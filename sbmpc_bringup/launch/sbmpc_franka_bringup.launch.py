from __future__ import annotations

import json
import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    EnvironmentVariable,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile, ParameterValue
from launch_ros.substitutions import FindPackageShare

from sbmpc_bringup.constants import (
    BRIDGE_DIAGNOSTICS_TOPIC,
    BRIDGE_NODE_NAME,
    CONTROLLER_MANAGER_NAME,
    CONTROLLER_SWITCH_TIMEOUT_SEC,
    DEFAULT_KEYFRAME,
    GRIPPER_ACTION_CONTROLLER_NAME,
    JOINT_STATE_BROADCASTER_NAME,
    JOINT_STATE_ESTIMATOR_NAME,
    LINEAR_FEEDBACK_CONTROLLER_NAME,
    RESET_WORLD_SERVICE,
)
from sbmpc_bringup.launch_preflight import assert_clean_ros_graph


TRUE_VALUES = {"1", "true", "yes", "on"}
SWITCH_TIMEOUT = str(CONTROLLER_SWITCH_TIMEOUT_SEC)


def _is_true(context, name: str) -> bool:
    return LaunchConfiguration(name).perform(context).strip().lower() in TRUE_VALUES


def _share(*parts: str):
    return PathJoinSubstitution([FindPackageShare("sbmpc_bringup"), *parts])


def launch_setup(context, *args, **kwargs):
    del args, kwargs

    # Fail closed on a stale ROS graph for both backends (D22): a leftover
    # controller_manager / bridge colliding with a fresh bringup is a real
    # footgun, and a stale graph on the live FCI is worse than in sim. This is
    # the backstop; clean teardown on stop is the primary mechanism.
    assert_clean_ros_graph(context)

    backend = LaunchConfiguration("backend").perform(context).strip().lower()
    if backend not in {"mujoco", "real"}:
        raise RuntimeError(f"backend must be 'mujoco' or 'real', got {backend!r}.")
    is_mujoco = backend == "mujoco"
    use_sim_time = is_mujoco

    # Which solver backs the bridge (orthogonal to the plant backend above).
    planner = LaunchConfiguration("planner").perform(context).strip().lower()
    if planner not in {"sbmpc", "hydrax"}:
        raise RuntimeError(f"planner must be 'sbmpc' or 'hydrax', got {planner!r}.")
    is_hydrax = planner == "hydrax"

    # Single config set for both backends; sim adds exactly one physical overlay.
    controllers_file = _share("config", "franka_controllers.yaml")
    lfc_params_file = _share("config", "franka_lfc_params.yaml")
    sim_lfc_params_file = _share("config", "franka_lfc_params_sim.yaml")
    bridge_params_file = _share(
        "config", "hydrax_bridge.yaml" if is_hydrax else "sbmpc_bridge.yaml"
    )
    rviz_config = _share("rviz", "pregrasp.rviz")

    # The bridge runs inside the selected planner's Python runtime (pixi for
    # sbmpc, uv for hydrax); these come from the environment, not the launch
    # interface (plan §3).
    bridge_runtime_script = EnvironmentVariable(
        "SBMPC_BRIDGE_RUNTIME_SCRIPT",
        default_value=(
            "/workspace/sbmpc_containers/scripts/uv_ros_run.sh"
            if is_hydrax
            else "/workspace/sbmpc_containers/scripts/pixi_ros_run.sh"
        ),
    )
    pixi_env = EnvironmentVariable("PIXI_ENV", default_value="cuda")
    sbmpc_dir = EnvironmentVariable("SBMPC_DIR", default_value="/workspace/sbmpc")
    hydrax_dir = EnvironmentVariable("HYDRAX_DIR", default_value="/workspace/hydrax")

    # --- sim start pose (the V-A5 placement seam, live) ---
    # random regenerates the scene's home keyframe: qpos drawn fresh within
    # the V-A5-certified ±0.2 rad/joint envelope AND gravity-comp ctrl
    # recomputed at the drawn pose (the keyframe ctrl holds the arm before
    # the LFC stack activates). Generation runs in the planner runtime —
    # the launch python has no mujoco. The reference plan still starts at
    # the nominal home pose, so the run shows the controller absorbing the
    # placement gap, exactly like a hand-placed real robot.
    initial_q = LaunchConfiguration("initial_q").perform(context).strip().lower()
    if initial_q not in {"home", "random"}:
        raise RuntimeError(f"initial_q must be 'home' or 'random', got {initial_q!r}.")
    initial_q_actions = []
    xacro_model_override = []
    if initial_q == "random":
        if not is_mujoco:
            raise RuntimeError(
                "initial_q:=random needs backend:=mujoco "
                "(the real robot starts wherever it physically is)."
            )
        generator = subprocess.run(
            [
                bridge_runtime_script.perform(context),
                "python",
                "-m",
                "sbmpc_bringup.initial_q",
                "--scene",
                os.path.join(
                    get_package_share_directory("sbmpc_bringup"),
                    "mujoco",
                    "fer_pick_place_ros2_control_scene.xml",
                ),
            ],
            capture_output=True,
            text=True,
        )
        if generator.returncode != 0:
            raise RuntimeError(
                "initial_q:=random keyframe generation failed:\n"
                + generator.stderr[-2000:]
            )
        draw = json.loads(generator.stdout.strip().splitlines()[-1])
        xacro_model_override = [" mujoco_model:=", draw["mujoco_model"]]
        initial_q_actions = [
            LogInfo(
                msg=(
                    "initial_q=random start pose [rad]: "
                    + " ".join(f"{q:.3f}" for q in draw["q_arm"])
                    + "  (offset from home: "
                    + " ".join(f"{o:+.3f}" for o in draw["offset"])
                    + ")"
                )
            )
        ]

    # --- robot_description: one xacro per backend, same agimus_franka macro ---
    if is_mujoco:
        robot_description_content = Command(
            [
                PathJoinSubstitution([FindExecutable(name="xacro")]),
                " ",
                _share("urdf", "franka_arm_with_sbmpc_mujoco.urdf.xacro"),
                " headless:=",
                LaunchConfiguration("headless"),
                *xacro_model_override,
            ]
        )
    else:
        robot_description_content = Command(
            [
                PathJoinSubstitution([FindExecutable(name="xacro")]),
                " ",
                _share("urdf", "franka_arm_with_sbmpc_real.urdf.xacro"),
                " robot_ip:=",
                LaunchConfiguration("robot_ip"),
                # Hand always present for kinematic parity with the sim (D21).
                " mount_end_effector:=true",
            ]
        )
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    # --- core nodes shared by both backends ---
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description, {"use_sim_time": use_sim_time}],
        on_exit=Shutdown(),
    )

    if is_mujoco:
        control_node = Node(
            package="mujoco_ros2_control",
            executable="ros2_control_node",
            output="both",
            emulate_tty=True,
            parameters=[{"use_sim_time": True}, ParameterFile(controllers_file)],
            remappings=(
                [("~/robot_description", "/robot_description")]
                if os.environ.get("ROS_DISTRO") == "humble"
                else []
            ),
            on_exit=Shutdown(),
        )
    else:
        control_node = Node(
            package="controller_manager",
            executable="ros2_control_node",
            output="screen",
            parameters=[
                ParameterFile(controllers_file),
                robot_description,
                {"robot_type": "fer", "load_gripper": True, "arm_prefix": ""},
            ],
            on_exit=Shutdown(),
        )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": use_sim_time}],
        additional_env={"LIBGL_ALWAYS_SOFTWARE": "1"},
        condition=IfCondition(LaunchConfiguration("use_rviz")),
        output="screen",
    )

    # The planner's gripper_command executes through ONE GripperCommand
    # action client in the bridge; the action name and the close position
    # differ per backend (the launch owns backend knowledge). Empty name =
    # no gripper wired. Close position: the sim's effort controller needs
    # the overdrive to 0 — its squeeze force IS the residual position
    # error times the PID p — while the real agimus node maps position to
    # a franka grasp width (2x, epsilon +-5 mm): it must be half the cube
    # width (0.04 m cube), or the grasp "fails" while physically holding
    # the cube and trips the bridge's fail-closed path.
    if _is_true(context, "use_gripper"):
        gripper_action_name = (
            "/gripper_action_controller/gripper_cmd"
            if is_mujoco
            else "/fer_gripper/gripper_action"
        )
    else:
        gripper_action_name = ""
    gripper_close_position = 0.0 if is_mujoco else 0.02

    bridge = Node(
        executable="python",
        arguments=["-m", "sbmpc_ros_bridge.lfc_bridge_node"],
        prefix=[bridge_runtime_script],
        parameters=[
            bridge_params_file,
            {
                "use_sim_time": use_sim_time,
                # Structural pairing with the runtime wrapper selected above:
                # the preset yaml repeats it, but the launch owns the choice
                # (a missing/stale preset must not silently flip the planner).
                "planner_impl": planner,
                "gripper_action_name": gripper_action_name,
                "gripper_close_position": gripper_close_position,
                # Always start disarmed; the warmup step arms after JIT/warmup.
                "enable_nonzero_control": False,
                "publish_rollout_markers": ParameterValue(
                    LaunchConfiguration("publish_rollout_markers"),
                    value_type=bool,
                ),
            },
        ],
        additional_env={"PIXI_ENV": pixi_env, "SBMPC_DIR": sbmpc_dir},
        output="screen",
        on_exit=Shutdown(),
    )

    # --- joint_state_broadcaster (real remaps its output to franka/joint_states) ---
    jsb_arguments = [
        JOINT_STATE_BROADCASTER_NAME,
        "--controller-manager",
        CONTROLLER_MANAGER_NAME,
        "--controller-manager-timeout",
        SWITCH_TIMEOUT,
        "--param-file",
        controllers_file,
    ]
    if not is_mujoco:
        jsb_arguments += [
            "--controller-ros-args=--remap",
            "--controller-ros-args=joint_states:=franka/joint_states",
        ]
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=jsb_arguments,
        output="screen",
    )

    # --- LFC stack, activated up front in both backends (D8) ---
    lfc_param_files = ["--param-file", controllers_file, "--param-file", lfc_params_file]
    if is_mujoco:
        # MuJoCo has no gravity compensation; keep it in the LFC output (D10).
        lfc_param_files += ["--param-file", sim_lfc_params_file]
    lfc_stack_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            JOINT_STATE_ESTIMATOR_NAME,
            LINEAR_FEEDBACK_CONTROLLER_NAME,
            "--controller-manager",
            CONTROLLER_MANAGER_NAME,
            "--controller-manager-timeout",
            SWITCH_TIMEOUT,
            "--switch-timeout",
            SWITCH_TIMEOUT,
            "--activate-as-group",
            *lfc_param_files,
        ],
        output="screen",
    )

    # --- optional replay recorder (D14): one path arg, full debug set ---
    record_replay_path = LaunchConfiguration("record_replay").perform(context).strip()
    recorder_nodes = []
    if record_replay_path:
        recorder_arguments = [
            "--output",
            record_replay_path,
            "--duration-sec",
            "0",  # record until shutdown
            "--record-lfc-output",  # always capture commanded torque (D14)
            "--include-warmup",  # record immediately; do not depend on /control timing
        ]
        if is_mujoco:
            # tau_J only exists on real hardware; the sim has no measured-torque topic.
            recorder_arguments += ["--measured-torque-topic", ""]
        recorder_nodes = [
            Node(
                package="sbmpc_bringup",
                executable="record_sbmpc_replay",
                arguments=recorder_arguments,
                output="screen",
            )
        ]

    # --- warmup + arm (D7/D8/D20): wait on diagnostics, reset (sim), arm if asked ---
    warmup_arguments = [
        "--diagnostics-topic",
        BRIDGE_DIAGNOSTICS_TOPIC,
        "--timeout-sec",
        "0",  # unbounded: planner JIT can exceed minutes (D20)
        "--bridge-node",
        BRIDGE_NODE_NAME,
        "--enable-nonzero-control",
        LaunchConfiguration("enable_nonzero_control"),
    ]
    if is_mujoco:
        warmup_arguments += [
            "--reset-world-service",
            RESET_WORLD_SERVICE,
            "--reset-keyframe",
            DEFAULT_KEYFRAME,
        ]
    warmup_step = Node(
        package="sbmpc_bringup",
        executable="wait_for_bridge_warmup",
        arguments=warmup_arguments,
        output="screen",
    )

    # --- backend-specific nodes + the required-spawner set ---
    # (node, label) pairs whose non-zero exit must abort the launch.
    required_spawners = [(joint_state_broadcaster_spawner, "joint_state_broadcaster")]
    backend_actions = []
    if is_mujoco:
        gripper_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                GRIPPER_ACTION_CONTROLLER_NAME,
                "--controller-manager",
                CONTROLLER_MANAGER_NAME,
                "--controller-manager-timeout",
                SWITCH_TIMEOUT,
                "--param-file",
                controllers_file,
            ],
            condition=IfCondition(LaunchConfiguration("use_gripper")),
            output="screen",
        )
        backend_actions = [gripper_spawner]
        required_spawners.append((gripper_spawner, GRIPPER_ACTION_CONTROLLER_NAME))
    else:
        franka_robot_state_broadcaster_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "franka_robot_state_broadcaster",
                "--controller-manager",
                CONTROLLER_MANAGER_NAME,
                "--controller-manager-timeout",
                SWITCH_TIMEOUT,
                "--param-file",
                controllers_file,
            ],
            output="screen",
        )
        joint_state_publisher = Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            name="joint_state_publisher",
            parameters=[
                {
                    "source_list": ["franka/joint_states", "fer_gripper/joint_states"],
                    "rate": 30,  # TF/RViz republish only — not in the control path
                }
            ],
            output="screen",
        )
        franka_gripper = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [
                        FindPackageShare("agimus_franka_gripper"),
                        "launch",
                        "gripper.launch.py",
                    ]
                )
            ),
            launch_arguments={
                "robot_ip": LaunchConfiguration("robot_ip"),
                "arm_id": "fer",
            }.items(),
            condition=IfCondition(LaunchConfiguration("use_gripper")),
        )
        backend_actions = [
            franka_robot_state_broadcaster_spawner,
            joint_state_publisher,
            franka_gripper,
        ]
        required_spawners.append(
            (franka_robot_state_broadcaster_spawner, "franka_robot_state_broadcaster")
        )

    # --- shutdown / sequencing handlers ---
    def shutdown_if_failed(reason: str):
        def on_exit(event, context):
            del context
            if event.returncode == 0:
                return []
            return [Shutdown(reason=reason)]

        return on_exit

    def on_lfc_ready(event, context):
        del context
        if event.returncode != 0:
            return [Shutdown(reason="Failed to activate the SB-MPC LFC controller stack.")]
        # The LFC now PD-holds the start pose; wait for bridge warmup, then
        # (sim) reset the world and arm if requested.
        return [warmup_step]

    event_handlers = [
        RegisterEventHandler(
            OnProcessExit(
                target_action=node,
                on_exit=shutdown_if_failed(f"Required controller spawner failed: {label}."),
            )
        )
        for node, label in required_spawners
    ]
    event_handlers.append(
        RegisterEventHandler(
            OnProcessExit(target_action=lfc_stack_spawner, on_exit=on_lfc_ready)
        )
    )
    event_handlers.append(
        RegisterEventHandler(
            OnProcessExit(
                target_action=warmup_step,
                on_exit=shutdown_if_failed(
                    "SB-MPC bridge warmup, MuJoCo reset, or arming failed."
                ),
            )
        )
    )

    summary = (
        f"SB-MPC Franka bringup | backend={backend} | planner={planner} | "
        f"initial_q={initial_q} | "
        f"rviz={_is_true(context, 'use_rviz')} | gripper={_is_true(context, 'use_gripper')} | "
        f"arm_after_warmup={_is_true(context, 'enable_nonzero_control')} | "
        f"rollout_markers={_is_true(context, 'publish_rollout_markers')} | "
        f"record_replay={record_replay_path or '(off)'}"
    )

    telemetry_hint = (
        "SB-MPC live telemetry: the bridge logs one line per planner solve "
        "(solve wall time vs deadline, plan/cmd/prep ms, |tau|, |gain|, "
        f"deadline_miss) to this console; full per-step diagnostics stream on "
        f"{BRIDGE_DIAGNOSTICS_TOPIC} (ros2 topic echo)."
    )

    return [
        LogInfo(msg=summary),
        *initial_q_actions,
        LogInfo(msg=telemetry_hint),
        robot_state_publisher,
        control_node,
        rviz,
        bridge,
        *recorder_nodes,
        joint_state_broadcaster_spawner,
        *backend_actions,
        lfc_stack_spawner,
        *event_handlers,
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "backend",
                default_value="mujoco",
                choices=["mujoco", "real"],
                description="mujoco (physics sim) or real (Franka FCI).",
            ),
            DeclareLaunchArgument(
                "planner",
                default_value="hydrax",
                choices=["sbmpc", "hydrax"],
                description=(
                    "Which solver backs the bridge (orthogonal to `backend`). "
                    "Selects the bridge preset yaml and the runtime wrapper: "
                    "sbmpc runs in its pixi env, hydrax in its uv env. "
                    "Defaults to hydrax; pass planner:=sbmpc for the legacy "
                    "backend."
                ),
            ),
            DeclareLaunchArgument(
                "enable_nonzero_control",
                default_value="true",
                description=(
                    "Arm the bridge (send nonzero control) after warmup. Default "
                    "true so the sim runs out of the box. On real hardware pass "
                    "enable_nonzero_control:=false for a disarmed PD-hold bringup."
                ),
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Launch RViz.",
            ),
            DeclareLaunchArgument(
                "headless",
                default_value="false",
                description="mujoco only: open the MuJoCo viewer (default). Set true to run without it. Ignored on real.",
            ),
            DeclareLaunchArgument(
                "initial_q",
                default_value="home",
                choices=["home", "random"],
                description=(
                    "mujoco only: arm start pose. home = the scene keyframe; "
                    "random = a fresh uniform draw within ±0.2 rad/joint "
                    "around home (the V-A5-certified placement envelope), "
                    "logged at launch. The reference plan still starts at "
                    "home, so the controller must absorb the gap — the "
                    "hand-placed-robot case, watchable in the viewer."
                ),
            ),
            DeclareLaunchArgument(
                "robot_ip",
                default_value="172.17.1.2",
                description="real only: Franka FCI IP. Ignored on mujoco.",
            ),
            DeclareLaunchArgument(
                "publish_rollout_markers",
                default_value="false",
                description=(
                    "Publish MPPI rollout markers for RViz (off the control hot "
                    "path). Off by default to protect controller timing."
                ),
            ),
            DeclareLaunchArgument(
                "record_replay",
                default_value="",
                description="Path to write a replay JSON; empty disables recording.",
            ),
            DeclareLaunchArgument(
                "use_gripper",
                default_value="true",
                description=(
                    "Actuate the gripper: mujoco spawns gripper_action_controller; "
                    "real includes the agimus_franka_gripper FCI node."
                ),
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
