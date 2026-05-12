from __future__ import annotations

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit, OnProcessIO
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
    GRIPPER_ACTION_CONTROLLER_NAME,
    JOINT_STATE_BROADCASTER_NAME,
    JOINT_STATE_ESTIMATOR_NAME,
    LINEAR_FEEDBACK_CONTROLLER_NAME,
)


def available_cpus() -> tuple[int, ...]:
    if hasattr(os, "sched_getaffinity"):
        return tuple(sorted(int(cpu) for cpu in os.sched_getaffinity(0)))
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return ()
    return tuple(range(cpu_count))


def taskset_prefix(cpus: tuple[int, ...]) -> str:
    return "taskset -c " + ",".join(str(cpu) for cpu in cpus)


def simulation_cpu_prefixes() -> tuple[str, str]:
    cpus = available_cpus()
    if len(cpus) < 4:
        return "", ""
    return taskset_prefix(cpus[:2]), taskset_prefix(cpus[2:])


def launch_setup(context, *args, **kwargs):
    control_cpu_prefix, bridge_cpu_prefix = simulation_cpu_prefixes()
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("sbmpc_bringup"),
                    "urdf",
                    "franka_arm_with_sbmpc_mujoco.urdf.xacro",
                ]
            ),
            " headless:=",
            LaunchConfiguration("headless"),
            " mujoco_model:=",
            LaunchConfiguration("mujoco_model"),
        ]
    )
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }
    controllers_file = ParameterFile(LaunchConfiguration("controllers_file"))
    lfc_params_file = LaunchConfiguration("lfc_params_file")
    sim_lfc_params_file = LaunchConfiguration("sim_lfc_params_file")
    record_replay_args = [
        "--duration-sec",
        LaunchConfiguration("record_replay_duration_sec"),
        "--output",
        LaunchConfiguration("record_replay_output"),
        "--startup-timeout-sec",
        LaunchConfiguration("record_replay_startup_timeout_sec"),
        "--autosave-period-sec",
        LaunchConfiguration("record_replay_autosave_period_sec"),
    ]
    if LaunchConfiguration("record_replay_include_warmup").perform(context).lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        record_replay_args.append("--include-warmup")

    control_node_kwargs = {
        "package": "mujoco_ros2_control",
        "executable": "ros2_control_node",
        "output": "both",
        "emulate_tty": True,
        "parameters": [
            {"use_sim_time": True},
            controllers_file,
        ],
        "remappings": (
            [("~/robot_description", "/robot_description")]
            if os.environ.get("ROS_DISTRO") == "humble"
            else []
        ),
        "on_exit": Shutdown(),
    }
    if control_cpu_prefix:
        control_node_kwargs["prefix"] = control_cpu_prefix
    control_node = Node(**control_node_kwargs)

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            JOINT_STATE_BROADCASTER_NAME,
            "--controller-manager",
            LaunchConfiguration("controller_manager_name"),
            "--controller-manager-timeout",
            "60",
            "--param-file",
            LaunchConfiguration("controllers_file"),
        ],
        output="screen",
    )

    gripper_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            GRIPPER_ACTION_CONTROLLER_NAME,
            "--controller-manager",
            LaunchConfiguration("controller_manager_name"),
            "--controller-manager-timeout",
            "60",
            "--param-file",
            LaunchConfiguration("controllers_file"),
        ],
        output="screen",
    )

    lfc_stack_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            JOINT_STATE_ESTIMATOR_NAME,
            LINEAR_FEEDBACK_CONTROLLER_NAME,
            "--controller-manager",
            LaunchConfiguration("controller_manager_name"),
            "--controller-manager-timeout",
            "60",
            "--switch-timeout",
            "60",
            "--inactive",
            "--param-file",
            LaunchConfiguration("controllers_file"),
            "--param-file",
            lfc_params_file,
            "--param-file",
            sim_lfc_params_file,
        ],
        output="screen",
    )
    bridge_prefix = [LaunchConfiguration("bridge_runtime_script")]
    if bridge_cpu_prefix:
        bridge_prefix = [
            bridge_cpu_prefix,
            " ",
            LaunchConfiguration("bridge_runtime_script"),
        ]

    bridge = Node(
        executable="python",
        arguments=["-m", "sbmpc_ros_bridge.lfc_bridge_node"],
        prefix=bridge_prefix,
        parameters=[
            LaunchConfiguration("bridge_params_file"),
            {
                "use_sim_time": True,
                "enable_nonzero_control": ParameterValue(
                    LaunchConfiguration("enable_nonzero_control"),
                    value_type=bool,
                ),
            },
        ],
        additional_env={
            "PIXI_ENV": LaunchConfiguration("pixi_env"),
            "SBMPC_DIR": LaunchConfiguration("sbmpc_dir"),
        },
        output="screen",
        on_exit=Shutdown(),
    )

    reset_after_bridge_warmup = Node(
        package="sbmpc_bringup",
        executable="wait_for_bridge_warmup",
        arguments=[
            "--diagnostics-topic",
            BRIDGE_DIAGNOSTICS_TOPIC,
            "--timeout-sec",
            "5",
            "--reset-world-service",
            "/mujoco_ros2_control_node/reset_world",
            "--reset-keyframe",
            "home",
            "--controller-manager",
            LaunchConfiguration("controller_manager_name"),
            "--activate-controller",
            JOINT_STATE_ESTIMATOR_NAME,
            "--activate-controller",
            LINEAR_FEEDBACK_CONTROLLER_NAME,
            "--switch-timeout-sec",
            "10",
        ],
        output="screen",
    )
    replay_recorder = Node(
        package="sbmpc_bringup",
        executable="record_sbmpc_replay",
        arguments=record_replay_args,
        condition=IfCondition(LaunchConfiguration("record_replay")),
        output="screen",
    )
    warmup_reset_started = {"value": False}

    def on_bridge_output(event):
        text = event.text.decode(errors="replace") if isinstance(event.text, bytes) else str(event.text)
        if (
            not warmup_reset_started["value"]
            and "Planner warmup/JIT compilation complete" in text
        ):
            warmup_reset_started["value"] = True
            return [reset_after_bridge_warmup]
        return []

    def on_reset_exit(event, context):
        del context
        if event.returncode == 0:
            return []
        return [
            Shutdown(
                reason=(
                    "MuJoCo reset or LFC activation after SB-MPC bridge "
                    "warmup failed."
                )
            )
        ]

    return [
        LogInfo(
            msg=[
                "SB-MPC bridge params: ",
                LaunchConfiguration("bridge_params_file"),
            ]
        ),
        LogInfo(
            msg=[
                "ros2_control controllers: ",
                LaunchConfiguration("controllers_file"),
            ]
        ),
        LogInfo(
            msg=[
                "LFC params: ",
                LaunchConfiguration("lfc_params_file"),
                " + ",
                LaunchConfiguration("sim_lfc_params_file"),
            ]
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="both",
            parameters=[robot_description, {"use_sim_time": True}],
        ),
        control_node,
        replay_recorder,
        bridge,
        joint_state_broadcaster_spawner,
        gripper_spawner,
        lfc_stack_spawner,
        RegisterEventHandler(
            OnProcessIO(target_action=bridge, on_stdout=on_bridge_output)
        ),
        RegisterEventHandler(
            OnProcessIO(target_action=bridge, on_stderr=on_bridge_output)
        ),
        RegisterEventHandler(
            OnProcessExit(target_action=reset_after_bridge_warmup, on_exit=on_reset_exit)
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("headless", default_value="true"),
            DeclareLaunchArgument("enable_nonzero_control", default_value="false"),
            DeclareLaunchArgument(
                "controller_manager_name",
                default_value="/controller_manager",
            ),
            DeclareLaunchArgument(
                "bridge_runtime_script",
                default_value=EnvironmentVariable(
                    "SBMPC_BRIDGE_RUNTIME_SCRIPT",
                    default_value="/workspace/sbmpc_containers/scripts/pixi_ros_run.sh",
                ),
            ),
            DeclareLaunchArgument(
                "pixi_env",
                default_value=EnvironmentVariable("PIXI_ENV", default_value="cuda"),
            ),
            DeclareLaunchArgument(
                "sbmpc_dir",
                default_value=EnvironmentVariable(
                    "SBMPC_DIR",
                    default_value="/workspace/sbmpc",
                ),
            ),
            DeclareLaunchArgument(
                "controllers_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("sbmpc_bringup"),
                        "config",
                        "franka_controllers.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "lfc_params_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("sbmpc_bringup"),
                        "config",
                        "franka_lfc_params.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "sim_lfc_params_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("sbmpc_bringup"),
                        "config",
                        "franka_lfc_params_sim.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "bridge_params_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("sbmpc_bringup"),
                        "config",
                        "sbmpc_bridge_exact_async.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "mujoco_model",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("sbmpc_bringup"),
                        "mujoco",
                        "panda_pick_place_ros2_control_scene.xml",
                    ]
                ),
            ),
            DeclareLaunchArgument("record_replay", default_value="false"),
            DeclareLaunchArgument(
                "record_replay_output",
                default_value="/tmp/sbmpc_ros_replay.json",
            ),
            DeclareLaunchArgument(
                "record_replay_duration_sec",
                default_value="0",
                description="0 records until the launch is stopped.",
            ),
            DeclareLaunchArgument(
                "record_replay_startup_timeout_sec",
                default_value="120",
            ),
            DeclareLaunchArgument(
                "record_replay_autosave_period_sec",
                default_value="5",
            ),
            DeclareLaunchArgument(
                "record_replay_include_warmup",
                default_value="false",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
