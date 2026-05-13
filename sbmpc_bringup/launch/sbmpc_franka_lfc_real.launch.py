from __future__ import annotations

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
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
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from sbmpc_bringup.constants import (
    BRIDGE_DIAGNOSTICS_TOPIC,
    JOINT_STATE_ESTIMATOR_NAME,
    LINEAR_FEEDBACK_CONTROLLER_NAME,
)


def generate_launch_description() -> LaunchDescription:
    robot_description_file = LaunchConfiguration("robot_description_file")
    robot_description = ParameterValue(
        Command(
            [
                PathJoinSubstitution([FindExecutable(name="xacro")]),
                " ",
                robot_description_file,
                " robot_ip:=",
                LaunchConfiguration("robot_ip"),
                " mount_end_effector:=",
                LaunchConfiguration("load_gripper"),
                " use_fake_hardware:=",
                LaunchConfiguration("use_fake_hardware"),
                " fake_sensor_commands:=",
                LaunchConfiguration("fake_sensor_commands"),
                " ee_id:=",
                LaunchConfiguration("ee_id"),
                " use_camera:=",
                LaunchConfiguration("use_camera"),
                " use_ft_sensor:=",
                LaunchConfiguration("use_ft_sensor"),
                " safety_distance:=",
                LaunchConfiguration("safety_distance"),
            ]
        ),
        value_type=str,
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace=LaunchConfiguration("namespace"),
        parameters=[{"robot_description": robot_description}],
        output="screen",
        on_exit=Shutdown(),
    )

    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        namespace=LaunchConfiguration("namespace"),
        parameters=[
            LaunchConfiguration("controllers_file"),
            {"robot_description": robot_description},
            {"robot_type": LaunchConfiguration("robot_type")},
            {
                "load_gripper": ParameterValue(
                    LaunchConfiguration("load_gripper"),
                    value_type=bool,
                )
            },
            {"arm_prefix": LaunchConfiguration("arm_prefix")},
        ],
        output="screen",
        on_exit=Shutdown(),
    )

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        namespace=LaunchConfiguration("namespace"),
        parameters=[
            {
                "source_list": [
                    "franka/joint_states",
                    "franka_gripper/joint_states",
                ],
                "rate": ParameterValue(
                    LaunchConfiguration("joint_state_rate"),
                    value_type=int,
                ),
            }
        ],
        output="screen",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        namespace=LaunchConfiguration("namespace"),
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            LaunchConfiguration("controller_manager_name"),
            "--controller-manager-timeout",
            "60",
            "--param-file",
            LaunchConfiguration("controllers_file"),
            "--controller-ros-args",
            "--remap",
            "joint_states:=franka/joint_states",
        ],
        output="screen",
    )

    franka_robot_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        namespace=LaunchConfiguration("namespace"),
        arguments=[
            "franka_robot_state_broadcaster",
            "--controller-manager",
            LaunchConfiguration("controller_manager_name"),
            "--controller-manager-timeout",
            "60",
            "--param-file",
            LaunchConfiguration("controllers_file"),
        ],
        output="screen",
    )

    franka_gripper = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("franka_gripper"), "launch", "gripper.launch.py"]
            )
        ),
        launch_arguments={
            "namespace": LaunchConfiguration("namespace"),
            "robot_ip": LaunchConfiguration("robot_ip"),
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("load_gripper")),
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
            LaunchConfiguration("lfc_params_file"),
        ],
        output="screen",
    )

    bridge = Node(
        executable="python",
        arguments=["-m", "sbmpc_ros_bridge.lfc_bridge_node"],
        prefix=[LaunchConfiguration("bridge_runtime_script")],
        parameters=[
            LaunchConfiguration("bridge_params_file"),
            {
                "use_sim_time": False,
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

    activate_after_bridge_warmup = Node(
        package="sbmpc_bringup",
        executable="wait_for_bridge_warmup",
        arguments=[
            "--diagnostics-topic",
            BRIDGE_DIAGNOSTICS_TOPIC,
            "--timeout-sec",
            LaunchConfiguration("bridge_warmup_timeout_sec"),
            "--controller-manager",
            LaunchConfiguration("controller_manager_name"),
            "--activate-controller",
            JOINT_STATE_ESTIMATOR_NAME,
            "--activate-controller",
            LINEAR_FEEDBACK_CONTROLLER_NAME,
            "--switch-timeout-sec",
            LaunchConfiguration("controller_switch_timeout_sec"),
        ],
        output="screen",
    )

    def on_lfc_stack_spawner_exit(event, context):
        del context
        if event.returncode != 0:
            return [
                Shutdown(
                    reason=(
                        "Failed to load the SB-MPC LFC controllers in inactive "
                        "state."
                    )
                )
            ]
        return [bridge, activate_after_bridge_warmup]

    def on_activation_exit(event, context):
        del context
        if event.returncode == 0:
            return []
        return [
            Shutdown(
                reason=(
                    "SB-MPC bridge warmup or real-robot LFC controller activation "
                    "failed."
                )
            )
        ]

    def shutdown_if_spawner_failed(name: str):
        def on_exit(event, context):
            del context
            if event.returncode == 0:
                return []
            return [Shutdown(reason=f"Failed to spawn {name}.")]

        return on_exit

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_type", default_value="fer"),
            DeclareLaunchArgument("arm_prefix", default_value=""),
            DeclareLaunchArgument("namespace", default_value=""),
            DeclareLaunchArgument("robot_ip", default_value="172.17.1.2"),
            DeclareLaunchArgument("load_gripper", default_value="false"),
            DeclareLaunchArgument("ee_id", default_value="agimus_franka_hand"),
            DeclareLaunchArgument("use_camera", default_value="false"),
            DeclareLaunchArgument("use_ft_sensor", default_value="false"),
            DeclareLaunchArgument("safety_distance", default_value="0.03"),
            DeclareLaunchArgument("use_fake_hardware", default_value="false"),
            DeclareLaunchArgument("fake_sensor_commands", default_value="false"),
            DeclareLaunchArgument("joint_state_rate", default_value="30"),
            DeclareLaunchArgument("enable_nonzero_control", default_value="true"),
            DeclareLaunchArgument(
                "robot_description_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("sbmpc_bringup"),
                        "urdf",
                        "franka_arm_with_sbmpc_real.urdf.xacro",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "bridge_runtime_script",
                default_value=EnvironmentVariable(
                    "SBMPC_BRIDGE_RUNTIME_SCRIPT",
                    default_value="/workspace/sbmpc_containers/scripts/pixi_ros_run.sh",
                ),
            ),
            DeclareLaunchArgument(
                "controller_manager_name",
                default_value="/controller_manager",
            ),
            DeclareLaunchArgument(
                "bridge_warmup_timeout_sec",
                default_value="120",
            ),
            DeclareLaunchArgument(
                "controller_switch_timeout_sec",
                default_value="10",
            ),
            DeclareLaunchArgument(
                "pixi_env",
                default_value=EnvironmentVariable(
                    "PIXI_ENV",
                    default_value="cuda",
                ),
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
                    [FindPackageShare("sbmpc_bringup"), "config", "franka_lfc_params.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "bridge_params_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("sbmpc_bringup"),
                        "config",
                        "sbmpc_bridge_exact_async_40hz.yaml",
                    ]
                ),
            ),
            LogInfo(
                msg=[
                    "SB-MPC real launch: LFC controllers are loaded inactive, "
                    "the bridge warms up, then activation is requested.",
                ]
            ),
            LogInfo(
                msg=[
                    "SB-MPC bridge params: ",
                    LaunchConfiguration("bridge_params_file"),
                ]
            ),
            LogInfo(
                msg=[
                    "SB-MPC real robot description: ",
                    LaunchConfiguration("robot_description_file"),
                ]
            ),
            LogInfo(
                msg=[
                    "SB-MPC nonzero control after readiness: ",
                    LaunchConfiguration("enable_nonzero_control"),
                ]
            ),
            robot_state_publisher,
            controller_manager,
            joint_state_publisher,
            joint_state_broadcaster_spawner,
            franka_robot_state_broadcaster_spawner,
            franka_gripper,
            lfc_stack_spawner,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=joint_state_broadcaster_spawner,
                    on_exit=shutdown_if_spawner_failed("joint_state_broadcaster"),
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=franka_robot_state_broadcaster_spawner,
                    on_exit=shutdown_if_spawner_failed(
                        "franka_robot_state_broadcaster"
                    ),
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=lfc_stack_spawner,
                    on_exit=on_lfc_stack_spawner_exit,
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=activate_after_bridge_warmup,
                    on_exit=on_activation_exit,
                )
            ),
        ]
    )
