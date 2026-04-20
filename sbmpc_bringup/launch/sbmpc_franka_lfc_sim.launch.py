from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit, OnShutdown
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
    GRIPPER_ACTION_CONTROLLER_NAME,
    JOINT_STATE_BROADCASTER_NAME,
    JOINT_STATE_ESTIMATOR_NAME,
    LINEAR_FEEDBACK_CONTROLLER_NAME,
)
from sbmpc_bringup.launch_preflight import assert_clean_ros_graph


def generate_launch_description() -> LaunchDescription:
    franka_description_share = get_package_share_directory("franka_description")
    franka_resource_root = os.path.dirname(franka_description_share)

    robot_description = ParameterValue(
        Command(
            [
                PathJoinSubstitution([FindExecutable(name="xacro")]),
                " ",
                PathJoinSubstitution(
                    [
                        FindPackageShare("sbmpc_bringup"),
                        "urdf",
                        "franka_arm_with_sbmpc_inertials.gazebo.xacro",
                    ]
                ),
                " robot_type:=",
                LaunchConfiguration("robot_type"),
                " inertials_file:=",
                LaunchConfiguration("inertials_file"),
                " hand:=",
                LaunchConfiguration("load_gripper"),
                " ros2_control:=true",
                " gazebo:=true",
                " ee_id:=",
                LaunchConfiguration("franka_hand"),
                " gazebo_effort:=true",
            ]
        ),
        value_type=str,
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"]
            )
        ),
        launch_arguments={"gz_args": LaunchConfiguration("gz_args")}.items(),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
    )

    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name",
            LaunchConfiguration("entity_name"),
            "-topic",
            "/robot_description",
        ],
        output="screen",
    )

    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
    )

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        parameters=[{"source_list": ["joint_states"], "rate": 30, "use_sim_time": True}],
        output="screen",
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=[
            "--display-config",
            PathJoinSubstitution(
                [FindPackageShare("franka_description"), "rviz", "visualize_franka.rviz"]
            ),
            "-f",
            "world",
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            JOINT_STATE_BROADCASTER_NAME,
            "--controller-manager",
            LaunchConfiguration("controller_manager_name"),
            "--controller-manager-timeout",
            "60",
            "--switch-timeout",
            "60",
            "--param-file",
            LaunchConfiguration("controllers_file"),
        ],
        output="screen",
    )

    gripper_action_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            GRIPPER_ACTION_CONTROLLER_NAME,
            "--controller-manager",
            LaunchConfiguration("controller_manager_name"),
            "--controller-manager-timeout",
            "60",
            "--switch-timeout",
            "60",
            "--param-file",
            LaunchConfiguration("controllers_file"),
        ],
        output="screen",
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
            "--activate-as-group",
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
        parameters=[LaunchConfiguration("bridge_params_file"), {"use_sim_time": True}],
        additional_env={
            "PIXI_ENV": LaunchConfiguration("pixi_env"),
            "SBMPC_DIR": LaunchConfiguration("sbmpc_dir"),
        },
        output="screen",
    )

    cleanup_on_shutdown = ExecuteProcess(
        cmd=[
            FindExecutable(name="ros2"),
            "run",
            "sbmpc_bringup",
            "cleanup_sbmpc_sim",
            "--timeout-sec",
            "3.0",
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_type", default_value="fer"),
            DeclareLaunchArgument("load_gripper", default_value="false"),
            DeclareLaunchArgument("franka_hand", default_value="franka_hand"),
            DeclareLaunchArgument("entity_name", default_value="franka"),
            DeclareLaunchArgument("gz_args", default_value="empty.sdf -r"),
            DeclareLaunchArgument("use_rviz", default_value="false"),
            DeclareLaunchArgument(
                "allow_existing_ros_graph",
                default_value="false",
                description=(
                    "Bypass the stale ROS/Gazebo graph guard. Keep false for normal "
                    "Milestone 5 validation runs."
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
                "inertials_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("sbmpc_bringup"), "config", "fer_sim_inertials.yaml"]
                ),
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
                "controller_manager_name",
                default_value="/controller_manager",
            ),
            DeclareLaunchArgument(
                "controllers_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("sbmpc_bringup"), "config", "franka_controllers.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "lfc_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("sbmpc_bringup"), "config", "franka_lfc_params_sim.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "bridge_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("sbmpc_bringup"), "config", "sbmpc_bridge.yaml"]
                ),
            ),
            SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", franka_resource_root),
            OpaqueFunction(function=assert_clean_ros_graph),
            gazebo,
            robot_state_publisher,
            spawn_entity,
            clock_bridge,
            joint_state_publisher,
            rviz,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=spawn_entity,
                    on_exit=[joint_state_broadcaster_spawner],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=joint_state_broadcaster_spawner,
                    on_exit=[gripper_action_controller_spawner, lfc_stack_spawner],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=lfc_stack_spawner,
                    on_exit=[bridge],
                )
            ),
            RegisterEventHandler(OnShutdown(on_shutdown=[cleanup_on_shutdown])),
        ]
    )
