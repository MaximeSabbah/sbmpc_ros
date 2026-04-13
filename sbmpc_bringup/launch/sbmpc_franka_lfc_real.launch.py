from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from sbmpc_bringup.constants import (
    JOINT_STATE_ESTIMATOR_NAME,
    LINEAR_FEEDBACK_CONTROLLER_NAME,
)


def generate_launch_description() -> LaunchDescription:
    franka_core = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("franka_bringup"), "launch", "franka.launch.py"]
            )
        ),
        launch_arguments={
            "robot_type": LaunchConfiguration("robot_type"),
            "arm_prefix": LaunchConfiguration("arm_prefix"),
            "namespace": LaunchConfiguration("namespace"),
            "robot_ip": LaunchConfiguration("robot_ip"),
            "load_gripper": LaunchConfiguration("load_gripper"),
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
            "fake_sensor_commands": LaunchConfiguration("fake_sensor_commands"),
            "joint_state_rate": LaunchConfiguration("joint_state_rate"),
            "controllers_yaml": LaunchConfiguration("controllers_file"),
        }.items(),
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
        package="sbmpc_ros_bridge",
        executable="sbmpc_lfc_bridge_node",
        parameters=[LaunchConfiguration("bridge_params_file"), {"use_sim_time": False}],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_type", default_value="fer"),
            DeclareLaunchArgument("arm_prefix", default_value=""),
            DeclareLaunchArgument("namespace", default_value=""),
            DeclareLaunchArgument("robot_ip", default_value="172.16.0.3"),
            DeclareLaunchArgument("load_gripper", default_value="false"),
            DeclareLaunchArgument("use_fake_hardware", default_value="false"),
            DeclareLaunchArgument("fake_sensor_commands", default_value="false"),
            DeclareLaunchArgument("joint_state_rate", default_value="30"),
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
                    [FindPackageShare("sbmpc_bringup"), "config", "franka_lfc_params.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "bridge_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("sbmpc_bringup"), "config", "sbmpc_bridge.yaml"]
                ),
            ),
            franka_core,
            lfc_stack_spawner,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=lfc_stack_spawner,
                    on_exit=[bridge],
                )
            ),
        ]
    )
