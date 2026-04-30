from __future__ import annotations

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
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
    GRIPPER_ACTION_CONTROLLER_NAME,
    JOINT_STATE_BROADCASTER_NAME,
    JOINT_STATE_ESTIMATOR_NAME,
    LINEAR_FEEDBACK_CONTROLLER_NAME,
)


def launch_setup(context, *args, **kwargs):
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

    control_node = Node(
        package="mujoco_ros2_control",
        executable="ros2_control_node",
        output="both",
        emulate_tty=True,
        parameters=[
            {"use_sim_time": True},
            controllers_file,
        ],
        remappings=(
            [("~/robot_description", "/robot_description")]
            if os.environ.get("ROS_DISTRO") == "humble"
            else []
        ),
        on_exit=Shutdown(),
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
            "--activate-as-group",
            "--param-file",
            LaunchConfiguration("controllers_file"),
            "--param-file",
            lfc_params_file,
            "--param-file",
            sim_lfc_params_file,
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

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="both",
            parameters=[robot_description, {"use_sim_time": True}],
        ),
        control_node,
        joint_state_broadcaster_spawner,
        gripper_spawner,
        lfc_stack_spawner,
        RegisterEventHandler(
            OnProcessExit(
                target_action=lfc_stack_spawner,
                on_exit=[bridge],
            )
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
            OpaqueFunction(function=launch_setup),
        ]
    )
