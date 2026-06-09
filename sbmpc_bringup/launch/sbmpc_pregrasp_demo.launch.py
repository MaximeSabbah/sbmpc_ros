from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    Shutdown,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("sbmpc_bringup"),
                    "launch",
                    "sbmpc_franka_lfc_mujoco_sim.launch.py",
                ]
            )
        ),
        launch_arguments={
            "headless": LaunchConfiguration("headless"),
            "use_rviz": LaunchConfiguration("use_rviz"),
            "enable_nonzero_control": "true",
            "bridge_params_file": LaunchConfiguration("bridge_params_file"),
            "record_replay": LaunchConfiguration("record_replay"),
            "record_replay_output": LaunchConfiguration("record_replay_output"),
            "record_replay_duration_sec": LaunchConfiguration(
                "record_replay_duration_sec"
            ),
        }.items(),
    )

    validator = Node(
        package="sbmpc_bringup",
        executable="validate_sbmpc_sim",
        arguments=[
            "--duration-sec",
            LaunchConfiguration("validation_duration_sec"),
            "--startup-timeout-sec",
            LaunchConfiguration("validation_startup_timeout_sec"),
            "--assert-stable",
            "--max-p95-planning-ms",
            LaunchConfiguration("max_p95_planning_ms"),
            "--max-final-position-error",
            "0.01",
            "--max-torque-fraction",
            "0.9",
            "--max-velocity-fraction",
            "0.9",
            "--max-position-fraction",
            "0.9",
        ],
        condition=IfCondition(LaunchConfiguration("validate")),
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "headless",
                default_value="false",
                description="Set true for reliable timing without the MuJoCo viewer.",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Show the ROS robot state alongside MuJoCo.",
            ),
            DeclareLaunchArgument("validate", default_value="true"),
            DeclareLaunchArgument(
                "shutdown_after_validation", default_value="false"
            ),
            DeclareLaunchArgument(
                "max_p95_planning_ms",
                default_value="0",
                description="Set to 40 for a strict headless 25 Hz timing gate.",
            ),
            DeclareLaunchArgument(
                "validation_duration_sec", default_value="16"
            ),
            DeclareLaunchArgument(
                "validation_startup_timeout_sec", default_value="180"
            ),
            DeclareLaunchArgument(
                "bridge_params_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("sbmpc_bringup"),
                        "config",
                        "sbmpc_bridge.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument("record_replay", default_value="false"),
            DeclareLaunchArgument(
                "record_replay_output",
                default_value="/tmp/sbmpc_pregrasp_replay.json",
            ),
            DeclareLaunchArgument(
                "record_replay_duration_sec", default_value="0"
            ),
            simulation,
            validator,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=validator,
                    on_exit=Shutdown(reason="SB-MPC validation completed."),
                ),
                condition=IfCondition(
                    LaunchConfiguration("shutdown_after_validation")
                ),
            ),
        ]
    )
