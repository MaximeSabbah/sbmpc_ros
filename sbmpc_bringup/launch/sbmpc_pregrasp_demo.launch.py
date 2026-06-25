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
            "max_sensor_age_sec": LaunchConfiguration("max_sensor_age_sec"),
            "max_planner_output_age_sec": LaunchConfiguration(
                "max_planner_output_age_sec"
            ),
            "bridge_params_file": LaunchConfiguration("bridge_params_file"),
            "record_replay": LaunchConfiguration("record_replay"),
            "record_replay_output": LaunchConfiguration("record_replay_output"),
            "record_replay_duration_sec": LaunchConfiguration(
                "record_replay_duration_sec"
            ),
            "record_lfc_output": LaunchConfiguration("record_lfc_output"),
            "publish_rollout_markers": LaunchConfiguration("publish_rollout_markers"),
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
                default_value="true",
                description="Set false to also open the MuJoCo viewer.",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Show the ROS robot state alongside MuJoCo.",
            ),
            DeclareLaunchArgument("validate", default_value="false"),
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
                "max_sensor_age_sec",
                default_value="0.0",
                description=(
                    "Simulation-only sensor stale guard. 0 disables this "
                    "fail-closed guard; the real launch still uses the strict "
                    "bridge config."
                ),
            ),
            DeclareLaunchArgument(
                "max_planner_output_age_sec",
                default_value="0.0",
                description=(
                    "Simulation-only planner-output stale guard. 0 disables "
                    "this fail-closed guard; the real launch still uses the "
                    "strict bridge config."
                ),
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
            DeclareLaunchArgument("record_lfc_output", default_value="false"),
            DeclareLaunchArgument(
                "publish_rollout_markers",
                default_value="false",
                description=(
                    "Publish representative MPPI end-effector rollouts for "
                    "RViz debugging. Disabled by default to preserve the "
                    "validated controller timing path."
                ),
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
