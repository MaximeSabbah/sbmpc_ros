from __future__ import annotations

import importlib.util
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument


LAUNCH_DIR = Path(__file__).resolve().parents[1] / "launch"


def load_launch_module(filename: str):
    launch_path = LAUNCH_DIR / filename
    spec = importlib.util.spec_from_file_location(launch_path.stem, launch_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_declared_arguments(
    launch_description: LaunchDescription,
    expected_argument_names: set[str],
) -> None:
    declared_argument_names = {
        entity.name
        for entity in launch_description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    assert expected_argument_names.issubset(declared_argument_names)


def declared_argument_defaults(launch_description: LaunchDescription) -> dict[str, str]:
    defaults: dict[str, str] = {}
    for entity in launch_description.entities:
        if not isinstance(entity, DeclareLaunchArgument):
            continue
        defaults[entity.name] = "".join(
            getattr(value, "text", str(value)) for value in entity.default_value
        )
    return defaults


def test_legacy_gazebo_sim_launch_is_removed() -> None:
    assert not (LAUNCH_DIR / "sbmpc_franka_lfc_sim.launch.py").exists()


def test_real_launch_imports_and_declares_expected_arguments() -> None:
    module = load_launch_module("sbmpc_franka_lfc_real.launch.py")
    launch_description = module.generate_launch_description()

    assert isinstance(launch_description, LaunchDescription)
    assert_declared_arguments(
        launch_description,
        {
            "arm_prefix",
            "bridge_warmup_timeout_sec",
            "bridge_runtime_script",
            "bridge_params_file",
            "controller_switch_timeout_sec",
            "controller_manager_name",
            "controller_watchdog_period_sec",
            "controller_watchdog_service_timeout_sec",
            "controllers_file",
            "ee_id",
            "enable_nonzero_control",
            "fake_sensor_commands",
            "joint_state_rate",
            "lfc_params_file",
            "load_gripper",
            "namespace",
            "pixi_env",
            "publish_rollout_markers",
            "record_replay",
            "record_replay_autosave_period_sec",
            "record_replay_duration_sec",
            "record_replay_output",
            "require_realtime",
            "robot_description_file",
            "robot_ip",
            "robot_type",
            "rviz_config",
            "safety_distance",
            "sbmpc_dir",
            "start_gripper_node",
            "use_camera",
            "use_fake_hardware",
            "use_ft_sensor",
            "use_rviz",
        },
    )
    defaults = declared_argument_defaults(launch_description)
    assert defaults["robot_ip"] == "172.17.1.2"
    assert defaults["load_gripper"] == "true"
    assert defaults["start_gripper_node"] == "false"
    assert defaults["enable_nonzero_control"] == "false"
    assert defaults["require_realtime"] == "false"
    assert defaults["use_rviz"] == "true"
    assert defaults["record_replay"] == "false"
    assert defaults["record_replay_output"] == "/tmp/sbmpc_real_replay.json"
    assert defaults["record_replay_duration_sec"] == "0"
    assert defaults["record_replay_autosave_period_sec"] == "2"
    assert defaults["publish_rollout_markers"] == "false"
    assert "pregrasp.rviz" in defaults["rviz_config"]
    assert "sbmpc_bridge_real_bringup.yaml" in defaults["bridge_params_file"]
    assert "franka_arm_with_sbmpc_real.urdf.xacro" in defaults["robot_description_file"]


def test_real_launch_joint_state_broadcaster_remap_is_spawner_safe() -> None:
    module = load_launch_module("sbmpc_franka_lfc_real.launch.py")
    launch_description = module.generate_launch_description()

    matching_spawners = [
        entity
        for entity in launch_description.entities
        if getattr(entity, "_Node__arguments", None)
        and getattr(entity, "_Node__arguments")[0] == "joint_state_broadcaster"
    ]

    assert len(matching_spawners) == 1
    arguments = getattr(matching_spawners[0], "_Node__arguments")
    assert "--controller-ros-args=--remap" in arguments
    assert "--controller-ros-args=joint_states:=franka/joint_states" in arguments
