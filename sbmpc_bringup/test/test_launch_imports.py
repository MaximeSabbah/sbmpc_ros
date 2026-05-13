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
            "controllers_file",
            "ee_id",
            "enable_nonzero_control",
            "fake_sensor_commands",
            "joint_state_rate",
            "lfc_params_file",
            "load_gripper",
            "namespace",
            "pixi_env",
            "robot_description_file",
            "robot_ip",
            "robot_type",
            "safety_distance",
            "sbmpc_dir",
            "use_camera",
            "use_fake_hardware",
            "use_ft_sensor",
        },
    )
    defaults = declared_argument_defaults(launch_description)
    assert defaults["robot_ip"] == "172.17.1.2"
    assert defaults["enable_nonzero_control"] == "true"
    assert "sbmpc_bridge_exact_async_40hz.yaml" in defaults["bridge_params_file"]
    assert "franka_arm_with_sbmpc_real.urdf.xacro" in defaults["robot_description_file"]
