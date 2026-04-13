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


def test_sim_launch_imports_and_declares_expected_arguments() -> None:
    module = load_launch_module("sbmpc_franka_lfc_sim.launch.py")
    launch_description = module.generate_launch_description()

    assert isinstance(launch_description, LaunchDescription)
    assert_declared_arguments(
        launch_description,
        {
            "bridge_params_file",
            "controller_manager_name",
            "controllers_file",
            "entity_name",
            "franka_hand",
            "gz_args",
            "lfc_params_file",
            "load_gripper",
            "robot_type",
            "use_rviz",
        },
    )


def test_real_launch_imports_and_declares_expected_arguments() -> None:
    module = load_launch_module("sbmpc_franka_lfc_real.launch.py")
    launch_description = module.generate_launch_description()

    assert isinstance(launch_description, LaunchDescription)
    assert_declared_arguments(
        launch_description,
        {
            "arm_prefix",
            "bridge_params_file",
            "controller_manager_name",
            "controllers_file",
            "fake_sensor_commands",
            "joint_state_rate",
            "lfc_params_file",
            "load_gripper",
            "namespace",
            "robot_ip",
            "robot_type",
            "use_fake_hardware",
        },
    )
