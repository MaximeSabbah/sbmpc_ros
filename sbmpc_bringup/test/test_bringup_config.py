from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

from sbmpc_bringup.constants import (
    BRIDGE_CONTROL_TOPIC,
    BRIDGE_DIAGNOSTICS_TOPIC,
    BRIDGE_SENSOR_TOPIC,
    FER_ARM_JOINT_NAMES,
    FER_GRIPPER_JOINT_NAME,
    GRIPPER_ACTION_CONTROLLER_NAME,
    JOINT_STATE_BROADCASTER_NAME,
    JOINT_STATE_ESTIMATOR_NAME,
    LINEAR_FEEDBACK_CONTROLLER_NAME,
    effort_command_interfaces,
    hardware_state_interfaces,
    lfc_reference_interfaces,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
LAUNCH_DIR = Path(__file__).resolve().parents[1] / "launch"
EXPECTED_CONFIG_FILES = {
    "franka_controllers.yaml",
    "franka_lfc_params.yaml",
    "franka_lfc_params_sim.yaml",
    "sbmpc_bridge.yaml",
    "sbmpc_bridge_feedforward.yaml",
}


def load_yaml(name: str) -> dict[str, object]:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_launch_module(name: str):
    path = LAUNCH_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_directory_contains_only_supported_presets() -> None:
    assert {path.name for path in CONFIG_DIR.glob("*.yaml")} == EXPECTED_CONFIG_FILES


def test_franka_controllers_yaml_declares_expected_controller_types() -> None:
    config = load_yaml("franka_controllers.yaml")
    cm_params = config["controller_manager"]["ros__parameters"]

    assert cm_params["update_rate"] == 1000
    assert cm_params["cpu_affinity"] == [0, 1]
    assert cm_params[JOINT_STATE_BROADCASTER_NAME]["type"] == (
        "joint_state_broadcaster/JointStateBroadcaster"
    )
    assert cm_params["franka_robot_state_broadcaster"]["type"] == (
        "agimus_franka_robot_state_broadcaster/AgimusFrankaRobotStateBroadcaster"
    )
    assert cm_params[JOINT_STATE_ESTIMATOR_NAME]["type"] == (
        "linear_feedback_controller/JointStateEstimator"
    )
    assert cm_params[LINEAR_FEEDBACK_CONTROLLER_NAME]["type"] == (
        "linear_feedback_controller/LinearFeedbackController"
    )
    assert cm_params[GRIPPER_ACTION_CONTROLLER_NAME]["type"] == (
        "position_controllers/GripperActionController"
    )

    gripper = config[GRIPPER_ACTION_CONTROLLER_NAME]["ros__parameters"]
    assert gripper["type"] == "position_controllers/GripperActionController"
    assert gripper["joint"] == FER_GRIPPER_JOINT_NAME
    assert gripper["allow_stalling"] is True

    state_broadcaster = config["franka_robot_state_broadcaster"]["ros__parameters"]
    assert state_broadcaster == {"arm_id": "fer"}


def test_franka_lfc_params_match_the_expected_fer_interface_layout() -> None:
    config = load_yaml("franka_lfc_params.yaml")

    estimator = config[JOINT_STATE_ESTIMATOR_NAME]["ros__parameters"]
    assert tuple(estimator["state_interfaces"]) == hardware_state_interfaces()
    assert tuple(estimator["command_interfaces"]) == lfc_reference_interfaces()

    controller = config[LINEAR_FEEDBACK_CONTROLLER_NAME]["ros__parameters"]
    assert tuple(controller["moving_joint_names"]) == FER_ARM_JOINT_NAMES
    assert tuple(controller["chainable_controller"]["command_interfaces"]) == (
        effort_command_interfaces()
    )
    assert controller["remove_gravity_compensation_effort"] is True


def test_sim_lfc_params_only_override_direct_effort_gravity_handling() -> None:
    config = load_yaml("franka_lfc_params_sim.yaml")

    assert config == {
        LINEAR_FEEDBACK_CONTROLLER_NAME: {
            "ros__parameters": {"remove_gravity_compensation_effort": False}
        }
    }


def test_bridge_params_file_points_to_the_lfc_topics_and_fer_joint_names() -> None:
    config = load_yaml("sbmpc_bridge.yaml")
    params = config["sbmpc_lfc_bridge_node"]["ros__parameters"]

    assert params["sensor_topic"] == BRIDGE_SENSOR_TOPIC
    assert params["control_topic"] == BRIDGE_CONTROL_TOPIC
    assert params["diagnostics_topic"] == BRIDGE_DIAGNOSTICS_TOPIC
    assert tuple(params["joint_names"]) == FER_ARM_JOINT_NAMES
    assert params["publish_rate_hz"] == 25.0
    assert params["planner_deadline_sec"] == 0.04
    assert params["enable_nonzero_control"] is False
    assert params["retime_control_initial_state"] is True
    assert params["planner_mode"] == "exact_feedback"
    assert params["planner_phase"] == "PREGRASP"
    assert params["planner_num_steps"] == 1
    assert params["planner_ocp"] == "pregrasp"
    assert params["planner_warmup_iterations"] == 3


# The sbmpc OCP yaml (planner_ocp) owns the MPPI knobs. The bridge presets
# must not duplicate them, otherwise the two repos drift apart.
MPPI_KNOBS_OWNED_BY_OCP_YAML = (
    "planner_num_samples",
    "planner_horizon",
    "planner_num_parallel_computations",
    "planner_num_control_points",
    "planner_temperature",
    "planner_dt",
    "planner_lambda_mpc",
    "planner_noise_scale",
    "planner_std_dev_scale",
    "planner_smoothing",
    "planner_num_gain_samples",
)


def test_bridge_presets_cover_feedforward_and_exact_feedback() -> None:
    feedforward = load_yaml("sbmpc_bridge_feedforward.yaml")["sbmpc_lfc_bridge_node"]["ros__parameters"]
    feedback = load_yaml("sbmpc_bridge.yaml")["sbmpc_lfc_bridge_node"]["ros__parameters"]

    assert feedforward["planner_mode"] == "feedforward"
    assert feedback["planner_mode"] == "exact_feedback"
    for params in (feedforward, feedback):
        assert params["publish_rate_hz"] == 25.0
        assert params["planner_ocp"] == "pregrasp"
        for knob in MPPI_KNOBS_OWNED_BY_OCP_YAML:
            assert knob not in params, (
                f"{knob} duplicates the sbmpc OCP yaml; tune it there instead."
            )


def test_mujoco_launch_partitions_simulation_and_bridge_cpus(monkeypatch) -> None:
    launch_module = load_launch_module("sbmpc_franka_lfc_mujoco_sim.launch.py")
    monkeypatch.setattr(
        launch_module.os,
        "sched_getaffinity",
        lambda pid: {0, 1, 2, 3},
    )

    assert launch_module.simulation_cpu_prefixes() == (
        "taskset -c 0,1",
        "taskset -c 2,3",
    )
