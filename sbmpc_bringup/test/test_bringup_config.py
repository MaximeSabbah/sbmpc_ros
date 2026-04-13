from __future__ import annotations

from pathlib import Path

import yaml

from sbmpc_bringup.constants import (
    BRIDGE_CONTROL_TOPIC,
    BRIDGE_DIAGNOSTICS_TOPIC,
    BRIDGE_SENSOR_TOPIC,
    FER_ARM_JOINT_NAMES,
    JOINT_STATE_BROADCASTER_NAME,
    JOINT_STATE_ESTIMATOR_NAME,
    LINEAR_FEEDBACK_CONTROLLER_NAME,
    effort_command_interfaces,
    hardware_state_interfaces,
    lfc_reference_interfaces,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def load_yaml(name: str) -> dict[str, object]:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_franka_controllers_yaml_declares_expected_controller_types() -> None:
    config = load_yaml("franka_controllers.yaml")
    cm_params = config["controller_manager"]["ros__parameters"]

    assert cm_params["update_rate"] == 1000
    assert cm_params[JOINT_STATE_BROADCASTER_NAME]["type"] == (
        "joint_state_broadcaster/JointStateBroadcaster"
    )
    assert cm_params[JOINT_STATE_ESTIMATOR_NAME]["type"] == (
        "linear_feedback_controller/JointStateEstimator"
    )
    assert cm_params[LINEAR_FEEDBACK_CONTROLLER_NAME]["type"] == (
        "linear_feedback_controller/LinearFeedbackController"
    )


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


def test_bridge_params_file_points_to_the_lfc_topics_and_fer_joint_names() -> None:
    config = load_yaml("sbmpc_bridge.yaml")
    params = config["sbmpc_lfc_bridge_node"]["ros__parameters"]

    assert params["sensor_topic"] == BRIDGE_SENSOR_TOPIC
    assert params["control_topic"] == BRIDGE_CONTROL_TOPIC
    assert params["diagnostics_topic"] == BRIDGE_DIAGNOSTICS_TOPIC
    assert tuple(params["joint_names"]) == FER_ARM_JOINT_NAMES
    assert params["publish_rate_hz"] == 50.0
