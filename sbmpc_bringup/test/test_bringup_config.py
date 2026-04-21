from __future__ import annotations

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
    assert cm_params[GRIPPER_ACTION_CONTROLLER_NAME]["type"] == (
        "position_controllers/GripperActionController"
    )

    gripper = config[GRIPPER_ACTION_CONTROLLER_NAME]["ros__parameters"]
    assert gripper["type"] == "position_controllers/GripperActionController"
    assert gripper["joint"] == FER_GRIPPER_JOINT_NAME
    assert gripper["allow_stalling"] is True


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
    assert params["enable_nonzero_control"] is False
    assert params["force_zero_control"] is False
    assert params["planner_phase"] == "PREGRASP"
    assert params["planner_gains"] is True
    assert params["planner_num_steps"] == 1
    assert params["planner_num_samples"] == 1024
    assert params["planner_horizon"] == 8
    assert params["planner_num_parallel_computations"] == 1024
    assert params["planner_num_control_points"] == 8
    assert params["planner_temperature"] == 0.05
    assert params["planner_dt"] == 0.02
    assert params["planner_lambda_mpc"] == 0.05
    assert params["planner_noise_scale"] == 0.05
    assert params["planner_std_dev_scale"] == 0.05
    assert params["planner_smoothing"] == "Spline"
    assert params["planner_gain_method"] == "finite_difference"
    assert params["planner_gain_fd_epsilon"] == 0.01
    assert params["planner_gain_fd_scheme"] == "forward"
    assert params["planner_gain_fd_num_samples"] == 256


def test_fer_sim_inertials_zero_only_the_problematic_link4_cross_terms() -> None:
    config = load_yaml("fer_sim_inertials.yaml")
    link4 = config["link4"]["inertia"]

    assert link4["xy"] == 0.0
    assert link4["xz"] == 0.0
    assert link4["yz"] == 0.0

    # Keep the rest of the FER inertials unchanged for the narrowest possible workaround.
    link3 = config["link3"]["inertia"]
    assert link3["xy"] != 0.0


def test_milestone5_bridge_presets_cover_feedforward_and_feedback_runs() -> None:
    feedforward = load_yaml("sbmpc_bridge_milestone5_feedforward.yaml")
    feedback = load_yaml("sbmpc_bridge_milestone5_feedback.yaml")
    local_lqr = load_yaml("sbmpc_bridge_m5_diag_local_lqr.yaml")
    exact_slow = load_yaml("sbmpc_bridge_m5_diag_exact_none_h10_5000.yaml")

    feedforward_params = feedforward["sbmpc_lfc_bridge_node"]["ros__parameters"]
    feedback_params = feedback["sbmpc_lfc_bridge_node"]["ros__parameters"]
    local_lqr_params = local_lqr["sbmpc_lfc_bridge_node"]["ros__parameters"]
    exact_slow_params = exact_slow["sbmpc_lfc_bridge_node"]["ros__parameters"]

    assert feedforward_params["planner_phase"] == "PREGRASP"
    assert feedback_params["planner_phase"] == "PREGRASP"
    assert local_lqr_params["planner_phase"] == "PREGRASP"
    assert feedforward_params["enable_nonzero_control"] is False
    assert feedback_params["enable_nonzero_control"] is False
    assert local_lqr_params["enable_nonzero_control"] is False
    assert exact_slow_params["enable_nonzero_control"] is False
    assert feedforward_params["planner_gains"] is False
    assert feedback_params["planner_gains"] is True
    assert local_lqr_params["planner_gains"] is True
    assert exact_slow_params["planner_gains"] is True
    assert local_lqr_params["planner_gain_method"] == "local_lqr"
    assert local_lqr_params["publish_rate_hz"] == 10.0
    assert exact_slow_params["planner_gain_method"] == "exact"
    assert exact_slow_params["planner_smoothing"] == "none"
    assert exact_slow_params["planner_num_parallel_computations"] == 5000
