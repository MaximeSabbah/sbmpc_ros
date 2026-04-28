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
URDF_DIR = Path(__file__).resolve().parents[1] / "urdf"
EXPECTED_CONFIG_FILES = {
    "fer_sim_inertials.yaml",
    "franka_controllers.yaml",
    "franka_lfc_params.yaml",
    "franka_lfc_params_sim.yaml",
    "sbmpc_bridge.yaml",
    "sbmpc_bridge_exact_async.yaml",
    "sbmpc_bridge_feedforward.yaml",
}


def load_yaml(name: str) -> dict[str, object]:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_config_directory_contains_only_supported_presets() -> None:
    assert {path.name for path in CONFIG_DIR.glob("*.yaml")} == EXPECTED_CONFIG_FILES


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
    assert params["retime_control_initial_state"] is True
    assert params["control_initial_state_prediction_sec"] == 0.0
    assert params["planner_mode"] == "fd_feedback"
    assert params["planner_phase"] == "PREGRASP"
    assert params["planner_num_steps"] == 1
    assert params["planner_num_samples"] == 1024
    assert params["planner_horizon"] == 8
    assert params["planner_num_control_points"] == 8
    assert params["planner_temperature"] == 0.05
    assert params["planner_dt"] == 0.02
    assert params["planner_noise_scale"] == 0.05
    assert params["planner_smoothing"] == "Spline"
    assert params["planner_gain_fd_epsilon"] == 0.05
    assert params["planner_gain_fd_scheme"] == "forward"
    assert params["planner_gain_fd_num_samples"] == 256


def test_gazebo_xacro_keeps_gravity_enabled_by_default_for_sbmpc() -> None:
    xacro_text = (
        URDF_DIR / "franka_arm_with_sbmpc_inertials.gazebo.xacro"
    ).read_text(encoding="utf-8")

    assert '<xacro:arg name="disable_gazebo_gravity" default="false"/>' in xacro_text
    assert '<xacro:if value="$(arg disable_gazebo_gravity)">' in xacro_text


def test_fer_sim_inertials_zero_only_the_problematic_link4_cross_terms() -> None:
    config = load_yaml("fer_sim_inertials.yaml")
    link4 = config["link4"]["inertia"]

    assert link4["xy"] == 0.0
    assert link4["xz"] == 0.0
    assert link4["yz"] == 0.0

    # Keep the rest of the FER inertials unchanged for the narrowest possible workaround.
    link3 = config["link3"]["inertia"]
    assert link3["xy"] != 0.0


def test_bridge_presets_cover_feedforward_and_exact_async_runs() -> None:
    feedforward = load_yaml("sbmpc_bridge_feedforward.yaml")
    feedback = load_yaml("sbmpc_bridge.yaml")
    exact_async = load_yaml("sbmpc_bridge_exact_async.yaml")

    feedforward_params = feedforward["sbmpc_lfc_bridge_node"]["ros__parameters"]
    feedback_params = feedback["sbmpc_lfc_bridge_node"]["ros__parameters"]
    exact_async_params = exact_async["sbmpc_lfc_bridge_node"]["ros__parameters"]

    assert feedforward_params["planner_phase"] == "PREGRASP"
    assert feedback_params["planner_phase"] == "PREGRASP"
    assert feedforward_params["planner_mode"] == "feedforward"
    assert feedback_params["planner_mode"] == "fd_feedback"
    assert feedforward_params["publish_rate_hz"] == 50.0
    assert feedback_params["publish_rate_hz"] == 50.0
    assert exact_async_params["publish_rate_hz"] == 50.0
    assert feedforward_params["planner_dt"] == 0.02
    assert feedback_params["planner_dt"] == 0.02
    assert exact_async_params["planner_dt"] == 0.02
    assert feedforward_params["enable_nonzero_control"] is False
    assert feedback_params["enable_nonzero_control"] is False
    assert exact_async_params["enable_nonzero_control"] is False
    assert exact_async_params["planner_mode"] == "exact_async_feedback"
    assert exact_async_params["planner_gain_samples_per_cycle"] == 128
    assert exact_async_params["planner_gain_buffer_size"] == 512
