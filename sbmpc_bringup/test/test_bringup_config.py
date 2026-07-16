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

# One config set serves both backends: shared controllers + LFC params, one
# bridge config per planner (sbmpc | hydrax), and a single physical sim
# overlay (gravity compensation).
EXPECTED_CONFIG_FILES = {
    "franka_controllers.yaml",
    "franka_lfc_params.yaml",
    "franka_lfc_params_sim.yaml",
    "sbmpc_bridge.yaml",
    "hydrax_bridge.yaml",
}

# The sbmpc OCP yaml (planner_ocp) owns the MPPI knobs. The bridge config must
# not duplicate them, otherwise the two repos drift apart.
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


def load_yaml(name: str) -> dict[str, object]:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_config_directory_contains_only_the_supported_files() -> None:
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
        "effort_controllers/GripperActionController"
    )

    gripper = config[GRIPPER_ACTION_CONTROLLER_NAME]["ros__parameters"]
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
    # The real FCI applies gravity compensation, so the LFC removes it on real.
    assert controller["remove_gravity_compensation_effort"] is True


def test_sim_lfc_params_only_override_direct_effort_gravity_handling() -> None:
    # MuJoCo has no gravity compensation, so the sim keeps it in the LFC output.
    config = load_yaml("franka_lfc_params_sim.yaml")

    assert config == {
        LINEAR_FEEDBACK_CONTROLLER_NAME: {
            "ros__parameters": {"remove_gravity_compensation_effort": False}
        }
    }


def test_bridge_config_points_to_the_lfc_topics_and_fer_joint_names() -> None:
    params = load_yaml("sbmpc_bridge.yaml")["sbmpc_lfc_bridge_node"]["ros__parameters"]

    assert params["sensor_topic"] == BRIDGE_SENSOR_TOPIC
    assert params["control_topic"] == BRIDGE_CONTROL_TOPIC
    assert params["diagnostics_topic"] == BRIDGE_DIAGNOSTICS_TOPIC
    assert tuple(params["joint_names"]) == FER_ARM_JOINT_NAMES
    assert params["publish_rate_hz"] == 25.0
    # exact_feedback is the deployed sbmpc mode (2026-07-01 direction); this
    # assertion had drifted from the committed yaml while the suite was not
    # collecting under the pixi runtime (pytest plugin incompatibility).
    assert params["planner_mode"] == "exact_feedback"
    assert params["planner_num_steps"] == 1
    assert params["planner_ocp"] == "pregrasp"
    assert params["planner_warmup_iterations"] == 3
    assert params["feedforward_position_gain"] == 0.0
    assert params["feedforward_velocity_damping_gain"] == 0.0


def test_hydrax_bridge_config_is_transport_only() -> None:
    params = load_yaml("hydrax_bridge.yaml")["sbmpc_lfc_bridge_node"]["ros__parameters"]

    assert params["sensor_topic"] == BRIDGE_SENSOR_TOPIC
    assert params["control_topic"] == BRIDGE_CONTROL_TOPIC
    assert params["diagnostics_topic"] == BRIDGE_DIAGNOSTICS_TOPIC
    assert tuple(params["joint_names"]) == FER_ARM_JOINT_NAMES
    assert params["publish_rate_hz"] == 25.0
    # exact_feedback is the deployed hydrax mode (port plan Phase 4,
    # 2026-07-06); this preset yaml is THE mode switch (user decision:
    # no launch-argument override).
    assert params["planner_mode"] == "exact_feedback"
    # planner_ocp SELECTS which hydrax tuning surface/task the adapter
    # loads (P3 design, user-approved 2026-07-10) — same yaml-key pattern
    # as planner_mode. Selection, not tuning: the values still live only
    # in the hydrax yaml (guard below).
    assert params["planner_ocp"] in ("pregrasp", "pick_place")
    assert params["planner_warmup_iterations"] == 3
    # The gripper action NAME is backend-dependent and injected by the
    # launch; the preset must not pin it.
    assert "gripper_action_name" not in params


def test_bridge_config_does_not_duplicate_mppi_knobs_owned_by_the_ocp_yaml() -> None:
    params = load_yaml("sbmpc_bridge.yaml")["sbmpc_lfc_bridge_node"]["ros__parameters"]
    for knob in MPPI_KNOBS_OWNED_BY_OCP_YAML:
        assert knob not in params, (
            f"{knob} duplicates the sbmpc OCP yaml; tune it there instead."
        )


def test_hydrax_bridge_config_does_not_duplicate_the_hydrax_tuning_surface() -> None:
    # The hydrax OCP tuning lives exclusively in the hydrax configs yaml
    # (pregrasp.yaml / pick_place.yaml, selected by planner_ocp); the
    # bridge preset is transport wiring only.
    params = load_yaml("hydrax_bridge.yaml")["sbmpc_lfc_bridge_node"]["ros__parameters"]
    for knob in MPPI_KNOBS_OWNED_BY_OCP_YAML + ("planner_cost_weights",):
        assert knob not in params, (
            f"{knob} duplicates the hydrax tuning surface; tune it there instead."
        )
