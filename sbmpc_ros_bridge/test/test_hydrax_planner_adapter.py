"""V-B1 contract test for the hydrax planner adapter (port plan Phase 1.5).

Pure Python, no ROS runtime. Requires the hydrax uv environment (run through
uv_ros_run.sh); skips wherever hydrax or the LFC messages are unavailable,
e.g. in the pixi-env regression suite.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hydrax")
lfc_msgs = pytest.importorskip("linear_feedback_controller_msgs.msg")

from sbmpc_ros_bridge.hydrax_planner_adapter import (  # noqa: E402
    HydraxPlannerAdapter,
)
from sbmpc_ros_bridge.lfc_msg_adapter import (  # noqa: E402
    float64_multi_array_to_numpy,
    planner_output_to_control,
)
from sbmpc_ros_bridge.planner_adapter import PlannerInput  # noqa: E402
from sbmpc_ros_bridge.safety import validate_planner_output  # noqa: E402


@pytest.fixture(scope="module")
def adapter() -> HydraxPlannerAdapter:
    # One instance for the whole module: construction JIT-compiles the solver.
    return HydraxPlannerAdapter()


def _planner_input(adapter: HydraxPlannerAdapter) -> PlannerInput:
    q = np.asarray(adapter._task.start_q, dtype=np.float64)
    v = np.zeros_like(q)
    sensor = lfc_msgs.Sensor()
    sensor.joint_state.name = [f"fer_joint{i}" for i in range(1, 8)]
    sensor.joint_state.position = q.tolist()
    sensor.joint_state.velocity = v.tolist()
    return PlannerInput(sensor=sensor, q=q, v=v)


def test_config_matches_the_single_tuning_surface(adapter):
    """The pairing glue must build exactly the yaml-loaded configuration."""
    from hydrax.configs import load_pregrasp_config

    options, config = load_pregrasp_config()
    ctrl = adapter._ctrl
    assert ctrl.num_samples == config.num_samples
    assert ctrl.temperature == config.temperature
    assert ctrl.plan_horizon == config.plan_horizon
    assert ctrl.num_knots == config.num_knots
    assert ctrl.spline_type == config.spline_type
    assert ctrl.iterations == config.iterations
    assert ctrl.mean_adaptation_rate == config.mean_adaptation_rate
    assert ctrl.num_gain_samples == config.num_gain_samples
    np.testing.assert_allclose(
        np.asarray(ctrl.noise_std),
        config.noise_scale * np.asarray(options.tau_max),
        rtol=1e-6,
    )


def test_warmup_reset_step_sequencing(adapter):
    out = adapter.warmup()
    assert out.tau_ff.shape == (7,)
    adapter.reset_runtime_state_after_warmup()
    assert adapter._step_index == 0

    planner_input = _planner_input(adapter)
    out = adapter.step(planner_input)
    assert adapter._step_index == 1
    assert out.tau_ff.shape == (7,)
    assert np.all(np.isfinite(out.tau_ff))
    assert out.K.shape == (7, 14)
    assert np.all(np.isfinite(out.K))
    validate_planner_output(out.tau_ff, out.K)


def test_plan_clock_reproduces_the_reference(adapter):
    """Step k must return the plan point k (catches time-indexing bugs)."""
    adapter.reset_runtime_state_after_warmup()
    planner_input = _planner_input(adapter)
    for k in range(3):
        out = adapter.step(planner_input)
        np.testing.assert_allclose(out.reference_q, adapter._plan_q[k])
        np.testing.assert_allclose(out.reference_v, adapter._plan_v[k])
    # First plan point is the start configuration at zero velocity
    adapter.reset_runtime_state_after_warmup()
    out = adapter.step(planner_input)
    np.testing.assert_allclose(out.reference_q, adapter._task.start_q)
    np.testing.assert_allclose(out.reference_v, np.zeros(7))


def test_constant_impedance_in_dudx_convention(adapter):
    """K is -[diag(kp) | diag(kd)]; the bridge's -1 scale yields +[Kp|Kd]."""
    opts = adapter._task.options
    kp = np.asarray(opts.kp_fixed)
    kd = np.asarray(opts.kd_fixed)
    expected = -np.hstack([np.diag(kp), np.diag(kd)])

    adapter.reset_runtime_state_after_warmup()
    planner_input = _planner_input(adapter)
    first = adapter.step(planner_input)
    second = adapter.step(planner_input)
    np.testing.assert_allclose(first.K, expected)
    np.testing.assert_allclose(second.K, expected)  # constant across steps

    control = planner_output_to_control(first, planner_input)
    lfc_gain = float64_multi_array_to_numpy(control.feedback_gain)
    np.testing.assert_allclose(
        lfc_gain, np.hstack([np.diag(kp), np.diag(kd)])
    )
    feedforward = float64_multi_array_to_numpy(control.feedforward).reshape(-1)
    np.testing.assert_allclose(feedforward, first.tau_ff)


def test_mode_and_diagnostics(adapter):
    adapter.reset_runtime_state_after_warmup()
    out = adapter.step(_planner_input(adapter))
    assert out.phase == "PREGRASP"
    assert out.next_phase == "PREGRASP"
    assert out.diagnostics.gain_mode == "feedforward"
    assert out.diagnostics.planning_time_ms > 0.0
    assert adapter.mpc_dt == pytest.approx(0.04)


def test_diagnostic_helpers(adapter):
    planner_input = _planner_input(adapter)
    ee = adapter.ee_position(planner_input.q)
    assert ee.shape == (3,)
    tau_g = adapter.gravity_torques(planner_input.q)
    assert tau_g.shape == (7,)
    predicted = adapter.predict_state(planner_input, tau_g, 0.04)
    assert predicted is not None
    q_pred, v_pred = predicted
    # Gravity feedforward from rest: the arm barely moves in one period
    np.testing.assert_allclose(q_pred, planner_input.q, atol=1e-3)
    assert v_pred.shape == (7,)

    assert adapter.diagnostics_snapshot() is None


def test_rollout_visualization_capture(adapter):
    planner_input = _planner_input(adapter)

    # Disabled: getters return None and nothing is stored.
    adapter.set_rollout_capture_enabled(False)
    adapter.step(planner_input)
    assert adapter.planned_end_effector_path(planner_input) is None
    assert (
        adapter.representative_end_effector_rollouts(
            planner_input, max_rollouts=4
        )
        is None
    )

    # Enabled: EE paths of the last solve, nominal first.
    adapter.set_rollout_capture_enabled(True)
    adapter.warmup_rollout_visualization(max_rollouts=4)  # no data yet: no-op
    adapter.step(planner_input)

    result = adapter.planned_end_effector_path(planner_input)
    assert result is not None
    _, ee_path = result
    horizon = ee_path.shape[0]
    assert ee_path.shape == (horizon, 3)
    assert np.all(np.isfinite(ee_path))

    rollouts = adapter.representative_end_effector_rollouts(
        planner_input, max_rollouts=4
    )
    assert rollouts.shape == (4, horizon, 3)
    assert np.all(np.isfinite(rollouts))
    # First representative rollout is the zero-noise nominal = planned path.
    np.testing.assert_allclose(rollouts[0], ee_path, rtol=1e-5, atol=1e-6)

    # Marker-publisher consumption path (same as the bridge uses).
    from builtin_interfaces.msg import Time

    from sbmpc_ros_bridge.rollout_markers import make_trajectory_marker_array

    markers = make_trajectory_marker_array(
        ee_path=np.asarray(ee_path, dtype=np.float64),
        rollout_paths=rollouts,
        goal_position=np.zeros(3),
        frame_id="base",
        stamp=Time(),
        line_width=0.01,
        sample_line_width=0.004,
        point_diameter=0.025,
        goal_diameter=0.045,
    )
    assert len(markers.markers) > 0

    adapter.set_rollout_capture_enabled(False)
    assert adapter.planned_end_effector_path(planner_input) is None


@pytest.fixture(scope="module")
def fb_adapter() -> HydraxPlannerAdapter:
    # exact_feedback pairs the same glue with compute_gains=True; the first
    # solve JIT-compiles the gain graph (the bridge does this in warmup).
    return HydraxPlannerAdapter(mode="exact_feedback")


def test_exact_feedback_publishes_the_solver_gains(fb_adapter):
    """K is the solve's F-MPPI gain: time-varying, finite, healthy."""
    assert fb_adapter._ctrl.compute_gains is True
    fb_adapter.warmup()
    fb_adapter.reset_runtime_state_after_warmup()
    planner_input = _planner_input(fb_adapter)

    first = fb_adapter.step(planner_input)
    assert first.K.shape == (7, 14)
    assert np.all(np.isfinite(first.K))
    validate_planner_output(first.tau_ff, first.K)
    # The solver's du/dx gains, not the feedforward-mode constant impedance
    opts = fb_adapter._task.options
    impedance = -np.hstack([np.diag(opts.kp_fixed), np.diag(opts.kd_fixed)])
    assert not np.allclose(first.K, impedance)
    # Recomputed every solve (fresh sampling noise ⇒ a new gain estimate)
    second = fb_adapter.step(planner_input)
    assert not np.allclose(second.K, first.K)
    # V-A3 health readouts flow out of the solver with the gains
    diag = first.diagnostics
    assert diag.gain_mode == "exact_feedback"
    assert diag.gain_ess >= 1.0
    assert 0.0 <= diag.gain_nominal_weight <= 1.0
    assert diag.gain_norm == pytest.approx(np.linalg.norm(first.K))


def test_exact_feedback_lfc_anchor_is_the_solve_state(fb_adapter):
    """The F-MPPI law through LFC: K_lfc = -K and the anchor is x0.

    control.initial_state must stay the measured snapshot the solve
    started from (the bridge's reference substitution is feedforward-only),
    so LFC applies tau_ff + (-K)(x0 - x) = tau_ff + K (x - x0).
    """
    fb_adapter.reset_runtime_state_after_warmup()
    planner_input = _planner_input(fb_adapter)
    out = fb_adapter.step(planner_input)

    control = planner_output_to_control(out, planner_input)
    lfc_gain = float64_multi_array_to_numpy(control.feedback_gain)
    np.testing.assert_allclose(lfc_gain, -out.K, rtol=1e-6)
    np.testing.assert_allclose(
        control.initial_state.joint_state.position, planner_input.q
    )
    np.testing.assert_allclose(
        control.initial_state.joint_state.velocity, planner_input.v
    )
    feedforward = float64_multi_array_to_numpy(control.feedforward).reshape(-1)
    np.testing.assert_allclose(feedforward, out.tau_ff)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        HydraxPlannerAdapter(mode="bogus")
