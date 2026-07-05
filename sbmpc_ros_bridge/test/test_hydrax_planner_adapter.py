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

    # Unimplemented visualization hooks must return None without raising
    assert adapter.planned_end_effector_path(planner_input) is None
    assert (
        adapter.representative_end_effector_rollouts(
            planner_input, max_rollouts=4
        )
        is None
    )
    assert adapter.diagnostics_snapshot() is None
    adapter.set_rollout_capture_enabled(True)
    adapter.warmup_rollout_visualization(max_rollouts=4)


def test_exact_feedback_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        HydraxPlannerAdapter(mode="exact_feedback")
