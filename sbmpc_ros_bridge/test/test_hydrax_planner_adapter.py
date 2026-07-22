"""V-B1 contract test for the hydrax planner adapter (port plan Phase 1.5).

Pure Python, no ROS runtime. Requires the hydrax uv environment (run through
uv_ros_run.sh); skips wherever hydrax or the LFC messages are unavailable,
e.g. in the pixi-env regression suite.
"""

from __future__ import annotations

import json

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
    planner_input = _planner_input(adapter)
    out = adapter.step(planner_input)
    assert out.phase == "PREGRASP"
    assert out.next_phase == "PREGRASP"
    assert out.diagnostics.gain_mode == "feedforward"
    assert out.diagnostics.planning_time_ms > 0.0
    assert out.diagnostics.phase_machine is None
    expected_position, expected_rotation = adapter._ee_pose(planner_input.q)
    expected_goal_position = np.asarray(adapter._task.options.goal_pos)
    expected_goal_rotation = np.asarray(adapter._task.options.goal_rot)
    expected_signed_error = expected_position - expected_goal_position
    relative_rotation = expected_goal_rotation.T @ expected_rotation
    expected_orientation_error = np.arccos(
        np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    )
    np.testing.assert_allclose(out.diagnostics.ee_position, expected_position)
    np.testing.assert_allclose(out.diagnostics.ee_rotation, expected_rotation)
    np.testing.assert_allclose(
        out.diagnostics.goal_rotation, expected_goal_rotation
    )
    np.testing.assert_allclose(
        out.diagnostics.position_error_signed, expected_signed_error
    )
    assert out.diagnostics.position_error == pytest.approx(
        np.linalg.norm(expected_signed_error)
    )
    assert out.diagnostics.orientation_error == pytest.approx(
        expected_orientation_error
    )
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


def test_unknown_ocp_is_rejected():
    with pytest.raises(ValueError):
        HydraxPlannerAdapter(ocp="bogus")


# --- pick-and-place (P-B1, P3) ------------------------------------------


@pytest.fixture(scope="module")
def pp_adapter() -> HydraxPlannerAdapter:
    # The deployment pairing: exact_feedback + the pick_place task. Warmed
    # once like the bridge does; each test resets the runtime state.
    adapter = HydraxPlannerAdapter(mode="exact_feedback", ocp="pick_place")
    adapter.warmup()
    adapter.reset_runtime_state_after_warmup()
    return adapter


def _converged_input(adapter: HydraxPlannerAdapter) -> PlannerInput:
    """Planner input sitting exactly on the current phase goal, at rest."""
    from hydrax.tasks.panda_pick_place import Phase

    pm = adapter._phase_machine
    q = np.asarray(
        adapter._task.phase_goal_q[min(pm.phase, Phase.RETREAT)],
        dtype=np.float64,
    )
    return _state_input(q, np.zeros_like(q))


def _state_input(q: np.ndarray, v: np.ndarray) -> PlannerInput:
    """Build the measured-state contract used by the adapter."""
    sensor = lfc_msgs.Sensor()
    sensor.joint_state.name = [f"fer_joint{i}" for i in range(1, 8)]
    sensor.joint_state.position = q.tolist()
    sensor.joint_state.velocity = v.tolist()
    return PlannerInput(sensor=sensor, q=q, v=v)


def test_pick_place_config_matches_the_tuning_surface(pp_adapter):
    from hydrax.configs import load_pick_place_config

    options, config = load_pick_place_config()
    ctrl = pp_adapter._ctrl
    assert ctrl.compute_gains is True
    assert ctrl.num_samples == config.num_samples
    assert ctrl.temperature == config.temperature
    assert ctrl.plan_horizon == config.plan_horizon
    assert ctrl.num_knots == config.num_knots
    assert ctrl.spline_type == config.spline_type
    assert ctrl.iterations == config.iterations
    assert ctrl.num_gain_samples == config.num_gain_samples
    np.testing.assert_allclose(
        np.asarray(ctrl.noise_std),
        config.noise_scale * np.asarray(options.tau_max),
        rtol=1e-6,
    )
    # The diagnostics goal is the placement target, not a pregrasp pose
    np.testing.assert_allclose(pp_adapter._goal_pos, options.target_pos)
    # Phase acceptance is a fixed task contract, never a YAML/controller knob.
    assert not hasattr(options, "transition_ee_position_tolerance")


def test_pick_place_diagnostics_expose_a_blocked_pregrasp_gate(pp_adapter):
    """The deployed payload explains why PREGRASP cannot advance."""
    from hydrax.tasks.panda_pick_place import Phase

    pp_adapter.reset_runtime_state_after_warmup()
    task = pp_adapter._task
    pm = pp_adapter._phase_machine
    pm.plan_time = float(task.segment_end_times[Phase.PREGRASP])

    # The start state is intentionally far from the pregrasp EE goal.
    q = np.asarray(task.start_q, dtype=np.float64).copy()
    v = np.zeros(7, dtype=np.float64)
    out = pp_adapter.step(_state_input(q, v))

    gate = out.diagnostics.phase_machine
    assert gate is not None
    assert out.phase == gate["phase"] == "PREGRASP"
    assert out.next_phase == gate["next_phase"] == "DESCEND"
    assert gate["gate_type"] == "task_space"
    assert gate["at_boundary"] is True
    assert gate["transition_status"].startswith("blocked_ee_position")
    assert gate["transition_blocked"] is True
    assert "ee_position" in gate["transition_blockers"]
    assert gate["ee_position_ok"] is False
    assert gate["q_error_max_rad"] > 0.0
    assert gate["velocity_abs_max_rad_s"] == pytest.approx(0.0)
    assert gate["ee_linear_speed_m_s"] == pytest.approx(0.0)
    assert gate["ee_angular_speed_rad_s"] == pytest.approx(0.0)
    assert gate["precision_hold"] is False
    assert gate["clock_paused"] is False
    assert gate["clock_pause_reason"] is None
    assert gate["gripper_command"] == out.gripper_command == "open"
    assert gate["ee_source"] == "planning_model_fk"
    ee = np.asarray(gate["ee_position_m"])
    ee_goal = np.asarray(gate["ee_goal_position_m"])
    ee_error = np.asarray(gate["ee_position_error_signed_m"])
    assert ee.shape == ee_goal.shape == ee_error.shape == (3,)
    np.testing.assert_allclose(ee_error, ee - ee_goal)
    assert gate["ee_position_error_norm_m"] == pytest.approx(
        np.linalg.norm(ee_error)
    )
    assert gate["ee_orientation_error_rad"] >= 0.0
    # This exact object is forwarded to BridgeDiagnostics.to_json().
    json.dumps(gate, allow_nan=False)


def test_pick_place_real_q5_offset_advances_the_ee_gate(pp_adapter):
    """Regression for the 2026-07-21 real PREGRASP deadlock."""
    from hydrax.tasks.panda_pick_place import Phase

    pp_adapter.reset_runtime_state_after_warmup()
    task = pp_adapter._task
    pm = pp_adapter._phase_machine
    pm.plan_time = float(task.segment_end_times[Phase.PREGRASP])
    q = np.asarray(task.phase_goal_q[Phase.PREGRASP], dtype=np.float64).copy()
    q[4] += 0.0528
    planner_input = _state_input(q, np.zeros(7, dtype=np.float64))

    required = None
    for index in range(5):
        out = pp_adapter.step(planner_input)
        gate = out.diagnostics.phase_machine
        if index == 0:
            required = gate["consecutive_required_cycles"]
            assert required == 5
            assert gate["q_error_max_rad"] == pytest.approx(0.0528)
            assert gate["ee_position_error_norm_m"] < 0.004
            assert gate["transition_blockers"] == ()
        if index < 4:
            assert pm.phase == Phase.PREGRASP

    assert required == 5
    assert pm.phase == Phase.DESCEND


def test_pick_place_uses_one_cpu_kinematics_pass_per_step(
    pp_adapter, monkeypatch
):
    pp_adapter.reset_runtime_state_after_warmup()
    calls = {"forward": 0, "velocity": 0}
    original_forward = pp_adapter._mujoco.mj_forward
    original_velocity = pp_adapter._mujoco.mj_objectVelocity

    def counted_forward(*args, **kwargs):
        calls["forward"] += 1
        return original_forward(*args, **kwargs)

    def counted_velocity(*args, **kwargs):
        calls["velocity"] += 1
        return original_velocity(*args, **kwargs)

    monkeypatch.setattr(pp_adapter._mujoco, "mj_forward", counted_forward)
    monkeypatch.setattr(
        pp_adapter._mujoco, "mj_objectVelocity", counted_velocity
    )

    planner_input = _converged_input(pp_adapter)
    planner_input.v[:] = np.linspace(-0.03, 0.03, 7)
    out = pp_adapter.step(planner_input)
    assert calls == {"forward": 1, "velocity": 1}

    # Pin MuJoCo's spatial-vector ordering: objectVelocity is angular then
    # linear, while mj_jacSite exposes the same world-frame components as
    # separate translational/rotational Jacobians.
    model = pp_adapter._cpu_model
    data = pp_adapter._mujoco.MjData(model)
    data.qpos[:] = planner_input.q
    data.qvel[:] = planner_input.v
    original_forward(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    pp_adapter._mujoco.mj_jacSite(
        model, data, jacp, jacr, pp_adapter._site_id
    )
    gate = out.diagnostics.phase_machine
    np.testing.assert_allclose(
        gate["ee_linear_velocity_m_s"], jacp @ planner_input.v, atol=1e-12
    )
    np.testing.assert_allclose(
        gate["ee_angular_velocity_rad_s"], jacr @ planner_input.v, atol=1e-12
    )


def test_pick_place_walks_the_sequence_and_keeps_exact_feedback(pp_adapter):
    """The full sequence through the adapter contract, converged inputs.

    The selected law is invariant across phases: every cycle publishes
    the solve's F-MPPI K in exact_feedback, including CLOSE/OPEN precision
    windows. The stationary dwell reference remains available for
    diagnostics but never changes the LFC anchor or gain mode.
    """
    from hydrax.tasks.panda_pick_place import Phase, _DWELL_PHASES

    pp_adapter.reset_runtime_state_after_warmup()
    task = pp_adapter._task
    opts = task.options
    impedance = -np.hstack([np.diag(opts.kp_fixed), np.diag(opts.kd_fixed)])
    pm = pp_adapter._phase_machine

    phases_seen: list[str] = []
    grip_flips: list[tuple[str, str, float]] = []  # (command, phase, t_in)
    last_command = None
    max_steps = int(task.duration / task.dt) + 100
    for _ in range(max_steps):
        out = pp_adapter.step(_converged_input(pp_adapter))
        phase = pm.phase
        if not phases_seen or phases_seen[-1] != out.phase:
            phases_seen.append(out.phase)
        assert out.phase == phase.name
        assert out.gripper_command in ("open", "close")
        if out.gripper_command != last_command and last_command is not None:
            start = float(task.segment_end_times[phase - 1])
            grip_flips.append(
                (out.gripper_command, out.phase, pm.plan_time - start)
            )
        last_command = out.gripper_command

        assert out.diagnostics.gain_mode == "exact_feedback"
        np.testing.assert_allclose(
            out.K,
            np.asarray(pp_adapter._params.gains, dtype=np.float64),
        )
        assert not np.allclose(out.K, impedance)
        assert out.diagnostics.gain_ess >= 1.0
        assert 0.0 <= out.diagnostics.gain_nominal_weight <= 1.0
        assert out.diagnostics.gain_norm == pytest.approx(np.linalg.norm(out.K))

        if phase in _DWELL_PHASES:
            # The precision window and stationary reference remain visible,
            # but neither selects a different feedback law.
            assert out.diagnostics.phase_machine["precision_hold"] is True
            np.testing.assert_allclose(
                out.reference_q, task.phase_goal_q[phase], atol=1e-5
            )
            np.testing.assert_allclose(out.reference_v, np.zeros(7), atol=1e-5)
        if phase == Phase.DONE:
            break

    # the full sequence, in order, no skips
    assert phases_seen == [p.name for p in Phase]
    # exactly two gripper flips, each at its dwell's settle instant
    assert [(c, p) for c, p, _ in grip_flips] == [
        ("close", "CLOSE"),
        ("open", "OPEN"),
    ]
    for _, _, t_in in grip_flips:
        assert t_in == pytest.approx(opts.dwell_settle_sec, abs=task.dt)
    # the plan clock stayed a python float (traced-dtype recompile guard)
    assert type(pm.plan_time) is float


def test_pick_place_blocked_dwell_entry_keeps_exact_feedback(pp_adapter):
    """The real-run precision wait must not request fixed impedance."""
    from hydrax.tasks.panda_pick_place import Phase

    pp_adapter.reset_runtime_state_after_warmup()
    task = pp_adapter._task
    pm = pp_adapter._phase_machine
    pm.plan_time = float(task.segment_end_times[Phase.DESCEND])

    q = np.asarray(task.phase_goal_q[Phase.DESCEND], dtype=np.float64).copy()
    # Exact action pose but still moving in task space: the gripper must wait,
    # while the selected exact-feedback law remains unchanged.
    v = np.zeros(7, dtype=np.float64)
    v[0] = 1.0
    planner_input = _state_input(q, v)
    out = pp_adapter.step(planner_input)

    gate = out.diagnostics.phase_machine
    assert out.phase == gate["phase"] == "DESCEND"
    assert gate["at_boundary"] is True
    assert gate["transition_status"].startswith("blocked_ee_")
    assert any(
        blocker in gate["transition_blockers"]
        for blocker in ("ee_linear_speed", "ee_angular_speed")
    )
    assert gate["precision_hold"] is True
    assert out.diagnostics.gain_mode == "exact_feedback"

    opts = task.options
    impedance = -np.hstack([np.diag(opts.kp_fixed), np.diag(opts.kd_fixed)])
    np.testing.assert_allclose(
        out.K,
        np.asarray(pp_adapter._params.gains, dtype=np.float64),
    )
    assert not np.allclose(out.K, impedance)

    # Exact feedback remains anchored at the measured solve state, not the
    # stationary dwell reference, while the precision gate is blocked.
    control = planner_output_to_control(out, planner_input)
    np.testing.assert_allclose(control.initial_state.joint_state.position, q)
    np.testing.assert_allclose(control.initial_state.joint_state.velocity, v)
    np.testing.assert_allclose(out.reference_q, q, atol=1e-6)
    assert not np.allclose(out.reference_v, v)


def test_pick_place_gripper_wait_freezes_the_clock(pp_adapter):
    from hydrax.tasks.panda_pick_place import Phase

    pp_adapter.reset_runtime_state_after_warmup()
    pm = pp_adapter._phase_machine
    pm.plan_time = float(
        pp_adapter._task.segment_end_times[Phase.PREGRASP]
    )
    planner_input = _converged_input(pp_adapter)

    out = pp_adapter.step(planner_input)
    t_after_first = pm.plan_time
    assert t_after_first > 0.0
    eligible_after_first = pm._eligible_cycles
    assert eligible_after_first == 1

    pp_adapter.set_gripper_wait(True)
    frozen_first = pp_adapter.step(planner_input)
    frozen_second = pp_adapter.step(planner_input)
    assert pm.plan_time == t_after_first  # the clock did not move
    assert pm._eligible_cycles == eligible_after_first
    assert frozen_first.phase == frozen_second.phase == out.phase
    frozen_gate = frozen_second.diagnostics.phase_machine
    assert frozen_gate is not None
    assert frozen_gate["clock_paused"] is True
    assert frozen_gate["clock_pause_reason"] == "gripper_action"
    assert frozen_gate["plan_time_sec"] == pytest.approx(t_after_first)
    # frozen cycles still publish a full, valid control
    validate_planner_output(frozen_second.tau_ff, frozen_second.K)

    pp_adapter.set_gripper_wait(False)
    resumed = pp_adapter.step(planner_input)
    assert pm.plan_time == t_after_first  # still completing the fixed streak
    assert pm._eligible_cycles == eligible_after_first + 1
    resumed_gate = resumed.diagnostics.phase_machine
    assert resumed_gate is not None
    assert resumed_gate["clock_paused"] is False
    assert resumed_gate["clock_pause_reason"] is None

    # reset rebuilds a fresh machine and clears the wait flag
    pp_adapter.reset_runtime_state_after_warmup()
    assert pp_adapter._phase_machine.plan_time == 0.0
    assert pp_adapter._gripper_wait is False
