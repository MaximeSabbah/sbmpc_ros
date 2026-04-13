from __future__ import annotations

import numpy as np
import pytest

from sbmpc_ros_bridge.safety import (
    AlwaysOnSafety,
    BridgeSafetyProfile,
    BringupLimits,
    MonitoringOnly,
    SBMPC_TO_LFC_GAIN_SCALE,
    ControlSafetyLimits,
    PlanningDeadlineMonitor,
    UnsafeControlError,
    apply_gain_norm_limit,
    apply_torque_limit,
    compute_lfc_control,
    compute_control_age_sec,
    compute_lfc_state_error,
    make_conservative_bringup_profile,
    make_default_safety_profile,
    sbmpc_gain_to_lfc_gain,
    validate_control_age,
    validate_planner_output,
)


def test_compute_lfc_state_error_uses_desired_minus_measured() -> None:
    desired = np.asarray([1.0, 0.5])
    measured = np.asarray([1.3, -0.5])

    diff = compute_lfc_state_error(desired, measured)

    np.testing.assert_allclose(diff, np.asarray([-0.3, 1.0]))


def test_positive_position_gain_is_restoring_under_lfc_convention() -> None:
    feedforward = np.zeros(7, dtype=np.float64)
    feedback_gain = np.zeros((7, 14), dtype=np.float64)
    feedback_gain[0, 0] = 20.0
    desired_state = np.zeros(14, dtype=np.float64)
    measured_state = np.zeros(14, dtype=np.float64)
    measured_state[0] = 0.1

    control = compute_lfc_control(
        feedforward,
        feedback_gain,
        desired_state,
        measured_state,
    )

    assert control[0] < 0.0


def test_positive_velocity_gain_is_damping_under_lfc_convention() -> None:
    feedforward = np.zeros(7, dtype=np.float64)
    feedback_gain = np.zeros((7, 14), dtype=np.float64)
    feedback_gain[0, 7] = 8.0
    desired_state = np.zeros(14, dtype=np.float64)
    measured_state = np.zeros(14, dtype=np.float64)
    measured_state[7] = 0.25

    control = compute_lfc_control(
        feedforward,
        feedback_gain,
        desired_state,
        measured_state,
    )

    assert control[0] < 0.0


def test_sbmpc_measured_state_gain_must_be_negated_for_lfc() -> None:
    planner_jacobian = np.zeros((7, 14), dtype=np.float64)
    planner_jacobian[0, 0] = -20.0
    desired_state = np.zeros(14, dtype=np.float64)
    measured_state = np.zeros(14, dtype=np.float64)
    measured_state[0] = 0.1

    raw_control = compute_lfc_control(
        np.zeros(7, dtype=np.float64),
        planner_jacobian,
        desired_state,
        measured_state,
    )
    lfc_gain = sbmpc_gain_to_lfc_gain(planner_jacobian)
    safe_control = compute_lfc_control(
        np.zeros(7, dtype=np.float64),
        lfc_gain,
        desired_state,
        measured_state,
    )

    assert SBMPC_TO_LFC_GAIN_SCALE == -1.0
    assert raw_control[0] > 0.0
    assert safe_control[0] < 0.0


def test_validate_planner_output_rejects_limit_violations_by_default() -> None:
    with pytest.raises(UnsafeControlError, match="max_abs_torque"):
        validate_planner_output(
            tau_ff=np.asarray([2.0] + [0.0] * 6, dtype=np.float64),
            feedback_gain=np.zeros((7, 14), dtype=np.float64),
            limits=ControlSafetyLimits(max_abs_torque=1.0),
        )

    with pytest.raises(UnsafeControlError, match="max_gain_norm"):
        validate_planner_output(
            tau_ff=np.zeros(7, dtype=np.float64),
            feedback_gain=np.full((7, 14), 10.0, dtype=np.float64),
            limits=ControlSafetyLimits(max_gain_norm=1.0),
        )


def test_validate_planner_output_can_clip_torque_when_requested() -> None:
    tau_ff, gain = validate_planner_output(
        tau_ff=np.asarray([2.0, -3.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        feedback_gain=np.eye(7, 14, dtype=np.float64),
        limits=ControlSafetyLimits(
            max_abs_torque=1.0,
            torque_limit_mode="clip",
        ),
    )

    np.testing.assert_allclose(
        tau_ff,
        np.asarray([1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    assert float(np.max(np.abs(tau_ff))) == 1.0
    assert gain.shape == (7, 14)


def test_validate_planner_output_can_scale_gain_norm_when_requested() -> None:
    _, gain = validate_planner_output(
        tau_ff=np.zeros(7, dtype=np.float64),
        feedback_gain=np.full((7, 14), 4.0, dtype=np.float64),
        limits=ControlSafetyLimits(
            max_gain_norm=2.0,
            gain_limit_mode="scale",
        ),
    )

    assert np.isclose(np.linalg.norm(gain), 2.0)


def test_apply_limit_helpers_reject_unknown_modes() -> None:
    with pytest.raises(ValueError, match="unsupported torque limit mode"):
        apply_torque_limit(
            np.asarray([2.0] + [0.0] * 6, dtype=np.float64),
            max_abs_torque=1.0,
            mode="bad",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="unsupported gain limit mode"):
        apply_gain_norm_limit(
            np.full((7, 14), 2.0, dtype=np.float64),
            max_gain_norm=1.0,
            mode="bad",  # type: ignore[arg-type]
        )


def test_compute_control_age_and_stale_detection() -> None:
    age = compute_control_age_sec(control_stamp_sec=10.0, now_sec=10.04)
    assert np.isclose(age, 0.04)

    fresh_age = validate_control_age(
        control_stamp_sec=10.0,
        now_sec=10.04,
        max_control_age_sec=0.05,
    )
    assert np.isclose(fresh_age, 0.04)

    with pytest.raises(UnsafeControlError, match="control is stale"):
        validate_control_age(
            control_stamp_sec=10.0,
            now_sec=10.2,
            max_control_age_sec=0.05,
        )

    with pytest.raises(UnsafeControlError, match="in the future"):
        compute_control_age_sec(control_stamp_sec=10.0, now_sec=9.9)


def test_planning_deadline_monitor_counts_misses_without_rejecting_when_configured() -> None:
    monitor = PlanningDeadlineMonitor(
        max_planning_duration_sec=0.02,
        fail_closed=False,
    )

    assert monitor.observe(0.015) is True
    assert monitor.deadline_miss_count == 0
    assert monitor.observe(0.03) is False
    assert monitor.deadline_miss_count == 1
    assert np.isclose(monitor.last_planning_duration_sec, 0.03)


def test_planning_deadline_monitor_can_fail_closed() -> None:
    monitor = PlanningDeadlineMonitor(
        max_planning_duration_sec=0.02,
        fail_closed=True,
    )

    with pytest.raises(UnsafeControlError, match="planner deadline missed"):
        monitor.observe(0.03)

    assert monitor.deadline_miss_count == 1


def test_planning_deadline_monitor_rejects_invalid_durations() -> None:
    monitor = PlanningDeadlineMonitor(max_planning_duration_sec=0.02)

    with pytest.raises(UnsafeControlError, match="must be finite"):
        monitor.observe(np.nan)

    with pytest.raises(UnsafeControlError, match="must be non-negative"):
        monitor.observe(-0.01)


def test_default_safety_profile_is_minimal_and_non_restrictive() -> None:
    profile = make_default_safety_profile()

    assert isinstance(profile.always_on, AlwaysOnSafety)
    assert isinstance(profile.bringup_limits, BringupLimits)
    assert isinstance(profile.monitoring_only, MonitoringOnly)
    assert profile.always_on.gain_scale == -1.0
    assert profile.bringup_limits.max_abs_torque is None
    assert profile.bringup_limits.max_gain_norm is None
    assert profile.make_deadline_monitor() is None


def test_conservative_bringup_profile_enables_optional_limits_and_monitoring() -> None:
    profile = make_conservative_bringup_profile(
        max_control_age_sec=0.05,
        max_planning_duration_sec=0.02,
        max_abs_torque=3.0,
        max_gain_norm=4.0,
    )

    assert profile.always_on.max_control_age_sec == 0.05
    assert profile.bringup_limits.max_abs_torque == 3.0
    assert profile.bringup_limits.torque_limit_mode == "clip"
    assert profile.bringup_limits.max_gain_norm == 4.0
    assert profile.bringup_limits.gain_limit_mode == "scale"

    monitor = profile.make_deadline_monitor()
    assert monitor is not None
    assert monitor.fail_closed is False
    assert np.isclose(monitor.max_planning_duration_sec, 0.02)


def test_always_on_safety_can_validate_control_age_directly() -> None:
    always_on = AlwaysOnSafety(max_control_age_sec=0.05)

    age = always_on.validate_control_age(control_stamp_sec=10.0, now_sec=10.02)
    assert np.isclose(age, 0.02)

    with pytest.raises(UnsafeControlError, match="control is stale"):
        always_on.validate_control_age(control_stamp_sec=10.0, now_sec=10.2)
