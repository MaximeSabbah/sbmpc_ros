from __future__ import annotations

import numpy as np
import pytest

from sbmpc_ros_bridge.safety import (
    SBMPC_TO_LFC_GAIN_SCALE,
    PlanningDeadlineMonitor,
    UnsafeControlError,
    compute_lfc_control,
    compute_lfc_state_error,
    sbmpc_gain_to_lfc_gain,
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


def test_validate_planner_output_passes_finite_correctly_shaped_outputs() -> None:
    tau = np.asarray([2.0, -3.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    gain = np.eye(7, 14, dtype=np.float64)

    checked_tau, checked_gain = validate_planner_output(tau, gain)

    np.testing.assert_allclose(checked_tau, tau)
    np.testing.assert_allclose(checked_gain, gain)


def test_validate_planner_output_rejects_bad_shape_or_non_finite_values() -> None:
    with pytest.raises(UnsafeControlError, match="feedforward must have shape"):
        validate_planner_output(
            np.zeros(6, dtype=np.float64),
            np.zeros((7, 14), dtype=np.float64),
        )

    with pytest.raises(UnsafeControlError, match="feedback_gain must have shape"):
        validate_planner_output(
            np.zeros(7, dtype=np.float64),
            np.zeros((7, 13), dtype=np.float64),
        )

    nan_tau = np.zeros(7, dtype=np.float64)
    nan_tau[0] = np.nan
    with pytest.raises(UnsafeControlError, match="feedforward contains non-finite"):
        validate_planner_output(nan_tau, np.zeros((7, 14), dtype=np.float64))

    nan_gain = np.zeros((7, 14), dtype=np.float64)
    nan_gain[0, 0] = np.inf
    with pytest.raises(UnsafeControlError, match="feedback_gain contains non-finite"):
        validate_planner_output(np.zeros(7, dtype=np.float64), nan_gain)


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
