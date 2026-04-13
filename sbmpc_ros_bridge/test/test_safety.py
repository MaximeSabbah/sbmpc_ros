from __future__ import annotations

import numpy as np
import pytest

from sbmpc_ros_bridge.safety import (
    ControlSafetyLimits,
    UnsafeControlError,
    compute_lfc_control,
    compute_lfc_state_error,
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


def test_validate_planner_output_rejects_limit_violations() -> None:
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
