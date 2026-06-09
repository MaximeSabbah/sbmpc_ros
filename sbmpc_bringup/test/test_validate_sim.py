from __future__ import annotations

import numpy as np
import pytest

from sbmpc_bringup.validate_sim import (
    JointRecord,
    assert_stable,
    joint_indices,
    summarize,
    vector_from_indices,
)


def test_joint_helpers_extract_fer_arm_order() -> None:
    names = ["foo", "fer_joint2", "fer_joint1", "fer_joint3", "fer_joint4", "fer_joint5", "fer_joint6", "fer_joint7"]
    indices = joint_indices(names)

    assert indices == (2, 1, 3, 4, 5, 6, 7)
    np.testing.assert_allclose(
        vector_from_indices([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], indices),
        [30.0, 20.0, 40.0, 50.0, 60.0, 70.0, 80.0],
    )


def test_summarize_reports_tail_joint_spans_and_stability() -> None:
    diagnostics = [
        {
            "state": "running",
            "planner_mode": "exact_feedback",
            "last_position_error": 0.04,
            "last_foreground_planning_time_ms": 18.0,
            "last_bridge_loop_time_ms": 19.0,
            "accepted_planner_output_count": 3,
            "rejected_planner_output_count": 0,
            "last_gain_norm": 2.0,
            "last_gain_norm": 5.0,
            "deadline_miss_count": 0,
        },
        {
            "state": "running",
            "planner_mode": "exact_feedback",
            "last_position_error": 0.02,
            "last_foreground_planning_time_ms": 17.0,
            "last_bridge_loop_time_ms": 18.0,
            "accepted_planner_output_count": 4,
            "rejected_planner_output_count": 0,
            "last_gain_norm": 3.0,
            "deadline_miss_count": 0,
        },
    ]
    joint_records = [
        JointRecord(0.0, np.zeros(7), np.zeros(7)),
        JointRecord(1.0, np.full(7, 0.01), np.full(7, 0.1)),
        JointRecord(2.0, np.full(7, 0.02), np.full(7, 0.2)),
        JointRecord(3.0, np.full(7, 0.03), np.full(7, 0.3)),
    ]

    summary = summarize(diagnostics, joint_records, tail_fraction=0.5)

    assert summary.running_count == 2
    assert summary.planner_mode == "exact_feedback"
    assert summary.final_position_error == 0.02
    assert summary.max_foreground_ms == 18.0
    assert summary.accepted_planner_output_count == 4
    assert summary.rejected_planner_output_count == 0
    assert summary.max_tail_joint_span == pytest.approx(0.01)
    assert summary.joint_velocity_abs_max == 0.3
    assert assert_stable(
        summary,
        max_tail_joint_span=0.02,
        max_final_position_error=0.05,
    )
    assert assert_stable(
        summary,
        max_tail_joint_span=0.02,
        max_final_position_error=0.05,
        max_foreground_ms=20.0,
    )


def _running_diagnostics() -> list[dict]:
    return [
        {
            "state": "running",
            "last_position_error": 0.0005,
            "accepted_planner_output_count": 2,
            "rejected_planner_output_count": 0,
            "last_gain_norm": 2.0,
        }
    ]


def test_limit_gate_fails_on_torque_or_velocity_over_limit() -> None:
    # j2 effort 86 Nm (99% of 87) and j2 velocity 2.7 rad/s (>2.62 FR3 limit).
    over_limit = [
        JointRecord(0.0, np.zeros(7), np.zeros(7), np.zeros(7)),
        JointRecord(
            1.0,
            np.full(7, 0.001),
            np.array([0.0, 2.7, 0, 0, 0, 0, 0]),
            np.array([0.0, 86.0, 0, 0, 0, 0, 0]),
        ),
    ]
    summary = summarize(_running_diagnostics(), over_limit, tail_fraction=0.5)

    assert summary.worst_velocity_fraction == pytest.approx(2.7 / 2.62)
    assert summary.worst_torque_fraction == pytest.approx(86.0 / 87.0)
    assert not assert_stable(
        summary,
        max_tail_joint_span=0.1,
        max_final_position_error=0.05,
        max_torque_fraction=0.9,
        max_velocity_fraction=0.9,
    )
    # Without the limit gates the same run is still considered stable.
    assert assert_stable(
        summary,
        max_tail_joint_span=0.1,
        max_final_position_error=0.05,
    )


def test_limit_gate_passes_within_limits() -> None:
    within = [
        JointRecord(0.0, np.zeros(7), np.zeros(7), np.zeros(7)),
        JointRecord(
            1.0,
            np.full(7, 0.001),
            np.full(7, 0.5),
            np.array([20.0, 30.0, 20.0, 15.0, 3.0, 5.0, 1.0]),
        ),
    ]
    summary = summarize(_running_diagnostics(), within, tail_fraction=0.5)

    assert summary.worst_torque_fraction < 0.9
    assert summary.worst_velocity_fraction < 0.9
    assert assert_stable(
        summary,
        max_tail_joint_span=0.1,
        max_final_position_error=0.05,
        max_torque_fraction=0.9,
        max_velocity_fraction=0.9,
    )
