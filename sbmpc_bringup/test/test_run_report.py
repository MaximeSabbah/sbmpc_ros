from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sbmpc_bringup.run_report import (
    ControlSeries,
    DiagnosticSeries,
    JointSeries,
    RunData,
    _arm_indices,
    _joint_row,
    _plot_agimus_robot_state,
    _plot_phase,
    _robot_state_row,
    _resolve_bag,
    align_run,
    causal_indices,
    nearest_indices,
    summarize,
)


def joint_series(times: list[float], q: np.ndarray) -> JointSeries:
    count = len(times)
    return JointSeries(
        receive=np.asarray(times),
        stamp=np.asarray(times),
        q=np.asarray(q, dtype=float),
        v=np.zeros((count, 7)),
        effort=np.zeros((count, 7)),
    )


def synthetic_run(*, phase_machine: bool = False) -> RunData:
    times = np.asarray([10.0, 10.04, 10.08])
    q = np.zeros((3, 7))
    q[:, 0] = [0.1, 0.05, 0.02]
    gain = np.zeros((3, 7, 14))
    gain[:, 0, 0] = 2.0
    diagnostics = []
    for index, error in enumerate((0.1, 0.05, 0.02), start=1):
        phase = None
        if phase_machine:
            phase = {
                "phase": "PREGRASP",
                "plan_time_sec": 0.04 * (index - 1),
                "at_boundary": index == 3,
                "transition_blocked": index == 3,
                "q_ok": False,
                "velocity_ok": True,
                "ee_position_error_norm_m": error,
            }
        diagnostics.append(
            {
                "planner_step_count": index,
                "planner_mode": "exact_feedback",
                "last_phase": "PREGRASP",
                "last_reference_q": [0.0] * 7,
                "last_reference_v": [0.0] * 7,
                "last_goal_position": [0.5, 0.0, 0.18],
                "last_position_error": error,
                "last_planning_time_ms": 25.0,
                "last_planner_step_wall_time_ms": 26.0,
                "accepted_planner_output_count": index,
                "published_control_count": index,
                "rejected_planner_output_count": 0,
                "deadline_miss_count": 0,
                "rejected_sensor_count": 0,
                "last_error": "",
                "phase_machine": phase,
                "gripper": {"stage": "idle", "goal_count": 0},
            }
        )
    state = joint_series(times.tolist(), q)
    return RunData(
        bag=Path("/tmp/example/rosbag"),
        control=ControlSeries(
            receive=times,
            stamp=times,
            anchor_stamp=times - 0.04,
            feedforward=np.ones((3, 7)),
            gain=gain,
            anchor_q=np.zeros((3, 7)),
            anchor_v=np.zeros((3, 7)),
        ),
        sensor=state,
        output=joint_series(times.tolist(), q),
        hardware=joint_series(times.tolist(), np.zeros((3, 7))),
        merged=None,
        desired=None,
        robot_state=[],
        diagnostics=DiagnosticSeries(receive=times, rows=diagnostics),
        rosout=[],
        topic_counts={
            "/control": 3,
            "/sensor": 3,
            "/output_joint_effort": 3,
            "/franka/joint_states": 3,
            "/franka_robot_state_broadcaster/robot_state": 0,
            "/franka_robot_state_broadcaster/desired_joint_states": 0,
        },
    )


def test_nearest_indices_selects_closest_rows() -> None:
    source = np.asarray([0.0, 1.0, 2.0])
    query = np.asarray([0.1, 0.6, 1.9])

    assert nearest_indices(source, query).tolist() == [0, 1, 2]


def test_causal_indices_never_select_a_row_before_the_query() -> None:
    source = np.asarray([0.0, 1.0, 2.0])
    query = np.asarray([0.1, 1.0, 1.9])

    assert causal_indices(source, query).tolist() == [1, 1, 2]


def test_sensor_row_uses_outer_sensor_header_and_named_joint_order() -> None:
    names = list(reversed([f"fer_joint{i}" for i in range(1, 8)]))
    joint_state = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0)),
        name=names,
        position=list(range(7)),
        velocity=[0.0] * 7,
        effort=[0.0] * 7,
    )
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=500_000_000)),
        joint_state=joint_state,
    )

    row = _joint_row(message, 13.0, sensor=True)

    assert row["stamp"] == pytest.approx(12.5)
    assert row["q"] == list(reversed(range(7)))


def test_arm_mapping_rejects_partial_or_unknown_names() -> None:
    with pytest.raises(ValueError, match="unambiguous"):
        _arm_indices(["wrong_joint"] * 7, 7)


def test_agimus_robot_state_row_preserves_tau_j_d_mode_and_errors() -> None:
    class BooleanFields(SimpleNamespace):
        def get_fields_and_field_types(self) -> dict[str, str]:
            return {name: "boolean" for name in vars(self)}

    vector = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    joint = SimpleNamespace(
        name=[f"fer_joint{i}" for i in range(1, 8)],
        position=[0.0] * 7,
        velocity=[0.0] * 7,
        effort=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    )
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=2, nanosec=0)),
        measured_joint_state=joint,
        desired_joint_state=joint,
        control_command_success_rate=0.99,
        robot_mode=4,
        current_errors=BooleanFields(joint_velocity_violation=True, joint_reflex=False),
        last_motion_errors=BooleanFields(controller_torque_discontinuity=True),
        collision_indicators=SimpleNamespace(
            is_cartesian_linear_collision=vector,
            is_cartesian_angular_collision=vector,
            is_joint_collision=[0.0] * 7,
        ),
    )

    row = _robot_state_row(message, 2.1)

    assert row["desired_effort"] == joint.effort
    assert row["robot_mode"] == 4
    assert row["current_errors"] == ["joint_velocity_violation"]
    assert row["last_motion_errors"] == ["controller_torque_discontinuity"]


def test_agimus_state_is_summarized_and_plotted_when_available(
    tmp_path: Path,
) -> None:
    robot_row = {
        "receive": 10.04,
        "stamp": 10.04,
        "measured_effort": [0.0] * 7,
        "desired_effort": [0.1] * 7,
        "control_command_success_rate": 0.99,
        "robot_mode": 2,
        "current_errors": [],
        "last_motion_errors": ["controller_torque_discontinuity"],
        "collision_indicator_max": 0.0,
    }
    run = replace(synthetic_run(), robot_state=[robot_row])
    aligned = align_run(run)

    summary = summarize(aligned, terminal_sec=1.0)
    filename = _plot_agimus_robot_state(aligned, tmp_path)

    assert summary["agimus_robot_state"]["available"] is True
    assert summary["agimus_robot_state"]["robot_mode_counts"] == {"MOVE": 1}
    assert summary["agimus_robot_state"]["last_motion_errors_seen"] == [
        "controller_torque_discontinuity"
    ]
    assert filename == "09_agimus_robot_state.png"
    assert (tmp_path / filename).is_file()


def test_resolve_bag_accepts_run_or_direct_bag(tmp_path: Path) -> None:
    bag = tmp_path / "run" / "rosbag"
    bag.mkdir(parents=True)
    (bag / "metadata.yaml").write_text("rosbag2_bagfile_information: {}")

    assert _resolve_bag(tmp_path / "run") == bag.resolve()
    assert _resolve_bag(bag) == bag.resolve()


def test_alignment_reconstructs_deployed_lfc_feedback_sign() -> None:
    aligned = align_run(synthetic_run())

    # K_msg=+2 and [anchor-current]=-0.1 => signed correction=-0.2 Nm.
    assert aligned.feedback[0, 0] == pytest.approx(-0.2)
    assert aligned.feedback[-1, 0] == pytest.approx(-0.04)


def test_summary_separates_bias_runtime_and_observability() -> None:
    summary = summarize(align_run(synthetic_run()), terminal_sec=1.0)

    assert summary["recorded_goal_position_m"] == [0.5, 0.0, 0.18]
    assert summary["final_q_minus_reference_rad"][0] == pytest.approx(0.02)
    assert summary["final_q_minus_local_anchor_rad"][0] == pytest.approx(0.02)
    assert summary["terminal_position_error_mm"]["median"] == pytest.approx(50.0)
    assert summary["control_period_ms"]["median"] == pytest.approx(40.0)
    assert summary["solve_anchor_age_ms"]["median"] == pytest.approx(40.0)
    assert summary["observability"]["post_rate_limit_tau_J_d_available"] is False
    assert summary["torque_rate_audit"]["component_steps_over_threshold_total"] == 0
    assert summary["state_stream_consistency"]["all_recorded_state_values_finite"]


def test_summary_prefers_phase_goal_without_losing_task_goal() -> None:
    run = synthetic_run(phase_machine=True)
    for row in run.diagnostics.rows:
        row["phase_machine"]["ee_goal_position_m"] = [0.5, 0.0, 0.18]
        row["last_goal_position"] = [0.65, 0.0, 0.105]

    summary = summarize(align_run(run), terminal_sec=1.0)

    assert summary["recorded_goal_position_m"] == [0.5, 0.0, 0.18]
    assert summary["recorded_phase_goal_position_m"] == [0.5, 0.0, 0.18]
    assert summary["recorded_task_goal_position_m"] == [0.65, 0.0, 0.105]


def test_phase_plot_is_automatic_only_when_phase_machine_is_recorded(
    tmp_path: Path,
) -> None:
    assert _plot_phase(align_run(synthetic_run()), tmp_path) is None

    filename = _plot_phase(align_run(synthetic_run(phase_machine=True)), tmp_path)
    assert filename == "07_phase_and_gripper.png"
    assert (tmp_path / filename).is_file()
