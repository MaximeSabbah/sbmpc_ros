from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from sbmpc_bringup.constants import FER_ARM_JOINT_NAMES
from sbmpc_bringup.replay import (
    _control_record_from_message,
    _print_record_summary,
    summarize_payload,
)


def stamp(sec: int = 1, nanosec: int = 2) -> SimpleNamespace:
    return SimpleNamespace(sec=sec, nanosec=nanosec)


def multi_array(data: list[float], shape: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        data=data,
        layout=SimpleNamespace(
            dim=[SimpleNamespace(size=size) for size in shape],
        ),
    )


def control_message() -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(stamp=stamp()),
        feedforward=multi_array([1.0, -2.0], [2, 1]),
        feedback_gain=multi_array([3.0, 4.0, 0.0, 0.0], [2, 2]),
        initial_state=SimpleNamespace(
            joint_state=SimpleNamespace(
                header=SimpleNamespace(stamp=stamp(1, 20)),
                name=list(FER_ARM_JOINT_NAMES),
                position=[0.1 * index for index in range(7)],
                velocity=[-0.2 * index for index in range(7)],
                effort=[],
            )
        ),
    )


def test_control_record_keeps_full_control_and_initial_state() -> None:
    record = _control_record_from_message(
        control_message(),
        receive_wall_sec=12.0,
    )

    assert record["receive_wall_sec"] == 12.0
    assert record["feedforward"] == [1.0, -2.0]
    assert record["feedforward_shape"] == [2, 1]
    assert record["feedforward_max_abs"] == 2.0
    assert record["feedback_gain"] == [3.0, 4.0, 0.0, 0.0]
    assert record["feedback_gain_shape"] == [2, 2]
    assert record["gain_norm"] == 5.0
    assert record["initial_state_velocity_abs_max"] == pytest.approx(1.2)
    initial_state = record["initial_state"]
    assert isinstance(initial_state, dict)
    assert initial_state["position"][-1] == pytest.approx(0.6)


def test_recorder_summary_reports_control_extrema(capsys) -> None:
    controls = [_control_record_from_message(control_message(), receive_wall_sec=12.0)]
    summary = summarize_payload([], [], controls, [])

    assert summary["control_feedforward_abs_max"] == 2.0
    assert summary["control_gain_norm_max"] == 5.0
    assert summary["control_initial_state_velocity_abs_max"] == pytest.approx(1.2)
    _print_record_summary(
        {
            "summary": summary,
        },
        Path("/tmp/example_replay.json"),
    )
    captured = capsys.readouterr()
    assert "control_feedforward_abs_max=2.0" in captured.out
