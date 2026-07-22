from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sbmpc_bringup.constants import FER_ARM_JOINT_NAMES
from sbmpc_bringup.replay import (
    _load_payload,
    export_replay_json,
    replay_payload_from_run_data,
    states_from_payload,
)


def decoded_joint_series(
    times: list[float],
    *,
    position_offset: float = 0.0,
    effort_offset: float = 0.0,
) -> SimpleNamespace:
    count = len(times)
    return SimpleNamespace(
        receive=np.asarray(times, dtype=float),
        stamp=np.asarray(times, dtype=float) - 0.001,
        q=np.asarray(
            [
                [position_offset + index + 0.1 * joint for joint in range(7)]
                for index in range(count)
            ],
            dtype=float,
        ),
        v=np.full((count, 7), 0.25, dtype=float),
        effort=np.asarray(
            [
                [effort_offset + index + joint for joint in range(7)]
                for index in range(count)
            ],
            dtype=float,
        ),
    )


def decoded_run_data(*, merged: bool = True) -> SimpleNamespace:
    times = [10.0, 10.04]
    return SimpleNamespace(
        merged=(
            decoded_joint_series(times, position_offset=1.0)
            if merged
            else None
        ),
        sensor=decoded_joint_series(times, position_offset=2.0),
        output=decoded_joint_series(times, effort_offset=3.0),
        hardware=decoded_joint_series(times, effort_offset=4.0),
        control=SimpleNamespace(
            receive=np.asarray(times),
            stamp=np.asarray(times) - 0.002,
            anchor_stamp=np.asarray(times) - 0.004,
            feedforward=np.asarray(
                [[1.0] * 7, [2.0] * 7],
                dtype=float,
            ),
            gain=np.asarray(
                [np.eye(7, 14), 2.0 * np.eye(7, 14)],
                dtype=float,
            ),
            anchor_q=np.asarray([[0.1] * 7, [0.2] * 7]),
            anchor_v=np.asarray([[0.3] * 7, [0.4] * 7]),
        ),
        diagnostics=SimpleNamespace(
            receive=np.asarray(times),
            rows=[
                {
                    "state": "running",
                    "planner_step_count": 1,
                    "published_control_count": 1,
                    "last_planning_time_ms": 4.0,
                },
                {
                    "state": "running",
                    "planner_step_count": 2,
                    "published_control_count": 2,
                    "last_planning_time_ms": 5.0,
                },
            ],
        ),
    )


def test_run_data_export_preserves_replay_fields_with_explicit_effort_semantics() -> None:
    payload = replay_payload_from_run_data(decoded_run_data())

    assert payload["schema"] == "sbmpc_ros_replay_v2"
    assert payload["backend"] == "real"
    assert payload["recorded_wall_time_sec"] == pytest.approx(0.04)
    assert payload["joint_names"] == list(FER_ARM_JOINT_NAMES)
    assert payload["topics"]["joint_states"] == "/joint_states"
    assert payload["joint_states"][0]["position"][0] == pytest.approx(1.0)
    assert payload["sensor_states"][0]["position"][0] == pytest.approx(2.0)

    control = payload["controls"][1]
    assert control["feedforward"] == [2.0] * 7
    assert control["feedforward_shape"] == [7, 1]
    assert control["feedback_gain_shape"] == [7, 14]
    assert control["initial_state"]["stamp_sec"] == pytest.approx(10.036)
    assert control["initial_state"]["position"] == [0.2] * 7
    assert control["initial_state_velocity_abs_max"] == pytest.approx(0.4)

    assert payload["lfc_output_efforts"][0]["effort"][0] == pytest.approx(3.0)
    assert payload["observed_joint_effort"][0]["effort"][0] == pytest.approx(4.0)
    assert "FCI measured" in payload["observed_joint_effort_semantics"]
    assert payload["diagnostics"][1]["planner_step_count"] == 2
    assert payload["summary"]["joint_state_count"] == 2
    assert payload["summary"]["sensor_state_count"] == 2
    assert payload["summary"]["control_count"] == 2
    assert payload["summary"]["lfc_output_effort_count"] == 2


def test_sim_export_does_not_label_actuator_effort_as_fci_torque() -> None:
    data = decoded_run_data()
    data.backend = "mujoco"
    data.hardware_source = "/joint_states"

    payload = replay_payload_from_run_data(data)

    assert payload["topics"]["observed_joint_effort"] == "/joint_states"
    assert "not Franka FCI" in payload["observed_joint_effort_semantics"]
    assert "measured_joint_torque" not in payload


def test_exported_run_data_loads_through_existing_replay_api(tmp_path: Path) -> None:
    output_path = tmp_path / "replay.json"
    written = export_replay_json(decoded_run_data(merged=False), output_path)

    loaded = _load_payload(output_path)
    assert loaded == written
    assert loaded["joint_states"] == []
    assert loaded["topics"]["joint_states"] is None
    states = states_from_payload(loaded, source="auto", time_source="receive")
    assert [state["position"][0] for state in states] == [2.0, 3.0]
    assert [state["t"] for state in states] == pytest.approx([0.0, 0.04])
