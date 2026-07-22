"""Offline, repeatable diagnostics for a recorded SB-MPC real-robot run.

The reporter reads an MCAP after the experiment has stopped.  It never runs in
the planner or low-level controller processes, so generating the figures has no
effect on controller real-time performance.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sbmpc_bringup.constants import FER_ARM_JOINT_NAMES


TOPICS = {
    "/control",
    "/sensor",
    "/output_joint_effort",
    "/franka/joint_states",
    "/joint_states",
    "/sbmpc/diagnostics",
    "/rosout",
}

OBSERVABILITY_TOPICS = {
    "/franka_robot_state_broadcaster/robot_state",
    "/franka_robot_state_broadcaster/desired_joint_states",
    "/franka_robot_state_broadcaster/current_pose",
    "/franka_robot_state_broadcaster/external_joint_torques",
    "/franka_robot_state_broadcaster/external_wrench_in_base_frame",
}

ROBOT_STATE_TOPIC = "/franka_robot_state_broadcaster/robot_state"
DESIRED_STATE_TOPIC = "/franka_robot_state_broadcaster/desired_joint_states"
DECODED_TOPICS = TOPICS | {ROBOT_STATE_TOPIC, DESIRED_STATE_TOPIC}

# Agimus limits a torque-command component to just under 1 Nm per 1 kHz
# hardware cycle.  A pre-limiter trace below this threshold is a useful
# necessary-condition check; without tau_J_d it cannot prove what was applied.
AGIMUS_COMPONENT_STEP_AUDIT_NM = 0.999999


@dataclass(frozen=True)
class JointSeries:
    receive: np.ndarray
    stamp: np.ndarray
    q: np.ndarray
    v: np.ndarray
    effort: np.ndarray


@dataclass(frozen=True)
class ControlSeries:
    receive: np.ndarray
    stamp: np.ndarray
    anchor_stamp: np.ndarray
    feedforward: np.ndarray
    gain: np.ndarray
    anchor_q: np.ndarray
    anchor_v: np.ndarray


@dataclass(frozen=True)
class DiagnosticSeries:
    receive: np.ndarray
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class RunData:
    bag: Path
    control: ControlSeries
    sensor: JointSeries
    output: JointSeries
    hardware: JointSeries
    merged: JointSeries | None
    desired: JointSeries | None
    robot_state: list[dict[str, Any]]
    diagnostics: DiagnosticSeries
    rosout: list[dict[str, Any]]
    topic_counts: dict[str, int]


@dataclass(frozen=True)
class AlignedRun:
    data: RunData
    time: np.ndarray
    q: np.ndarray
    v: np.ndarray
    reference_q: np.ndarray
    reference_v: np.ndarray
    final_reference_q: np.ndarray
    feedforward: np.ndarray
    gain: np.ndarray
    feedback: np.ndarray
    output: np.ndarray
    measured_tau: np.ndarray
    diagnostics: list[dict[str, Any]]
    diagnostic_time: np.ndarray


def _stamp_sec(stamp: Any) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def _resolve_bag(path: Path) -> Path:
    path = path.resolve()
    candidates = (path, path / "rosbag")
    for candidate in candidates:
        if (candidate / "metadata.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        f"{path} is neither a rosbag2 directory nor a run directory containing rosbag/"
    )


def nearest_indices(source_time: np.ndarray, query_time: np.ndarray) -> np.ndarray:
    """Return the nearest source row for every monotonically ordered query."""
    if not len(source_time):
        raise ValueError("cannot align against an empty time series")
    right = np.clip(np.searchsorted(source_time, query_time), 0, len(source_time) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(
        np.abs(source_time[left] - query_time)
        <= np.abs(source_time[right] - query_time),
        left,
        right,
    )


def causal_indices(source_time: np.ndarray, query_time: np.ndarray) -> np.ndarray:
    """Return the first source row at or after each query timestamp."""
    if not len(source_time):
        raise ValueError("cannot align against an empty time series")
    return np.clip(np.searchsorted(source_time, query_time), 0, len(source_time) - 1)


def _arm_indices(names: list[str], vector_length: int) -> list[int]:
    if all(name in names for name in FER_ARM_JOINT_NAMES):
        return [names.index(name) for name in FER_ARM_JOINT_NAMES]
    if not names and vector_length == len(FER_ARM_JOINT_NAMES):
        return list(range(len(FER_ARM_JOINT_NAMES)))
    raise ValueError(
        "joint message cannot provide an unambiguous seven-arm-joint mapping: "
        f"names={names!r}, vector_length={vector_length}"
    )


def _joint_row(message: Any, receive_sec: float, *, sensor: bool) -> dict[str, Any]:
    joint = message.joint_state if sensor else message
    header = message.header if sensor else joint.header
    indices = _arm_indices(list(joint.name), len(joint.position))

    def selected(values: Any) -> list[float]:
        array = np.asarray(values, dtype=np.float64)
        if not len(array):
            return [0.0] * len(indices)
        return array[indices].tolist()

    return {
        "receive": receive_sec,
        "stamp": _stamp_sec(header.stamp),
        "q": selected(joint.position),
        "v": selected(joint.velocity),
        "effort": selected(joint.effort),
    }


def _active_boolean_fields(message: Any) -> list[str]:
    fields = message.get_fields_and_field_types()
    return [name for name in fields if bool(getattr(message, name))]


def _robot_state_row(message: Any, receive_sec: float) -> dict[str, Any]:
    measured = message.measured_joint_state
    desired = message.desired_joint_state
    measured_indices = _arm_indices(list(measured.name), len(measured.position))
    desired_indices = _arm_indices(list(desired.name), len(desired.position))
    collision = message.collision_indicators
    collision_values = np.asarray(
        [
            collision.is_cartesian_linear_collision.x,
            collision.is_cartesian_linear_collision.y,
            collision.is_cartesian_linear_collision.z,
            collision.is_cartesian_angular_collision.x,
            collision.is_cartesian_angular_collision.y,
            collision.is_cartesian_angular_collision.z,
            *collision.is_joint_collision,
        ],
        dtype=np.float64,
    )

    def selected(values: Any, indices: list[int]) -> list[float]:
        return np.asarray(values, dtype=np.float64)[indices].tolist()

    return {
        "receive": receive_sec,
        "stamp": _stamp_sec(message.header.stamp),
        "measured_effort": selected(measured.effort, measured_indices),
        "desired_effort": selected(desired.effort, desired_indices),
        "control_command_success_rate": float(message.control_command_success_rate),
        "robot_mode": int(message.robot_mode),
        "current_errors": _active_boolean_fields(message.current_errors),
        "last_motion_errors": _active_boolean_fields(message.last_motion_errors),
        "collision_indicator_max": float(np.max(np.abs(collision_values))),
    }


def _joint_series(rows: list[dict[str, Any]]) -> JointSeries:
    if not rows:
        raise ValueError("required joint-state stream is empty")
    rows = sorted(rows, key=lambda row: row["receive"])
    return JointSeries(
        receive=np.asarray([row["receive"] for row in rows]),
        stamp=np.asarray([row["stamp"] for row in rows]),
        q=np.asarray([row["q"] for row in rows]),
        v=np.asarray([row["v"] for row in rows]),
        effort=np.asarray([row["effort"] for row in rows]),
    )


def _optional_joint_series(rows: list[dict[str, Any]]) -> JointSeries | None:
    return _joint_series(rows) if rows else None


def _control_series(rows: list[dict[str, Any]]) -> ControlSeries:
    if not rows:
        raise ValueError("bag contains no /control packets")
    rows = sorted(rows, key=lambda row: row["receive"])
    return ControlSeries(
        receive=np.asarray([row["receive"] for row in rows]),
        stamp=np.asarray([row["stamp"] for row in rows]),
        anchor_stamp=np.asarray([row["anchor_stamp"] for row in rows]),
        feedforward=np.asarray([row["feedforward"] for row in rows]),
        gain=np.asarray([row["gain"] for row in rows]),
        anchor_q=np.asarray([row["anchor_q"] for row in rows]),
        anchor_v=np.asarray([row["anchor_v"] for row in rows]),
    )


def read_bag(path: Path) -> RunData:
    """Load only the diagnostic topics; no messages are replayed or published."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    bag = _resolve_bag(path)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    available_types = {
        topic.name: topic.type for topic in reader.get_all_topics_and_types()
    }
    monitored = TOPICS | OBSERVABILITY_TOPICS
    selected = monitored.intersection(available_types)
    decoded = DECODED_TOPICS.intersection(selected)
    message_types = {
        topic: get_message(available_types[topic]) for topic in decoded
    }
    raw: dict[str, list[dict[str, Any]]] = {topic: [] for topic in DECODED_TOPICS}
    topic_counts = {topic: 0 for topic in sorted(monitored)}
    while reader.has_next():
        topic, serialized, receive_ns = reader.read_next()
        if topic not in selected:
            continue
        topic_counts[topic] += 1
        if topic not in decoded:
            continue
        receive_sec = receive_ns * 1e-9
        message = deserialize_message(serialized, message_types[topic])
        if topic == "/control":
            initial = message.initial_state.joint_state
            sensor_stamp = getattr(message.initial_state, "header", None)
            anchor_stamp = (
                _stamp_sec(sensor_stamp.stamp)
                if sensor_stamp is not None
                else _stamp_sec(initial.header.stamp)
            )
            raw[topic].append(
                {
                    "receive": receive_sec,
                    "stamp": _stamp_sec(message.header.stamp),
                    "anchor_stamp": anchor_stamp,
                    "feedforward": list(message.feedforward.data),
                    "gain": np.asarray(message.feedback_gain.data).reshape(7, 14).tolist(),
                    "anchor_q": list(initial.position),
                    "anchor_v": list(initial.velocity),
                }
            )
        elif topic == "/sensor":
            raw[topic].append(_joint_row(message, receive_sec, sensor=True))
        elif topic in {"/output_joint_effort", "/franka/joint_states", "/joint_states"}:
            raw[topic].append(_joint_row(message, receive_sec, sensor=False))
        elif topic == DESIRED_STATE_TOPIC:
            raw[topic].append(_joint_row(message, receive_sec, sensor=False))
        elif topic == ROBOT_STATE_TOPIC:
            raw[topic].append(_robot_state_row(message, receive_sec))
        elif topic == "/sbmpc/diagnostics":
            raw[topic].append({"receive": receive_sec, "data": json.loads(message.data)})
        elif topic == "/rosout":
            raw[topic].append(
                {
                    "receive": receive_sec,
                    "level": int(message.level),
                    "name": message.name,
                    "message": message.msg,
                }
            )

    missing = [
        topic
        for topic in ("/control", "/sensor", "/output_joint_effort", "/franka/joint_states")
        if not raw[topic]
    ]
    if missing:
        raise ValueError(f"bag is missing required populated topics: {', '.join(missing)}")

    # Diagnostics are latched/published independently. Keep one row per planner
    # step so references and counters cannot be biased by duplicates.
    by_step: dict[int, dict[str, Any]] = {}
    for item in raw["/sbmpc/diagnostics"]:
        step = int(item["data"].get("planner_step_count", 0))
        if step > 0:
            by_step[step] = item
    diagnostic_items = [by_step[key] for key in sorted(by_step)]
    if not diagnostic_items:
        raise ValueError("bag contains no running planner diagnostics")

    return RunData(
        bag=bag,
        control=_control_series(raw["/control"]),
        sensor=_joint_series(raw["/sensor"]),
        output=_joint_series(raw["/output_joint_effort"]),
        hardware=_joint_series(raw["/franka/joint_states"]),
        merged=_optional_joint_series(raw["/joint_states"]),
        desired=_optional_joint_series(raw[DESIRED_STATE_TOPIC]),
        robot_state=sorted(raw[ROBOT_STATE_TOPIC], key=lambda row: row["receive"]),
        diagnostics=DiagnosticSeries(
            receive=np.asarray([item["receive"] for item in diagnostic_items]),
            rows=[item["data"] for item in diagnostic_items],
        ),
        rosout=sorted(raw["/rosout"], key=lambda row: row["receive"]),
        topic_counts=topic_counts,
    )


def align_run(data: RunData) -> AlignedRun:
    control = data.control
    # The first output published after a control packet is the earliest sample
    # that can contain that packet. Its embedded q/dq is copied from the exact
    # state used by LFC for that output, avoiding a second timestamp alignment.
    output_index = causal_indices(data.output.receive, control.receive)
    output_receive = data.output.receive[output_index]
    hardware_index = nearest_indices(data.hardware.receive, output_receive)
    diagnostic_index = nearest_indices(data.diagnostics.receive, control.receive)
    diagnostics = [data.diagnostics.rows[index] for index in diagnostic_index]
    reference_q = np.asarray([row["last_reference_q"] for row in diagnostics])
    reference_v = np.asarray([row["last_reference_v"] for row in diagnostics])
    q = data.output.q[output_index]
    v = data.output.v[output_index]
    # The bag stores the already-negated LFC convention gain. For Panda
    # revolute coordinates LFC evaluates [anchor-current] before multiplying K.
    feedback_state = np.hstack((control.anchor_q - q, control.anchor_v - v))
    feedback = np.einsum("nij,nj->ni", control.gain, feedback_state)
    t0 = control.receive[0]
    return AlignedRun(
        data=data,
        time=control.receive - t0,
        q=q,
        v=v,
        reference_q=reference_q,
        reference_v=reference_v,
        final_reference_q=reference_q[-1],
        feedforward=control.feedforward,
        gain=control.gain,
        feedback=feedback,
        output=data.output.effort[output_index],
        measured_tau=data.hardware.effort[hardware_index],
        diagnostics=diagnostics,
        diagnostic_time=data.diagnostics.receive - t0,
    )


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(len(finite)),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _detrended(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return values - values[:1]
    x = np.arange(len(values), dtype=np.float64)
    result = np.empty_like(values)
    for joint in range(values.shape[1]):
        fit = np.polyfit(x, values[:, joint], 1)
        result[:, joint] = values[:, joint] - np.polyval(fit, x)
    return result


def _rms_by_column(values: np.ndarray) -> list[float]:
    return np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2, axis=0)).tolist()


def _max_abs_by_column(values: np.ndarray) -> list[float | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [None] * (array.shape[1] if array.ndim == 2 else 0)
    return np.max(np.abs(array), axis=0).tolist()


def _state_stream_summary(data: RunData) -> dict[str, Any]:
    """Compare the two state publications without claiming independent sensing."""
    indices = nearest_indices(data.hardware.receive, data.sensor.receive)
    receive_skew_ms = 1e3 * (
        data.sensor.receive - data.hardware.receive[indices]
    )
    q_delta = data.sensor.q - data.hardware.q[indices]
    v_delta = data.sensor.v - data.hardware.v[indices]
    effort_delta = data.sensor.effort - data.hardware.effort[indices]
    raw_header_period_ms = 1e3 * np.diff(data.hardware.stamp)
    return {
        "sensor_to_raw_receive_skew_ms": _stats(receive_skew_ms),
        "sensor_to_raw_abs_receive_skew_ms": _stats(np.abs(receive_skew_ms)),
        "sensor_minus_raw_q_rms_rad_by_joint": _rms_by_column(q_delta),
        "sensor_minus_raw_q_max_abs_rad_by_joint": _max_abs_by_column(q_delta),
        "sensor_minus_raw_velocity_rms_rad_s_by_joint": _rms_by_column(v_delta),
        "sensor_minus_raw_velocity_max_abs_rad_s_by_joint": _max_abs_by_column(
            v_delta
        ),
        "sensor_minus_raw_effort_rms_nm_by_joint": _rms_by_column(effort_delta),
        "sensor_minus_raw_effort_max_abs_nm_by_joint": _max_abs_by_column(
            effort_delta
        ),
        "raw_state_header_period_ms": _stats(raw_header_period_ms),
        "raw_state_nonpositive_header_period_count": int(
            np.sum(raw_header_period_ms <= 0.0)
        ),
        "raw_state_header_gap_over_2ms_count": int(
            np.sum(raw_header_period_ms > 2.0)
        ),
        "all_recorded_state_values_finite": bool(
            np.all(np.isfinite(data.sensor.q))
            and np.all(np.isfinite(data.sensor.v))
            and np.all(np.isfinite(data.hardware.q))
            and np.all(np.isfinite(data.hardware.v))
        ),
        "interpretation": (
            "/sensor and /franka/joint_states are two publications derived from "
            "the same Franka RobotState. Agreement detects ROS transport or "
            "mapping corruption but cannot prove absolute encoder calibration."
        ),
    }


def _agimus_robot_state_summary(data: RunData) -> dict[str, Any]:
    mode_names = {
        0: "OTHER",
        1: "IDLE",
        2: "MOVE",
        3: "GUIDING",
        4: "REFLEX",
        5: "USER_STOPPED",
        6: "AUTOMATIC_ERROR_RECOVERY",
    }
    if data.robot_state:
        desired_receive = np.asarray([row["receive"] for row in data.robot_state])
        desired_stamp = np.asarray([row["stamp"] for row in data.robot_state])
        desired_effort = np.asarray(
            [row["desired_effort"] for row in data.robot_state]
        )
        mode_counts: dict[str, int] = {}
        current_error_counts: dict[str, int] = {}
        last_motion_errors: set[str] = set()
        for row in data.robot_state:
            mode = mode_names.get(row["robot_mode"], f"UNKNOWN_{row['robot_mode']}")
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            for name in row["current_errors"]:
                current_error_counts[name] = current_error_counts.get(name, 0) + 1
            last_motion_errors.update(row["last_motion_errors"])
        state_details: dict[str, Any] = {
            "source": ROBOT_STATE_TOPIC,
            "robot_mode_counts": mode_counts,
            "current_error_active_message_counts": current_error_counts,
            "last_motion_errors_seen": sorted(last_motion_errors),
            "control_command_success_rate": _stats(
                np.asarray(
                    [
                        row["control_command_success_rate"]
                        for row in data.robot_state
                    ]
                )
            ),
            "collision_indicator_max": _stats(
                np.asarray(
                    [row["collision_indicator_max"] for row in data.robot_state]
                )
            ),
        }
    elif data.desired is not None:
        desired_receive = data.desired.receive
        desired_stamp = data.desired.stamp
        desired_effort = data.desired.effort
        state_details = {
            "source": DESIRED_STATE_TOPIC,
            "robot_mode_counts": None,
            "current_error_active_message_counts": None,
            "last_motion_errors_seen": None,
            "control_command_success_rate": None,
            "collision_indicator_max": None,
        }
    else:
        return {
            "available": False,
            "source": None,
            "note": (
                "The Agimus broadcaster recorded no desired torque or full FCI "
                "state, so post-limit commands, robot mode, and reflex/error bits "
                "cannot be diagnosed from this bag."
            ),
        }

    output_indices = nearest_indices(data.output.receive, desired_receive)
    output_receive = data.output.receive[output_indices]
    delta = desired_effort - data.output.effort[output_indices]
    state_details.update(
        {
            "available": True,
            "tau_J_d_count": int(len(desired_effort)),
            "tau_J_d_period_ms": _stats(1e3 * np.diff(desired_stamp)),
            "tau_J_d_to_pre_limiter_receive_skew_ms": _stats(
                1e3 * (desired_receive - output_receive)
            ),
            "tau_J_d_minus_pre_limiter_rms_nm_by_joint": _rms_by_column(delta),
            "tau_J_d_minus_pre_limiter_max_abs_nm_by_joint": _max_abs_by_column(
                delta
            ),
            "note": (
                "tau_J_d is the Agimus/Franka desired torque used as the rate-"
                "limiter state. It is compared with the nearest recorded "
                "pre-limiter LFC output."
            ),
        }
    )
    return state_details


def _recorded_position_error(row: dict[str, Any]) -> float | None:
    phase = row.get("phase_machine") or {}
    value = phase.get("ee_position_error_norm_m", row.get("last_position_error"))
    return float(value) if value is not None else None


def _recorded_orientation_error(row: dict[str, Any]) -> float | None:
    phase = row.get("phase_machine") or {}
    value = phase.get("ee_orientation_error_rad", row.get("last_orientation_error"))
    return float(value) if value is not None else None


def _phase(row: dict[str, Any]) -> str:
    phase = row.get("phase_machine") or {}
    return str(phase.get("phase", row.get("last_phase", "UNKNOWN")))


def summarize(run: AlignedRun, *, terminal_sec: float) -> dict[str, Any]:
    data = run.data
    terminal_start = max(0.0, run.time[-1] - terminal_sec)
    terminal_control = run.time >= terminal_start
    sensor_time = data.sensor.receive - data.control.receive[0]
    terminal_sensor = (sensor_time >= terminal_start) & (
        data.sensor.receive <= data.control.receive[-1]
    )
    terminal_q = data.sensor.q[terminal_sensor]
    terminal_v = data.sensor.v[terminal_sensor]
    terminal_q_detrended = _detrended(terminal_q)
    q_error = run.q - run.reference_q
    local_anchor_q_error = run.q - data.control.anchor_q
    local_anchor_v_error = run.v - data.control.anchor_v
    active_output = (
        (data.output.receive >= data.control.receive[0])
        & (data.output.receive <= data.control.receive[-1])
    )
    active_output_effort = data.output.effort[active_output]
    output_component_step = np.diff(active_output_effort, axis=0)
    active_output_period_ms = 1e3 * np.diff(data.output.stamp[active_output])
    one_cycle_step = (active_output_period_ms >= 0.5) & (
        active_output_period_ms <= 1.5
    )
    audited_component_step = output_component_step[one_cycle_step]
    position_error_mm = np.asarray(
        [
            np.nan if (value := _recorded_position_error(row)) is None else 1e3 * value
            for row in run.diagnostics
        ]
    )
    orientation_error_rad = np.asarray(
        [
            np.nan if (value := _recorded_orientation_error(row)) is None else value
            for row in run.diagnostics
        ]
    )
    warning_rows = [row for row in data.rosout if row["level"] >= 30]
    final = run.diagnostics[-1]
    final_phase = final.get("phase_machine") or {}
    task_goal_position = final.get("last_goal_position")
    phase_goal_position = final_phase.get("ee_goal_position_m")
    goal_position = phase_goal_position or task_goal_position
    agimus_state = _agimus_robot_state_summary(data)
    return {
        "schema": "sbmpc_run_report_v1",
        "bag": str(data.bag),
        "topic_counts": data.topic_counts,
        "active_control_duration_sec": float(run.time[-1]),
        "control_count": int(len(run.time)),
        "planner_mode": final.get("planner_mode"),
        "last_phase": _phase(final),
        "recorded_goal_position_m": goal_position,
        "recorded_phase_goal_position_m": phase_goal_position,
        "recorded_task_goal_position_m": task_goal_position,
        "initial_q_minus_reference_rad": (run.q[0] - run.reference_q[0]).tolist(),
        "final_q_minus_reference_rad": (run.q[-1] - run.final_reference_q).tolist(),
        "final_q_minus_local_anchor_rad": local_anchor_q_error[-1].tolist(),
        "final_max_abs_q_error_rad": float(np.max(np.abs(q_error[-1]))),
        "position_error_mm": _stats(position_error_mm),
        "terminal_position_error_mm": _stats(position_error_mm[terminal_control]),
        "orientation_error_rad": _stats(orientation_error_rad),
        "terminal_orientation_error_rad": _stats(
            orientation_error_rad[terminal_control]
        ),
        "terminal_detrended_q_rms_rad": np.sqrt(
            np.mean(terminal_q_detrended**2, axis=0)
        ).tolist(),
        "terminal_detrended_q_p2p_rad": np.ptp(terminal_q_detrended, axis=0).tolist(),
        "terminal_velocity_rms_rad_s": np.sqrt(
            np.mean(terminal_v**2, axis=0)
        ).tolist(),
        "gain_norm": _stats(np.linalg.norm(run.gain.reshape(len(run.gain), -1), axis=1)),
        "feedback_correction_norm_nm": _stats(np.linalg.norm(run.feedback, axis=1)),
        "local_anchor_tracking": {
            "q_error_norm_rad": _stats(np.linalg.norm(local_anchor_q_error, axis=1)),
            "velocity_error_norm_rad_s": _stats(
                np.linalg.norm(local_anchor_v_error, axis=1)
            ),
            "terminal_q_error_rms_rad_by_joint": _rms_by_column(
                local_anchor_q_error[terminal_control]
            ),
            "terminal_velocity_error_rms_rad_s_by_joint": _rms_by_column(
                local_anchor_v_error[terminal_control]
            ),
            "note": (
                "This is the state error actually seen by the deployed local "
                "Riccati gain. It is distinct from q minus the task reference."
            ),
        },
        "feedforward_step_norm_nm": _stats(
            np.linalg.norm(np.diff(run.feedforward, axis=0), axis=1)
        ),
        "planner_aligned_output_step_norm_nm": _stats(
            np.linalg.norm(np.diff(run.output, axis=0), axis=1)
        ),
        "torque_rate_audit": {
            "source": "/output_joint_effort at the hardware-controller rate",
            "semantics": "gravity-free request before the Agimus hardware limiter",
            "component_step_audit_threshold_nm": AGIMUS_COMPONENT_STEP_AUDIT_NM,
            "component_step_abs_max_nm_by_joint": _max_abs_by_column(
                audited_component_step
            ),
            "component_steps_over_threshold_by_joint": np.sum(
                np.abs(audited_component_step) > AGIMUS_COMPONENT_STEP_AUDIT_NM,
                axis=0,
            ).tolist(),
            "component_steps_over_threshold_total": int(
                np.sum(
                    np.abs(audited_component_step)
                    > AGIMUS_COMPONENT_STEP_AUDIT_NM
                )
            ),
            "recorded_output_period_ms": _stats(active_output_period_ms),
            "audited_0p5_to_1p5ms_step_count": int(np.sum(one_cycle_step)),
            "recorded_step_gap_over_1p5ms_count": int(
                np.sum(active_output_period_ms > 1.5)
            ),
            "note": (
                "Only recorded intervals from 0.5 to 1.5 ms are compared with "
                "the Agimus one-cycle limit. A zero count rules out an oversized "
                "observed pre-limiter step, but is not a substitute for the "
                "missing post-limit tau_J_d."
            ),
        },
        "control_period_ms": _stats(1e3 * np.diff(data.control.stamp)),
        "causal_output_after_control_ms": _stats(
            1e3
            * (
                data.output.receive[
                    causal_indices(data.output.receive, data.control.receive)
                ]
                - data.control.receive
            )
        ),
        "solve_anchor_age_ms": _stats(
            1e3 * (data.control.stamp - data.control.anchor_stamp)
        ),
        "planning_time_ms": _stats(
            np.asarray(
                [row.get("last_planning_time_ms", np.nan) for row in run.diagnostics]
            )
        ),
        "planner_step_wall_time_ms": _stats(
            np.asarray(
                [
                    row.get("last_planner_step_wall_time_ms", np.nan)
                    for row in run.diagnostics
                ]
            )
        ),
        "final_counters": {
            key: final.get(key)
            for key in (
                "accepted_planner_output_count",
                "published_control_count",
                "rejected_planner_output_count",
                "deadline_miss_count",
                "rejected_sensor_count",
                "last_error",
            )
        },
        "gripper": final.get("gripper"),
        "rosout_warning_or_error_count": len(warning_rows),
        "rosout_warning_or_error": warning_rows,
        "state_stream_consistency": _state_stream_summary(data),
        "agimus_robot_state": agimus_state,
        "observability": {
            "lfc_output_is_pre_agimus_rate_limit": True,
            "post_rate_limit_tau_J_d_available": bool(agimus_state["available"]),
            "fci_robot_state_available": bool(data.robot_state),
            "missing_populated_topics": [
                topic
                for topic in sorted(OBSERVABILITY_TOPICS)
                if data.topic_counts.get(topic, 0) == 0
            ],
            "note": (
                "Measured /franka/joint_states effort is total tau_J and must not "
                "be interpreted as the gravity-free commanded effort."
            ),
        },
    }


def _save(fig: plt.Figure, output: Path, name: str) -> str:
    path = output / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return name


def _plot_task(run: AlignedRun, output: Path) -> str:
    q_error = run.q - run.reference_q
    position_error = np.asarray(
        [
            np.nan if (value := _recorded_position_error(row)) is None else 1e3 * value
            for row in run.diagnostics
        ]
    )
    orientation_error = np.asarray(
        [
            np.nan if (value := _recorded_orientation_error(row)) is None else value
            for row in run.diagnostics
        ]
    )
    has_orientation = bool(np.any(np.isfinite(orientation_error)))
    phases = [_phase(row) for row in run.diagnostics]
    phase_names = list(dict.fromkeys(phases))
    phase_index = np.asarray([phase_names.index(value) for value in phases])
    row_count = 5 if has_orientation else 4
    fig, axes = plt.subplots(
        row_count,
        1,
        figsize=(13, 13 if has_orientation else 11),
        sharex=True,
    )
    axes[0].plot(run.time, position_error, color="black")
    axes[0].set_ylabel("recorded EE error [mm]")
    row = 1
    if has_orientation:
        axes[row].plot(run.time, orientation_error, color="tab:purple")
        axes[row].set_ylabel("EE orientation error [rad]")
        row += 1
    axes[row].plot(run.time, np.linalg.norm(q_error, axis=1), label="||q-q_ref||")
    axes[row].plot(run.time, np.max(np.abs(q_error), axis=1), label="max |q-q_ref|")
    axes[row].set_ylabel("joint error [rad]")
    axes[row].legend()
    row += 1
    axes[row].plot(run.time, np.linalg.norm(run.v, axis=1))
    axes[row].set_ylabel("||dq|| [rad/s]")
    row += 1
    axes[row].step(run.time, phase_index, where="post")
    axes[row].set_yticks(range(len(phase_names)), phase_names)
    axes[row].set_ylabel("phase")
    axes[row].set_xlabel("active-control time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("Task tracking and controller phase")
    return _save(fig, output, "01_task_tracking.png")


def _plot_joints(run: AlignedRun, output: Path) -> str:
    fig, axes = plt.subplots(7, 2, figsize=(15, 17), sharex=True)
    for joint in range(7):
        axes[joint, 0].plot(run.time, run.q[:, joint], label="measured")
        axes[joint, 0].plot(run.time, run.reference_q[:, joint], "--", label="reference")
        axes[joint, 0].set_ylabel(f"J{joint + 1} [rad]")
        axes[joint, 1].plot(run.time, run.q[:, joint] - run.reference_q[:, joint])
        axes[joint, 1].axhline(0.0, color="black", linewidth=0.5)
        axes[joint, 1].set_ylabel(f"J{joint + 1} error [rad]")
        for axis in axes[joint]:
            axis.grid(alpha=0.25)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("active-control time [s]")
    axes[-1, 1].set_xlabel("active-control time [s]")
    fig.suptitle("Measured joint state versus recorded planner reference")
    return _save(fig, output, "02_joint_tracking.png")


def _plot_terminal(run: AlignedRun, output: Path, terminal_sec: float) -> str:
    sensor_time = run.data.sensor.receive - run.data.control.receive[0]
    mask = (sensor_time >= max(0.0, run.time[-1] - terminal_sec)) & (
        run.data.sensor.receive <= run.data.control.receive[-1]
    )
    time = sensor_time[mask]
    q = run.data.sensor.q[mask]
    v = run.data.sensor.v[mask]
    detrended_q = _detrended(q)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(time, detrended_q)
    axes[0].set_ylabel("detrended q [rad]")
    axes[1].plot(time, v)
    axes[1].set_ylabel("dq [rad/s]")
    axes[2].bar(
        np.arange(7),
        run.q[-1] - run.final_reference_q,
    )
    axes[2].set_xticks(np.arange(7), [f"J{i}" for i in range(1, 8)])
    axes[2].set_ylabel("final q - final ref [rad]")
    axes[2].set_xlabel("joint")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(f"Terminal stability and reference bias (last {terminal_sec:g} s)")
    return _save(fig, output, "03_terminal_stability.png")


def _plot_controller(run: AlignedRun, output: Path) -> str:
    gain_norm = np.linalg.norm(run.gain.reshape(len(run.gain), -1), axis=1)
    feedback_norm = np.linalg.norm(run.feedback, axis=1)
    reference_q_error = np.linalg.norm(run.q - run.reference_q, axis=1)
    anchor_q_error = np.linalg.norm(run.q - run.data.control.anchor_q, axis=1)
    delta_ff = np.r_[np.nan, np.linalg.norm(np.diff(run.feedforward, axis=0), axis=1)]
    delta_output = np.r_[np.nan, np.linalg.norm(np.diff(run.output, axis=0), axis=1)]
    fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
    axes[0].plot(run.time, np.linalg.norm(run.feedforward, axis=1), label="feedforward")
    axes[0].plot(run.time, np.linalg.norm(run.output, axis=1), label="LFC output")
    axes[0].set_ylabel("torque norm [Nm]")
    axes[0].legend()
    axes[1].semilogy(
        run.time,
        np.maximum(anchor_q_error, 1e-9),
        label="local gain anchor error",
    )
    axes[1].semilogy(
        run.time,
        np.maximum(reference_q_error, 1e-9),
        label="task-reference error",
    )
    axes[1].set_ylabel("joint error norm [rad]")
    axes[1].legend()
    axes[2].plot(run.time, feedback_norm)
    axes[2].set_ylabel("||K(anchor-current)|| [Nm]")
    axes[3].plot(run.time, gain_norm)
    axes[3].set_ylabel("Frobenius ||K||")
    axes[4].plot(run.time, delta_ff, label="feedforward step")
    axes[4].plot(run.time, delta_output, label="planner-aligned output step")
    axes[4].set_ylabel("step norm [Nm]")
    axes[4].set_xlabel("active-control time [s]")
    axes[4].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("Controller command and local feedback activity")
    return _save(fig, output, "04_controller.png")


def _plot_runtime(run: AlignedRun, output: Path) -> str:
    planning = np.asarray(
        [row.get("last_planning_time_ms", np.nan) for row in run.diagnostics]
    )
    step = np.asarray(
        [row.get("last_planner_step_wall_time_ms", np.nan) for row in run.diagnostics]
    )
    period = 1e3 * np.diff(run.data.control.stamp)
    anchor_age = 1e3 * (run.data.control.stamp - run.data.control.anchor_stamp)
    final = run.diagnostics[-1]
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=False)
    axes[0].plot(run.time, planning, label="planning")
    axes[0].plot(run.time, step, label="whole step")
    axes[0].axhline(40.0, color="tab:red", linestyle="--", label="40 ms")
    axes[0].set_ylabel("time [ms]")
    axes[0].legend()
    axes[1].plot(run.time[1:], period)
    axes[1].axhline(40.0, color="black", linestyle="--", linewidth=0.7)
    axes[1].set_ylabel("control period [ms]")
    axes[2].plot(run.time, anchor_age)
    axes[2].set_ylabel("solve-anchor age [ms]")
    counters = (
        "accepted_planner_output_count",
        "rejected_planner_output_count",
        "deadline_miss_count",
    )
    for key in counters:
        axes[3].plot(run.time, [row.get(key, 0) for row in run.diagnostics], label=key)
    axes[3].set_ylabel("cumulative count")
    axes[3].set_xlabel("active-control time [s]")
    axes[3].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(f"Runtime health (last error: {final.get('last_error') or 'none'})")
    return _save(fig, output, "05_runtime.png")


def _plot_torque(run: AlignedRun, output: Path, terminal_sec: float) -> str:
    # Real LFC output is gravity-free and before the Agimus hardware limiter.
    # The reconstruction uses the first output after each control packet and
    # is exact after the startup PD-to-LF blend, subject to ROS callback order.
    inferred_gravity = run.feedforward + run.feedback - run.output
    final_error = run.q[-1] - run.final_reference_q
    focus = int(np.argmax(np.abs(final_error)))
    terminal = run.time >= max(0.0, run.time[-1] - terminal_sec)
    active_output = (
        (run.data.output.receive >= run.data.control.receive[0])
        & (run.data.output.receive <= run.data.control.receive[-1])
    )
    active_effort = run.data.output.effort[active_output]
    active_period_ms = 1e3 * np.diff(run.data.output.stamp[active_output])
    one_cycle = (active_period_ms >= 0.5) & (active_period_ms <= 1.5)
    component_step_max = np.max(
        np.abs(np.diff(active_effort, axis=0)[one_cycle]), axis=0
    )
    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=False)
    axes[0].plot(run.time, run.feedforward[:, focus], label="feedforward")
    axes[0].plot(
        run.time,
        inferred_gravity[:, focus],
        label="inferred removed gravity (causal sample)",
    )
    axes[0].plot(run.time, run.output[:, focus], label="pre-limiter output")
    axes[0].plot(run.time, run.measured_tau[:, focus], label="measured total tau_J")
    axes[0].set_ylabel(f"J{focus + 1} torque [Nm]")
    axes[0].legend(ncol=2)
    width = 0.20
    joints = np.arange(7)
    for offset, values, label in (
        (-1.5 * width, run.feedforward, "feedforward"),
        (-0.5 * width, inferred_gravity, "inferred removed gravity"),
        (+0.5 * width, run.output, "pre-limiter output"),
        (+1.5 * width, run.measured_tau, "measured total tau_J"),
    ):
        axes[1].bar(
            joints + offset,
            np.mean(values[terminal], axis=0),
            width,
            label=label,
        )
    axes[1].set_xticks(joints, [f"J{i}" for i in range(1, 8)])
    axes[1].set_ylabel("terminal mean [Nm]")
    axes[1].legend(ncol=2)
    axes[2].bar(joints, component_step_max)
    axes[2].axhline(
        AGIMUS_COMPONENT_STEP_AUDIT_NM,
        color="tab:red",
        linestyle="--",
        label="Agimus one-cycle component limit",
    )
    axes[2].set_xticks(joints, [f"J{i}" for i in range(1, 8)])
    axes[2].set_ylabel("max 1 kHz output step [Nm]")
    axes[2].legend()
    axes[3].axis("off")
    axes[3].text(
        0.02,
        0.85,
        "Interpretation boundary:\n"
        "• /output_joint_effort is the gravity-free LFC request before Agimus rate limiting.\n"
        "• inferred gravity uses the first output after each control packet; "
        "the startup blend is approximate.\n"
        "• measured tau_J is total link-side torque; it is not the commanded residual.\n"
        "• post-limit tau_J_d and FCI error state are unavailable when the "
        "robot-state broadcaster is inactive.",
        va="top",
        fontsize=11,
    )
    for axis in axes[:3]:
        axis.grid(alpha=0.25)
    fig.suptitle("Real torque path with command/measurement semantics")
    return _save(fig, output, "06_torque_path.png")


def _plot_phase(run: AlignedRun, output: Path) -> str | None:
    phase_rows = [row.get("phase_machine") for row in run.diagnostics]
    if not any(phase_rows):
        return None
    phase_rows = [row or {} for row in phase_rows]
    phases = [str(row.get("phase", "UNKNOWN")) for row in phase_rows]
    phase_names = list(dict.fromkeys(phases))
    phase_index = [phase_names.index(value) for value in phases]
    plan_time = [row.get("plan_time_sec", np.nan) for row in phase_rows]
    boundary = [bool(row.get("at_boundary", False)) for row in phase_rows]
    blocked = [bool(row.get("transition_blocked", False)) for row in phase_rows]
    q_ok = [bool(row.get("q_ok", False)) for row in phase_rows]
    velocity_ok = [bool(row.get("velocity_ok", False)) for row in phase_rows]
    gripper = [row.get("gripper") or {} for row in run.diagnostics]
    goal_count = [row.get("goal_count", 0) or 0 for row in gripper]
    stages = [str(row.get("stage", "none")) for row in gripper]
    stage_names = list(dict.fromkeys(stages))
    stage_index = [stage_names.index(value) for value in stages]
    fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True)
    axes[0].step(run.time, phase_index, where="post")
    axes[0].set_yticks(range(len(phase_names)), phase_names)
    axes[0].set_ylabel("phase")
    axes[1].plot(run.time, plan_time, label="plan time")
    axes[1].plot(run.time, run.time, "--", alpha=0.5, label="unpaused time")
    axes[1].set_ylabel("time [s]")
    axes[1].legend()
    axes[2].step(run.time, q_ok, where="post", label="q_ok")
    axes[2].step(run.time, velocity_ok, where="post", label="velocity_ok")
    axes[2].step(run.time, blocked, where="post", label="blocked")
    axes[2].step(run.time, boundary, where="post", label="at boundary")
    axes[2].set_ylabel("gate flags")
    axes[2].legend(ncol=4)
    axes[3].step(run.time, stage_index, where="post")
    axes[3].set_yticks(range(len(stage_names)), stage_names)
    axes[3].set_ylabel("gripper stage")
    axes[4].step(run.time, goal_count, where="post")
    axes[4].set_ylabel("gripper goals")
    axes[4].set_xlabel("active-control time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("State-machine gate and gripper timeline")
    return _save(fig, output, "07_phase_and_gripper.png")


def _plot_sensor_integrity(run: AlignedRun, output: Path) -> str:
    sensor = run.data.sensor
    hardware = run.data.hardware
    indices = nearest_indices(hardware.receive, sensor.receive)
    q_delta = sensor.q - hardware.q[indices]
    v_delta = sensor.v - hardware.v[indices]
    hardware_time = hardware.receive - run.data.control.receive[0]
    period = 1e3 * np.diff(hardware.stamp)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
    axes[0].plot(sensor.receive - run.data.control.receive[0], q_delta)
    axes[0].set_ylabel("sensor - raw q [rad]")
    axes[1].plot(sensor.receive - run.data.control.receive[0], v_delta)
    axes[1].set_ylabel("filtered - raw dq [rad/s]")
    axes[2].plot(hardware_time[1:], period)
    axes[2].set_ylabel("raw state period [ms]")
    axes[2].set_xlabel("active-control time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(
        "State-stream consistency (agreement cannot prove absolute encoder calibration)"
    )
    return _save(fig, output, "08_sensor_integrity.png")


def _plot_agimus_robot_state(run: AlignedRun, output: Path) -> str | None:
    data = run.data
    if data.robot_state:
        receive = np.asarray([row["receive"] for row in data.robot_state])
        desired = np.asarray([row["desired_effort"] for row in data.robot_state])
        modes = np.asarray([row["robot_mode"] for row in data.robot_state])
        success = np.asarray(
            [row["control_command_success_rate"] for row in data.robot_state]
        )
        active_errors = sorted(
            {
                name
                for row in data.robot_state
                for name in row["current_errors"]
            }
        )
        last_errors = sorted(
            {
                name
                for row in data.robot_state
                for name in row["last_motion_errors"]
            }
        )
    elif data.desired is not None:
        receive = data.desired.receive
        desired = data.desired.effort
        modes = None
        success = None
        active_errors = []
        last_errors = []
    else:
        return None

    output_indices = nearest_indices(data.output.receive, receive)
    pre_limiter = data.output.effort[output_indices]
    delta = desired - pre_limiter
    focus = int(np.argmax(np.sqrt(np.mean(delta**2, axis=0))))
    time = receive - data.control.receive[0]
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    axes[0].plot(time, pre_limiter[:, focus], label="LFC pre-limiter")
    axes[0].plot(time, desired[:, focus], label="Agimus tau_J_d")
    axes[0].set_ylabel(f"J{focus + 1} [Nm]")
    axes[0].legend()
    axes[1].plot(time, delta)
    axes[1].set_ylabel("tau_J_d - LFC [Nm]")
    if modes is None:
        axes[2].text(0.02, 0.5, "Full Agimus robot_state was not recorded.")
        axes[2].axis("off")
        axes[3].axis("off")
    else:
        axes[2].step(time, modes, where="post")
        axes[2].set_ylabel("robot mode code")
        axes[3].plot(time, success)
        axes[3].set_ylabel("command success rate")
        axes[3].set_ylim(-0.02, 1.02)
        error_text = (
            f"active errors: {active_errors or 'none'}; "
            f"last-motion errors: {last_errors or 'none'}"
        )
        axes[3].set_title(error_text, fontsize=9)
    axes[-1].set_xlabel("active-control time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("Agimus post-limiter torque and FCI state")
    return _save(fig, output, "09_agimus_robot_state.png")


def _write_steps_csv(run: AlignedRun, output: Path) -> None:
    fields = (
        "time_sec",
        "phase",
        "ee_error_mm",
        "q_error_norm_rad",
        "q_error_max_rad",
        "velocity_norm_rad_s",
        "feedforward_norm_nm",
        "output_norm_nm",
        "feedback_norm_nm",
        "gain_norm",
        "planning_time_ms",
    )
    with (output / "controller_steps.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(run.diagnostics):
            q_error = run.q[index] - run.reference_q[index]
            ee_error = _recorded_position_error(row)
            writer.writerow(
                {
                    "time_sec": run.time[index],
                    "phase": _phase(row),
                    "ee_error_mm": None if ee_error is None else 1e3 * ee_error,
                    "q_error_norm_rad": np.linalg.norm(q_error),
                    "q_error_max_rad": np.max(np.abs(q_error)),
                    "velocity_norm_rad_s": np.linalg.norm(run.v[index]),
                    "feedforward_norm_nm": np.linalg.norm(run.feedforward[index]),
                    "output_norm_nm": np.linalg.norm(run.output[index]),
                    "feedback_norm_nm": np.linalg.norm(run.feedback[index]),
                    "gain_norm": np.linalg.norm(run.gain[index]),
                    "planning_time_ms": row.get("last_planning_time_ms"),
                }
            )


def _write_html(output: Path, summary: dict[str, Any], images: list[str]) -> None:
    image_html = "\n".join(
        f'<section><h2>{html.escape(Path(image).stem.replace("_", " "))}</h2>'
        f'<a href="{html.escape(image)}"><img src="{html.escape(image)}"></a></section>'
        for image in images
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SB-MPC run report</title>
<style>body{{font-family:sans-serif;max-width:1500px;margin:auto;padding:1rem}}
img{{max-width:100%;border:1px solid #ddd}}
pre{{white-space:pre-wrap;background:#f5f5f5;padding:1rem}}</style>
</head><body><h1>SB-MPC offline run report</h1>
<p>Source: {html.escape(str(summary['bag']))}</p>
<p>This report was generated after the run and imposed no controller-loop load.</p>
<p><a href="summary.json">summary.json</a> ·
<a href="controller_steps.csv">controller_steps.csv</a></p>
{image_html}<h2>Machine-readable summary</h2>
<pre>{html.escape(json.dumps(summary, indent=2))}</pre>
</body></html>"""
    (output / "index.html").write_text(document)


def generate_report(
    bag_or_run: Path,
    output: Path,
    *,
    terminal_sec: float = 5.0,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    run = align_run(read_bag(bag_or_run))
    summary = summarize(run, terminal_sec=terminal_sec)
    images = [
        _plot_task(run, output),
        _plot_joints(run, output),
        _plot_terminal(run, output, terminal_sec),
        _plot_controller(run, output),
        _plot_runtime(run, output),
        _plot_torque(run, output, terminal_sec),
    ]
    phase_image = _plot_phase(run, output)
    if phase_image is not None:
        images.append(phase_image)
    images.append(_plot_sensor_integrity(run, output))
    agimus_image = _plot_agimus_robot_state(run, output)
    if agimus_image is not None:
        images.append(agimus_image)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_steps_csv(run, output)
    _write_html(output, summary, images)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run",
        type=Path,
        help="run directory containing rosbag/, or the rosbag2 directory itself",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: <run>/diagnostic_report (or sibling of a direct bag path)",
    )
    parser.add_argument("--terminal-sec", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.terminal_sec <= 0.0:
        parser.error("--terminal-sec must be positive")
    bag = _resolve_bag(args.run)
    if args.output_dir is not None:
        output = args.output_dir.resolve()
    elif bag.parent.name == "rosbag":
        output = bag.parent / "diagnostic_report"
    elif bag.name == "rosbag":
        output = bag.parent / "diagnostic_report"
    else:
        output = bag.parent / f"{bag.name}_diagnostic_report"
    summary = generate_report(bag, output, terminal_sec=args.terminal_sec)
    print(f"SB-MPC report: {output / 'index.html'}")
    print(
        "terminal: "
        f"EE median={summary['terminal_position_error_mm']['median']} mm, "
        f"max |q-ref|={summary['final_max_abs_q_error_rad']} rad, "
        f"deadline misses={summary['final_counters']['deadline_miss_count']}"
    )


if __name__ == "__main__":
    main()
