"""Offline, repeatable diagnostics for a recorded SB-MPC real-robot run.

The reporter reads an MCAP after the experiment has stopped.  It never runs in
the planner or low-level controller processes, so generating the figures has no
effect on controller real-time performance.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sbmpc_bringup.constants import FER_ARM_JOINT_NAMES  # noqa: E402
from sbmpc_bringup.run_topics import (  # noqa: E402
    CONTROLLER_ACTIVITY_TOPIC,
    CONTROL_TOPIC,
    DESIRED_STATE_TOPIC,
    GRIPPER_FEEDBACK_TOPIC,
    GRIPPER_JOINT_TOPIC,
    GRIPPER_STATUS_TOPIC,
    HARDWARE_JOINT_TOPIC,
    MERGED_JOINT_TOPIC,
    OUTPUT_TOPIC,
    PLANNER_DIAGNOSTICS_TOPIC,
    RUN_TOPICS,
    ROBOT_STATE_TOPIC,
    ROS_CLOCK_TOPIC,
    ROSOUT_TOPIC,
    ROS_DIAGNOSTICS_TOPIC,
    SENSOR_TOPIC,
    SIM_ACTUATOR_STATE_TOPIC,
    SIM_GRIPPER_FEEDBACK_TOPIC,
    SIM_GRIPPER_STATUS_TOPIC,
    SIM_OBJECT_POSE_TOPIC,
)


TOPICS = set(RUN_TOPICS)

OBSERVABILITY_TOPICS = {
    ROBOT_STATE_TOPIC,
    ROS_DIAGNOSTICS_TOPIC,
    CONTROLLER_ACTIVITY_TOPIC,
    GRIPPER_JOINT_TOPIC,
    GRIPPER_STATUS_TOPIC,
    GRIPPER_FEEDBACK_TOPIC,
}

DECODED_TOPICS = TOPICS | {DESIRED_STATE_TOPIC}

# Agimus limits a torque-command component to just under 1 Nm per 1 kHz
# hardware cycle.  A pre-limiter trace below this threshold is a useful
# necessary-condition check; without tau_J_d it cannot prove what was applied.
AGIMUS_COMPONENT_STEP_AUDIT_NM = 0.999999

_DIAGNOSTIC_RANGE = re.compile(
    r"Avg:\s*(?P<average>[-+0-9.eE]+)\s*"
    r"\[\s*(?P<minimum>[-+0-9.eE]+)\s*-\s*"
    r"(?P<maximum>[-+0-9.eE]+)\s*\]\s*"
    r"(?P<unit>[^,\s]+)\s*,\s*StdDev:\s*"
    r"(?P<stddev>[-+0-9.eE]+)"
    r"(?:\s*->\s*Desired\s*:\s*(?P<desired>[-+0-9.eE]+)\s*"
    r"(?P<desired_unit>\S+))?"
)


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
    ros_diagnostics: list[dict[str, Any]]
    controller_activity: list[dict[str, Any]]
    gripper_joint_states: list[dict[str, Any]]
    gripper_feedback: list[dict[str, Any]]
    gripper_status: list[dict[str, Any]]
    topic_counts: dict[str, int]
    hardware_source: str = HARDWARE_JOINT_TOPIC
    sim_object_pose: list[dict[str, Any]] = field(default_factory=list)
    sim_actuator_states: list[dict[str, Any]] = field(default_factory=list)
    sim_clock: list[dict[str, float]] = field(default_factory=list)
    backend: str = "real"
    backend_source: str = "default"
    backend_consistency_warning: str | None = None
    recorder_realtime: dict[str, Any] = field(default_factory=dict)


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


@dataclass(frozen=True)
class LimiterReconstruction:
    state_stamp: np.ndarray
    observed: np.ndarray
    predicted: np.ndarray
    prior_output: np.ndarray
    same_cycle_output: np.ndarray
    same_cycle_q_error: np.ndarray
    same_cycle_stamp_skew_ms: np.ndarray
    interval_output_count: np.ndarray
    interval_max_gap_ms: np.ndarray
    valid_interval: np.ndarray


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


def _read_manifest(bag: Path) -> dict[str, Any]:
    candidates = (bag / "manifest.json", bag.parent / "manifest.json")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            manifest = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(manifest, dict):
            return manifest
    return {}


def _manifest_backend(bag: Path) -> str | None:
    """Return the recorder-declared backend when this is a canonical run."""
    backend = _read_manifest(bag).get("backend")
    if backend in {"real", "mujoco"}:
        return str(backend)
    return None


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


def _joint_field(joint: Any, field: str) -> list[float]:
    values = getattr(joint, field)
    indices = _arm_indices(list(joint.name), len(values))
    return np.asarray(values, dtype=np.float64)[indices].tolist()


def _xyz(vector: Any) -> list[float]:
    return [float(vector.x), float(vector.y), float(vector.z)]


def _pose(stamped: Any) -> list[float]:
    pose = stamped.pose
    return [
        *_xyz(pose.position),
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    ]


def _spatial(linear: Any, angular: Any) -> list[float]:
    return [*_xyz(linear), *_xyz(angular)]


def _twist(stamped: Any) -> list[float]:
    return _spatial(stamped.twist.linear, stamped.twist.angular)


def _accel(stamped: Any) -> list[float]:
    return _spatial(stamped.accel.linear, stamped.accel.angular)


def _wrench(stamped: Any) -> list[float]:
    return _spatial(stamped.wrench.force, stamped.wrench.torque)


def _inertia(stamped: Any) -> list[float]:
    inertia = stamped.inertia
    return [
        float(inertia.m),
        *_xyz(inertia.com),
        float(inertia.ixx),
        float(inertia.ixy),
        float(inertia.ixz),
        float(inertia.iyy),
        float(inertia.iyz),
        float(inertia.izz),
    ]


def _robot_state_row(message: Any, receive_sec: float) -> dict[str, Any]:
    measured = message.measured_joint_state
    desired = message.desired_joint_state
    motor = message.measured_joint_motor_state
    external = message.tau_ext_hat_filtered
    collision = message.collision_indicators
    cartesian_collision = [
        *_xyz(collision.is_cartesian_linear_collision),
        *_xyz(collision.is_cartesian_angular_collision),
    ]
    cartesian_contact = [
        *_xyz(collision.is_cartesian_linear_contact),
        *_xyz(collision.is_cartesian_angular_contact),
    ]
    joint_collision = np.asarray(collision.is_joint_collision, dtype=np.float64).tolist()
    joint_contact = np.asarray(collision.is_joint_contact, dtype=np.float64).tolist()
    collision_values = np.asarray(
        [*cartesian_collision, *joint_collision], dtype=np.float64
    )
    contact_values = np.asarray([*cartesian_contact, *joint_contact], dtype=np.float64)
    elbow = message.elbow

    return {
        "receive": receive_sec,
        "stamp": _stamp_sec(message.header.stamp),
        "frame_id": str(message.header.frame_id),
        "robot_time": float(message.time),
        "measured_position": _joint_field(measured, "position"),
        "measured_velocity": _joint_field(measured, "velocity"),
        "measured_effort": _joint_field(measured, "effort"),
        "desired_position": _joint_field(desired, "position"),
        "desired_velocity": _joint_field(desired, "velocity"),
        "desired_effort": _joint_field(desired, "effort"),
        "motor_position": _joint_field(motor, "position"),
        "motor_velocity": _joint_field(motor, "velocity"),
        "desired_acceleration": np.asarray(message.ddq_d, dtype=np.float64).tolist(),
        "measured_torque_derivative": np.asarray(
            message.dtau_j, dtype=np.float64
        ).tolist(),
        "external_effort": _joint_field(external, "effort"),
        "elbow_position": np.asarray(elbow.position, dtype=np.float64).tolist(),
        "elbow_desired_position": np.asarray(
            elbow.desired_position, dtype=np.float64
        ).tolist(),
        "elbow_commanded_position": np.asarray(
            elbow.commanded_position, dtype=np.float64
        ).tolist(),
        "elbow_commanded_velocity": np.asarray(
            elbow.commanded_velocity, dtype=np.float64
        ).tolist(),
        "elbow_commanded_acceleration": np.asarray(
            elbow.commanded_acceleration, dtype=np.float64
        ).tolist(),
        "external_wrench_stiffness": _wrench(message.k_f_ext_hat_k),
        "external_wrench_base": _wrench(message.o_f_ext_hat_k),
        "inertia_ee": _inertia(message.inertia_ee),
        "inertia_load": _inertia(message.inertia_load),
        "inertia_total": _inertia(message.inertia_total),
        "measured_ee_pose": _pose(message.o_t_ee),
        "desired_ee_pose": _pose(message.o_t_ee_d),
        "commanded_ee_pose": _pose(message.o_t_ee_c),
        "flange_to_ee_pose": _pose(message.f_t_ee),
        "ee_to_stiffness_pose": _pose(message.ee_t_k),
        "desired_ee_twist": _twist(message.o_dp_ee_d),
        "commanded_ee_twist": _twist(message.o_dp_ee_c),
        "commanded_ee_acceleration": _accel(message.o_ddp_ee_c),
        "cartesian_collision": cartesian_collision,
        "cartesian_contact": cartesian_contact,
        "joint_collision": joint_collision,
        "joint_contact": joint_contact,
        "control_command_success_rate": float(message.control_command_success_rate),
        "robot_mode": int(message.robot_mode),
        "current_errors": _active_boolean_fields(message.current_errors),
        "last_motion_errors": _active_boolean_fields(message.last_motion_errors),
        "collision_indicator_max": float(np.max(np.abs(collision_values))),
        "contact_indicator_max": float(np.max(np.abs(contact_values))),
    }


def _byte_value(value: Any) -> int:
    if isinstance(value, bytes):
        return int(value[0]) if value else 0
    if isinstance(value, str):
        return ord(value) if value else 0
    return int(value)


def _diagnostic_statistics(value: str) -> dict[str, float | str | None] | None:
    match = _DIAGNOSTIC_RANGE.fullmatch(value.strip())
    if match is None:
        return None
    return {
        "average": float(match.group("average")),
        "minimum": float(match.group("minimum")),
        "maximum": float(match.group("maximum")),
        "stddev": float(match.group("stddev")),
        "unit": match.group("unit"),
        "desired": (
            None
            if match.group("desired") is None
            else float(match.group("desired"))
        ),
        "desired_unit": match.group("desired_unit"),
    }


def _ros_diagnostic_rows(message: Any, receive_sec: float) -> list[dict[str, Any]]:
    stamp = _stamp_sec(message.header.stamp)
    return [
        {
            "receive": receive_sec,
            "stamp": stamp,
            "level": _byte_value(status.level),
            "name": str(status.name),
            "message": str(status.message).strip(),
            "hardware_id": str(status.hardware_id),
            "values": {str(value.key): str(value.value) for value in status.values},
        }
        for status in message.status
    ]


def _controller_activity_rows(
    message: Any, receive_sec: float
) -> list[dict[str, Any]]:
    stamp = _stamp_sec(message.header.stamp)
    rows: list[dict[str, Any]] = []
    for kind, entries in (
        ("controller", message.controllers),
        ("hardware", message.hardware_components),
    ):
        for entry in entries:
            rows.append(
                {
                    "receive": receive_sec,
                    "stamp": stamp,
                    "kind": kind,
                    "name": str(entry.name),
                    "state_id": int(entry.state.id),
                    "state_label": str(entry.state.label),
                }
            )
    return rows


def _gripper_joint_row(
    message: Any,
    receive_sec: float,
) -> dict[str, Any] | None:
    indices = [
        index for index, name in enumerate(message.name) if "finger" in name.lower()
    ]
    if not indices:
        return None

    def selected(values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if len(array) < len(message.name):
            return np.zeros(len(indices), dtype=np.float64)
        return array[indices]

    names = [str(message.name[index]) for index in indices]
    position = selected(message.position)
    velocity = selected(message.velocity)
    effort = selected(message.effort)
    # MuJoCo exposes one actuated finger and couples the second with an equality
    # constraint. The real Agimus state normally exposes both fingers.
    width = 2.0 * position[0] if len(position) == 1 else np.sum(position)
    return {
        "receive": receive_sec,
        "stamp": _stamp_sec(message.header.stamp),
        "names": names,
        "position": position.tolist(),
        "velocity": velocity.tolist(),
        "effort": effort.tolist(),
        "width": float(width),
    }


def _generic_joint_row(message: Any, receive_sec: float) -> dict[str, Any]:
    return {
        "receive": receive_sec,
        "stamp": _stamp_sec(message.header.stamp),
        "names": [str(name) for name in message.name],
        "position": np.asarray(message.position, dtype=np.float64).tolist(),
        "velocity": np.asarray(message.velocity, dtype=np.float64).tolist(),
        "effort": np.asarray(message.effort, dtype=np.float64).tolist(),
    }


def _object_pose_row(message: Any, receive_sec: float) -> dict[str, Any]:
    pose = message.pose.pose
    twist = message.twist.twist
    return {
        "receive": receive_sec,
        "stamp": _stamp_sec(message.header.stamp),
        "frame_id": str(message.header.frame_id),
        "child_frame_id": str(message.child_frame_id),
        "pose": [
            *_xyz(pose.position),
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        ],
        "twist": _spatial(twist.linear, twist.angular),
    }


def _clock_row(message: Any, receive_sec: float) -> dict[str, float]:
    return {"receive": receive_sec, "sim_time": _stamp_sec(message.clock)}


def _goal_id(value: Any) -> str:
    return bytes(value.uuid).hex()


def _gripper_feedback_row(message: Any, receive_sec: float) -> dict[str, Any]:
    feedback = message.feedback
    return {
        "receive": receive_sec,
        "goal_id": _goal_id(message.goal_id),
        "position": float(feedback.position),
        "effort": float(feedback.effort),
        "stalled": bool(feedback.stalled),
        "reached_goal": bool(feedback.reached_goal),
    }


def _gripper_status_rows(message: Any, receive_sec: float) -> list[dict[str, Any]]:
    return [
        {
            "receive": receive_sec,
            "goal_stamp": _stamp_sec(status.goal_info.stamp),
            "goal_id": _goal_id(status.goal_info.goal_id),
            "status": int(status.status),
        }
        for status in message.status_list
    ]


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
    monitored = TOPICS | OBSERVABILITY_TOPICS | {DESIRED_STATE_TOPIC}
    selected = monitored.intersection(available_types)
    decoded = DECODED_TOPICS.intersection(selected)
    message_types = {
        topic: get_message(available_types[topic]) for topic in decoded
    }
    raw: dict[str, list[dict[str, Any]]] = {topic: [] for topic in DECODED_TOPICS}
    topic_counts = {topic: 0 for topic in sorted(monitored)}
    reader.set_filter(rosbag2_py.StorageFilter(topics=sorted(selected)))
    while reader.has_next():
        topic, serialized, receive_ns = reader.read_next()
        if topic not in selected:
            continue
        topic_counts[topic] += 1
        if topic not in decoded:
            continue
        receive_sec = receive_ns * 1e-9
        message = deserialize_message(serialized, message_types[topic])
        if topic == CONTROL_TOPIC:
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
        elif topic == SENSOR_TOPIC:
            raw[topic].append(_joint_row(message, receive_sec, sensor=True))
        elif topic in {OUTPUT_TOPIC, HARDWARE_JOINT_TOPIC}:
            raw[topic].append(_joint_row(message, receive_sec, sensor=False))
        elif topic == MERGED_JOINT_TOPIC:
            raw[topic].append(_joint_row(message, receive_sec, sensor=False))
            gripper_row = _gripper_joint_row(message, receive_sec)
            if gripper_row is not None:
                gripper_row["source_topic"] = topic
                raw[GRIPPER_JOINT_TOPIC].append(gripper_row)
        elif topic == DESIRED_STATE_TOPIC:
            raw[topic].append(_joint_row(message, receive_sec, sensor=False))
        elif topic == ROBOT_STATE_TOPIC:
            raw[topic].append(_robot_state_row(message, receive_sec))
        elif topic == PLANNER_DIAGNOSTICS_TOPIC:
            raw[topic].append({"receive": receive_sec, "data": json.loads(message.data)})
        elif topic == ROS_DIAGNOSTICS_TOPIC:
            raw[topic].extend(_ros_diagnostic_rows(message, receive_sec))
        elif topic == CONTROLLER_ACTIVITY_TOPIC:
            raw[topic].extend(_controller_activity_rows(message, receive_sec))
        elif topic == GRIPPER_JOINT_TOPIC:
            gripper_row = _gripper_joint_row(message, receive_sec)
            if gripper_row is not None:
                gripper_row["source_topic"] = topic
                raw[topic].append(gripper_row)
        elif topic in {GRIPPER_FEEDBACK_TOPIC, SIM_GRIPPER_FEEDBACK_TOPIC}:
            row = _gripper_feedback_row(message, receive_sec)
            row["source_topic"] = topic
            raw[topic].append(row)
        elif topic in {GRIPPER_STATUS_TOPIC, SIM_GRIPPER_STATUS_TOPIC}:
            rows = _gripper_status_rows(message, receive_sec)
            for row in rows:
                row["source_topic"] = topic
            raw[topic].extend(rows)
        elif topic == SIM_OBJECT_POSE_TOPIC:
            raw[topic].append(_object_pose_row(message, receive_sec))
        elif topic == SIM_ACTUATOR_STATE_TOPIC:
            raw[topic].append(_generic_joint_row(message, receive_sec))
        elif topic == ROS_CLOCK_TOPIC:
            raw[topic].append(_clock_row(message, receive_sec))
        elif topic == ROSOUT_TOPIC:
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
        for topic in (CONTROL_TOPIC, SENSOR_TOPIC, OUTPUT_TOPIC)
        if not raw[topic]
    ]
    hardware_source = (
        HARDWARE_JOINT_TOPIC if raw[HARDWARE_JOINT_TOPIC] else MERGED_JOINT_TOPIC
    )
    if not raw[hardware_source]:
        missing.append(f"{HARDWARE_JOINT_TOPIC} or {MERGED_JOINT_TOPIC}")
    if missing:
        raise ValueError(f"bag is missing required populated topics: {', '.join(missing)}")

    # Diagnostics are latched/published independently. Keep one row per planner
    # step so references and counters cannot be biased by duplicates.
    by_step: dict[int, dict[str, Any]] = {}
    for item in raw[PLANNER_DIAGNOSTICS_TOPIC]:
        step = int(item["data"].get("planner_step_count", 0))
        if step > 0:
            by_step[step] = item
    diagnostic_items = [by_step[key] for key in sorted(by_step)]
    if not diagnostic_items:
        raise ValueError("bag contains no running planner diagnostics")

    manifest = _read_manifest(bag)
    declared_backend = manifest.get("backend")
    if declared_backend not in {"real", "mujoco"}:
        declared_backend = None
    if raw[HARDWARE_JOINT_TOPIC]:
        evidence_backend = "real"
    elif raw[SIM_OBJECT_POSE_TOPIC] or raw[SIM_ACTUATOR_STATE_TOPIC] or raw[ROS_CLOCK_TOPIC]:
        evidence_backend = "mujoco"
    else:
        evidence_backend = None
    backend = declared_backend or evidence_backend or (
        "real" if hardware_source == HARDWARE_JOINT_TOPIC else "mujoco"
    )
    backend_source = "manifest" if declared_backend is not None else "topic_evidence"
    backend_consistency_warning = None
    if (
        declared_backend is not None
        and evidence_backend is not None
        and declared_backend != evidence_backend
    ):
        backend_consistency_warning = (
            f"manifest declares backend={declared_backend}, while populated "
            f"backend-specific topics indicate {evidence_backend}"
        )

    gripper_joint_rows = raw[GRIPPER_JOINT_TOPIC]
    if any(
        row.get("source_topic") == GRIPPER_JOINT_TOPIC
        for row in gripper_joint_rows
    ):
        gripper_joint_rows = [
            row
            for row in gripper_joint_rows
            if row.get("source_topic") == GRIPPER_JOINT_TOPIC
        ]

    return RunData(
        bag=bag,
        control=_control_series(raw[CONTROL_TOPIC]),
        sensor=_joint_series(raw[SENSOR_TOPIC]),
        output=_joint_series(raw[OUTPUT_TOPIC]),
        hardware=_joint_series(raw[hardware_source]),
        merged=_optional_joint_series(raw[MERGED_JOINT_TOPIC]),
        desired=_optional_joint_series(raw[DESIRED_STATE_TOPIC]),
        robot_state=sorted(raw[ROBOT_STATE_TOPIC], key=lambda row: row["receive"]),
        diagnostics=DiagnosticSeries(
            receive=np.asarray([item["receive"] for item in diagnostic_items]),
            rows=[item["data"] for item in diagnostic_items],
        ),
        rosout=sorted(raw[ROSOUT_TOPIC], key=lambda row: row["receive"]),
        ros_diagnostics=sorted(
            raw[ROS_DIAGNOSTICS_TOPIC], key=lambda row: row["receive"]
        ),
        controller_activity=sorted(
            raw[CONTROLLER_ACTIVITY_TOPIC], key=lambda row: row["receive"]
        ),
        gripper_joint_states=sorted(
            gripper_joint_rows, key=lambda row: row["receive"]
        ),
        gripper_feedback=sorted(
            raw[GRIPPER_FEEDBACK_TOPIC] + raw[SIM_GRIPPER_FEEDBACK_TOPIC],
            key=lambda row: row["receive"],
        ),
        gripper_status=sorted(
            raw[GRIPPER_STATUS_TOPIC] + raw[SIM_GRIPPER_STATUS_TOPIC],
            key=lambda row: row["receive"],
        ),
        topic_counts=topic_counts,
        hardware_source=hardware_source,
        sim_object_pose=sorted(
            raw[SIM_OBJECT_POSE_TOPIC], key=lambda row: row["receive"]
        ),
        sim_actuator_states=sorted(
            raw[SIM_ACTUATOR_STATE_TOPIC], key=lambda row: row["receive"]
        ),
        sim_clock=sorted(raw[ROS_CLOCK_TOPIC], key=lambda row: row["receive"]),
        backend=backend,
        backend_source=backend_source,
        backend_consistency_warning=backend_consistency_warning,
        recorder_realtime=(
            manifest.get("realtime", {})
            if isinstance(manifest.get("realtime", {}), dict)
            else {}
        ),
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


def _rms_by_column(values: np.ndarray) -> list[float | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [None] * (array.shape[1] if array.ndim == 2 else 0)
    return np.sqrt(np.mean(array**2, axis=0)).tolist()


def _max_abs_by_column(values: np.ndarray) -> list[float | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [None] * (array.shape[1] if array.ndim == 2 else 0)
    return np.max(np.abs(array), axis=0).tolist()


def _robot_matrix(data: RunData, key: str) -> np.ndarray:
    return np.asarray([row[key] for row in data.robot_state], dtype=np.float64)


def _reconstruct_agimus_limiter(data: RunData) -> LimiterReconstruction | None:
    """Reconstruct sparse tau_J_d observations using read-update-write order.

    A robot-state sample is read before the same-cycle LFC request is written.
    Between two 100 Hz broadcaster samples, replay every recorded 1 kHz request
    through the Agimus component-wise limiter.  Intervals with a missing output
    sample are retained for plotting but excluded from error statistics.
    """
    if not data.robot_state:
        return None

    state_stamp = np.asarray(
        [row["stamp"] for row in data.robot_state], dtype=np.float64
    )
    state_q = _robot_matrix(data, "measured_position")
    observed = _robot_matrix(data, "desired_effort")
    output_order = np.argsort(data.output.stamp, kind="stable")
    output_stamp = data.output.stamp[output_order]
    output_effort = data.output.effort[output_order]
    output_q = data.output.q[output_order]

    same_cycle = nearest_indices(output_stamp, state_stamp)
    same_cycle_output = output_effort[same_cycle]
    same_cycle_q_error = np.max(
        np.abs(output_q[same_cycle] - state_q), axis=1
    )
    same_cycle_stamp_skew_ms = 1e3 * (output_stamp[same_cycle] - state_stamp)

    prior_output = np.full_like(observed, np.nan)
    has_prior = same_cycle > 0
    prior_output[has_prior] = output_effort[same_cycle[has_prior] - 1]

    predicted = observed.copy()
    interval_output_count = np.zeros(len(state_stamp), dtype=np.int64)
    interval_max_gap_ms = np.full(len(state_stamp), np.nan)
    valid_interval = np.zeros(len(state_stamp), dtype=bool)
    for index in range(len(state_stamp) - 1):
        start = state_stamp[index]
        stop = state_stamp[index + 1]
        left = int(np.searchsorted(output_stamp, start, side="left"))
        right = int(np.searchsorted(output_stamp, stop, side="left"))
        interval_stamps = output_stamp[left:right]
        interval_output_count[index + 1] = len(interval_stamps)
        if stop <= start or not len(interval_stamps):
            continue
        command = observed[index].copy()
        for effort in output_effort[left:right]:
            command += np.clip(
                effort - command,
                -AGIMUS_COMPONENT_STEP_AUDIT_NM,
                AGIMUS_COMPONENT_STEP_AUDIT_NM,
            )
        predicted[index + 1] = command
        boundary_stamps = np.r_[start, interval_stamps, stop]
        interval_max_gap_ms[index + 1] = float(
            1e3 * np.max(np.diff(boundary_stamps))
        )
        valid_interval[index + 1] = interval_max_gap_ms[index + 1] <= 1.5

    return LimiterReconstruction(
        state_stamp=state_stamp,
        observed=observed,
        predicted=predicted,
        prior_output=prior_output,
        same_cycle_output=same_cycle_output,
        same_cycle_q_error=same_cycle_q_error,
        same_cycle_stamp_skew_ms=same_cycle_stamp_skew_ms,
        interval_output_count=interval_output_count,
        interval_max_gap_ms=interval_max_gap_ms,
        valid_interval=valid_interval,
    )


def _quaternion_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_norm = np.linalg.norm(first, axis=1)
    second_norm = np.linalg.norm(second, axis=1)
    valid = (first_norm > 0.0) & (second_norm > 0.0)
    result = np.full(len(first), np.nan)
    normalized_dot = np.sum(first[valid] * second[valid], axis=1) / (
        first_norm[valid] * second_norm[valid]
    )
    result[valid] = 2.0 * np.arccos(np.clip(np.abs(normalized_dot), 0.0, 1.0))
    return result


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
        "raw_state_source": data.hardware_source,
        "interpretation": (
            (
                "/sensor and /franka/joint_states are two publications derived "
                "from the same Franka RobotState. Agreement detects ROS transport "
                "or mapping corruption but cannot prove absolute calibration."
            )
            if data.backend == "real"
            else (
                "/sensor and /joint_states are two mappings of the same MuJoCo "
                "ros2_control state. Agreement checks transport and joint mapping; "
                "it is not a comparison with independent hardware sensing."
            )
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
    if not data.robot_state:
        if data.desired is None:
            return {
                "available": False,
                "source": None,
                "note": (
                    (
                        "Agimus/FCI state is backend-specific and is correctly "
                        "unavailable in this MuJoCo simulation."
                    )
                    if data.backend == "mujoco"
                    else (
                        "The real run recorded no canonical Agimus/FCI state, so "
                        "rate-limited desired torque, robot mode, and reflex/error "
                        "bits cannot be diagnosed from this bag."
                    )
                ),
            }
        output_order = np.argsort(data.output.stamp, kind="stable")
        output_stamp = data.output.stamp[output_order]
        output_effort = data.output.effort[output_order]
        prior = np.searchsorted(output_stamp, data.desired.stamp, side="left") - 1
        valid = prior >= 0
        delta = data.desired.effort[valid] - output_effort[prior[valid]]
        return {
            "available": True,
            "full_fci_state_available": False,
            "source": DESIRED_STATE_TOPIC,
            "tau_J_d_count": int(len(data.desired.effort)),
            "tau_J_d_period_ms": _stats(1e3 * np.diff(data.desired.stamp)),
            "tau_J_d_minus_causal_prior_lfc_rms_nm_by_joint": _rms_by_column(delta),
            "tau_J_d_minus_causal_prior_lfc_max_abs_nm_by_joint": (
                _max_abs_by_column(delta)
            ),
            "note": (
                "Only the derived desired-joint-state topic was available. "
                "tau_J_d is compared with the preceding LFC output by header "
                "stamp; complete FCI state and interval limiter reconstruction "
                "are unavailable."
            ),
        }

    receive = np.asarray([row["receive"] for row in data.robot_state])
    stamp = np.asarray([row["stamp"] for row in data.robot_state])
    robot_time = np.asarray([row["robot_time"] for row in data.robot_state])
    measured_position = _robot_matrix(data, "measured_position")
    measured_velocity = _robot_matrix(data, "measured_velocity")
    desired_position = _robot_matrix(data, "desired_position")
    desired_velocity = _robot_matrix(data, "desired_velocity")
    motor_position = _robot_matrix(data, "motor_position")
    external_effort = _robot_matrix(data, "external_effort")
    measured_pose = _robot_matrix(data, "measured_ee_pose")
    desired_pose = _robot_matrix(data, "desired_ee_pose")
    wrench_base = _robot_matrix(data, "external_wrench_base")
    success_rate = np.asarray(
        [row["control_command_success_rate"] for row in data.robot_state]
    )
    robot_mode = np.asarray([row["robot_mode"] for row in data.robot_state])
    collision_max = np.asarray(
        [row["collision_indicator_max"] for row in data.robot_state]
    )
    contact_max = np.asarray(
        [row["contact_indicator_max"] for row in data.robot_state]
    )

    mode_counts: dict[str, int] = {}
    current_error_counts: dict[str, int] = {}
    last_motion_error_counts: dict[str, int] = {}
    for row in data.robot_state:
        mode = mode_names.get(row["robot_mode"], f"UNKNOWN_{row['robot_mode']}")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        for name in row["current_errors"]:
            current_error_counts[name] = current_error_counts.get(name, 0) + 1
        for name in row["last_motion_errors"]:
            last_motion_error_counts[name] = last_motion_error_counts.get(name, 0) + 1

    reconstructed = _reconstruct_agimus_limiter(data)
    assert reconstructed is not None
    valid = reconstructed.valid_interval
    prediction_error = reconstructed.observed[valid] - reconstructed.predicted[valid]
    has_prior = np.all(np.isfinite(reconstructed.prior_output), axis=1)
    causal_prior_delta = (
        reconstructed.observed[has_prior] - reconstructed.prior_output[has_prior]
    )

    numeric_keys = (
        "measured_position",
        "measured_velocity",
        "measured_effort",
        "desired_position",
        "desired_velocity",
        "desired_effort",
        "motor_position",
        "motor_velocity",
        "desired_acceleration",
        "measured_torque_derivative",
        "external_effort",
        "elbow_position",
        "elbow_desired_position",
        "elbow_commanded_position",
        "elbow_commanded_velocity",
        "elbow_commanded_acceleration",
        "external_wrench_stiffness",
        "external_wrench_base",
        "inertia_ee",
        "inertia_load",
        "inertia_total",
        "measured_ee_pose",
        "desired_ee_pose",
        "commanded_ee_pose",
        "flange_to_ee_pose",
        "ee_to_stiffness_pose",
        "desired_ee_twist",
        "commanded_ee_twist",
        "commanded_ee_acceleration",
        "cartesian_collision",
        "cartesian_contact",
        "joint_collision",
        "joint_contact",
    )
    all_numeric = np.concatenate(
        [_robot_matrix(data, key).reshape(-1) for key in numeric_keys]
        + [robot_time, receive, stamp, success_rate, robot_mode, collision_max, contact_max]
    )
    measured_to_desired_position = np.linalg.norm(
        measured_pose[:, :3] - desired_pose[:, :3], axis=1
    )
    measured_to_desired_orientation = _quaternion_distance(
        measured_pose[:, 3:], desired_pose[:, 3:]
    )
    limiter_summary = {
        "component_step_limit_nm_per_1khz_cycle": AGIMUS_COMPONENT_STEP_AUDIT_NM,
        "valid_interval_count": int(np.sum(valid)),
        "incomplete_interval_count": int(np.sum(~valid[1:])),
        "outputs_per_100hz_interval": _stats(
            reconstructed.interval_output_count[1:]
        ),
        "interval_max_output_stamp_gap_ms": _stats(
            reconstructed.interval_max_gap_ms[1:]
        ),
        "prediction_minus_observed_rms_nm_by_joint": _rms_by_column(
            prediction_error
        ),
        "prediction_minus_observed_max_abs_nm_by_joint": _max_abs_by_column(
            prediction_error
        ),
        "tau_J_d_minus_causal_prior_lfc_rms_nm_by_joint": _rms_by_column(
            causal_prior_delta
        ),
        "tau_J_d_minus_causal_prior_lfc_max_abs_nm_by_joint": (
            _max_abs_by_column(causal_prior_delta)
        ),
        "same_cycle_state_q_match_max_abs_rad": _stats(
            reconstructed.same_cycle_q_error
        ),
        "same_cycle_output_minus_state_stamp_ms": _stats(
            reconstructed.same_cycle_stamp_skew_ms
        ),
        "note": (
            "The controller loop is read -> update -> write. For each pair of "
            "100 Hz FCI states, every recorded 1 kHz LFC request in the header-"
            "stamp interval is replayed through the Agimus component limiter. "
            "Intervals with an output gap above 1.5 ms are excluded. tau_J_d is "
            "desired link-side torque without gravity; tau_J is measured total "
            "link-side torque and is not directly subtracted as command error."
        ),
    }
    return {
        "available": True,
        "full_fci_state_available": True,
        "source": ROBOT_STATE_TOPIC,
        "message_count": int(len(data.robot_state)),
        "stamp_period_ms": _stats(1e3 * np.diff(stamp)),
        "robot_time_period_ms": _stats(1e3 * np.diff(robot_time)),
        "receive_latency_ms": _stats(1e3 * (receive - stamp)),
        "nonpositive_stamp_period_count": int(np.sum(np.diff(stamp) <= 0.0)),
        "all_exposed_numeric_values_finite": bool(np.all(np.isfinite(all_numeric))),
        "robot_mode_counts": mode_counts,
        "current_error_active_message_counts": current_error_counts,
        "last_motion_error_active_message_counts": last_motion_error_counts,
        "last_motion_errors_seen": sorted(last_motion_error_counts),
        "control_command_success_rate": _stats(success_rate),
        "collision_indicator_max": _stats(collision_max),
        "contact_indicator_max": _stats(contact_max),
        "joint_motion_generator_tracking": {
            "q_minus_q_d_rms_rad_by_joint": _rms_by_column(
                measured_position - desired_position
            ),
            "q_minus_q_d_max_abs_rad_by_joint": _max_abs_by_column(
                measured_position - desired_position
            ),
            "dq_minus_dq_d_rms_rad_s_by_joint": _rms_by_column(
                measured_velocity - desired_velocity
            ),
            "motor_theta_minus_link_q_rms_rad_by_joint": _rms_by_column(
                motor_position - measured_position
            ),
            "note": (
                "q_d/dq_d are FCI motion-generator state, not the SB-MPC task "
                "reference while the arm is torque controlled."
            ),
        },
        "end_effector_motion_generator_tracking": {
            "measured_minus_desired_position_norm_m": _stats(
                measured_to_desired_position
            ),
            "measured_minus_desired_orientation_rad": _stats(
                measured_to_desired_orientation
            ),
            "note": (
                "O_T_EE_d/O_T_EE_c are FCI motion-generator states, not the "
                "SB-MPC task-space goal while torque controlled."
            ),
        },
        "interaction": {
            "external_joint_torque_norm_nm": _stats(
                np.linalg.norm(external_effort, axis=1)
            ),
            "external_force_base_norm_n": _stats(
                np.linalg.norm(wrench_base[:, :3], axis=1)
            ),
            "external_torque_base_norm_nm": _stats(
                np.linalg.norm(wrench_base[:, 3:], axis=1)
            ),
        },
        "latest_inertia": {
            "ee_mass_kg": data.robot_state[-1]["inertia_ee"][0],
            "load_mass_kg": data.robot_state[-1]["inertia_load"][0],
            "total_mass_kg": data.robot_state[-1]["inertia_total"][0],
            "ee": data.robot_state[-1]["inertia_ee"],
            "load": data.robot_state[-1]["inertia_load"],
            "total": data.robot_state[-1]["inertia_total"],
        },
        "limiter_reconstruction": limiter_summary,
        "note": (
            "current_errors are active FCI faults. last_motion_errors are "
            "historical reasons for the previous aborted motion and are never "
            "reported as current faults."
        ),
    }


def _recorded_position_error(row: dict[str, Any]) -> float | None:
    phase = row.get("phase_machine") or {}
    value = phase.get("ee_position_error_norm_m", row.get("last_position_error"))
    return float(value) if value is not None else None


def _recorded_vector(
    row: dict[str, Any],
    *,
    phase_key: str | None,
    row_key: str,
    size: int,
) -> np.ndarray | None:
    """Return one finite diagnostic vector, preferring the active phase gate."""
    phase = row.get("phase_machine") or {}
    value = phase.get(phase_key) if phase_key is not None else None
    if value is None:
        value = row.get(row_key)
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        return None
    return array


def _recorded_ee_position(row: dict[str, Any]) -> np.ndarray | None:
    return _recorded_vector(
        row,
        phase_key="ee_position_m",
        row_key="last_ee_position",
        size=3,
    )


def _recorded_goal_position(row: dict[str, Any]) -> np.ndarray | None:
    return _recorded_vector(
        row,
        phase_key="ee_goal_position_m",
        row_key="last_goal_position",
        size=3,
    )


def _recorded_position_error_signed(row: dict[str, Any]) -> np.ndarray | None:
    return _recorded_vector(
        row,
        phase_key="ee_position_error_signed_m",
        row_key="last_position_error_signed",
        size=3,
    )


def _recorded_ee_rotation(row: dict[str, Any]) -> np.ndarray | None:
    return _recorded_vector(
        row,
        phase_key=None,
        row_key="last_ee_rotation",
        size=9,
    )


def _recorded_goal_rotation(row: dict[str, Any]) -> np.ndarray | None:
    return _recorded_vector(
        row,
        phase_key=None,
        row_key="last_goal_rotation",
        size=9,
    )


def _recorded_orientation_error(row: dict[str, Any]) -> float | None:
    phase = row.get("phase_machine") or {}
    value = phase.get("ee_orientation_error_rad", row.get("last_orientation_error"))
    return float(value) if value is not None else None


def _phase(row: dict[str, Any]) -> str:
    phase = row.get("phase_machine") or {}
    return str(phase.get("phase", row.get("last_phase", "UNKNOWN")))


def _numeric_text(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _ros_control_diagnostics_summary(data: RunData) -> dict[str, Any]:
    rows = data.ros_diagnostics
    if not rows:
        return {
            "available": False,
            "note": f"{ROS_DIAGNOSTICS_TOPIC} was not populated in this bag.",
        }

    level_names = {0: "OK", 1: "WARN", 2: "ERROR", 3: "STALE"}
    level_counts: dict[str, int] = {}
    event_counts: dict[tuple[int, str, str], int] = {}
    state_counts: dict[str, dict[str, int]] = {}
    range_values: dict[str, list[dict[str, Any]]] = {}
    scalar_values: dict[str, list[float]] = {}
    for row in rows:
        level = int(row["level"])
        level_name = level_names.get(level, f"UNKNOWN_{level}")
        level_counts[level_name] = level_counts.get(level_name, 0) + 1
        event = (level, row["name"], row["message"])
        event_counts[event] = event_counts.get(event, 0) + 1
        for key, value in row["values"].items():
            if key.endswith(".state"):
                counts = state_counts.setdefault(key, {})
                counts[value] = counts.get(value, 0) + 1
            parsed = _diagnostic_statistics(value)
            if parsed is not None:
                range_values.setdefault(key, []).append(parsed)
            elif (number := _numeric_text(value)) is not None:
                scalar_values.setdefault(key, []).append(number)

    range_summary: dict[str, Any] = {}
    for key, values in sorted(range_values.items()):
        range_summary[key] = {
            "unit": values[0]["unit"],
            "reported_average": _stats(
                np.asarray([value["average"] for value in values])
            ),
            "observed_min": float(min(value["minimum"] for value in values)),
            "observed_max": float(max(value["maximum"] for value in values)),
            "reported_stddev": _stats(
                np.asarray([value["stddev"] for value in values])
            ),
            "desired": next(
                (value["desired"] for value in reversed(values) if value["desired"] is not None),
                None,
            ),
        }

    latest_activity: dict[str, dict[str, Any]] = {}
    for row in data.controller_activity:
        latest_activity[f"{row['kind']}:{row['name']}"] = {
            "state_id": row["state_id"],
            "state_label": row["state_label"],
            "stamp": row["stamp"],
        }

    return {
        "available": True,
        "status_row_count": len(rows),
        "level_counts": level_counts,
        "status_events": [
            {
                "level": level_names.get(level, f"UNKNOWN_{level}"),
                "name": name,
                "message": message,
                "count": count,
            }
            for (level, name, message), count in sorted(
                event_counts.items(), key=lambda item: (-item[0][0], item[0][1], item[0][2])
            )
        ],
        "state_counts": state_counts,
        "range_metrics": range_summary,
        "scalar_metrics": {
            key: _stats(np.asarray(values))
            for key, values in sorted(scalar_values.items())
        },
        "latest_activity": latest_activity,
        "note": (
            "Diagnostic levels and messages are reported verbatim from "
            "ros2_control. Timing ranges are parsed without changing their "
            "meaning; controller-manager periodicity should be interpreted "
            "alongside recorded stream header periods."
        ),
    }


def _gripper_summary(data: RunData) -> dict[str, Any]:
    status_names = {
        0: "UNKNOWN",
        1: "ACCEPTED",
        2: "EXECUTING",
        3: "CANCELING",
        4: "SUCCEEDED",
        5: "CANCELED",
        6: "ABORTED",
    }
    if not (
        data.gripper_joint_states or data.gripper_feedback or data.gripper_status
    ):
        return {
            "available": False,
            "note": "No gripper joint, feedback, or action-status data was recorded.",
        }
    status_counts: dict[str, int] = {}
    latest_status_by_goal: dict[str, str] = {}
    for row in data.gripper_status:
        name = status_names.get(row["status"], f"UNKNOWN_{row['status']}")
        status_counts[name] = status_counts.get(name, 0) + 1
        latest_status_by_goal[row["goal_id"]] = name
    widths = np.asarray(
        [row["width"] for row in data.gripper_joint_states if row["width"] is not None]
    )
    feedback_positions = np.asarray(
        [row["position"] for row in data.gripper_feedback], dtype=np.float64
    )
    feedback_efforts = np.asarray(
        [row["effort"] for row in data.gripper_feedback], dtype=np.float64
    )
    return {
        "available": True,
        "joint_state_count": len(data.gripper_joint_states),
        "feedback_count": len(data.gripper_feedback),
        "status_count": len(data.gripper_status),
        "measured_width_m": _stats(widths),
        "feedback_position_m": _stats(feedback_positions),
        "feedback_effort_n": _stats(feedback_efforts),
        "feedback_stalled_count": int(
            sum(bool(row["stalled"]) for row in data.gripper_feedback)
        ),
        "feedback_reached_goal_count": int(
            sum(bool(row["reached_goal"]) for row in data.gripper_feedback)
        ),
        "status_counts": status_counts,
        "latest_status_by_goal": latest_status_by_goal,
    }


def _windowed_clock_rtf(
    receive: np.ndarray,
    simulation: np.ndarray,
    *,
    window_sec: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate RTF over wall-time windows without crossing clock resets."""
    if len(receive) < 2:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    break_after = np.flatnonzero(
        (np.diff(receive) <= 0.0) | (np.diff(simulation) < 0.0)
    )
    starts = np.concatenate(([0], break_after + 1))
    stops = np.concatenate((break_after + 1, [len(receive)]))
    sample_time: list[float] = []
    rtf: list[float] = []
    for start, stop in zip(starts, stops, strict=True):
        anchor = int(start)
        for index in range(anchor + 1, int(stop)):
            wall_delta = float(receive[index] - receive[anchor])
            if wall_delta < window_sec:
                continue
            sim_delta = float(simulation[index] - simulation[anchor])
            sample_time.append(float(receive[index]))
            rtf.append(sim_delta / wall_delta)
            anchor = index
    return np.asarray(sample_time), np.asarray(rtf)


def _simulation_summary(data: RunData) -> dict[str, Any]:
    if data.backend != "mujoco":
        return {
            "available": False,
            "note": "MuJoCo ground-truth streams are not expected on the real backend.",
        }

    clock_receive = np.asarray(
        [row["receive"] for row in data.sim_clock], dtype=np.float64
    )
    clock_time = np.asarray(
        [row["sim_time"] for row in data.sim_clock], dtype=np.float64
    )
    clock_receive_delta = np.diff(clock_receive)
    clock_delta = np.diff(clock_time)
    continuous = (clock_delta >= 0.0) & (clock_receive_delta > 0.0)
    aggregate_wall = float(np.sum(clock_receive_delta[continuous]))
    aggregate_rtf = (
        float(np.sum(clock_delta[continuous]) / aggregate_wall)
        if aggregate_wall > 0.0
        else None
    )
    _, windowed_rtf = _windowed_clock_rtf(clock_receive, clock_time)

    object_pose = np.asarray(
        [row["pose"] for row in data.sim_object_pose], dtype=np.float64
    )
    object_twist = np.asarray(
        [row["twist"] for row in data.sim_object_pose], dtype=np.float64
    )
    object_summary: dict[str, Any] = {
        "available": bool(len(object_pose)),
        "message_count": len(data.sim_object_pose),
    }
    if len(object_pose):
        initial = object_pose[0, :3]
        final = object_pose[-1, :3]
        object_summary.update(
            {
                "initial_position_m": initial.tolist(),
                "final_position_m": final.tolist(),
                "maximum_height_m": float(np.max(object_pose[:, 2])),
                "maximum_lift_from_initial_m": float(
                    np.max(object_pose[:, 2] - initial[2])
                ),
                "total_displacement_m": float(np.linalg.norm(final - initial)),
                "linear_speed_m_s": _stats(
                    np.linalg.norm(object_twist[:, :3], axis=1)
                ),
                "angular_speed_rad_s": _stats(
                    np.linalg.norm(object_twist[:, 3:], axis=1)
                ),
            }
        )
        final_diagnostic = data.diagnostics.rows[-1]
        task_goal = final_diagnostic.get("last_goal_position")
        is_pick_place = isinstance(final_diagnostic.get("phase_machine"), dict)
        if is_pick_place and task_goal is not None and len(task_goal) >= 2:
            object_summary["final_xy_error_to_recorded_task_goal_m"] = float(
                np.linalg.norm(final[:2] - np.asarray(task_goal[:2], dtype=np.float64))
            )

    actuator_values = [
        float(value)
        for row in data.sim_actuator_states
        for key in ("position", "velocity", "effort")
        for value in row[key]
    ]
    actuator_names = (
        data.sim_actuator_states[-1]["names"] if data.sim_actuator_states else []
    )
    return {
        "available": True,
        "clock": {
            "message_count": len(data.sim_clock),
            "sim_period_ms": _stats(1e3 * clock_delta),
            "receive_period_ms": _stats(1e3 * clock_receive_delta),
            "nonadvancing_or_reset_count": int(np.sum(clock_delta <= 0.0)),
            "reset_count": int(np.sum(clock_delta < 0.0)),
            "aggregate_real_time_factor": aggregate_rtf,
            "window_sec": 0.5,
            "windowed_real_time_factor": _stats(windowed_rtf),
        },
        "object": object_summary,
        "actuators": {
            "message_count": len(data.sim_actuator_states),
            "names": actuator_names,
            "all_values_finite": bool(
                actuator_values and np.all(np.isfinite(actuator_values))
            ),
        },
        "note": (
            "MuJoCo actuator effort is qfrc_actuator and is not Franka FCI "
            "measured total link torque."
        ),
    }


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
    position_error_signed_mm = np.asarray(
        [
            np.full(3, np.nan) if value is None else 1e3 * value
            for row in run.diagnostics
            for value in [_recorded_position_error_signed(row)]
        ]
    )
    gain_ess = np.asarray(
        [
            np.nan if row.get("last_gain_ess") is None else row["last_gain_ess"]
            for row in run.diagnostics
        ],
        dtype=np.float64,
    )
    gain_nominal_weight = np.asarray(
        [
            (
                np.nan
                if row.get("last_gain_nominal_weight") is None
                else row["last_gain_nominal_weight"]
            )
            for row in run.diagnostics
        ],
        dtype=np.float64,
    )
    warning_rows = [row for row in data.rosout if row["level"] >= 30]
    final = run.diagnostics[-1]
    final_phase = final.get("phase_machine") or {}
    task_goal_position = final.get("last_goal_position")
    phase_goal_position = final_phase.get("ee_goal_position_m")
    goal_position = phase_goal_position or task_goal_position
    final_ee_position = _recorded_ee_position(final)
    final_position_error_signed_mm = _recorded_position_error_signed(final)
    if final_position_error_signed_mm is not None:
        final_position_error_signed_mm = 1e3 * final_position_error_signed_mm
    final_ee_rotation = _recorded_ee_rotation(final)
    goal_rotation = _recorded_goal_rotation(final)
    agimus_state = _agimus_robot_state_summary(data)
    ros_control_diagnostics = _ros_control_diagnostics_summary(data)
    gripper_observability = _gripper_summary(data)
    simulation = _simulation_summary(data)
    is_real = data.backend == "real"
    gripper_goal_count = int((final.get("gripper") or {}).get("goal_count") or 0)
    if is_real:
        torque_rate_audit = {
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
                "the Agimus one-cycle limit. The separate causal reconstruction "
                "uses tau_J_d from canonical FCI state."
            ),
        }
        expected_observability = {
            ROBOT_STATE_TOPIC,
            ROS_DIAGNOSTICS_TOPIC,
            CONTROLLER_ACTIVITY_TOPIC,
            GRIPPER_JOINT_TOPIC,
        }
        if gripper_goal_count:
            expected_observability.add(GRIPPER_STATUS_TOPIC)
    else:
        torque_rate_audit = {
            "source": "/output_joint_effort at the MuJoCo controller rate",
            "semantics": (
                "controller effort including model compensation, sent directly "
                "to the MuJoCo effort interface"
            ),
            "component_step_audit_threshold_nm": None,
            "component_step_abs_max_nm_by_joint": _max_abs_by_column(
                audited_component_step
            ),
            "component_steps_over_threshold_by_joint": None,
            "component_steps_over_threshold_total": None,
            "recorded_output_period_ms": _stats(active_output_period_ms),
            "audited_0p5_to_1p5ms_step_count": int(np.sum(one_cycle_step)),
            "recorded_step_gap_over_1p5ms_count": int(
                np.sum(active_output_period_ms > 1.5)
            ),
            "note": (
                "The Agimus 1 Nm/cycle limiter does not exist in MuJoCo and is "
                "therefore not applied as a simulation threshold."
            ),
        }
        expected_observability = {
            ROS_DIAGNOSTICS_TOPIC,
            CONTROLLER_ACTIVITY_TOPIC,
            SIM_OBJECT_POSE_TOPIC,
            SIM_ACTUATOR_STATE_TOPIC,
            ROS_CLOCK_TOPIC,
        }
        if gripper_goal_count:
            expected_observability.add(SIM_GRIPPER_STATUS_TOPIC)
    return {
        "schema": "sbmpc_run_report_v3",
        "bag": str(data.bag),
        "backend": data.backend,
        "backend_source": data.backend_source,
        "backend_consistency_warning": data.backend_consistency_warning,
        "recorder_realtime": data.recorder_realtime,
        "raw_arm_state_source": data.hardware_source,
        "topic_counts": data.topic_counts,
        "active_control_duration_sec": float(run.time[-1]),
        "control_count": int(len(run.time)),
        "planner_mode": final.get("planner_mode"),
        "last_phase": _phase(final),
        "recorded_goal_position_m": goal_position,
        "recorded_phase_goal_position_m": phase_goal_position,
        "recorded_task_goal_position_m": task_goal_position,
        "final_ee_position_m": (
            None if final_ee_position is None else final_ee_position.tolist()
        ),
        "final_ee_position_error_signed_mm": (
            None
            if final_position_error_signed_mm is None
            else final_position_error_signed_mm.tolist()
        ),
        "final_ee_rotation_matrix": (
            None
            if final_ee_rotation is None
            else final_ee_rotation.reshape(3, 3).tolist()
        ),
        "recorded_goal_rotation_matrix": (
            None if goal_rotation is None else goal_rotation.reshape(3, 3).tolist()
        ),
        "initial_q_minus_reference_rad": (run.q[0] - run.reference_q[0]).tolist(),
        "final_q_minus_reference_rad": (run.q[-1] - run.final_reference_q).tolist(),
        "final_q_minus_local_anchor_rad": local_anchor_q_error[-1].tolist(),
        "final_max_abs_q_error_rad": float(np.max(np.abs(q_error[-1]))),
        "position_error_mm": _stats(position_error_mm),
        "terminal_position_error_mm": _stats(position_error_mm[terminal_control]),
        "ee_position_error_signed_mm_by_axis": {
            axis: _stats(position_error_signed_mm[:, index])
            for index, axis in enumerate(("x", "y", "z"))
        },
        "terminal_ee_position_error_signed_mm_by_axis": {
            axis: _stats(position_error_signed_mm[terminal_control, index])
            for index, axis in enumerate(("x", "y", "z"))
        },
        "orientation_error_rad": _stats(orientation_error_rad),
        "terminal_orientation_error_rad": _stats(
            orientation_error_rad[terminal_control]
        ),
        "gain_sampling": {
            "ess": _stats(gain_ess),
            "terminal_ess": _stats(gain_ess[terminal_control]),
            "nominal_weight": _stats(gain_nominal_weight),
            "terminal_nominal_weight": _stats(
                gain_nominal_weight[terminal_control]
            ),
        },
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
        "torque_rate_audit": torque_rate_audit,
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
        "gripper_observability": gripper_observability,
        "rosout_warning_or_error_count": len(warning_rows),
        "rosout_warning_or_error": warning_rows,
        "ros_control_diagnostics": ros_control_diagnostics,
        "state_stream_consistency": _state_stream_summary(data),
        "agimus_robot_state": agimus_state,
        "simulation": simulation,
        "observability": {
            "lfc_output_is_pre_agimus_rate_limit": is_real,
            "post_rate_limit_tau_J_d_available": bool(
                is_real and agimus_state["available"]
            ),
            "fci_robot_state_available": bool(data.robot_state),
            "unpopulated_expected_topics": [
                topic
                for topic in sorted(expected_observability)
                if data.topic_counts.get(topic, 0) == 0
            ],
            "note": (
                (
                    "Measured /franka/joint_states effort is total tau_J and must "
                    "not be interpreted as gravity-free commanded effort."
                )
                if is_real
                else (
                    "MuJoCo joint/actuator effort is simulator actuator force, not "
                    "Franka FCI measured total link torque."
                )
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
    position_error_signed = np.asarray(
        [
            np.full(3, np.nan) if value is None else 1e3 * value
            for row in run.diagnostics
            for value in [_recorded_position_error_signed(row)]
        ]
    )
    has_signed_position = bool(np.any(np.isfinite(position_error_signed)))
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
    axes[0].plot(run.time, position_error, color="black", linewidth=2.0, label="norm")
    if has_signed_position:
        for index, axis_name in enumerate(("x", "y", "z")):
            axes[0].plot(
                run.time,
                position_error_signed[:, index],
                label=f"signed {axis_name}",
            )
        axes[0].axhline(0.0, color="black", linewidth=0.6)
        axes[0].legend(ncol=4)
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
    gain_ess = np.asarray(
        [row.get("last_gain_ess", np.nan) for row in run.diagnostics],
        dtype=np.float64,
    )
    gain_nominal_weight = np.asarray(
        [row.get("last_gain_nominal_weight", np.nan) for row in run.diagnostics],
        dtype=np.float64,
    )
    has_gain_health = bool(
        np.any(np.isfinite(gain_ess)) or np.any(np.isfinite(gain_nominal_weight))
    )
    row_count = 6 if has_gain_health else 5
    fig, axes = plt.subplots(
        row_count,
        1,
        figsize=(13, 16 if has_gain_health else 14),
        sharex=True,
    )
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
    step_axis_index = 4
    if has_gain_health:
        gain_axis = axes[4]
        gain_axis.plot(run.time, gain_ess, color="tab:blue", label="gain ESS")
        gain_axis.set_ylabel("gain ESS", color="tab:blue")
        weight_axis = gain_axis.twinx()
        weight_axis.plot(
            run.time,
            gain_nominal_weight,
            color="tab:orange",
            label="nominal weight",
        )
        weight_axis.set_ylim(-0.02, 1.02)
        weight_axis.set_ylabel("nominal weight", color="tab:orange")
        gain_lines = gain_axis.get_lines() + weight_axis.get_lines()
        gain_axis.legend(gain_lines, [line.get_label() for line in gain_lines])
        step_axis_index = 5
    axes[step_axis_index].plot(run.time, delta_ff, label="feedforward step")
    axes[step_axis_index].plot(
        run.time,
        delta_output,
        label="planner-aligned output step",
    )
    axes[step_axis_index].set_ylabel("step norm [Nm]")
    axes[step_axis_index].set_xlabel("active-control time [s]")
    axes[step_axis_index].legend()
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
    is_real = run.data.backend == "real"
    secondary = (
        run.feedforward + run.feedback - run.output if is_real else run.feedback
    )
    secondary_label = (
        "inferred removed gravity" if is_real else "feedback correction"
    )
    output_label = "pre-limiter output" if is_real else "MuJoCo effort request"
    measured_label = (
        "measured total tau_J" if is_real else "MuJoCo reported joint effort"
    )
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
    audited_steps = np.abs(np.diff(active_effort, axis=0)[one_cycle])
    component_step_max = (
        np.max(audited_steps, axis=0) if len(audited_steps) else np.zeros(7)
    )
    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=False)
    axes[0].plot(run.time, run.feedforward[:, focus], label="feedforward")
    axes[0].plot(
        run.time,
        secondary[:, focus],
        label=secondary_label,
    )
    axes[0].plot(run.time, run.output[:, focus], label=output_label)
    axes[0].plot(run.time, run.measured_tau[:, focus], label=measured_label)
    axes[0].set_ylabel(f"J{focus + 1} torque [Nm]")
    axes[0].legend(ncol=2)
    width = 0.20
    joints = np.arange(7)
    for offset, values, label in (
        (-1.5 * width, run.feedforward, "feedforward"),
        (-0.5 * width, secondary, secondary_label),
        (+0.5 * width, run.output, output_label),
        (+1.5 * width, run.measured_tau, measured_label),
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
    if is_real:
        axes[2].axhline(
            AGIMUS_COMPONENT_STEP_AUDIT_NM,
            color="tab:red",
            linestyle="--",
            label="Agimus one-cycle component limit",
        )
    axes[2].set_xticks(joints, [f"J{i}" for i in range(1, 8)])
    axes[2].set_ylabel("max 1 kHz output step [Nm]")
    if is_real:
        axes[2].legend()
    axes[3].axis("off")
    axes[3].text(
        0.02,
        0.85,
        (
            "Real-backend interpretation:\n"
            "• /output_joint_effort is gravity-free and precedes Agimus rate limiting.\n"
            "• inferred gravity uses the first output after each control packet.\n"
            "• measured tau_J is total link-side torque, not command error.\n"
            "• the Agimus panel reconstructs limiting from canonical robot_state."
            if is_real
            else
            "MuJoCo interpretation:\n"
            "• /output_joint_effort includes model compensation and drives MuJoCo.\n"
            "• reported joint effort is simulator actuator force, not FCI tau_J.\n"
            "• no Agimus torque-rate threshold is applied in simulation."
        ),
        va="top",
        fontsize=11,
    )
    for axis in axes[:3]:
        axis.grid(alpha=0.25)
    fig.suptitle(
        "Real torque path with command/measurement semantics"
        if is_real
        else "MuJoCo effort path with simulator semantics"
    )
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
        (
            "State-stream consistency (agreement cannot prove absolute encoder calibration)"
            if run.data.backend == "real"
            else "MuJoCo state-stream and joint-mapping consistency"
        )
    )
    return _save(fig, output, "08_sensor_integrity.png")


def _plot_agimus_robot_state(run: AlignedRun, output: Path) -> str | None:
    data = run.data
    if data.robot_state:
        stamp = np.asarray([row["stamp"] for row in data.robot_state])
        desired = _robot_matrix(data, "desired_effort")
        measured = _robot_matrix(data, "measured_effort")
        external = _robot_matrix(data, "external_effort")
        modes = np.asarray([row["robot_mode"] for row in data.robot_state])
        success = np.asarray(
            [row["control_command_success_rate"] for row in data.robot_state]
        )
        current_error_active = np.asarray(
            [bool(row["current_errors"]) for row in data.robot_state]
        )
        historical_error_active = np.asarray(
            [bool(row["last_motion_errors"]) for row in data.robot_state]
        )
        collision = np.asarray(
            [row["collision_indicator_max"] for row in data.robot_state]
        )
        contact = np.asarray(
            [row["contact_indicator_max"] for row in data.robot_state]
        )
        reconstructed = _reconstruct_agimus_limiter(data)
        assert reconstructed is not None
        valid = reconstructed.valid_interval
        residual = reconstructed.observed - reconstructed.predicted
        residual[~valid] = np.nan
        if np.any(valid):
            focus = int(
                np.nanargmax(np.sqrt(np.nanmean(residual[valid] ** 2, axis=0)))
            )
        else:
            focus = int(np.argmax(np.ptp(desired, axis=0)))
        time = stamp - data.control.stamp[0]
        fig, axes = plt.subplots(5, 1, figsize=(13, 15), sharex=True)
        axes[0].plot(
            time,
            reconstructed.same_cycle_output[:, focus],
            label="same-cycle LFC request (written after state read)",
        )
        axes[0].plot(time, desired[:, focus], label="observed FCI tau_J_d")
        axes[0].plot(
            time,
            np.where(valid, reconstructed.predicted[:, focus], np.nan),
            "--",
            label="causal limiter reconstruction",
        )
        axes[0].set_ylabel(f"J{focus + 1} [Nm]")
        axes[0].legend(ncol=2)
        axes[1].plot(time, residual)
        axes[1].set_ylabel("observed - reconstructed\ntau_J_d [Nm]")
        axes[2].plot(time, measured[:, focus], label="measured total tau_J")
        axes[2].plot(time, external[:, focus], label="estimated external tau")
        axes[2].set_ylabel(f"J{focus + 1} [Nm]")
        axes[2].legend()
        axes[3].step(time, modes, where="post", label="robot mode")
        axes[3].plot(time, collision, label="collision max")
        axes[3].plot(time, contact, label="contact max")
        axes[3].set_ylabel("FCI state / flags")
        axes[3].legend(ncol=3)
        axes[4].plot(time, success, label="command success rate")
        axes[4].step(
            time,
            current_error_active,
            where="post",
            label="current error active",
        )
        axes[4].step(
            time,
            historical_error_active,
            where="post",
            label="historical last-motion error present",
        )
        axes[4].set_ylim(-0.05, 1.05)
        axes[4].set_ylabel("rate / booleans")
        axes[4].legend(ncol=3)
    elif data.desired is not None:
        stamp = data.desired.stamp
        desired = data.desired.effort
        order = np.argsort(data.output.stamp, kind="stable")
        output_stamp = data.output.stamp[order]
        output_effort = data.output.effort[order]
        prior = np.searchsorted(output_stamp, stamp, side="left") - 1
        valid = prior >= 0
        pre_limiter = np.full_like(desired, np.nan)
        pre_limiter[valid] = output_effort[prior[valid]]
        delta = desired - pre_limiter
        focus = int(np.nanargmax(np.sqrt(np.nanmean(delta**2, axis=0))))
        time = stamp - data.control.stamp[0]
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        axes[0].plot(time, pre_limiter[:, focus], label="causal prior LFC request")
        axes[0].plot(time, desired[:, focus], label="Agimus tau_J_d")
        axes[0].set_ylabel(f"J{focus + 1} [Nm]")
        axes[0].legend()
        axes[1].plot(time, delta)
        axes[1].set_ylabel("tau_J_d - prior LFC [Nm]")
        axes[2].axis("off")
        axes[2].text(
            0.02,
            0.8,
            "Only the derived desired-joint-state stream was recorded; full FCI "
            "state and interval limiter reconstruction are unavailable.",
            va="top",
        )
    else:
        return None
    axes[-1].set_xlabel("active-control time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("Agimus causal torque path and FCI state")
    return _save(fig, output, "09_agimus_robot_state.png")


def _plot_agimus_joint_motor(run: AlignedRun, output: Path) -> str | None:
    data = run.data
    if not data.robot_state:
        return None
    time = np.asarray([row["stamp"] for row in data.robot_state]) - data.control.stamp[0]
    q = _robot_matrix(data, "measured_position")
    dq = _robot_matrix(data, "measured_velocity")
    q_d = _robot_matrix(data, "desired_position")
    dq_d = _robot_matrix(data, "desired_velocity")
    theta = _robot_matrix(data, "motor_position")
    tau_ext = _robot_matrix(data, "external_effort")
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    axes[0].plot(time, q - q_d)
    axes[0].set_ylabel("q - q_d [rad]")
    axes[1].plot(time, dq - dq_d)
    axes[1].set_ylabel("dq - dq_d [rad/s]")
    axes[2].plot(time, theta - q)
    axes[2].set_ylabel("motor theta - link q [rad]")
    axes[3].plot(time, tau_ext)
    axes[3].set_ylabel("external joint torque [Nm]")
    axes[3].set_xlabel("active-control time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(
        "Agimus joint/motor state (q_d and dq_d are FCI motion-generator state)"
    )
    return _save(fig, output, "10_agimus_joint_motor.png")


def _plot_agimus_ee_wrench(run: AlignedRun, output: Path) -> str | None:
    data = run.data
    if not data.robot_state:
        return None
    time = np.asarray([row["stamp"] for row in data.robot_state]) - data.control.stamp[0]
    measured = _robot_matrix(data, "measured_ee_pose")
    desired = _robot_matrix(data, "desired_ee_pose")
    commanded = _robot_matrix(data, "commanded_ee_pose")
    wrench_base = _robot_matrix(data, "external_wrench_base")
    wrench_stiffness = _robot_matrix(data, "external_wrench_stiffness")
    desired_twist = _robot_matrix(data, "desired_ee_twist")
    commanded_twist = _robot_matrix(data, "commanded_ee_twist")
    commanded_acceleration = _robot_matrix(data, "commanded_ee_acceleration")
    position_error_desired = np.linalg.norm(measured[:, :3] - desired[:, :3], axis=1)
    position_error_commanded = np.linalg.norm(
        measured[:, :3] - commanded[:, :3], axis=1
    )
    orientation_error_desired = _quaternion_distance(
        measured[:, 3:], desired[:, 3:]
    )
    orientation_error_commanded = _quaternion_distance(
        measured[:, 3:], commanded[:, 3:]
    )
    fig, axes = plt.subplots(6, 1, figsize=(13, 17), sharex=True)
    for dimension, label in enumerate(("x", "y", "z")):
        color = f"C{dimension}"
        axes[0].plot(time, measured[:, dimension], color=color, label=f"measured {label}")
        axes[0].plot(
            time,
            desired[:, dimension],
            "--",
            color=color,
            alpha=0.7,
            label=f"FCI desired {label}",
        )
    axes[0].set_ylabel("O_T_EE position [m]")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].plot(time, position_error_desired, label="measured - FCI desired")
    axes[1].plot(time, position_error_commanded, label="measured - FCI commanded")
    axes[1].set_ylabel("position error norm [m]")
    axes[1].legend()
    axes[2].plot(time, orientation_error_desired, label="measured - FCI desired")
    axes[2].plot(time, orientation_error_commanded, label="measured - FCI commanded")
    axes[2].set_ylabel("orientation error [rad]")
    axes[2].legend()
    axes[3].plot(
        time, np.linalg.norm(wrench_base[:, :3], axis=1), label="base force"
    )
    axes[3].plot(
        time,
        np.linalg.norm(wrench_stiffness[:, :3], axis=1),
        label="stiffness-frame force",
    )
    axes[3].set_ylabel("external force norm [N]")
    axes[3].legend()
    axes[4].plot(
        time, np.linalg.norm(wrench_base[:, 3:], axis=1), label="base torque"
    )
    axes[4].plot(
        time,
        np.linalg.norm(wrench_stiffness[:, 3:], axis=1),
        label="stiffness-frame torque",
    )
    axes[4].set_ylabel("external torque norm [Nm]")
    axes[4].legend()
    axes[5].plot(time, np.linalg.norm(desired_twist, axis=1), label="desired twist")
    axes[5].plot(
        time, np.linalg.norm(commanded_twist, axis=1), label="commanded twist"
    )
    axes[5].plot(
        time,
        np.linalg.norm(commanded_acceleration, axis=1),
        label="commanded acceleration",
    )
    axes[5].set_ylabel("FCI Cartesian norm")
    axes[5].set_xlabel("active-control time [s]")
    axes[5].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(
        "Agimus end-effector, wrench, and motion-generator state\n"
        "(FCI desired/commanded pose is not the SB-MPC task goal in torque mode)"
    )
    return _save(fig, output, "11_agimus_ee_wrench.png")


def _diagnostic_range_series(
    data: RunData, key: str, component: str
) -> tuple[np.ndarray, np.ndarray]:
    time: list[float] = []
    values: list[float] = []
    for row in data.ros_diagnostics:
        raw = row["values"].get(key)
        parsed = None if raw is None else _diagnostic_statistics(raw)
        if parsed is None:
            continue
        value = parsed.get(component)
        if value is None:
            continue
        time.append(row["receive"] - data.control.receive[0])
        values.append(float(value))
    return np.asarray(time), np.asarray(values)


def _diagnostic_scalar_series(
    data: RunData, key: str
) -> tuple[np.ndarray, np.ndarray]:
    time: list[float] = []
    values: list[float] = []
    for row in data.ros_diagnostics:
        raw = row["values"].get(key)
        value = None if raw is None else _numeric_text(raw)
        if value is None:
            continue
        time.append(row["receive"] - data.control.receive[0])
        values.append(value)
    return np.asarray(time), np.asarray(values)


def _plot_ros_control_timing(run: AlignedRun, output: Path) -> str | None:
    data = run.data
    if not data.ros_diagnostics:
        return None
    fig, axes = plt.subplots(5, 1, figsize=(13, 15), sharex=False)
    range_keys = sorted(
        {
            key
            for row in data.ros_diagnostics
            for key, raw in row["values"].items()
            if _diagnostic_statistics(raw) is not None
        }
    )
    controller_execution_keys = [
        key
        for key in range_keys
        if key.endswith(".execution_time")
        and ".read_cycle." not in key
        and ".write_cycle." not in key
    ]
    for key in controller_execution_keys:
        label = key.removesuffix(".execution_time")
        time, average = _diagnostic_range_series(data, key, "average")
        _, maximum = _diagnostic_range_series(data, key, "maximum")
        axes[0].plot(time, average, label=f"{label} average")
        axes[0].plot(time, maximum, "--", alpha=0.7, label=f"{label} observed max")
    axes[0].set_ylabel("controller execution [us]")
    if axes[0].lines:
        axes[0].legend(ncol=2, fontsize=8)
    hardware_execution_keys = [
        key
        for key in range_keys
        if key.endswith(".read_cycle.execution_time")
        or key.endswith(".write_cycle.execution_time")
    ]
    for key in hardware_execution_keys:
        label = key.removesuffix(".execution_time")
        time, average = _diagnostic_range_series(data, key, "average")
        _, maximum = _diagnostic_range_series(data, key, "maximum")
        axes[1].plot(time, average, label=f"{label} average")
        axes[1].plot(time, maximum, "--", alpha=0.7, label=f"{label} observed max")
    axes[1].set_ylabel("hardware execution [us]")
    if axes[1].lines:
        axes[1].legend(ncol=2, fontsize=8)
    periodicity_keys = [key for key in range_keys if key.endswith(".periodicity")]
    for key in periodicity_keys:
        label = key.removesuffix(".periodicity")
        time, average = _diagnostic_range_series(data, key, "average")
        _, desired = _diagnostic_range_series(data, key, "desired")
        axes[2].plot(time, average, label=f"{label} average")
        if len(desired):
            axes[2].plot(time, desired, ":", label=f"{label} desired")
    axes[2].set_ylabel("component rate [Hz]")
    if axes[2].lines:
        axes[2].legend(ncol=2, fontsize=8)
    for key, label in (
        ("periodicity.average", "average"),
        ("periodicity.min", "minimum"),
        ("periodicity.max", "maximum"),
    ):
        scalar_time, value = _diagnostic_scalar_series(data, key)
        axes[3].plot(scalar_time, value, label=label)
    axes[3].axhline(1000.0, color="black", linestyle=":", label="configured 1 kHz")
    axes[3].set_ylabel("controller-manager rate [Hz]")
    axes[3].legend(ncol=2)
    diagnostic_time = np.asarray(
        [row["receive"] - data.control.receive[0] for row in data.ros_diagnostics]
    )
    levels = np.asarray([row["level"] for row in data.ros_diagnostics])
    axes[4].scatter(diagnostic_time, levels, s=12, alpha=0.65)
    axes[4].set_yticks((0, 1, 2, 3), ("OK", "WARN", "ERROR", "STALE"))
    axes[4].set_ylabel("diagnostic level")
    axes[4].set_xlabel("active-control time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("ros2_control controller, hardware, and scheduler diagnostics")
    return _save(fig, output, "12_ros_control_timing.png")


def _plot_gripper_observability(run: AlignedRun, output: Path) -> str | None:
    data = run.data
    if not (
        data.gripper_joint_states or data.gripper_feedback or data.gripper_status
    ):
        return None
    t0 = data.control.receive[0]
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    if data.gripper_joint_states:
        joint_time = np.asarray(
            [row["receive"] - t0 for row in data.gripper_joint_states]
        )
        width = np.asarray([row["width"] for row in data.gripper_joint_states])
        effort = np.asarray(
            [np.sum(np.abs(row["effort"])) for row in data.gripper_joint_states]
        )
        axes[0].plot(joint_time, width, label="measured finger width")
        axes[1].plot(joint_time, effort, label="joint effort sum")
    if data.gripper_feedback:
        feedback_time = np.asarray(
            [row["receive"] - t0 for row in data.gripper_feedback]
        )
        axes[0].plot(
            feedback_time,
            [row["position"] for row in data.gripper_feedback],
            "--",
            label="action feedback position",
        )
        axes[1].plot(
            feedback_time,
            [row["effort"] for row in data.gripper_feedback],
            "--",
            label="action feedback effort",
        )
        axes[2].step(
            feedback_time,
            [row["stalled"] for row in data.gripper_feedback],
            where="post",
            label="stalled",
        )
        axes[2].step(
            feedback_time,
            [row["reached_goal"] for row in data.gripper_feedback],
            where="post",
            label="reached goal",
        )
    if data.gripper_status:
        axes[3].step(
            [row["receive"] - t0 for row in data.gripper_status],
            [row["status"] for row in data.gripper_status],
            where="post",
        )
    axes[0].set_ylabel("gripper width [m]")
    axes[1].set_ylabel("effort")
    axes[2].set_ylabel("feedback booleans")
    axes[3].set_yticks(
        range(7),
        ("unknown", "accepted", "executing", "canceling", "succeeded", "canceled", "aborted"),
    )
    axes[3].set_ylabel("action status")
    axes[3].set_xlabel("active-control time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend()
    fig.suptitle("Gripper joint and action observability")
    return _save(fig, output, "13_gripper_observability.png")


def _plot_simulation_ground_truth(run: AlignedRun, output: Path) -> str | None:
    data = run.data
    if not data.sim_object_pose and not data.sim_clock:
        return None
    t0 = data.control.receive[0]
    fig, axes = plt.subplots(5, 1, figsize=(13, 15), sharex=False)
    if data.sim_object_pose:
        time_values = np.asarray(
            [row["receive"] - t0 for row in data.sim_object_pose]
        )
        pose = np.asarray([row["pose"] for row in data.sim_object_pose])
        twist = np.asarray([row["twist"] for row in data.sim_object_pose])
        for dimension, label in enumerate(("x", "y", "z")):
            axes[0].plot(time_values, pose[:, dimension], label=label)
        final_diagnostic = run.diagnostics[-1]
        goal = final_diagnostic.get("last_goal_position")
        is_pick_place = isinstance(final_diagnostic.get("phase_machine"), dict)
        if is_pick_place and goal is not None:
            for dimension, label in enumerate(("x goal", "y goal")):
                axes[0].axhline(
                    float(goal[dimension]),
                    color=f"C{dimension}",
                    linestyle=":",
                    label=label,
                )
        axes[0].set_ylabel("cube position [m]")
        axes[0].legend(ncol=3)
        axes[1].plot(time_values, pose[:, 2] - pose[0, 2], label="lift")
        axes[1].plot(
            time_values,
            np.linalg.norm(pose[:, :3] - pose[0, :3], axis=1),
            label="3D displacement",
        )
        axes[1].set_ylabel("cube displacement [m]")
        axes[1].legend()
        axes[2].plot(
            time_values,
            np.linalg.norm(twist[:, :3], axis=1),
            label="linear speed",
        )
        axes[2].plot(
            time_values,
            np.linalg.norm(twist[:, 3:], axis=1),
            label="angular speed",
        )
        axes[2].set_ylabel("cube speed norm")
        axes[2].legend()
    if data.sim_clock:
        receive = np.asarray([row["receive"] for row in data.sim_clock])
        sim_time = np.asarray([row["sim_time"] for row in data.sim_clock])
        axes[3].plot(receive - t0, sim_time - sim_time[0], label="simulation time")
        axes[3].plot(receive - t0, receive - receive[0], "--", label="wall receive time")
        axes[3].set_ylabel("elapsed time [s]")
        axes[3].legend()
        rtf_time, windowed_rtf = _windowed_clock_rtf(receive, sim_time)
        axes[4].plot(rtf_time - t0, windowed_rtf)
        axes[4].axhline(1.0, color="black", linestyle=":")
        axes[4].set_ylabel("0.5 s windowed RTF")
        axes[4].set_xlabel("active-control receive time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("MuJoCo cube ground truth and simulation clock")
    return _save(fig, output, "14_simulation_ground_truth.png")


def _plot_mujoco_actuators(run: AlignedRun, output: Path) -> str | None:
    rows = run.data.sim_actuator_states
    if not rows:
        return None
    time_values = np.asarray(
        [row["receive"] - run.data.control.receive[0] for row in rows]
    )
    names = rows[-1]["names"]
    position = np.asarray([row["position"] for row in rows])
    velocity = np.asarray([row["velocity"] for row in rows])
    effort = np.asarray([row["effort"] for row in rows])
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    for index, name in enumerate(names):
        if index < position.shape[1]:
            axes[0].plot(time_values, position[:, index], label=name)
        if index < velocity.shape[1]:
            axes[1].plot(time_values, velocity[:, index], label=name)
        if index < effort.shape[1]:
            axes[2].plot(time_values, effort[:, index], label=name)
    axes[0].set_ylabel("MuJoCo qpos")
    axes[1].set_ylabel("MuJoCo qvel")
    axes[2].set_ylabel("qfrc_actuator")
    axes[2].set_xlabel("active-control receive time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)
    fig.suptitle("MuJoCo raw actuator state")
    return _save(fig, output, "15_mujoco_actuators.png")


_ROTATION_COMPONENTS = tuple(
    f"r{row}{column}" for row in range(3) for column in range(3)
)


def _write_steps_csv(run: AlignedRun, output: Path) -> None:
    fields = (
        "time_sec",
        "phase",
        "ee_error_mm",
        "ee_error_x_mm",
        "ee_error_y_mm",
        "ee_error_z_mm",
        "ee_orientation_error_rad",
        "ee_position_x_m",
        "ee_position_y_m",
        "ee_position_z_m",
        "ee_goal_x_m",
        "ee_goal_y_m",
        "ee_goal_z_m",
        "q_error_norm_rad",
        "q_error_max_rad",
        "velocity_norm_rad_s",
        "feedforward_norm_nm",
        "output_norm_nm",
        "feedback_norm_nm",
        "gain_norm",
        "gain_ess",
        "gain_nominal_weight",
        "planning_time_ms",
    ) + tuple(
        f"ee_rotation_{component}" for component in _ROTATION_COMPONENTS
    ) + tuple(
        f"goal_rotation_{component}" for component in _ROTATION_COMPONENTS
    )
    with (output / "controller_steps.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(run.diagnostics):
            q_error = run.q[index] - run.reference_q[index]
            ee_error = _recorded_position_error(row)
            ee_error_signed = _recorded_position_error_signed(row)
            ee_position = _recorded_ee_position(row)
            ee_goal = _recorded_goal_position(row)
            ee_rotation = _recorded_ee_rotation(row)
            goal_rotation = _recorded_goal_rotation(row)
            record = {
                "time_sec": run.time[index],
                "phase": _phase(row),
                "ee_error_mm": None if ee_error is None else 1e3 * ee_error,
                "ee_orientation_error_rad": _recorded_orientation_error(row),
                "q_error_norm_rad": np.linalg.norm(q_error),
                "q_error_max_rad": np.max(np.abs(q_error)),
                "velocity_norm_rad_s": np.linalg.norm(run.v[index]),
                "feedforward_norm_nm": np.linalg.norm(run.feedforward[index]),
                "output_norm_nm": np.linalg.norm(run.output[index]),
                "feedback_norm_nm": np.linalg.norm(run.feedback[index]),
                "gain_norm": np.linalg.norm(run.gain[index]),
                "gain_ess": row.get("last_gain_ess"),
                "gain_nominal_weight": row.get("last_gain_nominal_weight"),
                "planning_time_ms": row.get("last_planning_time_ms"),
            }
            for component, value in zip(
                ("x", "y", "z"),
                np.full(3, np.nan)
                if ee_error_signed is None
                else 1e3 * ee_error_signed,
            ):
                record[f"ee_error_{component}_mm"] = value
            for prefix, vector in (("ee_position", ee_position), ("ee_goal", ee_goal)):
                for component, value in zip(
                    ("x", "y", "z"),
                    np.full(3, np.nan) if vector is None else vector,
                ):
                    record[f"{prefix}_{component}_m"] = value
            for prefix, rotation in (
                ("ee_rotation", ee_rotation),
                ("goal_rotation", goal_rotation),
            ):
                for component, value in zip(
                    _ROTATION_COMPONENTS,
                    np.full(9, np.nan) if rotation is None else rotation,
                ):
                    record[f"{prefix}_{component}"] = value
            writer.writerow(record)


_JOINT_COMPONENTS = tuple(f"j{index}" for index in range(1, 8))
_SPATIAL_COMPONENTS = ("x", "y", "z", "rx", "ry", "rz")
_ROBOT_VECTOR_COMPONENTS = {
    **{
        key: _JOINT_COMPONENTS
        for key in (
            "measured_position",
            "measured_velocity",
            "measured_effort",
            "desired_position",
            "desired_velocity",
            "desired_effort",
            "motor_position",
            "motor_velocity",
            "desired_acceleration",
            "measured_torque_derivative",
            "external_effort",
            "joint_collision",
            "joint_contact",
        )
    },
    **{
        key: ("value", "sign")
        for key in (
            "elbow_position",
            "elbow_desired_position",
            "elbow_commanded_position",
            "elbow_commanded_velocity",
            "elbow_commanded_acceleration",
        )
    },
    **{
        key: ("force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z")
        for key in ("external_wrench_stiffness", "external_wrench_base")
    },
    **{
        key: ("mass", "com_x", "com_y", "com_z", "ixx", "ixy", "ixz", "iyy", "iyz", "izz")
        for key in ("inertia_ee", "inertia_load", "inertia_total")
    },
    **{
        key: ("x", "y", "z", "qx", "qy", "qz", "qw")
        for key in (
            "measured_ee_pose",
            "desired_ee_pose",
            "commanded_ee_pose",
            "flange_to_ee_pose",
            "ee_to_stiffness_pose",
        )
    },
    **{
        key: _SPATIAL_COMPONENTS
        for key in (
            "desired_ee_twist",
            "commanded_ee_twist",
            "commanded_ee_acceleration",
            "cartesian_collision",
            "cartesian_contact",
        )
    },
}


def _flatten_robot_state_row(row: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in row.items():
        if key in {"current_errors", "last_motion_errors"}:
            flattened[key] = json.dumps(value, separators=(",", ":"))
        elif key in _ROBOT_VECTOR_COMPONENTS:
            labels = _ROBOT_VECTOR_COMPONENTS[key]
            if len(value) != len(labels):
                raise ValueError(
                    f"{key} has {len(value)} values; expected {len(labels)}"
                )
            flattened.update(
                {f"{key}_{label}": item for label, item in zip(labels, value)}
            )
        elif isinstance(value, (list, tuple)):
            flattened.update(
                {f"{key}_{index + 1}": item for index, item in enumerate(value)}
            )
        else:
            flattened[key] = value
    return flattened


def _write_wide_csv(path: Path, rows: list[dict[str, Any]], empty_fields: tuple[str, ...]) -> None:
    fields = list(empty_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_agimus_robot_state_csv(run: AlignedRun, output: Path) -> None:
    rows = [_flatten_robot_state_row(row) for row in run.data.robot_state]
    _write_wide_csv(
        output / "agimus_robot_state.csv",
        rows,
        ("receive", "stamp", "frame_id", "robot_time"),
    )


def _write_controller_diagnostics_csv(run: AlignedRun, output: Path) -> None:
    fields = (
        "time_sec",
        "receive_sec",
        "stamp_sec",
        "level",
        "name",
        "message",
        "hardware_id",
        "key",
        "value",
        "average",
        "minimum",
        "maximum",
        "stddev",
        "unit",
        "desired",
        "desired_unit",
    )
    with (output / "controller_diagnostics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in run.data.ros_diagnostics:
            values = row["values"] or {"": ""}
            for key, value in values.items():
                parsed = _diagnostic_statistics(value) or {}
                writer.writerow(
                    {
                        "time_sec": row["receive"] - run.data.control.receive[0],
                        "receive_sec": row["receive"],
                        "stamp_sec": row["stamp"],
                        "level": row["level"],
                        "name": row["name"],
                        "message": row["message"],
                        "hardware_id": row["hardware_id"],
                        "key": key,
                        "value": value,
                        **parsed,
                    }
                )


def _write_controller_activity_csv(run: AlignedRun, output: Path) -> None:
    rows = [
        {
            "time_sec": row["receive"] - run.data.control.receive[0],
            "receive_sec": row["receive"],
            "stamp_sec": row["stamp"],
            "kind": row["kind"],
            "name": row["name"],
            "state_id": row["state_id"],
            "state_label": row["state_label"],
        }
        for row in run.data.controller_activity
    ]
    _write_wide_csv(
        output / "controller_activity.csv",
        rows,
        (
            "time_sec",
            "receive_sec",
            "stamp_sec",
            "kind",
            "name",
            "state_id",
            "state_label",
        ),
    )


def _write_gripper_csv(run: AlignedRun, output: Path) -> None:
    rows: list[dict[str, Any]] = []
    t0 = run.data.control.receive[0]
    for source, source_rows in (
        ("joint_state", run.data.gripper_joint_states),
        ("feedback", run.data.gripper_feedback),
        ("status", run.data.gripper_status),
    ):
        for row in source_rows:
            flattened = {
                "source": source,
                "time_sec": row["receive"] - t0,
                **row,
            }
            for key, value in tuple(flattened.items()):
                if isinstance(value, (list, tuple)):
                    flattened[key] = json.dumps(value, separators=(",", ":"))
            rows.append(flattened)
    _write_wide_csv(
        output / "gripper_observability.csv",
        rows,
        ("source", "time_sec", "receive"),
    )


def _write_simulation_csvs(run: AlignedRun, output: Path) -> None:
    t0 = run.data.control.receive[0]
    object_rows = []
    for row in run.data.sim_object_pose:
        pose = row["pose"]
        twist = row["twist"]
        object_rows.append(
            {
                "time_sec": row["receive"] - t0,
                "receive_sec": row["receive"],
                "stamp_sec": row["stamp"],
                "frame_id": row["frame_id"],
                "child_frame_id": row["child_frame_id"],
                **{
                    name: value
                    for name, value in zip(
                        ("x", "y", "z", "qx", "qy", "qz", "qw"), pose
                    )
                },
                **{
                    name: value
                    for name, value in zip(
                        ("vx", "vy", "vz", "wx", "wy", "wz"), twist
                    )
                },
            }
        )
    _write_wide_csv(
        output / "simulation_object_pose.csv",
        object_rows,
        ("time_sec", "receive_sec", "stamp_sec", "frame_id", "child_frame_id"),
    )

    actuator_rows = []
    for row in run.data.sim_actuator_states:
        flattened: dict[str, Any] = {
            "time_sec": row["receive"] - t0,
            "receive_sec": row["receive"],
            "stamp_sec": row["stamp"],
        }
        for field_name in ("position", "velocity", "effort"):
            for name, value in zip(row["names"], row[field_name]):
                flattened[f"{field_name}_{name}"] = value
        actuator_rows.append(flattened)
    _write_wide_csv(
        output / "mujoco_actuator_states.csv",
        actuator_rows,
        ("time_sec", "receive_sec", "stamp_sec"),
    )

    clock_rows = [
        {
            "time_sec": row["receive"] - t0,
            "receive_sec": row["receive"],
            "simulation_time_sec": row["sim_time"],
        }
        for row in run.data.sim_clock
    ]
    _write_wide_csv(
        output / "simulation_clock.csv",
        clock_rows,
        ("time_sec", "receive_sec", "simulation_time_sec"),
    )


def generate_capture_health_report(
    bag_or_run: Path,
    output: Path,
    *,
    full_report_error: str,
) -> dict[str, Any]:
    """Generate a degraded report when the active-control contract is incomplete.

    This deliberately avoids constructing :class:`RunData`: startup, JIT, or
    controller failures often occur before /control exists.  The raw bag topic
    inventory, ROS events, controller lifecycle, planner snapshots, and clock
    remain useful in exactly those cases.
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    output.mkdir(parents=True, exist_ok=True)
    bag = _resolve_bag(bag_or_run)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    available_types = {
        topic.name: topic.type for topic in reader.get_all_topics_and_types()
    }
    decoded_topics = {
        PLANNER_DIAGNOSTICS_TOPIC,
        ROSOUT_TOPIC,
        ROS_DIAGNOSTICS_TOPIC,
        CONTROLLER_ACTIVITY_TOPIC,
        ROS_CLOCK_TOPIC,
    }.intersection(available_types)
    message_types: dict[str, Any] = {}
    decode_errors: list[str] = []
    for topic in sorted(decoded_topics):
        try:
            message_types[topic] = get_message(available_types[topic])
        except Exception as error:
            decode_errors.append(f"{topic}: cannot load {available_types[topic]}: {error}")

    statistics: dict[str, dict[str, Any]] = {
        topic: {
            "topic": topic,
            "type": type_name,
            "count": 0,
            "first_receive_sec": None,
            "last_receive_sec": None,
            "maximum_receive_gap_sec": None,
        }
        for topic, type_name in available_types.items()
    }
    planner_rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    activity_rows: list[dict[str, Any]] = []
    clock_rows: list[dict[str, float]] = []
    reader.set_filter(rosbag2_py.StorageFilter(topics=sorted(available_types)))
    while reader.has_next():
        topic, serialized, receive_ns = reader.read_next()
        receive_sec = receive_ns * 1e-9
        topic_statistics = statistics[topic]
        previous = topic_statistics["last_receive_sec"]
        topic_statistics["count"] += 1
        if topic_statistics["first_receive_sec"] is None:
            topic_statistics["first_receive_sec"] = receive_sec
        topic_statistics["last_receive_sec"] = receive_sec
        if previous is not None:
            gap = receive_sec - previous
            current_maximum = topic_statistics["maximum_receive_gap_sec"]
            topic_statistics["maximum_receive_gap_sec"] = (
                gap if current_maximum is None else max(current_maximum, gap)
            )
        if topic not in message_types:
            continue
        try:
            message = deserialize_message(serialized, message_types[topic])
            if topic == PLANNER_DIAGNOSTICS_TOPIC:
                row = json.loads(message.data)
                row["receive_sec"] = receive_sec
                planner_rows.append(row)
            elif topic == ROSOUT_TOPIC:
                if int(message.level) >= 20:
                    events.append(
                        {
                            "source": "rosout",
                            "receive_sec": receive_sec,
                            "level": int(message.level),
                            "name": str(message.name),
                            "message": str(message.msg),
                        }
                    )
            elif topic == ROS_DIAGNOSTICS_TOPIC:
                for row in _ros_diagnostic_rows(message, receive_sec):
                    if row["level"]:
                        events.append(
                            {
                                "source": "diagnostics",
                                "receive_sec": receive_sec,
                                "level": 20 + 10 * int(row["level"]),
                                "name": row["name"],
                                "message": row["message"],
                            }
                        )
            elif topic == CONTROLLER_ACTIVITY_TOPIC:
                activity_rows.extend(_controller_activity_rows(message, receive_sec))
            elif topic == ROS_CLOCK_TOPIC:
                clock_rows.append(_clock_row(message, receive_sec))
        except Exception as error:
            if len(decode_errors) < 100:
                decode_errors.append(f"{topic} at {receive_sec:.9f}: {error}")

    topic_rows = []
    for row in sorted(statistics.values(), key=lambda item: item["topic"]):
        first = row["first_receive_sec"]
        last = row["last_receive_sec"]
        duration = None if first is None or last is None else max(0.0, last - first)
        count = int(row["count"])
        topic_rows.append(
            {
                **row,
                "duration_sec": duration,
                "effective_rate_hz": (
                    (count - 1) / duration
                    if duration is not None and duration > 0.0 and count > 1
                    else None
                ),
            }
        )

    clock_receive = np.asarray([row["receive"] for row in clock_rows])
    clock_time = np.asarray([row["sim_time"] for row in clock_rows])
    _, clock_rtf = _windowed_clock_rtf(clock_receive, clock_time)
    manifest = _read_manifest(bag)
    summary = {
        "schema": "sbmpc_capture_health_v1",
        "report_kind": "capture_health_only",
        "bag": str(bag),
        "backend": _manifest_backend(bag),
        "recorder_realtime": (
            manifest.get("realtime", {})
            if isinstance(manifest.get("realtime", {}), dict)
            else {}
        ),
        "full_report_error": full_report_error,
        "topic_count": len(topic_rows),
        "populated_topic_count": sum(row["count"] > 0 for row in topic_rows),
        "topics": topic_rows,
        "planner_diagnostic_count": len(planner_rows),
        "last_planner_diagnostic": planner_rows[-1] if planner_rows else None,
        "event_count": len(events),
        "warning_or_error_count": sum(row["level"] >= 30 for row in events),
        "last_controller_activity": activity_rows[-20:],
        "clock": {
            "message_count": len(clock_rows),
            "reset_count": int(np.sum(np.diff(clock_time) < 0.0)),
            "windowed_real_time_factor": _stats(clock_rtf),
        },
        "decode_errors": decode_errors,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_wide_csv(
        output / "capture_topics.csv",
        topic_rows,
        ("topic", "type", "count"),
    )
    _write_wide_csv(
        output / "capture_events.csv",
        events,
        ("source", "receive_sec", "level", "name", "message"),
    )
    _write_wide_csv(
        output / "planner_diagnostics.csv",
        [
            {
                key: (
                    json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            }
            for row in planner_rows
        ],
        ("receive_sec", "state", "planner_step_count"),
    )
    _write_wide_csv(
        output / "controller_activity.csv",
        activity_rows,
        ("receive", "stamp", "kind", "name", "state_id", "state_label"),
    )

    populated = [row for row in topic_rows if row["count"]]
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=False)
    if populated:
        names = [row["topic"] for row in populated]
        axes[0].barh(names, [row["count"] for row in populated])
        axes[0].set_xscale("log")
        axes[0].set_xlabel("message count (log scale)")
        rated = [row for row in populated if row["effective_rate_hz"] is not None]
        axes[1].barh(
            [row["topic"] for row in rated],
            [row["effective_rate_hz"] for row in rated],
        )
        axes[1].set_xscale("log")
        axes[1].set_xlabel("effective receive rate [Hz] (log scale)")
    receive_candidates = [
        row["first_receive_sec"]
        for row in populated
        if row["first_receive_sec"] is not None
    ]
    t0 = min(receive_candidates, default=0.0)
    if planner_rows:
        planner_time = [row["receive_sec"] - t0 for row in planner_rows]
        axes[2].plot(
            planner_time,
            [row.get("planner_step_count", 0) for row in planner_rows],
            label="planner steps",
        )
        axes[2].plot(
            planner_time,
            [row.get("published_control_count", 0) for row in planner_rows],
            label="published controls",
        )
        axes[2].legend()
    axes[2].set_ylabel("cumulative count")
    if events:
        axes[3].scatter(
            [row["receive_sec"] - t0 for row in events],
            [row["level"] for row in events],
            s=10,
            alpha=0.65,
        )
    axes[3].set_ylabel("ROS severity")
    if clock_rows:
        axes[4].plot(clock_receive - t0, clock_time - clock_time[0], label="simulation")
        axes[4].plot(
            clock_receive - t0,
            clock_receive - clock_receive[0],
            "--",
            label="receive wall time",
        )
        axes[4].legend()
    axes[4].set_ylabel("elapsed time [s]")
    axes[4].set_xlabel("bag receive time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("SB-MPC capture health (full control report unavailable)")
    image = _save(fig, output, "00_capture_health.png")
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SB-MPC capture health</title>
<style>body{{font-family:sans-serif;max-width:1500px;margin:auto;padding:1rem}}
img{{max-width:100%;border:1px solid #ddd}}
pre{{white-space:pre-wrap;background:#f5f5f5;padding:1rem}}</style></head>
<body><h1>SB-MPC capture health report</h1>
<p>The complete active-control report could not be constructed:</p>
<pre>{html.escape(full_report_error)}</pre>
<p>The MCAP was retained and the available startup/runtime streams were analyzed.</p>
<p><a href="summary.json">summary.json</a> ·
<a href="capture_topics.csv">capture_topics.csv</a> ·
<a href="capture_events.csv">capture_events.csv</a> ·
<a href="planner_diagnostics.csv">planner_diagnostics.csv</a> ·
<a href="controller_activity.csv">controller_activity.csv</a></p>
<a href="{image}"><img src="{image}"></a>
<h2>Machine-readable summary</h2><pre>{html.escape(json.dumps(summary, indent=2))}</pre>
</body></html>"""
    (output / "index.html").write_text(document)
    return summary


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
<a href="controller_steps.csv">controller_steps.csv</a> ·
<a href="agimus_robot_state.csv">agimus_robot_state.csv</a> ·
<a href="controller_diagnostics.csv">controller_diagnostics.csv</a> ·
<a href="controller_activity.csv">controller_activity.csv</a> ·
<a href="gripper_observability.csv">gripper_observability.csv</a> ·
<a href="simulation_object_pose.csv">simulation_object_pose.csv</a> ·
<a href="mujoco_actuator_states.csv">mujoco_actuator_states.csv</a> ·
<a href="simulation_clock.csv">simulation_clock.csv</a></p>
{image_html}<h2>Machine-readable summary</h2>
<pre>{html.escape(json.dumps(summary, indent=2))}</pre>
</body></html>"""
    (output / "index.html").write_text(document)


def generate_report_from_data(
    data: RunData,
    output: Path,
    *,
    terminal_sec: float = 5.0,
) -> dict[str, Any]:
    """Generate every full report artifact from one decoded MCAP pass."""
    output.mkdir(parents=True, exist_ok=True)
    run = align_run(data)
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
    for optional_plot in (
        _plot_agimus_joint_motor,
        _plot_agimus_ee_wrench,
        _plot_ros_control_timing,
        _plot_gripper_observability,
        _plot_simulation_ground_truth,
        _plot_mujoco_actuators,
    ):
        image = optional_plot(run, output)
        if image is not None:
            images.append(image)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_steps_csv(run, output)
    _write_agimus_robot_state_csv(run, output)
    _write_controller_diagnostics_csv(run, output)
    _write_controller_activity_csv(run, output)
    _write_gripper_csv(run, output)
    _write_simulation_csvs(run, output)
    _write_html(output, summary, images)
    return summary


def generate_report(
    bag_or_run: Path,
    output: Path,
    *,
    terminal_sec: float = 5.0,
) -> dict[str, Any]:
    return generate_report_from_data(
        read_bag(bag_or_run),
        output,
        terminal_sec=terminal_sec,
    )


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
