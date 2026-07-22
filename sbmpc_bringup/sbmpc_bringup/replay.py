from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sbmpc_bringup.constants import (
    BRIDGE_CONTROL_TOPIC,
    BRIDGE_DIAGNOSTICS_TOPIC,
    BRIDGE_SENSOR_TOPIC,
    FER_ARM_JOINT_NAMES,
    LFC_OUTPUT_JOINT_EFFORT_TOPIC,
    SBMPC_JOINT_STATES_TOPIC,
)


@dataclass(frozen=True, slots=True)
class ReplayJointEffortCommand:
    stamp_sec: float
    receive_wall_sec: float | None
    position: list[float]
    velocity: list[float]
    effort: list[float]


@dataclass(frozen=True, slots=True)
class ReplayState:
    stamp_sec: float
    receive_wall_sec: float | None
    position: list[float]
    velocity: list[float]
    # Joint effort with source/backend semantics carried by the replay payload.
    effort: list[float] = field(default_factory=list)


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _series_replay_states(series: Any | None) -> list[ReplayState]:
    """Convert a decoded run-report joint series into replay states."""
    if series is None:
        return []
    count = len(series.receive)
    if not all(
        len(values) == count
        for values in (series.stamp, series.q, series.v, series.effort)
    ):
        raise ValueError("decoded joint series fields have inconsistent lengths")
    return [
        ReplayState(
            stamp_sec=float(series.stamp[index]),
            receive_wall_sec=float(series.receive[index]),
            position=np.asarray(series.q[index], dtype=np.float64).tolist(),
            velocity=np.asarray(series.v[index], dtype=np.float64).tolist(),
            effort=np.asarray(series.effort[index], dtype=np.float64).tolist(),
        )
        for index in range(count)
    ]


def _series_lfc_output(series: Any) -> list[ReplayJointEffortCommand]:
    count = len(series.receive)
    if not all(
        len(values) == count
        for values in (series.stamp, series.q, series.v, series.effort)
    ):
        raise ValueError("decoded LFC output fields have inconsistent lengths")
    return [
        ReplayJointEffortCommand(
            stamp_sec=float(series.stamp[index]),
            receive_wall_sec=float(series.receive[index]),
            position=np.asarray(series.q[index], dtype=np.float64).tolist(),
            velocity=np.asarray(series.v[index], dtype=np.float64).tolist(),
            effort=np.asarray(series.effort[index], dtype=np.float64).tolist(),
        )
        for index in range(count)
    ]


def _control_records_from_series(control: Any) -> list[dict[str, object]]:
    count = len(control.receive)
    if not all(
        len(values) == count
        for values in (
            control.stamp,
            control.anchor_stamp,
            control.feedforward,
            control.gain,
            control.anchor_q,
            control.anchor_v,
        )
    ):
        raise ValueError("decoded control series fields have inconsistent lengths")

    records = []
    for index in range(count):
        feedforward = np.asarray(control.feedforward[index], dtype=np.float64)
        gain = np.asarray(control.gain[index], dtype=np.float64)
        anchor_q = np.asarray(control.anchor_q[index], dtype=np.float64)
        anchor_v = np.asarray(control.anchor_v[index], dtype=np.float64)
        # The ROS adapter publishes a seven-element feedforward command as a
        # (7, 1) Float64MultiArray. The decoded report intentionally stores the
        # numerical vector only, so restore that wire shape during export.
        feedforward_shape = [7, 1] if feedforward.size == 7 else list(feedforward.shape)
        records.append(
            {
                "stamp_sec": float(control.stamp[index]),
                "receive_wall_sec": float(control.receive[index]),
                "feedforward": feedforward.reshape(-1).tolist(),
                "feedforward_shape": feedforward_shape,
                "feedforward_max_abs": (
                    float(np.max(np.abs(feedforward), initial=0.0))
                    if feedforward.size
                    else 0.0
                ),
                "feedback_gain": gain.reshape(-1).tolist(),
                "feedback_gain_shape": list(gain.shape),
                "gain_norm": float(np.linalg.norm(gain)) if gain.size else 0.0,
                "initial_state": asdict(
                    ReplayState(
                        stamp_sec=float(control.anchor_stamp[index]),
                        receive_wall_sec=None,
                        position=anchor_q.tolist(),
                        velocity=anchor_v.tolist(),
                    )
                ),
                "initial_state_velocity_abs_max": (
                    float(np.max(np.abs(anchor_v), initial=0.0))
                    if anchor_v.size
                    else None
                ),
            }
        )
    return records


def _effort_records(series: Any) -> list[dict[str, object]]:
    return [
        {
            "stamp_sec": float(series.stamp[index]),
            "receive_wall_sec": float(series.receive[index]),
            "effort": np.asarray(series.effort[index], dtype=np.float64).tolist(),
        }
        for index in range(len(series.receive))
    ]


def _recorded_wall_time_sec(*streams: Any) -> float:
    receive_times = [
        float(value)
        for stream in streams
        for value in getattr(stream, "receive", [])
        if np.isfinite(float(value))
    ]
    if len(receive_times) < 2:
        return 0.0
    return max(0.0, max(receive_times) - min(receive_times))


def replay_payload_from_run_data(data: Any) -> dict[str, object]:
    """Build an ``sbmpc_ros_replay_v2`` payload from decoded MCAP data.

    ``data`` follows the structural interface of :class:`run_report.RunData`.
    Keeping this converter independent of ``run_report`` avoids importing ROS
    bag and plotting dependencies when the replay viewer starts.
    """
    joint_states = _series_replay_states(getattr(data, "merged", None))
    sensor_states = _series_replay_states(data.sensor)
    controls = _control_records_from_series(data.control)
    diagnostics = list(data.diagnostics.rows)
    lfc_output_efforts = _series_lfc_output(data.output)
    observed_joint_effort = _effort_records(data.hardware)
    effort_source = getattr(data, "hardware_source", "/franka/joint_states")
    backend = getattr(
        data,
        "backend",
        "real" if effort_source == "/franka/joint_states" else "mujoco",
    )
    effort_semantics = (
        "Franka FCI measured total link-side joint torque tau_J"
        if backend == "real"
        else "MuJoCo ros2_control actuator effort; not Franka FCI measured torque"
    )
    recorded_wall_time_sec = _recorded_wall_time_sec(
        getattr(data, "merged", None),
        data.sensor,
        data.control,
        data.output,
        data.hardware,
        data.diagnostics,
    )
    return {
        "schema": "sbmpc_ros_replay_v2",
        "backend": backend,
        "recorded_wall_time_sec": recorded_wall_time_sec,
        "joint_names": list(FER_ARM_JOINT_NAMES),
        "topics": {
            "joint_states": (
                SBMPC_JOINT_STATES_TOPIC
                if getattr(data, "merged", None) is not None
                else None
            ),
            "sensor": BRIDGE_SENSOR_TOPIC,
            "control": BRIDGE_CONTROL_TOPIC,
            "diagnostics": BRIDGE_DIAGNOSTICS_TOPIC,
            "lfc_output_effort": LFC_OUTPUT_JOINT_EFFORT_TOPIC,
            "observed_joint_effort": effort_source,
        },
        "joint_states": [asdict(state) for state in joint_states],
        "sensor_states": [asdict(state) for state in sensor_states],
        "controls": controls,
        "diagnostics": diagnostics,
        "lfc_output_efforts": [
            asdict(command) for command in lfc_output_efforts
        ],
        "observed_joint_effort_semantics": effort_semantics,
        "observed_joint_effort": observed_joint_effort,
        "summary": summarize_payload(
            joint_states,
            sensor_states,
            controls,
            diagnostics,
            lfc_output_efforts,
        ),
    }


def export_replay_json(data: Any, output_path: Path) -> dict[str, object]:
    """Atomically export decoded MCAP data in the existing replay format."""
    payload = replay_payload_from_run_data(data)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temp_path, output_path)
    return payload


def _resolve_default_model_xml() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("sbmpc_bringup"))
        return str(share / "mujoco" / "fer_pick_place_ros2_control_scene.xml")
    except Exception:
        return (
            "/workspace/sbmpc_ros/sbmpc_bringup/mujoco/"
            "fer_pick_place_ros2_control_scene.xml"
        )


def _import_mujoco():
    try:
        import mujoco
    except Exception as exc:
        raise RuntimeError(
            "MuJoCo is not importable in this Python environment. Run replay "
            "through /workspace/sbmpc_containers/scripts/uv_ros_run.sh."
        ) from exc
    if not hasattr(mujoco, "MjModel"):
        raise RuntimeError(
            "The imported 'mujoco' module is not the official Python MuJoCo "
            "package used by the replay viewer. Run replay through "
            "/workspace/sbmpc_containers/scripts/uv_ros_run.sh."
        )
    return mujoco


def _relative_states(
    states: list[dict[str, object]],
    *,
    payload: dict[str, object],
    time_source: str,
) -> list[dict[str, object]]:
    if not states:
        return []
    times = _relative_times(states, payload=payload, time_source=time_source)
    result = []
    for row, row_time in zip(states, times, strict=True):
        result.append(
            {
                **row,
                "t": max(0.0, float(row_time)),
            }
        )
    return result


def states_from_payload(
    payload: dict[str, object],
    *,
    source: str,
    time_source: str = "auto",
) -> list[dict[str, object]]:
    if source == "joint_states":
        states = list(payload.get("joint_states", []))
    elif source == "sensor":
        states = list(payload.get("sensor_states", []))
    elif source == "auto":
        states = list(payload.get("joint_states", []))
        if not states:
            states = list(payload.get("sensor_states", []))
    else:
        raise ValueError(f"unsupported state source: {source!r}")
    return _relative_states(states, payload=payload, time_source=time_source)


def summarize_payload(
    joint_states: list[ReplayState],
    sensor_states: list[ReplayState],
    controls: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    lfc_output_efforts: list[ReplayJointEffortCommand] | None = None,
) -> dict[str, object]:
    states = joint_states if joint_states else sensor_states
    if states:
        q = np.asarray([state.position for state in states], dtype=np.float64)
        v = np.asarray([state.velocity for state in states], dtype=np.float64)
    else:
        q = np.zeros((0, len(FER_ARM_JOINT_NAMES)), dtype=np.float64)
        v = np.zeros_like(q)

    running = [row for row in diagnostics if row.get("state") == "running"]
    active_running = _active_running_rows(running)
    timing_rows = active_running if active_running else running
    planning = [
        _finite_float(row.get("last_planning_time_ms"))
        for row in timing_rows
    ]
    planning = [value for value in planning if value is not None]
    lfc_output_efforts = lfc_output_efforts or []
    lfc_effort_abs_max = _state_vector_abs_max(lfc_output_efforts, "effort")
    lfc_velocity_abs_max = _state_vector_abs_max(lfc_output_efforts, "velocity")
    return {
        "joint_state_count": len(joint_states),
        "sensor_state_count": len(sensor_states),
        "control_count": len(controls),
        "diagnostics_count": len(diagnostics),
        "running_diagnostics_count": len(running),
        "active_running_diagnostics_count": len(active_running),
        "duration_sec": _state_duration_sec(states, controls),
        "max_tail_joint_span": (
            float(np.max(np.ptp(q[len(q) // 2 :], axis=0))) if len(q) >= 2 else None
        ),
        "joint_velocity_abs_max": (
            float(np.max(np.abs(v), initial=0.0)) if len(v) else None
        ),
        "control_feedforward_abs_max": _row_max(controls, "feedforward_max_abs"),
        "control_gain_norm_max": _row_max(controls, "gain_norm"),
        "control_initial_state_velocity_abs_max": _row_max(
            controls,
            "initial_state_velocity_abs_max",
        ),
        "lfc_output_effort_count": len(lfc_output_efforts),
        "lfc_output_effort_abs_max": lfc_effort_abs_max,
        "lfc_output_velocity_abs_max": lfc_velocity_abs_max,
        "planning_ms_mean": float(np.mean(planning)) if planning else None,
        "planning_ms_max": float(np.max(planning)) if planning else None,
        "timing_ms": {
            "planning": _diagnostic_stats(
                timing_rows,
                "last_planning_time_ms",
            ),
            "planner_step_wall": _diagnostic_stats(
                timing_rows,
                "last_planner_step_wall_time_ms",
            ),
            "planner_prepare": _diagnostic_stats(
                timing_rows,
                "last_planner_prepare_time_ms",
            ),
            "planner_command": _diagnostic_stats(
                timing_rows,
                "last_planner_command_time_ms",
            ),
            "control_prepare": _diagnostic_stats(
                timing_rows,
                "last_control_prepare_time_ms",
            ),
            "control_publish": _diagnostic_stats(
                timing_rows,
                "last_control_publish_time_ms",
            ),
        },
        "control_cadence_sec": _control_cadence_summary(controls),
        "deadline_miss_count": _counter_delta(timing_rows, "deadline_miss_count"),
        "accepted_planner_output_count": _counter_delta(
            timing_rows,
            "accepted_planner_output_count",
        ),
        "rejected_planner_output_count": _counter_delta(
            timing_rows,
            "rejected_planner_output_count",
        ),
    }



def _state_vector_abs_max(states, field_name: str) -> float | None:
    values: list[float] = []
    for state in states:
        vector = getattr(state, field_name, None)
        if vector is None and isinstance(state, dict):
            vector = state.get(field_name)
        if vector is not None:
            values.extend(abs(float(value)) for value in vector)
    return float(np.max(values, initial=0.0)) if values else None


def _row_max(rows: list[dict[str, object]], key: str) -> float | None:
    values = [_finite_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return float(np.max(values, initial=0.0)) if values else None


def _row_values(rows: list[dict[str, object]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _span(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return max(0.0, float(values[-1] - values[0]))


def _relative_from_values(values: list[float], count: int) -> list[float] | None:
    if len(values) != count or _span(values) <= 1e-9:
        return None
    first = values[0]
    return [max(0.0, float(value - first)) for value in values]


def _scaled_times(total_sec: float, count: int) -> list[float] | None:
    if count <= 0 or total_sec <= 1e-9:
        return None
    if count == 1:
        return [0.0]
    return np.linspace(0.0, float(total_sec), count, dtype=np.float64).tolist()


def _control_span_sec(payload: dict[str, object]) -> float:
    controls = list(payload.get("controls", []))
    for key in ("receive_wall_sec", "stamp_sec"):
        span = _span(_row_values(controls, key))
        if span > 1e-9:
            return span
    return 0.0


def _relative_times(
    states: list[dict[str, object]],
    *,
    payload: dict[str, object],
    time_source: str,
) -> list[float]:
    if time_source == "receive":
        times = _relative_from_values(_row_values(states, "receive_wall_sec"), len(states))
    elif time_source == "stamp":
        times = _relative_from_values(_row_values(states, "stamp_sec"), len(states))
    elif time_source == "control":
        times = _scaled_times(_control_span_sec(payload), len(states))
    elif time_source == "index":
        recorded = float(payload.get("recorded_wall_time_sec") or 0.0)
        times = _scaled_times(recorded, len(states))
        if times is None:
            times = [0.001 * index for index in range(len(states))]
    elif time_source == "auto":
        # Replay what the recorder observed in wall time when possible. ROS/MuJoCo
        # header stamps can be zero or compressed during reset/warmup.
        times = _relative_from_values(_row_values(states, "receive_wall_sec"), len(states))
        if times is None:
            times = _relative_from_values(_row_values(states, "stamp_sec"), len(states))
        if times is None:
            times = _scaled_times(_control_span_sec(payload), len(states))
        if times is None:
            recorded = float(payload.get("recorded_wall_time_sec") or 0.0)
            times = _scaled_times(recorded, len(states))
        if times is None:
            times = [0.001 * index for index in range(len(states))]
    else:
        raise ValueError(f"unsupported time source: {time_source!r}")

    if times is None:
        raise ValueError(
            f"time source {time_source!r} is not available in this replay file."
        )
    return times


def _state_duration_sec(
    states: list[ReplayState],
    controls: list[dict[str, object]],
) -> float:
    if len(states) < 2:
        return 0.0
    receive_values = [
        state.receive_wall_sec
        for state in states
        if state.receive_wall_sec is not None
    ]
    if len(receive_values) >= 2:
        return max(0.0, float(receive_values[-1] - receive_values[0]))
    stamp_span = float(states[-1].stamp_sec - states[0].stamp_sec)
    if stamp_span > 1e-9:
        return stamp_span
    control_receive_span = _span(_row_values(controls, "receive_wall_sec"))
    if control_receive_span > 1e-9:
        return control_receive_span
    control_stamp_span = _span(_row_values(controls, "stamp_sec"))
    if control_stamp_span > 1e-9:
        return control_stamp_span
    return 0.0


def _active_running_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if int(row.get("accepted_planner_output_count") or 0) > 0
        or int(row.get("published_control_count") or 0) > 0
    ]


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _diagnostic_stats(rows: list[dict[str, object]], key: str) -> dict[str, float | None]:
    values = [_finite_float(row.get(key)) for row in rows]
    return _stats([value for value in values if value is not None])


def _deltas(values: list[float]) -> list[float]:
    if len(values) < 2:
        return []
    array = np.asarray(values, dtype=np.float64)
    return np.diff(array).tolist()


def _control_cadence_summary(controls: list[dict[str, object]]) -> dict[str, object]:
    header_stamps = [
        float(row["stamp_sec"])
        for row in controls
        if row.get("stamp_sec") is not None
    ]
    receive_stamps = [
        float(row["receive_wall_sec"])
        for row in controls
        if row.get("receive_wall_sec") is not None
    ]
    return {
        "header_delta": _stats(_deltas(header_stamps)),
        "receive_delta": _stats(_deltas(receive_stamps)),
    }


def _last_int(rows: list[dict[str, object]], key: str) -> int | None:
    for row in reversed(rows):
        value = row.get(key)
        if value is not None:
            return int(value)
    return None


def _first_int(rows: list[dict[str, object]], key: str) -> int | None:
    for row in rows:
        value = row.get(key)
        if value is not None:
            return int(value)
    return None


def _counter_delta(rows: list[dict[str, object]], key: str) -> int | None:
    first = _first_int(rows, key)
    last = _last_int(rows, key)
    if first is None or last is None:
        return None
    return max(0, last - first)


def _load_payload(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") not in {
        "sbmpc_ros_replay_v1",
        "sbmpc_ros_replay_v2",
    }:
        raise ValueError(f"{path} is not an sbmpc ROS replay file.")
    return payload


def _print_replay_summary(
    path: Path,
    payload: dict[str, object],
    states,
    *,
    time_source: str,
) -> None:
    summary = payload.get("summary", {})
    replay_duration = float(states[-1]["t"]) if states else 0.0
    print(f"replay file: {path}")
    print(f"states selected: {len(states)}")
    print(
        "summary: "
        f"duration_sec={summary.get('duration_sec')} "
        f"replay_duration_sec={replay_duration} "
        f"time_source={time_source} "
        f"planning_max_ms={summary.get('planning_ms_max')}"
    )


def replay_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Replay a recorded SB-MPC ROS trajectory in a MuJoCo viewer."
    )
    parser.add_argument("replay_json", type=Path)
    parser.add_argument("--model-xml", default=_resolve_default_model_xml())
    parser.add_argument(
        "--state-source",
        choices=("auto", "joint_states", "sensor"),
        default="auto",
    )
    parser.add_argument(
        "--time-source",
        choices=("auto", "receive", "stamp", "control", "index"),
        default="auto",
        help=(
            "Clock used for replay pacing. auto prefers recorder receive-wall "
            "time, then message stamps, then control cadence for older files."
        ),
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the replay file and model mapping without opening the viewer.",
    )
    args, _ros_args = parser.parse_known_args(argv)
    if args.speed <= 0.0:
        raise ValueError("--speed must be positive.")

    payload = _load_payload(args.replay_json)
    states = states_from_payload(
        payload,
        source=args.state_source,
        time_source=args.time_source,
    )
    if not states:
        raise ValueError("replay file does not contain any selected state samples.")
    _print_replay_summary(
        args.replay_json,
        payload,
        states,
        time_source=args.time_source,
    )

    mujoco = _import_mujoco()

    model = mujoco.MjModel.from_xml_path(args.model_xml)
    data = mujoco.MjData(model)
    joint_qpos_addr = []
    joint_qvel_addr = []
    for name in FER_ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"model does not contain joint {name!r}")
        joint_qpos_addr.append(int(model.jnt_qposadr[joint_id]))
        joint_qvel_addr.append(int(model.jnt_dofadr[joint_id]))

    if args.dry_run:
        print(f"model: {args.model_xml}")
        print(f"mapped_joints: {', '.join(FER_ARM_JOINT_NAMES)}")
        return

    import mujoco.viewer

    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    def apply_state(row: dict[str, object]) -> None:
        q = np.asarray(row["position"], dtype=np.float64)
        v = np.asarray(row["velocity"], dtype=np.float64)
        data.qpos[joint_qpos_addr] = q
        data.qvel[joint_qvel_addr] = v
        mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if home_key >= 0:
                mujoco.mj_resetDataKeyframe(model, data, home_key)
            for index, row in enumerate(states):
                if not viewer.is_running():
                    break
                start = time.perf_counter()
                apply_state(row)
                viewer.sync()
                if index + 1 < len(states):
                    dt = max(0.0, float(states[index + 1]["t"]) - float(row["t"]))
                    sleep_sec = dt / args.speed - (time.perf_counter() - start)
                    if sleep_sec > 0.0:
                        time.sleep(sleep_sec)
            if not args.loop:
                break


if __name__ == "__main__":
    replay_main()
