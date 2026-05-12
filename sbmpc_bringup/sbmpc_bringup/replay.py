from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import signal
import time
from typing import Any

import numpy as np

from sbmpc_bringup.constants import (
    BRIDGE_CONTROL_TOPIC,
    BRIDGE_DIAGNOSTICS_TOPIC,
    BRIDGE_SENSOR_TOPIC,
    FER_ARM_JOINT_NAMES,
    SBMPC_JOINT_STATES_TOPIC,
)


@dataclass(frozen=True, slots=True)
class ReplayState:
    stamp_sec: float
    receive_wall_sec: float | None
    position: list[float]
    velocity: list[float]


def _stamp_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def _joint_indices(names: list[str] | tuple[str, ...]) -> tuple[int, ...] | None:
    index_by_name = {name: index for index, name in enumerate(names)}
    if not all(name in index_by_name for name in FER_ARM_JOINT_NAMES):
        return None
    return tuple(index_by_name[name] for name in FER_ARM_JOINT_NAMES)


def _vector_from_indices(values, indices: tuple[int, ...]) -> list[float]:
    return [float(values[index]) for index in indices]


def _record_state_from_joint_state(
    message,
    *,
    receive_wall_sec: float | None = None,
) -> ReplayState | None:
    indices = _joint_indices(tuple(message.name))
    if indices is None:
        return None
    position = _vector_from_indices(message.position, indices)
    if len(message.velocity) >= len(message.name):
        velocity = _vector_from_indices(message.velocity, indices)
    else:
        velocity = [0.0] * len(indices)
    return ReplayState(
        stamp_sec=_stamp_sec(message.header.stamp),
        receive_wall_sec=receive_wall_sec,
        position=position,
        velocity=velocity,
    )


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _resolve_default_model_xml() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("sbmpc_bringup"))
        return str(share / "mujoco" / "panda_pick_place_ros2_control_scene.xml")
    except Exception:
        return (
            "/workspace/sbmpc_ros/sbmpc_bringup/mujoco/"
            "panda_pick_place_ros2_control_scene.xml"
        )


def _import_mujoco():
    try:
        import mujoco
    except Exception as exc:
        raise RuntimeError(
            "MuJoCo is not importable in this Python environment. Run replay "
            "through /workspace/sbmpc_containers/scripts/pixi_ros_run.sh."
        ) from exc
    if not hasattr(mujoco, "MjModel"):
        raise RuntimeError(
            "The imported 'mujoco' module is not the official Python MuJoCo "
            "package used by the replay viewer. Run replay through "
            "/workspace/sbmpc_containers/scripts/pixi_ros_run.sh."
        )
    return mujoco


class ReplayRecorder:
    def __init__(
        self,
        *,
        duration_sec: float,
        output_path: Path,
        joint_states_topic: str,
        sensor_topic: str,
        control_topic: str,
        diagnostics_topic: str,
        start_after_first_control: bool = True,
        startup_timeout_sec: float = 120.0,
        autosave_period_sec: float = 5.0,
    ) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        from linear_feedback_controller_msgs.msg import Control, Sensor

        class RecorderNode(Node):
            def __init__(self, outer: ReplayRecorder) -> None:
                super().__init__("sbmpc_replay_recorder")
                qos = QoSProfile(
                    history=HistoryPolicy.KEEP_LAST,
                    depth=10,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                )
                self.outer = outer
                self.create_subscription(
                    JointState,
                    joint_states_topic,
                    self._on_joint_state,
                    10,
                )
                self.create_subscription(Sensor, sensor_topic, self._on_sensor, qos)
                self.create_subscription(Control, control_topic, self._on_control, qos)
                self.create_subscription(String, diagnostics_topic, self._on_diagnostics, 10)

            def _on_joint_state(self, message) -> None:
                state = _record_state_from_joint_state(
                    message,
                    receive_wall_sec=time.monotonic(),
                )
                if state is not None:
                    self.outer.joint_states.append(state)

            def _on_sensor(self, message) -> None:
                state = _record_state_from_joint_state(
                    message.joint_state,
                    receive_wall_sec=time.monotonic(),
                )
                if state is not None:
                    self.outer.sensor_states.append(state)

            def _on_control(self, message) -> None:
                feedforward = np.asarray(message.feedforward.data, dtype=np.float64)
                gain = np.asarray(message.feedback_gain.data, dtype=np.float64)
                self.outer.controls.append(
                    {
                        "stamp_sec": _stamp_sec(message.header.stamp),
                        "receive_wall_sec": time.monotonic(),
                        "feedforward_max_abs": (
                            float(np.max(np.abs(feedforward), initial=0.0))
                            if feedforward.size
                            else 0.0
                        ),
                        "gain_norm": float(np.linalg.norm(gain)) if gain.size else 0.0,
                    }
                )

            def _on_diagnostics(self, message) -> None:
                try:
                    self.outer.diagnostics.append(json.loads(message.data))
                except json.JSONDecodeError:
                    return

        self._rclpy = rclpy
        self._node_class = RecorderNode
        self.duration_sec = float(duration_sec)
        self.recorded_wall_time_sec = 0.0
        self.output_path = output_path
        self.start_after_first_control = bool(start_after_first_control)
        self.startup_timeout_sec = float(startup_timeout_sec)
        self.autosave_period_sec = float(autosave_period_sec)
        self._stop_requested = False
        self._record_start_wall = 0.0
        self._last_save_wall = 0.0
        self.joint_states: list[ReplayState] = []
        self.sensor_states: list[ReplayState] = []
        self.controls: list[dict[str, object]] = []
        self.diagnostics: list[dict[str, object]] = []
        self.topics = {
            "joint_states": joint_states_topic,
            "sensor": sensor_topic,
            "control": control_topic,
            "diagnostics": diagnostics_topic,
        }

    def run(self) -> dict[str, object]:
        self._rclpy.init()
        node = self._node_class(self)
        previous_handlers = self._install_signal_handlers()
        record_start = time.monotonic()
        try:
            self._write_payload(self.payload())
            if self.start_after_first_control:
                print(
                    "record_sbmpc_replay: waiting for first /control message "
                    f"before recording -> {self.output_path}",
                    flush=True,
                )
                startup_deadline = time.monotonic() + self.startup_timeout_sec
                while (
                    self._rclpy.ok()
                    and not self._stop_requested
                    and not self.controls
                ):
                    if time.monotonic() >= startup_deadline:
                        raise TimeoutError(
                            "timed out waiting for the first control message; "
                            "use --include-warmup to record immediately."
                        )
                    self._rclpy.spin_once(node, timeout_sec=0.02)
                self._clear_samples()
                self._write_payload(self.payload())

            record_start = time.monotonic()
            print(
                "record_sbmpc_replay: recording "
                + (
                    f"for {self.duration_sec:.3f}s"
                    if self.duration_sec > 0.0
                    else "until ROS shutdown"
                )
                + f" -> {self.output_path}",
                flush=True,
            )
            deadline = (
                record_start + self.duration_sec if self.duration_sec > 0.0 else None
            )
            self._record_start_wall = record_start
            while (
                self._rclpy.ok()
                and not self._stop_requested
                and (deadline is None or time.monotonic() < deadline)
            ):
                self._rclpy.spin_once(node, timeout_sec=0.02)
                self._autosave_if_due()
        except KeyboardInterrupt:
            pass
        finally:
            self.recorded_wall_time_sec = max(0.0, time.monotonic() - record_start)
            self._restore_signal_handlers(previous_handlers)
            node.destroy_node()
            self._rclpy.shutdown()

        payload = self.payload()
        self._write_payload(payload)
        return payload

    def _install_signal_handlers(self):
        previous_handlers = {}

        def request_stop(signum, frame):  # noqa: ARG001
            self._stop_requested = True

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
        return previous_handlers

    def _restore_signal_handlers(self, previous_handlers) -> None:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    def _write_payload(self, payload: dict[str, object]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.output_path.with_name(
            f".{self.output_path.name}.{os.getpid()}.tmp"
        )
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temp_path, self.output_path)
        self._last_save_wall = time.monotonic()

    def _autosave_if_due(self) -> None:
        if self.autosave_period_sec <= 0.0:
            return
        if (time.monotonic() - self._last_save_wall) < self.autosave_period_sec:
            return
        self.recorded_wall_time_sec = max(
            self.recorded_wall_time_sec,
            time.monotonic() - self._record_start_wall,
        )
        self._write_payload(self.payload())

    def _clear_samples(self) -> None:
        self.joint_states.clear()
        self.sensor_states.clear()
        self.controls.clear()
        self.diagnostics.clear()

    def payload(self) -> dict[str, object]:
        return {
            "schema": "sbmpc_ros_replay_v1",
            "recorded_wall_time_sec": self.recorded_wall_time_sec,
            "joint_names": list(FER_ARM_JOINT_NAMES),
            "topics": self.topics,
            "joint_states": [asdict(state) for state in self.joint_states],
            "sensor_states": [asdict(state) for state in self.sensor_states],
            "controls": self.controls,
            "diagnostics": self.diagnostics,
            "summary": summarize_payload(
                self.joint_states,
                self.sensor_states,
                self.controls,
                self.diagnostics,
            ),
        }


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
    foreground = [
        _finite_float(row.get("last_foreground_planning_time_ms"))
        for row in timing_rows
    ]
    foreground = [value for value in foreground if value is not None]
    gain_age = [_finite_float(row.get("last_gain_age_cycles")) for row in timing_rows]
    gain_age = [value for value in gain_age if value is not None]
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
        "foreground_ms_mean": float(np.mean(foreground)) if foreground else None,
        "foreground_ms_max": float(np.max(foreground)) if foreground else None,
        "timing_ms": {
            "foreground": _diagnostic_stats(
                timing_rows,
                "last_foreground_planning_time_ms",
            ),
            "bridge_loop": _diagnostic_stats(timing_rows, "last_bridge_loop_time_ms"),
            "planner_step_wall": _diagnostic_stats(
                timing_rows,
                "last_planner_step_wall_time_ms",
            ),
            "planner_step_overhead": _diagnostic_stats(
                timing_rows,
                "last_planner_step_overhead_time_ms",
            ),
            "planner_api_wall": _diagnostic_stats(
                timing_rows,
                "last_planner_api_wall_time_ms",
            ),
            "planner_bridge_adapter_overhead": _diagnostic_stats(
                timing_rows,
                "last_planner_bridge_adapter_overhead_time_ms",
            ),
            "planner_prepare": _diagnostic_stats(
                timing_rows,
                "last_planner_prepare_time_ms",
            ),
            "planner_command": _diagnostic_stats(
                timing_rows,
                "last_planner_command_time_ms",
            ),
            "planner_tau_extract": _diagnostic_stats(
                timing_rows,
                "last_planner_tau_extract_time_ms",
            ),
            "planner_gain_fetch": _diagnostic_stats(
                timing_rows,
                "last_planner_gain_fetch_time_ms",
            ),
            "planner_task_diagnostics": _diagnostic_stats(
                timing_rows,
                "last_planner_task_diagnostics_time_ms",
            ),
            "planner_output_build": _diagnostic_stats(
                timing_rows,
                "last_planner_output_build_time_ms",
            ),
            "planner_loop_residual": _diagnostic_stats(
                timing_rows,
                "last_planner_loop_residual_time_ms",
            ),
            "background_gain": _diagnostic_stats(
                timing_rows,
                "last_background_gain_time_ms",
            ),
            "background_gain_wall": _diagnostic_stats(
                timing_rows,
                "last_background_gain_wall_time_ms",
            ),
            "gain_subset_select": _diagnostic_stats(
                timing_rows,
                "last_gain_subset_select_time_ms",
            ),
            "gain_snapshot_pack": _diagnostic_stats(
                timing_rows,
                "last_gain_snapshot_pack_time_ms",
            ),
            "gain_gradient": _diagnostic_stats(
                timing_rows,
                "last_gain_gradient_time_ms",
            ),
            "gain_synthesis": _diagnostic_stats(
                timing_rows,
                "last_gain_synthesis_time_ms",
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
        "gain_age_cycles_max": float(np.max(gain_age)) if gain_age else None,
        "final_gain_window_fill": _last_int(timing_rows, "last_gain_window_fill"),
        "final_completed_gain_batches": _last_int(
            timing_rows,
            "last_gain_completed_batch_count",
        ),
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


def _print_record_summary(payload: dict[str, object], output_path: Path) -> None:
    summary = payload["summary"]
    print(f"wrote replay -> {output_path}")
    print(
        "samples: "
        f"joint_states={summary['joint_state_count']} "
        f"sensor_states={summary['sensor_state_count']} "
        f"controls={summary['control_count']} "
        f"diagnostics={summary['diagnostics_count']} "
        f"active_diagnostics={summary['active_running_diagnostics_count']}"
    )
    print(
        "controller: "
        f"foreground_max_ms={summary['foreground_ms_max']} "
        f"gain_age_max={summary['gain_age_cycles_max']} "
        f"window_fill={summary['final_gain_window_fill']} "
        f"completed_gain_batches={summary['final_completed_gain_batches']}"
    )
    timing = summary["timing_ms"]
    cadence = summary["control_cadence_sec"]["receive_delta"]
    print(
        "full_stack_timing: "
        f"bridge_p99_ms={timing['bridge_loop']['p99']} "
        f"planner_wall_p99_ms={timing['planner_step_wall']['p99']} "
        f"planner_api_p99_ms={timing['planner_api_wall']['p99']} "
        f"adapter_overhead_p99_ms={timing['planner_bridge_adapter_overhead']['p99']} "
        f"background_gain_p99_ms={timing['background_gain']['p99']} "
        f"background_gain_wall_p99_ms={timing['background_gain_wall']['p99']} "
        f"planner_residual_p99_ms={timing['planner_loop_residual']['p99']} "
        f"control_receive_p99_sec={cadence['p99']} "
        f"deadline_misses={summary['deadline_miss_count']}"
    )


def record_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Record a headless ROS/MuJoCo SB-MPC run for offline replay."
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=8.0,
        help="Recording duration. Use 0 to record until ROS shutdown.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-warmup",
        action="store_true",
        help="Record immediately instead of waiting for the first /control message.",
    )
    parser.add_argument(
        "--startup-timeout-sec",
        type=float,
        default=120.0,
        help="Maximum time to wait for the first /control message.",
    )
    parser.add_argument(
        "--autosave-period-sec",
        type=float,
        default=5.0,
        help="Periodically rewrite the replay file so launch shutdown cannot lose it.",
    )
    parser.add_argument("--joint-states-topic", default=SBMPC_JOINT_STATES_TOPIC)
    parser.add_argument("--sensor-topic", default=BRIDGE_SENSOR_TOPIC)
    parser.add_argument("--control-topic", default=BRIDGE_CONTROL_TOPIC)
    parser.add_argument("--diagnostics-topic", default=BRIDGE_DIAGNOSTICS_TOPIC)
    args, _ros_args = parser.parse_known_args(argv)
    if args.duration_sec < 0.0:
        raise ValueError("--duration-sec must be non-negative.")
    if args.startup_timeout_sec <= 0.0:
        raise ValueError("--startup-timeout-sec must be positive.")
    if args.autosave_period_sec < 0.0:
        raise ValueError("--autosave-period-sec must be non-negative.")

    recorder = ReplayRecorder(
        duration_sec=args.duration_sec,
        output_path=args.output,
        joint_states_topic=args.joint_states_topic,
        sensor_topic=args.sensor_topic,
        control_topic=args.control_topic,
        diagnostics_topic=args.diagnostics_topic,
        start_after_first_control=not args.include_warmup,
        startup_timeout_sec=args.startup_timeout_sec,
        autosave_period_sec=args.autosave_period_sec,
    )
    payload = recorder.run()
    _print_record_summary(payload, args.output)


def _load_payload(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "sbmpc_ros_replay_v1":
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
        f"foreground_max_ms={summary.get('foreground_ms_max')} "
        f"gain_age_max={summary.get('gain_age_cycles_max')} "
        f"window_fill={summary.get('final_gain_window_fill')} "
        f"completed_gain_batches={summary.get('final_completed_gain_batches')}"
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
