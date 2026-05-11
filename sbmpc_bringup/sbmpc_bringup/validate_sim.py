from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from sbmpc_bringup.constants import FER_ARM_JOINT_NAMES, SBMPC_JOINT_STATES_TOPIC


@dataclass(frozen=True, slots=True)
class JointRecord:
    stamp_sec: float
    position: np.ndarray
    velocity: np.ndarray


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    diagnostics_count: int
    running_count: int
    joint_count: int
    planner_mode: str | None
    final_position_error: float | None
    min_position_error: float | None
    max_position_error: float | None
    mean_foreground_ms: float | None
    max_foreground_ms: float | None
    mean_bridge_loop_ms: float | None
    max_bridge_loop_ms: float | None
    mean_planner_step_wall_ms: float | None
    max_planner_step_wall_ms: float | None
    mean_control_prepare_ms: float | None
    max_control_prepare_ms: float | None
    mean_control_publish_ms: float | None
    max_control_publish_ms: float | None
    mean_background_gain_ms: float | None
    max_background_gain_ms: float | None
    accepted_planner_output_count: int | None
    rejected_planner_output_count: int | None
    max_gain_norm: float | None
    final_gain_norm: float | None
    max_gain_age_cycles: float | None
    final_gain_window_fill: int | None
    final_gain_completed_batch_count: int | None
    final_gain_dropped_snapshot_count: int | None
    gain_worker_running_samples: int
    gain_worker_error_count: int
    deadline_miss_count: int | None
    joint_velocity_rms_mean: float | None
    joint_velocity_abs_max: float | None
    tail_joint_spans: tuple[float, ...]

    @property
    def max_tail_joint_span(self) -> float | None:
        if not self.tail_joint_spans:
            return None
        return max(self.tail_joint_spans)


class SimulationValidationCollector(Node):
    def __init__(
        self,
        *,
        diagnostics_topic: str,
        joint_states_topic: str,
    ) -> None:
        super().__init__("sbmpc_sim_validation_collector")
        self.diagnostics: list[dict[str, object]] = []
        self.joint_records: list[JointRecord] = []
        self.create_subscription(String, diagnostics_topic, self._on_diagnostics, 10)
        self.create_subscription(JointState, joint_states_topic, self._on_joint_state, 10)

    def _on_diagnostics(self, message: String) -> None:
        try:
            self.diagnostics.append(json.loads(message.data))
        except json.JSONDecodeError:
            return

    def _on_joint_state(self, message: JointState) -> None:
        indices = joint_indices(message.name)
        if indices is None:
            return

        position = vector_from_indices(message.position, indices)
        if len(message.velocity) >= len(message.name):
            velocity = vector_from_indices(message.velocity, indices)
        else:
            velocity = np.zeros(len(indices), dtype=np.float64)
        stamp_sec = float(message.header.stamp.sec) + 1e-9 * float(
            message.header.stamp.nanosec
        )
        self.joint_records.append(
            JointRecord(stamp_sec=stamp_sec, position=position, velocity=velocity)
        )


def joint_indices(names: list[str]) -> tuple[int, ...] | None:
    name_to_index = {name: index for index, name in enumerate(names)}
    if not all(name in name_to_index for name in FER_ARM_JOINT_NAMES):
        return None
    return tuple(name_to_index[name] for name in FER_ARM_JOINT_NAMES)


def vector_from_indices(values, indices: tuple[int, ...]) -> np.ndarray:
    return np.asarray([values[index] for index in indices], dtype=np.float64)


def finite_values(rows: list[dict[str, object]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def last_finite_int(rows: list[dict[str, object]], key: str) -> int | None:
    values = finite_values(rows, key)
    if values.size == 0:
        return None
    return int(values[-1])


def last_text(rows: list[dict[str, object]], key: str) -> str | None:
    for row in reversed(rows):
        value = row.get(key)
        if value:
            return str(value)
    return None


def summarize(
    diagnostics: list[dict[str, object]],
    joint_records: list[JointRecord],
    *,
    tail_fraction: float,
) -> ValidationSummary:
    running = [row for row in diagnostics if row.get("state") == "running"]
    position_error = finite_values(running, "last_position_error")
    foreground_ms = finite_values(running, "last_foreground_planning_time_ms")
    if foreground_ms.size == 0:
        foreground_ms = finite_values(running, "last_planner_output_time_ms")
    bridge_ms = finite_values(running, "last_bridge_loop_time_ms")
    planner_step_wall_ms = finite_values(running, "last_planner_step_wall_time_ms")
    control_prepare_ms = finite_values(running, "last_control_prepare_time_ms")
    control_publish_ms = finite_values(running, "last_control_publish_time_ms")
    background_gain_ms = finite_values(running, "last_background_gain_time_ms")
    gain_norm = finite_values(running, "last_gain_norm")
    gain_age_cycles = finite_values(running, "last_gain_age_cycles")
    deadline_misses = finite_values(running, "deadline_miss_count")
    gain_worker_running_samples = sum(
        1 for row in running if row.get("last_gain_worker_running") is True
    )
    gain_worker_error_count = sum(
        1 for row in running if row.get("last_gain_worker_error")
    )

    tail_joint_spans: tuple[float, ...] = ()
    velocity_rms_mean: float | None = None
    velocity_abs_max: float | None = None
    if joint_records:
        positions = np.asarray([record.position for record in joint_records])
        velocities = np.asarray([record.velocity for record in joint_records])
        tail_start = int(
            max(
                0,
                min(len(positions) - 1, round(len(positions) * (1.0 - tail_fraction))),
            )
        )
        tail_joint_spans = tuple(
            float(value) for value in np.ptp(positions[tail_start:], axis=0)
        )
        velocity_rms_mean = float(
            np.mean(np.sqrt(np.mean(velocities * velocities, axis=1)))
        )
        velocity_abs_max = float(np.max(np.abs(velocities), initial=0.0))

    return ValidationSummary(
        diagnostics_count=len(diagnostics),
        running_count=len(running),
        joint_count=len(joint_records),
        planner_mode=last_text(running, "planner_mode"),
        final_position_error=float(position_error[-1]) if position_error.size else None,
        min_position_error=float(np.min(position_error)) if position_error.size else None,
        max_position_error=float(np.max(position_error)) if position_error.size else None,
        mean_foreground_ms=float(np.mean(foreground_ms)) if foreground_ms.size else None,
        max_foreground_ms=float(np.max(foreground_ms)) if foreground_ms.size else None,
        mean_bridge_loop_ms=float(np.mean(bridge_ms)) if bridge_ms.size else None,
        max_bridge_loop_ms=float(np.max(bridge_ms)) if bridge_ms.size else None,
        mean_planner_step_wall_ms=(
            float(np.mean(planner_step_wall_ms)) if planner_step_wall_ms.size else None
        ),
        max_planner_step_wall_ms=(
            float(np.max(planner_step_wall_ms)) if planner_step_wall_ms.size else None
        ),
        mean_control_prepare_ms=(
            float(np.mean(control_prepare_ms)) if control_prepare_ms.size else None
        ),
        max_control_prepare_ms=(
            float(np.max(control_prepare_ms)) if control_prepare_ms.size else None
        ),
        mean_control_publish_ms=(
            float(np.mean(control_publish_ms)) if control_publish_ms.size else None
        ),
        max_control_publish_ms=(
            float(np.max(control_publish_ms)) if control_publish_ms.size else None
        ),
        mean_background_gain_ms=(
            float(np.mean(background_gain_ms)) if background_gain_ms.size else None
        ),
        max_background_gain_ms=(
            float(np.max(background_gain_ms)) if background_gain_ms.size else None
        ),
        accepted_planner_output_count=last_finite_int(
            running,
            "accepted_planner_output_count",
        ),
        rejected_planner_output_count=last_finite_int(
            running,
            "rejected_planner_output_count",
        ),
        max_gain_norm=float(np.max(gain_norm)) if gain_norm.size else None,
        final_gain_norm=float(gain_norm[-1]) if gain_norm.size else None,
        max_gain_age_cycles=float(np.max(gain_age_cycles)) if gain_age_cycles.size else None,
        final_gain_window_fill=last_finite_int(running, "last_gain_window_fill"),
        final_gain_completed_batch_count=last_finite_int(
            running, "last_gain_completed_batch_count"
        ),
        final_gain_dropped_snapshot_count=last_finite_int(
            running, "last_gain_dropped_snapshot_count"
        ),
        gain_worker_running_samples=gain_worker_running_samples,
        gain_worker_error_count=gain_worker_error_count,
        deadline_miss_count=int(deadline_misses[-1]) if deadline_misses.size else None,
        joint_velocity_rms_mean=velocity_rms_mean,
        joint_velocity_abs_max=velocity_abs_max,
        tail_joint_spans=tail_joint_spans,
    )


def collect(duration_sec: float, *, diagnostics_topic: str, joint_states_topic: str):
    rclpy.init()
    node = SimulationValidationCollector(
        diagnostics_topic=diagnostics_topic,
        joint_states_topic=joint_states_topic,
    )
    deadline = time.monotonic() + duration_sec
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
        return list(node.diagnostics), list(node.joint_records)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def format_optional(value: float | int | None, *, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{precision}f}"


def print_summary(summary: ValidationSummary) -> None:
    print(
        "samples: "
        f"diagnostics={summary.diagnostics_count} "
        f"running={summary.running_count} "
        f"joint_states={summary.joint_count}"
    )
    print(f"planner_mode: {summary.planner_mode or 'n/a'}")
    print(
        "task_error_m: "
        f"final={format_optional(summary.final_position_error)} "
        f"min={format_optional(summary.min_position_error)} "
        f"max={format_optional(summary.max_position_error)}"
    )
    print(
        "foreground_ms: "
        f"mean={format_optional(summary.mean_foreground_ms, precision=2)} "
        f"max={format_optional(summary.max_foreground_ms, precision=2)}"
    )
    print(
        "bridge_loop_ms: "
        f"mean={format_optional(summary.mean_bridge_loop_ms, precision=2)} "
        f"max={format_optional(summary.max_bridge_loop_ms, precision=2)} "
        f"deadline_misses={format_optional(summary.deadline_miss_count)}"
    )
    print(
        "bridge_breakdown_ms: "
        f"planner_step_mean={format_optional(summary.mean_planner_step_wall_ms, precision=2)} "
        f"planner_step_max={format_optional(summary.max_planner_step_wall_ms, precision=2)} "
        f"prepare_mean={format_optional(summary.mean_control_prepare_ms, precision=2)} "
        f"prepare_max={format_optional(summary.max_control_prepare_ms, precision=2)} "
        f"publish_mean={format_optional(summary.mean_control_publish_ms, precision=2)} "
        f"publish_max={format_optional(summary.max_control_publish_ms, precision=2)}"
    )
    print(
        "planner_outputs: "
        f"accepted={format_optional(summary.accepted_planner_output_count)} "
        f"rejected={format_optional(summary.rejected_planner_output_count)}"
    )
    print(
        "gain_norm: "
        f"final={format_optional(summary.final_gain_norm)} "
        f"max={format_optional(summary.max_gain_norm)}"
    )
    print(
        "async_gain: "
        f"background_mean_ms={format_optional(summary.mean_background_gain_ms, precision=2)} "
        f"background_max_ms={format_optional(summary.max_background_gain_ms, precision=2)} "
        f"max_age_cycles={format_optional(summary.max_gain_age_cycles, precision=2)} "
        f"window_fill={format_optional(summary.final_gain_window_fill)} "
        f"completed_batches={format_optional(summary.final_gain_completed_batch_count)} "
        f"dropped_snapshots={format_optional(summary.final_gain_dropped_snapshot_count)} "
        f"worker_running_samples={summary.gain_worker_running_samples} "
        f"worker_errors={summary.gain_worker_error_count}"
    )
    print(
        "joint_velocity: "
        f"rms_mean={format_optional(summary.joint_velocity_rms_mean)} "
        f"abs_max={format_optional(summary.joint_velocity_abs_max)}"
    )
    print(
        "tail_joint_spans_rad: "
        + ", ".join(
            f"j{index + 1}={span:.4f}"
            for index, span in enumerate(summary.tail_joint_spans)
        )
    )


def assert_stable(
    summary: ValidationSummary,
    *,
    max_tail_joint_span: float,
    max_final_position_error: float,
    max_foreground_ms: float | None = None,
) -> bool:
    if (
        summary.final_position_error is None
        or summary.max_tail_joint_span is None
    ):
        return False
    foreground_ok = (
        True
        if max_foreground_ms is None
        else (
            summary.max_foreground_ms is not None
            and summary.max_foreground_ms <= max_foreground_ms
        )
    )
    return (
        summary.final_position_error <= max_final_position_error
        and summary.max_tail_joint_span <= max_tail_joint_span
        and foreground_ok
        and summary.rejected_planner_output_count == 0
        and summary.gain_worker_error_count == 0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-sec", type=float, default=12.0)
    parser.add_argument("--diagnostics-topic", default="/sbmpc/diagnostics")
    parser.add_argument("--joint-states-topic", default=SBMPC_JOINT_STATES_TOPIC)
    parser.add_argument("--tail-fraction", type=float, default=0.5)
    parser.add_argument("--assert-stable", action="store_true")
    parser.add_argument("--max-tail-joint-span", type=float, default=0.1)
    parser.add_argument("--max-final-position-error", type=float, default=0.001)
    parser.add_argument(
        "--max-foreground-ms",
        type=float,
        default=None,
        help=(
            "Optional controller-only foreground timing gate. Leave unset for "
            "MuJoCo behavior/visual checks, where simulation load is not part "
            "of the real robot controller budget."
        ),
    )
    args = parser.parse_args()

    if args.duration_sec <= 0.0:
        raise ValueError("--duration-sec must be positive.")
    if not 0.0 < args.tail_fraction <= 1.0:
        raise ValueError("--tail-fraction must be in (0, 1].")

    diagnostics, joint_records = collect(
        args.duration_sec,
        diagnostics_topic=args.diagnostics_topic,
        joint_states_topic=args.joint_states_topic,
    )
    summary = summarize(diagnostics, joint_records, tail_fraction=args.tail_fraction)
    print_summary(summary)

    if args.assert_stable:
        stable = assert_stable(
            summary,
            max_tail_joint_span=args.max_tail_joint_span,
            max_final_position_error=args.max_final_position_error,
            max_foreground_ms=args.max_foreground_ms,
        )
        print("verdict: " + ("stable" if stable else "unstable"))
        raise SystemExit(0 if stable else 1)


if __name__ == "__main__":
    main()
