from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from sbmpc_bringup.constants import FER_ARM_JOINT_NAMES


# FR3 ("fer") per-joint limits, in FER_ARM_JOINT_NAMES order.
FR3_TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
FR3_VELOCITY_LIMITS = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
FR3_POSITION_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
FR3_POSITION_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
FER_TCP_FRAME = "fer_hand_tcp"
PREGRASP_GOAL_POSITION = np.array([0.5, 0.0, 0.1], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class JointRecord:
    stamp_sec: float
    position: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray | None = None


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
    p95_foreground_ms: float | None
    max_foreground_ms: float | None
    mean_bridge_loop_ms: float | None
    max_bridge_loop_ms: float | None
    mean_planner_step_wall_ms: float | None
    max_planner_step_wall_ms: float | None
    mean_control_prepare_ms: float | None
    max_control_prepare_ms: float | None
    mean_control_publish_ms: float | None
    max_control_publish_ms: float | None
    accepted_planner_output_count: int | None
    rejected_planner_output_count: int | None
    max_gain_norm: float | None
    final_gain_norm: float | None
    deadline_miss_count: int | None
    joint_velocity_rms_mean: float | None
    joint_velocity_abs_max: float | None
    tail_joint_spans: tuple[float, ...]
    joint_velocity_abs_max_per_joint: tuple[float, ...] = ()
    joint_effort_abs_max_per_joint: tuple[float, ...] | None = None
    worst_velocity_fraction: float | None = None
    worst_torque_fraction: float | None = None
    worst_position_fraction: float | None = None

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
        self.robot_description: str | None = None
        # Only record joint samples while the controller is actively driving, so
        # the torque/velocity limit gate ignores any pre-activation gravity sag.
        self._armed = False
        self.create_subscription(String, diagnostics_topic, self._on_diagnostics, 10)
        robot_description_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            String,
            "/robot_description",
            self._on_robot_description,
            robot_description_qos,
        )
        self.create_subscription(JointState, joint_states_topic, self._on_joint_state, 10)

    def _on_robot_description(self, message: String) -> None:
        self.robot_description = message.data

    def _on_diagnostics(self, message: String) -> None:
        try:
            row = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if row.get("state") == "running" or int(row.get("accepted_planner_output_count") or 0) > 0:
            self._armed = True
        self.diagnostics.append(row)

    @property
    def armed(self) -> bool:
        return self._armed

    def _on_joint_state(self, message: JointState) -> None:
        if not self._armed:
            return
        indices = joint_indices(message.name)
        if indices is None:
            return

        position = vector_from_indices(message.position, indices)
        if len(message.velocity) >= len(message.name):
            velocity = vector_from_indices(message.velocity, indices)
        else:
            velocity = np.zeros(len(indices), dtype=np.float64)
        if len(message.effort) >= len(message.name):
            effort = vector_from_indices(message.effort, indices)
        else:
            effort = None
        stamp_sec = float(message.header.stamp.sec) + 1e-9 * float(
            message.header.stamp.nanosec
        )
        self.joint_records.append(
            JointRecord(
                stamp_sec=stamp_sec,
                position=position,
                velocity=velocity,
                effort=effort,
            )
        )


def joint_indices(names: list[str]) -> tuple[int, ...] | None:
    name_to_index = {name: index for index, name in enumerate(names)}
    if not all(name in name_to_index for name in FER_ARM_JOINT_NAMES):
        return None
    return tuple(name_to_index[name] for name in FER_ARM_JOINT_NAMES)


def vector_from_indices(values, indices: tuple[int, ...]) -> np.ndarray:
    return np.asarray([values[index] for index in indices], dtype=np.float64)


def ee_position_errors(
    joint_records: list[JointRecord],
    *,
    robot_description: str,
) -> np.ndarray:
    import pinocchio as pin

    model = pin.buildModelFromXML(robot_description)
    data = model.createData()
    frame_id = model.getFrameId(FER_TCP_FRAME)
    if frame_id >= len(model.frames):
        raise ValueError(f"frame {FER_TCP_FRAME!r} is missing from robot_description")

    joint_q_indices = []
    for name in FER_ARM_JOINT_NAMES:
        joint_id = model.getJointId(name)
        if joint_id == 0:
            raise ValueError(f"joint {name!r} is missing from robot_description")
        joint_q_indices.append(model.joints[joint_id].idx_q)

    errors = np.empty(len(joint_records), dtype=np.float64)
    for index, record in enumerate(joint_records):
        q = pin.neutral(model)
        q[np.asarray(joint_q_indices)] = record.position
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacement(model, data, frame_id)
        errors[index] = np.linalg.norm(
            np.asarray(data.oMf[frame_id].translation) - PREGRASP_GOAL_POSITION
        )
    return errors


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


def unique_planner_steps(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Keep one diagnostics sample for each completed planner step."""
    unique: list[dict[str, object]] = []
    last_step_count: int | None = None
    for row in rows:
        raw_count = row.get("planner_step_count")
        if raw_count is None:
            unique.append(row)
            continue
        step_count = int(raw_count)
        if step_count <= 0 or step_count == last_step_count:
            continue
        unique.append(row)
        last_step_count = step_count
    return unique


def summarize(
    diagnostics: list[dict[str, object]],
    joint_records: list[JointRecord],
    *,
    robot_description: str | None = None,
    tail_fraction: float,
) -> ValidationSummary:
    running = [row for row in diagnostics if row.get("state") == "running"]
    planner_steps = unique_planner_steps(running)
    position_error = finite_values(running, "last_position_error")
    if position_error.size == 0 and robot_description and joint_records:
        position_error = ee_position_errors(
            joint_records, robot_description=robot_description
        )
    foreground_ms = finite_values(planner_steps, "last_foreground_planning_time_ms")
    if foreground_ms.size == 0:
        foreground_ms = finite_values(planner_steps, "last_planner_output_time_ms")
    bridge_ms = finite_values(planner_steps, "last_bridge_loop_time_ms")
    planner_step_wall_ms = finite_values(
        planner_steps, "last_planner_step_wall_time_ms"
    )
    control_prepare_ms = finite_values(planner_steps, "last_control_prepare_time_ms")
    control_publish_ms = finite_values(planner_steps, "last_control_publish_time_ms")
    gain_norm = finite_values(planner_steps, "last_gain_norm")
    deadline_misses = finite_values(running, "deadline_miss_count")

    tail_joint_spans: tuple[float, ...] = ()
    velocity_rms_mean: float | None = None
    velocity_abs_max: float | None = None
    vel_per_joint: tuple[float, ...] = ()
    eff_per_joint: tuple[float, ...] | None = None
    worst_velocity_fraction: float | None = None
    worst_torque_fraction: float | None = None
    worst_position_fraction: float | None = None
    if joint_records:
        positions = np.asarray([record.position for record in joint_records])
        velocities = np.asarray([record.velocity for record in joint_records])
        tail_start = int(
            max(
                0,
                min(len(positions) - 1, round(len(positions) * (1.0 - tail_fraction))),
            )
        )
        position_center = 0.5 * (FR3_POSITION_MIN + FR3_POSITION_MAX)
        position_half_range = 0.5 * (FR3_POSITION_MAX - FR3_POSITION_MIN)
        peak_position_offset = np.max(
            np.abs(positions - position_center), axis=0
        )
        worst_position_fraction = float(
            np.max(peak_position_offset / position_half_range)
        )
        tail_joint_spans = tuple(
            float(value) for value in np.ptp(positions[tail_start:], axis=0)
        )
        velocity_rms_mean = float(
            np.mean(np.sqrt(np.mean(velocities * velocities, axis=1)))
        )
        velocity_abs_max = float(np.max(np.abs(velocities), initial=0.0))
        velocity_peaks = np.max(np.abs(velocities), axis=0)
        vel_per_joint = tuple(float(value) for value in velocity_peaks)
        worst_velocity_fraction = float(np.max(velocity_peaks / FR3_VELOCITY_LIMITS))
        if all(record.effort is not None for record in joint_records):
            efforts = np.asarray([record.effort for record in joint_records])
            effort_peaks = np.max(np.abs(efforts), axis=0)
            eff_per_joint = tuple(float(value) for value in effort_peaks)
            worst_torque_fraction = float(np.max(effort_peaks / FR3_TORQUE_LIMITS))

    return ValidationSummary(
        diagnostics_count=len(diagnostics),
        running_count=len(running),
        joint_count=len(joint_records),
        planner_mode=last_text(running, "planner_mode"),
        final_position_error=float(position_error[-1]) if position_error.size else None,
        min_position_error=float(np.min(position_error)) if position_error.size else None,
        max_position_error=float(np.max(position_error)) if position_error.size else None,
        mean_foreground_ms=float(np.mean(foreground_ms)) if foreground_ms.size else None,
        p95_foreground_ms=(
            float(np.percentile(foreground_ms, 95)) if foreground_ms.size else None
        ),
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
        deadline_miss_count=int(deadline_misses[-1]) if deadline_misses.size else None,
        joint_velocity_rms_mean=velocity_rms_mean,
        joint_velocity_abs_max=velocity_abs_max,
        tail_joint_spans=tail_joint_spans,
        joint_velocity_abs_max_per_joint=vel_per_joint,
        joint_effort_abs_max_per_joint=eff_per_joint,
        worst_velocity_fraction=worst_velocity_fraction,
        worst_torque_fraction=worst_torque_fraction,
        worst_position_fraction=worst_position_fraction,
    )


def collect(
    duration_sec: float,
    *,
    diagnostics_topic: str,
    joint_states_topic: str,
    startup_timeout_sec: float = 120.0,
):
    rclpy.init()
    node = SimulationValidationCollector(
        diagnostics_topic=diagnostics_topic,
        joint_states_topic=joint_states_topic,
    )
    startup_deadline = time.monotonic() + startup_timeout_sec
    try:
        while not node.armed and time.monotonic() < startup_deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
        if not node.armed:
            raise TimeoutError(
                "SB-MPC bridge did not enter running state before validation timeout."
            )
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
        return (
            list(node.diagnostics),
            list(node.joint_records),
            node.robot_description,
        )
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
        f"p95={format_optional(summary.p95_foreground_ms, precision=2)} "
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
    wt = summary.worst_torque_fraction
    wv = summary.worst_velocity_fraction
    wp = summary.worst_position_fraction
    print(
        "limit_usage_pct: "
        f"worst_torque={'n/a' if wt is None else f'{wt * 100:.0f}'} "
        f"worst_velocity={'n/a' if wv is None else f'{wv * 100:.0f}'} "
        f"worst_position={'n/a' if wp is None else f'{wp * 100:.0f}'}"
    )
    if summary.joint_effort_abs_max_per_joint is not None:
        print(
            "peak_torque_Nm: "
            + ", ".join(
                f"j{index + 1}={value:.1f}({value / limit * 100:.0f}%)"
                for index, (value, limit) in enumerate(
                    zip(summary.joint_effort_abs_max_per_joint, FR3_TORQUE_LIMITS)
                )
            )
        )
    if summary.joint_velocity_abs_max_per_joint:
        print(
            "peak_velocity_rad_s: "
            + ", ".join(
                f"j{index + 1}={value:.2f}({value / limit * 100:.0f}%)"
                for index, (value, limit) in enumerate(
                    zip(summary.joint_velocity_abs_max_per_joint, FR3_VELOCITY_LIMITS)
                )
            )
        )


def assert_stable(
    summary: ValidationSummary,
    *,
    max_tail_joint_span: float,
    max_final_position_error: float,
    max_foreground_ms: float | None = None,
    max_p95_planning_ms: float | None = None,
    max_torque_fraction: float | None = None,
    max_velocity_fraction: float | None = None,
    max_position_fraction: float | None = None,
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
    p95_ok = (
        True
        if max_p95_planning_ms is None
        else (
            summary.p95_foreground_ms is not None
            and summary.p95_foreground_ms <= max_p95_planning_ms
        )
    )
    # Limit gates fail closed: if a bound is requested but the measurement is
    # missing, the run is not certified.
    torque_ok = (
        True
        if max_torque_fraction is None
        else (
            summary.worst_torque_fraction is not None
            and summary.worst_torque_fraction <= max_torque_fraction
        )
    )
    velocity_ok = (
        True
        if max_velocity_fraction is None
        else (
            summary.worst_velocity_fraction is not None
            and summary.worst_velocity_fraction <= max_velocity_fraction
        )
    )
    position_ok = (
        True
        if max_position_fraction is None
        else (
            summary.worst_position_fraction is not None
            and summary.worst_position_fraction <= max_position_fraction
        )
    )
    return (
        summary.final_position_error <= max_final_position_error
        and summary.max_tail_joint_span <= max_tail_joint_span
        and foreground_ok
        and p95_ok
        and torque_ok
        and velocity_ok
        and position_ok
        and summary.rejected_planner_output_count == 0
        and summary.final_gain_norm is not None
        and np.isfinite(summary.final_gain_norm)
        and summary.final_gain_norm > 0.0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-sec", type=float, default=16.0)
    parser.add_argument("--startup-timeout-sec", type=float, default=120.0)
    parser.add_argument("--diagnostics-topic", default="/sbmpc/diagnostics")
    parser.add_argument("--joint-states-topic", default="/joint_states")
    parser.add_argument("--tail-fraction", type=float, default=0.5)
    parser.add_argument("--assert-stable", action="store_true")
    parser.add_argument("--max-tail-joint-span", type=float, default=0.1)
    parser.add_argument("--max-final-position-error", type=float, default=0.01)
    parser.add_argument(
        "--max-p95-planning-ms",
        type=float,
        default=None,
        help="Optional p95 same-cycle MPC timing gate in milliseconds.",
    )
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
    parser.add_argument(
        "--max-torque-fraction",
        type=float,
        default=0.9,
        help=(
            "Fail if peak measured joint effort exceeds this fraction of the "
            "FR3 per-joint torque limit during active control. Set <=0 to disable."
        ),
    )
    parser.add_argument(
        "--max-position-fraction",
        type=float,
        default=0.9,
        help="Fail if joint positions exceed this fraction of their range.",
    )
    parser.add_argument(
        "--max-velocity-fraction",
        type=float,
        default=0.9,
        help=(
            "Fail if peak measured joint velocity exceeds this fraction of the "
            "FR3 per-joint velocity limit during active control. Set <=0 to disable."
        ),
    )
    from rclpy.utilities import remove_ros_args

    args = parser.parse_args(remove_ros_args()[1:])

    if args.duration_sec <= 0.0:
        raise ValueError("--duration-sec must be positive.")
    if not 0.0 < args.tail_fraction <= 1.0:
        raise ValueError("--tail-fraction must be in (0, 1].")

    diagnostics, joint_records, robot_description = collect(
        args.duration_sec,
        diagnostics_topic=args.diagnostics_topic,
        joint_states_topic=args.joint_states_topic,
        startup_timeout_sec=args.startup_timeout_sec,
    )
    summary = summarize(
        diagnostics,
        joint_records,
        tail_fraction=args.tail_fraction,
        robot_description=robot_description,
    )
    print_summary(summary)

    if args.assert_stable:
        stable = assert_stable(
            summary,
            max_tail_joint_span=args.max_tail_joint_span,
            max_final_position_error=args.max_final_position_error,
            max_foreground_ms=args.max_foreground_ms,
            max_p95_planning_ms=(
                args.max_p95_planning_ms
                if args.max_p95_planning_ms is not None
                and args.max_p95_planning_ms > 0
                else None
            ),
            max_torque_fraction=(
                args.max_torque_fraction if args.max_torque_fraction > 0 else None
            ),
            max_velocity_fraction=(
                args.max_velocity_fraction if args.max_velocity_fraction > 0 else None
            ),
            max_position_fraction=(
                args.max_position_fraction if args.max_position_fraction > 0 else None
            ),
        )
        print("verdict: " + ("stable" if stable else "unstable"))
        raise SystemExit(0 if stable else 1)


if __name__ == "__main__":
    main()
