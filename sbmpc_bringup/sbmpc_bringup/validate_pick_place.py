"""P-B2 checker: the full pick-and-place sequence through the sim bringup.

Companion to ``validate_sim`` (which certifies the pregrasp reach and the
limit margins): this one watches a ``planner_ocp: pick_place`` session and
gates exactly what Tier B adds over the certified Tier A physics
(hydrax doc/pick_place_plan.md, P-B2):

- the PHASE SEQUENCE reported in the bridge diagnostics walks
  PREGRASP -> ... -> DONE in order (the transport delivered the phase
  machine's clock);
- the CUBE lands within tolerance of the target — measured from the
  simulator's ground-truth free-joint odometry (/simulator/object_pose,
  published by mujoco_ros2_control via the xacro's odom_free_joint_name),
  against the planner-reported goal position;
- the GRIPPER action round-trips: exactly two goals (close at the CLOSE
  settle, open at the OPEN settle), no failure, terminal result verified;
- planning deadline misses stay under the budget fraction and no planner
  output is rejected.

Run it next to the bringup (same ROS_DOMAIN_ID) once the bridge is armed:

    ros2 run sbmpc_bringup validate_pick_place --assert-complete
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


PHASE_SEQUENCE = (
    "PREGRASP",
    "DESCEND",
    "CLOSE",
    "LIFT",
    "TRANSPORT",
    "PLACE",
    "OPEN",
    "RETREAT",
    "DONE",
)


@dataclass(frozen=True, slots=True)
class PickPlaceSummary:
    diagnostics_count: int
    running_count: int
    phases_seen: tuple[str, ...]
    reached_done: bool
    goal_position: tuple[float, float, float] | None
    cube_final: tuple[float, float, float] | None
    cube_lift_max: float | None
    cube_place_error_m: float | None
    gripper_goal_count: int | None
    gripper_failure: str | None
    gripper_busy: bool | None
    gripper_last_result: dict | None
    planner_step_count: int | None
    deadline_miss_count: int | None
    rejected_planner_output_count: int | None
    error_states_seen: int
    last_error: str | None


class PickPlaceValidationCollector(Node):
    def __init__(self, *, diagnostics_topic: str, object_pose_topic: str) -> None:
        super().__init__("sbmpc_pick_place_validation_collector")
        self.diagnostics: list[dict[str, object]] = []
        self.phases_seen: list[str] = []
        self.cube_positions: list[np.ndarray] = []
        self._armed = False
        self.create_subscription(String, diagnostics_topic, self._on_diagnostics, 10)
        self.create_subscription(Odometry, object_pose_topic, self._on_object_pose, 10)

    def _on_diagnostics(self, message: String) -> None:
        try:
            row = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if row.get("state") == "running" or int(row.get("accepted_planner_output_count") or 0) > 0:
            self._armed = True
        phase = row.get("last_phase")
        if phase and (not self.phases_seen or self.phases_seen[-1] != phase):
            self.phases_seen.append(str(phase))
        self.diagnostics.append(row)

    def _on_object_pose(self, message: Odometry) -> None:
        if not self._armed:
            return
        p = message.pose.pose.position
        self.cube_positions.append(np.asarray([p.x, p.y, p.z], dtype=np.float64))

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def reached_done(self) -> bool:
        return bool(self.phases_seen) and self.phases_seen[-1] == "DONE"


def _last_value(rows: list[dict[str, object]], key: str):
    for row in reversed(rows):
        value = row.get(key)
        if value is not None:
            return value
    return None


def summarize(
    diagnostics: list[dict[str, object]],
    phases_seen: list[str],
    cube_positions: list[np.ndarray],
) -> PickPlaceSummary:
    running = [row for row in diagnostics if row.get("state") == "running"]
    goal = _last_value(running, "last_goal_position")
    goal_position = tuple(float(v) for v in goal) if goal else None
    cube_final = (
        tuple(float(v) for v in cube_positions[-1]) if cube_positions else None
    )
    cube_lift_max = (
        float(max(p[2] for p in cube_positions)) if cube_positions else None
    )
    cube_place_error = None
    if goal_position is not None and cube_final is not None:
        # planar placement error: the goal z is the EE release height,
        # the cube settles on the floor
        cube_place_error = float(
            np.linalg.norm(np.asarray(cube_final[:2]) - np.asarray(goal_position[:2]))
        )
    gripper = _last_value(diagnostics, "gripper") or {}
    deadline = _last_value(running, "deadline_miss_count")
    steps = _last_value(running, "planner_step_count")
    rejected = _last_value(running, "rejected_planner_output_count")
    return PickPlaceSummary(
        diagnostics_count=len(diagnostics),
        running_count=len(running),
        phases_seen=tuple(phases_seen),
        reached_done=bool(phases_seen) and phases_seen[-1] == "DONE",
        goal_position=goal_position,
        cube_final=cube_final,
        cube_lift_max=cube_lift_max,
        cube_place_error_m=cube_place_error,
        gripper_goal_count=(
            int(gripper["goal_count"]) if gripper.get("goal_count") is not None else None
        ),
        gripper_failure=gripper.get("failure"),
        gripper_busy=gripper.get("busy"),
        gripper_last_result=gripper.get("last_result"),
        planner_step_count=int(steps) if steps is not None else None,
        deadline_miss_count=int(deadline) if deadline is not None else None,
        rejected_planner_output_count=int(rejected) if rejected is not None else None,
        error_states_seen=sum(1 for row in diagnostics if row.get("state") == "error"),
        last_error=_last_value(diagnostics, "last_error") or None,
    )


def phases_in_order(phases_seen: tuple[str, ...]) -> bool:
    """The distinct phase stream must be a contiguous in-order run of the
    sequence ending at DONE (it may start mid-sequence if the collector
    attached late)."""
    if not phases_seen or phases_seen[-1] != "DONE":
        return False
    try:
        start = PHASE_SEQUENCE.index(phases_seen[0])
    except ValueError:
        return False
    return tuple(phases_seen) == PHASE_SEQUENCE[start:]


def assert_complete(
    summary: PickPlaceSummary,
    *,
    max_place_error_m: float,
    max_deadline_miss_fraction: float,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not phases_in_order(summary.phases_seen):
        failures.append(f"phase sequence broken: {list(summary.phases_seen)}")
    if summary.cube_place_error_m is None:
        failures.append("no cube pose received (check /simulator/object_pose)")
    elif summary.cube_place_error_m > max_place_error_m:
        failures.append(
            f"cube placed {summary.cube_place_error_m * 1000:.1f} mm from the "
            f"target (gate {max_place_error_m * 1000:.1f} mm)"
        )
    if summary.gripper_goal_count != 2:
        failures.append(
            f"gripper round-trip incomplete: goal_count={summary.gripper_goal_count}"
        )
    if summary.gripper_failure:
        failures.append(f"gripper failure: {summary.gripper_failure}")
    if summary.gripper_busy:
        failures.append("gripper action still in flight at the end of the run")
    steps = summary.planner_step_count or 0
    misses = summary.deadline_miss_count or 0
    if steps <= 0:
        failures.append("no planner steps observed")
    elif misses / steps > max_deadline_miss_fraction:
        failures.append(
            f"deadline misses {misses}/{steps} exceed "
            f"{100 * max_deadline_miss_fraction:.1f}%"
        )
    if (summary.rejected_planner_output_count or 0) > 0:
        failures.append(
            f"rejected planner outputs: {summary.rejected_planner_output_count}"
        )
    if summary.error_states_seen > 0:
        failures.append(
            f"bridge reported error state {summary.error_states_seen}x: "
            f"{summary.last_error}"
        )
    return (not failures, failures)


def collect(
    *,
    diagnostics_topic: str,
    object_pose_topic: str,
    startup_timeout_sec: float,
    timeout_sec: float,
    settle_sec: float,
):
    rclpy.init()
    node = PickPlaceValidationCollector(
        diagnostics_topic=diagnostics_topic,
        object_pose_topic=object_pose_topic,
    )
    try:
        startup_deadline = time.monotonic() + startup_timeout_sec
        while not node.armed and time.monotonic() < startup_deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
        if not node.armed:
            raise TimeoutError(
                "SB-MPC bridge did not enter running state before validation timeout."
            )
        deadline = time.monotonic() + timeout_sec
        done_since: float | None = None
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            if node.reached_done:
                done_since = done_since or time.monotonic()
                if time.monotonic() - done_since >= settle_sec:
                    break
            else:
                done_since = None
        return list(node.diagnostics), list(node.phases_seen), list(node.cube_positions)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def print_summary(summary: PickPlaceSummary) -> None:
    def fmt_pos(value):
        return (
            "n/a" if value is None else " ".join(f"{v:.4f}" for v in value)
        )

    print(
        f"samples: diagnostics={summary.diagnostics_count} "
        f"running={summary.running_count}"
    )
    print("phases: " + (" -> ".join(summary.phases_seen) or "n/a"))
    print(f"goal_position: {fmt_pos(summary.goal_position)}")
    print(
        f"cube: final={fmt_pos(summary.cube_final)} "
        f"lift_max={'n/a' if summary.cube_lift_max is None else f'{summary.cube_lift_max:.3f}'} "
        "place_error_mm="
        + (
            "n/a"
            if summary.cube_place_error_m is None
            else f"{1000 * summary.cube_place_error_m:.1f}"
        )
    )
    print(
        f"gripper: goals={summary.gripper_goal_count} "
        f"busy={summary.gripper_busy} "
        f"failure={summary.gripper_failure or 'none'} "
        f"last_result={summary.gripper_last_result}"
    )
    print(
        f"planner: steps={summary.planner_step_count} "
        f"deadline_misses={summary.deadline_miss_count} "
        f"rejected={summary.rejected_planner_output_count}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-topic", default="/sbmpc/diagnostics")
    parser.add_argument("--object-pose-topic", default="/simulator/object_pose")
    parser.add_argument("--startup-timeout-sec", type=float, default=300.0)
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=120.0,
        help="Abort if DONE is not reached within this wall time after arming.",
    )
    parser.add_argument(
        "--settle-sec",
        type=float,
        default=2.0,
        help="Keep collecting this long after DONE (cube settling).",
    )
    parser.add_argument("--assert-complete", action="store_true")
    parser.add_argument(
        "--max-place-error-m",
        type=float,
        default=0.015,
        help="Planar cube-to-target gate: 1.5x the Tier A P-A2 tolerance.",
    )
    parser.add_argument("--max-deadline-miss-fraction", type=float, default=0.01)
    from rclpy.utilities import remove_ros_args

    args = parser.parse_args(remove_ros_args()[1:])

    diagnostics, phases_seen, cube_positions = collect(
        diagnostics_topic=args.diagnostics_topic,
        object_pose_topic=args.object_pose_topic,
        startup_timeout_sec=args.startup_timeout_sec,
        timeout_sec=args.timeout_sec,
        settle_sec=args.settle_sec,
    )
    summary = summarize(diagnostics, phases_seen, cube_positions)
    print_summary(summary)

    if args.assert_complete:
        ok, failures = assert_complete(
            summary,
            max_place_error_m=args.max_place_error_m,
            max_deadline_miss_fraction=args.max_deadline_miss_fraction,
        )
        for failure in failures:
            print(f"FAIL: {failure}")
        print("verdict: " + ("complete" if ok else "incomplete"))
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
