"""Gripper action client for the pick-and-place bridge (P3).

One ``control_msgs/action/GripperCommand`` client serves both backends
(user decision 2026-07-10): the sim's ``gripper_action_controller``
exposes ``/gripper_action_controller/gripper_cmd`` and the real
``agimus_franka_gripper`` node exposes ``/fer_gripper/gripper_action`` of
the SAME action type, internally mapping close→franka grasp (with force)
and open→franka move. The action name is therefore the only per-backend
difference, injected by the launch.

The bridge stays transport: it executes the planner's ``gripper_command``
("open"/"close") and reports the outcome; the phase machine owns *when*.
Goals are sent async off the 25 Hz hot path; result callbacks ride the
node executor, so all state is lock-guarded.

Result verification (fed back for the plan's grasp verification):

- ``close`` verifies on ``reached_goal`` OR ``stalled`` — closing on the
  cube legitimately stalls the sim's effort controller
  (``allow_stalling: true``), while the real grasp reports
  ``reached_goal`` within its epsilon window;
- ``open`` verifies on ``reached_goal`` only;
- a rejected goal, a non-SUCCEEDED terminal status, or a failed
  verification latches ``failure`` — the bridge escalates it to the
  fail-closed fatal path (the plan's "safe abort on a failed grasp").
"""

from __future__ import annotations

from threading import Lock
from time import monotonic

from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node

GRIPPER_OPEN = "open"
GRIPPER_CLOSE = "close"

STAGE_IDLE = "idle"
STAGE_WAITING_FOR_GOAL_RESPONSE = "waiting_for_goal_response"
STAGE_WAITING_FOR_RESULT = "waiting_for_result"
STAGE_SUCCEEDED = "succeeded"
STAGE_FAILED = "failed"


class GripperCommandClient:
    """Sequenced open/close goals over one GripperCommand action."""

    def __init__(
        self,
        node: Node,
        *,
        action_name: str,
        close_position: float = 0.0,
        open_position: float = 0.04,
        max_effort: float = 40.0,
    ) -> None:
        self._action_name = str(action_name)
        self._close_position = float(close_position)
        self._open_position = float(open_position)
        self._max_effort = float(max_effort)
        self._client = ActionClient(node, GripperCommand, self._action_name)
        self._logger = node.get_logger()
        self._lock = Lock()
        self._busy = False
        self._busy_since: float | None = None
        self._failure: str | None = None
        self._goal_count = 0
        self._stage = STAGE_IDLE
        self._active_command: str | None = None
        self._last_command: str | None = None
        self._submitted_position: float | None = None
        self._submitted_max_effort: float | None = None
        self._goal_accepted_at: float | None = None
        self._goal_response_latency_sec: float | None = None
        self._result_latency_sec: float | None = None
        self._total_duration_sec: float | None = None
        self._last_result: dict | None = None

    # --- bridge-facing state --------------------------------------------

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    @property
    def failure(self) -> str | None:
        with self._lock:
            return self._failure

    def busy_duration_sec(self) -> float | None:
        with self._lock:
            if self._busy_since is None:
                return None
            return monotonic() - self._busy_since

    def snapshot(self) -> dict:
        """Diagnostics view of the most recent action lifecycle."""
        now = monotonic()
        with self._lock:
            busy_duration_sec = (
                now - self._busy_since
                if self._busy_since is not None
                else None
            )
            return {
                "action_name": self._action_name,
                "stage": self._stage,
                "busy": self._busy,
                "busy_duration_sec": busy_duration_sec,
                "goal_count": self._goal_count,
                "active_command": self._active_command,
                "last_command": self._last_command,
                "submitted_position": self._submitted_position,
                "submitted_max_effort": self._submitted_max_effort,
                "goal_response_latency_sec": self._goal_response_latency_sec,
                "result_latency_sec": self._result_latency_sec,
                "total_duration_sec": self._total_duration_sec,
                "last_result": (
                    dict(self._last_result)
                    if self._last_result is not None
                    else None
                ),
                "failure": self._failure,
            }

    # --- goal submission (planner worker thread) -------------------------

    def execute(self, command: str) -> None:
        command = str(command).strip().lower()
        if command not in (GRIPPER_OPEN, GRIPPER_CLOSE):
            self._set_failure(f"unknown gripper command {command!r}.")
            return
        position = (
            self._close_position
            if command == GRIPPER_CLOSE
            else self._open_position
        )
        overlap_failure: str | None = None
        goal_count = 0
        with self._lock:
            if self._failure is not None:
                return
            if self._busy:
                # The phase machine spaces commands by full dwells and the
                # bridge freezes its clock while a goal is in flight, so an
                # overlap means the sequencing contract broke — fail closed.
                overlap_failure = (
                    f"gripper command '{command}' issued while "
                    f"'{self._active_command}' is still in flight."
                )
                self._latch_failure_locked(overlap_failure, monotonic())
            else:
                self._busy = True
                self._busy_since = monotonic()
                self._goal_count += 1
                goal_count = self._goal_count
                self._stage = STAGE_WAITING_FOR_GOAL_RESPONSE
                self._active_command = command
                self._last_command = command
                self._submitted_position = position
                self._submitted_max_effort = self._max_effort
                self._goal_accepted_at = None
                self._goal_response_latency_sec = None
                self._result_latency_sec = None
                self._total_duration_sec = None
        if overlap_failure is not None:
            self._log_failure(overlap_failure)
            return
        if not self._client.server_is_ready():
            self._set_failure(
                f"gripper action server '{self._action_name}' is not "
                "available."
            )
            return
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = self._max_effort
        future = self._client.send_goal_async(goal)
        self._log_info(
            "[gripper action] submitted "
            f"goal={goal_count} command={command} "
            f"position={position:.5f}m max_effort={self._max_effort:.2f}N "
            f"action={self._action_name}"
        )
        future.add_done_callback(self._on_goal_response)

    # --- action callbacks (node executor thread) -------------------------

    def _on_goal_response(self, future) -> None:
        response_time = monotonic()
        with self._lock:
            if self._busy_since is not None:
                self._goal_response_latency_sec = (
                    response_time - self._busy_since
                )
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001 — any failure fails closed
            self._set_failure(f"gripper goal submission failed: {exc}")
            return
        if goal_handle is None or not goal_handle.accepted:
            self._set_failure(
                f"gripper goal rejected by '{self._action_name}'."
            )
            return
        with self._lock:
            self._goal_accepted_at = response_time
            if self._failure is None:
                self._stage = STAGE_WAITING_FOR_RESULT
            command = self._active_command
            goal_count = self._goal_count
            response_latency = self._goal_response_latency_sec
        result_future = goal_handle.get_result_async()
        self._log_info(
            "[gripper action] accepted "
            f"goal={goal_count} command={command} "
            f"response_latency={self._format_seconds(response_latency)}s"
        )
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future) -> None:
        try:
            wrapped = future.result()
        except Exception as exc:  # noqa: BLE001 — any failure fails closed
            self._set_failure(f"gripper action failed: {exc}")
            return
        result = wrapped.result
        status = int(wrapped.status)
        result_time = monotonic()
        with self._lock:
            command = self._active_command
            verified = bool(
                status == GoalStatus.STATUS_SUCCEEDED
                and (
                    (result.reached_goal or result.stalled)
                    if command == GRIPPER_CLOSE
                    else result.reached_goal
                )
            )
            if self._goal_accepted_at is not None:
                self._result_latency_sec = (
                    result_time - self._goal_accepted_at
                )
            if self._busy_since is not None:
                self._total_duration_sec = result_time - self._busy_since
            self._last_result = {
                "command": command,
                "position": float(result.position),
                "reached_goal": bool(result.reached_goal),
                "stalled": bool(result.stalled),
                "status": status,
                "verified": verified,
                "goal_response_latency_sec": self._goal_response_latency_sec,
                "result_latency_sec": self._result_latency_sec,
                "total_duration_sec": self._total_duration_sec,
            }
        if status != GoalStatus.STATUS_SUCCEEDED:
            self._set_failure(
                f"gripper {command} ended with action status {status}."
            )
            return
        if not verified:
            self._set_failure(
                f"gripper {command} did not verify: position "
                f"{float(result.position):.4f}, reached_goal "
                f"{bool(result.reached_goal)}, stalled {bool(result.stalled)}."
            )
            return
        with self._lock:
            if self._failure is None:
                self._stage = STAGE_SUCCEEDED
            self._busy = False
            self._busy_since = None
            self._active_command = None
            goal_count = self._goal_count
            stage = self._stage
            response_latency = self._goal_response_latency_sec
            result_latency = self._result_latency_sec
            total_duration = self._total_duration_sec
        self._log_info(
            "[gripper action] result "
            f"goal={goal_count} stage={stage} command={command} "
            f"status={status} verified=True "
            f"position={float(result.position):.5f}m "
            f"reached_goal={bool(result.reached_goal)} "
            f"stalled={bool(result.stalled)} "
            f"response_latency={self._format_seconds(response_latency)}s "
            f"result_latency={self._format_seconds(result_latency)}s "
            f"total={self._format_seconds(total_duration)}s"
        )

    def _set_failure(self, message: str) -> None:
        first_failure = False
        with self._lock:
            first_failure = self._latch_failure_locked(
                str(message), monotonic()
            )
        if first_failure:
            self._log_failure(str(message))

    def _latch_failure_locked(self, message: str, now: float) -> bool:
        """Latch a failure while the caller holds ``self._lock``."""
        if self._failure is not None:
            return False
        self._failure = str(message)
        self._stage = STAGE_FAILED
        if (
            self._busy_since is not None
            and self._total_duration_sec is None
        ):
            self._total_duration_sec = now - self._busy_since
        return True

    def _log_failure(self, message: str) -> None:
        """Emit one terminal event without holding the action-state lock."""
        snapshot = self.snapshot()
        self._log_error(
            "[gripper action] failed "
            f"goal={snapshot['goal_count']} "
            f"command={snapshot['active_command']} "
            f"position={snapshot['submitted_position']} "
            f"max_effort={snapshot['submitted_max_effort']} "
            "response_latency="
            f"{self._format_seconds(snapshot['goal_response_latency_sec'])}s "
            f"total={self._format_seconds(snapshot['total_duration_sec'])}s "
            f"result={snapshot['last_result']!r} reason={message}"
        )

    def _log_info(self, message: str) -> None:
        """Keep observability best-effort and outside action state changes."""
        try:
            self._logger.info(message)
        except Exception:  # noqa: BLE001 — logging must not affect transport
            pass

    def _log_error(self, message: str) -> None:
        """Keep observability best-effort and outside action state changes."""
        try:
            self._logger.error(message)
        except Exception:  # noqa: BLE001 — logging must not affect transport
            pass

    @staticmethod
    def _format_seconds(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"
