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
        self._lock = Lock()
        self._busy = False
        self._busy_since: float | None = None
        self._failure: str | None = None
        self._goal_count = 0
        self._active_command: str | None = None
        self._last_command: str | None = None
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
        """Diagnostics view: command/goal counters and the last result."""
        with self._lock:
            return {
                "action_name": self._action_name,
                "busy": self._busy,
                "goal_count": self._goal_count,
                "last_command": self._last_command,
                "last_result": self._last_result,
                "failure": self._failure,
            }

    # --- goal submission (planner worker thread) -------------------------

    def execute(self, command: str) -> None:
        command = str(command).strip().lower()
        if command not in (GRIPPER_OPEN, GRIPPER_CLOSE):
            self._set_failure(f"unknown gripper command {command!r}.")
            return
        with self._lock:
            if self._failure is not None:
                return
            if self._busy:
                # The phase machine spaces commands by full dwells and the
                # bridge freezes its clock while a goal is in flight, so an
                # overlap means the sequencing contract broke — fail closed.
                self._failure = (
                    f"gripper command '{command}' issued while "
                    f"'{self._active_command}' is still in flight."
                )
                return
            self._busy = True
            self._busy_since = monotonic()
            self._goal_count += 1
            self._active_command = command
            self._last_command = command
        if not self._client.server_is_ready():
            self._set_failure(
                f"gripper action server '{self._action_name}' is not "
                "available."
            )
            return
        goal = GripperCommand.Goal()
        goal.command.position = (
            self._close_position
            if command == GRIPPER_CLOSE
            else self._open_position
        )
        goal.command.max_effort = self._max_effort
        future = self._client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)

    # --- action callbacks (node executor thread) -------------------------

    def _on_goal_response(self, future) -> None:
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
        goal_handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future) -> None:
        try:
            wrapped = future.result()
        except Exception as exc:  # noqa: BLE001 — any failure fails closed
            self._set_failure(f"gripper action failed: {exc}")
            return
        result = wrapped.result
        status = int(wrapped.status)
        with self._lock:
            command = self._active_command
            self._last_result = {
                "position": float(result.position),
                "reached_goal": bool(result.reached_goal),
                "stalled": bool(result.stalled),
                "status": status,
            }
        if status != GoalStatus.STATUS_SUCCEEDED:
            self._set_failure(
                f"gripper {command} ended with action status {status}."
            )
            return
        verified = (
            (result.reached_goal or result.stalled)
            if command == GRIPPER_CLOSE
            else result.reached_goal
        )
        if not verified:
            self._set_failure(
                f"gripper {command} did not verify: position "
                f"{float(result.position):.4f}, reached_goal "
                f"{bool(result.reached_goal)}, stalled {bool(result.stalled)}."
            )
            return
        with self._lock:
            self._busy = False
            self._busy_since = None
            self._active_command = None

    def _set_failure(self, message: str) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = str(message)
