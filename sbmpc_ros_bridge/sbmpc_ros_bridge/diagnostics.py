from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True, slots=True)
class BridgeDiagnostics:
    state: str
    control_enabled: bool
    force_zero_control: bool
    received_sensor_count: int
    valid_sensor_count: int
    rejected_sensor_count: int
    published_control_count: int
    nonzero_control_count: int
    warmup_count: int
    planner_step_count: int
    accepted_planner_output_count: int
    rejected_planner_output_count: int
    deadline_miss_count: int
    # planning = the planner-reported foreground latency (blocked command +
    # gain fetch); step_wall = the same step measured at the bridge, including
    # adapter overhead. prepare/command are the planner-side components.
    last_planning_time_ms: float | None
    last_planner_step_wall_time_ms: float | None
    last_planner_prepare_time_ms: float | None
    last_planner_command_time_ms: float | None
    last_control_prepare_time_ms: float | None
    last_control_publish_time_ms: float | None
    last_phase: str | None
    last_next_phase: str | None
    last_running_cost: float | None
    last_gain_norm: float | None
    last_gain_ess: float | None
    last_gain_nominal_weight: float | None
    last_torque_norm: float | None
    last_position_error: float | None
    last_position_error_signed: list[float] | None
    last_orientation_error: float | None
    last_ee_position: list[float] | None
    # Row-major 3x3 matrices. Keeping matrices avoids quaternion sign
    # ambiguity and Euler-angle singularities in the recorded data.
    last_ee_rotation: list[float] | None
    last_goal_rotation: list[float] | None
    last_object_error: float | None
    last_goal_position: list[float] | None
    last_reference_q: list[float] | None
    last_reference_v: list[float] | None
    last_control_max_abs_feedforward: float | None
    last_control_gain_norm: float | None
    # Compact summary of the final feedback matrix actually published to LFC.
    # The complete 7x14 matrix remains available in the Control/replay stream.
    last_control_position_gain_diagonal: list[float] | None
    last_control_velocity_gain_diagonal: list[float] | None
    last_control_gain_max_abs_off_diagonal: float | None
    last_error: str
    planner_mode: str | None = None
    # Gripper action round-trip (pick-and-place, P3): the last planner
    # gripper command seen by the bridge and the client's state snapshot
    # (goal_count / last_result / failure). None when no gripper is wired
    # or the task never commands one (pregrasp).
    last_gripper_command: str | None = None
    gripper: dict | None = None
    # Pick-and-place event-gate snapshot. None for planners without a phase
    # machine (including the pregrasp OCP).
    phase_machine: dict | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)
