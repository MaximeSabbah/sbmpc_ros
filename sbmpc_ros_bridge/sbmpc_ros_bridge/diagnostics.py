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
    last_planning_time_ms: float | None
    last_planner_output_time_ms: float | None
    last_bridge_loop_time_ms: float | None
    last_planner_step_wall_time_ms: float | None
    last_planner_step_overhead_time_ms: float | None
    last_planner_loop_residual_time_ms: float | None
    last_planner_api_wall_time_ms: float | None
    last_planner_bridge_adapter_overhead_time_ms: float | None
    last_planner_prepare_time_ms: float | None
    last_planner_command_time_ms: float | None
    last_planner_tau_extract_time_ms: float | None
    last_planner_gain_fetch_time_ms: float | None
    last_planner_task_diagnostics_time_ms: float | None
    last_planner_output_build_time_ms: float | None
    last_control_prepare_time_ms: float | None
    last_control_publish_time_ms: float | None
    last_phase: str | None
    last_next_phase: str | None
    last_running_cost: float | None
    last_gain_norm: float | None
    last_torque_norm: float | None
    last_position_error: float | None
    last_orientation_error: float | None
    last_object_error: float | None
    last_goal_position: list[float] | None
    last_control_max_abs_feedforward: float | None
    last_control_gain_norm: float | None
    last_error: str
    planner_mode: str | None = None
    last_foreground_planning_time_ms: float | None = None
    last_background_gain_time_ms: float | None = None
    last_background_gain_wall_time_ms: float | None = None
    last_gain_subset_select_time_ms: float | None = None
    last_gain_snapshot_pack_time_ms: float | None = None
    last_gain_gradient_time_ms: float | None = None
    last_gain_synthesis_time_ms: float | None = None
    last_gain_age_cycles: float | None = None
    last_gain_window_fill: int | None = None
    last_gain_completed_batch_count: int | None = None
    last_gain_dropped_snapshot_count: int | None = None
    last_gain_worker_running: bool | None = None
    last_gain_worker_error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)
