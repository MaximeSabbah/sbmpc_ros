from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True, slots=True)
class BridgeDiagnostics:
    state: str
    valid_sensor_count: int
    rejected_sensor_count: int
    published_control_count: int
    nonzero_control_count: int
    warmup_count: int
    planner_step_count: int
    deadline_miss_count: int
    last_planning_time_ms: float | None
    last_error: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)
