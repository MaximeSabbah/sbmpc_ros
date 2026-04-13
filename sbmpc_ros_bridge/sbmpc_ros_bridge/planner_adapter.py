from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from linear_feedback_controller_msgs.msg import Sensor


@dataclass(frozen=True, slots=True)
class PlannerInput:
    """Canonical planner input extracted from an LFC sensor message."""

    sensor: Sensor
    q: np.ndarray
    v: np.ndarray


class SbMpcPlannerAdapter:
    """Thin wrapper around sbmpc's stable public planner API."""

    def __init__(self, controller: Any | None = None) -> None:
        if controller is None:
            controller = self._build_default_controller()
        self._controller = controller

    def warmup(self, **kwargs: Any) -> Any:
        return self._controller.warmup(**kwargs)

    def step(self, planner_input: PlannerInput, **kwargs: Any) -> Any:
        return self._controller.step(planner_input.q, planner_input.v, **kwargs)

    @staticmethod
    def _build_default_controller() -> Any:
        try:
            from sbmpc import PandaPickAndPlaceController
        except ImportError as exc:
            raise RuntimeError(
                "sbmpc is not importable. Install or path-expose the algorithm "
                "repository before using the runtime planner adapter."
            ) from exc
        return PandaPickAndPlaceController()
