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

    def __init__(
        self,
        controller: Any | None = None,
        *,
        warmup_kwargs: dict[str, Any] | None = None,
        step_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if controller is None:
            controller = self._build_default_controller()
        self._controller = controller
        self._warmup_kwargs = (
            self._build_default_step_kwargs()
            if warmup_kwargs is None
            else dict(warmup_kwargs)
        )
        self._step_kwargs = (
            self._build_default_step_kwargs()
            if step_kwargs is None
            else dict(step_kwargs)
        )

    def warmup(self, **kwargs: Any) -> Any:
        call_kwargs = dict(self._warmup_kwargs)
        call_kwargs.update(kwargs)
        return self._controller.warmup(**call_kwargs)

    def step(self, planner_input: PlannerInput, **kwargs: Any) -> Any:
        call_kwargs = dict(self._step_kwargs)
        call_kwargs.update(kwargs)
        return self._controller.step(planner_input.q, planner_input.v, **call_kwargs)

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

    @staticmethod
    def _build_default_step_kwargs() -> dict[str, Any]:
        try:
            from sbmpc.panda_pick_and_place import Phase
        except ImportError:
            return {}
        return {"phase": Phase.PREGRASP}
