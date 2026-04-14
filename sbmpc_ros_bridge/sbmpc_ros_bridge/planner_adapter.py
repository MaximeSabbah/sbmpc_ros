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


@dataclass(frozen=True, slots=True)
class PlannerConfigOverrides:
    """Optional ROS-side overrides for the public sbmpc planner adapter."""

    phase: str | None = None
    gains: bool | None = None
    num_steps: int | None = None
    horizon: int | None = None
    num_parallel_computations: int | None = None
    num_control_points: int | None = None
    dt: float | None = None
    lambda_mpc: float | None = None
    std_dev_scale: float | None = None
    smoothing: str | None = None
    gain_method: str | None = None
    gain_fd_epsilon: float | None = None
    gain_fd_scheme: str | None = None

    def active_items(self) -> dict[str, object]:
        values: dict[str, object] = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if value is not None:
                values[field_name] = value
        return values


def planner_config_overrides_from_values(
    *,
    phase: str | None = None,
    gains: bool | None = None,
    num_steps: int = 0,
    num_samples: int = 0,
    horizon: int = 0,
    num_parallel_computations: int = 0,
    num_control_points: int = 0,
    temperature: float = 0.0,
    dt: float = 0.0,
    lambda_mpc: float = 0.0,
    noise_scale: float = 0.0,
    std_dev_scale: float = 0.0,
    smoothing: str | None = None,
    gain_method: str | None = None,
    gain_fd_epsilon: float = 0.0,
    gain_fd_scheme: str | None = None,
) -> PlannerConfigOverrides:
    sample_count = _optional_positive_int(num_samples)
    if sample_count is None:
        sample_count = _optional_positive_int(num_parallel_computations)

    effective_temperature = _optional_positive_float(temperature)
    if effective_temperature is None:
        effective_temperature = _optional_positive_float(lambda_mpc)

    effective_noise_scale = _optional_positive_float(noise_scale)
    if effective_noise_scale is None:
        effective_noise_scale = _optional_positive_float(std_dev_scale)

    return PlannerConfigOverrides(
        phase=_clean_optional_text(phase),
        gains=gains,
        num_steps=_optional_positive_int(num_steps),
        horizon=_optional_positive_int(horizon),
        num_parallel_computations=sample_count,
        num_control_points=_optional_positive_int(num_control_points),
        dt=_optional_positive_float(dt),
        lambda_mpc=effective_temperature,
        std_dev_scale=effective_noise_scale,
        smoothing=_normalize_smoothing_value(smoothing),
        gain_method=_clean_optional_text(gain_method),
        gain_fd_epsilon=_optional_positive_float(gain_fd_epsilon),
        gain_fd_scheme=_clean_optional_text(gain_fd_scheme),
    )


def apply_config_overrides(
    config: Any,
    planner: Any,
    overrides: PlannerConfigOverrides,
    *,
    initial_guess_phase: Any,
) -> Any:
    xp = _array_namespace()
    if overrides.gains is not None:
        config.MPC.gains = overrides.gains
    if overrides.horizon is not None:
        config.MPC.horizon = overrides.horizon
    if overrides.num_parallel_computations is not None:
        config.MPC.num_parallel_computations = overrides.num_parallel_computations
    if overrides.num_control_points is not None:
        config.MPC.num_control_points = overrides.num_control_points
    if overrides.dt is not None:
        config.MPC.dt = overrides.dt
    if overrides.lambda_mpc is not None:
        config.MPC.lambda_mpc = overrides.lambda_mpc
    if overrides.std_dev_scale is not None:
        config.MPC.std_dev_mppi = (
            xp.asarray(overrides.std_dev_scale, dtype=getattr(config.general, "dtype", np.float32))
            * planner.torque_limits
        )
    if overrides.smoothing is not None:
        config.MPC.smoothing = None if overrides.smoothing == "__none__" else overrides.smoothing
    if overrides.gain_method is not None:
        config.MPC.gain_method = overrides.gain_method
    if overrides.gain_fd_epsilon is not None:
        config.MPC.gain_fd_epsilon = overrides.gain_fd_epsilon
    if overrides.gain_fd_scheme is not None:
        config.MPC.gain_fd_scheme = overrides.gain_fd_scheme

    if config.MPC.num_control_points > config.MPC.horizon:
        raise ValueError(
            "planner_num_control_points must not exceed planner_horizon after "
            "applying ROS-side overrides."
        )

    config.MPC.initial_guess = planner.nominal_torque_sequence_from_state(
        xp.concatenate(
            [
                xp.asarray(planner.home_q, dtype=np.float32),
                xp.zeros(planner.nv, dtype=np.float32),
            ]
        ),
        config.MPC.horizon,
        config.MPC.dt,
        initial_guess_phase,
    )
    return config


class SbMpcPlannerAdapter:
    """Thin wrapper around sbmpc's stable public planner API."""

    def __init__(
        self,
        controller: Any | None = None,
        *,
        config_overrides: PlannerConfigOverrides | None = None,
        warmup_kwargs: dict[str, Any] | None = None,
        step_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._config_overrides = (
            PlannerConfigOverrides() if config_overrides is None else config_overrides
        )
        if controller is None:
            controller = self._build_default_controller(self._config_overrides)
        self._controller = controller
        default_step_kwargs = self._build_step_kwargs(
            phase_name=self._config_overrides.phase,
            num_steps=self._config_overrides.num_steps,
        )
        self._warmup_kwargs = (
            default_step_kwargs
            if warmup_kwargs is None
            else dict(warmup_kwargs)
        )
        self._step_kwargs = (
            default_step_kwargs
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
    def _build_default_controller(config_overrides: PlannerConfigOverrides) -> Any:
        try:
            from sbmpc import PandaPickAndPlaceController
            from sbmpc.panda_pick_and_place import (
                PandaPickAndPlacePlanner,
                make_panda_pick_and_place_config,
            )
        except ImportError as exc:
            raise RuntimeError(
                "sbmpc is not importable. Install or path-expose the algorithm "
                "repository before using the runtime planner adapter."
            ) from exc
        planner = PandaPickAndPlacePlanner()
        gains = True if config_overrides.gains is None else config_overrides.gains
        config = make_panda_pick_and_place_config(
            planner,
            visualize=False,
            gains=gains,
        )
        phase = SbMpcPlannerAdapter._resolve_phase(config_overrides.phase)
        config = apply_config_overrides(
            config,
            planner,
            config_overrides,
            initial_guess_phase=phase,
        )
        return PandaPickAndPlaceController(planner=planner, config=config)

    @staticmethod
    def _build_default_step_kwargs(phase_name: str | None) -> dict[str, Any]:
        return SbMpcPlannerAdapter._build_step_kwargs(phase_name=phase_name)

    @staticmethod
    def _build_step_kwargs(
        *,
        phase_name: str | None,
        num_steps: int | None = None,
    ) -> dict[str, Any]:
        try:
            from sbmpc.panda_pick_and_place import Phase
        except ImportError:
            kwargs: dict[str, Any] = {}
            if num_steps is not None:
                kwargs["num_steps"] = num_steps
            return kwargs
        kwargs = {"phase": SbMpcPlannerAdapter._resolve_phase(phase_name, Phase)}
        if num_steps is not None:
            kwargs["num_steps"] = num_steps
        return kwargs

    @staticmethod
    def _resolve_phase(
        phase_name: str | None,
        phase_enum: Any | None = None,
    ) -> Any:
        try:
            Phase = phase_enum
            if Phase is None:
                from sbmpc.panda_pick_and_place import Phase
        except ImportError as exc:
            raise RuntimeError(
                "sbmpc is not importable, so the requested planner phase cannot be "
                "resolved."
            ) from exc
        if phase_name is None:
            return Phase.PREGRASP
        try:
            return Phase[phase_name.strip().upper()]
        except KeyError as exc:
            valid_names = ", ".join(member.name for member in Phase)
            raise ValueError(
                f"Unsupported planner phase '{phase_name}'. Choose from: {valid_names}."
            ) from exc


def _optional_positive_int(value: int) -> int | None:
    value = int(value)
    return value if value > 0 else None


def _optional_positive_float(value: float) -> float | None:
    value = float(value)
    return value if value > 0.0 else None


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalize_smoothing_value(value: str | None) -> str | None:
    cleaned = _clean_optional_text(value)
    if cleaned is None:
        return None
    if cleaned.lower() in {"none", "null", "off"}:
        return "__none__"
    return cleaned


def _array_namespace():
    try:
        import jax.numpy as jnp

        return jnp
    except ImportError:
        return np
