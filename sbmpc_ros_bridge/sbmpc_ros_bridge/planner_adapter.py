from __future__ import annotations

from dataclasses import dataclass
import inspect
import os
from pathlib import Path
from typing import Any

import numpy as np
from linear_feedback_controller_msgs.msg import Sensor


DEFAULT_CONTAINER_JAX_CACHE_DIR = Path("/workspace/sbmpc/.jax_cache")


@dataclass(frozen=True, slots=True)
class PlannerInput:
    """Canonical planner input extracted from an LFC sensor message."""

    sensor: Sensor
    q: np.ndarray
    v: np.ndarray


@dataclass(frozen=True, slots=True)
class PlannerConfigOverrides:
    """Optional ROS-side overrides for the public sbmpc planner adapter."""

    mode: str | None = None
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
    gain_samples_per_cycle: int | None = None
    gain_buffer_size: int | None = None
    reseed_every_step: bool | None = None

    def active_items(self) -> dict[str, object]:
        values: dict[str, object] = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if value is not None:
                values[field_name] = value
        return values


def planner_config_overrides_from_values(
    *,
    mode: str | None = None,
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
    gain_samples_per_cycle: int = 0,
    gain_buffer_size: int = 0,
    reseed_every_step: bool | None = None,
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
        mode=_clean_optional_text(mode),
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
        gain_samples_per_cycle=_optional_positive_int(gain_samples_per_cycle),
        gain_buffer_size=_optional_positive_int(gain_buffer_size),
        reseed_every_step=reseed_every_step,
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
    config.MPC.gain_method = "exact"
    if overrides.gain_samples_per_cycle is not None:
        config.MPC.gain_samples_per_cycle = overrides.gain_samples_per_cycle
    if overrides.gain_buffer_size is not None:
        config.MPC.gain_buffer_size = overrides.gain_buffer_size

    if config.MPC.num_control_points > config.MPC.horizon:
        raise ValueError(
            "planner_num_control_points must not exceed planner_horizon after "
            "applying ROS-side overrides."
        )

    config.MPC.initial_guess = _planner_initial_guess(
        planner,
        horizon=config.MPC.horizon,
        dt=config.MPC.dt,
        initial_guess_phase=initial_guess_phase,
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
        reset_runtime_after_warmup: bool = True,
    ) -> None:
        self._config_overrides = (
            PlannerConfigOverrides() if config_overrides is None else config_overrides
        )
        self.jax_cache_dir: str | None = None
        if controller is None:
            self.jax_cache_dir = configure_jax_compilation_cache()
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
        self._reset_runtime_after_warmup = bool(reset_runtime_after_warmup)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        start = getattr(self._controller, "start", None)
        if callable(start):
            start()
        self._started = True

    def close(self) -> None:
        close = getattr(self._controller, "close", None)
        if callable(close):
            close()
        self._started = False

    def warmup(self, **kwargs: Any) -> Any:
        self.start()
        call_kwargs = dict(self._warmup_kwargs)
        call_kwargs.update(kwargs)
        output = self._controller.warmup(**call_kwargs)
        reset_runtime = getattr(
            self._controller,
            "reset_runtime_state_after_warmup",
            None,
        )
        if self._reset_runtime_after_warmup and callable(reset_runtime):
            reset_runtime()
            self._started = False
        return output

    def step(self, planner_input: PlannerInput, **kwargs: Any) -> Any:
        self.start()
        call_kwargs = dict(self._step_kwargs)
        call_kwargs.update(kwargs)
        return self._controller.step(planner_input.q, planner_input.v, **call_kwargs)

    def step_feedforward(self, planner_input: PlannerInput, **kwargs: Any) -> Any:
        self.start()
        call_kwargs = dict(self._step_kwargs)
        call_kwargs.update(kwargs)
        step_feedforward = getattr(self._controller, "step_feedforward", None)
        if callable(step_feedforward):
            return step_feedforward(planner_input.q, planner_input.v, **call_kwargs)
        return self._controller.step(planner_input.q, planner_input.v, **call_kwargs)

    def latest_gain(self, diagnostics: Any | None = None) -> Any | None:
        latest_gain = getattr(self._controller, "latest_gain", None)
        if callable(latest_gain):
            return latest_gain(diagnostics)
        return None

    @property
    def gain_refresh_runs_in_planner_worker(self) -> bool:
        gain_mode = getattr(self._controller, "gain_mode", None)
        return gain_mode == "exact_async_feedback"

    def refresh_gain_if_budget(
        self,
        diagnostics: Any | None = None,
        *,
        budget_sec: float | None = None,
    ) -> Any | None:
        refresh_gain = getattr(self._controller, "refresh_gain_if_budget", None)
        if callable(refresh_gain):
            return refresh_gain(diagnostics, budget_sec=budget_sec)
        return None

    def predict_state(
        self,
        planner_input: PlannerInput,
        tau_ff: np.ndarray,
        duration_sec: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        predict_state = getattr(self._controller, "predict_state", None)
        if predict_state is None:
            return None
        return predict_state(planner_input.q, planner_input.v, tau_ff, duration_sec)

    def diagnostics_snapshot(self) -> Any:
        diagnostics = getattr(self._controller, "diagnostics_snapshot", None)
        if callable(diagnostics):
            return diagnostics()
        return None

    @staticmethod
    def _build_default_controller(config_overrides: PlannerConfigOverrides) -> Any:
        try:
            from sbmpc.examples.franka_emika_panda.planner_api import PandaPregraspController
            from sbmpc.examples.franka_emika_panda.panda_pregrasp import (
                PandaPregraspPlanner,
                make_panda_pregrasp_config,
            )
        except ImportError as exc:
            raise RuntimeError(
                "sbmpc is not importable. Install or path-expose the algorithm "
                "repository before using the runtime planner adapter."
            ) from exc
        planner = PandaPregraspPlanner()
        gains = True if config_overrides.gains is None else config_overrides.gains
        config = make_panda_pregrasp_config(
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
        return PandaPregraspController(
            planner=planner,
            config=config,
            reseed_every_step=(
                True
                if config_overrides.reseed_every_step is None
                else bool(config_overrides.reseed_every_step)
            ),
            gain_mode=config_overrides.mode,
            compute_running_cost=False,
            compute_task_diagnostics=False,
        )

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
            from sbmpc.examples.franka_emika_panda.panda_pick_and_place import Phase
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
                from sbmpc.examples.franka_emika_panda.panda_pick_and_place import Phase
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


def configure_jax_compilation_cache(
    cache_dir: str | os.PathLike[str] | None = None,
) -> str | None:
    """Enable JAX's persistent compilation cache before sbmpc creates jitted functions."""
    requested_cache_dir = _resolve_jax_cache_dir(cache_dir)
    if requested_cache_dir is None:
        return None

    requested_cache_dir.mkdir(parents=True, exist_ok=True)

    import jax

    jax.config.update("jax_compilation_cache_dir", str(requested_cache_dir))
    jax.config.update("jax_enable_compilation_cache", True)
    _jax_config_update_if_present(
        jax.config,
        "jax_persistent_cache_min_compile_time_secs",
        0.0,
    )
    _jax_config_update_if_present(
        jax.config,
        "jax_persistent_cache_min_entry_size_bytes",
        0,
    )
    return str(requested_cache_dir)


def _resolve_jax_cache_dir(cache_dir: str | os.PathLike[str] | None) -> Path | None:
    if cache_dir is not None:
        return _clean_jax_cache_dir(cache_dir)

    env_cache_dir = os.environ.get("SBMPC_JAX_CACHE_DIR")
    if env_cache_dir is not None:
        return _clean_jax_cache_dir(env_cache_dir)

    if DEFAULT_CONTAINER_JAX_CACHE_DIR.parent.is_dir():
        return DEFAULT_CONTAINER_JAX_CACHE_DIR
    return Path.cwd() / ".jax_cache"


def _clean_jax_cache_dir(cache_dir: str | os.PathLike[str]) -> Path | None:
    cache_dir_text = os.fspath(cache_dir).strip()
    if cache_dir_text.lower() in {"", "0", "false", "no", "off", "none"}:
        return None
    return Path(cache_dir_text).expanduser().resolve()


def _jax_config_update_if_present(config: Any, key: str, value: Any) -> None:
    values = getattr(config, "values", {})
    if key in values:
        config.update(key, value)


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


def _planner_initial_guess(
    planner: Any,
    *,
    horizon: int,
    dt: float,
    initial_guess_phase: Any,
) -> Any:
    xp = _array_namespace()
    state = xp.concatenate(
        [
            xp.asarray(planner.home_q, dtype=np.float32),
            xp.zeros(planner.nv, dtype=np.float32),
        ]
    )

    parameters = inspect.signature(planner.nominal_torque_sequence_from_state).parameters
    if len(parameters) >= 4:
        return planner.nominal_torque_sequence_from_state(
            state,
            horizon,
            dt,
            initial_guess_phase,
        )
    return planner.nominal_torque_sequence_from_state(
        state,
        horizon,
        dt,
    )


def _array_namespace():
    try:
        import jax.numpy as jnp

        return jnp
    except ImportError:
        return np
