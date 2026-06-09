from __future__ import annotations

from dataclasses import dataclass, field
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sbmpc_ros_bridge.planner_adapter import (
    PlannerConfigOverrides,
    SbMpcPlannerAdapter,
    apply_config_overrides,
    configure_jax_compilation_cache,
    planner_config_overrides_from_values,
)


@dataclass
class FakeMPCConfig:
    gains: bool = True
    horizon: int = 8
    num_parallel_computations: int = 14
    num_control_points: int = 4
    dt: float = 0.02
    lambda_mpc: float = 0.05
    std_dev_mppi: object = field(
        default_factory=lambda: np.ones(7, dtype=np.float32)
    )
    smoothing: str | None = "Spline"
    gain_method: str = "exact"
    num_gain_samples: int | None = 512
    initial_guess: object = field(
        default_factory=lambda: np.zeros((8, 7), dtype=np.float32)
    )


@dataclass
class FakeGeneralConfig:
    dtype: object = np.float32


@dataclass
class FakeConfig:
    MPC: FakeMPCConfig = field(default_factory=FakeMPCConfig)
    general: FakeGeneralConfig = field(default_factory=FakeGeneralConfig)


class FakePlanner:
    def __init__(self) -> None:
        self.home_q = np.asarray([0.1] * 7, dtype=np.float32)
        self.nv = 7
        self.torque_limits = np.asarray([2.0] * 7, dtype=np.float32)
        self.initial_guess_calls: list[dict[str, object]] = []

    def nominal_torque_sequence_from_state(self, state, horizon, dt, phase):
        self.initial_guess_calls.append(
            {
                "state": np.asarray(state, dtype=np.float32),
                "horizon": horizon,
                "dt": dt,
                "phase": phase,
            }
        )
        return np.full((horizon, 7), dt, dtype=np.float32)


class FakePregraspPlanner(FakePlanner):
    def nominal_torque_sequence_from_state(self, state, horizon, dt):
        self.initial_guess_calls.append(
            {
                "state": np.asarray(state, dtype=np.float32),
                "horizon": horizon,
                "dt": float(dt),
            }
        )
        return np.full((horizon, 7), float(dt), dtype=np.float32)


class FakeJaxConfig:
    def __init__(self) -> None:
        self.values = {
            "jax_compilation_cache_dir": None,
            "jax_enable_compilation_cache": True,
            "jax_persistent_cache_min_compile_time_secs": 1.0,
            "jax_persistent_cache_min_entry_size_bytes": 0,
        }
        self.updates: list[tuple[str, object]] = []

    def update(self, key: str, value: object) -> None:
        self.updates.append((key, value))
        self.values[key] = value


class FakeRuntimeController:
    def __init__(self) -> None:
        self.start_count = 0
        self.warmup_calls = []
        self.step_calls = []

    def start(self) -> None:
        self.start_count += 1

    def warmup(self, **kwargs):
        self.warmup_calls.append(kwargs)
        return {"kwargs": kwargs}

    def step(self, q, v, **kwargs):
        self.step_calls.append((q, v, kwargs))
        return {"tau_ff": np.ones(7), "K": np.eye(7, 14), "kwargs": kwargs}


def test_planner_config_overrides_from_values_maps_tuning_inputs_cleanly() -> None:
    overrides = planner_config_overrides_from_values(
        mode=" exact_feedback ",
        phase=" transport ",
        gains=False,
        num_steps=3,
        num_samples=128,
        horizon=24,
        num_control_points=6,
        temperature=0.15,
        dt=0.05,
        noise_scale=0.08,
        smoothing="none",
        num_gain_samples=64,
    )

    assert overrides == PlannerConfigOverrides(
        mode="exact_feedback",
        phase="transport",
        gains=False,
        num_steps=3,
        horizon=24,
        num_parallel_computations=128,
        num_control_points=6,
        dt=0.05,
        lambda_mpc=0.15,
        std_dev_scale=0.08,
        smoothing="__none__",
        num_gain_samples=64,
    )


def test_planner_config_overrides_carry_compute_task_diagnostics() -> None:
    assert (
        planner_config_overrides_from_values().compute_task_diagnostics is None
    )
    assert (
        planner_config_overrides_from_values(
            compute_task_diagnostics=True
        ).compute_task_diagnostics
        is True
    )


def test_planner_config_overrides_support_legacy_aliases_for_existing_config_names() -> None:
    overrides = planner_config_overrides_from_values(
        num_steps=2,
        num_parallel_computations=96,
        lambda_mpc=0.07,
        std_dev_scale=0.03,
    )

    assert overrides.num_steps == 2
    assert overrides.num_parallel_computations == 96
    assert np.isclose(overrides.lambda_mpc, 0.07)
    assert np.isclose(overrides.std_dev_scale, 0.03)


def test_apply_config_overrides_updates_core_mppi_settings_and_initial_guess() -> None:
    config = FakeConfig()
    planner = FakePlanner()
    overrides = PlannerConfigOverrides(
        gains=False,
        horizon=12,
        num_parallel_computations=64,
        num_control_points=3,
        dt=0.04,
        lambda_mpc=0.1,
        std_dev_scale=0.07,
        smoothing="__none__",
        num_gain_samples=32,
    )

    updated_config = apply_config_overrides(
        config,
        planner,
        overrides,
        initial_guess_phase="PREGRASP",
    )

    assert updated_config.MPC.gains is False
    assert updated_config.MPC.horizon == 12
    assert updated_config.MPC.num_parallel_computations == 64
    assert updated_config.MPC.num_control_points == 3
    assert np.isclose(updated_config.MPC.dt, 0.04)
    assert np.isclose(updated_config.MPC.lambda_mpc, 0.1)
    np.testing.assert_allclose(
        np.asarray(updated_config.MPC.std_dev_mppi, dtype=np.float32),
        np.asarray([0.14] * 7, dtype=np.float32),
    )
    assert updated_config.MPC.smoothing is None
    assert updated_config.MPC.gain_method == "exact"
    assert updated_config.MPC.num_gain_samples == 32
    assert np.asarray(updated_config.MPC.initial_guess).shape == (12, 7)
    np.testing.assert_allclose(
        np.asarray(updated_config.MPC.initial_guess, dtype=np.float32),
        np.zeros((12, 7), dtype=np.float32),
    )
    assert planner.initial_guess_calls == []


def test_apply_config_overrides_rejects_control_points_above_horizon() -> None:
    with pytest.raises(ValueError, match="must not exceed planner_horizon"):
        apply_config_overrides(
            FakeConfig(),
            FakePlanner(),
            PlannerConfigOverrides(horizon=3, num_control_points=4),
            initial_guess_phase="PREGRASP",
        )


def test_apply_config_overrides_uses_neutral_initial_guess() -> None:
    config = FakeConfig()
    planner = FakePregraspPlanner()

    updated_config = apply_config_overrides(
        config,
        planner,
        PlannerConfigOverrides(horizon=5, dt=0.03),
        initial_guess_phase="PREGRASP",
    )

    assert np.asarray(updated_config.MPC.initial_guess).shape == (5, 7)
    np.testing.assert_allclose(
        np.asarray(updated_config.MPC.initial_guess, dtype=np.float32),
        np.zeros((5, 7), dtype=np.float32),
    )
    assert planner.initial_guess_calls == []


def test_default_adapter_rejects_per_step_trajectory_reseed() -> None:
    with pytest.raises(ValueError, match="planner_reseed_every_step"):
        SbMpcPlannerAdapter._build_default_controller(
            PlannerConfigOverrides(reseed_every_step=True)
        )


def test_configure_jax_compilation_cache_enables_persistent_cache(
    tmp_path,
    monkeypatch,
) -> None:
    fake_config = FakeJaxConfig()
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(config=fake_config))

    cache_dir = configure_jax_compilation_cache(tmp_path / "jax-cache")

    assert cache_dir == str(tmp_path / "jax-cache")
    assert (tmp_path / "jax-cache").is_dir()
    assert fake_config.updates == [
        ("jax_compilation_cache_dir", str(tmp_path / "jax-cache")),
        ("jax_enable_compilation_cache", True),
        ("jax_persistent_cache_min_compile_time_secs", 0.0),
        ("jax_persistent_cache_min_entry_size_bytes", 0),
    ]


def test_configure_jax_compilation_cache_allows_env_disable(monkeypatch) -> None:
    monkeypatch.setenv("SBMPC_JAX_CACHE_DIR", "off")
    monkeypatch.delitem(sys.modules, "jax", raising=False)

    assert configure_jax_compilation_cache() is None


def test_adapter_preserves_warmup_state_and_delegates_coherent_steps() -> None:
    controller = FakeRuntimeController()
    adapter = SbMpcPlannerAdapter(controller=controller)
    planner_input = SimpleNamespace(
        q=np.asarray([0.1] * 7, dtype=np.float64),
        v=np.zeros(7, dtype=np.float64),
    )

    warmup = adapter.warmup(num_steps=3)
    output = adapter.step(planner_input, num_steps=2)

    assert warmup["kwargs"]["num_steps"] == 3
    assert output["kwargs"]["num_steps"] == 2
    assert controller.start_count == 1
    assert adapter._started is True
    assert len(controller.warmup_calls) == 1
    assert len(controller.step_calls) == 1
    np.testing.assert_allclose(controller.step_calls[0][0], planner_input.q)
    np.testing.assert_allclose(controller.step_calls[0][1], planner_input.v)
    np.testing.assert_allclose(output["K"], np.eye(7, 14))
