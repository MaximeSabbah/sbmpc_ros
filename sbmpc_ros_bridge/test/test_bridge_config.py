"""Config-parity guard between the sbmpc OCP yaml and the ROS bridge.

The bridge starts from `make_panda_pregrasp_config(planner, ocp=...)` (the
OCP yaml is the single source of truth for the MPPI knobs) and applies only
the explicitly set ROS overrides on top. This test pins down the invariant
that an empty `PlannerConfigOverrides()` is a strict no-op. If a future
change introduces a silent Config mutation in `apply_config_overrides`, this
test fails and the OCP yaml stops representing what the bridge actually runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sbmpc_ros_bridge.planner_adapter import (
    PlannerConfigOverrides,
    apply_config_overrides,
)


@dataclass
class FakeMPCConfig:
    gains: bool = True
    horizon: int = 8
    num_parallel_computations: int = 1024
    num_control_points: int = 8
    dt: float = 0.02
    lambda_mpc: float = 0.05
    std_dev_mppi: object = field(
        default_factory=lambda: np.full(7, 0.5, dtype=np.float32)
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
        self.home_q = np.zeros(7, dtype=np.float32)
        self.nv = 7
        self.torque_limits = np.full(7, 87.0, dtype=np.float32)


_MPC_FIELDS = (
    "gains",
    "horizon",
    "num_parallel_computations",
    "num_control_points",
    "dt",
    "lambda_mpc",
    "smoothing",
    "gain_method",
    "num_gain_samples",
)


def _snapshot_mpc(mpc: FakeMPCConfig) -> dict:
    snap = {f: getattr(mpc, f) for f in _MPC_FIELDS}
    snap["std_dev_mppi"] = np.asarray(mpc.std_dev_mppi, dtype=np.float32).copy()
    snap["initial_guess"] = np.asarray(mpc.initial_guess, dtype=np.float32).copy()
    return snap


def test_empty_overrides_is_strict_noop() -> None:
    config = FakeConfig()
    planner = FakePlanner()
    before = _snapshot_mpc(config.MPC)

    apply_config_overrides(config, planner, PlannerConfigOverrides())

    after = _snapshot_mpc(config.MPC)
    for f in _MPC_FIELDS:
        assert before[f] == after[f], f"apply_config_overrides mutated MPC.{f}"
    np.testing.assert_array_equal(before["std_dev_mppi"], after["std_dev_mppi"])
    np.testing.assert_array_equal(before["initial_guess"], after["initial_guess"])
