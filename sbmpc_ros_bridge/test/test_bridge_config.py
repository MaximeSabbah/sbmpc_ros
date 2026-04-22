"""Config-parity guard between the MuJoCo bench and the ROS bridge.

The MuJoCo bench (`sbmpc/tests/bench_lfc.py`) and the ROS bridge both start
from `make_panda_pregrasp_config(planner, visualize=False, gains=True)` and
apply overrides on top. This test pins down the invariant that an empty
`PlannerConfigOverrides()` is a no-op (except for `initial_guess`, which is
always recomputed so the shape matches the horizon). If a future change
introduces a silent Config mutation in `apply_config_overrides`, this test
fails and the MuJoCo baseline stops representing what the bridge actually
runs.
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
    gain_fd_epsilon: float = 1e-3
    gain_fd_scheme: str = "forward"
    gain_fd_num_samples: int | None = 256
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
        self.torque_limits = np.full(7, 87.0, dtype=np.float32)

    def nominal_torque_sequence_from_state(self, state, horizon, dt, phase):
        return np.zeros((horizon, 7), dtype=np.float32)


_MPC_FIELDS = (
    "gains",
    "horizon",
    "num_parallel_computations",
    "num_control_points",
    "dt",
    "lambda_mpc",
    "smoothing",
    "gain_method",
    "gain_fd_epsilon",
    "gain_fd_scheme",
    "gain_fd_num_samples",
)


def _snapshot_mpc(mpc: FakeMPCConfig) -> dict:
    snap = {f: getattr(mpc, f) for f in _MPC_FIELDS}
    snap["std_dev_mppi"] = np.asarray(mpc.std_dev_mppi, dtype=np.float32).copy()
    return snap


def test_empty_overrides_is_noop_on_mpc_fields() -> None:
    config = FakeConfig()
    planner = FakePlanner()
    before = _snapshot_mpc(config.MPC)

    apply_config_overrides(
        config,
        planner,
        PlannerConfigOverrides(),
        initial_guess_phase=None,
    )

    after = _snapshot_mpc(config.MPC)
    for f in _MPC_FIELDS:
        assert before[f] == after[f], f"apply_config_overrides mutated MPC.{f}"
    np.testing.assert_array_equal(before["std_dev_mppi"], after["std_dev_mppi"])


def test_empty_overrides_still_recomputes_initial_guess() -> None:
    config = FakeConfig()
    planner = FakePlanner()

    apply_config_overrides(
        config,
        planner,
        PlannerConfigOverrides(),
        initial_guess_phase=None,
    )

    guess = np.asarray(config.MPC.initial_guess)
    assert guess.shape == (config.MPC.horizon, 7)
