"""Planner adapter for the hydrax Feedback-MPPI backend.

Implements the same duck-typed surface the bridge consumes from
``SbMpcPlannerAdapter``, backed by ``hydrax`` (which must be importable —
the bridge process is launched through ``uv_ros_run.sh`` so the hydrax uv
environment provides it; hydrax is never installed into the sbmpc pixi
env).

Configuration model (see the port plan's "Parameter architecture"): the
OCP tuning (cost weights, solver knobs, plan timing) comes exclusively
from hydrax's single tuning surface, ``hydrax/configs/pregrasp.yaml``,
loaded here through the same ``load_pregrasp_config`` the Tier A example
uses — the ROS parameter layer carries transport and deployment wiring
only (``planner_impl``, ``planner_mode``) and can never override the OCP
values. The controller pairing glue mirrors the Tier A example; the V-B1
contract test asserts the two stay equivalent.

Conventions (audited against the bridge, 2026-07-03):

- ``K`` is expressed as the measured-state Jacobian du/dx, like sbmpc's
  F-MPPI gains: the bridge negates it (``SBMPC_TO_LFC_GAIN_SCALE = -1``)
  before LFC applies ``tau_ff + K_lfc (desired - measured)``. In
  feedforward mode this adapter therefore publishes the constant joint
  impedance as ``K = -[diag(kp) | diag(kd)]``, which LFC receives as
  ``+[Kp | Kd]``.
- ``reference_q``/``reference_v`` are the minimum-jerk plan point at the
  solve time; the bridge substitutes them into the published
  ``control.initial_state`` so LFC's desired state tracks the plan.
- The plan clock is the adapter's own step counter times the control
  period; it starts when the plan starts (after
  ``reset_runtime_state_after_warmup``), mirroring the trajectory mode of
  the sbmpc pregrasp controller.
- ``phase``/``next_phase`` are the constant ``"PREGRASP"``, exactly like
  sbmpc's pregrasp controller; a real phase machine is a planner-side
  concept for later task modes.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from sbmpc_ros_bridge.planner_adapter import (
    PlannerInput,
    configure_jax_compilation_cache,
)

DEFAULT_HYDRAX_JAX_CACHE_DIR = "/workspace/hydrax/.jax_cache"

GAIN_MODE_FEEDFORWARD = "feedforward"


@dataclass(frozen=True)
class HydraxPlannerDiagnostics:
    """Per-step planner report (duck-typed like sbmpc's PlannerDiagnostics)."""

    planning_time_ms: float
    gain_norm: float
    torque_norm: float
    goal_position: np.ndarray
    gain_mode: str
    running_cost: float | None = None
    position_error: float | None = None
    orientation_error: float | None = None
    object_error: float | None = None
    planner_prepare_time_ms: float | None = None
    planner_command_time_ms: float | None = None


@dataclass(frozen=True)
class HydraxPlannerOutput:
    """Duck-typed planner output consumed by the bridge."""

    tau_ff: np.ndarray
    K: np.ndarray
    diagnostics: HydraxPlannerDiagnostics
    reference_q: np.ndarray | None = None
    reference_v: np.ndarray | None = None
    phase: str = "PREGRASP"
    next_phase: str = "PREGRASP"
    gripper_command: Any = None


class HydraxPlannerAdapter:
    """Bridge-facing adapter around the hydrax Panda pregrasp controller."""

    def __init__(
        self,
        *,
        mode: str | None = None,
        config_path: str | None = None,
        jax_cache_dir: str | None = DEFAULT_HYDRAX_JAX_CACHE_DIR,
    ) -> None:
        self._mode = (mode or GAIN_MODE_FEEDFORWARD).strip().lower()
        if self._mode != GAIN_MODE_FEEDFORWARD:
            raise NotImplementedError(
                "the hydrax backend currently supports only feedforward "
                "mode; exact_feedback arrives with the FeedbackMPPI gain "
                "computation (port plan Phase 4)."
            )
        self.jax_cache_dir = configure_jax_compilation_cache(jax_cache_dir)

        try:
            import jax
            from hydrax.algs.feedback_mppi import FeedbackMPPI
            from hydrax.configs import load_pregrasp_config
            from hydrax.tasks.panda_pregrasp import PandaPregrasp
        except ImportError as exc:
            raise RuntimeError(
                "hydrax is not importable. Launch the bridge through "
                "uv_ros_run.sh so the hydrax uv environment provides it."
            ) from exc
        import mujoco
        from mujoco import mjx

        self._mujoco = mujoco
        self._jax = jax

        # OCP tuning from the single tuning surface (same file as Tier A)
        options, config = load_pregrasp_config(config_path)
        self.config = config
        self._task = PandaPregrasp(options=options)
        opts = self._task.options
        tau_max = np.asarray(opts.tau_max, dtype=np.float64)

        # Controller pairing glue (mirrors examples/panda_pregrasp.py)
        self._ctrl = FeedbackMPPI(
            self._task,
            num_samples=config.num_samples,
            noise_std=config.noise_scale * tau_max,
            temperature=config.temperature,
            plan_horizon=config.plan_horizon,
            spline_type=config.spline_type,
            num_knots=config.num_knots,
            iterations=config.iterations,
        )

        # Constant impedance in du/dx convention (see module docstring)
        kp = np.asarray(opts.kp_fixed, dtype=np.float64)
        kd = np.asarray(opts.kd_fixed, dtype=np.float64)
        self._gain_dudx = -np.hstack([np.diag(kp), np.diag(kd)])

        # Reference plan on the numpy side
        self._plan_q = np.asarray(self._task.reference_qpos, dtype=np.float64)
        self._plan_v = np.asarray(self._task.reference_qvel, dtype=np.float64)
        self._goal_pos = np.asarray(opts.goal_pos, dtype=np.float64)
        self._control_period = float(self._task.dt)

        # Warm-startable optimizer state, seeded from the torque plan
        knot_times = np.linspace(
            0.0, self._ctrl.plan_horizon, self._ctrl.num_knots
        )
        knot_idx = np.minimum(
            np.round(knot_times / self._task.dt).astype(int),
            self._plan_q.shape[0] - 1,
        )
        self._initial_knots = self._task.reference_ctrl[knot_idx]
        self._params = self._ctrl.init_params(
            initial_knots=self._initial_knots
        )
        self._jit_optimize = jax.jit(self._ctrl.optimize)
        self._mjx_data = mjx.make_data(self._task.model)
        self._step_index = 0

        # Rollout visualization capture (RViz markers). hydrax computes the
        # gripper-site path of every rollout during optimize
        # (Trajectory.trace_sites), so capture only keeps device references —
        # the host copy happens in the viz getters, off the hot path, at the
        # marker publish rate.
        self._capture_rollouts = False
        self._last_trace_sites = None
        self._last_costs = None

        # CPU-side model for FK / gravity / state prediction (off hot path)
        self._cpu_model = deepcopy(self._task.mj_model)
        self._cpu_model.opt.timestep = 0.001
        self._cpu_data = mujoco.MjData(self._cpu_model)
        self._site_id = mujoco.mj_name2id(
            self._cpu_model, mujoco.mjtObj.mjOBJ_SITE, "gripper"
        )
        self._started = False

    # --- lifecycle -----------------------------------------------------

    def start(self) -> None:
        self._started = True

    def close(self) -> None:
        self._started = False

    def warmup(self, **kwargs: Any) -> HydraxPlannerOutput:
        """Compile/warm the solver from the plan start state.

        Does not advance the plan clock; the bridge calls
        ``reset_runtime_state_after_warmup`` before arming.
        """
        del kwargs
        self.start()
        q = np.asarray(self._task.start_q, dtype=np.float64)
        v = np.zeros_like(q)
        return self._solve(q, v, advance_clock=False)

    def reset_runtime_state_after_warmup(self) -> None:
        """Discard warmup state while retaining compiled JAX executables."""
        self._params = self._ctrl.init_params(
            initial_knots=self._initial_knots
        )
        self._step_index = 0
        self._last_trace_sites = None
        self._last_costs = None
        self._started = False

    # --- control hot path ----------------------------------------------

    def step(
        self, planner_input: PlannerInput, **kwargs: Any
    ) -> HydraxPlannerOutput:
        del kwargs
        self.start()
        return self._solve(
            planner_input.q, planner_input.v, advance_clock=True
        )

    def _solve(
        self,
        q: np.ndarray,
        v: np.ndarray,
        *,
        advance_clock: bool,
    ) -> HydraxPlannerOutput:
        prepare_start = time.perf_counter()
        t = self._step_index * self._control_period
        i_ref = min(self._step_index, self._plan_q.shape[0] - 1)
        state = self._mjx_data.replace(
            qpos=np.asarray(q, dtype=np.float64),
            qvel=np.asarray(v, dtype=np.float64),
            time=t,
        )
        prepare_ms = 1000.0 * (time.perf_counter() - prepare_start)

        command_start = time.perf_counter()
        self._params, rollouts = self._jit_optimize(state, self._params)
        if self._capture_rollouts:
            self._last_trace_sites = rollouts.trace_sites
            self._last_costs = rollouts.costs
        tau_ff = np.asarray(
            self._jax.block_until_ready(
                self._ctrl.get_action(self._params, t)
            ),
            dtype=np.float64,
        )
        command_ms = 1000.0 * (time.perf_counter() - command_start)

        if advance_clock:
            self._step_index += 1

        ee = self.ee_position(q)
        position_error = float(np.linalg.norm(ee - self._goal_pos))

        return HydraxPlannerOutput(
            tau_ff=tau_ff,
            K=self._gain_dudx.copy(),
            reference_q=self._plan_q[i_ref].copy(),
            reference_v=self._plan_v[i_ref].copy(),
            diagnostics=HydraxPlannerDiagnostics(
                planning_time_ms=prepare_ms + command_ms,
                gain_norm=float(np.linalg.norm(self._gain_dudx)),
                torque_norm=float(np.linalg.norm(tau_ff)),
                goal_position=self._goal_pos.copy(),
                gain_mode=self._mode,
                position_error=position_error,
                planner_prepare_time_ms=prepare_ms,
                planner_command_time_ms=command_ms,
            ),
        )

    # --- diagnostics helpers (off the hot path) -------------------------

    def predict_state(
        self,
        planner_input: PlannerInput,
        tau_ff: np.ndarray,
        duration_sec: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        mujoco = self._mujoco
        data = self._cpu_data
        data.qpos[:] = np.asarray(planner_input.q, dtype=np.float64)
        data.qvel[:] = np.asarray(planner_input.v, dtype=np.float64)
        data.ctrl[:] = np.asarray(tau_ff, dtype=np.float64)
        n_steps = max(
            1, int(round(duration_sec / self._cpu_model.opt.timestep))
        )
        for _ in range(n_steps):
            mujoco.mj_step(self._cpu_model, data)
        return data.qpos.copy(), data.qvel.copy()

    def ee_position(self, q: np.ndarray) -> np.ndarray | None:
        mujoco = self._mujoco
        data = self._cpu_data
        data.qpos[:] = np.asarray(q, dtype=np.float64).reshape(-1)
        data.qvel[:] = 0.0
        mujoco.mj_forward(self._cpu_model, data)
        return data.site_xpos[self._site_id].copy()

    def gravity_torques(self, q: np.ndarray) -> np.ndarray | None:
        mujoco = self._mujoco
        data = self._cpu_data
        data.qpos[:] = np.asarray(q, dtype=np.float64).reshape(-1)
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_inverse(self._cpu_model, data)
        return data.qfrc_inverse.copy()

    def diagnostics_snapshot(self) -> Any:
        return None

    def set_rollout_capture_enabled(self, enabled: bool) -> None:
        self._capture_rollouts = bool(enabled)
        if not self._capture_rollouts:
            self._last_trace_sites = None
            self._last_costs = None

    def warmup_rollout_visualization(self, *, max_rollouts: int) -> None:
        """Exercise the viz getters once so their host paths are warm."""
        self.planned_end_effector_path(None)
        self.representative_end_effector_rollouts(
            None, max_rollouts=max_rollouts
        )

    def planned_end_effector_path(
        self, planner_input: PlannerInput | None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """EE path of the warm-started nominal (sample 0) of the last solve.

        The joint path is not tracked in hydrax's rollout traces; only the
        EE path (the part the marker publisher consumes) is returned.
        """
        del planner_input  # the last solve's rollouts already fix the state
        if self._last_trace_sites is None:
            return None
        # trace_sites: (num_samples, H+1, num_trace_sites, 3); site 0 is the
        # gripper. Sample 0 is FeedbackMPPI's zero-noise nominal.
        ee_path = np.asarray(
            self._last_trace_sites[0, :, 0, :], dtype=np.float64
        )
        q_path = np.empty((0, self._plan_q.shape[1]), dtype=np.float64)
        return q_path, ee_path

    def representative_end_effector_rollouts(
        self, planner_input: PlannerInput | None, *, max_rollouts: int
    ) -> np.ndarray | None:
        """Nominal plus lowest-cost sampled EE rollouts from the last solve."""
        del planner_input
        max_rollouts = int(max_rollouts)
        if self._last_trace_sites is None or self._last_costs is None:
            return None
        if max_rollouts <= 0:
            horizon = int(self._last_trace_sites.shape[1])
            return np.empty((0, horizon, 3), dtype=np.float32)

        totals = np.asarray(self._last_costs, dtype=np.float64).sum(axis=1)
        # Sample 0 (the nominal) first, then the lowest-cost others.
        others = np.argsort(totals[1:])[: max(0, max_rollouts - 1)] + 1
        selected = np.concatenate([[0], others]).astype(int)
        traces = np.asarray(
            self._last_trace_sites[selected, :, 0, :], dtype=np.float32
        )
        return traces

    @property
    def mpc_dt(self) -> float | None:
        """Resolved planner step duration [s], for coherence checks."""
        return self._control_period
