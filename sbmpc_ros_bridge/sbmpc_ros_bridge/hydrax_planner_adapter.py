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
- In **exact_feedback** mode the published K is the solve's F-MPPI gain
  (``FeedbackMPPIParams.gains``), and the anchor changes meaning: the
  bridge's reference substitution is gated on
  ``gain_mode == "feedforward"``, so ``control.initial_state`` stays the
  measured snapshot the solve started from — x₀, the linearization
  point. LFC then applies ``tau_ff + (-K)(x₀ - x) = tau_ff + K (x - x₀)``,
  the Feedback-MPPI law: K multiplies only the drift since planning
  (the tracking error is already baked into tau_ff, which replans from
  x₀ at 25 Hz). The K anchored at the *reference* would double-count
  the tracking error — that is why the anchor differs between modes.
- ``reference_q``/``reference_v`` are the minimum-jerk plan point at the
  solve time; in feedforward mode the bridge substitutes them into the
  published ``control.initial_state`` so LFC's desired state tracks the
  plan (in exact_feedback they remain diagnostics only).
- The plan clock is the adapter's own step counter times the control
  period; it starts when the plan starts (after
  ``reset_runtime_state_after_warmup``), mirroring the trajectory mode of
  the sbmpc pregrasp controller.
- ``phase``/``next_phase`` are the constant ``"PREGRASP"``, exactly like
  sbmpc's pregrasp controller; a real phase machine is a planner-side
  concept for later task modes.

Pick-and-place (P3, ``ocp="pick_place"``; see hydrax
doc/pick_place_plan.md). Same solver, same bridge contract; three things
change, all mirroring the Tier A example (examples/panda_pick_place.py):

- the plan clock is hydrax's ``PickPlacePhaseMachine``, stepped with the
  measured state before each solve (``state.time = pm.plan_time``); the
  clock refuses to cross segment boundaries until the arm converges, so
  phase/next_phase and ``gripper_command`` ("open"/"close") come from the
  machine and change at runtime.
- the deployed law flips PER CYCLE through the existing gain_mode anchor
  semantics: motion cycles publish the solve's F-MPPI K anchored at x₀
  (``exact_feedback``), precision-hold cycles (the CLOSE/OPEN dwells and
  the waits at their entry) publish the constant impedance with
  ``gain_mode="feedforward"`` and ``reference_q/v`` = the dwell goal — the
  bridge's reference substitution then anchors LFC on the stationary
  reference, which is exactly the P-A2-certified settle-then-actuate law.
  No bridge change is involved.
- while the bridge reports a gripper action in flight
  (``set_gripper_wait(True)``) the clock is not stepped: mid-dwell the
  reference is constant, so the pause is indistinguishable from a longer
  dwell (the plan's action-latency liveness fix, user-approved
  2026-07-10).
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from sbmpc_ros_bridge.planner_adapter import (
    PlannerInput,
    configure_jax_compilation_cache,
)

DEFAULT_HYDRAX_JAX_CACHE_DIR = "/workspace/hydrax/.jax_cache"

GAIN_MODE_FEEDFORWARD = "feedforward"
GAIN_MODE_EXACT_FEEDBACK = "exact_feedback"

OCP_PREGRASP = "pregrasp"
OCP_PICK_PLACE = "pick_place"


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
    # V-A3 gain-health readouts (exact_feedback only): effective sample
    # size of the gain-batch softmax and the zero-noise nominal's weight
    gain_ess: float | None = None
    gain_nominal_weight: float | None = None
    # Pick-and-place only: exact event-gate inputs and the resulting decision.
    # Values are Python-native for direct JSON diagnostics publication.
    phase_machine: dict[str, object] | None = None


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
        ocp: str | None = None,
        config_path: str | None = None,
        jax_cache_dir: str | None = DEFAULT_HYDRAX_JAX_CACHE_DIR,
    ) -> None:
        self._mode = (mode or GAIN_MODE_FEEDFORWARD).strip().lower()
        if self._mode not in (GAIN_MODE_FEEDFORWARD, GAIN_MODE_EXACT_FEEDBACK):
            raise ValueError(
                f"planner_mode must be '{GAIN_MODE_FEEDFORWARD}' or "
                f"'{GAIN_MODE_EXACT_FEEDBACK}', got {self._mode!r}."
            )
        self._exact_feedback = self._mode == GAIN_MODE_EXACT_FEEDBACK
        self._ocp = (ocp or OCP_PREGRASP).strip().lower()
        if self._ocp not in (OCP_PREGRASP, OCP_PICK_PLACE):
            raise ValueError(
                f"planner_ocp must be '{OCP_PREGRASP}' or "
                f"'{OCP_PICK_PLACE}', got {self._ocp!r}."
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

        # OCP tuning from the single tuning surface (same file as Tier A):
        # pregrasp.yaml or pick_place.yaml, selected by planner_ocp. The
        # pick-and-place additionally owns the event-gated plan clock (see
        # the module docstring); the pregrasp keeps the step-counter clock.
        if self._ocp == OCP_PICK_PLACE:
            from hydrax.configs import load_pick_place_config
            from hydrax.tasks.panda_pick_place import (
                PandaPickPlace,
                Phase,
                PickPlacePhaseMachine,
            )

            options, config = load_pick_place_config(config_path)
            self._task = PandaPickPlace(options=options)
            self._phase_enum = Phase
            self._phase_machine_cls = PickPlacePhaseMachine
            self._phase_machine = PickPlacePhaseMachine(self._task)
        else:
            options, config = load_pregrasp_config(config_path)
            self._task = PandaPregrasp(options=options)
            self._phase_machine = None
        self.config = config
        self._gripper_wait = False
        opts = self._task.options
        tau_max = np.asarray(opts.tau_max, dtype=np.float64)

        # Controller pairing glue (mirrors examples/panda_pregrasp.py).
        # The mode switch is the K source and nothing else: gains are
        # compiled into the solve only when they will be published, so the
        # feedforward mode pays nothing for the gain path.
        self._ctrl = FeedbackMPPI(
            self._task,
            num_samples=config.num_samples,
            noise_std=config.noise_scale * tau_max,
            temperature=config.temperature,
            mean_adaptation_rate=config.mean_adaptation_rate,
            num_gain_samples=config.num_gain_samples,
            compute_gains=self._exact_feedback,
            plan_horizon=config.plan_horizon,
            spline_type=config.spline_type,
            num_knots=config.num_knots,
            iterations=config.iterations,
        )

        # Constant impedance in du/dx convention (see module docstring)
        kp = np.asarray(opts.kp_fixed, dtype=np.float64)
        kd = np.asarray(opts.kd_fixed, dtype=np.float64)
        self._gain_dudx = -np.hstack([np.diag(kp), np.diag(kd)])

        # Reference plan on the numpy side. The diagnostics goal is the
        # task's terminal EE target: the pregrasp goal pose, or the
        # pick-and-place placement target.
        self._plan_q = np.asarray(self._task.reference_qpos, dtype=np.float64)
        self._plan_v = np.asarray(self._task.reference_qvel, dtype=np.float64)
        self._goal_pos = np.asarray(
            opts.target_pos if self._ocp == OCP_PICK_PLACE else opts.goal_pos,
            dtype=np.float64,
        )
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
        # Jitted spline query: eagerly it costs ~0.6 ms on the 25 Hz hot
        # path (measured Phase 1.5), jitted ~0.2 ms; warmup compiles it.
        self._jit_get_action = jax.jit(self._ctrl.get_action)
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
        self._phase_goal_ee_positions = None
        self._phase_goal_ee_rotations = None
        if self._phase_machine is not None:
            # Phase goals are constant. Cache their FK once so diagnostics do
            # not add another mj_forward call to the 25 Hz control path.
            phase_goal_poses = [
                self._ee_pose(q) for q in self._task.phase_goal_q
            ]
            self._phase_goal_ee_positions = np.stack(
                [pose[0] for pose in phase_goal_poses]
            )
            self._phase_goal_ee_rotations = np.stack(
                [pose[1] for pose in phase_goal_poses]
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
        if self._phase_machine is not None:
            self._phase_machine = self._phase_machine_cls(self._task)
        self._gripper_wait = False
        self._last_trace_sites = None
        self._last_costs = None
        self._started = False

    def set_gripper_wait(self, wait: bool) -> None:
        """Freeze the plan clock while a gripper action is in flight (P3).

        Called by the bridge around its gripper action client. Mid-dwell
        the reference is constant, so not stepping the phase machine is
        indistinguishable from a longer dwell — the certified task layer
        needs no result-gating of its own. No-op for the pregrasp.
        """
        self._gripper_wait = bool(wait)

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
        q_host = np.asarray(q, dtype=np.float64).reshape(-1)
        v_host = np.asarray(v, dtype=np.float64).reshape(-1)
        phase_machine_diagnostics = None
        if self._phase_machine is None:
            t = self._step_index * self._control_period
            i_ref = min(self._step_index, self._plan_q.shape[0] - 1)
        else:
            # Event-gated plan clock (Tier A ordering: update with the
            # measured state, then solve at pm.plan_time). The clock is
            # frozen while the bridge waits on a gripper action result.
            if advance_clock and not self._gripper_wait:
                self._phase_machine.update(q_host, v_host)
            t = self._phase_machine.plan_time
            i_ref = min(
                int(t * self._task.reference_fps), self._plan_q.shape[0] - 1
            )
            phase_machine_diagnostics = asdict(
                self._phase_machine.diagnostics_snapshot(q_host, v_host)
            )
            phase_machine_diagnostics.update(
                clock_paused=bool(self._gripper_wait),
                clock_pause_reason=(
                    "gripper_action" if self._gripper_wait else None
                ),
            )
        state = self._mjx_data.replace(
            qpos=q_host,
            qvel=v_host,
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
                self._jit_get_action(self._params, t)
            ),
            dtype=np.float64,
        )
        command_ms = 1000.0 * (time.perf_counter() - command_start)

        if advance_clock:
            self._step_index += 1

        # The published K: the solve's F-MPPI gains (du/dx, anchored at
        # this solve's state) in exact_feedback, the constant impedance in
        # feedforward. See the module docstring for the anchor semantics.
        # Pick-and-place precision-hold cycles publish the impedance under
        # gain_mode="feedforward" so the bridge anchors LFC on the
        # (stationary) dwell reference — the P-A2-certified dwell law.
        hold = (
            self._phase_machine is not None
            and self._phase_machine.precision_hold
        )
        if self._exact_feedback:
            gain_ess = float(self._params.gain_ess)
            gain_nominal_weight = float(self._params.gain_nominal_weight)
        else:
            gain_ess = None
            gain_nominal_weight = None
        if self._exact_feedback and not hold:
            gain_dudx = np.asarray(self._params.gains, dtype=np.float64)
            gain_mode = GAIN_MODE_EXACT_FEEDBACK
        else:
            gain_dudx = self._gain_dudx.copy()
            gain_mode = GAIN_MODE_FEEDFORWARD

        if self._phase_machine is None:
            phase_name = next_phase_name = "PREGRASP"
            gripper_command = None
        else:
            phase = self._phase_machine.phase
            phase_name = phase.name
            next_phase_name = self._phase_enum(
                min(phase + 1, self._phase_enum.DONE)
            ).name
            gripper_command = (
                "close" if self._phase_machine.gripper_closed else "open"
            )

        # This reuses the single CPU FK that was already performed here for
        # terminal-target diagnostics; reading site_xmat adds no FK call.
        ee, ee_rotation = self._ee_pose(q_host)
        position_error = float(np.linalg.norm(ee - self._goal_pos))
        if phase_machine_diagnostics is not None:
            phase = self._phase_machine.phase
            goal_phase = min(phase, self._phase_enum.RETREAT)
            phase_goal_ee = self._phase_goal_ee_positions[goal_phase]
            phase_goal_rotation = self._phase_goal_ee_rotations[goal_phase]
            ee_error = ee - phase_goal_ee
            orientation_error = self._rotation_error_angle(
                ee_rotation, phase_goal_rotation
            )
            phase_machine_diagnostics.update(
                ee_source="planning_model_fk",
                ee_position_m=ee.tolist(),
                ee_goal_position_m=phase_goal_ee.tolist(),
                ee_position_error_signed_m=ee_error.tolist(),
                ee_position_error_norm_m=float(np.linalg.norm(ee_error)),
                ee_orientation_error_rad=orientation_error,
            )

        return HydraxPlannerOutput(
            tau_ff=tau_ff,
            K=gain_dudx,
            reference_q=self._plan_q[i_ref].copy(),
            reference_v=self._plan_v[i_ref].copy(),
            phase=phase_name,
            next_phase=next_phase_name,
            gripper_command=gripper_command,
            diagnostics=HydraxPlannerDiagnostics(
                planning_time_ms=prepare_ms + command_ms,
                gain_norm=float(np.linalg.norm(gain_dudx)),
                torque_norm=float(np.linalg.norm(tau_ff)),
                goal_position=self._goal_pos.copy(),
                gain_mode=gain_mode,
                position_error=position_error,
                planner_prepare_time_ms=prepare_ms,
                planner_command_time_ms=command_ms,
                gain_ess=gain_ess,
                gain_nominal_weight=gain_nominal_weight,
                phase_machine=phase_machine_diagnostics,
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

    def _ee_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return gripper position and rotation from one CPU FK call."""
        mujoco = self._mujoco
        data = self._cpu_data
        data.qpos[:] = np.asarray(q, dtype=np.float64).reshape(-1)
        data.qvel[:] = 0.0
        mujoco.mj_forward(self._cpu_model, data)
        return (
            data.site_xpos[self._site_id].copy(),
            data.site_xmat[self._site_id].reshape(3, 3).copy(),
        )

    @staticmethod
    def _rotation_error_angle(
        rotation: np.ndarray, goal_rotation: np.ndarray
    ) -> float:
        """Shortest orientation error angle between two rotation matrices."""
        relative = goal_rotation.T @ rotation
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.arccos(cosine))

    def ee_position(self, q: np.ndarray) -> np.ndarray | None:
        return self._ee_pose(q)[0]

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
