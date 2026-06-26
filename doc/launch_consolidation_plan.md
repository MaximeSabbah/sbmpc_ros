# SB-MPC ROS Bringup Consolidation — Implementation Plan

> Canonical, step-by-step plan for reworking `sbmpc_bringup` (and a small, well-scoped
> change to `sbmpc_ros_bridge`). Written so it can be followed after a context compaction
> and reviewed change-by-change. **Do not deviate without updating this document.**

Status legend per step: `[ ]` pending · `[~]` in progress · `[x]` done & reviewed.

---

## 1. Goal

One bringup launch that serves **both** the MuJoCo simulation and the real Franka, with the
simulation behaving as close to the real hardware as physically possible, a minimal argument
surface, and none of the accumulated cruft. The simulation and real paths must differ **only**
where physics forces them to.

### Non-negotiable principles
- **Sim ≈ real.** Any divergence between backends must have a physical justification (HW plugin,
  gravity compensation, the inability to teleport the real robot). No accidental divergence.
- **Minimal arguments.** Anything that is not a knob a human turns at launch time lives in a
  config file, a constant, or an environment variable — not in the launch interface.
- **No software "safety" theater.** Joint torque/velocity/effort limits are enforced by the robot
  hardware. The bridge does not re-implement them. (Finite/NaN/shape validation and the gain-
  convention math are *correctness*, not safety, and stay.)
- **Rich, symmetric logging.** Recording captures the full controller-debug signal set and is byte-
  for-byte schema-identical in sim and real so the two can be compared directly.

---

## 2. Confirmed decisions (the design contract)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Single launch file `sbmpc_franka_bringup.launch.py`, switch `backend:=mujoco\|real`. | One code path; impossible to drift. |
| D2 | Delete `sbmpc_franka_lfc_mujoco_sim.launch.py`, `sbmpc_franka_lfc_real.launch.py`, `sbmpc_pregrasp_demo.launch.py`. | Replaced by D1. |
| D3 | Final arg set (8): `backend`, `enable_nonzero_control`, `use_rviz`, `headless`, `robot_ip`, `publish_rollout_markers`, `record_replay`, `use_gripper`. | Down from 27/35/15. |
| D4 | Remove the realtime preflight + `require_realtime` entirely. | Counterproductive; not wanted. |
| D5 | Remove `use_fake_hardware`/`fake_sensor_commands` + the mock-hardware xacro branch. | Undocumented, unused; MuJoCo is *the* sim. |
| D6 | Remove the in-launch validator (`validate_sbmpc_sim` stays as a standalone `ros2 run` tool). | Belongs outside the operational launch. |
| D7 | Replace the bridge stdout string-match warmup trigger with the diagnostics-topic wait (already used by `wait_for_bridge_warmup`). | The string-match is fragile. |
| D8 | Both backends **activate the LFC stack up front** (PD-hold), warm up, then arm. | Removes the real-only inactive-load/activate asymmetry. The LFC's own PD phase (`pd_to_lf_transition_duration`) does the holding in both. ⚠ validate on hardware before first armed real run. |
| D9 | **One** bridge config `sbmpc_bridge.yaml` for both backends. No `*_real_bringup` / `*_feedforward` variants. | Ground truth; user removed the others. |
| D10 | Keep `franka_lfc_params_sim.yaml` (mujoco-only LFC overlay: `remove_gravity_compensation_effort: false`). | Physical: MuJoCo has no gravity comp, the real FCI does. |
| D11 | **Drop** the sim latency model (`control_output_delay_sec`) and the `max_sensor_age_sec`/`max_planner_output_age_sec` launch overrides. | The real controller will be sized for no transport delay; sim runs on the same config values (0.12 s guards from `sbmpc_bridge.yaml`). |
| D12 | `planner_mode` stays a **config-file setting** in `sbmpc_bridge.yaml` (default `exact_feedback`, the reference using feedback gains). No launch arg — edit the yaml to check `feedforward`-only. | Keep the ability to test feedforward without growing the launch interface. |
| D13 | Rollout markers stay behind a flag, default **off** (`publish_rollout_markers:=false`). | Avoid polluting controller timing unless explicitly debugging. |
| D14 | Recording: single `record_replay:=<path>` (empty=off); when on, LFC-output (commanded torque) capture is **always enabled**, schema identical in sim/real. | Full controller-debug set; cross-backend comparison. |
| D15 | Arming `enable_nonzero_control` moves from a ROS **parameter** to a **`std_srvs/SetBool` service** on the bridge, with precondition checks. Param remains only as the initial state. | Params are config, not commands; a service gives request/response + precondition enforcement. |
| D16 | Remove the dead torque/gain **limit** code from `safety.py` and the bridge (the limiter is already inert under the single config). Keep gain-convention math, finite/shape validation, staleness checks, deadline diagnostics, and the disarm emergency-PD-hold. | Software limits are not wanted; the rest is correctness/control, not safety. |
| D17 | Remove the launch-level `taskset` CPU pinning. | `franka_controllers.yaml` already sets `cpu_affinity`; the launch pinning is redundant complexity. |
| D18 | Remove the `controller_watchdog` from the launch graph. | Monitoring/auto-shutdown layer; not wanted (review-flag: easy to keep if desired). |
| D19 | Drop the `allow_existing_ros_graph` arg; keep the mujoco clean-graph preflight fail-closed (no escape hatch). | One fewer arg; stale-graph collisions are a real sim footgun. |
| D20 | Warmup wait is **unbounded** (`wait_for_bridge_warmup --timeout-sec 0`, already supported at `warmup_wait.py:88`). No `WARMUP_TIMEOUT_SEC` constant. | Planner JIT can exceed minutes; a deadline risks aborting a legitimate compile. Crash-safety is covered by the bridge's `on_exit=Shutdown()` (tears down the launch → kills the waiter, which exits via `rclpy.ok()` going false); an alive-but-never-warming bridge is an operator Ctrl-C. The **switch / reset / arm** timeouts stay bounded (no JIT there — fail fast). |
| D21 | Gripper actuation kept in **both** backends behind one symmetric `use_gripper` arg (default `true`): mujoco spawns `gripper_action_controller`; real includes the `agimus_franka_gripper` FCI node (`robot_ip`, `arm_id:fer`). The hand stays in the URDF unconditionally (`mount_end_effector:=true`) for kinematic parity. | User requires the gripper usable in sim *and* real. The two grip through different physical paths (MuJoCo effort interface vs. Franka gripper FCI), so the actuation node legitimately differs by backend. |
| D22 | The clean-graph preflight (`assert_clean_ros_graph`) runs for **both** backends, not just mujoco. Message reworded "simulation" → "Franka bringup"; internal helper names keep their cosmetic `sim` suffix. | Its checks are backend-agnostic (the mujoco-only entries simply don't match on real); a stale `controller_manager` on the live FCI is *more* dangerous than in sim. It is the backstop; clean teardown (already handled in the current design per the maintainer) is the primary mechanism. |

**Deferred (separate task, NOT in this plan):** MuJoCo↔URDF dynamic-model parity (the MJCF in
`mujoco/fer_ros2_control.xml` is menagerie-Panda-derived and not reconciled with
`agimus_franka_description`). Tracked for Phase 5; needs its own investigation.

---

## 3. Final argument reference

| Arg | Default | Backends | Meaning |
|-----|---------|----------|---------|
| `backend` | `mujoco` | both | `mujoco` (physics sim) or `real` (Franka FCI). |
| `enable_nonzero_control` | `false` | both | If true, the warmup step arms the bridge (via the SetBool service) after warmup. |
| `use_rviz` | `true` | both | Launch RViz. |
| `headless` | `true` | mujoco | Open the MuJoCo viewer when false. Ignored for `real`. |
| `robot_ip` | `172.17.1.2` | real | Franka FCI IP. Ignored for `mujoco`. |
| `publish_rollout_markers` | `false` | both | Publish MPPI rollout markers for RViz (off the control hot path). |
| `record_replay` | `""` | both | Path to write a replay JSON; empty disables recording. |
| `use_gripper` | `true` | both | Actuate the gripper. mujoco: spawn `gripper_action_controller`; real: include `agimus_franka_gripper`. The hand is always in the URDF regardless (kinematic parity). |

`planner_mode` is intentionally **not** a launch arg — it lives in `sbmpc_bridge.yaml`
(`exact_feedback` by default; edit it to `feedforward` for open-loop checks).

**Environment (not launch args):** `SBMPC_BRIDGE_RUNTIME_SCRIPT`
(default `/workspace/sbmpc_containers/scripts/pixi_ros_run.sh`), `PIXI_ENV` (default `cuda`),
`SBMPC_DIR` (default `/workspace/sbmpc`).

**Moved to `constants.py`:** controller-manager name (`/controller_manager`), bridge node name
(`/sbmpc_lfc_bridge_node`), reset service (`/mujoco_ros2_control_node/reset_world`), keyframe
(`home`), controller-switch timeout. (The warmup wait is unbounded — see D20 — so there is no
warmup-timeout constant. The real-only `joint_state_publisher` rate is inlined as `30` — TF/RViz
only, not control.)

---

## 4. Unified launch flow

```
                         ┌── backend == mujoco ──┐         ┌── backend == real ──┐
preflight                clean ROS-graph (fail-closed) — BOTH backends (D22)
robot_description        mujoco xacro                       real xacro (robot_ip, mount_end_effector:true)
controller_manager host  mujoco_ros2_control/ros2_control_node   controller_manager/ros2_control_node
                         (use_sim_time:=true)               (use_sim_time:=false, +robot_type/load_gripper/arm_prefix)
broadcasters             joint_state_broadcaster            joint_state_broadcaster (remap → franka/joint_states)
                                                            franka_robot_state_broadcaster
                                                            joint_state_publisher (rate 30, TF only)
gripper (use_gripper)    gripper_action_controller          agimus_franka_gripper include (robot_ip, arm_id:fer)
LFC stack (D8)           joint_state_estimator + linear_feedback_controller, ACTIVATED up front
                         + franka_lfc_params_sim.yaml overlay    (no sim overlay)
bridge (disarmed)        -m sbmpc_ros_bridge.lfc_bridge_node ; params: sbmpc_bridge.yaml +
                         {use_sim_time, enable_nonzero_control:false, publish_rollout_markers}
                         (planner_mode comes from sbmpc_bridge.yaml, not the launch)
warmup + arm             wait_for_bridge_warmup (diagnostics topic, --timeout-sec 0 / unbounded, D20):
                          mujoco → reset world to `home`, then       both → arm (param now; SetBool
                          arm if requested                                  service in Phase 2) if requested
rviz                     if use_rviz (started up front)
recorder                 if record_replay != "" (--include-warmup + --record-lfc-output, identical schema)
shutdown handlers        Shutdown() on any required-spawner / bridge / control-host / warmup failure
```

Backend-conditional pieces (the **only** legitimate divergences): HW plugin + xacro; `use_sim_time`;
the gravity-comp overlay (D10, **physically mandatory**: MuJoCo has no gravity comp, the real FCI
does); the mujoco world-reset (real cannot teleport); the real-only broadcasters/joint_state_publisher;
the gripper actuation path (D21: sim `gripper_action_controller` vs. real FCI include). The clean-graph
preflight is **shared** by both backends (D22 — no longer a divergence).

---

## 5. Phase 1 — Unified launch (no bridge code touched)

> Outcome: the three launches become one; args slashed; cruft removed. The bridge runs unchanged
> (the safety limiter is already inert under the single config). Arming still uses the param at the
> end of Phase 1; Phase 2 swaps it for the service.

- [x] **1.1 Constants.** Add to `sbmpc_bringup/constants.py`: `CONTROLLER_MANAGER_NAME`,
  `BRIDGE_NODE_NAME`, `RESET_WORLD_SERVICE`, `DEFAULT_KEYFRAME`, `CONTROLLER_SWITCH_TIMEOUT_SEC`
  (5 constants). No `WARMUP_TIMEOUT_SEC` (warmup wait is unbounded, D20 — launch passes
  `--timeout-sec 0`); no `JOINT_STATE_RATE` (inline `30` in the real branch, TF-only). Verify:
  `python -c "import ..."` / unit import.
- [x] **1.2 Preflight.** In `launch_preflight.py`, `assert_clean_ros_graph(context, *args, **kwargs)`
  no longer reads the `allow_existing_ros_graph` LaunchConfiguration (D19) — fully fail-closed, no
  keyword bypass; removed the now-dead `_launch_bool` and `LaunchConfiguration` import. Helpers
  unchanged; `test_launch_preflight.py` still green (10/10).
- [x] **1.3 New launch file** `launch/sbmpc_franka_bringup.launch.py` using a single
  `OpaqueFunction(launch_setup)` that reads `backend` via `.perform(context)` and builds the
  backend-appropriate nodes per §4. Implements D3–D8, D11–D13, D17–D22. Verified: 8-arg surface +
  defaults + `backend` choices introspected; `launch_setup` runs for both backends (mujoco 7 nodes,
  real 8 nodes + gripper include, 4 event handlers each) with the preflight mocked.
- [x] **1.4 Recorder wiring (D14).** When `record_replay` is non-empty, add the
  `record_sbmpc_replay` node with `--record-lfc-output` + `--include-warmup` always passed,
  `--output <path>` and `--duration-sec 0` (record until shutdown). mujoco passes
  `--measured-torque-topic ""`; real uses the recorder's default real topic (τ lives in `joint_states`
  on sim). Wired inside the 1.3 launch.
- [x] **1.5 xacro (D5).** Removed the `use_fake_hardware`/`fake_sensor_commands` args and the
  `mock_components/GenericSystem` branch from `urdf/franka_arm_with_sbmpc_real.urdf.xacro`; only the
  `AgimusFrankaHardwareInterface` hardware block remains. Verified: renders to 429 lines, no
  `mock_components`; the two literal `"false"` values left are the agimus macro's own params.
- [x] **1.6 Delete** `launch/sbmpc_franka_lfc_mujoco_sim.launch.py`,
  `launch/sbmpc_franka_lfc_real.launch.py`, `launch/sbmpc_pregrasp_demo.launch.py` (D2). Done — only
  `sbmpc_franka_bringup.launch.py` remains. Remaining refs are tests (1.7) + docs (Phase 4).
- [x] **1.7 Tests — rewritten (suite was red, referenced deleted files).**
  - Replaced `test_launch_imports.py` + `test_mujoco_launch_imports.py` with one
    `test_bringup_launch.py`: asserts the 8-arg set, defaults, `backend` choices, and the
    backend-conditional node graph (mujoco vs real spawners, jsb remap, sim overlay, recorder).
  - `test_bringup_config.py`: expected config set = the 4 real files; gripper type
    `effort_controllers/GripperActionController` (the old test wrongly expected position_controllers
    *and* a `type` under the gripper params); dropped the `*_feedforward`/`*_real_bringup` preset
    test and the `simulation_cpu_prefixes` test (D17); kept the MPPI-knobs-not-duplicated guard.
  - `test_real_xacro.py`: deleted `test_real_xacro_can_render_fake_hardware_for_dry_checks`.
  - `test_ee_parity_smoke.py`: repointed launch name → `sbmpc_franka_bringup.launch.py` + `backend:=mujoco`.
  - `test_launch_preflight.py`: unaffected, green.
  - **Extra (pre-existing red, fixed):** `test_replay_recorder.py` `control_message` mock lacked the
    `effort` field that a real `JointState` always has, crashing `replay.py`; added `effort=[]`.
- [x] **1.8 Verify Phase 1.** Full `pytest` on the package green: **57 passed, 2 skipped** (the 2 skips
  are live hardware/sim smoke tests). Clean-rebuilt `sbmpc_bringup` (0.7 s) — install space now has only
  `sbmpc_franka_bringup.launch.py`. `ros2 launch sbmpc_bringup sbmpc_franka_bringup.launch.py --show-args`
  lists all 8 args with `backend` choices. **Live smoke (sim/hardware) is the user's runtime step.**

**Review checkpoint A:** new launch + tests green, old files gone, args = 8. ✅ Phase 1 done. ⟶ user review.

---

## 6. Phase 2 — Service-based arming (D15)

> Outcome: arming the robot is an explicit, precondition-checked RPC, not a silent param write.

- [ ] **2.1 Bridge service.** In `lfc_bridge_node.py`, add a `std_srvs/srv/SetBool` service
  `~/set_nonzero_control`. Handler: `data=true` → arm only if warmup complete (else
  `success=false`, message says why); `data=false` → disarm (always allowed). Back it with a
  thread-safe internal `self._control_enabled`; `enable_nonzero_control` param becomes the initial
  value only. Audit all readers of `_nonzero_control_enabled()` to use the internal flag.
- [ ] **2.2 Warmup tool.** In `warmup_wait.py`, replace `set_remote_bool_parameter(... enable_nonzero_control ...)`
  with a `SetBool` client call to `~/set_nonzero_control`; keep the `--enable-nonzero-control` CLI flag.
- [ ] **2.3 Launch.** Bridge still starts with `enable_nonzero_control:=false`; warmup step arms via
  the service when the `enable_nonzero_control` arg is true. No launch-interface change.
- [ ] **2.4 Tests.** Add a bridge test for the service (arm refused before warmup, allowed after,
  disarm always). Update `warmup_wait` tests for the service call. Update any test that armed via
  the param.
- [ ] **2.5 Verify.** Unit tests green; live: confirm `ros2 service call .../set_nonzero_control std_srvs/srv/SetBool "{data: true}"` arms and is rejected pre-warmup.

**Review checkpoint B.** ⟶ user review.

---

## 7. Phase 3 — Remove dead software limits (D16)

> The torque/gain limiter is already inert (no limits set in the single config). This is cleanup.
> ⚠ Surgical: `safety.py` mixes removable policy with load-bearing control math.

**Remove:** `BringupLimits`, `ControlSafetyLimits` (the limit fields), `make_conservative_bringup_profile`,
`apply_torque_limit`, `apply_joint_torque_limits`, `apply_gain_norm_limit`, the limit-application
block inside `validate_planner_output`; bridge params `max_abs_torque`, `max_abs_torque_by_joint`,
`torque_limit_mode`, `max_gain_norm`, `gain_limit_mode`; methods `_safety_profile_from_parameters`,
`_optional_positive_double_parameter`, `_optional_positive_double_array_parameter`,
`_log_safety_limits`. Delete `test_safety.py` limit cases and the `__init__.py` exports.

**Keep (do NOT touch):** `sbmpc_gain_to_lfc_gain` + `SBMPC_TO_LFC_GAIN_SCALE` (gain negation),
`validate_planner_output` shape/finite checks, `compute_lfc_state_error`/`compute_lfc_control`,
control-age/staleness (`AlwaysOnSafety`, `validate_control_age` — tied to `max_planner_output_age_sec`),
`PlanningDeadlineMonitor`/`deadline_miss_count` (timing diagnostics for debugging), and the disarm
emergency-PD-hold (`_publish_emergency_hold`, `hold_on_disarm_after_control`, `emergency_hold_*_gain`)
— this is graceful-stop control behavior, not a hardware-handled limit.

- [ ] **3.1** Strip the limiter from `safety.py`, keeping the control/validation/diagnostics core.
- [ ] **3.2** Remove the 5 limit params + the 4 helper methods from `lfc_bridge_node.py`; replace
  `validate_planner_output(..., limits=...)` call sites with the no-limits validation.
- [ ] **3.3** Update `lfc_msg_adapter.py` to call validation without limits.
- [ ] **3.4** Update `__init__.py` exports and `test_safety.py`; run the bridge test suite.
- [ ] **3.5 Verify.** `colcon test --packages-select sbmpc_ros_bridge` green; live armed run shows
  unchanged commanded torque (limiter was already inert).

**Review checkpoint C.** ⟶ user review.

---

## 8. Phase 4 — Docs & housekeeping

- [ ] **4.1** Update launch commands in `sbmpc_ros/README.md`, `sbmpc_containers/README.md`, and the
  `sbmpc/docs/*.md` references to `ros2 launch sbmpc_bringup sbmpc_franka_bringup.launch.py backend:=...`.
- [ ] **4.2** Update this plan's checkboxes; note any deviations.

---

## 9. Phase 5 — Model parity (DEFERRED, separate task)

Reconcile the MuJoCo physics model (`mujoco/fer_ros2_control.xml`, menagerie-Panda-derived) with
`agimus_franka_description` so sim dynamics match the real robot. Approach TBD after a focused
diff (single-source generation vs. a parity test). Not started here.

---

## 10. Recorded controller-debug signal set (reference)

When `record_replay` is on, the replay JSON contains (timestamped, identical schema both backends):
measured state q/dq/τ (`/sbmpc/joint_states`, `/sensor`); **LFC input** `/control` (MPC feedforward
τ, feedback gain K, reference x₀); **LFC output / commanded torque** `/output_joint_effort`;
diagnostics/timing (`/sbmpc/diagnostics`: planning ms, deadline misses, accept/reject counts);
and, on real only, measured link-side τ_J (`franka_robot_state_broadcaster/measured_joint_states`).

---

## 11. Open review-flags (decide during review, low cost to flip)

- **R1 (D18):** `controller_watchdog` removed from the real launch. Keep it if you want auto-shutdown
  on a post-arming controller/HW fault.
- **R2 (D8):** LFC activated up front on real — needs a hardware sanity check (it only PD-holds the
  current position) before the first armed real run.
- **R3 (D16/Phase 3):** confirm the disarm emergency-PD-hold stays (recommended) vs. full removal.
</content>
</invoke>
