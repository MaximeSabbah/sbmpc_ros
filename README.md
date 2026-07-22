# sbmpc_ros

ROS 2 wiring for the Franka pregrasp MPC controllers and the
`linear_feedback_controller` (LFC). One launch file serves both the MuJoCo
simulation and the real Franka, switched by `backend:=mujoco|real`, and two
solver backends, switched by `planner:=hydrax|sbmpc`.

## Planner backends

| | `planner:=hydrax` (default) | `planner:=sbmpc` (legacy) |
|---|---|---|
| Solver | Feedback-MPPI in the hydrax fork (`/workspace/hydrax`) | original `sbmpc` package |
| Python runtime | uv venv (`scripts/uv_ros_run.sh` in sbmpc_containers) | pixi env (`pixi_ros_run.sh`) |
| Bridge preset | `sbmpc_bringup/config/hydrax_bridge.yaml` | `sbmpc_bringup/config/sbmpc_bridge.yaml` |
| OCP tuning surface | `hydrax/hydrax/configs/pregrasp.yaml` | `sbmpc/sbmpc/ocp_configs/pregrasp.yaml` |

The bridge machinery (topics, safety, watchdog, arming, replay) is identical
for both; the `planner` argument selects the preset yaml, the Python runtime
wrapper, and the adapter inside the bridge node.

**Gain modes** (`planner_mode` in the preset yaml — the yaml is the mode
switch, there is no launch argument for it):

- `feedforward` — K published to LFC is a constant joint impedance
  (`Kp/Kd` from the task options) anchored on the plan reference. The
  proven, most-precise mode; the flip-back on the robot.
- `exact_feedback` — K is the solve's Feedback-MPPI gain `du*/dx0`,
  anchored at the state the plan was solved from (x0), so LFC applies
  `tau_ff + K (x - x0)`: the F-MPPI paper's law, no impedance anywhere.
  This is the deployed hydrax mode.

After editing a preset yaml, rebuild the install space (launch reads configs
from there, not from the source tree):

```bash
cd /workspace/ros2_ws && colcon build --packages-select sbmpc_bringup && source install/setup.bash
```

## Controller contract

The bridge publishes one coherent planner output per cycle (feedforward `tau_ff`
and feedback gain `K` together) at 25 Hz (`dt = 0.04 s`), retaining the shifted
MPPI solution as the warm start. The OCP tuning has a single source of truth
per backend (table above); the bridge presets carry deployment settings only —
nothing in the ROS parameter layer can override the OCP values (guard tests
enforce this).

Active files:

- `sbmpc_bringup/launch/sbmpc_franka_bringup.launch.py` — the one bringup launch (both backends, both planners).
- `sbmpc_bringup/config/hydrax_bridge.yaml` — hydrax bridge preset (topics, rate, `planner_mode`, joint names, rollout-marker styling).
- `sbmpc_bringup/config/sbmpc_bridge.yaml` — sbmpc bridge preset.
- `sbmpc_bringup/config/franka_controllers.yaml`, `franka_lfc_params.yaml` — controller + LFC params.
- `sbmpc_bringup/config/franka_lfc_params_sim.yaml` — MuJoCo-only overlay (no gravity compensation; the real FCI does it).
- `sbmpc_ros_bridge/hydrax_planner_adapter.py` — the hydrax adapter (K conventions and anchor semantics are documented in its module docstring).
- `hydrax/doc/feedback_mppi_panda_port_plan.md` (in the hydrax repo) — the canonical plan, decisions log, and validation gates of the Feedback-MPPI port.

## Build

```bash
cd /workspace/ros2_ws
source /opt/ros/jazzy/setup.bash
source /opt/sbmpc_deps_ws/install/setup.bash
colcon build --symlink-install --packages-select sbmpc_ros_bridge sbmpc_bringup
source install/setup.bash
```

## Run (MuJoCo simulation)

`backend` defaults to `mujoco`, the viewer opens, and the bridge arms itself once
warmup finishes — so the simulation runs out of the box. The one command that
loads the sim, shows the MPPI rollouts, and starts moving after warmup:

```bash
ros2 launch sbmpc_bringup sbmpc_franka_bringup.launch.py publish_rollout_markers:=true
```

The LFC activates in PD-hold while JAX compiles (~20 s); the warmup step then
resets MuJoCo to the `home` keyframe and arms the bridge. RViz starts after warmup
to avoid compile-time GPU contention.

Launch arguments (defaults in parentheses):

| Arg | Default | Meaning |
|-----|---------|---------|
| `backend` | `mujoco` | `mujoco` (physics sim) or `real` (Franka FCI). |
| `planner` | `hydrax` | `hydrax` (Feedback-MPPI, uv runtime) or `sbmpc` (legacy, pixi runtime). |
| `enable_nonzero_control` | `true` | Arm the bridge (via the SetBool service) after warmup. **Set `false` on real hardware for a disarmed bringup.** |
| `use_rviz` | `true` | Launch RViz. |
| `headless` | `false` | Run mujoco without the viewer when `true` (mujoco only). |
| `initial_q` | `home` | Arm start pose (mujoco only). `random` draws a fresh pose within ±0.2 rad/joint of home (the V-A5-certified placement envelope, logged at launch) — the reference plan still starts at home, so you watch the controller absorb the gap, as with a hand-placed real robot. |
| `robot_ip` | `172.17.1.2` | Franka FCI IP (real only). |
| `publish_rollout_markers` | `false` | Publish MPPI rollout markers for RViz. |
| `use_gripper` | `true` | Actuate the gripper (sim: `gripper_action_controller`; real: `agimus_franka_gripper`). |

```bash
# Headless, disarmed sim (e.g. for recording or CI)
ros2 launch sbmpc_bringup sbmpc_franka_bringup.launch.py \
  headless:=true use_rviz:=false enable_nonzero_control:=false
```

## Run (real robot)

`enable_nonzero_control` defaults to `true`, so on real hardware pass
`enable_nonzero_control:=false` for a disarmed bringup and arm explicitly once
you have verified the PD-hold (see **Arming**):

```bash
ros2 launch sbmpc_bringup sbmpc_franka_bringup.launch.py \
  backend:=real robot_ip:=172.17.1.2 enable_nonzero_control:=false
```

The real and sim paths differ only where physics forces it (HW plugin, gravity
compensation, no world-reset on hardware, the gripper FCI node). On real, the LFC
activates up front and PD-holds the current position until you arm the MPC —
verify that "it only holds, does not move" before the first armed run.

## Arming

Arming is an explicit, precondition-checked service on the bridge. The bridge
*process* starts disarmed (LFC PD-holds); the warmup step then arms it via the
service when `enable_nonzero_control` is true (the default in sim, opt-in on
real). Arming is rejected until planner warmup completes. Arm or disarm manually
at any time:

```bash
# Arm
ros2 service call /sbmpc_lfc_bridge_node/set_nonzero_control std_srvs/srv/SetBool "{data: true}"

# Disarm (always allowed)
ros2 service call /sbmpc_lfc_bridge_node/set_nonzero_control std_srvs/srv/SetBool "{data: false}"
```

## Record, diagnose, and replay a run

`record_sbmpc_run` is the one recording procedure for both MuJoCo and the real
robot. It starts the MCAP recorder first, starts the normal unified bringup,
captures the complete run, stops bringup before finalizing the bag, and only
then generates the report and replay data. Do not start a separate bringup for
the same experiment. A process lock rejects a second recorder/controller
supervisor while a run is active.

The controller-manager process must inherit realtime limits from the recorder
terminal. The compose configuration supplies them for newly created
containers. If an already-running container predates those settings, repair
only the current terminal (no reinstall or container recreation) and verify it
before recording:

```bash
sudo prlimit --pid $$ --rtprio=99 --memlock=unlimited
chrt -f 1 true && echo "SCHED_FIFO available"
```

The recorder stores this probe and the inherited limits in
`provenance/realtime.json` and prints a warning when FIFO is unavailable.

```bash
# MuJoCo (backend defaults to mujoco; the label is descriptive only).
ros2 run sbmpc_bringup record_sbmpc_run --label pregrasp_sim -- \
  publish_rollout_markers:=false

# Real robot: the recorder injects enable_nonzero_control:=false unless you
# explicitly override it. Inspect the PD hold, then arm from another shell.
ros2 run sbmpc_bringup record_sbmpc_run --label pregrasp_real -- \
  backend:=real robot_ip:=172.17.1.2
```

Everything after `--` is passed to
`sbmpc_franka_bringup.launch.py` as `name:=value`. Press Ctrl-C once in the
recorder terminal after the experiment. A second Ctrl-C accelerates bringup
shutdown if graceful teardown is stuck; MCAP still gets its own finalization
window. For a bounded unattended simulation, add `--duration-sec <seconds>`
before `--`.

Each run is stored under `/workspace/sbmpc_runs/<label>_<UTC timestamp>`, and
the latest path is written to `/tmp/sbmpc_run_path`. The run directory contains:

- `rosbag/` — the raw MCAP and source of truth;
- `manifest.json` and `provenance/` — launch arguments, environment, GPU
  information, and Git state/diffs for the planner, ROS overlay, container,
  Agimus, and low-level feedback-controller sources;
- `launch-console.log`, `rosbag-console.log`, `ros_logs/`, and
  `rosbag-info.txt`;
- `diagnostic_report/` — `index.html`, `summary.json`, CSV tables, and PNG plots;
- `replay.json` — automatically derived from the same MCAP.

The shared topic contract captures the 1 kHz LFC sensor/input/output path,
planner diagnostics, joint state, `ros2_control` diagnostics and lifecycle,
ROS logs, and gripper action/state observability. On hardware it additionally
decodes the canonical Agimus/FCI robot state; in MuJoCo it captures clock,
object-pose, actuator, and simulated-gripper signals. Missing backend-specific
topics are reported as unavailable rather than changing the procedure.

All decoding, plots, CSVs, summaries, and replay export run after bringup and
recording have stopped, so those operations cannot consume planner or 1 kHz
controller-loop time. The report covers reference and Riccati-anchor tracking,
terminal stability, controller output and timing, state-stream consistency,
phase transitions, and gripper behavior. When Agimus state is available, it
also covers the causal pre-/post-limiter torque path, FCI joint/motor and
Cartesian state, wrench/contact/collision signals, robot mode, command success,
and current versus historical error flags.

Open or replay the latest run with:

```bash
RUN_DIR="$(< /tmp/sbmpc_run_path)"

# Browser-open diagnostic_report/index.html using your usual host/container workflow.
/workspace/sbmpc_containers/scripts/uv_ros_run.sh \
  ros2 run sbmpc_bringup replay_sbmpc_trajectory "$RUN_DIR/replay.json" --dry-run
/workspace/sbmpc_containers/scripts/uv_ros_run.sh \
  ros2 run sbmpc_bringup replay_sbmpc_trajectory "$RUN_DIR/replay.json"
```

If automatic post-processing reports an error, the MCAP is retained. After
fixing the analysis environment, regenerate the report with:

```bash
ros2 run sbmpc_bringup report_sbmpc_bag "$RUN_DIR"
```

On hardware, `/output_joint_effort` is labelled as the gravity-free LFC request
before the Agimus torque-rate limiter and is never equated with measured total
`tau_J`. In MuJoCo, it is labelled as the simulator effort request; no Agimus
limit is applied and simulator actuator effort is never called FCI torque.

## Validate a live simulation

`validate_sbmpc_sim` is a standalone tool (not part of the launch). Start the
sim, then in another shell:

```bash
ros2 run sbmpc_bringup validate_sbmpc_sim --assert-stable
```

It reads `/sbmpc/diagnostics` and `/joint_states` and reports task error, gain
health, and planner timing over the run window (`--duration-sec`, default 16 s).

## Planner smoke (no ROS graph, sbmpc backend)

```bash
/workspace/sbmpc_containers/scripts/pixi_ros_run.sh \
  python -m sbmpc_ros_bridge.planner_smoke \
  --joint-set fer \
  --planner-mode exact_feedback
```

The sbmpc MPPI knobs come from `sbmpc/sbmpc/ocp_configs/pregrasp.yaml`
(override with `--planner-ocp` or the individual `--planner-*` flags only when
experimenting). The hydrax equivalent of a no-ROS check is the V-B1 contract
test (see the verification runbook below).

## Verifying the Feedback-MPPI stack from scratch

The complete check-out ritual for the hydrax Feedback-MPPI controller — what
to run, in which environment, and what "good" looks like. Assumes the
container from `sbmpc_containers` is running and
`check_unified_env.sh` passes (see that repo's README for the fresh-machine
setup). Deep background: `hydrax/doc/feedback_mppi_panda_port_plan.md`.

**Rules of the game, before anything else:**

1. **The GPU must be otherwise idle** during any timing-relevant run. A
   cohabiting job (another training run, even RViz + the MuJoCo viewer on
   the same card) shows up directly as planner deadline misses.
2. `pytest` through the pixi/uv wrappers needs
   `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` — the ROS-installed pytest plugins
   otherwise break collection *silently* (a suite that reports "1 skipped"
   while running zero tests).
3. Config yamls under `sbmpc_bringup/config/` are read from the **install
   space**: after editing one, `colcon build --packages-select sbmpc_bringup`.
   Python files are symlinked (live). The hydrax tuning yaml
   (`hydrax/hydrax/configs/pregrasp.yaml`) is read live, no rebuild needed.
4. The first solve after a config change JIT-compiles (~1 min for the gain
   graph; later runs hit the cache in `/workspace/hydrax/.jax_cache`).

### Step 1 — Tier A: the controller alone (no ROS)

Runs entirely inside the hydrax repo, in its uv env. The multi-rate example
(1 kHz plant / 25 Hz planner, the LFC law) is a faithful model of the
deployed loop — Tier A↔B parity was measured exact.

```bash
cd /workspace/hydrax

# Feedforward reach (V-A1). Expect: VERDICT PASS, terminal_ee_error ~0.002 m.
uv run python examples/panda_pregrasp.py

# Feedback reach (V-A4 nominal). Expect: terminal_ee_error ~0.007 m PASS,
# health_ess ~87 PASS, solve mean/p95 ~30/31 ms PASS. Two gates FAIL by
# design and are accepted (documented in the port plan):
#   tracking_vs_feedforward (pure feedback tracks 2-6x behind the stiff
#   impedance during motion) and health_k_smoothness (K wiggles relative
#   to its small norm; the applied-torque effect is ~0.01 Nm).
uv run python examples/panda_pregrasp.py --mode feedback

# Robustness scenarios (all must stay stable, margins PASS):
uv run python examples/panda_pregrasp.py --mode feedback --disturb
uv run python examples/panda_pregrasp.py --mode feedback --mass_scale 0.9
uv run python examples/panda_pregrasp.py --mode feedback --latency 0.04

# Initial-configuration robustness (V-A5) — run before any robot session
# where the arm is hand-placed rather than parked exactly at the home
# pose. The reference plan always starts at the hardcoded start_q, so the
# controller must close the placement gap; the sweep perturbs the start
# by 0.05/0.1/0.2 rad per joint x 8 seeds (~25 s per run, one process
# each; failures keep their trajectory for --replay). Requires the
# nominal V-A1/V-A4 report from above.
#
# feedback (the deployed mode): expect VERDICT PASS — certified 24/24 on
# 2026-07-07, placement error absorbed within the reach at no extra
# torque up to +-0.2 rad/joint.
uv run python examples/panda_pregrasp_q0_sweep.py --mode feedback
# feedforward: a tolerance MEASUREMENT, expect FAIL beyond small offsets
# (the fixed impedance yanks kp*dq at arming: 2026-07-07 grid passed
# 6/8 at +-0.05, 1/8 at +-0.1, 0/8 at +-0.2 rad, demand up to 2.5x
# tau_max). Only arm feedforward mode with the arm parked at the home
# pose.
uv run python examples/panda_pregrasp_q0_sweep.py --mode feedforward

# Watch a recorded run in the MuJoCo viewer (needs a display; the reference
# plan renders as a transparent ghost robot):
uv run python examples/panda_pregrasp.py --replay --show_reference
```

Reports land in `hydrax/validation/reports/` (fixed name per check+mode,
overwritten each run; `history.jsonl` keeps one line per run).

Gain correctness (V-A2): finite-difference proof of K = du*/dx0 plus the
25 Hz cycle-time gate. Expect `4 passed` in ~3 min, relative FD errors
below 0.2 %, cycle mean/p95 ~30/32 ms:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_feedback_gains.py -v -s
```

### Step 2 — Tier B contract and regression tests (no launch)

```bash
# V-B1: the adapter honors the bridge contract (10 tests; includes the
# exact_feedback ones: K = the solver's gains, LFC round-trip K_lfc = -K,
# anchor = the measured solve state). Runs in the hydrax uv env + ROS:
ROS_DOMAIN_ID=29 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /workspace/sbmpc_containers/scripts/uv_ros_run.sh \
  python -m pytest /workspace/sbmpc_ros/sbmpc_ros_bridge/test/test_hydrax_planner_adapter.py -v

# Full regression net (124 tests, pixi env; hydrax-only tests auto-skip):
ROS_DOMAIN_ID=29 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /workspace/sbmpc_containers/scripts/pixi_ros_run.sh \
  python -m pytest /workspace/sbmpc_ros/sbmpc_ros_bridge/test /workspace/sbmpc_ros/sbmpc_bringup/test -q
```

### Step 3 — the simulation bringup

```bash
cd /workspace/ros2_ws && colcon build --symlink-install \
  --packages-select sbmpc_ros_bridge sbmpc_bringup && source install/setup.bash

# Everything on defaults: MuJoCo viewer, RViz, hydrax planner, the mode
# from hydrax_bridge.yaml (exact_feedback), auto-arm after warmup.
ros2 launch sbmpc_bringup sbmpc_franka_bringup.launch.py publish_rollout_markers:=true
```

What to watch in the launch log:

- `Planner configuration from ROS parameters: {'mode': 'exact_feedback', ...}`
  — the mode that will run. If this says `feedforward`, the preset yaml (or
  its install-space copy) is not what you think it is.
- `Planner warmup/JIT compilation complete.` then `armed: nonzero control
  enabled.` — the robot starts moving only after this.
- Per-solve lines: `wall 30.7/40.0ms ... |gain|=15.3 ... deadline_miss=N`.
  In exact_feedback `|gain|` is the F-MPPI K norm and varies (~9–27); in
  feedforward it is the constant impedance norm (~2000). `deadline_miss`
  must stay rare (< 1 % of solves) — if it climbs, something else is on
  the GPU.

Quantitative verdict while the sim runs (second shell):

```bash
ros2 run sbmpc_bringup validate_sbmpc_sim
```

Expected at the frozen config, exact_feedback: `planning_ms` mean ~31 /
p95 ~32, `deadline_misses` < 1 % of solves, `planner_outputs` accepted with
0 rejected, task error dipping to a few mm. Known and accepted: during the
hold the pure feedback law has no position anchor, so the pose wanders
slowly within ~1–2 cm (`tail_joint_spans_rad` ~0.02–0.04); the feedforward
mode holds sub-mm.

To compare against the feedforward mode: set `planner_mode: feedforward` in
`sbmpc_bringup/config/hydrax_bridge.yaml`, rebuild `sbmpc_bringup`
(rule 3), relaunch.

To watch the hand-placed-start case (V-A5) live — the arm starts off-home
while the reference plan still begins at home, and the controller absorbs
the gap during the reach:

```bash
ros2 launch sbmpc_bringup sbmpc_franka_bringup.launch.py initial_q:=random
```

The drawn pose is logged at launch (`initial_q=random start pose [rad]:
...`). Expect the same `validate_sbmpc_sim` numbers as the nominal run
(verified 2026-07-07: 0 deadline misses / 3304 solves, task error min
1.7 mm from a draw with offsets up to 0.19 rad).

### Step 4 — the real robot

Staged protocol (details: port plan §V-B4): keep
`max_velocity_fraction: 0.10` in `hydrax/hydrax/configs/pregrasp.yaml` for
the first sessions (0.20 is the sim-validated value), start disarmed,
verify the PD-hold, then arm:

Arm placement (V-A5, measured 2026-07-07): the reference plan always
starts at the hardcoded home `start_q`, so a hand-placed start is a
tracking error the controller must absorb. In `exact_feedback` mode that
is certified up to ±0.2 rad per joint (24/24 runs, no extra torque). In
`feedforward` mode the fixed impedance demands `kp·Δq` the moment it
arms — ±0.1 rad already asks for more than τ_max and would trip the
robot's reflexes — so **only arm feedforward mode with the arm parked at
the home pose**.

```bash
ros2 launch sbmpc_bringup sbmpc_franka_bringup.launch.py \
  backend:=real robot_ip:=172.17.1.2 enable_nonzero_control:=false
# ... verify LFC only holds, watch one full warmup, then:
ros2 service call /sbmpc_lfc_bridge_node/set_nonzero_control std_srvs/srv/SetBool "{data: true}"
```

The instant flip-back if anything looks wrong: disarm (service above with
`data: false`), set `planner_mode: feedforward` in `hydrax_bridge.yaml`,
rebuild `sbmpc_bringup`, and start the next session with `record_sbmpc_run`
(see **Record, diagnose, and replay a run**).

## Tests

```bash
colcon test --packages-select sbmpc_ros_bridge sbmpc_bringup --event-handlers console_direct+
colcon test-result --verbose
```
