# sbmpc_ros

ROS 2 wiring for the synchronous SB-MPC Franka pregrasp controller and the
`linear_feedback_controller` (LFC). One launch file serves both the MuJoCo
simulation and the real Franka, switched by `backend:=mujoco|real`.

## Controller contract

The bridge publishes one coherent planner output per cycle (feedforward `tau_ff`
and feedback gain `K` together) at 25 Hz (`dt = 0.04 s`), retaining the shifted
MPPI solution as the warm start. The MPPI knobs (horizon, samples, dt, smoothing,
top-K gain samples) and the cost weights have a single source of truth:
`sbmpc/sbmpc/ocp_configs/<planner_ocp>.yaml`. The bridge config only carries
deployment settings.

Active files:

- `sbmpc_bringup/launch/sbmpc_franka_bringup.launch.py` — the one bringup launch (both backends).
- `sbmpc_bringup/config/sbmpc_bridge.yaml` — bridge deployment config (topics, rate, mode, task, joint names).
- `sbmpc_bringup/config/franka_controllers.yaml`, `franka_lfc_params.yaml` — controller + LFC params.
- `sbmpc_bringup/config/franka_lfc_params_sim.yaml` — MuJoCo-only overlay (no gravity compensation; the real FCI does it).

## Build

```bash
cd /workspace/ros2_ws
source /opt/ros/jazzy/setup.bash
source /opt/sbmpc_deps_ws/install/setup.bash
colcon build --symlink-install --packages-select sbmpc_ros_bridge sbmpc_bringup
source install/setup.bash
```

## Run (MuJoCo simulation)

`backend` defaults to `mujoco`:

```bash
ros2 launch sbmpc_bringup sbmpc_franka_bringup.launch.py
```

The LFC activates in PD-hold while JAX compiles; the warmup step then resets
MuJoCo to the `home` keyframe and arms the bridge only if requested (see
**Arming**). RViz starts after warmup to avoid compile-time GPU contention.

Launch arguments (defaults in parentheses):

| Arg | Default | Meaning |
|-----|---------|---------|
| `backend` | `mujoco` | `mujoco` (physics sim) or `real` (Franka FCI). |
| `enable_nonzero_control` | `false` | If true, the warmup step arms the bridge (via the SetBool service) after warmup. |
| `use_rviz` | `true` | Launch RViz. |
| `headless` | `true` | Open the MuJoCo viewer when `false` (mujoco only). |
| `robot_ip` | `172.17.1.2` | Franka FCI IP (real only). |
| `publish_rollout_markers` | `false` | Publish MPPI rollout markers for RViz. |
| `record_replay` | `""` | Path to write a replay JSON; empty disables recording. |
| `use_gripper` | `true` | Actuate the gripper (sim: `gripper_action_controller`; real: `agimus_franka_gripper`). |

```bash
# Open the MuJoCo viewer, no RViz
ros2 launch sbmpc_bringup sbmpc_franka_bringup.launch.py headless:=false use_rviz:=false

# Show representative MPPI end-effector rollouts in RViz
ros2 launch sbmpc_bringup sbmpc_franka_bringup.launch.py publish_rollout_markers:=true
```

`planner_mode` is intentionally not a launch arg — it lives in `sbmpc_bridge.yaml`
(`exact_feedback` by default; set to `feedforward` there for open-loop checks).

## Run (real robot)

```bash
ros2 launch sbmpc_bringup sbmpc_franka_bringup.launch.py backend:=real robot_ip:=172.17.1.2
```

The real and sim paths differ only where physics forces it (HW plugin, gravity
compensation, no world-reset on hardware, the gripper FCI node). On real, the LFC
activates up front and PD-holds the current position until you arm the MPC —
verify that "it only holds, does not move" before the first armed run.

## Arming

Arming is an explicit, precondition-checked service on the bridge; the bridge
always starts disarmed (LFC stays in PD-hold). Arming is rejected until planner
warmup completes. The story is identical in sim and real.

```bash
# Arm
ros2 service call /sbmpc_lfc_bridge_node/set_nonzero_control std_srvs/srv/SetBool "{data: true}"

# Disarm (always allowed)
ros2 service call /sbmpc_lfc_bridge_node/set_nonzero_control std_srvs/srv/SetBool "{data: false}"
```

Pass `enable_nonzero_control:=true` at launch to have the warmup step arm
automatically once warmup finishes; otherwise warm up disarmed and arm manually.

## Record and replay

```bash
# Record until shutdown (commanded LFC output is always captured; schema matches sim/real)
ros2 launch sbmpc_bringup sbmpc_franka_bringup.launch.py \
  record_replay:=/tmp/sbmpc_replay.json

# Summarize or visualize a recording
replay_sbmpc_trajectory /tmp/sbmpc_replay.json --dry-run
replay_sbmpc_trajectory /tmp/sbmpc_replay.json
```

## Validate a live simulation

`validate_sbmpc_sim` is a standalone tool (not part of the launch). Start the
sim, then in another shell:

```bash
ros2 run sbmpc_bringup validate_sbmpc_sim --assert-stable
```

It reads `/sbmpc/diagnostics` and `/joint_states` and reports task error, gain
health, and planner timing over the run window (`--duration-sec`, default 16 s).

## Planner smoke (no ROS graph)

```bash
/workspace/sbmpc_containers/scripts/pixi_ros_run.sh \
  python -m sbmpc_ros_bridge.planner_smoke \
  --joint-set fer \
  --planner-mode exact_feedback
```

The MPPI knobs come from `sbmpc/sbmpc/ocp_configs/pregrasp.yaml` (override with
`--planner-ocp` or the individual `--planner-*` flags only when experimenting).

## Tests

```bash
colcon test --packages-select sbmpc_ros_bridge sbmpc_bringup --event-handlers console_direct+
colcon test-result --verbose
```
