# sbmpc_ros

ROS 2 integration for running SB-MPC on the Franka through
`linear_feedback_controller`.

Use this repository from the ROS workspace:

```bash
cd /workspace/ros2_ws
```

The source checkout is available at both:

```text
/workspace/ros2_ws/src/sbmpc_ros
/workspace/sbmpc_ros
```

## Build

Rebuild after changing launch files, configs, or Python code:

```bash
cd /workspace/ros2_ws
colcon build --symlink-install --packages-select sbmpc_ros_bridge sbmpc_bringup
source install/setup.bash
```

Run the focused tests:

```bash
colcon test --packages-select sbmpc_ros_bridge sbmpc_bringup --event-handlers console_direct+
colcon test-result --verbose
```

## Bridge Configs

The relevant presets live in:

```text
/workspace/sbmpc_ros/sbmpc_bringup/config
```

- `sbmpc_bridge_exact_async_40hz.yaml`: current conservative deployment preset,
  `40 Hz`, `planner_dt=0.025`, `64/512` rolling gain chunks.
- `sbmpc_bridge_exact_async.yaml`: more aggressive `50 Hz` preset,
  `planner_dt=0.02`, `128/512` rolling gain chunks.
- `sbmpc_bridge_feedforward.yaml`: feedforward-only baseline.
- `sbmpc_bridge.yaml`: older/default exact-async config kept for comparison.

To try another controller preset, pass:

```bash
bridge_params_file:=/workspace/sbmpc_ros/sbmpc_bringup/config/<config-name>.yaml
```

## Simulation

Run the validated 40 Hz MuJoCo stack and record a replay:

```bash
cd /workspace/ros2_ws
source install/setup.bash

ros2 launch sbmpc_bringup sbmpc_franka_lfc_mujoco_sim.launch.py \
  headless:=true \
  enable_nonzero_control:=true \
  bridge_params_file:=/workspace/sbmpc_ros/sbmpc_bringup/config/sbmpc_bridge_exact_async_40hz.yaml \
  record_replay:=true \
  record_replay_output:=/tmp/sbmpc_ros_40hz_replay.json \
  record_replay_duration_sec:=30
```

Replay timing summary:

```bash
/workspace/sbmpc_containers/scripts/pixi_ros_run.sh \
  python -m sbmpc_bringup.trajectory_replay \
  /tmp/sbmpc_ros_40hz_replay.json --dry-run
```

Offline visualization:

```bash
/workspace/sbmpc_containers/scripts/pixi_ros_run.sh \
  python -m sbmpc_bringup.trajectory_replay \
  /tmp/sbmpc_ros_40hz_replay.json
```

Useful simulation arguments:

- `headless:=true`: no live viewer, preferred for timing runs.
- `enable_nonzero_control:=true`: send SB-MPC commands after bridge warmup.
- `bridge_params_file:=...`: select the bridge/controller preset.
- `record_replay:=true`: record `/sensor`, `/control`, and diagnostics.
- `record_replay_duration_sec:=30`: fixed recording duration; `0` records
  until launch shutdown.
- `record_replay_output:=/tmp/name.json`: replay output path.

## Real Robot

The real launch now follows the same readiness order as the simulation stack:
load LFC controllers inactive, start the SB-MPC bridge, wait for planner/JAX
warmup, activate `joint_state_estimator` and `linear_feedback_controller`, then
publish controller outputs. No separate arming command is needed for the normal
run. After arming, a controller watchdog monitors the LFC controllers and the
Agimus hardware component; if controller_manager deactivates them after a robot
reflex or hardware fault, the launch shuts down instead of hanging.

Default real robot launch, using robot IP `172.17.1.2` and the 40 Hz preset:

```bash
cd /workspace/ros2_ws
source install/setup.bash

ros2 launch sbmpc_bringup sbmpc_franka_lfc_real.launch.py
```

Equivalent explicit command:

```bash
ros2 launch sbmpc_bringup sbmpc_franka_lfc_real.launch.py \
  robot_ip:=172.17.1.2 \
  bridge_params_file:=/workspace/sbmpc_ros/sbmpc_bringup/config/sbmpc_bridge_exact_async_40hz.yaml
```

Try the 50 Hz preset:

```bash
ros2 launch sbmpc_bringup sbmpc_franka_lfc_real.launch.py \
  robot_ip:=172.17.1.2 \
  bridge_params_file:=/workspace/sbmpc_ros/sbmpc_bringup/config/sbmpc_bridge_exact_async.yaml
```

Dry arming check without SB-MPC commands:

```bash
ros2 launch sbmpc_bringup sbmpc_franka_lfc_real.launch.py \
  robot_ip:=172.17.1.2 \
  enable_nonzero_control:=false
```

This still waits for planner/JAX warmup and activates the LFC stack, but the
bridge remains silent after warmup so LFC stays in its PD holding mode. Use this
first when checking the live FCI control loop, then rerun with
`enable_nonzero_control:=true` once the controller stays active.

Useful real launch arguments:

- `robot_ip:=172.17.1.2`: Franka FCI address.
- `bridge_params_file:=...`: select 40 Hz, 50 Hz, or feedforward-only config.
- `enable_nonzero_control:=true`: default; publish SB-MPC outputs after
  readiness. Set false only for dry bringup.
- `max_abs_torque:=12.0`: default real-robot feedforward cap in Nm. Raise only
  after checking `/control` with the LFC command probe.
- `torque_limit_mode:=clip`: clip over-limit planner feedforward. Use `reject`
  to fail closed instead of publishing a clipped command.
- `controllers_file:=...`: ROS2-control controller config.
- `lfc_params_file:=...`: LFC/JSE parameters.
- `bridge_warmup_timeout_sec:=120`: maximum time to wait for planner/JAX warmup.
- `controller_switch_timeout_sec:=10`: controller activation timeout.
- `controller_watchdog_period_sec:=0.25`: post-arming controller/hardware state
  polling period.
- `load_gripper:=false`: arm-only bringup.
- `use_camera:=false`: set true to add the Agimus calibrated camera mount to
  the robot description.
- `ee_id:=agimus_franka_hand`: end-effector model used when `load_gripper` or
  `use_camera` requires one.
- `use_fake_hardware:=true`: controller-manager check without the robot.

Monitor the run:

```bash
ros2 control list_controllers
ros2 topic echo /sbmpc/diagnostics
```

Key diagnostics are `/control` cadence, fresh planner output rate, deadline
misses, foreground time, background gain timing, gain age, completed gain
batches, rejected planner outputs, and joint velocity.

To diagnose a hard Franka reflex, start the LFC command probe before arming or
before launching with automatic nonzero control:

```bash
ros2 run sbmpc_ros_bridge sbmpc_lfc_control_probe \
  --sensor-topic /sensor \
  --control-topic /control \
  --warn-abs-effort 15.0 \
  --json
```

The probe estimates the fixed-base LFC command from the actual ROS messages:
`tau = feedforward + K * ([q_des - q, v_des - v])`. If `max_abs_feedforward` is
large, the planner feedforward is the first suspect. If `max_abs_feedback_effort`
is large while the feedforward is modest, inspect the initial-state mismatch,
gain norm, and the PD-to-LF transition.

To replay the planner from a measured robot state without touching the robot,
copy `measured_position` and `measured_velocity` from one probe JSON line:

```bash
/workspace/sbmpc_containers/scripts/pixi_ros_run.sh \
  ros2 run sbmpc_ros_bridge sbmpc_planner_smoke \
  --joint-set fer \
  --planner-mode exact_async_feedback \
  --planner-horizon 8 \
  --planner-dt 0.025 \
  --planner-gain-samples-per-cycle 64 \
  --planner-gain-buffer-size 512 \
  --q '<measured_position JSON list>' \
  --v '<measured_velocity JSON list>'
```

The smoke report includes `feedforward`, `max_abs_feedforward`,
`predicted_q_delta_norm`, and `predicted_v_norm` for that state.

## Planner Smoke

Real planner smoke through the ROS adapter:

```bash
/workspace/sbmpc_containers/scripts/pixi_ros_run.sh \
  python -m sbmpc_ros_bridge.planner_smoke \
  --joint-set fer --planner-mode exact_async_feedback
```
