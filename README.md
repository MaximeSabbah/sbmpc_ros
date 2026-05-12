# sbmpc_ros

ROS 2 integration workspace for running SB-MPC with the Franka Panda through
`linear_feedback_controller`.

This milestone creates the initial `sbmpc_ros_bridge` package skeleton and
proves the core message contract against the installed
`linear_feedback_controller_msgs` interfaces:

- strict Panda joint-name validation
- `Sensor -> PlannerInput` conversion
- `PlannerOutput -> Control` conversion
- row-major `feedback_gain` serialization
- `(7, 1)` `feedforward` serialization, matching the installed LFC Eigen
  conversion path
- deep-copied `initial_state` snapshots for replayable control messages
- rejection of malformed or non-finite planner outputs

`sbmpc` remains the algorithm dependency. This repository intentionally keeps
the planner logic out of ROS and only adds thin adapters around the stable
planner API exposed in `/workspace/sbmpc/sbmpc/planner_api.py`.

In the container build, the ROS package name `franka_description` is expected
to be provided by the Agimus fork `agimus-project/agimus-franka-description`
rather than the upstream Franka repository.

This Git repository is not intended to be the colcon workspace root.

Use this repository through the canonical ROS workspace path:

```bash
/workspace/ros2_ws/src/sbmpc_ros
```

`/workspace/ros2_ws` is the colcon workspace root. `build/`, `install/`, and
`log/` belong there, not in this repository.

For compatibility with older tooling, the container also restores:

```bash
/workspace/sbmpc_ros
```

Use `/workspace/ros2_ws/src/sbmpc_ros` as the canonical development path and
`/workspace/sbmpc_ros` only when a legacy script or editor workflow still
expects it.

## Safety Philosophy

The bridge safety code is intentionally split into three layers:

- `always_on`: checks that should remain enabled in every deployment. Today this
  covers the gain sign convention, message validity, non-finite rejection, and
  optional stale-control checks.
- `bringup_limits`: optional conservative limits for early robot testing, such
  as torque clipping or gain-norm scaling. These should stay tunable and should
  not be confused with the permanent controller architecture.
- `monitoring_only`: signals worth observing, such as planner deadline misses,
  without automatically degrading controller authority in mature deployments.

Franka hardware limits are necessary, but they are not a replacement for
bridge-side validity checks. The bridge should catch integration mistakes and
obviously invalid outputs before they reach the robot, while avoiding
unnecessary long-term restrictions on the controller.

The current bringup defaults are adapted to the local `fer` robot description,
which uses runtime joint names `fer_joint1 ... fer_joint7`.

## Layout

```text
sbmpc_ros/
  README.md
  sbmpc_ros_bridge/
    package.xml
    setup.py
    setup.cfg
    resource/
    sbmpc_ros_bridge/
      __init__.py
      diagnostics.py
      joint_mapping.py
      lfc_msg_adapter.py
      planner_adapter.py
      safety.py
      lfc_bridge_node.py
    test/
  sbmpc_bringup/
    package.xml
    launch/
      sbmpc_franka_lfc_mujoco_sim.launch.py
      sbmpc_franka_lfc_real.launch.py
    config/
      franka_controllers.yaml
      franka_lfc_params.yaml
      franka_lfc_params_sim.yaml
      sbmpc_bridge.yaml
      sbmpc_bridge_exact_async.yaml
      sbmpc_bridge_exact_async_40hz.yaml
      sbmpc_bridge_feedforward.yaml
    test/
```

## Build And Test

```bash
cd /workspace/ros2_ws
colcon build --symlink-install --packages-select sbmpc_ros_bridge sbmpc_bringup
colcon test --packages-select sbmpc_ros_bridge sbmpc_bringup --event-handlers console_direct+
colcon test-result --verbose
```

The fake-loop integration coverage for Milestone 3 lives in:

```text
/workspace/ros2_ws/src/sbmpc_ros/sbmpc_ros_bridge/test/test_fake_ros_loop.py
```

It exercises a real `rclpy` timer loop with:

- a fake LFC `Sensor` publisher
- the `sbmpc_lfc_bridge_node` at 50 Hz
- a fake `Control` subscriber
- diagnostics publication

## Runtime Checks

Real planner smoke through the ROS bridge adapter:

```bash
/workspace/sbmpc_containers/scripts/pixi_ros_run.sh \
  python -m sbmpc_ros_bridge.planner_smoke --joint-set fer
```

Exact async planner smoke:

```bash
/workspace/sbmpc_containers/scripts/pixi_ros_run.sh \
  python -m sbmpc_ros_bridge.planner_smoke \
  --joint-set fer --planner-mode exact_async_feedback
```

## MuJoCo Simulation Validation

The main simulation launch runs the MuJoCo ROS2-control stack, the
`linear_feedback_controller`, and the SB-MPC bridge. Rebuild and source the ROS
workspace after changing launch files, configs, or Python packages:

```bash
cd /workspace/ros2_ws
colcon build --symlink-install --packages-select sbmpc_ros_bridge sbmpc_bringup
source install/setup.bash
```

Run the current conservative 40 Hz simulation preset and record a replay:

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

The relevant bridge presets are:

- `sbmpc_bridge_exact_async_40hz.yaml`: conservative simulation/deployment
  rehearsal preset, currently `40 Hz`, `planner_dt=0.025`, and `64/512`
  rolling exact-gain chunks.
- `sbmpc_bridge_exact_async.yaml`: more aggressive `50 Hz` exact-async preset,
  currently `planner_dt=0.02` and `128/512` rolling exact-gain chunks.
- `sbmpc_bridge_feedforward.yaml`: feedforward-only comparison, useful to
  isolate MPPI torque behavior from feedback-gain behavior.

To test another preset, change only `bridge_params_file`:

```bash
ros2 launch sbmpc_bringup sbmpc_franka_lfc_mujoco_sim.launch.py \
  headless:=true \
  enable_nonzero_control:=true \
  bridge_params_file:=/workspace/sbmpc_ros/sbmpc_bringup/config/sbmpc_bridge_exact_async.yaml \
  record_replay:=true \
  record_replay_output:=/tmp/sbmpc_ros_50hz_replay.json \
  record_replay_duration_sec:=30
```

Before opening the viewer, check the replay summary:

```bash
/workspace/sbmpc_containers/scripts/pixi_ros_run.sh \
  python -m sbmpc_bringup.trajectory_replay \
  /tmp/sbmpc_ros_40hz_replay.json --dry-run
```

Then replay the recorded trajectory offline:

```bash
/workspace/sbmpc_containers/scripts/pixi_ros_run.sh \
  python -m sbmpc_bringup.trajectory_replay \
  /tmp/sbmpc_ros_40hz_replay.json
```

The replay file contains recorded `/sensor`, `/control`, and
`/sbmpc/diagnostics` data. The viewer replays recorded joint positions after
the live run has stopped, so visualization does not add rendering load to the
timing-sensitive SB-MPC/JAX process.

Useful simulation launch arguments:

- `headless:=true`: run without a live viewer. Use this for timing runs.
- `enable_nonzero_control:=true`: arm the bridge so nonzero SB-MPC torques are
  sent to LFC after warmup.
- `bridge_params_file:=...`: choose the SB-MPC bridge preset.
- `record_replay:=true`: start the replay recorder as part of the launch.
- `record_replay_output:=/tmp/name.json`: choose the replay output path.
- `record_replay_duration_sec:=30`: record a fixed window. Use `0` to record
  until launch shutdown.
- `record_replay_include_warmup:=true`: include startup/JAX warmup in the
  recording. Leave this false for normal controller timing review.
- `record_replay_autosave_period_sec:=5`: autosave period while recording.

The dry-run summary is the quickest sanity check. For controller timing, look
at `/control` cadence, fresh planner output rate, deadline misses, foreground
time, background gain timing, gain age, completed gain batches, rejected
planner outputs, and joint velocity. For visual validation, check that the
offline replay duration matches the recording and that the arm motion is
smooth, bounded, and consistent with the PREGRASP behavior.

Standalone recording is also available through `ros2 run` while the stack is
already running:

```bash
ros2 run sbmpc_bringup record_sbmpc_replay \
  --duration-sec 8 --output /tmp/sbmpc_ros_replay.json
```

Direct shell commands such as `record_sbmpc_replay` are not guaranteed to be
on `PATH` in ROS 2; `ros2 run sbmpc_bringup ...` is the portable form. During
source-tree development, the modules can also be run without rebuilding:

```bash
PYTHONPATH=/workspace/sbmpc_ros/sbmpc_bringup:$PYTHONPATH \
  python3 -m sbmpc_bringup.record_replay \
  --duration-sec 8 --output /tmp/sbmpc_ros_replay.json

PYTHONPATH=/workspace/sbmpc_ros/sbmpc_bringup:$PYTHONPATH \
  python3 -m sbmpc_bringup.trajectory_replay \
  /tmp/sbmpc_ros_replay.json
```

Real or fake-hardware bringup:

```bash
cd /workspace/ros2_ws
source install/setup.bash
ros2 launch sbmpc_bringup sbmpc_franka_lfc_real.launch.py robot_ip:=<robot-ip>
```

Fake-hardware check:

```bash
cd /workspace/ros2_ws
source install/setup.bash
ros2 launch sbmpc_bringup sbmpc_franka_lfc_real.launch.py \
  use_fake_hardware:=true use_rviz:=true
```

Bringup launches the bridge through
`/workspace/sbmpc_containers/scripts/pixi_ros_run.sh`, so ROS sees the
`sbmpc` JAX/Pixi stack without requiring a separate manual wrapper command.

Useful launch overrides:

- `sbmpc_dir:=/workspace/sbmpc`
- `pixi_env:=cuda`
- `bridge_runtime_script:=/workspace/sbmpc_containers/scripts/pixi_ros_run.sh`
- `load_gripper:=false` only for arm-only debugging

## Planner Tuning From YAML

The bridge reads planner overrides from:

```bash
/workspace/ros2_ws/src/sbmpc_ros/sbmpc_bringup/config/sbmpc_bridge.yaml
```

This lets you tune the ROS-side runtime without editing the `sbmpc` codebase.
The bridge presets are:

- `sbmpc_bridge.yaml`: default exact async feedback.
- `sbmpc_bridge_feedforward.yaml`: feedforward baseline.
- `sbmpc_bridge_exact_async.yaml`: background exact-gain validation.
- `sbmpc_bridge_exact_async_40hz.yaml`: conservative 40 Hz exact-async
  simulation/deployment rehearsal preset.

The preferred controller selector is `planner_mode`:

- `feedforward`: MPPI feedforward only, zero gain sent to LFC.
- `exact_async_feedback`: MPPI feedforward plus background exact gain.

The main tuning knobs are:

- `planner_mode`
- `planner_phase`
- `planner_horizon`
- `planner_num_samples`
- `planner_num_control_points`
- `planner_dt`
- `planner_temperature`
- `planner_noise_scale`
- `planner_smoothing`
- `planner_gain_samples_per_cycle`
- `planner_gain_buffer_size`

For numeric overrides, leave the value at `0` or `0.0` to keep the current
`sbmpc` default. Set it to a positive value to override the planner config from
ROS.

## Notes

- The bridge package assumes the ROS environment already exposes
  `linear_feedback_controller_msgs` and `rclpy`.
- The runtime planner adapter lazy-loads `sbmpc` so the bridge package can be
  imported even before the algorithm repo is path-installed into the same
  Python environment.
- The high-level safety entry points live in
  `/workspace/ros2_ws/src/sbmpc_ros/sbmpc_ros_bridge/sbmpc_ros_bridge/safety.py`
  as `AlwaysOnSafety`, `BringupLimits`, `MonitoringOnly`, and
  `BridgeSafetyProfile`.
- When editing code in the container, open files from
  `/workspace/ros2_ws/src/sbmpc_ros`.
- For host-side Git operations, the underlying checkout may still physically
  live outside the workspace root, but there is only one repository; the
  workspace path is the canonical one to use during development.
- The helper `/workspace/sbmpc_containers/scripts/pixi_ros_run.sh` now sources
  `/workspace/ros2_ws/install/setup.bash` before entering the Pixi
  environment, so planner smoke commands can see both ROS packages and the
  `sbmpc` JAX stack.
- Simulation bringup uses `sbmpc_franka_lfc_mujoco_sim.launch.py`, backed by
  `mujoco_ros2_control` and the exact-async bridge preset.
- The bridge diagnostics topic is `/sbmpc/diagnostics`; for async exact mode it
  includes foreground planning time, background gain timing, gain age, rolling
  window fill, completed/dropped gain batches, worker running state, and worker
  errors.
