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
      sbmpc_franka_lfc_real.launch.py
    config/
      franka_controllers.yaml
      franka_lfc_params.yaml
      franka_lfc_params_sim.yaml
      sbmpc_bridge.yaml
      sbmpc_bridge_exact_async.yaml
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
Only three bridge presets are kept:

- `sbmpc_bridge.yaml`: default exact async feedback.
- `sbmpc_bridge_feedforward.yaml`: feedforward baseline.
- `sbmpc_bridge_exact_async.yaml`: background exact-gain validation.

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
- To visualize a full ROS-stack run without adding viewer load to the live
  controller/MuJoCo process, let the launch file record a replay and open it
  afterward:

  ```bash
  # Run the full stack headless and write a replay when the launch exits.
  ros2 launch sbmpc_bringup sbmpc_franka_lfc_mujoco_sim.launch.py \
    headless:=true enable_nonzero_control:=true \
    record_replay:=true \
    record_replay_output:=/tmp/sbmpc_ros_replay.json

  # After stopping the live stack, open the offline MuJoCo replay viewer.
  /workspace/sbmpc_containers/scripts/pixi_ros_run.sh \
    python -m sbmpc_bringup.trajectory_replay /tmp/sbmpc_ros_replay.json
  ```

  The replay file contains `/sbmpc/joint_states`, `/sensor`, `/control`
  summaries, and `/sbmpc/diagnostics`. The viewer only replays recorded joint
  positions, so it does not contend with the SB-MPC/JAX controller during the
  timing-sensitive run. By default the recorder waits for the first `/control`
  message before starting, so warmup/JAX compilation is excluded. Set
  `record_replay_include_warmup:=true` to capture the startup phase too, or
  `record_replay_duration_sec:=8` to record a fixed-length window instead of
  recording until launch shutdown. The launch recorder autosaves every
  `record_replay_autosave_period_sec:=5` seconds and writes an empty file as
  soon as it starts, so an old replay file cannot silently survive a failed or
  interrupted recording.

  Standalone recording is also available through `ros2 run`:

  ```bash
  ros2 run sbmpc_bringup record_sbmpc_replay \
    --duration-sec 8 --output /tmp/sbmpc_ros_replay.json
  ```

  Use the Pixi wrapper for replay and dry-run replay validation, because that
  environment provides the MuJoCo Python viewer stack. Replay pacing defaults
  to `--time-source auto`, which uses recorder wall-clock receive time when
  available and falls back to control cadence for older files whose state
  header stamps did not advance.

  Direct shell commands such as `record_sbmpc_replay` are not guaranteed to be
  on `PATH` in ROS 2; `ros2 run sbmpc_bringup ...` is the portable form. If
  `ros2 run` cannot find the executable, rebuild and source the ROS workspace:

  ```bash
  cd /workspace/ros2_ws
  colcon build --packages-select sbmpc_bringup
  source install/setup.bash
  ```

  During development, the source module can also be run without rebuilding:

  ```bash
  PYTHONPATH=/workspace/sbmpc_ros/sbmpc_bringup:$PYTHONPATH \
    python3 -m sbmpc_bringup.record_replay \
    --duration-sec 8 --output /tmp/sbmpc_ros_replay.json

  PYTHONPATH=/workspace/sbmpc_ros/sbmpc_bringup:$PYTHONPATH \
    python3 -m sbmpc_bringup.trajectory_replay \
    /tmp/sbmpc_ros_replay.json
  ```
