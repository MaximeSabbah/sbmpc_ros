# sbmpc_ros

ROS 2 wiring for the synchronous SB-MPC Franka pregrasp controller and `linear_feedback_controller`.

## Controller Contract

The default bridge uses the same YAML-defined controller as `sbmpc/scripts/panda_pregrasp.py`:

- requested publication rate: 25 Hz (`dt=0.04`)
- MPPI horizon: 10
- feedforward samples: 1024
- same-cycle exact-gain samples: 512
- one coherent planner output containing both `tau_ff` and `K`
- shifted MPPI solution retained as the warm start
- no rolling gain window, background gain worker, or trajectory reseeding

The active files are:

- `sbmpc_bringup/config/sbmpc_bridge.yaml`: exact feedback controller
- `sbmpc_bringup/config/sbmpc_bridge_feedforward.yaml`: feedforward-only diagnostic baseline
- `sbmpc_bringup/launch/sbmpc_pregrasp_demo.launch.py`: simulation, visualization, and validation entry point

## Build

```bash
cd /workspace/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select sbmpc_ros_bridge sbmpc_bringup
source install/setup.bash
```

## Live MuJoCo And RViz

```bash
cd /workspace/ros2_ws
source install/setup.bash
ros2 launch sbmpc_bringup sbmpc_pregrasp_demo.launch.py
```

This opens the live MuJoCo viewer immediately. RViz uses software OpenGL and is deliberately launched only after JAX warmup completes, avoiding the compilation-time GPU contention that previously crashed it. The validator prints task error, gain health, torque/velocity/position usage, and planner timing after 16 seconds; the simulation remains open for interactive tuning.

Useful arguments:

```bash
# MuJoCo only
ros2 launch sbmpc_bringup sbmpc_pregrasp_demo.launch.py use_rviz:=false

# Keep running without the automatic metric collector
ros2 launch sbmpc_bringup sbmpc_pregrasp_demo.launch.py validate:=false
```

## Headless Validation

```bash
ros2 launch sbmpc_bringup sbmpc_pregrasp_demo.launch.py \
  headless:=true \
  use_rviz:=false \
  max_p95_planning_ms:=0 \
  shutdown_after_validation:=true
```

Set `max_p95_planning_ms:=40` to enforce the strict 25 Hz controller budget. On the current RTX 5000 Ada setup, the measured synchronous exact controller is about 44-46 ms per update, so the strict gate currently fails and the planner produces roughly 22 Hz of fresh outputs. Robot limits remain comfortably respected, but this timing gap must be resolved before claiming a true 25 Hz deployment.

## Record And Replay

Recording remains available for repeatable RViz/offline inspection:

```bash
ros2 launch sbmpc_bringup sbmpc_pregrasp_demo.launch.py \
  headless:=true \
  use_rviz:=false \
  record_replay:=true \
  record_replay_output:=/tmp/sbmpc_pregrasp_replay.json
```

Replay summary or visualization:

```bash
python -m sbmpc_bringup.trajectory_replay /tmp/sbmpc_pregrasp_replay.json --dry-run
python -m sbmpc_bringup.trajectory_replay /tmp/sbmpc_pregrasp_replay.json
```

## Planner Smoke

```bash
/workspace/sbmpc_containers/scripts/pixi_ros_run.sh \
  python -m sbmpc_ros_bridge.planner_smoke \
  --joint-set fer \
  --planner-mode exact_feedback \
  --planner-horizon 10 \
  --planner-dt 0.04 \
  --planner-num-gain-samples 512
```

## Real Robot

The real launch uses the same bridge configuration and readiness sequence:

```bash
ros2 launch sbmpc_bringup sbmpc_franka_lfc_real.launch.py
```

Use `enable_nonzero_control:=false` for a dry bringup that warms the planner and keeps LFC in PD hold. The ROS simulation behavior and timing should be accepted before enabling commands on hardware.

## Tests

```bash
colcon test --packages-select sbmpc_ros_bridge sbmpc_bringup --event-handlers console_direct+
colcon test-result --verbose
```
