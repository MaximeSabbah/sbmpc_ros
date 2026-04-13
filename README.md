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

This Git repository is not intended to be the colcon workspace root.

Use this repository through the canonical ROS workspace path:

```bash
/workspace/ros2_ws/src/sbmpc_ros
```

`/workspace/ros2_ws` is the colcon workspace root. `build/`, `install/`, and
`log/` belong there, not in this repository.

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
      joint_mapping.py
      lfc_msg_adapter.py
      planner_adapter.py
      safety.py
      lfc_bridge_node.py
    test/
```

## Milestone 1 Test Command

```bash
cd /workspace/ros2_ws
colcon build --symlink-install --packages-select sbmpc_ros_bridge
colcon test --packages-select sbmpc_ros_bridge --event-handlers console_direct+
colcon test-result --verbose
```

## Notes

- The bridge package assumes the ROS environment already exposes
  `linear_feedback_controller_msgs` and `rclpy`.
- The runtime planner adapter lazy-loads `sbmpc` so the bridge package can be
  imported even before the algorithm repo is path-installed into the same
  Python environment.
- When editing code in the container, open files from
  `/workspace/ros2_ws/src/sbmpc_ros`.
- For host-side Git operations, the underlying checkout may still physically
  live outside the workspace root, but there is only one repository; the
  workspace path is the canonical one to use during development.
