from __future__ import annotations

import numpy as np
from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker

from sbmpc_ros_bridge.rollout_markers import make_trajectory_marker_array


def test_make_trajectory_marker_array_contains_sampled_and_optimized_paths() -> None:
    optimized_path = np.asarray(
        [
            [0.4, 0.0, 0.3],
            [0.5, 0.0, 0.35],
            [0.6, 0.0, 0.4],
        ],
        dtype=np.float64,
    )
    rollout_paths = np.stack(
        [
            optimized_path + np.asarray([0.0, 0.01, 0.0]),
            optimized_path + np.asarray([0.0, -0.01, 0.0]),
        ],
        axis=0,
    )
    goal = np.asarray([0.7, 0.1, 0.5], dtype=np.float64)

    marker_array = make_trajectory_marker_array(
        ee_path=optimized_path,
        rollout_paths=rollout_paths,
        goal_position=goal,
        frame_id="base",
        stamp=Time(sec=1, nanosec=2),
    )

    assert [marker.action for marker in marker_array.markers] == [
        Marker.ADD,
        Marker.ADD,
        Marker.ADD,
        Marker.ADD,
        Marker.ADD,
    ]
    assert marker_array.markers[0].type == Marker.LINE_LIST
    assert marker_array.markers[1].type == Marker.LINE_STRIP
    assert marker_array.markers[2].type == Marker.SPHERE_LIST
    assert marker_array.markers[3].type == Marker.SPHERE
    assert marker_array.markers[4].type == Marker.SPHERE
    assert len(marker_array.markers[0].points) == 2 * 2 * (optimized_path.shape[0] - 1)
    assert len(marker_array.markers[1].points) == optimized_path.shape[0]
    assert len(marker_array.markers[2].points) == optimized_path.shape[0]
    assert marker_array.markers[3].pose.position.x == optimized_path[0, 0]
    assert marker_array.markers[4].pose.position.z == goal[2]
