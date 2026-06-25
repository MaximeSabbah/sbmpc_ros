from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


def make_trajectory_marker_array(
    *,
    ee_path: np.ndarray,
    goal_position: np.ndarray,
    frame_id: str,
    stamp,
    rollout_paths: np.ndarray | None = None,
    namespace: str = "sbmpc_rollout",
    line_width: float = 0.01,
    sample_line_width: float = 0.004,
    point_diameter: float = 0.025,
    goal_diameter: float = 0.045,
) -> MarkerArray:
    optimized_path = np.asarray(ee_path, dtype=np.float64)
    if (
        optimized_path.ndim != 2
        or optimized_path.shape[1] != 3
        or optimized_path.shape[0] < 1
    ):
        raise ValueError(f"ee_path must have shape (N, 3), got {optimized_path.shape}.")
    goal = np.asarray(goal_position, dtype=np.float64).reshape(3)
    sample_paths = _normalize_rollout_paths(rollout_paths)

    sample_lines = _base_marker(frame_id, stamp, namespace, 1)
    sample_lines.type = Marker.LINE_LIST
    sample_lines.scale.x = float(sample_line_width)
    _set_color(sample_lines, (0.35, 0.60, 1.0, 0.22))
    sample_lines.points = _rollout_line_list_points(sample_paths)

    optimized_line = _base_marker(frame_id, stamp, namespace, 2)
    optimized_line.type = Marker.LINE_STRIP
    optimized_line.scale.x = float(line_width)
    _set_color(optimized_line, (1.0, 0.85, 0.20, 0.98))
    optimized_line.points = [_point(row) for row in optimized_path]

    optimized_points = _base_marker(frame_id, stamp, namespace, 3)
    optimized_points.type = Marker.SPHERE_LIST
    optimized_points.scale.x = float(point_diameter)
    optimized_points.scale.y = float(point_diameter)
    optimized_points.scale.z = float(point_diameter)
    _set_color(optimized_points, (1.0, 0.95, 0.55, 0.80))
    optimized_points.points = [_point(row) for row in optimized_path]

    current = _base_marker(frame_id, stamp, namespace, 4)
    current.type = Marker.SPHERE
    current.pose.position = _point(optimized_path[0])
    current.scale.x = float(goal_diameter)
    current.scale.y = float(goal_diameter)
    current.scale.z = float(goal_diameter)
    _set_color(current, (0.25, 1.0, 0.45, 0.95))

    goal_marker = _base_marker(frame_id, stamp, namespace, 5)
    goal_marker.type = Marker.SPHERE
    goal_marker.pose.position = _point(goal)
    goal_marker.scale.x = float(goal_diameter)
    goal_marker.scale.y = float(goal_diameter)
    goal_marker.scale.z = float(goal_diameter)
    _set_color(goal_marker, (1.0, 0.42, 0.22, 0.95))

    return MarkerArray(
        markers=[
            sample_lines,
            optimized_line,
            optimized_points,
            current,
            goal_marker,
        ]
    )


def goal_position_from_planner_output(
    planner_output: object,
    fallback: np.ndarray,
) -> np.ndarray:
    diagnostics = getattr(planner_output, "diagnostics", None)
    goal_position = getattr(diagnostics, "goal_position", None)
    if goal_position is None:
        return np.asarray(fallback, dtype=np.float64).reshape(3)
    goal = np.asarray(goal_position, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(goal)):
        return np.asarray(fallback, dtype=np.float64).reshape(3)
    return goal


def _normalize_rollout_paths(rollout_paths: np.ndarray | None) -> np.ndarray:
    if rollout_paths is None:
        return np.empty((0, 0, 3), dtype=np.float64)
    paths = np.asarray(rollout_paths, dtype=np.float64)
    if paths.size == 0:
        return np.empty((0, 0, 3), dtype=np.float64)
    if paths.ndim != 3 or paths.shape[2] != 3:
        raise ValueError(f"rollout_paths must have shape (N, T, 3), got {paths.shape}.")
    return paths


def _rollout_line_list_points(rollout_paths: np.ndarray) -> list[Point]:
    points: list[Point] = []
    if rollout_paths.shape[0] <= 0:
        return points
    for path in rollout_paths:
        if path.shape[0] < 2:
            continue
        for start, stop in zip(path[:-1], path[1:], strict=False):
            points.append(_point(start))
            points.append(_point(stop))
    return points


def _base_marker(frame_id: str, stamp, namespace: str, marker_id: int) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = namespace
    marker.id = int(marker_id)
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    return marker


def _set_color(marker: Marker, rgba: tuple[float, float, float, float]) -> None:
    marker.color.r = float(rgba[0])
    marker.color.g = float(rgba[1])
    marker.color.b = float(rgba[2])
    marker.color.a = float(rgba[3])


def _point(values: np.ndarray) -> Point:
    arr = np.asarray(values, dtype=np.float64).reshape(3)
    return Point(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))
