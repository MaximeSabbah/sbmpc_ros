from __future__ import annotations

from dataclasses import dataclass
import subprocess
import time
from typing import Callable, Iterable, Protocol

from launch.substitutions import LaunchConfiguration

from sbmpc_bringup.constants import (
    GRIPPER_ACTION_CONTROLLER_NAME,
    JOINT_STATE_BROADCASTER_NAME,
    JOINT_STATE_ESTIMATOR_NAME,
    LINEAR_FEEDBACK_CONTROLLER_NAME,
)


STALE_SIM_NODE_NAMES: tuple[str, ...] = (
    "/controller_manager",
    "/joint_state_publisher",
    "/mujoco_ros2_control_node",
    "/robot_state_publisher",
    "/rviz2",
    "/sbmpc_lfc_bridge_node",
    f"/{GRIPPER_ACTION_CONTROLLER_NAME}",
    f"/{JOINT_STATE_BROADCASTER_NAME}",
    f"/{JOINT_STATE_ESTIMATOR_NAME}",
    f"/{LINEAR_FEEDBACK_CONTROLLER_NAME}",
)


class _RunResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[..., _RunResult]


@dataclass(frozen=True)
class RosGraphPreflightResult:
    nodes: tuple[str, ...]
    stale_nodes: tuple[str, ...]


def parse_ros_node_list(output: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in output.splitlines() if line.strip())


def find_stale_sim_nodes(
    nodes: Iterable[str],
    *,
    protected_names: Iterable[str] = STALE_SIM_NODE_NAMES,
) -> tuple[str, ...]:
    protected = set(protected_names)
    return tuple(sorted({node for node in nodes if node in protected}))


def check_clean_ros_graph(
    *,
    runner: Runner = subprocess.run,
    protected_names: Iterable[str] = STALE_SIM_NODE_NAMES,
) -> RosGraphPreflightResult:
    result = runner(
        ["ros2", "node", "list"],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "no stderr captured"
        raise RuntimeError(f"Unable to inspect the existing ROS graph: {stderr}")

    nodes = parse_ros_node_list(result.stdout)
    stale_nodes = find_stale_sim_nodes(nodes, protected_names=protected_names)
    return RosGraphPreflightResult(nodes=nodes, stale_nodes=stale_nodes)


def wait_for_clean_ros_graph(
    *,
    runner: Runner = subprocess.run,
    protected_names: Iterable[str] = STALE_SIM_NODE_NAMES,
    timeout_sec: float = 20.0,
    poll_sec: float = 1.0,
) -> RosGraphPreflightResult:
    deadline = time.monotonic() + max(0.0, timeout_sec)
    result = check_clean_ros_graph(
        runner=runner,
        protected_names=protected_names,
    )
    while result.stale_nodes and time.monotonic() < deadline:
        time.sleep(max(0.0, poll_sec))
        result = check_clean_ros_graph(
            runner=runner,
            protected_names=protected_names,
        )
    return result


def assert_clean_ros_graph(context, *args, **kwargs) -> list[object]:
    del args, kwargs
    if _launch_bool(LaunchConfiguration("allow_existing_ros_graph").perform(context)):
        return []

    result = wait_for_clean_ros_graph()
    if not result.stale_nodes:
        return []

    stale_list = ", ".join(result.stale_nodes)
    raise RuntimeError(
        "Refusing to launch SB-MPC Franka simulation because an existing ROS "
        f"control graph is already running or still visible after waiting for graph "
        f"settling: {stale_list}. Stop the previous launch cleanly before retrying. "
        "If this is intentional, pass "
        "allow_existing_ros_graph:=true."
    )


def _launch_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
