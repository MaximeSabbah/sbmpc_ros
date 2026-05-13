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


@dataclass(frozen=True)
class StaleProcess:
    pid: int
    command: str


@dataclass(frozen=True)
class SimRuntimePreflightResult:
    nodes: tuple[str, ...]
    stale_nodes: tuple[str, ...]
    stale_processes: tuple[StaleProcess, ...]


STALE_SIM_PROCESS_PATTERNS: tuple[str, ...] = (
    "/mujoco_ros2_control/ros2_control_node",
    "/robot_state_publisher/robot_state_publisher",
    "/controller_manager/spawner",
    "python -m sbmpc_ros_bridge.lfc_bridge_node",
    "wait_for_bridge_warmup",
    "record_sbmpc_replay",
)


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


def find_stale_sim_processes(
    ps_output: str,
    *,
    protected_patterns: Iterable[str] = STALE_SIM_PROCESS_PATTERNS,
) -> tuple[StaleProcess, ...]:
    stale_processes: list[StaleProcess] = []
    patterns = tuple(protected_patterns)
    for line in ps_output.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            continue
        pid_text, command = fields
        if not any(pattern in command for pattern in patterns):
            continue
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        stale_processes.append(StaleProcess(pid=pid, command=command))
    return tuple(stale_processes)


def check_clean_sim_runtime(
    *,
    graph_runner: Runner = subprocess.run,
    process_runner: Runner = subprocess.run,
    protected_names: Iterable[str] = STALE_SIM_NODE_NAMES,
    protected_process_patterns: Iterable[str] = STALE_SIM_PROCESS_PATTERNS,
) -> SimRuntimePreflightResult:
    graph = check_clean_ros_graph(
        runner=graph_runner,
        protected_names=protected_names,
    )
    process_result = process_runner(
        ["ps", "-eo", "pid=,cmd="],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )
    if process_result.returncode != 0:
        stderr = process_result.stderr.strip() or "no stderr captured"
        raise RuntimeError(f"Unable to inspect existing processes: {stderr}")

    stale_processes = find_stale_sim_processes(
        process_result.stdout,
        protected_patterns=protected_process_patterns,
    )
    return SimRuntimePreflightResult(
        nodes=graph.nodes,
        stale_nodes=graph.stale_nodes,
        stale_processes=stale_processes,
    )


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


def wait_for_clean_sim_runtime(
    *,
    graph_runner: Runner = subprocess.run,
    process_runner: Runner = subprocess.run,
    protected_names: Iterable[str] = STALE_SIM_NODE_NAMES,
    protected_process_patterns: Iterable[str] = STALE_SIM_PROCESS_PATTERNS,
    timeout_sec: float = 20.0,
    poll_sec: float = 1.0,
) -> SimRuntimePreflightResult:
    deadline = time.monotonic() + max(0.0, timeout_sec)
    result = check_clean_sim_runtime(
        graph_runner=graph_runner,
        process_runner=process_runner,
        protected_names=protected_names,
        protected_process_patterns=protected_process_patterns,
    )
    while (
        result.stale_nodes or result.stale_processes
    ) and time.monotonic() < deadline:
        time.sleep(max(0.0, poll_sec))
        result = check_clean_sim_runtime(
            graph_runner=graph_runner,
            process_runner=process_runner,
            protected_names=protected_names,
            protected_process_patterns=protected_process_patterns,
        )
    return result


def assert_clean_ros_graph(context, *args, **kwargs) -> list[object]:
    del args, kwargs
    if _launch_bool(LaunchConfiguration("allow_existing_ros_graph").perform(context)):
        return []

    result = wait_for_clean_sim_runtime()
    if not result.stale_nodes and not result.stale_processes:
        return []

    stale_parts: list[str] = []
    if result.stale_nodes:
        stale_parts.append("nodes: " + ", ".join(result.stale_nodes))
    if result.stale_processes:
        process_list = ", ".join(
            f"{process.pid} ({process.command})"
            for process in result.stale_processes
        )
        stale_parts.append("processes: " + process_list)
    raise RuntimeError(
        "Refusing to launch SB-MPC Franka simulation because an existing ROS "
        "control graph or simulation process is already running or still visible "
        f"after waiting for graph settling: {'; '.join(stale_parts)}. Stop the "
        "previous launch cleanly before retrying. "
        "If this is intentional, pass "
        "allow_existing_ros_graph:=true."
    )


def _launch_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
