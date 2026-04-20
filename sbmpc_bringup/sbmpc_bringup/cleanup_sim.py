from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import signal
import subprocess
import time
from typing import Iterable, Sequence

import rclpy
from controller_manager_msgs.srv import (
    ListControllers,
    SwitchController,
    UnloadController,
)
from rclpy.node import Node

from sbmpc_bringup.constants import (
    GRIPPER_ACTION_CONTROLLER_NAME,
    JOINT_STATE_BROADCASTER_NAME,
    JOINT_STATE_ESTIMATOR_NAME,
    LINEAR_FEEDBACK_CONTROLLER_NAME,
)


SBMPC_SIM_CONTROLLERS: tuple[str, ...] = (
    GRIPPER_ACTION_CONTROLLER_NAME,
    LINEAR_FEEDBACK_CONTROLLER_NAME,
    JOINT_STATE_ESTIMATOR_NAME,
    JOINT_STATE_BROADCASTER_NAME,
)


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    command: str


@dataclass(frozen=True)
class ControllerCleanupReport:
    loaded_controllers: tuple[str, ...]
    deactivated_controllers: tuple[str, ...]
    unloaded_controllers: tuple[str, ...]
    controller_manager_available: bool


def parse_ps_output(output: str) -> tuple[ProcessInfo, ...]:
    processes: list[ProcessInfo] = []
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3:
            continue
        try:
            pid = int(fields[0])
            ppid = int(fields[1])
        except ValueError:
            continue
        processes.append(ProcessInfo(pid=pid, ppid=ppid, command=fields[2]))
    return tuple(processes)


def is_sbmpc_sim_process(command: str) -> bool:
    return any(
        (
            "ros2 launch sbmpc_bringup sbmpc_franka_lfc_sim.launch.py" in command,
            "gz sim" in command and "empty.sdf" in command,
            "/robot_state_publisher/robot_state_publisher" in command,
            "/ros_gz_bridge/parameter_bridge" in command,
            "/joint_state_publisher/joint_state_publisher" in command,
            "/rviz2" in command,
            "pixi_ros_run.sh" in command,
            "python -m sbmpc_ros_bridge.lfc_bridge_node" in command,
            "/controller_manager/spawner" in command,
        )
    )


def is_sbmpc_sim_launch_process(command: str) -> bool:
    return "ros2 launch sbmpc_bringup sbmpc_franka_lfc_sim.launch.py" in command


def find_sbmpc_sim_processes(
    processes: Iterable[ProcessInfo],
    *,
    exclude_pids: Iterable[int] = (),
) -> tuple[ProcessInfo, ...]:
    excluded = set(exclude_pids)
    return tuple(
        process
        for process in processes
        if process.pid not in excluded and is_sbmpc_sim_process(process.command)
    )


def list_system_processes() -> tuple[ProcessInfo, ...]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,args="],
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_ps_output(result.stdout)


def cleanup_controllers(
    *,
    controller_manager: str,
    controller_names: Sequence[str] = SBMPC_SIM_CONTROLLERS,
    timeout_sec: float,
    dry_run: bool = False,
) -> ControllerCleanupReport:
    rclpy.init(args=None)
    node = rclpy.create_node("sbmpc_sim_cleanup")
    try:
        list_client = node.create_client(
            ListControllers, f"{controller_manager}/list_controllers"
        )
        if not list_client.wait_for_service(timeout_sec=timeout_sec):
            return ControllerCleanupReport((), (), (), False)

        list_response = _call_service(
            node,
            list_client,
            ListControllers.Request(),
            timeout_sec=timeout_sec,
        )
        loaded_states = {
            controller.name: controller.state
            for controller in list_response.controller
            if controller.name in controller_names
        }
        loaded = tuple(name for name in controller_names if name in loaded_states)
        active = tuple(name for name in loaded if loaded_states[name] == "active")

        if dry_run:
            return ControllerCleanupReport(loaded, active, loaded, True)

        deactivated: tuple[str, ...] = ()
        if active:
            switch_client = node.create_client(
                SwitchController, f"{controller_manager}/switch_controller"
            )
            if not switch_client.wait_for_service(timeout_sec=timeout_sec):
                raise RuntimeError("controller switch service is unavailable")

            request = SwitchController.Request()
            request.activate_controllers = []
            request.deactivate_controllers = list(active)
            request.strictness = SwitchController.Request.BEST_EFFORT
            request.activate_asap = True
            request.timeout.sec = int(timeout_sec)
            request.timeout.nanosec = int((timeout_sec % 1.0) * 1_000_000_000)
            response = _call_service(
                node,
                switch_client,
                request,
                timeout_sec=timeout_sec,
            )
            if not response.ok:
                raise RuntimeError(f"failed to deactivate controllers: {response.message}")
            deactivated = active

        unload_client = node.create_client(
            UnloadController, f"{controller_manager}/unload_controller"
        )
        if loaded and not unload_client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError("controller unload service is unavailable")

        unloaded: list[str] = []
        for name in loaded:
            request = UnloadController.Request()
            request.name = name
            response = _call_service(
                node,
                unload_client,
                request,
                timeout_sec=timeout_sec,
            )
            if response.ok:
                unloaded.append(name)
        return ControllerCleanupReport(loaded, deactivated, tuple(unloaded), True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def terminate_processes(
    processes: Sequence[ProcessInfo],
    *,
    dry_run: bool = False,
    grace_sec: float = 3.0,
) -> tuple[int, ...]:
    if dry_run:
        return tuple(process.pid for process in processes)

    launch_processes = [
        process for process in processes if is_sbmpc_sim_launch_process(process.command)
    ]
    first_wave = launch_processes if launch_processes else list(processes)

    for process in first_wave:
        _signal_process(process.pid, signal.SIGINT)

    deadline = time.monotonic() + grace_sec
    while time.monotonic() < deadline:
        remaining = _remaining_pids(processes)
        if not remaining:
            return ()
        time.sleep(0.1)

    remaining = _remaining_pids(processes)
    for pid in remaining:
        _signal_process(pid, signal.SIGTERM)
    return _remaining_pids(processes)


def _call_service(
    node: Node,
    client,
    request,
    *,
    timeout_sec: float,
):
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    if not future.done():
        raise TimeoutError(f"service call to {client.srv_name} timed out")
    exception = future.exception()
    if exception is not None:
        raise exception
    return future.result()


def _remaining_pids(processes: Sequence[ProcessInfo]) -> tuple[int, ...]:
    remaining: list[int] = []
    for process in processes:
        try:
            os.kill(process.pid, 0)
        except ProcessLookupError:
            continue
        remaining.append(process.pid)
    return tuple(remaining)


def _signal_process(pid: int, signum: signal.Signals) -> None:
    try:
        os.kill(pid, signum)
    except ProcessLookupError:
        return


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cleanly stop the SB-MPC Franka Gazebo simulation stack."
    )
    parser.add_argument("--controller-manager", default="/controller_manager")
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--controllers-only",
        action="store_true",
        help="Deactivate/unload controllers but leave ROS/Gazebo processes alive.",
    )
    parser.add_argument(
        "--processes-only",
        action="store_true",
        help="Skip controller-manager cleanup and only stop matching sim processes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.controllers_only and args.processes_only:
        raise SystemExit("--controllers-only and --processes-only are mutually exclusive")

    if not args.processes_only:
        report = cleanup_controllers(
            controller_manager=args.controller_manager,
            timeout_sec=args.timeout_sec,
            dry_run=args.dry_run,
        )
        prefix = "would " if args.dry_run else ""
        if report.controller_manager_available:
            print(f"loaded controllers: {list(report.loaded_controllers)}")
            print(f"{prefix}deactivate controllers: {list(report.deactivated_controllers)}")
            print(f"{prefix}unload controllers: {list(report.unloaded_controllers)}")
        else:
            print("controller manager unavailable: skipping controller cleanup")

    if args.controllers_only:
        return 0

    processes = find_sbmpc_sim_processes(
        list_system_processes(),
        exclude_pids={os.getpid(), os.getppid()},
    )
    print(f"matched processes: {[(p.pid, p.command) for p in processes]}")
    if args.dry_run:
        return 0

    remaining = terminate_processes(
        processes,
        grace_sec=args.timeout_sec,
    )
    if remaining:
        print(f"processes still alive after SIGTERM: {list(remaining)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
