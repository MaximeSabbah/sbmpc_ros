from __future__ import annotations

from dataclasses import dataclass

import pytest

from sbmpc_bringup.launch_preflight import (
    check_clean_ros_graph,
    check_clean_sim_runtime,
    find_stale_sim_processes,
    find_stale_sim_nodes,
    parse_ros_node_list,
    wait_for_clean_sim_runtime,
    wait_for_clean_ros_graph,
)


@dataclass(frozen=True)
class FakeRunResult:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def test_parse_ros_node_list_ignores_empty_lines() -> None:
    assert parse_ros_node_list("\n/controller_manager\n\n/rviz2\n") == (
        "/controller_manager",
        "/rviz2",
    )


def test_find_stale_sim_nodes_flags_known_sbmpc_sim_nodes_once() -> None:
    stale = find_stale_sim_nodes(
        [
            "/controller_manager",
            "/some_unrelated_node",
            "/linear_feedback_controller",
            "/controller_manager",
        ]
    )

    assert stale == ("/controller_manager", "/linear_feedback_controller")


def test_find_stale_sim_processes_flags_orphaned_mujoco_control_node() -> None:
    stale = find_stale_sim_processes(
        "\n".join(
            [
                "11142 /opt/sbmpc_deps_ws/install/mujoco_ros2_control/lib/"
                "mujoco_ros2_control/ros2_control_node --ros-args",
                "22222 /usr/bin/python unrelated.py",
            ]
        )
    )

    assert len(stale) == 1
    assert stale[0].pid == 11142
    assert "ros2_control_node" in stale[0].command


def test_check_clean_ros_graph_passes_when_only_unrelated_nodes_exist() -> None:
    def runner(*args, **kwargs) -> FakeRunResult:
        del args, kwargs
        return FakeRunResult(stdout="/some_unrelated_node\n")

    result = check_clean_ros_graph(runner=runner)

    assert result.nodes == ("/some_unrelated_node",)
    assert result.stale_nodes == ()


def test_check_clean_ros_graph_reports_stale_controller_manager() -> None:
    def runner(*args, **kwargs) -> FakeRunResult:
        del args, kwargs
        return FakeRunResult(stdout="/controller_manager\n/joint_state_estimator\n")

    result = check_clean_ros_graph(runner=runner)

    assert result.stale_nodes == ("/controller_manager", "/joint_state_estimator")


def test_check_clean_sim_runtime_reports_stale_nodes_and_processes() -> None:
    def graph_runner(*args, **kwargs) -> FakeRunResult:
        del args, kwargs
        return FakeRunResult(stdout="/controller_manager\n")

    def process_runner(*args, **kwargs) -> FakeRunResult:
        del args, kwargs
        return FakeRunResult(
            stdout=(
                "11142 /opt/sbmpc_deps_ws/install/mujoco_ros2_control/lib/"
                "mujoco_ros2_control/ros2_control_node --ros-args\n"
            )
        )

    result = check_clean_sim_runtime(
        graph_runner=graph_runner,
        process_runner=process_runner,
    )

    assert result.stale_nodes == ("/controller_manager",)
    assert tuple(process.pid for process in result.stale_processes) == (11142,)


def test_check_clean_ros_graph_errors_when_node_list_fails() -> None:
    def runner(*args, **kwargs) -> FakeRunResult:
        del args, kwargs
        return FakeRunResult(returncode=1, stderr="daemon unavailable")

    with pytest.raises(RuntimeError, match="daemon unavailable"):
        check_clean_ros_graph(runner=runner)


def test_wait_for_clean_ros_graph_rechecks_transient_stale_nodes() -> None:
    calls = 0

    def runner(*args, **kwargs) -> FakeRunResult:
        nonlocal calls
        del args, kwargs
        calls += 1
        if calls == 1:
            return FakeRunResult(stdout="/mujoco_ros2_control_node\n")
        return FakeRunResult(stdout="")

    result = wait_for_clean_ros_graph(
        runner=runner,
        timeout_sec=1.0,
        poll_sec=0.0,
    )

    assert result.stale_nodes == ()
    assert calls == 2


def test_wait_for_clean_sim_runtime_rechecks_transient_stale_processes() -> None:
    calls = 0

    def graph_runner(*args, **kwargs) -> FakeRunResult:
        del args, kwargs
        return FakeRunResult(stdout="")

    def process_runner(*args, **kwargs) -> FakeRunResult:
        nonlocal calls
        del args, kwargs
        calls += 1
        if calls == 1:
            return FakeRunResult(
                stdout=(
                    "11142 /opt/sbmpc_deps_ws/install/mujoco_ros2_control/lib/"
                    "mujoco_ros2_control/ros2_control_node --ros-args\n"
                )
            )
        return FakeRunResult(stdout="")

    result = wait_for_clean_sim_runtime(
        graph_runner=graph_runner,
        process_runner=process_runner,
        timeout_sec=1.0,
        poll_sec=0.0,
    )

    assert result.stale_processes == ()
    assert calls == 2


def test_wait_for_clean_ros_graph_returns_persistent_stale_nodes() -> None:
    def runner(*args, **kwargs) -> FakeRunResult:
        del args, kwargs
        return FakeRunResult(stdout="/mujoco_ros2_control_node\n")

    result = wait_for_clean_ros_graph(
        runner=runner,
        timeout_sec=0.0,
        poll_sec=0.0,
    )

    assert result.stale_nodes == ("/mujoco_ros2_control_node",)
