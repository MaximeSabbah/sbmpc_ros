from __future__ import annotations

from dataclasses import dataclass

import pytest

from sbmpc_bringup.launch_preflight import (
    check_clean_ros_graph,
    find_stale_sim_nodes,
    parse_ros_node_list,
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


def test_check_clean_ros_graph_errors_when_node_list_fails() -> None:
    def runner(*args, **kwargs) -> FakeRunResult:
        del args, kwargs
        return FakeRunResult(returncode=1, stderr="daemon unavailable")

    with pytest.raises(RuntimeError, match="daemon unavailable"):
        check_clean_ros_graph(runner=runner)
