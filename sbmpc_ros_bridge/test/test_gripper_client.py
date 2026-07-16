"""P-B1 — the gripper action client and its bridge wiring (P3).

One control_msgs/GripperCommand client serves both backends; these tests
run it against a fake action server (goal semantics, result verification,
fail-closed paths) and drive the bridge end-to-end with a fake planner
(dispatch on command flips only, plan-clock freeze while a goal is in
flight, fatal escalation on gripper failure).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import rclpy
from rclpy.action import ActionServer, GoalResponse
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

control_msgs = pytest.importorskip("control_msgs.action")
from control_msgs.action import GripperCommand  # noqa: E402

from test_fake_ros_loop import (  # noqa: E402
    FakePlanner,
    FakePlannerOutput,
    FakeSensorPublisher,
    build_executor,
    spin_for,
    teardown_executor,
)

from sbmpc_ros_bridge.gripper_client import GripperCommandClient  # noqa: E402
from sbmpc_ros_bridge.lfc_bridge_node import SbMpcLfcBridgeNode  # noqa: E402

ACTION_NAME = "/fake_gripper/gripper_cmd"


class FakeGripperServer(Node):
    """Scriptable GripperCommand action server (one goal at a time)."""

    def __init__(
        self,
        *,
        action_name: str = ACTION_NAME,
        reached_goal: bool = True,
        stalled: bool = False,
        reject: bool = False,
        abort: bool = False,
        delay_sec: float = 0.0,
    ) -> None:
        super().__init__("fake_gripper_server")
        self.goals: list[GripperCommand.Goal] = []
        self.reached_goal = reached_goal
        self.stalled = stalled
        self.reject = reject
        self.abort = abort
        self.delay_sec = delay_sec
        self._server = ActionServer(
            self,
            GripperCommand,
            action_name,
            execute_callback=self._execute,
            goal_callback=self._on_goal,
        )

    def _on_goal(self, goal_request) -> GoalResponse:
        self.goals.append(goal_request)
        return GoalResponse.REJECT if self.reject else GoalResponse.ACCEPT

    def _execute(self, goal_handle):
        if self.delay_sec > 0.0:
            time.sleep(self.delay_sec)
        result = GripperCommand.Result()
        result.position = float(goal_handle.request.command.position)
        result.reached_goal = bool(self.reached_goal)
        result.stalled = bool(self.stalled)
        if self.abort:
            goal_handle.abort()
        else:
            goal_handle.succeed()
        return result


class ClientHost(Node):
    def __init__(self) -> None:
        super().__init__("gripper_client_host")


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


def run_command(
    command: str,
    server: FakeGripperServer,
    *,
    close_position: float = 0.0,
    open_position: float = 0.04,
    max_effort: float = 40.0,
    spin_sec: float = 1.5,
) -> GripperCommandClient:
    host = ClientHost()
    client = GripperCommandClient(
        host,
        action_name=ACTION_NAME,
        close_position=close_position,
        open_position=open_position,
        max_effort=max_effort,
    )
    executor = build_executor(host, server)
    try:
        spin_for(executor, 0.2)  # server discovery
        client.execute(command)
        deadline = time.monotonic() + spin_sec
        while (
            client.busy
            and client.failure is None
            and time.monotonic() < deadline
        ):
            executor.spin_once(timeout_sec=0.02)
        return client
    finally:
        teardown_executor(executor, host, server)


def test_close_goal_carries_the_configured_position_and_effort() -> None:
    server = FakeGripperServer(reached_goal=False, stalled=True)
    client = run_command(
        "close", server, close_position=0.005, max_effort=25.0
    )
    assert len(server.goals) == 1
    assert server.goals[0].command.position == pytest.approx(0.005)
    assert server.goals[0].command.max_effort == pytest.approx(25.0)
    # Closing on the cube stalls the sim's effort controller: verified.
    assert client.failure is None
    assert not client.busy
    assert client.snapshot()["goal_count"] == 1
    assert client.snapshot()["last_result"]["stalled"] is True


def test_open_verifies_on_reached_goal() -> None:
    server = FakeGripperServer(reached_goal=True)
    client = run_command("open", server, open_position=0.04)
    assert server.goals[0].command.position == pytest.approx(0.04)
    assert client.failure is None
    assert not client.busy


def test_open_that_only_stalls_is_a_failure() -> None:
    # An open that stalls did NOT release: fail closed.
    server = FakeGripperServer(reached_goal=False, stalled=True)
    client = run_command("open", server)
    assert client.failure is not None
    assert "did not verify" in client.failure


def test_close_with_neither_reach_nor_stall_is_a_failure() -> None:
    server = FakeGripperServer(reached_goal=False, stalled=False)
    client = run_command("close", server)
    assert client.failure is not None
    assert "did not verify" in client.failure


def test_aborted_goal_is_a_failure() -> None:
    server = FakeGripperServer(abort=True)
    client = run_command("close", server)
    assert client.failure is not None
    assert "status" in client.failure


def test_rejected_goal_is_a_failure() -> None:
    server = FakeGripperServer(reject=True)
    client = run_command("close", server)
    assert client.failure is not None
    assert "rejected" in client.failure


def test_missing_server_is_a_failure() -> None:
    host = ClientHost()
    client = GripperCommandClient(host, action_name="/nobody/home")
    try:
        client.execute("close")
        assert client.failure is not None
        assert "not available" in client.failure
    finally:
        host.destroy_node()


def test_overlapping_commands_fail_closed() -> None:
    server = FakeGripperServer(delay_sec=0.3)
    host = ClientHost()
    client = GripperCommandClient(host, action_name=ACTION_NAME)
    executor = build_executor(host, server)
    try:
        spin_for(executor, 0.2)
        client.execute("close")
        assert client.busy
        assert client.busy_duration_sec() is not None
        client.execute("open")
        assert client.failure is not None
        assert "in flight" in client.failure
    finally:
        teardown_executor(executor, host, server)


def test_unknown_command_is_a_failure() -> None:
    host = ClientHost()
    client = GripperCommandClient(host, action_name=ACTION_NAME)
    try:
        client.execute("grab")
        assert client.failure is not None
    finally:
        host.destroy_node()


# --- bridge wiring ------------------------------------------------------


class GripperPlanner(FakePlanner):
    """Fake planner whose gripper_command is scripted per step and that
    records the bridge's set_gripper_wait calls (the plan-clock freeze)."""

    def __init__(self, commands: list[str]) -> None:
        super().__init__()
        self.commands = commands
        self.wait_calls: list[bool] = []

    def set_gripper_wait(self, wait: bool) -> None:
        self.wait_calls.append(bool(wait))

    def step(self, planner_input) -> FakePlannerOutput:
        output = super().step(planner_input)
        command = self.commands[
            min(self.step_calls - 1, len(self.commands) - 1)
        ]
        return FakePlannerOutput(
            tau_ff=output.tau_ff,
            K=output.K,
            diagnostics=output.diagnostics,
            gripper_command=command,
        )


def make_gripper_bridge(planner: GripperPlanner) -> SbMpcLfcBridgeNode:
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    bridge._control_enabled = True
    bridge._gripper_client = bridge._build_gripper_client(ACTION_NAME)
    return bridge


def test_bridge_dispatches_only_on_command_flips() -> None:
    # first output seeds the baseline ("open"); the flip to "close" sends
    # EXACTLY one goal; repeated "close" sends nothing more
    planner = GripperPlanner(["open", "open", "close", "close", "close"])
    server = FakeGripperServer(reached_goal=False, stalled=True)
    bridge = make_gripper_bridge(planner)
    sensor_publisher = FakeSensorPublisher(enabled=True)
    executor = build_executor(bridge, sensor_publisher, server)
    try:
        spin_for(executor, 0.5)
        assert planner.step_calls >= 4
        assert len(server.goals) == 1
        snapshot = bridge.diagnostics_snapshot()
        assert snapshot.last_gripper_command == "close"
        assert snapshot.gripper["goal_count"] == 1
        assert snapshot.gripper["failure"] is None
        # the clock resumed after the action result
        assert planner.wait_calls[-1] is False
    finally:
        teardown_executor(executor, bridge, sensor_publisher, server)


def test_bridge_freezes_the_plan_clock_while_the_action_runs() -> None:
    planner = GripperPlanner(["open", "close", "close", "close", "close"])
    server = FakeGripperServer(
        reached_goal=False, stalled=True, delay_sec=0.15
    )
    bridge = make_gripper_bridge(planner)
    sensor_publisher = FakeSensorPublisher(enabled=True)
    # Multi-threaded: the fake server's blocking execute must not starve
    # the bridge timer, or no planner step lands inside the in-flight
    # window this test observes.
    from rclpy.executors import MultiThreadedExecutor

    executor = MultiThreadedExecutor(num_threads=3)
    for node in (bridge, sensor_publisher, server):
        executor.add_node(node)
    try:
        spin_for(executor, 0.6)
        # while the goal was in flight the bridge asked the adapter to
        # freeze; afterwards it released it
        assert True in planner.wait_calls
        assert planner.wait_calls[-1] is False
        assert bridge.diagnostics_snapshot().gripper["failure"] is None
    finally:
        teardown_executor(executor, bridge, sensor_publisher, server)


def test_bridge_latches_fatal_on_gripper_failure() -> None:
    planner = GripperPlanner(["open", "close", "close", "close"])
    server = FakeGripperServer(abort=True)
    bridge = make_gripper_bridge(planner)
    sensor_publisher = FakeSensorPublisher(enabled=True)
    executor = build_executor(bridge, sensor_publisher, server)
    try:
        with pytest.raises(RuntimeError, match="[Gg]ripper"):
            spin_for(executor, 0.8)
        snapshot = bridge.diagnostics_snapshot()
        assert snapshot.state == "error"
        assert "ripper" in snapshot.last_error
    finally:
        teardown_executor(executor, bridge, sensor_publisher, server)


def test_bridge_without_gripper_client_only_warns_on_commands() -> None:
    planner = GripperPlanner(["open", "close", "close"])
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    bridge._control_enabled = True
    assert bridge._gripper_client is None  # gripper_action_name defaults ""
    sensor_publisher = FakeSensorPublisher(enabled=True)
    executor = build_executor(bridge, sensor_publisher)
    try:
        spin_for(executor, 0.3)
        assert planner.step_calls >= 2
        snapshot = bridge.diagnostics_snapshot()
        assert snapshot.state == "running"  # no fatal, transport keeps going
        assert snapshot.last_gripper_command == "close"
        assert snapshot.gripper is None
    finally:
        teardown_executor(executor, bridge, sensor_publisher)
