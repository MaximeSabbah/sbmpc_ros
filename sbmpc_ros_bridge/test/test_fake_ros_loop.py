from __future__ import annotations

from dataclasses import dataclass
import json
import time

import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from linear_feedback_controller_msgs.msg import Control, Sensor
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, String

from sbmpc_ros_bridge.joint_mapping import PANDA_ARM_JOINT_NAMES
from sbmpc_ros_bridge.lfc_bridge_node import SbMpcLfcBridgeNode
from sbmpc_ros_bridge.lfc_msg_adapter import float64_multi_array_to_numpy


@dataclass(frozen=True)
class FakePlannerOutput:
    tau_ff: np.ndarray
    K: np.ndarray


class FakePlanner:
    def __init__(self, *, step_sleep_sec: float = 0.0) -> None:
        self.warmup_calls = 0
        self.step_calls = 0
        self.step_sleep_sec = step_sleep_sec

    def warmup(self) -> FakePlannerOutput:
        self.warmup_calls += 1
        return FakePlannerOutput(
            tau_ff=np.zeros(7, dtype=np.float64),
            K=np.zeros((7, 14), dtype=np.float64),
        )

    def step(self, planner_input) -> FakePlannerOutput:
        del planner_input
        self.step_calls += 1
        if self.step_sleep_sec > 0.0:
            time.sleep(self.step_sleep_sec)
        return FakePlannerOutput(
            tau_ff=np.asarray([0.5] * 7, dtype=np.float64),
            K=np.eye(7, 14, dtype=np.float64),
        )


class FakeSensorPublisher(Node):
    def __init__(self, *, period_sec: float = 0.01, enabled: bool = True) -> None:
        super().__init__("fake_lfc_sensor_publisher")
        self.publisher = self.create_publisher(Sensor, "sensor", 10)
        self.enabled = enabled
        self.timer = self.create_timer(period_sec, self._publish)

    def _publish(self) -> None:
        if not self.enabled:
            return
        msg = Sensor()
        now = self.get_clock().now().to_msg()
        msg.header.stamp = now
        msg.joint_state = JointState(
            header=Header(stamp=now),
            name=list(PANDA_ARM_JOINT_NAMES),
            position=[0.1 * i for i in range(7)],
            velocity=[0.0] * 7,
            effort=[0.0] * 7,
        )
        self.publisher.publish(msg)


class ControlCollector(Node):
    def __init__(self) -> None:
        super().__init__("fake_lfc_control_collector")
        self.controls: list[Control] = []
        self.control_receive_times: list[float] = []
        self.diagnostics_payloads: list[dict[str, object]] = []
        self.control_subscription = self.create_subscription(
            Control,
            "control",
            self._on_control,
            10,
        )
        self.diagnostics_subscription = self.create_subscription(
            String,
            "diagnostics",
            self._on_diagnostics,
            10,
        )

    def _on_control(self, message: Control) -> None:
        self.controls.append(message)
        self.control_receive_times.append(time.monotonic())

    def _on_diagnostics(self, message: String) -> None:
        self.diagnostics_payloads.append(json.loads(message.data))


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


def spin_for(executor: SingleThreadedExecutor, duration_sec: float) -> None:
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.01)


def build_executor(*nodes: Node) -> SingleThreadedExecutor:
    executor = SingleThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
    return executor


def teardown_executor(executor: SingleThreadedExecutor, *nodes: Node) -> None:
    for node in nodes:
        executor.remove_node(node)
        node.destroy_node()
    executor.shutdown()


def test_fake_ros_loop_waits_for_sensor_then_warmup_then_nonzero_control() -> None:
    planner = FakePlanner()
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    sensor_publisher = FakeSensorPublisher(enabled=False)
    collector = ControlCollector()
    executor = build_executor(bridge, sensor_publisher, collector)

    try:
        spin_for(executor, 0.08)
        assert collector.controls == []

        sensor_publisher.enabled = True
        spin_for(executor, 0.20)

        assert len(collector.controls) >= 2
        first_control = collector.controls[0]
        later_control = collector.controls[-1]
        np.testing.assert_allclose(
            float64_multi_array_to_numpy(first_control.feedforward),
            np.zeros((7, 1), dtype=np.float64),
        )
        assert np.all(
            float64_multi_array_to_numpy(later_control.feedforward).reshape(-1) > 0.0
        )
        assert planner.warmup_calls == 1
        assert planner.step_calls >= 1
        snapshot = bridge.diagnostics_snapshot()
        assert snapshot.state == "running"
        assert snapshot.valid_sensor_count >= 1
        assert snapshot.published_control_count >= 2
        assert snapshot.nonzero_control_count >= 1
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_fake_ros_loop_publishes_controls_near_target_rate_and_diagnostics() -> None:
    planner = FakePlanner()
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    sensor_publisher = FakeSensorPublisher(enabled=True)
    collector = ControlCollector()
    executor = build_executor(bridge, sensor_publisher, collector)

    try:
        spin_for(executor, 0.35)

        assert len(collector.controls) >= 8
        assert collector.diagnostics_payloads
        intervals = np.diff(collector.control_receive_times[1:])
        assert intervals.size >= 3
        mean_interval = float(np.mean(intervals))
        assert 0.01 <= mean_interval <= 0.05

        last_diag = collector.diagnostics_payloads[-1]
        assert int(last_diag["published_control_count"]) >= len(collector.controls)
        assert "deadline_miss_count" in last_diag
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_fake_ros_loop_counts_deadline_misses() -> None:
    planner = FakePlanner(step_sleep_sec=0.03)
    bridge = SbMpcLfcBridgeNode(
        planner=planner,
        publish_period_sec=0.02,
        planner_deadline_sec=0.02,
    )
    sensor_publisher = FakeSensorPublisher(enabled=True)
    collector = ControlCollector()
    executor = build_executor(bridge, sensor_publisher, collector)

    try:
        spin_for(executor, 0.22)

        snapshot = bridge.diagnostics_snapshot()
        assert snapshot.deadline_miss_count >= 1
        assert snapshot.last_planning_time_ms is not None
        assert snapshot.last_planning_time_ms > 20.0
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_fake_ros_loop_force_zero_control_gate_blocks_nonzero_outputs() -> None:
    planner = FakePlanner()
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    bridge.set_parameters(
        [
            Parameter(
                "force_zero_control",
                Parameter.Type.BOOL,
                True,
            )
        ]
    )
    sensor_publisher = FakeSensorPublisher(enabled=True)
    collector = ControlCollector()
    executor = build_executor(bridge, sensor_publisher, collector)

    try:
        spin_for(executor, 0.20)

        assert collector.controls
        for control in collector.controls:
            np.testing.assert_allclose(
                float64_multi_array_to_numpy(control.feedforward),
                np.zeros((7, 1), dtype=np.float64),
            )
            np.testing.assert_allclose(
                float64_multi_array_to_numpy(control.feedback_gain),
                np.zeros((7, 14), dtype=np.float64),
            )
        assert planner.warmup_calls == 1
        assert planner.step_calls == 0
        snapshot = bridge.diagnostics_snapshot()
        assert snapshot.state == "gated_zero_control"
        assert snapshot.nonzero_control_count == 0
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)
