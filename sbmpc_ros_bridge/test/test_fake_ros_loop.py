from __future__ import annotations

from dataclasses import dataclass
import json
import time

import numpy as np
import pytest
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from linear_feedback_controller_msgs.msg import Control, Sensor
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, String

from sbmpc_ros_bridge.joint_mapping import PANDA_ARM_JOINT_NAMES
from sbmpc_ros_bridge.lfc_bridge_node import SbMpcLfcBridgeNode
from sbmpc_ros_bridge.lfc_msg_adapter import float64_multi_array_to_numpy


LFC_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)
DEFAULT_WARMUP_ITERATIONS = 10


@dataclass(frozen=True)
class FakePlannerOutput:
    tau_ff: np.ndarray
    K: np.ndarray
    phase: str = "PREGRASP"
    next_phase: str = "DESCEND"
    diagnostics: object | None = None


@dataclass(frozen=True)
class FakeFeedforwardOutput:
    tau_ff: np.ndarray
    phase: str = "PREGRASP"
    next_phase: str = "DESCEND"
    diagnostics: object | None = None


@dataclass(frozen=True)
class FakeGainSnapshot:
    K: np.ndarray
    diagnostics: object | None = None


@dataclass(frozen=True)
class FakePlannerDiagnostics:
    planning_time_ms: float = 12.5
    running_cost: float = 1.25
    gain_norm: float = 2.65
    torque_norm: float = 1.32
    position_error: float = 0.25
    orientation_error: float = 0.01
    object_error: float | None = None
    goal_position: tuple[float, float, float] = (0.55, 0.0, 0.40)
    gain_mode: str = "exact_async_feedback"
    foreground_planning_time_ms: float = 9.5
    background_gain_time_ms: float = 4.25
    async_gain_worker_running: bool = True
    gain_age_cycles: float = 1.0
    gain_window_fill: int = 128
    gain_completed_batch_count: int = 3
    gain_dropped_snapshot_count: int = 1
    async_gain_worker_error: str | None = None


class FakePlanner:
    def __init__(self, *, step_sleep_sec: float = 0.0, on_step=None) -> None:
        self.warmup_calls = 0
        self.step_calls = 0
        self.close_calls = 0
        self.step_inputs = []
        self.step_sleep_sec = step_sleep_sec
        self.on_step = on_step

    def warmup(self) -> FakePlannerOutput:
        self.warmup_calls += 1
        return FakePlannerOutput(
            tau_ff=np.zeros(7, dtype=np.float64),
            K=np.zeros((7, 14), dtype=np.float64),
            diagnostics=FakePlannerDiagnostics(
                gain_norm=0.0,
                torque_norm=0.0,
            ),
        )

    def step(self, planner_input) -> FakePlannerOutput:
        self.step_inputs.append(planner_input)
        self.step_calls += 1
        if self.step_sleep_sec > 0.0:
            time.sleep(self.step_sleep_sec)
        if self.on_step is not None:
            self.on_step()
        return FakePlannerOutput(
            tau_ff=np.asarray([0.5] * 7, dtype=np.float64),
            K=np.eye(7, 14, dtype=np.float64),
            diagnostics=FakePlannerDiagnostics(),
        )

    def close(self) -> None:
        self.close_calls += 1


class SplitFeedforwardGainPlanner(FakePlanner):
    def __init__(self) -> None:
        super().__init__()
        self.feedforward_calls = 0
        self.latest_gain_calls = 0

    def step_feedforward(self, planner_input) -> FakeFeedforwardOutput:
        self.step_inputs.append(planner_input)
        self.feedforward_calls += 1
        return FakeFeedforwardOutput(
            tau_ff=np.asarray([0.75] * 7, dtype=np.float64),
            diagnostics=FakePlannerDiagnostics(
                planning_time_ms=8.0,
                foreground_planning_time_ms=8.0,
                gain_norm=0.0,
                torque_norm=1.98,
            ),
        )

    def latest_gain(self, diagnostics=None) -> FakeGainSnapshot:
        self.latest_gain_calls += 1
        return FakeGainSnapshot(
            K=2.0 * np.eye(7, 14, dtype=np.float64),
            diagnostics=diagnostics,
        )


class SlowGainSnapshotPlanner(SplitFeedforwardGainPlanner):
    def __init__(self, *, gain_sleep_sec: float) -> None:
        super().__init__()
        self.gain_sleep_sec = gain_sleep_sec

    def latest_gain(self, diagnostics=None) -> FakeGainSnapshot:
        time.sleep(self.gain_sleep_sec)
        return super().latest_gain(diagnostics)


class BadGainAfterFirstStepPlanner(FakePlanner):
    def step(self, planner_input) -> FakePlannerOutput:
        output = super().step(planner_input)
        if self.step_calls == 1:
            return output
        return FakePlannerOutput(
            tau_ff=output.tau_ff,
            K=np.full((7, 14), np.nan, dtype=np.float64),
            diagnostics=output.diagnostics,
        )


class BadGainOncePlanner(FakePlanner):
    def step(self, planner_input) -> FakePlannerOutput:
        output = super().step(planner_input)
        if self.step_calls == 2:
            return FakePlannerOutput(
                tau_ff=output.tau_ff,
                K=np.full((7, 14), np.nan, dtype=np.float64),
                diagnostics=output.diagnostics,
            )
        return output


class PredictingPlanner(FakePlanner):
    def predict_state(self, planner_input, tau_ff, duration_sec):
        del tau_ff
        offset = 10.0 * float(duration_sec)
        return planner_input.q + offset, planner_input.v + offset


def make_sensor(position_offset: float = 0.0) -> Sensor:
    now = rclpy.clock.Clock().now().to_msg()
    return Sensor(
        header=Header(stamp=now),
        joint_state=JointState(
            header=Header(stamp=now),
            name=list(PANDA_ARM_JOINT_NAMES),
            position=[position_offset + 0.1 * i for i in range(7)],
            velocity=[0.0] * 7,
            effort=[0.0] * 7,
        ),
    )


class FakeSensorPublisher(Node):
    def __init__(self, *, period_sec: float = 0.01, enabled: bool = True) -> None:
        super().__init__("fake_lfc_sensor_publisher")
        self.publisher = self.create_publisher(Sensor, "sensor", LFC_QOS)
        self.enabled = enabled
        self.timer = self.create_timer(period_sec, self._publish)

    def _publish(self) -> None:
        if not self.enabled:
            return
        self.publisher.publish(make_sensor())


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
            LFC_QOS,
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


class FailingPublisher:
    def publish(self, message) -> None:
        del message
        raise RCLError("publisher's context is invalid")


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
    bridge.set_parameters(
        [
            Parameter(
                "enable_nonzero_control",
                Parameter.Type.BOOL,
                True,
            )
        ]
    )
    sensor_publisher = FakeSensorPublisher(enabled=False)
    collector = ControlCollector()
    executor = build_executor(bridge, sensor_publisher, collector)

    try:
        spin_for(executor, 0.08)
        assert collector.controls == []

        sensor_publisher.enabled = True
        spin_for(executor, 0.20)

        assert len(collector.controls) >= 2
        later_control = collector.controls[-1]
        assert np.all(
            float64_multi_array_to_numpy(collector.controls[0].feedforward).reshape(-1) > 0.0
        )
        assert np.all(
            float64_multi_array_to_numpy(later_control.feedforward).reshape(-1) > 0.0
        )
        assert planner.warmup_calls == DEFAULT_WARMUP_ITERATIONS
        assert planner.step_calls >= 1
        snapshot = bridge.diagnostics_snapshot()
        assert snapshot.state == "running"
        assert snapshot.control_enabled is True
        assert snapshot.force_zero_control is False
        assert snapshot.valid_sensor_count >= 1
        assert snapshot.published_control_count >= 2
        assert snapshot.nonzero_control_count >= 1
        assert snapshot.accepted_planner_output_count >= 1
        assert snapshot.rejected_planner_output_count == 0
        assert snapshot.last_phase == "PREGRASP"
        assert snapshot.last_next_phase == "DESCEND"
        assert snapshot.last_planner_output_time_ms == pytest.approx(12.5)
        assert snapshot.last_bridge_loop_time_ms is not None
        assert snapshot.last_position_error == pytest.approx(0.25)
        assert snapshot.last_goal_position == [0.55, 0.0, 0.4]
        assert snapshot.planner_mode == "exact_async_feedback"
        assert snapshot.last_foreground_planning_time_ms == pytest.approx(9.5)
        assert snapshot.last_background_gain_time_ms == pytest.approx(4.25)
        assert snapshot.last_gain_worker_running is True
        assert snapshot.last_gain_window_fill == 128
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_bridge_destroy_closes_planner() -> None:
    planner = FakePlanner()
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)

    bridge.destroy_node()

    assert planner.close_calls == 1


def test_fake_ros_loop_publishes_controls_near_target_rate_and_diagnostics() -> None:
    planner = FakePlanner()
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    bridge.set_parameters(
        [
            Parameter(
                "enable_nonzero_control",
                Parameter.Type.BOOL,
                True,
            )
        ]
    )
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

        snapshot = bridge.diagnostics_snapshot()
        assert snapshot.published_control_count + 1 >= len(collector.controls)
        last_diag = collector.diagnostics_payloads[-1]
        assert "deadline_miss_count" in last_diag
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_control_thread_publishes_controls() -> None:
    planner = FakePlanner()
    bridge = SbMpcLfcBridgeNode(
        planner=planner,
        publish_period_sec=0.02,
    )
    bridge.set_parameters(
        [
            Parameter(
                "enable_nonzero_control",
                Parameter.Type.BOOL,
                True,
            )
        ]
    )
    sensor_publisher = FakeSensorPublisher(enabled=True)
    collector = ControlCollector()
    executor = build_executor(bridge, sensor_publisher, collector)

    try:
        spin_for(executor, 0.30)

        assert len(collector.controls) >= 4
        assert planner.step_calls >= 1
        snapshot = bridge.diagnostics_snapshot()
        assert snapshot.state == "running"
        assert snapshot.published_control_count >= len(collector.controls)
        assert snapshot.accepted_planner_output_count >= 1
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_fake_ros_loop_composes_feedforward_and_gain_from_separate_buffers() -> None:
    planner = SplitFeedforwardGainPlanner()
    bridge = SbMpcLfcBridgeNode(
        planner=planner,
        publish_period_sec=0.02,
    )
    bridge.set_parameters(
        [
            Parameter(
                "enable_nonzero_control",
                Parameter.Type.BOOL,
                True,
            )
        ]
    )
    sensor_publisher = FakeSensorPublisher(enabled=True)
    collector = ControlCollector()
    executor = build_executor(bridge, sensor_publisher, collector)

    try:
        spin_for(executor, 0.25)

        assert collector.controls
        assert planner.feedforward_calls >= 1
        assert planner.latest_gain_calls >= 1
        latest = collector.controls[-1]
        np.testing.assert_allclose(
            float64_multi_array_to_numpy(latest.feedforward),
            np.full((7, 1), 0.75, dtype=np.float64),
        )
        feedback_gain = float64_multi_array_to_numpy(latest.feedback_gain)
        assert feedback_gain.shape == (7, 14)
        assert np.linalg.norm(feedback_gain) > 0.0
        snapshot = bridge.diagnostics_snapshot()
        assert snapshot.accepted_planner_output_count >= 1
        assert snapshot.last_foreground_planning_time_ms == pytest.approx(8.0)
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_fake_ros_loop_slow_gain_snapshot_does_not_block_feedforward() -> None:
    planner = SlowGainSnapshotPlanner(gain_sleep_sec=0.10)
    bridge = SbMpcLfcBridgeNode(
        planner=planner,
        publish_period_sec=0.02,
    )
    bridge.set_parameters(
        [
            Parameter(
                "enable_nonzero_control",
                Parameter.Type.BOOL,
                True,
            )
        ]
    )
    sensor_publisher = FakeSensorPublisher(enabled=True)
    collector = ControlCollector()
    executor = build_executor(bridge, sensor_publisher, collector)

    try:
        spin_for(executor, 0.35)

        assert collector.controls
        assert planner.latest_gain_calls >= 1
        assert planner.feedforward_calls >= 5
        assert planner.feedforward_calls > planner.latest_gain_calls
        snapshot = bridge.diagnostics_snapshot()
        assert snapshot.accepted_planner_output_count >= 5
        assert snapshot.last_planner_step_wall_time_ms is not None
        assert snapshot.last_planner_step_wall_time_ms < 50.0
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_fake_ros_loop_counts_deadline_misses() -> None:
    planner = FakePlanner(step_sleep_sec=0.03)
    bridge = SbMpcLfcBridgeNode(
        planner=planner,
        publish_period_sec=0.02,
        planner_deadline_sec=0.02,
    )
    bridge.set_parameters(
        [
            Parameter(
                "enable_nonzero_control",
                Parameter.Type.BOOL,
                True,
            )
        ]
    )
    sensor_publisher = FakeSensorPublisher(enabled=True)
    collector = ControlCollector()
    executor = build_executor(bridge, sensor_publisher, collector)

    try:
        spin_for(executor, 0.22)

        assert len(collector.controls) >= 3
        intervals = np.diff(collector.control_receive_times[1:])
        assert intervals.size >= 1
        assert float(np.mean(intervals)) <= 0.04

        snapshot = bridge.diagnostics_snapshot()
        assert snapshot.deadline_miss_count >= 1
        assert snapshot.accepted_planner_output_count >= 1
        assert snapshot.rejected_planner_output_count == 0
        assert snapshot.last_planning_time_ms is not None
        assert snapshot.last_planning_time_ms > 20.0
        assert snapshot.last_bridge_loop_time_ms == pytest.approx(
            snapshot.last_planning_time_ms
        )
        assert snapshot.last_planner_output_time_ms == pytest.approx(12.5)
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_fake_ros_loop_keeps_last_valid_control_after_bad_gain() -> None:
    planner = BadGainAfterFirstStepPlanner()
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    bridge.set_parameters(
        [
            Parameter(
                "enable_nonzero_control",
                Parameter.Type.BOOL,
                True,
            )
        ]
    )
    sensor_publisher = FakeSensorPublisher(enabled=True)
    collector = ControlCollector()
    executor = build_executor(bridge, sensor_publisher, collector)

    try:
        spin_for(executor, 0.24)

        assert planner.step_calls >= 2
        assert len(collector.controls) >= 3
        for control in collector.controls:
            assert np.all(np.isfinite(control.feedback_gain.data))
        snapshot = bridge.diagnostics_snapshot()
        assert snapshot.accepted_planner_output_count >= 1
        assert snapshot.rejected_planner_output_count >= 1
        assert "feedback_gain contains non-finite values" in (
            snapshot.last_error
        )
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_fake_ros_loop_clears_error_after_next_valid_control() -> None:
    planner = BadGainOncePlanner()
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    bridge.set_parameters(
        [
            Parameter(
                "enable_nonzero_control",
                Parameter.Type.BOOL,
                True,
            )
        ]
    )
    sensor_publisher = FakeSensorPublisher(enabled=True)
    collector = ControlCollector()
    executor = build_executor(bridge, sensor_publisher, collector)

    try:
        spin_for(executor, 0.30)

        snapshot = bridge.diagnostics_snapshot()
        assert planner.step_calls >= 3
        assert snapshot.accepted_planner_output_count >= 2
        assert snapshot.rejected_planner_output_count == 1
        assert snapshot.last_error == ""
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_fake_ros_loop_stays_zero_until_nonzero_control_is_enabled() -> None:
    planner = FakePlanner()
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    sensor_publisher = FakeSensorPublisher(enabled=True)
    collector = ControlCollector()
    executor = build_executor(bridge, sensor_publisher, collector)

    try:
        spin_for(executor, 0.20)

        assert collector.controls == []
        assert planner.warmup_calls == DEFAULT_WARMUP_ITERATIONS
        assert planner.step_calls == 0
        snapshot = bridge.diagnostics_snapshot()
        assert snapshot.state == "armed_idle"
        assert snapshot.control_enabled is False
        assert snapshot.nonzero_control_count == 0

        control_count_before_enable = len(collector.controls)
        bridge.set_parameters(
            [
                Parameter(
                    "enable_nonzero_control",
                    Parameter.Type.BOOL,
                    True,
                )
            ]
        )
        spin_for(executor, 0.12)

        assert len(collector.controls) > control_count_before_enable
        assert any(
            np.any(
                float64_multi_array_to_numpy(control.feedforward).reshape(-1) > 0.0
            )
            for control in collector.controls[control_count_before_enable:]
        )
        assert planner.step_calls >= 1
        snapshot = bridge.diagnostics_snapshot()
        assert snapshot.state == "running"
        assert snapshot.control_enabled is True
        assert snapshot.last_control_max_abs_feedforward == pytest.approx(0.5)
        assert snapshot.last_control_gain_norm is not None
        assert snapshot.last_control_gain_norm > 0.0
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_fake_ros_loop_retimes_control_initial_state_to_latest_sensor() -> None:
    bridge: SbMpcLfcBridgeNode | None = None

    def publish_new_sensor_during_planning() -> None:
        assert bridge is not None
        bridge._on_sensor(make_sensor(position_offset=1.0))

    planner = FakePlanner(on_step=publish_new_sensor_during_planning)
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    bridge.set_parameters(
        [
            Parameter(
                "enable_nonzero_control",
                Parameter.Type.BOOL,
                True,
            )
        ]
    )
    sensor_publisher = FakeSensorPublisher(enabled=False)
    collector = ControlCollector()
    executor = build_executor(bridge, sensor_publisher, collector)

    try:
        bridge._on_sensor(make_sensor(position_offset=0.0))
        spin_for(executor, 0.12)

        assert collector.controls
        np.testing.assert_allclose(
            collector.controls[-1].initial_state.joint_state.position,
            [1.0 + 0.1 * i for i in range(7)],
        )
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_fake_ros_loop_can_keep_planned_initial_state_for_delayed_control() -> None:
    bridge: SbMpcLfcBridgeNode | None = None

    def publish_new_sensor_during_planning() -> None:
        assert bridge is not None
        bridge._on_sensor(make_sensor(position_offset=1.0))

    planner = FakePlanner(on_step=publish_new_sensor_during_planning)
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    bridge.set_parameters(
        [
            Parameter(
                "enable_nonzero_control",
                Parameter.Type.BOOL,
                True,
            ),
            Parameter(
                "retime_control_initial_state",
                Parameter.Type.BOOL,
                False,
            ),
        ]
    )
    sensor_publisher = FakeSensorPublisher(enabled=False)
    collector = ControlCollector()
    executor = build_executor(bridge, sensor_publisher, collector)

    try:
        bridge._on_sensor(make_sensor(position_offset=0.0))
        spin_for(executor, 0.12)

        assert collector.controls
        np.testing.assert_allclose(
            collector.controls[0].initial_state.joint_state.position,
            [0.1 * i for i in range(7)],
        )
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_fake_ros_loop_predicts_delayed_planning_state_when_not_retiming() -> None:
    planner = PredictingPlanner()
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    bridge.set_parameters(
        [
            Parameter(
                "enable_nonzero_control",
                Parameter.Type.BOOL,
                True,
            ),
            Parameter(
                "retime_control_initial_state",
                Parameter.Type.BOOL,
                False,
            ),
            Parameter(
                "control_initial_state_prediction_sec",
                Parameter.Type.DOUBLE,
                0.02,
            ),
        ]
    )
    sensor_publisher = FakeSensorPublisher(enabled=True)
    collector = ControlCollector()
    executor = build_executor(bridge, sensor_publisher, collector)

    try:
        spin_for(executor, 0.18)

        assert planner.step_calls >= 2
        np.testing.assert_allclose(
            planner.step_inputs[-1].q,
            np.asarray([0.1 * i for i in range(7)], dtype=np.float64) + 0.2,
        )
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_fake_ros_loop_force_zero_control_gate_blocks_nonzero_outputs() -> None:
    planner = FakePlanner()
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    bridge.set_parameters(
        [
            Parameter(
                "enable_nonzero_control",
                Parameter.Type.BOOL,
                True,
            ),
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
        assert planner.warmup_calls == DEFAULT_WARMUP_ITERATIONS
        assert planner.step_calls == 0
        snapshot = bridge.diagnostics_snapshot()
        assert snapshot.state == "gated_zero_control"
        assert snapshot.control_enabled is True
        assert snapshot.force_zero_control is True
        assert snapshot.nonzero_control_count == 0
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_fake_ros_loop_ignores_publish_failures_after_shutdown(monkeypatch) -> None:
    bridge = SbMpcLfcBridgeNode(planner=FakePlanner(), publish_period_sec=0.02)
    monkeypatch.setattr(rclpy, "ok", lambda context=None: False)
    bridge._control_publisher = FailingPublisher()
    bridge._diagnostics_publisher = FailingPublisher()

    try:
        bridge._publish_control(Control())
        bridge._publish_diagnostics()
        assert bridge.diagnostics_snapshot().published_control_count == 0
    finally:
        bridge.destroy_node()
