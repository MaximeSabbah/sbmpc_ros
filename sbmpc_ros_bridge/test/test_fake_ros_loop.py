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
from sbmpc_ros_bridge.lfc_control_probe import estimate_lfc_command
from sbmpc_ros_bridge.lfc_msg_adapter import float64_multi_array_to_numpy
from sbmpc_ros_bridge.safety import make_conservative_bringup_profile


LFC_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)
DEFAULT_WARMUP_ITERATIONS = 3


@dataclass(frozen=True)
class FakePlannerOutput:
    tau_ff: np.ndarray
    K: np.ndarray
    phase: str = "PREGRASP"
    next_phase: str = "DESCEND"
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
    gain_mode: str = "exact_feedback"
    planner_prepare_time_ms: float = 0.4
    planner_command_time_ms: float = 9.5


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

    def gravity_torques(self, q) -> np.ndarray:
        return np.asarray([0.25] * len(q), dtype=np.float64)

    def close(self) -> None:
        self.close_calls += 1


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


class HighTorquePlanner(FakePlanner):
    def step(self, planner_input) -> FakePlannerOutput:
        self.step_inputs.append(planner_input)
        self.step_calls += 1
        return FakePlannerOutput(
            tau_ff=np.asarray([20.0, -30.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            K=np.zeros((7, 14), dtype=np.float64),
            diagnostics=FakePlannerDiagnostics(),
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


def make_sensor(
    position_offset: float = 0.0,
    velocity: list[float] | None = None,
) -> Sensor:
    now = rclpy.clock.Clock().now().to_msg()
    return Sensor(
        header=Header(stamp=now),
        joint_state=JointState(
            header=Header(stamp=now),
            name=list(PANDA_ARM_JOINT_NAMES),
            position=[position_offset + 0.1 * i for i in range(7)],
            velocity=[0.0] * 7 if velocity is None else velocity,
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
        assert snapshot.last_planning_time_ms == pytest.approx(12.5)
        assert snapshot.last_planner_step_wall_time_ms is not None
        assert snapshot.last_position_error == pytest.approx(0.25)
        assert snapshot.last_goal_position == [0.55, 0.0, 0.4]
        assert snapshot.planner_mode == "exact_feedback"
        assert snapshot.last_planner_command_time_ms == pytest.approx(9.5)
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


def test_fake_ros_loop_clips_feedforward_with_bringup_safety_profile() -> None:
    planner = HighTorquePlanner()
    bridge = SbMpcLfcBridgeNode(
        planner=planner,
        safety_profile=make_conservative_bringup_profile(
            max_abs_torque=1.0,
            torque_limit_mode="clip",
        ),
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
        feedforward = float64_multi_array_to_numpy(
            collector.controls[-1].feedforward
        ).reshape(-1)
        np.testing.assert_allclose(
            feedforward,
            np.asarray([1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0]),
        )
        assert bridge.diagnostics_snapshot().last_control_max_abs_feedforward == 1.0
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
        assert snapshot.last_planner_step_wall_time_ms is not None
        assert snapshot.last_planner_step_wall_time_ms > 20.0
        assert snapshot.last_planning_time_ms == pytest.approx(12.5)
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_fake_ros_loop_fails_closed_after_bad_gain() -> None:
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
        with pytest.raises(RuntimeError, match="feedback_gain contains non-finite"):
            spin_for(executor, 0.24)

        assert planner.step_calls >= 2
        assert collector.controls
        for control in collector.controls:
            assert np.all(np.isfinite(control.feedback_gain.data))
        snapshot = bridge.diagnostics_snapshot()
        assert snapshot.state == "error"
        assert snapshot.accepted_planner_output_count >= 1
        assert snapshot.rejected_planner_output_count >= 1
        assert "feedback_gain contains non-finite values" in snapshot.last_error
        assert bridge._latest_planner_output is None
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_fake_ros_loop_does_not_recover_after_transient_bad_gain() -> None:
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
        with pytest.raises(RuntimeError, match="feedback_gain contains non-finite"):
            spin_for(executor, 0.30)

        snapshot = bridge.diagnostics_snapshot()
        assert planner.step_calls == 2
        assert snapshot.state == "error"
        assert snapshot.accepted_planner_output_count == 1
        assert snapshot.rejected_planner_output_count == 1
        assert "feedback_gain contains non-finite values" in snapshot.last_error
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_fake_ros_loop_fails_closed_on_stale_sensor() -> None:
    planner = FakePlanner()
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    bridge.set_parameters(
        [
            Parameter(
                "max_sensor_age_sec",
                Parameter.Type.DOUBLE,
                0.04,
            )
        ]
    )
    bridge._on_sensor(make_sensor())
    assert bridge._last_sensor_arrival_sec is not None
    bridge._last_sensor_arrival_sec -= 1.0

    try:
        with pytest.raises(RuntimeError, match="sensor stream is stale"):
            bridge._on_timer()

        snapshot = bridge.diagnostics_snapshot()
        assert snapshot.state == "error"
        assert "sensor stream is stale" in snapshot.last_error
    finally:
        bridge.destroy_node()


def test_fake_ros_loop_publishes_emergency_hold_before_stale_planner_fatal() -> None:
    class CapturingPublisher:
        def __init__(self) -> None:
            self.messages: list[Control] = []

        def publish(self, message: Control) -> None:
            self.messages.append(message)

    planner = FakePlanner()
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    publisher = CapturingPublisher()
    bridge._control_publisher = publisher
    bridge.set_parameters(
        [
            Parameter(
                "enable_nonzero_control",
                Parameter.Type.BOOL,
                True,
            ),
            Parameter(
                "max_planner_output_age_sec",
                Parameter.Type.DOUBLE,
                0.001,
            ),
        ]
    )
    bridge._warmup_complete = True
    bridge._published_control_count = 1
    bridge._on_sensor(make_sensor())
    planner_input = bridge._snapshot_planner_input()
    assert planner_input is not None
    bridge._store_latest_planner_output(
        planner_input,
        FakePlannerOutput(
            tau_ff=np.asarray([0.5] * 7, dtype=np.float64),
            K=np.eye(7, 14, dtype=np.float64),
        ),
    )
    time.sleep(0.01)

    try:
        with pytest.raises(RuntimeError, match="planner output is stale"):
            bridge._on_timer()

        assert len(publisher.messages) == 1
        hold = publisher.messages[0]
        np.testing.assert_allclose(
            float64_multi_array_to_numpy(hold.feedforward).reshape(-1),
            np.asarray([0.25] * 7, dtype=np.float64),
        )
        gain = float64_multi_array_to_numpy(hold.feedback_gain)
        np.testing.assert_allclose(gain[:, :7], np.eye(7) * 1.0)
        np.testing.assert_allclose(gain[:, 7:], np.eye(7) * 2.0)
        np.testing.assert_allclose(hold.initial_state.joint_state.velocity, [0.0] * 7)
        snapshot = bridge.diagnostics_snapshot()
        assert snapshot.state == "error"
        assert "planner output is stale" in snapshot.last_error
    finally:
        bridge.destroy_node()


def test_fake_ros_loop_adds_feedforward_velocity_damping_with_correct_sign() -> None:
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
                "feedforward_velocity_damping_gain",
                Parameter.Type.DOUBLE,
                2.0,
            ),
        ]
    )
    sensor_publisher = FakeSensorPublisher(enabled=True)
    collector = ControlCollector()
    executor = build_executor(bridge, sensor_publisher, collector)

    try:
        spin_for(executor, 0.20)

        assert collector.controls
        control = collector.controls[-1]
        gain = float64_multi_array_to_numpy(control.feedback_gain)
        np.testing.assert_allclose(gain[:, 7:], np.eye(7) * 2.0)
        np.testing.assert_allclose(control.initial_state.joint_state.velocity, [0.0] * 7)

        measured = make_sensor(velocity=[0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        measured.joint_state.position = list(control.initial_state.joint_state.position)
        estimate = estimate_lfc_command(control, measured)
        assert estimate.feedback_effort[1] == pytest.approx(-1.0)
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


def test_fake_ros_loop_publishes_hold_when_disarmed_after_control() -> None:
    planner = FakePlanner()
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    bridge.set_parameters(
        [
            Parameter(
                "enable_nonzero_control",
                Parameter.Type.BOOL,
                True,
            ),
        ]
    )
    sensor_publisher = FakeSensorPublisher(enabled=True)
    collector = ControlCollector()
    executor = build_executor(bridge, sensor_publisher, collector)

    try:
        spin_for(executor, 0.18)
        assert collector.controls

        control_count_before_disarm = len(collector.controls)
        bridge.set_parameters(
            [
                Parameter(
                    "enable_nonzero_control",
                    Parameter.Type.BOOL,
                    False,
                )
            ]
        )
        spin_for(executor, 0.08)

        assert len(collector.controls) > control_count_before_disarm
        hold = collector.controls[-1]
        np.testing.assert_allclose(
            float64_multi_array_to_numpy(hold.feedforward).reshape(-1),
            np.asarray([0.25] * 7, dtype=np.float64),
        )
        gain = float64_multi_array_to_numpy(hold.feedback_gain)
        np.testing.assert_allclose(gain[:, :7], np.eye(7) * 1.0)
        np.testing.assert_allclose(gain[:, 7:], np.eye(7) * 2.0)
        np.testing.assert_allclose(hold.initial_state.joint_state.velocity, [0.0] * 7)
        assert bridge.diagnostics_snapshot().state == "disarmed_hold"
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


def test_fake_ros_loop_predicts_delayed_planning_state() -> None:
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


def test_control_output_delay_selects_aged_planner_output() -> None:
    from sbmpc_ros_bridge.lfc_bridge_node import _LatestPlannerOutput

    bridge = SbMpcLfcBridgeNode(planner=FakePlanner(), publish_period_sec=0.04)
    try:
        # Three outputs, 40 ms apart, oldest first (the worker appends newest).
        entries = [
            _LatestPlannerOutput(
                planner_input=None,
                planner_output=f"plan_{i}",
                created_at_sec=float(i),
            )
            for i in range(3)
        ]
        for created_sec, entry in zip((0.00, 0.04, 0.08), entries):
            bridge._planner_output_history.append((created_sec, entry))

        # At t=0.13 s with an 80 ms modelled delay, the freshest output already
        # older than 80 ms is the one created at t=0.04 (age 90 ms), not the
        # newest at t=0.08 (age 50 ms).
        selected = bridge._select_delayed_planner_output(now_sec=0.13, delay_sec=0.08)
        assert selected is entries[1]

        # Zero delay falls through to the newest entry via the normal path.
        assert bridge._select_delayed_planner_output(now_sec=0.13, delay_sec=0.0) is entries[2]

        # Before any output is old enough, nothing is published (hold).
        assert bridge._select_delayed_planner_output(now_sec=0.05, delay_sec=0.08) is None

        # Clearing drops the modelled-latency history too.
        bridge._clear_latest_planner_output()
        assert len(bridge._planner_output_history) == 0
    finally:
        bridge.destroy_node()
