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
from std_srvs.srv import SetBool

from sbmpc_ros_bridge.joint_mapping import PANDA_ARM_JOINT_NAMES
from sbmpc_ros_bridge.lfc_bridge_node import SbMpcLfcBridgeNode
from sbmpc_ros_bridge.lfc_control_probe import estimate_lfc_command
from sbmpc_ros_bridge.lfc_msg_adapter import float64_multi_array_to_numpy


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
    reference_q: np.ndarray | None = None
    reference_v: np.ndarray | None = None
    gripper_command: object | None = None


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
    phase_machine: dict[str, object] | None = None


class FakePlanner:
    def __init__(
        self,
        *,
        step_sleep_sec: float = 0.0,
        on_step=None,
        gain_mode: str = "exact_feedback",
        feedback_gain: np.ndarray | None = None,
        reference_q: np.ndarray | None = None,
        reference_v: np.ndarray | None = None,
    ) -> None:
        self.warmup_calls = 0
        self.step_calls = 0
        self.close_calls = 0
        self.step_inputs = []
        self.step_sleep_sec = step_sleep_sec
        self.on_step = on_step
        self.gain_mode = gain_mode
        self.feedback_gain = (
            np.eye(7, 14, dtype=np.float64)
            if feedback_gain is None
            else np.asarray(feedback_gain, dtype=np.float64)
        )
        self.reference_q = reference_q
        self.reference_v = reference_v

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
            K=self.feedback_gain,
            diagnostics=FakePlannerDiagnostics(gain_mode=self.gain_mode),
            reference_q=self.reference_q,
            reference_v=self.reference_v,
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
    reference_q = np.asarray([0.1 * index for index in range(7)], dtype=np.float64)
    reference_v = np.asarray([-0.2 * index for index in range(7)], dtype=np.float64)
    planner = FakePlanner(reference_q=reference_q, reference_v=reference_v)
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    bridge._control_enabled = True
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
        assert snapshot.last_reference_q == pytest.approx(reference_q.tolist())
        assert snapshot.last_reference_v == pytest.approx(reference_v.tolist())
        assert snapshot.planner_mode == "exact_feedback"
        assert snapshot.last_planner_command_time_ms == pytest.approx(9.5)
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_bridge_destroy_closes_planner() -> None:
    planner = FakePlanner()
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)

    bridge.destroy_node()

    assert planner.close_calls == 1


def test_bridge_diagnostics_preserve_phase_machine_json() -> None:
    """The nested gate report survives planner -> bridge -> JSON unchanged."""
    gate: dict[str, object] = {
        "phase": "PREGRASP",
        "next_phase": "DESCEND",
        "gate_type": "task_space",
        "plan_time_sec": 2.4,
        "phase_elapsed_sec": 2.4,
        "boundary_time_sec": 2.4,
        "time_to_boundary_sec": 0.0,
        "at_boundary": True,
        "transition_status": "blocked_ee_linear_speed",
        "transition_blocked": True,
        "q_error_signed_by_joint_rad": [
            0.0,
            0.0,
            0.0,
            0.061,
            0.0,
            0.0,
            0.0,
        ],
        "q_error_max_rad": 0.061,
        "q_error_joint_index": 3,
        "velocity_by_joint_rad_s": [0.0] * 7,
        "velocity_abs_max_rad_s": 0.0,
        "velocity_joint_index": 0,
        "precision_hold": False,
        "gripper_command": "open",
        "clock_paused": False,
        "clock_pause_reason": None,
        "ee_source": "planning_model_fk",
        "ee_position_m": [0.5, 0.0, 0.11],
        "ee_goal_position_m": [0.5, 0.0, 0.1],
        "ee_position_error_signed_m": [0.0, 0.0, 0.01],
        "ee_position_error_norm_m": 0.01,
        "ee_position_tolerance_m": 0.03,
        "ee_position_ok": True,
        "ee_orientation_error_rad": 0.002,
        "ee_orientation_tolerance_rad": 0.1,
        "ee_orientation_ok": True,
        "ee_linear_velocity_m_s": [0.061, 0.0, 0.0],
        "ee_linear_speed_m_s": 0.061,
        "ee_linear_speed_tolerance_m_s": 0.06,
        "ee_linear_speed_ok": False,
        "ee_angular_velocity_rad_s": [0.0, 0.0, 0.0],
        "ee_angular_speed_rad_s": 0.0,
        "ee_angular_speed_tolerance_rad_s": 0.2,
        "ee_angular_speed_ok": True,
        "transition_blockers": ["ee_linear_speed"],
        "consecutive_eligible_cycles": 0,
        "consecutive_required_cycles": 5,
        "consecutive_ok": False,
    }
    planner = FakePlanner()
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    output = FakePlannerOutput(
        tau_ff=np.zeros(7, dtype=np.float64),
        K=np.zeros((7, 14), dtype=np.float64),
        diagnostics=FakePlannerDiagnostics(phase_machine=gate),
    )

    try:
        bridge._record_planner_diagnostics(output)
        # Recording makes an owned copy; later planner mutations cannot alter
        # the bridge's current diagnostic sample.
        gate["transition_status"] = "ready"
        snapshot = bridge.diagnostics_snapshot()
        assert snapshot.phase_machine is not None
        assert (
            snapshot.phase_machine["transition_status"]
            == "blocked_ee_linear_speed"
        )
        assert snapshot.phase_machine["q_error_joint_index"] == 3
        assert snapshot.phase_machine[
            "ee_position_error_norm_m"
        ] == pytest.approx(0.01)

        payload = json.loads(snapshot.to_json())
        assert payload["phase_machine"] == snapshot.phase_machine
        assert payload["phase_machine"]["transition_blocked"] is True
    finally:
        bridge.destroy_node()


def test_fake_ros_loop_publishes_controls_near_target_rate_and_diagnostics() -> None:
    planner = FakePlanner()
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    bridge._control_enabled = True
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
    bridge._control_enabled = True
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


def test_fake_ros_loop_counts_deadline_misses() -> None:
    planner = FakePlanner(step_sleep_sec=0.03)
    bridge = SbMpcLfcBridgeNode(
        planner=planner,
        publish_period_sec=0.02,
        planner_deadline_sec=0.02,
    )
    bridge._control_enabled = True
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
    bridge._control_enabled = True
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
    bridge._control_enabled = True
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


def test_fake_ros_loop_adds_feedforward_velocity_damping_with_correct_sign() -> None:
    planner = FakePlanner(
        gain_mode="feedforward",
        feedback_gain=np.zeros((7, 14), dtype=np.float64),
    )
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    bridge._control_enabled = True
    bridge.set_parameters(
        [
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


def test_feedforward_diagonal_helper_ignores_exact_feedback_mode() -> None:
    planner = FakePlanner(
        gain_mode="exact_feedback",
        feedback_gain=np.zeros((7, 14), dtype=np.float64),
    )
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    bridge._control_enabled = True
    bridge.set_parameters(
        [
            Parameter("feedforward_position_gain", Parameter.Type.DOUBLE, 3.0),
            Parameter("feedforward_velocity_damping_gain", Parameter.Type.DOUBLE, 2.0),
        ]
    )
    sensor_publisher = FakeSensorPublisher(enabled=True)
    collector = ControlCollector()
    executor = build_executor(bridge, sensor_publisher, collector)

    try:
        spin_for(executor, 0.20)

        assert collector.controls
        gain = float64_multi_array_to_numpy(collector.controls[-1].feedback_gain)
        np.testing.assert_allclose(gain, np.zeros((7, 14), dtype=np.float64))
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_fake_ros_loop_stays_zero_until_nonzero_control_is_enabled() -> None:
    planner = FakePlanner()
    planner.feedback_gain[0, 1] = 0.25
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
        bridge._control_enabled = True
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
        published_gain = float64_multi_array_to_numpy(
            collector.controls[-1].feedback_gain
        )
        np.testing.assert_allclose(
            snapshot.last_control_position_gain_diagonal,
            np.diag(published_gain[:, :7]),
        )
        np.testing.assert_allclose(
            snapshot.last_control_velocity_gain_diagonal,
            np.diag(published_gain[:, 7:]),
        )
        off_diagonal = published_gain.copy()
        rows = np.arange(7)
        off_diagonal[rows, rows] = 0.0
        off_diagonal[rows, 7 + rows] = 0.0
        assert snapshot.last_control_gain_max_abs_off_diagonal == pytest.approx(
            np.max(np.abs(off_diagonal))
        )
    finally:
        teardown_executor(executor, bridge, sensor_publisher, collector)


def test_fake_ros_loop_publishes_hold_when_disarmed_after_control() -> None:
    planner = FakePlanner()
    bridge = SbMpcLfcBridgeNode(planner=planner, publish_period_sec=0.02)
    bridge._control_enabled = True
    sensor_publisher = FakeSensorPublisher(enabled=True)
    collector = ControlCollector()
    executor = build_executor(bridge, sensor_publisher, collector)

    try:
        spin_for(executor, 0.18)
        assert collector.controls

        control_count_before_disarm = len(collector.controls)
        bridge._control_enabled = False
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
    bridge._control_enabled = True
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
    bridge._control_enabled = True
    bridge.set_parameters(
        [
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
    bridge._control_enabled = True
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


def test_set_nonzero_control_service_enforces_warmup_precondition() -> None:
    bridge = SbMpcLfcBridgeNode(planner=FakePlanner(), publish_period_sec=0.02)
    client_node = Node("set_nonzero_control_test_client")
    client = client_node.create_client(
        SetBool, "/sbmpc_lfc_bridge_node/set_nonzero_control"
    )
    executor = build_executor(bridge, client_node)

    def call(data: bool) -> SetBool.Response:
        assert client.wait_for_service(timeout_sec=5.0)
        future = client.call_async(SetBool.Request(data=data))
        deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
        response = future.result()
        assert response is not None
        return response

    try:
        # Arming is refused while warmup is incomplete (LFC must stay in PD-hold).
        bridge._warmup_complete = False
        refused = call(True)
        assert refused.success is False
        assert "warmup" in refused.message
        assert bridge._nonzero_control_enabled() is False

        # Once warmup completes, arming is accepted.
        bridge._warmup_complete = True
        armed = call(True)
        assert armed.success is True
        assert bridge._nonzero_control_enabled() is True

        # Disarming is always allowed, regardless of warmup state.
        disarmed = call(False)
        assert disarmed.success is True
        assert bridge._nonzero_control_enabled() is False
    finally:
        teardown_executor(executor, bridge, client_node)
