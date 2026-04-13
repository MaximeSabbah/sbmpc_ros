from __future__ import annotations

from time import perf_counter

from std_msgs.msg import String
import rclpy
from linear_feedback_controller_msgs.msg import Control, Sensor
from rclpy.node import Node

from sbmpc_ros_bridge.diagnostics import BridgeDiagnostics
from sbmpc_ros_bridge.joint_mapping import JointMapper
from sbmpc_ros_bridge.lfc_msg_adapter import (
    planner_output_to_control,
    sensor_to_planner_input,
    zero_control_from_sensor,
)
from sbmpc_ros_bridge.planner_adapter import SbMpcPlannerAdapter
from sbmpc_ros_bridge.safety import (
    BridgeSafetyProfile,
    PlanningDeadlineMonitor,
    UnsafeControlError,
    make_default_safety_profile,
)


class SbMpcLfcBridgeNode(Node):
    """Timer-driven SB-MPC to LFC bridge node."""

    def __init__(
        self,
        *,
        planner: object | None = None,
        safety_profile: BridgeSafetyProfile | None = None,
        publish_period_sec: float = 0.02,
        planner_deadline_sec: float | None = None,
    ) -> None:
        super().__init__("sbmpc_lfc_bridge_node")
        self.declare_parameter("sensor_topic", "sensor")
        self.declare_parameter("control_topic", "control")
        self.declare_parameter("diagnostics_topic", "diagnostics")
        self.declare_parameter("allow_joint_reordering", False)
        self.declare_parameter("publish_rate_hz", 1.0 / publish_period_sec)
        self.declare_parameter(
            "planner_deadline_sec",
            0.0 if planner_deadline_sec is None else planner_deadline_sec,
        )
        self.declare_parameter("joint_names", list(JointMapper.panda().expected_names))

        publish_rate_hz = (
            self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        )
        if publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be strictly positive.")
        publish_period_sec = 1.0 / publish_rate_hz

        configured_planner_deadline_sec = (
            self.get_parameter("planner_deadline_sec")
            .get_parameter_value()
            .double_value
        )
        if configured_planner_deadline_sec > 0.0:
            planner_deadline_sec = configured_planner_deadline_sec
        elif planner_deadline_sec is None:
            planner_deadline_sec = publish_period_sec

        joint_names = tuple(
            self.get_parameter("joint_names").get_parameter_value().string_array_value
        )
        self._joint_mapper = JointMapper(expected_names=joint_names)
        self._last_planner_input = None
        self._planner = planner if planner is not None else SbMpcPlannerAdapter()
        self._safety_profile = (
            make_default_safety_profile() if safety_profile is None else safety_profile
        )
        self._deadline_monitor = PlanningDeadlineMonitor(
            max_planning_duration_sec=planner_deadline_sec,
            fail_closed=False,
        )
        self._state = "waiting_for_sensor"
        self._valid_sensor_count = 0
        self._rejected_sensor_count = 0
        self._published_control_count = 0
        self._nonzero_control_count = 0
        self._warmup_count = 0
        self._planner_step_count = 0
        self._last_planning_time_ms: float | None = None
        self._last_error = ""
        self._warmup_complete = False

        sensor_topic = (
            self.get_parameter("sensor_topic").get_parameter_value().string_value
        )
        control_topic = (
            self.get_parameter("control_topic").get_parameter_value().string_value
        )
        diagnostics_topic = (
            self.get_parameter("diagnostics_topic").get_parameter_value().string_value
        )
        self._control_publisher = self.create_publisher(Control, control_topic, 10)
        self._diagnostics_publisher = self.create_publisher(String, diagnostics_topic, 10)
        self._sensor_subscription = self.create_subscription(
            Sensor,
            sensor_topic,
            self._on_sensor,
            10,
        )
        self._timer = self.create_timer(publish_period_sec, self._on_timer)

        self.get_logger().info(
            "SB-MPC LFC bridge active: waiting for valid sensors before warmup "
            f"and {publish_rate_hz:.1f} Hz control publication."
        )

    def _on_sensor(self, message: Sensor) -> None:
        allow_reordering = (
            self.get_parameter("allow_joint_reordering")
            .get_parameter_value()
            .bool_value
        )
        try:
            self._last_planner_input = sensor_to_planner_input(
                message,
                joint_mapper=self._joint_mapper,
                allow_reordering=allow_reordering,
            )
            self._valid_sensor_count += 1
            if self._state == "waiting_for_sensor":
                self._state = "warming_up"
        except Exception as exc:
            self._rejected_sensor_count += 1
            self._last_error = str(exc)
            self.get_logger().error(f"Rejected sensor message: {exc}")
            self._publish_diagnostics()

    def diagnostics_snapshot(self) -> BridgeDiagnostics:
        return BridgeDiagnostics(
            state=self._state,
            valid_sensor_count=self._valid_sensor_count,
            rejected_sensor_count=self._rejected_sensor_count,
            published_control_count=self._published_control_count,
            nonzero_control_count=self._nonzero_control_count,
            warmup_count=self._warmup_count,
            planner_step_count=self._planner_step_count,
            deadline_miss_count=self._deadline_monitor.deadline_miss_count,
            last_planning_time_ms=self._last_planning_time_ms,
            last_error=self._last_error,
        )

    def _on_timer(self) -> None:
        if self._last_planner_input is None:
            self._publish_diagnostics()
            return

        if not self._warmup_complete:
            self._run_warmup()
            self._publish_control(zero_control_from_sensor(self._last_planner_input))
            self._state = "running"
            self._publish_diagnostics()
            return

        start = perf_counter()
        try:
            planner_output = self._planner.step(self._last_planner_input)
            self._planner_step_count += 1
            control = planner_output_to_control(
                planner_output,
                self._last_planner_input,
                safety_profile=self._safety_profile,
            )
            planning_duration_sec = perf_counter() - start
            self._record_planning_duration(planning_duration_sec)
            self._publish_control(control)
        except UnsafeControlError as exc:
            self._state = "error"
            self._last_error = str(exc)
            self.get_logger().error(f"Rejected planner output: {exc}")
        except Exception as exc:
            self._state = "error"
            self._last_error = str(exc)
            self.get_logger().error(f"Planner loop failed: {exc}")
        finally:
            self._publish_diagnostics()

    def _run_warmup(self) -> None:
        start = perf_counter()
        self._planner.warmup()
        self._warmup_count += 1
        self._warmup_complete = True
        planning_duration_sec = perf_counter() - start
        self._record_planning_duration(planning_duration_sec)

    def _record_planning_duration(self, planning_duration_sec: float) -> None:
        self._last_planning_time_ms = 1000.0 * planning_duration_sec
        try:
            self._deadline_monitor.observe(planning_duration_sec)
        except UnsafeControlError as exc:
            self._last_error = str(exc)
            self.get_logger().warn(str(exc))

    def _publish_control(self, control: Control) -> None:
        self._control_publisher.publish(control)
        self._published_control_count += 1
        if any(abs(value) > 1e-12 for value in control.feedforward.data):
            self._nonzero_control_count += 1

    def _publish_diagnostics(self) -> None:
        self._diagnostics_publisher.publish(
            String(data=self.diagnostics_snapshot().to_json())
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SbMpcLfcBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
