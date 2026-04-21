from __future__ import annotations

from copy import deepcopy
from threading import Lock
from time import perf_counter

import numpy as np
from std_msgs.msg import String
import rclpy
from linear_feedback_controller_msgs.msg import Control, Sensor
from rclpy._rclpy_pybind11 import RCLError
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from sbmpc_ros_bridge.diagnostics import BridgeDiagnostics
from sbmpc_ros_bridge.joint_mapping import JointMapper
from sbmpc_ros_bridge.lfc_msg_adapter import (
    planner_output_to_control,
    sensor_to_planner_input,
    zero_control_from_sensor,
)
from sbmpc_ros_bridge.planner_adapter import (
    PlannerInput,
    SbMpcPlannerAdapter,
    planner_config_overrides_from_values,
)
from sbmpc_ros_bridge.safety import (
    BridgeSafetyProfile,
    PlanningDeadlineMonitor,
    UnsafeControlError,
    make_default_safety_profile,
)


class _NoopPlannerAdapter:
    """Warmup-only placeholder used when the bridge is explicitly gated to zero output."""

    def warmup(self, **kwargs) -> None:
        del kwargs


LFC_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
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
        self.declare_parameter("enable_nonzero_control", False)
        self.declare_parameter("force_zero_control", False)
        self.declare_parameter("allow_joint_reordering", False)
        self.declare_parameter("retime_control_initial_state", True)
        self.declare_parameter("control_initial_state_prediction_sec", 0.0)
        self.declare_parameter("publish_rate_hz", 1.0 / publish_period_sec)
        self.declare_parameter(
            "planner_deadline_sec",
            0.0 if planner_deadline_sec is None else planner_deadline_sec,
        )
        self.declare_parameter("planner_warmup_iterations", 10)
        self.declare_parameter("joint_names", list(JointMapper.panda().expected_names))
        self.declare_parameter("planner_phase", "PREGRASP")
        self.declare_parameter("planner_gains", True)
        self.declare_parameter("planner_num_steps", 1)
        self.declare_parameter("planner_num_samples", 1024)
        self.declare_parameter("planner_horizon", 8)
        self.declare_parameter("planner_num_parallel_computations", 1024)
        self.declare_parameter("planner_num_control_points", 8)
        self.declare_parameter("planner_temperature", 0.05)
        self.declare_parameter("planner_dt", 0.02)
        self.declare_parameter("planner_lambda_mpc", 0.05)
        self.declare_parameter("planner_noise_scale", 0.05)
        self.declare_parameter("planner_std_dev_scale", 0.05)
        self.declare_parameter("planner_smoothing", "Spline")
        self.declare_parameter("planner_gain_method", "finite_difference")
        self.declare_parameter("planner_gain_fd_epsilon", 1e-2)
        self.declare_parameter("planner_gain_fd_scheme", "forward")
        self.declare_parameter("planner_gain_fd_num_samples", 256)

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
        self._force_zero_control = self._force_zero_control_enabled()
        self._joint_mapper = JointMapper(expected_names=joint_names)
        self._sensor_lock = Lock()
        self._last_planner_input = None
        self._planner_warmup_iterations = max(
            1,
            int(
                self.get_parameter("planner_warmup_iterations")
                .get_parameter_value()
                .integer_value
            ),
        )
        planner_config = planner_config_overrides_from_values(
            phase=self.get_parameter("planner_phase").get_parameter_value().string_value,
            gains=self.get_parameter("planner_gains").get_parameter_value().bool_value,
            num_steps=(
                self.get_parameter("planner_num_steps").get_parameter_value().integer_value
            ),
            num_samples=(
                self.get_parameter("planner_num_samples").get_parameter_value().integer_value
            ),
            horizon=self.get_parameter("planner_horizon").get_parameter_value().integer_value,
            num_parallel_computations=(
                self.get_parameter("planner_num_parallel_computations")
                .get_parameter_value()
                .integer_value
            ),
            num_control_points=(
                self.get_parameter("planner_num_control_points")
                .get_parameter_value()
                .integer_value
            ),
            temperature=(
                self.get_parameter("planner_temperature").get_parameter_value().double_value
            ),
            dt=self.get_parameter("planner_dt").get_parameter_value().double_value,
            lambda_mpc=(
                self.get_parameter("planner_lambda_mpc").get_parameter_value().double_value
            ),
            noise_scale=(
                self.get_parameter("planner_noise_scale").get_parameter_value().double_value
            ),
            std_dev_scale=(
                self.get_parameter("planner_std_dev_scale").get_parameter_value().double_value
            ),
            smoothing=self.get_parameter("planner_smoothing").get_parameter_value().string_value,
            gain_method=(
                self.get_parameter("planner_gain_method").get_parameter_value().string_value
            ),
            gain_fd_epsilon=(
                self.get_parameter("planner_gain_fd_epsilon").get_parameter_value().double_value
            ),
            gain_fd_scheme=(
                self.get_parameter("planner_gain_fd_scheme").get_parameter_value().string_value
            ),
            gain_fd_num_samples=(
                self.get_parameter("planner_gain_fd_num_samples")
                .get_parameter_value()
                .integer_value
            ),
        )
        self._planner = (
            planner
            if planner is not None
            else (
                _NoopPlannerAdapter()
                if self._force_zero_control
                else SbMpcPlannerAdapter(config_overrides=planner_config)
            )
        )
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
        self._last_planner_output_time_ms: float | None = None
        self._last_bridge_loop_time_ms: float | None = None
        self._last_phase: str | None = None
        self._last_next_phase: str | None = None
        self._last_running_cost: float | None = None
        self._last_gain_norm: float | None = None
        self._last_torque_norm: float | None = None
        self._last_position_error: float | None = None
        self._last_orientation_error: float | None = None
        self._last_object_error: float | None = None
        self._last_goal_position: list[float] | None = None
        self._last_control_max_abs_feedforward: float | None = None
        self._last_control_gain_norm: float | None = None
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
        self._control_publisher = self.create_publisher(
            Control,
            control_topic,
            LFC_QOS,
        )
        self._diagnostics_publisher = self.create_publisher(String, diagnostics_topic, 10)
        self._sensor_callback_group = MutuallyExclusiveCallbackGroup()
        self._timer_callback_group = MutuallyExclusiveCallbackGroup()
        self._sensor_subscription = self.create_subscription(
            Sensor,
            sensor_topic,
            self._on_sensor,
            LFC_QOS,
            callback_group=self._sensor_callback_group,
        )
        self._timer = self.create_timer(
            publish_period_sec,
            self._on_timer,
            callback_group=self._timer_callback_group,
        )

        self.get_logger().info(
            "SB-MPC LFC bridge active: waiting for valid sensors before warmup "
            f"and {publish_rate_hz:.1f} Hz control publication."
        )
        if planner is None:
            if self._force_zero_control:
                self.get_logger().info(
                    "force_zero_control is enabled: the bridge will publish zero "
                    "feedforward and zero gains after warmup."
                )
            else:
                self.get_logger().info(
                    f"Planner configuration from ROS parameters: {planner_config.active_items()}"
                )
            if not self._nonzero_control_enabled():
                self.get_logger().info(
                    "enable_nonzero_control is false: the bridge will stay silent "
                    "after warmup so LFC remains in PD mode until you arm it."
                )

    def _force_zero_control_enabled(self) -> bool:
        self._force_zero_control = (
            self.get_parameter("force_zero_control").get_parameter_value().bool_value
        )
        return self._force_zero_control

    def _nonzero_control_enabled(self) -> bool:
        return (
            self.get_parameter("enable_nonzero_control")
            .get_parameter_value()
            .bool_value
        )

    def _retime_control_initial_state_enabled(self) -> bool:
        return (
            self.get_parameter("retime_control_initial_state")
            .get_parameter_value()
            .bool_value
        )

    def _snapshot_planner_input(self):
        with self._sensor_lock:
            return self._last_planner_input

    def _on_sensor(self, message: Sensor) -> None:
        allow_reordering = (
            self.get_parameter("allow_joint_reordering")
            .get_parameter_value()
            .bool_value
        )
        try:
            planner_input = sensor_to_planner_input(
                message,
                joint_mapper=self._joint_mapper,
                allow_reordering=allow_reordering,
            )
            with self._sensor_lock:
                self._last_planner_input = planner_input
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
            control_enabled=self._nonzero_control_enabled(),
            force_zero_control=self._force_zero_control_enabled(),
            valid_sensor_count=self._valid_sensor_count,
            rejected_sensor_count=self._rejected_sensor_count,
            published_control_count=self._published_control_count,
            nonzero_control_count=self._nonzero_control_count,
            warmup_count=self._warmup_count,
            planner_step_count=self._planner_step_count,
            deadline_miss_count=self._deadline_monitor.deadline_miss_count,
            last_planning_time_ms=self._last_planning_time_ms,
            last_planner_output_time_ms=self._last_planner_output_time_ms,
            last_bridge_loop_time_ms=self._last_bridge_loop_time_ms,
            last_phase=self._last_phase,
            last_next_phase=self._last_next_phase,
            last_running_cost=self._last_running_cost,
            last_gain_norm=self._last_gain_norm,
            last_torque_norm=self._last_torque_norm,
            last_position_error=self._last_position_error,
            last_orientation_error=self._last_orientation_error,
            last_object_error=self._last_object_error,
            last_goal_position=self._last_goal_position,
            last_control_max_abs_feedforward=self._last_control_max_abs_feedforward,
            last_control_gain_norm=self._last_control_gain_norm,
            last_error=self._last_error,
        )

    def _on_timer(self) -> None:
        planner_input = self._snapshot_planner_input()
        if planner_input is None:
            self._publish_diagnostics()
            return

        if not self._warmup_complete:
            self._run_warmup(planner_input)
            # Do NOT publish any Control here. Publishing a zero Control message
            # (tau_ff=0, K=0) would cause LFC to immediately start its PD→LF
            # transition (it triggers on the first non-NaN feedforward). After
            # 100 ms LFC would be in pure LF mode sending zero torques → the
            # robot falls. Keep LFC in PD mode until the bridge is explicitly armed.
            self._state = (
                "gated_zero_control"
                if self._force_zero_control_enabled()
                else ("running" if self._nonzero_control_enabled() else "armed_idle")
            )
            self._publish_diagnostics()
            return

        if self._force_zero_control_enabled():
            # Explicit test mode: caller deliberately wants LFC in LF mode with
            # zero torques (e.g. to verify the robot holds under PD-only).
            self._state = "gated_zero_control"
            self._publish_control(zero_control_from_sensor(planner_input))
            self._publish_diagnostics()
            return

        if not self._nonzero_control_enabled():
            # Not armed yet — do NOT publish. Keeping silent here leaves LFC in
            # PD mode (stiff hold) until the operator explicitly arms the bridge.
            self._state = "armed_idle"
            self._publish_diagnostics()
            return

        self._state = "running"
        start = perf_counter()
        try:
            planner_output = self._planner.step(planner_input)
            self._planner_step_count += 1
            self._record_planner_diagnostics(planner_output)
            control_initial_state = planner_input
            if self._retime_control_initial_state_enabled():
                control_initial_state = self._snapshot_planner_input() or planner_input
            control_initial_state = self._predict_control_initial_state(
                control_initial_state,
                planner_output,
            )
            control = planner_output_to_control(
                planner_output,
                control_initial_state,
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

    def _run_warmup(self, planner_input: PlannerInput) -> None:
        start = perf_counter()
        warmup_output = None
        for _ in range(self._planner_warmup_iterations):
            warmup_output = self._planner.warmup()
            if warmup_output is not None:
                self._record_planner_diagnostics(warmup_output)
                self._predict_control_initial_state(planner_input, warmup_output)
        self._warmup_count += 1
        self._warmup_complete = True
        planning_duration_sec = perf_counter() - start
        self._record_planning_duration(planning_duration_sec)

    def _record_planning_duration(self, planning_duration_sec: float) -> None:
        self._last_planning_time_ms = 1000.0 * planning_duration_sec
        self._last_bridge_loop_time_ms = self._last_planning_time_ms
        try:
            self._deadline_monitor.observe(planning_duration_sec)
        except UnsafeControlError as exc:
            self._last_error = str(exc)
            self.get_logger().warn(str(exc))

    def _control_initial_state_prediction_sec(self) -> float:
        value = (
            self.get_parameter("control_initial_state_prediction_sec")
            .get_parameter_value()
            .double_value
        )
        return max(0.0, float(value))

    def _predict_control_initial_state(
        self,
        planner_input: PlannerInput,
        planner_output: object,
    ) -> PlannerInput:
        prediction_sec = self._control_initial_state_prediction_sec()
        if prediction_sec <= 0.0:
            return planner_input

        predict_state = getattr(self._planner, "predict_state", None)
        if predict_state is None:
            return planner_input

        tau_ff = np.asarray(getattr(planner_output, "tau_ff"), dtype=np.float64)
        prediction = predict_state(planner_input, tau_ff, prediction_sec)
        if prediction is None:
            return planner_input

        q_pred, v_pred = prediction
        q = np.asarray(q_pred, dtype=np.float64).reshape(-1)
        v = np.asarray(v_pred, dtype=np.float64).reshape(-1)
        sensor = deepcopy(planner_input.sensor)
        sensor.joint_state.position = q.tolist()
        sensor.joint_state.velocity = v.tolist()
        return PlannerInput(sensor=sensor, q=q, v=v)

    def _record_planner_diagnostics(self, planner_output: object) -> None:
        self._last_phase = self._phase_name(getattr(planner_output, "phase", None))
        self._last_next_phase = self._phase_name(
            getattr(planner_output, "next_phase", None)
        )

        diagnostics = getattr(planner_output, "diagnostics", None)
        if diagnostics is None:
            return

        self._last_planner_output_time_ms = self._maybe_float(
            getattr(diagnostics, "planning_time_ms", None)
        )
        self._last_running_cost = self._maybe_float(
            getattr(diagnostics, "running_cost", None)
        )
        self._last_gain_norm = self._maybe_float(
            getattr(diagnostics, "gain_norm", None)
        )
        self._last_torque_norm = self._maybe_float(
            getattr(diagnostics, "torque_norm", None)
        )
        self._last_position_error = self._maybe_float(
            getattr(diagnostics, "position_error", None)
        )
        self._last_orientation_error = self._maybe_float(
            getattr(diagnostics, "orientation_error", None)
        )
        self._last_object_error = self._maybe_float(
            getattr(diagnostics, "object_error", None)
        )

        goal_position = getattr(diagnostics, "goal_position", None)
        if goal_position is None:
            self._last_goal_position = None
            return

        goal_array = np.asarray(goal_position, dtype=np.float64).reshape(-1)
        self._last_goal_position = goal_array.tolist()

    def _publish_control(self, control: Control) -> None:
        try:
            self._control_publisher.publish(control)
        except RCLError:
            if not rclpy.ok(context=self.context):
                return
            raise
        self._published_control_count += 1
        self._last_control_max_abs_feedforward = max(
            (abs(float(value)) for value in control.feedforward.data),
            default=0.0,
        )
        gain_data = np.asarray(control.feedback_gain.data, dtype=np.float64)
        self._last_control_gain_norm = (
            float(np.linalg.norm(gain_data)) if gain_data.size else 0.0
        )
        if (
            self._last_control_max_abs_feedforward > 1e-12
            or self._last_control_gain_norm > 1e-12
        ):
            self._nonzero_control_count += 1

    def _publish_diagnostics(self) -> None:
        try:
            self._diagnostics_publisher.publish(
                String(data=self.diagnostics_snapshot().to_json())
            )
        except RCLError:
            if not rclpy.ok(context=self.context):
                return
            raise

    @staticmethod
    def _phase_name(value: object | None) -> str | None:
        if value is None:
            return None
        name = getattr(value, "name", None)
        return str(name) if name is not None else str(value)

    @staticmethod
    def _maybe_float(value: object | None) -> float | None:
        if value is None:
            return None
        return float(value)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SbMpcLfcBridgeNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            executor.shutdown()
        except KeyboardInterrupt:
            pass
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok(context=node.context):
            rclpy.shutdown(context=node.context)


if __name__ == "__main__":
    main()
