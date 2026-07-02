from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
import signal
from threading import Event, Lock, Thread
from time import perf_counter

import numpy as np
from std_msgs.msg import String
from std_srvs.srv import SetBool
import rclpy
from linear_feedback_controller_msgs.msg import Control, Sensor
from rclpy._rclpy_pybind11 import RCLError
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from visualization_msgs.msg import MarkerArray

from sbmpc_ros_bridge.diagnostics import BridgeDiagnostics
from sbmpc_ros_bridge.joint_mapping import JointMapper
from sbmpc_ros_bridge.lfc_msg_adapter import (
    float64_multi_array_to_numpy,
    hold_control_from_sensor,
    numpy_to_float64_multi_array,
    planner_output_to_control,
    sensor_to_planner_input,
    zero_control_from_sensor,
)
from sbmpc_ros_bridge.planner_adapter import (
    PlannerInput,
    SbMpcPlannerAdapter,
    planner_config_overrides_from_values,
)
from sbmpc_ros_bridge.rollout_markers import (
    goal_position_from_planner_output,
    make_trajectory_marker_array,
)
from sbmpc_ros_bridge.safety import (
    PlanningDeadlineMonitor,
    UnsafeControlError,
    validate_planner_output,
)


CONTROL_QOS_DEPTH = 10
SENSOR_QOS_DEPTH = 1
CONTROLLER_MANAGER_CPU_COUNT = 2


@dataclass(frozen=True, slots=True)
class _LatestPlannerOutput:
    planner_input: PlannerInput
    planner_output: object


class _NoopPlannerAdapter:
    """Warmup-only placeholder used when the bridge is explicitly gated to zero output."""

    def warmup(self, **kwargs) -> None:
        del kwargs

    def reset_runtime_state_after_warmup(self) -> None:
        return None


def best_effort_qos(*, depth: int) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=max(1, int(depth)),
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )


def _prefer_bridge_cpu_affinity() -> tuple[int, ...] | None:
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    sched_setaffinity = getattr(os, "sched_setaffinity", None)
    if not callable(sched_getaffinity) or not callable(sched_setaffinity):
        return None

    current = set(int(cpu) for cpu in sched_getaffinity(0))
    if len(current) <= CONTROLLER_MANAGER_CPU_COUNT:
        return None

    preferred = {cpu for cpu in current if cpu >= CONTROLLER_MANAGER_CPU_COUNT}
    if not preferred or preferred == current:
        return None

    try:
        sched_setaffinity(0, preferred)
    except OSError:
        return None
    return tuple(sorted(preferred))


class SbMpcLfcBridgeNode(Node):
    """Timer-driven SB-MPC to LFC bridge node."""

    def __init__(
        self,
        *,
        planner: object | None = None,
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
        self.declare_parameter("control_initial_state_prediction_sec", 0.0)
        self.declare_parameter("publish_rate_hz", 1.0 / publish_period_sec)
        self.declare_parameter(
            "planner_deadline_sec",
            0.0 if planner_deadline_sec is None else planner_deadline_sec,
        )
        self.declare_parameter("feedforward_position_gain", 0.0)
        self.declare_parameter("feedforward_velocity_damping_gain", 0.0)
        self.declare_parameter("emergency_hold_position_gain", 1.0)
        self.declare_parameter("emergency_hold_velocity_gain", 2.0)
        self.declare_parameter("hold_on_disarm_after_control", True)
        self.declare_parameter("planner_warmup_iterations", 3)
        self.declare_parameter("planner_warmup_on_start", True)
        self.declare_parameter("joint_names", list(JointMapper.panda().expected_names))
        self.declare_parameter("planner_mode", "")
        self.declare_parameter("planner_num_steps", 1)
        # The MPPI/solver knobs below default to "unset" (0 / empty): the
        # sbmpc OCP yaml (planner_ocp) is the single source of truth, and only
        # explicitly configured ROS parameters override it.
        self.declare_parameter("planner_num_samples", 0)
        self.declare_parameter("planner_horizon", 0)
        self.declare_parameter("planner_num_parallel_computations", 0)
        self.declare_parameter("planner_num_control_points", 0)
        self.declare_parameter("planner_temperature", 0.0)
        self.declare_parameter("planner_dt", 0.0)
        self.declare_parameter("planner_lambda_mpc", 0.0)
        self.declare_parameter("planner_noise_scale", 0.0)
        self.declare_parameter("planner_std_dev_scale", 0.0)
        self.declare_parameter("planner_smoothing", "")
        self.declare_parameter("planner_num_gain_samples", 0)
        self.declare_parameter("planner_compute_task_diagnostics", False)
        self.declare_parameter("planner_ocp", "pregrasp")
        self.declare_parameter("publish_rollout_markers", False)
        self.declare_parameter("rollout_marker_topic", "/sbmpc/trajectory_markers")
        self.declare_parameter("rollout_marker_frame_id", "base")
        self.declare_parameter("rollout_marker_max_samples", 16)
        self.declare_parameter("rollout_marker_publish_rate_hz", 5.0)
        self.declare_parameter("rollout_marker_line_width", 0.01)
        self.declare_parameter("rollout_marker_sample_line_width", 0.004)
        self.declare_parameter("rollout_marker_point_diameter", 0.025)
        self.declare_parameter("rollout_marker_goal_diameter", 0.045)

        publish_rate_hz = (
            self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        )
        if publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be strictly positive.")
        publish_period_sec = 1.0 / publish_rate_hz
        self._publish_period_sec = publish_period_sec

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
        # Arming is a runtime command via the ~/set_nonzero_control service (D15).
        # The `enable_nonzero_control` parameter only seeds the initial value.
        self._control_lock = Lock()
        self._control_enabled = (
            self.get_parameter("enable_nonzero_control").get_parameter_value().bool_value
        )
        self._joint_names = joint_names
        self._closing = Event()
        self._planner_lock = Lock()
        self._warmup_lock = Lock()
        self._warmup_started = False
        self._warmup_thread: Thread | None = None
        self._joint_mapper = JointMapper(expected_names=joint_names)
        self._sensor_lock = Lock()
        self._last_sensor: Sensor | None = None
        self._latest_planner_output_lock = Lock()
        self._latest_planner_output: _LatestPlannerOutput | None = None
        self._fatal_error_lock = Lock()
        self._fatal_error: str | None = None
        self._planner_request = Event()
        self._planner_thread: Thread | None = None
        self._planner_warmup_iterations = max(
            1,
            int(
                self.get_parameter("planner_warmup_iterations")
                .get_parameter_value()
                .integer_value
            ),
        )
        # `phase` is intentionally not a ROS param: the controller owns its phase
        # (the pregrasp controller ignores it; the pick-and-place state machine
        # starts at PREGRASP and advances at runtime from the planner output's
        # next_phase). The adapter still accepts a phase override for that path.
        planner_config = planner_config_overrides_from_values(
            mode=self.get_parameter("planner_mode").get_parameter_value().string_value,
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
            num_gain_samples=(
                self.get_parameter("planner_num_gain_samples")
                .get_parameter_value()
                .integer_value
            ),
            compute_task_diagnostics=(
                self.get_parameter("planner_compute_task_diagnostics")
                .get_parameter_value()
                .bool_value
            ),
            ocp=self.get_parameter("planner_ocp").get_parameter_value().string_value,
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
        if self._publish_rollout_markers_enabled():
            enable_rollout_capture = getattr(
                self._planner, "set_rollout_capture_enabled", None
            )
            if callable(enable_rollout_capture):
                enable_rollout_capture(True)
        self._deadline_monitor = PlanningDeadlineMonitor(
            max_planning_duration_sec=planner_deadline_sec,
            fail_closed=False,
        )
        self._state = "waiting_for_sensor"
        self._received_sensor_count = 0
        self._valid_sensor_count = 0
        self._rejected_sensor_count = 0
        self._published_control_count = 0
        self._nonzero_control_count = 0
        self._warmup_count = 0
        self._planner_step_count = 0
        self._accepted_planner_output_count = 0
        self._rejected_planner_output_count = 0
        self._last_planning_time_ms: float | None = None
        self._last_planner_step_wall_time_ms: float | None = None
        self._last_planner_prepare_time_ms: float | None = None
        self._last_planner_command_time_ms: float | None = None
        self._last_control_prepare_time_ms: float | None = None
        self._last_control_publish_time_ms: float | None = None
        self._last_phase: str | None = None
        self._last_next_phase: str | None = None
        self._last_running_cost: float | None = None
        self._last_gain_norm: float | None = None
        self._last_torque_norm: float | None = None
        self._last_position_error: float | None = None
        self._last_orientation_error: float | None = None
        self._last_object_error: float | None = None
        self._last_goal_position: list[float] | None = None
        self._last_reference_q: list[float] | None = None
        self._last_reference_v: list[float] | None = None
        self._last_rollout_marker_publish_wall_sec: float | None = None
        self._last_control_max_abs_feedforward: float | None = None
        self._last_control_gain_norm: float | None = None
        self._planner_mode: str | None = planner_config.mode
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
            best_effort_qos(depth=CONTROL_QOS_DEPTH),
        )
        self._diagnostics_publisher = self.create_publisher(String, diagnostics_topic, 10)
        self._rollout_marker_publisher = (
            self.create_publisher(
                MarkerArray,
                self.get_parameter("rollout_marker_topic")
                .get_parameter_value()
                .string_value,
                10,
            )
            if self._publish_rollout_markers_enabled()
            else None
        )
        self._sensor_callback_group = MutuallyExclusiveCallbackGroup()
        self._sensor_subscription = self.create_subscription(
            Sensor,
            sensor_topic,
            self._on_sensor,
            best_effort_qos(depth=SENSOR_QOS_DEPTH),
            callback_group=self._sensor_callback_group,
        )
        self._set_nonzero_control_service = self.create_service(
            SetBool,
            "~/set_nonzero_control",
            self._on_set_nonzero_control,
        )
        self._timer = self.create_timer(publish_period_sec, self._on_timer)
        self._planner_thread = Thread(
            target=self._run_planner_worker,
            name="sbmpc_planner_worker",
            daemon=True,
        )
        self._planner_thread.start()

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
                mpc_dt = getattr(self._planner, "mpc_dt", None)
                if mpc_dt is not None and abs(mpc_dt - publish_period_sec) > 1e-9:
                    self.get_logger().warn(
                        f"Planner MPC dt ({mpc_dt:.4f} s, from the OCP yaml) does "
                        f"not match the bridge publish period "
                        f"({publish_period_sec:.4f} s). Align publish_rate_hz "
                        "with the OCP mpc.dt to keep both repos coherent."
                    )
                jax_cache_dir = getattr(self._planner, "jax_cache_dir", None)
                if jax_cache_dir:
                    self.get_logger().info(
                        f"JAX compilation cache enabled: {jax_cache_dir}"
                    )
            if not self._nonzero_control_enabled():
                self.get_logger().info(
                    "enable_nonzero_control is false: the bridge will stay silent "
                    "after warmup so LFC remains in PD mode until you arm it."
                )
            if (
                not self._force_zero_control
                and self._planner_warmup_on_start_enabled()
            ):
                self._start_warmup_thread()

    def _force_zero_control_enabled(self) -> bool:
        self._force_zero_control = (
            self.get_parameter("force_zero_control").get_parameter_value().bool_value
        )
        return self._force_zero_control

    def _nonzero_control_enabled(self) -> bool:
        with self._control_lock:
            return self._control_enabled

    def _on_set_nonzero_control(
        self,
        request: SetBool.Request,
        response: SetBool.Response,
    ) -> SetBool.Response:
        if request.data and not self._warmup_complete:
            response.success = False
            response.message = (
                "cannot arm: planner warmup is not complete; the bridge stays "
                "in PD-hold until warmup finishes."
            )
            return response

        with self._control_lock:
            self._control_enabled = bool(request.data)
        response.success = True
        response.message = (
            "armed: nonzero control enabled."
            if request.data
            else "disarmed: nonzero control disabled."
        )
        self.get_logger().info(response.message)
        return response

    def _planner_warmup_on_start_enabled(self) -> bool:
        return (
            self.get_parameter("planner_warmup_on_start")
            .get_parameter_value()
            .bool_value
        )

    def _publish_rollout_markers_enabled(self) -> bool:
        return (
            self.get_parameter("publish_rollout_markers")
            .get_parameter_value()
            .bool_value
        )

    def _snapshot_planner_input(self):
        with self._sensor_lock:
            sensor = (
                deepcopy(self._last_sensor)
                if self._last_sensor is not None
                else None
            )
        if sensor is None:
            return None

        allow_reordering = (
            self.get_parameter("allow_joint_reordering")
            .get_parameter_value()
            .bool_value
        )
        try:
            planner_input = sensor_to_planner_input(
                sensor,
                joint_mapper=self._joint_mapper,
                allow_reordering=allow_reordering,
            )
            self._valid_sensor_count += 1
            if self._state == "waiting_for_sensor":
                self._state = "warming_up" if not self._warmup_complete else "armed_idle"
            return planner_input
        except Exception as exc:
            self._rejected_sensor_count += 1
            self._last_error = str(exc)
            self.get_logger().error(f"Rejected sensor message: {exc}")
            return None

    def _on_sensor(self, message: Sensor) -> None:
        self._received_sensor_count += 1
        with self._sensor_lock:
            self._last_sensor = message

    def diagnostics_snapshot(self) -> BridgeDiagnostics:
        return BridgeDiagnostics(
            state=self._state,
            control_enabled=self._nonzero_control_enabled(),
            force_zero_control=self._force_zero_control_enabled(),
            received_sensor_count=self._received_sensor_count,
            valid_sensor_count=self._valid_sensor_count,
            rejected_sensor_count=self._rejected_sensor_count,
            published_control_count=self._published_control_count,
            nonzero_control_count=self._nonzero_control_count,
            warmup_count=self._warmup_count,
            planner_step_count=self._planner_step_count,
            accepted_planner_output_count=self._accepted_planner_output_count,
            rejected_planner_output_count=self._rejected_planner_output_count,
            deadline_miss_count=self._deadline_monitor.deadline_miss_count,
            last_planning_time_ms=self._last_planning_time_ms,
            last_planner_step_wall_time_ms=self._last_planner_step_wall_time_ms,
            last_planner_prepare_time_ms=self._last_planner_prepare_time_ms,
            last_planner_command_time_ms=self._last_planner_command_time_ms,
            last_control_prepare_time_ms=self._last_control_prepare_time_ms,
            last_control_publish_time_ms=self._last_control_publish_time_ms,
            last_phase=self._last_phase,
            last_next_phase=self._last_next_phase,
            last_running_cost=self._last_running_cost,
            last_gain_norm=self._last_gain_norm,
            last_torque_norm=self._last_torque_norm,
            last_position_error=self._last_position_error,
            last_orientation_error=self._last_orientation_error,
            last_object_error=self._last_object_error,
            last_goal_position=self._last_goal_position,
            last_reference_q=self._last_reference_q,
            last_reference_v=self._last_reference_v,
            last_control_max_abs_feedforward=self._last_control_max_abs_feedforward,
            last_control_gain_norm=self._last_control_gain_norm,
            last_error=self._last_error,
            planner_mode=self._planner_mode,
        )

    def _on_timer(self) -> None:
        self._publish_hold_before_fatal_if_needed()
        self._raise_if_fatal()

        planner_input = self._snapshot_planner_input()
        if planner_input is None:
            if self._warmup_started and not self._warmup_complete:
                self._state = "warming_up"
            self._publish_diagnostics()
            return

        if not self._warmup_complete:
            self._start_warmup_thread()
            # Do NOT publish any Control here. Publishing a zero Control message
            # (tau_ff=0, K=0) would cause LFC to immediately start its PD→LF
            # transition (it triggers on the first non-NaN feedforward). After
            # 100 ms LFC would be in pure LF mode sending zero torques → the
            # robot falls. Keep LFC in PD mode until the bridge is explicitly armed.
            self._state = "warming_up"
            self._publish_diagnostics()
            return

        if self._force_zero_control_enabled():
            # Explicit test mode: caller deliberately wants LFC in LF mode with
            # zero torques (e.g. to verify the robot holds under PD-only).
            self._state = "gated_zero_control"
            self._clear_latest_planner_output()
            self._publish_control(zero_control_from_sensor(planner_input))
            self._publish_diagnostics()
            return

        if not self._nonzero_control_enabled():
            # Before the first Control message, stay silent so LFC remains in
            # PD mode. After LF mode has been entered once, publish a hold
            # command when disarmed instead of leaving LFC with the previous
            # moving command.
            if self._published_control_count > 0 and self._hold_on_disarm_enabled():
                self._state = "disarmed_hold"
                self._publish_hold_control(
                    planner_input,
                    position_gain=self._nonnegative_double_parameter(
                        "emergency_hold_position_gain"
                    ),
                    velocity_gain=self._nonnegative_double_parameter(
                        "emergency_hold_velocity_gain"
                    ),
                    reason="disarm",
                )
            else:
                self._state = "armed_idle"
            self._clear_latest_planner_output()
            self._publish_diagnostics()
            return

        try:
            published = self._publish_latest_control(planner_input)
            self._state = "running" if published else "planning"
            self._planner_request.set()
        except UnsafeControlError as exc:
            self._rejected_planner_output_count += 1
            self._publish_emergency_hold(
                planner_input,
                reason="unsafe planner output",
            )
            self._latch_fatal_error(f"Rejected planner output: {exc}")
        except Exception as exc:
            self._publish_emergency_hold(
                planner_input,
                reason="planner/control loop exception",
            )
            self._latch_fatal_error(f"Planner loop failed: {exc}")
        finally:
            self._publish_diagnostics()
        self._raise_if_fatal()

    def _publish_latest_control(self, planner_input: PlannerInput) -> bool:
        with self._latest_planner_output_lock:
            latest = self._latest_planner_output
        if latest is None:
            return False

        prepare_start = perf_counter()
        planner_output = latest.planner_output
        # x_des sent to the LFC is the state the planner actually planned from
        # (its linearization point), so the LFC term K*(x_des - x_measured)
        # corrects deviation from the plan instead of collapsing to ~0.
        control = planner_output_to_control(
            planner_output,
            latest.planner_input,
        )
        control = self._apply_feedforward_feedback_regularization(
            control,
            planner_output=planner_output,
        )
        control.header = deepcopy(planner_input.sensor.header)
        self._last_control_prepare_time_ms = 1000.0 * (
            perf_counter() - prepare_start
        )
        publish_start = perf_counter()
        self._publish_control(control)
        self._last_control_publish_time_ms = 1000.0 * (
            perf_counter() - publish_start
        )
        return True

    def _clear_latest_planner_output(self) -> None:
        with self._latest_planner_output_lock:
            self._latest_planner_output = None

    def _store_latest_planner_output(
        self, planner_input: PlannerInput, planner_output: object
    ) -> None:
        entry = _LatestPlannerOutput(
            planner_input=planner_input,
            planner_output=planner_output,
        )
        with self._latest_planner_output_lock:
            self._latest_planner_output = entry

    def _run_planner_worker(self) -> None:
        while not self._closing.is_set():
            if not self._planner_request.wait(timeout=0.1):
                continue
            self._planner_request.clear()
            if self._closing.is_set():
                return
            if (
                not self._warmup_complete
                or self._force_zero_control_enabled()
                or not self._nonzero_control_enabled()
            ):
                continue

            planner_input = self._snapshot_planner_input()
            if planner_input is None:
                continue
            planner_input = self._predict_delayed_planner_input(planner_input)

            try:
                step_start = perf_counter()
                with self._planner_lock:
                    planner_output = self._planner.step(planner_input)
                step_wall_sec = perf_counter() - step_start
                self._last_planner_step_wall_time_ms = 1000.0 * step_wall_sec
                self._planner_step_count += 1
                self._record_planner_diagnostics(planner_output)
                validate_planner_output(
                    np.asarray(getattr(planner_output, "tau_ff")),
                    np.asarray(getattr(planner_output, "K")),
                )
                self._store_latest_planner_output(planner_input, planner_output)
                self._accepted_planner_output_count += 1
                self._last_error = ""
                self._state = "running"
                self._observe_planning_deadline(step_wall_sec)
                self._log_solve(step_wall_sec, planner_input, planner_output)
                self._publish_rollout_markers(planner_input, planner_output)
            except UnsafeControlError as exc:
                self._rejected_planner_output_count += 1
                self._latch_fatal_error(f"Rejected planner output: {exc}")
            except Exception as exc:
                self._latch_fatal_error(f"Planner loop failed: {exc}")

    def _predict_delayed_planner_input(
        self,
        planner_input: PlannerInput,
    ) -> PlannerInput:
        if self._control_initial_state_prediction_sec() <= 0.0:
            return planner_input

        with self._latest_planner_output_lock:
            latest = self._latest_planner_output
        if latest is None:
            return planner_input

        return self._predict_control_initial_state(
            planner_input,
            latest.planner_output,
        )

    def _start_warmup_thread(self) -> None:
        with self._warmup_lock:
            if self._warmup_complete or self._warmup_started:
                return
            self._warmup_started = True
            self._state = "warming_up"
            self.get_logger().info(
                "Starting planner warmup/JIT compilation before arming."
            )
            self._warmup_thread = Thread(
                target=self._run_warmup_in_background,
                name="sbmpc_planner_warmup",
                daemon=True,
            )
            self._warmup_thread.start()

    def _run_warmup_in_background(self) -> None:
        try:
            self._run_warmup()
            self.get_logger().info("Planner warmup/JIT compilation complete.")
        except Exception as exc:
            self._latch_fatal_error(f"Planner warmup failed: {exc}")
        finally:
            self._publish_diagnostics()

    def _run_warmup(self, planner_input: PlannerInput | None = None) -> None:
        warmup_output = None
        with self._warmup_lock:
            if self._warmup_complete:
                return
            for _ in range(self._planner_warmup_iterations):
                if self._closing.is_set():
                    return
                with self._planner_lock:
                    warmup_output = self._planner.warmup()
                if warmup_output is not None:
                    self._record_planner_diagnostics(warmup_output)
                    if planner_input is not None:
                        self._predict_control_initial_state(planner_input, warmup_output)
            self._warmup_rollout_marker_generation()
            reset = getattr(self._planner, "reset_runtime_state_after_warmup", None)
            if callable(reset):
                with self._planner_lock:
                    reset()
            self._clear_latest_planner_output()

        self._warmup_count += 1
        self._warmup_complete = True
        if self._snapshot_planner_input() is None:
            self._state = "waiting_for_sensor"

    def _warmup_rollout_marker_generation(self) -> None:
        if not self._publish_rollout_markers_enabled():
            return
        warmup = getattr(self._planner, "warmup_rollout_visualization", None)
        if not callable(warmup):
            return
        start = perf_counter()
        warmup(max_rollouts=self._rollout_marker_max_samples())
        self.get_logger().info(
            "SB-MPC rollout marker visualization warmed in "
            f"{1000.0 * (perf_counter() - start):.1f} ms."
        )

    def _observe_planning_deadline(self, planning_duration_sec: float) -> None:
        try:
            self._deadline_monitor.observe(planning_duration_sec)
        except UnsafeControlError as exc:
            self._last_error = str(exc)
            self.get_logger().warn(str(exc))

    def _log_solve(
        self,
        step_wall_sec: float,
        planner_input: PlannerInput,
        planner_output: object,
    ) -> None:
        """Log one line per planner solve with the time the controller spent.

        Runs on the planner worker thread (off the 25 Hz publish path) and is
        called AFTER ``step_wall_sec`` is measured, so the optional EE-error FK
        below never inflates ``planning_time_ms``, the reported wall time, or the
        control publish path.
        """
        diagnostics = getattr(planner_output, "diagnostics", None)
        budget_ms = 1000.0 * self._deadline_monitor.max_planning_duration_sec
        wall_ms = 1000.0 * step_wall_sec

        def fmt(value: object, spec: str = "{:.1f}") -> str:
            return "n/a" if value is None else spec.format(float(value))

        over = " OVER" if budget_ms > 0.0 and wall_ms > budget_ms else ""
        self.get_logger().info(
            f"[sbmpc solve #{self._planner_step_count}] "
            f"wall {wall_ms:.1f}/{budget_ms:.1f}ms{over} | "
            f"plan {fmt(getattr(diagnostics, 'planning_time_ms', None))} "
            f"(cmd {fmt(getattr(diagnostics, 'planner_command_time_ms', None))}, "
            f"prep {fmt(getattr(diagnostics, 'planner_prepare_time_ms', None))}) | "
            f"|tau|={fmt(getattr(diagnostics, 'torque_norm', None))} "
            f"|gain|={fmt(getattr(diagnostics, 'gain_norm', None))} "
            f"eeErr={fmt(self._ee_position_error(planner_input, diagnostics), '{:.4f}')}m | "
            f"deadline_miss={self._deadline_monitor.deadline_miss_count}"
        )

    def _ee_position_error(
        self,
        planner_input: PlannerInput,
        diagnostics: object,
    ) -> float | None:
        """Distance from the measured EE to the goal (one jitted MJX FK).

        Best-effort and never raises: a logging metric must not fault the loop.
        """
        goal = getattr(diagnostics, "goal_position", None)
        ee_position = getattr(self._planner, "ee_position", None)
        if goal is None or not callable(ee_position):
            return None
        try:
            ee = ee_position(planner_input.q)
            if ee is None:
                return None
            return float(
                np.linalg.norm(
                    np.asarray(ee, dtype=np.float64)
                    - np.asarray(goal, dtype=np.float64)
                )
            )
        except Exception:
            return None

    def _publish_emergency_hold(
        self,
        planner_input: PlannerInput,
        *,
        reason: str,
    ) -> None:
        if self._published_control_count <= 0:
            return

        self._publish_hold_control(
            planner_input,
            position_gain=self._nonnegative_double_parameter(
                "emergency_hold_position_gain"
            ),
            velocity_gain=self._nonnegative_double_parameter(
                "emergency_hold_velocity_gain"
            ),
            reason=reason,
        )

    def _publish_hold_before_fatal_if_needed(self) -> None:
        with self._fatal_error_lock:
            message = self._fatal_error
        if message is None or self._published_control_count <= 0:
            return

        planner_input = self._snapshot_planner_input()
        if planner_input is None:
            return
        self._publish_emergency_hold(
            planner_input,
            reason=f"fatal error handoff: {message}",
        )

    def _publish_hold_control(
        self,
        planner_input: PlannerInput,
        *,
        position_gain: float,
        velocity_gain: float,
        reason: str,
    ) -> None:
        feedforward = self._emergency_hold_feedforward(planner_input)
        control = hold_control_from_sensor(
            planner_input,
            feedforward=feedforward,
            position_gain=position_gain,
            velocity_gain=velocity_gain,
        )
        self._publish_control(control)
        self.get_logger().warn(
            f"published hold control for {reason}: "
            f"position_gain={position_gain:.3f}, "
            f"velocity_gain={velocity_gain:.3f}."
        )

    def _emergency_hold_feedforward(self, planner_input: PlannerInput) -> np.ndarray:
        gravity_torques = getattr(self._planner, "gravity_torques", None)
        if callable(gravity_torques):
            try:
                tau = np.asarray(
                    gravity_torques(planner_input.q),
                    dtype=np.float64,
                ).reshape(-1)
                if tau.shape == (len(self._joint_names),) and np.all(np.isfinite(tau)):
                    return tau
            except Exception as exc:
                self.get_logger().warn(
                    "failed to compute emergency gravity hold torques; "
                    f"falling back to latest feedforward: {exc}"
                )

        with self._latest_planner_output_lock:
            latest = self._latest_planner_output
        if latest is not None:
            tau = np.asarray(
                getattr(latest.planner_output, "tau_ff"),
                dtype=np.float64,
            ).reshape(-1)
            if tau.shape == (len(self._joint_names),) and np.all(np.isfinite(tau)):
                return tau

        self.get_logger().warn(
            "no valid emergency hold feedforward available; using zero feedforward."
        )
        return np.zeros(len(self._joint_names), dtype=np.float64)

    def _hold_on_disarm_enabled(self) -> bool:
        return (
            self.get_parameter("hold_on_disarm_after_control")
            .get_parameter_value()
            .bool_value
        )

    def _apply_feedforward_feedback_regularization(
        self,
        control: Control,
        *,
        planner_output: object | None = None,
    ) -> Control:
        position_gain = self._nonnegative_double_parameter(
            "feedforward_position_gain"
        )
        velocity_gain = self._nonnegative_double_parameter(
            "feedforward_velocity_damping_gain"
        )
        if position_gain <= 0.0 and velocity_gain <= 0.0:
            return control
        if self._planner_output_gain_mode(planner_output) != "feedforward":
            return control

        joint_count = len(control.initial_state.joint_state.position)
        reference_state = self._planner_reference_state(planner_output, joint_count)
        if reference_state is not None:
            q_ref, v_ref = reference_state
            control.initial_state.joint_state.position = q_ref.tolist()
            control.initial_state.joint_state.velocity = v_ref.tolist()

        gain = float64_multi_array_to_numpy(control.feedback_gain).copy()
        expected_shape = (joint_count, 2 * joint_count)
        if gain.shape != expected_shape:
            raise UnsafeControlError(
                "feedback_gain shape is incompatible with feedforward damping: "
                f"got {gain.shape}, expected {expected_shape}."
            )
        if position_gain > 0.0:
            gain[:, :joint_count] += np.eye(joint_count, dtype=np.float64) * position_gain
        if velocity_gain > 0.0:
            gain[:, joint_count:] += (
                np.eye(joint_count, dtype=np.float64) * velocity_gain
            )
            if reference_state is None:
                control.initial_state.joint_state.velocity = [0.0] * joint_count

        control.feedback_gain = numpy_to_float64_multi_array(gain)
        return control

    def _planner_output_gain_mode(self, planner_output: object | None) -> str | None:
        diagnostics = getattr(planner_output, "diagnostics", None)
        mode = getattr(diagnostics, "gain_mode", None)
        if mode is None:
            mode = self._planner_mode
        if mode is None:
            return None
        return str(mode).strip().lower()

    @staticmethod
    def _planner_reference_state(
        planner_output: object | None,
        joint_count: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if planner_output is None:
            return None
        q_ref = getattr(planner_output, "reference_q", None)
        v_ref = getattr(planner_output, "reference_v", None)
        if q_ref is None or v_ref is None:
            return None
        q = np.asarray(q_ref, dtype=np.float64).reshape(-1)
        v = np.asarray(v_ref, dtype=np.float64).reshape(-1)
        if q.shape != (joint_count,) or v.shape != (joint_count,):
            raise UnsafeControlError(
                "planner reference state is incompatible with feedforward tracking: "
                f"q_ref={q.shape}, v_ref={v.shape}, expected ({joint_count},)."
            )
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
            raise UnsafeControlError("planner reference state contains non-finite values.")
        return q, v

    def _latch_fatal_error(self, message: str) -> None:
        with self._fatal_error_lock:
            if self._fatal_error is not None:
                return
            self._fatal_error = str(message)
        self._state = "error"
        self._last_error = str(message)
        self._clear_latest_planner_output()
        self.get_logger().error(str(message))

    def _raise_if_fatal(self) -> None:
        with self._fatal_error_lock:
            message = self._fatal_error
        if message is not None:
            raise RuntimeError(message)

    def _control_initial_state_prediction_sec(self) -> float:
        value = (
            self.get_parameter("control_initial_state_prediction_sec")
            .get_parameter_value()
            .double_value
        )
        return max(0.0, float(value))

    def _nonnegative_double_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).get_parameter_value().double_value)
        if value < 0.0 or not np.isfinite(value):
            raise ValueError(f"{name} must be finite and non-negative.")
        return value

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

        self._last_planning_time_ms = self._maybe_float(
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

        self._planner_mode = self._maybe_text(
            getattr(diagnostics, "gain_mode", None)
        )
        self._last_planner_prepare_time_ms = self._maybe_float(
            getattr(diagnostics, "planner_prepare_time_ms", None)
        )
        self._last_planner_command_time_ms = self._maybe_float(
            getattr(diagnostics, "planner_command_time_ms", None)
        )
        self._last_reference_q = self._maybe_float_vector(
            getattr(planner_output, "reference_q", None),
            expected_size=len(self._joint_names),
        )
        self._last_reference_v = self._maybe_float_vector(
            getattr(planner_output, "reference_v", None),
            expected_size=len(self._joint_names),
        )

        goal_position = getattr(diagnostics, "goal_position", None)
        if goal_position is None:
            self._last_goal_position = None
            return

        goal_array = np.asarray(goal_position, dtype=np.float64).reshape(-1)
        self._last_goal_position = goal_array.tolist()

    def _rollout_marker_max_samples(self) -> int:
        return max(
            0,
            int(
                self.get_parameter("rollout_marker_max_samples")
                .get_parameter_value()
                .integer_value
            ),
        )

    def _rollout_marker_publish_period_sec(self) -> float | None:
        value = float(
            self.get_parameter("rollout_marker_publish_rate_hz")
            .get_parameter_value()
            .double_value
        )
        if value <= 0.0:
            return None
        return 1.0 / value

    def _rollout_marker_publish_due(self) -> bool:
        period_sec = self._rollout_marker_publish_period_sec()
        if period_sec is None:
            return False
        now_sec = perf_counter()
        last_sec = self._last_rollout_marker_publish_wall_sec
        if last_sec is not None and now_sec - last_sec < period_sec:
            return False
        self._last_rollout_marker_publish_wall_sec = now_sec
        return True

    def _publish_rollout_markers(
        self,
        planner_input: PlannerInput,
        planner_output: object,
    ) -> None:
        if self._rollout_marker_publisher is None:
            return
        if not self._rollout_marker_publish_due():
            return

        try:
            optimized_path_fn = getattr(self._planner, "planned_end_effector_path", None)
            if not callable(optimized_path_fn):
                return
            optimized_path_result = optimized_path_fn(planner_input)
            if optimized_path_result is None:
                return
            _, ee_path = optimized_path_result
            ee_path = np.asarray(ee_path, dtype=np.float64)
            if ee_path.ndim != 2 or ee_path.shape[1] != 3:
                raise ValueError(f"unexpected optimized EE rollout shape: {ee_path.shape}.")

            rollout_paths = None
            rollout_path_fn = getattr(
                self._planner,
                "representative_end_effector_rollouts",
                None,
            )
            if callable(rollout_path_fn):
                rollout_paths = rollout_path_fn(
                    planner_input,
                    max_rollouts=self._rollout_marker_max_samples(),
                )

            goal_position = goal_position_from_planner_output(
                planner_output,
                ee_path[-1],
            )
            markers = make_trajectory_marker_array(
                ee_path=ee_path,
                rollout_paths=rollout_paths,
                goal_position=goal_position,
                frame_id=(
                    self.get_parameter("rollout_marker_frame_id")
                    .get_parameter_value()
                    .string_value
                ),
                stamp=self.get_clock().now().to_msg(),
                line_width=(
                    self.get_parameter("rollout_marker_line_width")
                    .get_parameter_value()
                    .double_value
                ),
                sample_line_width=(
                    self.get_parameter("rollout_marker_sample_line_width")
                    .get_parameter_value()
                    .double_value
                ),
                point_diameter=(
                    self.get_parameter("rollout_marker_point_diameter")
                    .get_parameter_value()
                    .double_value
                ),
                goal_diameter=(
                    self.get_parameter("rollout_marker_goal_diameter")
                    .get_parameter_value()
                    .double_value
                ),
            )
            self._rollout_marker_publisher.publish(markers)
        except Exception as exc:
            self.get_logger().warn(f"failed to publish SB-MPC rollout markers: {exc}")

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

    def destroy_node(self) -> None:
        try:
            self._closing.set()
            self._planner_request.set()
            planner_thread = self._planner_thread
            if planner_thread is not None and planner_thread.is_alive():
                planner_thread.join(timeout=1.0)
            warmup_thread = self._warmup_thread
            if warmup_thread is not None and warmup_thread.is_alive():
                warmup_thread.join(timeout=1.0)
            close = getattr(self._planner, "close", None)
            if callable(close) and self._planner_lock.acquire(timeout=1.0):
                try:
                    close()
                finally:
                    self._planner_lock.release()
        finally:
            super().destroy_node()

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

    @staticmethod
    def _maybe_float_vector(
        value: object | None,
        *,
        expected_size: int | None = None,
    ) -> list[float] | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if expected_size is not None and array.shape != (expected_size,):
            return None
        if not np.all(np.isfinite(array)):
            return None
        return array.tolist()

    @staticmethod
    def _maybe_text(value: object | None) -> str | None:
        if value is None:
            return None
        return str(value)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    bridge_cpu_affinity = _prefer_bridge_cpu_affinity()
    node = SbMpcLfcBridgeNode()
    if bridge_cpu_affinity is not None:
        node.get_logger().info(
            "SB-MPC bridge CPU affinity set to "
            f"{list(bridge_cpu_affinity)}."
        )
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(signum, frame) -> None:
        del signum, frame
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok(context=node.context):
            raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
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
