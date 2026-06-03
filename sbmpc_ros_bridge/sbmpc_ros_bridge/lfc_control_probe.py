from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import json
from typing import Sequence

import numpy as np
import rclpy
from linear_feedback_controller_msgs.msg import Control, Sensor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from sbmpc_ros_bridge.lfc_msg_adapter import float64_multi_array_to_numpy


@dataclass(frozen=True, slots=True)
class LfcCommandEstimate:
    joint_names: tuple[str, ...]
    feedforward: np.ndarray
    feedback_effort: np.ndarray
    total_effort: np.ndarray
    desired_position: np.ndarray
    desired_velocity: np.ndarray
    measured_position: np.ndarray
    measured_velocity: np.ndarray
    q_error: np.ndarray
    v_error: np.ndarray
    control_age_sec: float
    feedback_gain_norm: float

    @property
    def max_abs_feedforward(self) -> float:
        return _max_abs(self.feedforward)

    @property
    def max_abs_feedback_effort(self) -> float:
        return _max_abs(self.feedback_effort)

    @property
    def max_abs_total_effort(self) -> float:
        return _max_abs(self.total_effort)

    @property
    def q_error_norm(self) -> float:
        return float(np.linalg.norm(self.q_error))

    @property
    def v_error_norm(self) -> float:
        return float(np.linalg.norm(self.v_error))

    def peak_joint(self) -> tuple[str, float]:
        if self.total_effort.size == 0:
            return "", 0.0
        index = int(np.argmax(np.abs(self.total_effort)))
        name = self.joint_names[index] if index < len(self.joint_names) else str(index)
        return name, float(self.total_effort[index])

    def to_dict(self) -> dict[str, object]:
        peak_joint, peak_effort = self.peak_joint()
        return {
            "joint_names": list(self.joint_names),
            "control_age_sec": self.control_age_sec,
            "max_abs_feedforward": self.max_abs_feedforward,
            "max_abs_feedback_effort": self.max_abs_feedback_effort,
            "max_abs_total_effort": self.max_abs_total_effort,
            "feedback_gain_norm": self.feedback_gain_norm,
            "q_error_norm": self.q_error_norm,
            "v_error_norm": self.v_error_norm,
            "peak_joint": peak_joint,
            "peak_effort": peak_effort,
            "feedforward": self.feedforward.tolist(),
            "feedback_effort": self.feedback_effort.tolist(),
            "total_effort": self.total_effort.tolist(),
            "desired_position": self.desired_position.tolist(),
            "desired_velocity": self.desired_velocity.tolist(),
            "measured_position": self.measured_position.tolist(),
            "measured_velocity": self.measured_velocity.tolist(),
            "q_error": self.q_error.tolist(),
            "v_error": self.v_error.tolist(),
        }


def estimate_lfc_command(
    control: Control,
    sensor: Sensor,
    *,
    now_sec: float | None = None,
) -> LfcCommandEstimate:
    """Estimate the fixed-base LFC torque command for a Control/Sensor pair."""

    desired_joint_state = control.initial_state.joint_state
    measured_joint_state = sensor.joint_state
    joint_names = _reference_joint_names(desired_joint_state.name, measured_joint_state.name)

    q_des = _joint_vector(
        desired_joint_state.position,
        desired_joint_state.name,
        joint_names,
        field_name="control.initial_state.joint_state.position",
    )
    v_des = _joint_vector(
        desired_joint_state.velocity,
        desired_joint_state.name,
        joint_names,
        field_name="control.initial_state.joint_state.velocity",
    )
    q_meas = _joint_vector(
        measured_joint_state.position,
        measured_joint_state.name,
        joint_names,
        field_name="sensor.joint_state.position",
    )
    v_meas = _joint_vector(
        measured_joint_state.velocity,
        measured_joint_state.name,
        joint_names,
        field_name="sensor.joint_state.velocity",
    )

    q_error = q_des - q_meas
    v_error = v_des - v_meas
    state_error = np.concatenate([q_error, v_error])

    feedforward = float64_multi_array_to_numpy(control.feedforward).reshape(-1)
    feedback_gain = float64_multi_array_to_numpy(control.feedback_gain)
    if feedforward.shape != (len(joint_names),):
        raise ValueError(
            "control.feedforward has shape "
            f"{feedforward.shape}, expected ({len(joint_names)},)."
        )
    expected_gain_shape = (len(joint_names), 2 * len(joint_names))
    if feedback_gain.shape != expected_gain_shape:
        raise ValueError(
            f"control.feedback_gain has shape {feedback_gain.shape}, "
            f"expected {expected_gain_shape}."
        )

    feedback_effort = feedback_gain @ state_error
    total_effort = feedforward + feedback_effort
    return LfcCommandEstimate(
        joint_names=tuple(joint_names),
        feedforward=feedforward,
        feedback_effort=feedback_effort,
        total_effort=total_effort,
        desired_position=q_des,
        desired_velocity=v_des,
        measured_position=q_meas,
        measured_velocity=v_meas,
        q_error=q_error,
        v_error=v_error,
        control_age_sec=_control_age_sec(control, now_sec=now_sec),
        feedback_gain_norm=float(np.linalg.norm(feedback_gain)),
    )


def _reference_joint_names(
    desired_names: Sequence[str],
    measured_names: Sequence[str],
) -> tuple[str, ...]:
    if desired_names:
        return tuple(str(name) for name in desired_names)
    if measured_names:
        return tuple(str(name) for name in measured_names)
    raise ValueError("joint names are required to estimate the LFC command.")


def _joint_vector(
    values: Sequence[float],
    names: Sequence[str],
    reference_names: Sequence[str],
    *,
    field_name: str,
) -> np.ndarray:
    value_array = np.asarray(values, dtype=np.float64).reshape(-1)
    if value_array.size != len(names):
        raise ValueError(
            f"{field_name} has {value_array.size} values but {len(names)} names."
        )
    if tuple(names) == tuple(reference_names):
        return value_array

    indexed_values = {name: value for name, value in zip(names, value_array)}
    missing = [name for name in reference_names if name not in indexed_values]
    if missing:
        raise ValueError(f"{field_name} is missing joints: {missing}.")
    return np.asarray([indexed_values[name] for name in reference_names], dtype=np.float64)


def _control_age_sec(control: Control, *, now_sec: float | None) -> float:
    if now_sec is None:
        return float("nan")
    stamp = control.header.stamp.sec + 1e-9 * control.header.stamp.nanosec
    return float(now_sec) - float(stamp)


def _max_abs(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


class LfcControlProbe(Node):
    def __init__(
        self,
        *,
        sensor_topic: str,
        control_topic: str,
        warn_abs_effort: float,
        max_samples: int,
        json_output: bool,
    ) -> None:
        super().__init__("sbmpc_lfc_control_probe")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._latest_sensor: Sensor | None = None
        self._warn_abs_effort = float(warn_abs_effort)
        self._max_samples = int(max_samples)
        self._json_output = bool(json_output)
        self._sample_count = 0
        self.create_subscription(Sensor, sensor_topic, self._on_sensor, qos)
        self.create_subscription(Control, control_topic, self._on_control, qos)
        self.get_logger().info(
            "Estimating LFC commands from "
            f"sensor_topic={sensor_topic!r}, control_topic={control_topic!r}."
        )

    def _on_sensor(self, message: Sensor) -> None:
        self._latest_sensor = message

    def _on_control(self, message: Control) -> None:
        if self._latest_sensor is None:
            self.get_logger().warn("Received control before any sensor sample.")
            return

        now = self.get_clock().now()
        now_sec = now.nanoseconds * 1e-9
        try:
            estimate = estimate_lfc_command(message, self._latest_sensor, now_sec=now_sec)
        except Exception as exc:
            self.get_logger().error(f"Failed to estimate LFC command: {exc}")
            return

        self._sample_count += 1
        if self._json_output:
            self.get_logger().info(json.dumps(estimate.to_dict(), sort_keys=True))
        else:
            peak_joint, peak_effort = estimate.peak_joint()
            text = (
                f"sample={self._sample_count} "
                f"age={estimate.control_age_sec:.4f}s "
                f"max|ff|={estimate.max_abs_feedforward:.3f} "
                f"max|fb|={estimate.max_abs_feedback_effort:.3f} "
                f"max|tau|={estimate.max_abs_total_effort:.3f} "
                f"|K|={estimate.feedback_gain_norm:.3f} "
                f"qerr={estimate.q_error_norm:.5f} "
                f"verr={estimate.v_error_norm:.5f} "
                f"peak={peak_joint}:{peak_effort:.3f}"
            )
            if estimate.max_abs_total_effort > self._warn_abs_effort:
                self.get_logger().warn(text)
            else:
                self.get_logger().info(text)

        if self._max_samples > 0 and self._sample_count >= self._max_samples:
            rclpy.shutdown(context=self.context)


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(description="Estimate LFC effort commands from ROS topics.")
    parser.add_argument("--sensor-topic", default="/sensor")
    parser.add_argument("--control-topic", default="/control")
    parser.add_argument("--warn-abs-effort", type=float, default=15.0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Exit after this many controls; 0 means run until interrupted.",
    )
    parser.add_argument("--json", action="store_true", help="Log full JSON payloads.")
    args = parser.parse_args(argv)

    rclpy.init(args=None)
    node = LfcControlProbe(
        sensor_topic=args.sensor_topic,
        control_topic=args.control_topic,
        warn_abs_effort=args.warn_abs_effort,
        max_samples=args.max_samples,
        json_output=args.json,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
