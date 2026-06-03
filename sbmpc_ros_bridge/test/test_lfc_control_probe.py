from __future__ import annotations

import numpy as np
import pytest
from linear_feedback_controller_msgs.msg import Control, Sensor
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from sbmpc_ros_bridge.lfc_control_probe import estimate_lfc_command
from sbmpc_ros_bridge.lfc_msg_adapter import numpy_to_float64_multi_array


def make_sensor(*, names: list[str], q: list[float], v: list[float]) -> Sensor:
    return Sensor(
        joint_state=JointState(
            name=names,
            position=q,
            velocity=v,
            effort=[0.0] * len(names),
        )
    )


def make_control() -> Control:
    control = Control()
    control.header = Header()
    control.header.stamp.sec = 10
    control.initial_state = make_sensor(
        names=["joint2", "joint1"],
        q=[2.0, 1.0],
        v=[0.2, 0.1],
    )
    control.feedforward = numpy_to_float64_multi_array(
        np.asarray([1.0, -1.0], dtype=np.float64)
    )
    gain = np.asarray(
        [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 20.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    control.feedback_gain = numpy_to_float64_multi_array(gain)
    return control


def test_estimate_lfc_command_reorders_sensor_to_control_joint_names() -> None:
    sensor = make_sensor(
        names=["joint1", "joint2"],
        q=[0.5, 1.5],
        v=[0.01, 0.02],
    )

    estimate = estimate_lfc_command(make_control(), sensor, now_sec=12.0)

    assert estimate.joint_names == ("joint2", "joint1")
    np.testing.assert_allclose(estimate.desired_position, [2.0, 1.0])
    np.testing.assert_allclose(estimate.desired_velocity, [0.2, 0.1])
    np.testing.assert_allclose(estimate.measured_position, [1.5, 0.5])
    np.testing.assert_allclose(estimate.measured_velocity, [0.02, 0.01])
    np.testing.assert_allclose(estimate.q_error, [0.5, 0.5])
    np.testing.assert_allclose(estimate.v_error, [0.18, 0.09])
    np.testing.assert_allclose(estimate.feedback_effort, [5.0, 10.0])
    np.testing.assert_allclose(estimate.total_effort, [6.0, 9.0])
    assert estimate.control_age_sec == pytest.approx(2.0)
    assert estimate.max_abs_feedforward == pytest.approx(1.0)
    assert estimate.max_abs_feedback_effort == pytest.approx(10.0)
    assert estimate.max_abs_total_effort == pytest.approx(9.0)
    assert estimate.feedback_gain_norm == pytest.approx(np.sqrt(500.0))
    assert estimate.peak_joint() == ("joint1", 9.0)


def test_estimate_lfc_command_rejects_missing_sensor_joint() -> None:
    sensor = make_sensor(
        names=["joint1", "joint3"],
        q=[0.5, 1.5],
        v=[0.01, 0.02],
    )

    with pytest.raises(ValueError, match="missing joints"):
        estimate_lfc_command(make_control(), sensor)
