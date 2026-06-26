from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from linear_feedback_controller_msgs.msg import Sensor
from sensor_msgs.msg import JointState

from sbmpc_ros_bridge.joint_mapping import PANDA_ARM_JOINT_NAMES
from sbmpc_ros_bridge.lfc_msg_adapter import (
    float64_multi_array_to_numpy,
    planner_output_to_control,
    sensor_to_planner_input,
)
from sbmpc_ros_bridge.safety import UnsafeControlError


@dataclass(frozen=True)
class FakePlannerOutput:
    tau_ff: np.ndarray
    K: np.ndarray


def make_sensor() -> Sensor:
    return Sensor(
        joint_state=JointState(
            name=list(PANDA_ARM_JOINT_NAMES),
            position=[0.1 * i for i in range(7)],
            velocity=[-0.2 * i for i in range(7)],
            effort=[0.0] * 7,
        )
    )


def test_sensor_to_planner_input_extracts_ros_ready_joint_arrays() -> None:
    planner_input = sensor_to_planner_input(make_sensor())

    np.testing.assert_allclose(planner_input.q, np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    np.testing.assert_allclose(
        planner_input.v,
        np.asarray([0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2]),
    )
    assert tuple(planner_input.sensor.joint_state.name) == PANDA_ARM_JOINT_NAMES


def test_planner_output_to_control_uses_lfc_matrix_layouts() -> None:
    sensor = make_sensor()
    planner_input = sensor_to_planner_input(sensor)
    planner_output = FakePlannerOutput(
        tau_ff=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        K=np.arange(7 * 14, dtype=np.float64).reshape(7, 14),
    )

    control = planner_output_to_control(planner_output, planner_input)

    feedback_gain = float64_multi_array_to_numpy(control.feedback_gain)
    feedforward = float64_multi_array_to_numpy(control.feedforward)

    np.testing.assert_allclose(feedback_gain, -planner_output.K)
    np.testing.assert_allclose(feedforward, planner_output.tau_ff.reshape(7, 1))
    assert control.feedback_gain.layout.dim[0].label == "rows"
    assert control.feedback_gain.layout.dim[0].size == 7
    assert control.feedback_gain.layout.dim[1].label == "cols"
    assert control.feedback_gain.layout.dim[1].size == 14
    assert control.feedforward.layout.dim[0].size == 7
    assert control.feedforward.layout.dim[1].size == 1


def test_planner_output_to_control_deep_copies_initial_state_snapshot() -> None:
    sensor = make_sensor()
    planner_input = sensor_to_planner_input(sensor)
    planner_output = FakePlannerOutput(
        tau_ff=np.zeros(7, dtype=np.float64),
        K=np.zeros((7, 14), dtype=np.float64),
    )

    control = planner_output_to_control(planner_output, planner_input)
    sensor.joint_state.position[0] = 99.0

    assert control.initial_state == planner_input.sensor
    assert control.initial_state is not planner_input.sensor
    assert control.initial_state.joint_state is not planner_input.sensor.joint_state
    assert control.initial_state.joint_state.position[0] == 0.0


@pytest.mark.parametrize(
    ("tau_ff", "feedback_gain", "message"),
    [
        (
            np.zeros(6, dtype=np.float64),
            np.zeros((7, 14), dtype=np.float64),
            "feedforward must have shape",
        ),
        (
            np.zeros(7, dtype=np.float64),
            np.zeros((7, 13), dtype=np.float64),
            "feedback_gain must have shape",
        ),
        (
            np.asarray([np.nan] + [0.0] * 6, dtype=np.float64),
            np.zeros((7, 14), dtype=np.float64),
            "feedforward contains non-finite",
        ),
        (
            np.zeros(7, dtype=np.float64),
            np.pad(np.asarray([[np.nan]], dtype=np.float64), ((0, 6), (0, 13))),
            "feedback_gain contains non-finite",
        ),
    ],
)
def test_planner_output_to_control_rejects_invalid_outputs(
    tau_ff: np.ndarray,
    feedback_gain: np.ndarray,
    message: str,
) -> None:
    planner_output = FakePlannerOutput(tau_ff=tau_ff, K=feedback_gain)

    with pytest.raises(UnsafeControlError, match=message):
        planner_output_to_control(planner_output, make_sensor())


def test_planner_output_to_control_can_override_default_sign_flip() -> None:
    planner_output = FakePlannerOutput(
        tau_ff=np.zeros(7, dtype=np.float64),
        K=np.eye(7, 14, dtype=np.float64),
    )

    control = planner_output_to_control(planner_output, make_sensor(), gain_scale=1.0)

    np.testing.assert_allclose(
        float64_multi_array_to_numpy(control.feedback_gain),
        planner_output.K,
    )
