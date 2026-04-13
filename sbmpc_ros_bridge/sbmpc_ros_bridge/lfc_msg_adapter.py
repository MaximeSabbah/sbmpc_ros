from __future__ import annotations

from copy import deepcopy
from typing import Protocol

import numpy as np
from linear_feedback_controller_msgs.msg import Control, Sensor
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout

from sbmpc_ros_bridge.joint_mapping import JointMapper
from sbmpc_ros_bridge.planner_adapter import PlannerInput
from sbmpc_ros_bridge.safety import (
    BridgeSafetyProfile,
    SBMPC_TO_LFC_GAIN_SCALE,
    ControlSafetyLimits,
    sbmpc_gain_to_lfc_gain,
    validate_planner_output,
)


class PlannerOutputLike(Protocol):
    tau_ff: np.ndarray
    K: np.ndarray


class ZeroPlannerOutput:
    def __init__(self, control_dim: int = 7, state_dim: int = 14) -> None:
        self.tau_ff = np.zeros(control_dim, dtype=np.float64)
        self.K = np.zeros((control_dim, state_dim), dtype=np.float64)


def sensor_to_planner_input(
    sensor: Sensor,
    *,
    joint_mapper: JointMapper | None = None,
    allow_reordering: bool = False,
) -> PlannerInput:
    mapper = JointMapper.panda() if joint_mapper is None else joint_mapper
    canonical_sensor = deepcopy(
        mapper.reorder_sensor(sensor, allow_reordering=allow_reordering)
    )
    q = np.asarray(canonical_sensor.joint_state.position, dtype=np.float64)
    v = np.asarray(canonical_sensor.joint_state.velocity, dtype=np.float64)
    return PlannerInput(sensor=canonical_sensor, q=q, v=v)


def planner_output_to_control(
    planner_output: PlannerOutputLike,
    planned_state: PlannerInput | Sensor,
    *,
    gain_scale: float | None = None,
    safety_limits: ControlSafetyLimits | None = None,
    safety_profile: BridgeSafetyProfile | None = None,
) -> Control:
    if safety_profile is None:
        effective_gain_scale = (
            SBMPC_TO_LFC_GAIN_SCALE if gain_scale is None else gain_scale
        )
        effective_limits = safety_limits
    else:
        effective_gain_scale = safety_profile.always_on.gain_scale
        if gain_scale is not None:
            effective_gain_scale = gain_scale
        effective_limits = safety_profile.planner_output_limits()
        if safety_limits is not None:
            effective_limits = safety_limits

    sensor_snapshot = (
        planned_state.sensor if isinstance(planned_state, PlannerInput) else planned_state
    )
    tau_ff, feedback_gain = validate_planner_output(
        planner_output.tau_ff,
        planner_output.K,
        limits=effective_limits,
    )

    control = Control()
    control.header = deepcopy(sensor_snapshot.header)
    control.feedback_gain = numpy_to_float64_multi_array(
        sbmpc_gain_to_lfc_gain(feedback_gain, gain_scale=effective_gain_scale)
    )
    control.feedforward = numpy_to_float64_multi_array(tau_ff.reshape((-1, 1)))
    control.initial_state = deepcopy(sensor_snapshot)
    return control


def zero_control_from_sensor(
    sensor_snapshot: PlannerInput | Sensor,
    *,
    control_dim: int = 7,
    state_dim: int = 14,
) -> Control:
    return planner_output_to_control(
        ZeroPlannerOutput(control_dim=control_dim, state_dim=state_dim),
        sensor_snapshot,
        gain_scale=1.0,
    )


def numpy_to_float64_multi_array(values: np.ndarray) -> Float64MultiArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape((-1, 1))
    if array.ndim != 2:
        raise ValueError(f"expected a 1D or 2D array, got shape {array.shape}.")

    rows, cols = array.shape
    return Float64MultiArray(
        layout=MultiArrayLayout(
            dim=[
                MultiArrayDimension(label="rows", size=rows, stride=rows * cols),
                MultiArrayDimension(label="cols", size=cols, stride=cols),
            ],
            data_offset=0,
        ),
        data=array.reshape(-1, order="C").tolist(),
    )


def float64_multi_array_to_numpy(message: Float64MultiArray) -> np.ndarray:
    if len(message.layout.dim) != 2:
        raise ValueError(
            "Float64MultiArray must contain exactly two dimensions for LFC."
        )

    rows = int(message.layout.dim[0].size)
    cols = int(message.layout.dim[1].size)
    stride0 = int(message.layout.dim[0].stride)
    stride1 = int(message.layout.dim[1].stride)

    if stride0 != rows * cols:
        raise ValueError(
            f"unexpected row stride {stride0}; expected {rows * cols} for {rows}x{cols}."
        )
    if stride1 != cols:
        raise ValueError(
            f"unexpected column stride {stride1}; expected {cols} for {rows}x{cols}."
        )
    if len(message.data) != rows * cols:
        raise ValueError(
            f"data length {len(message.data)} does not match {rows}x{cols} layout."
        )

    return np.asarray(message.data, dtype=np.float64).reshape((rows, cols), order="C")
