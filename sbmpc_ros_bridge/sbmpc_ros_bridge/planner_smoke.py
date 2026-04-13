from __future__ import annotations

import argparse
import json

import numpy as np
from linear_feedback_controller_msgs.msg import Sensor
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from sbmpc_ros_bridge.joint_mapping import JointMapper, PANDA_ARM_JOINT_NAMES
from sbmpc_ros_bridge.lfc_msg_adapter import (
    float64_multi_array_to_numpy,
    planner_output_to_control,
    sensor_to_planner_input,
)
from sbmpc_ros_bridge.planner_adapter import SbMpcPlannerAdapter


FR3_ARM_JOINT_NAMES: tuple[str, ...] = tuple(f"fr3_joint{i}" for i in range(1, 8))
FER_ARM_JOINT_NAMES: tuple[str, ...] = tuple(f"fer_joint{i}" for i in range(1, 8))
JOINT_NAME_SETS: dict[str, tuple[str, ...]] = {
    "fer": FER_ARM_JOINT_NAMES,
    "fr3": FR3_ARM_JOINT_NAMES,
    "panda": PANDA_ARM_JOINT_NAMES,
}


def build_sensor(
    *,
    joint_names: tuple[str, ...],
    q: np.ndarray,
    v: np.ndarray,
) -> Sensor:
    return Sensor(
        header=Header(),
        joint_state=JointState(
            header=Header(),
            name=list(joint_names),
            position=np.asarray(q, dtype=np.float64).tolist(),
            velocity=np.asarray(v, dtype=np.float64).tolist(),
            effort=[0.0] * len(joint_names),
        ),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Warm up and step the sbmpc planner through the ROS bridge adapter "
            "using a synthetic LFC Sensor message."
        )
    )
    parser.add_argument(
        "--joint-set",
        choices=tuple(sorted(JOINT_NAME_SETS)),
        default="fer",
        help=(
            "Runtime joint naming to stamp into the synthetic Sensor and Control "
            "messages. The FER robot in the local Franka stack uses 'fer'."
        ),
    )
    args = parser.parse_args(argv)

    adapter = SbMpcPlannerAdapter()
    joint_names = JOINT_NAME_SETS[args.joint_set]
    home_q = np.asarray(adapter._controller.planner.home_q, dtype=np.float64)
    zero_v = np.zeros_like(home_q)

    planner_input = sensor_to_planner_input(
        build_sensor(joint_names=joint_names, q=home_q, v=zero_v),
        joint_mapper=JointMapper(expected_names=joint_names),
    )
    warmup_output = adapter.warmup()
    step_output = adapter.step(planner_input)
    control = planner_output_to_control(step_output, planner_input)

    report = {
        "joint_set": args.joint_set,
        "joint_names": list(joint_names),
        "warmup_phase": str(warmup_output.phase),
        "step_phase": str(step_output.phase),
        "next_phase": str(step_output.next_phase),
        "planning_time_ms": float(step_output.diagnostics.planning_time_ms),
        "torque_norm": float(step_output.diagnostics.torque_norm),
        "gain_norm": float(step_output.diagnostics.gain_norm),
        "feedback_gain_shape": list(float64_multi_array_to_numpy(control.feedback_gain).shape),
        "feedforward_shape": list(float64_multi_array_to_numpy(control.feedforward).shape),
        "control_initial_state_names": list(control.initial_state.joint_state.name),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
