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
from sbmpc_ros_bridge.planner_adapter import PlannerConfigOverrides, SbMpcPlannerAdapter


FR3_ARM_JOINT_NAMES: tuple[str, ...] = tuple(f"fr3_joint{i}" for i in range(1, 8))
FER_ARM_JOINT_NAMES: tuple[str, ...] = tuple(f"fer_joint{i}" for i in range(1, 8))
JOINT_NAME_SETS: dict[str, tuple[str, ...]] = {
    "fer": FER_ARM_JOINT_NAMES,
    "fr3": FR3_ARM_JOINT_NAMES,
    "panda": PANDA_ARM_JOINT_NAMES,
}
PLANNER_MODES: tuple[str, ...] = (
    "feedforward",
    "exact_async_feedback",
)


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
    parser.add_argument(
        "--planner-mode",
        choices=PLANNER_MODES,
        default="exact_async_feedback",
        help="Planner mode to smoke-test through the ROS adapter.",
    )
    args = parser.parse_args(argv)

    config_overrides = PlannerConfigOverrides(
        mode=args.planner_mode,
        gain_samples_per_cycle=(
            128 if args.planner_mode != "feedforward" else None
        ),
        gain_buffer_size=(
            512 if args.planner_mode != "feedforward" else None
        ),
    )
    adapter = SbMpcPlannerAdapter(config_overrides=config_overrides)
    joint_names = JOINT_NAME_SETS[args.joint_set]
    try:
        home_q = np.asarray(adapter._controller.planner.home_q, dtype=np.float64)
        zero_v = np.zeros_like(home_q)

        planner_input = sensor_to_planner_input(
            build_sensor(joint_names=joint_names, q=home_q, v=zero_v),
            joint_mapper=JointMapper(expected_names=joint_names),
        )
        warmup_output = adapter.warmup()
        step_output = adapter.step(planner_input)
        control = planner_output_to_control(step_output, planner_input)
        diagnostics = step_output.diagnostics

        report = {
            "joint_set": args.joint_set,
            "joint_names": list(joint_names),
            "planner_mode": str(diagnostics.gain_mode),
            "warmup_phase": str(warmup_output.phase),
            "step_phase": str(step_output.phase),
            "next_phase": str(step_output.next_phase),
            "planning_time_ms": float(diagnostics.planning_time_ms),
            "foreground_planning_time_ms": diagnostics.foreground_planning_time_ms,
            "background_gain_time_ms": diagnostics.background_gain_time_ms,
            "gain_worker_running": diagnostics.async_gain_worker_running,
            "gain_window_fill": diagnostics.gain_window_fill,
            "gain_completed_batch_count": diagnostics.gain_completed_batch_count,
            "gain_dropped_snapshot_count": diagnostics.gain_dropped_snapshot_count,
            "gain_worker_error": diagnostics.async_gain_worker_error,
            "torque_norm": float(diagnostics.torque_norm),
            "gain_norm": float(diagnostics.gain_norm),
            "feedback_gain_shape": list(
                float64_multi_array_to_numpy(control.feedback_gain).shape
            ),
            "feedforward_shape": list(
                float64_multi_array_to_numpy(control.feedforward).shape
            ),
            "control_initial_state_names": list(control.initial_state.joint_state.name),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
