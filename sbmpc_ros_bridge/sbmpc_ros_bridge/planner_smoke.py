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
VECTOR_SIZE = 7


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


def parse_vector(value: str | None, *, name: str, default: np.ndarray) -> np.ndarray:
    if value is None or not value.strip():
        return np.asarray(default, dtype=np.float64)
    cleaned = value.strip()
    if cleaned.startswith("["):
        parsed = json.loads(cleaned)
    else:
        parsed = [part.strip() for part in cleaned.split(",") if part.strip()]
    vector = np.asarray(parsed, dtype=np.float64).reshape(-1)
    if vector.shape != (VECTOR_SIZE,):
        raise ValueError(f"{name} must contain {VECTOR_SIZE} values, got {vector.shape}.")
    return vector


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
    parser.add_argument("--q", help="Comma-separated or JSON list of 7 joint positions.")
    parser.add_argument("--v", help="Comma-separated or JSON list of 7 joint velocities.")
    parser.add_argument("--planner-horizon", type=int, default=8)
    parser.add_argument("--planner-dt", type=float, default=0.025)
    parser.add_argument("--planner-num-samples", type=int, default=1024)
    parser.add_argument("--planner-noise-scale", type=float, default=1.0)
    parser.add_argument("--planner-temperature", type=float, default=0.05)
    parser.add_argument("--planner-gain-samples-per-cycle", type=int, default=64)
    parser.add_argument("--planner-gain-buffer-size", type=int, default=512)
    args = parser.parse_args(argv)

    config_overrides = PlannerConfigOverrides(
        mode=args.planner_mode,
        horizon=args.planner_horizon,
        dt=args.planner_dt,
        num_parallel_computations=args.planner_num_samples,
        std_dev_scale=args.planner_noise_scale,
        lambda_mpc=args.planner_temperature,
        gain_samples_per_cycle=(
            args.planner_gain_samples_per_cycle
            if args.planner_mode != "feedforward"
            else None
        ),
        gain_buffer_size=(
            args.planner_gain_buffer_size
            if args.planner_mode != "feedforward"
            else None
        ),
    )
    adapter = SbMpcPlannerAdapter(config_overrides=config_overrides)
    joint_names = JOINT_NAME_SETS[args.joint_set]
    try:
        home_q = np.asarray(adapter._controller.planner.home_q, dtype=np.float64)
        q = parse_vector(args.q, name="--q", default=home_q)
        v = parse_vector(args.v, name="--v", default=np.zeros_like(home_q))

        planner_input = sensor_to_planner_input(
            build_sensor(joint_names=joint_names, q=q, v=v),
            joint_mapper=JointMapper(expected_names=joint_names),
        )
        warmup_output = adapter.warmup()
        step_output = adapter.step(planner_input)
        control = planner_output_to_control(step_output, planner_input)
        diagnostics = step_output.diagnostics
        feedforward = np.asarray(step_output.tau_ff, dtype=np.float64).reshape(-1)
        predicted = adapter.predict_state(planner_input, feedforward, args.planner_dt)
        if predicted is None:
            predicted_q = q
            predicted_v = v
        else:
            predicted_q, predicted_v = predicted

        report = {
            "joint_set": args.joint_set,
            "joint_names": list(joint_names),
            "planner_mode": str(diagnostics.gain_mode),
            "q": q.tolist(),
            "v": v.tolist(),
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
            "max_abs_feedforward": float(np.max(np.abs(feedforward), initial=0.0)),
            "feedforward": feedforward.tolist(),
            "predicted_q_delta_norm": float(np.linalg.norm(np.asarray(predicted_q) - q)),
            "predicted_v_norm": float(np.linalg.norm(predicted_v)),
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
