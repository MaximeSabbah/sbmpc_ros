from __future__ import annotations

import os
import time

import numpy as np
import pytest


def read_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


def read_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def test_controller_foreground_timing_with_synthetic_sensor() -> None:
    if os.environ.get("SBMPC_RUN_CONTROLLER_TIMING") != "1":
        pytest.skip(
            "set SBMPC_RUN_CONTROLLER_TIMING=1 to run the controller timing smoke"
        )

    from sbmpc_ros_bridge.joint_mapping import JointMapper
    from sbmpc_ros_bridge.lfc_msg_adapter import (
        float64_multi_array_to_numpy,
        planner_output_to_control,
        sensor_to_planner_input,
    )
    from sbmpc_ros_bridge.planner_adapter import (
        PlannerConfigOverrides,
        SbMpcPlannerAdapter,
    )
    from sbmpc_ros_bridge.planner_smoke import (
        FER_ARM_JOINT_NAMES,
        build_sensor,
    )

    steps = read_env_int("SBMPC_CONTROLLER_TIMING_STEPS", 30)
    max_foreground_ms = read_env_float(
        "SBMPC_CONTROLLER_TIMING_MAX_FOREGROUND_MS",
        21.0,
    )
    if steps <= 0:
        raise ValueError("SBMPC_CONTROLLER_TIMING_STEPS must be positive")

    adapter = SbMpcPlannerAdapter(
        config_overrides=PlannerConfigOverrides(
            mode="exact_async_feedback",
            gain_samples_per_cycle=128,
            gain_buffer_size=512,
        )
    )
    try:
        home_q = np.asarray(adapter._controller.planner.home_q, dtype=np.float64)
        zero_v = np.zeros_like(home_q)
        planner_input = sensor_to_planner_input(
            build_sensor(
                joint_names=FER_ARM_JOINT_NAMES,
                q=home_q,
                v=zero_v,
            ),
            joint_mapper=JointMapper(expected_names=FER_ARM_JOINT_NAMES),
        )

        adapter.warmup()
        foreground_ms: list[float] = []
        wall_ms: list[float] = []
        diagnostics = None
        for _ in range(steps):
            start = time.perf_counter()
            output = adapter.step(planner_input)
            wall_ms.append(1000.0 * (time.perf_counter() - start))
            diagnostics = output.diagnostics
            foreground_ms.append(float(diagnostics.foreground_planning_time_ms))

            control = planner_output_to_control(output, planner_input)
            assert np.all(np.isfinite(float64_multi_array_to_numpy(control.feedforward)))
            assert np.all(
                np.isfinite(float64_multi_array_to_numpy(control.feedback_gain))
            )

        assert diagnostics is not None
        assert diagnostics.async_gain_worker_error is None
        assert diagnostics.async_gain_worker_running is True
        assert diagnostics.gain_completed_batch_count > 0

        foreground = np.asarray(foreground_ms, dtype=np.float64)
        wall = np.asarray(wall_ms, dtype=np.float64)
        context = (
            f"foreground_ms={foreground.tolist()}, wall_ms={wall.tolist()}, "
            f"gain_age={diagnostics.gain_age_cycles!r}, "
            f"completed_batches={diagnostics.gain_completed_batch_count!r}, "
            f"dropped_snapshots={diagnostics.gain_dropped_snapshot_count!r}"
        )
        assert float(np.quantile(foreground, 0.99)) <= max_foreground_ms, context
    finally:
        adapter.close()
