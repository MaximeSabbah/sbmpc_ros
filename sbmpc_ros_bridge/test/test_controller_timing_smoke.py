from __future__ import annotations

import os
import time
from types import SimpleNamespace

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
    target_feedforward_ms = read_env_float(
        "SBMPC_CONTROLLER_TIMING_TARGET_FEEDFORWARD_MS",
        10.0,
    )
    max_feedforward_ms = read_env_float(
        "SBMPC_CONTROLLER_TIMING_MAX_FEEDFORWARD_MS",
        12.0,
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
        output = None
        for _ in range(steps):
            start = time.perf_counter()
            output = adapter.step_feedforward(planner_input)
            elapsed_ms = 1000.0 * (time.perf_counter() - start)
            wall_ms.append(elapsed_ms)
            foreground_ms.append(float(output.diagnostics.foreground_planning_time_ms))
            assert np.all(np.isfinite(np.asarray(output.tau_ff)))
            adapter.refresh_gain_if_budget(
                output.diagnostics,
                budget_sec=max(0.0, 0.020 - 0.001 - 1e-3 * elapsed_ms),
            )

        assert output is not None
        gain_snapshot = adapter.latest_gain(output.diagnostics)
        assert gain_snapshot is not None
        assert gain_snapshot.diagnostics.async_gain_worker_error is None
        assert gain_snapshot.diagnostics.async_gain_worker_running is False
        assert gain_snapshot.diagnostics.gain_completed_batch_count >= 0

        composed = SimpleNamespace(tau_ff=output.tau_ff, K=gain_snapshot.K)
        control = planner_output_to_control(composed, planner_input)
        assert np.all(np.isfinite(float64_multi_array_to_numpy(control.feedforward)))
        assert np.all(np.isfinite(float64_multi_array_to_numpy(control.feedback_gain)))

        foreground = np.asarray(foreground_ms, dtype=np.float64)
        wall = np.asarray(wall_ms, dtype=np.float64)
        p99_ms = float(np.quantile(foreground, 0.99))
        context = (
            f"feedforward_foreground_ms={foreground.tolist()}, wall_ms={wall.tolist()}, "
            f"p99_ms={p99_ms:.3f}, target_ms={target_feedforward_ms:.3f}, "
            f"max_ms={max_feedforward_ms:.3f}, "
            f"gain_age={gain_snapshot.diagnostics.gain_age_cycles!r}, "
            f"completed_batches={gain_snapshot.diagnostics.gain_completed_batch_count!r}, "
            f"dropped_snapshots={gain_snapshot.diagnostics.gain_dropped_snapshot_count!r}"
        )
        print(context)
        assert p99_ms <= max_feedforward_ms, context
    finally:
        adapter.close()
