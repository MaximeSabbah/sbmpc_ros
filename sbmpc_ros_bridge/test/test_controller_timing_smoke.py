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


def test_controller_planning_time_with_synthetic_sensor() -> None:
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
    from sbmpc_ros_bridge.planner_smoke import FER_ARM_JOINT_NAMES, build_sensor

    steps = read_env_int("SBMPC_CONTROLLER_TIMING_STEPS", 30)
    # Budget = the deployed 25 Hz control period; the MPPI knobs come from the
    # pregrasp OCP yaml exactly as in deployment.
    max_exact_ms = read_env_float("SBMPC_CONTROLLER_TIMING_MAX_EXACT_MS", 40.0)
    if steps <= 0:
        raise ValueError("SBMPC_CONTROLLER_TIMING_STEPS must be positive")

    adapter = SbMpcPlannerAdapter(
        config_overrides=PlannerConfigOverrides(
            mode="exact_feedback",
            ocp="pregrasp",
        )
    )
    try:
        home_q = np.asarray(adapter._controller.planner.home_q, dtype=np.float64)
        planner_input = sensor_to_planner_input(
            build_sensor(
                joint_names=FER_ARM_JOINT_NAMES,
                q=home_q,
                v=np.zeros_like(home_q),
            ),
            joint_mapper=JointMapper(expected_names=FER_ARM_JOINT_NAMES),
        )

        for _ in range(3):
            adapter.warmup()
        wall_ms: list[float] = []
        output = None
        for _ in range(steps):
            start = time.perf_counter()
            output = adapter.step(planner_input)
            wall_ms.append(1000.0 * (time.perf_counter() - start))
            assert np.all(np.isfinite(np.asarray(output.tau_ff)))
            assert np.all(np.isfinite(np.asarray(output.K)))

        assert output is not None
        control = planner_output_to_control(output, planner_input)
        assert np.all(np.isfinite(float64_multi_array_to_numpy(control.feedforward)))
        assert np.all(np.isfinite(float64_multi_array_to_numpy(control.feedback_gain)))

        steady = np.asarray(wall_ms[-min(20, len(wall_ms)):], dtype=np.float64)
        p95_ms = float(np.quantile(steady, 0.95))
        context = f"exact_wall_ms={steady.tolist()}, p95_ms={p95_ms:.3f}, max_ms={max_exact_ms:.3f}"
        print(context)
        assert p95_ms <= max_exact_ms, context
    finally:
        adapter.close()
