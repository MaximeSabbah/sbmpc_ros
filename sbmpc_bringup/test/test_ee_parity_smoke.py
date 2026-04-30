from __future__ import annotations

from collections import deque
import os
import re
import signal
import subprocess
import threading
import time

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from linear_feedback_controller_msgs.msg import Control, Sensor
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from sbmpc_bringup.constants import (
    BRIDGE_CONTROL_TOPIC,
    BRIDGE_DIAGNOSTICS_TOPIC,
    BRIDGE_SENSOR_TOPIC,
    SBMPC_JOINT_STATES_TOPIC,
)
from sbmpc_bringup.validate_sim import (
    JointRecord,
    assert_stable,
    joint_indices,
    summarize,
    vector_from_indices,
)


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
LFC_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)
LAUNCH_LOG_TAIL_LINES = 160
OBSERVATION_SEC = 8.0
STARTUP_TIMEOUT_SEC = 90.0


def stamp_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def read_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


def launch_log_tail_text(lines: deque[str]) -> str:
    if not lines:
        return "<no launch output captured>"
    return "\n".join(ANSI_ESCAPE.sub("", line) for line in lines)


def latest_diagnostics_text(diagnostics: list[dict[str, object]]) -> str:
    if not diagnostics:
        return "<no diagnostics received>"
    keys = (
        "state",
        "control_enabled",
        "valid_sensor_count",
        "rejected_sensor_count",
        "published_control_count",
        "planner_step_count",
        "deadline_miss_count",
        "last_foreground_planning_time_ms",
        "last_bridge_loop_time_ms",
        "last_error",
    )
    latest = diagnostics[-1]
    return ", ".join(f"{key}={latest.get(key)!r}" for key in keys)


def capture_launch_tail(stream) -> tuple[deque[str], threading.Thread]:
    lines: deque[str] = deque(maxlen=LAUNCH_LOG_TAIL_LINES)

    def read_lines() -> None:
        for line in iter(stream.readline, ""):
            lines.append(line.rstrip())

    thread = threading.Thread(target=read_lines, daemon=True)
    thread.start()
    return lines, thread


class ParityCollector(Node):
    def __init__(self) -> None:
        super().__init__("sbmpc_mujoco_parity_collector")
        self.diagnostics: list[dict[str, object]] = []
        self.control_stamps: list[float] = []
        self.joint_records: list[JointRecord] = []
        self.create_subscription(String, BRIDGE_DIAGNOSTICS_TOPIC, self._on_diagnostics, 10)
        self.create_subscription(
            Control,
            BRIDGE_CONTROL_TOPIC,
            self._on_control,
            LFC_QOS,
        )
        self.create_subscription(Sensor, BRIDGE_SENSOR_TOPIC, self._on_sensor, LFC_QOS)
        self.create_subscription(
            JointState,
            SBMPC_JOINT_STATES_TOPIC,
            self._on_joint_state,
            10,
        )

    def _on_diagnostics(self, message: String) -> None:
        import json

        try:
            self.diagnostics.append(json.loads(message.data))
        except json.JSONDecodeError:
            return

    def _on_control(self, message: Control) -> None:
        self.control_stamps.append(stamp_sec(message.header.stamp))

    def _on_sensor(self, message: Sensor) -> None:
        self._record_joint_state(message.joint_state)

    def _on_joint_state(self, message: JointState) -> None:
        self._record_joint_state(message)

    def _record_joint_state(self, message: JointState) -> None:
        indices = joint_indices(message.name)
        if indices is None:
            return
        velocity = (
            vector_from_indices(message.velocity, indices)
            if len(message.velocity) >= len(message.name)
            else np.zeros(len(indices), dtype=np.float64)
        )
        self.joint_records.append(
            JointRecord(
                stamp_sec=stamp_sec(message.header.stamp),
                position=vector_from_indices(message.position, indices),
                velocity=velocity,
            )
        )


def collect_after_first_control(
    *,
    process: subprocess.Popen,
    launch_tail: deque[str],
    startup_timeout_sec: float,
    observation_sec: float,
) -> ParityCollector:
    rclpy.init()
    node = ParityCollector()
    startup_deadline = time.monotonic() + startup_timeout_sec
    try:
        while not node.control_stamps and time.monotonic() < startup_deadline:
            if process.poll() is not None:
                raise AssertionError(
                    f"launch exited before publishing {BRIDGE_CONTROL_TOPIC}; "
                    f"exit={process.returncode}\n{launch_log_tail_text(launch_tail)}"
                )
            rclpy.spin_once(node, timeout_sec=0.05)

        if not node.control_stamps:
            raise AssertionError(
                f"no {BRIDGE_CONTROL_TOPIC} samples after "
                f"{startup_timeout_sec:.1f}s startup timeout\n"
                f"collector: diagnostics={len(node.diagnostics)} "
                f"joint_records={len(node.joint_records)} "
                f"last_diagnostics={latest_diagnostics_text(node.diagnostics)}\n"
                f"{launch_log_tail_text(launch_tail)}"
            )

        deadline = time.monotonic() + observation_sec
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(
                    f"launch exited during observation; exit={process.returncode}\n"
                    f"{launch_log_tail_text(launch_tail)}"
                )
            rclpy.spin_once(node, timeout_sec=0.02)

        return node
    finally:
        node.destroy_node()
        rclpy.shutdown()


def cadence_failure_context(
    *,
    control_deltas: np.ndarray,
    diagnostics: list[dict[str, object]],
) -> str:
    return (
        f"control_samples={control_deltas.size + 1}, "
        f"p50={float(np.quantile(control_deltas, 0.50)):.6f}s, "
        f"p95={float(np.quantile(control_deltas, 0.95)):.6f}s, "
        f"p99={float(np.quantile(control_deltas, 0.99)):.6f}s, "
        f"max={float(np.max(control_deltas)):.6f}s, "
        f"last_diagnostics={latest_diagnostics_text(diagnostics)}"
    )


def test_mujoco_pregrasp_parity_smoke() -> None:
    if os.environ.get("SBMPC_RUN_MUJOCO_PARITY") != "1":
        pytest.skip("set SBMPC_RUN_MUJOCO_PARITY=1 to run the live MuJoCo parity smoke")

    process = subprocess.Popen(
        [
            "ros2",
            "launch",
            "sbmpc_bringup",
            "sbmpc_franka_lfc_mujoco_sim.launch.py",
            "headless:=true",
            "enable_nonzero_control:=true",
        ],
        cwd="/workspace/ros2_ws",
        preexec_fn=os.setsid,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    launch_tail, launch_reader = capture_launch_tail(process.stdout)
    try:
        collector = collect_after_first_control(
            process=process,
            launch_tail=launch_tail,
            startup_timeout_sec=read_env_float(
                "SBMPC_MUJOCO_STARTUP_TIMEOUT_SEC",
                STARTUP_TIMEOUT_SEC,
            ),
            observation_sec=read_env_float(
                "SBMPC_MUJOCO_OBSERVATION_SEC",
                OBSERVATION_SEC,
            ),
        )
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5.0)
        launch_reader.join(timeout=1.0)

    control_deltas = np.diff(np.asarray(collector.control_stamps, dtype=np.float64))
    assert control_deltas.size >= 100, launch_log_tail_text(launch_tail)
    cadence_context = cadence_failure_context(
        control_deltas=control_deltas,
        diagnostics=collector.diagnostics,
    )
    assert float(np.quantile(control_deltas, 0.99)) <= 0.020, cadence_context

    summary = summarize(
        collector.diagnostics,
        collector.joint_records,
        tail_fraction=0.5,
    )
    assert summary.planner_mode == "exact_async_feedback"
    assert assert_stable(
        summary,
        max_tail_joint_span=0.02,
        max_final_position_error=0.001,
        max_foreground_ms=20.0,
    )
