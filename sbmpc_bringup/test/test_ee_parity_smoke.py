from __future__ import annotations

import os
import signal
import subprocess
import time

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
from rclpy.node import Node

from linear_feedback_controller_msgs.msg import Control
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from sbmpc_bringup.constants import (
    BRIDGE_CONTROL_TOPIC,
    BRIDGE_DIAGNOSTICS_TOPIC,
    SBMPC_JOINT_STATES_TOPIC,
)
from sbmpc_bringup.validate_sim import (
    JointRecord,
    assert_stable,
    joint_indices,
    summarize,
    vector_from_indices,
)


def stamp_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


class ParityCollector(Node):
    def __init__(self) -> None:
        super().__init__("sbmpc_mujoco_parity_collector")
        self.diagnostics: list[dict[str, object]] = []
        self.control_stamps: list[float] = []
        self.joint_records: list[JointRecord] = []
        self.create_subscription(String, BRIDGE_DIAGNOSTICS_TOPIC, self._on_diagnostics, 10)
        self.create_subscription(Control, BRIDGE_CONTROL_TOPIC, self._on_control, 10)
        self.create_subscription(JointState, SBMPC_JOINT_STATES_TOPIC, self._on_joint_state, 10)

    def _on_diagnostics(self, message: String) -> None:
        import json

        try:
            self.diagnostics.append(json.loads(message.data))
        except json.JSONDecodeError:
            return

    def _on_control(self, message: Control) -> None:
        self.control_stamps.append(stamp_sec(message.header.stamp))

    def _on_joint_state(self, message: JointState) -> None:
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


def collect_for(duration_sec: float) -> ParityCollector:
    rclpy.init()
    node = ParityCollector()
    deadline = time.monotonic() + duration_sec
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
        return node
    finally:
        node.destroy_node()
        rclpy.shutdown()


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
    )
    try:
        time.sleep(10.0)
        assert process.poll() is None
        collector = collect_for(5.0)
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5.0)

    control_deltas = np.diff(np.asarray(collector.control_stamps, dtype=np.float64))
    assert control_deltas.size >= 100
    assert float(np.quantile(control_deltas, 0.99)) <= 0.020

    summary = summarize(collector.diagnostics, collector.joint_records, tail_fraction=0.5)
    assert summary.planner_mode == "exact_async_feedback"
    assert assert_stable(
        summary,
        max_tail_joint_span=0.02,
        max_final_position_error=0.001,
        max_foreground_ms=20.0,
    )
