from __future__ import annotations

import argparse
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from sbmpc_bringup.constants import BRIDGE_DIAGNOSTICS_TOPIC


def diagnostics_warmup_complete(
    payload: dict[str, object],
    *,
    min_warmup_count: int = 1,
) -> bool:
    return int(payload.get("warmup_count", 0) or 0) >= min_warmup_count


def reset_service_requested(service_name: str | None) -> bool:
    return bool((service_name or "").strip())


def activation_requested(controller_names: list[str] | tuple[str, ...]) -> bool:
    return any(name.strip() for name in controller_names)


def switch_controller_service_name(controller_manager_name: str) -> str:
    manager = (controller_manager_name or "/controller_manager").strip()
    if not manager:
        manager = "/controller_manager"
    return f"{manager.rstrip('/')}/switch_controller"


class WarmupWaiter(Node):
    def __init__(
        self,
        *,
        diagnostics_topic: str,
        min_warmup_count: int,
    ) -> None:
        super().__init__("sbmpc_bridge_warmup_waiter")
        self._min_warmup_count = max(1, int(min_warmup_count))
        self.complete = False
        self.last_payload: dict[str, object] | None = None
        self.create_subscription(String, diagnostics_topic, self._on_diagnostics, 10)

    def _on_diagnostics(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        self.last_payload = payload
        self.complete = diagnostics_warmup_complete(
            payload,
            min_warmup_count=self._min_warmup_count,
        )


def wait_for_warmup(
    *,
    diagnostics_topic: str,
    timeout_sec: float,
    min_warmup_count: int,
    reset_world_service: str = "",
    reset_keyframe: str = "home",
    reset_timeout_sec: float = 10.0,
    activate_controllers: list[str] | tuple[str, ...] = (),
    controller_manager_name: str = "/controller_manager",
    switch_timeout_sec: float = 10.0,
) -> bool:
    rclpy.init()
    node = WarmupWaiter(
        diagnostics_topic=diagnostics_topic,
        min_warmup_count=min_warmup_count,
    )
    deadline = None if timeout_sec <= 0.0 else time.monotonic() + timeout_sec
    try:
        while rclpy.ok() and not node.complete:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            rclpy.spin_once(node, timeout_sec=0.1)
        if not node.complete:
            return False
        if reset_service_requested(reset_world_service) and not reset_world(
                node,
                service_name=reset_world_service,
                keyframe=reset_keyframe,
                timeout_sec=reset_timeout_sec,
        ):
            return False
        if activation_requested(activate_controllers):
            return switch_controllers(
                node,
                controller_manager_name=controller_manager_name,
                activate_controllers=activate_controllers,
                timeout_sec=switch_timeout_sec,
            )
        return True
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def reset_world(
    node: Node,
    *,
    service_name: str,
    keyframe: str,
    timeout_sec: float,
) -> bool:
    from mujoco_ros2_control_msgs.srv import ResetWorld

    client = node.create_client(ResetWorld, service_name)
    if not client.wait_for_service(timeout_sec=timeout_sec):
        return False

    request = ResetWorld.Request()
    request.keyframe = keyframe
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    result = future.result()
    return bool(result is not None and result.success)


def switch_controllers(
    node: Node,
    *,
    controller_manager_name: str,
    activate_controllers: list[str] | tuple[str, ...],
    timeout_sec: float,
) -> bool:
    from controller_manager_msgs.srv import SwitchController

    controller_names = [name.strip() for name in activate_controllers if name.strip()]
    if not controller_names:
        return True

    client = node.create_client(
        SwitchController,
        switch_controller_service_name(controller_manager_name),
    )
    if not client.wait_for_service(timeout_sec=timeout_sec):
        return False

    request = SwitchController.Request()
    request.activate_controllers = controller_names
    request.deactivate_controllers = []
    request.strictness = SwitchController.Request.STRICT
    request.activate_asap = True
    request.timeout.sec = int(timeout_sec)
    request.timeout.nanosec = int((timeout_sec - int(timeout_sec)) * 1e9)
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    result = future.result()
    return bool(result is not None and result.ok)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-topic", default=BRIDGE_DIAGNOSTICS_TOPIC)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--min-warmup-count", type=int, default=1)
    parser.add_argument("--reset-world-service", default="")
    parser.add_argument("--reset-keyframe", default="home")
    parser.add_argument("--reset-timeout-sec", type=float, default=10.0)
    parser.add_argument("--activate-controller", action="append", default=[])
    parser.add_argument("--controller-manager", default="/controller_manager")
    parser.add_argument("--switch-timeout-sec", type=float, default=10.0)
    args, _ = parser.parse_known_args(argv)

    if not wait_for_warmup(
        diagnostics_topic=args.diagnostics_topic,
        timeout_sec=args.timeout_sec,
        min_warmup_count=args.min_warmup_count,
        reset_world_service=args.reset_world_service,
        reset_keyframe=args.reset_keyframe,
        reset_timeout_sec=args.reset_timeout_sec,
        activate_controllers=args.activate_controller,
        controller_manager_name=args.controller_manager,
        switch_timeout_sec=args.switch_timeout_sec,
    ):
        raise SystemExit(
            "Timed out waiting for SB-MPC bridge warmup, MuJoCo reset, "
            "or controller activation."
        )


if __name__ == "__main__":
    main()
