from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import rclpy
from rclpy.node import Node


ACTIVE_STATE = "active"


def controller_manager_service_name(controller_manager_name: str, service_name: str) -> str:
    manager = (controller_manager_name or "/controller_manager").strip()
    if not manager:
        manager = "/controller_manager"
    return f"{manager.rstrip('/')}/{service_name.lstrip('/')}"


def controller_states_by_name(controllers: Sequence[object]) -> dict[str, str]:
    return {
        str(getattr(controller, "name", "")): str(getattr(controller, "state", ""))
        for controller in controllers
        if str(getattr(controller, "name", ""))
    }


def inactive_required_controllers(
    controllers: Sequence[object],
    required_controllers: Sequence[str],
) -> dict[str, str]:
    states = controller_states_by_name(controllers)
    return {
        name: states.get(name, "missing")
        for name in required_controllers
        if states.get(name) != ACTIVE_STATE
    }


def hardware_state_label(component: object) -> str:
    state = getattr(component, "state", None)
    return str(getattr(state, "label", "")).lower()


def hardware_states_by_name(components: Sequence[object]) -> dict[str, str]:
    return {
        str(getattr(component, "name", "")): hardware_state_label(component)
        for component in components
        if str(getattr(component, "name", ""))
    }


def inactive_required_hardware(
    components: Sequence[object],
    required_hardware: Sequence[str],
) -> dict[str, str]:
    states = hardware_states_by_name(components)
    return {
        name: states.get(name, "missing")
        for name in required_hardware
        if states.get(name) != ACTIVE_STATE
    }


class ControllerStateWatchdog(Node):
    def __init__(
        self,
        *,
        controller_manager_name: str,
        required_controllers: Sequence[str],
        required_hardware: Sequence[str],
    ) -> None:
        super().__init__("sbmpc_controller_state_watchdog")
        from controller_manager_msgs.srv import (
            ListControllers,
            ListHardwareComponents,
        )

        self._required_controllers = tuple(
            name.strip() for name in required_controllers if name.strip()
        )
        self._required_hardware = tuple(
            name.strip() for name in required_hardware if name.strip()
        )
        self._list_controllers = self.create_client(
            ListControllers,
            controller_manager_service_name(
                controller_manager_name,
                "list_controllers",
            ),
        )
        self._list_hardware = self.create_client(
            ListHardwareComponents,
            controller_manager_service_name(
                controller_manager_name,
                "list_hardware_components",
            ),
        )
        self._list_controllers_request = ListControllers.Request()
        self._list_hardware_request = ListHardwareComponents.Request()

    def wait_for_services(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_sec)
        clients = [self._list_controllers]
        if self._required_hardware:
            clients.append(self._list_hardware)
        for client in clients:
            remaining = max(0.0, deadline - time.monotonic())
            if not client.wait_for_service(timeout_sec=remaining):
                self.get_logger().error(
                    f"Timed out waiting for controller_manager service {client.srv_name}"
                )
                return False
        return True

    def _call_service(self, client: object, request: object, timeout_sec: float):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done():
            self.get_logger().error(
                f"Timed out calling controller_manager service {client.srv_name}"
            )
            return None
        try:
            return future.result()
        except Exception as exc:  # pragma: no cover - depends on rclpy transport
            self.get_logger().error(
                f"controller_manager service {client.srv_name} failed: {exc}"
            )
            return None

    def check_once(self, *, timeout_sec: float) -> bool:
        controllers = self._call_service(
            self._list_controllers,
            self._list_controllers_request,
            timeout_sec,
        )
        if controllers is None:
            return False
        inactive_controllers = inactive_required_controllers(
            controllers.controller,
            self._required_controllers,
        )
        if inactive_controllers:
            self.get_logger().error(
                "Required controllers are no longer active: "
                f"{inactive_controllers}"
            )
            return False

        if not self._required_hardware:
            return True
        hardware = self._call_service(
            self._list_hardware,
            self._list_hardware_request,
            timeout_sec,
        )
        if hardware is None:
            return False
        inactive_hardware = inactive_required_hardware(
            hardware.component,
            self._required_hardware,
        )
        if inactive_hardware:
            self.get_logger().error(
                "Required hardware components are no longer active: "
                f"{inactive_hardware}"
            )
            return False
        return True


def monitor_controller_state(
    *,
    controller_manager_name: str,
    required_controllers: Sequence[str],
    required_hardware: Sequence[str],
    period_sec: float,
    service_timeout_sec: float,
) -> bool:
    rclpy.init()
    node = ControllerStateWatchdog(
        controller_manager_name=controller_manager_name,
        required_controllers=required_controllers,
        required_hardware=required_hardware,
    )
    try:
        if not node.wait_for_services(timeout_sec=service_timeout_sec):
            return False
        node.get_logger().info(
            "Monitoring active controllers "
            f"{tuple(required_controllers)} and hardware {tuple(required_hardware)}."
        )
        while rclpy.ok():
            if not node.check_once(timeout_sec=service_timeout_sec):
                return False
            time.sleep(max(0.05, period_sec))
        return True
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-manager", default="/controller_manager")
    parser.add_argument("--required-controller", action="append", default=[])
    parser.add_argument("--required-hardware", action="append", default=[])
    parser.add_argument("--period-sec", type=float, default=0.25)
    parser.add_argument("--service-timeout-sec", type=float, default=2.0)
    args, _ = parser.parse_known_args(argv)

    if not monitor_controller_state(
        controller_manager_name=args.controller_manager,
        required_controllers=args.required_controller,
        required_hardware=args.required_hardware,
        period_sec=args.period_sec,
        service_timeout_sec=args.service_timeout_sec,
    ):
        raise SystemExit("Controller or hardware state watchdog failed.")


if __name__ == "__main__":
    main()
