from __future__ import annotations

import sys
import types

from sbmpc_bringup import warmup_wait


def test_diagnostics_warmup_complete_uses_warmup_count() -> None:
    assert warmup_wait.diagnostics_warmup_complete({"warmup_count": 1}) is True
    assert (
        warmup_wait.diagnostics_warmup_complete(
            {"warmup_count": 2},
            min_warmup_count=2,
        )
        is True
    )


def test_diagnostics_warmup_complete_rejects_missing_or_low_count() -> None:
    assert warmup_wait.diagnostics_warmup_complete({}) is False
    assert (
        warmup_wait.diagnostics_warmup_complete(
            {"warmup_count": 1},
            min_warmup_count=2,
        )
        is False
    )


def test_reset_service_requested_ignores_empty_values() -> None:
    assert warmup_wait.reset_service_requested("") is False
    assert warmup_wait.reset_service_requested("  ") is False
    assert warmup_wait.reset_service_requested("/reset_world") is True


def test_activation_requested_ignores_empty_values() -> None:
    assert warmup_wait.activation_requested([]) is False
    assert warmup_wait.activation_requested(["  "]) is False
    assert warmup_wait.activation_requested(["linear_feedback_controller"]) is True


def test_switch_controller_service_name_uses_manager_namespace() -> None:
    assert (
        warmup_wait.switch_controller_service_name("/controller_manager")
        == "/controller_manager/switch_controller"
    )
    assert (
        warmup_wait.switch_controller_service_name("/controller_manager/")
        == "/controller_manager/switch_controller"
    )


def test_reset_world_calls_mujoco_service(monkeypatch) -> None:
    class Result:
        success = True

    class Future:
        def result(self):
            return Result()

    class Client:
        def __init__(self) -> None:
            self.requests = []

        def wait_for_service(self, *, timeout_sec: float) -> bool:
            assert timeout_sec == 3.0
            return True

        def call_async(self, request):
            self.requests.append(request)
            return Future()

    class Node:
        def __init__(self) -> None:
            self.client = Client()
            self.service_name = ""

        def create_client(self, service_type, service_name: str):
            del service_type
            self.service_name = service_name
            return self.client

    node = Node()
    spun = []
    monkeypatch.setattr(
        warmup_wait.rclpy,
        "spin_until_future_complete",
        lambda node_arg, future, timeout_sec: spun.append(
            (node_arg, future, timeout_sec)
        ),
    )

    assert (
        warmup_wait.reset_world(
            node,
            service_name="/mujoco_ros2_control_node/reset_world",
            keyframe="home",
            timeout_sec=3.0,
        )
        is True
    )

    assert node.service_name == "/mujoco_ros2_control_node/reset_world"
    assert node.client.requests[0].keyframe == "home"
    assert spun and spun[0][0] is node


def test_switch_controllers_calls_controller_manager(monkeypatch) -> None:
    class Request:
        STRICT = 2

        def __init__(self) -> None:
            self.activate_controllers = []
            self.deactivate_controllers = []
            self.strictness = 0
            self.activate_asap = False
            self.timeout = types.SimpleNamespace(sec=0, nanosec=0)

    SwitchController = type("SwitchController", (), {"Request": Request})

    package_module = types.ModuleType("controller_manager_msgs")
    srv_module = types.ModuleType("controller_manager_msgs.srv")
    srv_module.SwitchController = SwitchController
    package_module.srv = srv_module
    monkeypatch.setitem(sys.modules, "controller_manager_msgs", package_module)
    monkeypatch.setitem(sys.modules, "controller_manager_msgs.srv", srv_module)

    class Result:
        ok = True

    class Future:
        def result(self):
            return Result()

    class Client:
        def __init__(self) -> None:
            self.requests = []

        def wait_for_service(self, *, timeout_sec: float) -> bool:
            assert timeout_sec == 4.25
            return True

        def call_async(self, request):
            self.requests.append(request)
            return Future()

    class Node:
        def __init__(self) -> None:
            self.client = Client()
            self.service_name = ""

        def create_client(self, service_type, service_name: str):
            assert service_type is SwitchController
            self.service_name = service_name
            return self.client

    node = Node()
    spun = []
    monkeypatch.setattr(
        warmup_wait.rclpy,
        "spin_until_future_complete",
        lambda node_arg, future, timeout_sec: spun.append(
            (node_arg, future, timeout_sec)
        ),
    )

    assert (
        warmup_wait.switch_controllers(
            node,
            controller_manager_name="/controller_manager",
            activate_controllers=[
                "joint_state_estimator",
                "linear_feedback_controller",
            ],
            timeout_sec=4.25,
        )
        is True
    )

    request = node.client.requests[0]
    assert node.service_name == "/controller_manager/switch_controller"
    assert request.activate_controllers == [
        "joint_state_estimator",
        "linear_feedback_controller",
    ]
    assert request.deactivate_controllers == []
    assert request.strictness == Request.STRICT
    assert request.activate_asap is True
    assert request.timeout.sec == 4
    assert request.timeout.nanosec == 250_000_000
    assert spun and spun[0][0] is node


def test_main_ignores_ros_args_appended_by_launch_node(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        warmup_wait,
        "wait_for_warmup",
        lambda **kwargs: calls.append(kwargs) or True,
    )

    warmup_wait.main(
        [
            "--timeout-sec",
            "1.5",
            "--activate-controller",
            "joint_state_estimator",
            "--activate-controller",
            "linear_feedback_controller",
            "--controller-manager",
            "/controller_manager",
            "--switch-timeout-sec",
            "2.5",
            "--ros-args",
            "--params-file",
            "/tmp/launch_params.yaml",
        ]
    )

    assert calls[0]["timeout_sec"] == 1.5
    assert calls[0]["activate_controllers"] == [
        "joint_state_estimator",
        "linear_feedback_controller",
    ]
    assert calls[0]["controller_manager_name"] == "/controller_manager"
    assert calls[0]["switch_timeout_sec"] == 2.5
