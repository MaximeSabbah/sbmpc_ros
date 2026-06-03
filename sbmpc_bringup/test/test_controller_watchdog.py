from __future__ import annotations

from types import SimpleNamespace

from sbmpc_bringup import controller_watchdog


def controller(name: str, state: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, state=state)


def hardware(name: str, label: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, state=SimpleNamespace(label=label))


def test_controller_manager_service_name_uses_manager_namespace() -> None:
    assert (
        controller_watchdog.controller_manager_service_name(
            "/controller_manager",
            "list_controllers",
        )
        == "/controller_manager/list_controllers"
    )
    assert (
        controller_watchdog.controller_manager_service_name(
            "/controller_manager/",
            "/list_hardware_components",
        )
        == "/controller_manager/list_hardware_components"
    )


def test_inactive_required_controllers_reports_missing_or_inactive() -> None:
    controllers = [
        controller("joint_state_estimator", "active"),
        controller("linear_feedback_controller", "inactive"),
    ]

    assert controller_watchdog.inactive_required_controllers(
        controllers,
        [
            "joint_state_estimator",
            "linear_feedback_controller",
            "missing_controller",
        ],
    ) == {
        "linear_feedback_controller": "inactive",
        "missing_controller": "missing",
    }


def test_inactive_required_hardware_reports_missing_or_inactive() -> None:
    components = [
        hardware("AgimusFrankaHardwareInterface", "active"),
        hardware("AuxHardware", "inactive"),
    ]

    assert controller_watchdog.inactive_required_hardware(
        components,
        [
            "AgimusFrankaHardwareInterface",
            "AuxHardware",
            "missing_hardware",
        ],
    ) == {
        "AuxHardware": "inactive",
        "missing_hardware": "missing",
    }
