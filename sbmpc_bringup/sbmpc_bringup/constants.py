from __future__ import annotations


FER_ARM_JOINT_NAMES: tuple[str, ...] = tuple(f"fer_joint{i}" for i in range(1, 8))

JOINT_STATE_BROADCASTER_NAME = "joint_state_broadcaster"
JOINT_STATE_ESTIMATOR_NAME = "joint_state_estimator"
LINEAR_FEEDBACK_CONTROLLER_NAME = "linear_feedback_controller"
GRIPPER_ACTION_CONTROLLER_NAME = "gripper_action_controller"
FER_GRIPPER_JOINT_NAME = "fer_finger_joint1"

BRIDGE_SENSOR_TOPIC = "/sensor"
BRIDGE_CONTROL_TOPIC = "/control"
BRIDGE_DIAGNOSTICS_TOPIC = "/sbmpc/diagnostics"


def hardware_state_interfaces(
    joint_names: tuple[str, ...] = FER_ARM_JOINT_NAMES,
) -> tuple[str, ...]:
    interfaces: list[str] = []
    for interface_name in ("position", "velocity", "effort"):
        interfaces.extend(f"{joint_name}/{interface_name}" for joint_name in joint_names)
    return tuple(interfaces)


def effort_command_interfaces(
    joint_names: tuple[str, ...] = FER_ARM_JOINT_NAMES,
) -> tuple[str, ...]:
    return tuple(f"{joint_name}/effort" for joint_name in joint_names)


def lfc_reference_interfaces(
    controller_name: str = LINEAR_FEEDBACK_CONTROLLER_NAME,
    joint_names: tuple[str, ...] = FER_ARM_JOINT_NAMES,
) -> tuple[str, ...]:
    interfaces: list[str] = []
    for interface_name in ("position", "velocity", "effort"):
        interfaces.extend(
            f"{controller_name}/{joint_name}/{interface_name}"
            for joint_name in joint_names
        )
    return tuple(interfaces)
