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
SBMPC_JOINT_STATES_TOPIC = "/joint_states"
SBMPC_DYNAMIC_JOINT_STATES_TOPIC = "/sbmpc/dynamic_joint_states"
LFC_OUTPUT_JOINT_EFFORT_TOPIC = "/output_joint_effort"

# Node / service names shared by the unified bringup launch and its helper nodes.
CONTROLLER_MANAGER_NAME = "/controller_manager"
BRIDGE_NODE_NAME = "/sbmpc_lfc_bridge_node"
RESET_WORLD_SERVICE = "/mujoco_ros2_control_node/reset_world"

# MuJoCo keyframe the hardware interface initializes to and that the warmup step
# resets the world to before arming.
DEFAULT_KEYFRAME = "home"

# Controller-spawner switch timeout (seconds). Bounds LFC activation only, which
# involves no JIT and must fail fast. The warmup wait, by contrast, is unbounded
# (the launch passes --timeout-sec 0) because planner JIT can exceed minutes.
CONTROLLER_SWITCH_TIMEOUT_SEC = 60


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
