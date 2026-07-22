"""Canonical topic contract for low-overhead sim and robot run capture."""

from __future__ import annotations

from sbmpc_bringup.constants import (
    BRIDGE_CONTROL_TOPIC,
    BRIDGE_DIAGNOSTICS_TOPIC,
    BRIDGE_SENSOR_TOPIC,
    LFC_OUTPUT_JOINT_EFFORT_TOPIC,
    SBMPC_JOINT_STATES_TOPIC,
)


CONTROL_TOPIC = BRIDGE_CONTROL_TOPIC
SENSOR_TOPIC = BRIDGE_SENSOR_TOPIC
OUTPUT_TOPIC = LFC_OUTPUT_JOINT_EFFORT_TOPIC
MERGED_JOINT_TOPIC = SBMPC_JOINT_STATES_TOPIC
PLANNER_DIAGNOSTICS_TOPIC = BRIDGE_DIAGNOSTICS_TOPIC

HARDWARE_JOINT_TOPIC = "/franka/joint_states"
ROSOUT_TOPIC = "/rosout"
ROS_CLOCK_TOPIC = "/clock"
SIM_OBJECT_POSE_TOPIC = "/simulator/object_pose"
SIM_ACTUATOR_STATE_TOPIC = "/mujoco_actuators_states"

ROBOT_STATE_TOPIC = "/franka_robot_state_broadcaster/robot_state"
DESIRED_STATE_TOPIC = "/franka_robot_state_broadcaster/desired_joint_states"
ROS_DIAGNOSTICS_TOPIC = "/diagnostics"
CONTROLLER_ACTIVITY_TOPIC = "/controller_manager/activity"

GRIPPER_JOINT_TOPIC = "/fer_gripper/joint_states"
GRIPPER_STATUS_TOPIC = "/fer_gripper/gripper_action/_action/status"
GRIPPER_FEEDBACK_TOPIC = "/fer_gripper/gripper_action/_action/feedback"
SIM_GRIPPER_STATUS_TOPIC = (
    "/gripper_action_controller/gripper_cmd/_action/status"
)
SIM_GRIPPER_FEEDBACK_TOPIC = (
    "/gripper_action_controller/gripper_cmd/_action/feedback"
)

# The 1 kHz streams below are already required to reconstruct the deployed LFC
# command path.  The complete Agimus/FCI state is recorded once, canonically,
# at the broadcaster's validated 100 Hz rate.  The remaining streams are
# low-rate diagnostics or gripper observability. Redundant Agimus convenience
# topics and the high-volume full statistics stream are intentionally omitted.
# Topics absent on one backend are harmless: rosbag discovery records those
# which appear and the report labels backend-specific availability explicitly.
RUN_TOPICS = (
    CONTROL_TOPIC,
    SENSOR_TOPIC,
    OUTPUT_TOPIC,
    HARDWARE_JOINT_TOPIC,
    MERGED_JOINT_TOPIC,
    PLANNER_DIAGNOSTICS_TOPIC,
    ROBOT_STATE_TOPIC,
    ROS_DIAGNOSTICS_TOPIC,
    CONTROLLER_ACTIVITY_TOPIC,
    ROSOUT_TOPIC,
    GRIPPER_JOINT_TOPIC,
    GRIPPER_STATUS_TOPIC,
    GRIPPER_FEEDBACK_TOPIC,
    SIM_GRIPPER_STATUS_TOPIC,
    SIM_GRIPPER_FEEDBACK_TOPIC,
    ROS_CLOCK_TOPIC,
    SIM_OBJECT_POSE_TOPIC,
    SIM_ACTUATOR_STATE_TOPIC,
)
