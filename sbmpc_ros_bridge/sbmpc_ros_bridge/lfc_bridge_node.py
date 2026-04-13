from __future__ import annotations

import rclpy
from linear_feedback_controller_msgs.msg import Control, Sensor
from rclpy.node import Node

from sbmpc_ros_bridge.joint_mapping import JointMapper
from sbmpc_ros_bridge.lfc_msg_adapter import sensor_to_planner_input


class SbMpcLfcBridgeNode(Node):
    """Milestone 1 skeleton for the future SB-MPC to LFC runtime bridge."""

    def __init__(self) -> None:
        super().__init__("sbmpc_lfc_bridge_node")
        self.declare_parameter("sensor_topic", "sensor")
        self.declare_parameter("control_topic", "control")
        self.declare_parameter("allow_joint_reordering", False)
        self.declare_parameter("joint_names", list(JointMapper.panda().expected_names))

        joint_names = tuple(
            self.get_parameter("joint_names").get_parameter_value().string_array_value
        )
        self._joint_mapper = JointMapper(expected_names=joint_names)
        self._last_planner_input = None

        sensor_topic = (
            self.get_parameter("sensor_topic").get_parameter_value().string_value
        )
        control_topic = (
            self.get_parameter("control_topic").get_parameter_value().string_value
        )
        self._control_publisher = self.create_publisher(Control, control_topic, 10)
        self._sensor_subscription = self.create_subscription(
            Sensor,
            sensor_topic,
            self._on_sensor,
            10,
        )

        self.get_logger().info(
            "Milestone 1 skeleton active: LFC sensor validation is wired, but "
            "the planner loop is intentionally left for later milestones."
        )

    def _on_sensor(self, message: Sensor) -> None:
        allow_reordering = (
            self.get_parameter("allow_joint_reordering")
            .get_parameter_value()
            .bool_value
        )
        try:
            self._last_planner_input = sensor_to_planner_input(
                message,
                joint_mapper=self._joint_mapper,
                allow_reordering=allow_reordering,
            )
        except Exception as exc:
            self.get_logger().error(f"Rejected sensor message: {exc}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SbMpcLfcBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
