from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from sbmpc_bringup.constants import FER_ARM_JOINT_NAMES


xacro = pytest.importorskip("xacro")

PACKAGE_DIR = Path(__file__).resolve().parents[1]
XACRO_FILE = PACKAGE_DIR / "urdf" / "franka_arm_with_sbmpc_real.urdf.xacro"


def render_real_urdf(**mappings: str) -> ET.Element:
    document = xacro.process_file(
        str(XACRO_FILE),
        mappings={"robot_ip": "172.17.1.2", **mappings},
    )
    return ET.fromstring(document.toxml())


def ros2_control(root: ET.Element) -> ET.Element:
    controls = root.findall("ros2_control")
    assert len(controls) == 1
    return controls[0]


def params_by_name(element: ET.Element) -> dict[str, str]:
    return {
        param.attrib["name"]: (param.text or "").strip()
        for param in element.findall(".//param")
    }


def interface_names(joint: ET.Element, tag: str) -> tuple[str, ...]:
    return tuple(child.attrib["name"] for child in joint.findall(tag))


def test_real_xacro_robot_name_matches_agimus_arm_id() -> None:
    root = render_real_urdf()
    hardware = ros2_control(root).find("hardware")
    assert hardware is not None

    params = params_by_name(hardware)
    assert root.attrib["name"] == params["arm_id"] == "fer"


def test_real_xacro_uses_agimus_franka_hardware_plugin() -> None:
    control = ros2_control(render_real_urdf())
    hardware = control.find("hardware")
    assert hardware is not None

    plugin = hardware.find("plugin")
    assert plugin is not None
    assert plugin.text == "agimus_franka_hardware/AgimusFrankaHardwareInterface"

    params = params_by_name(hardware)
    assert params["version"] == "1.0.0"
    assert params["arm_id"] == "fer"
    assert params["robot_ip"] == "172.17.1.2"
    assert "prefix" not in params
    assert "arm_prefix" not in params
    assert "franka_hardware/FrankaHardwareInterface" not in ET.tostring(
        control,
        encoding="unicode",
    )


def test_real_xacro_exposes_arm_position_velocity_effort_interfaces() -> None:
    control = ros2_control(render_real_urdf())
    joints = {joint.attrib["name"]: joint for joint in control.findall("joint")}

    assert set(joints) == set(FER_ARM_JOINT_NAMES)
    for joint_name in FER_ARM_JOINT_NAMES:
        joint = joints[joint_name]
        assert interface_names(joint, "command_interface") == (
            "position",
            "velocity",
            "effort",
        )
        assert interface_names(joint, "state_interface") == (
            "position",
            "velocity",
            "effort",
        )


def test_real_xacro_can_mount_regular_hand_without_ft_sensor() -> None:
    root = render_real_urdf(mount_end_effector="true")
    links = {link.attrib["name"] for link in root.findall("link")}
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}

    assert "fer_link8" in links
    assert "ati_mini45_tool_mount" not in links

    hand_joint = joints["fer_hand_joint"]
    parent = hand_joint.find("parent")
    assert parent is not None
    assert parent.attrib["link"] == "fer_link8"
