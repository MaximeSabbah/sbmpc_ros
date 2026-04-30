from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from sbmpc_bringup.constants import FER_ARM_JOINT_NAMES, FER_GRIPPER_JOINT_NAME


xacro = pytest.importorskip("xacro")

PACKAGE_DIR = Path(__file__).resolve().parents[1]
XACRO_FILE = PACKAGE_DIR / "urdf" / "franka_arm_with_sbmpc_mujoco.urdf.xacro"
MUJOCO_DIR = PACKAGE_DIR / "mujoco"
ROS2_CONTROL_SCENE = MUJOCO_DIR / "panda_pick_place_ros2_control_scene.xml"
ROS2_CONTROL_PANDA = MUJOCO_DIR / "panda_ros2_control.xml"
BENCHMARK_DIR = Path("/workspace/sbmpc/examples/panda_pick_place")
BENCHMARK_SCENE = BENCHMARK_DIR / "scene.xml"
BENCHMARK_PANDA = BENCHMARK_DIR / "panda.xml"


def render_mujoco_urdf() -> ET.Element:
    document = xacro.process_file(
        str(XACRO_FILE),
        mappings={"headless": "true", "mujoco_model": str(ROS2_CONTROL_SCENE)},
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


def test_mujoco_xacro_renders_mujoco_system_with_existing_model() -> None:
    control = ros2_control(render_mujoco_urdf())
    hardware = control.find("hardware")
    assert hardware is not None

    plugin = hardware.find("plugin")
    assert plugin is not None
    assert plugin.text == "mujoco_ros2_control/MujocoSystemInterface"

    params = params_by_name(hardware)
    assert Path(params["mujoco_model"]) == ROS2_CONTROL_SCENE
    assert Path(params["mujoco_model"]).is_file()
    assert params["initial_keyframe"] == "home"
    assert params["headless"] == "true"


def test_mujoco_xacro_exposes_fer_arm_effort_interfaces_and_gripper_position() -> None:
    control = ros2_control(render_mujoco_urdf())
    joints = {joint.attrib["name"]: joint for joint in control.findall("joint")}

    assert set(joints) == {*FER_ARM_JOINT_NAMES, FER_GRIPPER_JOINT_NAME}
    for joint_name in FER_ARM_JOINT_NAMES:
        joint = joints[joint_name]
        assert interface_names(joint, "command_interface") == ("effort",)
        assert interface_names(joint, "state_interface") == (
            "position",
            "velocity",
            "effort",
        )

    gripper = joints[FER_GRIPPER_JOINT_NAME]
    assert interface_names(gripper, "command_interface") == ("position",)
    assert interface_names(gripper, "state_interface") == (
        "position",
        "velocity",
        "effort",
    )


def without_approved_panda_changes(text: str) -> str:
    replacements = {
        "panda_ros2_control": "panda",
        "/workspace/sbmpc/examples/panda_pick_place/assets": "assets",
        "fer_finger_joint1": "finger_joint1",
        "fer_finger_joint2": "finger_joint2",
    }
    for index in range(1, 8):
        replacements[f"fer_joint{index}"] = f"joint{index}"
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s*<actuator>.*?</actuator>\s*", "\n  <actuator/>\n", text, flags=re.S)
    return text


def test_ros2_control_mjcf_copy_preserves_benchmark_physics_except_names_and_actuators() -> None:
    assert without_approved_panda_changes(ROS2_CONTROL_PANDA.read_text()) == (
        without_approved_panda_changes(BENCHMARK_PANDA.read_text())
    )


def test_ros2_control_scene_preserves_benchmark_scene_except_include_name() -> None:
    copied = ROS2_CONTROL_SCENE.read_text()
    assert 'timestep="0.001"' in copied
    copied = copied.replace("panda_ros2_control pick and place", "panda pick and place")
    copied = copied.replace("panda_ros2_control.xml", "panda.xml")
    copied = copied.replace('timestep="0.001"', 'timestep="0.005"')
    assert copied == BENCHMARK_SCENE.read_text()


def test_ros2_control_mjcf_uses_effort_compatible_arm_actuators() -> None:
    root = ET.parse(ROS2_CONTROL_PANDA).getroot()
    actuators = root.find("actuator")
    assert actuators is not None

    by_name = {actuator.attrib["name"]: actuator for actuator in actuators}
    for joint_name in FER_ARM_JOINT_NAMES:
        actuator = by_name[joint_name]
        assert actuator.tag == "motor"
        assert actuator.attrib["joint"] == joint_name
    assert by_name["fer_finger_joint1"].tag == "position"
    assert by_name["fer_finger_joint1"].attrib["joint"] == "fer_finger_joint1"


def test_ros2_control_scene_loads_with_mujoco_when_python_bindings_are_available() -> None:
    mujoco = pytest.importorskip("mujoco")
    if not hasattr(mujoco, "MjModel"):
        pytest.skip("MuJoCo Python bindings are not available in this environment")

    model = mujoco.MjModel.from_xml_path(str(ROS2_CONTROL_SCENE))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home") >= 0
    for joint_name in (*FER_ARM_JOINT_NAMES, "fer_finger_joint1", "fer_finger_joint2"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) >= 0
