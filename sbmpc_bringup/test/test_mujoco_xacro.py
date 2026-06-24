from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from sbmpc_bringup.constants import FER_ARM_JOINT_NAMES, FER_GRIPPER_JOINT_NAME


xacro = pytest.importorskip("xacro")

PACKAGE_DIR = Path(__file__).resolve().parents[1]
XACRO_FILE = PACKAGE_DIR / "urdf" / "franka_arm_with_sbmpc_mujoco.urdf.xacro"
MUJOCO_DIR = PACKAGE_DIR / "mujoco"
FER_ROS2_CONTROL_SCENE = MUJOCO_DIR / "fer_pick_place_ros2_control_scene.xml"
FER_ROS2_CONTROL_MODEL = MUJOCO_DIR / "fer_ros2_control.xml"


def render_mujoco_urdf() -> ET.Element:
    document = xacro.process_file(
        str(XACRO_FILE),
        mappings={"headless": "true", "mujoco_model": str(FER_ROS2_CONTROL_SCENE)},
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


def test_mujoco_xacro_renders_mujoco_system_with_existing_fer_model() -> None:
    control = ros2_control(render_mujoco_urdf())
    hardware = control.find("hardware")
    assert hardware is not None

    plugin = hardware.find("plugin")
    assert plugin is not None
    assert plugin.text == "mujoco_ros2_control/MujocoSystemInterface"

    params = params_by_name(hardware)
    assert Path(params["mujoco_model"]) == FER_ROS2_CONTROL_SCENE
    assert Path(params["mujoco_model"]).is_file()
    assert params["initial_keyframe"] == "home"
    assert params["headless"] == "true"


def test_home_keyframe_uses_gravity_compensation_motor_torques() -> None:
    root = ET.parse(FER_ROS2_CONTROL_SCENE).getroot()
    home = root.find("./keyframe/key[@name='home']")
    assert home is not None

    controls = tuple(float(value) for value in home.attrib["ctrl"].split())
    assert controls == pytest.approx(
        (0.0, -4.0002565, -0.64374495, 22.022167, 0.63384765, 2.2781773, 0.0, 0.04)
    )


def test_mujoco_xacro_attaches_regular_hand_without_ft_sensor() -> None:
    root = render_mujoco_urdf()
    links = {link.attrib["name"] for link in root.findall("link")}
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}

    assert "fer_link8" in links
    assert "ati_mini45_tool_mount" not in links

    hand_joint = joints["fer_hand_joint"]
    parent = hand_joint.find("parent")
    assert parent is not None
    assert parent.attrib["link"] == "fer_link8"


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
    assert interface_names(gripper, "command_interface") == ("effort",)
    assert interface_names(gripper, "state_interface") == (
        "position",
        "velocity",
        "effort",
    )


def test_ros2_control_mjcf_uses_calibrated_fer_joint_dynamics() -> None:
    urdf_root = render_mujoco_urdf()
    urdf_joints = {joint.attrib["name"]: joint for joint in urdf_root.findall("joint")}

    mjcf_root = ET.parse(FER_ROS2_CONTROL_MODEL).getroot()
    mjcf_joints = {
        joint.attrib["name"]: joint
        for joint in mjcf_root.findall(".//joint")
        if joint.attrib.get("name") in FER_ARM_JOINT_NAMES
    }

    assert set(mjcf_joints) == set(FER_ARM_JOINT_NAMES)
    for joint_name in FER_ARM_JOINT_NAMES:
        dynamics = urdf_joints[joint_name].find("dynamics")
        assert dynamics is not None

        # MuJoCo ignores Franka's non-standard mu_viscous URDF attribute.
        # We map mu_viscous to MuJoCo damping as an explicit calibrated
        # surrogate, while preserving the standard friction as frictionloss.
        assert float(dynamics.attrib["damping"]) == pytest.approx(0.003)
        assert float(mjcf_joints[joint_name].attrib["damping"]) == pytest.approx(
            float(dynamics.attrib["mu_viscous"])
        )
        assert float(mjcf_joints[joint_name].attrib["frictionloss"]) == pytest.approx(
            float(dynamics.attrib["friction"])
        )


def test_ros2_control_scene_declares_pick_place_task_contract() -> None:
    root = ET.parse(FER_ROS2_CONTROL_SCENE).getroot()
    assert root.attrib["model"] == "fer_ros2_control pick and place"

    include = root.find("include")
    assert include is not None
    assert include.attrib["file"] == "fer_ros2_control.xml"

    option = root.find("option")
    assert option is not None
    assert option.attrib["timestep"] == "0.001"

    assert root.find("./worldbody/body[@name='object']") is not None
    assert root.find("./worldbody/body[@name='target']") is not None

    sensors = {(sensor.tag, sensor.attrib["name"]) for sensor in root.findall("./sensor/*")}
    assert sensors >= {
        ("framepos", "ee_pos"),
        ("framexaxis", "ee_xaxis"),
        ("frameyaxis", "ee_yaxis"),
        ("framezaxis", "ee_zaxis"),
        ("framepos", "obj_pos"),
        ("framepos", "target_pos"),
    }


def test_ros2_control_mjcf_uses_effort_compatible_arm_actuators() -> None:
    root = ET.parse(FER_ROS2_CONTROL_MODEL).getroot()
    actuators = root.find("actuator")
    assert actuators is not None

    by_name = {actuator.attrib["name"]: actuator for actuator in actuators}
    for joint_name in FER_ARM_JOINT_NAMES:
        actuator = by_name[joint_name]
        assert actuator.tag == "motor"
        assert actuator.attrib["joint"] == joint_name
    assert by_name["fer_finger_joint1"].tag == "motor"
    assert by_name["fer_finger_joint1"].attrib["joint"] == "fer_finger_joint1"


def test_ros2_control_scene_loads_with_mujoco_when_python_bindings_are_available() -> None:
    mujoco = pytest.importorskip("mujoco")
    if not hasattr(mujoco, "MjModel"):
        pytest.skip("MuJoCo Python bindings are not available in this environment")

    model = mujoco.MjModel.from_xml_path(str(FER_ROS2_CONTROL_SCENE))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home") >= 0
    for joint_name in (*FER_ARM_JOINT_NAMES, "fer_finger_joint1", "fer_finger_joint2"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) >= 0
