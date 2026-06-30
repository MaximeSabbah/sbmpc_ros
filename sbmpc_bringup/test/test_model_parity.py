"""Dynamic-model parity: the MuJoCo sim model must track the real robot.

`agimus_franka_description` (the description used by the real Franka) is the
single source of truth for the arm's mass/inertia, joint limits, torque limits,
and joint damping/friction. The MuJoCo simulation model
(`mujoco/fer_ros2_control.xml`) carries those values inline, so it can silently
drift. These tests read the Agimus YAMLs and assert the MJCF still matches.

Known, intentional sim-only deviation (NOT checked here): `armature=0.1`, a
MuJoCo rotor-inertia surrogate the FER URDF does not model.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

MJCF_PATH = Path(__file__).resolve().parents[1] / "mujoco" / "fer_ros2_control.xml"

# MuJoCo body name -> Agimus inertials.yaml key.
BODY_TO_AGIMUS = {
    "link0": "link0",
    "link1": "link1",
    "link2": "link2",
    "link3": "link3",
    "link4": "link4",
    "link5": "link5",
    "link6": "link6",
    "link7": "link7",
    "hand": "hand",
    "left_finger": "leftfinger",
    "right_finger": "rightfinger",
}
ARM_JOINTS = tuple(f"fer_joint{i}" for i in range(1, 8))
# The MJCF maps the Agimus non-standard URDF attribute mu_viscous to MuJoCo
# joint damping, and friction to frictionloss (see the MJCF comment on link1).
DEFAULT_RANGE = (-2.8973, 2.8973)  # MJCF `panda` joint default class.


def _floats(text: str) -> list[float]:
    return [float(token) for token in text.split()]


def _agimus_share() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("agimus_franka_description"))
    except Exception:  # pragma: no cover - environment without the real stack
        pytest.skip("agimus_franka_description not available")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _agimus_inertials() -> dict:
    share = _agimus_share()
    arm = _load_yaml(share / "robots" / "fer" / "inertials.yaml")
    hand = _load_yaml(
        share / "end_effectors" / "agimus_franka_hand" / "inertials.yaml"
    )
    return {**arm, **hand}


def _mjcf_bodies() -> dict[str, ET.Element]:
    root = ET.parse(MJCF_PATH).getroot()
    return {body.get("name"): body for body in root.iter("body")}


def _mjcf_joints() -> dict[str, dict[str, str]]:
    root = ET.parse(MJCF_PATH).getroot()
    joints: dict[str, dict[str, str]] = {}
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name is not None:
            joints[name] = joint.attrib
    return joints


def _mjcf_actuator_forceranges() -> dict[str, tuple[float, float]]:
    root = ET.parse(MJCF_PATH).getroot()
    out: dict[str, tuple[float, float]] = {}
    for actuator in root.iter("motor"):
        joint = actuator.get("joint")
        forcerange = actuator.get("forcerange")
        if joint is not None and forcerange is not None:
            lo, hi = _floats(forcerange)
            out[joint] = (lo, hi)
    return out


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)


def test_mjcf_inertials_match_agimus_description() -> None:
    agimus = _agimus_inertials()
    bodies = _mjcf_bodies()

    for body_name, agimus_key in BODY_TO_AGIMUS.items():
        body = bodies[body_name]
        inertial = body.find("inertial")
        assert inertial is not None, f"{body_name} has no <inertial>"
        ref = agimus[agimus_key]

        assert _close(float(inertial.get("mass")), float(ref["mass"])), body_name

        pos = _floats(inertial.get("pos"))
        xyz = _floats(str(ref["origin"]["xyz"]))
        assert all(_close(p, q) for p, q in zip(pos, xyz)), f"{body_name} com"

        ref_inertia = ref["inertia"]
        if inertial.get("fullinertia") is not None:
            ixx, iyy, izz, ixy, ixz, iyz = _floats(inertial.get("fullinertia"))
        else:
            ixx, iyy, izz = _floats(inertial.get("diaginertia"))
            ixy = ixz = iyz = 0.0
        expected = {
            "xx": ixx, "yy": iyy, "zz": izz, "xy": ixy, "xz": ixz, "yz": iyz,
        }
        for axis, value in expected.items():
            assert _close(value, float(ref_inertia[axis])), f"{body_name} I{axis}"


def test_mjcf_joint_position_limits_match_agimus_description() -> None:
    share = _agimus_share()
    limits = _load_yaml(share / "robots" / "fer" / "joint_limits.yaml")
    joints = _mjcf_joints()

    for joint_name in ARM_JOINTS:
        attrib = joints[joint_name]
        lo, hi = (
            _floats(attrib["range"]) if "range" in attrib else DEFAULT_RANGE
        )
        ref = limits[joint_name.removeprefix("fer_")]["limit"]
        assert _close(lo, float(ref["lower"])), f"{joint_name} lower"
        assert _close(hi, float(ref["upper"])), f"{joint_name} upper"


def test_mjcf_torque_limits_match_agimus_description() -> None:
    share = _agimus_share()
    limits = _load_yaml(share / "robots" / "fer" / "joint_limits.yaml")
    forceranges = _mjcf_actuator_forceranges()

    for joint_name in ARM_JOINTS:
        lo, hi = forceranges[joint_name]
        effort = float(limits[joint_name.removeprefix("fer_")]["limit"]["effort"])
        assert _close(hi, effort), f"{joint_name} +effort"
        assert _close(lo, -effort), f"{joint_name} -effort"


def test_mjcf_joints_use_menagerie_damping_standard() -> None:
    # Joint dissipation uses the MuJoCo Menagerie Panda standard
    # (armature=0.1, damping=1, no frictionloss) for numerical stability. The
    # real joint friction (small viscous + Coulomb, per the De Luca/Gaz
    # identification) is a separate model not yet identified for this stack and
    # is intentionally NOT mirrored here.
    root = ET.parse(MJCF_PATH).getroot()
    panda_default = next(d for d in root.iter("default") if d.get("class") == "panda")
    default_joint = panda_default.find("joint")
    assert default_joint is not None
    assert _close(float(default_joint.get("armature")), 0.1)
    assert _close(float(default_joint.get("damping")), 1.0)
    assert default_joint.get("frictionloss") in (None, "0", "0.0")

    joints = _mjcf_joints()
    for joint_name in ARM_JOINTS:
        attrib = joints[joint_name]
        # No per-joint override: inherit the standard from the panda class.
        assert "damping" not in attrib, f"{joint_name} re-adds a damping override"
        assert "frictionloss" not in attrib, f"{joint_name} re-adds frictionloss"
