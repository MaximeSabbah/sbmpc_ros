"""Generate a randomized start-pose keyframe for the MuJoCo backend.

`initial_q:=random` (sim-only launch argument) validates through ROS what
V-A5 certified in hydrax: the reference plan always starts at the
hardcoded home pose, so a start pose near — not at — home is a placement
gap the controller must absorb. The perturbation is drawn fresh at each
launch, uniform per joint within the V-A5-certified envelope.

The sim reads its start pose from the scene's `home` keyframe twice (the
`initial_keyframe` xacro param at startup and the post-warmup ResetWorld
call), and that keyframe also carries the gravity-compensation `ctrl`
that holds the arm before the LFC stack activates. A perturbed start
therefore needs a regenerated keyframe — qpos moved AND ctrl recomputed
at the new pose — not just different numbers in a parameter. The output
is a copy of the scene xml with only the `home` key rewritten, written
next to the source so the relative include and assets resolve unchanged.

Runs inside the planner runtime (uv/pixi — the launch python has no real
mujoco); the launch consumes the JSON line on stdout.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from sbmpc_bringup.constants import DEFAULT_KEYFRAME, FER_ARM_JOINT_NAMES

# The V-A5-certified placement envelope (hydrax plan doc §V-A5): uniform
# per-joint offsets up to 0.2 rad are absorbed by the exact_feedback
# controller with no extra torque.
NOISE_RAD = 0.2
# Stay clear of the joint limits, as in the V-A5 sweep.
LIMIT_MARGIN_RAD = 0.05

GENERATED_SUFFIX = ".initial_q_random.xml"


def gravity_compensation_ctrl(
    model: mujoco.MjModel, data: mujoco.MjData, q_arm: np.ndarray
) -> np.ndarray:
    """Arm actuator ctrl holding q_arm at rest (the keyframe-ctrl role)."""
    data.qpos[:7] = q_arm
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    gear = model.actuator_gear[:7, 0]
    return data.qfrc_bias[:7] / gear


def perturb_home_keyframe(
    scene_path: Path, rng: np.random.Generator
) -> tuple[Path, dict[str, list[float]]]:
    """Write a copy of the scene whose `home` key starts the arm off-home.

    Returns the generated path and the draw (for the launch log).
    """
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    for i, name in enumerate(FER_ARM_JOINT_NAMES):
        assert model.joint(i).name == name, "arm joints must lead the model"
        assert model.actuator(i).trnid[0] == i, "actuator i must drive joint i"
    key = model.key(DEFAULT_KEYFRAME)

    home_q = np.array(key.qpos[:7])
    offset = rng.uniform(-NOISE_RAD, NOISE_RAD, 7)
    q_arm = np.clip(
        home_q + offset,
        model.jnt_range[:7, 0] + LIMIT_MARGIN_RAD,
        model.jnt_range[:7, 1] - LIMIT_MARGIN_RAD,
    )

    qpos = np.array(key.qpos)
    qpos[:7] = q_arm
    ctrl = np.array(key.ctrl)
    ctrl[:7] = gravity_compensation_ctrl(model, mujoco.MjData(model), q_arm)

    tree = ET.parse(scene_path)
    key_element = tree.getroot().find(
        f"./keyframe/key[@name='{DEFAULT_KEYFRAME}']"
    )
    if key_element is None:
        raise ValueError(f"no '{DEFAULT_KEYFRAME}' keyframe in {scene_path}")
    key_element.set("qpos", " ".join(repr(float(v)) for v in qpos))
    key_element.set("ctrl", " ".join(repr(float(v)) for v in ctrl))

    out_path = scene_path.with_name(scene_path.stem + GENERATED_SUFFIX)
    tree.write(out_path, xml_declaration=True, encoding="utf-8")
    # Regenerated every initial_q:=random launch; loading it revalidates it.
    mujoco.MjModel.from_xml_path(str(out_path))
    return out_path, {
        "q_arm": q_arm.tolist(),
        "offset": (q_arm - home_q).tolist(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    args = parser.parse_args(argv)

    out_path, draw = perturb_home_keyframe(
        args.scene, np.random.default_rng()
    )
    print(json.dumps({"mujoco_model": str(out_path), **draw}))


if __name__ == "__main__":
    main()
