from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sbmpc_bringup.initial_q import (  # noqa: E402
    LIMIT_MARGIN_RAD,
    NOISE_RAD,
    gravity_compensation_ctrl,
    perturb_home_keyframe,
)

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCENE = PACKAGE_DIR / "mujoco" / "fer_pick_place_ros2_control_scene.xml"


@pytest.fixture
def generated_scene():
    out_path, draw = perturb_home_keyframe(SCENE, np.random.default_rng(7))
    try:
        yield out_path, draw
    finally:
        out_path.unlink(missing_ok=True)


def test_gravity_compensation_reproduces_the_committed_home_ctrl() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    ctrl = gravity_compensation_ctrl(
        model, mujoco.MjData(model), np.array(model.key("home").qpos[:7])
    )
    assert np.allclose(ctrl, model.key("home").ctrl[:7], atol=1e-4)


def test_generated_scene_perturbs_only_the_arm_within_the_envelope(
    generated_scene,
) -> None:
    out_path, draw = generated_scene
    original = mujoco.MjModel.from_xml_path(str(SCENE))
    generated = mujoco.MjModel.from_xml_path(str(out_path))
    home, key = original.key("home"), generated.key("home")

    offset = np.asarray(draw["offset"])
    assert np.all(np.abs(offset) <= NOISE_RAD)
    assert np.any(np.abs(offset) > 0.0)
    assert np.allclose(key.qpos[:7], np.asarray(home.qpos[:7]) + offset)
    assert np.all(
        key.qpos[:7] >= original.jnt_range[:7, 0] + LIMIT_MARGIN_RAD - 1e-12
    )
    assert np.all(
        key.qpos[:7] <= original.jnt_range[:7, 1] - LIMIT_MARGIN_RAD + 1e-12
    )
    # Gripper and object stay exactly at the home keyframe
    assert np.array_equal(key.qpos[7:], home.qpos[7:])
    assert key.ctrl[7] == home.ctrl[7]


def test_generated_ctrl_is_gravity_compensation_at_the_drawn_pose(
    generated_scene,
) -> None:
    out_path, draw = generated_scene
    model = mujoco.MjModel.from_xml_path(str(out_path))
    expected = gravity_compensation_ctrl(
        model, mujoco.MjData(model), np.asarray(draw["q_arm"])
    )
    assert np.allclose(model.key("home").ctrl[:7], expected, atol=1e-6)


def test_generation_leaves_the_source_scene_untouched(generated_scene) -> None:
    out_path, _ = generated_scene
    assert out_path != SCENE
    original = mujoco.MjModel.from_xml_path(str(SCENE))
    assert np.allclose(
        original.key("home").qpos[:7], [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    )
