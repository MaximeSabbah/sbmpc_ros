from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch_ros.actions import Node


LAUNCH_DIR = Path(__file__).resolve().parents[1] / "launch"
LAUNCH_FILE = "sbmpc_franka_bringup.launch.py"

EXPECTED_ARGUMENTS = {
    "backend",
    "planner",
    "enable_nonzero_control",
    "use_rviz",
    "headless",
    "initial_q",
    "robot_ip",
    "publish_rollout_markers",
    "use_gripper",
}


def load_launch_module():
    launch_path = LAUNCH_DIR / LAUNCH_FILE
    spec = importlib.util.spec_from_file_location(launch_path.stem, launch_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_launch_module()


@pytest.fixture(autouse=True)
def _skip_preflight(monkeypatch):
    # The clean-graph preflight shells out to `ros2 node list` / `ps`; stub it
    # for these structural tests.
    monkeypatch.setattr(MODULE, "assert_clean_ros_graph", lambda *args, **kwargs: [])


def declared_argument_defaults(launch_description: LaunchDescription) -> dict[str, str]:
    defaults: dict[str, str] = {}
    for entity in launch_description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            defaults[entity.name] = "".join(
                getattr(value, "text", str(value)) for value in entity.default_value
            )
    return defaults


def node_arguments(node: Node) -> list[str]:
    return [
        "".join(getattr(part, "text", str(part)) for part in argument)
        if isinstance(argument, list)
        else getattr(argument, "text", str(argument))
        for argument in getattr(node, "_Node__arguments")
    ]


def setup_actions(backend: str, **overrides):
    launch_description = MODULE.generate_launch_description()
    defaults = declared_argument_defaults(launch_description)
    context = LaunchContext()
    for name, value in {**defaults, "backend": backend, **overrides}.items():
        context.launch_configurations[name] = value
    setup = next(
        entity
        for entity in launch_description.entities
        if isinstance(entity, OpaqueFunction)
    )
    return setup.execute(context)


def nodes_of(actions) -> list[Node]:
    return [action for action in actions if isinstance(action, Node)]


def node_specs(actions) -> set[tuple]:
    return {(node.node_package, node.node_executable) for node in nodes_of(actions)}


def spawners(actions) -> list[Node]:
    # Filter on the executable, not the package: the real control host is also a
    # `controller_manager` package node (ros2_control_node) but is not a spawner.
    return [n for n in nodes_of(actions) if n.node_executable == "spawner"]


def find_spawner(actions, controller: str) -> Node:
    return next(n for n in spawners(actions) if controller in node_arguments(n))


def includes(actions) -> list[IncludeLaunchDescription]:
    return [a for a in actions if isinstance(a, IncludeLaunchDescription)]


# --- argument surface -------------------------------------------------------


def test_declares_exactly_the_nine_arguments_with_expected_defaults() -> None:
    defaults = declared_argument_defaults(MODULE.generate_launch_description())

    assert set(defaults) == EXPECTED_ARGUMENTS
    assert defaults["backend"] == "mujoco"
    assert defaults["planner"] == "hydrax"
    assert defaults["enable_nonzero_control"] == "true"
    assert defaults["use_rviz"] == "true"
    assert defaults["headless"] == "false"
    assert defaults["initial_q"] == "home"
    assert defaults["robot_ip"] == "172.17.1.2"
    assert defaults["publish_rollout_markers"] == "false"
    assert defaults["use_gripper"] == "true"


def test_backend_argument_is_constrained_to_mujoco_or_real() -> None:
    backend_arg = next(
        entity
        for entity in MODULE.generate_launch_description().entities
        if isinstance(entity, DeclareLaunchArgument) and entity.name == "backend"
    )
    assert list(backend_arg.choices) == ["mujoco", "real"]


# --- mujoco backend ---------------------------------------------------------


def test_mujoco_backend_builds_the_simulation_node_set() -> None:
    actions = setup_actions("mujoco")

    assert node_specs(actions) == {
        ("robot_state_publisher", "robot_state_publisher"),
        ("mujoco_ros2_control", "ros2_control_node"),
        ("rviz2", "rviz2"),
        (None, "python"),  # the bridge runs under the pixi wrapper (no package)
        ("controller_manager", "spawner"),
    }
    # joint_state_broadcaster + gripper_action_controller + LFC stack.
    assert len(spawners(actions)) == 3
    # No real-only pieces.
    assert includes(actions) == []
    assert not any(n.node_package == "joint_state_publisher" for n in nodes_of(actions))


def test_mujoco_lfc_spawner_activates_up_front_with_the_gravity_overlay() -> None:
    lfc = find_spawner(setup_actions("mujoco"), "linear_feedback_controller")
    arguments = node_arguments(lfc)

    assert "--activate-as-group" in arguments
    assert "--inactive" not in arguments
    # controllers + base LFC params + the sim gravity-comp overlay (D10).
    assert arguments.count("--param-file") == 3


# --- real backend -----------------------------------------------------------


def test_real_backend_builds_the_hardware_node_set() -> None:
    actions = setup_actions("real")

    assert node_specs(actions) == {
        ("robot_state_publisher", "robot_state_publisher"),
        ("controller_manager", "ros2_control_node"),
        ("rviz2", "rviz2"),
        (None, "python"),  # the bridge
        ("controller_manager", "spawner"),
        ("joint_state_publisher", "joint_state_publisher"),
    }
    # joint_state_broadcaster + franka_robot_state_broadcaster + LFC stack.
    assert len(spawners(actions)) == 3
    # Exactly the agimus_franka_gripper include (D21).
    assert len(includes(actions)) == 1


def test_real_joint_state_broadcaster_remaps_to_franka_joint_states() -> None:
    jsb = find_spawner(setup_actions("real"), "joint_state_broadcaster")
    arguments = node_arguments(jsb)

    assert "--controller-ros-args=--remap" in arguments
    assert "--controller-ros-args=joint_states:=franka/joint_states" in arguments


def test_real_lfc_spawner_activates_up_front_without_the_sim_overlay() -> None:
    lfc = find_spawner(setup_actions("real"), "linear_feedback_controller")
    arguments = node_arguments(lfc)

    assert "--activate-as-group" in arguments
    assert "--inactive" not in arguments
    # controllers + base LFC params only (no sim overlay on real).
    assert arguments.count("--param-file") == 2


# --- gripper wiring ----------------------------------------------------------


def bridge_param_dict(actions) -> dict:
    """The inline parameter dict of the bridge node, flattened.

    launch_ros normalizes parameter dicts: keys become substitution tuples
    and string VALUES become substitution tuples of their yaml dump (with
    the trailing document-end marker); flatten both back to plain python.
    """
    import yaml

    def text(parts) -> str:
        return "".join(getattr(part, "text", str(part)) for part in parts)

    node = next(n for n in nodes_of(actions) if n.node_executable == "python")
    for entry in getattr(node, "_Node__parameters"):
        if isinstance(entry, dict):
            return {
                text(key): (
                    yaml.safe_load(text(value))
                    if isinstance(value, (list, tuple))
                    else value
                )
                for key, value in entry.items()
            }
    raise AssertionError("bridge node has no inline parameter dict")


def test_gripper_wiring_is_injected_per_backend() -> None:
    sim = bridge_param_dict(setup_actions("mujoco"))
    assert sim["gripper_action_name"] == "/gripper_action_controller/gripper_cmd"
    # sim overdrives to 0: the effort controller's squeeze force IS the
    # residual position error times its PID p (P-B2 fix, 2026-07-16)
    assert sim["gripper_close_position"] == 0.0

    real = bridge_param_dict(setup_actions("real"))
    assert real["gripper_action_name"] == "/fer_gripper/gripper_action"
    # real maps position -> franka grasp width (2x, epsilon +-5 mm): half
    # the 0.04 m cube, or the grasp reports failure while physically
    # holding it and trips the bridge's fail-closed path
    assert real["gripper_close_position"] == 0.02


def test_use_gripper_false_unwires_the_gripper_client() -> None:
    params = bridge_param_dict(setup_actions("mujoco", use_gripper="false"))
    assert params["gripper_action_name"] == ""
