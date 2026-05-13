from __future__ import annotations

import importlib.util
from pathlib import Path

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch_ros.actions import Node


LAUNCH_DIR = Path(__file__).resolve().parents[1] / "launch"


def load_launch_module(filename: str):
    launch_path = LAUNCH_DIR / filename
    spec = importlib.util.spec_from_file_location(launch_path.stem, launch_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def declared_argument_defaults(launch_description: LaunchDescription) -> dict[str, str]:
    defaults: dict[str, str] = {}
    for entity in launch_description.entities:
        if not isinstance(entity, DeclareLaunchArgument):
            continue
        defaults[entity.name] = "".join(
            getattr(value, "text", str(value)) for value in entity.default_value
        )
    return defaults


def configure_context(defaults: dict[str, str]) -> LaunchContext:
    context = LaunchContext()
    for name, value in defaults.items():
        context.launch_configurations[name] = value
    return context


def node_name(node: Node) -> str | None:
    value = getattr(node, "_Node__node_name")
    if isinstance(value, list):
        return "".join(getattr(part, "text", str(part)) for part in value) or None
    return value


def event_nodes(handler: RegisterEventHandler) -> list[Node]:
    event_handler = handler.event_handler
    actions = getattr(event_handler, "_OnActionEventBase__actions_on_event")
    return [action for action in actions if isinstance(action, Node)]


def test_mujoco_launch_imports_and_declares_expected_arguments() -> None:
    module = load_launch_module("sbmpc_franka_lfc_mujoco_sim.launch.py")
    launch_description = module.generate_launch_description()

    assert isinstance(launch_description, LaunchDescription)
    defaults = declared_argument_defaults(launch_description)
    assert set(defaults) == {
        "allow_existing_ros_graph",
        "bridge_params_file",
        "bridge_runtime_script",
        "controller_manager_name",
        "controllers_file",
        "enable_nonzero_control",
        "headless",
        "lfc_params_file",
        "mujoco_model",
        "pixi_env",
        "record_replay",
        "record_replay_autosave_period_sec",
        "record_replay_duration_sec",
        "record_replay_include_warmup",
        "record_replay_output",
        "record_replay_startup_timeout_sec",
        "sbmpc_dir",
        "sim_lfc_params_file",
    }
    assert defaults["headless"] == "true"
    assert defaults["enable_nonzero_control"] == "false"
    assert defaults["record_replay"] == "false"
    assert defaults["record_replay_output"] == "/tmp/sbmpc_ros_replay.json"
    assert defaults["record_replay_duration_sec"] == "0"
    assert defaults["controller_manager_name"] == "/controller_manager"
    assert defaults["allow_existing_ros_graph"] == "false"
    assert "sbmpc_bridge_exact_async.yaml" in defaults["bridge_params_file"]
    assert "panda_pick_place_ros2_control_scene.xml" in defaults["mujoco_model"]


def test_mujoco_launch_has_expected_node_set(monkeypatch) -> None:
    module = load_launch_module("sbmpc_franka_lfc_mujoco_sim.launch.py")
    monkeypatch.setattr(module, "assert_clean_ros_graph", lambda context: [])
    launch_description = module.generate_launch_description()
    defaults = declared_argument_defaults(launch_description)
    setup = next(
        entity for entity in launch_description.entities if isinstance(entity, OpaqueFunction)
    )

    actions = setup.execute(configure_context(defaults))
    nodes = [action for action in actions if isinstance(action, Node)]
    for handler in [action for action in actions if isinstance(action, RegisterEventHandler)]:
        nodes.extend(event_nodes(handler))
    node_specs = {
        (node.node_package, node.node_executable, node_name(node)) for node in nodes
    }

    assert node_specs == {
        ("robot_state_publisher", "robot_state_publisher", None),
        ("mujoco_ros2_control", "ros2_control_node", None),
        ("controller_manager", "spawner", None),
        (None, "python", None),
        ("sbmpc_bringup", "record_sbmpc_replay", None),
    }
    spawners = [node for node in nodes if node.node_package == "controller_manager"]
    assert len(spawners) == 3


def test_mujoco_launch_declarations_do_not_reintroduce_gazebo_args() -> None:
    module = load_launch_module("sbmpc_franka_lfc_mujoco_sim.launch.py")
    launch_description = module.generate_launch_description()

    for name, default in declared_argument_defaults(launch_description).items():
        text = f"{name} {default}".lower()
        assert "gazebo" not in text
        assert "ros_gz" not in text
        assert "gz_" not in text
