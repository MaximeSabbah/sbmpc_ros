from __future__ import annotations

import signal
import threading
from datetime import datetime, timezone

import pytest

import sbmpc_bringup.record_run as record_run
from sbmpc_bringup.record_run import (
    DEFAULT_CACHE_BYTES,
    _acquire_capture_lock,
    _bringup_command,
    _launch_arguments,
    _launch_value,
    _new_run_directory,
    _publish_run_path,
    _record_command,
    _stop_process,
    _validated_label,
    _with_backend_defaults,
    main,
)
from sbmpc_bringup.run_topics import RUN_TOPICS


def test_topic_contract_is_cross_backend_and_nonredundant() -> None:
    assert len(RUN_TOPICS) == len(set(RUN_TOPICS))
    for topic in (
        "/control",
        "/sensor",
        "/output_joint_effort",
        "/joint_states",
        "/franka/joint_states",
        "/franka_robot_state_broadcaster/robot_state",
        "/diagnostics",
        "/controller_manager/activity",
        "/fer_gripper/joint_states",
        "/fer_gripper/gripper_action/_action/status",
        "/gripper_action_controller/gripper_cmd/_action/status",
        "/simulator/object_pose",
        "/mujoco_actuators_states",
        "/clock",
    ):
        assert topic in RUN_TOPICS
    assert "/controller_manager/statistics/full" not in RUN_TOPICS
    assert "/franka_robot_state_broadcaster/current_pose" not in RUN_TOPICS


def test_record_command_uses_tuned_mcap_and_explicit_topics(tmp_path) -> None:
    command = _record_command(tmp_path / "run", "pregrasp", "real")

    assert command[:3] == ["ros2", "bag", "record"]
    assert command[command.index("--storage") + 1] == "mcap"
    assert command[command.index("--storage-preset-profile") + 1] == "fastwrite"
    assert command[command.index("--max-cache-size") + 1] == str(
        DEFAULT_CACHE_BYTES
    )
    assert "--all" not in command
    assert "--compression-mode" not in command
    assert "--include-hidden-topics" in command
    assert "--disable-keyboard-controls" in command
    assert "sbmpc_backend=real" in command
    assert command[command.index("--topics") + 1 :] == list(RUN_TOPICS)


def test_launch_arguments_are_forwarded_without_a_second_recorder() -> None:
    values = _launch_arguments(
        ["--", "backend:=mujoco", "headless:=true", "use_rviz:=false"]
    )

    assert _launch_value(values, "backend", "real") == "mujoco"
    assert _bringup_command(values) == [
        "ros2",
        "launch",
        "sbmpc_bringup",
        "sbmpc_franka_bringup.launch.py",
        "backend:=mujoco",
        "headless:=true",
        "use_rviz:=false",
    ]
    with pytest.raises(ValueError, match="record_replay was replaced"):
        _launch_arguments(["record_replay:=/tmp/legacy.json"])
    with pytest.raises(ValueError, match="name:=value"):
        _launch_arguments(["backend", "mujoco"])


def test_real_backend_defaults_to_disarmed_without_overriding_explicit_choice() -> None:
    assert _with_backend_defaults(["backend:=real"]) == [
        "backend:=real",
        "enable_nonzero_control:=false",
    ]
    assert _with_backend_defaults(
        ["backend:=real", "enable_nonzero_control:=true"]
    ) == ["backend:=real", "enable_nonzero_control:=true"]
    assert _with_backend_defaults(["backend:=mujoco"]) == ["backend:=mujoco"]


def test_stop_process_signals_surviving_group_after_leader_exits(monkeypatch) -> None:
    class ExitedLeader:
        pid = 4242
        returncode = 7

        def poll(self):
            return self.returncode

    group_states = iter((True, False, False, False))
    signals = []
    monkeypatch.setattr(
        record_run,
        "_process_group_exists",
        lambda process_group_id: next(group_states),
    )
    monkeypatch.setattr(
        record_run,
        "_signal_process_group",
        lambda process_group_id, signum: signals.append((process_group_id, signum)),
    )

    result = _stop_process(
        ExitedLeader(),
        name="test group",
        force_requested=threading.Event(),
        graceful_timeout_sec=0.0,
    )

    assert result == 7
    assert signals == [(4242, signal.SIGINT)]


def test_run_directory_and_path_file_are_deterministic_and_atomic(tmp_path) -> None:
    now = datetime(2026, 7, 22, 12, 34, 56, tzinfo=timezone.utc)
    run = _new_run_directory(tmp_path / "runs", "pregrasp", now=now)
    path_file = tmp_path / "current_run"

    _publish_run_path(path_file, run)

    assert run.name == "pregrasp_20260722T123456Z"
    assert (run / "ros_logs").is_dir()
    assert path_file.read_text() == f"{run.resolve()}\n"
    assert not (tmp_path / ".current_run.tmp").exists()


def test_capture_lock_rejects_a_second_supervisor(tmp_path) -> None:
    lock_path = tmp_path / "capture.lock"
    first = _acquire_capture_lock(lock_path)
    try:
        with pytest.raises(RuntimeError, match="already active"):
            _acquire_capture_lock(lock_path)
    finally:
        first.close()

    second = _acquire_capture_lock(lock_path)
    second.close()


def test_label_rejects_paths_and_shell_punctuation() -> None:
    assert _validated_label("grasp-2") == "grasp-2"
    for value in ("../run", "grasp test", "grasp;test", "_grasp"):
        with pytest.raises(ValueError):
            _validated_label(value)


def test_print_command_defaults_to_mujoco_and_creates_nothing(
    tmp_path,
    capsys,
) -> None:
    assert (
        main(
            [
                "--label",
                "pregrasp",
                "--output-root",
                str(tmp_path),
                "--print-command",
                "--",
                "headless:=true",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "ros2 bag record" in output
    assert "sbmpc_backend=mujoco" in output
    assert "ros2 launch sbmpc_bringup sbmpc_franka_bringup.launch.py" in output
    assert "headless:=true" in output
    assert list(tmp_path.iterdir()) == []


def test_print_command_defaults_real_capture_to_disarmed(tmp_path, capsys) -> None:
    assert (
        main(
            [
                "--label",
                "pregrasp_real",
                "--output-root",
                str(tmp_path),
                "--print-command",
                "--",
                "backend:=real",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "backend:=real enable_nonzero_control:=false" in output
    assert list(tmp_path.iterdir()) == []
