"""Capture one complete SB-MPC simulation or hardware experiment.

The MCAP is the single runtime recording.  This supervisor starts the canonical
Franka bringup and rosbag recorder as separate processes, captures provenance
and console logs, and stops both in a deterministic order.  Plot generation
and replay-JSON export happen only after the launch and bag have fully stopped.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import resource
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from sbmpc_bringup.run_topics import RUN_TOPICS


DEFAULT_OUTPUT_ROOT = Path("/workspace/sbmpc_runs")
DEFAULT_PATH_FILE = Path("/tmp/sbmpc_run_path")
DEFAULT_LOCK_FILE = Path("/tmp/sbmpc_record_run.lock")
DEFAULT_CACHE_BYTES = 256 * 1024 * 1024
MAX_PROVENANCE_TEXT_BYTES = 8 * 1024 * 1024
MAX_UNTRACKED_FILE_BYTES = 2 * 1024 * 1024
MAX_UNTRACKED_TOTAL_BYTES = 20 * 1024 * 1024
MAX_UNTRACKED_ENTRIES = 2000
MAX_DIFF_CHANGED_LINES = 100_000
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_SOURCE_SUFFIXES = {
    ".bash",
    ".c",
    ".cc",
    ".cfg",
    ".cmake",
    ".cpp",
    ".cu",
    ".diff",
    ".h",
    ".hpp",
    ".json",
    ".launch",
    ".md",
    ".orig",
    ".patch",
    ".py",
    ".rej",
    ".repos",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".urdf",
    ".xacro",
    ".xml",
    ".yaml",
    ".yml",
}
_SOURCE_FILENAMES = {
    "CMakeLists.txt",
    "Dockerfile",
    "LICENSE",
    "Makefile",
    "pixi.lock",
    "uv.lock",
}
_PROVENANCE_REPOSITORIES = {
    "hydrax": Path("/workspace/hydrax"),
    "sbmpc": Path("/workspace/sbmpc"),
    "sbmpc_ros": Path("/workspace/sbmpc_ros"),
    "sbmpc_containers": Path("/workspace/sbmpc_containers"),
    "agimus_franka_ros2": Path(
        "/opt/sbmpc_deps_ws/src/agimus_franka_ros2"
    ),
    "linear_feedback_controller": Path(
        "/opt/sbmpc_deps_ws/src/linear_feedback_controller"
    ),
}


def _validated_label(value: str) -> str:
    if _LABEL.fullmatch(value) is None:
        raise ValueError(
            "label must start with an alphanumeric character and contain only "
            "letters, digits, underscores, or hyphens"
        )
    return value


def _launch_arguments(values: Sequence[str]) -> list[str]:
    result = list(values)
    if result[:1] == ["--"]:
        result = result[1:]
    invalid = [value for value in result if ":=" not in value or value.startswith("-")]
    if invalid:
        raise ValueError(
            "launch arguments after '--' must use name:=value syntax; invalid: "
            + ", ".join(repr(value) for value in invalid)
        )
    if any(value.startswith("record_replay:=") for value in result):
        raise ValueError(
            "record_replay was replaced by this canonical MCAP capture; remove "
            "that launch argument"
        )
    return result


def _launch_value(arguments: Sequence[str], name: str, default: str) -> str:
    prefix = f"{name}:="
    for argument in reversed(arguments):
        if argument.startswith(prefix):
            return argument[len(prefix) :]
    return default


def _with_backend_defaults(arguments: Sequence[str]) -> list[str]:
    """Apply recorder-level defaults which must not depend on user memory."""
    result = list(arguments)
    backend = _launch_value(result, "backend", "mujoco")
    has_arm_choice = any(
        argument.startswith("enable_nonzero_control:=") for argument in result
    )
    if backend == "real" and not has_arm_choice:
        result.append("enable_nonzero_control:=false")
    return result


def _new_run_directory(
    root: Path,
    label: str,
    *,
    now: datetime | None = None,
) -> Path:
    instant = datetime.now(timezone.utc) if now is None else now.astimezone(timezone.utc)
    stamp = instant.strftime("%Y%m%dT%H%M%SZ")
    run = root.resolve() / f"{_validated_label(label)}_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "ros_logs").mkdir()
    return run


def _publish_run_path(path_file: Path, run: Path) -> None:
    path_file = path_file.resolve()
    path_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = path_file.with_name(f".{path_file.name}.tmp")
    temporary.write_text(f"{run.resolve()}\n")
    temporary.replace(path_file)


def _acquire_capture_lock(path: Path = DEFAULT_LOCK_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(
            "another record_sbmpc_run supervisor is already active"
        ) from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def _record_command(run: Path, label: str, backend: str) -> list[str]:
    return [
        "ros2",
        "bag",
        "record",
        "--storage",
        "mcap",
        "--storage-preset-profile",
        "fastwrite",
        "--max-cache-size",
        str(DEFAULT_CACHE_BYTES),
        "--output",
        str(run / "rosbag"),
        "--node-name",
        "sbmpc_run_recorder",
        "--log-level",
        "warn",
        "--disable-keyboard-controls",
        "--include-hidden-topics",
        "--custom-data",
        f"sbmpc_label={label}",
        f"sbmpc_backend={backend}",
        "sbmpc_capture_schema=3",
        "--topics",
        *RUN_TOPICS,
    ]


def _bringup_command(launch_arguments: Sequence[str]) -> list[str]:
    return [
        "ros2",
        "launch",
        "sbmpc_bringup",
        "sbmpc_franka_bringup.launch.py",
        *launch_arguments,
    ]


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _command_text(command: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return f"capture failed: {error}\n"
    text = result.stdout
    if result.stderr:
        text += result.stderr
    if result.returncode:
        text += f"\n[exit code {result.returncode}]\n"
    return text


def _limit_value(value: int) -> int | str:
    return "unlimited" if value == resource.RLIM_INFINITY else int(value)


def _realtime_status() -> dict[str, object]:
    rtprio = resource.getrlimit(resource.RLIMIT_RTPRIO)
    memlock = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    try:
        probe = subprocess.run(
            ["chrt", "-f", "1", "true"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        detail = (probe.stdout + probe.stderr).strip()
        fifo_available = probe.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        detail = str(error)
        fifo_available = False
    return {
        "rtprio_soft": _limit_value(rtprio[0]),
        "rtprio_hard": _limit_value(rtprio[1]),
        "memlock_soft_bytes": _limit_value(memlock[0]),
        "memlock_hard_bytes": _limit_value(memlock[1]),
        "sched_fifo_probe_ok": fifo_available,
        "sched_fifo_probe_detail": detail,
    }


def _bounded_text(text: str, limit: int = MAX_PROVENANCE_TEXT_BYTES) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return text
    retained = encoded[:limit].decode("utf-8", errors="ignore")
    return retained + f"\n[provenance output truncated at {limit} bytes]\n"


def _source_like(path: Path) -> bool:
    return path.name in _SOURCE_FILENAMES or path.suffix.lower() in _SOURCE_SUFFIXES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_untracked(repository: Path, destination: Path) -> dict[str, object]:
    listing = _command_text(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repository,
    )
    relative_names = listing.splitlines()
    entries: list[dict[str, object]] = []
    copied_bytes = 0
    for relative_text in relative_names[:MAX_UNTRACKED_ENTRIES]:
        relative = Path(relative_text)
        source = (repository / relative).resolve()
        try:
            source.relative_to(repository.resolve())
        except ValueError:
            entries.append(
                {
                    "path": relative.as_posix(),
                    "copied": False,
                    "reason": "outside repository",
                }
            )
            continue
        if not source.is_file():
            continue
        size = source.stat().st_size
        reason: str | None = None
        if not _source_like(relative):
            reason = "not a source-like file"
        elif size > MAX_UNTRACKED_FILE_BYTES:
            reason = f"larger than {MAX_UNTRACKED_FILE_BYTES} bytes"
        elif copied_bytes + size > MAX_UNTRACKED_TOTAL_BYTES:
            reason = f"total copy limit {MAX_UNTRACKED_TOTAL_BYTES} bytes reached"
        if reason is not None:
            entries.append(
                {
                    "path": relative.as_posix(),
                    "size_bytes": size,
                    "copied": False,
                    "reason": reason,
                }
            )
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied_bytes += size
        entries.append(
            {
                "path": relative.as_posix(),
                "size_bytes": size,
                "sha256": _sha256(source),
                "copied": True,
            }
        )
    return {
        "entries": entries,
        "copied_bytes": copied_bytes,
        "listing_entry_count": len(relative_names),
        "listing_truncated": len(relative_names) > MAX_UNTRACKED_ENTRIES,
    }


def _tracked_diff(repository: Path) -> str:
    numstat = _command_text(["git", "diff", "--numstat", "HEAD"], cwd=repository)
    changed_lines = 0
    for line in numstat.splitlines():
        fields = line.split("\t", maxsplit=2)
        if len(fields) < 2:
            continue
        for value in fields[:2]:
            if value.isdigit():
                changed_lines += int(value)
    if changed_lines > MAX_DIFF_CHANGED_LINES:
        stat = _command_text(["git", "diff", "--stat", "HEAD"], cwd=repository)
        return (
            f"Patch omitted: {changed_lines} changed lines exceeds the "
            f"{MAX_DIFF_CHANGED_LINES}-line provenance limit.\n\n{stat}"
        )
    return _bounded_text(
        _command_text(
            ["git", "diff", "--no-ext-diff", "--no-color", "HEAD"],
            cwd=repository,
        )
    )


def _capture_provenance(
    run: Path,
    *,
    launch_command: Sequence[str],
    record_command: Sequence[str],
    launch_arguments: Sequence[str],
) -> dict[str, object]:
    provenance = run / "provenance"
    provenance.mkdir()
    repositories: dict[str, object] = {}
    for name, repository in _PROVENANCE_REPOSITORIES.items():
        if not (repository / ".git").exists():
            repositories[name] = {"available": False, "path": str(repository)}
            continue
        commit = _command_text(["git", "rev-parse", "HEAD"], cwd=repository).strip()
        status = _bounded_text(
            _command_text(["git", "status", "--short", "--branch"], cwd=repository)
        )
        diff = _tracked_diff(repository)
        (provenance / f"{name}.commit").write_text(commit + "\n")
        (provenance / f"{name}.status").write_text(status)
        (provenance / f"{name}.diff").write_text(diff)
        untracked = _copy_untracked(repository, provenance / f"{name}.untracked")
        repositories[name] = {
            "available": True,
            "path": str(repository),
            "commit": commit,
            "untracked": untracked,
        }
    (provenance / "nvidia-smi.txt").write_text(_command_text(["nvidia-smi"]))
    realtime = _realtime_status()
    (provenance / "realtime.json").write_text(
        json.dumps(realtime, indent=2, sort_keys=True) + "\n"
    )
    environment = {
        key: os.environ.get(key)
        for key in (
            "ROS_DISTRO",
            "ROS_DOMAIN_ID",
            "RMW_IMPLEMENTATION",
            "CYCLONEDDS_URI",
            "PIXI_ENV",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
        )
    }
    return {
        "schema": "sbmpc_run_v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "backend": _launch_value(launch_arguments, "backend", "mujoco"),
        "launch_arguments": list(launch_arguments),
        "launch_command": list(launch_command),
        "record_command": list(record_command),
        "topics": list(RUN_TOPICS),
        "environment": environment,
        "realtime": realtime,
        "repositories": repositories,
    }


def _tee_launch_output(stream, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        for line in iter(stream.readline, ""):
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
    stream.close()


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_process_group(process_group_id: int, signum: signal.Signals) -> None:
    try:
        os.killpg(process_group_id, signum)
    except ProcessLookupError:
        pass


def _stop_process(
    process: subprocess.Popen,
    *,
    name: str,
    force_requested: threading.Event | None,
    graceful_timeout_sec: float = 30.0,
) -> int:
    process_group_id = process.pid

    def group_exists() -> bool:
        # Reap the group leader promptly; otherwise a zombie leader can make
        # killpg(..., 0) look alive until the full shutdown timeout expires.
        process.poll()
        return _process_group_exists(process_group_id)

    return_code = process.poll()
    if return_code is not None and not group_exists():
        return int(return_code)
    print(f"Stopping {name}...", flush=True)
    _signal_process_group(process_group_id, signal.SIGINT)
    deadline = time.monotonic() + graceful_timeout_sec
    terminated = False
    while group_exists() and time.monotonic() < deadline:
        if force_requested is not None and force_requested.is_set() and not terminated:
            _signal_process_group(process_group_id, signal.SIGTERM)
            terminated = True
            deadline = min(deadline, time.monotonic() + 2.0)
        time.sleep(0.1)
    if group_exists():
        if not terminated:
            _signal_process_group(process_group_id, signal.SIGTERM)
            deadline = time.monotonic() + 10.0
            while group_exists() and time.monotonic() < deadline:
                time.sleep(0.1)
    if group_exists():
        _signal_process_group(process_group_id, signal.SIGKILL)
    if process.poll() is None:
        return int(process.wait())
    return int(process.returncode)


def _write_bag_info(run: Path) -> None:
    (run / "rosbag-info.txt").write_text(
        _command_text(["ros2", "bag", "info", str(run / "rosbag")])
    )


def _postprocess(run: Path) -> dict[str, object]:
    from sbmpc_bringup.run_report import (
        generate_capture_health_report,
        generate_report_from_data,
        read_bag,
    )

    result: dict[str, object] = {
        "report_summary": None,
        "report_error": None,
        "capture_health_report_generated": False,
        "replay_generated": False,
        "replay_error": None,
    }
    try:
        data = read_bag(run)
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        result["report_error"] = message
        try:
            generate_capture_health_report(
                run,
                run / "diagnostic_report",
                full_report_error=message,
            )
            result["capture_health_report_generated"] = True
        except Exception as health_error:
            result["report_error"] = (
                f"{message}; capture-health report failed: "
                f"{type(health_error).__name__}: {health_error}"
            )
    else:
        try:
            result["report_summary"] = generate_report_from_data(
                data,
                run / "diagnostic_report",
            )
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            result["report_error"] = message
            try:
                generate_capture_health_report(
                    run,
                    run / "diagnostic_report",
                    full_report_error=message,
                )
                result["capture_health_report_generated"] = True
            except Exception as health_error:
                result["report_error"] = (
                    f"{message}; capture-health report failed: "
                    f"{type(health_error).__name__}: {health_error}"
                )
        try:
            from sbmpc_bringup.replay import export_replay_json

            export_replay_json(data, run / "replay.json")
            result["replay_generated"] = True
        except Exception as error:
            result["replay_error"] = f"{type(error).__name__}: {error}"

    errors = [
        str(result[key])
        for key in ("report_error", "replay_error")
        if result[key] is not None
    ]
    if errors:
        (run / "postprocess-error.txt").write_text("\n".join(errors) + "\n")
    return result


def _run_capture(
    run: Path,
    *,
    label: str,
    launch_arguments: Sequence[str],
    duration_sec: float,
) -> int:
    backend = _launch_value(launch_arguments, "backend", "mujoco")
    launch_command = _bringup_command(launch_arguments)
    record_command = _record_command(run, label, backend)
    manifest = _capture_provenance(
        run,
        launch_command=launch_command,
        record_command=record_command,
        launch_arguments=launch_arguments,
    )
    _atomic_json(run / "manifest.json", manifest)
    realtime = manifest["realtime"]
    if not realtime["sched_fifo_probe_ok"]:
        print(
            "WARNING: SCHED_FIFO is unavailable in this launch shell; "
            "controller timing in this capture will not represent the intended "
            "realtime deployment. See provenance/realtime.json.",
            file=sys.stderr,
            flush=True,
        )

    environment = os.environ.copy()
    environment["ROS_LOG_DIR"] = str(run / "ros_logs")
    environment.setdefault("RCUTILS_COLORIZED_OUTPUT", "0")
    stop_requested = threading.Event()
    force_requested = threading.Event()

    def request_stop(signum, frame) -> None:
        del signum, frame
        if stop_requested.is_set():
            force_requested.set()
        else:
            stop_requested.set()

    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous_handlers:
        signal.signal(signum, request_stop)

    bag_log = (run / "rosbag-console.log").open("w", encoding="utf-8")
    try:
        bag = subprocess.Popen(
            record_command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=bag_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except Exception as error:
        bag_log.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        message = f"failed to start rosbag recorder: {type(error).__name__}: {error}"
        manifest.update(
            {
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "capture_error": message,
            }
        )
        _atomic_json(run / "manifest.json", manifest)
        (run / "capture-error.txt").write_text(message + "\n")
        print(message, file=sys.stderr)
        return 1
    launch: subprocess.Popen | None = None
    tee_thread: threading.Thread | None = None
    deadline: float | None = None
    launch_return_code: int | None = None
    bag_return_code = 1
    bag_stop_attempted = False
    launch_stop_attempted = False
    user_stopped = False
    capture_error: str | None = None
    try:
        startup_deadline = time.monotonic() + 2.0
        while time.monotonic() < startup_deadline and bag.poll() is None:
            time.sleep(0.05)
        if bag.poll() is not None:
            raise RuntimeError(
                f"rosbag recorder exited during startup with code {bag.returncode}"
            )
        if stop_requested.is_set():
            user_stopped = True
            print("Stop requested before bringup; the robot was not launched.", flush=True)
        else:
            print(f"SB-MPC run directory: {run}", flush=True)
            print(f"Backend: {backend}", flush=True)
            print(f"Bag: {shlex.join(record_command)}", flush=True)
            print(f"Launch: {shlex.join(launch_command)}", flush=True)
            print(
                "Press Ctrl-C once to stop, finalize, and generate the report.",
                flush=True,
            )
            launch = subprocess.Popen(
                launch_command,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                errors="replace",
                start_new_session=True,
            )
            assert launch.stdout is not None
            tee_thread = threading.Thread(
                target=_tee_launch_output,
                args=(launch.stdout, run / "launch-console.log"),
                daemon=True,
            )
            tee_thread.start()
            if duration_sec:
                deadline = time.monotonic() + duration_sec
            while launch.poll() is None:
                if bag.poll() is not None:
                    print(
                        f"rosbag recorder exited unexpectedly with code {bag.returncode}",
                        file=sys.stderr,
                        flush=True,
                    )
                    stop_requested.set()
                if deadline is not None and time.monotonic() >= deadline:
                    stop_requested.set()
                if stop_requested.is_set():
                    user_stopped = bag.poll() is None
                    break
                time.sleep(0.1)
            launch_return_code = _stop_process(
                launch,
                name="Franka bringup",
                force_requested=force_requested,
            )
            launch_stop_attempted = True
        if tee_thread is not None:
            tee_thread.join(timeout=5.0)
        bag_return_code = _stop_process(
            bag,
            name="rosbag recorder",
            force_requested=None,
        )
        bag_stop_attempted = True
    except Exception as error:
        capture_error = f"{type(error).__name__}: {error}"
        print(f"Capture supervisor error: {capture_error}", file=sys.stderr)
    finally:
        if launch is not None and not launch_stop_attempted:
            try:
                launch_stop_attempted = True
                launch_return_code = _stop_process(
                    launch,
                    name="Franka bringup",
                    force_requested=force_requested,
                    graceful_timeout_sec=5.0,
                )
            except Exception as error:
                print(f"Failed to stop Franka bringup cleanly: {error}", file=sys.stderr)
        if not bag_stop_attempted:
            try:
                bag_stop_attempted = True
                bag_return_code = _stop_process(
                    bag,
                    name="rosbag recorder",
                    force_requested=None,
                    graceful_timeout_sec=5.0,
                )
            except Exception as error:
                print(
                    f"Failed to stop rosbag recorder cleanly: {error}",
                    file=sys.stderr,
                )
        if tee_thread is not None:
            tee_thread.join(timeout=5.0)
        bag_log.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    bag_metadata = run / "rosbag" / "metadata.yaml"
    postprocess: dict[str, object] = {
        "report_summary": None,
        "report_error": None,
        "capture_health_report_generated": False,
        "replay_generated": False,
        "replay_error": None,
    }
    if bag_metadata.is_file():
        _write_bag_info(run)
        print("Generating offline plots and replay artifact...", flush=True)
        try:
            postprocess = _postprocess(run)
        except Exception as error:
            postprocess["report_error"] = f"{type(error).__name__}: {error}"
            (run / "postprocess-error.txt").write_text(
                str(postprocess["report_error"]) + "\n"
            )
    else:
        postprocess["report_error"] = "rosbag did not produce metadata.yaml"
        (run / "postprocess-error.txt").write_text(
            str(postprocess["report_error"]) + "\n"
        )

    postprocess_errors = [
        str(postprocess[key])
        for key in ("report_error", "replay_error")
        if postprocess[key] is not None
    ]

    manifest.update(
        {
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "launch_return_code": launch_return_code,
            "rosbag_return_code": bag_return_code,
            "capture_error": capture_error,
            "postprocess_errors": postprocess_errors,
            "full_report_generated": postprocess["report_summary"] is not None,
            "capture_health_report_generated": postprocess[
                "capture_health_report_generated"
            ],
            "replay_generated": postprocess["replay_generated"],
        }
    )
    _atomic_json(run / "manifest.json", manifest)
    print(f"Recorded run: {run}")
    if (
        postprocess["report_summary"] is not None
        or postprocess["capture_health_report_generated"]
    ):
        print(f"Diagnostic report: {run / 'diagnostic_report' / 'index.html'}")
    if postprocess["replay_generated"]:
        print(f"Replay JSON: {run / 'replay.json'}")
    if postprocess_errors:
        print(
            "Post-processing incomplete: " + "; ".join(postprocess_errors),
            file=sys.stderr,
        )

    if (
        capture_error is not None
        or postprocess_errors
        or bag_return_code not in (0, -signal.SIGINT)
    ):
        return 1
    if user_stopped:
        return 0
    return 0 if launch_return_code == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="run")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--path-file", type=Path, default=DEFAULT_PATH_FILE)
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="stop automatically after this many seconds; zero records until Ctrl-C",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="print the launch and rosbag commands without starting a run",
    )
    parser.add_argument(
        "launch_arguments",
        nargs=argparse.REMAINDER,
        help="launch arguments after '--', each written as name:=value",
    )
    args = parser.parse_args(argv)
    try:
        label = _validated_label(args.label)
        launch_arguments = _launch_arguments(args.launch_arguments)
    except ValueError as error:
        parser.error(str(error))
    if args.duration_sec < 0.0:
        parser.error("--duration-sec must be non-negative")
    backend = _launch_value(launch_arguments, "backend", "mujoco")
    if backend not in {"mujoco", "real"}:
        parser.error(f"backend must be mujoco or real, got {backend!r}")
    launch_arguments = _with_backend_defaults(launch_arguments)
    if args.print_command:
        preview = args.output_root.resolve() / f"{label}_<UTC_TIMESTAMP>"
        print(shlex.join(_record_command(preview, label, backend)))
        print(shlex.join(_bringup_command(launch_arguments)))
        return 0
    try:
        capture_lock = _acquire_capture_lock()
    except RuntimeError as error:
        parser.error(str(error))
    try:
        run = _new_run_directory(args.output_root, label)
        _publish_run_path(args.path_file, run)
        return _run_capture(
            run,
            label=label,
            launch_arguments=launch_arguments,
            duration_sec=args.duration_sec,
        )
    finally:
        fcntl.flock(capture_lock.fileno(), fcntl.LOCK_UN)
        capture_lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
