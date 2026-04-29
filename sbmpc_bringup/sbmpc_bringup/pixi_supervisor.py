from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


DEFAULT_PIXI_ROS_RUN = "/workspace/sbmpc_containers/scripts/pixi_ros_run.sh"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    runner = os.environ.get("SBMPC_PIXI_ROS_RUN_SCRIPT", DEFAULT_PIXI_ROS_RUN)
    child = subprocess.Popen([runner, *args], start_new_session=True)
    shutting_down = False

    def handle_shutdown(signum, frame) -> None:
        nonlocal shutting_down
        del frame
        if shutting_down:
            os._exit(0)
        shutting_down = True
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        del signum
        _stop_child_group(child)
        os._exit(0)

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    try:
        return child.wait()
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def _stop_child_group(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return

    _send_process_group(child, signal.SIGTERM)
    if _wait_for_exit(child, _env_float("SBMPC_SUPERVISOR_TERM_GRACE_SEC", 1.5)):
        return

    _send_process_group(child, signal.SIGKILL)
    _wait_for_exit(child, 1.0)


def _send_process_group(child: subprocess.Popen, signum: int) -> None:
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        return


def _wait_for_exit(child: subprocess.Popen, timeout_sec: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_sec)
    while child.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    return child.poll() is not None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


if __name__ == "__main__":
    raise SystemExit(main())
