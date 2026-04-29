from __future__ import annotations

import signal
import subprocess
import sys
import time

from sbmpc_bringup import pixi_supervisor


def test_stop_child_group_kills_stubborn_process_group(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SBMPC_SUPERVISOR_TERM_GRACE_SEC", "0.05")
    ready_file = tmp_path / "ready"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import pathlib, signal, sys, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "pathlib.Path(sys.argv[1]).write_text('ready'); "
                "time.sleep(30)"
            ),
            str(ready_file),
        ],
        start_new_session=True,
    )

    try:
        deadline = time.monotonic() + 1.0
        while not ready_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready_file.exists()
        pixi_supervisor._stop_child_group(child)
        assert child.wait(timeout=1.0) == -signal.SIGKILL
    finally:
        if child.poll() is None:
            child.kill()
