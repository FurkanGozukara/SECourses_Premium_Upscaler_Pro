"""Small, dependency-light helpers for stopping subprocess worker trees."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from contextlib import suppress
from typing import Any


def terminate_process_tree(proc: Any, grace_sec: float = 1.0) -> None:
    """Stop a launcher and descendants without leaving GPU workers orphaned."""
    if proc is None:
        return

    descendants = []
    with suppress(Exception):
        import psutil

        descendants = psutil.Process(proc.pid).children(recursive=True)

    # Stop workers first. On Windows a venv python.exe launcher can otherwise
    # exit and orphan the real interpreter while it is still holding VRAM.
    for child in reversed(descendants):
        with suppress(Exception):
            child.terminate()
    with suppress(Exception):
        proc.terminate()

    deadline = time.time() + max(0.0, float(grace_sec))
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.05)

    for child in reversed(descendants):
        with suppress(Exception):
            if child.is_running():
                child.kill()

    if proc.poll() is None:
        if os.name == "nt":
            with suppress(Exception):
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3.0,
                    check=False,
                )
        else:
            with suppress(Exception):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        with suppress(Exception):
            proc.kill()
    with suppress(Exception):
        proc.wait(timeout=1.0)
