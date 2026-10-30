"""Calibration-only hook: every Python process started with this folder on PYTHONPATH and
SECOURSES_VRAM_PROFILE_OUT set appends its PyTorch peak VRAM to that JSONL file at exit.

Used by tools/vram_calibrate.py to measure engine phases (each SeedVR2 / SparkVSR stage runs
in its own child process). The app never puts this folder on PYTHONPATH.
"""

import atexit
import json
import os
import sys
import time

_OUT = os.environ.get("SECOURSES_VRAM_PROFILE_OUT")
_T0 = time.time()


def _dump():
    try:
        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return
        gib = 1024 ** 3
        rec = {
            "pid": os.getpid(),
            "argv": [os.path.basename(a) if i == 0 else a for i, a in enumerate(sys.argv[:12])],
            "seconds": round(time.time() - _T0, 2),
            "devices": [],
        }
        for d in range(torch.cuda.device_count()):
            try:
                rec["devices"].append({
                    "device": d,
                    "peak_reserved_gb": round(torch.cuda.max_memory_reserved(d) / gib, 4),
                    "peak_allocated_gb": round(torch.cuda.max_memory_allocated(d) / gib, 4),
                })
            except Exception:
                pass
        with open(_OUT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass


if _OUT:
    atexit.register(_dump)
