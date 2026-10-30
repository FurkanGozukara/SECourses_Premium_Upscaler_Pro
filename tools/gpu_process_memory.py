"""Per-process dedicated and shared GPU memory on Windows (PDH "GPU Process Memory" counters).

nvidia-smi reports N/A for per-process memory under WDDM, and the PyTorch allocator does not see
the CUDA context, kernels or SDK allocations. The Windows counters do, and their "Shared Usage"
shows whether the driver spilled device allocations into system RAM (sysmem fallback).

    sampler = ProcessGpuMemorySampler(pid_getter, interval=0.2)
    sampler.start(); ...; sampler.stop()
    sampler.peak_dedicated_gb, sampler.peak_shared_gb

A pid getter (instead of a pid) lets the sampler follow a launcher's child processes:
every pid it returns is summed.

Operator/measurement tool; the app does not import it.
"""

from __future__ import annotations

import ctypes
import os
import re
import threading
import time
from ctypes import wintypes
from typing import Callable, Dict, Iterable, List, Optional, Tuple

GIB = 1024 ** 3

_pdh = ctypes.WinDLL("pdh.dll") if os.name == "nt" else None
PDH_FMT_LARGE = 0x00000400
PDH_MORE_DATA = 0x800007D2
ERROR_SUCCESS = 0


class _FmtValue(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("largeValue", ctypes.c_longlong)]


class _FmtItem(ctypes.Structure):
    _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _FmtValue)]


class _Query:
    def __init__(self, paths: Iterable[str]):
        self.handle = wintypes.HANDLE()
        status = _pdh.PdhOpenQueryW(None, None, ctypes.byref(self.handle))
        if status != ERROR_SUCCESS:
            raise OSError(f"PdhOpenQueryW failed: 0x{status & 0xFFFFFFFF:08X}")
        self.counters: Dict[str, wintypes.HANDLE] = {}
        for path in paths:
            counter = wintypes.HANDLE()
            status = _pdh.PdhAddEnglishCounterW(self.handle, path, None, ctypes.byref(counter))
            if status != ERROR_SUCCESS:
                raise OSError(f"PdhAddEnglishCounterW({path}) failed: 0x{status & 0xFFFFFFFF:08X}")
            self.counters[path] = counter
        _pdh.PdhCollectQueryData(self.handle)

    def collect(self) -> Dict[str, Dict[str, int]]:
        _pdh.PdhCollectQueryData(self.handle)
        out: Dict[str, Dict[str, int]] = {}
        for path, counter in self.counters.items():
            size = wintypes.DWORD(0)
            count = wintypes.DWORD(0)
            status = _pdh.PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), None)
            if (status & 0xFFFFFFFF) != PDH_MORE_DATA:
                out[path] = {}
                continue
            buf = (ctypes.c_byte * size.value)()
            status = _pdh.PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), buf)
            values: Dict[str, int] = {}
            if status == ERROR_SUCCESS:
                items = ctypes.cast(buf, ctypes.POINTER(_FmtItem))
                for i in range(count.value):
                    values[items[i].szName] = int(items[i].FmtValue.largeValue)
            out[path] = values
        return out

    def close(self) -> None:
        _pdh.PdhCloseQuery(self.handle)


DEDICATED = r"\GPU Process Memory(*)\Dedicated Usage"
SHARED = r"\GPU Process Memory(*)\Shared Usage"
ADAPTER_DEDICATED = r"\GPU Adapter Memory(*)\Dedicated Usage"
_PID_RE = re.compile(r"pid_(\d+)_luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)_phys_(\d+)")


def _per_pid(values: Dict[str, int], luid: Optional[str]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for name, value in values.items():
        m = _PID_RE.search(name)
        if not m:
            continue
        if luid and m.group(2).lower() != luid.lower():
            continue
        pid = int(m.group(1))
        out[pid] = out.get(pid, 0) + int(value)
    return out


def adapter_luids() -> Dict[str, int]:
    """LUID -> dedicated usage of each adapter (to find the LUID of a GPU by its usage)."""
    q = _Query([ADAPTER_DEDICATED])
    try:
        time.sleep(0.05)
        vals = q.collect()[ADAPTER_DEDICATED]
    finally:
        q.close()
    out: Dict[str, int] = {}
    for name, value in vals.items():
        m = re.search(r"luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)_phys_(\d+)", name)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0) + int(value)
    return out


class ProcessGpuMemorySampler:
    def __init__(self, pids: Callable[[], Iterable[int]], interval: float = 0.2, luid: Optional[str] = None):
        self.pids = pids
        self.interval = float(interval)
        self.luid = luid
        self.peak_dedicated = 0
        self.peak_shared = 0
        self.first_shared: Optional[int] = None
        self.samples: List[Tuple[float, int, int]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        q = _Query([DEDICATED, SHARED])
        t0 = time.time()
        try:
            while not self._stop.is_set():
                vals = q.collect()
                wanted = set(int(p) for p in (self.pids() or []))
                ded = sum(v for p, v in _per_pid(vals.get(DEDICATED, {}), self.luid).items() if p in wanted)
                sh = sum(v for p, v in _per_pid(vals.get(SHARED, {}), self.luid).items() if p in wanted)
                if wanted:
                    if self.first_shared is None and ded > 0:
                        self.first_shared = sh
                    self.peak_dedicated = max(self.peak_dedicated, ded)
                    self.peak_shared = max(self.peak_shared, sh)
                    self.samples.append((time.time() - t0, ded, sh))
                self._stop.wait(self.interval)
        finally:
            q.close()

    def start(self) -> "ProcessGpuMemorySampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def peak_dedicated_gb(self) -> float:
        return self.peak_dedicated / GIB

    @property
    def peak_shared_gb(self) -> float:
        return self.peak_shared / GIB

    @property
    def shared_growth_gb(self) -> float:
        """Shared usage above the first sample (pinned host buffers start early; a spill grows late)."""
        return max(0, self.peak_shared - int(self.first_shared or 0)) / GIB


def process_tree_pids(root_pid: int) -> List[int]:
    """root + all descendants (psutil)."""
    try:
        import psutil

        root = psutil.Process(int(root_pid))
        return [root.pid] + [c.pid for c in root.children(recursive=True)]
    except Exception:
        return [int(root_pid)]


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(adapter_luids(), indent=1))
    if len(sys.argv) > 1:
        pid = int(sys.argv[1])
        s = ProcessGpuMemorySampler(lambda: process_tree_pids(pid)).start()
        time.sleep(3)
        s.stop()
        print(f"pid {pid}: dedicated {s.peak_dedicated_gb:.2f} GB, shared {s.peak_shared_gb:.2f} GB")
