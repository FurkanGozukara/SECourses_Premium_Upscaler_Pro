"""
GPU utility functions shared across all model services.

Provides consistent CUDA device handling for SeedVR2, GAN, RIFE, FlashVSR+.
"""
from __future__ import annotations

import csv
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from io import StringIO
from threading import Lock
from typing import Optional, List, Tuple

_GPU_INFO_LOCK = Lock()
_GPU_INFO_CACHE_TTL_SEC = 30.0
_GPU_INFO_CACHE: tuple[float, List["GPUInfo"]] = (0.0, [])

_CUDA_VER_LOCK = Lock()
_CUDA_VER_CACHE_TTL_SEC = 30.0
_CUDA_VER_CACHE: tuple[float, Optional[str]] = (0.0, None)


def _which_nvidia_smi() -> Optional[str]:
    return shutil.which("nvidia-smi")


def _run_nvidia_smi(args: List[str], timeout: float = 3.0) -> Optional[str]:
    """
    Run `nvidia-smi` and return stdout, or None if unavailable/fails.

    IMPORTANT: `nvidia-smi` talks to NVML and does NOT create a CUDA context.
    This is critical for subprocess-based pipelines: the parent Gradio process
    must not reserve VRAM (~300-800MB) just for "GPU detection".
    """
    exe = _which_nvidia_smi()
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            return None
        out = (proc.stdout or "").strip()
        return out or None
    except Exception:
        return None


def _probe_torch_cuda_gpus_via_subprocess(timeout: float = 8.0) -> List["GPUInfo"]:
    """
    Fallback GPU probe that runs *inside a short-lived Python subprocess*.

    Why this exists:
    - Some Windows setups don't have `nvidia-smi` on PATH even though CUDA works.
    - We still must avoid importing torch / initializing CUDA in the parent Gradio process.

    This subprocess may briefly allocate a CUDA context while probing, but it will be
    released immediately when the subprocess exits (so parent VRAM stays at 0).
    """
    try:
        import json

        code = r"""
import json
try:
    import torch
    if not torch.cuda.is_available():
        print("[]")
    else:
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            # Avoid memory_allocated/reserved; only static properties.
            gpus.append({
                "id": int(i),
                "name": str(props.name),
                "total_memory_gb": float(props.total_memory) / (1024 ** 3),
                "compute_capability": [int(props.major), int(props.minor)],
            })
        print(json.dumps(gpus))
except Exception:
    print("[]")
"""

        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        raw = (proc.stdout or "").strip()
        data = json.loads(raw) if raw else []
        gpus: List[GPUInfo] = []
        if isinstance(data, list):
            for item in data:
                try:
                    cc = item.get("compute_capability")
                    compute_cap = (int(cc[0]), int(cc[1])) if isinstance(cc, list) and len(cc) >= 2 else None
                    gpus.append(
                        GPUInfo(
                            id=int(item.get("id", 0)),
                            name=str(item.get("name", "")),
                            total_memory_gb=float(item.get("total_memory_gb", 0.0)),
                            available_memory_gb=0.0,  # Unknown without NVML; keep 0.0.
                            compute_capability=compute_cap,
                            is_available=True,
                        )
                    )
                except Exception:
                    continue
        return gpus
    except Exception:
        return []


@dataclass
class GPUInfo:
    """GPU information dataclass for health checks"""
    id: int
    name: str
    total_memory_gb: float
    available_memory_gb: float
    compute_capability: Optional[Tuple[int, int]]
    is_available: bool


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M1/M2/M3)"""
    if platform.system() != "Darwin":
        return False
    
    try:
        # Check for ARM64 architecture
        import subprocess
        result = subprocess.run(["uname", "-m"], capture_output=True, text=True)
        return "arm64" in result.stdout.lower()
    except Exception:
        return False


def get_cuda_version() -> Optional[str]:
    """
    Get CUDA version string without initializing a CUDA context in this process.

    We parse the `nvidia-smi` header (e.g. "CUDA Version: 12.2"). This keeps the
    parent (Gradio) process VRAM at 0 for subprocess-based pipelines.
    """
    global _CUDA_VER_CACHE
    now = time.time()
    with _CUDA_VER_LOCK:
        ts, cached = _CUDA_VER_CACHE
        if now - ts < _CUDA_VER_CACHE_TTL_SEC:
            return cached

    out = _run_nvidia_smi([], timeout=2.0)
    cuda_ver: Optional[str] = None
    if out:
        m = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", out)
        if m:
            cuda_ver = m.group(1)

    with _CUDA_VER_LOCK:
        _CUDA_VER_CACHE = (now, cuda_ver)
    return cuda_ver


def get_gpu_info() -> List[GPUInfo]:
    """
    Get detailed information about all available NVIDIA GPUs WITHOUT creating a CUDA context.

    Uses `nvidia-smi --query-gpu=...` (NVML) to avoid persistent VRAM reservations
    in the parent (Gradio) process.
    """
    global _GPU_INFO_CACHE
    now = time.time()
    with _GPU_INFO_LOCK:
        ts, cached = _GPU_INFO_CACHE
        if now - ts < _GPU_INFO_CACHE_TTL_SEC:
            return cached

    # Prefer including compute capability if supported by installed nvidia-smi.
    query_fields_sets = [
        ["index", "name", "memory.total", "memory.free", "compute_cap"],
        ["index", "name", "memory.total", "memory.free"],
        ["index", "name", "memory.total"],
    ]

    gpus: List[GPUInfo] = []
    for fields in query_fields_sets:
        out = _run_nvidia_smi(
            [f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            timeout=3.0,
        )
        if not out:
            continue

        try:
            rows = list(csv.reader(StringIO(out)))
        except Exception:
            continue

        parsed: List[GPUInfo] = []
        ok = True
        for row in rows:
            if not row:
                continue
            row = [c.strip() for c in row]
            try:
                idx = int(row[0])
                name = row[1] if len(row) > 1 else f"GPU {idx}"

                total_mb = float(row[2]) if len(row) > 2 and row[2] else 0.0
                free_mb = float(row[3]) if len(row) > 3 and row[3] else 0.0

                compute_cap: Optional[Tuple[int, int]] = None
                if "compute_cap" in fields and len(row) > 4 and row[4]:
                    parts = row[4].split(".")
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                        compute_cap = (int(parts[0]), int(parts[1]))

                parsed.append(
                    GPUInfo(
                        id=idx,
                        name=name,
                        total_memory_gb=total_mb / 1024.0 if total_mb else 0.0,
                        available_memory_gb=free_mb / 1024.0 if free_mb else 0.0,
                        compute_capability=compute_cap,
                        is_available=True,
                    )
                )
            except Exception:
                ok = False
                break

        if ok:
            gpus = parsed
            break

    # Fallback: if nvidia-smi is missing, try a subprocess torch probe.
    # This keeps the parent process VRAM clean (subprocess exits => VRAM freed).
    if not gpus and _which_nvidia_smi() is None:
        gpus = _probe_torch_cuda_gpus_via_subprocess()

    with _GPU_INFO_LOCK:
        _GPU_INFO_CACHE = (now, gpus)
    return gpus


def expand_cuda_device_spec(cuda_spec: str) -> str:
    """
    Expand 'all' to actual device list.
    
    Supports:
    - "0" -> "0" (single GPU)
    - "0,1,2" -> "0,1,2" (explicit multi-GPU)
    - "all" -> "0,1,2,3" (auto-detect all GPUs)
    - Whitespace-tolerant: "0, 1, 2" -> "0,1,2"
    
    Args:
        cuda_spec: User-provided CUDA device specification
        
    Returns:
        Normalized device specification (comma-separated GPU IDs)
    """
    spec_clean = str(cuda_spec).strip().lower()

    # Handle "all" keyword (NVML-based device count; avoids torch import).
    if spec_clean == "all":
        gpus = get_gpu_info()
        if gpus:
            return ",".join(str(i) for i in range(len(gpus)))
        # If no GPUs detected, leave keyword for validation to report properly.
        return "all"

    # Normalize whitespace in device lists: "0, 1, 2" -> "0,1,2"
    devices = [d.strip() for d in str(cuda_spec).replace(" ", "").split(",") if d.strip()]
    return ",".join(devices)


def validate_cuda_device_spec(cuda_spec: str) -> Optional[str]:
    """
    Validate CUDA device specification.
    
    Checks:
    - CUDA availability
    - Device ID validity (numeric and within range)
    - Multi-GPU compatibility warnings
    
    Args:
        cuda_spec: Device specification (already expanded via expand_cuda_device_spec)
        
    Returns:
        Error message if invalid, None if valid
    """
    if not cuda_spec or str(cuda_spec).strip() == "":
        return None  # Empty is valid (CPU fallback)

    spec = str(cuda_spec).strip()
    if spec.lower() == "all":
        gpus = get_gpu_info()
        if not gpus:
            return "No CUDA GPUs detected (nvidia-smi unavailable or no NVIDIA GPU)."
        return None

    gpus = get_gpu_info()
    device_count = len(gpus)
    if device_count == 0:
        return "No CUDA GPUs detected (nvidia-smi unavailable or no NVIDIA GPU), but CUDA devices were specified."

    devices = [d.strip() for d in spec.split(",") if d.strip()]
    invalid: List[str] = []
    for d in devices:
        if not d.isdigit():
            invalid.append(f"{d} (not numeric)")
            continue
        did = int(d)
        if did < 0 or did >= device_count:
            invalid.append(f"{d} (max: {device_count - 1})")

    if invalid:
        return f"Invalid CUDA device ID(s): {', '.join(invalid)}. Available devices: 0-{device_count-1}"

    return None


def clear_cuda_cache():
    """
    Clear CUDA cache to free VRAM.
    
    Safe to call even if CUDA unavailable (silently ignored).
    Used after subprocess completion or model switches.
    """
    try:
        # IMPORTANT: do NOT import torch here. Importing torch in the parent Gradio
        # process can permanently increase RAM usage, and any accidental CUDA init
        # will reserve hundreds of MB of VRAM for the life of the UI process.
        torch = sys.modules.get("torch")
        if torch is None:
            return
        if not hasattr(torch, "cuda"):
            return
        # IMPORTANT:
        # - On Windows, initializing a CUDA context in the *main* Gradio process can
        #   reserve a few hundred MB of VRAM (often ~300-800MB) even if no model is loaded.
        # - In subprocess-based pipelines (SeedVR2), we do NOT want to create a CUDA
        #   context in the parent process "just to clear cache" because it can look
        #   like leftover VRAM after cancel.
        #
        # So we only clear if CUDA is already initialized in THIS process.
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_initialized"):
            if not torch.cuda.is_initialized():
                return
        # If CUDA isn't available, empty_cache would raise; keep it best-effort.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()  # Wait for GPU operations to finish
            except Exception:
                # If no device/context is active, synchronize can fail; ignore.
                pass
    except Exception:
        pass  # Silently ignore if CUDA not available


def get_cuda_memory_info() -> str:
    """
    Get human-readable CUDA memory usage summary.
    
    Returns:
        Formatted string with memory stats for each GPU, or empty if unavailable
    """
    out = _run_nvidia_smi(
        ["--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"],
        timeout=3.0,
    )
    if not out:
        return "CUDA not available"
    try:
        rows = list(csv.reader(StringIO(out)))
        lines: List[str] = []
        for row in rows:
            if not row:
                continue
            row = [c.strip() for c in row]
            idx = int(row[0])
            name = row[1] if len(row) > 1 else f"GPU {idx}"
            used_mb = float(row[2]) if len(row) > 2 and row[2] else 0.0
            total_mb = float(row[3]) if len(row) > 3 and row[3] else 0.0
            lines.append(
                f"GPU {idx} ({name}): {used_mb/1024.0:.2f}GB used, {total_mb/1024.0:.2f}GB total"
            )
        return "\n".join(lines) if lines else "No CUDA devices found"
    except Exception as e:
        return f"Memory info unavailable: {e}"


def check_gpu_compatibility(model_type: str) -> tuple[bool, str]:
    """
    Check if GPU acceleration is available and compatible.
    
    Args:
        model_type: Model type for specific warnings (seedvr2, gan, rife, flashvsr)
        
    Returns:
        (is_compatible, warning_message) tuple
    """
    gpus = get_gpu_info()
    if not gpus:
        return False, "CUDA not available (no NVIDIA GPUs detected via nvidia-smi) - processing will use CPU (significantly slower)"

    device_count = len(gpus)
    compute_cap = gpus[0].compute_capability

    warnings: List[str] = []
    if compute_cap:
        # Flash attention requires Ampere+ (8.0) for optimal performance
        if model_type in ("seedvr2", "flashvsr") and compute_cap[0] < 8:
            warnings.append(
                f"GPU compute capability {compute_cap[0]}.{compute_cap[1]} detected. "
                f"Flash attention works best on Ampere (8.0+) or newer GPUs."
            )

        # torch.compile requires recent GPU
        if compute_cap[0] < 7:
            warnings.append(
                f"GPU compute capability {compute_cap[0]}.{compute_cap[1]} is very old. "
                f"torch.compile may not work properly. Consider upgrading GPU."
            )

    if warnings:
        return True, "\n".join(warnings)

    return True, f"✅ {device_count} CUDA GPU(s) detected"
