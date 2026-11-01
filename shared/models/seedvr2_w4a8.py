"""W4A8 7B Sharp, the default SeedVR2 model, and its automatic INT8 ConvRot fallback.

W4A8 runs on comfy_kitchen's CUDA kernels, which need compute capability 8.0 or newer (RTX 30 series
and later). On an older GPU, in CPU mode, or when the comfy_kitchen CUDA backend cannot load, a run
that selects W4A8 switches to the 7B Sharp INT8 ConvRot path (the same 7B Sharp model with INT8
ConvRot weights) and Auto Tune plans with the INT8 numbers. Presets that name another model keep it.

SECOURSES_FORCE_W4A8_FALLBACK=1 forces the fallback (tests).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

W4A8_MODEL = "seedvr2_7b_sharp_convrot_w4a8.safetensors"
W4A8_FALLBACK_MODEL = "seedvr2_ema_7b_sharp_fp16.safetensors"  # run with INT8 ConvRot on
W4A8_MIN_COMPUTE_CAPABILITY = (8, 0)
ENV_FORCE_FALLBACK = "SECOURSES_FORCE_W4A8_FALLBACK"

_BACKEND_ERROR: Dict[str, Optional[str]] = {}


def is_w4a8_model(model_name: Any) -> bool:
    return "w4a8" in Path(str(model_name or "")).name.lower()


def comfy_kitchen_backend_error() -> Optional[str]:
    """None when comfy_kitchen's CUDA backend (with the W4A8 kernel) loads, else the reason.

    Checked once per process in a hidden child process with no GPU visible, so the app process
    never loads CUDA libraries or creates a CUDA context for this check."""
    if "value" in _BACKEND_ERROR:
        return _BACKEND_ERROR["value"]
    code = (
        "import comfy_kitchen.backends.cuda as c\n"
        "ok = bool(getattr(c, '_EXT_AVAILABLE', False)) and hasattr(c, 'w4a8_int8_linear')\n"
        "print('W4A8_OK' if ok else 'W4A8_ERR ' + str(getattr(c, '_EXT_ERROR', None) or 'no W4A8 kernel'))\n"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    error: Optional[str]
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip().startswith("W4A8_")]
        if lines and lines[-1] == "W4A8_OK":
            error = None
        elif lines:
            error = "comfy_kitchen CUDA backend: " + lines[-1][len("W4A8_ERR "):]
        else:
            tail = [ln for ln in (proc.stderr or "").splitlines() if ln.strip()]
            error = "comfy_kitchen could not be loaded: " + (tail[-1].strip() if tail else f"exit code {proc.returncode}")
    except Exception:  # the check itself could not run (timeout): the engine checks again at load
        return None
    _BACKEND_ERROR["value"] = error
    return error


def _device_ids(device_spec: Any) -> List[int]:
    from shared.vram_autotune.common import parse_device_ids, pick_gpu_index

    ids = parse_device_ids(device_spec)
    if ids:
        return ids
    idx = pick_gpu_index(device_spec)
    return [idx] if idx is not None else []


def _compute_capabilities() -> Dict[int, Optional[Tuple[int, int]]]:
    from shared.gpu_utils import get_gpu_info

    return {int(g.id): g.compute_capability for g in (get_gpu_info() or [])}


def w4a8_fallback_reason(device_spec: Any) -> Optional[str]:
    """Why W4A8 cannot run on `device_spec` (a GPU index, list, '' for auto, or 'cpu'); None when it can."""
    if str(os.environ.get(ENV_FORCE_FALLBACK, "")).strip().lower() in {"1", "true", "yes", "on"}:
        return f"forced by {ENV_FORCE_FALLBACK}"
    spec = str(device_spec or "").strip().lower()
    if spec in {"cpu", "none"}:
        return "CPU mode (the W4A8 kernels run on NVIDIA GPUs only)"
    ids = _device_ids(device_spec)
    if not ids:
        return "no NVIDIA GPU detected (the W4A8 kernels run on NVIDIA GPUs only)"
    caps = _compute_capabilities()
    need = W4A8_MIN_COMPUTE_CAPABILITY
    for idx in ids:
        cap = caps.get(idx)
        if cap is not None and tuple(cap) < need:
            return (f"GPU {idx} has compute capability {cap[0]}.{cap[1]}; the W4A8 kernels need "
                    f"{need[0]}.{need[1]} or newer (RTX 30 series or later)")
    return comfy_kitchen_backend_error()


def effective_model(model_name: Any, int8_convrot: bool, device_spec: Any) -> Tuple[str, bool, Optional[str]]:
    """(model, int8_convrot, fallback reason or None) a run of `model_name` really uses on `device_spec`."""
    name = str(model_name or "")
    if not is_w4a8_model(name):
        return name, bool(int8_convrot), None
    reason = w4a8_fallback_reason(device_spec)
    if reason is None:
        return name, False, None
    return W4A8_FALLBACK_MODEL, True, reason


def fallback_message(reason: str) -> str:
    return (f"[SeedVR2] W4A8 7B Sharp cannot run here ({reason}); using the 7B Sharp INT8 ConvRot model "
            f"instead ({W4A8_FALLBACK_MODEL} + INT8 ConvRot).")


def apply_w4a8_fallback(settings: Dict[str, Any], device_spec: Any) -> Optional[str]:
    """Switch `settings` from W4A8 to the INT8 ConvRot fallback when needed; returns the log line."""
    model, int8, reason = effective_model(settings.get("dit_model"), bool(settings.get("int8_convrot")), device_spec)
    if reason is None:
        return None
    settings["dit_model"] = model
    settings["int8_convrot"] = int8
    settings["_w4a8_fallback_reason"] = reason
    return fallback_message(reason)
