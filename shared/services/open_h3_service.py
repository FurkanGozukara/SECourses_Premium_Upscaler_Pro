"""
Open-H3 Regenerate Service Module
Handles Open-H3 Regenerate processing logic, presets, and callbacks.

Open-H3 Regenerate (NVIDIA Open-H3-Regenerate-2K on MiniMax H3) regenerates a
low-resolution video natively at 1080P / 2K / 4K in three diffusion steps. The
engine (tools/open_h3_inference.py) plans its own 17k+5-frame windows and
prefers the whole clip as one window, so app-level scene chunking is OFF by
default here and only used when the user explicitly configures fixed chunking
in the Resolution tab.

Mirrors shared/services/ltx25_service.py for every shared app behavior.
"""

import hashlib
import json
import logging
import os
import queue
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr

from shared.preset_manager import PresetManager
from shared.path_utils import (
    normalize_path,
    get_media_dimensions,
    get_media_duration_seconds,
    get_media_fps,
    detect_input_type,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    list_directory_entries_sorted,
    resolve_batch_output_dir,
)
from shared.logging_utils import RunLogger
from shared.comparison_unified import create_unified_comparison
from shared.open_h3_constants import (
    OPENH3_DEFAULT_SEED,
    OPENH3_DIT_DEFAULT,
    OPENH3_DIT_INT8,
    OPENH3_DIT_NAMES,
    OPENH3_JXL_MAX_SIDE_DEFAULT,
    OPENH3_JXL_MAX_SIDES,
    OPENH3_JXL_VIDEO_MESSAGE,
    OPENH3_TE_NAMES,
    OPENH3_TE_NVFP4,
    OPENH3_VAE_NAMES,
    OPENH3_VAE_SOL_FP16,
    OPENH3_WORKFLOW_JXL,
    OPENH3_WORKFLOW_REGEN,
    is_jxl_workflow,
)
from shared.gpu_utils import get_global_gpu_override
from shared.vram_autotune.common import DEFAULT_SAVE_VRAM_GB, clamp_save_vram_gb
from shared.oom_alert import clear_vram_oom_alert, maybe_set_vram_oom_alert, show_vram_oom_modal
from shared.output_run_manager import (
    prepare_single_video_run,
    batch_item_dir,
    resolve_resume_input_from_run_dir,
    finalize_run_context,
)
from shared.batch_output_cleanup import keep_only_batch_outputs
from shared.global_rife import maybe_apply_global_rife
from shared.comparison_video_service import maybe_generate_input_vs_output_comparison
from shared.chunk_preview import build_chunk_preview_payload

logger = logging.getLogger("OpenH3Service")

# Cancel event for Open-H3 processing
_open_h3_cancel_event = threading.Event()

OPENH3_DEFAULT_PROMPT = (
    "High-quality, sharp and detailed footage with natural fine textures and crisp, clean edges. Every subject, "
    "object and background element keeps its original identity, shape, design, colors and position, with the same "
    "camera framing, motion and lighting; nothing new is added or removed."
)
# NVIDIA's release grid starts at 0.909375 and redraws a lot (identities and designs can change); the shift-6
# grid from 0.8 keeps the input's content while still adding real detail, so the app defaults to it.
OPENH3_DEFAULT_START_SIGMA = 0.8

OPENH3_RESOLUTION_PRESETS = [
    "2K (1440p)",
    "1080p",
    "4K (2160p)",
    "768p (refine only)",
    "Custom upscale factor",
]
OPENH3_DRAFT_SIZES = [
    "Auto (768p for 720p+ sources, 576p below)",
    "768p (H3 drafts, 1344x768 class)",
    "576p (low-res sources, 1024x576 class)",
    "Native (input size on the 32 px grid)",
]
OPENH3_FPS_MODES = [
    "Keep source frames (1:1, exact timing)",
    "Resample to 24 fps (H3 reference)",
]
OPENH3_AUDIO_MODES = [
    "Encoded silence (NVIDIA reference)",
    "Source soundtrack",
]
OPENH3_SINK_OPTIONS = ["exact_kv_and_rows", "exact_kv", "off"]
OPENH3_TOKEN_BUDGET_MODE_AUTO = "Auto (from free VRAM)"
OPENH3_TOKEN_BUDGET_MODE_MANUAL = "Manual"
OPENH3_TOKEN_BUDGET_MODES = [OPENH3_TOKEN_BUDGET_MODE_AUTO, OPENH3_TOKEN_BUDGET_MODE_MANUAL]
OPENH3_ATTENTION_OPTIONS = ["auto", "sage", "flash", "sdpa"]


def run_open_h3(*args, **kwargs):
    """Lazy-import the Open-H3 runner so the Gradio process stays backend-light."""
    from shared.open_h3_runner import run_open_h3 as _run_open_h3_impl

    return _run_open_h3_impl(*args, **kwargs)


def run_open_h3_plan(*args, **kwargs):
    """Lazy-import the plan-only helper (geometry / window plan preview, no sampling)."""
    from shared.open_h3_runner import run_open_h3_plan as _run_open_h3_plan_impl

    return _run_open_h3_plan_impl(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Live window-plan preview (in-process, instant)
# --------------------------------------------------------------------------- #
_PLANNER_MODULE_CACHE: Dict[str, Any] = {}
_PLAN_PROBE_CACHE: Dict[str, Tuple[Tuple[float, int], Tuple[int, int, int, float, bool]]] = {}
_PLAN_VRAM_CACHE: Dict[str, Any] = {"ts": 0.0, "gpus": []}


def _planner_module(base_dir: Path):
    """Import the engine's planner math by file path (pure Python, no torch)."""
    key = str(Path(base_dir).resolve())
    module = _PLANNER_MODULE_CACHE.get(key)
    if module is not None:
        return module
    import importlib.util
    import sys

    planner_path = Path(base_dir) / "OpenH3" / "open_h3_engine" / "planner.py"
    spec = importlib.util.spec_from_file_location("_open_h3_live_planner", str(planner_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["_open_h3_live_planner"] = module
    spec.loader.exec_module(module)
    _PLANNER_MODULE_CACHE[key] = module
    return module


def fps_mode_value(label: Any) -> str:
    return "24" if "24" in str(label or "") and "resample" in str(label or "").lower() else "source"


def audio_mode_value(label: Any) -> str:
    return "source" if "source" in str(label or "").lower() else "silence"

def _probe_video_for_plan(path: str) -> Optional[Tuple[int, int, int, float, bool]]:
    """ffprobe (width, height, frames, fps, frames_estimated); mtime+size cached."""
    try:
        stat = Path(path).stat()
        cache_key = str(Path(path).resolve())
        fingerprint = (float(stat.st_mtime), int(stat.st_size))
    except OSError:
        return None
    cached = _PLAN_PROBE_CACHE.get(cache_key)
    if cached and cached[0] == fingerprint:
        return cached[1]

    import subprocess

    ffprobe = shutil.which("ffprobe") or "ffprobe"
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,nb_frames,r_frame_rate,avg_frame_rate,duration",
                "-show_entries", "format=duration",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=20,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )
        data = json.loads(result.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)

        def _rate(text: str) -> float:
            try:
                num, _, den = str(text or "").partition("/")
                value = float(num) / float(den or 1)
                return value if value > 0 else 0.0
            except (ValueError, ZeroDivisionError):
                return 0.0

        fps = _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate")) or 30.0
        frames_estimated = False
        try:
            frames = int(stream.get("nb_frames") or 0)
        except (TypeError, ValueError):
            frames = 0
        if frames <= 0:
            duration = 0.0
            for source in (stream.get("duration"), (data.get("format") or {}).get("duration")):
                try:
                    duration = float(source or 0)
                except (TypeError, ValueError):
                    duration = 0.0
                if duration > 0:
                    break
            frames = int(round(duration * fps)) if duration > 0 else 0
            frames_estimated = True
        if width <= 0 or height <= 0 or frames <= 0:
            return None
        info = (width, height, frames, fps, frames_estimated)
        _PLAN_PROBE_CACHE[cache_key] = (fingerprint, info)
        return info
    except Exception:
        return None


def _free_vram_bytes_for_plan(device_value: str) -> Tuple[float, str]:
    """Free VRAM of the GPU the run will use, via nvidia-smi (no CUDA context); 3s cache."""
    from shared.vram_autotune.common import pick_gpu_index, query_gpu_memory

    now = time.time()
    if now - float(_PLAN_VRAM_CACHE.get("ts", 0.0)) > 3.0:
        _PLAN_VRAM_CACHE["gpus"] = query_gpu_memory()
        _PLAN_VRAM_CACHE["ts"] = now
    gpus = _PLAN_VRAM_CACHE.get("gpus") or {}
    idx = pick_gpu_index(device_value, gpus) if gpus else None
    if idx is None:
        return 0.0, "no NVIDIA GPU detected"
    free_gb = float(gpus[idx].free_gb)
    return free_gb * (1024.0 ** 3), f"GPU {idx} free {free_gb:.1f}GB"


def build_live_plan_text(settings: Dict[str, Any], base_dir: Path) -> str:
    """Geometry + window plan computed instantly in-process (same math as the engine)."""
    from shared.open_h3_runner import _draft_size, compute_target_size

    input_path = normalize_path(settings.get("_effective_input_path") or settings.get("input_path") or "")
    if not input_path or not Path(input_path).exists():
        return "Set an Input Path (or upload a video / image), then the live plan appears here."
    kind = detect_input_type(input_path)
    planner = _planner_module(base_dir)
    lines: List[str] = []
    if kind == "image":
        dims = get_media_dimensions(input_path)
        if not dims:
            return "[ERROR] Could not read the image size."
        width, height, total_frames, fps, frames_estimated = int(dims[0]), int(dims[1]), 1, 24.0, False
    elif kind == "video":
        probe = _probe_video_for_plan(input_path)
        if probe is None:
            return "[ERROR] Could not read video dimensions/frames (ffprobe failed)."
        width, height, total_frames, fps, frames_estimated = probe
    else:
        return "[ERROR] Open-H3 needs a video or an image input."

    start_frame = max(0, int(settings.get("start_frame") or 0))
    end_frame = int(settings.get("end_frame") if settings.get("end_frame") is not None else -1)
    if kind == "video":
        if end_frame >= 0:
            total_frames = max(0, min(total_frames, end_frame + 1) - start_frame)
        else:
            total_frames = max(0, total_frames - start_frame)
        if total_frames <= 0:
            return "[ERROR] Start/End Frame selection leaves no frames to process."
        if fps_mode_value(settings.get("fps_mode")) == "24" and abs(fps - 24.0) > 1e-3:
            total_frames = max(1, int(round(total_frames / fps * 24.0)))
            fps = 24.0

    target_w, target_h, size_note = compute_target_size(width, height, settings)
    if is_jxl_workflow(settings):
        what = "image" if kind == "image" else f"{total_frames} frame(s) @ {fps:.6g} fps"
        return "\n".join([
            f"[Plan] JonXL H3 Upscale: {width}x{height} -> {target_w}x{target_h} ({size_note})",
            f"[Plan] {what}: every picture is the first frame of its own 24-frame H3 clip (39 frames on the H3 grid), "
            "20 steps er_sde / beta57, CFG 1; the last frame is the result",
            "[Plan] models: the selected DiT + text encoder + VAE and the JonXL h3_upscale_1920 LoRA "
            "(Draft Size, Prompt, Refine Sampling and the window settings are not used)",
        ])
    draft = _draft_size(settings, width, height)
    geometry = planner.plan_geometry(width, height, target_w, target_h, draft)
    lines.append(f"[Plan] {geometry.describe()}")
    lines.append(f"[Plan] size rule: {size_note}; draft {draft}")

    manual_mode = str(settings.get("token_budget_mode") or "").strip().lower().startswith("manual")
    if manual_mode:
        budget = int(settings.get("max_tokens") or 160000)
    else:
        free_bytes, vram_label = _free_vram_bytes_for_plan(str(settings.get("device") or ""))
        if free_bytes <= 0:
            budget = int(settings.get("max_tokens") or 160000)
            lines.append(f"[Plan] auto budget unavailable ({vram_label}) - using Manual value {budget}")
        else:
            budget = planner.auto_token_budget(int(free_bytes), clamp_save_vram_gb(settings.get("reserve_vram_gb")), 145000.0)
            lines.append(f"[Plan] auto token budget {budget} | {vram_label}")
    if kind == "image":
        lines.append("[Plan] image: refined as a 5-frame still, first frame saved")
        tokens = planner.window_tokens(1, geometry)
        lines.append(f"[Plan] {tokens} tokens")
    else:
        window = planner.frames_for_token_budget(budget, geometry, int(settings.get("max_window_frames") or 121))
        plan = planner.plan_windows(total_frames, window, int(settings.get("overlap_frames") or 0))
        tokens = max(planner.window_tokens(length, geometry) for _, length in plan.windows)
        lines.append(f"[Plan] {total_frames} frames @ {fps:.3f} fps -> {plan.summary()}")
        lines.append(f"[Plan] up to {tokens} tokens per window")
        if frames_estimated:
            lines.append(f"[Plan] note: frame count estimated from duration x fps ({fps:.3f}).")
    steps = int(settings.get("steps") or 3)
    lines.append(f"[Plan] {steps} refine steps per window, start sigma "
                 f"{float(settings.get('start_sigma') or 0) or 'release grid (0.909375)'}")
    lines.append("[Live] Updates automatically as you change parameters.")
    return "\n".join(lines)


def _safe_ui_video_preview_path(video_path: Optional[str]) -> Optional[str]:
    """
    Return an isolated copy path for UI video widgets.

    Some UI/video preview stacks can normalize media in-place. By serving a copy
    to the UI, the original processing output remains untouched.
    """
    if not video_path:
        return video_path
    try:
        src = Path(str(video_path)).resolve()
        if not src.exists() or not src.is_file():
            return str(src)

        st = src.stat()
        sig = hashlib.sha256(
            f"{src.as_posix()}|{st.st_size}|{st.st_mtime_ns}|OPENH3_ui_preview_v1".encode("utf-8")
        ).hexdigest()[:20]
        cache_dir = src.parent / ".ui_preview_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dst = cache_dir / f"{src.stem}.__ui_preview_{sig}{src.suffix}"

        if dst.exists() and dst.is_file() and dst.stat().st_size > 1024:
            return str(dst)

        tmp = dst.with_name(f"{dst.name}.{os.getpid()}.tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        return str(dst)
    except Exception:
        return video_path


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _snap_down_8n1(value: int, low: int, high: int) -> int:
    """Clamp to [low, high] then snap DOWN onto the 8n+1 frame grid."""
    v = max(low, min(high, int(value)))
    snapped = ((v - 1) // 8) * 8 + 1
    return max(low, snapped)


def open_h3_defaults() -> Dict[str, Any]:
    """Defaults aligned to the NVIDIA Open-H3-Regenerate-2K recipe and the ComfyUI workflow."""
    try:
        from shared.gpu_utils import get_gpu_info

        cuda_default = "auto" if get_gpu_info() else "cpu"
    except Exception:
        cuda_default = "cpu"

    return {
        "input_path": "",
        "output_override": "",
        "output_format": "auto",
        "model_name": OPENH3_DIT_DEFAULT,
        "text_encoder": OPENH3_TE_NVFP4,
        "video_vae": OPENH3_VAE_SOL_FP16,
        "isolated_stages": False,
        "prompt": OPENH3_DEFAULT_PROMPT,
        "seed": OPENH3_DEFAULT_SEED,
        "randomize_seed": False,
        "resolution_preset": OPENH3_RESOLUTION_PRESETS[0],
        "upscale_factor": 2.0,
        "max_target_resolution": 0,
        "draft_size": OPENH3_DRAFT_SIZES[0],
        "steps": 3,
        "start_sigma": OPENH3_DEFAULT_START_SIGMA,
        "lora_strength": 1.0,
        "curve_strength": 1.0,
        "sparse_attention": True,
        # Upstream Sol-Attn quality preset (NVlabs/Sana PR #564): first refine step dense, then sparse at tau 1.0.
        # Measured: same structure as dense attention (37 dB), 23% faster refine; sparse first steps reshape objects.
        "sparse_tau": 1.0,
        "sparse_dense_steps": 1,
        "sparse_end_percent": 0.9,
        "dense_blocks": "0,1,49",
        "sink_conditioning": "exact_kv_and_rows",
        "attention_backend": "auto",
        "token_budget_mode": OPENH3_TOKEN_BUDGET_MODE_AUTO,
        "max_tokens": 160000,
        "max_window_frames": 121,
        "overlap_frames": 8,
        "ffn_chunk_tokens": 32768,
        "reserve_vram_gb": DEFAULT_SAVE_VRAM_GB,
        "max_gpu_blocks": 50,
        "fps_mode": OPENH3_FPS_MODES[0],
        # The refine attends to this audio; real speech keeps mouths in sync (measured +0.65 dB on the mouth).
        "audio_mode": OPENH3_AUDIO_MODES[1],
        "start_frame": 0,
        "end_frame": -1,
        "keep_latents": False,
        "fps": 0.0,
        "device": cuda_default,
        "save_metadata": True,
        "auto_transfer_output_to_input": False,
        "batch_enable": False,
        "batch_input_path": "",
        "batch_output_path": "",
        "resume_run_dir": "",
        "keep_only_output_files": False,
        "auto_tune_on_run": True,
        "workflow": OPENH3_WORKFLOW_REGEN,
        "jxl_max_side": OPENH3_JXL_MAX_SIDE_DEFAULT,
    }


OPENH3_ORDER: List[str] = [
    "input_path",
    "output_override",
    "output_format",
    "model_name",
    "text_encoder",
    "video_vae",
    "isolated_stages",
    "prompt",
    "seed",
    "randomize_seed",
    "resolution_preset",
    "upscale_factor",
    "max_target_resolution",
    "draft_size",
    "steps",
    "start_sigma",
    "lora_strength",
    "curve_strength",
    "sparse_attention",
    "sparse_tau",
    "sparse_end_percent",
    "dense_blocks",
    "sink_conditioning",
    "attention_backend",
    "token_budget_mode",
    "max_tokens",
    "max_window_frames",
    "overlap_frames",
    "ffn_chunk_tokens",
    "reserve_vram_gb",
    "max_gpu_blocks",
    "fps_mode",
    "audio_mode",
    "start_frame",
    "end_frame",
    "keep_latents",
    "fps",
    "device",
    "save_metadata",
    "auto_transfer_output_to_input",
    "batch_enable",
    "batch_input_path",
    "batch_output_path",
    "resume_run_dir",
    "keep_only_output_files",
    "auto_tune_on_run",
    "sparse_dense_steps",
    "workflow",
    "jxl_max_side",
]


def _open_h3_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    return dict(zip(OPENH3_ORDER, args))


def _pick(value: Any, options: List[str], default: str) -> str:
    text = str(value or "").strip()
    if text in options:
        return text
    lowered = text.lower()
    for option in options:
        if lowered and (option.lower().startswith(lowered) or lowered.startswith(option.lower().split(" ")[0])):
            return option
    return default


def _enforce_open_h3_guardrails(cfg: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {**defaults, **(cfg or {})}

    cfg["output_format"] = "auto"
    for key in ("save_metadata", "randomize_seed", "auto_transfer_output_to_input", "batch_enable",
                "keep_only_output_files", "sparse_attention", "keep_latents"):
        cfg[key] = _to_bool(cfg.get(key), _to_bool(defaults.get(key, False), False))

    cfg["model_name"] = _pick(cfg.get("model_name"), OPENH3_DIT_NAMES, OPENH3_DIT_DEFAULT)
    cfg["workflow"] = OPENH3_WORKFLOW_JXL if is_jxl_workflow(cfg) else OPENH3_WORKFLOW_REGEN
    side = _to_int(cfg.get("jxl_max_side"), OPENH3_JXL_MAX_SIDE_DEFAULT)
    cfg["jxl_max_side"] = side if side in OPENH3_JXL_MAX_SIDES else OPENH3_JXL_MAX_SIDE_DEFAULT
    cfg["text_encoder"] = _pick(cfg.get("text_encoder"), OPENH3_TE_NAMES, OPENH3_TE_NVFP4)
    cfg["video_vae"] = _pick(cfg.get("video_vae"), OPENH3_VAE_NAMES, OPENH3_VAE_SOL_FP16)
    cfg["prompt"] = str(cfg.get("prompt") if cfg.get("prompt") is not None else OPENH3_DEFAULT_PROMPT)[:6000]

    cfg["seed"] = max(0, min(2**31 - 1, _to_int(cfg.get("seed"), OPENH3_DEFAULT_SEED)))
    cfg["resolution_preset"] = _pick(cfg.get("resolution_preset"), OPENH3_RESOLUTION_PRESETS, OPENH3_RESOLUTION_PRESETS[0])
    cfg["upscale_factor"] = max(1.0, min(8.0, _to_float(cfg.get("upscale_factor"), 2.0)))
    cfg["max_target_resolution"] = max(0, min(8192, _to_int(cfg.get("max_target_resolution"), 0)))
    cfg["draft_size"] = _pick(cfg.get("draft_size"), OPENH3_DRAFT_SIZES, OPENH3_DRAFT_SIZES[0])

    cfg["steps"] = max(1, min(8, _to_int(cfg.get("steps"), 3)))
    start_sigma = max(0.0, min(1.0, _to_float(cfg.get("start_sigma"), OPENH3_DEFAULT_START_SIGMA)))
    if start_sigma <= 0 and cfg["steps"] not in (2, 3):
        start_sigma = 0.909375  # the release grids exist for 2 and 3 steps; other counts use the shift-6 rule
    cfg["start_sigma"] = start_sigma
    cfg["lora_strength"] = max(-2.0, min(2.0, _to_float(cfg.get("lora_strength"), 1.0)))
    cfg["curve_strength"] = max(-2.0, min(2.0, _to_float(cfg.get("curve_strength"), 1.0)))

    cfg["sparse_tau"] = max(0.0, min(4.0, _to_float(cfg.get("sparse_tau"), 1.0)))
    cfg["sparse_dense_steps"] = max(-1, min(8, _to_int(cfg.get("sparse_dense_steps"), 1)))
    cfg["sparse_end_percent"] = max(0.0, min(1.0, _to_float(cfg.get("sparse_end_percent"), 0.9)))
    cfg["dense_blocks"] = re.sub(r"[^0-9,\- ]", "", str(cfg.get("dense_blocks") or ""))[:200]
    sink = str(cfg.get("sink_conditioning") or "exact_kv_and_rows").strip()
    cfg["sink_conditioning"] = sink if sink in OPENH3_SINK_OPTIONS else "exact_kv_and_rows"
    attention = str(cfg.get("attention_backend") or "auto").strip().lower()
    cfg["attention_backend"] = attention if attention in OPENH3_ATTENTION_OPTIONS else "auto"

    token_mode = str(cfg.get("token_budget_mode") or "").strip()
    cfg["token_budget_mode"] = (OPENH3_TOKEN_BUDGET_MODE_MANUAL if "manual" in token_mode.lower()
                                else OPENH3_TOKEN_BUDGET_MODE_AUTO)
    cfg["max_tokens"] = max(20000, min(2000000, _to_int(cfg.get("max_tokens"), 160000)))
    cfg["max_window_frames"] = max(5, min(361, _to_int(cfg.get("max_window_frames"), 121)))
    overlap = max(0, min(64, _to_int(cfg.get("overlap_frames"), 8)))
    cfg["overlap_frames"] = min(overlap, max(0, cfg["max_window_frames"] - 5))
    cfg["ffn_chunk_tokens"] = max(0, min(1 << 20, _to_int(cfg.get("ffn_chunk_tokens"), 32768)))
    cfg["reserve_vram_gb"] = clamp_save_vram_gb(cfg.get("reserve_vram_gb"), defaults.get("reserve_vram_gb", DEFAULT_SAVE_VRAM_GB))
    cfg["auto_tune_on_run"] = _to_bool(cfg.get("auto_tune_on_run", defaults.get("auto_tune_on_run", True)), True)
    cfg["max_gpu_blocks"] = max(0, min(50, _to_int(cfg.get("max_gpu_blocks"), 50)))
    cfg["isolated_stages"] = _to_bool(cfg.get("isolated_stages"), False)

    cfg["fps_mode"] = OPENH3_FPS_MODES[1] if fps_mode_value(cfg.get("fps_mode")) == "24" else OPENH3_FPS_MODES[0]
    cfg["audio_mode"] = OPENH3_AUDIO_MODES[1] if audio_mode_value(cfg.get("audio_mode")) == "source" else OPENH3_AUDIO_MODES[0]
    cfg["start_frame"] = max(0, _to_int(cfg.get("start_frame"), 0))
    end_frame = _to_int(cfg.get("end_frame"), -1)
    cfg["end_frame"] = end_frame if end_frame >= 0 else -1
    cfg["fps"] = max(0.0, _to_float(cfg.get("fps"), 0.0))

    for key in ("input_path", "output_override", "batch_input_path", "batch_output_path", "resume_run_dir"):
        cfg[key] = str(cfg.get(key, defaults.get(key, "")) or "").strip()

    device_str = str(cfg.get("device", defaults.get("device", "auto")) or "auto").strip()
    if "," in device_str:
        cfg["device"] = device_str.split(",")[0].strip() or "auto"
        cfg["_multi_gpu_disabled_reason"] = "Open-H3 Regenerate is single-GPU."
    else:
        cfg["device"] = device_str or "auto"
    return cfg


def _target_note(settings: Dict[str, Any]) -> str:
    """Human-readable output size rule for logs and run summaries."""
    try:
        from shared.open_h3_runner import compute_target_size

        path = normalize_path(settings.get("_effective_input_path") or settings.get("input_path") or "")
        dims = get_media_dimensions(path) if path else None
        if dims:
            w, h, note = compute_target_size(int(dims[0]), int(dims[1]), settings)
            return f"{int(dims[0])}x{int(dims[1])} -> {w}x{h} ({note})"
    except Exception:
        pass
    return str(settings.get("resolution_preset") or "")


def _runner_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """UI labels -> engine values (the runner keeps everything else as is)."""
    out = dict(cfg)
    out["fps_mode"] = fps_mode_value(cfg.get("fps_mode"))
    out["audio_mode"] = audio_mode_value(cfg.get("audio_mode"))
    return out


def _apply_open_h3_preset(
    preset: Dict[str, Any],
    defaults: Dict[str, Any],
    preset_manager: PresetManager,
    current: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    base = defaults.copy()
    if current:
        base.update(current)
    merged = preset_manager.merge_config(base, preset)
    merged = _enforce_open_h3_guardrails(merged, defaults)
    return [merged[k] for k in OPENH3_ORDER]


def build_open_h3_callbacks(
    preset_manager: PresetManager,
    runner,
    run_logger: RunLogger,
    global_settings: Dict[str, Any],
    shared_state: gr.State,
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path,
):
    """Build Open-H3 callback functions for the UI."""
    defaults = open_h3_defaults()

    def refresh_presets(model_name: str, select_name: Optional[str] = None):
        """Refresh preset dropdown."""
        presets = preset_manager.list_presets("open_h3", model_name)
        last_used = preset_manager.get_last_used_name("open_h3", model_name)
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        return gr.update(choices=presets, value=value)

    def save_preset(preset_name: str, *args):
        """Save a preset."""
        if not preset_name.strip():
            return gr.update(), gr.update(value="[WARN] Enter a preset name"), *list(args)

        try:
            payload = _open_h3_dict_from_args(list(args))
            model_name = str(payload.get("model_name") or OPENH3_DIT_DEFAULT)

            preset_manager.save_preset_safe("open_h3", model_name, preset_name.strip(), payload)
            dropdown = refresh_presets(model_name, select_name=preset_name.strip())

            current_map = dict(zip(OPENH3_ORDER, list(args)))
            loaded_vals = _apply_open_h3_preset(payload, defaults, preset_manager, current=current_map)

            return dropdown, gr.update(value=f"[OK] Saved preset '{preset_name}'"), *loaded_vals
        except Exception as e:
            return gr.update(), gr.update(value=f"[ERROR] Error: {str(e)}"), *list(args)

    def load_preset(preset_name: str, model_name: str, current_values: List[Any]):
        """
        Load a preset.

        Returns (*values, status_message) to match UI output expectations
        (inputs_list + [preset_status]).
        """
        try:
            model_name = str(model_name or OPENH3_DIT_DEFAULT)
            preset = preset_manager.load_preset_safe("open_h3", model_name, preset_name)
            if preset:
                preset_manager.set_last_used("open_h3", model_name, preset_name)

            current_map = dict(zip(OPENH3_ORDER, current_values))
            values = _apply_open_h3_preset(preset or {}, defaults, preset_manager, current=current_map)

            status_msg = f"[OK] Loaded preset '{preset_name}'" if preset else "[INFO] Preset not found"
            return (*values, gr.update(value=status_msg))
        except Exception as e:
            logger.error(f"Error loading preset {preset_name}: {e}")
            return (*current_values, gr.update(value=f"[ERROR] Error: {str(e)}"))

    def safe_defaults():
        """Get safe default values."""
        normalized = _enforce_open_h3_guardrails(defaults.copy(), defaults)
        return [normalized[key] for key in OPENH3_ORDER]

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

    def _media_updates(out_path: Optional[str]) -> tuple[Any, Any]:
        """Return (output_video_update, output_image_update) for the output panel."""
        try:
            if out_path and not Path(out_path).is_dir():
                suf = Path(out_path).suffix.lower()
                if suf in video_exts:
                    safe_preview = _safe_ui_video_preview_path(out_path)
                    return gr.update(value=safe_preview, visible=True), gr.update(value=None, visible=False)
                if suf in image_exts:
                    return gr.update(value=None, visible=False), gr.update(value=out_path, visible=True)
        except Exception:
            pass
        return gr.update(value=None, visible=False), gr.update(value=None, visible=False)

    def _error_payload(state, status: str, detail: str = ""):
        vid_upd, img_upd = _media_updates(None)
        return (
            status,
            detail,
            vid_upd,
            img_upd,
            gr.update(visible=False),
            gr.update(value="", visible=False),
            state,
        )

    def live_plan_action(
        uploaded_file,
        *args,
        state: Dict[str, Any] = None,
        global_settings_snapshot: Dict[str, Any] | None = None,
        _global_settings: Dict[str, Any] = global_settings,
    ) -> str:
        """
        Instant in-process chunk plan (no subprocess, no sampling). Used by the
        Preview Chunk Plan button and by the live auto-refresh wiring so the
        plan re-renders as the user changes parameters - like the ComfyUI
        live-plan widget.
        """
        snapshot = (
            dict(global_settings_snapshot)
            if isinstance(global_settings_snapshot, dict)
            else dict(_global_settings)
        )
        state = state or {"seed_controls": {}}
        seed_controls = state.get("seed_controls", {}) if isinstance(state, dict) else {}

        settings = {**defaults, **_open_h3_dict_from_args(list(args))}
        settings = _enforce_open_h3_guardrails(settings, defaults)
        global_gpu_device = get_global_gpu_override(seed_controls, snapshot)
        settings["device"] = "cpu" if global_gpu_device == "cpu" else str(global_gpu_device)

        uploaded_path = ""
        if uploaded_file:
            if isinstance(uploaded_file, dict):
                uploaded_path = str(uploaded_file.get("path") or uploaded_file.get("name") or "")
            else:
                uploaded_path = str(getattr(uploaded_file, "name", "") or uploaded_file)
        candidate = normalize_path(settings.get("input_path") or "") or normalize_path(uploaded_path)
        settings["_effective_input_path"] = candidate

        try:
            return build_live_plan_text(settings, base_dir)
        except Exception as exc:
            logger.exception("Live chunk plan failed")
            return f"[ERROR] Live chunk plan failed: {exc}"

    def auto_tune_action(
        uploaded_file,
        *args,
        state: Dict[str, Any] = None,
        global_settings_snapshot: Dict[str, Any] | None = None,
        _global_settings: Dict[str, Any] = global_settings,
        **_ignored,
    ) -> Dict[str, Any]:
        """Instant Auto Tune (no test runs): opens the engine's quality/speed limits and shows the
        window plan for this input, output size and (GPU total - in use - Save VRAM)."""
        from shared.vram_autotune import open_h3 as h3_plan
        from shared.vram_autotune.common import resolve_budget

        snapshot = dict(global_settings_snapshot) if isinstance(global_settings_snapshot, dict) else dict(_global_settings)
        state = state or {"seed_controls": {}}
        seed_controls = state.setdefault("seed_controls", {})
        settings = {**defaults, **_open_h3_dict_from_args(list(args))}
        settings = _enforce_open_h3_guardrails(settings, defaults)
        global_gpu_device = get_global_gpu_override(seed_controls, snapshot)
        settings["device"] = "cpu" if global_gpu_device == "cpu" else str(global_gpu_device)
        uploaded_path = ""
        if uploaded_file:
            if isinstance(uploaded_file, dict):
                uploaded_path = str(uploaded_file.get("path") or uploaded_file.get("name") or "")
            else:
                uploaded_path = str(getattr(uploaded_file, "name", "") or uploaded_file)
        candidate = normalize_path(settings.get("input_path") or "") or normalize_path(uploaded_path)
        if not candidate or not Path(candidate).exists():
            return {"status": "Auto Tune needs an input first (upload a file or enter an Input Path).",
                    "report": "", "updates": {}, "state": state}
        settings["_effective_input_path"] = candidate
        budget = None if settings["device"] == "cpu" else resolve_budget(settings["device"], settings.get("reserve_vram_gb"))
        tp = h3_plan.plan(settings, base_dir, budget)
        report = tp.report("Auto Tune")
        seed_controls.setdefault("open_h3_settings", {}).update(tp.updates)
        state["preset_dirty"] = True
        for _line in report.splitlines():
            print(f"[Open-H3 Auto Tune] {_line}", flush=True)
        return {"status": report.splitlines()[0], "report": report, "updates": dict(tp.updates), "state": state}

    def cli_plan_preview_action(
        uploaded_file,
        *args,
        state: Dict[str, Any] = None,
        progress=None,
        global_settings_snapshot: Dict[str, Any] | None = None,
        _global_settings: Dict[str, Any] = global_settings,
    ):
        """
        Chunk-plan preview: runs the engine CLI with --plan_only against the
        current input and streams the "[Plan] ..." lines. Lightweight (no
        sampling, no model weights loaded) - it must NOT hold a GPU queue slot.
        """
        snapshot = (
            dict(global_settings_snapshot)
            if isinstance(global_settings_snapshot, dict)
            else dict(_global_settings)
        )
        state = state or {"seed_controls": {}}
        seed_controls = state.get("seed_controls", {}) if isinstance(state, dict) else {}

        settings = {**defaults, **_open_h3_dict_from_args(list(args))}
        settings = _enforce_open_h3_guardrails(settings, defaults)
        global_gpu_device = get_global_gpu_override(seed_controls, snapshot)
        settings["device"] = "cpu" if global_gpu_device == "cpu" else str(global_gpu_device)

        input_path = normalize_path(uploaded_file if uploaded_file else settings.get("input_path") or "")
        if not input_path or not Path(input_path).exists():
            yield _error_payload(
                state,
                "[ERROR] Input path missing",
                "Provide a video or image input (upload or path) before previewing the plan.",
            )
            return
        if detect_input_type(input_path) not in {"video", "image"}:
            yield _error_payload(
                state,
                "[ERROR] Plan needs a video or image input",
                "Open-H3 regenerates videos and single images; this input type cannot be planned.",
            )
            return
        settings["input_path"] = input_path
        settings["_effective_input_path"] = input_path

        _open_h3_cancel_event.clear()
        if progress:
            progress(0, desc="Planning Open-H3 chunks...")

        plan_lines: List[str] = []
        line_queue: "queue.Queue[str]" = queue.Queue()
        result_holder: Dict[str, Any] = {}

        def _plan_thread():
            try:
                rc, log_text = run_open_h3_plan(
                    settings,
                    base_dir,
                    on_progress=lambda msg: line_queue.put(str(msg)),
                    cancel_event=_open_h3_cancel_event,
                )
                result_holder["rc"] = rc
                result_holder["log"] = log_text
            except Exception as exc:
                result_holder["error"] = str(exc)

        worker = threading.Thread(target=_plan_thread, daemon=True)
        worker.start()

        vid_upd, img_upd = _media_updates(None)
        yield (
            "[RUNNING] Planning chunks (token budget preview)...",
            "Starting the Open-H3 planner. The engine reads the video, sizes the token budget "
            "from free VRAM (Auto mode), and prints the chunk plan without sampling.",
            vid_upd,
            img_upd,
            gr.update(visible=False),
            gr.update(value="", visible=False),
            state,
        )

        last_update = time.time()
        while worker.is_alive() or not line_queue.empty():
            try:
                msg = line_queue.get(timeout=0.1)
                text = str(msg or "").strip()
                if text:
                    plan_lines.append(text)
            except queue.Empty:
                pass
            now = time.time()
            if now - last_update > 0.5:
                last_update = now
                yield (
                    "[RUNNING] Planning chunks (token budget preview)...",
                    "\n".join(plan_lines[-120:]),
                    vid_upd,
                    img_upd,
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    state,
                )
        worker.join()

        if "error" in result_holder:
            yield _error_payload(state, "[ERROR] Chunk plan failed", f"Error: {result_holder['error']}")
            return

        rc = int(result_holder.get("rc", 1))
        full_log = str(result_holder.get("log") or "\n".join(plan_lines))
        if rc == 130 or _open_h3_cancel_event.is_set():
            yield _error_payload(state, "[STOP] Chunk plan cancelled", full_log)
            return
        if rc != 0:
            yield _error_payload(state, "[ERROR] Chunk plan failed", full_log)
            return

        plan_only = [ln for ln in full_log.splitlines() if ln.strip().startswith("[Plan]")]
        chunk_count_hint = ""
        for ln in plan_only:
            m = re.search(r"(\d+)\s+chunk", ln, flags=re.IGNORECASE)
            if m:
                chunk_count_hint = m.group(1)
        if chunk_count_hint == "1":
            recommendation = (
                "Recommendation: the whole video fits in ONE chunk - best quality. Run as-is."
            )
        elif chunk_count_hint:
            recommendation = (
                f"Recommendation: {chunk_count_hint} windows planned. For fewer windows, free VRAM, "
                "lower Save VRAM, choose a smaller target resolution, or trim the frame range."
            )
        else:
            recommendation = "Plan printed above. Auto mode prefers a single chunk when VRAM allows."

        status = "[OK] Chunk plan ready" + (f" ({chunk_count_hint} chunk(s))" if chunk_count_hint else "")
        if progress:
            progress(1.0, desc="Chunk plan ready")
        yield (
            status,
            (full_log + "\n\n" + recommendation).strip(),
            vid_upd,
            img_upd,
            gr.update(visible=False),
            gr.update(value="", visible=False),
            state,
        )

    def _chunk_handlers(chunk_template: Dict[str, Any]):
        """(prepass, process_chunk) for chunk_and_process.

        The prepass regenerates every pending chunk in ONE engine process (models load once);
        process_chunk then hands each finished chunk to the chunker at the path it expects and
        falls back to a normal per-chunk run for anything the prepass did not produce."""
        from shared.runner import RunResult

        ready: Dict[str, Any] = {}

        def _prepass(chunk_paths, start_index, on_progress=None):
            if len(chunk_paths) < 2 or float(chunk_template.get("fps") or 0) > 0:
                return
            try:
                from shared.open_h3_runner import run_open_h3_batch

                stage = Path(chunk_template.get("_run_dir") or temp_dir) / "open_h3_chunk_prepass"
                stage.mkdir(parents=True, exist_ok=True)
                items = []
                for chunk in chunk_paths:
                    item = dict(chunk_template)
                    item.update(input_path=str(chunk), _effective_input_path=str(chunk),
                                output_path=str(stage / f"{Path(chunk).stem}_upscaled.mp4"))
                    items.append(item)
                sink = on_progress if callable(on_progress) else (lambda _m: None)
                results = run_open_h3_batch(items, base_dir, on_progress=lambda m: sink(f"{m}\n"),
                                            cancel_event=_open_h3_cancel_event)
                for chunk, res in zip(chunk_paths, results):
                    if res.returncode == 0 and res.output_path and Path(res.output_path).exists():
                        ready[str(Path(chunk).resolve())] = res
            except Exception as exc:  # never block the chunker: chunks then run one by one
                logger.warning("Open-H3 chunk prepass failed: %s", exc)

        def _process(settings_chunk: Dict[str, Any], on_progress=None) -> RunResult:
            key = str(Path(settings_chunk.get("_effective_input_path") or settings_chunk.get("input_path") or "").resolve())
            done = ready.pop(key, None)
            if done is not None:
                target = Path(normalize_path(settings_chunk.get("output_override") or "") or done.output_path)
                if target.suffix.lower() != ".mp4":
                    target = target / Path(done.output_path).name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(done.output_path, target)
                if not ready:
                    try:  # the staging folder is empty once its last chunk moved out
                        Path(done.output_path).parent.rmdir()
                    except OSError:
                        pass
                return RunResult(0, str(target), done.log)
            r = run_open_h3(settings_chunk, base_dir, on_progress=on_progress,
                            cancel_event=_open_h3_cancel_event, process_handle=None)
            return RunResult(r.returncode, r.output_path, r.log)

        return _prepass, _process

    def _first_frame_preview(upload, settings, state, global_settings):
        """Regenerate only the first frame (as a 5-frame still) to check the look before a long run."""
        input_path = normalize_path(upload if upload else settings.get("input_path") or "")
        if not input_path or not Path(input_path).exists():
            yield _error_payload(state, "[ERROR] Input path missing", "Provide a video or image first.")
            return
        kind = detect_input_type(input_path)
        preview_dir = Path(global_settings.get("temp_dir", temp_dir)) / "open_h3_preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        still = input_path
        if kind == "video":
            import subprocess

            still = str(preview_dir / f"{Path(input_path).stem}_frame0.png")
            start = max(0, int(settings.get("start_frame") or 0))
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", input_path, "-vf", f"select=eq(n\\,{start})",
                   "-frames:v", "1", still]
            subprocess.run(cmd, capture_output=True, creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
            if not Path(still).exists():
                yield _error_payload(state, "[ERROR] Could not extract the first frame", input_path)
                return
        elif kind != "image":
            yield _error_payload(state, "[ERROR] Preview needs a video or image input", f"Detected: {kind}")
            return
        _open_h3_cancel_event.clear()
        item = dict(settings)
        item.update(input_path=still, _effective_input_path=still, _original_filename=Path(still).name,
                    output_override=str(preview_dir / f"{Path(input_path).stem}_open_h3_preview.png"),
                    batch_enable=False, resume_run_dir="")
        lines: List[str] = []
        holder: Dict[str, Any] = {}
        line_queue: "queue.Queue[str]" = queue.Queue()

        def _work():
            try:
                holder["result"] = run_open_h3(item, base_dir, on_progress=lambda m: line_queue.put(str(m)),
                                               cancel_event=_open_h3_cancel_event)
            except Exception as exc:
                holder["error"] = str(exc)

        worker = threading.Thread(target=_work, daemon=True)
        worker.start()
        started = time.time()
        last = 0.0
        while worker.is_alive() or not line_queue.empty():
            try:
                lines.append(line_queue.get(timeout=0.1))
            except queue.Empty:
                pass
            if time.time() - last > 0.5:
                last = time.time()
                yield (f"[RUNNING] First-frame preview... {int(time.time() - started)}s", "\n".join(lines[-60:]),
                       gr.skip(), gr.skip(), gr.skip(), gr.skip(), state)
        worker.join()
        result = holder.get("result")
        if "error" in holder or result is None or not result.output_path:
            detail = holder.get("error") or (result.log if result else "")
            yield _error_payload(state, "[ERROR] First-frame preview failed", detail or "\n".join(lines))
            return
        html_comp, img_slider = create_unified_comparison(input_path=still, output_path=result.output_path, mode="native")
        vid_upd, img_upd = _media_updates(result.output_path)
        done_text = ("[OK] First-frame preview ready (JonXL H3 Upscale of the first frame)" if is_jxl_workflow(item)
                     else "[OK] First-frame preview ready (5-frame still, first frame shown)")
        yield (done_text, "\n".join(lines[-200:]),
               vid_upd, img_upd, img_slider if img_slider else gr.update(visible=False),
               html_comp if html_comp else gr.update(value="", visible=False), state)

    def _convert_image_output(outp: Optional[str], settings: Dict[str, Any], seed_controls: Dict[str, Any],
                              output_settings: Dict[str, Any], logs_sink: Callable[[str], None]) -> Optional[str]:
        """Output-tab image format (png / jpg / webp) for still outputs; the engine writes PNG."""
        if not outp or Path(outp).suffix.lower() not in image_exts or not Path(outp).exists():
            return outp
        fmt = str(seed_controls.get("image_output_format_val", output_settings.get("image_output_format", "png")) or "png").lower()
        if fmt not in {"jpg", "webp"}:
            return outp
        try:
            quality = int(float(seed_controls.get("image_output_quality_val", output_settings.get("image_output_quality", 95)) or 95))
        except Exception:
            quality = 95
        try:
            from PIL import Image

            target = Path(outp).with_suffix(".jpg" if fmt == "jpg" else ".webp")
            with Image.open(outp) as img:
                img.convert("RGB").save(target, quality=max(1, min(100, quality)))
            Path(outp).unlink(missing_ok=True)
            logs_sink(f"Saved as {fmt.upper()} (quality {quality}): {target.name}")
            return str(target)
        except Exception as exc:
            logs_sink(f"[WARN] Image format conversion failed: {exc}")
            return outp

    def run_action(
        upload,
        *args,
        preview_only: bool = False,
        state=None,
        progress=None,
        global_settings_snapshot: Dict[str, Any] | None = None,
        _global_settings: Dict[str, Any] = global_settings,
    ):
        """Main processing action with gr.Progress integration and pre-flight checks."""
        try:
            global_settings = (
                dict(global_settings_snapshot)
                if isinstance(global_settings_snapshot, dict)
                else dict(_global_settings)
            )
            state = state or {"seed_controls": {}}
            clear_vram_oom_alert(state)
            seed_controls = state.get("seed_controls", {})
            output_settings = seed_controls.get("output_settings", {}) if isinstance(seed_controls, dict) else {}
            if not isinstance(output_settings, dict):
                output_settings = {}
            global_gpu_device = get_global_gpu_override(seed_controls, global_settings)
            seed_controls["global_gpu_device_val"] = global_gpu_device
            seed_controls["global_rife_cuda_device_val"] = "" if global_gpu_device == "cpu" else global_gpu_device
            seed_controls["open_h3_chunk_preview"] = {
                "message": "No chunk preview available yet.",
                "gallery": [],
                "videos": [],
                "count": 0,
            }
            seed_controls["open_h3_batch_outputs"] = []
            state["seed_controls"] = seed_controls
            settings_dict = _open_h3_dict_from_args(list(args))
            settings = {**defaults, **settings_dict}
            if settings.get("batch_enable"):
                settings["resume_run_dir"] = ""
            settings["device"] = "cpu" if global_gpu_device == "cpu" else str(global_gpu_device)

            settings = _enforce_open_h3_guardrails(settings, defaults)

            if preview_only:
                yield from _first_frame_preview(upload, settings, state, global_settings)
                return

            # Apply Output tab cached settings.
            fps_override_val = seed_controls.get("fps_override_val")
            if fps_override_val is None and isinstance(output_settings, dict):
                fps_override_val = output_settings.get("fps_override")
            if fps_override_val is not None:
                try:
                    fps_override_num = float(fps_override_val or 0.0)
                except Exception:
                    fps_override_num = 0.0
                settings["fps"] = fps_override_num if fps_override_num > 0 else settings.get("fps", 0.0)
            if seed_controls.get("comparison_mode_val"):
                settings["_comparison_mode"] = seed_controls["comparison_mode_val"]
            settings["save_metadata"] = bool(
                seed_controls.get(
                    "save_metadata_val",
                    output_settings.get("save_metadata", settings.get("save_metadata", True)),
                )
            )
            # Audio mux preferences (used by chunking + final output postprocessing).
            if seed_controls.get("audio_codec_val") is not None:
                settings["audio_codec"] = seed_controls.get("audio_codec_val") or "copy"
            elif output_settings.get("audio_codec") is not None:
                settings["audio_codec"] = output_settings.get("audio_codec") or "copy"
            if seed_controls.get("audio_bitrate_val") is not None:
                settings["audio_bitrate"] = seed_controls.get("audio_bitrate_val") or ""
            elif output_settings.get("audio_bitrate") is not None:
                settings["audio_bitrate"] = output_settings.get("audio_bitrate") or ""
            encode_overrides = {
                "video_codec": seed_controls.get("video_codec_val"),
                "video_quality": seed_controls.get("video_quality_val"),
                "video_preset": seed_controls.get("video_preset_val"),
                "h265_tune": seed_controls.get("h265_tune_val"),
                "av1_film_grain": seed_controls.get("av1_film_grain_val"),
                "av1_film_grain_denoise": seed_controls.get("av1_film_grain_denoise_val"),
                "pixel_format": seed_controls.get("pixel_format_val"),
                "two_pass_encoding": seed_controls.get("two_pass_encoding_val"),
            }
            for key in (
                "video_codec",
                "video_quality",
                "video_preset",
                "h265_tune",
                "av1_film_grain",
                "av1_film_grain_denoise",
                "pixel_format",
                "two_pass_encoding",
                "metadata_format",
                "log_level",
            ):
                value = encode_overrides.get(key)
                if value is None and output_settings:
                    value = output_settings.get(key)
                if value is not None:
                    settings[key] = value

            # Open-H3 tab uses global Output tab video settings for the engine writer.
            output_codec = settings.get("video_codec")
            if output_codec is None and isinstance(output_settings, dict):
                output_codec = output_settings.get("video_codec")
            codec_map = {
                "h264": "libx264",
                "h265": "libx265",
                "hevc": "libx265",
                "x265": "libx265",
                "av1": "libsvtav1",
                "libsvtav1": "libsvtav1",
            }
            output_codec_key = str(output_codec or "").strip().lower()
            mapped_codec = codec_map.get(output_codec_key, "libx264")
            settings["codec"] = mapped_codec
            if output_codec_key and output_codec_key not in codec_map:
                settings["_output_codec_note"] = (
                    f"Output tab codec '{output_codec_key}' is not supported by the Open-H3 writer; "
                    f"using '{mapped_codec}' instead."
                )

            output_quality = settings.get("video_quality")
            if output_quality is None and isinstance(output_settings, dict):
                output_quality = output_settings.get("video_quality")
            try:
                output_quality_i = int(float(output_quality if output_quality is not None else 15))
            except Exception:
                output_quality_i = 15
            settings["crf"] = max(0, min(51, output_quality_i))

            face_apply = bool(global_settings.get("face_global", False))
            face_strength = float(global_settings.get("face_strength", 0.5))

            _open_h3_cancel_event.clear()

            if progress:
                progress(0, desc="Initializing Open-H3...")

            def _cache_chunk_preview(run_dir: Optional[Path]) -> None:
                try:
                    if not run_dir:
                        seed_controls["open_h3_chunk_preview"] = {
                            "message": "No chunk preview available.",
                            "gallery": [],
                            "videos": [],
                            "count": 0,
                        }
                    else:
                        seed_controls["open_h3_chunk_preview"] = build_chunk_preview_payload(str(run_dir))
                    state["seed_controls"] = seed_controls
                except Exception:
                    pass

            # PRE-FLIGHT CHECKS (mirrors SparkVSR/SeedVR2 for consistency).
            from shared.error_handling import check_ffmpeg_available, check_disk_space

            ffmpeg_ok, ffmpeg_msg = check_ffmpeg_available()
            if not ffmpeg_ok:
                yield _error_payload(
                    state,
                    "[ERROR] ffmpeg not found in PATH",
                    ffmpeg_msg or "Install ffmpeg and add to PATH before processing",
                )
                return

            output_path_check = Path(global_settings.get("output_dir", output_dir))
            has_space, space_warning = check_disk_space(output_path_check, required_mb=5000)
            if not has_space:
                yield _error_payload(
                    state,
                    "[ERROR] Insufficient disk space",
                    space_warning or "Free up at least 5GB disk space before processing",
                )
                return

            # Universal chunking settings (Resolution tab). Open-H3 defaults to
            # NO app-level chunking: the engine has its own token-budget planner
            # and prefers a single chunk. App-level scene/fixed chunking is used
            # only when the user explicitly disabled Auto Chunk and set a fixed
            # chunk size in the Resolution tab.
            auto_chunk = bool(seed_controls.get("auto_chunk", True))
            chunk_size_sec = float(seed_controls.get("chunk_size_sec", 0) or 0)
            chunk_overlap_sec = 0.0 if auto_chunk else float(seed_controls.get("chunk_overlap_sec", 0) or 0)
            per_chunk_cleanup = bool(seed_controls.get("per_chunk_cleanup", False))
            scene_threshold = float(seed_controls.get("scene_threshold", 27.0))
            min_scene_len = float(seed_controls.get("min_scene_len", 1.0))
            frame_accurate_split = bool(seed_controls.get("frame_accurate_split", True))
            use_app_chunking = (not auto_chunk) and chunk_size_sec > 0
            settings["frame_accurate_split"] = frame_accurate_split
            if use_app_chunking and fps_mode_value(settings.get("fps_mode")) == "24":
                # App-level chunks are merged frame-for-frame; keep every source frame.
                settings["fps_mode"] = OPENH3_FPS_MODES[0]
                settings["_fps_mode_note"] = (
                    "[OpenH3] App-level chunking keeps source frames 1:1 (24 fps resampling is applied "
                    "only without chunking)."
                )

            def _run_summary_extras(chunk_count_value: int) -> Dict[str, Any]:
                if not chunk_count_value:
                    return {}
                return {
                    "chunking": {
                        "mode": "static",
                        "chunk_size_sec": float(chunk_size_sec or 0),
                        "chunk_overlap_sec": float(chunk_overlap_sec or 0),
                        "scene_threshold": float(scene_threshold or 27.0),
                        "min_scene_len": float(min_scene_len or 1.0),
                        "chunks": int(chunk_count_value or 0),
                        "frame_accurate_split": bool(frame_accurate_split),
                    }
                }

            def _apply_video_post_passes(
                outp: str,
                original_input: str,
                logs_sink: Callable[[str], None],
                chunk_count_value: int,
            ) -> str:
                """Face restore -> audio -> RIFE -> comparison video, in SparkVSR order (stills: face restore + format)."""
                if outp and Path(outp).exists() and Path(outp).suffix.lower() in image_exts:
                    if face_apply:
                        try:
                            from shared.face_restore import restore_image

                            logs_sink(f"Applying face restoration (strength {face_strength})...")
                            restored = restore_image(outp, strength=face_strength,
                                                     on_progress=lambda x: logs_sink(x) if x else None,
                                                     gpu_device=settings.get("device"))
                            if restored and Path(restored).exists():
                                outp = restored
                                logs_sink(f"[OK] Face restoration complete: {restored}")
                        except Exception as face_exc:
                            logs_sink(f"[WARN] Face restoration failed: {face_exc}")
                    return _convert_image_output(outp, settings, seed_controls, output_settings, logs_sink)
                if face_apply and outp and Path(outp).exists() and Path(outp).suffix.lower() in video_exts:
                    try:
                        from shared.face_restore import restore_video

                        logs_sink(f"Applying face restoration (strength {face_strength})...")
                        restored = restore_video(
                            outp,
                            strength=face_strength,
                            on_progress=lambda x: logs_sink(x) if x else None,
                            gpu_device=settings.get("device"),
                        )
                        if restored and Path(restored).exists():
                            outp = restored
                            logs_sink(f"[OK] Face restoration complete: {restored}")
                    except Exception as face_exc:
                        logs_sink(f"[WARN] Face restoration failed: {face_exc}")

                # Best-effort audio preservation. The engine already copies the
                # source audio track; this repairs runs where that step failed
                # or strips audio when the Output tab requests 'none'.
                if outp and Path(outp).exists() and Path(outp).suffix.lower() in video_exts:
                    try:
                        from shared.audio_utils import ensure_audio_on_video

                        audio_codec = str(settings.get("audio_codec") or "copy")
                        audio_bitrate = settings.get("audio_bitrate") or None
                        _changed, _final, _err = ensure_audio_on_video(
                            Path(outp),
                            Path(original_input),
                            audio_codec=audio_codec,
                            audio_bitrate=str(audio_bitrate) if audio_bitrate else None,
                            on_progress=lambda x: logs_sink(x) if x else None,
                        )
                        if _err:
                            logs_sink(f"WARNING: Audio mux: {_err}")
                        if _final and str(_final) != str(outp):
                            outp = str(_final)
                    except Exception as audio_exc:
                        logs_sink(f"WARNING: Audio mux failed: {audio_exc}")

                if outp and Path(outp).exists() and Path(outp).suffix.lower() in video_exts:
                    rife_out, rife_msg = maybe_apply_global_rife(
                        runner=runner,
                        output_video_path=outp,
                        seed_controls=seed_controls,
                        on_log=(lambda m: logs_sink(m.strip()) if m else None),
                        chunking_context={
                            "enabled": bool(chunk_count_value and chunk_count_value > 0),
                            "auto_chunk": False,
                            "chunk_size_sec": float(chunk_size_sec or 0),
                            "chunk_overlap_sec": float(chunk_overlap_sec or 0),
                            "scene_threshold": float(scene_threshold or 27.0),
                            "min_scene_len": float(min_scene_len or 1.0),
                            "frame_accurate_split": bool(frame_accurate_split),
                            "per_chunk_cleanup": bool(per_chunk_cleanup),
                        },
                    )
                    if rife_out and Path(rife_out).exists():
                        logs_sink(f"[OK] Global RIFE output: {Path(rife_out).name}")
                        outp = rife_out
                    elif rife_msg:
                        logs_sink(f"[WARN] {rife_msg}")

                    comp_vid_path, comp_vid_err = maybe_generate_input_vs_output_comparison(
                        original_input,
                        outp,
                        seed_controls,
                        label_output="open_h3",
                        on_progress=(lambda m: logs_sink(m.strip()) if m else None),
                    )
                    if comp_vid_path:
                        logs_sink(f"[OK] Comparison video created: {Path(comp_vid_path).name}")
                    elif comp_vid_err:
                        logs_sink(f"[WARN] Comparison video failed: {comp_vid_err}")
                return outp

            # -------------------------------------------------------------
            # Batch processing (folder of videos and images)
            # -------------------------------------------------------------
            if bool(settings.get("batch_enable")):
                batch_in = normalize_path(settings.get("batch_input_path") or "")

                if not batch_in or not Path(batch_in).exists() or not Path(batch_in).is_dir():
                    yield _error_payload(
                        state,
                        "[ERROR] Batch input folder missing/invalid",
                        "Provide a valid Batch Input Folder path.",
                    )
                    return

                in_dir = Path(batch_in)
                batch_root = resolve_batch_output_dir(
                    batch_input_path=str(in_dir),
                    batch_output_path=settings.get("batch_output_path"),
                    fallback_output_dir=Path(global_settings.get("output_dir", output_dir)),
                    default_subdir_name="upscaled_files",
                )
                settings["batch_output_path"] = str(batch_root)
                try:
                    batch_root.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

                def _is_within(path: Path, parent: Path) -> bool:
                    try:
                        path.resolve().relative_to(parent.resolve())
                        return True
                    except Exception:
                        return False

                excluded_subtree: Optional[Path] = None
                try:
                    if _is_within(batch_root, in_dir) and batch_root.resolve() != in_dir.resolve():
                        excluded_subtree = batch_root.resolve()
                except Exception:
                    excluded_subtree = None

                items: List[Path] = []
                skipped_non_video = 0
                try:
                    for p in list_directory_entries_sorted(
                        in_dir,
                        include_files=True,
                        include_dirs=False,
                        extensions=VIDEO_EXTENSIONS.union(IMAGE_EXTENSIONS),
                    ):
                        try:
                            if excluded_subtree and _is_within(p, excluded_subtree):
                                continue
                            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS.union(IMAGE_EXTENSIONS):
                                items.append(p)
                        except Exception:
                            continue
                except Exception:
                    items = []

                jxl_skipped: List[str] = []
                if is_jxl_workflow(settings):  # images only: the folder's videos are skipped
                    jxl_skipped = [p.name for p in items if p.suffix.lower() not in IMAGE_EXTENSIONS]
                    items = [p for p in items if p.suffix.lower() in IMAGE_EXTENSIONS]
                    if jxl_skipped:
                        gr.Warning(f"JonXL H3 Upscale is for images: {len(jxl_skipped)} video(s) in the batch folder "
                                   f"are skipped.")
                    if not items:
                        yield _error_payload(state, "[WARNING] JonXL H3 Upscale is for images: no images in the batch "
                                             "folder", OPENH3_JXL_VIDEO_MESSAGE)
                        return

                if not items:
                    yield _error_payload(
                        state,
                        "[ERROR] No video or image files found in batch input folder",
                        "Open-H3 batch processes video and image files (frame folders are not supported).",
                    )
                    return

                overwrite_existing = bool(seed_controls.get("overwrite_existing_batch_val", False))

                logs: List[str] = []
                outputs: List[str] = []
                last_input_path: Optional[str] = None
                last_output_path: Optional[str] = None
                last_chunk_run_dir: Optional[Path] = None
                total_items = max(1, len(items))

                def _batch_live_payload(status_text: str, preview_out: Optional[str] = None):
                    seed_controls["open_h3_batch_outputs"] = list(outputs)
                    state["seed_controls"] = seed_controls
                    preview_candidate = str(preview_out or "").strip() if preview_out else ""
                    if (not preview_candidate) and last_output_path:
                        preview_candidate = str(last_output_path)
                    vid_upd, img_upd = _media_updates(preview_candidate if preview_candidate else None)
                    return (
                        status_text,
                        "\n".join(logs[-200:]),
                        vid_upd,
                        img_upd,
                        gr.update(visible=False),
                        gr.update(value="", visible=False),
                        state,
                    )

                if progress:
                    progress(0, desc=f"Batch: {len(items)} item(s) queued")
                logs.append(f"Batch queued: {len(items)} item(s). Output: {batch_root}")
                if jxl_skipped:
                    logs.append(f"[SKIP] JonXL H3 Upscale is for images; {len(jxl_skipped)} video(s) skipped: "
                                f"{', '.join(jxl_skipped[:10])}{' ...' if len(jxl_skipped) > 10 else ''}")
                if not use_app_chunking:
                    logs.append(
                        "[OpenH3] App-level scene chunking is OFF (the engine plans its own windows). "
                        "Disable Auto Chunk and set a fixed chunk size in the Resolution tab to force it."
                    )
                yield _batch_live_payload(f"Starting Open-H3 batch ({len(items)} item(s))")

                def _prep_batch_item(idx: int, item_path: str):
                    """Run folder + settings for one batch item: ("ok", settings, out_dir, output, is_image) or skip/error."""
                    from shared.output_run_manager import prepare_batch_video_run_dir

                    item_settings = settings.copy()
                    item_settings["batch_enable"] = False
                    item_settings["input_path"] = item_path
                    item_settings["_effective_input_path"] = item_path
                    item_settings["_original_filename"] = Path(item_path).name
                    item_input = Path(item_path)
                    item_out_dir = batch_item_dir(batch_root, item_input.name)
                    item_is_image = item_input.suffix.lower() in IMAGE_EXTENSIONS
                    predicted_output_file = item_out_dir / f"{item_input.stem}{'.png' if item_is_image else '.mp4'}"
                    run_paths = prepare_batch_video_run_dir(
                        batch_root,
                        item_input.name,
                        input_path=str(item_input),
                        model_label="open_h3",
                        mode=str(getattr(runner, "get_mode", lambda: "subprocess")() or "subprocess"),
                        overwrite_existing=overwrite_existing,
                    )
                    if not run_paths:
                        if not overwrite_existing:
                            return ("skip", f"[SKIP] [{idx}/{len(items)}] {item_input.name} skipped (output folder exists)",
                                    str(predicted_output_file) if predicted_output_file.exists() else "")
                        return ("error", f"[ERROR] [{idx}/{len(items)}] {item_input.name} failed (could not create output folder)")
                    item_settings["global_output_dir"] = str(run_paths.run_dir)
                    item_settings["_run_dir"] = str(run_paths.run_dir)
                    item_settings["_processed_chunks_dir"] = str(run_paths.processed_chunks_dir)
                    item_settings["output_override"] = str(predicted_output_file)
                    return ("ok", item_settings, item_out_dir, predicted_output_file, item_is_image)

                # Every item that needs no app-level chunking runs in ONE engine process, so the 21 GB DiT,
                # the LoRA merge and the VAE load once for the whole folder instead of once per file.
                grouped_prep: Dict[int, Any] = {}
                grouped_results: Dict[int, Any] = {}
                if (not use_app_chunking) and len(items) > 1 and float(settings.get("fps") or 0) <= 0:
                    from shared.open_h3_runner import run_open_h3_batch

                    group_idx: List[int] = []
                    group_settings: List[Dict[str, Any]] = []
                    for idx, item in enumerate(items, 1):
                        prepared = _prep_batch_item(idx, str(item))
                        grouped_prep[idx] = prepared
                        if prepared[0] == "ok":
                            group_idx.append(idx)
                            group_settings.append(prepared[1])
                    if len(group_settings) > 1:
                        logs.append(f"[OpenH3] {len(group_settings)} item(s) run in one engine process (models load once).")
                        yield _batch_live_payload(f"Batch: regenerating {len(group_settings)} item(s) in one engine run")
                        group_queue: "queue.Queue[str]" = queue.Queue()
                        group_holder: Dict[str, Any] = {}

                        def _group_worker():
                            try:
                                group_holder["results"] = run_open_h3_batch(
                                    group_settings, base_dir, on_progress=lambda m: group_queue.put(str(m)),
                                    cancel_event=_open_h3_cancel_event)
                            except Exception as exc:
                                group_holder["error"] = str(exc)

                        group_thread = threading.Thread(target=_group_worker, daemon=True)
                        group_thread.start()
                        last_tick = 0.0
                        live_line = ""
                        while group_thread.is_alive() or not group_queue.empty():
                            try:
                                text = group_queue.get(timeout=0.1).strip()
                                if text.startswith("Processing:") or "still running" in text:
                                    live_line = text
                                    match = re.search(r"Processing:\s*(\d+)\s*/\s*(\d+)", text)
                                    if match and progress:
                                        progress(int(match.group(1)) / max(1, int(match.group(2))), desc=text[:100])
                                elif text:
                                    logs.append(text)
                            except queue.Empty:
                                pass
                            if time.time() - last_tick > 0.5:
                                last_tick = time.time()
                                yield _batch_live_payload(f"[RUNNING] Batch engine run: {live_line or 'loading models...'}")
                        group_thread.join()
                        if "error" in group_holder:
                            logs.append(f"[WARN] Grouped batch run failed ({group_holder['error']}); items run one by one.")
                        else:
                            for idx, res in zip(group_idx, group_holder.get("results") or []):
                                grouped_results[idx] = res

                for idx, item in enumerate(items, 1):
                    if _open_h3_cancel_event.is_set():
                        logs.append("Batch cancelled by user.")
                        payload = _batch_live_payload("Batch cancelled by user")
                        yield (
                            payload[0],
                            (payload[1] + "\n\n[Cancelled by user]").strip(),
                            payload[2],
                            payload[3],
                            payload[4],
                            payload[5],
                            payload[6],
                        )
                        return

                    item_path = str(item)
                    last_input_path = item_path
                    item_name = Path(item_path).name
                    logs.append(f"[{idx}/{len(items)}] Processing {item_name}")
                    yield _batch_live_payload(f"Batch {idx}/{len(items)}: processing {item_name}")

                    if progress:
                        progress((idx - 1) / max(1, len(items)), desc=f"Batch {idx}/{len(items)}: {item_name}")

                    if idx in grouped_prep:
                        prepared = grouped_prep[idx]
                    else:
                        prepared = _prep_batch_item(idx, item_path)
                    if prepared[0] == "skip":
                        logs.append(prepared[1])
                        if prepared[2]:
                            outputs.append(prepared[2])
                        yield _batch_live_payload(f"Batch {idx}/{len(items)}: skipped {item_name}")
                        continue
                    if prepared[0] == "error":
                        logs.append(prepared[1])
                        yield _batch_live_payload(f"Batch {idx}/{len(items)}: could not prepare output for {item_name}")
                        continue
                    _status, item_settings, item_out_dir, predicted_output_file, item_is_image = prepared

                    chunk_count_item = 0
                    if use_app_chunking and not item_is_image:
                        from shared.chunking import chunk_and_process
                        from shared.runner import RunResult

                        class _CancelProbe:
                            def is_canceled(self) -> bool:
                                return bool(_open_h3_cancel_event.is_set())

                        chunk_settings = item_settings.copy()
                        chunk_settings["frame_accurate_split"] = frame_accurate_split
                        try:
                            base_stem = Path(chunk_settings.get("_original_filename") or item_path).stem
                            chunk_settings["output_override"] = str(Path(item_out_dir) / f"{base_stem}.mp4")
                        except Exception:
                            pass

                        _prepass_chunks, _process_chunk = _chunk_handlers(chunk_settings)

                        rc, clog, final_output, chunk_count_item = chunk_and_process(
                            runner=_CancelProbe(),
                            settings=chunk_settings,
                            scene_threshold=scene_threshold,
                            min_scene_len=min_scene_len,
                            work_dir=Path(item_out_dir),
                            on_progress=lambda msg: None,
                            chunk_seconds=chunk_size_sec,
                            chunk_overlap=chunk_overlap_sec,
                            per_chunk_cleanup=per_chunk_cleanup,
                            allow_partial=True,
                            global_output_dir=str(item_out_dir),
                            resume_from_partial=False,
                            progress_tracker=None,
                            process_func=_process_chunk,
                            pre_process_chunks_func=_prepass_chunks,
                            model_type="open_h3",
                        )
                        result = RunResult(rc, final_output if final_output else None, clog)
                        outp = result.output_path
                    elif idx in grouped_results:
                        result = grouped_results[idx]
                        outp = result.output_path
                    else:
                        result = run_open_h3(
                            item_settings,
                            base_dir,
                            on_progress=None,
                            cancel_event=_open_h3_cancel_event,
                            process_handle=None,
                        )
                        outp = result.output_path

                    if outp and Path(outp).exists():
                        outp = _apply_video_post_passes(
                            outp,
                            item_path,
                            lambda m: logs.append(str(m)),
                            chunk_count_item,
                        )

                        if chunk_count_item:
                            last_chunk_run_dir = Path(item_settings.get("_run_dir") or item_out_dir)

                        outputs.append(outp)
                        last_output_path = outp
                        if chunk_count_item:
                            logs.append(
                                f"[OK] [{idx}/{len(items)}] {Path(item_path).name} -> {Path(outp).name} "
                                f"({int(chunk_count_item)} chunks)"
                            )
                        else:
                            logs.append(f"[OK] [{idx}/{len(items)}] {Path(item_path).name} -> {Path(outp).name}")
                    else:
                        if getattr(result, "returncode", 0) != 0:
                            maybe_set_vram_oom_alert(
                                state, model_label="open_h3", text=getattr(result, "log", ""), settings=item_settings
                            )
                        logs.append(f"[ERROR] [{idx}/{len(items)}] {Path(item_path).name} failed")

                    if bool(item_settings.get("save_metadata", True)):
                        try:
                            run_logger.write_summary(
                                Path(outp) if outp else output_dir,
                                {
                                    "input": item_path,
                                    "output": outp,
                                    "returncode": result.returncode,
                                    "args": item_settings,
                                    "face_apply": face_apply,
                                    "face_strength": face_strength,
                                    "pipeline": "open_h3",
                                    "batch": True,
                                    "target": _target_note(item_settings),
                                    **_run_summary_extras(chunk_count_item),
                                },
                            )
                        except Exception:
                            pass

                    succeeded = len(outputs)
                    with_issues = max(0, idx - succeeded)
                    yield _batch_live_payload(
                        f"Batch progress {idx}/{total_items}: {succeeded} succeeded, {with_issues} with issues",
                        preview_out=last_output_path,
                    )

                if bool(settings.get("keep_only_output_files", False)):
                    try:
                        cleanup_result = keep_only_batch_outputs(
                            batch_root,
                            outputs,
                            on_log=lambda msg: logs.append(str(msg)),
                        )
                        flattened_outputs = cleanup_result.get("final_outputs")
                        if isinstance(flattened_outputs, list) and flattened_outputs:
                            outputs = [str(p) for p in flattened_outputs if str(p).strip()]
                            last_output_path = outputs[-1] if outputs else last_output_path
                    except Exception as cleanup_err:
                        logs.append(f"Keep-only cleanup warning: {cleanup_err}")

                if progress:
                    progress(1.0, desc=f"Batch complete ({len(outputs)}/{len(items)} succeeded)")

                _cache_chunk_preview(last_chunk_run_dir)
                seed_controls["open_h3_batch_outputs"] = list(outputs)
                state["seed_controls"] = seed_controls

                if last_output_path:
                    try:
                        outp_path = Path(last_output_path)
                        seed_controls["last_output_dir"] = str(outp_path.parent if outp_path.is_file() else outp_path)
                        seed_controls["last_output_path"] = str(outp_path) if outp_path.is_file() else None
                        state["seed_controls"] = seed_controls
                    except Exception:
                        pass

                html_comp = gr.update(value="", visible=False)
                img_slider = gr.update(visible=False)
                if last_input_path and last_output_path:
                    try:
                        h, sld = create_unified_comparison(
                            input_path=last_input_path,
                            output_path=last_output_path,
                            mode=(
                                "slider"
                                if Path(last_output_path).suffix.lower() in video_exts
                                else "native"
                            ),
                        )
                        html_comp = h if h else gr.update(value="", visible=False)
                        img_slider = sld if sld else gr.update(visible=False)
                    except Exception:
                        pass

                status = (
                    f"[OK] Open-H3 batch complete ({len(outputs)}/{len(items)} succeeded, "
                    f"{len(items) - len(outputs)} failed)"
                )
                vid_upd, img_upd = _media_updates(last_output_path)
                yield (
                    status,
                    "\n".join(logs),
                    vid_upd,
                    img_upd,
                    img_slider if img_slider else gr.update(visible=False),
                    html_comp if html_comp else gr.update(value="", visible=False),
                    state,
                )
                return

            # -------------------------------------------------------------
            # Single input
            # -------------------------------------------------------------
            resume_run_dir_raw = str(settings.get("resume_run_dir") or "").strip()
            resume_mode = bool(resume_run_dir_raw and (not settings.get("batch_enable")))
            if resume_mode:
                resume_run_dir = Path(normalize_path(resume_run_dir_raw))
                if not (resume_run_dir.exists() and resume_run_dir.is_dir()):
                    yield _error_payload(
                        state,
                        "[ERROR] Resume folder not found",
                        f"Configured resume folder does not exist: {resume_run_dir}",
                    )
                    return
                recovered_input, recovered_name, _recovered_source = resolve_resume_input_from_run_dir(resume_run_dir)
                if recovered_input is None:
                    yield _error_payload(
                        state,
                        "[ERROR] Resume input not found",
                        (
                            f"Could not recover input source from resume folder: {resume_run_dir}. "
                            "Expected run_context.json input path or run_metadata artifact."
                        ),
                    )
                    return
                input_path = normalize_path(str(recovered_input))
                settings["_original_filename"] = recovered_name or Path(input_path).name
            else:
                input_path = normalize_path(upload if upload else settings["input_path"])
            if not input_path or not Path(input_path).exists():
                yield _error_payload(state, "[ERROR] Input path missing", "")
                return

            input_kind_single = detect_input_type(input_path)
            if input_kind_single not in {"video", "image"}:
                yield _error_payload(
                    state,
                    "[ERROR] Open-H3 expects a video or an image",
                    (
                        f"Detected input type: {input_kind_single}. Convert frame folders to a video first."
                    ),
                )
                return
            if input_kind_single == "video" and is_jxl_workflow(settings):
                gr.Warning(OPENH3_JXL_VIDEO_MESSAGE)
                yield _error_payload(state, "[WARNING] JonXL H3 Upscale is for images, not videos",
                                     OPENH3_JXL_VIDEO_MESSAGE)
                return
            if input_kind_single == "image" and use_app_chunking:
                use_app_chunking = False

            settings["input_path"] = input_path
            settings["_effective_input_path"] = input_path
            if not settings.get("_original_filename"):
                settings["_original_filename"] = Path(input_path).name
            seed_controls["open_h3_batch_outputs"] = []
            state["seed_controls"] = seed_controls

            # Per-run output folder (0001/0002/...) so all artifacts stay together.
            if resume_run_dir_raw:
                resume_run_dir = Path(normalize_path(resume_run_dir_raw))
                run_dir = resume_run_dir
                processed_chunks_dir = run_dir / "processed_chunks"
                processed_chunks_dir.mkdir(parents=True, exist_ok=True)
                seed_controls["last_run_dir"] = str(run_dir)
                settings["_run_dir"] = str(run_dir)
                settings["_processed_chunks_dir"] = str(processed_chunks_dir)
                settings["_resume_run_requested"] = True
                settings["_user_output_override_raw"] = settings.get("output_override") or ""

                base_stem = Path(settings.get("_original_filename") or input_path).stem
                default_final = run_dir / f"{base_stem}.mp4"
                # Resume mode ignores new output override paths.
                settings["output_override"] = str(default_final)
            else:
                try:
                    base_out_root = Path(global_settings.get("output_dir", output_dir))
                    run_paths, explicit_final = prepare_single_video_run(
                        output_root_fallback=base_out_root,
                        output_override_raw=settings.get("output_override"),
                        input_path=settings["input_path"],
                        original_filename=settings.get("_original_filename") or Path(settings["input_path"]).name,
                        model_label="open_h3",
                        mode="subprocess",
                    )
                    run_dir = Path(run_paths.run_dir)
                    seed_controls["last_run_dir"] = str(run_dir)
                    settings["_run_dir"] = str(run_dir)
                    settings["_processed_chunks_dir"] = str(run_paths.processed_chunks_dir)
                    settings["_user_output_override_raw"] = settings.get("output_override") or ""

                    base_stem = Path(settings.get("_original_filename") or input_path).stem
                    default_final = run_dir / f"{base_stem}.mp4"
                    settings["output_override"] = str(explicit_final) if explicit_final else str(default_final)
                except Exception:
                    pass

            progress_queue = queue.Queue()
            settings["global_output_dir"] = str(Path(settings.get("_run_dir") or output_dir))

            resume_requested = bool(settings.get("_resume_run_requested"))
            if resume_requested and not use_app_chunking:
                yield _error_payload(
                    state,
                    "Resume unavailable for current mode",
                    (
                        "Resume run folder works only with app-level chunked video processing. "
                        "Disable Auto Chunk and set a fixed chunk size in the Resolution tab "
                        "(same settings as the original run) to resume."
                    ),
                )
                return
            if resume_requested:
                from shared.chunking import check_resume_available

                resume_root = Path(settings.get("_run_dir") or Path(global_settings.get("output_dir", output_dir)))
                resume_ok, resume_msg = check_resume_available(resume_root, "mp4")
                if not resume_ok:
                    yield _error_payload(
                        state,
                        "Resume failed",
                        f"Resume requested but no resumable chunk outputs were found in {resume_root}. {resume_msg}",
                    )
                    return

            # Run Open-H3 in a worker thread with cancel support.
            result_holder: Dict[str, Any] = {}
            process_handle = {"proc": None}

            def processing_thread():
                try:
                    if use_app_chunking:
                        from shared.chunking import chunk_and_process
                        from shared.runner import RunResult

                        class _CancelProbe:
                            def is_canceled(self) -> bool:
                                return bool(_open_h3_cancel_event.is_set())

                        chunk_settings = settings.copy()
                        if not (chunk_settings.get("output_override") or "").strip():
                            try:
                                out_base = Path(global_settings.get("output_dir", output_dir))
                                base_stem = Path(chunk_settings.get("_original_filename") or input_path).stem
                                chunk_settings["output_override"] = str(out_base / f"{base_stem}.mp4")
                            except Exception:
                                pass
                        if resume_requested:
                            progress_queue.put(
                                "Resume run folder detected. Resuming from last processed chunk and "
                                "continuing remaining chunks (same settings required)."
                            )

                        _prepass_chunks, _process_chunk = _chunk_handlers(chunk_settings)

                        def _chunk_progress_cb(progress_val, desc="", **kwargs):
                            try:
                                pct = int(float(progress_val) * 100)
                                progress_queue.put(f"{pct}% {desc}".strip())
                            except Exception:
                                pass
                            if str(kwargs.get("phase", "")).strip().lower() == "completed":
                                try:
                                    run_root = Path(
                                        settings.get("_run_dir")
                                        or global_settings.get("output_dir", output_dir)
                                    )
                                    seed_controls["open_h3_chunk_preview"] = build_chunk_preview_payload(str(run_root))
                                    state["seed_controls"] = dict(seed_controls)
                                except Exception:
                                    pass

                        rc, clog, final_output, chunk_count = chunk_and_process(
                            runner=_CancelProbe(),
                            settings=chunk_settings,
                            scene_threshold=scene_threshold,
                            min_scene_len=min_scene_len,
                            work_dir=Path(settings.get("_run_dir") or Path(global_settings.get("output_dir", output_dir))),
                            on_progress=lambda msg: progress_queue.put(msg),
                            chunk_seconds=chunk_size_sec,
                            chunk_overlap=chunk_overlap_sec,
                            per_chunk_cleanup=per_chunk_cleanup,
                            allow_partial=True,
                            global_output_dir=str(Path(settings.get("_run_dir") or Path(global_settings.get("output_dir", output_dir)))),
                            resume_from_partial=resume_requested,
                            progress_tracker=_chunk_progress_cb,
                            process_func=_process_chunk,
                            pre_process_chunks_func=_prepass_chunks,
                            model_type="open_h3",
                        )

                        result_holder["result"] = RunResult(rc, final_output if final_output else None, clog)
                        result_holder["chunk_count"] = int(chunk_count or 0)
                    else:
                        progress_queue.put(
                            "[OpenH3] App-level scene chunking OFF - the engine plans its own 17k+5-frame "
                            "windows (the whole clip as one window when VRAM allows)."
                        )
                        result = run_open_h3(
                            settings,
                            base_dir,
                            on_progress=lambda msg: progress_queue.put(msg),
                            cancel_event=_open_h3_cancel_event,
                            process_handle=process_handle,
                        )
                        result_holder["result"] = result
                except Exception as e:
                    result_holder["error"] = str(e)

            thread = threading.Thread(target=processing_thread, daemon=True)
            thread.start()

            # Stream progress updates.
            run_started_ts = time.time()
            last_update = time.time()
            log_buffer: List[str] = []
            live_progress_pct: Optional[float] = None
            live_progress_desc = ""
            live_log_idx: Optional[int] = None
            cmd_inline_progress_active = False
            cmd_inline_progress_width = 0

            def _strip_ansi(text: str) -> str:
                try:
                    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
                except Exception:
                    return text

            def _upsert_log_entry(text: str, transient: bool) -> None:
                nonlocal live_log_idx
                if not text:
                    return
                if transient:
                    if live_log_idx is None or live_log_idx >= len(log_buffer):
                        log_buffer.append(text)
                        live_log_idx = len(log_buffer) - 1
                    else:
                        log_buffer[live_log_idx] = text
                    return
                live_log_idx = None
                log_buffer.append(text)

            def _print_cmd_progress(text: str, transient: bool) -> None:
                nonlocal cmd_inline_progress_active, cmd_inline_progress_width
                if use_app_chunking:
                    return
                payload = str(text or "").rstrip("\r\n")
                if not payload:
                    return
                try:
                    inline_payload = transient and (
                        payload.startswith("Processing:")
                        or payload.startswith("[OpenH3] still running")
                    )
                    if inline_payload:
                        padded = payload
                        if cmd_inline_progress_width > len(payload):
                            padded = payload + (" " * (cmd_inline_progress_width - len(payload)))
                        print(f"\r{padded}", end="", flush=True)
                        cmd_inline_progress_active = True
                        cmd_inline_progress_width = len(payload)
                    else:
                        if cmd_inline_progress_active:
                            print("", flush=True)
                            cmd_inline_progress_active = False
                            cmd_inline_progress_width = 0
                        print(payload, flush=True)
                except Exception:
                    pass

            def _normalize_live_progress_line(msg_text: str) -> Tuple[str, bool]:
                text = _strip_ansi(str(msg_text or "")).strip()
                if not text:
                    return "", False
                # Per-sampler-step lines: "Processing: i/N | chunk c/M step s/S"
                if re.match(r"^Processing:\s*\d+\s*/\s*\d+", text, flags=re.IGNORECASE):
                    return text, True
                if text.lower().startswith("[openh3] still running"):
                    return text, True
                if re.match(r"^\s*\d+(?:\.\d+)?\s*%", text):
                    return text, True
                return text, False

            def _extract_progress(msg_text: str) -> Tuple[Optional[float], str]:
                text = _strip_ansi(str(msg_text or "")).strip()
                if not text:
                    return None, ""

                m = re.search(r"Processing:\s*(\d+)\s*/\s*(\d+)", text, flags=re.IGNORECASE)
                if m:
                    try:
                        num = int(m.group(1))
                        den = max(1, int(m.group(2)))
                        return max(0.0, min(1.0, float(num) / float(den))), text
                    except Exception:
                        return None, text

                m = re.search(r"^\s*(\d+(?:\.\d+)?)\s*%", text)
                if m:
                    try:
                        return max(0.0, min(100.0, float(m.group(1)))) / 100.0, text
                    except Exception:
                        return None, text

                return None, text

            if settings.get("_output_codec_note"):
                log_buffer.append(str(settings.get("_output_codec_note")))
            if settings.get("_multi_gpu_disabled_reason"):
                log_buffer.append(str(settings.get("_multi_gpu_disabled_reason")))
            if settings.get("_fps_mode_note"):
                log_buffer.append(str(settings.get("_fps_mode_note")))
            log_buffer.append(f"[OpenH3] Output size: {_target_note(settings)}")

            while thread.is_alive() or not progress_queue.empty():
                if _open_h3_cancel_event.is_set():
                    if process_handle.get("proc"):
                        try:
                            import platform

                            proc = process_handle["proc"]
                            if platform.system() == "Windows":
                                proc.terminate()
                            else:
                                proc.kill()
                        except Exception:
                            pass

                    if progress:
                        progress(0, desc="Cancelled")

                    # Try to salvage partial outputs (chunked runs only).
                    compiled_output = None
                    temp_base = Path(global_settings.get("temp_dir", temp_dir))
                    temp_chunks_dir = temp_base / "chunks"

                    if temp_chunks_dir.exists():
                        try:
                            from shared.chunking import detect_resume_state, concat_videos
                            from shared.path_utils import collision_safe_path as _collision_safe_path_local

                            partial_video, completed_chunks = detect_resume_state(temp_chunks_dir, "mp4")

                            if completed_chunks and len(completed_chunks) > 0:
                                partial_target = _collision_safe_path_local(
                                    temp_chunks_dir / "cancelled_OpenH3_partial.mp4"
                                )
                                if concat_videos(completed_chunks, partial_target, encode_settings=settings):
                                    final_output = Path(output_dir) / "cancelled_OpenH3_partial_upscaled.mp4"
                                    final_output = _collision_safe_path_local(final_output)
                                    shutil.copy2(partial_target, final_output)
                                    compiled_output = str(final_output)
                                    log_buffer.append(f"\n[OK] Partial output salvaged: {final_output.name}")
                        except Exception as e:
                            log_buffer.append(f"\n[WARN] Could not salvage partials: {str(e)}")

                    status_msg = "[STOP] Processing cancelled"
                    if compiled_output:
                        status_msg += f" - Partial output saved: {Path(compiled_output).name}"

                    vid_upd, img_upd = _media_updates(compiled_output)
                    yield (
                        status_msg,
                        "\n".join(log_buffer[-50:]) + "\n\n[Cancelled by user]",
                        vid_upd,
                        img_upd,
                        gr.update(visible=False),
                        gr.update(value="", visible=False),
                        state,
                    )
                    return

                try:
                    msg = progress_queue.get(timeout=0.1)
                    msg_clean, is_live_line = _normalize_live_progress_line(msg)
                    _print_cmd_progress(msg_clean, transient=is_live_line)
                    _upsert_log_entry(msg_clean, transient=is_live_line)
                    pct_val, _ = _extract_progress(msg_clean)

                    if msg_clean:
                        msg_lc = msg_clean.lower()
                        has_real_hint = (
                            pct_val is not None
                            or msg_lc.startswith("processing:")
                            or "chunk" in msg_lc
                            or "vae encoding" in msg_lc
                            or "vae decoding" in msg_lc
                            or "elapsed=" in msg_lc
                            or msg_lc.startswith("[openh3] still running")
                        )
                        if has_real_hint:
                            live_progress_desc = msg_clean

                    if pct_val is not None:
                        if live_progress_pct is None:
                            live_progress_pct = pct_val
                        else:
                            live_progress_pct = max(live_progress_pct, pct_val)
                        if progress:
                            progress(pct_val, desc=(msg_clean or str(msg))[:100])

                except queue.Empty:
                    pass

                now = time.time()
                if now - last_update > 0.5:
                    last_update = now
                    elapsed_s = int(now - run_started_ts)
                    if live_progress_desc:
                        status_live = f"[RUNNING] {live_progress_desc} | UI elapsed {elapsed_s}s"
                    elif live_progress_pct is not None:
                        status_live = f"[RUNNING] Open-H3 processing... {int(live_progress_pct * 100)}% ({elapsed_s}s)"
                    else:
                        status_live = f"[RUNNING] Open-H3 processing... {elapsed_s}s elapsed"
                    # gr.skip() leaves the four media components untouched on live
                    # ticks (gradio 6 fast path) — only status and log repaint.
                    yield (
                        status_live,
                        "\n".join(log_buffer[-50:]),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        state,
                    )

            thread.join()

            if "error" in result_holder:
                if progress:
                    progress(0, desc="Error")
                if maybe_set_vram_oom_alert(state, model_label="open_h3", text=result_holder.get("error", ""), settings=settings):
                    show_vram_oom_modal(state, title="Out of VRAM (GPU) - Open-H3", duration=None)
                yield (
                    (
                        "[OOM] Out of VRAM (GPU) - see banner above"
                        if state.get("alerts", {}).get("oom", {}).get("visible")
                        else "[ERROR] Processing failed"
                    ),
                    f"Error: {result_holder['error']}",
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    state,
                )
                return

            result = result_holder.get("result")
            if not result:
                yield (
                    "[ERROR] No result",
                    "Processing did not complete",
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    state,
                )
                return

            if progress:
                progress(1.0, desc="Open-H3 complete!")

            output_path = result.output_path

            chunk_count = int(result_holder.get("chunk_count") or 0)
            if chunk_count > 0:
                _cache_chunk_preview(Path(settings.get("_run_dir") or global_settings.get("output_dir", output_dir)))
            else:
                _cache_chunk_preview(None)

            # Face restore -> audio -> global RIFE -> comparison video.
            if output_path and Path(output_path).exists():
                output_path = _apply_video_post_passes(
                    output_path,
                    input_path,
                    lambda m: log_buffer.append(str(m)),
                    chunk_count,
                )

            # Final media probe for debugging timeline/fps drift.
            if output_path and Path(output_path).exists() and Path(output_path).suffix.lower() in video_exts:
                try:
                    dims = get_media_dimensions(output_path)
                    fps_val = get_media_fps(output_path)
                    dur_val = get_media_duration_seconds(output_path)
                    dim_txt = (
                        f"{int(dims[0])}x{int(dims[1])}"
                        if isinstance(dims, tuple) and len(dims) == 2
                        else "unknown"
                    )
                    fps_txt = f"{float(fps_val):.6g}" if fps_val and float(fps_val) > 0 else "unknown"
                    dur_txt = f"{float(dur_val):.6g}s" if dur_val and float(dur_val) > 0 else "unknown"
                    log_buffer.append(
                        f"[final] probe: name={Path(output_path).name}, dims={dim_txt}, fps={fps_txt}, duration={dur_txt}"
                    )
                except Exception:
                    pass

            html_comp, img_slider = create_unified_comparison(
                input_path=input_path,
                output_path=output_path,
                mode="slider" if output_path and Path(output_path).suffix.lower() in video_exts else "native",
            )

            if output_path:
                try:
                    outp = Path(output_path)
                    seed_controls = state.get("seed_controls", {})
                    seed_controls["last_output_dir"] = str(outp.parent if outp.is_file() else outp)
                    seed_controls["last_output_path"] = str(outp) if outp.is_file() else None
                    state["seed_controls"] = seed_controls
                except Exception:
                    pass

            if bool(settings.get("save_metadata", True)):
                run_logger.write_summary(
                    Path(output_path) if output_path else output_dir,
                    {
                        "input": input_path,
                        "output": output_path,
                        "returncode": result.returncode,
                        "args": settings,
                        "face_apply": face_apply,
                        "face_strength": face_strength,
                        "pipeline": "open_h3",
                        "target": _target_note(settings),
                        **_run_summary_extras(chunk_count),
                    },
                )

            done_name = "JonXL H3 Upscale" if is_jxl_workflow(settings) else "Open-H3 regenerate"
            if chunk_count > 0:
                status = (
                    f"[OK] {done_name} complete ({chunk_count} chunks)"
                    if result.returncode == 0
                    else f"[WARN] Chunked upscale exited with code {result.returncode}"
                )
            else:
                status = (
                    f"[OK] {done_name} complete"
                    if result.returncode == 0
                    else f"[WARN] Exited with code {result.returncode}"
                )

            if result.returncode != 0 and maybe_set_vram_oom_alert(state, model_label="open_h3", text=result.log, settings=settings):
                status = "Out of VRAM (GPU) - see banner above"
                show_vram_oom_modal(state, title="Out of VRAM (GPU) - Open-H3", duration=None)

            # Keep run folder self-documented.
            try:
                run_dir_raw = settings.get("_run_dir")
                if run_dir_raw:
                    effective_input = settings.get("_effective_input_path") or settings.get("input_path")
                    finalize_run_context(
                        Path(run_dir_raw),
                        pipeline="open_h3",
                        status=str(status or ""),
                        returncode=int(result.returncode),
                        output_path=str(output_path) if output_path else None,
                        original_input_path=str(input_path),
                        effective_input_path=str(effective_input) if effective_input else None,
                        preprocessed_input_path=None,
                        input_kind=input_kind_single,
                    )
            except Exception:
                pass

            vid_upd, img_upd = _media_updates(output_path)
            yield (
                status,
                ("\n".join(log_buffer[-400:]) if log_buffer else result.log),
                vid_upd,
                img_upd,
                img_slider if img_slider else gr.update(visible=False),
                html_comp if html_comp else gr.update(value="", visible=False),
                state,
            )

        except Exception as e:
            if progress:
                progress(0, desc="Critical error")
            if maybe_set_vram_oom_alert(state, model_label="open_h3", text=str(e), settings=locals().get("settings")):
                show_vram_oom_modal(state, title="Out of VRAM (GPU) - Open-H3", duration=None)
            yield (
                "[ERROR] Critical error",
                f"Error: {str(e)}",
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(visible=False),
                gr.update(value="", visible=False),
                state or {},
            )

    def cancel_action():
        """Cancel Open-H3 processing."""
        _open_h3_cancel_event.set()
        return gr.update(value="[STOP] Cancellation requested - Open-H3 will stop at next checkpoint"), "Cancelling..."

    def open_outputs_folder_open_h3():
        """Open outputs folder - delegates to shared utility (no code duplication)."""
        from shared.services.global_service import open_outputs_folder

        return open_outputs_folder(str(output_dir))

    def clear_temp_folder_open_h3(confirm: bool):
        """Clear temp folder - delegates to shared utility (no code duplication)."""
        from shared.services.global_service import clear_temp_folder

        return clear_temp_folder(str(temp_dir), confirm)

    return {
        "defaults": defaults,
        "order": OPENH3_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "auto_tune_action": auto_tune_action,
        "cli_plan_preview": cli_plan_preview_action,
        "live_plan": live_plan_action,
        "run_action": run_action,
        "cancel_action": cancel_action,
        "open_outputs_folder": open_outputs_folder_open_h3,
        "clear_temp_folder": clear_temp_folder_open_h3,
    }
