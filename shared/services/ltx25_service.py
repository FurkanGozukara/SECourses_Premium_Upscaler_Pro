"""
LTX 2.5 Upscaler Service Module
Handles LTX 2.5 processing logic, presets, and callbacks.

LTX 2.5 is a FIXED 2x pixel spatial video upscaler (IC-LoRA on the 22B DiT).
The engine (tools/ltx25_inference.py) plans token-budget chunks internally and
prefers processing the whole video as ONE chunk, so app-level scene chunking is
OFF by default here and only used when the user explicitly configures fixed
chunking in the Resolution tab.

Mirrors shared/services/sparkvsr_service.py for every shared app behavior.
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
from shared.models.ltx25_meta import (
    get_ltx25_default_model,
    get_ltx25_model_names,
    ltx25_sampling_defaults,
)
from shared.ltx25_constants import (
    LTX25_MODEL_NAMES,
    LTX25_TE_INT8,
    LTX25_TE_NAMES,
    LTX25_VAE_CONV,
    LTX25_VAE_NAMES,
    ltx25_is_distilled,
)
from shared.gpu_utils import get_global_gpu_override
from shared.oom_alert import clear_vram_oom_alert, maybe_set_vram_oom_alert, show_vram_oom_modal
from shared.output_run_manager import (
    prepare_single_video_run,
    batch_item_dir,
    resolve_resume_input_from_run_dir,
    finalize_run_context,
)
from shared.batch_output_cleanup import keep_only_batch_outputs
from shared.fixed_scale_output import enforce_fixed_scale_output_size
from shared.global_rife import maybe_apply_global_rife
from shared.comparison_video_service import maybe_generate_input_vs_output_comparison
from shared.chunk_preview import build_chunk_preview_payload

logger = logging.getLogger("LTX25Service")

# Cancel event for LTX 2.5 processing
_ltx25_cancel_event = threading.Event()

# LTX 2.5 is a fixed 2x pixel spatial upscaler.
LTX25_FIXED_SCALE = 2.0

LTX25_DEFAULT_POSITIVE = (
    "Faithfully re-render the input video at exactly twice its spatial resolution. "
    "Preserve the same subjects, identities, composition, framing, camera motion, actions, timing, "
    "lighting, colors, and scene content. Preserve every visible letter, number, caption, title, logo, "
    "and symbol exactly as written and in the same position. Add only natural fine detail, clean edges, "
    "realistic textures, and stable temporal consistency. Do not introduce, remove, redesign, or "
    "reinterpret anything."
)
LTX25_DEFAULT_NEGATIVE = (
    "new or missing subjects, changed identity, changed face, changed objects, changed composition, "
    "changed framing, crop, reframing, changed motion, changed timing, hallucinated content, invented "
    "detail, flicker, jitter, warping, morphing, oversharpening, ringing, halos, blur, compression "
    "artifacts, altered text, misspelled text, changed captions, changed titles, changed logos, "
    "changed symbols"
)

LTX25_SAMPLER_OPTIONS = ["euler_ancestral", "euler", "res_multistep"]
LTX25_TOKEN_BUDGET_MODE_AUTO = "Auto (Single Chunk Preferred)"
LTX25_TOKEN_BUDGET_MODE_MANUAL = "Manual"
LTX25_TOKEN_BUDGET_MODES = [LTX25_TOKEN_BUDGET_MODE_AUTO, LTX25_TOKEN_BUDGET_MODE_MANUAL]
LTX25_ATTENTION_OPTIONS = ["auto", "sage", "flash", "sdpa"]


def run_ltx25(*args, **kwargs):
    """Lazy-import LTX 2.5 runner so the Gradio process stays backend-light."""
    from shared.ltx25_runner import run_ltx25 as _run_ltx25_impl

    return _run_ltx25_impl(*args, **kwargs)


def run_ltx25_plan(*args, **kwargs):
    """Lazy-import the plan-only helper (chunk plan preview, no sampling)."""
    from shared.ltx25_runner import run_ltx25_plan as _run_ltx25_plan_impl

    return _run_ltx25_plan_impl(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Live chunk-plan preview (in-process, instant — mirrors the ComfyUI live plan)
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

    planner_path = Path(base_dir) / "LTX25" / "ltx25_engine" / "planner.py"
    spec = importlib.util.spec_from_file_location("_ltx25_live_planner", str(planner_path))
    module = importlib.util.module_from_spec(spec)
    # Must be registered before exec_module: @dataclass resolves the module's
    # (string) annotations through sys.modules[cls.__module__].
    sys.modules["_ltx25_live_planner"] = module
    spec.loader.exec_module(module)
    _PLANNER_MODULE_CACHE[key] = module
    return module


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
    """Free VRAM via NVML/nvidia-smi (no CUDA context); 3s cache."""
    now = time.time()
    if now - float(_PLAN_VRAM_CACHE.get("ts", 0.0)) > 3.0:
        try:
            from shared.gpu_utils import get_gpu_info

            _PLAN_VRAM_CACHE["gpus"] = get_gpu_info() or []
        except Exception:
            _PLAN_VRAM_CACHE["gpus"] = []
        _PLAN_VRAM_CACHE["ts"] = now
    gpus = _PLAN_VRAM_CACHE.get("gpus") or []
    if not gpus:
        return 0.0, "no NVIDIA GPU detected"
    wanted = str(device_value or "").strip()
    chosen = None
    if wanted.isdigit():
        for gpu in gpus:
            if str(getattr(gpu, "id", "")) == wanted:
                chosen = gpu
                break
    if chosen is None:
        chosen = max(gpus, key=lambda g: float(getattr(g, "available_memory_gb", 0.0) or 0.0))
    free_gb = float(getattr(chosen, "available_memory_gb", 0.0) or 0.0)
    label = f"GPU {getattr(chosen, 'id', '?')} free {free_gb:.1f}GB"
    return free_gb * (1024.0 ** 3), label


def build_live_plan_text(settings: Dict[str, Any], base_dir: Path) -> str:
    """Compute the chunk plan instantly in-process (same math as the engine)."""
    input_path = normalize_path(settings.get("_effective_input_path") or settings.get("input_path") or "")
    if not input_path or not Path(input_path).exists():
        return "Set an Input Path (or upload a video), then the live plan appears here."
    if detect_input_type(input_path) != "video":
        return "[ERROR] Chunk plan needs a video input - LTX 2.5 upscales videos only."

    probe = _probe_video_for_plan(input_path)
    if probe is None:
        return "[ERROR] Could not read video dimensions/frames (ffprobe failed)."
    width, height, total_frames, fps, frames_estimated = probe

    # Frame subset (mirrors the engine's start/end handling).
    start_frame = max(0, int(settings.get("start_frame") or 0))
    end_frame = int(settings.get("end_frame") if settings.get("end_frame") is not None else -1)
    if end_frame >= 0:
        total_frames = max(0, min(total_frames, end_frame + 1) - start_frame)
    else:
        total_frames = max(0, total_frames - start_frame)
    if total_frames <= 0:
        return "[ERROR] Start/End Frame selection leaves no frames to process."

    # Pre-resize (mirrors Ltx25Pipeline._pre_resize rounding).
    pre_resize = int(settings.get("pre_resize_longer_edge") or 0)
    if pre_resize > 0 and max(width, height) > pre_resize:
        scale = pre_resize / max(width, height)
        width = max(2, int(round(width * scale / 2)) * 2)
        height = max(2, int(round(height * scale / 2)) * 2)

    planner = _planner_module(base_dir)
    lines: List[str] = []
    manual_mode = str(settings.get("token_budget_mode") or "").strip().lower().startswith("manual")
    if manual_mode:
        budget = int(settings.get("max_latent_tokens") or 18000)
    else:
        free_bytes, vram_label = _free_vram_bytes_for_plan(str(settings.get("device") or ""))
        if free_bytes <= 0:
            budget = int(settings.get("max_latent_tokens") or 18000)
            lines.append(
                f"[Plan] auto budget unavailable ({vram_label}) - using Manual value {budget}"
            )
        else:
            budget, reason = planner.auto_token_budget(
                total_frames,
                width,
                height,
                free_vram_bytes=int(free_bytes),
                weight_resident_bytes=0,
                reserve_gb=float(settings.get("reserve_vram_gb") or 1.0),
            )
            lines.append(f"[Plan] auto budget {budget} ({reason}) | {vram_label}")

    plan = planner.plan_chunks(
        total_frames,
        width,
        height,
        budget,
        max_chunk_frames=int(settings.get("max_chunk_frames") or 121),
        overlap_frames=int(settings.get("overlap_frames") or 1),
    )
    steps = int(settings.get("steps") or 8)
    lines.extend(f"[Plan] {line}" for line in plan.summary(steps).splitlines())
    if frames_estimated:
        lines.append(f"[Plan] note: frame count estimated from duration x fps ({fps:.3f}).")
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
            f"{src.as_posix()}|{st.st_size}|{st.st_mtime_ns}|LTX25_ui_preview_v1".encode("utf-8")
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


def ltx25_defaults() -> Dict[str, Any]:
    """Defaults aligned to the LTX 2.5 engine CLI and the ComfyUI upscaler workflow."""
    try:
        from shared.gpu_utils import get_gpu_info

        cuda_default = "auto" if get_gpu_info() else "cpu"
    except Exception:
        cuda_default = "cpu"

    return {
        "input_path": "",
        "output_override": "",
        "output_format": "mp4",
        "model_name": get_ltx25_default_model(),
        "text_encoder": LTX25_TE_INT8,
        "video_vae": LTX25_VAE_CONV,
        "positive_prompt": LTX25_DEFAULT_POSITIVE,
        "negative_prompt": LTX25_DEFAULT_NEGATIVE,
        "seed": 42,
        "randomize_seed": False,
        "steps": 8,
        "cfg": 1.0,
        "sampler": "euler_ancestral",
        "ic_lora_strength": 1.0,
        "guide_strength": 1.0,
        "token_budget_mode": LTX25_TOKEN_BUDGET_MODE_AUTO,
        "max_latent_tokens": 18000,
        "max_chunk_frames": 121,
        "overlap_frames": 1,
        "reserve_vram_gb": 1.0,
        "attention_backend": "auto",
        "encode_tile_size": 512,
        "encode_tile_overlap": 64,
        "decode_tile_size": 512,
        "decode_tile_overlap": 64,
        "decode_temporal_size": 128,
        "decode_temporal_overlap": 32,
        "pre_resize_longer_edge": 0,
        "fps": 0.0,
        "start_frame": 0,
        "end_frame": -1,
        "device": cuda_default,
        "save_metadata": True,
        "auto_transfer_output_to_input": False,
        "batch_enable": False,
        "batch_input_path": "",
        "batch_output_path": "",
        "resume_run_dir": "",
        "keep_only_output_files": False,
    }


LTX25_ORDER: List[str] = [
    "input_path",
    "output_override",
    "output_format",
    "model_name",
    "text_encoder",
    "video_vae",
    "positive_prompt",
    "negative_prompt",
    "seed",
    "randomize_seed",
    "steps",
    "cfg",
    "sampler",
    "ic_lora_strength",
    "guide_strength",
    "token_budget_mode",
    "max_latent_tokens",
    "max_chunk_frames",
    "overlap_frames",
    "reserve_vram_gb",
    "attention_backend",
    "encode_tile_size",
    "encode_tile_overlap",
    "decode_tile_size",
    "decode_tile_overlap",
    "decode_temporal_size",
    "decode_temporal_overlap",
    "pre_resize_longer_edge",
    "fps",
    "start_frame",
    "end_frame",
    "device",
    "save_metadata",
    "auto_transfer_output_to_input",
    "batch_enable",
    "batch_input_path",
    "batch_output_path",
    "resume_run_dir",
    "keep_only_output_files",
]


def _ltx25_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    return dict(zip(LTX25_ORDER, args))


def _enforce_ltx25_guardrails(cfg: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {**defaults, **(cfg or {})}

    cfg["output_format"] = "mp4"
    cfg["save_metadata"] = _to_bool(cfg.get("save_metadata"), _to_bool(defaults.get("save_metadata", True), True))
    cfg["randomize_seed"] = _to_bool(cfg.get("randomize_seed"), _to_bool(defaults.get("randomize_seed", False), False))
    cfg["auto_transfer_output_to_input"] = _to_bool(
        cfg.get("auto_transfer_output_to_input"),
        _to_bool(defaults.get("auto_transfer_output_to_input", False), False),
    )
    cfg["batch_enable"] = _to_bool(cfg.get("batch_enable"), _to_bool(defaults.get("batch_enable", False), False))
    cfg["keep_only_output_files"] = _to_bool(
        cfg.get("keep_only_output_files"),
        _to_bool(defaults.get("keep_only_output_files", False), False),
    )

    # Model / text encoder / VAE names validated against constants.
    try:
        valid_model_names = set(get_ltx25_model_names())
    except Exception:
        valid_model_names = set(LTX25_MODEL_NAMES)
    model_name = str(cfg.get("model_name", defaults.get("model_name")) or "").strip()
    if model_name not in valid_model_names:
        model_name = str(defaults.get("model_name") or get_ltx25_default_model())
    cfg["model_name"] = model_name

    text_encoder = str(cfg.get("text_encoder", defaults.get("text_encoder")) or "").strip()
    cfg["text_encoder"] = text_encoder if text_encoder in set(LTX25_TE_NAMES) else LTX25_TE_INT8
    video_vae = str(cfg.get("video_vae", defaults.get("video_vae")) or "").strip()
    cfg["video_vae"] = video_vae if video_vae in set(LTX25_VAE_NAMES) else LTX25_VAE_CONV

    # Schedule is derived from the variant family (Distilled -> distilled sigmas,
    # Dev -> 20-step quality schedule) and passed straight to the runner.
    cfg["schedule"] = "distilled" if ltx25_is_distilled(model_name) else "dev"

    # Prompts (long conditioning texts by design; keep a sane hard cap).
    cfg["positive_prompt"] = str(cfg.get("positive_prompt", LTX25_DEFAULT_POSITIVE) or "").strip()[:4000]
    cfg["negative_prompt"] = str(cfg.get("negative_prompt", LTX25_DEFAULT_NEGATIVE) or "").strip()[:4000]

    # Sampling (fallbacks are model-aware; the values stay fully user-editable).
    model_sampling = ltx25_sampling_defaults(model_name)
    cfg["seed"] = max(0, _to_int(cfg.get("seed"), 42))
    cfg["steps"] = max(1, min(100, _to_int(cfg.get("steps"), int(model_sampling["steps"]))))
    cfg["cfg"] = max(0.0, min(15.0, _to_float(cfg.get("cfg"), float(model_sampling["cfg"]))))
    sampler = str(cfg.get("sampler", defaults.get("sampler", "euler_ancestral")) or "").strip().lower()
    cfg["sampler"] = sampler if sampler in set(LTX25_SAMPLER_OPTIONS) else "euler_ancestral"
    cfg["ic_lora_strength"] = max(0.0, min(2.0, _to_float(cfg.get("ic_lora_strength"), 1.0)))
    cfg["guide_strength"] = max(0.0, min(1.0, _to_float(cfg.get("guide_strength"), 1.0)))

    # Token budget / chunk planning.
    token_mode_raw = str(cfg.get("token_budget_mode", LTX25_TOKEN_BUDGET_MODE_AUTO) or "").strip()
    if token_mode_raw in set(LTX25_TOKEN_BUDGET_MODES):
        cfg["token_budget_mode"] = token_mode_raw
    elif "manual" in token_mode_raw.lower():
        cfg["token_budget_mode"] = LTX25_TOKEN_BUDGET_MODE_MANUAL
    else:
        cfg["token_budget_mode"] = LTX25_TOKEN_BUDGET_MODE_AUTO
    # ComfyUI LTX25UpscaleControls parity: budget may go up to 9,999,999 so any
    # video can be forced through as a single chunk in Manual mode.
    cfg["max_latent_tokens"] = max(4096, min(9999999, _to_int(cfg.get("max_latent_tokens"), 18000)))
    cfg["max_chunk_frames"] = _snap_down_8n1(_to_int(cfg.get("max_chunk_frames"), 121), 9, 1025)
    overlap = _snap_down_8n1(_to_int(cfg.get("overlap_frames"), 1), 1, 257)
    if overlap >= cfg["max_chunk_frames"]:
        overlap = _snap_down_8n1(max(1, cfg["max_chunk_frames"] - 8), 1, 257)
    cfg["overlap_frames"] = overlap
    cfg["reserve_vram_gb"] = max(0.0, min(8.0, _to_float(cfg.get("reserve_vram_gb"), 1.0)))
    attention = str(cfg.get("attention_backend", "auto") or "auto").strip().lower()
    cfg["attention_backend"] = attention if attention in set(LTX25_ATTENTION_OPTIONS) else "auto"

    # VAE tiling (clamp only, no snapping).
    cfg["encode_tile_size"] = max(256, min(1024, _to_int(cfg.get("encode_tile_size"), 512)))
    cfg["encode_tile_overlap"] = max(16, min(256, _to_int(cfg.get("encode_tile_overlap"), 64)))
    if cfg["encode_tile_overlap"] >= cfg["encode_tile_size"]:
        cfg["encode_tile_overlap"] = max(16, cfg["encode_tile_size"] // 4)
    cfg["decode_tile_size"] = max(256, min(1024, _to_int(cfg.get("decode_tile_size"), 512)))
    cfg["decode_tile_overlap"] = max(16, min(256, _to_int(cfg.get("decode_tile_overlap"), 64)))
    if cfg["decode_tile_overlap"] >= cfg["decode_tile_size"]:
        cfg["decode_tile_overlap"] = max(16, cfg["decode_tile_size"] // 4)
    cfg["decode_temporal_size"] = max(16, min(256, _to_int(cfg.get("decode_temporal_size"), 128)))
    cfg["decode_temporal_overlap"] = max(8, min(128, _to_int(cfg.get("decode_temporal_overlap"), 32)))
    if cfg["decode_temporal_overlap"] >= cfg["decode_temporal_size"]:
        cfg["decode_temporal_overlap"] = max(8, cfg["decode_temporal_size"] // 4)

    # Input handling.
    pre_resize = _to_int(cfg.get("pre_resize_longer_edge"), 0)
    if pre_resize <= 0:
        pre_resize = 0
    else:
        pre_resize = max(256, min(8192, pre_resize))
    cfg["pre_resize_longer_edge"] = pre_resize
    cfg["fps"] = max(0.0, _to_float(cfg.get("fps"), 0.0))
    cfg["start_frame"] = max(0, _to_int(cfg.get("start_frame"), 0))
    end_frame = _to_int(cfg.get("end_frame"), -1)
    cfg["end_frame"] = end_frame if end_frame >= 0 else -1

    # Paths.
    for key in ("input_path", "output_override", "batch_input_path", "batch_output_path", "resume_run_dir"):
        cfg[key] = str(cfg.get(key, defaults.get(key, "")) or "").strip()

    # Single-GPU guardrail (mirrors SparkVSR).
    device_str = str(cfg.get("device", defaults.get("device", "auto")) or "auto").strip()
    if "," in device_str:
        cfg["device"] = device_str.split(",")[0].strip() or "auto"
        cfg["_multi_gpu_disabled_reason"] = "LTX 2.5 is single-GPU."
    else:
        cfg["device"] = device_str or "auto"

    return cfg


def _apply_ltx25_preset(
    preset: Dict[str, Any],
    defaults: Dict[str, Any],
    preset_manager: PresetManager,
    current: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    base = defaults.copy()
    if current:
        base.update(current)
    merged = preset_manager.merge_config(base, preset)
    merged = _enforce_ltx25_guardrails(merged, defaults)
    return [merged[k] for k in LTX25_ORDER]


def build_ltx25_callbacks(
    preset_manager: PresetManager,
    runner,
    run_logger: RunLogger,
    global_settings: Dict[str, Any],
    shared_state: gr.State,
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path,
):
    """Build LTX 2.5 callback functions for the UI."""
    defaults = ltx25_defaults()

    def refresh_presets(model_name: str, select_name: Optional[str] = None):
        """Refresh preset dropdown."""
        presets = preset_manager.list_presets("ltx25", model_name)
        last_used = preset_manager.get_last_used_name("ltx25", model_name)
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        return gr.update(choices=presets, value=value)

    def save_preset(preset_name: str, *args):
        """Save a preset."""
        if not preset_name.strip():
            return gr.update(), gr.update(value="[WARN] Enter a preset name"), *list(args)

        try:
            payload = _ltx25_dict_from_args(list(args))
            model_name = str(payload.get("model_name") or get_ltx25_default_model())

            preset_manager.save_preset_safe("ltx25", model_name, preset_name.strip(), payload)
            dropdown = refresh_presets(model_name, select_name=preset_name.strip())

            current_map = dict(zip(LTX25_ORDER, list(args)))
            loaded_vals = _apply_ltx25_preset(payload, defaults, preset_manager, current=current_map)

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
            model_name = str(model_name or get_ltx25_default_model())
            preset = preset_manager.load_preset_safe("ltx25", model_name, preset_name)
            if preset:
                preset_manager.set_last_used("ltx25", model_name, preset_name)

            current_map = dict(zip(LTX25_ORDER, current_values))
            values = _apply_ltx25_preset(preset or {}, defaults, preset_manager, current=current_map)

            status_msg = f"[OK] Loaded preset '{preset_name}'" if preset else "[INFO] Preset not found"
            return (*values, gr.update(value=status_msg))
        except Exception as e:
            logger.error(f"Error loading preset {preset_name}: {e}")
            return (*current_values, gr.update(value=f"[ERROR] Error: {str(e)}"))

    def safe_defaults():
        """Get safe default values."""
        normalized = _enforce_ltx25_guardrails(defaults.copy(), defaults)
        return [normalized[key] for key in LTX25_ORDER]

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

        settings = {**defaults, **_ltx25_dict_from_args(list(args))}
        settings = _enforce_ltx25_guardrails(settings, defaults)
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

        settings = {**defaults, **_ltx25_dict_from_args(list(args))}
        settings = _enforce_ltx25_guardrails(settings, defaults)
        global_gpu_device = get_global_gpu_override(seed_controls, snapshot)
        settings["device"] = "cpu" if global_gpu_device == "cpu" else str(global_gpu_device)

        input_path = normalize_path(uploaded_file if uploaded_file else settings.get("input_path") or "")
        if not input_path or not Path(input_path).exists():
            yield _error_payload(
                state,
                "[ERROR] Input path missing",
                "Provide a video input (upload or path) before previewing the chunk plan.",
            )
            return
        if detect_input_type(input_path) != "video":
            yield _error_payload(
                state,
                "[ERROR] Chunk plan needs a video input",
                "LTX 2.5 upscales videos only; the chunk planner cannot analyze this input type.",
            )
            return
        settings["input_path"] = input_path
        settings["_effective_input_path"] = input_path

        _ltx25_cancel_event.clear()
        if progress:
            progress(0, desc="Planning LTX 2.5 chunks...")

        plan_lines: List[str] = []
        line_queue: "queue.Queue[str]" = queue.Queue()
        result_holder: Dict[str, Any] = {}

        def _plan_thread():
            try:
                rc, log_text = run_ltx25_plan(
                    settings,
                    base_dir,
                    on_progress=lambda msg: line_queue.put(str(msg)),
                    cancel_event=_ltx25_cancel_event,
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
            "Starting the LTX 2.5 planner. The engine reads the video, sizes the token budget "
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
        if rc == 130 or _ltx25_cancel_event.is_set():
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
                f"Recommendation: {chunk_count_hint} chunks planned. For fewer chunks (better quality), "
                "free VRAM, lower Reserve VRAM, reduce Pre-Resize Longer Edge, or trim the frame range."
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
            seed_controls["ltx25_chunk_preview"] = {
                "message": "No chunk preview available yet.",
                "gallery": [],
                "videos": [],
                "count": 0,
            }
            seed_controls["ltx25_batch_outputs"] = []
            state["seed_controls"] = seed_controls
            settings_dict = _ltx25_dict_from_args(list(args))
            settings = {**defaults, **settings_dict}
            if settings.get("batch_enable"):
                settings["resume_run_dir"] = ""
            settings["device"] = "cpu" if global_gpu_device == "cpu" else str(global_gpu_device)

            settings = _enforce_ltx25_guardrails(settings, defaults)

            if preview_only:
                yield _error_payload(
                    state,
                    "[INFO] Preview is not available for LTX 2.5",
                    "The LTX 2.5 engine upscales whole videos (temporal model); a single-frame preview "
                    "is not meaningful. Use the Preview Chunk Plan button instead.",
                )
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

            # LTX 2.5 tab uses global Output tab video settings for the engine writer.
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
                    f"Output tab codec '{output_codec_key}' is not supported by the LTX 2.5 writer; "
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

            _ltx25_cancel_event.clear()

            if progress:
                progress(0, desc="Initializing LTX 2.5...")

            def _cache_chunk_preview(run_dir: Optional[Path]) -> None:
                try:
                    if not run_dir:
                        seed_controls["ltx25_chunk_preview"] = {
                            "message": "No chunk preview available.",
                            "gallery": [],
                            "videos": [],
                            "count": 0,
                        }
                    else:
                        seed_controls["ltx25_chunk_preview"] = build_chunk_preview_payload(str(run_dir))
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

            # Universal chunking settings (Resolution tab). LTX 2.5 defaults to
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
                """Face restore -> audio -> RIFE -> comparison video, in SparkVSR order."""
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
                        label_output="ltx25",
                        on_progress=(lambda m: logs_sink(m.strip()) if m else None),
                    )
                    if comp_vid_path:
                        logs_sink(f"[OK] Comparison video created: {Path(comp_vid_path).name}")
                    elif comp_vid_err:
                        logs_sink(f"[WARN] Comparison video failed: {comp_vid_err}")
                return outp

            # -------------------------------------------------------------
            # Batch processing (folder of videos; LTX 2.5 is video-only)
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
                            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                                items.append(p)
                            elif p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                                skipped_non_video += 1
                        except Exception:
                            continue
                except Exception:
                    items = []

                if not items:
                    yield _error_payload(
                        state,
                        "[ERROR] No video files found in batch input folder",
                        "LTX 2.5 batch processes video files only (images/frame folders are not supported "
                        "by this fixed-2x video engine).",
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
                    seed_controls["ltx25_batch_outputs"] = list(outputs)
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
                    progress(0, desc=f"Batch: {len(items)} video(s) queued")
                logs.append(f"Batch queued: {len(items)} video(s). Output: {batch_root}")
                if skipped_non_video:
                    logs.append(
                        f"[WARN] Skipped {skipped_non_video} non-video item(s); LTX 2.5 is a video-only 2x upscaler."
                    )
                if not use_app_chunking:
                    logs.append(
                        "[LTX25] App-level scene chunking is OFF (engine token-budget chunking active). "
                        "Disable Auto Chunk and set a fixed chunk size in the Resolution tab to force it."
                    )
                yield _batch_live_payload(f"Starting LTX 2.5 batch ({len(items)} video(s))")

                for idx, item in enumerate(items, 1):
                    if _ltx25_cancel_event.is_set():
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

                    item_settings = settings.copy()
                    item_settings["batch_enable"] = False
                    item_settings["input_path"] = item_path
                    item_settings["_effective_input_path"] = item_path
                    item_settings["_original_filename"] = Path(item_path).name
                    item_input = Path(item_path)

                    item_out_dir = batch_item_dir(batch_root, item_input.name)
                    base_no_ext = item_input.stem
                    predicted_output_file = item_out_dir / f"{base_no_ext}.mp4"

                    from shared.output_run_manager import prepare_batch_video_run_dir

                    run_paths = prepare_batch_video_run_dir(
                        batch_root,
                        item_input.name,
                        input_path=str(item_input),
                        model_label="ltx25",
                        mode=str(getattr(runner, "get_mode", lambda: "subprocess")() or "subprocess"),
                        overwrite_existing=overwrite_existing,
                    )
                    if not run_paths:
                        if not overwrite_existing:
                            logs.append(
                                f"[SKIP] [{idx}/{len(items)}] {Path(item_path).name} skipped (output folder exists)"
                            )
                            if predicted_output_file.exists():
                                outputs.append(str(predicted_output_file))
                            yield _batch_live_payload(f"Batch {idx}/{len(items)}: skipped {item_name}")
                            continue
                        logs.append(
                            f"[ERROR] [{idx}/{len(items)}] {Path(item_path).name} failed (could not create output folder)"
                        )
                        yield _batch_live_payload(f"Batch {idx}/{len(items)}: could not prepare output for {item_name}")
                        continue

                    item_settings["global_output_dir"] = str(run_paths.run_dir)
                    item_settings["_run_dir"] = str(run_paths.run_dir)
                    item_settings["_processed_chunks_dir"] = str(run_paths.processed_chunks_dir)
                    item_settings["output_override"] = str(predicted_output_file)

                    chunk_count_item = 0
                    if use_app_chunking:
                        from shared.chunking import chunk_and_process
                        from shared.runner import RunResult

                        class _CancelProbe:
                            def is_canceled(self) -> bool:
                                return bool(_ltx25_cancel_event.is_set())

                        chunk_settings = item_settings.copy()
                        chunk_settings["frame_accurate_split"] = frame_accurate_split
                        try:
                            base_stem = Path(chunk_settings.get("_original_filename") or item_path).stem
                            chunk_settings["output_override"] = str(Path(item_out_dir) / f"{base_stem}.mp4")
                        except Exception:
                            pass

                        def _process_chunk(s: Dict[str, Any], on_progress=None) -> RunResult:
                            r = run_ltx25(
                                s,
                                base_dir,
                                on_progress=on_progress,
                                cancel_event=_ltx25_cancel_event,
                                process_handle=None,
                            )
                            return RunResult(r.returncode, r.output_path, r.log)

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
                            pre_process_chunks_func=None,
                            model_type="ltx25",
                        )
                        result = RunResult(rc, final_output if final_output else None, clog)
                        outp = result.output_path
                    else:
                        result = run_ltx25(
                            item_settings,
                            base_dir,
                            on_progress=None,
                            cancel_event=_ltx25_cancel_event,
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
                                state, model_label="ltx25", text=getattr(result, "log", ""), settings=item_settings
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
                                    "pipeline": "ltx25",
                                    "batch": True,
                                    "fixed_scale": LTX25_FIXED_SCALE,
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
                seed_controls["ltx25_batch_outputs"] = list(outputs)
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
                    f"[OK] LTX 2.5 batch complete ({len(outputs)}/{len(items)} succeeded, "
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
            if input_kind_single != "video":
                yield _error_payload(
                    state,
                    "[ERROR] LTX 2.5 expects a video input",
                    (
                        f"Detected input type: {input_kind_single}. This fixed-2x engine upscales videos only. "
                        "Use SeedVR2/GAN tabs for single images or convert frame folders to a video first."
                    ),
                )
                return

            settings["input_path"] = input_path
            settings["_effective_input_path"] = input_path
            if not settings.get("_original_filename"):
                settings["_original_filename"] = Path(input_path).name
            seed_controls["ltx25_batch_outputs"] = []
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
                        model_label="ltx25",
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

            # Run LTX 2.5 in a worker thread with cancel support.
            result_holder: Dict[str, Any] = {}
            process_handle = {"proc": None}

            def processing_thread():
                try:
                    if use_app_chunking:
                        from shared.chunking import chunk_and_process
                        from shared.runner import RunResult

                        class _CancelProbe:
                            def is_canceled(self) -> bool:
                                return bool(_ltx25_cancel_event.is_set())

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

                        def _process_chunk(s: Dict[str, Any], on_progress=None) -> RunResult:
                            r = run_ltx25(
                                s,
                                base_dir,
                                on_progress=on_progress,
                                cancel_event=_ltx25_cancel_event,
                                process_handle=None,
                            )
                            return RunResult(r.returncode, r.output_path, r.log)

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
                                    seed_controls["ltx25_chunk_preview"] = build_chunk_preview_payload(str(run_root))
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
                            pre_process_chunks_func=None,
                            model_type="ltx25",
                        )

                        result_holder["result"] = RunResult(rc, final_output if final_output else None, clog)
                        result_holder["chunk_count"] = int(chunk_count or 0)
                    else:
                        progress_queue.put(
                            "[LTX25] App-level scene chunking OFF - the engine plans token-budget chunks "
                            "internally (single chunk preferred)."
                        )
                        result = run_ltx25(
                            settings,
                            base_dir,
                            on_progress=lambda msg: progress_queue.put(msg),
                            cancel_event=_ltx25_cancel_event,
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
                        or payload.startswith("[LTX25] still running")
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
                if text.lower().startswith("[ltx25] still running"):
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
            if int(settings.get("pre_resize_longer_edge") or 0) > 0:
                log_buffer.append(
                    f"[LTX25] Pre-resize longer edge: {int(settings['pre_resize_longer_edge'])}px "
                    "(output = exactly 2x the pre-resized input)."
                )

            while thread.is_alive() or not progress_queue.empty():
                if _ltx25_cancel_event.is_set():
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
                                    temp_chunks_dir / "cancelled_LTX25_partial.mp4"
                                )
                                if concat_videos(completed_chunks, partial_target, encode_settings=settings):
                                    final_output = Path(output_dir) / "cancelled_LTX25_partial_upscaled.mp4"
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
                            or msg_lc.startswith("[ltx25] still running")
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
                        status_live = f"[RUNNING] LTX 2.5 processing... {int(live_progress_pct * 100)}% ({elapsed_s}s)"
                    else:
                        status_live = f"[RUNNING] LTX 2.5 processing... {elapsed_s}s elapsed"
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
                if maybe_set_vram_oom_alert(state, model_label="ltx25", text=result_holder.get("error", ""), settings=settings):
                    show_vram_oom_modal(state, title="Out of VRAM (GPU) - LTX 2.5", duration=None)
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
                progress(1.0, desc="LTX 2.5 complete!")

            output_path = result.output_path

            # Fixed-scale enforcement: LTX 2.5 is exactly 2x. When a pre-resize
            # is configured, the engine defines the output size (2x the resized
            # input), so the 2x-vs-original check is skipped.
            if output_path and Path(output_path).exists():
                if int(settings.get("pre_resize_longer_edge") or 0) <= 0:
                    output_path_new, resized_final, resize_msg = enforce_fixed_scale_output_size(
                        output_path=output_path,
                        source_input_path=settings.get("_effective_input_path") or input_path,
                        requested_scale=LTX25_FIXED_SCALE,
                        model_scale=LTX25_FIXED_SCALE,
                        max_edge=0,
                        pre_downscale_then_upscale=False,
                        settings=settings,
                        label="LTX 2.5 final sizing",
                        on_log=lambda x: log_buffer.append(x) if x else None,
                    )
                    if output_path_new and output_path_new != output_path:
                        output_path = output_path_new
                        result.output_path = output_path
                    if resize_msg and (resized_final or "failed" in resize_msg.lower()):
                        log_buffer.append(resize_msg)
                else:
                    log_buffer.append(
                        "[LTX25] Fixed-scale check skipped: output = exactly 2x the pre-resized input."
                    )

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
                mode="slider" if output_path and output_path.endswith(".mp4") else "native",
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
                        "pipeline": "ltx25",
                        "fixed_scale": LTX25_FIXED_SCALE,
                        **_run_summary_extras(chunk_count),
                    },
                )

            if chunk_count > 0:
                status = (
                    f"[OK] LTX 2.5 chunked 2x upscale complete ({chunk_count} chunks)"
                    if result.returncode == 0
                    else f"[WARN] Chunked upscale exited with code {result.returncode}"
                )
            else:
                status = (
                    "[OK] LTX 2.5 2x upscaling complete"
                    if result.returncode == 0
                    else f"[WARN] Exited with code {result.returncode}"
                )

            if result.returncode != 0 and maybe_set_vram_oom_alert(state, model_label="ltx25", text=result.log, settings=settings):
                status = "Out of VRAM (GPU) - see banner above"
                show_vram_oom_modal(state, title="Out of VRAM (GPU) - LTX 2.5", duration=None)

            # Keep run folder self-documented.
            try:
                run_dir_raw = settings.get("_run_dir")
                if run_dir_raw:
                    effective_input = settings.get("_effective_input_path") or settings.get("input_path")
                    finalize_run_context(
                        Path(run_dir_raw),
                        pipeline="ltx25",
                        status=str(status or ""),
                        returncode=int(result.returncode),
                        output_path=str(output_path) if output_path else None,
                        original_input_path=str(input_path),
                        effective_input_path=str(effective_input) if effective_input else None,
                        preprocessed_input_path=None,
                        input_kind="video",
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
            if maybe_set_vram_oom_alert(state, model_label="ltx25", text=str(e), settings=locals().get("settings")):
                show_vram_oom_modal(state, title="Out of VRAM (GPU) - LTX 2.5", duration=None)
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
        """Cancel LTX 2.5 processing."""
        _ltx25_cancel_event.set()
        return gr.update(value="[STOP] Cancellation requested - LTX 2.5 will stop at next checkpoint"), "Cancelling..."

    def open_outputs_folder_LTX25():
        """Open outputs folder - delegates to shared utility (no code duplication)."""
        from shared.services.global_service import open_outputs_folder

        return open_outputs_folder(str(output_dir))

    def clear_temp_folder_LTX25(confirm: bool):
        """Clear temp folder - delegates to shared utility (no code duplication)."""
        from shared.services.global_service import clear_temp_folder

        return clear_temp_folder(str(temp_dir), confirm)

    return {
        "defaults": defaults,
        "order": LTX25_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "auto_tune_action": auto_tune_action,
        "live_plan": live_plan_action,
        "run_action": run_action,
        "cancel_action": cancel_action,
        "open_outputs_folder": open_outputs_folder_LTX25,
        "clear_temp_folder": clear_temp_folder_LTX25,
    }
