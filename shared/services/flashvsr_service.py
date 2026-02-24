"""
FlashVSR+ Service Module
Handles FlashVSR+ processing logic, presets, and callbacks
"""

import queue
import re
import hashlib
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr

from shared.preset_manager import PresetManager
from shared.flashvsr_runner import run_flashvsr, FlashVSRResult
from shared.path_utils import (
    normalize_path,
    collision_safe_path,
    collision_safe_dir,
    get_media_dimensions,
    get_media_duration_seconds,
    get_media_fps,
    detect_input_type,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    resolve_batch_output_dir,
)
from shared.logging_utils import RunLogger
from shared.comparison_unified import create_unified_comparison
from shared.models.flashvsr_meta import (
    get_flashvsr_metadata,
    get_flashvsr_default_model,
    flashvsr_version_to_internal,
    flashvsr_version_to_ui,
)
from shared.gpu_utils import expand_cuda_device_spec, get_global_gpu_override, validate_cuda_device_spec
from shared.error_handling import logger as error_logger
from shared.resolution_calculator import estimate_fixed_scale_upscale_plan_from_dims
from shared.oom_alert import clear_vram_oom_alert, maybe_set_vram_oom_alert, show_vram_oom_modal
from shared.output_run_manager import (
    prepare_single_video_run,
    batch_item_dir,
    downscaled_video_path,
    resolve_resume_input_from_run_dir,
    ensure_image_input_artifact,
    finalize_run_context,
)
from shared.ffmpeg_utils import scale_video
from shared.global_rife import maybe_apply_global_rife
from shared.comparison_video_service import maybe_generate_input_vs_output_comparison
from shared.chunk_preview import build_chunk_preview_payload
from shared.preview_utils import prepare_preview_input

# Cancel event for FlashVSR+ processing
_flashvsr_cancel_event = threading.Event()


def _save_preprocessed_artifact(pre_path: Path, output_path_str: str) -> Optional[str]:
    """
    Save the preprocessed (downscaled) input next to outputs for inspection.

    Requirement:
    - Save into a `pre_processed/` folder inside the output folder
    - Use the SAME base name as the final output
    """
    try:
        if not pre_path or not pre_path.exists():
            return None

        outp = Path(output_path_str)
        parent = outp.parent if outp.suffix else outp.parent
        pre_dir = parent / "pre_processed"
        pre_dir.mkdir(parents=True, exist_ok=True)

        base = outp.stem if outp.suffix else outp.name

        if pre_path.is_dir():
            dest_dir = collision_safe_dir(pre_dir / base)
            shutil.copytree(pre_path, dest_dir, dirs_exist_ok=False)
            return str(dest_dir)

        dest_file = collision_safe_path(pre_dir / f"{base}{pre_path.suffix}")
        shutil.copy2(pre_path, dest_file)
        return str(dest_file)
    except Exception:
        return None


def _apply_image_output_preferences(
    image_path: Optional[str],
    image_output_format: Any,
    image_output_quality: Any,
) -> Optional[str]:
    """Convert a finalized image output to the globally configured format/quality."""
    if not image_path:
        return image_path
    try:
        src = Path(str(image_path))
        if (not src.exists()) or src.is_dir():
            return image_path

        fmt = str(image_output_format or "png").strip().lower()
        if fmt not in {"png", "jpg", "webp"}:
            fmt = "png"

        try:
            quality = int(float(image_output_quality or 95))
        except Exception:
            quality = 95
        quality = max(1, min(100, quality))

        target_ext = ".jpg" if fmt == "jpg" else f".{fmt}"
        needs_reencode = src.suffix.lower() != target_ext or fmt in {"jpg", "webp"}
        if not needs_reencode:
            return image_path

        from PIL import Image

        dst = collision_safe_path(src.with_suffix(target_ext))
        with Image.open(src) as img:
            if fmt == "jpg":
                img = img.convert("RGB")
                img.save(dst, format="JPEG", quality=quality)
            elif fmt == "webp":
                img.save(dst, format="WEBP", quality=quality)
            else:
                img.save(dst, format="PNG")

        try:
            if dst.resolve() != src.resolve():
                src.unlink(missing_ok=True)
        except Exception:
            pass

        return str(dst)
    except Exception:
        return image_path


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
            f"{src.as_posix()}|{st.st_size}|{st.st_mtime_ns}|flashvsr_ui_preview_v1".encode("utf-8")
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


FLASHVSR_VAE_OPTIONS = ["Wan2.1", "Wan2.2", "LightVAE_W2.1", "TAE_W2.2", "LightTAE_HY1.5"]
FLASHVSR_PRECISION_OPTIONS = ["auto", "bf16", "fp16"]
FLASHVSR_ATTENTION_OPTIONS = ["sparse_sage_attention", "block_sparse_attention", "flash_attention_2", "sdpa"]
FLASHVSR_CODEC_OPTIONS = ["libx264", "libx265", "h264_nvenc"]


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


def _nearest_supported_scale(value: Any, default: int = 4) -> int:
    raw = _to_float(value, float(default))
    return 2 if raw <= 3.0 else 4


def canonical_flashvsr_scale(
    *,
    scale_value: Any = None,
    upscale_factor_value: Any = None,
    default: Any = 4,
) -> int:
    """
    Resolve FlashVSR scale to backend-supported values (2x or 4x).

    `upscale_factor_value` takes precedence over `scale_value` so UI slider
    updates cannot be overridden by stale hidden/state fields.
    """
    default_scale = _nearest_supported_scale(default, 4)
    if upscale_factor_value is not None:
        return _nearest_supported_scale(upscale_factor_value, default_scale)
    return _nearest_supported_scale(scale_value, default_scale)


def flashvsr_defaults(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Defaults aligned to ComfyUI-FlashVSR_Stable with quality-first tuning.
    """
    try:
        from shared.gpu_utils import get_gpu_info

        cuda_default = "auto" if get_gpu_info() else "cpu"
    except Exception:
        cuda_default = "cpu"

    default_model = model_name or get_flashvsr_default_model()
    model_meta = get_flashvsr_metadata(default_model)

    if model_meta:
        default_precision = "bf16"
        default_tile_size = model_meta.default_tile_size
        default_overlap = model_meta.default_overlap
        default_attention = "flash_attention_2"
        default_vae = "Wan2.2"
        mode = model_meta.mode
        scale = int(model_meta.scale)
        frame_chunk_size = int(model_meta.recommended_frame_chunk_size)
        keep_models_on_cpu = bool(model_meta.default_keep_models_on_cpu)
        version = flashvsr_version_to_ui(model_meta.version)
        tiled_vae = False
        tiled_dit = True
    else:
        default_precision = "bf16"
        default_tile_size = 256
        default_overlap = 24
        default_attention = "flash_attention_2"
        default_vae = "Wan2.2"
        version = "1.1"
        mode = "full"
        scale = 4
        frame_chunk_size = 64
        keep_models_on_cpu = True
        tiled_vae = False
        tiled_dit = True

    return {
        "input_path": "",
        "output_override": "",
        "output_format": "mp4",
        "scale": scale,
        "version": version,
        "mode": mode,
        "vae_model": default_vae,
        "precision": default_precision,
        "attention_mode": default_attention,
        "tiled_vae": tiled_vae,
        "tiled_dit": tiled_dit,
        "tile_size": default_tile_size,
        "overlap": default_overlap,
        "unload_dit": True,
        "stream_decode": False,
        "cfg_scale": 1.0,
        "denoise_amount": 1.0,
        "sparse_ratio": 2.0,
        "kv_ratio": 3.0,
        "local_range": 11,
        "frame_chunk_size": frame_chunk_size,
        "keep_models_on_cpu": keep_models_on_cpu,
        "force_offload": True,
        "enable_debug": False,
        "color_fix": True,
        "seed": 0,
        "device": cuda_default,
        "fps": 0.0,
        "codec": "libx264",
        "crf": 18,
        "start_frame": 0,
        "end_frame": -1,
        "models_dir": str((Path(__file__).resolve().parents[2] / "ComfyUI-FlashVSR_Stable" / "models")),
        "save_metadata": True,
        "face_restore_after_upscale": False,
        "batch_enable": False,
        "batch_input_path": "",
        "batch_output_path": "",
        "resume_run_dir": "",
        "use_resolution_tab": True,
        "upscale_factor": float(scale),
        "max_target_resolution": 1920,
        "pre_downscale_then_upscale": True,
    }


FLASHVSR_ORDER: List[str] = [
    "input_path",
    "output_override",
    "output_format",
    "scale",
    "version",
    "mode",
    "vae_model",
    "precision",
    "attention_mode",
    "tiled_vae",
    "tiled_dit",
    "tile_size",
    "overlap",
    "unload_dit",
    "stream_decode",
    "cfg_scale",
    "denoise_amount",
    "sparse_ratio",
    "kv_ratio",
    "local_range",
    "frame_chunk_size",
    "keep_models_on_cpu",
    "force_offload",
    "enable_debug",
    "color_fix",
    "seed",
    "device",
    "fps",
    "codec",
    "crf",
    "start_frame",
    "end_frame",
    "models_dir",
    "save_metadata",
    "face_restore_after_upscale",
    "batch_enable",
    "batch_input_path",
    "batch_output_path",
    "use_resolution_tab",
    "upscale_factor",
    "max_target_resolution",
    "pre_downscale_then_upscale",
    "resume_run_dir",
]


def _flashvsr_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    return dict(zip(FLASHVSR_ORDER, args))


def _enforce_flashvsr_guardrails(cfg: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {**defaults, **(cfg or {})}

    cfg["output_format"] = "mp4"
    cfg["save_metadata"] = _to_bool(cfg.get("save_metadata"), _to_bool(defaults.get("save_metadata", True), True))
    cfg["face_restore_after_upscale"] = _to_bool(
        cfg.get("face_restore_after_upscale"),
        _to_bool(defaults.get("face_restore_after_upscale", False), False),
    )

    cfg["version"] = flashvsr_version_to_ui(cfg.get("version", defaults.get("version", "1.1")))
    mode = str(cfg.get("mode", defaults.get("mode", "full")) or "full").strip().lower()
    cfg["mode"] = mode if mode in {"tiny", "tiny-long", "full"} else "full"
    scale = canonical_flashvsr_scale(
        scale_value=cfg.get("scale", defaults.get("scale", 4)),
        upscale_factor_value=cfg.get("upscale_factor"),
        default=defaults.get("upscale_factor", defaults.get("scale", 4)),
    )
    cfg["scale"] = str(scale)
    cfg["upscale_factor"] = float(scale)

    precision = str(cfg.get("precision", cfg.get("dtype", defaults.get("precision", "auto")))).strip().lower()
    cfg["precision"] = precision if precision in {"auto", "bf16", "fp16"} else "auto"

    att_raw = str(cfg.get("attention_mode", cfg.get("attention", defaults.get("attention_mode", "flash_attention_2")))).strip().lower()
    att_map = {
        "sage": "sparse_sage_attention",
        "sparse_sage": "sparse_sage_attention",
        "sparse_sage_attention": "sparse_sage_attention",
        "block": "block_sparse_attention",
        "block_sparse": "block_sparse_attention",
        "block_sparse_attention": "block_sparse_attention",
        "flash_attn_2": "flash_attention_2",
        "flash_attention_2": "flash_attention_2",
        "sdpa": "sdpa",
    }
    cfg["attention_mode"] = att_map.get(att_raw, "flash_attention_2")

    vae_model = str(cfg.get("vae_model", defaults.get("vae_model", "Wan2.2"))).strip()
    cfg["vae_model"] = vae_model if vae_model in FLASHVSR_VAE_OPTIONS else str(defaults.get("vae_model", "Wan2.2"))

    cfg["tiled_vae"] = _to_bool(cfg.get("tiled_vae"), _to_bool(defaults.get("tiled_vae", True), True))
    cfg["tiled_dit"] = _to_bool(cfg.get("tiled_dit"), _to_bool(defaults.get("tiled_dit", False), False))
    # Guardrail: full mode + tiled_vae can produce heavily corrupted output on some systems.
    if cfg.get("mode") == "full" and cfg.get("tiled_vae"):
        cfg["tiled_vae"] = False
        cfg["_tiled_vae_note"] = "Disabled tiled_vae in full mode to avoid known corruption artifacts."
    cfg["unload_dit"] = _to_bool(cfg.get("unload_dit"), _to_bool(defaults.get("unload_dit", False), False))
    cfg["stream_decode"] = _to_bool(cfg.get("stream_decode"), _to_bool(defaults.get("stream_decode", False), False))
    cfg["cfg_scale"] = max(0.5, min(2.0, _to_float(cfg.get("cfg_scale"), _to_float(defaults.get("cfg_scale"), 1.0))))
    cfg["denoise_amount"] = max(
        0.5,
        min(
            2.0,
            _to_float(
                cfg.get("denoise_amount", cfg.get("denoising_strength")),
                _to_float(defaults.get("denoise_amount", defaults.get("denoising_strength", 1.0)), 1.0),
            ),
        ),
    )
    cfg["color_fix"] = _to_bool(cfg.get("color_fix"), _to_bool(defaults.get("color_fix", True), True))
    cfg["keep_models_on_cpu"] = _to_bool(
        cfg.get("keep_models_on_cpu"),
        _to_bool(defaults.get("keep_models_on_cpu", True), True),
    )
    cfg["force_offload"] = _to_bool(cfg.get("force_offload"), _to_bool(defaults.get("force_offload", True), True))
    cfg["enable_debug"] = _to_bool(cfg.get("enable_debug"), _to_bool(defaults.get("enable_debug", False), False))

    cfg["seed"] = max(0, _to_int(cfg.get("seed"), 0))
    cfg["tile_size"] = max(32, min(1024, _to_int(cfg.get("tile_size"), 256)))
    cfg["overlap"] = max(8, min(512, _to_int(cfg.get("overlap"), 24)))
    if cfg["overlap"] >= cfg["tile_size"]:
        cfg["overlap"] = max(8, cfg["tile_size"] - 8)

    cfg["sparse_ratio"] = max(1.5, min(2.0, _to_float(cfg.get("sparse_ratio"), 2.0)))
    cfg["kv_ratio"] = max(1.0, min(3.0, _to_float(cfg.get("kv_ratio"), 3.0)))
    cfg["local_range"] = 9 if _to_int(cfg.get("local_range"), 11) == 9 else 11
    cfg["frame_chunk_size"] = max(0, _to_int(cfg.get("frame_chunk_size"), 0))

    cfg["fps"] = max(0.0, _to_float(cfg.get("fps"), 0.0))
    codec = str(cfg.get("codec", defaults.get("codec", "libx264")) or "libx264").strip()
    cfg["codec"] = codec if codec else "libx264"
    cfg["crf"] = max(0, min(51, _to_int(cfg.get("crf"), 18)))
    cfg["start_frame"] = max(0, _to_int(cfg.get("start_frame"), 0))
    cfg["end_frame"] = _to_int(cfg.get("end_frame"), -1)

    cfg["models_dir"] = str(cfg.get("models_dir", defaults.get("models_dir", "")) or "").strip()
    cfg["batch_enable"] = _to_bool(cfg.get("batch_enable"), _to_bool(defaults.get("batch_enable", False), False))
    cfg["use_resolution_tab"] = _to_bool(
        cfg.get("use_resolution_tab"),
        _to_bool(defaults.get("use_resolution_tab", True), True),
    )
    cfg["max_target_resolution"] = max(0, min(8192, _to_int(cfg.get("max_target_resolution"), 1920)))
    cfg["pre_downscale_then_upscale"] = _to_bool(
        cfg.get("pre_downscale_then_upscale"),
        _to_bool(defaults.get("pre_downscale_then_upscale", True), True),
    )

    # Single-GPU guardrail
    device_str = str(cfg.get("device", defaults.get("device", "auto")) or "auto").strip()
    if "," in device_str:
        cfg["device"] = device_str.split(",")[0].strip() or "auto"
        cfg["_multi_gpu_disabled_reason"] = "FlashVSR is single-GPU."
    else:
        cfg["device"] = device_str or "auto"

    # Apply metadata defaults for selected version/mode/scale.
    internal_version = flashvsr_version_to_internal(cfg.get("version", "1.1"))
    model_id = f"v{internal_version}_{cfg.get('mode', 'full')}_{cfg.get('scale', '4')}x"
    model_meta = get_flashvsr_metadata(model_id)
    if model_meta:
        if cfg.get("tile_size", 0) <= 0:
            cfg["tile_size"] = int(model_meta.default_tile_size)
        if cfg.get("overlap", 0) <= 0:
            cfg["overlap"] = int(model_meta.default_overlap)

    # Compatibility aliases for older presets/components.
    cfg["dtype"] = cfg.get("precision", "auto")
    cfg["attention"] = cfg.get("attention_mode", "flash_attention_2")
    cfg["quality"] = cfg.get("crf", 18)

    return cfg


def _apply_flashvsr_preset(
    preset: Dict[str, Any],
    defaults: Dict[str, Any],
    preset_manager: PresetManager,
    current: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    base = defaults.copy()
    if current:
        base.update(current)
    merged = preset_manager.merge_config(base, preset)
    # Apply guardrails to merged preset
    merged = _enforce_flashvsr_guardrails(merged, defaults)
    return [merged[k] for k in FLASHVSR_ORDER]


def build_flashvsr_callbacks(
    preset_manager: PresetManager,
    runner,
    run_logger: RunLogger,
    global_settings: Dict[str, Any],
    shared_state: gr.State,
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path,
):
    """Build FlashVSR+ callback functions for the UI."""
    defaults = flashvsr_defaults()

    def refresh_presets(model_name: str, select_name: Optional[str] = None):
        """Refresh preset dropdown."""
        presets = preset_manager.list_presets("flashvsr", model_name)
        last_used = preset_manager.get_last_used_name("flashvsr", model_name)
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        return gr.update(choices=presets, value=value)

    def save_preset(preset_name: str, *args):
        """Save a preset."""
        if not preset_name.strip():
            return gr.update(), gr.update(value="⚠️ Enter a preset name"), *list(args)

        try:
            payload = _flashvsr_dict_from_args(list(args))
            internal_version = flashvsr_version_to_internal(payload.get("version", "1.1"))
            model_name = f"v{internal_version}_{payload['mode']}"
            
            preset_manager.save_preset_safe("flashvsr", model_name, preset_name.strip(), payload)
            dropdown = refresh_presets(model_name, select_name=preset_name.strip())

            current_map = dict(zip(FLASHVSR_ORDER, list(args)))
            loaded_vals = _apply_flashvsr_preset(payload, defaults, preset_manager, current=current_map)

            return dropdown, gr.update(value=f"✅ Saved preset '{preset_name}'"), *loaded_vals
        except Exception as e:
            return gr.update(), gr.update(value=f"❌ Error: {str(e)}"), *list(args)

    def load_preset(preset_name: str, version: str, mode: str, current_values: List[Any]):
        """
        Load a preset.
        
        FIXED: Now returns (*values, status_message) to match UI output expectations.
        UI expects: inputs_list + [preset_status]
        """
        try:
            internal_version = flashvsr_version_to_internal(version)
            model_name = f"v{internal_version}_{mode}"
            preset = preset_manager.load_preset_safe("flashvsr", model_name, preset_name)
            if preset:
                preset_manager.set_last_used("flashvsr", model_name, preset_name)

            current_map = dict(zip(FLASHVSR_ORDER, current_values))
            values = _apply_flashvsr_preset(preset or {}, defaults, preset_manager, current=current_map)
            
            # Return values + status message (status is LAST)
            status_msg = f"✅ Loaded preset '{preset_name}'" if preset else "ℹ️ Preset not found"
            return (*values, gr.update(value=status_msg))
        except Exception as e:
            print(f"Error loading preset {preset_name}: {e}")
            # Return current values + error status
            return (*current_values, gr.update(value=f"❌ Error: {str(e)}"))

    def safe_defaults():
        """Get safe default values."""
        normalized = _enforce_flashvsr_guardrails(defaults.copy(), defaults)
        return [normalized[key] for key in FLASHVSR_ORDER]

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
            # Clear any previous VRAM OOM banner at the start of a new run.
            clear_vram_oom_alert(state)
            seed_controls = state.get("seed_controls", {})
            output_settings = seed_controls.get("output_settings", {}) if isinstance(seed_controls, dict) else {}
            if not isinstance(output_settings, dict):
                output_settings = {}
            global_gpu_device = get_global_gpu_override(seed_controls, global_settings)
            seed_controls["global_gpu_device_val"] = global_gpu_device
            seed_controls["global_rife_cuda_device_val"] = "" if global_gpu_device == "cpu" else global_gpu_device
            seed_controls["flashvsr_chunk_preview"] = {
                "message": "No chunk preview available yet.",
                "gallery": [],
                "videos": [],
                "count": 0,
            }
            seed_controls["flashvsr_batch_outputs"] = []
            state["seed_controls"] = seed_controls
            settings_dict = _flashvsr_dict_from_args(list(args))
            settings = {**defaults, **settings_dict}
            if preview_only:
                settings["resume_run_dir"] = ""
            if settings.get("batch_enable"):
                settings["resume_run_dir"] = ""
            settings["device"] = "cpu" if global_gpu_device == "cpu" else str(global_gpu_device)
            
            # Apply FlashVSR+ guardrails (single GPU, tile validation, etc.)
            settings = _enforce_flashvsr_guardrails(settings, defaults)
            if preview_only:
                settings["batch_enable"] = False
            use_global_resolution = bool(settings.get("use_resolution_tab", True))
            
            # Apply shared Resolution tab scale with FlashVSR backend limits (2x or 4x only).
            if use_global_resolution:
                if seed_controls.get("upscale_factor_val") is not None:
                    try:
                        global_scale = float(seed_controls["upscale_factor_val"])
                        fixed_scale = canonical_flashvsr_scale(
                            scale_value=settings.get("scale", 4),
                            upscale_factor_value=global_scale,
                            default=settings.get("upscale_factor", settings.get("scale", 4)),
                        )
                        settings["scale"] = str(fixed_scale)
                        settings["upscale_factor"] = float(fixed_scale)
                        if abs(global_scale - float(fixed_scale)) > 0.01:
                            settings["_scale_clamp_note"] = (
                                f"Resolution tab requested {global_scale:.2f}x; FlashVSR supports only 2x/4x, "
                                f"so {fixed_scale}x will be used."
                            )
                    except Exception:
                        pass
                # Max-edge cap and pre-downscale toggle are local to FlashVSR tab.
                # Do NOT override them from Resolution-tab defaults (which are chunking-focused).
                # Legacy bridge: only use a global max if local max is unset (<= 0) and a legacy
                # explicit value is present in shared state.
                try:
                    local_max_edge = int(settings.get("max_target_resolution", 0) or 0)
                    legacy_max_edge = seed_controls.get("max_resolution_val")
                    if local_max_edge <= 0 and legacy_max_edge is not None:
                        merged_max = max(0, min(8192, int(float(legacy_max_edge or 0))))
                        settings["max_target_resolution"] = merged_max
                        settings["_max_edge_merge_note"] = (
                            f"Applied legacy shared max edge fallback: {merged_max}px "
                            "(local max was unset)."
                        )
                except Exception:
                    pass

            # Apply Output tab cached settings
            fps_override_val = seed_controls.get("fps_override_val")
            if fps_override_val is None and isinstance(output_settings, dict):
                fps_override_val = output_settings.get("fps_override")
            if fps_override_val is not None:
                try:
                    fps_override_num = float(fps_override_val or 0.0)
                except Exception:
                    fps_override_num = 0.0
                settings["fps"] = fps_override_num if fps_override_num > 0 else 0.0
            if seed_controls.get("comparison_mode_val"):
                settings["_comparison_mode"] = seed_controls["comparison_mode_val"]
            settings["save_metadata"] = bool(
                seed_controls.get(
                    "save_metadata_val",
                    output_settings.get("save_metadata", settings.get("save_metadata", True)),
                )
            )
            image_fmt = str(
                seed_controls.get(
                    "image_output_format_val",
                    output_settings.get("image_output_format", "png"),
                )
                or "png"
            ).strip().lower()
            if image_fmt not in {"png", "jpg", "webp"}:
                image_fmt = "png"
            settings["image_output_format"] = image_fmt
            try:
                image_quality = int(
                    float(
                        seed_controls.get(
                            "image_output_quality_val",
                            output_settings.get("image_output_quality", 95),
                        )
                        or 95
                    )
                )
            except Exception:
                image_quality = 95
            settings["image_output_quality"] = max(1, min(100, image_quality))
            # Audio mux preferences (used by chunking + final output postprocessing)
            if seed_controls.get("audio_codec_val") is not None:
                settings["audio_codec"] = seed_controls.get("audio_codec_val") or "copy"
            if seed_controls.get("audio_bitrate_val") is not None:
                settings["audio_bitrate"] = seed_controls.get("audio_bitrate_val") or ""
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

            # FlashVSR tab uses global Output tab video settings for model-pass encoding.
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
                    f"Output tab codec '{output_codec_key}' is not supported by FlashVSR CLI writer; "
                    f"using '{mapped_codec}' instead."
                )

            output_quality = settings.get("video_quality")
            if output_quality is None and isinstance(output_settings, dict):
                output_quality = output_settings.get("video_quality")
            try:
                output_quality_i = int(float(output_quality if output_quality is not None else 18))
            except Exception:
                output_quality_i = 18
            settings["crf"] = max(0, min(51, output_quality_i))

            face_apply = bool(settings.get("face_restore_after_upscale", False)) or bool(global_settings.get("face_global", False))
            face_strength = float(global_settings.get("face_strength", 0.5))
            
            # Clear cancel event
            _flashvsr_cancel_event.clear()

            def _apply_vnext_preprocess(cfg: Dict[str, Any], src_input_path: str) -> None:
                """
                vNext preprocessing for fixed-scale FlashVSR models (2x/4x).

                If the requested upscale (and/or max-edge cap) implies an effective scale < model_scale,
                pre-downscale the input so the model still runs at its native scale without post-downscale.

                Supports:
                - Video files (ffmpeg scale)
                - Single image files (resize to temporary image)
                - Frame directories (resize images into a temp directory)
                """
                try:
                    dims = get_media_dimensions(src_input_path)
                    if not dims:
                        return
                    w, h = dims
                    model_scale = int(cfg.get("scale", 4) or 4)
                    requested_scale = float(cfg.get("upscale_factor") or float(model_scale))
                    max_edge = int(cfg.get("max_target_resolution", 0) or 0)
                    pre_downscale_enabled = bool(cfg.get("pre_downscale_then_upscale", True))

                    plan = estimate_fixed_scale_upscale_plan_from_dims(
                        int(w),
                        int(h),
                        requested_scale=requested_scale,
                        model_scale=model_scale,
                        max_edge=max_edge,
                        force_pre_downscale=pre_downscale_enabled,
                    )

                    cfg["_preprocess_plan_note"] = (
                        f"Sizing plan: input={int(w)}x{int(h)}, model_scale={int(model_scale)}x, "
                        f"requested_scale={float(requested_scale):.3f}x, max_edge={int(max_edge)}, "
                        f"pre_scale={float(plan.preprocess_scale):.6f}, "
                        f"target={int(plan.resize_width)}x{int(plan.resize_height)}."
                    )

                    if plan.preprocess_scale >= 0.999999:
                        return

                    in_type = detect_input_type(src_input_path)

                    if in_type == "video":
                        out_root = Path(
                            cfg.get("_run_dir") or cfg.get("global_output_dir") or global_settings.get("output_dir", output_dir)
                        )
                        original_name = cfg.get("_original_filename") or Path(src_input_path).name
                        pre_out = downscaled_video_path(out_root, str(original_name))
                        if progress:
                            progress(0, desc=f"Preprocessing input → {int(plan.preprocess_width)}×{int(plan.preprocess_height)}")
                        ok, _err = scale_video(
                            Path(src_input_path),
                            Path(pre_out),
                            int(plan.preprocess_width),
                            int(plan.preprocess_height),
                            lossless=True,
                            audio_copy_first=True,
                        )
                        if ok and Path(pre_out).exists():
                            cfg["_original_input_path_before_preprocess"] = src_input_path
                            cfg["_preprocessed_input_path"] = str(pre_out)
                            cfg["_effective_input_path"] = str(pre_out)
                            cfg["_preprocess_note"] = (
                                f"Pre-downscale applied: {int(w)}x{int(h)} → "
                                f"{int(plan.preprocess_width)}x{int(plan.preprocess_height)} "
                                f"(max edge cap {int(max_edge) if int(max_edge) > 0 else 'off'})."
                            )
                        return

                    if in_type == "image":
                        out_root = Path(
                            cfg.get("_run_dir") or cfg.get("global_output_dir") or global_settings.get("output_dir", output_dir)
                        )
                        pre_dir = out_root / "pre_processed"
                        pre_dir.mkdir(parents=True, exist_ok=True)
                        src_file = Path(src_input_path)
                        pre_file = collision_safe_path(
                            pre_dir / f"{src_file.stem}_pre{int(plan.preprocess_width)}x{int(plan.preprocess_height)}{src_file.suffix}"
                        )

                        if progress:
                            progress(0, desc=f"Preprocessing image → {int(plan.preprocess_width)}×{int(plan.preprocess_height)}")

                        saved = False
                        try:
                            import cv2  # type: ignore

                            img = cv2.imread(str(src_file), cv2.IMREAD_UNCHANGED)
                            if img is not None:
                                resized = cv2.resize(
                                    img,
                                    (int(plan.preprocess_width), int(plan.preprocess_height)),
                                    interpolation=cv2.INTER_AREA,
                                )
                                saved = bool(cv2.imwrite(str(pre_file), resized))
                        except Exception:
                            saved = False

                        if not saved:
                            try:
                                from PIL import Image  # type: ignore

                                with Image.open(src_file) as im:
                                    im2 = im.resize(
                                        (int(plan.preprocess_width), int(plan.preprocess_height)),
                                        resample=Image.LANCZOS,
                                    )
                                    im2.save(pre_file)
                                    saved = True
                            except Exception:
                                saved = False

                        if saved and pre_file.exists():
                            cfg["_original_input_path_before_preprocess"] = src_input_path
                            cfg["_preprocessed_input_path"] = str(pre_file)
                            cfg["_effective_input_path"] = str(pre_file)
                            cfg["_preprocess_note"] = (
                                f"Pre-downscale applied: {int(w)}x{int(h)} → "
                                f"{int(plan.preprocess_width)}x{int(plan.preprocess_height)} "
                                f"(max edge cap {int(max_edge) if int(max_edge) > 0 else 'off'})."
                            )
                        return

                    if in_type == "directory":
                        temp_root = Path(global_settings.get("temp_dir", temp_dir))
                        temp_root.mkdir(parents=True, exist_ok=True)
                        src_dir = Path(src_input_path)
                        img_files = [p for p in sorted(src_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
                        if not img_files:
                            return

                        pre_dir = collision_safe_dir(
                            temp_root
                            / f"{src_dir.name}_pre{int(plan.preprocess_width)}x{int(plan.preprocess_height)}"
                        )
                        pre_dir.mkdir(parents=True, exist_ok=True)

                        if progress:
                            progress(0, desc=f"Preprocessing frames → {int(plan.preprocess_width)}×{int(plan.preprocess_height)}")

                        # Prefer OpenCV for speed; fall back to Pillow.
                        try:
                            import cv2  # type: ignore
                            for f in img_files:
                                img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                                if img is None:
                                    continue
                                resized = cv2.resize(
                                    img,
                                    (int(plan.preprocess_width), int(plan.preprocess_height)),
                                    interpolation=cv2.INTER_AREA,
                                )
                                cv2.imwrite(str(pre_dir / f.name), resized)
                        except Exception:
                            try:
                                from PIL import Image  # type: ignore
                                for f in img_files:
                                    with Image.open(f) as im:
                                        im2 = im.resize(
                                            (int(plan.preprocess_width), int(plan.preprocess_height)),
                                            resample=Image.LANCZOS,
                                        )
                                        im2.save(pre_dir / f.name)
                            except Exception:
                                return

                        if any(pre_dir.iterdir()):
                            cfg["_original_input_path_before_preprocess"] = src_input_path
                            cfg["_preprocessed_input_path"] = str(pre_dir)
                            cfg["_effective_input_path"] = str(pre_dir)
                            cfg["_preprocess_note"] = (
                                f"Pre-downscale applied: {int(w)}x{int(h)} → "
                                f"{int(plan.preprocess_width)}x{int(plan.preprocess_height)} "
                                f"(max edge cap {int(max_edge) if int(max_edge) > 0 else 'off'})."
                            )
                        return
                except Exception:
                    return

            def _enforce_preprocess_requirements(cfg: Dict[str, Any], src_input_path: str) -> None:
                """
                Verify preprocess actually took effect when a max-edge cap requires it.
                If missing, run a robust fallback preprocess (ffmpeg/cv2/Pillow) and
                mark the run as failed if we still cannot enforce the cap path.
                """
                cfg["_preprocess_required_but_missing"] = False
                try:
                    dims = get_media_dimensions(src_input_path)
                    if not dims:
                        return
                    w, h = int(dims[0]), int(dims[1])
                    model_scale = int(cfg.get("scale", 4) or 4)
                    requested_scale = float(cfg.get("upscale_factor") or float(model_scale))
                    max_edge = int(cfg.get("max_target_resolution", 0) or 0)
                    pre_downscale_enabled = bool(cfg.get("pre_downscale_then_upscale", True))

                    plan = estimate_fixed_scale_upscale_plan_from_dims(
                        w,
                        h,
                        requested_scale=requested_scale,
                        model_scale=model_scale,
                        max_edge=max_edge,
                        force_pre_downscale=pre_downscale_enabled,
                    )
                    if plan.preprocess_scale >= 0.999999:
                        return

                    src_norm = normalize_path(src_input_path) or str(src_input_path)
                    eff_norm = normalize_path(cfg.get("_effective_input_path")) or src_norm
                    if eff_norm != src_norm and Path(eff_norm).exists():
                        return

                    in_type = detect_input_type(src_input_path)
                    fallback_done = False

                    def _mark_applied(preprocessed_path: str) -> None:
                        cfg["_original_input_path_before_preprocess"] = src_input_path
                        cfg["_preprocessed_input_path"] = str(preprocessed_path)
                        cfg["_effective_input_path"] = str(preprocessed_path)
                        cfg["_preprocess_note"] = (
                            f"Pre-downscale applied (fallback): {int(w)}x{int(h)} -> "
                            f"{int(plan.preprocess_width)}x{int(plan.preprocess_height)} "
                            f"(max edge cap {int(max_edge) if int(max_edge) > 0 else 'off'})."
                        )

                    if in_type == "video":
                        out_root = Path(
                            cfg.get("_run_dir") or cfg.get("global_output_dir") or global_settings.get("output_dir", output_dir)
                        )
                        original_name = cfg.get("_original_filename") or Path(src_input_path).name
                        pre_out = downscaled_video_path(out_root, str(original_name))
                        ok, err = scale_video(
                            Path(src_input_path),
                            Path(pre_out),
                            int(plan.preprocess_width),
                            int(plan.preprocess_height),
                            lossless=True,
                            audio_copy_first=True,
                        )
                        if ok and Path(pre_out).exists():
                            _mark_applied(str(pre_out))
                            fallback_done = True
                        else:
                            cfg["_preprocess_error_note"] = (
                                "Pre-downscale fallback failed for video input: "
                                f"{str(err or 'ffmpeg scale failed').strip()}"
                            )
                    elif in_type in {"image", "unknown"} and Path(src_input_path).is_file():
                        out_root = Path(
                            cfg.get("_run_dir") or cfg.get("global_output_dir") or global_settings.get("output_dir", output_dir)
                        )
                        pre_dir = out_root / "pre_processed"
                        pre_dir.mkdir(parents=True, exist_ok=True)
                        src_file = Path(src_input_path)
                        pre_file = collision_safe_path(
                            pre_dir / f"{src_file.stem}_pre{int(plan.preprocess_width)}x{int(plan.preprocess_height)}{src_file.suffix}"
                        )

                        saved = False
                        ffmpeg_err = ""
                        try:
                            ff = subprocess.run(
                                [
                                    "ffmpeg",
                                    "-y",
                                    "-i",
                                    str(src_file),
                                    "-vf",
                                    f"scale={int(plan.preprocess_width)}:{int(plan.preprocess_height)}:flags=lanczos",
                                    "-frames:v",
                                    "1",
                                    str(pre_file),
                                ],
                                capture_output=True,
                                text=True,
                            )
                            saved = bool(ff.returncode == 0 and pre_file.exists() and pre_file.stat().st_size > 0)
                            if not saved:
                                ffmpeg_err = (ff.stderr or ff.stdout or "").strip()
                        except Exception as e:
                            ffmpeg_err = str(e)

                        if not saved:
                            try:
                                import cv2  # type: ignore

                                img = cv2.imread(str(src_file), cv2.IMREAD_UNCHANGED)
                                if img is not None:
                                    resized = cv2.resize(
                                        img,
                                        (int(plan.preprocess_width), int(plan.preprocess_height)),
                                        interpolation=cv2.INTER_AREA,
                                    )
                                    saved = bool(cv2.imwrite(str(pre_file), resized))
                            except Exception:
                                saved = False

                        if not saved:
                            try:
                                from PIL import Image  # type: ignore

                                with Image.open(src_file) as im:
                                    im2 = im.resize(
                                        (int(plan.preprocess_width), int(plan.preprocess_height)),
                                        resample=Image.LANCZOS,
                                    )
                                    im2.save(pre_file)
                                    saved = True
                            except Exception:
                                saved = False

                        if saved and pre_file.exists():
                            _mark_applied(str(pre_file))
                            fallback_done = True
                        else:
                            cfg["_preprocess_error_note"] = (
                                "Pre-downscale fallback failed for image input. "
                                + (
                                    f"ffmpeg error: {ffmpeg_err}"
                                    if ffmpeg_err
                                    else "Unable to resize with ffmpeg/cv2/Pillow."
                                )
                            )
                    elif in_type == "directory":
                        temp_root = Path(global_settings.get("temp_dir", temp_dir))
                        temp_root.mkdir(parents=True, exist_ok=True)
                        src_dir = Path(src_input_path)
                        img_files = [p for p in sorted(src_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
                        if img_files:
                            pre_dir = collision_safe_dir(
                                temp_root / f"{src_dir.name}_pre{int(plan.preprocess_width)}x{int(plan.preprocess_height)}"
                            )
                            pre_dir.mkdir(parents=True, exist_ok=True)

                            try:
                                import cv2  # type: ignore
                                for f in img_files:
                                    img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                                    if img is None:
                                        continue
                                    resized = cv2.resize(
                                        img,
                                        (int(plan.preprocess_width), int(plan.preprocess_height)),
                                        interpolation=cv2.INTER_AREA,
                                    )
                                    cv2.imwrite(str(pre_dir / f.name), resized)
                            except Exception:
                                try:
                                    from PIL import Image  # type: ignore
                                    for f in img_files:
                                        with Image.open(f) as im:
                                            im2 = im.resize(
                                                (int(plan.preprocess_width), int(plan.preprocess_height)),
                                                resample=Image.LANCZOS,
                                            )
                                            im2.save(pre_dir / f.name)
                                except Exception as e:
                                    cfg["_preprocess_error_note"] = (
                                        "Pre-downscale fallback failed for frame directory input: "
                                        f"{str(e)}"
                                    )
                            try:
                                if any(pre_dir.iterdir()):
                                    _mark_applied(str(pre_dir))
                                    fallback_done = True
                            except Exception:
                                pass

                    if fallback_done:
                        return

                    cfg["_preprocess_required_but_missing"] = True
                    if not cfg.get("_preprocess_error_note"):
                        cfg["_preprocess_error_note"] = (
                            "Pre-downscale was required to enforce Max Resolution, but preprocessing did not produce "
                            "a valid capped input."
                        )
                except Exception as e:
                    cfg["_preprocess_required_but_missing"] = True
                    cfg["_preprocess_error_note"] = f"Pre-downscale fallback failed with unexpected error: {str(e)}"
            
            # Initialize progress
            if progress:
                progress(0, desc="Initializing FlashVSR+...")

            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}
            image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

            def _media_updates(out_path: Optional[str]) -> tuple[Any, Any]:
                """
                Return (output_video_update, output_image_update) for the merged output panel.
                """
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

            def _cache_chunk_preview(run_dir: Optional[Path]) -> None:
                try:
                    if not run_dir:
                        seed_controls["flashvsr_chunk_preview"] = {
                            "message": "No chunk preview available.",
                            "gallery": [],
                            "videos": [],
                            "count": 0,
                        }
                    else:
                        seed_controls["flashvsr_chunk_preview"] = build_chunk_preview_payload(str(run_dir))
                    state["seed_controls"] = seed_controls
                except Exception:
                    pass
            
            # PRE-FLIGHT CHECKS (mirrors SeedVR2/GAN for consistency)
            from shared.error_handling import check_ffmpeg_available, check_disk_space
            
            # Check ffmpeg availability
            ffmpeg_ok, ffmpeg_msg = check_ffmpeg_available()
            if not ffmpeg_ok:
                vid_upd, img_upd = _media_updates(None)
                yield (
                    "❌ ffmpeg not found in PATH",
                    ffmpeg_msg or "Install ffmpeg and add to PATH before processing",
                    vid_upd,
                    img_upd,
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    state
                )
                return
            
            # Check disk space (require at least 5GB free)
            output_path_check = Path(global_settings.get("output_dir", output_dir))
            has_space, space_warning = check_disk_space(output_path_check, required_mb=5000)
            if not has_space:
                vid_upd, img_upd = _media_updates(None)
                yield (
                    "❌ Insufficient disk space",
                    space_warning or "Free up at least 5GB disk space before processing",
                    vid_upd,
                    img_upd,
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    state
                )
                return

            # -------------------------------------------------------------
            # ✅ Batch processing (folder of videos and/or frame directories)
            # -------------------------------------------------------------
            if bool(settings.get("batch_enable")):
                batch_in = normalize_path(settings.get("batch_input_path") or "")

                if not batch_in or not Path(batch_in).exists() or not Path(batch_in).is_dir():
                    vid_upd, img_upd = _media_updates(None)
                    yield (
                        "❌ Batch input folder missing/invalid",
                        "Provide a valid Batch Input Folder path.",
                        vid_upd,
                        img_upd,
                        gr.update(visible=False),
                        gr.update(value="", visible=False),
                        state,
                    )
                    return

                in_dir = Path(batch_in)
                items: List[Path] = [
                    p
                    for p in sorted(in_dir.iterdir())
                    if (p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS) or p.is_dir()
                ]
                if not items:
                    # If the batch folder itself is a frame directory, treat it as a single job.
                    items = [in_dir]

                # Universal chunking (Resolution tab) applies to FlashVSR+ batch runs too.
                auto_chunk = bool(seed_controls.get("auto_chunk", True))
                chunk_size_sec = float(seed_controls.get("chunk_size_sec", 0) or 0)
                chunk_overlap_sec = 0.0 if auto_chunk else float(seed_controls.get("chunk_overlap_sec", 0) or 0)
                per_chunk_cleanup = bool(seed_controls.get("per_chunk_cleanup", False))
                scene_threshold = float(seed_controls.get("scene_threshold", 27.0))
                min_scene_len = float(seed_controls.get("min_scene_len", 1.0))
                frame_accurate_split = bool(seed_controls.get("frame_accurate_split", True))
                overwrite_existing = bool(seed_controls.get("overwrite_existing_batch_val", False))

                logs: List[str] = []
                outputs: List[str] = []
                last_input_path: Optional[str] = None
                last_output_path: Optional[str] = None
                last_chunk_run_dir: Optional[Path] = None
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

                if progress:
                    progress(0, desc=f"Batch: {len(items)} item(s) queued")

                for idx, item in enumerate(items, 1):
                    if _flashvsr_cancel_event.is_set():
                        vid_upd, img_upd = _media_updates(None)
                        yield (
                            "⏹️ Batch cancelled",
                            "\n".join(logs[-200:]) + "\n\n[Cancelled by user]",
                            vid_upd,
                            img_upd,
                            gr.update(visible=False),
                            gr.update(value="", visible=False),
                            state,
                        )
                        return

                    item_path = str(item)
                    last_input_path = item_path

                    if progress:
                        progress((idx - 1) / max(1, len(items)), desc=f"Batch {idx}/{len(items)}: {Path(item_path).name}")

                    item_settings = settings.copy()
                    item_settings["batch_enable"] = False
                    item_settings["input_path"] = item_path
                    item_settings["_effective_input_path"] = item_path
                    item_settings["_original_filename"] = Path(item_path).name
                    item_out_dir = batch_item_dir(batch_root, Path(item_path).name)

                    mode_val = str(item_settings.get("mode", "tiny") or "tiny")
                    seed_val = int(item_settings.get("seed", 0) or 0)
                    base_no_ext = Path(item_path).stem
                    predicted_output_file = item_out_dir / f"FlashVSR_{mode_val}_{base_no_ext}_{seed_val}.mp4"

                    from shared.output_run_manager import prepare_batch_video_run_dir

                    run_paths = prepare_batch_video_run_dir(
                        batch_root,
                        Path(item_path).name,
                        input_path=str(item_path),
                        model_label="FlashVSR+",
                        mode=str(getattr(runner, "get_mode", lambda: "subprocess")() or "subprocess"),
                        overwrite_existing=overwrite_existing,
                    )
                    if not run_paths:
                        if not overwrite_existing:
                            logs.append(
                                f"⏭️ [{idx}/{len(items)}] {Path(item_path).name} skipped (output folder exists)"
                            )
                            if predicted_output_file.exists():
                                outputs.append(str(predicted_output_file))
                            continue
                        logs.append(
                            f"❌ [{idx}/{len(items)}] {Path(item_path).name} failed (could not create output folder)"
                        )
                        continue

                    item_settings["global_output_dir"] = str(run_paths.run_dir)
                    item_settings["_run_dir"] = str(run_paths.run_dir)
                    item_settings["_processed_chunks_dir"] = str(run_paths.processed_chunks_dir)
                    # Explicit output file path inside the per-item folder.
                    item_settings["output_override"] = str(predicted_output_file)

                    _apply_vnext_preprocess(item_settings, item_path)
                    _enforce_preprocess_requirements(item_settings, item_path)
                    if item_settings.get("_preprocess_required_but_missing"):
                        logs.append(
                            f"❌ [{idx}/{len(items)}] {Path(item_path).name} failed: "
                            f"{item_settings.get('_preprocess_error_note') or 'required pre-downscale failed'}"
                        )
                        continue

                    effective_for_chunk = normalize_path(item_settings.get("_effective_input_path") or item_path)
                    should_chunk_video = (detect_input_type(effective_for_chunk) == "video") and (auto_chunk or chunk_size_sec > 0)

                    chunk_count_item = 0
                    if should_chunk_video:
                        from shared.chunking import chunk_and_process
                        from shared.runner import RunResult

                        class _CancelProbe:
                            def is_canceled(self) -> bool:
                                return bool(_flashvsr_cancel_event.is_set())

                        chunk_settings = item_settings.copy()
                        chunk_settings["input_path"] = effective_for_chunk
                        chunk_settings["frame_accurate_split"] = frame_accurate_split

                        # For batch output, prefer an explicit file name to match FlashVSR naming.
                        try:
                            out_base = Path(item_out_dir)
                            base_stem = Path(chunk_settings.get("_original_filename") or item_path).stem
                            seed_val = int(chunk_settings.get("seed", 0) or 0)
                            mode_val = str(chunk_settings.get("mode", "tiny") or "tiny")
                            chunk_settings["output_override"] = str(out_base / f"FlashVSR_{mode_val}_{base_stem}_{seed_val}.mp4")
                        except Exception:
                            pass

                        def _process_chunk(s: Dict[str, Any], on_progress=None) -> RunResult:
                            r = run_flashvsr(
                                s,
                                base_dir,
                                on_progress=on_progress,
                                cancel_event=_flashvsr_cancel_event,
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
                            chunk_seconds=0.0 if auto_chunk else chunk_size_sec,
                            chunk_overlap=0.0 if auto_chunk else chunk_overlap_sec,
                            per_chunk_cleanup=per_chunk_cleanup,
                            allow_partial=True,
                            global_output_dir=str(item_out_dir),
                            resume_from_partial=False,
                            progress_tracker=None,
                            process_func=_process_chunk,
                            model_type="flashvsr",
                        )
                        result = RunResult(rc, final_output if final_output else None, clog)
                        outp = result.output_path
                    else:
                        result = run_flashvsr(
                            item_settings,
                            base_dir,
                            on_progress=None,
                            cancel_event=_flashvsr_cancel_event,
                            process_handle=None,
                        )
                        outp = result.output_path

                    if outp and Path(outp).exists():
                        # Optional face restoration (video-first)
                        if face_apply:
                            try:
                                from shared.face_restore import restore_video, restore_image
                                if Path(outp).suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                                    restored = restore_video(
                                        outp,
                                        strength=face_strength,
                                        on_progress=lambda x: logs.append(x) if x else None,
                                    )
                                    if restored and Path(restored).exists():
                                        outp = restored
                                else:
                                    restored_img = restore_image(outp, strength=face_strength)
                                    if restored_img and Path(restored_img).exists():
                                        outp = restored_img
                            except Exception:
                                pass

                        # Apply global image output format/quality preferences.
                        if Path(outp).suffix.lower() in image_exts:
                            converted = _apply_image_output_preferences(
                                outp,
                                settings.get("image_output_format", "png"),
                                settings.get("image_output_quality", 95),
                            )
                            if converted and Path(converted).exists():
                                outp = converted

                        # Save preprocessed input (if created) alongside outputs
                        pre_in = item_settings.get("_preprocessed_input_path")
                        if pre_in and outp:
                            # Preprocessed inputs (e.g., downscaled videos) are already written into the run folder
                            # (see `downscaled_<orig>.mp4`), so we don't duplicate them under `pre_processed/`.
                            saved_pre = None
                            if saved_pre:
                                logs.append(f"🧩 Preprocessed input saved: {saved_pre}")

                        if Path(outp).suffix.lower() in video_exts:
                            rife_out, rife_msg = maybe_apply_global_rife(
                                runner=runner,
                                output_video_path=outp,
                                seed_controls=seed_controls,
                                on_log=(lambda m: logs.append(m.strip()) if m else None),
                                chunking_context={
                                    "enabled": bool(chunk_count_item and chunk_count_item > 0),
                                    "auto_chunk": bool(auto_chunk),
                                    "chunk_size_sec": float(chunk_size_sec or 0),
                                    "chunk_overlap_sec": 0.0 if auto_chunk else float(chunk_overlap_sec or 0),
                                    "scene_threshold": float(scene_threshold or 27.0),
                                    "min_scene_len": float(min_scene_len or 1.0),
                                    "frame_accurate_split": bool(frame_accurate_split),
                                    "per_chunk_cleanup": bool(per_chunk_cleanup),
                                },
                            )
                            if rife_out and Path(rife_out).exists():
                                logs.append(f"✅ Global RIFE output: {Path(rife_out).name}")
                                outp = rife_out
                            elif rife_msg:
                                logs.append(f"⚠️ {rife_msg}")
                            comp_vid_path, comp_vid_err = maybe_generate_input_vs_output_comparison(
                                item_settings.get("_original_input_path_before_preprocess") or item_path,
                                outp,
                                seed_controls,
                                label_output="FlashVSR+",
                                on_progress=None,
                            )
                            if comp_vid_path:
                                logs.append(f"✅ Comparison video created: {Path(comp_vid_path).name}")
                            elif comp_vid_err:
                                logs.append(f"⚠️ Comparison video failed: {comp_vid_err}")

                        if chunk_count_item:
                            last_chunk_run_dir = Path(item_settings.get("_run_dir") or item_out_dir)

                        outputs.append(outp)
                        last_output_path = outp
                        if chunk_count_item:
                            logs.append(f"✅ [{idx}/{len(items)}] {Path(item_path).name} → {Path(outp).name} ({int(chunk_count_item)} chunks)")
                        else:
                            logs.append(f"✅ [{idx}/{len(items)}] {Path(item_path).name} → {Path(outp).name}")
                    else:
                        if getattr(result, "returncode", 0) != 0:
                            maybe_set_vram_oom_alert(state, model_label="FlashVSR+", text=getattr(result, "log", ""), settings=item_settings)
                        logs.append(f"❌ [{idx}/{len(items)}] {Path(item_path).name} failed")

                    # Log run summary per-item
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
                                    "pipeline": "flashvsr",
                                    "batch": True,
                                    **(
                                        {
                                            "chunking": {
                                                "mode": "auto" if auto_chunk else "static",
                                                "chunk_size_sec": 0.0 if auto_chunk else float(chunk_size_sec or 0),
                                                "chunk_overlap_sec": 0.0 if auto_chunk else float(chunk_overlap_sec or 0),
                                                "scene_threshold": float(scene_threshold or 27.0),
                                                "min_scene_len": float(min_scene_len or 1.0),
                                                "chunks": int(chunk_count_item or 0),
                                                "frame_accurate_split": bool(frame_accurate_split),
                                            }
                                        }
                                        if chunk_count_item
                                        else {}
                                    ),
                                },
                            )
                        except Exception:
                            pass

                if progress:
                    progress(1.0, desc=f"Batch complete ({len(outputs)}/{len(items)} succeeded)")

                _cache_chunk_preview(last_chunk_run_dir)
                seed_controls["flashvsr_batch_outputs"] = list(outputs)
                state["seed_controls"] = seed_controls

                # Track output path for pinned comparison feature (last output)
                if last_output_path:
                    try:
                        outp_path = Path(last_output_path)
                        seed_controls["last_output_dir"] = str(outp_path.parent if outp_path.is_file() else outp_path)
                        seed_controls["last_output_path"] = str(outp_path) if outp_path.is_file() else None
                        state["seed_controls"] = seed_controls
                    except Exception:
                        pass

                # Comparison for last item only
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

                status = f"✅ FlashVSR+ batch complete ({len(outputs)}/{len(items)} succeeded, {len(items) - len(outputs)} failed)"
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
            
            # Resolve input
            resume_run_dir_raw = str(settings.get("resume_run_dir") or "").strip()
            resume_mode = bool(resume_run_dir_raw and (not settings.get("batch_enable")) and (not preview_only))
            if resume_mode:
                resume_run_dir = Path(normalize_path(resume_run_dir_raw))
                if not (resume_run_dir.exists() and resume_run_dir.is_dir()):
                    vid_upd, img_upd = _media_updates(None)
                    yield (
                        "❌ Resume folder not found",
                        f"Configured resume folder does not exist: {resume_run_dir}",
                        vid_upd,
                        img_upd,
                        gr.update(visible=False),
                        gr.update(value="", visible=False),
                        state,
                    )
                    return
                recovered_input, recovered_name, _recovered_source = resolve_resume_input_from_run_dir(resume_run_dir)
                if recovered_input is None:
                    vid_upd, img_upd = _media_updates(None)
                    yield (
                        "❌ Resume input not found",
                        (
                            f"Could not recover input source from resume folder: {resume_run_dir}. "
                            "Expected run_context.json input path or run_metadata/downscaled artifact."
                        ),
                        vid_upd,
                        img_upd,
                        gr.update(visible=False),
                        gr.update(value="", visible=False),
                        state,
                    )
                    return
                input_path = normalize_path(str(recovered_input))
                settings["_original_filename"] = recovered_name or Path(input_path).name
            else:
                input_path = normalize_path(upload if upload else settings["input_path"])
            if not input_path or not Path(input_path).exists():
                vid_upd, img_upd = _media_updates(None)
                yield (
                    "❌ Input path missing",
                    "",
                    vid_upd,
                    img_upd,
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    state
                )
                return
            
            settings["input_path"] = input_path
            settings["_effective_input_path"] = input_path  # may be overridden by preprocessing
            if not settings.get("_original_filename"):
                settings["_original_filename"] = Path(input_path).name
            seed_controls["flashvsr_batch_outputs"] = []
            state["seed_controls"] = seed_controls

            if preview_only:
                preview_src, preview_note = prepare_preview_input(
                    input_path,
                    Path(global_settings.get("temp_dir", temp_dir)),
                    prefix="flashvsr",
                    as_single_frame_dir=True,
                )
                if not preview_src:
                    vid_upd, img_upd = _media_updates(None)
                    yield (
                        "Preview preparation failed",
                        preview_note or "Failed to prepare preview input.",
                        vid_upd,
                        img_upd,
                        gr.update(visible=False),
                        gr.update(value="", visible=False),
                        state,
                    )
                    return
                settings["_preview_original_input"] = input_path
                settings["input_path"] = normalize_path(preview_src)
                settings["_effective_input_path"] = settings["input_path"]
                settings["_original_filename"] = f"{Path(input_path).stem}_preview"
                input_path = settings["input_path"]
                if preview_note and progress:
                    try:
                        progress(0, desc=preview_note[:120])
                    except Exception:
                        pass

            # Per-run output folder for single video/image runs (0001/0002/...) to avoid collisions
            # and keep all artifacts user-visible in one folder.
            input_kind_single = detect_input_type(settings["input_path"])
            if input_kind_single in {"video", "image"}:
                resume_run_dir_raw = str(settings.get("resume_run_dir") or "").strip()
                if resume_run_dir_raw:
                    resume_run_dir = Path(normalize_path(resume_run_dir_raw))
                    if not (resume_run_dir.exists() and resume_run_dir.is_dir()):
                        vid_upd, img_upd = _media_updates(None)
                        yield (
                            "Resume folder not found",
                            f"Configured resume folder does not exist: {resume_run_dir}",
                            vid_upd,
                            img_upd,
                            gr.update(visible=False),
                            gr.update(value="", visible=False),
                            state,
                        )
                        return
                    run_dir = resume_run_dir
                    processed_chunks_dir = run_dir / "processed_chunks"
                    processed_chunks_dir.mkdir(parents=True, exist_ok=True)
                    seed_controls["last_run_dir"] = str(run_dir)
                    settings["_run_dir"] = str(run_dir)
                    settings["_processed_chunks_dir"] = str(processed_chunks_dir)
                    settings["_resume_run_requested"] = bool(input_kind_single == "video")
                    settings["_user_output_override_raw"] = settings.get("output_override") or ""

                    base_stem = Path(settings.get("_original_filename") or input_path).stem
                    seed_val = int(settings.get("seed", 0) or 0)
                    mode_val = str(settings.get("mode", "tiny") or "tiny")
                    default_final = run_dir / f"FlashVSR_{mode_val}_{base_stem}_{seed_val}.mp4"
                    # Resume mode ignores new output override paths and keeps output in the selected run folder.
                    settings["output_override"] = str(default_final)
                    if input_kind_single != "video":
                        settings["resume_run_dir"] = ""
                else:
                    try:
                        base_out_root = Path(global_settings.get("output_dir", output_dir))
                        run_paths, explicit_final = prepare_single_video_run(
                            output_root_fallback=base_out_root,
                            output_override_raw=settings.get("output_override"),
                            input_path=settings["input_path"],
                            original_filename=settings.get("_original_filename") or Path(settings["input_path"]).name,
                            model_label="FlashVSR+",
                            mode="subprocess",
                        )
                        run_dir = Path(run_paths.run_dir)
                        seed_controls["last_run_dir"] = str(run_dir)
                        settings["_run_dir"] = str(run_dir)
                        settings["_processed_chunks_dir"] = str(run_paths.processed_chunks_dir)
                        settings["_user_output_override_raw"] = settings.get("output_override") or ""

                        base_stem = Path(settings.get("_original_filename") or input_path).stem
                        seed_val = int(settings.get("seed", 0) or 0)
                        mode_val = str(settings.get("mode", "tiny") or "tiny")
                        default_final = run_dir / f"FlashVSR_{mode_val}_{base_stem}_{seed_val}.mp4"
                        settings["output_override"] = str(explicit_final) if explicit_final else str(default_final)
                    except Exception:
                        pass
            
            # Output root for artifacts + preprocessing
            settings["global_output_dir"] = str(Path(settings.get("_run_dir") or output_dir))
            _apply_vnext_preprocess(settings, input_path)
            _enforce_preprocess_requirements(settings, input_path)
            if settings.get("_preprocess_required_but_missing"):
                vid_upd, img_upd = _media_updates(None)
                yield (
                    "❌ Max resolution preprocess failed",
                    str(
                        settings.get("_preprocess_error_note")
                        or "Pre-downscale was required but failed, so FlashVSR run was stopped to avoid uncapped output."
                    ),
                    vid_upd,
                    img_upd,
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    state,
                )
                return

            # Pull universal PySceneDetect chunking settings from Resolution tab (global).
            auto_chunk = bool(seed_controls.get("auto_chunk", True))
            chunk_size_sec = float(seed_controls.get("chunk_size_sec", 0) or 0)
            chunk_overlap_sec = 0.0 if auto_chunk else float(seed_controls.get("chunk_overlap_sec", 0) or 0)
            per_chunk_cleanup = bool(seed_controls.get("per_chunk_cleanup", False))
            scene_threshold = float(seed_controls.get("scene_threshold", 27.0))
            min_scene_len = float(seed_controls.get("min_scene_len", 1.0))
            frame_accurate_split = bool(seed_controls.get("frame_accurate_split", True))
            settings["frame_accurate_split"] = frame_accurate_split

            # Chunk against the effective (preprocessed) input, but keep output naming from the original filename.
            effective_for_chunk = normalize_path(settings.get("_effective_input_path") or input_path)
            should_use_chunking = (detect_input_type(effective_for_chunk) == "video") and (auto_chunk or chunk_size_sec > 0)
            resume_requested = bool(settings.get("_resume_run_requested") or str(settings.get("resume_run_dir") or "").strip())
            if resume_requested and not should_use_chunking:
                vid_upd, img_upd = _media_updates(None)
                yield (
                    "Resume unavailable for current mode",
                    (
                        "Resume run folder works only with chunk/scene-based video processing. "
                        "Enable chunking and keep the same settings as the original run."
                    ),
                    vid_upd,
                    img_upd,
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    state,
                )
                return
            if resume_requested:
                from shared.chunking import check_resume_available

                resume_root = Path(settings.get("_run_dir") or Path(global_settings.get("output_dir", output_dir)))
                resume_ok, resume_msg = check_resume_available(resume_root, "mp4")
                if not resume_ok:
                    vid_upd, img_upd = _media_updates(None)
                    yield (
                        "Resume failed",
                        f"Resume requested but no resumable chunk outputs were found in {resume_root}. {resume_msg}",
                        vid_upd,
                        img_upd,
                        gr.update(visible=False),
                        gr.update(value="", visible=False),
                        state,
                    )
                    return
            
            # Run FlashVSR+ in thread with cancel support
            result_holder = {}
            progress_queue = queue.Queue()
            process_handle = {"proc": None}  # Store subprocess handle
            
            def processing_thread():
                try:
                    if should_use_chunking:
                        from shared.chunking import chunk_and_process
                        from shared.runner import RunResult

                        # Minimal runner shim for cancellation checks inside chunk_and_process.
                        class _CancelProbe:
                            def is_canceled(self) -> bool:
                                return bool(_flashvsr_cancel_event.is_set())

                        chunk_settings = settings.copy()
                        chunk_settings["input_path"] = effective_for_chunk

                        # Default output path: match FlashVSR naming unless user overrides.
                        if not (chunk_settings.get("output_override") or "").strip():
                            try:
                                out_base = Path(global_settings.get("output_dir", output_dir))
                                base_stem = Path(chunk_settings.get("_original_filename") or input_path).stem
                                seed_val = int(chunk_settings.get("seed", 0) or 0)
                                mode_val = str(chunk_settings.get("mode", "tiny") or "tiny")
                                chunk_settings["output_override"] = str(out_base / f"FlashVSR_{mode_val}_{base_stem}_{seed_val}.mp4")
                            except Exception:
                                pass
                        if resume_requested:
                            progress_queue.put(
                                "Resume run folder detected. Resuming from last processed chunk and "
                                "continuing remaining chunks (same settings required)."
                            )

                        def _process_chunk(s: Dict[str, Any], on_progress=None) -> RunResult:
                            r = run_flashvsr(
                                s,
                                base_dir,
                                on_progress=on_progress,
                                cancel_event=_flashvsr_cancel_event,
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
                                    seed_controls["flashvsr_chunk_preview"] = build_chunk_preview_payload(str(run_root))
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
                            chunk_seconds=0.0 if auto_chunk else chunk_size_sec,
                            chunk_overlap=0.0 if auto_chunk else chunk_overlap_sec,
                            per_chunk_cleanup=per_chunk_cleanup,
                            allow_partial=True,
                            global_output_dir=str(Path(settings.get("_run_dir") or Path(global_settings.get("output_dir", output_dir)))),
                            resume_from_partial=resume_requested,
                            progress_tracker=_chunk_progress_cb,
                            process_func=_process_chunk,
                            model_type="flashvsr",
                        )

                        result_holder["result"] = RunResult(rc, final_output if final_output else None, clog)
                        result_holder["chunk_count"] = int(chunk_count or 0)
                    else:
                        result = run_flashvsr(
                            settings,
                            base_dir,
                            on_progress=lambda msg: progress_queue.put(msg),
                            cancel_event=_flashvsr_cancel_event,
                            process_handle=process_handle
                        )
                        result_holder["result"] = result
                except Exception as e:
                    result_holder["error"] = str(e)
            
            thread = threading.Thread(target=processing_thread, daemon=True)
            thread.start()
            
            # Apply face restoration if enabled (per-run toggle OR global setting)
            
            # Stream progress updates
            run_started_ts = time.time()
            last_update = time.time()
            log_buffer = []
            live_progress_pct: Optional[float] = None
            live_progress_desc = ""
            live_log_idx: Optional[int] = None
            progress_tile_idx: Optional[int] = None
            progress_tile_total: Optional[int] = None
            progress_iter_idx: Optional[int] = None
            progress_iter_ts: Optional[float] = None
            progress_iter_ema_s: Optional[float] = None

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

            def _normalize_live_progress_line(msg_text: str) -> Tuple[str, bool]:
                nonlocal progress_tile_idx, progress_tile_total
                nonlocal progress_iter_idx, progress_iter_ts, progress_iter_ema_s

                text = _strip_ansi(str(msg_text or "")).strip()
                if not text:
                    return "", False

                iter_match = re.search(
                    r"(?:^|\|)\s*(?:Processing|Processed):\s*(\d+)\s*/\s*(\d+)",
                    text,
                    flags=re.IGNORECASE,
                )
                if iter_match:
                    tile_match = re.search(
                        r"Processing\s+Tiles:\s*(\d+)\s*/\s*(\d+)",
                        text,
                        flags=re.IGNORECASE,
                    )
                    if tile_match:
                        tile_idx = int(tile_match.group(1))
                        tile_total = max(1, int(tile_match.group(2)))
                        if progress_tile_idx != tile_idx:
                            progress_tile_idx = tile_idx
                            progress_tile_total = tile_total
                            progress_iter_idx = None
                            progress_iter_ts = None
                            progress_iter_ema_s = None

                    now_ts = time.time()
                    iter_idx = int(iter_match.group(1))
                    iter_total = max(1, int(iter_match.group(2)))

                    if progress_iter_idx is None or iter_idx <= progress_iter_idx:
                        progress_iter_ema_s = None
                    elif progress_iter_ts is not None:
                        delta_s = max(0.0, now_ts - progress_iter_ts)
                        if progress_iter_ema_s is None:
                            progress_iter_ema_s = delta_s
                        else:
                            progress_iter_ema_s = (progress_iter_ema_s * 0.7) + (delta_s * 0.3)

                    progress_iter_idx = iter_idx
                    progress_iter_ts = now_ts

                    iter_pct = (float(iter_idx) / float(iter_total)) * 100.0
                    parts: List[str] = []
                    if progress_tile_idx is not None and progress_tile_total:
                        parts.append(f"Processing Tiles: {int(progress_tile_idx)}/{int(progress_tile_total)}")
                    parts.append(f"Processing: {iter_idx}/{iter_total} ({iter_pct:.1f}%)")
                    if progress_iter_ema_s is not None and progress_iter_ema_s > 0:
                        parts.append(f"{progress_iter_ema_s:.2f}s/iter")
                    return " | ".join(parts), True

                tile_match = re.search(
                    r"^\s*Processing\s+Tiles:\s*(\d+)\s*/\s*(\d+)",
                    text,
                    flags=re.IGNORECASE,
                )
                if tile_match:
                    tile_idx = int(tile_match.group(1))
                    tile_total = max(1, int(tile_match.group(2)))
                    if progress_tile_idx != tile_idx:
                        progress_tile_idx = tile_idx
                        progress_tile_total = tile_total
                        progress_iter_idx = None
                        progress_iter_ts = None
                        progress_iter_ema_s = None
                    tile_pct = (float(tile_idx) / float(tile_total)) * 100.0
                    return f"Processing Tiles: {tile_idx}/{tile_total} ({tile_pct:.1f}%)", True

                text_lc = text.lower()
                if text_lc.startswith("[flashvsr] processing..."):
                    return text, True

                return text, False

            def _extract_progress(msg_text: str) -> Tuple[Optional[float], str]:
                text = _strip_ansi(str(msg_text or "")).strip()
                if not text:
                    return None, ""

                pct: Optional[float] = None
                m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
                if m:
                    try:
                        pct = max(0.0, min(100.0, float(m.group(1)))) / 100.0
                    except Exception:
                        pct = None
                else:
                    m = re.search(r"Processed:\s*(\d+)\s*/\s*(\d+)", text, flags=re.IGNORECASE)
                    if m:
                        try:
                            num = int(m.group(1))
                            den = max(1, int(m.group(2)))
                            pct = max(0.0, min(1.0, float(num) / float(den)))
                        except Exception:
                            pct = None
                    else:
                        m = re.search(r"Processing:\s*(\d+)\s*/\s*(\d+)", text, flags=re.IGNORECASE)
                        if m:
                            try:
                                num = int(m.group(1))
                                den = max(1, int(m.group(2)))
                                pct = max(0.0, min(1.0, float(num) / float(den)))
                            except Exception:
                                pct = None

                return pct, text

            if settings.get("_scale_clamp_note"):
                log_buffer.append(str(settings.get("_scale_clamp_note")))
            if settings.get("_max_edge_merge_note"):
                log_buffer.append(str(settings.get("_max_edge_merge_note")))
            if settings.get("_preprocess_plan_note"):
                log_buffer.append(str(settings.get("_preprocess_plan_note")))
            if settings.get("_preprocess_note"):
                log_buffer.append(str(settings.get("_preprocess_note")))
            if settings.get("_preprocess_error_note"):
                log_buffer.append(str(settings.get("_preprocess_error_note")))
            if settings.get("_output_codec_note"):
                log_buffer.append(str(settings.get("_output_codec_note")))
            
            while thread.is_alive() or not progress_queue.empty():
                # Check for cancellation
                if _flashvsr_cancel_event.is_set():
                    # Kill the subprocess if still running
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
                    
                    # Try to salvage partial outputs (mirrors SeedVR2/GAN behavior)
                    compiled_output = None
                    temp_base = Path(global_settings.get("temp_dir", temp_dir))
                    temp_chunks_dir = temp_base / "chunks"
                    
                    if temp_chunks_dir.exists():
                        try:
                            from shared.chunking import detect_resume_state, concat_videos
                            from shared.path_utils import collision_safe_path as _collision_safe_path_local
                            import shutil
                            
                            # Check for completed video chunks
                            partial_video, completed_chunks = detect_resume_state(temp_chunks_dir, "mp4")
                            
                            if completed_chunks and len(completed_chunks) > 0:
                                partial_target = _collision_safe_path_local(temp_chunks_dir / "cancelled_flashvsr_partial.mp4")
                                if concat_videos(completed_chunks, partial_target, encode_settings=settings):
                                    final_output = Path(output_dir) / f"cancelled_flashvsr_partial_upscaled.mp4"
                                    final_output = _collision_safe_path_local(final_output)
                                    shutil.copy2(partial_target, final_output)
                                    compiled_output = str(final_output)
                                    log_buffer.append(f"\n✅ Partial output salvaged: {final_output.name}")
                        except Exception as e:
                            log_buffer.append(f"\n⚠️ Could not salvage partials: {str(e)}")
                    
                    status_msg = "⏹️ Processing cancelled"
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
                        state
                    )
                    return
                
                try:
                    msg = progress_queue.get(timeout=0.1)
                    msg_clean, is_live_line = _normalize_live_progress_line(msg)
                    _upsert_log_entry(msg_clean, transient=is_live_line)
                    pct_val, _ = _extract_progress(msg_clean)

                    if msg_clean:
                        msg_lc = msg_clean.lower()
                        has_real_hint = (
                            pct_val is not None
                            or "processed:" in msg_lc
                            or msg_lc.startswith("processing:")
                            or msg_lc.startswith("processing tiles:")
                            or "eta:" in msg_lc
                            or "speed:" in msg_lc
                            or "s/iter" in msg_lc
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
                
                # Yield updates every 0.5s
                now = time.time()
                if now - last_update > 0.5:
                    last_update = now
                    vid_upd, img_upd = _media_updates(None)
                    elapsed_s = int(now - run_started_ts)
                    if live_progress_desc:
                        status_live = f"⚙️ {live_progress_desc} | UI elapsed {elapsed_s}s"
                    elif live_progress_pct is not None:
                        status_live = f"⚙️ FlashVSR processing... {int(live_progress_pct * 100)}% ({elapsed_s}s)"
                    else:
                        status_live = f"⚙️ FlashVSR processing... {elapsed_s}s elapsed"
                    yield (
                        status_live,
                        "\n".join(log_buffer[-50:]),
                        vid_upd,
                        img_upd,
                        gr.update(visible=False),
                        gr.update(value="Processing...", visible=False),
                        state
                    )
            
            thread.join()
            
            # Get result
            if "error" in result_holder:
                if progress:
                    progress(0, desc="Error")
                if maybe_set_vram_oom_alert(state, model_label="FlashVSR+", text=result_holder.get("error", ""), settings=settings):
                    show_vram_oom_modal(state, title="Out of VRAM (GPU) — FlashVSR+", duration=None)
                yield (
                    ("🚫 Out of VRAM (GPU) — see banner above" if state.get("alerts", {}).get("oom", {}).get("visible") else "❌ Processing failed"),
                    f"Error: {result_holder['error']}",
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    state
                )
                return
            
            result = result_holder.get("result")
            if not result:
                yield (
                    "❌ No result",
                    "Processing did not complete",
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    state
                )
                return
            
            # Update progress to 100%
            if progress:
                progress(1.0, desc="FlashVSR+ complete!")
            
            # Apply face restoration if enabled (per-run toggle OR global setting)
            output_path = result.output_path
            
            if face_apply and output_path and Path(output_path).exists():
                from shared.face_restore import restore_video, restore_image
                
                log_buffer.append(f"Applying face restoration (strength {face_strength})...")
                
                if Path(output_path).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    # Video restoration
                    restored = restore_video(
                        output_path,
                        strength=face_strength,
                        on_progress=lambda x: log_buffer.append(x) if x else None
                    )
                    if restored and Path(restored).exists():
                        output_path = restored
                        log_buffer.append(f"✅ Face restoration complete: {restored}")
                else:
                    # Image restoration
                    restored_img = restore_image(output_path, strength=face_strength)
                    if restored_img and Path(restored_img).exists():
                        output_path = restored_img
                        log_buffer.append(f"✅ Face restoration complete: {restored_img}")

            # Apply global image output format/quality preferences.
            if output_path and Path(output_path).exists() and Path(output_path).suffix.lower() in image_exts:
                converted = _apply_image_output_preferences(
                    output_path,
                    settings.get("image_output_format", "png"),
                    settings.get("image_output_quality", 95),
                )
                if converted and Path(converted).exists() and converted != output_path:
                    output_path = converted
                    log_buffer.append(
                        f"Converted image output to {Path(converted).suffix.lower()} "
                        f"(quality {int(settings.get('image_output_quality', 95) or 95)})."
                    )

            # Save preprocessed input (if we created one) alongside outputs
            pre_in = settings.get("_preprocessed_input_path")
            if pre_in and output_path:
                # Preprocessed inputs are already saved into the run folder (downscaled_<orig>.mp4).
                saved_pre = None
                if saved_pre:
                    log_buffer.append(f"🧩 Preprocessed input saved: {saved_pre}")
            
            # Preserve audio for video outputs (best-effort; chunked runs are handled by chunk_and_process).
            if output_path and Path(output_path).exists() and Path(output_path).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                try:
                    from shared.audio_utils import ensure_audio_on_video

                    audio_src = settings.get("_original_input_path_before_preprocess") or input_path
                    audio_codec = str(settings.get("audio_codec") or "copy")
                    audio_bitrate = settings.get("audio_bitrate") or None
                    _changed, _final, _err = ensure_audio_on_video(
                        Path(output_path),
                        Path(audio_src),
                        audio_codec=audio_codec,
                        audio_bitrate=str(audio_bitrate) if audio_bitrate else None,
                        on_progress=lambda x: log_buffer.append(x) if x else None,
                    )
                    if _err:
                        log_buffer.append(f"WARNING: Audio mux: {_err}")
                    if _final and str(_final) != str(output_path):
                        output_path = str(_final)
                except Exception as e:
                    log_buffer.append(f"WARNING: Audio mux failed: {str(e)}")

            chunk_count = int(result_holder.get("chunk_count") or 0)
            if chunk_count > 0:
                _cache_chunk_preview(Path(settings.get("_run_dir") or global_settings.get("output_dir", output_dir)))
            else:
                _cache_chunk_preview(None)

            # Global RIFE post-process (keep original + create *_xFPS).
            if (
                output_path
                and Path(output_path).exists()
                and Path(output_path).suffix.lower() in video_exts
            ):
                rife_out, rife_msg = maybe_apply_global_rife(
                    runner=runner,
                    output_video_path=output_path,
                    seed_controls=seed_controls,
                    on_log=(lambda m: log_buffer.append(m.strip()) if m else None),
                    chunking_context={
                        "enabled": bool(chunk_count and chunk_count > 0),
                        "auto_chunk": bool(auto_chunk),
                        "chunk_size_sec": float(chunk_size_sec or 0),
                        "chunk_overlap_sec": 0.0 if auto_chunk else float(chunk_overlap_sec or 0),
                        "scene_threshold": float(scene_threshold or 27.0),
                        "min_scene_len": float(min_scene_len or 1.0),
                        "frame_accurate_split": bool(frame_accurate_split),
                        "per_chunk_cleanup": bool(per_chunk_cleanup),
                    },
                )
                if rife_out and Path(rife_out).exists():
                    output_path = rife_out
                elif rife_msg:
                    log_buffer.append(rife_msg)
                comp_vid_path, comp_vid_err = maybe_generate_input_vs_output_comparison(
                    settings.get("_original_input_path_before_preprocess") or input_path,
                    output_path,
                    seed_controls,
                    label_output="FlashVSR+",
                    on_progress=(lambda m: log_buffer.append(m.strip()) if m else None),
                )
                if comp_vid_path:
                    log_buffer.append(f"Comparison video created: {Path(comp_vid_path).name}")
                elif comp_vid_err:
                    log_buffer.append(f"Comparison video failed: {comp_vid_err}")

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

            # Create comparison
            html_comp, img_slider = create_unified_comparison(
                input_path=input_path,
                output_path=output_path,
                mode="slider" if output_path and output_path.endswith(".mp4") else "native"
            )
            
            # Track output path for pinned comparison feature
            if output_path:
                try:
                    outp = Path(output_path)
                    seed_controls = state.get("seed_controls", {})
                    seed_controls["last_output_dir"] = str(outp.parent if outp.is_file() else outp)
                    seed_controls["last_output_path"] = str(outp) if outp.is_file() else None
                    state["seed_controls"] = seed_controls
                except Exception:
                    pass
            
            # Log run
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
                        "pipeline": "flashvsr",
                        **(
                            {
                                "chunking": {
                                    "mode": "auto" if auto_chunk else "static",
                                    "chunk_size_sec": 0.0 if auto_chunk else float(chunk_size_sec or 0),
                                    "chunk_overlap_sec": 0.0 if auto_chunk else float(chunk_overlap_sec or 0),
                                    "scene_threshold": float(scene_threshold or 27.0),
                                    "min_scene_len": float(min_scene_len or 1.0),
                                    "chunks": chunk_count,
                                    "frame_accurate_split": bool(frame_accurate_split),
                                }
                            }
                            if chunk_count > 0
                            else {}
                        ),
                    }
                )
            if chunk_count > 0:
                status = (
                    f"✅ FlashVSR+ chunked upscale complete ({chunk_count} chunks)"
                    if result.returncode == 0
                    else f"⚠️ Chunked upscale exited with code {result.returncode}"
                )
            else:
                status = "✅ FlashVSR+ upscaling complete" if result.returncode == 0 else f"⚠️ Exited with code {result.returncode}"
            if preview_only:
                if result.returncode == 0:
                    status = "Preview complete"
                elif _flashvsr_cancel_event.is_set():
                    status = "Preview canceled"
                else:
                    status = "Preview failed"

            if result.returncode != 0 and maybe_set_vram_oom_alert(state, model_label="FlashVSR+", text=result.log, settings=settings):
                status = "Out of VRAM (GPU) - see banner above"
                show_vram_oom_modal(state, title="Out of VRAM (GPU) - FlashVSR+", duration=None)

            # Keep run folder self-documented for both video and image runs.
            try:
                run_dir_raw = settings.get("_run_dir")
                if run_dir_raw:
                    effective_input = settings.get("_effective_input_path") or settings.get("input_path")
                    input_kind_for_ctx = detect_input_type(effective_input or "")
                    pre_in = settings.get("_preprocessed_input_path")
                    if input_kind_for_ctx == "image" and (not pre_in):
                        snap = ensure_image_input_artifact(
                            Path(run_dir_raw),
                            str(effective_input or ""),
                            preferred_stem=Path(settings.get("_original_filename") or "used_input").stem,
                        )
                        if snap:
                            pre_in = str(snap)
                            settings["_preprocessed_input_path"] = pre_in
                    finalize_run_context(
                        Path(run_dir_raw),
                        pipeline="flashvsr",
                        status=str(status or ""),
                        returncode=int(result.returncode),
                        output_path=str(output_path) if output_path else None,
                        original_input_path=str(
                            settings.get("_original_input_path_before_preprocess")
                            or settings.get("_preview_original_input")
                            or input_path
                        ),
                        effective_input_path=str(effective_input) if effective_input else None,
                        preprocessed_input_path=str(pre_in) if pre_in else None,
                        input_kind=input_kind_for_ctx,
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
                state
            )
            
        except Exception as e:
            if progress:
                progress(0, desc="Critical error")
            if maybe_set_vram_oom_alert(state, model_label="FlashVSR+", text=str(e), settings=locals().get("settings")):
                show_vram_oom_modal(state, title="Out of VRAM (GPU) — FlashVSR+", duration=None)
            yield (
                "❌ Critical error",
                f"Error: {str(e)}",
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(visible=False),
                gr.update(value="", visible=False),
                state or {}
            )

    def cancel_action():
        """Cancel FlashVSR+ processing"""
        _flashvsr_cancel_event.set()
        return gr.update(value="⏹️ Cancellation requested - FlashVSR+ will stop at next checkpoint"), "Cancelling..."

    def open_outputs_folder_flashvsr():
        """Open outputs folder - delegates to shared utility (no code duplication)"""
        from shared.services.global_service import open_outputs_folder
        return open_outputs_folder(str(output_dir))
    
    def clear_temp_folder_flashvsr(confirm: bool):
        """Clear temp folder - delegates to shared utility (no code duplication)"""
        from shared.services.global_service import clear_temp_folder
        return clear_temp_folder(str(temp_dir), confirm)

    return {
        "defaults": defaults,
        "order": FLASHVSR_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "run_action": run_action,
        "cancel_action": cancel_action,
        "open_outputs_folder": open_outputs_folder_flashvsr,
        "clear_temp_folder": clear_temp_folder_flashvsr,
    }

