"""
Universal Preset System - Unified preset management for ALL tabs

This module provides a centralized preset system that saves/loads ALL settings
from ALL tabs in a single preset file. No more per-tab, per-model presets.

Features:
- Single preset contains ALL settings from all tabs (including Global Settings)
- Save/load from any tab updates ALL tabs simultaneously
- Last used preset tracked in .last_used_preset.txt
- Auto-load last preset on app startup
- Backward compatible with merge_config for missing keys
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import ORDER constants from all services
from shared.services.seedvr2_service import SEEDVR2_ORDER, seedvr2_defaults
from shared.services.gan_service import GAN_ORDER, gan_defaults
from shared.services.rife_service import RIFE_ORDER, rife_defaults
from shared.services.flashvsr_service import FLASHVSR_ORDER, flashvsr_defaults
from shared.services.face_service import FACE_ORDER, face_defaults
from shared.services.resolution_service import RESOLUTION_ORDER, resolution_defaults
from shared.services.output_service import OUTPUT_ORDER, output_defaults
from shared.models.rife_meta import get_rife_default_model
from shared.models.flashvsr_meta import flashvsr_version_to_ui
from shared.gpu_utils import auto_select_global_gpu_device, resolve_global_gpu_device
from shared.video_codec_options import (
    get_codec_choices,
    get_pixel_format_choices,
    ENCODING_PRESETS,
    AUDIO_CODECS,
)


GLOBAL_ORDER = [
    "output_dir",
    "temp_dir",
    "telemetry",
    "face_global",
    "face_strength",
    "queue_enabled",
    "global_gpu_device",
    "mode",
    "models_dir",
    "hf_home",
    "transformers_cache",
    "pinned_reference_path",
]


def global_defaults(base_dir: Path = None) -> Dict[str, Any]:
    base = Path(base_dir) if base_dir else Path.cwd()
    models_dir = str(base / "models")
    default_gpu = resolve_global_gpu_device(auto_select_global_gpu_device())
    return {
        "output_dir": str(base / "outputs"),
        "temp_dir": str(base / "temp"),
        "telemetry": True,
        "face_global": False,
        "face_strength": 0.5,
        "queue_enabled": True,
        "global_gpu_device": default_gpu,
        "mode": "subprocess",
        "models_dir": models_dir,
        "hf_home": models_dir,
        "transformers_cache": models_dir,
        "pinned_reference_path": None,
    }


# Tab configuration: maps tab name to (ORDER, defaults_function)
TAB_CONFIGS = {
    "global": {
        "order": GLOBAL_ORDER,
        "defaults_fn": global_defaults,
        "needs_model_arg": True,  # global_defaults optionally uses base_dir
    },
    "seedvr2": {
        "order": SEEDVR2_ORDER,
        "defaults_fn": seedvr2_defaults,
        "needs_model_arg": False,  # seedvr2_defaults takes optional model_name
    },
    "gan": {
        "order": GAN_ORDER, 
        "defaults_fn": gan_defaults,
        "needs_model_arg": True,  # gan_defaults needs base_dir
    },
    "rife": {
        "order": RIFE_ORDER,
        "defaults_fn": rife_defaults,
        "needs_model_arg": False,
    },
    "flashvsr": {
        "order": FLASHVSR_ORDER,
        "defaults_fn": flashvsr_defaults,
        "needs_model_arg": False,
    },
    "face": {
        "order": FACE_ORDER,
        "defaults_fn": face_defaults,
        "needs_model_arg": True,  # face_defaults needs models list
    },
    "resolution": {
        "order": RESOLUTION_ORDER,
        "defaults_fn": resolution_defaults,
        "needs_model_arg": True,  # resolution_defaults needs models list
    },
    "output": {
        "order": OUTPUT_ORDER,
        "defaults_fn": output_defaults,
        "needs_model_arg": True,  # output_defaults needs models list
    },
}


def _normalize_flashvsr_settings(data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(data or {})
    try:
        from shared.services.flashvsr_service import _enforce_flashvsr_guardrails

        cfg = _enforce_flashvsr_guardrails(cfg, flashvsr_defaults())
    except Exception:
        pass

    cfg["output_format"] = "mp4"
    scale_raw = str(cfg.get("scale", "4")).strip()
    cfg["scale"] = "2" if scale_raw == "2" else "4"
    cfg["upscale_factor"] = 2.0 if cfg["scale"] == "2" else 4.0
    cfg["version"] = flashvsr_version_to_ui(cfg.get("version", "1.0"))
    mode_raw = str(cfg.get("mode", "tiny") or "tiny").strip().lower()
    cfg["mode"] = mode_raw if mode_raw in {"tiny", "tiny-long", "full"} else "tiny"
    precision_raw = str(cfg.get("precision", cfg.get("dtype", "auto")) or "auto").strip().lower()
    cfg["precision"] = precision_raw if precision_raw in {"auto", "bf16", "fp16"} else "auto"
    cfg["dtype"] = cfg["precision"]
    att_raw = str(cfg.get("attention_mode", cfg.get("attention", "sparse_sage_attention")) or "sparse_sage_attention").strip().lower()
    if att_raw in {"sage", "sparse_sage", "sparse_sage_attention"}:
        cfg["attention_mode"] = "sparse_sage_attention"
    elif att_raw in {"block", "block_sparse", "block_sparse_attention"}:
        cfg["attention_mode"] = "block_sparse_attention"
    elif att_raw in {"flash_attn_2", "flash_attention_2"}:
        cfg["attention_mode"] = "flash_attention_2"
    else:
        cfg["attention_mode"] = "sdpa" if att_raw == "sdpa" else "sparse_sage_attention"
    cfg["attention"] = cfg["attention_mode"]
    vae_raw = str(cfg.get("vae_model", "Wan2.1") or "Wan2.1").strip()
    if vae_raw not in {"Wan2.1", "Wan2.2", "LightVAE_W2.1", "TAE_W2.2", "LightTAE_HY1.5"}:
        vae_raw = "Wan2.1"
    cfg["vae_model"] = vae_raw
    cfg["codec"] = str(cfg.get("codec", "libx264") or "libx264").strip() or "libx264"
    try:
        cfg["crf"] = max(0, min(51, int(float(cfg.get("crf", cfg.get("quality", 18)) or 18))))
    except Exception:
        cfg["crf"] = 18
    cfg["quality"] = cfg["crf"]
    cfg["save_metadata"] = bool(cfg.get("save_metadata", True))
    cfg["face_restore_after_upscale"] = bool(cfg.get("face_restore_after_upscale", False))
    return cfg


def _normalize_rife_settings(data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(data or {})
    cfg["rife_enabled"] = True

    # Ensure model dropdown always gets a non-empty value.
    model_name = str(cfg.get("model", "") or "").strip()
    if not model_name:
        try:
            model_name = get_rife_default_model()
        except Exception:
            model_name = "rife-v4.6"
    cfg["model"] = model_name

    # Normalize precision so dropdown always gets a valid choice.
    precision_raw = str(cfg.get("fp16_mode", "fp32")).strip().lower()
    if precision_raw in ("true", "1", "yes", "on"):
        precision_raw = "fp16"
    elif precision_raw in ("false", "0", "no", "off"):
        precision_raw = "fp32"
    cfg["fp16_mode"] = "fp16" if precision_raw == "fp16" else "fp32"

    # Normalize FPS multiplier to valid dropdown options.
    fps_raw = str(cfg.get("fps_multiplier", "x2")).strip().lower()
    if fps_raw.startswith("x"):
        fps_raw = fps_raw[1:]
    try:
        fps_int = int(float(fps_raw))
    except Exception:
        fps_int = 2
    fps_int = 1 if fps_int <= 1 else (2 if fps_int <= 2 else (4 if fps_int <= 4 else 8))
    cfg["fps_multiplier"] = f"x{fps_int}"
    return cfg


def _normalize_output_settings(data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(data or {})
    # Removed setting: keep universal preset schema clean when loading old presets.
    cfg.pop("temporal_padding", None)
    cfg["output_format"] = str(cfg.get("output_format", "auto") or "auto").strip().lower()
    if cfg["output_format"] not in {"auto", "mp4", "png"}:
        cfg["output_format"] = "auto"

    image_fmt = str(cfg.get("image_output_format", "png") or "png").strip().lower()
    if image_fmt not in {"png", "jpg", "webp"}:
        image_fmt = "png"
    cfg["image_output_format"] = image_fmt
    try:
        image_quality = int(float(cfg.get("image_output_quality", 95) or 95))
    except Exception:
        image_quality = 95
    cfg["image_output_quality"] = max(1, min(100, image_quality))

    backend = str(cfg.get("seedvr2_video_backend", "opencv") or "opencv").strip().lower()
    if backend not in {"opencv", "ffmpeg"}:
        backend = "opencv"
    cfg["seedvr2_video_backend"] = backend
    cfg["seedvr2_use_10bit"] = bool(cfg.get("seedvr2_use_10bit", False)) and backend == "ffmpeg"

    codec = str(cfg.get("video_codec", "h264") or "h264").strip().lower()
    if codec not in set(get_codec_choices()):
        codec = "h264"
    cfg["video_codec"] = codec

    try:
        video_quality = int(float(cfg.get("video_quality", 18) or 18))
    except Exception:
        video_quality = 18
    cfg["video_quality"] = max(0, min(51, video_quality))

    preset = str(cfg.get("video_preset", "medium") or "medium").strip().lower()
    cfg["video_preset"] = preset if preset in ENCODING_PRESETS else "medium"
    cfg["two_pass_encoding"] = bool(cfg.get("two_pass_encoding", False))

    pix_fmt_choices = get_pixel_format_choices(codec)
    pix_fmt_fallback = pix_fmt_choices[0] if pix_fmt_choices else "yuv420p"
    pix_fmt = str(cfg.get("pixel_format", pix_fmt_fallback) or pix_fmt_fallback).strip().lower()
    if pix_fmt not in pix_fmt_choices:
        pix_fmt = pix_fmt_fallback
    cfg["pixel_format"] = pix_fmt

    audio_codec = str(cfg.get("audio_codec", "copy") or "copy").strip().lower()
    if audio_codec not in set(AUDIO_CODECS.keys()):
        audio_codec = "copy"
    cfg["audio_codec"] = audio_codec
    cfg["audio_bitrate"] = str(cfg.get("audio_bitrate", "") or "").strip()

    cfg["overwrite_existing_batch"] = bool(cfg.get("overwrite_existing_batch", False))
    cfg["frame_interpolation"] = bool(cfg.get("frame_interpolation", False))
    mult_raw = str(cfg.get("global_rife_multiplier", "x2") or "x2").strip().lower()
    if mult_raw.startswith("x"):
        mult_raw = mult_raw[1:]
    try:
        mult_val = int(float(mult_raw))
    except Exception:
        mult_val = 2
    mult_val = 2 if mult_val <= 2 else (4 if mult_val <= 4 else 8)
    cfg["global_rife_multiplier"] = f"x{mult_val}"
    model_name = str(cfg.get("global_rife_model", "") or "").strip()
    if not model_name:
        try:
            model_name = get_rife_default_model()
        except Exception:
            model_name = "rife-v4.6"
    cfg["global_rife_model"] = model_name
    precision = str(cfg.get("global_rife_precision", "fp32") or "fp32").lower()
    cfg["global_rife_precision"] = "fp16" if precision == "fp16" else "fp32"
    cfg["global_rife_cuda_device"] = str(cfg.get("global_rife_cuda_device", "") or "")
    cfg["global_rife_process_chunks"] = bool(cfg.get("global_rife_process_chunks", True))
    return cfg


def _normalize_global_settings(data: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(defaults or {})
    cfg.update(dict(data or {}))

    cfg["output_dir"] = str(cfg.get("output_dir", defaults.get("output_dir", "")) or "").strip()
    cfg["temp_dir"] = str(cfg.get("temp_dir", defaults.get("temp_dir", "")) or "").strip()
    cfg["telemetry"] = bool(cfg.get("telemetry", defaults.get("telemetry", True)))
    cfg["face_global"] = bool(cfg.get("face_global", defaults.get("face_global", False)))
    try:
        face_strength = float(cfg.get("face_strength", defaults.get("face_strength", 0.5)) or 0.5)
    except Exception:
        face_strength = 0.5
    cfg["face_strength"] = max(0.0, min(1.0, face_strength))
    cfg["queue_enabled"] = bool(cfg.get("queue_enabled", defaults.get("queue_enabled", True)))
    cfg["global_gpu_device"] = resolve_global_gpu_device(
        cfg.get("global_gpu_device", defaults.get("global_gpu_device"))
    )
    mode_raw = str(cfg.get("mode", defaults.get("mode", "subprocess")) or "subprocess").strip().lower()
    cfg["mode"] = mode_raw if mode_raw in {"subprocess", "in_app"} else "subprocess"
    cfg["models_dir"] = str(cfg.get("models_dir", defaults.get("models_dir", "")) or "").strip()
    cfg["hf_home"] = str(cfg.get("hf_home", defaults.get("hf_home", "")) or "").strip()
    cfg["transformers_cache"] = str(cfg.get("transformers_cache", defaults.get("transformers_cache", "")) or "").strip()
    pinned = cfg.get("pinned_reference_path")
    cfg["pinned_reference_path"] = str(pinned).strip() if pinned else ""
    return cfg


def _normalize_tab_settings(tab_name: str, data: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(data or {})
    if tab_name == "global":
        return _normalize_global_settings(cfg, defaults or global_defaults())
    if tab_name == "seedvr2":
        try:
            from shared.services.seedvr2_service import _enforce_seedvr2_guardrails

            cfg = _enforce_seedvr2_guardrails(cfg, defaults or seedvr2_defaults(), state=None, silent_migration=True)
        except Exception:
            pass
        return cfg
    if tab_name == "flashvsr":
        return _normalize_flashvsr_settings(cfg)
    if tab_name == "rife":
        return _normalize_rife_settings(cfg)
    if tab_name == "output":
        return _normalize_output_settings(cfg)
    return cfg


def get_all_defaults(base_dir: Path = None, models_list: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get default values for ALL tabs in universal preset format.
    
    Args:
        base_dir: Base directory for the application (needed for GAN defaults)
        models_list: List of available models (needed for some defaults)
    
    Returns:
        Dict with structure: {"seedvr2": {...}, "gan": {...}, ...}
    """
    if models_list is None:
        models_list = ["default"]
    
    defaults = {}

    # Global Settings
    defaults["global"] = global_defaults(base_dir)
    
    # SeedVR2
    defaults["seedvr2"] = seedvr2_defaults()
    
    # GAN - needs base_dir  
    if base_dir:
        defaults["gan"] = gan_defaults(base_dir)
    else:
        # Fallback without base_dir - get minimal defaults
        defaults["gan"] = {k: "" if k.endswith("_path") else None for k in GAN_ORDER}
        defaults["gan"].update({
            "input_path": "",
            "output_override": "",
            "batch_enable": False,
            "batch_input_path": "",
            "batch_output_path": "",
            "model": "",
            "use_resolution_tab": True,
            "tile_size": 0,
            "overlap": 32,
            "denoising_strength": 0.0,
            "sharpening": 0.0,
            "color_correction": True,
            "gpu_acceleration": True,
            "gpu_device": "0",
            "batch_size": 1,
            "output_format": "auto",
            "output_quality": 95,
            "save_metadata": True,
            "fps_override": 0,
            "face_restore_after_upscale": False,
            "create_subfolders": False,
            # vNext sizing
            "upscale_factor": 4.0,
            "max_resolution": 0,
            "pre_downscale_then_upscale": True,
        })
    
    # RIFE
    defaults["rife"] = rife_defaults()
    
    # FlashVSR
    defaults["flashvsr"] = flashvsr_defaults()
    
    # Face
    defaults["face"] = face_defaults(models_list)
    
    # Resolution
    defaults["resolution"] = resolution_defaults(models_list)
    
    # Output
    defaults["output"] = output_defaults(models_list)
    
    return defaults


def values_to_dict(tab_name: str, values: List[Any]) -> Dict[str, Any]:
    """
    Convert a list of values to a dict using the tab's ORDER.
    
    Args:
        tab_name: Name of the tab (seedvr2, gan, rife, etc.)
        values: List of values in ORDER sequence
    
    Returns:
        Dict mapping key names to values
    """
    config = TAB_CONFIGS.get(tab_name)
    if not config:
        raise ValueError(f"Unknown tab: {tab_name}")
    
    order = config["order"]
    if len(values) != len(order):
        raise ValueError(f"{tab_name}: Expected {len(order)} values, got {len(values)}")
    
    return dict(zip(order, values))


def dict_to_values(tab_name: str, data: Dict[str, Any], defaults: Dict[str, Any] = None) -> List[Any]:
    """
    Convert a dict to a list of values using the tab's ORDER.
    Missing keys use defaults.
    
    Args:
        tab_name: Name of the tab
        data: Dict of settings
        defaults: Default values for missing keys
    
    Returns:
        List of values in ORDER sequence
    """
    config = TAB_CONFIGS.get(tab_name)
    if not config:
        raise ValueError(f"Unknown tab: {tab_name}")
    
    order = config["order"]
    defaults = defaults or {}
    normalized = _normalize_tab_settings(tab_name, data, defaults)
    return [normalized.get(key, defaults.get(key, None)) for key in order]


def create_universal_preset(
    global_values: List[Any] = None,
    seedvr2_values: List[Any] = None,
    gan_values: List[Any] = None,
    rife_values: List[Any] = None,
    flashvsr_values: List[Any] = None,
    face_values: List[Any] = None,
    resolution_values: List[Any] = None,
    output_values: List[Any] = None,
    base_dir: Path = None,
    models_list: List[str] = None,
) -> Dict[str, Any]:
    """
    Create a universal preset from tab values.
    
    Values not provided will use defaults.
    
    Returns:
        Universal preset dict with all tabs + metadata
    """
    defaults = get_all_defaults(base_dir, models_list)
    
    preset = {
        "_meta": {
            "version": "2.0",
            "format": "universal",
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
        }
    }
    
    # Convert values to dicts, using defaults for missing tabs
    if global_values is not None:
        preset["global"] = values_to_dict("global", global_values)
    else:
        preset["global"] = defaults["global"]

    if seedvr2_values is not None:
        preset["seedvr2"] = values_to_dict("seedvr2", seedvr2_values)
    else:
        preset["seedvr2"] = defaults["seedvr2"]
    
    if gan_values is not None:
        preset["gan"] = values_to_dict("gan", gan_values)
    else:
        preset["gan"] = defaults["gan"]
    
    if rife_values is not None:
        preset["rife"] = values_to_dict("rife", rife_values)
    else:
        preset["rife"] = defaults["rife"]
    
    if flashvsr_values is not None:
        preset["flashvsr"] = values_to_dict("flashvsr", flashvsr_values)
    else:
        preset["flashvsr"] = defaults["flashvsr"]
    
    if face_values is not None:
        preset["face"] = values_to_dict("face", face_values)
    else:
        preset["face"] = defaults["face"]
    
    if resolution_values is not None:
        preset["resolution"] = values_to_dict("resolution", resolution_values)
    else:
        preset["resolution"] = defaults["resolution"]
    
    if output_values is not None:
        preset["output"] = values_to_dict("output", output_values)
    else:
        preset["output"] = defaults["output"]
    
    return preset


def extract_tab_values(
    preset: Dict[str, Any], 
    tab_name: str, 
    defaults: Dict[str, Any] = None
) -> List[Any]:
    """
    Extract values for a specific tab from a universal preset.
    
    Args:
        preset: Universal preset dict
        tab_name: Tab to extract (seedvr2, gan, etc.)
        defaults: Default values for missing keys
    
    Returns:
        List of values in ORDER sequence for the tab
    """
    tab_data = preset.get(tab_name, {})
    return dict_to_values(tab_name, tab_data, defaults)


def merge_preset_with_defaults(
    preset: Dict[str, Any],
    base_dir: Path = None,
    models_list: List[str] = None,
) -> Dict[str, Any]:
    """
    Merge a loaded preset with defaults to fill in any missing keys.
    
    This ensures backward compatibility when loading old presets
    that don't have all the new settings.
    """
    defaults = get_all_defaults(base_dir, models_list)
    merged = {"_meta": preset.get("_meta", {})}
    
    for tab_name in TAB_CONFIGS:
        if tab_name == "global" and tab_name not in preset:
            # Preserve current runtime/global state when loading legacy presets
            # that predate global fields in universal schema.
            merged[tab_name] = {}
            continue
        tab_defaults = defaults.get(tab_name, {})
        tab_preset = preset.get(tab_name, {})
        
        # Merge: start with defaults, overlay preset values
        merged_tab = tab_defaults.copy()
        for key, value in tab_preset.items():
            if value is not None:
                merged_tab[key] = value
        merged_tab = _normalize_tab_settings(tab_name, merged_tab, tab_defaults)
        
        merged[tab_name] = merged_tab
    
    return merged


# Shared state keys for syncing between tabs
SHARED_STATE_KEYS = {
    "current_preset_name": None,  # Currently loaded preset name
    "preset_dirty": False,  # True if settings changed since last save/load
}


def update_shared_state_from_preset(
    state: Dict[str, Any],
    preset: Dict[str, Any],
    preset_name: str = None,
) -> Dict[str, Any]:
    """
    Update shared_state with values from a universal preset.
    
    This populates the seed_controls cache for all tabs.
    """
    seed_controls = state.get("seed_controls", {})
    
    current_global_settings = seed_controls.get("global_settings", {})
    current_global_settings = (
        dict(current_global_settings)
        if isinstance(current_global_settings, dict)
        else {}
    )
    global_settings = _normalize_global_settings(
        preset.get("global", {}),
        current_global_settings or global_defaults(),
    )

    # Store tab settings in shared state
    seed_controls["global_settings"] = global_settings
    seed_controls["seedvr2_settings"] = preset.get("seedvr2", {})
    seed_controls["gan_settings"] = preset.get("gan", {})
    seed_controls["rife_settings"] = preset.get("rife", {})
    seed_controls["flashvsr_settings"] = preset.get("flashvsr", {})
    seed_controls["face_settings"] = preset.get("face", {})
    seed_controls["resolution_settings"] = preset.get("resolution", {})
    seed_controls["output_settings"] = preset.get("output", {})
    
    # Track current preset
    seed_controls["current_preset_name"] = preset_name
    seed_controls["preset_dirty"] = False
    
    # Also update individual cached values that other parts of the app use
    res_settings = preset.get("resolution", {})
    # Enforce: overlap is not meaningful for scene cuts (auto chunking).
    # Keep resolution_settings consistent so preset save/load stays stable.
    res_settings = dict(res_settings) if isinstance(res_settings, dict) else {}
    auto_chunk = bool(res_settings.get("auto_chunk", True))
    res_settings.setdefault("auto_detect_scenes", True)
    res_settings.setdefault("frame_accurate_split", True)
    if auto_chunk:
        res_settings["chunk_overlap"] = 0.0
    seed_controls["resolution_settings"] = res_settings
    seed_controls["auto_chunk"] = auto_chunk
    seed_controls["auto_detect_scenes"] = bool(res_settings.get("auto_detect_scenes", True))
    seed_controls["frame_accurate_split"] = bool(res_settings.get("frame_accurate_split", True))
    seed_controls["chunk_size_sec"] = res_settings.get("chunk_size", 0)
    seed_controls["chunk_overlap_sec"] = 0.0 if auto_chunk else float(res_settings.get("chunk_overlap", 0.0) or 0.0)
    seed_controls["per_chunk_cleanup"] = res_settings.get("per_chunk_cleanup", False)
    seed_controls["scene_threshold"] = res_settings.get("scene_threshold", 27.0)
    seed_controls["min_scene_len"] = res_settings.get("min_scene_len", 1.0)
    
    out_settings = _normalize_output_settings(preset.get("output", {}))
    seed_controls["output_settings"] = out_settings
    seed_controls["png_padding_val"] = out_settings.get("png_padding", 6)
    seed_controls["png_keep_basename_val"] = out_settings.get("png_keep_basename", True)
    seed_controls["png_sequence_enabled_val"] = bool(out_settings.get("png_sequence_enabled", False))
    seed_controls["overwrite_existing_batch_val"] = bool(out_settings.get("overwrite_existing_batch", False))
    seed_controls["skip_first_frames_val"] = out_settings.get("skip_first_frames", 0)
    seed_controls["load_cap_val"] = out_settings.get("load_cap", 0)
    seed_controls["fps_override_val"] = out_settings.get("fps_override", 0)
    seed_controls["image_output_format_val"] = out_settings.get("image_output_format", "png")
    seed_controls["image_output_quality_val"] = out_settings.get("image_output_quality", 95)
    seed_controls["seedvr2_video_backend_val"] = out_settings.get("seedvr2_video_backend", "opencv")
    seed_controls["seedvr2_use_10bit_val"] = bool(out_settings.get("seedvr2_use_10bit", False))
    seed_controls["video_codec_val"] = str(out_settings.get("video_codec", "h264") or "h264")
    seed_controls["video_quality_val"] = int(out_settings.get("video_quality", 18) or 18)
    seed_controls["video_preset_val"] = str(out_settings.get("video_preset", "medium") or "medium")
    seed_controls["two_pass_encoding_val"] = bool(out_settings.get("two_pass_encoding", False))
    seed_controls["pixel_format_val"] = str(out_settings.get("pixel_format", "yuv420p") or "yuv420p")
    seed_controls["frame_interpolation_val"] = bool(out_settings.get("frame_interpolation", False))
    seed_controls["global_rife_enabled_val"] = bool(out_settings.get("frame_interpolation", False))
    seed_controls["global_rife_multiplier_val"] = out_settings.get("global_rife_multiplier", "x2")
    seed_controls["global_rife_model_val"] = out_settings.get("global_rife_model", get_rife_default_model())
    seed_controls["global_rife_precision_val"] = out_settings.get("global_rife_precision", "fp32")
    global_gpu_device = resolve_global_gpu_device(global_settings.get("global_gpu_device"))
    out_settings["global_rife_cuda_device"] = "" if global_gpu_device == "cpu" else global_gpu_device
    seed_controls["global_gpu_device_val"] = global_gpu_device
    seed_controls["global_rife_cuda_device_val"] = out_settings["global_rife_cuda_device"]
    seed_controls["global_rife_process_chunks_val"] = bool(out_settings.get("global_rife_process_chunks", True))
    seed_controls["output_format_val"] = out_settings.get("output_format", "auto")
    seed_controls["comparison_mode_val"] = out_settings.get("comparison_mode", "slider")
    seed_controls["pin_reference_val"] = out_settings.get("pin_reference", False)
    seed_controls["fullscreen_val"] = out_settings.get("fullscreen_enabled", True)
    seed_controls["save_metadata_val"] = out_settings.get("save_metadata", True)
    seed_controls["telemetry_enabled_val"] = out_settings.get("telemetry_enabled", True)
    seed_controls["audio_codec_val"] = str(out_settings.get("audio_codec", "copy") or "copy")
    seed_controls["audio_bitrate_val"] = str(out_settings.get("audio_bitrate", "") or "")
    seed_controls["generate_comparison_video_val"] = bool(out_settings.get("generate_comparison_video", True))
    seed_controls["comparison_video_layout_val"] = str(out_settings.get("comparison_video_layout", "auto") or "auto")
    seed_controls["face_strength_val"] = float(global_settings.get("face_strength", 0.5))
    seed_controls["queue_enabled_val"] = bool(global_settings.get("queue_enabled", True))
    seed_controls["pinned_reference_path"] = global_settings.get("pinned_reference_path")

    state["seed_controls"] = seed_controls
    return state


def collect_preset_from_shared_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collect current settings from shared_state into a universal preset.
    """
    seed_controls = state.get("seed_controls", {})
    
    return {
        "_meta": {
            "version": "2.0",
            "format": "universal",
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
        },
        "global": seed_controls.get("global_settings", {}),
        "seedvr2": seed_controls.get("seedvr2_settings", {}),
        "gan": seed_controls.get("gan_settings", {}),
        "rife": seed_controls.get("rife_settings", {}),
        "flashvsr": seed_controls.get("flashvsr_settings", {}),
        "face": seed_controls.get("face_settings", {}),
        "resolution": seed_controls.get("resolution_settings", {}),
        "output": seed_controls.get("output_settings", {}),
    }

