"""
Universal Preset System - Unified preset management for ALL tabs

This module provides a centralized preset system that saves/loads ALL settings
from ALL tabs in a single preset file. No more per-tab, per-model presets.

Features:
- Single preset contains ALL 178+ settings from all 7 tabs
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


# Tab configuration: maps tab name to (ORDER, defaults_function)
TAB_CONFIGS = {
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

    # Dropdown-safe values to avoid Gradio "value not in choices" errors.
    scale_raw = str(cfg.get("scale", "4")).strip()
    cfg["scale"] = "2" if scale_raw == "2" else "4"
    cfg["version"] = str(cfg.get("version", "10") or "10")
    cfg["mode"] = str(cfg.get("mode", "tiny") or "tiny")
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


def _normalize_tab_settings(tab_name: str, data: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(data or {})
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
            "batch_enable": False,
            "batch_input_path": "",
            "batch_output_path": "",
            "model": "",
            "target_resolution": 1080,
            "downscale_first": False,
            "auto_calculate_input": True,
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
    
    # Store tab settings in shared state
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
    # NEW (vNext): unified Upscale-x sizing cache (applies to SeedVR2/GAN/FlashVSR)
    seed_controls["upscale_factor_val"] = float(res_settings.get("upscale_factor", 4.0) or 4.0)
    seed_controls["max_resolution_val"] = int(res_settings.get("max_target_resolution", 0) or 0)
    seed_controls["auto_chunk"] = auto_chunk
    seed_controls["auto_detect_scenes"] = bool(res_settings.get("auto_detect_scenes", True))
    seed_controls["frame_accurate_split"] = bool(res_settings.get("frame_accurate_split", True))
    seed_controls["chunk_size_sec"] = res_settings.get("chunk_size", 0)
    seed_controls["chunk_overlap_sec"] = 0.0 if auto_chunk else float(res_settings.get("chunk_overlap", 0.0) or 0.0)
    seed_controls["ratio_downscale"] = res_settings.get("ratio_downscale_then_upscale", True)
    seed_controls["enable_max_target"] = res_settings.get("enable_max_target", True)
    seed_controls["auto_resolution"] = res_settings.get("auto_resolution", True)
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
    seed_controls["frame_interpolation_val"] = bool(out_settings.get("frame_interpolation", False))
    seed_controls["global_rife_enabled_val"] = bool(out_settings.get("frame_interpolation", False))
    seed_controls["global_rife_multiplier_val"] = out_settings.get("global_rife_multiplier", "x2")
    seed_controls["global_rife_model_val"] = out_settings.get("global_rife_model", get_rife_default_model())
    seed_controls["global_rife_precision_val"] = out_settings.get("global_rife_precision", "fp32")
    seed_controls["global_rife_cuda_device_val"] = out_settings.get("global_rife_cuda_device", "")
    seed_controls["global_rife_process_chunks_val"] = bool(out_settings.get("global_rife_process_chunks", True))
    seed_controls["output_format_val"] = out_settings.get("output_format", "auto")
    seed_controls["comparison_mode_val"] = out_settings.get("comparison_mode", "slider")
    seed_controls["pin_reference_val"] = out_settings.get("pin_reference", False)
    seed_controls["fullscreen_val"] = out_settings.get("fullscreen_enabled", True)
    seed_controls["save_metadata_val"] = out_settings.get("save_metadata", True)
    seed_controls["telemetry_enabled_val"] = out_settings.get("telemetry_enabled", True)
    
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
        "seedvr2": seed_controls.get("seedvr2_settings", {}),
        "gan": seed_controls.get("gan_settings", {}),
        "rife": seed_controls.get("rife_settings", {}),
        "flashvsr": seed_controls.get("flashvsr_settings", {}),
        "face": seed_controls.get("face_settings", {}),
        "resolution": seed_controls.get("resolution_settings", {}),
        "output": seed_controls.get("output_settings", {}),
    }

