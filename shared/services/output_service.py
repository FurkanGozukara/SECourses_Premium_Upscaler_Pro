from typing import Any, Dict, List, Optional
import gradio as gr

from shared.preset_manager import PresetManager
from shared.video_codec_options import (
    get_codec_info,
    get_pixel_format_info,
    get_recommended_settings
)
from shared.models.rife_meta import get_rife_default_model


def output_defaults(models: List[str]) -> Dict[str, Any]:
    return {
        "model": models[0] if models else "",
        "output_format": "auto",
        "overwrite_existing_batch": False,
        "png_sequence_enabled": False,
        "png_padding": 6,  # Match SeedVR2 CLI default (6-digit padding)
        "png_keep_basename": True,
        "fps_override": 0,
        "video_codec": "h264",
        "video_quality": 18,
        "video_preset": "medium",
        "two_pass_encoding": False,
        "skip_first_frames": 0,
        "pixel_format": "yuv420p",
        "audio_codec": "copy",
        "audio_bitrate": "",
        "load_cap": 0,
        "temporal_padding": 0,
        # Repurposed legacy key as global cross-model RIFE switch.
        "frame_interpolation": False,
        "global_rife_multiplier": "x2",
        "global_rife_model": get_rife_default_model(),
        "global_rife_precision": "fp32",
        "global_rife_cuda_device": "",
        "global_rife_process_chunks": True,
        "comparison_mode": "slider",
        "pin_reference": False,
        "fullscreen_enabled": True,
        "comparison_zoom": 100,
        "show_difference": False,
        "generate_comparison_video": True,  # Generate input vs output comparison video
        "comparison_video_layout": "auto",  # auto, horizontal, or vertical
        "save_metadata": True,
        "metadata_format": "json",
        "telemetry_enabled": True,
        "log_level": "info",
    }


OUTPUT_ORDER: List[str] = [
    "output_format",
    "png_sequence_enabled",
    "png_padding",
    "png_keep_basename",
    "fps_override",
    "video_codec",
    "video_quality",
    "video_preset",
    "two_pass_encoding",
    "skip_first_frames",
    "load_cap",
    "pixel_format",
    "audio_codec",
    "audio_bitrate",
    "temporal_padding",
    "frame_interpolation",
    "global_rife_multiplier",
    "global_rife_model",
    "global_rife_precision",
    "global_rife_cuda_device",
    "comparison_mode",
    "pin_reference",
    "fullscreen_enabled",
    "comparison_zoom",
    "show_difference",
    "generate_comparison_video",
    "comparison_video_layout",
    "save_metadata",
    "metadata_format",
    "telemetry_enabled",
    "log_level",
    "overwrite_existing_batch",
    "global_rife_process_chunks",
]


def _output_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    return dict(zip(OUTPUT_ORDER, args))


def _normalize_output_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(data or {})
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

    cfg["global_rife_model"] = str(cfg.get("global_rife_model", "") or "").strip() or get_rife_default_model()
    precision_raw = str(cfg.get("global_rife_precision", "fp32") or "fp32").strip().lower()
    cfg["global_rife_precision"] = "fp16" if precision_raw == "fp16" else "fp32"
    cfg["global_rife_cuda_device"] = str(cfg.get("global_rife_cuda_device", "") or "")
    cfg["global_rife_process_chunks"] = bool(cfg.get("global_rife_process_chunks", True))
    return cfg


def _apply_output_preset(
    preset: Dict[str, Any],
    defaults: Dict[str, Any],
    preset_manager: PresetManager,
    current: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    base = defaults.copy()
    if current:
        base.update(current)
    merged = _normalize_output_fields(preset_manager.merge_config(base, preset))
    return [merged[k] for k in OUTPUT_ORDER]


def build_output_callbacks(
    preset_manager: PresetManager,
    shared_state: gr.State,
    models: List[str],
    global_settings: Optional[Dict[str, Any]] = None,
):
    defaults = output_defaults(models)
    
    # Load persisted pinned reference from global settings if available
    if global_settings and "pinned_reference_path" in global_settings:
        if shared_state and shared_state.value:
            seed_controls = shared_state.value.get("seed_controls", {})
            seed_controls["pinned_reference_path"] = global_settings["pinned_reference_path"]

    def refresh_presets(model_name: str, select_name: Optional[str] = None):
        presets = preset_manager.list_presets("output", model_name)
        last_used = preset_manager.get_last_used_name("output", model_name)
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        return gr.update(choices=presets, value=value)

    def save_preset(preset_name: str, *args):
        if not preset_name.strip():
            return gr.update(), gr.update(value="⚠️ Enter a preset name before saving"), *list(args)

        try:
            payload = _output_dict_from_args(list(args))
            model_name = defaults.get("model", "global")
            preset_manager.save_preset_safe("output", model_name, preset_name.strip(), payload)
            dropdown = refresh_presets(model_name, select_name=preset_name.strip())

            current_map = dict(zip(OUTPUT_ORDER, list(args)))
            loaded_vals = _apply_output_preset(payload, defaults, preset_manager, current=current_map)

            return dropdown, gr.update(value=f"✅ Saved preset '{preset_name}' for {model_name}"), *loaded_vals
        except Exception as e:
            return gr.update(), gr.update(value=f"❌ Error saving preset: {str(e)}"), *list(args)

    def load_preset(preset_name: str, model_name: str, current_values: List[Any]):
        try:
            model_name = model_name or defaults["model"]
            preset = preset_manager.load_preset_safe("output", model_name, preset_name)
            if preset:
                preset_manager.set_last_used("output", model_name, preset_name)

            defaults_with_model = defaults.copy()
            defaults_with_model["model"] = model_name
            current_map = dict(zip(OUTPUT_ORDER, current_values))
            values = _apply_output_preset(preset or {}, defaults_with_model, preset_manager, current=current_map)
            status = gr.update(value=f"✅ Loaded preset '{preset_name}' for {model_name}")
            return values + [status]
        except Exception as e:
            print(f"Error loading preset {preset_name}: {e}")
            error_status = gr.update(value=f"❌ Error loading preset: {str(e)}")
            return current_values + [error_status]

    def safe_defaults():
        return [defaults[k] for k in OUTPUT_ORDER]
    
    def apply_codec_preset(preset_name: str, current_values: List[Any]):
        """Apply a codec preset (youtube, archival, editing, web)"""
        try:
            recommended = get_recommended_settings(preset_name)
            current_dict = _output_dict_from_args(current_values)
            
            # Update only codec-related fields
            current_dict["video_codec"] = recommended["codec"]
            current_dict["video_quality"] = recommended["quality"]
            current_dict["pixel_format"] = recommended["pixel_format"]
            current_dict["video_preset"] = recommended["preset"]
            current_dict["audio_codec"] = recommended["audio_codec"]
            current_dict["audio_bitrate"] = recommended["audio_bitrate"] or ""
            
            return [current_dict[k] for k in OUTPUT_ORDER]
        except Exception:
            return current_values
    
    def update_codec_info(codec_key: str):
        """Update codec information display"""
        return gr.update(value=get_codec_info(codec_key))
    
    def update_pixel_format_info(pix_fmt: str):
        """Update pixel format information display"""
        return gr.update(value=get_pixel_format_info(pix_fmt))

    # Cache helpers used by tab_output UI
    def cache_output(fmt, state):
        state["seed_controls"]["output_format_val"] = fmt
        return gr.update(value="Output format cached for runs."), state

    def cache_fps(fps_val, state):
        state["seed_controls"]["fps_override_val"] = fps_val
        return gr.update(value="FPS override cached for runs."), state

    def cache_comparison(mode, state):
        state["seed_controls"]["comparison_mode_val"] = mode
        return gr.update(value="Comparison mode cached for runs."), state

    def cache_pin(val, state):
        state["seed_controls"]["pin_reference_val"] = bool(val)
        return gr.update(value="Pin reference preference cached."), state

    def cache_fullscreen(val, state):
        state["seed_controls"]["fullscreen_val"] = bool(val)
        return gr.update(value="Fullscreen preference cached."), state

    def cache_png_padding(val, state):
        try:
            state["seed_controls"]["png_padding_val"] = max(1, int(val))
        except Exception:
            state["seed_controls"]["png_padding_val"] = defaults["png_padding"]
        return gr.update(value="PNG padding cached for runs."), state

    def cache_png_basename(val, state):
        state["seed_controls"]["png_keep_basename_val"] = bool(val)
        return gr.update(value="PNG base-name preference cached."), state

    def cache_skip(val, state):
        try:
            state["seed_controls"]["skip_first_frames_val"] = max(0, int(val))
        except Exception:
            state["seed_controls"]["skip_first_frames_val"] = 0
        return gr.update(value="Skip-first-frames cached for runs."), state

    def cache_cap(val, state):
        try:
            state["seed_controls"]["load_cap_val"] = max(0, int(val))
        except Exception:
            state["seed_controls"]["load_cap_val"] = 0
        return gr.update(value="Load-cap cached for runs."), state

    def cache_overwrite_batch(val, state):
        """
        Cache a global preference for batch mode overwrite/skip behavior.

        - False (default): skip if output already exists.
        - True: overwrite existing outputs.
        """
        state["seed_controls"]["overwrite_existing_batch_val"] = bool(val)
        return gr.update(value="Batch overwrite preference cached."), state

    def cache_generate_comparison_video(val, state):
        """Cache preference for generating input vs output comparison video."""
        state["seed_controls"]["generate_comparison_video_val"] = bool(val)
        return gr.update(value="Comparison video generation preference cached."), state

    def cache_comparison_video_layout(val, state):
        """Cache preference for comparison video layout (auto, horizontal, vertical)."""
        state["seed_controls"]["comparison_video_layout_val"] = val
        return gr.update(value=f"Comparison video layout set to: {val}"), state

    def apply_to_pipeline(*args):
        """Apply all output settings to pipeline at once"""
        state = args[-1]
        values = list(args[:-1])
        settings_dict = _normalize_output_fields(_output_dict_from_args(values))

        seed_controls = state.get("seed_controls", {})
        seed_controls["output_format_val"] = settings_dict.get("output_format", "auto")
        seed_controls["png_sequence_enabled_val"] = bool(settings_dict.get("png_sequence_enabled", False))
        seed_controls["overwrite_existing_batch_val"] = bool(settings_dict.get("overwrite_existing_batch", False))
        seed_controls["fps_override_val"] = settings_dict.get("fps_override", 0)
        seed_controls["frame_interpolation_val"] = bool(settings_dict.get("frame_interpolation", False))
        seed_controls["global_rife_enabled_val"] = bool(settings_dict.get("frame_interpolation", False))
        seed_controls["global_rife_multiplier_val"] = settings_dict.get("global_rife_multiplier", "x2")
        seed_controls["global_rife_model_val"] = settings_dict.get("global_rife_model", get_rife_default_model())
        seed_controls["global_rife_precision_val"] = settings_dict.get("global_rife_precision", "fp32")
        seed_controls["global_rife_cuda_device_val"] = settings_dict.get("global_rife_cuda_device", "")
        seed_controls["global_rife_process_chunks_val"] = bool(settings_dict.get("global_rife_process_chunks", True))
        seed_controls["comparison_mode_val"] = settings_dict.get("comparison_mode", "slider")
        seed_controls["pin_reference_val"] = settings_dict.get("pin_reference", False)
        seed_controls["fullscreen_val"] = settings_dict.get("fullscreen_enabled", True)
        seed_controls["png_padding_val"] = settings_dict.get("png_padding", 6)
        seed_controls["png_keep_basename_val"] = settings_dict.get("png_keep_basename", True)
        seed_controls["skip_first_frames_val"] = settings_dict.get("skip_first_frames", 0)
        seed_controls["load_cap_val"] = settings_dict.get("load_cap", 0)
        # NEW: Audio preferences (used for muxing audio into model outputs)
        seed_controls["audio_codec_val"] = settings_dict.get("audio_codec", "copy")
        seed_controls["audio_bitrate_val"] = settings_dict.get("audio_bitrate", "")
        # Wire metadata and telemetry settings
        seed_controls["save_metadata_val"] = settings_dict.get("save_metadata", True)
        seed_controls["telemetry_enabled_val"] = settings_dict.get("telemetry_enabled", True)
        # Wire comparison video generation settings
        seed_controls["generate_comparison_video_val"] = settings_dict.get("generate_comparison_video", True)
        seed_controls["comparison_video_layout_val"] = settings_dict.get("comparison_video_layout", "auto")
        # Persist full Output tab state so tab switches keep values stable.
        seed_controls["output_settings"] = dict(settings_dict)
        seed_controls["preset_dirty"] = True
        state["seed_controls"] = seed_controls

        comp_video_status = "enabled" if seed_controls["generate_comparison_video_val"] else "disabled"
        global_rife_status = "enabled" if seed_controls["global_rife_enabled_val"] else "disabled"
        global_rife_chunk_mode = "per-chunk" if seed_controls["global_rife_process_chunks_val"] else "final-only"
        status = (
            f"Applied output settings\n"
            f"- Format: {seed_controls['output_format_val']}\n"
            f"- PNG Sequence: {seed_controls['png_sequence_enabled_val']}\n"
            f"- Overwrite Existing Batch Outputs: {seed_controls['overwrite_existing_batch_val']}\n"
            f"- Global RIFE: {global_rife_status} ({seed_controls['global_rife_multiplier_val']}, {seed_controls['global_rife_precision_val']}, {global_rife_chunk_mode})\n"
            f"- Comparison: {seed_controls['comparison_mode_val']}\n"
            f"- Comparison Video: {comp_video_status}\n"
            f"- Metadata: {seed_controls['save_metadata_val']}"
        )
        return gr.update(value=status), state

    def pin_reference_frame(image_path, state):
        """
        Pin a reference frame for iterative comparison.
        Persists to global settings for cross-session persistence.
        """
        from pathlib import Path
        
        seed_controls = state.get("seed_controls", {})
        if image_path and Path(image_path).exists():
            seed_controls["pinned_reference_path"] = image_path
            # Persist to global settings
            if global_settings is not None:
                global_settings["pinned_reference_path"] = image_path
                preset_manager.save_global_settings(global_settings)
            msg = f"✅ Reference pinned: {Path(image_path).name}\n💾 Saved to global settings (persists across sessions)"
        else:
            seed_controls["pinned_reference_path"] = None
            msg = "⚠️ No valid image to pin"
        state["seed_controls"] = seed_controls
        return gr.update(value=msg), state

    def unpin_reference(state):
        """
        Clear pinned reference.
        Removes from both runtime state and global settings.
        """
        seed_controls = state.get("seed_controls", {})
        seed_controls["pinned_reference_path"] = None
        # Clear from global settings
        if global_settings is not None:
            global_settings["pinned_reference_path"] = None
            preset_manager.save_global_settings(global_settings)
        state["seed_controls"] = seed_controls
        return gr.update(value="✅ Reference unpinned and cleared from global settings"), state

    return {
        "defaults": defaults,
        "order": OUTPUT_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "apply_codec_preset": apply_codec_preset,
        "update_codec_info": update_codec_info,
        "update_pixel_format_info": update_pixel_format_info,
        "cache_output": lambda *args: cache_output(*args[:-1], args[-1]),
        "cache_fps": lambda *args: cache_fps(*args[:-1], args[-1]),
        "cache_comparison": lambda *args: cache_comparison(*args[:-1], args[-1]),
        "cache_pin": lambda *args: cache_pin(*args[:-1], args[-1]),
        "cache_fullscreen": lambda *args: cache_fullscreen(*args[:-1], args[-1]),
        "cache_png_padding": lambda *args: cache_png_padding(*args[:-1], args[-1]),
        "cache_png_basename": lambda *args: cache_png_basename(*args[:-1], args[-1]),
        "cache_skip": lambda *args: cache_skip(*args[:-1], args[-1]),
        "cache_cap": lambda *args: cache_cap(*args[:-1], args[-1]),
        "cache_overwrite_batch": lambda *args: cache_overwrite_batch(*args[:-1], args[-1]),
        "cache_generate_comparison_video": lambda *args: cache_generate_comparison_video(*args[:-1], args[-1]),
        "cache_comparison_video_layout": lambda *args: cache_comparison_video_layout(*args[:-1], args[-1]),
        "apply_to_pipeline": apply_to_pipeline,
        "pin_reference_frame": pin_reference_frame,
        "unpin_reference": unpin_reference,
    }


