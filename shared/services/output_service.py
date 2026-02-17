from typing import Any, Dict, List, Optional
import gradio as gr

from shared.preset_manager import PresetManager
from shared.video_codec_options import (
    get_codec_info,
    get_pixel_format_info,
    get_recommended_settings,
    get_codec_choices,
    get_pixel_format_choices,
    ENCODING_PRESETS,
    AUDIO_CODECS,
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
        # Global image output settings (applies to image results from all model tabs).
        "image_output_format": "png",
        "image_output_quality": 95,
        # SeedVR2 encoding controls moved from SeedVR2 tab into global Output tab.
        "seedvr2_video_backend": "opencv",
        "seedvr2_use_10bit": False,
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
    "image_output_format",
    "image_output_quality",
    "seedvr2_video_backend",
    "seedvr2_use_10bit",
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
    cfg["output_format"] = str(cfg.get("output_format", "auto") or "auto").strip().lower()
    if cfg["output_format"] not in {"auto", "mp4", "png"}:
        cfg["output_format"] = "auto"

    image_format = str(cfg.get("image_output_format", "png") or "png").strip().lower()
    if image_format not in {"png", "jpg", "webp"}:
        image_format = "png"
    cfg["image_output_format"] = image_format
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
    codec_choices = set(get_codec_choices())
    if codec not in codec_choices:
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
    }


