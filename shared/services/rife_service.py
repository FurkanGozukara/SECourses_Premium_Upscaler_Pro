"""
RIFE Service Module - Clean Implementation
Handles RIFE/FPS/Edit Videos processing logic, presets, and callbacks
"""

import shutil
import subprocess
import queue
import threading
import time
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr

from shared.preset_manager import PresetManager
from shared.runner import Runner
from shared.path_utils import normalize_path, ffmpeg_set_fps, get_media_dimensions
from shared.face_restore import restore_video
from shared.logging_utils import RunLogger
from shared.models.rife_meta import get_rife_metadata, get_rife_default_model
from shared.gpu_utils import expand_cuda_device_spec, validate_cuda_device_spec
from shared.error_handling import logger as error_logger
from shared.oom_alert import clear_vram_oom_alert, maybe_set_vram_oom_alert, show_vram_oom_modal
from shared.output_run_manager import prepare_single_video_run
from shared.video_codec_options import build_ffmpeg_video_encode_args
from shared.comparison_video_service import maybe_generate_input_vs_output_comparison
from shared.global_rife import global_rife_enabled


# Defaults and ordering --------------------------------------------------------
def rife_defaults(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get default RIFE settings aligned with RIFE CLI.
    Applies model-specific metadata when model_name is provided.
    """
    # IMPORTANT: do not import torch in the parent Gradio process.
    # Use NVML-based detection (nvidia-smi) via shared.gpu_utils instead.
    try:
        from shared.gpu_utils import get_gpu_info
        cuda_default = "0" if get_gpu_info() else ""
    except Exception:
        cuda_default = ""
    
    # Get model metadata if specific model is provided
    default_model = model_name or get_rife_default_model()
    model_meta = get_rife_metadata(default_model)
    
    # Apply model-specific defaults if metadata available
    if model_meta:
        recommended_uhd = model_meta.recommended_uhd_threshold
    else:
        recommended_uhd = 2160
    # Default requested by app UX: FP32.
    default_precision = "fp32"
    
    return {
        "input_path": "",
        "rife_enabled": True,
        "output_override": "",
        "output_format": "mp4",
        "model_dir": "",
        "model": default_model,  # Fixed: was "rife_model", should match RIFE_ORDER
        "fps_multiplier": "x2",
        "fps_override": 0,  # Fixed: was "target_fps", should match RIFE_ORDER
        "scale": 1.0,
        "uhd_mode": False,
        # Kept as string to align with UI dropdown choices ("fp16"/"fp32").
        "fp16_mode": default_precision,
        "png_output": False,
        "no_audio": False,
        "show_ffmpeg": False,
        "montage": False,
        "img_mode": False,
        "skip_static_frames": False,
        "exp": 1,
        "multi": 2,
        "batch_enable": False,
        "batch_input_path": "",
        "batch_output_path": "",
        "skip_first_frames": 0,
        "load_cap": 0,
        "cuda_device": cuda_default,
        # Video editing parameters
        "edit_mode": "none",
        "start_time": "",
        "end_time": "",
        "speed_factor": 1.0,
        "video_codec": "libx264",
        "output_quality": 23,
        "concat_videos": "",
        "_recommended_uhd_threshold": recommended_uhd,  # Store for validation
    }


"""
ðŸ“‹ RIFE PRESET ORDER - MUST match inputs_list in ui/rife_tab.py
Adding controls? Update rife_defaults(), RIFE_ORDER, and inputs_list in sync.
Current count: 32 components
"""

RIFE_ORDER: List[str] = [
    "input_path",
    "rife_enabled",
    "output_override",
    "output_format",
    "model_dir",
    "model",
    "fps_multiplier",
    "fps_override",
    "scale",
    "uhd_mode",
    "fp16_mode",
    "png_output",
    "no_audio",
    "show_ffmpeg",
    "montage",
    "img_mode",
    "skip_static_frames",
    "exp",
    "multi",
    "batch_enable",
    "batch_input_path",
    "batch_output_path",
    "skip_first_frames",
    "load_cap",
    "cuda_device",
    # Video editing parameters
    "edit_mode",
    "start_time",
    "end_time",
    "speed_factor",
    "video_codec",
    "output_quality",
    "concat_videos",
]


# Video editing functions -----------------------------------------------------

def _parse_time_to_seconds(time_str: str) -> float:
    """Parse time string in HH:MM:SS or seconds format to total seconds."""
    if not time_str or time_str.strip() == "":
        return 0.0

    time_str = time_str.strip()

    # Check if it's already a number (seconds)
    try:
        return float(time_str)
    except ValueError:
        pass

    # Parse HH:MM:SS format
    match = re.match(r'^(\d{1,2}):(\d{1,2}):(\d{1,2}(?:\.\d+)?)$', time_str)
    if match:
        hours, minutes, seconds = match.groups()
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

    # Try MM:SS format
    match = re.match(r'^(\d{1,2}):(\d{1,2}(?:\.\d+)?)$', time_str)
    if match:
        minutes, seconds = match.groups()
        return float(minutes) * 60 + float(seconds)

    # If nothing matches, return 0
    return 0.0


def _normalize_video_codec_key(codec_raw: str) -> str:
    codec = str(codec_raw or "h264").strip().lower()
    codec_map = {
        "libx264": "h264",
        "x264": "h264",
        "h264": "h264",
        "avc": "h264",
        "libx265": "h265",
        "x265": "h265",
        "h265": "h265",
        "hevc": "h265",
        "libvpx-vp9": "vp9",
        "vp9": "vp9",
        "libaom-av1": "av1",
        "av1": "av1",
        "prores": "prores",
        "prores_ks": "prores",
    }
    return codec_map.get(codec, "h264")


def _trim_video(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    video_codec: str = "libx264",
    quality: int = 23,
    no_audio: bool = False,
    video_preset: str = "fast",
    pixel_format: str = "yuv420p",
    audio_codec: str = "aac",
    audio_bitrate: Optional[str] = None,
) -> Tuple[bool, str]:
    """Trim video using FFmpeg."""
    try:
        cmd = ["ffmpeg", "-y"]

        # Input
        cmd.extend(["-i", input_path])

        # Time range
        if start_time > 0:
            cmd.extend(["-ss", str(start_time)])
        if end_time > 0:
            cmd.extend(["-to", str(end_time)])

        codec_key = _normalize_video_codec_key(video_codec)
        selected_audio_codec = "none" if no_audio else str(audio_codec or "aac")
        cmd.extend(
            build_ffmpeg_video_encode_args(
                codec=codec_key,
                quality=int(quality),
                pixel_format=str(pixel_format or "yuv420p"),
                preset=str(video_preset or "fast"),
                audio_codec=selected_audio_codec,
                audio_bitrate=str(audio_bitrate) if audio_bitrate else None,
            )
        )

        cmd.append(output_path)

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        return result.returncode == 0, result.stderr or result.stdout

    except Exception as e:
        return False, f"Trimming failed: {str(e)}"


def _change_video_speed(
    input_path: str,
    output_path: str,
    speed_factor: float,
    video_codec: str = "libx264",
    quality: int = 23,
    no_audio: bool = False,
    video_preset: str = "fast",
    pixel_format: str = "yuv420p",
    audio_codec: str = "aac",
    audio_bitrate: Optional[str] = None,
) -> Tuple[bool, str]:
    """Change video speed using FFmpeg."""
    try:
        if speed_factor <= 0:
            return False, "Speed factor must be greater than 0"

        # For speed changes, we need to adjust both video and audio
        # Video filter for speed change
        video_filter = f"setpts={1/speed_factor}*PTS"

        # Audio filter for speed change (if not removing audio)
        audio_filter = ""
        if not no_audio:
            audio_filter = f"atempo={speed_factor}"

        cmd = ["ffmpeg", "-y", "-i", input_path]

        # Apply filters
        if video_filter:
            cmd.extend(["-filter:v", video_filter])
        if audio_filter:
            cmd.extend(["-filter:a", audio_filter])

        codec_key = _normalize_video_codec_key(video_codec)
        selected_audio_codec = "none" if no_audio else str(audio_codec or "aac")
        cmd.extend(
            build_ffmpeg_video_encode_args(
                codec=codec_key,
                quality=int(quality),
                pixel_format=str(pixel_format or "yuv420p"),
                preset=str(video_preset or "fast"),
                audio_codec=selected_audio_codec,
                audio_bitrate=str(audio_bitrate) if audio_bitrate else None,
            )
        )

        cmd.append(output_path)

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        return result.returncode == 0, result.stderr or result.stdout

    except Exception as e:
        return False, f"Speed change failed: {str(e)}"


def _concatenate_videos(
    video_paths: List[str],
    output_path: str,
    video_codec: str = "libx264",
    quality: int = 23,
    no_audio: bool = False,
    video_preset: str = "fast",
    pixel_format: str = "yuv420p",
    audio_codec: str = "copy",
    audio_bitrate: Optional[str] = None,
) -> Tuple[bool, str]:
    """Concatenate multiple videos using FFmpeg."""
    try:
        if len(video_paths) < 2:
            return False, "Need at least 2 videos to concatenate"

        # Create a temporary concat file
        concat_file = output_path + ".concat.txt"
        with open(concat_file, 'w', encoding='utf-8') as f:
            for video_path in video_paths:
                f.write(f"file '{video_path.replace(chr(39), chr(39) + chr(39))}'\n")

        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file]

        codec_key = _normalize_video_codec_key(video_codec)
        selected_audio_codec = "none" if no_audio else str(audio_codec or "copy")
        cmd.extend(
            build_ffmpeg_video_encode_args(
                codec=codec_key,
                quality=int(quality),
                pixel_format=str(pixel_format or "yuv420p"),
                preset=str(video_preset or "fast"),
                audio_codec=selected_audio_codec,
                audio_bitrate=str(audio_bitrate) if audio_bitrate else None,
            )
        )

        cmd.append(output_path)

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

        # Clean up concat file
        try:
            Path(concat_file).unlink(missing_ok=True)
        except:
            pass

        return result.returncode == 0, result.stderr or result.stdout

    except Exception as e:
        # Clean up concat file on error
        try:
            Path(concat_file).unlink(missing_ok=True)
        except:
            pass
        return False, f"Concatenation failed: {str(e)}"


def _apply_video_editing(input_path: str, output_path: str, settings: Dict[str, Any],
                        temp_dir: Path, progress_cb: Optional[Callable[[str], None]] = None) -> Tuple[bool, str, str]:
    """
    Apply video editing operations to the input video.
    Returns (success, log_message, final_output_path)
    """
    edit_mode = settings.get("edit_mode", "none")
    if edit_mode == "none":
        # No editing, just copy the file
        shutil.copy2(input_path, output_path)
        return True, "No video editing applied", output_path

    try:
        # Parse common parameters
        start_time = _parse_time_to_seconds(settings.get("start_time", ""))
        end_time = _parse_time_to_seconds(settings.get("end_time", ""))
        speed_factor = float(settings.get("speed_factor", 1.0))
        video_codec = settings.get("video_codec", "libx264")
        quality = int(settings.get("video_quality", settings.get("output_quality", 23)))
        video_preset = str(settings.get("video_preset", "fast") or "fast")
        pixel_format = str(settings.get("pixel_format", "yuv420p") or "yuv420p")
        audio_codec = str(settings.get("audio_codec", "copy") or "copy")
        audio_bitrate = settings.get("audio_bitrate") or None
        no_audio = bool(settings.get("no_audio", False)) or audio_codec.strip().lower() in ("none", "no", "off", "disable", "disabled")

        temp_output = temp_dir / f"temp_edit_{Path(output_path).name}"

        if edit_mode == "trim":
            if progress_cb:
                progress_cb("Trimming video...")
            success, log = _trim_video(
                input_path, str(temp_output), start_time, end_time,
                video_codec, quality, no_audio, video_preset, pixel_format, audio_codec, audio_bitrate
            )
            if not success:
                return False, f"Trimming failed: {log}", ""

        elif edit_mode == "speed_change":
            if progress_cb:
                progress_cb(f"Changing video speed by factor {speed_factor}...")
            success, log = _change_video_speed(
                input_path, str(temp_output), speed_factor,
                video_codec, quality, no_audio, video_preset, pixel_format, audio_codec, audio_bitrate
            )
            if not success:
                return False, f"Speed change failed: {log}", ""

        elif edit_mode == "concatenate":
            # For concatenation, we need additional video paths
            concat_videos_str = settings.get("concat_videos", "").strip()
            if not concat_videos_str:
                return False, "Concatenation mode requires additional video paths", ""

            # Parse the concatenation videos (comma-separated paths)
            concat_paths = [path.strip() for path in concat_videos_str.split(",") if path.strip()]
            if not concat_paths:
                return False, "No valid video paths found for concatenation", ""

            # Add the main input video as the first video
            all_paths = [input_path] + concat_paths

            if progress_cb:
                progress_cb(f"Concatenating {len(all_paths)} videos...")
            success, log = _concatenate_videos(
                all_paths,
                str(temp_output),
                video_codec,
                quality,
                no_audio,
                video_preset,
                pixel_format,
                audio_codec,
                audio_bitrate,
            )
            if not success:
                return False, f"Concatenation failed: {log}", ""

        elif edit_mode == "effects":
            # Placeholder for future effects
            if progress_cb:
                progress_cb("Applying video effects...")
            # For now, just copy the file
            shutil.copy2(input_path, temp_output)
            success, log = True, "Basic effects applied (placeholder)"

        else:
            return False, f"Unknown edit mode: {edit_mode}", ""

        if success:
            # Move temp output to final output
            temp_output.replace(output_path)
            return True, f"Video editing completed: {edit_mode}", output_path
        else:
            return False, log, ""

    except Exception as e:
        return False, f"Video editing error: {str(e)}", ""


# Helper functions -------------------------------------------------------------
def _validate_cuda_devices(cuda_spec: str) -> Optional[str]:
    """Validate CUDA device specification."""
    try:
        if not cuda_spec:
            return None
        expanded = expand_cuda_device_spec(cuda_spec)
        return validate_cuda_device_spec(expanded)
    except Exception as exc:
        return f"CUDA validation failed: {exc}"


def _ffmpeg_available() -> bool:
    """Check if ffmpeg is available in PATH."""
    return shutil.which("ffmpeg") is not None


def _normalize_global_multiplier(raw: Any) -> int:
    """Normalize global multiplier input into {1,2,4,8}."""
    text = str(raw or "x2").strip().lower()
    if text.startswith("x"):
        text = text[1:]
    try:
        val = int(float(text))
    except Exception:
        val = 2
    if val <= 1:
        return 1
    if val <= 2:
        return 2
    if val <= 4:
        return 4
    return 8


def _rife_processing_requested(cfg: Dict[str, Any]) -> bool:
    """Return True when any RIFE-processing-affecting flag is enabled."""
    try:
        fps_mult_val = int(cfg.get("fps_multiplier", 1) or 1)
    except Exception:
        fps_mult_val = 1

    return (
        fps_mult_val > 1
        or float(cfg.get("fps_override", 0.0) or 0.0) > 0.0
        or float(cfg.get("scale", 1.0) or 1.0) != 1.0
        or bool(cfg.get("png_output", False))
        or bool(cfg.get("montage", False))
        or bool(cfg.get("skip_static_frames", False))
        or int(cfg.get("exp", 1) or 1) != 1
    )


def _apply_global_rife_override(
    settings: Dict[str, Any],
    seed_controls: Dict[str, Any],
    output_settings: Dict[str, Any],
) -> bool:
    """
    Force RIFE processing from global settings when local RIFE would be skipped.

    Returns:
        True if global override was applied.
    """
    if not global_rife_enabled(seed_controls):
        return False
    if _rife_processing_requested(settings):
        return False

    mult_val = _normalize_global_multiplier(
        output_settings.get(
            "global_rife_multiplier",
            seed_controls.get("global_rife_multiplier_val", "x2"),
        )
    )
    # Global RIFE is intended as an enabled interpolation pass; keep at least x2.
    if mult_val < 2:
        mult_val = 2
    settings["fps_multiplier"] = int(mult_val)

    model_name = str(
        output_settings.get(
            "global_rife_model",
            seed_controls.get("global_rife_model_val", ""),
        ) or ""
    ).strip() or get_rife_default_model()
    settings["model"] = model_name

    precision = str(
        output_settings.get(
            "global_rife_precision",
            seed_controls.get("global_rife_precision_val", "fp32"),
        ) or "fp32"
    ).strip().lower()
    settings["fp16_mode"] = precision == "fp16"

    cuda_device = str(
        output_settings.get(
            "global_rife_cuda_device",
            seed_controls.get("global_rife_cuda_device_val", ""),
        ) or ""
    ).strip()
    if "," in cuda_device:
        cuda_device = cuda_device.split(",", 1)[0].strip()
    if cuda_device.lower() == "all":
        cuda_device = "0"
    if cuda_device:
        settings["cuda_device"] = cuda_device

    settings["_forced_global_rife"] = True
    return True


# Preset helpers ---------------------------------------------------------------
def _rife_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    """Convert argument list to settings dictionary."""
    return dict(zip(RIFE_ORDER, args))


def _enforce_rife_guardrails(cfg: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply RIFE-specific validation rules using metadata registry.
    
    Enforces:
    - Single GPU requirement (RIFE doesn't support multi-GPU)
    - UHD mode recommendations for 4K+ content
    - FPS multiplier constraints from model metadata
    - Precision compatibility checks
    """
    cfg = cfg.copy()
    if not str(cfg.get("model", "") or "").strip():
        cfg["model"] = get_rife_default_model()
    
    # Get model metadata
    model_name = cfg.get("model", get_rife_default_model())
    model_meta = get_rife_metadata(model_name)
    
    if model_meta:
        # Enforce single GPU (RIFE is single-GPU optimized)
        gpu_device_str = str(cfg.get("cuda_device", ""))
        if gpu_device_str and gpu_device_str not in ("", "cpu"):
            devices = [d.strip() for d in gpu_device_str.replace(" ", "").split(",") if d.strip()]
            if len(devices) > 1:
                error_logger.warning(f"RIFE doesn't support multi-GPU - forcing single GPU (using first: {devices[0]})")
                cfg["cuda_device"] = devices[0]
                cfg["_multi_gpu_disabled_reason"] = "RIFE is single-GPU optimized"
        
        # Validate FPS multiplier against model limits
        fps_mult_str = str(cfg.get("fps_multiplier", "x2"))
        try:
            # Extract numeric multiplier (e.g., "x2" -> 2)
            mult_value = int(fps_mult_str.replace("x", "").strip())
            max_mult = model_meta.max_fps_multiplier
            
            if mult_value > max_mult:
                error_logger.warning(f"FPS multiplier {mult_value}x exceeds model limit {max_mult}x, clamping")
                cfg["fps_multiplier"] = f"x{max_mult}"
                cfg["_fps_mult_clamped_reason"] = f"Model {model_name} max: {max_mult}x"
        except (ValueError, AttributeError):
            pass  # Keep original value if parsing fails
        
        # Auto-enable UHD mode for high-resolution content
        # This is a recommendation check - actual enabling happens in processing
        if model_meta.supports_uhd:
            cfg["_uhd_threshold"] = model_meta.recommended_uhd_threshold
    
    # Validate scale factor (must be positive)
    try:
        scale = float(cfg.get("scale", 1.0))
        if scale <= 0:
            cfg["scale"] = 1.0
            error_logger.warning(f"Invalid scale {scale}, resetting to 1.0")
    except (ValueError, TypeError):
        cfg["scale"] = 1.0

    # Normalize precision value for dropdown/runtime compatibility.
    precision_raw = cfg.get("fp16_mode", defaults.get("fp16_mode", "fp32"))
    if isinstance(precision_raw, bool):
        cfg["fp16_mode"] = "fp16" if precision_raw else "fp32"
    else:
        text = str(precision_raw).strip().lower()
        cfg["fp16_mode"] = "fp16" if text in ("fp16", "true", "1", "yes", "on") else "fp32"

    # Normalize FPS multiplier into xN format accepted by the UI dropdown.
    fps_raw = str(cfg.get("fps_multiplier", "x2")).strip().lower()
    if fps_raw.startswith("x"):
        fps_raw = fps_raw[1:]
    try:
        fps_val = int(float(fps_raw))
    except Exception:
        fps_val = 2
    fps_val = 1 if fps_val <= 1 else (2 if fps_val <= 2 else (4 if fps_val <= 4 else 8))
    cfg["fps_multiplier"] = f"x{fps_val}"
    
    return cfg


def _apply_rife_preset(
    preset: Dict[str, Any],
    defaults: Dict[str, Any],
    preset_manager: PresetManager,
    current: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Apply preset values to current settings."""
    base = defaults.copy()
    if current:
        base.update(current)
    merged = preset_manager.merge_config(base, preset)
    # Apply guardrails to merged preset
    merged = _enforce_rife_guardrails(merged, defaults)
    return [merged[key] for key in RIFE_ORDER]


# Core run/cancel/preset callbacks --------------------------------------------
def build_rife_callbacks(
    preset_manager: PresetManager,
    runner: Runner,
    run_logger: RunLogger,
    global_settings: Dict[str, Any],
    output_dir: Path,
    temp_dir: Path,
    shared_state: gr.State,
):
    """Build RIFE callback functions for the UI."""
    defaults = rife_defaults()

    def refresh_presets(model_name: str, select_name: Optional[str] = None):
        """Refresh preset dropdown."""
        presets = preset_manager.list_presets("rife", model_name)
        last_used = preset_manager.get_last_used_name("rife", model_name)
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        return gr.update(choices=presets, value=value)

    def save_preset(preset_name: str, *args):
        """Save preset with validation"""
        if not preset_name.strip():
            return gr.update(), gr.update(value="âš ï¸ Enter a preset name before saving"), *list(args)

        try:
            # Validate component count
            if len(args) != len(RIFE_ORDER):
                error_msg = f"âš ï¸ Preset mismatch: {len(args)} values vs {len(RIFE_ORDER)} expected. Check inputs_list in rife_tab.py"
                return gr.update(), gr.update(value=error_msg), *list(args)
            
            payload = _rife_dict_from_args(list(args))
            model_name = payload["model"]
            preset_manager.save_preset_safe("rife", model_name, preset_name.strip(), payload)
            dropdown = refresh_presets(model_name, select_name=preset_name.strip())

            current_map = dict(zip(RIFE_ORDER, list(args)))
            loaded_vals = _apply_rife_preset(payload, defaults, preset_manager, current=current_map)

            return dropdown, gr.update(value=f"âœ… Saved preset '{preset_name}' for {model_name}"), *loaded_vals
        except Exception as e:
            return gr.update(), gr.update(value=f"âŒ Error saving preset: {str(e)}"), *list(args)

    def load_preset(preset_name: str, model_name: str, current_values: List[Any]):
        """
        Load a preset.
        
        FIXED: Now returns (*values, status_message) to match UI output expectations.
        UI expects: inputs_list + [preset_status]
        """
        try:
            model_name = model_name or defaults["model"]
            preset = preset_manager.load_preset_safe("rife", model_name, preset_name)
            if preset:
                preset_manager.set_last_used("rife", model_name, preset_name)

            defaults_with_model = defaults.copy()
            defaults_with_model["model"] = model_name
            current_map = dict(zip(RIFE_ORDER, current_values))
            values = _apply_rife_preset(preset or {}, defaults_with_model, preset_manager, current=current_map)
            
            # Return values + status message (status is LAST)
            status_msg = f"âœ… Loaded preset '{preset_name}'" if preset else "â„¹ï¸ Preset not found"
            return (*values, gr.update(value=status_msg))
        except Exception as e:
            print(f"Error loading preset {preset_name}: {e}")
            # Return current values + error status
            return (*current_values, gr.update(value=f"âŒ Error: {str(e)}"))

    def safe_defaults():
        """Get safe default values."""
        return [defaults[k] for k in RIFE_ORDER]

    def run_action(
        uploaded_file,
        img_folder,
        *args,
        state=None,
        global_settings_snapshot: Dict[str, Any] | None = None,
        _global_settings: Dict[str, Any] = global_settings,
    ):
        """Main RIFE processing action with pre-flight checks."""
        try:
            global_settings = (
                dict(global_settings_snapshot)
                if isinstance(global_settings_snapshot, dict)
                else dict(_global_settings)
            )
            state = state or {"seed_controls": {}, "operation_status": "ready"}
            state["operation_status"] = "running"
            # Clear any previous VRAM OOM banner at the start of a new run.
            clear_vram_oom_alert(state)
            seed_controls = state.get("seed_controls", {})
            output_settings = seed_controls.get("output_settings", {}) if isinstance(seed_controls, dict) else {}
            if not isinstance(output_settings, dict):
                output_settings = {}
            
            settings_dict = _rife_dict_from_args(list(args))
            settings = {**defaults, **settings_dict}
            
            # Apply RIFE guardrails (single GPU, FPS limits, etc.)
            settings = _enforce_rife_guardrails(settings, defaults)
            precision_raw = settings.get("fp16_mode", "fp32")
            settings["fp16_mode"] = str(precision_raw).strip().lower() == "fp16"
            # Normalize fps multiplier to an int for runtime (UI uses strings like "x2").
            try:
                fps_mult_raw = settings.get("fps_multiplier", 2)
                if isinstance(fps_mult_raw, str):
                    fps_mult = int(fps_mult_raw.lower().replace("x", "").strip() or "2")
                else:
                    fps_mult = int(fps_mult_raw)
                settings["fps_multiplier"] = max(1, int(fps_mult))
            except Exception:
                settings["fps_multiplier"] = 2

            # PRE-FLIGHT CHECKS (mirrors SeedVR2/GAN for consistency)
            from shared.error_handling import check_ffmpeg_available, check_disk_space
            
            # Check ffmpeg availability
            ffmpeg_ok, ffmpeg_msg = check_ffmpeg_available()
            if not ffmpeg_ok:
                yield ("âŒ ffmpeg not found in PATH", ffmpeg_msg or "Install ffmpeg and add to PATH before processing", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                return
            
            # Check disk space (require at least 5GB free)
            output_path_check = Path(global_settings.get("output_dir", output_dir))
            has_space, space_warning = check_disk_space(output_path_check, required_mb=5000)
            if not has_space:
                yield ("âŒ Insufficient disk space", space_warning or "Free up at least 5GB disk space before processing", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                return

            input_path = normalize_path(uploaded_file if uploaded_file else img_folder)
            if not input_path or not Path(input_path).exists():
                yield ("âŒ Input missing or not found", "", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                return

            # Validate input type based on mode
            if settings.get("img_mode"):
                # In --img mode, require a frames folder or images
                if Path(input_path).is_file() and Path(input_path).suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"):
                    yield ("âš ï¸ --img mode expects frames folder or images, not a video file.", "", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                    return
            else:
                # In video mode, require a video file
                if Path(input_path).is_dir():
                    yield ("âš ï¸ Video mode expects a video file. Enable --img for frame folders.", "", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                    return

            settings["input_path"] = input_path
            settings["output_override"] = settings.get("output_override") or None

            # Expand "all" to device list if specified (using shared GPU utility)
            cuda_device_raw = settings.get("cuda_device", "")
            if cuda_device_raw:
                settings["cuda_device"] = expand_cuda_device_spec(cuda_device_raw)

            # Validate CUDA devices (using shared GPU utility)
            cuda_warning = validate_cuda_device_spec(settings.get("cuda_device", ""))
            if cuda_warning:
                yield (f"âš ï¸ {cuda_warning}", "", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                return

            # Check ffmpeg availability
            if not _ffmpeg_available():
                yield ("âŒ ffmpeg not found in PATH. Install ffmpeg and retry.", "", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                return

            # Apply cached values from Resolution & Scene Split tab (vNext Upscale-x)
            scale_x = seed_controls.get("upscale_factor_val")
            max_edge = int(seed_controls.get("max_resolution_val", 0) or 0)
            enable_max = seed_controls.get("enable_max_target", True)
            if not enable_max:
                max_edge = 0

            if scale_x is not None:
                try:
                    scale_x = float(scale_x)
                except Exception:
                    scale_x = None

            if scale_x is not None:
                input_dims = get_media_dimensions(input_path)
                if input_dims:
                    input_w, input_h = input_dims
                    long_side = max(input_w, input_h)

                    # Apply max-edge cap (LONG side) to compute effective scale
                    effective_scale = float(scale_x)
                    if max_edge > 0 and long_side > 0:
                        capped = max_edge / long_side
                        effective_scale = min(effective_scale, capped)

                    # Clamp to RIFE's reasonable range
                    effective_scale = max(0.5, min(4.0, effective_scale))
                    settings["scale"] = effective_scale

            # Apply Output tab format preferences when RIFE tab format is not explicitly set.
            cached_fmt = str(seed_controls.get("output_format_val") or "").strip().lower()
            png_seq_enabled = bool(
                seed_controls.get(
                    "png_sequence_enabled_val",
                    output_settings.get("png_sequence_enabled", False),
                )
            )
            if settings.get("output_format") in (None, "auto"):
                if cached_fmt in ("mp4", "png"):
                    settings["output_format"] = cached_fmt
                elif png_seq_enabled:
                    settings["output_format"] = "png"

            # Audio preferences (used by chunking + final muxing). "Remove Audio" overrides everything.
            audio_codec = str(seed_controls.get("audio_codec_val") or "copy")
            audio_bitrate = seed_controls.get("audio_bitrate_val") or ""
            if bool(settings.get("no_audio", False)):
                audio_codec = "none"
            settings["audio_codec"] = audio_codec
            settings["audio_bitrate"] = audio_bitrate
            if output_settings:
                for key in (
                    "video_codec",
                    "video_quality",
                    "video_preset",
                    "pixel_format",
                    "two_pass_encoding",
                    "metadata_format",
                    "log_level",
                    "temporal_padding",
                ):
                    if output_settings.get(key) is not None:
                        settings[key] = output_settings.get(key)

            # Global RIFE override:
            # If user enabled Global RIFE and local RIFE settings would skip processing,
            # force an interpolation pass using global model/multiplier/precision/device.
            global_rife_forced = _apply_global_rife_override(settings, seed_controls, output_settings)
            
            # Pull PySceneDetect chunking settings from Resolution tab (universal chunking)
            auto_chunk = bool(seed_controls.get("auto_chunk", True))
            frame_accurate_split = bool(seed_controls.get("frame_accurate_split", True))
            chunk_size_sec = float(seed_controls.get("chunk_size_sec", 0) or 0)
            chunk_overlap_sec = 0.0 if auto_chunk else float(seed_controls.get("chunk_overlap_sec", 0) or 0)
            per_chunk_cleanup = seed_controls.get("per_chunk_cleanup", False)
            # PySceneDetect parameters now managed centrally in Resolution tab
            scene_threshold = float(seed_controls.get("scene_threshold", 27.0))
            min_scene_len = float(seed_controls.get("min_scene_len", 1.0))
            settings["frame_accurate_split"] = frame_accurate_split
            
            # Determine if PySceneDetect chunking should be used
            from shared.path_utils import detect_input_type as detect_type
            input_type_check = detect_type(input_path)

            # NEW: Per-run output folder for single video runs (0001/0002/...) to avoid collisions and
            # to keep chunk artifacts user-visible.
            if (
                input_type_check == "video"
                and not settings.get("batch_enable", False)
                and not settings.get("img_mode", False)
            ):
                try:
                    base_out_root = Path(global_settings.get("output_dir", output_dir))
                    run_paths, explicit_final = prepare_single_video_run(
                        output_root_fallback=base_out_root,
                        output_override_raw=settings.get("output_override") or "",
                        input_path=input_path,
                        original_filename=Path(input_path).name,
                        model_label="RIFE",
                        mode=str(getattr(runner, "get_mode", lambda: "subprocess")() or "subprocess"),
                    )
                    run_dir = Path(run_paths.run_dir)
                    seed_controls["last_run_dir"] = str(run_dir)
                    settings["_run_dir"] = str(run_dir)
                    settings["_processed_chunks_dir"] = str(run_paths.processed_chunks_dir)
                    settings["_user_output_override_raw"] = str(settings.get("output_override") or "")

                    png_output = bool(settings.get("png_output", False))
                    if png_output:
                        # PNG sequence output: force an output DIRECTORY inside the run folder.
                        # If user provided a file override, reuse its stem as the directory name.
                        if explicit_final:
                            default_final = run_dir / Path(explicit_final).stem
                        else:
                            default_final = run_dir / f"{Path(input_path).stem}_png"
                        settings["output_override"] = str(default_final)
                    else:
                        out_ext = str(settings.get("output_format") or "mp4")
                        if out_ext == "auto":
                            out_ext = "mp4"
                        out_ext = out_ext.lstrip(".")
                        default_final = run_dir / f"{Path(input_path).stem}.{out_ext}"
                        settings["output_override"] = str(explicit_final) if explicit_final else str(default_final)
                except Exception:
                    pass

            should_use_chunking = (
                (auto_chunk or chunk_size_sec > 0) and
                input_type_check == "video" and
                not settings.get("batch_enable", False) and
                not settings.get("img_mode", False) and  # Don't chunk image sequences
                not bool(settings.get("png_output", False))  # PNG export uses directory outputs; keep flow simple
            )
            
            # If chunking enabled, use universal chunk_and_process for RIFE
            if should_use_chunking:
                from shared.chunking import chunk_and_process
                
                mode_label = "Auto Chunk (PySceneDetect scenes)" if auto_chunk else f"Static Chunk ({chunk_size_sec:g}s)"
                init_desc = "Initializing scene detection..." if auto_chunk else "Initializing chunking..."
                yield (
                    f"âš™ï¸ Starting {mode_label} for RIFE processing...",
                    init_desc,
                    gr.update(value="Chunking...", visible=True),
                    None,
                    gr.update(value=None),
                    gr.update(value="", visible=False),
                    state,
                )
                
                # Prepare settings for chunking
                settings["chunk_size_sec"] = chunk_size_sec
                settings["chunk_overlap_sec"] = chunk_overlap_sec
                settings["per_chunk_cleanup"] = per_chunk_cleanup
                settings["frame_accurate_split"] = frame_accurate_split
                
                def chunk_progress_cb(progress_val, desc=""):
                    yield (f"âš™ï¸ Chunking: {desc}", f"Processing chunks... {desc}", gr.update(value=desc, visible=True), None, gr.update(value=None), gr.update(value="", visible=False), state)
                
                # Run chunked RIFE processing
                rc, clog, final_output, chunk_count = chunk_and_process(
                    runner=runner,
                    settings=settings,
                    scene_threshold=scene_threshold,
                    min_scene_len=min_scene_len,
                    work_dir=Path(settings.get("_run_dir") or output_dir),
                    on_progress=lambda msg: None,
                    chunk_seconds=0.0 if auto_chunk else chunk_size_sec,
                    chunk_overlap=0.0 if auto_chunk else chunk_overlap_sec,
                    per_chunk_cleanup=per_chunk_cleanup,
                    allow_partial=True,
                    global_output_dir=str(Path(settings.get("_run_dir") or output_dir)),
                    resume_from_partial=False,
                    progress_tracker=chunk_progress_cb,
                    process_func=None,
                    model_type="rife",  # Route to runner.run_rife
                )
                
                status = "âœ… RIFE chunked processing complete" if rc == 0 else f"âš ï¸ RIFE chunking failed (code {rc})"
                if rc != 0 and maybe_set_vram_oom_alert(state, model_label="RIFE", text=clog, settings=settings):
                    state["operation_status"] = "error"
                    status = "ðŸš« Out of VRAM (GPU) â€” see banner above"
                    show_vram_oom_modal(state, title="Out of VRAM (GPU) â€” RIFE", duration=None)
                
                # Build comparison for chunked output
                video_comp_html_update = gr.update(value="", visible=False)
                image_slider_update = gr.update(value=None)
                
                if final_output and Path(final_output).exists():
                    if Path(final_output).suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
                        from shared.video_comparison_slider import create_video_comparison_html
                        video_comp_html_value = create_video_comparison_html(
                            original_video=settings["input_path"],
                            upscaled_video=final_output,
                            height=600,
                            slider_position=50.0
                        )
                        video_comp_html_update = gr.update(value=video_comp_html_value, visible=True)
                    elif not Path(final_output).is_dir():
                        image_slider_update = gr.update(value=(settings["input_path"], final_output), visible=True)
                
                meta_md = f"Chunking ({'Auto scenes' if auto_chunk else 'Static'}): {chunk_count} chunks processed\nOutput: {final_output}"
                
                yield (status, clog, gr.update(value="", visible=False), final_output if final_output and Path(final_output).suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv') else None, image_slider_update, video_comp_html_update, state)
                return

            # Check for batch processing
            if settings.get("batch_enable"):
                # Use the batch processor for multiple files
                from shared.batch_processor import BatchProcessor, BatchJob

                batch_input_path = Path(settings.get("batch_input_path", ""))
                batch_output_path = Path(settings.get("batch_output_path", ""))

                if not batch_input_path.exists():
                    yield ("âŒ Batch input path does not exist", "", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                    return

                # Collect all video files for RIFE
                rife_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
                batch_files = []
                if batch_input_path.is_dir():
                    for ext in rife_exts:
                        batch_files.extend(batch_input_path.glob(f"**/*{ext}"))
                elif batch_input_path.suffix.lower() in rife_exts:
                    batch_files = [batch_input_path]

                if not batch_files:
                    yield ("âŒ No supported video files found in batch input", "", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                    return

                # Create batch processor
                batch_processor = BatchProcessor(
                    output_dir=str(batch_output_path) if batch_output_path.exists() else str(output_dir),
                    max_workers=1,  # Sequential processing for memory management
                    telemetry_enabled=global_settings.get("telemetry", True),
                )

                # Create batch jobs
                jobs = []
                for input_file in sorted(set(batch_files)):
                    job = BatchJob(
                        input_path=str(input_file),
                        metadata={
                            "settings": settings.copy(),
                            "global_settings": global_settings,
                            "face_apply": bool(global_settings.get("face_global", False)),
                            "face_strength": float(global_settings.get("face_strength", 0.5)),
                        }
                    )
                    jobs.append(job)

                # Define processing function for each job (reuses the single-file runner)
                def process_single_rife_job(job: BatchJob) -> bool:
                    try:
                        # Process single file with current settings
                        single_settings = job.metadata["settings"].copy()
                        single_settings["input_path"] = job.input_path
                        single_settings["batch_enable"] = False  # Disable batch for individual processing
                        single_settings["output_override"] = None  # Will be set per-item for batch
                        single_settings["_original_filename"] = Path(job.input_path).name

                        overwrite_existing = bool(seed_controls.get("overwrite_existing_batch_val", False))

                        batch_output_folder = Path(batch_output_path) if batch_output_path.exists() else output_dir
                        batch_output_folder.mkdir(parents=True, exist_ok=True)

                        # Determine desired container extension (best-effort; RIFE runner uses output path suffix)
                        out_ext = str(single_settings.get("output_format") or "mp4")
                        if out_ext == "auto":
                            out_ext = "mp4"

                        from shared.output_run_manager import batch_item_dir, prepare_batch_video_run_dir

                        item_out_dir = batch_item_dir(batch_output_folder, Path(job.input_path).name)
                        png_output_job = bool(single_settings.get("png_output", False))
                        if png_output_job:
                            predicted_final = item_out_dir / f"{Path(job.input_path).stem}_png"
                        else:
                            predicted_final = item_out_dir / f"{Path(job.input_path).stem}.{out_ext.lstrip('.')}"

                        run_paths = prepare_batch_video_run_dir(
                            batch_output_folder,
                            Path(job.input_path).name,
                            input_path=str(job.input_path),
                            model_label="RIFE",
                            mode=str(getattr(runner, "get_mode", lambda: "subprocess")() or "subprocess"),
                            overwrite_existing=overwrite_existing,
                        )
                        if not run_paths:
                            if not overwrite_existing:
                                job.status = "skipped"
                                job.output_path = str(predicted_final)
                                return True
                            job.error_message = f"Could not create batch output folder: {item_out_dir}"
                            return False
                        single_settings["_run_dir"] = str(run_paths.run_dir)
                        single_settings["_processed_chunks_dir"] = str(run_paths.processed_chunks_dir)
                        single_settings["_user_output_override_raw"] = str(job.metadata["settings"].get("output_override") or "")
                        single_settings["output_override"] = str(predicted_final)

                        # Apply universal chunking (Resolution tab) for batch videos, same as single-file runs.
                        from shared.path_utils import detect_input_type as _detect_type
                        should_chunk = (_detect_type(job.input_path) == "video") and (auto_chunk or chunk_size_sec > 0) and (not png_output_job)
                        if should_chunk:
                            from shared.chunking import chunk_and_process

                            rc, clog, final_output, _chunk_count = chunk_and_process(
                                runner=runner,
                                settings=single_settings,
                                scene_threshold=scene_threshold,
                                min_scene_len=min_scene_len,
                                work_dir=Path(single_settings.get("_run_dir") or item_out_dir),
                                on_progress=lambda msg: None,
                                chunk_seconds=0.0 if auto_chunk else chunk_size_sec,
                                chunk_overlap=0.0 if auto_chunk else chunk_overlap_sec,
                                per_chunk_cleanup=per_chunk_cleanup,
                                allow_partial=True,
                                global_output_dir=str(Path(single_settings.get("_run_dir") or item_out_dir)),
                                resume_from_partial=False,
                                progress_tracker=None,
                                process_func=None,
                                model_type="rife",
                            )

                            if final_output and Path(final_output).exists():
                                job.output_path = final_output
                                ok = (rc == 0)
                                if not ok:
                                    job.error_message = clog

                                # Apply face restoration if enabled (on final output)
                                if job.metadata["face_apply"] and Path(job.output_path).exists():
                                    restored = restore_video(
                                        job.output_path,
                                        strength=job.metadata["face_strength"],
                                        on_progress=None
                                    )
                                    if restored:
                                        job.output_path = restored
                                # Face restoration can drop audio; re-mux/strip according to preferences.
                                try:
                                    from shared.audio_utils import ensure_audio_on_video

                                    audio_codec = str(single_settings.get("audio_codec") or "copy")
                                    audio_bitrate = single_settings.get("audio_bitrate") or None
                                    if Path(job.output_path).exists() and Path(job.output_path).is_file():
                                        if audio_codec.strip().lower() in ("none", "no", "off", "disable", "disabled"):
                                            ensure_audio_on_video(
                                                Path(job.output_path),
                                                Path(job.output_path),
                                                audio_codec="none",
                                                audio_bitrate=None,
                                                on_progress=None,
                                            )
                                        elif Path(job.input_path).exists():
                                            ensure_audio_on_video(
                                                Path(job.output_path),
                                                Path(job.input_path),
                                                audio_codec=audio_codec,
                                                audio_bitrate=str(audio_bitrate) if audio_bitrate else None,
                                                on_progress=None,
                                            )
                                except Exception:
                                    pass
                                comp_vid_path, _comp_vid_err = maybe_generate_input_vs_output_comparison(
                                    str(job.input_path),
                                    job.output_path,
                                    seed_controls,
                                    label_output="RIFE",
                                    on_progress=None,
                                )
                                if comp_vid_path:
                                    try:
                                        job.metadata["comparison_video_path"] = str(comp_vid_path)
                                    except Exception:
                                        pass
                                return ok

                            job.error_message = clog
                            return False

                        # If no processing flags are enabled, copy-only mode.
                        should_run_rife_job = _rife_processing_requested(single_settings) or global_rife_forced
                        if not should_run_rife_job:
                            try:
                                dest = Path(normalize_path(str(predicted_final)))
                                if dest.suffix == "":
                                    dest.mkdir(parents=True, exist_ok=True)
                                else:
                                    dest.parent.mkdir(parents=True, exist_ok=True)
                                    shutil.copy2(job.input_path, dest)
                                # Respect "Remove Audio" even in copy-only mode.
                                try:
                                    from shared.audio_utils import ensure_audio_on_video

                                    audio_codec = str(single_settings.get("audio_codec") or "copy")
                                    if audio_codec.strip().lower() in ("none", "no", "off", "disable", "disabled"):
                                        ensure_audio_on_video(dest, dest, audio_codec="none", audio_bitrate=None, on_progress=None)
                                except Exception:
                                    pass
                                job.output_path = str(dest)
                                return True
                            except Exception as e:
                                job.error_message = str(e)
                                return False

                        result = runner.run_rife(single_settings, on_progress=None)

                        if result.output_path and Path(result.output_path).exists():
                            job.output_path = result.output_path
                            ok = True

                            # Apply face restoration if enabled
                            if job.metadata["face_apply"] and Path(job.output_path).exists():
                                restored = restore_video(
                                    job.output_path,
                                    strength=job.metadata["face_strength"],
                                    on_progress=None
                                )
                                if restored:
                                    job.output_path = restored
                            # Ensure audio is correct (RIFE/face-restore can produce video-only outputs).
                            try:
                                from shared.audio_utils import ensure_audio_on_video

                                audio_codec = str(single_settings.get("audio_codec") or "copy")
                                audio_bitrate = single_settings.get("audio_bitrate") or None
                                if Path(job.output_path).exists() and Path(job.output_path).is_file():
                                    if audio_codec.strip().lower() in ("none", "no", "off", "disable", "disabled"):
                                        ensure_audio_on_video(
                                            Path(job.output_path),
                                            Path(job.output_path),
                                            audio_codec="none",
                                            audio_bitrate=None,
                                            on_progress=None,
                                        )
                                    elif Path(job.input_path).exists():
                                        ensure_audio_on_video(
                                            Path(job.output_path),
                                            Path(job.input_path),
                                            audio_codec=audio_codec,
                                            audio_bitrate=str(audio_bitrate) if audio_bitrate else None,
                                            on_progress=None,
                                        )
                            except Exception:
                                pass
                            comp_vid_path, _comp_vid_err = maybe_generate_input_vs_output_comparison(
                                str(job.input_path),
                                job.output_path,
                                seed_controls,
                                label_output="RIFE",
                                on_progress=None,
                            )
                            if comp_vid_path:
                                try:
                                    job.metadata["comparison_video_path"] = str(comp_vid_path)
                                except Exception:
                                    pass
                        else:
                            job.error_message = result.log
                            ok = False

                        return ok
                    except Exception as e:
                        job.error_message = str(e)
                        return False

                # Run batch processing
                batch_result = batch_processor.process_batch(
                    jobs=jobs,
                    processor_func=process_single_rife_job,
                    max_concurrent=1,
                )

                # Summarize results
                completed = int(batch_result.completed_files)
                failed = int(batch_result.failed_files)

                summary_msg = f"RIFE batch complete: {completed}/{len(jobs)} succeeded"
                if failed > 0:
                    summary_msg += f", {failed} failed"

                log_lines = [summary_msg]
                if failed:
                    for j in [x for x in jobs if x.status == "failed"][:10]:
                        name = Path(j.input_path).name
                        err = (j.error_message or "").strip()
                        err = (err[:180] + "â€¦") if len(err) > 180 else err
                        log_lines.append(f"âŒ {name}: {err}" if err else f"âŒ {name}")

                yield (f"âœ… {summary_msg}", "\n".join(log_lines), gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                return

            # Single file processing with streaming updates
            processing_complete = False
            last_progress_update = 0

            def progress_callback(message: str):
                nonlocal last_progress_update
                current_time = time.time()
                # Throttle updates to every 0.5 seconds to avoid UI spam
                if current_time - last_progress_update > 0.5:
                    last_progress_update = current_time
                    yield (f"âš™ï¸ Processing: {message}", f"Progress: {message}", gr.update(value=message, visible=True), None, gr.update(value=None), gr.update(value="", visible=False), state)

            # Start processing with progress tracking
            yield ("âš™ï¸ Starting processing...", "Initializing...", gr.update(value="Initializing...", visible=True), None, gr.update(value=None), gr.update(value="", visible=False), state)

            # Determine processing workflow
            edit_mode = settings.get("edit_mode", "none")
            should_run_rife = _rife_processing_requested(settings) or global_rife_forced

            current_input = settings["input_path"]
            final_output_path = None

            # Step 1: Apply video editing (if any)
            if edit_mode != "none":
                yield ("âš™ï¸ Applying video editing...", "Processing video edits...", gr.update(value="Video editing in progress...", visible=True), None, gr.update(value=None), gr.update(value="", visible=False), state)

                edit_temp_output = temp_dir / f"edit_temp_{Path(current_input).stem}_{int(time.time())}.mp4"
                edit_success, edit_log, edited_path = _apply_video_editing(
                    current_input, str(edit_temp_output), settings, temp_dir,
                    lambda msg: progress_callback(f"Editing: {msg}")
                )

                if not edit_success:
                    yield (f"âŒ Video editing failed: {edit_log}", "Edit failed", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                    return

                current_input = edited_path
                yield ("âœ… Video editing completed", "Edit completed successfully", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)

            # Step 2: Apply RIFE processing (if enabled)
            if should_run_rife:
                # Update settings to use the edited video as input
                rife_settings = settings.copy()
                rife_settings["input_path"] = current_input

                yield ("âš™ï¸ Running RIFE frame interpolation...", "Starting RIFE processing...", gr.update(value="RIFE processing...", visible=True), None, gr.update(value=None), gr.update(value="", visible=False), state)

                # Create a queue for progress updates
                progress_queue = queue.Queue()

                def processing_thread():
                    try:
                        result = runner.run_rife(rife_settings, on_progress=lambda msg: progress_queue.put(("progress", msg)))
                        progress_queue.put(("complete", result))
                    except Exception as e:
                        progress_queue.put(("error", str(e)))

                # Start processing in background thread
                proc_thread = threading.Thread(target=processing_thread, daemon=True)
                proc_thread.start()

                # Stream progress updates
                while proc_thread.is_alive() or not progress_queue.empty():
                    try:
                        update_type, data = progress_queue.get(timeout=0.1)
                        if update_type == "progress":
                            yield (f"âš™ï¸ RIFE Processing: {data}", f"Progress: {data}", gr.update(value=data, visible=True), None, gr.update(value=None), gr.update(value="", visible=False), state)
                        elif update_type == "complete":
                            result = data
                            processing_complete = True
                            break
                        elif update_type == "error":
                            if maybe_set_vram_oom_alert(state, model_label="RIFE", text=data, settings=rife_settings):
                                state["operation_status"] = "error"
                                show_vram_oom_modal(state, title="Out of VRAM (GPU) â€” RIFE", duration=None)
                                yield ("ðŸš« Out of VRAM (GPU) â€” see banner above", f"Error: {data}", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                            else:
                                yield ("âŒ RIFE processing failed", f"Error: {data}", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                            return
                    except queue.Empty:
                        continue

                if not processing_complete:
                    yield ("âŒ Processing timed out", "RIFE processing did not complete within expected time", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                    return

                status = "âœ… RIFE complete" if result.returncode == 0 else f"âš ï¸ RIFE exited with code {result.returncode}"
                if result.returncode != 0 and maybe_set_vram_oom_alert(state, model_label="RIFE", text=result.log, settings=rife_settings):
                    state["operation_status"] = "error"
                    status = "ðŸš« Out of VRAM (GPU) â€” see banner above"
                    show_vram_oom_modal(state, title="Out of VRAM (GPU) â€” RIFE", duration=None)
                final_output_path = result.output_path
            else:
                # No RIFE processing: copy/move into the run output location so the user always
                # gets a file in the output folder (and we avoid touching the original input).
                final_output_path = current_input
                status = "âœ… Processing complete" if edit_mode != "none" else "âœ… File copied (no processing)"
                try:
                    dest_override = (settings.get("output_override") or "").strip()
                    if dest_override:
                        from shared.path_utils import collision_safe_path

                        src_p = Path(final_output_path)
                        dest_p = Path(normalize_path(dest_override))
                        if dest_p.suffix == "":
                            dest_p = dest_p / src_p.name
                        dest_p = collision_safe_path(dest_p)
                        dest_p.parent.mkdir(parents=True, exist_ok=True)

                        try:
                            same = src_p.resolve() == dest_p.resolve()
                        except Exception:
                            same = str(src_p) == str(dest_p)

                        if (not same) and src_p.exists():
                            # Move edited temp outputs; copy original inputs.
                            orig_raw = str(settings.get("input_path", "") or "").strip()
                            orig_p = Path(normalize_path(orig_raw)) if orig_raw else None

                            is_original_input = False
                            if orig_p:
                                try:
                                    is_original_input = src_p.resolve() == orig_p.resolve()
                                except Exception:
                                    is_original_input = str(src_p) == str(orig_p)

                            if is_original_input:
                                shutil.copy2(src_p, dest_p)
                            else:
                                shutil.move(str(src_p), str(dest_p))
                            final_output_path = str(dest_p)
                except Exception:
                    pass

            # Pick a suitable audio source for post-processing (face-restore can drop audio).
            audio_source_for_mux = current_input if should_run_rife else final_output_path

            # Apply face restoration if enabled
            face_apply = bool(global_settings.get("face_global", False))
            if face_apply and final_output_path and Path(final_output_path).exists():
                yield ("âš™ï¸ Applying face restoration...", "Face restoration in progress...", gr.update(value="Face restoration...", visible=True), None, gr.update(value=None), gr.update(value="", visible=False), state)
                pre_face_output = str(final_output_path)
                face_strength = float(global_settings.get("face_strength", 0.5))
                restored = restore_video(final_output_path, strength=face_strength,
                                       on_progress=lambda x: progress_callback(f"Face restoration: {x}"))
                if restored:
                    final_output_path = restored
                    # Prefer audio from the pre-face output only if it actually contains audio;
                    # otherwise keep the previously chosen source (often the original/edited input).
                    try:
                        from shared.audio_utils import has_audio_stream

                        if Path(pre_face_output).exists() and has_audio_stream(Path(pre_face_output)):
                            audio_source_for_mux = pre_face_output
                    except Exception:
                        pass
                    yield ("âœ… Face restoration completed", "Face restoration done", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)

            # Audio normalization (best-effort): ensure output has (or does not have) audio per Output tab.
            try:
                from shared.audio_utils import ensure_audio_on_video

                if final_output_path and Path(final_output_path).exists() and Path(final_output_path).is_file():
                    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}
                    if Path(final_output_path).suffix.lower() in video_exts:
                        audio_codec = str(settings.get("audio_codec") or "copy")
                        audio_bitrate = settings.get("audio_bitrate") or None
                        if audio_codec.strip().lower() in ("none", "no", "off", "disable", "disabled"):
                            _changed, _final, _err = ensure_audio_on_video(
                                Path(final_output_path),
                                Path(final_output_path),
                                audio_codec="none",
                                audio_bitrate=None,
                                on_progress=None,
                            )
                        else:
                            src = audio_source_for_mux or settings.get("input_path")
                            if src and Path(src).exists():
                                _changed, _final, _err = ensure_audio_on_video(
                                    Path(final_output_path),
                                    Path(src),
                                    audio_codec=audio_codec,
                                    audio_bitrate=str(audio_bitrate) if audio_bitrate else None,
                                    on_progress=None,
                                )
                            else:
                                _changed, _final, _err = (False, None, "")
                        if _final and str(_final) != str(final_output_path):
                            final_output_path = str(_final)
            except Exception:
                pass

            comp_vid_path, comp_vid_err = maybe_generate_input_vs_output_comparison(
                settings.get("_original_input_path_before_preprocess") or settings.get("input_path"),
                final_output_path,
                seed_controls,
                label_output="RIFE",
                on_progress=None,
            )
            comp_vid_note = ""
            if comp_vid_path:
                comp_vid_note = f"\nComparison Video: {comp_vid_path}"
            elif comp_vid_err:
                progress_callback(f"Comparison video failed: {comp_vid_err}")

            # Create metadata string
            processing_steps = []
            if edit_mode != "none":
                processing_steps.append(f"Edit: {edit_mode}")
            if global_rife_forced:
                processing_steps.append("Global RIFE override")
            if should_run_rife:
                processing_steps.append(f"RIFE: {settings.get('fps_multiplier')}x")
            if face_apply:
                processing_steps.append(f"Face: {face_strength}")

            steps_str = ", ".join(processing_steps) if processing_steps else "None"

            meta_md = f"Input: {settings['input_path']}\nOutput: {final_output_path}\nProcessing: {steps_str}{comp_vid_note}"

            # Log the run and update state with output path
            if final_output_path:
                # Track output path for pinned comparison feature
                try:
                    outp = Path(final_output_path)
                    seed_controls["last_output_dir"] = str(outp.parent if outp.is_file() else outp)
                    seed_controls["last_output_path"] = str(outp) if outp.is_file() else None
                except Exception:
                    pass
                
                run_logger.write_summary(
                    Path(final_output_path),
                    {
                        "input": settings["input_path"],
                        "output": final_output_path,
                        "returncode": (result.returncode if should_run_rife and "result" in locals() else 0),
                        "args": settings,
                        "edit_mode": edit_mode,
                        "rife_enabled": bool(should_run_rife),
                        "face_restoration": face_apply,
                    },
                )

            # Build comparison outputs (match UI expectations: 5 outputs total)
            comparison_mode = seed_controls.get("comparison_mode_val", "native")
            image_slider_update = gr.update(value=None)
            video_comparison_html_update = gr.update(value="", visible=False)
            
            if final_output_path and Path(final_output_path).exists():
                original_input = settings["input_path"]
                
                # Video comparison with custom HTML slider
                if Path(final_output_path).suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
                    if Path(original_input).exists():
                        from shared.video_comparison_slider import create_video_comparison_html
                        
                        video_comp_html_value = create_video_comparison_html(
                            original_video=original_input,
                            upscaled_video=final_output_path,
                            height=600,
                            slider_position=50.0
                        )
                        video_comparison_html_update = gr.update(value=video_comp_html_value, visible=True)
                
                # Image comparison with ImageSlider (for single-frame outputs or image mode)
                elif Path(final_output_path).suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    if Path(original_input).exists():
                        image_slider_update = gr.update(
                            value=(original_input, final_output_path),
                            visible=True
                        )

            state["operation_status"] = "completed" if "âœ…" in status else "ready"
            
            # Return 7 outputs to match UI expectations: status, log, progress_indicator, output_video, image_slider, video_comparison_html, state
            yield (
                status,
                result.log if should_run_rife and "result" in locals() else "",
                gr.update(value="", visible=False),  # progress_indicator (clear on completion)
                final_output_path if final_output_path and Path(final_output_path).suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv') else None,
                image_slider_update,
                video_comparison_html_update,
                state
            )

        except Exception as e:
            error_msg = f"Critical error in RIFE processing: {str(e)}"
            state = state or {}
            state["operation_status"] = "error"
            if maybe_set_vram_oom_alert(state, model_label="RIFE", text=str(e), settings=locals().get("settings")):
                show_vram_oom_modal(state, title="Out of VRAM (GPU) â€” RIFE", duration=None)
            yield ("âŒ Critical error", error_msg, gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)


    def cancel():
        """Cancel current processing and compile partial outputs if available."""
        canceled = runner.cancel()
        if not canceled:
            return gr.update(value="No active process to cancel"), ""

        compiled_output: Optional[str] = None
        live_output_root = Path(global_settings.get("output_dir", output_dir))
        state_snapshot = {}
        try:
            state_snapshot = shared_state.value if isinstance(shared_state.value, dict) else {}
        except Exception:
            state_snapshot = {}
        seed_controls = state_snapshot.get("seed_controls", {}) if isinstance(state_snapshot, dict) else {}
        last_run_dir = seed_controls.get("last_run_dir")
        audio_source = seed_controls.get("last_input_path") or None
        audio_codec = str(seed_controls.get("audio_codec_val") or "copy")
        audio_bitrate = seed_controls.get("audio_bitrate_val") or None
        output_settings = seed_controls.get("output_settings", {}) if isinstance(seed_controls, dict) else {}
        if not isinstance(output_settings, dict):
            output_settings = {}

        try:
            from shared.chunking import salvage_partial_from_run_dir
            from shared.output_run_manager import recent_output_run_dirs

            for run_dir in recent_output_run_dirs(
                live_output_root,
                last_run_dir=str(last_run_dir) if last_run_dir else None,
                limit=20,
            ):
                partial_path, _method = salvage_partial_from_run_dir(
                    run_dir,
                    partial_basename="cancelled_rife_partial",
                    audio_source=str(audio_source) if audio_source else None,
                    audio_codec=audio_codec,
                    audio_bitrate=str(audio_bitrate) if audio_bitrate else None,
                    encode_settings=output_settings,
                )
                if partial_path and Path(partial_path).exists():
                    compiled_output = str(partial_path)
                    break
        except Exception as e:
            return (
                gr.update(value=f"Cancelled - Error salvaging partials: {str(e)}"),
                "Processing cancelled but partial compilation failed",
            )

        if not compiled_output:
            try:
                temp_chunks_dir = Path(global_settings.get("temp_dir", temp_dir)) / "chunks"
                if temp_chunks_dir.exists():
                    from shared.chunking import salvage_partial_from_run_dir

                    partial_path, _method = salvage_partial_from_run_dir(
                        temp_chunks_dir,
                        partial_basename="cancelled_rife_partial",
                        encode_settings=output_settings,
                    )
                    if partial_path and Path(partial_path).exists():
                        compiled_output = str(partial_path)
            except Exception:
                pass

        if compiled_output:
            return (
                gr.update(value=f"Cancelled - Partial RIFE output saved: {Path(compiled_output).name}"),
                f"Partial results salvaged and saved to: {compiled_output}",
            )
        return gr.update(value="Cancelled - No partial outputs found"), "Processing cancelled - no partial outputs found to salvage"

    def open_outputs_folder_rife():
        """Open outputs folder - delegates to shared utility (no code duplication)"""
        from shared.services.global_service import open_outputs_folder
        return open_outputs_folder(str(output_dir))
    
    def clear_temp_folder_rife(confirm: bool):
        """Clear temp folder - delegates to shared utility (no code duplication)"""
        from shared.services.global_service import clear_temp_folder
        return clear_temp_folder(str(temp_dir), confirm)

    return {
        "defaults": defaults,
        "order": RIFE_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "run_action": run_action,
        "cancel_action": cancel,
        "open_outputs_folder": open_outputs_folder_rife,
        "clear_temp_folder": clear_temp_folder_rife,
    }
