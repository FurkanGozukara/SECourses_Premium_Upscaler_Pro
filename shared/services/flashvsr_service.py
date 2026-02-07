"""
FlashVSR+ Service Module
Handles FlashVSR+ processing logic, presets, and callbacks
"""

import queue
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
    detect_input_type,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from shared.logging_utils import RunLogger
from shared.comparison_unified import create_unified_comparison
from shared.models.flashvsr_meta import get_flashvsr_metadata, get_flashvsr_default_model
from shared.gpu_utils import expand_cuda_device_spec, validate_cuda_device_spec
from shared.error_handling import logger as error_logger
from shared.resolution_calculator import estimate_fixed_scale_upscale_plan_from_dims
from shared.oom_alert import clear_vram_oom_alert, maybe_set_vram_oom_alert, show_vram_oom_modal
from shared.output_run_manager import prepare_single_video_run, batch_item_dir, downscaled_video_path
from shared.ffmpeg_utils import scale_video
from shared.global_rife import maybe_apply_global_rife
from shared.comparison_video_service import maybe_generate_input_vs_output_comparison
from shared.chunk_preview import build_chunk_preview_payload

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


def flashvsr_defaults(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get default FlashVSR+ settings aligned with CLI defaults.
    Applies model-specific metadata when model_name is provided.
    """
    # IMPORTANT: do not import torch in the parent Gradio process.
    # Use NVML-based detection (nvidia-smi) via shared.gpu_utils instead.
    try:
        from shared.gpu_utils import get_gpu_info
        cuda_default = "auto" if get_gpu_info() else "cpu"
    except Exception:
        cuda_default = "cpu"
    
    # Get model metadata if specific model is provided
    default_model = model_name or get_flashvsr_default_model()
    model_meta = get_flashvsr_metadata(default_model)
    
    # Apply model-specific defaults if metadata available
    if model_meta:
        default_dtype = model_meta.default_dtype
        default_tile_size = model_meta.default_tile_size
        default_overlap = model_meta.default_overlap
        default_attention = model_meta.default_attention
        version = model_meta.version
        mode = model_meta.mode
        scale = model_meta.scale
    else:
        default_dtype = "bf16"
        default_tile_size = 256
        default_overlap = 24
        default_attention = "sage"
        version = "10"
        mode = "tiny"
        scale = 4
    
    return {
        "input_path": "",
        "output_override": "",
        "scale": scale,
        "version": version,
        "mode": mode,
        "tiled_vae": False,
        "tiled_dit": False,
        "tile_size": default_tile_size,
        "overlap": default_overlap,
        "unload_dit": False,
        "color_fix": True,
        "seed": 0,
        "dtype": default_dtype,
        "device": cuda_default,
        "fps": 30,
        "quality": 6,
        "attention": default_attention,
        "batch_enable": False,
        "batch_input_path": "",
        "batch_output_path": "",
        # vNext sizing (app-level)
        "use_resolution_tab": True,
        "upscale_factor": float(scale),
        "max_target_resolution": 0,  # Max edge cap (0 = no cap)
        "pre_downscale_then_upscale": False,
    }


FLASHVSR_ORDER: List[str] = [
    "input_path",
    "output_override",
    "scale",
    "version",
    "mode",
    "tiled_vae",
    "tiled_dit",
    "tile_size",
    "overlap",
    "unload_dit",
    "color_fix",
    "seed",
    "dtype",
    "device",
    "fps",
    "quality",
    "attention",
    "batch_enable",
    "batch_input_path",
    "batch_output_path",
    # vNext sizing
    "use_resolution_tab",
    "upscale_factor",
    "max_target_resolution",
    "pre_downscale_then_upscale",
]


def _flashvsr_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    return dict(zip(FLASHVSR_ORDER, args))


def _enforce_flashvsr_guardrails(cfg: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply FlashVSR+-specific validation rules using metadata registry.
    
    Enforces:
    - Single GPU requirement (FlashVSR+ doesn't support multi-GPU well)
    - Valid CUDA device IDs (checks against available devices)
    - VRAM-based tile size recommendations
    - Resolution constraints from model metadata
    - Valid version/mode/scale combinations
    """
    cfg = cfg.copy()
    # Dropdown-safe normalization to avoid Gradio choice mismatches.
    try:
        cfg["scale"] = "2" if str(cfg.get("scale", defaults.get("scale", "4"))).strip() == "2" else "4"
    except Exception:
        cfg["scale"] = "4"
    cfg["version"] = str(cfg.get("version", defaults.get("version", "10")) or "10")
    cfg["mode"] = str(cfg.get("mode", defaults.get("mode", "tiny")) or "tiny")
    
    # Build model identifier and get metadata
    model_id = f"v{cfg.get('version', '10')}_{cfg.get('mode', 'tiny')}_{cfg.get('scale', '4')}x"
    model_meta = get_flashvsr_metadata(model_id)
    
    if model_meta:
        # Enforce single GPU (FlashVSR+ doesn't support multi-GPU)
        device_str = str(cfg.get("device", "auto"))
        if device_str not in ("auto", "cpu", ""):
            # Parse device specification
            devices = [d.strip() for d in device_str.replace(" ", "").split(",") if d.strip()]
            if len(devices) > 1:
                error_logger.warning(f"FlashVSR+ doesn't support multi-GPU - forcing single GPU (using first: {devices[0]})")
                cfg["device"] = devices[0]
                cfg["_multi_gpu_disabled_reason"] = "FlashVSR+ is single-GPU optimized"
            
            # Validate CUDA device ID is actually available
            try:
                from shared.gpu_utils import get_gpu_info

                device_count = len(get_gpu_info())
                device_id_str = str(cfg.get("device", "")).replace("cuda:", "").strip()
                if device_id_str.isdigit():
                    device_id = int(device_id_str)
                    if device_count == 0 or device_id >= device_count:
                        error_logger.warning(
                            f"Device ID {device_id} not available (detected {device_count} GPU(s)) - falling back to auto"
                        )
                        cfg["device"] = "auto"
                        cfg["_device_validation_warning"] = f"Requested GPU {device_id} not found, using auto-select"
            except Exception as e:
                error_logger.warning(f"Failed to validate device ID: {e}")
        
        # Apply model-specific defaults if not set
        if not cfg.get("dtype"):
            cfg["dtype"] = model_meta.default_dtype
        
        if cfg.get("tile_size", 0) == 0:
            cfg["tile_size"] = model_meta.default_tile_size
        
        if cfg.get("overlap", 0) == 0:
            cfg["overlap"] = model_meta.default_overlap
        
        if not cfg.get("attention"):
            cfg["attention"] = model_meta.default_attention
        
        # Validate tile overlap constraint
        if cfg.get("tiled_vae") or cfg.get("tiled_dit"):
            tile_size = int(cfg.get("tile_size", model_meta.default_tile_size))
            overlap = int(cfg.get("overlap", model_meta.default_overlap))
            
            if overlap >= tile_size:
                cfg["overlap"] = max(0, tile_size - 1)
                error_logger.warning(f"Tile overlap ({overlap}) >= tile size ({tile_size}), correcting to {cfg['overlap']}")
            
            if tile_size < 64:
                cfg["tile_size"] = model_meta.default_tile_size
                error_logger.warning(f"Tile size too small, resetting to {cfg['tile_size']}")
    
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
            model_name = f"v{payload['version']}_{payload['mode']}"
            
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
            model_name = f"v{version}_{mode}"
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
            seed_controls["flashvsr_chunk_preview"] = {
                "message": "No chunk preview available yet.",
                "gallery": [],
                "videos": [],
                "count": 0,
            }
            state["seed_controls"] = seed_controls
            settings_dict = _flashvsr_dict_from_args(list(args))
            settings = {**defaults, **settings_dict}
            
            # Apply FlashVSR+ guardrails (single GPU, tile validation, etc.)
            settings = _enforce_flashvsr_guardrails(settings, defaults)
            use_global_resolution = bool(settings.get("use_resolution_tab", True))
            
            # Apply shared Resolution & Scene Split tab settings (vNext Upscale-x)
            if use_global_resolution:
                if seed_controls.get("upscale_factor_val") is not None:
                    try:
                        settings["upscale_factor"] = float(seed_controls["upscale_factor_val"])
                    except Exception:
                        pass
                if seed_controls.get("max_resolution_val") is not None:
                    settings["max_target_resolution"] = int(seed_controls["max_resolution_val"] or 0)
                # Repurposed global flag: "pre-downscale then upscale when capped"
                if "ratio_downscale" in seed_controls:
                    settings["pre_downscale_then_upscale"] = bool(seed_controls.get("ratio_downscale", False))
            
            # Apply Output tab cached settings
            if seed_controls.get("fps_override_val") is not None and seed_controls["fps_override_val"] > 0:
                settings["fps"] = seed_controls["fps_override_val"]
            if seed_controls.get("comparison_mode_val"):
                settings["_comparison_mode"] = seed_controls["comparison_mode_val"]
            if seed_controls.get("save_metadata_val") is not None:
                settings["save_metadata"] = seed_controls["save_metadata_val"]
            # Audio mux preferences (used by chunking + final output postprocessing)
            if seed_controls.get("audio_codec_val") is not None:
                settings["audio_codec"] = seed_controls.get("audio_codec_val") or "copy"
            if seed_controls.get("audio_bitrate_val") is not None:
                settings["audio_bitrate"] = seed_controls.get("audio_bitrate_val") or ""
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
            
            # Clear cancel event
            _flashvsr_cancel_event.clear()

            def _apply_vnext_preprocess(cfg: Dict[str, Any], src_input_path: str) -> None:
                """
                vNext preprocessing for fixed-scale FlashVSR models (2x/4x).

                If the requested upscale (and/or max-edge cap) implies an effective scale < model_scale,
                pre-downscale the input so the model still runs at its native scale without post-downscale.

                Supports:
                - Video files (ffmpeg scale)
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

                    # Respect global enable_max_target when using Resolution tab
                    if use_global_resolution and not bool(seed_controls.get("enable_max_target", True)):
                        max_edge = 0

                    plan = estimate_fixed_scale_upscale_plan_from_dims(
                        int(w),
                        int(h),
                        requested_scale=requested_scale,
                        model_scale=model_scale,
                        max_edge=max_edge,
                        force_pre_downscale=True,
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
                        return
                except Exception:
                    return
            
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
                            return gr.update(value=out_path, visible=True), gr.update(value=None, visible=False)
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
                batch_out = normalize_path(settings.get("batch_output_path") or "") if settings.get("batch_output_path") else ""

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

                if batch_out:
                    try:
                        Path(batch_out).mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass

                in_dir = Path(batch_in)
                items: List[Path] = [
                    p
                    for p in sorted(in_dir.iterdir())
                    if (p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS) or p.is_dir()
                ]
                if not items:
                    # If the batch folder itself is a frame directory, treat it as a single job.
                    items = [in_dir]

                face_apply = bool(global_settings.get("face_global", False))
                face_strength = float(global_settings.get("face_strength", 0.5))

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
                batch_root = Path(batch_out) if batch_out else Path(global_settings.get("output_dir", output_dir))
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
            settings["_original_filename"] = Path(input_path).name

            # NEW: Per-run output folder for videos (0001/0002/...) to avoid collisions and
            # to keep chunk artifacts user-visible.
            if detect_input_type(input_path) == "video":
                try:
                    base_out_root = Path(global_settings.get("output_dir", output_dir))
                    run_paths, explicit_final = prepare_single_video_run(
                        output_root_fallback=base_out_root,
                        output_override_raw=settings.get("output_override"),
                        input_path=input_path,
                        original_filename=settings.get("_original_filename") or Path(input_path).name,
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

                        def _process_chunk(s: Dict[str, Any], on_progress=None) -> RunResult:
                            r = run_flashvsr(
                                s,
                                base_dir,
                                on_progress=on_progress,
                                cancel_event=_flashvsr_cancel_event,
                                process_handle=None,
                            )
                            return RunResult(r.returncode, r.output_path, r.log)

                        def _chunk_progress_cb(progress_val, desc=""):
                            try:
                                pct = int(float(progress_val) * 100)
                                progress_queue.put(f"{pct}% {desc}".strip())
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
                            resume_from_partial=False,
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
            
            # Apply face restoration if globally enabled
            face_apply = global_settings.get("face_global", False)
            face_strength = global_settings.get("face_strength", 0.5)
            
            # Stream progress updates
            last_update = time.time()
            log_buffer = []
            
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
                            from shared.path_utils import collision_safe_path
                            import shutil
                            
                            # Check for completed video chunks
                            partial_video, completed_chunks = detect_resume_state(temp_chunks_dir, "mp4")
                            
                            if completed_chunks and len(completed_chunks) > 0:
                                partial_target = collision_safe_path(temp_chunks_dir / "cancelled_flashvsr_partial.mp4")
                                if concat_videos(completed_chunks, partial_target, encode_settings=settings):
                                    final_output = Path(output_dir) / f"cancelled_flashvsr_partial_upscaled.mp4"
                                    final_output = collision_safe_path(final_output)
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
                    log_buffer.append(msg)
                    
                    # Update gr.Progress from messages
                    if progress and "%" in msg:
                        import re
                        match = re.search(r'(\d+)%', msg)
                        if match:
                            pct = int(match.group(1)) / 100.0
                            progress(pct, desc=msg[:100])
                    
                except queue.Empty:
                    pass
                
                # Yield updates every 0.5s
                now = time.time()
                if now - last_update > 0.5:
                    last_update = now
                    vid_upd, img_upd = _media_updates(None)
                    yield (
                        "⚙️ Processing with FlashVSR+...",
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
            
            # Apply face restoration if globally enabled
            output_path = result.output_path
            face_apply = global_settings.get("face_global", False)
            face_strength = global_settings.get("face_strength", 0.5)
            
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
                        log_buffer.append(f"âš ï¸ Audio mux: {_err}")
                    if _final and str(_final) != str(output_path):
                        output_path = str(_final)
                except Exception as e:
                    log_buffer.append(f"âš ï¸ Audio mux failed: {str(e)}")

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
            if result.returncode != 0 and maybe_set_vram_oom_alert(state, model_label="FlashVSR+", text=result.log, settings=settings):
                status = "🚫 Out of VRAM (GPU) — see banner above"
                show_vram_oom_modal(state, title="Out of VRAM (GPU) — FlashVSR+", duration=None)
            
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
