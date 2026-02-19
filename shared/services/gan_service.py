import queue
import shutil
import subprocess
import threading
import time
import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
import gradio as gr

from shared.preset_manager import PresetManager
from shared.path_utils import (
    normalize_path,
    collision_safe_dir,
    collision_safe_path,
    ffmpeg_set_fps,
    get_media_dimensions,
    detect_input_type,
    IMAGE_EXTENSIONS,
    resolve_batch_output_dir,
)
from shared.resolution_calculator import estimate_fixed_scale_upscale_plan_from_dims
from shared.face_restore import restore_image, restore_video
from shared.logging_utils import RunLogger
from shared.output_run_manager import (
    prepare_single_video_run,
    downscaled_video_path,
    resolve_resume_input_from_run_dir,
    ensure_image_input_artifact,
    finalize_run_context,
)
from shared.realesrgan_runner import run_realesrgan
from shared.gan_runner import run_gan_upscale, GanResult, get_gan_model_metadata
from shared.ffmpeg_utils import scale_video
from shared.comparison_unified import create_unified_comparison, create_video_comparison_slider
from shared.video_comparison_slider import create_video_comparison_html
from shared.gpu_utils import expand_cuda_device_spec, get_global_gpu_override, validate_cuda_device_spec
from shared.oom_alert import clear_vram_oom_alert, maybe_set_vram_oom_alert, show_vram_oom_modal
from shared.global_rife import maybe_apply_global_rife
from shared.comparison_video_service import maybe_generate_input_vs_output_comparison
from shared.chunk_preview import build_chunk_preview_payload
from shared.preview_utils import prepare_preview_input


GAN_MODEL_EXTS = {".pth", ".safetensors"}
GAN_META_CACHE: Dict[str, Dict[str, Any]] = {}
PREFERRED_GAN_DEFAULT_MODEL = "4x-UltraSharpV2.safetensors"


def _normalize_key(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _parse_scale_from_name(name: str) -> int:
    """Legacy fallback scale detection - now superseded by metadata system"""
    import re
    lowered = name.lower()
    m = re.search(r"(\d+)x", lowered)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return 4


def _calculate_input_resolution_for_target(*args, **kwargs) -> Tuple[int, int]:
    """
    LEGACY (deprecated): kept to avoid breaking imports from older code paths.

    vNext sizing is handled via `estimate_fixed_scale_upscale_plan_from_dims()`.
    """
    try:
        input_dims = args[0] if args else None
        return input_dims if input_dims else (0, 0)
    except Exception:
        return (0, 0)


def _load_gan_catalog(base_dir: Path):
    if GAN_META_CACHE:
        return
    data_dir = base_dir / "open-model-database" / "data" / "models"
    if not data_dir.exists():
        return
    for jf in data_dir.glob("*.json"):
        try:
            import json

            with jf.open("r", encoding="utf-8") as f:
                data = json.load(f)
            name = data.get("name") or jf.stem
            scale = data.get("scale") or _parse_scale_from_name(name)
            GAN_META_CACHE[_normalize_key(name)] = {"name": name, "scale": scale}
        except Exception:
            continue


def _get_gan_meta(filename: str, base_dir: Path) -> Dict[str, Any]:
    """Legacy metadata function - now uses comprehensive registry"""
    from shared.gan_runner import get_gan_model_metadata
    metadata = get_gan_model_metadata(filename, base_dir)
    return {
        "scale": metadata.scale,
        "canonical": metadata.name,
        "supports_multi_gpu": False,
        "architecture": metadata.architecture,
        "description": metadata.description,
        "author": metadata.author,
        "tags": metadata.tags
    }


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


def _is_realesrgan_builtin(name: str) -> bool:
    key = _normalize_key(name)
    builtins = [
        "realesrganx4plus",
        "realesrganx4plusanime6b",
        "realesrnetx4plus",
        "realesrganx2plus",
        "realesranimevideov3",
        "realesrgeneralx4v3",
    ]
    return key in builtins


def _scan_gan_models(base_dir: Path) -> List[str]:
    """Scan for GAN models with comprehensive metadata"""
    # Support both the legacy folder (`Image_Upscale_Models/`) and the current layout (`models/`).
    models: set[str] = set()
    for folder_name in ("models", "Image_Upscale_Models"):
        models_dir = base_dir / folder_name
        if not models_dir.exists():
            continue
        try:
            for f in models_dir.iterdir():
                if f.is_file() and f.suffix.lower() in GAN_MODEL_EXTS:
                    models.add(f.name)
        except Exception:
            continue
    
    # Trigger cache reload for metadata
    from shared.gan_runner import reload_gan_models_cache
    reload_gan_models_cache(base_dir)
    
    return sorted(models)


def gan_defaults(base_dir: Path) -> Dict[str, Any]:
    models = _scan_gan_models(base_dir)
    preferred_lookup = {str(m).lower(): str(m) for m in models}
    default_model = preferred_lookup.get(PREFERRED_GAN_DEFAULT_MODEL.lower(), models[0] if models else "")

    return {
        "input_path": "",
        "output_override": "",
        "batch_enable": False,
        "batch_input_path": "",
        "batch_output_path": "",
        "model": default_model,
        "use_resolution_tab": True,  # Enable Resolution tab integration by default
        # NEW (vNext): unified Upscale-x sizing (applies to both images and videos)
        "upscale_factor": 4.0,
        "max_resolution": 1920,  # Max edge cap (0 = no cap)
        "pre_downscale_then_upscale": True,
        "tile_size": 0,
        "overlap": 32,
        "denoising_strength": 0.0,
        "sharpening": 0.0,
        "color_correction": True,
        "gpu_acceleration": True,
        "gpu_device": "0",
        "batch_size": 1,
        "output_format": "png",
        "output_quality": 100,
        "save_metadata": True,
        "fps_override": 0,
        "face_restore_after_upscale": False,
        "create_subfolders": False,
        "resume_run_dir": "",
    }


"""
 GAN PRESET ORDER

MUST match inputs_list order in ui/gan_tab.py.

 TO ADD NEW CONTROLS:
1. Add default to gan_defaults()
2. Append key to GAN_ORDER below
3. Add component to ui/gan_tab.py inputs_list
4. Old presets auto-merge (no migration needed)
"""

GAN_ORDER: List[str] = [
    "input_path",
    "output_override",
    "batch_enable",
    "batch_input_path",
    "batch_output_path",
    "model",
    "use_resolution_tab",  # Added to support Resolution tab integration
    "tile_size",
    "overlap",
    "denoising_strength",
    "sharpening",
    "color_correction",
    "gpu_acceleration",
    "gpu_device",
    "batch_size",
    "face_restore_after_upscale",
    "create_subfolders",
    # vNext sizing (app-level; preserves backward compatibility by appending)
    "upscale_factor",
    "max_resolution",
    "pre_downscale_then_upscale",
    # Resume from an existing chunk run folder.
    "resume_run_dir",
]


def _gan_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    return dict(zip(GAN_ORDER, args))


def _apply_gan_preset(
    preset: Dict[str, Any],
    defaults: Dict[str, Any],
    preset_manager: PresetManager,
    current: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    base = defaults.copy()
    if current:
        base.update(current)
    merged = preset_manager.merge_config(base, preset)
    return [merged[k] for k in GAN_ORDER]


from shared.gpu_utils import expand_cuda_device_spec, get_global_gpu_override, validate_cuda_device_spec


def _expand_cuda_spec_gan(cuda_spec: str) -> str:
    """
    DEPRECATED: Use shared.gpu_utils.expand_cuda_device_spec instead.
    Kept for backward compatibility.
    """
    return expand_cuda_device_spec(cuda_spec)


def _validate_cuda_devices(cuda_spec: str) -> Optional[str]:
    """
    DEPRECATED: Use shared.gpu_utils.validate_cuda_device_spec instead.
    Kept for backward compatibility.
    """
    return validate_cuda_device_spec(cuda_spec)
    return None


def build_gan_callbacks(
    preset_manager: PresetManager,
    runner,  # Runner instance for universal chunking support
    run_logger: RunLogger,
    global_settings: Dict[str, Any],
    shared_state: gr.State,
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path,
):
    defaults = gan_defaults(base_dir)
    cancel_event = threading.Event()

    def refresh_presets(model_name: str, select_name: Optional[str] = None):
        presets = preset_manager.list_presets("gan", model_name)
        last_used = preset_manager.get_last_used_name("gan", model_name)
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        return gr.update(choices=presets, value=value)

    def save_preset(preset_name: str, *args):
        """Save preset with validation"""
        if not preset_name.strip():
            return gr.update(), gr.update(value="WARNING: Enter a preset name before saving"), *list(args)

        try:
            # Validate component count
            if len(args) != len(GAN_ORDER):
                error_msg = f"WARNING: Preset mismatch: {len(args)} values vs {len(GAN_ORDER)} expected. Check inputs_list in gan_tab.py"
                return gr.update(), gr.update(value=error_msg), *list(args)
            
            payload = _gan_dict_from_args(list(args))
            model_name = payload["model"]
            preset_manager.save_preset_safe("gan", model_name, preset_name.strip(), payload)
            dropdown = refresh_presets(model_name, select_name=preset_name.strip())

            current_map = dict(zip(GAN_ORDER, list(args)))
            loaded_vals = _apply_gan_preset(payload, defaults, preset_manager, current=current_map)

            return dropdown, gr.update(value=f"SUCCESS: Saved preset '{preset_name}' for {model_name}"), *loaded_vals
        except Exception as e:
            return gr.update(), gr.update(value=f"ERROR: Error saving preset: {str(e)}"), *list(args)

    def load_preset(preset_name: str, model_name: str, current_values: List[Any]):
        """
        Load a preset.
        
        FIXED: Now returns (*values, status_message) to match UI output expectations.
        UI expects: inputs_list + [preset_status]
        """
        try:
            model_name = model_name or defaults["model"]
            preset = preset_manager.load_preset_safe("gan", model_name, preset_name)
            if preset:
                preset_manager.set_last_used("gan", model_name, preset_name)
                # Apply validation constraints
                preset = preset_manager.validate_preset_constraints(preset, "gan", model_name)

            defaults_with_model = defaults.copy()
            defaults_with_model["model"] = model_name
            current_map = dict(zip(GAN_ORDER, current_values))
            values = _apply_gan_preset(preset or {}, defaults_with_model, preset_manager, current=current_map)
            
            # Return values + status message (status is LAST)
            status_msg = f"SUCCESS: Loaded preset '{preset_name}'" if preset else "INFO: Preset not found"
            return (*values, gr.update(value=status_msg))
        except Exception as e:
            print(f"Error loading preset {preset_name}: {e}")
            # Return current values + error status
            return (*current_values, gr.update(value=f"ERROR: Error: {str(e)}"))

    def safe_defaults():
        return [defaults[k] for k in GAN_ORDER]

    def prepare_single(
        single_path: str,
        settings_base: Dict[str, Any],
        seed_controls_local: Dict[str, Any],
    ) -> Dict[str, Any]:
        s = settings_base.copy()
        s["input_path"] = normalize_path(single_path)
        s["base_dir"] = str(base_dir)  # Pass base directory for metadata lookup
        meta = _get_gan_meta(s.get("model", ""), base_dir)
        s["scale"] = meta.get("scale", s.get("scale", 4))  # Use metadata scale, default to 4x
        s["supports_multi_gpu"] = meta.get("supports_multi_gpu", False)
        s["model_name"] = meta.get("canonical", s.get("model", ""))
        # PNG padding (from Output tab cache if present)
        s["png_padding"] = int(seed_controls_local.get("png_padding_val", 6))  # Match CLI default
        return s

    def maybe_downscale(
        s: Dict[str, Any],
        seed_controls_local: Dict[str, Any],
        current_output_dir_local: Path,
        current_temp_dir_local: Path,
    ):
            """
            Dynamic resolution adjustment for fixed-scale GAN models.

            vNext behavior:
            - Use Upscale-x + max-edge cap (LONG side) to compute an effective scale.
            - For fixed-scale models (2x/4x), pre-downscale the input so that one model pass
              reaches the capped target (mandatory to avoid post-downscale).
            """
            # Get model scale factor
            model_scale = s.get("scale", 4)
            if model_scale <= 1:  # Not a scaling model
                return s

            # Decide whether to use global Resolution tab cache or local per-tab values
            use_global = bool(s.get("use_resolution_tab", True))
            model_cache = seed_controls_local.get("resolution_cache", {}).get(s.get("model"), {}) if use_global else {}

            if use_global:
                scale_raw = model_cache.get("upscale_factor_val")
                if scale_raw is None:
                    scale_raw = seed_controls_local.get("upscale_factor_val")
                if scale_raw is None:
                    scale_raw = s.get("upscale_factor", 4.0)
            else:
                scale_raw = s.get("upscale_factor", 4.0)

            scale_x = float(scale_raw or 4.0)
            # Max-edge cap is local to GAN settings (no global max propagation).
            max_edge = int(s.get("max_resolution", 0) or 0)

            # Get input dimensions
            dims = get_media_dimensions(s["input_path"])
            if not dims:
                return s

            input_w, input_h = dims
            plan = estimate_fixed_scale_upscale_plan_from_dims(
                input_w,
                input_h,
                requested_scale=float(scale_x),
                model_scale=int(model_scale),
                max_edge=int(max_edge or 0),
                force_pre_downscale=True,
            )

            optimal_w, optimal_h = int(plan.preprocess_width), int(plan.preprocess_height)

            # Skip if already at optimal resolution (within tolerance)
            tolerance = 32  # Allow some tolerance for rounding
            if abs(input_w - optimal_w) <= tolerance and abs(input_h - optimal_h) <= tolerance:
                return s

            in_type = detect_input_type(s["input_path"])

            if in_type == "video":
                # Save downscaled artifact into the user-visible output folder for this run.
                out_root = Path(s.get("_run_dir") or current_output_dir_local)
                original_name = s.get("_original_filename") or Path(s["input_path"]).name
                pre_out = downscaled_video_path(out_root, str(original_name))
                ok, _err = scale_video(
                    Path(s["input_path"]),
                    Path(pre_out),
                    int(optimal_w),
                    int(optimal_h),
                    lossless=True,
                    audio_copy_first=True,
                )
                if ok and Path(pre_out).exists():
                    s["_original_input_path_before_preprocess"] = s["input_path"]
                    s["_preprocessed_input_path"] = str(pre_out)
                    s["input_path"] = str(pre_out)
                    s["resolution_adjusted"] = True
            elif in_type == "image":
                # Image resolution adjustment with OpenCV
                try:
                    import cv2
                    img = cv2.imread(s["input_path"], cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        adjusted = cv2.resize(img, (optimal_w, optimal_h), interpolation=cv2.INTER_AREA)
                        pre_dir = Path(s.get("_run_dir") or current_output_dir_local) / "pre_processed"
                        pre_dir.mkdir(parents=True, exist_ok=True)
                        tmp_path = collision_safe_path(
                            pre_dir / f"gan_input_adjust_{Path(s['input_path']).stem}_{optimal_w}x{optimal_h}{Path(s['input_path']).suffix}"
                        )
                        cv2.imwrite(str(tmp_path), adjusted)
                        if tmp_path.exists():
                            s["_original_input_path_before_preprocess"] = s["input_path"]
                            s["_preprocessed_input_path"] = str(tmp_path)
                            s["input_path"] = str(tmp_path)
                            s["resolution_adjusted"] = True
                except Exception:
                    pass
            elif in_type == "directory":
                # Directory of frames: pre-downscale each image into a temp directory.
                try:
                    src_dir = Path(s["input_path"])
                    img_files = [p for p in sorted(src_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
                    if not img_files:
                        return s

                    tmp_dir = collision_safe_dir(
                        (Path(s.get("_run_dir") or current_output_dir_local) / "pre_processed")
                        / f"gan_input_adjust_{src_dir.name}_pre{optimal_w}x{optimal_h}"
                    )
                    tmp_dir.mkdir(parents=True, exist_ok=True)

                    import cv2
                    for f in img_files:
                        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                        if img is None:
                            continue
                        adjusted = cv2.resize(img, (optimal_w, optimal_h), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(str(tmp_dir / f.name), adjusted)

                    if any(tmp_dir.iterdir()):
                        s["_original_input_path_before_preprocess"] = s["input_path"]
                        s["_preprocessed_input_path"] = str(tmp_dir)
                        s["input_path"] = str(tmp_dir)
                        s["resolution_adjusted"] = True
                except Exception:
                    pass

            return s

    def run_action(
        upload,
        *args,
        preview_only: bool = False,
        state=None,
        progress=None,
        global_settings_snapshot: Dict[str, Any] | None = None,
        _global_settings: Dict[str, Any] = global_settings,
    ):
        # Streaming: run in background thread, stream log lines if available
        global_settings = (
            dict(global_settings_snapshot)
            if isinstance(global_settings_snapshot, dict)
            else dict(_global_settings)
        )
        progress_q: "queue.Queue[str]" = queue.Queue()
        result_holder: Dict[str, Any] = {}

        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

        def _media_updates(out_path: Optional[str]) -> tuple[Any, Any]:
            """
            Return (output_image_update, output_video_update) for the merged output panel.
            """
            try:
                if out_path and not Path(out_path).is_dir():
                    suf = Path(out_path).suffix.lower()
                    if suf in video_exts:
                        return gr.update(value=None, visible=False), gr.update(value=out_path, visible=True)
                    if suf in image_exts:
                        return gr.update(value=out_path, visible=True), gr.update(value=None, visible=False)
            except Exception:
                pass
            return gr.update(value=None, visible=False), gr.update(value=None, visible=False)

        def _running_indicator(title: str, subtitle: str, spinning: bool = True):
            safe_title = html.escape(str(title or ""))
            safe_subtitle = html.escape(str(subtitle or ""))
            spinner_style = "" if spinning else ' style="opacity:0.45; animation:none;"'
            indicator_html = (
                '<div class="processing-banner">'
                f'<div class="processing-spinner"{spinner_style}></div>'
                '<div class="processing-col">'
                f'<div class="processing-text">{safe_title}</div>'
                f'<div class="processing-sub">{safe_subtitle}</div>'
                "</div></div>"
            )
            return gr.update(value=indicator_html, visible=True)
        
        # Initialize progress if provided
        if progress:
            progress(0, desc="Initializing GAN upscaling...")

        # PRE-FLIGHT CHECKS (mirrors SeedVR2 for consistency)
        # Check ffmpeg availability
        from shared.error_handling import check_ffmpeg_available, check_disk_space
        ffmpeg_ok, ffmpeg_msg = check_ffmpeg_available()
        if not ffmpeg_ok:
            yield (
                "ERROR: ffmpeg not found in PATH",
                ffmpeg_msg or "Install ffmpeg and add to PATH before processing",
                gr.update(value="", visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                "ffmpeg missing",
                gr.update(value=None),
                gr.update(value="", visible=False),
                gr.update(visible=False),
                state or {}
            )
            return
        
        # Check disk space (require at least 5GB free)
        output_path_check = Path(global_settings.get("output_dir", output_dir))
        has_space, space_warning = check_disk_space(output_path_check, required_mb=5000)
        if not has_space:
            yield (
                "ERROR: Insufficient disk space",
                space_warning or f"Free up disk space before processing. Required: 5GB+, Available: {output_path_check}",
                gr.update(value="", visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                "Low disk space",
                gr.update(value=None),
                gr.update(value="", visible=False),
                gr.update(visible=False),
                state or {}
            )
            return
        elif space_warning:
            # Low space warning - continue but warn user
            if progress:
                progress(0, desc="WARNING: Low disk space detected")

        # Define worker functions (moved outside try block)
        def worker_single(prepped_settings):
            status, lg, outp, cmp_html, slider_upd = run_single(prepped_settings, progress_cb=progress_q.put)
            result_holder["payload"] = (status, lg, outp, cmp_html, slider_upd)

        def worker_batch(batch_items):
            """Process batch using BatchProcessor - no duplicate code"""
            from shared.batch_processor import BatchProcessor, BatchJob
            
            outputs = []
            logs = []
            last_cmp = ""
            last_slider = gr.update(value=None)
            
            # Create batch jobs
            jobs = []
            for item in batch_items:
                job = BatchJob(
                    input_path=str(item),
                    metadata={"settings": settings.copy()}
                )
                jobs.append(job)
            
            # Define processor function that reuses run_single
            def process_job(job: BatchJob) -> bool:
                if cancel_event.is_set():
                    return False
                
                try:
                    input_file = Path(job.input_path)
                    input_kind = detect_input_type(str(input_file))

                    from shared.path_utils import sanitize_filename, resolve_output_location
                    from shared.output_run_manager import batch_item_dir, prepare_batch_video_run_dir

                    batch_root = Path(current_output_dir)
                    batch_root.mkdir(parents=True, exist_ok=True)

                    if input_kind != "video":
                        # Batch images: write directly into the batch output folder using the original name.
                        safe_name = sanitize_filename(input_file.name)
                        desired_out = batch_root / safe_name

                        if desired_out.exists() and not overwrite_existing_batch:
                            job.status = "skipped"
                            job.output_path = str(desired_out)
                            job.metadata["log"] = f"Skipped (exists): {desired_out}"
                            return True
                        if desired_out.exists() and overwrite_existing_batch:
                            try:
                                desired_out.unlink(missing_ok=True)
                            except Exception:
                                pass

                        ps_base = prepare_single(job.input_path, settings, seed_controls)
                        ps_base["_original_filename"] = input_file.name
                        ps_base["_desired_output_path"] = str(desired_out)
                        ps = maybe_downscale(ps_base, seed_controls, current_output_dir, current_temp_dir)
                        status, lg, outp, cmp_html, slider_upd = run_single(ps, progress_cb=progress_q.put)
                    else:
                        # Batch videos: stable per-input folder under outputs/<input_stem>/ with chunk artifacts inside.
                        item_out_dir = batch_item_dir(batch_root, input_file.name)
                        predicted_final = resolve_output_location(
                            input_path=str(input_file),
                            output_format="mp4",
                            global_output_dir=str(item_out_dir),
                            batch_mode=False,
                            original_filename=input_file.name,
                        )
                        run_paths = prepare_batch_video_run_dir(
                            batch_root,
                            input_file.name,
                            input_path=str(input_file),
                            model_label="GAN",
                            mode=str(getattr(runner, "get_mode", lambda: "subprocess")() or "subprocess"),
                            overwrite_existing=overwrite_existing_batch,
                        )
                        if not run_paths:
                            if not overwrite_existing_batch:
                                job.status = "skipped"
                                job.output_path = str(predicted_final)
                                job.metadata["log"] = f"Skipped (exists): {item_out_dir}"
                                return True
                            job.status = "failed"
                            job.error_message = f"Could not create batch output folder: {item_out_dir}"
                            return False

                        ps_base = prepare_single(job.input_path, settings, seed_controls)
                        ps_base["_original_filename"] = input_file.name
                        ps_base["_run_dir"] = str(run_paths.run_dir)
                        ps_base["_processed_chunks_dir"] = str(run_paths.processed_chunks_dir)
                        # Keep outputs inside the per-item folder (GAN runner chooses filenames).
                        ps_base["output_override"] = str(run_paths.run_dir)
                        ps = maybe_downscale(ps_base, seed_controls, current_output_dir, current_temp_dir)
                        status, lg, outp, cmp_html, slider_upd = run_single(ps, progress_cb=progress_q.put)
                     
                    # Store results in job
                    job.output_path = outp
                    job.metadata["log"] = lg
                    job.metadata["comparison"] = cmp_html
                    job.metadata["slider"] = slider_upd
                    job.status = "completed" if outp else "failed"
                    
                    return bool(outp)
                except Exception as e:
                    job.status = "failed"
                    job.error_message = str(e)
                    return False
            
            # Use BatchProcessor for controlled execution
            batch_processor = BatchProcessor(max_workers=1)  # Sequential for GPU
            batch_result = batch_processor.process_batch(
                jobs=jobs,
                processor_func=process_job,
                max_concurrent=1  # Sequential processing for GPU-bound operations
            )
            
            # Collect results from jobs
            for job in jobs:
                if job.status == "completed" and job.output_path:
                    outputs.append(job.output_path)
                if "log" in job.metadata:
                    logs.append(job.metadata["log"])
                if "comparison" in job.metadata and job.metadata["comparison"]:
                    last_cmp = job.metadata["comparison"]
                if "slider" in job.metadata:
                    last_slider = job.metadata["slider"]
            
            # FIXED: Write consolidated metadata for GAN batch (matching SeedVR2 behavior)
            # Requirement: "only have 1 metadata for batch image upscale and have individual metadata for videos upscale"
            try:
                # Detect if this is image-only batch vs video/mixed
                video_count = sum(1 for item in batch_items if Path(item).suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'))
                image_count = len(batch_items) - video_count
                is_image_only_batch = image_count > 0 and video_count == 0
                
                metadata_dir = Path(current_output_dir)
                
                if is_image_only_batch:
                    # Image-only batch: Single consolidated metadata (no per-file metadata to avoid spam)
                    batch_metadata = {
                        "batch_type": "images",
                        "pipeline": "gan",
                        "total_files": len(batch_items),
                        "completed": batch_result.completed_files,
                        "failed": batch_result.failed_files,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "settings": settings,
                        "outputs": outputs,
                        "failed_files": [
                            {"input": job.input_path, "error": job.error_message}
                            for job in jobs if job.status == "failed"
                        ]
                    }
                    metadata_path = metadata_dir / "batch_gan_images_metadata.json"
                else:
                    # Video/mixed batch: Batch summary only (individual metadata written by run_logger per video)
                    batch_metadata = {
                        "batch_type": "videos" if video_count > 0 and image_count == 0 else "mixed",
                        "pipeline": "gan",
                        "total_files": len(batch_items),
                        "video_files": video_count,
                        "image_files": image_count,
                        "completed": batch_result.completed_files,
                        "failed": batch_result.failed_files,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "settings": settings,
                        "outputs": outputs,
                        "individual_metadata_note": "Each video has its own run_metadata.json in its output directory",
                        "failed_files": [
                            {"input": job.input_path, "error": job.error_message}
                            for job in jobs if job.status == "failed"
                        ]
                    }
                    metadata_path = metadata_dir / "batch_gan_videos_summary.json"
                
                import json
                with metadata_path.open("w", encoding="utf-8") as f:
                    json.dump(batch_metadata, f, indent=2)
                    
            except Exception as e:
                # Don't fail batch on metadata error
                if progress_q:
                    progress_q.put(f"WARNING: Failed to write batch metadata: {e}\n")
            
            result_holder["payload"] = (
                f"SUCCESS: Batch complete: {len(outputs)}/{len(batch_items)} processed ({batch_result.failed_files} failed)",
                "\n\n".join(logs),
                outputs[0] if outputs else None,
                last_cmp,
                last_slider,
            )

        try:
            state = state or {"seed_controls": {}}
            seed_controls = state.get("seed_controls", {})
            output_settings = seed_controls.get("output_settings", {}) if isinstance(seed_controls, dict) else {}
            if not isinstance(output_settings, dict):
                output_settings = {}
            global_gpu_device = get_global_gpu_override(seed_controls, global_settings)
            seed_controls["global_gpu_device_val"] = global_gpu_device
            seed_controls["global_rife_cuda_device_val"] = "" if global_gpu_device == "cpu" else global_gpu_device
            seed_controls["gan_chunk_preview"] = {
                "message": "No chunk preview available yet.",
                "gallery": [],
                "videos": [],
                "count": 0,
            }
            state["seed_controls"] = seed_controls
            # Clear any previous VRAM OOM banner at the start of a new run.
            clear_vram_oom_alert(state)
            cancel_event.clear()
            settings_dict = _gan_dict_from_args(list(args))
            settings = {**defaults, **settings_dict}
            if preview_only:
                settings["resume_run_dir"] = ""
            if settings.get("batch_enable"):
                settings["resume_run_dir"] = ""
            settings["output_override"] = str(settings.get("output_override") or "").strip()
            resolved_cuda_device = "" if global_gpu_device == "cpu" else global_gpu_device
            settings["cuda_device"] = resolved_cuda_device
            settings["gpu_device"] = resolved_cuda_device

            # Pull latest global paths in case user changed them in Global tab
            current_output_dir = Path(global_settings.get("output_dir", output_dir))
            current_temp_dir = Path(global_settings.get("temp_dir", temp_dir))

            resume_run_dir_raw = str(settings.get("resume_run_dir") or "").strip()
            resume_mode = bool(resume_run_dir_raw and (not settings.get("batch_enable")) and (not preview_only))

            original_filename = None
            if resume_mode:
                resume_run_dir = Path(normalize_path(resume_run_dir_raw))
                if not (resume_run_dir.exists() and resume_run_dir.is_dir()):
                    yield ("ERROR: Resume folder not found", f"Configured resume folder does not exist: {resume_run_dir}", gr.update(value="", visible=False), gr.update(value=None, visible=False), gr.update(value=None, visible=False), "Resume folder invalid", gr.update(value=None), gr.update(value="", visible=False), gr.update(visible=False), state)
                    return
                recovered_input, recovered_name, _recovered_source = resolve_resume_input_from_run_dir(resume_run_dir)
                if recovered_input is None:
                    yield (
                        "ERROR: Resume input not found",
                        (
                            f"Could not recover input source from resume folder: {resume_run_dir}. "
                            "Expected run_context.json input path or run_metadata/downscaled artifact."
                        ),
                        gr.update(value="", visible=False),
                        gr.update(value=None, visible=False),
                        gr.update(value=None, visible=False),
                        "Resume input missing",
                        gr.update(value=None),
                        gr.update(value="", visible=False),
                        gr.update(visible=False),
                        state,
                    )
                    return
                raw_inp = str(recovered_input)
                original_filename = recovered_name or Path(raw_inp).name
            else:
                # Input selection: in batch mode prefer `batch_input_path`.
                raw_inp = upload if upload else (
                    settings.get("batch_input_path") if settings.get("batch_enable") else settings.get("input_path")
                )
                # Gradio may provide FileData as a dict (path + orig_name). Handle both forms.
                if isinstance(raw_inp, dict):
                    original_filename = raw_inp.get("orig_name") or raw_inp.get("name")
                    raw_inp = raw_inp.get("path") or ""

            inp = normalize_path(str(raw_inp)) if raw_inp else ""
            settings["_original_filename"] = original_filename or (Path(inp).name if inp else "")
            try:
                state.setdefault("seed_controls", {})["_original_filename"] = settings["_original_filename"]
            except Exception:
                pass
            if settings.get("batch_enable"):
                if not inp or not Path(inp).exists() or not Path(inp).is_dir():
                    yield ("ERROR: Batch input folder missing", "", gr.update(value="", visible=False), gr.update(value=None, visible=False), gr.update(value=None, visible=False), "Error", gr.update(value=None), gr.update(value="", visible=False), gr.update(visible=False), state)
                    return
            else:
                if not inp or not Path(inp).exists():
                    yield ("ERROR: Input missing", "", gr.update(value="", visible=False), gr.update(value=None, visible=False), gr.update(value=None, visible=False), "Error", gr.update(value=None), gr.update(value="", visible=False), gr.update(visible=False), state)
                    return

            # Preview runs are single-item and operate on a prepared first-frame image.
            if preview_only and settings.get("batch_enable"):
                settings["batch_enable"] = False

            preview_original_input = inp
            if preview_only:
                preview_src, preview_note = prepare_preview_input(
                    inp,
                    Path(global_settings.get("temp_dir", temp_dir)),
                    prefix="gan",
                    as_single_frame_dir=False,
                )
                if not preview_src:
                    msg = preview_note or "Failed to prepare preview input."
                    yield (
                        "Preview preparation failed",
                        msg,
                        gr.update(value="", visible=False),
                        gr.update(value=None, visible=False),
                        gr.update(value=None, visible=False),
                        "Preview error",
                        gr.update(value=None),
                        gr.update(value="", visible=False),
                        gr.update(visible=False),
                        state,
                    )
                    return
                inp = normalize_path(preview_src)
                settings["output_format"] = "png"
                if preview_note:
                    progress_q.put(preview_note)

            settings["input_path"] = inp
            if preview_only:
                settings["_preview_original_input"] = preview_original_input
                settings["_preview_effective_input"] = inp
            if settings.get("batch_enable"):
                settings["batch_input_path"] = inp

                # Batch default: write to <batch_input>/upscaled_files when no explicit output folder is set.
                current_output_dir = resolve_batch_output_dir(
                    batch_input_path=inp,
                    batch_output_path=settings.get("batch_output_path"),
                    fallback_output_dir=current_output_dir,
                    default_subdir_name="upscaled_files",
                )
                settings["batch_output_path"] = str(current_output_dir)

            # Ensure output/temp directories exist.
            try:
                current_output_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            try:
                current_temp_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            # Expand "all" to device list if specified (uses shared GPU utility)
            cuda_device_raw = settings.get("cuda_device", "")
            if cuda_device_raw:
                settings["cuda_device"] = expand_cuda_device_spec(cuda_device_raw)

            cuda_warn = validate_cuda_device_spec(settings.get("cuda_device", ""))
            if cuda_warn:
                yield (f"WARNING: {cuda_warn}", "", gr.update(value="", visible=False), gr.update(value=None, visible=False), gr.update(value=None, visible=False), "CUDA Error", gr.update(value=None), gr.update(value="", visible=False), gr.update(visible=False), state)
                return
            devices = [d.strip() for d in str(settings.get("cuda_device") or "").split(",") if d.strip()]
            if len(devices) > 1:
                yield (
                    "WARNING: GAN backends currently use a single GPU; select one CUDA device.",
                    "",
                    gr.update(value="", visible=False),
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                    "Multi-GPU Error",
                    gr.update(value=None),
                    gr.update(value="", visible=False),
                    gr.update(visible=False),
                    state
                )
                return

            # Per-run toggle matches SeedVR2 behavior: local OR global face restore enables post-pass.
            face_apply = bool(settings.get("face_restore_after_upscale", False)) or bool(global_settings.get("face_global", False))
            face_strength = float(global_settings.get("face_strength", 0.5))
            backend_val = settings.get("backend", "realesrgan")

            # Apply shared Output-tab image settings (global source of truth for image saves).
            image_fmt = str(
                seed_controls.get(
                    "image_output_format_val",
                    output_settings.get("image_output_format", settings.get("output_format", "png")),
                )
                or "png"
            ).strip().lower()
            if image_fmt not in {"png", "jpg", "webp"}:
                image_fmt = "png"
            settings["output_format"] = image_fmt
            try:
                image_quality = int(
                    float(
                        seed_controls.get(
                            "image_output_quality_val",
                            output_settings.get("image_output_quality", settings.get("output_quality", 95)),
                        )
                        or 95
                    )
                )
            except Exception:
                image_quality = 95
            settings["output_quality"] = max(1, min(100, image_quality))

            settings["save_metadata"] = bool(
                seed_controls.get(
                    "save_metadata_val",
                    output_settings.get("save_metadata", settings.get("save_metadata", True)),
                )
            )
            if seed_controls.get("fps_override_val") is not None:
                try:
                    settings["fps_override"] = float(seed_controls.get("fps_override_val") or 0)
                except Exception:
                    settings["fps_override"] = 0.0
            # Audio mux preferences (used by chunking + final output postprocessing)
            if seed_controls.get("audio_codec_val") is not None:
                settings["audio_codec"] = seed_controls.get("audio_codec_val") or "copy"
            if seed_controls.get("audio_bitrate_val") is not None:
                settings["audio_bitrate"] = seed_controls.get("audio_bitrate_val") or ""
            # Advanced output encoding settings are defined globally in Output tab.
            # Propagate them into GAN runtime so ffmpeg reconstruction paths don't use hardcoded defaults.
            encode_overrides = {
                "video_codec": seed_controls.get("video_codec_val"),
                "video_quality": seed_controls.get("video_quality_val"),
                "video_preset": seed_controls.get("video_preset_val"),
                "pixel_format": seed_controls.get("pixel_format_val"),
                "two_pass_encoding": seed_controls.get("two_pass_encoding_val"),
            }
            for key in (
                "video_codec",
                "video_quality",
                "video_preset",
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
            cmp_mode = seed_controls.get("comparison_mode_val", "native")
            pin_pref = bool(seed_controls.get("pin_reference_val", False))
            fs_pref = bool(seed_controls.get("fullscreen_val", False))
            
            # Pull PySceneDetect chunking settings from Resolution tab (universal chunking)
            auto_chunk = bool(seed_controls.get("auto_chunk", True))
            frame_accurate_split = bool(seed_controls.get("frame_accurate_split", True))
            chunk_size_sec = float(seed_controls.get("chunk_size_sec", 0) or 0)
            chunk_overlap_sec = 0.0 if auto_chunk else float(seed_controls.get("chunk_overlap_sec", 0) or 0)
            per_chunk_cleanup = seed_controls.get("per_chunk_cleanup", False)
            overwrite_existing_batch = bool(seed_controls.get("overwrite_existing_batch_val", False))
            # PySceneDetect parameters now managed centrally in Resolution tab
            scene_threshold = float(seed_controls.get("scene_threshold", 27.0))
            min_scene_len = float(seed_controls.get("min_scene_len", 1.0))
            
            # Determine if PySceneDetect chunking should be used for video inputs
            from shared.path_utils import detect_input_type as detect_type
            input_type = detect_type(inp)
            if preview_only:
                input_type = detect_type(settings.get("input_path") or inp)
            # Chunking is handled in `run_single()` so it also works in batch mode.
            should_use_chunking = False
            
            # If chunking is enabled for video, use universal chunk_and_process
            if should_use_chunking:
                from shared.chunking import chunk_and_process
                from shared.runner import RunResult
                
                if progress:
                    mode_label = "Auto Chunk (PySceneDetect scenes)" if auto_chunk else f"Static Chunk ({chunk_size_sec:g}s)"
                    progress(0, desc=f"Starting {mode_label} for GAN processing...")
                
                # Prepare settings for chunking
                settings["chunk_size_sec"] = chunk_size_sec
                settings["chunk_overlap_sec"] = chunk_overlap_sec
                settings["per_chunk_cleanup"] = per_chunk_cleanup
                
                # Fixed: Make chunk_progress_cb a proper callback function (not a generator)
                # Store chunk status in a dict for UI updates
                chunk_status = {"completed": 0, "total": 0}
                
                def chunk_progress_cb(progress_val, desc=""):
                    """Non-generator callback for chunk progress (called by chunk_and_process)"""
                    # Update gr.Progress if available
                    if progress:
                        progress(progress_val, desc=desc)
                    
                    # Extract chunk numbers if present
                    import re
                    match = re.search(r'(\d+)/(\d+)', desc)
                    if match:
                        chunk_status["completed"] = int(match.group(1))
                        chunk_status["total"] = int(match.group(2))
                
                # Run chunked processing using universal chunk_and_process
                # This will route to runner.run_gan for each chunk based on model_type="gan"
                rc, clog, final_output, chunk_count = chunk_and_process(
                    runner=runner,  # Pass runner instance for model routing
                    settings=settings,
                    scene_threshold=scene_threshold,
                    min_scene_len=min_scene_len,
                    work_dir=Path(settings.get("_run_dir") or current_output_dir),
                    on_progress=lambda msg: progress(0, desc=msg) if progress else None,
                    chunk_seconds=0.0 if auto_chunk else chunk_size_sec,
                    chunk_overlap=0.0 if auto_chunk else chunk_overlap_sec,
                    per_chunk_cleanup=per_chunk_cleanup,
                    allow_partial=True,
                    global_output_dir=str(current_output_dir),
                    resume_from_partial=False,
                    progress_tracker=chunk_progress_cb,  # Now properly wired as callback
                    process_func=None,  # Use model_type routing to runner.run_gan
                    model_type="gan",  # Route to runner.run_gan for each chunk
                )
                
                if progress:
                    progress(1.0, desc="Chunking complete!")
                
                status = "SUCCESS: GAN chunked upscale complete" if rc == 0 else f"WARNING: GAN chunking failed (code {rc})"
                if rc != 0 and maybe_set_vram_oom_alert(state, model_label="GAN", text=clog, settings=settings):
                    status = "OOM: Out of VRAM (GPU) - see banner above"
                    show_vram_oom_modal(state, title="Out of VRAM (GPU) - GAN", duration=None)
                
                # Build comparison for chunked output
                video_comp_html_update = gr.update(value="", visible=False)
                slider_update = gr.update(value=None)
                
                if final_output and Path(final_output).exists():
                    if Path(final_output).suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
                        video_comp_html_value = create_video_comparison_html(
                            original_video=settings["input_path"],
                            upscaled_video=final_output,
                            height=600,
                            slider_position=50.0
                        )
                        video_comp_html_update = gr.update(value=video_comp_html_value, visible=True)
                    elif not Path(final_output).is_dir():
                        slider_update = gr.update(value=(settings["input_path"], final_output), visible=True)
                
                img_upd, vid_upd = _media_updates(final_output)
                yield (
                    status,
                    clog,
                    gr.update(value="", visible=False),
                    img_upd,
                    vid_upd,
                    f"Chunking: {'Auto (PySceneDetect scenes)' if auto_chunk else 'Static'} - {chunk_count} chunks",
                    slider_update,
                    video_comp_html_update,
                    gr.update(visible=False),
                    state
                )
                return
            
            def relocate_output(path_str: Optional[str]) -> Optional[str]:
                if not path_str:
                    return None
                target_root = settings.get("output_override")
                if not target_root:
                    return path_str
                src = Path(path_str)
                target_root_path = Path(normalize_path(target_root))
                target_root_path.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    dest = collision_safe_dir(target_root_path / src.name)
                    shutil.copytree(src, dest)
                    return str(dest)
                else:
                    dest = collision_safe_path(target_root_path / src.name)
                    shutil.copyfile(src, dest)
                    return str(dest)

            # Define run_single function (moved outside try block)
            def run_single(prepped_settings: Dict[str, Any], progress_cb: Optional[Callable[[str], None]] = None):
                if cancel_event.is_set():
                    return ("CANCELED: Canceled", "\n".join(["Canceled before start"]), None, "", gr.update(value=None))
                run_output_root = Path(prepped_settings.get("_run_dir") or current_output_dir)
                try:
                    run_output_root.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

                def _finalize_run_context(status_text: str, output_path_text: Optional[str], rc_value: Optional[int], cfg: Dict[str, Any]) -> None:
                    try:
                        run_dir_raw = cfg.get("_run_dir")
                        if not run_dir_raw:
                            return
                        effective_input = cfg.get("_effective_input_path") or cfg.get("input_path")
                        input_kind_for_ctx = detect_input_type(str(effective_input or ""))
                        pre_in = cfg.get("_preprocessed_input_path")
                        if input_kind_for_ctx == "image" and (not pre_in):
                            snap = ensure_image_input_artifact(
                                Path(run_dir_raw),
                                str(effective_input or ""),
                                preferred_stem=Path(cfg.get("_original_filename") or "used_input").stem,
                            )
                            if snap:
                                pre_in = str(snap)
                                cfg["_preprocessed_input_path"] = pre_in
                        finalize_run_context(
                            Path(run_dir_raw),
                            pipeline="gan",
                            status=str(status_text or ""),
                            returncode=(int(rc_value) if rc_value is not None else None),
                            output_path=str(output_path_text) if output_path_text else None,
                            original_input_path=str(
                                cfg.get("_original_input_path_before_preprocess")
                                or cfg.get("input_path")
                                or ""
                            ),
                            effective_input_path=str(effective_input) if effective_input else None,
                            preprocessed_input_path=str(pre_in) if pre_in else None,
                            input_kind=input_kind_for_ctx,
                        )
                    except Exception:
                        pass

                runtime_settings = prepped_settings
                runtime_input_type = detect_input_type(prepped_settings.get("input_path", ""))
                if runtime_input_type == "video":
                    runtime_settings = prepped_settings.copy()
                    runtime_settings["_requested_output_format"] = runtime_settings.get("output_format", "auto")
                    runtime_settings["output_format"] = "mp4"
                    if progress_cb:
                        progress_cb("Video output format forced to MP4 for playback/comparison; frame artifacts remain in frames/ and frames_upscaled/.")

                header_log = [
                    f"Model: {runtime_settings['model_name']}",
                    f"Backend: {backend_val}",
                    f"Scale: {runtime_settings['scale']}x",
                    f"GPU: {runtime_settings.get('cuda_device') or 'auto/CPU'} (single GPU enforced)",
                    f"Input: {runtime_settings['input_path']}",
                ]
                if progress_cb:
                    for line in header_log:
                        progress_cb(line)
                # Run backend and collect logs
                try:
                    # Use new unified GAN runner
                    # Support both `models/` and legacy `Image_Upscale_Models/`.
                    builtin_key = prepped_settings.get("model_name") or prepped_settings.get("model") or ""
                    if not _is_realesrgan_builtin(builtin_key):
                        candidates = [
                            prepped_settings.get("model_name"),
                            prepped_settings.get("model"),
                        ]
                        candidates = [c for c in candidates if c]
                        found = False
                        for name in candidates:
                            for folder in ("models", "Image_Upscale_Models"):
                                if (base_dir / folder / name).exists():
                                    found = True
                                    break
                            if found:
                                break
                        if not found:
                            return ("ERROR: Model weights not found", "\n".join(header_log + ["Missing model file."]), None, "", gr.update(value=None))
                    
                    # Add face restoration settings
                    runtime_settings["face_restore"] = face_apply
                    runtime_settings["face_strength"] = face_strength

                    # Apply universal chunking (Resolution tab) for video inputs (single + batch).
                    should_chunk_video = (detect_input_type(runtime_settings["input_path"]) == "video") and (auto_chunk or chunk_size_sec > 0)
                    resume_requested_local = bool(
                        runtime_settings.get("_resume_run_requested")
                        or str(runtime_settings.get("resume_run_dir") or "").strip()
                    )
                    if resume_requested_local and not should_chunk_video:
                        msg = (
                            "Resume run folder works only with chunk/scene-based video processing. "
                            "Enable chunking and keep the same settings as the original run."
                        )
                        return ("Resume unavailable for current mode", "\n".join(header_log + [msg]), None, "", gr.update(value=None))
                    if should_chunk_video:
                        from shared.chunking import chunk_and_process, check_resume_available

                        chunk_settings = runtime_settings.copy()
                        # Disable per-chunk face restore; apply once on final output to match non-chunked behavior.
                        chunk_settings["face_restore"] = False
                        chunk_settings["face_strength"] = face_strength
                        chunk_settings["chunk_size_sec"] = chunk_size_sec
                        chunk_settings["chunk_overlap_sec"] = chunk_overlap_sec
                        chunk_settings["per_chunk_cleanup"] = per_chunk_cleanup
                        chunk_settings["frame_accurate_split"] = frame_accurate_split
                        if resume_requested_local:
                            resume_ok, resume_msg = check_resume_available(run_output_root, "mp4")
                            if not resume_ok:
                                msg = (
                                    f"Resume requested but no resumable chunk outputs were found in {run_output_root}. "
                                    f"{resume_msg}"
                                )
                                return ("Resume failed", "\n".join(header_log + [msg]), None, "", gr.update(value=None))
                            if progress_cb:
                                progress_cb(
                                    "Resume run folder detected. Resuming from the last processed chunk "
                                    "and continuing remaining chunks (same settings required)."
                                )

                        def _chunk_progress_cb(_progress_val, desc=""):
                            desc_s = str(desc or "").strip()
                            if desc_s.lower().startswith("completed chunk"):
                                try:
                                    seed_controls["gan_chunk_preview"] = build_chunk_preview_payload(str(run_output_root))
                                except Exception:
                                    pass
                            if progress_cb and desc_s:
                                progress_cb(desc_s)

                        rc, clog, final_output, chunk_count = chunk_and_process(
                            runner=runner,
                            settings=chunk_settings,
                            scene_threshold=scene_threshold,
                            min_scene_len=min_scene_len,
                            work_dir=run_output_root,
                            on_progress=(lambda msg: progress_cb(msg) if progress_cb else None),
                            chunk_seconds=0.0 if auto_chunk else chunk_size_sec,
                            chunk_overlap=0.0 if auto_chunk else chunk_overlap_sec,
                            per_chunk_cleanup=per_chunk_cleanup,
                            allow_partial=True,
                            global_output_dir=str(run_output_root),
                            resume_from_partial=resume_requested_local,
                            progress_tracker=_chunk_progress_cb,
                            process_func=None,
                            model_type="gan",
                        )

                        status = "SUCCESS: GAN chunked upscale complete" if rc == 0 else f"WARNING: GAN chunking failed (code {rc})"
                        if rc != 0 and maybe_set_vram_oom_alert(state, model_label="GAN", text=clog, settings=chunk_settings):
                            status = "OOM: Out of VRAM (GPU) - see banner above"

                        outp = final_output
                        if outp and Path(outp).exists() and face_apply:
                            restored = restore_video(
                                outp,
                                strength=face_strength,
                                on_progress=progress_cb if progress_cb else None,
                            )
                            if restored:
                                outp = restored

                        # Global RIFE post-process (adds *_xFPS output and keeps original).
                        if (
                            outp
                            and Path(outp).exists()
                            and Path(outp).suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm")
                        ):
                            rife_out, rife_msg = maybe_apply_global_rife(
                                runner=runner,
                                output_video_path=outp,
                                seed_controls=seed_controls,
                                on_log=(lambda m: progress_cb(m) if progress_cb and m else None),
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
                                outp = rife_out
                            elif rife_msg and progress_cb:
                                progress_cb(rife_msg)
                            comp_vid_path, comp_vid_err = maybe_generate_input_vs_output_comparison(
                                runtime_settings.get("_original_input_path_before_preprocess") or runtime_settings.get("input_path"),
                                outp,
                                seed_controls,
                                label_output="GAN",
                                on_progress=progress_cb if progress_cb else None,
                            )
                            if comp_vid_path:
                                if progress_cb:
                                    progress_cb(f"Comparison video created: {comp_vid_path}")
                            elif comp_vid_err and progress_cb:
                                progress_cb(f"Comparison video failed: {comp_vid_err}")

                        # Update shared state output pointers.
                        if outp and Path(outp).exists():
                            try:
                                out_path = Path(outp)
                                seed_controls["last_output_dir"] = str(out_path.parent if out_path.is_file() else out_path)
                                seed_controls["last_output_path"] = str(out_path) if out_path.is_file() else None
                            except Exception:
                                pass

                        if bool(chunk_settings.get("save_metadata", True)):
                            try:
                                run_logger.write_summary(
                                    Path(outp) if outp else current_output_dir,
                                    {
                                        "input": runtime_settings.get("_original_input_path_before_preprocess") or runtime_settings.get("input_path"),
                                        "output": outp,
                                        "returncode": rc,
                                        "args": chunk_settings,
                                        "face_apply": face_apply,
                                        "pipeline": "gan",
                                        "chunking": {
                                            "mode": "auto" if auto_chunk else "static",
                                            "chunk_size_sec": 0.0 if auto_chunk else chunk_size_sec,
                                            "scene_threshold": scene_threshold,
                                            "min_scene_len": min_scene_len,
                                            "chunks": chunk_count,
                                        },
                                    },
                                )
                            except Exception:
                                pass

                        try:
                            seed_controls["gan_chunk_preview"] = build_chunk_preview_payload(str(run_output_root))
                        except Exception:
                            pass

                        full_log = "\n".join(header_log + [clog])
                        if progress_cb:
                            progress_cb(status)

                        cmp_html = ""
                        slider_update = gr.update(value=None)
                        src = runtime_settings.get("_original_input_path_before_preprocess") or runtime_settings.get("input_path")
                        if outp and Path(outp).exists():
                            if Path(outp).suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"):
                                cmp_html = create_video_comparison_html(
                                    original_video=src,
                                    upscaled_video=outp,
                                    height=600,
                                    slider_position=50.0
                                )
                            elif Path(outp).is_dir():
                                cmp_html = f"<p>PNG frames saved to {outp}</p>"
                            else:
                                slider_update = gr.update(value=(src, outp), visible=True)
                        _finalize_run_context(status, outp, rc, runtime_settings)
                        return status, full_log, outp, cmp_html, slider_update
                    
                    # Use GAN runner with proper backend integration
                    result = run_gan_upscale(
                        input_path=runtime_settings["input_path"],
                        model_name=runtime_settings["model"],
                        settings=runtime_settings,
                        base_dir=base_dir,
                        temp_dir=current_temp_dir,
                        output_dir=run_output_root,
                        on_progress=progress_cb if progress_cb else None,
                        cancel_event=cancel_event  # Fixed: Pass cancel event to enable cancellation
                    )
                except Exception as exc:  # surface ffmpeg or other runtime issues
                    err_msg = f"ERROR: GAN upscale failed: {exc}"
                    if maybe_set_vram_oom_alert(state, model_label="GAN", text=str(exc), settings=runtime_settings):
                        # NOTE: Modal popups must be triggered from the main Gradio event thread.
                        # We show the modal after the worker thread finishes (see below).
                        pass
                    if progress_cb:
                        progress_cb(err_msg)
                    return (err_msg, "\n".join(header_log + [str(exc)]), None, "", gr.update(value=None))
                if cancel_event.is_set():
                    status = "CANCELED: Canceled"
                else:
                    status = "SUCCESS: GAN upscale complete" if result.returncode == 0 else f"WARNING: GAN upscale failed"
                log_body = result.log or ""
                if result.returncode != 0 and maybe_set_vram_oom_alert(state, model_label="GAN", text=log_body, settings=runtime_settings):
                    status = "OOM: Out of VRAM (GPU) - see banner above"
                full_log = "\n".join(header_log + [log_body])
                if progress_cb:
                    progress_cb(status)
                final_out_path = result.output_path

                # Relocate/rename outputs when a deterministic destination is requested (batch images, single-image numbering).
                desired_out = prepped_settings.get("_desired_output_path")
                if desired_out and final_out_path and Path(final_out_path).exists():
                    try:
                        src_path = Path(final_out_path)
                        dest_path = Path(normalize_path(str(desired_out)))
                        dest_path.parent.mkdir(parents=True, exist_ok=True)

                        if src_path.is_dir():
                            if dest_path.exists():
                                shutil.rmtree(dest_path, ignore_errors=True)
                            shutil.copytree(src_path, dest_path, dirs_exist_ok=False)
                            final_out_path = str(dest_path)
                            result.output_path = final_out_path
                        else:
                            if dest_path.exists():
                                dest_path.unlink(missing_ok=True)

                            # If the requested extension differs, convert via PIL for correctness.
                            if dest_path.suffix.lower() != src_path.suffix.lower():
                                from PIL import Image

                                img = Image.open(src_path)
                                fmt = dest_path.suffix.lower().lstrip(".")
                                quality = int(runtime_settings.get("output_quality", 95) or 95)
                                if fmt in ("jpg", "jpeg"):
                                    img = img.convert("RGB")
                                    img.save(dest_path, format="JPEG", quality=quality)
                                elif fmt == "webp":
                                    img.save(dest_path, format="WEBP", quality=quality)
                                else:
                                    img.save(dest_path)
                                try:
                                    src_path.unlink(missing_ok=True)
                                except Exception:
                                    pass
                            else:
                                shutil.move(str(src_path), str(dest_path))

                            final_out_path = str(dest_path)
                            result.output_path = final_out_path
                    except Exception:
                        pass

                # Preserve audio for video outputs (GAN video pipeline reconstructs video from frames -> no audio by default).
                try:
                    if final_out_path and Path(final_out_path).exists() and Path(final_out_path).suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
                        from shared.audio_utils import ensure_audio_on_video

                        audio_src = runtime_settings.get("_original_input_path_before_preprocess") or runtime_settings.get("input_path")
                        if audio_src and Path(audio_src).exists():
                            audio_codec = str(runtime_settings.get("audio_codec") or settings.get("audio_codec") or "copy")
                            audio_bitrate = runtime_settings.get("audio_bitrate") or settings.get("audio_bitrate") or None
                            _changed, _final, _err = ensure_audio_on_video(
                                Path(final_out_path),
                                Path(audio_src),
                                audio_codec=audio_codec,
                                audio_bitrate=str(audio_bitrate) if audio_bitrate else None,
                                on_progress=progress_cb if progress_cb else None,
                            )
                            if _err and progress_cb:
                                progress_cb(f"WARNING: Audio mux: {_err}")
                            if _final and str(_final) != str(final_out_path):
                                final_out_path = str(_final)
                                result.output_path = final_out_path
                except Exception:
                    pass

                # Global RIFE post-process (adds *_xFPS output while keeping original).
                if final_out_path and Path(final_out_path).exists() and Path(final_out_path).suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
                    rife_out, rife_msg = maybe_apply_global_rife(
                        runner=runner,
                        output_video_path=final_out_path,
                        seed_controls=seed_controls,
                        on_log=(lambda m: progress_cb(m) if progress_cb and m else None),
                    )
                    if rife_out and Path(rife_out).exists():
                        final_out_path = rife_out
                        result.output_path = final_out_path
                        full_log = "\n".join([full_log, f"Global RIFE output: {rife_out}"])
                    elif rife_msg:
                        full_log = "\n".join([full_log, rife_msg])
                    comp_vid_path, comp_vid_err = maybe_generate_input_vs_output_comparison(
                        runtime_settings.get("_original_input_path_before_preprocess") or runtime_settings.get("input_path"),
                        final_out_path,
                        seed_controls,
                        label_output="GAN",
                        on_progress=progress_cb if progress_cb else None,
                    )
                    if comp_vid_path:
                        full_log = "\n".join([full_log, f"Comparison video created: {comp_vid_path}"])
                    elif comp_vid_err:
                        full_log = "\n".join([full_log, f"Comparison video failed: {comp_vid_err}"])

                # Save preprocessed input (if we created one) alongside outputs
                pre_in = runtime_settings.get("_preprocessed_input_path")
                if pre_in and Path(pre_in).exists():
                    full_log = "\n".join([full_log, f"Downscaled input saved: {pre_in}"])
                    if progress_cb:
                        progress_cb(f"Downscaled input saved: {pre_in}")

                if final_out_path:
                    try:
                        outp = Path(final_out_path)
                        seed_controls["last_output_dir"] = str(outp.parent if outp.is_file() else outp)
                        seed_controls["last_output_path"] = str(outp) if outp.is_file() else None
                    except Exception:
                        pass
                if bool(runtime_settings.get("save_metadata", True)):
                    try:
                        run_logger.write_summary(
                            Path(final_out_path) if final_out_path else run_output_root,
                            {
                                "input": runtime_settings.get("_original_input_path_before_preprocess")
                                or runtime_settings.get("input_path"),
                                "output": final_out_path,
                                "returncode": result.returncode,
                                "args": runtime_settings,
                                "face_apply": face_apply,
                                "pipeline": "gan",
                            },
                        )
                    except Exception:
                        pass
                cmp_html = ""
                slider_update = gr.update(value=None)
                if final_out_path:
                    src = runtime_settings.get("_original_input_path_before_preprocess") or runtime_settings.get("input_path")
                    outp = final_out_path
                    if Path(outp).exists():
                        if Path(outp).suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
                            # Use imported video comparison function
                            cmp_html = create_video_comparison_html(
                                original_video=src,
                                upscaled_video=outp,
                                height=600,
                                slider_position=50.0
                            )
                        elif Path(outp).is_dir():
                            cmp_html = f"<p>PNG frames saved to {outp}</p>"
                        else:
                            # For images, use ImageSlider directly
                            slider_update = gr.update(value=(src, outp), visible=True)
                            cmp_html = ""  # No HTML comparison for images, use slider instead
                _finalize_run_context(status, final_out_path, result.returncode, runtime_settings)
                return status, full_log, final_out_path, cmp_html, slider_update

            # Kick off worker thread
            if settings.get("batch_enable"):
                folder = Path(settings["input_path"])
                # Check if this is a frame folder (contains only images, treated as a sequence)
                image_files = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")]
                video_files = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi")]

                if image_files and len(image_files) > 1 and not video_files:
                    # Treat as frame folder - process as single unit
                    items = [folder]
                else:
                    # Regular batch processing - individual files
                    items = image_files + video_files

                if not items:
                    yield (
                        "ERROR: No media files or frame folders found in batch folder",
                        "",
                        gr.update(value="", visible=False),
                        gr.update(value=None, visible=False),
                        gr.update(value=None, visible=False),
                        "Error",
                        gr.update(value=None),
                        gr.update(value="", visible=False),
                        gr.update(visible=False),
                        state,
                    )
                    return
                t = threading.Thread(target=worker_batch, args=(items,), daemon=True)
            else:
                # Per-run output folder for single video/image runs (0001/0002/...)
                # Keeps all artifacts user-visible and prevents collisions between app instances.
                if input_type in {"video", "image"}:
                    resume_run_dir_raw = str(settings.get("resume_run_dir") or "").strip()
                    if resume_run_dir_raw:
                        resume_run_dir = Path(normalize_path(resume_run_dir_raw))
                        if not (resume_run_dir.exists() and resume_run_dir.is_dir()):
                            yield (
                                "ERROR: Resume folder not found",
                                f"Configured resume folder does not exist: {resume_run_dir}",
                                gr.update(value="", visible=False),
                                gr.update(value=None, visible=False),
                                gr.update(value=None, visible=False),
                                "Resume folder invalid",
                                gr.update(value=None),
                                gr.update(value="", visible=False),
                                gr.update(visible=False),
                                state,
                            )
                            return
                        current_output_dir = resume_run_dir
                        processed_chunks_dir = current_output_dir / "processed_chunks"
                        processed_chunks_dir.mkdir(parents=True, exist_ok=True)
                        seed_controls["last_run_dir"] = str(current_output_dir)
                        settings["_run_dir"] = str(current_output_dir)
                        settings["_processed_chunks_dir"] = str(processed_chunks_dir)
                        settings["_resume_run_requested"] = bool(input_type == "video")
                        settings["_user_output_override_raw"] = settings.get("output_override") or ""
                        # Resume mode ignores new output overrides and keeps writing in the selected run folder.
                        settings["output_override"] = str(current_output_dir)
                        if input_type != "video":
                            settings["resume_run_dir"] = ""
                        progress_q.put(
                            f"Resume run folder detected: {current_output_dir}. "
                            "Processing will continue from the next chunk (same settings required)."
                        )
                    else:
                        try:
                            run_paths, _explicit_final = prepare_single_video_run(
                                output_root_fallback=current_output_dir,
                                output_override_raw=settings.get("output_override"),
                                input_path=settings["input_path"],
                                original_filename=Path(settings["input_path"]).name,
                                model_label="GAN",
                                mode=str(getattr(runner, "get_mode", lambda: "subprocess")() or "subprocess"),
                            )
                            current_output_dir = Path(run_paths.run_dir)
                            seed_controls["last_run_dir"] = str(current_output_dir)
                            settings["_run_dir"] = str(current_output_dir)
                            settings["_user_output_override_raw"] = settings.get("output_override") or ""
                            # For GAN, we use the run folder as the output root; explicit file naming is handled later.
                            settings["output_override"] = str(current_output_dir)
                        except Exception:
                            pass

                prepped = maybe_downscale(
                    prepare_single(settings["input_path"], settings, seed_controls),
                    seed_controls,
                    current_output_dir,
                    current_temp_dir,
                )
                # Single-image runs: save final output inside the per-run folder with original filename.
                try:
                    if detect_input_type(prepped.get("input_path") or "") == "image":
                        fmt = str(prepped.get("output_format") or "auto").lower()
                        if fmt in ("", "auto"):
                            ext = Path(prepped.get("_original_filename") or prepped["input_path"]).suffix or ".png"
                        else:
                            ext = f".{fmt}"
                        if ext.lower() == ".jpeg":
                            ext = ".jpg"
                        orig_name = prepped.get("_original_filename") or Path(prepped["input_path"]).name
                        stem = Path(orig_name).stem or "image"
                        prepped["_desired_output_path"] = str(
                            collision_safe_path(Path(current_output_dir) / f"{stem}{ext}")
                        )
                except Exception:
                    pass
                t = threading.Thread(target=worker_single, args=(prepped,), daemon=True)
            t.start()

            last_yield = time.time()
            last_progress_update = 0.0
            
            while t.is_alive() or not progress_q.empty():
                try:
                    line = progress_q.get(timeout=0.2)
                    if line:
                        line_s = str(line).strip()
                        if not line_s:
                            continue
                        live_logs = result_holder.setdefault("live_logs", [])
                        frame_marker = "FRAME_PROGRESS "
                        frame_progress_detail = None
                        if line_s.startswith(frame_marker):
                            frame_progress_detail = line_s[len(frame_marker):].strip()
                            result_holder["runtime_status_detail"] = frame_progress_detail
                            compact_line = f"{frame_marker}{frame_progress_detail}"
                            if live_logs and str(live_logs[-1]).startswith(frame_marker):
                                live_logs[-1] = compact_line
                            else:
                                live_logs.append(compact_line)
                        else:
                            live_logs.append(line_s)
                            result_holder["runtime_status_detail"] = line_s
                        
                        # Update gr.Progress from log messages
                        if progress:
                            import re
                            if frame_progress_detail:
                                nm_match = re.search(r'(\d+)\s*/\s*(\d+)', frame_progress_detail)
                                if nm_match:
                                    current = int(nm_match.group(1))
                                    total = int(nm_match.group(2))
                                    if total > 0:
                                        pct = current / total
                                        progress(pct, desc=f"{current}/{total} frames ({int(pct * 100)}%)")
                                        last_progress_update = pct
                            else:
                                # Look for progress indicators in logs
                                # Pattern: "Progress: 50%", "Processing 5/10 frames", etc.
                                pct_match = re.search(r'(\d+(?:\.\d+)?)%', line_s)
                                if pct_match:
                                    pct = float(pct_match.group(1)) / 100.0
                                    progress(pct, desc=line_s[:100])
                                    last_progress_update = pct
                                elif re.search(r'(\d+)/(\d+)', line_s):
                                    nm_match = re.search(r'(\d+)/(\d+)', line_s)
                                    if nm_match:
                                        current = int(nm_match.group(1))
                                        total = int(nm_match.group(2))
                                        if total > 0:
                                            pct = current / total
                                            progress(pct, desc=line_s[:100])
                                            last_progress_update = pct
                except queue.Empty:
                    pass
                now = time.time()
                if now - last_yield > 0.5:
                    last_yield = now
                    live_logs = result_holder.get("live_logs", [])
                    img_upd, vid_upd = _media_updates(None)
                    status_title = "RUNNING: GAN upscaling in progress"
                    status_detail = str(result_holder.get("runtime_status_detail") or "").strip()
                    if not status_detail:
                        for cand in reversed(live_logs):
                            cand_s = str(cand or "").strip()
                            if cand_s.startswith("FRAME_PROGRESS "):
                                cand_s = cand_s[len("FRAME_PROGRESS "):].strip()
                            if cand_s:
                                status_detail = cand_s
                                break
                    if not status_detail:
                        status_detail = "Waiting for model output..."
                    yield (
                        gr.update(value=status_title),
                        "\n".join(live_logs[-400:]),
                        _running_indicator(status_title, status_detail, spinning=True),
                        img_upd,
                        vid_upd,
                        status_detail,
                        gr.update(value=None),
                        gr.update(value="", visible=False),
                        gr.update(visible=False),
                        state
                    )
                time.sleep(0.1)
            t.join()

            status, lg, outp, cmp_html, slider_upd = result_holder.get(
                "payload",
                ("ERROR: Failed", "", None, "", gr.update(value=None)),
            )
            live_logs = result_holder.get("live_logs", [])
            merged_logs = lg if lg else "\n".join(live_logs)
            if preview_only:
                status_lc = str(status).lower()
                if "failed" in status_lc or "error" in status_lc:
                    status = "Preview failed"
                elif "cancel" in status_lc:
                    status = "Preview canceled"
                else:
                    status = "Preview complete"
            
            # Update progress to 100% on completion
            if progress:
                status_lc = str(status or "").lower()
                if ("success" in status_lc or "complete" in status_lc) and ("error" not in status_lc and "fail" not in status_lc):
                    progress(1.0, desc="GAN upscaling complete")
                elif ("error" in status_lc or "fail" in status_lc or "oom" in status_lc):
                    progress(0, desc="Failed")
            
            # Build video comparison for videos
            video_comp_html_update = gr.update(value="", visible=False)
            if cmp_html:
                video_comp_html_update = gr.update(value=cmp_html, visible=True)
            elif outp and Path(outp).exists() and Path(outp).suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
                original_path = settings.get("input_path", "")
                if original_path and Path(original_path).exists() and Path(original_path).is_file():
                    video_comp_html_value = create_video_comparison_html(
                        original_video=original_path,
                        upscaled_video=outp,
                        height=600,
                        slider_position=50.0
                    )
                    video_comp_html_update = gr.update(value=video_comp_html_value, visible=True)

            # If VRAM OOM was detected during the worker run, show a modal popup (easy to notice).
            if isinstance(state, dict) and state.get("alerts", {}).get("oom", {}).get("visible"):
                show_vram_oom_modal(state, title="Out of VRAM (GPU) - GAN", duration=None)
            
            img_upd, vid_upd = _media_updates(outp)
            yield (
                status,
                merged_logs,
                gr.update(value="", visible=False),
                img_upd,
                vid_upd,
                f"Output: {outp}" if outp else "No output",
                slider_upd,
                video_comp_html_update,
                gr.update(visible=False),
                state
            )
        except Exception as e:
            error_msg = f"Critical error in GAN processing: {str(e)}"
            yield (
                "ERROR: Critical error",
                error_msg,
                gr.update(value="", visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                "Error",
                gr.update(value=None),
                gr.update(value="", visible=False),
                gr.update(visible=False),
                state or {}
            )


    def cancel_action(state=None):
        """
        Cancel GAN processing and attempt to compile partial outputs if available.
        Uses output-run folders first, then legacy temp fallback.
        """
        cancel_event.set()
        try:
            runner.cancel()
        except Exception:
            pass

        compiled_output: Optional[str] = None
        live_output_root = Path(global_settings.get("output_dir", output_dir))
        state_obj = state or {}
        seed_controls = state_obj.get("seed_controls", {}) if isinstance(state_obj, dict) else {}
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
                    partial_basename="cancelled_gan_partial",
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
                gr.update(value=f"Cancelled - Error compiling partials: {str(e)}"),
                "Cancellation successful but partial compilation failed",
                state_obj,
            )

        if not compiled_output:
            try:
                temp_chunks_dir = Path(global_settings.get("temp_dir", temp_dir)) / "chunks"
                if temp_chunks_dir.exists():
                    from shared.chunking import salvage_partial_from_run_dir

                    partial_path, _method = salvage_partial_from_run_dir(
                        temp_chunks_dir,
                        partial_basename="cancelled_gan_partial",
                        encode_settings=output_settings,
                    )
                    if partial_path and Path(partial_path).exists():
                        compiled_output = str(partial_path)
            except Exception:
                pass

        if compiled_output:
            return (
                gr.update(value=f"Cancelled - Partial GAN output compiled: {Path(compiled_output).name}"),
                f"Partial results saved to: {compiled_output}",
                state_obj,
            )
        return (
            gr.update(value="Cancelled - No partial outputs to compile"),
            "Processing was cancelled before any chunks were completed",
            state_obj,
        )

    def get_model_scale(model_name: str) -> int:
        """Get the scale factor for a specific model"""
        if not model_name:
            return 4
        try:
            from shared.gan_runner import get_gan_model_metadata
            meta = get_gan_model_metadata(model_name, Path(defaults["base_dir"]))
            return meta.scale
        except Exception:
            return _parse_scale_from_name(model_name)

    def open_outputs_folder_gan(state: Dict[str, Any]):
        """Open outputs folder - delegates to shared utility"""
        from shared.services.global_service import open_outputs_folder
        live_output_dir = str(global_settings.get("output_dir", output_dir))
        return open_outputs_folder(live_output_dir)
    
    def clear_temp_folder_gan(confirm: bool):
        """Clear temp folder - delegates to shared utility"""
        from shared.services.global_service import clear_temp_folder
        live_temp_dir = str(global_settings.get("temp_dir", temp_dir))
        return clear_temp_folder(live_temp_dir, confirm)

    return {
        "defaults": defaults,
        "order": GAN_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "run_action": run_action,
        "model_scanner": lambda: _scan_gan_models(base_dir),
        "cancel_action": lambda *args: cancel_action(args[0] if args else None),
        "get_model_scale": get_model_scale,
        "open_outputs_folder": open_outputs_folder_gan,
        "clear_temp_folder": clear_temp_folder_gan,
    }
