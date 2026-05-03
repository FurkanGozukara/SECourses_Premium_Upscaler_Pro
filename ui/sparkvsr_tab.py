"""
SparkVSR Tab - Self-contained modular implementation
Real-time diffusion-based streaming video super-resolution
UPDATED: Now uses Universal Preset System
"""

import gradio as gr
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Any
import html
import re
import threading
import time

from shared.services.sparkvsr_service import (
    build_sparkvsr_callbacks,
    SPARKVSR_ORDER,
    SPARKVSR_PRECISION_OPTIONS,
    SPARKVSR_UPSCALE_MODE_OPTIONS,
    SPARKVSR_REF_MODE_OPTIONS,
    SPARKVSR_SAVE_FORMAT_OPTIONS,
    canonical_sparkvsr_scale,
)
from shared.services.gan_service import PREFERRED_GAN_DEFAULT_MODEL, get_gan_model_metadata_lightweight
from shared.fixed_scale_analysis import build_fixed_scale_analysis_update
from shared.models.sparkvsr_meta import get_sparkvsr_default_model, get_sparkvsr_model_names
from shared.path_utils import get_media_dimensions, normalize_path
from shared.resolution_calculator import estimate_fixed_scale_upscale_plan_from_dims
from ui.universal_preset_section import (
    universal_preset_section,
    wire_universal_preset_events,
)
from shared.universal_preset import dict_to_values
from ui.media_preview import preview_updates
from shared.video_comparison_slider import get_video_comparison_js_on_load
from shared.processing_queue import get_processing_queue_manager, resolve_queue_gpu_resources
from shared.queue_state import (
    snapshot_queue_state,
    snapshot_global_settings,
    merge_payload_state,
)


def resolve_shared_upscale_factor(state: Dict[str, Any] | None) -> float | None:
    """
    Resolve shared/global upscale value from app state.
    """
    if not isinstance(state, dict):
        return None
    try:
        seed_controls = state.get("seed_controls", {}) or {}
        raw = seed_controls.get("upscale_factor_val")
        if raw is None:
            return None
        val = float(raw)
        if val <= 0:
            return None
        return val
    except Exception:
        return None


def resolve_sparkvsr_effective_scale(
    *,
    scale_state_val,
    use_global: bool,
    local_upscale_factor,
    state: Dict[str, Any] | None,
) -> int:
    """
    Resolve SparkVSR effective integer scale from UI + shared-state inputs.
    """
    default_scale = canonical_sparkvsr_scale(scale_value=scale_state_val, default=4)
    global_scale = resolve_shared_upscale_factor(state if bool(use_global) else None)
    return canonical_sparkvsr_scale(
        scale_value=scale_state_val,
        upscale_factor_value=(global_scale if global_scale is not None else local_upscale_factor),
        default=default_scale,
    )


def sparkvsr_tab(
    preset_manager,
    runner,
    run_logger,
    global_settings: Dict[str, Any],
    shared_state: gr.State,
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path
):
    """
    Self-contained SparkVSR tab following SECourses modular pattern.
    """

    # Build service callbacks
    service = build_sparkvsr_callbacks(
        preset_manager, runner, run_logger, global_settings, shared_state,
        base_dir, temp_dir, output_dir
    )
    queue_manager = get_processing_queue_manager()

    # Get defaults
    defaults = service["defaults"]
    
    # UNIVERSAL PRESET: Load from shared_state
    seed_controls = shared_state.value.get("seed_controls", {})
    sparkvsr_settings = seed_controls.get("sparkvsr_settings", {})
    models_list = seed_controls.get("available_models", ["default"])
    
    # Merge with defaults
    merged_defaults = defaults.copy()
    for key, value in sparkvsr_settings.items():
        if value is not None:
            merged_defaults[key] = value
    
    values = [merged_defaults[k] for k in SPARKVSR_ORDER]

    def _value(key: str, default=None):
        try:
            idx = SPARKVSR_ORDER.index(key)
            if 0 <= idx < len(values):
                raw = values[idx]
                if raw is None and default is not None:
                    return default
                return raw
        except Exception:
            pass
        return default

    def _compute_spatial_tile_count_text(
        path_val,
        tile_height_val,
        tile_width_val,
        overlap_height_val,
        overlap_width_val,
        scale_state_val,
        use_global,
        local_upscale_factor,
        max_edge_val,
        pre_downscale_val,
        state,
    ) -> tuple[str, bool]:
        try:
            tile_h = int(float(tile_height_val))
        except Exception:
            tile_h = 0
        try:
            tile_w = int(float(tile_width_val))
        except Exception:
            tile_w = 0
        tile_h = max(0, min(4096, tile_h))
        tile_w = max(0, min(4096, tile_w))

        if tile_h <= 0 or tile_w <= 0:
            return "**Spatial tiles:** disabled (`0 x 0` processes the full frame).", True

        try:
            overlap_h = int(float(overlap_height_val))
        except Exception:
            overlap_h = 32
        try:
            overlap_w = int(float(overlap_width_val))
        except Exception:
            overlap_w = 32
        overlap_h = max(0, min(1024, overlap_h))
        overlap_w = max(0, min(1024, overlap_w))

        notes = []
        if overlap_h >= tile_h:
            overlap_h = max(0, tile_h - 8)
            notes.append("height overlap auto-corrected")
        if overlap_w >= tile_w:
            overlap_w = max(0, tile_w - 8)
            notes.append("width overlap auto-corrected")
        stride_h = max(1, int(tile_h - overlap_h))
        stride_w = max(1, int(tile_w - overlap_w))

        normalized_path = normalize_path(path_val) if path_val else None
        if not normalized_path:
            return "**Spatial tiles:** set an input path to estimate tile count.", True

        dims = get_media_dimensions(normalized_path)
        if not dims:
            return "**Spatial tiles:** unable to read input dimensions yet.", True

        src_w, src_h = int(dims[0]), int(dims[1])
        tiling_w, tiling_h = src_w, src_h

        try:
            resolved_scale = resolve_sparkvsr_effective_scale(
                scale_state_val=scale_state_val,
                use_global=bool(use_global),
                local_upscale_factor=local_upscale_factor,
                state=state if isinstance(state, dict) else None,
            )
            model_scale = int(resolved_scale or 4)
            max_edge = int(max_edge_val or 0)

            plan = estimate_fixed_scale_upscale_plan_from_dims(
                src_w,
                src_h,
                requested_scale=float(model_scale),
                model_scale=int(model_scale),
                max_edge=max_edge,
                force_pre_downscale=bool(pre_downscale_val),
            )
            tiling_w = int(plan.resize_width or (src_w * model_scale))
            tiling_h = int(plan.resize_height or (src_h * model_scale))
        except Exception:
            try:
                model_scale = int(float(scale_state_val or 4))
            except Exception:
                model_scale = 4
            tiling_w, tiling_h = src_w * model_scale, src_h * model_scale

        rows = max(1, int(math.ceil((tiling_h - overlap_h) / float(stride_h))))
        cols = max(1, int(math.ceil((tiling_w - overlap_w) / float(stride_w))))
        total = int(rows * cols)

        preprocess_note = ""
        if tiling_w != src_w or tiling_h != src_h:
            preprocess_note = f"; output estimate from `{src_w}x{src_h}`"

        guardrail_note = ""
        if notes:
            guardrail_note = f" ({'; '.join(notes)})"

        text = (
            f"**Spatial tiles:** `{total}` (`{cols} x {rows}`)  \n"
            f"Target for tiling: `{tiling_w}x{tiling_h}`; stride: `{stride_w}x{stride_h}`{preprocess_note}{guardrail_note}."
        )
        return text, True

    def _tile_count_info_update(
        path_val,
        tile_height_val,
        tile_width_val,
        overlap_height_val,
        overlap_width_val,
        scale_state_val,
        use_global,
        local_upscale_factor,
        max_edge_val,
        pre_downscale_val,
        state,
    ):
        text, visible = _compute_spatial_tile_count_text(
            path_val,
            tile_height_val,
            tile_width_val,
            overlap_height_val,
            overlap_width_val,
            scale_state_val,
            use_global,
            local_upscale_factor,
            max_edge_val,
            pre_downscale_val,
            state,
        )
        return gr.update(value=text, visible=visible)
    # GPU detection and warnings (parent-process safe: NO torch import)
    cuda_available = False
    cuda_count = 0
    gpu_hint = "CUDA detection in progress..."
    
    try:
        from shared.gpu_utils import get_gpu_info

        gpus = get_gpu_info()
        cuda_count = len(gpus)
        cuda_available = cuda_count > 0
        
        if cuda_available:
            gpu_hint = f" Detected {cuda_count} CUDA GPU(s) - GPU acceleration available\n SparkVSR uses single GPU only (multi-GPU not supported)"
        else:
            gpu_hint = " CUDA not detected (nvidia-smi unavailable or no NVIDIA GPU) - Processing will use CPU (significantly slower)"
    except Exception as e:
        gpu_hint = f" CUDA detection failed: {str(e)}"
        cuda_available = False

    available_spark_models = sorted(
        {
            str(model_name).strip()
            for model_name in (models_list or [])
            if str(model_name).strip().lower().startswith("sparkvsr")
        }
    )
    if not available_spark_models:
        available_spark_models = get_sparkvsr_model_names()
    if not available_spark_models:
        available_spark_models = [get_sparkvsr_default_model()]
    model_name_value = str(_value("model_name", get_sparkvsr_default_model()) or get_sparkvsr_default_model())
    if model_name_value not in available_spark_models:
        available_spark_models = [model_name_value] + [m for m in available_spark_models if m != model_name_value]

    def _scan_gan_reference_models() -> list[str]:
        models: set[str] = set()
        for folder_name in ("models", "Image_Upscale_Models"):
            models_dir = base_dir / folder_name
            if not models_dir.exists():
                continue
            try:
                for f in models_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in {".pth", ".safetensors"}:
                        models.add(f.name)
            except Exception:
                continue
        ordered = sorted(models)
        preferred = {m.lower(): m for m in ordered}.get(PREFERRED_GAN_DEFAULT_MODEL.lower())
        if preferred:
            ordered = [preferred] + [m for m in ordered if m != preferred]
        return ordered

    spark_auto_ref_choices: list[tuple[str, str]] = [("SeedVR2", "SeedVR2")]
    for _gan_model in _scan_gan_reference_models():
        try:
            _meta = get_gan_model_metadata_lightweight(_gan_model, base_dir)
            _label = f"GAN: {_gan_model} (x{int(getattr(_meta, 'scale', 4) or 4)})"
        except Exception:
            _label = f"GAN: {_gan_model}"
        spark_auto_ref_choices.append((_label, f"GAN::{_gan_model}"))
    spark_auto_ref_choices.append(("FlashVSR+", "FlashVSR+"))
    spark_auto_ref_values = {str(v) for _, v in spark_auto_ref_choices}
    auto_reference_upscaler_value = str(_value("auto_reference_upscaler", "SeedVR2") or "SeedVR2").strip()
    if (
        auto_reference_upscaler_value not in spark_auto_ref_values
        and auto_reference_upscaler_value.lower().endswith((".pth", ".safetensors"))
    ):
        auto_reference_upscaler_value = f"GAN::{auto_reference_upscaler_value}"
    if auto_reference_upscaler_value not in spark_auto_ref_values:
        auto_reference_upscaler_value = "SeedVR2"

    # Show GPU warning if not available
    if not cuda_available:
        gr.Markdown(
            f'<div style="background: #fff3cd; padding: 12px; border-radius: 8px; border: 1px solid #ffc107;">'
            f'<strong> GPU Acceleration Unavailable</strong><br>'
            f'{gpu_hint}<br><br>'
            f'SparkVSR is designed for CUDA GPUs. CPU mode is possible but very slow.'
            f'</div>',
            elem_classes="warning-text"
        )

    with gr.Row():
        # Left Column: Input & Settings
        with gr.Column(scale=3):
            gr.Markdown("####  Input")

            with gr.Group():
                with gr.Row():
                    with gr.Column(scale=2, elem_classes=["SparkVSR-input-source-col"]):
                        input_file = gr.File(
                            label="Upload video or image (optional)",
                            type="filepath",
                            file_types=["video", "image"]
                        )
                        input_path = gr.Textbox(
                            label="Input Path",
                            value=_value("input_path", ""),
                            placeholder="C:/path/to/video.mp4 or C:/path/to/frames/",
                            info="Video file or image sequence folder"
                        )
                        copy_output_into_input_btn = gr.Button(
                            "Copy Output Into Input",
                            elem_classes=["action-btn", "action-btn-source-seed", "SparkVSR-copy-output-compact"],
                            size="md",
                        )

                    with gr.Column(scale=2):
                        # Backend-only state (kept for preset/schema compatibility).
                        _initial_scale_state = canonical_sparkvsr_scale(
                            scale_value=_value("scale", "4"),
                            upscale_factor_value=_value("upscale_factor", None),
                            default=4,
                        )
                        scale = gr.State(value=str(int(_initial_scale_state)))
                        model_name = gr.Dropdown(
                            label="SparkVSR Model",
                            choices=available_spark_models,
                            value=model_name_value,
                            info="Stage-2 is the official final SparkVSR checkpoint. Stage-1 is exposed for compatibility.",
                        )
                        model_path = gr.Textbox(
                            label="Model Path Override",
                            value=str(_value("model_path", "") or ""),
                            placeholder="Leave empty to use SparkVSR/models/<selected model>",
                            info="Optional local Diffusers model folder. Overrides the selected model when set.",
                        )
                        lora_path = gr.Textbox(
                            label="LoRA Path",
                            value=str(_value("lora_path", "") or ""),
                            placeholder="Optional .safetensors LoRA file",
                            info="Optional SparkVSR LoRA adapter path.",
                        )
                        upscale_mode = gr.Dropdown(
                            label="Input Upscale Mode",
                            choices=list(SPARKVSR_UPSCALE_MODE_OPTIONS),
                            value=(
                                str(_value("upscale_mode", "bilinear"))
                                if str(_value("upscale_mode", "bilinear")) in set(SPARKVSR_UPSCALE_MODE_OPTIONS)
                                else "bilinear"
                            ),
                            info="Interpolation used before SparkVSR diffusion refinement.",
                        )

                    with gr.Column(scale=2):
                        input_image_preview = gr.Image(
                            label=" Input Preview (Image)",
                            type="filepath",
                            interactive=False,
                            height=250,
                            visible=False
                        )
                        input_video_preview = gr.Video(
                            label=" Input Preview (Video)",
                            interactive=False,
                            height=250,
                            visible=False
                        )

                input_cache_msg = gr.Markdown("", visible=False)
                sizing_info = gr.Markdown("", visible=False, elem_classes=["resolution-info"])
                input_detection_result = gr.Markdown("", visible=False)

            # vNext sizing controls are placed in the right column to mirror SeedVR2 layout.

            # Processing Settings
            gr.Markdown("#### Core Runtime Settings")
            
            with gr.Group():
                with gr.Row():
                    precision = gr.Dropdown(
                        label="Precision",
                        choices=list(SPARKVSR_PRECISION_OPTIONS),
                        value=(
                            str(_value("precision", _value("dtype", "bfloat16")))
                            if str(_value("precision", _value("dtype", "bfloat16"))) in set(SPARKVSR_PRECISION_OPTIONS)
                            else "bfloat16"
                        ),
                        info="Default is bfloat16, matching the official SparkVSR script.",
                    )
                    noise_step = gr.Slider(
                        label="Noise Step",
                        minimum=0,
                        maximum=999,
                        step=1,
                        value=int(_value("noise_step", 0) or 0),
                        info="Reference/noise timestep. Official default is 0.",
                    )
                    sr_noise_step = gr.Slider(
                        label="SR Noise Step",
                        minimum=1,
                        maximum=999,
                        step=1,
                        value=int(_value("sr_noise_step", 399) or 399),
                        info="Super-resolution denoise timestep. Official default is 399.",
                    )
                    with gr.Column(scale=1):
                        seed = gr.Number(
                            label="Random Seed",
                            value=_value("seed", 0),
                            precision=0,
                            info="Seed for reproducibility."
                        )
                        auto_transfer_output_to_input = gr.Checkbox(
                            label="Auto Transfer Output to Input",
                            value=bool(_value("auto_transfer_output_to_input", False)),
                            info="After upscale completes, automatically copy the latest output path into Input Path.",
                        )
                    with gr.Column(scale=1):
                        auto_tune_btn = gr.Button(
                            "Auto Tune for Max Quality - VRAM Optimized",
                            size="md",
                            min_width=220,
                            elem_classes=["action-btn", "action-btn-optimize", "SparkVSR-autotune-tall"],
                        )
                        save_vram_gb = gr.Slider(
                            label="Save VRAM (GB)",
                            minimum=0.0,
                            maximum=9.9,
                            step=0.1,
                            value=float(_value("save_vram_gb", 2.0) or 2.0),
                        )
                optimize_summary = gr.Markdown(value="", visible=False, elem_classes=["resolution-info"])
                gr.Markdown(
                    (
                        "**Auto Tune:** Runs a short SparkVSR probe clip, measures live GPU VRAM, and applies the highest-quality "
                        "spatial tile / temporal chunk settings that keep the selected free-VRAM headroom."
                    )
                )

                with gr.Row():
                    chunk_len = gr.Slider(
                        label="Temporal Chunk Length",
                        minimum=0,
                        maximum=10000,
                        step=1,
                        value=int(_value("chunk_len", 0) or 0),
                        info=(
                            "0 = process all frames at once. Use shorter chunks for long clips or limited VRAM."
                        ),
                    )
                    overlap_t = gr.Slider(
                        label="Temporal Overlap",
                        minimum=0,
                        maximum=128,
                        step=1,
                        value=int(_value("overlap_t", 8) or 8),
                        info=(
                            "Overlap between temporal chunks. Official default is 8; must be smaller than chunk length."
                        ),
                    )

                with gr.Row():
                    cpu_offload = gr.Checkbox(
                        label="CPU Offload",
                        value=bool(_value("cpu_offload", True)),
                        info="Enable Diffusers sequential CPU offload. Recommended for redistribution and limited VRAM.",
                    )
                    vae_tiling = gr.Checkbox(
                        label="VAE Slicing/Tiling",
                        value=bool(_value("vae_tiling", True)),
                        info="Enable VAE slicing and tiling to reduce peak VRAM.",
                    )
                    group_offload = gr.Checkbox(
                        label="Group Offload",
                        value=bool(_value("group_offload", False)),
                        info="Optional Diffusers group offload for the transformer when available.",
                    )

                with gr.Row():
                    num_blocks_per_group = gr.Slider(
                        label="Blocks per Offload Group",
                        minimum=1,
                        maximum=64,
                        step=1,
                        value=int(_value("num_blocks_per_group", 1) or 1),
                        info="Used only when Group Offload is enabled.",
                    )
                    force_offload = gr.Checkbox(
                        label="Force Offload After Run",
                        value=bool(_value("force_offload", True)),
                        info="Clear CUDA memory after a run in the parent app workflow.",
                    )
                    enable_debug = gr.Checkbox(
                        label="Enable Debug Logging",
                        value=bool(_value("enable_debug", False)),
                        info="Verbose backend logging (timings, memory behavior, diagnostics).",
                    )
                gr.Markdown(
                    "**GPU Device:** Controlled globally from the top app header selector. "
                    "This tab no longer has a per-tab GPU override."
                )
                device = gr.State(str(_value("device", "auto") or "auto"))
            
            # Memory Optimization
            gr.Markdown("#### Spatial Tiling")
            
            with gr.Group():
                _initial_tile_count_text, _initial_tile_count_visible = _compute_spatial_tile_count_text(
                    _value("input_path", ""),
                    _value("tile_height", 0),
                    _value("tile_width", 0),
                    _value("overlap_height", 32),
                    _value("overlap_width", 32),
                    _value("scale", "4"),
                    _value("use_resolution_tab", True),
                    _value("upscale_factor", _value("scale", 4)),
                    _value("max_target_resolution", 1920),
                    _value("pre_downscale_then_upscale", True),
                    shared_state.value if isinstance(shared_state.value, dict) else {},
                )

                with gr.Row():
                    with gr.Column():
                        tile_height = gr.Slider(
                            label="Tile Height",
                            minimum=0, maximum=4096, step=16,
                            value=int(_value("tile_height", 0) or 0),
                            info="0 disables spatial tiling. Use 256-512 on lower VRAM GPUs."
                        )
                        tile_width = gr.Slider(
                            label="Tile Width",
                            minimum=0, maximum=4096, step=16,
                            value=int(_value("tile_width", 0) or 0),
                            info="0 disables spatial tiling. Use the same value as Tile Height for square tiles."
                        )
                        dit_tile_count = gr.Markdown(
                            value=_initial_tile_count_text,
                            visible=bool(_initial_tile_count_visible),
                            elem_classes=["resolution-info"],
                        )

                    with gr.Column():
                        overlap_height = gr.Slider(
                            label="Overlap Height",
                            minimum=0, maximum=1024, step=8,
                            value=int(_value("overlap_height", 32) or 32),
                            info="Vertical overlap between spatial tiles. Official default is 32."
                        )
                        overlap_width = gr.Slider(
                            label="Overlap Width",
                            minimum=0, maximum=1024, step=8,
                            value=int(_value("overlap_width", 32) or 32),
                            info="Horizontal overlap between spatial tiles. Official default is 32."
                        )

            gr.Markdown("#### Reference Guidance")
            with gr.Group():
                with gr.Row():
                    ref_mode = gr.Dropdown(
                        label="Reference Mode",
                        choices=list(SPARKVSR_REF_MODE_OPTIONS),
                        value=(
                            str(_value("ref_mode", "sr_image"))
                            if str(_value("ref_mode", "sr_image")) in set(SPARKVSR_REF_MODE_OPTIONS)
                            else "sr_image"
                        ),
                        info="sr_image is the local default. If the reference path is blank, SparkVSR uses the input video frame as the reference.",
                    )
                    ref_guidance_scale = gr.Slider(
                        label="Reference Guidance Scale",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.05,
                        value=float(_value("ref_guidance_scale", 1.0) or 1.0),
                        info="Strength of reference guidance. Official default is 1.0.",
                    )
                ref_indices = gr.Textbox(
                    label="Reference Indices",
                    value=str(_value("ref_indices", "0") or "0"),
                    placeholder="Example: 0",
                    info="Comma-separated source frame indices. For short clips, SparkVSR recommends using only frame 0.",
                )
                ref_source_path = gr.Textbox(
                    label="Local SR Reference Path",
                    value=str(_value("ref_source_path", "") or ""),
                    placeholder="Leave blank for input video fallback, or provide image/video/folder",
                    info=(
                        "Used by sr_image mode. Blank uses the source video; a locally upscaled keyframe/video gives better detail. "
                        "Auto first-frame references override this per chunk when enabled."
                    ),
                )
                with gr.Row():
                    auto_reference_prepass = gr.Checkbox(
                        label="Auto Upscale First Frame per Chunk",
                        value=bool(_value("auto_reference_prepass", True)),
                        info=(
                            "Before SparkVSR processes a chunk, upscale that chunk's first frame locally and use it "
                            "as the sr_image reference for that chunk."
                        ),
                        scale=1,
                    )
                    auto_reference_upscaler = gr.Dropdown(
                        label="Reference Frame Upscaler",
                        choices=spark_auto_ref_choices,
                        value=auto_reference_upscaler_value,
                        info="Default is SeedVR2. GAN models use the selected GAN weight; FlashVSR+ is available as the final option.",
                        scale=1,
                    )
                with gr.Accordion("Local Reference Backends", open=False):
                    gr.Markdown(
                        "`sr_image`: local default; blank path uses the input video. `pisasr`: strongest fully local reference path when PiSA-SR is installed. `no_ref`: baseline without reference guidance."
                    )
                    ref_pisa_cache_dir = gr.Textbox(
                        label="PiSA-SR Reference Cache Directory",
                        value=str(_value("ref_pisa_cache_dir", "") or ""),
                    )
                    pisa_python_executable = gr.Textbox(
                        label="PiSA Python Executable",
                        value=str(_value("pisa_python_executable", "") or ""),
                    )
                    pisa_script_path = gr.Textbox(
                        label="PiSA Script Path",
                        value=str(_value("pisa_script_path", "") or ""),
                    )
                    pisa_sd_model_path = gr.Textbox(
                        label="PiSA SD Model Path",
                        value=str(_value("pisa_sd_model_path", "") or ""),
                    )
                    pisa_chkpt_path = gr.Textbox(
                        label="PiSA Checkpoint Path",
                        value=str(_value("pisa_chkpt_path", "") or ""),
                    )
                    pisa_gpu = gr.Textbox(
                        label="PiSA GPU",
                        value=str(_value("pisa_gpu", "0") or "0"),
                    )
            
            # Quality / I/O Settings
            gr.Markdown("#### Output / I/O Settings")
            
            with gr.Group():
                with gr.Row():
                    png_save = gr.Checkbox(
                        label="Save PNG Frames",
                        value=bool(_value("png_save", False)),
                        info="Also save restored frames as PNG files beside the video output.",
                    )
                    save_format = gr.Dropdown(
                        label="Video Pixel Format",
                        choices=list(SPARKVSR_SAVE_FORMAT_OPTIONS),
                        value=(
                            str(_value("save_format", "yuv444p"))
                            if str(_value("save_format", "yuv444p")) in set(SPARKVSR_SAVE_FORMAT_OPTIONS)
                            else "yuv444p"
                        ),
                        info="Official default is yuv444p. yuv420p is smaller and more compatible.",
                    )
                    start_frame = gr.Number(
                        label="Start Frame",
                        value=int(_value("start_frame", 0) or 0),
                        precision=0,
                        info="0-indexed start frame for partial processing."
                    )
                    end_frame = gr.Number(
                        label="End Frame",
                        value=int(_value("end_frame", -1) or -1),
                        precision=0,
                        info="-1 = process until the end of input."
                    )

                gr.Markdown(
                    "**Video Codec / CRF / FPS Override:** Controlled from `Output > Video Output` "
                    "and automatically applied here."
                )
                # Keep schema compatibility fields without exposing duplicate controls.
                fps_SparkVSR = gr.State(float(_value("fps", 0.0) or 0.0))
                codec = gr.State(str(_value("codec", "libx264") or "libx264"))
                crf = gr.State(int(_value("crf", _value("quality", 18)) or 18))

                models_dir = gr.Textbox(
                    label="SparkVSR Models Directory",
                    value=str(_value("models_dir", str(base_dir / "SparkVSR" / "models")) or ""),
                    placeholder="G:/.../SparkVSR/models",
                    info=(
                        "Folder containing SparkVSR Diffusers model folders, normally downloaded by Models_Downloader.py."
                    ),
                )

                output_override = gr.Textbox(
                    label="Output Override (folder or .mp4 file)",
                    value=_value("output_override", ""),
                    placeholder="Leave empty for auto naming",
                    info="Optional custom output location. A folder saves into that folder. A .mp4 file path renames the final output to that exact file.",
                )

                with gr.Row():
                    output_format = gr.Dropdown(
                        label="Output Format",
                        choices=["mp4"],
                        value=str(_value("output_format", "mp4") or "mp4"),
                        info="SparkVSR CLI outputs MP4.",
                        interactive=False,
                    )
                    save_metadata = gr.Checkbox(
                        label="Save Processing Metadata",
                        value=bool(_value("save_metadata", True)),
                        info="Save run metadata to output folder."
                    )
                    face_restore_after_upscale = gr.Checkbox(
                        label="Apply Face Restoration after upscale",
                        value=bool(_value("face_restore_after_upscale", False)),
                        info="Per-run face restore toggle. Also respects Global Settings face toggle."
                    )

            # UNIVERSAL PRESET MANAGEMENT
            (
                preset_dropdown,
                preset_name_input,
                save_preset_btn,
                load_preset_btn,
                preset_status,
                reset_defaults_btn,
                delete_preset_btn,
                preset_callbacks,
            ) = universal_preset_section(
                preset_manager=preset_manager,
                shared_state=shared_state,
                tab_name="sparkvsr",
                inputs_list=[],
                base_dir=base_dir,
                models_list=models_list,
                open_accordion=True,
            )
        
        # Right Column: Output & Controls
        with gr.Column(scale=2):
            gr.Markdown("####  Output & Actions")
            status_box = gr.Markdown(value="Ready.", visible=False, elem_classes=["runtime-status-box"])
            progress_indicator = gr.Markdown(value="", visible=False, elem_classes=["runtime-progress-box"])
            with gr.Group(visible=False, elem_classes=["autotune-modal-overlay"]) as flash_autotune_notice_modal:
                with gr.Group(elem_classes=["autotune-modal-card"]):
                    with gr.Row(elem_classes=["autotune-modal-header"]):
                        gr.Markdown("Auto Tune Update", elem_classes=["autotune-modal-title"])
                        flash_autotune_notice_close_btn = gr.Button("X", size="sm", elem_classes=["autotune-modal-close"])
                    flash_autotune_notice_text = gr.Markdown("", elem_classes=["autotune-modal-body"])
                    with gr.Row(elem_classes=["autotune-modal-actions"]):
                        flash_autotune_notice_ok_btn = gr.Button("OK", variant="primary", elem_classes=["autotune-modal-ok"])

            with gr.Group():
                _upscale_factor_default = _value("upscale_factor", _value("scale", 4))
                try:
                    _upscale_factor_default = float(_upscale_factor_default)
                except Exception:
                    _upscale_factor_default = 4.0
                _upscale_factor_default = 2.0 if _upscale_factor_default <= 3.0 else 4.0

                _initial_use_global_scale = bool(_value("use_resolution_tab", True))
                _initial_shared_scale = resolve_shared_upscale_factor(
                    shared_state.value if _initial_use_global_scale else None
                )
                if _initial_use_global_scale and _initial_shared_scale is not None:
                    _upscale_factor_default = float(
                        canonical_sparkvsr_scale(
                            scale_value=None,
                            upscale_factor_value=_initial_shared_scale,
                            default=_upscale_factor_default,
                        )
                    )

                _max_resolution_default = _value("max_target_resolution", 1920)
                try:
                    _max_resolution_default = int(_max_resolution_default)
                except Exception:
                    _max_resolution_default = 1920
                _max_resolution_default = min(8192, max(0, _max_resolution_default))

                with gr.Row():
                    upscale_factor = gr.Slider(
                        label="SparkVSR Upscale Factor",
                        minimum=1.0,
                        maximum=16.0,
                        step=1.0,
                        value=_upscale_factor_default,
                        info="Integer scale passed to SparkVSR. Official default is 4x.",
                        interactive=True,
                        scale=2,
                    )
                    max_target_resolution = gr.Slider(
                        label="Max Resolution (max edge, 0 = no cap)",
                        minimum=0,
                        maximum=8192,
                        step=16,
                        value=_max_resolution_default,
                        info="Caps the LONG side (max(width,height)) of the target. 0 = unlimited.",
                        scale=2,
                    )
                    pre_downscale_then_upscale = gr.Checkbox(
                        label="Pre-downscale then upscale (when capped)",
                        value=bool(_value("pre_downscale_then_upscale", True)),
                        info="If max edge would reduce effective scale, downscale input first so the model still runs at the full Upscale x.",
                        scale=1,
                    )

                use_resolution_tab = gr.Checkbox(
                    label="Use Resolution & Scene Split Tab Settings",
                    value=bool(_value("use_resolution_tab", True)),
                    info=(
                        "When enabled, Upscale x follows shared app scale cache (e.g., SeedVR2/Resolution workflow). "
                        "Moving this local slider will switch this toggle OFF and use local sizing."
                    ),
                )

                with gr.Row():
                    upscale_btn = gr.Button(
                        " Start Upscaling",
                        variant="primary",
                        size="lg",
                        elem_classes=["action-btn", "action-btn-upscale"],
                    )
                    preview_btn = gr.Button(
                        "Preview First Frame",
                        size="lg",
                        elem_classes=["action-btn", "action-btn-preview"],
                    )

                with gr.Row():
                    cancel_confirm = gr.Checkbox(
                        label=" Confirm cancel (required for safety)",
                        value=False,
                        info="Enable this checkbox to confirm cancellation of processing",
                        scale=3,
                    )
                    cancel_btn = gr.Button(
                        " Cancel",
                        variant="stop",
                        size="md",
                        min_width=170,
                        scale=1,
                        elem_classes=["action-btn", "action-btn-cancel"],
                    )

            resume_run_dir = gr.Textbox(
                label="Resume Run Folder (chunk/scene resume)",
                value=(
                    _value("resume_run_dir", "")
                    if "resume_run_dir" in SPARKVSR_ORDER and len(values) > SPARKVSR_ORDER.index("resume_run_dir")
                    else ""
                ),
                placeholder="Optional: G:/.../outputs/0019",
                info=(
                    "Optional. When set, chunk/scene processing resumes from the last completed chunk in that folder. "
                    "Use the same settings as the original run to continue remaining chunks. "
                    "Fresh output path overrides are ignored while resume is active."
                ),
            )
            
            with gr.Accordion(" Upscaled Output", open=True):
                output_video = gr.Video(
                    label=" Upscaled Video",
                    interactive=False,
                    visible=False,
                    buttons=["download"],
                )
                output_image = gr.Image(
                    label=" Upscaled Image",
                    interactive=False,
                    visible=False,
                    buttons=["download"],
                )
            
            # Comparison
            image_slider = gr.ImageSlider(
                label=" Comparison",
                interactive=False,
                slider_position=50,
                max_height=1000,
                buttons=["download", "fullscreen"],
                elem_classes=["native-image-comparison-slider"],
            )
            
            video_comparison_html = gr.HTML(
                label=" Video Comparison",
                value="",
                js_on_load=get_video_comparison_js_on_load(),
                visible=False
            )
            
            chunk_status = gr.Markdown("", visible=False)
            chunk_gallery = gr.Gallery(
                label=" Chunk Preview",
                visible=False,
                columns=4,
                rows=2,
                height=220,
                object_fit="contain",
            )
            chunk_preview_video = gr.Video(
                label=" Selected Chunk",
                interactive=False,
                visible=False,
                buttons=["download"],
            )
            batch_gallery = gr.Gallery(
                label=" Batch Results",
                visible=False,
                columns=4,
                rows=2,
                height="auto",
                object_fit="contain",
                buttons=["download"],
            )
            last_processed = gr.Markdown("Batch processing results will appear here.")
            
            # Utility buttons
            with gr.Row():
                open_outputs_btn = gr.Button(
                    " Open Outputs",
                    elem_classes=["action-btn", "action-btn-open"],
                )
                clear_temp_btn = gr.Button(
                    " Clear Temp",
                    elem_classes=["action-btn", "action-btn-clear"],
                )

            with gr.Accordion(" Batch Processing", open=True):
                with gr.Row():
                    batch_enable = gr.Checkbox(
                        label="Enable Batch",
                        value=bool(_value("batch_enable", False)),
                        info="Process multiple files",
                        scale=2,
                    )
                    keep_only_output_files = gr.Checkbox(
                        label="Keep only output files",
                        value=bool(_value("keep_only_output_files", False)),
                        info="After batch completion, remove metadata/chunks/temp artifacts and keep only final outputs.",
                        scale=2,
                    )
                batch_input = gr.Textbox(
                    label="Batch Input Folder",
                    value=_value("batch_input_path", ""),
                    placeholder="Folder with videos"
                )
                batch_output = gr.Textbox(
                    label="Batch Output Folder",
                    value=_value("batch_output_path", ""),
                    placeholder="Output directory"
                )
            
            log_box = gr.Textbox(
                label=" Processing Log",
                value="",
                lines=12,
                buttons=["copy"]
            )
            
            # Info
            gr.Markdown("""
            #### SparkVSR Guide

            **Model defaults**
            - `SparkVSR-S2` is the official final-stage checkpoint and is selected by default.
            - `sr_image` is selected by default. If `Local SR Reference Path` is blank, the input video is used automatically as the local reference source.
            - Enable `Auto Upscale First Frame per Chunk` to generate one local SR reference per chunk before SparkVSR processing starts.
            - Best local quality comes from `sr_image` with a locally upscaled keyframe/video, or `pisasr` when PiSA-SR is installed. `no_ref` is only a baseline.

            **Runtime notes**
            - SparkVSR supports integer upscale factors. The default is **4x**.
            - Use `Max Resolution` + `Pre-downscale then upscale` to keep output size safe on limited VRAM.
            - For long videos, combine this tab with Resolution tab chunking (scene split or fixed chunk seconds).
            - Spatial tiling uses output-resolution tiles; `0 x 0` disables tiling.
            - CPU offload and VAE tiling are enabled by default for safer redistribution across mixed GPUs.
            """)
    
    # Collect inputs
    inputs_list = [
        input_path, output_override, output_format,
        model_name, model_path, lora_path,
        scale, precision, upscale_mode,
        noise_step, sr_noise_step, cpu_offload, vae_tiling, group_offload, num_blocks_per_group,
        tile_height, tile_width, overlap_height, overlap_width,
        chunk_len, overlap_t, ref_mode, ref_indices, ref_guidance_scale,
        ref_source_path, auto_reference_prepass, auto_reference_upscaler,
        ref_pisa_cache_dir, pisa_python_executable, pisa_script_path,
        pisa_sd_model_path, pisa_chkpt_path, pisa_gpu, png_save, save_format,
        force_offload, enable_debug, seed, auto_transfer_output_to_input, device, fps_SparkVSR,
        codec, crf, start_frame, end_frame, models_dir,
        save_metadata, face_restore_after_upscale, batch_enable, batch_input, batch_output,
        use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale,
        resume_run_dir, save_vram_gb, keep_only_output_files,
    ]

    # Development validation: inputs_list must stay aligned with SPARKVSR_ORDER
    if len(inputs_list) != len(SPARKVSR_ORDER):
        import logging
        logging.getLogger("SparkVSRTab").error(
            f"ERROR: inputs_list ({len(inputs_list)}) != SPARKVSR_ORDER ({len(SPARKVSR_ORDER)})"
        )
    
    # Wire up events
    def _analysis_progress_note(state: Dict[str, Any], pct: int) -> str:
        seed_controls = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
        scene_mode = bool(seed_controls.get("auto_chunk", True)) and bool(seed_controls.get("auto_detect_scenes", True))
        if pct < 15:
            return "Reading media metadata..."
        if pct < 35:
            return "Computing resize target..."
        if pct < 70:
            return "Scanning scenes for chunk stats..." if scene_mode else "Building runtime summary..."
        if pct < 95:
            return "Preparing analysis panel..."
        return "Finalizing analysis..."

    def _analysis_banner_html(state: Dict[str, Any], progress_pct: int, progress_note: str = "") -> str:
        safe_pct = max(0, min(100, int(progress_pct)))
        seed_controls = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
        scene_mode = bool(seed_controls.get("auto_chunk", True)) and bool(seed_controls.get("auto_detect_scenes", True))
        title = "Analyzing input (resolution + scene detection)" if scene_mode else "Analyzing input"
        sub = f"{safe_pct}%"
        if progress_note:
            sub = f"{sub}<br>{html.escape(str(progress_note))}"
        return (
            '<div class="processing-banner">'
            '<div class="processing-spinner"></div>'
            '<div class="processing-col">'
            f'<div class="processing-text">{html.escape(title)}</div>'
            f'<div class="processing-sub">{sub}</div>'
            "</div></div>"
        )

    def _build_input_detection_md(path_val: str) -> gr.update:
        from shared.input_detector import detect_input
        if not path_val or not str(path_val).strip():
            # Hide when empty (clearing input should clear this panel).
            return gr.update(value="", visible=False)
        try:
            info = detect_input(path_val)
            if not info.is_valid:
                return gr.update(value=f"ERROR: **Invalid Input**\n\n{info.error_message}", visible=True)
            parts = [f"OK: **Input Detected: {info.input_type.upper()}**"]
            if info.input_type == "frame_sequence":
                parts.append(f"&nbsp;&nbsp;Pattern: `{info.frame_pattern}`")
                parts.append(f"&nbsp;&nbsp;Frames: {info.frame_start}-{info.frame_end}")
                if info.missing_frames:
                    parts.append(f"&nbsp;&nbsp;Missing: {len(info.missing_frames)}")
            elif info.input_type == "directory":
                parts.append(f"&nbsp;&nbsp;Files: {info.total_files}")
            elif info.input_type in ["video", "image"]:
                parts.append(f"&nbsp;&nbsp;Format: **{info.format.upper()}**")
            return gr.update(value=" ".join(parts), visible=True)
        except Exception as e:
            return gr.update(value=f"ERROR: **Detection Error**\n\n{str(e)}", visible=True)

    def _build_sizing_info(path_val, model_scale_val, use_global, local_scale_x, local_max_edge, local_pre_down, state):
        resolved_scale = resolve_sparkvsr_effective_scale(
            scale_state_val=model_scale_val,
            use_global=bool(use_global),
            local_upscale_factor=local_scale_x,
            state=state if isinstance(state, dict) else None,
        )
        ms = int(resolved_scale or 4)
        return build_fixed_scale_analysis_update(
            input_path_val=path_val,
            model_scale=ms,
            # Scale is resolved locally with Resolution-tab fallback and integer guardrails.
            # so the preview cannot desync from hidden backend state.
            use_global=False,
            local_scale_x=float(ms),
            local_max_edge=int(local_max_edge or 0),
            local_pre_down=bool(local_pre_down),
            state=state,
            model_label="sparkvsr",
            runtime_label=f"SparkVSR pipeline (fixed {ms}x pass)",
            auto_scene_scan=True,
        )

    def _run_analysis_payload(path_val, scale_val, use_global, scale_x, max_edge, pre_down, state):
        det = _build_input_detection_md(path_val or "")
        info = _build_sizing_info(path_val or "", int(scale_val), bool(use_global), scale_x, max_edge, pre_down, state)
        img_prev, vid_prev = preview_updates(path_val)
        return img_prev, vid_prev, det, info, state

    def _iter_analysis_with_progress(path_val, scale_val, use_global, scale_x, max_edge, pre_down, state):
        result: Dict[str, Any] = {}

        def _worker():
            try:
                result["payload"] = _run_analysis_payload(path_val, scale_val, use_global, scale_x, max_edge, pre_down, state)
            except Exception as exc:
                result["error"] = exc

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        fallback_steps = [4, 10, 18, 26, 35, 46, 58, 70, 80, 88, 94]
        step_idx = 0
        last_emit = 0.0

        while worker.is_alive():
            now = time.monotonic()
            if now - last_emit >= 0.2:
                pct = fallback_steps[min(step_idx, len(fallback_steps) - 1)]
                if step_idx < len(fallback_steps) - 1:
                    step_idx += 1
                yield "progress", pct, _analysis_progress_note(state, pct)
                last_emit = now
            time.sleep(0.05)

        worker.join()
        if "error" in result:
            raise result["error"]
        yield "progress", 100, _analysis_progress_note(state, 100)
        yield "result", result.get("payload"), ""

    def cache_input_upload(val, scale_val, use_global, scale_x, max_edge, pre_down, state):
        try:
            state = state or {}
            state.setdefault("seed_controls", {})
            state["seed_controls"]["last_input_path"] = val if val else ""
        except Exception:
            pass

        if not val:
            img_prev, vid_prev = preview_updates(None)
            yield (
                "",
                gr.update(value="", visible=False),
                img_prev,
                vid_prev,
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                state,
            )
            return

        yield (
            val or "",
            gr.update(value="", visible=False),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.update(value=_analysis_banner_html(state, 0, _analysis_progress_note(state, 0)), visible=True),
            state,
        )

        try:
            for event_type, payload_a, _ in _iter_analysis_with_progress(
                val, scale_val, use_global, scale_x, max_edge, pre_down, state
            ):
                if event_type == "progress":
                    pct = int(payload_a)
                    yield (
                        val or "",
                        gr.update(value="", visible=False),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.update(value=_analysis_banner_html(state, pct, _analysis_progress_note(state, pct)), visible=True),
                        state,
                    )
                    continue

                img_prev, vid_prev, det, info, state_out = payload_a
                yield (
                    val or "",
                    gr.update(value="OK: Input cached for processing.", visible=True),
                    img_prev,
                    vid_prev,
                    det,
                    info,
                    state_out,
                )
                return
        except Exception as e:
            img_prev, vid_prev = preview_updates(val)
            yield (
                val or "",
                gr.update(value=f"Input cached (analysis error: {str(e)[:120]})", visible=True),
                img_prev,
                vid_prev,
                _build_input_detection_md(val or ""),
                gr.update(value="", visible=False),
                state,
            )

    def refresh_tile_count(
        path_val,
        tile_height_val,
        tile_width_val,
        overlap_height_val,
        overlap_width_val,
        scale_val,
        use_global,
        local_scale_x,
        max_edge,
        pre_down,
        state,
    ):
        return _tile_count_info_update(
            path_val,
            tile_height_val,
            tile_width_val,
            overlap_height_val,
            overlap_width_val,
            scale_val,
            use_global,
            local_scale_x,
            max_edge,
            pre_down,
            state,
        )

    tile_count_inputs = [
        input_path,
        tile_height,
        tile_width,
        overlap_height,
        overlap_width,
        scale,
        use_resolution_tab,
        upscale_factor,
        max_target_resolution,
        pre_downscale_then_upscale,
        shared_state,
    ]

    upload_evt = input_file.upload(
        fn=cache_input_upload,
        inputs=[input_file, scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state]
    )
    upload_evt.then(
        fn=refresh_tile_count,
        inputs=tile_count_inputs,
        outputs=[dit_tile_count],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    # When upload is cleared, also clear the textbox path and hide dependent panels.
    def clear_input_path_on_upload_clear(file_path, state):
        if file_path:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), state
        try:
            state = state or {}
            state.setdefault("seed_controls", {})
            state["seed_controls"]["last_input_path"] = ""
        except Exception:
            pass
        img_prev, vid_prev = preview_updates(None)
        return (
            "",
            gr.update(value="", visible=False),
            img_prev,
            vid_prev,
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            state,
        )

    upload_clear_evt = input_file.change(
        fn=clear_input_path_on_upload_clear,
        inputs=[input_file, shared_state],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )
    upload_clear_evt.then(
        fn=refresh_tile_count,
        inputs=tile_count_inputs,
        outputs=[dit_tile_count],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    def cache_input_path(path_val, scale_val, use_global, scale_x, max_edge, pre_down, state):
        try:
            state = state or {}
            state.setdefault("seed_controls", {})
            state["seed_controls"]["last_input_path"] = path_val if path_val else ""
        except Exception:
            pass

        if not path_val or not str(path_val).strip():
            img_prev, vid_prev = preview_updates(None)
            yield (
                gr.update(value="", visible=False),
                img_prev,
                vid_prev,
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                state,
            )
            return

        yield (
            gr.update(value="", visible=False),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.update(value=_analysis_banner_html(state, 0, _analysis_progress_note(state, 0)), visible=True),
            state,
        )

        try:
            for event_type, payload_a, _ in _iter_analysis_with_progress(
                path_val, scale_val, use_global, scale_x, max_edge, pre_down, state
            ):
                if event_type == "progress":
                    pct = int(payload_a)
                    yield (
                        gr.update(value="", visible=False),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.update(value=_analysis_banner_html(state, pct, _analysis_progress_note(state, pct)), visible=True),
                        state,
                    )
                    continue

                img_prev, vid_prev, det, info, state_out = payload_a
                yield (
                    gr.update(value="OK: Input path updated.", visible=True),
                    img_prev,
                    vid_prev,
                    det,
                    info,
                    state_out,
                )
                return
        except Exception as e:
            img_prev, vid_prev = preview_updates(path_val)
            yield (
                gr.update(value=f"Input path updated (analysis error: {str(e)[:120]})", visible=True),
                img_prev,
                vid_prev,
                _build_input_detection_md(path_val or ""),
                gr.update(value="", visible=False),
                state,
            )

    def _output_path_signature(path_val):
        normalized = normalize_path(path_val) if path_val else ""
        if not normalized:
            return ""
        try:
            cand = Path(normalized)
            if not cand.exists() or not cand.is_file():
                return ""
            stat = cand.stat()
            return f"{normalized}|{int(stat.st_size)}|{int(stat.st_mtime_ns)}"
        except Exception:
            return ""

    def _resolve_latest_output_path(state):
        seed_controls = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
        candidates = []

        last_output = normalize_path(seed_controls.get("last_output_path")) if seed_controls.get("last_output_path") else ""
        if last_output:
            candidates.append(last_output)

        batch_outputs = seed_controls.get("sparkvsr_batch_outputs", [])
        if isinstance(batch_outputs, list):
            for item in reversed(batch_outputs):
                normalized = normalize_path(item) if item else ""
                if normalized:
                    candidates.append(normalized)

        seen = set()
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            try:
                if Path(cand).exists():
                    return cand
            except Exception:
                continue
        return ""

    def _apply_output_path_to_input(
        output_path_val,
        scale_val,
        use_global,
        scale_x,
        max_edge,
        pre_down,
        state,
        source_label="Output transferred to input.",
    ):
        state = state or {}
        state.setdefault("seed_controls", {})
        state["seed_controls"]["last_input_path"] = output_path_val or ""

        try:
            img_prev, vid_prev, det, info, state_out = _run_analysis_payload(
                output_path_val or "",
                scale_val,
                use_global,
                scale_x,
                max_edge,
                pre_down,
                state,
            )
            return (
                output_path_val or "",
                gr.update(value=f"OK: {source_label}", visible=True),
                img_prev,
                vid_prev,
                det,
                info,
                state_out,
            )
        except Exception as exc:
            img_prev, vid_prev = preview_updates(output_path_val)
            return (
                output_path_val or "",
                gr.update(
                    value=f"OK: {source_label} Analysis warning: {str(exc)[:120]}",
                    visible=True,
                ),
                img_prev,
                vid_prev,
                _build_input_detection_md(output_path_val or ""),
                gr.update(value="", visible=False),
                state,
            )

    def copy_latest_output_to_input(scale_val, use_global, scale_x, max_edge, pre_down, state):
        state = state if isinstance(state, dict) else {}
        output_path_val = _resolve_latest_output_path(state)
        if not output_path_val:
            return (
                gr.skip(),
                gr.update(value="[WARN] No generated output found to transfer.", visible=True),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                state,
            )
        return _apply_output_path_to_input(
            output_path_val,
            scale_val,
            use_global,
            scale_x,
            max_edge,
            pre_down,
            state,
            source_label="Output path copied into input.",
        )

    def capture_latest_output_signature(state):
        return _output_path_signature(_resolve_latest_output_path(state))

    def auto_transfer_latest_output_to_input(
        scale_val,
        use_global,
        scale_x,
        max_edge,
        pre_down,
        auto_enabled,
        previous_signature,
        state,
    ):
        state = state if isinstance(state, dict) else {}
        if not bool(auto_enabled):
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                state,
            )

        output_path_val = _resolve_latest_output_path(state)
        if not output_path_val:
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                state,
            )

        latest_signature = _output_path_signature(output_path_val)
        if previous_signature and latest_signature and str(previous_signature) == str(latest_signature):
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                state,
            )

        return _apply_output_path_to_input(
            output_path_val,
            scale_val,
            use_global,
            scale_x,
            max_edge,
            pre_down,
            state,
            source_label="Auto-transferred latest output into input.",
        )

    # Keep heavy media/path analysis user-triggered only.
    # `.change()` also fires on function updates (e.g. tab sync), which can fan out
    # into expensive cascades on large UIs.
    input_submit_evt = input_path.submit(
        fn=cache_input_path,
        inputs=[input_path, scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )
    input_submit_evt.then(
        fn=refresh_tile_count,
        inputs=tile_count_inputs,
        outputs=[dit_tile_count],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    copy_output_evt = copy_output_into_input_btn.click(
        fn=copy_latest_output_to_input,
        inputs=[scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )
    copy_output_evt.then(
        fn=refresh_tile_count,
        inputs=tile_count_inputs,
        outputs=[dit_tile_count],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    input_path.change(
        fn=refresh_tile_count,
        inputs=tile_count_inputs,
        outputs=[dit_tile_count],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    def refresh_sizing(path_val, scale_val, use_global, local_scale_x, max_edge, pre_down, state):
        return _build_sizing_info(path_val, int(scale_val), bool(use_global), local_scale_x, max_edge, pre_down, state)

    scale_evt = scale.change(
        fn=refresh_sizing,
        inputs=[input_path, scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[sizing_info],
        trigger_mode="always_last",
    )
    scale_evt.then(
        fn=refresh_tile_count,
        inputs=tile_count_inputs,
        outputs=[dit_tile_count],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    pre_down_evt = pre_downscale_then_upscale.change(
        fn=refresh_sizing,
        inputs=[input_path, scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[sizing_info],
        trigger_mode="always_last",
    )
    pre_down_evt.then(
        fn=refresh_tile_count,
        inputs=tile_count_inputs,
        outputs=[dit_tile_count],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    max_res_evt = max_target_resolution.release(
        fn=refresh_sizing,
        inputs=[input_path, scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[sizing_info],
        preprocess=False,
        trigger_mode="always_last",
    )
    max_res_evt.then(
        fn=refresh_tile_count,
        inputs=tile_count_inputs,
        outputs=[dit_tile_count],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    def _sync_upscale_ui(use_global, local_x, state):
        shared_scale = resolve_shared_upscale_factor(state if bool(use_global) else None)
        if bool(use_global) and shared_scale is not None:
            effective = canonical_sparkvsr_scale(
                scale_value=None,
                upscale_factor_value=shared_scale,
                default=local_x if local_x is not None else 4,
            )
            return (
                gr.update(value=float(effective), interactive=True),
                gr.update(value=str(int(effective))),
            )

        effective_local = canonical_sparkvsr_scale(
            scale_value=None,
            upscale_factor_value=local_x,
            default=4,
        )
        return (
            gr.update(value=float(effective_local), interactive=True),
            gr.update(value=str(int(effective_local))),
        )

    def _sync_upscale_ui_and_sizing(use_global, local_x, path_val, max_edge, pre_down, state):
        slider_upd, scale_upd = _sync_upscale_ui(use_global, local_x, state)
        shared_scale = resolve_shared_upscale_factor(state if bool(use_global) else None)
        effective_scale = shared_scale if (bool(use_global) and shared_scale is not None) else local_x
        resolved = canonical_sparkvsr_scale(
            scale_value=None,
            upscale_factor_value=effective_scale,
            default=4,
        )
        info = _build_sizing_info(
            path_val or "",
            int(resolved),
            bool(use_global),
            float(resolved),
            max_edge,
            pre_down,
            state,
        )
        return slider_upd, scale_upd, info

    def _on_local_upscale_changed(local_x, use_global, state):
        resolved = canonical_sparkvsr_scale(
            scale_value=None,
            upscale_factor_value=local_x,
            default=4,
        )
        # If a shared/global scale is currently active, local slider interaction
        # switches to local mode instead of staying locked behind global mode.
        shared_scale = resolve_shared_upscale_factor(state if bool(use_global) else None)
        next_use_global = bool(use_global)
        if bool(use_global) and shared_scale is not None:
            next_use_global = False
        return gr.update(value=next_use_global), gr.update(value=str(int(resolved)))

    local_scale_evt = upscale_factor.release(
        fn=_on_local_upscale_changed,
        inputs=[upscale_factor, use_resolution_tab, shared_state],
        outputs=[use_resolution_tab, scale],
        preprocess=False,
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )
    local_scale_evt.then(
        fn=refresh_sizing,
        inputs=[input_path, scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[sizing_info],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )
    local_scale_evt.then(
        fn=refresh_tile_count,
        inputs=tile_count_inputs,
        outputs=[dit_tile_count],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    use_resolution_evt = use_resolution_tab.change(
        fn=_sync_upscale_ui_and_sizing,
        inputs=[use_resolution_tab, upscale_factor, input_path, max_target_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[upscale_factor, scale, sizing_info],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )
    use_resolution_evt.then(
        fn=refresh_tile_count,
        inputs=tile_count_inputs,
        outputs=[dit_tile_count],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    def refresh_chunk_preview_ui(state):
        preview = (state or {}).get("seed_controls", {}).get("sparkvsr_chunk_preview", {})
        if not isinstance(preview, dict):
            return gr.update(value="", visible=False), gr.update(value=[], visible=False), gr.update(value=None, visible=False)

        gallery = preview.get("gallery") or []
        videos = preview.get("videos") or []
        message = str(preview.get("message") or "")

        first_video = None
        for v in videos:
            if v and Path(v).exists():
                first_video = v
                break

        return (
            gr.update(value=message, visible=bool(message or gallery)),
            gr.update(value=gallery, visible=bool(gallery)),
            gr.update(value=first_video, visible=bool(first_video)),
        )

    def on_chunk_gallery_select(evt: gr.SelectData, state):
        try:
            idx = int(evt.index)
            videos = (state or {}).get("seed_controls", {}).get("sparkvsr_chunk_preview", {}).get("videos", [])
            if 0 <= idx < len(videos):
                cand = videos[idx]
                if cand and Path(cand).exists():
                    return gr.update(value=cand, visible=True)
        except Exception:
            pass
        return gr.update(value=None, visible=False)

    chunk_gallery.select(
        fn=on_chunk_gallery_select,
        inputs=[shared_state],
        outputs=[chunk_preview_video],
    )
    
    def _queue_status_indicator(title: str, subtitle: str, spinning: bool = True):
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

    def _extract_update_value(update_obj):
        try:
            if isinstance(update_obj, dict):
                return update_obj.get("value")
        except Exception:
            pass
        return None

    def _compact_single_line(text: Any, max_len: int = 120) -> str:
        raw = str(text or "")
        try:
            raw = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw)
        except Exception:
            pass
        raw = re.sub(r"\s+", " ", raw.replace("\r", " ").replace("\n", " ")).strip()
        if len(raw) > max_len:
            raw = raw[: max(0, max_len - 3)].rstrip() + "..."
        return raw

    def _log_tail_line(logs: Any) -> str:
        if not isinstance(logs, str):
            return ""
        for line in reversed(logs.splitlines()):
            compact = _compact_single_line(line)
            if compact:
                return compact
        return ""

    def _batch_gallery_update_from_state(state):
        outputs = (state or {}).get("seed_controls", {}).get("sparkvsr_batch_outputs", [])
        if not isinstance(outputs, list):
            outputs = []
        outputs = [str(p) for p in outputs if p and Path(str(p)).exists()]
        return gr.update(value=outputs, visible=bool(outputs))

    def _last_processed_text(state, vid_upd, img_upd) -> str:
        outputs = (state or {}).get("seed_controls", {}).get("sparkvsr_batch_outputs", [])
        if isinstance(outputs, list) and outputs:
            last_out = str(outputs[-1])
            return f"Batch results: {len(outputs)} item(s). Last output: {Path(last_out).name}"

        single = _extract_update_value(img_upd) or _extract_update_value(vid_upd)
        if single:
            return f"Output: {single}"
        return "Batch processing results will appear here."

    def _expand_service_payload(payload, live_state):
        merged = merge_payload_state(payload, live_state)
        if not isinstance(merged, tuple) or len(merged) < 7:
            safe_state = live_state if isinstance(live_state, dict) else {}
            return (
                gr.update(value="ERROR: Invalid SparkVSR payload"),
                "",
                gr.update(value="", visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                "Error",
                gr.update(value=None),
                gr.update(value="", visible=False),
                gr.update(value=[], visible=False),
                safe_state,
            )

        status, logs, vid_upd, img_upd, slider_upd, html_upd, state_out = merged
        status_text = _extract_update_value(status) if isinstance(status, dict) else status
        status_text = str(status_text or "").strip()
        log_tail = _log_tail_line(logs)
        status_lc = status_text.lower()
        log_lc = log_tail.lower()

        terminal_tokens = (
            "complete",
            "completed",
            "failed",
            "error",
            "critical",
            "cancel",
            "no result",
            "out of vram",
            "oom",
            "timed out",
            "timeout",
            "aborted",
            "input path missing",
            "input missing",
            "batch input folder missing",
            "resume folder not found",
            "resume input not found",
            "resume unavailable",
            "ffmpeg not found",
            "max resolution preprocess failed",
            "insufficient disk space",
        )
        is_terminal = any(tok in status_lc for tok in terminal_tokens) or any(
            tok in log_lc for tok in ("critical error", "processing failed", "cancelled", "out of vram")
        )

        if (status_text or log_tail) and not is_terminal:
            title = _compact_single_line(status_text or "SparkVSR processing...", max_len=96)
            subtitle = log_tail or "Processing..."
            progress_update = _queue_status_indicator(title, subtitle, spinning=True)
        else:
            progress_update = gr.update(value="", visible=False)

        return (
            status,
            logs,
            progress_update,
            img_upd if img_upd is not None else gr.update(value=None, visible=False),
            vid_upd if vid_upd is not None else gr.update(value=None, visible=False),
            _last_processed_text(state_out, vid_upd, img_upd),
            slider_upd if slider_upd is not None else gr.update(value=None),
            html_upd if html_upd is not None else gr.update(value="", visible=False),
            _batch_gallery_update_from_state(state_out),
            state_out,
        )

    def _starting_runtime_output(state, action_label: str):
        safe_state = state or {}
        title = f"{action_label} started"
        subtitle = "Initializing runtime and preparing input..."
        return (
            gr.update(value=title),
            gr.update(value=f"{action_label} requested. Preparing backend..."),
            _queue_status_indicator(title, subtitle, spinning=True),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            "Initializing...",
            gr.update(value=None),
            gr.update(value="", visible=False),
            _batch_gallery_update_from_state(safe_state),
            safe_state,
        )

    def _queued_waiting_output(state, ticket_id: str, position: int):
        safe_state = state or {}
        pos = max(1, int(position)) if position else "-"
        title = f"Queue waiting: {ticket_id} (position {pos})"
        subtitle = (
            f"Queued and waiting for active processing slot. Queue position: {pos}. "
            "Run logs and chunk previews will update once processing starts."
        )
        return (
            gr.update(value=title),
            gr.update(value=f"Queued and waiting for active processing slot. Queue position: {pos}."),
            _queue_status_indicator(title, subtitle, spinning=True),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            "Waiting in queue",
            gr.update(value=None),
            gr.update(value="", visible=False),
            _batch_gallery_update_from_state(safe_state),
            safe_state,
        )

    def _queued_cancelled_output(state, ticket_id: str):
        safe_state = state or {}
        title = f"Queue item removed: {ticket_id}"
        subtitle = "This queued request was removed before processing started."
        return (
            gr.update(value=title),
            gr.update(value=subtitle),
            _queue_status_indicator(title, subtitle, spinning=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            "Removed from queue",
            gr.update(value=None),
            gr.update(value="", visible=False),
            _batch_gallery_update_from_state(safe_state),
            safe_state,
        )

    def _queue_disabled_busy_output(state):
        safe_state = state or {}
        title = "Processing already in progress (queue disabled)."
        subtitle = "Enable 'Enable Queue' in Global Settings to stack additional requests."
        return (
            gr.update(value=title),
            gr.update(value=subtitle),
            _queue_status_indicator(title, subtitle, spinning=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            "Queue disabled: request ignored",
            gr.update(value=None),
            gr.update(value="", visible=False),
            _batch_gallery_update_from_state(safe_state),
            safe_state,
        )

    def run_upscale_with_queue(*args, progress=gr.Progress()):
        live_state = args[-1] if (args and isinstance(args[-1], dict)) else {}
        queued_state = snapshot_queue_state(live_state)
        queued_global_settings = snapshot_global_settings(global_settings)
        queue_enabled = bool(queued_global_settings.get("queue_enabled", True))
        queue_resource_keys, queue_resource_label = resolve_queue_gpu_resources(queued_state, queued_global_settings)
        ticket = queue_manager.submit(
            "sparkvsr",
            "Upscale",
            resource_keys=queue_resource_keys,
            resource_label=queue_resource_label,
        )
        acquired_slot = queue_manager.is_active(ticket.job_id)

        try:
            yield _starting_runtime_output(live_state, "Upscale")

            if not queue_enabled:
                if not acquired_slot:
                    queue_manager.cancel_waiting([ticket.job_id])
                    yield _queue_disabled_busy_output(live_state)
                    return
                for payload in service["run_action"](
                    args[0],
                    *args[1:-1],
                    preview_only=False,
                    state=queued_state,
                    progress=progress,
                    global_settings_snapshot=queued_global_settings,
                ):
                    yield _expand_service_payload(payload, live_state)
                return

            wait_notice_sent = False
            while not ticket.start_event.wait(timeout=0.5):
                if ticket.cancel_event.is_set():
                    yield _queued_cancelled_output(live_state, ticket.job_id)
                    return
                if not wait_notice_sent:
                    try:
                        pos = queue_manager.waiting_position(ticket.job_id)
                        pos_text = max(1, int(pos)) if pos else "-"
                        gr.Info(f"Queued: {ticket.job_id} (position {pos_text})")
                    except Exception:
                        pass
                    wait_notice_sent = True

            if ticket.cancel_event.is_set() and not queue_manager.is_active(ticket.job_id):
                yield _queued_cancelled_output(live_state, ticket.job_id)
                return

            acquired_slot = True
            for payload in service["run_action"](
                args[0],
                *args[1:-1],
                preview_only=False,
                state=queued_state,
                progress=progress,
                global_settings_snapshot=queued_global_settings,
            ):
                yield _expand_service_payload(payload, live_state)
        finally:
            if acquired_slot:
                queue_manager.complete(ticket.job_id)
            else:
                queue_manager.cancel_waiting([ticket.job_id])

    def run_preview_with_snapshot(*args, progress=gr.Progress()):
        live_state = args[-1] if (args and isinstance(args[-1], dict)) else {}
        queued_state = snapshot_queue_state(live_state)
        queued_global_settings = snapshot_global_settings(global_settings)
        yield _starting_runtime_output(live_state, "Preview")
        for payload in service["run_action"](
            args[0],
            *args[1:-1],
            preview_only=True,
            state=queued_state,
            progress=progress,
            global_settings_snapshot=queued_global_settings,
        ):
            yield _expand_service_payload(payload, live_state)

    def _autotune_waiting_output(state, ticket_id: str, position: int):
        safe_state = state or {}
        pos = max(1, int(position)) if position else "-"
        title = f"Queue waiting: {ticket_id} (position {pos})"
        subtitle = "Auto Tune is queued and will start when a processing slot becomes available."
        return (
            gr.update(value=title),
            gr.update(value=f"Queued and waiting for processing slot. Queue position: {pos}."),
            _queue_status_indicator(title, subtitle, spinning=True),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            safe_state,
        )

    def _autotune_cancelled_output(state, ticket_id: str):
        safe_state = state or {}
        title = f"Queue item removed: {ticket_id}"
        subtitle = "This Auto Tune request was removed before it started."
        return (
            gr.update(value=title),
            gr.update(value=subtitle),
            _queue_status_indicator(title, subtitle, spinning=False),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            safe_state,
        )

    def _autotune_busy_output(state):
        safe_state = state or {}
        title = "Processing already in progress (queue disabled)."
        subtitle = "Enable queue in Global Settings to run Auto Tune after the active job."
        return (
            gr.update(value=title),
            gr.update(value=subtitle),
            _queue_status_indicator(title, subtitle, spinning=False),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            safe_state,
        )

    def _merge_autotune_payload_state(payload, live_state):
        merged = merge_payload_state(payload, live_state)
        if not (isinstance(payload, tuple) and payload):
            return merged
        worker_state = payload[-1] if isinstance(payload[-1], dict) else {}
        if not worker_state:
            return merged
        merged_state = merged[-1] if (isinstance(merged, tuple) and isinstance(merged[-1], dict)) else {}
        if not merged_state:
            return merged
        worker_seed = worker_state.get("seed_controls", {})
        merged_seed = merged_state.get("seed_controls", {})
        if not (isinstance(worker_seed, dict) and isinstance(merged_seed, dict)):
            return merged
        if "sparkvsr_settings" in worker_seed:
            try:
                merged_seed["sparkvsr_settings"] = dict(worker_seed.get("sparkvsr_settings") or {})
                merged_state["seed_controls"] = merged_seed
                return (*merged[:-1], merged_state)
            except Exception:
                return merged
        return merged

    def _autotune_modal_message(payload) -> str | None:
        if not (isinstance(payload, tuple) and payload):
            return None
        status_text = str(payload[0] or "").strip()
        if not status_text:
            return None
        status_lower = status_text.lower()
        if "reused a matching cached result" in status_lower:
            return (
                "Auto Tune found a matching cached config and applied it instantly.\n"
                "No new scan was needed for this input and GPU profile."
            )
        state_payload = payload[-1] if isinstance(payload[-1], dict) else {}
        operation_status = str(state_payload.get("operation_status") or "").strip().lower()
        if operation_status in {"completed", "ready", "error"} and status_lower.startswith("auto tune"):
            return f"{status_text}\nCheck Run Log for full details."
        return None

    def _with_autotune_modal(base_payload, payload=None, *, hide_if_no_message: bool = False):
        if not (isinstance(base_payload, tuple) and base_payload):
            return base_payload
        modal_message = _autotune_modal_message(payload if payload is not None else base_payload)
        if modal_message:
            return (
                *base_payload,
                gr.update(value=str(modal_message)),
                gr.update(visible=True),
            )
        if hide_if_no_message:
            return (
                *base_payload,
                gr.update(value=""),
                gr.update(visible=False),
            )
        return (*base_payload, gr.update(), gr.update())

    def run_autotune_wrapper(*args, progress=gr.Progress()):
        live_state = args[-1] if (args and isinstance(args[-1], dict)) else {}
        queued_state = snapshot_queue_state(live_state)
        queued_global_settings = snapshot_global_settings(global_settings)
        queue_enabled = bool(queued_global_settings.get("queue_enabled", True))
        queue_resource_keys, queue_resource_label = resolve_queue_gpu_resources(queued_state, queued_global_settings)
        ticket = queue_manager.submit(
            "sparkvsr",
            "Auto Tune",
            resource_keys=queue_resource_keys,
            resource_label=queue_resource_label,
        )
        acquired_slot = queue_manager.is_active(ticket.job_id)

        try:
            if not queue_enabled:
                if not acquired_slot:
                    queue_manager.cancel_waiting([ticket.job_id])
                    yield _with_autotune_modal(_autotune_busy_output(live_state), hide_if_no_message=True)
                    return
                for payload in service["auto_tune_action"](
                    *args[:-1],
                    state=queued_state,
                    progress=progress,
                    global_settings_snapshot=queued_global_settings,
                ):
                    merged_payload = _merge_autotune_payload_state(payload, live_state)
                    yield _with_autotune_modal(merged_payload, merged_payload, hide_if_no_message=True)
                return

            wait_notice_sent = False
            while not ticket.start_event.wait(timeout=0.5):
                if ticket.cancel_event.is_set():
                    yield _with_autotune_modal(
                        _autotune_cancelled_output(live_state, ticket.job_id),
                        hide_if_no_message=True,
                    )
                    return
                if not wait_notice_sent:
                    try:
                        pos = queue_manager.waiting_position(ticket.job_id)
                        pos_text = max(1, int(pos)) if pos else "-"
                        gr.Info(f"Queued: {ticket.job_id} (position {pos_text})")
                        yield _with_autotune_modal(
                            _autotune_waiting_output(live_state, ticket.job_id, int(pos) if pos else 0),
                            hide_if_no_message=True,
                        )
                    except Exception:
                        pass
                    wait_notice_sent = True

            if ticket.cancel_event.is_set() and not queue_manager.is_active(ticket.job_id):
                yield _with_autotune_modal(
                    _autotune_cancelled_output(live_state, ticket.job_id),
                    hide_if_no_message=True,
                )
                return

            acquired_slot = True
            for payload in service["auto_tune_action"](
                *args[:-1],
                state=queued_state,
                progress=progress,
                global_settings_snapshot=queued_global_settings,
            ):
                merged_payload = _merge_autotune_payload_state(payload, live_state)
                yield _with_autotune_modal(merged_payload, merged_payload, hide_if_no_message=True)
        finally:
            if acquired_slot:
                queue_manager.complete(ticket.job_id)
            else:
                queue_manager.cancel_waiting([ticket.job_id])

    SparkVSR_upscale_sync_signature = gr.State(value="")
    SparkVSR_chunk_sync_signature = gr.State(value="")
    SparkVSR_pre_run_output_signature = gr.State(value="")

    def _sync_signature(payload: Dict[str, Any]) -> str:
        try:
            blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str, separators=(",", ":"))
        except Exception:
            blob = str(payload)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def _sync_upscale_ui_and_sizing_if_needed(
        use_global,
        local_x,
        path_val,
        max_edge,
        pre_down,
        state,
        previous_signature: str = "",
    ):
        seed_controls = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
        signature = _sync_signature(
            {
                "use_global": bool(use_global),
                "local_x": local_x,
                "path_val": path_val,
                "max_edge": max_edge,
                "pre_down": bool(pre_down),
                "shared_scale": resolve_shared_upscale_factor(state if bool(use_global) else None),
                "resolution_settings": seed_controls.get("resolution_settings", {}),
            }
        )
        if signature == str(previous_signature or ""):
            return gr.skip(), gr.skip(), gr.skip(), previous_signature
        slider_upd, scale_upd, sizing_upd = _sync_upscale_ui_and_sizing(
            use_global, local_x, path_val, max_edge, pre_down, state
        )
        return slider_upd, scale_upd, sizing_upd, signature

    def _refresh_chunk_preview_ui_if_needed(state, previous_signature: str = ""):
        preview = (state or {}).get("seed_controls", {}).get("sparkvsr_chunk_preview", {})
        signature = _sync_signature(preview if isinstance(preview, dict) else {"preview": None})
        if signature == str(previous_signature or ""):
            return gr.skip(), gr.skip(), gr.skip(), previous_signature
        chunk_status_upd, chunk_gallery_upd, chunk_video_upd = refresh_chunk_preview_ui(state)
        return chunk_status_upd, chunk_gallery_upd, chunk_video_upd, signature

    # Main processing
    upscale_btn.click(
        fn=capture_latest_output_signature,
        inputs=[shared_state],
        outputs=[SparkVSR_pre_run_output_signature],
        queue=False,
        show_progress="hidden",
    )

    run_evt = upscale_btn.click(
        fn=run_upscale_with_queue,
        inputs=[input_file] + inputs_list + [shared_state],
        outputs=[
            status_box,
            log_box,
            progress_indicator,
            output_image,
            output_video,
            last_processed,
            image_slider,
            video_comparison_html,
            batch_gallery,
            shared_state,
        ],
        concurrency_limit=32,
        concurrency_id="app_processing_queue",
        trigger_mode="multiple",
    )
    run_evt.then(
        fn=_refresh_chunk_preview_ui_if_needed,
        inputs=[shared_state, SparkVSR_chunk_sync_signature],
        outputs=[chunk_status, chunk_gallery, chunk_preview_video, SparkVSR_chunk_sync_signature],
    )
    auto_transfer_evt = run_evt.then(
        fn=auto_transfer_latest_output_to_input,
        inputs=[
            scale,
            use_resolution_tab,
            upscale_factor,
            max_target_resolution,
            pre_downscale_then_upscale,
            auto_transfer_output_to_input,
            SparkVSR_pre_run_output_signature,
            shared_state,
        ],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )
    auto_transfer_evt.then(
        fn=refresh_tile_count,
        inputs=tile_count_inputs,
        outputs=[dit_tile_count],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    preview_evt = preview_btn.click(
        fn=run_preview_with_snapshot,
        inputs=[input_file] + inputs_list + [shared_state],
        outputs=[
            status_box,
            log_box,
            progress_indicator,
            output_image,
            output_video,
            last_processed,
            image_slider,
            video_comparison_html,
            batch_gallery,
            shared_state,
        ],
    )
    preview_evt.then(
        fn=_refresh_chunk_preview_ui_if_needed,
        inputs=[shared_state, SparkVSR_chunk_sync_signature],
        outputs=[chunk_status, chunk_gallery, chunk_preview_video, SparkVSR_chunk_sync_signature],
    )

    autotune_evt = auto_tune_btn.click(
        fn=run_autotune_wrapper,
        inputs=[input_file] + inputs_list + [shared_state],
        outputs=[
            status_box,
            log_box,
            progress_indicator,
            tile_height,
            tile_width,
            overlap_height,
            overlap_width,
            chunk_len,
            overlap_t,
            vae_tiling,
            optimize_summary,
            shared_state,
            flash_autotune_notice_text,
            flash_autotune_notice_modal,
        ],
        concurrency_limit=32,
        concurrency_id="app_processing_queue",
        trigger_mode="multiple",
    )
    autotune_evt.then(
        fn=refresh_tile_count,
        inputs=tile_count_inputs,
        outputs=[dit_tile_count],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    def _dismiss_flash_autotune_notice():
        return gr.update(value=""), gr.update(visible=False)

    flash_autotune_notice_ok_btn.click(
        fn=_dismiss_flash_autotune_notice,
        outputs=[flash_autotune_notice_text, flash_autotune_notice_modal],
        queue=False,
        show_progress="hidden",
    )
    flash_autotune_notice_close_btn.click(
        fn=_dismiss_flash_autotune_notice,
        outputs=[flash_autotune_notice_text, flash_autotune_notice_modal],
        queue=False,
        show_progress="hidden",
    )

    shared_scale_sync_evt = shared_state.change(
        fn=_sync_upscale_ui_and_sizing_if_needed,
        inputs=[use_resolution_tab, upscale_factor, input_path, max_target_resolution, pre_downscale_then_upscale, shared_state, SparkVSR_upscale_sync_signature],
        outputs=[upscale_factor, scale, sizing_info, SparkVSR_upscale_sync_signature],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )
    shared_scale_sync_evt.then(
        fn=refresh_tile_count,
        inputs=tile_count_inputs,
        outputs=[dit_tile_count],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    shared_state.change(
        fn=_refresh_chunk_preview_ui_if_needed,
        inputs=[shared_state, SparkVSR_chunk_sync_signature],
        outputs=[chunk_status, chunk_gallery, chunk_preview_video, SparkVSR_chunk_sync_signature],
        queue=False,
        show_progress="hidden",
    )

    tile_height.change(
        fn=refresh_tile_count,
        inputs=tile_count_inputs,
        outputs=[dit_tile_count],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    tile_width.change(
        fn=refresh_tile_count,
        inputs=tile_count_inputs,
        outputs=[dit_tile_count],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    overlap_height.change(
        fn=refresh_tile_count,
        inputs=tile_count_inputs,
        outputs=[dit_tile_count],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    overlap_width.change(
        fn=refresh_tile_count,
        inputs=tile_count_inputs,
        outputs=[dit_tile_count],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    def _cancel_with_confirmation_reset(ok):
        if ok:
            status_upd, log_msg = service["cancel_action"]()
            return status_upd, log_msg, gr.update(value=False)
        return gr.update(value="WARNING: Enable 'Confirm cancel' to stop."), "", gr.update(value=False)

    cancel_btn.click(
        fn=_cancel_with_confirmation_reset,
        inputs=[cancel_confirm],
        outputs=[status_box, log_box, cancel_confirm]
    )
    
    open_outputs_btn.click(
        fn=service["open_outputs_folder"],
        outputs=status_box
    )

    clear_temp_btn.click(
        fn=lambda: service["clear_temp_folder"](False),
        outputs=status_box
    )
    
    # UNIVERSAL PRESET EVENT WIRING
    wire_universal_preset_events(
        preset_dropdown=preset_dropdown,
        preset_name_input=preset_name_input,
        save_btn=save_preset_btn,
        load_btn=load_preset_btn,
        preset_status=preset_status,
        reset_btn=reset_defaults_btn,
        delete_btn=delete_preset_btn,
        callbacks=preset_callbacks,
        inputs_list=inputs_list,
        shared_state=shared_state,
        tab_name="sparkvsr",
    )

    return {
        "inputs_list": inputs_list,
        "preset_dropdown": preset_dropdown,
        "preset_status": preset_status,
    }
