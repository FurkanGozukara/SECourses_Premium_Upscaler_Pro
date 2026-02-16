"""
FlashVSR+ Tab - Self-contained modular implementation
Real-time diffusion-based streaming video super-resolution
UPDATED: Now uses Universal Preset System
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any
import html

from shared.services.flashvsr_service import (
    build_flashvsr_callbacks, FLASHVSR_ORDER
)
from shared.fixed_scale_analysis import build_fixed_scale_analysis_update
from shared.models.flashvsr_meta import flashvsr_version_to_internal, flashvsr_version_to_ui
from ui.universal_preset_section import (
    universal_preset_section,
    wire_universal_preset_events,
)
from shared.universal_preset import dict_to_values
from ui.media_preview import preview_updates
from shared.video_comparison_slider import get_video_comparison_js_on_load
from shared.processing_queue import get_processing_queue_manager
from shared.queue_state import (
    snapshot_queue_state,
    snapshot_global_settings,
    merge_payload_state,
)


def flashvsr_tab(
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
    Self-contained FlashVSR+ tab following SECourses modular pattern.
    """

    # Build service callbacks
    service = build_flashvsr_callbacks(
        preset_manager, runner, run_logger, global_settings, shared_state,
        base_dir, temp_dir, output_dir
    )
    queue_manager = get_processing_queue_manager()

    # Get defaults
    defaults = service["defaults"]
    
    # UNIVERSAL PRESET: Load from shared_state
    seed_controls = shared_state.value.get("seed_controls", {})
    flashvsr_settings = seed_controls.get("flashvsr_settings", {})
    models_list = seed_controls.get("available_models", ["default"])
    
    # Merge with defaults
    merged_defaults = defaults.copy()
    for key, value in flashvsr_settings.items():
        if value is not None:
            merged_defaults[key] = value
    
    values = [merged_defaults[k] for k in FLASHVSR_ORDER]

    def _value(key: str, default=None):
        try:
            idx = FLASHVSR_ORDER.index(key)
            if 0 <= idx < len(values):
                raw = values[idx]
                if raw is None and default is not None:
                    return default
                return raw
        except Exception:
            pass
        return default
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
            gpu_hint = f" Detected {cuda_count} CUDA GPU(s) - GPU acceleration available\n FlashVSR+ uses single GPU only (multi-GPU not supported)"
        else:
            gpu_hint = " CUDA not detected (nvidia-smi unavailable or no NVIDIA GPU) - Processing will use CPU (significantly slower)"
    except Exception as e:
        gpu_hint = f" CUDA detection failed: {str(e)}"
        cuda_available = False

    # Layout
    gr.Markdown("###  FlashVSR+ - Real-Time Diffusion Video Super-Resolution")
    gr.Markdown("*High-quality real-time video upscaling with diffusion models*")
    
    # Show GPU warning if not available
    if not cuda_available:
        gr.Markdown(
            f'<div style="background: #fff3cd; padding: 12px; border-radius: 8px; border: 1px solid #ffc107;">'
            f'<strong> GPU Acceleration Unavailable</strong><br>'
            f'{gpu_hint}<br><br>'
            f'FlashVSR+ requires CUDA for optimal performance. CPU mode is extremely slow.'
            f'</div>',
            elem_classes="warning-text"
        )

    with gr.Row():
        # Left Column: Input & Settings
        with gr.Column(scale=3):
            gr.Markdown("####  Input")

            with gr.Group():
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
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

                    with gr.Column(scale=2):
                        # Hidden model scale: FlashVSR pipeline runs fixed model scale internally.
                        # UI keeps only Upscale-x controls in the right column as requested.
                        scale = gr.Dropdown(
                            label="Upscale Factor",
                            choices=["2", "4"],
                            value="4",
                            visible=False,
                            interactive=False,
                        )
                        version = gr.Dropdown(
                            label="Model Version",
                            choices=["1.0", "1.1"],
                            value=flashvsr_version_to_ui(_value("version", "1.0")),
                            info="1.0 = faster, 1.1 = higher quality"
                        )
                        mode = gr.Dropdown(
                            label="Pipeline Mode",
                            choices=["tiny", "tiny-long", "full"],
                            value=str(_value("mode", "tiny")) if str(_value("mode", "tiny")) in {"tiny", "tiny-long", "full"} else "tiny",
                            info="tiny = fastest (4-6GB VRAM), tiny-long = balanced (5-7GB), full = best quality (8-12GB)"
                        )
                        model_info_display = gr.Markdown("")

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

            def update_flashvsr_model_info(version_val, mode_val, scale_val):
                """Display model metadata information"""
                from shared.models.flashvsr_meta import get_flashvsr_metadata

                model_id = f"v{flashvsr_version_to_internal(version_val)}_{mode_val}_{scale_val}x"
                metadata = get_flashvsr_metadata(model_id)

                if metadata:
                    info_lines = [
                        f"** Model: {metadata.name}**",
                        f"**VRAM Estimate:** ~{metadata.estimated_vram_gb:.1f}GB",
                        f"**Speed:** {metadata.speed_tier.title()} | **Quality:** {metadata.quality_tier.replace('_', ' ').title()}",
                        f"**Multi-GPU:** {' Not supported' if not metadata.supports_multi_gpu else ' Supported'}",
                        f"**Compile:** {' Compatible' if metadata.compile_compatible else ' Not supported'}",
                    ]
                    if metadata.notes:
                        info_lines.append(f"\n {metadata.notes}")

                    return gr.update(value="\n".join(info_lines), visible=True)
                else:
                    return gr.update(value="Model metadata not available", visible=False)

            # Wire up model info updates
            for component in [version, mode, scale]:
                component.change(
                    fn=update_flashvsr_model_info,
                    inputs=[version, mode, scale],
                    outputs=model_info_display
                )
            
            # Processing Settings
            gr.Markdown("####  Processing Settings")
            
            with gr.Group():
                with gr.Row():
                    dtype = gr.Dropdown(
                        label="Precision",
                        choices=["fp16", "bf16"],
                        value=str(_value("dtype", "bf16")) if str(_value("dtype", "bf16")) in {"fp16", "bf16"} else "bf16",
                        info="bf16 = faster, more stable. fp16 = broader compatibility"
                    )
                    
                    attention = gr.Dropdown(
                        label="Attention Mode",
                        choices=["sage", "block"],
                        value=str(_value("attention", "sage")) if str(_value("attention", "sage")) in {"sage", "block"} else "sage",
                        info="sage = default, block = alternative implementation"
                    )

                with gr.Row():
                    seed = gr.Number(
                        label="Random Seed",
                        value=_value("seed", 0),
                        precision=0,
                        info="Seed for reproducibility. 0 = random"
                    )
                    
                    device = gr.Textbox(
                        label="Device (Single GPU Only)",
                        value=_value("device", "auto") if cuda_available else "cpu",
                        placeholder="auto, cuda:0, cpu" if cuda_available else "CPU only (no CUDA)",
                        info=f"{gpu_hint}\nauto = automatic GPU selection, cuda:0 = specific GPU, cpu = CPU mode. Multi-GPU NOT supported by FlashVSR+.",
                        interactive=cuda_available
                    )
            
            # Memory Optimization
            gr.Markdown("####  Memory Optimization (Tiling)")
            
            with gr.Group():
                with gr.Row():
                    tiled_vae = gr.Checkbox(
                        label="Enable VAE Tiling",
                        value=bool(_value("tiled_vae", False)),
                        info="Reduce VRAM usage during VAE encoding/decoding. Essential for high resolutions."
                    )
                    
                    tiled_dit = gr.Checkbox(
                        label="Enable DiT Tiling",
                        value=bool(_value("tiled_dit", False)),
                        info="Reduce VRAM usage during diffusion inference. Enables processing larger videos."
                    )
                    
                    unload_dit = gr.Checkbox(
                        label="Unload DiT Before Decoding",
                        value=bool(_value("unload_dit", False)),
                        info="Free VRAM before VAE decoding. Slower but uses less memory."
                    )

                with gr.Row():
                    tile_size = gr.Slider(
                        label="Tile Size",
                        minimum=128, maximum=512, step=32,
                        value=int(_value("tile_size", 256) or 256),
                        info="Size of each tile. Larger = faster but more VRAM"
                    )
                    
                    overlap = gr.Slider(
                        label="Tile Overlap",
                        minimum=8, maximum=64, step=8,
                        value=int(_value("overlap", 24) or 24),
                        info="Overlap between tiles to reduce seams. Higher = smoother"
                    )
            
            # Quality Settings
            gr.Markdown("####  Quality Settings")
            
            with gr.Group():
                with gr.Row():
                    color_fix = gr.Checkbox(
                        label="Color Correction",
                        value=bool(_value("color_fix", True)),
                        info="Maintain color accuracy. Recommended ON."
                    )
                    
                    fps_flashvsr = gr.Number(
                        label="Output FPS (image sequences only)",
                        value=int(_value("fps", 30) or 30),
                        precision=0,
                        info="Frame rate for image sequence outputs. Ignored for video inputs."
                    )
                    
                    quality = gr.Slider(
                        label="Video Quality",
                        minimum=1, maximum=10, step=1,
                        value=int(_value("quality", 6) or 6),
                        info="Output quality. 1 = lowest, 10 = highest. 6 is recommended."
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
                        info="FlashVSR+ currently outputs MP4.",
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
                tab_name="flashvsr",
                inputs_list=[],
                base_dir=base_dir,
                models_list=models_list,
                open_accordion=True,
            )
        
        # Right Column: Output & Controls
        with gr.Column(scale=2):
            gr.Markdown("####  Output & Actions")

            with gr.Group():
                _upscale_factor_default = _value("upscale_factor", _value("scale", 4))
                try:
                    _upscale_factor_default = float(_upscale_factor_default)
                except Exception:
                    _upscale_factor_default = 4.0
                _upscale_factor_default = min(9.9, max(1.0, _upscale_factor_default))

                _max_resolution_default = _value("max_target_resolution", 0)
                try:
                    _max_resolution_default = int(_max_resolution_default)
                except Exception:
                    _max_resolution_default = 0
                _max_resolution_default = min(8192, max(0, _max_resolution_default))

                with gr.Row():
                    upscale_factor = gr.Slider(
                        label="Upscale x (any factor)",
                        minimum=1.0,
                        maximum=9.9,
                        step=0.1,
                        value=_upscale_factor_default,
                        info="e.g., 4.0 = 4x. Target size is computed from input, then capped by Max Resolution (max edge).",
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
                        value=bool(_value("pre_downscale_then_upscale", False)),
                        info="If max edge would reduce effective scale, downscale input first so the model still runs at the full Upscale x.",
                        scale=1,
                    )

                use_resolution_tab = gr.Checkbox(
                    label="Use Resolution & Scene Split Tab Settings",
                    value=bool(_value("use_resolution_tab", True)),
                    info="Apply Upscale-x, Max Resolution, and Pre-downscale settings from Resolution tab. Recommended ON.",
                )
            
            status_box = gr.Markdown(value="Ready.")
            progress_indicator = gr.Markdown(value="", visible=True)

            resume_run_dir = gr.Textbox(
                label="Resume Run Folder (chunk/scene resume)",
                value=(
                    _value("resume_run_dir", "")
                    if "resume_run_dir" in FLASHVSR_ORDER and len(values) > FLASHVSR_ORDER.index("resume_run_dir")
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

            # Action buttons
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
                cancel_btn = gr.Button(
                    " Cancel",
                    variant="stop",
                    elem_classes=["action-btn", "action-btn-cancel"],
                )
            cancel_confirm = gr.Checkbox(
                label=" Confirm cancel (required for safety)",
                value=False,
                info="Enable this checkbox to confirm cancellation of processing"
            )
            
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

            with gr.Accordion(" Batch Processing", open=False):
                batch_enable = gr.Checkbox(
                    label="Enable Batch",
                    value=bool(_value("batch_enable", False)),
                    info="Process multiple files"
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
            ####  About FlashVSR+
            
            **Real-time Diffusion Video SR:**
            - Streaming processing for memory efficiency
            - Multiple pipeline modes for speed/quality tradeoff
            - Automatic model download from HuggingFace
            
            **Recommended Settings:**
            - Mode: `tiny` for real-time, `full` for best quality
            - Enable tiling for high-res or limited VRAM
            - Use color fix for accurate colors
            """)
    
    # Collect inputs
    inputs_list = [
        input_path, output_override, output_format, scale, version, mode,
        tiled_vae, tiled_dit, tile_size, overlap, unload_dit,
        color_fix, seed, dtype, device, fps_flashvsr,
        quality, attention, save_metadata, face_restore_after_upscale, batch_enable, batch_input, batch_output,
        use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale,
        resume_run_dir,
    ]

    # Development validation: inputs_list must stay aligned with FLASHVSR_ORDER
    if len(inputs_list) != len(FLASHVSR_ORDER):
        import logging
        logging.getLogger("FlashVSRTab").error(
            f"ERROR: inputs_list ({len(inputs_list)}) != FLASHVSR_ORDER ({len(FLASHVSR_ORDER)})"
        )
    
    # Wire up events
    def cache_input(val, scale_val, use_global, scale_x, max_edge, pre_down, state):
        try:
            state = state or {}
            state.setdefault("seed_controls", {})
            state["seed_controls"]["last_input_path"] = val if val else ""
        except Exception:
            pass
        det = _build_input_detection_md(val or "")
        info = _build_sizing_info(val or "", int(scale_val), bool(use_global), scale_x, max_edge, pre_down, state)
        img_prev, vid_prev = preview_updates(val)
        return (
            val or "",
            gr.update(value="OK: Input cached for processing.", visible=True),
            img_prev,
            vid_prev,
            det,
            info,
            state,
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
        ms = int(model_scale_val or 4)
        return build_fixed_scale_analysis_update(
            input_path_val=path_val,
            model_scale=ms,
            use_global=bool(use_global),
            local_scale_x=float(local_scale_x or 4.0),
            local_max_edge=int(local_max_edge or 0),
            local_pre_down=bool(local_pre_down),
            state=state,
            model_label="FlashVSR+",
            runtime_label=f"FlashVSR+ pipeline (fixed {ms}x pass)",
            auto_scene_scan=True,
        )

    input_file.upload(
        fn=cache_input,
        inputs=[input_file, scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state]
    )

    # Preview + sizing + detection refresh on input changes
    def refresh_panels(path_val, scale_val, use_global, scale_x, max_edge, pre_down, state):
        img_prev, vid_prev = preview_updates(path_val)
        det = _build_input_detection_md(path_val or "")
        info = _build_sizing_info(path_val, int(scale_val), bool(use_global), scale_x, max_edge, pre_down, state)
        return img_prev, vid_prev, det, info, state

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

    input_file.change(
        fn=clear_input_path_on_upload_clear,
        inputs=[input_file, shared_state],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )

    def cache_input_path_only(path_val, state):
        try:
            state = state or {}
            state.setdefault("seed_controls", {})
            state["seed_controls"]["last_input_path"] = path_val if path_val else ""
        except Exception:
            pass
        return gr.update(value="OK: Input path updated.", visible=True), state

    input_path.change(
        fn=cache_input_path_only,
        inputs=[input_path, shared_state],
        outputs=[input_cache_msg, shared_state],
    )

    input_path.change(
        fn=refresh_panels,
        inputs=[input_path, scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )

    for comp in [scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale]:
        comp.change(
            fn=lambda p, s, ug, sx, me, pd, st: (_build_sizing_info(p, int(s), bool(ug), sx, me, pd, st), st),
            inputs=[input_path, scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale, shared_state],
            outputs=[sizing_info, shared_state],
            trigger_mode="always_last",
        )

    def _lock_flashvsr_scale(scale_val):
        if str(scale_val).strip() == "4":
            return gr.skip()
        return gr.update(value="4")

    scale.change(
        fn=_lock_flashvsr_scale,
        inputs=[scale],
        outputs=[scale],
        queue=False,
        show_progress="hidden",
    )

    def refresh_chunk_preview_ui(state):
        preview = (state or {}).get("seed_controls", {}).get("flashvsr_chunk_preview", {})
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
            videos = (state or {}).get("seed_controls", {}).get("flashvsr_chunk_preview", {}).get("videos", [])
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

    def _batch_gallery_update_from_state(state):
        outputs = (state or {}).get("seed_controls", {}).get("flashvsr_batch_outputs", [])
        if not isinstance(outputs, list):
            outputs = []
        outputs = [str(p) for p in outputs if p and Path(str(p)).exists()]
        return gr.update(value=outputs, visible=bool(outputs))

    def _last_processed_text(state, vid_upd, img_upd) -> str:
        outputs = (state or {}).get("seed_controls", {}).get("flashvsr_batch_outputs", [])
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
                gr.update(value="ERROR: Invalid FlashVSR+ payload"),
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
        status_lc = status_text.lower()

        terminal_tokens = (
            "complete",
            "failed",
            "error",
            "critical",
            "cancel",
            "no result",
            "out of vram",
            "oom",
            "missing",
            "insufficient",
        )
        is_terminal = any(tok in status_lc for tok in terminal_tokens)
        if status_text and not is_terminal:
            subtitle = "Processing..."
            if isinstance(logs, str):
                for line in reversed(logs.splitlines()):
                    line = line.strip()
                    if line:
                        subtitle = line
                        break
            progress_update = _queue_status_indicator(status_text, subtitle, spinning=True)
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

    def _queued_waiting_output(state, ticket_id: str, position: int):
        safe_state = state or {}
        pos = max(1, int(position)) if position else "?"
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
        ticket = queue_manager.submit("FlashVSR+", "Upscale")
        acquired_slot = queue_manager.is_active(ticket.job_id)

        try:
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
                        pos_text = max(1, int(pos)) if pos else "?"
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
        for payload in service["run_action"](
            args[0],
            *args[1:-1],
            preview_only=True,
            state=queued_state,
            progress=progress,
            global_settings_snapshot=queued_global_settings,
        ):
            yield _expand_service_payload(payload, live_state)

    # Main processing
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
        fn=refresh_chunk_preview_ui,
        inputs=[shared_state],
        outputs=[chunk_status, chunk_gallery, chunk_preview_video],
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
        fn=refresh_chunk_preview_ui,
        inputs=[shared_state],
        outputs=[chunk_status, chunk_gallery, chunk_preview_video],
    )

    cancel_btn.click(
        fn=lambda ok: service["cancel_action"]() if ok else (gr.update(value="WARNING: Enable 'Confirm cancel' to stop."), ""),
        inputs=[cancel_confirm],
        outputs=[status_box, log_box]
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
        tab_name="flashvsr",
    )

    return {
        "inputs_list": inputs_list,
        "preset_dropdown": preset_dropdown,
        "preset_status": preset_status,
    }

