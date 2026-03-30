"""
RIFE / FPS / Edit Videos Tab - Self-contained modular implementation
UPDATED: Now uses Universal Preset System
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any
import html

from shared.services.rife_service import (
    build_rife_callbacks, RIFE_ORDER
)
from shared.models import get_rife_model_names
from shared.models.rife_meta import get_rife_default_model
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


def rife_tab(
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
    Self-contained RIFE / FPS / Edit Videos tab.
    Handles frame interpolation, FPS changes, and video editing.
    """

    # Build service callbacks
    service = build_rife_callbacks(
        preset_manager, runner, run_logger, global_settings,
        output_dir, temp_dir, shared_state
    )
    queue_manager = get_processing_queue_manager()

    # Get defaults
    defaults = service["defaults"]
    
    # UNIVERSAL PRESET: Load from shared_state
    seed_controls = shared_state.value.get("seed_controls", {})
    rife_settings = seed_controls.get("rife_settings", {})
    models_list = seed_controls.get("available_models", ["default"])
    
    # Merge with defaults
    merged_defaults = defaults.copy()
    for key, value in rife_settings.items():
        if value is not None:
            merged_defaults[key] = value
    
    values = [merged_defaults[k] for k in RIFE_ORDER]

    def _value(key: str, fallback=None):
        raw = merged_defaults.get(key, fallback)
        if raw is None and fallback is not None:
            return fallback
        return raw

    shared_rife_models = sorted(
        {
            str(model_name).strip()
            for model_name in (models_list or [])
            if str(model_name).strip() and (
                str(model_name).strip().lower().startswith("rife")
                or str(model_name).strip().lower() in {"anime", "rife-anime"}
                or str(model_name).strip().lower().replace("v", "", 1).replace(".", "", 1).isdigit()
            )
        }
    )
    # GPU availability check (parent-process safe: NO torch import)
    import platform
    cuda_available = False
    cuda_count = 0
    gpu_hint = "CUDA detection in progress..."
    
    try:
        from shared.gpu_utils import get_gpu_info
        gpus = get_gpu_info()
        cuda_count = len(gpus)
        cuda_available = cuda_count > 0
        
        if cuda_available:
            gpu_hint = f"SUCCESS: Detected {cuda_count} CUDA GPU(s) - GPU acceleration available"
        else:
            gpu_hint = "WARNING: CUDA not detected (nvidia-smi unavailable or no NVIDIA GPU) - GPU acceleration disabled. Processing will use CPU (significantly slower)"
    except Exception as e:
        gpu_hint = f"ERROR: CUDA detection failed: {str(e)}"
        cuda_available = False

    # Layout: Two-column design (left=controls, right=output)
    gr.Markdown("### RIFE / FPS / Edit Videos")
    gr.Markdown("*Frame interpolation, FPS adjustment, and video editing tools*")
    
    # Import shared layout helpers
    from ui.shared_layouts import create_gpu_warning_banner
    
    # Show GPU warning if not available
    create_gpu_warning_banner(cuda_available, gpu_hint, "RIFE")

    # Two-column layout
    with gr.Row():
        # ===== LEFT COLUMN: Input & Controls =====
        with gr.Column(scale=3):
            gr.Markdown("### Input / Controls")
            
            # Input section
            with gr.Accordion("Input Configuration", open=True):
                with gr.Row():
                    input_file = gr.File(
                        label="Upload Video or Image",
                        type="filepath",
                        file_types=["video", "image"]
                    )
                    with gr.Column():
                        input_image_preview = gr.Image(
                            label="Input Preview (Image)",
                            type="filepath",
                            interactive=False,
                            height=220,
                            visible=False,
                        )
                        input_video_preview = gr.Video(
                            label="Input Preview (Video)",
                            interactive=False,
                            height=220,
                            visible=False,
                        )
                input_path = gr.Textbox(
                    label="Input Path",
                    value=_value("input_path", ""),
                    placeholder="C:/path/to/video.mp4 or C:/path/to/images/",
                    info="Direct path to video file or image folder"
                )
                input_cache_msg = gr.Markdown("", visible=False)
                
                # Batch processing controls
                batch_enable = gr.Checkbox(
                    label="Enable Batch Processing",
                    value=bool(_value("batch_enable", False)),
                    info="Process multiple files from directory"
                )
                batch_input = gr.Textbox(
                    label="Batch Input Folder",
                    value=_value("batch_input_path", ""),
                    placeholder="Folder containing videos",
                    info="Directory with files to process in batch mode"
                )
                batch_output = gr.Textbox(
                    label="Batch Output Folder Override",
                    value=_value("batch_output_path", ""),
                    placeholder="Optional override for batch outputs",
                    info="Custom output directory for batch results"
                )

            # Processing settings
            with gr.Tabs():
                # Frame Interpolation (RIFE)
                with gr.TabItem("Frame Interpolation"):
                    gr.Markdown("#### RIFE - Real-Time Intermediate Flow Estimation")

                    # Output controls at top (more important than RIFE toggle for workflow)
                    with gr.Group():
                        gr.Markdown("#### Output Configuration")
                        
                        output_override = gr.Textbox(
                            label="Output Override (custom path)",
                            value=_value("output_override", ""),
                            placeholder="Leave empty for auto naming",
                            info="Specify custom output path. Auto-naming creates files in output folder."
                        )
                        
                        output_format_rife = gr.Dropdown(
                            label="Output Format",
                            choices=["auto", "mp4", "avi", "mov", "webm"],
                            value=_value("output_format", "mp4"),
                            info="Container format for output video"
                        )

                        with gr.Row(equal_height=True):
                            png_output = gr.Checkbox(
                                label="Export as PNG/JPG Sequence",
                                value=bool(_value("png_output", False)),
                                info="Save output as numbered PNG or JPG frames instead of a video file. Useful for further editing."
                            )
                            sequence_format = gr.Dropdown(
                                label="Sequence Format",
                                choices=["png", "jpg"],
                                value=str(_value("sequence_format", "png") or "png"),
                                interactive=bool(_value("png_output", False)),
                                info="PNG = lossless. JPG = smaller files."
                            )
                        sequence_quality = gr.Slider(
                            label="JPEG Sequence Quality (%)",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=int(_value("sequence_quality", 95) or 95),
                            interactive=bool(_value("png_output", False))
                            and str(_value("sequence_format", "png") or "png").strip().lower() == "jpg",
                            info="Used only when Sequence Format = jpg."
                        )
                    
                    with gr.Group():
                        gr.Markdown("#### RIFE Interpolation")
                        gr.Markdown(
                            "RIFE interpolation is always active in this tab based on FPS/settings below. "
                            "Cross-model post-upscale FPS generation is controlled from Output & Comparison > Global Enable RIFE."
                        )

                        def _discover_rife_models():
                            """Dynamically discover available RIFE models."""
                            # Only list locally installed models from supported layouts.
                            return get_rife_model_names(base_dir)
                        
                        model_dir = gr.Textbox(
                            label="Model Directory Override",
                            value=_value("model_dir", ""),
                            placeholder="Leave empty for default (RIFE/train_log)",
                            info="Custom path to RIFE model directory. Only needed if models are in non-standard location."
                        )
                        
                        _rife_model_choices = list(shared_rife_models) if shared_rife_models else _discover_rife_models()
                        _rife_model_value = str(_value("model", "") or "").strip()
                        if _rife_model_value not in _rife_model_choices:
                            preferred_default = get_rife_default_model()
                            if preferred_default in _rife_model_choices:
                                _rife_model_value = preferred_default
                            else:
                                _rife_model_value = _rife_model_choices[0] if _rife_model_choices else None

                        with gr.Row(equal_height=True):
                            rife_model = gr.Dropdown(
                                label="RIFE Model",
                                choices=_rife_model_choices,
                                value=_rife_model_value,
                                allow_custom_value=True,
                                info="RIFE model version. v4.6 = fastest. v4.15+ = best quality. 'anime' optimized for animation. Newer versions slower but smoother."
                            )
                            fps_multiplier = gr.Dropdown(
                                label="FPS Multiplier",
                                choices=["x1", "x2", "x4", "x8"],
                                value=_value("fps_multiplier", "x2"),
                                info="Multiply original FPS. x2 = double smoothness (30 to 60fps). x4 = 4x smoother. x8 = extreme slow-mo. Higher = more processing time."
                            )
                            target_fps = gr.Number(
                                label="Target FPS Override",
                                value=_value("fps_override", 0),
                                precision=1,
                                info="Desired output frame rate. 0 = use multiplier instead. 60 = smooth 60fps. 120 = ultra-smooth. Higher FPS = larger file size."
                            )
                        
                        # Model info display with metadata
                        rife_model_info = gr.Markdown("")
                        
                        def update_rife_model_info(model_name_val):
                            """Display RIFE model metadata information"""
                            from shared.models.rife_meta import get_rife_metadata
                            
                            metadata = get_rife_metadata(model_name_val)
                            
                            if metadata:
                                info_lines = [
                                    f"**Model: {metadata.name}**",
                                    f"**Version:** {metadata.version} | **Variant:** {metadata.variant.title()}",
                                    f"**VRAM Estimate:** ~{metadata.estimated_vram_gb:.1f}GB",
                                    f"**Multi-GPU:** {'Not supported (single GPU only)' if not metadata.supports_multi_gpu else 'Supported'}",
                                    f"**Max FPS Multiplier:** {metadata.max_fps_multiplier}x",
                                    f"**UHD Mode:** {'Supported (recommended for 4K+)' if metadata.supports_uhd else 'Not available'}",
                                ]
                                if metadata.notes:
                                    info_lines.append(f"\nNotes: {metadata.notes}")
                                
                                return gr.update(value="\n".join(info_lines), visible=True)
                            else:
                                return gr.update(value="Model metadata not available", visible=False)
                        
                        # Wire up model info update
                        rife_model.change(
                            fn=update_rife_model_info,
                            inputs=rife_model,
                            outputs=rife_model_info
                        )

                        with gr.Row(equal_height=True):
                            scale = gr.Slider(
                                label="Spatial Scale Factor",
                                minimum=0.5, maximum=4.0, step=0.1,
                                value=_value("scale", 1.0),
                                info="Scale video resolution. 1.0 = original size, 2.0 = double resolution. Can combine with interpolation. >1.0 significantly increases processing time."
                            )
                            
                            uhd_mode = gr.Checkbox(
                                label="UHD Mode (4K+ Processing)",
                                value=bool(_value("uhd_mode", False)) if cuda_available else False,  # Force False if no CUDA
                                info=f"{gpu_hint} | Enable optimizations for 4K/8K videos. Uses more memory. Enable for 3840x2160+ inputs.",
                                interactive=cuda_available  # Disable if no CUDA
                            )

                            rife_precision = gr.Dropdown(
                                label="Precision",
                                choices=["fp16", "fp32"],
                                value=(
                                    ("fp16" if str(_value("fp16_mode", "fp32")).strip().lower() in {"fp16", "true", "1", "yes", "on"} else "fp32")
                                    if cuda_available else "fp32"
                                ),
                                info=f"fp16 = half precision, 2x faster, less VRAM. fp32 = full precision. {'(fp16 requires GPU)' if not cuda_available else 'Use fp16 for speed.'}",
                                interactive=cuda_available  # Disable if no CUDA (CPU uses fp32 only)
                            )
                        
                        
                        img_mode = gr.State(bool(_value("img_mode", False)))

                        with gr.Row(equal_height=True):
                            montage = gr.Checkbox(
                                label="Create Montage (Side-by-Side Comparison)",
                                value=bool(_value("montage", False)),
                                info="Generate side-by-side comparison video showing original vs interpolated. Useful for quality checking."
                            )
                            
                            skip_static_frames = gr.Checkbox(
                                label="Skip Static Frames (Auto-Detect)",
                                value=bool(_value("skip_static_frames", False)),
                                info="Automatically skip static/duplicate frames. Saves processing time for videos with static scenes. May miss subtle motion."
                            )
                            
                            exp = gr.Number(
                                label="Temporal Recursion Depth",
                                value=_value("exp", 1),
                                precision=0,
                                info="Exponential frame generation depth. 1 = direct interpolation, 2+ = recursive. Higher = smoother but exponentially slower. Use 1 for most cases."
                            )

                        gr.Markdown(
                            "**GPU Device:** Controlled globally from the top app header selector. "
                            "This tab no longer has a per-tab GPU override."
                        )
                        rife_gpu = gr.State(_value("cuda_device", ""))

                # Video Editing
                with gr.TabItem("Video Editing"):
                    gr.Markdown("#### Video Trimming & Effects")

                    with gr.Group():
                        edit_mode = gr.Dropdown(
                            label="Edit Mode",
                            choices=["none", "trim", "concatenate", "speed_change", "effects"],
                            value=_value("edit_mode", "none"),
                            info="Type of video editing to perform"
                        )

                        start_time = gr.Textbox(
                            label="Start Time (HH:MM:SS or seconds)",
                            value=_value("start_time", ""),
                            placeholder="00:00:30 or 30",
                            info="Where to start the edit"
                        )

                        end_time = gr.Textbox(
                            label="End Time (HH:MM:SS or seconds)",
                            value=_value("end_time", ""),
                            placeholder="00:01:30 or 90",
                            info="Where to end the edit"
                        )

                        speed_factor = gr.Slider(
                            label="Speed Factor",
                            minimum=0.25, maximum=4.0, step=0.25,
                            value=_value("speed_factor", 1.0),
                            info="1.0 = normal speed, 2.0 = 2x faster, 0.5 = 2x slower"
                        )

                        concat_videos = gr.Textbox(
                            label="Additional Videos for Concatenation",
                            value=_value("concat_videos", ""),
                            placeholder="C:/path/to/video1.mp4, C:/path/to/video2.mp4",
                            info="Comma-separated list of video files to concatenate with the main input",
                            lines=2
                        )

                # Frame Control & Advanced
                with gr.TabItem("Frame Control"):
                    gr.Markdown("#### Advanced Frame Processing")

                    with gr.Group():
                        skip_first_frames = gr.Number(
                            label="Skip First Frames",
                            value=_value("skip_first_frames", 0),
                            precision=0,
                            info="Skip N frames from start of video. Useful to skip intros/logos. 0 = process from beginning."
                        )

                        load_cap = gr.Number(
                            label="Frame Load Cap (0 = all)",
                            value=_value("load_cap", 0),
                            precision=0,
                            info="Process only first N frames. Useful for quick tests. 0 = process entire video. Combine with skip for specific range."
                        )

                # Output Settings
                with gr.TabItem("Output Settings"):
                    gr.Markdown("#### Video Export Configuration")

                    with gr.Group():
                        video_codec_rife = gr.Dropdown(
                            label="Video Codec",
                            choices=["libx264", "libx265", "libvpx-vp9"],
                            value=_value("video_codec", "libx264"),
                            info="Compression codec"
                        )

                        output_quality_rife = gr.Slider(
                            label="Quality (CRF)",
                            minimum=0, maximum=51, step=1,
                            value=_value("output_quality", 23),
                            info="Lower = higher quality, larger file"
                        )

                        no_audio = gr.Checkbox(
                            label="Remove Audio",
                            value=bool(_value("no_audio", False)),
                            info="Strip audio track from output"
                        )

                        show_ffmpeg_output = gr.Checkbox(
                            label="Show FFmpeg Output",
                            value=bool(_value("show_ffmpeg", False)),
                            info="Display detailed processing logs"
                        )
        
        # ===== RIGHT COLUMN: Output & Actions =====
        with gr.Column(scale=2):
            gr.Markdown("### Output / Actions")
            
            # Status and progress
            status_box = gr.Markdown(value="Ready for processing.", visible=False, elem_classes=["runtime-status-box"])
            progress_indicator = gr.Markdown(value="", visible=False, elem_classes=["runtime-progress-box"])
            log_box = gr.Textbox(
                label="Processing Log",
                value="",
                lines=10,
                buttons=["copy"]
            )

            resume_run_dir = gr.Textbox(
                label="Resume Run Folder (chunk/scene resume)",
                value=(
                    values[RIFE_ORDER.index("resume_run_dir")]
                    if "resume_run_dir" in RIFE_ORDER and len(values) > RIFE_ORDER.index("resume_run_dir")
                    else ""
                ),
                placeholder="Optional: G:/.../outputs/0019",
                info=(
                    "Optional. When set, chunk/scene processing resumes from the last completed chunk in that folder. "
                    "Use the same settings as the original run to continue remaining chunks. "
                    "Fresh output path overrides are ignored while resume is active."
                ),
            )

            # Output displays
            output_video = gr.Video(
                label="Processed Video",
                interactive=False,
                buttons=["download"]
            )
            
            # Comparison outputs (matching SeedVR2/GAN tabs)
            image_slider = gr.ImageSlider(
                label="Before/After Comparison",
                interactive=False,
                slider_position=50,
                max_height=1000,
                buttons=["download", "fullscreen"],
                elem_classes=["native-image-comparison-slider"],
            )
            
            video_comparison_html = gr.HTML(
                label="Video Comparison Slider",
                value="",
                js_on_load=get_video_comparison_js_on_load(),
                visible=False
            )

            # Action buttons
            with gr.Row():
                process_btn = gr.Button(
                    "Process Video",
                    variant="primary",
                    size="lg",
                    elem_classes=["action-btn", "action-btn-upscale"],
                )
                cancel_btn = gr.Button(
                    "Cancel",
                    variant="stop",
                    size="lg",
                    elem_classes=["action-btn", "action-btn-cancel"],
                )
            
            cancel_confirm = gr.Checkbox(
                label="Confirm cancel (required for safety)",
                value=False,
                info="Enable this checkbox to confirm cancellation"
            )

            # Utility buttons
            with gr.Row():
                open_outputs_btn = gr.Button(
                    "Open Outputs Folder",
                    elem_classes=["action-btn", "action-btn-open"],
                )
                clear_temp_btn = gr.Button(
                    "Clear Temp Files",
                    elem_classes=["action-btn", "action-btn-clear"],
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
                tab_name="rife",
                inputs_list=[],
                base_dir=base_dir,
                models_list=models_list,
                open_accordion=True,
            )

    # Info section (outside columns, full width)
    with gr.Accordion("About RIFE & FPS", open=False):
        gr.Markdown("""
        #### RIFE (Real-Time Intermediate Flow Estimation)

        **What it does:**
        - Generates smooth intermediate frames between existing frames
        - Converts 30fps video to 60fps, 120fps, etc.
        - Creates natural motion without stuttering

        **Use cases:**
        - Smooth slow-motion video
        - Fix stuttering from low frame rate sources
        - Enhance video playback quality

        **Performance notes:**
        - Processing time increases with multiplier
        - Higher quality models are slower
        - GPU acceleration highly recommended

        #### Video Editing Features

        **Trimming:** Cut specific time ranges
        **Speed Change:** Slow down or speed up video
        **Effects:** Apply various video filters
        **Format Conversion:** Change codecs/containers
        """)

    # Collect all inputs matching RIFE_ORDER exactly
    # IMPORTANT: Order must match RIFE_ORDER in shared/services/rife_service.py
    # ============================================================================
    # RIFE PRESET INPUT LIST - MUST match RIFE_ORDER in rife_service.py
    # Adding controls? Update rife_defaults(), RIFE_ORDER, and this list in sync.
    # Current count: 33 components
    # ============================================================================

    inputs_list = [
        input_path,           # 0: input_path
        output_override,      # 1: output_override
        output_format_rife,   # 2: output_format
        model_dir,            # 3: model_dir
        rife_model,           # 4: model
        fps_multiplier,       # 5: fps_multiplier
        target_fps,           # 6: fps_override
        scale,                # 7: scale
        uhd_mode,             # 8: uhd_mode
        rife_precision,       # 9: fp16_mode
        png_output,           # 10: png_output
        sequence_format,      # 11: sequence_format
        sequence_quality,     # 12: sequence_quality
        no_audio,             # 13: no_audio
        show_ffmpeg_output,   # 14: show_ffmpeg
        montage,              # 15: montage
        img_mode,             # 16: img_mode
        skip_static_frames,   # 17: skip_static_frames
        exp,                  # 18: exp
        batch_enable,         # 19: batch_enable
        batch_input,          # 20: batch_input_path
        batch_output,         # 21: batch_output_path
        skip_first_frames,    # 22: skip_first_frames
        load_cap,             # 23: load_cap
        rife_gpu,             # 24: cuda_device
        edit_mode,            # 25: edit_mode
        start_time,           # 26: start_time
        end_time,             # 27: end_time
        speed_factor,         # 28: speed_factor
        video_codec_rife,     # 29: video_codec
        output_quality_rife,  # 30: output_quality
        concat_videos,        # 31: concat_videos
        resume_run_dir,       # 32: resume_run_dir
    ]
    
    # Development validation
    if len(inputs_list) != len(RIFE_ORDER):
        import logging
        logging.getLogger("RIFETab").error(
            f"ERROR: inputs_list ({len(inputs_list)}) != RIFE_ORDER ({len(RIFE_ORDER)})"
        )

    # Wire up event handlers
    
    # Per-tab GPU override removed: global selector controls all runs.

    # Input handling
    def cache_input(val, state):
        state["seed_controls"]["last_input_path"] = val if val else ""
        return val or "", gr.update(value="SUCCESS: Input cached for processing.", visible=True), state

    input_upload_evt = input_file.upload(
        fn=lambda val, state: cache_input(val, state),
        inputs=[input_file, shared_state],
        outputs=[input_path, input_cache_msg, shared_state]
    )

    # Input preview (image + video)
    input_file.change(
        fn=lambda p: preview_updates(p),
        inputs=[input_file],
        outputs=[input_image_preview, input_video_preview],
    )

    # If user clears the upload (clicks "X"), clear the textbox + hide the cached message.
    def clear_on_upload_clear(file_path, state):
        if file_path:
            return gr.update(), gr.update(), state
        try:
            state = state or {}
            state.setdefault("seed_controls", {})
            state["seed_controls"]["last_input_path"] = ""
        except Exception:
            pass
        return "", gr.update(value="", visible=False), state

    input_file.change(
        fn=clear_on_upload_clear,
        inputs=[input_file, shared_state],
        outputs=[input_path, input_cache_msg, shared_state],
    )

    input_path_submit_evt = input_path.submit(
        fn=lambda val, state: (gr.update(value="SUCCESS: Input path updated.", visible=True), state),
        inputs=[input_path, shared_state],
        outputs=[input_cache_msg, shared_state]
    )

    input_path_submit_evt.then(
        fn=lambda p: preview_updates(p),
        inputs=[input_path],
        outputs=[input_image_preview, input_video_preview],
        queue=False,
        show_progress="hidden",
    )

    input_upload_evt.then(
        fn=lambda p: preview_updates(p),
        inputs=[input_path],
        outputs=[input_image_preview, input_video_preview],
        queue=False,
        show_progress="hidden",
    )

    def _queue_status_indicator(title: str, subtitle: str, spinning: bool = True):
        """Render queue/runtime state in the same rich card style as processing."""
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

    def _queued_waiting_output(state, ticket_id: str, position: int):
        safe_state = state or {}
        pos = max(1, int(position)) if position else "?"
        title = f"Queue waiting: {ticket_id} (position {pos})"
        subtitle = (
            f"Queued and waiting for active processing slot. Queue position: {pos}. "
            "Run logs will begin once this request starts."
        )
        return (
            gr.update(value=title),
            gr.update(value=f"Queued and waiting for active processing slot. Queue position: {pos}."),
            _queue_status_indicator(title, subtitle, spinning=True),
            gr.update(),
            gr.update(),
            gr.update(),
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
            gr.update(),
            gr.update(),
            gr.update(),
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
            gr.update(),
            gr.update(),
            gr.update(),
            safe_state,
        )

    def run_process_with_queue(*args):
        live_state = args[-1] if (args and isinstance(args[-1], dict)) else {}
        queued_state = snapshot_queue_state(live_state)
        queued_global_settings = snapshot_global_settings(global_settings)
        queue_enabled = bool(queued_global_settings.get("queue_enabled", True))
        queue_resource_keys, queue_resource_label = resolve_queue_gpu_resources(queued_state, queued_global_settings)
        ticket = queue_manager.submit(
            "RIFE",
            "Process",
            resource_keys=queue_resource_keys,
            resource_label=queue_resource_label,
        )
        acquired_slot = queue_manager.is_active(ticket.job_id)

        try:
            if not queue_enabled:
                if not acquired_slot:
                    queue_manager.cancel_waiting([ticket.job_id])
                    yield _queue_disabled_busy_output(live_state)
                    return
                for payload in service["run_action"](
                    *args[:-1],
                    state=queued_state,
                    global_settings_snapshot=queued_global_settings,
                ):
                    yield merge_payload_state(payload, live_state)
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
                *args[:-1],
                state=queued_state,
                global_settings_snapshot=queued_global_settings,
            ):
                yield merge_payload_state(payload, live_state)
        finally:
            if acquired_slot:
                queue_manager.complete(ticket.job_id)
            else:
                queue_manager.cancel_waiting([ticket.job_id])

    # Main processing
    process_btn.click(
        fn=run_process_with_queue,
        inputs=inputs_list + [shared_state],
        outputs=[status_box, log_box, progress_indicator, output_video, image_slider, video_comparison_html, shared_state],
        concurrency_limit=32,
        concurrency_id="app_processing_queue",
        trigger_mode="multiple",
    )

    cancel_btn.click(
        fn=lambda ok, state: (*service["cancel_action"](), state) if ok else (gr.update(value="WARNING: Enable 'Confirm cancel' to stop."), "", state),
        inputs=[cancel_confirm, shared_state],
        outputs=[status_box, log_box, shared_state]
    )

    # Utility functions
    open_outputs_btn.click(
        fn=lambda: service["open_outputs_folder"](),
        outputs=status_box
    )

    clear_temp_btn.click(
        fn=lambda: service["clear_temp_folder"](False),
        outputs=status_box
    )

    def _sync_sequence_controls(enabled: bool, fmt: str):
        active = bool(enabled)
        fmt_value = str(fmt or "png").strip().lower()
        if fmt_value not in {"png", "jpg"}:
            fmt_value = "png"
        quality_active = active and fmt_value == "jpg"
        return (
            gr.update(interactive=active, value=fmt_value),
            gr.update(interactive=quality_active),
        )

    png_output.change(
        fn=_sync_sequence_controls,
        inputs=[png_output, sequence_format],
        outputs=[sequence_format, sequence_quality],
    )
    sequence_format.change(
        fn=_sync_sequence_controls,
        inputs=[png_output, sequence_format],
        outputs=[sequence_format, sequence_quality],
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
        tab_name="rife",
    )

    return {
        "inputs_list": inputs_list,
        "preset_dropdown": preset_dropdown,
        "preset_status": preset_status,
    }

