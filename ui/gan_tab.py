"""
Image-Based (GAN) Tab - Self-contained modular implementation
UPDATED: Now uses Universal Preset System
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any
import html

from shared.services.gan_service import (
    build_gan_callbacks, GAN_ORDER
)
from shared.video_comparison_slider import create_video_comparison_html, get_video_comparison_js_on_load

from ui.universal_preset_section import (
    universal_preset_section,
    wire_universal_preset_events,
)
from shared.universal_preset import dict_to_values
from shared.fixed_scale_analysis import build_fixed_scale_analysis_update
from ui.media_preview import preview_updates
from shared.processing_queue import get_processing_queue_manager
from shared.queue_state import (
    snapshot_queue_state,
    snapshot_global_settings,
    merge_payload_state,
)


def gan_tab(
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
    Self-contained Image-Based (GAN) tab.
    Handles GAN-based upscaling with fixed scale factors.
    """

    # Build service callbacks
    service = build_gan_callbacks(
        preset_manager, runner, run_logger, global_settings, shared_state,
        base_dir, temp_dir, output_dir
    )
    queue_manager = get_processing_queue_manager()

    # Get defaults
    defaults = service["defaults"]
    
    # UNIVERSAL PRESET: Load from shared_state
    seed_controls = shared_state.value.get("seed_controls", {})
    gan_settings = seed_controls.get("gan_settings", {})
    models_list = seed_controls.get("available_models", ["default"])

    # Merge with defaults
    merged_defaults = defaults.copy()
    for key, value in gan_settings.items():
        if value is not None:
            merged_defaults[key] = value

    values = [merged_defaults[k] for k in GAN_ORDER]

    # Only show GAN weights in this tab.
    gan_model_choices = [m for m in service["model_scanner"]() if str(m).strip()]
    if not gan_model_choices:
        gan_model_choices = sorted(
            {
                str(model_name).strip()
                for model_name in (models_list or [])
                if str(model_name).strip().lower().endswith((".pth", ".safetensors"))
            }
        )

    gan_model_value = values[4] if len(values) > 4 else ""
    if gan_model_choices and gan_model_value not in gan_model_choices:
        gan_model_value = gan_model_choices[0]

    from shared.gan_runner import get_gan_model_metadata

    model_meta_cache: Dict[str, Any] = {}

    def _get_model_meta(model_name: str):
        key = str(model_name or "").strip()
        if not key:
            return None
        if key not in model_meta_cache:
            try:
                model_meta_cache[key] = get_gan_model_metadata(key, base_dir)
            except Exception:
                model_meta_cache[key] = None
        return model_meta_cache[key]

    gan_model_dropdown_choices = []
    for model_name in gan_model_choices:
        meta = _get_model_meta(model_name)
        if meta is not None and getattr(meta, "scale", None):
            label = f"{model_name} (x{int(meta.scale)})"
        else:
            label = model_name
        gan_model_dropdown_choices.append((label, model_name))

    def update_model_info(model_name):
        if not model_name:
            return "Select a model to see details."
        metadata = _get_model_meta(model_name)
        if metadata is None:
            return f"**{model_name}**\n\nMetadata unavailable."

        info_lines = [f"**{model_name}**"]
        info_lines.append(f"- **Upscale**: x{int(metadata.scale)}")
        info_lines.append(f"- **Architecture**: {metadata.architecture}")
        if metadata.description and metadata.description != f"{model_name}":
            info_lines.append(f"- **Description**: {metadata.description}")
        if metadata.author and metadata.author != "unknown":
            info_lines.append(f"- **Author**: {metadata.author}")
        if metadata.tags:
            info_lines.append(f"- **Tags**: {', '.join(metadata.tags)}")
        return "\n".join(info_lines)

    # GPU availability check (parent-process safe: NO torch import)
    cuda_available = False
    cuda_count = 0
    gpu_hint = "CUDA detection in progress..."

    try:
        from shared.gpu_utils import get_gpu_info
        gpus = get_gpu_info()
        cuda_count = len(gpus)
        cuda_available = cuda_count > 0

        if cuda_available:
            gpu_hint = f"Detected {cuda_count} CUDA GPU(s) - GPU acceleration available"
        else:
            gpu_hint = "CUDA not detected (nvidia-smi unavailable or no NVIDIA GPU) - GPU acceleration disabled. Processing will use CPU (significantly slower)"
    except Exception as e:
        gpu_hint = f"CUDA detection failed: {str(e)}"
        cuda_available = False

    gr.Markdown("### Image-Based (GAN) Upscaling")
    gr.Markdown("*High-quality image and video upscaling using fixed-scale GAN models.*")

    if not cuda_available:
        gr.Markdown(
            f'<div style="background: #fff3cd; padding: 12px; border-radius: 8px; border: 1px solid #ffc107;">'
            f'<strong>GPU Acceleration Unavailable</strong><br>'
            f'{gpu_hint}<br><br>'
            f'GAN processing will run on CPU and can be significantly slower.'
            f'</div>',
            elem_classes="warning-text"
        )

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Input and Processing")

            with gr.Group():
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        input_file = gr.File(
                            label="Upload Image or Video",
                            type="filepath",
                            file_types=["image", "video"]
                        )
                        input_path = gr.Textbox(
                            label="Image/Video Path",
                            value=values[0],
                            placeholder="C:/path/to/image.jpg or C:/path/to/video.mp4",
                            info="Direct path to image, video, or frame-sequence folder"
                        )

                    with gr.Column(scale=2):
                        input_image_preview = gr.Image(
                            label="Input Preview (Image)",
                            type="filepath",
                            interactive=False,
                            height=250,
                            visible=False
                        )
                        input_video_preview = gr.Video(
                            label="Input Preview (Video)",
                            interactive=False,
                            height=250,
                            visible=False
                        )

                    with gr.Column(scale=2):
                        gan_model = gr.Dropdown(
                            label="GAN Model",
                            choices=gan_model_dropdown_choices,
                            value=gan_model_value,
                            info="Only GAN model weights are shown here. Model labels include native upscale metadata."
                        )
                        model_info = gr.Markdown(update_model_info(gan_model_value))

                input_cache_msg = gr.Markdown("", visible=False)
                input_detection_result = gr.Markdown("", visible=False)
                sizing_info = gr.Markdown("", visible=False, elem_classes=["resolution-info"])

            with gr.Accordion("Batch Processing", open=False):
                batch_enable = gr.Checkbox(
                    label="Enable Batch Processing",
                    value=values[1],
                    info="Process multiple files from directory"
                )
                with gr.Row():
                    batch_input = gr.Textbox(
                        label="Batch Input Folder",
                        value=values[2],
                        placeholder="Folder containing images/videos",
                        info="Directory with images, videos, or frame-sequence subfolders"
                    )
                    batch_output = gr.Textbox(
                        label="Batch Output Folder",
                        value=values[3],
                        placeholder="Output directory for batch results"
                    )

            gr.Markdown("#### Processing Settings")
            with gr.Group():
                # Legacy controls (kept for old presets, no longer used by vNext sizing)
                target_resolution = gr.Slider(
                    label="(Legacy) Target Resolution (longest side) [internal]",
                    minimum=512, maximum=4096, step=64,
                    value=values[5],
                    visible=False,
                    interactive=False,
                )
                target_res_warning = gr.Markdown("", visible=False)

                downscale_first = gr.Checkbox(
                    label="(Legacy) Downscale First if Needed [internal]",
                    value=values[6],
                    visible=False,
                    interactive=False,
                )

                auto_calculate_input = gr.Checkbox(
                    label="(Legacy) Auto-Calculate Input Resolution [internal]",
                    value=values[7],
                    visible=False,
                    interactive=False,
                )

                upscale_factor = gr.Number(
                    label="Upscale x (any factor)",
                    value=values[21] if len(values) > 21 else 4.0,
                    precision=2,
                    info="Target scale factor relative to input. For fixed-scale GAN models, input is pre-downscaled so one model pass reaches the target."
                )

                with gr.Row():
                    max_resolution = gr.Slider(
                        label="Max Resolution (max edge, 0 = no cap)",
                        minimum=0, maximum=8192, step=16,
                        value=values[22] if len(values) > 22 else 0,
                        info="Caps the LONG side (max(width,height)) of the target."
                    )
                    pre_downscale_then_upscale = gr.Checkbox(
                        label="Pre-downscale then upscale (auto when needed)",
                        value=values[23] if len(values) > 23 else False,
                        info="For fixed-scale GAN models this is applied automatically when needed to satisfy Upscale-x / Max Resolution without post-resize."
                    )
                    use_resolution_tab = gr.Checkbox(
                        label="Use Resolution & Scene Split Tab Settings",
                        value=values[8],
                        info="Apply Upscale-x, Max Resolution, and Pre-downscale settings from the Resolution tab. Recommended ON."
                    )

                with gr.Row():
                    tile_size = gr.Number(
                        label="Tile Size",
                        value=values[9],
                        precision=0,
                        info="Process image in tiles to reduce VRAM usage. 0 = process whole image. Try 512 for 8GB GPUs, 1024 for 12GB+."
                    )
                    overlap = gr.Number(
                        label="Tile Overlap",
                        value=values[10],
                        precision=0,
                        info="Pixels of overlap between tiles to reduce seam artifacts. Must be less than tile size."
                    )
                    batch_size = gr.Slider(
                        label="Batch Size (Frames per Iteration)",
                        minimum=1, maximum=16, step=1,
                        value=values[16],
                        info="Frames processed simultaneously for videos. Higher is faster but uses more VRAM."
                    )

            gr.Markdown("#### Quality and Performance")
            with gr.Group():
                with gr.Row():
                    denoising_strength = gr.Slider(
                        label="Denoising Strength",
                        minimum=0.0, maximum=1.0, step=0.05,
                        value=values[11],
                        info="Reduce noise/compression artifacts. 0 = no denoising, 1 = maximum."
                    )
                    sharpening = gr.Slider(
                        label="Output Sharpening",
                        minimum=0.0, maximum=2.0, step=0.1,
                        value=values[12],
                        info="Post-process sharpening. 0 = none, 1 = moderate, 2 = strong."
                    )

                with gr.Row():
                    color_correction = gr.Checkbox(
                        label="Color Correction",
                        value=values[13],
                        info="Maintain color accuracy by matching output colors to input."
                    )
                    gpu_acceleration = gr.Checkbox(
                        label="GPU Acceleration",
                        value=values[14] if cuda_available else False,
                        info=f"{gpu_hint} | Use GPU for processing. CPU fallback is significantly slower.",
                        interactive=cuda_available
                    )
                    gpu_device = gr.Textbox(
                        label="GPU Device",
                        value=values[15] if cuda_available else "",
                        placeholder="0 or all" if cuda_available else "CUDA not available",
                        info=f"GPU device ID(s). {cuda_count} GPU(s) detected. Single ID (0) for one GPU, 'all' for all available.",
                        interactive=cuda_available
                    )
                gpu_device_warning = gr.Markdown("", visible=False)

            gr.Markdown("#### Output Settings")
            with gr.Group():
                output_format_default = str(values[17]).strip().lower()
                if output_format_default not in {"auto", "png", "jpg", "webp"}:
                    output_format_default = "png"
                with gr.Row():
                    output_format_gan = gr.Dropdown(
                        label="Output Format",
                        choices=["auto", "png", "jpg", "webp"],
                        value=output_format_default,
                        info="'auto' matches input format"
                    )
                    output_quality_gan = gr.Slider(
                        label="Output Quality",
                        minimum=70, maximum=100, step=5,
                        value=values[18],
                        info="Quality for lossy formats (JPG/WebP)"
                    )

                with gr.Row():
                    save_metadata = gr.Checkbox(
                        label="Save Processing Metadata",
                        value=values[19],
                        info="Embed processing information in output files"
                    )
                    create_subfolders = gr.Checkbox(
                        label="Create Subfolders by Model",
                        value=values[20],
                        info="Organize outputs in model-named subdirectories"
                    )

        with gr.Column(scale=2):
            gr.Markdown("### Output / Actions")

            status_box = gr.Markdown(value="Ready for processing.")
            progress_indicator = gr.Markdown(value="", visible=True)
            log_box = gr.Textbox(
                label="Processing Log",
                value="",
                lines=10,
                buttons=["copy"]
            )

            with gr.Accordion("Upscaled Output", open=True):
                output_video = gr.Video(
                    label="Upscaled Video",
                    interactive=False,
                    visible=False,
                    buttons=["download"],
                )
                output_image = gr.Image(
                    label="Upscaled Image",
                    interactive=False,
                    visible=False,
                    buttons=["download"],
                )

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

            chunk_status = gr.Markdown("", visible=False)
            chunk_gallery = gr.Gallery(
                label="Chunk Preview",
                visible=False,
                columns=4,
                rows=2,
                height=220,
                object_fit="contain",
            )
            chunk_preview_video = gr.Video(
                label="Selected Chunk",
                interactive=False,
                visible=False,
                buttons=["download"],
            )

            batch_gallery = gr.Gallery(
                label="Batch Results",
                visible=False,
                columns=4,
                rows=2,
                height="auto",
                object_fit="contain",
                buttons=["download"],
            )

            last_processed = gr.Markdown("Batch processing results will appear here.")

            with gr.Row():
                upscale_btn = gr.Button(
                    "Start Upscaling",
                    variant="primary",
                    size="lg",
                    elem_classes=["action-btn", "action-btn-upscale"],
                )
                cancel_btn = gr.Button(
                    "Cancel",
                    variant="stop",
                    elem_classes=["action-btn", "action-btn-cancel"],
                )
                preview_btn = gr.Button(
                    "Preview First Frame",
                    size="lg",
                    elem_classes=["action-btn", "action-btn-preview"],
                )

            cancel_confirm = gr.Checkbox(
                label="Confirm cancel (required for safety)",
                value=False,
                info="Enable this checkbox to confirm cancellation of processing"
            )

            with gr.Row():
                open_outputs_btn = gr.Button(
                    "Open Outputs Folder",
                    elem_classes=["action-btn", "action-btn-open"],
                )
                clear_temp_btn = gr.Button(
                    "Clear Temp Files",
                    elem_classes=["action-btn", "action-btn-clear"],
                )

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
                tab_name="gan",
                inputs_list=[],
                base_dir=base_dir,
                models_list=models_list,
                open_accordion=True,
            )

    gan_model.change(
        fn=update_model_info,
        inputs=gan_model,
        outputs=model_info,
    )

    with gr.Accordion("About GAN Upscaling", open=False):
        gr.Markdown("""
        #### GAN-Based Image and Video Upscaling

        **Fixed Scale Factors:**
        - Models are trained for specific upscale ratios (2x, 4x, etc.)
        - Arbitrary upscale factors are handled with cap-aware pre-downscale + upscale

        **Use Cases:**
        - High-quality image enlargement
        - Video upscaling by frame extraction, frame upscaling, and video reconstruction
        - Batch processing of mixed image/video collections
        """)

    # ============================================================================
    # GAN PRESET INPUT LIST - MUST match GAN_ORDER in gan_service.py
    # Adding controls? Update gan_defaults(), GAN_ORDER, and this list in sync.
    # Current count: 24 components (includes vNext sizing)
    # ============================================================================
    
    inputs_list = [
        input_path, batch_enable, batch_input, batch_output, gan_model,
        target_resolution, downscale_first, auto_calculate_input, use_resolution_tab, tile_size, overlap,
        denoising_strength, sharpening, color_correction, gpu_acceleration, gpu_device,
        batch_size, output_format_gan, output_quality_gan, save_metadata, create_subfolders,
        # vNext sizing
        upscale_factor, max_resolution, pre_downscale_then_upscale
    ]
    
    # Development validation
    if len(inputs_list) != len(GAN_ORDER):
        import logging
        logging.getLogger("GANTab").error(
            f"ERROR: inputs_list ({len(inputs_list)}) != GAN_ORDER ({len(GAN_ORDER)})"
        )

    # Wire up event handlers
    
    # FIXED: Live CUDA device validation for GAN tab
    def validate_cuda_device_live_gan(cuda_device_val):
        """Live CUDA validation for GAN models (enforces single GPU)"""
        if not cuda_device_val or not cuda_device_val.strip():
            return gr.update(value="", visible=False)
        
        try:
            if not cuda_available or cuda_count <= 0:
                return gr.update(value="WARNING: CUDA not detected. GPU acceleration disabled.", visible=True)
            
            device_str = str(cuda_device_val).strip()
            device_count = cuda_count
            
            if device_str.lower() == "all":
                return gr.update(value=f"WARNING: GAN models use single GPU. 'all' will use GPU 0 (of {device_count} available)", visible=True)
            
            devices = [d.strip() for d in device_str.replace(" ", "").split(",") if d.strip()]
            
            invalid_devices = []
            valid_devices = []
            
            for device in devices:
                if not device.isdigit():
                    invalid_devices.append(device)
                else:
                    device_id = int(device)
                    if device_id >= device_count:
                        invalid_devices.append(f"{device} (max: {device_count-1})")
                    else:
                        valid_devices.append(device_id)
            
            if invalid_devices:
                return gr.update(
                    value=f"ERROR: Invalid device ID(s): {', '.join(invalid_devices)}. Available: 0-{device_count-1}",
                    visible=True
                )
            
            if len(valid_devices) > 1:
                return gr.update(
                    value=f"WARNING: GAN models use single GPU. Will use GPU {valid_devices[0]} (ignoring others)",
                    visible=True
                )
            elif len(valid_devices) == 1:
                return gr.update(
                    value=f"OK: Using GPU {valid_devices[0]}",
                    visible=True
                )
            
            return gr.update(value="", visible=False)
        except Exception as e:
            return gr.update(value=f"WARNING: Validation error: {str(e)}", visible=True)
    
    # Wire up live CUDA validation
    gpu_device.change(
        fn=validate_cuda_device_live_gan,
        inputs=gpu_device,
        outputs=gpu_device_warning
    )

    # Input handling
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

    def _build_sizing_info(
        input_path_val: str,
        model_name: str,
        use_global: bool,
        local_scale_x: float,
        local_max_edge: int,
        local_pre_down: bool,
        state,
    ) -> gr.update:
        try:
            from shared.gan_runner import get_gan_model_metadata
            meta = get_gan_model_metadata(model_name, base_dir)
            model_scale = int(meta.scale or 4)
        except Exception:
            model_scale = 4

        return build_fixed_scale_analysis_update(
            input_path_val=input_path_val,
            model_scale=int(model_scale),
            use_global=bool(use_global),
            local_scale_x=float(local_scale_x or 4.0),
            local_max_edge=int(local_max_edge or 0),
            local_pre_down=bool(local_pre_down),
            state=state,
            model_label="GAN",
            runtime_label=f"GAN pipeline (fixed {int(model_scale)}x pass)",
            auto_scene_scan=True,
        )

    def cache_input(val, model_val, use_global, scale_x, max_edge, pre_down, state):
        state["seed_controls"]["last_input_path"] = val if val else ""
        det = _build_input_detection_md(val or "")
        info = _build_sizing_info(val or "", model_val, bool(use_global), scale_x, max_edge, pre_down, state)
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

    input_file.upload(
        fn=cache_input,
        inputs=[input_file, gan_model, use_resolution_tab, upscale_factor, max_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state]
    )

    # When the user clears the upload, clear the path and hide dependent panels.
    def clear_on_upload_clear(file_path, state):
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
            "",  # input_path
            gr.update(value="", visible=False),  # input_cache_msg
            img_prev,
            vid_prev,
            gr.update(value="", visible=False),  # input_detection_result
            gr.update(value="", visible=False),  # sizing_info
            state,
        )

    input_file.change(
        fn=clear_on_upload_clear,
        inputs=[input_file, shared_state],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )

    def update_from_path(val, model_val, use_global, scale_x, max_edge, pre_down, state):
        det = _build_input_detection_md(val or "")
        info = _build_sizing_info(val or "", model_val, bool(use_global), scale_x, max_edge, pre_down, state)
        img_prev, vid_prev = preview_updates(val)
        return (
            gr.update(value="OK: Input path updated.", visible=True),
            img_prev,
            vid_prev,
            det,
            info,
            state,
        )

    input_path.change(
        fn=update_from_path,
        inputs=[input_path, gan_model, use_resolution_tab, upscale_factor, max_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state]
    )

    # Refresh sizing info when settings change
    def refresh_sizing(scale_x, max_edge, pre_down, use_global, model_val, path_val, state):
        info = _build_sizing_info(path_val or "", model_val, bool(use_global), scale_x, max_edge, pre_down, state)
        return info, state

    for comp in [upscale_factor, max_resolution, pre_downscale_then_upscale, use_resolution_tab, gan_model]:
        comp.change(
            fn=refresh_sizing,
            inputs=[upscale_factor, max_resolution, pre_downscale_then_upscale, use_resolution_tab, gan_model, input_path, shared_state],
            outputs=[sizing_info, shared_state],
            trigger_mode="always_last",
        )

    def refresh_chunk_preview_ui(state):
        preview = (state or {}).get("seed_controls", {}).get("gan_chunk_preview", {})
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
            videos = (state or {}).get("seed_controls", {}).get("gan_chunk_preview", {}).get("videos", [])
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
            "Run logs and chunk previews will update once processing starts."
        )
        return (
            gr.update(value=title),
            gr.update(value=f"Queued and waiting for active processing slot. Queue position: {pos}."),
            _queue_status_indicator(title, subtitle, spinning=True),
            gr.update(),
            gr.update(),
            "Waiting in queue",
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
            "Removed from queue",
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
            "Queue disabled: request ignored",
            gr.update(),
            gr.update(),
            gr.update(),
            safe_state,
        )

    def run_upscale_with_queue(upload, *args, progress=gr.Progress()):
        live_state = args[-1] if (args and isinstance(args[-1], dict)) else {}
        queued_state = snapshot_queue_state(live_state)
        queued_global_settings = snapshot_global_settings(global_settings)
        queue_enabled = bool(queued_global_settings.get("queue_enabled", True))
        ticket = queue_manager.submit("GAN", "Upscale")
        acquired_slot = queue_manager.is_active(ticket.job_id)

        try:
            if not queue_enabled:
                if not acquired_slot:
                    queue_manager.cancel_waiting([ticket.job_id])
                    yield _queue_disabled_busy_output(live_state)
                    return
                for payload in service["run_action"](
                    upload,
                    *args[:-1],
                    preview_only=False,
                    state=queued_state,
                    progress=progress,
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
                upload,
                *args[:-1],
                preview_only=False,
                state=queued_state,
                progress=progress,
                global_settings_snapshot=queued_global_settings,
            ):
                yield merge_payload_state(payload, live_state)
        finally:
            if acquired_slot:
                queue_manager.complete(ticket.job_id)
            else:
                queue_manager.cancel_waiting([ticket.job_id])

    def run_preview_with_snapshot(upload, *args, progress=gr.Progress()):
        live_state = args[-1] if (args and isinstance(args[-1], dict)) else {}
        queued_state = snapshot_queue_state(live_state)
        queued_global_settings = snapshot_global_settings(global_settings)
        for payload in service["run_action"](
            upload,
            *args[:-1],
            preview_only=True,
            state=queued_state,
            progress=progress,
            global_settings_snapshot=queued_global_settings,
        ):
            yield merge_payload_state(payload, live_state)

    # Main processing with gr.Progress - include input_file upload
    run_evt = upscale_btn.click(
        fn=run_upscale_with_queue,
        inputs=[input_file] + inputs_list + [shared_state],
        outputs=[
            status_box, log_box, progress_indicator, output_image, output_video,
            last_processed, image_slider, video_comparison_html, batch_gallery, shared_state
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
            status_box, log_box, progress_indicator, output_image, output_video,
            last_processed, image_slider, video_comparison_html, batch_gallery, shared_state
        ]
    )
    preview_evt.then(
        fn=refresh_chunk_preview_ui,
        inputs=[shared_state],
        outputs=[chunk_status, chunk_gallery, chunk_preview_video],
    )
    
    # NOTE: Legacy target_resolution is hidden; sizing is now driven by Upscale-x.

    cancel_btn.click(
        fn=lambda ok, state: service["cancel_action"](state) if ok else (gr.update(value="WARNING: Enable 'Confirm cancel' to stop."), "", state),
        inputs=[cancel_confirm, shared_state],
        outputs=[status_box, log_box, shared_state]
    )

    # Utility functions
    open_outputs_btn.click(
        fn=lambda state: (service["open_outputs_folder"](state), state),
        inputs=shared_state,
        outputs=[status_box, shared_state]
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
        tab_name="gan",
    )

    return {
        "inputs_list": inputs_list,
        "preset_dropdown": preset_dropdown,
        "preset_status": preset_status,
    }



