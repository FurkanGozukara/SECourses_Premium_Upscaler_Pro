"""
FlashVSR+ Tab - Self-contained modular implementation
Real-time diffusion-based streaming video super-resolution
UPDATED: Now uses Universal Preset System
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any

from shared.services.flashvsr_service import (
    build_flashvsr_callbacks, FLASHVSR_ORDER
)
from shared.path_utils import get_media_dimensions, normalize_path
from shared.resolution_calculator import estimate_fixed_scale_upscale_plan_from_dims
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
            gpu_hint = f"✅ Detected {cuda_count} CUDA GPU(s) - GPU acceleration available\n⚠️ FlashVSR+ uses single GPU only (multi-GPU not supported)"
        else:
            gpu_hint = "⚠️ CUDA not detected (nvidia-smi unavailable or no NVIDIA GPU) - Processing will use CPU (significantly slower)"
    except Exception as e:
        gpu_hint = f"❌ CUDA detection failed: {str(e)}"
        cuda_available = False

    # Layout
    gr.Markdown("### ⚡ FlashVSR+ - Real-Time Diffusion Video Super-Resolution")
    gr.Markdown("*High-quality real-time video upscaling with diffusion models*")
    
    # Show GPU warning if not available
    if not cuda_available:
        gr.Markdown(
            f'<div style="background: #fff3cd; padding: 12px; border-radius: 8px; border: 1px solid #ffc107;">'
            f'<strong>⚠️ GPU Acceleration Unavailable</strong><br>'
            f'{gpu_hint}<br><br>'
            f'FlashVSR+ requires CUDA for optimal performance. CPU mode is extremely slow.'
            f'</div>',
            elem_classes="warning-text"
        )

    with gr.Row():
        # Left Column: Input & Settings
        with gr.Column(scale=3):
            gr.Markdown("#### 📁 Input")

            with gr.Row():
                input_file = gr.File(
                    label="Upload video or image (optional)",
                    type="filepath",
                    file_types=["video", "image"]
                )
                with gr.Column():
                    input_image_preview = gr.Image(
                        label="📸 Input Preview (Image)",
                        type="filepath",
                        interactive=False,
                        height=250,
                        visible=False
                    )
                    input_video_preview = gr.Video(
                        label="🎬 Input Preview (Video)",
                        interactive=False,
                        height=250,
                        visible=False
                    )
            input_path = gr.Textbox(
                label="Input Path",
                value=values[0],
                placeholder="C:/path/to/video.mp4 or C:/path/to/frames/",
                info="Video file or image sequence folder"
            )
            input_cache_msg = gr.Markdown("", visible=False)
            sizing_info = gr.Markdown("", visible=False, elem_classes=["resolution-info"])
            input_detection_result = gr.Markdown("", visible=False)
            
            # Model Configuration
            gr.Markdown("#### 🤖 Model Configuration")
            
            with gr.Row():
                scale = gr.Dropdown(
                    label="Upscale Factor",
                    choices=["2", "4"],
                    value="2" if str(values[2]).strip() == "2" else "4",
                    info="2x or 4x upscaling"
                )
                version = gr.Dropdown(
                    label="Model Version",
                    choices=["10", "11"],
                    value=str(values[3]) if str(values[3]) in {"10", "11"} else "10",
                    info="v10 = faster, v11 = higher quality"
                )
                mode = gr.Dropdown(
                    label="Pipeline Mode",
                    choices=["tiny", "tiny-long", "full"],
                    value=str(values[4]) if str(values[4]) in {"tiny", "tiny-long", "full"} else "tiny",
                    info="tiny = fastest (4-6GB VRAM), tiny-long = balanced (5-7GB), full = best quality (8-12GB)"
                )

            # vNext sizing controls (any x + max edge + pre-downscale)
            with gr.Group():
                use_resolution_tab = gr.Checkbox(
                    label="🔗 Use Resolution & Scene Split Tab Settings",
                    value=values[20] if len(values) > 20 else True,
                    info="Apply Upscale-x, Max Resolution, and Pre-downscale settings from Resolution tab. Recommended ON."
                )

                upscale_factor = gr.Number(
                    label="Upscale x (any factor)",
                    value=values[21] if len(values) > 21 else float(values[2]),
                    precision=2,
                    info="Desired effective upscale. For FlashVSR fixed 2x/4x models, input is pre-downscaled to hit the target without exceeding Max Resolution."
                )

                with gr.Row():
                    max_target_resolution = gr.Slider(
                        label="Max Resolution (max edge, 0 = no cap)",
                        minimum=0, maximum=8192, step=16,
                        value=values[22] if len(values) > 22 else 0,
                        info="Caps the LONG side (max(width,height)) of the target."
                    )
                    pre_downscale_then_upscale = gr.Checkbox(
                        label="⬇️➡️⬆️ Pre-downscale then upscale (auto when needed)",
                        value=values[23] if len(values) > 23 else False,
                        info="For fixed-scale FlashVSR models this is applied automatically when needed to satisfy Upscale-x / Max Resolution without post-resize."
                    )
            
            # Model info display with metadata
            model_info_display = gr.Markdown("")
            
            def update_flashvsr_model_info(version_val, mode_val, scale_val):
                """Display model metadata information"""
                from shared.models.flashvsr_meta import get_flashvsr_metadata
                
                model_id = f"v{version_val}_{mode_val}_{scale_val}x"
                metadata = get_flashvsr_metadata(model_id)
                
                if metadata:
                    info_lines = [
                        f"**📊 Model: {metadata.name}**",
                        f"**VRAM Estimate:** ~{metadata.estimated_vram_gb:.1f}GB",
                        f"**Speed:** {metadata.speed_tier.title()} | **Quality:** {metadata.quality_tier.replace('_', ' ').title()}",
                        f"**Multi-GPU:** {'❌ Not supported' if not metadata.supports_multi_gpu else '✅ Supported'}",
                        f"**Compile:** {'✅ Compatible' if metadata.compile_compatible else '❌ Not supported'}",
                    ]
                    if metadata.notes:
                        info_lines.append(f"\n💡 {metadata.notes}")
                    
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
            gr.Markdown("#### ⚙️ Processing Settings")
            
            with gr.Group():
                dtype = gr.Dropdown(
                    label="Precision",
                    choices=["fp16", "bf16"],
                    value=str(values[12]) if str(values[12]) in {"fp16", "bf16"} else "bf16",
                    info="bf16 = faster, more stable. fp16 = broader compatibility"
                )
                
                device = gr.Textbox(
                    label="Device (Single GPU Only)",
                    value=values[13] if cuda_available else "cpu",
                    placeholder="auto, cuda:0, cpu" if cuda_available else "CPU only (no CUDA)",
                    info=f"{gpu_hint}\nauto = automatic GPU selection, cuda:0 = specific GPU, cpu = CPU mode. Multi-GPU NOT supported by FlashVSR+.",
                    interactive=cuda_available
                )
                
                seed = gr.Number(
                    label="Random Seed",
                    value=values[11],
                    precision=0,
                    info="Seed for reproducibility. 0 = random"
                )
                
                attention = gr.Dropdown(
                    label="Attention Mode",
                    choices=["sage", "block"],
                    value=str(values[16]) if str(values[16]) in {"sage", "block"} else "sage",
                    info="sage = default, block = alternative implementation"
                )
            
            # Memory Optimization
            gr.Markdown("#### 💾 Memory Optimization (Tiling)")
            
            with gr.Group():
                tiled_vae = gr.Checkbox(
                    label="Enable VAE Tiling",
                    value=values[5],
                    info="Reduce VRAM usage during VAE encoding/decoding. Essential for high resolutions."
                )
                
                tiled_dit = gr.Checkbox(
                    label="Enable DiT Tiling",
                    value=values[6],
                    info="Reduce VRAM usage during diffusion inference. Enables processing larger videos."
                )
                
                tile_size = gr.Slider(
                    label="Tile Size",
                    minimum=128, maximum=512, step=32,
                    value=values[7],
                    info="Size of each tile. Larger = faster but more VRAM"
                )
                
                overlap = gr.Slider(
                    label="Tile Overlap",
                    minimum=8, maximum=64, step=8,
                    value=values[8],
                    info="Overlap between tiles to reduce seams. Higher = smoother"
                )
                
                unload_dit = gr.Checkbox(
                    label="Unload DiT Before Decoding",
                    value=values[9],
                    info="Free VRAM before VAE decoding. Slower but uses less memory."
                )
            
            # Quality Settings
            gr.Markdown("#### 🎨 Quality Settings")
            
            with gr.Group():
                color_fix = gr.Checkbox(
                    label="Color Correction",
                    value=values[10],
                    info="Maintain color accuracy. Recommended ON."
                )
                
                fps_flashvsr = gr.Number(
                    label="Output FPS (image sequences only)",
                    value=values[14],
                    precision=0,
                    info="Frame rate for image sequence outputs. Ignored for video inputs."
                )
                
                quality = gr.Slider(
                    label="Video Quality",
                    minimum=1, maximum=10, step=1,
                    value=values[15],
                    info="Output quality. 1 = lowest, 10 = highest. 6 is recommended."
                )
        
        # Right Column: Output & Controls
        with gr.Column(scale=2):
            gr.Markdown("#### 🎯 Output & Actions")
            
            status_box = gr.Markdown(value="Ready.")
            progress_indicator = gr.Markdown(value="", visible=True)
            
            log_box = gr.Textbox(
                label="📋 Processing Log",
                value="",
                lines=12,
                buttons=["copy"]
            )

            output_override = gr.Textbox(
                label="Output Override (folder or .mp4 file)",
                value=values[1],
                placeholder="Leave empty for auto naming",
                info="Optional custom output location. A folder saves into that folder. A .mp4 file path renames the final output to that exact file.",
            )
            
            with gr.Accordion("🎬 Upscaled Output", open=True):
                output_video = gr.Video(
                    label="🎬 Upscaled Video",
                    interactive=False,
                    visible=False,
                    buttons=["download"],
                )
                output_image = gr.Image(
                    label="🖼️ Upscaled Image",
                    interactive=False,
                    visible=False,
                    buttons=["download"],
                )
            
            # Comparison
            image_slider = gr.ImageSlider(
                label="🔍 Comparison",
                interactive=False,
                slider_position=50,
                max_height=1000,
                buttons=["download", "fullscreen"],
                elem_classes=["native-image-comparison-slider"],
            )
            
            video_comparison_html = gr.HTML(
                label="🎬 Video Comparison",
                value="",
                js_on_load=get_video_comparison_js_on_load(),
                visible=False
            )
            
            chunk_status = gr.Markdown("", visible=False)
            chunk_gallery = gr.Gallery(
                label="🧩 Chunk Preview",
                visible=False,
                columns=4,
                rows=2,
                height=220,
                object_fit="contain",
            )
            chunk_preview_video = gr.Video(
                label="🎬 Selected Chunk",
                interactive=False,
                visible=False,
                buttons=["download"],
            )

            # Action buttons
            with gr.Row():
                upscale_btn = gr.Button(
                    "🚀 Start Upscaling",
                    variant="primary",
                    size="lg",
                    elem_classes=["action-btn", "action-btn-upscale"],
                )
                cancel_btn = gr.Button(
                    "⏹️ Cancel",
                    variant="stop",
                    elem_classes=["action-btn", "action-btn-cancel"],
                )
            
            cancel_confirm = gr.Checkbox(
                label="⚠️ Confirm cancel (required for safety)",
                value=False,
                info="Enable this checkbox to confirm cancellation of processing"
            )
            
            # Utility buttons
            with gr.Row():
                open_outputs_btn = gr.Button(
                    "📂 Open Outputs",
                    elem_classes=["action-btn", "action-btn-open"],
                )
                clear_temp_btn = gr.Button(
                    "🗑️ Clear Temp",
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
                tab_name="flashvsr",
                inputs_list=[],
                base_dir=base_dir,
                models_list=models_list,
                open_accordion=True,
            )
            
            # Info
            gr.Markdown("""
            #### ℹ️ About FlashVSR+
            
            **Real-time Diffusion Video SR:**
            - Streaming processing for memory efficiency
            - Multiple pipeline modes for speed/quality tradeoff
            - Automatic model download from HuggingFace
            
            **Recommended Settings:**
            - Mode: `tiny` for real-time, `full` for best quality
            - Enable tiling for high-res or limited VRAM
            - Use color fix for accurate colors
            """)
    
    # Batch processing (optional)
    with gr.Accordion("📦 Batch Processing", open=False):
        batch_enable = gr.Checkbox(
            label="Enable Batch",
            value=values[17],
            info="Process multiple files"
        )
        batch_input = gr.Textbox(
            label="Batch Input Folder",
            value=values[18],
            placeholder="Folder with videos"
        )
        batch_output = gr.Textbox(
            label="Batch Output Folder",
            value=values[19],
            placeholder="Output directory"
        )
    
    # Collect inputs
    inputs_list = [
        input_path, output_override, scale, version, mode,
        tiled_vae, tiled_dit, tile_size, overlap, unload_dit,
        color_fix, seed, dtype, device, fps_flashvsr,
        quality, attention, batch_enable, batch_input, batch_output
        , use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale
    ]

    # Development validation: inputs_list must stay aligned with FLASHVSR_ORDER
    if len(inputs_list) != len(FLASHVSR_ORDER):
        import logging
        logging.getLogger("FlashVSRTab").error(
            f"❌ inputs_list ({len(inputs_list)}) != FLASHVSR_ORDER ({len(FLASHVSR_ORDER)})"
        )
    
    # Wire up events
    def cache_input(val, state):
        try:
            state = state or {}
            state.setdefault("seed_controls", {})
            state["seed_controls"]["last_input_path"] = val if val else ""
        except Exception:
            pass
        return val or "", gr.update(value="✅ Input cached", visible=True), state

    def _build_input_detection_md(path_val: str) -> gr.update:
        from shared.input_detector import detect_input
        if not path_val or not str(path_val).strip():
            # Hide when empty (clearing input should clear this panel).
            return gr.update(value="", visible=False)
        try:
            info = detect_input(path_val)
            if not info.is_valid:
                return gr.update(value=f"❌ **Invalid Input**\n\n{info.error_message}", visible=True)
            parts = [f"✅ **Input Detected: {info.input_type.upper()}**"]
            if info.input_type == "frame_sequence":
                parts.append(f"&nbsp;&nbsp;📁 Pattern: `{info.frame_pattern}`")
                parts.append(f"&nbsp;&nbsp;🎞️ Frames: {info.frame_start}-{info.frame_end}")
                if info.missing_frames:
                    parts.append(f"&nbsp;&nbsp;⚠️ Missing: {len(info.missing_frames)}")
            elif info.input_type == "directory":
                parts.append(f"&nbsp;&nbsp;📂 Files: {info.total_files}")
            elif info.input_type in ["video", "image"]:
                parts.append(f"&nbsp;&nbsp;📄 Format: **{info.format.upper()}**")
            return gr.update(value=" ".join(parts), visible=True)
        except Exception as e:
            return gr.update(value=f"❌ **Detection Error**\n\n{str(e)}", visible=True)

    def _build_sizing_info(path_val, model_scale_val, use_global, local_scale_x, local_max_edge, local_pre_down, state):
        if not path_val or not str(path_val).strip():
            return gr.update(visible=False)
        seed_controls = (state or {}).get("seed_controls", {})
        enable_max = bool(seed_controls.get("enable_max_target", True)) if use_global else True
        scale_x = float(seed_controls.get("upscale_factor_val", 4.0) or 4.0) if use_global else float(local_scale_x or 4.0)
        max_edge = int(seed_controls.get("max_resolution_val", 0) or 0) if use_global else int(local_max_edge or 0)
        pre_down = bool(seed_controls.get("ratio_downscale", False)) if use_global else bool(local_pre_down)
        if not enable_max:
            max_edge = 0

        # Dimensions (file only; directory sizing preview is best-effort)
        p = Path(normalize_path(path_val))
        rep = None
        if p.exists() and p.is_dir():
            # pick first media file
            exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".mp4", ".mov", ".mkv", ".avi", ".webm")
            items = [x for x in sorted(p.iterdir()) if x.is_file() and x.suffix.lower() in exts]
            rep = str(items[0]) if items else None
        else:
            rep = str(p)
        dims = get_media_dimensions(rep) if rep else None
        if not dims:
            return gr.update(value="⚠️ Could not determine input dimensions for sizing preview.", visible=True)
        w, h = dims

        ms = int(model_scale_val or 4)
        plan = estimate_fixed_scale_upscale_plan_from_dims(
            int(w), int(h),
            requested_scale=float(scale_x),
            model_scale=ms,
            max_edge=int(max_edge or 0),
            force_pre_downscale=True,
        )

        out_w = plan.final_saved_width or plan.resize_width
        out_h = plan.final_saved_height or plan.resize_height
        input_short = min(plan.input_width, plan.input_height)
        out_short = min(int(out_w), int(out_h))

        items = []
        items.append(f"📐 <strong>Input:</strong> {plan.input_width}×{plan.input_height} (short side: {input_short}px)")
        t = f"🎯 <strong>Target setting:</strong> upscale {scale_x:g}x"
        if max_edge and max_edge > 0:
            t += f", max edge {max_edge}px (effective {plan.effective_scale:.2f}x)"
        items.append(t)
        # For fixed-scale models, pre-downscale is mandatory when the effective scale < model_scale.
        if plan.pre_downscale_then_upscale and plan.preprocess_scale < 0.999999:
            items.append(f"🧩 <strong>Preprocess:</strong> {plan.input_width}×{plan.input_height} → {plan.preprocess_width}×{plan.preprocess_height} (×{plan.preprocess_scale:.3f})")
        items.append(f"🧱 <strong>Model pass:</strong> fixed {ms}x FlashVSR")
        items.append(f"✅ <strong>Final saved output:</strong> {int(out_w)}×{int(out_h)}")
        if out_short < input_short:
            items.append(f"📉 <strong>Mode:</strong> Downscaling (output short side {out_short}px < input short side {input_short}px)")
        elif out_short > input_short:
            items.append(f"📈 <strong>Mode:</strong> Upscaling (output short side {out_short}px > input short side {input_short}px)")
        else:
            items.append("➡️ <strong>Mode:</strong> Keep size (output short side matches input short side)")
        if plan.notes:
            for n in plan.notes:
                items.append(f"ℹ️ {n}")
        html = '<div style="font-size: 1.15em; line-height: 1.8;">' + "<br>".join(items) + "</div>"
        return gr.update(value=html, visible=True)
    
    input_file.upload(
        fn=cache_input,
        inputs=[input_file, shared_state],
        outputs=[input_path, input_cache_msg, shared_state]
    )

    # Preview + sizing + detection refresh on input changes
    def refresh_panels(path_val, scale_val, use_global, scale_x, max_edge, pre_down, state):
        img_prev, vid_prev = preview_updates(path_val)
        det = _build_input_detection_md(path_val or "")
        info = _build_sizing_info(path_val, int(scale_val), bool(use_global), scale_x, max_edge, pre_down, state)
        return img_prev, vid_prev, det, info, state

    input_file.change(
        fn=lambda p, state: (*preview_updates(p), _build_input_detection_md(p or ""), gr.update(visible=False), state),
        inputs=[input_file, shared_state],
        outputs=[input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state]
    )

    # When upload is cleared, also clear the textbox path so sizing/detection panels disappear.
    def clear_input_path_on_upload_clear(file_path, state):
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
        fn=clear_input_path_on_upload_clear,
        inputs=[input_file, shared_state],
        outputs=[input_path, input_cache_msg, shared_state],
    )

    def cache_input_path_only(path_val, state):
        try:
            state = state or {}
            state.setdefault("seed_controls", {})
            state["seed_controls"]["last_input_path"] = path_val if path_val else ""
        except Exception:
            pass
        return gr.update(value="✅ Input path updated.", visible=True), state

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
    
    def _queued_waiting_output(state, ticket_id: str, position: int):
        safe_state = state or {}
        pos = max(1, int(position)) if position else "?"
        return (
            gr.update(value=f"Queue waiting: {ticket_id} (position {pos})"),
            gr.update(value=f"Queued and waiting for active processing slot. Queue position: {pos}."),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            safe_state,
        )

    def _queued_cancelled_output(state, ticket_id: str):
        safe_state = state or {}
        return (
            gr.update(value=f"Queue item removed: {ticket_id}"),
            gr.update(value="This queued request was removed before processing started."),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            safe_state,
        )

    def _queue_disabled_busy_output(state):
        safe_state = state or {}
        return (
            gr.update(value="Processing already in progress (queue disabled)."),
            gr.update(value="Enable 'Enable Queue' in Global Settings to stack additional requests."),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
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
                args[0],
                *args[1:-1],
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

    # Main processing
    run_evt = upscale_btn.click(
        fn=run_upscale_with_queue,
        inputs=[input_file] + inputs_list + [shared_state],
        outputs=[status_box, log_box, output_video, output_image, image_slider, video_comparison_html, shared_state],
        concurrency_limit=32,
        concurrency_id="app_processing_queue",
        trigger_mode="multiple",
    )
    run_evt.then(
        fn=refresh_chunk_preview_ui,
        inputs=[shared_state],
        outputs=[chunk_status, chunk_gallery, chunk_preview_video],
    )
    
    cancel_btn.click(
        fn=lambda ok: service["cancel_action"]() if ok else (gr.update(value="⚠️ Enable 'Confirm cancel' to stop."), ""),
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

