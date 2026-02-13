"""
Output & Comparison Tab - Self-contained modular implementation
UPDATED: Now uses Universal Preset System
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any

from shared.services.output_service import (
    build_output_callbacks, OUTPUT_ORDER
)
from shared.models import (
    get_seedvr2_model_names,
    get_flashvsr_model_names,
    get_rife_model_names,
    scan_gan_models
)
from shared.models.rife_meta import get_rife_default_model
from shared.video_codec_options import (
    get_codec_choices,
    get_pixel_format_choices,
    get_codec_info,
    get_pixel_format_info,
    ENCODING_PRESETS,
    AUDIO_CODECS
)
from shared.video_comparison_slider import (
    create_video_comparison_html,
    get_video_comparison_js_on_load,
)
from ui.universal_preset_section import (
    universal_preset_section,
    wire_universal_preset_events,
)
from shared.universal_preset import dict_to_values


def output_tab(preset_manager, shared_state: gr.State, base_dir: Path, global_settings: Dict[str, Any] = None):
    """
    Self-contained Output & Comparison tab.
    Handles output format and comparison settings shared across ALL upscaler models.
    """

    seed_controls = shared_state.value.get("seed_controls", {})
    shared_models = seed_controls.get("available_models", [])
    if not isinstance(shared_models, list):
        shared_models = []
    combined_models = sorted(
        {str(model_name).strip() for model_name in (shared_models or []) if str(model_name).strip()}
    )
    if not combined_models:
        # Fallback for legacy state/migrations: discover model lists directly.
        seedvr2_models = get_seedvr2_model_names()
        gan_models = scan_gan_models(base_dir)
        flashvsr_models = get_flashvsr_model_names()
        rife_models = get_rife_model_names(base_dir)
        combined_models = sorted(list({
            *seedvr2_models,
            *gan_models,
            *flashvsr_models,
            *rife_models
        }))
    else:
        rife_models = get_rife_model_names(base_dir)

    # Build service callbacks
    service = build_output_callbacks(preset_manager, shared_state, combined_models, global_settings)

    # Get defaults
    defaults = service["defaults"]
    
    # UNIVERSAL PRESET: Load from shared_state
    seed_controls = shared_state.value.get("seed_controls", {})
    output_settings = seed_controls.get("output_settings", {})
    models_list = seed_controls.get("available_models", combined_models)
    
    # Merge with defaults
    merged_defaults = defaults.copy()
    for key, value in output_settings.items():
        if value is not None:
            merged_defaults[key] = value
    
    values = [merged_defaults[k] for k in OUTPUT_ORDER]
    # Layout
    gr.Markdown("### 🎭 Output & Comparison Settings")
    gr.Markdown("*Configure output formats, FPS handling, and comparison display options shared across all upscaler models*")

    try:
        overwrite_idx = OUTPUT_ORDER.index("overwrite_existing_batch")
        overwrite_existing_batch_default = bool(values[overwrite_idx])
    except Exception:
        overwrite_existing_batch_default = bool(seed_controls.get("overwrite_existing_batch_val", False))
    global_rife_multiplier_value = str(values[16] or "x2").strip().lower()
    if not global_rife_multiplier_value.startswith("x"):
        global_rife_multiplier_value = f"x{global_rife_multiplier_value}"
    if global_rife_multiplier_value not in {"x2", "x4", "x8"}:
        global_rife_multiplier_value = "x2"
    try:
        global_rife_process_chunks_value = bool(values[OUTPUT_ORDER.index("global_rife_process_chunks")])
    except Exception:
        global_rife_process_chunks_value = bool(seed_controls.get("global_rife_process_chunks_val", True))

    # Global RIFE controls (shared across all upscaler tabs).
    with gr.Group():
        gr.Markdown("#### 🌐 Global RIFE")
        frame_interpolation = gr.Checkbox(
            label="Global Enable RIFE",
            value=values[15],
            info="When enabled, each upscaler keeps its original output and also generates an additional RIFE FPS output (e.g., _2xFPS, _4xFPS)."
        )
        with gr.Row():
            global_rife_multiplier = gr.Dropdown(
                label="RIFE FPS Multiplier",
                choices=["x2", "x4", "x8"],
                value=global_rife_multiplier_value,
                info="FPS increase applied to finalized video outputs."
            )
            global_rife_model_value = str(values[17] or "").strip()
            if rife_models and global_rife_model_value not in rife_models:
                preferred_default = get_rife_default_model()
                if preferred_default in rife_models:
                    global_rife_model_value = preferred_default
                else:
                    global_rife_model_value = rife_models[0]
            global_rife_model = gr.Dropdown(
                label="RIFE Model",
                choices=rife_models,
                value=global_rife_model_value,
                allow_custom_value=True,
                info="Model used for global post-upscale frame interpolation."
            )
        with gr.Row():
            global_rife_precision = gr.Dropdown(
                label="RIFE Precision",
                choices=["fp32", "fp16"],
                value=values[18] if str(values[18]).lower() in {"fp32", "fp16"} else "fp32",
                info="Default is fp32 for maximum compatibility."
            )
            global_rife_cuda_device = gr.Textbox(
                label="RIFE CUDA Device Override",
                value=values[19],
                placeholder="Leave empty for default GPU (single GPU recommended)",
                info="Optional single CUDA device ID for global RIFE post-process."
            )
        global_rife_process_chunks = gr.Checkbox(
            label="Chunk-Safe Global RIFE (process chunks before merge)",
            value=global_rife_process_chunks_value,
            info="Recommended. When chunking is active, applies Global RIFE to each chunk before final merge to avoid morphing around chunk/scene boundaries."
        )

    with gr.Tabs():
        # Output Format Settings
        with gr.TabItem("📁 Output Format"):
            gr.Markdown("#### File Output Configuration")

            with gr.Group():
                output_format = gr.Dropdown(
                    label="Output Format",
                    choices=["auto", "mp4", "png"],
                    value=values[0],
                    info="'auto' uses mp4 for videos, png for images. Explicit format overrides auto-detection."
                )

                overwrite_existing_batch = gr.Checkbox(
                    label="Overwrite existing outputs (batch mode)",
                    value=overwrite_existing_batch_default,
                    info="When OFF (default), existing batch outputs are skipped. When ON, existing outputs are overwritten.",
                )

                png_sequence_enabled = gr.Checkbox(
                    label="Enable PNG Sequence Output",
                    value=values[1],
                    info="Save as numbered PNG frames instead of video (useful for further processing)"
                )

                png_padding = gr.Slider(
                    label="PNG Frame Number Padding",
                    minimum=1, maximum=10, step=1,
                    value=values[2],
                    info="Number of digits for frame numbers (e.g., 5 = 00001.png, 6 = 000001.png).\n\n"
                         "⚠️ **CRITICAL MODEL-SPECIFIC LIMITATION:**\n"
                         "• **SeedVR2**: ❌ HARDCODED to 6 digits in CLI (line 728 of inference_cli.py)\n"
                         "  → This slider has NO EFFECT on SeedVR2 PNG outputs\n"
                         "  → SeedVR2 will ALWAYS use 6-digit padding regardless of this setting\n"
                         "• **GAN/RIFE/FlashVSR**: ✅ Fully respects this setting\n\n"
                         "💡 **Recommendation**: Keep at 6 for consistency. If using SeedVR2 + other models,\n"
                         "setting to 6 ensures all outputs match. Custom padding only works for GAN/RIFE/FlashVSR.",
                    interactive=True
                )

                png_keep_basename = gr.Checkbox(
                    label="Keep Original Basename in PNG Names",
                    value=values[3],
                    info="Preserve input filename as base for PNG frames (e.g., 'video.mp4' → 'video_00001.png').\n\n"
                         "⚠️ **MODEL-SPECIFIC BEHAVIOR:**\n"
                         "• **SeedVR2**: ✅ Always preserves input basename (CLI design)\n"
                         "  → This checkbox has NO EFFECT on SeedVR2 outputs\n"
                         "  → SeedVR2 will ALWAYS keep basename regardless of this setting\n"
                         "• **GAN/RIFE/FlashVSR**: ✅ Fully respects this setting\n\n"
                         "💡 **Note**: All models use collision-safe naming (_0001, _0002, etc.) to prevent overwrites."
                )

        # Video Settings
        with gr.TabItem("🎬 Video Output"):
            gr.Markdown("#### Video Encoding & FPS")

            with gr.Group():
                fps_override = gr.Number(
                    label="FPS Override",
                    value=values[4],
                    precision=2,
                    info="Override output FPS (0 = use source FPS)"
                )
                
                gr.Markdown("---\n**Codec Selection**")

                video_codec = gr.Dropdown(
                    label="Video Codec",
                    choices=get_codec_choices(),
                    value=values[5] if values[5] in get_codec_choices() else "h264",
                    info="Choose encoding codec based on your use case"
                )
                
                codec_info_display = gr.Markdown("")
                
                pixel_format = gr.Dropdown(
                    label="Pixel Format",
                    choices=["yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv444p10le", "rgb24"],
                    value=values[11] if len(values) > 11 else "yuv420p",
                    info="Color subsampling and bit depth. yuv420p = best compatibility"
                )
                
                pixel_format_info = gr.Markdown("")

                video_quality = gr.Slider(
                    label="Video Quality (CRF - lower is better)",
                    minimum=0, maximum=51, step=1,
                    value=values[6],
                    info="0 = lossless (huge files), 18 = visually lossless, 23 = high quality, 28 = medium, 35+ = low quality"
                )

                video_preset = gr.Dropdown(
                    label="Encoding Preset",
                    choices=ENCODING_PRESETS,
                    value=values[7],
                    info="ultrafast = fastest encoding, veryslow = best compression. medium = balanced"
                )
                
                gr.Markdown("---\n**Audio Options**")
                
                audio_codec = gr.Dropdown(
                    label="Audio Codec",
                    choices=list(AUDIO_CODECS.keys()),
                    value=values[12] if len(values) > 12 else "copy",
                    info="Audio encoding: copy = no re-encode (fastest), aac = compatible, flac = lossless, none = remove audio"
                )
                
                audio_bitrate = gr.Textbox(
                    label="Audio Bitrate (optional)",
                    value=values[13] if len(values) > 13 else "",
                    placeholder="192k, 320k, etc.",
                    info="Only used when re-encoding audio (not for 'copy')"
                )

                two_pass_encoding = gr.Checkbox(
                    label="Two-Pass Encoding",
                    value=values[8],
                    info="Slower but better quality/filesize ratio. Recommended for archival."
                )
                
                # Quick preset buttons
                gr.Markdown("---\n**Quick Presets**")
                with gr.Row():
                    preset_youtube = gr.Button("🎬 YouTube", size="lg")
                    preset_archival = gr.Button("💾 Archival", size="lg")
                    preset_editing = gr.Button("✂️ Editing", size="lg")
                    preset_web = gr.Button("🌐 Web", size="lg")

        # Frame Handling
        with gr.TabItem("🎭 Frame Processing"):
            gr.Markdown("#### Frame Trimming & Timing")

            with gr.Group():
                skip_first_frames = gr.Number(
                    label="Skip First Frames",
                    value=values[9],
                    precision=0,
                    info="Number of frames to skip from start"
                )

                load_cap = gr.Number(
                    label="Frame Load Cap",
                    value=values[10],
                    precision=0,
                    info="Maximum frames to process (0 = all frames)"
                )

                temporal_padding = gr.Number(
                    label="Temporal Padding",
                    value=values[14],
                    precision=0,
                    info="Extra frames for temporal processing"
                )

                gr.Markdown("Global RIFE controls moved to top of this tab.")

        # Comparison Settings
        with gr.TabItem("🔍 Comparison Display"):
            gr.Markdown("#### Comparison Viewer Configuration")

            with gr.Group():
                comparison_mode = gr.Dropdown(
                    label="Comparison Mode",
                    choices=["native", "slider", "side_by_side", "overlay"],
                    value=values[20],
                    info="How to display before/after comparison"
                )

                pin_reference = gr.Checkbox(
                    label="Pin Reference Image",
                    value=values[21],
                    info="Keep original as fixed reference when changing settings"
                )

                fullscreen_enabled = gr.Checkbox(
                    label="Enable Fullscreen Comparison",
                    value=values[22],
                    info="Allow fullscreen viewing of comparisons"
                )

                comparison_zoom = gr.Slider(
                    label="Default Zoom Level",
                    minimum=25, maximum=400, step=25,
                    value=values[23],
                    info="Default zoom percentage for comparison viewer"
                )

                show_difference = gr.Checkbox(
                    label="Show Difference Overlay",
                    value=values[24],
                    info="Highlight differences between original and upscaled"
                )

            gr.Markdown("---")
            gr.Markdown("#### Comparison Video Generation")
            gr.Markdown("*Generate a side-by-side or stacked video comparing original input vs upscaled output*")

            with gr.Group():
                generate_comparison_video = gr.Checkbox(
                    label="Generate Input vs Output Comparison Video",
                    value=values[25] if len(values) > 25 else True,
                    info="Create a merged video showing original input (scaled up) beside the upscaled output. "
                         "The original input is scaled to match the output resolution before merging."
                )

                comparison_video_layout = gr.Dropdown(
                    label="Comparison Video Layout",
                    choices=["auto", "horizontal", "vertical"],
                    value=values[26] if len(values) > 26 else "auto",
                    info="Layout for comparison video:\n"
                         "• auto: Choose based on aspect ratio (landscape → horizontal, portrait → vertical)\n"
                         "• horizontal: Side-by-side (left: original, right: upscaled)\n"
                         "• vertical: Stacked (top: original, bottom: upscaled)"
                )

        # Direct manual video comparison
        with gr.TabItem("🎞️ Direct Video Compare"):
            gr.Markdown("#### Upload Any 2 Videos for Fullscreen Comparison")
            gr.Markdown("*Use this independent viewer to compare any two videos, even outside processed outputs.*")

            with gr.Row():
                direct_video_a_upload = gr.File(
                    label="Video A (Reference)",
                    type="filepath",
                    file_types=["video"]
                )
                direct_video_b_upload = gr.File(
                    label="Video B (Compared)",
                    type="filepath",
                    file_types=["video"]
                )

            with gr.Row():
                direct_video_a_path = gr.Textbox(
                    label="Video A Path",
                    placeholder="C:/path/to/video_a.mp4",
                )
                direct_video_b_path = gr.Textbox(
                    label="Video B Path",
                    placeholder="C:/path/to/video_b.mp4",
                )

            with gr.Row():
                direct_compare_height = gr.Slider(
                    label="Viewer Height",
                    minimum=300,
                    maximum=1000,
                    step=20,
                    value=620,
                )
                direct_compare_slider = gr.Slider(
                    label="Initial Slider Position",
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=50,
                )

            with gr.Row():
                direct_compare_btn = gr.Button("🔍 Generate Comparison", variant="primary", size="lg")
                direct_compare_clear_btn = gr.Button("🧹 Clear", size="lg")

            direct_compare_status = gr.Markdown("")
            direct_comparison_html = gr.HTML(
                label="Direct Video Comparison",
                value="",
                js_on_load=get_video_comparison_js_on_load(),
                visible=False,
            )

        # Metadata & Logging
        with gr.TabItem("📊 Metadata & Logging"):
            gr.Markdown("#### Output Metadata & Telemetry")

            with gr.Group():
                save_metadata = gr.Checkbox(
                    label="Save Processing Metadata",
                    value=values[27] if len(values) > 27 else True,
                    info="Embed processing info in output files"
                )

                metadata_format = gr.Dropdown(
                    label="Metadata Format",
                    choices=["json", "xml", "exif", "none"],
                    value=values[28] if len(values) > 28 else "json",
                    info="Format for embedded metadata"
                )

                telemetry_enabled = gr.Checkbox(
                    label="Enable Run Telemetry",
                    value=values[29] if len(values) > 29 else True,
                    info="Log processing stats for troubleshooting"
                )

                log_level = gr.Dropdown(
                    label="Log Verbosity",
                    choices=["error", "warning", "info", "debug"],
                    value=values[30] if len(values) > 30 else "info",
                    info="Detail level for processing logs"
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
        tab_name="output",
        inputs_list=[],
        base_dir=base_dir,
        models_list=models_list,
        open_accordion=True,
    )

    # Collect inputs for callbacks (must match OUTPUT_ORDER exactly)
    inputs_list = [
        output_format, png_sequence_enabled, png_padding, png_keep_basename,
        fps_override, video_codec, video_quality, video_preset, two_pass_encoding,
        skip_first_frames, load_cap, pixel_format, audio_codec, audio_bitrate,
        temporal_padding, frame_interpolation, global_rife_multiplier, global_rife_model,
        global_rife_precision, global_rife_cuda_device, comparison_mode, pin_reference,
        fullscreen_enabled, comparison_zoom, show_difference,
        generate_comparison_video, comparison_video_layout,  # NEW: Comparison video options
        save_metadata, metadata_format, telemetry_enabled, log_level, overwrite_existing_batch,
        global_rife_process_chunks
    ]

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
        tab_name="output",
    )

    # Codec info updates
    video_codec.change(
        fn=service["update_codec_info"],
        inputs=video_codec,
        outputs=codec_info_display
    )
    
    pixel_format.change(
        fn=service["update_pixel_format_info"],
        inputs=pixel_format,
        outputs=pixel_format_info
    )
    
    # Quick codec preset buttons
    preset_youtube.click(
        fn=lambda *vals: service["apply_codec_preset"]("youtube", list(vals)),
        inputs=inputs_list,
        outputs=inputs_list
    )
    
    preset_archival.click(
        fn=lambda *vals: service["apply_codec_preset"]("archival", list(vals)),
        inputs=inputs_list,
        outputs=inputs_list
    )
    
    preset_editing.click(
        fn=lambda *vals: service["apply_codec_preset"]("editing", list(vals)),
        inputs=inputs_list,
        outputs=inputs_list
    )
    
    preset_web.click(
        fn=lambda *vals: service["apply_codec_preset"]("web", list(vals)),
        inputs=inputs_list,
        outputs=inputs_list
    )

    # Direct comparison events
    def _upload_to_path(file_val):
        if not file_val:
            return ""
        if isinstance(file_val, dict):
            return str(file_val.get("path") or "")
        return str(file_val)

    direct_video_a_upload.change(
        fn=_upload_to_path,
        inputs=[direct_video_a_upload],
        outputs=[direct_video_a_path],
    )
    direct_video_b_upload.change(
        fn=_upload_to_path,
        inputs=[direct_video_b_upload],
        outputs=[direct_video_b_path],
    )

    def _build_direct_video_compare(path_a, path_b, height, slider_pos):
        pa = str(path_a or "").strip()
        pb = str(path_b or "").strip()
        if not pa or not pb:
            return gr.update(value="⚠️ Select both videos first."), gr.update(value="", visible=False)
        if not Path(pa).exists() or not Path(pb).exists():
            return gr.update(value="⚠️ One or both video paths do not exist."), gr.update(value="", visible=False)

        html = create_video_comparison_html(
            original_video=pa,
            upscaled_video=pb,
            height=int(height or 620),
            slider_position=float(slider_pos or 50.0),
        )
        return (
            gr.update(value="✅ Direct video comparison ready. Use fullscreen inside the viewer controls."),
            gr.update(value=html, visible=True),
        )

    def _clear_direct_video_compare():
        return (
            "",
            "",
            gr.update(value=""),
            gr.update(value="", visible=False),
        )

    direct_compare_btn.click(
        fn=_build_direct_video_compare,
        inputs=[direct_video_a_path, direct_video_b_path, direct_compare_height, direct_compare_slider],
        outputs=[direct_compare_status, direct_comparison_html],
    )
    direct_compare_clear_btn.click(
        fn=_clear_direct_video_compare,
        outputs=[direct_video_a_path, direct_video_b_path, direct_compare_status, direct_comparison_html],
    )

    return {
        "inputs_list": inputs_list,
        "preset_dropdown": preset_dropdown,
        "preset_status": preset_status,
    }

