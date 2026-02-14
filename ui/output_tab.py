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
    for key, value in (output_settings or {}).items():
        if value is not None:
            merged_defaults[key] = value

    def _value(key: str, fallback=None):
        raw = merged_defaults.get(key, fallback)
        if raw is None and fallback is not None:
            return fallback
        return raw

    gr.Markdown("### Output & Comparison Settings")
    gr.Markdown("*Configure output formats, FPS handling, metadata, and comparison behavior shared across all models*")

    overwrite_existing_batch_default = bool(_value("overwrite_existing_batch", False))

    global_rife_multiplier_value = str(_value("global_rife_multiplier", "x2") or "x2").strip().lower()
    if not global_rife_multiplier_value.startswith("x"):
        global_rife_multiplier_value = f"x{global_rife_multiplier_value}"
    if global_rife_multiplier_value not in {"x2", "x4", "x8"}:
        global_rife_multiplier_value = "x2"

    global_rife_process_chunks_value = bool(_value("global_rife_process_chunks", True))

    global_rife_model_value = str(_value("global_rife_model", "") or "").strip()
    if rife_models and global_rife_model_value not in rife_models:
        preferred_default = get_rife_default_model()
        if preferred_default in rife_models:
            global_rife_model_value = preferred_default
        else:
            global_rife_model_value = rife_models[0]

    image_output_format_value = str(_value("image_output_format", "png") or "png").strip().lower()
    if image_output_format_value not in {"png", "jpg", "webp"}:
        image_output_format_value = "png"

    try:
        image_output_quality_value = int(float(_value("image_output_quality", 95) or 95))
    except Exception:
        image_output_quality_value = 95
    image_output_quality_value = max(1, min(100, image_output_quality_value))

    seedvr2_video_backend_value = str(_value("seedvr2_video_backend", "opencv") or "opencv").strip().lower()
    if seedvr2_video_backend_value not in {"opencv", "ffmpeg"}:
        seedvr2_video_backend_value = "opencv"
    seedvr2_use_10bit_value = bool(_value("seedvr2_use_10bit", False))

    default_codec = _value("video_codec", "h264")
    codec_choices = get_codec_choices()
    if default_codec not in codec_choices:
        default_codec = "h264"

    pix_fmt_choices = get_pixel_format_choices()
    default_pix_fmt = str(_value("pixel_format", "yuv420p") or "yuv420p")
    if default_pix_fmt not in pix_fmt_choices:
        default_pix_fmt = "yuv420p"

    with gr.Tabs():
        with gr.TabItem("Global RIFE"):
            gr.Markdown("#### Global Frame Interpolation")
            gr.Markdown("Runs after upscaling to create additional FPS outputs while preserving the original upscaled result.")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    frame_interpolation = gr.Checkbox(
                        label="Enable Global RIFE",
                        value=bool(_value("frame_interpolation", False)),
                        info="Applies RIFE post-processing to finalized video outputs from all model tabs."
                    )
                    global_rife_multiplier = gr.Dropdown(
                        label="RIFE FPS Multiplier",
                        choices=["x2", "x4", "x8"],
                        value=global_rife_multiplier_value,
                        info="FPS multiplier for global post-process outputs."
                    )
                    global_rife_precision = gr.Dropdown(
                        label="RIFE Precision",
                        choices=["fp32", "fp16"],
                        value=str(_value("global_rife_precision", "fp32") or "fp32").lower() if str(_value("global_rife_precision", "fp32") or "fp32").lower() in {"fp32", "fp16"} else "fp32",
                        info="Use fp32 for compatibility, fp16 for speed when stable."
                    )

                with gr.Column(scale=1):
                    global_rife_model = gr.Dropdown(
                        label="RIFE Model",
                        choices=rife_models,
                        value=global_rife_model_value,
                        allow_custom_value=True,
                        info="Model used for global post-upscale frame interpolation."
                    )
                    global_rife_cuda_device = gr.Textbox(
                        label="RIFE CUDA Device Override",
                        value=str(_value("global_rife_cuda_device", "") or ""),
                        placeholder="Leave empty for default GPU",
                        info="Optional single CUDA device ID for global RIFE post-process."
                    )
                    global_rife_process_chunks = gr.Checkbox(
                        label="Chunk-Safe Global RIFE",
                        value=global_rife_process_chunks_value,
                        info="When chunking is active, process chunks before merge to avoid seam artifacts."
                    )

        with gr.TabItem("Output Format"):
            gr.Markdown("#### File Container and Sequence Output")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    output_format = gr.Dropdown(
                        label="Output Format",
                        choices=["auto", "mp4", "png"],
                        value=str(_value("output_format", "auto") or "auto"),
                        info="auto = mp4 for videos, png for images."
                    )
                    overwrite_existing_batch = gr.Checkbox(
                        label="Overwrite Existing Outputs (Batch)",
                        value=overwrite_existing_batch_default,
                        info="When OFF, existing batch outputs are skipped."
                    )

                with gr.Column(scale=1):
                    png_sequence_enabled = gr.Checkbox(
                        label="Enable PNG Sequence Output",
                        value=bool(_value("png_sequence_enabled", False)),
                        info="Save numbered PNG frames instead of encoded video where supported."
                    )
                    png_padding = gr.Slider(
                        label="PNG Frame Number Padding",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=int(_value("png_padding", 6) or 6),
                        info="Digit count for frame numbers (e.g. 6 -> 000001)."
                    )
                    png_keep_basename = gr.Checkbox(
                        label="Keep Original Basename in PNG Names",
                        value=bool(_value("png_keep_basename", True)),
                        info="Preserve input name prefix for exported PNG frames."
                    )

        with gr.TabItem("Image Output"):
            gr.Markdown("#### Global Image Save Format")
            gr.Markdown("Applies to final image outputs from SeedVR2, GAN, and FlashVSR image runs.")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    image_output_format = gr.Dropdown(
                        label="Image Output Format",
                        choices=["png", "jpg", "webp"],
                        value=image_output_format_value,
                        info="Final saved format for image results across model tabs."
                    )
                with gr.Column(scale=1):
                    image_output_quality = gr.Slider(
                        label="Image Quality (%)",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=image_output_quality_value,
                        info="Used for JPG/WEBP quality. PNG ignores this value."
                    )

        with gr.TabItem("Video Output"):
            gr.Markdown("#### Video Encoding and FPS")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    fps_override = gr.Number(
                        label="FPS Override (0 = source FPS)",
                        value=float(_value("fps_override", 0) or 0),
                        precision=2,
                        info="0 keeps source FPS. Set >0 to force a target FPS."
                    )

                    gr.Markdown("**SeedVR2 Video Encoding (Global)**")
                    seedvr2_video_backend = gr.Dropdown(
                        label="SeedVR2 Video Backend",
                        choices=["opencv", "ffmpeg"],
                        value=seedvr2_video_backend_value,
                        info="opencv: fast 8-bit | ffmpeg: required for 10-bit output"
                    )
                    seedvr2_use_10bit = gr.Checkbox(
                        label="SeedVR2 Enable 10-bit Encoding",
                        value=seedvr2_use_10bit_value,
                        info="Uses x265 yuv420p10le. Requires backend=ffmpeg."
                    )
                    seedvr2_encoding_warning = gr.Markdown("", visible=False)

                    video_codec = gr.Dropdown(
                        label="Video Codec",
                        choices=codec_choices,
                        value=default_codec,
                        info="Select codec based on compatibility/quality goals."
                    )
                    codec_info_display = gr.Markdown(get_codec_info(default_codec))

                    video_preset = gr.Dropdown(
                        label="Encoding Preset",
                        choices=ENCODING_PRESETS,
                        value=str(_value("video_preset", "medium") or "medium"),
                        info="ultrafast = fastest encode, veryslow = best compression."
                    )
                    two_pass_encoding = gr.Checkbox(
                        label="Two-Pass Encoding",
                        value=bool(_value("two_pass_encoding", False)),
                        info="Slower but can improve quality/filesize efficiency."
                    )

                with gr.Column(scale=1):
                    pixel_format = gr.Dropdown(
                        label="Pixel Format",
                        choices=pix_fmt_choices,
                        value=default_pix_fmt,
                        info="Color subsampling and bit depth profile."
                    )
                    pixel_format_info = gr.Markdown(get_pixel_format_info(default_pix_fmt))

                    video_quality = gr.Slider(
                        label="Video Quality (CRF, lower = better)",
                        minimum=0,
                        maximum=51,
                        step=1,
                        value=int(_value("video_quality", 18) or 18),
                        info="18 ~ visually lossless, 23 ~ high quality, 28 ~ medium."
                    )

                    audio_codec = gr.Dropdown(
                        label="Audio Codec",
                        choices=list(AUDIO_CODECS.keys()),
                        value=str(_value("audio_codec", "copy") or "copy"),
                        info="copy = no re-encode, none = remove audio."
                    )
                    audio_bitrate = gr.Textbox(
                        label="Audio Bitrate (optional)",
                        value=str(_value("audio_bitrate", "") or ""),
                        placeholder="192k, 320k, ...",
                        info="Used only when audio codec re-encodes."
                    )

                    gr.Markdown("**Quick Presets**")
                    with gr.Row():
                        preset_youtube = gr.Button("YouTube", size="lg")
                        preset_archival = gr.Button("Archival", size="lg")
                    with gr.Row():
                        preset_editing = gr.Button("Editing", size="lg")
                        preset_web = gr.Button("Web", size="lg")

        with gr.TabItem("Frame Processing"):
            gr.Markdown("#### Frame Range and Temporal Options")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    skip_first_frames = gr.Number(
                        label="Skip First Frames",
                        value=int(_value("skip_first_frames", 0) or 0),
                        precision=0,
                        info="Frames to skip from the start of input."
                    )
                    load_cap = gr.Number(
                        label="Frame Load Cap",
                        value=int(_value("load_cap", 0) or 0),
                        precision=0,
                        info="Maximum frames to process (0 = all)."
                    )

                with gr.Column(scale=1):
                    temporal_padding = gr.Number(
                        label="Temporal Padding",
                        value=int(_value("temporal_padding", 0) or 0),
                        precision=0,
                        info="Extra context frames for temporal pipelines."
                    )
                    gr.Markdown("Global RIFE controls are in the **Global RIFE** tab.")

        with gr.TabItem("Comparison Display"):
            gr.Markdown("#### Comparison Viewer and Auto-Generated Comparison Video")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    comparison_mode = gr.Dropdown(
                        label="Comparison Mode",
                        choices=["native", "slider", "side_by_side", "overlay"],
                        value=str(_value("comparison_mode", "slider") or "slider"),
                        info="Default visual mode for before/after comparisons."
                    )
                    pin_reference = gr.Checkbox(
                        label="Pin Reference Image",
                        value=bool(_value("pin_reference", False)),
                        info="Keep original as fixed reference between runs."
                    )
                    fullscreen_enabled = gr.Checkbox(
                        label="Enable Fullscreen Comparison",
                        value=bool(_value("fullscreen_enabled", True)),
                        info="Allow fullscreen controls in comparison viewer."
                    )

                with gr.Column(scale=1):
                    comparison_zoom = gr.Slider(
                        label="Default Zoom Level",
                        minimum=25,
                        maximum=400,
                        step=25,
                        value=int(_value("comparison_zoom", 100) or 100),
                        info="Default zoom percentage for image comparison UI."
                    )
                    show_difference = gr.Checkbox(
                        label="Show Difference Overlay",
                        value=bool(_value("show_difference", False)),
                        info="Overlay highlights differences between source and output."
                    )
                    generate_comparison_video = gr.Checkbox(
                        label="Generate Input vs Output Comparison Video",
                        value=bool(_value("generate_comparison_video", True)),
                        info="Generate a merged comparison video for video outputs."
                    )
                    comparison_video_layout = gr.Dropdown(
                        label="Comparison Video Layout",
                        choices=["auto", "horizontal", "vertical"],
                        value=str(_value("comparison_video_layout", "auto") or "auto"),
                        info="auto picks layout from aspect ratio."
                    )

        with gr.TabItem("Direct Video Compare"):
            gr.Markdown("#### Upload Any Two Videos for Independent Fullscreen Comparison")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
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
                        direct_compare_btn = gr.Button("Generate Comparison", variant="primary", size="lg")
                        direct_compare_clear_btn = gr.Button("Clear", size="lg")

                    direct_compare_status = gr.Markdown("")

                with gr.Column(scale=1):
                    direct_comparison_html = gr.HTML(
                        label="Direct Video Comparison",
                        value="",
                        js_on_load=get_video_comparison_js_on_load(),
                        visible=False,
                    )

        with gr.TabItem("Metadata & Logging"):
            gr.Markdown("#### Metadata and Runtime Logging")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    save_metadata = gr.Checkbox(
                        label="Save Processing Metadata",
                        value=bool(_value("save_metadata", True)),
                        info="Save run metadata into output folder."
                    )
                    telemetry_enabled = gr.Checkbox(
                        label="Enable Run Telemetry",
                        value=bool(_value("telemetry_enabled", True)),
                        info="Enable runtime stats logging for troubleshooting."
                    )

                with gr.Column(scale=1):
                    metadata_format = gr.Dropdown(
                        label="Metadata Format",
                        choices=["json", "xml", "exif", "none"],
                        value=str(_value("metadata_format", "json") or "json"),
                        info="Serialization format for run metadata payloads."
                    )
                    log_level = gr.Dropdown(
                        label="Log Verbosity",
                        choices=["error", "warning", "info", "debug"],
                        value=str(_value("log_level", "info") or "info"),
                        info="Verbosity level for per-run logs."
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
        output_format,
        png_sequence_enabled,
        png_padding,
        png_keep_basename,
        fps_override,
        image_output_format,
        image_output_quality,
        seedvr2_video_backend,
        seedvr2_use_10bit,
        video_codec,
        video_quality,
        video_preset,
        two_pass_encoding,
        skip_first_frames,
        load_cap,
        pixel_format,
        audio_codec,
        audio_bitrate,
        temporal_padding,
        frame_interpolation,
        global_rife_multiplier,
        global_rife_model,
        global_rife_precision,
        global_rife_cuda_device,
        comparison_mode,
        pin_reference,
        fullscreen_enabled,
        comparison_zoom,
        show_difference,
        generate_comparison_video,
        comparison_video_layout,
        save_metadata,
        metadata_format,
        telemetry_enabled,
        log_level,
        overwrite_existing_batch,
        global_rife_process_chunks,
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

    # SeedVR2 encoding validation
    def _validate_seedvr2_encoding(backend: str, use_10bit: bool):
        if bool(use_10bit) and str(backend or "").strip().lower() != "ffmpeg":
            return gr.update(
                value="10-bit encoding requires SeedVR2 Video Backend = ffmpeg.",
                visible=True,
            )
        if bool(use_10bit):
            return gr.update(
                value="10-bit enabled: SeedVR2 will encode x265 yuv420p10le.",
                visible=True,
            )
        return gr.update(value="", visible=False)

    seedvr2_video_backend.change(
        fn=_validate_seedvr2_encoding,
        inputs=[seedvr2_video_backend, seedvr2_use_10bit],
        outputs=seedvr2_encoding_warning,
    )
    seedvr2_use_10bit.change(
        fn=_validate_seedvr2_encoding,
        inputs=[seedvr2_video_backend, seedvr2_use_10bit],
        outputs=seedvr2_encoding_warning,
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
            return gr.update(value="Select both videos first."), gr.update(value="", visible=False)
        if not Path(pa).exists() or not Path(pb).exists():
            return gr.update(value="One or both video paths do not exist."), gr.update(value="", visible=False)

        html = create_video_comparison_html(
            original_video=pa,
            upscaled_video=pb,
            height=int(height or 620),
            slider_position=float(slider_pos or 50.0),
        )
        return (
            gr.update(value="Direct video comparison ready."),
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
