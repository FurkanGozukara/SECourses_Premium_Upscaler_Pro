"""
Output & Comparison Tab - Self-contained modular implementation
UPDATED: Now uses Universal Preset System
"""

import gradio as gr
import hashlib
import re
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

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
from shared.gpu_utils import describe_gpu_selection, resolve_global_gpu_device
from shared.video_codec_options import (
    get_codec_choices,
    get_pixel_format_choices,
    get_codec_info,
    get_pixel_format_info,
    ENCODING_PRESETS,
    AUDIO_CODECS,
    H265_TUNE_OPTIONS,
    DEFAULT_AV1_FILM_GRAIN,
    DEFAULT_AV1_FILM_GRAIN_DENOISE,
)
from shared.video_comparison_slider import (
    create_image_comparison_html,
    create_video_comparison_html,
    get_video_comparison_js_on_load,
)
from shared.frame_utils import extract_video_thumbnail
from ui.universal_preset_section import (
    universal_preset_section,
    wire_universal_preset_events,
    sync_tab_to_shared_state,
)


def output_tab(preset_manager, shared_state: gr.State, base_dir: Path, global_settings: Dict[str, Any] = None):
    """
    Self-contained Output & Comparison tab.
    Handles output format and comparison settings shared across ALL upscaler models.
    """

    seed_controls = shared_state.value.get("seed_controls", {})
    global_settings_state = seed_controls.get("global_settings", {}) if isinstance(seed_controls, dict) else {}
    if not isinstance(global_settings_state, dict):
        global_settings_state = {}
    resolved_global_gpu = resolve_global_gpu_device(
        global_settings_state.get("global_gpu_device", seed_controls.get("global_gpu_device_val"))
    )
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

    pix_fmt_choices = get_pixel_format_choices(default_codec)
    default_pix_fmt = str(_value("pixel_format", pix_fmt_choices[0] if pix_fmt_choices else "yuv420p") or "yuv420p")
    if default_pix_fmt not in pix_fmt_choices:
        default_pix_fmt = pix_fmt_choices[0] if pix_fmt_choices else "yuv420p"
    default_h265_tune = str(_value("h265_tune", "none") or "none").strip().lower()
    if default_h265_tune not in H265_TUNE_OPTIONS:
        default_h265_tune = "none"
    try:
        default_av1_film_grain = int(
            float(_value("av1_film_grain", DEFAULT_AV1_FILM_GRAIN) or DEFAULT_AV1_FILM_GRAIN)
        )
    except Exception:
        default_av1_film_grain = DEFAULT_AV1_FILM_GRAIN
    default_av1_film_grain = max(0, min(50, default_av1_film_grain))
    default_av1_film_grain_denoise = bool(
        _value("av1_film_grain_denoise", DEFAULT_AV1_FILM_GRAIN_DENOISE)
    )

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
                    gr.Markdown(
                        f"**GPU Source:** Global selector (`{describe_gpu_selection(resolved_global_gpu)}`) is used for Global RIFE and all tabs."
                    )
                    global_rife_cuda_device = gr.State(
                        "" if resolved_global_gpu == "cpu" else resolved_global_gpu
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
                    gr.Markdown("**Film Grain (FFmpeg post-encode)**")
                    h265_tune = gr.Dropdown(
                        label="H.265 Tune",
                        choices=H265_TUNE_OPTIONS,
                        value=default_h265_tune,
                        info="Used only when Video Codec = h265. Use 'grain' for grain-preserving tuning."
                    )
                    av1_film_grain = gr.Slider(
                        label="AV1 Film Grain",
                        minimum=0,
                        maximum=50,
                        step=1,
                        value=default_av1_film_grain,
                        info="Used only when Video Codec = av1 (libsvtav1). 0 disables synthetic film grain."
                    )
                    av1_film_grain_denoise = gr.Checkbox(
                        label="AV1 Film Grain Denoise",
                        value=default_av1_film_grain_denoise,
                        info="Maps to film-grain-denoise. Off = 0 (requested default)."
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
            gr.Markdown("#### Frame Range Controls")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    skip_first_frames = gr.Number(
                        label="Skip First Frames",
                        value=int(_value("skip_first_frames", 0) or 0),
                        precision=0,
                        info="Skip N frames from the beginning before processing. Useful for intros, logos, or dead lead-in frames."
                    )
                    load_cap = gr.Number(
                        label="Frame Load Cap",
                        value=int(_value("load_cap", 0) or 0),
                        precision=0,
                        info="Process only the first N frames after skipping. Use small values for quick tests (0 = process entire input)."
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

        with gr.TabItem("Videos Comparison Slider"):
            gr.Markdown("#### Multi-Video Comparison Slider")
            gr.Markdown("Add videos to the pool at any time, then pick left/right using thumbnails.")
            multi_video_pool_state = gr.State([])
            multi_video_left_state = gr.State("")
            multi_video_right_state = gr.State("")
            multi_video_thumb_cache_state = gr.State({})

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    multi_video_upload = gr.File(
                        label="Add Videos to Comparison Pool",
                        type="filepath",
                        file_types=["video"],
                        file_count="multiple",
                    )
                    gr.Markdown("Click a thumbnail in each picker to choose left and right videos.")

                    with gr.Row():
                        multi_video_left_gallery = gr.Gallery(
                            label="Left Video Picker (Reference)",
                            value=[],
                            columns=4,
                            rows=1,
                            height=180,
                            object_fit="contain",
                            allow_preview=False,
                        )
                        multi_video_right_gallery = gr.Gallery(
                            label="Right Video Picker (Compared)",
                            value=[],
                            columns=4,
                            rows=1,
                            height=180,
                            object_fit="contain",
                            allow_preview=False,
                        )

                    with gr.Row():
                        multi_video_left_preview = gr.Image(
                            label="Left Selection Thumbnail",
                            value=None,
                            visible=False,
                            interactive=False,
                            height=140,
                        )
                        multi_video_right_preview = gr.Image(
                            label="Right Selection Thumbnail",
                            value=None,
                            visible=False,
                            interactive=False,
                            height=140,
                        )

                    with gr.Row():
                        multi_video_height = gr.Slider(
                            label="Viewer Height",
                            minimum=300,
                            maximum=1000,
                            step=20,
                            value=620,
                        )
                        multi_video_slider_pos = gr.Slider(
                            label="Initial Slider Position",
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=50,
                        )

                    with gr.Row():
                        multi_video_swap_btn = gr.Button("Swap", size="lg")
                        multi_video_render_btn = gr.Button("Render", variant="primary", size="lg")
                        multi_video_clear_btn = gr.Button("Clear", size="lg")

                    multi_video_status = gr.Markdown("Upload at least 2 videos to compare.")

                with gr.Column(scale=1):
                    multi_video_html = gr.HTML(
                        label="Video Pair Comparison",
                        value="",
                        js_on_load=get_video_comparison_js_on_load(),
                        visible=False,
                    )

        with gr.TabItem("Images Comparison Slider"):
            gr.Markdown("#### Multi-Image Comparison Slider")
            gr.Markdown("Add images to the pool at any time, then pick left/right using thumbnails.")
            multi_image_pool_state = gr.State([])
            multi_image_left_state = gr.State("")
            multi_image_right_state = gr.State("")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    multi_image_upload = gr.File(
                        label="Add Images to Comparison Pool",
                        type="filepath",
                        file_types=["image"],
                        file_count="multiple",
                    )
                    gr.Markdown("Click a thumbnail in each picker to choose left and right images.")

                    with gr.Row():
                        multi_image_left_gallery = gr.Gallery(
                            label="Left Image Picker (Reference)",
                            value=[],
                            columns=4,
                            rows=1,
                            height=180,
                            object_fit="contain",
                            allow_preview=False,
                        )
                        multi_image_right_gallery = gr.Gallery(
                            label="Right Image Picker (Compared)",
                            value=[],
                            columns=4,
                            rows=1,
                            height=180,
                            object_fit="contain",
                            allow_preview=False,
                        )

                    with gr.Row():
                        multi_image_left_preview = gr.Image(
                            label="Left Selection Thumbnail",
                            value=None,
                            visible=False,
                            interactive=False,
                            height=140,
                        )
                        multi_image_right_preview = gr.Image(
                            label="Right Selection Thumbnail",
                            value=None,
                            visible=False,
                            interactive=False,
                            height=140,
                        )

                    with gr.Row():
                        multi_image_swap_btn = gr.Button("Swap", size="lg")
                        multi_image_render_btn = gr.Button("Render", variant="primary", size="lg")
                        multi_image_clear_btn = gr.Button("Clear", size="lg")

                    multi_image_status = gr.Markdown("Upload at least 2 images to compare.")

                with gr.Column(scale=1):
                    multi_image_slider = gr.HTML(
                        label="Image Pair Comparison",
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
        h265_tune,
        av1_film_grain,
        av1_film_grain_denoise,
        two_pass_encoding,
        skip_first_frames,
        load_cap,
        pixel_format,
        audio_codec,
        audio_bitrate,
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

    def _sync_codec_and_pixfmt(codec_key: str, current_pix_fmt: str):
        codec = str(codec_key or "h264").strip().lower()
        codec_choices_local = get_codec_choices()
        if codec not in codec_choices_local:
            codec = "h264"
        pix_choices_local = get_pixel_format_choices(codec)
        pix_fallback = pix_choices_local[0] if pix_choices_local else "yuv420p"
        pix_fmt = str(current_pix_fmt or pix_fallback).strip().lower()
        if pix_fmt not in pix_choices_local:
            pix_fmt = pix_fallback
        return (
            gr.update(value=get_codec_info(codec)),
            gr.update(choices=pix_choices_local, value=pix_fmt),
            gr.update(value=get_pixel_format_info(pix_fmt)),
        )

    # Codec + pixel format sync updates
    video_codec.change(
        fn=_sync_codec_and_pixfmt,
        inputs=[video_codec, pixel_format],
        outputs=[codec_info_display, pixel_format, pixel_format_info],
    )

    pixel_format.change(
        fn=service["update_pixel_format_info"],
        inputs=pixel_format,
        outputs=pixel_format_info
    )

    codec_index = OUTPUT_ORDER.index("video_codec")
    pix_fmt_index = OUTPUT_ORDER.index("pixel_format")

    def _apply_codec_preset_with_sync(preset_name: str, *vals):
        current_values = list(vals[:-1])
        state = vals[-1] if isinstance(vals[-1], dict) else {"seed_controls": {}}
        next_values = service["apply_codec_preset"](preset_name, current_values)

        codec = str(next_values[codec_index] or "h264").strip().lower()
        codec_choices_local = get_codec_choices()
        if codec not in codec_choices_local:
            codec = "h264"
            next_values[codec_index] = codec

        pix_choices_local = get_pixel_format_choices(codec)
        pix_fallback = pix_choices_local[0] if pix_choices_local else "yuv420p"
        pix_fmt = str(next_values[pix_fmt_index] or pix_fallback).strip().lower()
        if pix_fmt not in pix_choices_local:
            pix_fmt = pix_fallback
            next_values[pix_fmt_index] = pix_fmt

        synced_state = sync_tab_to_shared_state("output", list(next_values), state)
        ui_values = list(next_values)
        ui_values[pix_fmt_index] = gr.update(choices=pix_choices_local, value=pix_fmt)

        return (
            *ui_values,
            gr.update(value=get_codec_info(codec)),
            gr.update(value=get_pixel_format_info(pix_fmt)),
            synced_state,
        )

    # Quick codec preset buttons
    preset_youtube.click(
        fn=lambda *vals: _apply_codec_preset_with_sync("youtube", *vals),
        inputs=inputs_list + [shared_state],
        outputs=inputs_list + [codec_info_display, pixel_format_info, shared_state],
    )

    preset_archival.click(
        fn=lambda *vals: _apply_codec_preset_with_sync("archival", *vals),
        inputs=inputs_list + [shared_state],
        outputs=inputs_list + [codec_info_display, pixel_format_info, shared_state],
    )

    preset_editing.click(
        fn=lambda *vals: _apply_codec_preset_with_sync("editing", *vals),
        inputs=inputs_list + [shared_state],
        outputs=inputs_list + [codec_info_display, pixel_format_info, shared_state],
    )

    preset_web.click(
        fn=lambda *vals: _apply_codec_preset_with_sync("web", *vals),
        inputs=inputs_list + [shared_state],
        outputs=inputs_list + [codec_info_display, pixel_format_info, shared_state],
    )

    # Direct and multi-comparison helpers/events
    def _upload_to_path(file_val):
        if not file_val:
            return ""
        if isinstance(file_val, dict):
            return str(file_val.get("path") or "")
        return str(file_val)

    def _natural_filename_key(file_path: str) -> Tuple[Tuple[int, object], ...]:
        name = Path(str(file_path or "")).name.lower()
        parts = re.split(r"(\d+)", name)
        key_parts: List[Tuple[int, object]] = []
        for part in parts:
            if part.isdigit():
                key_parts.append((0, int(part)))
            else:
                key_parts.append((1, part))
        return tuple(key_parts)

    def _normalize_uploaded_paths(file_val) -> List[str]:
        if not file_val:
            return []

        items = list(file_val) if isinstance(file_val, (list, tuple)) else [file_val]
        seen = set()
        normalized: List[str] = []

        for item in items:
            if not item:
                continue
            if isinstance(item, dict):
                raw_path = str(item.get("path") or item.get("name") or "").strip()
            else:
                raw_path = str(item).strip()

            if not raw_path:
                continue

            candidate = Path(raw_path)
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate

            if not resolved.exists() or not resolved.is_file():
                continue

            resolved_str = str(resolved)
            if resolved_str in seen:
                continue
            seen.add(resolved_str)
            normalized.append(resolved_str)

        normalized.sort(key=_natural_filename_key)
        return normalized

    def _resolve_pair(paths: List[str], left_value: str, right_value: str) -> Tuple[str, str]:
        if not paths:
            return "", ""

        left = str(left_value or "").strip()
        right = str(right_value or "").strip()

        if left not in paths:
            left = paths[0]

        if right not in paths or right == left:
            right = next((p for p in paths if p != left), "")

        return left, right

    def _merge_media_pools(existing_pool, new_files) -> List[str]:
        existing_paths = _normalize_uploaded_paths(existing_pool)
        new_paths = _normalize_uploaded_paths(new_files)
        merged = existing_paths + new_paths
        seen = set()
        unique_paths: List[str] = []
        for p in merged:
            if p in seen:
                continue
            seen.add(p)
            unique_paths.append(p)
        unique_paths.sort(key=_natural_filename_key)
        return unique_paths

    def _preview_update(preview_path: Optional[str]):
        preview = str(preview_path or "").strip()
        if preview and Path(preview).exists():
            return gr.update(value=preview, visible=True)
        return gr.update(value=None, visible=False)

    def _build_video_thumbnail_path(video_path: str, thumb_cache) -> Tuple[str, Dict[str, str]]:
        updated_cache = dict(thumb_cache or {})
        cached = str(updated_cache.get(video_path) or "").strip()
        if cached and Path(cached).exists():
            return cached, updated_cache

        src = Path(video_path)
        try:
            src_stat = src.stat()
            signature = f"{src.resolve()}|{src_stat.st_size}|{src_stat.st_mtime_ns}"
        except Exception:
            signature = str(src)

        thumb_dir = Path(tempfile.gettempdir()) / "secourses_compare_thumbs"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        thumb_name = f"{hashlib.sha1(signature.encode('utf-8', errors='ignore')).hexdigest()[:24]}.jpg"
        thumb_path = thumb_dir / thumb_name

        if not thumb_path.exists():
            ok, generated_thumb, _ = extract_video_thumbnail(
                video_path=str(src),
                output_path=str(thumb_path),
                width=320,
            )
            if ok and generated_thumb and Path(generated_thumb).exists():
                thumb_path = Path(generated_thumb)

        if thumb_path.exists():
            thumb_str = str(thumb_path)
            updated_cache[video_path] = thumb_str
            return thumb_str, updated_cache

        return "", updated_cache

    def _build_video_gallery(pool_paths: List[str], thumb_cache):
        gallery_items: List[str] = []
        preview_map: Dict[str, str] = {}
        updated_cache = dict(thumb_cache or {})

        for p in pool_paths:
            thumb_path, updated_cache = _build_video_thumbnail_path(p, updated_cache)
            display_path = thumb_path if thumb_path else p
            gallery_items.append(display_path)
            preview_map[p] = thumb_path or ""

        return gallery_items, preview_map, updated_cache

    def _build_image_gallery(pool_paths: List[str]):
        gallery_items = [p for p in pool_paths if Path(p).exists()]
        preview_map = {p: p for p in gallery_items}
        return gallery_items, preview_map

    def _extract_select_index(evt: gr.SelectData) -> int:
        idx = getattr(evt, "index", -1)
        if isinstance(idx, (tuple, list)):
            idx = idx[0] if idx else -1
        try:
            return int(idx)
        except Exception:
            return -1

    def _render_video_pair(
        pool_paths: List[str],
        left_value: str,
        right_value: str,
        height: float,
        slider_pos: float,
        thumb_cache,
    ):
        left, right = _resolve_pair(pool_paths, left_value, right_value)
        gallery_items, preview_map, updated_cache = _build_video_gallery(pool_paths, thumb_cache)
        left_gallery_update = gr.update(value=list(gallery_items), visible=bool(gallery_items))
        right_gallery_update = gr.update(value=list(gallery_items), visible=bool(gallery_items))
        left_preview = _preview_update(preview_map.get(left, ""))
        right_preview = _preview_update(preview_map.get(right, ""))

        if len(pool_paths) < 2:
            if len(pool_paths) == 1:
                msg = "Pool has 1 video. Add at least one more video to compare."
            else:
                msg = "Upload at least 2 videos to compare."
            return (
                left,
                right,
                left_gallery_update,
                right_gallery_update,
                left_preview,
                right_preview,
                gr.update(value=msg),
                gr.update(value="", visible=False),
                updated_cache,
            )

        if not left or not right:
            return (
                left,
                right,
                left_gallery_update,
                right_gallery_update,
                left_preview,
                right_preview,
                gr.update(value="Select two different videos to compare."),
                gr.update(value="", visible=False),
                updated_cache,
            )

        safe_height = max(300, int(float(height or 620)))
        safe_slider = max(0.0, min(100.0, float(slider_pos or 50.0)))
        html = create_video_comparison_html(
            original_video=left,
            upscaled_video=right,
            height=safe_height,
            slider_position=safe_slider,
            selectable_videos=pool_paths,
        )
        status = f"Comparing `{Path(left).name}` vs `{Path(right).name}`"
        return (
            left,
            right,
            left_gallery_update,
            right_gallery_update,
            left_preview,
            right_preview,
            gr.update(value=status),
            gr.update(value=html, visible=True),
            updated_cache,
        )

    def _render_image_pair(pool_paths: List[str], left_value: str, right_value: str):
        left, right = _resolve_pair(pool_paths, left_value, right_value)
        gallery_items, preview_map = _build_image_gallery(pool_paths)
        left_gallery_update = gr.update(value=list(gallery_items), visible=bool(gallery_items))
        right_gallery_update = gr.update(value=list(gallery_items), visible=bool(gallery_items))
        left_preview = _preview_update(preview_map.get(left, ""))
        right_preview = _preview_update(preview_map.get(right, ""))

        if len(pool_paths) < 2:
            if len(pool_paths) == 1:
                msg = "Pool has 1 image. Add at least one more image to compare."
            else:
                msg = "Upload at least 2 images to compare."
            return (
                left,
                right,
                left_gallery_update,
                right_gallery_update,
                left_preview,
                right_preview,
                gr.update(value=msg),
                gr.update(value="", visible=False),
            )

        if not left or not right:
            return (
                left,
                right,
                left_gallery_update,
                right_gallery_update,
                left_preview,
                right_preview,
                gr.update(value="Select two different images to compare."),
                gr.update(value="", visible=False),
            )

        html = create_image_comparison_html(
            original_image=left,
            upscaled_image=right,
            height=620,
            slider_position=50.0,
            selectable_images=pool_paths,
        )
        status = f"Comparing `{Path(left).name}` vs `{Path(right).name}`"
        return (
            left,
            right,
            left_gallery_update,
            right_gallery_update,
            left_preview,
            right_preview,
            gr.update(value=status),
            gr.update(value=html, visible=True),
        )

    def _sync_multi_video_upload(files, pool_paths, left_value, right_value, height, slider_pos, thumb_cache):
        merged_pool = _merge_media_pools(pool_paths, files)
        (
            left_update,
            right_update,
            left_gallery_update,
            right_gallery_update,
            left_preview_update,
            right_preview_update,
            status_update,
            html_update,
            thumb_cache_update,
        ) = _render_video_pair(
            merged_pool, left_value, right_value, height, slider_pos, thumb_cache
        )
        return (
            merged_pool,
            left_update,
            right_update,
            left_gallery_update,
            right_gallery_update,
            left_preview_update,
            right_preview_update,
            status_update,
            html_update,
            thumb_cache_update,
        )

    def _refresh_multi_video(pool_paths, left_value, right_value, height, slider_pos, thumb_cache):
        normalized = _normalize_uploaded_paths(pool_paths)
        return _render_video_pair(normalized, left_value, right_value, height, slider_pos, thumb_cache)

    def _select_left_video(evt: gr.SelectData, pool_paths, left_value, right_value, height, slider_pos, thumb_cache):
        normalized = _normalize_uploaded_paths(pool_paths)
        idx = _extract_select_index(evt)
        next_left = normalized[idx] if 0 <= idx < len(normalized) else left_value
        return _render_video_pair(normalized, next_left, right_value, height, slider_pos, thumb_cache)

    def _select_right_video(evt: gr.SelectData, pool_paths, left_value, right_value, height, slider_pos, thumb_cache):
        normalized = _normalize_uploaded_paths(pool_paths)
        idx = _extract_select_index(evt)
        next_right = normalized[idx] if 0 <= idx < len(normalized) else right_value
        return _render_video_pair(normalized, left_value, next_right, height, slider_pos, thumb_cache)

    def _swap_multi_video(pool_paths, left_value, right_value, height, slider_pos, thumb_cache):
        normalized = _normalize_uploaded_paths(pool_paths)
        return _render_video_pair(normalized, right_value, left_value, height, slider_pos, thumb_cache)

    def _clear_multi_video_compare():
        left_empty_gallery = gr.update(value=[], visible=False)
        right_empty_gallery = gr.update(value=[], visible=False)
        hidden_preview = gr.update(value=None, visible=False)
        return (
            gr.update(value=None),
            [],
            "",
            "",
            left_empty_gallery,
            right_empty_gallery,
            hidden_preview,
            hidden_preview,
            gr.update(value="Upload at least 2 videos to compare."),
            gr.update(value="", visible=False),
            {},
        )

    def _sync_multi_image_upload(files, pool_paths, left_value, right_value):
        merged_pool = _merge_media_pools(pool_paths, files)
        (
            left_update,
            right_update,
            left_gallery_update,
            right_gallery_update,
            left_preview_update,
            right_preview_update,
            status_update,
            slider_update,
        ) = _render_image_pair(
            merged_pool, left_value, right_value
        )
        return (
            merged_pool,
            left_update,
            right_update,
            left_gallery_update,
            right_gallery_update,
            left_preview_update,
            right_preview_update,
            status_update,
            slider_update,
        )

    def _refresh_multi_image(pool_paths, left_value, right_value):
        normalized = _normalize_uploaded_paths(pool_paths)
        return _render_image_pair(normalized, left_value, right_value)

    def _select_left_image(evt: gr.SelectData, pool_paths, left_value, right_value):
        normalized = _normalize_uploaded_paths(pool_paths)
        idx = _extract_select_index(evt)
        next_left = normalized[idx] if 0 <= idx < len(normalized) else left_value
        return _render_image_pair(normalized, next_left, right_value)

    def _select_right_image(evt: gr.SelectData, pool_paths, left_value, right_value):
        normalized = _normalize_uploaded_paths(pool_paths)
        idx = _extract_select_index(evt)
        next_right = normalized[idx] if 0 <= idx < len(normalized) else right_value
        return _render_image_pair(normalized, left_value, next_right)

    def _swap_multi_image(pool_paths, left_value, right_value):
        normalized = _normalize_uploaded_paths(pool_paths)
        return _render_image_pair(normalized, right_value, left_value)

    def _clear_multi_image_compare():
        left_empty_gallery = gr.update(value=[], visible=False)
        right_empty_gallery = gr.update(value=[], visible=False)
        hidden_preview = gr.update(value=None, visible=False)
        return (
            gr.update(value=None),
            [],
            "",
            "",
            left_empty_gallery,
            right_empty_gallery,
            hidden_preview,
            hidden_preview,
            gr.update(value="Upload at least 2 images to compare."),
            gr.update(value="", visible=False),
        )

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

    multi_video_upload.change(
        fn=_sync_multi_video_upload,
        inputs=[
            multi_video_upload,
            multi_video_pool_state,
            multi_video_left_state,
            multi_video_right_state,
            multi_video_height,
            multi_video_slider_pos,
            multi_video_thumb_cache_state,
        ],
        outputs=[
            multi_video_pool_state,
            multi_video_left_state,
            multi_video_right_state,
            multi_video_left_gallery,
            multi_video_right_gallery,
            multi_video_left_preview,
            multi_video_right_preview,
            multi_video_status,
            multi_video_html,
            multi_video_thumb_cache_state,
        ],
    )
    multi_video_left_gallery.select(
        fn=_select_left_video,
        inputs=[
            multi_video_pool_state,
            multi_video_left_state,
            multi_video_right_state,
            multi_video_height,
            multi_video_slider_pos,
            multi_video_thumb_cache_state,
        ],
        outputs=[
            multi_video_left_state,
            multi_video_right_state,
            multi_video_left_gallery,
            multi_video_right_gallery,
            multi_video_left_preview,
            multi_video_right_preview,
            multi_video_status,
            multi_video_html,
            multi_video_thumb_cache_state,
        ],
    )
    multi_video_right_gallery.select(
        fn=_select_right_video,
        inputs=[
            multi_video_pool_state,
            multi_video_left_state,
            multi_video_right_state,
            multi_video_height,
            multi_video_slider_pos,
            multi_video_thumb_cache_state,
        ],
        outputs=[
            multi_video_left_state,
            multi_video_right_state,
            multi_video_left_gallery,
            multi_video_right_gallery,
            multi_video_left_preview,
            multi_video_right_preview,
            multi_video_status,
            multi_video_html,
            multi_video_thumb_cache_state,
        ],
    )
    multi_video_render_btn.click(
        fn=_refresh_multi_video,
        inputs=[
            multi_video_pool_state,
            multi_video_left_state,
            multi_video_right_state,
            multi_video_height,
            multi_video_slider_pos,
            multi_video_thumb_cache_state,
        ],
        outputs=[
            multi_video_left_state,
            multi_video_right_state,
            multi_video_left_gallery,
            multi_video_right_gallery,
            multi_video_left_preview,
            multi_video_right_preview,
            multi_video_status,
            multi_video_html,
            multi_video_thumb_cache_state,
        ],
    )
    multi_video_swap_btn.click(
        fn=_swap_multi_video,
        inputs=[
            multi_video_pool_state,
            multi_video_left_state,
            multi_video_right_state,
            multi_video_height,
            multi_video_slider_pos,
            multi_video_thumb_cache_state,
        ],
        outputs=[
            multi_video_left_state,
            multi_video_right_state,
            multi_video_left_gallery,
            multi_video_right_gallery,
            multi_video_left_preview,
            multi_video_right_preview,
            multi_video_status,
            multi_video_html,
            multi_video_thumb_cache_state,
        ],
    )
    multi_video_height.release(
        fn=_refresh_multi_video,
        inputs=[
            multi_video_pool_state,
            multi_video_left_state,
            multi_video_right_state,
            multi_video_height,
            multi_video_slider_pos,
            multi_video_thumb_cache_state,
        ],
        outputs=[
            multi_video_left_state,
            multi_video_right_state,
            multi_video_left_gallery,
            multi_video_right_gallery,
            multi_video_left_preview,
            multi_video_right_preview,
            multi_video_status,
            multi_video_html,
            multi_video_thumb_cache_state,
        ],
    )
    multi_video_slider_pos.release(
        fn=_refresh_multi_video,
        inputs=[
            multi_video_pool_state,
            multi_video_left_state,
            multi_video_right_state,
            multi_video_height,
            multi_video_slider_pos,
            multi_video_thumb_cache_state,
        ],
        outputs=[
            multi_video_left_state,
            multi_video_right_state,
            multi_video_left_gallery,
            multi_video_right_gallery,
            multi_video_left_preview,
            multi_video_right_preview,
            multi_video_status,
            multi_video_html,
            multi_video_thumb_cache_state,
        ],
    )
    multi_video_clear_btn.click(
        fn=_clear_multi_video_compare,
        outputs=[
            multi_video_upload,
            multi_video_pool_state,
            multi_video_left_state,
            multi_video_right_state,
            multi_video_left_gallery,
            multi_video_right_gallery,
            multi_video_left_preview,
            multi_video_right_preview,
            multi_video_status,
            multi_video_html,
            multi_video_thumb_cache_state,
        ],
    )

    multi_image_upload.change(
        fn=_sync_multi_image_upload,
        inputs=[multi_image_upload, multi_image_pool_state, multi_image_left_state, multi_image_right_state],
        outputs=[
            multi_image_pool_state,
            multi_image_left_state,
            multi_image_right_state,
            multi_image_left_gallery,
            multi_image_right_gallery,
            multi_image_left_preview,
            multi_image_right_preview,
            multi_image_status,
            multi_image_slider,
        ],
    )
    multi_image_left_gallery.select(
        fn=_select_left_image,
        inputs=[multi_image_pool_state, multi_image_left_state, multi_image_right_state],
        outputs=[
            multi_image_left_state,
            multi_image_right_state,
            multi_image_left_gallery,
            multi_image_right_gallery,
            multi_image_left_preview,
            multi_image_right_preview,
            multi_image_status,
            multi_image_slider,
        ],
    )
    multi_image_right_gallery.select(
        fn=_select_right_image,
        inputs=[multi_image_pool_state, multi_image_left_state, multi_image_right_state],
        outputs=[
            multi_image_left_state,
            multi_image_right_state,
            multi_image_left_gallery,
            multi_image_right_gallery,
            multi_image_left_preview,
            multi_image_right_preview,
            multi_image_status,
            multi_image_slider,
        ],
    )
    multi_image_render_btn.click(
        fn=_refresh_multi_image,
        inputs=[multi_image_pool_state, multi_image_left_state, multi_image_right_state],
        outputs=[
            multi_image_left_state,
            multi_image_right_state,
            multi_image_left_gallery,
            multi_image_right_gallery,
            multi_image_left_preview,
            multi_image_right_preview,
            multi_image_status,
            multi_image_slider,
        ],
    )
    multi_image_swap_btn.click(
        fn=_swap_multi_image,
        inputs=[multi_image_pool_state, multi_image_left_state, multi_image_right_state],
        outputs=[
            multi_image_left_state,
            multi_image_right_state,
            multi_image_left_gallery,
            multi_image_right_gallery,
            multi_image_left_preview,
            multi_image_right_preview,
            multi_image_status,
            multi_image_slider,
        ],
    )
    multi_image_clear_btn.click(
        fn=_clear_multi_image_compare,
        outputs=[
            multi_image_upload,
            multi_image_pool_state,
            multi_image_left_state,
            multi_image_right_state,
            multi_image_left_gallery,
            multi_image_right_gallery,
            multi_image_left_preview,
            multi_image_right_preview,
            multi_image_status,
            multi_image_slider,
        ],
    )

    return {
        "inputs_list": inputs_list,
        "preset_dropdown": preset_dropdown,
        "preset_status": preset_status,
    }
