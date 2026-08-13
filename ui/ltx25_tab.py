"""
LTX 2.5 Upscaler Tab - Self-contained modular implementation
Fixed 2x pixel spatial video upscaling (IC-LoRA on the 22B DiT).
Uses the Universal Preset System. Mirrors ui/sparkvsr_tab.py wiring with a
SeedVR2-like layout (left: input + model settings, right: output/status).
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any
import html
import threading
import time

from shared.services.ltx25_service import (
    build_ltx25_callbacks,
    LTX25_ORDER,
    LTX25_DEFAULT_POSITIVE,
    LTX25_DEFAULT_NEGATIVE,
    LTX25_SAMPLER_OPTIONS,
    LTX25_TOKEN_BUDGET_MODES,
    LTX25_TOKEN_BUDGET_MODE_AUTO,
    LTX25_ATTENTION_OPTIONS,
)
from shared.models.ltx25_meta import (
    get_ltx25_default_model,
    get_ltx25_metadata,
    get_ltx25_model_names,
    ltx25_sampling_defaults,
)
from shared.ltx25_constants import LTX25_TE_NAMES, LTX25_VAE_NAMES
from shared.path_utils import get_media_dimensions, normalize_path
from ui.universal_preset_section import (
    universal_preset_section,
    wire_universal_preset_events,
)
from ui.media_preview import preview_updates
from ui.shared_components import warn_cancel_confirmation
from shared.video_comparison_slider import get_video_comparison_js_on_load
from shared.processing_queue import get_processing_queue_manager, resolve_queue_gpu_resources
from shared.queue_state import (
    snapshot_queue_state,
    snapshot_global_settings,
    merge_payload_state,
)
from ui.model_tab_common import (
    TERMINAL_STATUS_TOKENS,
    sync_signature as _sync_signature,
    compact_single_line as _compact_single_line,
    log_tail_line as _log_tail_line,
    extract_update_value as _extract_update_value,
)


def ltx25_tab(
    preset_manager,
    runner,
    run_logger,
    global_settings: Dict[str, Any],
    shared_state: gr.State,
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path,
):
    """
    Self-contained LTX 2.5 Upscaler tab following the SECourses modular pattern.
    """

    # Build service callbacks
    service = build_ltx25_callbacks(
        preset_manager, runner, run_logger, global_settings, shared_state,
        base_dir, temp_dir, output_dir
    )
    queue_manager = get_processing_queue_manager()

    # Get defaults
    defaults = service["defaults"]

    # UNIVERSAL PRESET: Load from shared_state
    seed_controls = shared_state.value.get("seed_controls", {})
    ltx25_settings = seed_controls.get("ltx25_settings", {})

    # Merge with defaults
    merged_defaults = defaults.copy()
    for key, value in ltx25_settings.items():
        if value is not None:
            merged_defaults[key] = value

    values = [merged_defaults[k] for k in LTX25_ORDER]

    def _value(key: str, default=None):
        try:
            idx = LTX25_ORDER.index(key)
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
    device_choices: list[tuple[str, str]] = [("Auto (best available GPU)", "auto")]

    try:
        from shared.gpu_utils import get_gpu_info

        gpus = get_gpu_info()
        cuda_count = len(gpus)
        cuda_available = cuda_count > 0
        for g in gpus:
            try:
                device_choices.append((f"GPU {g.id}: {g.name} ({float(g.total_memory_gb):.1f} GB)", str(g.id)))
            except Exception:
                device_choices.append((f"GPU {getattr(g, 'id', '?')}", str(getattr(g, "id", "0"))))

        if cuda_available:
            gpu_hint = (
                f" Detected {cuda_count} CUDA GPU(s) - GPU acceleration available\n"
                " LTX 2.5 uses a single GPU only (multi-GPU not supported)"
            )
        else:
            gpu_hint = (
                " CUDA not detected (nvidia-smi unavailable or no NVIDIA GPU) - "
                "LTX 2.5 (22B DiT) realistically requires a CUDA GPU"
            )
    except Exception as e:
        gpu_hint = f" CUDA detection failed: {str(e)}"
        cuda_available = False
    device_choices.append(("CPU (not recommended for a 22B DiT)", "cpu"))
    device_valid_values = {v for _, v in device_choices}
    device_value = str(_value("device", "auto") or "auto")
    if device_value not in device_valid_values:
        device_value = "auto"

    available_ltx25_models = get_ltx25_model_names()
    if not available_ltx25_models:
        available_ltx25_models = [get_ltx25_default_model()]
    model_name_value = str(_value("model_name", get_ltx25_default_model()) or get_ltx25_default_model())
    if model_name_value not in available_ltx25_models:
        model_name_value = get_ltx25_default_model()

    def _build_model_info_md(selected_model: str) -> str:
        meta = get_ltx25_metadata(selected_model)
        if not meta:
            return "Model metadata unavailable."
        return (
            f"**{meta.name}** &nbsp;`{meta.transformer_file}`\n\n"
            f"- Download size: **{meta.size_gb:g} GB** (downloaded automatically on first use from `{meta.repo_id}`)\n"
            f"- Estimated VRAM class: **~{meta.estimated_vram_gb:g} GB**\n"
            f"- Recommended sampling: **{int(meta.default_steps)} steps, CFG {meta.default_cfg:g}, "
            f"{meta.default_sampler}** (`{meta.schedule}` schedule)\n"
            f"- {meta.notes}"
        )

    def _build_ltx25_sizing_md(path_val, pre_resize_val) -> gr.update:
        normalized = normalize_path(path_val) if path_val else ""
        if not normalized:
            return gr.update(value="", visible=False)
        dims = get_media_dimensions(normalized)
        if not dims:
            return gr.update(value="**Output size:** unable to read input dimensions yet.", visible=True)
        src_w, src_h = int(dims[0]), int(dims[1])
        try:
            pre = int(float(pre_resize_val or 0))
        except Exception:
            pre = 0
        if pre > 0 and max(src_w, src_h) > pre:
            ratio = float(pre) / float(max(src_w, src_h))
            rw = max(2, int(round(src_w * ratio / 2.0)) * 2)
            rh = max(2, int(round(src_h * ratio / 2.0)) * 2)
            text = (
                f"**Output size (fixed 2x):** `{src_w}x{src_h}` -> pre-resize `{rw}x{rh}` -> "
                f"output `{rw * 2}x{rh * 2}` (approx; the engine snaps to its own grid)."
            )
        else:
            text = f"**Output size (fixed 2x):** `{src_w}x{src_h}` -> output `{src_w * 2}x{src_h * 2}`."
        return gr.update(value=text, visible=True)

    # Show GPU warning if not available
    if not cuda_available:
        gr.Markdown(
            f'<div style="background: #fff3cd; padding: 12px; border-radius: 8px; border: 1px solid #ffc107;">'
            f'<strong> GPU Acceleration Unavailable</strong><br>'
            f'{gpu_hint}<br><br>'
            f'LTX 2.5 is a 22B DiT designed for CUDA GPUs. CPU mode is technically possible but impractical.'
            f'</div>',
            elem_classes="warning-text",
        )

    with gr.Row():
        # Left Column: Input & Settings
        with gr.Column(scale=3):
            gr.Markdown("#### 📥 Input Source")

            with gr.Group():
                with gr.Row():
                    with gr.Column(scale=2, elem_classes=["LTX25-input-source-col"]):
                        input_file = gr.File(
                            label="Upload video (optional)",
                            type="filepath",
                            file_types=["video"],
                        )
                        input_path = gr.Textbox(
                            label="Input Path",
                            value=_value("input_path", ""),
                            placeholder="C:/path/to/video.mp4",
                            info="Video file only. LTX 2.5 is a fixed 2x VIDEO upscaler.",
                        )
                        copy_output_into_input_btn = gr.Button(
                            "Copy Output Into Input",
                            elem_classes=["action-btn", "action-btn-source-seed", "LTX25-copy-output-compact", "sec-btn-teal"],
                            size="md",
                        )
                        auto_transfer_output_to_input = gr.Checkbox(
                            label="Auto Transfer Output to Input",
                            value=bool(_value("auto_transfer_output_to_input", False)),
                            info="After upscale completes, automatically copy the latest output path into Input Path.",
                        )

                    with gr.Column(scale=2):
                        input_image_preview = gr.Image(
                            label="🖼️ Input Preview (Image)",
                            type="filepath",
                            interactive=False,
                            height=250,
                            visible=False,
                        )
                        input_video_preview = gr.Video(
                            label="🎞️ Input Preview (Video)",
                            interactive=False,
                            height=250,
                            visible=False,
                        )

                input_cache_msg = gr.Markdown("", visible=False)
                input_detection_result = gr.Markdown("", visible=False)
                sizing_info = gr.Markdown("", visible=False, elem_classes=["resolution-info"])
                gr.Markdown(
                    "**Fixed scale:** Output = exactly **2x** input (or 2x the pre-resize). "
                    "This tab does not use the Resolution tab's upscale factor."
                )

                output_override = gr.Textbox(
                    label="Output Override (folder or .mp4 file)",
                    value=_value("output_override", ""),
                    placeholder="Leave empty for auto naming",
                    info=(
                        "Optional custom output location. A folder saves into that folder. "
                        "A .mp4 file path renames the final output to that exact file."
                    ),
                )
                resume_run_dir = gr.Textbox(
                    label="Resume Run Folder (chunk/scene resume)",
                    value=str(_value("resume_run_dir", "") or ""),
                    placeholder="Optional: G:/.../outputs/0019",
                    info=(
                        "Optional. Works only when app-level chunking is explicitly enabled in the Resolution tab "
                        "(Auto Chunk OFF + fixed chunk size). The engine's internal token-budget chunking cannot resume."
                    ),
                )

                with gr.Accordion("📁 Batch Processing", open=False):
                    batch_enable = gr.Checkbox(
                        label="Enable Batch",
                        value=bool(_value("batch_enable", False)),
                        info="Process every video in a folder (LTX 2.5 skips images).",
                    )
                    batch_input = gr.Textbox(
                        label="Batch Input Folder",
                        value=_value("batch_input_path", ""),
                        placeholder="Folder with videos",
                    )
                    batch_output = gr.Textbox(
                        label="Batch Output Folder",
                        value=_value("batch_output_path", ""),
                        placeholder="Output directory (empty = <input>/upscaled_files)",
                    )

            with gr.Accordion("🎛️ LTX 2.5 Model", open=True):
                model_name = gr.Dropdown(
                    label="LTX 2.5 Transformer",
                    choices=available_ltx25_models,
                    value=model_name_value,
                    info="All variants upscale exactly 2x. INT8 ConvRot = recommended balance. NVFP4 = fastest on RTX 50.",
                )
                with gr.Row():
                    text_encoder = gr.Dropdown(
                        label="Text Encoder",
                        choices=list(LTX25_TE_NAMES),
                        value=(
                            str(_value("text_encoder", LTX25_TE_NAMES[0]))
                            if str(_value("text_encoder", LTX25_TE_NAMES[0])) in set(LTX25_TE_NAMES)
                            else LTX25_TE_NAMES[0]
                        ),
                        info="Gemma 4 12B prompt encoder. INT8 ConvRot saves ~12GB of downloads/VRAM vs BF16.",
                    )
                    video_vae = gr.Dropdown(
                        label="Video VAE",
                        choices=list(LTX25_VAE_NAMES),
                        value=(
                            str(_value("video_vae", LTX25_VAE_NAMES[0]))
                            if str(_value("video_vae", LTX25_VAE_NAMES[0])) in set(LTX25_VAE_NAMES)
                            else LTX25_VAE_NAMES[0]
                        ),
                        info="Conv VAE matches the ComfyUI upscaler workflow default.",
                    )
                model_info_md = gr.Markdown(_build_model_info_md(model_name_value))

            with gr.Accordion("✍️ Prompts", open=True):
                positive_prompt = gr.Textbox(
                    label="Positive Prompt",
                    value=str(_value("positive_prompt", LTX25_DEFAULT_POSITIVE) or ""),
                    lines=5,
                    max_lines=10,
                    info="Faithful re-render conditioning. The default is tuned for text/logo preservation.",
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value=str(_value("negative_prompt", LTX25_DEFAULT_NEGATIVE) or ""),
                    lines=3,
                    max_lines=8,
                    info="Artifacts and content drift to suppress.",
                )
                reset_prompts_btn = gr.Button(
                    "Reset Prompts",
                    size="md",
                    elem_classes=["action-btn", "sec-btn-slate"],
                )

            with gr.Accordion("🧠 VRAM & Chunking", open=True):
                token_budget_mode = gr.Radio(
                    label="Token Budget Mode",
                    choices=list(LTX25_TOKEN_BUDGET_MODES),
                    value=(
                        str(_value("token_budget_mode", LTX25_TOKEN_BUDGET_MODE_AUTO))
                        if str(_value("token_budget_mode", LTX25_TOKEN_BUDGET_MODE_AUTO)) in set(LTX25_TOKEN_BUDGET_MODES)
                        else LTX25_TOKEN_BUDGET_MODE_AUTO
                    ),
                    info=(
                        "Auto sizes the token budget from free VRAM and prefers processing the whole video as "
                        "ONE chunk - this model works best that way."
                    ),
                )
                with gr.Row():
                    max_latent_tokens = gr.Slider(
                        label="Max Latent Tokens (Manual mode only)",
                        minimum=4096,
                        maximum=9999999,
                        step=8,
                        value=int(_value("max_latent_tokens", 18000) or 18000),
                        info=(
                            "Hard token budget per chunk (Manual mode). Type a large value "
                            "(up to 9,999,999 - same as ComfyUI) to force SINGLE-chunk processing; "
                            "Preview Chunk Plan shows the tokens a single chunk needs."
                        ),
                    )
                    reserve_vram_gb = gr.Slider(
                        label="Reserve VRAM (GB)",
                        minimum=0.0,
                        maximum=8.0,
                        step=0.5,
                        value=float(_value("reserve_vram_gb", 1.0) or 1.0),
                        info="Headroom kept free when Auto mode sizes the token budget.",
                    )
                with gr.Row():
                    max_chunk_frames = gr.Slider(
                        label="Max Chunk Frames (8n+1)",
                        minimum=9,
                        maximum=1025,
                        step=8,
                        value=int(_value("max_chunk_frames", 121) or 121),
                        info="Upper bound on frames per chunk. Snapped down to the 8n+1 grid.",
                    )
                    overlap_frames = gr.Slider(
                        label="Chunk Overlap Frames (8n+1)",
                        minimum=1,
                        maximum=257,
                        step=8,
                        value=int(_value("overlap_frames", 1) or 1),
                        info="Cross-faded overlap between chunks. 1 is enough for most content.",
                    )
                attention_backend = gr.Dropdown(
                    label="Attention Backend",
                    choices=list(LTX25_ATTENTION_OPTIONS),
                    value=(
                        str(_value("attention_backend", "auto"))
                        if str(_value("attention_backend", "auto")) in set(LTX25_ATTENTION_OPTIONS)
                        else "auto"
                    ),
                    info="auto picks the fastest installed backend (Sage > Flash > SDPA).",
                )
                plan_chunk_btn = gr.Button(
                    "📊 Preview Chunk Plan",
                    size="md",
                    elem_classes=["action-btn", "action-btn-optimize", "sec-btn-crimson"],
                )
                plan_output_box = gr.Textbox(
                    label="📊 Chunk Plan Result",
                    value="",
                    lines=7,
                    max_lines=14,
                    interactive=False,
                    visible=False,
                    buttons=["copy"],
                )
                gr.Markdown(
                    "**Preview Chunk Plan:** reads the video, sizes the token budget from free VRAM, and prints "
                    "the chunk plan (`[Plan]` lines) right here WITHOUT sampling. After the first click the plan "
                    "goes **live** and re-renders automatically as you change parameters - like the ComfyUI live "
                    "plan readout."
                )

            with gr.Accordion("🧩 VAE Tiling (Advanced)", open=False):
                with gr.Row():
                    encode_tile_size = gr.Slider(
                        label="Encode Tile Size",
                        minimum=256, maximum=1024, step=32,
                        value=int(_value("encode_tile_size", 512) or 512),
                        info="Spatial tile for VAE-encoding the guide video.",
                    )
                    encode_tile_overlap = gr.Slider(
                        label="Encode Tile Overlap",
                        minimum=16, maximum=256, step=16,
                        value=int(_value("encode_tile_overlap", 64) or 64),
                    )
                with gr.Row():
                    decode_tile_size = gr.Slider(
                        label="Decode Tile Size",
                        minimum=256, maximum=1024, step=32,
                        value=int(_value("decode_tile_size", 512) or 512),
                        info="Spatial tile for decoding the upscaled latents.",
                    )
                    decode_tile_overlap = gr.Slider(
                        label="Decode Tile Overlap",
                        minimum=16, maximum=256, step=16,
                        value=int(_value("decode_tile_overlap", 64) or 64),
                    )
                with gr.Row():
                    decode_temporal_size = gr.Slider(
                        label="Decode Temporal Size",
                        minimum=16, maximum=256, step=8,
                        value=int(_value("decode_temporal_size", 128) or 128),
                        info="Frames decoded per temporal VAE window.",
                    )
                    decode_temporal_overlap = gr.Slider(
                        label="Decode Temporal Overlap",
                        minimum=8, maximum=128, step=8,
                        value=int(_value("decode_temporal_overlap", 32) or 32),
                    )

            with gr.Accordion("🎬 Input Handling", open=False):
                with gr.Row():
                    pre_resize_longer_edge = gr.Number(
                        label="Pre-Resize Longer Edge (px)",
                        value=int(_value("pre_resize_longer_edge", 0) or 0),
                        precision=0,
                        info=(
                            "0 = native 2x. Set e.g. 1536 to downscale the longer edge first "
                            "(output = 2x that). Big VRAM/time saver for large sources."
                        ),
                    )
                    fps = gr.Number(
                        label="FPS Override",
                        value=float(_value("fps", 0.0) or 0.0),
                        info="0 = keep source FPS. Otherwise the engine resamples timing to this FPS.",
                    )
                with gr.Row():
                    start_frame = gr.Number(
                        label="Start Frame",
                        value=int(_value("start_frame", 0) or 0),
                        precision=0,
                        info="0-indexed start frame for partial processing.",
                    )
                    end_frame = gr.Number(
                        label="End Frame",
                        value=int(_value("end_frame", -1) or -1),
                        precision=0,
                        info="-1 = process until the end of input.",
                    )
                device = gr.Dropdown(
                    label="GPU Device",
                    choices=device_choices,
                    value=device_value,
                    info="The global GPU selector in the app header takes precedence at run time.",
                )
                with gr.Row():
                    output_format = gr.Dropdown(
                        label="Output Format",
                        choices=["mp4"],
                        value=str(_value("output_format", "mp4") or "mp4"),
                        info="LTX 2.5 engine outputs MP4 (audio copied from source).",
                        interactive=False,
                    )
                    save_metadata = gr.Checkbox(
                        label="Save Processing Metadata",
                        value=bool(_value("save_metadata", True)),
                        info="Save run metadata to output folder.",
                    )
                    keep_only_output_files = gr.Checkbox(
                        label="Keep only output files",
                        value=bool(_value("keep_only_output_files", False)),
                        info="After batch completion, remove metadata/chunks/temp artifacts and keep only final outputs.",
                    )
                gr.Markdown(
                    "**Video Codec / CRF / FPS Override defaults:** Controlled from `Output > Video Output` "
                    "and automatically applied here."
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
                tab_name="ltx25",
                inputs_list=[],
                base_dir=base_dir,
                models_list=get_ltx25_model_names(),
                open_accordion=True,
            )

        # Right Column: Output & Controls
        with gr.Column(scale=2):
            gr.Markdown("#### 📤 Output & Actions")
            status_box = gr.Markdown(value="Ready.", visible=False, elem_classes=["runtime-status-box"])
            progress_indicator = gr.Markdown(value="", visible=False, elem_classes=["runtime-progress-box"])

            with gr.Group():
                upscale_btn = gr.Button(
                    "🚀 Upscale 2x",
                    variant="primary",
                    size="lg",
                    elem_classes=["action-btn", "action-btn-upscale", "sec-btn-emerald"],
                )
                with gr.Row():
                    cancel_confirm = gr.Checkbox(
                        label="⚠️ Confirm cancel (required for safety)",
                        value=False,
                        info="Enable this checkbox to confirm cancellation of processing",
                        scale=3,
                    )
                    cancel_btn = gr.Button(
                        "🛑 Cancel",
                        variant="stop",
                        size="md",
                        min_width=170,
                        scale=1,
                        elem_classes=["action-btn", "action-btn-cancel", "sec-btn-red"],
                    )
                with gr.Row():
                    open_outputs_btn = gr.Button(
                        "📂 Open Outputs",
                        elem_classes=["action-btn", "action-btn-open", "sec-btn-blue"],
                    )
                    clear_temp_btn = gr.Button(
                        "🧹 Clear Temp",
                        elem_classes=["action-btn", "action-btn-clear", "sec-btn-orange"],
                    )
                    plan_chunk_btn_row = gr.Button(
                        "📊 Preview Chunk Plan",
                        elem_classes=["action-btn", "action-btn-optimize", "sec-btn-crimson"],
                    )

            with gr.Accordion("📺 Upscaled Output", open=True):
                output_video = gr.Video(
                    label="🎥 Upscaled Video (2x)",
                    interactive=False,
                    visible=False,
                    height=420,
                    buttons=["download"],
                )
                output_image = gr.Image(
                    label="🖼️ Upscaled Image",
                    interactive=False,
                    visible=False,
                    buttons=["download"],
                )
            last_processed = gr.Markdown("Processing results will appear here.")

            # Comparison
            image_slider = gr.ImageSlider(
                label="🆚 Comparison",
                interactive=False,
                slider_position=50,
                max_height=1000,
                buttons=["download", "fullscreen"],
                elem_classes=["native-image-comparison-slider"],
            )

            video_comparison_html = gr.HTML(
                label="🆚 Video Comparison",
                value="",
                js_on_load=get_video_comparison_js_on_load(),
                visible=False,
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
                label="🧩 Selected Chunk",
                interactive=False,
                visible=False,
                buttons=["download"],
            )
            refresh_chunk_btn = gr.Button(
                "🔄 Refresh Chunk Preview",
                size="sm",
                elem_classes=["action-btn", "sec-btn-slate"],
            )
            batch_gallery = gr.Gallery(
                label="🗂️ Batch Results",
                visible=False,
                columns=4,
                rows=2,
                height="auto",
                object_fit="contain",
                buttons=["download"],
            )

            log_box = gr.Textbox(
                label="📋 Processing Log",
                value="",
                lines=18,
                buttons=["copy"],
            )

            with gr.Accordion("⚙️ Sampling", open=True):
                with gr.Row():
                    steps = gr.Slider(
                        label="Steps (auto-set per model, editable)",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=int(_value("steps", 8) or 8),
                        info="Distilled variants: 8. Dev variants: 20. Switching the model resets this.",
                    )
                    cfg = gr.Slider(
                        label="CFG",
                        minimum=0.0,
                        maximum=15.0,
                        step=0.1,
                        value=float(_value("cfg", 1.0) or 1.0),
                        info="Distilled: 1.0 (no CFG). Dev: 3.0.",
                    )
                    sampler = gr.Dropdown(
                        label="Sampler",
                        choices=list(LTX25_SAMPLER_OPTIONS),
                        value=(
                            str(_value("sampler", "euler_ancestral"))
                            if str(_value("sampler", "euler_ancestral")) in set(LTX25_SAMPLER_OPTIONS)
                            else "euler_ancestral"
                        ),
                        info="euler_ancestral matches the reference workflow.",
                    )
                with gr.Row():
                    seed = gr.Number(
                        label="Seed",
                        value=int(_value("seed", 42) or 42),
                        precision=0,
                        info="Seed for reproducibility.",
                    )
                    randomize_seed = gr.Checkbox(
                        label="Randomize Seed",
                        value=bool(_value("randomize_seed", False)),
                        info="Pick a fresh random seed for every run (logged in the run log).",
                    )
                with gr.Row():
                    ic_lora_strength = gr.Slider(
                        label="IC-LoRA Strength",
                        minimum=0.0,
                        maximum=2.0,
                        step=0.05,
                        value=float(_value("ic_lora_strength", 1.0) or 1.0),
                        info="Strength of the pixel spatial upscaler IC-LoRA. Official default 1.0.",
                    )
                    guide_strength = gr.Slider(
                        label="Guide Strength",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=float(_value("guide_strength", 1.0) or 1.0),
                        info="How strongly the low-res guide latents constrain sampling. 1.0 = faithful.",
                    )

            gr.Markdown("""
            #### LTX 2.5 Upscaler Guide

            **Model defaults**
            - `Distilled INT8 ConvRot` is the recommended default everywhere: ComfyUI-parity INT8 kernels, best speed/quality balance.
            - `Distilled NVFP4` is fastest on RTX 50 (native FP4); older GPUs emulate it (same quality, slower).
            - `Distilled INT4 W4A8 ConvRot` is the smallest download; `BF16` variants need weight streaming below 48GB VRAM.
            - `Dev` variants run 20 steps at CFG 3 for maximum quality; `Distilled` variants run 8 steps at CFG 1.
            - Switching the model auto-fills Steps/CFG/Sampler (still fully editable afterwards).

            **Runtime notes**
            - LTX 2.5 upscales exactly **2x**. Use `Pre-Resize Longer Edge` to control the final size (output = 2x the pre-resize).
            - `Auto (Single Chunk Preferred)` token budgeting processes the whole video as ONE chunk whenever VRAM allows - best quality.
            - Use `📊 Preview Chunk Plan` before running to see chunk count/token budget without sampling.
            - App-level scene chunking stays OFF by default for this engine; enable it only via the Resolution tab (Auto Chunk OFF + fixed chunk size).
            - Audio is copied from the source video automatically; the Output tab's audio codec settings apply afterwards.
            """)

    # Collect inputs - MUST match LTX25_ORDER positions exactly.
    inputs_list = [
        input_path, output_override, output_format,
        model_name, text_encoder, video_vae,
        positive_prompt, negative_prompt,
        seed, randomize_seed,
        steps, cfg, sampler, ic_lora_strength, guide_strength,
        token_budget_mode, max_latent_tokens, max_chunk_frames, overlap_frames,
        reserve_vram_gb, attention_backend,
        encode_tile_size, encode_tile_overlap, decode_tile_size, decode_tile_overlap,
        decode_temporal_size, decode_temporal_overlap,
        pre_resize_longer_edge, fps, start_frame, end_frame,
        device, save_metadata, auto_transfer_output_to_input,
        batch_enable, batch_input, batch_output,
        resume_run_dir, keep_only_output_files,
    ]

    # Development validation: inputs_list must stay aligned with LTX25_ORDER
    if len(inputs_list) != len(LTX25_ORDER):
        import logging
        logging.getLogger("LTX25Tab").error(
            f"ERROR: inputs_list ({len(inputs_list)}) != LTX25_ORDER ({len(LTX25_ORDER)})"
        )

    # ------------------------------------------------------------------ #
    # Wire up events
    # ------------------------------------------------------------------ #

    def _on_model_switch(selected_model):
        """
        Auto-fill sampling defaults for the selected variant.

        Wired to .input() (NOT .change) so it only fires on real user
        interaction - preset loading never clobbers saved steps/cfg/sampler.
        """
        sampling = ltx25_sampling_defaults(selected_model)
        return (
            gr.update(value=int(sampling["steps"])),
            gr.update(value=float(sampling["cfg"])),
            gr.update(value=str(sampling["sampler"])),
            gr.update(value=_build_model_info_md(selected_model)),
        )

    model_name.input(
        fn=_on_model_switch,
        inputs=[model_name],
        outputs=[steps, cfg, sampler, model_info_md],
        queue=False,
        show_progress="hidden",
    )

    def _reset_prompts():
        return (
            gr.update(value=LTX25_DEFAULT_POSITIVE),
            gr.update(value=LTX25_DEFAULT_NEGATIVE),
        )

    reset_prompts_btn.click(
        fn=_reset_prompts,
        outputs=[positive_prompt, negative_prompt],
        queue=False,
        show_progress="hidden",
    )

    # ---------------- Input caching / detection / sizing ---------------- #

    def _analysis_progress_note(pct: int) -> str:
        if pct < 15:
            return "Reading media metadata..."
        if pct < 45:
            return "Computing fixed 2x output size..."
        if pct < 95:
            return "Preparing analysis panel..."
        return "Finalizing analysis..."

    def _analysis_banner_html(progress_pct: int, progress_note: str = "") -> str:
        safe_pct = max(0, min(100, int(progress_pct)))
        sub = f"{safe_pct}%"
        if progress_note:
            sub = f"{sub}<br>{html.escape(str(progress_note))}"
        return (
            '<div class="processing-banner">'
            '<div class="processing-spinner"></div>'
            '<div class="processing-col">'
            '<div class="processing-text">Analyzing input (fixed 2x sizing)</div>'
            f'<div class="processing-sub">{sub}</div>'
            "</div></div>"
        )

    def _build_input_detection_md(path_val: str) -> gr.update:
        from shared.input_detector import detect_input
        if not path_val or not str(path_val).strip():
            return gr.update(value="", visible=False)
        try:
            info = detect_input(path_val)
            if not info.is_valid:
                return gr.update(value=f"ERROR: **Invalid Input**\n\n{info.error_message}", visible=True)
            parts = [f"OK: **Input Detected: {info.input_type.upper()}**"]
            if info.input_type in ["video", "image"]:
                parts.append(f"&nbsp;&nbsp;Format: **{info.format.upper()}**")
            if info.input_type != "video":
                parts.append("&nbsp;&nbsp;⚠️ LTX 2.5 processes VIDEO inputs only.")
            return gr.update(value=" ".join(parts), visible=True)
        except Exception as e:
            return gr.update(value=f"ERROR: **Detection Error**\n\n{str(e)}", visible=True)

    def _run_analysis_payload(path_val, pre_resize_val, state):
        det = _build_input_detection_md(path_val or "")
        info = _build_ltx25_sizing_md(path_val or "", pre_resize_val)
        img_prev, vid_prev = preview_updates(path_val)
        return img_prev, vid_prev, det, info, state

    def _iter_analysis_with_progress(path_val, pre_resize_val, state):
        result: Dict[str, Any] = {}

        def _worker():
            try:
                result["payload"] = _run_analysis_payload(path_val, pre_resize_val, state)
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
                yield "progress", pct, _analysis_progress_note(pct)
                last_emit = now
            time.sleep(0.05)

        worker.join()
        if "error" in result:
            raise result["error"]
        yield "progress", 100, _analysis_progress_note(100)
        yield "result", result.get("payload"), ""

    def cache_input_upload(val, pre_resize_val, state):
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
            gr.update(value=_analysis_banner_html(0, _analysis_progress_note(0)), visible=True),
            state,
        )

        try:
            for event_type, payload_a, _ in _iter_analysis_with_progress(val, pre_resize_val, state):
                if event_type == "progress":
                    pct = int(payload_a)
                    yield (
                        val or "",
                        gr.update(value="", visible=False),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.update(value=_analysis_banner_html(pct, _analysis_progress_note(pct)), visible=True),
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

    def cache_input_path(path_val, pre_resize_val, state):
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
            gr.update(value=_analysis_banner_html(0, _analysis_progress_note(0)), visible=True),
            state,
        )

        try:
            for event_type, payload_a, _ in _iter_analysis_with_progress(path_val, pre_resize_val, state):
                if event_type == "progress":
                    pct = int(payload_a)
                    yield (
                        gr.update(value="", visible=False),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.update(value=_analysis_banner_html(pct, _analysis_progress_note(pct)), visible=True),
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

    input_file.upload(
        fn=cache_input_upload,
        inputs=[input_file, pre_resize_longer_edge, shared_state],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )

    input_file.change(
        fn=clear_input_path_on_upload_clear,
        inputs=[input_file, shared_state],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )

    # Keep heavy media/path analysis user-triggered only (submit, not change).
    input_path.submit(
        fn=cache_input_path,
        inputs=[input_path, pre_resize_longer_edge, shared_state],
        outputs=[input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )

    def refresh_sizing(path_val, pre_resize_val):
        return _build_ltx25_sizing_md(path_val or "", pre_resize_val)

    pre_resize_longer_edge.change(
        fn=refresh_sizing,
        inputs=[input_path, pre_resize_longer_edge],
        outputs=[sizing_info],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    # ---------------- Output -> input transfer helpers ---------------- #

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

        batch_outputs = seed_controls.get("ltx25_batch_outputs", [])
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
        pre_resize_val,
        state,
        source_label="Output transferred to input.",
    ):
        state = state or {}
        state.setdefault("seed_controls", {})
        state["seed_controls"]["last_input_path"] = output_path_val or ""

        try:
            img_prev, vid_prev, det, info, state_out = _run_analysis_payload(
                output_path_val or "",
                pre_resize_val,
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

    def copy_latest_output_to_input(pre_resize_val, state):
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
            pre_resize_val,
            state,
            source_label="Output path copied into input.",
        )

    def capture_latest_output_signature(state):
        return _output_path_signature(_resolve_latest_output_path(state))

    def auto_transfer_latest_output_to_input(
        pre_resize_val,
        auto_enabled,
        previous_signature,
        state,
    ):
        state = state if isinstance(state, dict) else {}
        if not bool(auto_enabled):
            return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), state)

        output_path_val = _resolve_latest_output_path(state)
        if not output_path_val:
            return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), state)

        latest_signature = _output_path_signature(output_path_val)
        if previous_signature and latest_signature and str(previous_signature) == str(latest_signature):
            return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), state)

        return _apply_output_path_to_input(
            output_path_val,
            pre_resize_val,
            state,
            source_label="Auto-transferred latest output into input.",
        )

    copy_output_into_input_btn.click(
        fn=copy_latest_output_to_input,
        inputs=[pre_resize_longer_edge, shared_state],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )

    # ---------------- Chunk preview ---------------- #

    def refresh_chunk_preview_ui(state):
        preview = (state or {}).get("seed_controls", {}).get("ltx25_chunk_preview", {})
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
            videos = (state or {}).get("seed_controls", {}).get("ltx25_chunk_preview", {}).get("videos", [])
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

    refresh_chunk_btn.click(
        fn=refresh_chunk_preview_ui,
        inputs=[shared_state],
        outputs=[chunk_status, chunk_gallery, chunk_preview_video],
        queue=False,
        show_progress="hidden",
    )

    # ---------------- Queue-aware run wiring ---------------- #

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

    def _batch_gallery_update_from_state(state):
        outputs = (state or {}).get("seed_controls", {}).get("ltx25_batch_outputs", [])
        if not isinstance(outputs, list):
            outputs = []
        outputs = [str(p) for p in outputs if p and Path(str(p)).exists()]
        return gr.update(value=outputs, visible=bool(outputs))

    def _last_processed_text(state, vid_upd, img_upd) -> str:
        outputs = (state or {}).get("seed_controls", {}).get("ltx25_batch_outputs", [])
        if isinstance(outputs, list) and outputs:
            last_out = str(outputs[-1])
            return f"Batch results: {len(outputs)} item(s). Last output: {Path(last_out).name}"

        single = _extract_update_value(img_upd) or _extract_update_value(vid_upd)
        if single:
            return f"Output: {single}"
        return "Processing results will appear here."

    def _expand_service_payload(payload, live_state):
        merged = merge_payload_state(payload, live_state)
        if not isinstance(merged, tuple) or len(merged) < 7:
            safe_state = live_state if isinstance(live_state, dict) else {}
            return (
                gr.update(value="ERROR: Invalid LTX 2.5 payload"),
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

        terminal_tokens = TERMINAL_STATUS_TOKENS + (
            "chunk plan ready",
            "chunk plan needs a video",
            "expects a video input",
            "preview is not available",
            "no video files found",
        )
        is_terminal = any(tok in status_lc for tok in terminal_tokens) or any(
            tok in log_lc for tok in ("critical error", "processing failed", "cancelled", "out of vram")
        )

        if (status_text or log_tail) and not is_terminal:
            title = _compact_single_line(status_text or "LTX 2.5 processing...", max_len=96)
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
            "ltx25",
            "Upscale",
            resource_keys=queue_resource_keys,
            resource_label=queue_resource_label,
        )
        acquired_slot = queue_manager.is_active(ticket.job_id)

        try:
            yield _starting_runtime_output(live_state, "Upscale 2x")

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
                        yield _queued_waiting_output(live_state, ticket.job_id, int(pos) if pos else 0)
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

    def _compute_live_plan(args):
        """args = [input_file] + inputs_list values + [shared_state]."""
        live_state = args[-1] if (args and isinstance(args[-1], dict)) else {}
        try:
            return service["live_plan"](
                args[0],
                *args[1:-1],
                state=snapshot_queue_state(live_state),
                global_settings_snapshot=snapshot_global_settings(global_settings),
            )
        except Exception as exc:
            return f"[ERROR] Chunk plan failed: {exc}"

    def run_plan_preview(*args):
        """
        Instant in-process chunk plan under the button. Clicking also ARMS the
        live mode: from then on the plan re-renders automatically whenever a
        plan-relevant parameter changes (ComfyUI live-plan behavior).
        """
        text = _compute_live_plan(list(args))
        return gr.update(value=text, visible=True), True

    def live_plan_refresh(armed, *args):
        """Auto-refresh handler; inert until the user clicks Preview once."""
        if not armed:
            return gr.skip()
        return gr.update(value=_compute_live_plan(list(args)), visible=True)

    LTX25_chunk_sync_signature = gr.State(value="")
    LTX25_pre_run_output_signature = gr.State(value="")

    def _refresh_chunk_preview_ui_if_needed(state, previous_signature: str = ""):
        preview = (state or {}).get("seed_controls", {}).get("ltx25_chunk_preview", {})
        signature = _sync_signature(preview if isinstance(preview, dict) else {"preview": None})
        if signature == str(previous_signature or ""):
            return gr.skip(), gr.skip(), gr.skip(), previous_signature
        chunk_status_upd, chunk_gallery_upd, chunk_video_upd = refresh_chunk_preview_ui(state)
        return chunk_status_upd, chunk_gallery_upd, chunk_video_upd, signature

    run_outputs = [
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
    ]

    # Main processing
    upscale_btn.click(
        fn=capture_latest_output_signature,
        inputs=[shared_state],
        outputs=[LTX25_pre_run_output_signature],
        queue=False,
        show_progress="hidden",
    )

    run_evt = upscale_btn.click(
        fn=run_upscale_with_queue,
        inputs=[input_file] + inputs_list + [shared_state],
        outputs=run_outputs,
        concurrency_limit=32,
        concurrency_id="app_processing_queue",
        trigger_mode="multiple",
    )
    run_evt.then(
        fn=_refresh_chunk_preview_ui_if_needed,
        inputs=[shared_state, LTX25_chunk_sync_signature],
        outputs=[chunk_status, chunk_gallery, chunk_preview_video, LTX25_chunk_sync_signature],
    )
    run_evt.then(
        fn=auto_transfer_latest_output_to_input,
        inputs=[
            pre_resize_longer_edge,
            auto_transfer_output_to_input,
            LTX25_pre_run_output_signature,
            shared_state,
        ],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )

    # Chunk plan preview (both buttons -> same lightweight handler, no GPU queue ticket).
    plan_live_armed = gr.State(value=False)

    plan_chunk_btn.click(
        fn=run_plan_preview,
        inputs=[input_file] + inputs_list + [shared_state],
        outputs=[plan_output_box, plan_live_armed],
        show_progress="hidden",
    )
    plan_chunk_btn_row.click(
        fn=run_plan_preview,
        inputs=[input_file] + inputs_list + [shared_state],
        outputs=[plan_output_box, plan_live_armed],
        show_progress="hidden",
    )

    # Live plan: once armed by a Preview click, every plan-relevant parameter
    # change re-renders the plan instantly (pure math + cached ffprobe/NVML,
    # no subprocess). gradio 6.18+ fires .change on programmatic updates too,
    # so model switches that auto-set Steps also refresh the plan.
    _plan_relevant_events = [
        input_path.change,
        input_file.change,
        token_budget_mode.change,
        max_latent_tokens.change,
        max_chunk_frames.change,
        overlap_frames.change,
        reserve_vram_gb.change,
        steps.change,
        pre_resize_longer_edge.change,
        start_frame.change,
        end_frame.change,
        model_name.change,
        device.change,
    ]
    gr.on(
        triggers=_plan_relevant_events,
        fn=live_plan_refresh,
        inputs=[plan_live_armed, input_file] + inputs_list + [shared_state],
        outputs=[plan_output_box],
        show_progress="hidden",
        trigger_mode="always_last",
    )

    shared_state.change(
        fn=_refresh_chunk_preview_ui_if_needed,
        inputs=[shared_state, LTX25_chunk_sync_signature],
        outputs=[chunk_status, chunk_gallery, chunk_preview_video, LTX25_chunk_sync_signature],
        queue=False,
        show_progress="hidden",
    )

    def _cancel_with_confirmation_reset(ok):
        if ok:
            status_upd, log_msg = service["cancel_action"]()
            return status_upd, log_msg, gr.update(value=False)
        message = warn_cancel_confirmation()
        return gr.update(value=f"WARNING: {message}", visible=True), message, gr.update(value=False)

    cancel_btn.click(
        fn=_cancel_with_confirmation_reset,
        inputs=[cancel_confirm],
        outputs=[status_box, log_box, cancel_confirm],
    )

    open_outputs_btn.click(
        fn=service["open_outputs_folder"],
        outputs=status_box,
    )

    clear_temp_btn.click(
        fn=lambda: service["clear_temp_folder"](False),
        outputs=status_box,
    )

    # UNIVERSAL PRESET EVENT WIRING (must be last, after inputs_list is final)
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
        tab_name="ltx25",
    )

    return {
        "inputs_list": inputs_list,
        "preset_dropdown": preset_dropdown,
        "preset_status": preset_status,
    }
