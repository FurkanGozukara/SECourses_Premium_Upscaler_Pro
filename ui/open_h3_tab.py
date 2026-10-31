"""
Open-H3 Regenerate Tab - Self-contained modular implementation
NVIDIA Open-H3-Regenerate-2K on MiniMax H3: a low-resolution video (or image) regenerated
natively at 1080P / 2K / 4K in three diffusion steps.
Uses the Universal Preset System. Mirrors ui/ltx25_tab.py wiring (left: input + model
settings, right: output/status).
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any
import html
import threading
import time

from shared.services.open_h3_service import (
    build_open_h3_callbacks,
    OPENH3_ORDER,
    OPENH3_DEFAULT_PROMPT,
    OPENH3_DEFAULT_START_SIGMA,
    OPENH3_RESOLUTION_PRESETS,
    OPENH3_DRAFT_SIZES,
    OPENH3_FPS_MODES,
    OPENH3_AUDIO_MODES,
    OPENH3_SINK_OPTIONS,
    OPENH3_TOKEN_BUDGET_MODES,
    OPENH3_TOKEN_BUDGET_MODE_AUTO,
    OPENH3_ATTENTION_OPTIONS,
)
from shared.open_h3_constants import (
    OPENH3_DEFAULT_SEED,
    OPENH3_DIT_FILES,
    OPENH3_DIT_INT8,
    OPENH3_DIT_NAMES,
    OPENH3_DIT_SIZES_GB,
    OPENH3_TE_NAMES,
    OPENH3_TE_NVFP4,
    OPENH3_VAE_NAMES,
    OPENH3_VAE_SOL_FP16,
    openh3_download_size_gb,
)
from shared.path_utils import get_media_dimensions, normalize_path
from ui.universal_preset_section import (
    universal_preset_section,
    wire_universal_preset_events,
)
from ui.media_preview import preview_updates
from ui.shared_components import (
    autotune_modal_dismiss_js,
    autotune_vram_controls,
    warn_cancel_confirmation,
    wire_auto_tune,
    wire_auto_tune_on_run,
)
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

_DIT_NOTES = {
    "FL2VA Pruned INT8 ConvRot": "Reference speed setup; INT8 kernels on every RTX GPU.",
    "FL2VA Pruned FP8 Scaled": "FP8 weights; native FP8 math on RTX 40/50, emulated on older GPUs.",
    "FL2VA Pruned W6A8": "Smallest download (6-bit weights).",
    "FL2VA Pruned BF16": "Unquantized base; needs weight streaming below 48 GB VRAM.",
}


def open_h3_tab(
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
    Self-contained Open-H3 Regenerate tab following the SECourses modular pattern.
    """

    service = build_open_h3_callbacks(
        preset_manager, runner, run_logger, global_settings, shared_state,
        base_dir, temp_dir, output_dir
    )
    queue_manager = get_processing_queue_manager()
    defaults = service["defaults"]

    # UNIVERSAL PRESET: Load from shared_state
    seed_controls = shared_state.value.get("seed_controls", {})
    open_h3_settings = seed_controls.get("open_h3_settings", {})
    merged_defaults = defaults.copy()
    for key, value in open_h3_settings.items():
        if value is not None:
            merged_defaults[key] = value
    values = [merged_defaults[k] for k in OPENH3_ORDER]

    def _value(key: str, default=None):
        try:
            idx = OPENH3_ORDER.index(key)
            if 0 <= idx < len(values):
                raw = values[idx]
                if raw is None and default is not None:
                    return default
                return raw
        except Exception:
            pass
        return default

    def _choice(key: str, options, default):
        val = str(_value(key, default) or default)
        return val if val in set(options) else default

    # GPU detection (parent-process safe: NO torch import)
    cuda_available = False
    gpu_hint = ""
    device_choices: list[tuple[str, str]] = [("Auto (best available GPU)", "auto")]
    try:
        from shared.gpu_utils import get_gpu_info

        gpus = get_gpu_info()
        cuda_available = len(gpus) > 0
        for g in gpus:
            try:
                device_choices.append((f"GPU {g.id}: {g.name} ({float(g.total_memory_gb):.1f} GB)", str(g.id)))
            except Exception:
                device_choices.append((f"GPU {getattr(g, 'id', '?')}", str(getattr(g, "id", "0"))))
        if not cuda_available:
            gpu_hint = "CUDA not detected (nvidia-smi unavailable or no NVIDIA GPU)."
    except Exception as e:
        gpu_hint = f"CUDA detection failed: {str(e)}"
    device_choices.append(("CPU (not practical for MiniMax H3)", "cpu"))
    device_valid_values = {v for _, v in device_choices}
    device_value = str(_value("device", "auto") or "auto")
    if device_value not in device_valid_values:
        device_value = "auto"

    def _build_model_info_md(dit: str, te: str, vae: str) -> str:
        total = openh3_download_size_gb(dit, te, vae)
        return (
            f"**{dit}** &nbsp;`{OPENH3_DIT_FILES.get(dit, '')}` ({OPENH3_DIT_SIZES_GB.get(dit, 0):g} GB) - "
            f"{_DIT_NOTES.get(dit, '')}\n\n"
            f"- First run downloads only the selected files into `OpenH3_Models` (about **{total:.0f} GB** with this "
            "selection: DiT + text encoder + VAE + the Open-H3 LoRA, Sharp curve branches, latent upscaler and audio VAE).\n"
            "- Recipe: FL2VA + merged LoRA (refiner + LightX2V FL2VA Turbo + LMS Sharp) + 50 Sharp curve branches, "
            "3 Euler steps (shift-6 grid from the Start Sigma; 0 = NVIDIA's 0.909375 release grid), CFG 1, "
            "Sol sparse attention, fast Sol VAE decoder."
        )

    def _build_sizing_md(path_val, preset_val, factor_val, max_edge_val, draft_val) -> gr.update:
        normalized = normalize_path(path_val) if path_val else ""
        if not normalized:
            return gr.update(value="", visible=False)
        dims = get_media_dimensions(normalized)
        if not dims:
            return gr.update(value="**Output size:** unable to read input dimensions yet.", visible=True)
        try:
            from shared.open_h3_runner import _draft_size, compute_target_size

            settings = {"resolution_preset": preset_val, "upscale_factor": factor_val,
                        "max_target_resolution": max_edge_val, "draft_size": draft_val}
            tw, th, note = compute_target_size(int(dims[0]), int(dims[1]), settings)
            draft = _draft_size(settings, int(dims[0]), int(dims[1]))
            text = (f"**Output size:** `{int(dims[0])}x{int(dims[1])}` -> `{tw}x{th}` ({note}) | "
                    f"draft `{draft}` | aspect ratio kept, no cropping.")
        except Exception as exc:
            text = f"**Output size:** could not be computed ({exc})"
        return gr.update(value=text, visible=True)

    if not cuda_available:
        gr.Markdown(
            f'<div style="background: #fff3cd; padding: 12px; border-radius: 8px; border: 1px solid #ffc107;">'
            f'<strong> GPU Acceleration Unavailable</strong><br>{gpu_hint}<br><br>'
            f'Open-H3 Regenerate runs the MiniMax H3 DiT and needs a CUDA GPU.</div>',
            elem_classes="warning-text",
        )

    with gr.Row():
        # Left Column: Input & Settings
        with gr.Column(scale=3):
            gr.Markdown("#### 📥 Input Source")

            with gr.Group():
                with gr.Row():
                    with gr.Column(scale=2, elem_classes=["OPENH3-input-source-col"]):
                        input_file = gr.File(
                            label="Upload video or image (optional)",
                            type="filepath",
                            file_types=["video", "image"],
                        )
                        input_path = gr.Textbox(
                            label="Input Path",
                            value=_value("input_path", ""),
                            placeholder="C:/path/to/video.mp4",
                            info="Video file or a single image (images are refined as a 5-frame still).",
                        )
                        copy_output_into_input_btn = gr.Button(
                            "Copy Output Into Input",
                            elem_classes=["action-btn", "action-btn-source-seed", "OPENH3-copy-output-compact", "sec-btn-teal"],
                            size="md",
                        )
                        auto_transfer_output_to_input = gr.Checkbox(
                            label="Auto Transfer Output to Input",
                            value=bool(_value("auto_transfer_output_to_input", False)),
                            info="After a run completes, automatically copy the latest output path into Input Path.",
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

                output_override = gr.Textbox(
                    label="Output Override (folder or file)",
                    value=_value("output_override", ""),
                    placeholder="Leave empty for auto naming",
                    info=(
                        "Optional custom output location. A folder saves into that folder. "
                        "A file path renames the final output (.mp4 for videos, .png for images)."
                    ),
                )
                resume_run_dir = gr.Textbox(
                    label="Resume Run Folder (chunk/scene resume)",
                    value=str(_value("resume_run_dir", "") or ""),
                    placeholder="Optional: G:/.../outputs/0019",
                    info=(
                        "Optional. Works only when app-level chunking is enabled in the Resolution tab "
                        "(Auto Chunk OFF + fixed chunk size)."
                    ),
                )

                with gr.Accordion("📁 Batch Processing", open=False):
                    batch_enable = gr.Checkbox(
                        label="Enable Batch",
                        value=bool(_value("batch_enable", False)),
                        info="Process every video and image in a folder.",
                    )
                    batch_input = gr.Textbox(
                        label="Batch Input Folder",
                        value=_value("batch_input_path", ""),
                        placeholder="Folder with videos / images",
                    )
                    batch_output = gr.Textbox(
                        label="Batch Output Folder",
                        value=_value("batch_output_path", ""),
                        placeholder="Output directory (empty = <input>/upscaled_files)",
                    )

            with gr.Accordion("📐 Output Resolution", open=True):
                with gr.Row():
                    resolution_preset = gr.Dropdown(
                        label="Target Resolution",
                        choices=list(OPENH3_RESOLUTION_PRESETS),
                        value=_choice("resolution_preset", OPENH3_RESOLUTION_PRESETS, OPENH3_RESOLUTION_PRESETS[0]),
                        info=("Short side of the output (1440 for 2K). Aspect ratio is kept. 2K is the trained "
                              "target of Open-H3-Regenerate-2K; 768p refines without upscaling."),
                    )
                    draft_size = gr.Dropdown(
                        label="Draft Size",
                        choices=list(OPENH3_DRAFT_SIZES),
                        value=_choice("draft_size", OPENH3_DRAFT_SIZES, OPENH3_DRAFT_SIZES[0]),
                        info=("Size the input is encoded at before the latent upscale. NVIDIA uses 768p for H3 drafts "
                              "and 576p for drafts from other models."),
                    )
                with gr.Row():
                    upscale_factor = gr.Slider(
                        label="Upscale Factor (Custom mode)",
                        minimum=1.0,
                        maximum=8.0,
                        step=0.05,
                        value=float(_value("upscale_factor", 2.0) or 2.0),
                        info="Used when Target Resolution is 'Custom upscale factor'.",
                    )
                    max_target_resolution = gr.Number(
                        label="Max Edge (px, 0 = off)",
                        value=int(_value("max_target_resolution", 0) or 0),
                        precision=0,
                        minimum=0,
                        info="Caps the longer output edge in every mode.",
                    )
                auto_tune_btn, reserve_vram_gb, auto_tune_on_run = autotune_vram_controls(
                    _value("reserve_vram_gb", None),
                    _value("auto_tune_on_run", True),
                )
                gr.Markdown(
                    "**Auto Tune (instant, no test runs):** reads your GPU's total and in-use VRAM, the input and the "
                    "output size, then sets the token budget to Auto, the longest window (one window over the whole "
                    "clip when it fits, up to 361 frames) and allows all 50 DiT blocks on the GPU, and shows the "
                    "window plan. It keeps `Save VRAM (GB)` free; the engine never uses shared GPU memory."
                )

            with gr.Accordion("🎛️ Open-H3 Models", open=True):
                isolated_stages = gr.Checkbox(
                    label="🧩 Run each stage as a separate process (lowest RAM and VRAM)",
                    value=bool(_value("isolated_stages", False)),
                    info=("Text encoding, VAE encode + latent upscale, DiT refine and VAE decode each run in their own "
                          "process and hand their results to the next one through files, so a finished stage (text "
                          "encoder, VAE) keeps no RAM or VRAM. Adds about 15-30 s per run for the extra process starts."),
                )
                model_name = gr.Dropdown(
                    label="MiniMax H3 Base (FL2VA)",
                    choices=list(OPENH3_DIT_NAMES),
                    value=_choice("model_name", OPENH3_DIT_NAMES, OPENH3_DIT_INT8),
                    info="FL2VA is the reference base of the regenerate recipe. INT8 ConvRot = recommended.",
                )
                with gr.Row():
                    text_encoder = gr.Dropdown(
                        label="Text Encoder",
                        choices=list(OPENH3_TE_NAMES),
                        value=_choice("text_encoder", OPENH3_TE_NAMES, OPENH3_TE_NVFP4),
                        info="Qwen3-VL 32B (50 layers). The prompt embedding is cached, so it runs once per prompt.",
                    )
                    video_vae = gr.Dropdown(
                        label="Video VAE",
                        choices=list(OPENH3_VAE_NAMES),
                        value=_choice("video_vae", OPENH3_VAE_NAMES, OPENH3_VAE_SOL_FP16),
                        info="Sol = H3 encoder + NVIDIA's fast distilled decoder (default). Released = original decoder, slower.",
                    )
                model_info_md = gr.Markdown(_build_model_info_md(
                    _choice("model_name", OPENH3_DIT_NAMES, OPENH3_DIT_INT8),
                    _choice("text_encoder", OPENH3_TE_NAMES, OPENH3_TE_NVFP4),
                    _choice("video_vae", OPENH3_VAE_NAMES, OPENH3_VAE_SOL_FP16),
                ))

            with gr.Accordion("✍️ Prompt", open=True):
                prompt = gr.Textbox(
                    label="Prompt",
                    value=str(_value("prompt", OPENH3_DEFAULT_PROMPT) or ""),
                    lines=5,
                    max_lines=14,
                    info=("H3 regenerates the clip from this text plus the draft. Describing the scene (subjects, "
                          "clothing, setting, light) keeps identities closest; the default is a generic faithful prompt."),
                )
                reset_prompt_btn = gr.Button(
                    "Reset Prompt",
                    size="md",
                    elem_classes=["action-btn", "sec-btn-slate"],
                )

            with gr.Accordion("🧠 VRAM, Windows & Attention", open=False):
                gr.Markdown(
                    "**Shared GPU memory is never used.** The engine limits itself to *free VRAM - Save VRAM*. "
                    "When a step needs more, it moves just enough DiT blocks to system RAM (block swap) and redoes "
                    "that window automatically, so the run still completes, with as many blocks as possible kept "
                    "on the GPU for speed. Windows tip: NVIDIA Control Panel > Manage 3D settings > "
                    "*CUDA - Sysmem Fallback Policy* > *Prefer No Sysmem Fallback* stops every app from spilling.",
                    elem_classes=["open-h3-vram-note"],
                )
                with gr.Row():
                    max_gpu_blocks = gr.Slider(
                        label="Max DiT Blocks on GPU (Block Swap)",
                        minimum=0,
                        maximum=50,
                        step=1,
                        value=int(_value("max_gpu_blocks", 50) if _value("max_gpu_blocks", 50) is not None else 50),
                        info=("50 = keep as many of the 50 blocks on the GPU as fit (fastest). Lower = less VRAM, "
                              "slower: the rest stream from RAM every step."),
                    )
                with gr.Row():
                    token_budget_mode = gr.Radio(
                        label="Token Budget Mode",
                        choices=list(OPENH3_TOKEN_BUDGET_MODES),
                        value=_choice("token_budget_mode", OPENH3_TOKEN_BUDGET_MODES, OPENH3_TOKEN_BUDGET_MODE_AUTO),
                        info="Auto sizes each window from free VRAM; long clips are split into cross-faded windows.",
                    )
                    max_tokens = gr.Number(
                        label="Max Tokens per Window (Manual mode)",
                        value=int(_value("max_tokens", 160000) or 160000),
                        precision=0,
                        info="Packed sequence length limit (text + audio + video tokens).",
                    )
                with gr.Row():
                    max_window_frames = gr.Slider(
                        label="Max Window Frames",
                        minimum=5,
                        maximum=361,
                        step=1,
                        value=int(_value("max_window_frames", 121) or 121),
                        info="121 = 5 s at 24 fps (reference). Fewer frames per window = less VRAM per window.",
                    )
                    overlap_frames = gr.Slider(
                        label="Window Overlap Frames",
                        minimum=0,
                        maximum=64,
                        step=1,
                        value=int(_value("overlap_frames", 8) or 8),
                        info="Frames cross-faded between consecutive windows.",
                    )
                ffn_chunk_tokens = gr.Number(
                    label="FFN Chunk Tokens (0 = off)",
                    value=int(_value("ffn_chunk_tokens", 32768) or 0),
                    precision=0,
                    info="Runs the MLP in token slices: same result, far less activation VRAM.",
                )
                gr.Markdown("**Attention**", elem_classes=["open-h3-vram-note"])
                with gr.Row():
                    sparse_attention = gr.Checkbox(
                        label="Sparse Attention (Sol-Attn)",
                        value=bool(_value("sparse_attention", True)),
                        info="Faster refine than dense attention with the same structure (2K, 49 frames: 34 s vs 44 s).",
                    )
                    attention_backend = gr.Dropdown(
                        label="Dense Attention Backend",
                        choices=list(OPENH3_ATTENTION_OPTIONS),
                        value=_choice("attention_backend", OPENH3_ATTENTION_OPTIONS, "auto"),
                        info="Used for dense blocks and when Sol-Attn has no kernel for the GPU.",
                    )
                with gr.Row():
                    sparse_tau = gr.Slider(
                        label="Sol-Attn Tau",
                        minimum=0.0,
                        maximum=4.0,
                        step=0.05,
                        value=float(_value("sparse_tau", 1.0) or 0.0),
                        info="Higher = sparser and faster. Upstream quality preset 1.0, fast preset 1.75.",
                    )
                    sparse_dense_steps = gr.Slider(
                        label="Sol-Attn Dense Steps",
                        minimum=-1,
                        maximum=8,
                        step=1,
                        value=int(_value("sparse_dense_steps", 1)),
                        info=("First N refine steps run dense, the rest sparse. 1 = upstream quality preset (same "
                              "structure as dense). 0 = all sparse (fast preset, objects can reshape). -1 = use End Percent."),
                    )
                    sparse_end_percent = gr.Slider(
                        label="Sol-Attn End Percent",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=float(_value("sparse_end_percent", 0.9) or 0.9),
                        info="Only with Dense Steps -1: steps above this sigma window run sparse.",
                    )
                with gr.Row():
                    dense_blocks = gr.Textbox(
                        label="Dense Blocks",
                        value=str(_value("dense_blocks", "0,1,49") or ""),
                        info="Blocks that always run dense. Reference: 0,1,49.",
                    )
                    sink_conditioning = gr.Dropdown(
                        label="Sink Conditioning",
                        choices=list(OPENH3_SINK_OPTIONS),
                        value=_choice("sink_conditioning", OPENH3_SINK_OPTIONS, "exact_kv_and_rows"),
                        info="Text/audio rows attended exactly by every query (reference: exact_kv_and_rows).",
                    )
                plan_chunk_btn = gr.Button(
                    "📊 Preview Window Plan",
                    size="md",
                    elem_classes=["action-btn", "action-btn-optimize", "sec-btn-crimson"],
                )
                plan_output_box = gr.Textbox(
                    label="📊 Plan Result",
                    value="",
                    lines=8,
                    max_lines=16,
                    interactive=False,
                    visible=False,
                    buttons=["copy"],
                )

            with gr.Accordion("🎬 Input Handling", open=False):
                with gr.Row():
                    fps_mode = gr.Dropdown(
                        label="Frame Rate Handling",
                        choices=list(OPENH3_FPS_MODES),
                        value=_choice("fps_mode", OPENH3_FPS_MODES, OPENH3_FPS_MODES[0]),
                        info="Keep = every source frame, same timing and audio sync. Resample = NVIDIA's 24 fps preprocessing.",
                    )
                    audio_mode = gr.Dropdown(
                        label="Audio Conditioning",
                        choices=list(OPENH3_AUDIO_MODES),
                        value=_choice("audio_mode", OPENH3_AUDIO_MODES, OPENH3_AUDIO_MODES[0]),
                        info=("Audio the refine attends to. Source soundtrack keeps talking mouths in sync (default; "
                              "videos without audio use silence). The output soundtrack is always the source audio."),
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
                    fps = gr.Number(
                        label="FPS Override",
                        value=float(_value("fps", 0.0) or 0.0),
                        info="0 = keep source FPS (the Output tab FPS override applies too).",
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
                        choices=["auto"],
                        value="auto",
                        info="MP4 for videos (source audio copied), the Output tab image format for images.",
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
                        info="After batch completion, keep only final outputs.",
                    )
                keep_latents = gr.Checkbox(
                    label="Keep Refined Latents",
                    value=bool(_value("keep_latents", False)),
                    info="The refined latents are saved before decoding; keep the file after a successful run.",
                )
                gr.Markdown(
                    "**Video Codec / CRF / FPS Override defaults:** Controlled from `Output > Video Output` "
                    "and automatically applied here."
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
                tab_name="open_h3",
                inputs_list=[],
                base_dir=base_dir,
                models_list=list(OPENH3_DIT_NAMES),
                open_accordion=True,
            )

        # Right Column: Output & Controls
        with gr.Column(scale=2):
            gr.Markdown("#### 📤 Output & Actions")
            status_box = gr.Markdown(value="Ready.", visible=False, elem_classes=["runtime-status-box"])
            with gr.Group(
                visible=False,
                elem_id="openh3-autotune-notice-modal",
                elem_classes=["autotune-modal-overlay"],
            ) as h3_autotune_notice_modal:
                with gr.Group(elem_classes=["autotune-modal-card"]):
                    with gr.Row(elem_classes=["autotune-modal-header"]):
                        gr.Markdown("Auto Tune Update", elem_classes=["autotune-modal-title"])
                        h3_autotune_notice_close_btn = gr.Button("X", size="sm", elem_classes=["autotune-modal-close", "sec-btn-slate"])
                    h3_autotune_notice_text = gr.Markdown("", elem_classes=["autotune-modal-body"])
                    with gr.Row(elem_classes=["autotune-modal-actions"]):
                        h3_autotune_notice_ok_btn = gr.Button("OK", variant="primary", elem_classes=["autotune-modal-ok", "sec-btn-pink"])
            progress_indicator = gr.Markdown(value="", visible=False, elem_classes=["runtime-progress-box"])

            with gr.Group():
                upscale_btn = gr.Button(
                    "🚀 Regenerate / Upscale",
                    variant="primary",
                    size="lg",
                    elem_classes=["action-btn", "action-btn-upscale", "sec-btn-emerald"],
                )
                preview_btn = gr.Button(
                    "🖼️ Preview First Frame",
                    size="md",
                    elem_classes=["action-btn", "action-btn-preview", "sec-btn-violet"],
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
                        "📊 Preview Window Plan",
                        elem_classes=["action-btn", "action-btn-optimize", "sec-btn-crimson"],
                    )

            with gr.Accordion("📺 Regenerated Output", open=True):
                output_video = gr.Video(
                    label="🎥 Regenerated Video",
                    interactive=False,
                    visible=False,
                    height=420,
                    buttons=["download"],
                )
                output_image = gr.Image(
                    label="🖼️ Regenerated Image",
                    interactive=False,
                    visible=False,
                    buttons=["download"],
                )
            last_processed = gr.Markdown("Processing results will appear here.")

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

            with gr.Accordion("⚙️ Refine Sampling", open=True):
                with gr.Row():
                    steps = gr.Slider(
                        label="Steps",
                        minimum=1,
                        maximum=8,
                        step=1,
                        value=int(_value("steps", 3) or 3),
                        info="3 = reference. 2 = about 30% faster, softer fine texture.",
                    )
                    start_sigma = gr.Slider(
                        label="Start Sigma (regenerate strength)",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.005,
                        value=float(_value("start_sigma", OPENH3_DEFAULT_START_SIGMA) or 0.0),
                        info=("0.8 = faithful default: same content, new detail. Lower = closer to the input. "
                              "0 = NVIDIA release grid (0.909): strongest redraw, identities and designs can change."),
                    )
                with gr.Row():
                    seed = gr.Number(
                        label="Seed",
                        value=int(_value("seed", OPENH3_DEFAULT_SEED) or OPENH3_DEFAULT_SEED),
                        precision=0,
                        info="Reference seed 1766580932.",
                    )
                    randomize_seed = gr.Checkbox(
                        label="Randomize Seed",
                        value=bool(_value("randomize_seed", False)),
                        info="Pick a fresh random seed for every run (logged in the run log).",
                    )
                with gr.Row():
                    lora_strength = gr.Slider(
                        label="Open-H3 LoRA Strength",
                        minimum=-2.0,
                        maximum=2.0,
                        step=0.01,
                        value=float(_value("lora_strength", 1.0) or 0.0) if _value("lora_strength", 1.0) is not None else 1.0,
                        info="The merge is exact at 1.0 (refiner + Turbo + Sharp strengths are folded in).",
                    )
                    curve_strength = gr.Slider(
                        label="Sharp Curve Branches Strength",
                        minimum=-2.0,
                        maximum=2.0,
                        step=0.01,
                        value=float(_value("curve_strength", 1.0) or 0.0) if _value("curve_strength", 1.0) is not None else 1.0,
                        info="LMS Sharp's 50 AdaLN curve branches (part of the reference recipe). 0 = off.",
                    )

            gr.Markdown("""
            #### Open-H3 Regenerate Guide

            - **What it does:** the input is encoded at the draft size, upscaled in latent space onto the target grid and
              regenerated in 3 diffusion steps by MiniMax H3 with NVIDIA's Open-H3-Regenerate-2K adapters. It adds real
              new detail (skin, fabric, foliage) instead of sharpening.
            - **Faithfulness:** the default `Start Sigma` 0.8 keeps the input's content and adds detail. `0` uses
              NVIDIA's release grid (0.909), which redraws a lot: faces, designs and text can change. Describing the
              scene in the prompt also helps.
            - **Long videos:** split into windows of up to `Max Window Frames` and cross-faded. Enable fixed app-level
              chunking in the Resolution tab to cut at scenes instead.
            - **Speed (RTX 5090 class):** about 15-25 s per 5 s window at 1080p and about 2 minutes per 5 s window at
              2K (portrait 1440x2560), after a one-time model load. The prompt embedding is cached on disk.
            - **VRAM:** shared GPU memory is never used; see `🧠 VRAM, Windows & Attention` for Save VRAM and Block Swap.
            - Use `🖼️ Preview First Frame` to check the look in seconds before a long run.
            - Audio is copied from the source; the Output tab's audio and codec settings apply afterwards.
            """)

    # Collect inputs - MUST match OPENH3_ORDER positions exactly.
    inputs_list = [
        input_path, output_override, output_format,
        model_name, text_encoder, video_vae, isolated_stages,
        prompt, seed, randomize_seed,
        resolution_preset, upscale_factor, max_target_resolution, draft_size,
        steps, start_sigma, lora_strength, curve_strength,
        sparse_attention, sparse_tau, sparse_end_percent, dense_blocks, sink_conditioning,
        attention_backend, token_budget_mode, max_tokens, max_window_frames, overlap_frames,
        ffn_chunk_tokens, reserve_vram_gb, max_gpu_blocks,
        fps_mode, audio_mode, start_frame, end_frame, keep_latents, fps,
        device, save_metadata, auto_transfer_output_to_input,
        batch_enable, batch_input, batch_output,
        resume_run_dir, keep_only_output_files,
        auto_tune_on_run,
        sparse_dense_steps,
    ]

    if len(inputs_list) != len(OPENH3_ORDER):
        import logging
        logging.getLogger("OpenH3Tab").error(
            f"ERROR: inputs_list ({len(inputs_list)}) != OPENH3_ORDER ({len(OPENH3_ORDER)})"
        )

    # ------------------------------------------------------------------ #
    # Wire up events
    # ------------------------------------------------------------------ #

    def _on_model_switch(dit, te, vae):
        return gr.update(value=_build_model_info_md(dit, te, vae))

    gr.on(
        triggers=[model_name.input, text_encoder.input, video_vae.input],
        fn=_on_model_switch,
        inputs=[model_name, text_encoder, video_vae],
        outputs=[model_info_md],
        queue=False,
        show_progress="hidden",
    )

    reset_prompt_btn.click(
        fn=lambda: gr.update(value=OPENH3_DEFAULT_PROMPT),
        outputs=[prompt],
        queue=False,
        show_progress="hidden",
    )

    sizing_inputs = [resolution_preset, upscale_factor, max_target_resolution, draft_size]

    # ---------------- Input caching / detection / sizing ---------------- #

    def _analysis_progress_note(pct: int) -> str:
        if pct < 15:
            return "Reading media metadata..."
        if pct < 45:
            return "Computing the output size..."
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
            '<div class="processing-text">Analyzing input</div>'
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
            else:
                parts.append("&nbsp;&nbsp;⚠️ Open-H3 processes videos and single images.")
            return gr.update(value=" ".join(parts), visible=True)
        except Exception as e:
            return gr.update(value=f"ERROR: **Detection Error**\n\n{str(e)}", visible=True)

    def _run_analysis_payload(path_val, sizing_vals, state):
        det = _build_input_detection_md(path_val or "")
        info = _build_sizing_md(path_val or "", *sizing_vals)
        img_prev, vid_prev = preview_updates(path_val)
        return img_prev, vid_prev, det, info, state

    def _iter_analysis_with_progress(path_val, sizing_vals, state):
        result: Dict[str, Any] = {}

        def _worker():
            try:
                result["payload"] = _run_analysis_payload(path_val, sizing_vals, state)
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

    def cache_input_upload(val, *rest):
        sizing_vals, state = rest[:-1], rest[-1]
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
            for event_type, payload_a, _ in _iter_analysis_with_progress(val, sizing_vals, state):
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

    def cache_input_path(path_val, *rest):
        sizing_vals, state = rest[:-1], rest[-1]
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
            for event_type, payload_a, _ in _iter_analysis_with_progress(path_val, sizing_vals, state):
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
        inputs=[input_file] + sizing_inputs + [shared_state],
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
        inputs=[input_path] + sizing_inputs + [shared_state],
        outputs=[input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )

    def refresh_sizing(path_val, *sizing_vals):
        return _build_sizing_md(path_val or "", *sizing_vals)

    gr.on(
        triggers=[c.change for c in sizing_inputs],
        fn=refresh_sizing,
        inputs=[input_path] + sizing_inputs,
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

        batch_outputs = seed_controls.get("open_h3_batch_outputs", [])
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
        sizing_vals,
        state,
        source_label="Output transferred to input.",
    ):
        state = state or {}
        state.setdefault("seed_controls", {})
        state["seed_controls"]["last_input_path"] = output_path_val or ""

        try:
            img_prev, vid_prev, det, info, state_out = _run_analysis_payload(
                output_path_val or "",
                sizing_vals,
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

    def copy_latest_output_to_input(*args):
        sizing_vals, state = args[:-1], args[-1]
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
            sizing_vals,
            state,
            source_label="Output path copied into input.",
        )

    def capture_latest_output_signature(state):
        return _output_path_signature(_resolve_latest_output_path(state))

    def auto_transfer_latest_output_to_input(*args):
        sizing_vals, auto_enabled, previous_signature, state = args[:4], args[4], args[5], args[6]
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
            sizing_vals,
            state,
            source_label="Auto-transferred latest output into input.",
        )

    copy_output_into_input_btn.click(
        fn=copy_latest_output_to_input,
        inputs=sizing_inputs + [shared_state],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )

    # ---------------- Chunk preview ---------------- #

    def refresh_chunk_preview_ui(state):
        preview = (state or {}).get("seed_controls", {}).get("open_h3_chunk_preview", {})
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
            videos = (state or {}).get("seed_controls", {}).get("open_h3_chunk_preview", {}).get("videos", [])
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
        outputs = (state or {}).get("seed_controls", {}).get("open_h3_batch_outputs", [])
        if not isinstance(outputs, list):
            outputs = []
        outputs = [str(p) for p in outputs if p and Path(str(p)).exists()]
        return gr.update(value=outputs, visible=bool(outputs))

    def _last_processed_text(state, vid_upd, img_upd) -> str:
        outputs = (state or {}).get("seed_controls", {}).get("open_h3_batch_outputs", [])
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
                gr.update(value="ERROR: Invalid Open-H3 payload"),
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
            "plan needs a video",
            "expects a video or an image",
            "preview ready",
            "preview failed",
            "no video or image files found",
        )
        is_terminal = any(tok in status_lc for tok in terminal_tokens) or any(
            tok in log_lc for tok in ("critical error", "processing failed", "cancelled", "out of vram")
        )

        if (status_text or log_tail) and not is_terminal:
            title = _compact_single_line(status_text or "Open-H3 processing...", max_len=96)
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
        yield from _run_with_queue(False, "Regenerate", *args, progress=progress)

    def run_preview_with_queue(*args, progress=gr.Progress()):
        yield from _run_with_queue(True, "First-frame preview", *args, progress=progress)

    def _run_with_queue(preview_only, action_label, *args, progress=None):
        live_state = args[-1] if (args and isinstance(args[-1], dict)) else {}
        queued_state = snapshot_queue_state(live_state)
        queued_global_settings = snapshot_global_settings(global_settings)
        queue_enabled = bool(queued_global_settings.get("queue_enabled", True))
        queue_resource_keys, queue_resource_label = resolve_queue_gpu_resources(queued_state, queued_global_settings)
        ticket = queue_manager.submit(
            "open_h3",
            "Upscale",
            resource_keys=queue_resource_keys,
            resource_label=queue_resource_label,
        )
        acquired_slot = queue_manager.is_active(ticket.job_id)

        try:
            yield _starting_runtime_output(live_state, action_label)

            if not queue_enabled:
                if not acquired_slot:
                    queue_manager.cancel_waiting([ticket.job_id])
                    yield _queue_disabled_busy_output(live_state)
                    return
                for payload in service["run_action"](
                    args[0],
                    *args[1:-1],
                    preview_only=preview_only,
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
                preview_only=preview_only,
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
            return f"[ERROR] Window plan failed: {exc}"

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

    OPENH3_chunk_sync_signature = gr.State(value="")
    OPENH3_pre_run_output_signature = gr.State(value="")

    def _refresh_chunk_preview_ui_if_needed(state, previous_signature: str = ""):
        preview = (state or {}).get("seed_controls", {}).get("open_h3_chunk_preview", {})
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

    # Registered before the Upscale job: Gradio runs a button's listeners in order and waits for a
    # queued job to finish before the next one, so a later listener would update the UI only after the run.
    h3_tuned = {
        "token_budget_mode": token_budget_mode,
        "max_window_frames": max_window_frames,
        "max_gpu_blocks": max_gpu_blocks,
        "ffn_chunk_tokens": ffn_chunk_tokens,
    }
    wire_auto_tune_on_run(
        upscale_btn=upscale_btn,
        action=service["auto_tune_action"],
        inputs=[input_file] + inputs_list,
        tuned=h3_tuned,
        shared_state=shared_state,
        auto_tune_on_run=auto_tune_on_run,
        global_settings=global_settings,
    )

    # Main processing
    upscale_btn.click(
        fn=capture_latest_output_signature,
        inputs=[shared_state],
        outputs=[OPENH3_pre_run_output_signature],
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
        inputs=[shared_state, OPENH3_chunk_sync_signature],
        outputs=[chunk_status, chunk_gallery, chunk_preview_video, OPENH3_chunk_sync_signature],
    )
    run_evt.then(
        fn=auto_transfer_latest_output_to_input,
        inputs=sizing_inputs + [
            auto_transfer_output_to_input,
            OPENH3_pre_run_output_signature,
            shared_state,
        ],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )

    preview_btn.click(
        fn=run_preview_with_queue,
        inputs=[input_file] + inputs_list + [shared_state],
        outputs=run_outputs,
        concurrency_limit=32,
        concurrency_id="app_processing_queue",
        trigger_mode="multiple",
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
        max_tokens.change,
        max_window_frames.change,
        overlap_frames.change,
        reserve_vram_gb.change,
        steps.change,
        start_sigma.change,
        resolution_preset.change,
        upscale_factor.change,
        max_target_resolution.change,
        draft_size.change,
        fps_mode.change,
        start_frame.change,
        end_frame.change,
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
        inputs=[shared_state, OPENH3_chunk_sync_signature],
        outputs=[chunk_status, chunk_gallery, chunk_preview_video, OPENH3_chunk_sync_signature],
        queue=False,
        show_progress="hidden",
    )

    def _cancel_with_confirmation_reset(ok):
        if ok:
            status_upd, log_msg = service["cancel_action"]()
            return status_upd, log_msg, gr.update(value=False)
        message = warn_cancel_confirmation()
        return gr.update(value=f"WARNING: {message}", visible=True), message, gr.update(value=False)

    wire_auto_tune(
        auto_tune_btn=auto_tune_btn,
        action=service["auto_tune_action"],
        inputs=[input_file] + inputs_list,
        tuned=h3_tuned,
        status_box=status_box,
        shared_state=shared_state,
        notice_text=h3_autotune_notice_text,
        notice_modal=h3_autotune_notice_modal,
        modal_id="openh3-autotune-notice-modal",
        global_settings=global_settings,
    )

    def _dismiss_h3_autotune_notice():
        return gr.update(value=""), gr.update(visible=False)

    for _btn in (h3_autotune_notice_ok_btn, h3_autotune_notice_close_btn):
        _btn.click(
            fn=_dismiss_h3_autotune_notice,
            outputs=[h3_autotune_notice_text, h3_autotune_notice_modal],
            queue=False,
            show_progress="hidden",
            js=autotune_modal_dismiss_js("openh3-autotune-notice-modal"),
        )

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
        tab_name="open_h3",
    )

    return {
        "inputs_list": inputs_list,
        "preset_dropdown": preset_dropdown,
        "preset_status": preset_status,
    }
