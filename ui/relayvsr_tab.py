"""
RelayVSR tab: the shared video SR tab (ui/video_sr_tab.py) with RelayVSR's own controls - Wan model file, precision,
keyframe window and Wan blocks kept on the GPU.
"""
from pathlib import Path
from typing import Any, Dict

import gradio as gr

from shared.relayvsr_constants import (
    RELAYVSR_PRECISIONS,
    RELAYVSR_WAN_DEFAULT,
    RELAYVSR_WAN_NAMES,
    RELAYVSR_WINDOW_DEFAULT,
    RELAYVSR_WINDOW_MAX,
    RELAYVSR_WINDOW_MIN,
)
from shared.services.relayvsr_service import RELAYVSR_ORDER, build_relayvsr_callbacks, canonical_relayvsr_scale
from ui.video_sr_tab import TabContext, VideoSRTabSpec, video_sr_tab


def _build_model_column(ctx: TabContext) -> None:
    v = ctx.value
    wan = str(v("wan_model", RELAYVSR_WAN_DEFAULT))
    ctx.c["wan_model"] = gr.Dropdown(
        label="Wan Model",
        choices=list(RELAYVSR_WAN_NAMES),
        value=wan if wan in RELAYVSR_WAN_NAMES else RELAYVSR_WAN_DEFAULT,
        info=("RelayVSR's large model, which writes the reference keyframes. INT8 ConvRot (default) is half the size "
              "of BF16 with the same look; both download on first use from the SECourses Hugging Face mirror "
              "(+ the 0.13 GB Flash decoder)."),
    )
    precision = str(v("precision", "auto")).lower()
    ctx.c["precision"] = gr.Dropdown(
        label="Precision",
        choices=list(RELAYVSR_PRECISIONS),
        value=precision if precision in RELAYVSR_PRECISIONS else "auto",
        info="auto = BF16 on GPUs that support it (RTX 30 and newer), FP16 otherwise.",
    )
    ctx.c["window_size"] = gr.Slider(
        label="Keyframe Window (frames)",
        minimum=RELAYVSR_WINDOW_MIN,
        maximum=RELAYVSR_WINDOW_MAX,
        step=1,
        value=int(v("window_size", RELAYVSR_WINDOW_DEFAULT) or RELAYVSR_WINDOW_DEFAULT),
        info=("Wan writes one reference keyframe every (window - 1) frames and the small Flash decoder relays the "
              "frames between them. 16 = the trained default. Smaller = more keyframes (steadier detail, slower); "
              "larger = faster."),
    )


def _build_runtime_row(ctx: TabContext) -> None:
    ctx.c["wan_gpu_blocks"] = gr.Slider(
        label="Wan Blocks on GPU (-1 = Auto)",
        minimum=-1,
        maximum=30,
        step=1,
        value=int(ctx.value("wan_gpu_blocks", -1) or -1),
        info=("-1 = every block that fits. Blocks that do not fit stay in RAM with their memory of recent keyframes "
              "and are copied to the GPU only when Wan runs (once per keyframe window): same result, somewhat slower. "
              "Auto Tune sets this."),
        scale=2,
    )


def _nothing(_ctx: TabContext) -> None:
    return None


RELAYVSR_TAB = VideoSRTabSpec(
    key="relayvsr",
    label="RelayVSR",
    build_service=build_relayvsr_callbacks,
    order=RELAYVSR_ORDER,
    canonical_scale=canonical_relayvsr_scale,
    model_label="RelayVSR",
    runtime_label=lambda ms: f"RelayVSR streaming pipeline (fixed {ms}x pass)",
    upscale_slider=dict(label="RelayVSR Upscale Factor", minimum=1.0, maximum=4.0, step=0.5,
                        info="The model always runs at 4x; smaller factors downscale the input first (Pre-downscale)."),
    input_path_info="Video file, image, or (with Enable Batch) a folder",
    input_placeholder="C:/path/to/video.mp4 or C:/path/to/image.png",
    gpu_note="RelayVSR uses a single GPU",
    autotune_help=(
        "**Auto Tune (instant, no test runs):** reads your GPU's total and in-use VRAM, the output size and the Wan "
        "model, then keeps as many Wan blocks on the GPU as fit, each with its memory of the last 4 keyframes (about "
        "0.4 GB per block at a 4K output, so 4K needs about 19 GB with every block on the GPU; a 12 GB GPU keeps about "
        "half of them there). RelayVSR streams one frame at a time, so memory does not grow with clip length. It keeps "
        "`Save VRAM (GB)` free, and every run is capped at GPU total - in use - Save VRAM, so it never uses shared "
        "GPU memory."
    ),
    guide_md="""
    #### RelayVSR Guide

    **How it works**
    - A large Wan model writes a reference keyframe every (window - 1) frames; a small Flash decoder turns every
      low-resolution frame into a 4x frame, relaying the detail between the keyframes. One frame streams at a time.
    - The model always upscales **4x**. Smaller factors and `Max Resolution` downscale the input first, so the model
      still runs at 4x.

    **Speed and memory**
    - With every Wan block on the GPU a 4K output needs about 19 GB (INT8). Smaller GPUs keep part of Wan and its
      keyframe memory in RAM and copy it in when Wan runs: the same result, roughly half the speed on a 12 GB GPU.
    - Memory depends on the output size, not on the clip length. For long videos use the Resolution tab chunking
      (scene split or fixed chunk seconds) only when you want resumable chunks.
    """,
    tuned_keys=["wan_gpu_blocks"],
    build_model_column=_build_model_column,
    build_runtime_row=_build_runtime_row,
    build_runtime_extra=_nothing,
    build_sections=_nothing,
    build_output_leading=_nothing,
    output_format_info="RelayVSR writes MP4 for videos and PNG for images (Output tab image format applies).",
    fixed_model_scale=4,
)


def relayvsr_tab(
    preset_manager,
    runner,
    run_logger,
    global_settings: Dict[str, Any],
    shared_state: gr.State,
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path,
):
    """RelayVSR tab (the shared video SR tab with the RelayVSR spec)."""
    return video_sr_tab(RELAYVSR_TAB, preset_manager, runner, run_logger, global_settings, shared_state, base_dir,
                        temp_dir, output_dir)
