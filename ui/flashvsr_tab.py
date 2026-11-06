"""
FlashVSR tab: the shared video SR tab (ui/video_sr_tab.py) with FlashVSR's own controls - model / pipeline mode /
VAE, precision, attention, chunking, sparse / KV ratios, offloading, DiT / VAE tiling with the live tile count,
diffusion controls, color correction and the models folder.
"""

import math
from pathlib import Path
from typing import Any, Dict

import gradio as gr

from shared.models.flashvsr_meta import FLASHVSR_VERSION_CHOICES, flashvsr_version_to_ui
from shared.path_utils import get_media_dimensions, normalize_path
from shared.resolution_calculator import estimate_fixed_scale_upscale_plan_from_dims
from shared.services.flashvsr_service import (
    FLASHVSR_ATTENTION_OPTIONS,
    FLASHVSR_ORDER,
    FLASHVSR_PRECISION_OPTIONS,
    FLASHVSR_VAE_OPTIONS,
    build_flashvsr_callbacks,
    canonical_flashvsr_scale,
)
from ui.model_tab_common import resolve_shared_upscale_factor
from ui.path_picker import path_textbox
from ui.video_sr_tab import TabContext, VideoSRTabSpec, video_sr_tab


def resolve_flashvsr_effective_scale(
    *,
    scale_state_val,
    use_global: bool,
    local_upscale_factor,
    state: Dict[str, Any] | None,
) -> int:
    """
    Resolve FlashVSR effective scale (2x/4x) from UI + shared-state inputs.
    """
    default_scale = canonical_flashvsr_scale(scale_value=scale_state_val, default=4)
    global_scale = resolve_shared_upscale_factor(state if bool(use_global) else None)
    return canonical_flashvsr_scale(
        scale_value=scale_state_val,
        upscale_factor_value=(global_scale if global_scale is not None else local_upscale_factor),
        default=default_scale,
    )

def _vae_mode_note(mode_value: str) -> str:
    mode_norm = str(mode_value or "full").strip().lower()
    if mode_norm == "full":
        return "**Decoder:** `full` mode decodes with the selected **VAE Model**."
    return (
        "**Decoder:** `tiny` / `tiny-long` decode with FlashVSR's TC decoder, so **VAE Model** "
        "applies only to `full` mode."
    )

def _compute_dit_tile_count_text(
    path_val,
    tile_size_val,
    overlap_val,
    scale_state_val,
    use_global,
    local_upscale_factor,
    max_edge_val,
    pre_downscale_val,
    state,
    tiled_dit_val,
    mode_val,
    stream_decode_val,
) -> tuple[str, bool]:
    if not bool(tiled_dit_val):
        return "DiT tile count: disabled (`Enable DiT Tiling` is OFF).", True

    mode_norm = str(mode_val or "full").strip().lower()
    if bool(stream_decode_val) and mode_norm in {"tiny", "tiny-long"}:
        return "DiT tile count: disabled at runtime (`Stream Decode` is ON in tiny modes).", True

    try:
        tile_size = int(float(tile_size_val))
    except Exception:
        tile_size = 256
    tile_size = max(32, min(1024, tile_size))

    try:
        overlap = int(float(overlap_val))
    except Exception:
        overlap = 24
    overlap = max(8, min(512, overlap))

    notes = []
    if tile_size < 128:
        tile_size = 128
        notes.append("tile_size auto-raised to 128")
    if overlap >= tile_size:
        overlap = max(8, tile_size - 8)
        notes.append("overlap auto-corrected")
    stride = max(1, int(tile_size - overlap))

    normalized_path = normalize_path(path_val) if path_val else None
    if not normalized_path:
        return "**DiT tiles:** set an input path to estimate tile count.", True

    dims = get_media_dimensions(normalized_path)
    if not dims:
        return "**DiT tiles:** unable to read input dimensions yet.", True

    src_w, src_h = int(dims[0]), int(dims[1])
    tiling_w, tiling_h = src_w, src_h

    try:
        resolved_scale = resolve_flashvsr_effective_scale(
            scale_state_val=scale_state_val,
            use_global=bool(use_global),
            local_upscale_factor=local_upscale_factor,
            state=state if isinstance(state, dict) else None,
        )
        model_scale = int(resolved_scale or 4)
        max_edge = int(max_edge_val or 0)

        plan = estimate_fixed_scale_upscale_plan_from_dims(
            src_w,
            src_h,
            requested_scale=float(model_scale),
            model_scale=int(model_scale),
            max_edge=max_edge,
            force_pre_downscale=bool(pre_downscale_val),
        )
        tiling_w = int(plan.preprocess_width or src_w)
        tiling_h = int(plan.preprocess_height or src_h)
    except Exception:
        tiling_w, tiling_h = src_w, src_h

    rows = max(1, int(math.ceil((tiling_h - overlap) / float(stride))))
    cols = max(1, int(math.ceil((tiling_w - overlap) / float(stride))))
    total = int(rows * cols)

    preprocess_note = ""
    if tiling_w != src_w or tiling_h != src_h:
        preprocess_note = f"; preprocessed from `{src_w}x{src_h}`"

    guardrail_note = ""
    if notes:
        guardrail_note = f" ({'; '.join(notes)})"

    text = (
        f"**DiT tiles:** `{total}` (`{cols} x {rows}`)  \n"
        f"Input for tiling: `{tiling_w}x{tiling_h}`; stride: `{stride}`{preprocess_note}{guardrail_note}."
    )
    return text, True

def _tile_count_info_update(
    path_val,
    tile_size_val,
    overlap_val,
    scale_state_val,
    use_global,
    local_upscale_factor,
    max_edge_val,
    pre_downscale_val,
    state,
    tiled_dit_val,
    mode_val,
    stream_decode_val,
):
    text, visible = _compute_dit_tile_count_text(
        path_val,
        tile_size_val,
        overlap_val,
        scale_state_val,
        use_global,
        local_upscale_factor,
        max_edge_val,
        pre_downscale_val,
        state,
        tiled_dit_val,
        mode_val,
        stream_decode_val,
    )
    return gr.update(value=text, visible=visible)


def _build_model_column(ctx: TabContext) -> None:
    _value = ctx.value
    version = gr.Dropdown(
        label="Model",
        choices=list(FLASHVSR_VERSION_CHOICES),
        value=flashvsr_version_to_ui(_value("version", "1.1")),
        info=(
            "FlashVSR v1.1 (`FlashVSR-v1.1`, v1.0 = legacy). FlashVSR+ Turbo / Base = the official "
            "FlashVSR+ release: one stream over the whole clip (no chunk restarts), reward-tuned "
            "detail; Turbo uses hybrid local attention (less VRAM at high resolution, the better "
            "default); Base needs more VRAM. FlashVSR+ ignores Pipeline Mode / VAE Model, decodes with "
            "the TC decoder and follows the Precision setting (INT8 ConvRot: slightly faster and "
            "0.5-1 GB less VRAM than BF16 at the same quality; its 1.7 GB INT8 cache downloads on first use)."
        ),
    )
    mode = gr.Dropdown(
        label="Pipeline Mode",
        choices=["tiny", "tiny-long", "full"],
        value=(
            str(_value("mode", "full"))
            if str(_value("mode", "full")) in {"tiny", "tiny-long", "full"}
            else "full"
        ),
        info=(
            "`tiny` = best speed/quality balance. `tiny-long` = lowest VRAM for long clips. "
            "`full` = maximum quality (default), but highest VRAM usage."
        )
    )
    vae_model = gr.Dropdown(
        label="VAE Model",
        choices=list(FLASHVSR_VAE_OPTIONS),
        value=(
            str(_value("vae_model", "LightVAE_W2.1"))
            if str(_value("vae_model", "LightVAE_W2.1")) in set(FLASHVSR_VAE_OPTIONS)
            else "LightVAE_W2.1"
        ),
        info=(
            "Full mode decoder. LightVAE_W2.1 (default) = closest to the original, least flicker, "
            "fastest. Wan2.1 = full Wan VAE: crisper but invents more detail, more VRAM."
        ),
        interactive=str(_value("mode", "full")).strip().lower() == "full",
    )
    vae_mode_note = gr.Markdown(
        _vae_mode_note(str(_value("mode", "full"))),
        elem_classes=["resolution-info"],
    )
    with gr.Row():
        temporal_compression = gr.Dropdown(
            label="FlashVSR+ Temporal Compression",
            choices=[4, 2, 1],
            value=int(_value("temporal_compression", 4) or 4) if int(_value("temporal_compression", 4) or 4) in (1, 2, 4) else 4,
            info="FlashVSR+ only. Input frames per latent frame: 4 = default; 2 or 1 keep more detail in large motion (about 2x / 4x the work).",
        )
        decode_every = gr.Slider(
            label="FlashVSR+ Decode Every (latent frames)",
            minimum=4,
            maximum=64,
            step=4,
            value=int(_value("decode_every", 16) or 16),
            info="FlashVSR+ only. The clip streams through the model once; every N latent frames (4N video frames) are decoded and written, so RAM stays bounded on long clips.",
        )
    ctx.c.update(version=version, mode=mode, vae_model=vae_model, vae_mode_note=vae_mode_note,
                 temporal_compression=temporal_compression, decode_every=decode_every)


def _build_runtime_row(ctx: TabContext) -> None:
    _value = ctx.value
    precision = gr.Dropdown(
        label="Precision",
        choices=list(FLASHVSR_PRECISION_OPTIONS),
        value=(
            str(_value("precision", _value("dtype", "int8_convrot")))
            if str(_value("precision", _value("dtype", "int8_convrot"))) in set(FLASHVSR_PRECISION_OPTIONS)
            else "int8_convrot"
        ),
        info="`int8_convrot` is the default on every platform: same quality as BF16, about 1 GB less VRAM and 3-6% faster (FlashVSR spends most of its time in attention and decoding, not in the layers INT8 speeds up). It downloads a prebuilt Hadamard-rotated DiT cache on first use (1.7 GB); local generation remains the fallback."
    )

    attention_mode = gr.Dropdown(
        label="Attention Backend",
        choices=list(FLASHVSR_ATTENTION_OPTIONS),
        value=(
            str(_value("attention_mode", _value("attention", "flash_attention_2")))
            if str(_value("attention_mode", _value("attention", "flash_attention_2"))) in set(FLASHVSR_ATTENTION_OPTIONS)
            else "flash_attention_2"
        ),
        info=(
            "Default `flash_attention_2` (dense, fast on modern NVIDIA). "
            "`Local Range` / `Sparse Ratio` / `KV Ratio` mainly affect sparse backends "
            "(`sparse_sage_attention`, `block_sparse_attention`)."
        )
    )
    ctx.c.update(precision=precision, attention_mode=attention_mode)


def _build_runtime_extra(ctx: TabContext) -> None:
    _value = ctx.value
    with gr.Row():
        frame_chunk_size = gr.Slider(
            label="Frame Chunk Size",
            minimum=0,
            maximum=10000,
            step=1,
            value=int(_value("frame_chunk_size", 0) or 0),
            info=(
                "0 = process all frames at once. Larger chunks are usually faster and improve temporal quality, "
                "but require more VRAM. Smaller chunks use less VRAM on long clips, but may be slower."
            ),
        )
        local_range = gr.Dropdown(
            label="Local Range",
            choices=[9, 11],
            value=9 if str(_value("local_range", 11)).strip() == "9" else 11,
            info=(
                "Backend-supported values: `9` or `11` only. Sparse local window size: "
                "`9` = tighter/sharper and slightly lighter, `11` = wider/stabler motion and slightly heavier. "
                "Limited impact on dense backends."
            ),
        )

    with gr.Row():
        sparse_ratio = gr.Slider(
            label="Sparse Ratio",
            minimum=1.5,
            maximum=2.0,
            step=0.1,
            value=float(_value("sparse_ratio", 2.0) or 2.0),
            info=(
                "True range in this build: `1.5-2.0` (not `1-3`). Controls sparse top-k density: "
                "lower = less compute/VRAM and faster, higher = more detail/stability and slower. "
                "Primarily affects sparse backends."
            ),
        )
        kv_ratio = gr.Slider(
            label="KV Ratio",
            minimum=1.0,
            maximum=3.0,
            step=0.1,
            value=float(_value("kv_ratio", 3.0) or 3.0),
            info=(
                "True range in this build: `1.0-3.0` (not `1-5`). Controls temporal KV cache length. "
                "Higher keeps more history (better consistency, more VRAM/time). Internally cast to int "
                "(effective levels are roughly `1`, `2`, `3`)."
            ),
        )

    with gr.Row():
        keep_models_on_cpu = gr.Checkbox(
            label="Keep Models on CPU",
            value=bool(_value("keep_models_on_cpu", True)),
            info=(
                "Keeps idle model parts in system RAM and moves them to GPU only when needed. "
                "Reduces peak VRAM but adds transfer overhead (can lower FPS). Usually good for low-VRAM "
                "or multi-job stability."
            ),
        )
        force_offload = gr.Checkbox(
            label="Force Offload After Run",
            value=bool(_value("force_offload", True)),
            info=(
                "After processing, explicitly offloads models from VRAM to CPU RAM. "
                "Effective only when `Keep Models on CPU` is ON. Useful for long-lived app sessions; "
                "in one-job-per-subprocess usage, benefit is usually smaller."
            ),
        )
        enable_debug = gr.Checkbox(
            label="Enable Debug Logging",
            value=bool(_value("enable_debug", False)),
            info="Verbose backend logging (timings, memory behavior, diagnostics).",
        )
    ctx.c.update(frame_chunk_size=frame_chunk_size, local_range=local_range, sparse_ratio=sparse_ratio,
                 kv_ratio=kv_ratio, keep_models_on_cpu=keep_models_on_cpu, force_offload=force_offload,
                 enable_debug=enable_debug)


def _build_sections(ctx: TabContext) -> None:
    _value = ctx.value
    shared_state = ctx.shared_state
    gr.Markdown("#### Memory Optimization (Tiling)")

    with gr.Group():
        with gr.Row():
            tiled_vae = gr.Checkbox(
                label="Enable VAE Tiling",
                value=bool(_value("tiled_vae", False)),
                info="Decodes in tiles: in full mode the VAE (lower VRAM), in tiny modes the TC decoder."
            )

            tiled_dit = gr.Checkbox(
                label="Enable DiT Tiling",
                value=bool(_value("tiled_dit", True)),
                info="Splits diffusion transformer inference into tiles. Default ON."
            )

            unload_dit = gr.Checkbox(
                label="Unload DiT Before Decoding",
                value=bool(_value("unload_dit", True)),
                info="Releases DiT from VRAM before decoder step (TCDecoder in tiny modes, VAE in full mode). Default ON."
            )

        with gr.Row():
            stream_decode = gr.Checkbox(
                label="Enable Stream Decode (Tiny/Tiny-Long)",
                value=bool(_value("stream_decode", False)),
                info=(
                    "Forces streaming decode path in tiny modes by disabling DiT tiling at runtime. "
                    "Can improve responsiveness and lower host RAM, but typically increases peak VRAM."
                ),
            )

        _initial_tile_count_text, _initial_tile_count_visible = _compute_dit_tile_count_text(
            _value("input_path", ""),
            _value("tile_size", 256),
            _value("overlap", 24),
            _value("scale", "4"),
            _value("use_resolution_tab", True),
            _value("upscale_factor", _value("scale", 4)),
            _value("max_target_resolution", 1920),
            _value("pre_downscale_then_upscale", True),
            shared_state.value if isinstance(shared_state.value, dict) else {},
            _value("tiled_dit", True),
            _value("mode", "full"),
            _value("stream_decode", False),
        )

        with gr.Row():
            with gr.Column():
                tile_size = gr.Slider(
                    label="Tile Size",
                    minimum=32, maximum=1024, step=32,
                    value=int(_value("tile_size", 256) or 256),
                    info="Larger tiles are faster but use more VRAM. 256 is a balanced default."
                )
                dit_tile_count = gr.Markdown(
                    value=_initial_tile_count_text,
                    visible=bool(_initial_tile_count_visible),
                    elem_classes=["resolution-info"],
                )

            with gr.Column():
                overlap = gr.Slider(
                    label="Tile Overlap",
                    minimum=8, maximum=512, step=8,
                    value=int(_value("overlap", 24) or 24),
                    info="Overlap between tiles to hide seams. Higher overlap is smoother but slower."
                )

    gr.Markdown("#### Diffusion Controls (Experimental)")
    with gr.Group():
        with gr.Row():
            cfg_scale = gr.Slider(
                label="CFG Scale",
                minimum=0.5,
                maximum=2.0,
                step=0.05,
                value=float(_value("cfg_scale", 1.0) or 1.0),
                info=(
                    "Experimental latent update gain. `1.0` = default behavior. "
                    "Higher values push stronger hallucination/detail; lower values are more conservative."
                ),
            )
            denoise_amount = gr.Slider(
                label="Denoise Amount",
                minimum=0.5,
                maximum=2.0,
                step=0.05,
                value=float(_value("denoise_amount", 1.0) or 1.0),
                info=(
                    "Experimental one-step denoise strength. `1.0` = default. "
                    "Higher can increase diffusion impact but may introduce artifacts/flicker."
                ),
            )
    ctx.c.update(tiled_vae=tiled_vae, tiled_dit=tiled_dit, unload_dit=unload_dit, stream_decode=stream_decode,
                 tile_size=tile_size, dit_tile_count=dit_tile_count, overlap=overlap, cfg_scale=cfg_scale,
                 denoise_amount=denoise_amount)


def _build_output_leading(ctx: TabContext) -> None:
    _value = ctx.value
    color_fix = gr.Checkbox(
        label="Color Correction",
        value=bool(_value("color_fix", True)),
        info="Wavelet color transfer. Recommended ON unless you intentionally want different colors."
    )
    ctx.c["color_fix"] = color_fix


def _build_models_dir(ctx: TabContext) -> None:
    _value = ctx.value
    base_dir = ctx.base_dir
    models_dir = path_textbox("folder", 
        label="FlashVSR Models Directory",
        value=str(_value("models_dir", str(base_dir / "ComfyUI-FlashVSR_Stable" / "models")) or ""),
        placeholder="G:/.../ComfyUI-FlashVSR_Stable/models",
        info=(
            "Folder containing `FlashVSR` and/or `FlashVSR-v1.1` subfolders. "
            "VAE files can auto-download if missing."
        ),
    )
    ctx.c["models_dir"] = models_dir


def _tile_count_inputs(c: Dict[str, Any]):
    return [c["input_path"], c["tile_size"], c["overlap"], c["scale"], c["use_resolution_tab"], c["upscale_factor"],
            c["max_target_resolution"], c["pre_downscale_then_upscale"], c["shared_state"], c["tiled_dit"], c["mode"],
            c["stream_decode"]]


def _after_sizing(ctx: TabContext):
    ctx.c["shared_state"] = ctx.shared_state
    return _tile_count_info_update, _tile_count_inputs(ctx.c), [ctx.c["dit_tile_count"]]


def _wire_extra(ctx: TabContext) -> None:
    c = ctx.c
    c["shared_state"] = ctx.shared_state
    c["mode"].change(
        fn=lambda m: (gr.update(value=_vae_mode_note(m)),
                      gr.update(interactive=str(m or "").strip().lower() == "full")),
        inputs=[c["mode"]],
        outputs=[c["vae_mode_note"], c["vae_model"]],
        queue=False,
        show_progress="hidden",
    )
    for key in ("mode", "tile_size", "overlap", "tiled_dit", "stream_decode"):
        c[key].change(fn=_tile_count_info_update, inputs=_tile_count_inputs(c), outputs=[c["dit_tile_count"]],
                      queue=False, show_progress="hidden", trigger_mode="always_last")


FLASHVSR_TAB = VideoSRTabSpec(
    key="flashvsr",
    label="FlashVSR",
    build_service=build_flashvsr_callbacks,
    order=FLASHVSR_ORDER,
    canonical_scale=canonical_flashvsr_scale,
    model_label="FlashVSR Stable",
    runtime_label=lambda ms: f"FlashVSR Stable pipeline (fixed {ms}x pass)",
    upscale_slider=dict(label="FlashVSR Upscale Factor", minimum=2.0, maximum=4.0, step=2.0,
                        info="FlashVSR backend supports only 2x or 4x."),
    input_path_info="Video file or image sequence folder",
    input_placeholder="C:/path/to/video.mp4 or C:/path/to/frames/",
    gpu_note="FlashVSR uses single GPU only (multi-GPU not supported)",
    autotune_help=(
    "**Auto Tune (instant, no test runs):** reads your GPU's total and in-use VRAM, the input's size and "
    "length, the output size, version and mode, then sets the longest **Frame Chunk Size** that fits "
    "(the whole clip in one pass when possible: every chunk restarts FlashVSR's temporal memory), "
    "**DiT tiling OFF** when the full frame fits (otherwise the fewest tiles, with a wider overlap), "
    "**KV Ratio 3** and no CPU offloading for speed. It keeps `Save VRAM (GB)` free. Every run is "
    "capped at GPU total - in use - Save VRAM, so it never uses shared GPU memory."
),
    guide_md="""
    #### FlashVSR Stable Guide

    **Decoder**
    - `full` mode decodes with the selected VAE: `Wan2.1` (full Wan VAE) or `LightVAE_W2.1` (quarter width, less VRAM).
    - `tiny` / `tiny-long` decode with FlashVSR's TC decoder; the VAE choice does not apply there.

    **Important backend limits**
    - FlashVSR supports only **2x** or **4x** upscale factors.
    - Use `Max Resolution` + `Pre-downscale then upscale` to keep output size safe on limited VRAM.
    - For long videos, combine this tab with Resolution tab chunking (scene split or fixed chunk seconds).
    - `Local Range` is truly `9` or `11` only. `Sparse Ratio` is `1.5-2.0`. `KV Ratio` is `1.0-3.0`.
    - `Sparse Ratio`/`Local Range` matter most on sparse attention backends; on `flash_attention_2`/`sdpa` their effect is limited.
    - `CFG Scale` and `Denoise Amount` are experimental one-step multipliers; keep both near `1.0` for baseline behavior.
    - `Force Offload After Run` is most useful in long-lived processes; in subprocess-per-job flows it usually has smaller impact.
    - `Enable Stream Decode` works only in tiny modes and automatically disables DiT tiling for that run.
    """,
    tuned_keys=["frame_chunk_size", "tiled_dit", "tile_size", "overlap", "kv_ratio", "keep_models_on_cpu",
                "force_offload", "unload_dit"],
    build_model_column=_build_model_column,
    build_runtime_row=_build_runtime_row,
    build_runtime_extra=_build_runtime_extra,
    build_sections=_build_sections,
    build_output_leading=_build_output_leading,
    build_models_dir=_build_models_dir,
    output_format_info="FlashVSR Stable CLI outputs MP4.",
    after_sizing=_after_sizing,
    wire_extra=_wire_extra,
)


def flashvsr_tab(
    preset_manager,
    runner,
    run_logger,
    global_settings: Dict[str, Any],
    shared_state: gr.State,
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path,
):
    """FlashVSR tab (the shared video SR tab with the FlashVSR spec)."""
    return video_sr_tab(FLASHVSR_TAB, preset_manager, runner, run_logger, global_settings, shared_state, base_dir,
                        temp_dir, output_dir)
