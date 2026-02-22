"""
FlashVSR+ Tab - Self-contained modular implementation
Real-time diffusion-based streaming video super-resolution
UPDATED: Now uses Universal Preset System
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any
import html
import threading
import time

from shared.services.flashvsr_service import (
    build_flashvsr_callbacks,
    FLASHVSR_ORDER,
    FLASHVSR_VAE_OPTIONS,
    FLASHVSR_PRECISION_OPTIONS,
    FLASHVSR_ATTENTION_OPTIONS,
)
from shared.fixed_scale_analysis import build_fixed_scale_analysis_update
from shared.models.flashvsr_meta import flashvsr_version_to_ui
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
from shared.gpu_utils import get_gpu_info


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

    def _value(key: str, default=None):
        try:
            idx = FLASHVSR_ORDER.index(key)
            if 0 <= idx < len(values):
                raw = values[idx]
                if raw is None and default is not None:
                    return default
                return raw
        except Exception:
            pass
        return default

    def _vae_mode_note(mode_value: str) -> str:
        mode_norm = str(mode_value or "full").strip().lower()
        if mode_norm == "full":
            return (
                "**Mode note:** In `full` mode, the selected **VAE Model** directly changes the active decoder "
                "(quality/VRAM/speed tradeoff is real here)."
            )
        return (
            "**Mode note:** In `tiny` / `tiny-long`, decoding is TCDecoder-based. "
            "Changing **VAE Model** has limited effect on VRAM/decoder behavior in this backend path."
        )
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
            gpu_hint = f" Detected {cuda_count} CUDA GPU(s) - GPU acceleration available\n FlashVSR uses single GPU only (multi-GPU not supported)"
        else:
            gpu_hint = " CUDA not detected (nvidia-smi unavailable or no NVIDIA GPU) - Processing will use CPU (significantly slower)"
    except Exception as e:
        gpu_hint = f" CUDA detection failed: {str(e)}"
        cuda_available = False

    # Show GPU warning if not available
    if not cuda_available:
        gr.Markdown(
            f'<div style="background: #fff3cd; padding: 12px; border-radius: 8px; border: 1px solid #ffc107;">'
            f'<strong> GPU Acceleration Unavailable</strong><br>'
            f'{gpu_hint}<br><br>'
            f'FlashVSR is designed for CUDA GPUs. CPU mode is possible but very slow.'
            f'</div>',
            elem_classes="warning-text"
        )

    with gr.Row():
        # Left Column: Input & Settings
        with gr.Column(scale=3):
            gr.Markdown("####  Input")

            with gr.Group():
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        input_file = gr.File(
                            label="Upload video or image (optional)",
                            type="filepath",
                            file_types=["video", "image"]
                        )
                        input_path = gr.Textbox(
                            label="Input Path",
                            value=_value("input_path", ""),
                            placeholder="C:/path/to/video.mp4 or C:/path/to/frames/",
                            info="Video file or image sequence folder"
                        )

                    with gr.Column(scale=2):
                        # Backend-only state (kept for preset/schema compatibility).
                        scale = gr.State(
                            value="2" if str(_value("scale", "4")).strip() == "2" else "4"
                        )
                        version = gr.Dropdown(
                            label="Model Version",
                            choices=["1.0", "1.1"],
                            value=flashvsr_version_to_ui(_value("version", "1.1")),
                            info="1.1 = latest recommended model folder (`FlashVSR-v1.1`), 1.0 = legacy (`FlashVSR`)."
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
                                str(_value("vae_model", "Wan2.2"))
                                if str(_value("vae_model", "Wan2.2")) in set(FLASHVSR_VAE_OPTIONS)
                                else "Wan2.2"
                            ),
                            info=(
                                "Choose quality vs VRAM tradeoff: Wan2.1/Wan2.2 = highest quality; "
                                "LightVAE/LightTAE = much lower VRAM and faster."
                            ),
                        )
                        vae_mode_note = gr.Markdown(
                            _vae_mode_note(str(_value("mode", "full"))),
                            elem_classes=["resolution-info"],
                        )

                    with gr.Column(scale=2):
                        input_image_preview = gr.Image(
                            label=" Input Preview (Image)",
                            type="filepath",
                            interactive=False,
                            height=250,
                            visible=False
                        )
                        input_video_preview = gr.Video(
                            label=" Input Preview (Video)",
                            interactive=False,
                            height=250,
                            visible=False
                        )

                input_cache_msg = gr.Markdown("", visible=False)
                sizing_info = gr.Markdown("", visible=False, elem_classes=["resolution-info"])
                input_detection_result = gr.Markdown("", visible=False)

            # vNext sizing controls are placed in the right column to mirror SeedVR2 layout.

            # Processing Settings
            gr.Markdown("#### Core Runtime Settings")
            
            with gr.Group():
                with gr.Row():
                    precision = gr.Dropdown(
                        label="Precision",
                        choices=list(FLASHVSR_PRECISION_OPTIONS),
                        value=(
                            str(_value("precision", _value("dtype", "bf16")))
                            if str(_value("precision", _value("dtype", "bf16"))) in set(FLASHVSR_PRECISION_OPTIONS)
                            else "bf16"
                        ),
                        info="Default is `bf16`. Use `auto` to let backend fall back to fp16 when needed."
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
                    seed = gr.Number(
                        label="Random Seed",
                        value=_value("seed", 0),
                        precision=0,
                        info="Seed for reproducibility."
                    )

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
                gr.Markdown(
                    "**GPU Device:** Controlled globally from the top app header selector. "
                    "This tab no longer has a per-tab GPU override."
                )
                device = gr.State(str(_value("device", "auto") or "auto"))
            
            # Memory Optimization
            gr.Markdown("#### Memory Optimization (Tiling)")
            
            with gr.Group():
                with gr.Row():
                    tiled_vae = gr.Checkbox(
                        label="Enable VAE Tiling",
                        value=bool(_value("tiled_vae", False)),
                        info="Splits VAE processing into tiles to reduce VRAM. Default OFF."
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

                with gr.Row():
                    tile_size = gr.Slider(
                        label="Tile Size",
                        minimum=32, maximum=1024, step=32,
                        value=int(_value("tile_size", 256) or 256),
                        info="Larger tiles are faster but use more VRAM. 256 is a balanced default."
                    )
                    
                    overlap = gr.Slider(
                        label="Tile Overlap",
                        minimum=8, maximum=512, step=8,
                        value=int(_value("overlap", 24) or 24),
                        info="Overlap between tiles to hide seams. Higher overlap is smoother but slower."
                    )
            
            # Quality / I/O Settings
            gr.Markdown("#### Output / I/O Settings")
            
            with gr.Group():
                with gr.Row():
                    color_fix = gr.Checkbox(
                        label="Color Correction",
                        value=bool(_value("color_fix", True)),
                        info="Wavelet color transfer. Recommended ON unless you intentionally want different colors."
                    )
                    start_frame = gr.Number(
                        label="Start Frame",
                        value=int(_value("start_frame", 0) or 0),
                        precision=0,
                        info="0-indexed start frame for partial processing."
                    )
                    end_frame = gr.Number(
                        label="End Frame",
                        value=int(_value("end_frame", -1) or -1),
                        precision=0,
                        info="-1 = process until the end of input."
                    )

                gr.Markdown(
                    "**Video Codec / CRF / FPS Override:** Controlled from `Output > Video Output` "
                    "and automatically applied here."
                )
                # Keep schema compatibility fields without exposing duplicate controls.
                fps_flashvsr = gr.State(float(_value("fps", 0.0) or 0.0))
                codec = gr.State(str(_value("codec", "libx264") or "libx264"))
                crf = gr.State(int(_value("crf", _value("quality", 18)) or 18))

                models_dir = gr.Textbox(
                    label="FlashVSR Models Directory",
                    value=str(_value("models_dir", str(base_dir / "ComfyUI-FlashVSR_Stable" / "models")) or ""),
                    placeholder="G:/.../ComfyUI-FlashVSR_Stable/models",
                    info=(
                        "Folder containing `FlashVSR` and/or `FlashVSR-v1.1` subfolders. "
                        "VAE files can auto-download if missing."
                    ),
                )

                output_override = gr.Textbox(
                    label="Output Override (folder or .mp4 file)",
                    value=_value("output_override", ""),
                    placeholder="Leave empty for auto naming",
                    info="Optional custom output location. A folder saves into that folder. A .mp4 file path renames the final output to that exact file.",
                )

                with gr.Row():
                    output_format = gr.Dropdown(
                        label="Output Format",
                        choices=["mp4"],
                        value=str(_value("output_format", "mp4") or "mp4"),
                        info="FlashVSR Stable CLI outputs MP4.",
                        interactive=False,
                    )
                    save_metadata = gr.Checkbox(
                        label="Save Processing Metadata",
                        value=bool(_value("save_metadata", True)),
                        info="Save run metadata to output folder."
                    )
                    face_restore_after_upscale = gr.Checkbox(
                        label="Apply Face Restoration after upscale",
                        value=bool(_value("face_restore_after_upscale", False)),
                        info="Per-run face restore toggle. Also respects Global Settings face toggle."
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
        
        # Right Column: Output & Controls
        with gr.Column(scale=2):
            status_box = gr.Markdown(value="Ready.")
            progress_indicator = gr.Markdown(value="", visible=True)

            gr.Markdown("####  Output & Actions")

            with gr.Group():
                _upscale_factor_default = _value("upscale_factor", _value("scale", 4))
                try:
                    _upscale_factor_default = float(_upscale_factor_default)
                except Exception:
                    _upscale_factor_default = 4.0
                _upscale_factor_default = 2.0 if _upscale_factor_default <= 3.0 else 4.0

                _max_resolution_default = _value("max_target_resolution", 1920)
                try:
                    _max_resolution_default = int(_max_resolution_default)
                except Exception:
                    _max_resolution_default = 1920
                _max_resolution_default = min(8192, max(0, _max_resolution_default))

                with gr.Row():
                    upscale_factor = gr.Slider(
                        label="FlashVSR Upscale Factor",
                        minimum=2.0,
                        maximum=4.0,
                        step=2.0,
                        value=_upscale_factor_default,
                        info="FlashVSR backend supports only 2x or 4x.",
                        scale=2,
                    )
                    max_target_resolution = gr.Slider(
                        label="Max Resolution (max edge, 0 = no cap)",
                        minimum=0,
                        maximum=8192,
                        step=16,
                        value=_max_resolution_default,
                        info="Caps the LONG side (max(width,height)) of the target. 0 = unlimited.",
                        scale=2,
                    )
                    pre_downscale_then_upscale = gr.Checkbox(
                        label="Pre-downscale then upscale (when capped)",
                        value=bool(_value("pre_downscale_then_upscale", True)),
                        info="If max edge would reduce effective scale, downscale input first so the model still runs at the full Upscale x.",
                        scale=1,
                    )

                use_resolution_tab = gr.Checkbox(
                    label="Use Resolution & Scene Split Tab Settings",
                    value=bool(_value("use_resolution_tab", True)),
                    info="Apply Resolution tab sizing controls. Non-2x/4x global ratios are auto-clamped to 2x or 4x.",
                )

                with gr.Row():
                    upscale_btn = gr.Button(
                        " Start Upscaling",
                        variant="primary",
                        size="lg",
                        elem_classes=["action-btn", "action-btn-upscale"],
                    )
                    preview_btn = gr.Button(
                        "Preview First Frame",
                        size="lg",
                        elem_classes=["action-btn", "action-btn-preview"],
                    )

                with gr.Row():
                    cancel_confirm = gr.Checkbox(
                        label=" Confirm cancel (required for safety)",
                        value=False,
                        info="Enable this checkbox to confirm cancellation of processing",
                        scale=3,
                    )
                    cancel_btn = gr.Button(
                        " Cancel",
                        variant="stop",
                        size="md",
                        min_width=170,
                        scale=1,
                        elem_classes=["action-btn", "action-btn-cancel"],
                    )

            resume_run_dir = gr.Textbox(
                label="Resume Run Folder (chunk/scene resume)",
                value=(
                    _value("resume_run_dir", "")
                    if "resume_run_dir" in FLASHVSR_ORDER and len(values) > FLASHVSR_ORDER.index("resume_run_dir")
                    else ""
                ),
                placeholder="Optional: G:/.../outputs/0019",
                info=(
                    "Optional. When set, chunk/scene processing resumes from the last completed chunk in that folder. "
                    "Use the same settings as the original run to continue remaining chunks. "
                    "Fresh output path overrides are ignored while resume is active."
                ),
            )
            
            with gr.Accordion(" Upscaled Output", open=True):
                output_video = gr.Video(
                    label=" Upscaled Video",
                    interactive=False,
                    visible=False,
                    buttons=["download"],
                )
                output_image = gr.Image(
                    label=" Upscaled Image",
                    interactive=False,
                    visible=False,
                    buttons=["download"],
                )
            
            # Comparison
            image_slider = gr.ImageSlider(
                label=" Comparison",
                interactive=False,
                slider_position=50,
                max_height=1000,
                buttons=["download", "fullscreen"],
                elem_classes=["native-image-comparison-slider"],
            )
            
            video_comparison_html = gr.HTML(
                label=" Video Comparison",
                value="",
                js_on_load=get_video_comparison_js_on_load(),
                visible=False
            )
            
            chunk_status = gr.Markdown("", visible=False)
            chunk_gallery = gr.Gallery(
                label=" Chunk Preview",
                visible=False,
                columns=4,
                rows=2,
                height=220,
                object_fit="contain",
            )
            chunk_preview_video = gr.Video(
                label=" Selected Chunk",
                interactive=False,
                visible=False,
                buttons=["download"],
            )
            batch_gallery = gr.Gallery(
                label=" Batch Results",
                visible=False,
                columns=4,
                rows=2,
                height="auto",
                object_fit="contain",
                buttons=["download"],
            )
            last_processed = gr.Markdown("Batch processing results will appear here.")
            
            # Utility buttons
            with gr.Row():
                open_outputs_btn = gr.Button(
                    " Open Outputs",
                    elem_classes=["action-btn", "action-btn-open"],
                )
                clear_temp_btn = gr.Button(
                    " Clear Temp",
                    elem_classes=["action-btn", "action-btn-clear"],
                )

            with gr.Accordion(" Batch Processing", open=False):
                batch_enable = gr.Checkbox(
                    label="Enable Batch",
                    value=bool(_value("batch_enable", False)),
                    info="Process multiple files"
                )
                batch_input = gr.Textbox(
                    label="Batch Input Folder",
                    value=_value("batch_input_path", ""),
                    placeholder="Folder with videos"
                )
                batch_output = gr.Textbox(
                    label="Batch Output Folder",
                    value=_value("batch_output_path", ""),
                    placeholder="Output directory"
                )
            
            log_box = gr.Textbox(
                label=" Processing Log",
                value="",
                lines=12,
                buttons=["copy"]
            )
            
            # Info
            gr.Markdown("""
            #### FlashVSR Stable Guide

            **VAE selection**
            - `Wan2.1` / `Wan2.2`: best quality, highest VRAM demand.
            - `LightVAE_W2.1`: lower VRAM + faster; strong default for 8-16GB GPUs.
            - `TAE_W2.2` / `LightTAE_HY1.5`: strong temporal consistency and low VRAM variants.

            **Important backend limits**
            - FlashVSR supports only **2x** or **4x** upscale factors.
            - Use `Max Resolution` + `Pre-downscale then upscale` to keep output size safe on limited VRAM.
            - For long videos, combine this tab with Resolution tab chunking (scene split or fixed chunk seconds).
            - `Local Range` is truly `9` or `11` only. `Sparse Ratio` is `1.5-2.0`. `KV Ratio` is `1.0-3.0`.
            - `Sparse Ratio`/`Local Range` matter most on sparse attention backends; on `flash_attention_2`/`sdpa` their effect is limited.
            - `Force Offload After Run` is most useful in long-lived processes; in subprocess-per-job flows it usually has smaller impact.
            - `Enable Stream Decode` works only in tiny modes and automatically disables DiT tiling for that run.
            """)
    
    # Collect inputs
    inputs_list = [
        input_path, output_override, output_format, scale, version, mode,
        vae_model, precision, attention_mode,
        tiled_vae, tiled_dit, tile_size, overlap, unload_dit, stream_decode,
        sparse_ratio, kv_ratio, local_range, frame_chunk_size,
        keep_models_on_cpu, force_offload, enable_debug,
        color_fix, seed, device, fps_flashvsr, codec, crf, start_frame, end_frame, models_dir,
        save_metadata, face_restore_after_upscale, batch_enable, batch_input, batch_output,
        use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale,
        resume_run_dir,
    ]

    # Development validation: inputs_list must stay aligned with FLASHVSR_ORDER
    if len(inputs_list) != len(FLASHVSR_ORDER):
        import logging
        logging.getLogger("FlashVSRTab").error(
            f"ERROR: inputs_list ({len(inputs_list)}) != FLASHVSR_ORDER ({len(FLASHVSR_ORDER)})"
        )
    
    # Wire up events
    def _analysis_progress_note(state: Dict[str, Any], pct: int) -> str:
        seed_controls = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
        scene_mode = bool(seed_controls.get("auto_chunk", True)) and bool(seed_controls.get("auto_detect_scenes", True))
        if pct < 15:
            return "Reading media metadata..."
        if pct < 35:
            return "Computing resize target..."
        if pct < 70:
            return "Scanning scenes for chunk stats..." if scene_mode else "Building runtime summary..."
        if pct < 95:
            return "Preparing analysis panel..."
        return "Finalizing analysis..."

    def _analysis_banner_html(state: Dict[str, Any], progress_pct: int, progress_note: str = "") -> str:
        safe_pct = max(0, min(100, int(progress_pct)))
        seed_controls = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
        scene_mode = bool(seed_controls.get("auto_chunk", True)) and bool(seed_controls.get("auto_detect_scenes", True))
        title = "Analyzing input (resolution + scene detection)" if scene_mode else "Analyzing input"
        sub = f"{safe_pct}%"
        if progress_note:
            sub = f"{sub}<br>{html.escape(str(progress_note))}"
        return (
            '<div class="processing-banner">'
            '<div class="processing-spinner"></div>'
            '<div class="processing-col">'
            f'<div class="processing-text">{html.escape(title)}</div>'
            f'<div class="processing-sub">{sub}</div>'
            "</div></div>"
        )

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

    def _build_sizing_info(path_val, model_scale_val, use_global, local_scale_x, local_max_edge, local_pre_down, state):
        ms = int(model_scale_val or 4)
        return build_fixed_scale_analysis_update(
            input_path_val=path_val,
            model_scale=ms,
            use_global=bool(use_global),
            local_scale_x=float(local_scale_x or 4.0),
            local_max_edge=int(local_max_edge or 0),
            local_pre_down=bool(local_pre_down),
            state=state,
            model_label="FlashVSR Stable",
            runtime_label=f"FlashVSR Stable pipeline (fixed {ms}x pass)",
            auto_scene_scan=True,
        )

    def _run_analysis_payload(path_val, scale_val, use_global, scale_x, max_edge, pre_down, state):
        det = _build_input_detection_md(path_val or "")
        info = _build_sizing_info(path_val or "", int(scale_val), bool(use_global), scale_x, max_edge, pre_down, state)
        img_prev, vid_prev = preview_updates(path_val)
        return img_prev, vid_prev, det, info, state

    def _iter_analysis_with_progress(path_val, scale_val, use_global, scale_x, max_edge, pre_down, state):
        result: Dict[str, Any] = {}

        def _worker():
            try:
                result["payload"] = _run_analysis_payload(path_val, scale_val, use_global, scale_x, max_edge, pre_down, state)
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
                yield "progress", pct, _analysis_progress_note(state, pct)
                last_emit = now
            time.sleep(0.05)

        worker.join()
        if "error" in result:
            raise result["error"]
        yield "progress", 100, _analysis_progress_note(state, 100)
        yield "result", result.get("payload"), ""

    def cache_input_upload(val, scale_val, use_global, scale_x, max_edge, pre_down, state):
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
            gr.update(value=_analysis_banner_html(state, 0, _analysis_progress_note(state, 0)), visible=True),
            state,
        )

        try:
            for event_type, payload_a, _ in _iter_analysis_with_progress(
                val, scale_val, use_global, scale_x, max_edge, pre_down, state
            ):
                if event_type == "progress":
                    pct = int(payload_a)
                    yield (
                        val or "",
                        gr.update(value="", visible=False),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.update(value=_analysis_banner_html(state, pct, _analysis_progress_note(state, pct)), visible=True),
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

    input_file.upload(
        fn=cache_input_upload,
        inputs=[input_file, scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state]
    )

    # When upload is cleared, also clear the textbox path and hide dependent panels.
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

    input_file.change(
        fn=clear_input_path_on_upload_clear,
        inputs=[input_file, shared_state],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )

    def cache_input_path(path_val, scale_val, use_global, scale_x, max_edge, pre_down, state):
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
            gr.update(value=_analysis_banner_html(state, 0, _analysis_progress_note(state, 0)), visible=True),
            state,
        )

        try:
            for event_type, payload_a, _ in _iter_analysis_with_progress(
                path_val, scale_val, use_global, scale_x, max_edge, pre_down, state
            ):
                if event_type == "progress":
                    pct = int(payload_a)
                    yield (
                        gr.update(value="", visible=False),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.update(value=_analysis_banner_html(state, pct, _analysis_progress_note(state, pct)), visible=True),
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

    input_path.change(
        fn=cache_input_path,
        inputs=[input_path, scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )

    for comp in [scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale]:
        comp.change(
            fn=lambda p, s, ug, sx, me, pd, st: (_build_sizing_info(p, int(s), bool(ug), sx, me, pd, st), st),
            inputs=[input_path, scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale, shared_state],
            outputs=[sizing_info, shared_state],
            trigger_mode="always_last",
        )

    def _sync_scale_from_upscale(local_x):
        try:
            return gr.update(value=("2" if float(local_x) <= 3.0 else "4"))
        except Exception:
            return gr.update(value="4")

    upscale_factor.change(
        fn=_sync_scale_from_upscale,
        inputs=[upscale_factor],
        outputs=[scale],
        queue=False,
        show_progress="hidden",
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

    def _extract_update_value(update_obj):
        try:
            if isinstance(update_obj, dict):
                return update_obj.get("value")
        except Exception:
            pass
        return None

    def _batch_gallery_update_from_state(state):
        outputs = (state or {}).get("seed_controls", {}).get("flashvsr_batch_outputs", [])
        if not isinstance(outputs, list):
            outputs = []
        outputs = [str(p) for p in outputs if p and Path(str(p)).exists()]
        return gr.update(value=outputs, visible=bool(outputs))

    def _last_processed_text(state, vid_upd, img_upd) -> str:
        outputs = (state or {}).get("seed_controls", {}).get("flashvsr_batch_outputs", [])
        if isinstance(outputs, list) and outputs:
            last_out = str(outputs[-1])
            return f"Batch results: {len(outputs)} item(s). Last output: {Path(last_out).name}"

        single = _extract_update_value(img_upd) or _extract_update_value(vid_upd)
        if single:
            return f"Output: {single}"
        return "Batch processing results will appear here."

    def _expand_service_payload(payload, live_state):
        merged = merge_payload_state(payload, live_state)
        if not isinstance(merged, tuple) or len(merged) < 7:
            safe_state = live_state if isinstance(live_state, dict) else {}
            return (
                gr.update(value="ERROR: Invalid FlashVSR+ payload"),
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
        status_lc = status_text.lower()

        terminal_tokens = (
            "complete",
            "failed",
            "error",
            "critical",
            "cancel",
            "no result",
            "out of vram",
            "oom",
            "missing",
            "insufficient",
        )
        is_terminal = any(tok in status_lc for tok in terminal_tokens)
        if status_text and not is_terminal:
            subtitle = "Processing..."
            if isinstance(logs, str):
                for line in reversed(logs.splitlines()):
                    line = line.strip()
                    if line:
                        subtitle = line
                        break
            progress_update = _queue_status_indicator(status_text, subtitle, spinning=True)
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
                yield _expand_service_payload(payload, live_state)
        finally:
            if acquired_slot:
                queue_manager.complete(ticket.job_id)
            else:
                queue_manager.cancel_waiting([ticket.job_id])

    def run_preview_with_snapshot(*args, progress=gr.Progress()):
        live_state = args[-1] if (args and isinstance(args[-1], dict)) else {}
        queued_state = snapshot_queue_state(live_state)
        queued_global_settings = snapshot_global_settings(global_settings)
        for payload in service["run_action"](
            args[0],
            *args[1:-1],
            preview_only=True,
            state=queued_state,
            progress=progress,
            global_settings_snapshot=queued_global_settings,
        ):
            yield _expand_service_payload(payload, live_state)

    # Main processing
    run_evt = upscale_btn.click(
        fn=run_upscale_with_queue,
        inputs=[input_file] + inputs_list + [shared_state],
        outputs=[
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
        ],
    )
    preview_evt.then(
        fn=refresh_chunk_preview_ui,
        inputs=[shared_state],
        outputs=[chunk_status, chunk_gallery, chunk_preview_video],
    )

    shared_state.change(
        fn=refresh_chunk_preview_ui,
        inputs=[shared_state],
        outputs=[chunk_status, chunk_gallery, chunk_preview_video],
        queue=False,
        show_progress="hidden",
    )

    mode.change(
        fn=lambda m: gr.update(value=_vae_mode_note(m)),
        inputs=[mode],
        outputs=[vae_mode_note],
        queue=False,
        show_progress="hidden",
    )

    def _cancel_with_confirmation_reset(ok):
        if ok:
            status_upd, log_msg = service["cancel_action"]()
            return status_upd, log_msg, gr.update(value=False)
        return gr.update(value="WARNING: Enable 'Confirm cancel' to stop."), "", gr.update(value=False)

    cancel_btn.click(
        fn=_cancel_with_confirmation_reset,
        inputs=[cancel_confirm],
        outputs=[status_box, log_box, cancel_confirm]
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

