"""
Shared tab of the video SR engines (FlashVSR, RelayVSR): input with upload / path / previews and sizing analysis,
seed, Auto Transfer, Auto Tune + Save VRAM, Start / End Frame, output settings, universal presets, Output & Actions
(scale, Max Resolution, pre-downscale, Start / Preview / Cancel, resume, outputs, image / video comparison, chunk
preview, batch results, open / clear folders), batch processing, queue, log. An engine supplies a VideoSRTabSpec with
its service and the builders of its own controls; everything else is this one implementation.
"""

import html
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import gradio as gr

from shared.fixed_scale_analysis import build_fixed_scale_analysis_update
from shared.path_utils import normalize_path
from shared.processing_queue import get_processing_queue_manager, resolve_queue_gpu_resources
from shared.queue_state import merge_payload_state, snapshot_global_settings, snapshot_queue_state
from shared.video_comparison_slider import get_video_comparison_js_on_load
from ui.media_preview import preview_updates
from ui.model_tab_common import (
    TERMINAL_STATUS_TOKENS,
    build_input_detection_md as _build_input_detection_md,
    compact_single_line as _compact_single_line,
    extract_update_value as _extract_update_value,
    log_tail_line as _log_tail_line,
    resolve_shared_upscale_factor,
    sync_signature as _sync_signature,
)
from ui.path_picker import path_textbox
from ui.shared_components import (
    autotune_modal_dismiss_js,
    autotune_vram_controls,
    warn_cancel_confirmation,
    wire_auto_tune,
    wire_auto_tune_on_run,
    with_pre_run_output_signature,
)
from ui.universal_preset_section import universal_preset_section, wire_universal_preset_events


@dataclass
class TabContext:
    """What engine builders get: initial values, the component registry and the app objects."""

    value: Callable[..., Any]
    c: Dict[str, Any]
    base_dir: Path
    shared_state: Any
    spec: "VideoSRTabSpec"


@dataclass
class VideoSRTabSpec:
    key: str                                   # "flashvsr", "relayvsr"
    label: str                                 # user-facing engine name
    build_service: Callable[..., Dict[str, Any]]
    order: List[str]
    canonical_scale: Callable[..., int]
    model_label: str                           # sizing analysis label
    runtime_label: Callable[[int], str]        # sizing analysis runtime line for a model scale
    upscale_slider: Dict[str, Any]             # label / minimum / maximum / step / info of the Upscale slider
    input_path_info: str
    input_placeholder: str
    gpu_note: str                              # appended to the GPU detection hint
    autotune_help: str                         # markdown under the Auto Tune controls
    guide_md: str                              # markdown at the bottom of the right column
    tuned_keys: List[str]                      # components Auto Tune may update
    build_model_column: Callable[[TabContext], None]
    build_runtime_row: Callable[[TabContext], None]
    build_runtime_extra: Callable[[TabContext], None]
    build_sections: Callable[[TabContext], None]
    build_output_leading: Callable[[TabContext], None]
    build_models_dir: Optional[Callable[[TabContext], None]] = None
    output_format_choices: List[str] = field(default_factory=lambda: ["mp4"])
    output_format_info: str = "The engine writes MP4."
    output_override_label: str = "Output Override (folder or .mp4 file)"
    after_sizing: Optional[Callable[[TabContext], Any]] = None   # -> (fn, inputs, outputs) chained after sizing events
    fixed_model_scale: int = 0                 # see VideoSREngine.fixed_model_scale (RelayVSR: 4)
    wire_extra: Optional[Callable[[TabContext], None]] = None


def video_sr_tab(
    spec: VideoSRTabSpec,
    preset_manager,
    runner,
    run_logger,
    global_settings: Dict[str, Any],
    shared_state: gr.State,
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path,
):
    service = spec.build_service(
        preset_manager, runner, run_logger, global_settings, shared_state, base_dir, temp_dir, output_dir
    )
    queue_manager = get_processing_queue_manager()
    defaults = service["defaults"]
    seed_controls = shared_state.value.get("seed_controls", {})
    tab_settings = seed_controls.get(f"{spec.key}_settings", {})
    models_list = seed_controls.get("available_models", ["default"])
    merged_defaults = defaults.copy()
    for key, value in tab_settings.items():
        if value is not None:
            merged_defaults[key] = value
    values = [merged_defaults[k] for k in spec.order]

    def _value(key: str, default=None):
        try:
            i = spec.order.index(key)
            if 0 <= i < len(values):
                raw = values[i]
                if raw is None and default is not None:
                    return default
                return raw
        except Exception:
            pass
        return default

    c: Dict[str, Any] = {}
    ctx = TabContext(value=_value, c=c, base_dir=base_dir, shared_state=shared_state, spec=spec)

    def _effective_scale(*, scale_state_val, use_global: bool, local_upscale_factor, state) -> int:
        default_scale = spec.canonical_scale(scale_value=scale_state_val, default=4)
        global_scale = resolve_shared_upscale_factor(state if bool(use_global) else None)
        return spec.canonical_scale(
            scale_value=scale_state_val,
            upscale_factor_value=(global_scale if global_scale is not None else local_upscale_factor),
            default=default_scale,
        )

    ctx.effective_scale = _effective_scale

    def _model_scale_text(requested) -> str:
        """The model-pass scale kept in the hidden `scale` state (fixed for RelayVSR, = requested for FlashVSR)."""
        return str(int(spec.fixed_model_scale or float(requested)))

    cuda_available = False
    try:
        from shared.gpu_utils import get_gpu_info

        gpus = get_gpu_info()
        cuda_available = len(gpus) > 0
        gpu_hint = (f" Detected {len(gpus)} CUDA GPU(s) - GPU acceleration available\n {spec.gpu_note}" if cuda_available
                    else " CUDA not detected (nvidia-smi unavailable or no NVIDIA GPU) - Processing will use CPU (significantly slower)")
    except Exception as e:
        gpu_hint = f" CUDA detection failed: {str(e)}"
    if not cuda_available:
        gr.Markdown(
            f'<div style="background: #fff3cd; padding: 12px; border-radius: 8px; border: 1px solid #ffc107;">'
            f'<strong> GPU Acceleration Unavailable</strong><br>{gpu_hint}<br><br>'
            f'{spec.label} is designed for CUDA GPUs. CPU mode is possible but very slow.</div>',
            elem_classes="warning-text",
        )

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("####  Input")
            with gr.Group():
                with gr.Row():
                    with gr.Column(scale=2, elem_classes=["video-sr-input-source-col"]):
                        input_file = gr.File(label="Upload video or image (optional)", type="filepath",
                                             file_types=["video", "image"])
                        input_path = path_textbox("media+folder", label="Input Path", value=_value("input_path", ""),
                                                  placeholder=spec.input_placeholder, info=spec.input_path_info)
                        copy_output_into_input_btn = gr.Button(
                            "Copy Output Into Input",
                            elem_classes=["action-btn", "action-btn-source-seed", "video-sr-copy-output-compact", "sec-btn-teal"],
                            size="md",
                        )
                    with gr.Column(scale=2):
                        _initial_scale_state = spec.canonical_scale(
                            scale_value=_value("scale", "4"), upscale_factor_value=_value("upscale_factor", None), default=4)
                        scale = gr.State(value=_model_scale_text(_initial_scale_state))
                        c["scale"] = scale
                        spec.build_model_column(ctx)
                    with gr.Column(scale=2):
                        input_image_preview = gr.Image(label=" Input Preview (Image)", type="filepath", interactive=False,
                                                       height=250, visible=False)
                        input_video_preview = gr.Video(label=" Input Preview (Video)", interactive=False, height=250,
                                                       visible=False)
                input_cache_msg = gr.Markdown("", visible=False)
                sizing_info = gr.Markdown("", visible=False, elem_classes=["resolution-info"])
                input_detection_result = gr.Markdown("", visible=False)

            gr.Markdown("#### Core Runtime Settings")
            with gr.Group():
                with gr.Row():
                    spec.build_runtime_row(ctx)
                    with gr.Column(scale=1):
                        seed = gr.Number(label="Random Seed", value=_value("seed", 0), precision=0,
                                         info="Seed for reproducibility.")
                        auto_transfer_output_to_input = gr.Checkbox(
                            label="Auto Transfer Output to Input",
                            value=bool(_value("auto_transfer_output_to_input", False)),
                            info="After upscale completes, automatically copy the latest output path into Input Path.",
                        )
                    with gr.Column(scale=2):
                        auto_tune_btn, save_vram_gb, auto_tune_on_run = autotune_vram_controls(
                            _value("save_vram_gb", None), _value("auto_tune_on_run", True))
                gr.Markdown(spec.autotune_help)
                spec.build_runtime_extra(ctx)
                gr.Markdown("**GPU Device:** Controlled globally from the top app header selector. "
                            "This tab no longer has a per-tab GPU override.")
                device = gr.State(str(_value("device", "auto") or "auto"))

            spec.build_sections(ctx)

            gr.Markdown("#### Output / I/O Settings")
            with gr.Group():
                with gr.Row():
                    spec.build_output_leading(ctx)
                    start_frame = gr.Number(label="Start Frame", value=int(_value("start_frame", 0) or 0), precision=0,
                                            info="0-indexed start frame for partial processing.")
                    end_frame = gr.Number(label="End Frame", value=int(_value("end_frame", -1) or -1), precision=0,
                                          info="-1 = process until the end of input.")
                gr.Markdown("**Video Codec / CRF / FPS Override:** Controlled from `Output > Video Output` "
                            "and automatically applied here.")
                fps_state = gr.State(float(_value("fps", 0.0) or 0.0))
                codec = gr.State(str(_value("codec", "libx264") or "libx264"))
                crf = gr.State(int(_value("crf", _value("quality", 18)) or 18))
                if spec.build_models_dir is not None:
                    spec.build_models_dir(ctx)
                elif "models_dir" in spec.order:
                    c["models_dir"] = gr.State(str(_value("models_dir", "") or ""))
                output_override = path_textbox(
                    "folder", label=spec.output_override_label, value=_value("output_override", ""),
                    placeholder="Leave empty for auto naming",
                    info="Optional custom output location. A folder saves into that folder. A .mp4 file path renames the final output to that exact file.",
                )
                with gr.Row():
                    output_format = gr.Dropdown(label="Output Format", choices=list(spec.output_format_choices),
                                                value=str(_value("output_format", spec.output_format_choices[0]) or spec.output_format_choices[0]),
                                                info=spec.output_format_info, interactive=len(spec.output_format_choices) > 1)
                    save_metadata = gr.Checkbox(label="Save Processing Metadata", value=bool(_value("save_metadata", True)),
                                                info="Save run metadata to output folder.")
                    face_restore_after_upscale = gr.Checkbox(
                        label="Apply Face Restoration after upscale",
                        value=bool(_value("face_restore_after_upscale", False)),
                        info="Per-run face restore toggle. Also respects Global Settings face toggle.",
                    )

            (preset_dropdown, preset_name_input, save_preset_btn, load_preset_btn, preset_status, reset_defaults_btn,
             delete_preset_btn, preset_callbacks) = universal_preset_section(
                preset_manager=preset_manager, shared_state=shared_state, tab_name=spec.key, inputs_list=[],
                base_dir=base_dir, models_list=models_list, open_accordion=True)

        with gr.Column(scale=2):
            gr.Markdown("####  Output & Actions")
            status_box = gr.Markdown(value="Ready.", visible=False, elem_classes=["runtime-status-box"])
            progress_indicator = gr.Markdown(value="", visible=False, elem_classes=["runtime-progress-box"])
            _modal_id = f"{spec.key}-autotune-notice-modal"
            with gr.Group(visible=False, elem_id=_modal_id, elem_classes=["autotune-modal-overlay"]) as autotune_notice_modal:
                with gr.Group(elem_classes=["autotune-modal-card"]):
                    with gr.Row(elem_classes=["autotune-modal-header"]):
                        gr.Markdown("Auto Tune Update", elem_classes=["autotune-modal-title"])
                        autotune_notice_close_btn = gr.Button("X", size="sm", elem_classes=["autotune-modal-close", "sec-btn-slate"])
                    autotune_notice_text = gr.Markdown("", elem_classes=["autotune-modal-body"])
                    with gr.Row(elem_classes=["autotune-modal-actions"]):
                        autotune_notice_ok_btn = gr.Button("OK", variant="primary", elem_classes=["autotune-modal-ok", "sec-btn-pink"])

            with gr.Group():
                sl = spec.upscale_slider
                try:
                    _upscale_factor_default = float(_value("upscale_factor", _value("scale", 4)))
                except Exception:
                    _upscale_factor_default = 4.0
                _upscale_factor_default = float(spec.canonical_scale(scale_value=None, upscale_factor_value=_upscale_factor_default, default=4))
                _initial_use_global_scale = bool(_value("use_resolution_tab", True))
                _initial_shared_scale = resolve_shared_upscale_factor(shared_state.value if _initial_use_global_scale else None)
                if _initial_use_global_scale and _initial_shared_scale is not None:
                    _upscale_factor_default = float(spec.canonical_scale(
                        scale_value=None, upscale_factor_value=_initial_shared_scale, default=_upscale_factor_default))
                try:
                    _max_resolution_default = int(_value("max_target_resolution", 1920))
                except Exception:
                    _max_resolution_default = 1920
                _max_resolution_default = min(8192, max(0, _max_resolution_default))
                with gr.Row():
                    upscale_factor = gr.Slider(label=sl["label"], minimum=sl["minimum"], maximum=sl["maximum"],
                                               step=sl["step"], value=_upscale_factor_default, info=sl["info"],
                                               interactive=True, scale=2)
                    max_target_resolution = gr.Slider(
                        label="Max Resolution (max edge, 0 = no cap)", minimum=0, maximum=8192, step=16,
                        value=_max_resolution_default,
                        info="Caps the LONG side (max(width,height)) of the target. 0 = unlimited.", scale=2)
                    pre_downscale_then_upscale = gr.Checkbox(
                        label="Pre-downscale then upscale (when capped)",
                        value=bool(_value("pre_downscale_then_upscale", True)),
                        info="If max edge would reduce effective scale, downscale input first so the model still runs at the full Upscale x.",
                        scale=1)
                use_resolution_tab = gr.Checkbox(
                    label="Use Resolution & Scene Split Tab Settings",
                    value=bool(_value("use_resolution_tab", True)),
                    info=("When enabled, Upscale x follows shared app scale cache (e.g., SeedVR2/Resolution workflow). "
                          "Moving this local slider will switch this toggle OFF and use local sizing."),
                )
                with gr.Row():
                    upscale_btn = gr.Button(" Start Upscaling", variant="primary", size="lg",
                                            elem_classes=["action-btn", "action-btn-upscale", "sec-btn-emerald"])
                    preview_btn = gr.Button("Preview First Frame", size="lg",
                                            elem_classes=["action-btn", "action-btn-preview", "sec-btn-cyan"])
                with gr.Row():
                    cancel_confirm = gr.Checkbox(label=" Confirm cancel (required for safety)", value=False,
                                                 info="Enable this checkbox to confirm cancellation of processing", scale=3)
                    cancel_btn = gr.Button(" Cancel", variant="stop", size="md", min_width=170, scale=1,
                                           elem_classes=["action-btn", "action-btn-cancel", "sec-btn-red"])

            resume_run_dir = path_textbox(
                "folder", label="Resume Run Folder (chunk/scene resume)", value=_value("resume_run_dir", ""),
                placeholder="Optional: G:/.../outputs/0019",
                info=("Optional. When set, chunk/scene processing resumes from the last completed chunk in that folder. "
                      "Use the same settings as the original run to continue remaining chunks. "
                      "Fresh output path overrides are ignored while resume is active."),
            )
            with gr.Accordion(" Upscaled Output", open=True):
                output_video = gr.Video(label=" Upscaled Video", interactive=False, visible=False, buttons=["download"])
                output_image = gr.Image(label=" Upscaled Image", interactive=False, visible=False, buttons=["download"])
            image_slider = gr.ImageSlider(label=" Comparison", interactive=False, slider_position=50, max_height=1000,
                                          buttons=["download", "fullscreen"], elem_classes=["native-image-comparison-slider"])
            video_comparison_html = gr.HTML(label=" Video Comparison", value="",
                                            js_on_load=get_video_comparison_js_on_load(), visible=False)
            chunk_status = gr.Markdown("", visible=False)
            chunk_gallery = gr.Gallery(label=" Chunk Preview", visible=False, columns=4, rows=2, height=220,
                                       object_fit="contain")
            chunk_preview_video = gr.Video(label=" Selected Chunk", interactive=False, visible=False, buttons=["download"])
            batch_gallery = gr.Gallery(label=" Batch Results", visible=False, columns=4, rows=2, height="auto",
                                       object_fit="contain", buttons=["download"])
            last_processed = gr.Markdown("Batch processing results will appear here.")
            with gr.Row():
                open_outputs_btn = gr.Button(" Open Outputs", elem_classes=["action-btn", "action-btn-open", "sec-btn-blue"])
                clear_temp_btn = gr.Button(" Clear Temp", elem_classes=["action-btn", "action-btn-clear", "sec-btn-orange"])
            with gr.Accordion(" Batch Processing", open=True):
                with gr.Row():
                    batch_enable = gr.Checkbox(label="Enable Batch", value=bool(_value("batch_enable", False)),
                                               info="Process multiple files", scale=2)
                    keep_only_output_files = gr.Checkbox(
                        label="Keep only output files", value=bool(_value("keep_only_output_files", False)),
                        info="After batch completion, remove metadata/chunks/temp artifacts and keep only final outputs.",
                        scale=2)
                batch_input = path_textbox("folder", label="Batch Input Folder", value=_value("batch_input_path", ""),
                                           placeholder="Folder with videos")
                batch_output = path_textbox("folder", label="Batch Output Folder", value=_value("batch_output_path", ""),
                                            placeholder="Output directory")
            log_box = gr.Textbox(label=" Processing Log", value="", lines=12, buttons=["copy"])
            gr.Markdown(spec.guide_md)

    c.update({
        "input_path": input_path, "output_override": output_override, "output_format": output_format,
        "seed": seed, "auto_transfer_output_to_input": auto_transfer_output_to_input, "device": device,
        "fps": fps_state, "codec": codec, "crf": crf, "start_frame": start_frame, "end_frame": end_frame,
        "save_metadata": save_metadata, "face_restore_after_upscale": face_restore_after_upscale,
        "batch_enable": batch_enable, "batch_input_path": batch_input, "batch_output_path": batch_output,
        "use_resolution_tab": use_resolution_tab, "upscale_factor": upscale_factor,
        "max_target_resolution": max_target_resolution, "pre_downscale_then_upscale": pre_downscale_then_upscale,
        "resume_run_dir": resume_run_dir, "save_vram_gb": save_vram_gb,
        "keep_only_output_files": keep_only_output_files, "auto_tune_on_run": auto_tune_on_run,
        "input_file": input_file,
    })
    missing = [k for k in spec.order if k not in c]
    if missing:
        raise RuntimeError(f"{spec.label} tab: no component for settings {missing}")
    inputs_list = [c[k] for k in spec.order]

    def _after_sizing(evt):
        if spec.after_sizing is None:
            return evt
        fn, ins, outs = spec.after_sizing(ctx)
        return evt.then(fn=fn, inputs=ins, outputs=outs, queue=False, show_progress="hidden", trigger_mode="always_last")

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

    def _build_sizing_info(path_val, model_scale_val, use_global, local_scale_x, local_max_edge, local_pre_down, state):
        resolved_scale = _effective_scale(
            scale_state_val=model_scale_val,
            use_global=bool(use_global),
            local_upscale_factor=local_scale_x,
            state=state if isinstance(state, dict) else None,
        )
        requested = float(resolved_scale or 4)
        ms = int(spec.fixed_model_scale or requested)
        return build_fixed_scale_analysis_update(
            input_path_val=path_val,
            model_scale=ms,
            # Scale is resolved locally (with Resolution-tab fallback + engine clamp)
            # so the preview cannot desync from hidden backend state.
            use_global=False,
            local_scale_x=requested,
            local_max_edge=int(local_max_edge or 0),
            local_pre_down=bool(local_pre_down),
            state=state,
            model_label=spec.model_label,
            runtime_label=spec.runtime_label(ms),
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

    upload_evt = input_file.upload(
        fn=cache_input_upload,
        inputs=[input_file, scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state]
    )
    _after_sizing(upload_evt)

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

    upload_clear_evt = input_file.change(
        fn=clear_input_path_on_upload_clear,
        inputs=[input_file, shared_state],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )
    _after_sizing(upload_clear_evt)

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

        batch_outputs = seed_controls.get(f"{spec.key}_batch_outputs", [])
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
        scale_val,
        use_global,
        scale_x,
        max_edge,
        pre_down,
        state,
        source_label="Output transferred to input.",
    ):
        state = state or {}
        state.setdefault("seed_controls", {})
        state["seed_controls"]["last_input_path"] = output_path_val or ""

        try:
            img_prev, vid_prev, det, info, state_out = _run_analysis_payload(
                output_path_val or "",
                scale_val,
                use_global,
                scale_x,
                max_edge,
                pre_down,
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

    def copy_latest_output_to_input(scale_val, use_global, scale_x, max_edge, pre_down, state):
        state = state if isinstance(state, dict) else {}
        output_path_val = _resolve_latest_output_path(state)
        if not output_path_val:
            return (
                gr.skip(),
                gr.update(value="⚠️ No generated output found to transfer.", visible=True),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                state,
            )
        return _apply_output_path_to_input(
            output_path_val,
            scale_val,
            use_global,
            scale_x,
            max_edge,
            pre_down,
            state,
            source_label="Output path copied into input.",
        )

    def capture_latest_output_signature(state):
        return _output_path_signature(_resolve_latest_output_path(state))

    def auto_transfer_latest_output_to_input(
        scale_val,
        use_global,
        scale_x,
        max_edge,
        pre_down,
        auto_enabled,
        previous_signature,
        state,
    ):
        state = state if isinstance(state, dict) else {}
        if not bool(auto_enabled):
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                state,
            )

        output_path_val = _resolve_latest_output_path(state)
        if not output_path_val:
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                state,
            )

        latest_signature = _output_path_signature(output_path_val)
        if previous_signature and latest_signature and str(previous_signature) == str(latest_signature):
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                state,
            )

        return _apply_output_path_to_input(
            output_path_val,
            scale_val,
            use_global,
            scale_x,
            max_edge,
            pre_down,
            state,
            source_label="Auto-transferred latest output into input.",
        )

    # Keep heavy media/path analysis user-triggered only.
    # `.change()` also fires on function updates (e.g. tab sync), which can fan out
    # into expensive cascades on large UIs.
    input_submit_evt = input_path.submit(
        fn=cache_input_path,
        inputs=[input_path, scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )
    _after_sizing(input_submit_evt)

    copy_output_evt = copy_output_into_input_btn.click(
        fn=copy_latest_output_to_input,
        inputs=[scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )
    _after_sizing(copy_output_evt)


    def refresh_sizing(path_val, scale_val, use_global, local_scale_x, max_edge, pre_down, state):
        return _build_sizing_info(path_val, int(scale_val), bool(use_global), local_scale_x, max_edge, pre_down, state)

    scale_evt = scale.change(
        fn=refresh_sizing,
        inputs=[input_path, scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[sizing_info],
        trigger_mode="always_last",
    )
    _after_sizing(scale_evt)

    pre_down_evt = pre_downscale_then_upscale.change(
        fn=refresh_sizing,
        inputs=[input_path, scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[sizing_info],
        trigger_mode="always_last",
    )
    _after_sizing(pre_down_evt)

    max_res_evt = max_target_resolution.release(
        fn=refresh_sizing,
        inputs=[input_path, scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[sizing_info],
        preprocess=False,
        trigger_mode="always_last",
    )
    _after_sizing(max_res_evt)

    def _sync_upscale_ui(use_global, local_x, state):
        shared_scale = resolve_shared_upscale_factor(state if bool(use_global) else None)
        if bool(use_global) and shared_scale is not None:
            effective = spec.canonical_scale(
                scale_value=None,
                upscale_factor_value=shared_scale,
                default=local_x if local_x is not None else 4,
            )
            return (
                gr.update(value=float(effective), interactive=True),
                gr.update(value=_model_scale_text(effective)),
            )

        effective_local = spec.canonical_scale(
            scale_value=None,
            upscale_factor_value=local_x,
            default=4,
        )
        return (
            gr.update(value=float(effective_local), interactive=True),
            gr.update(value=_model_scale_text(effective_local)),
        )

    def _sync_upscale_ui_and_sizing(use_global, local_x, path_val, max_edge, pre_down, state):
        slider_upd, scale_upd = _sync_upscale_ui(use_global, local_x, state)
        shared_scale = resolve_shared_upscale_factor(state if bool(use_global) else None)
        effective_scale = shared_scale if (bool(use_global) and shared_scale is not None) else local_x
        resolved = spec.canonical_scale(
            scale_value=None,
            upscale_factor_value=effective_scale,
            default=4,
        )
        info = _build_sizing_info(
            path_val or "",
            int(resolved),
            bool(use_global),
            float(resolved),
            max_edge,
            pre_down,
            state,
        )
        return slider_upd, scale_upd, info

    def _on_local_upscale_changed(local_x, use_global, state):
        resolved = spec.canonical_scale(
            scale_value=None,
            upscale_factor_value=local_x,
            default=4,
        )
        # If a shared/global scale is currently active, local slider interaction
        # switches to local mode instead of staying locked behind global mode.
        shared_scale = resolve_shared_upscale_factor(state if bool(use_global) else None)
        next_use_global = bool(use_global)
        if bool(use_global) and shared_scale is not None:
            next_use_global = False
        return gr.update(value=next_use_global), gr.update(value=_model_scale_text(resolved))

    local_scale_evt = upscale_factor.release(
        fn=_on_local_upscale_changed,
        inputs=[upscale_factor, use_resolution_tab, shared_state],
        outputs=[use_resolution_tab, scale],
        preprocess=False,
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )
    local_scale_evt.then(
        fn=refresh_sizing,
        inputs=[input_path, scale, use_resolution_tab, upscale_factor, max_target_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[sizing_info],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )
    _after_sizing(local_scale_evt)

    use_resolution_evt = use_resolution_tab.change(
        fn=_sync_upscale_ui_and_sizing,
        inputs=[use_resolution_tab, upscale_factor, input_path, max_target_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[upscale_factor, scale, sizing_info],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )
    _after_sizing(use_resolution_evt)

    def refresh_chunk_preview_ui(state):
        preview = (state or {}).get("seed_controls", {}).get(f"{spec.key}_chunk_preview", {})
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
            videos = (state or {}).get("seed_controls", {}).get(f"{spec.key}_chunk_preview", {}).get("videos", [])
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

    def _batch_gallery_update_from_state(state):
        outputs = (state or {}).get("seed_controls", {}).get(f"{spec.key}_batch_outputs", [])
        if not isinstance(outputs, list):
            outputs = []
        outputs = [str(p) for p in outputs if p and Path(str(p)).exists()]
        return gr.update(value=outputs, visible=bool(outputs))

    def _last_processed_text(state, vid_upd, img_upd) -> str:
        outputs = (state or {}).get("seed_controls", {}).get(f"{spec.key}_batch_outputs", [])
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
                gr.update(value=f"ERROR: Invalid {spec.label} payload"),
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
            "max resolution preprocess failed",
        )
        is_terminal = any(tok in status_lc for tok in terminal_tokens) or any(
            tok in log_lc for tok in ("critical error", "processing failed", "cancelled", "out of vram")
        )

        if (status_text or log_tail) and not is_terminal:
            title = _compact_single_line(status_text or f"{spec.label} processing...", max_len=96)
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
        queue_resource_keys, queue_resource_label = resolve_queue_gpu_resources(queued_state, queued_global_settings)
        ticket = queue_manager.submit(
            spec.label,
            "Upscale",
            resource_keys=queue_resource_keys,
            resource_label=queue_resource_label,
        )
        acquired_slot = queue_manager.is_active(ticket.job_id)

        try:
            yield _starting_runtime_output(live_state, "Upscale")

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
        yield _starting_runtime_output(live_state, "Preview")
        for payload in service["run_action"](
            args[0],
            *args[1:-1],
            preview_only=True,
            state=queued_state,
            progress=progress,
            global_settings_snapshot=queued_global_settings,
        ):
            yield _expand_service_payload(payload, live_state)

    upscale_sync_signature = gr.State(value="")
    chunk_sync_signature = gr.State(value="")
    pre_run_output_signature = gr.State(value="")

    def _sync_upscale_ui_and_sizing_if_needed(
        use_global,
        local_x,
        path_val,
        max_edge,
        pre_down,
        state,
        previous_signature: str = "",
    ):
        seed_controls = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
        signature = _sync_signature(
            {
                "use_global": bool(use_global),
                "local_x": local_x,
                "path_val": path_val,
                "max_edge": max_edge,
                "pre_down": bool(pre_down),
                "shared_scale": resolve_shared_upscale_factor(state if bool(use_global) else None),
                "resolution_settings": seed_controls.get("resolution_settings", {}),
            }
        )
        if signature == str(previous_signature or ""):
            return gr.skip(), gr.skip(), gr.skip(), previous_signature
        slider_upd, scale_upd, sizing_upd = _sync_upscale_ui_and_sizing(
            use_global, local_x, path_val, max_edge, pre_down, state
        )
        return slider_upd, scale_upd, sizing_upd, signature

    def _refresh_chunk_preview_ui_if_needed(state, previous_signature: str = ""):
        preview = (state or {}).get("seed_controls", {}).get(f"{spec.key}_chunk_preview", {})
        signature = _sync_signature(preview if isinstance(preview, dict) else {"preview": None})
        if signature == str(previous_signature or ""):
            return gr.skip(), gr.skip(), gr.skip(), previous_signature
        chunk_status_upd, chunk_gallery_upd, chunk_video_upd = refresh_chunk_preview_ui(state)
        return chunk_status_upd, chunk_gallery_upd, chunk_video_upd, signature

    # Registered before the Upscale job: Gradio runs a button's listeners in order and waits for a
    # queued job to finish before the next one, so a later listener would update the UI only after the run.
    tuned = {k: c[k] for k in spec.tuned_keys}
    wire_auto_tune_on_run(
        upscale_btn=upscale_btn,
        action=service["auto_tune_action"],
        inputs=[input_file] + inputs_list,
        tuned=tuned,
        shared_state=shared_state,
        auto_tune_on_run=auto_tune_on_run,
        global_settings=global_settings,
    )

    # Main processing
    run_evt = upscale_btn.click(
        fn=with_pre_run_output_signature(run_upscale_with_queue, capture_latest_output_signature),
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
            pre_run_output_signature,
        ],
        concurrency_limit=32,
        concurrency_id="app_processing_queue",
        trigger_mode="multiple",
    )
    run_evt.then(
        fn=_refresh_chunk_preview_ui_if_needed,
        inputs=[shared_state, chunk_sync_signature],
        outputs=[chunk_status, chunk_gallery, chunk_preview_video, chunk_sync_signature],
    )
    auto_transfer_evt = run_evt.then(
        fn=auto_transfer_latest_output_to_input,
        inputs=[
            scale,
            use_resolution_tab,
            upscale_factor,
            max_target_resolution,
            pre_downscale_then_upscale,
            auto_transfer_output_to_input,
            pre_run_output_signature,
            shared_state,
        ],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )
    _after_sizing(auto_transfer_evt)

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
        fn=_refresh_chunk_preview_ui_if_needed,
        inputs=[shared_state, chunk_sync_signature],
        outputs=[chunk_status, chunk_gallery, chunk_preview_video, chunk_sync_signature],
    )

    autotune_evt = wire_auto_tune(
        auto_tune_btn=auto_tune_btn,
        action=service["auto_tune_action"],
        inputs=[input_file] + inputs_list,
        tuned=tuned,
        status_box=status_box,
        shared_state=shared_state,
        notice_text=autotune_notice_text,
        notice_modal=autotune_notice_modal,
        modal_id=_modal_id,
        global_settings=global_settings,
    )
    _after_sizing(autotune_evt)

    def _dismiss_autotune_notice():
        return gr.update(value=""), gr.update(visible=False)

    autotune_notice_ok_btn.click(
        fn=_dismiss_autotune_notice,
        outputs=[autotune_notice_text, autotune_notice_modal],
        queue=False,
        show_progress="hidden",
        js=autotune_modal_dismiss_js(_modal_id),
    )
    autotune_notice_close_btn.click(
        fn=_dismiss_autotune_notice,
        outputs=[autotune_notice_text, autotune_notice_modal],
        queue=False,
        show_progress="hidden",
        js=autotune_modal_dismiss_js(_modal_id),
    )

    shared_scale_sync_evt = shared_state.change(
        fn=_sync_upscale_ui_and_sizing_if_needed,
        inputs=[use_resolution_tab, upscale_factor, input_path, max_target_resolution, pre_downscale_then_upscale, shared_state, upscale_sync_signature],
        outputs=[upscale_factor, scale, sizing_info, upscale_sync_signature],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )
    _after_sizing(shared_scale_sync_evt)

    shared_state.change(
        fn=_refresh_chunk_preview_ui_if_needed,
        inputs=[shared_state, chunk_sync_signature],
        outputs=[chunk_status, chunk_gallery, chunk_preview_video, chunk_sync_signature],
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
        tab_name=spec.key,
    )

    if spec.after_sizing is not None:
        _fn, _ins, _outs = spec.after_sizing(ctx)
        input_path.change(fn=_fn, inputs=_ins, outputs=_outs, queue=False, show_progress="hidden",
                          trigger_mode="always_last")
    if spec.wire_extra is not None:
        spec.wire_extra(ctx)

    return {
        "inputs_list": inputs_list,
        "run_action": service["run_action"],
        "cancel_action": service["cancel_action"],
        "preset_dropdown": preset_dropdown,
        "preset_status": preset_status,
    }
