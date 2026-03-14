"""
RTX Super Resolution service callbacks.

Implements a tab-level service with:
- universal preset support
- queue-friendly streaming updates
- chunk/scene split + resume support
- preview mode
- cancel handling + partial salvage
"""

from __future__ import annotations

import html
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import gradio as gr

from shared.chunk_preview import build_chunk_preview_payload
from shared.comparison_unified import create_video_comparison_slider
from shared.comparison_video_service import maybe_generate_input_vs_output_comparison
from shared.gpu_utils import (
    expand_cuda_device_spec,
    get_global_gpu_override,
    validate_cuda_device_spec,
)
from shared.logging_utils import RunLogger
from shared.oom_alert import clear_vram_oom_alert, maybe_set_vram_oom_alert, show_vram_oom_modal
from shared.output_run_manager import (
    batch_item_dir,
    ensure_image_input_artifact,
    finalize_run_context,
    prepare_batch_video_run_dir,
    prepare_single_video_run,
    resolve_resume_input_from_run_dir,
)
from shared.path_utils import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    detect_input_type,
    get_media_fps,
    normalize_path,
    resolve_batch_output_dir,
)
from shared.preset_manager import PresetManager
from shared.preview_utils import prepare_preview_input
from shared.global_rife import maybe_apply_global_rife
from shared.rtx_superres_runner import run_rtx_superres


RTX_QUALITY_PRESETS: List[str] = [
    "ULTRA",
    "HIGH",
    "MEDIUM",
    "LOW",
    "BICUBIC",
    "DENOISE_ULTRA",
    "DENOISE_HIGH",
    "DENOISE_MEDIUM",
    "DENOISE_LOW",
    "DEBLUR_ULTRA",
    "DEBLUR_HIGH",
    "DEBLUR_MEDIUM",
    "DEBLUR_LOW",
    "HIGHBITRATE_ULTRA",
    "HIGHBITRATE_HIGH",
    "HIGHBITRATE_MEDIUM",
    "HIGHBITRATE_LOW",
]


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _resolve_shared_upscale_factor(state: Dict[str, Any] | None) -> Optional[float]:
    if not isinstance(state, dict):
        return None
    seed_controls = state.get("seed_controls", {}) if isinstance(state, dict) else {}
    try:
        val = float(seed_controls.get("upscale_factor_val"))
    except Exception:
        return None
    if val <= 0:
        return None
    return val


def rtx_super_resolution_defaults() -> Dict[str, Any]:
    return {
        "input_path": "",
        "output_override": "",
        "batch_enable": False,
        "batch_input_path": "",
        "batch_output_path": "",
        "quality_preset": "HIGHBITRATE_ULTRA",
        "use_resolution_tab": True,
        "upscale_factor": 4.0,
        "max_resolution": 3840,
        "pre_downscale_then_upscale": True,
        "non_blocking_inference": True,
        "disable_auto_scene_detection_split": True,
        "cuda_stream_ptr": 0,
        "output_format": "auto",
        "face_restore_after_upscale": False,
        "resume_run_dir": "",
        "auto_transfer_output_to_input": False,
        "streaming_chunk_size_frames": 0,
        "resume_partial_chunks": False,
    }


RTX_ORDER: List[str] = [
    "input_path",
    "output_override",
    "batch_enable",
    "batch_input_path",
    "batch_output_path",
    "quality_preset",
    "use_resolution_tab",
    "upscale_factor",
    "max_resolution",
    "pre_downscale_then_upscale",
    "non_blocking_inference",
    "disable_auto_scene_detection_split",
    "cuda_stream_ptr",
    "output_format",
    "face_restore_after_upscale",
    "resume_run_dir",
    "auto_transfer_output_to_input",
    "streaming_chunk_size_frames",
    "resume_partial_chunks",
]


def _rtx_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    return dict(zip(RTX_ORDER, args))


def _apply_rtx_preset(
    preset: Dict[str, Any],
    defaults: Dict[str, Any],
    preset_manager: PresetManager,
    current: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    base = defaults.copy()
    if current:
        base.update(current)
    merged = preset_manager.merge_config(base, preset or {})
    return [merged.get(k, defaults.get(k)) for k in RTX_ORDER]


def build_rtx_super_resolution_callbacks(
    preset_manager: PresetManager,
    runner,
    run_logger: RunLogger,
    global_settings: Dict[str, Any],
    shared_state: gr.State,
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path,
):
    defaults = rtx_super_resolution_defaults()
    cancel_event = threading.Event()

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

    def _media_updates(out_path: Optional[str]) -> tuple[Any, Any]:
        try:
            if out_path and not Path(out_path).is_dir():
                suf = Path(out_path).suffix.lower()
                if suf in video_exts:
                    return gr.update(value=None, visible=False), gr.update(value=out_path, visible=True)
                if suf in image_exts:
                    return gr.update(value=out_path, visible=True), gr.update(value=None, visible=False)
        except Exception:
            pass
        return gr.update(value=None, visible=False), gr.update(value=None, visible=False)

    def _running_indicator(title: str, subtitle: str, spinning: bool = True):
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

    def _build_payload(
        *,
        status: str,
        logs: str,
        progress_upd: Any,
        output_path: Optional[str],
        last_processed: str,
        slider_upd: Any = None,
        html_upd: Any = None,
        batch_upd: Any = None,
        state: Optional[Dict[str, Any]] = None,
    ):
        img_upd, vid_upd = _media_updates(output_path)
        return (
            status,
            logs,
            progress_upd,
            img_upd,
            vid_upd,
            last_processed,
            slider_upd if slider_upd is not None else gr.update(value=None),
            html_upd if html_upd is not None else gr.update(value="", visible=False),
            batch_upd if batch_upd is not None else gr.update(value=[], visible=False),
            state or {},
        )

    def refresh_presets(select_name: Optional[str] = None):
        presets = preset_manager.list_presets("rtx_super_resolution", None)
        last_used = preset_manager.get_last_used_name("rtx_super_resolution", None)
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        return gr.update(choices=presets, value=value)

    def save_preset(preset_name: str, *args):
        if not str(preset_name or "").strip():
            return gr.update(), gr.update(value="WARNING: Enter a preset name before saving"), *list(args)
        if len(args) != len(RTX_ORDER):
            return (
                gr.update(),
                gr.update(value=f"WARNING: Preset mismatch ({len(args)} vs {len(RTX_ORDER)})"),
                *list(args),
            )
        payload = _rtx_dict_from_args(list(args))
        preset_manager.save_preset_safe("rtx_super_resolution", None, str(preset_name).strip(), payload)
        dropdown = refresh_presets(select_name=str(preset_name).strip())
        current_map = dict(zip(RTX_ORDER, list(args)))
        loaded_vals = _apply_rtx_preset(payload, defaults, preset_manager, current=current_map)
        return (
            dropdown,
            gr.update(value=f"SUCCESS: Saved preset '{preset_name}'"),
            *loaded_vals,
        )

    def load_preset(preset_name: str, current_values: List[Any]):
        try:
            preset = preset_manager.load_preset_safe("rtx_super_resolution", None, preset_name)
            if preset:
                preset_manager.set_last_used("rtx_super_resolution", None, preset_name)
                preset = preset_manager.validate_preset_constraints(preset, "rtx_super_resolution", None)
            current_map = dict(zip(RTX_ORDER, current_values))
            values = _apply_rtx_preset(preset or {}, defaults, preset_manager, current=current_map)
            status_msg = f"SUCCESS: Loaded preset '{preset_name}'" if preset else "INFO: Preset not found"
            return (*values, gr.update(value=status_msg))
        except Exception as e:
            return (*current_values, gr.update(value=f"ERROR: {str(e)}"))

    def safe_defaults():
        return [defaults[k] for k in RTX_ORDER]

    def _resolve_quality(raw: Any) -> str:
        cand = str(raw or "HIGHBITRATE_ULTRA").strip().upper()
        if cand in RTX_QUALITY_PRESETS:
            return cand
        return "HIGHBITRATE_ULTRA"

    def _resolve_output_format(raw: Any) -> str:
        fmt = str(raw or "auto").strip().lower()
        if fmt in {"auto", "mp4", "png"}:
            return fmt
        return "auto"

    def _prepare_single_settings(
        *,
        item_input: str,
        settings_base: Dict[str, Any],
        seed_controls_local: Dict[str, Any],
    ) -> Dict[str, Any]:
        s = settings_base.copy()
        s["input_path"] = normalize_path(item_input)
        s["quality_preset"] = _resolve_quality(s.get("quality_preset"))
        s["output_format"] = _resolve_output_format(s.get("output_format"))
        s["global_output_dir"] = str(global_settings.get("output_dir", output_dir))

        use_global = bool(s.get("use_resolution_tab", True))
        shared_scale = _resolve_shared_upscale_factor({"seed_controls": seed_controls_local} if use_global else None)
        if shared_scale is not None:
            s["upscale_factor"] = float(shared_scale)
        else:
            s["upscale_factor"] = max(1.0, min(9.9, _to_float(s.get("upscale_factor"), 4.0)))

        s["max_resolution"] = max(0, _to_int(s.get("max_resolution"), 0))
        s["pre_downscale_then_upscale"] = _to_bool(s.get("pre_downscale_then_upscale"), True)
        s["non_blocking_inference"] = _to_bool(s.get("non_blocking_inference"), True)
        s["disable_auto_scene_detection_split"] = _to_bool(s.get("disable_auto_scene_detection_split"), True)
        s["cuda_stream_ptr"] = _to_int(s.get("cuda_stream_ptr"), 0)

        s["image_output_format"] = str(
            seed_controls_local.get("image_output_format_val", "png")
        ).strip().lower() or "png"
        s["image_output_quality"] = int(seed_controls_local.get("image_output_quality_val", 95) or 95)
        s["fps"] = float(seed_controls_local.get("fps_override_val", 0) or 0) or float(
            get_media_fps(s["input_path"]) or 30.0
        )

        s["audio_codec"] = seed_controls_local.get("audio_codec_val") or "copy"
        s["audio_bitrate"] = seed_controls_local.get("audio_bitrate_val") or ""
        s["save_metadata"] = bool(seed_controls_local.get("save_metadata_val", True))
        return s

    def _apply_face_restore_if_enabled(
        output_path: Optional[str],
        face_apply: bool,
        face_strength: float,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        if not output_path or not face_apply:
            return output_path
        path_obj = Path(output_path)
        if not path_obj.exists() or path_obj.is_dir():
            return output_path
        try:
            from shared.face_restore import restore_image, restore_video
        except Exception:
            return output_path

        try:
            if path_obj.suffix.lower() in video_exts:
                restored = restore_video(
                    str(path_obj),
                    strength=float(face_strength),
                    on_progress=(lambda m: progress_cb(m) if (progress_cb and m) else None),
                )
            else:
                restored = restore_image(str(path_obj), strength=float(face_strength))
            if restored and Path(restored).exists():
                return str(restored)
        except Exception:
            return output_path
        return output_path

    def _apply_audio_policy(
        output_path: Optional[str],
        audio_source_path: Optional[str],
        settings_local: Dict[str, Any],
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        if not output_path:
            return output_path
        outp = Path(output_path)
        if not outp.exists() or outp.is_dir() or outp.suffix.lower() not in video_exts:
            return output_path
        if not audio_source_path:
            return output_path
        src = Path(audio_source_path)
        if not src.exists():
            return output_path
        try:
            from shared.audio_utils import ensure_audio_on_video

            changed, final_path, err = ensure_audio_on_video(
                outp,
                src,
                audio_codec=str(settings_local.get("audio_codec") or "copy"),
                audio_bitrate=str(settings_local.get("audio_bitrate")) if settings_local.get("audio_bitrate") else None,
                on_progress=(lambda m: progress_cb(m) if (progress_cb and m) else None),
            )
            if err and progress_cb:
                progress_cb(f"WARNING: Audio mux: {err}")
            if changed and final_path and Path(final_path).exists():
                return str(final_path)
        except Exception:
            return output_path
        return output_path

    def _create_comparison_updates(input_path_val: str, output_path_val: Optional[str]) -> tuple[Any, Any]:
        if not output_path_val:
            return gr.update(value=None), gr.update(value="", visible=False)
        try:
            outp = Path(output_path_val)
            if not outp.exists():
                return gr.update(value=None), gr.update(value="", visible=False)
            if outp.is_dir():
                return gr.update(value=None), gr.update(value=f"<p>PNG frames saved to {outp}</p>", visible=True)
            if outp.suffix.lower() in video_exts:
                html_comp = create_video_comparison_slider(input_path_val, output_path_val)
                return gr.update(value=None), gr.update(value=html_comp, visible=True)
            return gr.update(value=(input_path_val, output_path_val), visible=True), gr.update(value="", visible=False)
        except Exception:
            return gr.update(value=None), gr.update(value="", visible=False)

    def _append_run_summary(
        *,
        settings_local: Dict[str, Any],
        output_path_val: Optional[str],
        status_val: str,
        returncode_val: int,
    ) -> None:
        if not bool(settings_local.get("save_metadata", True)):
            return
        try:
            run_logger.write_summary(
                Path(output_path_val) if output_path_val else Path(global_settings.get("output_dir", output_dir)),
                {
                    "input": settings_local.get("_original_input_path_before_preprocess") or settings_local.get("input_path"),
                    "output": output_path_val,
                    "returncode": int(returncode_val),
                    "args": settings_local,
                    "pipeline": "rtx_super_resolution",
                    "status": status_val,
                },
            )
        except Exception:
            pass

    def _update_last_output_state(seed_controls: Dict[str, Any], out_path: Optional[str]) -> None:
        if not out_path:
            return
        try:
            p = Path(out_path)
            seed_controls["rtx_last_output_path"] = str(p)
            seed_controls["last_output_dir"] = str(p.parent if p.is_file() else p)
            seed_controls["last_output_path"] = str(p) if p.is_file() else None
        except Exception:
            pass

    def run_action(
        upload,
        *args,
        preview_only: bool = False,
        state=None,
        progress=None,
        global_settings_snapshot: Dict[str, Any] | None = None,
        _global_settings: Dict[str, Any] = global_settings,
    ):
        global_settings_local = (
            dict(global_settings_snapshot)
            if isinstance(global_settings_snapshot, dict)
            else dict(_global_settings)
        )

        progress_q: "queue.Queue[str]" = queue.Queue()
        result_holder: Dict[str, Any] = {}

        def _emit_progress_line(text: str) -> None:
            line = str(text or "").strip()
            if not line:
                return
            progress_q.put(line)

        def _run_single(prepped_settings: Dict[str, Any], seed_controls_local: Dict[str, Any]):
            from shared.runner import RunResult

            runtime_settings = prepped_settings.copy()
            input_path_local = str(runtime_settings.get("input_path") or "")
            run_output_root = Path(runtime_settings.get("_run_dir") or global_settings_local.get("output_dir", output_dir))
            run_output_root.mkdir(parents=True, exist_ok=True)

            face_apply = bool(runtime_settings.get("face_restore_after_upscale", False)) or bool(
                global_settings_local.get("face_global", False)
            )
            face_strength = float(global_settings_local.get("face_strength", 0.5))

            disable_scene_split = bool(runtime_settings.get("disable_auto_scene_detection_split", True))
            auto_chunk = bool(seed_controls_local.get("auto_chunk", True))
            chunk_size_sec = float(seed_controls_local.get("chunk_size_sec", 0) or 0)
            if disable_scene_split:
                auto_chunk = False
                chunk_size_sec = 0.0
            chunk_overlap_sec = 0.0 if auto_chunk else float(seed_controls_local.get("chunk_overlap_sec", 0) or 0)
            scene_threshold = float(seed_controls_local.get("scene_threshold", 27.0))
            min_scene_len = float(seed_controls_local.get("min_scene_len", 1.0))
            frame_accurate_split = bool(seed_controls_local.get("frame_accurate_split", True))
            per_chunk_cleanup = bool(seed_controls_local.get("per_chunk_cleanup", False))

            if disable_scene_split:
                _emit_progress_line(
                    "[RTX] Override is active: Resolution tab scene detection/split is disabled for this run."
                )

            native_chunk_frames = max(0, _to_int(runtime_settings.get("streaming_chunk_size_frames"), 0))
            input_kind = detect_input_type(input_path_local)
            resolution_chunk_requested = input_kind == "video" and (auto_chunk or chunk_size_sec > 0)
            native_chunk_requested = input_kind == "video" and native_chunk_frames > 0
            should_chunk = input_kind == "video" and (resolution_chunk_requested or native_chunk_requested)
            resume_requested = bool(
                runtime_settings.get("_resume_run_requested")
                or str(runtime_settings.get("resume_run_dir") or "").strip()
                or bool(runtime_settings.get("resume_partial_chunks", False))
            )

            if resume_requested and not should_chunk:
                return (
                    "Resume unavailable for current mode",
                    "Resume folder works only with chunk/scene processing mode.",
                    None,
                    "",
                    gr.update(value=None),
                    1,
                )

            def _finalize_context(status_text: str, output_path_text: Optional[str], returncode_val: int) -> None:
                try:
                    run_dir_raw = runtime_settings.get("_run_dir")
                    if not run_dir_raw:
                        return
                    effective_input = runtime_settings.get("_effective_input_path") or runtime_settings.get("input_path")
                    input_kind_for_ctx = detect_input_type(str(effective_input or ""))
                    pre_in = runtime_settings.get("_preprocessed_input_path")
                    if input_kind_for_ctx == "image" and (not pre_in):
                        snap = ensure_image_input_artifact(
                            Path(run_dir_raw),
                            str(effective_input or ""),
                            preferred_stem=Path(runtime_settings.get("_original_filename") or "used_input").stem,
                        )
                        if snap:
                            pre_in = str(snap)
                    finalize_run_context(
                        Path(run_dir_raw),
                        pipeline="rtx_super_resolution",
                        status=str(status_text or ""),
                        returncode=int(returncode_val),
                        output_path=str(output_path_text) if output_path_text else None,
                        original_input_path=str(
                            runtime_settings.get("_original_input_path_before_preprocess")
                            or runtime_settings.get("_preview_original_input")
                            or runtime_settings.get("input_path")
                            or ""
                        ),
                        effective_input_path=str(effective_input) if effective_input else None,
                        preprocessed_input_path=str(pre_in) if pre_in else None,
                        input_kind=input_kind_for_ctx,
                    )
                except Exception:
                    pass

            if should_chunk:
                from shared.chunking import check_resume_available, chunk_and_process

                if resume_requested:
                    ok_resume, msg_resume = check_resume_available(run_output_root, "mp4")
                    if not ok_resume:
                        return (
                            "Resume failed",
                            f"Resume requested but no resumable chunk outputs were found in {run_output_root}. {msg_resume}",
                            None,
                            "",
                            gr.update(value=None),
                            1,
                        )
                    _emit_progress_line(
                        "Resume folder detected. Continuing from last completed chunk with same settings."
                    )

                chunk_seconds_effective = 0.0
                if resolution_chunk_requested:
                    if auto_chunk:
                        chunk_seconds_effective = 0.0
                    elif chunk_size_sec > 0:
                        chunk_seconds_effective = float(chunk_size_sec)
                elif native_chunk_requested:
                    fps_src = float(get_media_fps(input_path_local) or 30.0)
                    chunk_seconds_effective = max(0.1, float(native_chunk_frames) / max(1.0, fps_src))

                if resolution_chunk_requested and native_chunk_requested and auto_chunk:
                    _emit_progress_line(
                        "[RTX] Native chunk-size is ignored while auto scene split is active. "
                        "Disable auto scene split to enforce fixed frame chunks."
                    )

                class _CancelProbe:
                    def is_canceled(self) -> bool:
                        return bool(cancel_event.is_set())

                chunk_settings = runtime_settings.copy()
                chunk_settings["frame_accurate_split"] = frame_accurate_split
                chunk_settings["chunk_size_sec"] = chunk_seconds_effective
                chunk_settings["chunk_overlap_sec"] = chunk_overlap_sec
                chunk_settings["per_chunk_cleanup"] = per_chunk_cleanup
                chunk_settings["output_format"] = _resolve_output_format(chunk_settings.get("output_format"))

                def _process_chunk(s: Dict[str, Any], on_progress=None) -> RunResult:
                    res = run_rtx_superres(
                        s,
                        base_dir=base_dir,
                        on_progress=on_progress,
                        cancel_event=cancel_event,
                    )
                    return RunResult(res.returncode, res.output_path, res.log)

                def _chunk_progress_cb(progress_val: float, desc: str = "", **kwargs):
                    if desc:
                        _emit_progress_line(str(desc))
                    phase = str(kwargs.get("phase", "")).strip().lower()
                    if phase == "completed":
                        try:
                            seed_controls_local["rtx_chunk_preview"] = build_chunk_preview_payload(str(run_output_root))
                        except Exception:
                            pass

                rc, clog, final_output, chunk_count = chunk_and_process(
                    runner=_CancelProbe(),
                    settings=chunk_settings,
                    scene_threshold=scene_threshold,
                    min_scene_len=min_scene_len,
                    work_dir=run_output_root,
                    on_progress=lambda msg: _emit_progress_line(str(msg).strip()),
                    chunk_seconds=chunk_seconds_effective,
                    chunk_overlap=chunk_overlap_sec,
                    per_chunk_cleanup=per_chunk_cleanup,
                    allow_partial=True,
                    global_output_dir=str(run_output_root),
                    resume_from_partial=resume_requested,
                    progress_tracker=_chunk_progress_cb,
                    process_func=_process_chunk,
                    model_type="rtx",
                )

                output_path_local = final_output if final_output else None
                if output_path_local:
                    output_path_local = _apply_face_restore_if_enabled(
                        output_path_local, face_apply, face_strength, progress_cb=_emit_progress_line
                    )
                    output_path_local = _apply_audio_policy(
                        output_path_local,
                        runtime_settings.get("_original_input_path_before_preprocess") or input_path_local,
                        runtime_settings,
                        progress_cb=_emit_progress_line,
                    )

                if (
                    output_path_local
                    and Path(output_path_local).exists()
                    and Path(output_path_local).suffix.lower() in video_exts
                ):
                    rife_out, rife_msg = maybe_apply_global_rife(
                        runner=runner,
                        output_video_path=output_path_local,
                        seed_controls=seed_controls_local,
                        on_log=(lambda m: _emit_progress_line(m) if m else None),
                        chunking_context={
                            "enabled": bool(chunk_count and chunk_count > 0),
                            "auto_chunk": bool(auto_chunk),
                            "chunk_size_sec": float(chunk_seconds_effective or 0),
                            "chunk_overlap_sec": float(chunk_overlap_sec or 0),
                            "scene_threshold": float(scene_threshold),
                            "min_scene_len": float(min_scene_len),
                            "frame_accurate_split": bool(frame_accurate_split),
                            "per_chunk_cleanup": bool(per_chunk_cleanup),
                        },
                    )
                    if rife_out and Path(rife_out).exists():
                        output_path_local = rife_out
                    elif rife_msg:
                        _emit_progress_line(rife_msg)

                    comp_vid_path, comp_vid_err = maybe_generate_input_vs_output_comparison(
                        runtime_settings.get("_original_input_path_before_preprocess") or input_path_local,
                        output_path_local,
                        seed_controls_local,
                        label_output="RTX Super Resolution",
                        on_progress=(lambda m: _emit_progress_line(m) if m else None),
                    )
                    if comp_vid_path:
                        _emit_progress_line(f"Comparison video created: {comp_vid_path}")
                    elif comp_vid_err:
                        _emit_progress_line(f"Comparison video failed: {comp_vid_err}")

                status = (
                    f"SUCCESS: RTX chunked upscale complete ({int(chunk_count)} chunks)"
                    if int(rc) == 0
                    else f"WARNING: RTX chunked upscale failed (code {int(rc)})"
                )
                if int(rc) != 0 and maybe_set_vram_oom_alert(state, model_label="RTX Super Resolution", text=clog, settings=chunk_settings):
                    status = "OOM: Out of VRAM (GPU) - see banner above"

                _append_run_summary(
                    settings_local=chunk_settings,
                    output_path_val=output_path_local,
                    status_val=status,
                    returncode_val=int(rc),
                )
                _finalize_context(status, output_path_local, int(rc))
                slider_upd, html_upd = _create_comparison_updates(input_path_local, output_path_local)
                return status, clog, output_path_local, html_upd, slider_upd, int(rc)

            # Non-chunk path
            res = run_rtx_superres(
                runtime_settings,
                base_dir=base_dir,
                on_progress=lambda msg: _emit_progress_line(str(msg).strip()),
                cancel_event=cancel_event,
            )

            status = (
                "CANCELED: RTX processing canceled"
                if cancel_event.is_set()
                else (
                    "SUCCESS: RTX upscaling complete"
                    if int(res.returncode) == 0
                    else f"WARNING: RTX upscaling failed (code {int(res.returncode)})"
                )
            )

            output_path_local = res.output_path
            if output_path_local:
                output_path_local = _apply_face_restore_if_enabled(
                    output_path_local, face_apply, face_strength, progress_cb=_emit_progress_line
                )
                output_path_local = _apply_audio_policy(
                    output_path_local,
                    runtime_settings.get("_original_input_path_before_preprocess") or input_path_local,
                    runtime_settings,
                    progress_cb=_emit_progress_line,
                )

            if (
                output_path_local
                and Path(output_path_local).exists()
                and Path(output_path_local).suffix.lower() in video_exts
            ):
                rife_out, rife_msg = maybe_apply_global_rife(
                    runner=runner,
                    output_video_path=output_path_local,
                    seed_controls=seed_controls_local,
                    on_log=(lambda m: _emit_progress_line(m) if m else None),
                )
                if rife_out and Path(rife_out).exists():
                    output_path_local = rife_out
                elif rife_msg:
                    _emit_progress_line(rife_msg)
                comp_vid_path, comp_vid_err = maybe_generate_input_vs_output_comparison(
                    runtime_settings.get("_original_input_path_before_preprocess") or input_path_local,
                    output_path_local,
                    seed_controls_local,
                    label_output="RTX Super Resolution",
                    on_progress=(lambda m: _emit_progress_line(m) if m else None),
                )
                if comp_vid_path:
                    _emit_progress_line(f"Comparison video created: {comp_vid_path}")
                elif comp_vid_err:
                    _emit_progress_line(f"Comparison video failed: {comp_vid_err}")

            if int(res.returncode) != 0 and maybe_set_vram_oom_alert(state, model_label="RTX Super Resolution", text=res.log, settings=runtime_settings):
                status = "OOM: Out of VRAM (GPU) - see banner above"

            _append_run_summary(
                settings_local=runtime_settings,
                output_path_val=output_path_local,
                status_val=status,
                returncode_val=int(res.returncode),
            )
            _finalize_context(status, output_path_local, int(res.returncode))
            slider_upd, html_upd = _create_comparison_updates(input_path_local, output_path_local)
            return status, res.log, output_path_local, html_upd, slider_upd, int(res.returncode)

        def _worker_single(prepped_settings: Dict[str, Any], seed_controls_local: Dict[str, Any]):
            status, log_text, out_path, html_upd, slider_upd, rc = _run_single(prepped_settings, seed_controls_local)
            result_holder["payload"] = (status, log_text, out_path, html_upd, slider_upd, rc)

        def _worker_batch(batch_items: List[str], settings_local: Dict[str, Any], seed_controls_local: Dict[str, Any]):
            from shared.batch_processor import BatchJob, BatchProcessor

            jobs = []
            for item in batch_items:
                jobs.append(BatchJob(input_path=str(item), metadata={"settings": settings_local.copy()}))

            outputs: List[str] = []
            logs: List[str] = []
            last_html = gr.update(value="", visible=False)
            last_slider = gr.update(value=None)

            batch_root = Path(settings_local.get("batch_output_path") or global_settings_local.get("output_dir", output_dir))
            batch_root.mkdir(parents=True, exist_ok=True)
            overwrite_existing = bool(seed_controls_local.get("overwrite_existing_batch_val", False))

            def _process_job(job: BatchJob) -> bool:
                if cancel_event.is_set():
                    return False
                try:
                    item_path = Path(job.input_path)
                    item_kind = detect_input_type(str(item_path))
                    item_settings = settings_local.copy()
                    item_settings["input_path"] = str(item_path)
                    item_settings["batch_enable"] = False
                    item_settings["_original_filename"] = item_path.name

                    if item_kind == "video":
                        run_paths = prepare_batch_video_run_dir(
                            batch_root,
                            item_path.name,
                            input_path=str(item_path),
                            model_label="RTX",
                            mode=str(getattr(runner, "get_mode", lambda: "subprocess")() or "subprocess"),
                            overwrite_existing=overwrite_existing,
                        )
                        if not run_paths:
                            job.status = "skipped"
                            job.output_path = str(batch_item_dir(batch_root, item_path.name))
                            return True
                        item_settings["_run_dir"] = str(run_paths.run_dir)
                        item_settings["_processed_chunks_dir"] = str(run_paths.processed_chunks_dir)
                        item_settings["output_override"] = str(run_paths.run_dir)
                    else:
                        item_dir = batch_item_dir(batch_root, item_path.name)
                        item_dir.mkdir(parents=True, exist_ok=True)
                        item_settings["output_override"] = str(item_dir)
                        item_settings["_run_dir"] = str(item_dir)
                        item_settings["_processed_chunks_dir"] = str(item_dir / "processed_chunks")

                    status, log_text, out_path, html_upd, slider_upd, rc = _run_single(item_settings, seed_controls_local)
                    if out_path:
                        job.output_path = out_path
                        outputs.append(out_path)
                    logs.append(log_text or status)
                    if isinstance(html_upd, dict) and html_upd.get("value"):
                        last_html = html_upd
                    if isinstance(slider_upd, dict) and slider_upd.get("value"):
                        last_slider = slider_upd
                    job.status = "completed" if int(rc) == 0 else "failed"
                    if int(rc) != 0:
                        job.error_message = log_text
                    return int(rc) == 0
                except Exception as e:
                    job.status = "failed"
                    job.error_message = str(e)
                    return False

            processor = BatchProcessor(max_workers=1)
            batch_result = processor.process_batch(jobs=jobs, processor_func=_process_job, max_concurrent=1)
            if outputs:
                _update_last_output_state(seed_controls_local, outputs[-1])
            seed_controls_local["rtx_batch_outputs"] = list(outputs)
            state["seed_controls"] = seed_controls_local

            result_holder["payload"] = (
                f"SUCCESS: Batch complete: {batch_result.completed_files}/{batch_result.total_files} processed ({batch_result.failed_files} failed)",
                "\n\n".join(logs),
                outputs[-1] if outputs else None,
                last_html,
                last_slider,
                0 if batch_result.failed_files == 0 else 1,
            )

        try:
            state = state or {"seed_controls": {}}
            seed_controls = state.get("seed_controls", {}) if isinstance(state, dict) else {}
            if not isinstance(seed_controls, dict):
                seed_controls = {}
                state["seed_controls"] = seed_controls

            seed_controls["rtx_chunk_preview"] = {
                "message": "No chunk preview available yet.",
                "gallery": [],
                "videos": [],
                "count": 0,
            }
            seed_controls["rtx_batch_outputs"] = []
            seed_controls["rtx_last_output_path"] = ""
            state["seed_controls"] = seed_controls

            clear_vram_oom_alert(state)
            cancel_event.clear()

            settings_dict = _rtx_dict_from_args(list(args))
            settings = {**defaults, **settings_dict}

            if preview_only:
                settings["resume_run_dir"] = ""
            if bool(settings.get("batch_enable")):
                settings["resume_run_dir"] = ""

            # Resolve global GPU
            global_gpu = get_global_gpu_override(seed_controls, global_settings_local)
            seed_controls["global_gpu_device_val"] = global_gpu
            settings["device"] = "" if global_gpu == "cpu" else global_gpu

            # Validate CUDA selector for consistency with other tabs
            cuda_device_raw = str(settings.get("device") or "").strip()
            if cuda_device_raw:
                settings["device"] = expand_cuda_device_spec(cuda_device_raw)
            cuda_warn = validate_cuda_device_spec(str(settings.get("device") or ""))
            if cuda_warn:
                yield _build_payload(
                    status=f"WARNING: {cuda_warn}",
                    logs="",
                    progress_upd=gr.update(value="", visible=False),
                    output_path=None,
                    last_processed="CUDA Error",
                    state=state,
                )
                return

            # Input resolve
            original_filename = None
            if str(settings.get("resume_run_dir") or "").strip():
                resume_dir = Path(normalize_path(str(settings.get("resume_run_dir"))))
                if not (resume_dir.exists() and resume_dir.is_dir()):
                    yield _build_payload(
                        status="ERROR: Resume folder not found",
                        logs=f"Configured resume folder does not exist: {resume_dir}",
                        progress_upd=gr.update(value="", visible=False),
                        output_path=None,
                        last_processed="Resume folder invalid",
                        state=state,
                    )
                    return
                recovered_input, recovered_name, _source = resolve_resume_input_from_run_dir(resume_dir)
                if recovered_input is None:
                    yield _build_payload(
                        status="ERROR: Resume input not found",
                        logs=(
                            f"Could not recover input source from resume folder: {resume_dir}. "
                            "Expected run_context.json input path or run_metadata/downscaled artifact."
                        ),
                        progress_upd=gr.update(value="", visible=False),
                        output_path=None,
                        last_processed="Resume input missing",
                        state=state,
                    )
                    return
                raw_input = str(recovered_input)
                original_filename = recovered_name or Path(raw_input).name
            else:
                raw_input = upload if upload else (
                    settings.get("batch_input_path") if bool(settings.get("batch_enable")) else settings.get("input_path")
                )
                if isinstance(raw_input, dict):
                    original_filename = raw_input.get("orig_name") or raw_input.get("name")
                    raw_input = raw_input.get("path") or ""

            input_path_val = normalize_path(str(raw_input)) if raw_input else ""
            settings["_original_filename"] = original_filename or (Path(input_path_val).name if input_path_val else "")

            if bool(settings.get("batch_enable")):
                if not input_path_val or not Path(input_path_val).exists() or not Path(input_path_val).is_dir():
                    yield _build_payload(
                        status="ERROR: Batch input folder missing",
                        logs="",
                        progress_upd=gr.update(value="", visible=False),
                        output_path=None,
                        last_processed="Error",
                        state=state,
                    )
                    return
            else:
                if not input_path_val or not Path(input_path_val).exists():
                    yield _build_payload(
                        status="ERROR: Input missing",
                        logs="",
                        progress_upd=gr.update(value="", visible=False),
                        output_path=None,
                        last_processed="Error",
                        state=state,
                    )
                    return

            if preview_only and bool(settings.get("batch_enable")):
                settings["batch_enable"] = False

            preview_original = input_path_val
            if preview_only:
                preview_src, preview_note = prepare_preview_input(
                    input_path_val,
                    Path(global_settings_local.get("temp_dir", temp_dir)),
                    prefix="rtx",
                    as_single_frame_dir=False,
                )
                if not preview_src:
                    yield _build_payload(
                        status="Preview preparation failed",
                        logs=preview_note or "Failed to prepare preview input.",
                        progress_upd=gr.update(value="", visible=False),
                        output_path=None,
                        last_processed="Preview error",
                        state=state,
                    )
                    return
                input_path_val = normalize_path(preview_src)
                settings["output_format"] = "png"
                if preview_note:
                    _emit_progress_line(preview_note)

            settings["input_path"] = input_path_val
            seed_controls["last_input_path"] = input_path_val
            if preview_only:
                settings["_preview_original_input"] = preview_original
                settings["_preview_effective_input"] = input_path_val
            if bool(settings.get("batch_enable")):
                settings["batch_input_path"] = input_path_val
                batch_out = resolve_batch_output_dir(
                    batch_input_path=input_path_val,
                    batch_output_path=settings.get("batch_output_path"),
                    fallback_output_dir=Path(global_settings_local.get("output_dir", output_dir)),
                    default_subdir_name="upscaled_files",
                )
                settings["batch_output_path"] = str(batch_out)

            current_output_dir = Path(global_settings_local.get("output_dir", output_dir))
            current_temp_dir = Path(global_settings_local.get("temp_dir", temp_dir))
            current_output_dir.mkdir(parents=True, exist_ok=True)
            current_temp_dir.mkdir(parents=True, exist_ok=True)

            # Pre-flight checks
            from shared.error_handling import check_disk_space, check_ffmpeg_available

            ffmpeg_ok, ffmpeg_msg = check_ffmpeg_available()
            if not ffmpeg_ok:
                yield _build_payload(
                    status="ERROR: ffmpeg not found in PATH",
                    logs=ffmpeg_msg or "Install ffmpeg and add to PATH before processing",
                    progress_upd=gr.update(value="", visible=False),
                    output_path=None,
                    last_processed="ffmpeg missing",
                    state=state,
                )
                return
            has_space, space_warning = check_disk_space(current_output_dir, required_mb=5000)
            if not has_space:
                yield _build_payload(
                    status="ERROR: Insufficient disk space",
                    logs=space_warning or "Free up disk space before processing",
                    progress_upd=gr.update(value="", visible=False),
                    output_path=None,
                    last_processed="Low disk space",
                    state=state,
                )
                return

            if progress:
                progress(0, desc="Initializing RTX Super Resolution...")

            # Per-run output folder for single runs
            if not bool(settings.get("batch_enable")) and detect_input_type(settings["input_path"]) in {"video", "image", "directory"}:
                resume_dir_raw = str(settings.get("resume_run_dir") or "").strip()
                if resume_dir_raw:
                    resume_dir = Path(normalize_path(resume_dir_raw))
                    run_dir = resume_dir
                    processed_chunks_dir = run_dir / "processed_chunks"
                    processed_chunks_dir.mkdir(parents=True, exist_ok=True)
                    settings["_run_dir"] = str(run_dir)
                    settings["_processed_chunks_dir"] = str(processed_chunks_dir)
                    settings["_resume_run_requested"] = bool(detect_input_type(settings["input_path"]) == "video")
                    settings["_user_output_override_raw"] = settings.get("output_override") or ""
                    settings["output_override"] = str(run_dir)
                    _emit_progress_line(
                        f"Resume run folder detected: {run_dir}. Processing continues from next remaining chunk."
                    )
                else:
                    try:
                        run_paths, _explicit = prepare_single_video_run(
                            output_root_fallback=current_output_dir,
                            output_override_raw=settings.get("output_override"),
                            input_path=settings["input_path"],
                            original_filename=Path(settings["input_path"]).name,
                            model_label="RTX",
                            mode=str(getattr(runner, "get_mode", lambda: "subprocess")() or "subprocess"),
                        )
                        settings["_run_dir"] = str(run_paths.run_dir)
                        settings["_processed_chunks_dir"] = str(run_paths.processed_chunks_dir)
                        settings["_user_output_override_raw"] = settings.get("output_override") or ""
                        settings["output_override"] = str(run_paths.run_dir)
                    except Exception:
                        pass

            def _worker_single_start():
                prepped = _prepare_single_settings(
                    item_input=settings["input_path"],
                    settings_base=settings,
                    seed_controls_local=seed_controls,
                )
                _worker_single(prepped, seed_controls)

            def _worker_batch_start():
                batch_folder = Path(settings["batch_input_path"])
                entries: List[str] = []
                has_video_or_image_files = False
                try:
                    for child in sorted(batch_folder.iterdir()):
                        if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS.union(IMAGE_EXTENSIONS):
                            entries.append(str(child))
                            has_video_or_image_files = True
                        elif child.is_dir():
                            has_images = any(
                                p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in child.iterdir()
                            )
                            if has_images:
                                entries.append(str(child))
                    if not entries and not has_video_or_image_files:
                        # If folder itself is a frame folder, treat as one item.
                        folder_has_images = any(
                            p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in batch_folder.iterdir()
                        )
                        if folder_has_images:
                            entries = [str(batch_folder)]
                except Exception:
                    entries = []
                if not entries:
                    result_holder["payload"] = (
                        "ERROR: No media files found for batch processing",
                        "",
                        None,
                        gr.update(value="", visible=False),
                        gr.update(value=None),
                        1,
                    )
                    return
                _worker_batch(entries, settings, seed_controls)

            worker = threading.Thread(
                target=_worker_batch_start if bool(settings.get("batch_enable")) else _worker_single_start,
                daemon=True,
            )
            worker.start()

            last_update = time.time()
            run_started = time.time()
            live_logs: List[str] = []
            live_detail = ""
            live_progress = 0.0

            while worker.is_alive() or not progress_q.empty():
                try:
                    line = progress_q.get(timeout=0.15)
                    line_s = str(line or "").strip()
                    if not line_s:
                        continue
                    frame_marker = "FRAME_PROGRESS "
                    if line_s.startswith(frame_marker):
                        detail = line_s[len(frame_marker):].strip()
                        live_detail = detail
                        compact = f"{frame_marker}{detail}"
                        if live_logs and str(live_logs[-1]).startswith(frame_marker):
                            live_logs[-1] = compact
                        else:
                            live_logs.append(compact)
                    else:
                        live_logs.append(line_s)
                        live_detail = line_s

                    if progress:
                        import re

                        nm = re.search(r"(\d+)\s*/\s*(\d+)", line_s)
                        if nm:
                            cur = int(nm.group(1))
                            total = max(1, int(nm.group(2)))
                            live_progress = max(live_progress, float(cur) / float(total))
                            progress(live_progress, desc=line_s[:100])
                        else:
                            pct = re.search(r"(\d+(?:\.\d+)?)\s*%", line_s)
                            if pct:
                                live_progress = max(live_progress, float(pct.group(1)) / 100.0)
                                progress(live_progress, desc=line_s[:100])
                except queue.Empty:
                    pass

                now = time.time()
                if now - last_update > 0.5:
                    last_update = now
                    elapsed = int(now - run_started)
                    subtitle = live_detail or "Waiting for model output..."
                    yield _build_payload(
                        status="RUNNING: RTX Super Resolution in progress",
                        logs="\n".join(live_logs[-400:]),
                        progress_upd=_running_indicator(
                            "RUNNING: RTX Super Resolution in progress",
                            f"{subtitle} | elapsed {elapsed}s",
                            spinning=True,
                        ),
                        output_path=None,
                        last_processed=subtitle,
                        state=state,
                    )
                time.sleep(0.05)

            worker.join()
            payload = result_holder.get("payload")
            if not payload:
                yield _build_payload(
                    status="ERROR: Failed",
                    logs="\n".join(live_logs[-400:]),
                    progress_upd=gr.update(value="", visible=False),
                    output_path=None,
                    last_processed="Error",
                    state=state,
                )
                return

            status, log_text, out_path, html_upd, slider_upd, rc = payload
            merged_logs = str(log_text or "\n".join(live_logs[-400:]))
            if preview_only:
                status_lc = str(status).lower()
                if "fail" in status_lc or "error" in status_lc:
                    status = "Preview failed"
                elif "cancel" in status_lc:
                    status = "Preview canceled"
                else:
                    status = "Preview complete"

            if out_path and Path(out_path).exists():
                _update_last_output_state(seed_controls, out_path)
                state["seed_controls"] = seed_controls
                try:
                    seed_controls["rtx_chunk_preview"] = build_chunk_preview_payload(
                        str(Path(settings.get("_run_dir") or Path(out_path).parent))
                    )
                except Exception:
                    pass

            if int(rc) != 0 and maybe_set_vram_oom_alert(state, model_label="RTX Super Resolution", text=merged_logs, settings=settings):
                show_vram_oom_modal(state, title="Out of VRAM (GPU) - RTX Super Resolution", duration=None)

            if progress:
                if int(rc) == 0 and "error" not in str(status).lower():
                    progress(1.0, desc="RTX Super Resolution complete")
                else:
                    progress(0, desc="Failed")

            yield _build_payload(
                status=str(status),
                logs=merged_logs,
                progress_upd=gr.update(value="", visible=False),
                output_path=out_path,
                last_processed=(f"Output: {out_path}" if out_path else "No output"),
                slider_upd=slider_upd,
                html_upd=html_upd,
                batch_upd=gr.update(value=[], visible=False),
                state=state,
            )

        except Exception as e:
            yield _build_payload(
                status="ERROR: Critical error",
                logs=f"Critical error in RTX processing: {str(e)}",
                progress_upd=gr.update(value="", visible=False),
                output_path=None,
                last_processed="Error",
                state=state or {},
            )

    def cancel_action(state=None):
        cancel_event.set()
        try:
            runner.cancel()
        except Exception:
            pass

        compiled_output: Optional[str] = None
        state_obj = state or {}
        seed_controls = state_obj.get("seed_controls", {}) if isinstance(state_obj, dict) else {}
        last_run_dir = seed_controls.get("last_run_dir")
        output_root = Path(global_settings.get("output_dir", output_dir))
        output_settings = seed_controls.get("output_settings", {}) if isinstance(seed_controls, dict) else {}
        if not isinstance(output_settings, dict):
            output_settings = {}

        try:
            from shared.chunking import salvage_partial_from_run_dir
            from shared.output_run_manager import recent_output_run_dirs

            for run_dir in recent_output_run_dirs(output_root, last_run_dir=str(last_run_dir) if last_run_dir else None, limit=20):
                partial_path, _method = salvage_partial_from_run_dir(
                    run_dir,
                    partial_basename="cancelled_rtx_partial",
                    audio_source=str(seed_controls.get("last_input_path") or "") or None,
                    audio_codec=str(seed_controls.get("audio_codec_val") or "copy"),
                    audio_bitrate=str(seed_controls.get("audio_bitrate_val") or "") or None,
                    encode_settings=output_settings,
                )
                if partial_path and Path(partial_path).exists():
                    compiled_output = str(partial_path)
                    break
        except Exception:
            compiled_output = None

        if compiled_output:
            return (
                gr.update(value=f"Cancelled - Partial RTX output compiled: {Path(compiled_output).name}"),
                f"Partial results saved to: {compiled_output}",
                state_obj,
            )
        return (
            gr.update(value="Cancelled - No partial outputs to compile"),
            "Processing was cancelled before any chunk was completed",
            state_obj,
        )

    def auto_tune_action(
        upload,
        input_path,
        current_quality,
        use_resolution_tab,
        local_upscale_factor,
        max_resolution,
        pre_downscale_then_upscale,
        non_blocking_inference,
        disable_auto_scene_detection_split,
        cuda_stream_ptr,
        state,
    ):
        state = state if isinstance(state, dict) else {"seed_controls": {}}
        seed_controls = state.get("seed_controls", {}) if isinstance(state.get("seed_controls"), dict) else {}
        if not isinstance(seed_controls, dict):
            seed_controls = {}

        def _persist_autotune_settings(selected_quality: str) -> None:
            rtx_settings = seed_controls.get("rtx_settings", {})
            if not isinstance(rtx_settings, dict):
                rtx_settings = {}
            rtx_settings["quality_preset"] = str(selected_quality or current_quality or "HIGHBITRATE_ULTRA")
            rtx_settings["use_resolution_tab"] = bool(use_resolution_tab)
            rtx_settings["upscale_factor"] = float(_to_float(local_upscale_factor, 4.0))
            rtx_settings["max_resolution"] = int(_to_int(max_resolution, 3840))
            rtx_settings["pre_downscale_then_upscale"] = bool(_to_bool(pre_downscale_then_upscale, True))
            rtx_settings["non_blocking_inference"] = bool(_to_bool(non_blocking_inference, True))
            rtx_settings["disable_auto_scene_detection_split"] = bool(_to_bool(disable_auto_scene_detection_split, True))
            rtx_settings["cuda_stream_ptr"] = int(_to_int(cuda_stream_ptr, 0))
            seed_controls["rtx_settings"] = rtx_settings
            seed_controls["preset_dirty"] = True
            state["seed_controls"] = seed_controls

        raw_input = upload if upload else input_path
        if isinstance(raw_input, dict):
            raw_input = raw_input.get("path") or ""
        input_path_val = normalize_path(str(raw_input or ""))
        if not input_path_val or not Path(input_path_val).exists():
            _persist_autotune_settings(str(current_quality or "HIGHBITRATE_ULTRA"))
            return (
                gr.update(value=current_quality),
                "Auto Tune: input path is missing or invalid.",
                state,
            )

        preview_src, preview_note = prepare_preview_input(
            input_path_val,
            Path(global_settings.get("temp_dir", temp_dir)),
            prefix="rtx_autotune",
            as_single_frame_dir=False,
        )
        if not preview_src:
            _persist_autotune_settings(str(current_quality or "HIGHBITRATE_ULTRA"))
            return (
                gr.update(value=current_quality),
                f"Auto Tune failed: {preview_note or 'could not prepare preview input'}",
                state,
            )

        use_global = bool(use_resolution_tab)
        shared_scale = _resolve_shared_upscale_factor(state if use_global else None)
        effective_scale = shared_scale if shared_scale is not None else _to_float(local_upscale_factor, 4.0)
        effective_scale = max(1.0, min(9.9, float(effective_scale)))

        quality_order = [
            "ULTRA",
            "HIGH",
            "MEDIUM",
            "LOW",
            "BICUBIC",
            "HIGHBITRATE_ULTRA",
            "HIGHBITRATE_HIGH",
            "HIGHBITRATE_MEDIUM",
            "HIGHBITRATE_LOW",
            "DENOISE_ULTRA",
            "DENOISE_HIGH",
            "DENOISE_MEDIUM",
            "DENOISE_LOW",
            "DEBLUR_ULTRA",
            "DEBLUR_HIGH",
            "DEBLUR_MEDIUM",
            "DEBLUR_LOW",
        ]

        logs: List[str] = []
        best_quality: Optional[str] = None
        for quality in quality_order:
            trial_settings = {
                "input_path": preview_src,
                "quality_preset": quality,
                "upscale_factor": effective_scale,
                "max_resolution": _to_int(max_resolution, 0),
                "pre_downscale_then_upscale": _to_bool(pre_downscale_then_upscale, True),
                "non_blocking_inference": _to_bool(non_blocking_inference, True),
                "disable_auto_scene_detection_split": _to_bool(disable_auto_scene_detection_split, True),
                "cuda_stream_ptr": _to_int(cuda_stream_ptr, 0),
                "output_format": "png",
                "global_output_dir": str(Path(global_settings.get("temp_dir", temp_dir)) / "rtx_autotune"),
                "image_output_format": "png",
                "image_output_quality": 95,
                "device": "" if get_global_gpu_override(seed_controls, global_settings) == "cpu" else get_global_gpu_override(seed_controls, global_settings),
            }
            try:
                res = run_rtx_superres(trial_settings, base_dir=base_dir, on_progress=None, cancel_event=None)
                if int(res.returncode) == 0 and res.output_path and Path(res.output_path).exists():
                    best_quality = quality
                    logs.append(f"PASS: {quality}")
                    break
                logs.append(f"FAIL: {quality} ({str(res.log).splitlines()[-1] if res.log else 'unknown error'})")
            except Exception as e:
                logs.append(f"FAIL: {quality} ({str(e)})")

        if best_quality:
            _persist_autotune_settings(best_quality)
            return (
                gr.update(value=best_quality),
                f"Auto Tune selected {best_quality} for current VRAM/input constraints.\n" + "\n".join(logs[:12]),
                state,
            )
        _persist_autotune_settings(str(current_quality or "HIGHBITRATE_ULTRA"))
        return (
            gr.update(value=current_quality),
            "Auto Tune could not find a stable preset. Check logs and lower upscale/max resolution.\n" + "\n".join(logs[:12]),
            state,
        )

    def open_outputs_folder_rtx(state: Dict[str, Any]):
        from shared.services.global_service import open_outputs_folder

        live_output_dir = str(global_settings.get("output_dir", output_dir))
        return open_outputs_folder(live_output_dir)

    def clear_temp_folder_rtx(confirm: bool):
        from shared.services.global_service import clear_temp_folder

        live_temp_dir = str(global_settings.get("temp_dir", temp_dir))
        return clear_temp_folder(live_temp_dir, confirm)

    return {
        "defaults": defaults,
        "order": RTX_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "run_action": run_action,
        "cancel_action": lambda *args: cancel_action(args[0] if args else None),
        "auto_tune_action": auto_tune_action,
        "quality_presets": RTX_QUALITY_PRESETS,
        "open_outputs_folder": open_outputs_folder_rtx,
        "clear_temp_folder": clear_temp_folder_rtx,
    }
