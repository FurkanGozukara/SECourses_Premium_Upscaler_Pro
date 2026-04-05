"""RTX Super Resolution tab."""

from __future__ import annotations

import html
import hashlib
import json
import math
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import gradio as gr

from shared.path_utils import (
    detect_input_type,
    get_media_dimensions,
    get_media_duration_seconds,
    get_media_fps,
    normalize_path,
)
from shared.services.rtx_super_resolution_service import RTX_ORDER, build_rtx_super_resolution_callbacks
from shared.video_fps_utils import build_output_fps_summary
from shared.video_comparison_slider import get_video_comparison_js_on_load
from ui.media_preview import preview_updates
from ui.universal_preset_section import universal_preset_section, wire_universal_preset_events


def resolve_shared_upscale_factor(state: Dict[str, Any] | None) -> Optional[float]:
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


def _rtx_scene_override_enabled(
    state: Dict[str, Any] | None,
    override_flag: Optional[bool] = None,
) -> bool:
    if override_flag is not None:
        return bool(override_flag)
    if not isinstance(state, dict):
        return True
    seed_controls = state.get("seed_controls", {}) if isinstance(state, dict) else {}
    rtx_settings = seed_controls.get("rtx_settings", {}) if isinstance(seed_controls, dict) else {}
    if isinstance(rtx_settings, dict):
        if "disable_auto_scene_detection_split" in rtx_settings:
            return bool(rtx_settings.get("disable_auto_scene_detection_split", True))
    return True


def _effective_rtx_scene_flags(
    state: Dict[str, Any] | None,
    override_flag: Optional[bool] = None,
) -> tuple[bool, bool, bool]:
    seed_controls = state.get("seed_controls", {}) if isinstance(state, dict) else {}
    auto_chunk = bool(seed_controls.get("auto_chunk", True))
    auto_detect_scenes = bool(seed_controls.get("auto_detect_scenes", True))
    override_enabled = _rtx_scene_override_enabled(state, override_flag=override_flag)
    if override_enabled:
        return False, False, True
    return auto_chunk, auto_detect_scenes, False


def _build_dimensions_info(path_val: str, use_global: bool, local_scale_x: float, max_edge: int, pre_downscale: bool, quality_preset: str, state: Dict[str, Any] | None) -> str:
    normalized = normalize_path(path_val) if path_val else ""
    if not normalized:
        return ""
    dims = get_media_dimensions(normalized)
    if not dims:
        return "Could not detect input dimensions."

    in_w, in_h = int(dims[0]), int(dims[1])
    shared_scale = resolve_shared_upscale_factor(state if use_global else None)
    req_scale = float(shared_scale if shared_scale is not None else local_scale_x or 2.0)
    req_scale = max(1.0, min(9.9, req_scale))

    quality_upper = str(quality_preset or "HIGHBITRATE_ULTRA").strip().upper()
    same_res_mode = quality_upper.startswith("DENOISE_") or quality_upper.startswith("DEBLUR_")
    if same_res_mode:
        req_scale = 1.0

    long_side = max(in_w, in_h)
    max_edge_val = max(0, int(max_edge or 0))
    cap_ratio = 1.0
    if max_edge_val > 0:
        requested_long = long_side * req_scale
        if requested_long > float(max_edge_val):
            cap_ratio = max(0.01, float(max_edge_val) / float(requested_long))

    preprocess_scale = cap_ratio if (bool(pre_downscale) and cap_ratio < 0.999999) else 1.0
    pre_w = max(1, int(round(in_w * preprocess_scale)))
    pre_h = max(1, int(round(in_h * preprocess_scale)))
    out_w = max(2, int(round(pre_w * req_scale)))
    out_h = max(2, int(round(pre_h * req_scale)))
    if out_w % 2:
        out_w -= 1
    if out_h % 2:
        out_h -= 1
    out_w = max(2, out_w)
    out_h = max(2, out_h)

    lines = [
        "**RTX Sizing Plan**",
        f"- Input: `{in_w}x{in_h}`",
        f"- Preprocess: `{pre_w}x{pre_h}`",
        f"- Output: `{out_w}x{out_h}`",
        f"- Requested scale: `{req_scale:.3g}x`",
    ]
    if max_edge_val > 0:
        lines.append(f"- Max edge cap: `{max_edge_val}`")
    if same_res_mode:
        lines.append("- Same-resolution quality mode detected (`DENOISE/DEBLUR`), scale forced to `1x`.")
    return "\n".join(lines)


def _quality_preset_details(preset_val: str) -> str:
    raw = str(preset_val or "HIGHBITRATE_ULTRA").strip().upper()
    if raw.startswith("DENOISE_"):
        level = raw.split("_", 1)[1] if "_" in raw else "HIGH"
        return (
            f"**{raw}**\n\n"
            f"- Mode: Same-resolution denoise (`{level}`).\n"
            "- Best for noisy/compressed sources where you want cleanup without enlarging.\n"
            "- Upscale is forced to `1x` for this mode."
        )
    if raw.startswith("DEBLUR_"):
        level = raw.split("_", 1)[1] if "_" in raw else "HIGH"
        return (
            f"**{raw}**\n\n"
            f"- Mode: Same-resolution deblur/sharpen (`{level}`).\n"
            "- Best for soft footage that needs clarity recovery at original size.\n"
            "- Upscale is forced to `1x` for this mode."
        )
    if raw.startswith("HIGHBITRATE_"):
        level = raw.split("_", 1)[1] if "_" in raw else "HIGH"
        return (
            f"**{raw}**\n\n"
            f"- Mode: AI upscale tuned for high-bitrate/cleaner sources (`{level}`).\n"
            "- Preserves texture/detail with lighter cleanup than aggressive denoise modes.\n"
            "- Works with normal upscale factors."
        )
    if raw == "BICUBIC":
        return (
            "**BICUBIC**\n\n"
            "- Mode: Non-AI interpolation baseline.\n"
            "- Fastest and lowest VRAM use.\n"
            "- Lower detail recovery than AI quality levels."
        )
    return (
        f"**{raw}**\n\n"
        "- Mode: Standard AI super-resolution.\n"
        "- `LOW/MEDIUM`: faster, less detail recovery.\n"
        "- `HIGH/ULTRA`: best detail quality, higher VRAM/compute."
    )


def _format_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def _probe_video_frame_count(video_path: str) -> tuple[Optional[int], str]:
    path = str(video_path or "").strip()
    if not path:
        return None, "unknown"
    try:
        proc_exact = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if proc_exact.returncode == 0:
            raw_exact = str(proc_exact.stdout or "").strip()
            if raw_exact.isdigit():
                val = int(raw_exact)
                if val > 0:
                    return val, "exact"
    except Exception:
        pass

    try:
        proc_meta = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if proc_meta.returncode == 0:
            raw_meta = str(proc_meta.stdout or "").strip()
            if raw_meta.isdigit():
                val = int(raw_meta)
                if val > 0:
                    return val, "metadata"
    except Exception:
        pass

    try:
        fps_val = float(get_media_fps(path) or 0.0)
        dur_val = float(get_media_duration_seconds(path) or 0.0)
        if fps_val > 0 and dur_val > 0:
            est = int(round(fps_val * dur_val))
            if est > 0:
                return est, "estimated"
    except Exception:
        pass
    return None, "unknown"


def _calculate_scene_frame_stats(scenes: list[tuple[float, float]], fps: Optional[float]) -> Dict[str, Any]:
    if not scenes:
        return {}
    try:
        fps_val = float(fps or 0.0)
    except Exception:
        fps_val = 0.0
    if fps_val <= 0:
        return {"scene_count": len(scenes)}

    frame_counts: list[int] = []
    for start_sec, end_sec in scenes:
        try:
            start_f = float(start_sec)
            end_f = float(end_sec)
        except Exception:
            continue
        if end_f <= start_f:
            continue
        start_frame = int(math.floor(start_f * fps_val + 1e-9))
        end_frame = int(math.ceil(end_f * fps_val - 1e-9))
        if end_frame <= start_frame:
            end_frame = start_frame + 1
        frame_counts.append(max(1, end_frame - start_frame))

    if not frame_counts:
        return {"scene_count": len(scenes)}
    avg_frames = float(sum(frame_counts)) / float(len(frame_counts))
    return {
        "scene_count": len(frame_counts),
        "chunk_min_frames": int(min(frame_counts)),
        "chunk_avg_frames": float(round(avg_frames, 1)),
        "chunk_max_frames": int(max(frame_counts)),
    }


def _estimate_rtx_plan(
    *,
    in_w: int,
    in_h: int,
    scale_x: float,
    max_edge: int,
    pre_down: bool,
    quality: str,
) -> Dict[str, Any]:
    w = max(1, int(in_w))
    h = max(1, int(in_h))
    requested_scale = max(1.0, float(scale_x or 1.0))
    quality_upper = str(quality or "HIGHBITRATE_ULTRA").strip().upper()
    same_res_mode = quality_upper.startswith("DENOISE_") or quality_upper.startswith("DEBLUR_")
    if same_res_mode:
        requested_scale = 1.0

    max_edge_val = max(0, int(max_edge or 0))
    long_side = max(w, h)
    cap_ratio = 1.0
    if max_edge_val > 0:
        requested_long = float(long_side) * float(requested_scale)
        if requested_long > float(max_edge_val):
            cap_ratio = max(0.01, float(max_edge_val) / requested_long)

    use_pre_down = bool(pre_down) and cap_ratio < 0.999999
    preprocess_scale = cap_ratio if use_pre_down else 1.0
    model_scale = requested_scale if use_pre_down else (requested_scale * cap_ratio)
    pre_w = max(1, int(round(float(w) * float(preprocess_scale))))
    pre_h = max(1, int(round(float(h) * float(preprocess_scale))))

    raw_out_w = max(2, int(round(float(pre_w) * float(model_scale))))
    raw_out_h = max(2, int(round(float(pre_h) * float(model_scale))))
    out_w = int(raw_out_w)
    out_h = int(raw_out_h)
    if out_w % 2:
        out_w -= 1
    if out_h % 2:
        out_h -= 1
    out_w = max(2, out_w)
    out_h = max(2, out_h)

    return {
        "input_width": w,
        "input_height": h,
        "requested_scale": float(scale_x or 1.0),
        "effective_scale": float(requested_scale * cap_ratio),
        "model_scale": float(model_scale),
        "same_res_mode": bool(same_res_mode),
        "cap_ratio": float(cap_ratio),
        "preprocess_scale": float(preprocess_scale),
        "preprocess_width": int(pre_w),
        "preprocess_height": int(pre_h),
        "raw_output_width": int(raw_out_w),
        "raw_output_height": int(raw_out_h),
        "output_width": int(out_w),
        "output_height": int(out_h),
    }


def _build_rtx_sizing_report(
    path_val: str,
    use_global: bool,
    local_scale_x: float,
    max_edge: int,
    pre_downscale: bool,
    quality_preset: str,
    disable_auto_scene_detection_split: bool,
    state: Dict[str, Any] | None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> tuple[gr.update, Dict[str, Any]]:
    state_obj: Dict[str, Any] = state if isinstance(state, dict) else {}
    state_obj.setdefault("seed_controls", {})
    seed_controls = state_obj.get("seed_controls", {})
    if not isinstance(seed_controls, dict):
        seed_controls = {}
        state_obj["seed_controls"] = seed_controls

    def _emit_progress(pct: int, note: str = "") -> None:
        if not on_progress:
            return
        try:
            on_progress(max(0, min(100, int(pct))), str(note or ""))
        except Exception:
            pass

    normalized = normalize_path(path_val) if path_val else ""
    if not normalized:
        return gr.update(value="", visible=False), state_obj
    input_path = Path(normalized)
    if not input_path.exists():
        return gr.update(value="Input path not found.", visible=True), state_obj

    _emit_progress(2, "Reading media metadata...")
    dims = get_media_dimensions(str(input_path))
    if not dims:
        return gr.update(value="Could not determine input dimensions.", visible=True), state_obj

    in_w, in_h = int(dims[0]), int(dims[1])
    input_short = min(in_w, in_h)
    input_long = max(in_w, in_h)

    shared_scale = resolve_shared_upscale_factor(state_obj if bool(use_global) else None)
    req_scale = float(shared_scale if shared_scale is not None else local_scale_x or 2.0)
    req_scale = max(1.0, min(9.9, req_scale))

    _emit_progress(18, "Calculating target resize plan...")
    plan = _estimate_rtx_plan(
        in_w=in_w,
        in_h=in_h,
        scale_x=req_scale,
        max_edge=int(max_edge or 0),
        pre_down=bool(pre_downscale),
        quality=str(quality_preset or "HIGHBITRATE_ULTRA"),
    )

    out_w = int(plan["output_width"])
    out_h = int(plan["output_height"])
    out_short = min(out_w, out_h)
    out_long = max(out_w, out_h)
    cap_ratio = float(plan["cap_ratio"])

    input_kind = detect_input_type(str(input_path))
    input_format = (input_path.suffix or "").replace(".", "").upper() if input_path.suffix else "N/A"
    is_video_input = input_kind == "video"

    def _safe(text: Any) -> str:
        return html.escape(str(text))

    def _stat_row(label: str, value: str, value_class: str = "") -> str:
        cls = "resolution-stat-val"
        if value_class:
            cls = f"{cls} {value_class}"
        return (
            '<div class="resolution-stat-row">'
            f'<div class="resolution-stat-key">{_safe(label)}</div>'
            f'<div class="{_safe(cls)}">{_safe(value)}</div>'
            "</div>"
        )

    def _build_card(title: str, rows: list[str]) -> str:
        body = "".join(rows) if rows else _stat_row("Info", "n/a")
        return (
            '<div class="resolution-stat-card">'
            f'<div class="resolution-stat-card-title">{_safe(title)}</div>'
            f"{body}"
            "</div>"
        )

    sizing_rows: list[str] = []
    runtime_rows: list[str] = []
    input_rows: list[str] = []
    chunk_rows: list[str] = []
    notes: list[str] = []

    sizing_rows.append(_stat_row("Input", f"{in_w}x{in_h} (short side: {input_short}px)"))
    target_line = f"upscale {req_scale:g}x"
    max_edge_val = max(0, int(max_edge or 0))
    if max_edge_val > 0:
        target_line += f", max edge {max_edge_val}px"
    if max_edge_val > 0 and cap_ratio < 0.999999:
        target_line += f" (effective {float(plan['effective_scale']):.2f}x)"
    sizing_rows.append(_stat_row("Target Setting", target_line))

    if max_edge_val > 0 and cap_ratio < 0.999999:
        if bool(pre_downscale) and float(plan["preprocess_scale"]) < 0.999999:
            cap_line = (
                f"actual preprocess: downscaled to {int(plan['preprocess_width'])}x{int(plan['preprocess_height'])}px, "
                f"then upscaled {req_scale:g}x"
            )
        else:
            cap_base_w = max(1, int(round(float(out_w) / float(req_scale))))
            cap_base_h = max(1, int(round(float(out_h) / float(req_scale))))
            cap_line = (
                f"equivalent cap-aware base: ~{cap_base_w}x{cap_base_h}px for {req_scale:g}x "
                "(no actual preprocess unless checkbox is ON)"
            )
        sizing_rows.append(_stat_row("Cap-Aware Upscale Path", cap_line))

    if bool(pre_downscale) and float(plan["preprocess_scale"]) < 0.999999:
        sizing_rows.append(
            _stat_row(
                "Preprocess",
                f"{in_w}x{in_h} -> {int(plan['preprocess_width'])}x{int(plan['preprocess_height'])} (x{float(plan['preprocess_scale']):.3f})",
            )
        )

    sizing_rows.append(_stat_row("Resize Result", f"{out_w}x{out_h} (short side: {out_short}px)"))
    if int(plan["raw_output_width"]) != out_w or int(plan["raw_output_height"]) != out_h:
        sizing_rows.append(
            _stat_row(
                "Padded for Runtime (2)",
                f"{int(plan['raw_output_width'])}x{int(plan['raw_output_height'])} -> {out_w}x{out_h} (trimmed to even numbers)",
            )
        )
    sizing_rows.append(_stat_row("Final Saved Output", f"{out_w}x{out_h} (trimmed to even numbers)"))

    mode_class = "is-neutral"
    if out_short < input_short:
        mode_line = f"Downscaling vs original input ({out_short}px < {input_short}px short side)"
        mode_class = "is-down"
        notes.append("Tip: raise Upscale x and/or Max Resolution to avoid downscaling.")
    elif out_short > input_short:
        mode_line = f"Upscaling vs original input ({out_short}px > {input_short}px short side)"
        mode_class = "is-up"
    else:
        mode_line = "Keep size vs original input (output short side matches input)"
    runtime_rows.append(_stat_row("Mode", mode_line, value_class=mode_class))

    quality_upper = str(quality_preset or "HIGHBITRATE_ULTRA").strip().upper()
    runtime_rows.append(_stat_row("RTX Quality", quality_upper))
    if bool(plan["same_res_mode"]):
        runtime_rows.append(_stat_row("RTX Mode", "Same-resolution cleanup (DENOISE/DEBLUR), upscale forced to 1x"))

    if max_edge_val > 0 and cap_ratio < 0.999999:
        runtime_rows.append(
            _stat_row(
                "Actual Preprocess",
                (
                    f"ON: input is pre-downscaled to {int(plan['preprocess_width'])}x{int(plan['preprocess_height'])} before model pass"
                    if (bool(pre_downscale) and float(plan["preprocess_scale"]) < 0.999999)
                    else "OFF: input is not pre-downscaled; effective scale is reduced by max-edge cap"
                ),
            )
        )
        requested_long = int(round(float(input_long) * float(req_scale)))
        runtime_rows.append(
            _stat_row(
                "Max Edge Clamp",
                f"original request ~{requested_long}px long side; final path saved ~{out_long}px (ratio {cap_ratio:.3f})",
            )
        )
    runtime_rows.append(_stat_row("RTX Runtime", f"quality={quality_upper}, output={out_w}x{out_h}"))

    input_rows.append(_stat_row("Input Detected", input_kind.upper()))
    input_rows.append(_stat_row("Format", input_format))

    duration_sec: Optional[float] = None
    fps_val: Optional[float] = None
    total_frames: Optional[int] = None
    frame_count_source = "unknown"

    if is_video_input:
        _emit_progress(42, "Probing duration/FPS/frame stats...")
        cache_key = str(input_path)
        try:
            stat = input_path.stat()
            cache_key = f"{str(input_path)}|{int(stat.st_size)}|{int(stat.st_mtime_ns)}"
        except Exception:
            pass

        video_probe_cache = seed_controls.get("rtx_last_video_probe") or {}
        if video_probe_cache.get("cache_key") == cache_key:
            duration_sec = video_probe_cache.get("duration_sec")
            fps_val = video_probe_cache.get("fps")
            total_frames = video_probe_cache.get("total_frames")
            frame_count_source = str(video_probe_cache.get("frame_count_source") or "unknown")
        else:
            duration_sec = get_media_duration_seconds(str(input_path))
            fps_val = get_media_fps(str(input_path))
            total_frames, frame_count_source = _probe_video_frame_count(str(input_path))
            seed_controls["rtx_last_video_probe"] = {
                "cache_key": cache_key,
                "duration_sec": duration_sec,
                "fps": fps_val,
                "total_frames": total_frames,
                "frame_count_source": frame_count_source,
            }
            state_obj["seed_controls"] = seed_controls

        if duration_sec and duration_sec > 0:
            input_rows.append(_stat_row("Duration", f"{float(duration_sec):.2f}s"))
        else:
            input_rows.append(_stat_row("Duration", "Unavailable"))
        if fps_val and fps_val > 0:
            input_rows.append(_stat_row("FPS", f"{float(fps_val):.3f}".rstrip("0").rstrip(".")))
        else:
            input_rows.append(_stat_row("FPS", "Unavailable"))
        if total_frames is not None and int(total_frames) > 0:
            frame_src_label = {
                "exact": "exact",
                "metadata": "stream metadata",
                "estimated": "estimated",
            }.get(frame_count_source, "unknown")
            input_rows.append(_stat_row("Total Frames", f"{_format_int(total_frames)} ({frame_src_label})"))
        else:
            input_rows.append(_stat_row("Total Frames", "Unavailable"))

        output_settings = seed_controls.get("output_settings", {}) if isinstance(seed_controls, dict) else {}
        if not isinstance(output_settings, dict):
            output_settings = {}
        fps_summary = build_output_fps_summary(
            input_fps=fps_val,
            seed_controls=seed_controls,
            output_settings=output_settings,
        )
        input_rows.append(
            _stat_row(
                str(fps_summary.get("label") or "Output FPS"),
                str(fps_summary.get("value") or "Unavailable"),
                str(fps_summary.get("value_class") or ""),
            )
        )

        _emit_progress(62, "Preparing chunk analysis...")
        auto_chunk, auto_detect_scenes, override_scene_split = _effective_rtx_scene_flags(
            state_obj,
            override_flag=disable_auto_scene_detection_split,
        )
        if auto_chunk:
            scene_threshold = float(seed_controls.get("scene_threshold", 27.0) or 27.0)
            min_scene_len = float(seed_controls.get("min_scene_len", 1.0) or 1.0)
            scan = seed_controls.get("rtx_last_scene_scan") or {}
            scan_path = normalize_path(scan.get("input_path")) if scan.get("input_path") else None
            cached_valid = (
                bool(scan_path)
                and scan_path == str(input_path)
                and abs(float(scan.get("scene_threshold", scene_threshold)) - scene_threshold) < 1e-6
                and abs(float(scan.get("min_scene_len", min_scene_len)) - min_scene_len) < 1e-6
                and "scene_count" in scan
            )
            scene_count = int(scan.get("scene_count", 0) or 0) if cached_valid else 0
            chunk_min_frames = scan.get("chunk_min_frames") if cached_valid else None
            chunk_avg_frames = scan.get("chunk_avg_frames") if cached_valid else None
            chunk_max_frames = scan.get("chunk_max_frames") if cached_valid else None
            scene_scan_error = str(scan.get("error") or "").strip() if cached_valid else ""

            if (not cached_valid) and auto_detect_scenes:
                try:
                    from shared.chunking import detect_scenes

                    _emit_progress(68, "Running scene detection...")

                    def _scene_scan_progress(scene_pct: int) -> None:
                        safe_scene_pct = max(0, min(100, int(scene_pct)))
                        mapped_pct = 68 + int(round((safe_scene_pct / 100.0) * 28.0))
                        _emit_progress(mapped_pct, f"Running scene detection... {safe_scene_pct}%")

                    scenes = detect_scenes(
                        str(input_path),
                        threshold=scene_threshold,
                        min_scene_len=min_scene_len,
                        on_progress_pct=_scene_scan_progress,
                    )
                    scene_stats = _calculate_scene_frame_stats(scenes or [], fps_val)
                    scene_count = int(scene_stats.get("scene_count", len(scenes or [])) or 0)
                    chunk_min_frames = scene_stats.get("chunk_min_frames")
                    chunk_avg_frames = scene_stats.get("chunk_avg_frames")
                    chunk_max_frames = scene_stats.get("chunk_max_frames")
                    scene_scan_error = ""
                    seed_controls["rtx_last_scene_scan"] = {
                        "input_path": str(input_path),
                        "scene_threshold": scene_threshold,
                        "min_scene_len": min_scene_len,
                        "scene_count": scene_count,
                        "chunk_min_frames": chunk_min_frames,
                        "chunk_avg_frames": chunk_avg_frames,
                        "chunk_max_frames": chunk_max_frames,
                        "success": scene_count > 0,
                    }
                    state_obj["seed_controls"] = seed_controls
                except Exception as scene_exc:
                    scene_count = 0
                    chunk_min_frames = None
                    chunk_avg_frames = None
                    chunk_max_frames = None
                    scene_scan_error = str(scene_exc)
                    seed_controls["rtx_last_scene_scan"] = {
                        "input_path": str(input_path),
                        "scene_threshold": scene_threshold,
                        "min_scene_len": min_scene_len,
                        "scene_count": 0,
                        "success": False,
                        "error": str(scene_exc),
                    }
                    state_obj["seed_controls"] = seed_controls
            elif not auto_detect_scenes:
                _emit_progress(74, "Scene detection disabled; using static chunk rules.")
            else:
                _emit_progress(82, "Using cached scene detection results...")

            chunk_rows.append(_stat_row("Chunk Mode", "Auto Scene Detect (PySceneDetect)"))
            chunk_rows.append(_stat_row("Scene Settings", f"threshold={scene_threshold:g}, min_len={min_scene_len:g}s, overlap=0"))
            if scene_count > 0:
                chunk_rows.append(_stat_row("Detected Scenes", _format_int(scene_count)))
                if chunk_min_frames is not None:
                    chunk_rows.append(_stat_row("Min Frames / Chunk", _format_int(chunk_min_frames)))
                if chunk_avg_frames is not None:
                    chunk_rows.append(_stat_row("Avg Frames / Chunk", f"{float(chunk_avg_frames):.1f}"))
                if chunk_max_frames is not None:
                    chunk_rows.append(_stat_row("Max Frames / Chunk", _format_int(chunk_max_frames)))
            elif scene_scan_error:
                chunk_rows.append(_stat_row("Auto Chunk Status", f"Scene scan failed: {scene_scan_error}"))
            elif not auto_detect_scenes:
                chunk_rows.append(_stat_row("Auto Chunk Status", "Auto scene detection is disabled in Resolution tab."))
            else:
                chunk_rows.append(_stat_row("Auto Chunk Status", "Scene scan failed."))
        else:
            if override_scene_split:
                _emit_progress(74, "RTX override disables global scene detection/splitting.")
                chunk_rows.append(_stat_row("Chunk Mode", "Disabled by RTX override"))
                chunk_rows.append(_stat_row("Auto Chunk Status", "Resolution tab scene detection/split is ignored for RTX."))
                chunk_rows.append(_stat_row("Chunking", "Global auto/static chunking disabled for RTX runs."))
            else:
                _emit_progress(74, "Static chunking mode selected.")
            chunk_size = float(seed_controls.get("chunk_size_sec", 0) or 0)
            chunk_overlap = float(seed_controls.get("chunk_overlap_sec", 0) or 0)
            if not override_scene_split:
                chunk_rows.append(_stat_row("Chunk Mode", "Static"))
                if chunk_size <= 0:
                    chunk_rows.append(_stat_row("Chunking", "Disabled (chunk size = 0)"))
                elif chunk_overlap >= chunk_size:
                    chunk_rows.append(_stat_row("Chunking", "Invalid settings: overlap must be smaller than chunk size."))
                else:
                    if duration_sec and duration_sec > 0:
                        est_chunks = math.ceil(float(duration_sec) / max(0.001, chunk_size - chunk_overlap))
                        chunk_rows.append(_stat_row("Estimated Chunks", f"~{_format_int(est_chunks)}"))
                        chunk_rows.append(_stat_row("Window", f"{chunk_size:g}s with {chunk_overlap:g}s overlap"))
                    if fps_val and fps_val > 0:
                        chunk_rows.append(_stat_row("Approx Frames / Chunk", _format_int(int(round(float(chunk_size) * float(fps_val))))))
    else:
        _emit_progress(74, "Input is not a video; scene detection skipped.")
        chunk_rows.append(_stat_row("Chunk Stats", "Chunk frame stats are available for video inputs."))

    left_cards = [
        _build_card("Sizing", sizing_rows),
        _build_card("Runtime", runtime_rows),
    ]
    right_cards = [
        _build_card("Input Stats", input_rows),
        _build_card("Chunk Stats", chunk_rows),
    ]

    notes_html = ""
    if notes:
        note_items = "".join(
            f'<div class="resolution-note-item">{_safe(note)}</div>' for note in notes if str(note).strip()
        )
        if note_items:
            notes_html = f'<div class="resolution-notes">{note_items}</div>'

    html_block = (
        '<div class="resolution-stats-shell">'
        '<div class="resolution-stats-grid">'
        f'<div class="resolution-stats-col">{"".join(left_cards)}</div>'
        f'<div class="resolution-stats-col">{"".join(right_cards)}</div>'
        "</div>"
        f"{notes_html}"
        "</div>"
    )
    _emit_progress(100, "Finalizing analysis report...")
    return gr.update(value=html_block, visible=True), state_obj


def _processing_banner_html(
    state,
    progress_pct: Optional[int] = None,
    progress_note: str = "",
    override_scene_split: Optional[bool] = None,
) -> str:
    auto_chunk, auto_detect_scenes, override_active = _effective_rtx_scene_flags(
        state if isinstance(state, dict) else {},
        override_flag=override_scene_split,
    )
    if auto_chunk and auto_detect_scenes:
        title = "Analyzing input (resolution + scene detection)"
        sub = (
            "PySceneDetect scans the video to find scene cuts; on long videos this can take a while. "
            "Disable <strong>Auto Detect Scenes</strong> in the Resolution tab to speed this up."
        )
    elif override_active:
        title = "Analyzing input (RTX override active)"
        sub = "Resolution tab scene detection and split are disabled for RTX."
    else:
        title = "Analyzing input"
        sub = "Reading media metadata and calculating target sizing."
    if progress_pct is not None:
        safe_pct = max(0, min(100, int(progress_pct)))
        title = f"{title} ({safe_pct}%)"
        if progress_note:
            sub = f"{sub}<br>{progress_note}"
    return (
        '<div class="processing-banner">'
        '<div class="processing-spinner"></div>'
        '<div class="processing-col">'
        f'<div class="processing-text">{title}</div>'
        f'<div class="processing-sub">{sub}</div>'
        "</div></div>"
    )


def _analysis_progress_note(scene_mode: bool, pct: int) -> str:
    safe_pct = max(0, min(100, int(pct)))
    if safe_pct >= 100:
        return "Finalizing analysis report..."
    if scene_mode:
        if safe_pct < 20:
            return "Reading media metadata..."
        if safe_pct < 45:
            return "Calculating target resize plan..."
        if safe_pct < 70:
            return "Probing duration/FPS/frame stats..."
        return "Running scene detection..."
    if safe_pct < 35:
        return "Reading media metadata..."
    if safe_pct < 75:
        return "Calculating target resize plan..."
    return "Building summary cards..."


def rtx_super_resolution_tab(
    preset_manager,
    runner,
    run_logger,
    global_settings: Dict[str, Any],
    shared_state: gr.State,
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path,
):
    service = build_rtx_super_resolution_callbacks(
        preset_manager,
        runner,
        run_logger,
        global_settings,
        shared_state,
        base_dir,
        temp_dir,
        output_dir,
    )

    defaults = service["defaults"]
    seed_controls = shared_state.value.get("seed_controls", {}) if isinstance(shared_state.value, dict) else {}
    rtx_settings = seed_controls.get("rtx_settings", {}) if isinstance(seed_controls, dict) else {}

    merged_defaults = defaults.copy()
    if isinstance(rtx_settings, dict):
        for key, value in rtx_settings.items():
            if value is not None:
                merged_defaults[key] = value
    values = [merged_defaults.get(k) for k in RTX_ORDER]

    def _value(key: str, default=None):
        try:
            idx = RTX_ORDER.index(key)
            raw = values[idx]
            if raw is None and default is not None:
                return default
            return raw
        except Exception:
            return default

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### RTX Super Resolution")

            with gr.Group():
                with gr.Row():
                    with gr.Column(scale=3):
                        input_file = gr.File(label="Upload video or image (optional)", type="filepath", file_types=["video", "image"])
                        input_path = gr.Textbox(label="Input Path", value=_value("input_path", ""), placeholder="C:/path/to/video.mp4 or C:/path/to/frames/")
                        copy_output_into_input_btn = gr.Button("Copy Output Into Input", elem_classes=["action-btn", "action-btn-source-seed"])
                        auto_transfer_output_to_input = gr.Checkbox(label="Auto Transfer Output to Input", value=bool(_value("auto_transfer_output_to_input", False)))
                    with gr.Column(scale=3):
                        input_image_preview = gr.Image(label="Input Preview (Image)", type="filepath", interactive=False, height=220, visible=False)
                        input_video_preview = gr.Video(label="Input Preview (Video)", interactive=False, height=220, visible=False)
                    with gr.Column(scale=2):
                        quality_preset = gr.Dropdown(label="RTX Quality Preset", choices=service.get("quality_presets", []), value=str(_value("quality_preset", "HIGHBITRATE_ULTRA") or "HIGHBITRATE_ULTRA"))
                        quality_preset_details = gr.Markdown(
                            _quality_preset_details(str(_value("quality_preset", "HIGHBITRATE_ULTRA") or "HIGHBITRATE_ULTRA")),
                            visible=True,
                            elem_classes=["resolution-info"],
                        )
                input_cache_msg = gr.Markdown("", visible=False)
                input_detection_result = gr.Markdown("", visible=False)
                sizing_info = gr.Markdown("", visible=False, elem_classes=["resolution-info"])

            with gr.Accordion("RTX Super Resolution Native Streaming (Advanced)", open=True):
                gr.Markdown(
                    "Optional chunk sizing and resume controls for long videos. "
                    "For most users, configure scene/chunking in Resolution tab."
                )
                streaming_chunk_size_frames = gr.Number(
                    label="Streaming Chunk Size (frames, 0=disabled)",
                    value=int(_value("streaming_chunk_size_frames", 0) or 0),
                    precision=0,
                )
                resume_partial_chunks = gr.Checkbox(
                    label="Resume from partial chunks",
                    value=bool(_value("resume_partial_chunks", False)),
                )
            face_restore_after_upscale = gr.Checkbox(
                label="Apply Face Restoration after upscale",
                value=bool(_value("face_restore_after_upscale", False)),
            )
            auto_tune_btn = gr.Button("Auto Tune Max Quality (VRAM Optimized)", elem_classes=["action-btn", "action-btn-source-seed"])
            gr.Markdown(
                (
                    "**How Auto Tune works (RTX Super Resolution):** Tests a short preview input against RTX presets "
                    "from highest to lower quality and picks the first stable one for your current VRAM/input setup. "
                    "It uses your current Upscale x, Max Resolution, pre-downscale mode, non-blocking setting, and CUDA stream pointer."
                ),
                elem_classes=["resolution-info"],
            )
            auto_tune_status = gr.Markdown("", visible=False)

            with gr.Row():
                output_override = gr.Textbox(
                    label="Output Override (single run)",
                    value=_value("output_override", ""),
                    placeholder="Leave empty for auto naming",
                    info="Specify custom output path. Auto-naming creates '_upscaled' files in output folder. Supports both file paths and directories.",
                    scale=3,
                )
                output_format = gr.Dropdown(
                    label="Output Format",
                    choices=["auto", "mp4", "png"],
                    value=str(_value("output_format", "auto") or "auto"),
                    info="'auto' chooses based on input type. 'mp4' for video output. 'png' exports frame sequence. Note: MP4 drops alpha channels.",
                    scale=1,
                )
            with gr.Row():
                open_outputs_btn = gr.Button("Open Outputs Folder", elem_classes=["action-btn", "action-btn-open"])
                clear_temp_btn = gr.Button("Delete Temp Folder", elem_classes=["action-btn", "action-btn-clear"])
            delete_temp_confirm = gr.Checkbox(label="Confirm delete temp folder (required for safety)", value=False)

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
                tab_name="rtx",
                inputs_list=[],
                base_dir=base_dir,
                models_list=seed_controls.get("available_models", ["default"]),
                open_accordion=True,
            )
            image_slider = gr.ImageSlider(
                label="Before / After",
                interactive=False,
                visible=False,
                max_height=1000,
                buttons=["download", "fullscreen"],
                elem_classes=["native-image-comparison-slider"],
            )
            video_comparison_html = gr.HTML(
                label="Video Comparison",
                value="",
                js_on_load=get_video_comparison_js_on_load(),
                visible=False,
            )

        with gr.Column(scale=2):
            gr.Markdown("### Output / Actions")
            status_box = gr.Markdown("Ready for processing.", elem_classes=["runtime-status-box"])
            progress_indicator = gr.Markdown("", visible=False, elem_classes=["runtime-progress-box"])
            with gr.Group():
                use_resolution_tab = gr.Checkbox(
                    label="Use shared Upscale x from Resolution tab",
                    value=bool(_value("use_resolution_tab", True)),
                )
                with gr.Row():
                    upscale_factor = gr.Slider(
                        label="Upscale x (any factor)",
                        minimum=1.0,
                        maximum=9.9,
                        step=0.1,
                        value=float(_value("upscale_factor", 4.0) or 4.0),
                        info="e.g., 4.0 = 4x. Target size is computed from input, then capped by Max Resolution (max edge).",
                        scale=2,
                    )
                    max_resolution = gr.Slider(
                        label="Max Resolution (max edge, 0 = no cap)",
                        minimum=0,
                        maximum=8192,
                        step=16,
                        value=int(_value("max_resolution", 3840) or 3840),
                        info="Caps the LONG side (max(width,height)) of the target. 0 = unlimited.",
                        scale=2,
                    )
                    pre_downscale_then_upscale = gr.Checkbox(
                        label="Pre-downscale then upscale (when capped)",
                        value=bool(_value("pre_downscale_then_upscale", True)),
                        info="If max edge would reduce effective scale, downscale input first so model still runs at full Upscale x.",
                        scale=1,
                    )
                with gr.Row():
                    non_blocking_inference = gr.Checkbox(
                        label="Non-blocking Inference",
                        value=bool(_value("non_blocking_inference", True)),
                        scale=2,
                    )
                    disable_auto_scene_detection_split = gr.Checkbox(
                        label="Disable Auto Scene Detection/Split (RTX Override)",
                        value=bool(_value("disable_auto_scene_detection_split", True)),
                        info="When enabled, RTX ignores Resolution tab scene detection/chunk split settings for this run.",
                        scale=2,
                    )
                    cuda_stream_ptr = gr.Number(
                        label="CUDA Stream Pointer (0 = default)",
                        value=int(_value("cuda_stream_ptr", 0) or 0),
                        info="Advanced: raw CUDA stream handle for external integrations. Use 0 to use the default/internal stream. Leave at 0 unless you explicitly manage CUDA streams.",
                        precision=0,
                        scale=1,
                    )
                with gr.Row():
                    upscale_btn = gr.Button("Upscale", variant="primary", elem_classes=["action-btn", "action-btn-upscale"])
                    preview_btn = gr.Button("First-frame Preview", elem_classes=["action-btn", "action-btn-preview"])
                with gr.Row():
                    cancel_confirm = gr.Checkbox(
                        label="Confirm cancel (subprocess mode only)",
                        value=False,
                        info="Cancel only works in subprocess mode. Check Global Settings to verify mode.",
                        scale=3,
                    )
                    cancel_btn = gr.Button("Cancel (subprocess only)", variant="stop", elem_classes=["action-btn", "action-btn-cancel"], scale=1)
                resume_run_dir = gr.Textbox(
                    label="Resume Run Folder (chunk/scene resume)",
                    value=str(_value("resume_run_dir", "") or ""),
                    placeholder="Optional: G:/.../outputs/0019",
                    info=(
                        "Optional. When set, this run resumes from the last processed chunk in that folder. "
                        "Use the same settings as the original run; processing continues from the next remaining chunk. "
                        "Output override for this run is ignored while resume is active."
                    ),
                )
            with gr.Accordion("Upscaled Output", open=True):
                output_video = gr.Video(label="Upscaled Output (Video)", interactive=False, visible=False, buttons=["download"])
                output_image = gr.Image(label="Upscaled Output (Image)", interactive=False, visible=False, buttons=["download"])

            last_processed = gr.Markdown("Output path will appear here.")
            chunk_status = gr.Markdown("", visible=False)
            with gr.Accordion("Completed Chunks Gallery", open=True):
                gr.Markdown("*Completed chunks appear here during processing. Click a thumbnail to preview the video.*")
                chunk_gallery = gr.Gallery(label="Completed Chunks", visible=False, columns=4, rows=2, height=220, object_fit="cover")
                chunk_preview_video = gr.Video(label="Chunk Preview", interactive=False, visible=False, buttons=["download"])

            batch_gallery = gr.Gallery(label="Batch Results", visible=False, columns=4, rows=2, height="auto", object_fit="contain", buttons=["download"])

            with gr.Accordion("Batch Processing", open=True):
                with gr.Row():
                    batch_enable = gr.Checkbox(
                        label="Enable Batch Processing",
                        value=bool(_value("batch_enable", False)),
                        scale=2,
                    )
                    keep_only_output_files = gr.Checkbox(
                        label="Keep only output files",
                        value=bool(_value("keep_only_output_files", False)),
                        info="After batch completion, remove metadata/chunks/temp artifacts and keep only final outputs.",
                        scale=2,
                    )
                batch_input = gr.Textbox(label="Batch Input Folder", value=_value("batch_input_path", ""))
                batch_output = gr.Textbox(label="Batch Output Folder", value=_value("batch_output_path", ""))

            log_box = gr.Textbox(label="Run Log", value="", lines=14, buttons=["copy"])

    inputs_list = [
        input_path, output_override, batch_enable, keep_only_output_files, batch_input, batch_output,
        quality_preset, use_resolution_tab, upscale_factor, max_resolution,
        pre_downscale_then_upscale, non_blocking_inference, disable_auto_scene_detection_split, cuda_stream_ptr,
        output_format, face_restore_after_upscale, resume_run_dir,
        auto_transfer_output_to_input, streaming_chunk_size_frames, resume_partial_chunks,
    ]

    def _build_input_detection_md(path_val: str) -> gr.update:
        from shared.input_detector import detect_input
        if not path_val or not str(path_val).strip():
            return gr.update(value="", visible=False)
        try:
            info = detect_input(path_val)
            if not info.is_valid:
                return gr.update(value=f"ERROR: **Invalid Input**\n\n{info.error_message}", visible=True)
            parts = [f"OK: **Input Detected: {info.input_type.upper()}**"]
            if info.input_type == "frame_sequence":
                parts.append(f"Pattern: `{info.frame_pattern}`")
                parts.append(f"Frames: {info.frame_start}-{info.frame_end}")
            elif info.input_type == "directory":
                parts.append(f"Files: {info.total_files}")
            elif info.input_type in ["video", "image"]:
                parts.append(f"Format: **{info.format.upper()}**")
            return gr.update(value=" | ".join(parts), visible=True)
        except Exception as e:
            return gr.update(value=f"ERROR: **Detection Error**\n\n{str(e)}", visible=True)

    def _build_sizing_update(path_val, use_global, scale_x, max_edge, pre_down, quality, disable_scene_split, state):
        upd, _ = _build_rtx_sizing_report(
            str(path_val or ""),
            bool(use_global),
            float(scale_x or 2.0),
            int(max_edge or 0),
            bool(pre_down),
            str(quality or "HIGHBITRATE_ULTRA"),
            bool(disable_scene_split),
            state if isinstance(state, dict) else None,
            on_progress=None,
        )
        return upd

    def _iter_sizing_progress(path_val, use_global, scale_x, max_edge, pre_down, quality, disable_scene_split, state):
        result_box: Dict[str, Any] = {}
        error_box: Dict[str, Exception] = {}
        progress_box: "queue.Queue[tuple[int, str]]" = queue.Queue()
        state_obj = state if isinstance(state, dict) else {}
        auto_chunk_local, auto_detect_local, _override_active = _effective_rtx_scene_flags(
            state_obj,
            override_flag=bool(disable_scene_split),
        )
        scene_mode = bool(auto_chunk_local) and bool(auto_detect_local)
        cap_pct = 96 if scene_mode else 92
        speed = 5.0 if scene_mode else 12.0
        poll_sec = 0.12
        real_progress_seen = False
        last_progress_at = time.monotonic()

        def _push_progress(pct: int, note: str = "") -> None:
            nonlocal last_progress_at
            safe_pct = max(0, min(99, int(pct)))
            progress_box.put((safe_pct, str(note or "")))
            last_progress_at = time.monotonic()

        def _worker():
            try:
                result_box["value"] = _build_rtx_sizing_report(
                    str(path_val or ""),
                    bool(use_global),
                    float(scale_x or 2.0),
                    int(max_edge or 0),
                    bool(pre_down),
                    str(quality or "HIGHBITRATE_ULTRA"),
                    bool(disable_scene_split),
                    state_obj,
                    on_progress=_push_progress,
                )
            except Exception as exc:
                error_box["value"] = exc

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        started = time.monotonic()
        last_pct = -1
        last_note = ""
        while worker.is_alive():
            while True:
                try:
                    pct, note = progress_box.get_nowait()
                except queue.Empty:
                    break
                real_progress_seen = True
                if pct > last_pct or (pct == last_pct and note and note != last_note):
                    last_pct = max(last_pct, pct)
                    last_note = note or _analysis_progress_note(scene_mode, last_pct)
                    yield (
                        "progress",
                        gr.update(
                            value=_processing_banner_html(
                                state_obj,
                                last_pct,
                                last_note,
                                override_scene_split=bool(disable_scene_split),
                            ),
                            visible=True,
                        ),
                        state_obj,
                    )
            elapsed = max(0.0, time.monotonic() - started)
            fallback_pct = int(min(cap_pct, elapsed * speed))
            stalled = (time.monotonic() - last_progress_at) > 1.4
            if fallback_pct > last_pct and (not real_progress_seen or stalled):
                last_pct = fallback_pct
                last_note = _analysis_progress_note(scene_mode, fallback_pct)
                yield (
                    "progress",
                    gr.update(
                        value=_processing_banner_html(
                            state_obj,
                            fallback_pct,
                            last_note,
                            override_scene_split=bool(disable_scene_split),
                        ),
                        visible=True,
                    ),
                    state_obj,
                )
            time.sleep(poll_sec)

        worker.join()
        if "value" in error_box:
            msg = f"Analysis failed: {str(error_box['value'])}"
            yield ("done", gr.update(value=msg, visible=True), state_obj)
            return
        final_upd, final_state = result_box.get("value", (gr.update(value="", visible=False), state_obj))
        yield ("done", final_upd, final_state)

    def cache_input(val, use_global, scale_x, max_edge, pre_down, quality, disable_scene_split, state):
        state = state if isinstance(state, dict) else {}
        state.setdefault("seed_controls", {})
        state["seed_controls"]["last_input_path"] = val if val else ""
        det = _build_input_detection_md(val or "")
        img_prev, vid_prev = preview_updates(val)
        for event_type, sizing_upd, next_state in _iter_sizing_progress(
            val,
            use_global,
            scale_x,
            max_edge,
            pre_down,
            quality,
            disable_scene_split,
            state,
        ):
            if event_type == "progress":
                yield (
                    val or "",
                    gr.update(value="Analyzing input...", visible=True),
                    img_prev,
                    vid_prev,
                    det,
                    sizing_upd,
                    next_state,
                )
            else:
                yield (
                    val or "",
                    gr.update(value="OK: Input cached for processing.", visible=True),
                    img_prev,
                    vid_prev,
                    det,
                    sizing_upd,
                    next_state,
                )

    input_file.upload(
        fn=cache_input,
        inputs=[
            input_file,
            use_resolution_tab,
            upscale_factor,
            max_resolution,
            pre_downscale_then_upscale,
            quality_preset,
            disable_auto_scene_detection_split,
            shared_state,
        ],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )

    def clear_on_upload_clear(file_path, state):
        if file_path:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), state
        state = state if isinstance(state, dict) else {}
        state.setdefault("seed_controls", {})
        state["seed_controls"]["last_input_path"] = ""
        img_prev, vid_prev = preview_updates(None)
        return "", gr.update(value="", visible=False), img_prev, vid_prev, gr.update(value="", visible=False), gr.update(value="", visible=False), state

    input_file.change(
        fn=clear_on_upload_clear,
        inputs=[input_file, shared_state],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )

    def update_from_path(val, use_global, scale_x, max_edge, pre_down, quality, disable_scene_split, state):
        state = state if isinstance(state, dict) else {}
        state.setdefault("seed_controls", {})
        state["seed_controls"]["last_input_path"] = val if val else ""
        det = _build_input_detection_md(val or "")
        img_prev, vid_prev = preview_updates(val)
        for event_type, sizing_upd, next_state in _iter_sizing_progress(
            val,
            use_global,
            scale_x,
            max_edge,
            pre_down,
            quality,
            disable_scene_split,
            state,
        ):
            if event_type == "progress":
                yield (
                    gr.update(value="Analyzing input...", visible=True),
                    img_prev,
                    vid_prev,
                    det,
                    sizing_upd,
                    next_state,
                )
            else:
                yield (
                    gr.update(value="OK: Input path updated.", visible=True),
                    img_prev,
                    vid_prev,
                    det,
                    sizing_upd,
                    next_state,
                )

    input_path.submit(
        fn=update_from_path,
        inputs=[
            input_path,
            use_resolution_tab,
            upscale_factor,
            max_resolution,
            pre_downscale_then_upscale,
            quality_preset,
            disable_auto_scene_detection_split,
            shared_state,
        ],
        outputs=[input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )

    def refresh_sizing(scale_x, max_edge, pre_down, use_global, quality, disable_scene_split, path_val, state):
        return _build_sizing_update(path_val or "", use_global, scale_x, max_edge, pre_down, quality, disable_scene_split, state)

    for comp in [pre_downscale_then_upscale, use_resolution_tab, quality_preset, output_format, disable_auto_scene_detection_split]:
        comp.change(
            fn=refresh_sizing,
            inputs=[
                upscale_factor,
                max_resolution,
                pre_downscale_then_upscale,
                use_resolution_tab,
                quality_preset,
                disable_auto_scene_detection_split,
                input_path,
                shared_state,
            ],
            outputs=[sizing_info],
            trigger_mode="always_last",
        )

    quality_preset.change(
        fn=lambda q: gr.update(value=_quality_preset_details(str(q or "HIGHBITRATE_ULTRA")), visible=True),
        inputs=[quality_preset],
        outputs=[quality_preset_details],
        trigger_mode="always_last",
    )

    upscale_factor.release(
        fn=refresh_sizing,
        inputs=[
            upscale_factor,
            max_resolution,
            pre_downscale_then_upscale,
            use_resolution_tab,
            quality_preset,
            disable_auto_scene_detection_split,
            input_path,
            shared_state,
        ],
        outputs=[sizing_info],
        preprocess=False,
        trigger_mode="always_last",
    )

    max_resolution.change(
        fn=refresh_sizing,
        inputs=[
            upscale_factor,
            max_resolution,
            pre_downscale_then_upscale,
            use_resolution_tab,
            quality_preset,
            disable_auto_scene_detection_split,
            input_path,
            shared_state,
        ],
        outputs=[sizing_info],
        trigger_mode="always_last",
    )

    def _sync_upscale_ui(use_global, local_x, state):
        shared_scale = resolve_shared_upscale_factor(state if bool(use_global) else None)
        if bool(use_global) and shared_scale is not None:
            return gr.update(value=min(9.9, max(1.0, float(shared_scale))), interactive=True)
        try:
            local_val = float(local_x)
        except Exception:
            local_val = 2.0
        return gr.update(value=min(9.9, max(1.0, local_val)), interactive=True)

    use_resolution_tab.change(
        fn=_sync_upscale_ui,
        inputs=[use_resolution_tab, upscale_factor, shared_state],
        outputs=[upscale_factor],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    def _on_local_upscale_changed(local_x, use_global, state):
        shared_scale = resolve_shared_upscale_factor(state if bool(use_global) else None)
        return gr.update(value=(False if (bool(use_global) and shared_scale is not None) else bool(use_global)))

    upscale_factor.release(
        fn=_on_local_upscale_changed,
        inputs=[upscale_factor, use_resolution_tab, shared_state],
        outputs=[use_resolution_tab],
        preprocess=False,
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )
    if hasattr(upscale_factor, "input"):
        upscale_factor.input(
            fn=_on_local_upscale_changed,
            inputs=[upscale_factor, use_resolution_tab, shared_state],
            outputs=[use_resolution_tab],
            queue=False,
            show_progress="hidden",
            trigger_mode="always_last",
        )

    def _resolve_latest_output(state) -> str:
        seed_controls_local = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
        candidates = []
        preferred = seed_controls_local.get("rtx_last_output_path")
        if preferred:
            candidates.append(normalize_path(preferred))
        for item in reversed(seed_controls_local.get("rtx_batch_outputs", []) or []):
            candidates.append(normalize_path(item))
        fallback = seed_controls_local.get("last_output_path")
        if fallback:
            candidates.append(normalize_path(fallback))
        seen = set()
        for cand in candidates:
            if not cand or cand in seen:
                continue
            seen.add(cand)
            try:
                if Path(cand).exists():
                    return cand
            except Exception:
                pass
        return ""

    def _output_path_signature(path_val: str) -> str:
        normalized = normalize_path(path_val) if path_val else ""
        if not normalized:
            return ""
        try:
            p = Path(normalized)
            if not p.exists():
                return ""
            st = p.stat()
            if p.is_file():
                return f"file|{normalized}|{int(st.st_size)}|{int(st.st_mtime_ns)}"
            return f"dir|{normalized}|{int(st.st_mtime_ns)}"
        except Exception:
            return ""

    def _apply_output_to_input(
        output_path_val,
        use_global,
        scale_x,
        max_edge,
        pre_down,
        quality,
        disable_scene_split,
        state,
        message: str,
    ):
        state = state if isinstance(state, dict) else {}
        state.setdefault("seed_controls", {})
        state["seed_controls"]["last_input_path"] = output_path_val if output_path_val else ""
        det = _build_input_detection_md(output_path_val or "")
        info, state = _build_rtx_sizing_report(
            str(output_path_val or ""),
            bool(use_global),
            float(scale_x or 2.0),
            int(max_edge or 0),
            bool(pre_down),
            str(quality or "HIGHBITRATE_ULTRA"),
            bool(disable_scene_split),
            state,
            on_progress=None,
        )
        img_prev, vid_prev = preview_updates(output_path_val)
        return output_path_val or "", gr.update(value=message, visible=True), img_prev, vid_prev, det, info, state

    def copy_latest_output_to_input(use_global, scale_x, max_edge, pre_down, quality, disable_scene_split, state):
        outp = _resolve_latest_output(state)
        if not outp:
            return gr.skip(), gr.update(value="No generated RTX output found to transfer.", visible=True), gr.skip(), gr.skip(), gr.skip(), gr.skip(), state
        return _apply_output_to_input(
            outp,
            use_global,
            scale_x,
            max_edge,
            pre_down,
            quality,
            disable_scene_split,
            state,
            "Output path copied into input.",
        )

    copy_output_into_input_btn.click(
        fn=copy_latest_output_to_input,
        inputs=[
            use_resolution_tab,
            upscale_factor,
            max_resolution,
            pre_downscale_then_upscale,
            quality_preset,
            disable_auto_scene_detection_split,
            shared_state,
        ],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )

    def refresh_chunk_preview_ui(state):
        preview = (state or {}).get("seed_controls", {}).get("rtx_chunk_preview", {})
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
        return gr.update(value=message, visible=bool(message or gallery)), gr.update(value=gallery, visible=bool(gallery)), gr.update(value=first_video, visible=bool(first_video))

    def on_chunk_gallery_select(evt: gr.SelectData, state):
        try:
            idx = int(evt.index)
            videos = (state or {}).get("seed_controls", {}).get("rtx_chunk_preview", {}).get("videos", [])
            if 0 <= idx < len(videos):
                cand = videos[idx]
                if cand and Path(cand).exists():
                    return gr.update(value=cand, visible=True)
        except Exception:
            pass
        return gr.update(value=None, visible=False)

    chunk_gallery.select(fn=on_chunk_gallery_select, inputs=[shared_state], outputs=[chunk_preview_video])

    def _expand_service_payload(payload, live_state):
        safe_state = live_state if isinstance(live_state, dict) else {}
        if not isinstance(payload, tuple) or len(payload) < 10:
            chunk_status_upd, chunk_gallery_upd, chunk_preview_upd = refresh_chunk_preview_ui(safe_state)
            return (
                gr.update(value="ERROR: Invalid RTX payload"),
                "",
                gr.update(value="", visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                "Error",
                gr.update(value=None),
                gr.update(value="", visible=False),
                gr.update(value=[], visible=False),
                chunk_status_upd,
                chunk_gallery_upd,
                chunk_preview_upd,
                safe_state,
            )

        status, logs, prog_upd, img_upd, vid_upd, last_txt, slider_upd, html_upd, batch_upd, state_out = payload[:10]
        chunk_status_upd, chunk_gallery_upd, chunk_preview_upd = refresh_chunk_preview_ui(state_out)
        return (
            status,
            logs,
            prog_upd if prog_upd is not None else gr.update(value="", visible=False),
            img_upd if img_upd is not None else gr.update(value=None, visible=False),
            vid_upd if vid_upd is not None else gr.update(value=None, visible=False),
            last_txt,
            slider_upd if slider_upd is not None else gr.update(value=None),
            html_upd if html_upd is not None else gr.update(value="", visible=False),
            batch_upd if batch_upd is not None else gr.update(value=[], visible=False),
            chunk_status_upd,
            chunk_gallery_upd,
            chunk_preview_upd,
            state_out,
        )

    def run_upscale(upload, *args, progress=gr.Progress()):
        live_state = args[-1] if (args and isinstance(args[-1], dict)) else {}
        for payload in service["run_action"](upload, *args[:-1], preview_only=False, state=live_state, progress=progress):
            yield _expand_service_payload(payload, live_state)

    def run_preview(upload, *args, progress=gr.Progress()):
        live_state = args[-1] if (args and isinstance(args[-1], dict)) else {}
        for payload in service["run_action"](upload, *args[:-1], preview_only=True, state=live_state, progress=progress):
            yield _expand_service_payload(payload, live_state)

    rtx_pre_run_output_signature = gr.State(value="")

    def capture_latest_output_signature(state):
        return _output_path_signature(_resolve_latest_output(state))

    def auto_transfer_latest_output_to_input(
        use_global,
        scale_x,
        max_edge,
        pre_down,
        quality,
        disable_scene_split,
        auto_enabled,
        previous_signature,
        state,
    ):
        state = state if isinstance(state, dict) else {}
        if not bool(auto_enabled):
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), state
        outp = _resolve_latest_output(state)
        if not outp:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), state
        sig = _output_path_signature(outp)
        if previous_signature and sig and str(previous_signature) == str(sig):
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), state
        return _apply_output_to_input(
            outp,
            use_global,
            scale_x,
            max_edge,
            pre_down,
            quality,
            disable_scene_split,
            state,
            "Auto-transferred latest output into input.",
        )

    upscale_btn.click(
        fn=capture_latest_output_signature,
        inputs=[shared_state],
        outputs=[rtx_pre_run_output_signature],
        queue=False,
        show_progress="hidden",
    )

    run_evt = upscale_btn.click(
        fn=run_upscale,
        inputs=[input_file] + inputs_list + [shared_state],
        outputs=[
            status_box, log_box, progress_indicator, output_image, output_video,
            last_processed, image_slider, video_comparison_html, batch_gallery,
            chunk_status, chunk_gallery, chunk_preview_video, shared_state,
        ],
        concurrency_limit=32,
        concurrency_id="app_processing_queue",
        trigger_mode="multiple",
    )

    preview_btn.click(
        fn=run_preview,
        inputs=[input_file] + inputs_list + [shared_state],
        outputs=[
            status_box, log_box, progress_indicator, output_image, output_video,
            last_processed, image_slider, video_comparison_html, batch_gallery,
            chunk_status, chunk_gallery, chunk_preview_video, shared_state,
        ],
    )

    run_evt.then(
        fn=auto_transfer_latest_output_to_input,
        inputs=[
            use_resolution_tab, upscale_factor, max_resolution, pre_downscale_then_upscale,
            quality_preset, disable_auto_scene_detection_split, auto_transfer_output_to_input, rtx_pre_run_output_signature, shared_state,
        ],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )

    auto_tune_btn.click(
        fn=service["auto_tune_action"],
        inputs=[
            input_file, input_path, quality_preset, use_resolution_tab, upscale_factor,
            max_resolution, pre_downscale_then_upscale, non_blocking_inference,
            disable_auto_scene_detection_split, cuda_stream_ptr, shared_state,
        ],
        outputs=[quality_preset, auto_tune_status, shared_state],
    ).then(fn=lambda: gr.update(visible=True), outputs=[auto_tune_status])

    cancel_btn.click(
        fn=lambda ok, state: service["cancel_action"](state) if ok else (gr.update(value="Enable confirm cancel first."), "", state),
        inputs=[cancel_confirm, shared_state],
        outputs=[status_box, log_box, shared_state],
    )

    open_outputs_btn.click(
        fn=lambda state: (service["open_outputs_folder"](state), state),
        inputs=shared_state,
        outputs=[status_box, shared_state],
    )

    clear_temp_btn.click(
        fn=lambda confirm: service["clear_temp_folder"](bool(confirm)),
        inputs=[delete_temp_confirm],
        outputs=[status_box],
    )

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
        tab_name="rtx",
    )

    rtx_sync_signature = gr.State(value="")

    def _sync_signature(payload: Dict[str, Any]) -> str:
        try:
            blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str, separators=(",", ":"))
        except Exception:
            blob = str(payload)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def _sync_upscale_and_sizing_if_needed(
        use_global,
        local_x,
        max_edge,
        pre_down,
        quality,
        disable_scene_split,
        path_val,
        state,
        previous_signature: str = "",
    ):
        signature = _sync_signature(
            {
                "use_global": bool(use_global),
                "local_x": local_x,
                "max_edge": max_edge,
                "pre_down": bool(pre_down),
                "quality": quality,
                "disable_scene_split": bool(disable_scene_split),
                "path_val": path_val,
                "shared_scale": resolve_shared_upscale_factor(state if bool(use_global) else None),
                "resolution_settings": ((state or {}).get("seed_controls", {}) or {}).get("resolution_settings", {}),
            }
        )
        if signature == str(previous_signature or ""):
            return gr.skip(), gr.skip(), previous_signature
        slider_upd = _sync_upscale_ui(use_global, local_x, state)
        sizing_upd = _build_sizing_update(
            path_val or "",
            use_global,
            local_x,
            max_edge,
            pre_down,
            quality,
            disable_scene_split,
            state,
        )
        return slider_upd, sizing_upd, signature

    shared_state.change(
        fn=_sync_upscale_and_sizing_if_needed,
        inputs=[
            use_resolution_tab, upscale_factor, max_resolution, pre_downscale_then_upscale,
            quality_preset, disable_auto_scene_detection_split, input_path, shared_state, rtx_sync_signature,
        ],
        outputs=[upscale_factor, sizing_info, rtx_sync_signature],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    return {
        "inputs_list": inputs_list,
        "preset_dropdown": preset_dropdown,
        "preset_status": preset_status,
    }
