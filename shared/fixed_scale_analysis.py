"""
SeedVR2-style sizing/runtime/input cards for fixed-scale pipelines.

Used by GAN and FlashVSR+ tabs so they can expose the same analysis flow as
SeedVR2 while keeping model-specific runtime details.
"""

from __future__ import annotations

import html
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from .input_detector import detect_input
from .path_utils import (
    VIDEO_EXTENSIONS,
    detect_input_type,
    get_media_dimensions,
    get_media_duration_seconds,
    get_media_fps,
    normalize_path,
)
from .resolution_calculator import estimate_fixed_scale_upscale_plan_from_dims


def detect_scenes(*args, **kwargs):
    """Lazy-import scene detection so fixed-scale UI cards stay OpenCV-free at startup."""
    from .chunking import detect_scenes as _detect_scenes_impl

    return _detect_scenes_impl(*args, **kwargs)


def _safe(text: Any) -> str:
    return html.escape(str(text))


def _format_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "n/a"


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


def _build_card(title: str, rows: List[str]) -> str:
    body = "".join(rows) if rows else _stat_row("Info", "n/a")
    return (
        '<div class="resolution-stat-card">'
        f'<div class="resolution-stat-card-title">{_safe(title)}</div>'
        f"{body}"
        "</div>"
    )


def _probe_video_frame_count(video_path: str) -> Tuple[Optional[int], str]:
    """
    Return (frame_count, source).
    source is one of: exact | metadata | estimated | unknown.
    """

    def _parse_positive_int(raw_text: str) -> Optional[int]:
        raw = str(raw_text or "").strip()
        if not raw:
            return None
        for line in raw.splitlines():
            token = line.strip()
            if token.isdigit():
                val = int(token)
                if val > 0:
                    return val
        return None

    try:
        proc = subprocess.run(
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
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            val = _parse_positive_int(proc.stdout)
            if val is not None:
                return val, "exact"
    except Exception:
        pass

    try:
        proc = subprocess.run(
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
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            val = _parse_positive_int(proc.stdout)
            if val is not None:
                return val, "metadata"
    except Exception:
        pass

    try:
        duration = get_media_duration_seconds(str(video_path))
        fps = get_media_fps(str(video_path))
        if duration and duration > 0 and fps and fps > 0:
            return max(1, int(round(float(duration) * float(fps)))), "estimated"
    except Exception:
        pass

    return None, "unknown"


def _calculate_scene_frame_stats(
    scenes: List[Tuple[float, float]],
    fps: Optional[float],
    total_frames: Optional[int] = None,
) -> Dict[str, Any]:
    if not scenes:
        return {}

    normalized_scenes: List[Tuple[float, float]] = []
    for start_sec, end_sec in scenes:
        try:
            start_f = float(start_sec)
            end_f = float(end_sec)
        except Exception:
            continue
        if end_f <= start_f:
            continue
        normalized_scenes.append((start_f, end_f))

    if not normalized_scenes:
        return {}

    scene_count = len(normalized_scenes)
    total_frames_val: Optional[int] = None
    try:
        if total_frames is not None:
            tf = int(total_frames)
            if tf > 0:
                total_frames_val = tf
    except Exception:
        total_frames_val = None

    # Prefer total-frame-aware stats so chunk frame totals stay consistent with Input Stats.
    if total_frames_val is not None:
        durations: List[float] = [max(0.0, end_f - start_f) for start_f, end_f in normalized_scenes]
        total_duration = float(sum(durations))
        if total_duration > 0.0:
            baseline = 1 if total_frames_val >= scene_count else 0
            frame_counts: List[int] = [baseline for _ in range(scene_count)]
            remaining = max(0, total_frames_val - (baseline * scene_count))

            if remaining > 0:
                raw = [remaining * (d / total_duration) for d in durations]
                integer = [int(math.floor(v)) for v in raw]
                fractions = [v - i for v, i in zip(raw, integer)]
                frame_counts = [c + i for c, i in zip(frame_counts, integer)]

                leftover = remaining - sum(integer)
                if leftover > 0:
                    order = sorted(range(scene_count), key=lambda idx: fractions[idx], reverse=True)
                    for idx in order[:leftover]:
                        frame_counts[idx] += 1

            # Final correction for safety: force exact sum match.
            delta = total_frames_val - sum(frame_counts)
            if delta > 0:
                for i in range(delta):
                    frame_counts[i % scene_count] += 1
            elif delta < 0:
                to_remove = -delta
                order = sorted(range(scene_count), key=lambda idx: frame_counts[idx], reverse=True)
                floor_val = 1 if total_frames_val >= scene_count else 0
                for idx in order:
                    if to_remove <= 0:
                        break
                    removable = max(0, frame_counts[idx] - floor_val)
                    if removable <= 0:
                        continue
                    take = min(removable, to_remove)
                    frame_counts[idx] -= take
                    to_remove -= take

            avg_frames = float(sum(frame_counts)) / float(len(frame_counts))
            return {
                "scene_count": len(frame_counts),
                "chunk_min_frames": int(min(frame_counts)),
                "chunk_avg_frames": float(round(avg_frames, 1)),
                "chunk_max_frames": int(max(frame_counts)),
                "frame_stats_method": "proportional_total_frames_v1",
            }

    # Fallback when total-frame information is unavailable.
    try:
        fps_val = float(fps or 0.0)
    except Exception:
        fps_val = 0.0

    if fps_val <= 0:
        return {"scene_count": scene_count}

    frame_counts = []
    for start_f, end_f in normalized_scenes:
        start_frame = int(math.floor(start_f * fps_val + 1e-9))
        end_frame = int(math.ceil(end_f * fps_val - 1e-9))
        if end_frame <= start_frame:
            end_frame = start_frame + 1
        frame_counts.append(max(1, end_frame - start_frame))

    if not frame_counts:
        return {"scene_count": scene_count}

    avg_frames = float(sum(frame_counts)) / float(len(frame_counts))
    return {
        "scene_count": len(frame_counts),
        "chunk_min_frames": int(min(frame_counts)),
        "chunk_avg_frames": float(round(avg_frames, 1)),
        "chunk_max_frames": int(max(frame_counts)),
        "frame_stats_method": "fps_timecode_v1",
    }


def _resolve_dims_for_preview(path_val: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    if not path_val:
        return None, None, None

    p = Path(normalize_path(path_val))
    if not p.exists():
        return None, None, None

    if p.is_file():
        dims = get_media_dimensions(str(p))
        return (dims[0], dims[1], str(p)) if dims else (None, None, str(p))

    # Directory: pick a representative file for dimension probing.
    items: List[Path] = []
    try:
        for item in sorted(p.iterdir()):
            if item.is_file():
                items.append(item)
    except Exception:
        return None, None, None

    if not items:
        return None, None, None

    # Prefer images first; if none, try videos.
    rep = None
    for item in items:
        if item.suffix.lower() not in VIDEO_EXTENSIONS:
            rep = item
            break
    if rep is None:
        rep = items[0]

    dims = get_media_dimensions(str(rep))
    return (dims[0], dims[1], str(rep)) if dims else (None, None, str(rep))


def build_fixed_scale_analysis_update(
    *,
    input_path_val: str,
    model_scale: int,
    use_global: bool,
    local_scale_x: float,
    local_max_edge: int,
    local_pre_down: bool,
    state: Optional[Dict[str, Any]],
    model_label: str,
    runtime_label: str,
    auto_scene_scan: bool = True,
) -> gr.update:
    """
    Build SeedVR2-style analysis cards for fixed-scale pipelines.
    """
    if not input_path_val or not str(input_path_val).strip():
        return gr.update(value="", visible=False)

    state = state or {}
    state.setdefault("seed_controls", {})
    seed_controls = state.get("seed_controls", {})

    try:
        ms = max(2, int(model_scale or 4))
    except Exception:
        ms = 4

    if use_global:
        # Global Resolution tab can still provide shared upscale ratio,
        # but max-edge capping is now local per upscaler tab.
        scale_x = float(seed_controls.get("upscale_factor_val", local_scale_x or 4.0) or (local_scale_x or 4.0))
    else:
        scale_x = float(local_scale_x or 4.0)
    max_edge = int(local_max_edge or 0)
    pre_down = bool(local_pre_down)

    w, h, rep = _resolve_dims_for_preview(input_path_val)
    if not w or not h:
        return gr.update(value="Could not determine input dimensions for sizing preview.", visible=True)

    plan = estimate_fixed_scale_upscale_plan_from_dims(
        int(w),
        int(h),
        requested_scale=float(scale_x),
        model_scale=int(ms),
        max_edge=int(max_edge or 0),
        force_pre_downscale=True,
    )

    out_w = int(plan.final_saved_width or plan.resize_width)
    out_h = int(plan.final_saved_height or plan.resize_height)
    out_short = min(out_w, out_h)
    out_long = max(out_w, out_h)
    input_short = min(plan.input_width, plan.input_height)
    input_long = max(plan.input_width, plan.input_height)

    # Input detection (format/type/details) on the original user path.
    input_info = detect_input(input_path_val)
    input_kind = str(input_info.input_type or detect_input_type(str(rep or input_path_val)))
    input_format = (
        str(input_info.format).upper()
        if input_info and input_info.format
        else (Path(rep or input_path_val).suffix.replace(".", "").upper() or "N/A")
    )

    sizing_rows: List[str] = []
    runtime_rows: List[str] = []
    input_rows: List[str] = []
    chunk_rows: List[str] = []
    notes: List[str] = []

    sizing_rows.append(_stat_row("Input", f"{plan.input_width}x{plan.input_height} (short side: {input_short}px)"))

    target_line = f"upscale {scale_x:g}x"
    if use_global and seed_controls.get("upscale_factor_val") is not None:
        target_line += " (shared scale)"
    if max_edge and max_edge > 0:
        target_line += f", max edge {max_edge}px"
        if plan.cap_ratio < 0.999999:
            target_line += f" (effective {plan.effective_scale:.2f}x)"
    sizing_rows.append(_stat_row("Target Setting", target_line))

    if max_edge and max_edge > 0 and plan.cap_ratio < 0.999999:
        if plan.pre_downscale_then_upscale and plan.preprocess_scale < 0.999999:
            cap_path = (
                f"actual preprocess: downscaled to {plan.preprocess_width}x{plan.preprocess_height}px, "
                f"then upscaled {ms}x"
            )
        else:
            cap_base_w = max(1, int(round(float(out_w) / float(ms))))
            cap_base_h = max(1, int(round(float(out_h) / float(ms))))
            cap_path = f"equivalent base: ~{cap_base_w}x{cap_base_h}px for fixed {ms}x pass"
        sizing_rows.append(_stat_row("Cap-Aware Upscale Path", cap_path))

    if plan.pre_downscale_then_upscale and plan.preprocess_scale < 0.999999:
        sizing_rows.append(
            _stat_row(
                "Preprocess",
                f"{plan.input_width}x{plan.input_height} -> {plan.preprocess_width}x{plan.preprocess_height} (x{plan.preprocess_scale:.3f})",
            )
        )

    resized_short = min(plan.resize_width, plan.resize_height)
    sizing_rows.append(
        _stat_row("Resize Result", f"{plan.resize_width}x{plan.resize_height} (short side: {resized_short}px)")
    )

    if plan.padded_width and plan.padded_height:
        sizing_rows.append(
            _stat_row(
                "Padded for Model (16)",
                f"{plan.padded_width}x{plan.padded_height} (trimmed after processing)",
            )
        )

    sizing_rows.append(_stat_row("Final Saved Output", f"{out_w}x{out_h} (trimmed to even numbers)"))

    mode_class = "is-neutral"
    if out_short < input_short:
        mode_line = f"Downscaling vs original input ({out_short}px < {input_short}px short side)"
        mode_class = "is-down"
        notes.append("Tip: increase Upscale x and/or Max Resolution to avoid downscaling.")
    elif out_short > input_short:
        mode_line = f"Upscaling vs original input ({out_short}px > {input_short}px short side)"
        mode_class = "is-up"
    else:
        mode_line = "Keep size vs original input (output short side matches input)"
    runtime_rows.append(_stat_row("Mode", mode_line, value_class=mode_class))

    if max_edge and max_edge > 0 and plan.cap_ratio < 0.999999:
        if plan.pre_downscale_then_upscale and plan.preprocess_scale < 0.999999:
            runtime_rows.append(
                _stat_row(
                    "Actual Preprocess",
                    f"ON: input is pre-downscaled to {plan.preprocess_width}x{plan.preprocess_height} before model pass",
                )
            )
        else:
            runtime_rows.append(_stat_row("Actual Preprocess", "OFF: no input pre-downscale before model pass"))

        requested_long = int(round(input_long * scale_x))
        clamp_line = (
            f"requested ~{requested_long}px long side, final ~{out_long}px "
            f"(ratio {plan.cap_ratio:.3f})"
        )
        runtime_rows.append(_stat_row("Max Edge Clamp", clamp_line))

    runtime_rows.append(_stat_row("Pipeline", runtime_label))
    runtime_rows.append(_stat_row("Model Runtime", f"{model_label} fixed {ms}x"))

    if plan.notes:
        notes.extend([str(n) for n in plan.notes if str(n).strip()])

    input_rows.append(_stat_row("Input Detected", str(input_kind).upper()))
    input_rows.append(_stat_row("Format", input_format))

    duration_sec: Optional[float] = None
    fps_val: Optional[float] = None
    total_frames: Optional[int] = None
    frame_count_source = "unknown"

    if input_kind == "video":
        target_for_stats = rep or input_path_val
        duration_sec = get_media_duration_seconds(str(target_for_stats))
        fps_val = get_media_fps(str(target_for_stats))
        total_frames, frame_count_source = _probe_video_frame_count(str(target_for_stats))

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
    elif input_kind == "frame_sequence":
        input_rows.append(_stat_row("Frames", f"{_format_int(input_info.frame_count)}"))
        if input_info.frame_start or input_info.frame_end:
            input_rows.append(_stat_row("Frame Range", f"{input_info.frame_start}-{input_info.frame_end}"))
        if input_info.missing_frames:
            input_rows.append(_stat_row("Missing Frames", _format_int(len(input_info.missing_frames))))
    elif input_kind == "directory":
        input_rows.append(_stat_row("Files", _format_int(input_info.total_files)))

    # Chunk stats card
    if input_kind == "video":
        auto_chunk = bool(seed_controls.get("auto_chunk", True))
        if auto_chunk:
            scene_threshold = float(seed_controls.get("scene_threshold", 27.0) or 27.0)
            min_scene_len = float(seed_controls.get("min_scene_len", 1.0) or 1.0)
            auto_detect_scenes = bool(seed_controls.get("auto_detect_scenes", True))

            chunk_rows.append(_stat_row("Chunk Mode", "Auto Scene Detect (PySceneDetect)"))
            chunk_rows.append(
                _stat_row(
                    "Scene Settings",
                    f"threshold={scene_threshold:g}, min_len={min_scene_len:g}s, overlap=0",
                )
            )

            scan = seed_controls.get("last_scene_scan") or {}
            scan_path = normalize_path(scan.get("input_path")) if scan.get("input_path") else None
            cached_valid = (
                bool(scan_path)
                and scan_path == normalize_path(input_path_val)
                and abs(float(scan.get("scene_threshold", scene_threshold)) - scene_threshold) < 1e-6
                and abs(float(scan.get("min_scene_len", min_scene_len)) - min_scene_len) < 1e-6
                and "scene_count" in scan
            )
            if cached_valid and total_frames is not None:
                try:
                    cached_total_frames = int(scan.get("total_frames"))
                    if cached_total_frames != int(total_frames):
                        cached_valid = False
                except Exception:
                    cached_valid = False
                if (
                    cached_valid
                    and int(scan.get("scene_count", 0) or 0) > 0
                    and not str(scan.get("frame_stats_method") or "").strip()
                ):
                    # Invalidate legacy cached stats that predate total-frame-aware method.
                    cached_valid = False

            scene_count = int(scan.get("scene_count", 0) or 0) if cached_valid else 0
            chunk_min_frames = scan.get("chunk_min_frames") if cached_valid else None
            chunk_avg_frames = scan.get("chunk_avg_frames") if cached_valid else None
            chunk_max_frames = scan.get("chunk_max_frames") if cached_valid else None
            scene_scan_error = str(scan.get("error") or "").strip() if cached_valid else ""

            if (not cached_valid or scene_count <= 0) and auto_scene_scan and auto_detect_scenes:
                try:
                    scenes = detect_scenes(
                        str(rep or input_path_val),
                        threshold=scene_threshold,
                        min_scene_len=min_scene_len,
                    )
                    scene_stats = _calculate_scene_frame_stats(scenes or [], fps_val, total_frames=total_frames)
                    scene_count = int(scene_stats.get("scene_count", len(scenes or [])) or 0)
                    chunk_min_frames = scene_stats.get("chunk_min_frames")
                    chunk_avg_frames = scene_stats.get("chunk_avg_frames")
                    chunk_max_frames = scene_stats.get("chunk_max_frames")
                    frame_stats_method = str(scene_stats.get("frame_stats_method") or "").strip()
                    scene_scan_error = ""

                    scan_payload: Dict[str, Any] = {
                        "input_path": normalize_path(input_path_val),
                        "scene_threshold": scene_threshold,
                        "min_scene_len": min_scene_len,
                        "scene_count": scene_count,
                        "success": scene_count > 0,
                    }
                    if chunk_min_frames is not None:
                        scan_payload["chunk_min_frames"] = int(chunk_min_frames)
                    if chunk_avg_frames is not None:
                        scan_payload["chunk_avg_frames"] = float(chunk_avg_frames)
                    if chunk_max_frames is not None:
                        scan_payload["chunk_max_frames"] = int(chunk_max_frames)
                    if total_frames is not None:
                        scan_payload["total_frames"] = int(total_frames)
                    if frame_stats_method:
                        scan_payload["frame_stats_method"] = frame_stats_method
                    seed_controls["last_scene_scan"] = scan_payload
                    state["seed_controls"] = seed_controls
                except Exception as exc:
                    scene_count = 0
                    scene_scan_error = str(exc)
                    seed_controls["last_scene_scan"] = {
                        "input_path": normalize_path(input_path_val),
                        "scene_threshold": scene_threshold,
                        "min_scene_len": min_scene_len,
                        "scene_count": 0,
                        "success": False,
                        "error": scene_scan_error,
                    }
                    state["seed_controls"] = seed_controls

            if scene_count > 0:
                chunk_rows.append(_stat_row("Detected Scenes", _format_int(scene_count)))
                if chunk_min_frames is not None:
                    chunk_rows.append(_stat_row("Min Frames / Chunk", _format_int(chunk_min_frames)))
                if chunk_avg_frames is not None:
                    chunk_rows.append(_stat_row("Avg Frames / Chunk", f"{float(chunk_avg_frames):.1f}"))
                if chunk_max_frames is not None:
                    chunk_rows.append(_stat_row("Max Frames / Chunk", _format_int(chunk_max_frames)))
                if (
                    chunk_min_frames is None
                    or chunk_avg_frames is None
                    or chunk_max_frames is None
                ):
                    chunk_rows.append(_stat_row("Chunk Frame Stats", "Unavailable for current cached scan."))
            elif scene_scan_error:
                chunk_rows.append(_stat_row("Auto Chunk Status", f"Scene scan failed: {scene_scan_error}"))
            elif not auto_detect_scenes:
                chunk_rows.append(_stat_row("Auto Chunk Status", "Auto scene detection is disabled in Resolution tab."))
            else:
                chunk_rows.append(_stat_row("Auto Chunk Status", "No scenes detected yet."))
        else:
            chunk_size = float(seed_controls.get("chunk_size_sec", 0) or 0)
            chunk_overlap = float(seed_controls.get("chunk_overlap_sec", 0) or 0)
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
                    est_chunk_frames = max(1, int(round(float(chunk_size) * float(fps_val))))
                    chunk_rows.append(_stat_row("Approx Frames / Chunk", _format_int(est_chunk_frames)))
    else:
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
    return gr.update(value=html_block, visible=True)
