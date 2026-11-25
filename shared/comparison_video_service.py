from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from shared.video_comparison_advanced import create_input_vs_output_comparison_video


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}


def prepare_frame_range_comparison_source(source: str, start: int, end: int, temp_dir: Path) -> str:
    """Render the exact decoded source frames on the same CFR timeline as a RelayVSR export."""
    from shared.video_io import find_ffmpeg, format_fps_rate, probe_video, resolve_frame_range

    info = probe_video(source)
    first, last = resolve_frame_range(info.frames, start, end)
    if first == 0 and last == info.frames - 1:
        return source

    temp_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(tempfile.mkdtemp(prefix="relayvsr_comparison_", dir=temp_dir))
    reference = cache_dir / f"{Path(source).stem}_frames_{first}-{last}.mp4"
    fps = format_fps_rate(info.fps)
    command = [find_ffmpeg(), "-y", "-v", "error", "-nostdin", "-i", source]
    if info.has_audio:
        command += [
            "-ss", f"{first / float(info.fps):.6f}", "-t", f"{(last - first + 1) / float(info.fps):.6f}",
            "-i", source, "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k",
        ]
    command += [
        "-map", "0:v:0",
        "-vf", f"trim=start_frame={first}:end_frame={last + 1},setpts=N/({fps}*TB),pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-r", fps, "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p",
        "-threads", "2", "-movflags", "+faststart", str(reference),
    ]
    flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800, creationflags=flags)
    if result.returncode or not reference.is_file() or not reference.stat().st_size:
        raise RuntimeError(result.stderr.strip() or "FFmpeg produced no comparison source")
    return str(reference)


def maybe_generate_input_vs_output_comparison(
    original_input_path: Optional[str],
    output_video_path: Optional[str],
    seed_controls: Optional[Dict[str, Any]],
    *,
    label_input: str = "Original",
    label_output: str = "Upscaled",
    on_progress: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Optionally generate input-vs-output comparison video according to Output-tab settings.

    Returns:
        (comparison_path, message_or_error)
        - comparison_path is set when generation succeeds.
        - message_or_error is set only when generation was attempted and failed.
    """
    seed_controls = seed_controls or {}
    if not bool(seed_controls.get("generate_comparison_video_val", True)):
        return None, None

    src = Path(str(original_input_path or "").strip())
    out = Path(str(output_video_path or "").strip())
    if not src.exists() or not out.exists():
        return None, None
    if src.suffix.lower() not in VIDEO_EXTS or out.suffix.lower() not in VIDEO_EXTS:
        return None, None

    comparison_layout = str(seed_controls.get("comparison_video_layout_val", "auto") or "auto")
    comparison_path = out.parent / f"{out.stem}_comparison.mp4"

    try:
        if on_progress:
            on_progress("Generating input vs output comparison video...\n")
        success, comp_path, err = create_input_vs_output_comparison_video(
            original_input_video=str(src),
            upscaled_output_video=str(out),
            comparison_output=str(comparison_path),
            layout=comparison_layout,
            label_input=label_input,
            label_output=label_output,
            on_progress=on_progress,
        )
        if success and comp_path:
            return str(comp_path), None
        return None, str(err or "Comparison video generation failed")
    except Exception as e:
        return None, str(e)
