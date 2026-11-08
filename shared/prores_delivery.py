"""
Final encode for runs whose chunk pipeline works in MP4 files: ProRes and custom FFmpeg video arguments.

The scene / fixed-size chunk pipeline (split, per-chunk outputs, resume, merge) is built around MP4 files, and an
MP4 file cannot hold ProRes (nor many codecs custom arguments may name). When the Output tab asks for ProRes or
gives its own FFmpeg video arguments, a chunked run therefore writes its chunks as lossless H.264 (MP4), and the
merged result is encoded once with the requested settings. The delivered file goes through one lossy step only,
as in a single-pass run that encodes the engine's frames directly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .path_utils import collision_safe_path
from .video_codec_options import (
    check_custom_video_args,
    describe_custom_video_args,
    is_prores_codec,
    normalize_custom_video_args,
    parse_custom_video_args,
    prores_pixel_format,
    prores_profile_for_pixel_format,
)

_NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW

# Output tab keys the chunk intermediates override; the run keeps the requested values for its records.
ENCODE_KEYS = ("video_codec", "video_quality", "video_preset", "pixel_format", "two_pass_encoding", "codec", "crf",
               "video_custom_args")


def wants_prores(settings: Optional[Dict[str, Any]]) -> bool:
    """True when the run's Output tab codec is ProRes."""
    if not isinstance(settings, dict):
        return False
    return is_prores_codec(settings.get("video_codec")) or is_prores_codec(settings.get("codec"))


def prores_label(pixel_format: Optional[str]) -> str:
    return "ProRes 4444" if prores_profile_for_pixel_format(pixel_format) == "4444" else "ProRes 422 HQ"


def prores_video_args(pixel_format: Optional[str]) -> List[str]:
    pix = prores_pixel_format(pixel_format)
    return ["-c:v", "prores_ks", "-profile:v", prores_profile_for_pixel_format(pix), "-pix_fmt", pix,
            "-vendor", "apl0"]


def final_encode_plan(settings: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The final encode a chunked run needs: custom FFmpeg video arguments, ProRes, or None (plain MP4 codecs).

    Returns {"args": [...], "suffix": ".mov", "label": "..."}. Invalid custom arguments give None here; the
    services check them before a run starts.
    """
    if not isinstance(settings, dict):
        return None
    custom = normalize_custom_video_args(settings.get("video_custom_args"))
    if custom:
        args, suffix, err = parse_custom_video_args(custom, default_codec=settings.get("video_codec"))
        if err or not args:
            return None
        return {"args": args, "suffix": suffix,
                "label": f"custom FFmpeg arguments ({describe_custom_video_args(args, suffix)})"}
    if wants_prores(settings):
        pix = prores_pixel_format(settings.get("pixel_format"))
        return {"args": prores_video_args(pix), "suffix": ".mov", "label": prores_label(pix)}
    return None


def _output_value(seed_controls: Dict[str, Any], key: str) -> Any:
    value = seed_controls.get(f"{key}_val")
    if value is None:
        out = seed_controls.get("output_settings")
        value = out.get(key) if isinstance(out, dict) else None
    return value


def custom_video_args_error(seed_controls: Optional[Dict[str, Any]]) -> Optional[str]:
    """None when the Output tab's custom FFmpeg video arguments are empty or FFmpeg accepts them (test encode)."""
    if not isinstance(seed_controls, dict):
        return None
    text = normalize_custom_video_args(_output_value(seed_controls, "video_custom_args"))
    if not text:
        return None
    args, suffix, err = parse_custom_video_args(text, default_codec=_output_value(seed_controls, "video_codec"))
    return err or check_custom_video_args(args, suffix)


def lossless_chunk_settings(settings: Dict[str, Any], *, deep: bool = False) -> Dict[str, Any]:
    """Settings for the chunks of such a run: lossless H.264 4:4:4 in MP4 (10-bit when the engine writes 10-bit)."""
    out = dict(settings)
    out.update(
        video_codec="h264",
        video_quality=0,
        video_preset="veryfast",
        pixel_format="yuv444p10le" if deep else "yuv444p",
        two_pass_encoding=False,
        codec="libx264",
        crf=0,
        video_custom_args="",
    )
    return out


def restore_requested_encode(settings: Dict[str, Any], requested: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of chunk settings with the Output tab encode values the user asked for (for run records)."""
    out = dict(settings)
    for key in ENCODE_KEYS:
        if key in requested:
            out[key] = requested[key]
        else:
            out.pop(key, None)
    return out


def _ffprobe_exe() -> Optional[str]:
    return shutil.which("ffprobe")


def _ffmpeg_exe() -> Optional[str]:
    env = os.environ.get("FFMPEG_PATH")
    if env and os.path.isfile(env):
        return env
    return shutil.which("ffmpeg")


def probe_video_codec(path: Path) -> str:
    exe = _ffprobe_exe()
    if not exe:
        return ""
    try:
        proc = subprocess.run(
            [exe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30, creationflags=_NO_WINDOW,
        )
        lines = str(proc.stdout or "").strip().splitlines()
        return lines[0].strip().lower() if lines else ""
    except Exception:
        return ""


def _probe_frame_count(path: Path) -> int:
    exe = _ffprobe_exe()
    if not exe:
        return -1
    try:
        proc = subprocess.run(
            [exe, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
             "stream=nb_read_packets", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=600, creationflags=_NO_WINDOW,
        )
        return int(str(proc.stdout or "").strip().splitlines()[0])
    except Exception:
        return -1


def _unlink_quietly(path: Path, attempts: int = 1) -> None:
    """Remove a file of our own; on Windows a reader (preview, antivirus) can hold it briefly, so retry."""
    import time

    for attempt in range(max(1, attempts)):
        try:
            Path(path).unlink(missing_ok=True)
            return
        except Exception:
            if attempt + 1 < attempts:
                time.sleep(0.5)


def deliver_final_encode(
    video_path: Optional[str],
    plan: Dict[str, Any],
    on_progress: Optional[Callable[[str], None]] = None,
    keep_source: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """Encode a finished (merged) video with a final encode plan; the result takes the source's name.

    Returns (path, None) on success or (None, error). The source is removed after a verified encode unless
    keep_source is set; when the plan keeps the suffix (.mp4), the source is replaced in place.
    """
    if not video_path:
        return None, "no video to encode"
    src = Path(video_path)
    if not src.is_file():
        return None, f"video not found: {src}"
    ffmpeg = _ffmpeg_exe()
    if not ffmpeg:
        return None, "ffmpeg not found"

    suffix = str(plan.get("suffix") or ".mp4")
    label = str(plan.get("label") or "final encode")
    args = [str(a) for a in plan.get("args") or []]
    in_place = src.suffix.lower() == suffix.lower()
    target = src if in_place else src.with_suffix(suffix)
    if not in_place and target.exists():
        target = collision_safe_path(target)
    tmp = target.with_name(f"{target.stem}__final_tmp{suffix}")

    def _say(msg: str) -> None:
        if on_progress:
            try:
                on_progress(msg)
            except Exception:
                pass

    _say(f"Encoding {label}: {target.name}")
    base = [ffmpeg, "-y", "-v", "error", "-i", str(src), "-map", "0:v:0", "-map", "0:a?", "-map_metadata", "0",
            *args]
    tail = ["-movflags", "+faststart"] if suffix.lower() in (".mp4", ".mov") else []
    # Copy the audio first; a codec the container refuses is re-encoded (PCM in MOV/MKV, AAC in MP4).
    fallback_audio = ["-c:a", "aac", "-b:a", "192k"] if suffix.lower() == ".mp4" else ["-c:a", "pcm_s24le"]
    last_err = ""
    for audio_args in (["-c:a", "copy"], fallback_audio):
        _unlink_quietly(tmp)
        proc = subprocess.run(base + audio_args + tail + [str(tmp)],
                              capture_output=True, text=True, creationflags=_NO_WINDOW)
        if proc.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
            last_err = ""
            break
        last_err = (proc.stderr or "").strip()[-600:] or f"ffmpeg exit code {proc.returncode}"
    if last_err:
        _unlink_quietly(tmp)
        return None, f"{label} failed: {last_err}"

    src_frames = _probe_frame_count(src)
    out_frames = _probe_frame_count(tmp)
    if src_frames > 0 and out_frames >= 0 and out_frames != src_frames:
        _unlink_quietly(tmp)
        return None, f"{label} wrote {out_frames} frames for {src_frames}"
    if in_place and keep_source:
        keep = collision_safe_path(src.with_name(f"{src.stem}_lossless{src.suffix}"))
        try:
            os.replace(src, keep)
        except Exception as exc:
            _unlink_quietly(tmp)
            return None, f"could not keep the source: {exc}"
    try:
        os.replace(tmp, target)
    except Exception as exc:
        _unlink_quietly(tmp)
        return None, f"could not save {target.name}: {exc}"
    if not keep_source and not in_place:
        _unlink_quietly(src, attempts=20)
        if src.exists():
            _say(f"Could not remove the intermediate file (in use): {src}")
    _say(f"{label} saved: {target}")
    return str(target), None


def deliver_prores(
    video_path: Optional[str],
    pixel_format: Optional[str],
    on_progress: Optional[Callable[[str], None]] = None,
    keep_source: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """Encode a finished video to ProRes in a .mov next to it (unchanged if it is a ProRes .mov already)."""
    if video_path and Path(video_path).suffix.lower() == ".mov" and Path(video_path).is_file() \
            and probe_video_codec(Path(video_path)).startswith("prores"):
        return str(video_path), None
    pix = prores_pixel_format(pixel_format)
    plan = {"args": prores_video_args(pix), "suffix": ".mov", "label": f"{prores_label(pix)} (.mov)"}
    out, err = deliver_final_encode(video_path, plan, on_progress=on_progress, keep_source=keep_source)
    if out and not probe_video_codec(Path(out)).startswith("prores"):
        return None, "the encode produced no ProRes stream"
    return out, err
