"""
Shared media preview helpers for Gradio inputs.

Goal:
- Provide the same UX for video inputs as image inputs (preview next to upload/path).
- Keep logic centralized to avoid duplicate per-tab extension checks.

Works with Gradio 6.x by returning `gr.update(...)` objects that can be used
directly as event outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from shared.path_utils import normalize_path


# Keep this aligned with the formats the app supports broadly.
# (We intentionally keep preview formats conservative; exotic formats may not render in browser.)
IMAGE_PREVIEW_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
VIDEO_PREVIEW_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}

BROWSER_VIDEO_FORMATS = {
    (".mp4", "h264"),
    (".mp4", "av1"),
    (".ogg", "theora"),
    (".webm", "vp8"),
    (".webm", "vp9"),
    (".webm", "av1"),
}
MP4_VIDEO_CODECS = {"h264", "av1"}
MP4_AUDIO_CODECS = {"aac", "mp3"}


def _first_stream_codecs(video_path: Path) -> Optional[Tuple[Optional[str], Optional[str]]]:
    """Return the first video and audio codecs reported by ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None

    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_streams",
                "-print_format",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        streams = json.loads(result.stdout).get("streams", [])
    except (OSError, subprocess.CalledProcessError, TypeError, ValueError):
        return None

    codecs: dict[str, str] = {}
    for stream in streams:
        codec_type = stream.get("codec_type")
        if codec_type in {"video", "audio"} and codec_type not in codecs:
            codecs[codec_type] = str(stream.get("codec_name") or "").lower()
    if "video" not in codecs:
        return None
    return codecs.get("video"), codecs.get("audio")


def _is_browser_playable(
    video_path: Path,
    codecs: Optional[Tuple[Optional[str], Optional[str]]] = None,
) -> bool:
    codecs = codecs if codecs is not None else _first_stream_codecs(video_path)
    return bool(codecs and (video_path.suffix.lower(), codecs[0]) in BROWSER_VIDEO_FORMATS)


def _can_remux_to_mp4(codecs: Tuple[Optional[str], Optional[str]]) -> bool:
    video_codec, audio_codec = codecs
    return bool(
        video_codec in MP4_VIDEO_CODECS
        and (audio_codec is None or audio_codec in MP4_AUDIO_CODECS)
    )


def _ffmpeg_preview_command(
    ffmpeg: str,
    source: Path,
    destination: Path,
    *,
    copy_streams: bool,
) -> list[str]:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
    ]
    if copy_streams:
        command.extend(["-c", "copy"])
    else:
        command.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
            ]
        )
    command.extend(["-movflags", "+faststart", str(destination)])
    return command


def _run_ffmpeg_preview(
    ffmpeg: str,
    source: Path,
    destination: Path,
    *,
    copy_streams: bool,
) -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            _ffmpeg_preview_command(
                ffmpeg, source, destination, copy_streams=copy_streams
            ),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return False, str(exc)
    return result.returncode == 0, result.stderr.strip()


def _preview_cache_root() -> Path:
    try:
        from gradio import utils as gradio_utils

        return Path(gradio_utils.get_upload_folder()) / "secourses-video-previews"
    except Exception:
        return Path(tempfile.gettempdir()) / "secourses-video-previews"


def safe_video_preview_path(video_path: Path) -> Optional[str]:
    """Return a browser-playable preview without modifying the processing source."""
    try:
        source = video_path.resolve()
        stat = source.stat()
    except (OSError, RuntimeError):
        return None

    ffmpeg = shutil.which("ffmpeg")
    codecs = _first_stream_codecs(source)
    if not ffmpeg or codecs is None or _is_browser_playable(source, codecs):
        return str(source)

    identity = f"{source}\0{stat.st_size}\0{stat.st_mtime_ns}".encode(
        "utf-8", errors="surrogatepass"
    )
    cache_dir = _preview_cache_root() / hashlib.sha256(identity).hexdigest()
    cached = cache_dir / "preview.mp4"
    temporary = cache_dir / (
        f".{cached.stem}.{os.getpid()}.{threading.get_ident()}.mp4"
    )

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if cached.is_file() and _is_browser_playable(cached):
            return str(cached)

        converted, error = _run_ffmpeg_preview(
            ffmpeg,
            source,
            temporary,
            copy_streams=_can_remux_to_mp4(codecs),
        )
        if not converted and _can_remux_to_mp4(codecs):
            temporary.unlink(missing_ok=True)
            converted, error = _run_ffmpeg_preview(
                ffmpeg, source, temporary, copy_streams=False
            )

        if converted and temporary.is_file() and _is_browser_playable(temporary):
            os.replace(temporary, cached)
            return str(cached)

        if error:
            print(f"[Video Preview] FFmpeg conversion failed: {error[-500:]}", flush=True)
    except OSError:
        pass

    try:
        temporary.unlink(missing_ok=True)
    except OSError:
        pass
    # Hiding the preview is safer than handing Gradio the processing source.
    return None


def _clean_path(path_val: Optional[str]) -> Optional[str]:
    if path_val is None:
        return None
    s = str(path_val).strip()
    if not s:
        return None
    # Strip surrounding quotes users sometimes paste.
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    if not s:
        return None
    try:
        # normalize_path resolves environment variables and makes absolute.
        return normalize_path(s) or s
    except Exception:
        return s


def _first_file_with_ext(folder: Path, exts: set[str]) -> Optional[Path]:
    try:
        for item in sorted(folder.iterdir()):
            if item.is_file() and item.suffix.lower() in exts:
                return item
    except Exception:
        return None
    return None


def pick_preview_paths(path_val: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (image_path, video_path) where at most one is non-None.

    Rules:
    - If `path_val` is an image file → show image preview
    - If `path_val` is a video file → show video preview
    - If `path_val` is a directory → show first image if present, else first video
    - Otherwise → no previews
    """
    cleaned = _clean_path(path_val)
    if not cleaned:
        return None, None

    try:
        p = Path(cleaned)
    except Exception:
        return None, None

    if not p.exists():
        return None, None

    if p.is_file():
        ext = p.suffix.lower()
        if ext in IMAGE_PREVIEW_EXTS:
            return str(p), None
        if ext in VIDEO_PREVIEW_EXTS:
            return None, safe_video_preview_path(p)
        return None, None

    if p.is_dir():
        img = _first_file_with_ext(p, IMAGE_PREVIEW_EXTS)
        if img:
            return str(img), None
        vid = _first_file_with_ext(p, VIDEO_PREVIEW_EXTS)
        if vid:
            return None, safe_video_preview_path(vid)
        return None, None

    return None, None


def preview_updates(path_val: Optional[str]) -> Tuple[gr.update, gr.update]:
    """
    Build (image_update, video_update) for Gradio component outputs.
    """
    img_path, vid_path = pick_preview_paths(path_val)
    img_upd = gr.update(value=img_path if img_path else None, visible=bool(img_path))
    vid_upd = gr.update(value=vid_path if vid_path else None, visible=bool(vid_path))
    return img_upd, vid_upd






