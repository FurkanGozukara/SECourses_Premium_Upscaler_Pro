"""
Preview input helpers shared by fixed-scale pipelines.

These helpers build lightweight preview inputs so "Preview First Frame" can run
quickly for videos and frame-sequence directories.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from .path_utils import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    collision_safe_dir,
    collision_safe_path,
    detect_input_type,
    list_files_sorted,
    normalize_path,
)


def _first_file_with_exts(folder: Path, exts: set[str]) -> Optional[Path]:
    files = list_files_sorted(folder, exts)
    return files[0] if files else None


def _extract_first_frame(video_path: Path, temp_root: Path, prefix: str) -> Optional[Path]:
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    out_path = collision_safe_path(temp_root / f"{prefix}_{video_path.stem}_preview_frame.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        "select=eq(n\\,0)",
        "-vframes",
        "1",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode == 0 and out_path.exists():
            return out_path
    except Exception:
        return None
    return None


def _as_single_frame_dir(image_path: Path, temp_root: Path, prefix: str) -> Optional[Path]:
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    dst_dir = collision_safe_dir(temp_root / f"{prefix}_{image_path.stem}_preview_frames")
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / "frame_000001.png"
        shutil.copy2(image_path, dst_path)
    except Exception:
        return None
    return dst_dir if dst_path.exists() else None


def prepare_preview_input(
    input_path: str,
    temp_root: Path,
    *,
    prefix: str,
    as_single_frame_dir: bool = False,
) -> Tuple[Optional[str], str]:
    """
    Build a lightweight preview input path.

    Returns:
      (preview_path, note)
    """
    normalized = normalize_path(str(input_path or "").strip()) if input_path else None
    if not normalized:
        return None, "Preview input is empty."

    src = Path(normalized)
    if not src.exists():
        return None, f"Preview input does not exist: {src}"

    input_kind = detect_input_type(str(src))
    preview_image: Optional[Path] = None

    if input_kind == "image":
        preview_image = src
    elif input_kind == "video":
        preview_image = _extract_first_frame(src, temp_root, prefix)
        if not preview_image:
            return None, f"Failed to extract preview frame from video: {src.name}"
    elif input_kind == "directory":
        img = _first_file_with_exts(src, IMAGE_EXTENSIONS)
        if img:
            preview_image = img
        else:
            vid = _first_file_with_exts(src, VIDEO_EXTENSIONS)
            if not vid:
                return None, f"No previewable media found in folder: {src}"
            preview_image = _extract_first_frame(vid, temp_root, prefix)
            if not preview_image:
                return None, f"Failed to extract preview frame from folder video: {vid.name}"
    else:
        return None, f"Unsupported preview input type: {input_kind}"

    if not preview_image or not preview_image.exists():
        return None, "Preview frame could not be prepared."

    if as_single_frame_dir:
        folder = _as_single_frame_dir(preview_image, temp_root, prefix)
        if not folder:
            return None, f"Failed to prepare single-frame preview folder from: {preview_image.name}"
        return str(folder), f"Preview input prepared from first frame ({preview_image.name})."

    return str(preview_image), f"Preview input prepared from first frame ({preview_image.name})."
