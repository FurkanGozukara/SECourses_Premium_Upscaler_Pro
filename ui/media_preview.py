"""
Shared media preview helpers for Gradio inputs.

Goal:
- Provide the same UX for video inputs as image inputs (preview next to upload/path).
- Keep logic centralized to avoid duplicate per-tab extension checks.

Works with Gradio 6.x by returning `gr.update(...)` objects that can be used
directly as event outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from shared.path_utils import normalize_path


# Keep this aligned with the formats the app supports broadly.
# (We intentionally keep preview formats conservative; exotic formats may not render in browser.)
IMAGE_PREVIEW_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
VIDEO_PREVIEW_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


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
            return None, str(p)
        return None, None

    if p.is_dir():
        img = _first_file_with_ext(p, IMAGE_PREVIEW_EXTS)
        if img:
            return str(img), None
        vid = _first_file_with_ext(p, VIDEO_PREVIEW_EXTS)
        if vid:
            return None, str(vid)
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






