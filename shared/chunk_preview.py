from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _collect_chunk_outputs(run_dir: Path) -> List[Path]:
    processed_dir = run_dir / "processed_chunks"
    if not processed_dir.exists():
        # Best-effort fallback: search direct chunk files in run dir.
        direct = sorted([p for p in run_dir.glob("chunk_*") if p.exists()])
        return direct

    items: List[Path] = []
    files = sorted(
        [
            p
            for p in processed_dir.iterdir()
            if p.exists()
            and (
                (p.is_file() and p.suffix.lower() in VIDEO_EXTS.union(IMAGE_EXTS))
                or p.is_dir()
            )
        ]
    )
    items.extend(files)
    return items


def _pick_preview_from_dir(chunk_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    videos = sorted([p for p in chunk_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
    if videos:
        return str(videos[0]), str(videos[0])
    images = sorted([p for p in chunk_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if images:
        return str(images[0]), None
    return None, None


def _safe_video_preview_copy(video_path: str) -> str:
    """
    Return a UI-safe copy path for video previews.

    We intentionally avoid hard links because some UI stacks may transcode/normalize
    preview videos in-place, which could mutate the original output artifact.
    """
    src = Path(str(video_path)).resolve()
    if not src.exists() or not src.is_file():
        return str(src)

    try:
        st = src.stat()
        sig = hashlib.sha256(
            f"{src.as_posix()}|{st.st_size}|{st.st_mtime_ns}|ui_safe_v1".encode("utf-8")
        ).hexdigest()[:20]
        cache_dir = src.parent / ".ui_preview_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dst = cache_dir / f"{src.stem}.__ui_preview_{sig}{src.suffix}"

        if dst.exists() and dst.is_file() and dst.stat().st_size > 1024:
            return str(dst)

        tmp = dst.with_name(f"{dst.name}.{os.getpid()}.tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        return str(dst)
    except Exception:
        return str(src)


def build_chunk_preview_payload(run_dir: str, max_items: int = 24) -> Dict[str, Any]:
    root = Path(str(run_dir))
    if not root.exists():
        return {"message": "Chunk preview unavailable (run directory missing).", "gallery": [], "videos": [], "count": 0}

    chunk_items = _collect_chunk_outputs(root)
    if not chunk_items:
        return {"message": "No processed chunks found for this run.", "gallery": [], "videos": [], "count": 0}

    gallery: List[Tuple[str, str]] = []
    videos: List[Optional[str]] = []

    for idx, item in enumerate(chunk_items[:max_items], 1):
        display_path: Optional[str] = None
        video_path: Optional[str] = None

        if item.is_file():
            if item.suffix.lower() in VIDEO_EXTS:
                video_path = _safe_video_preview_copy(str(item))
                display_path = str(item)
            elif item.suffix.lower() in IMAGE_EXTS:
                display_path = str(item)
        elif item.is_dir():
            display_path, video_path = _pick_preview_from_dir(item)

        if not display_path:
            continue

        # For videos, show thumbnail in gallery when possible.
        if video_path:
            try:
                from shared.frame_utils import extract_video_thumbnail

                ok, thumb_path, _ = extract_video_thumbnail(video_path, width=320)
                if ok and thumb_path:
                    display_path = str(thumb_path)
            except Exception:
                pass

        gallery.append((display_path, f"Chunk {idx}"))
        videos.append(video_path)

    count = len(gallery)
    if count == 0:
        return {"message": "No previewable chunk artifacts found.", "gallery": [], "videos": [], "count": 0}

    msg = f"Detected {count} processed chunk preview(s)."
    return {"message": msg, "gallery": gallery, "videos": videos, "count": count}
