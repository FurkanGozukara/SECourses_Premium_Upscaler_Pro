"""Batch outputs keep their input's name.

User decision (2026-11-08): in batch processing an output file has the same name as its input
file inside the batch output folder. Post-steps that write a new file next to the output
(face restoration ``_face_restored`` / ``_gfpgan`` / ``_codeformer``, global DLSS 5 ``_DLSS5``,
global RIFE ``_2xFPS``, the RIFE FPS override ``_fps``, fixed-scale sizing ``_final{W}x{H}``)
therefore hand their result back to the item's own output name in batch runs. Single runs keep
their existing names.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional

# Settings key a batch loop sets on each item's settings so shared single-run code knows
# that its post-steps must keep the item's output name.
BATCH_KEEP_NAME_KEY = "_batch_keep_name"


def batch_keep_name(settings: Optional[Mapping[str, Any]]) -> bool:
    """True when ``settings`` belong to a batch item whose outputs keep the input name."""
    try:
        return bool(settings and settings.get(BATCH_KEEP_NAME_KEY))
    except Exception:
        return False


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


def adopt_output_name(base_path: Any, produced_path: Any) -> Optional[str]:
    """Move a post-step result onto the item's own output name.

    ``base_path`` is the output the post-step started from (``clip/clip.mp4``) and
    ``produced_path`` the file it wrote (``clip/clip_face_restored.mp4``). The produced file
    replaces the earlier version under the base name (with the produced file's extension when
    the two differ), so no suffixed copy is left behind. Returns the path that now holds the
    result; on any problem the produced path is returned unchanged, so a post-step result is
    never lost.
    """
    if not produced_path:
        return produced_path
    if not base_path:
        return str(produced_path)
    produced = Path(str(produced_path))
    base = Path(str(base_path))
    try:
        if not produced.exists() or _same_path(produced, base):
            return str(produced_path)
    except OSError:
        return str(produced_path)

    target = base
    if produced.is_file() and produced.suffix.lower() != base.suffix.lower():
        target = base.with_suffix(produced.suffix)
    if target.parent != produced.parent and not target.parent.exists():
        return str(produced_path)
    try:
        if produced.is_dir():
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            os.replace(str(produced), str(target))
        else:
            if target.is_dir():
                return str(produced_path)
            os.replace(str(produced), str(target))
        if not _same_path(target, base) and base.is_file():
            # The earlier version under another extension is now superseded.
            base.unlink()
    except OSError:
        return str(produced_path) if produced.exists() else (str(target) if target.exists() else None)
    return str(target)


def same_name_target(out_dir: Path, source: Path, ext: Optional[str] = None, fallback_tag: str = "") -> Path:
    """``out_dir / <source stem><ext>``: a batch output under its input's own name.

    When that path is the source file itself (the batch output folder is the input folder),
    the output cannot take the input's name without destroying the input, so the previous
    tagged name (``<stem><fallback_tag><ext>``) is used instead.
    """
    out_dir = Path(out_dir)
    source = Path(source)
    suffix = ext if ext is not None else source.suffix
    target = out_dir / f"{source.stem}{suffix}"
    if _same_path(target, source) and fallback_tag:
        target = out_dir / f"{source.stem}{fallback_tag}{suffix}"
    return target
