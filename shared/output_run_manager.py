import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .path_utils import ensure_dir, sanitize_filename


RUN_CONTEXT_FILENAME = "run_context.json"


@dataclass(frozen=True)
class VideoRunPaths:
    """
    Paths for a single video-processing run.

    The `run_dir` is the user-visible folder (e.g., outputs/0000/).
    Chunks live under `input_chunks_dir` and `processed_chunks_dir`.
    """

    run_dir: Path
    input_chunks_dir: Path
    processed_chunks_dir: Path
    thumbs_dir: Path
    context_path: Path


def _iter_numeric_children(root: Path, width: int) -> list[int]:
    out: list[int] = []
    try:
        if not root.exists():
            return out
        for child in root.iterdir():
            try:
                if not child.is_dir():
                    continue
                name = child.name
                if len(name) != width or not name.isdigit():
                    continue
                out.append(int(name))
            except Exception:
                continue
    except Exception:
        return out
    return out


def allocate_sequential_run_dir(output_root: Path, width: int = 4, start: int = 0) -> Path:
    """
    Allocate a new, collision-free sequential directory under `output_root`.

    Guarantees uniqueness across multiple processes by using atomic mkdir.
    Example output: outputs/0000, outputs/0001, outputs/0002, ...
    """
    output_root = Path(output_root)
    ensure_dir(output_root)

    existing = _iter_numeric_children(output_root, width=width)
    next_idx = max(existing) + 1 if existing else int(start)
    next_idx = max(int(start), int(next_idx))

    # Atomic claim: mkdir(exist_ok=False) is process-safe on both Windows and Linux.
    while True:
        cand = output_root / f"{next_idx:0{width}d}"
        try:
            cand.mkdir(parents=False, exist_ok=False)
            return cand
        except FileExistsError:
            next_idx += 1


def prepare_batch_video_run_dir(
    output_root: Path,
    original_filename: str,
    *,
    input_path: str,
    model_label: str,
    mode: str,
    overwrite_existing: bool = False,
) -> Optional[VideoRunPaths]:
    """
    Prepare a stable, per-input batch run folder (outputs/<input_stem>/) with
    chunk subfolders inside.

    Concurrency-safe: claims the folder via atomic mkdir. If it already exists and
    `overwrite_existing` is False, returns None (caller should skip).
    """
    run_dir = Path(batch_item_dir(output_root, original_filename))
    if overwrite_existing and run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return None
    except Exception:
        # If creation fails for any reason (permissions/path), let caller fall back.
        return None
    return init_existing_video_run_dir(
        run_dir=run_dir,
        input_path=input_path,
        original_filename=original_filename,
        model_label=model_label,
        mode=mode,
    )


def batch_item_dir(output_root: Path, original_filename: str) -> Path:
    """
    Batch outputs use a stable, user-visible folder named after the input (stem).
    Example: outputs/35sec_360x240/
    """
    stem = Path(original_filename).stem if original_filename else "unnamed"
    safe = sanitize_filename(stem)
    return Path(output_root) / safe


def downscaled_video_path(run_dir: Path, original_filename: str) -> Path:
    """
    User-visible downscaled artifact name:
      downscaled_<original_stem>.mp4
    """
    stem = Path(original_filename).stem if original_filename else "unnamed"
    safe_stem = sanitize_filename(stem)
    return Path(run_dir) / f"downscaled_{safe_stem}.mp4"


def _reserve_sequential_id(reserve_root: Path, width: int = 4, start: int = 1) -> int:
    reserve_root = Path(reserve_root)
    ensure_dir(reserve_root)

    pat = re.compile(rf"^(\d{{{width}}})$")
    max_idx = 0
    try:
        for child in reserve_root.iterdir():
            try:
                if not child.is_dir():
                    continue
                m = pat.match(child.name)
                if not m:
                    continue
                max_idx = max(max_idx, int(m.group(1)))
            except Exception:
                continue
    except Exception:
        max_idx = 0

    idx = max(int(start), max_idx + 1)
    while True:
        cand = reserve_root / f"{idx:0{width}d}"
        try:
            cand.mkdir(parents=False, exist_ok=False)
            return idx
        except FileExistsError:
            idx += 1


def numbered_single_image_output_path(output_root: Path, original_filename: str, ext: str = ".png") -> Path:
    """
    Single-image outputs go directly into the output root with sequential prefixes:
      0001_<original_stem>.png

    Uses a reservation directory under output_root/.run_ids/images/ to avoid
    conflicts between multiple concurrent app instances.
    """
    output_root = Path(output_root)
    ensure_dir(output_root)

    reserve_dir = output_root / ".run_ids" / "images"
    idx = _reserve_sequential_id(reserve_dir, width=4, start=1)

    stem = Path(original_filename).stem if original_filename else "unnamed"
    safe_stem = sanitize_filename(stem)
    safe_ext = ext if ext.startswith(".") else f".{ext}"
    return output_root / f"{idx:04d}_{safe_stem}{safe_ext}"


def init_video_run_dir(
    output_root: Path,
    input_path: str,
    original_filename: Optional[str],
    model_label: str,
    mode: str,
) -> VideoRunPaths:
    """
    Create a new run folder and standard subfolders for chunking + artifacts.
    Writes `run_context.json` for resuming/debugging.
    """
    run_dir = allocate_sequential_run_dir(Path(output_root))
    return init_existing_video_run_dir(
        run_dir=run_dir,
        input_path=input_path,
        original_filename=original_filename,
        model_label=model_label,
        mode=mode,
    )


def init_existing_video_run_dir(
    run_dir: Path,
    input_path: str,
    original_filename: Optional[str],
    model_label: str,
    mode: str,
) -> VideoRunPaths:
    """
    Initialize standard subfolders inside an already-determined run directory.

    This is used for batch items where the run folder name is stable (e.g., outputs/<input_stem>/),
    while preserving the same internal structure as sequential single runs.
    """
    run_dir = Path(run_dir)
    ensure_dir(run_dir)

    input_chunks_dir = run_dir / "input_chunks"
    processed_chunks_dir = run_dir / "processed_chunks"
    thumbs_dir = processed_chunks_dir / "thumbs"
    for d in (input_chunks_dir, processed_chunks_dir, thumbs_dir):
        d.mkdir(parents=True, exist_ok=True)

    context_path = run_dir / RUN_CONTEXT_FILENAME
    payload: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_path": str(input_path or ""),
        "original_filename": str(original_filename or ""),
        "model": str(model_label or ""),
        "mode": str(mode or ""),
    }
    try:
        with context_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass

    return VideoRunPaths(
        run_dir=run_dir,
        input_chunks_dir=input_chunks_dir,
        processed_chunks_dir=processed_chunks_dir,
        thumbs_dir=thumbs_dir,
        context_path=context_path,
    )


def parse_output_override_as_root(override: Optional[str]) -> tuple[Optional[Path], Optional[str]]:
    """
    Interpret the UI "Output Override" as a *root location* for run folders.

    - If override is a directory path (no suffix): root_dir=override, file_name_override=None
    - If override is a file path (has suffix): root_dir=override.parent, file_name_override=override.name

    The final output file (if file_name_override is provided) will be created INSIDE the run folder.
    """
    if not override:
        return None, None
    try:
        p = Path(str(override))
        # Treat any suffix as a file name override (e.g., .mp4, .png).
        if p.suffix:
            return p.parent, p.name
        return p, None
    except Exception:
        return None, None


def recent_output_run_dirs(
    output_root: Path,
    *,
    last_run_dir: Optional[str] = None,
    limit: int = 20,
) -> list[Path]:
    """
    Return recent candidate run directories from the output root.

    Priority:
    1) Explicit `last_run_dir` (if provided and exists)
    2) Most-recent subdirectories under `output_root` (mtime desc)
    """
    out: list[Path] = []
    seen: set[str] = set()

    def _push(path_like: Optional[Path]) -> None:
        try:
            if not path_like:
                return
            p = Path(path_like)
            if not p.exists() or not p.is_dir():
                return
            key = str(p.resolve())
            if key in seen:
                return
            seen.add(key)
            out.append(p)
        except Exception:
            return

    if last_run_dir:
        _push(Path(last_run_dir))

    try:
        root = Path(output_root)
        if root.exists() and root.is_dir():
            dirs = [d for d in root.iterdir() if d.is_dir()]
            dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for d in dirs:
                _push(d)
                if len(out) >= int(limit):
                    break
    except Exception:
        pass

    return out[: max(1, int(limit))]


def prepare_single_video_run(
    output_root_fallback: Path,
    output_override_raw: Optional[str],
    input_path: str,
    original_filename: Optional[str],
    model_label: str,
    mode: str,
) -> tuple[VideoRunPaths, Optional[Path]]:
    """
    Create the run folder for a single video run and return:
      (run_paths, explicit_final_output_path_or_None)
    """
    root_override, file_name_override = parse_output_override_as_root(output_override_raw)
    output_root = root_override if root_override else Path(output_root_fallback)
    run_paths = init_video_run_dir(
        output_root=output_root,
        input_path=input_path,
        original_filename=original_filename,
        model_label=model_label,
        mode=mode,
    )

    if file_name_override:
        safe_name = sanitize_filename(file_name_override)
        if not safe_name:
            safe_name = "output.mp4"
        return run_paths, (run_paths.run_dir / safe_name)
    return run_paths, None
