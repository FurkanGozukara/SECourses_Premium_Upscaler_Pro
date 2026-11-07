from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set

from shared.path_utils import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, collision_safe_path, sanitize_filename

# Media files directly in the batch output folder are results (this run's, a skipped item's, or an
# earlier run's): the keep-only cleanup never deletes them.
_ROOT_MEDIA_EXTS = {str(e).lower() for e in (*IMAGE_EXTENSIONS, *VIDEO_EXTENSIONS)}


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _keep_parent_chain(path: Path, root: Path, out: Set[Path]) -> None:
    cur = path
    root_resolved = root.resolve()
    while True:
        try:
            cur_resolved = cur.resolve()
        except Exception:
            cur_resolved = cur
        out.add(cur_resolved)
        if cur_resolved == root_resolved:
            break
        parent = cur.parent
        if parent == cur:
            break
        cur = parent


def _cleanup_keep_files(
    root_resolved: Path,
    preserve_files: Set[Path],
) -> Dict[str, int]:
    root = Path(root_resolved)
    keep_dirs: Set[Path] = {root_resolved}
    for p in preserve_files:
        _keep_parent_chain(p, root_resolved, keep_dirs)

    deleted_files = 0
    for candidate in sorted(root.rglob("*")):
        try:
            if not candidate.is_file():
                continue
        except Exception:
            continue

        try:
            cand_resolved = candidate.resolve()
        except Exception:
            cand_resolved = candidate

        if cand_resolved in preserve_files:
            continue
        if cand_resolved.parent == root_resolved and cand_resolved.suffix.lower() in _ROOT_MEDIA_EXTS:
            continue

        try:
            candidate.unlink(missing_ok=True)
            deleted_files += 1
        except Exception:
            pass

    deleted_dirs = 0
    all_dirs = [d for d in root.rglob("*") if d.is_dir()]
    all_dirs.sort(key=lambda d: len(d.parts), reverse=True)
    for directory in all_dirs:
        try:
            dir_resolved = directory.resolve()
        except Exception:
            dir_resolved = directory

        if dir_resolved in keep_dirs:
            continue

        try:
            directory.rmdir()
            deleted_dirs += 1
        except Exception:
            pass

    return {
        "deleted_files": int(deleted_files),
        "deleted_dirs": int(deleted_dirs),
    }


def _flatten_target_for_file(
    root_resolved: Path,
    src_file: Path,
    src_output_root: Optional[Path] = None,
    taken: Optional[Set[Path]] = None,
    protected: Optional[Set[Path]] = None,
) -> Path:
    # If already at root, keep as-is.
    try:
        if src_file.resolve().parent == root_resolved:
            return src_file.resolve()
    except Exception:
        pass

    # Files of a directory output (a PNG frame sequence) stay where they are, inside the folder
    # named after the input, so the frames keep their names (no `<dir>_<frame>` renaming).
    if src_output_root is not None:
        try:
            return src_file.resolve()
        except Exception:
            return src_file

    # Normal file output: keep the filename and move it to the root. A file already there under
    # that name is an earlier run's result for the same input and is replaced; only two results of
    # this run with one name, or a batch input itself (output folder = input folder), get a
    # numbered name instead.
    safe_name = sanitize_filename(src_file.name) or src_file.name
    target = Path(root_resolved) / safe_name
    try:
        target_resolved = target.resolve()
    except Exception:
        target_resolved = target
    if target.exists() and (
        target.is_dir()
        or (taken is not None and target_resolved in taken)
        or (protected is not None and target_resolved in protected)
    ):
        return collision_safe_path(target).resolve()
    return target_resolved


def keep_only_batch_outputs(
    batch_output_root: Path,
    output_paths: Iterable[str],
    on_log: Optional[Callable[[str], None]] = None,
    *,
    preserve_recovery: bool = False,
    also_keep: Optional[Iterable[str]] = None,
    inputs: Optional[Iterable[str]] = None,
) -> Dict[str, int | str | List[str]]:
    """
    Flatten final outputs into the batch output folder root, then remove all
    other generated artifacts. With preserve_recovery, keep recovery folders in
    place without counting their contents as successful outputs.

    also_keep: existing results of items this run skipped (output already there). They are
    flattened and kept like outputs but not reported in final_outputs; without them the cleanup
    deleted the earlier results of skipped items.
    inputs: the batch's input files; a flattened output never replaces one of them (when the
    batch output folder is the input folder it gets a numbered name instead).
    """
    root = Path(batch_output_root)
    if not root.exists() or not root.is_dir():
        msg = f"Keep-only cleanup skipped: output root not found: {root}"
        if on_log:
            on_log(msg)
        return {
            "deleted_files": 0,
            "deleted_dirs": 0,
            "kept_outputs": 0,
            "status": "skipped",
            "final_outputs": [],
        }

    try:
        root_resolved = root.resolve()
    except Exception:
        root_resolved = root

    preserve_files: Set[Path] = set()
    final_outputs: List[str] = []
    moved_files = 0
    warned = 0
    counted_kept = 0
    protected: Set[Path] = set()
    for raw_input in list(inputs or []):
        try:
            protected.add(Path(str(raw_input)).resolve())
        except Exception:
            continue

    entries = [(raw, True) for raw in list(output_paths or [])]
    entries += [(raw, False) for raw in list(also_keep or [])]
    for raw, counted in entries:
        if not counted and not counted_kept:
            break  # No new outputs: the cleanup is skipped below, so leave skipped results untouched.
        p_text = str(raw or "").strip()
        if not p_text:
            continue
        p = Path(p_text)
        if not p.exists():
            continue
        try:
            p_resolved = p.resolve()
        except Exception:
            p_resolved = p
        if not _is_within(p_resolved, root_resolved):
            continue

        # File output: one final file.
        if p_resolved.is_file():
            dest = _flatten_target_for_file(
                root_resolved, p_resolved, src_output_root=None, taken=preserve_files, protected=protected
            )
            if p_resolved != dest:
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.is_file():
                        os.replace(str(p_resolved), str(dest))
                    else:
                        shutil.move(str(p_resolved), str(dest))
                    moved_files += 1
                except Exception as move_err:
                    warned += 1
                    if on_log:
                        on_log(f"Keep-only cleanup warning: failed to move '{p_resolved.name}': {move_err}")
                    dest = p_resolved
            preserve_files.add(dest)
            if counted:
                final_outputs.append(str(dest))
                counted_kept += 1
            continue

        # Directory output: treat contained files as final output files.
        if p_resolved.is_dir():
            dir_files = [f for f in sorted(p_resolved.rglob("*")) if f.is_file()]
            for src_file in dir_files:
                dest = _flatten_target_for_file(root_resolved, src_file, src_output_root=p_resolved)
                if src_file.resolve() != dest:
                    try:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(src_file), str(dest))
                        moved_files += 1
                    except Exception as move_err:
                        warned += 1
                        if on_log:
                            on_log(f"Keep-only cleanup warning: failed to move '{src_file.name}': {move_err}")
                        dest = src_file.resolve()
                preserve_files.add(dest)
                if counted:
                    final_outputs.append(str(dest))
                    counted_kept += 1

    if not preserve_files or not counted_kept:
        msg = (
            "Keep-only cleanup skipped: no final output files were found "
            "inside the batch output folder."
        )
        if on_log:
            on_log(msg)
        return {
            "deleted_files": 0,
            "deleted_dirs": 0,
            "kept_outputs": 0,
            "status": "skipped",
            "final_outputs": [],
        }

    deduped_outputs: List[str] = []
    seen_outputs: Set[str] = set()
    for outp in final_outputs:
        key = str(outp).strip()
        if not key:
            continue
        key_norm = key.lower() if os.name == "nt" else key
        if key_norm in seen_outputs:
            continue
        seen_outputs.add(key_norm)
        deduped_outputs.append(key)
    final_outputs = deduped_outputs

    if preserve_recovery:
        recovery_files: Set[Path] = set()
        for recovery_dir in root_resolved.rglob("recovery"):
            if not recovery_dir.is_dir() or not _is_within(recovery_dir, root_resolved):
                continue
            for recovery_file in recovery_dir.rglob("*"):
                if recovery_file.is_file() and _is_within(recovery_file, root_resolved):
                    recovery_files.add(recovery_file.resolve())
        preserve_files.update(recovery_files)
        if recovery_files and on_log:
            on_log(f"Keep-only cleanup preserved {len(recovery_files)} recovery file(s).")

    stats = _cleanup_keep_files(root_resolved, preserve_files)

    kept_outputs = len(final_outputs)
    msg = (
        f"Keep-only cleanup complete: flattened {moved_files} file(s), "
        f"removed {stats['deleted_files']} file(s) and {stats['deleted_dirs']} folder(s), "
        f"kept {kept_outputs} final output file(s)."
    )
    if warned > 0:
        msg += f" ({warned} move warning(s))"
    if on_log:
        on_log(msg)

    return {
        "deleted_files": int(stats["deleted_files"]),
        "deleted_dirs": int(stats["deleted_dirs"]),
        "kept_outputs": int(kept_outputs),
        "status": "ok",
        "final_outputs": final_outputs,
    }
