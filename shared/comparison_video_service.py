from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from shared.video_comparison_advanced import create_input_vs_output_comparison_video


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}


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
