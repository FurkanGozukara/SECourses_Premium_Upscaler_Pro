"""
Lightweight native path dialogs for Gradio callbacks.

This mirrors the Musubi trainer strategy:
- keep a small icon button beside textbox fields
- open a native Tk file dialog on click
- return the chosen path directly to the textbox
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

from shared.path_utils import normalize_path


_ENV_EXCLUSION = ("COLAB_GPU", "RUNPOD_POD_ID")


def _display_available() -> bool:
    """Return True when a GUI display is likely available for Tk dialogs."""
    if any(name in os.environ for name in _ENV_EXCLUSION):
        return False
    if sys.platform == "darwin":
        return False
    if sys.platform.startswith("linux"):
        if ("DISPLAY" not in os.environ) and ("WAYLAND_DISPLAY" not in os.environ):
            return False
    return True


def _split_path_for_dialog(path_value: str) -> Tuple[str, str]:
    """Split user path into dialog initial dir + initial file name."""
    raw = str(path_value or "").strip()
    if not raw:
        return ".", ""

    normalized = normalize_path(raw) or raw
    p = Path(normalized)

    if p.exists() and p.is_dir():
        return str(p), ""

    parent = p.parent if str(p.parent) not in ("", ".") else Path(".")
    return str(parent), p.name


def get_any_file_path(current_path: str = "") -> str:
    """
    Open native file picker and return selected file path.

    If selection is cancelled or dialog is unavailable, returns `current_path`.
    """
    current_path = str(current_path or "")
    if not _display_available():
        return current_path

    try:
        from tkinter import Tk, filedialog
    except Exception:
        return current_path

    initial_dir, initial_file = _split_path_for_dialog(current_path)
    root = Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    try:
        selected = filedialog.askopenfilename(
            initialdir=initial_dir,
            initialfile=initial_file,
        )
    finally:
        root.destroy()

    if not selected:
        return current_path
    return normalize_path(selected) or selected

