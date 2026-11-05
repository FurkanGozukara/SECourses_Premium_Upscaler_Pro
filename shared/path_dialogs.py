"""
Native folder / file pickers for the path fields of every tab (the Musubi trainer's 📂 buttons, improved).

- The dialog runs in its own short-lived Python process (tkinter, hidden console), so it never shares
  a Tk interpreter with Gradio's worker threads, never freezes the UI and cannot crash the app.
- It opens on top of the browser at the folder of the current value (or the last folder picked for the
  same kind of field), with file-type filters per field (videos, images, models, ...).
- One dialog at a time; Cancel keeps the current value. On a server without a display (RunPod, Colab)
  or without tkinter the buttons are hidden and typing a path works as before.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from shared.path_utils import normalize_path


_ENV_EXCLUSION = ("COLAB_GPU", "RUNPOD_POD_ID")
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

VIDEO_PATTERNS = "*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.flv *.wmv *.ts *.mts *.m2ts"
IMAGE_PATTERNS = "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"
MODEL_PATTERNS = "*.safetensors *.pth *.pt *.ckpt *.bin *.gguf *.onnx"

FILETYPES: Dict[str, List[Tuple[str, str]]] = {
    "media": [("Videos and images", f"{VIDEO_PATTERNS} {IMAGE_PATTERNS}"), ("Videos", VIDEO_PATTERNS),
              ("Images", IMAGE_PATTERNS), ("All files", "*.*")],
    "video": [("Videos", VIDEO_PATTERNS), ("All files", "*.*")],
    "image": [("Images", IMAGE_PATTERNS), ("All files", "*.*")],
    "model": [("Model files", MODEL_PATTERNS), ("All files", "*.*")],
    "script": [("Python scripts", "*.py"), ("All files", "*.*")],
    "file": [("All files", "*.*")],
}

TITLES = {
    "folder": "Select a folder",
    "media": "Select a video or image",
    "video": "Select a video",
    "image": "Select an image",
    "model": "Select a model file",
    "script": "Select a script",
    "file": "Select a file",
}

_DIALOG_SCRIPT = r"""
import json, sys
args = json.loads(sys.argv[1])
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
if args.get("selftest"):
    root.destroy()
    sys.stdout.write(json.dumps({"path": "", "ok": True}))
    raise SystemExit(0)
root.attributes("-topmost", True)
root.update_idletasks()
root.lift()
root.focus_force()
kw = {"parent": root, "title": args["title"]}
if args.get("initialdir"):
    kw["initialdir"] = args["initialdir"]
if args["mode"] == "folder":
    result = filedialog.askdirectory(mustexist=True, **kw)
else:
    if args.get("initialfile"):
        kw["initialfile"] = args["initialfile"]
    kw["filetypes"] = [tuple(item) for item in args.get("filetypes") or [("All files", "*.*")]]
    result = filedialog.askopenfilename(**kw)
root.destroy()
sys.stdout.write(json.dumps({"path": result or ""}))
"""

_lock = threading.Lock()
_last_dir: Dict[str, str] = {}
_available: Optional[bool] = None


def _display_available() -> bool:
    """Return True when a GUI display is likely available for Tk dialogs."""
    if any(name in os.environ for name in _ENV_EXCLUSION):
        return False
    if os.environ.get("SECOURSES_DISABLE_PATH_PICKERS", "").strip() in ("1", "true", "yes"):
        return False
    if sys.platform == "darwin":
        return False
    if sys.platform.startswith("linux"):
        if ("DISPLAY" not in os.environ) and ("WAYLAND_DISPLAY" not in os.environ):
            return False
    return True


def _python() -> str:
    """The interpreter of this app (pythonw.exe on Windows when present: no console at all)."""
    exe = Path(sys.executable)
    if os.name == "nt":
        windowed = exe.with_name("pythonw.exe")
        if windowed.is_file():
            return str(windowed)
    return str(exe)


def _run_dialog(args: Dict[str, object], timeout: Optional[float] = None) -> Optional[Dict[str, object]]:
    try:
        completed = subprocess.run(
            [_python(), "-I", "-c", _DIALOG_SCRIPT, json.dumps(args)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        return json.loads((completed.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def dialogs_available() -> bool:
    """True when the native pickers can open here (display present and tkinter importable). Cached."""
    global _available
    if os.environ.get("SECOURSES_PATH_PICKER_TEST_VALUE") is not None:
        return True
    if _available is None:
        if not _display_available():
            _available = False
        else:
            result = _run_dialog({"selftest": True, "mode": "folder", "title": ""}, timeout=30)
            _available = bool(result and result.get("ok"))
    return _available


def _existing_dir(path_value: str) -> str:
    """The folder a dialog should open at for ``path_value`` (nearest existing parent), or ""."""
    raw = str(path_value or "").strip().strip('"')
    if not raw:
        return ""
    p = Path(normalize_path(raw) or raw)
    for candidate in (p, *p.parents):
        try:
            if candidate.is_dir():
                return str(candidate)
        except OSError:
            continue
    return ""


def _split_path_for_dialog(path_value: str) -> Tuple[str, str]:
    """Split user path into dialog initial dir + initial file name."""
    raw = str(path_value or "").strip().strip('"')
    if not raw:
        return "", ""
    p = Path(normalize_path(raw) or raw)
    if p.is_dir():
        return str(p), ""
    return _existing_dir(str(p.parent)), (p.name if p.suffix else "")


def pick_path(kind: str = "folder", current_path: str = "", title: str = "",
              filetypes: Optional[Sequence[Tuple[str, str]]] = None) -> str:
    """Open the native picker for ``kind`` ("folder", "media", "video", "image", "model", "script", "file")
    and return the chosen path; Cancel, a busy picker or no display returns ``current_path`` unchanged."""
    current_path = "" if current_path is None else str(current_path)
    kind = kind if kind in TITLES else "file"
    scripted = os.environ.get("SECOURSES_PATH_PICKER_TEST_VALUE")
    if scripted is not None:  # automated UI tests only: the "dialog" answers this path at once
        return scripted.strip() or current_path
    if not dialogs_available():
        return current_path
    if not _lock.acquire(blocking=False):  # a dialog is already open
        return current_path
    try:
        if kind == "folder":
            initial_dir, initial_file = _existing_dir(current_path), ""
        else:
            initial_dir, initial_file = _split_path_for_dialog(current_path)
        initial_dir = initial_dir or _last_dir.get(kind) or _last_dir.get("any") or ""
        args = {
            "mode": "folder" if kind == "folder" else "file",
            "title": title or TITLES[kind],
            "initialdir": initial_dir,
            "initialfile": initial_file,
            "filetypes": [list(item) for item in (filetypes or FILETYPES.get(kind, FILETYPES["file"]))],
        }
        result = _run_dialog(args)
    finally:
        _lock.release()
    selected = str((result or {}).get("path") or "").strip()
    if not selected:
        return current_path
    selected = normalize_path(selected) or selected
    folder = selected if kind == "folder" else str(Path(selected).parent)
    _last_dir[kind] = folder
    _last_dir["any"] = folder
    return selected


def pick_folder(current_path: str = "") -> str:
    return pick_path("folder", current_path)


def get_any_file_path(current_path: str = "") -> str:
    """Open native file picker and return selected file path (kept for existing callers)."""
    return pick_path("media", current_path)
