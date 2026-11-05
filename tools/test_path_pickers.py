"""Folder / file picker checks without opening a dialog.

Covers: the dialog subprocess starts tkinter (self-test), the scripted test answer, Cancel / busy keeping the value,
the folder a dialog opens at, file-type filters, the kinds a field accepts, the hidden buttons on a server without a
display, and the path_textbox layout (row, buttons, events) in a real gr.Blocks.
Run: venv/Scripts/python.exe tools/test_path_pickers.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import path_dialogs as pd  # noqa: E402

FAILS: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + (f" ({detail})" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def main() -> int:
    os.environ.pop("SECOURSES_PATH_PICKER_TEST_VALUE", None)
    check("dialog process starts tkinter", pd.dialogs_available())
    check("pythonw on Windows", os.name != "nt" or pd._python().lower().endswith("pythonw.exe"), pd._python())

    media = ROOT / "test_media"
    check("opens at the folder of a file", pd._split_path_for_dialog(str(media / "test_image.png")) == (str(media), "test_image.png"))
    check("opens at the nearest existing parent", pd._existing_dir(str(media / "missing" / "x.mp4")) == str(media))
    check("empty path opens at the default", pd._split_path_for_dialog("") == ("", ""))
    check("video filter", "*.mp4" in pd.FILETYPES["video"][0][1] and "*.png" not in pd.FILETYPES["video"][0][1])
    check("media filter has both", "*.mp4" in pd.FILETYPES["media"][0][1] and "*.png" in pd.FILETYPES["media"][0][1])
    check("model filter", "*.safetensors" in pd.FILETYPES["model"][0][1])

    os.environ["SECOURSES_PATH_PICKER_TEST_VALUE"] = str(media)
    check("scripted answer", pd.pick_path("folder", "old") == str(media))
    os.environ["SECOURSES_PATH_PICKER_TEST_VALUE"] = ""
    check("cancel keeps the value", pd.pick_path("folder", "keep me") == "keep me")
    os.environ.pop("SECOURSES_PATH_PICKER_TEST_VALUE", None)

    assert pd._lock.acquire(blocking=False)
    try:
        check("a second dialog while one is open keeps the value", pd.pick_path("folder", "busy") == "busy")
    finally:
        pd._lock.release()

    saved = pd._available
    os.environ["RUNPOD_POD_ID"] = "test"
    pd._available = None
    try:
        check("no display: pickers off, value kept", not pd.dialogs_available() and pd.pick_path("folder", "x") == "x")
    finally:
        os.environ.pop("RUNPOD_POD_ID", None)
        pd._available = saved

    from ui.path_picker import _kinds

    check("kinds", _kinds("media+folder") == ["media", "folder"] and _kinds("bogus") == ["folder"] and _kinds("model") == ["model"])

    import gradio as gr
    from ui.path_picker import path_textbox

    with gr.Blocks() as demo:
        with gr.Row():
            a = path_textbox("folder", label="A", scale=3)
            b = path_textbox("media+folder", label="B", visible=False)
    rows = [blk for blk in demo.blocks.values() if isinstance(blk, gr.Row) and "path-row" in (blk.elem_classes or [])]
    buttons = [blk for blk in demo.blocks.values() if isinstance(blk, gr.Button) and "path-pick-btn" in (blk.elem_classes or [])]
    check("one row per field, scale moved to the row", len(rows) == 2 and rows[0].scale == 3, str([r.scale for r in rows]))
    check("one button per kind", len(buttons) == 3 and len(a.path_pick_events) == 1 and len(b.path_pick_events) == 2)
    check("hidden field stays hidden", b.visible is False)
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
