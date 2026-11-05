"""Path fields with native folder / file picker buttons (📂 folder, 📄 file) next to them.

``path_textbox(kind, **textbox_kwargs)`` is a drop-in for ``gr.Textbox(...)``: it returns the Textbox and
puts it in a small row with its picker button(s). ``kind``:

- "folder"                         📂 folder dialog
- "media" / "video" / "image"      📄 file dialog filtered to videos and / or images
- "model" / "script" / "file"      📄 file dialog (model files, Python scripts, any file)
- "<file kind>+folder"             both buttons (fields that take a file OR a folder, e.g. "media+folder")

A pick writes the path into the field (Cancel keeps it); ``textbox.path_pick_events`` holds the click
events so a tab can chain ``.then(...)`` (e.g. refresh a preview). On a server without a display the
buttons are hidden and the field works as a plain textbox.
"""
from __future__ import annotations

import itertools
from typing import Any, List

import gradio as gr

from shared.path_dialogs import dialogs_available, pick_path

ICONS = {"folder": "📂"}
FILE_KINDS = ("media", "video", "image", "model", "script", "file")
_ids = itertools.count(1)

# Before the pick: remember the field's value. After it: when the value changed, press Enter in the field
# (Gradio "submit"), so every tab refreshes its preview / input info exactly as for a typed path.
_REMEMBER_JS = "(v) => { (window.__secPathPick = window.__secPathPick || {})['__ID__'] = v; return v; }"
_SUBMIT_JS = """() => {
  const btn = document.getElementById('__ID__');
  const row = btn && btn.closest('.path-row');
  const field = row && row.querySelector('textarea, input[type=text], input:not([type])');
  const before = (window.__secPathPick || {})['__ID__'];
  if (!field || field.value === before) return [];
  field.dispatchEvent(new KeyboardEvent('keypress', {key: 'Enter', code: 'Enter', bubbles: true}));
  return [];
}"""


def _kinds(kind: str) -> List[str]:
    parts = [p.strip() for p in str(kind or "folder").split("+") if p.strip()]
    return [p for p in parts if p == "folder" or p in FILE_KINDS] or ["folder"]


def add_path_buttons(textbox: gr.Textbox, kind: str = "folder") -> List[Any]:
    """Create the picker button(s) for ``textbox`` in the current layout and wire them."""
    events = []
    available = dialogs_available()
    for part in _kinds(kind):
        icon = ICONS.get(part, "📄")
        button_id = f"path-pick-{next(_ids)}"
        button = gr.Button(
            icon, size="sm", scale=0, min_width=0, visible=available, elem_id=button_id,
            elem_classes=["path-pick-btn", "path-pick-folder" if part == "folder" else "path-pick-file"],
        )
        event = button.click(
            fn=lambda current, _kind=part: pick_path(_kind, current),
            inputs=[textbox], outputs=[textbox], queue=False, show_progress="hidden",
            js=_REMEMBER_JS.replace("__ID__", button_id),
        ).then(fn=None, js=_SUBMIT_JS.replace("__ID__", button_id), queue=False, show_progress="hidden")
        events.append(event)
    textbox.path_pick_events = getattr(textbox, "path_pick_events", []) + events
    return events


def path_textbox(kind: str = "folder", **kwargs: Any) -> gr.Textbox:
    """A ``gr.Textbox`` with picker button(s) beside it (same arguments as gr.Textbox)."""
    row_kwargs = {"elem_classes": ["path-row"], "equal_height": False}
    if "scale" in kwargs and kwargs["scale"] is not None:
        row_kwargs["scale"] = kwargs.pop("scale")
    # A hidden field hides its buttons too (CSS: .path-row:has(> .hidden)), so tabs can keep toggling the field.
    with gr.Row(**row_kwargs):
        textbox = gr.Textbox(scale=1, **kwargs)
        add_path_buttons(textbox, kind)
    return textbox
