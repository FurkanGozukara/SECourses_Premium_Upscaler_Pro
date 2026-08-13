"""
Shared helpers used by the model tab modules.

These were extracted verbatim from byte-identical copies that lived in
ui/flashvsr_tab.py, ui/sparkvsr_tab.py, ui/ltx25_tab.py, ui/gan_tab.py,
ui/rtx_super_resolution_tab.py and secourses_app.py. Tabs import them with
aliases so call sites keep their original local names, e.g.:

    from ui.model_tab_common import sync_signature as _sync_signature

This module intentionally does NOT import gradio at module import time;
build_input_detection_md imports it lazily when called.
"""

import hashlib
import json
import re
from typing import Any, Dict


# Terminal-status substrings shared by the FlashVSR+/SparkVSR/LTX 2.5 tabs'
# _expand_service_payload progress logic. Each tab appends its own extra
# tokens (e.g. terminal_tokens = TERMINAL_STATUS_TOKENS + ("...",)) so the
# per-tab effective sets stay exactly what they were before extraction.
TERMINAL_STATUS_TOKENS = (
    "complete",
    "completed",
    "failed",
    "error",
    "critical",
    "cancel",
    "no result",
    "out of vram",
    "oom",
    "timed out",
    "timeout",
    "aborted",
    "input path missing",
    "input missing",
    "batch input folder missing",
    "resume folder not found",
    "resume input not found",
    "resume unavailable",
    "ffmpeg not found",
    "insufficient disk space",
)


def sync_signature(payload: Dict[str, Any]) -> str:
    try:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str, separators=(",", ":"))
    except Exception:
        blob = str(payload)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def resolve_shared_upscale_factor(state: Dict[str, Any] | None) -> float | None:
    """
    Resolve shared/global upscale value from app state.
    """
    if not isinstance(state, dict):
        return None
    try:
        seed_controls = state.get("seed_controls", {}) or {}
        raw = seed_controls.get("upscale_factor_val")
        if raw is None:
            return None
        val = float(raw)
        if val <= 0:
            return None
        return val
    except Exception:
        return None


def build_input_detection_md(path_val: str):
    import gradio as gr
    from shared.input_detector import detect_input
    if not path_val or not str(path_val).strip():
        # Hide when empty (clearing input should clear this panel).
        return gr.update(value="", visible=False)
    try:
        info = detect_input(path_val)
        if not info.is_valid:
            return gr.update(value=f"ERROR: **Invalid Input**\n\n{info.error_message}", visible=True)
        parts = [f"OK: **Input Detected: {info.input_type.upper()}**"]
        if info.input_type == "frame_sequence":
            parts.append(f"&nbsp;&nbsp;Pattern: `{info.frame_pattern}`")
            parts.append(f"&nbsp;&nbsp;Frames: {info.frame_start}-{info.frame_end}")
            if info.missing_frames:
                parts.append(f"&nbsp;&nbsp;Missing: {len(info.missing_frames)}")
        elif info.input_type == "directory":
            parts.append(f"&nbsp;&nbsp;Files: {info.total_files}")
        elif info.input_type in ["video", "image"]:
            parts.append(f"&nbsp;&nbsp;Format: **{info.format.upper()}**")
        return gr.update(value=" ".join(parts), visible=True)
    except Exception as e:
        return gr.update(value=f"ERROR: **Detection Error**\n\n{str(e)}", visible=True)


def extract_update_value(update_obj):
    try:
        if isinstance(update_obj, dict):
            return update_obj.get("value")
    except Exception:
        pass
    return None


def compact_single_line(text: Any, max_len: int = 120) -> str:
    raw = str(text or "")
    try:
        raw = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw)
    except Exception:
        pass
    raw = re.sub(r"\s+", " ", raw.replace("\r", " ").replace("\n", " ")).strip()
    if len(raw) > max_len:
        raw = raw[: max(0, max_len - 3)].rstrip() + "..."
    return raw


def log_tail_line(logs: Any) -> str:
    if not isinstance(logs, str):
        return ""
    for line in reversed(logs.splitlines()):
        compact = compact_single_line(line)
        if compact:
            return compact
    return ""
