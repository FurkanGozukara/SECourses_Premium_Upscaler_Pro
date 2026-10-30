"""
LTX 2.5 Auto Tune.

The engine sizes its token budget from the VRAM free when it starts (after Save VRAM: 4.9 GB of
non-block weights, 3.0 GB fixed, then 0.25 MB x 1.15 per sampled token; LTX25/ltx25_engine/planner.py)
and keeps as many of the 48 transformer blocks on the GPU as fit, streaming the rest from RAM. This model works best on ONE chunk: every chunk boundary
is a cross-faded regeneration seam. Auto Tune sets the limits that let it reach that:

- Quality: Auto token budget and the longest chunk the model's 20 s position range allows (up to 1025
  frames), so a video up to 20 s is one chunk whenever it fits, and otherwise the fewest, longest chunks.
- Speed: block residency is automatic (all blocks stay on the GPU when they fit).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .common import TunePlan, VramBudget

TOKEN_BUDGET_AUTO = "Auto (Single Chunk Preferred)"
MAX_CHUNK_FRAMES = 1025
# The transformer's time positions are seconds (frame / fps) over max_pos 20: a longer chunk runs past the
# trained range. Measured (720 frames, 30 fps): one 24 s chunk scored 0.2-0.3 dB below 20 s chunks in every
# segment, not only after 20 s, so chunks stop at 20 s.
MAX_CHUNK_SECONDS = 20.0


def max_chunk_frames(fps: float) -> int:
    """Longest 8n+1 chunk whose last frame stays inside the 20 s position range at this frame rate."""
    if not fps or fps <= 0:
        fps = 24.0
    frames = int(MAX_CHUNK_SECONDS * float(fps)) + 1
    return max(9, min(MAX_CHUNK_FRAMES, ((frames - 1) // 8) * 8 + 1))


def _run_fps(settings: Dict[str, Any]) -> float:
    try:
        override = float(settings.get("fps") or 0.0)
    except (TypeError, ValueError):
        override = 0.0
    if override > 0:
        return override
    from .common import probe_media

    info = probe_media(settings.get("input_path"))
    return float(info.fps) if info is not None and info.fps else 24.0


def tuned_values(settings: Dict[str, Any]) -> Dict[str, Any]:
    return {"token_budget_mode": TOKEN_BUDGET_AUTO, "max_chunk_frames": max_chunk_frames(_run_fps(settings))}


def plan(settings: Dict[str, Any], base_dir: Path, budget: Optional[VramBudget]) -> TunePlan:
    from shared.services.ltx25_service import build_live_plan_text

    out = TunePlan(budget=budget)
    out.updates = tuned_values(settings)
    try:
        text = build_live_plan_text(dict(settings, **out.updates), base_dir)
    except Exception as exc:  # informative only
        text = f"[Plan] unavailable: {exc}"
    for line in text.splitlines():
        line = line.replace("[Plan] ", "").strip()
        if line and not line.startswith("[Live]"):
            out.lines.append(line)
    out.lines.append(f"Token budget Auto (single chunk preferred), longest chunk {out.updates['max_chunk_frames']} frames "
                     "(20 s, the model's position range); transformer blocks that do not fit stream from RAM.")
    if budget is not None and budget.limit_gb < 9.0:
        out.warnings.append(f"Only {budget.limit_gb:.1f} GB is available for this run; LTX 2.5 needs about 9 GB "
                            "(a 12 GB GPU at Save VRAM 1.5: 8 and 10 GB cards stopped with an out-of-memory error, "
                            "never shared GPU memory). Lower Save VRAM, close other GPU apps or use a larger GPU.")
    elif budget is not None and budget.limit_gb < 10.0:
        out.warnings.append(f"Only {budget.limit_gb:.1f} GB is available for this run: chunks will be short and "
                            "most transformer blocks will stream from RAM (slow, but no shared GPU memory).")
    return out


def auto_apply(settings: Dict[str, Any], log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Runner hook: the engine sizes chunks and block residency from the VRAM free at its start."""
    updates = tuned_values(settings)
    settings.update(updates)
    if log:
        log(f"[LTX25 Auto Tune] Token budget Auto (single chunk preferred), longest chunk {updates['max_chunk_frames']} frames (20 s); chunk "
            "length and block streaming are sized from the VRAM free at engine start (Save VRAM "
            f"{settings.get('reserve_vram_gb')} GB kept free).")
    return updates
