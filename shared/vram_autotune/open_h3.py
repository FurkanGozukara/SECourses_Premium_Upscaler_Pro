"""
Open-H3 Regenerate Auto Tune.

The engine already sizes each window from the VRAM that is free when it starts (token budget =
free - Save VRAM - minimum resident weights, about 145 KB per packed token) and streams just
enough DiT blocks to RAM when a step needs more. Auto Tune therefore sets the limits that let it
reach the best quality first and the best speed second:

- Quality: one window over the whole clip when it fits (up to the model's 361 frames); windows
  are cross-faded otherwise, and every window boundary is a regeneration seam.
- Speed: all 50 DiT blocks may stay on the GPU (the engine only streams what does not fit).
- FFN chunking stays on: it is exact and frees activation VRAM for longer windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .common import TunePlan, VramBudget

MAX_WINDOW_FRAMES = 361
ALL_BLOCKS = 50
FFN_CHUNK_TOKENS = 32768
TOKEN_BUDGET_AUTO = "Auto (from free VRAM)"


def tuned_values(settings: Dict[str, Any]) -> Dict[str, Any]:
    ffn = settings.get("ffn_chunk_tokens")
    try:
        ffn_val = int(ffn or 0)
    except Exception:
        ffn_val = 0
    return {
        "token_budget_mode": TOKEN_BUDGET_AUTO,
        "max_window_frames": MAX_WINDOW_FRAMES,
        "max_gpu_blocks": ALL_BLOCKS,
        "ffn_chunk_tokens": ffn_val if ffn_val > 0 else FFN_CHUNK_TOKENS,
    }


def plan(settings: Dict[str, Any], base_dir: Path, budget: Optional[VramBudget]) -> TunePlan:
    from shared.services.open_h3_service import build_live_plan_text

    out = TunePlan(budget=budget)
    out.updates = tuned_values(settings)
    tuned = dict(settings, **out.updates)
    try:
        text = build_live_plan_text(tuned, base_dir)
    except Exception as exc:  # the plan text is informative only
        text = f"[Plan] unavailable: {exc}"
    for line in text.splitlines():
        line = line.replace("[Plan] ", "").strip()
        if line and not line.startswith("[Live]"):
            out.lines.append(line)
    out.lines.append("Token budget Auto, longest window 361 frames, all 50 DiT blocks allowed on the GPU "
                     "(the engine streams only what does not fit), FFN chunking on.")
    if budget is not None and budget.limit_gb < 8.0:
        out.warnings.append(f"Only {budget.limit_gb:.1f} GB is available for this run: windows will be short "
                            "and most DiT blocks will stream from RAM (slow but no shared GPU memory).")
    return out


def auto_apply(settings: Dict[str, Any], log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Runner hook: the engine does the VRAM sizing at start; only open its limits."""
    updates = tuned_values(settings)
    settings.update(updates)
    if log:
        log("[Open-H3 Auto Tune] Token budget Auto, max window 361 frames, max 50 DiT blocks on GPU; "
            "window length and block streaming are sized from the VRAM free at engine start "
            f"(Save VRAM {settings.get('reserve_vram_gb')} GB kept free).")
    return updates
