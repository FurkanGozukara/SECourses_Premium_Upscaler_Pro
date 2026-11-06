"""
RelayVSR Auto Tune and VRAM placement (one formula for the tab's Auto Tune and the engine's run-time placement).

RelayVSR streams one frame at a time, so memory does not grow with clip length. What uses VRAM:
- weights: Flash decoder 0.13 GB, Wan non-block 0.23 GB, 30 Wan blocks of 0.163 GB (INT8 ConvRot) / 0.305 GB (BF16);
- Wan's KV cache: the last 4 keyframes' keys and values for every block = tokens x 4 x 3072 x 2 x 2 bytes per block,
  tokens = ceil(out_h / 32) x ceil(out_w / 32): 0.4 GB per block at a 3840x2160 output (12 GB for all 30 blocks);
- transient activations of one step: 0.4 GB + 0.25 GB per output megapixel.
Measured on an RTX 5090 (relayvsr_work/mem_profile*.py, out/probe_*.log): INT8, all blocks on the GPU, peak allocated
8.8 GB at 1920x1080 and 18.4-18.8 GB at 3840x2160 output; the formula gives 8.9 / 19.0 GB. Capped at 16 / 10 GB
(relayvsr_work/capped_run.py) the 4K run kept 24 / 13 blocks on the GPU, peaked at 13.9 / 8.3 GB and gave the same output.
A block that does not fit stays in pinned RAM together with its KV cache and is copied to the GPU only while it runs
(Wan runs once per keyframe window, so the copies cost a fraction of the time). Quality is identical either way.
"""
from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, Optional

from .common import TunePlan, VramBudget, mark_tightness, probe_media, resolve_budget

RELAYVSR_PROCESS_OVERHEAD_GB = 0.8           # CUDA context + kernels of the engine process
WAN_BLOCKS = 30
WAN_FIXED_GB = 0.23
WAN_BLOCK_GB = {"int8": 0.163, "bf16": 0.305}
FLASH_GB = 0.13
KV_FRAMES, HIDDEN, KV_BYTES = 4, 3072, 2
ACT_BASE_GB, ACT_GB_PER_OUT_MP = 0.4, 0.25  # measured 0.74 GB at 2.1 MP, 2.34 GB at 8.3 MP output (peak - weights - KV)
SAFETY_GB = 0.5
GIB = 1024 ** 3


def wan_kind(settings_or_name: Any) -> str:
    name = settings_or_name.get("wan_model") if isinstance(settings_or_name, dict) else settings_or_name
    return "bf16" if "bf16" in str(name or "").lower() else "int8"


def kv_block_gb(out_w: int, out_h: int) -> float:
    tokens = math.ceil(out_h / 32) * math.ceil(out_w / 32)
    return tokens * KV_FRAMES * HIDDEN * 2 * KV_BYTES / GIB


def activation_gb(out_w: int, out_h: int) -> float:
    return ACT_BASE_GB + ACT_GB_PER_OUT_MP * out_w * out_h / 1e6


def blocks_on_gpu(cap_gb: float, kind: str, out_w: int, out_h: int) -> int:
    """Wan blocks (weights + KV cache) that fit in cap_gb next to everything else."""
    fixed = FLASH_GB + WAN_FIXED_GB + activation_gb(out_w, out_h) + SAFETY_GB
    per = WAN_BLOCK_GB[kind] + kv_block_gb(out_w, out_h)
    return max(0, min(WAN_BLOCKS, int(math.floor((cap_gb - fixed) / per))))


def peak_gb(kind: str, out_w: int, out_h: int, resident: int) -> float:
    per = WAN_BLOCK_GB[kind] + kv_block_gb(out_w, out_h)
    streamed = 1 if resident < WAN_BLOCKS else 0     # one streamed block (weights + KV) on the GPU at a time
    return FLASH_GB + WAN_FIXED_GB + activation_gb(out_w, out_h) + (resident + streamed) * per


def plan(settings: Dict[str, Any], *, in_w: int, in_h: int, frames: int, budget: Optional[VramBudget],
         image: bool = False) -> TunePlan:
    out = TunePlan(budget=budget)
    out_w, out_h = in_w * 4, in_h * 4
    kind = wan_kind(settings)
    if budget is None:
        out.updates["wan_gpu_blocks"] = -1
        out.lines.append("No GPU budget known: every Wan block that fits stays on the GPU (decided at run time).")
        return out
    cap = budget.limit_gb - RELAYVSR_PROCESS_OVERHEAD_GB
    blocks = blocks_on_gpu(cap, kind, out_w, out_h)
    out.estimate_gb = peak_gb(kind, out_w, out_h, blocks) + RELAYVSR_PROCESS_OVERHEAD_GB
    out.updates["wan_gpu_blocks"] = -1 if blocks >= WAN_BLOCKS else blocks
    out.lines.append(f"Output {out_w}x{out_h} ({out_w * out_h / 1e6:.1f} MP), {'image' if image else f'{frames} frames'}: "
                     f"Wan KV cache {kv_block_gb(out_w, out_h) * WAN_BLOCKS:.1f} GB, activations "
                     f"{activation_gb(out_w, out_h):.1f} GB")
    if blocks >= WAN_BLOCKS:
        out.lines.append(f"All {WAN_BLOCKS} Wan blocks and their KV cache on the GPU (fastest)")
    else:
        out.lines.append(f"{blocks}/{WAN_BLOCKS} Wan blocks on the GPU; the rest stay in RAM with their KV cache and are "
                         f"copied to the GPU when Wan runs (once per window: same quality, somewhat slower)")
    if cap < peak_gb(kind, out_w, out_h, 0):
        out.fits = False
        out.warnings.append(f"this output size needs about {peak_gb(kind, out_w, out_h, 0) + RELAYVSR_PROCESS_OVERHEAD_GB:.1f} GB "
                            f"even with Wan streamed from RAM; lower Max Resolution or use pre-downscale")
    return out


def auto_apply(settings: Dict[str, Any], input_path: str, log: Optional[Callable[[str], None]] = None,
               scale: int = 4) -> Optional[TunePlan]:
    """Runner hook: tune for this exact input and the VRAM free right now."""
    media = probe_media(input_path)
    if media is None:
        return None
    try:
        start = max(0, int(settings.get("start_frame") or 0))
        end = int(settings.get("end_frame") if settings.get("end_frame") is not None else -1)
    except Exception:
        start, end = 0, -1
    frames = max(1, (min(media.frames, end + 1) if end >= 0 else media.frames) - start)
    budget = resolve_budget(settings.get("device") or os.environ.get("SECOURSES_GLOBAL_GPU_DEVICE"),
                            settings.get("save_vram_gb"), overhead_gb=RELAYVSR_PROCESS_OVERHEAD_GB)
    tp = plan(settings, in_w=media.width, in_h=media.height, frames=frames, budget=budget,
              image=media.kind == "image")
    settings.update(tp.updates)
    mark_tightness(settings, tp)
    if log:
        for line in tp.report("[RelayVSR Auto Tune]").splitlines():
            log(line)
    return tp
