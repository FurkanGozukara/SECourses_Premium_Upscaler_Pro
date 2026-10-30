"""
SparkVSR Auto Tune: best quality, then best speed, for one input on one VRAM budget, computed
instantly from measured memory formulas (no test runs).

Quality: Temporal Chunk Length. SparkVSR (a CogVideoX-1.5-class 5.6B video transformer) restores
a chunk of frames jointly, and chunks are joined with a hard cut inside their overlap, so longer
chunks are steadier (references also reach more frames). Then no spatial tiling (tiles are joined
with a hard cut too); when tiling is needed, the largest tiles.
Speed: the whole transformer stays on the GPU; block-level group offload only when its weights
do not fit next to the activations.

Memory (PyTorch allocated, RTX 5090; benchmarks/vram/sparkvsr.jsonl), split-stage mode: the
transformer stage dominates = weights + activations per token, where a chunk of 8n+1 frames at
HxW (output, padded to 16) has (n+1) * (H/16) * (W/16) video tokens (+226 text tokens).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .common import TunePlan, mark_tightness, VramBudget, probe_media, resolve_budget, round_down_8n1, round_up_8n1

# Non-allocator VRAM of a SparkVSR run: the CLI parent plus one stage child (split mode).
SPARKVSR_PROCESS_OVERHEAD_GB = 1.0

MAX_CHUNK = 129            # longest temporal window the previous probe campaigns validated
MIN_CHUNK = 17
TEXT_TOKENS = 226
# Transformer stage, PyTorch allocated GiB = weights + per 1k tokens (benchmarks/vram/sparkvsr.jsonl):
# BF16 10.4 + 0.076 (11.78 / 12.88 / 15.06 / 15.99 GB at 18k / 33k / 61k / 74k tokens);
# INT8 ConvRot 6.3 + 0.170 (9.40 GB at 18k, 18.84 GB at 74k: its per-token activation quantization
# buffers make it use MORE VRAM than BF16 above about 44k tokens). FP8 Scaled is unmeasured: budgeted
# between the two, on the safe side.
WEIGHTS_GB = {"bf16": 10.4, "fp8": 10.4, "int8": 6.3}
GB_PER_KTOK_BY_KIND = {"bf16": 0.076, "fp8": 0.12, "int8": 0.170}
EXTRA_GB = 0.0             # measured intercepts equal the weights
GROUP_OFFLOAD_RESIDENT_GB = 0.35  # block-level group offload keeps ~0.3 GB of the transformer resident
                                 # (BF16 1080p/65 frames: 5.90 GB instead of 15.99, 152 s vs 160 s: one pass per
                                 # chunk streams the weights once, so it costs almost no speed). BF16/FP8 only:
                                 # INT8 ConvRot with group offload used MORE (1080p x 65 frames failed at 19.6 GB
                                 # allocated + 3.4 GB asked, planned 14.9; 1024 tiles peaked at 20.35 GB, +6.7 GB
                                 # over INT8 without it), so INT8 plans never use it.
VAE_STAGE_GB_PER_MP = 0.12  # tiled VAE decode: 4.1 GB at both 720p and 1080p (fixed-size tiles)
NOSPLIT_EXTRA_GB = 1.0     # one process keeps the VAE resident next to the transformer (+0.91 GB measured)
# One process also loads the 8.9 GB BF16 text encoder onto the GPU (with Group Offload all of it at once:
# a 10 GB card failed at load); separate stages stream it layer by layer when it does not fit.
TEXT_ENCODER_GB = 9.2
SAFETY_GB = 0.6
SAFETY_FRACTION = 0.10
# Larger tiles first: fewer seams and less per-tile overhead (4K x 33 frames peaked at 11.8 GB with
# 1024 px tiles on a 29.6 GB budget; the whole frame would need about 34 GB).
TILE_CHOICES = (2048, 1536, 1024, 896, 768, 640, 512, 384, 256)
# Plans stay inside the measured range: a 2048 px tile x 33 frames (82k tokens) peaked 2.5 GB above its
# estimate and ran 51% slower than 1024 px tiles (attention grows with the square of the tile's tokens).
MAX_PLAN_TOKENS = 75_000
TILE_OVERLAP = 32


def _pad16(v: float) -> int:
    return int(math.ceil(max(1.0, v) / 16.0) * 16)


def model_kind(model_name: str) -> str:
    n = str(model_name or "").lower()
    if "int8" in n:
        return "int8"
    if "fp8" in n:
        return "fp8"
    return "bf16"


def tokens(frames: int, w: int, h: int) -> int:
    f = round_up_8n1(max(1, int(frames)))
    n = (f - 1) // 8
    return (n + 1) * (_pad16(w) // 16) * (_pad16(h) // 16) + TEXT_TOKENS


def transform_gb(kind: str, frames: int, w: int, h: int, group_offload: bool = False, cfg_double: bool = False) -> float:
    weights = WEIGHTS_GB.get(kind, WEIGHTS_GB["bf16"])
    if group_offload and GROUP_OFFLOAD_RESIDENT_GB is not None:
        weights = GROUP_OFFLOAD_RESIDENT_GB
    toks = tokens(frames, w, h) * (2 if cfg_double else 1)
    return weights + EXTRA_GB + GB_PER_KTOK_BY_KIND.get(kind, 0.170) * toks / 1000.0


def vae_gb(w: int, h: int) -> float:
    return 4.0 + VAE_STAGE_GB_PER_MP * (_pad16(w) * _pad16(h) / 1e6)


def _with_margin(gb: float) -> float:
    return gb * (1.0 + SAFETY_FRACTION) + SAFETY_GB


@dataclass
class SparkChoice:
    chunk: int
    tile: int           # 0 = full frame
    group_offload: bool
    gb: float


def choose(kind: str, out_w: int, out_h: int, total_frames: int, cap_gb: float,
           cfg_double: bool = False) -> Tuple[Optional[SparkChoice], List[str]]:
    notes: List[str] = []
    want = min(MAX_CHUNK, round_up_8n1(max(1, int(total_frames))))
    want = max(MIN_CHUNK if total_frames > 1 else 1, want)
    chunk = want
    tiles = [0] + [t for t in TILE_CHOICES if t < max(out_w, out_h)]
    while chunk >= MIN_CHUNK or (chunk == want):
        for tile in tiles:
            tw = min(out_w, tile) if tile else out_w
            th = min(out_h, tile) if tile else out_h
            if _with_margin(vae_gb(tw, th)) > cap_gb:
                continue
            if tokens(chunk, tw, th) > MAX_PLAN_TOKENS:
                continue
            for group in ((False, True) if GROUP_OFFLOAD_RESIDENT_GB is not None and kind != "int8" else (False,)):
                gb = transform_gb(kind, chunk, tw, th, group, cfg_double)
                if _with_margin(gb) <= cap_gb:
                    return SparkChoice(chunk=chunk, tile=tile, group_offload=group, gb=gb), notes
        if chunk <= MIN_CHUNK:
            break
        chunk = max(MIN_CHUNK, round_down_8n1(chunk - 8))
    return None, notes


def plan(settings: Dict[str, Any], *, out_w: int, out_h: int, frames: int, budget: Optional[VramBudget],
         image: bool = False) -> TunePlan:
    cap_scale = max(0.3, min(1.0, float(settings.get("_autotune_cap_scale") or 1.0)))
    out = TunePlan(budget=budget)
    if budget is None:
        out.warnings.append("No NVIDIA GPU selected or detected; settings left unchanged.")
        return out
    kind = model_kind(settings.get("model_name"))
    cfg_double = abs(float(settings.get("ref_guidance_scale") or 1.0) - 1.0) > 1e-6
    cap = (budget.limit_gb - SPARKVSR_PROCESS_OVERHEAD_GB) * cap_scale
    choice, notes = choose(kind, out_w, out_h, 1 if image else frames, cap, cfg_double)
    out.lines.append(f"Output {out_w}x{out_h}, " + ("single image" if image else f"{frames} frame(s)")
                     + f", model {settings.get('model_name')}")
    out.lines.extend(notes)
    if choice is None:
        out.fits = False
        hint = ""
        if kind == "int8" and choose("bf16", out_w, out_h, 1 if image else frames, cap, cfg_double)[0] is not None:
            hint = (" The SparkVSR-bf16 model fits this card: its transformer streams from RAM with Group Offload, "
                    "while INT8 ConvRot has to stay on the GPU.")
        out.warnings.append(f"Even {MIN_CHUNK}-frame chunks with 256px tiles do not fit the {max(0.0, cap):.1f} GB "
                            "available. Lower the output size, close other GPU apps or lower Save VRAM." + hint)
        out.updates.update({"chunk_len": MIN_CHUNK, "overlap_t": 8, "tile_height": 256, "tile_width": 256,
                            "overlap_height": TILE_OVERLAP, "overlap_width": TILE_OVERLAP,
                            "group_offload": kind != "int8", "num_blocks_per_group": 1, "vae_tiling": True})
        return out
    whole = (not image) and choice.chunk >= round_up_8n1(frames)
    # Speed: one process (no stage subprocesses) avoids reloading the weights for every chunk and
    # tile (1080p, 33 frames: 79 s instead of 122 s) at about 1 GB more peak VRAM.
    single_process = (not choice.group_offload and _with_margin(choice.gb + NOSPLIT_EXTRA_GB) <= cap
                      and _with_margin(TEXT_ENCODER_GB) <= cap)
    out.updates = {
        "chunk_len": 0 if (whole or image) else int(choice.chunk),
        "overlap_t": 8 if not whole else int(settings.get("overlap_t") or 8),
        "tile_height": int(choice.tile),
        "tile_width": int(choice.tile),
        "overlap_height": TILE_OVERLAP,
        "overlap_width": TILE_OVERLAP,
        "group_offload": bool(choice.group_offload),
        "num_blocks_per_group": 1 if choice.group_offload else int(settings.get("num_blocks_per_group") or 1),
        "vae_tiling": True,
        "split_stage_subprocesses": not single_process,
    }
    out.estimate_gb = _with_margin(choice.gb + (NOSPLIT_EXTRA_GB if single_process else 0.0)) + SPARKVSR_PROCESS_OVERHEAD_GB
    if not image:
        out.lines.append("Temporal Chunk Length " + ("0 (whole clip as one chunk)" if whole else
                         f"{choice.chunk} (longest that fits; {math.ceil(frames / max(1, choice.chunk - 8))} chunks)"))
    out.lines.append("Spatial tiling OFF (full frame)" if not choice.tile else
                     f"Spatial tiles {choice.tile}x{choice.tile} (largest that fits)")
    out.lines.append("Transformer fully on the GPU" if not choice.group_offload else
                     "Block group offload ON (transformer weights stream from RAM; about the same speed for SparkVSR)")
    out.lines.append("Stage Subprocess Isolation OFF (one process, faster)" if single_process else
                     "Stage Subprocess Isolation ON (each stage frees its VRAM; needed for this budget)")
    return out


def auto_apply(settings: Dict[str, Any], input_path: str, scale: int,
               log: Optional[Callable[[str], None]] = None) -> Optional[TunePlan]:
    """Runner hook: tune for this exact input and the VRAM free right now."""
    media = probe_media(input_path)
    if media is None:
        return None
    try:
        start = max(0, int(settings.get("start_frame") or 0))
        end = int(settings.get("end_frame") if settings.get("end_frame") is not None else -1)
    except Exception:
        start, end = 0, -1
    frames = media.frames
    frames = max(1, (min(frames, end + 1) if end >= 0 else frames) - start)
    budget = resolve_budget(settings.get("device") or os.environ.get("SECOURSES_GLOBAL_GPU_DEVICE"),
                            settings.get("save_vram_gb"), overhead_gb=SPARKVSR_PROCESS_OVERHEAD_GB)
    tp = plan(settings, out_w=media.width * scale, out_h=media.height * scale, frames=frames, budget=budget,
              image=media.kind == "image")
    settings.update(tp.updates)
    if not settings.get("_autotune_cap_scale"):
        mark_tightness(settings, tp)
    if log:
        for line in tp.report("[SparkVSR Auto Tune]").splitlines():
            log(line)
    return tp
