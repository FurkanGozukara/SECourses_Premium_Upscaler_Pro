"""
FlashVSR+ Auto Tune: best quality, then best speed, for one input on one VRAM budget, computed
instantly from measured memory formulas (no test runs).

Quality: frames per pass (Frame Chunk Size). FlashVSR streams a video with a KV cache of recent
frames; every chunk restarts that memory and pads its tail, so the whole clip in one pass is best.
Then DiT tiling OFF (tiles are diffused separately and feather-blended); when tiling is needed,
the largest tiles with a wider overlap. KV Ratio 3 keeps the most temporal history.
Speed: models stay on the GPU (no CPU offload, no DiT unload between steps).

Memory (PyTorch allocated, RTX 5090, native allocator; benchmarks/vram/flashvsr.jsonl):
- weights + LQ projection + decoder: about 3.4 GB
- KV cache and activations: per padded output megapixel of the frame (or of one tile)
- the pass's upscaled frames on the GPU (full / tiny modes): 6 bytes per pixel per frame
- INT8 ConvRot DiT: about 1 GB less (1.75 GB of weights instead of 2.84 GB; -0.98 GB measured at 720p and 1080p)
- full mode decodes each step with the LightVAE: +0.44 GB per output megapixel (+0.39 GB at 720p, +0.97 GB
  at 1080p); the full Wan2.1 VAE needs about 8 GB per megapixel streamed, so Auto Tune decodes it in tiles
  (less than the tiny modes' TC decoder: 9.8 vs 11.3 GB at 720p)
System RAM also bounds a pass: the whole chunk is decoded (and, when tiled, blended) in RAM.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .common import TunePlan, mark_tightness, VramBudget, probe_media, resolve_budget

# Non-allocator VRAM of a FlashVSR+ CLI process (CUDA context, kernels, flash-attention).
FLASHVSR_PROCESS_OVERHEAD_GB = 0.7

WEIGHTS_GB = 3.4
GB_PER_MP = 7.7            # kv_ratio 3, dense attention, per padded output megapixel
KV_SHARE = 0.24            # share of GB_PER_MP that scales with kv_ratio (kv 3 -> 1 saved 2.7 GB at 2.2 MP)
FRAME_GB_PER_MP = 0.006    # upscaled frames kept on the GPU, per frame per megapixel (full/tiny)
INT8_WEIGHT_SAVING_GB = 1.0
LIGHTVAE_GB_PER_MP = 0.44  # full mode, LightVAE decoded per step (Wan2.1 is tiled instead)
SAFETY_GB = 0.6
SAFETY_FRACTION = 0.15     # reserved ran 1.18-1.19x allocated in calibration
MIN_TILE = 128
TILE_CHOICES = tuple(range(1024, MIN_TILE - 1, -32))
TILED_OVERLAP = 48         # wider overlap = smoother tile seams (24 is the minimum default)
RAM_SHARE = 0.5            # use at most half of the free system RAM for one pass
# The DiT's temporal RoPE table holds 1024 latent positions and a pass advances 2 per 8 frames, so one
# pass breaks after about 4,090 frames (a 4,200-frame pass crashed); longer clips stream in passes.
MAX_PASS_FRAMES = 4000


def _pad128(v: float) -> int:
    return int(math.ceil(max(1.0, v) / 128.0) * 128)


def frames_in_pass(frames: int) -> int:
    """Frames FlashVSR actually diffuses for a pass of `frames` (padded to 8n+5, then 8n+1)."""
    n = max(1, int(frames))
    padded = 21 if n < 21 else ((n - 5 + 7) // 8) * 8 + 5
    return ((padded + 4 - 1) // 8) * 8 + 1


def out_mp(in_w: int, in_h: int, scale: int) -> float:
    return _pad128(in_w * scale) * _pad128(in_h * scale) / 1e6


def tile_mp(tile: int, in_w: int, in_h: int, scale: int) -> float:
    return _pad128(min(tile, in_w) * scale) * _pad128(min(tile, in_h) * scale) / 1e6


def streams_lightvae(mode: str, vae_model: str) -> bool:
    return mode == "full" and str(vae_model or "") != "Wan2.1"


def peak_gb(mp: float, frames: int, kv_ratio: float, mode: str, precision: str = "bf16",
            vae_model: str = "LightVAE_W2.1") -> float:
    kv = max(1.0, min(3.0, float(kv_ratio or 3.0)))
    per_mp = GB_PER_MP * ((1.0 - KV_SHARE) + KV_SHARE * kv / 3.0)
    frame_term = 0.0 if mode == "tiny-long" else FRAME_GB_PER_MP * mp * frames_in_pass(frames)
    weights = WEIGHTS_GB - (INT8_WEIGHT_SAVING_GB if str(precision) == "int8_convrot" else 0.0)
    decode = LIGHTVAE_GB_PER_MP * mp if streams_lightvae(mode, vae_model) else 0.0
    return weights + per_mp * mp + frame_term + decode


def _with_margin(gb: float) -> float:
    return gb * (1.0 + SAFETY_FRACTION) + SAFETY_GB


def ram_frames_limit(in_w: int, in_h: int, scale: int, tiled: bool) -> int:
    """Longest pass the free system RAM holds (input frames + decoded output, x2 when tiled)."""
    try:
        import psutil

        avail = float(psutil.virtual_memory().available)
    except Exception:
        return 100000
    per_frame = in_w * in_h * 3 * 4 + (in_w * scale) * (in_h * scale) * 3 * (8 if tiled else 4)
    return max(21, int(avail * RAM_SHARE / max(1.0, per_frame)))


@dataclass
class FlashChoice:
    frames: int        # frames per pass (0 in the UI = the whole clip)
    tiled: bool
    tile: int
    overlap: int
    gb: float


def choose(in_w: int, in_h: int, scale: int, total_frames: int, cap_gb: float, kv_ratio: float = 3.0,
           mode: str = "full", precision: str = "bf16",
           vae_model: str = "LightVAE_W2.1") -> Tuple[Optional[FlashChoice], List[str]]:
    notes: List[str] = []
    total = max(1, int(total_frames))
    full_mp = out_mp(in_w, in_h, scale)
    # 1) untiled, whole clip (or the longest pass RAM and VRAM allow)
    best: Optional[FlashChoice] = None
    for tiled, tile in [(False, 0)] + [(True, t) for t in TILE_CHOICES]:
        if tiled and tile >= max(in_w, in_h):
            continue
        mp = tile_mp(tile, in_w, in_h, scale) if tiled else full_mp
        if _with_margin(peak_gb(mp, 21, kv_ratio, mode, precision, vae_model)) > cap_gb:
            continue  # not even a short pass fits at this tiling
        frames = min(total, ram_frames_limit(in_w, in_h, scale, tiled), MAX_PASS_FRAMES)
        while frames > 21 and _with_margin(peak_gb(mp, frames, kv_ratio, mode, precision, vae_model)) > cap_gb:
            frames = max(21, int(frames * 0.85))
        overlap = min(TILED_OVERLAP, max(8, tile // 4)) if tiled else 24
        cand = FlashChoice(frames=frames, tiled=tiled, tile=tile if tiled else 256, overlap=overlap,
                           gb=peak_gb(mp, frames, kv_ratio, mode, precision, vae_model))
        # Prefer the longest pass; among equal passes the first (untiled, then largest tile) wins.
        if best is None or cand.frames > best.frames:
            best = cand
        if cand.frames >= total:
            break
    if best is not None and best.frames < total:
        why = "model limit" if best.frames >= MAX_PASS_FRAMES else "VRAM/RAM limit"
        notes.append(f"The clip is split into passes of {best.frames} frames ({why}).")
    return best, notes


def plan(settings: Dict[str, Any], *, in_w: int, in_h: int, frames: int, budget: Optional[VramBudget],
         image: bool = False) -> TunePlan:
    out = TunePlan(budget=budget)
    if budget is None:
        out.warnings.append("No NVIDIA GPU selected or detected; settings left unchanged.")
        return out
    scale = 2 if str(settings.get("scale", "4")).strip() == "2" else 4
    mode = str(settings.get("mode") or "full").strip().lower()
    precision = str(settings.get("precision") or "bf16").strip().lower()
    vae_model = str(settings.get("vae_model") or "LightVAE_W2.1")
    cap = budget.limit_gb - FLASHVSR_PROCESS_OVERHEAD_GB
    choice, notes = choose(in_w, in_h, scale, 1 if image else frames, cap, 3.0, mode, precision, vae_model)
    out.lines.append(f"Model input {in_w}x{in_h} x{scale} -> {in_w * scale}x{in_h * scale} "
                     f"(works at {_pad128(in_w * scale)}x{_pad128(in_h * scale)}), "
                     + ("single image" if image else f"{frames} frame(s)") + f", mode {mode}")
    out.lines.extend(notes)
    speed = {"keep_models_on_cpu": False, "force_offload": False, "unload_dit": False, "kv_ratio": 3.0}
    if mode == "full":
        # LightVAE decodes each step untiled; the full Wan2.1 VAE only fits when decoded in tiles.
        speed["tiled_vae"] = not streams_lightvae(mode, vae_model)
    if choice is None:
        out.fits = False
        out.warnings.append(f"Even {MIN_TILE}px DiT tiles need about "
                            f"{_with_margin(peak_gb(tile_mp(MIN_TILE, in_w, in_h, scale), 21, 3.0, mode, precision, vae_model)):.1f} GB but only "
                            f"{max(0.0, cap):.1f} GB is available. Lower the output size, close other GPU apps or lower Save VRAM.")
        out.updates.update({"tiled_dit": True, "tile_size": MIN_TILE, "overlap": 24, "frame_chunk_size": 21, **speed,
                            "kv_ratio": 1.0})
        return out
    whole = choice.frames >= frames
    out.updates = {
        "frame_chunk_size": 0 if (whole or image) else int(choice.frames),
        "tiled_dit": bool(choice.tiled),
        "tile_size": int(choice.tile),
        "overlap": int(choice.overlap),
        **speed,
    }
    out.estimate_gb = _with_margin(choice.gb) + FLASHVSR_PROCESS_OVERHEAD_GB
    if not image:
        out.lines.append("Frame Chunk Size " + ("0 (whole clip in one pass)" if whole else str(choice.frames)))
    out.lines.append("DiT tiling OFF (full frame)" if not choice.tiled else
                     f"DiT tiling ON: {choice.tile}px tiles, {choice.overlap}px overlap (largest that fits)")
    if mode == "full":
        out.lines.append(f"Decoder {vae_model}" + (" in tiles (untiled it needs about 8 GB per output megapixel)"
                                                    if speed.get("tiled_vae") else " untiled"))
    out.lines.append("KV Ratio 3, models kept on the GPU (no CPU offload / DiT unload)")
    return out


def auto_apply(settings: Dict[str, Any], input_path: str, scale: int,
               log: Optional[Callable[[str], None]] = None) -> Optional[TunePlan]:
    """Runner hook: tune for this exact (already pre-downscaled) input and the VRAM free now."""
    media = probe_media(input_path)
    if media is None:
        return None
    try:
        start = max(0, int(settings.get("start_frame") or 0))
        end = int(settings.get("end_frame") if settings.get("end_frame") is not None else -1)
    except Exception:
        start, end = 0, -1
    frames = media.frames
    if end >= 0:
        frames = max(1, min(frames, end + 1) - start)
    else:
        frames = max(1, frames - start)
    budget = resolve_budget(settings.get("device") or os.environ.get("SECOURSES_GLOBAL_GPU_DEVICE"),
                            settings.get("save_vram_gb"), overhead_gb=FLASHVSR_PROCESS_OVERHEAD_GB)
    tp = plan(dict(settings, scale=scale), in_w=media.width, in_h=media.height, frames=frames, budget=budget,
              image=media.kind == "image")
    settings.update(tp.updates)
    mark_tightness(settings, tp)
    if log:
        for line in tp.report("[FlashVSR Auto Tune]").splitlines():
            log(line)
    return tp
