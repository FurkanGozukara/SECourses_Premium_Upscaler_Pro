"""
FlashVSR+ Auto Tune: best quality, then best speed, for one input on one VRAM budget, computed
instantly from measured memory formulas (no test runs).

Quality: frames per pass (Frame Chunk Size). FlashVSR streams a video with a KV cache of recent
frames; every chunk restarts that memory and pads its tail, so the whole clip in one pass is best.
Then DiT tiling OFF (tiles are diffused separately and feather-blended); when tiling is needed,
the fewest tiles (fewest seams) with a wider overlap, at the smallest tile size that keeps that tile
grid: a larger tile in the same grid only repeats work in the overlap (320x480 input, 2 tiles: 448 px
took 146 s and 22.4 GB, 320 px 75 s and 16.7 GB, the same picture within run-to-run noise).
KV Ratio 3 keeps the most temporal history.
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
# Reserved runs 1.18-1.26x allocated (1.26x with the LightVAE decoded each step). At 0.15 a 320x480 x4
# portrait clip (120 frames, INT8, LightVAE) was planned untiled at a 28.6 GB cap and ran out of VRAM;
# it passed at 29.0, 29.4 and 30.0 GB caps (benchmarks/vram/flashvsr.jsonl, v92_*). 0.20 plans it at 29.8 GB.
SAFETY_FRACTION = 0.20
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


def tile_overlap(tile: int) -> int:
    return min(TILED_OVERLAP, max(8, tile // 4))


def tile_grid(tile: int, overlap: int, in_w: int, in_h: int) -> Tuple[int, int]:
    """Rows and columns of DiT tiles, as the FlashVSR engine lays them out (calculate_tile_coords)."""
    stride = max(1, tile - overlap)
    return (max(1, math.ceil(max(1, in_h - overlap) / stride)), max(1, math.ceil(max(1, in_w - overlap) / stride)))


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
        overlap = tile_overlap(tile) if tiled else 24
        cand = FlashChoice(frames=frames, tiled=tiled, tile=tile if tiled else 256, overlap=overlap,
                           gb=peak_gb(mp, frames, kv_ratio, mode, precision, vae_model))
        # Prefer the longest pass; among equal passes the first (untiled, then largest tile) wins.
        if best is None or cand.frames > best.frames:
            best = cand
        if cand.frames >= total:
            break
    if best is not None and best.tiled:
        # Same tile grid (same seams) and seam overlap at the smallest tile: less repeated work in the
        # overlaps, less VRAM. Smaller tiles never need more memory, so the pass length still fits.
        rows, cols = tile_grid(best.tile, best.overlap, in_w, in_h)
        for tile in sorted(t for t in TILE_CHOICES if t < best.tile):
            overlap = tile_overlap(tile)
            r, c = tile_grid(tile, overlap, in_w, in_h)
            if overlap >= best.overlap and r <= rows and c <= cols:
                best = FlashChoice(frames=best.frames, tiled=True, tile=tile, overlap=overlap,
                                   gb=peak_gb(tile_mp(tile, in_w, in_h, scale), best.frames, kv_ratio, mode,
                                              precision, vae_model))
                break
    if best is not None and best.frames < total:
        why = "model limit" if best.frames >= MAX_PASS_FRAMES else "VRAM/RAM limit"
        notes.append(f"The clip is split into passes of {best.frames} frames ({why}).")
    return best, notes


# --- FlashVSR+ (official Turbo / Base presets) -------------------------------------------------------------
# One stream over the whole clip: memory does not grow with the clip length (decoded in segments), so the plan
# only chooses KV-cache residency and, when the frame does not fit, DiT tiles. The TC decoder always runs serial:
# its parallel executor took 17.6 GB instead of 5.7-6.8 GB at 720p and was not faster (results/WP-FLASHVSR/calib).
# Peak PyTorch-RESERVED GB per padded output megapixel, measured on the RTX 5090 (bf16, decode_every 16,
# temporal compression 4, serial decoder); linear between points, beyond the last point with the steeper of the
# last-segment and the average slope. Turbo 4K untiled without KV offload and Base 4K untiled ran out of memory
# at a 28.4 GB cap.
PLUS_PRESETS = {"plus-turbo": "turbo", "plus-base": "base"}
PLUS_RESERVED_GB = {  # (padded output MP, reserved GB)
    ("turbo", "never"): ((0.983, 6.75), (2.212, 11.05), (3.932, 15.38)),
    ("turbo", "always"): ((3.932, 14.47), (8.356, 26.76)),
    ("base", "never"): ((0.983, 8.04), (2.212, 17.94), (3.932, 19.60)),
    ("base", "always"): ((3.932, 14.47),),
}
PLUS_INT8_EXTRA_GB = 0.0        # INT8 ConvRot planned like BF16: Turbo 1080p 10.24 vs 10.74 GB reserved (2026-11-02, after
                                # the FP32-activation fix; the old FP32 path reserved 12.16 vs 11.05 GB)
PLUS_TC_FACTOR = {4: 1.0, 2: 1.45, 1: 1.45}  # non-weight memory vs temporal compression 4 (robot 1080p: 14.17 vs 11.05)
PLUS_WEIGHTS_GB = 3.6
PLUS_MARGIN = 1.10
PLUS_PINNED_RAM_SHARE = 0.25  # pinned host memory granted reliably (Base 4K needs ~36 GB: allocation failed on 96 GB)


def plus_kv_offload_gb(mp: float, preset: str) -> float:
    """Pinned RAM the KV offload needs: per offloaded layer K+V of 3 windows x 2 latent frames (bf16)."""
    layers = 5 if preset == "turbo" else 30
    tokens = mp * 1e6 / 256.0 * 6
    return layers * 2 * tokens * 1536 * 2 / 1e9


def _interp(points, mp: float) -> float:
    pts = sorted(points)
    x0, y0 = pts[0]
    if mp <= x0:
        return PLUS_WEIGHTS_GB + (y0 - PLUS_WEIGHTS_GB) * mp / x0
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        if mp <= xb:
            return ya + (yb - ya) * (mp - xa) / (xb - xa)
    if len(pts) == 1:
        return PLUS_WEIGHTS_GB + (y0 - PLUS_WEIGHTS_GB) * mp / x0
    (xa, ya), (xb, yb) = pts[-2], pts[-1]
    slope = max((yb - ya) / (xb - xa), (yb - y0) / max(1e-6, xb - x0))
    return yb + slope * (mp - xb)


def plus_peak_gb(mp: float, preset: str, precision: str, kv: str, temporal_compression: int = 4) -> float:
    """Estimated reserved GB of one FlashVSR+ stream (untiled frame or one tile), with margin."""
    kv = "always" if kv == "always" else "never"
    gb = _interp(PLUS_RESERVED_GB[(preset, kv)], mp)
    if kv == "always":
        gb = min(gb, _interp(PLUS_RESERVED_GB[(preset, "never")], mp))
    gb = PLUS_WEIGHTS_GB + (gb - PLUS_WEIGHTS_GB) * PLUS_TC_FACTOR.get(int(temporal_compression), 1.45)
    if precision == "int8_convrot":
        gb += PLUS_INT8_EXTRA_GB
    return gb * PLUS_MARGIN


def _plus_ram_gb() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().total) / 1e9
    except Exception:
        return 64.0


def plan_plus(settings: Dict[str, Any], *, in_w: int, in_h: int, frames: int, budget: VramBudget,
              image: bool = False) -> TunePlan:
    out = TunePlan(budget=budget)
    preset = PLUS_PRESETS[str(settings.get("version")).strip().lower()]
    scale = 2 if str(settings.get("scale", "4")).strip() == "2" else 4
    precision = str(settings.get("precision") or "bf16").strip().lower()
    try:
        tc = int(settings.get("temporal_compression") or 4)
    except Exception:
        tc = 4
    cap = budget.limit_gb - FLASHVSR_PROCESS_OVERHEAD_GB
    mp = out_mp(in_w, in_h, scale)
    pinned_cap = _plus_ram_gb() * PLUS_PINNED_RAM_SHARE
    speed = {"keep_models_on_cpu": False, "force_offload": False, "unload_dit": False, "plus_tiny_decoder": "serial"}
    out.lines.append(f"FlashVSR+ {preset.title()}: input {in_w}x{in_h} x{scale} -> {in_w * scale}x{in_h * scale} "
                     f"(works at {_pad128(in_w * scale)}x{_pad128(in_h * scale)}), "
                     + ("single image" if image else f"{frames} frame(s)") + f", temporal compression {tc}, "
                     "one stream over the whole clip")
    for kv in ("never", "always"):
        if kv == "always" and plus_kv_offload_gb(mp, preset) > pinned_cap:
            continue
        gb = plus_peak_gb(mp, preset, precision, kv, tc)
        if gb <= cap:
            out.updates = {"tiled_dit": False, "plus_kv_offload": kv, **speed}
            out.estimate_gb = gb + FLASHVSR_PROCESS_OVERHEAD_GB
            out.lines.append("DiT tiling OFF (full frame), KV cache " + (
                "of the full-attention layers in pinned RAM between steps" if kv == "always" else "on the GPU"))
            return out
    tile_pick = None
    for tile in TILE_CHOICES:
        if tile >= max(in_w, in_h):
            continue
        gb = plus_peak_gb(tile_mp(tile, in_w, in_h, scale), preset, precision, "never", tc)
        if gb <= cap:
            tile_pick = (tile, gb)
            break
    if tile_pick is None:
        out.fits = False
        out.warnings.append(f"Even {MIN_TILE}px FlashVSR+ tiles do not fit the {max(0.0, cap):.1f} GB budget.")
        out.updates = {"tiled_dit": True, "tile_size": MIN_TILE, "overlap": 24, "plus_kv_offload": "never", **speed}
        return out
    tile, gb = tile_pick
    overlap = tile_overlap(tile)
    rows, cols = tile_grid(tile, overlap, in_w, in_h)
    for smaller in sorted(t for t in TILE_CHOICES if t < tile):  # same grid, less repeated work in the overlaps
        ov = tile_overlap(smaller)
        r, c = tile_grid(smaller, ov, in_w, in_h)
        if ov >= overlap and r <= rows and c <= cols:
            tile, overlap = smaller, ov
            gb = plus_peak_gb(tile_mp(tile, in_w, in_h, scale), preset, precision, "never", tc)
            break
    chunk = min(max(1, int(frames)), ram_frames_limit(in_w, in_h, scale, True), MAX_PASS_FRAMES)
    out.updates = {"tiled_dit": True, "tile_size": tile, "overlap": overlap, "plus_kv_offload": "never",
                   "frame_chunk_size": 0 if chunk >= frames else chunk, **speed}
    out.estimate_gb = gb + FLASHVSR_PROCESS_OVERHEAD_GB
    out.lines.append(f"DiT tiling ON: {tile}px tiles, {overlap}px overlap "
                     f"({'x'.join(map(str, tile_grid(tile, overlap, in_w, in_h)))} tiles); each frame chunk "
                     f"of {chunk} frames streams per tile")
    return out


def plan(settings: Dict[str, Any], *, in_w: int, in_h: int, frames: int, budget: Optional[VramBudget],
         image: bool = False) -> TunePlan:
    out = TunePlan(budget=budget)
    if budget is None:
        out.warnings.append("No NVIDIA GPU selected or detected; settings left unchanged.")
        return out
    if str(settings.get("version") or "").strip().lower() in PLUS_PRESETS:
        return plan_plus(settings, in_w=in_w, in_h=in_h, frames=frames, budget=budget, image=image)
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
                     f"DiT tiling ON: {choice.tile}px tiles, {choice.overlap}px overlap "
                     f"({'x'.join(map(str, tile_grid(choice.tile, choice.overlap, in_w, in_h)))} tiles, the fewest that fit)")
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
