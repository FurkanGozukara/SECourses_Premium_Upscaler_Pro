"""
Spatial transformer tiling for large LTX 2.5 canvases.

One latent canvas and one sampler trajectory; every model call is evaluated as
overlapping fixed-size spatial windows whose predictions are blended back onto
the canvas (LTX-2 1.4 `TiledDiffusionModel` / `FixedSizeSpatialTiling` with
trapezoidal weights, or the ComfyUI-LTXVideo tiled fusion sampler's Gaussian
weights). Each window is the problem the IC-LoRA was trained on: its own crop of
the guide, positions normalised to the window (so a 4K canvas never runs RoPE
past the 2048 px the model was trained on), the same per-token timesteps.

Blending the per-call velocity is the same as blending x0 (x0 = x - v * sigma is
affine with one sigma per call) and, for a deterministic sampler, the same as
fusing the stepped tiles; ancestral noise is drawn once on the canvas, so it
cannot seam.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch

SPATIAL_SCALE = 32  # pixels per latent cell


def plan_axis(canvas: int, tile: int, min_overlap: int, align: int = 1) -> List[int]:
    """Tile origins covering [0, canvas) with at least min_overlap; ends pinned (tiled_fusion_plan.plan_grid),
    interior origins rounded to `align` (2 for a factor-2 dilated guide)."""
    if canvas <= tile:
        return [0]
    tile = max(1, int(tile))
    min_overlap = min(max(0, int(min_overlap)), tile - 1)
    n = 2
    while n < canvas and (n * tile - canvas) / (n - 1) < min_overlap:
        n += 1
    step = (canvas - tile) / (n - 1)
    out = []
    for i in range(n):
        v = int(round(i * step))
        if align > 1:
            v = int(round(v / align)) * align
        v = max(0, min(canvas - tile, v))
        out.append(v)
    out[-1] = canvas - tile
    return sorted(set(out))


def _trapezoid_axis(origins: Sequence[int], i: int, tile: int, canvas: int) -> torch.Tensor:
    """Linear ramps over the overlap with each neighbour, flat 1 elsewhere (and at the canvas border)."""
    w = torch.ones(tile, dtype=torch.float32)
    o = origins[i]
    if i > 0:
        ov = origins[i - 1] + tile - o
        if ov > 0:
            ramp = (torch.arange(ov, dtype=torch.float32) + 0.5) / ov
            w[:ov] = torch.minimum(w[:ov], ramp)
    if i < len(origins) - 1:
        ov = o + tile - origins[i + 1]
        if ov > 0:
            ramp = (torch.arange(ov, dtype=torch.float32).flip(0) + 0.5) / ov
            w[tile - ov:] = torch.minimum(w[tile - ov:], ramp)
    return w


def _gaussian_axis(tile: int, var: float) -> torch.Tensor:
    x = torch.arange(tile, dtype=torch.float32)
    mid = (tile - 1) / 2
    return torch.exp(-((x - mid) ** 2) / (tile * tile * 2 * var))


@dataclass
class SpatialTiling:
    canvas_h: int
    canvas_w: int
    tile_h: int
    tile_w: int
    ys: List[int]
    xs: List[int]
    weights: List[torch.Tensor]  # one [tile_h, tile_w] weight per (y, x) origin, row-major
    blend: str

    @property
    def origins(self) -> List[Tuple[int, int]]:
        return [(y, x) for y in self.ys for x in self.xs]

    def describe(self) -> str:
        return (f"{len(self.xs)}x{len(self.ys)} windows of {self.tile_w * SPATIAL_SCALE}x{self.tile_h * SPATIAL_SCALE} px "
                f"over {self.canvas_w * SPATIAL_SCALE}x{self.canvas_h * SPATIAL_SCALE} ({self.blend} blend)")


def build_tiling(
    canvas_h: int,
    canvas_w: int,
    tile_h_px: int,
    tile_w_px: int,
    overlap: float = 0.5,
    blend: str = "trapezoid",
    blend_var: float = 0.05,
    align: int = 1,
) -> Optional[SpatialTiling]:
    """Latent-grid tiling; None when the canvas fits one window."""
    th = max(align, (int(tile_h_px) // SPATIAL_SCALE) // align * align)
    tw = max(align, (int(tile_w_px) // SPATIAL_SCALE) // align * align)
    th, tw = min(th, canvas_h), min(tw, canvas_w)
    ys = plan_axis(canvas_h, th, max(1, int(th * overlap)), align)
    xs = plan_axis(canvas_w, tw, max(1, int(tw * overlap)), align)
    if len(ys) == 1 and len(xs) == 1:
        return None
    weights = []
    for iy, _y in enumerate(ys):
        for ix, _x in enumerate(xs):
            if blend == "gaussian":
                wy = _gaussian_axis(th, blend_var) if len(ys) > 1 else torch.ones(th)
                wx = _gaussian_axis(tw, blend_var) if len(xs) > 1 else torch.ones(tw)
            else:
                wy = _trapezoid_axis(ys, iy, th, canvas_h)
                wx = _trapezoid_axis(xs, ix, tw, canvas_w)
            weights.append(wy[:, None] * wx[None, :])
    return SpatialTiling(canvas_h, canvas_w, th, tw, ys, xs, weights, blend)


def crop_keyframes(keyframe_idxs: torch.Tensor, guide_frames: int, canvas_h: int, canvas_w: int,
                   y0: int, x0: int, th: int, tw: int) -> torch.Tensor:
    """Crop the guide's pixel coordinates [1, 3, Fg*H*W, 2] to one window and normalise them to its origin."""
    b = keyframe_idxs.shape[0]
    blk = keyframe_idxs.reshape(b, 3, guide_frames, canvas_h, canvas_w, 2)
    blk = blk[:, :, :, y0:y0 + th, x0:x0 + tw, :].clone()
    blk[:, 1] -= y0 * SPATIAL_SCALE
    blk[:, 2] -= x0 * SPATIAL_SCALE
    return blk.reshape(b, 3, guide_frames * th * tw, 2)


def crop_entries(entries: List[dict], guide_frames: int, th: int, tw: int, factor: int) -> List[dict]:
    out = []
    for e in entries:
        ne = dict(e)
        ne["pre_filter_count"] = int(guide_frames * th * tw)
        ne["latent_shape"] = [int(guide_frames), int(th // max(1, factor)), int(tw // max(1, factor))]
        out.append(ne)
    return out


class TiledVelocity:
    """Evaluate `run_tile(x_tile, mask_tile, keyframes_tile, entries_tile)` per window and blend the outputs."""

    def __init__(self, tiling: SpatialTiling, guide_frames: int, factor: int,
                 keyframe_idxs: torch.Tensor, entries: List[dict], device: torch.device):
        self.t = tiling
        self.guide_frames = int(guide_frames)
        self.device = device
        self.kf = []
        self.entries = []
        self.weights = []
        for (y0, x0), w in zip(tiling.origins, tiling.weights):
            self.kf.append(crop_keyframes(keyframe_idxs, self.guide_frames, tiling.canvas_h, tiling.canvas_w,
                                          y0, x0, tiling.tile_h, tiling.tile_w).to(device))
            self.entries.append(crop_entries(entries, self.guide_frames, tiling.tile_h, tiling.tile_w, factor))
            self.weights.append(w.to(device))
        wsum = torch.zeros((tiling.canvas_h, tiling.canvas_w), dtype=torch.float32, device=device)
        for (y0, x0), w in zip(tiling.origins, self.weights):
            wsum[y0:y0 + tiling.tile_h, x0:x0 + tiling.tile_w] += w
        self.wsum = wsum.clamp_min(1e-12)

    def __call__(self, x: torch.Tensor, mask: torch.Tensor,
                 run_tile: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, List[dict]], torch.Tensor]) -> torch.Tensor:
        t = self.t
        acc = torch.zeros_like(x, dtype=torch.float32)
        for i, (y0, x0) in enumerate(t.origins):
            xs = x[..., y0:y0 + t.tile_h, x0:x0 + t.tile_w]
            ms = mask[..., y0:y0 + t.tile_h, x0:x0 + t.tile_w]
            out = run_tile(xs, ms, self.kf[i], self.entries[i]).float()
            acc[..., y0:y0 + t.tile_h, x0:x0 + t.tile_w] += out * self.weights[i]
        return acc / self.wsum


def tiling_needed(canvas_h_px: int, canvas_w_px: int, limit_px: int = 2048) -> bool:
    """Auto rule: the model's spatial positions are trained up to 2048 px (max_pos 2048)."""
    return max(int(canvas_h_px), int(canvas_w_px)) > int(limit_px)
