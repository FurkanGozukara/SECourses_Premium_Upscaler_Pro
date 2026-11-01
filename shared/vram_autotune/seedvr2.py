"""
SeedVR2 Auto Tune: the settings that give the best quality, then the best speed, for one input
on one VRAM budget, computed instantly from measured memory formulas (no test runs).

Quality: frames per batch. SeedVR2 restores a whole batch jointly, so the more frames share a
batch the steadier the result (no flicker or boundary pops between batches). The best batch holds
the whole clip (or the whole chunk/scene), up to SEEDVR2_MAX_BATCH_SIZE.
Speed: fewer swapped DiT blocks (BlockSwap streams them from RAM every step) and no VAE tiling
(tiling also adds slight seams, so larger tiles first).

The run is split into three processes (encode, DiT upscale, decode); each phase's peak must fit.
All formulas give PyTorch reserved memory in GB (what the cap limits) and were fitted on an
RTX 5090 with the native caching allocator (benchmarks/vram/seedvr2.jsonl):
- DiT: resident weights + activations per latent token (+ block swap staging).
- VAE: per-pixel working memory of one temporal slice of a tile (+ the batch's latents/frames).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .common import (
    TunePlan,
    mark_tightness,
    VramBudget,
    frames_per_unit,
    probe_media,
    resolve_budget,
    round_down_4n1,
    round_up_4n1,
)

# Non-allocator VRAM of a SeedVR2 run: the CLI parent's CUDA context plus the phase
# child's context and kernels (split-phase mode runs both at once). nvidia-smi totals minus the
# PyTorch reserved peak, 2026-11-01: 1.0 GB on the RTX 5090, up to 1.43 GB on the RTX 3090 (W4A8 with
# comfy_kitchen's kernels loaded).
SEEDVR2_PROCESS_OVERHEAD_GB = 1.45

MAX_BATCH = 601  # mirrors shared.ui_validators.SEEDVR2_MAX_BATCH_SIZE (kept import-free)
VAE_TILE_CHOICES = (1024, 896, 768, 640, 512, 384, 256)
VAE_TILE_OVERLAP = 128
# Estimates are PyTorch *allocated* peaks; the cap limits *reserved* memory, which the caching
# allocator keeps above allocated (fragmentation). Margins validated by boundary runs.
SAFETY_GB = 0.6
SAFETY_FRACTION = 0.08

# --- DiT -----------------------------------------------------------------------------------
# GiB of the DiT checkpoint as it sits on the GPU (local file size wins when present).
_DIT_WEIGHT_GB = {
    ("3b", "fp16"): 6.3, ("3b", "fp8"): 3.2, ("3b", "q8"): 3.4, ("3b", "q4"): 1.9, ("3b", "int8"): 3.4,
    ("7b", "fp16"): 15.4, ("7b", "fp8"): 8.3, ("7b", "q8"): 8.3, ("7b", "q4"): 4.6, ("7b", "int8"): 8.1,
    ("7b", "w4a8"): 4.45, ("3b", "w4a8"): 1.9,
}
_ARCH = {
    # blocks; PyTorch-allocated GiB per 1k latent tokens; fixed DiT-phase extra GiB on top of the weights
    # (text embeddings, RoPE tables, buffers). The engine runs the MLP and RoPE in token chunks and frees
    # the pre-window QKV (2026-11-01, benchmarks/vram/seedvr2.jsonl `r4_*`, INT8 ConvRot runs): 0.144 (7B) /
    # 0.12 (3B) per 1k tokens. torch.compile'd DiTs run unchunked: the earlier fit (0.176 / 0.137) applies.
    "7b": {"blocks": 36, "gb_per_ktok": 0.144, "gb_per_ktok_compiled": 0.176, "extra_gb": 0.30},
    "3b": {"blocks": 32, "gb_per_ktok": 0.12, "gb_per_ktok_compiled": 0.137, "extra_gb": 0.50},
}
# Activation slope by weight format (2026-11-01, RTX 5090 + RTX 3090 runs, results/WP-SEEDVR2 "W4A8 default
# + 3090 speed"): BF16, W4A8 and INT8 ConvRot all need 0.104-0.105 GB per 1k latent tokens on the 7B and
# 0.117 on the 3B at 720p-1080p (INT8 needed 0.140 / 0.147 until its Linear layers stopped returning float32
# under autocast, 2026-11-01 23:00). 4K batches need about 5 % more per token (W4A8 0.109-0.110 on both GPUs),
# so every value carries that 5 %: 720p-1080p estimates sit 1-5 % above the measured peaks, 4K within 1-2 %.
_GB_PER_KTOK = {
    ("7b", "int8"): 0.110, ("7b", "fp16"): 0.110, ("7b", "w4a8"): 0.111,
    ("7b", "fp8"): 0.110, ("7b", "q8"): 0.110, ("7b", "q4"): 0.110,
    ("3b", "int8"): 0.123, ("3b", "fp16"): 0.123, ("3b", "fp8"): 0.123, ("3b", "q8"): 0.123, ("3b", "q4"): 0.123,
}
# Share of each checkpoint inside the swappable transformer blocks (measured on the files: 7B BF16 0.990,
# 7B INT8 ConvRot 0.981, 7B W4A8 0.965, 3B BF16 0.982; the rest is the embeddings and output head).
_BLOCK_SHARE = {"fp16": 0.99, "int8": 0.98, "w4a8": 0.965}
BLOCK_SHARE = 0.97  # other formats

# --- VAE (shared by every DiT model; ema_vae_fp16) ---------------------------------------------
# Allocated GiB = base + per-megapixel-of-tile slope + the batch's frames, for batches of 17+ frames
# (4+ latent frames fill a temporal slice); shorter batches use about 55-75% of the slope. Refitted
# 2026-11-01 with the fused VAE path on 12 tile sizes 0.26-2.06 MP (RTX 3090 sweep + RTX 5090 runs,
# results/WP-SEEDVR2 "W4A8 default + 3090 speed"): encode 1.75 + 9.0/MP (-3.5..+13 %, most at 512 px tiles),
# decode 1.65 + 15.9/MP (-0.9..+3.4 %); the batch's frames add 0.005 (encode) / 0.001 (decode) GB per frame-MP.
# The earlier 3.85 + 7.1/MP encode undershot untiled 1080p by 7 % and overshot 512-768 px tiles by 10-50 %.
VAE_ENC_BASE_GB = 1.75
VAE_ENC_GB_PER_MP = 9.0
VAE_DEC_BASE_GB = 1.65
VAE_DEC_GB_PER_MP = 15.9
VAE_ENC_FRAME_GB_PER_MP = 0.005  # per frame per megapixel of the batch kept by the phase
VAE_DEC_FRAME_GB_PER_MP = 0.001
VAE_FRAME_GB_PER_MP = VAE_ENC_FRAME_GB_PER_MP  # earlier name
# Single images (one frame): measured separately (2026-11-01, robot 1080p 2.09 MP and face 720p 0.92 MP,
# untiled; results/WP-SEEDVR2): encode 5.86 / 3.61 GB, decode 7.62 / 4.97 GB with the 3D convs, and
# 3.52 / 1.84 and 5.25 / 2.61 GB with the default single-frame 2D path; a 4K image (8.29 MP) took encode
# 14.90 / 12.44 GB and decode 23.76 / 19.31 GB (3D / 2D). The video terms above (fitted on 17+ frame
# batches) overestimated images 2.6-3.8x. The default single-frame 2D path (SEEDVR2_VAE_2D_SINGLE_FRAME,
# on unless set to 0) uses the 2D terms (fitted through all three sizes within 0.05 GB); the 3D terms
# apply when it is switched off.
VAE_IMG_ENC_BASE_GB = 1.85
VAE_IMG_ENC_GB_PER_MP = 1.95
VAE_IMG_DEC_BASE_GB = 2.65
VAE_IMG_DEC_GB_PER_MP = 2.56
VAE_IMG2D_ENC_BASE_GB = 0.52
VAE_IMG2D_ENC_GB_PER_MP = 1.44
VAE_IMG2D_DEC_BASE_GB = 0.53
VAE_IMG2D_DEC_GB_PER_MP = 2.27


def _image_2d_path() -> bool:
    return os.environ.get("SEEDVR2_VAE_2D_SINGLE_FRAME", "1").strip() != "0"


def _model_key(model_name: str, int8: bool) -> Tuple[str, str]:
    name = Path(str(model_name or "")).name.lower()
    size = "3b" if re.search(r"(?:^|[._-])3b(?:[._-]|$)", name) else "7b"
    if name.endswith(".gguf"):
        prec = "q4" if "q4" in name else "q8"
    elif "w4a8" in name:
        prec = "w4a8"  # prequantized; the INT8 ConvRot switch does not apply
    elif int8:
        prec = "int8"
    elif "fp8" in name:
        prec = "fp8"
    else:
        prec = "fp16"
    return size, prec


def dit_weight_gb(model_name: str, int8: bool, model_dir: Optional[str] = None) -> float:
    size, prec = _model_key(model_name, int8)
    gb = _DIT_WEIGHT_GB.get((size, prec), 16.5 if size == "7b" else 6.8)
    try:
        base = Path(model_dir) if model_dir else Path(__file__).resolve().parents[2] / "SeedVR2" / "models"
        name = Path(str(model_name)).name
        cand = base / (name.replace(".safetensors", "_int8_convrot.safetensors") if prec == "int8" else name)
        if cand.is_file():
            gb = cand.stat().st_size / 1024 ** 3
    except Exception:
        pass
    return gb


def max_blocks(model_name: str) -> int:
    size, _ = _model_key(model_name, False)
    try:
        from shared.models.seedvr2_meta import model_meta_map

        meta = model_meta_map().get(Path(str(model_name)).name)
        if meta is not None:
            return min(_ARCH[size]["blocks"], int(getattr(meta, "max_blocks_to_swap", 36) or 36))
    except Exception:
        pass
    return _ARCH[size]["blocks"]


@dataclass
class Geometry:
    width: int    # padded model width (multiple of 16)
    height: int
    out_w: int    # saved width
    out_h: int

    @property
    def mp(self) -> float:
        return self.width * self.height / 1e6

    def tokens_per_latent_frame(self) -> int:
        return (self.width // 16) * (self.height // 16)


def geometry_from_cli(in_w: int, in_h: int, resolution: int, max_resolution: int) -> Geometry:
    """Mirror SeedVR2's resize: short side -> resolution, long side capped at max_resolution, pad to 16."""
    res = max(16, int(resolution or 0))
    if in_w <= in_h:
        w, h = res, int(res * in_h / max(1, in_w))
    else:
        h, w = res, int(res * in_w / max(1, in_h))
    cap = int(max_resolution or 0)
    if cap > 0 and max(w, h) > cap:
        s = cap / float(max(w, h))
        w, h = int(round(w * s)), int(round(h * s))
    pad = lambda v: int(math.ceil(v / 16.0) * 16)  # noqa: E731
    return Geometry(width=pad(w), height=pad(h), out_w=(w // 2) * 2, out_h=(h // 2) * 2)


def latent_frames(frames: int) -> int:
    t = round_up_4n1(max(1, int(frames)))
    return (t - 1) // 4 + 1


def dit_peak_gb(model_name: str, int8: bool, geo: Geometry, frames: int, swap: int,
                model_dir: Optional[str] = None, weights: Optional[float] = None,
                compiled: bool = False) -> float:
    size, prec = _model_key(model_name, int8)
    arch = _ARCH[size]
    weights = dit_weight_gb(model_name, int8, model_dir) if weights is None else weights
    per_block = weights * _BLOCK_SHARE.get(prec, BLOCK_SHARE) / arch["blocks"]
    swap = max(0, min(int(swap), arch["blocks"]))
    resident = weights - swap * per_block + (per_block if swap else 0.0)  # one block in flight
    tokens_k = latent_frames(frames) * geo.tokens_per_latent_frame() / 1000.0
    slope = arch["gb_per_ktok_compiled"] if compiled else _GB_PER_KTOK.get((size, prec), arch["gb_per_ktok"])
    return resident + slope * tokens_k + arch["extra_gb"]


def _tile_mp(geo: Geometry, tiled: bool, tile: int) -> float:
    if not tiled:
        return geo.mp
    t = max(64, int(tile))
    return min(t, geo.width) * min(t, geo.height) / 1e6


def _slice_share(frames: int) -> float:
    """The VAE works on temporal slices of 4 latent frames (16 video frames); short batches use less."""
    lat = latent_frames(frames)
    return 0.75 if lat <= 2 else min(1.0, 0.75 + 0.125 * (lat - 2))


def encode_peak_gb(geo: Geometry, frames: int, tiled: bool, tile: int) -> float:
    if int(frames) <= 1:
        if _image_2d_path():
            return VAE_IMG2D_ENC_BASE_GB + VAE_IMG2D_ENC_GB_PER_MP * _tile_mp(geo, tiled, tile)
        return VAE_IMG_ENC_BASE_GB + VAE_IMG_ENC_GB_PER_MP * _tile_mp(geo, tiled, tile)
    return (VAE_ENC_BASE_GB + VAE_ENC_GB_PER_MP * _tile_mp(geo, tiled, tile) * _slice_share(frames)
            + VAE_ENC_FRAME_GB_PER_MP * geo.mp * round_up_4n1(frames))


def decode_peak_gb(geo: Geometry, frames: int, tiled: bool, tile: int) -> float:
    if int(frames) <= 1:
        if _image_2d_path():
            return VAE_IMG2D_DEC_BASE_GB + VAE_IMG2D_DEC_GB_PER_MP * _tile_mp(geo, tiled, tile)
        return VAE_IMG_DEC_BASE_GB + VAE_IMG_DEC_GB_PER_MP * _tile_mp(geo, tiled, tile)
    return (VAE_DEC_BASE_GB + VAE_DEC_GB_PER_MP * _tile_mp(geo, tiled, tile) * _slice_share(frames)
            + VAE_DEC_FRAME_GB_PER_MP * geo.mp * round_up_4n1(frames))


# --- Phase 4 (post-processing, in the decode process after the decoder) ---------------------------
# Colour correction runs on each whole batch of output frames in float32. Allocated GB per frame-megapixel
# of the batch, measured 2026-11-01 on the RTX 3090 (720p 61f, 1080p 25f, 4K 9f, 720p-4K images): lab
# 0.043 (0.13 per frame-MP while a batch holds at most 16.7 M pixels: exact sort-based matcher), wavelet 0.041,
# adain 0.040, hsv 0.105, wavelet_adaptive 0.14, none 0.006. The decoder's cached blocks are released
# before Phase 4 (generation_phases.py), so the decode process peaks at the larger of the two phases.
POST_GB_PER_FRAME_MP = {"lab": 0.0435, "wavelet": 0.042, "adain": 0.041, "hsv": 0.106, "wavelet_adaptive": 0.142,
                        "none": 0.0065}
POST_LAB_EXACT_GB_PER_FRAME_MP = 0.132
POST_LAB_EXACT_MAX_PIXELS = 16_777_216  # SeedVR2/src/utils/color_fix.py LAB_SORT_MATCH_MAX_ELEMENTS
POST_BASE_GB = 0.05


def post_peak_gb(geo: Geometry, frames: int, method: str = "lab") -> float:
    method = str(method or "lab").strip().lower()
    frames = max(1, int(frames))
    fmp = geo.out_w * geo.out_h * frames / 1e6
    if method == "lab" and geo.out_w * geo.out_h * frames <= POST_LAB_EXACT_MAX_PIXELS:
        return POST_BASE_GB + POST_LAB_EXACT_GB_PER_FRAME_MP * fmp
    return POST_BASE_GB + POST_GB_PER_FRAME_MP.get(method, max(POST_GB_PER_FRAME_MP.values())) * fmp


def _with_margin(gb: float) -> float:
    return gb * (1.0 + SAFETY_FRACTION) + SAFETY_GB


@dataclass
class SeedVR2Choice:
    batch: int
    swap: int
    enc_tiled: bool
    enc_tile: int
    dec_tiled: bool
    dec_tile: int
    dit_gb: float
    enc_gb: float
    dec_gb: float
    post_gb: float = 0.0

    @property
    def peak_gb(self) -> float:
        return max(self.dit_gb, self.enc_gb, self.dec_gb, self.post_gb)


def _vae_choice(fn, geo: Geometry, frames: int, cap: float) -> Optional[Tuple[bool, int, float]]:
    gb = fn(geo, frames, False, 0)
    if _with_margin(gb) <= cap:
        return False, 1024, gb
    for tile in VAE_TILE_CHOICES:
        if tile >= max(geo.width, geo.height):
            continue  # same as untiled
        gb = fn(geo, frames, True, tile)
        if _with_margin(gb) <= cap:
            return True, tile, gb
    return None


def batch_count(frames: int, batch: int, overlap: int = 0) -> int:
    """Batches SeedVR2 runs for `frames` (same stepping as generation_phases: step = batch - overlap)."""
    frames, batch, overlap = int(frames), max(1, int(batch)), max(0, int(overlap))
    step = batch - overlap if 0 < overlap < batch else batch
    count = 0
    for idx in range(0, max(1, frames), step):
        if idx > 0 and min(idx + batch, frames) - idx <= overlap:
            break
        count += 1
    return count


def balanced_batch(frames: int, batch: int, overlap: int = 0) -> int:
    """Smallest 4n+1 batch with as few batches as `batch`: 72 frames at batch 65 run 65 + 7 frames, at
    batch 37 they run 37 + 35, with the same single seam and no short, flicker-prone last batch."""
    count = batch_count(frames, batch, overlap)
    best = int(batch)
    while best - 4 >= 5 and best - 4 > overlap and batch_count(frames, best - 4, overlap) == count:
        best -= 4
    return best


def choose(model_name: str, int8: bool, geo: Geometry, frames_wanted: int, cap_gb: float,
           model_dir: Optional[str] = None, image: bool = False,
           overlap: int = 0, compiled: bool = False,
           color_correction: str = "lab") -> Tuple[Optional[SeedVR2Choice], List[str]]:
    """Fewest batches (largest batch that fits), split evenly, then the least BlockSwap, then the least
    VAE tiling."""
    notes: List[str] = []
    blocks = max_blocks(model_name)
    weights = dit_weight_gb(model_name, int8, model_dir)  # file stat once, not per candidate

    def fit(batch: int) -> Optional[SeedVR2Choice]:
        for swap in range(0, blocks + 1):
            dit = dit_peak_gb(model_name, int8, geo, batch, swap, model_dir, weights, compiled=compiled)
            if _with_margin(dit) > cap_gb:
                continue
            enc = _vae_choice(encode_peak_gb, geo, batch, cap_gb)
            dec = _vae_choice(decode_peak_gb, geo, batch, cap_gb)
            post = post_peak_gb(geo, batch, color_correction)
            if enc is None or dec is None or _with_margin(post) > cap_gb:
                return None  # VAE / post-processing cannot fit this batch; try a smaller batch
            return SeedVR2Choice(batch=batch, swap=swap, enc_tiled=enc[0], enc_tile=enc[1], dec_tiled=dec[0],
                                 dec_tile=dec[1], dit_gb=dit, enc_gb=enc[2], dec_gb=dec[2], post_gb=post)
        return None

    top = 1 if image else min(MAX_BATCH, max(5, round_up_4n1(frames_wanted)))
    batch = top
    while batch >= (1 if image else 5):
        choice = fit(batch)
        if choice is not None:
            even = batch if image else balanced_batch(frames_wanted, batch, overlap)
            if even < batch:
                choice = fit(even) or choice
            return choice, notes
        if image:
            break
        nxt = round_down_4n1(batch - 4, 5) if batch > 5 else 0
        batch = nxt if nxt < batch else batch - 4
    return None, notes


def plan(settings: Dict[str, Any], *, in_w: int, in_h: int, frames: int, budget: Optional[VramBudget],
         image: bool = False, cap_scale: float = 1.0) -> TunePlan:
    out = TunePlan(budget=budget)
    if budget is None:
        out.warnings.append("No NVIDIA GPU selected or detected; settings left unchanged.")
        return out
    model = str(settings.get("dit_model") or "")
    int8 = bool(settings.get("int8_convrot")) and not model.lower().endswith(".gguf") and "w4a8" not in model.lower()
    fallback_reason = None
    if "w4a8" in Path(model).name.lower():
        # W4A8 needs compute capability 8.0+ and comfy_kitchen's CUDA backend; otherwise the run
        # switches to 7B Sharp INT8 ConvRot (shared.models.seedvr2_w4a8), so plan for that model.
        from shared.models.seedvr2_w4a8 import effective_model

        model, int8, fallback_reason = effective_model(model, False, str(budget.gpu.index))
    geo = geometry_from_cli(in_w, in_h, int(settings.get("resolution") or 1080), int(settings.get("max_resolution") or 0))
    cap = (budget.limit_gb - SEEDVR2_PROCESS_OVERHEAD_GB) * max(0.3, min(1.0, float(cap_scale)))
    overlap = max(0, int(float(settings.get("temporal_overlap") or 0)))
    compiled = bool(settings.get("compile_dit"))
    choice, notes = choose(model, int8, geo, frames, cap, settings.get("model_dir"), image=image, overlap=overlap,
                           compiled=compiled, color_correction=str(settings.get("color_correction") or "lab"))
    out.lines.append(f"Output {geo.out_w}x{geo.out_h} (model works at {geo.width}x{geo.height}), "
                     + ("single image" if image else f"{frames} frame(s) per run") + f", model {Path(model).name}"
                     + (" + INT8 ConvRot" if int8 else ""))
    if fallback_reason:
        out.lines.append(f"W4A8 cannot run here ({fallback_reason}): planned for the 7B Sharp INT8 ConvRot "
                         "model the run switches to")
    out.lines.extend(notes)
    if choice is None:
        out.fits = False
        blocks = max_blocks(model)
        smallest = dit_peak_gb(model, int8, geo, 1 if image else 5, blocks, settings.get("model_dir"))
        out.warnings.append(
            f"Even batch {1 if image else 5} with all {blocks} blocks swapped needs about {_with_margin(smallest):.1f} GB "
            f"but only {max(0.0, cap):.1f} GB is available. The run still tries these smallest settings (the estimate "
            "is cautious: 1080p on an 8 GB card worked at 4.2 GB) and stops with an out-of-memory message, never "
            "shared GPU memory, if they do not fit. Otherwise lower the output size, pick a smaller/quantized model, "
            "close other GPU apps or lower Save VRAM.")
        out.updates.update({"blocks_to_swap": blocks, "vae_encode_tiled": True, "vae_decode_tiled": True,
                            "vae_encode_tile_size": 256, "vae_decode_tile_size": 256})
        if not image:
            out.updates["batch_size"] = 5
        return out
    upd: Dict[str, Any] = {
        "blocks_to_swap": int(choice.swap),
        "swap_io_components": False,
        "vae_encode_tiled": bool(choice.enc_tiled),
        "vae_encode_tile_size": int(choice.enc_tile),
        "vae_encode_tile_overlap": VAE_TILE_OVERLAP if choice.enc_tiled else int(settings.get("vae_encode_tile_overlap") or VAE_TILE_OVERLAP),
        "vae_decode_tiled": bool(choice.dec_tiled),
        "vae_decode_tile_size": int(choice.dec_tile),
        "vae_decode_tile_overlap": VAE_TILE_OVERLAP if choice.dec_tiled else int(settings.get("vae_decode_tile_overlap") or VAE_TILE_OVERLAP),
    }
    if not image:
        upd["batch_size"] = int(choice.batch)  # estimates assume full batches, so Uniform Batch Size is left as is
    out.updates = upd
    out.estimate_gb = _with_margin(choice.peak_gb) + SEEDVR2_PROCESS_OVERHEAD_GB
    whole = (not image) and choice.batch >= round_up_4n1(frames)
    if not image:
        out.lines.append(f"Batch Size {choice.batch}" + (" (whole clip in one batch)" if whole else
                         f" ({batch_count(frames, choice.batch, overlap)} even batches, the fewest that fit)"))
    out.lines.append(f"BlockSwap {choice.swap}" + (" (all DiT blocks stay on the GPU)" if choice.swap == 0 else " (needed to fit)"))
    out.lines.append("VAE encode " + (f"tiled {choice.enc_tile}px" if choice.enc_tiled else "untiled")
                     + ", VAE decode " + (f"tiled {choice.dec_tile}px" if choice.dec_tiled else "untiled"))
    out.lines.append(f"Phase peaks: encode {choice.enc_gb:.1f} GB, DiT {choice.dit_gb:.1f} GB, decode {choice.dec_gb:.1f} GB, "
                     f"post-processing {choice.post_gb:.1f} GB")
    out.lines.append(f"The run is capped at the estimate below (PyTorch would otherwise keep freed memory cached up to "
                     f"the {budget.limit_gb:.1f} GB limit); if it ever runs out there, it retries once with the whole "
                     f"limit. Never shared GPU memory.")
    return out


def auto_apply(settings: Dict[str, Any], input_path: str, log: Optional[Callable[[str], None]] = None,
               preview_only: bool = False, cap_scale: float = 1.0) -> Optional[TunePlan]:
    """Runner hook: tune `settings` in place for this exact input and the VRAM free right now."""
    media = probe_media(input_path)
    if media is None:
        if log:
            log("[SeedVR2 Auto Tune] Could not read the input size; keeping the current settings.")
        return None
    budget = resolve_budget(settings.get("cuda_device") or os.environ.get("SECOURSES_GLOBAL_GPU_DEVICE"),
                            settings.get("save_vram_gb"), overhead_gb=SEEDVR2_PROCESS_OVERHEAD_GB)
    frames = media.frames
    try:
        cap = int(settings.get("load_cap") or 0)
        skip = int(settings.get("skip_first_frames") or 0)
    except Exception:
        cap, skip = 0, 0
    frames = max(1, frames - max(0, skip))
    if cap > 0:
        frames = min(frames, cap)
    frames = frames_per_unit(type(media)(media.path, media.kind, media.width, media.height, frames, media.fps),
                             None, int(settings.get("chunk_size") or 0))
    frames += max(0, int(settings.get("prepend_frames") or 0))
    image = preview_only or media.kind == "image" or (media.kind == "directory" and frames == 1)
    tp = plan(settings, in_w=media.width, in_h=media.height, frames=frames, budget=budget, image=image,
              cap_scale=cap_scale)
    settings.update(tp.updates)
    if cap_scale >= 1.0:
        mark_tightness(settings, tp)
    # The run's allocator is capped at this estimate (shared.runner; SECOURSES_AUTOTUNE_CAP=0 restores the whole
    # budget), so VRAM in use stays at the estimate instead of filling the budget with cached blocks (RTX 5090,
    # 720p 61 frames: 22.1 GB in use with the whole budget, 19.4-20.0 GB capped, same speed); the cap adds
    # max_split_size_mb (an OOM at the cap without it), and an OOM there retries once with the whole budget.
    settings["_vram_cap_gb"] = round(tp.estimate_gb, 2) if (tp.fits and tp.estimate_gb > 0) else None
    if log:
        for line in tp.report("[SeedVR2 Auto Tune]").splitlines():
            log(line)
    return tp
