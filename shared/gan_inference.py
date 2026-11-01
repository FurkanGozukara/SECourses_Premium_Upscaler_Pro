"""Spandrel inference for the GAN tab: model cache, half precision and VRAM-aware tiling.

Both the image and the video paths of ``gan_runner`` call ``upscale_tensor``.

Precision ("gan_precision" setting):
  auto  FP16 on CUDA when the model's spandrel descriptor supports it and a one-time probe
        (the first input, area-downscaled to PROBE_SIZE px on its short side, FP16 vs FP32)
        reaches PROBE_MIN_PSNR; FP32 otherwise. A centre crop missed a RealPLKSR-LayerNorm model
        whose full-image FP16 output was 40 dB from FP32 (crop: 52 dB; downscaled whole: 43 dB).
  fp16 / bf16  forced (still FP32 when the descriptor does not support the type).
  fp32  always FP32 (the behavior before this module).
A non-finite output in FP16/BF16 switches that model to FP32 for the rest of the process.

Tiling ("tile_size" / "overlap" settings, input pixels):
  tile_size 0  VRAM-aware: a small probe forward measures the model's activation memory per
               input pixel; the whole image runs when that estimate fits in VRAM_BUDGET of the free
               VRAM, otherwise the largest square tile that fits. (Windows drivers may page CUDA
               memory into system RAM instead of failing, which is far slower than tiling.) A CUDA
               out-of-memory error still halves the tile size. The choice is remembered per model
               and input size, so video frames reuse it.
  tile_size N  fixed N x N tiles. Tiles overlap by "overlap" pixels and are blended with
               linear ramps inside the overlap; image borders keep full weight.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

PRECISION_CHOICES = ("auto", "fp16", "bf16", "fp32")
PROBE_MIN_PSNR = 50.0
PROBE_SIZE = 256
MIN_AUTO_TILE = 64
VRAM_PROBE_SIZE = 192
# Share of free VRAM the measured activations may use. PyTorch's caching allocator reserved
# up to 2.1x the allocated peak on DAT2 tiles (16.6 GB allocated, 33.4 GB reserved on a 32 GB
# card, which Windows then pages to system RAM), so the estimate keeps allocations under 40 %.
VRAM_BUDGET = 0.40

# Per model file: result of the FP16/BF16 probe and measured bytes per input pixel, kept for the
# app session so batch runs (one run_gan_upscale call per file) do not repeat the probes.
_PROBE_RESULTS: Dict[Tuple[str, str], Tuple[bool, str]] = {}
_BYTES_PER_PIXEL: Dict[Tuple[str, str, int], float] = {}

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


@dataclass
class LoadedGanModel:
    descriptor: Any
    path: str
    device: torch.device
    precision: str  # requested ("auto", "fp16", "bf16", "fp32")
    dtype: torch.dtype = torch.float32
    probed: bool = False
    notes: List[str] = field(default_factory=list)
    # (n, h, w) -> tile size that fit (0 = whole image)
    tile_memory: Dict[Tuple[int, int, int], int] = field(default_factory=dict)
    # Exact FP32 weights (CPU), so switching back from FP16/BF16 never keeps rounded weights.
    fp32_state: Optional[Dict[str, torch.Tensor]] = None

    @property
    def scale(self) -> int:
        return int(getattr(self.descriptor, "scale", 1) or 1)

    @property
    def dtype_name(self) -> str:
        return {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(self.dtype, "fp32")


_CACHE: Dict[Tuple[str, int, int, str, str], LoadedGanModel] = {}
_LOCK = threading.RLock()


def normalize_precision(value: Any) -> str:
    token = str(value or "auto").strip().lower()
    aliases = {"half": "fp16", "float16": "fp16", "bfloat16": "bf16", "float32": "fp32", "full": "fp32", "float": "fp32"}
    token = aliases.get(token, token)
    return token if token in PRECISION_CHOICES else "auto"


def resolve_device(cuda_id: str) -> torch.device:
    """cuda_id: "<n>" for a CUDA device, "" for CPU (same contract as gan_runner._resolve_cuda_device_id)."""
    if not torch.cuda.is_available() or str(cuda_id) == "":
        return torch.device("cpu")
    try:
        did = int(cuda_id)
        if 0 <= did < torch.cuda.device_count():
            return torch.device(f"cuda:{did}")
    except Exception:
        pass
    return torch.device("cuda")


def _cast(entry: LoadedGanModel, dtype: torch.dtype) -> None:
    if entry.dtype != torch.float32 and entry.fp32_state is not None:
        entry.descriptor.model.to(dtype=torch.float32)
        entry.descriptor.model.load_state_dict(entry.fp32_state, strict=False)
    entry.dtype = torch.float32
    if dtype != torch.float32:
        entry.descriptor.to(dtype=dtype)  # raises UnsupportedDtypeError when not supported
        entry.dtype = dtype


def load_gan_model(model_path: Path, device: torch.device, precision: str = "auto") -> LoadedGanModel:
    """Load (or reuse) a spandrel model on ``device``. One model is cached at a time."""
    import spandrel

    precision = normalize_precision(precision)
    path = Path(model_path)
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns, str(device), precision)
    with _LOCK:
        entry = _CACHE.get(key)
        if entry is not None:
            return entry
        release_gan_models()
        descriptor = spandrel.ModelLoader().load_from_file(str(path))
        descriptor.eval()
        descriptor.to(device)
        entry = LoadedGanModel(descriptor=descriptor, path=str(path), device=device, precision=precision)
        target = _initial_dtype(entry)
        if target != torch.float32:
            entry.fp32_state = {k: v.detach().to("cpu", copy=True) for k, v in descriptor.model.state_dict().items()}
            try:
                _cast(entry, target)
            except Exception as exc:  # UnsupportedDtypeError or a cast failure
                entry.notes.append(f"{_name(target)} unavailable ({exc}); using fp32")
                _cast(entry, torch.float32)
        _CACHE[key] = entry
        return entry


def _name(dtype: torch.dtype) -> str:
    return {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(dtype, "fp32")


def _initial_dtype(entry: LoadedGanModel) -> torch.dtype:
    d = entry.descriptor
    if entry.device.type != "cuda" or entry.precision == "fp32":
        return torch.float32
    if entry.precision == "fp16":
        return torch.float16 if d.supports_half else torch.float32
    if entry.precision == "bf16":
        ok = d.supports_bfloat16 and torch.cuda.is_bf16_supported()
        return torch.bfloat16 if ok else torch.float32
    # auto
    return torch.float16 if d.supports_half else torch.float32


def release_gan_models() -> None:
    with _LOCK:
        for entry in _CACHE.values():
            try:
                entry.descriptor.to("cpu")
            except Exception:
                pass
        _CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _forward(entry: LoadedGanModel, x: torch.Tensor) -> torch.Tensor:
    """x: float32 (N, C, h, w) on any device -> float32 output on the CPU."""
    xin = x.to(entry.device, dtype=entry.dtype, non_blocking=True)
    out = entry.descriptor(xin)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out.float().cpu()


def _model_key(entry: LoadedGanModel) -> str:
    try:
        st = Path(entry.path).stat()
        return f"{Path(entry.path).resolve()}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        return entry.path


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(torch.mean((a.clamp(0, 1) - b.clamp(0, 1)) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def _probe(entry: LoadedGanModel, x: torch.Tensor, log: Callable[[str], None]) -> None:
    """Auto precision: compare a centre crop in the reduced type with FP32 once per model."""
    entry.probed = True
    if entry.dtype == torch.float32 or entry.precision != "auto":
        return
    cache_key = (_model_key(entry), entry.dtype_name)
    if cache_key in _PROBE_RESULTS:
        ok, note = _PROBE_RESULTS[cache_key]
        entry.notes.append(note + " (cached)")
        if not ok:
            _cast(entry, torch.float32)
        return
    _, _, h, w = x.shape
    factor = min(1.0, PROBE_SIZE / float(min(h, w)))
    crop = x[:1]
    if factor < 1.0:
        size = (max(16, int(round(h * factor))), max(16, int(round(w * factor))))
        crop = torch.nn.functional.interpolate(crop, size=size, mode="area")
    reduced = _forward(entry, crop)
    low = entry.dtype
    _cast(entry, torch.float32)
    full = _forward(entry, crop)
    finite = bool(torch.isfinite(reduced).all())
    score = _psnr(reduced, full) if finite else 0.0
    if finite and score >= PROBE_MIN_PSNR:
        _cast(entry, low)
        note = f"{_name(low)} probe: {score:.1f} dB vs fp32"
        _PROBE_RESULTS[cache_key] = (True, note)
    else:
        note = f"{_name(low)} probe failed ({'non-finite' if not finite else f'{score:.1f} dB'}); using fp32"
        _PROBE_RESULTS[cache_key] = (False, note)
        log(f"GAN precision: {_name(low)} rejected for this model ({score:.1f} dB vs fp32), using fp32")
    entry.notes.append(note)


def _auto_tile_for_vram(entry: LoadedGanModel, x: torch.Tensor, log: Callable[[str], None]) -> int:
    """0 (whole image) when the measured per-pixel activation memory fits the free VRAM, else a tile size."""
    if entry.device.type != "cuda":
        return 0
    n, _, h, w = x.shape
    p = min(VRAM_PROBE_SIZE, h, w)
    if p * p >= h * w:
        return 0
    dev = entry.device
    bpp_key = (_model_key(entry), entry.dtype_name, n)
    per_px = _BYTES_PER_PIXEL.get(bpp_key)
    if per_px is None:
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        _forward(entry, x[:, :, :p, :p])
        torch.cuda.synchronize(dev)
        per_px = max(1.0, (torch.cuda.max_memory_allocated(dev) - base) / float(p * p))
        _BYTES_PER_PIXEL[bpp_key] = per_px
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info(dev)
    budget = free * VRAM_BUDGET
    need = per_px * h * w
    if need <= budget:
        return 0
    tile = int(math.sqrt(budget / per_px)) // 32 * 32
    tile = max(MIN_AUTO_TILE, tile)
    log(f"GAN auto tiling: whole {w}x{h} needs about {need / 2**30:.1f} GB of {free / 2**30:.1f} GB free VRAM; "
        f"using {tile}px tiles")
    return 0 if tile >= max(h, w) else tile


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cudnn_status_not_enough_workspace" in text or "cudnn_status_alloc_failed" in text


def _ramp(length: int, ov_start: int, ov_end: int) -> np.ndarray:
    """Weights along one tile axis: linear ramps over the shared overlaps (never 0), 1 elsewhere."""
    w = np.ones(length, dtype=np.float32)
    for ov, at_start in ((ov_start, True), (ov_end, False)):
        ov = min(int(ov), length)
        if ov <= 0:
            continue
        r = (np.arange(ov, dtype=np.float32) + 0.5) / ov
        if at_start:
            w[:ov] *= r
        else:
            w[length - ov:] *= r[::-1]
    return w


def _positions(size: int, tile: int, overlap: int) -> List[int]:
    if size <= tile:
        return [0]
    step = max(1, tile - overlap)
    count = int(math.ceil((size - tile) / step)) + 1
    return [int(round(i * (size - tile) / (count - 1))) for i in range(count)]


def _run_tiled(entry: LoadedGanModel, x: torch.Tensor, tile: int, overlap: int,
               cancel_event=None) -> torch.Tensor:
    n, c, h, w = x.shape
    s = entry.scale
    overlap = max(0, min(int(overlap), tile // 2))
    ys, xs = _positions(h, tile, overlap), _positions(w, tile, overlap)
    out: Optional[torch.Tensor] = None
    weight = torch.zeros((1, 1, h * s, w * s), dtype=torch.float32)
    for yi, y0 in enumerate(ys):
        th = min(tile, h - y0)
        for xi, x0 in enumerate(xs):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Canceled")
            tw = min(tile, w - x0)
            part = _forward(entry, x[:, :, y0:y0 + th, x0:x0 + tw])
            if entry.device.type == "cuda" and len(ys) * len(xs) > 1:
                torch.cuda.empty_cache()  # tiles of a large image: hand cached blocks back between tiles
            if out is None:
                out = torch.zeros((n, part.shape[1], h * s, w * s), dtype=torch.float32)
            # Overlap actually shared with the neighbouring tile (positions are evenly spread).
            ov_top = (ys[yi - 1] + tile - y0) if yi > 0 else 0
            ov_bot = (y0 + th - ys[yi + 1]) if yi + 1 < len(ys) else 0
            ov_left = (xs[xi - 1] + tile - x0) if xi > 0 else 0
            ov_right = (x0 + tw - xs[xi + 1]) if xi + 1 < len(xs) else 0
            wy = _ramp(th * s, ov_top * s, ov_bot * s)
            wx = _ramp(tw * s, ov_left * s, ov_right * s)
            wmap = torch.from_numpy(np.outer(wy, wx))[None, None]
            ys0, xs0 = y0 * s, x0 * s
            out[:, :, ys0:ys0 + th * s, xs0:xs0 + tw * s] += part * wmap
            weight[:, :, ys0:ys0 + th * s, xs0:xs0 + tw * s] += wmap
    assert out is not None
    return out / weight.clamp_min(1e-8)


def upscale_tensor(
    entry: LoadedGanModel,
    x: torch.Tensor,
    tile_size: int = 0,
    overlap: int = 32,
    log: Optional[Callable[[str], None]] = None,
    cancel_event=None,
) -> torch.Tensor:
    """x: float32 (N, C, H, W) in [0, 1] on the CPU -> float32 (N, C', H*s, W*s) on the CPU."""
    log = log or (lambda _m: None)
    if not entry.probed:
        _probe(entry, x, log)
    n, _, h, w = x.shape
    try:
        tile_size = int(tile_size or 0)
    except Exception:
        tile_size = 0
    try:
        overlap = max(0, int(overlap if overlap is not None else 32))
    except Exception:
        overlap = 32

    def run(tile: int) -> torch.Tensor:
        if tile <= 0 or (h <= tile and w <= tile):
            return _forward(entry, x)
        return _run_tiled(entry, x, tile, overlap, cancel_event)

    def run_checked(tile: int) -> torch.Tensor:
        out = run(tile)
        if entry.dtype != torch.float32 and not bool(torch.isfinite(out).all()):
            log(f"GAN precision: {entry.dtype_name} produced non-finite values; switching this model to fp32")
            entry.notes.append(f"{entry.dtype_name} non-finite output; switched to fp32")
            _cast(entry, torch.float32)
            out = run(tile)
        return out

    if tile_size > 0:
        return run_checked(tile_size)

    key = (n, h, w)
    if key in entry.tile_memory:
        tile = entry.tile_memory[key]
    else:
        try:
            tile = _auto_tile_for_vram(entry, x, log)
        except Exception as exc:
            if not _is_oom(exc):
                raise
            torch.cuda.empty_cache()
            tile = max(MIN_AUTO_TILE, VRAM_PROBE_SIZE // 2)
    while True:
        oom = False
        try:
            out = run_checked(tile)
            if entry.tile_memory.get(key) != tile:
                entry.tile_memory[key] = tile
                if tile:
                    log(f"GAN auto tiling: {tile}px tiles (overlap {overlap}px) for {w}x{h}")
            return out
        except Exception as exc:
            current = tile or max(h, w)
            nxt = 1 << max(0, int(math.ceil(math.log2(current))) - 1)
            if nxt >= current:
                nxt = current // 2
            if entry.device.type != "cuda" or not _is_oom(exc) or nxt < MIN_AUTO_TILE:
                raise
            oom = True
        if oom:
            # Outside the except block, so the failed attempt's frames (and tensors) are gone.
            torch.cuda.empty_cache()
            log(f"GAN auto tiling: out of VRAM at {'whole image' if not tile else f'{tile}px tiles'}; trying {nxt}px tiles")
            tile = nxt


def to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) float in [0, 1] -> (H, W, C) uint8, rounded (not truncated)."""
    arr = t.clamp(0, 1).mul(255.0).round_().to(torch.uint8)
    return arr.permute(1, 2, 0).contiguous().numpy()
