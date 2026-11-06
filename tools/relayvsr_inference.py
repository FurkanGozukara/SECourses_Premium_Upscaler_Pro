"""RelayVSR 4x video / image super-resolution (https://github.com/kopperx/RelayVSR) for SECourses Premium Upscaler Pro.

A large Wan model writes sparse reference latents (one keyframe every window_size - 1 frames, one step, KV cache of
recent keyframes) and a small conditional Flash decoder turns every low-resolution frame into a 4x frame, streaming
one frame at a time. The vendored inference package lives in RelayVSR/refvsr_inference.

App integration: the shared ffmpeg writer (codec / CRF / preset / pixel format, source audio with the frame-range
offset), image input -> PNG, Start / End Frame, FRAME_PROGRESS lines, the INT8 ConvRot Wan file, and VRAM placement:
Wan blocks that do not fit the budget (SECOURSES_VRAM_LIMIT_GB from the runner, or the free VRAM) stay in pinned RAM
and are copied to the GPU right before they run. Wan runs once per window, so streamed blocks cost little speed.

    python tools/relayvsr_inference.py --input in.mp4 --output out.mp4 --wan W.safetensors --flash F.safetensors
    python tools/relayvsr_inference.py --convert_int8 out_int8.safetensors --wan wan.safetensors   (one-time build)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
for _p in (APP_DIR / "RelayVSR", APP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from shared import video_io  # noqa: E402
from shared.int8_convrot_modules import is_int8_file, load_int8_state, quantize_linears, save_int8_model  # noqa: E402
from shared.vram_autotune import relayvsr as tune  # noqa: E402

GIB = 1024 ** 3
INT8_FORMAT = "relayvsr-wan-int8-convrot-v1"
INT8_MARKER = "relayvsr_int8_convrot"
SCALE = 4


def emit(msg: str) -> None:
    print(f"[RelayVSR] {msg}", flush=True)


# ------------------------------------------------------------------ models

def _wan_header(path: Path):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        prompt_shape = tuple(handle.get_slice("prompt_embeds").get_shape())
    if metadata.get("format") != "refvsr-wan-v1" and metadata.get("relayvsr_format") != "refvsr-wan-v1":
        raise ValueError(f"{path.name} is not a RelayVSR Wan checkpoint")
    return json.loads(metadata["config"]), prompt_shape, metadata


def build_wan(path: Path, dtype: torch.dtype) -> nn.Module:
    """The Wan model from the bf16 release file or from the INT8 ConvRot file (built on meta, weights assigned)."""
    from accelerate import init_empty_weights
    from refvsr_inference.models.wan.transformer import WanTransformer3DModel

    config, prompt_shape, _ = _wan_header(path)
    with init_empty_weights():
        wan = WanTransformer3DModel.from_config(config)
        wan.lq_proj = wan.build_lq_proj()
        wan.register_buffer("prompt_embeds", torch.empty(prompt_shape))
    state = load_file(str(path), device="cpu")
    if is_int8_file(path, INT8_MARKER):
        n = load_int8_state(wan, state, dtype, log_prefix="[RelayVSR INT8]")
        emit(f"Wan: {n} INT8 ConvRot layers ({path.name})")
    else:
        wan.load_state_dict(state, strict=True, assign=True)
        emit(f"Wan: {path.name} ({next(wan.parameters()).dtype})")
    del state
    for module in wan.modules():  # float weights in the compute dtype; INT8 layers keep int8 + float32 scales
        for key, param in list(module._parameters.items()):
            if param is not None and param.is_floating_point() and param.dtype != dtype:
                module._parameters[key] = nn.Parameter(param.data.to(dtype), requires_grad=False)
        for key, buf in list(module._buffers.items()):
            if buf is not None and buf.is_floating_point() and buf.dtype != dtype and key not in (
                    "scale_weight", "int8_ara_down", "int8_ara_up"):
                module._buffers[key] = buf.to(dtype)
    return wan.eval().requires_grad_(False)


def build_flash(path: Path, dtype: torch.dtype, device: torch.device) -> nn.Module:
    from accelerate import init_empty_weights
    from refvsr_inference.models.flash.model import FlashVSRConditionNet

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("format") != "refvsr-flash-v1":
        raise ValueError(f"{path.name} is not a RelayVSR Flash checkpoint")
    config = json.loads(metadata["config"])
    variant = config.pop("variant", "S")
    with init_empty_weights():
        flash = FlashVSRConditionNet.from_variant(variant, **config)
    flash.load_state_dict(load_file(str(path), device="cpu"), strict=True, assign=True)
    return flash.to(device=device, dtype=dtype).eval().requires_grad_(False)


def module_bytes(module: nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in list(module.parameters()) + list(module.buffers()))


class _BlockSwap:
    """Keeps a block's tensors in pinned RAM and copies them to the GPU only while the block (or one of its layers,
    called directly, e.g. the one-time cross-attention KV precompute) runs; one swapped block is on the GPU at a time."""

    def __init__(self, block: nn.Module, device: torch.device):
        self.device = device
        self.slots = []
        self.loaded = False
        self.in_block = False
        holders = []
        for module in block.modules():
            own = False
            for key, t in module._parameters.items():
                if t is not None:
                    self.slots.append((module, "_parameters", key, t.data.to("cpu").pin_memory()))
                    own = True
            for key, t in module._buffers.items():
                if t is not None:
                    self.slots.append((module, "_buffers", key, t.to("cpu").pin_memory()))
                    own = True
            if own and module is not block:
                holders.append(module)
        self._park()
        self._kv_pinned = None
        block._relay_swapped = True
        block.register_forward_pre_hook(self._block_pre, with_kwargs=True)
        block.register_forward_hook(self._block_post, with_kwargs=True)
        for module in holders:  # direct calls of a layer outside the block's forward
            module.register_forward_pre_hook(self._layer_pre)
            module.register_forward_hook(self._layer_post)

    @staticmethod
    def _kv_cache(args, kwargs):
        from refvsr_inference.models.wan.transformer import WanSelfAttentionKVCache

        for value in list(args) + list(kwargs.values()):
            if isinstance(value, WanSelfAttentionKVCache):
                return value
        return None

    @staticmethod
    def _point(cache, storage):
        valid = cache.key_value[0].shape[1] if cache.key_value is not None else 0
        cache._storage = storage
        if cache.key_value is not None:
            cache.key_value = (storage[0][:, :valid], storage[1][:, :valid])

    def _set(self, module, kind, key, value):
        if kind == "_parameters":
            module._parameters[key] = nn.Parameter(value, requires_grad=False)
        else:
            module._buffers[key] = value

    def _load(self):
        if not self.loaded:
            for module, kind, key, cpu in self.slots:
                self._set(module, kind, key, cpu.to(self.device, non_blocking=True))
            self.loaded = True

    def _park(self):
        for module, kind, key, cpu in self.slots:
            self._set(module, kind, key, cpu)
        self.loaded = False

    def _block_pre(self, _m, args, kwargs):
        self.in_block = True
        self._load()
        cache = self._kv_cache(args, kwargs)  # the block's KV cache lives in pinned RAM between its calls
        if cache is not None and cache._storage is not None and cache._storage[0].device.type == "cpu":
            self._point(cache, tuple(t.to(self.device, non_blocking=True) for t in cache._storage))

    def _block_post(self, _m, args, kwargs, _o):
        self.in_block = False
        self._park()
        cache = self._kv_cache(args, kwargs)
        if cache is not None and cache._storage is not None and cache._storage[0].device.type == "cuda":
            fresh = self._kv_pinned is None or self._kv_pinned[0].shape != cache._storage[0].shape
            if fresh:
                self._kv_pinned = tuple(torch.empty(t.shape, dtype=t.dtype, pin_memory=True) for t in cache._storage)
                for dst, src in zip(self._kv_pinned, cache._storage):
                    dst.copy_(src, non_blocking=True)
            else:
                # one Wan call writes one keyframe's slice of the ring: copy only that slice back to RAM
                tokens = int(cache.tokens_per_frame or 0)
                start = ((cache.next_frame - 1) % cache.window_size) * tokens
                for dst, src in zip(self._kv_pinned, cache._storage):
                    dst[:, start:start + tokens].copy_(src[:, start:start + tokens], non_blocking=True)
            self._point(cache, self._kv_pinned)

    def _layer_pre(self, _m, _a):
        self._load()

    def _layer_post(self, _m, _a, _o):
        if not self.in_block:
            self._park()


def place_wan(wan: nn.Module, device: torch.device, resident: int) -> int:
    """Everything but the transformer blocks on the GPU, then the first `resident` blocks; the other blocks stay in
    pinned RAM with their KV cache and are copied to the GPU while they run."""
    blocks = list(wan.blocks)
    for name, child in wan.named_children():
        if name != "blocks":
            child.to(device)
    for key, buf in list(wan._buffers.items()):
        if buf is not None:
            wan._buffers[key] = buf.to(device)
    for key, param in list(wan._parameters.items()):
        if param is not None:
            wan._parameters[key] = nn.Parameter(param.data.to(device), requires_grad=False)
    resident = max(0, min(len(blocks), int(resident)))
    for block in blocks[:resident]:
        block.to(device)
    for block in blocks[resident:]:
        _BlockSwap(block, device)
    if resident < len(blocks):
        emit(f"VRAM placement: {resident}/{len(blocks)} Wan blocks on the GPU; {len(blocks) - resident} stay in RAM with "
             f"their KV cache and are copied to the GPU when Wan runs (once per window)")
    else:
        emit(f"VRAM placement: all {len(blocks)} Wan blocks and their KV cache on the GPU")
    return resident


# ------------------------------------------------------------------ input / output

class Source:
    def __init__(self, path: str, start: int, end: int, fps_override: float):
        self.path = path
        self.is_image = video_io.is_image(path)
        if self.is_image:
            img = video_io.read_image(path)
            self.height, self.width = img.shape[:2]
            self.fps, self.total, self.first, self.has_audio = 1.0, 1, 0, False
            self._image = img
            return
        # A folder is an image sequence (sorted by name); otherwise a video file.
        self.sequence = sorted(f for f in Path(path).iterdir() if video_io.is_image(str(f))) if Path(path).is_dir() else None
        if self.sequence is not None:
            if not self.sequence:
                raise SystemExit(f"No images in folder: {path}")
            h, w = video_io.read_image(str(self.sequence[0])).shape[:2]
            frames, fps, self.has_audio = len(self.sequence), 30.0, False
        else:
            info = video_io.probe_video(path)
            w, h, frames, fps, self.has_audio = info.width, info.height, info.frames, float(info.fps), info.has_audio
        self.width, self.height = w, h
        self.fps = float(fps_override) if fps_override and fps_override > 0 else fps
        last = frames - 1 if end < 0 else min(frames - 1, end)
        self.first = max(0, min(start, last))
        self.total = last - self.first + 1

    def frames(self):
        if self.is_image:
            yield self._image
            return
        if self.sequence is not None:
            for f in self.sequence[self.first:self.first + self.total]:
                yield video_io.read_image(str(f))
            return
        for i, frame in enumerate(video_io.iter_video_frames(self.path)):
            if i < self.first:
                continue
            if i >= self.first + self.total:
                break
            yield frame


# ------------------------------------------------------------------ run

def _budget_bytes(device: torch.device, save_gb: float) -> int:
    torch.empty(1, device=device)
    free, total = torch.cuda.mem_get_info(device)
    limit_gb = float(os.environ.get("SECOURSES_VRAM_LIMIT_GB") or 0.0)
    if limit_gb > 0:
        return int(min(limit_gb * GIB, free + torch.cuda.memory_reserved(device)))
    return int(max(0, free - save_gb * GIB))


def run(args) -> int:
    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        dtype = torch.float16
        emit("This GPU has no bf16: running in fp16")
    src = Source(args.input, args.start_frame, args.end_frame, args.fps)
    out_w, out_h = src.width * SCALE, src.height * SCALE
    emit(f"Plan: {src.width}x{src.height} -> {out_w}x{out_h} ({'image' if src.is_image else f'{src.total} frames @ {src.fps:.3f} fps'}), "
         f"window {args.window_size}, {args.dtype}")
    t_load = time.time()
    flash = build_flash(Path(args.flash), dtype, device)
    wan = build_wan(Path(args.wan), dtype)
    budget = _budget_bytes(device, args.save_vram_gb)
    kind = "int8" if is_int8_file(Path(args.wan), INT8_MARKER) else "bf16"
    fit = tune.blocks_on_gpu(budget / GIB, kind, out_w, out_h)
    resident = fit if args.max_gpu_blocks < 0 else min(fit, int(args.max_gpu_blocks))
    emit(f"VRAM budget {budget / GIB:.1f} GB at {out_w}x{out_h}: Wan {module_bytes(wan) / GIB:.1f} GB + KV cache "
         f"{tune.kv_block_gb(out_w, out_h) * tune.WAN_BLOCKS:.1f} GB, activations ~{tune.activation_gb(out_w, out_h):.1f} GB"
         f" -> estimated peak {tune.peak_gb(kind, out_w, out_h, resident):.1f} GB")
    resident = place_wan(wan, device, resident)
    emit(f"Models ready in {time.time() - t_load:.1f}s")

    from refvsr_inference.models.wan.sampling import WanVSRPipeline
    from refvsr_inference.runtime.streaming import CollaborativeFrameStream, InputFrame

    attempt = 0
    while True:
        try:
            return _stream(args, src, wan, flash, device, dtype, WanVSRPipeline, CollaborativeFrameStream, InputFrame)
        except torch.OutOfMemoryError as exc:
            gc.collect()
            torch.cuda.empty_cache()
            on_gpu = [b for b in wan.blocks if not getattr(b, "_relay_swapped", False)]
            if not on_gpu or attempt >= 6:
                emit(f"ERROR: out of VRAM even with Wan streamed from RAM ({str(exc).splitlines()[0]}). "
                     f"Lower the resolution (Max Resolution / pre-downscale) or free GPU memory.")
                return 2
            move = max(1, math.ceil(len(on_gpu) / 3))
            for block in on_gpu[-move:]:
                block.to("cpu")
                _BlockSwap(block, device)
            attempt += 1
            emit(f"[VRAM] Out of VRAM; {move} more Wan blocks now copied from RAM, restarting ({attempt}/6)")


def _stream(args, src, wan, flash, device, dtype, WanVSRPipeline, CollaborativeFrameStream, InputFrame) -> int:
    sampler = WanVSRPipeline(
        scheduler_config={"num_train_timesteps": 1000, "shift": 5.0},
        sampling={"execution_mode": "streaming", "num_inference_steps": 1, "timesteps": [1000],
                  "guidance_scale": 1.0, "seed": args.seed, "kv_cache_window_size": wan.temporal_context_frames},
    )
    sampler.prepare_streaming(model=wan, prompt_embeds=wan.prompt_embeds, device=device, compute_dtype=dtype)
    stream = CollaborativeFrameStream(wan_model=wan, flash_model=flash, wan_pipeline=sampler,
                                      prompt_embeds=wan.prompt_embeds, device=device, compute_dtype=dtype,
                                      window_size=args.window_size, seed=args.seed)
    out_w, out_h = src.width * SCALE, src.height * SCALE
    writer = None
    if not src.is_image:
        audio = args.input if (args.keep_audio and src.has_audio) else None
        writer = video_io.FfmpegWriter(args.output, out_w, out_h, src.fps, audio_source=audio, codec=args.codec,
                                       crf=args.crf, preset=args.preset, pixel_format=args.pixel_format,
                                       audio_offset_s=(src.first / src.fps) if src.first else 0.0)
    packets = (InputFrame(i, torch.from_numpy(f).permute(2, 0, 1).contiguous(), source_time=i / src.fps)
               for i, f in enumerate(src.frames()))
    started = time.time()
    done = 0
    try:
        for result in stream.frames(packets):
            rgb = result.rgb[0].add(1).mul_(127.5).clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().cpu()
            if writer is None:
                video_io.write_image(args.output, rgb.numpy())
            else:
                writer.write(rgb.unsqueeze(0))
            done += 1
            if done == 1 or done % 5 == 0 or done == src.total:
                elapsed = max(1e-6, time.time() - started)
                speed = done / elapsed
                eta = (src.total - done) / speed if speed > 0 else 0
                print(f"FRAME_PROGRESS {done}/{src.total} frames | {100.0 * done / max(1, src.total):.1f}% | "
                      f"{speed:.2f} fps | ETA {int(eta // 60)}m {int(eta % 60):02d}s", flush=True)
        if writer is not None:
            writer.close()
            writer = None
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    finally:
        packets.close()
    total_s = time.time() - started
    peak = (f", peak VRAM {torch.cuda.max_memory_allocated(device) / GIB:.1f} GB" if device.type == "cuda" else "")
    emit(f"Done: {done} frame(s) in {total_s:.1f}s ({done / max(total_s, 1e-6):.2f} fps){peak}")
    print(f"OUTPUT_PATH: {args.output}", flush=True)
    return 0


def convert_int8(args) -> int:
    """One-time build of the INT8 ConvRot Wan file from the bf16 release file."""
    src = Path(args.wan)
    config, _prompt_shape, metadata = _wan_header(src)
    wan = build_wan(src, torch.bfloat16)
    n = quantize_linears(wan, src, log_prefix="[RelayVSR INT8]",
                         include=lambda name: name.startswith("blocks."))
    out = save_int8_model(wan, args.convert_int8, src, format_id=INT8_FORMAT, marker_key=INT8_MARKER,
                          extra_metadata={"config": metadata["config"], "relayvsr_format": "refvsr-wan-v1"})
    emit(f"INT8 ConvRot Wan: {n} layers -> {out} ({out.stat().st_size / GIB:.2f} GB)")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="RelayVSR 4x video / image super-resolution")
    p.add_argument("--input", default="")
    p.add_argument("--output", default="")
    p.add_argument("--wan", required=True, help="Wan checkpoint (bf16 release file or the INT8 ConvRot file)")
    p.add_argument("--flash", default="", help="Flash decoder checkpoint")
    p.add_argument("--window_size", type=int, default=16, help="keyframe interval + 1 (Wan reference every N-1 frames)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, default=-1)
    p.add_argument("--fps", type=float, default=0.0, help="override the input frame rate (0 = source)")
    p.add_argument("--save_vram_gb", type=float, default=1.0)
    p.add_argument("--max_gpu_blocks", type=int, default=-1, help="cap of GPU-resident Wan blocks (-1 = as many as fit)")
    p.add_argument("--codec", default="libx264")
    p.add_argument("--crf", type=int, default=15)
    p.add_argument("--preset", default="medium")
    p.add_argument("--pixel_format", default="yuv420p")
    p.add_argument("--no_audio", dest="keep_audio", action="store_false")
    p.add_argument("--convert_int8", default="", help="build the INT8 ConvRot Wan file at this path and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.convert_int8:
        return convert_int8(args)
    if not args.input or not args.output or not args.flash:
        raise SystemExit("--input, --output and --flash are required")
    if args.window_size < 2:
        raise SystemExit("--window_size must be at least 2")
    with torch.no_grad():
        return run(args)


if __name__ == "__main__":
    sys.exit(main())
