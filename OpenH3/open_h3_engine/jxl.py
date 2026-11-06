"""
JonXL H3 Upscale: JonXL's MiniMax H3 image upscaler (Civitai "Minimax H3 Image Upscaler (LORA + Workflow & Batch)",
workflow JXL_H3_Upscale_Easy_v1, LoRA h3_upscale_1920) on the app's own Open-H3 engine (no ComfyUI backend).

Per picture (an image, or every frame of a video):
  canvas: ImageScaleToTotalPixels (max W x max H pixels) -> fit under max W x max H, sides floored to 32
  first frame: keep-proportion Lanczos resize onto the canvas (even sides) -> centre crop + Lanczos to the canvas
  -> Qwen3-VL-32B conditioning of "<Picture 1>" (the first frame) + the fixed JonXL prompt
  -> FL2VA + h3_upscale_1920 LoRA (strength 1), first-frame keyframe, 24 frames (H3 grid: 39) at 24 fps,
     joint audio stream, er_sde on the beta57 schedule (beta 0.5 / 0.7, 20 steps, shift-12 flow), CFG 1
  -> H3 VAE decode -> the LAST frame is the upscaled picture (the clip cross-fades from the blurry input to sharp).

Models load once per run: text encoder for every picture, then the VAE encoder, then the DiT, then the decoder.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

import comfy.model_management as mm
import comfy.utils
from comfy.ldm.minimax.model import PackedLayout

from . import loader as loader_mod
from . import lora as lora_mod
from . import video_io

JXL_PROMPT = "A static still blurry pixelated picture crossfades into a clear sharp detailed picture, 4psc4l3,"
JXL_SEED = 505817956432961          # the workflow's fixed "Seed control"
JXL_STEPS = 20
JXL_LENGTH = 24                     # "Duration (Frames)"; the H3 grid snaps it to 39 (17k + 5)
JXL_MAX_SIDE = 1920
JXL_SHIFT = 12.0                    # MiniMax H3 flow shift (video), audio 3.0
JXL_AUDIO_SHIFT = 3.0
AUDIO_LATENTS_PER_SECOND = 40
CANVAS_MULTIPLE = 32


@dataclass
class JXLSettings:
    input_path: str = ""
    output_path: str = ""
    dit_path: str = ""
    text_encoder_path: str = ""
    vae_path: str = ""
    lora_path: str = ""
    cache_dir: str = ""
    prompt: str = JXL_PROMPT
    seed: int = JXL_SEED
    steps: int = JXL_STEPS
    length: int = JXL_LENGTH
    max_width: int = JXL_MAX_SIDE
    max_height: int = JXL_MAX_SIDE
    lora_strength: float = 1.0
    geometry: str = "keep"           # keep (exact aspect, padded canvas) | workflow (parity tests)
    start_frame: int = 0
    end_frame: int = -1
    reserve_vram_gb: float = 2.0
    max_gpu_blocks: int = 50
    bytes_per_token: float = 145_000.0
    ffn_chunk_tokens: int = 32768
    codec: str = "libx264"
    crf: int = 15
    preset: str = "medium"
    pixel_format: str = "yuv420p"
    keep_audio: bool = True


# ------------------------------------------------------------------ geometry and first frame

def jxl_canvas(src_w: int, src_h: int, max_w: int = JXL_MAX_SIDE, max_h: int = JXL_MAX_SIDE) -> Tuple[int, int]:
    """Output canvas of the workflow: ImageScaleToTotalPixels(max_w * max_h / 1e6 MP) then ImageResizeKJv2
    'resize' into max_w x max_h with divisible_by 32."""
    total = (max_w * max_h / 1e6) * 1024 * 1024
    scale = math.sqrt(total / (src_w * src_h))
    w1, h1 = round(src_w * scale), round(src_h * scale)
    ratio = min(max_w / w1, max_h / h1)
    w2, h2 = round(w1 * ratio), round(h1 * ratio)
    w2, h2 = w2 - w2 % CANVAS_MULTIPLE, h2 - h2 % CANVAS_MULTIPLE
    if w2 < CANVAS_MULTIPLE or h2 < CANVAS_MULTIPLE:
        raise ValueError(f"{src_w}x{src_h} is too narrow for a {max_w}x{max_h} JonXL canvas")
    return w2, h2


def _lanczos(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """[1, H, W, 3] float -> [1, height, width, 3] float (comfy.utils.common_upscale 'lanczos', crop disabled)."""
    return comfy.utils.common_upscale(image.movedim(-1, 1), int(width), int(height), "lanczos", "disabled").movedim(1, -1)


@dataclass
class Geometry:
    width: int         # canvas the model renders (multiple of 32)
    height: int
    out_w: int         # picture kept from the canvas
    out_h: int
    x0: int = 0
    y0: int = 0
    mode: str = "keep"


def plan_geometry(src_w: int, src_h: int, max_w: int = JXL_MAX_SIDE, max_h: int = JXL_MAX_SIDE,
                  mode: str = "keep") -> Geometry:
    """'keep' (the app): the picture fitted under max_w x max_h at its exact aspect (even sides), edge-padded onto
    the 32-pixel canvas the model needs and cropped back after decoding: no stretch, nothing cut off.
    'workflow': the workflow's own canvas (jxl_canvas), whose first frame is cropped to sides floored to 32 and then
    stretched onto the canvas (a 960x540 picture becomes 1856x1056 stretched to 1920x1056); kept for parity tests."""
    if mode == "workflow":
        w, h = jxl_canvas(src_w, src_h, max_w, max_h)
        return Geometry(w, h, w, h, 0, 0, mode)
    total = (max_w * max_h / 1e6) * 1024 * 1024  # same area rule as the workflow, then fit under the max sides
    scale = math.sqrt(total / (src_w * src_h))
    scale = min(scale, max_w / src_w, max_h / src_h)
    out_w, out_h = max(2, int(round(src_w * scale / 2)) * 2), max(2, int(round(src_h * scale / 2)) * 2)
    out_w, out_h = min(out_w, max_w - max_w % 2), min(out_h, max_h - max_h % 2)
    width = -(-out_w // CANVAS_MULTIPLE) * CANVAS_MULTIPLE
    height = -(-out_h // CANVAS_MULTIPLE) * CANVAS_MULTIPLE
    return Geometry(width, height, out_w, out_h, (width - out_w) // 2, (height - out_h) // 2, mode)


def prepare_first_frame(frame_u8: np.ndarray, geo: Geometry) -> np.ndarray:
    """First frame on the canvas (uint8 [height, width, 3]); every resize is the workflow's Lanczos pass over 8-bit
    pictures (comfy.utils.common_upscale), so the result is exact in uint8."""
    img = torch.from_numpy(np.ascontiguousarray(frame_u8)).to(torch.float32).div(255.0).unsqueeze(0)
    src_h, src_w = img.shape[1], img.shape[2]
    if geo.mode != "workflow":
        img = _lanczos(img, geo.out_w, geo.out_h)
        pic = np.ascontiguousarray(np.clip(np.rint(img[0].numpy() * 255.0), 0, 255).astype(np.uint8))
        pad = ((geo.y0, geo.height - geo.out_h - geo.y0), (geo.x0, geo.width - geo.out_w - geo.x0), (0, 0))
        return np.pad(pic, pad, mode="edge") if any(sum(p) for p in pad[:2]) else pic
    # ImageResizeKJv2 'resize' to the canvas (divisible by 2) ...
    ratio = min(geo.width / src_w, geo.height / src_h)
    w1, h1 = round(src_w * ratio), round(src_h * ratio)
    w1, h1 = w1 - w1 % 2, h1 - h1 % 2
    img = _lanczos(img, w1, h1)
    # ... then ImageResizeKJv2 'crop' to that size floored to 32 (centre) ...
    cw, ch = w1 - w1 % CANVAS_MULTIPLE, h1 - h1 % CANVAS_MULTIPLE
    if w1 / h1 > cw / ch:
        crop_w, crop_h = round(h1 * (cw / ch)), h1
    else:
        crop_w, crop_h = w1, round(w1 / (cw / ch))
    img = img.narrow(-2, (w1 - crop_w) // 2, crop_w).narrow(-3, (h1 - crop_h) // 2, crop_h)
    img = _lanczos(img, cw, ch)
    # ... then MiniMaxH3ImageToVideo stretches the first frame onto the canvas
    img = _lanczos(img, geo.width, geo.height)
    return np.ascontiguousarray(np.clip(np.rint(img[0].numpy() * 255.0), 0, 255).astype(np.uint8))


def frame_count(length: int) -> int:
    n = max(5, int(length))
    return n + (5 - n) % 17


def video_latent_t(frames: int) -> int:
    return 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2


# ------------------------------------------------------------------ schedule and sampler

def _time_snr_shift(alpha: float, t: torch.Tensor) -> torch.Tensor:
    return t if alpha == 1.0 else alpha * t / (1 + (alpha - 1) * t)


def flow_sigma_table(shift: float = JXL_SHIFT) -> torch.Tensor:
    """ModelSamplingDiscreteFlow.sigmas (1000 entries, float32)."""
    ts = (torch.arange(1, 1001, 1) / 1000) * 1000
    return _time_snr_shift(shift, ts / 1000)


def beta57_sigmas(steps: int, shift: float = JXL_SHIFT) -> torch.Tensor:
    """BasicScheduler 'beta57' (RES4LYF: comfy beta_scheduler with alpha 0.5, beta 0.7), denoise 1."""
    import scipy.stats

    table = flow_sigma_table(shift)
    total = len(table) - 1
    ts = 1 - np.linspace(0, 1, int(steps), endpoint=False)
    ts = np.rint(scipy.stats.beta.ppf(ts, 0.5, 0.7) * total)
    sigs, last = [], -1
    for t in ts:
        if t != last:
            sigs.append(float(table[int(t)]))
        last = t
    sigs.append(0.0)
    return torch.FloatTensor(sigs)


def _percent_to_sigma(percent: float, shift: float) -> float:
    if percent <= 0.0:
        return 1.0
    if percent >= 1.0:
        return 0.0
    return float(_time_snr_shift(shift, torch.tensor(1.0 - percent)))


@torch.no_grad()
def sample_er_sde(model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x: torch.Tensor,
                  sigmas: torch.Tensor, seed: int, shift: float = JXL_SHIFT, step_cb=None,
                  max_stage: int = 3) -> torch.Tensor:
    """k-diffusion er_sde (VP ER-SDE-Solver-3) on a flow (CONST) model, ComfyUI sample_er_sde parity:
    the first sigma moved off 1, noise from a generator seeded with the noise seed on x's device."""
    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed) + (1 if x.device.type == "cpu" else 0))

    def noise():
        return torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)

    def scaler(v):
        return v * ((v ** 0.3).exp() + 10.0)

    points = 200.0
    point_indice = torch.arange(0, points, dtype=torch.float32, device=x.device)
    sigmas = sigmas.clone().to(x.device)
    if len(sigmas) > 1 and sigmas[0] >= 1:
        sigmas[0] = _percent_to_sigma(1e-4, shift)
    er_lambdas = sigmas.logit().exp()  # half-logSNR = -logit(sigma) for a flow model; er_lambda = exp(-half-logSNR)
    s_in = x.new_ones([x.shape[0]])
    old_denoised = old_denoised_d = None
    for i in range(len(sigmas) - 1):
        denoised = model_fn(x, sigmas[i] * s_in)
        stage = min(max_stage, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            lam_s, lam_t = er_lambdas[i], er_lambdas[i + 1]
            alpha_s = sigmas[i] / lam_s
            alpha_t = sigmas[i + 1] / lam_t
            r_alpha = alpha_t / alpha_s
            r = scaler(lam_t) / scaler(lam_s)
            x = r_alpha * r * x + alpha_t * (1 - r) * denoised
            if stage >= 2:
                dt = lam_t - lam_s
                step = -dt / points
                pos = lam_t + point_indice * step
                scaled = scaler(pos)
                s = torch.sum(1 / scaled) * step
                denoised_d = (denoised - old_denoised) / (lam_s - er_lambdas[i - 1])
                x = x + alpha_t * (dt + s * scaler(lam_t)) * denoised_d
                if stage >= 3:
                    s_u = torch.sum((pos - lam_s) / scaled) * step
                    denoised_u = (denoised_d - old_denoised_d) / ((lam_s - er_lambdas[i - 2]) / 2)
                    x = x + alpha_t * ((dt ** 2) / 2 + s_u * scaler(lam_t)) * denoised_u
                old_denoised_d = denoised_d
            x = x + alpha_t * noise() * (lam_t ** 2 - lam_s ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)
        old_denoised = denoised
        if step_cb is not None:
            step_cb(i)
    return x


# ------------------------------------------------------------------ run

@dataclass
class _Unit:
    item: int          # input index
    frame: int         # frame position within the input (0 for an image)
    width: int
    height: int


class _Item:
    def __init__(self, s: JXLSettings):
        self.s = s
        self.is_image = video_io.is_image(s.input_path)
        if self.is_image:
            img = video_io.read_image(s.input_path)
            self.src_w, self.src_h = img.shape[1], img.shape[0]
            self.fps = Fraction(24, 1)
            self.first, self.count, self.has_audio = 0, 1, False
        else:
            info = video_io.probe_video(s.input_path)
            self.src_w, self.src_h, self.fps, self.has_audio = info.width, info.height, info.fps, info.has_audio
            self.first = max(0, int(s.start_frame))
            last = info.frames - 1 if int(s.end_frame) < 0 else min(info.frames - 1, int(s.end_frame))
            if last < self.first:
                raise RuntimeError("The selected frame range is empty.")
            self.count = last - self.first + 1
        self.geo = plan_geometry(self.src_w, self.src_h, int(s.max_width), int(s.max_height), str(s.geometry))
        self.width, self.height = self.geo.width, self.geo.height

    def frames(self):
        """Source uint8 frames of the selected range."""
        if self.is_image:
            yield video_io.read_image(self.s.input_path)
            return
        for pos, frame in enumerate(video_io.iter_video_frames(self.s.input_path)):
            if pos < self.first:
                continue
            if pos >= self.first + self.count:
                return
            yield frame


def _tokens(width: int, height: int, latent_t: int, text_len: int, audio_t: int) -> int:
    rows = (height // 16 + 1) // 2 * ((width // 16 + 1) // 2)
    return text_len + rows * (latent_t + 1) + audio_t * 2


def run_jxl(items: List[JXLSettings], emit: Callable[[str], None]) -> List[Tuple[Optional[str], Optional[BaseException]]]:
    """Upscale every input (images, or every frame of videos) with the models loaded once.
    Returns [(output path or None, error or None)] in input order; a failing input never stops the others."""
    torch.set_grad_enabled(False)
    started = time.time()
    s0 = items[0]
    device = mm.get_torch_device()
    loader_mod.apply_vram_cap(device, float(s0.reserve_vram_gb), emit=emit)
    loader_mod.guard_attention_oom()
    results: List[list] = [[None, None] for _ in items]
    cache_dir = Path(s0.cache_dir or (Path(s0.output_path).parent / ".open_h3_cache"))
    work = cache_dir / "jxl_runs" / f"run_{os.getpid()}_{int(time.time() * 1000)}"
    work.mkdir(parents=True, exist_ok=True)

    def fail(index: int, exc: BaseException) -> None:
        results[index][1] = exc
        emit(f"[OpenH3] ITEM_ERROR {index}: {exc}")

    plans: List[Optional[_Item]] = []
    for i, s in enumerate(items):
        try:
            it = _Item(s)
            plans.append(it)
            what = "image" if it.is_image else f"{it.count} frame(s) @ {float(it.fps):.6g} fps"
            emit(f"[Plan] {f'[{i + 1}/{len(items)}] ' if len(items) > 1 else ''}JonXL H3 Upscale: "
                 f"{it.src_w}x{it.src_h} -> {it.geo.out_w}x{it.geo.out_h} ({what}; model canvas {it.width}x{it.height}), "
                 f"{frame_count(s.length)}-frame clip, "
                 f"{int(s.steps)} steps er_sde / beta57, seed {int(s.seed)}")
        except Exception as exc:
            plans.append(None)
            fail(i, exc)
    units = [_Unit(i, k, it.width, it.height) for i, it in enumerate(plans) if it is not None for k in range(it.count)]
    total = len(units)
    if not total:
        return [tuple(r) for r in results]

    def unit_file(u: _Unit, kind: str) -> Path:
        return work / f"{kind}_{u.item:04d}_{u.frame:06d}.safetensors"

    try:
        # 1) first frames + text conditioning (Qwen3-VL sees the first frame as <Picture 1>)
        emit(f"Phase 1/4: first frames + text encoding (Qwen3-VL-32B with the picture, {total} picture(s))...")
        t0 = time.time()
        from safetensors.torch import load_file, save_file

        te = None
        cond_len = 0
        for i, it in enumerate(plans):
            if it is None:
                continue
            s = items[i]
            try:
                for k, frame in enumerate(it.frames()):
                    u = _Unit(i, k, it.width, it.height)
                    first = prepare_first_frame(frame, it.geo)
                    key = hashlib.sha256(first.tobytes() + f"|{first.shape}|{s.prompt}|"
                                         f"{_sig(s.text_encoder_path)}".encode("utf-8")).hexdigest()[:32]
                    cached = cache_dir / "jxl_text" / f"{key}.safetensors"
                    if it.is_image and cached.is_file():
                        tensors = load_file(str(cached))
                    else:
                        if te is None:
                            te = loader_mod.H3TextEncoder(s.text_encoder_path, device, emit=emit)
                            te.place(int(loader_mod.usable_vram_bytes(device) - 2.5 * loader_mod.GIB), emit=emit)
                            te.place_vision(emit=emit)
                        picture = torch.from_numpy(first).to(torch.float32).div(255.0).unsqueeze(0)
                        cond, tags = te.encode(s.prompt, images=[picture])
                        tensors = {"cond": cond.detach().to(torch.bfloat16).cpu().contiguous()}
                        if tags is not None:
                            tensors["tags"] = tags.detach().cpu().contiguous()
                        if it.is_image:
                            cached.parent.mkdir(parents=True, exist_ok=True)
                            save_file(tensors, str(cached))
                    tensors["first"] = torch.from_numpy(np.ascontiguousarray(first)).contiguous()
                    save_file(tensors, str(unit_file(u, "cond")))
                    cond_len = max(cond_len, int(tensors["cond"].shape[1]))
                    if total > 1:
                        emit(f"Text conditioning {sum(1 for x in units if x.item < i) + k + 1}/{total}")
                if it.count != sum(1 for x in units if x.item == i):
                    raise RuntimeError("fewer frames decoded than the video reports")
            except Exception as exc:
                fail(i, exc)
        if te is not None:
            te.unload()
            del te
        gc.collect()
        mm.soft_empty_cache()
        emit(f"Text conditioning done in {time.time() - t0:.1f}s")
        units = [u for u in units if results[u.item][1] is None]
        if not units:
            return [tuple(r) for r in results]

        # 2) keyframe latents
        emit("Phase 2/4: VAE encode of the first frames...")
        t0 = time.time()
        vae = loader_mod.H3VideoVAE(s0.vae_path, device, emit=emit)
        for u in units:
            path = unit_file(u, "cond")
            data = load_file(str(path))
            first = data["first"].to(torch.float32).div(255.0).unsqueeze(0)
            data["key"] = vae.encode(first).contiguous()
            save_file(data, str(path))
        vae.offload()
        gc.collect()
        mm.soft_empty_cache()
        emit(f"Encode done in {time.time() - t0:.1f}s")

        # 3) DiT: FL2VA + JonXL LoRA, er_sde / beta57
        emit("Phase 3/4: loading MiniMax H3 FL2VA + JonXL upscale LoRA...")
        model, _config, dtype, _quantized = loader_mod.load_dit(s0.dit_path, device, emit=emit)
        frames_n = frame_count(s0.length)
        latent_t = video_latent_t(frames_n)
        audio_t = round(frames_n / 24 * AUDIO_LATENTS_PER_SECOND)
        max_tokens = max(_tokens(u.width, u.height, latent_t, cond_len, audio_t) for u in units)
        activation = int(max_tokens * float(s0.bytes_per_token))
        usable = loader_mod.usable_vram_bytes(device)
        emit(f"VRAM plan: {usable / loader_mod.GIB:.1f} GB usable, {activation / loader_mod.GIB:.1f} GB kept for "
             f"activations ({max_tokens} tokens), up to {int(s0.max_gpu_blocks)} DiT blocks on GPU")
        blocks = list(model.blocks)
        loader_mod.place_blocks(model, blocks, device, int(usable - activation), emit=emit, label="DiT blocks",
                                max_resident=int(s0.max_gpu_blocks))
        t_lora = time.time()
        lora_mod.apply_kohya_lora(model, s0.lora_path, float(s0.lora_strength), device, dtype, emit=emit)
        emit(f"LoRA merged in {time.time() - t_lora:.1f}s")
        if int(s0.ffn_chunk_tokens) > 0:
            from .pipeline import _ChunkedMLP

            _ChunkedMLP(int(s0.ffn_chunk_tokens)).install(model)
        sigmas = beta57_sigmas(int(s0.steps))
        steps = len(sigmas) - 1
        attention = _attention_override(device, emit)
        all_steps = steps * len(units)
        done = [0]
        t0 = time.time()
        for n, u in enumerate(units):
            if results[u.item][1] is not None:
                continue
            s = items[u.item]
            f_start = time.time()
            data = load_file(str(unit_file(u, "cond")))
            try:
                latent = None
                for attempt in range(40):
                    done_before = done[0]
                    try:
                        latent = _sample(model, dtype, device, data, u, s, sigmas, latent_t, audio_t, steps,
                                         lambda st: _progress(emit, done, all_steps, n, len(units), st, steps),
                                         attention)
                        break
                    except BaseException as exc:
                        if not loader_mod.is_oom(exc):
                            raise
                        failed = loader_mod.oom_note(exc)
                        shortfall = loader_mod.oom_shortfall(exc, device)
                    done[0] = done_before
                    gc.collect()
                    mm.soft_empty_cache()
                    emit(f"[VRAM] Picture {n + 1} reached the VRAM limit ({failed}); streaming more DiT blocks "
                         f"from RAM and retrying - no shared GPU memory is used")
                    if not loader_mod.stream_more_blocks(blocks, max(shortfall, int(0.5 * loader_mod.GIB)), emit=emit):
                        raise RuntimeError("Not enough VRAM even with every DiT block streamed from RAM. Lower the "
                                           "JonXL max size, lower Save VRAM, or close other GPU apps.")
                save_file({"video": latent.to(torch.float16).contiguous()}, str(unit_file(u, "latent")))
                unit_file(u, "cond").unlink(missing_ok=True)
                emit(f"Picture {n + 1}/{len(units)} sampled in {time.time() - f_start:.1f}s")
            except Exception as exc:
                fail(u.item, exc)
        emit(f"Sampling done in {time.time() - t0:.1f}s")
        del model, blocks
        gc.collect()
        mm.soft_empty_cache()

        # 4) decode, keep the last frame, write
        emit("Phase 4/4: VAE decode (last frame of each clip) + write...")
        t0 = time.time()
        for i, it in enumerate(plans):
            if it is None or results[i][1] is not None:
                continue
            s = items[i]
            writer = None
            try:
                if not it.is_image:
                    audio = s.input_path if (s.keep_audio and it.has_audio) else None
                    writer = video_io.FfmpegWriter(s.output_path, it.geo.out_w, it.geo.out_h, it.fps, audio_source=audio,
                                                   codec=s.codec, crf=int(s.crf), preset=s.preset,
                                                   pixel_format=s.pixel_format,
                                                   audio_offset_s=float(it.first / it.fps) if it.first else 0.0)
                for k in range(it.count):
                    u = _Unit(i, k, it.width, it.height)
                    latent = load_file(str(unit_file(u, "latent")))["video"].to(torch.float32)
                    pixels = vae.decode(latent)                      # [1, 3, frames, H, W] in [0, 1]
                    g = it.geo
                    last = pixels[0, :, -1].movedim(0, -1)[g.y0:g.y0 + g.out_h, g.x0:g.x0 + g.out_w]
                    frame = video_io.to_uint8(last.float())
                    if writer is None:
                        video_io.write_image(s.output_path, frame.numpy())
                    else:
                        writer.write(frame.unsqueeze(0))
                    unit_file(u, "latent").unlink(missing_ok=True)
                    if it.count > 1:
                        emit(f"Decoded {k + 1}/{it.count}")
                if writer is not None:
                    writer.close()
                    writer = None
                results[i][0] = s.output_path
                if len(items) > 1:
                    emit(f"[OpenH3] ITEM_DONE {i} OUTPUT_PATH: {s.output_path}")
            except Exception as exc:
                if writer is not None:
                    writer.abort()
                fail(i, exc)
        vae.offload()
        emit(f"Decode + write done in {time.time() - t0:.1f}s")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    ok = sum(1 for r in results if r[0])
    emit(f"JonXL H3 Upscale complete -> {ok}/{len(items)} output(s) | total {time.time() - started:.1f}s"
         if len(items) > 1 else f"JonXL H3 Upscale complete -> {results[0][0]} | total {time.time() - started:.1f}s")
    return [tuple(r) for r in results]


def _sig(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{Path(path).name}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        return Path(path).name


def _attention_override(device, emit):
    """The workflow's ModelAttentionBackend 'comfy kitchen attention' (INT8 attention) when this GPU has it,
    otherwise the engine's default attention (Sage / PyTorch)."""
    try:
        import comfy_kitchen
        from comfy.ldm.modules import attention as attn

        if attn.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE and comfy_kitchen.int8_attention_is_available(device):
            fn = attn.get_attention_function("comfy_kitchen_int8", None)
            if fn is not None:
                def override(_func, *args, **kwargs):
                    return fn(*args, **kwargs)

                emit("Attention: Comfy Kitchen INT8 (the workflow's attention backend)")
                return override
    except Exception as exc:  # noqa: BLE001 - fall back to the default attention
        emit(f"Attention: default ({exc})")
        return None
    emit("Attention: default (Comfy Kitchen INT8 attention is not available on this GPU)")
    return None


def _progress(emit, done, total, n, units, step, steps) -> None:
    done[0] += 1
    emit(f"Processing: {done[0]}/{total} | picture {n + 1}/{units} step {step + 1}/{steps}")


@torch.no_grad()
def _sample(model, dtype, device, data, u: _Unit, s: JXLSettings, sigmas, latent_t, audio_t, steps, step_cb,
            attention=None):
    """One picture: SamplerCustom (add noise, CFG 1) over the MiniMaxH3ImageToVideo latent and conditioning."""
    lat_h, lat_w = u.height // 16, u.width // 16
    video_shape = (1, 24, latent_t, lat_h, lat_w)
    audio_shape = (1, 32, 2, audio_t)
    seed = int(s.seed)
    gen = torch.manual_seed(seed)  # comfy.sample.prepare_noise: video stream, then audio stream, CPU
    noise_v = torch.randn(video_shape, dtype=torch.float32, generator=gen, device="cpu")
    noise_a = torch.randn(audio_shape, dtype=torch.float32, generator=gen, device="cpu")
    nv = noise_v.numel()
    x = torch.cat([noise_v.reshape(1, 1, -1), noise_a.reshape(1, 1, -1)], dim=-1).to(device)  # sigma 1: pure noise
    del noise_v, noise_a
    context = model.preprocess_text_embeds(data["cond"].to(device, dtype))
    tags = data.get("tags")
    keyframes = [{"resolved_frame_index": 0, "latent": data["key"]}]
    layout = PackedLayout(context.shape[1], latent_t, (lat_h + 1) // 2 * 2, (lat_w + 1) // 2 * 2, audio_t,
                          keyframes=keyframes)
    payload = {"keyframes": keyframes, "cond_video_latents": [data["key"]], "cond_audio_latents": [],
               "seed": seed, "audio_scale": JXL_SHIFT / JXL_AUDIO_SHIFT, "layout": layout}
    if tags is not None:
        payload["text_token_tags"] = tags
    sample_sigmas = sigmas.to(device)

    def model_fn(xp: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        xv = xp[..., :nv].reshape(video_shape)
        xa = xp[..., nv:].reshape(audio_shape)
        options = {"sigmas": sigma.view(1), "sample_sigmas": sample_sigmas}
        if attention is not None:
            options["optimized_attention_override"] = attention
        out = model([xv.to(dtype), xa.to(dtype)], (sigma * 1000.0).view(1).float(), context,
                    transformer_options=options, minimax_payload=payload, denoise_mask=None, audio_denoise_mask=None)
        v = torch.cat([out[0].float().reshape(1, 1, -1), out[1].float().reshape(1, 1, -1)], dim=-1)
        del out
        return xp - v * sigma.view(1, 1, 1)  # CONST.calculate_denoised

    x = sample_er_sde(model_fn, x, sigmas, seed, step_cb=step_cb)
    return x[..., :nv].reshape(video_shape).cpu()


def describe(path: str, max_side: int = JXL_MAX_SIDE) -> str:
    """'W x H -> W' x H'' for a plan line, without loading models."""
    info = video_io.probe_video(path, count=False) if not video_io.is_image(path) else None
    if info is None:
        img = video_io.read_image(path)
        w, h = img.shape[1], img.shape[0]
    else:
        w, h = info.width, info.height
    g = plan_geometry(w, h, max_side, max_side)
    return json.dumps({"source": [w, h], "output": [g.out_w, g.out_h], "canvas": [g.width, g.height]})
