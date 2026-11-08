"""
Open-H3-Regenerate-2K pipeline (NVIDIA recipe, ComfyUI-Open-H3-Regenerate parity).

Per window of 17k + 5 frames:
  draft (whole frame resized to the 768P / 576P class size) -> H3 VAE encode
  -> H3 latent upscaler onto the target grid (two passes above the 2K area)
  -> re-noise to sigma 0.909375 -> 3 Euler steps (0.909375 -> 0.857 -> 0.667 -> 0), CFG 1,
     FL2VA + merged LoRA (refiner + LightX2V FL2VA Turbo + LMS Sharp) + 50 Sharp curve branches,
     audio stream held clean (encoded silence, or the source soundtrack)
  -> H3 VAE decode (Sol fast decoder by default) -> trim padding -> exact output size.

Every model loads once per run: text encoder (skipped when the prompt is cached),
then VAE + upscaler for all windows, then the DiT for all windows, then the decoder.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.ops
from comfy.ldm.minimax.model import PackedLayout

from . import loader as loader_mod
from . import lora as lora_mod
from . import planner as planner_mod
from . import regen_core as core
from . import sparse as sparse_mod
from . import video_io

log = logging.getLogger("openh3.pipeline")

DEFAULT_SEED = 1766580932
AUDIO_LATENTS_PER_SECOND = 40  # H3 audio latent rate (32 kHz / 800)


@dataclass
class OpenH3Settings:
    input_path: str = ""
    output_path: str = ""
    dit_path: str = ""
    text_encoder_path: str = ""
    vae_path: str = ""
    decode_vae_path: str = ""          # empty = same file as vae_path
    audio_vae_path: str = ""
    upscaler_path: str = ""
    lora_path: str = ""
    curve_path: str = ""
    table_path: str = ""
    cache_dir: str = ""

    prompt: str = ""
    seed: int = DEFAULT_SEED
    steps: int = 3
    start_sigma: float = 0.0
    lora_strength: float = 1.0
    curve_strength: float = 1.0

    target_width: int = 0
    target_height: int = 0
    draft_size: str = "768p"           # 768p | 576p | native
    max_window_frames: int = 121
    overlap_frames: int = 8
    fps_mode: str = "source"           # source (1:1 frames) | 24 (reference resample)
    audio_mode: str = "silence"        # silence | source
    start_frame: int = 0
    end_frame: int = -1

    sparse_attention: bool = True
    sparse_tau: float = 1.2
    sparse_start_percent: float = 0.0
    sparse_end_percent: float = 0.9
    sparse_dense_steps: int = -1       # -1 = start/end percent window; N >= 0 = first N refine steps dense, rest sparse
    second_pass_sigma: float = 0.0     # > 0: refine the refined latent again at the same size from this noise level
    dense_blocks: str = "0,1,49"
    sparse_min_tokens: int = 12288
    sparse_extra_tokens: int = 256
    sink_conditioning: str = "exact_kv_and_rows"

    ffn_chunk_tokens: int = 32768      # 0 = off; exact per-token split of the MLP
    reserve_vram_gb: float = 2.0       # "Save VRAM": always left free; the engine caps itself below it
    max_gpu_blocks: int = 50           # block swap: most DiT blocks kept on the GPU (fewer if they do not fit)
    bytes_per_token: float = 145_000.0  # refine VRAM per packed token: 116 KB live at 2K + allocator slack
    token_budget_mode: str = "auto"    # auto | manual
    max_tokens: int = 160000

    codec: str = "libx264"
    crf: int = 15
    preset: str = "medium"
    pixel_format: str = "yuv420p"
    keep_audio: bool = True
    keep_latents: bool = False
    video_args: str = ""               # own FFmpeg video encoder arguments (replace codec/crf/preset/pix_fmt)


@dataclass
class OpenH3Progress:
    on_line: Callable[[str], None] = field(default=lambda s: None)

    def line(self, text: str) -> None:
        try:
            self.on_line(text)
        except Exception:
            pass


def _file_signature(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{Path(path).name}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        return Path(path).name


class _ChunkedMLP:
    """Exact per-token split of the H3 SwiGLU MLP: same rows, a fraction of the activation memory."""

    def __init__(self, chunk: int):
        self.chunk = int(chunk)

    def install(self, model: nn.Module) -> int:
        count = 0
        for block in model.blocks:
            mlp = block.mlp
            if getattr(mlp, "_open_h3_chunked", False):
                continue
            mlp.forward = self._make_forward(mlp)
            mlp._open_h3_chunked = True
            count += 1
        return count

    def _make_forward(self, mlp):
        chunk = self.chunk

        def plain(x):
            return comfy.ops.linear_input_act(mlp.fc2, mlp.fc1(x), "swiglu")

        def forward(x):
            # the blocks pass [1, tokens, channels]: split over every token row, not the first dimension
            lead = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
            n = x.shape[0]
            if n <= chunk:
                return plain(x).view(*lead, -1)
            swapped = []
            for lin in (mlp.fc1, mlp.fc2):
                weight = lin.weight
                if weight is not None and weight.device != x.device and not getattr(lin, "weight_function", None):
                    lin.weight = nn.Parameter(weight.to(x.device), requires_grad=False)
                    swapped.append((lin, weight))
            try:
                out = torch.empty((n, mlp.fc2.out_features), dtype=x.dtype, device=x.device)
                for i in range(0, n, chunk):
                    out[i:i + chunk] = plain(x[i:i + chunk])
                return out.view(*lead, -1)
            finally:
                for lin, weight in swapped:
                    lin.weight = weight

        return forward


class OpenH3Pipeline:
    def __init__(self, settings: OpenH3Settings, progress: Optional[OpenH3Progress] = None):
        self.s = settings
        self.progress = progress or OpenH3Progress()
        self.device = mm.get_torch_device()
        self.cache_dir = Path(settings.cache_dir or (Path(settings.output_path).parent / ".open_h3_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def line(self, text: str) -> None:
        self.progress.line(text)

    # ------------------------------------------------------------- input
    def _timeline(self) -> Tuple[List[int], Fraction, int, int, bool, bool]:
        """Output-timeline -> source frame indices, output fps, width, height, has_audio, is_image."""
        s = self.s
        if video_io.is_image(s.input_path):
            frame = video_io.read_image(s.input_path)
            self._image = frame
            return [0], Fraction(24, 1), frame.shape[1], frame.shape[0], False, True
        info = video_io.probe_video(s.input_path)
        first = max(0, int(s.start_frame))
        last = info.frames - 1 if int(s.end_frame) < 0 else min(info.frames - 1, int(s.end_frame))
        if last < first:
            raise RuntimeError("The selected frame range is empty.")
        count = last - first + 1
        if str(s.fps_mode) == "24" and abs(float(info.fps) - 24.0) > 1e-3:
            indices = [first + i for i in core.resample_indices(count, float(info.fps))]
            fps = Fraction(24, 1)
        else:
            indices = list(range(first, last + 1))
            fps = info.fps
        return indices, fps, info.width, info.height, info.has_audio, False

    def _iter_source(self, indices: List[int]):
        """Yields uint8 frames for each output-timeline position (decode once, repeat when resampling)."""
        if getattr(self, "_image", None) is not None:
            yield self._image
            return
        wanted = iter(indices)
        target = next(wanted, None)
        current = None
        for pos, frame in enumerate(video_io.iter_video_frames(self.s.input_path)):
            current = frame
            while target is not None and target == pos:
                yield current
                target = next(wanted, None)
            if target is None:
                return
        while target is not None and current is not None:  # stream ended early: hold the last frame
            yield current
            target = next(wanted, None)

    def _draft(self, frames_u8: List[np.ndarray], geometry: planner_mod.Geometry) -> torch.Tensor:
        """uint8 frames -> float [T_padded, draft_h, draft_w, 3] in [0, 1], last frame held to 17k+5."""
        dw, dh = geometry.draft_w, geometry.draft_h
        out = []
        device = self.device
        for i in range(0, len(frames_u8), 16):
            part = torch.from_numpy(np.stack(frames_u8[i:i + 16])).to(device)
            part = part.permute(0, 3, 1, 2).to(torch.float32) / 255.0
            src_h, src_w = part.shape[-2:]
            if (src_w, src_h) != (dw, dh):
                part = F.interpolate(part, size=(dh, dw), mode="bilinear", align_corners=False,
                                     antialias=dw < src_w or dh < src_h)
            out.append(part.clamp(0, 1).permute(0, 2, 3, 1).cpu())
        frames = torch.cat(out)
        padded = planner_mod.align_frame_count(frames.shape[0])
        if padded > frames.shape[0]:
            frames = torch.cat([frames, frames[-1:].expand(padded - frames.shape[0], -1, -1, -1)])
        return frames.contiguous()

    # ------------------------------------------------------- conditioning
    def _text_condition(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        s = self.s
        prompt = str(s.prompt or "")
        key = hashlib.sha256(f"{_file_signature(s.text_encoder_path)}|{prompt}".encode("utf-8")).hexdigest()[:32]
        path = self.cache_dir / "text" / f"{key}.safetensors"
        if path.is_file():
            from safetensors.torch import load_file

            data = load_file(str(path))
            self.line("Text conditioning: cached (text encoder skipped)")
            return data["cond"], data.get("tags")
        self.line("Phase 1/4: text encoding (Qwen3-VL-32B, 50 layers)...")
        start = time.time()
        te = loader_mod.H3TextEncoder(s.text_encoder_path, self.device, emit=self.line)
        te.place(int(loader_mod.usable_vram_bytes(self.device) - 1.5 * loader_mod.GIB), emit=self.line)
        for attempt in range(12):
            failed = None
            try:
                cond, tags = te.encode(prompt)
                break
            except BaseException as exc:
                if not loader_mod.is_oom(exc):
                    raise
                failed = loader_mod.oom_note(exc)
            gc.collect()
            mm.soft_empty_cache()
            self.line(f"[VRAM] Text encoder hit the VRAM limit ({failed}); streaming more layers from RAM and retrying")
            if not loader_mod.stream_more_blocks(te.layers, int(loader_mod.GIB * 1.5 ** attempt), emit=self.line,
                                                 label="text encoder layers"):
                raise RuntimeError("Not enough VRAM for the text encoder even with every layer streamed from RAM. "
                                   "Lower Save VRAM or close other GPU apps.")
        cond = cond.detach().float().cpu()
        te.unload()
        del te
        gc.collect()
        mm.soft_empty_cache()
        from safetensors.torch import save_file

        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {"cond": cond.contiguous()}
        if tags is not None:
            tensors["tags"] = tags.detach().cpu().contiguous()
        save_file(tensors, str(path), metadata={"prompt_sha": key})
        self.line(f"Text encoding done in {time.time() - start:.1f}s ({cond.shape[1]} tokens, cached)")
        return cond, tags

    def _audio_latents(self, windows, fps: Fraction, has_audio: bool, indices: List[int]) -> List[torch.Tensor]:
        """One clean audio latent per window: encoded silence (reference) or the source soundtrack."""
        s = self.s
        use_source = str(s.audio_mode) == "source" and has_audio and not getattr(self, "_is_image", False)
        audio_vae = None
        out = []
        for start, length in windows:
            padded = planner_mod.align_frame_count(length)
            audio_t = round(padded / 24 * AUDIO_LATENTS_PER_SECOND)
            if not use_source:
                key = f"{_file_signature(s.audio_vae_path)}|silence|{padded}"
                cache = self.cache_dir / "audio" / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".safetensors")
                if cache.is_file():
                    from safetensors.torch import load_file

                    out.append(load_file(str(cache))["audio"])
                    continue
            if audio_vae is None:
                audio_vae = loader_mod.H3AudioVAE(s.audio_vae_path, self.device, emit=self.line)
            rate = audio_vae.sample_rate
            samples = round(padded / 24 * rate)  # reference: 124 frames -> 5.17 s -> 207 latents
            waveform = torch.zeros(1, 2, samples)
            if use_source:
                # the window's real-time soundtrack, stretched onto H3's 24 fps clock so it stays frame-aligned
                start_s = indices[0] / float(self._source_fps) + start / float(fps)
                clip = video_io.extract_audio(s.input_path, rate, start_s=start_s, duration_s=length / float(fps))
                if clip is not None and clip.shape[-1] > 1:
                    want = round(length / 24 * rate)
                    if clip.shape[-1] != want:
                        clip = F.interpolate(clip, size=want, mode="linear", align_corners=False)
                    n = min(samples, clip.shape[-1])
                    waveform[..., :n] = clip[..., :n]
            sound = audio_vae.encode(waveform)
            if sound.shape[-1] < audio_t:
                sound = torch.cat([sound, sound[..., -1:].expand(*sound.shape[:-1], audio_t - sound.shape[-1])], dim=-1)
            sound = sound[..., :audio_t].contiguous()
            if not use_source:
                from safetensors.torch import save_file

                cache.parent.mkdir(parents=True, exist_ok=True)
                save_file({"audio": sound}, str(cache))
            out.append(sound)
        if audio_vae is not None:
            del audio_vae
            gc.collect()
            mm.soft_empty_cache()
        return out

    # ------------------------------------------------------------- DiT
    def _load_dit(self, result: dict) -> None:
        try:
            s = self.s
            model, config, dtype, quantized = loader_mod.load_dit(s.dit_path, self.device, emit=self.line)
            result.update(model=model, config=config, dtype=dtype, quantized=quantized)
            if s.lora_path and float(s.lora_strength) != 0.0:
                result["lora_pairs_ready"] = True
        except BaseException as exc:  # surfaced on the main thread
            result["error"] = exc

    def _prepare_dit(self, loaded: dict, max_tokens: int) -> Tuple[nn.Module, torch.dtype, dict, Optional[object]]:
        s = self.s
        model = loaded["model"]
        dtype = loaded["dtype"]
        activation = int(max_tokens * float(s.bytes_per_token))
        usable = loader_mod.usable_vram_bytes(self.device)
        budget = int(usable - activation)
        self.line(f"VRAM plan: {usable / loader_mod.GIB:.1f} GB usable, {activation / loader_mod.GIB:.1f} GB kept for "
                  f"activations ({max_tokens} tokens), up to {int(s.max_gpu_blocks)} DiT blocks on GPU")
        loader_mod.place_blocks(model, list(model.blocks), self.device, budget, emit=self.line, label="DiT blocks",
                                max_resident=int(s.max_gpu_blocks))
        self.line("Applying Open-H3 merged LoRA (refiner + LightX2V Turbo + LMS Sharp)...")
        t0 = time.time()
        lora_mod.apply_merged_lora(model, s.lora_path, float(s.lora_strength), self.device, dtype, emit=self.line)
        self.line(f"LoRA merged in {time.time() - t0:.1f}s")
        if s.curve_path and float(s.curve_strength) != 0.0:
            table = lora_mod.read_table(s.table_path)
            lora_mod.apply_sharp_curves(model, s.curve_path, table, float(s.curve_strength), emit=self.line)
        else:
            self.line("Sharp curve branches: off (not the reference recipe)")
        if int(s.ffn_chunk_tokens) > 0:
            _ChunkedMLP(int(s.ffn_chunk_tokens)).install(model)
        options: dict = {}
        patch = None
        if s.sparse_attention:
            if sparse_mod.sol_attn_available(self.device):
                options, patch = sparse_mod.build_transformer_options(
                    model, tau=float(s.sparse_tau), start_percent=float(s.sparse_start_percent),
                    end_percent=float(s.sparse_end_percent),
                    dense_blocks=sparse_mod.parse_block_list(s.dense_blocks),
                    min_tokens=int(s.sparse_min_tokens), extra_tokens=int(s.sparse_extra_tokens),
                    sink_conditioning=str(s.sink_conditioning), shift=12.0)
                self.line(f"Sparse attention (Sol-Attn): on, tau {float(s.sparse_tau):g}, dense blocks {s.dense_blocks or 'none'}")
            else:
                self.line("Sparse attention (Sol-Attn): no compiled kernel for this GPU - dense attention")
        else:
            self.line("Sparse attention: off (dense attention)")
        return model, dtype, options, patch

    @torch.no_grad()
    def _refine(self, model, dtype, base_options, patch, context, tags, video0: torch.Tensor,
                audio0: torch.Tensor, seed: int, step_cb, start_sigma=None) -> torch.Tensor:
        s = self.s
        device = self.device
        start = float(s.start_sigma) if start_sigma is None else float(start_sigma)
        sigmas = torch.tensor(core.refine_sigmas(int(s.steps), start), dtype=torch.float32)
        if patch is not None and int(s.sparse_dense_steps) >= 0:
            # Step-based window (Open-H3 sol_engine presets): the first N steps dense, every later step sparse.
            k = int(s.sparse_dense_steps)
            patch.sigma_end = 0.0
            if k <= 0:
                patch.sigma_start = 1.0
            elif k < len(sigmas) - 1:
                patch.sigma_start = float(sigmas[k - 1] + sigmas[k]) / 2.0
            else:
                patch.sigma_start = -1.0
        generator = torch.manual_seed(int(seed))  # comfy.sample.prepare_noise: video stream first
        noise = torch.randn(video0.size(), dtype=torch.float32, generator=generator, device="cpu")
        x0 = video0.to(device, torch.float32)
        x = float(sigmas[0]) * noise.to(device) + (1.0 - float(sigmas[0])) * x0  # CONST.noise_scaling
        del noise
        audio = audio0.to(device, dtype)
        audio_t = audio0.shape[-1]
        lat_t, lat_h, lat_w = x0.shape[2], x0.shape[3], x0.shape[4]
        layout = PackedLayout(context.shape[1], lat_t, (lat_h + 1) // 2 * 2, (lat_w + 1) // 2 * 2, audio_t)
        payload = {"audio_scale": 1.0, "seed": int(seed), "layout": layout}
        if tags is not None:
            payload["text_token_tags"] = tags
        audio_mask = torch.zeros((1, 1, 2, audio_t), dtype=dtype, device=device)
        ctx = context.to(device, dtype)
        sample_sigmas = sigmas.to(device)
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i].to(device)
            options = dict(base_options)
            options["sigmas"] = sigma.view(1)
            options["sample_sigmas"] = sample_sigmas
            timestep = (sigma * 1000.0).view(1).float()
            out = model([x.to(dtype), audio], timestep, ctx, transformer_options=options,
                        minimax_payload=payload, denoise_mask=None, audio_denoise_mask=audio_mask)
            v = out[0].float()
            del out
            denoised = x - v * sigma          # CONST.calculate_denoised
            d = (x - denoised) / sigma        # k-diffusion euler
            x = x + d * (sigmas[i + 1].to(device) - sigma)
            del v, denoised, d
            step_cb(i)
        if patch is not None:
            patch.reset()
        return x.float().cpu()

    # -------------------------------------------------------------- plan
    def plan(self):
        s = self.s
        self._image = None
        indices, fps, src_w, src_h, has_audio, is_image = self._timeline()
        self._is_image = is_image
        self._source_fps = None if is_image else video_io.probe_video(s.input_path, count=False).fps
        target_w = int(s.target_width) or src_w * 2
        target_h = int(s.target_height) or src_h * 2
        geometry = planner_mod.plan_geometry(src_w, src_h, target_w, target_h, s.draft_size)
        if str(s.token_budget_mode) == "manual":
            budget = int(s.max_tokens)
        else:
            free = loader_mod.device_free_bytes(self.device)
            budget = planner_mod.auto_token_budget(free, float(s.reserve_vram_gb), float(s.bytes_per_token))
        window = planner_mod.frames_for_token_budget(budget, geometry, int(s.max_window_frames))
        if is_image:
            plan = planner_mod.WindowPlan(5, 0, 1, [(0, 1)])
        else:
            plan = planner_mod.plan_windows(len(indices), window, int(s.overlap_frames))
        max_tokens = max(planner_mod.window_tokens(length, geometry) for _, length in plan.windows)
        lines = [
            geometry.describe(),
            f"input {len(indices)} frame(s) on the output timeline @ {float(fps):.6g} fps"
            + (" (audio present)" if has_audio else "") + (" (image: 5-frame still)" if is_image else ""),
            f"{plan.summary()} | up to {max_tokens} tokens per window (budget {budget})",
        ]
        return indices, fps, has_audio, is_image, geometry, plan, max_tokens, lines

    # -------------------------------------------------------------- run
    def run(self) -> str:
        result = run_many([self], self.progress)[0]
        if result[1] is not None:
            raise result[1]
        return result[0]

    def encode_windows(self, vae, upscaler, plan, geometry, indices) -> List[torch.Tensor]:
        """Phase 2 for this input: streamed windows -> draft -> VAE encode -> latent upscale (CPU latents)."""
        upscaled: List[torch.Tensor] = []
        source = self._iter_source(indices)
        buffer: List[np.ndarray] = []
        pos = 0
        for w_index, (start, length) in enumerate(plan.windows):
            while pos < start + length:
                frame = next(source, None)
                if frame is None:
                    break
                buffer.append(frame)
                pos += 1
            offset = start - (pos - len(buffer))
            frames = buffer[offset:offset + length]
            if not frames:
                raise RuntimeError("Ran out of input frames while building windows.")
            latent = vae.encode(self._draft(frames, geometry))
            for pw, ph in geometry.passes:
                latent = upscaler.upscale(latent, ph // 16, pw // 16)
            upscaled.append(latent)
            keep = plan.overlap_frames
            buffer = buffer[-keep:] if keep else []
            self.line(f"Window {w_index + 1}/{len(plan.windows)} encoded: latent {tuple(latent.shape)}")
        return upscaled

    def _save_latents(self, refined: List[torch.Tensor], plan, fps) -> Optional[str]:
        """The refined result is written before decoding, so a failed decode never costs the refine."""
        try:
            from safetensors.torch import save_file

            path = Path(self.s.output_path).with_suffix(".open_h3_latents.safetensors")
            tensors = {f"window_{i:04d}": t.contiguous() for i, t in enumerate(refined)}
            meta = {"format": "open_h3_refined_latent_v1",
                    "windows": json.dumps(plan.windows), "fps": str(fps)}
            save_file(tensors, str(path), metadata=meta)
            self.line(f"Refined latents saved before decoding: {path.name}")
            return str(path)
        except Exception as exc:
            self.line(f"[WARN] Could not save refined latents: {exc}")
            return None

    def _finish(self, decoded: torch.Tensor, count: int, geometry: planner_mod.Geometry) -> torch.Tensor:
        """Decoder buffer [1, 3, F, grid_h, grid_w] -> uint8 [count, target_h, target_w, 3] (resized on the GPU)."""
        resize = (decoded.shape[-1], decoded.shape[-2]) != (geometry.target_w, geometry.target_h)
        out = []
        for i in range(0, min(count, decoded.shape[2]), 8):
            part = decoded[0, :, i:min(i + 8, count)].to(self.device).float().movedim(1, 0)  # [n, 3, H, W]
            if resize:
                part = F.interpolate(part, size=(geometry.target_h, geometry.target_w), mode="bicubic",
                                     align_corners=False, antialias=True)
            out.append(video_io.to_uint8(part.permute(0, 2, 3, 1)).cpu())
        return torch.cat(out)

    def _decode_and_write(self, vae, refined, plan, geometry, fps, has_audio, indices, is_image) -> str:
        s = self.s
        if is_image:
            frames = self._finish(vae.decode(refined[0]), 1, geometry)
            image = frames[0].numpy()
            out_path = str(Path(s.output_path).with_suffix(Path(s.output_path).suffix or ".png"))
            video_io.write_image(out_path, image)
            self.line(f"Image written: {Path(out_path).name} ({image.shape[1]}x{image.shape[0]})")
            return out_path
        audio_source = s.input_path if (s.keep_audio and has_audio and str(s.fps_mode) != "24") else None
        audio_offset = 0.0
        if audio_source and indices and self._source_fps:
            audio_offset = indices[0] / float(self._source_fps)
        writer = video_io.FfmpegWriter(s.output_path, geometry.target_w, geometry.target_h, fps,
                                       audio_source=audio_source, codec=s.codec, crf=int(s.crf),
                                       preset=s.preset, pixel_format=s.pixel_format, audio_offset_s=audio_offset,
                                       video_args=s.video_args)
        merger = video_io.StreamingChunkMerger(writer, plan.overlap_frames, plan.total_frames)
        ok = False
        try:
            for w_index, ((start, length), latent) in enumerate(zip(plan.windows, refined)):
                frames = self._finish(vae.decode(latent), length, geometry)
                merger.add_chunk(frames, is_last=(w_index == len(plan.windows) - 1))
                del frames
                gc.collect()
                self.line(f"Window {w_index + 1}/{len(plan.windows)} decoded")
            writer.close()
            ok = True
        finally:
            if not ok:
                writer.abort()
                try:
                    Path(s.output_path).unlink(missing_ok=True)
                except Exception:
                    pass
        if writer.frames_written != plan.total_frames:
            raise RuntimeError(f"Open-H3 writer produced {writer.frames_written}/{plan.total_frames} frames")
        return s.output_path


STAGES = ("text", "encode", "refine", "decode")


class _StageStore:
    """Hands each stage's results to the next one: in memory, or through work_dir files when
    every stage runs in its own process (nothing of a finished stage stays in RAM or VRAM)."""

    def __init__(self, work_dir: Optional[str]):
        self.dir = Path(work_dir) if work_dir else None
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)

    def errors(self) -> Dict[int, str]:
        if self.dir is None or not (self.dir / "errors.json").is_file():
            return {}
        return {int(k): v for k, v in json.loads((self.dir / "errors.json").read_text(encoding="utf-8")).items()}

    def add_error(self, index: int, message: str) -> None:
        if self.dir is None:
            return
        errors = self.errors()
        errors[index] = message
        (self.dir / "errors.json").write_text(json.dumps(errors), encoding="utf-8")

    def save_contexts(self, contexts: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]) -> None:
        if self.dir is None:
            return
        from safetensors.torch import save_file

        keys = list(contexts)
        tensors = {}
        for n, key in enumerate(keys):
            cond, tags = contexts[key]
            tensors[f"cond_{n}"] = cond.contiguous()
            if tags is not None:
                tensors[f"tags_{n}"] = tags.contiguous()
        save_file(tensors, str(self.dir / "contexts.safetensors"))
        (self.dir / "contexts.json").write_text(json.dumps(keys), encoding="utf-8")

    def load_contexts(self) -> Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        from safetensors.torch import load_file

        keys = json.loads((self.dir / "contexts.json").read_text(encoding="utf-8"))
        data = load_file(str(self.dir / "contexts.safetensors"))
        return {key: (data[f"cond_{n}"], data.get(f"tags_{n}")) for n, key in enumerate(keys)}

    def save_encoded(self, index: int, latents: List[torch.Tensor], audio: List[torch.Tensor]) -> None:
        if self.dir is None:
            return
        from safetensors.torch import save_file

        tensors = {f"latent_{w}": t.contiguous() for w, t in enumerate(latents)}
        tensors.update({f"audio_{w}": t.contiguous() for w, t in enumerate(audio)})
        save_file(tensors, str(self.dir / f"encoded_{index}.safetensors"), metadata={"windows": str(len(latents))})

    def load_encoded(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        from safetensors import safe_open
        from safetensors.torch import load_file

        path = str(self.dir / f"encoded_{index}.safetensors")
        with safe_open(path, framework="pt") as handle:
            windows = int(handle.metadata()["windows"])
        data = load_file(path)
        return [data[f"latent_{w}"] for w in range(windows)], [data[f"audio_{w}"] for w in range(windows)]

    def drop(self, name: str) -> None:
        if self.dir is not None:
            (self.dir / name).unlink(missing_ok=True)


def _refined_latent_path(p: "OpenH3Pipeline") -> Path:
    return Path(p.s.output_path).with_suffix(".open_h3_latents.safetensors")


def _load_refined(p: "OpenH3Pipeline", windows: int) -> List[torch.Tensor]:
    from safetensors.torch import load_file

    data = load_file(str(_refined_latent_path(p)))
    return [data[f"window_{w:04d}"] for w in range(windows)]


def run_many(pipelines: List["OpenH3Pipeline"], progress: Optional[OpenH3Progress] = None,
             stages: Optional[Tuple[str, ...]] = None, work_dir: Optional[str] = None):
    """Run one or more inputs with every model loaded once (text encoder -> VAE + upscaler -> DiT -> decoder).

    stages/work_dir run a subset of the four stages, reading and writing the hand-over files in
    work_dir (each stage in its own process). Returns [(output_path or None, exception or None)]
    in input order; one failing input never stops the others."""
    progress = progress or OpenH3Progress()
    line = progress.line
    torch.set_grad_enabled(False)
    started = time.time()
    stages = tuple(stages or STAGES)
    store = _StageStore(work_dir)
    lead = pipelines[0]
    s0 = lead.s
    device = lead.device
    count = len(pipelines)
    results: List[list] = [[None, None] for _ in pipelines]
    loader_mod.apply_vram_cap(device, float(s0.reserve_vram_gb), emit=line)
    loader_mod.guard_attention_oom()

    def fail(index: int, exc: BaseException) -> None:
        results[index][1] = exc
        store.add_error(index, str(exc))
        line(f"[OpenH3] ITEM_ERROR {index}: {exc}")

    earlier = store.errors()
    jobs = []
    for i, p in enumerate(pipelines):
        if i in earlier:
            results[i][1] = RuntimeError(earlier[i])
            continue
        tag = f"[{i + 1}/{count}] " if count > 1 else ""
        try:
            plan_info = p.plan()
            if stages[0] == "text":
                for text in plan_info[-1]:
                    line(f"[Plan] {tag}{text}")
            jobs.append((i, p, plan_info))
        except Exception as exc:
            fail(i, exc)
    if not jobs:
        return [tuple(r) for r in results]

    # 1) text conditioning, one per distinct prompt
    contexts: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
    if "text" in stages:
        for _i, p, _plan in jobs:
            key = str(p.s.prompt or "")
            if key not in contexts:
                contexts[key] = p._text_condition()
        store.save_contexts(contexts)
    elif "refine" in stages:
        contexts = store.load_contexts()

    # 2) draft encode + latent upscale, every input
    encoded: Dict[int, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
    vae = None
    if "encode" in stages:
        total_windows = sum(len(info[5].windows) for _i, _p, info in jobs)
        line(f"Phase 2/4: VAE encode + latent upscale ({total_windows} window(s)"
             f"{f', {len(jobs)} inputs' if count > 1 else ''})...")
        t0 = time.time()
        vae = loader_mod.H3VideoVAE(s0.vae_path, device, emit=line)
        needs_upscaler = any(info[4].passes for _i, _p, info in jobs)
        upscaler = loader_mod.H3LatentUpscaler(s0.upscaler_path, device, emit=line) if needs_upscaler else None
        for i, p, info in jobs:
            indices, fps, has_audio, is_image, geometry, plan, _max_tokens, _lines = info
            try:
                latents = p.encode_windows(vae, upscaler, plan, geometry, indices)
                audio = p._audio_latents(plan.windows, fps, has_audio, indices)
                encoded[i] = (latents, audio)
                store.save_encoded(i, latents, audio)
            except Exception as exc:
                fail(i, exc)
        vae.offload()
        if upscaler is not None:
            upscaler.offload()
            del upscaler
        if "decode" not in stages:
            del vae
            vae = None
        gc.collect()
        mm.soft_empty_cache()
        line(f"Encode + upscale done in {time.time() - t0:.1f}s")
        jobs = [job for job in jobs if job[0] in encoded]
    elif "refine" in stages:
        for i, _p, _info in jobs:
            encoded[i] = store.load_encoded(i)
    if not jobs:
        return [tuple(r) for r in results]

    # 3) refine: one DiT load for every input
    refined: Dict[int, Tuple[List[torch.Tensor], Optional[str]]] = {}
    if "refine" in stages:
        line("Phase 3/4: loading MiniMax H3 FL2VA + Open-H3 adapters...")
        loaded: dict = {}
        lead._load_dit(loaded)
        if "error" in loaded:
            raise loaded["error"]
        max_tokens = max(info[6] for _i, _p, info in jobs)
        model, dtype, options, patch = lead._prepare_dit(loaded, max_tokens)
        blocks = list(model.blocks)
        ctx_dev = {}
        with torch.no_grad():
            for key, (context, tags) in contexts.items():
                ctx_dev[key] = (model.preprocess_text_embeds(context.to(device, dtype)), tags)
        def _steps_per_window(ps):
            n = len(core.refine_sigmas(int(ps.steps), float(ps.start_sigma))) - 1
            if float(ps.second_pass_sigma) > 0:
                n += len(core.refine_sigmas(int(ps.steps), float(ps.second_pass_sigma))) - 1
            return n

        total_steps = sum(_steps_per_window(p.s) * len(info[5].windows) for _i, p, info in jobs)
        done = [0]
        t0 = time.time()
        for i, p, info in jobs:
            plan = info[5]
            latents, audio = encoded.pop(i)
            context_dev, tags = ctx_dev[str(p.s.prompt or "")]
            steps = _steps_per_window(p.s)
            outs = []
            try:
                for w_index, latent in enumerate(latents):
                    w_start = time.time()
                    done_before = done[0]

                    def _cb(step, _w=w_index, _n=len(latents)):
                        done[0] += 1
                        where = f"input {i + 1}/{count} " if count > 1 else ""
                        line(f"Processing: {done[0]}/{total_steps} | {where}window {_w + 1}/{_n} step {step + 1}/{steps}")

                    out = None
                    for attempt in range(40):
                        weights_now = 0
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats(device)
                            weights_now = torch.cuda.memory_allocated(device)
                        failed = None
                        try:
                            out = p._refine(model, dtype, options, patch, context_dev, tags, latent, audio[w_index],
                                            int(p.s.seed) + w_index, _cb)
                            if float(p.s.second_pass_sigma) > 0:
                                # same size again from a lower noise level (Open-H3 4K ladder: 0.7, then 0.6)
                                if patch is not None:
                                    patch.reset()
                                first_steps = len(core.refine_sigmas(int(p.s.steps), float(p.s.start_sigma))) - 1
                                out = p._refine(model, dtype, options, patch, context_dev, tags, out, audio[w_index],
                                                int(p.s.seed) + w_index, lambda st: _cb(first_steps + st),
                                                start_sigma=float(p.s.second_pass_sigma))
                            break
                        except BaseException as exc:
                            if not loader_mod.is_oom(exc):
                                raise
                            failed = loader_mod.oom_note(exc)
                            shortfall = loader_mod.oom_shortfall(exc, device)
                        # auto-fix: free a little more VRAM by streaming more blocks, then redo this window
                        if patch is not None:
                            patch.reset()
                        done[0] = done_before
                        gc.collect()
                        mm.soft_empty_cache()
                        line(f"[VRAM] Window {w_index + 1} reached the VRAM limit ({failed}); "
                             f"streaming more DiT blocks from RAM and retrying - no shared GPU memory is used")
                        if not loader_mod.stream_more_blocks(blocks, max(shortfall, int(0.5 * loader_mod.GIB)),
                                                             emit=line):
                            raise RuntimeError(
                                "Not enough VRAM for this window even with every DiT block streamed from RAM. "
                                "Lower Max Window Frames (or use a manual token budget), lower Save VRAM, "
                                "or close other GPU apps.")
                    outs.append(out)
                    if torch.cuda.is_available() and w_index + 1 < len(latents):
                        # spare VRAM left by this window: bring streamed blocks back for the next one
                        cap = loader_mod.vram_cap_bytes(device)
                        spare = int(cap - torch.cuda.max_memory_reserved(device) - 0.75 * loader_mod.GIB)
                        if spare > 0:
                            loader_mod.restore_blocks(blocks, device, spare, emit=line)
                    peak = ""
                    if torch.cuda.is_available():
                        top = torch.cuda.max_memory_allocated(device)
                        tokens = planner_mod.window_tokens(plan.windows[w_index][1], info[4])
                        peak = (f" | peak VRAM {top / loader_mod.GIB:.1f}GB, activations "
                                f"{(top - weights_now) / loader_mod.GIB:.1f}GB "
                                f"({(top - weights_now) / max(1, tokens) / 1000:.0f} KB/token)")
                    line(f"Window {w_index + 1}/{len(latents)} refined in {time.time() - w_start:.1f}s{peak}")
                refined[i] = (outs, p._save_latents(outs, plan, info[1]))
            except Exception as exc:
                fail(i, exc)
        line(f"Refine done in {time.time() - t0:.1f}s")
        del model, loaded, options, patch, ctx_dev, blocks
        gc.collect()
        mm.soft_empty_cache()
        jobs = [job for job in jobs if job[0] in refined]
        for i in range(count):
            store.drop(f"encoded_{i}.safetensors")
    elif "decode" in stages:
        for i, p, info in jobs:
            path = _refined_latent_path(p)
            if path.is_file():
                refined[i] = (_load_refined(p, len(info[5].windows)), str(path))
            else:
                fail(i, RuntimeError(f"Refined latents missing: {path.name}"))

    # 4) decode + write every input
    if "decode" in stages:
        line("Phase 4/4: VAE decode + write...")
        t0 = time.time()
        decode_path = s0.decode_vae_path or s0.vae_path
        same_vae = os.path.normcase(os.path.abspath(decode_path)) == os.path.normcase(os.path.abspath(s0.vae_path))
        if vae is None or not same_vae:
            if vae is not None:
                vae.offload()
                del vae
                gc.collect()
            vae = loader_mod.H3VideoVAE(decode_path, device, emit=line)
        for i, p, info in jobs:
            if i not in refined:
                continue
            indices, fps, has_audio, is_image, geometry, plan, _max_tokens, _lines = info
            outs, latent_file = refined.pop(i)
            try:
                output = p._decode_and_write(vae, outs, plan, geometry, fps, has_audio, indices, is_image)
                results[i][0] = output
                if latent_file and not p.s.keep_latents:
                    Path(latent_file).unlink(missing_ok=True)
                if count > 1:
                    line(f"[OpenH3] ITEM_DONE {i} OUTPUT_PATH: {output}")
            except Exception as exc:
                fail(i, exc)
        vae.offload()
        line(f"Decode + write done in {time.time() - t0:.1f}s")
        ok = sum(1 for r in results if r[0])
        line(f"Open-H3 regenerate complete -> {ok}/{count} output(s) | total {time.time() - started:.1f}s"
             if count > 1 else f"Open-H3 regenerate complete -> {results[0][0]} | total {time.time() - started:.1f}s")
    else:
        line(f"Stage {'+'.join(stages)} done in {time.time() - started:.1f}s")
        for i, _p, _info in jobs:
            results[i][0] = "__stage_ok__"
    return [tuple(r) for r in results]


def plan_only(settings: OpenH3Settings) -> List[str]:
    """Geometry + window plan without loading any model."""
    pipeline = OpenH3Pipeline(settings)
    return pipeline.plan()[-1]
