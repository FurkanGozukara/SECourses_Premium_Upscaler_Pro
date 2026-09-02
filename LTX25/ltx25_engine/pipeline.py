"""
End-to-end LTX 2.5 pixel spatial 2x upscale pipeline (ComfyUI-parity).

Per chunk:
  pad -> IC-LoRA guide (tiled VAE encode + dilate + keyframe coords)
       -> empty gen latent + frozen zero audio latent
       -> flow sampling (per-token timesteps, masked pinning, CFG)
       -> crop guides -> tiled VAE decode -> crop padding -> merge/write.
"""

from __future__ import annotations

import gc
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import torch

import comfy.model_management as mm
import comfy.utils
from comfy.ldm.lightricks.symmetric_patchifier import SymmetricPatchifier, AudioPatchifier

from . import guide as guide_mod
from . import loader as loader_mod
from . import planner as planner_mod
from . import sampling as sampling_mod
from .vae import Ltx25VideoVae
from .video_io import FfmpegWriter, StreamingChunkMerger, read_video_frames

log = logging.getLogger("ltx25.pipeline")

_V_PATCHIFIER = SymmetricPatchifier(1, start_end=True)
_A_PATCHIFIER = AudioPatchifier(1, start_end=True)


@dataclass
class Ltx25Settings:
    input_path: str = ""
    output_path: str = ""
    transformer_path: str = ""
    text_encoder_path: str = ""
    video_vae_path: str = ""
    audio_vae_path: str = ""
    ic_lora_path: str = ""

    positive_prompt: str = ""
    negative_prompt: str = ""
    seed: int = 42
    steps: int = 8
    cfg: float = 1.0
    sampler: str = "euler_ancestral"
    schedule: str = "distilled"          # "distilled" | "dev"
    ic_lora_strength: float = 1.0
    guide_strength: float = 1.0

    # Chunk planning (ComfyUI LTX25UpscaleControls parity)
    token_budget_mode: str = "auto"       # "auto" (single-chunk preferred) | "manual"
    max_latent_tokens: int = 18000
    max_chunk_frames: int = 121
    overlap_frames: int = 1

    # VAE tiling (workflow defaults)
    encode_tile_size: int = 512
    encode_tile_overlap: int = 64
    decode_tile_size: int = 512
    decode_tile_overlap: int = 64
    decode_temporal_size: int = 128
    decode_temporal_overlap: int = 32

    # Input handling
    pre_resize_longer_edge: int = 0       # 0 = off (native 2x)
    fps_override: float = 0.0             # 0 = keep source fps
    start_frame: int = 0
    end_frame: int = -1                   # -1 = all

    # Output encode (final app-level re-encode may follow)
    codec: str = "libx264"
    crf: int = 15
    preset: str = "medium"
    pixel_format: str = "yuv420p"
    keep_audio: bool = True

    # VRAM
    reserve_vram_gb: float = 1.0
    bytes_per_token: float = 1_400_000.0

    dev_max_shift: float = 2.05
    dev_base_shift: float = 0.95
    dev_terminal: float = 0.1


@dataclass
class Ltx25Progress:
    on_line: Callable[[str], None] = field(default=lambda s: None)

    def line(self, text: str) -> None:
        try:
            self.on_line(text)
        except Exception:
            pass


class Ltx25Pipeline:
    def __init__(self, settings: Ltx25Settings, progress: Optional[Ltx25Progress] = None):
        self.s = settings
        self.progress = progress or Ltx25Progress()
        self.device = mm.get_torch_device()
        self.dtype = torch.bfloat16
        self.model = None
        self.vae: Optional[Ltx25VideoVae] = None
        self.audio_constants = loader_mod.read_audio_latent_constants(settings.audio_vae_path or None)
        self._ctx_pos = None
        self._ctx_neg = None

    # ------------------------------------------------------------------ text
    def encode_prompts(self) -> None:
        s = self.s
        self.progress.line("Phase 1/4: text encoding (Gemma 4 12B)...")
        start = time.time()
        te = loader_mod.Ltx25TextEncoder(
            s.text_encoder_path, self.device, self.dtype, on_progress=self.progress.line
        )
        te.to_gpu()
        raw_pos = te.encode(s.positive_prompt or "")
        raw_neg = None
        if float(s.cfg) != 1.0:
            raw_neg = te.encode(s.negative_prompt or "")
        te.unload()
        del te
        gc.collect()
        mm.soft_empty_cache()
        self._raw_ctx_pos = raw_pos
        self._raw_ctx_neg = raw_neg
        self.progress.line(f"Text encoding done in {time.time() - start:.1f}s")

    # ----------------------------------------------------------------- model
    def load_models(self, plan: planner_mod.ChunkPlan) -> None:
        s = self.s
        self.progress.line("Phase 2/4: loading LTX 2.5 transformer...")
        start = time.time()
        model, quant_config, _config = loader_mod.load_transformer(
            s.transformer_path, self.device, self.dtype, on_progress=self.progress.line
        )
        loader_mod.apply_ic_lora(
            model, s.ic_lora_path, strength=float(s.ic_lora_strength), on_progress=self.progress.line
        )

        # Activation budget: total sequence = video*1.25 + text + audio slack.
        seq_tokens = plan.effective_tokens * 1.25 + 2048
        activation_budget = int(seq_tokens * float(s.bytes_per_token))
        free_vram = mm.get_free_memory(self.device)
        loader_mod.place_transformer(
            model,
            self.device,
            free_vram_bytes=int(free_vram),
            activation_budget_bytes=activation_budget,
            reserve_bytes=int(float(s.reserve_vram_gb) * (1024 ** 3)),
            on_progress=self.progress.line,
        )
        self.model = model

        self.vae = Ltx25VideoVae(s.video_vae_path, self.device, self.dtype)
        self.progress.line(f"Models ready in {time.time() - start:.1f}s")

        # Run the embedding connectors once per prompt (ComfyUI does this in
        # extra_conds via preprocess_text_embeds with unprocessed=True).
        with torch.no_grad():
            self._ctx_pos = self.model.preprocess_text_embeds(
                self._raw_ctx_pos.to(device=self.device, dtype=self.dtype), unprocessed=True
            ).float()
            if self._raw_ctx_neg is not None:
                self._ctx_neg = self.model.preprocess_text_embeds(
                    self._raw_ctx_neg.to(device=self.device, dtype=self.dtype), unprocessed=True
                ).float()

    # -------------------------------------------------------------- sampling
    def _make_apply_fn(
        self,
        bundle: guide_mod.GuideBundle,
        audio_latent: torch.Tensor,
        audio_mask: torch.Tensor,
        frame_rate: float,
        cfg: float,
    ):
        model = self.model
        device = self.device
        dtype = self.dtype
        v_mask = bundle.noise_mask.to(device)
        a_mask = audio_mask.to(device)
        ax = audio_latent.to(device=device, dtype=dtype)
        kf = bundle.keyframe_idxs.to(device)
        entries = bundle.guide_attention_entries
        ctx_pos = self._ctx_pos.to(device=device, dtype=dtype)
        ctx_neg = self._ctx_neg.to(device=device, dtype=dtype) if self._ctx_neg is not None else None

        def _timesteps(sigma: float):
            v_t = _V_PATCHIFIER.patchify((v_mask * sigma)[:, :1])[0]
            a_t = _A_PATCHIFIER.patchify((a_mask * sigma)[:, :1, :, :1])[0]
            return v_t, a_t

        def _run(vx: torch.Tensor, sigma: float, context: torch.Tensor) -> torch.Tensor:
            v_t, a_t = _timesteps(sigma)
            out = model(
                [vx.to(dtype), ax.clone()],
                (v_t, a_t),
                context=context,
                attention_mask=None,
                frame_rate=float(frame_rate),
                transformer_options={},
                keyframe_idxs=kf,
                denoise_mask=v_mask,
                guide_attention_entries=entries,
            )
            v_out = out[0] if isinstance(out, (list, tuple)) else out
            return v_out.float()

        def apply_fn(x: torch.Tensor, sigma: float) -> torch.Tensor:
            with torch.no_grad():
                cond_out = _run(x, sigma, ctx_pos)
                denoised = x - cond_out * sigma
                if ctx_neg is not None and not math.isclose(float(cfg), 1.0):
                    uncond_out = _run(x, sigma, ctx_neg)
                    denoised_u = x - uncond_out * sigma
                    denoised = denoised_u + (denoised - denoised_u) * float(cfg)
                return denoised

        return apply_fn

    def _sigmas(self, tokens_with_guides: int) -> torch.Tensor:
        s = self.s
        if str(s.schedule).lower() == "dev":
            return sampling_mod.ltxv_scheduler_sigmas(
                int(s.steps), int(tokens_with_guides),
                max_shift=float(s.dev_max_shift), base_shift=float(s.dev_base_shift),
                stretch=True, terminal=float(s.dev_terminal),
            )
        return sampling_mod.distilled_sigmas(int(s.steps))

    # ------------------------------------------------------------------- run
    def run(self, cancel_check: Optional[Callable[[], bool]] = None) -> str:
        s = self.s
        total_start = time.time()
        cancel_check = cancel_check or (lambda: False)

        frames_u8, src_fps, has_audio = read_video_frames(s.input_path)
        fps = float(s.fps_override) if float(s.fps_override or 0) > 0 else src_fps
        start_f = max(0, int(s.start_frame))
        end_f = int(s.end_frame)
        if end_f is not None and end_f >= 0:
            frames_u8 = frames_u8[start_f:end_f + 1]
        elif start_f:
            frames_u8 = frames_u8[start_f:]
        if frames_u8.shape[0] == 0:
            raise RuntimeError("No frames selected from the input video.")

        if int(s.pre_resize_longer_edge or 0) > 0:
            frames_u8 = self._pre_resize(frames_u8, int(s.pre_resize_longer_edge))

        total_frames, height, width = frames_u8.shape[0], frames_u8.shape[1], frames_u8.shape[2]
        self.progress.line(
            f"Input: {width}x{height} x {total_frames} frames @ {fps:.3f} fps"
            + (" (audio present)" if has_audio else " (no audio)")
        )

        # --- plan chunks -------------------------------------------------
        if str(s.token_budget_mode).lower() == "auto":
            weight_bytes_hint = 0  # weights stream; activation budget dominates
            free_vram = mm.get_free_memory(self.device)
            budget, reason = planner_mod.auto_token_budget(
                total_frames, width, height,
                free_vram_bytes=int(free_vram),
                weight_resident_bytes=weight_bytes_hint,
                reserve_gb=float(s.reserve_vram_gb),
                bytes_per_token=float(s.bytes_per_token),
            )
            self.progress.line(f"Auto VRAM token budget: {budget} ({reason})")
        else:
            budget = int(s.max_latent_tokens)
        plan = planner_mod.plan_chunks(
            total_frames, width, height, budget,
            max_chunk_frames=int(s.max_chunk_frames), overlap_frames=int(s.overlap_frames),
        )
        for line in plan.summary(int(s.steps)).splitlines():
            self.progress.line(f"[Plan] {line}")

        # --- prompts + models ---------------------------------------------
        self.encode_prompts()
        if cancel_check():
            raise KeyboardInterrupt()
        self.load_models(plan)

        latent_h = plan.generation_height // 32
        latent_w = plan.generation_width // 32
        latent_frames = plan.latent_frames

        audio_t = loader_mod.audio_latent_frames(
            plan.chunk_length, fps, self.audio_constants["latents_per_second"]
        )
        audio_latent = torch.zeros(
            (1, self.audio_constants["z_channels"], max(audio_t, 1), self.audio_constants["freq_bins"]),
            dtype=torch.float32,
        )
        audio_mask = torch.zeros((1, 1, audio_latent.shape[2], 1), dtype=torch.float32)

        # Start ffmpeg lazily after the first chunk has actually decoded. Starting it before
        # VAE/sampling work leaves an audio-only MP4 when inference fails early; older runner
        # code then mistook that artifact for a successful video and entered concat fallbacks.
        writer: Optional[FfmpegWriter] = None
        merger: Optional[StreamingChunkMerger] = None
        pipeline_failed = True
        writer_close_error: Optional[Exception] = None

        total_steps = plan.chunk_count * int(s.steps)
        done_steps = 0
        try:
            for chunk_index, (start, keep_length) in enumerate(plan.chunk_ranges):
                if cancel_check():
                    raise KeyboardInterrupt()
                chunk_t0 = time.time()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(self.device)
                self.progress.line(
                    f"Chunk {chunk_index + 1}/{plan.chunk_count}: frames {start}-{start + keep_length - 1}"
                    f" (window {plan.chunk_length})"
                )

                chunk_float = frames_u8[start:start + keep_length].to(torch.float32) / 255.0
                chunk_float = planner_mod.slice_chunk(
                    chunk_float, 0, keep_length, plan.chunk_length
                )
                chunk_float = planner_mod.pad_frames_to_grid(
                    chunk_float, plan.padded_width, plan.padded_height
                )

                self.progress.line("VAE encoding guide (tiled)...")
                bundle = guide_mod.build_guide(
                    self.vae, chunk_float, latent_frames, latent_h, latent_w,
                    guide_strength=float(s.guide_strength),
                    latent_downscale_factor=2.0,
                    tile_size=int(s.encode_tile_size), tile_overlap=int(s.encode_tile_overlap),
                    use_tiled_encode=True,
                )
                del chunk_float
                self.vae.to_cpu()

                sigmas = self._sigmas(
                    bundle.latent.shape[2] * latent_h * latent_w
                )
                clean = bundle.latent.to(self.device)
                noise = sampling_mod.prepare_noise(bundle.latent, int(s.seed) + chunk_index).to(self.device)
                apply_fn = self._make_apply_fn(bundle, audio_latent, audio_mask, fps, float(s.cfg))
                masked = sampling_mod.MaskedFlowModel(apply_fn, bundle.noise_mask.to(self.device), clean)

                def _step_cb(i, _chunk=chunk_index):
                    nonlocal done_steps
                    done_steps += 1
                    self.progress.line(
                        f"Processing: {done_steps}/{total_steps} | chunk {_chunk + 1}/{plan.chunk_count}"
                        f" step {i + 1}/{int(self.s.steps)}"
                    )
                    if cancel_check():
                        raise KeyboardInterrupt()

                latent_out = sampling_mod.run_sampler(
                    s.sampler, masked, noise, sigmas, seed=int(s.seed) + chunk_index, callback=_step_cb
                )
                del noise, clean, masked, apply_fn
                latent_out = guide_mod.crop_guides(latent_out, bundle.num_guide_latent_frames)
                del bundle
                mm.soft_empty_cache()

                self.progress.line("VAE decoding (tiled)...")
                pixels = self.vae.decode_tiled(
                    latent_out.float().cpu(),
                    tile_size_px=int(s.decode_tile_size),
                    overlap_px=int(s.decode_tile_overlap),
                    temporal_size_frames=int(s.decode_temporal_size),
                    temporal_overlap_frames=int(s.decode_temporal_overlap),
                )
                del latent_out
                self.vae.to_cpu()

                pixels = pixels[:plan.chunk_length, : plan.output_height, : plan.output_width, :]
                if writer is None:
                    writer = FfmpegWriter(
                        s.output_path,
                        width=plan.output_width,
                        height=plan.output_height,
                        fps=fps,
                        audio_source=(s.input_path if (s.keep_audio and has_audio) else None),
                        codec=s.codec,
                        crf=int(s.crf),
                        preset=s.preset,
                        pixel_format=s.pixel_format,
                    )
                    merger = StreamingChunkMerger(writer, plan.overlap_frames, plan.total_frames)
                assert merger is not None
                merger.add_chunk(
                    pixels, keep_length, is_last=(chunk_index == plan.chunk_count - 1)
                )
                del pixels
                gc.collect()
                mm.soft_empty_cache()
                peak_note = ""
                if torch.cuda.is_available():
                    peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                    reserved = torch.cuda.max_memory_reserved(self.device) / (1024 ** 3)
                    peak_note = f" | peak VRAM {peak:.1f}GB (reserved {reserved:.1f}GB)"
                self.progress.line(
                    f"Chunk {chunk_index + 1}/{plan.chunk_count} done in {time.time() - chunk_t0:.1f}s{peak_note}"
                )
            pipeline_failed = False
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception as exc:
                    writer_close_error = exc
                    log.warning(f"Writer close: {exc}")
            if pipeline_failed or writer_close_error is not None:
                # This path belongs to the current collision-safe run. Do not leave a partial
                # chunk that resume/concat discovery could later classify as completed.
                try:
                    Path(s.output_path).unlink(missing_ok=True)
                except Exception:
                    pass

        if writer_close_error is not None:
            raise RuntimeError(f"LTX 2.5 video writer failed to finalize: {writer_close_error}") from writer_close_error

        if writer is None or int(writer.frames_written) != int(plan.total_frames):
            written = int(writer.frames_written) if writer is not None else 0
            try:
                Path(s.output_path).unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError(
                f"LTX 2.5 writer produced {written}/{int(plan.total_frames)} required frames"
            )

        self.progress.line(
            f"LTX 2.5 upscale complete: {writer.frames_written} frames -> {s.output_path}"
            f" | total {time.time() - total_start:.1f}s"
        )
        return s.output_path

    @staticmethod
    def _pre_resize(frames_u8: torch.Tensor, longer_edge: int) -> torch.Tensor:
        _, h, w, _ = frames_u8.shape
        longest = max(h, w)
        if longest <= longer_edge:
            return frames_u8
        scale = longer_edge / longest
        new_w = max(2, int(round(w * scale / 2)) * 2)
        new_h = max(2, int(round(h * scale / 2)) * 2)
        out = comfy.utils.common_upscale(
            frames_u8.to(torch.float32).movedim(-1, 1) / 255.0, new_w, new_h, "lanczos", "disabled"
        ).movedim(1, -1)
        return (out.clamp(0, 1) * 255.0).round().to(torch.uint8)
