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
    decode_tile_size: int = 512           # 0 = auto: the largest measured tile set that fits the free VRAM
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
    bytes_per_token: float = planner_mod.BYTES_PER_TOKEN

    dev_max_shift: float = 2.05
    dev_base_shift: float = 0.95
    dev_terminal: float = 0.1

    # Options added 2026-11-01 (upstream scan); every default reproduces the earlier output.
    # Frames repeated after the last frame, generated and trimmed (the last 1-2 frames of an LTX
    # generation lose detail; Refine-Details card).
    tail_pad_frames: int = 0
    # Guide VAE encode: "tiled" (512/64 tiles), "untiled" (one pass, tiled fallback on OOM), or "auto"
    # (untiled while the guide frame is at most guide_untiled_max_pixels).
    guide_encode: str = "tiled"
    guide_untiled_max_pixels: int = 1920 * 1088
    # Spatial transformer tiling: "off", "on", or "auto" (on when the canvas exceeds the model's 2048 px
    # position range).
    spatial_tiling: str = "off"
    spatial_tile_width: int = 1536
    spatial_tile_height: int = 1024
    spatial_tile_overlap: float = 0.5
    spatial_tile_blend: str = "trapezoid"   # "trapezoid" (LTX-2 TiledDiffusionModel) | "gaussian" (tiled fusion)
    # Mode: "x2" (pixel spatial x2 IC-LoRA, half-size guide) or "refine" (a factor-1 IC-LoRA such as
    # Refine-Details or Restore on a lanczos 2x canvas; ic_lora_path is that LoRA).
    mode: str = "x2"
    # Optional 1x pre-clean pass before the x2 pass (factor-1 IC-LoRA such as Decompression or Deblur).
    preclean_lora_path: str = ""
    preclean_strength: float = 1.0
    preclean_prompt: str = ""
    preclean_negative_prompt: str = ""
    # Sol-Attn block-sparse self-attention (comfy_kitchen kernel). Measured 2026-11-01 (RTX 5090): every step
    # sparse at tau 1.2 sampled 9 % (49 f 1080p) to 14 % (241 f 720p) faster with GT metrics and crops unchanged;
    # the Sana refiner policy (first 5 steps dense, tau 1.0/1.25/1.5) gained only 0-6 %.
    sparse_attention: bool = False
    sparse_taus: str = "1.2"
    sparse_dense_steps: int = 0
    sparse_min_tokens: int = 12288


# Fixed VRAM next to the resident blocks during sampling/VAE: two streaming cast buffers (2 x 0.72 GB
# blocks) and the tiled VAE working set (measured 2026-10-30, RTX 5090; benchmarks/vram/ltx25.jsonl).
ACTIVATION_BASE_BYTES = planner_mod.ACTIVATION_BASE_BYTES


@dataclass
class Ltx25Progress:
    on_line: Callable[[str], None] = field(default=lambda s: None)

    def line(self, text: str) -> None:
        try:
            self.on_line(text)
        except Exception:
            pass


class _FrameCollector:
    """Writer stand-in for an intermediate pass (pre-clean): keeps uint8 frames in RAM."""

    def __init__(self):
        self.frames: List[torch.Tensor] = []
        self.frames_written = 0

    def write(self, frames_uint8: torch.Tensor) -> None:
        self.frames.append(frames_uint8.contiguous().clone())
        self.frames_written += frames_uint8.shape[0]

    def close(self) -> None:
        pass

    def result(self) -> torch.Tensor:
        return torch.cat(self.frames, dim=0)


@dataclass
class _PassSpec:
    """One sampling pass over the clip."""
    label: str
    scale: int                      # output size / input size (2 = x2 upscale, 1 = pre-clean)
    factor: int                     # IC-LoRA reference downscale factor (2 = half-size guide, 1 = same size)
    lanczos_guide: bool             # resize the source to the canvas with lanczos before the guide encode
    ctx_pos: torch.Tensor = None
    ctx_neg: Optional[torch.Tensor] = None
    sampler: str = "euler_ancestral"
    seed_offset: int = 0
    tiling_mode: str = "off"
    final: bool = False             # the last pass: the transformer is released before its last chunk's decode


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
        self._ctx_pre_pos = None
        self._ctx_pre_neg = None
        self._raw_ctx_pre_pos = None
        self._raw_ctx_pre_neg = None
        self._main_lora: Optional[loader_mod.LoraPatchSet] = None
        self._pre_lora: Optional[loader_mod.LoraPatchSet] = None
        self._topts: dict = {}
        self.sparse_policy = None

    # ------------------------------------------------------------------ text
    def encode_prompts(self) -> None:
        s = self.s
        self.progress.line("Phase 1/4: text encoding (Gemma 4 12B)...")
        start = time.time()
        te = loader_mod.Ltx25TextEncoder(
            s.text_encoder_path, self.device, self.dtype, on_progress=self.progress.line
        )
        # Leave room for the prompt's activations (about 2 GB for the 12B encoder) under Save VRAM.
        te_budget = loader_mod.budget_free_bytes(self.device) - int((float(s.reserve_vram_gb) + 2.0) * (1024 ** 3))
        te.to_gpu(budget_bytes=max(0, te_budget), emit=self.progress.line)
        raw_pos = te.encode(s.positive_prompt or "")
        raw_neg = None
        if float(s.cfg) != 1.0:
            raw_neg = te.encode(s.negative_prompt or "")
        if s.preclean_lora_path:
            self._raw_ctx_pre_pos = te.encode(s.preclean_prompt or "")
            if float(s.cfg) != 1.0:
                self._raw_ctx_pre_neg = te.encode(s.preclean_negative_prompt or "")
        te.unload()
        del te
        gc.collect()
        mm.soft_empty_cache()
        self._raw_ctx_pos = raw_pos
        self._raw_ctx_neg = raw_neg
        self.progress.line(f"Text encoding done in {time.time() - start:.1f}s")

    # ----------------------------------------------------------------- model
    def load_models(self, plan: planner_mod.ChunkPlan, call_tokens: Optional[int] = None,
                    guide_ratio: float = 0.25) -> None:
        s = self.s
        self.progress.line("Phase 2/4: loading LTX 2.5 transformer...")
        start = time.time()
        model, quant_config, _config = loader_mod.load_transformer(
            s.transformer_path, self.device, self.dtype, on_progress=self.progress.line
        )
        if s.preclean_lora_path:
            # Two LoRAs switched between passes: both stay runtime patches.
            self._pre_lora = loader_mod.load_lora_patches(
                model, s.preclean_lora_path, strength=float(s.preclean_strength), on_progress=self.progress.line)
            self._main_lora = loader_mod.load_lora_patches(
                model, s.ic_lora_path, strength=float(s.ic_lora_strength), on_progress=self.progress.line)
        else:
            loader_mod.apply_ic_lora(
                model, s.ic_lora_path, strength=float(s.ic_lora_strength), on_progress=self.progress.line
            )

        # Activation budget: total sequence = video*(1 + guide share) + text + audio slack. With spatial
        # tiling one call only holds one window, so the budget follows the window.
        tokens = int(call_tokens) if call_tokens else plan.effective_tokens
        seq_tokens = tokens * (1.0 + float(guide_ratio)) + 2048
        # + a fixed part for the two streaming cast buffers and the tiled VAE that runs beside the blocks.
        activation_budget = int(ACTIVATION_BASE_BYTES
                                + seq_tokens * float(s.bytes_per_token) * planner_mod.ACTIVATION_SAFETY)
        free_vram = loader_mod.budget_free_bytes(self.device)
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
            if self._raw_ctx_pre_pos is not None:
                self._ctx_pre_pos = self.model.preprocess_text_embeds(
                    self._raw_ctx_pre_pos.to(device=self.device, dtype=self.dtype), unprocessed=True
                ).float()
            if self._raw_ctx_pre_neg is not None:
                self._ctx_pre_neg = self.model.preprocess_text_embeds(
                    self._raw_ctx_pre_neg.to(device=self.device, dtype=self.dtype), unprocessed=True
                ).float()

        if s.sparse_attention:
            from . import sparse as sparse_mod

            if sparse_mod.sol_attn_available(self.device):
                taus = [float(t) for t in str(s.sparse_taus or "1.2").replace(";", ",").split(",") if t.strip()]
                self.sparse_policy = sparse_mod.SparsePolicy(
                    taus, dense_steps=int(s.sparse_dense_steps), dense_blocks=(0,),
                    min_tokens=int(s.sparse_min_tokens))
                self._topts = sparse_mod.install(self.model, self.sparse_policy)
                self.progress.line(
                    f"Sparse attention (Sol-Attn): on, block 0 dense, first {int(s.sparse_dense_steps)} steps dense, "
                    f"tau {taus}")
            else:
                self.progress.line("Sparse attention (Sol-Attn): no kernel for this GPU; running dense attention")

    # -------------------------------------------------------------- sampling
    def _make_apply_fn(
        self,
        bundle: guide_mod.GuideBundle,
        audio_latent: torch.Tensor,
        audio_mask: torch.Tensor,
        frame_rate: float,
        cfg: float,
        ctx_pos: Optional[torch.Tensor] = None,
        ctx_neg: Optional[torch.Tensor] = None,
        tiling=None,
        factor: int = 2,
    ):
        model = self.model
        device = self.device
        dtype = self.dtype
        v_mask = bundle.noise_mask.to(device)
        a_mask = audio_mask.to(device)
        ax = audio_latent.to(device=device, dtype=dtype)
        kf = bundle.keyframe_idxs.to(device)
        entries = bundle.guide_attention_entries
        ctx_pos = (self._ctx_pos if ctx_pos is None else ctx_pos).to(device=device, dtype=dtype)
        if ctx_neg is None:
            ctx_neg = self._ctx_neg
        ctx_neg = ctx_neg.to(device=device, dtype=dtype) if ctx_neg is not None else None
        cond_fps = float(frame_rate)
        topts = self._topts
        policy = self.sparse_policy
        calls = [0]
        tiled = None
        if tiling is not None:
            from .tiling import TiledVelocity

            tiled = TiledVelocity(tiling, bundle.num_guide_latent_frames, factor, kf, entries, device)

        n_guide_frames = int(bundle.num_guide_latent_frames)

        def _call(vx: torch.Tensor, mask: torch.Tensor, kf_, entries_, sigma: float, context: torch.Tensor):
            if policy is not None:
                policy.sink_tokens = int((mask[:, :1, -n_guide_frames:] >= 0).sum()) if n_guide_frames else 0
            v_t = _V_PATCHIFIER.patchify((mask * sigma)[:, :1])[0]
            a_t = _A_PATCHIFIER.patchify((a_mask * sigma)[:, :1, :, :1])[0]
            out = model(
                [vx.to(dtype), ax.clone()],
                (v_t, a_t),
                context=context,
                attention_mask=None,
                frame_rate=cond_fps,
                transformer_options=dict(topts),
                keyframe_idxs=kf_,
                denoise_mask=mask,
                guide_attention_entries=entries_,
            )
            v_out = out[0] if isinstance(out, (list, tuple)) else out
            return v_out.float()

        def _run(vx: torch.Tensor, sigma: float, context: torch.Tensor) -> torch.Tensor:
            if tiled is None:
                return _call(vx, v_mask, kf, entries, sigma, context)
            return tiled(vx, v_mask, lambda xs, ms, k_, e_: _call(xs, ms, k_, e_, sigma, context))

        def apply_fn(x: torch.Tensor, sigma: float) -> torch.Tensor:
            if policy is not None:
                policy.step = calls[0]
            calls[0] += 1
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

    # VAE decode tiles measured 2026-11-01 on an RTX 5090 (results/WP-LTX decode bench): peak allocation of the
    # decode alone, worst case over 1080p 49/145 f and 4K 49 f. Larger tiles decode 1.4-2.2x faster and stay
    # within 47 dB of the 512 px default (tile seams only).
    _DECODE_DEFAULT = planner_mod.DECODE_TILE_DEFAULT

    def _decode_available_bytes(self) -> int:
        """VRAM this process may still allocate for the decode: its cap (Save VRAM already excluded) minus what it
        holds, and never more than the device has free minus Save VRAM (other processes)."""
        if not torch.cuda.is_available():
            return 0
        gc.collect()
        mm.soft_empty_cache()
        total = int(torch.cuda.get_device_properties(self.device).total_memory)
        cap = int(loader_mod._VRAM_CAP["bytes"] or total)
        save = int(loader_mod._VRAM_CAP["save"] or float(self.s.reserve_vram_gb) * (1024 ** 3))
        return planner_mod.decode_available_bytes(cap, int(torch.cuda.memory_reserved(self.device)),
                                                  int(loader_mod.device_free_bytes(self.device)), save)

    def _decode_tiles(self, latent_frames: int, available_bytes: int):
        s = self.s
        manual = (int(s.decode_tile_size), int(s.decode_tile_overlap), int(s.decode_temporal_size),
                  int(s.decode_temporal_overlap))
        if int(s.decode_tile_size) > 0:
            return manual
        return planner_mod.choose_decode_tiles(int(available_bytes), int(latent_frames))

    def _release_transformer(self) -> None:
        """Last chunk of the last pass: the transformer is not needed again, so its resident blocks leave the GPU
        before the VAE decode (the decode then has the whole budget)."""
        self.model = None
        self._main_lora = None
        self._pre_lora = None
        gc.collect()
        mm.soft_empty_cache()

    # ---------------------------------------------------------------- tiling
    def _tiling_for(self, plan: planner_mod.ChunkPlan, mode: str, factor: int):
        s = self.s
        mode = str(mode or "off").lower()
        if mode == "off":
            return None
        from . import tiling as tiling_mod

        if mode == "auto" and not tiling_mod.tiling_needed(plan.generation_height, plan.generation_width):
            return None
        return tiling_mod.build_tiling(
            plan.generation_height // 32, plan.generation_width // 32,
            int(s.spatial_tile_height), int(s.spatial_tile_width),
            overlap=float(s.spatial_tile_overlap), blend=str(s.spatial_tile_blend or "trapezoid"),
            align=max(1, int(factor)),
        )

    def _plan(self, total_frames: int, width: int, height: int, scale: int, factor: int, tiling_mode: str):
        """Chunk plan for one pass; with spatial tiling the VRAM budget follows one window."""
        s = self.s
        guide_ratio = 1.0 / float(factor * factor)
        if str(s.token_budget_mode).lower() == "auto":
            weight_bytes_hint = 0  # weights stream; activation budget dominates
            free_vram = loader_mod.budget_free_bytes(self.device)
            budget, reason = planner_mod.auto_token_budget(
                total_frames, width, height,
                free_vram_bytes=int(free_vram),
                weight_resident_bytes=weight_bytes_hint,
                reserve_gb=float(s.reserve_vram_gb),
                bytes_per_token=float(s.bytes_per_token),
                scale=scale,
                guide_token_ratio=guide_ratio,
            )
            tiling = None
            if str(tiling_mode).lower() != "off":
                probe = planner_mod.plan_chunks(total_frames, width, height, 10 ** 9,
                                                max_chunk_frames=int(s.max_chunk_frames),
                                                overlap_frames=int(s.overlap_frames), scale=scale)
                tiling = self._tiling_for(probe, tiling_mode, factor)
            if tiling is not None:
                # Budget in window tokens -> latent frames -> canvas tokens.
                tw_px, th_px = tiling.tile_w * 32 // scale, tiling.tile_h * 32 // scale
                tile_budget, _ = planner_mod.auto_token_budget(
                    total_frames, tw_px, th_px, free_vram_bytes=int(free_vram), weight_resident_bytes=0,
                    reserve_gb=float(s.reserve_vram_gb), bytes_per_token=float(s.bytes_per_token),
                    scale=scale, guide_token_ratio=guide_ratio)
                lat_frames = max(2, tile_budget // max(1, tiling.tile_h * tiling.tile_w))
                budget = lat_frames * probe.spatial_tokens
                reason = f"spatial tiling: {lat_frames} latent frames per window fit"
            self.progress.line(f"Auto VRAM token budget: {budget} ({reason})")
        else:
            budget = int(s.max_latent_tokens)
        plan = planner_mod.plan_chunks(
            total_frames, width, height, budget,
            max_chunk_frames=int(s.max_chunk_frames), overlap_frames=int(s.overlap_frames), scale=scale,
        )
        tiling = self._tiling_for(plan, tiling_mode, factor)
        return plan, tiling

    def _guide_untiled(self, guide_w: int, guide_h: int) -> bool:
        mode = str(self.s.guide_encode or "tiled").lower()
        if mode == "untiled":
            return True
        if mode == "auto":
            return int(guide_w) * int(guide_h) <= int(self.s.guide_untiled_max_pixels)
        return False

    # ------------------------------------------------------------------ pass
    def _run_pass(self, spec: _PassSpec, frames_u8: torch.Tensor, fps: float, plan: planner_mod.ChunkPlan,
                  tiling, make_writer: Callable[[planner_mod.ChunkPlan], object], out_total: int,
                  cancel_check: Callable[[], bool]):
        s = self.s
        latent_h = plan.generation_height // 32
        latent_w = plan.generation_width // 32
        latent_frames = plan.latent_frames
        if tiling is not None:
            self.progress.line(f"[{spec.label}] Spatial tiling: {tiling.describe()}")

        audio_t = loader_mod.audio_latent_frames(
            plan.chunk_length, fps, self.audio_constants["latents_per_second"]
        )
        audio_latent = torch.zeros(
            (1, self.audio_constants["z_channels"], max(audio_t, 1), self.audio_constants["freq_bins"]),
            dtype=torch.float32,
        )
        audio_mask = torch.zeros((1, 1, audio_latent.shape[2], 1), dtype=torch.float32)

        writer = None
        merger: Optional[StreamingChunkMerger] = None
        total_steps = plan.chunk_count * int(s.steps)
        done_steps = 0
        guide_w = plan.generation_width // spec.factor
        guide_h = plan.generation_height // spec.factor
        untiled_guide = self._guide_untiled(guide_w, guide_h)
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
                if spec.lanczos_guide and (chunk_float.shape[2] != guide_w or chunk_float.shape[1] != guide_h):
                    # Refine mode: the source is lanczos-resized to the guide canvas (the output canvas for a
                    # factor-1 LoRA) before the encode, as the Refine-Details / Restore workflows do.
                    chunk_float = comfy.utils.common_upscale(
                        chunk_float.movedim(-1, 1), guide_w, guide_h, "lanczos", "disabled",
                    ).movedim(1, -1).clamp(0.0, 1.0)

                self.progress.line("VAE encoding guide" + (" (untiled)..." if untiled_guide else " (tiled)..."))
                bundle = guide_mod.build_guide(
                    self.vae, chunk_float, latent_frames, latent_h, latent_w,
                    guide_strength=float(s.guide_strength),
                    latent_downscale_factor=float(spec.factor),
                    tile_size=int(s.encode_tile_size), tile_overlap=int(s.encode_tile_overlap),
                    use_tiled_encode=not untiled_guide,
                )
                del chunk_float

                self.vae.to_cpu()

                sigmas = self._sigmas(
                    bundle.latent.shape[2] * latent_h * latent_w
                )
                clean = bundle.latent.to(self.device)
                noise = sampling_mod.prepare_noise(
                    bundle.latent, int(s.seed) + chunk_index + spec.seed_offset).to(self.device)
                apply_fn = self._make_apply_fn(bundle, audio_latent, audio_mask, fps, float(s.cfg),
                                               ctx_pos=spec.ctx_pos, ctx_neg=spec.ctx_neg,
                                               tiling=tiling, factor=spec.factor)
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

                sample_t0 = time.time()
                latent_out = sampling_mod.run_sampler(
                    spec.sampler, masked, noise, sigmas, seed=int(s.seed) + chunk_index + spec.seed_offset,
                    callback=_step_cb
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                sample_s = time.time() - sample_t0
                del noise, clean, masked, apply_fn
                latent_out = guide_mod.crop_guides(latent_out, bundle.num_guide_latent_frames)
                del bundle
                mm.soft_empty_cache()

                if spec.final and chunk_index == plan.chunk_count - 1:
                    self._release_transformer()
                peak_before = torch.cuda.max_memory_allocated(self.device) if torch.cuda.is_available() else 0
                reserved_before = torch.cuda.max_memory_reserved(self.device) if torch.cuda.is_available() else 0
                avail = self._decode_available_bytes() if torch.cuda.is_available() else 0
                latent_out_frames = int(latent_out.shape[2])
                dec = self._decode_tiles(latent_out_frames, avail)
                self.progress.line("VAE decoding (tiled %d/%d px, %d/%d frames)..." % dec)
                held = 0
                if torch.cuda.is_available():
                    held = torch.cuda.memory_allocated(self.device)
                    torch.cuda.reset_peak_memory_stats(self.device)
                decode_t0 = time.time()
                try:
                    pixels = self.vae.decode_tiled(
                        latent_out.float().cpu(), tile_size_px=dec[0], overlap_px=dec[1],
                        temporal_size_frames=dec[2], temporal_overlap_frames=dec[3],
                    )
                except mm.OOM_EXCEPTION:
                    if int(s.decode_tile_size) > 0 or dec == self._DECODE_DEFAULT:
                        raise
                    self.vae.to_cpu()
                    gc.collect()
                    mm.soft_empty_cache()
                    self.progress.line("VAE decode ran out of VRAM with large tiles; decoding with 512 px tiles")
                    dec = self._DECODE_DEFAULT
                    pixels = self.vae.decode_tiled(
                        latent_out.float().cpu(), tile_size_px=dec[0], overlap_px=dec[1],
                        temporal_size_frames=dec[2], temporal_overlap_frames=dec[3],
                    )
                del latent_out
                self.vae.to_cpu()
                self.progress.line(f"Timing: sampling {sample_s:.1f}s, VAE decode {time.time() - decode_t0:.1f}s")
                if torch.cuda.is_available():
                    decode_peak = torch.cuda.max_memory_allocated(self.device)
                    self.progress.line(
                        f"VAE decode memory: +{(decode_peak - held) / 1024 ** 3:.2f} GB over {held / 1024 ** 3:.2f} GB held, "
                        f"{avail / 1024 ** 3:.2f} GB available, {int(latent_out_frames)} latent frames")
                    peak_before = max(peak_before, decode_peak)

                pixels = pixels[:plan.chunk_length, : plan.output_height, : plan.output_width, :]
                if writer is None:
                    writer = make_writer(plan)
                    merger = StreamingChunkMerger(writer, plan.overlap_frames, out_total)
                assert merger is not None
                merger.add_chunk(
                    pixels, keep_length, is_last=(chunk_index == plan.chunk_count - 1)
                )
                del pixels
                gc.collect()
                mm.soft_empty_cache()
                peak_note = ""
                if torch.cuda.is_available():
                    peak = max(peak_before, torch.cuda.max_memory_allocated(self.device)) / (1024 ** 3)
                    reserved = max(reserved_before, torch.cuda.max_memory_reserved(self.device)) / (1024 ** 3)
                    peak_note = f" | peak VRAM {peak:.1f}GB (reserved {reserved:.1f}GB)"
                self.progress.line(
                    f"Chunk {chunk_index + 1}/{plan.chunk_count} done in {time.time() - chunk_t0:.1f}s{peak_note}"
                )
        except BaseException:
            if writer is not None and not isinstance(writer, _FrameCollector):
                self._failed_writer = writer
            raise
        return writer

    # ------------------------------------------------------------------- run
    def run(self, cancel_check: Optional[Callable[[], bool]] = None) -> str:
        s = self.s
        total_start = time.time()
        cancel_check = cancel_check or (lambda: False)

        # Save VRAM: never ask for more than (free now - Save VRAM); no shared GPU memory.
        loader_mod.apply_vram_cap(self.device, float(s.reserve_vram_gb), emit=self.progress.line)

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
        out_total = total_frames
        tail_pad = max(0, int(s.tail_pad_frames or 0))
        work_frames = total_frames + tail_pad

        mode = str(s.mode or "x2").lower()
        if mode == "refine":
            main_factor = max(1, int(round(loader_mod.read_lora_downscale(s.ic_lora_path))))
            main = _PassSpec("Refine", scale=2, factor=main_factor, lanczos_guide=True,
                             sampler=s.sampler, tiling_mode=s.spatial_tiling, final=True)
        else:
            main = _PassSpec("x2", scale=2, factor=2, lanczos_guide=False, sampler=s.sampler,
                             tiling_mode=s.spatial_tiling, final=True)

        # --- plan chunks -------------------------------------------------
        plan, tiling = self._plan(work_frames, width, height, main.scale, main.factor, main.tiling_mode)
        if tail_pad:
            # The padding is worth its frames only while it does not add a chunk (a 4K clip that just fits one
            # chunk would otherwise run as two: measured +62 % sampling time).
            plain, plain_tiling = self._plan(total_frames, width, height, main.scale, main.factor, main.tiling_mode)
            if plan.chunk_count > plain.chunk_count:
                self.progress.line(f"Tail padding skipped: {tail_pad} more frames would need {plan.chunk_count} chunks "
                                   f"instead of {plain.chunk_count}")
                plan, tiling, tail_pad, work_frames = plain, plain_tiling, 0, total_frames
        if tail_pad:
            frames_u8 = torch.cat((frames_u8, frames_u8[-1:].repeat(tail_pad, 1, 1, 1)), dim=0)
            self.progress.line(f"Tail padding: {tail_pad} repeated last frames are generated and trimmed")
        for line in plan.summary(int(s.steps)).splitlines():
            self.progress.line(f"[Plan] {line}")
        pre_plan = None
        if s.preclean_lora_path:
            pre_plan, _ = self._plan(work_frames, width, height, 1, 1, "off")
            self.progress.line(f"[Plan] Pre-clean pass (1x, {Path(s.preclean_lora_path).name}): "
                               f"{pre_plan.chunk_count} chunk(s) of {pre_plan.chunk_length} frames")

        # --- prompts + models ---------------------------------------------
        self.encode_prompts()
        if cancel_check():
            raise KeyboardInterrupt()
        call_tokens = None
        if tiling is not None:
            call_tokens = tiling.tile_h * tiling.tile_w * plan.latent_frames
        guide_ratio = 1.0 / float(main.factor * main.factor)
        self.load_models(plan, call_tokens=call_tokens, guide_ratio=guide_ratio)
        main.ctx_pos, main.ctx_neg = self._ctx_pos, self._ctx_neg

        # --- optional pre-clean pass (1x) ----------------------------------
        if pre_plan is not None:
            self._main_lora.disable()
            self._pre_lora.enable()
            pre = _PassSpec("Pre-clean", scale=1, factor=1, lanczos_guide=False,
                            ctx_pos=self._ctx_pre_pos, ctx_neg=self._ctx_pre_neg, sampler="euler",
                            seed_offset=1000)
            t0 = time.time()
            self.progress.line("Pre-clean pass: same-size IC-LoRA pass before the x2 upscale")
            collector = self._run_pass(pre, frames_u8, fps, pre_plan, None, lambda _p: _FrameCollector(),
                                       work_frames, cancel_check)
            frames_u8 = collector.result()[:work_frames, :height, :width].contiguous()
            del collector
            self._pre_lora.disable()
            self._main_lora.enable()
            self.progress.line(f"Pre-clean pass done in {time.time() - t0:.1f}s")
        elif self._main_lora is not None:
            self._main_lora.enable()

        # Start ffmpeg lazily after the first chunk has actually decoded. Starting it before
        # VAE/sampling work leaves an audio-only MP4 when inference fails early; older runner
        # code then mistook that artifact for a successful video and entered concat fallbacks.
        def _make_writer(p: planner_mod.ChunkPlan) -> FfmpegWriter:
            return FfmpegWriter(
                s.output_path,
                width=p.output_width,
                height=p.output_height,
                fps=fps,
                audio_source=(s.input_path if (s.keep_audio and has_audio) else None),
                codec=s.codec,
                crf=int(s.crf),
                preset=s.preset,
                pixel_format=s.pixel_format,
            )

        writer: Optional[FfmpegWriter] = None
        pipeline_failed = True
        writer_close_error: Optional[Exception] = None
        self._failed_writer = None
        try:
            writer = self._run_pass(main, frames_u8, fps, plan, tiling, _make_writer, out_total, cancel_check)
            pipeline_failed = False
        finally:
            if writer is None:
                writer = self._failed_writer
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

        if writer is None or int(writer.frames_written) != int(out_total):
            written = int(writer.frames_written) if writer is not None else 0
            try:
                Path(s.output_path).unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError(
                f"LTX 2.5 writer produced {written}/{int(out_total)} required frames"
            )

        if self.sparse_policy is not None:
            self.progress.line(f"Sparse attention calls: {self.sparse_policy.calls_sparse} sparse, "
                               f"{self.sparse_policy.calls_dense} dense")
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
