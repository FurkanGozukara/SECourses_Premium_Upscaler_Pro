"""
Chunk planning for the LTX 2.5 pixel spatial upscaler.

Exact port of the `ltx25_dynamic_upscaler` ComfyUI custom nodes
(LTX25PrepareVideoChunks / LTX25MergeVideoChunks), plus an automatic
token-budget mode that prefers processing the whole video as a single
chunk — the sweet spot for this model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

# torch is only needed by the tensor helpers (pad/slice/merge). The planning
# math is pure Python so the app process can import this module for the LIVE
# chunk-plan preview without initializing torch.
if TYPE_CHECKING:  # pragma: no cover
    import torch


def round_down_8n1(value: int) -> int:
    """Round down to the 8n+1 frame grid (minimum 9)."""
    value = max(9, int(value))
    return ((value - 1) // 8) * 8 + 1


def one_chunk_length(total_frames: int) -> int:
    """Smallest 8n+1 length that covers the whole video in one chunk."""
    return max(9, math.ceil((max(1, int(total_frames)) - 1) / 8) * 8 + 1)


def valid_chunk_length(requested: int, spatial_tokens: int, token_budget: int) -> int:
    """LTX25PrepareVideoChunks._valid_chunk_length — verbatim math."""
    max_latent_frames = max(2, int(token_budget) // max(1, int(spatial_tokens)))
    token_limited = 8 * (max_latent_frames - 1) + 1
    requested = round_down_8n1(requested)
    return max(9, min(requested, token_limited))


def normalize_overlap(overlap_frames: int, chunk_length: int) -> int:
    """Snap overlap to the 8n+1 grid, min 1, and keep it <= chunk_length - 8."""
    overlap_frames = min(int(overlap_frames), chunk_length - 8)
    return max(1, ((overlap_frames - 1) // 8) * 8 + 1)


@dataclass
class ChunkPlan:
    total_frames: int
    source_width: int
    source_height: int
    padded_width: int
    padded_height: int
    generation_width: int
    generation_height: int
    output_width: int
    output_height: int
    spatial_tokens: int
    latent_frames: int
    chunk_length: int
    overlap_frames: int
    step: int
    token_budget: int
    effective_tokens: int
    chunk_ranges: List[Tuple[int, int]] = field(default_factory=list)  # (start, keep_length)
    active_limiter: str = ""

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_ranges)

    def summary(self, steps: int) -> str:
        util = self.effective_tokens / max(1, self.token_budget) * 100.0
        lines = [
            f"Source {self.source_width}x{self.source_height} x {self.total_frames} frames"
            f" -> output {self.output_width}x{self.output_height}",
            f"Generation grid {self.generation_width}x{self.generation_height}"
            f" ({self.spatial_tokens} tokens/frame)",
            f"Chunk length {self.chunk_length} frames ({self.latent_frames} latent frames),"
            f" overlap {self.overlap_frames}, chunks: {self.chunk_count}",
            f"Token budget {self.token_budget} | video tokens/chunk {self.effective_tokens}"
            f" ({util:.0f}% of budget) | limiter: {self.active_limiter}",
            f"Total sampler iterations: {self.chunk_count * int(steps)}"
            f" ({steps} steps x {self.chunk_count} chunk(s))",
        ]
        if self.chunk_count > 1:
            single = one_chunk_length(self.total_frames)
            single_tokens = self.spatial_tokens * (((single - 1) // 8) + 1)
            hints = []
            if single > self.chunk_length:
                hints.append(f"Max Chunk Frames >= {single}")
            if single_tokens > self.token_budget:
                hints.append(f"Manual token budget >= {single_tokens}")
            if hints:
                lines.append(
                    "Single-chunk hint: set "
                    + " and ".join(hints)
                    + f" to process all {self.total_frames} frames in ONE chunk"
                    " (needs enough VRAM; watch peak usage)."
                )
        return "\n".join(lines)


def plan_chunks(
    total_frames: int,
    width: int,
    height: int,
    token_budget: int,
    max_chunk_frames: int = 121,
    overlap_frames: int = 1,
) -> ChunkPlan:
    """Port of LTX25PrepareVideoChunks geometry + slicing (no tensors involved)."""
    total_frames = max(1, int(total_frames))
    padded_width = math.ceil(width / 32) * 32
    padded_height = math.ceil(height / 32) * 32
    generation_width = padded_width * 2
    generation_height = padded_height * 2
    spatial_tokens = (generation_width // 32) * (generation_height // 32)

    requested_valid = round_down_8n1(min(int(max_chunk_frames), one_chunk_length(total_frames)))
    chunk_length = valid_chunk_length(max_chunk_frames, spatial_tokens, token_budget)
    max_latent_frames = max(2, int(token_budget) // max(1, spatial_tokens))
    budget_frame_limit = 8 * (max_latent_frames - 1) + 1
    active_limiter = (
        "VRAM token budget" if budget_frame_limit < requested_valid else "requested frame cap"
    )

    overlap = normalize_overlap(overlap_frames, chunk_length)
    step = chunk_length - overlap

    ranges: List[Tuple[int, int]] = []
    start = 0
    while start < total_frames:
        end = min(start + chunk_length, total_frames)
        ranges.append((start, end - start))
        if end >= total_frames:
            break
        start += step

    latent_frames = ((chunk_length - 1) // 8) + 1
    return ChunkPlan(
        total_frames=total_frames,
        source_width=width,
        source_height=height,
        padded_width=padded_width,
        padded_height=padded_height,
        generation_width=generation_width,
        generation_height=generation_height,
        output_width=width * 2,
        output_height=height * 2,
        spatial_tokens=spatial_tokens,
        latent_frames=latent_frames,
        chunk_length=chunk_length,
        overlap_frames=overlap,
        step=step,
        token_budget=int(token_budget),
        effective_tokens=spatial_tokens * latent_frames,
        chunk_ranges=ranges,
        active_limiter=active_limiter,
    )


def auto_token_budget(
    total_frames: int,
    width: int,
    height: int,
    free_vram_bytes: int,
    weight_resident_bytes: int,
    reserve_gb: float = 1.0,
    bytes_per_token: float = 1_400_000.0,
    base_overhead_bytes: float = 2.5e9,
) -> Tuple[int, str]:
    """
    Choose the largest safe token budget for the available VRAM, preferring a
    single chunk for the whole video (the model performs best that way).

    The activation cost model is linear in the *total* transformer sequence:
    video tokens x 1.25 (surviving IC-LoRA guide tokens) + text/audio slack,
    calibrated with a conservative bytes/token so the first run never OOMs.
    Returns (token_budget, reason).
    """
    padded_w = math.ceil(width / 32) * 32
    padded_h = math.ceil(height / 32) * 32
    spatial_tokens = ((padded_w * 2) // 32) * ((padded_h * 2) // 32)

    usable = float(free_vram_bytes) - float(weight_resident_bytes)
    usable -= reserve_gb * (1024 ** 3) + base_overhead_bytes
    if usable <= 0:
        # Nothing left after weights: fall back to the minimum viable chunk.
        return max(4096, spatial_tokens * 2), "minimum (weights consume nearly all VRAM)"

    # tokens the sampler actually attends to = budget * 1.25 (+ ~1.2k text tokens)
    max_video_tokens = int(usable / (bytes_per_token * 1.25))
    max_video_tokens = max(4096, max_video_tokens)

    single = one_chunk_length(total_frames)
    single_latent = ((single - 1) // 8) + 1
    needed_single = spatial_tokens * single_latent
    if needed_single <= max_video_tokens:
        # Give a little headroom above the exact need so the planner lands on
        # the single-chunk length.
        return needed_single, f"single chunk fits ({single} frames in one pass)"
    return max_video_tokens, (
        f"largest budget that fits VRAM (single chunk would need {needed_single} tokens)"
    )


def pad_frames_to_grid(frames: "torch.Tensor", padded_width: int, padded_height: int) -> "torch.Tensor":
    """Edge-replicate pad bottom/right to the /32 grid (LTX25PrepareVideoChunks)."""
    import torch.nn.functional as F

    _, height, width, _ = frames.shape
    pad_right = padded_width - width
    pad_bottom = padded_height - height
    if pad_right or pad_bottom:
        frames = frames.movedim(-1, 1)
        frames = F.pad(frames, (0, pad_right, 0, pad_bottom), mode="replicate")
        frames = frames.movedim(1, -1)
    return frames


def slice_chunk(frames: "torch.Tensor", start: int, keep_length: int, chunk_length: int) -> "torch.Tensor":
    """Slice one chunk and pad the tail by repeating the last frame."""
    import torch

    chunk = frames[start:start + keep_length]
    if keep_length < chunk_length:
        padding = chunk[-1:].repeat(chunk_length - keep_length, 1, 1, 1)
        chunk = torch.cat((chunk, padding), dim=0)
    return chunk


class ChunkMerger:
    """Streaming port of LTX25MergeVideoChunks (linear cross-fade on overlap)."""

    def __init__(self, overlap_frames: int, total_frames: int):
        self.overlap = int(overlap_frames)
        self.total = int(total_frames)
        self.merged: Optional[torch.Tensor] = None

    def add_chunk(self, images: "torch.Tensor", keep_length: int) -> None:
        import torch

        current = images[: min(int(keep_length), images.shape[0])]
        if self.merged is None:
            self.merged = current.clone()
            return
        shared = min(self.overlap, self.merged.shape[0], current.shape[0])
        if shared:
            alpha = torch.linspace(
                1.0 / (shared + 1),
                shared / (shared + 1),
                shared,
                device=self.merged.device,
                dtype=self.merged.dtype,
            ).view(-1, 1, 1, 1)
            self.merged[-shared:] = self.merged[-shared:] * (1.0 - alpha) + current[:shared] * alpha
        if current.shape[0] > shared:
            self.merged = torch.cat((self.merged, current[shared:]), dim=0)

    def result(self) -> "torch.Tensor":
        import torch

        assert self.merged is not None, "No chunks were merged"
        merged = self.merged
        if merged.shape[0] < self.total:
            merged = torch.cat(
                (merged, merged[-1:].repeat(self.total - merged.shape[0], 1, 1, 1)), dim=0
            )
        return merged[: self.total]
