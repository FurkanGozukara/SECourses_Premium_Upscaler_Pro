"""
Geometry and window planning for Open-H3 Regenerate (pure Python, no torch).

Geometry: the draft is the whole input frame resized to the draft size (768P /
576P class area, aspect kept, 32 px grid); the latent upscaler maps it onto the
compute grid (the target rounded up to 32 px), one pass up to the 2K area and a
second pass above it (as the reference does for 4K); the decoded grid frame is
resized to the exact target size when the two differ.

Windows: H3 refines clips of 17k + 5 frames (5k + 2 latent frames). Long inputs
run as overlapping windows that are cross-faded on output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

DRAFT_AREAS = {"768p": 1344 * 768, "576p": 1024 * 576}
UPSCALER_PASS_AREA = 2560 * 1440      # one latent-upscaler pass stays within its trained range
GRID = 32                              # 16 px VAE patch x 2x2 DiT patch
MIN_FRAMES = 5
MAX_FRAMES = 361                       # 15 s at 24 fps (the refine's supported range)
TEXT_TOKENS_ESTIMATE = 512


def align_frame_count(n: int) -> int:
    n = max(MIN_FRAMES, int(n))
    return n + (5 - n) % 17


def video_latent_t(frames: int) -> int:
    return 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2


def snap(value: float, step: int = GRID, minimum: int = GRID) -> int:
    return max(minimum, int(round(float(value) / step)) * step)


def ceil_to(value: int, step: int = GRID) -> int:
    return max(step, int(math.ceil(int(value) / step)) * step)


def even(value: float) -> int:
    return max(2, int(round(float(value) / 2.0)) * 2)


def _area_dims(area: float, aspect: float) -> Tuple[int, int]:
    w = math.sqrt(area * aspect)
    h = math.sqrt(area / aspect)
    return snap(w), snap(h)


@dataclass
class Geometry:
    src_w: int
    src_h: int
    target_w: int
    target_h: int
    grid_w: int
    grid_h: int
    draft_w: int
    draft_h: int
    passes: List[Tuple[int, int]] = field(default_factory=list)  # pixel sizes of each latent-upscaler pass

    @property
    def tokens_per_latent_frame(self) -> int:
        return (self.grid_h // GRID) * (self.grid_w // GRID)

    def describe(self) -> str:
        chain = " -> ".join(f"{w}x{h}" for w, h in [(self.draft_w, self.draft_h)] + self.passes)
        tail = "" if (self.grid_w, self.grid_h) == (self.target_w, self.target_h) else \
            f" -> resize {self.grid_w}x{self.grid_h} to {self.target_w}x{self.target_h}"
        return (f"input {self.src_w}x{self.src_h} | draft {self.draft_w}x{self.draft_h} | latent upscale {chain}"
                f"{tail} | output {self.target_w}x{self.target_h}")


def plan_geometry(src_w: int, src_h: int, target_w: int, target_h: int, draft_size: str = "768p") -> Geometry:
    target_w, target_h = even(target_w), even(target_h)
    grid_w, grid_h = ceil_to(target_w), ceil_to(target_h)
    aspect = float(src_w) / float(max(1, src_h))
    key = str(draft_size or "768p").lower()
    if key in DRAFT_AREAS:
        draft_w, draft_h = _area_dims(DRAFT_AREAS[key], aspect)
    else:  # "native": the input size on the 32 px grid
        draft_w, draft_h = snap(src_w), snap(src_h)
    if draft_w * draft_h >= grid_w * grid_h:
        # target at or below the draft size: refine only, encoded straight at the compute grid
        draft_w, draft_h = grid_w, grid_h
    passes: List[Tuple[int, int]] = []
    if (draft_w, draft_h) != (grid_w, grid_h):
        if grid_w * grid_h > UPSCALER_PASS_AREA and draft_w * draft_h < UPSCALER_PASS_AREA:
            mid_w, mid_h = _area_dims(UPSCALER_PASS_AREA, grid_w / grid_h)
            if (mid_w, mid_h) != (grid_w, grid_h) and mid_w * mid_h > draft_w * draft_h:
                passes.append((mid_w, mid_h))
        passes.append((grid_w, grid_h))
    return Geometry(src_w, src_h, target_w, target_h, grid_w, grid_h, draft_w, draft_h, passes)


def window_tokens(frames: int, geometry: Geometry) -> int:
    padded = align_frame_count(frames)
    audio_t = round(padded / 24 * 40)
    return video_latent_t(padded) * geometry.tokens_per_latent_frame + 2 * audio_t + TEXT_TOKENS_ESTIMATE


def frames_for_token_budget(budget_tokens: int, geometry: Geometry, max_frames: int) -> int:
    """Largest window (<= max_frames content frames, padded to 17k+5) whose packed sequence fits the token budget."""
    limit = max(MIN_FRAMES, min(MAX_FRAMES, int(max_frames)))
    if window_tokens(limit, geometry) <= budget_tokens:
        return limit
    frames = align_frame_count(limit) - 17
    while frames > MIN_FRAMES and window_tokens(frames, geometry) > budget_tokens:
        frames -= 17
    return max(MIN_FRAMES, frames)


def auto_token_budget(free_vram_bytes: int, reserve_gb: float, bytes_per_token: float, min_weight_gb: float = 4.0) -> int:
    budget = float(free_vram_bytes) - (reserve_gb + min_weight_gb) * (1024 ** 3)
    return max(20000, int(budget / max(1.0, float(bytes_per_token))))


@dataclass
class WindowPlan:
    window_frames: int
    overlap_frames: int
    total_frames: int
    windows: List[Tuple[int, int]]  # (start, length) in output-timeline frames

    def summary(self) -> str:
        if len(self.windows) == 1:
            return f"1 window of {self.windows[0][1]} frames (padded to {align_frame_count(self.windows[0][1])})"
        return (f"{len(self.windows)} windows of up to {self.window_frames} frames, "
                f"{self.overlap_frames} frames cross-faded between windows")


def plan_windows(total_frames: int, window_frames: int, overlap_frames: int) -> WindowPlan:
    total = max(1, int(total_frames))
    window = max(MIN_FRAMES, min(MAX_FRAMES, int(window_frames)))
    overlap = max(0, min(int(overlap_frames), window - MIN_FRAMES))
    if total <= window:
        return WindowPlan(window, overlap, total, [(0, total)])
    windows = []
    start = 0
    step = window - overlap
    while True:
        length = min(window, total - start)
        windows.append((start, length))
        if start + length >= total:
            break
        start += step
        if total - start <= overlap:
            break
    return WindowPlan(window, overlap, total, windows)
