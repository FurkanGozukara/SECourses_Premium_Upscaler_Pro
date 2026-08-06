"""
Shared auto-tune search primitives.

All three VSR auto-tuners (SeedVR2 / FlashVSR+ / SparkVSR) look for the
highest-quality candidate on a resource-monotone axis: probing a more
aggressive candidate never uses less VRAM than a less aggressive one.
That makes the pass/fail frontier bisectable, so instead of walking the
candidate grid linearly we probe the top candidate first and bisect on
failure.

This module is intentionally dependency-free (no torch/gradio) so the
search logic is unit-testable without a GPU.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple


def next_bisect_index(lo: int, hi: int) -> Optional[int]:
    """
    Next index to probe on an ascending-aggressiveness candidate grid.

    lo: highest index known to pass (-1 when none passed yet).
    hi: lowest index known to fail (len(candidates) when none failed yet).
    Returns None when the frontier is settled (hi - lo <= 1).
    """
    lo = int(lo)
    hi = int(hi)
    if hi - lo <= 1:
        return None
    return (lo + hi) // 2


def frontier_bisect(
    num_candidates: int,
    probe,
    initial_index: Optional[int] = None,
) -> Generator[Any, Any, Dict[str, Any]]:
    """
    Find the highest-index passing candidate with top-first probing + bisection.

    `probe(index, require_full)` must be a generator (its yields are forwarded,
    so UI progress payloads flow through) returning a dict:
        {"outcome": "pass" | "boundary_fail" | "hard_fail" | "cancelled",
         "early_stopped_pass": bool}   # pass measured via early-success stop

    `initial_index` seeds the first probe (e.g. a frontier predicted from
    measured peaks in other logs). After a seeded first probe the immediate
    NEIGHBOR is probed next, so an accurate prediction settles the frontier in
    two probes; an inaccurate one simply falls back to normal bisection.

    A settled best whose pass came from an early-success stop is re-probed with
    require_full=True; if the full-length probe fails the candidate is demoted
    and bisection resumes below it, so the returned best is always validated
    at full length.

    Returns {"best_idx": Optional[int], "stopped": None | "hard_fail" | "cancelled",
             "boundary_failed": bool}.
    """
    passes: Dict[int, bool] = {}
    hi = int(num_candidates)
    boundary_failed = False
    stopped: Optional[str] = None

    def _lo() -> int:
        return max(passes) if passes else -1

    seeded = (
        initial_index is not None
        and num_candidates > 0
        and 0 <= int(initial_index) < num_candidates
    )
    pending: Optional[int] = (
        int(initial_index) if seeded else ((num_candidates - 1) if num_candidates > 0 else None)
    )
    seed_neighbor: Optional[int] = None
    first_probe_is_seeded = bool(seeded)
    while stopped is None:
        if pending is not None:
            res = yield from probe(int(pending), False)
            kind = str((res or {}).get("outcome") or "hard_fail")
            if kind == "pass":
                passes[int(pending)] = bool((res or {}).get("early_stopped_pass", False))
                if first_probe_is_seeded:
                    seed_neighbor = int(pending) + 1
            elif kind == "boundary_fail":
                boundary_failed = True
                hi = min(hi, int(pending))
                if first_probe_is_seeded:
                    seed_neighbor = int(pending) - 1
            elif kind == "cancelled":
                stopped = "cancelled"
                break
            else:
                stopped = "hard_fail"
                break
            first_probe_is_seeded = False
            if seed_neighbor is not None and _lo() < seed_neighbor < hi:
                pending = seed_neighbor
                seed_neighbor = None
                continue
            seed_neighbor = None
            pending = next_bisect_index(_lo(), hi)
            continue

        # Frontier settled: the applied best must be a full-length measurement.
        lo = _lo()
        if lo >= 0 and passes.get(lo):
            res = yield from probe(int(lo), True)
            kind = str((res or {}).get("outcome") or "hard_fail")
            if kind == "pass":
                passes[lo] = False
                continue
            if kind == "cancelled":
                stopped = "cancelled"
                break
            if kind == "hard_fail":
                stopped = "hard_fail"
                break
            boundary_failed = True
            passes.pop(lo, None)
            hi = min(hi, lo)
            pending = next_bisect_index(_lo(), hi)
            continue
        break

    lo = _lo()
    return {
        "best_idx": (lo if lo >= 0 else None),
        "stopped": stopped,
        "boundary_failed": bool(boundary_failed),
    }


def predict_frontier_index(
    candidate_values: Sequence[Any],
    measured_peaks_by_value: Dict[Any, float],
    budget_gb: float,
) -> Optional[int]:
    """
    Predict the pass/fail frontier from measured peaks recorded on a different
    GPU (peak VRAM used by a config is largely hardware-portable).

    candidate_values are ordered ascending by resource demand; budget_gb is
    this machine's total VRAM minus the free-headroom target. Returns the
    largest candidate index whose measured peak fits the budget, 0 when
    measurements exist but none fit (seed a fast bottom check), or None when
    no candidate has a measurement (no prediction possible).
    """
    try:
        budget = float(budget_gb)
    except Exception:
        return None
    best: Optional[int] = None
    covered = False
    for idx, value in enumerate(candidate_values):
        try:
            peak = measured_peaks_by_value.get(value)
        except TypeError:
            peak = None
        if peak is None:
            continue
        covered = True
        try:
            if float(peak) > 0 and float(peak) <= budget:
                best = idx
        except Exception:
            continue
    if best is not None:
        return best
    return 0 if covered else None


# --------------------------------------------------------------------------- #
# FlashVSR+ DiT tile grid (mirrors FlashVSR_plus/run.py calculate_tile_coords)
# --------------------------------------------------------------------------- #
def flashvsr_tile_grid_signature(
    height: int,
    width: int,
    tile_size: int,
    overlap: int,
) -> Tuple[Tuple[int, int, int, int], ...]:
    height = int(height)
    width = int(width)
    tile_size = int(tile_size)
    overlap = int(overlap)
    stride = tile_size - overlap
    if stride <= 0 or height <= 0 or width <= 0:
        return ((0, max(0, height), 0, max(0, width)),)
    num_rows = max(1, math.ceil(max(1, height - overlap) / stride))
    num_cols = max(1, math.ceil(max(1, width - overlap) / stride))
    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(num_rows):
        for j in range(num_cols):
            y1 = i * stride
            x1 = j * stride
            y2 = min(y1 + tile_size, height)
            x2 = min(x1 + tile_size, width)
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
            boxes.append((y1, y2, x1, x2))
    return tuple(boxes)


def dedupe_flashvsr_tile_candidates(
    height: int,
    width: int,
    tile_candidates: Sequence[int],
    overlap: int,
) -> List[int]:
    """
    Drop tile sizes whose DiT tile boxes are byte-identical for this input.

    Once a tile covers the frame every larger tile produces the same single
    box, so probing them separately measures the same run repeatedly (observed
    in real logs: 15 of 26 probes identical). Keeps the LARGEST tile of each
    distinct grid so the stored setting matches the old sweep's top value.
    Returns an ascending list.
    """
    by_grid: Dict[Tuple[Tuple[int, int, int, int], ...], int] = {}
    for tile in tile_candidates:
        tile = int(tile)
        if tile <= 0:
            continue
        grid = flashvsr_tile_grid_signature(height, width, tile, overlap)
        prev = by_grid.get(grid)
        if prev is None or tile > prev:
            by_grid[grid] = tile
    return sorted(by_grid.values())


# --------------------------------------------------------------------------- #
# SparkVSR spatial tile grid (mirrors tools/sparkvsr_inference.py make_spatial_tiles)
# --------------------------------------------------------------------------- #
def sparkvsr_tile_grid_signature(
    height: int,
    width: int,
    tile: int,
    overlap: int,
) -> Tuple[Tuple[int, int, int, int], ...]:
    height = int(height)
    width = int(width)
    tile = int(tile)
    overlap = int(overlap)
    if tile <= 0:
        return ((0, height, 0, width),)
    stride = tile - overlap
    if stride <= 0:
        return ((0, height, 0, width),)

    def tile_starts(length: int) -> List[int]:
        if length <= tile:
            return [0]
        starts = [0]
        while starts[-1] + tile < length:
            next_start = starts[-1] + stride
            if next_start + tile >= length:
                if next_start < length and next_start != starts[-1]:
                    starts.append(next_start)
                break
            starts.append(next_start)
        return sorted(set(max(0, min(int(s), max(0, length - 1))) for s in starts))

    boxes: List[Tuple[int, int, int, int]] = []
    for h_start in tile_starts(height):
        h_end = min(h_start + tile, height)
        for w_start in tile_starts(width):
            w_end = min(w_start + tile, width)
            boxes.append((h_start, h_end, w_start, w_end))
    return tuple(boxes)


def dedupe_sparkvsr_tile_candidates(
    height: int,
    width: int,
    tile_candidates: Sequence[int],
    overlap: int,
) -> List[int]:
    """
    Filter SparkVSR square-tile candidates for the effective input dims.

    - A tile covering the whole frame is identical work to tile=0 (full-frame),
      so such candidates are dropped whenever 0 is among the candidates.
    - Remaining candidates are deduped by their exact tile-box grid, keeping
      the largest tile per grid (higher quality, same memory).
    Returns the surviving tile sizes sorted ascending (0 excluded).
    """
    full_frame_grid = ((0, int(height), 0, int(width)),)
    has_full_frame = any(int(t) <= 0 for t in tile_candidates)
    by_grid: Dict[Tuple[Tuple[int, int, int, int], ...], int] = {}
    for tile in tile_candidates:
        tile = int(tile)
        if tile <= 0:
            continue
        grid = sparkvsr_tile_grid_signature(height, width, tile, overlap)
        if has_full_frame and grid == full_frame_grid:
            continue
        prev = by_grid.get(grid)
        if prev is None or tile > prev:
            by_grid[grid] = tile
    return sorted(by_grid.values())
