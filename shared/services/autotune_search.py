"""
Shared auto-tune search primitives.

All three VSR auto-tuners (SeedVR2 / FlashVSR+ / SparkVSR) look for the
highest-quality candidate on a resource-monotone axis: probing a more
aggressive candidate never uses less VRAM than a less aggressive one.
The search starts from the lightest candidate, grows in bounded steps, and
bisects after it brackets a clean VRAM boundary.  This is slightly more work
than top-first bisection, but it avoids launching the most dangerous probe on
an unknown machine and still converges quickly.

This module is intentionally dependency-free (no torch/gradio) so the
search logic is unit-testable without a GPU.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple


DEFAULT_MAX_GROWTH_STEP = 4
DEFAULT_MAX_INITIAL_HARD_FAILURES = 3
DEFAULT_MAX_PIXEL_DIFF = 0.05
DEFAULT_MAX_ASPECT_DIFF = 0.03
DEFAULT_MAX_DIMENSION_DIFF = 0.05
DEFAULT_LAUNCH_MARGIN_GB = 1.0
DEFAULT_MAX_AMBIENT_DROP_FOR_REUSE_GB = 0.5


def vram_drain_target_gb(
    ambient_used_gb: Any,
    *,
    slack_gb: Any = 1.5,
    floor_gb: Any = 3.0,
) -> float:
    """Return an achievable post-probe VRAM-settle threshold.

    The target is deliberately relative to the measured pre-tune baseline.
    Capping it to a fraction of total VRAM is incorrect when another legitimate
    workload already uses more than that fraction: the wait can never finish
    early even after the probe has released all of its memory.
    """
    try:
        ambient = max(0.0, float(ambient_used_gb))
        slack = max(0.0, float(slack_gb))
        floor = max(0.0, float(floor_gb))
    except Exception:
        return 3.0
    if not all(math.isfinite(value) for value in (ambient, slack, floor)):
        return 3.0
    return max(floor, ambient + slack)


def frontier_bisect(
    num_candidates: int,
    probe,
    initial_index: Optional[int] = None,
    *,
    max_growth_step: int = DEFAULT_MAX_GROWTH_STEP,
    max_initial_hard_failures: int = DEFAULT_MAX_INITIAL_HARD_FAILURES,
    trusted_pass_index: Optional[int] = None,
    known_boundary_index: Optional[int] = None,
) -> Generator[Any, Any, Dict[str, Any]]:
    """
    Find the highest-index usable candidate with safety-first growth + bisection.

    `probe(index, require_full)` must be a generator (its yields are forwarded,
    so UI progress payloads flow through) returning a dict:
        {"outcome": "pass" | "boundary_fail" | "hard_fail" | "cancelled",
         "reused_outcome": bool}

    A fresh campaign probes candidate zero first. A resumed campaign may start
    from `trusted_pass_index`, whose adapter must return a saved full-pass
    measurement without executing it. `known_boundary_index` caps growth at
    the lowest saved OOM/threshold point. Its adapter is still called so it can
    reuse that measurement (or recover via a different config), but no more
    aggressive candidate is launched first. Until a boundary is known, probes
    grow by at most `max_growth_step`; after that, bisection settles the missing
    interval.

    A persistent non-VRAM `hard_fail` is treated as an indeterminate hole, not
    as an OOM and not as a reason to abandon every lower candidate.  The search
    continues around the hole.  If such a hole remains above the selected pass,
    `indeterminate_above_best` is true and callers must not cache the result as
    a proven frontier.

    Returns the selected index plus boundary/indeterminate metadata. To avoid
    rerunning a global CLI/model error across the whole grid, the search stops
    after `max_initial_hard_failures` consecutive safety candidates fail before
    any pass or clean VRAM boundary.
    """
    candidate_count = max(0, int(num_candidates))
    passes: set[int] = set()
    observed: set[int] = set()
    hard_failures: set[int] = set()
    live_initial_hard_failures: set[int] = set()
    hi = int(candidate_count)
    boundary_failed = False
    stopped: Optional[str] = None

    try:
        saved_boundary = int(known_boundary_index) if known_boundary_index is not None else None
    except Exception:
        saved_boundary = None
    if saved_boundary is None or not (0 <= saved_boundary < candidate_count):
        saved_boundary = None

    try:
        saved_pass = int(trusted_pass_index) if trusted_pass_index is not None else None
    except Exception:
        saved_pass = None
    if (
        saved_pass is None
        or not (0 <= saved_pass < hi)
        or (saved_boundary is not None and saved_pass >= saved_boundary)
    ):
        saved_pass = None

    try:
        growth_step = max(1, int(max_growth_step))
    except Exception:
        growth_step = DEFAULT_MAX_GROWTH_STEP
    try:
        initial_hard_limit = max(1, int(max_initial_hard_failures))
    except Exception:
        initial_hard_limit = DEFAULT_MAX_INITIAL_HARD_FAILURES

    hinted_index: Optional[int] = None
    try:
        if initial_index is not None and 0 <= int(initial_index) < candidate_count:
            hinted_index = int(initial_index)
    except Exception:
        hinted_index = None

    def _lo() -> int:
        eligible = [idx for idx in passes if idx < hi]
        return max(eligible) if eligible else -1

    def _next_unknown_index() -> Optional[int]:
        lo = _lo()
        search_hi = hi
        if hi == candidate_count and saved_boundary is not None:
            search_hi = min(search_hi, saved_boundary + 1)
        unknown = [idx for idx in range(lo + 1, search_hi) if idx not in observed]
        if not unknown:
            return None

        # With no safe baseline, always choose the lightest remaining point.
        if lo < 0:
            return min(unknown)

        if hi < candidate_count:
            # A clean resource boundary is bracketed.  Pick the unmeasured point
            # nearest the midpoint; prefer the higher point on an exact tie.
            midpoint = (lo + hi) / 2.0
            return min(unknown, key=lambda idx: (abs(float(idx) - midpoint), -idx))

        # No boundary yet: grow only a bounded distance beyond the last pass.
        desired = min(candidate_count - 1, lo + growth_step)
        if saved_boundary is not None:
            desired = min(desired, saved_boundary)
        if hinted_index is not None and hinted_index > lo:
            desired = min(desired, hinted_index)
        in_window = [idx for idx in unknown if idx <= desired]
        if in_window:
            return max(in_window)

        # The bounded window can consist entirely of hard-failure holes.  Move
        # to the next lightest unknown candidate rather than making a large jump.
        return min(unknown)

    pending: Optional[int] = saved_pass if saved_pass is not None else (0 if candidate_count > 0 else None)
    while stopped is None:
        if pending is not None:
            was_saved_boundary = saved_boundary is not None and int(pending) == saved_boundary
            res = yield from probe(int(pending), False)
            kind = str((res or {}).get("outcome") or "hard_fail")
            observed.add(int(pending))
            if kind == "pass":
                passes.add(int(pending))
            elif kind == "boundary_fail":
                boundary_failed = True
                hi = min(hi, int(pending))
            elif kind == "cancelled":
                stopped = "cancelled"
                break
            else:
                hard_failures.add(int(pending))
                passes.discard(int(pending))
                if not bool((res or {}).get("reused_outcome", False)):
                    live_initial_hard_failures.add(int(pending))
                if (
                    not passes
                    and not boundary_failed
                    and len(live_initial_hard_failures) >= initial_hard_limit
                ):
                    stopped = "hard_fail"
                    break
            if was_saved_boundary:
                saved_boundary = None
            pending = _next_unknown_index()
            continue
        break

    lo = _lo()
    if stopped is None and lo < 0 and hard_failures and not boundary_failed:
        stopped = "hard_fail"
    indeterminate_above_best = bool(
        lo >= 0 and any(lo < idx < hi for idx in hard_failures)
    )
    frontier_verified = bool(
        lo >= 0
        and not indeterminate_above_best
        and (hi < candidate_count or lo == candidate_count - 1)
    )
    return {
        "best_idx": (lo if lo >= 0 else None),
        "stopped": stopped,
        "boundary_failed": bool(boundary_failed),
        "frontier_verified": frontier_verified,
        "hard_failed_indices": sorted(hard_failures),
        "indeterminate_above_best": indeterminate_above_best,
    }


def ratio_diff(lhs: Any, rhs: Any) -> float:
    """Return a stable relative difference, or infinity for invalid values."""
    try:
        lhs_f = float(lhs)
        rhs_f = float(rhs)
    except Exception:
        return float("inf")
    return abs(lhs_f - rhs_f) / max(abs(lhs_f), abs(rhs_f), 1e-9)


def resolution_signature_distance(
    candidate: Dict[str, Any], expected: Dict[str, Any]
) -> Tuple[float, float, float]:
    """Return target pixel, aspect-ratio, and maximum dimension differences."""
    if not isinstance(candidate, dict) or not isinstance(expected, dict):
        return (float("inf"), float("inf"), float("inf"))
    try:
        cand_w = int(candidate.get("target_width") or 0)
        cand_h = int(candidate.get("target_height") or 0)
        exp_w = int(expected.get("target_width") or 0)
        exp_h = int(expected.get("target_height") or 0)
        cand_px = float(candidate.get("target_pixels") or (cand_w * cand_h))
        exp_px = float(expected.get("target_pixels") or (exp_w * exp_h))
    except Exception:
        return (float("inf"), float("inf"), float("inf"))
    if min(cand_w, cand_h, exp_w, exp_h) <= 0 or min(cand_px, exp_px) <= 0:
        return (float("inf"), float("inf"), float("inf"))
    pixel_diff = ratio_diff(cand_px, exp_px)
    aspect_diff = ratio_diff(float(cand_w) / float(cand_h), float(exp_w) / float(exp_h))
    dimension_diff = max(ratio_diff(cand_w, exp_w), ratio_diff(cand_h, exp_h))
    return (pixel_diff, aspect_diff, dimension_diff)


def resolution_signatures_compatible(
    candidate: Dict[str, Any],
    expected: Dict[str, Any],
    *,
    max_pixel_diff: float = DEFAULT_MAX_PIXEL_DIFF,
    max_aspect_diff: float = DEFAULT_MAX_ASPECT_DIFF,
    max_dimension_diff: float = DEFAULT_MAX_DIMENSION_DIFF,
) -> bool:
    """Only reuse measurements for similar output and effective model-input shapes."""
    pixel_diff, aspect_diff, dimension_diff = resolution_signature_distance(candidate, expected)
    target_compatible = bool(
        pixel_diff <= float(max_pixel_diff)
        and aspect_diff <= float(max_aspect_diff)
        and dimension_diff <= float(max_dimension_diff)
    )
    if not target_compatible:
        return False

    try:
        cand_w = int(candidate.get("effective_input_width") or 0)
        cand_h = int(candidate.get("effective_input_height") or 0)
        exp_w = int(expected.get("effective_input_width") or 0)
        exp_h = int(expected.get("effective_input_height") or 0)
    except Exception:
        return False
    has_candidate_shape = cand_w > 0 and cand_h > 0
    has_expected_shape = exp_w > 0 and exp_h > 0
    if not has_candidate_shape and not has_expected_shape:
        return True
    if not has_candidate_shape or not has_expected_shape:
        return False

    cand_pixels = cand_w * cand_h
    exp_pixels = exp_w * exp_h
    return bool(
        ratio_diff(cand_pixels, exp_pixels) <= float(max_pixel_diff)
        and ratio_diff(float(cand_w) / float(cand_h), float(exp_w) / float(exp_h))
        <= float(max_aspect_diff)
        and max(ratio_diff(cand_w, exp_w), ratio_diff(cand_h, exp_h))
        <= float(max_dimension_diff)
    )


def resolution_signatures_identical(candidate: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    """Require the exact resolved output and effective model-input dimensions."""
    if not isinstance(candidate, dict) or not isinstance(expected, dict):
        return False
    keys = (
        "target_width",
        "target_height",
        "effective_input_width",
        "effective_input_height",
    )
    try:
        candidate_values = tuple(int(candidate.get(key) or 0) for key in keys)
        expected_values = tuple(int(expected.get(key) or 0) for key in keys)
    except Exception:
        return False
    return bool(min(candidate_values) > 0 and candidate_values == expected_values)


def is_verified_autotune_payload(payload: Dict[str, Any]) -> bool:
    """Return whether a saved campaign is eligible for instant reuse."""
    if not isinstance(payload, dict):
        return False
    # A frontier result is reusable only after the whole campaign reached its
    # final checkpoint. Partial, cancelled, and failed campaigns are resumable
    # probe history, never final cache entries.
    if str(payload.get("status") or "").strip().lower() != "completed":
        return False
    if not bool(payload.get("finalized", False)) or not bool(payload.get("frontier_verified", False)):
        return False
    best = payload.get("best_config")
    return isinstance(best, dict) and bool(best)


def persisted_autotune_status(
    result_reason: Any,
    *,
    finalized: bool,
    frontier_verified: bool,
) -> str:
    """Store one unambiguous completion state while preserving result_reason."""
    reason = str(result_reason or "running")
    return "completed" if finalized and frontier_verified else reason


def ambient_adjusted_peak_gb(
    peak_gb: Any,
    payload: Dict[str, Any],
    current_ambient_used_gb: float,
) -> float:
    """Conservatively carry extra current background VRAM into an old peak."""
    try:
        peak = max(0.0, float(peak_gb or 0.0))
        current_ambient = max(0.0, float(current_ambient_used_gb or 0.0))
        cached_gpu = payload.get("gpu") if isinstance(payload.get("gpu"), dict) else {}
        cached_ambient = max(
            0.0,
            float(
                (cached_gpu or {}).get("ambient_used_gb")
                or payload.get("ambient_used_gb")
                or 0.0
            ),
        )
    except Exception:
        return 0.0
    return peak + max(0.0, current_ambient - cached_ambient)


def ambient_adjusted_min_device_free_gb(
    recorded_min_device_free_gb: Any,
    original_peak_gb: Any,
    adjusted_peak_gb: Any,
    aggregate_free_gb: Any,
) -> float:
    """Conservatively update per-device headroom when only total ambient drift is known."""
    try:
        aggregate_free = max(0.0, float(aggregate_free_gb))
        recorded_free = float(recorded_min_device_free_gb)
        original_peak = max(0.0, float(original_peak_gb))
        adjusted_peak = max(0.0, float(adjusted_peak_gb))
    except Exception:
        return 0.0
    if not all(
        math.isfinite(value)
        for value in (aggregate_free, recorded_free, original_peak, adjusted_peak)
    ):
        return 0.0
    ambient_increase = max(0.0, adjusted_peak - original_peak)
    return min(aggregate_free, max(0.0, recorded_free - ambient_increase))


def cached_ambient_allows_frontier_reuse(
    payload: Dict[str, Any],
    current_ambient_used_gb: Any,
    *,
    max_drop_gb: Any = DEFAULT_MAX_AMBIENT_DROP_FOR_REUSE_GB,
) -> bool:
    """Return whether an old failure boundary is still an optimality proof."""
    if not isinstance(payload, dict):
        return False
    try:
        cached_gpu = payload.get("gpu") if isinstance(payload.get("gpu"), dict) else {}
        cached_ambient = max(
            0.0,
            float(
                (cached_gpu or {}).get("ambient_used_gb")
                or payload.get("ambient_used_gb")
                or 0.0
            ),
        )
        current_ambient = max(0.0, float(current_ambient_used_gb or 0.0))
        tolerance = max(0.0, float(max_drop_gb))
    except Exception:
        return False
    if not all(math.isfinite(value) for value in (cached_ambient, current_ambient, tolerance)):
        return False
    return current_ambient + tolerance >= cached_ambient


def cached_best_has_headroom(
    payload: Dict[str, Any],
    expected_signature: Dict[str, Any],
    min_free_vram_gb: float,
    *,
    current_ambient_used_gb: float = 0.0,
) -> bool:
    """Re-evaluate a cached peak against today's GPU size, reserve, and ambient use.

    A materially lower current baseline intentionally invalidates direct reuse:
    the old winner may be safe but unnecessarily conservative because a higher
    candidate that failed while another workload occupied VRAM may fit now.
    """
    if not isinstance(payload, dict) or not isinstance(expected_signature, dict):
        return False
    best = payload.get("best_config")
    if not isinstance(best, dict):
        return False
    try:
        peak_gb = float(best.get("measured_peak_vram_used_gb") or 0.0)
        total_gb = float(expected_signature.get("gpu_total_vram_gb") or 0.0)
    except Exception:
        return False
    if peak_gb <= 0 or total_gb <= 0:
        return False
    try:
        current_ambient_gb = max(0.0, float(current_ambient_used_gb or 0.0))
    except Exception:
        return False
    if not cached_ambient_allows_frontier_reuse(payload, current_ambient_gb):
        return False
    adjusted_peak_gb = ambient_adjusted_peak_gb(
        peak_gb,
        payload,
        current_ambient_gb,
    )
    if (total_gb - adjusted_peak_gb) < float(min_free_vram_gb):
        return False
    try:
        measured_min_device_free_gb = float(best.get("min_device_free_vram_gb"))
    except Exception:
        return True
    if not math.isfinite(measured_min_device_free_gb):
        return False
    ambient_increase_gb = max(0.0, adjusted_peak_gb - peak_gb)
    return (
        measured_min_device_free_gb - ambient_increase_gb
    ) >= float(min_free_vram_gb)


def autotune_launch_headroom(
    total_vram_gb: Any,
    ambient_used_gb: Any,
    min_free_vram_gb: Any,
    *,
    launch_margin_gb: Any = DEFAULT_LAUNCH_MARGIN_GB,
) -> Tuple[bool, float, float]:
    """Check that the GPU is not already too full to launch a probe safely."""
    try:
        total = float(total_vram_gb)
        ambient = max(0.0, float(ambient_used_gb))
        reserve = max(0.0, float(min_free_vram_gb))
        margin = max(0.0, float(launch_margin_gb))
    except Exception:
        return (False, 0.0, 0.0)
    if not all(math.isfinite(value) for value in (total, ambient, reserve, margin)):
        return (False, 0.0, 0.0)
    if total <= 0:
        return (False, 0.0, reserve + margin)
    available = max(0.0, total - ambient)
    required = reserve + margin
    return (available >= required, available, required)


def is_vram_boundary_outcome(outcome: Dict[str, Any], min_free_vram_gb: float) -> bool:
    """Separate clean VRAM boundaries from unrelated process/runtime failures."""
    if not isinstance(outcome, dict):
        return False
    # Explicit allocator OOMs and watchdog stops are resource proof even when
    # the telemetry sampler raced process termination and missed its last row.
    if bool(outcome.get("oom", False)):
        return True
    if str(outcome.get("probe_cancel_reason") or "").strip().lower() == "threshold_reached":
        return True
    if not bool(outcome.get("telemetry_ok", False)):
        return False
    try:
        free_gb = float(outcome.get("estimated_free_gb", float("inf")))
    except Exception:
        free_gb = float("inf")
    # Low measured headroom is a resource boundary even if process termination
    # raced the watchdog.  A nonzero exit with ample headroom remains a hard fail.
    return free_gb < float(min_free_vram_gb)


def resume_frontier_hints(
    outcomes: Sequence[Optional[Dict[str, Any]]],
    min_free_vram_gb: float,
) -> Tuple[Optional[int], Optional[int]]:
    """Return the strongest saved full pass and first saved VRAM boundary."""
    passes: List[int] = []
    boundaries: List[int] = []
    for index, outcome in enumerate(outcomes or ()):
        if not isinstance(outcome, dict):
            continue
        try:
            returncode_ok = int(outcome.get("returncode", 1)) == 0
        except Exception:
            returncode_ok = False
        clean_full_pass = bool(
            outcome.get("passed", False)
            and returncode_ok
            and bool(outcome.get("telemetry_ok", False))
            and not bool(outcome.get("oom", False))
            and not str(outcome.get("probe_cancel_reason") or "").strip()
        )
        if clean_full_pass:
            passes.append(index)
        elif is_vram_boundary_outcome(outcome, min_free_vram_gb):
            boundaries.append(index)

    boundary = min(boundaries) if boundaries else None
    eligible_passes = [index for index in passes if boundary is None or index < boundary]
    return (max(eligible_passes) if eligible_passes else None, boundary)


def is_reusable_vram_boundary_outcome(
    outcome: Dict[str, Any],
    min_free_vram_gb: float,
    *,
    recorded_min_free_vram_gb: Optional[float] = None,
) -> bool:
    """Classify a saved failure against the current reserve without stale cutoffs.

    A watchdog stop from an equal or stricter historical reserve remains a valid
    boundary. When the user lowers the reserve, however, an old early stop that
    still has enough headroom for the new target is incomplete and must be run
    again rather than reused as a failure.
    """
    if not isinstance(outcome, dict):
        return False
    if bool(outcome.get("oom", False)):
        return True

    try:
        current_target = max(0.0, float(min_free_vram_gb))
        free_gb = float(outcome.get("estimated_free_gb", float("inf")))
    except Exception:
        return False
    telemetry_ok = bool(outcome.get("telemetry_ok", False))
    if telemetry_ok and free_gb < current_target:
        return True

    reason = str(outcome.get("probe_cancel_reason") or "").strip().lower()
    if reason != "threshold_reached":
        return False
    try:
        recorded_target = float(recorded_min_free_vram_gb)
    except Exception:
        return False
    if not math.isfinite(recorded_target):
        return False
    return current_target + 0.05 >= max(0.0, recorded_target)


def predict_frontier_index(
    candidate_values: Sequence[Any],
    measured_peaks_by_value: Dict[Any, float],
    budget_gb: float,
) -> Optional[int]:
    """
    Predict the pass/fail frontier from compatible historical measured peaks.

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
# FlashVSR+ DiT tile grid (mirrors ComfyUI-FlashVSR_Stable/nodes.py)
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
