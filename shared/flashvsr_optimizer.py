"""
FlashVSR parameter optimization helpers.

The optimizer targets a configurable VRAM budget and returns a constraint-aware
setting proposal for FlashVSR+ runs.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

from shared.gpu_utils import (
    GPUInfo,
    get_gpu_info,
    get_most_powerful_gpu,
    normalize_global_gpu_device,
)
from shared.path_utils import get_media_dimensions, normalize_path
from shared.resolution_calculator import UpscalePlan, estimate_fixed_scale_upscale_plan_from_dims

MIN_TILE_SIZE = 128
MAX_TILE_SIZE = 1024
TILE_STEP = 32
TARGET_OVERLAP = 48
MIN_OVERLAP = 24
TARGET_FRAME_CHUNK = 450
MIN_FRAME_CHUNK = 32
DEFAULT_VRAM_RESERVE_GB = 2.0
DEFAULT_ESTIMATION_MARGIN_GB = 1.25
MIN_ESTIMATION_MARGIN_GB = 0.35
MAX_ESTIMATION_MARGIN_GB = 3.5

FRAME_CHUNK_FALLBACKS: Tuple[int, ...] = (450, 384, 320, 256, 224, 192, 160, 128, 96, 64, 48, 32)
MAX_EDGE_REDUCTION_CANDIDATES: Tuple[int, ...] = (
    8192,
    7680,
    6144,
    5120,
    4096,
    3840,
    3072,
    2880,
    2560,
    2304,
    2160,
    2048,
    1920,
    1792,
    1600,
    1536,
    1440,
    1280,
    1152,
    1080,
    960,
    900,
    864,
    768,
    720,
)


@dataclass(frozen=True)
class FlashVSRBudget:
    total_vram_gb: float
    target_vram_gb: float
    reserve_vram_gb: float
    gpu_label: str
    gpu_id: Optional[int]


@dataclass(frozen=True)
class FlashVSRResolutionState:
    scale: int
    max_target_resolution: int
    preprocess_width: int
    preprocess_height: int
    output_width: int
    output_height: int
    plan: UpscalePlan
    stage_label: str


@dataclass(frozen=True)
class FlashVSROptimizedSettings:
    success: bool
    tile_size: int
    overlap: int
    frame_chunk_size: int
    scale: int
    max_target_resolution: int
    tiled_dit: bool
    tiled_vae: bool
    estimated_peak_vram_gb: float
    budget: FlashVSRBudget
    preprocess_width: int
    preprocess_height: int
    output_width: int
    output_height: int
    stage_label: str
    calibration_multiplier: float
    calibration_samples: int
    notes: Tuple[str, ...]
    estimation_safety_margin_gb: float = 0.0
    estimated_guarded_vram_gb: float = 0.0


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        v = int(float(value))
    except Exception:
        v = int(default)
    return max(int(low), min(int(high), v))


def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = float(default)
    return max(float(low), min(float(high), v))


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _default_records_path() -> Path:
    base_dir = Path(__file__).resolve().parents[1]
    return base_dir / "outputs" / "flashvsr_vram_sweeps" / "flashvsr_vram_records.csv"


def _resolve_gpu_budget(
    *,
    selected_gpu_value: Any,
    reserve_vram_gb: float = DEFAULT_VRAM_RESERVE_GB,
    gpus: Optional[List[GPUInfo]] = None,
) -> FlashVSRBudget:
    devices = list(gpus) if gpus is not None else list(get_gpu_info())
    reserve = max(0.0, float(reserve_vram_gb))
    normalized = normalize_global_gpu_device(selected_gpu_value)

    if normalized == "cpu" or not devices:
        return FlashVSRBudget(
            total_vram_gb=0.0,
            target_vram_gb=0.0,
            reserve_vram_gb=reserve,
            gpu_label="CPU",
            gpu_id=None,
        )

    picked: Optional[GPUInfo] = None
    if normalized and normalized.isdigit():
        for gpu in devices:
            if str(gpu.id) == normalized:
                picked = gpu
                break
    if picked is None:
        picked = get_most_powerful_gpu(devices)
    if picked is None:
        return FlashVSRBudget(
            total_vram_gb=0.0,
            target_vram_gb=0.0,
            reserve_vram_gb=reserve,
            gpu_label="CPU",
            gpu_id=None,
        )

    total = max(0.0, float(picked.total_memory_gb or 0.0))
    target = max(0.0, total - reserve)
    label = f"GPU {int(picked.id)}: {picked.name}"
    return FlashVSRBudget(
        total_vram_gb=total,
        target_vram_gb=target,
        reserve_vram_gb=reserve,
        gpu_label=label,
        gpu_id=int(picked.id),
    )


def _safe_normalized_path(path_value: Any) -> Optional[str]:
    try:
        return normalize_path(str(path_value or "").strip())
    except Exception:
        return None


def _build_max_edge_reductions(
    *,
    input_width: int,
    input_height: int,
    scale: int,
    current_max_edge: int,
) -> List[int]:
    natural_edge = max(1, int(max(input_width, input_height) * int(scale)))
    current = int(current_max_edge)
    effective_start = natural_edge if current <= 0 else min(current, natural_edge)

    out: List[int] = []
    seen = set()
    for cand in MAX_EDGE_REDUCTION_CANDIDATES:
        c = int(cand)
        if c >= effective_start:
            continue
        if c < 720:
            continue
        if c in seen:
            continue
        out.append(c)
        seen.add(c)
    return out


def _build_resolution_states(
    *,
    input_width: int,
    input_height: int,
    requested_scale: int,
    current_max_edge: int,
    pre_downscale_then_upscale: bool,
) -> List[FlashVSRResolutionState]:
    states: List[FlashVSRResolutionState] = []
    seen = set()

    def add_state(scale_val: int, max_edge_val: int, stage_label: str) -> None:
        plan = estimate_fixed_scale_upscale_plan_from_dims(
            int(input_width),
            int(input_height),
            requested_scale=float(scale_val),
            model_scale=int(scale_val),
            max_edge=int(max_edge_val),
            force_pre_downscale=bool(pre_downscale_then_upscale),
        )
        key = (
            int(scale_val),
            int(max_edge_val),
            int(plan.preprocess_width),
            int(plan.preprocess_height),
            int(plan.resize_width),
            int(plan.resize_height),
        )
        if key in seen:
            return
        seen.add(key)
        states.append(
            FlashVSRResolutionState(
                scale=int(scale_val),
                max_target_resolution=int(max_edge_val),
                preprocess_width=int(plan.preprocess_width),
                preprocess_height=int(plan.preprocess_height),
                output_width=int(plan.resize_width),
                output_height=int(plan.resize_height),
                plan=plan,
                stage_label=str(stage_label),
            )
        )

    max_edge_i = max(0, int(current_max_edge))
    if requested_scale == 4:
        add_state(4, max_edge_i, "base")
        add_state(2, max_edge_i, "scale_4_to_2")
        for reduced in _build_max_edge_reductions(
            input_width=input_width,
            input_height=input_height,
            scale=2,
            current_max_edge=max_edge_i,
        ):
            add_state(2, int(reduced), "scale_4_to_2_then_reduce_max_edge")
    else:
        add_state(2, max_edge_i, "base")
        for reduced in _build_max_edge_reductions(
            input_width=input_width,
            input_height=input_height,
            scale=2,
            current_max_edge=max_edge_i,
        ):
            add_state(2, int(reduced), "reduce_max_edge")

    return states


def _iter_tile_sizes_desc() -> Iterable[int]:
    v = int(MAX_TILE_SIZE)
    while v >= int(MIN_TILE_SIZE):
        yield int(v)
        v -= int(TILE_STEP)


def _raw_estimate_peak_vram_gb(
    *,
    preprocess_width: int,
    preprocess_height: int,
    scale: int,
    mode: str,
    precision: str,
    vae_model: str,
    tile_size: int,
    overlap: int,
    frame_chunk_size: int,
    keep_models_on_cpu: bool,
    tiled_dit: bool,
    tiled_vae: bool,
    stream_decode: bool,
) -> float:
    mode_key = str(mode or "full").strip().lower()
    mode_base = {
        "tiny": 4.3,
        "tiny-long": 3.9,
        "full": 6.5,
    }.get(mode_key, 6.5)

    scale_i = 2 if int(scale) <= 2 else 4
    scale_add = 1.0 if scale_i == 2 else 2.3

    vae_key = str(vae_model or "Wan2.2").strip().lower()
    vae_add = {
        "wan2.1": 0.9,
        "wan2.2": 1.2,
        "lightvae_w2.1": -0.6,
        "tae_w2.2": -0.8,
        "lighttae_hy1.5": -1.0,
    }.get(vae_key, 0.8)

    precision_key = str(precision or "bf16").strip().lower()
    precision_mul = {
        "fp16": 0.95,
        "bf16": 1.0,
        "auto": 1.0,
    }.get(precision_key, 1.0)

    pixels = max(1.0, float(preprocess_width) * float(preprocess_height))
    pixel_ref = 1920.0 * 1080.0
    pixel_term = 1.30 * ((pixels / pixel_ref) ** 0.38 - 1.0)
    if not tiled_dit:
        pixel_term *= 1.8

    chunk = max(float(MIN_FRAME_CHUNK), float(frame_chunk_size))
    chunk_term = 6.2 * ((chunk / float(TARGET_FRAME_CHUNK)) ** 0.9)

    tile = max(float(MIN_TILE_SIZE), float(tile_size))
    ov = max(8.0, float(overlap))
    tile_effective = tile + max(0.0, ov - 24.0) * 0.60
    tile_term = 3.2 * ((tile_effective / 256.0) ** 2.05) if tiled_dit else 8.0

    cpu_offload_term = -0.6 if bool(keep_models_on_cpu) else 0.0
    tiled_vae_term = 0.8 if bool(tiled_vae) else 0.0
    stream_decode_term = 1.0 if bool(stream_decode) and mode_key in {"tiny", "tiny-long"} else 0.0

    estimate = (
        mode_base
        + scale_add
        + vae_add
        + pixel_term
        + chunk_term
        + tile_term
        + cpu_offload_term
        + tiled_vae_term
        + stream_decode_term
    )
    estimate = max(2.0, estimate * precision_mul)
    return float(estimate)


def _load_vram_calibration_samples(records_csv_path: Optional[str]) -> List[Dict[str, Any]]:
    path = Path(records_csv_path) if records_csv_path else _default_records_path()
    if not path.exists() or not path.is_file():
        return []

    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                try:
                    success = _to_bool(row.get("success"), False)
                    peak = float(row.get("peak_vram_gb") or 0.0)
                    if not success or peak <= 0:
                        continue
                    rows.append(row)
                except Exception:
                    continue
    except Exception:
        return []
    return rows


def _quantile(values: List[float], q: float, default: float = 0.0) -> float:
    if not values:
        return float(default)
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return float(vals[0])
    qf = _clamp_float(q, 0.0, 1.0, 0.5)
    pos = qf * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def _safe_ratio_distance(a: float, b: float) -> float:
    aa = max(1e-9, float(a))
    bb = max(1e-9, float(b))
    return float(abs(math.log(aa / bb)))


def _normalized_scale(value: Any) -> int:
    try:
        return 2 if int(float(value)) <= 2 else 4
    except Exception:
        return 4


def _calibration_stats(
    *,
    mode: str,
    scale: int,
    precision: str,
    vae_model: str,
    preprocess_width: int,
    preprocess_height: int,
    tile_size: int,
    overlap: int,
    frame_chunk_size: int,
    keep_models_on_cpu: bool,
    tiled_dit: bool,
    tiled_vae: bool,
    stream_decode: bool,
    records_csv_path: Optional[str],
    rows: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[float, int, float]:
    samples = list(rows) if rows is not None else _load_vram_calibration_samples(records_csv_path)
    if not samples:
        return 1.0, 0, float(DEFAULT_ESTIMATION_MARGIN_GB)

    target_mode = str(mode or "full").strip().lower()
    target_scale = _normalized_scale(scale)
    target_pixels = max(1.0, float(preprocess_width) * float(preprocess_height))
    target_precision = str(precision or "bf16").strip().lower()
    target_vae = str(vae_model or "Wan2.2").strip().lower()
    target_chunk = max(float(MIN_FRAME_CHUNK), float(frame_chunk_size))
    target_overlap = float(max(8, int(overlap)))
    target_tile_effective = max(
        float(MIN_TILE_SIZE),
        float(tile_size) + max(0.0, target_overlap - 24.0) * 0.60,
    )

    candidates: List[Tuple[float, float, float, float]] = []
    global_ratios: List[float] = []

    for row in samples:
        try:
            row_mode = str(row.get("mode") or "").strip().lower()
            row_scale = _normalized_scale(row.get("scale"))
            if row_mode != target_mode or row_scale != target_scale:
                continue

            row_w = max(1, int(float(row.get("preprocess_width") or 0)))
            row_h = max(1, int(float(row.get("preprocess_height") or 0)))
            row_pixels = float(row_w * row_h)
            row_tile = _clamp_int(row.get("tile_size"), MIN_TILE_SIZE, MAX_TILE_SIZE, 256)
            row_overlap = _clamp_int(row.get("overlap"), 8, 512, TARGET_OVERLAP)
            row_chunk = _clamp_int(row.get("frame_chunk_size"), MIN_FRAME_CHUNK, 10000, TARGET_FRAME_CHUNK)
            row_precision = str(row.get("precision") or target_precision).strip().lower()
            row_vae = str(row.get("vae_model") or vae_model or "Wan2.2").strip().lower()
            row_keep_cpu = _to_bool(row.get("keep_models_on_cpu"), keep_models_on_cpu)
            row_tiled_dit = _to_bool(row.get("tiled_dit"), tiled_dit)
            row_tiled_vae = _to_bool(row.get("tiled_vae"), tiled_vae)
            row_stream_decode = _to_bool(row.get("stream_decode"), stream_decode)

            predicted = _raw_estimate_peak_vram_gb(
                preprocess_width=row_w,
                preprocess_height=row_h,
                scale=row_scale,
                mode=row_mode,
                precision=row_precision,
                vae_model=row_vae,
                tile_size=row_tile,
                overlap=row_overlap,
                frame_chunk_size=row_chunk,
                keep_models_on_cpu=row_keep_cpu,
                tiled_dit=row_tiled_dit,
                tiled_vae=row_tiled_vae,
                stream_decode=row_stream_decode,
            )
            measured = float(row.get("peak_vram_gb") or 0.0)
            if predicted <= 0.01 or measured <= 0.01:
                continue

            ratio = measured / predicted
            global_ratios.append(float(ratio))

            row_tile_effective = max(float(MIN_TILE_SIZE), float(row_tile) + max(0.0, float(row_overlap) - 24.0) * 0.60)
            dist = 0.0
            dist += 1.35 * _safe_ratio_distance(row_pixels, target_pixels)
            dist += 0.92 * _safe_ratio_distance(float(row_chunk), target_chunk)
            dist += 0.78 * _safe_ratio_distance(row_tile_effective, target_tile_effective)
            dist += abs(float(row_overlap) - target_overlap) / 90.0
            if row_precision != target_precision:
                dist += 0.45
            if row_vae != target_vae:
                dist += 0.65
            if row_keep_cpu != bool(keep_models_on_cpu):
                dist += 0.20
            if row_tiled_dit != bool(tiled_dit):
                dist += 1.00
            if row_tiled_vae != bool(tiled_vae):
                dist += 0.70
            if row_stream_decode != bool(stream_decode):
                dist += 0.60

            candidates.append((float(dist), float(ratio), float(measured), float(predicted)))
        except Exception:
            continue

    if not global_ratios:
        return 1.0, 0, float(DEFAULT_ESTIMATION_MARGIN_GB)

    if not candidates:
        med = float(median(global_ratios))
        mul = _clamp_float(med, 0.65, 1.55, 1.0)
        return mul, len(global_ratios), float(DEFAULT_ESTIMATION_MARGIN_GB)

    candidates.sort(key=lambda item: item[0])
    top_k = min(len(candidates), 24)
    top = candidates[:top_k]

    weighted_num = 0.0
    weighted_den = 0.0
    for dist, ratio, _, _ in top:
        w = 1.0 / (0.20 + dist * dist)
        weighted_num += w * ratio
        weighted_den += w
    if weighted_den <= 0.0:
        mul = _clamp_float(float(median(global_ratios)), 0.65, 1.55, 1.0)
    else:
        mul = _clamp_float(weighted_num / weighted_den, 0.65, 1.55, 1.0)

    positive_residuals: List[float] = []
    for _, _, measured, predicted in top:
        positive_residuals.append(max(0.0, float(measured - predicted * mul)))

    margin_core = _quantile(positive_residuals, 0.85, default=0.0)
    if len(top) >= 8:
        margin = margin_core + 0.12
    elif len(top) >= 4:
        margin = margin_core + 0.28
    else:
        margin = max(float(DEFAULT_ESTIMATION_MARGIN_GB), margin_core + 0.48)

    margin = _clamp_float(margin, MIN_ESTIMATION_MARGIN_GB, MAX_ESTIMATION_MARGIN_GB, DEFAULT_ESTIMATION_MARGIN_GB)
    return float(mul), len(candidates), float(margin)


def _calibration_multiplier(
    *,
    mode: str,
    scale: int,
    precision: str,
    vae_model: str,
    preprocess_width: int,
    preprocess_height: int,
    keep_models_on_cpu: bool,
    tiled_dit: bool,
    tiled_vae: bool,
    stream_decode: bool,
    records_csv_path: Optional[str],
) -> Tuple[float, int]:
    mul, n, _ = _calibration_stats(
        mode=mode,
        scale=scale,
        precision=precision,
        vae_model=vae_model,
        preprocess_width=preprocess_width,
        preprocess_height=preprocess_height,
        tile_size=MIN_TILE_SIZE,
        overlap=TARGET_OVERLAP,
        frame_chunk_size=TARGET_FRAME_CHUNK,
        keep_models_on_cpu=keep_models_on_cpu,
        tiled_dit=tiled_dit,
        tiled_vae=tiled_vae,
        stream_decode=stream_decode,
        records_csv_path=records_csv_path,
        rows=None,
    )
    return float(mul), int(n)


def _estimate_flashvsr_peak_vram_with_margin_gb(
    *,
    preprocess_width: int,
    preprocess_height: int,
    scale: int,
    mode: str,
    precision: str,
    vae_model: str,
    tile_size: int,
    overlap: int,
    frame_chunk_size: int,
    keep_models_on_cpu: bool,
    tiled_dit: bool,
    tiled_vae: bool,
    stream_decode: bool,
    records_csv_path: Optional[str] = None,
    calibration_rows: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[float, float, float, int, float]:
    raw = _raw_estimate_peak_vram_gb(
        preprocess_width=preprocess_width,
        preprocess_height=preprocess_height,
        scale=scale,
        mode=mode,
        precision=precision,
        vae_model=vae_model,
        tile_size=tile_size,
        overlap=overlap,
        frame_chunk_size=frame_chunk_size,
        keep_models_on_cpu=keep_models_on_cpu,
        tiled_dit=tiled_dit,
        tiled_vae=tiled_vae,
        stream_decode=stream_decode,
    )
    calib_mul, calib_samples, margin = _calibration_stats(
        mode=mode,
        scale=scale,
        precision=precision,
        vae_model=vae_model,
        preprocess_width=preprocess_width,
        preprocess_height=preprocess_height,
        tile_size=tile_size,
        overlap=overlap,
        frame_chunk_size=frame_chunk_size,
        keep_models_on_cpu=keep_models_on_cpu,
        tiled_dit=tiled_dit,
        tiled_vae=tiled_vae,
        stream_decode=stream_decode,
        records_csv_path=records_csv_path,
        rows=calibration_rows,
    )
    estimated = float(raw * calib_mul)
    guarded = float(estimated + margin)
    return estimated, guarded, float(calib_mul), int(calib_samples), float(margin)


def estimate_flashvsr_peak_vram_gb(
    *,
    preprocess_width: int,
    preprocess_height: int,
    scale: int,
    mode: str,
    precision: str,
    vae_model: str,
    tile_size: int,
    overlap: int,
    frame_chunk_size: int,
    keep_models_on_cpu: bool,
    tiled_dit: bool,
    tiled_vae: bool,
    stream_decode: bool,
    records_csv_path: Optional[str] = None,
) -> Tuple[float, float, int]:
    estimated, _guarded, calib_mul, calib_samples, _margin = _estimate_flashvsr_peak_vram_with_margin_gb(
        preprocess_width=preprocess_width,
        preprocess_height=preprocess_height,
        scale=scale,
        mode=mode,
        precision=precision,
        vae_model=vae_model,
        tile_size=tile_size,
        overlap=overlap,
        frame_chunk_size=frame_chunk_size,
        keep_models_on_cpu=keep_models_on_cpu,
        tiled_dit=tiled_dit,
        tiled_vae=tiled_vae,
        stream_decode=stream_decode,
        records_csv_path=records_csv_path,
        calibration_rows=None,
    )
    return float(estimated), float(calib_mul), int(calib_samples)


def _pick_best_tile_for_state(
    *,
    state: FlashVSRResolutionState,
    mode: str,
    precision: str,
    vae_model: str,
    frame_chunk_size: int,
    overlap: int,
    keep_models_on_cpu: bool,
    tiled_dit: bool,
    tiled_vae: bool,
    stream_decode: bool,
    target_vram_gb: float,
    records_csv_path: Optional[str],
    calibration_rows: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[Tuple[int, float, float, float, int, float]], float]:
    best_over = float("inf")
    for tile in _iter_tile_sizes_desc():
        est, guarded_est, cal_mul, cal_n, margin = _estimate_flashvsr_peak_vram_with_margin_gb(
            preprocess_width=state.preprocess_width,
            preprocess_height=state.preprocess_height,
            scale=state.scale,
            mode=mode,
            precision=precision,
            vae_model=vae_model,
            tile_size=tile,
            overlap=overlap,
            frame_chunk_size=frame_chunk_size,
            keep_models_on_cpu=keep_models_on_cpu,
            tiled_dit=tiled_dit,
            tiled_vae=tiled_vae,
            stream_decode=stream_decode,
            records_csv_path=records_csv_path,
            calibration_rows=calibration_rows,
        )
        if guarded_est <= target_vram_gb:
            return (tile, est, guarded_est, cal_mul, cal_n, margin), 0.0
        best_over = min(best_over, guarded_est - target_vram_gb)
    return None, (best_over if best_over != float("inf") else 0.0)


def _fallback_minimum_candidate(
    *,
    final_state: FlashVSRResolutionState,
    mode: str,
    precision: str,
    vae_model: str,
    keep_models_on_cpu: bool,
    tiled_dit: bool,
    tiled_vae: bool,
    stream_decode: bool,
    overlap: int,
    records_csv_path: Optional[str],
    calibration_rows: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[float, float, float, int, float]:
    return _estimate_flashvsr_peak_vram_with_margin_gb(
        preprocess_width=final_state.preprocess_width,
        preprocess_height=final_state.preprocess_height,
        scale=final_state.scale,
        mode=mode,
        precision=precision,
        vae_model=vae_model,
        tile_size=MIN_TILE_SIZE,
        overlap=overlap,
        frame_chunk_size=MIN_FRAME_CHUNK,
        keep_models_on_cpu=keep_models_on_cpu,
        tiled_dit=tiled_dit,
        tiled_vae=tiled_vae,
        stream_decode=stream_decode,
        records_csv_path=records_csv_path,
        calibration_rows=calibration_rows,
    )


def optimize_flashvsr_settings(
    *,
    input_path: str,
    requested_scale: int,
    mode: str,
    precision: str,
    vae_model: str,
    keep_models_on_cpu: bool,
    stream_decode: bool,
    selected_gpu_value: Any,
    max_target_resolution: int,
    pre_downscale_then_upscale: bool,
    reserve_vram_gb: float = DEFAULT_VRAM_RESERVE_GB,
    records_csv_path: Optional[str] = None,
) -> FlashVSROptimizedSettings:
    notes: List[str] = []
    normalized_path = _safe_normalized_path(input_path)
    if not normalized_path:
        budget = _resolve_gpu_budget(selected_gpu_value=selected_gpu_value, reserve_vram_gb=reserve_vram_gb)
        return FlashVSROptimizedSettings(
            success=False,
            tile_size=256,
            overlap=TARGET_OVERLAP,
            frame_chunk_size=TARGET_FRAME_CHUNK,
            scale=2 if int(requested_scale) <= 2 else 4,
            max_target_resolution=max(0, int(max_target_resolution or 0)),
            tiled_dit=True,
            tiled_vae=False,
            estimated_peak_vram_gb=0.0,
            budget=budget,
            preprocess_width=0,
            preprocess_height=0,
            output_width=0,
            output_height=0,
            stage_label="invalid_input",
            calibration_multiplier=1.0,
            calibration_samples=0,
            notes=("Input path is empty or invalid.",),
        )

    dims = get_media_dimensions(normalized_path)
    if not dims:
        budget = _resolve_gpu_budget(selected_gpu_value=selected_gpu_value, reserve_vram_gb=reserve_vram_gb)
        return FlashVSROptimizedSettings(
            success=False,
            tile_size=256,
            overlap=TARGET_OVERLAP,
            frame_chunk_size=TARGET_FRAME_CHUNK,
            scale=2 if int(requested_scale) <= 2 else 4,
            max_target_resolution=max(0, int(max_target_resolution or 0)),
            tiled_dit=True,
            tiled_vae=False,
            estimated_peak_vram_gb=0.0,
            budget=budget,
            preprocess_width=0,
            preprocess_height=0,
            output_width=0,
            output_height=0,
            stage_label="missing_dimensions",
            calibration_multiplier=1.0,
            calibration_samples=0,
            notes=("Could not read input dimensions.",),
        )

    budget = _resolve_gpu_budget(selected_gpu_value=selected_gpu_value, reserve_vram_gb=reserve_vram_gb)
    if budget.target_vram_gb <= 0:
        return FlashVSROptimizedSettings(
            success=False,
            tile_size=MIN_TILE_SIZE,
            overlap=TARGET_OVERLAP,
            frame_chunk_size=MIN_FRAME_CHUNK,
            scale=2 if int(requested_scale) <= 2 else 4,
            max_target_resolution=max(0, int(max_target_resolution or 0)),
            tiled_dit=True,
            tiled_vae=False,
            estimated_peak_vram_gb=0.0,
            budget=budget,
            preprocess_width=int(dims[0]),
            preprocess_height=int(dims[1]),
            output_width=int(dims[0]),
            output_height=int(dims[1]),
            stage_label="cpu_mode",
            calibration_multiplier=1.0,
            calibration_samples=0,
            notes=("CUDA GPU not detected; optimization requires GPU VRAM data.",),
        )

    req_scale = 2 if int(requested_scale) <= 2 else 4
    current_max_edge = max(0, int(max_target_resolution or 0))
    states = _build_resolution_states(
        input_width=int(dims[0]),
        input_height=int(dims[1]),
        requested_scale=req_scale,
        current_max_edge=current_max_edge,
        pre_downscale_then_upscale=bool(pre_downscale_then_upscale),
    )
    if not states:
        fallback_plan = estimate_fixed_scale_upscale_plan_from_dims(
            int(dims[0]),
            int(dims[1]),
            requested_scale=float(req_scale),
            model_scale=int(req_scale),
            max_edge=current_max_edge,
            force_pre_downscale=bool(pre_downscale_then_upscale),
        )
        states = [
            FlashVSRResolutionState(
                scale=req_scale,
                max_target_resolution=current_max_edge,
                preprocess_width=int(fallback_plan.preprocess_width),
                preprocess_height=int(fallback_plan.preprocess_height),
                output_width=int(fallback_plan.resize_width),
                output_height=int(fallback_plan.resize_height),
                plan=fallback_plan,
                stage_label="base",
            )
        ]

    chosen_state: Optional[FlashVSRResolutionState] = None
    chosen_tile = MIN_TILE_SIZE
    chosen_overlap = TARGET_OVERLAP
    chosen_chunk = TARGET_FRAME_CHUNK
    chosen_est = float("inf")
    chosen_guarded_est = float("inf")
    chosen_cal_mul = 1.0
    chosen_cal_samples = 0
    chosen_margin = float(DEFAULT_ESTIMATION_MARGIN_GB)
    success = False
    calibration_rows = _load_vram_calibration_samples(records_csv_path)

    # Phase 1: strict policy (overlap=48, chunk=450), resolution fallbacks first.
    for state in states:
        picked, _ = _pick_best_tile_for_state(
            state=state,
            mode=mode,
            precision=precision,
            vae_model=vae_model,
            frame_chunk_size=TARGET_FRAME_CHUNK,
            overlap=TARGET_OVERLAP,
            keep_models_on_cpu=keep_models_on_cpu,
            tiled_dit=True,
            tiled_vae=False,
            stream_decode=stream_decode,
            target_vram_gb=budget.target_vram_gb,
            records_csv_path=records_csv_path,
            calibration_rows=calibration_rows,
        )
        if picked is None:
            continue
        chosen_state = state
        chosen_tile, chosen_est, chosen_guarded_est, chosen_cal_mul, chosen_cal_samples, chosen_margin = picked
        chosen_overlap = TARGET_OVERLAP
        chosen_chunk = TARGET_FRAME_CHUNK
        success = True
        break

    # Phase 2: if still over budget, keep most-reduced resolution and lower chunk size.
    if not success and states:
        final_state = states[-1]
        for chunk in FRAME_CHUNK_FALLBACKS[1:]:
            picked, _ = _pick_best_tile_for_state(
                state=final_state,
                mode=mode,
                precision=precision,
                vae_model=vae_model,
                frame_chunk_size=int(chunk),
                overlap=TARGET_OVERLAP,
                keep_models_on_cpu=keep_models_on_cpu,
                tiled_dit=True,
                tiled_vae=False,
                stream_decode=stream_decode,
                target_vram_gb=budget.target_vram_gb,
                records_csv_path=records_csv_path,
                calibration_rows=calibration_rows,
            )
            if picked is None:
                continue
            chosen_state = final_state
            chosen_tile, chosen_est, chosen_guarded_est, chosen_cal_mul, chosen_cal_samples, chosen_margin = picked
            chosen_overlap = TARGET_OVERLAP
            chosen_chunk = int(chunk)
            success = True
            notes.append("Frame chunk size was reduced after exhausting scale/max-resolution fallbacks.")
            break

    # Phase 3: emergency overlap fallback to 24 if needed.
    if not success and states:
        final_state = states[-1]
        for chunk in FRAME_CHUNK_FALLBACKS:
            picked, _ = _pick_best_tile_for_state(
                state=final_state,
                mode=mode,
                precision=precision,
                vae_model=vae_model,
                frame_chunk_size=int(chunk),
                overlap=MIN_OVERLAP,
                keep_models_on_cpu=keep_models_on_cpu,
                tiled_dit=True,
                tiled_vae=False,
                stream_decode=stream_decode,
                target_vram_gb=budget.target_vram_gb,
                records_csv_path=records_csv_path,
                calibration_rows=calibration_rows,
            )
            if picked is None:
                continue
            chosen_state = final_state
            chosen_tile, chosen_est, chosen_guarded_est, chosen_cal_mul, chosen_cal_samples, chosen_margin = picked
            chosen_overlap = MIN_OVERLAP
            chosen_chunk = int(chunk)
            success = True
            notes.append("Emergency fallback used tile overlap 24 to stay within VRAM budget.")
            break

    if chosen_state is None:
        chosen_state = states[-1]
        est_min, guarded_min, cal_mul, cal_n, margin_min = _fallback_minimum_candidate(
            final_state=chosen_state,
            mode=mode,
            precision=precision,
            vae_model=vae_model,
            keep_models_on_cpu=keep_models_on_cpu,
            tiled_dit=True,
            tiled_vae=False,
            stream_decode=stream_decode,
            overlap=MIN_OVERLAP,
            records_csv_path=records_csv_path,
            calibration_rows=calibration_rows,
        )
        chosen_est = float(est_min)
        chosen_guarded_est = float(guarded_min)
        chosen_cal_mul = float(cal_mul)
        chosen_cal_samples = int(cal_n)
        chosen_margin = float(margin_min)
        chosen_tile = MIN_TILE_SIZE
        chosen_overlap = MIN_OVERLAP
        chosen_chunk = MIN_FRAME_CHUNK
        notes.append("Even minimum constraints are estimated above VRAM budget.")

    if chosen_state.scale != req_scale:
        notes.append(f"Upscale factor reduced from {req_scale}x to {chosen_state.scale}x for VRAM safety.")
    if chosen_state.max_target_resolution != current_max_edge:
        if chosen_state.max_target_resolution <= 0:
            notes.append("Max edge cap remains disabled.")
        else:
            notes.append(f"Max edge cap set to {chosen_state.max_target_resolution}px.")

    safety_gap = budget.target_vram_gb - chosen_guarded_est
    if safety_gap < 0:
        notes.append(f"Guarded estimate is {abs(safety_gap):.2f} GB above target.")
    else:
        notes.append(f"Guarded estimated VRAM headroom: {safety_gap:.2f} GB.")

    return FlashVSROptimizedSettings(
        success=bool(success),
        tile_size=int(chosen_tile),
        overlap=int(chosen_overlap),
        frame_chunk_size=int(chosen_chunk),
        scale=int(chosen_state.scale),
        max_target_resolution=int(chosen_state.max_target_resolution),
        tiled_dit=True,
        tiled_vae=False,
        estimated_peak_vram_gb=float(chosen_est),
        budget=budget,
        preprocess_width=int(chosen_state.preprocess_width),
        preprocess_height=int(chosen_state.preprocess_height),
        output_width=int(chosen_state.output_width),
        output_height=int(chosen_state.output_height),
        stage_label=str(chosen_state.stage_label),
        calibration_multiplier=float(chosen_cal_mul),
        calibration_samples=int(chosen_cal_samples),
        notes=tuple(notes),
        estimation_safety_margin_gb=float(chosen_margin),
        estimated_guarded_vram_gb=float(chosen_guarded_est),
    )


def format_flashvsr_optimization_summary(result: FlashVSROptimizedSettings) -> str:
    status = "OK" if result.success else "WARN"
    budget = result.budget
    lines = [
        f"### {status}: FlashVSR Parameter Optimization",
        f"- GPU: `{budget.gpu_label}`",
        (
            f"- VRAM target: `{budget.target_vram_gb:.2f} GB` "
            f"(total `{budget.total_vram_gb:.2f} GB` - reserve `{budget.reserve_vram_gb:.2f} GB`)"
        ),
        f"- Estimated peak VRAM: `{result.estimated_peak_vram_gb:.2f} GB`",
        (
            f"- Guarded estimate: `{result.estimated_guarded_vram_gb:.2f} GB` "
            f"(includes `{result.estimation_safety_margin_gb:.2f} GB` safety margin)"
        ),
        f"- Tiled DiT: `ON` | Tiled VAE: `OFF`",
        (
            f"- Upscale: `{result.scale}x` | Max edge: "
            f"`{result.max_target_resolution if result.max_target_resolution > 0 else 0}`"
        ),
        f"- Frame chunk size: `{result.frame_chunk_size}`",
        f"- Tile size / overlap: `{result.tile_size}` / `{result.overlap}`",
        (
            f"- Working resolution: `{result.preprocess_width}x{result.preprocess_height}` "
            f"-> output `{result.output_width}x{result.output_height}`"
        ),
    ]
    if result.calibration_samples > 0:
        lines.append(
            f"- Calibration: `{result.calibration_multiplier:.3f}x` from `{result.calibration_samples}` sweep samples"
        )
    else:
        lines.append("- Calibration: `1.000x` (no sweep samples found)")
    if result.notes:
        for note in result.notes:
            lines.append(f"- Note: {note}")
    return "\n".join(lines)
