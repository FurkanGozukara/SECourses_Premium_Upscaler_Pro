"""
SparkVSR runtime auto-tune implementation.

The tuner runs real SparkVSR probes on a short demo clip, measures live VRAM,
and applies the highest-quality spatial/temporal chunking settings that keep
the requested free-VRAM headroom.
"""

from __future__ import annotations

import hashlib
import html
import json
import queue
import re
import shutil
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr

from shared.gpu_utils import get_global_gpu_override, get_gpu_info
from shared.oom_alert import clear_vram_oom_alert
from shared.path_utils import (
    IMAGE_EXTENSIONS,
    detect_input_type,
    get_media_dimensions,
    get_media_duration_seconds,
    get_media_fps,
    normalize_path,
)
from shared.resolution_calculator import estimate_fixed_scale_upscale_plan_from_dims
from shared.services.autotune_search import (
    ambient_adjusted_min_device_free_gb,
    ambient_adjusted_peak_gb,
    autotune_launch_headroom,
    cached_best_has_headroom,
    dedupe_sparkvsr_tile_candidates,
    frontier_bisect,
    is_verified_autotune_payload,
    is_vram_boundary_outcome,
    persisted_autotune_status,
    predict_frontier_index,
    resolution_signatures_compatible,
    resolution_signatures_identical,
    resume_frontier_hints,
)
from shared.services.flashvsr_autotune import (
    _clamp_save_vram_target_gb,
    _create_autotune_demo_video,
    _looks_like_oom,
    _parse_cuda_device_ids,
    _query_gpu_memory_snapshot_gb,
    _sample_peak_vram_gb,
    _wait_for_vram_drain,
)
from shared.sparkvsr_runner import SparkVSRResult, run_sparkvsr


AUTOTUNE_MODEL_ID = "sparkvsr"
AUTOTUNE_LOG_PREFIX = "sparkvsr_autotune"
AUTOTUNE_STRATEGY_VERSION = 16
AUTOTUNE_TARGET_FRAMES = 65
AUTOTUNE_MAX_PROBE_FRAMES = 129
AUTOTUNE_INITIAL_PROBE_FRAMES = AUTOTUNE_TARGET_FRAMES
AUTOTUNE_MIN_FREE_VRAM_GB = 2.0
AUTOTUNE_EMERGENCY_FREE_VRAM_GB = 1.0
AUTOTUNE_TEMPORAL_CANDIDATES = (65, 49, 33, 17)
AUTOTUNE_TEMPORAL_GROWTH_CANDIDATES = tuple(range(AUTOTUNE_TARGET_FRAMES + 8, AUTOTUNE_MAX_PROBE_FRAMES + 1, 8))
AUTOTUNE_SPATIAL_CANDIDATES = (0, 1024, 768, 512, 384, 256)
AUTOTUNE_TEMPORAL_OVERLAP = 8
AUTOTUNE_SPATIAL_OVERLAP = 32
AUTOTUNE_MAX_PIXEL_DIFF_FOR_REUSE = 0.05
AUTOTUNE_MAX_VRAM_DIFF_FOR_REUSE = 0.02
AUTOTUNE_FULL_SEQUENCE_FRAME_LIMIT = AUTOTUNE_TARGET_FRAMES
AUTOTUNE_VRAM_SAMPLE_INTERVAL_SEC = 0.10
AUTOTUNE_UI_UPDATE_INTERVAL_SEC = 0.50


def _resolve_uploaded_path(uploaded_file: Any) -> str:
    if uploaded_file is None:
        return ""
    if isinstance(uploaded_file, str):
        return normalize_path(uploaded_file)
    if isinstance(uploaded_file, dict):
        for key in ("path", "name", "orig_name"):
            val = uploaded_file.get(key)
            if val:
                return normalize_path(str(val))
    path_attr = getattr(uploaded_file, "name", None)
    if path_attr:
        return normalize_path(str(path_attr))
    raw = str(uploaded_file).strip()
    return normalize_path(raw) if raw else ""


def _write_autotune_log(log_dir: Path, payload: Dict[str, Any], existing_path: Optional[Path] = None) -> Optional[Path]:
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = existing_path
        if path is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = log_dir / f"{AUTOTUNE_LOG_PREFIX}_{stamp}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path
    except Exception:
        return None


def _estimate_input_frame_count(input_path: str) -> Tuple[Optional[int], str]:
    """Best-effort frame count for deciding whether full-sequence probes are safe to reuse."""
    path = Path(normalize_path(input_path))
    try:
        kind = detect_input_type(str(path))
    except Exception:
        kind = ""

    if kind == "image":
        return 1, "image"

    if kind == "directory":
        try:
            count = sum(1 for child in path.iterdir() if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS)
            if count > 0:
                return int(count), "directory"
        except Exception:
            pass
        return None, "directory"

    if kind == "video" or path.is_file():
        try:
            import cv2  # type: ignore

            cap = cv2.VideoCapture(str(path))
            try:
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            finally:
                cap.release()
            if count > 0:
                return count, "cv2"
        except Exception:
            pass

        duration = get_media_duration_seconds(str(path))
        fps = get_media_fps(str(path))
        if duration and fps and duration > 0 and fps > 0:
            return max(1, int(round(float(duration) * float(fps)))), "duration_fps"
        return None, "video_unknown"

    return None, "unknown"


def _autotune_reference_count(settings: Dict[str, Any]) -> int:
    """Number of reference latents the SparkVSR probe must keep resident."""
    ref_mode = str(settings.get("ref_mode") or "sr_image").strip().lower()
    if ref_mode == "no_ref":
        return 0
    if bool(settings.get("auto_reference_prepass", False)):
        # The prepass produces one local reference for each temporal chunk.
        return 1
    values: set[int] = set()
    for part in str(settings.get("ref_indices") or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        with suppress(Exception):
            values.add(int(float(part)))
    # SparkVSR auto-selects first/middle/last when the field is empty.
    return len(values) if values else 3


def _build_autotune_signature(
    settings: Dict[str, Any],
    *,
    target_w: int,
    target_h: int,
    effective_in_w: int,
    effective_in_h: int,
    total_vram_gb: float,
    min_free_target_gb: float,
    allow_full_sequence: bool,
    growth_probe_frames: int,
) -> Dict[str, Any]:
    ref_mode = str(settings.get("ref_mode") or "sr_image").strip().lower()
    if bool(settings.get("auto_reference_prepass", False)) and ref_mode != "no_ref":
        ref_mode = "sr_image"
    if ref_mode in {"pisasr", "gt"}:
        ref_mode = "sr_image"
    exact_payload = {
        "autotune_model": AUTOTUNE_MODEL_ID,
        "autotune_strategy_version": int(AUTOTUNE_STRATEGY_VERSION),
        "input_kind": str(settings.get("_autotune_input_kind") or "video"),
        "autotune_target_frames": int(AUTOTUNE_TARGET_FRAMES),
        "autotune_initial_probe_frames": int(AUTOTUNE_INITIAL_PROBE_FRAMES),
        "autotune_max_probe_frames": int(AUTOTUNE_MAX_PROBE_FRAMES),
        "growth_probe_frames": int(growth_probe_frames),
        "autotune_growth_candidates": list(AUTOTUNE_TEMPORAL_GROWTH_CANDIDATES),
        "model_name": str(settings.get("model_name") or ""),
        "model_path": str(settings.get("model_path") or ""),
        "lora_path": str(settings.get("lora_path") or ""),
        "precision": str(settings.get("precision") or "bfloat16"),
        "scale": str(settings.get("scale") or "4"),
        "upscale_mode": str(settings.get("upscale_mode") or "bilinear"),
        "noise_step": int(settings.get("noise_step") or 0),
        "sr_noise_step": int(settings.get("sr_noise_step") or 399),
        "cpu_offload": bool(settings.get("cpu_offload", True)),
        "vae_tiling": bool(settings.get("vae_tiling", True)),
        "group_offload": bool(settings.get("group_offload", False)),
        "num_blocks_per_group": (
            int(settings.get("num_blocks_per_group") or 1)
            if bool(settings.get("group_offload", False))
            else 0
        ),
        "force_offload": bool(settings.get("force_offload", True)),
        "ref_mode": ref_mode,
        "reference_count": int(_autotune_reference_count(settings)),
        "ref_guidance_scale": round(float(settings.get("ref_guidance_scale") or 1.0), 4),
        "save_format": str(settings.get("save_format") or "yuv444p"),
        "save_vram_gb": _clamp_save_vram_target_gb(min_free_target_gb, AUTOTUNE_MIN_FREE_VRAM_GB),
        "split_stage_subprocesses": bool(settings.get("split_stage_subprocesses", True)),
        "temporal_policy": "full_sequence_allowed" if allow_full_sequence else "bounded_chunks_required",
        "full_sequence_frame_limit": int(AUTOTUNE_FULL_SEQUENCE_FRAME_LIMIT),
        "long_sequence_chunk_len": int(AUTOTUNE_TARGET_FRAMES),
    }
    exact_blob = json.dumps(exact_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "exact": exact_payload,
        "exact_hash": hashlib.sha1(exact_blob.encode("utf-8")).hexdigest(),
        "target_pixels": int(max(1, int(target_w) * int(target_h))),
        "target_width": int(target_w),
        "target_height": int(target_h),
        "effective_input_width": int(effective_in_w),
        "effective_input_height": int(effective_in_h),
        "gpu_total_vram_gb": float(total_vram_gb or 0.0),
    }


def _signature_matches(candidate: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    if not isinstance(candidate, dict) or not isinstance(expected, dict):
        return False
    if candidate.get("exact_hash") != expected.get("exact_hash"):
        cand_exact = candidate.get("exact")
        exp_exact = expected.get("exact")
        if not isinstance(cand_exact, dict) or not isinstance(exp_exact, dict):
            return False
        cand_exact = dict(cand_exact)
        exp_exact = dict(exp_exact)
        for key in ("global_gpu_device", "gpu_identity"):
            cand_exact.pop(key, None)
            exp_exact.pop(key, None)
        if cand_exact != exp_exact:
            return False
    try:
        if not resolution_signatures_identical(candidate, expected):
            return False
        cand_vram = float(candidate.get("gpu_total_vram_gb") or 0)
        exp_vram = float(expected.get("gpu_total_vram_gb") or 0)
        if (
            cand_vram > 0
            and exp_vram > 0
            and abs(cand_vram - exp_vram) / max(cand_vram, exp_vram)
            > AUTOTUNE_MAX_VRAM_DIFF_FOR_REUSE
        ):
            return False
    except Exception:
        return False
    return True


def _find_cached_autotune_log(
    log_dir: Path,
    expected_signature: Dict[str, Any],
    min_free_vram_target_gb: float,
    current_ambient_used_gb: float = 0.0,
) -> Optional[Dict[str, Any]]:
    def _has_strict_search_proof(payload: Dict[str, Any]) -> bool:
        results = payload.get("search_results")
        best = payload.get("best_config")
        tests = payload.get("tests")
        if not isinstance(results, dict) or not isinstance(best, dict) or not isinstance(tests, list):
            return False
        for axis in ("temporal", "spatial"):
            result = results.get(axis)
            if not isinstance(result, dict) or not bool(result.get("frontier_verified", False)):
                return False
            if bool(result.get("indeterminate_above_best", False)) or result.get("hard_failed_indices"):
                return False
        for item in tests:
            if not isinstance(item, dict) or not bool(item.get("passed", False)):
                continue
            if str(item.get("probe_cancel_reason") or "").strip():
                continue
            try:
                returncode_ok = int(item.get("returncode", 1)) == 0
            except Exception:
                returncode_ok = False
            if not bool(item.get("telemetry_ok", False)) or not returncode_ok:
                continue
            if bool(item.get("oom", False)):
                continue
            if int(item.get("chunk_len") or 0) != int(best.get("chunk_len") or 0):
                continue
            if int(item.get("tile_height") or 0) != int(best.get("tile_height") or 0):
                continue
            if int(item.get("tile_width") or 0) != int(best.get("tile_width") or 0):
                continue
            if int(item.get("overlap_t") or 0) != int(best.get("overlap_t") or 0):
                continue
            if int(item.get("overlap_height") or 0) != int(best.get("overlap_height") or 0):
                continue
            if int(item.get("overlap_width") or 0) != int(best.get("overlap_width") or 0):
                continue
            best_probe_frames = int(best.get("probe_frames") or 0)
            if best_probe_frames > 0 and int(item.get("probe_frames") or 0) != best_probe_frames:
                continue
            sig_exact = ((payload.get("signature") or {}).get("exact") or {})
            if bool(sig_exact.get("split_stage_subprocesses", True)) and not bool(
                item.get("split_stage_validation_ok", False)
            ):
                continue
            if int(item.get("reference_count") or 0) != int(sig_exact.get("reference_count") or 0):
                continue
            return True
        return False

    if not log_dir.exists():
        return None
    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for path in sorted(log_dir.glob(f"{AUTOTUNE_LOG_PREFIX}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not is_verified_autotune_payload(payload):
            continue
        if not _has_strict_search_proof(payload):
            continue
        signature = payload.get("signature")
        best = payload.get("best_config")
        if not isinstance(signature, dict) or not isinstance(best, dict):
            continue
        if not _signature_matches(signature, expected_signature):
            continue
        try:
            cached_reserve = float(((signature or {}).get("exact") or {}).get("save_vram_gb"))
        except Exception:
            continue
        if abs(cached_reserve - float(min_free_vram_target_gb)) > 0.05:
            continue
        if not cached_best_has_headroom(
            payload,
            expected_signature,
            min_free_vram_target_gb,
            current_ambient_used_gb=current_ambient_used_gb,
        ):
            continue
        candidates.append((float(path.stat().st_mtime), {**payload, "_path": str(path)}))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _extract_demo_reference(demo_video_path: Path, ref_path: Path) -> Optional[Path]:
    try:
        import cv2  # type: ignore

        ref_path.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(demo_video_path))
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            return None
        if cv2.imwrite(str(ref_path), frame):
            return ref_path
    except Exception:
        return None
    return None


def _normalize_probe_ref_mode(settings: Dict[str, Any]) -> str:
    ref_mode = str(settings.get("ref_mode") or "sr_image").strip().lower()
    if ref_mode == "no_ref":
        return "no_ref"
    # PiSA/GT/manual auto-reference are external to SparkVSR memory tuning.
    # Use a local first-frame reference so reference-latent VRAM is still represented.
    return "sr_image"


def _detect_sparkvsr_oom_phase(log_text: str) -> str:
    text = str(log_text or "")
    current = "unknown"
    oom_tokens = (
        "out of memory",
        "cuda out of memory",
        "torch.cuda.outofmemoryerror",
        "allocation on device",
        "failed to allocate memory",
    )
    for raw in text.splitlines():
        line = raw.strip().lower()
        if not line:
            continue
        if "phase=references" in line or "encoding reference keyframes" in line:
            current = "references"
        elif "phase=tile" in line or "running sparkvsr transformer" in line:
            current = "tile"
        elif "phase=model_load" in line or "loading pipeline" in line:
            current = "model_load"
        elif "phase=resize_input" in line or "upscaled input" in line:
            current = "resize_input"
        if any(tok in line for tok in oom_tokens):
            return current
    return "unknown"


def _strip_ansi(text: str) -> str:
    try:
        return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", str(text or ""))
    except Exception:
        return str(text or "")


def _parse_sparkvsr_progress_fraction(text: str) -> Optional[float]:
    line = _strip_ansi(text).strip()
    if not line:
        return None
    m = re.search(r"SparkVSR\s+Progress:\s*(\d+(?:\.\d+)?)\s*%", line, flags=re.IGNORECASE)
    if m:
        with suppress(Exception):
            return max(0.0, min(1.0, float(m.group(1)) / 100.0))
    m = re.search(r"Processing\s+Tiles:\s*(\d+)\s*/\s*(\d+).*?\((\d+(?:\.\d+)?)%\)", line, flags=re.IGNORECASE)
    if m:
        with suppress(Exception):
            return max(0.0, min(1.0, float(m.group(3)) / 100.0))
    m = re.search(r"Processing\s+Tiles:\s*(\d+)\s*/\s*(\d+)", line, flags=re.IGNORECASE)
    if m:
        with suppress(Exception):
            return max(0.0, min(1.0, float(max(0, int(m.group(1)) - 1)) / float(max(1, int(m.group(2))))))
    m = re.search(r"(?:Processed|Processing):\s*(\d+)\s*/\s*(\d+)", line, flags=re.IGNORECASE)
    if m:
        with suppress(Exception):
            return max(0.0, min(1.0, float(int(m.group(1))) / float(max(1, int(m.group(2))))))
    return None


def _normalize_probe_progress_line(text: str) -> Tuple[str, bool]:
    line = _strip_ansi(text).strip()
    if not line:
        return "", False
    lc = line.lower()
    transient = (
        lc.startswith("sparkvsr progress:")
        or lc.startswith("processing tiles:")
        or lc.startswith("[sparkvsr] still running")
        or "loading pipeline components" in lc
        or "loading checkpoint shards" in lc
    )
    return line, transient


def _candidate_settings(
    *,
    allow_full_sequence: bool = False,
    effective_h: int = 0,
    effective_w: int = 0,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int, int]] = set()

    def add_candidate(chunk_len: int, tile: int) -> None:
        if int(chunk_len) > 0 and int(chunk_len) <= AUTOTUNE_TEMPORAL_OVERLAP:
            return
        key = (int(chunk_len), int(tile), int(tile))
        if key in seen:
            return
        seen.add(key)
        out.append(
            {
                "chunk_len": int(chunk_len),
                "overlap_t": int(AUTOTUNE_TEMPORAL_OVERLAP),
                "tile_height": int(tile),
                "tile_width": int(tile),
                "overlap_height": int(AUTOTUNE_SPATIAL_OVERLAP),
                "overlap_width": int(AUTOTUNE_SPATIAL_OVERLAP),
            }
        )

    _ = allow_full_sequence
    spatial_candidates: List[int] = [int(t) for t in AUTOTUNE_SPATIAL_CANDIDATES]
    if int(effective_h) > 0 and int(effective_w) > 0:
        # Tiles covering the whole frame are byte-identical to full-frame
        # (tile=0), so probing them separately is wasted work.
        deduped = dedupe_sparkvsr_tile_candidates(
            int(effective_h), int(effective_w), spatial_candidates, int(AUTOTUNE_SPATIAL_OVERLAP)
        )
        spatial_candidates = ([0] if 0 in spatial_candidates else []) + list(deduped)
    for chunk_len in AUTOTUNE_TEMPORAL_CANDIDATES:
        if int(chunk_len) <= 0:
            continue
        for tile in spatial_candidates:
            add_candidate(int(chunk_len), int(tile))
    return sorted(out, key=_quality_rank, reverse=True)


def _growth_candidate_settings(probe_frames: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    upper = min(int(AUTOTUNE_MAX_PROBE_FRAMES), max(0, int(probe_frames)))
    chunk_values = {
        int(chunk_len)
        for chunk_len in AUTOTUNE_TEMPORAL_GROWTH_CANDIDATES
        if int(AUTOTUNE_TARGET_FRAMES) < int(chunk_len) <= upper
    }
    # Include the exact bounded clip length. Otherwise a 72- or 100-frame
    # source stops at 65 or 97 even when the whole bounded sequence would fit.
    if int(AUTOTUNE_TARGET_FRAMES) < upper:
        chunk_values.add(upper)
    for chunk_len in sorted(chunk_values):
        out.append(
            {
                "chunk_len": int(chunk_len),
                "overlap_t": int(AUTOTUNE_TEMPORAL_OVERLAP),
                "tile_height": 0,
                "tile_width": 0,
                "overlap_height": int(AUTOTUNE_SPATIAL_OVERLAP),
                "overlap_width": int(AUTOTUNE_SPATIAL_OVERLAP),
            }
        )
    return out


def _quality_rank(cfg: Dict[str, Any]) -> int:
    chunk_len = int(cfg.get("chunk_len") or 0)
    tile = int(cfg.get("tile_height") or cfg.get("tile_width") or 0)
    # Lexicographic quality: every extra temporal frame must outrank every
    # possible spatial improvement. The previous coefficients allowed a
    # full-frame 73-frame candidate to outrank an 81-frame tiled candidate.
    temporal = 1_000_000_000 if chunk_len <= 0 else int(chunk_len) * 1_000_000
    spatial = 200_000 if tile <= 0 else int(tile) * 100
    return temporal + spatial


def sparkvsr_auto_tune_action(
    *,
    uploaded_file,
    args: Tuple[Any, ...],
    state: Dict[str, Any] | None,
    progress,
    global_settings_snapshot: Dict[str, Any] | None,
    global_settings_fallback: Dict[str, Any],
    defaults: Dict[str, Any],
    sparkvsr_order: List[str],
    parse_args_fn: Callable[[List[Any]], Dict[str, Any]],
    guardrail_fn: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    canonical_scale_fn: Callable[..., int],
    base_dir: Path,
    temp_dir: Path,
    cancel_event: threading.Event,
):
    global_cfg = (
        dict(global_settings_snapshot)
        if isinstance(global_settings_snapshot, dict)
        else dict(global_settings_fallback)
    )
    state = state or {"seed_controls": {}, "operation_status": "ready"}
    state.setdefault("seed_controls", {})
    seed_controls = state.get("seed_controls", {})
    log_lines: List[str] = []
    live_log_idx: Optional[int] = None
    cmd_inline_progress_active = False
    cmd_inline_progress_width = 0

    def _finish_cmd_inline() -> None:
        nonlocal cmd_inline_progress_active, cmd_inline_progress_width
        if not cmd_inline_progress_active:
            return
        try:
            print("", flush=True)
        except Exception:
            pass
        cmd_inline_progress_active = False
        cmd_inline_progress_width = 0

    def _append_live_log(text: str, *, transient: bool = False) -> None:
        nonlocal live_log_idx
        msg = str(text or "").strip()
        if not msg:
            return
        if transient:
            if live_log_idx is None or live_log_idx >= len(log_lines):
                log_lines.append(msg)
                live_log_idx = len(log_lines) - 1
            else:
                log_lines[live_log_idx] = msg
            return
        live_log_idx = None
        log_lines.append(msg)

    def _print_probe_cmd_line(text: str, *, transient: bool = False) -> None:
        nonlocal cmd_inline_progress_active, cmd_inline_progress_width
        payload = str(text or "").rstrip("\r\n")
        if not payload:
            return
        try:
            if transient:
                padded = payload
                if cmd_inline_progress_width > len(payload):
                    padded = payload + (" " * (cmd_inline_progress_width - len(payload)))
                print(f"\r[SparkVSR AutoTune] {padded}", end="", flush=True)
                cmd_inline_progress_active = True
                cmd_inline_progress_width = len(payload)
                return
            _finish_cmd_inline()
            print(f"[SparkVSR AutoTune] {payload}", flush=True)
        except Exception:
            pass

    def _append_probe_output(text: str, *, transient: bool = False) -> None:
        msg = str(text or "").strip()
        if not msg:
            return
        _append_live_log(msg, transient=transient)
        _print_probe_cmd_line(msg, transient=transient)

    def _indicator(title: str, subtitle: str) -> Dict[str, Any]:
        return gr.update(
            value=(
                '<div class="processing-banner">'
                '<div class="processing-spinner"></div>'
                '<div class="processing-col">'
                f'<div class="processing-text">{html.escape(str(title or ""))}</div>'
                f'<div class="processing-sub">{html.escape(str(subtitle or ""))}</div>'
                "</div></div>"
            ),
            visible=True,
        )

    def _payload(
        status_text: str,
        *,
        show_indicator: bool,
        tile_value: Optional[int] = None,
        overlap_hw_value: Optional[int] = None,
        chunk_value: Optional[int] = None,
        overlap_t_value: Optional[int] = None,
        vae_tiling_value: Optional[bool] = None,
        summary_text: Optional[str] = None,
    ):
        tile_i = int(tile_value) if tile_value is not None else None
        overlap_i = int(overlap_hw_value) if overlap_hw_value is not None else None
        return (
            str(status_text or ""),
            "\n".join(log_lines[-240:]),
            (_indicator("Auto Tune running", status_text) if show_indicator else gr.update(value="", visible=False)),
            (gr.update(value=tile_i) if tile_i is not None else gr.update()),
            (gr.update(value=tile_i) if tile_i is not None else gr.update()),
            (gr.update(value=overlap_i) if overlap_i is not None else gr.update()),
            (gr.update(value=overlap_i) if overlap_i is not None else gr.update()),
            (gr.update(value=int(chunk_value)) if chunk_value is not None else gr.update()),
            (gr.update(value=int(overlap_t_value)) if overlap_t_value is not None else gr.update()),
            (gr.update(value=bool(vae_tiling_value)) if vae_tiling_value is not None else gr.update()),
            (gr.update(value=str(summary_text), visible=bool(summary_text)) if summary_text is not None else gr.update()),
            state,
        )

    def _append_log(text: str) -> None:
        msg = str(text or "").strip()
        if not msg:
            return
        _append_live_log(msg, transient=False)
        _print_probe_cmd_line(msg, transient=False)

    session_dir: Optional[Path] = None
    autotune_log_path: Optional[Path] = None
    tests: List[Dict[str, Any]] = []
    best_config: Optional[Dict[str, Any]] = None
    status_reason = "running"

    try:
        state["operation_status"] = "running"
        clear_vram_oom_alert(state)
        cancel_event.clear()

        if len(args) != len(sparkvsr_order):
            _append_log(f"Schema mismatch: received {len(args)} settings values but expected {len(sparkvsr_order)}.")
            yield _payload("Auto Tune aborted: schema mismatch.", show_indicator=False)
            return

        settings = {**defaults, **parse_args_fn(list(args))}
        settings = guardrail_fn(settings, defaults)
        settings["batch_enable"] = False
        settings["batch_input_path"] = ""
        settings["batch_output_path"] = ""
        settings["resume_run_dir"] = ""
        settings["save_metadata"] = False
        settings["output_format"] = "mp4"
        settings["start_frame"] = 0
        settings["end_frame"] = -1
        settings["vae_tiling"] = True

        input_path = _resolve_uploaded_path(uploaded_file) or normalize_path(settings.get("input_path"))
        if not input_path or not Path(input_path).exists():
            _append_log("Input path is missing or does not exist.")
            yield _payload("Auto Tune requires a valid input file/path.", show_indicator=False)
            return
        settings["_autotune_input_kind"] = detect_input_type(input_path)

        global_gpu_device = get_global_gpu_override(seed_controls, global_cfg)
        settings["device"] = "cpu" if global_gpu_device == "cpu" else str(global_gpu_device)
        settings = guardrail_fn(settings, defaults)
        min_free_vram_target_gb = _clamp_save_vram_target_gb(
            settings.get("save_vram_gb", AUTOTUNE_MIN_FREE_VRAM_GB),
            AUTOTUNE_MIN_FREE_VRAM_GB,
        )
        if min_free_vram_target_gb < float(AUTOTUNE_EMERGENCY_FREE_VRAM_GB):
            _append_log(
                f"Raised Auto Tune's effective VRAM reserve from {min_free_vram_target_gb:.1f}GB "
                f"to the {AUTOTUNE_EMERGENCY_FREE_VRAM_GB:.1f}GB emergency safety floor."
            )
            min_free_vram_target_gb = float(AUTOTUNE_EMERGENCY_FREE_VRAM_GB)
        settings["save_vram_gb"] = float(min_free_vram_target_gb)
        try:
            import os as _os

            campaign_profile_min_peak_gb = max(
                0.0,
                float(_os.environ.get("SECOURSES_AUTOTUNE_PROFILE_MIN_VRAM_GB", "0") or 0.0),
            )
        except Exception:
            campaign_profile_min_peak_gb = 0.0
        campaign_force_fresh = bool(campaign_profile_min_peak_gb > 0.0)
        if global_gpu_device == "cpu":
            _append_log("Global GPU selector is set to CPU. Auto Tune requires CUDA GPU mode.")
            yield _payload("Auto Tune unavailable in CPU mode.", show_indicator=False)
            return
        split_stage_label = "ON" if bool(settings.get("split_stage_subprocesses", True)) else "OFF"
        prompt_text = str(settings.get("prompt") or "")
        prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        _append_log(
            "Using current UI settings for SparkVSR probes: "
            f"model={settings.get('model_name')}, precision={settings.get('precision')}, "
            f"scale={settings.get('scale')}x, ref_mode={settings.get('ref_mode')}, "
            f"prompt_sha256={prompt_hash[:12]}, prompt_chars={len(prompt_text)}, "
            f"cpu_offload={'ON' if bool(settings.get('cpu_offload', True)) else 'OFF'}, "
            "vae_tiling=ON (forced during autotune), "
            f"stage_subprocess_isolation={split_stage_label}, "
            f"device={settings.get('device')}, save_vram={min_free_vram_target_gb:.1f}GB."
        )

        dims = get_media_dimensions(input_path)
        if not dims:
            _append_log("Could not read input dimensions.")
            yield _payload("Failed to probe input dimensions.", show_indicator=False)
            return
        input_w, input_h = int(dims[0]), int(dims[1])
        source_frame_count, frame_count_source = _estimate_input_frame_count(input_path)
        growth_probe_frames = (
            int(AUTOTUNE_MAX_PROBE_FRAMES)
            if source_frame_count is None
            else max(
                int(AUTOTUNE_INITIAL_PROBE_FRAMES),
                min(int(AUTOTUNE_MAX_PROBE_FRAMES), int(source_frame_count)),
            )
        )
        allow_full_sequence = False
        if source_frame_count is None:
            _append_log(
                "Could not determine source frame count. Auto Tune will start with a "
                f"{AUTOTUNE_INITIAL_PROBE_FRAMES}-frame bounded probe and only create the "
                f"{AUTOTUNE_MAX_PROBE_FRAMES}-frame growth clip if the top candidate passes."
            )
        else:
            _append_log(
                f"Source frame count: {source_frame_count} ({frame_count_source}). "
                f"Auto Tune starts with a {AUTOTUNE_INITIAL_PROBE_FRAMES}-frame bounded probe; "
                f"{AUTOTUNE_MAX_PROBE_FRAMES}-frame growth probes are deferred until needed. "
                "chunk_len=0 full-sequence probes are skipped so longer clips stay inside the measured VRAM envelope."
            )
        temporal_policy_label = "bounded chunks required"
        source_frames_note = (
            f" ({int(source_frame_count)} source frames)"
            if source_frame_count is not None
            else ""
        )

        resolved_scale = canonical_scale_fn(
            scale_value=settings.get("scale", 4),
            upscale_factor_value=settings.get("upscale_factor"),
            default=settings.get("upscale_factor", settings.get("scale", 4)),
        )
        if bool(settings.get("use_resolution_tab", True)):
            raw_shared_scale = seed_controls.get("upscale_factor_val")
            if raw_shared_scale is not None:
                with suppress(Exception):
                    resolved_scale = canonical_scale_fn(
                        scale_value=settings.get("scale", resolved_scale),
                        upscale_factor_value=float(raw_shared_scale),
                        default=resolved_scale,
                    )
        settings["scale"] = str(int(resolved_scale))
        settings["upscale_factor"] = float(int(resolved_scale))

        plan = estimate_fixed_scale_upscale_plan_from_dims(
            int(input_w),
            int(input_h),
            requested_scale=float(settings.get("upscale_factor") or resolved_scale),
            model_scale=int(resolved_scale),
            max_edge=int(settings.get("max_target_resolution") or 0),
            force_pre_downscale=bool(settings.get("pre_downscale_then_upscale", True)),
        )
        target_w = int(plan.final_saved_width or plan.resize_width or 0)
        target_h = int(plan.final_saved_height or plan.resize_height or 0)
        effective_in_w = int(plan.preprocess_width if plan.pre_downscale_then_upscale else input_w)
        effective_in_h = int(plan.preprocess_height if plan.pre_downscale_then_upscale else input_h)
        if target_w <= 0 or target_h <= 0 or effective_in_w <= 0 or effective_in_h <= 0:
            _append_log("Could not determine target output dimensions for autotune.")
            yield _payload("Auto Tune failed to calculate target dimensions.", show_indicator=False)
            return

        gpu_ids = _parse_cuda_device_ids(settings.get("device", ""))
        if not gpu_ids and str(global_gpu_device).isdigit():
            gpu_ids = [int(global_gpu_device)]
        gpu_snapshot = _query_gpu_memory_snapshot_gb()
        selected_gpu_ids = list(gpu_ids)
        if (not selected_gpu_ids) and gpu_snapshot:
            selected_gpu_ids = [int(sorted(gpu_snapshot.keys())[0])]
        total_vram_gb = 0.0
        if selected_gpu_ids and gpu_snapshot:
            total_vram_gb = sum(float(gpu_snapshot[g][1]) for g in selected_gpu_ids if g in gpu_snapshot)
        if total_vram_gb <= 0:
            try:
                gpus = get_gpu_info()
            except Exception:
                gpus = []
            if gpus:
                if not selected_gpu_ids:
                    with suppress(Exception):
                        selected_gpu_ids = [int(gpus[0].id)]
                by_id = {int(g.id): g for g in gpus}
                total_vram_gb = sum(float(by_id[g].total_memory_gb) for g in selected_gpu_ids if g in by_id)
                if total_vram_gb <= 0:
                    total_vram_gb = float(gpus[0].total_memory_gb)
        if total_vram_gb <= 0:
            _append_log("Could not detect total VRAM for selected GPU.")
            yield _payload("Auto Tune failed to detect GPU VRAM.", show_indicator=False)
            return
        telemetry_gpu_ids = [idx for idx in selected_gpu_ids if idx in gpu_snapshot] if gpu_snapshot else []
        if not telemetry_gpu_ids:
            _append_log("Live VRAM telemetry is unavailable. Auto Tune requires nvidia-smi memory query support.")
            yield _payload("Auto Tune requires live VRAM telemetry from nvidia-smi.", show_indicator=False)
            return
        multi_gpu_autotune = len(telemetry_gpu_ids) > 1
        if multi_gpu_autotune:
            _append_log(
                "Multiple GPUs selected: enforcing per-GPU headroom and running fresh probes "
                "instead of reusing aggregate-memory history."
            )
        ambient_snap = _query_gpu_memory_snapshot_gb()
        if not ambient_snap:
            ambient_snap = gpu_snapshot
        ambient_used_gb = (
            sum(float(ambient_snap[g][0]) for g in telemetry_gpu_ids if g in ambient_snap)
            if ambient_snap
            else 0.0
        )
        launch_ok, ambient_free_gb, launch_required_gb = autotune_launch_headroom(
            total_vram_gb,
            ambient_used_gb,
            min_free_vram_target_gb,
        )
        per_device_free = [
            max(0.0, float(ambient_snap[g][1]) - float(ambient_snap[g][0]))
            for g in telemetry_gpu_ids
            if g in ambient_snap
        ]
        if per_device_free:
            ambient_free_gb = min(ambient_free_gb, min(per_device_free))
            launch_ok = bool(launch_ok and ambient_free_gb >= launch_required_gb)
        if not launch_ok:
            state["operation_status"] = "error"
            _append_log(
                f"Auto Tune stopped before launching a model: only {ambient_free_gb:.2f}GB "
                f"is currently free, but at least {launch_required_gb:.2f}GB is required "
                "for the reserve plus launch safety margin."
            )
            yield _payload(
                "Auto Tune did not start because the GPU is already too full. "
                "Stop other GPU jobs and try again.",
                show_indicator=False,
            )
            return

        signature = _build_autotune_signature(
            settings,
            target_w=target_w,
            target_h=target_h,
            effective_in_w=effective_in_w,
            effective_in_h=effective_in_h,
            total_vram_gb=total_vram_gb,
            min_free_target_gb=min_free_vram_target_gb,
            allow_full_sequence=allow_full_sequence,
            growth_probe_frames=growth_probe_frames,
        )
        logs_dir = Path(base_dir) / "vram_usages"
        cached = (
            None
            if campaign_force_fresh or multi_gpu_autotune
            else _find_cached_autotune_log(
                logs_dir,
                signature,
                min_free_vram_target_gb,
                ambient_used_gb,
            )
        )
        if cached and isinstance(cached.get("best_config"), dict):
            best = dict(cached["best_config"])
            spark_cfg = state.setdefault("seed_controls", {}).setdefault("sparkvsr_settings", {})
            spark_cfg["tile_height"] = int(best.get("tile_height") or 0)
            spark_cfg["tile_width"] = int(best.get("tile_width") or 0)
            spark_cfg["overlap_height"] = int(best.get("overlap_height") or 0)
            spark_cfg["overlap_width"] = int(best.get("overlap_width") or 0)
            spark_cfg["chunk_len"] = int(best.get("chunk_len") or 0)
            spark_cfg["overlap_t"] = (
                int(best.get("overlap_t") or 0)
                if int(spark_cfg["chunk_len"]) > 0
                else int(AUTOTUNE_TEMPORAL_OVERLAP)
            )
            spark_cfg["vae_tiling"] = True
            spark_cfg["save_vram_gb"] = float(min_free_vram_target_gb)
            state["operation_status"] = "completed"
            cached_path = str(cached.get("_path") or "cached log")
            _append_log(
                "A previous Auto Tune already found the best settings for this exact setup - "
                f"no new tests needed (saved result: {cached_path})."
            )
            _append_log(
                f"Applied: Chunk Length {spark_cfg['chunk_len']}, Temporal Overlap {spark_cfg['overlap_t']}, "
                f"Spatial Tile {'full-frame' if int(spark_cfg['tile_height']) <= 0 else spark_cfg['tile_height']}, "
                f"Spatial Overlap {spark_cfg['overlap_height']} "
                f"(measured peak {float(best.get('measured_peak_vram_used_gb') or 0.0):.2f}GB, "
                f"keeps {float(best.get('estimated_free_vram_gb') or 0.0):.2f}GB free)."
            )
            summary_md = (
                "**SparkVSR Auto Tune Result (cached)**\n"
                f"- Temporal Policy: `{temporal_policy_label}`{source_frames_note}\n"
                f"- Spatial Tile: `{spark_cfg['tile_height']}x{spark_cfg['tile_width']}` "
                f"({'disabled/full-frame' if int(spark_cfg['tile_height']) <= 0 else 'enabled'})\n"
                f"- Spatial Overlap: `{spark_cfg['overlap_height']}x{spark_cfg['overlap_width']}`\n"
                f"- Temporal Chunk Length: `{spark_cfg['chunk_len']}` "
                f"({'disabled/full sequence' if int(spark_cfg['chunk_len']) <= 0 else 'enabled'})\n"
                f"- Temporal Overlap: `{spark_cfg['overlap_t']}`\n"
                f"- Stage Subprocess Isolation: `{'ON' if bool(settings.get('split_stage_subprocesses', True)) else 'OFF'}`\n"
                f"- Peak VRAM: `{float(best.get('measured_peak_vram_used_gb') or 0.0):.2f} GB`\n"
                f"- Free VRAM estimate: `{float(best.get('estimated_free_vram_gb') or 0.0):.2f} GB` "
                f"(target >= `{min_free_vram_target_gb:.1f} GB`)\n"
                f"- Log file: `{cached_path}`"
            )
            yield _payload(
                (
                    f"Auto Tune reused a saved result - applied Chunk Length {spark_cfg['chunk_len']}, "
                    f"Spatial Tile {'full-frame' if int(spark_cfg['tile_height']) <= 0 else spark_cfg['tile_height']}, "
                    f"Temporal Overlap {spark_cfg['overlap_t']}."
                ),
                show_indicator=False,
                tile_value=int(spark_cfg["tile_height"]),
                overlap_hw_value=int(spark_cfg["overlap_height"]),
                chunk_value=int(spark_cfg["chunk_len"]),
                overlap_t_value=int(spark_cfg["overlap_t"]),
                vae_tiling_value=True,
                summary_text=summary_md,
            )
            return

        # Per-test history resume: reuse individual probe outcomes from earlier
        # (possibly interrupted) runs with a matching signature so a restarted
        # Auto Tune continues instead of starting from zero.
        history_outcomes_by_key: Dict[Tuple[int, int, int, int, int, int], Dict[str, Any]] = {}
        history_sources: List[str] = []
        # Capacity-shifted priors can hint at the frontier, but never replace a
        # live safety-first probe on the selected GPU.
        prior_peaks_by_config: Dict[Tuple[int, int], float] = {}
        prior_source_vrams: set = set()
        try:
            history_logs = (
                sorted(logs_dir.glob(f"{AUTOTUNE_LOG_PREFIX}_*.json"), key=lambda p: p.stat().st_mtime)
                if logs_dir.exists() and (not campaign_force_fresh) and (not multi_gpu_autotune)
                else []
            )
        except Exception:
            history_logs = []
        for history_path in history_logs:
            try:
                history_payload = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(history_payload, dict):
                continue
            if not _signature_matches(history_payload.get("signature"), signature):
                # A materially different reported capacity cannot be reused
                # directly, but compatible full-run peaks can still be hints.
                cand_sig = history_payload.get("signature")
                if isinstance(cand_sig, dict):
                    cand_exact = dict(cand_sig.get("exact") or {})
                    current_exact = dict(signature.get("exact") or {})
                    cand_exact.pop("save_vram_gb", None)
                    current_exact.pop("save_vram_gb", None)
                else:
                    cand_exact = {}
                    current_exact = {}
                if cand_exact and cand_exact == current_exact:
                    if resolution_signatures_compatible(
                        cand_sig,
                        signature,
                        max_pixel_diff=AUTOTUNE_MAX_PIXEL_DIFF_FOR_REUSE,
                    ):
                        for raw in history_payload.get("tests") or []:
                            if not isinstance(raw, dict):
                                continue
                            try:
                                if int(raw.get("returncode", 1)) != 0:
                                    continue
                                if (not bool(raw.get("telemetry_ok", False))) or bool(raw.get("oom", False)):
                                    continue
                                if str(raw.get("probe_cancel_reason") or "").strip():
                                    continue
                                peak = ambient_adjusted_peak_gb(
                                    raw.get("max_vram_used_gb"),
                                    history_payload,
                                    ambient_used_gb,
                                )
                                if peak <= 0:
                                    continue
                                cfg_key = (int(raw.get("chunk_len") or 0), int(raw.get("tile_height") or 0))
                            except Exception:
                                continue
                            prev = prior_peaks_by_config.get(cfg_key)
                            if prev is None or peak < prev:
                                prior_peaks_by_config[cfg_key] = peak
                            with suppress(Exception):
                                prior_source_vrams.add(
                                    round(float(cand_sig.get("gpu_total_vram_gb") or 0.0), 1)
                                )
                continue
            tests_blob = history_payload.get("tests")
            if not isinstance(tests_blob, list):
                continue
            try:
                recorded_target_gb = float(
                    (((history_payload.get("signature") or {}).get("exact") or {}).get("save_vram_gb"))
                )
            except Exception:
                recorded_target_gb = float("nan")
            matched_any = False
            for raw in tests_blob:
                if not isinstance(raw, dict):
                    continue
                item = dict(raw)
                probe_stop = str(item.get("probe_cancel_reason") or "").strip().lower()
                try:
                    original_peak = float(item.get("max_vram_used_gb") or 0.0)
                    peak = ambient_adjusted_peak_gb(
                        original_peak,
                        history_payload,
                        ambient_used_gb,
                    )
                except Exception:
                    original_peak = 0.0
                    peak = 0.0
                item["max_vram_used_gb"] = round(float(peak), 3)
                item["recorded_min_free_vram_target_gb"] = recorded_target_gb
                try:
                    total_hist = float(item.get("total_vram_gb") or 0.0)
                except Exception:
                    total_hist = 0.0
                total_for_eval = float(total_vram_gb if total_vram_gb > 0 else total_hist)
                if peak > 0 and total_for_eval > 0:
                    aggregate_free_gb = max(0.0, total_for_eval - peak)
                    if item.get("min_device_free_gb") is not None:
                        item["min_device_free_gb"] = ambient_adjusted_min_device_free_gb(
                            item.get("min_device_free_gb"),
                            original_peak,
                            peak,
                            aggregate_free_gb,
                        )
                    else:
                        item["min_device_free_gb"] = aggregate_free_gb
                    item["estimated_free_gb"] = min(
                        aggregate_free_gb,
                        float(item["min_device_free_gb"]),
                    )
                try:
                    returncode_ok = int(item.get("returncode", 1)) == 0
                except Exception:
                    returncode_ok = False
                split_expected = bool(
                    (((history_payload.get("signature") or {}).get("exact") or {}).get(
                        "split_stage_subprocesses",
                        True,
                    ))
                )
                split_validation_ok = bool(
                    (not split_expected)
                    or item.get("split_stage_validation_ok", False)
                )
                item["passed"] = bool(
                    returncode_ok
                    and bool(item.get("telemetry_ok", False))
                    and (not bool(item.get("oom", False)))
                    and split_validation_ok
                    and float(item.get("estimated_free_gb") or 0.0) >= float(min_free_vram_target_gb)
                    and not probe_stop
                )
                history_key = (
                    int(item.get("chunk_len") or 0),
                    int(item.get("tile_height") or 0),
                    int(item.get("tile_width") or 0),
                    int(item.get("overlap_t") or 0),
                    int(item.get("overlap_height") or 0),
                    int(item.get("overlap_width") or 0),
                )
                history_outcomes_by_key[history_key] = item
                matched_any = True
            if matched_any:
                history_sources.append(str(history_path))
        if history_outcomes_by_key:
            _append_log(
                f"Loaded {len(history_outcomes_by_key)} matching historical probe result(s) from "
                f"{len(history_sources)} log(s). Auto Tune will continue from previous progress."
            )
        tests[:] = list(history_outcomes_by_key.values())

        initial_probe_frames = int(AUTOTUNE_INITIAL_PROBE_FRAMES)
        probe_target_frames = initial_probe_frames

        stamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = Path(temp_dir) / "sparkvsr_autotune" / stamp
        session_dir.mkdir(parents=True, exist_ok=True)
        demo_video_path = session_dir / f"sparkvsr_autotune_demo_{probe_target_frames}f.mp4"
        demo_ref_path = session_dir / "sparkvsr_autotune_reference.png"
        growth_demo_video_path = session_dir / f"sparkvsr_autotune_demo_{growth_probe_frames}f.mp4"
        growth_demo_ref_path = session_dir / "sparkvsr_autotune_reference_growth.png"
        growth_demo_meta: Optional[Dict[str, Any]] = None
        demo_meta = _create_autotune_demo_video(
            input_path,
            demo_video_path,
            target_frames=probe_target_frames,
            resize_to=(effective_in_w, effective_in_h),
        )
        _extract_demo_reference(demo_video_path, demo_ref_path)
        _append_log(
            f"Initial demo clip: {demo_meta['written_frames']} frames, {demo_meta['width']}x{demo_meta['height']} -> "
            f"target {target_w}x{target_h}; keeping >= {min_free_vram_target_gb:.1f}GB free."
        )
        yield _payload(
            "Auto Tune setup complete. Starting SparkVSR probes...",
            show_indicator=True,
            tile_value=0,
            overlap_hw_value=AUTOTUNE_SPATIAL_OVERLAP,
            chunk_value=AUTOTUNE_TARGET_FRAMES,
            overlap_t_value=AUTOTUNE_TEMPORAL_OVERLAP,
            vae_tiling_value=True,
        )

        started_at = time.time()
        autotune_payload = {
            "status": "running",
            "signature": signature,
            "source_frame_count": int(source_frame_count) if source_frame_count is not None else None,
            "source_frame_count_source": str(frame_count_source or ""),
            "allow_full_sequence": bool(allow_full_sequence),
            "probe_target_frames": int(probe_target_frames),
            "initial_probe_frames": int(initial_probe_frames),
            "growth_probe_frames": int(growth_probe_frames),
            "gpu": {
                "selected_ids": list(selected_gpu_ids),
                "total_vram_gb": float(total_vram_gb),
                "ambient_used_gb": float(ambient_used_gb),
                "min_free_target_gb": float(min_free_vram_target_gb),
            },
            "tests": tests,
            "best_config": None,
            "frontier_verified": False,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        def _persist(status_override: Optional[str] = None, finalized: bool = False, frontier_verified: bool = False) -> None:
            nonlocal autotune_log_path
            result_reason = str(status_override or status_reason)
            autotune_payload.update(
                {
                    "status": persisted_autotune_status(
                        result_reason,
                        finalized=finalized,
                        frontier_verified=frontier_verified,
                    ),
                    "result_reason": result_reason,
                    "tests": list(tests),
                    "best_config": best_config,
                    "frontier_verified": bool(frontier_verified),
                    "finalized": bool(finalized),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_sec": round(time.time() - started_at, 3),
                }
            )
            saved = _write_autotune_log(logs_dir, autotune_payload, existing_path=autotune_log_path)
            if saved and autotune_log_path is None:
                autotune_log_path = Path(saved)
                _append_log(f"Auto Tune log file created: {autotune_log_path}")

        _persist("running")
        run_counter = 0
        candidates = _candidate_settings(
            allow_full_sequence=allow_full_sequence,
            effective_h=int(effective_in_h),
            effective_w=int(effective_in_w),
        )
        growth_candidates = _growth_candidate_settings(growth_probe_frames)
        spatial_templates = sorted(
            [
                dict(candidate)
                for candidate in candidates
                if int(candidate["chunk_len"]) == min(AUTOTUNE_TEMPORAL_CANDIDATES)
            ],
            key=_quality_rank,
        )
        if not spatial_templates:
            raise RuntimeError("SparkVSR Auto Tune could not build a safe spatial candidate.")
        safest_spatial = dict(spatial_templates[0])
        safest_tile = int(safest_spatial["tile_height"])
        temporal_values = sorted(
            {
                *[int(value) for value in AUTOTUNE_TEMPORAL_CANDIDATES if int(value) > 0],
                *[int(candidate["chunk_len"]) for candidate in growth_candidates],
            }
        )
        temporal_candidates = [
            {
                **safest_spatial,
                "chunk_len": int(chunk_len),
            }
            for chunk_len in temporal_values
        ]
        spatial_per_chunk = [int(candidate["tile_height"]) for candidate in spatial_templates]
        _append_log(
            f"Spatial candidates deduped for {effective_in_w}x{effective_in_h}: "
            f"{list(AUTOTUNE_SPATIAL_CANDIDATES)} -> {spatial_per_chunk} "
            "(tiles covering the whole frame equal full-frame and are skipped)."
        )
        _append_log(
            "Safety-first search: establish a low-memory baseline at "
            f"Chunk Length {temporal_values[0]}, Spatial Tile "
            f"{'full-frame' if safest_tile <= 0 else safest_tile}; then grow temporal context "
            "and spatial quality in bounded steps."
        )
        # Temporal context dominates SparkVSR's quality ranking. Find its safe
        # frontier at the smallest tile first, then maximize the tile at that
        # chunk length. This covers the same quality grid without top-first OOMs.
        total_estimated_runs = max(1, min(24, len(temporal_candidates) + len(spatial_templates) + 2))
        max_temporal = int(temporal_values[-1])
        top_rank = max(
            _quality_rank({**template, "chunk_len": max_temporal})
            for template in spatial_templates
        )

        def _ensure_growth_demo():
            nonlocal growth_demo_meta
            if growth_demo_meta is not None or int(growth_probe_frames) <= int(initial_probe_frames):
                return
            _append_log(
                f"Creating {growth_probe_frames}-frame growth demo clip only after the initial safe probe passed."
            )
            yield _payload(
                f"Preparing {growth_probe_frames}-frame growth probe clip...",
                show_indicator=True,
                tile_value=0,
                overlap_hw_value=AUTOTUNE_SPATIAL_OVERLAP,
                chunk_value=AUTOTUNE_TARGET_FRAMES,
                overlap_t_value=AUTOTUNE_TEMPORAL_OVERLAP,
                vae_tiling_value=True,
            )
            growth_demo_meta = _create_autotune_demo_video(
                input_path,
                growth_demo_video_path,
                target_frames=int(growth_probe_frames),
                resize_to=(effective_in_w, effective_in_h),
            )
            _extract_demo_reference(growth_demo_video_path, growth_demo_ref_path)
            autotune_payload["growth_demo_clip"] = dict(growth_demo_meta)
            _append_log(
                f"Growth demo clip: {growth_demo_meta['written_frames']} frames, "
                f"{growth_demo_meta['width']}x{growth_demo_meta['height']}."
            )
            _persist("running")

        def _run_probe_once(
            candidate: Dict[str, Any],
        ) -> Dict[str, Any]:
            nonlocal run_counter
            run_counter += 1
            tile = int(candidate["tile_height"])
            chunk_len = int(candidate["chunk_len"])
            use_growth_demo = bool(chunk_len > int(initial_probe_frames))
            active_demo_path = growth_demo_video_path if use_growth_demo else demo_video_path
            active_demo_ref_path = growth_demo_ref_path if use_growth_demo else demo_ref_path
            active_demo_meta = growth_demo_meta if use_growth_demo else demo_meta
            if active_demo_meta is None:
                raise RuntimeError(f"Growth probe clip was not prepared for chunk_len={chunk_len}.")

            probe_settings = settings.copy()
            probe_settings.update(candidate)
            probe_settings["input_path"] = str(active_demo_path)
            probe_settings["_effective_input_path"] = str(active_demo_path)
            probe_settings["_original_filename"] = Path(input_path).name
            probe_settings["_run_dir"] = str(session_dir)
            probe_settings["global_output_dir"] = str(session_dir)
            probe_settings["output_override"] = str(session_dir / f"probe_{run_counter:03d}_c{chunk_len}_t{tile}.mp4")
            probe_settings["auto_reference_prepass"] = False
            probe_settings["save_metadata"] = False
            probe_settings["fps"] = float(active_demo_meta.get("fps") or 30.0)
            probe_settings["vae_tiling"] = True
            probe_ref_mode = _normalize_probe_ref_mode(settings)
            probe_settings["ref_mode"] = probe_ref_mode
            probe_reference_count = int(_autotune_reference_count(settings))
            if probe_ref_mode == "sr_image" and active_demo_ref_path.exists():
                probe_settings["ref_source_path"] = str(active_demo_ref_path)
                probe_settings["ref_indices"] = ",".join(
                    str(index * 4) for index in range(max(1, probe_reference_count))
                )
            elif probe_ref_mode == "no_ref":
                probe_settings["ref_source_path"] = ""
                probe_settings["ref_indices"] = ""

            label = f"Test {run_counter}: chunk_len={chunk_len}, tile={tile if tile > 0 else 'full-frame'}"
            split_stage_requested = bool(probe_settings.get("split_stage_subprocesses", True))
            _append_log(
                f"{label} - starting | probe_frames={active_demo_meta.get('written_frames')} | "
                f"split_stage={'ON' if split_stage_requested else 'OFF'}"
            )
            if progress:
                progress(min(0.99, float(run_counter - 1) / float(total_estimated_runs)), desc=label)

            _wait_for_vram_drain(telemetry_gpu_ids, ambient_used_gb, float(total_vram_gb), _append_log)

            phase_state: Dict[str, Any] = {
                "phase": "startup",
                "chunks": 0,
                "tiles": 0,
                "threshold_any_phase": True,
            }
            probe_cancel_event = threading.Event()
            setattr(probe_cancel_event, "sparkvsr_cancel_reason", "vram_threshold")
            sampler_stop = threading.Event()
            sampler_box: Dict[str, Any] = {}
            live_queue: "queue.Queue[str]" = queue.Queue()
            result_box: Dict[str, Any] = {}
            process_re = re.compile(r"Processing:\s*F=(\d+)\s+H=(\d+)\s+W=(\d+)\s*\|\s*Chunks=(\d+)\s+Tiles=(\d+)", re.IGNORECASE)

            def _probe_progress(msg: str) -> None:
                line = str(msg or "").strip()
                if not line:
                    return
                with suppress(Exception):
                    live_queue.put_nowait(line)
                lc = line.lower()
                if "phase=tile" in lc or "running sparkvsr transformer" in lc:
                    phase_state["phase"] = "phase2"
                elif "phase=references" in lc:
                    phase_state["phase"] = "references"
                elif "phase=model_load" in lc or "loading pipeline" in lc:
                    phase_state["phase"] = "model_load"
                elif "phase=encode_output" in lc or "phase=complete" in lc:
                    phase_state["phase"] = "finish"
                m = process_re.search(line)
                if m:
                    phase_state["frames"] = int(m.group(1))
                    phase_state["height"] = int(m.group(2))
                    phase_state["width"] = int(m.group(3))
                    phase_state["chunks"] = int(m.group(4))
                    phase_state["tiles"] = int(m.group(5))
                if cancel_event.is_set():
                    setattr(probe_cancel_event, "sparkvsr_cancel_reason", "user_cancelled")
                    probe_cancel_event.set()

            def _run_probe_worker() -> None:
                try:
                    result_box["result"] = run_sparkvsr(
                        probe_settings,
                        base_dir,
                        on_progress=_probe_progress,
                        cancel_event=probe_cancel_event,
                        process_handle=None,
                    )
                except Exception as exc:
                    result_box["result"] = SparkVSRResult(1, None, f"SparkVSR probe error: {exc}")

            sampler_thread = threading.Thread(
                target=_sample_peak_vram_gb,
                args=(
                    sampler_stop,
                    sampler_box,
                    telemetry_gpu_ids,
                    AUTOTUNE_VRAM_SAMPLE_INTERVAL_SEC,
                    phase_state,
                    probe_cancel_event,
                    min_free_vram_target_gb,
                    float(total_vram_gb),
                ),
                daemon=True,
            )
            probe_thread = threading.Thread(target=_run_probe_worker, daemon=True)
            sampler_thread.start()
            probe_thread.start()
            try:
                last_ui_emit = 0.0
                last_live_line = ""
                last_outer_progress = min(0.99, float(run_counter - 1) / float(total_estimated_runs))
                while probe_thread.is_alive() or not live_queue.empty():
                    drained = False
                    for _ in range(80):
                        try:
                            raw_line = live_queue.get_nowait()
                        except queue.Empty:
                            break
                        drained = True
                        display_line, transient = _normalize_probe_progress_line(raw_line)
                        if not display_line:
                            continue
                        last_live_line = display_line
                        _append_probe_output(f"{label} | {display_line}", transient=transient)
                        pct = _parse_sparkvsr_progress_fraction(display_line)
                        if pct is not None and progress:
                            outer_pct = min(
                                0.99,
                                (float(run_counter - 1) + max(0.0, min(1.0, float(pct)))) / float(total_estimated_runs),
                            )
                            outer_pct = max(last_outer_progress, outer_pct)
                            last_outer_progress = outer_pct
                            progress(outer_pct, desc=display_line[:120])

                    if cancel_event.is_set():
                        setattr(probe_cancel_event, "sparkvsr_cancel_reason", "user_cancelled")
                        probe_cancel_event.set()

                    now = time.time()
                    if now - last_ui_emit >= AUTOTUNE_UI_UPDATE_INTERVAL_SEC:
                        last_ui_emit = now
                        peak_live = float(sampler_box.get("max_used_gb", 0.0) or 0.0)
                        total_live = float(sampler_box.get("total_gb", 0.0) or total_vram_gb or 0.0)
                        free_live = max(0.0, total_live - peak_live) if total_live > 0 and peak_live > 0 else 0.0
                        vram_note = (
                            f" | live peak={peak_live:.2f}GB, free={free_live:.2f}GB"
                            if peak_live > 0
                            else ""
                        )
                        detail = last_live_line or "waiting for SparkVSR progress output"
                        yield _payload(
                            f"{label} | {detail}{vram_note}",
                            show_indicator=True,
                            tile_value=int(tile),
                            overlap_hw_value=int(candidate["overlap_height"]),
                            chunk_value=int(chunk_len),
                            overlap_t_value=int(candidate["overlap_t"]),
                            vae_tiling_value=True,
                        )

                    if not drained:
                        time.sleep(0.10)

                probe_thread.join(timeout=1.0)
            finally:
                sampler_stop.set()
                sampler_thread.join(timeout=2.0)
                _finish_cmd_inline()

            result = result_box.get("result")
            if not isinstance(result, SparkVSRResult):
                result = SparkVSRResult(1, None, "SparkVSR probe ended without a result.")

            with suppress(Exception):
                outp = Path(str(probe_settings.get("output_override") or ""))
                if outp.exists():
                    outp.unlink(missing_ok=True)

            max_used_gb = float(sampler_box.get("max_used_gb", 0.0) or 0.0)
            measured_total = float(sampler_box.get("total_gb", 0.0) or total_vram_gb or 0.0)
            telemetry_ok = bool(sampler_box.get("telemetry_ok", False))
            early_stop_reason = str(sampler_box.get("early_stop_reason") or "")
            total_for_eval = measured_total if measured_total > 0 else float(total_vram_gb)
            aggregate_free_gb = (
                max(0.0, total_for_eval - max_used_gb)
                if total_for_eval > 0
                else 0.0
            )
            min_device_free_gb = float(
                sampler_box.get("min_device_free_gb", aggregate_free_gb)
                or 0.0
            )
            free_gb = min(aggregate_free_gb, min_device_free_gb)
            oom = bool(_looks_like_oom(result.log))
            oom_phase = _detect_sparkvsr_oom_phase(result.log) if oom else ""
            canceled_by_user = bool(cancel_event.is_set())
            result_log = str(result.log or "")
            result_log_lc = result_log.lower()
            split_cli_flag = bool("--split_stage_subprocesses" in result_log)
            split_runtime_seen = bool(
                "stage subprocess isolation enabled" in result_log_lc
                or "[sparkvsr split] internal stage started" in result_log_lc
            )
            split_validation_ok = bool(
                (not split_stage_requested)
                or (split_cli_flag and split_runtime_seen)
            )
            if split_stage_requested and not split_cli_flag:
                _append_log(f"{label} warning: split-stage was requested but the CLI flag was not found in the runner log.")
            elif split_stage_requested and not split_runtime_seen:
                _append_log(f"{label} warning: split-stage workers were not observed in the runner log.")
            passed = bool(
                int(result.returncode) == 0
                and telemetry_ok
                and (not oom)
                and (not canceled_by_user)
                and not early_stop_reason
                and split_validation_ok
                and free_gb >= float(min_free_vram_target_gb)
            )
            return {
                "chunk_len": int(chunk_len),
                "overlap_t": int(candidate["overlap_t"]),
                "tile_height": int(tile),
                "tile_width": int(tile),
                "overlap_height": int(candidate["overlap_height"]),
                "overlap_width": int(candidate["overlap_width"]),
                "returncode": int(result.returncode),
                "oom": bool(oom),
                "oom_phase": str(oom_phase),
                "max_vram_used_gb": round(max_used_gb, 3),
                "total_vram_gb": round(total_for_eval, 3),
                "estimated_free_gb": round(free_gb, 3),
                "min_device_free_gb": round(min_device_free_gb, 3),
                "telemetry_ok": bool(telemetry_ok),
                "samples": int(sampler_box.get("samples", 0) or 0),
                "phase2_samples": int(sampler_box.get("phase2_samples", 0) or 0),
                "probe_cancel_reason": early_stop_reason,
                "split_stage_subprocesses": bool(split_stage_requested),
                "split_stage_cli_flag": bool(split_cli_flag),
                "split_stage_runtime_seen": bool(split_runtime_seen),
                "split_stage_validation_ok": bool(split_validation_ok),
                "probe_frames": int(active_demo_meta.get("written_frames") or 0),
                "reference_count": int(probe_reference_count),
                "passed": bool(passed),
                "quality_rank": int(_quality_rank(candidate)),
                "frames": int(phase_state.get("frames") or 0),
                "chunks": int(phase_state.get("chunks") or 0),
                "tiles": int(phase_state.get("tiles") or 0),
            }

        def _apply_passed_outcome(outcome: Dict[str, Any]) -> None:
            nonlocal best_config, status_reason
            candidate_rank = int(outcome["quality_rank"])
            if best_config is not None and candidate_rank < int(best_config.get("quality_rank") or 0):
                return
            best_config = {
                "chunk_len": int(outcome["chunk_len"]),
                "overlap_t": int(outcome["overlap_t"]),
                "tile_height": int(outcome["tile_height"]),
                "tile_width": int(outcome["tile_width"]),
                "overlap_height": int(outcome["overlap_height"]),
                "overlap_width": int(outcome["overlap_width"]),
                "vae_tiling": True,
                "min_free_vram_target_gb": float(min_free_vram_target_gb),
                "measured_peak_vram_used_gb": float(outcome["max_vram_used_gb"]),
                "estimated_free_vram_gb": float(outcome["estimated_free_gb"]),
                "min_device_free_vram_gb": float(
                    outcome.get("min_device_free_gb", outcome["estimated_free_gb"])
                ),
                "quality_rank": candidate_rank,
                "temporal_policy": "full_sequence_allowed" if allow_full_sequence else "bounded_chunks_required",
                "source_frame_count": int(source_frame_count) if source_frame_count is not None else None,
                "split_stage_subprocesses": bool(outcome.get("split_stage_subprocesses", True)),
                "probe_frames": int(outcome.get("probe_frames") or 0),
            }
            status_reason = "completed" if candidate_rank >= int(top_rank) else "threshold_reached"

        def _candidate_history_key(candidate: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
            return (
                int(candidate["chunk_len"]),
                int(candidate["tile_height"]),
                int(candidate["tile_width"]),
                int(candidate["overlap_t"]),
                int(candidate["overlap_height"]),
                int(candidate["overlap_width"]),
            )

        def _classify_outcome(outcome: Optional[Dict[str, Any]]) -> str:
            if outcome is None:
                return "hard_fail"
            if bool(outcome.get("passed", False)):
                return "pass"
            if is_vram_boundary_outcome(outcome, min_free_vram_target_gb):
                return "boundary_fail"
            return "hard_fail"

        def _probe_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
            if cancel_event.is_set():
                _append_log("Auto Tune cancelled before next test.")
                raise KeyboardInterrupt()
            history_key = _candidate_history_key(candidate)
            known_outcome = history_outcomes_by_key.get(history_key)
            if isinstance(known_outcome, dict):
                outcome = dict(known_outcome)
                outcome["_autotune_reused"] = True
                saved_verdict = (
                    "fits"
                    if bool(outcome.get("passed", False))
                    else (
                        "is a saved VRAM boundary"
                        if is_vram_boundary_outcome(outcome, min_free_vram_target_gb)
                        else "was already attempted"
                    )
                )
                _append_log(
                    f"Chunk {outcome.get('chunk_len')}, tile "
                    f"{outcome.get('tile_height') or 'full-frame'}: already tested earlier - "
                    f"reusing that measurement ({saved_verdict})."
                )
                return outcome
            if int(candidate["chunk_len"]) > int(initial_probe_frames):
                yield from _ensure_growth_demo()
            yield _payload(
                (
                    f"Testing chunk_len={candidate['chunk_len']}, "
                    f"tile={candidate['tile_height'] if candidate['tile_height'] > 0 else 'full-frame'}"
                ),
                show_indicator=True,
                tile_value=int(candidate["tile_height"]),
                overlap_hw_value=int(candidate["overlap_height"]),
                chunk_value=int(candidate["chunk_len"]),
                overlap_t_value=int(candidate["overlap_t"]),
                vae_tiling_value=True,
            )
            outcome = yield from _run_probe_once(candidate)
            outcome["autotune_attempt"] = 1
            tests.append(outcome)
            history_outcomes_by_key[history_key] = dict(outcome)
            verdict = _classify_outcome(outcome)
            if verdict == "pass":
                verdict_note = "PASSED (full run)"
            elif verdict == "boundary_fail" and str(outcome.get("probe_cancel_reason") or "") == "threshold_reached":
                verdict_note = "needs too much VRAM (test stopped early to save time)"
            elif verdict == "boundary_fail" and outcome.get("oom"):
                verdict_note = "ran out of VRAM"
            elif not bool(outcome.get("telemetry_ok", False)):
                verdict_note = "failed (VRAM telemetry unavailable)"
            else:
                verdict_note = f"failed (exit code {outcome.get('returncode')})"
            _append_log(
                f"Tested Chunk {outcome['chunk_len']}, tile {outcome['tile_height'] or 'full-frame'}: "
                f"peak {outcome['max_vram_used_gb']:.2f}GB, "
                f"{outcome['estimated_free_gb']:.2f}GB left free - {verdict_note}."
            )
            _persist("running")
            return outcome

        temporal_frontier_verified = False
        spatial_frontier_verified = False
        search_indeterminate = False
        search_results: Dict[str, Any] = {}

        def _record_search_health(label: str, result: Dict[str, Any]) -> None:
            nonlocal search_indeterminate
            hard_indices = list(result.get("hard_failed_indices") or [])
            if hard_indices:
                _append_log(
                    f"{label} search saw persistent non-VRAM failures at candidate indices "
                    f"{hard_indices}; they were not treated as OOM boundaries."
                )
            if bool(
                result.get("indeterminate_above_best", False)
                or (hard_indices and not result.get("frontier_verified", False))
            ):
                search_indeterminate = True
                _append_log(
                    f"{label} search could not prove every setting above the selected result. "
                    "The best measured setting can be applied, but this run will not be cached as final."
                )

        try:
            vram_budget_gb = float(total_vram_gb) - float(min_free_vram_target_gb)
            temporal_peaks = {
                int(chunk_k): float(peak)
                for (chunk_k, tile_k), peak in prior_peaks_by_config.items()
                if int(tile_k) == int(safest_tile)
            }
            temporal_seed = predict_frontier_index(temporal_values, temporal_peaks, vram_budget_gb)
            if temporal_seed is not None:
                vram_list = ", ".join(
                    f"{v:.0f}GB" for v in sorted(v for v in prior_source_vrams if v > 0)
                ) or "different-size"
                _append_log(
                    f"Measurements from a {vram_list} GPU predict a temporal frontier near "
                    f"Chunk Length {temporal_values[int(temporal_seed)]}. The prediction is only a "
                    "hint; this GPU still starts at the safest candidate and grows gradually."
                )
            _append_log(
                "Finding the longest safe Temporal Chunk at the smallest spatial tile "
                f"(candidates {temporal_values}; longer improves consistency and usually speed)."
            )

            def _temporal_probe(idx: int, _require_full: bool):
                outcome = yield from _probe_candidate(temporal_candidates[int(idx)])
                verdict = "cancelled" if cancel_event.is_set() else _classify_outcome(outcome)
                if verdict == "pass":
                    _apply_passed_outcome(outcome)
                return {
                    "outcome": verdict,
                    "early_stopped_pass": False,
                    "reused_outcome": bool(outcome.get("_autotune_reused", False)),
                }

            temporal_saved_pass, temporal_saved_boundary = resume_frontier_hints(
                [
                    history_outcomes_by_key.get(_candidate_history_key(candidate))
                    for candidate in temporal_candidates
                ],
                min_free_vram_target_gb,
            )
            temporal_res = yield from frontier_bisect(
                len(temporal_candidates),
                _temporal_probe,
                initial_index=temporal_seed,
                max_growth_step=1,
                trusted_pass_index=temporal_saved_pass,
                known_boundary_index=temporal_saved_boundary,
            )
            search_results["temporal"] = dict(temporal_res)
            _record_search_health("Temporal", temporal_res)
            temporal_frontier_verified = bool(temporal_res.get("frontier_verified", False))

            temporal_best_idx = temporal_res.get("best_idx")
            if temporal_res.get("stopped") == "cancelled":
                status_reason = "cancelled"
            elif temporal_best_idx is None:
                status_reason = "failed"
            else:
                selected_chunk = int(temporal_candidates[int(temporal_best_idx)]["chunk_len"])
                spatial_candidates = [
                    {**template, "chunk_len": selected_chunk}
                    for template in spatial_templates
                ]
                tile_values = [int(candidate["tile_height"]) for candidate in spatial_candidates]
                tile_peaks = {
                    int(tile_k): float(peak)
                    for (chunk_k, tile_k), peak in prior_peaks_by_config.items()
                    if int(chunk_k) == selected_chunk
                }
                tile_seed = predict_frontier_index(tile_values, tile_peaks, vram_budget_gb)
                if tile_seed is not None:
                    _append_log(
                        "Compatible historical measurements predict the spatial frontier near "
                        f"{tile_values[int(tile_seed)] if tile_values[int(tile_seed)] > 0 else 'full-frame'} "
                        f"at Chunk Length {selected_chunk}. Safe bounded growth will verify it locally."
                    )
                _append_log(
                    f"Temporal frontier settled at Chunk Length {selected_chunk}. Now maximizing "
                    f"the spatial tile among {['full-frame' if t <= 0 else t for t in tile_values]}."
                )

                def _spatial_probe(idx: int, _require_full: bool):
                    outcome = yield from _probe_candidate(spatial_candidates[int(idx)])
                    verdict = "cancelled" if cancel_event.is_set() else _classify_outcome(outcome)
                    if verdict == "pass":
                        _apply_passed_outcome(outcome)
                    return {
                        "outcome": verdict,
                        "early_stopped_pass": False,
                        "reused_outcome": bool(outcome.get("_autotune_reused", False)),
                    }

                spatial_saved_pass, spatial_saved_boundary = resume_frontier_hints(
                    [
                        history_outcomes_by_key.get(_candidate_history_key(candidate))
                        for candidate in spatial_candidates
                    ],
                    min_free_vram_target_gb,
                )
                spatial_res = yield from frontier_bisect(
                    len(spatial_candidates),
                    _spatial_probe,
                    initial_index=tile_seed,
                    max_growth_step=1,
                    trusted_pass_index=spatial_saved_pass,
                    known_boundary_index=spatial_saved_boundary,
                )
                search_results["spatial"] = dict(spatial_res)
                _record_search_health("Spatial", spatial_res)
                spatial_frontier_verified = bool(spatial_res.get("frontier_verified", False))
                if spatial_res.get("stopped") == "cancelled":
                    status_reason = "cancelled"
                elif spatial_res.get("best_idx") is None:
                    status_reason = "failed"

            if search_indeterminate and best_config is not None and status_reason != "cancelled":
                status_reason = "failed"
        except KeyboardInterrupt:
            status_reason = "cancelled"

        if campaign_force_fresh and status_reason != "cancelled":
            _append_log(
                "Campaign profiling: the 2GB reserve cutoff remains active; continuing the "
                f"lowest spatial-tile ladder toward {campaign_profile_min_peak_gb:.1f}GB peak VRAM."
            )
            profile_floor_reached = False
            for profile_chunk in sorted(AUTOTUNE_TEMPORAL_CANDIDATES):
                chunk_candidates = [
                    candidate
                    for candidate in candidates
                    if int(candidate.get("chunk_len") or 0) == int(profile_chunk)
                    and int(candidate.get("tile_height") or 0) > 0
                ]
                if not chunk_candidates:
                    continue
                profile_candidate = min(
                    chunk_candidates,
                    key=lambda item: int(item.get("tile_height") or 0),
                )
                outcome = yield from _probe_candidate(profile_candidate)
                if bool(outcome.get("passed", False)):
                    _apply_passed_outcome(outcome)
                peak_gb = float(outcome.get("max_vram_used_gb", 0.0) or 0.0)
                clean_measurement = bool(
                    int(outcome.get("returncode", 1)) == 0
                    and bool(outcome.get("telemetry_ok", False))
                    and (not bool(outcome.get("oom", False)))
                    and (not str(outcome.get("probe_cancel_reason") or ""))
                )
                _persist("running")
                if clean_measurement and peak_gb <= float(campaign_profile_min_peak_gb):
                    profile_floor_reached = True
                    _append_log(
                        f"Campaign profiling reached {peak_gb:.2f}GB peak at Chunk Length "
                        f"{int(profile_chunk)}, Spatial Tile {int(profile_candidate['tile_height'])}."
                    )
                    break
                if clean_measurement:
                    _append_log(
                        f"The safest Chunk Length {int(profile_chunk)} already peaks at {peak_gb:.2f}GB, "
                        f"above the {campaign_profile_min_peak_gb:.1f}GB profiling target. "
                        "Larger chunks cannot lower the memory floor, so profiling stopped safely."
                    )
                    break
            if not profile_floor_reached and status_reason != "cancelled":
                measured = [
                    float(item.get("max_vram_used_gb", 0.0) or 0.0)
                    for item in tests
                    if bool(item.get("telemetry_ok", False))
                    and int(item.get("returncode", 1)) == 0
                    and (not bool(item.get("oom", False)))
                    and float(item.get("max_vram_used_gb", 0.0) or 0.0) > 0
                ]
                floor_note = min(measured) if measured else 0.0
                _append_log(
                    "Campaign profiling exhausted the low-memory ladder; "
                    f"lowest clean measured peak was {floor_note:.2f}GB."
                )

        autotune_payload["search_results"] = dict(search_results)
        frontier_ok = bool(
            best_config
            and temporal_frontier_verified
            and spatial_frontier_verified
            and not search_indeterminate
            and status_reason in {"completed", "threshold_reached"}
        )
        _persist(status_reason, finalized=True, frontier_verified=frontier_ok)
        if autotune_log_path:
            _append_log(f"Saved autotune log: {autotune_log_path}")

        with suppress(Exception):
            if session_dir:
                shutil.rmtree(session_dir, ignore_errors=True)
        if progress:
            progress(1.0, desc="Auto Tune complete")

        if best_config:
            spark_cfg = state.setdefault("seed_controls", {}).setdefault("sparkvsr_settings", {})
            spark_cfg["tile_height"] = int(best_config["tile_height"])
            spark_cfg["tile_width"] = int(best_config["tile_width"])
            spark_cfg["overlap_height"] = int(best_config["overlap_height"])
            spark_cfg["overlap_width"] = int(best_config["overlap_width"])
            spark_cfg["chunk_len"] = int(best_config["chunk_len"])
            spark_cfg["overlap_t"] = (
                int(best_config["overlap_t"])
                if int(best_config["chunk_len"]) > 0
                else int(AUTOTUNE_TEMPORAL_OVERLAP)
            )
            spark_cfg["vae_tiling"] = True
            spark_cfg["save_vram_gb"] = float(min_free_vram_target_gb)
            state["operation_status"] = "completed" if status_reason in {"completed", "threshold_reached"} else "ready"
            summary_md = (
                "**SparkVSR Auto Tune Result**\n"
                f"- Temporal Policy: `{temporal_policy_label}`{source_frames_note}\n"
                f"- Spatial Tile: `{spark_cfg['tile_height']}x{spark_cfg['tile_width']}` "
                f"({'disabled/full-frame' if int(spark_cfg['tile_height']) <= 0 else 'enabled'})\n"
                f"- Spatial Overlap: `{spark_cfg['overlap_height']}x{spark_cfg['overlap_width']}`\n"
                f"- Temporal Chunk Length: `{spark_cfg['chunk_len']}` "
                f"({'disabled/full sequence' if int(spark_cfg['chunk_len']) <= 0 else 'enabled'})\n"
                f"- Temporal Overlap: `{spark_cfg['overlap_t']}`\n"
                f"- VAE Tiling: `ON`\n"
                f"- Stage Subprocess Isolation: `{'ON' if bool(settings.get('split_stage_subprocesses', True)) else 'OFF'}`\n"
                f"- Peak VRAM: `{float(best_config.get('measured_peak_vram_used_gb') or 0.0):.2f} GB`\n"
                f"- Free VRAM estimate: `{float(best_config.get('estimated_free_vram_gb') or 0.0):.2f} GB` "
                f"(target >= `{min_free_vram_target_gb:.1f} GB`)\n"
                f"- Log file: `{str(autotune_log_path) if autotune_log_path else 'not saved'}`"
            )
            applied_note = (
                f"applied Chunk Length {spark_cfg['chunk_len']}, "
                f"Spatial Tile {'full-frame' if int(spark_cfg['tile_height']) <= 0 else spark_cfg['tile_height']}, "
                f"Temporal Overlap {spark_cfg['overlap_t']}"
            )
            if status_reason == "completed":
                final_status = f"Auto Tune complete - {applied_note}."
            elif status_reason == "threshold_reached":
                final_status = f"Auto Tune finished - {applied_note} (highest config that fits your VRAM)."
            elif status_reason == "cancelled":
                final_status = f"Auto Tune cancelled - {applied_note} (best found so far)."
            else:
                final_status = f"Auto Tune hit an error - {applied_note} (best found before the error)."
            _append_log(
                f"Result: Chunk Length {spark_cfg['chunk_len']}, "
                f"Spatial Tile {'full-frame' if int(spark_cfg['tile_height']) <= 0 else spark_cfg['tile_height']}, "
                f"Temporal Overlap {spark_cfg['overlap_t']} "
                f"(measured peak {float(best_config.get('measured_peak_vram_used_gb') or 0.0):.2f}GB, "
                f"keeps {float(best_config.get('estimated_free_vram_gb') or 0.0):.2f}GB of "
                f"{total_vram_gb:.1f}GB free)."
            )
            yield _payload(
                final_status,
                show_indicator=False,
                tile_value=int(spark_cfg["tile_height"]),
                overlap_hw_value=int(spark_cfg["overlap_height"]),
                chunk_value=int(spark_cfg["chunk_len"]),
                overlap_t_value=int(spark_cfg["overlap_t"]),
                vae_tiling_value=True,
                summary_text=summary_md,
            )
            return

        state["operation_status"] = "error" if status_reason == "failed" else "ready"
        if status_reason == "cancelled":
            final_msg = "Auto Tune cancelled before a stable config was found."
        elif status_reason == "failed":
            final_msg = "Auto Tune failed before finding a safe config."
        else:
            final_msg = "Auto Tune stopped at VRAM threshold before finding a safe config."
        yield _payload(final_msg, show_indicator=False)
        return

    except Exception as exc:
        state["operation_status"] = "error"
        _append_log(f"Auto Tune error: {exc}")
        yield _payload("Auto Tune failed due to an internal error.", show_indicator=False)
        return
