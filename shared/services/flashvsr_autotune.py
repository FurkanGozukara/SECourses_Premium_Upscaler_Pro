"""
FlashVSR+ runtime auto-tune implementation.
"""

import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr

from shared.flashvsr_runner import FlashVSRResult, run_flashvsr
from shared.gpu_utils import get_global_gpu_override
from shared.oom_alert import clear_vram_oom_alert
from shared.path_utils import IMAGE_EXTENSIONS, detect_input_type, get_media_dimensions, normalize_path
from shared.resolution_calculator import estimate_fixed_scale_upscale_plan_from_dims


AUTOTUNE_MODEL_ID = "flashvsrplus"
AUTOTUNE_LOG_PREFIX = "flashvsrplus_autotune"
AUTOTUNE_TARGET_FRAMES = 450
AUTOTUNE_MIN_FREE_VRAM_GB = 2.0
AUTOTUNE_BASE_TILE_SIZE = 256
AUTOTUNE_FALLBACK_TILES = (128, 64)
AUTOTUNE_TILE_STEP = 32
AUTOTUNE_TILE_MAX = 1024
AUTOTUNE_PRIMARY_OVERLAP = 48
AUTOTUNE_FALLBACK_OVERLAP = 24
AUTOTUNE_FRAME_CHUNK_SEQUENCE = (450, 350, 250, 150, 100, 64, 48, 32)
AUTOTUNE_PHASE2_MIN_SAMPLES_FOR_EARLY_STOP = 4
AUTOTUNE_PHASE2_MIN_SECONDS_FOR_EARLY_STOP = 0.60
AUTOTUNE_PHASE2_MIN_ITER_FOR_GATE = 3


def _within_ratio(lhs: float, rhs: float, tolerance: float) -> bool:
    try:
        lhs_f = float(lhs)
        rhs_f = float(rhs)
        tol_f = abs(float(tolerance))
    except Exception:
        return False
    base = max(abs(lhs_f), abs(rhs_f), 1e-9)
    return abs(lhs_f - rhs_f) / base <= tol_f


def _exact_payload_for_cache_lookup(signature: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize signature.exact for cache matching across different input-scale routes."""
    exact = signature.get("exact")
    if not isinstance(exact, dict):
        return {}
    out = dict(exact)
    # Same output target can be reached via different requested scales (e.g. 960x540@x4 vs 1920x1080@x2).
    # Reuse the same cache lane by ignoring scale-only route differences.
    out.pop("scale", None)
    return out


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


def _parse_cuda_device_ids(cuda_spec: Any) -> List[int]:
    raw = str(cuda_spec or "").strip().lower()
    if not raw:
        return []
    out: List[int] = []
    for token in raw.split(","):
        norm = token.strip()
        if norm.startswith("cuda:"):
            norm = norm.split(":", 1)[1].strip()
        if not norm.isdigit():
            continue
        idx = int(norm)
        if idx not in out:
            out.append(idx)
    return out


def _query_gpu_memory_snapshot_gb() -> Dict[int, Tuple[float, float]]:
    """Return per-GPU memory usage as {gpu_index: (used_gb, total_gb)}."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.5,
            check=False,
        )
        if proc.returncode != 0:
            return {}
        rows = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        out: Dict[int, Tuple[float, float]] = {}
        for row in rows:
            parts = [p.strip() for p in row.split(",")]
            if len(parts) < 3 or (not parts[0].isdigit()):
                continue
            idx = int(parts[0])
            used_mb = float(parts[1]) if parts[1] else 0.0
            total_mb = float(parts[2]) if parts[2] else 0.0
            out[idx] = (used_mb / 1024.0, total_mb / 1024.0)
        return out
    except Exception:
        return {}


def _sample_peak_vram_gb(
    stop_event: threading.Event,
    result_box: Dict[str, Any],
    device_ids: List[int],
    interval_sec: float = 0.25,
    phase_state: Optional[Dict[str, Any]] = None,
    probe_cancel_event: Optional[threading.Event] = None,
    min_free_target_gb: Optional[float] = None,
    total_vram_hint_gb: float = 0.0,
) -> None:
    """
    Track peak VRAM and optionally stop probe early once phase2 has enough samples.

    Probe cancellation is triggered by:
    - crossing free-VRAM threshold (<2GB free), or
    - collecting enough phase2 samples (~0.6s+) for fast pass/fail inference.
    """
    peak_used = 0.0
    peak_phase2 = 0.0
    total_seen = 0.0
    telemetry_ok = False
    samples = 0
    phase2_samples = 0
    phase2_started_at: Optional[float] = None

    while not stop_event.is_set():
        snap = _query_gpu_memory_snapshot_gb()
        if snap:
            selected = [d for d in device_ids if d in snap] if device_ids else sorted(snap.keys())
            if selected:
                telemetry_ok = True
                used_sum = sum(float(snap[d][0]) for d in selected)
                total_sum = sum(float(snap[d][1]) for d in selected)
                peak_used = max(peak_used, used_sum)
                samples += 1

                cur_phase = str((phase_state or {}).get("phase") or "").strip().lower()
                if cur_phase == "phase2":
                    peak_phase2 = max(peak_phase2, used_sum)
                    phase2_samples += 1
                    if phase2_started_at is None:
                        phase2_started_at = time.time()

                if total_sum > 0:
                    total_seen = total_sum

                # Write live stats for probe-side decisions.
                result_box["max_used_gb"] = float(peak_used)
                result_box["max_used_phase2_gb"] = float(peak_phase2)
                result_box["samples"] = int(samples)
                result_box["phase2_samples"] = int(phase2_samples)
                result_box["total_gb"] = float(total_seen if total_seen > 0 else total_vram_hint_gb)
                result_box["telemetry_ok"] = bool(telemetry_ok)

                if probe_cancel_event is not None and (not probe_cancel_event.is_set()) and phase2_samples > 0:
                    total_for_eval = float(total_seen if total_seen > 0 else total_vram_hint_gb)
                    free_est = (
                        max(0.0, total_for_eval - peak_phase2)
                        if total_for_eval > 0
                        else max(0.0, float(total_vram_hint_gb) - peak_phase2)
                    )
                    phase2_gate_ready = bool((phase_state or {}).get("phase2_gate_ready", False))

                    if min_free_target_gb is not None and free_est < float(min_free_target_gb):
                        result_box["early_stop_reason"] = "threshold_reached"
                        probe_cancel_event.set()
                    elif (
                        phase2_gate_ready
                        and
                        phase2_started_at is not None
                        and phase2_samples >= int(AUTOTUNE_PHASE2_MIN_SAMPLES_FOR_EARLY_STOP)
                        and (time.time() - phase2_started_at) >= float(AUTOTUNE_PHASE2_MIN_SECONDS_FOR_EARLY_STOP)
                    ):
                        result_box["early_stop_reason"] = "enough_phase2_samples"
                        probe_cancel_event.set()

        stop_event.wait(max(0.05, float(interval_sec)))

    result_box["max_used_gb"] = float(peak_used)
    result_box["max_used_phase2_gb"] = float(peak_phase2)
    result_box["samples"] = int(samples)
    result_box["phase2_samples"] = int(phase2_samples)
    if total_seen > 0:
        result_box["total_gb"] = float(total_seen)
    elif float(total_vram_hint_gb) > 0:
        result_box["total_gb"] = float(total_vram_hint_gb)
    result_box["telemetry_ok"] = bool(telemetry_ok)


def _looks_like_oom(log_text: str) -> bool:
    lc = str(log_text or "").lower()
    tokens = (
        "out of memory",
        "cuda out of memory",
        "torch.cuda.outofmemoryerror",
        "failed to allocate memory",
        "cannot recover",
    )
    return any(tok in lc for tok in tokens)


def _detect_flashvsr_oom_phase(log_text: str) -> str:
    lc = str(log_text or "").lower()
    if ("vae" in lc and ("encoding" in lc or "encode" in lc)) or ("phase 1" in lc):
        return "phase1_encode"
    if ("vae" in lc and ("decoding" in lc or "decode" in lc)) or ("phase 3" in lc) or ("tcdecoder" in lc):
        return "phase3_decode"
    if ("processing tiles" in lc) or ("dit" in lc) or ("upscal" in lc) or ("phase 2" in lc):
        return "phase2_upscale"
    return "unknown"


def _build_autotune_signature(
    settings: Dict[str, Any],
    *,
    target_w: int,
    target_h: int,
    effective_in_w: int,
    effective_in_h: int,
    global_gpu_device: str,
    total_vram_gb: float,
) -> Dict[str, Any]:
    exact_payload = {
        "autotune_model": AUTOTUNE_MODEL_ID,
        "version": str(settings.get("version") or "1.1"),
        "mode": str(settings.get("mode") or "full"),
        "scale": int(settings.get("scale") or 4),
        "vae_model": str(settings.get("vae_model") or "Wan2.2"),
        "precision": str(settings.get("precision") or "bf16"),
        "attention_mode": str(settings.get("attention_mode") or "flash_attention_2"),
        "tiled_vae": bool(settings.get("tiled_vae", False)),
        "unload_dit": bool(settings.get("unload_dit", True)),
        "keep_models_on_cpu": bool(settings.get("keep_models_on_cpu", True)),
        "force_offload": bool(settings.get("force_offload", True)),
        "stream_decode": bool(settings.get("stream_decode", False)),
        "sparse_ratio": round(float(settings.get("sparse_ratio") or 2.0), 4),
        "kv_ratio": round(float(settings.get("kv_ratio") or 3.0), 4),
        "local_range": int(settings.get("local_range") or 11),
        "cfg_scale": round(float(settings.get("cfg_scale") or 1.0), 4),
        "denoise_amount": round(float(settings.get("denoise_amount") or 1.0), 4),
        "color_fix": bool(settings.get("color_fix", True)),
        "max_target_resolution": int(settings.get("max_target_resolution") or 0),
        "pre_downscale_then_upscale": bool(settings.get("pre_downscale_then_upscale", True)),
        "use_resolution_tab": bool(settings.get("use_resolution_tab", True)),
        "global_gpu_device": str(global_gpu_device or ""),
        "test_target_frames": int(AUTOTUNE_TARGET_FRAMES),
        "test_overlap_primary": int(AUTOTUNE_PRIMARY_OVERLAP),
        "test_overlap_fallback": int(AUTOTUNE_FALLBACK_OVERLAP),
    }
    exact_blob = json.dumps(exact_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "exact": exact_payload,
        "exact_hash": hashlib.sha1(exact_blob.encode("utf-8")).hexdigest(),
        "target_pixels": int(max(1, int(target_w) * int(target_h))),
        "effective_input_pixels": int(max(1, int(effective_in_w) * int(effective_in_h))),
        "target_width": int(target_w),
        "target_height": int(target_h),
        "effective_input_width": int(effective_in_w),
        "effective_input_height": int(effective_in_h),
        "gpu_total_vram_gb": float(total_vram_gb or 0.0),
    }


def _autotune_signature_matches(candidate: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    if not isinstance(candidate, dict) or not isinstance(expected, dict):
        return False
    cand_hash = str(candidate.get("exact_hash") or "")
    exp_hash = str(expected.get("exact_hash") or "")
    if cand_hash != exp_hash:
        cand_exact = _exact_payload_for_cache_lookup(candidate)
        exp_exact = _exact_payload_for_cache_lookup(expected)
        if (not cand_exact) or (not exp_exact) or cand_exact != exp_exact:
            return False
    if not _within_ratio(
        float(candidate.get("target_pixels", 0) or 0),
        float(expected.get("target_pixels", 0) or 0),
        tolerance=0.05,
    ):
        return False
    cand_vram = float(candidate.get("gpu_total_vram_gb", 0) or 0)
    exp_vram = float(expected.get("gpu_total_vram_gb", 0) or 0)
    if cand_vram > 0 and exp_vram > 0 and (not _within_ratio(cand_vram, exp_vram, tolerance=0.05)):
        return False
    return True


def _find_cached_autotune_log(log_dir: Path, expected_signature: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    def _payload_has_frontier_proof(payload: Dict[str, Any]) -> bool:
        best = payload.get("best_config")
        tests = payload.get("tests")
        if not isinstance(best, dict) or (not best) or (not isinstance(tests, list)):
            return False

        def _rank(cfg: Dict[str, Any]) -> int:
            chunk = int(cfg.get("frame_chunk_size") or 0)
            tile = int(cfg.get("tile_size") or 0)
            overlap_val = int(cfg.get("overlap") or 0)
            tiled_dit_val = bool(cfg.get("tiled_dit", True))
            if not tiled_dit_val:
                return 1_000_000_000 + (chunk * 10_000) - overlap_val
            return (chunk * 10_000) + (tile * 10) - overlap_val

        best_rank = _rank(best)
        for item in tests:
            if not isinstance(item, dict):
                continue
            if bool(item.get("passed", False)):
                continue
            if not bool(item.get("telemetry_ok", False)):
                continue
            cand_rank = _rank(
                {
                    "frame_chunk_size": int(item.get("frame_chunk_size") or 0),
                    "tile_size": int(item.get("tile_size") or 0),
                    "overlap": int(item.get("overlap") or 0),
                    "tiled_dit": bool(item.get("tiled_dit", True)),
                }
            )
            if cand_rank <= best_rank:
                continue
            try:
                free_gb = float(item.get("estimated_free_gb", 1e9) or 1e9)
            except Exception:
                free_gb = 1e9
            if bool(item.get("oom", False)):
                return True
            if str(item.get("probe_cancel_reason") or "").strip().lower() == "threshold_reached":
                return True
            if free_gb < float(AUTOTUNE_MIN_FREE_VRAM_GB):
                return True

        # Top-of-search candidate reached; no higher-quality candidate exists.
        if (
            int(best.get("frame_chunk_size") or 0) >= int(AUTOTUNE_FRAME_CHUNK_SEQUENCE[0])
            and int(best.get("tile_size") or 0) >= int(AUTOTUNE_TILE_MAX)
            and (not bool(best.get("tiled_dit", True)))
        ):
            return True
        return False

    def _is_finalized_and_frontier_verified(payload: Dict[str, Any]) -> bool:
        status = str(payload.get("status") or "").strip().lower()
        if status not in {"completed", "threshold_reached"}:
            return False
        if not bool(payload.get("finalized", False)):
            return False
        # Require actual frontier proof from saved tests; do not rely on legacy flags alone.
        if not _payload_has_frontier_proof(payload):
            return False
        return True

    def _has_strict_phase2_validation(payload: Dict[str, Any]) -> bool:
        tests = payload.get("tests")
        if not isinstance(tests, list):
            return False
        for item in tests:
            if not isinstance(item, dict):
                continue
            if not bool(item.get("passed", False)):
                continue
            # New strict probe quality markers (added after initial implementation).
            if not bool(item.get("phase2_gate_ready", False)):
                continue
            if int(item.get("tile_idx_max", 0) or 0) < 1:
                continue
            iter_idx = int(item.get("iter_idx_max", 0) or 0)
            if iter_idx < int(AUTOTUNE_PHASE2_MIN_ITER_FOR_GATE):
                continue
            return True
        return False

    try:
        if not log_dir.exists():
            return None
        candidates = sorted(
            [p for p in log_dir.glob("*.json") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        return None

    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        is_flash_file = path.name.lower().startswith(f"{AUTOTUNE_LOG_PREFIX}_")
        model_id = str(payload.get("model") or "").strip().lower()
        if model_id and model_id != AUTOTUNE_MODEL_ID:
            continue
        if (not model_id) and (not is_flash_file):
            continue
        best = payload.get("best_config")
        if not isinstance(best, dict) or (not best):
            continue
        if not _is_finalized_and_frontier_verified(payload):
            continue
        if not _has_strict_phase2_validation(payload):
            # Ignore stale cache logs produced before strict DiT-phase validation.
            continue
        if _autotune_signature_matches(payload.get("signature", {}), expected_signature):
            payload["_log_path"] = str(path)
            return payload
    return None


def _write_autotune_log(
    log_dir: Path,
    payload: Dict[str, Any],
    *,
    existing_path: Optional[Path] = None,
) -> Optional[Path]:
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = Path(existing_path) if existing_path else None
        if not path:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = log_dir / f"{AUTOTUNE_LOG_PREFIX}_{ts}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
    except Exception:
        return None


def _create_autotune_demo_video(
    input_path: str,
    output_path: Path,
    *,
    target_frames: int = AUTOTUNE_TARGET_FRAMES,
    resize_to: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    import cv2  # type: ignore

    input_kind = detect_input_type(str(input_path or ""))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink(missing_ok=True)

    resize_w = int(resize_to[0]) if resize_to else 0
    resize_h = int(resize_to[1]) if resize_to else 0
    if resize_w <= 0 or resize_h <= 0:
        dims = get_media_dimensions(str(input_path))
        if not dims:
            raise RuntimeError("Could not determine input dimensions for autotune demo.")
        resize_w, resize_h = int(dims[0]), int(dims[1])
    resize_w = max(16, resize_w - (resize_w % 2))
    resize_h = max(16, resize_h - (resize_h % 2))

    fps = 30.0
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (resize_w, resize_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create demo video writer: {output_path}")

    written = 0
    source_frames = 0
    last_frame = None

    def _resize_if_needed(frame):
        if frame is None:
            return None
        if frame.shape[1] == resize_w and frame.shape[0] == resize_h:
            return frame
        interpolation = (
            cv2.INTER_AREA
            if (frame.shape[1] >= resize_w and frame.shape[0] >= resize_h)
            else cv2.INTER_LANCZOS4
        )
        return cv2.resize(frame, (resize_w, resize_h), interpolation=interpolation)

    try:
        if input_kind == "video":
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open input video for autotune: {input_path}")
            source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            while written < int(target_frames):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame = _resize_if_needed(frame)
                if frame is None:
                    continue
                writer.write(frame)
                last_frame = frame
                written += 1
            cap.release()

        elif input_kind == "image":
            frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read input image for autotune: {input_path}")
            frame = _resize_if_needed(frame)
            if frame is None:
                raise RuntimeError("Autotune image resize failed.")
            source_frames = 1
            last_frame = frame
            while written < int(target_frames):
                writer.write(frame)
                written += 1

        elif input_kind == "directory":
            frames = [
                p
                for p in sorted(Path(input_path).iterdir())
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if not frames:
                raise RuntimeError("No image frames found in input directory for autotune.")
            source_frames = len(frames)
            for fp in frames:
                if written >= int(target_frames):
                    break
                frame = cv2.imread(str(fp), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                frame = _resize_if_needed(frame)
                if frame is None:
                    continue
                writer.write(frame)
                last_frame = frame
                written += 1

        else:
            raise RuntimeError(f"Unsupported input type for autotune: {input_kind}")

        if last_frame is None or written <= 0:
            raise RuntimeError("No frames available for autotune demo clip.")
        while written < int(target_frames):
            writer.write(last_frame)
            written += 1
    finally:
        writer.release()

    return {
        "input_kind": input_kind,
        "fps": float(fps),
        "width": int(resize_w),
        "height": int(resize_h),
        "source_frames": int(source_frames),
        "written_frames": int(written),
        "path": str(output_path),
    }


def flashvsr_auto_tune_action(
    *,
    uploaded_file,
    args: Tuple[Any, ...],
    state: Dict[str, Any] | None,
    progress,
    global_settings_snapshot: Dict[str, Any] | None,
    global_settings_fallback: Dict[str, Any],
    defaults: Dict[str, Any],
    flashvsr_order: List[str],
    parse_args_fn: Callable[[List[Any]], Dict[str, Any]],
    guardrail_fn: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    canonical_scale_fn: Callable[..., int],
    runner,
    base_dir: Path,
    temp_dir: Path,
    cancel_event: threading.Event,
):
    global_cfg = (
        dict(global_settings_snapshot)
        if isinstance(global_settings_snapshot, dict)
        else dict(global_settings_fallback or {})
    )
    state = state or {"seed_controls": {}, "operation_status": "ready"}
    state.setdefault("seed_controls", {})
    seed_controls = state.get("seed_controls", {})
    if not isinstance(seed_controls, dict):
        seed_controls = {}
        state["seed_controls"] = seed_controls

    def _indicator(title: str, subtitle: str) -> Dict[str, Any]:
        safe_title = html.escape(str(title or ""))
        safe_sub = html.escape(str(subtitle or ""))
        block = (
            '<div class="processing-banner">'
            '<div class="processing-spinner"></div>'
            '<div class="processing-col">'
            f'<div class="processing-text">{safe_title}</div>'
            f'<div class="processing-sub">{safe_sub}</div>'
            "</div></div>"
        )
        return gr.update(value=block, visible=True)

    log_lines: List[str] = []

    def _payload(
        status_text: str,
        *,
        show_indicator: bool,
        indicator_title: str = "Auto Tune in progress...",
        tile_value: Optional[int] = None,
        overlap_value: Optional[int] = None,
        chunk_value: Optional[int] = None,
        tiled_dit_value: Optional[bool] = None,
        summary_text: Optional[str] = None,
        summary_visible: Optional[bool] = None,
    ):
        summary_update = gr.update()
        if summary_text is not None:
            summary_update = gr.update(
                value=str(summary_text),
                visible=bool(summary_visible if summary_visible is not None else bool(summary_text)),
            )
        return (
            str(status_text or ""),
            "\n".join(log_lines[-260:]),
            (_indicator(indicator_title, status_text) if show_indicator else gr.update(value="", visible=False)),
            (gr.update(value=int(tile_value)) if tile_value is not None else gr.update()),
            (gr.update(value=int(overlap_value)) if overlap_value is not None else gr.update()),
            (gr.update(value=int(chunk_value)) if chunk_value is not None else gr.update()),
            (gr.update(value=bool(tiled_dit_value)) if tiled_dit_value is not None else gr.update()),
            summary_update,
            state,
        )

    def _append_log(text: str) -> None:
        msg = str(text or "").strip()
        if not msg:
            return
        print(f"[FlashVSR AutoTune] {msg}", flush=True)
        log_lines.append(msg)

    def _cancel_requested() -> bool:
        if cancel_event.is_set():
            return True
        is_canceled = getattr(runner, "is_canceled", None)
        try:
            return bool(callable(is_canceled) and is_canceled())
        except Exception:
            return False

    def _quality_rank(cfg: Dict[str, Any]) -> int:
        chunk = int(cfg.get("frame_chunk_size") or 0)
        tile = int(cfg.get("tile_size") or 0)
        overlap_val = int(cfg.get("overlap") or 0)
        tiled_dit_val = bool(cfg.get("tiled_dit", True))
        if not tiled_dit_val:
            return 1_000_000_000 + (chunk * 10_000) - overlap_val
        return (chunk * 10_000) + (tile * 10) - overlap_val

    session_dir: Optional[Path] = None
    autotune_log_path: Optional[Path] = None
    try:
        state["operation_status"] = "running"
        clear_vram_oom_alert(state)
        cancel_event.clear()
        reset_cancel = getattr(runner, "reset_cancel_state", None)
        if callable(reset_cancel):
            reset_cancel()

        if len(args) != len(flashvsr_order):
            _append_log(f"Schema mismatch: received {len(args)} settings values but expected {len(flashvsr_order)}.")
            yield _payload("Auto Tune aborted: schema mismatch.", show_indicator=False)
            return

        settings = {**defaults, **parse_args_fn(list(args))}
        settings = guardrail_fn(settings, defaults)
        settings["batch_enable"] = False
        settings["batch_input_path"] = ""
        settings["batch_output_path"] = ""
        settings["resume_run_dir"] = ""
        settings["save_metadata"] = False
        settings["face_restore_after_upscale"] = False
        settings["output_format"] = "mp4"
        settings["start_frame"] = 0
        settings["end_frame"] = -1

        input_path = _resolve_uploaded_path(uploaded_file) or normalize_path(settings.get("input_path"))
        if not input_path or not Path(input_path).exists():
            _append_log("Input path is missing or does not exist.")
            yield _payload("Auto Tune requires a valid input file/path.", show_indicator=False)
            return

        global_gpu_device = get_global_gpu_override(seed_controls, global_cfg)
        settings["device"] = "cpu" if global_gpu_device == "cpu" else str(global_gpu_device)
        settings = guardrail_fn(settings, defaults)
        if global_gpu_device == "cpu":
            _append_log("Global GPU selector is set to CPU. Auto Tune requires CUDA GPU mode.")
            yield _payload("Auto Tune unavailable in CPU mode.", show_indicator=False)
            return

        dims = get_media_dimensions(input_path)
        if not dims:
            _append_log("Could not read input dimensions.")
            yield _payload("Failed to probe input dimensions.", show_indicator=False)
            return
        input_w, input_h = int(dims[0]), int(dims[1])

        resolved_scale = canonical_scale_fn(
            scale_value=settings.get("scale", 4),
            upscale_factor_value=settings.get("upscale_factor"),
            default=settings.get("upscale_factor", settings.get("scale", 4)),
        )
        if bool(settings.get("use_resolution_tab", True)):
            raw_shared_scale = seed_controls.get("upscale_factor_val")
            if raw_shared_scale is not None:
                try:
                    resolved_scale = canonical_scale_fn(
                        scale_value=settings.get("scale", resolved_scale),
                        upscale_factor_value=float(raw_shared_scale),
                        default=resolved_scale,
                    )
                except Exception:
                    pass
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
        if target_w <= 0 or target_h <= 0:
            _append_log("Could not determine target output dimensions for autotune.")
            yield _payload("Auto Tune failed to calculate target dimensions.", show_indicator=False)
            return

        effective_in_w = int(plan.preprocess_width if plan.pre_downscale_then_upscale else input_w)
        effective_in_h = int(plan.preprocess_height if plan.pre_downscale_then_upscale else input_h)
        if effective_in_w <= 0 or effective_in_h <= 0:
            effective_in_w, effective_in_h = input_w, input_h

        gpu_ids = _parse_cuda_device_ids(settings.get("device", ""))
        if not gpu_ids and str(global_gpu_device).isdigit():
            gpu_ids = [int(global_gpu_device)]
        gpu_snapshot = _query_gpu_memory_snapshot_gb()
        total_vram_gb = 0.0
        if gpu_ids and gpu_snapshot:
            total_vram_gb = sum(float(gpu_snapshot[g][1]) for g in gpu_ids if g in gpu_snapshot)
        if total_vram_gb <= 0:
            try:
                from shared.gpu_utils import get_gpu_info

                gpus = get_gpu_info()
            except Exception:
                gpus = []
            if gpus:
                if gpu_ids:
                    by_id = {int(g.id): g for g in gpus}
                    total_vram_gb = sum(float(by_id[g].total_memory_gb) for g in gpu_ids if g in by_id)
                else:
                    total_vram_gb = float(gpus[0].total_memory_gb)
        if total_vram_gb <= 0:
            _append_log("Could not detect total VRAM for selected GPU.")
            yield _payload("Auto Tune failed to detect GPU VRAM.", show_indicator=False)
            return

        telemetry_gpu_ids: List[int] = []
        if gpu_snapshot:
            preferred_ids = list(gpu_ids) if gpu_ids else sorted(gpu_snapshot.keys())
            telemetry_gpu_ids = [idx for idx in preferred_ids if idx in gpu_snapshot]
        if not telemetry_gpu_ids:
            _append_log(
                "Live VRAM telemetry is unavailable. Auto Tune requires nvidia-smi "
                "memory query support to enforce the 2GB free headroom target."
            )
            yield _payload("Auto Tune requires live VRAM telemetry from nvidia-smi.", show_indicator=False)
            return

        signature = _build_autotune_signature(
            settings,
            target_w=target_w,
            target_h=target_h,
            effective_in_w=effective_in_w,
            effective_in_h=effective_in_h,
            global_gpu_device=str(global_gpu_device),
            total_vram_gb=total_vram_gb,
        )
        logs_dir = Path(getattr(runner, "base_dir", Path.cwd())) / "vram_usages"
        history_sources: List[str] = []
        history_outcomes_by_key: Dict[Tuple[int, int, int, bool], Dict[str, Any]] = {}

        def _history_key_from_outcome(item: Dict[str, Any]) -> Tuple[int, int, int, bool]:
            return (
                int(item.get("frame_chunk_size") or 0),
                int(item.get("tile_size") or 0),
                int(item.get("overlap") or 0),
                bool(item.get("tiled_dit", True)),
            )

        def _history_boundary_fail(item: Dict[str, Any]) -> bool:
            try:
                free_gb = float(item.get("estimated_free_gb", 1e9) or 1e9)
            except Exception:
                free_gb = 1e9
            if bool(item.get("oom", False)):
                return True
            if str(item.get("probe_cancel_reason") or "").strip().lower() == "threshold_reached":
                return True
            if bool(item.get("telemetry_ok", False)) and free_gb < float(AUTOTUNE_MIN_FREE_VRAM_GB):
                return True
            return False

        try:
            candidate_logs = sorted(
                [p for p in logs_dir.glob("*.json") if p.is_file()],
                key=lambda p: p.stat().st_mtime,
            ) if logs_dir.exists() else []
        except Exception:
            candidate_logs = []

        for log_path in candidate_logs:
            try:
                payload = json.loads(log_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            is_flash_file = log_path.name.lower().startswith(f"{AUTOTUNE_LOG_PREFIX}_")
            model_id = str(payload.get("model") or "").strip().lower()
            if model_id and model_id != AUTOTUNE_MODEL_ID:
                continue
            if (not model_id) and (not is_flash_file):
                continue
            if not _autotune_signature_matches(payload.get("signature", {}), signature):
                continue
            history_sources.append(str(log_path))
            tests_blob = payload.get("tests")
            if not isinstance(tests_blob, list):
                continue
            for raw in tests_blob:
                if not isinstance(raw, dict):
                    continue
                item = dict(raw)
                item["frame_chunk_size"] = int(item.get("frame_chunk_size") or 0)
                item["tile_size"] = int(item.get("tile_size") or 0)
                item["overlap"] = int(item.get("overlap") or 0)
                item["tiled_dit"] = bool(item.get("tiled_dit", True))
                item["telemetry_ok"] = bool(item.get("telemetry_ok", False))
                item["oom"] = bool(item.get("oom", False))
                try:
                    item["estimated_free_gb"] = float(item.get("estimated_free_gb", 0.0) or 0.0)
                except Exception:
                    item["estimated_free_gb"] = 0.0
                returncode_ok = bool(int(item.get("returncode", 1) or 1) == 0)
                fast_probe_ok = bool(item.get("fast_probe_stop", False))
                probe_stop = str(item.get("probe_cancel_reason") or "").strip().lower()
                item["passed"] = bool(
                    (returncode_ok or fast_probe_ok)
                    and item["telemetry_ok"]
                    and (not item["oom"])
                    and float(item["estimated_free_gb"]) >= float(AUTOTUNE_MIN_FREE_VRAM_GB)
                    and probe_stop != "threshold_reached"
                )
                history_outcomes_by_key[_history_key_from_outcome(item)] = item

        history_tests = list(history_outcomes_by_key.values())
        history_best_safe: Optional[Dict[str, Any]] = None
        history_best_rank = -1
        for item in history_tests:
            if not bool(item.get("passed", False)):
                continue
            candidate_cfg = {
                "tile_size": int(item.get("tile_size") or AUTOTUNE_BASE_TILE_SIZE),
                "overlap": int(item.get("overlap") or AUTOTUNE_PRIMARY_OVERLAP),
                "frame_chunk_size": int(item.get("frame_chunk_size") or AUTOTUNE_FRAME_CHUNK_SEQUENCE[0]),
                "tiled_dit": bool(item.get("tiled_dit", True)),
                "min_free_vram_target_gb": float(AUTOTUNE_MIN_FREE_VRAM_GB),
                "measured_peak_vram_used_gb": float(item.get("max_vram_used_gb") or 0.0),
                "estimated_free_vram_gb": float(item.get("estimated_free_gb") or 0.0),
            }
            rank = _quality_rank(candidate_cfg)
            if rank > history_best_rank:
                history_best_rank = rank
                history_best_safe = candidate_cfg

        history_frontier_best: Optional[Dict[str, Any]] = None
        if history_best_safe is not None:
            if (
                int(history_best_safe.get("frame_chunk_size") or 0) >= int(AUTOTUNE_FRAME_CHUNK_SEQUENCE[0])
                and int(history_best_safe.get("tile_size") or 0) >= int(AUTOTUNE_TILE_MAX)
                and (not bool(history_best_safe.get("tiled_dit", True)))
            ):
                history_frontier_best = dict(history_best_safe)
            else:
                for item in history_tests:
                    if bool(item.get("passed", False)):
                        continue
                    if not _history_boundary_fail(item):
                        continue
                    fail_rank = _quality_rank(
                        {
                            "frame_chunk_size": int(item.get("frame_chunk_size") or 0),
                            "tile_size": int(item.get("tile_size") or 0),
                            "overlap": int(item.get("overlap") or 0),
                            "tiled_dit": bool(item.get("tiled_dit", True)),
                        }
                    )
                    if fail_rank > history_best_rank:
                        history_frontier_best = dict(history_best_safe)
                        break

        if history_frontier_best:
            flash_cfg = state.setdefault("seed_controls", {}).setdefault("flashvsr_settings", {})
            flash_cfg["tile_size"] = int(history_frontier_best.get("tile_size") or AUTOTUNE_BASE_TILE_SIZE)
            flash_cfg["overlap"] = int(history_frontier_best.get("overlap") or AUTOTUNE_PRIMARY_OVERLAP)
            flash_cfg["frame_chunk_size"] = int(
                history_frontier_best.get("frame_chunk_size") or AUTOTUNE_FRAME_CHUNK_SEQUENCE[0]
            )
            flash_cfg["tiled_dit"] = bool(history_frontier_best.get("tiled_dit", True))
            source_hint = history_sources[-1] if history_sources else "(matched logs)"
            _append_log(
                f"Matched historical autotune tests across {len(history_sources)} log(s). Using frontier-safe config."
            )
            _append_log(
                "Reusing best config -> "
                f"tile_size={flash_cfg['tile_size']}, overlap={flash_cfg['overlap']}, "
                f"frame_chunk_size={flash_cfg['frame_chunk_size']}, tiled_dit={flash_cfg['tiled_dit']}"
            )
            summary_md = (
                "**FlashVSR+ Auto Tune Result (cached from historical logs)**\n"
                f"- DiT Tiling: {'Disabled' if not flash_cfg['tiled_dit'] else 'Enabled'}\n"
                f"- Tile Size: `{flash_cfg['tile_size']}`\n"
                f"- Tile Overlap: `{flash_cfg['overlap']}`\n"
                f"- Frame Chunk Size: `{flash_cfg['frame_chunk_size']}`\n"
                f"- Matched logs: `{len(history_sources)}`\n"
                f"- Latest matched log: `{source_hint}`"
            )
            state["operation_status"] = "completed"
            yield _payload(
                "Auto Tune reused a matching cached result.",
                show_indicator=False,
                tile_value=flash_cfg["tile_size"],
                overlap_value=flash_cfg["overlap"],
                chunk_value=flash_cfg["frame_chunk_size"],
                tiled_dit_value=flash_cfg["tiled_dit"],
                summary_text=summary_md,
                summary_visible=True,
            )
            return

        live_temp_dir = Path(global_cfg.get("temp_dir", temp_dir))
        session_tag = time.strftime("%Y%m%d_%H%M%S")
        session_dir = live_temp_dir / "flashvsrplus_autotune" / session_tag
        session_dir.mkdir(parents=True, exist_ok=True)
        demo_video_path = session_dir / "flashvsrplus_autotune_demo_450f.mp4"

        _append_log(
            f"Input: {input_path} ({input_w}x{input_h}) | "
            f"effective input: {effective_in_w}x{effective_in_h} | "
            f"target: {target_w}x{target_h} | total VRAM: {total_vram_gb:.2f}GB"
        )
        _append_log("Creating 450-frame demo clip for deterministic VRAM testing...")
        yield _payload(
            "Preparing 450-frame demo clip...",
            show_indicator=True,
            indicator_title="Auto Tune setup",
            tile_value=int(settings.get("tile_size") or AUTOTUNE_BASE_TILE_SIZE),
            overlap_value=int(AUTOTUNE_PRIMARY_OVERLAP),
            chunk_value=AUTOTUNE_FRAME_CHUNK_SEQUENCE[0],
            tiled_dit_value=True,
        )

        demo_meta = _create_autotune_demo_video(
            input_path=input_path,
            output_path=demo_video_path,
            target_frames=AUTOTUNE_TARGET_FRAMES,
            resize_to=(effective_in_w, effective_in_h),
        )
        _append_log(
            f"Demo clip ready: {demo_meta['path']} ({demo_meta['width']}x{demo_meta['height']}, "
            f"{demo_meta['written_frames']} frames)"
        )

        working_settings = settings.copy()
        working_settings["input_path"] = str(demo_video_path)
        working_settings["output_format"] = "mp4"
        working_settings["load_cap"] = 0
        working_settings["start_frame"] = 0
        working_settings["end_frame"] = -1
        working_settings["batch_enable"] = False
        working_settings["batch_input_path"] = ""
        working_settings["batch_output_path"] = ""
        working_settings["resume_run_dir"] = ""
        working_settings["save_metadata"] = False
        working_settings["face_restore_after_upscale"] = False
        working_settings["output_override"] = ""
        working_settings["stream_decode"] = False
        working_settings["tile_size"] = AUTOTUNE_BASE_TILE_SIZE
        working_settings["overlap"] = AUTOTUNE_PRIMARY_OVERLAP
        working_settings["tiled_dit"] = True
        working_settings["frame_chunk_size"] = AUTOTUNE_FRAME_CHUNK_SEQUENCE[0]
        working_settings["device"] = settings.get("device", "")

        tests: List[Dict[str, Any]] = list(history_tests)
        known_outcomes_by_key: Dict[Tuple[int, int, int, bool], Dict[str, Any]] = dict(history_outcomes_by_key)
        best_config: Optional[Dict[str, Any]] = dict(history_best_safe) if isinstance(history_best_safe, dict) else None
        best_rank = _quality_rank(best_config) if isinstance(best_config, dict) else -1
        status_reason = "running"
        created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        run_counter = 0
        total_estimated_runs = 96

        if history_tests:
            _append_log(
                f"Loaded {len(history_tests)} matching historical test result(s) from {len(history_sources)} log(s). "
                "Auto Tune will continue from previous progress."
            )

        def _persist_autotune_progress(
            status_override: Optional[str] = None,
            *,
            finalized: bool = False,
            frontier_verified: bool = False,
        ) -> None:
            nonlocal autotune_log_path
            payload = {
                "model": AUTOTUNE_MODEL_ID,
                "created_at": created_at,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": str(status_override or status_reason),
                "finalized": bool(finalized),
                "frontier_verified": bool(frontier_verified),
                "signature": signature,
                "input": {
                    "path": input_path,
                    "width": int(input_w),
                    "height": int(input_h),
                    "effective_input_width": int(effective_in_w),
                    "effective_input_height": int(effective_in_h),
                },
                "target": {
                    "width": int(target_w),
                    "height": int(target_h),
                    "pixels": int(target_w * target_h),
                },
                "gpu": {
                    "selected_device": str(global_gpu_device),
                    "selected_ids": list(gpu_ids),
                    "total_vram_gb": float(total_vram_gb),
                    "min_free_target_gb": float(AUTOTUNE_MIN_FREE_VRAM_GB),
                },
                "demo_clip": dict(demo_meta) if isinstance(demo_meta, dict) else {},
                "tests": list(tests),
                "best_config": dict(best_config) if isinstance(best_config, dict) else {},
            }
            saved = _write_autotune_log(logs_dir, payload, existing_path=autotune_log_path)
            if saved and autotune_log_path is None:
                autotune_log_path = Path(saved)
                _append_log(f"Auto Tune log file created: {autotune_log_path}")

        _persist_autotune_progress(status_override="running")

        def _run_probe_once(
            test_chunk: int,
            test_tile: int,
            test_overlap: int,
            test_tiled_dit: bool,
        ) -> Tuple[FlashVSRResult, Dict[str, Any]]:
            nonlocal run_counter
            run_counter += 1
            probe_settings = working_settings.copy()
            probe_settings["frame_chunk_size"] = int(test_chunk)
            probe_settings["tile_size"] = int(test_tile)
            probe_settings["overlap"] = int(test_overlap)
            probe_settings["tiled_dit"] = bool(test_tiled_dit)
            probe_settings["stream_decode"] = False
            probe_settings["output_override"] = str(
                session_dir
                / (
                    f"probe_c{int(test_chunk)}_t{int(test_tile)}_"
                    f"{'dit1' if test_tiled_dit else 'dit0'}_{run_counter:03d}.mp4"
                )
            )
            probe_settings["_original_filename"] = (
                Path(str(settings.get("_original_filename") or "")).name or Path(input_path).name
            )

            probe_label = (
                f"Test {run_counter}: chunk={int(test_chunk)}, tile={int(test_tile)}, "
                f"overlap={int(test_overlap)}, tiled_dit={bool(test_tiled_dit)}"
            )
            _append_log(f"{probe_label} - starting")
            if progress:
                pct = min(0.99, float(run_counter) / float(max(1, total_estimated_runs)))
                progress(pct, desc=probe_label[:120])
            yield _payload(
                f"{probe_label} | target {target_w}x{target_h} | keep >= {AUTOTUNE_MIN_FREE_VRAM_GB:.1f}GB free",
                show_indicator=True,
                indicator_title="Auto Tune running",
                tile_value=int(test_tile),
                overlap_value=int(test_overlap),
                chunk_value=int(test_chunk),
                tiled_dit_value=bool(test_tiled_dit),
            )

            phase_state: Dict[str, Any] = {
                "phase": "startup",
                "tile_idx_max": 0,
                "tile_total_max": 0,
                "iter_idx_max": 0,
                "iter_total_max": 0,
                "phase2_gate_ready": False,
                "step_full_seen": False,
            }
            probe_cancel_event = threading.Event()
            sampler_stop = threading.Event()
            sampler_box: Dict[str, Any] = {}
            tile_prog_re = re.compile(r"processing tiles:\s*(\d+)\s*/\s*(\d+)", flags=re.IGNORECASE)
            iter_prog_re = re.compile(r"processing:\s*(\d+)\s*/\s*(\d+)", flags=re.IGNORECASE)

            def _probe_progress(msg: str) -> None:
                line = str(msg or "").strip()
                if not line:
                    return
                lc = line.lower()
                tile_idx = 0
                tile_total = 0
                iter_idx = 0
                iter_total = 0

                m_tile = tile_prog_re.search(line)
                if m_tile:
                    tile_idx = int(m_tile.group(1))
                    tile_total = int(m_tile.group(2))
                    phase_state["tile_idx_max"] = max(int(phase_state.get("tile_idx_max") or 0), tile_idx)
                    phase_state["tile_total_max"] = max(int(phase_state.get("tile_total_max") or 0), tile_total)

                m_iter = iter_prog_re.search(line)
                if m_iter:
                    iter_idx = int(m_iter.group(1))
                    iter_total = int(m_iter.group(2))
                    phase_state["iter_idx_max"] = max(int(phase_state.get("iter_idx_max") or 0), iter_idx)
                    phase_state["iter_total_max"] = max(int(phase_state.get("iter_total_max") or 0), iter_total)
                    if iter_total > 0 and iter_idx >= iter_total:
                        phase_state["step_full_seen"] = True

                if ("processing tiles:" in lc) or ("starting tiled processing" in lc):
                    phase_state["phase"] = "phase2"
                elif (not bool(test_tiled_dit)) and ("processing:" in lc):
                    phase_state["phase"] = "phase2"
                elif ("vae" in lc and ("encode" in lc or "encoding" in lc)) or ("phase 1" in lc):
                    phase_state["phase"] = "phase1"
                elif ("vae" in lc and ("decode" in lc or "decoding" in lc)) or ("phase 3" in lc):
                    phase_state["phase"] = "phase3"

                # Require at least one tile + a few DiT iterations before permitting early stop.
                if bool(test_tiled_dit):
                    tile_seen = int(phase_state.get("tile_idx_max") or 0) >= 1
                    iter_seen = int(phase_state.get("iter_idx_max") or 0) >= int(AUTOTUNE_PHASE2_MIN_ITER_FOR_GATE)
                    if tile_seen and iter_seen:
                        phase_state["phase2_gate_ready"] = True
                else:
                    if int(phase_state.get("iter_idx_max") or 0) >= int(AUTOTUNE_PHASE2_MIN_ITER_FOR_GATE):
                        phase_state["phase2_gate_ready"] = True

                if _cancel_requested():
                    probe_cancel_event.set()

            sampler_thread = threading.Thread(
                target=_sample_peak_vram_gb,
                args=(
                    sampler_stop,
                    sampler_box,
                    telemetry_gpu_ids,
                    0.20,
                    phase_state,
                    probe_cancel_event,
                    AUTOTUNE_MIN_FREE_VRAM_GB,
                    float(total_vram_gb),
                ),
                daemon=True,
            )
            sampler_thread.start()
            prev_disable_oom = os.environ.get("FLASHVSR_DISABLE_OOM_RECOVERY")
            os.environ["FLASHVSR_DISABLE_OOM_RECOVERY"] = "1"
            try:
                result = run_flashvsr(
                    probe_settings,
                    base_dir,
                    on_progress=_probe_progress,
                    cancel_event=probe_cancel_event,
                )
            finally:
                if prev_disable_oom is None:
                    os.environ.pop("FLASHVSR_DISABLE_OOM_RECOVERY", None)
                else:
                    os.environ["FLASHVSR_DISABLE_OOM_RECOVERY"] = prev_disable_oom
                sampler_stop.set()
                sampler_thread.join(timeout=2.0)

            with suppress(Exception):
                probe_path = Path(str(probe_settings.get("output_override") or ""))
                if probe_path and probe_path.exists():
                    probe_path.unlink(missing_ok=True)

            max_used_gb = float(sampler_box.get("max_used_gb", 0.0) or 0.0)
            max_used_phase2_gb = float(sampler_box.get("max_used_phase2_gb", 0.0) or 0.0)
            measured_total = float(sampler_box.get("total_gb", 0.0) or 0.0)
            telemetry_ok = bool(sampler_box.get("telemetry_ok", False))
            phase2_samples = int(sampler_box.get("phase2_samples", 0) or 0)
            early_stop_reason = str(sampler_box.get("early_stop_reason") or "")
            phase2_gate_ready = bool(phase_state.get("phase2_gate_ready", False))
            total_for_eval = measured_total if measured_total > 0 else float(total_vram_gb)
            peak_source = "phase2" if (phase2_samples > 0 and max_used_phase2_gb > 0) else "whole_run"
            selected_peak_gb = max_used_phase2_gb if peak_source == "phase2" else max_used_gb
            free_gb = max(0.0, total_for_eval - selected_peak_gb) if total_for_eval > 0 else 0.0

            canceled_by_user = _cancel_requested()
            oom = bool(_looks_like_oom(result.log))
            oom_phase = _detect_flashvsr_oom_phase(result.log) if oom else ""
            fast_probe_stop = bool(early_stop_reason in {"threshold_reached", "enough_phase2_samples"})
            return_ok = bool(result.returncode == 0) or (fast_probe_stop and (not canceled_by_user) and (not oom))
            passed = bool(
                telemetry_ok
                and return_ok
                and (not oom)
                and (not canceled_by_user)
                and float(free_gb) >= float(AUTOTUNE_MIN_FREE_VRAM_GB)
                and early_stop_reason != "threshold_reached"
            )

            outcome = {
                "frame_chunk_size": int(test_chunk),
                "tile_size": int(test_tile),
                "overlap": int(test_overlap),
                "tiled_dit": bool(test_tiled_dit),
                "returncode": int(result.returncode),
                "oom": bool(oom),
                "oom_phase": str(oom_phase),
                "max_vram_used_gb": round(selected_peak_gb, 3),
                "peak_source": str(peak_source),
                "phase2_peak_vram_used_gb": round(max_used_phase2_gb, 3),
                "whole_run_peak_vram_used_gb": round(max_used_gb, 3),
                "total_vram_gb": round(total_for_eval, 3),
                "estimated_free_gb": round(free_gb, 3),
                "telemetry_ok": bool(telemetry_ok),
                "phase2_samples": int(phase2_samples),
                "phase2_gate_ready": bool(phase2_gate_ready),
                "tile_idx_max": int(phase_state.get("tile_idx_max") or 0),
                "tile_total_max": int(phase_state.get("tile_total_max") or 0),
                "iter_idx_max": int(phase_state.get("iter_idx_max") or 0),
                "iter_total_max": int(phase_state.get("iter_total_max") or 0),
                "probe_cancel_reason": str(early_stop_reason),
                "fast_probe_stop": bool(fast_probe_stop),
                "passed": bool(passed),
            }
            return result, outcome

        def _consider_best(outcome: Dict[str, Any]) -> None:
            nonlocal best_config, best_rank
            if not bool(outcome.get("passed", False)):
                return
            candidate = {
                "tile_size": int(outcome.get("tile_size") or AUTOTUNE_BASE_TILE_SIZE),
                "overlap": int(outcome.get("overlap") or AUTOTUNE_PRIMARY_OVERLAP),
                "frame_chunk_size": int(outcome.get("frame_chunk_size") or AUTOTUNE_FRAME_CHUNK_SEQUENCE[0]),
                "tiled_dit": bool(outcome.get("tiled_dit", True)),
                "min_free_vram_target_gb": float(AUTOTUNE_MIN_FREE_VRAM_GB),
                "measured_peak_vram_used_gb": float(outcome.get("max_vram_used_gb") or 0.0),
                "estimated_free_vram_gb": float(outcome.get("estimated_free_gb") or 0.0),
            }
            rank = _quality_rank(candidate)
            if rank > best_rank:
                best_rank = rank
                best_config = candidate

        def _run_candidate_with_vae_retry(
            chunk_candidate: int,
            tile_candidate: int,
            tiled_dit_candidate: bool,
            overlap_candidate: int,
        ) -> Tuple[Optional[Dict[str, Any]], bool]:
            attempted: set[int] = set()
            cur_tile = int(tile_candidate)
            run_overlap = int(overlap_candidate)
            if tiled_dit_candidate:
                cur_tile = max(AUTOTUNE_FALLBACK_TILES[-1], cur_tile)
            else:
                cur_tile = max(32, cur_tile)

            while True:
                if _cancel_requested():
                    return None, False
                if cur_tile in attempted:
                    return None, False
                attempted.add(cur_tile)
                outcome_key = (
                    int(chunk_candidate),
                    int(cur_tile),
                    int(run_overlap),
                    bool(tiled_dit_candidate),
                )
                known_outcome = known_outcomes_by_key.get(outcome_key)
                if isinstance(known_outcome, dict):
                    outcome = dict(known_outcome)
                    _append_log(
                        f"History hit chunk={outcome['frame_chunk_size']} tile={outcome['tile_size']} "
                        f"tiled_dit={outcome['tiled_dit']} -> reusing prior VRAM result."
                    )
                else:
                    _result, outcome = yield from _run_probe_once(
                        int(chunk_candidate),
                        int(cur_tile),
                        int(run_overlap),
                        bool(tiled_dit_candidate),
                    )
                    tests.append(outcome)
                    known_outcomes_by_key[outcome_key] = dict(outcome)
                    _persist_autotune_progress()
                    _append_log(
                        f"Result chunk={outcome['frame_chunk_size']} tile={outcome['tile_size']} "
                        f"tiled_dit={outcome['tiled_dit']} -> rc={outcome['returncode']}, "
                        f"peak={outcome['max_vram_used_gb']:.2f}GB ({outcome.get('peak_source','whole_run')}), "
                        f"free={outcome['estimated_free_gb']:.2f}GB, "
                        f"phase2_samples={outcome.get('phase2_samples', 0)}, "
                        f"phase2_gate_ready={bool(outcome.get('phase2_gate_ready', False))}, "
                        f"tile_prog={outcome.get('tile_idx_max', 0)}/{outcome.get('tile_total_max', 0)}, "
                        f"iter_prog={outcome.get('iter_idx_max', 0)}/{outcome.get('iter_total_max', 0)}, "
                        f"probe_stop={outcome.get('probe_cancel_reason') or 'none'}, "
                        f"passed={bool(outcome.get('passed', False))}"
                    )

                if bool(outcome.get("passed", False)):
                    _consider_best(outcome)
                    _persist_autotune_progress()
                    return outcome, True

                if bool(outcome.get("oom")) and str(outcome.get("oom_phase")) in {"phase1_encode", "phase3_decode"}:
                    if not tiled_dit_candidate:
                        return outcome, False
                    reduced_tile = max(AUTOTUNE_FALLBACK_TILES[-1], int(cur_tile // 2))
                    if reduced_tile < cur_tile and reduced_tile not in attempted:
                        phase_label = "VAE encode" if outcome.get("oom_phase") == "phase1_encode" else "VAE decode"
                        _append_log(f"{phase_label} OOM at tile={cur_tile}. Retrying with tile={reduced_tile}.")
                        cur_tile = reduced_tile
                        continue

                return outcome, False

        primary_chunk = int(AUTOTUNE_FRAME_CHUNK_SEQUENCE[0])
        passed_primary = False
        current_tile = int(AUTOTUNE_BASE_TILE_SIZE)
        active_overlap = int(AUTOTUNE_PRIMARY_OVERLAP)

        while current_tile <= int(AUTOTUNE_TILE_MAX):
            if _cancel_requested():
                status_reason = "cancelled"
                break

            _outcome, passed = yield from _run_candidate_with_vae_retry(
                primary_chunk,
                current_tile,
                True,
                active_overlap,
            )
            if _cancel_requested():
                status_reason = "cancelled"
                break
            if passed:
                passed_primary = True
                current_tile += int(AUTOTUNE_TILE_STEP)
                continue

            if (not passed_primary) and int(current_tile) == int(AUTOTUNE_BASE_TILE_SIZE):
                # Requested behavior: start overlap=48, then fallback to overlap=24
                # only when tile=256 fails.
                if int(active_overlap) != int(AUTOTUNE_FALLBACK_OVERLAP):
                    _append_log(
                        f"tile=256 with overlap={active_overlap} failed; retrying with overlap={AUTOTUNE_FALLBACK_OVERLAP}."
                    )
                    _outcome_ov, passed_ov = yield from _run_candidate_with_vae_retry(
                        primary_chunk,
                        int(AUTOTUNE_BASE_TILE_SIZE),
                        True,
                        int(AUTOTUNE_FALLBACK_OVERLAP),
                    )
                    if _cancel_requested():
                        status_reason = "cancelled"
                        break
                    if passed_ov:
                        active_overlap = int(AUTOTUNE_FALLBACK_OVERLAP)
                        passed_primary = True
                        current_tile += int(AUTOTUNE_TILE_STEP)
                        continue

                fallback_pass = False
                for fallback_tile in AUTOTUNE_FALLBACK_TILES:
                    _outcome_fb, passed_fb = yield from _run_candidate_with_vae_retry(
                        primary_chunk,
                        int(fallback_tile),
                        True,
                        int(AUTOTUNE_FALLBACK_OVERLAP),
                    )
                    if _cancel_requested():
                        status_reason = "cancelled"
                        break
                    if passed_fb:
                        fallback_pass = True
                        passed_primary = True
                        active_overlap = int(AUTOTUNE_FALLBACK_OVERLAP)
                        break
                if status_reason == "cancelled":
                    break
                if fallback_pass:
                    status_reason = "threshold_reached"
                else:
                    status_reason = "need_lower_chunk"
                break

            status_reason = "threshold_reached" if passed_primary else "need_lower_chunk"
            break

        if status_reason == "running":
            status_reason = "completed" if passed_primary else "need_lower_chunk"

        if (
            status_reason in {"completed", "threshold_reached"}
            and best_config
            and int(best_config.get("frame_chunk_size") or 0) == primary_chunk
            and int(best_config.get("tile_size") or 0) >= int(AUTOTUNE_TILE_MAX)
        ):
            _append_log("Tile 1024 passed. Running one no-tiling quality probe (tiled_dit=False)...")
            _outcome_nt, _passed_nt = yield from _run_candidate_with_vae_retry(
                primary_chunk,
                int(AUTOTUNE_TILE_MAX),
                False,
                active_overlap,
            )

        if status_reason == "need_lower_chunk":
            if int(active_overlap) != int(AUTOTUNE_FALLBACK_OVERLAP):
                active_overlap = int(AUTOTUNE_FALLBACK_OVERLAP)
            for chunk_candidate in AUTOTUNE_FRAME_CHUNK_SEQUENCE[1:]:
                if _cancel_requested():
                    status_reason = "cancelled"
                    break
                _append_log(
                    f"No safe config yet. Reducing frame_chunk_size to {int(chunk_candidate)} for lower-VRAM fallback."
                )
                safe_found = False
                for tile_candidate in (AUTOTUNE_BASE_TILE_SIZE,) + AUTOTUNE_FALLBACK_TILES:
                    _outcome_fallback, _passed_fallback = yield from _run_candidate_with_vae_retry(
                        int(chunk_candidate),
                        int(tile_candidate),
                        True,
                        active_overlap,
                    )
                    if _cancel_requested():
                        status_reason = "cancelled"
                        break
                    if _passed_fallback:
                        safe_found = True
                        break
                if status_reason == "cancelled":
                    break
                if safe_found:
                    status_reason = "threshold_reached"
                    break
            if status_reason == "need_lower_chunk":
                status_reason = "failed"

        def _is_vram_boundary_failure(test_item: Dict[str, Any]) -> bool:
            try:
                free_gb = float(test_item.get("estimated_free_gb", 1e9) or 1e9)
            except Exception:
                free_gb = 1e9
            if bool(test_item.get("oom", False)):
                return True
            if str(test_item.get("probe_cancel_reason") or "").strip().lower() == "threshold_reached":
                return True
            if free_gb < float(AUTOTUNE_MIN_FREE_VRAM_GB):
                return True
            return False

        def _frontier_verified() -> bool:
            if not isinstance(best_config, dict) or (not best_config):
                return False
            best_rank_local = _quality_rank(best_config)
            for item in tests:
                if not isinstance(item, dict):
                    continue
                if bool(item.get("passed", False)):
                    continue
                if not bool(item.get("telemetry_ok", False)):
                    continue
                candidate_rank = _quality_rank(
                    {
                        "frame_chunk_size": int(item.get("frame_chunk_size") or 0),
                        "tile_size": int(item.get("tile_size") or 0),
                        "overlap": int(item.get("overlap") or 0),
                        "tiled_dit": bool(item.get("tiled_dit", True)),
                    }
                )
                if candidate_rank <= best_rank_local:
                    continue
                if _is_vram_boundary_failure(item):
                    return True
            # If no higher-quality candidate exists in this sweep space, frontier is inherently verified.
            if (
                int(best_config.get("frame_chunk_size") or 0) >= int(AUTOTUNE_FRAME_CHUNK_SEQUENCE[0])
                and int(best_config.get("tile_size") or 0) >= int(AUTOTUNE_TILE_MAX)
                and (not bool(best_config.get("tiled_dit", True)))
            ):
                return True
            return False

        frontier_ok = bool(_frontier_verified())
        if best_config and (not frontier_ok):
            _append_log(
                "Boundary validation incomplete: no higher-quality VRAM-boundary failure was observed. "
                "This run will not be reused from cache."
            )

        _persist_autotune_progress(
            status_override=status_reason,
            finalized=True,
            frontier_verified=bool(
                frontier_ok and status_reason in {"completed", "threshold_reached"}
            ),
        )
        if autotune_log_path:
            _append_log(f"Saved autotune log: {autotune_log_path}")

        if progress:
            progress(1.0, desc="Auto Tune complete")

        if best_config:
            flash_cfg = state.setdefault("seed_controls", {}).setdefault("flashvsr_settings", {})
            flash_cfg["tile_size"] = int(best_config.get("tile_size") or AUTOTUNE_BASE_TILE_SIZE)
            flash_cfg["overlap"] = int(best_config.get("overlap") or AUTOTUNE_PRIMARY_OVERLAP)
            flash_cfg["frame_chunk_size"] = int(best_config.get("frame_chunk_size") or AUTOTUNE_FRAME_CHUNK_SEQUENCE[0])
            flash_cfg["tiled_dit"] = bool(best_config.get("tiled_dit", True))

            summary_md = (
                "**FlashVSR+ Auto Tune Result**\n"
                f"- DiT Tiling: {'Disabled (highest quality)' if not flash_cfg['tiled_dit'] else 'Enabled'}\n"
                f"- Tile Size: `{flash_cfg['tile_size']}`\n"
                f"- Tile Overlap: `{flash_cfg['overlap']}`\n"
                f"- Frame Chunk Size: `{flash_cfg['frame_chunk_size']}`\n"
                f"- Peak VRAM: `{float(best_config.get('measured_peak_vram_used_gb') or 0.0):.2f} GB`\n"
                f"- Free VRAM estimate: `{float(best_config.get('estimated_free_vram_gb') or 0.0):.2f} GB` "
                f"(target >= `{AUTOTUNE_MIN_FREE_VRAM_GB:.1f} GB`)\n"
                f"- Log file: `{str(autotune_log_path) if autotune_log_path else 'not saved'}`"
            )

            if status_reason == "cancelled":
                final_status = "Auto Tune cancelled. Applied best-so-far config."
                state["operation_status"] = "ready"
            elif status_reason == "threshold_reached":
                final_status = "Auto Tune reached VRAM threshold. Applied best-so-far config."
                state["operation_status"] = "completed"
            elif status_reason == "failed":
                final_status = "Auto Tune hit an error. Applied best-so-far config."
                state["operation_status"] = "error"
            else:
                final_status = "Auto Tune complete. Applied recommended config."
                state["operation_status"] = "completed"

            _append_log(
                "Recommended -> "
                f"tile_size={flash_cfg['tile_size']}, overlap={flash_cfg['overlap']}, "
                f"frame_chunk_size={flash_cfg['frame_chunk_size']}, tiled_dit={flash_cfg['tiled_dit']}"
            )
            yield _payload(
                final_status,
                show_indicator=False,
                tile_value=flash_cfg["tile_size"],
                overlap_value=flash_cfg["overlap"],
                chunk_value=flash_cfg["frame_chunk_size"],
                tiled_dit_value=flash_cfg["tiled_dit"],
                summary_text=summary_md,
                summary_visible=True,
            )
            return

        state["operation_status"] = "error" if status_reason == "failed" else "ready"
        if status_reason == "cancelled":
            final_msg = "Auto Tune cancelled before a stable config was found."
        elif status_reason == "threshold_reached":
            final_msg = "Auto Tune stopped at VRAM threshold before finding a safe config."
        else:
            final_msg = "Auto Tune failed before finding a safe config."
        yield _payload(final_msg, show_indicator=False)
        return

    except Exception as e:
        state["operation_status"] = "error"
        _append_log(f"Auto Tune error: {e}")
        yield _payload("Auto Tune failed due to an internal error.", show_indicator=False)
        return

    finally:
        if session_dir is not None:
            with suppress(Exception):
                shutil.rmtree(session_dir, ignore_errors=True)
