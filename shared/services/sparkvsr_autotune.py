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
import re
import shutil
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr

from shared.gpu_utils import get_global_gpu_override
from shared.oom_alert import clear_vram_oom_alert
from shared.path_utils import get_media_dimensions, normalize_path
from shared.resolution_calculator import estimate_fixed_scale_upscale_plan_from_dims
from shared.services.flashvsr_autotune import (
    _clamp_save_vram_target_gb,
    _create_autotune_demo_video,
    _looks_like_oom,
    _parse_cuda_device_ids,
    _query_gpu_memory_snapshot_gb,
    _sample_peak_vram_gb,
)
from shared.sparkvsr_runner import run_sparkvsr


AUTOTUNE_MODEL_ID = "sparkvsr"
AUTOTUNE_LOG_PREFIX = "sparkvsr_autotune"
AUTOTUNE_STRATEGY_VERSION = 2
AUTOTUNE_TARGET_FRAMES = 65
AUTOTUNE_MIN_FREE_VRAM_GB = 2.0
AUTOTUNE_TEMPORAL_CANDIDATES = (0, 49, 33, 17)
AUTOTUNE_SPATIAL_CANDIDATES = (0, 1024, 768, 512, 384, 256)
AUTOTUNE_TEMPORAL_OVERLAP = 8
AUTOTUNE_SPATIAL_OVERLAP = 32
AUTOTUNE_MAX_PIXEL_DIFF_FOR_REUSE = 0.05


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


def _build_autotune_signature(
    settings: Dict[str, Any],
    *,
    target_w: int,
    target_h: int,
    effective_in_w: int,
    effective_in_h: int,
    global_gpu_device: str,
    total_vram_gb: float,
    min_free_target_gb: float,
) -> Dict[str, Any]:
    ref_mode = str(settings.get("ref_mode") or "sr_image").strip().lower()
    if bool(settings.get("auto_reference_prepass", False)) and ref_mode != "no_ref":
        ref_mode = "sr_image"
    if ref_mode in {"pisasr", "gt"}:
        ref_mode = "sr_image"
    exact_payload = {
        "autotune_model": AUTOTUNE_MODEL_ID,
        "autotune_strategy_version": int(AUTOTUNE_STRATEGY_VERSION),
        "autotune_target_frames": int(AUTOTUNE_TARGET_FRAMES),
        "model_name": str(settings.get("model_name") or ""),
        "precision": str(settings.get("precision") or "bfloat16"),
        "scale": str(settings.get("scale") or "4"),
        "upscale_mode": str(settings.get("upscale_mode") or "bilinear"),
        "noise_step": int(settings.get("noise_step") or 0),
        "sr_noise_step": int(settings.get("sr_noise_step") or 399),
        "cpu_offload": bool(settings.get("cpu_offload", True)),
        "vae_tiling": bool(settings.get("vae_tiling", True)),
        "ref_mode": ref_mode,
        "ref_guidance_scale": round(float(settings.get("ref_guidance_scale") or 1.0), 4),
        "save_format": str(settings.get("save_format") or "yuv444p"),
        "global_gpu_device": str(global_gpu_device or ""),
        "save_vram_gb": _clamp_save_vram_target_gb(min_free_target_gb, AUTOTUNE_MIN_FREE_VRAM_GB),
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
        return False
    try:
        cand_pixels = float(candidate.get("target_pixels") or 0)
        exp_pixels = float(expected.get("target_pixels") or 0)
        if cand_pixels <= 0 or exp_pixels <= 0:
            return False
        if abs(cand_pixels - exp_pixels) / max(cand_pixels, exp_pixels) > AUTOTUNE_MAX_PIXEL_DIFF_FOR_REUSE:
            return False
        cand_vram = float(candidate.get("gpu_total_vram_gb") or 0)
        exp_vram = float(expected.get("gpu_total_vram_gb") or 0)
        if cand_vram > 0 and exp_vram > 0 and abs(cand_vram - exp_vram) / max(cand_vram, exp_vram) > 0.10:
            return False
    except Exception:
        return False
    return True


def _find_cached_autotune_log(log_dir: Path, expected_signature: Dict[str, Any], min_free_vram_target_gb: float) -> Optional[Dict[str, Any]]:
    if not log_dir.exists():
        return None
    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for path in sorted(log_dir.glob(f"{AUTOTUNE_LOG_PREFIX}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        signature = payload.get("signature")
        best = payload.get("best_config")
        if not isinstance(signature, dict) or not isinstance(best, dict):
            continue
        if not _signature_matches(signature, expected_signature):
            continue
        if not bool(payload.get("frontier_verified", False)):
            continue
        try:
            free_gb = float(best.get("estimated_free_vram_gb") or 0.0)
        except Exception:
            free_gb = 0.0
        if free_gb < float(min_free_vram_target_gb):
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


def _candidate_settings() -> List[Dict[str, Any]]:
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

    # Measured SparkVSR behavior: spatial tiling gives the best VRAM reduction
    # for the smallest output drift. Keep full temporal context while shrinking
    # spatial tiles first, then combine temporal chunking with spatial tiling.
    for tile in AUTOTUNE_SPATIAL_CANDIDATES:
        add_candidate(0, int(tile))

    spatial_fallbacks = [int(t) for t in AUTOTUNE_SPATIAL_CANDIDATES if int(t) > 0]
    for chunk_len in AUTOTUNE_TEMPORAL_CANDIDATES:
        if int(chunk_len) <= 0:
            continue
        for tile in spatial_fallbacks:
            add_candidate(int(chunk_len), int(tile))
        add_candidate(int(chunk_len), 0)
    return out


def _quality_rank(cfg: Dict[str, Any]) -> int:
    chunk_len = int(cfg.get("chunk_len") or 0)
    tile = int(cfg.get("tile_height") or cfg.get("tile_width") or 0)
    temporal = 1_000_000 if chunk_len <= 0 else int(chunk_len) * 10_000
    spatial = 100_000 if tile <= 0 else int(tile) * 50
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
        print(f"[SparkVSR AutoTune] {msg}", flush=True)
        log_lines.append(msg)

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

        global_gpu_device = get_global_gpu_override(seed_controls, global_cfg)
        settings["device"] = "cpu" if global_gpu_device == "cpu" else str(global_gpu_device)
        settings = guardrail_fn(settings, defaults)
        min_free_vram_target_gb = _clamp_save_vram_target_gb(
            settings.get("save_vram_gb", AUTOTUNE_MIN_FREE_VRAM_GB),
            AUTOTUNE_MIN_FREE_VRAM_GB,
        )
        settings["save_vram_gb"] = float(min_free_vram_target_gb)
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
                from shared.gpu_utils import get_gpu_info

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

        signature = _build_autotune_signature(
            settings,
            target_w=target_w,
            target_h=target_h,
            effective_in_w=effective_in_w,
            effective_in_h=effective_in_h,
            global_gpu_device=str(global_gpu_device),
            total_vram_gb=total_vram_gb,
            min_free_target_gb=min_free_vram_target_gb,
        )
        logs_dir = Path(base_dir) / "vram_usages"
        cached = _find_cached_autotune_log(logs_dir, signature, min_free_vram_target_gb)
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
            state["operation_status"] = "completed"
            summary_md = (
                "**SparkVSR Auto Tune Result (cached)**\n"
                f"- Spatial Tile: `{spark_cfg['tile_height']}x{spark_cfg['tile_width']}` "
                f"({'disabled/full-frame' if int(spark_cfg['tile_height']) <= 0 else 'enabled'})\n"
                f"- Spatial Overlap: `{spark_cfg['overlap_height']}x{spark_cfg['overlap_width']}`\n"
                f"- Temporal Chunk Length: `{spark_cfg['chunk_len']}` "
                f"({'disabled/full sequence' if int(spark_cfg['chunk_len']) <= 0 else 'enabled'})\n"
                f"- Temporal Overlap: `{spark_cfg['overlap_t']}`\n"
                f"- Peak VRAM: `{float(best.get('measured_peak_vram_used_gb') or 0.0):.2f} GB`\n"
                f"- Free VRAM estimate: `{float(best.get('estimated_free_vram_gb') or 0.0):.2f} GB` "
                f"(target >= `{min_free_vram_target_gb:.1f} GB`)\n"
                f"- Log file: `{cached.get('_path') or 'cached log'}`"
            )
            yield _payload(
                "Auto Tune reused a matching cached result.",
                show_indicator=False,
                tile_value=int(spark_cfg["tile_height"]),
                overlap_hw_value=int(spark_cfg["overlap_height"]),
                chunk_value=int(spark_cfg["chunk_len"]),
                overlap_t_value=int(spark_cfg["overlap_t"]),
                vae_tiling_value=True,
                summary_text=summary_md,
            )
            return

        stamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = Path(temp_dir) / "sparkvsr_autotune" / stamp
        session_dir.mkdir(parents=True, exist_ok=True)
        demo_video_path = session_dir / "sparkvsr_autotune_demo_65f.mp4"
        demo_ref_path = session_dir / "sparkvsr_autotune_reference.png"
        demo_meta = _create_autotune_demo_video(
            input_path,
            demo_video_path,
            target_frames=AUTOTUNE_TARGET_FRAMES,
            resize_to=(effective_in_w, effective_in_h),
        )
        _extract_demo_reference(demo_video_path, demo_ref_path)
        _append_log(
            f"Demo clip: {demo_meta['written_frames']} frames, {demo_meta['width']}x{demo_meta['height']} -> "
            f"target {target_w}x{target_h}; keeping >= {min_free_vram_target_gb:.1f}GB free."
        )
        yield _payload(
            "Auto Tune setup complete. Starting SparkVSR probes...",
            show_indicator=True,
            tile_value=0,
            overlap_hw_value=AUTOTUNE_SPATIAL_OVERLAP,
            chunk_value=0,
            overlap_t_value=AUTOTUNE_TEMPORAL_OVERLAP,
            vae_tiling_value=True,
        )

        started_at = time.time()
        autotune_payload = {
            "status": "running",
            "signature": signature,
            "tests": tests,
            "best_config": None,
            "frontier_verified": False,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        def _persist(status_override: Optional[str] = None, finalized: bool = False, frontier_verified: bool = False) -> None:
            nonlocal autotune_log_path
            autotune_payload.update(
                {
                    "status": status_override or status_reason,
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
        candidates = _candidate_settings()
        top_rank = max(_quality_rank(c) for c in candidates)

        def _run_probe_once(candidate: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal run_counter
            run_counter += 1
            tile = int(candidate["tile_height"])
            chunk_len = int(candidate["chunk_len"])
            probe_settings = settings.copy()
            probe_settings.update(candidate)
            probe_settings["input_path"] = str(demo_video_path)
            probe_settings["_effective_input_path"] = str(demo_video_path)
            probe_settings["_original_filename"] = Path(input_path).name
            probe_settings["_run_dir"] = str(session_dir)
            probe_settings["global_output_dir"] = str(session_dir)
            probe_settings["output_override"] = str(session_dir / f"probe_{run_counter:03d}_c{chunk_len}_t{tile}.mp4")
            probe_settings["auto_reference_prepass"] = False
            probe_settings["save_metadata"] = False
            probe_settings["fps"] = float(demo_meta.get("fps") or 30.0)
            probe_settings["vae_tiling"] = True
            probe_ref_mode = _normalize_probe_ref_mode(settings)
            probe_settings["ref_mode"] = probe_ref_mode
            if probe_ref_mode == "sr_image" and demo_ref_path.exists():
                probe_settings["ref_source_path"] = str(demo_ref_path)
                probe_settings["ref_indices"] = "0"
            elif probe_ref_mode == "no_ref":
                probe_settings["ref_source_path"] = ""

            label = f"Test {run_counter}: chunk_len={chunk_len}, tile={tile if tile > 0 else 'full-frame'}"
            _append_log(f"{label} - starting")
            if progress:
                progress(min(0.99, run_counter / max(1, len(candidates))), desc=label)
            yield_status = f"{label} | target {target_w}x{target_h} | keep >= {min_free_vram_target_gb:.1f}GB free"

            phase_state: Dict[str, Any] = {"phase": "startup", "chunks": 0, "tiles": 0}
            probe_cancel_event = threading.Event()
            sampler_stop = threading.Event()
            sampler_box: Dict[str, Any] = {}
            process_re = re.compile(r"Processing:\s*F=(\d+)\s+H=(\d+)\s+W=(\d+)\s*\|\s*Chunks=(\d+)\s+Tiles=(\d+)", re.IGNORECASE)

            def _probe_progress(msg: str) -> None:
                line = str(msg or "").strip()
                if not line:
                    return
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
                    min_free_vram_target_gb,
                    float(total_vram_gb),
                ),
                daemon=True,
            )
            sampler_thread.start()
            try:
                result = run_sparkvsr(
                    probe_settings,
                    base_dir,
                    on_progress=_probe_progress,
                    cancel_event=probe_cancel_event,
                    process_handle=None,
                )
            finally:
                sampler_stop.set()
                sampler_thread.join(timeout=2.0)

            with suppress(Exception):
                outp = Path(str(probe_settings.get("output_override") or ""))
                if outp.exists():
                    outp.unlink(missing_ok=True)

            max_used_gb = float(sampler_box.get("max_used_gb", 0.0) or 0.0)
            measured_total = float(sampler_box.get("total_gb", 0.0) or total_vram_gb or 0.0)
            telemetry_ok = bool(sampler_box.get("telemetry_ok", False))
            early_stop_reason = str(sampler_box.get("early_stop_reason") or "")
            total_for_eval = measured_total if measured_total > 0 else float(total_vram_gb)
            free_gb = max(0.0, total_for_eval - max_used_gb) if total_for_eval > 0 else 0.0
            oom = bool(_looks_like_oom(result.log))
            oom_phase = _detect_sparkvsr_oom_phase(result.log) if oom else ""
            canceled_by_user = bool(cancel_event.is_set())
            fast_probe_stop = bool(early_stop_reason == "threshold_reached")
            passed = bool(
                int(result.returncode) == 0
                and telemetry_ok
                and (not oom)
                and (not canceled_by_user)
                and (not fast_probe_stop)
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
                "telemetry_ok": bool(telemetry_ok),
                "samples": int(sampler_box.get("samples", 0) or 0),
                "phase2_samples": int(sampler_box.get("phase2_samples", 0) or 0),
                "probe_cancel_reason": early_stop_reason,
                "passed": bool(passed),
                "quality_rank": int(_quality_rank(candidate)),
                "frames": int(phase_state.get("frames") or 0),
                "chunks": int(phase_state.get("chunks") or 0),
                "tiles": int(phase_state.get("tiles") or 0),
            }

        boundary_failed = False
        for candidate in candidates:
            if cancel_event.is_set():
                status_reason = "cancelled"
                _append_log("Auto Tune cancelled before next test.")
                break
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
            outcome = _run_probe_once(candidate)
            tests.append(outcome)
            _append_log(
                f"Result chunk_len={outcome['chunk_len']} tile={outcome['tile_height'] or 'full-frame'} -> "
                f"rc={outcome['returncode']}, peak={outcome['max_vram_used_gb']:.2f}GB, "
                f"free={outcome['estimated_free_gb']:.2f}GB, telemetry_ok={outcome['telemetry_ok']}, "
                f"oom={outcome['oom']}, stop={outcome.get('probe_cancel_reason') or 'none'}, "
                f"chunks={outcome.get('chunks')}, tiles={outcome.get('tiles')}, passed={outcome['passed']}"
            )
            _persist("running")
            if bool(outcome.get("passed", False)):
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
                    "quality_rank": int(outcome["quality_rank"]),
                }
                status_reason = "completed" if int(outcome["quality_rank"]) >= int(top_rank) else "threshold_reached"
                break
            boundary_failed = bool(boundary_failed or outcome.get("oom") or outcome.get("probe_cancel_reason") == "threshold_reached")
            if not bool(outcome.get("telemetry_ok", False)):
                status_reason = "failed"
                break

        frontier_ok = bool(best_config and (best_config.get("quality_rank") == top_rank or boundary_failed))
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
                f"- Spatial Tile: `{spark_cfg['tile_height']}x{spark_cfg['tile_width']}` "
                f"({'disabled/full-frame' if int(spark_cfg['tile_height']) <= 0 else 'enabled'})\n"
                f"- Spatial Overlap: `{spark_cfg['overlap_height']}x{spark_cfg['overlap_width']}`\n"
                f"- Temporal Chunk Length: `{spark_cfg['chunk_len']}` "
                f"({'disabled/full sequence' if int(spark_cfg['chunk_len']) <= 0 else 'enabled'})\n"
                f"- Temporal Overlap: `{spark_cfg['overlap_t']}`\n"
                f"- VAE Tiling: `ON`\n"
                f"- Peak VRAM: `{float(best_config.get('measured_peak_vram_used_gb') or 0.0):.2f} GB`\n"
                f"- Free VRAM estimate: `{float(best_config.get('estimated_free_vram_gb') or 0.0):.2f} GB` "
                f"(target >= `{min_free_vram_target_gb:.1f} GB`)\n"
                f"- Log file: `{str(autotune_log_path) if autotune_log_path else 'not saved'}`"
            )
            if status_reason == "completed":
                final_status = "Auto Tune complete. Applied recommended config."
            elif status_reason == "threshold_reached":
                final_status = "Auto Tune reached VRAM threshold. Applied best-so-far config."
            elif status_reason == "cancelled":
                final_status = "Auto Tune cancelled. Applied best-so-far config."
            else:
                final_status = "Auto Tune hit an error. Applied best-so-far config."
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
