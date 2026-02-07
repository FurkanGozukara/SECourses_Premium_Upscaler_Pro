from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from shared.models.rife_meta import get_rife_default_model
from shared.path_utils import collision_safe_dir, collision_safe_path


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}


def _normalize_multiplier(raw: Any) -> Tuple[int, str]:
    if raw is None:
        return 2, "x2"
    text = str(raw).strip().lower()
    if text.startswith("x"):
        text = text[1:]
    try:
        val = int(float(text))
    except Exception:
        val = 2
    if val <= 1:
        val = 1
    elif val <= 2:
        val = 2
    elif val <= 4:
        val = 4
    else:
        val = 8
    return val, f"x{val}"


def _normalize_precision(raw: Any) -> str:
    text = str(raw or "fp32").strip().lower()
    return "fp16" if text == "fp16" else "fp32"


def _to_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return False
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return bool(raw)


def global_rife_enabled(seed_controls: Dict[str, Any]) -> bool:
    if not isinstance(seed_controls, dict):
        return False

    # Source-of-truth preference:
    # 1) output_settings.frame_interpolation (tab state)
    # 2) cached compatibility keys (legacy/apply-button paths)
    output_settings = seed_controls.get("output_settings", {})
    if isinstance(output_settings, dict) and "frame_interpolation" in output_settings:
        return _to_bool(output_settings.get("frame_interpolation", False))

    if "global_rife_enabled_val" in seed_controls:
        return _to_bool(seed_controls.get("global_rife_enabled_val"))
    if "frame_interpolation_val" in seed_controls:
        return _to_bool(seed_controls.get("frame_interpolation_val"))
    return False


def global_rife_process_chunks_enabled(seed_controls: Dict[str, Any]) -> bool:
    """
    Whether Global RIFE should be applied per chunk before chunk merge.

    Default is True to avoid cross-scene interpolation artifacts on already
    chunked pipelines.
    """
    if not isinstance(seed_controls, dict):
        return True

    output_settings = seed_controls.get("output_settings", {})
    if isinstance(output_settings, dict) and "global_rife_process_chunks" in output_settings:
        return _to_bool(output_settings.get("global_rife_process_chunks", True))

    if "global_rife_process_chunks_val" in seed_controls:
        return _to_bool(seed_controls.get("global_rife_process_chunks_val"))

    return True


def _as_bool(raw: Any, default: bool) -> bool:
    if raw is None:
        return bool(default)
    return _to_bool(raw)


def _as_float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _build_output_path(input_video: Path, multiplier: int) -> Path:
    target = input_video.with_name(f"{input_video.stem}_{multiplier}xFPS{input_video.suffix}")
    return collision_safe_path(target)


def build_global_rife_settings(
    input_video: str,
    seed_controls: Dict[str, Any],
) -> Dict[str, Any]:
    input_path = Path(str(input_video))
    output_settings = seed_controls.get("output_settings", {}) if isinstance(seed_controls, dict) else {}
    if not isinstance(output_settings, dict):
        output_settings = {}

    mult_int, mult_label = _normalize_multiplier(
        output_settings.get(
            "global_rife_multiplier",
            seed_controls.get("global_rife_multiplier_val", "x2"),
        )
    )
    precision = _normalize_precision(
        output_settings.get(
            "global_rife_precision",
            seed_controls.get("global_rife_precision_val", "fp32"),
        )
    )

    # RIFE is single-GPU; if user provides multi-device spec, keep only first token.
    cuda_device = str(
        output_settings.get(
            "global_rife_cuda_device",
            seed_controls.get("global_rife_cuda_device_val", ""),
        ) or ""
    ).strip()
    if "," in cuda_device:
        cuda_device = cuda_device.split(",", 1)[0].strip()
    if cuda_device.lower() == "all":
        cuda_device = "0"

    model_name = str(
        output_settings.get(
            "global_rife_model",
            seed_controls.get("global_rife_model_val", ""),
        ) or ""
    ).strip() or get_rife_default_model()

    settings: Dict[str, Any] = {
        "input_path": str(input_path),
        "rife_enabled": True,
        "output_override": str(_build_output_path(input_path, mult_int)),
        "output_format": input_path.suffix.lstrip(".").lower() or "mp4",
        "model_dir": "",
        "model": model_name,
        "fps_multiplier": mult_label,
        "fps_override": 0,
        "scale": 1.0,
        "uhd_mode": False,
        "fp16_mode": precision == "fp16",
        "png_output": False,
        "no_audio": False,
        "show_ffmpeg": False,
        "montage": False,
        "img_mode": False,
        "skip_static_frames": False,
        "exp": 1,
        "multi": mult_int,
        "batch_enable": False,
        "batch_input_path": "",
        "batch_output_path": "",
        "skip_first_frames": 0,
        "load_cap": 0,
        "cuda_device": cuda_device,
        "edit_mode": "none",
        "start_time": "",
        "end_time": "",
        "speed_factor": 1.0,
        "video_codec": output_settings.get("video_codec", "h264"),
        "output_quality": output_settings.get("video_quality", 18),
        "video_preset": output_settings.get("video_preset", "medium"),
        "pixel_format": output_settings.get("pixel_format", "yuv420p"),
        "two_pass_encoding": bool(output_settings.get("two_pass_encoding", False)),
        "concat_videos": "",
        "save_metadata": bool(seed_controls.get("save_metadata_val", True)),
        "audio_codec": str(seed_controls.get("audio_codec_val", "copy") or "copy"),
        "audio_bitrate": str(seed_controls.get("audio_bitrate_val", "") or ""),
    }
    return settings


def _run_global_rife_single(
    runner,
    outp: Path,
    seed_controls: Dict[str, Any],
    on_log: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[str], str]:
    settings = build_global_rife_settings(str(outp), seed_controls)
    mult = settings.get("fps_multiplier", "x2")
    precision = "fp16" if settings.get("fp16_mode") else "fp32"
    model_name = settings.get("model", "")
    if on_log:
        on_log(f"Global RIFE: starting {mult} with model '{model_name}' ({precision})...\n")

    result = runner.run_rife(
        settings,
        on_progress=(lambda msg: on_log(f"[Global RIFE] {msg}") if on_log else None),
    )
    if result.returncode == 0 and result.output_path and Path(result.output_path).exists():
        msg = f"Global RIFE complete: {result.output_path}"
        if on_log:
            on_log(f"{msg}\n")
        return str(result.output_path), msg

    detail = ""
    if getattr(result, "log", None):
        try:
            lines = [str(x).strip() for x in str(result.log).splitlines() if str(x).strip()]
            if lines:
                detail = lines[-1]
        except Exception:
            detail = ""
    if detail:
        msg = f"Global RIFE failed (code {result.returncode}): {detail}"
    else:
        msg = f"Global RIFE failed (code {result.returncode})."
    if on_log:
        on_log(f"{msg}\n")
    return None, msg


def _run_global_rife_chunked(
    runner,
    outp: Path,
    seed_controls: Dict[str, Any],
    chunking_context: Dict[str, Any],
    on_log: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[str], str]:
    try:
        from shared.chunking import chunk_and_process
    except Exception as e:
        return None, f"Global RIFE chunked mode unavailable: {e}"

    settings = build_global_rife_settings(str(outp), seed_controls)

    auto_chunk = _as_bool(
        chunking_context.get("auto_chunk", seed_controls.get("auto_chunk", True)),
        True,
    )
    chunk_size_sec = _as_float(
        chunking_context.get("chunk_size_sec", seed_controls.get("chunk_size_sec", 0)),
        0.0,
    )
    chunk_overlap_sec = _as_float(
        chunking_context.get("chunk_overlap_sec", seed_controls.get("chunk_overlap_sec", 0.0)),
        0.0,
    )
    if auto_chunk:
        chunk_overlap_sec = 0.0

    scene_threshold = _as_float(
        chunking_context.get("scene_threshold", seed_controls.get("scene_threshold", 27.0)),
        27.0,
    )
    min_scene_len = _as_float(
        chunking_context.get("min_scene_len", seed_controls.get("min_scene_len", 1.0)),
        1.0,
    )
    frame_accurate_split = _as_bool(
        chunking_context.get("frame_accurate_split", seed_controls.get("frame_accurate_split", True)),
        True,
    )
    per_chunk_cleanup = _as_bool(
        chunking_context.get("per_chunk_cleanup", seed_controls.get("per_chunk_cleanup", False)),
        False,
    )
    settings["frame_accurate_split"] = frame_accurate_split

    work_dir = collision_safe_dir(outp.parent / f"{outp.stem}_global_rife_chunks")
    work_dir.mkdir(parents=True, exist_ok=True)

    if on_log:
        mode_label = "Auto scenes" if auto_chunk else f"Static {chunk_size_sec:g}s"
        on_log(f"Global RIFE: chunk-safe mode ({mode_label})\n")

    rc, log_blob, final_out, chunk_count = chunk_and_process(
        runner=runner,
        settings=settings,
        scene_threshold=scene_threshold,
        min_scene_len=min_scene_len,
        work_dir=work_dir,
        on_progress=(lambda msg: on_log(f"[Global RIFE chunked] {msg}") if on_log and msg else None),
        chunk_seconds=0.0 if auto_chunk else chunk_size_sec,
        chunk_overlap=chunk_overlap_sec,
        per_chunk_cleanup=per_chunk_cleanup,
        allow_partial=False,
        global_output_dir=str(outp.parent),
        resume_from_partial=False,
        progress_tracker=None,
        process_func=None,
        model_type="rife",
    )

    if rc == 0 and final_out and Path(final_out).exists():
        msg = f"Global RIFE complete (chunked, {int(chunk_count or 0)} chunks): {final_out}"
        if on_log:
            on_log(f"{msg}\n")
        return str(final_out), msg

    detail = ""
    try:
        lines = [str(x).strip() for x in str(log_blob or "").splitlines() if str(x).strip()]
        if lines:
            detail = lines[-1]
    except Exception:
        detail = ""

    if detail:
        return None, f"Global RIFE chunked failed (code {rc}): {detail}"
    return None, f"Global RIFE chunked failed (code {rc})."


def maybe_apply_global_rife(
    runner,
    output_video_path: Optional[str],
    seed_controls: Dict[str, Any],
    on_log: Optional[Callable[[str], None]] = None,
    chunking_context: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], str]:
    """
    Apply global RIFE post-process to an already-produced video.

    Returns:
        (rife_output_path_or_none, message)
    """
    if not output_video_path:
        return None, "Global RIFE skipped: no output video path."
    outp = Path(output_video_path)
    if not outp.exists() or outp.suffix.lower() not in VIDEO_EXTS:
        return None, "Global RIFE skipped: output is not a supported video file."
    if not global_rife_enabled(seed_controls):
        return None, "Global RIFE disabled (enable in Output & Comparison > Global Enable RIFE)."

    wants_chunked = bool((chunking_context or {}).get("enabled", False))
    if wants_chunked and global_rife_process_chunks_enabled(seed_controls):
        chunked_out, chunked_msg = _run_global_rife_chunked(
            runner=runner,
            outp=outp,
            seed_controls=seed_controls,
            chunking_context=chunking_context or {},
            on_log=on_log,
        )
        if chunked_out and Path(chunked_out).exists():
            return chunked_out, chunked_msg
        if on_log:
            on_log(f"{chunked_msg} Falling back to single-pass Global RIFE.\n")

    return _run_global_rife_single(
        runner=runner,
        outp=outp,
        seed_controls=seed_controls,
        on_log=on_log,
    )
