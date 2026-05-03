"""
SparkVSR subprocess runner for the SECourses pipeline.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .command_logger import get_command_logger
from .models.sparkvsr_meta import get_sparkvsr_metadata
from .path_utils import (
    IMAGE_EXTENSIONS,
    collision_safe_path,
    detect_input_type,
    get_media_fps,
    list_files_sorted,
    normalize_path,
    resolve_output_location,
)


@dataclass
class SparkVSRResult:
    returncode: int
    output_path: Optional[str]
    log: str
    input_fps: float = 30.0
    output_fps: float = 30.0


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _bool(value: Any, default: bool = False) -> bool:
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


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds or 0))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _resolve_python_executable(base_dir: Path) -> str:
    if os.name == "nt":
        candidate = base_dir / "venv" / "Scripts" / "python.exe"
    else:
        candidate = base_dir / "venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def _normalize_cuda_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("cuda:"):
        text = text.split(":", 1)[1].strip()
    return text


def _resolve_sparkvsr_device(device_value: Any) -> tuple[str, Optional[str], Optional[str]]:
    raw = str(device_value or "").strip()
    raw_lower = raw.lower()
    if raw_lower in {"cpu", "none", "off"}:
        return "cpu", "", "[SparkVSR] GPU isolation: CPU mode (CUDA_VISIBLE_DEVICES cleared)"
    if raw_lower in {"", "auto", "cuda"}:
        return "cuda", None, None
    gpu_id = _normalize_cuda_token(raw)
    if gpu_id.isdigit():
        return "cuda", gpu_id, f"[SparkVSR] GPU isolation: CUDA_VISIBLE_DEVICES={gpu_id}, device remapped to cuda"
    return raw, None, None


def _resolve_model_path(base_dir: Path, settings: Dict[str, Any]) -> tuple[Optional[Path], str]:
    override = normalize_path(settings.get("model_path") or "")
    if override:
        path = Path(override)
        if path.exists():
            return path, f"[SparkVSR] Using model path override: {path}"
        return None, f"[SparkVSR] Model path override not found: {path}"

    models_dir = Path(normalize_path(settings.get("models_dir") or "")) if settings.get("models_dir") else base_dir / "SparkVSR" / "models"
    model_name = str(settings.get("model_name") or "SparkVSR-S2").strip()
    meta = get_sparkvsr_metadata(model_name, base_dir=base_dir)
    relative = meta.relative_path if meta else ("SparkVSR" if model_name == "SparkVSR-S2" else model_name)
    candidate = Path(relative)
    if not candidate.is_absolute():
        candidate = models_dir / relative
    if candidate.exists():
        return candidate, f"[SparkVSR] Using local model directory: {candidate}"
    repo_hint = f" ({meta.repo_id})" if meta and meta.repo_id else ""
    return None, (
        f"[SparkVSR] Model directory missing: {candidate}{repo_hint}. "
        "Run Models_Downloader.py --sparkvsr or set Model Path Override."
    )


def _build_temp_video_from_frames(image_paths: List[Path], output_path: Path, fps: float) -> tuple[bool, str]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        return False, f"OpenCV (cv2) is required for SparkVSR image/folder conversion: {exc}"

    if not image_paths:
        return False, "No image frames found to build temporary SparkVSR video input."

    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        return False, f"Failed to read first image frame: {image_paths[0]}"
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, float(fps or 30.0)),
        (width, height),
    )
    if not writer.isOpened():
        return False, f"Failed to create temporary video writer: {output_path}"
    try:
        for frame_path in image_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()

    if not output_path.exists():
        return False, f"Temporary SparkVSR video was not created: {output_path}"
    return True, ""


def _prepare_cli_input(
    input_path: str,
    fps_hint: float,
    log: Callable[[str], None],
) -> tuple[Optional[str], Optional[Path], Optional[str], str]:
    kind = detect_input_type(input_path)
    if kind == "video":
        return input_path, None, None, kind

    tmp_root = Path(tempfile.mkdtemp(prefix="sparkvsr_cli_input_"))
    temp_video = tmp_root / "input_video.mp4"
    if kind == "image":
        ok, err = _build_temp_video_from_frames([Path(input_path)], temp_video, fps_hint)
        if not ok:
            shutil.rmtree(tmp_root, ignore_errors=True)
            return None, None, err, kind
        log(f"[SparkVSR] Converted single image to temporary video input: {temp_video}")
        return str(temp_video), tmp_root, None, kind

    if kind == "directory":
        image_paths = list_files_sorted(input_path, IMAGE_EXTENSIONS)
        if not image_paths:
            shutil.rmtree(tmp_root, ignore_errors=True)
            return None, None, f"Input directory has no supported image files: {input_path}", kind
        ok, err = _build_temp_video_from_frames(image_paths, temp_video, fps_hint)
        if not ok:
            shutil.rmtree(tmp_root, ignore_errors=True)
            return None, None, err, kind
        log(f"[SparkVSR] Converted image directory to temporary video input: {temp_video}")
        return str(temp_video), tmp_root, None, kind

    shutil.rmtree(tmp_root, ignore_errors=True)
    return None, None, f"Unsupported SparkVSR input type: {kind}", kind


def _extract_single_image_from_video(
    video_path: Path,
    image_format: str = "png",
    image_quality: int = 95,
) -> tuple[Optional[str], Optional[str]]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        return None, f"OpenCV is required for SparkVSR single-image export: {exc}"

    if not video_path.exists():
        return None, f"Output video not found for single-image export: {video_path}"

    fmt = str(image_format or "png").strip().lower()
    if fmt not in {"png", "jpg", "webp"}:
        fmt = "png"
    ext = ".jpg" if fmt == "jpg" else f".{fmt}"
    output_path = collision_safe_path(video_path.with_suffix(ext))
    quality = max(1, min(100, int(image_quality or 95)))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, f"Failed to open SparkVSR output video: {video_path}"
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None, f"No frames decoded from SparkVSR output video: {video_path}"

    params: List[int] = []
    if fmt == "jpg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    elif fmt == "webp":
        params = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    if not cv2.imwrite(str(output_path), frame, params):
        return None, f"Failed to save SparkVSR image output: {output_path}"
    return str(output_path), None


def run_sparkvsr(
    settings: Dict[str, Any],
    base_dir: Path,
    on_progress: Optional[Callable[[str], None]] = None,
    cancel_event=None,
    process_handle: Optional[Dict] = None,
) -> SparkVSRResult:
    start_time = time.time()
    log_lines: List[str] = []
    cmd: List[str] = []
    result: Optional[SparkVSRResult] = None
    temp_input_dir: Optional[Path] = None

    def log(message: str) -> None:
        text = str(message or "")
        if text:
            log_lines.append(text)
            if on_progress:
                with suppress(Exception):
                    on_progress(text)

    try:
        original_input_path = normalize_path(settings.get("_effective_input_path") or settings.get("input_path") or "")
        if not original_input_path or not Path(original_input_path).exists():
            return SparkVSRResult(1, None, f"SparkVSR input path not found: {original_input_path}")

        model_path, model_msg = _resolve_model_path(base_dir, settings)
        log(model_msg)
        if model_path is None:
            return SparkVSRResult(1, None, "\n".join(log_lines))

        fps = _parse_float(settings.get("fps"), 0.0)
        input_fps = get_media_fps(original_input_path) or 30.0
        fps_hint = fps if fps > 0 else input_fps
        prepared_input_path, tmp_root, prep_err, original_input_kind = _prepare_cli_input(original_input_path, fps_hint, log)
        if prep_err or not prepared_input_path:
            return SparkVSRResult(1, None, prep_err or "Failed to prepare SparkVSR input.")
        temp_input_dir = tmp_root

        base_name = settings.get("_original_filename") or Path(original_input_path).name
        base_stem = Path(str(base_name)).stem or "output"
        output_override = str(settings.get("output_override") or "").strip()
        if output_override:
            override_path = Path(normalize_path(output_override))
            if override_path.suffix:
                output_file = collision_safe_path(override_path)
            else:
                output_file = collision_safe_path(override_path / f"{base_stem}.mp4")
        else:
            resolved = resolve_output_location(
                input_path=original_input_path,
                output_format="mp4",
                global_output_dir=settings.get("global_output_dir", str(base_dir / "outputs")),
                batch_mode=False,
                original_filename=settings.get("_original_filename"),
            )
            output_file = collision_safe_path(resolved if resolved.suffix else (resolved / f"{base_stem}.mp4"))
        output_file.parent.mkdir(parents=True, exist_ok=True)

        python_exe = _resolve_python_executable(base_dir)
        script_path = base_dir / "tools" / "sparkvsr_inference.py"
        if not script_path.exists():
            return SparkVSRResult(1, None, f"SparkVSR inference script not found: {script_path}")

        device_arg, visible_gpu, gpu_note = _resolve_sparkvsr_device(settings.get("device", "auto"))
        if gpu_note:
            log(gpu_note)
        if python_exe != sys.executable:
            log(f"[SparkVSR] Using venv python: {python_exe}")

        tile_h = max(0, _parse_int(settings.get("tile_height"), 0))
        tile_w = max(0, _parse_int(settings.get("tile_width"), 0))
        overlap_h = max(0, _parse_int(settings.get("overlap_height"), 32))
        overlap_w = max(0, _parse_int(settings.get("overlap_width"), 32))
        if tile_h <= 0 or tile_w <= 0:
            tile_h = 0
            tile_w = 0
            overlap_h = 0
            overlap_w = 0
        scale = max(1, min(16, _parse_int(settings.get("scale"), 4)))

        cmd = [
            python_exe,
            "-u",
            str(script_path),
            "--input_dir",
            str(prepared_input_path),
            "--output_path",
            str(output_file.parent),
            "--output_file",
            str(output_file),
            "--model_path",
            str(model_path),
            "--fps",
            str(fps_hint),
            "--dtype",
            str(settings.get("precision") or "bfloat16"),
            "--seed",
            str(_parse_int(settings.get("seed"), 0)),
            "--upscale_mode",
            str(settings.get("upscale_mode") or "bilinear"),
            "--upscale",
            str(scale),
            "--noise_step",
            str(max(0, _parse_int(settings.get("noise_step"), 0))),
            "--sr_noise_step",
            str(max(1, _parse_int(settings.get("sr_noise_step"), 399))),
            "--tile_size_hw",
            str(tile_h),
            str(tile_w),
            "--overlap_hw",
            str(overlap_h),
            str(overlap_w),
            "--chunk_len",
            str(max(0, _parse_int(settings.get("chunk_len"), 0))),
            "--overlap_t",
            str(max(0, _parse_int(settings.get("overlap_t"), 8))),
            "--ref_mode",
            str(settings.get("ref_mode") or "no_ref"),
            "--ref_prompt_mode",
            str(settings.get("ref_prompt_mode") or "fixed"),
            "--ref_indices",
            str(settings.get("ref_indices") or ""),
            "--ref_guidance_scale",
            str(_parse_float(settings.get("ref_guidance_scale"), 1.0)),
            "--save_format",
            str(settings.get("save_format") or "yuv444p"),
            "--start_frame",
            str(max(0, _parse_int(settings.get("start_frame"), 0))),
            "--end_frame",
            str(_parse_int(settings.get("end_frame"), -1)),
            "--device",
            device_arg,
        ]
        lora_path = normalize_path(settings.get("lora_path") or "")
        if lora_path:
            cmd.extend(["--lora_path", lora_path])
        if _bool(settings.get("cpu_offload"), True):
            cmd.append("--is_cpu_offload")
        if _bool(settings.get("vae_tiling"), True):
            cmd.append("--is_vae_st")
        if _bool(settings.get("group_offload"), False):
            cmd.extend(["--group_offload", "--num_blocks_per_group", str(max(1, _parse_int(settings.get("num_blocks_per_group"), 1)))])
        if _bool(settings.get("png_save"), False):
            cmd.append("--png_save")
        for key, flag in [
            ("ref_api_cache_dir", "--ref_api_cache_dir"),
            ("ref_pisa_cache_dir", "--ref_pisa_cache_dir"),
            ("pisa_python_executable", "--pisa_python_executable"),
            ("pisa_script_path", "--pisa_script_path"),
            ("pisa_sd_model_path", "--pisa_sd_model_path"),
            ("pisa_chkpt_path", "--pisa_chkpt_path"),
            ("pisa_gpu", "--pisa_gpu"),
        ]:
            value = str(settings.get(key) or "").strip()
            if value:
                cmd.extend([flag, value])

        log(
            f"Running SparkVSR: model={settings.get('model_name')}, scale={scale}x, "
            f"dtype={settings.get('precision')}, chunk_len={settings.get('chunk_len')}, "
            f"tile={tile_h}x{tile_w}"
        )
        log(f"Command: {' '.join(cmd)}")

        env = {
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
        legacy_alloc_conf = env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        if legacy_alloc_conf and not env.get("PYTORCH_ALLOC_CONF"):
            env["PYTORCH_ALLOC_CONF"] = legacy_alloc_conf
            log("[SparkVSR] Migrated PYTORCH_CUDA_ALLOC_CONF -> PYTORCH_ALLOC_CONF")
        if visible_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = visible_gpu
        triton_cache_dir = base_dir / "temp" / "triton_cache"
        with suppress(Exception):
            triton_cache_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault("TRITON_CACHE_DIR", str(triton_cache_dir))

        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        preexec_fn = None if os.name == "nt" else os.setsid
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(base_dir),
            env=env,
            creationflags=creationflags,
            preexec_fn=preexec_fn,
        )
        if process_handle is not None:
            process_handle["proc"] = proc

        output_lines: List[str] = []
        last_activity = time.time()
        proc_started = time.time()
        last_progress_text = ""
        while True:
            if cancel_event and cancel_event.is_set():
                log("Cancellation requested - terminating SparkVSR process")
                with suppress(Exception):
                    proc.terminate()
                    proc.wait(timeout=5.0)
                if process_handle is not None:
                    process_handle["proc"] = None
                return SparkVSRResult(1, None, "\n".join(log_lines + ["[Cancelled by user]"]))

            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                text = line.strip()
                if text:
                    output_lines.append(text)
                    log(text)
                    last_progress_text = text
                    last_activity = time.time()
                continue
            if proc.poll() is not None:
                break
            if time.time() - last_activity > 10:
                elapsed = _format_duration(time.time() - proc_started)
                if last_progress_text:
                    heartbeat = f"[SparkVSR] still running | elapsed={elapsed} | last={last_progress_text[:180]}"
                else:
                    heartbeat = f"[SparkVSR] still running | elapsed={elapsed} | waiting for first progress output"
                output_lines.append(heartbeat)
                log(heartbeat)
                last_activity = time.time()
            time.sleep(0.2)

        returncode = int(proc.wait())
        if process_handle is not None:
            process_handle["proc"] = None

        output_path: Optional[str] = str(output_file) if output_file.exists() else None
        if output_path and returncode != 0:
            log(f"SparkVSR exited with code {returncode} after producing output; treating run as successful.")
            returncode = 0
        if not output_path:
            log("No SparkVSR output file generated.")
            result = SparkVSRResult(returncode or 1, None, "\n".join(log_lines + output_lines))
        else:
            log(f"Output saved: {output_path}")
            result = SparkVSRResult(
                int(returncode),
                output_path,
                "\n".join(log_lines),
                input_fps=input_fps,
                output_fps=fps_hint,
            )
            if original_input_kind == "image":
                fmt_pref = str(settings.get("image_output_format", "png") or "png").strip().lower()
                quality_pref = max(1, min(100, _parse_int(settings.get("image_output_quality"), 95)))
                img_path, img_err = _extract_single_image_from_video(Path(output_path), fmt_pref, quality_pref)
                if img_path:
                    log(f"[SparkVSR] Single-image input detected. Exported image output: {img_path}")
                    result.output_path = img_path
                elif img_err:
                    log(f"[SparkVSR] Single-image export skipped: {img_err}")
                    result.returncode = 1
                    result.output_path = None

    except Exception as exc:
        log_lines.append(f"SparkVSR error: {exc}")
        result = SparkVSRResult(1, None, "\n".join(log_lines))
    finally:
        if temp_input_dir:
            with suppress(Exception):
                shutil.rmtree(temp_input_dir, ignore_errors=True)
        if cmd:
            try:
                execution_time = time.time() - start_time
                get_command_logger(base_dir / "executed_commands").log_command(
                    tab_name="sparkvsr",
                    command=cmd,
                    settings=settings,
                    returncode=result.returncode if result else -1,
                    output_path=result.output_path if result else None,
                    error_logs=log_lines[-50:] if result and result.returncode != 0 else None,
                    execution_time=execution_time,
                    additional_info={
                        "model_name": settings.get("model_name"),
                        "scale": settings.get("scale"),
                        "precision": settings.get("precision"),
                        "chunk_len": settings.get("chunk_len"),
                        "tile_height": settings.get("tile_height"),
                        "tile_width": settings.get("tile_width"),
                        "ref_mode": settings.get("ref_mode"),
                    },
                )
                log_lines.append("Command logged to executed_commands folder")
            except Exception as exc:
                log_lines.append(f"Failed to log command: {exc}")

    return result if result else SparkVSRResult(1, None, "\n".join(log_lines))


def discover_sparkvsr_models(base_dir: Path) -> List[str]:
    models_dir = base_dir / "SparkVSR" / "models"
    expected = ["SparkVSR", "SparkVSR-S1"]
    if not models_dir.exists():
        return ["SparkVSR-S2"]
    available = []
    for name in expected:
        if (models_dir / name / "model_index.json").exists():
            available.append("SparkVSR-S2" if name == "SparkVSR" else name)
    return available or ["SparkVSR-S2"]
