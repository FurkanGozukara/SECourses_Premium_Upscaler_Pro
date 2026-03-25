"""
RTX Super Resolution runner backed by NVIDIA Maxine nvvfx Python bindings.
"""

from __future__ import annotations

import re
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from shared.path_utils import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    collision_safe_dir,
    collision_safe_path,
    detect_input_type,
    get_media_dimensions,
    get_media_fps,
    list_files_sorted,
    normalize_path,
    resolve_output_location,
)


@dataclass
class RTXSuperResResult:
    returncode: int
    output_path: Optional[str]
    log: str
    input_fps: float = 30.0
    output_fps: float = 30.0
    frames_processed: int = 0
    total_frames: int = 0
    elapsed_seconds: float = 0.0


SAME_RES_QUALITY_RE = re.compile(r"^(DENOISE_|DEBLUR_)", flags=re.IGNORECASE)


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


def _to_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _format_seconds(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    sec = int(round(seconds))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _format_finish_eta(eta_seconds: Optional[float]) -> str:
    if eta_seconds is None:
        return "ETA unknown"
    if eta_seconds <= 0:
        return "ETA 0s"
    finish_ts = time.time() + float(eta_seconds)
    finish_local = time.strftime("%H:%M:%S", time.localtime(finish_ts))
    return f"ETA {_format_seconds(eta_seconds)} (finish ~{finish_local})"


def _sorted_image_files(folder: Path) -> List[Path]:
    return list_files_sorted(folder, IMAGE_EXTENSIONS)


def _make_image_output_path(
    *,
    input_path: str,
    image_format: str,
    global_output_dir: Optional[str],
    output_override: Optional[str],
    original_filename: Optional[str] = None,
) -> Path:
    fmt = str(image_format or "png").strip().lower()
    if fmt not in {"png", "jpg", "webp"}:
        fmt = "png"
    ext = ".jpg" if fmt == "jpg" else f".{fmt}"

    override_raw = str(output_override or "").strip()
    if override_raw:
        override_path = Path(normalize_path(override_raw))
        if override_path.suffix:
            if override_path.suffix.lower() != ext:
                override_path = override_path.with_suffix(ext)
            return collision_safe_path(override_path)
        base_name = Path(original_filename or Path(input_path).name).stem or "image"
        return collision_safe_path(override_path / f"{base_name}{ext}")

    default_png = resolve_output_location(
        input_path=input_path,
        output_format="png",
        global_output_dir=global_output_dir,
        batch_mode=False,
        original_filename=original_filename,
    )
    target = default_png if default_png.suffix else default_png / f"{Path(input_path).stem}.png"
    if target.suffix.lower() != ext:
        target = target.with_suffix(ext)
    return collision_safe_path(target)


def _make_video_output_path(
    *,
    input_path: str,
    global_output_dir: Optional[str],
    output_override: Optional[str],
    original_filename: Optional[str] = None,
) -> Path:
    override_raw = str(output_override or "").strip()
    if override_raw:
        override_path = Path(normalize_path(override_raw))
        if override_path.suffix:
            if override_path.suffix.lower() != ".mp4":
                override_path = override_path.with_suffix(".mp4")
            return collision_safe_path(override_path)
        base_stem = Path(original_filename or Path(input_path).name).stem or "video"
        return collision_safe_path(override_path / f"{base_stem}.mp4")

    resolved = resolve_output_location(
        input_path=input_path,
        output_format="mp4",
        global_output_dir=global_output_dir,
        batch_mode=False,
        original_filename=original_filename,
    )
    target = resolved if resolved.suffix else (resolved / f"{Path(input_path).stem}.mp4")
    if target.suffix.lower() != ".mp4":
        target = target.with_suffix(".mp4")
    return collision_safe_path(target)


def _make_sequence_output_dir(
    *,
    input_path: str,
    global_output_dir: Optional[str],
    output_override: Optional[str],
    original_filename: Optional[str] = None,
) -> Path:
    override_raw = str(output_override or "").strip()
    if override_raw:
        override_path = Path(normalize_path(override_raw))
        if override_path.suffix:
            return collision_safe_dir(override_path.parent / override_path.stem)
        return collision_safe_dir(override_path)

    resolved = resolve_output_location(
        input_path=input_path,
        output_format="png",
        global_output_dir=global_output_dir,
        batch_mode=False,
        original_filename=original_filename,
    )
    return collision_safe_dir(resolved if not resolved.suffix else (resolved.parent / resolved.stem))


def _build_dimensions_plan(
    *,
    input_width: int,
    input_height: int,
    upscale_factor: float,
    max_edge: int,
    pre_downscale_then_upscale: bool,
    quality_name: str,
) -> Dict[str, Any]:
    in_w = max(1, int(input_width))
    in_h = max(1, int(input_height))
    requested_scale = max(1.0, float(upscale_factor or 1.0))
    max_edge_val = max(0, int(max_edge or 0))
    long_side = max(in_w, in_h)

    cap_ratio = 1.0
    if max_edge_val > 0:
        requested_long = long_side * requested_scale
        if requested_long > float(max_edge_val):
            cap_ratio = max(0.01, float(max_edge_val) / float(requested_long))

    same_res_mode = bool(SAME_RES_QUALITY_RE.match(str(quality_name or "")))
    if same_res_mode:
        # DENOISE/DEBLUR modes are same-resolution modes.
        cap_ratio = 1.0
        requested_scale = 1.0

    effective_scale = requested_scale * cap_ratio
    # Cap must always be enforced:
    # - ON: pre-downscale input first, then run model at requested scale.
    # - OFF: keep input as-is, reduce model scale to effective scale.
    use_pre_down = bool(pre_downscale_then_upscale and cap_ratio < 0.999999)
    preprocess_scale = cap_ratio if use_pre_down else 1.0
    model_scale = requested_scale if use_pre_down else effective_scale
    preprocess_w = max(1, int(round(in_w * preprocess_scale)))
    preprocess_h = max(1, int(round(in_h * preprocess_scale)))

    out_w = max(1, int(round(preprocess_w * model_scale)))
    out_h = max(1, int(round(preprocess_h * model_scale)))

    # Keep encoded video dimensions codec-friendly.
    out_w = max(2, int(out_w))
    out_h = max(2, int(out_h))
    preprocess_w = max(1, int(preprocess_w))
    preprocess_h = max(1, int(preprocess_h))

    if out_w % 2:
        out_w -= 1
    if out_h % 2:
        out_h -= 1
    if preprocess_w % 2:
        preprocess_w -= 1
    if preprocess_h % 2:
        preprocess_h -= 1
    out_w = max(2, out_w)
    out_h = max(2, out_h)
    preprocess_w = max(2, preprocess_w)
    preprocess_h = max(2, preprocess_h)

    return {
        "input_width": in_w,
        "input_height": in_h,
        "requested_scale": float(upscale_factor or 1.0),
        "effective_scale": float(effective_scale),
        "model_scale": float(model_scale),
        "cap_ratio": float(cap_ratio),
        "same_res_mode": same_res_mode,
        "preprocess_width": preprocess_w,
        "preprocess_height": preprocess_h,
        "output_width": out_w,
        "output_height": out_h,
    }


def run_rtx_superres(
    settings: Dict[str, Any],
    base_dir: Path,
    on_progress: Optional[Callable[[str], None]] = None,
    cancel_event=None,
) -> RTXSuperResResult:
    start_ts = time.time()
    logs: List[str] = []
    processed = 0
    total_frames = 0
    input_fps = 30.0
    output_fps = 30.0
    inline_progress_active = False
    inline_progress_width = 0

    def log(msg: str) -> None:
        nonlocal inline_progress_active, inline_progress_width
        text = str(msg or "").rstrip("\r\n")
        if not text:
            return
        logs.append(text)
        is_frame_progress = text.startswith("FRAME_PROGRESS ")
        try:
            if is_frame_progress:
                padded = text
                if inline_progress_width > len(text):
                    padded = text + (" " * (inline_progress_width - len(text)))
                print(f"\r{padded}", end="", flush=True)
                inline_progress_active = True
                inline_progress_width = len(text)
            else:
                if inline_progress_active:
                    print("", flush=True)
                    inline_progress_active = False
                    inline_progress_width = 0
                print(text, flush=True)
        except Exception:
            pass
        if on_progress:
            try:
                on_progress(text + "\n")
            except Exception:
                pass

    try:
        try:
            import cv2  # type: ignore
            import torch  # type: ignore
            from nvvfx import VideoSuperRes  # type: ignore
        except Exception as import_err:
            return RTXSuperResResult(
                returncode=1,
                output_path=None,
                log=f"RTX Super Resolution dependencies missing: {import_err}",
                elapsed_seconds=max(0.0, time.time() - start_ts),
            )

        input_path = normalize_path(settings.get("input_path", ""))
        if not input_path or not Path(input_path).exists():
            return RTXSuperResResult(
                returncode=1,
                output_path=None,
                log="Input path is missing or does not exist.",
                elapsed_seconds=max(0.0, time.time() - start_ts),
            )

        input_kind = detect_input_type(input_path)
        if input_kind not in {"video", "image", "directory"}:
            return RTXSuperResResult(
                returncode=1,
                output_path=None,
                log=f"Unsupported input type for RTX Super Resolution: {input_kind}",
                elapsed_seconds=max(0.0, time.time() - start_ts),
            )

        device_raw = str(settings.get("device", "0") or "0").strip().lower()
        if device_raw in {"cpu", "none", "off"}:
            return RTXSuperResResult(
                returncode=1,
                output_path=None,
                log="RTX Super Resolution requires CUDA GPU; CPU mode is not supported.",
                elapsed_seconds=max(0.0, time.time() - start_ts),
            )
        if device_raw.startswith("cuda:"):
            device_raw = device_raw.split(":", 1)[1].strip()
        if not device_raw.isdigit():
            device_raw = "0"
        device_idx = int(device_raw)

        if not torch.cuda.is_available():
            return RTXSuperResResult(
                returncode=1,
                output_path=None,
                log="CUDA is not available. RTX Super Resolution requires CUDA.",
                elapsed_seconds=max(0.0, time.time() - start_ts),
            )
        torch.cuda.set_device(device_idx)

        quality_name = str(settings.get("quality_preset", "ULTRA") or "ULTRA").strip().upper()
        if quality_name not in VideoSuperRes.QualityLevel.__members__:
            quality_name = "ULTRA"
        quality_enum = VideoSuperRes.QualityLevel[quality_name]

        upscale_factor = _to_float(settings.get("upscale_factor"), 2.0)
        max_edge = _to_int(settings.get("max_resolution"), 0)
        pre_down = _bool(settings.get("pre_downscale_then_upscale"), True)
        non_blocking = _bool(settings.get("non_blocking_inference"), True)
        stream_ptr = _to_int(settings.get("cuda_stream_ptr"), 0)

        global_output_dir = settings.get("global_output_dir")
        output_override = settings.get("output_override")
        original_filename = settings.get("_original_filename")

        image_output_format = str(settings.get("image_output_format", "png") or "png").strip().lower()
        if image_output_format not in {"png", "jpg", "webp"}:
            image_output_format = "png"
        image_output_quality = max(1, min(100, _to_int(settings.get("image_output_quality"), 95)))

        output_format_setting = str(settings.get("output_format", "auto") or "auto").strip().lower()
        if output_format_setting not in {"auto", "mp4", "png"}:
            output_format_setting = "auto"

        if output_format_setting == "auto":
            if input_kind == "video":
                resolved_output_format = "mp4"
            elif input_kind == "directory":
                resolved_output_format = "png"
            else:
                resolved_output_format = image_output_format
        elif output_format_setting == "png":
            resolved_output_format = "png" if input_kind != "image" else image_output_format
        else:
            resolved_output_format = "mp4"

        if input_kind == "image" and resolved_output_format == "mp4":
            # Allow it, but note this is a single-frame video.
            pass

        dims = get_media_dimensions(input_path)
        if not dims:
            return RTXSuperResResult(
                returncode=1,
                output_path=None,
                log="Failed to detect input dimensions.",
                elapsed_seconds=max(0.0, time.time() - start_ts),
            )
        in_w, in_h = int(dims[0]), int(dims[1])
        plan = _build_dimensions_plan(
            input_width=in_w,
            input_height=in_h,
            upscale_factor=upscale_factor,
            max_edge=max_edge,
            pre_downscale_then_upscale=pre_down,
            quality_name=quality_name,
        )
        preprocess_w = int(plan["preprocess_width"])
        preprocess_h = int(plan["preprocess_height"])
        out_w = int(plan["output_width"])
        out_h = int(plan["output_height"])

        if resolved_output_format == "mp4" and (out_w > 8192 or out_h > 8192):
            return RTXSuperResResult(
                returncode=1,
                output_path=None,
                log=(
                    f"Target resolution {out_w}x{out_h} exceeds 8192 for MP4 output. "
                    "Lower Upscale x / Max Resolution, or switch output format to PNG frames."
                ),
                elapsed_seconds=max(0.0, time.time() - start_ts),
            )

        if bool(plan.get("same_res_mode")):
            log(
                f"[RTX] Quality preset {quality_name} is same-resolution mode. "
                f"Output dimensions forced to {out_w}x{out_h}."
            )
        else:
            log(
                f"[RTX] Sizing plan: input {in_w}x{in_h} -> preprocess {preprocess_w}x{preprocess_h} -> output {out_w}x{out_h} "
                f"(requested {upscale_factor:g}x, effective {float(plan.get('effective_scale', 1.0)):.3f}x)."
            )

        output_path: Optional[str] = None
        output_dir_path: Optional[Path] = None

        if input_kind == "image":
            output_img_path = _make_image_output_path(
                input_path=input_path,
                image_format=image_output_format if resolved_output_format != "mp4" else "png",
                global_output_dir=global_output_dir,
                output_override=output_override,
                original_filename=original_filename,
            )
            output_img_path.parent.mkdir(parents=True, exist_ok=True)
            output_path = str(output_img_path)
            total_frames = 1
            input_fps = _to_float(settings.get("fps"), 30.0)
            output_fps = input_fps
        elif resolved_output_format == "png":
            output_dir = _make_sequence_output_dir(
                input_path=input_path,
                global_output_dir=global_output_dir,
                output_override=output_override,
                original_filename=original_filename,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir_path = output_dir
            output_path = str(output_dir)
        else:
            output_video_path = _make_video_output_path(
                input_path=input_path,
                global_output_dir=global_output_dir,
                output_override=output_override,
                original_filename=original_filename,
            )
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            output_path = str(output_video_path)

        input_fps = float(get_media_fps(input_path) or _to_float(settings.get("fps"), 30.0) or 30.0)
        if input_fps <= 0:
            input_fps = 30.0
        output_fps = float(_to_float(settings.get("fps"), input_fps) or input_fps)
        if output_fps <= 0:
            output_fps = input_fps

        sr = VideoSuperRes(device=device_idx, quality=quality_enum)
        sr.output_width = out_w
        sr.output_height = out_h
        sr.load()

        log(
            f"[RTX] Runtime ready: quality={quality_name}, non_blocking={bool(non_blocking)}, "
            f"stream_ptr={int(stream_ptr)}, device=cuda:{device_idx}"
        )

        def preprocess_frame(frame_bgr):
            if frame_bgr is None:
                return None
            if preprocess_w != int(frame_bgr.shape[1]) or preprocess_h != int(frame_bgr.shape[0]):
                return cv2.resize(frame_bgr, (preprocess_w, preprocess_h), interpolation=cv2.INTER_AREA)
            return frame_bgr

        def run_frame(frame_bgr):
            prep = preprocess_frame(frame_bgr)
            if prep is None:
                return None
            rgb = cv2.cvtColor(prep, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).to(f"cuda:{device_idx}")
            tensor = tensor.permute(2, 0, 1).contiguous().float() / 255.0

            out = sr.run(tensor, non_blocking=bool(non_blocking), stream_ptr=int(stream_ptr or 0))
            if non_blocking:
                torch.cuda.synchronize(device_idx)
            out_t = torch.from_dlpack(out.image).clone()
            out_np = (
                out_t.clamp(0.0, 1.0)
                .mul(255.0)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

        # Build frame source.
        cap = None
        frame_paths: List[Path] = []
        single_image = None
        try:
            if input_kind == "video":
                cap = cv2.VideoCapture(str(input_path))
                if not cap.isOpened():
                    return RTXSuperResResult(
                        returncode=1,
                        output_path=None,
                        log=f"Failed to open input video: {input_path}",
                        elapsed_seconds=max(0.0, time.time() - start_ts),
                    )
                raw_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                total_frames = raw_total if raw_total > 0 else 0
            elif input_kind == "directory":
                frame_paths = _sorted_image_files(Path(input_path))
                if not frame_paths:
                    return RTXSuperResResult(
                        returncode=1,
                        output_path=None,
                        log=f"No supported images found in frame folder: {input_path}",
                        elapsed_seconds=max(0.0, time.time() - start_ts),
                    )
                total_frames = len(frame_paths)
            else:
                single_image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
                if single_image is None:
                    return RTXSuperResResult(
                        returncode=1,
                        output_path=None,
                        log=f"Failed to read input image: {input_path}",
                        elapsed_seconds=max(0.0, time.time() - start_ts),
                    )
                total_frames = 1
        except Exception as io_err:
            return RTXSuperResResult(
                returncode=1,
                output_path=None,
                log=f"Input read failed: {io_err}",
                elapsed_seconds=max(0.0, time.time() - start_ts),
            )

        # Build sink.
        video_writer = None
        if output_path and Path(output_path).suffix.lower() in VIDEO_EXTENSIONS:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(output_path), fourcc, float(output_fps), (out_w, out_h))
            if not video_writer.isOpened():
                return RTXSuperResResult(
                    returncode=1,
                    output_path=None,
                    log=f"Failed to create output video writer: {output_path}",
                    elapsed_seconds=max(0.0, time.time() - start_ts),
                )

        ema_frame_sec: Optional[float] = None
        log_eta_interval = 0.8
        last_eta_log_ts = 0.0

        def emit_progress(frame_idx: int, total: int) -> None:
            nonlocal ema_frame_sec, last_eta_log_ts
            now_ts = time.time()
            elapsed = max(1e-6, now_ts - start_ts)
            if frame_idx > 0:
                frame_sec = elapsed / float(frame_idx)
                if ema_frame_sec is None:
                    ema_frame_sec = frame_sec
                else:
                    ema_frame_sec = (ema_frame_sec * 0.7) + (frame_sec * 0.3)
            pct = (float(frame_idx) / float(total)) if total > 0 else 0.0
            eta_seconds = None
            if ema_frame_sec is not None and total > 0 and frame_idx > 0:
                eta_seconds = max(0.0, (float(total - frame_idx) * float(ema_frame_sec)))
            eta_text = _format_finish_eta(eta_seconds)
            line = (
                f"FRAME_PROGRESS {frame_idx}/{max(1, total)} | {pct * 100.0:.1f}% | "
                f"elapsed {_format_seconds(elapsed)} | {eta_text}"
            )
            if now_ts - last_eta_log_ts >= log_eta_interval or frame_idx >= total:
                log(line)
                last_eta_log_ts = now_ts
            elif on_progress:
                try:
                    on_progress(line + "\n")
                except Exception:
                    pass

        frame_idx = 0

        if single_image is not None:
            if cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)()):
                return RTXSuperResResult(
                    returncode=1,
                    output_path=None,
                    log="\n".join(logs + ["Cancelled before inference"]),
                    input_fps=input_fps,
                    output_fps=output_fps,
                    frames_processed=0,
                    total_frames=1,
                    elapsed_seconds=max(0.0, time.time() - start_ts),
                )
            out_frame = run_frame(single_image)
            if out_frame is None:
                return RTXSuperResResult(
                    returncode=1,
                    output_path=None,
                    log="\n".join(logs + ["Failed to process input image frame"]),
                    input_fps=input_fps,
                    output_fps=output_fps,
                    frames_processed=0,
                    total_frames=1,
                    elapsed_seconds=max(0.0, time.time() - start_ts),
                )
            frame_idx = 1
            emit_progress(frame_idx, total_frames or 1)
            out_path = Path(output_path or "")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if video_writer is not None:
                video_writer.write(out_frame)
            else:
                save_params: List[int] = []
                if out_path.suffix.lower() in {".jpg", ".jpeg"}:
                    save_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(image_output_quality)]
                elif out_path.suffix.lower() == ".webp":
                    save_params = [int(cv2.IMWRITE_WEBP_QUALITY), int(image_output_quality)]
                ok = cv2.imwrite(str(out_path), out_frame, save_params)
                if not ok:
                    return RTXSuperResResult(
                        returncode=1,
                        output_path=None,
                        log="\n".join(logs + [f"Failed to save output image: {out_path}"]),
                        input_fps=input_fps,
                        output_fps=output_fps,
                        frames_processed=0,
                        total_frames=1,
                        elapsed_seconds=max(0.0, time.time() - start_ts),
                    )
        else:
            while True:
                if cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)()):
                    return RTXSuperResResult(
                        returncode=1,
                        output_path=None,
                        log="\n".join(logs + ["Cancelled by user"]),
                        input_fps=input_fps,
                        output_fps=output_fps,
                        frames_processed=frame_idx,
                        total_frames=total_frames,
                        elapsed_seconds=max(0.0, time.time() - start_ts),
                    )

                frame_bgr = None
                if cap is not None:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_bgr = frame
                    if total_frames <= 0:
                        total_frames = max(total_frames, frame_idx + 1)
                else:
                    if frame_idx >= len(frame_paths):
                        break
                    frame_file = frame_paths[frame_idx]
                    frame_bgr = cv2.imread(str(frame_file), cv2.IMREAD_COLOR)
                    if frame_bgr is None:
                        log(f"[RTX] Skipping unreadable frame: {frame_file.name}")
                        frame_idx += 1
                        continue

                out_frame = run_frame(frame_bgr)
                if out_frame is None:
                    return RTXSuperResResult(
                        returncode=1,
                        output_path=None,
                        log="\n".join(logs + [f"Failed to process frame {frame_idx + 1}"]),
                        input_fps=input_fps,
                        output_fps=output_fps,
                        frames_processed=frame_idx,
                        total_frames=total_frames,
                        elapsed_seconds=max(0.0, time.time() - start_ts),
                    )

                next_idx = frame_idx + 1
                emit_progress(next_idx, total_frames if total_frames > 0 else max(next_idx, 1))

                if video_writer is not None:
                    video_writer.write(out_frame)
                elif output_dir_path is not None:
                    frame_name = f"{Path(input_path).stem}_{next_idx:06d}.png"
                    out_file = output_dir_path / frame_name
                    cv2.imwrite(str(out_file), out_frame)
                frame_idx = next_idx

        processed = frame_idx
        elapsed = max(0.0, time.time() - start_ts)
        fps_proc = (float(processed) / elapsed) if elapsed > 0 else 0.0
        log(
            f"[RTX] Complete: {processed} frame(s) processed in {_format_seconds(elapsed)} "
            f"({fps_proc:.2f} FPS)."
        )

        final_output = output_path
        if output_dir_path is not None:
            final_output = str(output_dir_path)

        return RTXSuperResResult(
            returncode=0,
            output_path=final_output,
            log="\n".join(logs),
            input_fps=float(input_fps),
            output_fps=float(output_fps),
            frames_processed=int(processed),
            total_frames=int(total_frames),
            elapsed_seconds=float(elapsed),
        )

    except Exception as e:
        logs.append(f"[RTX] Error: {e}")
        return RTXSuperResResult(
            returncode=1,
            output_path=None,
            log="\n".join(logs),
            input_fps=float(input_fps),
            output_fps=float(output_fps),
            frames_processed=int(processed),
            total_frames=int(total_frames),
            elapsed_seconds=max(0.0, time.time() - start_ts),
        )
    finally:
        # Ensure resources are released even on early returns.
        try:
            import cv2  # type: ignore
        except Exception:
            cv2 = None
        with suppress(Exception):
            if "cap" in locals() and cap is not None:
                cap.release()
        with suppress(Exception):
            if "video_writer" in locals() and video_writer is not None:
                video_writer.release()
