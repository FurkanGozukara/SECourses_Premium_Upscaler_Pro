from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .path_utils import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    collision_safe_dir,
    collision_safe_path,
    detect_input_type,
    get_media_dimensions,
    list_files_sorted,
)
from .resolution_calculator import estimate_fixed_scale_upscale_plan_from_dims


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _floor_even(value: int) -> int:
    return max(2, int((int(value) // 2) * 2))


def _normalize_codec(value: Any) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "": "libx264",
        "h264": "libx264",
        "x264": "libx264",
        "hevc": "libx265",
        "h265": "libx265",
        "x265": "libx265",
        "av1": "libsvtav1",
    }
    return aliases.get(raw, raw or "libx264")


def _target_matches(width: int, height: int, target_w: int, target_h: int) -> bool:
    return abs(int(width) - int(target_w)) <= 1 and abs(int(height) - int(target_h)) <= 1


def _emit(on_log: Optional[Callable[[str], None]], message: str) -> None:
    if on_log:
        try:
            on_log(str(message))
        except Exception:
            pass


def _resize_video(
    src: Path,
    dst: Path,
    target_w: int,
    target_h: int,
    settings: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    if shutil.which("ffmpeg") is None:
        return False, "ffmpeg not available"

    settings = settings or {}
    codec = _normalize_codec(settings.get("video_codec") or settings.get("codec"))
    preset = str(settings.get("video_preset") or settings.get("preset") or "medium").strip() or "medium"
    crf = max(0, min(51, _to_int(settings.get("video_quality", settings.get("crf", 18)), 18)))
    pixel_format = str(settings.get("pixel_format") or "yuv420p").strip() or "yuv420p"
    audio_bitrate = str(settings.get("audio_bitrate") or "").strip()

    scale_filter = f"scale={int(target_w)}:{int(target_h)}:flags=lanczos"
    video_args = ["-c:v", codec]
    if codec in {"libx264", "libx265", "libsvtav1"}:
        video_args.extend(["-preset", preset, "-crf", str(crf)])
    if pixel_format and codec in {"libx264", "libx265", "libsvtav1"}:
        video_args.extend(["-pix_fmt", pixel_format])

    def run(cmd: list[str]) -> Tuple[bool, str]:
        try:
            dst.unlink(missing_ok=True)
        except Exception:
            pass
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and dst.exists() and dst.stat().st_size > 1024:
            return True, ""
        return False, (proc.stderr or proc.stdout or "ffmpeg resize failed").strip()

    base = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        scale_filter,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        *video_args,
    ]
    ok, err = run(
        [
            *base,
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            "-avoid_negative_ts",
            "make_zero",
            str(dst),
        ]
    )
    if ok:
        return True, ""

    audio_args = ["-c:a", "aac"]
    if audio_bitrate:
        audio_args.extend(["-b:a", audio_bitrate])
    else:
        audio_args.extend(["-b:a", "192k"])
    return run(
        [
            *base,
            *audio_args,
            "-movflags",
            "+faststart",
            "-avoid_negative_ts",
            "make_zero",
            str(dst),
        ]
    )


def _resize_image(src: Path, dst: Path, target_w: int, target_h: int, settings: Optional[Dict[str, Any]]) -> bool:
    from PIL import Image

    settings = settings or {}
    quality = max(1, min(100, _to_int(settings.get("image_output_quality", settings.get("output_quality", 95)), 95)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        resized = image.resize((int(target_w), int(target_h)), Image.Resampling.LANCZOS)
        suffix = dst.suffix.lower()
        save_kwargs: Dict[str, Any] = {}
        if suffix in {".jpg", ".jpeg"}:
            if resized.mode not in {"RGB", "L"}:
                resized = resized.convert("RGB")
            save_kwargs["quality"] = quality
        elif suffix == ".webp":
            save_kwargs["quality"] = quality
        resized.save(dst, **save_kwargs)
    return dst.exists() and dst.stat().st_size > 0


def _resize_directory(
    src_dir: Path,
    dst_dir: Path,
    target_w: int,
    target_h: int,
    settings: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    files = list_files_sorted(src_dir, IMAGE_EXTENSIONS)
    if not files:
        return False, f"No image frames found in output directory: {src_dir}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    try:
        for frame in files:
            out_frame = dst_dir / frame.name
            dims = get_media_dimensions(str(frame))
            if dims and _target_matches(int(dims[0]), int(dims[1]), target_w, target_h):
                shutil.copy2(frame, out_frame)
            else:
                _resize_image(frame, out_frame, target_w, target_h, settings)
        return any(dst_dir.iterdir()), ""
    except Exception as exc:
        return False, str(exc)


def enforce_fixed_scale_output_size(
    *,
    output_path: Any,
    source_input_path: Any,
    requested_scale: Any,
    model_scale: Any,
    max_edge: Any = 0,
    pre_downscale_then_upscale: Any = True,
    min_side: int = 64,
    settings: Optional[Dict[str, Any]] = None,
    label: str = "fixed-scale",
    on_log: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[str], bool, str]:
    """
    Enforce the final saved dimensions for a fixed-scale output.

    When pre-downscale is disabled, fixed-scale models still produce their raw
    native-scale output. This post-resize step is what makes Upscale x and
    Max Resolution exact without pretending that preprocessing happened.
    """
    if not output_path:
        return None, False, "No output path."
    out_path = Path(str(output_path))
    if not out_path.exists():
        return str(out_path), False, f"Output not found: {out_path}"

    src_path = Path(str(source_input_path or ""))
    if not src_path.exists():
        return str(out_path), False, f"Source input not found for final-size enforcement: {src_path}"

    source_dims = get_media_dimensions(str(src_path))
    if not source_dims:
        return str(out_path), False, f"Could not read source dimensions: {src_path}"

    try:
        plan = estimate_fixed_scale_upscale_plan_from_dims(
            int(source_dims[0]),
            int(source_dims[1]),
            requested_scale=_to_float(requested_scale, _to_float(model_scale, 4.0)),
            model_scale=max(2, _to_int(model_scale, 4)),
            max_edge=max(0, _to_int(max_edge, 0)),
            force_pre_downscale=bool(pre_downscale_then_upscale),
            min_side=max(2, int(min_side or 2)),
        )
    except Exception as exc:
        return str(out_path), False, f"Could not build final-size plan: {exc}"

    target_w = _floor_even(int(plan.final_saved_width or plan.resize_width or 0))
    target_h = _floor_even(int(plan.final_saved_height or plan.resize_height or 0))
    if target_w <= 0 or target_h <= 0:
        return str(out_path), False, "Final-size plan did not produce valid dimensions."

    output_dims = get_media_dimensions(str(out_path))
    if output_dims and _target_matches(int(output_dims[0]), int(output_dims[1]), target_w, target_h):
        return str(out_path), False, f"Output already matches final size {target_w}x{target_h}."

    kind = detect_input_type(str(out_path))
    suffix = out_path.suffix.lower()
    if out_path.is_file() and (kind == "video" or suffix in VIDEO_EXTENSIONS):
        dst = collision_safe_path(out_path.with_name(f"{out_path.stem}_final{target_w}x{target_h}{out_path.suffix}"))
        _emit(on_log, f"{label}: resizing final video to {target_w}x{target_h}.")
        ok, err = _resize_video(out_path, dst, target_w, target_h, settings)
        if ok:
            return str(dst), True, f"Final output resized to {target_w}x{target_h}: {dst.name}"
        return str(out_path), False, f"Final video resize failed: {err}"

    if out_path.is_file() and (kind == "image" or suffix in IMAGE_EXTENSIONS):
        dst = collision_safe_path(out_path.with_name(f"{out_path.stem}_final{target_w}x{target_h}{out_path.suffix}"))
        _emit(on_log, f"{label}: resizing final image to {target_w}x{target_h}.")
        try:
            if _resize_image(out_path, dst, target_w, target_h, settings):
                return str(dst), True, f"Final output resized to {target_w}x{target_h}: {dst.name}"
        except Exception as exc:
            return str(out_path), False, f"Final image resize failed: {exc}"
        return str(out_path), False, "Final image resize failed."

    if out_path.is_dir():
        dst_dir = collision_safe_dir(out_path.with_name(f"{out_path.name}_final{target_w}x{target_h}"))
        _emit(on_log, f"{label}: resizing final frame directory to {target_w}x{target_h}.")
        ok, err = _resize_directory(out_path, dst_dir, target_w, target_h, settings)
        if ok:
            return str(dst_dir), True, f"Final frame directory resized to {target_w}x{target_h}: {dst_dir.name}"
        return str(out_path), False, f"Final frame directory resize failed: {err}"

    return str(out_path), False, f"Unsupported output type for final-size enforcement: {out_path}"
