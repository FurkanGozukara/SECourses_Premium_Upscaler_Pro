from __future__ import annotations

import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .path_utils import collision_safe_path, detect_input_type, get_media_fps, normalize_path, sanitize_filename


def format_fps_value(value: Optional[float]) -> Optional[str]:
    try:
        if value is None:
            return None
        fps_val = float(value)
        if fps_val <= 0:
            return None
        return f"{fps_val:.3f}".rstrip("0").rstrip(".")
    except Exception:
        return None


def _probe_video_rate_pair(path: Path) -> tuple[Optional[float], Optional[float]]:
    """Return ffprobe's (nominal, average) video rates."""
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,avg_frame_rate",
                "-of", "json", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            return None, None
        stream = (json.loads(proc.stdout or "{}").get("streams") or [{}])[0]

        def _rate(raw: Any) -> Optional[float]:
            try:
                value = float(Fraction(str(raw)))
                return value if math.isfinite(value) and value > 0 else None
            except Exception:
                return None

        return _rate(stream.get("r_frame_rate")), _rate(stream.get("avg_frame_rate"))
    except Exception:
        return None, None


def _packet_durations_match_rate(path: Path, fps: float) -> bool:
    """Confirm CFR packet timing; r/avg_frame_rate alone can both lie for VFR files."""
    if not math.isfinite(float(fps)) or float(fps) <= 0:
        return False
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-read_intervals", "%+#512",
                "-select_streams", "v:0", "-show_entries", "packet=duration_time",
                "-of", "csv=p=0", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            return False
        durations: list[float] = []
        for raw in (proc.stdout or "").splitlines():
            try:
                value = float(raw.strip().strip(","))
            except Exception:
                continue
            if math.isfinite(value) and value > 0:
                durations.append(value)
        if not durations:
            return False
        expected = 1.0 / float(fps)
        # Matroska and MPEG-TS commonly quantize a CFR duration to milliseconds.
        tolerance = max(0.0011, 0.03 * expected)
        return all(abs(value - expected) <= tolerance for value in durations)
    except Exception:
        return False


def normalize_rife_multiplier(raw: Any) -> int:
    text = str(raw or "x2").strip().lower()
    if text.startswith("x"):
        text = text[1:]
    try:
        val = int(float(text))
    except Exception:
        val = 2
    if val <= 1:
        return 1
    if val <= 2:
        return 2
    if val <= 4:
        return 4
    return 8


def build_output_fps_summary(
    *,
    input_fps: Optional[float],
    seed_controls: Optional[Dict[str, Any]],
    output_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from .global_rife import global_rife_enabled

    seed_controls = seed_controls if isinstance(seed_controls, dict) else {}
    output_settings = output_settings if isinstance(output_settings, dict) else {}

    try:
        fps_override = float(seed_controls.get("fps_override_val", output_settings.get("fps_override", 0)) or 0.0)
    except Exception:
        fps_override = 0.0

    base_fps: Optional[float]
    try:
        src_fps = float(input_fps) if input_fps is not None else 0.0
    except Exception:
        src_fps = 0.0
    base_fps = fps_override if fps_override > 0 else (src_fps if src_fps > 0 else None)

    rife_on = bool(global_rife_enabled(seed_controls))
    mult_val = normalize_rife_multiplier(
        seed_controls.get(
            "global_rife_multiplier_val",
            output_settings.get("global_rife_multiplier", "x2"),
        )
    )

    final_fps = (base_fps * float(mult_val)) if (rife_on and base_fps and base_fps > 0) else base_fps
    value_text = format_fps_value(final_fps)

    if rife_on and fps_override > 0:
        label = f"Output FPS (FPS Override + Global RIFE x{mult_val})"
        value_class = "is-override"
    elif rife_on:
        label = f"Output FPS (Global RIFE x{mult_val})"
        value_class = "is-override"
    elif fps_override > 0:
        label = "Output FPS (FPS Override)"
        value_class = "is-override"
    else:
        label = "Output FPS (Base)"
        value_class = ""

    if not value_text:
        value_text = "Unavailable (input FPS unknown)"

    return {
        "label": label,
        "value": value_text,
        "value_class": value_class,
        "base_fps": base_fps,
        "final_fps": final_fps,
        "fps_override": fps_override,
        "global_rife_enabled": rife_on,
        "global_rife_multiplier": mult_val,
    }


def remux_video_fps(
    input_path: Path,
    output_path: Path,
    fps: float,
    *,
    on_progress: Optional[Callable[[str], None]] = None,
) -> tuple[bool, str]:
    """
    Convert a video to a real CFR timeline while preserving playback duration.

    Stream-copy plus ``-r`` does not change H.264/HEVC packet timestamps; the old
    implementation therefore reported success while leaving the source FPS untouched.
    A true override necessarily duplicates/drops decoded frames, so use a lossless x264
    intermediate and validate the resulting rate, duration, frame count, and timeline origin.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if fps <= 0:
        return False, "Invalid FPS override"
    if not input_path.exists():
        return False, f"Input not found: {input_path}"
    if shutil.which("ffmpeg") is None:
        return False, "ffmpeg not available"

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    target_rate = Fraction(str(float(fps))).limit_denominator(1_000_000)
    fps_text = f"{target_rate.numerator}/{target_rate.denominator}"

    def _probe(path: Path, *, count_frames: bool = False) -> dict[str, Any]:
        try:
            cmd_probe = ["ffprobe", "-v", "error"]
            if count_frames:
                cmd_probe.append("-count_frames")
            cmd_probe += [
                "-show_entries",
                (
                    "stream=index,codec_type,pix_fmt,r_frame_rate,avg_frame_rate,start_time,duration,"
                    "nb_frames,nb_read_frames"
                ),
                "-of",
                "json",
                str(path),
            ]
            proc_probe = subprocess.run(cmd_probe, capture_output=True, text=True, timeout=1800)
            if proc_probe.returncode == 0:
                return json.loads(proc_probe.stdout or "{}")
        except Exception:
            pass
        return {}

    source_probe = _probe(input_path)
    source_streams = source_probe.get("streams") or []
    source_video = next((s for s in source_streams if s.get("codec_type") == "video"), {})
    source_audio = next((s for s in source_streams if s.get("codec_type") == "audio"), {})

    def _number(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) else default
        except Exception:
            return default

    source_video_start = _number(source_video.get("start_time"), 0.0)
    source_audio_start = _number(source_audio.get("start_time"), source_video_start)
    relative_audio_start = source_audio_start - source_video_start
    source_duration = _number(source_video.get("duration"), 0.0)
    source_pix_fmt = str(source_video.get("pix_fmt") or "").strip().lower()
    x264_pix_fmts = {
        "yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p10le",
        "yuv444p10le", "gray", "gray10le",
    }
    preferred_pix_fmt = source_pix_fmt if source_pix_fmt in x264_pix_fmts else "yuv420p"
    has_audio = bool(source_audio)

    def _build_cmd(pixel_format: str) -> list[str]:
        video_filter = f"fps=fps={fps_text}:round=near,setpts=N/({fps_text}*TB)"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(input_path),
            "-map", "0:v:0",
        ]
        if has_audio:
            cmd += ["-map", "0:a:0?"]
        cmd += [
            "-vf", video_filter,
            "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0",
            "-pix_fmt", pixel_format,
            "-r", fps_text,
            "-fps_mode", "cfr",
        ]
        if has_audio:
            cmd += [
                "-af", f"asetpts=PTS-STARTPTS+({relative_audio_start:.12f})/TB",
                "-c:a", "aac", "-b:a", "192k",
            ]
        else:
            cmd.append("-an")
        cmd += ["-map_metadata", "0"]
        if output_path.suffix.lower() in {".mp4", ".m4v", ".mov"}:
            cmd += ["-movflags", "+faststart"]
        cmd.append(str(output_path))
        return cmd

    def _validate() -> tuple[bool, str]:
        if not output_path.exists() or output_path.stat().st_size <= 1024:
            return False, "FPS preprocess produced no usable output"
        payload = _probe(output_path, count_frames=True)
        streams = payload.get("streams") or []
        video = next((s for s in streams if s.get("codec_type") == "video"), {})
        if not video:
            return False, "FPS preprocess output has no video stream"

        def _rate(value: Any) -> float:
            try:
                return float(Fraction(str(value)))
            except Exception:
                return 0.0

        actual_fps = _rate(video.get("r_frame_rate")) or _rate(video.get("avg_frame_rate"))
        if actual_fps <= 0 or abs(actual_fps - float(target_rate)) > max(1e-6, float(target_rate) * 1e-5):
            return False, f"FPS preprocess rate mismatch: {actual_fps:.6f} vs {float(target_rate):.6f}"
        frame_raw = str(video.get("nb_read_frames") or video.get("nb_frames") or "").strip()
        if not frame_raw.isdigit() or int(frame_raw) <= 0:
            return False, "FPS preprocess frame count unavailable"
        actual_frames = int(frame_raw)
        start_time = _number(video.get("start_time"), 0.0)
        if abs(start_time) > 0.001:
            return False, f"FPS preprocess video starts at {start_time:.6f}s instead of zero"
        actual_duration = _number(video.get("duration"), 0.0)
        frame_timeline_duration = float(actual_frames) / float(target_rate)
        if actual_duration <= 0 or abs(actual_duration - frame_timeline_duration) > max(
            0.02, 1.25 / float(target_rate)
        ):
            return False, (
                f"FPS preprocess timeline mismatch: duration={actual_duration:.6f}s, "
                f"frames/rate={frame_timeline_duration:.6f}s"
            )
        if source_duration > 0 and actual_duration > 0:
            duration_tol = max(0.10, 2.0 / float(target_rate))
            if abs(actual_duration - source_duration) > duration_tol:
                return False, (
                    f"FPS preprocess duration mismatch: {actual_duration:.6f}s vs {source_duration:.6f}s"
                )
            expected_frames = int(round(source_duration * float(target_rate)))
            if abs(actual_frames - expected_frames) > 2:
                return False, (
                    f"FPS preprocess frame inventory mismatch: {actual_frames} vs about {expected_frames} "
                    f"for {source_duration:.6f}s at {float(target_rate):.9f} fps"
                )
        return True, ""

    if on_progress:
        try:
            on_progress(f"Applying FPS override preprocess -> {output_path.name}\n")
        except Exception:
            pass

    try:
        last_error = ""
        for pixel_format in dict.fromkeys((preferred_pix_fmt, "yuv420p")):
            try:
                output_path.unlink(missing_ok=True)
            except Exception:
                pass
            cmd = _build_cmd(pixel_format)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                valid, validation_error = _validate()
                if valid:
                    return True, ""
                last_error = validation_error
            else:
                last_error = (proc.stderr or proc.stdout or "ffmpeg FPS conversion failed").strip()
    except Exception as exc:
        return False, str(exc)
    return False, last_error or "ffmpeg FPS conversion failed"


def apply_video_fps_override_preprocess(
    settings: Dict[str, Any],
    *,
    fps_key: str,
    run_dir: Path,
    on_progress: Optional[Callable[[str], None]] = None,
    input_key: str = "input_path",
    effective_input_key: Optional[str] = "_effective_input_path",
    original_input_key: str = "_original_input_path_before_preprocess",
    preprocessed_input_key: str = "_preprocessed_input_path",
) -> tuple[bool, str]:
    if not isinstance(settings, dict):
        return True, ""

    try:
        target_fps = float(settings.get(fps_key) or 0.0)
    except Exception:
        target_fps = 0.0

    if target_fps <= 0:
        return True, ""

    input_path_raw = settings.get(effective_input_key) if effective_input_key else None
    if not input_path_raw:
        input_path_raw = settings.get(input_key)
    input_path = normalize_path(str(input_path_raw or ""))
    if not input_path:
        return True, ""
    if detect_input_type(input_path) != "video":
        return True, ""

    source_fps = get_media_fps(input_path)
    source_fps_text = format_fps_value(source_fps)
    target_fps_text = format_fps_value(target_fps) or str(float(target_fps))

    nominal_fps, average_fps = _probe_video_rate_pair(Path(input_path))
    source_is_cfr = bool(
        nominal_fps
        and average_fps
        and abs(float(nominal_fps) - float(average_fps))
        <= max(1e-6, 0.002 * float(nominal_fps))
        and _packet_durations_match_rate(Path(input_path), float(nominal_fps))
    )
    if source_fps and source_is_cfr and abs(float(source_fps) - float(target_fps)) <= 0.01:
        settings[fps_key] = 0.0
        settings["_fps_override_requested"] = float(target_fps)
        settings["_fps_override_source_fps"] = float(source_fps)
        settings["_fps_override_target_fps"] = float(target_fps)
        return True, f"FPS override matches source FPS ({target_fps_text}); no remux needed."

    run_dir = Path(run_dir)
    original_name = str(settings.get("_original_filename") or Path(input_path).name or "input.mp4")
    safe_stem = sanitize_filename(Path(original_name).stem or "input")
    fps_token = target_fps_text.replace(".", "_")
    # The validated CFR intermediate is lossless H.264/AAC. Always use MP4 rather than
    # inheriting an incompatible source suffix such as .webm or .wmv.
    output_path = collision_safe_path(run_dir / f"fps_override_{safe_stem}_{fps_token}.mp4")

    ok, err = remux_video_fps(Path(input_path), output_path, target_fps, on_progress=on_progress)
    if not ok:
        return False, (
            f"FPS override preprocess failed for {Path(input_path).name}: "
            f"{err or 'ffmpeg remux failed'}"
        )

    original_input = normalize_path(str(settings.get(original_input_key) or "")) or input_path
    settings[original_input_key] = original_input
    settings[preprocessed_input_key] = str(output_path)
    settings["_fps_override_preprocessed_input_path"] = str(output_path)
    settings["_fps_override_requested"] = float(target_fps)
    settings["_fps_override_source_fps"] = float(source_fps) if source_fps else 0.0
    settings["_fps_override_target_fps"] = float(target_fps)
    settings[input_key] = str(output_path)
    if effective_input_key:
        settings[effective_input_key] = str(output_path)
    settings[fps_key] = 0.0

    if source_fps_text:
        return True, (
            f"FPS override preprocess applied: {source_fps_text} -> {target_fps_text} FPS "
            f"({output_path.name})"
        )
    return True, f"FPS override preprocess applied: target {target_fps_text} FPS ({output_path.name})"
