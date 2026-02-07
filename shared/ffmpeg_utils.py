import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional, Tuple


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def scale_video(
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    *,
    video_codec: str = "libx264",
    preset: str = "veryfast",
    lossless: bool = True,
    audio_copy_first: bool = True,
    audio_bitrate: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """
    Resize a video with ffmpeg and preserve audio when possible.

    Behavior:
    - Tries audio passthrough first (copy) when `audio_copy_first=True`.
    - If audio copy fails (common with incompatible codecs/containers), retries with AAC.
    - Uses lossless x264 by default (`-qp 0`) to minimize quality loss in preprocessing.

    Returns: (ok, error_message)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not _has_ffmpeg():
        return False, "ffmpeg not available"
    if not input_path.exists():
        return False, f"Input not found: {input_path}"

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    scale_filter = f"scale={int(width)}:{int(height)}:flags=lanczos"

    v_args = ["-c:v", str(video_codec), "-preset", str(preset)]
    if lossless and str(video_codec).lower() in ("libx264", "h264"):
        v_args += ["-qp", "0"]
    elif lossless and str(video_codec).lower() in ("libx265", "hevc", "h265"):
        v_args += ["-x265-params", "lossless=1"]
    elif lossless:
        # Generic fallback: avoid forcing lossless flags for unknown codecs.
        pass

    def _run(cmd: list[str]) -> Tuple[bool, str]:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1024:
                return True, ""
            return False, (proc.stderr or proc.stdout or "ffmpeg scale failed").strip()
        except Exception as e:
            return False, str(e)

    # Try audio copy first (best quality) when requested.
    attempts: list[list[str]] = []
    if audio_copy_first:
        attempts.append(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-vf",
                scale_filter,
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                *v_args,
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                "-avoid_negative_ts",
                "make_zero",
                str(output_path),
            ]
        )

    # AAC fallback (robust across containers/codecs).
    aac_args = ["-c:a", "aac"]
    if audio_bitrate:
        aac_args += ["-b:a", str(audio_bitrate)]
    else:
        aac_args += ["-b:a", "192k"]

    attempts.append(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            scale_filter,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            *v_args,
            *aac_args,
            "-movflags",
            "+faststart",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]
    )

    last_err = ""
    for cmd in attempts:
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
        if on_progress:
            on_progress(f"Preprocessing with ffmpeg -> {output_path.name}\n")
        ok, err = _run(cmd)
        if ok:
            return True, ""
        last_err = err

    return False, last_err

