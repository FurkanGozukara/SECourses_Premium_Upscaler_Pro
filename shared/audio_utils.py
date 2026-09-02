import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional, Tuple


def _emit_log(message: str, on_progress: Optional[Callable[[str], None]] = None) -> None:
    line = str(message)
    if not line.endswith("\n"):
        line += "\n"
    try:
        print(line, end="", flush=True)
    except Exception:
        pass
    if on_progress:
        try:
            on_progress(line)
        except Exception:
            pass


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(f"\"{c}\"" if " " in str(c) else str(c) for c in cmd)


def _display_media_ref(path_a: Path, path_b: Path) -> tuple[str, str]:
    """
    Format media path labels for logs.

    If both files share the same basename, return full paths so logs remain
    unambiguous and do not look like self-muxing.
    """
    pa = Path(path_a)
    pb = Path(path_b)
    if pa.name.lower() == pb.name.lower():
        return str(pa), str(pb)
    return pa.name, pb.name


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def has_audio_stream(path: Path) -> bool:
    """
    Return True if `path` contains at least one audio stream (ffprobe-based).
    """
    try:
        if not path.exists() or path.stat().st_size < 1024:
            return False
    except Exception:
        return False

    if not _has_ffmpeg():
        return False

    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.returncode == 0 and bool((proc.stdout or "").strip())
    except Exception:
        return False


def _probe_stream_fields(path: Path, select_streams: str, fields: str) -> dict[str, str]:
    """
    Best-effort ffprobe helper used only for logging diagnostics.
    """
    out: dict[str, str] = {}
    try:
        if not path.exists() or path.stat().st_size < 1024:
            return out
    except Exception:
        return out
    if not _has_ffmpeg():
        return out
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                select_streams,
                "-show_entries",
                fields,
                "-of",
                "default=noprint_wrappers=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return out
        for raw in str(proc.stdout or "").splitlines():
            line = str(raw or "").strip()
            if not line or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = str(key).strip().lower()
            if key.startswith("tag:"):
                key = key[4:]
            val = str(val).strip()
            if not key or not val:
                continue
            out[key] = val
    except Exception:
        return {}
    return out


def _probe_video_stream(path: Path) -> dict[str, str]:
    return _probe_stream_fields(
        path,
        "v:0",
        "stream=codec_name,profile,pix_fmt,codec_tag_string:stream_tags=encoder",
    )


def _probe_audio_stream(path: Path) -> dict[str, str]:
    return _probe_stream_fields(
        path,
        "a:0",
        "stream=codec_name,profile,sample_rate,channel_layout,channels",
    )


def _probe_stream_duration_seconds(path: Path, select_streams: str) -> Optional[float]:
    """Duration (seconds) of the first selected stream, or None."""
    try:
        if not path.exists() or path.stat().st_size < 1024:
            return None
    except Exception:
        return None
    if not _has_ffmpeg():
        return None
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                select_streams,
                "-show_entries",
                "stream=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            return None
        raw = (proc.stdout or "").strip().splitlines()
        if not raw:
            return None
        val = float(raw[0].strip())
        return val if val > 0 else None
    except Exception:
        return None


def _probe_stream_start_seconds(path: Path, select_streams: str) -> Optional[float]:
    """Start timestamp of the first selected stream, including an exact zero."""
    try:
        fields = _probe_stream_fields(path, select_streams, "stream=start_time")
        raw = fields.get("start_time")
        if raw is None:
            return None
        value = float(raw)
        return value if value == value and value not in {float("inf"), float("-inf")} else None
    except Exception:
        return None


def _probe_decodable_video_packets(path: Path) -> Optional[int]:
    """Count decoded video frames, with playable packets as a last-resort fallback."""
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=nb_read_frames",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        raw = (proc.stdout or "").strip() if proc.returncode == 0 else ""
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    except Exception:
        pass
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "packet=flags", "-of", "csv=p=0", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            return None
        count = 0
        seen = False
        for raw in (proc.stdout or "").splitlines():
            flags = raw.strip().strip(",")
            if not flags:
                continue
            seen = True
            if "D" not in flags:
                count += 1
        return count if seen and count > 0 else None
    except Exception:
        return None


def _video_timing_preserved(
    source_path: Path,
    candidate_path: Path,
    source_frames: Optional[int] = None,
    source_duration: Optional[float] = None,
) -> tuple[bool, str]:
    """Audio muxing must be a bitstream-copy operation for the video timeline."""
    source_frames = source_frames or _probe_decodable_video_packets(source_path)
    candidate_frames = _probe_decodable_video_packets(candidate_path)
    if source_frames is not None and candidate_frames is not None and candidate_frames != source_frames:
        return False, f"video frame count changed {source_frames}->{candidate_frames}"
    source_duration = source_duration or _probe_stream_duration_seconds(source_path, "v:0")
    candidate_duration = _probe_stream_duration_seconds(candidate_path, "v:0")
    if source_duration and candidate_duration:
        tolerance = max(0.10, 0.002 * float(source_duration))
        if abs(float(candidate_duration) - float(source_duration)) > tolerance:
            return False, (
                f"video duration changed {float(source_duration):.6f}s->"
                f"{float(candidate_duration):.6f}s"
            )
    candidate_start = _probe_stream_start_seconds(candidate_path, "v:0")
    if candidate_start is not None and abs(float(candidate_start)) > 0.001:
        return False, f"video timeline starts at {float(candidate_start):.6f}s instead of zero"
    return True, ""


def _build_audio_args(audio_codec: str, audio_bitrate: Optional[str]) -> list[str]:
    codec = (audio_codec or "copy").strip().lower()
    if codec in ("none", "no", "off", "disable", "disabled"):
        return ["-an"]
    if codec in ("copy", "passthrough"):
        return ["-c:a", "copy"]

    args = ["-c:a", codec]
    if audio_bitrate:
        args += ["-b:a", str(audio_bitrate)]
    elif codec == "aac":
        # Sensible default for compatibility/quality.
        args += ["-b:a", "192k"]
    return args


def mux_audio(
    video_path: Path,
    audio_source_path: Path,
    output_path: Path,
    audio_codec: str = "copy",
    audio_bitrate: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """
    Mux audio from `audio_source_path` into `video_path`, writing to `output_path`.
    Keeps video stream bit-exact via `-c:v copy`.
    """
    if not _has_ffmpeg():
        _emit_log("[audio] ffmpeg/ffprobe not available; mux skipped.", on_progress)
        return False, "ffmpeg/ffprobe not available"

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    args_audio = _build_audio_args(audio_codec, audio_bitrate)
    codec_norm = (audio_codec or "copy").strip().lower()
    want_audio = codec_norm not in {"none", "no", "off", "disable", "disabled"}
    src_has_audio = has_audio_stream(audio_source_path) if want_audio else False

    # Never let the audio mux shorten the VIDEO. The old unconditional `-shortest`
    # trimmed the video whenever it was longer than the audio, which silently hid
    # chunk-pipeline bugs (duplicated boundary frames) as "the ending is missing" +
    # progressive A/V drift. Only clamp the AUDIO when it is clearly longer than the
    # video (e.g. partial/preview runs muxed against a full-length source track).
    video_dur = _probe_stream_duration_seconds(video_path, "v:0")
    video_frames = _probe_decodable_video_packets(video_path)
    audio_dur = _probe_stream_duration_seconds(audio_source_path, "a:0") if want_audio else None
    clamp_audio_to_video = False
    if video_dur is not None and audio_dur is not None:
        delta = float(video_dur) - float(audio_dur)
        if delta < -0.5:
            clamp_audio_to_video = True
        if abs(delta) > 0.5:
            _emit_log(
                "[audio] WARNING: A/V duration mismatch: "
                f"video={float(video_dur):.3f}s audio={float(audio_dur):.3f}s (video-audio={delta:+.3f}s). "
                + (
                    "Audio will be clamped to the video length."
                    if clamp_audio_to_video
                    else "Video is kept intact (NOT truncated); audio will end early."
                ),
                on_progress,
            )

    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    if want_audio:
        cmd += ["-i", str(audio_source_path)]
    cmd += ["-map", "0:v:0"]
    if want_audio:
        cmd += ["-map", "1:a?"]
    cmd += ["-c:v", "copy", *args_audio]
    if clamp_audio_to_video:
        cmd.append("-shortest")
    # NOTE: no `-avoid_negative_ts make_zero` here. Shifting all timestamps so the first DTS
    # is zero moves the video's first PTS to +N B-frames (e.g. 0.08 s at 25 fps); depending on
    # the ffmpeg build that offset ends up in the reported track duration and is added at every
    # chunk boundary by the concat demuxer (2 frames per chunk = ~50 s on a 42-min film), and it
    # also introduced a constant ~21 ms audio lead in final outputs. The MP4 muxer default
    # handles the B-frame delay with an edit list, exactly like a plain `ffmpeg -c:v libx264`.

    # MP4 nicety: faststart if output is mp4-ish.
    if output_path.suffix.lower() in (".mp4", ".m4v", ".mov"):
        cmd += ["-movflags", "+faststart"]

    cmd.append(str(output_path))

    dst_has_audio_before = has_audio_stream(video_path)
    video_ref, source_ref = _display_media_ref(video_path, audio_source_path)
    _emit_log(
        f"[audio] mux start: video='{video_ref}', source='{source_ref}', "
        f"codec='{audio_codec}', bitrate='{audio_bitrate or ''}'",
        on_progress,
    )
    _emit_log(
        f"[audio] probe: source_has_audio={src_has_audio}, target_has_audio_before={dst_has_audio_before}",
        on_progress,
    )
    _emit_log(f"[audio] ffmpeg cmd: {_format_cmd(cmd)}", on_progress)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if (
            proc.returncode != 0
            and src_has_audio
            and (audio_codec or "copy").strip().lower() in {"copy", "passthrough"}
            and output_path.suffix.lower() in {".mp4", ".m4v", ".mov"}
        ):
            # MP4 cannot carry every source codec (for example PCM variants). Preserve
            # the video bitstream and transparently fall back to AAC audio rather than
            # returning a silent "successful" production output.
            retry_cmd = list(cmd)
            try:
                audio_arg_idx = retry_cmd.index("-c:a")
                retry_cmd[audio_arg_idx + 1] = "aac"
                retry_cmd[audio_arg_idx + 2:audio_arg_idx + 2] = ["-b:a", str(audio_bitrate or "192k")]
            except (ValueError, IndexError):
                retry_cmd = []
            if retry_cmd:
                _emit_log("[audio] passthrough is incompatible with the output container; retrying as AAC.", on_progress)
                _emit_log(f"[audio] ffmpeg retry: {_format_cmd(retry_cmd)}", on_progress)
                proc = subprocess.run(retry_cmd, capture_output=True, text=True)
                cmd = retry_cmd
        if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1024:
            timing_ok, timing_error = _video_timing_preserved(
                video_path,
                output_path,
                source_frames=video_frames,
                source_duration=video_dur,
            )
            if not timing_ok and clamp_audio_to_video:
                # Some ffmpeg/muxer combinations make -shortest drop the final video
                # packet. Retry without it; a slightly longer audio track is preferable
                # to a missing ending, and callers merge/validate the video stream itself.
                _emit_log(
                    f"[audio] WARNING: -shortest altered video timing ({timing_error}); retrying without it.",
                    on_progress,
                )
                output_path.unlink(missing_ok=True)
                cmd = [part for part in cmd if part != "-shortest"]
                _emit_log(f"[audio] ffmpeg retry: {_format_cmd(cmd)}", on_progress)
                proc = subprocess.run(cmd, capture_output=True, text=True)
                timing_ok = bool(
                    proc.returncode == 0
                    and output_path.exists()
                    and output_path.stat().st_size > 1024
                )
                if timing_ok:
                    timing_ok, timing_error = _video_timing_preserved(
                        video_path,
                        output_path,
                        source_frames=video_frames,
                        source_duration=video_dur,
                    )
            if not timing_ok:
                output_path.unlink(missing_ok=True)
                _emit_log(f"[audio] mux rejected: {timing_error}", on_progress)
                return False, f"Audio mux changed video timing: {timing_error}"
            out_has_audio = has_audio_stream(output_path)
            if want_audio and src_has_audio and not out_has_audio:
                output_path.unlink(missing_ok=True)
                return False, "Audio mux completed without the requested audio stream"
            out_video_info = _probe_video_stream(output_path)
            _emit_log(
                f"[audio] mux success: output='{output_path.name}', has_audio={out_has_audio}",
                on_progress,
            )
            if out_video_info:
                _emit_log(
                    "[audio] mux output video stream: "
                    f"codec={out_video_info.get('codec_name', 'unknown')}, "
                    f"profile={out_video_info.get('profile', 'unknown')}, "
                    f"pix_fmt={out_video_info.get('pix_fmt', 'unknown')}, "
                    f"encoder={out_video_info.get('encoder', 'unknown')}",
                    on_progress,
                )
            return True, ""
        err = (proc.stderr or proc.stdout or "ffmpeg mux failed").strip()
        if err:
            err = err[-1200:]
        _emit_log(f"[audio] mux failed (code={proc.returncode}): {err}", on_progress)
        return False, err
    except Exception as e:
        _emit_log(f"[audio] mux exception: {e}", on_progress)
        return False, str(e)


def ensure_audio_on_video(
    video_path: Path,
    audio_source_path: Path,
    audio_codec: str = "copy",
    audio_bitrate: Optional[str] = None,
    force_replace: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, Optional[Path], str]:
    """
    Ensure `video_path` has (or does not have) audio according to `audio_codec`.

    - If audio_codec is "none": strip audio if present.
    - Otherwise:
      - default behavior: if video already has audio, keep it;
        if it doesn't, attempt to mux from `audio_source_path`.
      - when force_replace=True: replace output audio with the source audio.
        If source has no audio, output audio is stripped to match source.

    Returns: (changed, final_path_or_None, error_message)
    """
    video_path = Path(video_path)
    audio_source_path = Path(audio_source_path)

    codec_norm = (audio_codec or "copy").strip().lower()
    want_audio = codec_norm not in ("none", "no", "off", "disable", "disabled")

    video_ref, source_ref = _display_media_ref(video_path, audio_source_path)
    _emit_log(
        f"[audio] ensure start: video='{video_ref}', source='{source_ref}', "
        f"codec='{codec_norm}', force_replace={bool(force_replace)}",
        on_progress,
    )

    if not _has_ffmpeg():
        _emit_log("[audio] ffmpeg/ffprobe not available; ensure skipped.", on_progress)
        return False, None, "ffmpeg/ffprobe not available"

    video_has_audio = has_audio_stream(video_path)
    src_has_audio = has_audio_stream(audio_source_path)
    _emit_log(
        f"[audio] ensure probe: source_has_audio={src_has_audio}, target_has_audio={video_has_audio}",
        on_progress,
    )
    target_video_info = _probe_video_stream(video_path)
    if target_video_info:
        _emit_log(
            "[audio] ensure target video before mux: "
            f"codec={target_video_info.get('codec_name', 'unknown')}, "
            f"profile={target_video_info.get('profile', 'unknown')}, "
            f"pix_fmt={target_video_info.get('pix_fmt', 'unknown')}, "
            f"encoder={target_video_info.get('encoder', 'unknown')}",
            on_progress,
        )
    else:
        _emit_log("[audio] ensure target video before mux: probe unavailable", on_progress)

    if src_has_audio:
        source_audio_info = _probe_audio_stream(audio_source_path)
        if source_audio_info:
            _emit_log(
                "[audio] ensure source audio before mux: "
                f"codec={source_audio_info.get('codec_name', 'unknown')}, "
                f"profile={source_audio_info.get('profile', 'unknown')}, "
                f"sample_rate={source_audio_info.get('sample_rate', 'unknown')}, "
                f"channels={source_audio_info.get('channels', 'unknown')}, "
                f"layout={source_audio_info.get('channel_layout', 'unknown')}",
                on_progress,
            )
        else:
            _emit_log("[audio] ensure source audio before mux: probe unavailable", on_progress)

    if not want_audio:
        if not video_has_audio:
            _emit_log("[audio] ensure no-op: target already has no audio.", on_progress)
            return False, video_path, ""
        tmp = video_path.with_name(f"{video_path.stem}.__noaudio_tmp{video_path.suffix}")
        ok, err = mux_audio(
            video_path=video_path,
            audio_source_path=audio_source_path,
            output_path=tmp,
            audio_codec="none",
            audio_bitrate=None,
            on_progress=on_progress,
        )
        if not ok:
            tmp.unlink(missing_ok=True)
            return False, None, err
        try:
            os.replace(str(tmp), str(video_path))
        except Exception:
            _emit_log(f"[audio] ensure complete via temp path: {tmp.name}", on_progress)
            return True, tmp, ""
        _emit_log("[audio] ensure complete: stripped audio.", on_progress)
        return True, video_path, ""

    # want audio
    if force_replace:
        # Match source exactly for robust final muxing.
        if not src_has_audio:
            if not video_has_audio:
                _emit_log("[audio] ensure no-op: source and target both have no audio.", on_progress)
                return False, video_path, ""
            tmp = video_path.with_name(f"{video_path.stem}.__noaudio_tmp{video_path.suffix}")
            ok, err = mux_audio(
                video_path=video_path,
                audio_source_path=audio_source_path,
                output_path=tmp,
                audio_codec="none",
                audio_bitrate=None,
                on_progress=on_progress,
            )
            if not ok:
                tmp.unlink(missing_ok=True)
                return False, None, err
            try:
                os.replace(str(tmp), str(video_path))
            except Exception:
                _emit_log(f"[audio] ensure complete via temp path: {tmp.name}", on_progress)
                return True, tmp, ""
            _emit_log("[audio] ensure complete: source has no audio, target audio stripped.", on_progress)
            return True, video_path, ""
    else:
        if video_has_audio:
            _emit_log("[audio] ensure no-op: target already has audio and force_replace is False.", on_progress)
            return False, video_path, ""
        if not src_has_audio:
            _emit_log("[audio] ensure no-op: source has no audio.", on_progress)
            return False, video_path, ""

    tmp = video_path.with_name(f"{video_path.stem}.__audio_tmp{video_path.suffix}")

    ok, err = mux_audio(
        video_path=video_path,
        audio_source_path=audio_source_path,
        output_path=tmp,
        audio_codec=audio_codec,
        audio_bitrate=audio_bitrate,
        on_progress=on_progress,
    )
    if not ok:
        tmp.unlink(missing_ok=True)
        return False, None, err

    try:
        os.replace(str(tmp), str(video_path))
    except Exception:
        # As a fallback, keep the muxed file as a sibling and return it.
        _emit_log(f"[audio] ensure complete via sibling temp: {tmp.name}", on_progress)
        return True, tmp, ""
    final_video_info = _probe_video_stream(video_path)
    if final_video_info:
        _emit_log(
            "[audio] ensure final target video stream: "
            f"codec={final_video_info.get('codec_name', 'unknown')}, "
            f"profile={final_video_info.get('profile', 'unknown')}, "
            f"pix_fmt={final_video_info.get('pix_fmt', 'unknown')}, "
            f"encoder={final_video_info.get('encoder', 'unknown')}",
            on_progress,
        )
    _emit_log("[audio] ensure complete: audio applied to target.", on_progress)
    return True, video_path, ""


def replace_audio_from_original(
    video_path: Path,
    original_input_path: Path,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """
    Robustly replace audio in video_path with audio from original_input_path.

    This is designed to be 100% safe:
    - If original has no audio: silently succeeds (video keeps as-is or gets audio stripped)
    - If video doesn't exist: returns False with error
    - If ffmpeg fails: tries fallback approaches
    - Never throws exceptions to caller

    Args:
        video_path: Path to the video file to update
        original_input_path: Path to the original input file with source audio
        on_progress: Optional progress callback

    Returns:
        (success: bool, error_message: str)
    """
    video_path = Path(video_path)
    original_input_path = Path(original_input_path)

    def _video_duration_seconds(path: Path) -> Optional[float]:
        try:
            proc = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if proc.returncode == 0:
                raw = (proc.stdout or "").strip()
                if raw:
                    v = float(raw)
                    if v > 0:
                        return v
        except Exception:
            pass
        return None

    def _duration_preserved(src_duration: Optional[float], candidate_path: Path, min_ratio: float = 0.95) -> bool:
        if src_duration is None or src_duration <= 0:
            return True
        cand = _video_duration_seconds(candidate_path)
        if cand is None or cand <= 0:
            return False
        return float(cand) >= float(src_duration) * float(min_ratio)

    # Validate paths
    if not video_path.exists():
        return False, f"Video not found: {video_path}"

    if not original_input_path.exists():
        # Original doesn't exist - nothing to do, consider success
        if on_progress:
            on_progress("Original input not found, skipping audio replacement\n")
        return True, ""

    if not _has_ffmpeg():
        return False, "ffmpeg/ffprobe not available"

    # Check if original has audio
    original_has_audio = has_audio_stream(original_input_path)

    if not original_has_audio:
        # Original has no audio - strip audio from output if present, or do nothing
        if on_progress:
            on_progress("Original has no audio, keeping video as-is\n")
        return True, ""

    if on_progress:
        on_progress(f"Replacing audio from original: {original_input_path.name}\n")

    # Create temp output
    tmp = video_path.with_name(f"{video_path.stem}.__audio_replace_tmp{video_path.suffix}")
    source_video_duration = _video_duration_seconds(video_path)

    # Try copy first (fastest, preserves quality)
    cmd_copy = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(original_input_path),
        "-map", "0:v:0",           # Video from first input
        "-map", "1:a:0",           # Audio from second input (original)
        "-c:v", "copy",            # Don't re-encode video
        "-c:a", "copy",            # Don't re-encode audio
        "-movflags", "+faststart",
        str(tmp)
    ]

    try:
        proc = subprocess.run(cmd_copy, capture_output=True, text=True, timeout=600)
        if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 1024:
            if not _duration_preserved(source_video_duration, tmp):
                tmp.unlink(missing_ok=True)
            else:
                # Success with copy - replace original
                try:
                    os.replace(str(tmp), str(video_path))
                    if on_progress:
                        on_progress("Audio replaced successfully (copy mode)\n")
                    return True, ""
                except Exception:
                    # Couldn't replace, but tmp exists
                    if on_progress:
                        on_progress(f"Audio replaced to: {tmp.name}\n")
                    return True, ""
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
    except Exception:
        tmp.unlink(missing_ok=True)

    # Fallback: re-encode audio to AAC (handles codec incompatibilities)
    cmd_aac = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(original_input_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(tmp)
    ]

    try:
        proc = subprocess.run(cmd_aac, capture_output=True, text=True, timeout=600)
        if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 1024:
            if not _duration_preserved(source_video_duration, tmp):
                tmp.unlink(missing_ok=True)
                return False, "Audio replacement aborted: muxed file shortened video unexpectedly"
            # Success with copy - replace original
            try:
                os.replace(str(tmp), str(video_path))
                if on_progress:
                    on_progress("Audio replaced successfully (AAC fallback)\n")
                return True, ""
            except Exception:
                # Couldn't replace, but tmp exists
                if on_progress:
                    on_progress(f"Audio replaced to: {tmp.name}\n")
                return True, ""
        tmp.unlink(missing_ok=True)
        error = (proc.stderr or proc.stdout or "Unknown error")[:200]
        return False, f"Audio replacement failed: {error}"
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return False, "Audio replacement timed out"
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return False, f"Audio replacement error: {str(e)}"
