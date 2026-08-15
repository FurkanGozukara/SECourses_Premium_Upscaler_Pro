import os
import math
import json
import re
import inspect
import shutil
import subprocess
import threading
import tempfile
import time
from statistics import median
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .path_utils import (
    collision_safe_dir,
    collision_safe_path,
    normalize_path,
    resolve_output_location,
    detect_input_type,
    emit_metadata,
    get_media_fps,
    get_media_duration_seconds,
)
from .audio_utils import has_audio_stream, ensure_audio_on_video
from .video_codec_options import build_ffmpeg_video_encode_args

# Try to import PySceneDetect (optional dependency)
try:
    import scenedetect
    PYSCENEDETECT_AVAILABLE = True
except ImportError:
    PYSCENEDETECT_AVAILABLE = False


def _has_scenedetect() -> bool:
    """Check if PySceneDetect is installed"""
    try:
        import scenedetect  # noqa: F401
        return True
    except ImportError:
        return False
    except Exception:
        return False


def detect_scenes(
    video_path: str,
    threshold: float = 27.0,
    min_scene_len: float = 1.0,
    fade_detection: bool = False,
    overlap_sec: float = 0.0,
    on_progress: Optional[Callable[[str], None]] = None,
    on_progress_pct: Optional[Callable[[int], None]] = None,
) -> List[Tuple[float, float]]:
    """
    Detect scenes using PySceneDetect with proper API usage and overlap support.

    Args:
        video_path: Path to video file
        threshold: Content threshold for scene detection (lower = more sensitive)
        min_scene_len: Minimum scene length in seconds
        fade_detection: Enable fade in/out detection
        overlap_sec: Seconds of overlap between chunks (for temporal consistency)
        on_progress: Optional callback for text progress updates
        on_progress_pct: Optional callback for numeric progress (0-100)

    Returns:
        List of (start_seconds, end_seconds) tuples for each scene with overlap applied
    """
    if not _has_scenedetect():
        if on_progress:
            on_progress("⚠️ PySceneDetect not installed, using fallback chunking\n")
        return []

    try:
        # PySceneDetect 0.6+ API (VideoManager is deprecated).
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        def _emit_pct(pct: int) -> None:
            if not on_progress_pct:
                return
            try:
                on_progress_pct(max(0, min(100, int(pct))))
            except Exception:
                pass

        def _timecode_to_frames(value: Any) -> Optional[int]:
            if value is None:
                return None
            for attr in ("get_frames", "frame_num", "frames", "frame"):
                try:
                    raw = getattr(value, attr, None)
                    if raw is None:
                        continue
                    frame_val = raw() if callable(raw) else raw
                    if frame_val is None:
                        continue
                    frame_i = int(frame_val)
                    if frame_i >= 0:
                        return frame_i
                except Exception:
                    continue
            try:
                frame_i = int(value)
                if frame_i >= 0:
                    return frame_i
            except Exception:
                pass
            return None

        if on_progress:
            on_progress(f"Detecting scenes: threshold={threshold}, min_len={min_scene_len}s\n")
        _emit_pct(0)

        video = open_video(video_path)
        fps = float(getattr(video, "frame_rate", None) or 30.0)
        min_scene_frames = max(1, int(round(float(min_scene_len) * fps)))

        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=float(threshold), min_scene_len=min_scene_frames))

        # Optional fade detector (best-effort).
        if fade_detection:
            try:
                from scenedetect.detectors import ThresholdDetector

                scene_manager.add_detector(
                    ThresholdDetector(
                        threshold=12,  # Default fade threshold
                        min_scene_len=min_scene_frames,
                        fade_bias=0.0,
                    )
                )
            except Exception:
                pass

        total_frames = _timecode_to_frames(getattr(video, "duration", None))
        if (not total_frames or total_frames <= 0) and fps > 0:
            try:
                duration_guess = get_media_duration_seconds(video_path)
                if duration_guess and duration_guess > 0:
                    total_frames = max(1, int(round(float(duration_guess) * float(fps))))
            except Exception:
                total_frames = None

        poller_stop = threading.Event()
        poller_thread: Optional[threading.Thread] = None
        last_pct = -1
        frames_seen = 0

        def _publish_frame_progress(frame_value: Any, force: bool = False) -> None:
            nonlocal last_pct
            if not on_progress_pct or not total_frames or total_frames <= 0:
                return
            frame_i = _timecode_to_frames(frame_value)
            if frame_i is None:
                return
            pct = int((float(frame_i) / float(total_frames)) * 100.0)
            pct = max(0, min(99, pct))
            if force or pct > last_pct:
                last_pct = pct
                _emit_pct(pct)

        def _publish_position_progress(force: bool = False) -> None:
            _publish_frame_progress(getattr(video, "position", None), force=force)

        # Fallback for backends that do not expose `position` updates during detection.
        # We wrap `read()` and track decoded frame count directly.
        if on_progress_pct and total_frames and total_frames > 0:
            original_read = getattr(video, "read", None)
            if callable(original_read):
                try:
                    def _read_with_progress(*args, **kwargs):
                        nonlocal frames_seen
                        frame_data = original_read(*args, **kwargs)
                        has_frame = frame_data is not None
                        if isinstance(frame_data, tuple) and frame_data:
                            first = frame_data[0]
                            if isinstance(first, bool):
                                has_frame = bool(first)
                            else:
                                has_frame = first is not None
                        if has_frame:
                            frames_seen += 1
                            _publish_frame_progress(frames_seen)
                        return frame_data

                    setattr(video, "read", _read_with_progress)
                except Exception:
                    pass

        if on_progress_pct and total_frames and total_frames > 0:

            def _progress_poller() -> None:
                while not poller_stop.wait(0.20):
                    _publish_position_progress()

            poller_thread = threading.Thread(target=_progress_poller, daemon=True)
            poller_thread.start()

        detect_kwargs: Dict[str, Any] = {"video": video, "show_progress": False}
        if on_progress_pct:

            def _detect_callback(*_args, **_kwargs) -> None:
                for value in _args:
                    frame_i = _timecode_to_frames(value)
                    if frame_i is not None:
                        _publish_frame_progress(frame_i, force=True)
                        return
                for value in _kwargs.values():
                    frame_i = _timecode_to_frames(value)
                    if frame_i is not None:
                        _publish_frame_progress(frame_i, force=True)
                        return
                _publish_position_progress(force=True)

            detect_kwargs["callback"] = _detect_callback

        try:
            scene_manager.detect_scenes(**detect_kwargs)
        except TypeError as callback_exc:
            # Some PySceneDetect versions do not support `callback=`.
            if "callback" not in str(callback_exc).lower():
                raise
            detect_kwargs.pop("callback", None)
            scene_manager.detect_scenes(**detect_kwargs)
        finally:
            poller_stop.set()
            if poller_thread and poller_thread.is_alive():
                poller_thread.join(timeout=0.5)

        _publish_position_progress(force=True)
        _emit_pct(100)
        scene_list = scene_manager.get_scene_list(start_in_scene=True)

        ranges: List[Tuple[float, float]] = []
        for start_tc, end_tc in scene_list:
            ranges.append((float(start_tc.get_seconds()), float(end_tc.get_seconds())))

        # If we somehow end up with an empty list, treat the whole video as one scene.
        if not ranges:
            try:
                duration = get_media_duration_seconds(video_path)
                if duration and duration > 0:
                    ranges = [(0.0, float(duration))]
            except Exception:
                pass

        # Overlap is generally not desirable for scene cuts; only apply if explicitly requested.
        if overlap_sec and overlap_sec > 0 and ranges:
            try:
                duration = get_media_duration_seconds(video_path)
                if duration and duration > 0:
                    ranges = apply_overlap_to_scenes(ranges, float(overlap_sec), float(duration))
            except Exception:
                pass

        if on_progress:
            on_progress(f"✅ Detected {len(ranges)} scenes\n")

        return ranges

    except ImportError as e:
        if on_progress:
            on_progress(f"⚠️ PySceneDetect import error: {e}, using fallback\n")
        return []
    except Exception as e:
        if on_progress:
            on_progress(f"⚠️ Scene detection error: {e}, using fallback\n")
        return []


def apply_overlap_to_scenes(
    scenes: List[Tuple[float, float]], 
    overlap_sec: float,
    total_duration: float
) -> List[Tuple[float, float]]:
    """
    Apply overlap to scene boundaries for temporal consistency.
    
    Args:
        scenes: List of (start, end) tuples without overlap
        overlap_sec: Seconds of overlap to add
        total_duration: Total video duration to clamp overlaps
        
    Returns:
        List of (start, end) tuples with overlap applied
    """
    if overlap_sec <= 0 or not scenes:
        return scenes
    
    overlapped = []
    for i, (start, end) in enumerate(scenes):
        # Extend start backwards (except first chunk)
        if i > 0:
            new_start = max(0, start - overlap_sec / 2)
        else:
            new_start = start
        
        # Extend end forwards (except last chunk)
        if i < len(scenes) - 1:
            new_end = min(total_duration, end + overlap_sec / 2)
        else:
            new_end = end
        
        overlapped.append((new_start, new_end))
    
    return overlapped


def fallback_scenes(video_path: str, chunk_seconds: float = 60.0, overlap_seconds: float = 0.0) -> List[Tuple[float, float]]:
    """
    Fallback to fixed-length segments using ffprobe duration with optional overlap.
    
    Args:
        video_path: Path to video file
        chunk_seconds: Length of each chunk in seconds
        overlap_seconds: Overlap between chunks in seconds
        
    Returns:
        List of (start_sec, end_sec) tuples with overlap applied
    """
    from .path_utils import get_media_duration_seconds
    
    try:
        duration = get_media_duration_seconds(video_path)
        if not duration or duration <= 0:
            # Try ffprobe as fallback
            proc = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            duration = float(proc.stdout.strip())
    except Exception:
        # If all fails, use default
        duration = chunk_seconds * 2
    
    scenes = []
    start = 0.0
    
    # First pass: create chunks without overlap
    while start < duration:
        end = min(start + chunk_seconds, duration)
        scenes.append((start, end))
        start += chunk_seconds
        
        # Avoid tiny last chunk
        if start < duration and (duration - start) < (chunk_seconds * 0.3):
            scenes[-1] = (scenes[-1][0], duration)
            break
    
    # Second pass: apply overlap
    if overlap_seconds > 0:
        scenes = apply_overlap_to_scenes(scenes, overlap_seconds, duration)
    
    return scenes


def split_video(
    video_path: str,
    scenes: List[Tuple[float, float]],
    work_dir: Path,
    precise: bool = True,
    preserve_quality: bool = True,
    include_audio: bool = True,
    on_progress: Optional[Callable[[str], None]] = None,
) -> List[Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: List[Path] = []

    if shutil.which("ffmpeg") is None:
        if on_progress:
            on_progress("⚠️ ffmpeg not found in PATH; skipping chunk splitting.\n")
        return [Path(video_path)]

    # If PySceneDetect says "1 scene" and it spans the whole file, don't physically split.
    # This avoids unnecessary remux/transcode and improves robustness for short clips.
    if len(scenes) == 1:
        try:
            from .path_utils import get_media_duration_seconds

            total_dur = float(get_media_duration_seconds(video_path) or 0.0)
            s0, e0 = float(scenes[0][0]), float(scenes[0][1])
            fps_guess = float(get_media_fps(video_path) or 30.0)
            tol = max(0.02, 1.0 / max(1.0, fps_guess))  # within ~1 frame
            if total_dur > 0 and abs(s0 - 0.0) <= tol and abs(e0 - total_dur) <= tol:
                return [Path(video_path)]
        except Exception:
            pass

    def _is_decodable(p: Path) -> bool:
        try:
            if not p.exists() or p.stat().st_size < 1024:
                return False
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                return False
            ok, frame = cap.read()
            cap.release()
            return bool(ok) and frame is not None
        except Exception:
            return False

    def _probe_pix_fmt(src: str) -> Optional[str]:
        try:
            proc = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=pix_fmt",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    src,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            pix = (proc.stdout or "").strip()
            return pix if pix else None
        except Exception:
            return None

    def _to_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            raw = str(value).strip()
            if not raw or raw.lower() in {"n/a", "nan"}:
                return None
            return float(raw)
        except Exception:
            return None

    def _probe_av_timing(path: Path) -> Dict[str, Optional[float]]:
        timing: Dict[str, Optional[float]] = {
            "video_start": None,
            "video_duration": None,
            "audio_start": None,
            "audio_duration": None,
        }
        try:
            proc = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=codec_type,start_time,duration",
                    "-of",
                    "json",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode != 0:
                return timing
            payload = json.loads(proc.stdout or "{}")
            streams = payload.get("streams") or []
            for st in streams:
                ctype = str(st.get("codec_type") or "").strip().lower()
                if ctype == "video" and timing["video_duration"] is None:
                    timing["video_start"] = _to_float(st.get("start_time"))
                    timing["video_duration"] = _to_float(st.get("duration"))
                elif ctype == "audio" and timing["audio_duration"] is None:
                    timing["audio_start"] = _to_float(st.get("start_time"))
                    timing["audio_duration"] = _to_float(st.get("duration"))
        except Exception:
            pass
        return timing

    def _has_reasonable_av_timing(path: Path) -> bool:
        """
        Validate basic A/V alignment for split chunks.
        Prevent accepting chunks where copied audio drifts far from video.
        """
        timing = _probe_av_timing(path)
        vs = timing.get("video_start")
        vd = timing.get("video_duration")
        as_ = timing.get("audio_start")
        ad = timing.get("audio_duration")
        if as_ is not None and vs is not None:
            # AAC priming often causes tiny offsets; allow a small tolerance.
            if abs(as_ - vs) > 0.25:
                return False
        if ad is not None and vd is not None:
            # Audio can be slightly longer due to codec frame boundaries.
            if ad > (vd + max(0.25, 0.05 * max(0.0, vd))):
                return False
            # Reject severe audio truncation.
            if ad < max(0.0, vd - 1.0):
                return False
        return True

    # ------------------------------------------------------------------
    # Boundary normalization (frame-exact, contiguous chunks).
    #
    # Scene boundaries arrive as floats that can carry sub-frame noise:
    #   * PySceneDetect >= 0.7 rounds timecodes to microseconds
    #     (e.g. frame 256 @ 23.976 fps -> 10.677333 s instead of 10.6773333...),
    #   * fixed-second chunking rarely lands on a frame boundary at NTSC rates.
    # The previous floor()/ceil() alignment rounded the END of chunk N up and the
    # START of chunk N+1 down, so every boundary that was not exactly on the frame
    # grid produced a one-frame overlap -> one duplicated frame per boundary in
    # the merged output, and (after the final audio mux) a truncated ending plus
    # progressive A/V drift.
    #
    # New rule: each boundary is snapped ONCE (round-to-nearest frame when the
    # frame rate is known) and consecutive chunks share the identical boundary
    # timestamp. Timestamps are kept as integer microseconds so ffmpeg -ss/-t
    # parsing (microsecond precision) cannot re-introduce drift.
    # ------------------------------------------------------------------
    fps_for_align = 0.0
    try:
        fps_for_align = float(get_media_fps(video_path) or 0.0)
    except Exception:
        fps_for_align = 0.0
    if not (fps_for_align > 0 and math.isfinite(fps_for_align)):
        fps_for_align = 0.0

    # Source frame-rate facts used to make the cut independent of the ffmpeg build:
    #  * exact rational (e.g. "24000/1001") -> forced on the re-encoded chunk with `-r`
    #    for CFR sources, so every chunk gets the source's timing and a complete last
    #    frame regardless of muxer quirks (ffmpeg 7.1 drops the last frame's duration
    #    otherwise);
    #  * the video stream's start_time -> absolute timestamps for `trim`.
    src_r_fps_str = ""
    src_avg_fps: Optional[float] = None
    src_is_cfr = False
    src_video_start = 0.0
    try:
        proc_src = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,avg_frame_rate,start_time",
                "-of", "json", str(video_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        st0 = (json.loads(proc_src.stdout or "{}").get("streams") or [{}])[0]
        src_r_fps_str = str(st0.get("r_frame_rate") or "").strip()
        src_avg_fps = _parse_fraction_to_float(st0.get("avg_frame_rate"))
        r_val = _parse_fraction_to_float(src_r_fps_str)
        if r_val and src_avg_fps:
            src_is_cfr = abs(r_val - src_avg_fps) <= max(1e-6, 0.002 * r_val)
        try:
            src_video_start = float(st0.get("start_time") or 0.0)
            if not math.isfinite(src_video_start) or src_video_start < 0:
                src_video_start = 0.0
        except Exception:
            src_video_start = 0.0
    except Exception:
        pass
    if not src_r_fps_str or _parse_fraction_to_float(src_r_fps_str) is None:
        src_r_fps_str = ""
        src_is_cfr = False

    def _snap_us(t: float) -> int:
        t = max(0.0, float(t))
        if fps_for_align > 0:
            frame_i = int(round(t * fps_for_align))
            return int(round(frame_i * 1_000_000.0 / fps_for_align))
        return int(round(t * 1_000_000.0))

    def _frame_index_of_us(t_us: int) -> Optional[int]:
        if fps_for_align <= 0:
            return None
        return int(round(t_us * fps_for_align / 1_000_000.0))

    half_frame_sec = (0.5 / fps_for_align) if fps_for_align > 0 else 0.0005
    normalized_us: List[Tuple[int, int]] = []
    prev_end_us: Optional[int] = None
    for start, end in scenes:
        try:
            start_f = float(start)
            end_f = float(end)
        except Exception:
            continue
        if not (math.isfinite(start_f) and math.isfinite(end_f)):
            continue
        s_us = _snap_us(start_f)
        e_us = _snap_us(end_f)
        # Contiguous scenes must share the exact same boundary timestamp.
        if prev_end_us is not None and abs(start_f - (prev_end_us / 1_000_000.0)) <= (half_frame_sec + 1e-6):
            s_us = prev_end_us
        if e_us <= s_us:
            e_us = s_us + (int(round(1_000_000.0 / fps_for_align)) if fps_for_align > 0 else 1000)
        normalized_us.append((s_us, e_us))
        prev_end_us = e_us

    if not normalized_us:
        return [Path(video_path)]

    src_has_audio = has_audio_stream(Path(video_path)) if include_audio else False
    src_pix_fmt = _probe_pix_fmt(video_path) if preserve_quality else None
    manifest_entries: List[Dict[str, Any]] = []
    # Seek this far before the chunk start (lands on an earlier keyframe), then cut by
    # absolute timestamps with the trim filters. Bound the amount of input read with an
    # input-side -t so early chunks of long videos do not decode to EOF.
    SEEK_LEAD_SEC = 1.0
    READ_MARGIN_SEC = 1.0
    last_ffmpeg_error = ""
    for idx, (s_us, e_us) in enumerate(normalized_us, 1):
        out = work_dir / f"chunk_{idx:04d}.mp4"
        dur_us = int(e_us - s_us)
        if dur_us <= 0:
            continue
        start_rel = s_us / 1_000_000.0
        duration = dur_us / 1_000_000.0
        start_str = f"{start_rel:.6f}"
        dur_str = f"{duration:.6f}"
        start_abs_str = f"{start_rel + src_video_start:.6f}"
        end_abs_str = f"{start_rel + duration + src_video_start:.6f}"
        seek_str = f"{max(0.0, start_rel - SEEK_LEAD_SEC):.6f}"
        read_limit_str = f"{duration + SEEK_LEAD_SEC + READ_MARGIN_SEC:.6f}"
        start_frame_i = _frame_index_of_us(s_us)
        end_frame_i = _frame_index_of_us(e_us)
        expected_frames: Optional[int] = None
        if src_is_cfr and start_frame_i is not None and end_frame_i is not None and end_frame_i > start_frame_i:
            expected_frames = int(end_frame_i - start_frame_i)
        split_mode_used = "unknown"

        def _run_ffmpeg(cmd: List[str]) -> int:
            nonlocal last_ffmpeg_error
            try:
                proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="ignore")
            except Exception as exc:
                last_ffmpeg_error = str(exc)
                return 1
            if proc.returncode != 0:
                tail = [ln.strip() for ln in (proc.stderr or "").splitlines() if ln.strip()]
                last_ffmpeg_error = " | ".join(tail[-3:])
            return int(proc.returncode)

        def _unlink_out() -> None:
            try:
                out.unlink(missing_ok=True)
            except Exception:
                pass

        def _split_copy() -> None:
            # IMPORTANT: -ss must be BEFORE -i when stream-copying or ffmpeg can output empty files.
            # NOTE: stream copy is keyframe-limited: ffmpeg keeps everything from the last keyframe
            # before -ss, so the chunk may start early. Exactness is verified afterwards.
            cmd = [
                "ffmpeg", "-y", "-ss", start_str, "-i", video_path, "-t", dur_str,
                "-c", "copy", "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(out),
            ]
            _run_ffmpeg(cmd)

        def _split_copy_video_only() -> None:
            # Stream-copy video only. Useful when audio codecs cannot be muxed into MP4.
            cmd = [
                "ffmpeg", "-y", "-ss", start_str, "-i", video_path, "-t", dur_str,
                "-map", "0:v:0", "-c:v", "copy", "-an",
                "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(out),
            ]
            _run_ffmpeg(cmd)

        def _split_copy_aac_audio() -> None:
            # Stream-copy video but re-encode audio to AAC for MP4 compatibility.
            cmd = [
                "ffmpeg", "-y", "-ss", start_str, "-i", video_path, "-t", dur_str,
                "-map", "0:v:0", "-map", "0:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(out),
            ]
            _run_ffmpeg(cmd)

        def _lossless_video_args(encoder: str) -> List[str]:
            if encoder == "libx265":
                return ["-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "lossless=1:log-level=error"]
            return ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0"]

        def _split_precise_lossless(pix_fmt: Optional[str], with_audio: bool, encoder: str = "libx264") -> int:
            """
            Frame-accurate cut that does not depend on how a given ffmpeg build implements
            `-ss`/`-t` accuracy or edit lists:
              * seek a little BEFORE the chunk (fast, keyframe-bound), keep original timestamps
                (-copyts) and select the exact [start, end) window with trim/atrim on absolute
                timestamps;
              * bound the input read with an input-side -t so we never decode to EOF;
              * for CFR sources force the exact source frame rate on the output (-r) so the
                chunk gets complete, uniform frame timing (some builds otherwise drop the last
                frame's duration -> one lost frame + a wrong avg_frame_rate);
              * lossless x264 (or x265) to preserve input quality; chunk audio (preview-only,
                the final output re-muxes the original audio) is re-encoded to AAC and trimmed
                with the same window.
            """
            cmd = ["ffmpeg", "-y", "-ss", seek_str, "-t", read_limit_str, "-copyts", "-i", video_path, "-map", "0:v:0"]
            if with_audio:
                cmd += ["-map", "0:a?"]
            cmd += ["-vf", f"trim=start={start_abs_str}:end={end_abs_str},setpts=PTS-STARTPTS"]
            if with_audio:
                cmd += ["-af", f"atrim=start={start_abs_str}:end={end_abs_str},asetpts=PTS-STARTPTS"]
            if src_is_cfr and src_r_fps_str:
                cmd += ["-r", src_r_fps_str]
            cmd += _lossless_video_args(encoder)
            if pix_fmt:
                cmd += ["-pix_fmt", pix_fmt]
            if with_audio:
                cmd += ["-c:a", "aac", "-b:a", "192k"]
            else:
                cmd += ["-an"]
            cmd += ["-movflags", "+faststart", str(out)]
            return _run_ffmpeg(cmd)

        def _ok_with_audio() -> bool:
            if not _is_decodable(out):
                return False
            if src_has_audio and not has_audio_stream(out):
                return False
            if src_has_audio and not _has_reasonable_av_timing(out):
                try:
                    if on_progress:
                        on_progress(
                            f"WARN: Split chunk {idx} has A/V timing drift; "
                            "retrying alternate split path.\n"
                        )
                except Exception:
                    pass
                return False
            return True

        def _chunk_is_exact() -> Tuple[bool, str]:
            """
            Verify the produced chunk covers exactly the requested range: decodable frame count
            for CFR sources (exact), duration otherwise (tolerance below one frame).
            """
            actual_dur = _probe_video_stream_duration(out)
            if actual_dur is None or actual_dur <= 0:
                return False, "duration probe failed"
            tol = (0.75 / fps_for_align) if fps_for_align > 0 else 0.03
            if expected_frames is not None:
                n_frames = _probe_chunk_nb_frames(out)
                if n_frames is not None and int(n_frames) != int(expected_frames):
                    return False, f"{int(n_frames)} decodable frames vs expected {int(expected_frames)}"
                return True, ""
            if abs(float(actual_dur) - float(duration)) > tol:
                return False, (
                    f"video duration {float(actual_dur):.3f}s vs requested {float(duration):.3f}s"
                )
            return True, ""

        def _attempt_precise_chain() -> bool:
            """
            Frame-accurate (lossless re-encode) split. Returns True when `out` is a decodable,
            frame-accurate video chunk. Chunk audio is preview-only (the final output re-muxes
            audio from the original input), so audio problems must never degrade the VIDEO
            split to a keyframe-limited stream copy.
            """
            nonlocal split_mode_used
            split_mode_used = "lossless"
            want_audio = bool(include_audio and src_has_audio)
            for encoder in ("libx264", "libx265"):
                for pix in ((src_pix_fmt, None) if src_pix_fmt else (None,)):
                    _unlink_out()
                    rc = _split_precise_lossless(pix, want_audio, encoder)
                    if want_audio and _ok_with_audio():
                        return True
                    if not want_audio and _is_decodable(out):
                        return True
                    if want_audio and _is_decodable(out):
                        # Video is frame-accurate; only chunk-audio validation failed
                        # (e.g. source audio shorter than video). Keep the accurate video.
                        try:
                            if on_progress:
                                on_progress(
                                    f"WARN: Split chunk {idx}: chunk audio could not be validated; "
                                    "keeping frame-accurate video (chunk audio is preview-only).\n"
                                )
                        except Exception:
                            pass
                        return True
                    if want_audio:
                        # Audio mapping/encoding may be the culprit: retry video-only.
                        _unlink_out()
                        rc = _split_precise_lossless(pix, False, encoder)
                        if _is_decodable(out):
                            try:
                                if on_progress:
                                    on_progress(
                                        f"WARN: Split chunk {idx}: chunk audio could not be encoded "
                                        f"({last_ffmpeg_error or 'unknown error'}); keeping video-only chunk.\n"
                                    )
                            except Exception:
                                pass
                            return True
                    if rc != 0 and last_ffmpeg_error and on_progress:
                        try:
                            on_progress(f"WARN: Split chunk {idx}: {encoder} lossless cut failed: {last_ffmpeg_error}\n")
                        except Exception:
                            pass
            return False

        def _attempt_copy_chain(require_exact: bool) -> bool:
            """
            Fast stream-copy split (bit-exact video, keyframe-limited). When `require_exact` is
            True the result is only accepted if it covers exactly the requested range.
            """
            nonlocal split_mode_used
            split_mode_used = "stream-copy"
            if include_audio:
                _unlink_out()
                _split_copy()
                if not _ok_with_audio():
                    _unlink_out()
                    _split_copy_aac_audio()
                if not _ok_with_audio():
                    if not _is_decodable(out):
                        _unlink_out()
                        _split_copy_video_only()
                    if not _is_decodable(out):
                        return False
            else:
                _unlink_out()
                _split_copy_video_only()
                if not _is_decodable(out):
                    return False
            if require_exact:
                exact, why = _chunk_is_exact()
                if not exact:
                    try:
                        if on_progress:
                            on_progress(
                                f"Chunk {idx}: fast stream-copy split is not frame-accurate ({why}); "
                                "re-encoding this chunk losslessly to keep A/V sync.\n"
                            )
                    except Exception:
                        pass
                    return False
            return True

        # Strategy:
        # - precise=True: lossless re-encode (frame-accurate). Stream copy is only a last resort
        #   when the re-encode cannot produce a decodable file at all.
        # - precise=False: stream copy (bit-exact) when it happens to be frame-exact (keyframe at the
        #   boundary), otherwise fall back to the lossless re-encode for that chunk.
        _unlink_out()
        if on_progress:
            mode = "precise-lossless" if precise else "stream-copy"
            on_progress(f"Splitting chunk {idx}/{len(scenes)} ({mode})...\n")

        ok_chunk = False
        if precise:
            ok_chunk = _attempt_precise_chain()
            if not ok_chunk:
                if on_progress:
                    on_progress(
                        f"WARN: Split chunk {idx}: lossless re-encode failed "
                        f"({last_ffmpeg_error or 'no decodable output'}); trying stream copy as last resort.\n"
                    )
                ok_chunk = _attempt_copy_chain(require_exact=False)
                if ok_chunk:
                    exact, why = _chunk_is_exact()
                    if not exact and on_progress:
                        on_progress(
                            f"WARN: Split chunk {idx} used a keyframe-limited stream copy as last resort "
                            f"({why}). Chunk coverage will be verified before processing.\n"
                        )
        else:
            ok_chunk = _attempt_copy_chain(require_exact=True)
            if not ok_chunk:
                ok_chunk = _attempt_precise_chain()

        if ok_chunk and split_mode_used == "lossless":
            exact, why = _chunk_is_exact()
            if not exact and on_progress:
                on_progress(
                    f"WARN: Split chunk {idx}: lossless cut is not exact ({why}). "
                    "Chunk coverage will be verified before processing.\n"
                )

        if ok_chunk and _is_decodable(out):
            chunk_paths.append(out)
            manifest_entries.append(
                {
                    "index": int(idx),
                    "file": out.name,
                    "start_us": int(s_us),
                    "end_us": int(e_us),
                    "start_frame": start_frame_i,
                    "end_frame": end_frame_i,
                    "expected_frames": expected_frames,
                    "split_mode": split_mode_used,
                }
            )

    # Safety: never return a partial set of chunks. If splitting failed for any scene,
    # fall back to processing the original video as a single chunk.
    if len(chunk_paths) != len(normalized_us):
        if on_progress:
            on_progress("⚠️ Split produced an incomplete set of chunks; falling back to single-pass input.\n")
        return [Path(video_path)]

    # Persist the split manifest so the caller can verify coverage and compute exact overlaps.
    try:
        manifest = {
            "source": str(video_path),
            "fps_for_align": float(fps_for_align) if fps_for_align > 0 else None,
            "source_fps_rational": src_r_fps_str or None,
            "source_avg_fps": float(src_avg_fps) if src_avg_fps else None,
            "source_is_cfr": bool(src_is_cfr),
            "source_video_start": float(src_video_start),
            "precise_requested": bool(precise),
            "chunks": manifest_entries,
        }
        with (work_dir / SPLIT_MANIFEST_NAME).open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception:
        pass

    return chunk_paths


def blend_overlapping_frames_opencv(
    prev_frames: np.ndarray,
    cur_frames: np.ndarray,
    overlap_frames: int
) -> np.ndarray:
    """
    Blend overlapping frames using smooth crossfade (OpenCV implementation).
    
    Args:
        prev_frames: Last `overlap_frames` from previous chunk [N, H, W, C]
        cur_frames: First `overlap_frames` from current chunk [N, H, W, C]
        overlap_frames: Number of frames to blend
        
    Returns:
        Blended frames [overlap_frames, H, W, C]
    """
    if overlap_frames <= 0:
        return cur_frames
    
    if overlap_frames >= 3:
        # Smooth Hann window for better blending
        t = np.linspace(0.0, 1.0, overlap_frames)
        blend_start = 1.0 / 3.0
        blend_end = 2.0 / 3.0
        u = np.clip((t - blend_start) / (blend_end - blend_start), 0.0, 1.0)
        w_prev = 0.5 + 0.5 * np.cos(np.pi * u)  # Hann window
    else:
        # Linear blend for short overlaps
        w_prev = np.linspace(1.0, 0.0, overlap_frames)
    
    # Reshape weights for broadcasting [N, 1, 1, 1]
    w_prev = w_prev.reshape(-1, 1, 1, 1)
    w_cur = 1.0 - w_prev
    
    # Blend frames
    blended = prev_frames.astype(np.float32) * w_prev + cur_frames.astype(np.float32) * w_cur
    
    return blended.astype(prev_frames.dtype)


def _sum_chunk_durations(chunk_paths: List[Path]) -> Optional[float]:
    """
    Sum durations for all chunk files.
    Returns None when duration probing is incomplete, so callers can skip plausibility checks.
    """
    total = 0.0
    for p in chunk_paths:
        dur = _probe_video_stream_duration(Path(p))
        if dur is None or dur <= 0:
            return None
        total += float(dur)
    return total if total > 0 else None


def _duration_is_plausible(
    output_path: Path,
    expected_duration: Optional[float],
    min_ratio: float = 0.85,
    max_ratio: float = 1.15,
    slack_sec: float = 0.35,
) -> bool:
    if expected_duration is None or expected_duration <= 0:
        return output_path.exists() and output_path.stat().st_size > 1024
    actual = _probe_video_stream_duration(Path(output_path))
    if actual is None or actual <= 0:
        return False
    exp = float(expected_duration)
    min_allowed = exp * float(min_ratio)
    max_allowed = exp * float(max_ratio) + max(0.0, float(slack_sec))
    return min_allowed <= float(actual) <= max_allowed


def _probe_video_stream_duration(path: Path) -> Optional[float]:
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
            timeout=10,
        )
        if proc.returncode == 0:
            raw = (proc.stdout or "").strip()
            if raw:
                val = float(raw)
                if val > 0:
                    return val
    except Exception:
        pass
    try:
        val2 = get_media_duration_seconds(str(path))
        if val2 and val2 > 0:
            return float(val2)
    except Exception:
        pass
    return None


SPLIT_MANIFEST_NAME = "chunks_manifest.json"


def _probe_decodable_frames(path: Path) -> Optional[int]:
    """
    Number of video frames a decoder will actually deliver: video packets minus the ones the
    demuxer flags as discard (e.g. samples cut off by an edit list). Container `nb_frames`
    can over-count those on some ffmpeg builds (7.1 writes a discard-flagged trailing sample),
    which is exactly the kind of off-by-one that accumulates into A/V drift over many chunks.
    """
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "packet=flags", "-of", "csv=p=0", str(path),
            ],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            return None
        n = 0
        seen = False
        for line in (proc.stdout or "").splitlines():
            flags = line.strip().strip(",")
            if not flags:
                continue
            seen = True
            if "D" in flags:
                continue
            n += 1
        return n if seen and n > 0 else None
    except Exception:
        return None


def _probe_chunk_nb_frames(path: Path) -> Optional[int]:
    """
    Frame count of a chunk video: decodable frames first (packet scan, no decode), then
    container metadata as fallback.
    """
    decodable = _probe_decodable_frames(Path(path))
    if decodable is not None and decodable > 0:
        return int(decodable)
    for entries, flag in (("stream=nb_frames", None), ("stream=nb_read_packets", "-count_packets")):
        try:
            cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
            if flag:
                cmd.append(flag)
            cmd += ["-show_entries", entries, "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if proc.returncode == 0:
                raw = (proc.stdout or "").strip()
                if raw.isdigit() and int(raw) > 0:
                    return int(raw)
        except Exception:
            continue
    return None


def _probe_source_frame_count(path: Path) -> Tuple[Optional[int], str]:
    """
    Frame count of the source video: (count, method). method is "nb_frames" (exact, MP4/MOV),
    "packets" (fast, one packet per frame for normal video codecs) or "" when unavailable.
    """
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        raw = (proc.stdout or "").strip() if proc.returncode == 0 else ""
        if raw.isdigit() and int(raw) > 0:
            return int(raw), "nb_frames"
    except Exception:
        pass
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                "-show_entries", "stream=nb_read_packets",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            capture_output=True, text=True, timeout=600,
        )
        raw = (proc.stdout or "").strip() if proc.returncode == 0 else ""
        if raw.isdigit() and int(raw) > 0:
            return int(raw), "packets"
    except Exception:
        pass
    return None, ""


def _probe_frame_rates(path: Path) -> Tuple[Optional[float], Optional[float]]:
    """Return (r_frame_rate, avg_frame_rate) as floats."""
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,avg_frame_rate",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            return None, None
        st = (json.loads(proc.stdout or "{}").get("streams") or [{}])[0]
        return _parse_fraction_to_float(st.get("r_frame_rate")), _parse_fraction_to_float(st.get("avg_frame_rate"))
    except Exception:
        return None, None


def _load_split_manifest(work_dir: Path) -> Optional[Dict[str, Any]]:
    try:
        p = Path(work_dir) / SPLIT_MANIFEST_NAME
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("chunks"), list):
            return data
    except Exception:
        pass
    return None


def _manifest_overlap_frames(manifest: Optional[Dict[str, Any]], chunk_count: int) -> Optional[List[int]]:
    """
    Exact number of overlapping frames at every chunk boundary (len == chunk_count - 1),
    derived from the split manifest. None when unavailable.
    """
    if not manifest:
        return None
    chunks = manifest.get("chunks") or []
    if len(chunks) != chunk_count or chunk_count < 2:
        return None
    overlaps: List[int] = []
    for prev, cur in zip(chunks, chunks[1:]):
        try:
            pe = prev.get("end_frame")
            cs = cur.get("start_frame")
            if pe is None or cs is None:
                return None
            overlaps.append(max(0, int(pe) - int(cs)))
        except Exception:
            return None
    return overlaps


# Models whose chunk output was verified to contain exactly as many frames as the chunk
# input at exactly the source frame rate. For these a per-chunk mismatch is a hard error.
# Other models are checked the same way but only warned about (their frame handling was
# not verified here), and the final merged-duration check still reports any drift.
# RIFE changes the frame count/fps by design and is checked by duration only.
_STRICT_TIMING_MODELS = {"flashvsr", "gan", "rtx"}
_FRAME_PRESERVING_MODELS = _STRICT_TIMING_MODELS  # backward-compatible alias


def _processed_chunk_matches_input(
    processed_path: Path,
    input_chunk_path: Path,
    model_type: str = "",
    expected_fps: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Verify that a model's output for one chunk has the same duration (and, for
    frame-preserving models, the same frame count) as the chunk it was made from.

    Any per-chunk drift here accumulates linearly over the run: +2 frames per chunk on a
    600-chunk movie is ~50 s of audio/video desync in the merged output. Returns (ok, detail).
    """
    try:
        inp = Path(input_chunk_path)
        outp = Path(processed_path)
        if not (inp.exists() and inp.is_file() and outp.exists() and outp.is_file()):
            return True, "skipped (missing file)"
        in_dur = _probe_video_stream_duration(inp)
        out_dur = _probe_video_stream_duration(outp)
        if in_dur is None or out_dur is None or in_dur <= 0 or out_dur <= 0:
            return True, "skipped (duration probe unavailable)"
        fps = 0.0
        try:
            fps = float(get_media_fps(str(inp)) or 0.0)
        except Exception:
            fps = 0.0
        frame_dur = (1.0 / fps) if fps > 0 else (1.0 / 30.0)
        tol = max(1.5 * frame_dur, 0.02)
        delta = float(out_dur) - float(in_dur)
        detail = f"input {float(in_dur):.3f}s vs output {float(out_dur):.3f}s (delta {delta:+.3f}s, tolerance {tol:.3f}s)"
        if abs(delta) > tol:
            return False, detail
        # For CFR sources the output must carry the source frame rate exactly: a chunk written
        # at e.g. 25.098 or 24.49 fps instead of 25 passes a per-chunk duration tolerance yet
        # accumulates into seconds of drift over hundreds of chunks.
        model_key = str(model_type or "").strip().lower()
        strict = model_key in _STRICT_TIMING_MODELS
        if expected_fps and expected_fps > 0 and model_key != "rife":
            out_r, out_avg = _probe_frame_rates(outp)
            out_fps = out_r or out_avg
            fps_rel_tol = 1e-4 if strict else 5e-3
            if out_fps and abs(float(out_fps) - float(expected_fps)) > fps_rel_tol * float(expected_fps):
                return False, f"output frame rate {float(out_fps):.6f} vs source {float(expected_fps):.6f} ({detail})"
            detail += f"; fps {float(out_fps) if out_fps else 0.0:.6f}"
        if model_key != "rife":
            n_in = _probe_chunk_nb_frames(inp)
            n_out = _probe_chunk_nb_frames(outp)
            if n_in is not None and n_out is not None and int(n_in) != int(n_out):
                return False, f"input {int(n_in)} frames vs output {int(n_out)} frames ({detail})"
            detail += f"; frames {n_in} -> {n_out}"
        return True, detail
    except Exception as exc:
        return True, f"skipped ({exc})"


def _verify_split_coverage(
    source_path: str,
    chunk_paths: List[Path],
    work_dir: Path,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """
    Verify that the split chunks cover the source exactly once (plus any intentional
    overlap recorded in the manifest). Returns (ok, message). When ok is False the caller
    must NOT start processing: duplicated/missing frames at chunk boundaries turn into
    progressive A/V desync and a truncated ending in the merged output.
    """
    def _say(msg: str) -> None:
        if on_progress:
            try:
                on_progress(msg if msg.endswith("\n") else msg + "\n")
            except Exception:
                pass

    if len(chunk_paths) < 2:
        return True, ""
    manifest = _load_split_manifest(work_dir)
    src_frames, method = _probe_source_frame_count(Path(source_path))
    r_fps, avg_fps = _probe_frame_rates(Path(source_path))
    is_cfr = bool(r_fps and avg_fps and abs(r_fps - avg_fps) <= max(1e-6, 0.002 * r_fps))

    chunk_frames: List[Optional[int]] = [_probe_chunk_nb_frames(Path(p)) for p in chunk_paths]
    if any(v is None for v in chunk_frames):
        _say("Split verification: could not count frames of every chunk; skipping strict check.")
        return True, ""
    total_chunk_frames = int(sum(int(v) for v in chunk_frames))  # type: ignore[arg-type]

    intended_overlap = 0
    overlaps = _manifest_overlap_frames(manifest, len(chunk_paths))
    if overlaps:
        intended_overlap = int(sum(overlaps))

    if manifest:
        expected_list = [c.get("expected_frames") for c in (manifest.get("chunks") or [])]
        mismatched = [
            (i + 1, int(chunk_frames[i]), int(exp))  # type: ignore[arg-type]
            for i, exp in enumerate(expected_list)
            if exp is not None and i < len(chunk_frames) and int(chunk_frames[i]) != int(exp)  # type: ignore[arg-type]
        ]
        if mismatched and is_cfr:
            preview = ", ".join(f"chunk {i}: {got} vs {exp}" for i, got, exp in mismatched[:8])
            _say(f"Split verification: {len(mismatched)} chunk(s) differ from the requested frame range ({preview}).")

    if src_frames is None:
        _say(
            f"Split verification: source frame count unavailable; chunks contain {total_chunk_frames} frames "
            f"(intended overlap {intended_overlap})."
        )
        return True, ""

    expected_total = int(src_frames) + int(intended_overlap)
    delta = total_chunk_frames - expected_total
    tolerance = 0 if method == "nb_frames" else 1
    summary = (
        f"Split verification: {len(chunk_paths)} chunks, {total_chunk_frames} frames total; "
        f"source has {src_frames} frames ({method}); intended overlap {intended_overlap}; delta {delta:+d}."
    )
    if abs(delta) <= tolerance:
        _say(summary + " OK")
        return True, ""

    if is_cfr:
        msg = (
            summary
            + f" FAILED: chunks would {'duplicate' if delta > 0 else 'lose'} {abs(delta)} frame(s), which causes "
            "progressive audio/video desync and a truncated ending in the merged output. "
            "Aborting before processing. Enable 'Frame-Accurate Split (Lossless)' in the Resolution tab "
            "(or report this input) and retry."
        )
        _say("ERROR: " + msg)
        return False, msg

    _say(
        summary
        + " WARNING: source is variable-frame-rate, so this check is advisory only; "
        "if the final output drifts, enable 'Frame-Accurate Split (Lossless)'."
    )
    return True, ""

def _remux_video_with_fresh_timestamps(
    src_path: Path,
    dst_path: Path,
    on_progress: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    Stream-copy remux with regenerated timestamps.

    This is used only when merge output duration drifts too high/low and we want
    to avoid full re-encoding.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-fflags",
        "+genpts",
        "-i",
        str(src_path),
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-an",
        "-movflags",
        "+faststart",
        "-avoid_negative_ts",
        "make_zero",
        str(dst_path),
    ]
    proc = _run_ffmpeg(cmd)
    if proc.returncode == 0 and dst_path.exists() and dst_path.stat().st_size > 1024:
        return True
    if on_progress:
        tail = (proc.stderr or proc.stdout or "").strip()[-400:]
        on_progress("WARN: Timestamp remux failed.\n")
        if tail:
            on_progress(f"ffmpeg: {tail}\n")
    return False


_CHUNK_INDEX_RE = re.compile(r"chunk_(\d+)", flags=re.IGNORECASE)


def _extract_chunk_index(path: Path) -> Optional[int]:
    try:
        m = _CHUNK_INDEX_RE.search(path.stem)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def _wait_for_media_file_ready(
    media_path: Path,
    *,
    expected_duration: Optional[float] = None,
    timeout_sec: float = 20.0,
    poll_sec: float = 0.35,
) -> bool:
    """
    Wait until a media file exists, has non-trivial size, and appears stable.
    This mitigates races where chunk files are still being finalized.
    """
    media_path = Path(media_path)
    deadline = time.time() + max(0.5, float(timeout_sec))
    last_size = -1
    stable_ticks = 0

    while time.time() < deadline:
        try:
            if media_path.exists() and media_path.is_file():
                size = int(media_path.stat().st_size or 0)
                if size > 1024:
                    dur = get_media_duration_seconds(str(media_path))
                    dur_ok = bool(dur and dur > 0)
                    if dur_ok and expected_duration and expected_duration > 0:
                        # Use a forgiving threshold; some pipelines trim a small tail.
                        dur_ok = float(dur) >= float(expected_duration) * 0.55

                    if dur_ok and size == last_size:
                        stable_ticks += 1
                    else:
                        stable_ticks = 0
                    last_size = size

                    if dur_ok and stable_ticks >= 2:
                        return True
        except Exception:
            pass
        time.sleep(max(0.05, float(poll_sec)))

    try:
        return media_path.exists() and media_path.is_file() and media_path.stat().st_size > 1024
    except Exception:
        return False


def _collect_merge_chunk_paths(
    preferred_chunks: List[Path],
    *,
    processed_dir: Optional[Path] = None,
    expected_count: Optional[int] = None,
) -> List[Path]:
    """
    Build an ordered, deduplicated list of chunk video files for merging.
    Prefers explicit `preferred_chunks`, then fills gaps from processed dir.
    """
    by_index: Dict[int, Path] = {}
    extras: List[Path] = []

    def _add_candidate(p: Path, prefer: bool) -> None:
        try:
            p = Path(p)
            if not (p.exists() and p.is_file()):
                return
            idx = _extract_chunk_index(p)
            if idx is None:
                extras.append(p)
                return
            if idx not in by_index or prefer:
                by_index[idx] = p
        except Exception:
            return

    for p in preferred_chunks or []:
        _add_candidate(Path(p), prefer=True)

    if processed_dir and Path(processed_dir).exists():
        pd = Path(processed_dir)
        for pat in ("chunk_*_upscaled.mp4", "chunk_*_out.mp4"):
            for p in sorted(pd.glob(pat)):
                _add_candidate(p, prefer=False)

    ordered = [by_index[i] for i in sorted(by_index.keys())]
    if extras:
        seen = {str(p.resolve()) for p in ordered}
        for p in sorted(extras):
            try:
                key = str(p.resolve())
            except Exception:
                key = str(p)
            if key not in seen:
                ordered.append(p)
                seen.add(key)

    if expected_count is not None and expected_count > 0:
        ordered = ordered[: int(expected_count)]

    return ordered


def _write_concat_list(
    txt_path: Path,
    paths: List[Path],
    durations: Optional[List[Optional[float]]] = None,
) -> None:
    """
    Write an ffmpeg concat-demuxer list.

    `durations` (seconds, per file) should be the VIDEO stream duration of each chunk.
    Without an explicit `duration` directive the concat demuxer offsets the next file by
    the container duration, i.e. the LONGEST stream. Processed chunks carry their own
    (preview) audio track, which is typically a few ms longer than the video, so a
    video-only merge would otherwise get a small timestamp gap at every chunk boundary
    (up to one frame per boundary -> seconds of A/V drift on long videos).
    """
    with txt_path.open("w", encoding="utf-8") as f:
        for i, p in enumerate(paths):
            f.write(f"file '{p.resolve().as_posix()}'\n")
            if durations is not None and i < len(durations):
                dur = durations[i]
                if dur is not None and dur > 0:
                    f.write(f"duration {float(dur):.6f}\n")


def _run_ffmpeg(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _parse_fraction_to_float(value: Any) -> Optional[float]:
    """
    Parse ffprobe fraction-like values (e.g. "30000/1001", "20/1", "19.97").
    """
    try:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw or raw.lower() in {"n/a", "nan"}:
            return None
        if "/" in raw:
            num_raw, den_raw = raw.split("/", 1)
            num = float(num_raw.strip())
            den = float(den_raw.strip())
            if abs(den) < 1e-12:
                return None
            out = num / den
        else:
            out = float(raw)
        if not math.isfinite(out) or out <= 0:
            return None
        return float(out)
    except Exception:
        return None


def _probe_concat_video_signature(video_path: Path) -> Optional[Dict[str, Any]]:
    """
    Probe merge-relevant video stream fields for concat compatibility decisions.
    """
    try:
        p = Path(video_path)
        if not p.exists() or not p.is_file():
            return None
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,width,height,time_base,r_frame_rate,avg_frame_rate,start_time,duration,nb_frames",
                "-of",
                "json",
                str(p),
            ],
            capture_output=True,
            text=True,
            timeout=12,
        )
        if proc.returncode != 0:
            return None
        payload = json.loads(proc.stdout or "{}")
        streams = payload.get("streams") or []
        if not streams:
            return None
        st = streams[0] or {}

        codec = str(st.get("codec_name") or "").strip().lower()
        pix_fmt = str(st.get("pix_fmt") or "").strip().lower()
        time_base = str(st.get("time_base") or "").strip()
        r_fps_raw = str(st.get("r_frame_rate") or "").strip()
        avg_fps_raw = str(st.get("avg_frame_rate") or "").strip()

        try:
            width = int(float(st.get("width") or 0))
            height = int(float(st.get("height") or 0))
        except Exception:
            width = 0
            height = 0

        fps_r = _parse_fraction_to_float(r_fps_raw)
        fps_avg = _parse_fraction_to_float(avg_fps_raw)
        fps = fps_r or fps_avg

        duration = _parse_fraction_to_float(st.get("duration"))
        start_time = _parse_fraction_to_float(st.get("start_time"))

        frame_count: Optional[int] = None
        try:
            raw_frames = str(st.get("nb_frames") or "").strip()
            if raw_frames and raw_frames.lower() not in {"n/a", "nan"}:
                frame_count = int(float(raw_frames))
        except Exception:
            frame_count = None

        if not codec:
            return None
        return {
            "codec_name": codec,
            "pix_fmt": pix_fmt,
            "width": width,
            "height": height,
            "time_base": time_base,
            "r_frame_rate": r_fps_raw,
            "avg_frame_rate": avg_fps_raw,
            "fps": fps,
            "duration": duration,
            "start_time": start_time,
            "nb_frames": frame_count,
        }
    except Exception:
        return None


def _merge_stream_copy_is_safe(signatures: List[Optional[Dict[str, Any]]]) -> Tuple[bool, str]:
    """
    Decide whether ffmpeg stream-copy concat is safe for the full chunk set.
    """
    sigs: List[Dict[str, Any]] = [s for s in signatures if isinstance(s, dict)]
    if not sigs or len(sigs) != len(signatures):
        return False, "missing ffprobe stream signatures"

    # Stream-copy concat is sensitive to per-chunk timing fields.
    required_fields = ["codec_name", "pix_fmt", "width", "height", "time_base", "r_frame_rate"]
    mismatched: List[str] = []
    for field in required_fields:
        values = {str(sig.get(field) or "").strip().lower() for sig in sigs}
        values.discard("")
        if len(values) > 1:
            mismatched.append(field)

    if mismatched:
        return False, f"mixed chunk stream fields: {', '.join(mismatched)}"
    return True, ""


def _pick_merge_fps(
    signatures: List[Optional[Dict[str, Any]]],
    chunk_paths: List[Path],
) -> Optional[float]:
    """
    Pick a robust target FPS for merge re-encode fallback.
    """
    fps_values: List[float] = []
    for sig in signatures:
        if not isinstance(sig, dict):
            continue
        fps_val = _parse_fraction_to_float(sig.get("r_frame_rate")) or _parse_fraction_to_float(
            sig.get("avg_frame_rate")
        )
        if fps_val and 1.0 <= fps_val <= 240.0:
            fps_values.append(float(fps_val))

    if not fps_values and chunk_paths:
        try:
            guessed = float(get_media_fps(str(chunk_paths[0])) or 0.0)
            if guessed > 0:
                fps_values.append(guessed)
        except Exception:
            pass

    if not fps_values:
        return None

    picked = float(median(fps_values))
    # Only absorb float noise (e.g. 25.0000001 -> 25). NEVER snap NTSC rates
    # (23.976 -> 24, 29.97 -> 30): that is a 0.1% speed change, i.e. ~2.4 s of
    # audio/video drift on a 40-minute video.
    nearest_int = round(picked)
    if abs(picked - nearest_int) <= 1e-3:
        picked = float(nearest_int)
    return max(1.0, min(240.0, picked))


def _pick_merge_fps_str(
    signatures: List[Optional[Dict[str, Any]]],
    chunk_paths: List[Path],
) -> Optional[str]:
    """
    Target FPS for merge re-encode fallback as an ffmpeg-parsable string.
    Prefer the exact rational (e.g. "24000/1001") when all chunks agree on it so the
    merged output keeps the exact source timing.
    """
    rationals = set()
    for sig in signatures:
        if not isinstance(sig, dict):
            continue
        raw = str(sig.get("r_frame_rate") or "").strip()
        val = _parse_fraction_to_float(raw)
        if raw and val and 1.0 <= val <= 240.0:
            rationals.add(raw)
    if len(rationals) == 1:
        return next(iter(rationals))
    picked = _pick_merge_fps(signatures, chunk_paths)
    if not picked or picked <= 0:
        return None
    fps_str = f"{float(picked):.6f}".rstrip("0").rstrip(".")
    return fps_str or None


def _normalize_video_encode_settings(encode_settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(encode_settings or {})
    codec_raw = str(cfg.get("video_codec", "h264") or "h264").strip().lower()
    codec_map = {
        "libx264": "h264",
        "x264": "h264",
        "h264": "h264",
        "avc": "h264",
        "libx265": "h265",
        "x265": "h265",
        "h265": "h265",
        "hevc": "h265",
        "libvpx-vp9": "vp9",
        "vp9": "vp9",
        "libaom-av1": "av1",
        "av1": "av1",
        "prores": "prores",
        "prores_ks": "prores",
    }
    codec = codec_map.get(codec_raw, "h264")
    try:
        quality = int(cfg.get("video_quality", 18) or 18)
    except Exception:
        quality = 18
    preset = str(cfg.get("video_preset", "medium") or "medium")
    h265_tune = str(cfg.get("h265_tune", "none") or "none").strip().lower() or "none"
    try:
        av1_film_grain = int(float(cfg.get("av1_film_grain", 8) or 8))
    except Exception:
        av1_film_grain = 8
    av1_film_grain = max(0, min(50, av1_film_grain))
    av1_film_grain_denoise = bool(cfg.get("av1_film_grain_denoise", False))
    pixel_format = str(cfg.get("pixel_format", "yuv420p") or "yuv420p").strip().lower()
    use_10bit = bool(cfg.get("use_10bit", False) or cfg.get("seedvr2_use_10bit", False))
    # SeedVR2's `--10bit` should dominate downstream ffmpeg enforcement/merge settings.
    # Without this, a stale/default Output-tab pix_fmt (yuv420p) can silently collapse
    # chunk/final outputs back to 8-bit during best-effort normalization.
    if codec == "h265" and use_10bit and "10le" not in pixel_format:
        pixel_format = "yuv420p10le"
    return {
        "codec": codec,
        "quality": quality,
        "preset": preset,
        "h265_tune": h265_tune,
        "av1_film_grain": av1_film_grain,
        "av1_film_grain_denoise": av1_film_grain_denoise,
        "pixel_format": pixel_format,
        "use_10bit": use_10bit,
    }


def _probe_video_stream_info(video_path: Path) -> Optional[Dict[str, str]]:
    """
    Probe the first video stream and normalize useful codec fields.
    """
    try:
        if not Path(video_path).exists() or not Path(video_path).is_file():
            return None
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt",
                "-of",
                "default=noprint_wrappers=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return None
        parsed: Dict[str, str] = {}
        for raw_line in str(proc.stdout or "").splitlines():
            if "=" not in raw_line:
                continue
            key, val = raw_line.split("=", 1)
            k = str(key or "").strip().lower()
            v = str(val or "").strip()
            if not k or not v:
                continue
            parsed[k] = v
        codec_raw = str(parsed.get("codec_name", "")).strip().lower()
        if not codec_raw:
            return None
        codec_map = {
            "h264": "h264",
            "avc": "h264",
            "hevc": "h265",
            "h265": "h265",
            "vp9": "vp9",
            "av1": "av1",
            "prores": "prores",
            "prores_ks": "prores",
        }
        codec_norm = codec_map.get(codec_raw, codec_raw)
        pix_fmt = str(parsed.get("pix_fmt", "")).strip().lower()
        info: Dict[str, str] = {"codec": codec_norm}
        if pix_fmt:
            info["pix_fmt"] = pix_fmt
        return info
    except Exception:
        return None


def _probe_video_codec_key(video_path: Path) -> Optional[str]:
    info = _probe_video_stream_info(video_path)
    if not info:
        return None
    return str(info.get("codec") or "").strip().lower() or None


def _probe_video_codec_key_with_retry(
    video_path: Path,
    *,
    attempts: int = 6,
    delay_sec: float = 0.25,
) -> Optional[str]:
    """
    Probe codec with short retries to avoid transient race/lock windows.
    """
    tries = max(1, int(attempts))
    for i in range(tries):
        codec = _probe_video_codec_key(video_path)
        if codec:
            return codec
        if i < tries - 1:
            time.sleep(max(0.05, float(delay_sec)))
    return None


def _probe_video_stream_info_with_retry(
    video_path: Path,
    *,
    attempts: int = 6,
    delay_sec: float = 0.25,
) -> Optional[Dict[str, str]]:
    tries = max(1, int(attempts))
    for i in range(tries):
        info = _probe_video_stream_info(video_path)
        if info:
            return info
        if i < tries - 1:
            time.sleep(max(0.05, float(delay_sec)))
    return None


def _reencode_video_to_match_settings(
    video_path: Path,
    encode_settings: Optional[Dict[str, Any]],
    on_progress: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    Disabled by design.

    Post-processing codec re-encode breaks the "no extra generation loss" contract for
    chunk pipelines. Keep this function as an explicit no-op for backward compatibility.
    """
    if on_progress:
        on_progress("INFO: Post codec enforcement re-encode is disabled.\n")
    return False


def _enforce_final_video_codec(
    video_path: Path,
    encode_settings: Optional[Dict[str, Any]],
    on_progress: Optional[Callable[[str], None]] = None,
    context_label: str = "Final merged",
) -> Path:
    """
    Disabled by design.

    Keep final video untouched to avoid hidden post re-encode quality loss.
    """
    outp = Path(video_path)
    if on_progress:
        on_progress(f"INFO: {context_label} codec enforcement skipped (no post re-encode).\n")
    return outp


def _enforce_merge_input_chunk_codecs(
    chunk_paths: List[Path],
    encode_settings: Optional[Dict[str, Any]],
    on_progress: Optional[Callable[[str], None]] = None,
    context_prefix: str = "Merge input chunk",
) -> List[Path]:
    """
    Disabled by design.

    Keep merge inputs untouched to avoid hidden post re-encode quality loss.
    """
    if on_progress:
        on_progress(f"INFO: {context_prefix} codec enforcement skipped (no post re-encode).\n")
    return [Path(p) for p in chunk_paths]


def concat_videos(
    chunk_paths: List[Path],
    output_path: Path,
    encode_settings: Optional[Dict[str, Any]] = None,
    on_progress: Optional[Callable[[str], None]] = None,
    nominal_fps: Optional[str] = None,
) -> bool:
    """
    Concatenate chunk videos into a single MP4.
    Merge is always done as video-only; caller can remux original audio afterward.

    `nominal_fps` (ffmpeg rational such as "24000/1001") is the exact frame rate of a CFR
    source. When given, per-chunk durations for the concat demuxer are computed as
    decodable_frames / fps instead of being read from container metadata (which some ffmpeg
    builds report including start offsets or minus the last frame).
    """
    if not chunk_paths:
        return False

    # Filter and stabilize candidate chunk files before writing concat list.
    stable_chunks: List[Path] = []
    for p in chunk_paths:
        p = Path(p)
        if _wait_for_media_file_ready(p, timeout_sec=20.0):
            stable_chunks.append(p)

    if not stable_chunks:
        if on_progress:
            on_progress("ERROR: No stable chunk files found for merge.\n")
        return False
    if len(stable_chunks) < len(chunk_paths):
        if on_progress:
            on_progress(
                f"ERROR: Only {len(stable_chunks)}/{len(chunk_paths)} chunk file(s) are ready; "
                "aborting merge to prevent truncated output.\n"
            )
        return False

    # Video-stream durations per chunk: used both for the merge duration check and as explicit
    # concat-demuxer `duration` directives (so a slightly longer preview-audio track inside a
    # chunk cannot push the next chunk's timestamps and open gaps in the merged video).
    chunk_video_durations: List[Optional[float]] = [_probe_video_stream_duration(Path(p)) for p in stable_chunks]
    nominal_fps_val = _parse_fraction_to_float(nominal_fps) if nominal_fps else None
    if nominal_fps_val and nominal_fps_val > 0:
        frame_based: List[Optional[float]] = []
        for p in stable_chunks:
            n = _probe_chunk_nb_frames(Path(p))
            frame_based.append((float(n) / float(nominal_fps_val)) if n else None)
        if all(d is not None and d > 0 for d in frame_based):
            chunk_video_durations = frame_based
            if on_progress:
                on_progress(f"Merge timing: per-chunk durations derived from frame counts at {nominal_fps} fps.\n")

    txt = output_path.parent / "concat.txt"
    _write_concat_list(txt, stable_chunks, durations=chunk_video_durations)

    expected_duration: Optional[float] = None
    if all(d is not None and d > 0 for d in chunk_video_durations):
        expected_duration = float(sum(float(d) for d in chunk_video_durations))  # type: ignore[arg-type]
    if len(stable_chunks) > 1 and (expected_duration is None or expected_duration <= 0):
        if on_progress:
            on_progress(
                "ERROR: Could not probe duration for every chunk; aborting merge to prevent short output.\n"
            )
        return False
    if on_progress:
        on_progress(f"Concatenating {len(stable_chunks)} chunk(s) (video-only merge)...\n")

    def _validate_or_fix_duration(path: Path, min_ratio: float = 0.90) -> bool:
        if not (path.exists() and path.stat().st_size > 1024):
            return False
        if expected_duration is None or expected_duration <= 0:
            return True

        actual_now = _probe_video_stream_duration(path)
        if on_progress and actual_now is not None:
            on_progress(
                "Merge duration check: "
                f"expected~{float(expected_duration):.3f}s, actual={float(actual_now):.3f}s.\n"
            )

        if _duration_is_plausible(path, expected_duration, min_ratio=min_ratio):
            return True

        actual_before = _probe_video_stream_duration(path)
        if on_progress:
            on_progress(
                "WARN: Merge duration drift detected "
                f"(expected~{float(expected_duration):.3f}s, "
                f"actual={float(actual_before or 0.0):.3f}s). "
                "Attempting timestamp remux fix.\n"
            )

        tmp_fixed = path.with_name(f"{path.stem}.__tsfix{path.suffix}")
        with suppress(Exception):
            tmp_fixed.unlink(missing_ok=True)
        if not _remux_video_with_fresh_timestamps(path, tmp_fixed, on_progress=on_progress):
            with suppress(Exception):
                tmp_fixed.unlink(missing_ok=True)
            return False

        fixed_ok = _duration_is_plausible(tmp_fixed, expected_duration, min_ratio=min_ratio)
        actual_after = _probe_video_stream_duration(tmp_fixed)
        if fixed_ok:
            try:
                os.replace(str(tmp_fixed), str(path))
                if on_progress:
                    on_progress(
                        "Timestamp remux fix applied "
                        f"(new duration={float(actual_after or 0.0):.3f}s).\n"
                    )
                return True
            except Exception:
                # If replacement fails, keep original path outcome and fail closed.
                with suppress(Exception):
                    tmp_fixed.unlink(missing_ok=True)
                return False

        if on_progress:
            on_progress(
                "WARN: Timestamp remux did not fix duration drift "
                f"(new duration={float(actual_after or 0.0):.3f}s).\n"
            )
        with suppress(Exception):
            tmp_fixed.unlink(missing_ok=True)
        return False

    # Trivial fast-path: only one chunk.
    if len(stable_chunks) == 1:
        try:
            output_path.unlink(missing_ok=True)
            shutil.copy2(stable_chunks[0], output_path)
            return _validate_or_fix_duration(output_path, min_ratio=0.90)
        except Exception:
            return False

    signatures = [_probe_concat_video_signature(p) for p in stable_chunks]
    stream_copy_safe, stream_copy_reason = _merge_stream_copy_is_safe(signatures)

    def _build_fallback_encode_args() -> Tuple[List[str], str]:
        enc = _normalize_video_encode_settings(encode_settings)
        # Exact rational when possible ("24000/1001"); a snapped/rounded value would
        # change playback speed and desync audio.
        fps_str = _pick_merge_fps_str(signatures, stable_chunks) or "30"
        video_encode_args = build_ffmpeg_video_encode_args(
            codec=enc["codec"],
            quality=int(enc["quality"]),
            pixel_format=str(enc["pixel_format"]),
            preset=str(enc["preset"]),
            audio_codec="none",
            audio_bitrate=None,
            h265_tune=str(enc["h265_tune"]),
            av1_film_grain=int(enc["av1_film_grain"]),
            av1_film_grain_denoise=bool(enc["av1_film_grain_denoise"]),
        )
        return video_encode_args, fps_str

    def _try_framepipe_reencode_concat(reason: str) -> bool:
        video_encode_args, fps_str = _build_fallback_encode_args()
        if on_progress:
            on_progress(
                "WARN: Falling back to frame-stream concat re-encode "
                f"(reason: {reason}; target_fps={fps_str}).\n"
            )

        output_path.unlink(missing_ok=True)

        first_frame = None
        width = 0
        height = 0
        for chunk_path in stable_chunks:
            cap = cv2.VideoCapture(str(chunk_path))
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                first_frame = frame
                height, width = frame.shape[:2]
                break

        if first_frame is None or width <= 0 or height <= 0:
            if on_progress:
                on_progress("WARN: Frame-stream fallback could not decode any input frames.\n")
            return False

        cmd_pipe = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{int(width)}x{int(height)}",
            "-r",
            fps_str,
            "-i",
            "-",
            "-map",
            "0:v:0",
            *video_encode_args,
            "-movflags",
            "+faststart",
            str(output_path),
        ]

        proc_pipe: Optional[subprocess.Popen] = None
        frames_written = 0
        try:
            proc_pipe = subprocess.Popen(
                cmd_pipe,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if proc_pipe.stdin is None:
                raise RuntimeError("ffmpeg stdin pipe unavailable")

            for chunk_path in stable_chunks:
                cap = cv2.VideoCapture(str(chunk_path))
                if not cap.isOpened():
                    cap.release()
                    if on_progress:
                        on_progress(f"WARN: Frame-stream fallback could not open chunk: {chunk_path.name}\n")
                    continue
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            break
                        if frame.shape[1] != width or frame.shape[0] != height:
                            # Keep merge resilient when a rare chunk has mismatched dimensions.
                            frame = cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
                        proc_pipe.stdin.write(frame.tobytes())
                        frames_written += 1
                finally:
                    cap.release()

            proc_pipe.stdin.close()
            _stdout, _stderr = proc_pipe.communicate()
            if proc_pipe.returncode == 0 and frames_written > 0 and _validate_or_fix_duration(output_path, min_ratio=0.90):
                if on_progress:
                    on_progress(
                        f"Concatenated {len(stable_chunks)} chunk(s) via frame-stream fallback "
                        f"(frames={frames_written}).\n"
                    )
                return True

            if on_progress:
                tail = (_stderr.decode(errors="ignore") if isinstance(_stderr, (bytes, bytearray)) else str(_stderr or ""))[-500:]
                on_progress("WARN: Frame-stream fallback failed.\n")
                if tail:
                    on_progress(f"ffmpeg: {tail}\n")
            return False
        except Exception as e:
            if on_progress:
                on_progress(f"WARN: Frame-stream fallback exception: {str(e)}\n")
            return False
        finally:
            if proc_pipe is not None:
                with suppress(Exception):
                    if proc_pipe.stdin:
                        proc_pipe.stdin.close()
                with suppress(Exception):
                    proc_pipe.kill()

    def _try_demuxer_reencode_concat(reason: str) -> bool:
        video_encode_args, fps_str = _build_fallback_encode_args()
        if on_progress:
            on_progress(
                "WARN: Falling back to concat-demuxer re-encode "
                f"(reason: {reason}; target_fps={fps_str}).\n"
            )

        output_path.unlink(missing_ok=True)
        cmd_reencode = [
            "ffmpeg",
            "-y",
            "-fflags",
            "+genpts",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(txt),
            "-map",
            "0:v:0",
            "-vf",
            f"setpts=N/({fps_str}*TB)",
            "-r",
            fps_str,
            "-fps_mode",
            "cfr",
            *video_encode_args,
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        proc_reencode = _run_ffmpeg(cmd_reencode)
        if proc_reencode.returncode == 0 and _validate_or_fix_duration(output_path, min_ratio=0.90):
            if on_progress:
                on_progress(
                    f"Concatenated {len(stable_chunks)} chunk(s) via concat-demuxer re-encode fallback.\n"
                )
            return True

        if on_progress:
            tail = (proc_reencode.stderr or proc_reencode.stdout or "").strip()[-500:]
            on_progress("ERROR: Concat-demuxer re-encode fallback failed.\n")
            if tail:
                on_progress(f"ffmpeg: {tail}\n")
        return False

    if not stream_copy_safe and on_progress:
        on_progress(f"WARN: Stream-copy concat skipped: {stream_copy_reason}.\n")

    # Stream-copy path (fast, no extra generation loss) only for homogeneous streams.
    if stream_copy_safe:
        # Use a tight duration check for copy-based merges. A relaxed threshold can accept
        # timestamp-compressed outputs that play in tolerant media players but decode poorly
        # in stricter NLEs such as Resolve.
        copy_merge_min_ratio = 0.999

        # Prefer direct concat demuxer + copy first. For MP4 chunk sets this preserves the
        # original MP4 timing more reliably than the TS bridge when chunk streams already
        # match cleanly.
        output_path.unlink(missing_ok=True)
        cmd_copy = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(txt),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        proc_copy = _run_ffmpeg(cmd_copy)
        if proc_copy.returncode == 0 and _validate_or_fix_duration(output_path, min_ratio=copy_merge_min_ratio):
            if on_progress:
                on_progress(
                    f"Concatenated {len(stable_chunks)} chunk(s) via direct stream copy.\n"
                )
            return True
        if on_progress:
            tail = (proc_copy.stderr or proc_copy.stdout or "").strip()[-400:]
            on_progress("WARN: Direct stream-copy concat failed validation; trying TS fallback.\n")
            if tail:
                on_progress(f"ffmpeg: {tail}\n")

        # Secondary path for H.264/H.265: convert each MP4 segment to MPEG-TS (Annex B),
        # then concat-copy back to MP4. Keep this as a fallback for chunk sets where direct
        # MP4 concat-copy is rejected by ffmpeg or produces incompatible bitstream metadata.
        codec_keys = [_probe_video_codec_key_with_retry(p, attempts=4, delay_sec=0.15) for p in stable_chunks]
        common_codec: Optional[str] = None
        if codec_keys and all(k == codec_keys[0] and k for k in codec_keys):
            common_codec = str(codec_keys[0] or "").strip().lower()

        if common_codec in {"h264", "h265"}:
            bsf = "h264_mp4toannexb" if common_codec == "h264" else "hevc_mp4toannexb"
            output_path.unlink(missing_ok=True)
            with tempfile.TemporaryDirectory(prefix="merge_ts_copy_") as td:
                td_path = Path(td)
                ts_paths: List[Path] = []
                ts_ok = True
                for i, src in enumerate(stable_chunks, 1):
                    ts_path = td_path / f"chunk_{i:04d}.ts"
                    cmd_to_ts = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(src),
                        "-map",
                        "0:v:0",
                        "-c:v",
                        "copy",
                        "-bsf:v",
                        bsf,
                        "-an",
                        "-f",
                        "mpegts",
                        str(ts_path),
                    ]
                    proc_to_ts = _run_ffmpeg(cmd_to_ts)
                    if proc_to_ts.returncode != 0 or not ts_path.exists() or ts_path.stat().st_size <= 512:
                        ts_ok = False
                        if on_progress:
                            tail = (proc_to_ts.stderr or proc_to_ts.stdout or "").strip()[-300:]
                            on_progress(f"WARN: TS conversion failed for chunk {i}: {tail}\n")
                        break
                    ts_paths.append(ts_path)

                if ts_ok and len(ts_paths) == len(stable_chunks):
                    ts_txt = td_path / "concat_ts.txt"
                    _write_concat_list(ts_txt, ts_paths)
                    cmd_ts_concat = [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        str(ts_txt),
                        "-map",
                        "0:v:0",
                        "-c:v",
                        "copy",
                        "-an",
                        "-movflags",
                        "+faststart",
                        str(output_path),
                    ]
                    proc_ts_concat = _run_ffmpeg(cmd_ts_concat)
                    if proc_ts_concat.returncode == 0 and _validate_or_fix_duration(output_path, min_ratio=copy_merge_min_ratio):
                        if on_progress:
                            on_progress(
                                f"Concatenated {len(stable_chunks)} chunk(s) via TS stream copy (codec={common_codec}).\n"
                            )
                        return True
                    if on_progress:
                        tail = (proc_ts_concat.stderr or proc_ts_concat.stdout or "").strip()[-400:]
                        on_progress("WARN: TS stream-copy concat failed validation; trying robust fallback.\n")
                        if tail:
                            on_progress(f"ffmpeg: {tail}\n")

    # Robust fallback path for mixed-timestamp chunks or failed stream-copy concat.
    if _try_framepipe_reencode_concat(stream_copy_reason or "stream-copy merge failed"):
        return True
    if _try_demuxer_reencode_concat(stream_copy_reason or "stream-copy merge failed"):
        return True

    # Avoid leaving a broken file with the final output name.
    with suppress(Exception):
        output_path.unlink(missing_ok=True)
    return False


def concat_videos_with_blending(
    chunk_paths: List[Path],
    output_path: Path,
    overlap_frames: Any = 0,
    fps: Optional[float] = None,
    encode_settings: Optional[Dict[str, Any]] = None,
    on_progress: Optional[Callable[[str], None]] = None,
    nominal_fps: Optional[str] = None,
) -> bool:
    """
    Concatenate video chunks with smooth blending of overlapping regions.
    
    Args:
        chunk_paths: List of video chunk file paths
        output_path: Output video path
        overlap_frames: Number of overlapping frames between chunks. Either a single int
            (same overlap at every boundary) or a list with one entry per boundary
            (len == len(chunk_paths) - 1) holding the EXACT overlap of that boundary.
        fps: Frame rate (detected from first chunk if None)
        on_progress: Progress callback
        
    Returns:
        True if successful, False otherwise
    """
    if not chunk_paths:
        return False

    # Normalize overlap spec to one exact value per boundary.
    boundary_overlaps: List[int] = []
    if isinstance(overlap_frames, (list, tuple)):
        boundary_overlaps = [max(0, int(v or 0)) for v in overlap_frames]
    else:
        try:
            uniform = max(0, int(overlap_frames or 0))
        except Exception:
            uniform = 0
        boundary_overlaps = [uniform] * max(0, len(chunk_paths) - 1)
    if len(boundary_overlaps) < max(0, len(chunk_paths) - 1):
        pad_val = boundary_overlaps[-1] if boundary_overlaps else 0
        boundary_overlaps += [pad_val] * (len(chunk_paths) - 1 - len(boundary_overlaps))
    
    # If no overlap, use simple concat
    if not any(v > 0 for v in boundary_overlaps):
        return concat_videos(
            chunk_paths, output_path, encode_settings=encode_settings, on_progress=on_progress, nominal_fps=nominal_fps
        )
    
    try:
        if on_progress:
            on_progress("Concatenating chunks with frame blending...\n")
        
        # Create temp directory for blended output
        with tempfile.TemporaryDirectory(prefix="blend_") as temp_dir:
            temp_path = Path(temp_dir)
            
            # Read all chunks and blend overlaps
            all_frames = []
            
            for i, chunk_path in enumerate(chunk_paths):
                if on_progress:
                    on_progress(f"Loading chunk {i+1}/{len(chunk_paths)}...\n")
                
                # Read chunk frames
                cap = cv2.VideoCapture(str(chunk_path))
                if not cap.isOpened():
                    if on_progress:
                        on_progress(f"⚠️ Failed to open chunk {chunk_path}, skipping\n")
                    continue
                
                # Detect FPS from first chunk
                if fps is None and i == 0:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                
                chunk_frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    chunk_frames.append(frame)
                
                cap.release()
                
                if not chunk_frames:
                    continue
                
                # Convert to numpy array for blending
                chunk_array = np.array(chunk_frames)
                
                if i == 0:
                    # First chunk - add all frames
                    all_frames.extend(chunk_frames)
                else:
                    # Subsequent chunks - blend overlap region (exact overlap for THIS boundary)
                    overlap_frames = int(boundary_overlaps[i - 1]) if i - 1 < len(boundary_overlaps) else 0
                    if overlap_frames <= 0:
                        all_frames.extend(chunk_frames)
                    elif len(all_frames) >= overlap_frames and len(chunk_frames) >= overlap_frames:
                        # Get overlapping regions
                        prev_tail = np.array(all_frames[-overlap_frames:])
                        cur_head = chunk_array[:overlap_frames]
                        
                        # Blend
                        if on_progress:
                            on_progress(f"Blending {overlap_frames} frames between chunks {i} and {i+1}...\n")
                        
                        blended = blend_overlapping_frames_opencv(prev_tail, cur_head, overlap_frames)
                        
                        # Replace tail of all_frames with blended, add rest of chunk
                        all_frames = all_frames[:-overlap_frames]
                        all_frames.extend(blended)
                        all_frames.extend(chunk_frames[overlap_frames:])
                    else:
                        # Not enough frames to blend, just append
                        non_overlap_start = min(overlap_frames, len(chunk_frames))
                        all_frames.extend(chunk_frames[non_overlap_start:])
            
            if not all_frames:
                if on_progress:
                    on_progress("❌ No frames to write\n")
                return False
            
            # Write blended frames to temp video
            if on_progress:
                on_progress(f"Writing {len(all_frames)} blended frames to output...\n")
            
            # Get dimensions from first frame
            height, width = all_frames[0].shape[:2]
            
            # Create temp video file
            temp_output = temp_path / "blended_temp.mp4"
            
            # Use ffmpeg to encode (better quality than cv2.VideoWriter)
            enc = _normalize_video_encode_settings(encode_settings)
            video_encode_args = build_ffmpeg_video_encode_args(
                codec=enc["codec"],
                quality=enc["quality"],
                pixel_format=enc["pixel_format"],
                preset=enc["preset"],
                h265_tune=enc["h265_tune"],
                av1_film_grain=enc["av1_film_grain"],
                av1_film_grain_denoise=enc["av1_film_grain_denoise"],
                audio_codec="none",
            )
            bf_args = ["-bf", "0"] if enc["codec"] in {"h264", "h265", "vp9", "av1"} else []
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "bgr24",
                "-r", str(fps or 30.0),
                "-i", "-",
                *bf_args,
                *video_encode_args,
                str(temp_output)
            ]
            
            # NOTE: ffmpeg's stderr must be drained continuously. With an undrained PIPE the
            # (small, 4 KiB on Windows) pipe buffer fills up with ffmpeg's banner/progress and
            # ffmpeg blocks forever on its final log write -> proc.wait() never returns.
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            stderr_tail: List[str] = []

            def _drain_stderr() -> None:
                try:
                    if proc.stderr is None:
                        return
                    for raw_line in iter(proc.stderr.readline, b""):
                        try:
                            text_line = raw_line.decode("utf-8", errors="ignore").rstrip()
                        except Exception:
                            text_line = str(raw_line)
                        if text_line:
                            stderr_tail.append(text_line)
                            if len(stderr_tail) > 60:
                                del stderr_tail[:-60]
                except Exception:
                    pass

            drain_thread = threading.Thread(target=_drain_stderr, daemon=True)
            drain_thread.start()

            # Write frames to ffmpeg
            try:
                for frame in all_frames:
                    proc.stdin.write(frame.tobytes())
            finally:
                with suppress(Exception):
                    proc.stdin.close()
            proc.wait()
            drain_thread.join(timeout=5.0)

            if proc.returncode != 0 or not temp_output.exists():
                if on_progress:
                    on_progress(f"❌ FFmpeg encoding failed: {' | '.join(stderr_tail[-12:])}\n")
                return False
            
            # Move to final output
            shutil.move(str(temp_output), str(output_path))
            
            if on_progress:
                on_progress(f"✅ Blended video saved to {output_path}\n")
            
            return True
            
    except Exception as e:
        if on_progress:
            on_progress(f"❌ Blending failed: {e}\n")
        # Fallback to simple concat
        return concat_videos(chunk_paths, output_path, encode_settings=encode_settings, on_progress=on_progress)


def detect_resume_state(work_dir: Path, output_format: str) -> Tuple[Optional[Path], List[Path]]:
    """
    Detect if there's a resumable chunking session.
    Returns (partial_output_path, completed_chunks) or (None, []) if no resume possible.
    """
    if not work_dir.exists():
        return None, []

    processed_dir = work_dir / "processed_chunks"
    if processed_dir.exists() and processed_dir.is_dir():
        chunks_root = processed_dir
    else:
        # Backward compatibility: older versions stored chunks directly in work_dir.
        chunks_root = work_dir

    def _load_metadata_entries(meta_path: Path) -> List[Dict[str, Any]]:
        try:
            if not meta_path.exists() or not meta_path.is_file():
                return []
            with meta_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                return [payload]
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
        except Exception:
            return []
        return []

    def _collect_contiguous_video_chunks() -> List[Path]:
        """
        Gather completed chunk outputs using both filesystem patterns and run_metadata.json,
        then keep only a contiguous prefix (chunk_0001..chunk_00NN) for safe resume.
        """
        by_index: Dict[int, Path] = {}

        def _put(path_like: Any) -> Optional[int]:
            try:
                p = Path(normalize_path(str(path_like)))
            except Exception:
                return None
            idx = _extract_chunk_index(p)
            if idx is None:
                return None
            if p.exists() and p.is_file():
                by_index[idx] = p
            return idx

        # Primary discovery from processed chunk files.
        for pat in ("chunk_*_upscaled.mp4", "chunk_*_out.mp4"):
            for cand in sorted(chunks_root.glob(pat)):
                _put(cand)

        # Supplement with run metadata (useful when filenames differ but include chunk index).
        metadata_candidates = [
            chunks_root / "run_metadata.json",
            work_dir / "run_metadata.json",
        ]
        for meta_path in metadata_candidates:
            for entry in _load_metadata_entries(meta_path):
                status = str(entry.get("status") or "").strip().lower()
                returncode = entry.get("returncode")
                is_success = (status in {"success", "completed", "ok"}) or (str(returncode).strip() == "0")
                if not is_success:
                    continue

                args_blob = entry.get("args")
                args: Dict[str, Any] = args_blob if isinstance(args_blob, dict) else {}
                candidates: List[Any] = []

                for key in ("output", "output_path"):
                    val = entry.get(key)
                    if val:
                        candidates.append(val)
                for key in ("output_override",):
                    val = args.get(key)
                    if val:
                        candidates.append(val)

                found_idx: Optional[int] = None
                for cand in candidates:
                    found_idx = _put(cand)
                    if found_idx is not None:
                        break

                # If metadata indicates chunk index but path is missing, fall back to canonical names.
                if found_idx is not None and found_idx not in by_index:
                    for fallback in (
                        chunks_root / f"chunk_{found_idx:04d}_upscaled.mp4",
                        chunks_root / f"chunk_{found_idx:04d}_out.mp4",
                    ):
                        if fallback.exists() and fallback.is_file():
                            by_index[found_idx] = fallback
                            break

        # Resume is safe only across a contiguous prefix.
        contiguous: List[Path] = []
        idx = 1
        while idx in by_index:
            contiguous.append(by_index[idx])
            idx += 1
        return contiguous

    # Check for partial outputs
    if output_format == "png":
        partial_candidates = list(work_dir.glob("*_partial"))
        if partial_candidates:
            partial_dir = partial_candidates[0]
            completed_chunks = []
            chunk_pattern = partial_dir / "chunk_*.png"
            for chunk_file in sorted(chunk_pattern.parent.glob("chunk_*.png")):
                if chunk_file.exists():
                    completed_chunks.append(chunk_file)
            return partial_dir, completed_chunks
    else:
        # Video: detect completed per-chunk outputs (file scan + metadata), contiguous only.
        completed_chunks = _collect_contiguous_video_chunks()

        # If a stitched partial exists inside the chunks dir, prefer it as the "partial indicator".
        partial_candidates = list(work_dir.glob("*_partial.mp4"))
        if partial_candidates:
            partial_file = partial_candidates[0]
            return partial_file, completed_chunks

        # If we have any completed chunk outputs, consider this resumable even without a stitched partial.
        if completed_chunks:
            return work_dir, completed_chunks

    return None, []


def check_resume_available(work_dir: Path, output_format: str) -> Tuple[bool, str]:
    """
    Check if resume is available for chunking.
    Returns (available, status_message).
    """
    partial_path, completed_chunks = detect_resume_state(work_dir, output_format)

    if not partial_path:
        return False, "No partial chunking session found to resume."

    if output_format == "png" and completed_chunks:
        return True, f"Found {len(completed_chunks)} completed chunks ready to resume."
    elif output_format != "png" and completed_chunks:
        return True, f"Found {len(completed_chunks)} completed chunk outputs ready to stitch/resume."
    elif output_format != "png" and partial_path and partial_path.exists():
        return True, "Found partial video output ready to resume from."
    else:
        return False, "Partial output found but no completed chunks to resume from."


def salvage_partial_from_run_dir(
    run_dir: Path,
    *,
    partial_basename: str = "cancelled_partial",
    audio_source: Optional[str] = None,
    audio_codec: str = "copy",
    audio_bitrate: Optional[str] = None,
    encode_settings: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Path], str]:
    """
    Best-effort salvage of partial chunk outputs from a run directory.

    Returns:
        (path, method) where method is one of: "simple", "png_collection", "none"
    """
    run_dir = Path(run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        return None, "none"

    # Prefer video chunk salvage first.
    _partial_video, completed_chunks = detect_resume_state(run_dir, "mp4")
    if completed_chunks:
        target = collision_safe_path(run_dir / f"{partial_basename}.mp4")
        ok = concat_videos(completed_chunks, target, encode_settings=encode_settings)
        if ok and target.exists():
            try:
                if audio_source and Path(audio_source).exists():
                    _changed, maybe_final, audio_err = ensure_audio_on_video(
                        video_path=target,
                        audio_source_path=Path(audio_source),
                        audio_codec=str(audio_codec or "copy"),
                        audio_bitrate=str(audio_bitrate) if audio_bitrate else None,
                        force_replace=True,
                        on_progress=None,
                    )
                    if maybe_final and Path(maybe_final).exists():
                        target = Path(maybe_final)
                    if audio_err:
                        # Keep the salvaged video even when audio replacement fails.
                        pass
            except Exception:
                pass
            return target, "simple"

    # Fallback: PNG chunks.
    _partial_png, completed_png_chunks = detect_resume_state(run_dir, "png")
    if completed_png_chunks:
        target_dir = collision_safe_dir(run_dir / f"{partial_basename}_png")
        target_dir.mkdir(parents=True, exist_ok=True)
        for idx, chunk_path in enumerate(completed_png_chunks, 1):
            dest = target_dir / f"chunk_{idx:04d}"
            try:
                if Path(chunk_path).is_dir():
                    shutil.copytree(chunk_path, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(chunk_path, dest)
            except Exception:
                continue
        return target_dir, "png_collection"

    return None, "none"


def chunk_and_process(
    runner,
    settings: dict,
    scene_threshold: float,
    min_scene_len: float,
    work_dir: Path,
    on_progress: Callable[[str], None],
    chunk_seconds: float = 0.0,
    chunk_overlap: float = 0.0,
    per_chunk_cleanup: bool = False,
    allow_partial: bool = True,
    global_output_dir: Optional[str] = None,
    resume_from_partial: bool = False,
    progress_tracker=None,
    process_func: Optional[Callable] = None,
    pre_process_chunks_func: Optional[Callable] = None,
    model_type: str = "seedvr2",
) -> Tuple[int, str, str, int]:
    """
    🎬 UNIVERSAL PySceneDetect Chunking System - Works with ALL Models
    
    This is the PREFERRED chunking method that works universally across:
    - SeedVR2 (diffusion-based video upscaling)
    - GAN models (Real-ESRGAN, etc.)
    - RIFE (frame interpolation)
    - FlashVSR+ (real-time diffusion)
    
    How it works:
    1. Splits video into scenes using PySceneDetect (intelligent scene detection)
    2. OR splits into fixed-duration chunks if scene detection disabled
    3. Processes each chunk independently with the selected model
    4. Concatenates results with optional frame blending for smooth transitions
    
    Configuration (from Resolution & Scene Split tab):
    - chunk_seconds: Duration of each chunk (0 = use scene detection)
    - scene_threshold: Sensitivity for scene detection
    - chunk_overlap: Overlap between chunks for temporal consistency
    
    Note: For SeedVR2, this can work ALONGSIDE native streaming (--chunk_size in frames).
    PySceneDetect creates scene chunks, then each chunk can use native streaming internally.
    
    Args:
        runner: Runner instance with model-specific run methods
        settings: Processing settings dict (must include input_path, output_format, etc.)
        scene_threshold: PySceneDetect sensitivity (lower = more cuts, 27 = default)
        min_scene_len: Minimum scene duration in seconds
        work_dir: Run folder for chunk artifacts (creates input_chunks/ and processed_chunks/)
        on_progress: Progress callback for UI updates
        chunk_seconds: Fixed chunk size in seconds (0 = use intelligent scene detection)
        chunk_overlap: Overlap between chunks in seconds (for smooth transitions)
        per_chunk_cleanup: Delete chunk artifacts from the run output folder to save disk space
        allow_partial: Save partial results on cancel/error
        global_output_dir: Output directory override
        resume_from_partial: Resume from previous interrupted run
        progress_tracker: Additional progress tracking callback
        process_func: Optional custom processing function (takes settings, returns RunResult)
                     If None, uses model_type to select runner method
        pre_process_chunks_func: Optional callback run after chunk splitting and before
                                 per-chunk model processing starts.
        model_type: Model type ("seedvr2", "gan", "rife", "flashvsr", "sparkvsr") - used if process_func is None
    
    Returns:
        (returncode, log, final_output_path, chunk_count)
    """
    # Clear stale cancellation state from previous jobs so a fresh run can start
    # cleanly after a cancel. Mid-run cancels still work via runner.cancel().
    try:
        reset_cancel = getattr(runner, "reset_cancel_state", None)
        if callable(reset_cancel):
            reset_cancel()
    except Exception:
        pass
    run_start_ts = time.time()

    input_path = normalize_path(settings["input_path"])
    # When inputs are preprocessed (e.g., downscaled) we still want to preserve the original audio.
    audio_source_for_mux = normalize_path(settings.get("_original_input_path_before_preprocess")) or input_path
    input_type = detect_input_type(input_path)
    output_format = settings.get("output_format") or "mp4"
    if output_format in (None, "auto"):
        output_format = "mp4"
    work_root = Path(work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    input_chunks_dir = work_root / "input_chunks"
    processed_chunks_dir = work_root / "processed_chunks"
    input_chunks_dir.mkdir(parents=True, exist_ok=True)
    processed_chunks_dir.mkdir(parents=True, exist_ok=True)

    existing_partial, existing_chunks = detect_resume_state(work_root, output_format)
    exact_boundary_overlaps: Optional[List[int]] = None
    source_fps_rational: Optional[str] = None
    source_fps_float: Optional[float] = None

    # Initialize variables
    start_chunk_idx = 0
    resuming = False
    
    if resume_from_partial and existing_partial and existing_chunks:
        on_progress(f"Resuming from partial output: {existing_partial} with {len(existing_chunks)} completed chunks\n")
        resuming = True
        # Don't clean work directory - we're resuming!
        # chunk_paths will be set later from actual input, not from existing chunks
        start_chunk_idx = len(existing_chunks)
    else:
        # Fresh start - clean ONLY the chunk subfolders (never delete the run folder itself)
        shutil.rmtree(input_chunks_dir, ignore_errors=True)
        shutil.rmtree(processed_chunks_dir, ignore_errors=True)
        input_chunks_dir.mkdir(parents=True, exist_ok=True)
        processed_chunks_dir.mkdir(parents=True, exist_ok=True)
        start_chunk_idx = 0

    # Predict final output locations for partial/cancel handling
    global_override = settings.get("output_override") or global_output_dir
    explicit_final_path: Optional[Path] = None
    if global_override and output_format != "png":
        try:
            cand = Path(normalize_path(str(global_override)))
            video_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".wmv", ".m4v", ".flv"}
            if cand.exists() and cand.is_dir():
                explicit_final_path = None
            elif cand.suffix.lower() in video_exts:
                explicit_final_path = cand
        except Exception:
            explicit_final_path = None

    if explicit_final_path is not None:
        predicted_final_path = explicit_final_path
    else:
        predicted_final = resolve_output_location(
            input_path=input_path,
            output_format=output_format,
            global_output_dir=global_override,
            batch_mode=False,
            png_padding=settings.get("png_padding"),
            png_keep_basename=settings.get("png_keep_basename", False),
            original_filename=settings.get("_original_filename"),
        )
        predicted_final_path = Path(predicted_final)
    if output_format == "png":
        # For PNG sequences, ensure we point to a directory; single-image PNG still gets a sibling folder for partials
        base_dir = (
            predicted_final_path.parent / predicted_final_path.stem
            if predicted_final_path.suffix.lower() == ".png"
            else predicted_final_path
        )
        partial_png_target = collision_safe_dir(base_dir.with_name(f"{base_dir.name}_partial"))
        partial_video_target = None
    else:
        partial_png_target = None
        partial_video_target = collision_safe_path(
            predicted_final_path.with_name(f"{predicted_final_path.stem}_partial{predicted_final_path.suffix}")
        )

    # Special handling for frame-folder inputs (image sequences)
    if input_type == "directory":
        frames = sorted(
            [
                f
                for f in Path(input_path).iterdir()
                if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
            ]
        )
        if not frames:
            return 1, "No frames found in folder for chunking", "", 0
        fps_guess = 30.0
        frame_window = len(frames) if chunk_seconds <= 0 else max(1, int(chunk_seconds * fps_guess))
        overlap_frames = 0 if chunk_overlap <= 0 else int(chunk_overlap * fps_guess)
        if overlap_frames >= frame_window:
            overlap_frames = max(0, frame_window - 1)
        chunk_specs = []
        start = 0
        idx = 1
        while start < len(frames):
            end = min(len(frames), start + frame_window)
            chunk_specs.append((idx, frames[start:end]))
            if end == len(frames):
                break
            start = end - overlap_frames
            idx += 1
        on_progress(f"Detected {len(chunk_specs)} frame chunks\n")
        chunk_paths = []
        for idx, frame_list in chunk_specs:
            cdir = input_chunks_dir / f"chunk_{idx:04d}"
            cdir.mkdir(parents=True, exist_ok=True)
            for f in frame_list:
                shutil.copy2(f, cdir / f.name)
            chunk_paths.append(cdir)
        # Frame-folder chunks overlap by exactly `overlap_frames` frames at every boundary.
        exact_boundary_overlaps = [int(overlap_frames)] * max(0, len(chunk_paths) - 1)
    else:
        scenes = detect_scenes(input_path, threshold=scene_threshold, min_scene_len=min_scene_len)
        if not scenes or chunk_seconds > 0:
            effective_seconds = chunk_seconds if chunk_seconds > 0 else max(min_scene_len, 30)
            scenes = fallback_scenes(input_path, chunk_seconds=effective_seconds, overlap_seconds=max(0.0, chunk_overlap))
        on_progress(f"Detected {len(scenes)} scenes for chunking\n")

        precise_split = bool(settings.get("frame_accurate_split", True))
        audio_codec_pref = str(settings.get("audio_codec") or "copy").strip().lower()
        include_chunk_audio = audio_codec_pref not in {"none", "no", "off", "disable", "disabled"}
        try:
            on_progress(
                f"Chunk split audio: {'enabled' if include_chunk_audio else 'disabled'} "
                f"(audio_codec={audio_codec_pref or 'copy'})\n"
            )
        except Exception:
            pass
        chunk_paths = split_video(
            input_path,
            scenes,
            input_chunks_dir,
            precise=precise_split,
            preserve_quality=True,
            include_audio=include_chunk_audio,
            on_progress=on_progress,
        )
        on_progress(f"Split into {len(chunk_paths)} chunks\n")

        # Fail closed BEFORE spending GPU time: chunks must cover the source exactly once
        # (plus any intentional overlap). Duplicated/missing boundary frames would otherwise
        # surface only at the very end as progressive A/V desync and a truncated ending.
        if len(chunk_paths) > 1:
            split_ok, split_msg = _verify_split_coverage(
                input_path,
                chunk_paths,
                input_chunks_dir,
                on_progress=on_progress,
            )
            if not split_ok:
                return 1, f"Chunk split verification failed: {split_msg}", "", len(chunk_paths)
        # Exact per-boundary overlaps (frames) from the split manifest, when available.
        split_manifest = _load_split_manifest(input_chunks_dir)
        exact_boundary_overlaps = _manifest_overlap_frames(split_manifest, len(chunk_paths))
        if split_manifest and split_manifest.get("source_is_cfr") and split_manifest.get("source_fps_rational"):
            source_fps_rational = str(split_manifest.get("source_fps_rational") or "").strip() or None
            source_fps_float = _parse_fraction_to_float(source_fps_rational)
            if source_fps_float and source_fps_float > 0:
                on_progress(
                    f"[timing] source is CFR at {source_fps_rational} fps; chunk outputs will be held to this "
                    f"frame rate and per-chunk frame counts.\n"
                )
            else:
                source_fps_rational = None
                source_fps_float = None

    split_stage_weight = 0.10 if input_type != "directory" else 0.04
    merge_stage_weight = 0.06 if output_format != "png" else 0.03
    process_stage_weight = max(0.0, 1.0 - split_stage_weight - merge_stage_weight)
    split_stage_progress = 1.0
    merge_stage_progress = 0.0

    def _safe_chunk_work_units(chunk_path: Path) -> float:
        try:
            p = Path(chunk_path)
            if p.is_dir():
                frame_count = 0
                try:
                    for item in p.iterdir():
                        if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}:
                            frame_count += 1
                except Exception:
                    frame_count = 0
                return float(max(1, frame_count))
            duration = float(get_media_duration_seconds(str(p)) or 0.0)
            if duration > 0:
                fps = float(get_media_fps(str(p)) or get_media_fps(input_path) or 30.0)
                return float(max(1.0, duration * max(1.0, fps)))
        except Exception:
            pass
        return 1.0

    chunk_work_units: List[float] = [max(1.0, _safe_chunk_work_units(Path(c))) for c in chunk_paths]
    total_chunk_units = float(sum(chunk_work_units)) if chunk_work_units else 1.0
    completed_chunk_units = 0.0
    current_chunk_index = 0
    current_chunk_inner_fraction = 0.0

    last_overall_emit_ts = 0.0
    last_rate_sample_ts = run_start_ts
    last_rate_sample_progress = 0.0
    ema_progress_rate: Optional[float] = None
    last_emitted_fraction = -1.0
    inline_frame_progress_active = False
    inline_frame_progress_width = 0
    progress_report_interval_sec = max(
        1.0,
        min(60.0, float(settings.get("progress_report_interval_sec") or 10.0)),
    )

    frame_progress_re = re.compile(r"(\d+)\s*/\s*(\d+)")
    pct_progress_re = re.compile(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*%")
    spark_progress_re = re.compile(r"SparkVSR\s+Progress:\s*(\d{1,3}(?:\.\d+)?)\s*%", flags=re.IGNORECASE)
    processing_tiles_re = re.compile(
        r"Processing\s+Tiles:\s*(\d+)\s*/\s*(\d+)(?:.*?\((\d{1,3}(?:\.\d+)?)%\))?",
        flags=re.IGNORECASE,
    )
    ratio_hint_re = re.compile(
        r"\b(?:processed|processing|frame|frames|step|steps|batch|batches)\b[^0-9]{0,20}(\d+)\s*/\s*(\d+)",
        flags=re.IGNORECASE,
    )

    def _format_elapsed(seconds: float) -> str:
        sec = max(0, int(round(float(seconds or 0.0))))
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s"
        if m > 0:
            return f"{m}m {s:02d}s"
        return f"{s}s"

    def _emit_progress_line(line: str, console: bool = True) -> None:
        nonlocal inline_frame_progress_active, inline_frame_progress_width
        payload = str(line or "").rstrip("\r\n")
        if not payload:
            return
        try:
            if not console:
                pass
            elif (
                payload.startswith("FRAME_PROGRESS ")
                or payload.startswith("SparkVSR Progress:")
                or payload.startswith("COMPARISON_PROGRESS")
            ):
                padded = payload
                if inline_frame_progress_width > len(payload):
                    padded = payload + (" " * (inline_frame_progress_width - len(payload)))
                print(f"\r{padded}", end="", flush=True)
                inline_frame_progress_active = True
                inline_frame_progress_width = len(payload)
            else:
                if inline_frame_progress_active:
                    print("", flush=True)
                    inline_frame_progress_active = False
                    inline_frame_progress_width = 0
                print(payload, flush=True)
        except Exception:
            pass
        try:
            on_progress(payload + "\n")
        except Exception:
            pass

    def _parse_chunk_inner_fraction(message: str) -> Optional[float]:
        text = str(message or "").strip()
        if not text:
            return None
        spark_match = spark_progress_re.search(text)
        if spark_match:
            return max(0.0, min(1.0, float(spark_match.group(1)) / 100.0))
        if text.startswith("FRAME_PROGRESS "):
            body = text[len("FRAME_PROGRESS ") :].strip()
            m = frame_progress_re.search(body)
            if m:
                cur = int(m.group(1))
                total = max(1, int(m.group(2)))
                return max(0.0, min(1.0, float(cur) / float(total)))
            p = pct_progress_re.search(body)
            if p:
                return max(0.0, min(1.0, float(p.group(1)) / 100.0))
            return None
        tile_match = processing_tiles_re.search(text)
        if tile_match:
            if tile_match.group(3) is not None:
                return max(0.0, min(1.0, float(tile_match.group(3)) / 100.0))
            cur = max(0, int(tile_match.group(1)) - 1)
            total = max(1, int(tile_match.group(2)))
            return max(0.0, min(1.0, float(cur) / float(total)))
        m = ratio_hint_re.search(text)
        if m:
            cur = int(m.group(1))
            total = max(1, int(m.group(2)))
            return max(0.0, min(1.0, float(cur) / float(total)))
        p = pct_progress_re.search(text)
        if p and any(tok in text.lower() for tok in ("progress", "processing", "processed", "frame", "batch", "step")):
            return max(0.0, min(1.0, float(p.group(1)) / 100.0))
        return None

    def _overall_process_fraction() -> float:
        nonlocal completed_chunk_units, current_chunk_index, current_chunk_inner_fraction
        if total_chunk_units <= 0:
            return 0.0
        current_units = 0.0
        if 1 <= int(current_chunk_index) <= len(chunk_work_units):
            current_units = float(chunk_work_units[int(current_chunk_index) - 1]) * max(
                0.0, min(1.0, float(current_chunk_inner_fraction))
            )
        frac = (float(completed_chunk_units) + current_units) / float(total_chunk_units)
        return max(0.0, min(1.0, frac))

    def _overall_fraction() -> float:
        frac = (
            float(split_stage_weight) * max(0.0, min(1.0, float(split_stage_progress)))
            + float(process_stage_weight) * _overall_process_fraction()
            + float(merge_stage_weight) * max(0.0, min(1.0, float(merge_stage_progress)))
        )
        return max(0.0, min(1.0, frac))

    def _format_eta(eta_seconds: Optional[float]) -> str:
        if eta_seconds is None:
            return "ETA unknown"
        if eta_seconds <= 0:
            return "ETA 0s"
        finish_ts = time.time() + float(eta_seconds)
        finish_local = time.strftime("%H:%M:%S", time.localtime(finish_ts))
        return f"ETA {_format_elapsed(eta_seconds)} (finish ~{finish_local})"

    def _estimate_eta_from_progress(progress_fraction: float) -> Optional[float]:
        nonlocal last_rate_sample_ts, last_rate_sample_progress, ema_progress_rate
        p = max(0.0, min(1.0, float(progress_fraction)))
        if p <= 1e-6:
            return None
        now = time.time()
        elapsed = max(1e-6, now - run_start_ts)
        inst_rate = p / elapsed

        dt = max(0.0, now - float(last_rate_sample_ts))
        dp = max(0.0, p - float(last_rate_sample_progress))
        if dt >= 0.25 and dp >= 0:
            sample_rate = dp / dt if dt > 0 else inst_rate
            if sample_rate > 0:
                if ema_progress_rate is None:
                    ema_progress_rate = sample_rate
                else:
                    ema_progress_rate = (ema_progress_rate * 0.7) + (sample_rate * 0.3)
            last_rate_sample_ts = now
            last_rate_sample_progress = p

        rate = max(inst_rate, float(ema_progress_rate or 0.0))
        if rate <= 1e-9:
            return None
        return max(0.0, (1.0 - p) / rate)

    def _emit_overall_progress(stage_label: str = "", force: bool = False) -> None:
        nonlocal last_overall_emit_ts, last_emitted_fraction
        now = time.time()
        frac = _overall_fraction()
        if not force:
            frac_delta = abs(frac - last_emitted_fraction)
            elapsed_since_emit = now - last_overall_emit_ts
            if frac_delta < 0.01 and elapsed_since_emit < progress_report_interval_sec:
                return
            if frac_delta >= 0.01 and elapsed_since_emit < 0.8:
                return
        elapsed = max(0.0, now - run_start_ts)
        eta_seconds = _estimate_eta_from_progress(frac)
        done = max(0, min(100, int(round(frac * 100.0))))
        line = (
            f"FRAME_PROGRESS {done}/100 | {frac * 100.0:.1f}% | "
            f"elapsed {_format_elapsed(elapsed)} | {_format_eta(eta_seconds)}"
        )
        stage_clean = str(stage_label or "").strip()
        if stage_clean:
            line += f" | {stage_clean}"
        _emit_progress_line(line)
        last_overall_emit_ts = now
        last_emitted_fraction = frac

    _emit_overall_progress("Chunk split complete; starting processing", force=True)

    output_chunks: List[Path] = []
    chunk_logs: List[dict] = []
    custom_process_accepts_kw_on_progress = False
    custom_process_accepts_pos_on_progress = False
    if process_func:
        try:
            sig = inspect.signature(process_func)
            params = list(sig.parameters.values())
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
            has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
            custom_process_accepts_kw_on_progress = has_var_kw or ("on_progress" in sig.parameters)
            positional_count = sum(
                1
                for p in params
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            )
            custom_process_accepts_pos_on_progress = has_var_pos or positional_count >= 2
        except Exception:
            # Safe default: legacy single-arg process_func(settings) invocation.
            custom_process_accepts_kw_on_progress = False
            custom_process_accepts_pos_on_progress = False

    def _get_merge_fps_hint(paths: Optional[List[Path]] = None) -> Optional[float]:
        candidates = list(paths or [])
        if not candidates and output_chunks:
            candidates = list(output_chunks)
        for p in candidates:
            try:
                fps_val = float(get_media_fps(str(p)) or 0.0)
                if fps_val > 0:
                    return fps_val
            except Exception:
                continue
        try:
            base_fps = float(get_media_fps(input_path) or 0.0)
            if base_fps > 0:
                return base_fps
        except Exception:
            pass
        return None

    def _notify_progress(progress_val: float, desc: str, **kwargs) -> None:
        """
        Call the optional `progress_tracker` in a backward-compatible way.

        Some callers expect `progress_tracker(progress_val, desc="...")`, while newer
        callers may accept additional keyword args (chunk paths, indices, etc.).
        """
        if not progress_tracker:
            return
        try:
            progress_tracker(progress_val, desc=desc, **kwargs)
        except TypeError:
            try:
                progress_tracker(progress_val, desc=desc)
            except TypeError:
                try:
                    progress_tracker(progress_val, desc)
                except Exception:
                    pass
        except Exception:
            pass

    def _emit_diag(message: str) -> None:
        nonlocal inline_frame_progress_active, inline_frame_progress_width
        """
        Emit key diagnostics to both console (CMD) and progress callback.
        """
        line = str(message)
        if not line.endswith("\n"):
            line += "\n"
        try:
            if inline_frame_progress_active:
                print("", flush=True)
                inline_frame_progress_active = False
                inline_frame_progress_width = 0
            print(line, end="", flush=True)
        except Exception:
            pass
        try:
            on_progress(line)
        except Exception:
            pass

    def _cleanup_chunk_dirs(preserve_thumbs: bool = True) -> None:
        """
        Best-effort cleanup for chunk artifacts when `per_chunk_cleanup` is enabled.

        We preserve `processed_chunks/thumbs/` by default so the UI gallery can still
        show completed thumbnails even when chunk videos are deleted.
        """
        try:
            shutil.rmtree(input_chunks_dir, ignore_errors=True)
        except Exception:
            pass
        try:
            if not processed_chunks_dir.exists():
                return
            for child in processed_chunks_dir.iterdir():
                if preserve_thumbs and child.is_dir() and child.name == "thumbs":
                    continue
                try:
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
                except Exception:
                    continue
        except Exception:
            pass

    def _resolve_merge_chunks(expected_count: Optional[int] = None) -> List[Path]:
        """
        Resolve chunk outputs for merge using both in-memory paths and processed_chunks/ scan.
        Also waits for each candidate to be fully finalized on disk.
        When an expected count is known, retry for a short window so late-finalizing
        chunk files are included instead of merging only an early subset.
        """
        start_ts = time.time()
        wait_deadline = start_ts + (35.0 if expected_count and expected_count > 1 else 6.0)
        best_ready: List[Path] = []

        while True:
            candidates = _collect_merge_chunk_paths(
                output_chunks,
                processed_dir=processed_chunks_dir,
                expected_count=expected_count,
            )
            ready: List[Path] = []
            for p in candidates:
                idx = _extract_chunk_index(Path(p))
                expected_dur = None
                if idx is not None and 1 <= idx <= len(chunk_paths):
                    try:
                        expected_dur = get_media_duration_seconds(str(chunk_paths[idx - 1]))
                    except Exception:
                        expected_dur = None
                # Short per-candidate wait; outer loop handles longer retries.
                _wait_for_media_file_ready(Path(p), expected_duration=expected_dur, timeout_sec=2.0, poll_sec=0.2)
                if Path(p).exists() and Path(p).is_file():
                    ready.append(Path(p))

            if len(ready) > len(best_ready):
                best_ready = list(ready)

            if expected_count and expected_count > 0 and len(ready) >= int(expected_count):
                return ready[: int(expected_count)]

            if time.time() >= wait_deadline:
                return best_ready

            time.sleep(0.25)

    def _finalize_partial_output(
        *,
        idx: int,
        returncode: int,
        canceled: bool,
        reason: str,
    ) -> Optional[Tuple[int, str, str, int]]:
        """
        Build and return a partial output from completed chunks.
        Returns None when no usable partial could be produced.
        """
        if not (allow_partial and output_chunks):
            return None

        if output_format == "png":
            partial_target = partial_png_target or collision_safe_dir(work_root / "partial_chunks")
            partial_target.mkdir(parents=True, exist_ok=True)
            for i, outp in enumerate(output_chunks, 1):
                dest = partial_target / f"chunk_{i:04d}"
                if Path(outp).is_dir():
                    shutil.copytree(outp, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(outp, dest)
            log_blob = f"Chunking {reason} at chunk {idx}; partial PNG outputs saved to {partial_target}"
            try:
                emit_metadata(
                    partial_target,
                    {
                        "returncode": returncode,
                        "chunks": chunk_logs,
                        "partial": True,
                        "chunk_index": idx,
                        "processed_chunks": len(output_chunks),
                        "canceled": canceled,
                    },
                )
            except Exception:
                pass
            if per_chunk_cleanup:
                _cleanup_chunk_dirs(preserve_thumbs=True)
            return returncode, log_blob, str(partial_target), len(chunk_paths)

        merge_chunks = _resolve_merge_chunks(expected_count=len(output_chunks))
        if not merge_chunks:
            return None
        partial_target = partial_video_target or collision_safe_path(work_root / "partial_concat.mp4")
        merge_fps_hint = _get_merge_fps_hint(merge_chunks) or 30.0
        overlap_frames_for_blend: Any = int(chunk_overlap * merge_fps_hint) if chunk_overlap > 0 else 0
        if chunk_overlap > 0 and exact_boundary_overlaps and len(exact_boundary_overlaps) >= len(merge_chunks) - 1:
            overlap_frames_for_blend = list(exact_boundary_overlaps[: max(0, len(merge_chunks) - 1)])
        ok = concat_videos_with_blending(
            merge_chunks,
            partial_target,
            overlap_frames=overlap_frames_for_blend,
            fps=merge_fps_hint,
            encode_settings=settings,
            on_progress=on_progress,
            nominal_fps=source_fps_rational,
        )
        if ok:
            try:
                _changed, maybe_final, audio_err = ensure_audio_on_video(
                    video_path=Path(partial_target),
                    audio_source_path=Path(audio_source_for_mux),
                    audio_codec=str(settings.get("audio_codec") or "copy"),
                    audio_bitrate=str(settings.get("audio_bitrate")) if settings.get("audio_bitrate") else None,
                    force_replace=True,
                    on_progress=on_progress,
                )
                if maybe_final and Path(maybe_final).exists():
                    partial_target = Path(maybe_final)
                if audio_err:
                    on_progress(f"Audio replacement note: {audio_err}\n")
            except Exception as e:
                on_progress(f"Audio replacement skipped: {str(e)}\n")
            on_progress(f"Partial output stitched to {partial_target}\n")

        meta = {
            "partial": True,
            "chunk_index": idx,
            "returncode": returncode,
            "processed_chunks": len(output_chunks),
            "canceled": canceled,
        }
        log_blob = f"Chunking {reason} at chunk {idx}; partial output saved: {partial_target}\n{meta}"
        try:
            emit_metadata(
                partial_target,
                {
                    "returncode": returncode,
                    "chunks": chunk_logs,
                    "partial": True,
                    "chunk_index": idx,
                    "processed_chunks": len(output_chunks),
                    "canceled": canceled,
                },
            )
        except Exception:
            pass
        if per_chunk_cleanup:
            _cleanup_chunk_dirs(preserve_thumbs=True)
        if ok:
            return returncode, log_blob, str(partial_target), len(chunk_paths)
        return None

    def _largest_4n_plus_1_leq(n: int) -> int:
        if n <= 0:
            return 1
        return max(1, ((int(n) - 1) // 4) * 4 + 1)

    def _count_frames_in_chunk(chunk_path: Path) -> Optional[int]:
        try:
            p = Path(chunk_path)
            if p.is_dir():
                exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
                return sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts)
            if shutil.which("ffprobe") is not None:
                proc = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-count_frames",
                        "-show_entries",
                        "stream=nb_read_frames",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        str(p),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if proc.returncode == 0:
                    raw = (proc.stdout or "").strip()
                    if raw.isdigit():
                        val = int(raw)
                        if val > 0:
                            return val
            try:
                cap = cv2.VideoCapture(str(p))
                if cap.isOpened():
                    val = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    cap.release()
                    if val > 0:
                        return val
            except Exception:
                return None
        except Exception:
            return None
        return None

    def _probe_video_stream_verbose(path: Path) -> Dict[str, str]:
        info: Dict[str, str] = {}
        try:
            proc = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,profile,pix_fmt,codec_tag_string:stream_tags=encoder",
                    "-of",
                    "default=noprint_wrappers=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=12,
            )
            if proc.returncode != 0:
                return info
            for raw in str(proc.stdout or "").splitlines():
                line = str(raw or "").strip()
                if not line or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                key = str(k).strip().lower()
                val = str(v).strip()
                if not val:
                    continue
                if key.startswith("tag:"):
                    key = key[4:]
                info[key] = val
        except Exception:
            return info
        return info

    def _log_chunk_codec_probe(chunk_idx: int, label: str, media_path: Path) -> None:
        try:
            p = Path(media_path)
            if not p.exists() or not p.is_file():
                _emit_diag(f"[chunk {chunk_idx}] codec probe ({label}): missing file: {p}\n")
                return
            v = _probe_video_stream_verbose(p)
            codec = str(v.get("codec_name") or "unknown").strip().lower()
            profile = str(v.get("profile") or "unknown").strip()
            pix_fmt = str(v.get("pix_fmt") or "unknown").strip().lower()
            encoder = str(v.get("encoder") or "unknown").strip()
            has_aud = has_audio_stream(p)
            _emit_diag(
                f"[chunk {chunk_idx}] codec probe ({label}): "
                f"codec={codec}, profile={profile}, pix_fmt={pix_fmt}, encoder={encoder}, has_audio={has_aud}\n"
            )
        except Exception as e:
            _emit_diag(f"[chunk {chunk_idx}] codec probe ({label}) failed: {str(e)}\n")

    expected_encode = _normalize_video_encode_settings(settings)
    expected_codec_key = str(expected_encode.get("codec") or "").strip().lower()
    expected_codec_name = {
        "h264": "h264",
        "h265": "hevc",
        "vp9": "vp9",
        "av1": "av1",
        "prores": "prores",
    }.get(expected_codec_key, "")
    expected_use_10bit = bool(expected_encode.get("use_10bit", False)) and expected_codec_key in {
        "h264",
        "h265",
        "vp9",
        "av1",
    }
    strict_codec_validation = True
    if model_type == "seedvr2":
        # SeedVR2 with OpenCV backend can legitimately emit mp4v/mpeg4 regardless of
        # global output codec preference. Enforcing strict codec checks here creates
        # false "codec drift" failures even when processing succeeds.
        seed_backend = str(settings.get("video_backend") or "").strip().lower()
        if seed_backend != "ffmpeg":
            strict_codec_validation = False
            _emit_diag(
                f"[codec] strict validation disabled for SeedVR2 backend='{seed_backend or 'opencv'}' "
                "(OpenCV output codec may differ by runtime build).\n"
            )
    elif model_type == "flashvsr":
        # FlashVSR backends can emit different codecs depending on runtime path:
        # - Legacy FlashVSR_plus often emits h264
        # - ComfyUI-FlashVSR_Stable CLI can emit mpeg4/mp4v via OpenCV writer
        # Enforcing strict codec equality here causes false failures even when the
        # chunk output is valid and mergeable.
        strict_codec_validation = False
        _emit_diag(
            "[codec] strict validation disabled for FlashVSR "
            f"(backend output codec may vary by runtime; requested codec={expected_codec_name or 'auto'}, "
            f"10bit={expected_use_10bit}).\n"
        )
    elif model_type == "sparkvsr":
        # SparkVSR writes MP4 internally via imageio/libx264, while the app-level
        # Output tab may request a different final codec. Chunk merge validation
        # should only require valid/mergeable chunk media here.
        strict_codec_validation = False
        _emit_diag(
            "[codec] strict validation disabled for SparkVSR "
            f"(backend output codec may vary; requested codec={expected_codec_name or 'auto'}, "
            f"10bit={expected_use_10bit}).\n"
        )
    elif model_type == "rtx":
        # RTX Super Resolution currently writes MP4 via OpenCV's mp4v path in the runner.
        # Enforcing strict equality against requested/source codec (often h264) can produce
        # false codec-drift failures even when chunk output is valid and mergeable.
        strict_codec_validation = False
        _emit_diag(
            "[codec] strict validation disabled for RTX Super Resolution "
            f"(runner output codec may differ by OpenCV runtime; requested codec={expected_codec_name or 'auto'}, "
            f"10bit={expected_use_10bit}).\n"
        )
    elif model_type == "ltx25":
        # The LTX 2.5 engine writes MP4 itself (ffmpeg libx264 by default) while
        # the Output tab may request a different final codec for the merge step.
        strict_codec_validation = False
        _emit_diag(
            "[codec] strict validation disabled for LTX 2.5 "
            f"(engine writes its own MP4; requested codec={expected_codec_name or 'auto'}, "
            f"10bit={expected_use_10bit}).\n"
        )
    if output_format != "png" and expected_codec_name and strict_codec_validation:
        _emit_diag(
            f"[codec] expected output codec={expected_codec_name}, 10bit={expected_use_10bit}\n"
        )

    def _codec_matches_expected(media_path: Path) -> bool:
        if output_format == "png" or not expected_codec_name or not strict_codec_validation:
            return True
        v = _probe_video_stream_verbose(Path(media_path))
        codec = str(v.get("codec_name") or "").strip().lower()
        pix_fmt = str(v.get("pix_fmt") or "").strip().lower()
        if expected_codec_name == "prores":
            codec_ok = codec.startswith("prores")
        else:
            codec_ok = codec == expected_codec_name
        if not codec_ok:
            return False
        if expected_use_10bit and "10" not in pix_fmt:
            return False
        return True

    def _find_expected_codec_sibling(media_path: Path) -> Optional[Path]:
        try:
            p = Path(media_path)
            if not p.exists() and not p.parent.exists():
                return None
            suffix = p.suffix if p.suffix else ".mp4"
            stem = p.stem
            candidates = sorted(
                p.parent.glob(f"{stem}*{suffix}"),
                key=lambda x: x.stat().st_mtime if x.exists() else 0.0,
                reverse=True,
            )
            for cand in candidates:
                if not cand.exists() or not cand.is_file():
                    continue
                name_lc = cand.name.lower()
                if "__audio_tmp" in name_lc or "__noaudio_tmp" in name_lc:
                    continue
                try:
                    if cand.resolve() == p.resolve():
                        continue
                except Exception:
                    if str(cand) == str(p):
                        continue
                if _codec_matches_expected(cand):
                    return cand
        except Exception:
            return None
        return None

    def _ensure_expected_chunk_codec(
        chunk_idx: int,
        label: str,
        media_path: Path,
    ) -> Tuple[bool, Path]:
        p = Path(media_path)
        _log_chunk_codec_probe(chunk_idx, label, p)
        if output_format == "png" or not expected_codec_name or not strict_codec_validation:
            return True, p
        if _codec_matches_expected(p):
            return True, p
        alt = _find_expected_codec_sibling(p)
        if alt is not None:
            _emit_diag(
                f"[chunk {chunk_idx}] codec mismatch at {label}; "
                f"switching to sibling output: {alt.name}\n"
            )
            _log_chunk_codec_probe(chunk_idx, f"{label}/sibling", alt)
            if _codec_matches_expected(alt):
                return True, alt
        _emit_diag(
            f"[chunk {chunk_idx}] ERROR: codec drift at {label}. "
            f"Expected codec={expected_codec_name}, 10bit={expected_use_10bit}.\n"
        )
        return False, p

    # If resuming, load existing completed chunks and skip them
    if resuming and existing_chunks:
        validated_existing_chunks: List[Path] = []
        for i, chunk_path in enumerate(existing_chunks, 1):
            ok_codec, resolved_chunk = _ensure_expected_chunk_codec(
                i,
                "resume_existing",
                Path(chunk_path),
            )
            if not ok_codec:
                return (
                    1,
                    f"Resume blocked: existing chunk {i} does not match requested output codec settings.",
                    str(chunk_path),
                    len(chunk_paths),
                )
            # A previously processed chunk is only reusable if it matches the chunk that the
            # CURRENT split produced (chunk outputs made by an older/buggy split or a different
            # chunking setup would silently desync the merged output).
            if output_format != "png":
                if i > len(chunk_paths):
                    _emit_diag(
                        f"Resume: previous run has more chunk outputs ({len(existing_chunks)}) than the current "
                        f"split ({len(chunk_paths)}); ignoring extra outputs from chunk {i}." + "\n"
                    )
                    break
                if Path(chunk_paths[i - 1]).is_file():
                    match_ok, match_detail = _processed_chunk_matches_input(
                        Path(resolved_chunk), Path(chunk_paths[i - 1]), model_type, source_fps_float
                    )
                    if not match_ok:
                        _emit_diag(
                            f"Resume: chunk {i} output from the previous run does not match its input chunk "
                            f"({match_detail}); reprocessing from chunk {i} onward." + "\n"
                        )
                        break
            validated_existing_chunks.append(Path(resolved_chunk))
            chunk_logs.append({
                "chunk_index": i,
                "input": "resumed",
                "output": str(resolved_chunk),
                "returncode": 0,
                "resumed": True,
            })
        output_chunks = validated_existing_chunks.copy()
        start_chunk_idx = len(validated_existing_chunks)
        if chunk_work_units:
            completed_chunk_units = float(sum(chunk_work_units[: len(validated_existing_chunks)]))
        current_chunk_index = 0
        current_chunk_inner_fraction = 0.0
        on_progress(f"✅ Loaded {len(validated_existing_chunks)} completed chunks from previous run - skipping to chunk {start_chunk_idx + 1}\n")
        _emit_overall_progress(
            f"Resumed {len(validated_existing_chunks)}/{len(chunk_paths)} chunks",
            force=True,
        )
        for i, chunk_path in enumerate(validated_existing_chunks, 1):
            _notify_progress(
                i / max(1, len(chunk_paths)),
                desc=f"Completed chunk {i}/{len(chunk_paths)} (resumed)",
                chunk_index=i,
                chunk_total=len(chunk_paths),
                chunk_output=str(chunk_path),
                resumed=True,
            )

    if callable(pre_process_chunks_func) and start_chunk_idx < len(chunk_paths):
        pending_chunks = list(chunk_paths[start_chunk_idx:])
        try:
            on_progress(
                f"Preparing per-chunk reference assets for {len(pending_chunks)} chunk(s) before processing\n"
            )
        except Exception:
            pass
        _notify_progress(
            max(0.0, float(start_chunk_idx) / max(1, len(chunk_paths))),
            desc="Preparing per-chunk reference assets",
            chunk_index=max(1, start_chunk_idx + 1),
            chunk_total=len(chunk_paths),
            phase="reference_prepass",
        )
        try:
            pre_process_chunks_func(pending_chunks, start_chunk_idx, on_progress)
        except Exception as exc:
            msg = f"Reference prepass failed before chunk processing: {exc}"
            try:
                on_progress(f"{msg}\n")
            except Exception:
                pass
            return 1, msg, "", len(chunk_paths)
        _emit_overall_progress("Reference prepass complete; starting chunk processing", force=True)

    for idx, chunk in enumerate(chunk_paths[start_chunk_idx:], start_chunk_idx + 1):
        current_chunk_index = int(idx)
        current_chunk_inner_fraction = 0.0
        _emit_overall_progress(f"Processing chunk {idx}/{len(chunk_paths)}", force=True)
        # Respect external cancellation
        try:
            if getattr(runner, "is_canceled", lambda: False)():
                partial = _finalize_partial_output(
                    idx=idx,
                    returncode=1,
                    canceled=True,
                    reason="canceled",
                )
                if partial:
                    return partial
                _emit_overall_progress(
                    f"Canceled before chunk {idx}/{len(chunk_paths)} started",
                    force=True,
                )
                return 1, "Canceled before processing current chunk", "", len(chunk_paths)
        except Exception:
            pass
        # Emit in-progress state before running the chunk so UI can show "processing chunk X/Y".
        _notify_progress(
            max(0.0, (idx - 1) / max(1, len(chunk_paths))),
            desc=f"Processing chunk {idx}/{len(chunk_paths)}",
            chunk_index=idx,
            chunk_total=len(chunk_paths),
            chunk_input=str(chunk),
            phase="processing",
        )
        chunk_settings = settings.copy()
        chunk_settings["input_path"] = str(chunk)
        # Some pipelines (e.g., FlashVSR+) support preprocessing via `_effective_input_path`.
        # Ensure per-chunk runs always point to the chunk itself.
        chunk_settings["_effective_input_path"] = str(chunk)
        # FlashVSR's CLI otherwise takes the output frame rate from OpenCV's CAP_PROP_FPS of the
        # chunk file, which is derived from container metadata and differs between ffmpeg
        # builds (e.g. 25.098 or 24.49 instead of 25). For CFR sources pass the exact source
        # frame rate explicitly so every chunk output has the correct timing.
        if model_type == "flashvsr" and source_fps_float and source_fps_float > 0:
            try:
                user_fps_override = float(chunk_settings.get("fps") or 0.0)
            except Exception:
                user_fps_override = 0.0
            if user_fps_override <= 0:
                chunk_settings["fps"] = float(source_fps_float)
        # Direct per-chunk outputs to the run folder (processed_chunks/).
        if output_format == "png":
            chunk_settings["output_override"] = str(processed_chunks_dir / f"{chunk.stem}_upscaled")
        else:
            chunk_settings["output_override"] = str(processed_chunks_dir / f"{chunk.stem}_upscaled.mp4")

        # Safety: SeedVR2 batch_size can exceed very short chunk lengths (e.g., user batch_size=29, chunk=14 frames).
        # Clamp per-chunk batch_size to the largest valid 4n+1 <= frame_count to avoid runtime errors.
        if model_type == "seedvr2":
            try:
                user_bs = int(chunk_settings.get("batch_size") or 0)
            except Exception:
                user_bs = 0
            if user_bs > 0:
                frame_count = _count_frames_in_chunk(Path(chunk))
                if frame_count and frame_count > 0 and user_bs > int(frame_count):
                    adj = _largest_4n_plus_1_leq(int(frame_count))
                    if adj != user_bs:
                        chunk_settings["batch_size"] = adj
                        try:
                            on_progress(f"Adjusting SeedVR2 batch_size {user_bs}->{adj} for short chunk ({frame_count} frames)\n")
                        except Exception:
                            pass

        def _chunk_progress_proxy(message: str) -> None:
            nonlocal current_chunk_inner_fraction
            text = str(message or "")
            stripped = text.strip()
            if not stripped:
                return
            parsed_fraction = _parse_chunk_inner_fraction(stripped)
            if parsed_fraction is not None:
                current_chunk_inner_fraction = max(
                    float(current_chunk_inner_fraction),
                    max(0.0, min(1.0, float(parsed_fraction))),
                )
                _emit_overall_progress(
                    f"Processing chunk {idx}/{len(chunk_paths)}",
                    force=False,
                )
            elif stripped.lower().startswith("[sparkvsr] still running"):
                _emit_overall_progress(
                    f"Processing chunk {idx}/{len(chunk_paths)} - {stripped}",
                    force=False,
                )
            # Replace chunk-local FRAME_PROGRESS with overall progress to keep UI/CMD consistent.
            if stripped.startswith("FRAME_PROGRESS "):
                return
            # FlashVSR's runner (and its model downloader) already echo every line
            # to the console themselves; skip the local echo to avoid double printing.
            _emit_progress_line(text, console=(model_type != "flashvsr"))
        
        # Use provided processing function or select based on model type
        if process_func:
            if custom_process_accepts_kw_on_progress:
                res = process_func(chunk_settings, on_progress=_chunk_progress_proxy)
            elif custom_process_accepts_pos_on_progress:
                res = process_func(chunk_settings, _chunk_progress_proxy)
            else:
                res = process_func(chunk_settings)
        elif model_type == "seedvr2":
            res = runner.run_seedvr2(chunk_settings, on_progress=_chunk_progress_proxy, preview_only=False)
        elif model_type == "gan":
            # Forward GAN progress so frame-level runtime status can be shown in the UI.
            res = runner.run_gan(chunk_settings, on_progress=_chunk_progress_proxy)
        elif model_type == "rife":
            res = runner.run_rife(chunk_settings, on_progress=_chunk_progress_proxy)
        elif model_type == "flashvsr":
            if hasattr(runner, "run_flashvsr"):
                res = runner.run_flashvsr(chunk_settings, on_progress=_chunk_progress_proxy)
            else:
                raise AttributeError("chunk_and_process: model_type='flashvsr' requires runner.run_flashvsr() or a custom process_func")
        elif model_type == "sparkvsr":
            if hasattr(runner, "run_sparkvsr"):
                res = runner.run_sparkvsr(chunk_settings, on_progress=_chunk_progress_proxy)
            else:
                raise AttributeError("chunk_and_process: model_type='sparkvsr' requires runner.run_sparkvsr() or a custom process_func")
        else:
            # Fallback to seedvr2 for backward compatibility
            res = runner.run_seedvr2(chunk_settings, on_progress=_chunk_progress_proxy, preview_only=False)

        if res.returncode != 0 or getattr(runner, "is_canceled", lambda: False)():
            on_progress(f"Chunk {idx} failed with code {res.returncode}\n")
            try:
                if int(getattr(res, "returncode", 0) or 0) != 0:
                    err_blob = str(getattr(res, "log", "") or "").strip()
                    if err_blob:
                        tail_lines = [ln for ln in err_blob.splitlines() if str(ln).strip()]
                        if tail_lines:
                            on_progress(f"[chunk {idx}] error details (tail):\n")
                            for ln in tail_lines[-12:]:
                                on_progress(f"[chunk {idx}] {ln}\n")
            except Exception:
                pass
            is_canceled_now = bool(getattr(runner, "is_canceled", lambda: False)())
            partial_returncode = res.returncode if res.returncode != 0 else 1
            partial = _finalize_partial_output(
                idx=idx,
                returncode=partial_returncode,
                canceled=is_canceled_now,
                reason="canceled" if is_canceled_now else "stopped early",
            )
            _emit_overall_progress(
                f"{'Canceled' if is_canceled_now else 'Failed'} at chunk {idx}/{len(chunk_paths)}",
                force=True,
            )
            if partial:
                return partial
            return res.returncode, res.log, res.output_path or "", len(chunk_paths)
        outp: Optional[Path] = Path(res.output_path) if res.output_path else None
        if outp:
            expected_chunk_duration = None
            try:
                if Path(chunk).is_file():
                    expected_chunk_duration = get_media_duration_seconds(str(chunk))
            except Exception:
                expected_chunk_duration = None

            if not _wait_for_media_file_ready(
                outp,
                expected_duration=expected_chunk_duration,
                timeout_sec=25.0,
            ):
                # Fallback discovery in case the runner returned early with a predictable path.
                fallback_candidates = [
                    processed_chunks_dir / f"{Path(chunk).stem}_upscaled.mp4",
                    processed_chunks_dir / f"{Path(chunk).stem}_out.mp4",
                ]
                for cand in fallback_candidates:
                    if _wait_for_media_file_ready(
                        cand,
                        expected_duration=expected_chunk_duration,
                        timeout_sec=8.0,
                    ):
                        outp = cand
                        break

            if outp.exists() and outp.is_file():
                codec_ok, resolved_outp = _ensure_expected_chunk_codec(
                    idx,
                    "post_model",
                    Path(outp),
                )
                if not codec_ok:
                    return (
                        1,
                        f"Chunk {idx} codec drift detected immediately after model output.",
                        str(outp),
                        len(chunk_paths),
                    )
                outp = Path(resolved_outp)
                # Keep each processed chunk muxed with its own source-chunk audio.
                # Video stays bit-exact via -c:v copy in ensure_audio_on_video/mux_audio.
                if output_format != "png" and Path(chunk).is_file():
                    try:
                        on_progress(
                            f"[chunk {idx}] Post-processing audio transfer "
                            f"(codec={str(settings.get('audio_codec') or 'copy')})...\n"
                        )
                        _changed, maybe_chunk_final, chunk_audio_err = ensure_audio_on_video(
                            video_path=Path(outp),
                            audio_source_path=Path(chunk),
                            audio_codec=str(settings.get("audio_codec") or "copy"),
                            audio_bitrate=str(settings.get("audio_bitrate")) if settings.get("audio_bitrate") else None,
                            force_replace=True,
                            on_progress=on_progress,
                        )
                        if maybe_chunk_final and Path(maybe_chunk_final).exists():
                            outp = Path(maybe_chunk_final)
                        if chunk_audio_err:
                            on_progress(f"[chunk {idx}] Audio transfer note: {chunk_audio_err}\n")
                    except Exception as e:
                        on_progress(f"[chunk {idx}] Audio transfer skipped: {str(e)}\n")
                codec_ok, resolved_outp = _ensure_expected_chunk_codec(
                    idx,
                    "post_audio",
                    Path(outp),
                )
                if not codec_ok:
                    return (
                        1,
                        f"Chunk {idx} codec drift detected after audio transfer.",
                        str(outp),
                        len(chunk_paths),
                    )
                outp = Path(resolved_outp)
                try:
                    st = Path(outp).stat()
                    saved_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
                    _emit_diag(
                        f"[chunk {idx}] final chunk file saved: "
                        f"name={Path(outp).name}, size={st.st_size}, mtime={saved_ts}\n"
                    )
                except Exception:
                    _emit_diag(f"[chunk {idx}] final chunk file saved: name={Path(outp).name}\n")
                _log_chunk_codec_probe(idx, "ready_for_next_chunk", Path(outp))
                # The chunk output must line up with the chunk input frame-for-frame in time;
                # otherwise the merged video drifts against the original audio.
                if output_format != "png" and Path(chunk).is_file():
                    match_ok, match_detail = _processed_chunk_matches_input(
                        Path(outp), Path(chunk), model_type, source_fps_float
                    )
                    if match_ok:
                        _emit_diag(f"[chunk {idx}] output timing check OK: {match_detail}" + "\n")
                    elif str(model_type or "").strip().lower() in _STRICT_TIMING_MODELS:
                        _emit_diag(
                            f"[chunk {idx}] ERROR: model output does not match the input chunk timing: {match_detail}. "
                            "Merging it would desync audio/video, so processing is stopped." + "\n"
                        )
                        return (
                            1,
                            f"Chunk {idx} output timing mismatch ({model_type}): {match_detail}",
                            str(outp),
                            len(chunk_paths),
                        )
                    else:
                        _emit_diag(
                            f"[chunk {idx}] WARNING: model output does not match the input chunk timing: {match_detail}. "
                            "The merged output may drift against the original audio (see the final duration check)." + "\n"
                        )
                output_chunks.append(outp)
            else:
                try:
                    on_progress(f"WARN: Chunk {idx} output missing/unready at merge stage: {res.output_path}\n")
                except Exception:
                    pass
        # Update progress only after successful chunk completion, include paths for UI preview.
        _notify_progress(
            idx / max(1, len(chunk_paths)),
            desc=f"Completed chunk {idx}/{len(chunk_paths)}",
            chunk_index=idx,
            chunk_total=len(chunk_paths),
            chunk_input=str(chunk),
            chunk_output=str(outp) if outp else None,
            output_format=str(output_format),
            phase="completed",
        )
        current_chunk_inner_fraction = 1.0
        _emit_overall_progress(f"Completed chunk {idx}/{len(chunk_paths)}", force=True)
        if idx - 1 < len(chunk_work_units):
            completed_chunk_units = min(
                float(total_chunk_units),
                float(completed_chunk_units) + float(chunk_work_units[idx - 1]),
            )
        current_chunk_inner_fraction = 0.0
        chunk_logs.append(
            {
                "chunk_index": idx,
                "input": str(chunk),
                "output": str(outp) if outp else (res.output_path or None),
                "returncode": res.returncode,
            }
        )

        # Optional: free disk space by deleting the *input* chunk file after it is processed.
        # This is safe because we only concatenate processed outputs, not the split inputs.
        if per_chunk_cleanup:
            try:
                chunk_path = Path(chunk)
                if chunk_path.is_file():
                    in_root = input_chunks_dir.resolve()
                    try:
                        parent_resolved = chunk_path.resolve().parent
                    except Exception:
                        parent_resolved = chunk_path.parent
                    if parent_resolved == in_root:
                        chunk_path.unlink(missing_ok=True)
            except Exception:
                pass

    if output_format == "png":
        # Aggregate chunk PNG outputs into a collision-safe parent directory
        target_dir = resolve_output_location(
            input_path=input_path,
            output_format="png",
            global_output_dir=global_override,
            batch_mode=False,
            png_padding=settings.get("png_padding"),
            png_keep_basename=settings.get("png_keep_basename", False),
            original_filename=settings.get("_original_filename"),
        )
        target_dir = collision_safe_dir(Path(target_dir))
        target_dir.mkdir(parents=True, exist_ok=True)
        pad_val = max(1, int(settings.get("png_padding") or 5))
        for i, outp in enumerate(output_chunks, 1):
            dest = target_dir / f"chunk_{i:0{pad_val}d}"
            if Path(outp).is_dir():
                shutil.copytree(outp, dest, dirs_exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(outp, dest)
        if per_chunk_cleanup:
            _cleanup_chunk_dirs(preserve_thumbs=True)
        log_blob = "Chunked processing complete (PNG)\n" + "\n".join([str(c) for c in chunk_logs])
        try:
            emit_metadata(
                target_dir,
                {
                    "returncode": 0,
                    "chunks": chunk_logs,
                    "partial": False,
                    "output_format": output_format,
                },
            )
        except Exception:
            pass
        merge_stage_progress = 1.0
        _emit_overall_progress("Chunk processing complete", force=True)
        return 0, log_blob, str(target_dir), len(chunk_paths)

    if explicit_final_path is not None:
        final_path = collision_safe_path(explicit_final_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        final_path = resolve_output_location(
            input_path=input_path,
            output_format="mp4",
            global_output_dir=global_override,
            batch_mode=False,
            png_padding=settings.get("png_padding"),
            png_keep_basename=settings.get("png_keep_basename", False),
            original_filename=settings.get("_original_filename"),
        )
        final_path = collision_safe_path(Path(final_path))
    
    merge_stage_progress = 0.05
    _emit_overall_progress("Preparing chunk merge", force=True)
    merge_chunks = _resolve_merge_chunks(expected_count=len(chunk_paths))
    if not merge_chunks:
        _emit_overall_progress("Merge failed: no mergeable chunks", force=True)
        return 1, "Concat failed: no mergeable chunk outputs were found", str(final_path), len(chunk_paths)
    if len(merge_chunks) < len(chunk_paths):
        _emit_overall_progress("Merge failed: missing chunk outputs", force=True)
        return (
            1,
            f"Concat failed: discovered {len(merge_chunks)}/{len(chunk_paths)} chunk outputs; refusing best-effort merge.",
            str(final_path),
            len(chunk_paths),
        )
    validated_merge_chunks: List[Path] = []
    for merge_idx, merge_path in enumerate(merge_chunks, 1):
        idx_hint = _extract_chunk_index(Path(merge_path)) or merge_idx
        ok_codec, resolved_merge = _ensure_expected_chunk_codec(
            int(idx_hint),
            "pre_merge",
            Path(merge_path),
        )
        if not ok_codec:
            return (
                1,
                f"Concat blocked: chunk {idx_hint} codec mismatch before merge.",
                str(final_path),
                len(chunk_paths),
            )
        validated_merge_chunks.append(Path(resolved_merge))
    merge_chunks = validated_merge_chunks

    # Use blending concat if overlap specified.
    merge_fps_hint = _get_merge_fps_hint(merge_chunks) or 30.0
    overlap_frames_for_blend: Any = int(chunk_overlap * merge_fps_hint) if chunk_overlap > 0 else 0
    if chunk_overlap > 0 and exact_boundary_overlaps and len(exact_boundary_overlaps) == len(merge_chunks) - 1:
        overlap_frames_for_blend = list(exact_boundary_overlaps)
    merge_stage_progress = 0.30
    _emit_overall_progress("Merging processed chunks", force=True)
    ok = concat_videos_with_blending(
        merge_chunks,
        final_path,
        overlap_frames=overlap_frames_for_blend,
        fps=merge_fps_hint,
        encode_settings=settings,
        on_progress=on_progress,
        nominal_fps=source_fps_rational,
    )
    
    if not ok:
        _emit_overall_progress("Merge failed", force=True)
        return 1, "Concat failed", str(final_path), len(chunk_paths)
    merge_stage_progress = 0.70
    _emit_overall_progress("Chunk merge complete, applying audio", force=True)
    on_progress(f"Chunks concatenated with blending to {final_path}\n")
    final_codec_ok, _final_probe_path = _ensure_expected_chunk_codec(
        0,
        "post_merge_video",
        Path(final_path),
    )
    if not final_codec_ok:
        return (
            1,
            "Concat failed: merged video codec drift detected before final audio mux.",
            str(final_path),
            len(chunk_paths),
        )

    # Sanity check: the merged video must have the source's duration. Any drift here means
    # frames were duplicated or lost in the chunk pipeline; say so loudly instead of letting
    # the audio mux silently hide it.
    try:
        merged_dur = _probe_video_stream_duration(Path(final_path))
        source_dur = _probe_video_stream_duration(Path(input_path))
        if merged_dur and source_dur:
            fps_hint_for_tol = float(merge_fps_hint or 0.0) or 30.0
            dur_tol = max(0.25, 2.0 / fps_hint_for_tol)
            dur_delta = float(merged_dur) - float(source_dur)
            if abs(dur_delta) > dur_tol:
                _emit_diag(
                    "WARNING: merged video duration "
                    f"{float(merged_dur):.3f}s differs from source {float(source_dur):.3f}s "
                    f"({dur_delta:+.3f}s). Audio/video may drift; the video will NOT be truncated. "
                    "Check the per-chunk frame counts above.\n"
                )
            else:
                _emit_diag(
                    f"Merged video duration check OK: {float(merged_dur):.3f}s vs source {float(source_dur):.3f}s.\n"
                )
    except Exception:
        pass

    # Audio normalization for merged output using user-configured codec/bitrate.
    # This is robust: if source has no audio, output remains valid.
    try:
        on_progress(f"Replacing audio from original input (codec={str(settings.get('audio_codec') or 'copy')})...\n")
        _changed, maybe_final, audio_err = ensure_audio_on_video(
            video_path=Path(final_path),
            audio_source_path=Path(audio_source_for_mux),
            audio_codec=str(settings.get("audio_codec") or "copy"),
            audio_bitrate=str(settings.get("audio_bitrate")) if settings.get("audio_bitrate") else None,
            force_replace=True,
            on_progress=on_progress,
        )
        if maybe_final and Path(maybe_final).exists():
            final_path = Path(maybe_final)
        if audio_err:
            on_progress(f"Audio replacement note: {audio_err}\n")
    except Exception as e:
        # Never fail the whole operation due to audio issues
        on_progress(f"Audio replacement skipped: {str(e)}\n")
    final_codec_ok, _final_probe_path = _ensure_expected_chunk_codec(
        0,
        "post_merge_audio",
        Path(final_path),
    )
    if not final_codec_ok:
        _emit_overall_progress("Final output validation failed", force=True)
        return (
            1,
            "Final output codec drift detected after audio mux.",
            str(final_path),
            len(chunk_paths),
        )
    merge_stage_progress = 1.0
    _emit_overall_progress("Chunked processing complete", force=True)
    if per_chunk_cleanup:
        _cleanup_chunk_dirs(preserve_thumbs=True)
    # Write chunk metadata
    meta_path = final_path.parent / f"{final_path.stem}_chunk_metadata.json"
    try:
        import json
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(chunk_logs, f, indent=2)
    except Exception:
        pass
    log_blob = "Chunked processing complete\n" + "\n".join([str(c) for c in chunk_logs])
    # Emit consolidated metadata for chunked runs
    try:
        emit_metadata(
            final_path,
            {
                "returncode": 0,
                "chunks": chunk_logs,
                "partial": False,
                "output_format": output_format,
            },
        )
    except Exception:
        pass
    return 0, log_blob, str(final_path), len(chunk_paths)
