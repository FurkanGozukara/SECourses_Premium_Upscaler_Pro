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

    # Filter invalid scenes and optionally frame-align boundaries for precision.
    fps_for_align = float(get_media_fps(video_path) or 30.0) if precise else 0.0

    normalized_scenes: List[Tuple[float, float]] = []
    for start, end in scenes:
        try:
            start_f = float(start)
            end_f = float(end)
        except Exception:
            continue

        if precise and fps_for_align and fps_for_align > 0:
            # Align to frame boundaries to avoid float rounding drift and ensure frame-level accuracy.
            # Use floor for start and ceil for end to avoid gaps.
            start_frame = int(math.floor(start_f * fps_for_align + 1e-9))
            end_frame = int(math.ceil(end_f * fps_for_align - 1e-9))
            if end_frame <= start_frame:
                end_frame = start_frame + 1
            start_f = max(0.0, start_frame / fps_for_align)
            end_f = max(start_f, end_frame / fps_for_align)

        if (end_f - start_f) > 0:
            normalized_scenes.append((start_f, end_f))

    if not normalized_scenes:
        return [Path(video_path)]

    src_has_audio = has_audio_stream(Path(video_path)) if include_audio else False
    src_pix_fmt = _probe_pix_fmt(video_path) if preserve_quality else None
    for idx, (start_f, end_f) in enumerate(normalized_scenes, 1):
        out = work_dir / f"chunk_{idx:04d}.mp4"
        duration = max(0.0, end_f - start_f)
        if duration <= 0:
            continue

        def _run_ffmpeg(cmd: List[str]) -> None:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def _split_copy() -> None:
            # IMPORTANT: `-ss` must be BEFORE `-i` when stream-copying or ffmpeg can output empty files.
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_f),
                "-i",
                video_path,
                "-t",
                str(duration),
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "+faststart",
                str(out),
            ]
            _run_ffmpeg(cmd)

        def _split_copy_video_only() -> None:
            # Stream-copy video only. Useful when audio codecs cannot be muxed into MP4.
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_f),
                "-i",
                video_path,
                "-t",
                str(duration),
                "-map",
                "0:v:0",
                "-c:v",
                "copy",
                "-an",
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "+faststart",
                str(out),
            ]
            _run_ffmpeg(cmd)

        def _split_copy_aac_audio() -> None:
            # Stream-copy video but re-encode audio to AAC for MP4 compatibility.
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_f),
                "-i",
                video_path,
                "-t",
                str(duration),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "+faststart",
                str(out),
            ]
            _run_ffmpeg(cmd)

        def _split_precise_lossless(pix_fmt: Optional[str]) -> None:
            # Frame-accurate trimming requires re-encoding (stream-copy is keyframe-limited).
            # Use lossless x264 to preserve input quality as much as possible.
            cmd = [
                "ffmpeg",
                "-y",
                "-fflags",
                "+genpts",
                "-ss",
                str(start_f),
                "-i",
                video_path,
                "-t",
                str(duration),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-vf",
                "setpts=PTS-STARTPTS",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-qp",
                "0",
                "-c:a",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "+faststart",
            ]
            if pix_fmt:
                cmd += ["-pix_fmt", pix_fmt]
            cmd += [str(out)]
            _run_ffmpeg(cmd)

        def _split_precise_lossless_aac_audio(pix_fmt: Optional[str]) -> None:
            # Frame-accurate trimming with lossless video + AAC audio (robust across containers/codecs).
            cmd = [
                "ffmpeg",
                "-y",
                "-fflags",
                "+genpts",
                "-ss",
                str(start_f),
                "-i",
                video_path,
                "-t",
                str(duration),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-vf",
                "setpts=PTS-STARTPTS",
                "-af",
                "asetpts=PTS-STARTPTS",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-qp",
                "0",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "+faststart",
            ]
            if pix_fmt:
                cmd += ["-pix_fmt", pix_fmt]
            cmd += [str(out)]
            _run_ffmpeg(cmd)

        def _split_precise_lossless_video_only(pix_fmt: Optional[str]) -> None:
            # Lossless re-encode video only. Useful when audio codecs cannot be muxed into MP4.
            cmd = [
                "ffmpeg",
                "-y",
                "-fflags",
                "+genpts",
                "-ss",
                str(start_f),
                "-i",
                video_path,
                "-t",
                str(duration),
                "-map",
                "0:v:0",
                "-vf",
                "setpts=PTS-STARTPTS",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-qp",
                "0",
                "-an",
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "+faststart",
            ]
            if pix_fmt:
                cmd += ["-pix_fmt", pix_fmt]
            cmd += [str(out)]
            _run_ffmpeg(cmd)

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

        # Strategy:
        # - precise=True: prefer lossless re-encode (frame-accurate), fall back to stream copy.
        # - precise=False: prefer stream copy (bit-exact), fall back to lossless re-encode if needed.
        try:
            out.unlink(missing_ok=True)
        except Exception:
            pass

        if on_progress:
            mode = "precise-lossless" if precise else "stream-copy"
            on_progress(f"Splitting chunk {idx}/{len(scenes)} ({mode})...\n")

        if precise:
            if include_audio:
                _split_precise_lossless(src_pix_fmt)
                if not _ok_with_audio() and src_pix_fmt:
                    # Retry without forcing pixel format (better compatibility with non-x264 pix_fmts).
                    try:
                        out.unlink(missing_ok=True)
                    except Exception:
                        pass
                    _split_precise_lossless(None)
                if not _ok_with_audio():
                    # Audio-copy can fail on some codecs/containers; retry with AAC audio.
                    try:
                        out.unlink(missing_ok=True)
                    except Exception:
                        pass
                    _split_precise_lossless_aac_audio(src_pix_fmt)
                    if not _ok_with_audio() and src_pix_fmt:
                        try:
                            out.unlink(missing_ok=True)
                        except Exception:
                            pass
                        _split_precise_lossless_aac_audio(None)
                if not _ok_with_audio():
                    try:
                        out.unlink(missing_ok=True)
                    except Exception:
                        pass
                    _split_copy()
                if not _ok_with_audio():
                    try:
                        out.unlink(missing_ok=True)
                    except Exception:
                        pass
                    _split_copy_aac_audio()
            else:
                _split_precise_lossless_video_only(src_pix_fmt)
                if not _is_decodable(out) and src_pix_fmt:
                    try:
                        out.unlink(missing_ok=True)
                    except Exception:
                        pass
                    _split_precise_lossless_video_only(None)

            # Last-resort fallbacks: keep video even if audio cannot be preserved.
            if not _is_decodable(out):
                try:
                    out.unlink(missing_ok=True)
                except Exception:
                    pass
                _split_precise_lossless_video_only(src_pix_fmt)
                if not _is_decodable(out) and src_pix_fmt:
                    try:
                        out.unlink(missing_ok=True)
                    except Exception:
                        pass
                    _split_precise_lossless_video_only(None)
            if not _is_decodable(out):
                try:
                    out.unlink(missing_ok=True)
                except Exception:
                    pass
                _split_copy_video_only()
        else:
            if include_audio:
                _split_copy()
                if not _ok_with_audio():
                    try:
                        out.unlink(missing_ok=True)
                    except Exception:
                        pass
                    _split_copy_aac_audio()
                if not _ok_with_audio():
                    try:
                        out.unlink(missing_ok=True)
                    except Exception:
                        pass
                    _split_precise_lossless(src_pix_fmt)
                    if not _ok_with_audio() and src_pix_fmt:
                        try:
                            out.unlink(missing_ok=True)
                        except Exception:
                            pass
                        _split_precise_lossless(None)
                if not _ok_with_audio():
                    try:
                        out.unlink(missing_ok=True)
                    except Exception:
                        pass
                    _split_precise_lossless_aac_audio(src_pix_fmt)
                    if not _ok_with_audio() and src_pix_fmt:
                        try:
                            out.unlink(missing_ok=True)
                        except Exception:
                            pass
                        _split_precise_lossless_aac_audio(None)
            else:
                _split_copy_video_only()
                if not _is_decodable(out):
                    try:
                        out.unlink(missing_ok=True)
                    except Exception:
                        pass
                    _split_precise_lossless_video_only(src_pix_fmt)
                    if not _is_decodable(out) and src_pix_fmt:
                        try:
                            out.unlink(missing_ok=True)
                        except Exception:
                            pass
                        _split_precise_lossless_video_only(None)

            # Last-resort fallbacks: keep video even if audio cannot be preserved.
            if not _is_decodable(out):
                try:
                    out.unlink(missing_ok=True)
                except Exception:
                    pass
                _split_copy_video_only()
            if not _is_decodable(out):
                try:
                    out.unlink(missing_ok=True)
                except Exception:
                    pass
                _split_precise_lossless_video_only(src_pix_fmt)
                if not _is_decodable(out) and src_pix_fmt:
                    try:
                        out.unlink(missing_ok=True)
                    except Exception:
                        pass
                    _split_precise_lossless_video_only(None)

        if _is_decodable(out):
            chunk_paths.append(out)

    # Safety: never return a partial set of chunks. If splitting failed for any scene,
    # fall back to processing the original video as a single chunk.
    if len(chunk_paths) != len(normalized_scenes):
        if on_progress:
            on_progress("⚠️ Split produced an incomplete set of chunks; falling back to single-pass input.\n")
        return [Path(video_path)]

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


def _write_concat_list(txt_path: Path, paths: List[Path]) -> None:
    with txt_path.open("w", encoding="utf-8") as f:
        for p in paths:
            f.write(f"file '{p.resolve().as_posix()}'\n")


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
    # Stabilize near-integer frame rates to avoid tiny rational drift.
    nearest_int = round(picked)
    if abs(picked - nearest_int) <= 0.05:
        picked = float(nearest_int)
    return max(1.0, min(240.0, picked))


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
) -> bool:
    """
    Concatenate chunk videos into a single MP4.
    Merge is always done as video-only; caller can remux original audio afterward.
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

    txt = output_path.parent / "concat.txt"
    _write_concat_list(txt, stable_chunks)

    expected_duration = _sum_chunk_durations(stable_chunks)
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
        fallback_fps = _pick_merge_fps(signatures, stable_chunks)
        if not fallback_fps or fallback_fps <= 0:
            fallback_fps = 30.0
        fps_str = f"{float(fallback_fps):.6f}".rstrip("0").rstrip(".")
        if not fps_str:
            fps_str = "30"
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
        # Preferred path for H.264/H.265: convert each MP4 segment to MPEG-TS (Annex B),
        # then concat-copy back to MP4. This avoids timestamp/PPS issues seen with direct
        # MP4 stream-copy concat and keeps video bit-exact (no re-encode).
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
                    if proc_ts_concat.returncode == 0 and _validate_or_fix_duration(output_path, min_ratio=0.90):
                        if on_progress:
                            on_progress(
                                f"Concatenated {len(stable_chunks)} chunk(s) via TS stream copy (codec={common_codec}).\n"
                            )
                        return True
                    if on_progress:
                        tail = (proc_ts_concat.stderr or proc_ts_concat.stdout or "").strip()[-400:]
                        on_progress("WARN: TS stream-copy concat failed; trying generic copy merge.\n")
                        if tail:
                            on_progress(f"ffmpeg: {tail}\n")

        # Secondary copy path: direct concat demuxer + stream copy.
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
        if proc_copy.returncode == 0 and _validate_or_fix_duration(output_path, min_ratio=0.90):
            if on_progress:
                on_progress(
                    f"Concatenated {len(stable_chunks)} chunk(s) via direct stream copy.\n"
                )
            return True
        if on_progress:
            tail = (proc_copy.stderr or proc_copy.stdout or "").strip()[-400:]
            on_progress("WARN: Direct stream-copy concat failed validation; trying robust fallback.\n")
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
    overlap_frames: int = 0,
    fps: Optional[float] = None,
    encode_settings: Optional[Dict[str, Any]] = None,
    on_progress: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Concatenate video chunks with smooth blending of overlapping regions.
    
    Args:
        chunk_paths: List of video chunk file paths
        output_path: Output video path
        overlap_frames: Number of overlapping frames between chunks
        fps: Frame rate (detected from first chunk if None)
        on_progress: Progress callback
        
    Returns:
        True if successful, False otherwise
    """
    if not chunk_paths:
        return False
    
    # If no overlap, use simple concat
    if overlap_frames <= 0:
        return concat_videos(chunk_paths, output_path, encode_settings=encode_settings, on_progress=on_progress)
    
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
                    # Subsequent chunks - blend overlap region
                    if len(all_frames) >= overlap_frames and len(chunk_frames) >= overlap_frames:
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
            
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Write frames to ffmpeg
            for frame in all_frames:
                proc.stdin.write(frame.tobytes())
            
            proc.stdin.close()
            proc.wait()
            
            if proc.returncode != 0 or not temp_output.exists():
                if on_progress:
                    on_progress(f"❌ FFmpeg encoding failed: {proc.stderr.read().decode()}\n")
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
        model_type: Model type ("seedvr2", "gan", "rife", "flashvsr") - used if process_func is None
    
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

    frame_progress_re = re.compile(r"(\d+)\s*/\s*(\d+)")
    pct_progress_re = re.compile(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*%")
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

    def _emit_progress_line(line: str) -> None:
        nonlocal inline_frame_progress_active, inline_frame_progress_width
        payload = str(line or "").rstrip("\r\n")
        if not payload:
            return
        try:
            if payload.startswith("FRAME_PROGRESS "):
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
            if (now - last_overall_emit_ts) < 0.45 and abs(frac - last_emitted_fraction) < 0.004:
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
        """
        Emit key diagnostics to both console (CMD) and progress callback.
        """
        line = str(message)
        if not line.endswith("\n"):
            line += "\n"
        try:
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
        overlap_frames_for_blend = int(chunk_overlap * merge_fps_hint) if chunk_overlap > 0 else 0
        ok = concat_videos_with_blending(
            merge_chunks,
            partial_target,
            overlap_frames=overlap_frames_for_blend,
            fps=merge_fps_hint,
            encode_settings=settings,
            on_progress=on_progress,
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
            validated_existing_chunks.append(Path(resolved_chunk))
            chunk_logs.append({
                "chunk_index": i,
                "input": "resumed",
                "output": str(resolved_chunk),
                "returncode": 0,
                "resumed": True,
            })
        output_chunks = validated_existing_chunks.copy()
        if chunk_work_units:
            completed_chunk_units = float(sum(chunk_work_units[: len(validated_existing_chunks)]))
        current_chunk_index = 0
        current_chunk_inner_fraction = 0.0
        on_progress(f"✅ Loaded {len(existing_chunks)} completed chunks from previous run - skipping to chunk {start_chunk_idx + 1}\n")
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
            # Replace chunk-local FRAME_PROGRESS with overall progress to keep UI/CMD consistent.
            if stripped.startswith("FRAME_PROGRESS "):
                return
            try:
                if text.endswith("\n"):
                    on_progress(text)
                else:
                    on_progress(text + "\n")
            except Exception:
                pass
        
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
    overlap_frames_for_blend = int(chunk_overlap * merge_fps_hint) if chunk_overlap > 0 else 0
    merge_stage_progress = 0.30
    _emit_overall_progress("Merging processed chunks", force=True)
    ok = concat_videos_with_blending(
        merge_chunks,
        final_path,
        overlap_frames=overlap_frames_for_blend,
        fps=merge_fps_hint,
        encode_settings=settings,
        on_progress=on_progress
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
