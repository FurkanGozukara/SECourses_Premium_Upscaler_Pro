"""Run DLSS 5 neural rendering over a video, an image or a folder of images.

Video path: ffmpeg decodes RGB frames (the source's own colour matrix), a CPU
worker estimates the optical-flow guides one frame ahead, the GPU renders with
temporal history, and ffmpeg encodes with the Output tab's codec settings; the
source audio is muxed back afterwards. Images render as single frames.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from shared.dlss5 import DLSS5_MODELS_DIRNAME, DLSS5_WEIGHTS_FILE
from shared.dlss5.motion import DEFAULT_MOTION_QUALITY, MOTION_QUALITY

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv", ".wmv", ".ts", ".mts", ".m2ts"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

PROFILE_CHOICES = ["Standard", "Natural", "Cinematic", "Neutral"]
MOTION_CHOICES = list(MOTION_QUALITY) + ["Static camera (no flow)"]
PRECISION_CHOICES = ["FP16 (fast, recommended)", "FP32 (reference precision)"]
SPEED_CHOICES = ["CUDA Graphs (fast)", "Eager (safest)"]

DLSS5_DEFAULTS: Dict[str, Any] = {
    "profile": "Standard",
    "intensity": 1.0,
    "detail_strength": 1.0,
    "colour_strength": 1.0,
    "detail_radius": 4.0,
    "processing_scale": 1.0,
    "temporal": True,
    "motion_quality": DEFAULT_MOTION_QUALITY,
    "scene_cut_threshold": 0.3,
    "tone_strength": 1.0,
    "structure_strength": 1.0,
    "precision": PRECISION_CHOICES[0],
    "speed_mode": SPEED_CHOICES[0],
    "max_frames": 0,
    "keep_model_loaded": False,
}


@dataclass
class DLSS5Result:
    returncode: int
    output_path: Optional[str]
    log: str
    frames_processed: int = 0
    total_frames: int = 0
    elapsed_seconds: float = 0.0
    fps: float = 0.0


def weights_path(base_dir: Path) -> Path:
    return Path(base_dir) / DLSS5_MODELS_DIRNAME / DLSS5_WEIGHTS_FILE


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def build_options(settings: Dict[str, Any]):
    from shared.dlss5.engine import DLSS5Options

    profile = str(settings.get("profile") or "Standard")
    if profile not in PROFILE_CHOICES:
        profile = "Standard"
    tone = _float(settings.get("tone_strength"), 1.0)
    structure = _float(settings.get("structure_strength"), 1.0)
    base = {"Neutral": (0.0, 0.0)}.get(profile, (1.0, 1.0))
    return DLSS5Options(
        profile=profile,
        intensity=max(0.0, min(1.0, _float(settings.get("intensity"), 1.0))),
        detail_strength=max(0.0, min(4.0, _float(settings.get("detail_strength"), 1.0))),
        colour_strength=max(0.0, min(4.0, _float(settings.get("colour_strength"), 1.0))),
        detail_radius=max(1.0, min(16.0, _float(settings.get("detail_radius"), 4.0))),
        processing_scale=max(1.0, min(2.0, _float(settings.get("processing_scale"), 1.0))),
        temporal=_bool(settings.get("temporal"), True),
        tone_strength=None if abs(tone - base[0]) < 1e-6 else max(0.0, min(2.0, tone)),
        structure_strength=None if abs(structure - base[1]) < 1e-6 else max(0.0, min(2.0, structure)),
    )


def _resolve_device(settings: Dict[str, Any]):
    import torch

    raw = str(settings.get("cuda_device") or settings.get("device") or "").strip().lower()
    if raw.startswith("cuda:"):
        raw = raw.split(":", 1)[1]
    raw = raw.split(",")[0].strip()
    if not torch.cuda.is_available():
        raise RuntimeError("DLSS 5 needs an NVIDIA RTX GPU with CUDA; CUDA is not available.")
    index = int(raw) if raw.isdigit() else 0
    if index >= torch.cuda.device_count():
        index = 0
    device = torch.device("cuda", index)
    torch.cuda.set_device(device)
    return device


def _probe(path: str) -> Dict[str, Any]:
    exe = shutil.which("ffprobe")
    if not exe:
        return {}
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
             "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_packets,pix_fmt,bits_per_raw_sample",
             "-of", "json", str(path)],
            capture_output=True, text=True, creationflags=_NO_WINDOW, timeout=300,
        ).stdout
        return (json.loads(out or "{}").get("streams") or [{}])[0]
    except Exception:
        return {}


def _rate_text(stream: Dict[str, Any], fallback: float) -> str:
    for key in ("r_frame_rate", "avg_frame_rate"):
        text = str(stream.get(key) or "")
        if "/" in text:
            num, den = text.split("/", 1)
            if num.isdigit() and den.isdigit() and int(den) > 0 and int(num) > 0:
                return text
    return f"{fallback:.8f}"


def _rate_value(text: str) -> float:
    if "/" in text:
        num, den = text.split("/", 1)
        return float(num) / float(den)
    return float(text)


def default_output_path(input_path: str, output_dir: Optional[str], suffix: str = "_DLSS5", ext: str = ".mp4") -> Path:
    from shared.path_utils import collision_safe_path

    source = Path(input_path)
    folder = Path(output_dir) if output_dir else source.parent
    folder.mkdir(parents=True, exist_ok=True)
    return collision_safe_path(folder / f"{source.stem}{suffix}{ext}")


def _plan_engine(settings: Dict[str, Any], base_dir: Path, device, width: int, height: int, log):
    import torch
    from shared.dlss5 import ops
    from shared.dlss5.engine import DLSS5Engine

    precision = "fp32" if "fp32" in str(settings.get("precision") or "").lower() else "fp16"
    engine = DLSS5Engine.get(weights_path(base_dir), device, precision)
    speed = str(settings.get("speed_mode") or SPEED_CHOICES[0])
    engine.set_speed_mode(speed, log=log)
    scale = max(1.0, min(2.0, _float(settings.get("processing_scale"), 1.0)))
    geometry = ops.NetworkGeometry.vendor_aligned(int(round(width * scale)), int(round(height * scale)))
    chunk = engine.plan_chunk(geometry.tokens)
    props = torch.cuda.get_device_properties(device)
    log(f"[DLSS 5] GPU {props.name} (sm{props.major}{props.minor}), {precision.upper()}, {engine.speed_label}, "
        f"network {geometry.network_width}x{geometry.network_height}, chunk {chunk} tokens")
    return engine


def run_dlss5(
    settings: Dict[str, Any],
    base_dir: Path,
    on_progress: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> DLSS5Result:
    """Render ``settings['input_path']`` (video, image or image folder) and write the result."""
    started = time.time()
    logs: List[str] = []

    def log(message: str) -> None:
        text = str(message or "").rstrip()
        if not text:
            return
        logs.append(text)
        try:
            print(text, flush=True)
        except Exception:
            pass
        if on_progress:
            with suppress(Exception):
                on_progress(text + "\n")

    def result(code: int, path: Optional[str], frames: int = 0, total: int = 0, message: str = "") -> DLSS5Result:
        if message:
            log(message)
        elapsed = max(0.0, time.time() - started)
        return DLSS5Result(code, path, "\n".join(logs), frames, total, elapsed, frames / elapsed if elapsed > 0 else 0.0)

    input_path = str(settings.get("input_path") or "").strip().strip('"')
    if not input_path or not Path(input_path).exists():
        return result(1, None, message=f"[DLSS 5] Input not found: {input_path or '(empty)'}")
    if not weights_path(base_dir).is_file():
        from shared.model_downloads import ensure_dlss5_model

        ok, error = ensure_dlss5_model(Path(base_dir), on_progress=on_progress)
        if not ok:
            return result(1, None, message=f"[DLSS 5] Model download failed: {error}")
    try:
        device = _resolve_device(settings)
    except Exception as exc:
        return result(1, None, message=f"[DLSS 5] {exc}")

    source = Path(input_path)
    try:
        if source.is_dir():
            return _run_folder(settings, base_dir, device, source, log, result, cancel_event)
        if source.suffix.lower() in IMAGE_EXTS:
            return _run_image(settings, base_dir, device, source, log, result)
        return _run_video(settings, base_dir, device, source, log, result, cancel_event)
    except Exception as exc:  # report OOM and runtime failures as a failed run, never a crash
        import traceback

        detail = traceback.format_exc(limit=6)
        text = str(exc)
        if "out of memory" in text.lower():
            text += " - lower Processing scale, close other GPU apps or use a smaller input."
        return result(1, None, message=f"[DLSS 5] Failed: {text}\n{detail}")
    finally:
        if not _bool(settings.get("keep_model_loaded"), False):
            from shared.dlss5.engine import DLSS5Engine

            DLSS5Engine.release_all()


def _load_image_rgb(path: Path) -> tuple[np.ndarray, bool]:
    from PIL import Image

    image = Image.open(path)
    deep = image.mode in {"I;16", "I;16B", "I;16L", "I", "RGB;16", "RGBA;16"}
    if deep:
        array = np.asarray(image, dtype=np.uint16) if image.mode.startswith("I;16") else np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        return array.astype(np.uint16), True
    return np.asarray(image.convert("RGB"), dtype=np.uint8), False


def _save_image(array: np.ndarray, path: Path, quality: int = 95) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    if array.dtype == np.uint16:
        import cv2

        cv2.imwrite(str(path), array[..., ::-1])
        return
    image = Image.fromarray(array)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=int(quality))
    elif path.suffix.lower() == ".webp":
        image.save(path, quality=int(quality))
    else:
        image.save(path)


def _render_still(engine, settings, frame: np.ndarray, deep: bool):
    from shared.dlss5.engine import DLSS5Session, to_array, to_tensor

    options = build_options(settings)
    options.temporal = False
    session = DLSS5Session(engine, options)
    out = session.render(to_tensor(frame, engine.device))
    return to_array(out, deep=deep)


def _image_target(settings: Dict[str, Any], source: Path) -> Path:
    override = str(settings.get("output_path") or "").strip()
    fmt = str(settings.get("image_output_format") or "png").lower()
    ext = {"jpg": ".jpg", "jpeg": ".jpg", "webp": ".webp"}.get(fmt, ".png")
    if override:
        target = Path(override)
        return target if target.suffix else default_output_path(str(source), str(target), ext=ext)
    return default_output_path(str(source), settings.get("output_dir"), ext=ext)


def _run_image(settings, base_dir, device, source: Path, log, result) -> DLSS5Result:
    frame, deep = _load_image_rgb(source)
    height, width = frame.shape[:2]
    engine = _plan_engine(settings, base_dir, device, width, height, log)
    out = _render_still(engine, settings, frame, deep)
    target = _image_target(settings, source)
    if deep and target.suffix.lower() != ".png":
        target = target.with_suffix(".png")
    _save_image(out, target, _int(settings.get("image_output_quality"), 95))
    return result(0, str(target), 1, 1, f"[DLSS 5] Image saved: {target}")


def _run_folder(settings, base_dir, device, folder: Path, log, result, cancel_event) -> DLSS5Result:
    from shared.path_utils import list_files_sorted

    files = [p for p in list_files_sorted(folder, IMAGE_EXTS)]
    if not files:
        return result(1, None, message=f"[DLSS 5] No images in {folder}")
    override = str(settings.get("output_path") or "").strip()
    out_dir = Path(override) if override else Path(settings.get("output_dir") or folder.parent) / f"{folder.name}_DLSS5"
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = None
    done = 0
    for index, path in enumerate(files, 1):
        if cancel_event is not None and cancel_event.is_set():
            return result(1, str(out_dir), done, len(files), "[DLSS 5] Cancelled.")
        frame, deep = _load_image_rgb(path)
        if engine is None:
            engine = _plan_engine(settings, base_dir, device, frame.shape[1], frame.shape[0], log)
        out = _render_still(engine, settings, frame, deep)
        _save_image(out, out_dir / (path.stem + (".png" if deep else path.suffix.lower())), _int(settings.get("image_output_quality"), 95))
        done += 1
        log(f"FRAME_PROGRESS {done}/{len(files)} images")
    return result(0, str(out_dir), done, len(files), f"[DLSS 5] {done} images saved to {out_dir}")


class _MotionWorker:
    """Reads frames and estimates the guide of frame i (against frame i-1) ahead of the GPU.

    Optical flow needs only the source frames, so several frames are estimated at once on a thread
    pool (OpenCV releases the GIL; each thread owns its DIS object). Results come back in frame order.
    Measured 2026-11-05 (960x540, Balanced): one thread 134 ms/frame, the GPU render 55-79 ms/frame.
    """

    def __init__(self, reader, settings: Dict[str, Any], temporal: bool, max_frames: int):
        import os
        from concurrent.futures import ThreadPoolExecutor

        from shared.dlss5.motion import make_estimator

        self.reader = reader
        self.max_frames = max_frames
        self._make = None
        if temporal:
            quality = str(settings.get("motion_quality") or DEFAULT_MOTION_QUALITY)
            threshold = _float(settings.get("scene_cut_threshold"), 0.3)
            self._make = lambda: make_estimator(quality, threshold)
        self.workers = max(2, min(8, (os.cpu_count() or 4) // 4))
        self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="dlss5-flow") if self._make else None
        self._local = threading.local()
        self.queue: "queue.Queue" = queue.Queue(maxsize=self.workers * 2 + 2)
        self.error: Optional[BaseException] = None
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="dlss5-reader", daemon=True)
        self.thread.start()

    def _estimate(self, frame, previous):
        estimator = getattr(self._local, "estimator", None)
        if estimator is None:
            estimator = self._local.estimator = self._make()
        return estimator.estimate(frame, previous)

    def _run(self) -> None:
        previous = None
        count = 0
        try:
            while not self._stop.is_set():
                frame = self.reader.read()
                if frame is None or (self.max_frames and count >= self.max_frames):
                    break
                future = None
                if self.pool is not None and previous is not None:
                    future = self.pool.submit(self._estimate, frame, previous)
                self.queue.put((frame, future))
                previous = frame
                count += 1
        except BaseException as exc:  # surfaced to the render loop
            self.error = exc
        finally:
            self.queue.put(None)

    def get(self):
        item = self.queue.get()
        if item is None:
            if self.error is not None:
                raise self.error
            return None
        frame, future = item
        return frame, (future.result() if future is not None else None)

    def close(self) -> None:
        self._stop.set()
        with suppress(Exception):
            while True:
                self.queue.get_nowait()
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)


def _encoder_command(settings: Dict[str, Any], output: Path, width: int, height: int, rate: str, deep: bool,
                     color_args: List[str]) -> List[str]:
    from shared.video_codec_options import build_ffmpeg_video_encode_args

    codec = str(settings.get("video_codec") or "h264").strip().lower()
    codec = {"libx264": "h264", "x264": "h264", "avc": "h264", "libx265": "h265", "x265": "h265", "hevc": "h265",
             "libsvtav1": "av1", "svtav1": "av1"}.get(codec, codec)
    if codec not in ("h264", "h265", "av1"):
        codec = "h264"
    pixel_format = str(settings.get("pixel_format") or "yuv420p")
    if deep and "10" not in pixel_format:
        pixel_format = "yuv420p10le"
    args = build_ffmpeg_video_encode_args(
        codec=codec,
        quality=_int(settings.get("video_quality"), 18),
        pixel_format=pixel_format,
        preset=str(settings.get("video_preset") or "medium"),
        h265_tune=str(settings.get("h265_tune") or "none"),
        av1_film_grain=_int(settings.get("av1_film_grain"), 8),
        av1_film_grain_denoise=_bool(settings.get("av1_film_grain_denoise"), False),
        audio_codec="none",
    )
    if deep and "-pix_fmt" in args:
        args[args.index("-pix_fmt") + 1] = pixel_format if "10" in pixel_format else "yuv420p10le"
    return [shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-v", "error", "-nostats", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le" if deep else "rgb24",
            "-s:v", f"{width}x{height}", "-r", rate, "-i", "-", *color_args, *args, "-movflags", "+faststart", str(output)]


def _run_video(settings, base_dir, device, source: Path, log, result, cancel_event) -> DLSS5Result:
    import torch
    from shared.dlss5.engine import DLSS5Session, to_array, to_tensor
    from shared.path_utils import get_media_fps
    from shared.rtx_superres_runner import _FrameReader, _probe_video_stream, _sdr_color_args

    if source.suffix.lower() not in VIDEO_EXTS:
        return result(1, None, message=f"[DLSS 5] Unsupported input type: {source.suffix}")
    if not shutil.which("ffmpeg"):
        return result(1, None, message="[DLSS 5] ffmpeg was not found in PATH.")
    color = _probe_video_stream(str(source))
    size = color.get("size")
    if not size:
        return result(1, None, message=f"[DLSS 5] Could not read the video size of {source}")
    width, height = int(size[0]), int(size[1])
    stream = _probe(str(source))
    total = _int(stream.get("nb_frames") or stream.get("nb_read_packets"), 0)
    fps_value = float(get_media_fps(str(source)) or 30.0)
    rate = _rate_text(stream, fps_value)
    max_frames = max(0, _int(settings.get("max_frames"), 0))
    if max_frames and (not total or max_frames < total):
        total = max_frames
    src_pix = str(stream.get("pix_fmt") or color.get("pix_fmt") or "")
    want_deep = any(tag in src_pix for tag in ("10", "12", "16")) and "10" in str(settings.get("pixel_format") or "")
    output = Path(str(settings.get("output_path") or "").strip() or default_output_path(str(source), settings.get("output_dir")))
    if output.suffix.lower() != ".mp4":
        output = output.with_suffix(".mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    silent = output.with_name(output.stem + ".video_only.mp4")

    engine = _plan_engine(settings, base_dir, device, width, height, log)
    options = build_options(settings)
    session = DLSS5Session(engine, options)
    log(f"[DLSS 5] {source.name}: {width}x{height} @ {_rate_value(rate):.3f} fps, {total or '?'} frames, "
        f"profile {options.profile}, intensity {options.intensity:g}, "
        f"{'temporal ' + str(settings.get('motion_quality') or DEFAULT_MOTION_QUALITY) if options.temporal else 'independent frames'}")

    reader = _FrameReader(str(source), width, height, deep=want_deep, prefetch=0)
    motion = _MotionWorker(reader, settings, options.temporal, max_frames)
    writer = subprocess.Popen(
        _encoder_command(settings, silent, width, height, rate, want_deep, _sdr_color_args(color)),
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=_NO_WINDOW,
    )
    write_q: "queue.Queue" = queue.Queue(maxsize=4)
    write_error: List[str] = []
    stderr_lines: List[str] = []

    def _drain_stderr() -> None:
        # ffmpeg blocks once an undrained stderr pipe fills; keep the last lines for the error report.
        for raw in iter(writer.stderr.readline, b""):
            stderr_lines.append(raw.decode("utf-8", "replace").rstrip())
            del stderr_lines[:-40]

    threading.Thread(target=_drain_stderr, name="dlss5-ffmpeg-log", daemon=True).start()

    def _writer() -> None:
        try:
            while True:
                item = write_q.get()
                if item is None:
                    break
                writer.stdin.write(memoryview(np.ascontiguousarray(item)).cast("B"))
        except Exception as exc:
            write_error.append(str(exc))

    writer_thread = threading.Thread(target=_writer, name="dlss5-writer", daemon=True)
    writer_thread.start()
    done = 0
    t_render = time.time()
    last_report = 0.0
    cancelled = False
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            item = motion.get()
            if item is None:
                break
            frame, estimate = item
            out = session.render(to_tensor(frame, device), estimate)
            write_q.put(to_array(out, deep=want_deep))
            done += 1
            if write_error:
                break
            now = time.time()
            if now - last_report > 0.75 or done == total:
                last_report = now
                speed = done / max(1e-6, now - t_render)
                eta = (total - done) / speed if total and speed > 0 else 0
                log(f"FRAME_PROGRESS {done}/{total or '?'} frames | {speed:.2f} fps | ETA {int(eta // 60)}m {int(eta % 60):02d}s")
    finally:
        write_q.put(None)
        writer_thread.join(timeout=600)
        with suppress(Exception):
            writer.stdin.close()
        code = writer.wait(timeout=3600)
        stderr_tail = " | ".join(stderr_lines[-12:])
        motion.close()
        reader.close()
        torch.cuda.synchronize(device)
    if cancelled:
        with suppress(Exception):
            silent.unlink()
        return result(1, None, done, total, "[DLSS 5] Cancelled.")
    if write_error or code != 0 or not silent.is_file():
        return result(1, None, done, total, f"[DLSS 5] Encoding failed (code {code}): {(write_error or [stderr_tail])[0]}")
    if session.scene_cuts:
        log(f"[DLSS 5] History reset at {session.scene_cuts} scene cut(s).")
    final = silent
    try:
        from shared.audio_utils import ensure_audio_on_video

        changed, muxed, audio_err = ensure_audio_on_video(
            video_path=silent,
            audio_source_path=source,
            audio_codec=str(settings.get("audio_codec") or "copy"),
            audio_bitrate=str(settings.get("audio_bitrate")) if settings.get("audio_bitrate") else None,
            force_replace=True,
            on_progress=None,
        )
        if muxed and Path(muxed).is_file():
            final = Path(muxed)
        if audio_err:
            log(f"[DLSS 5] Audio note: {audio_err}")
    except Exception as exc:
        log(f"[DLSS 5] Audio copy skipped: {exc}")
    if final != output:
        with suppress(Exception):
            if output.exists():
                output.unlink()
            os.replace(final, output)
            final = output
    elapsed = max(1e-6, time.time() - t_render)
    return result(0, str(final), done, total, f"[DLSS 5] Done: {done} frames in {elapsed:.1f}s ({done / elapsed:.2f} fps) -> {final}")
