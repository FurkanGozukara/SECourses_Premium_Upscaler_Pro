"""
Video / image IO shared by the Open-H3 and RelayVSR engines (Windows and Linux).

- probe_video / iter_video_frames: PyAV, frames decoded in order and handed out
  one at a time, so long inputs never sit in RAM at full size.
- FfmpegWriter: raw RGB pipe into ffmpeg, source audio copied in the same pass.
- StreamingChunkMerger: linear cross-fade over the window overlap, frames are
  flushed to the encoder as soon as they can no longer change.
- extract_audio: the soundtrack as float32 stereo at a given sample rate.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

log = logging.getLogger("shared.video_io")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def format_fps_rate(value) -> str:
    try:
        rate = value if isinstance(value, Fraction) else Fraction(str(float(value))).limit_denominator(1_000_000)
        if rate > 0:
            return f"{rate.numerator}/{rate.denominator}"
    except Exception:
        pass
    raise ValueError(f"Invalid video frame rate: {value!r}")


def find_ffmpeg() -> str:
    candidates = []
    env = os.environ.get("FFMPEG_PATH") or os.environ.get("IMAGEIO_FFMPEG_EXE")
    if env:
        candidates.append(env)
    found = shutil.which("ffmpeg")
    if found:
        candidates.append(found)
    try:
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    for candidate in candidates:
        if candidate and (os.path.isfile(candidate) or shutil.which(candidate)):
            return candidate
    raise RuntimeError("ffmpeg not found (checked FFMPEG_PATH, PATH, imageio-ffmpeg)")


def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    frames: int
    has_audio: bool


def probe_video(path: str, count: bool = True) -> VideoInfo:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate or Fraction(30, 1)
        width, height = int(stream.codec_context.width), int(stream.codec_context.height)
        has_audio = len(container.streams.audio) > 0
        frames = int(stream.frames or 0)
        if count or frames <= 0:
            stream.thread_type = "AUTO"
            frames = sum(1 for _ in container.demux(stream) if _.size > 0)
    if frames <= 0:
        raise RuntimeError(f"No decodable video frames in {path}")
    return VideoInfo(width, height, Fraction(rate), frames, has_audio)


def iter_video_frames(path: str) -> Iterator[np.ndarray]:
    """Decoded RGB frames (uint8 [H, W, 3]) in presentation order."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="rgb24")


def read_image(path: str) -> np.ndarray:
    from PIL import Image, ImageOps

    with Image.open(str(path)) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.asarray(img, dtype=np.float32)
            arr = arr / (65535.0 if arr.max() > 255 else 255.0)
            arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
            return np.stack([arr] * 3, axis=-1)
        return np.asarray(img.convert("RGB"), dtype=np.uint8).copy()


def write_image(path: str, frame_uint8: np.ndarray) -> None:
    from PIL import Image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame_uint8).save(str(path))


def extract_audio(path: str, sample_rate: int, start_s: float = 0.0, duration_s: Optional[float] = None) -> Optional[torch.Tensor]:
    """Soundtrack as float32 [1, 2, samples] at sample_rate, or None when the input has no audio."""
    cmd = [find_ffmpeg(), "-v", "error", "-nostdin"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.6f}"]
    cmd += ["-i", str(path)]
    if duration_s is not None:
        cmd += ["-t", f"{duration_s:.6f}"]
    cmd += ["-map", "0:a:0?", "-vn", "-ac", "2", "-ar", str(int(sample_rate)), "-f", "f32le", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW)
    if proc.returncode != 0 or not proc.stdout:
        return None
    data = np.frombuffer(proc.stdout, dtype=np.float32)
    if data.size < 2:
        return None
    data = data[: data.size // 2 * 2].reshape(-1, 2).T.copy()
    return torch.from_numpy(data).unsqueeze(0)


class FfmpegWriter:
    """Raw RGB pipe into ffmpeg; copies source audio when available."""

    def __init__(self, output_path: str, width: int, height: int, fps, audio_source: Optional[str] = None,
                 codec: str = "libx264", crf: int = 15, preset: str = "medium", pixel_format: str = "yuv420p",
                 audio_offset_s: float = 0.0, video_args: str = ""):
        self.output_path = str(output_path)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [find_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", format_fps_rate(fps), "-i", "pipe:0"]
        if audio_source:
            if audio_offset_s > 0:
                cmd += ["-ss", f"{audio_offset_s:.6f}"]
            cmd += ["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a:0?"]
        codec = str(codec or "libx264")
        if codec in ("prores", "prores_aw"):
            codec = "prores_ks"
        if str(video_args or "").strip():
            # The Output tab's custom FFmpeg video arguments (checked by the app) replace the codec settings.
            import shlex

            cmd += shlex.split(str(video_args), posix=True)
        else:
            cmd += ["-c:v", codec]
            if codec in ("libx264", "libx265"):
                cmd += ["-crf", str(int(crf)), "-preset", str(preset)]
            elif codec == "libsvtav1":
                cmd += ["-crf", str(int(crf)), "-preset", "8"]
            elif codec == "prores_ks":
                # ProRes (in a .mov): the profile follows the pixel format, 422 HQ or 4444; CRF does not apply.
                from .video_codec_options import prores_pixel_format, prores_profile_for_pixel_format

                pixel_format = prores_pixel_format(pixel_format)
                cmd += ["-profile:v", prores_profile_for_pixel_format(pixel_format), "-vendor", "apl0"]
            cmd += ["-pix_fmt", pixel_format]
        if audio_source:
            # never let a shorter audio track stop the video before every frame is written
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        cmd += [self.output_path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, creationflags=_NO_WINDOW)
        self.frames_written = 0

    def write(self, frames_uint8: torch.Tensor) -> None:
        data = frames_uint8.contiguous().numpy().tobytes()
        assert self.proc.stdin is not None
        self.proc.stdin.write(data)
        self.frames_written += frames_uint8.shape[0]

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        stderr = self.proc.stderr.read() if self.proc.stderr is not None else b""
        code = self.proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg encode failed ({code}): {stderr.decode(errors='replace')[-800:]}")

    def abort(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass


def to_uint8(frames_float: torch.Tensor) -> torch.Tensor:
    return (frames_float.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


class StreamingChunkMerger:
    """Linear cross-fade over the window overlap on uint8 frames; settled frames go straight to the writer."""

    def __init__(self, writer: FfmpegWriter, overlap_frames: int, total_frames: int):
        self.writer = writer
        self.overlap = max(0, int(overlap_frames))
        self.total = int(total_frames)
        self.tail: Optional[torch.Tensor] = None
        self.emitted = 0

    def _flush(self, frames: torch.Tensor) -> None:
        remaining = self.total - self.emitted
        if frames.shape[0] == 0 or remaining <= 0:
            return
        frames = frames[:remaining]
        self.writer.write(frames)
        self.emitted += frames.shape[0]

    def add_chunk(self, frames_u8: torch.Tensor, is_last: bool) -> None:
        current = frames_u8
        if self.tail is not None:
            shared = min(self.overlap, self.tail.shape[0], current.shape[0])
            self._flush(self.tail[:self.tail.shape[0] - shared])
            if shared:
                alpha = torch.linspace(1.0 / (shared + 1), shared / (shared + 1), shared).view(-1, 1, 1, 1)
                blended = self.tail[-shared:].float() * (1.0 - alpha) + current[:shared].float() * alpha
                self._flush(blended.round().clamp(0, 255).to(torch.uint8))
            current = current[shared:]
            self.tail = None
        if is_last:
            self._flush(current)
            if self.emitted < self.total and current.shape[0] > 0:
                self._flush(current[-1:].repeat(self.total - self.emitted, 1, 1, 1))
        elif self.overlap > 0 and current.shape[0] > self.overlap:
            self._flush(current[:-self.overlap])
            self.tail = current[-self.overlap:].clone()
        else:
            self.tail = current
