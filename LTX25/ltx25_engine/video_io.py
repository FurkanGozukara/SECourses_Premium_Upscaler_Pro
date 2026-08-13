"""
Video IO for the LTX 2.5 engine.

- Reading: PyAV, frames stored as uint8 [F, H, W, 3] (RAM-friendly).
- Writing: ffmpeg subprocess fed raw RGB frames, source audio copied in the
  same pass. Works identically on Windows and Linux.
- StreamingChunkMerger: LTX25MergeVideoChunks cross-fade, but frames are
  flushed to the encoder as soon as they can no longer change, so RAM stays
  bounded by (overlap + chunk) frames instead of the whole video.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from fractions import Fraction
from typing import List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger("ltx25.video_io")


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


def read_video_frames(path: str) -> Tuple[torch.Tensor, float, bool]:
    """Returns (frames uint8 [F,H,W,3], fps, has_audio)."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = stream.average_rate or stream.guessed_rate or Fraction(30, 1)
        fps = float(rate)
        has_audio = len(container.streams.audio) > 0
        frames: List[np.ndarray] = []
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise RuntimeError(f"No decodable video frames in {path}")
    array = np.stack(frames)
    return torch.from_numpy(array), fps, has_audio


class FfmpegWriter:
    """Raw RGB pipe into ffmpeg; copies source audio when available."""

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        audio_source: Optional[str] = None,
        codec: str = "libx264",
        crf: int = 15,
        preset: str = "medium",
        pixel_format: str = "yuv420p",
    ):
        self.output_path = str(output_path)
        ffmpeg = find_ffmpeg()
        fps_str = f"{fps:.6f}".rstrip("0").rstrip(".")
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", fps_str,
            "-i", "pipe:0",
        ]
        if audio_source:
            cmd += ["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a:0?"]
        codec = str(codec or "libx264")
        cmd += ["-c:v", codec]
        if codec in ("libx264", "libx265"):
            cmd += ["-crf", str(int(crf)), "-preset", str(preset)]
        elif codec == "libsvtav1":
            cmd += ["-crf", str(int(crf)), "-preset", "8"]
        cmd += ["-pix_fmt", pixel_format]
        if audio_source:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += [self.output_path]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, creationflags=creationflags,
        )
        self.frames_written = 0

    def write(self, frames_uint8: torch.Tensor) -> None:
        """frames_uint8: [F, H, W, 3] uint8 on CPU."""
        data = frames_uint8.contiguous().numpy().tobytes()
        assert self.proc.stdin is not None
        self.proc.stdin.write(data)
        self.frames_written += frames_uint8.shape[0]

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        stderr = b""
        if self.proc.stderr is not None:
            stderr = self.proc.stderr.read()
        code = self.proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg encode failed ({code}): {stderr.decode(errors='replace')[-800:]}")


def to_uint8(frames_float: torch.Tensor) -> torch.Tensor:
    return (frames_float.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


class StreamingChunkMerger:
    """LTX25MergeVideoChunks cross-fade with immediate flush of settled frames."""

    def __init__(self, writer: FfmpegWriter, overlap_frames: int, total_frames: int):
        self.writer = writer
        self.overlap = int(overlap_frames)
        self.total = int(total_frames)
        self.tail: Optional[torch.Tensor] = None  # float frames still blendable
        self.emitted = 0

    def _flush(self, frames_float: torch.Tensor) -> None:
        if frames_float.shape[0] == 0:
            return
        remaining = self.total - self.emitted
        if remaining <= 0:
            return
        frames_float = frames_float[:remaining]
        self.writer.write(to_uint8(frames_float))
        self.emitted += frames_float.shape[0]

    def add_chunk(self, images_float: torch.Tensor, keep_length: int, is_last: bool) -> None:
        current = images_float[: min(int(keep_length), images_float.shape[0])]
        if self.tail is None:
            merged = current
        else:
            shared = min(self.overlap, self.tail.shape[0], current.shape[0])
            if shared:
                alpha = torch.linspace(
                    1.0 / (shared + 1), shared / (shared + 1), shared,
                    device=self.tail.device, dtype=self.tail.dtype,
                ).view(-1, 1, 1, 1)
                blended = self.tail[-shared:] * (1.0 - alpha) + current[:shared] * alpha
                merged = torch.cat((self.tail[:-shared], blended, current[shared:]), dim=0)
            else:
                merged = torch.cat((self.tail, current), dim=0)
        if is_last:
            self._flush(merged)
            # Pad by repeating the last frame if the plan came up short.
            if self.emitted < self.total and merged.shape[0] > 0:
                pad = merged[-1:].repeat(self.total - self.emitted, 1, 1, 1)
                self._flush(pad)
            self.tail = None
        else:
            keep = max(self.overlap, 0)
            if keep > 0 and merged.shape[0] > keep:
                self._flush(merged[:-keep])
                self.tail = merged[-keep:].clone()
            else:
                self.tail = merged
