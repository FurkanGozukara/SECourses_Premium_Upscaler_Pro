"""Temporal flicker of a processed clip against its source: mean |frame(t) - frame(t-1)| of each,
plus the change the processing adds per frame. A temporally stable render keeps the output's
frame-to-frame change close to the source's.

    venv\\Scripts\\python.exe tools/dlss5_flicker.py <source.mp4> <processed.mp4> [frames]
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np


def frames(path: str, count: int, width: int = 480):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-frames:v", str(count), "-vf", f"scale={width}:-2:flags=area",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip().split(",")
    w0, h0 = int(probe[0]), int(probe[1])
    height = int(round(h0 * width / w0 / 2)) * 2
    data = subprocess.run(cmd, capture_output=True).stdout
    arr = np.frombuffer(data, np.uint8)
    n = arr.size // (width * height * 3)
    return arr[: n * width * height * 3].reshape(n, height, width, 3).astype(np.float32) / 255.0


def main() -> int:
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    src = frames(sys.argv[1], count)
    out = frames(sys.argv[2], count)
    n = min(len(src), len(out))
    src, out = src[:n], out[:n]
    d_src = np.abs(np.diff(src, axis=0)).mean(axis=(1, 2, 3))
    d_out = np.abs(np.diff(out, axis=0)).mean(axis=(1, 2, 3))
    effect = np.abs(out - src).mean(axis=(1, 2, 3))
    # Flicker of the added change: how much the residual (out - src) jumps between frames.
    resid = out - src
    d_resid = np.abs(np.diff(resid, axis=0)).mean(axis=(1, 2, 3))
    print(f"frames {n}: source change/frame {d_src.mean():.4f}, output change/frame {d_out.mean():.4f}, "
          f"ratio {d_out.mean() / max(1e-9, d_src.mean()):.3f}")
    print(f"effect |out-src| {effect.mean():.4f}; residual jump/frame {d_resid.mean():.4f} "
          f"({d_resid.mean() / max(1e-9, effect.mean()):.2f} of the effect)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
