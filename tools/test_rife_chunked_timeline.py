"""Chunked RIFE equals one whole-video run (real RIFE on CPU, tiny clips, app chunk path).

Every chunk renders only its own part of the whole-video output grid (--timeline-start /
--timeline-end from the split manifest), so the merge is a plain concatenation:
* exact frame count (N-1) * target / source + 1 at a non-integer rate (24 -> 60),
* the same frames as an unchunked run (no grid shift at chunk boundaries),
* a hard cut on a chunk boundary is held, never blended,
whether the chunk boundaries fall on the cut or not.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.chunking import chunk_and_process  # noqa: E402
from shared.runner import Runner  # noqa: E402

HAVE = (shutil.which("ffmpeg") is not None and (ROOT / "RIFE" / "models" / "4.26" / "flownet.pkl").is_file())
W, H = 96, 64
FPS = 24
SCENE = 18  # frames per scene; the cut is between frame 17 and 18


def _run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-2000:])
    return proc


def _frames(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3).astype(np.float32)


@unittest.skipUnless(HAVE, "needs ffmpeg and the RIFE 4.26 model")
class RifeChunkedTimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU: deterministic and off the shared GPU
        cls.tmp = Path(tempfile.mkdtemp(prefix="rife_chunks_"))
        a, b, src = cls.tmp / "a.mp4", cls.tmp / "b.mp4", cls.tmp / "src.mp4"
        # Scene A: dark background, white box moving right; scene B: bright background, dark box.
        for path, bg, fg in ((a, "0x202020", "white"), (b, "0xd0c040", "0x101060")):
            _run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c={bg}:s={W}x{H}:r={FPS}",
                  "-f", "lavfi", "-i", f"color=c={fg}:s=16x16:r={FPS}",
                  "-filter_complex", "[0:v][1:v]overlay=x='4+2*n':y=20,format=yuv420p",
                  "-frames:v", str(SCENE), "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast", str(path)])
        (cls.tmp / "list.txt").write_text(f"file '{a.as_posix()}'\nfile '{b.as_posix()}'\n", encoding="utf-8")
        _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(cls.tmp / "list.txt"),
              "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-r", str(FPS), str(src)])
        cls.src = src
        cls.n = 2 * SCENE
        cls.expected = int(Fraction(cls.n - 1) * 60 / FPS) + 1
        cls.whole = cls._interpolate("whole", chunk_seconds=0.0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def _interpolate(cls, name: str, chunk_seconds: float):
        out_dir = cls.tmp / name
        out_dir.mkdir(parents=True, exist_ok=True)
        runner = Runner(ROOT, ROOT / "temp", out_dir, telemetry_enabled=False)
        settings = {
            "input_path": str(cls.src), "output_override": str(out_dir / f"{name}.mp4"), "output_format": "mp4",
            "model_dir": "", "model": "4.26", "fps_multiplier": "x2", "fps_override": 60.0, "scale": 1.0,
            "fp16_mode": False, "png_output": False, "no_audio": True, "img_mode": False, "exp": 1,
            "cuda_device": "", "video_codec": "libx264", "output_quality": 0, "video_quality": 0,
            "video_preset": "ultrafast", "pixel_format": "yuv444p", "audio_codec": "none", "save_metadata": False,
            "scene_detect": "content", "global_output_dir": str(out_dir), "frame_accurate_split": True,
        }
        if chunk_seconds <= 0:
            res = runner.run_rife(settings)
            assert res.returncode == 0, res.log[-3000:]
            return Path(res.output_path), ""

        class Adapter:
            def reset_cancel_state(self):
                runner.reset_cancel_state()

            def is_canceled(self):
                return False

            def run_rife(self, s, on_progress=None):
                return runner.run_rife(s, on_progress=on_progress)

        messages = []
        settings["_run_dir"] = str(out_dir)
        rc, log, output, chunks = chunk_and_process(
            Adapter(), settings, scene_threshold=1_000_000.0, min_scene_len=0.1, work_dir=out_dir,
            on_progress=messages.append, chunk_seconds=chunk_seconds, chunk_overlap=0.0,
            per_chunk_cleanup=False, allow_partial=False, global_output_dir=str(out_dir), model_type="rife",
        )
        assert rc == 0, log[-3000:]
        assert chunks > 1, chunks
        return Path(output), "".join(str(m) for m in messages)

    def _check(self, name: str, chunk_seconds: float):
        whole_path, _ = self.whole
        out, messages = self._interpolate(name, chunk_seconds)
        self.assertIn("RIFE merge: every chunk rendered its own segment", messages)
        whole, chunked = _frames(whole_path), _frames(out)
        self.assertEqual(len(whole), self.expected)
        self.assertEqual(len(chunked), self.expected)
        # Same frames as the unchunked run (separately encoded chunks: allow tiny encode noise).
        diff = np.abs(whole - chunked).mean(axis=(1, 2, 3))
        self.assertLess(float(diff.max()), 2.0, f"frame {int(diff.argmax())} differs by {float(diff.max()):.2f}")
        # Around the cut every frame is plainly scene A (dark) or scene B (bright): held, not blended.
        cut_k = int(Fraction(SCENE) * 60 / FPS)
        for k in range(cut_k - 3, cut_k + 4):
            mean = float(np.median(chunked[k]))
            self.assertTrue(mean < 60 or mean > 120, f"frame {k} looks blended (median {mean:.1f})")

    def test_boundary_on_the_cut(self):
        # 0.75 s chunks at 24 fps = 18 frames: the chunk boundary sits on the cut.
        self._check("on_cut", chunk_seconds=SCENE / FPS)

    def test_boundary_off_the_cut(self):
        # 0.5 s chunks: boundaries at frames 12 and 24, the cut (18) is inside a chunk.
        self._check("off_cut", chunk_seconds=0.5)


if __name__ == "__main__":
    unittest.main()
