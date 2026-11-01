"""Real RIFE (CPU, tiny clips) regressions for the exact-timestep timeline (2026-11).

* A target FPS is rendered directly: (N-1) * target / source + 1 frames (floor), at the exact
  rational rate, with no intermediate x4 file and no ffmpeg frame dropping.
* An integer multiplier keeps q*(N-1)+1 frames.
* Duplicate-frame removal keeps the timeline length.
* Source colour tags are carried to the output.
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

ROOT = Path(__file__).resolve().parents[1]
RIFE = ROOT / "RIFE"
MODEL = RIFE / "models" / "4.26"
HAVE = (shutil.which("ffmpeg") is not None and (MODEL / "flownet.pkl").is_file()
        and (RIFE / "train_log" / "RIFE_HDv3.py").is_file())


def _run(cmd, **kw):
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise RuntimeError((proc.stdout or "") + (proc.stderr or ""))
    return proc


def _probe(path: Path) -> dict:
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                "stream=r_frame_rate,nb_read_frames,color_space,color_primaries,color_transfer", "-of", "json",
                str(path)]).stdout
    return json.loads(out)["streams"][0]


@unittest.skipUnless(HAVE, "needs ffmpeg and the RIFE 4.26 model")
class RifeExactFpsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rife_exact_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _src(self, rate: str, frames: int, name: str, dup: bool = False) -> Path:
        path = self.tmp / name
        # ffmpeg 8/9 take colour tags from the frames: label them with setparams.
        tag = "scale=out_color_matrix=bt709,setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv"
        vf = f"scale=64:48,{tag},format=yuv420p"
        if dup:
            vf = f"scale=64:48,fps={Fraction(rate) / 2},fps={rate},{tag},format=yuv420p"
        _run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size=128x96:rate={rate}",
              "-vf", vf, "-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast", "-crf", "8", str(path)])
        return path

    def _rife(self, src: Path, out: Path, *flags: str) -> dict:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONWARNINGS="ignore")
        _run([sys.executable, str(ROOT / "tools" / "rife_inference_wrapper.py"), str(RIFE / "inference_video.py"),
              "--video", str(src), "--output", str(out), "--model", str(MODEL), "--no-audio", "--pad", "64",
              "--video-preset", "ultrafast", "--video-crf", "10", *flags], cwd=str(RIFE), env=env)
        return _probe(out)

    def test_24_and_23976_to_60_are_exact(self):
        for rate in ("24", "24000/1001"):
            with self.subTest(rate=rate):
                src = self._src(rate, 12, f"src_{rate.replace('/', '_')}.mp4")
                info = self._rife(src, self.tmp / f"o_{rate.replace('/', '_')}.mp4", "--fps", "60")
                expected = int(Fraction(11) * 60 / Fraction(rate)) + 1
                self.assertEqual(int(info["nb_read_frames"]), expected)
                self.assertEqual(info["r_frame_rate"], "60/1")
                self.assertEqual(info.get("color_space"), "bt709")
                self.assertEqual(info.get("color_primaries"), "bt709")

    def test_integer_multiplier_keeps_q_n_minus_1_plus_1(self):
        src = self._src("24", 12, "m.mp4")
        info = self._rife(src, self.tmp / "m_x2.mp4")
        self.assertEqual(int(info["nb_read_frames"]), 2 * 11 + 1)
        self.assertEqual(info["r_frame_rate"], "48/1")

    def test_dedup_keeps_the_timeline(self):
        src = self._src("24", 12, "d.mp4", dup=True)
        plain = self._rife(src, self.tmp / "d_plain.mp4")
        dedup = self._rife(src, self.tmp / "d_dedup.mp4", "--dedup")
        self.assertEqual(int(plain["nb_read_frames"]), int(dedup["nb_read_frames"]))


if __name__ == "__main__":
    unittest.main()
