"""
Regression tests for the RIFE / FPS / Edit Videos tab (V10.2 fixes, 7 November 2026).

1. The tab's Spatial Scale Factor ("scale": the resolution at which RIFE estimates motion; the output size never
   changes) was replaced by the Resolution tab's Upscale x. With Upscale x = 4 every run estimated motion at 4x the
   input size: a 4K input filled 32 GB of VRAM and stalled, and every run interpolated even at FPS Multiplier 1x
   (scale != 1 counts as a RIFE request).
2. With Auto Chunk on (the Resolution tab's default) every single video went through the chunk path, which only runs
   RIFE per scene: Video Editing (here a 0.5x speed change) was skipped and the frames were doubled.

The run_action generator runs for real (ffmpeg does the speed change); only the RIFE runner is a fake that records
what it is asked to do. These tests need ffmpeg/ffprobe on PATH; they are skipped otherwise. CPU only.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.runner import RunResult  # noqa: E402
from shared.services import rife_service  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _make_clip(path: Path, seconds: float = 2.0, fps: int = 25) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=128x72:rate={fps}:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=nb_read_frames,r_frame_rate:format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return {"frames": int(stream["nb_read_frames"]), "fps": float(num) / float(den),
            "duration": float(data["format"]["duration"])}


class _FakeRunner:
    """Records every RIFE call; 'interpolates' by copying the input to the requested output."""

    def __init__(self):
        self.calls = []

    def run_rife(self, settings, on_progress=None):
        self.calls.append(dict(settings))
        dest = Path(settings.get("output_override") or (Path(settings["input_path"]).with_suffix(".rife.mp4")))
        if dest.suffix == "":
            dest = dest / (Path(settings["input_path"]).stem + ".mp4")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(settings["input_path"], dest)
        return RunResult(0, str(dest), "fake RIFE")

    def __getattr__(self, name):  # anything else the service asks (is_canceled() and the like): "no"
        return lambda *args, **kwargs: False


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not on PATH")
class RifeEditAndScaleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rife_edit_scale_"))
        self.clip = self.tmp / "clip.mp4"
        _make_clip(self.clip)
        self.out_dir = self.tmp / "outputs"
        self.temp_dir = self.tmp / "temp"
        self.out_dir.mkdir()
        self.temp_dir.mkdir()
        self.runner = _FakeRunner()
        self.callbacks = rife_service.build_rife_callbacks(
            preset_manager=mock.MagicMock(),
            runner=self.runner,
            run_logger=mock.MagicMock(),
            global_settings={"output_dir": str(self.out_dir), "temp_dir": str(self.temp_dir)},
            output_dir=self.out_dir,
            temp_dir=self.temp_dir,
            shared_state=None,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, **overrides):
        settings = dict(self.callbacks["defaults"])
        settings.update(input_path=str(self.clip), output_override="", batch_enable=False, resume_run_dir="")
        settings.update(overrides)
        values = [settings[k] for k in rife_service.RIFE_ORDER]
        # the Resolution tab's state as the app starts it: Auto Chunk on, and here Upscale x = 4
        state = {"seed_controls": {"upscale_factor_val": 4.0, "auto_chunk": True, "chunk_size_sec": 0},
                 "operation_status": "ready"}
        payloads = list(self.callbacks["run_action"](None, None, *values, state=state))
        return payloads, state

    def _outputs(self):
        return [p for p in self.out_dir.rglob("*.mp4") if "comparison" not in p.name]

    def test_speed_change_with_auto_chunk_applies_the_edit_and_runs_no_rife(self):
        payloads, _state = self._run(edit_mode="speed_change", speed_factor=0.5, fps_multiplier="x1", fps_override=0)
        self.assertEqual(self.runner.calls, [], "nothing to interpolate at 1x: RIFE must not run")
        outputs = self._outputs()
        self.assertTrue(outputs, f"no output video; last status: {payloads[-1][:2] if payloads else None}")
        info = _probe(outputs[0])
        self.assertAlmostEqual(info["duration"], 4.0, delta=0.25)   # 2 s at half speed
        self.assertEqual(info["frames"], 50)                        # the same 50 frames, none interpolated
        self.assertAlmostEqual(info["fps"], 12.5, delta=0.5)        # spread over twice the time

    def test_spatial_scale_is_the_tabs_own_value_not_the_upscale_factor(self):
        self._run(fps_multiplier="x2", scale=1.0)
        self.assertTrue(self.runner.calls, "x2 must run RIFE")
        self.assertTrue(all(float(c.get("scale", 1.0)) == 1.0 for c in self.runner.calls),
                        [c.get("scale") for c in self.runner.calls])

    def test_chunking_decision(self):
        decide = rife_service._rife_chunking_applies
        base = {"edit_mode": "none", "batch_enable": False, "img_mode": False, "png_output": False}
        self.assertTrue(decide(base, input_type="video", chunking_enabled=True, rife_requested=True))
        self.assertFalse(decide({**base, "edit_mode": "speed_change"}, input_type="video", chunking_enabled=True,
                                rife_requested=True))
        self.assertFalse(decide(base, input_type="video", chunking_enabled=True, rife_requested=False))
        self.assertFalse(decide(base, input_type="image", chunking_enabled=True, rife_requested=True))


if __name__ == "__main__":
    unittest.main()
