"""A chunked RIFE run saves its settings, so Past Generations can reload it (V10.2 fix, 7 November 2026).

Seen in the real app: a RIFE x2 run with Auto Chunk on (outputs/0089) wrote its result but no run_summary.json; its
Past Generations card had "Reload parameters" disabled ("No saved settings"). The whole-file path saved the settings,
the chunk path (chunk_and_process) did not.

The real run_action generator runs with static 1 s chunks of a 2 s clip and a fake RIFE runner that copies each chunk;
the real RunLogger writes the summary (its journal goes to a temporary file). CPU only; ffmpeg on PATH.
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

from shared import generation_history as gh  # noqa: E402
from shared.logging_utils import RunLogger  # noqa: E402
from shared.runner import RunResult  # noqa: E402
from shared.services import rife_service  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class _CopyRunner:
    def __init__(self):
        self.calls = 0

    def run_rife(self, settings, on_progress=None):
        self.calls += 1
        dest = Path(settings["output_override"])
        if dest.suffix == "":
            dest = dest / (Path(settings["input_path"]).stem + ".mp4")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(settings["input_path"], dest)
        return RunResult(0, str(dest), "fake RIFE")

    def __getattr__(self, name):
        return lambda *args, **kwargs: False


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not on PATH")
class RifeChunkedRunSettingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rife_chunked_settings_"))
        self.clip = self.tmp / "inputs" / "clip.mp4"
        self.clip.parent.mkdir(parents=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=128x72:rate=25:duration=2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(self.clip)], check=True)
        self.out_dir = self.tmp / "outputs"
        self.temp_dir = self.tmp / "temp"
        self.out_dir.mkdir()
        self.temp_dir.mkdir()
        self._journal = gh.JOURNAL_PATH
        gh.JOURNAL_PATH = self.tmp / "history_data" / "journal.jsonl"

    def tearDown(self):
        gh.JOURNAL_PATH = self._journal
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_chunked_run_saves_settings_that_reload(self):
        runner = _CopyRunner()
        callbacks = rife_service.build_rife_callbacks(
            preset_manager=mock.MagicMock(), runner=runner, run_logger=RunLogger(enabled=True),
            global_settings={"output_dir": str(self.out_dir), "temp_dir": str(self.temp_dir)},
            output_dir=self.out_dir, temp_dir=self.temp_dir, shared_state=None,
        )
        settings = dict(callbacks["defaults"])
        settings.update(input_path=str(self.clip), output_override="", batch_enable=False, resume_run_dir="",
                        fps_multiplier="x2", edit_mode="none", scene_detect="ssim", fp16_mode="fp32")
        values = [settings[k] for k in rife_service.RIFE_ORDER]
        state = {"seed_controls": {"auto_chunk": False, "chunk_size_sec": 1.0}, "operation_status": "ready"}
        payloads = list(callbacks["run_action"](None, None, *values, state=state))
        self.assertGreaterEqual(runner.calls, 2, "the run did not go through the chunk path")
        status = payloads[-1][0]
        self.assertIn("chunked processing complete", str(status))

        run_dir = Path(state["seed_controls"]["last_run_dir"])
        summary_path = run_dir / "run_summary.json"
        self.assertTrue(summary_path.is_file(), f"no run_summary.json in {run_dir}: {sorted(p.name for p in run_dir.iterdir())}")
        entry = json.loads(summary_path.read_text(encoding="utf-8"))[-1]
        self.assertEqual(entry["returncode"], 0)
        self.assertTrue(Path(entry["output"]).is_file())
        self.assertEqual(entry["args"]["scene_detect"], "ssim")
        self.assertEqual(entry["args"]["fps_multiplier"], 2)

        # Past Generations: the card has saved settings (Reload parameters enabled) and they reload into the tab
        rec = gh.record_from_run_dir(run_dir, {})
        self.assertEqual(rec["engine"], "rife")
        self.assertTrue(rec["args"], "the card has no saved settings, so Reload parameters stays disabled")
        merged, applied = gh.settings_for_reload(rec, {}, list(rife_service.RIFE_ORDER))
        self.assertGreater(applied, 20)
        self.assertEqual(merged["input_path"], str(self.clip))
        self.assertEqual(merged["scene_detect"], "ssim")
        self.assertEqual(merged["output_override"], "")  # the run's own output path is not reused


if __name__ == "__main__":
    unittest.main()
