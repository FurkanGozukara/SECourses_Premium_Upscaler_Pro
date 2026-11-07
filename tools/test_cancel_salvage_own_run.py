"""Cancel saves a partial result only from the cancelled job's own run folder (V10.2 fix, 7 November 2026).

Seen in the real app: a SeedVR2 run (outputs/0087) was cancelled before its first chunk finished. The cancel salvage
scanned the outputs root for "recent unfinished" run folders, skipped the job's own empty folder and joined the chunks
of an OLDER run (outputs/0041, a chunked RIFE run from days before, which had no run_summary.json) into
outputs/0041/cancelled_partial_0001.mp4, and showed that file as the cancelled job's partial result.

The RIFE test drives the real run_action generator (static 1 s chunks of a 2 s clip, a fake RIFE runner that presses
Cancel while it works on a chunk). The SeedVR2, GAN and RTX tests call each tab's cancel with the job's run folder
recorded as run_action records it. CPU only; ffmpeg/ffprobe on PATH (skipped otherwise).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.runner import RunResult  # noqa: E402
from shared.services import rife_service  # noqa: E402

try:  # the run-folder record the services keep for Cancel (absent before the fix)
    from shared.output_run_manager import begin_active_run, set_active_run_dir
except ImportError:  # pragma: no cover - old code: the cancel falls back to its folder scan
    def begin_active_run(engine):
        return None

    def set_active_run_dir(engine, run_dir):
        return None

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _make_clip(path: Path, seconds: float, fps: int = 25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=128x72:rate={fps}:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


def _files(folder: Path):
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())


class _Fixture:
    """An outputs root holding an older, unfinished run (0041) with one finished chunk and no run_summary.json."""

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cancel_salvage_"))
        self.out_dir = self.tmp / "outputs"
        self.temp_dir = self.tmp / "temp"
        self.temp_dir.mkdir(parents=True)
        self.old_run = self.out_dir / "0041"
        _make_clip(self.old_run / "processed_chunks" / "chunk_0001_upscaled.mp4", 1.0)
        (self.old_run / "run_context.json").write_text(json.dumps(
            {"created_at": "2026-11-01 22:19:15", "input_path": "old_input.mp4", "model": "RIFE",
             "mode": "subprocess"}), encoding="utf-8")
        self.old_files = _files(self.old_run)

    def cancelled_files(self):
        return [p for p in self.out_dir.rglob("cancelled*") if p.is_file()]

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class _CancellingRunner:
    """A RIFE runner that copies each chunk ('interpolates') and presses Cancel while working on chunk N."""

    def __init__(self, cancel_on_call: int):
        self.cancel_on_call = cancel_on_call
        self.calls = 0
        self.cancel_action = None
        self.cancel_result = None

    def cancel(self):
        return True

    def run_rife(self, settings, on_progress=None):
        self.calls += 1
        if self.calls == self.cancel_on_call:
            self.cancel_result = self.cancel_action()
            return RunResult(1, None, "Cancelled by user")
        dest = Path(settings["output_override"])
        if dest.suffix == "":
            dest = dest / (Path(settings["input_path"]).stem + ".mp4")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(settings["input_path"], dest)
        return RunResult(0, str(dest), "fake RIFE")

    def __getattr__(self, name):  # is_canceled() and the like: "no"
        return lambda *args, **kwargs: False


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not on PATH")
class RifeCancelSalvageTests(unittest.TestCase):
    def setUp(self):
        self.fx = _Fixture()
        self.clip = self.fx.tmp / "inputs" / "clip.mp4"
        _make_clip(self.clip, 2.0)

    def tearDown(self):
        self.fx.cleanup()

    def _run(self, cancel_on_call: int):
        runner = _CancellingRunner(cancel_on_call)
        callbacks = rife_service.build_rife_callbacks(
            preset_manager=mock.MagicMock(), runner=runner, run_logger=mock.MagicMock(),
            global_settings={"output_dir": str(self.fx.out_dir), "temp_dir": str(self.fx.temp_dir)},
            output_dir=self.fx.out_dir, temp_dir=self.fx.temp_dir, shared_state=None,
        )
        runner.cancel_action = callbacks["cancel_action"]
        settings = dict(callbacks["defaults"])
        settings.update(input_path=str(self.clip), output_override="", batch_enable=False, resume_run_dir="",
                        fps_multiplier="x2", edit_mode="none")
        values = [settings[k] for k in rife_service.RIFE_ORDER]
        # Resolution tab: static 1 s chunks (two chunks for the 2 s clip)
        state = {"seed_controls": {"auto_chunk": False, "chunk_size_sec": 1.0}, "operation_status": "ready"}
        list(callbacks["run_action"](None, None, *values, state=state))
        self.assertIsNotNone(runner.cancel_result, "the fake runner never pressed Cancel")
        own = Path(state["seed_controls"]["last_run_dir"])
        self.assertNotEqual(own.resolve(), self.fx.old_run.resolve())
        return runner.cancel_result, own

    def test_cancel_before_the_first_chunk_finished_writes_nothing(self):
        result, own = self._run(cancel_on_call=1)
        self.assertEqual(_files(self.fx.old_run), self.fx.old_files, "the older run's folder was written to")
        self.assertEqual(self.fx.cancelled_files(), [], "a partial result was made from another run's chunks")
        self.assertIn("No partial outputs", str(result[0].get("value") if isinstance(result[0], dict) else result[0]))

    def test_cancel_after_a_chunk_finished_saves_this_runs_partial(self):
        _result, own = self._run(cancel_on_call=2)
        self.assertEqual(_files(self.fx.old_run), self.fx.old_files)
        partials = self.fx.cancelled_files()
        self.assertTrue(partials, "the finished chunk of this run was not saved")
        self.assertTrue(all(p.parent.resolve() == own.resolve() for p in partials), partials)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not on PATH")
class OtherTabsCancelSalvageTests(unittest.TestCase):
    """SeedVR2 (where it was seen), Image-Based (GAN) and RTX Super Resolution share the salvage."""

    def setUp(self):
        self.fx = _Fixture()
        self.own = self.fx.out_dir / "0042"
        (self.own / "processed_chunks").mkdir(parents=True)
        self.settings = {"output_dir": str(self.fx.out_dir), "temp_dir": str(self.fx.temp_dir)}

    def tearDown(self):
        self.fx.cleanup()

    def _cancel(self, engine: str):
        runner = mock.MagicMock()
        runner.cancel.return_value = True
        state = SimpleNamespace(value={"seed_controls": {}})
        if engine == "seedvr2":
            from shared.services.seedvr2_service import build_seedvr2_callbacks

            cb = build_seedvr2_callbacks(mock.MagicMock(), runner, mock.MagicMock(), self.settings, state,
                                         self.fx.out_dir, self.fx.temp_dir)
            return cb["cancel_action"]()
        if engine == "gan":
            from shared.services.gan_service import build_gan_callbacks

            cb = build_gan_callbacks(mock.MagicMock(), runner, mock.MagicMock(), self.settings, state, ROOT,
                                     self.fx.temp_dir, self.fx.out_dir)
            return cb["cancel_action"]({"seed_controls": {}})
        from shared.services.rtx_super_resolution_service import build_rtx_super_resolution_callbacks

        cb = build_rtx_super_resolution_callbacks(mock.MagicMock(), runner, mock.MagicMock(), self.settings, state,
                                                  ROOT, self.fx.temp_dir, self.fx.out_dir)
        return cb["cancel_action"]({"seed_controls": {}})

    def test_cancel_with_no_finished_chunk_leaves_older_runs_alone(self):
        for engine in ("seedvr2", "gan", "rtx"):
            with self.subTest(engine=engine):
                begin_active_run(engine)
                set_active_run_dir(engine, self.own)  # what run_action records when it makes the run folder
                self._cancel(engine)
                self.assertEqual(_files(self.fx.old_run), self.fx.old_files, "the older run's folder was written to")
                self.assertEqual(self.fx.cancelled_files(), [])

    def test_seedvr2_cancel_saves_the_own_runs_finished_chunks(self):
        _make_clip(self.own / "processed_chunks" / "chunk_0001_upscaled.mp4", 1.0)
        begin_active_run("seedvr2")
        set_active_run_dir("seedvr2", self.own)
        self._cancel("seedvr2")
        partials = self.fx.cancelled_files()
        self.assertEqual([p.parent.resolve() for p in partials], [self.own.resolve()])
        self.assertEqual(_files(self.fx.old_run), self.fx.old_files)


if __name__ == "__main__":
    unittest.main()
