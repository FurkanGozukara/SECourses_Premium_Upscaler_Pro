r"""LTX output timing/recovery regressions; CPU only, with real FFmpeg/PyAV files.

The model subprocess is replaced only in runner tests: these exercise the work
after inference, using already encoded videos. Artifacts stay in temp for review.

    venv\Scripts\python.exe -m unittest tools.test_ltx25_output_timing -v
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unittest
import uuid
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared import ltx25_runner as runner

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
REPORTED_AVERAGE = 30.084310352
REPORTED_NOMINAL = 30000 / 1001


def _metadata(rate: float = REPORTED_AVERAGE, frames: int = 300) -> dict:
    return dict(
        codec_name="h264", width=128, height=96, nb_read_frames=str(frames),
        r_frame_rate=str(rate), avg_frame_rate=str(rate),
        start_time="0", duration=str(frames / rate),
    )


class Ltx25MetadataTests(unittest.TestCase):
    def test_reported_source_uses_average_even_below_one_percent(self):
        stream = dict(_metadata(), r_frame_rate="30000/1001")
        self.assertEqual(runner._ltx25_video_fps(stream), REPORTED_AVERAGE)

    def test_fractional_and_large_vfr_rates_are_not_rounded(self):
        for nominal, average in [(30, Fraction(24000, 1001)), (60, Fraction(30000, 1001))]:
            with self.subTest(average=average):
                self.assertEqual(
                    runner._ltx25_video_fps(dict(r_frame_rate=str(nominal), avg_frame_rate=str(average))),
                    float(average),
                )

    def test_unusable_average_falls_back_to_nominal(self):
        for average in [None, "0/0", "N/A", "nan", "inf", "-1", "0"]:
            with self.subTest(average=average):
                self.assertEqual(
                    runner._ltx25_video_fps(dict(r_frame_rate="30000/1001", avg_frame_rate=average)),
                    REPORTED_NOMINAL,
                )
        self.assertEqual(runner._ltx25_video_fps(None), 0.0)

    def _validate(self, output: dict, expected_fps: float = REPORTED_AVERAGE, **kwargs):
        with mock.patch.object(runner, "_probe_ltx25_video", side_effect=[_metadata(), output]):
            return runner._validate_ltx25_video_output(
                Path("source.mp4"), Path("output.mp4"), expected_fps=expected_fps, **kwargs,
            )

    def test_output_average_is_checked_instead_of_nominal_metadata(self):
        output = dict(_metadata(), r_frame_rate="30000/1001")
        ok, detail = self._validate(output)
        self.assertTrue(ok, detail)

    def test_real_rate_errors_still_fail(self):
        ok, detail = self._validate(_metadata(25.0))
        self.assertFalse(ok)
        self.assertIn("frame rate mismatch", detail)

    def test_missing_frame_still_fails(self):
        ok, detail = self._validate(_metadata(frames=299))
        self.assertFalse(ok)
        self.assertIn("decoded frame count mismatch", detail)

    def test_wrong_duration_still_fails(self):
        ok, detail = self._validate(dict(_metadata(), duration="20"))
        self.assertFalse(ok)
        self.assertIn("output duration", detail)

    def test_nonzero_origin_still_fails(self):
        ok, detail = self._validate(dict(_metadata(), start_time="0.1"))
        self.assertFalse(ok)
        self.assertIn("instead of zero", detail)

    def test_frame_range_is_inclusive_and_preserved(self):
        ok, detail = self._validate(_metadata(frames=49), start_frame=3, end_frame=51)
        self.assertTrue(ok, detail)


class BatchRecoveryCleanupTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "temp" / "ltx25_output_timing" / uuid.uuid4().hex
        self.final = self.root / "success" / "finished.mp4"
        self.recovery = self.root / "failure" / "recovery" / "rejected.mp4"
        self.temporary = self.root / "failure" / "input_chunk.mp4"
        for path in [self.final, self.recovery, self.temporary]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(path.name.encode("utf-8"))
            path.resolve().relative_to(self.root.resolve())
            self.assertFalse(path.is_symlink())

    def test_ltx_mixed_batch_cleanup_keeps_recovery_separate_from_successes(self):
        from shared.batch_output_cleanup import keep_only_batch_outputs
        original_bytes = self.recovery.read_bytes()
        logs = []
        result = keep_only_batch_outputs(self.root, [str(self.final)], logs.append, preserve_recovery=True)
        self.assertEqual(self.recovery.read_bytes(), original_bytes)
        self.assertEqual(result["final_outputs"], [str((self.root / "finished.mp4").resolve())])
        self.assertEqual(result["kept_outputs"], 1)
        self.assertFalse(self.temporary.exists())
        self.assertTrue(any("preserved 1 recovery file" in line for line in logs))

    def test_other_callers_keep_existing_cleanup_behavior(self):
        from shared.batch_output_cleanup import keep_only_batch_outputs
        result = keep_only_batch_outputs(self.root, [str(self.final)])
        self.assertEqual(result["kept_outputs"], 1)
        self.assertFalse(self.recovery.exists())
        self.assertFalse(self.temporary.exists())

    def test_all_failed_batch_retains_recovery_and_resume_artifacts(self):
        from shared.batch_output_cleanup import keep_only_batch_outputs
        result = keep_only_batch_outputs(self.root, [], preserve_recovery=True)
        self.assertEqual(result["status"], "skipped")
        self.assertTrue(self.recovery.is_file())
        self.assertTrue(self.temporary.is_file())


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available")
class Ltx25EncodedOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = ROOT / "temp" / "ltx25_output_timing" / uuid.uuid4().hex
        cls.artifacts.mkdir(parents=True)
        cls.source = cls.artifacts / "near_nominal_vfr.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=30000/1001:duration=10.01",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=10.01",
                "-vf", "select=not(eq(n\\,150))", "-fps_mode", "vfr",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                "-video_track_timescale", "30000", "-c:a", "aac", str(cls.source),
            ],
            capture_output=True, text=True, check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        sys.path.insert(0, str(ROOT / "LTX25"))
        from ltx25_engine import video_io
        cls.video_io = video_io
        cls.frames, cls.fps, cls.has_audio = video_io.read_video_frames(str(cls.source))
        cls.frames = cls.frames.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        cls.source_info = runner._probe_ltx25_video(cls.source, count_frames=True)
        nominal = float(Fraction(cls.source_info["r_frame_rate"]))
        assert 0 < abs(cls.fps - nominal) < 0.01 * nominal, cls.source_info
        print(f"\nLTX timing artifacts: {cls.artifacts}", flush=True)

    def setUp(self):
        self.case_dir = self.artifacts / self._testMethodName
        self.case_dir.mkdir()

    def _encode(self, *, frame_count=None, fps=None):
        output = self.case_dir / "finished.mp4"
        writer = self.video_io.FfmpegWriter(
            str(output), 128, 96, self.fps if fps is None else fps,
            audio_source=str(self.source), preset="ultrafast",
        )
        writer.write(self.frames if frame_count is None else self.frames[:frame_count])
        writer.close()
        writer.proc.stderr.close()
        return output

    def _run_completed(self, output: Path, *, returncode=0, extra_settings=None):
        # Only this task's generated media may reach the old destructive handler.
        output.resolve().relative_to(self.artifacts.resolve())
        self.assertFalse(output.is_symlink())
        original_popen = subprocess.Popen
        commands = []
        script_path = ROOT / "tools" / "ltx25_inference.py"

        def spawn(cmd, *args, **kwargs):
            if len(cmd) > 2 and str(cmd[2]) == str(script_path):
                commands.append(cmd)
                return SimpleNamespace(pid=-1)
            return original_popen(cmd, *args, **kwargs)

        paths = {key: script_path for key in ["transformer", "text_encoder", "video_vae", "ic_lora"]}
        paths["audio_vae"] = self.case_dir / "no_audio_vae"
        settings = dict(
            input_path=str(self.source), output_path=str(output), model_path=str(script_path),
            device="cpu", seed=42, _run_dir=str(self.case_dir),
        )
        settings.update(extra_settings or {})
        with (
            mock.patch.object(runner, "_resolve_model_files", return_value=(paths, [], None)),
            mock.patch.object(runner, "_build_env", return_value=dict(os.environ, CUDA_VISIBLE_DEVICES="-1")),
            mock.patch.object(runner.subprocess, "Popen", side_effect=spawn),
            mock.patch.object(runner, "_stream_process", return_value=(returncode, str(output), False)),
            mock.patch.object(runner, "get_command_logger"),
        ):
            result = runner._run_ltx25_once(settings, ROOT)
        self.assertEqual(len(commands), 1)
        return result, commands[0]

    def test_finished_near_nominal_vfr_render_succeeds(self):
        output = self._encode()
        result, command = self._run_completed(output)
        self.assertEqual(result.returncode, 0, result.log)
        self.assertEqual(result.output_path, str(output))
        self.assertTrue(output.is_file())
        self.assertAlmostEqual(result.input_fps, self.fps, places=8)
        self.assertAlmostEqual(result.output_fps, self.fps, places=8)
        self.assertEqual(float(command[command.index("--fps_override") + 1]), self.fps)
        self.assertIn("output validation passed", result.log)
        # FFmpeg writer copied the source soundtrack into the completed file.
        import av
        with av.open(str(output)) as media:
            self.assertEqual(len(media.streams.audio), 1)
            audio = media.streams.audio[0]
            audio_seconds = float(audio.duration * audio.time_base)
        self.assertLess(abs(audio_seconds - self.frames.shape[0] / self.fps), 0.1)

    def test_standard_fractional_rate_is_retained(self):
        output = self._encode(fps=REPORTED_NOMINAL)
        # Use the CFR output as source; both rates must stay exactly 30000/1001.
        with mock.patch.object(self, "source", output):
            result, command = self._run_completed(output)
        self.assertEqual(result.returncode, 0, result.log)
        self.assertAlmostEqual(result.output_fps, REPORTED_NOMINAL, places=8)
        self.assertAlmostEqual(float(command[command.index("--fps_override") + 1]), REPORTED_NOMINAL, places=8)

    def test_exact_reported_metadata_passes_the_complete_runner(self):
        output = self._encode()
        source = dict(_metadata(frames=self.frames.shape[0]), r_frame_rate="30000/1001")
        encoded = _metadata(frames=self.frames.shape[0])
        with mock.patch.object(runner, "_probe_ltx25_video", side_effect=[source, source, encoded]):
            result, command = self._run_completed(output)
        self.assertEqual(result.returncode, 0, result.log)
        self.assertEqual(float(command[command.index("--fps_override") + 1]), REPORTED_AVERAGE)
        self.assertEqual(result.output_fps, REPORTED_AVERAGE)
        self.assertTrue(output.exists())

    def test_user_fps_override_still_resamples_before_inference(self):
        from shared.video_fps_utils import remux_video_fps
        converted = self.case_dir / "expected_30fps_input.mp4"
        ok, detail = remux_video_fps(self.source, converted, 30.0)
        self.assertTrue(ok, detail)
        frames, fps, _ = self.video_io.read_video_frames(str(converted))
        with mock.patch.object(self, "frames", frames.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)):
            output = self._encode(fps=fps)
            result, command = self._run_completed(output, extra_settings=dict(fps=30.0))
        self.assertEqual(result.returncode, 0, result.log)
        self.assertEqual(float(command[command.index("--fps_override") + 1]), 30.0)
        self.assertIn("fps_preprocess", command[command.index("--input") + 1])
        self.assertEqual(result.output_fps, 30.0)
        self.assertNotEqual(len(frames), len(self.frames))

    def test_invalid_completed_render_is_preserved_outside_chunk_name(self):
        output = self._encode(frame_count=self.frames.shape[0] - 1)
        encoded_bytes = output.read_bytes()
        result, _ = self._run_completed(output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(result.output_path)
        self.assertFalse(output.exists())
        recovery = list((self.case_dir / "recovery").glob("*.mp4"))
        self.assertEqual(len(recovery), 1, result.log)
        self.assertEqual(recovery[0].read_bytes(), encoded_bytes)
        self.assertIn(str(recovery[0]), result.log)

    def test_failed_chunk_is_recovered_in_the_run_folder(self):
        output = self._encode(frame_count=self.frames.shape[0] - 1)
        chunk_path = self.case_dir / "processed_chunks" / "chunk_0001.mp4"
        chunk_path.parent.mkdir()
        output.rename(chunk_path)
        result, _ = self._run_completed(chunk_path)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(chunk_path.exists())
        self.assertFalse(list(chunk_path.parent.glob("*.mp4")))
        self.assertTrue((self.case_dir / "recovery" / chunk_path.name).is_file(), result.log)

    def test_validation_exception_preserves_completed_work(self):
        output = self._encode()
        encoded_bytes = output.read_bytes()
        with mock.patch.object(runner, "_validate_ltx25_video_output", side_effect=RuntimeError("test: probe failed")):
            result, _ = self._run_completed(output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(result.output_path)
        recovered = self.case_dir / "recovery" / output.name
        self.assertEqual(recovered.read_bytes(), encoded_bytes)
        self.assertIn("test: probe failed", result.log)

    def test_nonzero_engine_exit_preserves_encoded_work_but_stays_failed(self):
        output = self._encode()
        encoded_bytes = output.read_bytes()
        result, _ = self._run_completed(output, returncode=1)
        self.assertEqual(result.returncode, 1)
        self.assertIsNone(result.output_path)
        self.assertFalse(output.exists())
        recovery = list((self.case_dir / "recovery").glob("*.mp4"))
        self.assertEqual(len(recovery), 1, result.log)
        self.assertEqual(recovery[0].read_bytes(), encoded_bytes)

    def test_recovery_does_not_overwrite_previous_render(self):
        previous = self.case_dir / "recovery" / "finished.mp4"
        previous.parent.mkdir()
        previous.write_bytes(b"previous recovered render")
        output = self._encode(frame_count=self.frames.shape[0] - 1)
        encoded_bytes = output.read_bytes()
        result, _ = self._run_completed(output)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(previous.read_bytes(), b"previous recovered render")
        recovered = list(previous.parent.glob("*.mp4"))
        self.assertEqual(len(recovered), 2, result.log)
        self.assertIn(encoded_bytes, [p.read_bytes() for p in recovered])

    def test_recovery_move_error_leaves_original_file_intact(self):
        output = self._encode(frame_count=self.frames.shape[0] - 1)
        encoded_bytes = output.read_bytes()
        with mock.patch.object(Path, "rename", side_effect=PermissionError("test: recovery move denied")):
            result, _ = self._run_completed(output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(result.output_path)
        self.assertTrue(output.is_file(), result.log)
        self.assertEqual(output.read_bytes(), encoded_bytes)
        self.assertIn("test: recovery move denied", result.log)
        self.assertIn(str(output), result.log)


if __name__ == "__main__":
    unittest.main()
