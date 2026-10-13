"""Real-FFmpeg regressions for V8.6's RIFE chunk-merge hardening.

* MPEG-TS sources print every stream value twice; probes must still read them.
* A source whose audio ends early must not let RIFE's own ``-shortest`` audio mux
  shorten the interpolated video (it used to fail the final timeline check).
* A chunk that lost tail frames only changes its own boundary.
* Every frame-stream merge retries with the system ffmpeg decoder.
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import chunking  # noqa: E402
from shared.audio_utils import has_audio_stream  # noqa: E402
from shared.path_utils import first_ffprobe_value, get_media_dimensions, get_media_fps  # noqa: E402
from shared.runner import Runner, RunResult  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
FPS = 17

# Mirrors upstream RIFE inference_video.py timing: q*(n-1)+1 frames at q*fps, then, unless
# --no-audio is passed, the source audio is muxed with `-c copy -shortest`.
FAKE_RIFE_CLI = r'''
import argparse, json, os, subprocess
from fractions import Fraction

parser = argparse.ArgumentParser()
parser.add_argument("--video")
parser.add_argument("--output")
parser.add_argument("--exp", type=int, default=1)
parser.add_argument("--multi", type=int, default=2)
parser.add_argument("--no-audio", dest="no_audio", action="store_true")
args, _unknown = parser.parse_known_args()
multi = 2 ** args.exp if args.exp != 1 else args.multi
probe = subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
     "-show_entries", "stream=r_frame_rate,nb_read_frames", "-of", "json", args.video],
    capture_output=True, text=True, check=True,
)
stream = json.loads(probe.stdout)["streams"][0]
rate = Fraction(stream["r_frame_rate"]) * multi
frames = multi * (int(stream["nb_read_frames"]) - 1) + 1
video_only = args.output + ".video.mp4"
subprocess.run(
    ["ffmpeg", "-y", "-v", "error", "-i", args.video, "-map", "0:v:0", "-vf", f"fps={rate}",
     "-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
     "-an", video_only],
    check=True,
)
if args.no_audio:
    os.replace(video_only, args.output)
else:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", video_only, "-i", args.video, "-map", "0:v:0",
         "-map", "1:a?", "-c", "copy", "-shortest", args.output],
        check=True,
    )
    os.remove(video_only)
'''


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "command failed")[-2000:])


def _make_levels_video(path: Path, frames: int, *, offset: int = 0, size: str = "64x48") -> None:
    """Flat frames whose gray level identifies the source frame (lossless x264)."""
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"nullsrc=size={size}:rate={FPS}",
        "-vf", f"geq=lum='16+mod((N+{offset})*37\\,200)':cb=128:cr=128,format=yuv420p",
        "-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0",
        "-pix_fmt", "yuv420p", "-an", str(path),
    ])


def _with_audio(video: Path, output: Path, *, seconds: float, container_args: list[str] | None = None) -> None:
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds:.6f}",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "64k",
        *(container_args or []), str(output),
    ])


def _frame_levels(path: Path) -> list[float]:
    cap = cv2.VideoCapture(str(path))
    levels: list[float] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            levels.append(float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))
    finally:
        cap.release()
    return levels


def _fake_rife(multiplier: int, truncate_chunks: dict[int, int] | None = None):
    """Interpolation stand-in: q*(n-1)+1 frames; optionally drop tail frames of chunk N."""

    def process(settings: dict, on_progress=None) -> RunResult:
        input_path = Path(settings["input_path"])
        output_path = Path(settings["output_override"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        input_frames = int(chunking._probe_chunk_nb_frames(input_path) or 0)
        frames = multiplier * max(0, input_frames - 1) + 1
        index = chunking._extract_chunk_index(input_path) or 0
        frames -= int((truncate_chunks or {}).get(index, 0))
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_path),
                "-map", "0:v:0", "-vf", f"fps={FPS * multiplier}", "-frames:v", str(frames),
                "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv420p",
                "-an", str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        return RunResult(proc.returncode, str(output_path), proc.stderr or "")

    return process


class _NoopRunner:
    def reset_cancel_state(self) -> None:
        pass

    def is_canceled(self) -> bool:
        return False


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available")
class RifeRobustMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="rife_robust_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _chunked_rife(self, source: Path, case: str, multiplier: int = 2, **kwargs) -> tuple[Path, list[str]]:
        messages: list[str] = []
        settings = {
            "input_path": str(source),
            "output_override": str(self.tmp / case / "final.mp4"),
            "output_format": "mp4",
            "frame_accurate_split": True,
            "video_codec": "h264",
            "video_preset": "ultrafast",
            "video_quality": 12,
            "pixel_format": "yuv420p",
            "audio_codec": "copy",
            "fps_multiplier": f"x{multiplier}",
            "progress_report_interval_sec": 60,
        }
        with (
            mock.patch.object(chunking, "detect_scenes", return_value=[]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rc, log, output_path, chunk_count = chunking.chunk_and_process(
                runner=_NoopRunner(),
                settings=settings,
                scene_threshold=27.0,
                min_scene_len=1.0,
                work_dir=self.tmp / case / "work",
                on_progress=messages.append,
                chunk_seconds=kwargs.pop("chunk_seconds", 10 / FPS),
                chunk_overlap=0.0,
                per_chunk_cleanup=False,
                allow_partial=False,
                process_func=kwargs.pop("process", _fake_rife(multiplier)),
                model_type="rife",
            )
        self.assertEqual(rc, 0, f"{log}\n{''.join(messages)}")
        self.assertGreater(chunk_count, 1)
        return Path(output_path), messages

    def test_mpegts_probes_read_the_first_stream_value(self):
        self.assertEqual(first_ffprobe_value("24/1\n24/1\n"), "24/1")
        self.assertEqual(first_ffprobe_value("\n64,48,\n64,48\n"), "64,48")
        video = self.tmp / "video.mp4"
        _make_levels_video(video, 34)
        source = self.tmp / "source.ts"
        _with_audio(video, source, seconds=2.5, container_args=["-output_ts_offset", "1.4"])
        self.assertEqual(get_media_fps(str(source)), float(FPS))
        self.assertEqual(get_media_dimensions(str(source)), (64, 48))
        duration = chunking._probe_video_stream_duration(source)
        self.assertAlmostEqual(float(duration or 0.0), 34 / FPS, delta=0.02)

    def test_matroska_video_duration_ignores_a_longer_audio_track(self):
        video = self.tmp / "video.mp4"
        _make_levels_video(video, 34)
        source = self.tmp / "source.mkv"
        _with_audio(video, source, seconds=3.0)
        duration = chunking._probe_video_stream_duration(source)
        self.assertAlmostEqual(float(duration or 0.0), 34 / FPS, delta=0.02)

    def test_chunked_rife_on_mpegts_source_keeps_every_interpolated_frame(self):
        video = self.tmp / "video.mp4"
        _make_levels_video(video, 30)
        source = self.tmp / "source.ts"
        _with_audio(video, source, seconds=30 / FPS, container_args=["-output_ts_offset", "1.4"])
        output, messages = self._chunked_rife(source, "ts_case")
        self.assertEqual(chunking._probe_chunk_nb_frames(output), 2 * (30 - 1) + 1)
        self.assertTrue(any("RIFE chunk timing" in message for message in messages), messages)
        self.assertTrue(has_audio_stream(output))

    def test_truncated_chunk_tail_only_changes_its_own_boundary(self):
        video = self.tmp / "video.mp4"
        _make_levels_video(video, 30)
        source = self.tmp / "source.mp4"
        _with_audio(video, source, seconds=30 / FPS)
        reference = self.tmp / "whole_video_rife.mp4"
        whole = _fake_rife(2)({"input_path": str(video), "output_override": str(reference)})
        self.assertEqual(whole.returncode, 0, whole.log)

        # Chunk 2 lost its last frame: exactly its shared boundary frame with chunk 3.
        output, messages = self._chunked_rife(source, "truncated", process=_fake_rife(2, {2: 1}))
        plan = [message for message in messages if "RIFE merge overlap plan" in message]
        self.assertTrue(plan and "method=per-chunk" in plan[-1] and "output=[1, 0]" in plan[-1], plan)

        expected_levels = _frame_levels(reference)
        actual_levels = _frame_levels(output)
        self.assertEqual(len(actual_levels), 2 * (30 - 1) + 1)
        self.assertEqual(len(actual_levels), len(expected_levels))
        # Neighbouring source frames differ by >= 43 gray levels, so a duplicated, dropped
        # or blended boundary frame is off by ~20+; re-encoding alone shifts levels by ~2-3.
        worst = max(abs(a - b) for a, b in zip(actual_levels, expected_levels))
        self.assertLess(worst, 6.0, "boundary frames were duplicated, dropped or ghosted")

    def _two_chunks(self) -> list[Path]:
        first = self.tmp / "chunk_0001.mp4"
        second = self.tmp / "chunk_0002.mp4"
        _make_levels_video(first, 12)
        _make_levels_video(second, 9, offset=11)  # frame 0 repeats chunk 1's last frame
        return [first, second]

    def test_overlap_blend_retries_with_ffmpeg_decoder(self):
        chunks = self._two_chunks()
        real_open = chunking._open_frame_reader

        class _DropsLastFrame:
            def __init__(self, path: Path):
                self._cap = cv2.VideoCapture(str(path))
                self._left = int(chunking._probe_chunk_nb_frames(path) or 0) - 1

            def isOpened(self):
                return self._cap.isOpened()

            def read(self):
                if self._left <= 0:
                    return False, None
                self._left -= 1
                return self._cap.read()

            def release(self):
                self._cap.release()

        def flaky_open(path, decoder, scale_to=None):
            if decoder == "opencv":
                return _DropsLastFrame(Path(path))
            return real_open(path, decoder, scale_to=scale_to)

        output = self.tmp / "blended.mp4"
        messages: list[str] = []
        with mock.patch.object(chunking, "_open_frame_reader", side_effect=flaky_open):
            ok = chunking.concat_videos_with_blending(
                chunks, output, overlap_frames=[1],
                encode_settings={"video_codec": "h264", "video_preset": "ultrafast"},
                on_progress=messages.append,
            )
        self.assertTrue(ok, "".join(messages))
        self.assertTrue(any("retrying with the ffmpeg decoder" in m for m in messages), messages)
        self.assertTrue(any("decoder=ffmpeg" in m for m in messages), messages)
        self.assertEqual(chunking._probe_chunk_nb_frames(output), 12 + 9 - 1)

    def test_frame_stream_concat_retries_with_ffmpeg_decoder(self):
        chunks = self._two_chunks()
        real_open = chunking._open_frame_reader

        class _Closed:
            def isOpened(self):
                return False

            def read(self):
                return False, None

            def release(self):
                pass

        def broken_opencv(path, decoder, scale_to=None):
            return _Closed() if decoder == "opencv" else real_open(path, decoder, scale_to=scale_to)

        output = self.tmp / "framepipe.mp4"
        messages: list[str] = []
        with (
            mock.patch.object(chunking, "_open_frame_reader", side_effect=broken_opencv),
            mock.patch.object(chunking, "_merge_stream_copy_is_safe", return_value=(False, "forced re-encode")),
        ):
            ok = chunking.concat_videos(
                chunks, output,
                encode_settings={"video_codec": "h264", "video_preset": "ultrafast"},
                on_progress=messages.append,
            )
        self.assertTrue(ok, "".join(messages))
        self.assertTrue(any("frame-stream fallback (frames=21, decoder=ffmpeg)" in m for m in messages), messages)
        self.assertEqual(chunking._probe_chunk_nb_frames(output), 21)

    def test_target_fps_above_the_multiplier_raises_interpolation(self):
        video = self.tmp / "video24.mp4"
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "testsrc2=size=64x48:rate=24", "-frames:v", "24", "-c:v", "libx264",
            "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(video),
        ])
        runner = Runner(self.tmp / "app", self.tmp / "temp", self.tmp / "out", telemetry_enabled=False)

        def build(**extra) -> list[str]:
            settings = {"fps_multiplier": "x2", "model": "4.26", **extra}
            return runner._build_rife_cmd(Path("inference_video.py"), str(video), self.tmp / "o.mp4", settings)

        # x2 of 24 fps is 48 fps: a 60 fps target needs x4 (96 fps) resampled down to 60.
        cmd = build(fps_override=60.0)
        self.assertEqual(cmd[cmd.index("--multi") + 1], "4")
        self.assertEqual(float(cmd[cmd.index("--fps") + 1]), 60.0)
        self.assertNotIn("--multi", build(fps_override=30.0))
        self.assertNotIn("--multi", build())

    def test_run_rife_never_lets_rife_shorten_the_video(self):
        base = self.tmp / "app"
        (base / "RIFE").mkdir(parents=True)
        (base / "tools").mkdir()
        (base / "RIFE" / "inference_video.py").write_text(FAKE_RIFE_CLI, encoding="utf-8")
        shutil.copy2(ROOT / "tools" / "rife_inference_wrapper.py", base / "tools" / "rife_inference_wrapper.py")
        model_dir = self.tmp / "model"
        model_dir.mkdir()
        (model_dir / "flownet.pkl").write_bytes(b"0")

        video = self.tmp / "video.mp4"
        _make_levels_video(video, 34)
        source = self.tmp / "source_short_audio.mp4"
        _with_audio(video, source, seconds=1.4)  # audio ends 0.6 s before the picture

        runner = Runner(base, self.tmp / "temp", self.tmp / "out", telemetry_enabled=False)
        settings = {
            "input_path": str(source),
            "output_override": str(self.tmp / "out" / "rife.mp4"),
            "output_format": "mp4",
            "model_dir": str(model_dir),
            "fps_multiplier": "x2",
            "audio_codec": "copy",
            "video_codec": "h264",
            "save_metadata": False,
        }
        with contextlib.redirect_stdout(io.StringIO()):
            result = runner.run_rife(settings)
        self.assertEqual(result.returncode, 0, result.log[-2000:])
        output = Path(result.output_path)
        self.assertEqual(chunking._probe_chunk_nb_frames(output), 2 * (34 - 1) + 1)
        self.assertTrue(has_audio_stream(output))
        stream = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=r_frame_rate,start_time", "-of", "json", str(output)],
            capture_output=True, text=True, check=True,
        ).stdout)["streams"][0]
        self.assertEqual(stream["r_frame_rate"], f"{2 * FPS}/1")
        self.assertAlmostEqual(float(stream["start_time"]), 0.0, delta=0.001)


if __name__ == "__main__":
    unittest.main()
