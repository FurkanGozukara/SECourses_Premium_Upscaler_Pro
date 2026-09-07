"""Real-FFmpeg regressions for V8.2's source/output FPS merge mismatch.

RIFE's progress counts input intervals, not encoded output frames. Two 24 fps
chunks can produce 119 and 121 frames at 48 fps: their combined duration is five
seconds, even if stale metadata or the source rate suggests ten seconds.
"""
from __future__ import annotations

import contextlib
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

from shared import chunking
from tools.test_chunk_av_sync import HAVE_FFMPEG, _make_source, _probe


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available")
class RifeMergeTimingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rife_merge_timing_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.encoding = {"video_codec": "h264", "video_preset": "ultrafast"}

    def _chunks(self, fps="48/1"):
        rate = float(chunking._parse_fraction_to_float(fps))
        paths = [self.tmp / f"chunk_{i:04d}.mp4" for i in (1, 2)]
        for path, count in zip(paths, (121, 119)):
            self.assertEqual(
                _make_source(path, fps=fps, seconds=count / rate, audio=False), count
            )
        return paths

    def _assert_timeline(self, output, fps, frames=240):
        stream = _probe(output, "v:0", "r_frame_rate,start_time,duration")
        self.assertEqual(stream["r_frame_rate"], fps)
        self.assertEqual(chunking._probe_chunk_nb_frames(output), frames)
        self.assertAlmostEqual(float(stream["start_time"]), 0.0, delta=0.001)
        rate = float(chunking._parse_fraction_to_float(fps))
        self.assertAlmostEqual(float(stream["duration"]), frames / rate, delta=0.001)

    def test_24_to_48_uneven_chunks_ignore_stale_durations_in_every_merge_path(self):
        chunks = self._chunks()
        probe_duration = chunking._probe_video_stream_duration
        run_ffmpeg = chunking._run_ffmpeg

        def stale_source_rate_duration(path):
            # Simulate the exact doubled expectation in the V8.2 report. Only
            # input metadata is stale; final output is always genuinely probed.
            if Path(path) in chunks:
                return (121 if Path(path) == chunks[0] else 119) / 24.0
            return probe_duration(path)

        for route, success_message in (
            ("direct", "stream copy"),
            ("ts", "TS stream copy"),
            ("demuxer", "concat-demuxer re-encode fallback"),
            ("framepipe", "frame-stream fallback"),
        ):
            with self.subTest(route=route), contextlib.ExitStack() as stack:
                messages = []
                output = self.tmp / f"merged_{route}.mp4"

                def fail_earlier_copy_routes(cmd):
                    is_copy_concat = "concat" in cmd and "copy" in cmd
                    is_ts_concat = any(Path(str(arg)).name == "concat_ts.txt" for arg in cmd)
                    if is_copy_concat and (
                        route == "demuxer" or (route == "ts" and not is_ts_concat)
                    ):
                        return subprocess.CompletedProcess(cmd, 1, "", "injected copy failure")
                    return run_ffmpeg(cmd)

                stack.enter_context(mock.patch.object(
                    chunking, "_probe_video_stream_duration", side_effect=stale_source_rate_duration
                ))
                stack.enter_context(mock.patch.object(
                    chunking, "_run_ffmpeg", side_effect=fail_earlier_copy_routes
                ))
                if route == "framepipe":
                    stack.enter_context(mock.patch.object(
                        chunking, "_merge_stream_copy_is_safe", return_value=(False, "incompatible time bases")
                    ))
                self.assertTrue(chunking.concat_videos(
                    chunks, output, encode_settings=self.encoding, on_progress=messages.append
                ), "".join(messages))
                self._assert_timeline(output, "48/1")
                self.assertTrue(any(success_message in message for message in messages), messages)
                self.assertFalse(any("duration drift" in message for message in messages), messages)

    def test_rife_rates_above_240_keep_the_exact_output_timeline(self):
        for fps in ("480/1", "480000/1001"):
            with self.subTest(fps=fps):
                chunks = self._chunks(fps)
                output = self.tmp / "high_fps_merged.mp4"
                messages = []
                self.assertTrue(chunking.concat_videos(
                    chunks, output, encode_settings=self.encoding, on_progress=messages.append
                ), "".join(messages))
                self._assert_timeline(output, fps)

    def test_framepipe_rejects_a_missing_frame_even_with_plausible_duration(self):
        chunks = self._chunks()
        probe_count = chunking._probe_chunk_nb_frames

        def incomplete_inventory(path):
            count = probe_count(path)
            return count + 1 if Path(path) == chunks[1] and count else count

        output = self.tmp / "must_not_publish.mp4"
        messages = []
        with (
            mock.patch.object(chunking, "_probe_chunk_nb_frames", side_effect=incomplete_inventory),
            mock.patch.object(chunking, "_merge_stream_copy_is_safe", return_value=(False, "incompatible time bases")),
        ):
            self.assertFalse(chunking.concat_videos(
                chunks, output, encode_settings=self.encoding, on_progress=messages.append
            ))
        self.assertFalse(output.exists())
        self.assertTrue(any("decoded 119/120 frames" in message for message in messages), messages)


if __name__ == "__main__":
    unittest.main()
