"""
Regression tests for chunk-split / merge audio-video sync.

Background (v8.0 bug report): a 40-minute FlashVSR+ run produced a final video with the
same duration as the source but progressively desynced audio and a missing ending.
Root causes:
  * PySceneDetect >= 0.7 returns scene boundaries rounded to microseconds, and
    `split_video()` aligned them with floor(start)/ceil(end) -> one duplicated frame at
    ~2/3 of all boundaries at 23.976/29.97/24/30/60 fps.
  * "Frame-Accurate Split" OFF used a keyframe-limited stream copy that keeps the frames
    between the previous keyframe and the requested start (up to a whole GOP per chunk).
  * `mux_audio()` used `-shortest` unconditionally, so the longer merged video was silently
    clipped to the audio length ("same duration, ending missing").
  * The concat demuxer offset chunks by container duration (audio track slightly longer
    than video) -> tiny timestamp gaps at every boundary.

These tests need ffmpeg/ffprobe on PATH; they are skipped otherwise.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import chunking  # noqa: E402
from shared.audio_utils import mux_audio  # noqa: E402
from shared.ltx25_runner import _validate_ltx25_video_output  # noqa: E402
from shared.rife_runner import run_video_concatenate  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _probe(path: Path, stream: str, entries: str) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries", f"stream={entries}", "-of", "json", str(path)],
        capture_output=True,
        text=True,
    )
    streams = json.loads(proc.stdout or "{}").get("streams") or [{}]
    return streams[0]


def _make_source(
    path: Path,
    fps: str = "24000/1001",
    seconds: float = 8.0,
    gop_seconds: float = 3.0,
    audio: bool = True,
    b_frames: int | None = None,
) -> int:
    """Synthetic CFR test clip with a long GOP (so chunk starts fall between keyframes)."""
    num, den = fps.split("/")
    fps_f = float(num) / float(den)
    gop = max(1, int(round(gop_seconds * fps_f)))
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=size=96x64:rate={fps}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "64k"]
    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
    ]
    if b_frames is not None:
        cmd += ["-bf", str(max(0, int(b_frames)))]
    cmd += ["-t", f"{seconds:.6f}", "-movflags", "+faststart", str(path)]
    subprocess.run(cmd, check=True)
    return int(_probe(path, "v:0", "nb_frames").get("nb_frames") or 0)


def _microsecond_rounded_scenes(fps_f: float, boundaries_frames: list[int], total_frames: int) -> list[tuple[float, float]]:
    """Mimic PySceneDetect 0.7 (OpenCV backend): timecodes rounded to whole microseconds."""
    pts = [0] + list(boundaries_frames) + [total_frames]
    secs = [round(round(f / fps_f * 1_000_000) / 1_000_000, 6) for f in pts]
    return [(secs[i], secs[i + 1]) for i in range(len(secs) - 1)]


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available")
class SplitVideoFrameCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="avsync_test_"))
        cls.src = cls.tmp / "src.mp4"
        cls.total_frames = _make_source(cls.src)
        cls.fps_f = 24000.0 / 1001.0

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _chunk_frames(self, chunks):
        return [int(_probe(p, "v:0", "nb_frames").get("nb_frames") or 0) for p in chunks]

    def test_microsecond_rounded_scene_boundaries_do_not_duplicate_frames(self):
        # Boundaries at frames not divisible by 3 -> their microsecond timecodes are inexact.
        scenes = _microsecond_rounded_scenes(self.fps_f, [37, 74, 112, 149], self.total_frames)
        work = self.tmp / "precise"
        chunks = chunking.split_video(str(self.src), scenes, work, precise=True, include_audio=True)
        self.assertEqual(len(chunks), 5)
        self.assertEqual(sum(self._chunk_frames(chunks)), self.total_frames)
        manifest = chunking._load_split_manifest(work)
        self.assertIsNotNone(manifest)
        entries = manifest["chunks"]
        for prev, cur in zip(entries, entries[1:]):
            self.assertEqual(prev["end_frame"], cur["start_frame"], "consecutive chunks must share the boundary frame")
        ok, msg = chunking._verify_split_coverage(str(self.src), chunks, work)
        self.assertTrue(ok, msg)

    def test_fixed_second_chunks_at_ntsc_rate_are_contiguous(self):
        scenes = chunking.fallback_scenes(str(self.src), chunk_seconds=1.7)
        work = self.tmp / "fixed"
        chunks = chunking.split_video(str(self.src), scenes, work, precise=True, include_audio=True)
        self.assertGreater(len(chunks), 2)
        self.assertEqual(sum(self._chunk_frames(chunks)), self.total_frames)

    def test_stream_copy_mode_falls_back_to_lossless_when_keyframe_limited(self):
        # GOP is 3 s; a chunk starting at 1.7 s cannot be cut with stream copy.
        scenes = chunking.fallback_scenes(str(self.src), chunk_seconds=1.7)
        work = self.tmp / "copy"
        messages = []
        chunks = chunking.split_video(
            str(self.src), scenes, work, precise=False, include_audio=True, on_progress=messages.append
        )
        self.assertEqual(sum(self._chunk_frames(chunks)), self.total_frames)
        self.assertTrue(any("not frame-accurate" in m for m in messages), messages)
        ok, msg = chunking._verify_split_coverage(str(self.src), chunks, work)
        self.assertTrue(ok, msg)

    def test_verify_split_coverage_rejects_overlapping_chunks(self):
        # Two chunks that both contain the same 20 frames.
        work = self.tmp / "bad"
        work.mkdir(exist_ok=True)
        f = self.fps_f
        a = work / "chunk_0001.mp4"
        b = work / "chunk_0002.mp4"
        for out, ss, t in ((a, 0.0, 100 / f), (b, 80 / f, (self.total_frames - 80) / f)):
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{ss:.6f}", "-i", str(self.src), "-t", f"{t:.6f}",
                 "-map", "0:v:0", "-vf", "setpts=PTS-STARTPTS", "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-an", str(out)],
                check=True,
            )
        ok, msg = chunking._verify_split_coverage(str(self.src), [a, b], work)
        self.assertFalse(ok)
        self.assertIn("duplicate", msg)

    def test_mp4_edit_list_discard_packet_is_not_counted_as_a_source_frame(self):
        """Regression for the field-reported odd-frame ``delta -1`` failure.

        A stream-copy cut commonly leaves one pre-roll packet in MP4 metadata with the
        DISCARD flag. ``nb_frames`` includes it; decoders and all processed chunks do not.
        """
        base = self.tmp / "edit_base.mp4"
        edited = self.tmp / "edit_list_157.mp4"
        # Disable B-frames so this fixture contains exactly one discarded pre-roll
        # packet. This keeps the regression pinned to the reported delta=-1 shape.
        _make_source(
            base,
            fps="25/1",
            seconds=8.0,
            gop_seconds=4.0,
            audio=False,
            b_frames=0,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", "0.04", "-i", str(base), "-t", "6.28",
                "-map", "0:v:0", "-c:v", "copy", "-an", str(edited),
            ],
            check=True,
        )
        metadata_frames = int(_probe(edited, "v:0", "nb_frames").get("nb_frames") or 0)
        self.assertEqual(metadata_frames, 158)
        self.assertEqual(chunking._probe_decodable_frames(edited), 157)
        self.assertEqual(chunking._probe_source_frame_count(edited), (157, "decoded_frames"))

        scenes = [(0.0, 1.56), (1.56, 3.12), (3.12, 4.68), (4.68, 6.28)]
        work = self.tmp / "edit_list_split"
        chunks = chunking.split_video(str(edited), scenes, work, precise=True, include_audio=False)
        self.assertEqual(sum(self._chunk_frames(chunks)), 157)
        ok, msg = chunking._verify_split_coverage(str(edited), chunks, work)
        self.assertTrue(ok, msg)

    def test_last_scene_is_anchored_to_decoded_odd_frame_inventory(self):
        # Simulate a detector/container reporting the end one frame early.
        scenes = [(0.0, 2.0), (2.0, (self.total_frames - 1) / self.fps_f)]
        work = self.tmp / "short_advertised_end"
        chunks = chunking.split_video(str(self.src), scenes, work, precise=True, include_audio=False)
        self.assertEqual(sum(self._chunk_frames(chunks)), self.total_frames)
        ok, msg = chunking._verify_split_coverage(str(self.src), chunks, work)
        self.assertTrue(ok, msg)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available")
class MergeAndMuxTests(unittest.TestCase):
    def test_merge_fps_keeps_exact_ntsc_rational(self):
        sigs = [{"r_frame_rate": "24000/1001", "avg_frame_rate": "24000/1001"}] * 3
        self.assertEqual(chunking._pick_merge_fps_str(sigs, []), "24000/1001")
        self.assertAlmostEqual(chunking._pick_merge_fps(sigs, []), 24000 / 1001, places=6)
        sigs30 = [{"r_frame_rate": "30000/1001", "avg_frame_rate": "30000/1001"}] * 3
        self.assertAlmostEqual(chunking._pick_merge_fps(sigs30, []), 30000 / 1001, places=6)

    def test_concat_list_writes_video_durations(self):
        tmp = Path(tempfile.mkdtemp(prefix="avsync_concat_"))
        try:
            txt = tmp / "concat.txt"
            chunking._write_concat_list(txt, [tmp / "a.mp4", tmp / "b.mp4"], durations=[1.5, None])
            lines = txt.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[1], "duration 1.500000000000")
            self.assertEqual(len([l for l in lines if l.startswith("duration")]), 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_mux_audio_never_truncates_video(self):
        tmp = Path(tempfile.mkdtemp(prefix="avsync_mux_"))
        try:
            long_video = tmp / "video6.mp4"
            short_video = tmp / "video4.mp4"
            long_audio = tmp / "audio6.mp4"
            short_audio = tmp / "audio4.mp4"
            n_long = _make_source(long_video, seconds=6.0, audio=False)
            n_short = _make_source(short_video, seconds=4.0, audio=False)
            _make_source(long_audio, seconds=6.0, audio=True)
            _make_source(short_audio, seconds=4.0, audio=True)

            # Video longer than audio: the video must stay complete (old code clipped it).
            out = tmp / "muxed_video_longer.mp4"
            ok, err = mux_audio(long_video, short_audio, out, audio_codec="copy")
            self.assertTrue(ok, err)
            self.assertEqual(int(_probe(out, "v:0", "nb_frames").get("nb_frames") or 0), n_long)

            # Audio longer than video (preview/partial runs): audio is clamped, video intact.
            out2 = tmp / "muxed_audio_longer.mp4"
            ok2, err2 = mux_audio(short_video, long_audio, out2, audio_codec="copy")
            self.assertTrue(ok2, err2)
            self.assertEqual(int(_probe(out2, "v:0", "nb_frames").get("nb_frames") or 0), n_short)
            a_dur = float(_probe(out2, "a:0", "duration").get("duration") or 0.0)
            v_dur = float(_probe(out2, "v:0", "duration").get("duration") or 0.0)
            self.assertLessEqual(a_dur, v_dur + 0.5)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_bframe_timestamp_repairs_and_single_chunk_merge_start_at_zero(self):
        tmp = Path(tempfile.mkdtemp(prefix="avsync_timestamp_"))
        try:
            shifted = tmp / "shifted.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=25",
                    "-frames:v", "100", "-c:v", "libx264", "-preset", "ultrafast",
                    "-crf", "20", "-bf", "3", "-avoid_negative_ts", "make_zero",
                    str(shifted),
                ],
                check=True,
            )
            shifted_start = float(_probe(shifted, "v:0", "start_time").get("start_time") or 0.0)
            self.assertGreater(shifted_start, 0.02)

            repaired = tmp / "repaired.mp4"
            self.assertTrue(chunking._remux_video_with_fresh_timestamps(shifted, repaired))
            self.assertAlmostEqual(
                float(_probe(repaired, "v:0", "start_time").get("start_time") or 0.0),
                0.0,
                delta=0.001,
            )
            self.assertEqual(chunking._probe_chunk_nb_frames(repaired), 100)

            merged = tmp / "single_merged.mp4"
            self.assertTrue(
                chunking.concat_videos(
                    [shifted],
                    merged,
                    encode_settings={"video_codec": "h264", "video_preset": "ultrafast"},
                    nominal_fps="25/1",
                )
            )
            self.assertEqual(chunking._probe_chunk_nb_frames(merged), 100)
            self.assertAlmostEqual(
                float(_probe(merged, "v:0", "start_time").get("start_time") or 0.0),
                0.0,
                delta=0.001,
            )

            stripped = tmp / "stripped.mp4"
            ok, err = mux_audio(shifted, tmp / "missing_audio_source.mp4", stripped, audio_codec="none")
            self.assertTrue(ok, err)
            self.assertEqual(chunking._probe_chunk_nb_frames(stripped), 100)
            self.assertAlmostEqual(
                float(_probe(stripped, "v:0", "start_time").get("start_time") or 0.0),
                0.0,
                delta=0.001,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_concat_demuxer_paths_support_apostrophes_spaces_and_unicode(self):
        tmp = Path(tempfile.mkdtemp(prefix="avsync_quote_"))
        try:
            media_dir = tmp / "editor's caf\u00e9 files"
            media_dir.mkdir(parents=True)
            source = media_dir / "source clip.mp4"
            _make_source(source, fps="25/1", seconds=1.0, audio=False)
            first = media_dir / "first's clip.mp4"
            second = media_dir / "second clip \u00e7.mp4"
            shutil.copy2(source, first)
            shutil.copy2(source, second)
            concat_file = tmp / "quoted_concat.txt"
            chunking._write_concat_list(concat_file, [first, second], durations=[1.0, 1.0])
            output = tmp / "quoted_merged.mp4"
            proc = subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", str(concat_file),
                    "-map", "0:v:0", "-c:v", "copy", "-an", str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(chunking._probe_chunk_nb_frames(output), 50)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_rife_editor_concat_uses_the_same_safe_path_grammar(self):
        tmp = Path(tempfile.mkdtemp(prefix="avsync_rife_quote_"))
        try:
            media_dir = tmp / "editor's clips"
            media_dir.mkdir(parents=True)
            source = media_dir / "source café.mp4"
            _make_source(source, fps="25/1", seconds=1.0, audio=False)
            first = media_dir / "first's clip.mp4"
            second = media_dir / "second clip.mp4"
            shutil.copy2(source, first)
            shutil.copy2(source, second)
            output = tmp / "rife editor output.mp4"
            self.assertTrue(run_video_concatenate([str(first), str(second)], str(output)))
            self.assertEqual(chunking._probe_chunk_nb_frames(output), 50)
            self.assertFalse(list(tmp.rglob("*.concat.txt")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_many_finished_chunks_are_stabilized_on_one_shared_poll_clock(self):
        tmp = Path(tempfile.mkdtemp(prefix="avsync_batch_ready_"))
        try:
            source = tmp / "source.mp4"
            _make_source(source, fps="25/1", seconds=0.4, audio=False)
            chunks = []
            for index in range(40):
                chunk = tmp / f"chunk_{index + 1:04d}.mp4"
                shutil.copy2(source, chunk)
                chunks.append(chunk)
            started = time.monotonic()
            ready = chunking._wait_for_media_files_ready(chunks, timeout_sec=5.0, poll_sec=0.1)
            elapsed = time.monotonic() - started
            self.assertEqual(ready, chunks)
            # The old sequential waiter took roughly 0.7 s * 40 chunks (>28 s).
            self.assertLess(elapsed, 6.0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_audio_mux_preserves_delayed_audio_relative_to_zero_video(self):
        tmp = Path(tempfile.mkdtemp(prefix="avsync_audio_offset_"))
        try:
            video = tmp / "video.mp4"
            source = tmp / "source_delayed.mp4"
            _make_source(video, fps="25/1", seconds=4.0, audio=False)
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(video), "-itsoffset", "0.4", "-f", "lavfi", "-t", "3.2",
                    "-i", "sine=frequency=660:sample_rate=48000", "-map", "0:v:0",
                    "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "64k",
                    str(source),
                ],
                check=True,
            )
            source_audio_start = float(_probe(source, "a:0", "start_time").get("start_time") or 0.0)
            output = tmp / "muxed.mp4"
            ok, err = mux_audio(video, source, output, audio_codec="copy")
            self.assertTrue(ok, err)
            output_video_start = float(_probe(output, "v:0", "start_time").get("start_time") or 0.0)
            output_audio_start = float(_probe(output, "a:0", "start_time").get("start_time") or 0.0)
            self.assertAlmostEqual(output_video_start, 0.0, delta=0.001)
            self.assertAlmostEqual(output_audio_start, source_audio_start, delta=0.03)
            self.assertEqual(chunking._probe_chunk_nb_frames(output), 100)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_mixed_time_bases_use_nonblocking_exact_frame_fallback(self):
        tmp = Path(tempfile.mkdtemp(prefix="avsync_mixed_tb_"))
        try:
            src = tmp / "src.mp4"
            _make_source(src, fps="25/1", seconds=4.0, audio=False)
            chunks = []
            for idx, timescale in enumerate((1000, 12800), 1):
                out = tmp / f"chunk_{idx:04d}.mp4"
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-ss", str((idx - 1) * 2), "-i", str(src), "-t", "2",
                        "-map", "0:v:0", "-c:v", "libx264", "-preset", "ultrafast",
                        "-crf", "20", "-pix_fmt", "yuv420p", "-video_track_timescale",
                        str(timescale), "-an", str(out),
                    ],
                    check=True,
                )
                chunks.append(out)
            messages = []
            merged = tmp / "merged.mp4"
            self.assertTrue(
                chunking.concat_videos(
                    chunks,
                    merged,
                    encode_settings={"video_codec": "h264", "video_preset": "ultrafast"},
                    on_progress=messages.append,
                    nominal_fps="25/1",
                )
            )
            self.assertEqual(chunking._probe_chunk_nb_frames(merged), 100)
            self.assertTrue(any("frame-stream fallback" in m for m in messages), messages)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_overlap_blend_removes_each_boundary_once(self):
        tmp = Path(tempfile.mkdtemp(prefix="avsync_blend_"))
        try:
            src = tmp / "src.mp4"
            total = _make_source(src, fps="25/1", seconds=6.28, audio=False)
            scenes = chunking.fallback_scenes(str(src), chunk_seconds=1.7, overlap_seconds=0.4)
            work = tmp / "chunks"
            chunks = chunking.split_video(str(src), scenes, work, precise=True, include_audio=False)
            overlaps = chunking._manifest_overlap_frames(chunking._load_split_manifest(work), len(chunks))
            self.assertIsNotNone(overlaps)
            merged = tmp / "blended.mp4"
            self.assertTrue(
                chunking.concat_videos_with_blending(
                    chunks,
                    merged,
                    overlap_frames=overlaps,
                    fps=25.0,
                    encode_settings={"video_codec": "h264", "video_preset": "ultrafast"},
                    nominal_fps="25/1",
                )
            )
            self.assertEqual(chunking._probe_chunk_nb_frames(merged), total)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available")
class ChunkOutputTimingGuardTests(unittest.TestCase):
    """The per-chunk output check must catch the drift classes seen in the field:
    wrong output frame rate (e.g. 24.49 instead of 25) and extra/missing frames."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="avsync_guard_"))
        cls.chunk = cls.tmp / "chunk_0001.mp4"
        cls.n = _make_source(cls.chunk, fps="25/1", seconds=4.0, gop_seconds=1.0, audio=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _reencode(self, out: Path, extra_args: list[str]) -> Path:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(self.chunk), "-map", "0:v:0",
             *extra_args, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p", str(out)],
            check=True,
        )
        return out

    def test_decodable_frame_count_matches_metadata_for_clean_file(self):
        self.assertEqual(chunking._probe_decodable_frames(self.chunk), self.n)
        self.assertEqual(chunking._probe_chunk_nb_frames(self.chunk), self.n)

    def test_identical_output_passes(self):
        out = self._reencode(self.tmp / "same.mp4", [])
        ok, detail = chunking._processed_chunk_matches_input(out, self.chunk, "flashvsr", 25.0)
        self.assertTrue(ok, detail)

    def test_wrong_output_fps_is_detected(self):
        # Same frames, but the writer stamped 24.4908 fps (the reporter's merged-file rate).
        out = self._reencode(self.tmp / "wrongfps.mp4", ["-vf", "setpts=N/(24.4908*TB)", "-r", "24.4908"])
        ok, detail = chunking._processed_chunk_matches_input(out, self.chunk, "flashvsr", 25.0)
        self.assertFalse(ok, detail)

    def test_extra_frames_are_detected(self):
        # Two frames appended (like a keyframe pre-roll or padding leak): +2 frames = 0.08 s,
        # beyond the 1.5-frame duration tolerance -> rejected (the reporter's +2 frames/chunk case).
        out = self._reencode(self.tmp / "extra.mp4", ["-vf", "tpad=stop=2:stop_mode=clone"])
        ok, detail = chunking._processed_chunk_matches_input(out, self.chunk, "flashvsr", 25.0)
        self.assertFalse(ok, detail)

    def test_one_extra_frame_is_detected_by_frame_count(self):
        out = self._reencode(self.tmp / "extra1.mp4", ["-vf", "tpad=stop=1:stop_mode=clone"])
        ok, detail = chunking._processed_chunk_matches_input(out, self.chunk, "flashvsr", 25.0)
        self.assertFalse(ok, detail)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available")
class IncompleteModelArtifactTests(unittest.TestCase):
    """An interrupted writer may leave audio-only MP4 data that must never reach concat."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="avsync_incomplete_model_"))
        self.source = self.tmp / "source.mp4"
        _make_source(self.source, fps="17/1", seconds=1.0, audio=True)
        self.audio_only = self.tmp / "audio_only.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000",
                "-t", "1", "-c:a", "aac", str(self.audio_only),
            ],
            check=True,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_audio_only_mp4_is_not_a_ready_model_video(self):
        started = time.monotonic()
        self.assertFalse(
            chunking._wait_for_media_file_ready(
                self.audio_only,
                expected_duration=1.0,
                timeout_sec=0.6,
                poll_sec=0.05,
            )
        )
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(
            chunking._wait_for_media_files_ready(
                [self.audio_only], timeout_sec=0.6, poll_sec=0.05
            ),
            [],
        )

    def test_ltx_output_validation_rejects_audio_only_artifact(self):
        ok, detail = _validate_ltx25_video_output(
            self.source,
            self.audio_only,
            expected_fps=17.0,
        )
        self.assertFalse(ok)
        self.assertIn("no usable video stream", detail)

        timing_ok, timing_detail = chunking._processed_chunk_matches_input(
            self.audio_only,
            self.source,
            "ltx25",
            17.0,
        )
        self.assertFalse(timing_ok)
        self.assertIn("no usable video stream", timing_detail)

    def test_ltx_output_validation_accepts_complete_frame_inventory(self):
        ok, detail = _validate_ltx25_video_output(
            self.source,
            self.source,
            expected_fps=17.0,
        )
        self.assertTrue(ok, detail)

    def test_success_result_with_unready_file_is_a_structured_chunk_failure(self):
        class _Runner:
            @staticmethod
            def reset_cancel_state():
                return None

            @staticmethod
            def is_canceled() -> bool:
                return False

        class _Result:
            returncode = 0
            output_path = str(self.audio_only)
            log = ""

        settings = {
            "input_path": str(self.source),
            "output_override": str(self.tmp / "final.mp4"),
            "output_format": "mp4",
            "frame_accurate_split": True,
            "audio_codec": "copy",
            "video_codec": "h264",
        }
        with (
            mock.patch.object(chunking, "detect_scenes", return_value=[(0.0, 1.0)]),
            mock.patch.object(chunking, "_wait_for_media_file_ready", return_value=False),
            mock.patch.object(chunking, "concat_videos_with_blending") as concat_mock,
        ):
            rc, log, _output, _count = chunking.chunk_and_process(
                runner=_Runner(),
                settings=settings,
                scene_threshold=27.0,
                min_scene_len=1.0,
                work_dir=self.tmp / "invalid_result_work",
                on_progress=lambda _message: None,
                process_func=lambda _settings: _Result(),
                model_type="ltx25",
                allow_partial=False,
            )

        self.assertEqual(rc, 1)
        self.assertIn("missing or was not finalized", log)
        concat_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
