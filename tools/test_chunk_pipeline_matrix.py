"""Low-resolution combinatorial regression matrix for the universal chunk pipeline.

The suite intentionally exercises timing/container combinations rather than model math.
Per-model GPU inference is covered by model smoke runs; here a deterministic ffmpeg
processor lets every model route traverse the same split, validation, merge, and audio
code that production uses.
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
from shared.audio_utils import ensure_audio_on_video, has_audio_stream, mux_audio  # noqa: E402
from shared.runner import RunResult  # noqa: E402
from shared.video_fps_utils import apply_video_fps_override_preprocess, remux_video_fps  # noqa: E402
from LTX25.ltx25_engine.video_io import _format_fps_rate  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "command failed")[-2000:])


def _fraction(value: str) -> float:
    num, den = value.split("/", 1)
    return float(num) / float(den)


def _probe_stream(path: Path, selector: str, fields: str) -> dict:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", selector,
            "-show_entries", f"stream={fields}", "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(proc.stdout or "{}").get("streams") or []
    return streams[0] if streams else {}


def _encoder_available(name: str) -> bool:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-h", f"encoder={name}"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and "not recognized" not in (proc.stderr or "").lower()


def _make_video(
    path: Path,
    *,
    fps: str,
    frames: int,
    encoder: str = "libx264",
    gop: int | None = None,
    size: str = "64x48",
) -> None:
    encoder_args = {
        "libx264": ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "24", "-pix_fmt", "yuv420p"],
        "libx265": [
            "-c:v", "libx265", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
            "-x265-params", "log-level=error",
        ],
        "libvpx-vp9": [
            "-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8",
            "-crf", "38", "-b:v", "0", "-pix_fmt", "yuv420p",
        ],
        "libaom-av1": [
            "-c:v", "libaom-av1", "-cpu-used", "8", "-crf", "42", "-b:v", "0",
            "-row-mt", "1", "-pix_fmt", "yuv420p",
        ],
        "prores_ks": ["-c:v", "prores_ks", "-profile:v", "0", "-pix_fmt", "yuv422p10le"],
        "mpeg4": ["-c:v", "mpeg4", "-q:v", "4", "-pix_fmt", "yuv420p"],
        "ffv1": ["-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", "yuv420p"],
        "mpeg2video": ["-c:v", "mpeg2video", "-q:v", "4", "-pix_fmt", "yuv420p"],
        "wmv2": ["-c:v", "wmv2", "-q:v", "4", "-pix_fmt", "yuv420p"],
        "mjpeg": ["-c:v", "mjpeg", "-q:v", "4", "-pix_fmt", "yuvj420p"],
    }[encoder]
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={fps}",
        *encoder_args,
    ]
    if gop:
        cmd += ["-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0"]
    cmd += ["-frames:v", str(frames), "-an", str(path)]
    _run(cmd)


def _add_audio(
    video: Path,
    output: Path,
    *,
    duration: float,
    codec: str,
    delay: float = 0.0,
) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video),
    ]
    if delay:
        cmd += ["-itsoffset", f"{delay:.6f}"]
    cmd += [
        "-f", "lavfi", "-t", f"{duration:.6f}",
        "-i", "sine=frequency=440:sample_rate=48000",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", codec,
    ]
    if codec == "aac":
        cmd += ["-b:a", "64k"]
    elif codec == "libopus":
        cmd += ["-b:a", "64k"]
    cmd.append(str(output))
    _run(cmd)


def _scenes_for_frames(frame_count: int, fps: str) -> list[tuple[float, float]]:
    rate = _fraction(fps)
    boundaries = sorted({0, max(1, frame_count // 3), max(2, (2 * frame_count) // 3), frame_count})
    boundaries = [n for n in boundaries if 0 <= n <= frame_count]
    if boundaries[-1] != frame_count:
        boundaries.append(frame_count)
    return [
        (round(boundaries[i] / rate, 6), round(boundaries[i + 1] / rate, 6))
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] > boundaries[i]
    ]


def _frame_hashes(path: Path) -> list[str]:
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-map", "0:v:0", "-f", "framemd5", "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.rsplit(",", 1)[-1].strip() for line in proc.stdout.splitlines() if line and not line.startswith("#")]


def _frame_samples(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    samples: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(cv2.resize(frame, (24, 18), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            samples.append(gray.astype(np.int16))
    finally:
        cap.release()
    return samples


class _FakeRunner:
    def reset_cancel_state(self) -> None:
        pass

    def is_canceled(self) -> bool:
        return False


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available")
class ChunkPipelineMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(tempfile.mkdtemp(prefix="chunk_matrix_"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def _roundtrip(
        self,
        source: Path,
        *,
        fps: str,
        frame_count: int,
        case_name: str,
        precise: bool = True,
        include_audio: bool = False,
    ) -> Path:
        case_dir = self.root / case_name
        chunks_dir = case_dir / "chunks"
        messages: list[str] = []
        chunks = chunking.split_video(
            str(source),
            _scenes_for_frames(frame_count, fps),
            chunks_dir,
            precise=precise,
            include_audio=include_audio,
            on_progress=messages.append,
        )
        self.assertGreater(len(chunks), 1, messages)
        self.assertEqual(sum(int(chunking._probe_chunk_nb_frames(p) or 0) for p in chunks), frame_count)
        split_ok, split_detail = chunking._verify_split_coverage(
            str(source), chunks, chunks_dir, on_progress=messages.append
        )
        self.assertTrue(split_ok, split_detail + "\n" + "".join(messages))

        merged = case_dir / "merged.mp4"
        self.assertTrue(
            chunking.concat_videos(
                chunks,
                merged,
                encode_settings={"video_codec": "h264", "video_preset": "ultrafast", "video_quality": 28},
                on_progress=messages.append,
                nominal_fps=fps,
            ),
            "".join(messages),
        )
        self.assertEqual(chunking._probe_chunk_nb_frames(merged), frame_count)
        actual_duration = chunking._probe_video_stream_duration(merged)
        self.assertIsNotNone(actual_duration)
        self.assertLessEqual(abs(float(actual_duration) - (frame_count / _fraction(fps))), 1.5 / _fraction(fps))
        self.assertFalse(list(case_dir.glob("*.ffconcat")), "merge list was not cleaned up")
        return merged

    def test_odd_even_and_short_frame_inventory_matrix(self) -> None:
        counts = [1, 2, 3, 4, 5, 31, 32, 33, 79, 80, 81, 120, 121, 155, 156, 157, 158, 159, 160, 161]
        for count in counts:
            with self.subTest(frames=count):
                source = self.root / f"inventory_{count}.mp4"
                _make_video(source, fps="25/1", frames=count, gop=100)
                self.assertEqual(chunking._probe_source_frame_count(source)[0], count)
                if count < 3:
                    self.assertEqual(chunking.fallback_scenes(str(source), chunk_seconds=0), [(0.0, count / 25.0)])
                    continue
                self._roundtrip(
                    source,
                    fps="25/1",
                    frame_count=count,
                    case_name=f"inventory_{count}_roundtrip",
                    precise=(count % 2 == 1),
                )

    def test_fractional_and_integer_frame_rate_matrix(self) -> None:
        rates = ["16/1", "24000/1001", "24/1", "25/1", "30000/1001", "30/1", "50/1", "60/1"]
        for index, fps in enumerate(rates):
            with self.subTest(fps=fps):
                source = self.root / f"rate_{index}.mp4"
                _make_video(source, fps=fps, frames=47, gop=120)
                self._roundtrip(
                    source,
                    fps=fps,
                    frame_count=47,
                    case_name=f"rate_{index}_roundtrip",
                    precise=(index % 2 == 0),
                )

    def test_source_codec_and_container_matrix(self) -> None:
        cases = [
            ("h264_mp4", "libx264", ".mp4"),
            ("hevc_mp4", "libx265", ".mp4"),
            ("h264_mov", "libx264", ".mov"),
            ("prores_mov", "prores_ks", ".mov"),
            ("vp9_webm", "libvpx-vp9", ".webm"),
            ("av1_mkv", "libaom-av1", ".mkv"),
            ("mpeg4_avi", "mpeg4", ".avi"),
            ("ffv1_mkv", "ffv1", ".mkv"),
            ("h264_ts", "libx264", ".ts"),
            ("mpeg2_ts", "mpeg2video", ".ts"),
            ("h264_flv", "libx264", ".flv"),
            ("wmv_asf", "wmv2", ".wmv"),
            ("mjpeg_avi", "mjpeg", ".avi"),
        ]
        for name, encoder, suffix in cases:
            if not _encoder_available(encoder):
                continue
            with self.subTest(case=name):
                source = self.root / f"{name}{suffix}"
                _make_video(source, fps="30000/1001", frames=61, encoder=encoder, gop=120)
                self._roundtrip(
                    source,
                    fps="30000/1001",
                    frame_count=61,
                    case_name=f"{name}_roundtrip",
                )

    def test_high_bit_depth_and_chroma_format_matrix(self) -> None:
        cases = [
            ("hevc_10bit", "libx265", "yuv420p10le", ["-x265-params", "log-level=error"]),
            ("h264_444", "libx264", "yuv444p", []),
            ("prores_10bit", "prores_ks", "yuv422p10le", ["-profile:v", "2"]),
        ]
        for name, encoder, pix_fmt, extra in cases:
            if not _encoder_available(encoder):
                continue
            with self.subTest(case=name):
                source = self.root / f"{name}.mov"
                cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=24000/1001",
                    "-frames:v", "61", "-c:v", encoder,
                ]
                if encoder in {"libx264", "libx265"}:
                    cmd += ["-preset", "ultrafast", "-crf", "24"]
                cmd += ["-pix_fmt", pix_fmt, *extra, "-an", str(source)]
                _run(cmd)
                self._roundtrip(
                    source,
                    fps="24000/1001",
                    frame_count=61,
                    case_name=f"{name}_roundtrip",
                )

    def test_incompatible_chunk_fallback_honors_output_codec_matrix(self) -> None:
        """Every selectable output codec must survive the robust per-file fallback path."""
        first = self.root / "fallback_source_h264.mp4"
        second = self.root / "fallback_source_mpeg4.avi"
        _make_video(first, fps="25/1", frames=7, encoder="libx264", size="128x96")
        _make_video(second, fps="25/1", frames=7, encoder="mpeg4", size="128x96")

        cases = [
            ("h264", "libx264", "h264", "yuv420p", ".mp4", {}),
            ("h265", "libx265", "hevc", "yuv420p10le", ".mp4", {"use_10bit": True}),
            ("vp9", "libvpx-vp9", "vp9", "yuv420p", ".mp4", {}),
            ("av1", "libsvtav1", "av1", "yuv420p", ".mp4", {}),
            # FFmpeg intentionally rejects ProRes in an MP4 muxer; MOV is its supported
            # ISO-BMFF container and is also accepted by concat_videos callers.
            ("prores", "prores_ks", "prores", "yuv422p10le", ".mov", {}),
        ]
        for codec, encoder, expected_codec, pixel_format, suffix, extra in cases:
            if not _encoder_available(encoder):
                continue
            with self.subTest(codec=codec):
                output = self.root / f"fallback_output_{codec}{suffix}"
                messages: list[str] = []
                settings = {
                    "video_codec": codec,
                    "video_preset": "ultrafast",
                    "video_quality": 24,
                    "pixel_format": pixel_format,
                    **extra,
                }
                self.assertTrue(
                    chunking.concat_videos(
                        [first, second],
                        output,
                        encode_settings=settings,
                        on_progress=messages.append,
                        nominal_fps="25/1",
                    ),
                    "".join(messages),
                )
                stream = _probe_stream(
                    output,
                    "v:0",
                    "codec_name,pix_fmt,r_frame_rate,start_time,duration,nb_frames",
                )
                self.assertEqual(stream.get("codec_name"), expected_codec)
                self.assertEqual(stream.get("r_frame_rate"), "25/1")
                self.assertAlmostEqual(float(stream.get("start_time") or 0.0), 0.0, delta=0.001)
                self.assertEqual(chunking._probe_chunk_nb_frames(output), 14)
                self.assertAlmostEqual(float(stream.get("duration") or 0.0), 14 / 25.0, delta=0.06)
                self.assertFalse(list(self.root.glob(f".{output.stem}.*.ffconcat")))

    def test_vfr_input_is_canonicalized_without_losing_frames(self) -> None:
        first = self.root / "vfr_30.mp4"
        second = self.root / "vfr_15.mp4"
        source = self.root / "vfr.mkv"
        _make_video(first, fps="30/1", frames=30)
        _make_video(second, fps="15/1", frames=15)
        concat_file = self.root / "vfr_concat.txt"
        concat_file.write_text(
            f"file '{first.as_posix()}'\nfile '{second.as_posix()}'\n",
            encoding="utf-8",
        )
        _run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(concat_file),
                "-map", "0:v:0", "-c:v", "libx264", "-preset", "ultrafast",
                "-crf", "24", "-fps_mode", "vfr", "-an", str(source),
            ]
        )
        count = int(chunking._probe_source_frame_count(source)[0] or 0)
        self.assertEqual(count, 45)
        avg_rate = _probe_stream(source, "v:0", "avg_frame_rate").get("avg_frame_rate")
        self.assertTrue(avg_rate)
        merged = self._roundtrip(
            source,
            fps=str(avg_rate),
            frame_count=count,
            case_name="vfr_roundtrip",
        )
        self.assertEqual(chunking._probe_chunk_nb_frames(merged), 45)
        manifest = chunking._load_split_manifest(self.root / "vfr_roundtrip" / "chunks")
        self.assertIsNotNone(manifest)
        self.assertTrue(manifest.get("timeline_fps_rational"))

        # A VFR stream can advertise r_frame_rate=30 even though half of its frames
        # are timed at 15 fps. A same-nominal override must still build a real CFR
        # inventory instead of taking the old "FPS already matches" shortcut.
        settings = {"input_path": str(source), "fps": 30.0}
        ok, note = apply_video_fps_override_preprocess(
            settings,
            fps_key="fps",
            run_dir=self.root / "vfr_same_nominal_override",
        )
        self.assertTrue(ok, note)
        converted = Path(settings["input_path"])
        self.assertNotEqual(converted.resolve(), source.resolve())
        self.assertEqual(chunking._probe_chunk_nb_frames(converted), 60)
        converted_stream = _probe_stream(converted, "v:0", "r_frame_rate,start_time,duration")
        self.assertEqual(converted_stream.get("r_frame_rate"), "30/1")
        self.assertAlmostEqual(float(converted_stream.get("start_time") or 0.0), 0.0, delta=0.001)
        self.assertAlmostEqual(float(converted_stream.get("duration") or 0.0), 2.0, delta=0.06)

    def test_split_merge_preserves_decoded_frame_order_and_content(self) -> None:
        for index, (fps, frames, precise) in enumerate(
            (("24000/1001", 157, True), ("30000/1001", 161, False), ("25/1", 81, True))
        ):
            with self.subTest(fps=fps, frames=frames, precise=precise):
                source = self.root / f"content_{index}.mp4"
                _make_video(source, fps=fps, frames=frames, gop=120)
                merged = self._roundtrip(
                    source,
                    fps=fps,
                    frame_count=frames,
                    case_name=f"content_{index}_roundtrip",
                    precise=precise,
                )
                if precise:
                    self.assertEqual(_frame_hashes(merged), _frame_hashes(source))
                else:
                    expected = _frame_samples(source)
                    actual = _frame_samples(merged)
                    self.assertEqual(len(actual), len(expected))
                    frame_errors = [float(np.abs(got - want).mean()) for got, want in zip(actual, expected)]
                    self.assertLess(max(frame_errors, default=0.0), 8.0)

    def test_bframe_and_long_gop_split_matrix(self) -> None:
        for index, (b_frames, gop, precise) in enumerate(
            ((0, 1, False), (0, 120, True), (2, 15, False), (3, 120, True), (8, 240, False))
        ):
            if not _encoder_available("libx264"):
                continue
            with self.subTest(b_frames=b_frames, gop=gop, precise=precise):
                source = self.root / f"bframes_{index}.mp4"
                cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=25",
                    "-frames:v", "157", "-c:v", "libx264", "-preset", "ultrafast",
                    "-crf", "24", "-pix_fmt", "yuv420p", "-bf", str(b_frames),
                    "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
                    "-an", str(source),
                ]
                _run(cmd)
                self._roundtrip(
                    source,
                    fps="25/1",
                    frame_count=157,
                    case_name=f"bframes_{index}_roundtrip",
                    precise=precise,
                )

    def test_real_fps_override_changes_inventory_not_playback_duration(self) -> None:
        video = self.root / "fps_override_base.mp4"
        source = self.root / "fps_override_source.mp4"
        _make_video(video, fps="25/1", frames=100, gop=100)
        _add_audio(video, source, duration=4.0, codec="aac")

        for target, expected_frames in ((30.0, 120), (15.0, 60), (24000 / 1001, 96)):
            with self.subTest(target=target):
                output = self.root / f"fps_override_{str(target).replace('.', '_')}.mp4"
                ok, error = remux_video_fps(source, output, target)
                self.assertTrue(ok, error)
                stream = _probe_stream(
                    output,
                    "v:0",
                    "r_frame_rate,avg_frame_rate,start_time,duration,nb_frames",
                )
                actual_rate = _fraction(stream["r_frame_rate"])
                self.assertAlmostEqual(actual_rate, target, delta=max(1e-6, target * 1e-5))
                self.assertAlmostEqual(float(stream.get("start_time") or 0.0), 0.0, delta=0.001)
                self.assertEqual(chunking._probe_chunk_nb_frames(output), expected_frames)
                self.assertAlmostEqual(float(stream.get("duration") or 0.0), 4.0, delta=0.11)
                self.assertTrue(has_audio_stream(output))

        self.assertEqual(_format_fps_rate(24000 / 1001), "24000/1001")
        self.assertEqual(_format_fps_rate(25.0), "25/1")

    def test_wrong_backend_fps_with_same_frames_is_repaired_at_merge(self) -> None:
        video_only = self.root / "wrong_backend_fps_base.mp4"
        source = self.root / "wrong_backend_fps_source.mp4"
        _make_video(video_only, fps="25/1", frames=81, gop=80)
        _add_audio(video_only, source, duration=81 / 25.0, codec="aac")

        def process(settings: dict, on_progress=None) -> RunResult:
            input_path = Path(settings["input_path"])
            output_path = Path(settings["output_override"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(input_path), "-map", "0:v:0",
                    "-vf", "setpts=N/(24.4908*TB)", "-r", "24.4908",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                    "-pix_fmt", "yuv420p", "-an", str(output_path),
                ],
                capture_output=True,
                text=True,
            )
            return RunResult(proc.returncode, str(output_path), proc.stderr or "")

        case_dir = self.root / "wrong_backend_fps_pipeline"
        messages: list[str] = []
        settings = {
            "input_path": str(source),
            "output_override": str(case_dir / "final.mp4"),
            "output_format": "mp4",
            "frame_accurate_split": True,
            "video_codec": "h264",
            "video_preset": "ultrafast",
            "video_quality": 28,
            "pixel_format": "yuv420p",
            "audio_codec": "copy",
            "progress_report_interval_sec": 60,
        }
        with (
            mock.patch.object(chunking, "detect_scenes", return_value=[]),
            mock.patch.object(
                chunking,
                "_wait_for_media_file_ready",
                side_effect=lambda path, **_kwargs: Path(path).exists() and Path(path).stat().st_size > 1024,
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rc, log, output_path, chunk_count = chunking.chunk_and_process(
                runner=_FakeRunner(),
                settings=settings,
                scene_threshold=27.0,
                min_scene_len=1.0,
                work_dir=case_dir / "work",
                on_progress=messages.append,
                chunk_seconds=1.2,
                chunk_overlap=0.0,
                per_chunk_cleanup=False,
                allow_partial=False,
                process_func=process,
                model_type="flashvsr",
            )
        self.assertEqual(rc, 0, f"{log}\n{''.join(messages)}")
        self.assertGreaterEqual(chunk_count, 2)
        output = Path(output_path)
        self.assertEqual(chunking._probe_chunk_nb_frames(output), 81)
        stream = _probe_stream(output, "v:0", "r_frame_rate,start_time,duration")
        self.assertEqual(stream.get("r_frame_rate"), "25/1")
        self.assertAlmostEqual(float(stream.get("start_time") or 0.0), 0.0, delta=0.001)
        self.assertAlmostEqual(float(stream.get("duration") or 0.0), 81 / 25.0, delta=0.06)
        self.assertTrue(has_audio_stream(output))
        self.assertTrue(any("repairable" in message.lower() for message in messages), messages)

    def test_ltx_app_chunking_applies_fps_override_once_before_splitting(self) -> None:
        video_only = self.root / "ltx_fps_base.mp4"
        source = self.root / "ltx_fps_source.mp4"
        _make_video(video_only, fps="25/1", frames=100, gop=100)
        _add_audio(video_only, source, duration=4.0, codec="aac")

        def process(settings: dict, on_progress=None) -> RunResult:
            input_path = Path(settings["input_path"])
            output_path = Path(settings["output_override"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(input_path), "-map", "0:v:0", "-c:v", "libx264",
                    "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p",
                    "-an", str(output_path),
                ],
                capture_output=True,
                text=True,
            )
            return RunResult(proc.returncode, str(output_path), proc.stderr or "")

        case_dir = self.root / "ltx_fps_pipeline"
        messages: list[str] = []
        settings = {
            "input_path": str(source),
            "output_override": str(case_dir / "final.mp4"),
            "output_format": "mp4",
            "frame_accurate_split": True,
            "video_codec": "h264",
            "video_preset": "ultrafast",
            "video_quality": 28,
            "pixel_format": "yuv420p",
            "audio_codec": "copy",
            "fps": 30.0,
            "progress_report_interval_sec": 60,
        }
        with (
            mock.patch.object(chunking, "detect_scenes", return_value=[]),
            mock.patch.object(
                chunking,
                "_wait_for_media_file_ready",
                side_effect=lambda path, **_kwargs: Path(path).exists() and Path(path).stat().st_size > 1024,
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rc, log, output_path, chunk_count = chunking.chunk_and_process(
                runner=_FakeRunner(),
                settings=settings,
                scene_threshold=27.0,
                min_scene_len=1.0,
                work_dir=case_dir / "work",
                on_progress=messages.append,
                chunk_seconds=1.1,
                chunk_overlap=0.0,
                per_chunk_cleanup=False,
                allow_partial=False,
                process_func=process,
                model_type="ltx25",
            )
        self.assertEqual(rc, 0, f"{log}\n{''.join(messages)}")
        self.assertGreaterEqual(chunk_count, 3)
        output = Path(output_path)
        self.assertEqual(chunking._probe_chunk_nb_frames(output), 120)
        stream = _probe_stream(output, "v:0", "r_frame_rate,start_time,duration")
        self.assertEqual(stream.get("r_frame_rate"), "30/1")
        self.assertAlmostEqual(float(stream.get("duration") or 0.0), 4.0, delta=0.06)
        self.assertAlmostEqual(float(stream.get("start_time") or 0.0), 0.0, delta=0.001)
        self.assertTrue(has_audio_stream(output))
        self.assertEqual(float(settings.get("fps") or 0.0), 0.0)
        self.assertTrue(settings.get("_fps_override_preprocessed_input_path"))

    def test_audio_codec_length_and_offset_matrix(self) -> None:
        video = self.root / "audio_base.mp4"
        _make_video(video, fps="25/1", frames=157)
        merged = self._roundtrip(
            video,
            fps="25/1",
            frame_count=157,
            case_name="audio_video_roundtrip",
        )
        variants = [
            ("aac_short", ".mp4", "aac", 4.0, 0.0),
            ("aac_equal", ".mp4", "aac", 6.28, 0.0),
            ("aac_long", ".mp4", "aac", 8.0, 0.0),
            ("aac_delayed", ".mp4", "aac", 5.5, 0.4),
            ("pcm_mov", ".mov", "pcm_s16le", 6.28, 0.0),
            ("opus_mkv", ".mkv", "libopus", 6.28, 0.0),
            ("adpcm_mkv", ".mkv", "adpcm_ima_wav", 6.28, 0.0),
        ]
        for name, suffix, codec, duration, delay in variants:
            if not _encoder_available(codec):
                continue
            with self.subTest(audio=name):
                source = self.root / f"{name}{suffix}"
                _add_audio(video, source, duration=duration, codec=codec, delay=delay)
                output = self.root / f"{name}_muxed.mp4"
                ok, detail = mux_audio(merged, source, output, audio_codec="copy")
                self.assertTrue(ok, detail)
                self.assertEqual(chunking._probe_chunk_nb_frames(output), 157)
                self.assertTrue(has_audio_stream(output))
                if codec == "adpcm_ima_wav":
                    self.assertEqual(_probe_stream(output, "a:0", "codec_name").get("codec_name"), "aac")

        no_audio = self.root / "no_audio_copy.mp4"
        shutil.copy2(merged, no_audio)
        changed, final_path, error = ensure_audio_on_video(
            no_audio, video, audio_codec="copy", force_replace=True
        )
        self.assertFalse(changed, error)
        self.assertEqual(final_path, no_audio)
        self.assertFalse(has_audio_stream(no_audio))

    def test_five_minute_audio_many_chunk_timeline_has_no_cumulative_drift(self) -> None:
        """Exercise the long-video path cheaply: 7,501 frames across 43 chunks."""
        frames = 7_501
        fps = 25
        duration = frames / fps
        video_only = self.root / "long_many_chunk_base.mp4"
        source = self.root / "long_many_chunk_source.mp4"
        _run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", f"color=c=black:size=32x24:rate={fps}",
                "-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast",
                "-crf", "35", "-g", "250", "-bf", "3", "-pix_fmt", "yuv420p",
                "-an", str(video_only),
            ]
        )
        _add_audio(video_only, source, duration=duration, codec="aac")

        def copy_processor(settings: dict, on_progress=None) -> RunResult:
            output_path = Path(settings["output_override"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(settings["input_path"]), output_path)
            return RunResult(0, str(output_path), "")

        case_dir = self.root / "long_many_chunk_roundtrip"
        messages: list[str] = []
        settings = {
            "input_path": str(source),
            "output_override": str(case_dir / "final.mp4"),
            "output_format": "mp4",
            "frame_accurate_split": True,
            "video_codec": "h264",
            "video_preset": "ultrafast",
            "video_quality": 28,
            "pixel_format": "yuv420p",
            "audio_codec": "copy",
            "progress_report_interval_sec": 60,
        }
        with (
            mock.patch.object(chunking, "detect_scenes", return_value=[]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rc, log, output_path, chunk_count = chunking.chunk_and_process(
                runner=_FakeRunner(),
                settings=settings,
                scene_threshold=27.0,
                min_scene_len=1.0,
                work_dir=case_dir / "work",
                on_progress=messages.append,
                chunk_seconds=7.0,
                chunk_overlap=0.0,
                per_chunk_cleanup=False,
                allow_partial=False,
                process_func=copy_processor,
                model_type="gan",
            )

        self.assertEqual(rc, 0, f"{log}\n{''.join(messages)}")
        self.assertGreaterEqual(chunk_count, 40)
        output = Path(output_path)
        self.assertEqual(chunking._probe_chunk_nb_frames(output), frames)
        video_stream = _probe_stream(output, "v:0", "r_frame_rate,start_time,duration")
        audio_stream = _probe_stream(output, "a:0", "start_time,duration")
        self.assertEqual(video_stream.get("r_frame_rate"), "25/1")
        self.assertAlmostEqual(float(video_stream.get("start_time") or 0.0), 0.0, delta=0.001)
        self.assertAlmostEqual(float(audio_stream.get("start_time") or 0.0), 0.0, delta=0.03)
        self.assertAlmostEqual(float(video_stream.get("duration") or 0.0), duration, delta=0.04)
        self.assertLessEqual(
            abs(float(video_stream.get("duration") or 0.0) - float(audio_stream.get("duration") or 0.0)),
            0.08,
        )

    def test_resume_reprocesses_an_audio_only_stale_chunk(self) -> None:
        video_only = self.root / "resume_video.mp4"
        source = self.root / "resume_source.mp4"
        _make_video(video_only, fps="25/1", frames=50, gop=100)
        _add_audio(video_only, source, duration=2.0, codec="aac")

        case_dir = self.root / "resume_audio_only"
        processed_dir = case_dir / "work" / "processed_chunks"
        processed_dir.mkdir(parents=True, exist_ok=True)
        stale_chunk = processed_dir / "chunk_0001_upscaled.mp4"
        _run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-t", "1.0", "-i", "sine=frequency=330:sample_rate=48000",
                "-c:a", "aac", "-b:a", "64k", "-vn", str(stale_chunk),
            ]
        )

        calls: list[str] = []

        def copy_processor(settings: dict, on_progress=None) -> RunResult:
            calls.append(Path(settings["input_path"]).name)
            output_path = Path(settings["output_override"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(settings["input_path"]), output_path)
            return RunResult(0, str(output_path), "")

        messages: list[str] = []
        settings = {
            "input_path": str(source),
            "output_override": str(case_dir / "final.mp4"),
            "output_format": "mp4",
            "frame_accurate_split": True,
            "video_codec": "h264",
            "video_preset": "ultrafast",
            "video_quality": 28,
            "pixel_format": "yuv420p",
            "audio_codec": "copy",
            "progress_report_interval_sec": 60,
        }
        with (
            mock.patch.object(chunking, "detect_scenes", return_value=[]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rc, log, output_path, chunk_count = chunking.chunk_and_process(
                runner=_FakeRunner(),
                settings=settings,
                scene_threshold=27.0,
                min_scene_len=1.0,
                work_dir=case_dir / "work",
                on_progress=messages.append,
                chunk_seconds=1.0,
                chunk_overlap=0.0,
                per_chunk_cleanup=False,
                allow_partial=False,
                resume_from_partial=True,
                process_func=copy_processor,
                model_type="gan",
            )

        self.assertEqual(rc, 0, f"{log}\n{''.join(messages)}")
        self.assertEqual(chunk_count, 2)
        self.assertEqual(calls, ["chunk_0001.mp4", "chunk_0002.mp4"])
        self.assertTrue(any("not a finalized video" in message for message in messages), messages)
        self.assertEqual(chunking._probe_chunk_nb_frames(Path(output_path)), 50)
        self.assertTrue(has_audio_stream(Path(output_path)))

    def test_multiple_audio_tracks_sample_rates_and_offsets_survive_final_mux(self) -> None:
        video = self.root / "multi_audio_video.mp4"
        source = self.root / "multi_audio_source.mp4"
        _make_video(video, fps="25/1", frames=157, gop=100)
        _run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video),
                "-f", "lavfi", "-t", "6.28", "-i", "sine=frequency=440:sample_rate=44100",
                "-itsoffset", "0.35", "-f", "lavfi", "-t", "5.6", "-i",
                "sine=frequency=880:sample_rate=48000",
                "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "64k",
                "-metadata:s:a:0", "language=eng", "-metadata:s:a:1", "language=tur",
                str(source),
            ]
        )
        video_only = self._roundtrip(
            video,
            fps="25/1",
            frame_count=157,
            case_name="multi_audio_video_roundtrip",
        )
        output = self.root / "multi_audio_muxed.mp4"
        ok, detail = mux_audio(video_only, source, output, audio_codec="copy")
        self.assertTrue(ok, detail)
        self.assertEqual(chunking._probe_chunk_nb_frames(output), 157)
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=index,start_time,sample_rate:stream_tags=language",
                "-of", "json", str(output),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        streams = json.loads(probe.stdout or "{}").get("streams") or []
        self.assertEqual(len(streams), 2)
        self.assertEqual({stream.get("sample_rate") for stream in streams}, {"44100", "48000"})
        self.assertEqual({(stream.get("tags") or {}).get("language") for stream in streams}, {"eng", "tur"})
        starts = sorted(float(stream.get("start_time") or 0.0) for stream in streams)
        self.assertLessEqual(abs(starts[0]), 0.03)
        self.assertAlmostEqual(starts[1], 0.35, delta=0.04)

    def test_all_supported_model_routes_complete_the_same_exact_pipeline(self) -> None:
        video_only = self.root / "models_base.mp4"
        source = self.root / "models_source.mp4"
        _make_video(video_only, fps="25/1", frames=81)
        _add_audio(video_only, source, duration=81 / 25.0, codec="aac")

        active_model = ""

        def process(settings: dict, on_progress=None) -> RunResult:
            input_path = Path(settings["input_path"])
            output_path = Path(settings["output_override"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            rife_frames = None
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(input_path), "-map", "0:v:0",
            ]
            if active_model == "rife":
                cmd += ["-vf", "fps=50"]
                input_frames = int(chunking._probe_chunk_nb_frames(input_path) or 0)
                rife_frames = max(1, (input_frames * 2) - 1)
            cmd += [
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                "-pix_fmt", "yuv420p", "-an",
            ]
            if rife_frames is not None:
                cmd += ["-frames:v", str(rife_frames)]
            cmd.append(str(output_path))
            proc = subprocess.run(cmd, capture_output=True, text=True)
            return RunResult(proc.returncode, str(output_path), proc.stderr or "")

        def ready(path: Path, **_kwargs) -> bool:
            path = Path(path)
            return path.exists() and path.is_file() and path.stat().st_size > 1024

        runner = _FakeRunner()
        model_types = ["seedvr2", "gan", "rife", "flashvsr", "sparkvsr", "ltx25", "rtx"]
        for model in model_types:
            with self.subTest(model=model):
                active_model = model
                case_dir = self.root / f"model_{model}"
                messages: list[str] = []
                settings = {
                    "input_path": str(source),
                    "output_override": str(case_dir / "final.mp4"),
                    "output_format": "mp4",
                    "frame_accurate_split": True,
                    "video_codec": "h264",
                    "video_preset": "ultrafast",
                    "video_quality": 28,
                    "pixel_format": "yuv420p",
                    "audio_codec": "copy",
                    "progress_report_interval_sec": 60,
                }
                with (
                    mock.patch.object(chunking, "detect_scenes", return_value=[]),
                    mock.patch.object(chunking, "_wait_for_media_file_ready", side_effect=ready),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    rc, log, output_path, chunk_count = chunking.chunk_and_process(
                        runner=runner,
                        settings=settings,
                        scene_threshold=27.0,
                        min_scene_len=1.0,
                        work_dir=case_dir / "work",
                        on_progress=messages.append,
                        chunk_seconds=1.2,
                        chunk_overlap=0.0,
                        per_chunk_cleanup=False,
                        allow_partial=False,
                        process_func=process,
                        model_type=model,
                    )
                self.assertEqual(rc, 0, f"{log}\n{''.join(messages)}")
                self.assertGreaterEqual(chunk_count, 2)
                output = Path(output_path)
                self.assertTrue(output.exists(), output_path)
                self.assertTrue(has_audio_stream(output))
                self.assertAlmostEqual(
                    float(chunking._probe_video_stream_duration(output) or 0.0),
                    81 / 25.0,
                    delta=0.12,
                )
                expected_frames = 161 if model == "rife" else 81
                self.assertEqual(chunking._probe_chunk_nb_frames(output), expected_frames)

    def test_rife_multiplier_chunk_boundaries_keep_the_full_interpolated_timeline(self) -> None:
        video_only = self.root / "rife_boundary_base.mp4"
        source = self.root / "rife_boundary_source.mp4"
        _make_video(video_only, fps="17/1", frames=17)
        _add_audio(video_only, source, duration=1.0, codec="aac")

        for multiplier in (2, 4, 8):
            with self.subTest(multiplier=multiplier):
                case_dir = self.root / f"rife_x{multiplier}_boundary"

                def process(settings: dict, on_progress=None) -> RunResult:
                    input_path = Path(settings["input_path"])
                    output_path = Path(settings["output_override"])
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    input_frames = int(chunking._probe_chunk_nb_frames(input_path) or 0)
                    output_frames = (multiplier * max(0, input_frames - 1)) + 1
                    proc = subprocess.run(
                        [
                            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                            "-i", str(input_path), "-map", "0:v:0",
                            "-vf", f"fps={17 * multiplier}",
                            "-frames:v", str(output_frames),
                            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                            "-pix_fmt", "yuv420p", "-an", str(output_path),
                        ],
                        capture_output=True,
                        text=True,
                    )
                    return RunResult(proc.returncode, str(output_path), proc.stderr or "")

                settings = {
                    "input_path": str(source),
                    "output_override": str(case_dir / "final.mp4"),
                    "output_format": "mp4",
                    "frame_accurate_split": True,
                    "video_codec": "h264",
                    "video_preset": "ultrafast",
                    "video_quality": 28,
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
                        runner=_FakeRunner(),
                        settings=settings,
                        scene_threshold=27.0,
                        min_scene_len=1.0,
                        work_dir=case_dir / "work",
                        on_progress=lambda _message: None,
                        chunk_seconds=9 / 17,
                        chunk_overlap=0.0,
                        per_chunk_cleanup=False,
                        allow_partial=False,
                        process_func=process,
                        model_type="rife",
                    )
                self.assertEqual(rc, 0, log)
                self.assertEqual(chunk_count, 2)
                output = Path(output_path)
                self.assertEqual(
                    chunking._probe_chunk_nb_frames(output),
                    (multiplier * (17 - 1)) + 1,
                )
                stream = _probe_stream(output, "v:0", "r_frame_rate,start_time")
                self.assertEqual(stream.get("r_frame_rate"), f"{17 * multiplier}/1")
                self.assertAlmostEqual(float(stream.get("start_time") or 0.0), 0.0, delta=0.001)
                self.assertTrue(has_audio_stream(output))

    def test_rife_noninteger_target_fps_reconciles_per_chunk_rounding(self) -> None:
        video_only = self.root / "rife_target_base.mp4"
        source = self.root / "rife_target_source.mp4"
        _make_video(video_only, fps="17/1", frames=20)
        _add_audio(video_only, source, duration=20 / 17, codec="aac")
        target_fps = 30

        def process(settings: dict, on_progress=None) -> RunResult:
            input_path = Path(settings["input_path"])
            output_path = Path(settings["output_override"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            input_frames = int(chunking._probe_chunk_nb_frames(input_path) or 0)
            output_frames = int(round((input_frames - 1) * target_fps / 17)) + 1
            proc = subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(input_path), "-map", "0:v:0", "-vf", f"fps={target_fps}",
                    "-frames:v", str(output_frames), "-c:v", "libx264", "-preset", "ultrafast",
                    "-crf", "30", "-pix_fmt", "yuv420p", "-an", str(output_path),
                ],
                capture_output=True,
                text=True,
            )
            return RunResult(proc.returncode, str(output_path), proc.stderr or "")

        case_dir = self.root / "rife_target_rounding"
        messages: list[str] = []
        settings = {
            "input_path": str(source),
            "output_override": str(case_dir / "final.mp4"),
            "output_format": "mp4",
            "frame_accurate_split": True,
            "video_codec": "h264",
            "video_preset": "ultrafast",
            "video_quality": 28,
            "pixel_format": "yuv420p",
            "audio_codec": "copy",
            "target_fps": float(target_fps),
            "fps_multiplier": "x2",
            "progress_report_interval_sec": 60,
        }
        with (
            mock.patch.object(chunking, "detect_scenes", return_value=[]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rc, log, output_path, chunk_count = chunking.chunk_and_process(
                runner=_FakeRunner(),
                settings=settings,
                scene_threshold=27.0,
                min_scene_len=1.0,
                work_dir=case_dir / "work",
                on_progress=messages.append,
                chunk_seconds=9 / 17,
                chunk_overlap=0.0,
                per_chunk_cleanup=False,
                allow_partial=False,
                process_func=process,
                model_type="rife",
            )
        self.assertEqual(rc, 0, f"{log}\n{''.join(messages)}")
        self.assertEqual(chunk_count, 2)
        output = Path(output_path)
        self.assertEqual(chunking._probe_chunk_nb_frames(output), 35)
        stream = _probe_stream(output, "v:0", "r_frame_rate,start_time,duration")
        self.assertEqual(stream.get("r_frame_rate"), "30/1")
        self.assertAlmostEqual(float(stream.get("duration") or 0.0), 35 / 30, delta=0.04)
        self.assertTrue(any("required_boundary_removal=0" in message for message in messages), messages)
        self.assertTrue(has_audio_stream(output))

    def test_model_none_and_exception_are_structured_failures(self) -> None:
        source = self.root / "failure_source.mp4"
        _make_video(source, fps="25/1", frames=31)
        runner = _FakeRunner()

        def raises(_settings: dict):
            raise RuntimeError("synthetic model failure")

        for name, processor in (("none", lambda _settings: None), ("raises", raises)):
            with self.subTest(failure=name):
                settings = {
                    "input_path": str(source),
                    "output_override": str(self.root / f"failure_{name}.mp4"),
                    "output_format": "mp4",
                    "frame_accurate_split": True,
                    "video_codec": "h264",
                    "audio_codec": "none",
                }
                with (
                    mock.patch.object(chunking, "detect_scenes", return_value=[]),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    rc, log, _output, count = chunking.chunk_and_process(
                        runner=runner,
                        settings=settings,
                        scene_threshold=27.0,
                        min_scene_len=1.0,
                        work_dir=self.root / f"failure_{name}_work",
                        on_progress=lambda _message: None,
                        chunk_seconds=0.6,
                        allow_partial=False,
                        process_func=processor,
                        model_type="gan",
                    )
                self.assertEqual(rc, 1)
                self.assertGreater(count, 0)
                self.assertIn("Chunk 1", log)


if __name__ == "__main__":
    unittest.main(verbosity=2)
