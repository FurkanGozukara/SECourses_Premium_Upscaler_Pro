"""End-to-end chunk regressions using public Wan 2.2 and LTX 2.3 outputs.

The media is intentionally kept outside git. Tests skip when the downloaded field samples
are absent; on the release workstation they exercise the real DiT-generated H.264/AAC files
at low spatial resolution while retaining their original frame inventories and timing.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import chunking  # noqa: E402
from shared.audio_utils import has_audio_stream  # noqa: E402
from shared.runner import RunResult  # noqa: E402

FIELD_ROOT = ROOT.parent / "test_media" / "field_samples"
WAN_SAMPLE = FIELD_ROOT / "wan22_official_demo.mp4"
LTX_SAMPLE = FIELD_ROOT / "ltx23_replicate_demo.mp4"
HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "ffmpeg failed")[-2000:])


def _probe(path: Path) -> dict:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,start_time,duration,nb_read_frames",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return (json.loads(proc.stdout or "{}").get("streams") or [{}])[0]


class _Runner:
    def reset_cancel_state(self) -> None:
        pass

    def is_canceled(self) -> bool:
        return False


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available")
class PublicDiTFieldSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(tempfile.mkdtemp(prefix="chunk_field_samples_"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def _prepare(self, source: Path, name: str, *, frames: int | None = None) -> Path:
        output = self.root / f"{name}.mp4"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a:0?", "-vf", "scale=160:90:flags=lanczos",
        ]
        if frames is not None:
            source_count = int(_probe(source).get("nb_read_frames") or 0)
            pad = max(0, int(frames) - source_count)
            rate = _probe(source).get("r_frame_rate") or "25/1"
            filters = ["scale=160:90:flags=lanczos"]
            if pad:
                filters.append(f"tpad=stop={pad}:stop_mode=clone")
            filters.append(f"setpts=N/({rate}*TB)")
            vf_index = cmd.index("-vf") + 1
            cmd[vf_index] = ",".join(filters)
            cmd += ["-frames:v", str(int(frames)), "-r", str(rate)]
        cmd += [
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", str(output),
        ]
        _run(cmd)
        return output

    def _pipeline(
        self,
        source: Path,
        *,
        model: str,
        expected_frames: int,
        fps: str,
        chunk_seconds: float,
        precise: bool,
        overlap: float = 0.0,
    ) -> None:
        case_dir = self.root / f"{source.stem}_{model}_{'precise' if precise else 'copy'}_{overlap}"
        messages: list[str] = []

        def process(settings: dict, on_progress=None) -> RunResult:
            src = Path(settings["input_path"])
            dst = Path(settings["output_override"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
                    "-map", "0:v:0", "-c:v", "libx264", "-preset", "ultrafast",
                    "-crf", "30", "-pix_fmt", "yuv420p", "-an", str(dst),
                ],
                capture_output=True,
                text=True,
            )
            return RunResult(proc.returncode, str(dst), proc.stderr or "")

        settings = {
            "input_path": str(source),
            "output_override": str(case_dir / "final.mp4"),
            "output_format": "mp4",
            "frame_accurate_split": precise,
            "video_codec": "h264",
            "video_preset": "ultrafast",
            "video_quality": 28,
            "pixel_format": "yuv420p",
            "audio_codec": "copy",
            "progress_report_interval_sec": 60,
        }
        with mock.patch.object(chunking, "detect_scenes", return_value=[]), contextlib.redirect_stdout(io.StringIO()):
            rc, log, output_path, chunk_count = chunking.chunk_and_process(
                runner=_Runner(),
                settings=settings,
                scene_threshold=27.0,
                min_scene_len=1.0,
                work_dir=case_dir / "work",
                on_progress=messages.append,
                chunk_seconds=chunk_seconds,
                chunk_overlap=overlap,
                per_chunk_cleanup=False,
                allow_partial=False,
                process_func=process,
                model_type=model,
            )
        self.assertEqual(rc, 0, f"{log}\n{''.join(messages)}")
        self.assertGreater(chunk_count, 1)
        output = Path(output_path)
        stream = _probe(output)
        self.assertEqual(int(stream.get("nb_read_frames") or 0), expected_frames)
        self.assertEqual(stream.get("r_frame_rate"), fps)
        self.assertAlmostEqual(float(stream.get("start_time") or 0.0), 0.0, delta=0.001)
        fps_value = float(chunking._parse_fraction_to_float(fps) or 0.0)
        self.assertAlmostEqual(float(stream.get("duration") or 0.0), expected_frames / fps_value, delta=0.06)
        self.assertEqual(has_audio_stream(output), has_audio_stream(source))
        self.assertNotIn("concat failed", "".join(messages).lower())

    @unittest.skipUnless(WAN_SAMPLE.exists(), "public Wan 2.2 sample not downloaded")
    def test_official_wan22_odd_2401_frame_video(self) -> None:
        source = self._prepare(WAN_SAMPLE, "wan22_2401")
        self._pipeline(
            source,
            model="flashvsr",
            expected_frames=2401,
            fps="30/1",
            chunk_seconds=7.3,
            precise=True,
        )

    @unittest.skipUnless(WAN_SAMPLE.exists(), "public Wan 2.2 sample not downloaded")
    def test_official_wan22_157_frame_copy_mode(self) -> None:
        source = self._prepare(WAN_SAMPLE, "wan22_157", frames=157)
        self._pipeline(
            source,
            model="flashvsr",
            expected_frames=157,
            fps="30/1",
            chunk_seconds=1.7,
            precise=False,
        )

    @unittest.skipUnless(LTX_SAMPLE.exists(), "public LTX 2.3 sample not downloaded")
    def test_ltx23_native_audio_and_odd_157_frame_overlap(self) -> None:
        source = self._prepare(LTX_SAMPLE, "ltx23_157", frames=157)
        self._pipeline(
            source,
            model="ltx25",
            expected_frames=157,
            fps="25/1",
            chunk_seconds=1.8,
            precise=True,
            overlap=0.24,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
