"""
Regression tests for the Resolution & Scene Split tab's "Split Scenes to MP4".

Bug (7 November 2026): the standalone split called `split_video()` once per scene with a
one-range list. `split_video()` anchors the first start and the last end of its list to the
source's decoded edges, so every one-range call became the whole video: scene_0001.mp4 ..
scene_0007.mp4 each held all 768 frames of a 32 s movie. The service now calls
`split_video()` once with the complete scene list, like the chunked upscale does.

These tests need ffmpeg/ffprobe on PATH; they are skipped otherwise. CPU only.
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

from shared import chunking  # noqa: E402
from shared.services.resolution_service import build_resolution_callbacks  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
FPS = 24
TOTAL_FRAMES = 72  # 3 s


def _frame_md5s(path: Path) -> list[str]:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "framemd5", "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [ln.rsplit(",", 1)[-1].strip() for ln in proc.stdout.splitlines() if ln and not ln.startswith("#")]


def _has_audio(path: Path) -> bool:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "json", str(path)],
        capture_output=True,
        text=True,
    )
    return bool(json.loads(proc.stdout or "{}").get("streams"))


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available")
class StandaloneSceneSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="standalone_split_", ignore_cleanup_errors=True)
        cls.root = Path(cls._tmp.name)
        cls.src = cls.root / "movie.mp4"
        # Long GOP so the scene starts fall between keyframes; AAC audio.
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", f"testsrc2=size=96x64:rate={FPS}",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-g", "72", "-keyint_min", "72", "-sc_threshold", "0",
                "-c:a", "aac", "-b:a", "64k",
                "-frames:v", str(TOTAL_FRAMES), "-t", "3.0", str(cls.src),
            ],
            check=True,
        )
        cls.src_md5 = _frame_md5s(cls.src)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _split(self, case: str, scenes: list[tuple[float, float]], frame_accurate: bool = True) -> tuple[str, list[Path]]:
        out_root = self.root / case
        callbacks = build_resolution_callbacks(preset_manager=None, shared_state=None, models=[], base_dir=out_root)
        state = {"seed_controls": {"global_settings": {"output_dir": str(out_root)}}}
        with mock.patch.object(chunking, "detect_scenes", return_value=list(scenes)):
            updates = list(
                callbacks["standalone_scene_split"](
                    str(self.src), True, frame_accurate, 0, 0.0, 27.0, 1.0, state, progress=lambda *a, **k: None
                )
            )
        message = str(updates[-1][0].get("value") or "")
        split_dir = Path(state["seed_controls"].get("last_standalone_scene_split_dir") or out_root / "missing")
        return message, sorted(split_dir.glob("scene_*.mp4"))

    def _assert_scene_files(self, files: list[Path], expected_frames: list[int]) -> None:
        self.assertEqual([p.name for p in files], [f"scene_{i:04d}.mp4" for i in range(1, len(expected_frames) + 1)])
        self.assertEqual([chunking._probe_chunk_nb_frames(p) for p in files], expected_frames)
        start = 0
        for path, count in zip(files, expected_frames):
            self.assertEqual(_frame_md5s(path), self.src_md5[start:start + count], f"{path.name} frames differ")
            self.assertTrue(_has_audio(path), f"{path.name} lost the audio")
            start += count
        self.assertEqual(start, TOTAL_FRAMES)

    def test_each_scene_file_holds_only_its_own_frames(self) -> None:
        scenes = [(0.0, 1.0), (1.0, 1.583333), (1.583333, 3.0)]  # frames 0-24, 24-38, 38-72
        message, files = self._split("frame_accurate", scenes, frame_accurate=True)
        self.assertTrue(message.startswith("SUCCESS"), message)
        self._assert_scene_files(files, [24, 14, 34])

    def test_fast_stream_copy_split_is_also_exact(self) -> None:
        scenes = [(0.0, 1.0), (1.0, 1.583333), (1.583333, 3.0)]
        message, files = self._split("stream_copy", scenes, frame_accurate=False)
        self.assertTrue(message.startswith("SUCCESS"), message)
        self._assert_scene_files(files, [24, 14, 34])

    def test_single_whole_file_scene_is_exported_whole(self) -> None:
        message, files = self._split("single_scene", [(0.0, 3.0)])
        self.assertTrue(message.startswith("SUCCESS"), message)
        self._assert_scene_files(files, [TOTAL_FRAMES])


if __name__ == "__main__":
    unittest.main()
