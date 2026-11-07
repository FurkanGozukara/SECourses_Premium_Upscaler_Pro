"""The chunk preview gallery lists each chunk once, and Start clears the previous run's gallery (V10.1 fix).

Seen in the tutorial recording on 7 November 2026 (RelayVSR, a seven-scene movie): the gallery showed chunk 1 twice,
and pressing Start Upscaling showed the previous run's gallery until the new job reported its first progress.

Cause 1: build_chunk_preview_payload() listed every folder inside processed_chunks as a chunk. Its own preview copies
live in processed_chunks/.ui_preview_cache, so from the second refresh on that folder was listed first (its first
video is chunk 1's copy) and every real chunk was numbered one too high.
Cause 2: the tabs clear the preview only inside the job (the service's first report), not when Start is pressed.

The first test builds real chunk files with ffmpeg; the second builds the real interface without launching it and
takes the first update of each tab's Start event. CPU only.
"""
from __future__ import annotations

import copy
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from shared.chunk_preview import build_chunk_preview_payload  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
GALLERY_TABS = {"flashvsr": "flashvsr_chunk_preview", "relayvsr": "relayvsr_chunk_preview",
                "open_h3": "open_h3_chunk_preview", "ltx25": "ltx25_chunk_preview",
                "sparkvsr": "sparkvsr_chunk_preview", "gan": "gan_chunk_preview"}


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not on PATH")
class ChunkPreviewPayloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="chunk_gallery_"))
        self.run = self.tmp / "0031"
        chunks = self.run / "processed_chunks"
        chunks.mkdir(parents=True)
        for i, color in enumerate(("red", "green", "blue"), 1):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:size=64x36:rate=10:d=1",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(chunks / f"chunk_{i:04d}_upscaled.mp4")],
                           check=True)
        (chunks / "thumbs").mkdir()  # the run's thumbnails folder (output_run_manager makes it)
        shutil.copy2(chunks / "chunk_0001_upscaled.mp4", chunks / "thumbs" / "chunk_0001.mp4")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_each_chunk_is_listed_once_in_order(self):
        for refresh in (1, 2, 3):  # the first refresh writes .ui_preview_cache, later ones must not list it
            with self.subTest(refresh=refresh):
                payload = build_chunk_preview_payload(str(self.run))
                self.assertEqual([label for _thumb, label in payload["gallery"]], ["Chunk 1", "Chunk 2", "Chunk 3"])
                self.assertEqual(payload["count"], 3)
                stems = [Path(v).name.split(".__ui_preview_")[0] for v in payload["videos"]]
                self.assertEqual(stems, ["chunk_0001_upscaled", "chunk_0002_upscaled", "chunk_0003_upscaled"])


class _Built(Exception):
    def __init__(self, demo):
        super().__init__("UI built")
        self.demo = demo


def _build_demo():
    import secourses_app

    def _capture(self, *args, **kwargs):
        raise _Built(self)

    with patch.object(gr.Blocks, "launch", _capture):
        try:
            secourses_app.main([])
        except _Built as built:
            return built.demo
    raise AssertionError("secourses_app.main() returned without launching the UI")


def _walk(block):
    yield block
    for child in getattr(block, "children", None) or []:
        yield from _walk(child)


class StartClearsTheGalleryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = _build_demo()

    def _start_dep(self, key):
        tab = next(b for b in self.demo.blocks.values() if isinstance(b, gr.Tab) and b.elem_id == f"sec-tab-{key}")
        inside = {b._id for b in _walk(tab)}
        deps = [d for d in self.demo.fns.values() if d.fn is not None and getattr(d.fn, "__name__", "") == "_run"
                and any(t[0] in inside and t[1] == "click" for t in (d.targets or []))]
        self.assertEqual(len(deps), 1, f"{key}: Start events found: {len(deps)}")
        return deps[0]

    def test_start_clears_the_previous_runs_gallery(self):
        old_preview = {"message": "Detected 7 processed chunk preview(s).", "count": 7,
                       "gallery": [(f"thumb_{i}.jpg", f"Chunk {i}") for i in range(1, 8)],
                       "videos": [f"chunk_{i:04d}_upscaled.mp4" for i in range(1, 8)]}
        for key, preview_key in GALLERY_TABS.items():
            with self.subTest(tab=key):
                dep = self._start_dep(key)
                state_comp = dep.inputs[-1]
                self.assertIsInstance(state_comp, gr.State)
                state = copy.deepcopy(state_comp.value)
                state.setdefault("seed_controls", {})[preview_key] = copy.deepcopy(old_preview)
                args = [copy.deepcopy(getattr(c, "value", None)) for c in dep.inputs[:-1]] + [state]
                updates = dep.fn(*args)
                try:
                    first = next(updates)  # what the browser gets the moment Start is pressed
                finally:
                    updates.close()
                out_state = first[dep.outputs.index(state_comp)]
                preview = out_state["seed_controls"][preview_key]
                self.assertEqual(preview.get("gallery"), [], f"{key}: the previous run's gallery is still shown")
                self.assertEqual(preview.get("videos"), [])


if __name__ == "__main__":
    unittest.main()
