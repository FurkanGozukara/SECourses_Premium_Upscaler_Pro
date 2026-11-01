"""LTX 2.5 planning regressions (no GPU needed):

- the live plan / Auto Tune report plans the window the engine runs: Tail Padding Frames included, skipped when
  it would add a chunk (2026-11-01 acceptance: Auto Tune planned 49 frames, the engine ran 57);
- the VAE decode tile choice never counts Save VRAM as available and only picks large tiles that fit with margin
  (the same run chose 1024 px tiles from an estimate that included Save VRAM and ran out of memory).

    venv\Scripts\python.exe -m unittest tools.test_ltx25_plan
"""
import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
GB = 1024 ** 3


def _planner():
    ltx_dir = str(ROOT / "LTX25")
    if ltx_dir not in sys.path:
        sys.path.insert(0, ltx_dir)
    return importlib.import_module("ltx25_engine.planner")


class TailPadPlanTests(unittest.TestCase):
    def test_tail_pad_is_planned(self):
        p = _planner()

        def budget(frames):
            return p.auto_token_budget(frames, 960, 540, free_vram_bytes=int(30.8 * GB), weight_resident_bytes=0,
                                       reserve_gb=1.5)[0]

        plan, used = p.plan_with_tail_pad(49, 960, 540, 8, budget, max_chunk_frames=497)
        self.assertEqual(used, 8)
        self.assertEqual(plan.chunk_count, 1)
        self.assertEqual(plan.chunk_length, 57)
        self.assertEqual(plan.latent_frames, 8)
        self.assertEqual(plan.effective_tokens, 16320)

    def test_tail_pad_skipped_when_it_adds_a_chunk(self):
        p = _planner()

        def budget(frames):  # 1080p -> 4K with the VRAM free on a 32 GB card: 49 frames fit one chunk, 57 do not
            return p.auto_token_budget(frames, 1920, 1080, free_vram_bytes=int(28.2 * GB), weight_resident_bytes=0,
                                       reserve_gb=1.5)[0]

        plain = p.plan_chunks(49, 1920, 1080, budget(49), max_chunk_frames=497)
        plan, used = p.plan_with_tail_pad(49, 1920, 1080, 8, budget, max_chunk_frames=497)
        self.assertEqual(plain.chunk_count, 1)
        self.assertEqual(used, 0)
        self.assertEqual(plan.chunk_count, 1)
        self.assertEqual(plan.chunk_length, 49)

    def test_live_plan_text_counts_tail_pad(self):
        source = Path(r"G:/SECourses_Premium_Upscaler_Pro_v8/upstream_scan_20261101/eval/inputs/robot_lr2x_960x540_49f.mp4")
        if not source.is_file():
            self.skipTest("eval input not present")
        sys.path.insert(0, str(ROOT))
        from shared.services import ltx25_service as svc

        settings = dict(svc.ltx25_defaults(), input_path=str(source), max_chunk_frames=497, tail_pad_frames=8)
        with mock.patch.object(svc, "_free_vram_bytes_for_plan", return_value=(int(30.8 * GB), "test GPU")):
            text = svc.build_live_plan_text(settings, ROOT)
        self.assertIn("Tail padding: 8", text)
        self.assertIn("Chunk length 57 frames (8 latent frames)", text)
        self.assertIn("16320", text)


class DecodeTileTests(unittest.TestCase):
    def test_save_vram_is_not_available(self):
        p = _planner()
        # Acceptance run: cap 28.8 GB, 19.5 GB reserved after sampling, device free 12.3 GB, Save VRAM 1.5 GB.
        avail = p.decode_available_bytes(28.8 * GB, 19.5 * GB, 12.3 * GB, 1.5 * GB)
        self.assertAlmostEqual(avail / GB, 9.3, places=1)

    def test_acceptance_case_keeps_512(self):
        p = _planner()
        avail = p.decode_available_bytes(28.8 * GB, 19.5 * GB, 12.3 * GB, 1.5 * GB)
        self.assertEqual(p.choose_decode_tiles(avail, 8), p.DECODE_TILE_DEFAULT)

    def test_large_tiles_only_with_room(self):
        p = _planner()
        self.assertEqual(p.choose_decode_tiles(25 * GB, 8)[0], 2048)       # transformer released before decode
        self.assertEqual(p.choose_decode_tiles(14 * GB, 8)[0], 1536)
        self.assertEqual(p.choose_decode_tiles(15 * GB, 19)[0], 1024)      # long windows need more
        self.assertEqual(p.choose_decode_tiles(10 * GB, 19), p.DECODE_TILE_DEFAULT)
        for tiles, short, long in p.DECODE_TILE_SETS:                       # every pick keeps the full margin
            for t in (8, 19, 32):
                need = p.decode_tile_need(short, long, t)
                self.assertNotEqual(
                    p.choose_decode_tiles(need * p.DECODE_SAFETY + p.DECODE_MARGIN_BYTES - 1, t)[0], tiles)


if __name__ == "__main__":
    unittest.main()
