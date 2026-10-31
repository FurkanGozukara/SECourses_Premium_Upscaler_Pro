"""Unit tests for the instant VRAM-budget Auto Tune (no GPU needed).

    venv\\Scripts\\python.exe -m unittest tools.test_vram_autotune
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.vram_autotune import common  # noqa: E402
from shared.vram_autotune import flashvsr as flash_plan  # noqa: E402
from shared.vram_autotune import rtx as rtx_plan  # noqa: E402
from shared.vram_autotune import seedvr2 as seed_plan  # noqa: E402
from shared.vram_autotune import sparkvsr as spark_plan  # noqa: E402

GPU_SIZES_GB = (8.0, 10.0, 12.0, 16.0, 24.0, 32.0, 48.0, 80.0)
# (input w, h) covering 16:9, 9:16, 1:1, 21:9, 4:3, 5:2
INPUTS = ((640, 360), (360, 640), (512, 512), (840, 360), (640, 480), (800, 320), (1280, 720), (1920, 1080))


def fake_gpu(total_gb: float, used_gb: float = 1.0, index: int = 0) -> common.GpuMemory:
    return common.GpuMemory(index=index, name="Test GPU", uuid=f"GPU-test-{index}", total_gb=total_gb, reserved_gb=0.3,
                            used_gb=used_gb, free_gb=total_gb - used_gb - 0.3)


def budget(total_gb: float, save_gb: float = 1.5, used_gb: float = 1.0, overhead: float = 1.0) -> common.VramBudget:
    return common.VramBudget(gpu=fake_gpu(total_gb, used_gb), save_gb=save_gb, overhead_gb=overhead)


class SaveVramTests(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(common.clamp_save_vram_gb(None), 1.5)
        self.assertEqual(common.clamp_save_vram_gb(""), 1.5)
        self.assertEqual(common.clamp_save_vram_gb(0), 0.0)
        self.assertEqual(common.clamp_save_vram_gb("abc"), 1.5)
        self.assertEqual(common.clamp_save_vram_gb(1000), common.SAVE_VRAM_MAX_GB)
        self.assertEqual(common.clamp_save_vram_gb(-3), 0.0)
        self.assertEqual(common.clamp_save_vram_gb(5.04), 5.0)

    def test_budget_is_total_minus_in_use_minus_save(self):
        # The user's example: 31.8 GB GPU, 1.5 GB in use, Save VRAM 1 GB -> 29.3 GB for the run.
        gpu = common.GpuMemory(0, "RTX 5090", "u", total_gb=31.8, reserved_gb=0.0, used_gb=1.5, free_gb=30.3)
        b = common.VramBudget(gpu=gpu, save_gb=1.0, overhead_gb=0.6)
        self.assertAlmostEqual(b.limit_gb, 29.3, places=3)
        self.assertAlmostEqual(b.allocator_cap_gb, 28.7, places=3)
        self.assertAlmostEqual(b.fraction, 28.7 / 31.8, places=4)

    def test_merge_alloc_conf_replaces_fraction(self):
        merged = common.merge_alloc_conf("expandable_segments:True,per_process_memory_fraction:0.5,garbage_collection_threshold:0.8", 0.75)
        self.assertIn("per_process_memory_fraction:0.7500", merged)
        self.assertEqual(merged.count("per_process_memory_fraction"), 1)
        self.assertIn("garbage_collection_threshold:0.8", merged)

    def test_cap_env_strips_async_backend_and_legacy_var(self):
        gpus = {0: fake_gpu(32.0, 2.0)}
        env = {"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync,garbage_collection_threshold:0.8"}
        with mock.patch.object(common, "query_gpu_memory", return_value=gpus):
            b = common.apply_vram_cap_env(env, "0", 1.5)
        self.assertIsNotNone(b)
        self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", env)
        self.assertNotIn("cudaMallocAsync", env["PYTORCH_ALLOC_CONF"])
        self.assertIn("per_process_memory_fraction", env["PYTORCH_ALLOC_CONF"])
        self.assertEqual(env[common.ENV_SAVE_VRAM], "1.5")

    def test_cap_env_skips_cpu(self):
        env = {}
        self.assertIsNone(common.apply_vram_cap_env(env, "cpu", 1.5))
        self.assertEqual(env, {})

    def test_multi_gpu_uses_tightest_budget(self):
        gpus = {0: fake_gpu(32.0, 2.0, 0), 1: fake_gpu(24.0, 4.0, 1)}
        env = {}
        with mock.patch.object(common, "query_gpu_memory", return_value=gpus):
            b = common.apply_vram_cap_env(env, "0,1", 1.5)
        self.assertEqual(b.gpu.index, 1)

    def test_round_helpers(self):
        self.assertEqual(common.round_up_4n1(363), 365)
        self.assertEqual(common.round_down_4n1(363), 361)
        self.assertEqual(common.round_up_8n1(66), 73)
        self.assertEqual(common.round_down_8n1(66), 65)


class SeedVR2PlannerTests(unittest.TestCase):
    MODEL = "seedvr2_ema_7b_sharp_fp16.safetensors"

    def _plan(self, total_gb, in_wh, resolution=1080, frames=121, save=1.5, model=None, int8=True):
        settings = {"dit_model": model or self.MODEL, "int8_convrot": int8, "resolution": resolution,
                    "max_resolution": 0}
        return seed_plan.plan(settings, in_w=in_wh[0], in_h=in_wh[1], frames=frames,
                              budget=budget(total_gb, save, overhead=seed_plan.SEEDVR2_PROCESS_OVERHEAD_GB))

    def test_estimates_stay_inside_budget_for_every_gpu_and_aspect(self):
        for total in GPU_SIZES_GB:
            for wh in INPUTS:
                for res in (720, 1080):
                    tp = self._plan(total, wh, res)
                    if tp.fits:
                        self.assertLessEqual(tp.estimate_gb, tp.budget.limit_gb + 1e-6, (total, wh, res))
                        self.assertEqual((tp.updates["batch_size"] - 1) % 4, 0)

    def test_more_vram_never_lowers_quality(self):
        last = 0
        for total in GPU_SIZES_GB:
            tp = self._plan(total, (640, 360), 1080, frames=301)
            if tp.fits:
                self.assertGreaterEqual(tp.updates["batch_size"], last)
                last = tp.updates["batch_size"]
        self.assertGreaterEqual(last, 101)

    def test_bigger_save_vram_never_raises_batch(self):
        big = self._plan(32.0, (640, 360), 1080, frames=301, save=1.5)
        small = self._plan(32.0, (640, 360), 1080, frames=301, save=10.0)
        self.assertGreaterEqual(big.updates["batch_size"], small.updates["batch_size"])

    def test_whole_clip_in_one_batch_when_it_fits(self):
        tp = self._plan(80.0, (640, 360), 720, frames=45)
        self.assertEqual(tp.updates["batch_size"], 45)
        self.assertEqual(tp.updates["blocks_to_swap"], 0)

    def test_matches_calibration_points(self):
        # Allocated GiB measured on an RTX 5090 (benchmarks/vram/seedvr2.jsonl); estimates must not undershoot by > 0.2.
        geo = seed_plan.Geometry(1920, 1088, 1920, 1088)
        self.assertGreaterEqual(seed_plan.dit_peak_gb(self.MODEL, True, geo, 21, 0) + 0.2, 17.10)
        self.assertGreaterEqual(seed_plan.decode_peak_gb(geo, 21, True, 1024) + 0.2, 18.05)
        self.assertGreaterEqual(seed_plan.encode_peak_gb(geo, 21, True, 1024) + 0.2, 11.26)
        geo720 = seed_plan.Geometry(1280, 720, 1280, 720)
        self.assertGreaterEqual(seed_plan.dit_peak_gb(self.MODEL, True, geo720, 85, 0) + 0.2, 22.20)
        self.assertGreaterEqual(seed_plan.decode_peak_gb(geo720, 45, False, 0) + 0.2, 16.39)

    def test_impossible_budget_reports_instead_of_crashing(self):
        tp = self._plan(4.0, (1920, 1080), 2160, frames=121, save=2.0)
        self.assertFalse(tp.fits)
        self.assertTrue(tp.warnings)

    def test_image_keeps_batch(self):
        settings = {"dit_model": self.MODEL, "int8_convrot": True, "resolution": 2160, "max_resolution": 0}
        tp = seed_plan.plan(settings, in_w=1024, in_h=1024, frames=1, budget=budget(24.0), image=True)
        self.assertNotIn("batch_size", tp.updates)


class SeedVR2EvenBatchTests(unittest.TestCase):
    def test_batch_count_matches_engine_stepping(self):
        self.assertEqual(seed_plan.batch_count(72, 65), 2)
        self.assertEqual(seed_plan.batch_count(72, 73), 1)
        self.assertEqual(seed_plan.batch_count(100, 45, 3), 3)

    def test_last_batch_is_never_left_short(self):
        for frames in range(5, 400, 7):
            for batch in range(5, 130, 4):
                for overlap in (0, 3):
                    even = seed_plan.balanced_batch(frames, batch, overlap)
                    count = seed_plan.batch_count(frames, batch, overlap)
                    self.assertLessEqual(even, batch)
                    self.assertEqual((even - 1) % 4, 0)
                    self.assertEqual(seed_plan.batch_count(frames, even, overlap), count, (frames, batch, overlap))
        self.assertEqual(seed_plan.balanced_batch(72, 65), 37)

    def test_plan_splits_evenly(self):
        gpu = fake_gpu(32.0, used_gb=0.4)
        budget = common.resolve_budget("0", 1.5, gpus={0: gpu})
        plan = seed_plan.plan({"dit_model": "seedvr2_ema_7b_sharp_fp16.safetensors", "int8_convrot": True,
                               "resolution": 1072, "max_resolution": 1920}, in_w=640, in_h=360, frames=72,
                              budget=budget)
        batch = plan.updates["batch_size"]
        self.assertEqual(seed_plan.balanced_batch(72, batch), batch, plan.report())


class FlashVSRPlannerTests(unittest.TestCase):
    def test_every_gpu_and_aspect(self):
        for total in GPU_SIZES_GB:
            for wh in INPUTS:
                tp = flash_plan.plan({"scale": 4, "mode": "full"}, in_w=wh[0] // 2, in_h=wh[1] // 2, frames=240,
                                     budget=budget(total, overhead=flash_plan.FLASHVSR_PROCESS_OVERHEAD_GB))
                if tp.fits:
                    self.assertLessEqual(tp.estimate_gb, tp.budget.limit_gb + 1e-6, (total, wh))
                    self.assertFalse(tp.updates["keep_models_on_cpu"])

    def test_pass_never_exceeds_rope_table(self):
        # A 4,200-frame pass overran the DiT's 1024-position temporal RoPE table and crashed.
        for mode in ("full", "tiny-long"):
            tp = flash_plan.plan({"scale": 4, "mode": mode}, in_w=160, in_h=90, frames=20000, budget=budget(48.0))
            self.assertTrue(0 < tp.updates["frame_chunk_size"] <= flash_plan.MAX_PASS_FRAMES, (mode, tp.updates))

    def test_untiled_when_it_fits(self):
        tp = flash_plan.plan({"scale": 4, "mode": "full"}, in_w=320, in_h=180, frames=120, budget=budget(32.0))
        self.assertFalse(tp.updates["tiled_dit"])
        self.assertEqual(tp.updates["frame_chunk_size"], 0)

    def test_portrait_lightvae_keeps_reserved_headroom(self):
        # 320x480 x4 (1280x1920), 120 frames, INT8 + LightVAE: planned untiled at a 28.6 GB cap it ran out
        # of VRAM (reserved runs ~1.26x allocated); it passed at 29.0-30.0 GB caps.
        settings = {"scale": 4, "mode": "full", "precision": "int8_convrot", "vae_model": "LightVAE_W2.1"}
        for limit_gb, tiled in ((29.27, True), (30.7, False)):
            gpu = common.GpuMemory(index=0, name="Test GPU", uuid="GPU-test-0", total_gb=31.8, reserved_gb=0.0,
                                   used_gb=31.8 - limit_gb - 1.5, free_gb=limit_gb + 1.5)
            b = common.VramBudget(gpu=gpu, save_gb=1.5, overhead_gb=flash_plan.FLASHVSR_PROCESS_OVERHEAD_GB)
            tp = flash_plan.plan(settings, in_w=320, in_h=480, frames=120, budget=b)
            self.assertEqual(tp.updates["tiled_dit"], tiled, (limit_gb, tp.updates))
            self.assertEqual(tp.updates["frame_chunk_size"], 0)
            self.assertLessEqual(tp.estimate_gb, limit_gb)
            if tiled:
                # 2 tiles either way; 320 px instead of 448 px halved the time (75 s vs 146 s).
                self.assertEqual(tp.updates["tile_size"], 320)
                self.assertEqual(flash_plan.tile_grid(320, tp.updates["overlap"], 320, 480), (2, 1))

    def test_smallest_tile_keeps_the_grid(self):
        for total in GPU_SIZES_GB:
            for wh in INPUTS:
                w, h = wh[0] // 2, wh[1] // 2
                choice, _ = flash_plan.choose(w, h, 4, 120, total - 3.0, 3.0, "full", "int8_convrot")
                if choice is None or not choice.tiled:
                    continue
                rows, cols = flash_plan.tile_grid(choice.tile, choice.overlap, w, h)
                for smaller in (t for t in flash_plan.TILE_CHOICES if t < choice.tile):
                    ov = flash_plan.tile_overlap(smaller)
                    if ov >= choice.overlap:
                        r, c = flash_plan.tile_grid(smaller, ov, w, h)
                        self.assertTrue(r > rows or c > cols, (total, wh, choice.tile, smaller))

    def test_small_gpu_tiles(self):
        tp = flash_plan.plan({"scale": 4, "mode": "full"}, in_w=480, in_h=270, frames=120, budget=budget(12.0))
        self.assertTrue(tp.fits)
        self.assertTrue(tp.updates["tiled_dit"])

    def test_decoder_and_int8_calibration_points(self):
        # Allocated peaks, 480x270 x4 (1080p), 81 frames untiled (benchmarks: fdec v2_* / v_tiny_tc_1080p).
        mp = flash_plan.out_mp(480, 270, 4)
        for mode, precision, measured in (("tiny", "bf16", 21.25), ("tiny", "int8_convrot", 20.26),
                                          ("full", "bf16", 22.22), ("full", "int8_convrot", 21.23)):
            est = flash_plan.peak_gb(mp, 81, 3, mode, precision, "LightVAE_W2.1")
            self.assertGreaterEqual(est + 0.2, measured, (mode, precision, est))
            self.assertLess(est, measured + 1.5, (mode, precision, est))

    def test_full_mode_decoder_tiling(self):
        light = flash_plan.plan({"scale": 4, "mode": "full", "vae_model": "LightVAE_W2.1"}, in_w=320, in_h=180,
                                frames=81, budget=budget(32.0))
        wan = flash_plan.plan({"scale": 4, "mode": "full", "vae_model": "Wan2.1"}, in_w=320, in_h=180,
                              frames=81, budget=budget(32.0))
        tiny = flash_plan.plan({"scale": 4, "mode": "tiny"}, in_w=320, in_h=180, frames=81, budget=budget(32.0))
        self.assertFalse(light.updates["tiled_vae"])
        self.assertTrue(wan.updates["tiled_vae"])  # the Wan2.1 VAE needs ~8 GB per megapixel untiled
        self.assertNotIn("tiled_vae", tiny.updates)

    def test_calibration_points(self):
        self.assertGreaterEqual(flash_plan.peak_gb(flash_plan.out_mp(480, 270, 4), 45, 3, "full") + 0.2, 21.00)
        self.assertGreaterEqual(flash_plan.peak_gb(flash_plan.tile_mp(256, 480, 270, 4), 45, 3, "full") + 0.2, 11.75)
        self.assertGreaterEqual(flash_plan.peak_gb(flash_plan.tile_mp(128, 480, 270, 4), 45, 3, "full") + 0.2, 5.49)


class SparkVSRPlannerTests(unittest.TestCase):
    def test_every_gpu_and_aspect(self):
        for total in GPU_SIZES_GB:
            for wh in INPUTS:
                tp = spark_plan.plan({"model_name": "SparkVSR-int8-convrot"}, out_w=wh[0] * 2, out_h=wh[1] * 2,
                                     frames=200, budget=budget(total, overhead=spark_plan.SPARKVSR_PROCESS_OVERHEAD_GB))
                if tp.fits:
                    self.assertLessEqual(tp.estimate_gb, tp.budget.limit_gb + 1e-6, (total, wh))
                    chunk = tp.updates["chunk_len"]
                    self.assertTrue(chunk == 0 or (chunk - 1) % 8 == 0, chunk)

    def test_short_clip_is_one_chunk(self):
        tp = spark_plan.plan({"model_name": "SparkVSR-bf16"}, out_w=1280, out_h=720, frames=30, budget=budget(32.0))
        self.assertEqual(tp.updates["chunk_len"], 0)

    def test_longer_clips_use_best_quality_chunk(self):
        # 33-frame chunks measured best (two clips); longer chunks only bought speed.
        tp = spark_plan.plan({"model_name": "SparkVSR-int8-convrot"}, out_w=1920, out_h=1080, frames=121,
                             budget=budget(32.0, used_gb=0.7, overhead=spark_plan.SPARKVSR_PROCESS_OVERHEAD_GB))
        self.assertEqual(tp.updates["chunk_len"], spark_plan.MAX_CHUNK)
        self.assertEqual(tp.updates["tile_height"], 0)  # the shorter chunk fits untiled
        self.assertEqual(tp.updates["tile_height"], 0)


class RTXPlannerTests(unittest.TestCase):
    def test_family_is_kept(self):
        self.assertEqual(rtx_plan.ladder("HIGHBITRATE_LOW")[0], "HIGHBITRATE_ULTRA")
        self.assertEqual(rtx_plan.ladder("DENOISE_HIGH")[0], "DENOISE_ULTRA")
        self.assertEqual(rtx_plan.ladder("BICUBIC")[0], "ULTRA")

    def test_picks_highest_level_that_fits(self):
        q, _ = rtx_plan.choose("HIGHBITRATE_MEDIUM", 1920, 1080, 3840, 2160, budget(24.0))
        self.assertEqual(q, "HIGHBITRATE_ULTRA")
        q, need = rtx_plan.choose("ULTRA", 1920, 1080, 7680, 4320, budget(2.4, save_gb=0.0, used_gb=0.0))
        self.assertTrue(q is None or need <= 2.4)


class LTX25BudgetTests(unittest.TestCase):
    @staticmethod
    def planner():
        import importlib

        ltx_dir = str(ROOT / "LTX25")
        if ltx_dir not in sys.path:
            sys.path.insert(0, ltx_dir)
        return importlib.import_module("ltx25_engine.planner")

    def test_budget_fits_measured_runs(self):
        p = self.planner()
        gib = 1024 ** 3
        # 28.8 GB free: 720p input x 121 frames ran in one chunk (58,880 tokens, 25.2 GB peak); 1080p input
        # (8,160 tokens per latent frame) planned at 73,554 tokens hit the cap, so it gets less now.
        budget, reason = p.auto_token_budget(121, 1280, 720, int(28.8 * gib), 0, 1.5)
        self.assertEqual(budget, 58880, reason)
        budget, reason = p.auto_token_budget(121, 1920, 1080, int(28.8 * gib), 0, 1.5)
        self.assertLess(budget, 73554 * 0.9, reason)
        # Every chunk keeps the non-block weights + fixed part + per-token activations inside the free VRAM.
        for free_gb in (9.5, 14.0, 22.0, 28.8, 46.0, 78.0):
            for w, h in INPUTS:
                budget, _ = p.auto_token_budget(361, w, h, int(free_gb * gib), 0, 1.5)
                need = p.NON_BLOCK_WEIGHT_BYTES + p.ACTIVATION_BASE_BYTES + budget * 1.25 * p.BYTES_PER_TOKEN * p.ACTIVATION_SAFETY
                if budget > 4096:
                    self.assertLessEqual(need, (free_gb - 1.5) * gib, (free_gb, w, h, budget))

    def test_chunks_stay_inside_20s_positions(self):
        from shared.vram_autotune import ltx25 as ltx_plan

        for fps in (12, 23.976, 24, 25, 30, 50, 60, 120):
            frames = ltx_plan.max_chunk_frames(fps)
            self.assertEqual((frames - 1) % 8, 0, fps)
            self.assertLessEqual((frames - 1) / fps, 20.0 + 1e-9, fps)
            self.assertLessEqual(frames, ltx_plan.MAX_CHUNK_FRAMES)
        self.assertEqual(ltx_plan.max_chunk_frames(24), 481)

    def test_more_vram_never_shrinks_chunks(self):
        p = self.planner()
        gib = 1024 ** 3
        for w, h in INPUTS:
            budgets = [p.auto_token_budget(361, w, h, int(g * gib), 0, 1.5)[0] for g in (14.0, 22.0, 28.8, 46.0)]
            self.assertEqual(budgets, sorted(budgets), (w, h))


class PresetWiringTests(unittest.TestCase):
    def test_every_upscaler_has_save_vram_and_auto_apply(self):
        from shared.universal_preset import TAB_CONFIGS, dict_to_values

        keys = {"seedvr2": "save_vram_gb", "flashvsr": "save_vram_gb", "sparkvsr": "save_vram_gb",
                "rtx": "save_vram_gb", "ltx25": "reserve_vram_gb", "open_h3": "reserve_vram_gb"}
        for tab, save_key in keys.items():
            cfg = TAB_CONFIGS[tab]
            order = cfg["order"]
            defaults_fn = cfg.get("defaults_fn") or cfg.get("defaults")
            defaults = defaults_fn() if callable(defaults_fn) else dict(defaults_fn)
            self.assertIn(save_key, order, tab)
            self.assertIn("auto_tune_on_run", order, tab)
            self.assertEqual(float(defaults[save_key]), 1.5, tab)
            self.assertIs(defaults["auto_tune_on_run"], True, tab)
            self.assertTrue(set(order) <= set(defaults), (tab, set(order) - set(defaults)))
            values = dict_to_values(tab, {}, defaults)
            self.assertNotIn(None, [v for k, v in zip(order, values) if k not in {"cuda_device", "device"}], tab)
            # A saved preset round-trips the two new keys.
            round_trip = dict(zip(order, dict_to_values(tab, {save_key: 7.5, "auto_tune_on_run": False}, defaults)))
            self.assertEqual(float(round_trip[save_key]), 7.5, tab)
            self.assertIs(round_trip["auto_tune_on_run"], False, tab)


if __name__ == "__main__":
    unittest.main()
