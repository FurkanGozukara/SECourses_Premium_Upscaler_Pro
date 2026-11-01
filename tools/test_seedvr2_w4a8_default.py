"""CPU tests: W4A8 7B Sharp is the default SeedVR2 model, with an automatic INT8 ConvRot fallback.

Run with the GPU hidden:  set CUDA_VISIBLE_DEVICES=-1 && venv\\Scripts\\python.exe -m unittest tools.test_seedvr2_w4a8_default
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "SeedVR2"), str(ROOT.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from shared.models import seedvr2_w4a8 as w4  # noqa: E402

W4A8 = "seedvr2_7b_sharp_convrot_w4a8.safetensors"
INT8_SOURCE = "seedvr2_ema_7b_sharp_fp16.safetensors"


def _gpu(idx: int, cap):
    return SimpleNamespace(id=idx, name=f"GPU{idx}", total_memory_gb=24.0, available_memory_gb=20.0,
                           compute_capability=cap, is_available=True)


class _Env:
    """Patch the GPU list and the comfy_kitchen check; clear the forcing switch."""

    def __init__(self, caps, backend_error=None):
        self.caps = caps
        self.backend_error = backend_error

    def __enter__(self):
        gpus = [_gpu(i, c) for i, c in enumerate(self.caps)]
        self.patches = [
            mock.patch("shared.gpu_utils.get_gpu_info", return_value=gpus),
            mock.patch.object(w4, "comfy_kitchen_backend_error", return_value=self.backend_error),
            mock.patch.dict(os.environ, {}, clear=False),
        ]
        for p in self.patches:
            p.start()
        os.environ.pop(w4.ENV_FORCE_FALLBACK, None)
        return self

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.stop()


class DefaultModelTests(unittest.TestCase):
    def test_w4a8_is_the_default_and_int8_stays_selectable(self):
        from shared.models.seedvr2_meta import get_seedvr2_model_names
        from shared.services.seedvr2_service import seedvr2_defaults

        names = get_seedvr2_model_names()
        self.assertEqual(names[0], W4A8)
        self.assertIn(INT8_SOURCE, names)
        self.assertEqual(seedvr2_defaults()["dit_model"], W4A8)

    def test_saved_int8_preset_keeps_int8(self):
        from shared.services.seedvr2_service import _enforce_seedvr2_guardrails, seedvr2_defaults

        defaults = seedvr2_defaults()
        cfg = dict(defaults, dit_model=INT8_SOURCE, int8_convrot=True)
        out = _enforce_seedvr2_guardrails(cfg, defaults, state=None, silent_migration=True)
        self.assertEqual(out["dit_model"], INT8_SOURCE)
        self.assertTrue(out["int8_convrot"])


class FallbackTests(unittest.TestCase):
    def test_ampere_and_newer_run_w4a8(self):
        with _Env([(12, 0), (8, 6)]):
            self.assertIsNone(w4.w4a8_fallback_reason("0"))
            self.assertIsNone(w4.w4a8_fallback_reason("1"))
            self.assertEqual(w4.effective_model(W4A8, True, "1"), (W4A8, False, None))

    def test_old_gpu_falls_back_to_int8(self):
        with _Env([(12, 0), (7, 5)]):
            reason = w4.w4a8_fallback_reason("1")
            self.assertIn("7.5", reason)
            self.assertEqual(w4.effective_model(W4A8, False, "1")[:2], (INT8_SOURCE, True))
            self.assertIsNotNone(w4.w4a8_fallback_reason("0,1"))  # any listed GPU too old
            self.assertIsNone(w4.w4a8_fallback_reason("0"))

    def test_backend_failure_falls_back(self):
        with _Env([(12, 0)], backend_error="comfy_kitchen CUDA backend: DLL load failed"):
            self.assertIn("DLL load failed", w4.w4a8_fallback_reason("0"))

    def test_cpu_and_switch_force_fallback(self):
        with _Env([(12, 0)]):
            self.assertIsNotNone(w4.w4a8_fallback_reason("cpu"))
            os.environ[w4.ENV_FORCE_FALLBACK] = "1"
            self.assertIn(w4.ENV_FORCE_FALLBACK, w4.w4a8_fallback_reason("0"))

    def test_other_models_never_change(self):
        with _Env([(7, 5)]):
            self.assertEqual(w4.effective_model(INT8_SOURCE, True, "0"), (INT8_SOURCE, True, None))
            self.assertEqual(w4.effective_model("seedvr2_ema_3b_fp16.safetensors", False, "0")[2], None)

    def test_runner_switch_and_log_line(self):
        with _Env([(7, 5)]):
            settings = {"dit_model": W4A8, "int8_convrot": False}
            msg = w4.apply_w4a8_fallback(settings, "0")
            self.assertEqual(settings["dit_model"], INT8_SOURCE)
            self.assertTrue(settings["int8_convrot"])
            self.assertIn("INT8 ConvRot", msg)
            self.assertIn("7.5", msg)

    def test_real_backend_check_loads(self):
        w4._BACKEND_ERROR.clear()
        self.assertIsNone(w4.comfy_kitchen_backend_error())

    def test_engine_refuses_w4a8_without_cuda(self):
        import torch
        from src.optimization.w4a8_runtime import w4a8_unavailable_reason

        self.assertIsNotNone(w4a8_unavailable_reason(torch.device("cpu")))


class AutoTuneFallbackTests(unittest.TestCase):
    """When the fallback triggers, Auto Tune plans with the INT8 ConvRot numbers, not W4A8's."""

    def _plan(self, model, int8, caps):
        from shared.vram_autotune import seedvr2 as sp
        from shared.vram_autotune.common import GpuMemory, VramBudget

        budget = VramBudget(gpu=GpuMemory(index=0, name="t", uuid="u", total_gb=24.0, reserved_gb=0.0,
                                          used_gb=7.0, free_gb=17.0), save_gb=1.5, overhead_gb=1.2)
        settings = {"dit_model": model, "int8_convrot": int8, "resolution": 1080, "max_resolution": 1920}
        with _Env(caps):
            return sp.plan(settings, in_w=480, in_h=270, frames=49, budget=budget)

    def test_fallback_plan_equals_int8_plan(self):
        w4a8 = self._plan(W4A8, True, [(8, 6)])
        fallback = self._plan(W4A8, True, [(7, 5)])
        int8 = self._plan(INT8_SOURCE, True, [(7, 5)])
        self.assertEqual(fallback.updates, int8.updates)
        self.assertAlmostEqual(fallback.estimate_gb, int8.estimate_gb)
        dit = lambda tp: next(line for line in tp.lines if line.startswith("Phase peaks")).split("DiT ")[1].split(" GB")[0]
        self.assertEqual(dit(fallback), dit(int8))
        self.assertNotEqual(dit(w4a8), dit(int8))  # W4A8 weights (4.4 GB) vs INT8 (8.1 GB) at the same budget
        self.assertTrue(any("W4A8 cannot run here" in line for line in fallback.lines))


class DownloaderSourceTests(unittest.TestCase):
    """W4A8 downloads from our mirror first, then the original upload, only when size and sha256 match."""

    def setUp(self):
        import Models_Downloader as md
        self.md = md

    def _index(self, files_by_repo):
        def fake(repo_id):
            if repo_id not in files_by_repo:
                raise OSError("offline")
            return "c", files_by_repo[repo_id]
        return fake

    def _rf(self, repo, name, size=None, digest=None):
        md = self.md
        return md.RemoteFile(repo, name, size or md.SEEDVR2_W4A8_SIZE, "sha256",
                             digest or md.SEEDVR2_W4A8_SHA256, "c")

    def test_mirror_first(self):
        md = self.md
        files = {md.SEEDVR2_REPO_ID: {md.SEEDVR2_W4A8_MIRROR_NAME: self._rf(md.SEEDVR2_REPO_ID, md.SEEDVR2_W4A8_MIRROR_NAME)},
                 md.SEEDVR2_W4A8_REPO_ID: {md.SEEDVR2_W4A8_FILENAME: self._rf(md.SEEDVR2_W4A8_REPO_ID, md.SEEDVR2_W4A8_FILENAME)}}
        with mock.patch.object(md, "repo_index", self._index(files)):
            self.assertEqual(md.seedvr2_w4a8_source(), (md.SEEDVR2_REPO_ID, md.SEEDVR2_W4A8_MIRROR_NAME))

    def test_fallback_when_mirror_missing_or_different(self):
        md = self.md
        orig = {md.SEEDVR2_W4A8_FILENAME: self._rf(md.SEEDVR2_W4A8_REPO_ID, md.SEEDVR2_W4A8_FILENAME)}
        for mirror in ({}, {md.SEEDVR2_W4A8_MIRROR_NAME: self._rf(md.SEEDVR2_REPO_ID, md.SEEDVR2_W4A8_MIRROR_NAME, digest="0" * 64)}):
            with mock.patch.object(md, "repo_index", self._index({md.SEEDVR2_REPO_ID: mirror, md.SEEDVR2_W4A8_REPO_ID: orig})):
                self.assertEqual(md.seedvr2_w4a8_source(), (md.SEEDVR2_W4A8_REPO_ID, md.SEEDVR2_W4A8_FILENAME))

    def test_no_matching_source(self):
        md = self.md
        bad = {md.SEEDVR2_W4A8_FILENAME: self._rf(md.SEEDVR2_W4A8_REPO_ID, md.SEEDVR2_W4A8_FILENAME, size=1)}
        with mock.patch.object(md, "repo_index", self._index({md.SEEDVR2_W4A8_REPO_ID: bad})):
            self.assertIsNone(md.seedvr2_w4a8_source())

    def test_default_menu_entry_is_w4a8(self):
        md = self.md
        entry = next(e for e in md.MODEL_MENU if "default SeedVR2 model" in e["about"])
        self.assertIn("W4A8", entry["name"])


if __name__ == "__main__":
    unittest.main()
