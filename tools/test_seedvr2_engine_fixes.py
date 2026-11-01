"""CPU regression tests for the SeedVR2 engine fixes and memory helpers (upstream scan 2026-11-01).

Run with the GPU hidden:  set CUDA_VISIBLE_DEVICES=-1 && venv\\Scripts\\python.exe -m unittest tools.test_seedvr2_engine_fixes
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import types
import unittest
import weakref
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SEEDVR2 = ROOT / "SeedVR2"
for p in (str(ROOT), str(SEEDVR2)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch  # noqa: E402


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


class PrependFramesTests(unittest.TestCase):
    """Bug 1: prepended warm-up frames must leave the output once on one GPU, never twice."""

    def setUp(self):
        import inference_cli as cli
        self.cli = cli
        self.args = argparse.Namespace(split_phase_subprocesses=True, prepend_frames=4)
        self.debug = mock.MagicMock()

    def _run(self, strip: bool) -> torch.Tensor:
        out = torch.arange(65, dtype=torch.float32).view(65, 1, 1, 1)  # 61 frames + 4 prepended
        with mock.patch.object(self.cli, "_run_split_phase_pipeline", return_value=out):
            return self.cli._process_frames_core(out, self.args, "0", self.debug, strip_prepend=strip)

    def test_single_gpu_strips(self):
        res = self._run(True)
        self.assertEqual(res.shape[0], 61)
        self.assertEqual(float(res[0].item()), 4.0)

    def test_worker_keeps_for_main_process(self):
        self.assertEqual(self._run(False).shape[0], 65)

    def test_no_prepend_is_untouched(self):
        self.args.prepend_frames = 0
        self.assertEqual(self._run(True).shape[0], 65)


class CompiledModelCacheTests(unittest.TestCase):
    """Bug 2: a cached torch.compile'd DiT must be found without calling bool() on it."""

    def test_cached_compiled_dit_is_reused(self):
        from src.core import model_configuration as mc

        class NoLen(torch.nn.Module):
            def __len__(self):
                raise TypeError("CompatibleDiT does not support len()")

        compiled = torch.compile(NoLen())
        compiled._model_name = "m.safetensors"
        cache = mock.MagicMock()
        cache.get_dit.return_value = compiled
        cache.get_vae.return_value = None
        with mock.patch.object(mc, "get_global_cache", return_value=cache):
            ctx = mc._initialize_cache_context(True, True, 1, 2, "m.safetensors", "v.safetensors", mock.MagicMock())
        self.assertIs(ctx["cached_dit"], compiled)


class BlockSwapCycleTests(unittest.TestCase):
    """Bug 6: restoring a wrapped method must not leave a self-referencing bound method."""

    def test_restore_frees_without_gc(self):
        from src.optimization.blockswap import _restore_method

        gc.disable()
        try:
            m = torch.nn.Linear(8, 8)
            m._original_forward = m.forward
            m.forward = types.MethodType(lambda self, x: self._original_forward(x), m)
            _restore_method(m, "forward", m._original_forward)
            del m._original_forward
            ref = weakref.ref(m)
            del m
            self.assertIsNone(ref())
        finally:
            gc.enable()


class CompatibilityStubTests(unittest.TestCase):
    """Bug 7: absent packages are not stubbed (find_spec must stay honest)."""

    def test_absent_package_not_stubbed(self):
        import ast
        import importlib.machinery

        src = (SEEDVR2 / "src/optimization/compatibility.py").read_text(encoding="utf-8")
        fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "ensure_flash_attn_safe")
        ns = {"sys": sys, "types": types, "importlib": importlib}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "compat", "exec"), ns)
        with mock.patch.object(importlib.machinery.PathFinder, "find_spec", return_value=None), \
                mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("flash_attn", None)
            real_import = __import__

            def fake_import(name, *a, **k):
                if name == "flash_attn":
                    raise ImportError("absent")
                return real_import(name, *a, **k)

            with mock.patch("builtins.__import__", side_effect=fake_import):
                ns["ensure_flash_attn_safe"]()
            self.assertNotIn("flash_attn", sys.modules)


class SeedTests(unittest.TestCase):
    def test_64bit_and_negative_seeds(self):
        from src.common.seed import set_seed

        set_seed(42)
        a = torch.randn(3)
        for s in (2 ** 33, -1, 2 ** 64 + 5):
            set_seed(s)
        set_seed(42)
        self.assertTrue(torch.equal(a, torch.randn(3)))


class DitMemoryHelperTests(unittest.TestCase):
    """Row 4: chunked RoPE and preallocated varlen SDPA are bit-exact; MLP chunking runs."""

    def test_rope_chunked_matches_reference(self):
        from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
        from src.optimization import dit_memory as dm

        freqs = RotaryEmbedding(dim=21, freqs_for="pixel", max_freq=256).get_axial_freqs(3, 8, 10)
        freqs = freqs.reshape(-1, freqs.shape[-1]).to(torch.bfloat16)
        t = torch.randn(freqs.shape[0], 4, 64, dtype=torch.bfloat16)
        ref = apply_rotary_emb(freqs, t.permute(1, 0, 2).float()).to(t.dtype).permute(1, 0, 2)
        with torch.no_grad(), mock.patch.object(dm, "ROPE_CHUNK_TOKENS", 37):
            out = dm.rope_apply(apply_rotary_emb, freqs, t.clone())
        self.assertTrue(_bits_equal(out.contiguous(), ref.contiguous()))

    def test_varlen_sdpa_matches_list_and_cat(self):
        import torch.nn.functional as F
        from src.optimization import dit_memory as dm

        lens = [37, 37, 20, 5]
        q, k, v = (torch.randn(sum(lens), 4, 16) for _ in range(3))
        cu = torch.tensor([0, 37, 74, 94, 99], dtype=torch.int32)
        parts = []
        for s, e in zip(cu[:-1].tolist(), cu[1:].tolist()):
            o = F.scaled_dot_product_attention(q[s:e].permute(1, 0, 2)[None], k[s:e].permute(1, 0, 2)[None],
                                               v[s:e].permute(1, 0, 2)[None])
            parts.append(o[0].permute(1, 0, 2))
        ref = torch.cat(parts)
        for batched in (False, True):
            with torch.no_grad(), mock.patch.object(dm, "SDPA_BATCH_WINDOWS", batched):
                self.assertTrue(_bits_equal(dm.varlen_sdpa(q, k, v, cu, cu), ref), batched)

    def test_chunked_mlp_runs_and_matches_closely(self):
        from src.models.dit_7b import mlp as mlp_mod

        m = mlp_mod.MLP(32, 4)
        x = torch.randn(300, 32)
        with torch.no_grad():
            with mock.patch.object(mlp_mod, "MLP_CHUNK_TOKENS", 0):
                ref = m(x)
            with mock.patch.object(mlp_mod, "MLP_CHUNK_TOKENS", 64):
                out = m(x)
        self.assertTrue(torch.allclose(ref, out, atol=1e-5))


class W4A8LayerTests(unittest.TestCase):
    """Row 12: a W4A8 ConvRot layer from the shipped checkpoint matches its dequantized weight."""

    CKPT = SEEDVR2 / "models" / "seedvr2_7b_sharp_convrot_w4a8.safetensors"

    def test_layer_forward_matches_dequantized(self):
        if not self.CKPT.is_file():
            self.skipTest("W4A8 checkpoint not downloaded")
        from safetensors import safe_open
        from comfy_kitchen.tensor.w4a8_int8 import dequantize_w4a8_int8_weight
        from src.optimization.w4a8_runtime import is_w4a8_state_dict, prepare_model_for_w4a8

        prefix = "blocks.0.mlp.vid.proj_in."
        with safe_open(str(self.CKPT), "pt", device="cpu") as h:
            st = {k[len("blocks.0.mlp.vid."):]: h.get_tensor(k) for k in h.keys() if k.startswith(prefix)}
        self.assertTrue(is_w4a8_state_dict(st))
        m = torch.nn.Module()
        m.proj_in = torch.nn.Linear(3072, 12288)
        self.assertEqual(prepare_model_for_w4a8(m, st), 1)
        m.load_state_dict(st, strict=False, assign=True)
        x = torch.randn(32, 3072, dtype=torch.bfloat16)
        with torch.no_grad():
            y = m.proj_in(x).float()
            w = dequantize_w4a8_int8_weight(m.proj_in.weight, m.proj_in.weight_s_rel, m.proj_in.weight_s_channel,
                                            codebook=m.proj_in.weight_codebook, output_dtype=torch.float32)
            ref = torch.nn.functional.linear(x.float(), w, m.proj_in.bias.float())
        self.assertLess(float((y - ref).norm() / ref.norm()), 0.03)


class ModelConfigTests(unittest.TestCase):
    def test_config_folders(self):
        from src.core.model_configuration import config_directory_for_dit_model as cfg

        self.assertEqual(cfg("seedvr2_distill_6L_1.4B_sharp_fp16.safetensors"), "./configs_1_4b")
        self.assertEqual(cfg("seedvr2_ema_7b_sharp_fp16.safetensors"), "./configs_7b")
        self.assertEqual(cfg("seedvr2_7b_sharp_convrot_w4a8.safetensors"), "./configs_7b")
        self.assertEqual(cfg("seedvr2_ema_3b_fp16.safetensors"), "./configs_3b")
        self.assertTrue((SEEDVR2 / "configs_1_4b" / "main.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
