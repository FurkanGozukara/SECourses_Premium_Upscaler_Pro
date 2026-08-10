from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from shared.flashvsr_runner import (
    _flashvsr_cli_supports_option,
    _resolve_flashvsr_cli_precision,
)
from shared.services.sparkvsr_service import (
    _enforce_sparkvsr_guardrails,
    sparkvsr_defaults,
)
from shared.preset_manager import PresetManager
from tools.sparkvsr_inference import (
    _cogvideox_untiled_decode_needs_spatial_tiling,
    _vae_decode_to_cpu_low_vram,
    make_spatial_tiles,
    pad_video_chunk_to_vae_grid,
    pad_video_spatial_to_multiple,
)
from tools.rife_inference_wrapper import install_numpy_binary_fromstring_compat
from ui import shared_components


class AutoTuneModalTests(unittest.TestCase):
    def test_dismiss_and_reset_scripts_target_the_specific_modal(self):
        dismiss = shared_components.autotune_modal_dismiss_js("seed-modal")
        reset = shared_components.autotune_modal_reset_js("seed-modal")
        self.assertIn("seed-modal", dismiss)
        self.assertIn('dataset.dismissed = "true"', dismiss)
        self.assertIn("seed-modal", reset)
        self.assertIn("delete modal.dataset.dismissed", reset)

    def test_cancel_warning_is_explicit_and_raises_a_toast(self):
        with mock.patch.object(shared_components.gr, "Warning") as warning:
            message = shared_components.warn_cancel_confirmation()
        self.assertIn("Confirm cancel", message)
        warning.assert_called_once()


class FlashVSRCompatibilityTests(unittest.TestCase):
    def test_legacy_cli_falls_back_without_unsupported_int8_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            cli = Path(tmp) / "cli_main.py"
            cli.write_text("parser.add_argument('--precision', choices=['fp16', 'bf16', 'auto'])", encoding="utf-8")
            precision, note = _resolve_flashvsr_cli_precision("int8_convrot", cli)
            self.assertEqual(precision, "bf16")
            self.assertIsNotNone(note)
            self.assertFalse(_flashvsr_cli_supports_option(cli, "--int8_cache_dir"))

    def test_int8_capable_cli_keeps_requested_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            cli = Path(tmp) / "cli_main.py"
            cli.write_text(
                "parser.add_argument('--precision', choices=['int8_convrot']); "
                "parser.add_argument('--int8_cache_dir')",
                encoding="utf-8",
            )
            precision, note = _resolve_flashvsr_cli_precision("int8_convrot", cli)
            self.assertEqual(precision, "int8_convrot")
            self.assertIsNone(note)


class SparkVSRSpatialRegressionTests(unittest.TestCase):
    def test_edge_tiles_are_full_sized(self):
        tiles = make_spatial_tiles(960, 1320, (256, 256), (32, 32))
        self.assertTrue(tiles)
        self.assertTrue(all((h1 - h0) == 256 for h0, h1, _, _ in tiles))
        self.assertTrue(all((w1 - w0) == 256 for _, _, w0, w1 in tiles))
        self.assertEqual(max(h1 for _, h1, _, _ in tiles), 960)
        self.assertEqual(max(w1 for _, _, _, w1 in tiles), 1320)

    def test_spatial_padding_uses_transformer_grid_and_replicates_edges(self):
        source = torch.arange(2 * 3 * 7 * 13, dtype=torch.float32).reshape(2, 3, 7, 13)
        padded, pad_h, pad_w = pad_video_spatial_to_multiple(source)
        self.assertEqual((pad_h, pad_w), (9, 3))
        self.assertEqual(tuple(padded.shape[-2:]), (16, 16))
        torch.testing.assert_close(padded[..., :7, :13], source)
        torch.testing.assert_close(padded[..., -1, -1], source[..., -1, -1])

    def test_reference_prepass_is_opt_in_and_tile_values_are_grid_aligned(self):
        defaults = sparkvsr_defaults()
        self.assertFalse(defaults["auto_reference_prepass"])
        guarded = _enforce_sparkvsr_guardrails(
            {
                **defaults,
                "tile_height": 255,
                "tile_width": 257,
                "overlap_height": 31,
                "overlap_width": 33,
            },
            defaults,
        )
        self.assertEqual(guarded["tile_height"] % 16, 0)
        self.assertEqual(guarded["tile_width"] % 16, 0)
        self.assertEqual(guarded["overlap_height"] % 16, 0)
        self.assertEqual(guarded["overlap_width"] % 16, 0)


class SparkVSRTemporalDecodeRegressionTests(unittest.TestCase):
    def test_65_frame_chunks_stay_on_the_cogvideox_vae_grid(self):
        source = torch.arange(65, dtype=torch.float32).reshape(1, 1, 65, 1, 1)
        padded, pad_t = pad_video_chunk_to_vae_grid(source)
        self.assertEqual(pad_t, 0)
        self.assertIs(padded, source)

        tail = source[:, :, :58]
        padded_tail, pad_t = pad_video_chunk_to_vae_grid(tail)
        self.assertEqual(pad_t, 7)
        self.assertEqual(tuple(padded_tail.shape), (1, 1, 65, 1, 1))
        torch.testing.assert_close(padded_tail[:, :, -1], tail[:, :, -1])

    def test_real_diffusers_shape_probe_detects_the_1080p_safeconv_failure(self):
        class Config:
            block_out_channels = (128, 256, 256, 512)

        class Vae:
            config = Config()
            num_latent_frames_batch_size = 2

        full_hd_latents = torch.empty((1, 16, 17, 136, 240), device="meta")
        small_latents = torch.empty((1, 16, 17, 60, 80), device="meta")
        self.assertTrue(_cogvideox_untiled_decode_needs_spatial_tiling(Vae(), full_hd_latents))
        self.assertFalse(_cogvideox_untiled_decode_needs_spatial_tiling(Vae(), small_latents))

    def test_split_decode_retries_the_exact_temporal_kernel_error_with_tiles(self):
        class DecodeResult:
            def __init__(self, sample):
                self.sample = sample

        class FakeVae:
            use_tiling = False
            tile_latent_min_height = 2
            tile_latent_min_width = 2
            tile_sample_min_height = 16
            tile_sample_min_width = 16
            tile_overlap_factor_height = 0.5
            tile_overlap_factor_width = 0.5

            def __init__(self):
                self.decode_shapes = []

            def decode(self, value):
                self.decode_shapes.append(tuple(value.shape))
                if value.shape[-2:] == (4, 4):
                    raise RuntimeError(
                        "Calculated padded input size per channel: (2 x 34 x 34). "
                        "Kernel size: (3 x 3 x 3). Kernel size can't be greater than actual input size"
                    )
                sample = torch.ones(
                    (value.shape[0], 3, value.shape[2], value.shape[3] * 8, value.shape[4] * 8),
                    dtype=value.dtype,
                )
                return DecodeResult(sample)

            @staticmethod
            def blend_v(_above, current, _extent):
                return current

            @staticmethod
            def blend_h(_left, current, _extent):
                return current

        class Pipe:
            def __init__(self):
                self.vae = FakeVae()

        pipe = Pipe()
        decoded = _vae_decode_to_cpu_low_vram(pipe, torch.zeros((1, 16, 2, 4, 4)))
        self.assertEqual(pipe.vae.decode_shapes[0][-2:], (4, 4))
        self.assertTrue(all(shape[-2] <= 2 and shape[-1] <= 2 for shape in pipe.vae.decode_shapes[1:]))
        self.assertEqual(tuple(decoded.shape), (1, 3, 2, 32, 32))
        self.assertFalse(pipe.vae.use_tiling)


class RIFECompatibilityTests(unittest.TestCase):
    def test_binary_fromstring_uses_frombuffer_on_current_numpy(self):
        class FakeNumpy:
            frombuffer_calls = []

            @staticmethod
            def fromstring(value, **kwargs):
                return ("text", value, kwargs)

            @classmethod
            def frombuffer(cls, value, dtype=float, count=-1):
                cls.frombuffer_calls.append((value, dtype, count))
                return "binary"

        fake = FakeNumpy()
        install_numpy_binary_fromstring_compat(fake)
        first_patch = fake.fromstring
        install_numpy_binary_fromstring_compat(fake)
        self.assertIs(fake.fromstring, first_patch)
        self.assertEqual(fake.fromstring(b"abc", dtype="u1", count=3), "binary")
        self.assertEqual(fake.frombuffer_calls, [(b"abc", "u1", 3)])
        self.assertEqual(fake.fromstring("1 2", dtype=int, sep=" ")[0], "text")


class UniversalPresetRegressionTests(unittest.TestCase):
    def test_deleting_last_used_preset_clears_stale_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = PresetManager(Path(tmp))
            manager.save_universal_preset("temporary", {"global": {}})
            self.assertEqual(manager.get_last_used_universal_preset(), "temporary")
            self.assertTrue(manager.delete_universal_preset("temporary"))
            self.assertIsNone(manager.get_last_used_universal_preset())


if __name__ == "__main__":
    unittest.main()
