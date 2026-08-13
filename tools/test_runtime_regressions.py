from __future__ import annotations

import sys
import tempfile
import types
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
from shared.rtx_superres_runner import _build_dimensions_plan
from shared.sparkvsr_ref_utils import resolve_pisa_runtime
from shared.ui_validators import SEEDVR2_MAX_BATCH_SIZE, validate_batch_size_seedvr2
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


class PiSAIntegrationRegressionTests(unittest.TestCase):
    def test_standard_install_is_auto_resolved_before_sparkvsr_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pisa = base / "PiSA-SR"
            script = pisa / "test_pisasr.py"
            sd_model = pisa / "preset" / "models" / "stable-diffusion-2-1-base"
            checkpoint = pisa / "preset" / "models" / "pisa_sr.pkl"
            python_exe = pisa / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            for directory in (script.parent, sd_model, checkpoint.parent, python_exe.parent):
                directory.mkdir(parents=True, exist_ok=True)
            script.write_text("print('pisa')", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            python_exe.write_bytes(b"python")

            resolved, error, notes = resolve_pisa_runtime({"ref_mode": "pisasr"}, base)

            self.assertIsNone(error)
            self.assertEqual(Path(resolved["pisa_script_path"]), script.resolve())
            self.assertEqual(Path(resolved["pisa_sd_model_path"]), sd_model.resolve())
            self.assertEqual(Path(resolved["pisa_chkpt_path"]), checkpoint.resolve())
            self.assertEqual(Path(resolved["pisa_python_executable"]), python_exe.resolve())
            self.assertEqual(len(notes), 4)

    def test_missing_install_fails_preflight_with_actionable_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            _resolved, error, _notes = resolve_pisa_runtime({"ref_mode": "pisasr"}, Path(tmp))
        self.assertIsNotNone(error)
        self.assertIn("test_pisasr.py", error)
        self.assertIn("stable-diffusion-2-1-base", error)
        self.assertIn("pisa_sr.pkl", error)
        self.assertIn("SEC_PISA_SCRIPT_PATH", error)


class SeedVR2BatchLimitRegressionTests(unittest.TestCase):
    def test_601_is_the_supported_seedvr2_maximum(self):
        self.assertEqual(SEEDVR2_MAX_BATCH_SIZE, 601)
        self.assertEqual(validate_batch_size_seedvr2(601), (True, None, 601))
        valid, message, corrected = validate_batch_size_seedvr2(605)
        self.assertFalse(valid)
        self.assertIn("max 601", str(message))
        self.assertEqual(corrected, 601)

        from shared.services.seedvr2_service import _enforce_seedvr2_guardrails, seedvr2_defaults

        seed_defaults = seedvr2_defaults()
        guarded = _enforce_seedvr2_guardrails(
            {**seed_defaults, "batch_size": 1001},
            seed_defaults,
            silent_migration=True,
        )
        self.assertEqual(guarded["batch_size"], 601)


class RTXSuperResolutionRegressionTests(unittest.TestCase):
    def test_denoise_and_deblur_modes_are_always_same_resolution(self):
        for quality in ("DENOISE_ULTRA", "DEBLUR_HIGH"):
            plan = _build_dimensions_plan(
                input_width=1920,
                input_height=1080,
                upscale_factor=4.0,
                max_edge=8192,
                pre_downscale_then_upscale=True,
                quality_name=quality,
            )
            self.assertEqual((plan["output_width"], plan["output_height"]), (1920, 1080))

    def test_large_requested_scale_keeps_native_vfx_pass_at_or_below_4x(self):
        plan = _build_dimensions_plan(
            input_width=640,
            input_height=360,
            upscale_factor=9.9,
            max_edge=0,
            pre_downscale_then_upscale=True,
            quality_name="HIGHBITRATE_ULTRA",
        )
        self.assertTrue(plan["post_resize_required"])
        self.assertLessEqual(plan["model_scale"], 4.0)
        self.assertEqual((plan["output_width"], plan["output_height"]), (6336, 3564))

    def test_linux_health_check_uses_installed_nvvfx_instead_of_skipping(self):
        from shared import health

        fake_module = types.ModuleType("nvvfx")

        class FakeVideoSuperRes:
            QualityLevel = object()

        fake_module.VideoSuperRes = FakeVideoSuperRes
        with mock.patch.dict(sys.modules, {"nvvfx": fake_module}), mock.patch.object(
            health.platform, "system", return_value="Linux"
        ):
            result = health._check_nvidia_vfx()
        self.assertEqual(result["status"], "ok")
        self.assertIn("Linux", str(result["detail"]))

    def test_installer_requirements_include_nvidia_vfx(self):
        requirements = (ROOT.parent / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("nvidia-vfx==0.1.0.1", requirements)


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
