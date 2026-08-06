from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from shared import model_downloads


class ModelDownloadBridgeTests(unittest.TestCase):
    def test_windows_defaults_are_explicitly_flagged(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(model_downloads.windows_int8_defaults_enabled())
        with patch.dict(os.environ, {"SECOURSES_WINDOWS_INT8_DEFAULTS": "1"}, clear=True):
            self.assertTrue(model_downloads.windows_int8_defaults_enabled())

    def test_downloader_is_one_folder_up_and_streams_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_dir = root / "app"
            app_dir.mkdir()
            downloader = root / "Models_Downloader.py"
            downloader.touch()
            process = MagicMock()
            process.stdout = iter(["[DOWNLOADING] 25%\n", "[READY] selected model\n"])
            process.wait.return_value = 0
            progress = []
            with patch.object(model_downloads.subprocess, "Popen", return_value=process) as popen:
                ok, error = model_downloads.run_model_downloader(
                    app_dir, ["--ensure-rife", "4.26"], progress.append
                )

        self.assertTrue(ok)
        self.assertEqual(error, "")
        self.assertEqual(Path(popen.call_args.args[0][2]), downloader)
        self.assertEqual(Path(popen.call_args.kwargs["cwd"]), root)
        self.assertTrue(any("25%" in line for line in progress))
        self.assertTrue(any("Continuing upscale" in line for line in progress))

    def test_int8_cache_header_validation(self):
        metadata = {
            "seedvr2_int8_convrot": "true",
            "int8_convrot_format": "seedvr2-dit-int8-convrot-v2",
        }
        header = {
            "__metadata__": metadata,
            "block.weight.int8_convrot_groupsize": {
                "dtype": "I32",
                "shape": [],
                "data_offsets": [0, 4],
            },
        }
        payload = json.dumps(header).encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.safetensors"
            path.write_bytes(len(payload).to_bytes(8, "little") + payload + b"\0\0\0\0")
            self.assertTrue(model_downloads._valid_int8_cache(path, "seedvr2"))

    def test_seed_request_selects_only_model_and_int8_modifier(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            model_downloads, "run_model_downloader", return_value=(True, "")
        ) as run:
            ok, _ = model_downloads.ensure_seedvr2_model(
                Path(tmp),
                "seedvr2_ema_7b_sharp_fp16.safetensors",
                True,
            )
        self.assertTrue(ok)
        self.assertEqual(
            run.call_args.args[1],
            [
                "--ensure-seedvr2",
                "seedvr2_ema_7b_sharp_fp16.safetensors",
                "--int8-convrot",
            ],
        )

    def test_flash_request_keeps_selected_version_and_vae(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            model_downloads, "run_model_downloader", return_value=(True, "")
        ) as run:
            ok, _ = model_downloads.ensure_flashvsr_model(
                Path(tmp), "1.1", "int8_convrot", "Wan2.2"
            )
        self.assertTrue(ok)
        self.assertEqual(
            run.call_args.args[1],
            [
                "--ensure-flashvsr",
                "1.1",
                "--flashvsr-vae",
                "Wan2.2",
                "--int8-convrot",
            ],
        )

    def test_other_model_families_use_targeted_commands(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            model_downloads, "run_model_downloader", return_value=(True, "")
        ) as run:
            base = Path(tmp)
            model_downloads.ensure_sparkvsr_model(base, "SparkVSR-int8-convrot")
            model_downloads.ensure_gan_model(base, "4x-UltraSharpV2.safetensors")
            model_downloads.ensure_rife_model(base, "4.26")
        commands = [call.args[1] for call in run.call_args_list]
        self.assertEqual(
            commands,
            [
                ["--ensure-sparkvsr", "SparkVSR-int8-convrot"],
                ["--ensure-gan", "4x-UltraSharpV2.safetensors"],
                ["--ensure-rife", "4.26"],
            ],
        )

    def test_int8_download_failure_keeps_local_generation_fallbacks(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            model_downloads, "run_model_downloader", return_value=(False, "offline")
        ):
            base = Path(tmp)

            seed_dir = base / "SeedVR2" / "models"
            seed_dir.mkdir(parents=True)
            (seed_dir / "seedvr2_ema_3b_fp16.safetensors").touch()
            (seed_dir / "ema_vae_fp16.safetensors").touch()
            self.assertTrue(
                model_downloads.ensure_seedvr2_model(
                    base, "seedvr2_ema_3b_fp16.safetensors", True
                )[0]
            )

            flash_root = base / "ComfyUI-FlashVSR_Stable"
            flash_dir = flash_root / "models" / "FlashVSR-v1.1"
            flash_dir.mkdir(parents=True)
            for name in (
                "LQ_proj_in.ckpt",
                "TCDecoder.ckpt",
                "Wan2.2_VAE.pth",
                "diffusion_pytorch_model_streaming_dmd.safetensors",
            ):
                (flash_dir / name).touch()
            (flash_root / "posi_prompt.pth").touch()
            self.assertTrue(
                model_downloads.ensure_flashvsr_model(
                    base, "1.1", "int8_convrot", "Wan2.2"
                )[0]
            )

            spark_dir = base / "SparkVSR" / "models" / "SparkVSR-bf16"
            for relative in (
                "model_index.json",
                "transformer/config.json",
                "transformer/diffusion_pytorch_model.safetensors",
                "text_encoder/model.safetensors",
                "vae/diffusion_pytorch_model.safetensors",
            ):
                path = spark_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertTrue(
                model_downloads.ensure_sparkvsr_model(base, "SparkVSR-int8-convrot")[0]
            )


if __name__ == "__main__":
    unittest.main()
