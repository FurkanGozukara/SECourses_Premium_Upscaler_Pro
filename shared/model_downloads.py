"""Small bridge from app model selections to the root model downloader."""

from __future__ import annotations

import os
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence


ProgressCallback = Optional[Callable[[str], None]]

DOWNLOADABLE_GAN_MODELS = (
    "2x-AnimeSharpV4_Fast_RCAN_PU.safetensors",
    "2xLiveActionV1_SPAN_490000.pth",
    "2xNomosUni_span_multijpg_ldl.pth",
    "2x_AniScale2_Omni_i16_40K.pth",
    "4x-AnimeSharp.safetensors",
    "4x-UltraSharp.safetensors",
    "4x-UltraSharpV2.safetensors",
    "4xNomos2_hq_dat2.safetensors",
    "4xRealWebPhoto_v4_dat2.safetensors",
    "HAT-L_SRx4_ImageNet-pretrain.safetensors",
    "Kim2091-4x-UltraSharp.safetensors",
    "RealESRGAN_x4plus.safetensors",
    "RealESRGAN_x4plus_anime_6B.safetensors",
)

DOWNLOADABLE_RIFE_MODELS = (
    "4.14",
    "4.15",
    "4.17",
    "4.18",
    "4.20",
    "4.21",
    "4.22",
    "4.25",
    "4.26",
)

_SEED_INT8_CACHE_NAMES = {
    "seedvr2_ema_3b_fp16.safetensors": "seedvr2_ema_3b_fp16_int8_convrot.safetensors",
    "seedvr2_ema_7b_fp16.safetensors": "seedvr2_ema_7b_fp16_int8_convrot.safetensors",
    "seedvr2_ema_7b_sharp_fp16.safetensors": "seedvr2_ema_7b_sharp_fp16_int8_convrot.safetensors",
}
_FLASH_VAE_FILENAMES = {
    "Wan2.1": "Wan2.1_VAE.pth",
    "Wan2.2": "Wan2.2_VAE.pth",
    "LightVAE_W2.1": "lightvaew2_1.pth",
    "TAE_W2.2": "taew2_2.safetensors",
    "LightTAE_HY1.5": "lighttaehy1_5.pth",
}


def windows_int8_defaults_enabled() -> bool:
    return os.environ.get("SECOURSES_WINDOWS_INT8_DEFAULTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _emit(message: str, on_progress: ProgressCallback = None) -> None:
    text = str(message or "")
    if not text:
        return
    if not text.endswith("\n"):
        text += "\n"
    print(text, end="", flush=True)
    if on_progress:
        try:
            on_progress(text)
        except Exception:
            pass


def _all_files_exist(paths: Iterable[Path]) -> bool:
    return all(path.is_file() for path in paths)


def _valid_int8_cache(path: Path, family: str) -> bool:
    markers = {
        "seedvr2": ("seedvr2_int8_convrot", "seedvr2-dit-int8-convrot-v2"),
        "flashvsr": ("flashvsr_int8_convrot", "flashvsr-dit-int8-convrot-v2"),
        "sparkvsr": ("sparkvsr_int8_convrot", "sparkvsr-transformer-int8-convrot-v3"),
    }
    marker_key, format_value = markers[family]
    try:
        with path.open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little")
            if header_size <= 0 or header_size > 256 * 1024 * 1024:
                return False
            header = json.loads(handle.read(header_size))
        metadata = header.get("__metadata__", {})
        return (
            metadata.get(marker_key) == "true"
            and metadata.get("int8_convrot_format") == format_value
            and any(str(key).endswith(".int8_convrot_groupsize") for key in header)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def run_model_downloader(
    base_dir: Path,
    arguments: Sequence[str],
    on_progress: ProgressCallback = None,
    cancel_event=None,
) -> tuple[bool, str]:
    """Run the downloader one folder above the app and stream every line.

    When ``cancel_event`` (a threading.Event) is set mid-download the
    downloader process tree is terminated and (False, "Cancelled by user")
    is returned. Verified partial ranges are preserved, so the next attempt
    resumes where it stopped.
    """
    app_dir = Path(base_dir).resolve()
    downloader_path = app_dir.parent / "Models_Downloader.py"
    if not downloader_path.is_file():
        error = f"Model downloader not found: {downloader_path}"
        _emit(f"[Model Downloader] ERROR: {error}", on_progress)
        return False, error

    command = [sys.executable, "-u", str(downloader_path), *map(str, arguments)]
    _emit(f"[Model Downloader] Preparing the selected model...", on_progress)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cancelled = False
    try:
        process = subprocess.Popen(
            command,
            cwd=str(downloader_path.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        output_lines: list[str] = []
        assert process.stdout is not None

        def _cancel_requested() -> bool:
            return cancel_event is not None and cancel_event.is_set()

        def _watch_cancel() -> None:
            while process.poll() is None:
                if _cancel_requested():
                    try:
                        from shared.process_control import terminate_process_tree

                        terminate_process_tree(process)
                    except Exception:
                        try:
                            process.kill()
                        except Exception:
                            pass
                    return
                time.sleep(0.25)

        watcher = None
        if cancel_event is not None:
            watcher = threading.Thread(target=_watch_cancel, daemon=True)
            watcher.start()

        for line in process.stdout:
            output_lines.append(line.rstrip("\r\n"))
            _emit(line, on_progress)
        return_code = process.wait()
        if watcher is not None:
            watcher.join(timeout=1.0)
        cancelled = _cancel_requested()
    except (OSError, subprocess.SubprocessError) as exc:
        error = f"Could not start model downloader: {exc}"
        _emit(f"[Model Downloader] ERROR: {error}", on_progress)
        return False, error

    if cancelled:
        _emit(
            "[Model Downloader] Download cancelled by user. "
            "Verified partial data is kept; the next run resumes it.",
            on_progress,
        )
        return False, "Cancelled by user"
    if return_code != 0:
        tail = "\n".join(line for line in output_lines[-8:] if line).strip()
        error = tail or f"Model downloader exited with code {return_code}"
        _emit(f"[Model Downloader] ERROR: download failed (code {return_code}).", on_progress)
        return False, error
    _emit("[Model Downloader] Selected model is ready. Continuing upscale.", on_progress)
    return True, ""


def ensure_seedvr2_model(
    base_dir: Path,
    model_filename: str,
    int8_convrot: bool,
    on_progress: ProgressCallback = None,
) -> tuple[bool, str]:
    models_dir = Path(base_dir) / "SeedVR2" / "models"
    model_filename = Path(str(model_filename or "")).name
    vae_path = models_dir / "ema_vae_fp16.safetensors"
    if int8_convrot:
        cache_name = _SEED_INT8_CACHE_NAMES.get(model_filename)
        cache_path = models_dir / cache_name if cache_name else Path()
        if cache_name and vae_path.is_file() and _valid_int8_cache(cache_path, "seedvr2"):
            return True, ""
    elif _all_files_exist((models_dir / model_filename, vae_path)):
        return True, ""

    args = ["--ensure-seedvr2", model_filename]
    if int8_convrot:
        args.append("--int8-convrot")
    ok, error = run_model_downloader(base_dir, args, on_progress)
    if not ok and int8_convrot and _all_files_exist((models_dir / model_filename, vae_path)):
        _emit(
            "[Model Downloader] Prebuilt SeedVR2 cache unavailable; generating it from the local source model.",
            on_progress,
        )
        return True, ""
    return ok, error


def ensure_flashvsr_model(
    base_dir: Path,
    version: str,
    precision: str,
    vae_model: str,
    on_progress: ProgressCallback = None,
) -> tuple[bool, str]:
    version = "1.1" if str(version).strip() in {"1.1", "11"} else "1.0"
    model_dir_name = "FlashVSR-v1.1" if version == "1.1" else "FlashVSR"
    flash_root = Path(base_dir) / "ComfyUI-FlashVSR_Stable"
    models_root = flash_root / "models"
    int8_cache_root = Path(base_dir) / "FlashVSR_plus" / "models"
    model_dir = models_root / model_dir_name
    int8_convrot = str(precision or "").strip().lower() == "int8_convrot"
    required = [
        model_dir / "LQ_proj_in.ckpt",
        model_dir / "TCDecoder.ckpt",
        model_dir / _FLASH_VAE_FILENAMES.get(vae_model, "Wan2.1_VAE.pth"),
        flash_root / "posi_prompt.pth",
    ]
    if int8_convrot:
        required.append(int8_cache_root / f"{model_dir_name}_int8_convrot.safetensors")
    else:
        required.append(model_dir / "diffusion_pytorch_model_streaming_dmd.safetensors")
    cache_ready = True
    if int8_convrot:
        cache_ready = _valid_int8_cache(required[-1], "flashvsr")
    if cache_ready and _all_files_exist(required):
        return True, ""

    args = ["--ensure-flashvsr", version, "--flashvsr-vae", vae_model]
    if int8_convrot:
        args.append("--int8-convrot")
    ok, error = run_model_downloader(base_dir, args, on_progress)
    source_path = model_dir / "diffusion_pytorch_model_streaming_dmd.safetensors"
    if not ok and int8_convrot and _all_files_exist((*required[:-1], source_path)):
        _emit(
            "[Model Downloader] Prebuilt FlashVSR cache unavailable; generating it from the local source model.",
            on_progress,
        )
        return True, ""
    return ok, error


def ensure_sparkvsr_model(
    base_dir: Path,
    model_name: str,
    on_progress: ProgressCallback = None,
) -> tuple[bool, str]:
    models_dir = Path(base_dir) / "SparkVSR" / "models"
    bf16_dir = models_dir / "SparkVSR-bf16"
    model_name = str(model_name or "").strip()
    if model_name == "SparkVSR-int8-convrot":
        required = (
            models_dir / "SparkVSR-int8-convrot.safetensors",
            bf16_dir / "model_index.json",
            bf16_dir / "transformer" / "config.json",
            bf16_dir / "text_encoder" / "model.safetensors",
            bf16_dir / "vae" / "diffusion_pytorch_model.safetensors",
        )
    else:
        required = (
            bf16_dir / "model_index.json",
            bf16_dir / "transformer" / "diffusion_pytorch_model.safetensors",
        )
    cache_ready = True
    if model_name == "SparkVSR-int8-convrot":
        cache_ready = _valid_int8_cache(required[0], "sparkvsr")
    if cache_ready and _all_files_exist(required):
        return True, ""
    ok, error = run_model_downloader(base_dir, ["--ensure-sparkvsr", model_name], on_progress)
    source_path = bf16_dir / "transformer" / "diffusion_pytorch_model.safetensors"
    if not ok and model_name == "SparkVSR-int8-convrot" and _all_files_exist((*required[1:], source_path)):
        _emit(
            "[Model Downloader] Prebuilt SparkVSR cache unavailable; generating it from the local BF16 model.",
            on_progress,
        )
        return True, ""
    return ok, error


def ensure_ltx25_model(
    base_dir: Path,
    model_name: str,
    text_encoder: str,
    video_vae: str,
    on_progress: ProgressCallback = None,
    cancel_event=None,
) -> tuple[bool, str]:
    """Ensure every file the selected LTX 2.5 variant needs exists in LTX25_Models."""
    from shared.ltx25_constants import (
        LTX25_ALWAYS_FILES_TUPLE,
        LTX25_MODELS_DIRNAME,
        LTX25_TE_FILES,
        LTX25_TE_INT8,
        LTX25_TRANSFORMER_FILES,
        LTX25_VAE_CONV,
        LTX25_VAE_FILES,
    )

    models_dir = Path(base_dir) / LTX25_MODELS_DIRNAME
    model_name = str(model_name or "").strip()
    text_encoder = str(text_encoder or "").strip()
    video_vae = str(video_vae or "").strip()
    transformer_file = LTX25_TRANSFORMER_FILES.get(model_name)
    if transformer_file is None:
        return False, f"Unknown LTX 2.5 model: {model_name}"
    te_file = LTX25_TE_FILES.get(text_encoder) or LTX25_TE_FILES[LTX25_TE_INT8]
    vae_file = LTX25_VAE_FILES.get(video_vae) or LTX25_VAE_FILES[LTX25_VAE_CONV]
    required = [
        models_dir / transformer_file,
        models_dir / te_file,
        models_dir / vae_file,
        *(models_dir / name for name in LTX25_ALWAYS_FILES_TUPLE),
    ]
    if _all_files_exist(required):
        return True, ""
    args = [
        "--ensure-ltx25",
        model_name,
        "--ltx25-te",
        text_encoder if text_encoder in LTX25_TE_FILES else LTX25_TE_INT8,
        "--ltx25-vae",
        video_vae if video_vae in LTX25_VAE_FILES else LTX25_VAE_CONV,
    ]
    return run_model_downloader(base_dir, args, on_progress, cancel_event=cancel_event)


def ensure_gan_model(
    base_dir: Path,
    model_filename: str,
    on_progress: ProgressCallback = None,
) -> tuple[bool, str]:
    model_filename = Path(str(model_filename or "")).name
    if any((Path(base_dir) / folder / model_filename).is_file() for folder in ("models", "Image_Upscale_Models")):
        return True, ""
    return run_model_downloader(base_dir, ["--ensure-gan", model_filename], on_progress)


def ensure_rife_model(
    base_dir: Path,
    version: str,
    on_progress: ProgressCallback = None,
) -> tuple[bool, str]:
    version = str(version or "").strip()
    model_dir = Path(base_dir) / "RIFE" / "models" / version
    if (model_dir / "flownet.pkl").is_file() or any(model_dir.glob("*/flownet.pkl")):
        return True, ""
    return run_model_downloader(base_dir, ["--ensure-rife", version], on_progress)
