"""Run an opt-in video restoration model and package its source-aligned MP4."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from shared.video_io import find_ffmpeg, probe_video


MODELS = {
    "ReCaVSR": (
        "kopperx/ReCaVSR", "b98b1f2ec7fc550aed99fd7ec45a5e4b757e4306",
        "kopper/ReCaVSR", "9c874580eedeea516f31ee1f7c562a311dbf9cf7", "recavsr",
        ("transformer.safetensors", "prompt.safetensors", "flashdecoder.safetensors"),
    ),
    "ReCaVSR INT8": (
        "AmesianX/ComfyUI-ReCaVSR", "831769467c89b35ee47926189ac8189be63883d1",
        "PedroMarinhoDev/ReCaVSR-ComfyUI", "2fffebd4413433ad4b8be7aec44b7fb60c897d6a", "recavsr-int8",
        ("recavsr/transformer_int8_convrot.safetensors",),
    ),
    "SwiftVR": (
        "H-oliday/SwiftVR", "4fb82651efee06a90ac47fd2d9af2f632e84ab45",
        "H-oliday/SwiftVR", "a0b09461bd89847712dd91dbb4c63541b4693936", "swift",
        ("reae.safetensors", "prompt_embedding.safetensors", "transformer/config.json",
         "transformer/diffusion_pytorch_model.safetensors"),
    ),
    "VOSR 2.0": (
        "cswry/VOSR", "516f292b99cf23c76fdc33351e86dc4f97711fe8",
        "CSWRY/VOSR", "f24b3061b7f350b81e1907bdcfc27f71fb4ff3f3", "vosr",
        (
            "VOSR2/args.json", "VOSR2/checkpoints/ema_model.safetensors",
            "Qwen-Image-vae-2d/config.json", "Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors",
            "torch_cache/checkpoints/dinov2_vitl14_pretrain.pth",
        ),
    ),
    "HEART": (
        "Rift73/spandrel", "652482479c92b385457b482820381b94221c9d44",
        "Phips/HEART", "cdad6800ddb011a729b1a1f2b4cf6e0e519bdbe6", "spandrel",
        ("models/heart_2x.safetensors", "models/heart_4x_otf_v2.safetensors"),
    ),
    "Row-banded ESRGAN": (
        "Rift73/spandrel", "652482479c92b385457b482820381b94221c9d44",
        "MonsterMMORPG/BestImageUpscalers", "1c71740a5e4c8515e15d304ad96907946e5fc633", "spandrel",
        ("RealESRGAN_x4plus.safetensors",),
    ),
    "SMoVSR": (
        "Rift73/spandrel", "652482479c92b385457b482820381b94221c9d44",
        "", "", "spandrel", (),
    ),
}
IMAGE_MODELS = ("VOSR 2.0", "HEART", "Row-banded ESRGAN")
HDR_MODEL = "LTX 2.5 HDR Restore"
SIZES = {
    "ReCaVSR INT8": {"recavsr/transformer_int8_convrot.safetensors": 6266935048},
    "ReCaVSR": {"transformer.safetensors": 10603673624, "prompt.safetensors": 4194392,
               "flashdecoder.safetensors": 228723776},
    "SwiftVR": {"reae.safetensors": 163797568, "prompt_embedding.safetensors": 4202976,
                "transformer/config.json": 495, "transformer/diffusion_pytorch_model.safetensors": 19999235584},
    "VOSR 2.0": {"VOSR2/args.json": 816, "VOSR2/checkpoints/ema_model.safetensors": 5576385424,
                 "Qwen-Image-vae-2d/config.json": 811, "Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors": 178418892,
                 "torch_cache/checkpoints/dinov2_vitl14_pretrain.pth": 1217586395},
}
SCRIPTS = {
    "ReCaVSR": "inference.py",
    "ReCaVSR INT8": "recavsr/session.py",
    "SwiftVR": "scripts/inference.py",
    "VOSR 2.0": "inference_vosr_onestep.py",
    "HEART": "libs/spandrel/spandrel/__init__.py",
    "Row-banded ESRGAN": "libs/spandrel/spandrel/__init__.py",
    "SMoVSR": "libs/spandrel/spandrel/__init__.py",
}
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def ensure_source(repo: str, commit: str, source: Path, script: str) -> None:
    import requests

    if not (source / script).is_file():
        print(f"Preparing {repo} source at {commit}", flush=True)
        source.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=source.parent) as temp:
            stage = Path(temp) / "checkout"
            archive = Path(temp) / "source.zip"
            with requests.get(f"https://codeload.github.com/{repo}/zip/{commit}", stream=True, timeout=60) as response:
                response.raise_for_status()
                with archive.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
            with ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    parts = member.filename.rstrip("/").split("/")[1:]
                    if not parts or member.is_dir():
                        continue
                    if any(part in ("", ".", "..") or ":" in part or "\\" in part for part in parts):
                        raise ValueError(f"Invalid archive member: {member.filename}")
                    target = stage.joinpath(*parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as data, target.open("wb") as output:
                        shutil.copyfileobj(data, output)
            if not (stage / script).is_file():
                raise RuntimeError(f"{repo} source archive is missing {script}")
            try:
                stage.rename(source)
            except OSError:
                if not (source / script).is_file():
                    raise


def prepare_assets(model: str, root: Path) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    repo, commit, hub_repo, revision, folder, files = MODELS[model]
    source = root / "source" / f"{folder}-{commit}"
    weights = source / "preset/ckpts" if model == "VOSR 2.0" else root / folder
    ensure_source(repo, commit, source, SCRIPTS[model])
    weights.mkdir(parents=True, exist_ok=True)
    for file in files:
        path = weights / file
        expected_size = SIZES.get(model, {}).get(file)
        cached_size = path.stat().st_size if path.is_file() else 0
        if not cached_size or (expected_size and cached_size != expected_size):
            print(f"Downloading {model} {file}", flush=True)
            hf_hub_download(hub_repo, file, revision=revision, local_dir=weights, force_download=path.is_file())
        if expected_size and path.stat().st_size != expected_size:
            raise RuntimeError(f"{model} received an unexpected file size for {file}")
    if model == "ReCaVSR":
        for name in ("model_config.json", "vae_config.json", "flashdecoder_config.json"):
            shutil.copyfile(source / "checkpoints" / name, weights / name)
    return source, weights


def ensure_dependencies(root: Path, packages: dict[str, str]) -> Path:
    from importlib.util import find_spec
    from filelock import FileLock

    target = root / "python-deps"
    target.mkdir(parents=True, exist_ok=True)
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))
    with FileLock(str(target / "install.lock")):
        missing = [package for module, package in packages.items() if find_spec(module) is None]
        if missing:
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps",
                            "--target", str(target), *missing], check=True, creationflags=CREATE_NO_WINDOW)
    return target


def restore_vosr_image(args):
    from PIL import Image

    dependencies = ensure_dependencies(args.assets, {"timm": "timm==1.0.29", "fairscale": "fairscale==0.4.13"})
    source, weights = prepare_assets(args.model, args.assets)
    from huggingface_hub import snapshot_download

    repo_dir = weights / "torch_cache/facebookresearch_dinov2_main"
    if not (repo_dir / "hubconf.py").is_file():
        _, _, repo, revision, _, _ = MODELS[args.model]
        snapshot_download(repo, revision=revision, local_dir=weights,
                          allow_patterns=("torch_cache/facebookresearch_dinov2_main/*.py",
                                          "torch_cache/facebookresearch_dinov2_main/*LICENSE*"))
    with Image.open(args.input) as image:
        size = (image.width * args.scale, image.height * args.scale)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vosr-", dir=output.parent) as work:
        env = dict(os.environ, PYTHONPATH=str(dependencies))
        if sys.platform == "win32":
            env["TORCH_COMPILE_DISABLE"] = "1"
        subprocess.run([
            sys.executable, "-u", SCRIPTS[args.model], "-c", str(weights / "VOSR2"),
            "-i", str(args.input), "-o", work, "-u", str(args.scale),
            "--tile_size", "512", "--vae_tile_size", "1024",
        ], cwd=source, check=True, creationflags=CREATE_NO_WINDOW, env=env, stdout=sys.stdout, stderr=sys.stderr)
        rendered = Path(work) / f"{args.input.stem}.png"
        with Image.open(rendered) as image:
            if image.size != size:
                raise RuntimeError(f"Expected VOSR output size {size}, found {image.size}")
        rendered.replace(output)
    print(f"OUTPUT_PATH: {output}", flush=True)


def restore_spandrel_image(args):
    import torch
    from shared.video_io import read_image, write_image

    source, weights = prepare_assets(args.model, args.assets)
    sys.path.insert(0, str(source / "libs/spandrel"))
    from spandrel import ModelLoader

    if args.model == "HEART":
        file = "models/heart_2x.safetensors" if args.scale == 2 else "models/heart_4x_otf_v2.safetensors"
    else:
        file = "RealESRGAN_x4plus.safetensors"
    model = ModelLoader().load_from_file(str(weights / file)).eval().to("cuda")
    if model.supports_half:
        model.half()
    print(f"{args.model} {model.scale}× · {model.dtype} · {model.device}", flush=True)
    original = torch.from_numpy(read_image(args.input)).permute(2, 0, 1).unsqueeze(0)
    with torch.inference_mode():
        rgb = original.to(device=model.device, dtype=model.dtype).div_(255)
        restored = model(rgb)
        if model.scale != args.scale:
            restored = torch.nn.functional.interpolate(
                restored.float(), scale_factor=args.scale / model.scale, mode="bicubic",
                align_corners=False, antialias=True,
            )
        pixels = restored[0].float().mul_(255).round_().clamp_(0, 255).to(torch.uint8)
        output = pixels.permute(1, 2, 0).contiguous().cpu().numpy()
    write_image(args.output, output)
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB", flush=True)
    print(f"OUTPUT_PATH: {args.output.resolve()}", flush=True)


def restore_smovsr(args):
    import numpy as np
    import torch
    from itertools import islice
    from shared.video_io import FfmpegWriter, iter_video_frames

    if not args.checkpoint or not args.checkpoint.is_file() or args.checkpoint.suffix.lower() != ".safetensors":
        raise ValueError("Choose a trained SMoVSR safetensors checkpoint.")
    source, _ = prepare_assets(args.model, args.assets)
    sys.path.insert(0, str(source / "libs/spandrel"))
    from spandrel import ModelLoader
    from spandrel.architectures.SMoVSR import LOOK_AHEAD, run_window

    desc = ModelLoader().load_from_file(str(args.checkpoint))
    if str(desc.architecture.id) != "SMoVSR":
        raise ValueError(f"Expected an SMoVSR checkpoint, detected {desc.architecture.id}.")
    if desc.scale != args.scale:
        raise ValueError(f"The checkpoint's native scale is {desc.scale}×. Select that scale for this render.")
    desc.eval().to("cuda").half()
    media = probe_video(args.input, count=False)
    frame_count = min(media.frames, args.frames) if args.frames else media.frames
    frames = islice(iter_video_frames(args.input), frame_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smovsr-", dir=args.output.parent) as temp:
        part = Path(temp) / "output.mp4"
        writer = FfmpegWriter(str(part), media.width * args.scale, media.height * args.scale, media.fps,
                              audio_source=str(args.input) if media.has_audio else None,
                              audio_duration_s=float(frame_count / media.fps) if frame_count < media.frames else None)
        carry = None
        done = 0
        step = 12
        pending = list(islice(frames, step + LOOK_AHEAD))
        try:
            with torch.inference_mode():
                while pending:
                    keep = min(step, len(pending))
                    chunk = torch.from_numpy(np.stack(pending)).to(desc.device)
                    chunk = chunk.permute(0, 3, 1, 2).unsqueeze(0).to(desc.dtype).div_(255)
                    out, carry = run_window(desc.model, chunk, keep, carry)
                    writer.write(out[0].permute(0, 2, 3, 1).mul_(255).round_().to(torch.uint8).cpu())
                    pending = pending[keep:] + list(islice(frames, keep))
                    done += keep
                    print(f"FRAME_PROGRESS {done}/{frame_count} frames", flush=True)
            writer.close()
        except BaseException:
            writer.abort()
            raise
        if done != frame_count:
            raise RuntimeError(f"Expected {frame_count} frames, decoded {done}.")
        part.replace(args.output)
    print(f"OUTPUT_PATH: {args.output.resolve()}", flush=True)


def ffmpeg(*args):
    subprocess.run(
        [find_ffmpeg(), "-nostdin", "-v", "error", "-y", *map(str, args)],
        check=True, creationflags=CREATE_NO_WINDOW,
    )


def restore(args):
    media = probe_video(args.input, count=False)
    frames = min(media.frames, args.frames) if args.frames > 0 else media.frames
    width, height = media.width * args.scale, media.height * args.scale
    source, weights = prepare_assets(args.model, args.assets)
    source_audio = args.input
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(source), str(APP)]))
    with tempfile.TemporaryDirectory(prefix="video-restore-", dir=output.parent) as temp:
        temp = Path(temp)
        source_video, intermediate = args.input, temp / "inference.mp4"
        if args.model == "ReCaVSR":
            command = [
                "inference.py", "--input", str(source_video), "--output", str(intermediate),
                "--model-dir", str(weights), "--scale", str(args.scale), "--device", "cuda:0",
                "--color-fix", "wavelet", "--no-compile-blocks", "--no-compile-decoder",
            ]
            if frames < media.frames:
                command += ["--frames", str(frames)]
        else:
            import torch

            step = 16 // args.scale
            padded_w = media.width + (-media.width) % step
            padded_h = media.height + (-media.height) % step
            if (padded_w, padded_h) != (media.width, media.height) or frames < media.frames:
                source_video = temp / "model-input.mp4"
                filters = (
                    f"pad={padded_w}:{padded_h}:0:0,"
                    f"fillborders=right={padded_w - media.width}:bottom={padded_h - media.height}:mode=smear"
                )
                ffmpeg("-i", args.input, "-frames:v", frames, "-vf", filters, "-an", "-threads", 2,
                       "-c:v", "libx264", "-preset", "ultrafast", "-crf", 0, source_video)
            free_gib = torch.cuda.mem_get_info(0)[0] / 1024**3
            if width * height > 5_000_000:
                clip_length = 12 if free_gib > 42 else 4
            else:
                clip_length = 24 if free_gib > 42 else 8
            command = [
                "scripts/inference.py", "--input", str(source_video), "--output", str(intermediate),
                "--checkpoint", str(weights), "--resolution", f"{padded_w * args.scale}x{padded_h * args.scale}",
                "--device", "cuda:0", "--dtype", "bfloat16", "--attention_backend", "sdpa",
                "--fps", str(float(media.fps)), "--quality", "100", "--ffmpeg-preset", "ultrafast",
                "--clip-len", str(clip_length), "--queue-size", "1",
            ]
        subprocess.run([sys.executable, "-u", *command], cwd=source, env=env, check=True,
                       creationflags=CREATE_NO_WINDOW, stdout=sys.stdout, stderr=sys.stderr)
        rendered = probe_video(intermediate, count=False)
        if rendered.frames != frames:
            raise RuntimeError(f"Expected {frames} frames, found {rendered.frames} in {args.model} output")
        if rendered.width < width or rendered.height < height:
            raise RuntimeError(f"{args.model} output {rendered.width}×{rendered.height} is below {width}×{height}")
        audio_input, audio_args = [], []
        if media.has_audio:
            if frames < media.frames:
                audio_input += ["-t", f"{float(frames / media.fps):.9f}"]
            audio_input += ["-i", source_audio]
            audio_args = ["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
        # Use a single common browser-compatible format for the app and comparison.
        video_args = ["-c:v", "copy"]
        if args.model == "SwiftVR" or (rendered.width, rendered.height) != (width, height):
            video_args = ["-vf", f"crop={width}:{height}:0:0", "-c:v", "libx264", "-crf", "14",
                          "-threads", "2", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
        packaged = temp / "packaged.mp4"
        ffmpeg("-i", intermediate, *audio_input, "-map", "0:v:0", *audio_args, *video_args,
               "-movflags", "+faststart", packaged)
        final = probe_video(packaged, count=False)
        if (final.width, final.height, final.frames, final.has_audio) != (width, height, frames, media.has_audio):
            raise RuntimeError("Final video geometry, frame count or audio differs from the export plan")
        packaged.replace(output)
    print(f"OUTPUT_PATH: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, HDR_MODEL), required=True)
    parser.add_argument("--input", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=lambda path: Path(path).resolve(), default=APP / "experimental_models")
    parser.add_argument("--scale", type=int, choices=(2, 4), default=2)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--codec", choices=("libx264", "h264_nvenc"), default="libx264")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hdr-space", choices=("ACEScg", "ACEScct", "Linear Rec.709", "ACES2065-1"), default="ACEScg")
    parser.add_argument("--hdr-fps", type=float, default=24.0)
    parsed = parser.parse_args()
    if parsed.model == "VOSR 2.0":
        restore_vosr_image(parsed)
    elif parsed.model in IMAGE_MODELS:
        restore_spandrel_image(parsed)
    elif parsed.model == "SMoVSR":
        restore_smovsr(parsed)
    elif parsed.model == HDR_MODEL:
        from experimental_ltx_hdr import restore_hdr
        restore_hdr(parsed)
    elif parsed.model == "ReCaVSR INT8":
        from experimental_recavsr_stream import restore_stream
        restore_stream(parsed)
    else:
        restore(parsed)
