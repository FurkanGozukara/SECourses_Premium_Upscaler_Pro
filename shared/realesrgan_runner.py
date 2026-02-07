import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .path_utils import (
    collision_safe_dir,
    collision_safe_path,
    detect_input_type,
    ffmpeg_set_fps,
    get_media_fps,
    normalize_path,
    resolve_output_location,
)
from .face_restore import restore_image, restore_video


class RealESRGANResult:
    def __init__(self, returncode: int, output_path: Optional[str], log: str):
        self.returncode = returncode
        self.output_path = output_path
        self.log = log


def _run_realesrgan_image(model_name: str, inp: Path, out_dir: Path, face_enhance: bool, outscale: int, model_path: Optional[str] = None, gpu_id: Optional[int] = None) -> Tuple[int, str]:
    cmd = [
        "python",
        str(Path(__file__).parent.parent / "Real-ESRGAN" / "inference_realesrgan.py"),
        "-i",
        str(inp),
        "-o",
        str(out_dir),
        "-n",
        model_name,
        "-s",
        str(outscale),
    ]
    if model_path:
        cmd.extend(["--model_path", model_path])
    if gpu_id is not None:
        cmd.extend(["-g", str(gpu_id)])
    if face_enhance:
        cmd.append("--face_enhance")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout + "\n" + proc.stderr


def _run_realesrgan_video(
    model_name: str,
    inp: Path,
    out_dir: Path,
    face_enhance: bool,
    outscale: int,
    model_path: Optional[str] = None,
    gpu_id: Optional[int] = None,
    cancel_event=None,
) -> Tuple[int, str, Optional[Path], Path]:
    frames = out_dir / "frames"
    frames_up = out_dir / "frames_up"
    frames.mkdir(parents=True, exist_ok=True)
    frames_up.mkdir(parents=True, exist_ok=True)

    if cancel_event and cancel_event.is_set():
        return 1, "Canceled before processing", None, frames_up

    # Extract PNG padding from settings (default 6 for consistency)
    png_padding = 6  # Default padding for Real-ESRGAN (will be configurable)
    frame_pattern = f"frame_%0{png_padding}d.png"
    
    subprocess.run(["ffmpeg", "-y", "-i", str(inp), str(frames / frame_pattern)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    rc, log = _run_realesrgan_image(model_name, frames, frames_up, face_enhance, outscale, model_path=model_path, gpu_id=gpu_id)
    if rc != 0:
        return rc, log, None, frames_up

    if cancel_event and cancel_event.is_set():
        return 1, "Canceled after upscale", None, frames_up

    out_path = collision_safe_path(inp.with_name(f"{inp.stem}_realESR.mp4"))
    source_fps = get_media_fps(str(inp)) or 30.0
    
    # Use same padding pattern for upscaled frames
    upscaled_pattern = f"frame_%0{png_padding}d_out.png"
    
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", "30", "-i", str(frames_up / upscaled_pattern), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if source_fps and out_path.exists():
        out_path = ffmpeg_set_fps(out_path, source_fps)
    return 0, log, out_path if out_path.exists() else None, frames_up


def run_realesrgan(
    settings: Dict[str, Any],
    apply_face: bool = False,
    face_strength: float = 0.5,
    cancel_event=None,
    global_output_dir: Optional[str] = None,
) -> RealESRGANResult:
    """
    Run Real-ESRGAN for images or videos using inference_realesrgan.py.
    """
    input_path = Path(normalize_path(settings.get("input_path", "")))
    if not input_path.exists():
        return RealESRGANResult(1, None, "Input missing")
    model_name = settings.get("model") or settings.get("model_name") or "RealESRGAN_x4plus"
    model_name = Path(model_name).stem
    model_path = settings.get("model_path")
    outscale = int(settings.get("scale") or 4)
    fps_override = float(settings.get("fps_override") or 0)
    output_format = settings.get("output_format") or "auto"
    input_type = detect_input_type(str(input_path))
    if output_format == "auto":
        output_format = "png" if input_type == "image" else "mp4"
    gpu_id = None
    if settings.get("cuda_device"):
        try:
            first = str(settings["cuda_device"]).split(",")[0].strip()
            if first.isdigit():
                gpu_id = int(first)
        except Exception:
            gpu_id = None

    work = Path(tempfile.mkdtemp(prefix="realesrgan_"))
    try:
        if cancel_event and cancel_event.is_set():
            return RealESRGANResult(1, None, "Canceled")

        if input_path.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"):
            rc, log, out_path, frames_up = _run_realesrgan_video(
                model_name,
                input_path,
                work,
                apply_face,
                outscale,
                model_path=model_path,
                gpu_id=gpu_id,
                cancel_event=cancel_event,
            )
            if rc != 0 or not out_path:
                return RealESRGANResult(rc, None, log)
            if output_format == "png":
                target_dir = resolve_output_location(str(input_path), "png", global_output_dir, batch_mode=False)
                target_dir = collision_safe_dir(Path(target_dir))
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(frames_up, target_dir, dirs_exist_ok=True)
                out_path = target_dir
            if fps_override and isinstance(out_path, Path) and out_path.suffix.lower() == ".mp4":
                out_path = ffmpeg_set_fps(out_path, fps_override)
            if apply_face and out_path and Path(out_path).exists():
                restored = restore_video(out_path, strength=face_strength, on_progress=None)
                if restored:
                    out_path = Path(restored)
            if output_format == "mp4":
                predicted = resolve_output_location(str(input_path), "mp4", global_output_dir, batch_mode=False)
                dest = collision_safe_path(Path(predicted))
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(out_path), dest)
                out_path = dest
            return RealESRGANResult(0, str(out_path), log)
        else:
            rc, log = _run_realesrgan_image(model_name, input_path, work, apply_face, outscale, model_path=model_path, gpu_id=gpu_id)
            if rc != 0:
                return RealESRGANResult(rc, None, log)
            candidates = sorted(work.glob("*_out.*"))
            if not candidates:
                return RealESRGANResult(1, None, "No outputs produced")
            predicted = resolve_output_location(str(input_path), output_format, global_output_dir, batch_mode=False)
            if Path(predicted).suffix:
                out_path = collision_safe_path(Path(predicted))
            else:
                out_path = collision_safe_path(Path(predicted) / f"{input_path.stem}_realESR{candidates[0].suffix}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(candidates[0], out_path)
            if apply_face and out_path.exists():
                restored = restore_image(str(out_path), strength=face_strength)
                if restored:
                    out_path = Path(restored)
            return RealESRGANResult(0, str(out_path), log)
    finally:
        shutil.rmtree(work, ignore_errors=True)

