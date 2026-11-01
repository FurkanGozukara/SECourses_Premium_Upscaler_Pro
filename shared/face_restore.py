"""
Face Restoration Module
Supports GFPGAN (v1.3, v1.4) and CodeFormer for face enhancement.

Every backend shares one pipeline: facexlib RetinaFace detection, 512x512 alignment,
the restoration network, and parsing-mask paste-back (the same steps GFPGANer runs).
"strength" blends the restored faces over the input (0 = unchanged, 1 = full restoration).
CodeFormer also takes a fidelity weight (0 = strongest restoration, 1 = closest to the input).

The restorer (networks + detector) is loaded once and reused for every frame of a video
or a GAN run; it is released when the outermost face-restore call finishes, so no face
model stays in VRAM between runs.

The "auto" backend follows the Face tab's model choice (SECOURSES_FACE_MODEL, set by the
Face tab) and otherwise uses DEFAULT_AUTO_BACKEND. The CodeFormer fidelity used by "auto"
runs comes from SECOURSES_CODEFORMER_FIDELITY.
"""

import os
import shutil
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from .image_io import imread, imwrite, load_image_bgr8
from .path_utils import first_ffprobe_value


APP_DIR = Path(__file__).resolve().parents[1]
# facexlib detection/parsing weights already live here (GFPGANer's "gfpgan/weights" relative to the app).
FACE_WEIGHTS_DIR = APP_DIR / "gfpgan" / "weights"

# Backend id -> weights. "gfpgan" stays GFPGAN v1.3 so older presets keep their result.
FACE_MODELS: Dict[str, Dict[str, str]] = {
    "gfpgan": {
        "label": "GFPGAN v1.3",
        "file": "GFPGANv1.3.pth",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
    },
    "gfpgan_v1.4": {
        "label": "GFPGAN v1.4",
        "file": "GFPGANv1.4.pth",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    },
    "codeformer": {
        "label": "CodeFormer",
        "file": "codeformer.pth",
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    },
}
FACE_BACKEND_CHOICES = ("auto", "gfpgan", "gfpgan_v1.4", "codeformer")
# "auto" = GFPGAN v1.4: on the eval kit's face clip and images it kept closer to the ground truth
# than v1.3 (face PSNR +0.9 to +1.0 dB) with less frame-to-frame flicker (results/WP-GAN-FACE).
# When its weights cannot be fetched, auto falls back to v1.3.
DEFAULT_AUTO_BACKEND = "gfpgan_v1.4"
AUTO_FALLBACK_BACKEND = "gfpgan"
# 0.7: on the eval kit slightly closer to the ground truth and steadier than 0.5 (results/WP-GAN-FACE).
DEFAULT_CODEFORMER_FIDELITY = 0.7
# GFPGAN's decoder injects fresh random noise on every call (GFPGANer's default). Fixed noise
# (the trained noise buffers) makes a frame's result repeatable; see GFPGAN_FIXED_NOISE use below.
GFPGAN_FIXED_NOISE = False

_ALIASES = {
    "gfpgan": "gfpgan",
    "gfpgan1.3": "gfpgan",
    "gfpganv1.3": "gfpgan",
    "gfpgan_v1.3": "gfpgan",
    "gfpgan-v1.3": "gfpgan",
    "gfpgan1.4": "gfpgan_v1.4",
    "gfpganv1.4": "gfpgan_v1.4",
    "gfpgan_v1.4": "gfpgan_v1.4",
    "gfpgan-v1.4": "gfpgan_v1.4",
    "codeformer": "codeformer",
    "auto": "auto",
}


# Check available face restoration backends
def _module_available(name: str) -> bool:
    try:
        import importlib.util
        # Use spec lookup to avoid importing heavy ML stacks (torch/CUDA) in the parent process.
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _check_gfpgan() -> bool:
    """Check if GFPGAN is available"""
    return _module_available("gfpgan") and _module_available("facexlib")


def _check_codeformer() -> bool:
    """CodeFormer's network ships with the app (shared/face_archs); it needs facexlib for detection."""
    return _module_available("facexlib") and _module_available("basicsr")


GFPGAN_AVAILABLE = _check_gfpgan()
CODEFORMER_AVAILABLE = _check_codeformer()


def _resolve_global_cuda_device_token(gpu_device: Any = None) -> str:
    """Resolve a face-restoration GPU token from an explicit override or app env."""
    source = gpu_device if gpu_device is not None else os.environ.get("SECOURSES_GLOBAL_GPU_DEVICE", "")
    token = str(source or "").strip().lower()
    if token.startswith("cuda:"):
        token = token.split(":", 1)[1].strip()
    if "," in token:
        token = token.split(",", 1)[0].strip()
    return token


def get_available_backends() -> list:
    """Get list of available face restoration backends"""
    backends = []
    if GFPGAN_AVAILABLE:
        backends.extend(["gfpgan", "gfpgan_v1.4"])
    if CODEFORMER_AVAILABLE:
        backends.append("codeformer")
    return backends


def normalize_face_backend(value: Any) -> str:
    """Map UI/preset spellings to a backend id; unknown names become "auto"."""
    token = str(value or "auto").strip().lower().replace(" ", "")
    return _ALIASES.get(token, "auto")


def codeformer_fidelity_default() -> float:
    try:
        value = float(os.environ.get("SECOURSES_CODEFORMER_FIDELITY", DEFAULT_CODEFORMER_FIDELITY))
    except Exception:
        value = DEFAULT_CODEFORMER_FIDELITY
    return max(0.0, min(1.0, value))


def resolve_face_backend(backend: Any = "auto") -> str:
    """Concrete backend for a request; "" when none is available."""
    available = get_available_backends()
    choice = normalize_face_backend(backend)
    if choice == "auto":
        preferred = normalize_face_backend(os.environ.get("SECOURSES_FACE_MODEL", "auto"))
        choice = preferred if preferred != "auto" else DEFAULT_AUTO_BACKEND
    if choice in available:
        return choice
    return available[0] if available else ""


def face_backend_label(backend: Any) -> str:
    resolved = resolve_face_backend(backend)
    return FACE_MODELS.get(resolved, {}).get("label", resolved or "none")


def _resolve_device(use_gpu: Optional[bool], gpu_device: Any):
    import torch

    global_token = _resolve_global_cuda_device_token(gpu_device)
    if use_gpu is False or global_token in {"cpu", "none", "off"} or not torch.cuda.is_available():
        return torch.device("cpu")
    if global_token.isdigit():
        did = int(global_token)
        if 0 <= did < torch.cuda.device_count():
            return torch.device(f"cuda:{did}")
    return torch.device("cuda")


def _download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "SECourses-Upscaler"})
    with urllib.request.urlopen(req, timeout=120) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle, 1 << 22)
    tmp.replace(dest)


def ensure_face_weights(backend: str, on_progress: Optional[Callable[[str], None]] = None) -> Path:
    """Path of a backend's weights: app face folder, older GFPGAN download folder, model
    downloader (Hugging Face mirror), then the original release URL."""
    info = FACE_MODELS[backend]
    target = FACE_WEIGHTS_DIR / info["file"]
    if target.is_file():
        return target
    try:
        import gfpgan as _gfpgan_pkg

        legacy = Path(_gfpgan_pkg.__file__).resolve().parent / "weights" / info["file"]
        if legacy.is_file():
            return legacy
    except Exception:
        pass
    try:
        from .model_downloads import ensure_face_model

        ok, _err = ensure_face_model(APP_DIR, info["file"], on_progress)
        if ok and target.is_file():
            return target
    except Exception:
        pass
    if on_progress:
        on_progress(f"Downloading {info['label']} weights...\n")
    _download(info["url"], target)
    return target


class _FaceRestorer:
    """Detection + alignment + restoration network + paste-back for one backend on one device."""

    def __init__(self, backend: str, device, on_progress: Optional[Callable[[str], None]] = None):
        import torch
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper

        self.backend = backend
        self.device = device
        FACE_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        self.helper = FaceRestoreHelper(
            1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=device,
            model_rootpath=str(FACE_WEIGHTS_DIR),
        )
        weights = ensure_face_weights(backend, on_progress)
        checkpoint = torch.load(str(weights), map_location="cpu", weights_only=True)
        if backend == "codeformer":
            from .face_archs.codeformer_arch import CodeFormer

            state = checkpoint.get("params_ema", checkpoint)
            connect = [k for k in ("16", "32", "64", "128", "256", "512") if f"fuse_convs_dict.{k}.scale.0.weight" in state]
            n_layers = len({key.split(".")[1] for key in state if key.startswith("ft_layers.")})
            net = CodeFormer(
                dim_embd=int(state["position_emb"].shape[1]),
                n_head=8,
                n_layers=n_layers,
                codebook_size=int(state["idx_pred_layer.1.weight"].shape[0]),
                latent_size=int(state["position_emb"].shape[0]),
                connect_list=connect,
                fix_modules=["quantize", "generator"],
            )
        else:
            from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

            state = checkpoint["params_ema"] if "params_ema" in checkpoint else checkpoint["params"]
            net = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=2,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        net.load_state_dict(state, strict=True)
        self.net = net.eval().to(device)
        self.lock = threading.Lock()

    def restore(self, img_bgr: np.ndarray, strength: float, fidelity: float,
                fixed_noise: Optional[bool] = None) -> Tuple[np.ndarray, int]:
        """BGR uint8 image -> (restored BGR uint8 image, number of faces)."""
        if fixed_noise is None:
            fixed_noise = GFPGAN_FIXED_NOISE
        import torch
        from basicsr.utils import img2tensor, tensor2img
        from torchvision.transforms.functional import normalize

        strength = max(0.0, min(1.0, float(strength)))
        if strength <= 0.0:
            return img_bgr, 0
        with self.lock, torch.no_grad():
            helper = self.helper
            helper.clean_all()
            helper.read_image(img_bgr)
            helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
            helper.align_warp_face()
            faces = list(helper.cropped_faces)
            if not faces:
                return img_bgr, 0
            for cropped in faces:
                tensor = img2tensor(cropped / 255.0, bgr2rgb=True, float32=True)
                normalize(tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                tensor = tensor.unsqueeze(0).to(self.device)
                try:
                    if self.backend == "codeformer":
                        output = self.net(tensor, weight=float(fidelity))[0]
                    else:
                        output = self.net(tensor, return_rgb=False, randomize_noise=not fixed_noise)[0]
                    restored = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1)).astype("uint8")
                except RuntimeError as error:
                    print(f"Face restoration failed for one face: {error}")
                    restored = cropped
                helper.add_restored_face(restored)
            helper.get_inverse_affine(None)
            result = helper.paste_faces_to_input_image(upsample_img=None)
        if strength < 1.0:
            blended = cv2.addWeighted(result.astype(np.float32), strength, img_bgr.astype(np.float32), 1.0 - strength, 0.0)
            result = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
        return result, len(faces)


_RESTORERS: Dict[Tuple[str, str], _FaceRestorer] = {}
_RESTORER_LOCK = threading.RLock()
_SESSION_DEPTH = 0


def release_face_restorers() -> None:
    with _RESTORER_LOCK:
        _RESTORERS.clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@contextmanager
def face_restore_session():
    """Keep the loaded restorer across calls inside the block; release it at the outermost exit."""
    global _SESSION_DEPTH
    with _RESTORER_LOCK:
        _SESSION_DEPTH += 1
    try:
        yield
    finally:
        with _RESTORER_LOCK:
            _SESSION_DEPTH -= 1
            last = _SESSION_DEPTH == 0
        if last:
            release_face_restorers()


def _get_restorer(backend: str, use_gpu: Optional[bool], gpu_device: Any,
                  on_progress: Optional[Callable[[str], None]] = None) -> _FaceRestorer:
    device = _resolve_device(use_gpu, gpu_device)
    key = (backend, str(device))
    with _RESTORER_LOCK:
        restorer = _RESTORERS.get(key)
        if restorer is None:
            _RESTORERS.clear()
            if on_progress:
                on_progress(f"Loading {FACE_MODELS[backend]['label']} face restoration...\n")
            restorer = _FaceRestorer(backend, device, on_progress)
            _RESTORERS[key] = restorer
        return restorer


def restore_array(
    img_bgr: np.ndarray,
    strength: float = 0.5,
    backend: str = "auto",
    use_gpu: Optional[bool] = None,
    gpu_device: Any = None,
    fidelity: Optional[float] = None,
    on_progress: Optional[Callable[[str], None]] = None,
    fixed_noise: Optional[bool] = None,
) -> np.ndarray:
    """Restore faces in a BGR uint8 array (returned unchanged when no backend is available)."""
    resolved = resolve_face_backend(backend)
    if not resolved:
        return img_bgr
    if fidelity is None:
        fidelity = codeformer_fidelity_default()
    with face_restore_session():
        try:
            restorer = _get_restorer(resolved, use_gpu, gpu_device, on_progress)
        except Exception as exc:
            # An "auto" choice whose weights cannot be fetched falls back to GFPGAN v1.3.
            if normalize_face_backend(backend) != "auto" or resolved == AUTO_FALLBACK_BACKEND:
                raise
            if on_progress:
                on_progress(f"{FACE_MODELS[resolved]['label']} unavailable ({exc}); using {FACE_MODELS[AUTO_FALLBACK_BACKEND]['label']}\n")
            restorer = _get_restorer(AUTO_FALLBACK_BACKEND, use_gpu, gpu_device, on_progress)
        restored, _faces = restorer.restore(img_bgr, strength, fidelity, fixed_noise)
    return restored


def restore_image(
    image_path: str,
    strength: float = 0.5,
    backend: str = "auto",
    on_progress: Optional[Callable[[str], None]] = None,
    use_gpu: Optional[bool] = None,
    output_path: Optional[str] = None,
    gpu_device: Any = None,
    fidelity: Optional[float] = None,
) -> Optional[str]:
    """
    Restore faces in an image using GFPGAN or CodeFormer.

    Args:
        image_path: Path to input image
        strength: Restoration strength (0.0-1.0)
        backend: Which backend to use ("auto", "gfpgan", "gfpgan_v1.4", "codeformer")
        on_progress: Optional callback for progress updates
        fidelity: CodeFormer fidelity (None = Face tab setting)

    Returns:
        Path to restored image, or None if restoration failed
    """
    if not Path(image_path).exists():
        if on_progress:
            on_progress("⚠️ Input image not found\n")
        return None

    resolved = resolve_face_backend(backend)
    if not resolved:
        if on_progress:
            on_progress("⚠️ No face restoration backend available (install GFPGAN or CodeFormer)\n")
        return image_path  # Return original if no backend available

    try:
        img = load_image_bgr8(image_path)
        if img is None:
            if on_progress:
                on_progress("⚠️ Failed to read image\n")
            return None
        restored_img = restore_array(
            img,
            strength=strength,
            backend=backend,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            fidelity=fidelity,
            on_progress=on_progress,
        )

        # Save restored image (allow override). The "_gfpgan" suffix is kept for GFPGAN
        # outputs so existing output names do not change.
        suffix = "_codeformer" if resolved == "codeformer" else "_gfpgan"
        if output_path:
            outp = Path(output_path)
            if outp.suffix == "":
                outp.mkdir(parents=True, exist_ok=True)
                outp = outp / f"{Path(image_path).stem}{suffix}{Path(image_path).suffix}"
        else:
            outp = Path(image_path).with_stem(f"{Path(image_path).stem}{suffix}")

        outp.parent.mkdir(parents=True, exist_ok=True)
        imwrite(outp, restored_img)
        return str(outp)
    except ImportError as e:
        if on_progress:
            on_progress(f"⚠️ Face restoration package missing: {e}\n")
        return image_path
    except Exception as e:
        if on_progress:
            on_progress(f"⚠️ Face restoration error ({resolved}): {str(e)}\n")
        return image_path


def restore_video(
    video_path: str,
    strength: float = 0.5,
    backend: str = "auto",
    on_progress: Optional[Callable[[str], None]] = None,
    use_gpu: Optional[bool] = None,
    output_path: Optional[str] = None,
    preserve_audio: bool = True,
    gpu_device: Any = None,
    fidelity: Optional[float] = None,
) -> Optional[str]:
    """
    Restore faces in a video frame-by-frame.

    Args:
        video_path: Path to input video
        strength: Restoration strength (0.0-1.0)
        backend: Which backend to use ("auto", "gfpgan", "gfpgan_v1.4", "codeformer")
        on_progress: Optional callback for progress updates
        fidelity: CodeFormer fidelity (None = Face tab setting)

    Returns:
        Path to restored video, or None if restoration failed
    """
    if not Path(video_path).exists():
        if on_progress:
            on_progress("⚠️ Input video not found\n")
        return None

    resolved = resolve_face_backend(backend)
    if not resolved:
        if on_progress:
            on_progress("⚠️ No face restoration backend available, skipping face restoration\n")
        return video_path
    if fidelity is None:
        fidelity = codeformer_fidelity_default()

    if on_progress:
        detail = f", fidelity {fidelity:g}" if resolved == "codeformer" else ""
        on_progress(f"Restoring faces in video using {FACE_MODELS[resolved]['label']} (strength {float(strength):g}{detail})...\n")

    # Create temp directory for frames
    work_dir = Path(tempfile.mkdtemp(prefix="face_restore_"))
    frames_dir = work_dir / "frames"
    restored_dir = work_dir / "restored"
    frames_dir.mkdir(parents=True, exist_ok=True)
    restored_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Extract frames
        if on_progress:
            on_progress("Extracting frames...\n")

        extract_cmd = [
            "ffmpeg", "-y", "-i", video_path,
            str(frames_dir / "frame_%05d.png")
        ]
        subprocess.run(extract_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Get frame list
        frame_files = sorted(frames_dir.glob("*.png"))
        total_frames = len(frame_files)

        if total_frames == 0:
            if on_progress:
                on_progress("⚠️ No frames extracted\n")
            shutil.rmtree(work_dir, ignore_errors=True)
            return video_path

        if on_progress:
            on_progress(f"Restoring {total_frames} frames...\n")

        # Restore each frame with one loaded restorer.
        with face_restore_session():
            for i, frame_path in enumerate(frame_files):
                if on_progress and i % 10 == 0:
                    progress_pct = int((i / total_frames) * 100)
                    on_progress(f"Progress: {progress_pct}% ({i}/{total_frames} frames)\n")
                dest_path = restored_dir / frame_path.name
                img = imread(frame_path, cv2.IMREAD_COLOR)
                if img is None:
                    shutil.copy(frame_path, dest_path)
                    continue
                restored = restore_array(
                    img,
                    strength=strength,
                    backend=backend,
                    use_gpu=use_gpu,
                    gpu_device=gpu_device,
                    fidelity=fidelity,
                    on_progress=on_progress if i == 0 else None,
                )
                imwrite(dest_path, restored, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        # Reconstruct video
        if on_progress:
            on_progress("Reconstructing video...\n")

        # Get original FPS
        fps_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        fps_result = subprocess.run(fps_cmd, capture_output=True, text=True)
        fps_str = first_ffprobe_value(fps_result.stdout)

        # Parse FPS (format: "30/1" or "30000/1001")
        if "/" in fps_str:
            num, den = map(int, fps_str.split("/"))
            fps = num / den
        else:
            fps = float(fps_str) if fps_str else 30.0
        fps_arg = fps_str if fps_str and fps > 0 else str(fps)

        # Create output path (allow override)
        if output_path:
            outp = Path(output_path)
            # If a directory is provided, generate a file name inside it.
            if outp.suffix == "":
                outp.mkdir(parents=True, exist_ok=True)
                outp = outp / f"{Path(video_path).stem}_face_restored{Path(video_path).suffix}"
        else:
            outp = Path(video_path).with_stem(f"{Path(video_path).stem}_face_restored")

        outp.parent.mkdir(parents=True, exist_ok=True)

        # Encode video (preserve audio when possible)
        frames_pattern = str(restored_dir / "frame_%05d.png")

        base_cmd = [
            "ffmpeg", "-y",
            "-framerate", fps_arg,
            "-i", frames_pattern,
        ]

        if preserve_audio:
            base_cmd += [
                "-i", video_path,
                "-map", "0:v:0",
                "-map", "1:a?",
                "-c:a", "copy",
            ]
        else:
            base_cmd += ["-an"]

        encode_cmd = base_cmd + [
            "-c:v", "libx264",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(outp),
        ]

        try:
            subprocess.run(encode_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception:
            # If audio copy fails, fall back to AAC (still optional)
            if preserve_audio:
                encode_cmd_fallback = base_cmd.copy()
                # Replace "-c:a copy" with AAC
                try:
                    idx = encode_cmd_fallback.index("-c:a")
                    if idx + 1 < len(encode_cmd_fallback):
                        encode_cmd_fallback[idx + 1] = "aac"
                    encode_cmd_fallback += ["-b:a", "192k"]
                except Exception:
                    # If structure changes, just append audio settings
                    encode_cmd_fallback += ["-c:a", "aac", "-b:a", "192k"]

                encode_cmd_fallback += [
                    "-c:v", "libx264",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    str(outp),
                ]
                subprocess.run(encode_cmd_fallback, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            else:
                raise

        # Cleanup
        shutil.rmtree(work_dir, ignore_errors=True)

        if on_progress:
            on_progress(f"✅ Face restoration complete: {outp}\n")

        return str(outp)

    except Exception as e:
        if on_progress:
            on_progress(f"❌ Face restoration error: {str(e)}\n")
        shutil.rmtree(work_dir, ignore_errors=True)
        return video_path


# Batch face restoration
def restore_images_batch(
    image_paths: list,
    strength: float = 0.5,
    backend: str = "auto",
    on_progress: Optional[Callable[[str], None]] = None
) -> list:
    """
    Restore faces in multiple images.

    Args:
        image_paths: List of paths to input images
        strength: Restoration strength (0.0-1.0)
        backend: Which backend to use
        on_progress: Optional callback for progress updates

    Returns:
        List of paths to restored images
    """
    restored_paths = []
    total = len(image_paths)

    with face_restore_session():
        for i, img_path in enumerate(image_paths):
            if on_progress:
                progress_pct = int((i / total) * 100)
                on_progress(f"Batch progress: {progress_pct}% ({i}/{total})\n")

            restored = restore_image(img_path, strength, backend, on_progress=None)
            restored_paths.append(restored if restored else img_path)

    return restored_paths
