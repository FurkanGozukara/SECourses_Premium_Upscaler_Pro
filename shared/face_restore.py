"""
Face Restoration Module
Supports GFPGAN and CodeFormer for face enhancement
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np


# Check available face restoration backends
def _check_gfpgan() -> bool:
    """Check if GFPGAN is available"""
    try:
        import importlib.util
        # Use spec lookup to avoid importing heavy ML stacks (torch/CUDA) in the parent process.
        return importlib.util.find_spec("gfpgan") is not None
    except Exception:
        return False


def _check_codeformer() -> bool:
    """Check if CodeFormer is available"""
    try:
        import importlib.util
        return importlib.util.find_spec("codeformer") is not None
    except Exception:
        return False


GFPGAN_AVAILABLE = _check_gfpgan()
CODEFORMER_AVAILABLE = _check_codeformer()


def get_available_backends() -> list:
    """Get list of available face restoration backends"""
    backends = []
    if GFPGAN_AVAILABLE:
        backends.append("gfpgan")
    if CODEFORMER_AVAILABLE:
        backends.append("codeformer")
    return backends


def restore_image(
    image_path: str,
    strength: float = 0.5,
    backend: str = "auto",
    on_progress: Optional[Callable[[str], None]] = None,
    use_gpu: Optional[bool] = None,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """
    Restore faces in an image using GFPGAN or CodeFormer.
    
    Args:
        image_path: Path to input image
        strength: Restoration strength (0.0-1.0)
        backend: Which backend to use ("auto", "gfpgan", "codeformer")
        on_progress: Optional callback for progress updates
        
    Returns:
        Path to restored image, or None if restoration failed
    """
    if not Path(image_path).exists():
        if on_progress:
            on_progress("⚠️ Input image not found\n")
        return None
    
    # Determine backend
    if backend == "auto":
        if GFPGAN_AVAILABLE:
            backend = "gfpgan"
        elif CODEFORMER_AVAILABLE:
            backend = "codeformer"
        else:
            if on_progress:
                on_progress("⚠️ No face restoration backend available (install GFPGAN or CodeFormer)\n")
            return image_path  # Return original if no backend available
    
    if backend == "gfpgan":
        return _restore_with_gfpgan(image_path, strength, on_progress, use_gpu=use_gpu, output_path=output_path)
    elif backend == "codeformer":
        return _restore_with_codeformer(image_path, strength, on_progress, output_path=output_path)
    else:
        if on_progress:
            on_progress(f"⚠️ Unknown backend: {backend}\n")
        return image_path


def restore_video(
    video_path: str,
    strength: float = 0.5,
    backend: str = "auto",
    on_progress: Optional[Callable[[str], None]] = None,
    use_gpu: Optional[bool] = None,
    output_path: Optional[str] = None,
    preserve_audio: bool = True,
) -> Optional[str]:
    """
    Restore faces in a video frame-by-frame.
    
    Args:
        video_path: Path to input video
        strength: Restoration strength (0.0-1.0)
        backend: Which backend to use ("auto", "gfpgan", "codeformer")
        on_progress: Optional callback for progress updates
        
    Returns:
        Path to restored video, or None if restoration failed
    """
    if not Path(video_path).exists():
        if on_progress:
            on_progress("⚠️ Input video not found\n")
        return None
    
    # Check if any backend is available
    available_backends = get_available_backends()
    if not available_backends:
        if on_progress:
            on_progress("⚠️ No face restoration backend available, skipping face restoration\n")
        return video_path
    
    # Determine backend
    if backend == "auto":
        backend = available_backends[0]
    
    if on_progress:
        on_progress(f"Restoring faces in video using {backend}...\n")
    
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
        
        # Restore each frame
        for i, frame_path in enumerate(frame_files):
            if on_progress and i % 10 == 0:
                progress_pct = int((i / total_frames) * 100)
                on_progress(f"Progress: {progress_pct}% ({i}/{total_frames} frames)\n")
            
            restored_frame = restore_image(
                str(frame_path),
                strength=strength,
                backend=backend,
                on_progress=None,  # Don't spam progress for each frame
                use_gpu=use_gpu,
                output_path=None,
            )
            
            if restored_frame and restored_frame != str(frame_path):
                # Move restored frame to restored directory
                dest_path = restored_dir / frame_path.name
                shutil.move(restored_frame, dest_path)
            else:
                # Copy original frame if restoration failed
                dest_path = restored_dir / frame_path.name
                shutil.copy(frame_path, dest_path)
        
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
        fps_str = fps_result.stdout.strip()
        
        # Parse FPS (format: "30/1" or "30000/1001")
        if "/" in fps_str:
            num, den = map(int, fps_str.split("/"))
            fps = num / den
        else:
            fps = float(fps_str) if fps_str else 30.0
        
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
            "-framerate", str(fps),
            "-i", frames_pattern,
        ]

        if preserve_audio:
            base_cmd += [
                "-i", video_path,
                "-map", "0:v:0",
                "-map", "1:a?",
                "-c:a", "copy",
                "-shortest",
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


def _restore_with_gfpgan(
    image_path: str,
    strength: float,
    on_progress: Optional[Callable[[str], None]] = None,
    use_gpu: Optional[bool] = None,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Restore image using GFPGAN"""
    try:
        from gfpgan import GFPGANer
        import torch
        
        if on_progress:
            on_progress("Loading GFPGAN model...\n")
        
        # Determine device
        if use_gpu is False:
            device = torch.device("cpu")
        elif use_gpu is True:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize GFPGAN
        model_path = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
        upscaler = GFPGANer(
            model_path=model_path,
            upscale=1,  # Don't upscale, just restore
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device=device
        )
        
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            if on_progress:
                on_progress("⚠️ Failed to read image\n")
            return None
        
        # Restore faces
        _, _, restored_img = upscaler.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=strength
        )
        
        # Save restored image (allow override)
        if output_path:
            outp = Path(output_path)
            if outp.suffix == "":
                outp.mkdir(parents=True, exist_ok=True)
                outp = outp / f"{Path(image_path).stem}_gfpgan{Path(image_path).suffix}"
        else:
            outp = Path(image_path).with_stem(f"{Path(image_path).stem}_gfpgan")

        outp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(outp), restored_img)

        return str(outp)
        
    except ImportError:
        if on_progress:
            on_progress("⚠️ GFPGAN not installed\n")
        return image_path
    except Exception as e:
        if on_progress:
            on_progress(f"⚠️ GFPGAN error: {str(e)}\n")
        return image_path


def _restore_with_codeformer(
    image_path: str,
    strength: float,
    on_progress: Optional[Callable[[str], None]] = None,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Restore image using CodeFormer"""
    try:
        # CodeFormer implementation would go here
        # This is a placeholder - actual implementation requires CodeFormer installation
        if on_progress:
            on_progress("⚠️ CodeFormer integration not yet implemented\n")
        # For now, return original (no-op). If output_path specified, still no-op.
        return image_path
        
    except ImportError:
        if on_progress:
            on_progress("⚠️ CodeFormer not installed\n")
        return image_path
    except Exception as e:
        if on_progress:
            on_progress(f"⚠️ CodeFormer error: {str(e)}\n")
        return image_path


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
    
    for i, img_path in enumerate(image_paths):
        if on_progress:
            progress_pct = int((i / total) * 100)
            on_progress(f"Batch progress: {progress_pct}% ({i}/{total})\n")
        
        restored = restore_image(img_path, strength, backend, on_progress=None)
        restored_paths.append(restored if restored else img_path)
    
    return restored_paths
