"""
RIFE Frame Interpolation Runner

Provides interface to RIFE for:
- Frame interpolation (FPS increase)
- Video slow-motion generation
- Image interpolation mode
- Video editing features (trim, speed change, etc.)
"""

import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .path_utils import (
    normalize_path,
    collision_safe_path,
    detect_input_type,
    get_media_fps,
)


@dataclass
class RifeResult:
    """Result of RIFE processing"""
    returncode: int
    output_path: Optional[str]
    log: str
    input_fps: float = 30.0
    output_fps: float = 60.0
    frames_generated: int = 0


def discover_rife_models(base_dir: Path) -> List[str]:
    """
    Discover available RIFE models in train_log directory.
    
    Args:
        base_dir: Base directory containing RIFE folder
        
    Returns:
        List of available model names
    """
    rife_dir = base_dir / "RIFE" / "train_log"
    
    # Default models list (known versions)
    default_models = [
        "rife-v4.6",
        "rife-v4.13", 
        "rife-v4.14",
        "rife-v4.15",
        "rife-v4.16",
        "rife-v4.17",
        "rife-anime"
    ]
    
    if not rife_dir.exists():
        return default_models
    
    # Scan for actual model directories
    discovered = []
    for item in rife_dir.iterdir():
        if item.is_dir() and not item.name.startswith("_") and not item.name.startswith("."):
            # Verify it has model files (*.pkl or *.pth)
            has_model = any(
                f.suffix.lower() in (".pkl", ".pth") 
                for f in item.rglob("*")
            )
            if has_model:
                discovered.append(item.name)
    
    # Return discovered if found, else defaults
    return sorted(discovered) if discovered else default_models


def resolve_rife_model_dir(model_name: str, base_dir: Path) -> Optional[Path]:
    """
    Resolve a RIFE model name to its weights directory (contains flownet.pkl).

    Supports both local layouts (same rules as shared/models/rife_meta.py):
    - RIFE/models/<name>/flownet.pkl (current repo layout, e.g. models/4.25)
    - RIFE/train_log/<name>/flownet.pkl (legacy layout)
    plus a one-level nested folder inside either (zip extraction artifacts),
    and 'rife-v4.25' aliases for a plain '4.25' folder name.
    """
    from shared.models.rife_meta import _resolve_model_bundle_dir

    name = str(model_name or "").strip()
    candidates = [name]
    if name.lower().startswith("rife-v"):
        candidates.append(name[len("rife-v"):])
    for root in (base_dir / "RIFE" / "models", base_dir / "RIFE" / "train_log"):
        for candidate in candidates:
            bundle = _resolve_model_bundle_dir(root / candidate)
            if bundle is not None:
                return bundle
    return None


def _ensure_train_log_for_model(resolved_model_dir: Path, base_dir: Path, log) -> bool:
    """
    Stage the selected RIFE model into RIFE/train_log.

    inference_video.py (and the model bundles themselves) hardcode imports like
    `from train_log.RIFE_HDv3 import Model`, so whichever model the user picks
    must be present directly inside RIFE/train_log/. The repo ships models in
    versioned folders (RIFE/models/4.25/...), so we copy the selected bundle
    over (a ~50 MB one-time copy per model switch, tracked via a marker file).
    """
    import shutil

    rife_root = base_dir / "RIFE"
    train_log = rife_root / "train_log"
    resolved = Path(resolved_model_dir).resolve()
    marker = train_log / ".model_source.txt"
    try:
        if train_log.exists() and marker.exists():
            if marker.read_text(encoding="utf-8").strip() == str(resolved):
                return True
        if train_log.exists():
            if marker.exists() or not (train_log / "flownet.pkl").exists():
                shutil.rmtree(train_log, ignore_errors=True)
            else:
                # A real train_log the user populated manually - keep it safe.
                backup = rife_root / "train_log_backup"
                if backup.exists():
                    shutil.rmtree(backup, ignore_errors=True)
                train_log.rename(backup)
                log(f"Existing train_log preserved as {backup}")
        train_log.mkdir(parents=True, exist_ok=True)
        for item in resolved.iterdir():
            if item.is_file():
                shutil.copy2(item, train_log / item.name)
        marker.write_text(str(resolved), encoding="utf-8")
        return True
    except Exception as exc:
        log(f"Failed to stage RIFE model into train_log: {exc}")
        return False


def validate_rife_model(model_name: str, base_dir: Path) -> Tuple[bool, str]:
    """
    Validate that RIFE model exists and has required files.
    
    Args:
        model_name: Model name (e.g., "rife-v4.6" or "4.25")
        base_dir: Base directory
        
    Returns:
        (is_valid, message)
    """
    model_dir = resolve_rife_model_dir(model_name, base_dir)
    if model_dir is None:
        return False, (
            f"Model directory not found for '{model_name}' "
            f"(searched RIFE/models and RIFE/train_log, including rife-v aliases)"
        )
    return True, f"Model {model_name} validated"


def run_rife_interpolation(
    settings: Dict[str, Any],
    base_dir: Path,
    on_progress: Optional[Callable[[str], None]] = None
) -> RifeResult:
    """
    Run RIFE frame interpolation.
    
    settings must include:
    - input_path: str
    - rife_model: str (model name)
    - target_fps: float (or 0 for multiplier mode)
    - fps_multiplier: str (x2, x4, x8)
    - rife_precision: str (fp16, fp32)
    - gpu_device: str
    - include_audio: bool
    - output_format: str (mp4, png)
    
    Returns:
        RifeResult with processing outcome
    """
    log_lines = []
    
    def log(msg: str):
        log_lines.append(msg)
        if on_progress:
            on_progress(msg + "\n")
    
    try:
        # Check for deprecated parameters and warn
        legacy_warnings = []
        if settings.get("skip_first_frames", 0) > 0:
            legacy_warnings.append(
                "⚠️ DEPRECATED: 'skip_first_frames' is a legacy parameter. "
                "Please use the 'Video Editing' tab's trim feature instead for better control."
            )
        if settings.get("load_cap", 0) > 0:
            legacy_warnings.append(
                "⚠️ DEPRECATED: 'load_cap' is a legacy parameter. "
                "Please use the 'Video Editing' tab's trim feature to specify end time."
            )
        
        # Log warnings
        for warning in legacy_warnings:
            log(warning)
        
        # Validate input
        input_path = normalize_path(settings.get("input_path", ""))
        if not input_path or not Path(input_path).exists():
            return RifeResult(returncode=1, output_path=None, log="Invalid input path")
        
        input_type = detect_input_type(input_path)
        
        # Get settings
        model_name = settings.get("rife_model", "rife-v4.6")
        target_fps = float(settings.get("target_fps", 0))
        fps_multiplier = settings.get("fps_multiplier", "x2")
        precision_fp16 = settings.get("rife_precision", "fp16") == "fp16"
        gpu_device = settings.get("gpu_device", "0")
        include_audio = settings.get("include_audio", True)
        output_format = settings.get("output_format", "mp4")
        
        # Validate model
        is_valid, msg = validate_rife_model(model_name, base_dir)
        if not is_valid:
            log(f"⚠️ {msg}")
            return RifeResult(returncode=1, output_path=None, log="\n".join(log_lines))
        
        log(f"Model: {model_name}")
        log(f"Input: {input_path}")
        
        # Build RIFE command
        rife_script = base_dir / "RIFE" / "inference_video.py"
        
        compat_wrapper = base_dir / "tools" / "rife_inference_wrapper.py"
        if not compat_wrapper.is_file():
            return RifeResult(
                returncode=1,
                output_path=None,
                log=f"RIFE compatibility wrapper not found: {compat_wrapper}",
            )

        if input_type == "image" and settings.get("img_mode", False):
            # Image interpolation mode
            rife_script = base_dir / "RIFE" / "inference_img.py"
            cmd = [
                sys.executable,
                str(compat_wrapper),
                str(rife_script),
                "--img", input_path,
            ]
        else:
            # Video interpolation
            cmd = [
                sys.executable,
                str(compat_wrapper),
                str(rife_script),
                "--video", input_path,
            ]
        
        # Output path
        output_override = settings.get("output_override")
        if output_override:
            output_path = Path(normalize_path(output_override))
        else:
            # Auto-generate output path
            input_file = Path(input_path)
            suffix = "_interpolated"
            if fps_multiplier != "x2":
                suffix = f"_rife_{fps_multiplier}"
            output_path = collision_safe_path(
                input_file.with_stem(f"{input_file.stem}{suffix}")
            )
        
        cmd.extend(["--output", str(output_path)])
        
        # Model directory: inference_video.py expects a directory path for
        # --model (dest=modelDir) and imports the model classes from the
        # train_log package, so stage the selected bundle there first.
        resolved_model_dir = resolve_rife_model_dir(model_name, base_dir)
        if resolved_model_dir is not None:
            _ensure_train_log_for_model(resolved_model_dir, base_dir, log)
        staged_dir = base_dir / "RIFE" / "train_log"
        cmd.extend(["--model", str(staged_dir if (staged_dir / "flownet.pkl").exists() else (resolved_model_dir or model_name))])
        
        # FPS settings
        if target_fps > 0:
            cmd.extend(["--fps", str(target_fps)])
        else:
            # Use multiplier
            exp = {"x2": 1, "x4": 2, "x8": 3}.get(fps_multiplier, 1)
            cmd.extend(["--exp", str(exp)])
        
        # Precision
        if precision_fp16:
            cmd.append("--fp16")
        
        # UHD support for 4K
        resolution = int(settings.get("resolution", 1080))
        if resolution >= 2160:
            cmd.append("--UHD")
            cmd.extend(["--scale", "0.5"])
        
        # Output format
        if output_format == "png":
            cmd.append("--png")
        else:
            cmd.extend(["--ext", output_format])
        
        # Audio handling
        if not include_audio:
            cmd.append("--no-audio")
        
        # Montage mode (side-by-side comparison)
        if settings.get("montage", False):
            cmd.append("--montage")
            log("📊 Montage mode enabled - will create side-by-side comparison")
        
        # Skip deprecated flag warning
        if settings.get("skip_static_frames", False):
            log("⚠️ --skip flag is deprecated in RIFE (see issue #207). Ignoring.")
        
        # Log command
        cmd_str = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        log(f"Executing: {cmd_str}")
        
        # Set environment
        import os
        env = os.environ.copy()
        if gpu_device:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_device).split(",")[0]  # RIFE uses first GPU
        
        # Execute
        log("Starting RIFE interpolation...")
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=base_dir / "RIFE"
        )
        
        # Stream output
        if proc.stdout:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    log(line)
        
        proc.wait()
        
        if proc.returncode == 0 and output_path.exists():
            log(f"✅ RIFE completed: {output_path}")
            
            # Get FPS info
            input_fps = get_media_fps(input_path) or 30.0
            output_fps = get_media_fps(str(output_path)) or input_fps * 2
            
            return RifeResult(
                returncode=0,
                output_path=str(output_path),
                log="\n".join(log_lines),
                input_fps=input_fps,
                output_fps=output_fps
            )
        else:
            log(f"❌ RIFE failed with code {proc.returncode}")
            return RifeResult(
                returncode=proc.returncode,
                output_path=None,
                log="\n".join(log_lines)
            )
            
    except Exception as e:
        log(f"❌ RIFE error: {e}")
        import traceback
        log(traceback.format_exc())
        return RifeResult(
            returncode=1,
            output_path=None,
            log="\n".join(log_lines)
        )


def run_video_trim(
    input_path: str,
    output_path: str,
    start_time: str,
    end_time: str,
    on_progress: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Trim video to specified time range.
    
    Args:
        input_path: Input video path
        output_path: Output video path
        start_time: Start time (HH:MM:SS or seconds)
        end_time: End time (HH:MM:SS or seconds)
        on_progress: Progress callback
        
    Returns:
        True if successful
    """
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ss", start_time,
            "-to", end_time,
            "-c", "copy",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and Path(output_path).exists():
            if on_progress:
                on_progress(f"✅ Video trimmed: {start_time} to {end_time}\n")
            return True
        else:
            if on_progress:
                on_progress(f"❌ Trim failed: {result.stderr}\n")
            return False
            
    except Exception as e:
        if on_progress:
            on_progress(f"❌ Trim error: {e}\n")
        return False


def run_video_speed_change(
    input_path: str,
    output_path: str,
    speed_factor: float,
    on_progress: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Change video playback speed.
    
    Args:
        input_path: Input video path
        output_path: Output video path
        speed_factor: Speed multiplier (0.5 = half speed, 2.0 = double speed)
        on_progress: Progress callback
        
    Returns:
        True if successful
    """
    try:
        # Use setpts filter to change speed
        # speed_factor 2.0 = double speed = setpts=0.5*PTS
        pts_factor = 1.0 / speed_factor
        
        # Also adjust audio if present
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter_complex",
            f"[0:v]setpts={pts_factor}*PTS[v];[0:a]atempo={speed_factor}[a]",
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and Path(output_path).exists():
            if on_progress:
                on_progress(f"✅ Speed changed to {speed_factor}x\n")
            return True
        else:
            if on_progress:
                on_progress(f"❌ Speed change failed: {result.stderr}\n")
            return False
            
    except Exception as e:
        if on_progress:
            on_progress(f"❌ Speed change error: {e}\n")
        return False


def run_video_concatenate(
    input_paths: List[str],
    output_path: str,
    on_progress: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Concatenate multiple videos.
    
    Args:
        input_paths: List of input video paths
        output_path: Output video path
        on_progress: Progress callback
        
    Returns:
        True if successful
    """
    concat_file: Optional[Path] = None
    try:
        if not input_paths:
            return False

        normalized_paths = [Path(normalize_path(path)).resolve() for path in input_paths]
        if any(not path.is_file() for path in normalized_paths):
            return False

        # Reuse the chunker's tested ffconcat quoting. Doubling an apostrophe is not
        # valid ffconcat escaping on Windows.
        from .chunking import _write_concat_list

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        concat_file = collision_safe_path(output.with_suffix(".concat.txt"))
        _write_concat_list(concat_file, normalized_paths)
        
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and output.exists() and output.stat().st_size > 1024:
            if on_progress:
                on_progress(f"✅ Concatenated {len(input_paths)} videos\n")
            return True
        else:
            if on_progress:
                on_progress(f"❌ Concatenation failed: {result.stderr}\n")
            return False
            
    except Exception as e:
        if on_progress:
            on_progress(f"❌ Concatenation error: {e}\n")
        return False
    finally:
        if concat_file is not None:
            with suppress(Exception):
                concat_file.unlink(missing_ok=True)

