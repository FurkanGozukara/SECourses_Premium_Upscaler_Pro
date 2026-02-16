"""
FlashVSR+ Runner - Interface for FlashVSR+ Video Super-Resolution

Provides subprocess wrapper for FlashVSR+ CLI (run.py) with:
- Automatic model download from HuggingFace
- Support for video and image sequence inputs
- Tiled processing for memory efficiency
- Color correction and FPS control
- Multiple pipeline modes (tiny, tiny-long, full)
"""

import subprocess
import sys
import time
import shutil
import tempfile
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from contextlib import suppress

from .path_utils import (
    normalize_path,
    collision_safe_path,
    detect_input_type,
    IMAGE_EXTENSIONS,
    get_media_fps,
    resolve_output_location
)
from .command_logger import get_command_logger
from .models.flashvsr_meta import flashvsr_version_to_internal, flashvsr_version_to_ui


@dataclass
class FlashVSRResult:
    """Result of FlashVSR+ processing"""
    returncode: int
    output_path: Optional[str]
    log: str
    input_fps: float = 30.0
    output_fps: float = 30.0


def _normalize_cuda_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("cuda:"):
        text = text.split(":", 1)[1].strip()
    return text


def _resolve_flashvsr_device(device_value: Any) -> tuple[str, Optional[str], Optional[str]]:
    """
    Convert app GPU selection into a FlashVSR-safe runtime tuple:
    - CLI device arg (`-d`)
    - CUDA_VISIBLE_DEVICES value (or None to leave unchanged)
    - Optional log note
    """
    raw = str(device_value or "").strip()
    raw_lower = raw.lower()

    if raw_lower in {"cpu", "none", "off"}:
        return "cpu", "", "[FlashVSR] GPU isolation: CPU mode (CUDA_VISIBLE_DEVICES cleared)"

    if raw_lower in {"", "auto"}:
        return "auto", None, None

    gpu_id = _normalize_cuda_token(raw)
    if gpu_id.isdigit():
        # Restrict to the selected physical GPU and remap to local cuda:0.
        return "cuda:0", gpu_id, f"[FlashVSR] GPU isolation: CUDA_VISIBLE_DEVICES={gpu_id}, device remapped to cuda:0"

    # Unknown format: pass through untouched.
    return raw, None, None


def run_flashvsr(
    settings: Dict[str, Any],
    base_dir: Path,
    on_progress: Optional[Callable[[str], None]] = None,
    cancel_event=None,
    process_handle: Optional[Dict] = None
) -> FlashVSRResult:
    """
    Run FlashVSR+ upscaling.
    
    settings must include:
    - input_path: str (video or image folder)
    - output_path: str (output directory)
    - scale: int (2 or 4)
    - version: str ("1.0"/"1.1" UI, or legacy "10"/"11")
    - mode: str ("tiny", "tiny-long", "full")
    - tiled_vae: bool
    - tiled_dit: bool
    - tile_size: int
    - overlap: int
    - unload_dit: bool
    - color_fix: bool
    - seed: int
    - dtype: str ("fp16" or "bf16")
    - device: str (GPU ID or "auto")
    - fps: int (for image sequences)
    - quality: int (1-10, video quality)
    - attention: str ("sage" or "block")
    
    Returns:
        FlashVSRResult with processing outcome
    """
    start_time = time.time()
    log_lines = []
    cmd: List[str] = []
    
    def log(msg: str):
        log_lines.append(msg)
        if on_progress:
            try:
                on_progress(msg + "\n")
            except Exception:
                # Never let UI/console callback issues fail the actual run.
                pass
    temp_input_dir: Optional[Path] = None
    
    try:
        # Validate input (support preprocessed effective input path)
        original_input_path = normalize_path(settings.get("input_path", ""))
        effective_input_path = normalize_path(settings.get("_effective_input_path") or original_input_path)
        if not effective_input_path or not Path(effective_input_path).exists():
            return FlashVSRResult(returncode=1, output_path=None, log="Invalid input path")

        # FlashVSR CLI expects either a video path or a directory of images.
        # For single-image inputs, wrap the image into a temporary one-frame folder.
        effective_type = detect_input_type(effective_input_path)
        if effective_type == "image":
            try:
                src_img = Path(effective_input_path)
                temp_input_dir = Path(tempfile.mkdtemp(prefix="flashvsr_single_frame_"))
                ext = src_img.suffix.lower()
                if ext not in IMAGE_EXTENSIONS:
                    ext = ".png"
                dst_img = temp_input_dir / f"frame_000001{ext}"
                shutil.copy2(src_img, dst_img)
                effective_input_path = str(temp_input_dir)
                log(f"[FlashVSR] Single image input wrapped as frame sequence: {effective_input_path}")
            except Exception as e:
                return FlashVSRResult(
                    returncode=1,
                    output_path=None,
                    log=f"Failed to prepare single-image input for FlashVSR: {e}",
                )
        
        # Determine output path (naming should follow ORIGINAL input)
        output_override = settings.get("output_override", "")
        explicit_output_file: Optional[Path] = None
        if output_override:
            override_path = Path(normalize_path(output_override))
            # FlashVSR CLI writes into an output FOLDER; support file-path override by
            # running in the parent folder and renaming the produced mp4 after.
            if override_path.suffix:
                explicit_output_file = collision_safe_path(override_path)
                output_folder = str(explicit_output_file.parent)
            else:
                output_folder = str(override_path)
        else:
            # Use default output naming
            output_folder = resolve_output_location(
                input_path=original_input_path,
                output_format="mp4",
                global_output_dir=settings.get("global_output_dir", str(base_dir / "outputs")),
                batch_mode=False,
                original_filename=settings.get("_original_filename"),
            )
            # FlashVSR expects a folder, not a file
            if Path(output_folder).suffix:
                output_folder = str(Path(output_folder).parent)
        
        output_folder_path = Path(output_folder)
        output_folder_path.mkdir(parents=True, exist_ok=True)
        # Track pre-existing outputs so we can reliably locate the output from THIS run.
        pre_existing_mp4 = {p.resolve() for p in output_folder_path.glob("*.mp4")}
        
        # Get settings with defaults
        scale = int(settings.get("scale", 4))
        raw_version = str(settings.get("version", "1.0"))
        version = flashvsr_version_to_internal(raw_version)
        version_ui = flashvsr_version_to_ui(raw_version)
        mode = settings.get("mode", "tiny")
        dtype = settings.get("dtype", "bf16")
        device = settings.get("device", "auto")
        device_arg, visible_gpu, gpu_note = _resolve_flashvsr_device(device)
        fps = int(settings.get("fps", 30))
        quality = int(settings.get("quality", 6))
        attention = settings.get("attention", "sage")
        seed = int(settings.get("seed", 0))

        # FlashVSR+ naming (mirrors FlashVSR_plus/run.py):
        #   FlashVSR_{mode}_{name.split('.')[0]}_{seed}.mp4
        base_name = os.path.basename(original_input_path.rstrip("/\\"))
        base_no_ext = base_name.split(".")[0] if base_name else "FlashVSR"
        expected_output_path = output_folder_path / f"FlashVSR_{mode}_{base_no_ext}_{seed}.mp4"
        
        # Tile settings
        tiled_vae = bool(settings.get("tiled_vae", False))
        tiled_dit = bool(settings.get("tiled_dit", False))
        tile_size = int(settings.get("tile_size", 256))
        overlap = int(settings.get("overlap", 24))
        unload_dit = bool(settings.get("unload_dit", False))
        color_fix = bool(settings.get("color_fix", False))
        
        # Build command
        flashvsr_script = base_dir / "FlashVSR_plus" / "run.py"
        if not flashvsr_script.exists():
            return FlashVSRResult(
                returncode=1,
                output_path=None,
                log=f"FlashVSR+ script not found at {flashvsr_script}"
            )
        
        cmd = [
            sys.executable,
            str(flashvsr_script),
            "-i", effective_input_path,
            "-s", str(scale),
            "-v", version,
            "-m", mode,
            "-t", dtype,
            "-d", device_arg,
            "-f", str(fps),
            "-q", str(quality),
            "-a", attention,
            "--seed", str(seed),
            output_folder  # Positional arg
        ]
        
        # Add flags
        if tiled_vae:
            cmd.append("--tiled-vae")
        if tiled_dit:
            cmd.append("--tiled-dit")
        if tiled_dit or tiled_vae:
            cmd.extend(["--tile-size", str(tile_size)])
            cmd.extend(["--overlap", str(overlap)])
        if unload_dit:
            cmd.append("--unload-dit")
        if color_fix:
            cmd.append("--color-fix")
        
        log(f"Running FlashVSR+ with scale={scale}, mode={mode}, version={version_ui}")
        if original_input_path != effective_input_path:
            log(f"Preprocessed input: {effective_input_path} (original: {original_input_path})")
        log(f"Command: {' '.join(cmd)}")
        
        # Run subprocess with cancellation support
        import platform
        
        # Platform-specific process group creation
        creationflags = 0
        preexec_fn = None
        if platform.system() == "Windows":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            preexec_fn = os.setsid
        
        triton_cache_dir = base_dir / "temp" / "triton_cache"
        with suppress(Exception):
            triton_cache_dir.mkdir(parents=True, exist_ok=True)

        proc_env = {
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        if visible_gpu is not None:
            proc_env["CUDA_VISIBLE_DEVICES"] = visible_gpu
        if gpu_note:
            log(gpu_note)
        proc_env.setdefault("TRITON_CACHE_DIR", str(triton_cache_dir))

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=base_dir,
            env=proc_env,
            creationflags=creationflags,
            preexec_fn=preexec_fn
        )
        
        # Store process handle for cancellation
        if process_handle is not None:
            process_handle["proc"] = proc
        
        # Read output with cancel checking
        output_lines = []
        while True:
            # Check for cancellation
            if cancel_event and cancel_event.is_set():
                log("⏹️ Cancellation requested - terminating FlashVSR+ process")
                try:
                    if platform.system() == "Windows":
                        proc.terminate()
                    else:
                        proc.kill()
                    proc.wait(timeout=5.0)
                except Exception:
                    pass
                return FlashVSRResult(
                    returncode=1,
                    output_path=None,
                    log="\n".join(log_lines + output_lines + ["[Cancelled by user]"])
                )
            
            line = proc.stdout.readline()
            if not line:
                break
            
            line = line.rstrip()
            if line:
                output_lines.append(line)
                log(line)
        
        # Wait for completion
        returncode = proc.wait()
        
        # Collect any remaining output
        remaining = proc.stdout.read()
        if remaining:
            output_lines.append(remaining)
        
        # Find output file reliably (avoid lexicographic "last file" bugs).
        output_path: Optional[str] = None
        if expected_output_path.exists():
            output_path = str(expected_output_path)
        else:
            post_mp4 = {p.resolve() for p in output_folder_path.glob("*.mp4")}
            new_files = list(post_mp4 - pre_existing_mp4)
            if new_files:
                newest = max(new_files, key=lambda p: p.stat().st_mtime)
                output_path = str(newest)
            elif post_mp4:
                newest = max(post_mp4, key=lambda p: p.stat().st_mtime)
                output_path = str(newest)

        if output_path:
            # Optional: rename/move to an explicit file path override.
            if explicit_output_file:
                try:
                    import shutil
                    desired = explicit_output_file
                    desired.parent.mkdir(parents=True, exist_ok=True)
                    if Path(output_path).resolve() != desired.resolve():
                        shutil.move(output_path, desired)
                    output_path = str(desired)
                    log(f"✅ Output renamed to: {output_path}")
                except Exception as exc:
                    log(f"⚠️ Failed to rename output to override path: {exc}")

            log(f"✅ Output saved: {output_path}")
            
            effective_returncode = int(returncode)
            if effective_returncode != 0:
                log(
                    f"FlashVSR+ exited with code {effective_returncode} after producing output; "
                    "treating run as successful."
                )
                effective_returncode = 0

            result = FlashVSRResult(
                returncode=effective_returncode,
                output_path=output_path,
                log="\n".join(log_lines),
                input_fps=get_media_fps(original_input_path) or 30.0,
                output_fps=fps
            )
        else:
            log("❌ No output file generated")
            result = FlashVSRResult(
                returncode=1,
                output_path=None,
                log="\n".join(log_lines)
            )
            
    except Exception as e:
        error_msg = f"FlashVSR+ error: {str(e)}"
        log_lines.append(error_msg)
        result = FlashVSRResult(
            returncode=1,
            output_path=None,
            log="\n".join(log_lines)
        )
    
    finally:
        if temp_input_dir:
            try:
                shutil.rmtree(temp_input_dir, ignore_errors=True)
            except Exception:
                pass
        # Log command to executed_commands folder
        execution_time = time.time() - start_time
        try:
            command_logger = get_command_logger(base_dir.parent / "executed_commands")
            
            command_logger.log_command(
                tab_name="flashvsr",
                command=cmd if cmd else ["flashvsr_run.py", "--input", settings.get("input_path", "unknown")],
                settings=settings,
                returncode=result.returncode if result else -1,
                output_path=result.output_path if result else None,
                error_logs=log_lines[-50:] if result and result.returncode != 0 else None,
                execution_time=execution_time,
                additional_info={
                    "scale": settings.get("scale", "unknown"),
                    "version": settings.get("version", "unknown"),
                    "mode": settings.get("mode", "unknown")
                }
            )
            log("✅ Command logged to executed_commands folder")
        except Exception as e:
            log(f"⚠️ Failed to log command: {e}")
    
    return result


def discover_flashvsr_models(base_dir: Path) -> List[str]:
    """
    Discover available FlashVSR+ models.
    
    Args:
        base_dir: Base directory containing FlashVSR_plus
        
    Returns:
        List of available model versions
    """
    # FlashVSR+ has versioned models
    models_dir = base_dir / "FlashVSR_plus" / "models"
    
    available = []
    
    # Check for downloaded models
    if models_dir.exists():
        for item in models_dir.iterdir():
            if item.is_dir() and not item.name.startswith("_"):
                available.append(item.name)
    
    # Fallback to known versions if nothing found
    if not available:
        available = ["FlashVSR"]  # Default model from HuggingFace
    
    return sorted(available)

