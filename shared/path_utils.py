import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Supported extensions match SeedVR2 CLI
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def normalize_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    expanded = os.path.expanduser(os.path.expandvars(str(path)))
    return str(Path(expanded).resolve())


def detect_input_type(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    if p.is_dir():
        return "directory"
    ext = p.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    return "unknown"


def get_media_dimensions(path: str) -> Optional[Tuple[int, int]]:
    """
    Return (width, height) for an image or video using ffprobe first, Pillow fallback.
    """
    try:
        p = Path(path)
        if p.exists() and p.is_dir():
            # Directory input (frame sequence): pick a representative file.
            # Prefer images, then videos.
            for ext_set in (IMAGE_EXTENSIONS, VIDEO_EXTENSIONS):
                items = [x for x in sorted(p.iterdir()) if x.is_file() and x.suffix.lower() in ext_set]
                if items:
                    return get_media_dimensions(str(items[0]))
            return None
    except Exception:
        # Fall through to file-based probes
        pass

    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parts = proc.stdout.strip().split(",")
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
    except Exception:
        pass

    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            return im.width, im.height
    except Exception:
        return None


def get_media_duration_seconds(path: str) -> Optional[float]:
    """
    Return duration in seconds for a video using ffprobe.
    """
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return float(proc.stdout.strip())
    except Exception:
        return None
    return None


def get_media_fps(path: str) -> Optional[float]:
    """Return frames per second for a video using ffprobe."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            rate = proc.stdout.strip()
            if "/" in rate:
                num, den = rate.split("/")
                return float(num) / float(den)
            return float(rate)
    except Exception:
        return None
    return None

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_default_output_dir(base_dir: Path, global_settings: dict) -> Path:
    # Prefer saved global setting; fallback to ./outputs
    override = global_settings.get("output_dir")
    if override:
        return Path(normalize_path(override))
    return Path(base_dir / "outputs").resolve()


def get_default_temp_dir(base_dir: Path, global_settings: dict) -> Path:
    # Use launcher-provided TEMP/TMP if present; otherwise fallback to ./temp
    override = global_settings.get("temp_dir") or os.environ.get("TEMP") or os.environ.get("TMP")
    if override:
        return Path(normalize_path(override))
    return Path(base_dir / "temp").resolve()


def collision_safe_path(path: Path) -> Path:
    """
    Append numeric suffixes to avoid overwriting existing files or folders.
    
    Handles special cases:
    - file.mp4 -> file_0001.mp4 if exists
    - file_upscaled.mp4 -> file_upscaled_0001.mp4 if exists
    - file_upscaled_0001.mp4 -> file_upscaled_0002.mp4 if exists
    """
    if not path.exists():
        return path
    
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    
    # Pattern to match existing _NNNN suffix
    pattern = re.compile(r"^(.+?)_(\d{4})$")
    match = pattern.match(stem)
    
    if match:
        # Already has numeric suffix, extract base and increment
        base_stem = match.group(1)
        counter = int(match.group(2)) + 1
    else:
        # No numeric suffix yet, start from 0001
        base_stem = stem
        counter = 1
    
    # Find next available number
    max_attempts = 10000  # Prevent infinite loop
    for attempt in range(max_attempts):
        candidate = parent / f"{base_stem}_{counter:04d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1
    
    # If we somehow reach here, add timestamp
    import time
    timestamp = int(time.time())
    return parent / f"{base_stem}_{timestamp}{suffix}"


def collision_safe_dir(path: Path) -> Path:
    """
    Collision-safe directory naming with _0001 suffix.
    
    Similar to collision_safe_path but for directories.
    """
    if not path.exists():
        return path
    
    base = path.name
    parent = path.parent
    
    # Pattern to match existing _NNNN suffix
    pattern = re.compile(r"^(.+?)_(\d{4})$")
    match = pattern.match(base)
    
    if match:
        base_name = match.group(1)
        counter = int(match.group(2)) + 1
    else:
        base_name = base
        counter = 1
    
    # Find next available directory number
    max_attempts = 10000
    for attempt in range(max_attempts):
        candidate = parent / f"{base_name}_{counter:04d}"
        if not candidate.exists():
            return candidate
        counter += 1
    
    # Fallback with timestamp
    import time
    timestamp = int(time.time())
    return parent / f"{base_name}_{timestamp}"


def rife_output_path(
    input_path: str,
    png_output: bool,
    override: Optional[str],
    global_output_dir: Optional[str] = None,
    png_padding: Optional[int] = None,
    png_keep_basename: bool = False,
) -> Path:
    """
    Collision-safe output for RIFE, reusing the shared SeedVR2-style helper:
    - If override is provided, use it (file or dir) as-is.
    - Otherwise mirror generate_output_path semantics with optional global override.
    """
    if override:
        target = Path(normalize_path(override))
        if target.suffix == "":
            if png_output:
                ensure_dir(target)
                return collision_safe_path(target)
            target = target.with_suffix(".mp4")
        ensure_dir(target.parent)
        return collision_safe_path(target)

    output_fmt = "png" if png_output else "mp4"
    target = resolve_output_location(
        input_path=input_path,
        output_format=output_fmt,
        global_output_dir=global_output_dir,
        batch_mode=False,
        png_padding=png_padding,
        png_keep_basename=png_keep_basename,
    )
    # resolve_output_location may return file or dir; ensure collision safety
    target_path = Path(target)
    if target_path.suffix.lower() in (".mp4", ".png"):
        return collision_safe_path(target_path)
    return collision_safe_dir(target_path)


def write_png_metadata(dir_path: Path, payload: dict):
    """
    Write metadata JSON into a PNG sequence directory.
    """
    ensure_dir(dir_path)
    meta_path = dir_path / "run_metadata.json"
    try:
        import json
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def get_png_frame_pattern(output_dir: Path, base_name: str, padding: int = 6) -> str:
    """
    Get PNG frame naming pattern for video->PNG exports.
    
    Args:
        output_dir: Directory where frames will be saved
        base_name: Base name for frames (usually input video stem)
        padding: Number of digits for frame numbering (default 6 to match SeedVR2 CLI)
    
    Returns:
        Pattern string like "frame_%06d.png" for use with ffmpeg or manual frame saving
    """
    # Match SeedVR2 CLI behavior: {base_name}_{index:0Nd}.png
    return f"{base_name}_%0{padding}d.png"


def read_png_settings(output_dir: Path) -> Dict[str, Any]:
    """
    Read PNG settings metadata from a PNG sequence directory.
    
    Returns:
        Dict with 'padding', 'keep_basename', 'base_name' or defaults
    """
    metadata_path = output_dir / ".png_settings.json"
    if metadata_path.exists():
        try:
            import json
            with metadata_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    
    # Return defaults matching SeedVR2 CLI
    return {
        "padding": 6,
        "keep_basename": True,
        "base_name": "frame"
    }


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to be filesystem-safe while preserving readability.
    
    Removes/replaces problematic characters but keeps spaces, hyphens, underscores.
    Handles Windows reserved names (CON, PRN, AUX, etc.).
    
    Args:
        filename: Original filename (with or without extension)
        
    Returns:
        Sanitized filename safe for Windows and Linux
    """
    import re
    
    # Remove path separators and null bytes
    filename = filename.replace('/', '_').replace('\\', '_').replace('\0', '')
    
    # Remove or replace problematic characters
    # Keep: letters, digits, spaces, hyphens, underscores, dots, parentheses
    filename = re.sub(r'[<>:"|?*]', '_', filename)
    
    # Remove leading/trailing dots and spaces (Windows restriction)
    filename = filename.strip('. ')
    
    # Handle Windows reserved names (CON, PRN, AUX, NUL, COM1-9, LPT1-9)
    windows_reserved = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    name_upper = filename.upper()
    # Check both with and without extension
    base_name = name_upper.split('.')[0] if '.' in name_upper else name_upper
    if base_name in windows_reserved:
        filename = f"file_{filename}"
    
    # Ensure not empty
    if not filename:
        filename = "unnamed"
    
    # Limit length (Windows MAX_PATH is 260, leave room for directory and extension)
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename


def generate_output_path(
    input_path: str,
    output_format: str,
    output_dir: Optional[str] = None,
    input_type: Optional[str] = None,
    from_directory: bool = False,
    png_padding: Optional[int] = None,
    png_keep_basename: bool = False,
    original_filename: Optional[str] = None,
) -> Path:
    """
    Mirror SeedVR2 CLI generate_output_path behavior, then apply collision safety.

    We keep the `_upscaled` suffix for single runs even when an explicit output
    directory/global override is provided, but avoid the suffix for directory
    (batch) mode to preserve original names inside the batch folder.
    
    For PNG sequences (video -> PNG frames):
    - Creates directory with name pattern: {input_name}_upscaled/
    - Individual frames are named by the CLI: {base_name}_{frame_num:0Nd}.png
    - png_padding controls N (default 6 to match SeedVR2 CLI)
    - SeedVR2 CLI handles actual frame file creation with padding
    
    Args:
        original_filename: Original user filename from upload (e.g., FileData.orig_name)
                          If provided, this is used for output naming instead of temp path
    """
    input_path_obj = Path(input_path)
    
    # Use original filename if provided (from Gradio upload), otherwise use input path stem
    if original_filename:
        # Sanitize the original filename
        sanitized = sanitize_filename(original_filename)
        # Remove extension to get stem
        input_name = Path(sanitized).stem
    else:
        input_name = input_path_obj.stem

    if input_type is None:
        input_type = detect_input_type(input_path)

    # Determine base directory; suffix is applied for single runs.
    if output_dir:
        base_dir = Path(normalize_path(output_dir))
    elif from_directory:
        original_dir = input_path_obj.parent
        base_dir = original_dir.parent / f"{original_dir.name}_upscaled"
    else:
        base_dir = input_path_obj.parent

    # FIXED: Match CLI behavior - no suffix when output_dir is provided OR in batch mode
    # CLI logic: add_suffix = False when output_dir is set OR from_directory is True
    add_suffix = not output_dir and not from_directory
    file_suffix = "_upscaled" if add_suffix else ""
    if png_keep_basename and output_format == "png":
        file_suffix = ""

    if output_format == "png":
        if input_type == "image":
            # Single image -> single PNG file
            target = base_dir / f"{input_name}{file_suffix}.png"
            ensure_dir(target.parent)
            return collision_safe_path(target)
        else:
            # Video -> PNG sequence directory
            # NOTE: png_padding is stored as metadata in the directory for CLI usage
            # The actual frame files are created by SeedVR2 CLI with pattern: {base}_{idx:0Nd}.png
            # where N comes from png_padding (default 6 to match CLI hardcoded value)
            target_dir = base_dir / f"{input_name}{file_suffix}"
            ensure_dir(target_dir.parent)
            target_dir = collision_safe_dir(target_dir)
            ensure_dir(target_dir)
            
            # Store PNG settings as metadata for CLI/processor to use
            # This allows frame extraction utilities to respect user's padding preference
            if png_padding is not None:
                try:
                    metadata_path = target_dir / ".png_settings.json"
                    import json
                    with metadata_path.open("w", encoding="utf-8") as f:
                        json.dump({
                            "padding": png_padding,
                            "keep_basename": png_keep_basename,
                            "base_name": input_name
                        }, f, indent=2)
                except Exception:
                    pass  # Non-critical metadata write
            
            return target_dir
    else:
        target = base_dir / f"{input_name}{file_suffix}.mp4"
        ensure_dir(target.parent)
        return collision_safe_path(target)


def resolve_output_location(
    input_path: str,
    output_format: str,
    global_output_dir: Optional[str],
    batch_mode: bool,
    png_padding: Optional[int] = None,
    png_keep_basename: bool = False,
    original_filename: Optional[str] = None,
) -> Path:
    """
    Resolve where outputs should go respecting CLI semantics first, then applying
    global overrides when provided.
    
    Args:
        original_filename: Original user filename from Gradio upload (preserves user's filename)
    """
    target_dir = global_output_dir if global_output_dir else None
    return generate_output_path(
        input_path=input_path,
        output_format=output_format,
        output_dir=target_dir,
        input_type=detect_input_type(input_path),
        from_directory=batch_mode,
        png_padding=png_padding,
        png_keep_basename=png_keep_basename,
        original_filename=original_filename,
    )


def emit_metadata(path: Path, payload: dict):
    """
    Append metadata JSON to run_metadata.json (maintains history of all runs).
    Each run is stored as an entry in a JSON array, allowing users to find
    generated file names and track processing history.
    """
    target_dir = path if path.is_dir() else path.parent
    ensure_dir(target_dir)
    meta_path = target_dir / "run_metadata.json"
    
    import json
    
    # Load existing entries if file exists
    existing_entries = []
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    existing_entries = json.loads(content)
                    # Ensure it's a list
                    if not isinstance(existing_entries, list):
                        existing_entries = [existing_entries]
        except (json.JSONDecodeError, Exception):
            # If file is corrupted or not valid JSON, start fresh
            existing_entries = []
    
    # Append new entry
    existing_entries.append(payload)
    
    # Write back all entries
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(existing_entries, f, indent=2)
    
    return meta_path


def ffmpeg_set_fps(input_path: Path, fps: float) -> Path:
    """
    Re-mux video with a different fps (no re-encode). Falls back silently if ffmpeg missing.
    """
    if fps <= 0:
        return input_path
    if shutil.which("ffmpeg") is None:
        return input_path
    out = collision_safe_path(input_path.with_name(f"{input_path.stem}_fps{input_path.suffix}"))
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0",
        "-c",
        "copy",
        "-r",
        str(fps),
        str(out),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode == 0 and out.exists():
        return out
    return input_path


def is_writable(path: Path) -> bool:
    try:
        ensure_dir(path)
        test_file = path / ".write_test"
        with test_file.open("w") as f:
            f.write("ok")
        test_file.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def get_disk_free_gb(path: Path) -> float:
    try:
        usage = shutil.disk_usage(path)
        return round(usage.free / (1024 ** 3), 2)
    except Exception:
        return 0.0


