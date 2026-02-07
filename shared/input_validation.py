"""
Input Validation System - Pre-Processing Checks

Validates user inputs before processing to prevent:
- Memory exhaustion from huge files
- Disk overflow from insufficient space
- Unrealistic resolution requests
- Invalid parameter combinations

All validation functions return (is_valid, error_message)
"""

import os
from pathlib import Path
from typing import Tuple, Optional

from .path_utils import get_media_dimensions, get_media_duration_seconds


# Configuration constants
MAX_FILE_SIZE_GB = 20  # Block files larger than this
WARN_FILE_SIZE_GB = 5  # Warn files larger than this
MAX_RESOLUTION = 8192  # Maximum supported resolution
WARN_RESOLUTION = 4096  # Warn above this resolution
MIN_DISK_SPACE_GB = 10  # Minimum free disk space required


def validate_file_size(
    file_path: str,
    max_size_gb: float = MAX_FILE_SIZE_GB,
    warn_size_gb: float = WARN_FILE_SIZE_GB
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate input file size.
    
    Args:
        file_path: Path to file
        max_size_gb: Maximum allowed size in GB
        warn_size_gb: Size threshold for warnings in GB
        
    Returns:
        (is_valid, error_message, warning_message)
        - is_valid: True if file size acceptable
        - error_message: Blocking error if too large
        - warning_message: Warning if large but acceptable
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return False, f"File not found: {file_path}", None
        
        if path.is_dir():
            # For directories, estimate total size
            total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        else:
            total_size = path.stat().st_size
        
        size_gb = total_size / (1024**3)
        
        if size_gb > max_size_gb:
            return False, f"File too large: {size_gb:.1f}GB (max: {max_size_gb}GB). Split into smaller chunks or use scene-based chunking.", None
        elif size_gb > warn_size_gb:
            return True, None, f"⚠️ Large file detected: {size_gb:.1f}GB. Processing may take significant time and disk space. Ensure sufficient VRAM and storage."
        else:
            return True, None, None
            
    except Exception as e:
        return True, None, f"Warning: Could not check file size: {e}"


def validate_resolution(
    target_resolution: int,
    max_resolution: int = MAX_RESOLUTION,
    warn_resolution: int = WARN_RESOLUTION
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate target resolution request.
    
    Args:
        target_resolution: Requested resolution
        max_resolution: Maximum allowed resolution
        warn_resolution: Warning threshold
        
    Returns:
        (is_valid, error_message, warning_message)
    """
    if target_resolution > max_resolution:
        return False, f"Resolution too high: {target_resolution}px (max: {max_resolution}px). Reduce target resolution or max_resolution setting.", None
    elif target_resolution > warn_resolution:
        return True, None, f"⚠️ High resolution requested: {target_resolution}px. Ensure adequate VRAM (16GB+ recommended for 4K+). Processing will be slow."
    else:
        return True, None, None


def validate_disk_space(
    output_dir: str,
    required_space_gb: Optional[float] = None,
    min_free_gb: float = MIN_DISK_SPACE_GB
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate available disk space.
    
    Args:
        output_dir: Output directory path
        required_space_gb: Estimated space needed (None = use minimum)
        min_free_gb: Minimum free space required
        
    Returns:
        (is_valid, error_message, warning_message)
    """
    try:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        # Get disk usage
        if os.name == 'nt':  # Windows
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(str(path)),
                None,
                None,
                ctypes.pointer(free_bytes)
            )
            free_gb = free_bytes.value / (1024**3)
        else:  # Unix/Linux/Mac
            stat = os.statvfs(str(path))
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        
        required = required_space_gb if required_space_gb else min_free_gb
        
        if free_gb < required:
            return False, f"Insufficient disk space: {free_gb:.1f}GB free, {required:.1f}GB required. Free up space before processing.", None
        elif free_gb < required * 2:
            return True, None, f"⚠️ Limited disk space: {free_gb:.1f}GB free. Recommended: {required * 2:.1f}GB+ for safety. Monitor space during processing."
        else:
            return True, None, None
            
    except Exception as e:
        return True, None, f"Warning: Could not check disk space: {e}"


def estimate_output_size(
    input_path: str,
    scale_factor: int = 4,
    output_format: str = "mp4"
) -> float:
    """
    Estimate output file size in GB.
    
    Args:
        input_path: Input file path
        scale_factor: Upscale factor (2x, 4x, etc.)
        output_format: Output format (mp4, png, etc.)
        
    Returns:
        Estimated size in GB
    """
    try:
        path = Path(input_path)
        
        if not path.exists():
            return 0.0
        
        # Get input size
        if path.is_dir():
            input_size_gb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024**3)
        else:
            input_size_gb = path.stat().st_size / (1024**3)
        
        # Estimate based on format and scale
        if output_format == "png":
            # PNG sequences are HUGE - roughly linear with pixel count
            estimate_gb = input_size_gb * (scale_factor ** 2) * 3  # Conservative estimate
        else:
            # Video compression helps, but still scales with resolution
            estimate_gb = input_size_gb * (scale_factor ** 2) * 0.5  # Compressed estimate
        
        # Add temp space estimate (1.5x final output)
        total_estimate = estimate_gb * 2.5
        
        return total_estimate
        
    except Exception:
        # Conservative fallback
        return 10.0


def validate_input_comprehensive(
    input_path: str,
    target_resolution: int,
    output_dir: str,
    scale_factor: int = 4,
    output_format: str = "mp4"
) -> Tuple[bool, str]:
    """
    Comprehensive input validation combining all checks.
    
    Args:
        input_path: Input file path
        target_resolution: Target resolution
        output_dir: Output directory
        scale_factor: Upscale factor
        output_format: Output format
        
    Returns:
        (is_valid, message) - message includes errors and warnings
    """
    messages = []
    blocking_errors = []
    
    # Check file size
    size_valid, size_error, size_warn = validate_file_size(input_path)
    if not size_valid:
        blocking_errors.append(size_error)
    elif size_warn:
        messages.append(size_warn)
    
    # Check resolution
    res_valid, res_error, res_warn = validate_resolution(target_resolution)
    if not res_valid:
        blocking_errors.append(res_error)
    elif res_warn:
        messages.append(res_warn)
    
    # Estimate output size
    estimated_gb = estimate_output_size(input_path, scale_factor, output_format)
    
    # Check disk space
    disk_valid, disk_error, disk_warn = validate_disk_space(output_dir, estimated_gb)
    if not disk_valid:
        blocking_errors.append(disk_error)
    elif disk_warn:
        messages.append(disk_warn)
    else:
        messages.append(f"ℹ️ Estimated output size: ~{estimated_gb:.1f}GB (includes temp files)")
    
    # Build final message
    if blocking_errors:
        full_message = "❌ **Validation Failed:**\n\n" + "\n\n".join(blocking_errors)
        if messages:
            full_message += "\n\n**Additional Info:**\n" + "\n".join(messages)
        return False, full_message
    else:
        if messages:
            full_message = "✅ **Validation Passed**\n\n" + "\n".join(messages)
        else:
            full_message = "✅ All validation checks passed"
        return True, full_message


def validate_settings_compatibility(
    settings: dict,
    model_type: str = "seedvr2"
) -> Tuple[bool, Optional[str]]:
    """
    Validate that settings are compatible with each other.
    
    Args:
        settings: Settings dictionary
        model_type: Type of model (seedvr2, gan, rife, flashvsr)
        
    Returns:
        (is_valid, error_message)
    """
    errors = []
    
    if model_type == "seedvr2":
        # BlockSwap requires offload device
        if settings.get("blocks_to_swap", 0) > 0 or settings.get("swap_io_components"):
            if str(settings.get("dit_offload_device", "none")).lower() in ("none", ""):
                errors.append("BlockSwap requires dit_offload_device to be set (typically 'cpu')")
        
        # Cache requires single GPU
        cuda_devices = str(settings.get("cuda_device", "0")).split(",")
        if len([d.strip() for d in cuda_devices if d.strip()]) > 1:
            if settings.get("cache_dit") or settings.get("cache_vae"):
                errors.append("DiT/VAE caching only works with single GPU. Use one device or disable caching.")
        
        # VAE tile overlap validation
        if settings.get("vae_encode_tiled"):
            if settings.get("vae_encode_tile_overlap", 0) >= settings.get("vae_encode_tile_size", 1024):
                errors.append(f"VAE encode overlap ({settings.get('vae_encode_tile_overlap')}) must be < tile size ({settings.get('vae_encode_tile_size')})")
        
        if settings.get("vae_decode_tiled"):
            if settings.get("vae_decode_tile_overlap", 0) >= settings.get("vae_decode_tile_size", 1024):
                errors.append(f"VAE decode overlap ({settings.get('vae_decode_tile_overlap')}) must be < tile size ({settings.get('vae_decode_tile_size')})")
    
    elif model_type == "flashvsr":
        # Tile overlap validation
        if settings.get("tiled_vae") or settings.get("tiled_dit"):
            if settings.get("overlap", 0) >= settings.get("tile_size", 256):
                errors.append(f"Tile overlap ({settings.get('overlap')}) must be < tile size ({settings.get('tile_size')})")
    
    if errors:
        return False, "❌ **Settings Incompatible:**\n\n" + "\n".join(f"• {e}" for e in errors)
    else:
        return True, None

