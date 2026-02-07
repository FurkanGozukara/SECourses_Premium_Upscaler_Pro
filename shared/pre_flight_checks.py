"""
Pre-flight Validation System

Comprehensive pre-flight checks before processing to catch issues early:
- Disk space estimation and validation
- GPU VRAM estimation
- Input file validation
- FFmpeg availability
- Model availability
- Settings validation
"""

import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from .path_utils import (
    get_media_dimensions,
    get_media_duration_seconds,
    get_media_fps,
    detect_input_type,
    normalize_path,
)
from .health import (
    is_ffmpeg_available,
    get_cuda_device_count,
    get_cuda_memory_info,
    is_vs_build_tools_available,
)


@dataclass
class PreFlightResult:
    """Result of pre-flight validation"""
    passed: bool
    errors: List[str]
    warnings: List[str]
    info: List[str]
    estimated_vram_mb: int = 0
    estimated_disk_mb: int = 0
    estimated_duration_seconds: float = 0.0


def estimate_vram_required(
    resolution: int,
    batch_size: int,
    model_type: str = "seedvr2",
    model_size: str = "3b",
    compile_enabled: bool = False,
    vae_tiled: bool = False
) -> int:
    """
    Estimate VRAM requirements in MB.
    
    Args:
        resolution: Target resolution (short side)
        batch_size: Processing batch size
        model_type: Type of model (seedvr2, gan, rife)
        model_size: Model size variant (3b, 7b for SeedVR2)
        compile_enabled: torch.compile adds overhead
        vae_tiled: VAE tiling reduces VRAM
        
    Returns:
        Estimated VRAM in MB
    """
    if model_type == "seedvr2":
        # Base model size
        if model_size == "7b":
            base_vram = 8000  # ~8GB for 7B model
        else:
            base_vram = 4000  # ~4GB for 3B model
        
        # Resolution impact (quadratic)
        resolution_factor = (resolution / 1080) ** 2
        resolution_vram = 2000 * resolution_factor
        
        # Batch size impact (linear but with diminishing returns)
        batch_factor = (batch_size / 5) ** 0.7  # Sublinear
        batch_vram = 1000 * batch_factor
        
        # Compile overhead
        compile_vram = 2000 if compile_enabled else 0
        
        # VAE tiling reduces VRAM
        vae_reduction = 0.6 if vae_tiled else 1.0
        
        total_vram = int((base_vram + resolution_vram + batch_vram + compile_vram) * vae_reduction)
        
    elif model_type == "gan":
        # GAN models are lighter
        base_vram = 1500
        
        resolution_factor = (resolution / 1080) ** 2
        resolution_vram = 1000 * resolution_factor
        
        batch_factor = (batch_size / 4) ** 0.5
        batch_vram = 500 * batch_factor
        
        total_vram = int(base_vram + resolution_vram + batch_vram)
        
    elif model_type == "rife":
        # RIFE is relatively light
        base_vram = 2000
        resolution_factor = (resolution / 1080) ** 2
        total_vram = int(base_vram * resolution_factor)
        
    else:
        # Unknown model type
        total_vram = 4000  # Conservative estimate
    
    # Add 1GB safety margin
    return total_vram + 1024


def estimate_disk_space_required(
    input_path: str,
    output_format: str,
    scale_factor: float = 1.0,
    is_chunked: bool = False,
    chunk_count: int = 1,
    is_batch: bool = False,
    batch_count: int = 1,
    compression_quality: int = 18  # CRF value
) -> int:
    """
    Estimate required disk space in MB.
    
    Args:
        input_path: Input file or directory path
        output_format: Output format (mp4, png, etc.)
        scale_factor: Resolution scale factor (e.g., 2.0 for 2x upscale)
        is_chunked: Whether chunking is enabled
        chunk_count: Number of chunks (if chunked)
        is_batch: Batch processing
        batch_count: Number of files in batch
        compression_quality: CRF quality (lower = larger files)
        
    Returns:
        Estimated disk space in MB
    """
    try:
        path = Path(normalize_path(input_path))
        
        if not path.exists():
            return 5000  # Default estimate
        
        # Get input file size
        if path.is_file():
            input_size_mb = path.stat().st_size / (1024 * 1024)
        else:
            # Directory - sum all files
            input_size_mb = sum(
                f.stat().st_size for f in path.rglob("*") if f.is_file()
            ) / (1024 * 1024)
        
        # Upscaling multiplier (resolution squared)
        upscale_multiplier = scale_factor ** 2
        
        # Output size estimate based on format
        if output_format == "png":
            # PNG sequences are large (lossless)
            output_size_mb = input_size_mb * upscale_multiplier * 3.0
        elif output_format == "mp4":
            # MP4 compression factor based on CRF
            # CRF 18 (high quality) ~ 1.5x input size at same resolution
            # With upscaling, it's less than raw pixel increase due to compression
            compression_efficiency = 0.4 + (compression_quality / 51) * 0.6
            output_size_mb = input_size_mb * upscale_multiplier * compression_efficiency
        else:
            # Conservative estimate
            output_size_mb = input_size_mb * upscale_multiplier * 2.0
        
        # Temp space for processing (chunks, frames, etc.)
        if is_chunked:
            # Need space for all chunks + intermediate frames
            temp_space_multiplier = chunk_count * 1.5
        else:
            temp_space_multiplier = 2.0  # Space for frames extraction
        
        temp_size_mb = output_size_mb * temp_space_multiplier
        
        # Batch processing
        if is_batch:
            output_size_mb *= batch_count
            temp_size_mb *= 1.5  # Don't multiply temp by full batch count
        
        # Total estimate
        total_mb = int(output_size_mb + temp_size_mb)
        
        # Add 20% safety margin
        return int(total_mb * 1.2)
        
    except Exception:
        # Fallback estimate
        return 10000  # 10GB default


def check_pre_flight(
    settings: Dict[str, Any],
    model_type: str = "seedvr2",
    on_progress: Optional[callable] = None
) -> PreFlightResult:
    """
    Comprehensive pre-flight validation.
    
    Args:
        settings: Processing settings dictionary
        model_type: Type of model being used
        on_progress: Optional progress callback
        
    Returns:
        PreFlightResult with validation results
    """
    errors = []
    warnings = []
    info = []
    
    # Check FFmpeg
    if not is_ffmpeg_available():
        errors.append("❌ FFmpeg not found in PATH. Install FFmpeg to continue.")
    else:
        info.append("✓ FFmpeg available")
    
    # Validate input path
    input_path = settings.get("input_path", "")
    if not input_path:
        errors.append("❌ No input path provided")
    else:
        input_path = normalize_path(input_path)
        if not Path(input_path).exists():
            errors.append(f"❌ Input path does not exist: {input_path}")
        else:
            info.append(f"✓ Input path valid: {input_path}")
            
            # Detect input type
            input_type = detect_input_type(input_path)
            info.append(f"✓ Input type: {input_type}")
    
    # Check CUDA availability and VRAM
    estimated_vram = 0
    if model_type in ["seedvr2", "gan", "rife"]:
        try:
            import torch
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                info.append(f"✓ CUDA available: {device_count} device(s)")
                
                # Estimate VRAM requirements
                resolution = int(settings.get("resolution", 1080))
                batch_size = int(settings.get("batch_size", 5))
                compile_enabled = bool(settings.get("compile_dit") or settings.get("compile_vae"))
                vae_tiled = bool(settings.get("vae_encode_tiled") or settings.get("vae_decode_tiled"))
                
                model_size = "7b" if "7b" in settings.get("dit_model", "") else "3b"
                
                estimated_vram = estimate_vram_required(
                    resolution, batch_size, model_type, model_size, compile_enabled, vae_tiled
                )
                
                # Check if we have enough VRAM
                cuda_devices = settings.get("cuda_device", "0")
                device_ids = [int(d.strip()) for d in str(cuda_devices).split(",") if d.strip().isdigit()]
                
                if device_ids:
                    try:
                        total_vram = 0
                        for device_id in device_ids:
                            if device_id < device_count:
                                mem_info = get_cuda_memory_info(device_id)
                                if mem_info:
                                    total_vram += mem_info.get("total_mb", 0)
                        
                        if total_vram > 0:
                            if estimated_vram > total_vram:
                                warnings.append(f"⚠️ Estimated VRAM ({estimated_vram}MB) exceeds available ({total_vram}MB). Consider enabling VAE tiling or reducing batch size.")
                            else:
                                free_vram_pct = int(((total_vram - estimated_vram) / total_vram) * 100)
                                info.append(f"✓ VRAM estimate: {estimated_vram}MB / {total_vram}MB available ({free_vram_pct}% free)")
                    except Exception:
                        warnings.append(f"⚠️ Could not check VRAM. Estimated requirement: {estimated_vram}MB")
            else:
                warnings.append("⚠️ CUDA not available - will use CPU (much slower)")
        except Exception:
            warnings.append("⚠️ Could not check GPU availability")
    
    # Check disk space
    estimated_disk = 0
    output_dir = settings.get("output_dir", "outputs")
    try:
        output_path = Path(normalize_path(output_dir)) if output_dir else Path("outputs")
        
        # Estimate disk space required
        if input_path and Path(input_path).exists():
            scale_factor = int(settings.get("resolution", 1080)) / 720.0  # Rough estimate
            output_format = settings.get("output_format", "mp4")
            is_chunked = bool(settings.get("chunk_enable"))
            chunk_count = 5 if is_chunked else 1  # Rough estimate
            is_batch = bool(settings.get("batch_enable"))
            batch_input = settings.get("batch_input_path", "")
            batch_count = 1
            
            if is_batch and batch_input:
                batch_path = Path(normalize_path(batch_input))
                if batch_path.exists() and batch_path.is_dir():
                    from .input_detector import VIDEO_FORMATS, IMAGE_FORMATS
                    batch_files = [
                        f for f in batch_path.iterdir()
                        if f.is_file() and f.suffix.lower() in (VIDEO_FORMATS | IMAGE_FORMATS)
                    ]
                    batch_count = len(batch_files)
            
            estimated_disk = estimate_disk_space_required(
                input_path,
                output_format,
                scale_factor,
                is_chunked,
                chunk_count,
                is_batch,
                batch_count,
                int(settings.get("video_quality", 18))
            )
            
            # Check available disk space
            stat = shutil.disk_usage(output_path)
            available_mb = stat.free / (1024 * 1024)
            
            if estimated_disk > available_mb:
                errors.append(f"❌ Insufficient disk space: need ~{estimated_disk}MB, have {int(available_mb)}MB")
            elif estimated_disk > available_mb * 0.8:
                warnings.append(f"⚠️ Low disk space: need ~{estimated_disk}MB, have {int(available_mb)}MB ({int((available_mb - estimated_disk) / 1024)}GB remaining)")
            else:
                info.append(f"✓ Disk space OK: ~{estimated_disk}MB required, {int(available_mb)}MB available")
    
    except Exception as e:
        warnings.append(f"⚠️ Could not check disk space: {e}")
    
    # Check VS Build Tools on Windows (for torch.compile)
    if settings.get("compile_dit") or settings.get("compile_vae"):
        import platform
        if platform.system() == "Windows":
            if not is_vs_build_tools_available():
                warnings.append("⚠️ VS Build Tools not detected. torch.compile may fail. Install 'Desktop development with C++' workload.")
            else:
                info.append("✓ VS Build Tools detected")
    
    # Validate resolution settings
    resolution = settings.get("resolution", 1080)
    max_resolution = settings.get("max_resolution", 0)
    
    if resolution % 16 != 0:
        warnings.append(f"⚠️ Resolution {resolution} not multiple of 16. Will be adjusted to {(resolution // 16) * 16}")
    
    if max_resolution > 0 and resolution > max_resolution:
        warnings.append(f"⚠️ Target resolution ({resolution}) exceeds max resolution ({max_resolution})")
    
    # Model-specific validation
    if model_type == "seedvr2":
        batch_size = int(settings.get("batch_size", 5))
        if batch_size % 4 != 1:
            warnings.append(f"⚠️ SeedVR2 batch size must be 4n+1. {batch_size} will be adjusted to {(batch_size // 4) * 4 + 1}")
        
        # VAE tiling validation
        if settings.get("vae_encode_tiled"):
            tile_size = int(settings.get("vae_encode_tile_size", 1024))
            overlap = int(settings.get("vae_encode_tile_overlap", 128))
            if overlap >= tile_size:
                errors.append(f"❌ VAE encode tile overlap ({overlap}) must be less than tile size ({tile_size})")
        
        if settings.get("vae_decode_tiled"):
            tile_size = int(settings.get("vae_decode_tile_size", 1024))
            overlap = int(settings.get("vae_decode_tile_overlap", 128))
            if overlap >= tile_size:
                errors.append(f"❌ VAE decode tile overlap ({overlap}) must be less than tile size ({tile_size})")
        
        # BlockSwap validation
        blocks_to_swap = int(settings.get("blocks_to_swap", 0))
        swap_io = bool(settings.get("swap_io_components"))
        dit_offload = str(settings.get("dit_offload_device", "none")).lower()
        
        if (blocks_to_swap > 0 or swap_io) and dit_offload in ("none", ""):
            errors.append("❌ BlockSwap requires dit_offload_device to be set (recommend 'cpu')")
        
        # Multi-GPU validation for cache options
        cuda_device = str(settings.get("cuda_device", "0"))
        device_list = [d.strip() for d in cuda_device.replace(" ", "").split(",") if d.strip()]
        
        if len(device_list) > 1:
            if settings.get("cache_dit") or settings.get("cache_vae"):
                warnings.append("⚠️ CUDA graph caching (cache_dit/cache_vae) requires single GPU. Will be disabled.")
    
    # Estimate duration
    estimated_duration = 0.0
    if input_path and Path(input_path).exists():
        estimated_duration = get_media_duration_seconds(input_path) or 0.0
        
        # Adjust for RIFE time modification if applicable
        if model_type == "rife":
            speed_factor = float(settings.get("speed_factor", 1.0))
            if speed_factor > 0:
                estimated_duration /= speed_factor

    # Build result
    result = PreFlightResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        info=info,
        estimated_vram_mb=estimated_vram,
        estimated_disk_mb=estimated_disk,
        estimated_duration_seconds=estimated_duration
    )
    
    return result


def format_pre_flight_report(result: PreFlightResult) -> str:
    """Format pre-flight result as user-friendly string"""
    lines = []
    
    if result.passed:
        lines.append("✅ **Pre-flight Check PASSED**\n")
    else:
        lines.append("❌ **Pre-flight Check FAILED**\n")
    
    if result.errors:
        lines.append("**Errors:**")
        for error in result.errors:
            lines.append(f"  {error}")
        lines.append("")
    
    if result.warnings:
        lines.append("**Warnings:**")
        for warning in result.warnings:
            lines.append(f"  {warning}")
        lines.append("")
    
    if result.info:
        lines.append("**Info:**")
        for info_item in result.info:
            lines.append(f"  {info_item}")
        lines.append("")
    
    # Resource estimates
    if result.estimated_vram_mb > 0:
        lines.append(f"**Estimated VRAM:** ~{result.estimated_vram_mb}MB ({result.estimated_vram_mb / 1024:.1f}GB)")
    
    if result.estimated_disk_mb > 0:
        lines.append(f"**Estimated Disk:** ~{result.estimated_disk_mb}MB ({result.estimated_disk_mb / 1024:.1f}GB)")
    
    return "\n".join(lines)


def quick_validate_settings(settings: Dict[str, Any], model_type: str = "seedvr2") -> Tuple[bool, str]:
    """
    Quick validation of settings without deep checks.
    
    Returns:
        (is_valid, message)
    """
    # Input validation
    if not settings.get("input_path"):
        return False, "No input path provided"
    
    input_path = normalize_path(settings["input_path"])
    if not Path(input_path).exists():
        return False, f"Input path does not exist: {input_path}"
    
    # Model-specific quick checks
    if model_type == "seedvr2":
        batch_size = int(settings.get("batch_size", 5))
        if batch_size < 1:
            return False, "Batch size must be at least 1"
        
        resolution = int(settings.get("resolution", 1080))
        if resolution < 256:
            return False, "Resolution too low (minimum 256px)"
    
    return True, "Settings valid"

