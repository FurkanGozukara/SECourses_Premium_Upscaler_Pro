"""
Centralized Error Handling and Validation

Provides comprehensive error handling utilities for the upscaler application:
- Input validation with detailed error messages
- Graceful error recovery
- User-friendly error formatting
- Logging and telemetry for debugging
"""

import logging
import traceback
from pathlib import Path
from typing import Optional, Tuple, Any, Callable
from functools import wraps


# Configure logging
logger = logging.getLogger("SECourses_Upscaler")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class UpscalerError(Exception):
    """Base exception for upscaler errors"""
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class ValidationError(UpscalerError):
    """Raised when input validation fails"""
    pass


class ProcessingError(UpscalerError):
    """Raised when processing fails"""
    pass


class ModelError(UpscalerError):
    """Raised when model operations fail"""
    pass


def validate_input_path(path: str, must_exist: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Validate input path.
    
    Args:
        path: Path to validate
        must_exist: Whether path must exist
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not path or not path.strip():
        return False, "❌ Input path is empty"
    
    try:
        p = Path(path.strip())
        
        if must_exist and not p.exists():
            return False, f"❌ Input path does not exist: {path}"
        
        if must_exist and not (p.is_file() or p.is_dir()):
            return False, f"❌ Input path is neither a file nor directory: {path}"
        
        return True, None
        
    except Exception as e:
        return False, f"❌ Invalid path format: {str(e)}"


def validate_output_path(path: str, allow_create: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Validate output path.
    
    Args:
        path: Path to validate
        allow_create: Whether to allow creating new paths
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not path or not path.strip():
        return True, None  # Empty output path is OK (will use defaults)
    
    try:
        p = Path(path.strip())
        
        # Check if parent directory exists
        if not allow_create and not p.parent.exists():
            return False, f"❌ Output directory does not exist: {p.parent}"
        
        # Check if parent is writable
        if p.parent.exists() and not p.parent.is_dir():
            return False, f"❌ Parent path is not a directory: {p.parent}"
        
        return True, None
        
    except Exception as e:
        return False, f"❌ Invalid output path format: {str(e)}"


def validate_cuda_device(device_spec: str) -> Tuple[bool, Optional[str]]:
    """
    Validate CUDA device specification.
    
    Args:
        device_spec: Device specification (e.g., "0", "0,1,2")
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not device_spec or not device_spec.strip():
        return True, None  # Empty is OK (will use defaults)
    
    try:
        # Use NVML-based validation (nvidia-smi) to avoid importing torch in the parent process.
        from .gpu_utils import expand_cuda_device_spec, validate_cuda_device_spec

        expanded = expand_cuda_device_spec(device_spec)
        err = validate_cuda_device_spec(expanded)
        return (err is None), err
    except Exception as e:
        return False, f"❌ Error validating CUDA devices: {str(e)}"


def validate_batch_size(batch_size: int, must_be_4n_plus_1: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Validate batch size.
    
    Args:
        batch_size: Batch size to validate
        must_be_4n_plus_1: Whether batch size must follow 4n+1 formula (for SeedVR2)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if batch_size <= 0:
        return False, f"❌ Batch size must be positive, got: {batch_size}"
    
    if must_be_4n_plus_1:
        if (batch_size - 1) % 4 != 0:
            closest = max(1, ((batch_size // 4) * 4) + 1)
            return False, f"❌ Batch size must be 4n+1 (e.g., 5, 9, 13, 17...). Try: {closest}"
    
    return True, None


def validate_resolution(resolution: int, min_val: int = 256, max_val: int = 8192) -> Tuple[bool, Optional[str]]:
    """
    Validate resolution value.
    
    Args:
        resolution: Resolution to validate
        min_val: Minimum allowed resolution
        max_val: Maximum allowed resolution
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if resolution < min_val:
        return False, f"❌ Resolution too low: {resolution} < {min_val}"
    
    if resolution > max_val:
        return False, f"❌ Resolution too high: {resolution} > {max_val}"
    
    # Recommend multiples of 16 for best performance
    if resolution % 16 != 0:
        recommended = (resolution // 16) * 16
        return True, f"⚠️ Resolution {resolution} is not a multiple of 16. Recommended: {recommended} for optimal performance"
    
    return True, None


def safe_execute(
    func: Callable,
    *args,
    error_prefix: str = "Operation failed",
    default_return: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Tuple[bool, Any, Optional[str]]:
    """
    Safely execute a function with comprehensive error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for function
        error_prefix: Prefix for error messages
        default_return: Value to return on error
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for function
        
    Returns:
        Tuple of (success, result, error_message)
    """
    try:
        result = func(*args, **kwargs)
        return True, result, None
    except ValidationError as e:
        error_msg = f"{error_prefix}: {e.message}"
        if e.details:
            error_msg += f"\n{e.details}"
        if log_errors:
            logger.warning(error_msg)
        return False, default_return, error_msg
    except ProcessingError as e:
        error_msg = f"{error_prefix}: {e.message}"
        if e.details:
            error_msg += f"\n{e.details}"
        if log_errors:
            logger.error(error_msg)
        return False, default_return, error_msg
    except Exception as e:
        error_msg = f"{error_prefix}: {str(e)}"
        if log_errors:
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return False, default_return, error_msg


def with_error_handling(error_prefix: str = "Operation failed", log_errors: bool = True):
    """
    Decorator for adding comprehensive error handling to functions.
    
    Usage:
        @with_error_handling("Processing failed")
        def process_video(path):
            # ... processing code ...
            return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ValidationError as e:
                error_msg = f"{error_prefix}: {e.message}"
                if e.details:
                    error_msg += f"\n{e.details}"
                if log_errors:
                    logger.warning(error_msg)
                raise
            except ProcessingError as e:
                error_msg = f"{error_prefix}: {e.message}"
                if e.details:
                    error_msg += f"\n{e.details}"
                if log_errors:
                    logger.error(error_msg)
                raise
            except Exception as e:
                error_msg = f"{error_prefix}: {str(e)}"
                if log_errors:
                    logger.error(f"{error_msg}\n{traceback.format_exc()}")
                raise ProcessingError(error_msg) from e
        return wrapper
    return decorator


def format_user_error(error: Exception) -> str:
    """
    Format error message for user display.
    
    Args:
        error: Exception to format
        
    Returns:
        User-friendly error message
    """
    if isinstance(error, ValidationError):
        return f"⚠️ Validation Error: {error.message}"
    elif isinstance(error, ProcessingError):
        return f"❌ Processing Error: {error.message}"
    elif isinstance(error, ModelError):
        return f"❌ Model Error: {error.message}"
    else:
        return f"❌ Error: {str(error)}"


def check_ffmpeg_available() -> Tuple[bool, Optional[str]]:
    """
    Check if ffmpeg is available in PATH.
    
    Returns:
        Tuple of (is_available, error_message)
    """
    import shutil
    
    if not shutil.which("ffmpeg"):
        return False, "❌ ffmpeg not found in PATH. Please install ffmpeg and add it to your system PATH."
    
    return True, None


def check_disk_space(path: Path, required_mb: int = 1000) -> Tuple[bool, Optional[str]]:
    """
    Check if sufficient disk space is available.
    
    Args:
        path: Path to check
        required_mb: Required space in MB
        
    Returns:
        Tuple of (is_sufficient, warning_message)
    """
    try:
        import shutil
        
        stat = shutil.disk_usage(path)
        available_mb = stat.free // (1024 * 1024)
        
        if available_mb < required_mb:
            return False, f"⚠️ Low disk space: {available_mb}MB available, {required_mb}MB recommended"
        
        return True, None
        
    except Exception as e:
        return True, f"⚠️ Could not check disk space: {str(e)}"


__all__ = [
    'UpscalerError',
    'ValidationError', 
    'ProcessingError',
    'ModelError',
    'validate_input_path',
    'validate_output_path',
    'validate_cuda_device',
    'validate_batch_size',
    'validate_resolution',
    'safe_execute',
    'with_error_handling',
    'format_user_error',
    'check_ffmpeg_available',
    'check_disk_space',
    'logger',
]

