"""
UI Input Validators

Provides client-side and server-side validation for user inputs
to catch errors before processing starts.

Features:
- Batch size validation (4n+1 formula for SeedVR2)
- Resolution validation (multiple of 16)
- GPU device validation
- Path validation
- FPS validation
"""

from typing import Tuple, Optional
import re


def validate_batch_size_seedvr2(batch_size: int) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Validate and correct batch size for SeedVR2 (must be 4n+1).
    
    Args:
        batch_size: User-entered batch size
        
    Returns:
        Tuple of (is_valid, error_message, corrected_value)
    """
    try:
        bs = int(batch_size)
        
        if bs < 1:
            return False, "Batch size must be at least 1", 5
        
        if bs > 201:
            return False, "Batch size too large (max 201)", 201
        
        # Check if it's 4n+1
        if (bs - 1) % 4 != 0:
            # Find nearest valid value
            corrected = ((bs - 1) // 4) * 4 + 1
            if corrected < 1:
                corrected = 5
            
            return (
                False,
                f"⚠️ Batch size must be 4n+1 (5, 9, 13, 17...). Corrected to {corrected}",
                corrected
            )
        
        return True, None, bs
        
    except (ValueError, TypeError):
        return False, "Invalid batch size (must be a number)", 5


def validate_resolution(resolution: int, must_be_multiple_of: int = 16) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Validate resolution is a multiple of required value.
    
    Args:
        resolution: User-entered resolution
        must_be_multiple_of: Required multiple (default 16 for SeedVR2)
        
    Returns:
        Tuple of (is_valid, error_message, corrected_value)
    """
    try:
        res = int(resolution)
        
        if res < 256:
            return False, "Resolution too small (min 256)", 256
        
        if res > 8192:
            return False, "Resolution too large (max 8192)", 4096
        
        if res % must_be_multiple_of != 0:
            corrected = (res // must_be_multiple_of) * must_be_multiple_of
            return (
                False,
                f"⚠️ Resolution must be multiple of {must_be_multiple_of}. Corrected to {corrected}",
                corrected
            )
        
        return True, None, res
        
    except (ValueError, TypeError):
        return False, "Invalid resolution (must be a number)", 1080


def validate_gpu_device(device_string: str, max_devices: int = None) -> Tuple[bool, Optional[str]]:
    """
    Validate GPU device string format.
    
    Args:
        device_string: GPU device string (e.g., "0" or "0,1,2")
        max_devices: Maximum number of available devices (if known)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not device_string or device_string.strip() == "":
        return True, None  # Empty is valid (use default)
    
    # Remove whitespace
    device_string = device_string.strip()
    
    # Check for valid characters (digits and commas only)
    if not re.match(r'^[\d,\s]+$', device_string):
        return False, "GPU device must contain only digits and commas (e.g., '0' or '0,1,2')"
    
    # Parse device IDs
    try:
        device_ids = [int(x.strip()) for x in device_string.split(',') if x.strip()]
    except ValueError:
        return False, "Invalid GPU device format"
    
    # Check for negative IDs
    if any(d < 0 for d in device_ids):
        return False, "GPU device IDs must be non-negative"
    
    # Check against max if provided
    if max_devices is not None:
        invalid = [d for d in device_ids if d >= max_devices]
        if invalid:
            return (
                False,
                f"GPU device IDs {invalid} exceed available devices (0-{max_devices-1})"
            )
    
    # Check for duplicates
    if len(device_ids) != len(set(device_ids)):
        return False, "Duplicate GPU device IDs detected"
    
    return True, None


def validate_fps(fps: float) -> Tuple[bool, Optional[str]]:
    """
    Validate FPS value.
    
    Args:
        fps: Frames per second value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        fps_val = float(fps)
        
        if fps_val <= 0:
            return False, "FPS must be positive"
        
        if fps_val > 240:
            return False, "FPS too high (max 240)"
        
        return True, None
        
    except (ValueError, TypeError):
        return False, "Invalid FPS (must be a number)"


def validate_tile_overlap(tile_size: int, overlap: int) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Validate tile overlap is less than tile size.
    
    Args:
        tile_size: Tile size in pixels
        overlap: Overlap in pixels
        
    Returns:
        Tuple of (is_valid, error_message, corrected_overlap)
    """
    try:
        ts = int(tile_size)
        ov = int(overlap)
        
        if ts <= 0:
            return True, None, ov  # Tiling disabled
        
        if ov < 0:
            return False, "Overlap cannot be negative", 0
        
        if ov >= ts:
            corrected = max(0, ts - 1)
            return (
                False,
                f"⚠️ Overlap must be less than tile size. Corrected to {corrected}",
                corrected
            )
        
        return True, None, ov
        
    except (ValueError, TypeError):
        return False, "Invalid tile/overlap values", 0


def create_validation_callback(validator_func, update_component_func=None):
    """
    Create a Gradio callback for validation that updates the UI.
    
    Args:
        validator_func: Validation function that returns (is_valid, message, corrected_value)
        update_component_func: Optional function to get Gradio update for the component
        
    Returns:
        Callback function suitable for Gradio .change() or .submit()
    """
    def callback(value):
        is_valid, message, corrected = validator_func(value)
        
        if not is_valid and corrected is not None:
            # Return corrected value and warning message
            import gradio as gr
            if update_component_func:
                return (
                    update_component_func(corrected),
                    gr.update(value=f"<span style='color: orange;'>{message}</span>", visible=True)
                )
            else:
                return (
                    corrected,
                    gr.update(value=f"<span style='color: orange;'>{message}</span>", visible=True)
                )
        elif not is_valid:
            # Return error message
            import gradio as gr
            return (
                value,
                gr.update(value=f"<span style='color: red;'>{message}</span>", visible=True)
            )
        else:
            # Valid
            import gradio as gr
            return (
                value,
                gr.update(value="", visible=False)
            )
    
    return callback

