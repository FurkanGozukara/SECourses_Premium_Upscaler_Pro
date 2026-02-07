"""
Input Path Auto-Detection System

Automatically detects and validates:
- Video files vs image files vs frame sequences
- Frame sequence ordering and naming patterns
- Missing frames in sequences
- Compatible formats
- Path validity on Windows and Linux
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class InputInfo:
    """Information about detected input"""
    input_type: str  # "video", "image", "frame_sequence", "directory"
    path: str
    is_valid: bool
    error_message: Optional[str] = None
    
    # For frame sequences
    frame_count: int = 0
    frame_pattern: Optional[str] = None
    frame_start: int = 0
    frame_end: int = 0
    missing_frames: List[int] = None
    
    # For videos/images
    format: Optional[str] = None
    
    # General
    total_files: int = 0
    info_message: str = ""


# Supported formats
VIDEO_FORMATS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", 
    ".m4v", ".mpg", ".mpeg", ".3gp", ".ogv"
}

IMAGE_FORMATS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", 
    ".webp", ".exr", ".hdr"
}

FRAME_SEQUENCE_PATTERNS = [
    r"(.+?)(\d+)\.(png|jpg|jpeg|tiff|tif|exr)$",  # name0001.png
    r"(.+?)_(\d+)\.(png|jpg|jpeg|tiff|tif|exr)$",  # name_0001.png
    r"(.+?)\.(\d+)\.(png|jpg|jpeg|tiff|tif|exr)$",  # name.0001.png
]


def detect_input(path_str: str) -> InputInfo:
    """
    Automatically detect input type and validate.
    
    Args:
        path_str: Path to video file, image file, or directory containing frames
        
    Returns:
        InputInfo with detection results
    """
    from .path_utils import normalize_path
    
    try:
        path_str = normalize_path(path_str)
        if not path_str:
            return InputInfo(
                input_type="unknown",
                path="",
                is_valid=False,
                error_message="Empty path provided"
            )
        
        path = Path(path_str)
        
        # Check if path exists
        if not path.exists():
            return InputInfo(
                input_type="unknown",
                path=path_str,
                is_valid=False,
                error_message=f"Path does not exist: {path_str}"
            )
        
        # Check if it's a file
        if path.is_file():
            return _detect_file(path)
        
        # Check if it's a directory
        elif path.is_dir():
            return _detect_directory(path)
        
        else:
            return InputInfo(
                input_type="unknown",
                path=path_str,
                is_valid=False,
                error_message="Path is neither file nor directory"
            )
            
    except Exception as e:
        return InputInfo(
            input_type="unknown",
            path=path_str,
            is_valid=False,
            error_message=f"Error detecting input: {str(e)}"
        )


def _detect_file(path: Path) -> InputInfo:
    """Detect single file (video or image)"""
    suffix = path.suffix.lower()
    
    if suffix in VIDEO_FORMATS:
        return InputInfo(
            input_type="video",
            path=str(path),
            is_valid=True,
            format=suffix[1:],  # Remove dot
            total_files=1,
            info_message=f"✓ Video file detected: {suffix[1:].upper()}"
        )
    
    elif suffix in IMAGE_FORMATS:
        return InputInfo(
            input_type="image",
            path=str(path),
            is_valid=True,
            format=suffix[1:],
            total_files=1,
            info_message=f"✓ Image file detected: {suffix[1:].upper()}"
        )
    
    else:
        return InputInfo(
            input_type="unknown",
            path=str(path),
            is_valid=False,
            error_message=f"Unsupported file format: {suffix}"
        )


def _detect_directory(path: Path) -> InputInfo:
    """Detect directory contents (frame sequence or multiple files)"""
    # Get all files in directory (not recursive)
    all_files = [f for f in path.iterdir() if f.is_file()]
    
    if not all_files:
        return InputInfo(
            input_type="directory",
            path=str(path),
            is_valid=False,
            error_message="Directory is empty"
        )
    
    # Try to detect frame sequence
    sequence_info = _detect_frame_sequence(path, all_files)
    if sequence_info and sequence_info.is_valid:
        return sequence_info
    
    # Check for multiple video/image files (batch processing)
    video_files = [f for f in all_files if f.suffix.lower() in VIDEO_FORMATS]
    image_files = [f for f in all_files if f.suffix.lower() in IMAGE_FORMATS]
    
    if video_files:
        return InputInfo(
            input_type="directory",
            path=str(path),
            is_valid=True,
            total_files=len(video_files),
            info_message=f"✓ Directory with {len(video_files)} video file(s) detected\n(Use batch processing mode)"
        )
    
    elif image_files:
        # Could be batch images or unordered frame sequence
        if len(image_files) > 10:  # Likely a frame sequence
            return InputInfo(
                input_type="directory",
                path=str(path),
                is_valid=True,
                total_files=len(image_files),
                info_message=f"⚠️ Directory with {len(image_files)} image files (no sequence pattern detected)\n(Will process as-is, ensure proper naming)"
            )
        else:  # Likely batch images
            return InputInfo(
                input_type="directory",
                path=str(path),
                is_valid=True,
                total_files=len(image_files),
                info_message=f"✓ Directory with {len(image_files)} image file(s) detected\n(Use batch processing mode)"
            )
    
    else:
        return InputInfo(
            input_type="directory",
            path=str(path),
            is_valid=False,
            error_message=f"No supported files found in directory"
        )


def _detect_frame_sequence(path: Path, files: List[Path]) -> Optional[InputInfo]:
    """
    Detect frame sequence pattern in directory.
    
    Looks for numbered sequences like:
    - frame_0001.png, frame_0002.png, ...
    - image.0000.exr, image.0001.exr, ...
    - shot0000.jpg, shot0001.jpg, ...
    """
    # Group files by pattern
    pattern_groups: Dict[str, List[Tuple[int, Path]]] = {}
    
    for file_path in files:
        if file_path.suffix.lower() not in IMAGE_FORMATS:
            continue
        
        filename = file_path.name
        
        # Try each pattern
        for pattern in FRAME_SEQUENCE_PATTERNS:
            match = re.match(pattern, filename, re.IGNORECASE)
            if match:
                base_name = match.group(1)
                frame_num = int(match.group(2))
                ext = match.group(3)
                
                # Create pattern key
                pattern_key = f"{base_name}_%d.{ext}"
                
                if pattern_key not in pattern_groups:
                    pattern_groups[pattern_key] = []
                
                pattern_groups[pattern_key].append((frame_num, file_path))
                break
    
    if not pattern_groups:
        return None
    
    # Find the largest sequence
    largest_sequence = max(pattern_groups.items(), key=lambda x: len(x[1]))
    pattern_key, frame_list = largest_sequence
    
    if len(frame_list) < 3:  # Need at least 3 frames to be considered a sequence
        return None
    
    # Sort frames by number
    frame_list.sort(key=lambda x: x[0])
    
    frame_numbers = [f[0] for f in frame_list]
    frame_start = min(frame_numbers)
    frame_end = max(frame_numbers)
    frame_count = len(frame_numbers)
    
    # Check for missing frames
    expected_frames = set(range(frame_start, frame_end + 1))
    actual_frames = set(frame_numbers)
    missing_frames = sorted(expected_frames - actual_frames)
    
    # Build info message
    info_parts = [
        f"✓ Frame sequence detected:",
        f"Pattern: {pattern_key}",
        f"Frames: {frame_start} - {frame_end} ({frame_count} files)"
    ]
    
    if missing_frames:
        if len(missing_frames) <= 10:
            info_parts.append(f"⚠️ Missing frames: {missing_frames}")
        else:
            info_parts.append(f"⚠️ Missing {len(missing_frames)} frames")
    else:
        info_parts.append("✓ No missing frames")
    
    info_message = "\n".join(info_parts)
    
    # Get first frame format
    first_frame = frame_list[0][1]
    format_ext = first_frame.suffix[1:].lower()
    
    return InputInfo(
        input_type="frame_sequence",
        path=str(path),
        is_valid=True,
        frame_count=frame_count,
        frame_pattern=pattern_key,
        frame_start=frame_start,
        frame_end=frame_end,
        missing_frames=missing_frames if missing_frames else None,
        format=format_ext,
        total_files=frame_count,
        info_message=info_message
    )


def validate_batch_directory(
    path_str: str,
    supported_formats: Optional[List[str]] = None
) -> Tuple[bool, List[str], str]:
    """
    Validate directory for batch processing.
    
    Args:
        path_str: Directory path
        supported_formats: List of supported extensions (e.g., ['.mp4', '.avi'])
        
    Returns:
        (is_valid, list_of_files, message)
    """
    from .path_utils import normalize_path
    
    try:
        path_str = normalize_path(path_str)
        if not path_str:
            return False, [], "Empty path"
        
        path = Path(path_str)
        
        if not path.exists():
            return False, [], f"Directory does not exist: {path_str}"
        
        if not path.is_dir():
            return False, [], "Path is not a directory"
        
        # Get all files
        all_files = [f for f in path.iterdir() if f.is_file()]
        
        if not all_files:
            return False, [], "Directory is empty"
        
        # Filter by supported formats if provided
        if supported_formats:
            supported_exts = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                            for ext in supported_formats}
            filtered_files = [f for f in all_files if f.suffix.lower() in supported_exts]
        else:
            # Default: videos and images
            filtered_files = [f for f in all_files 
                            if f.suffix.lower() in VIDEO_FORMATS or f.suffix.lower() in IMAGE_FORMATS]
        
        if not filtered_files:
            return False, [], f"No supported files found in directory"
        
        file_paths = sorted([str(f) for f in filtered_files])
        
        message = f"✓ Found {len(file_paths)} file(s) for batch processing"
        
        return True, file_paths, message
        
    except Exception as e:
        return False, [], f"Error validating directory: {e}"


def get_batch_summary(file_list: List[str]) -> str:
    """
    Generate summary of batch files.
    
    Returns formatted string with file counts by type.
    """
    from collections import Counter
    
    if not file_list:
        return "No files"
    
    # Count by extension
    extensions = [Path(f).suffix.lower() for f in file_list]
    counts = Counter(extensions)
    
    summary_parts = [f"Total: {len(file_list)} files"]
    
    for ext, count in counts.most_common():
        ext_name = ext[1:].upper() if ext else "unknown"
        summary_parts.append(f"  • {ext_name}: {count}")
    
    return "\n".join(summary_parts)

