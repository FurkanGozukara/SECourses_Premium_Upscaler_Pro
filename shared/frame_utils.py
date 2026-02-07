"""
Frame Extraction and Manipulation Utilities

Provides utilities for:
- Extracting specific frames from videos
- Converting videos to frame sequences
- Merging frames back to video
- Frame comparison and difference calculation
- Thumbnail generation for galleries
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List


def extract_video_thumbnail(
    video_path: str, 
    output_path: Optional[str] = None, 
    timestamp: float = 0.0, 
    width: int = 320
) -> Tuple[bool, Optional[str], str]:
    """
    Extract a thumbnail from a video for gallery display.
    
    Args:
        video_path: Path to video file
        output_path: Optional output path (if None, creates temp file)
        timestamp: Time in seconds to extract thumbnail from (default: first frame)
        width: Width of thumbnail in pixels (height auto-calculated, default: 320)
        
    Returns:
        (success: bool, thumbnail_path: str, error_message: str)
    """
    try:
        from .path_utils import normalize_path
        from .error_handling import check_ffmpeg_available
        
        if not check_ffmpeg_available():
            return False, None, "FFmpeg not available"
        
        video_path_obj = Path(normalize_path(video_path))
        if not video_path_obj.exists():
            return False, None, f"Video not found: {video_path}"
        
        # Create output path if not provided
        if output_path is None:
            temp_dir = Path(tempfile.gettempdir())
            output_path = temp_dir / f"thumb_{video_path_obj.stem}_{timestamp:.1f}s.jpg"
        else:
            output_path = Path(normalize_path(output_path))
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use ffmpeg to extract thumbnail (fast and reliable)
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite
            "-ss", str(timestamp),  # Seek to timestamp
            "-i", str(video_path_obj),
            "-vframes", "1",  # Extract 1 frame
            "-vf", f"scale={width}:-1",  # Resize to width, maintain aspect ratio
            "-q:v", "2",  # High quality JPEG
            str(output_path)
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10
        )
        
        if result.returncode == 0 and output_path.exists():
            return True, str(output_path), ""
        else:
            error = result.stderr.decode('utf-8', errors='ignore') if result.stderr else "Unknown error"
            return False, None, f"FFmpeg failed: {error[:200]}"
            
    except subprocess.TimeoutExpired:
        return False, None, "Thumbnail extraction timed out"
    except Exception as e:
        return False, None, f"Thumbnail extraction error: {str(e)}"


def extract_multiple_thumbnails(
    video_paths: List[str], 
    output_dir: Optional[str] = None, 
    width: int = 320
) -> List[Tuple[str, str]]:
    """
    Extract thumbnails from multiple videos for gallery display.
    
    Args:
        video_paths: List of video file paths
        output_dir: Directory to save thumbnails (if None, uses temp)
        width: Thumbnail width in pixels
        
    Returns:
        List of (thumbnail_path, caption) tuples for gr.Gallery
    """
    thumbnails = []
    
    if output_dir:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        output_dir_path = None
    
    for idx, video_path in enumerate(video_paths):
        video_path_obj = Path(video_path)
        
        if not video_path_obj.exists():
            continue
        
        # Generate output path
        if output_dir_path:
            output_path = output_dir_path / f"thumb_chunk_{idx+1:04d}.jpg"
        else:
            output_path = None
        
        success, thumb_path, error = extract_video_thumbnail(
            video_path, 
            output_path=str(output_path) if output_path else None,
            width=width
        )
        
        if success and thumb_path:
            # Create caption with chunk info
            caption = f"Chunk {idx+1}: {video_path_obj.name}"
            thumbnails.append((thumb_path, caption))
    
    return thumbnails


def extract_first_frame(
    video_path: str,
    output_path: Optional[str] = None,
    format: str = "png"
) -> Tuple[bool, Optional[str], str]:
    """
    Extract the first frame from a video.
    
    Args:
        video_path: Path to input video
        output_path: Path for output frame (optional, will auto-generate if not provided)
        format: Output format (png, jpg, etc.)
        
    Returns:
        (success, frame_path, error_message)
    """
    from .path_utils import normalize_path
    from .error_handling import check_ffmpeg_available
    
    if not check_ffmpeg_available():
        return False, None, "FFmpeg not available"
    
    try:
        video_path = normalize_path(video_path)
        if not Path(video_path).exists():
            return False, None, f"Video not found: {video_path}"
        
        # Generate output path if not provided
        if output_path is None:
            temp_dir = Path(tempfile.mkdtemp(prefix="frame_extract_"))
            output_path = str(temp_dir / f"frame_0001.{format}")
        else:
            output_path = normalize_path(output_path)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Extract first frame using ffmpeg
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vframes", "1",  # Extract only 1 frame
            "-q:v", "1",  # Highest quality
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and Path(output_path).exists():
            return True, output_path, ""
        else:
            error = result.stderr or result.stdout or "Unknown ffmpeg error"
            return False, None, f"FFmpeg error: {error}"
            
    except subprocess.TimeoutExpired:
        return False, None, "Frame extraction timed out"
    except Exception as e:
        return False, None, f"Error extracting frame: {str(e)}"


def extract_frame_at_time(
    video_path: str,
    timestamp_sec: float,
    output_path: Optional[str] = None,
    format: str = "png"
) -> Tuple[bool, Optional[str], str]:
    """
    Extract a frame at a specific timestamp.
    
    Args:
        video_path: Path to input video
        timestamp_sec: Timestamp in seconds
        output_path: Path for output frame
        format: Output format
        
    Returns:
        (success, frame_path, error_message)
    """
    from .path_utils import normalize_path
    from .error_handling import check_ffmpeg_available
    
    if not check_ffmpeg_available():
        return False, None, "FFmpeg not available"
    
    try:
        video_path = normalize_path(video_path)
        
        if output_path is None:
            temp_dir = Path(tempfile.mkdtemp(prefix="frame_extract_"))
            output_path = str(temp_dir / f"frame_{int(timestamp_sec * 1000):06d}.{format}")
        else:
            output_path = normalize_path(output_path)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Extract frame at timestamp
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(timestamp_sec),  # Seek to timestamp
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "1",
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and Path(output_path).exists():
            return True, output_path, ""
        else:
            error = result.stderr or "Frame extraction failed"
            return False, None, error
            
    except Exception as e:
        return False, None, f"Error: {str(e)}"


def extract_frame_by_number(
    video_path: str,
    frame_number: int,
    output_path: Optional[str] = None,
    format: str = "png"
) -> Tuple[bool, Optional[str], str]:
    """
    Extract a specific frame by frame number.
    
    Args:
        video_path: Path to input video
        frame_number: Frame index (0-based)
        output_path: Path for output frame
        format: Output format
        
    Returns:
        (success, frame_path, error_message)
    """
    from .path_utils import get_media_fps, normalize_path
    
    # Get FPS to convert frame number to timestamp
    fps = get_media_fps(video_path)
    if not fps or fps <= 0:
        fps = 30.0  # Default fallback
    
    timestamp = frame_number / fps
    return extract_frame_at_time(video_path, timestamp, output_path, format)


def create_frame_difference(
    frame1_path: str,
    frame2_path: str,
    output_path: Optional[str] = None,
    diff_mode: str = "absolute"
) -> Tuple[bool, Optional[str], str]:
    """
    Create a difference image between two frames.
    
    Args:
        frame1_path: Path to first frame
        frame2_path: Path to second frame
        output_path: Path for difference output
        diff_mode: "absolute", "relative", or "heatmap"
        
    Returns:
        (success, diff_path, error_message)
    """
    try:
        import cv2
        import numpy as np
        from .path_utils import normalize_path
        
        # Read images
        img1 = cv2.imread(normalize_path(frame1_path))
        img2 = cv2.imread(normalize_path(frame2_path))
        
        if img1 is None or img2 is None:
            return False, None, "Failed to load one or both images"
        
        # Resize if dimensions don't match
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Calculate difference
        if diff_mode == "absolute":
            diff = cv2.absdiff(img1, img2)
        elif diff_mode == "relative":
            diff = cv2.subtract(img2, img1)
        elif diff_mode == "heatmap":
            # Create heatmap of differences
            diff_gray = cv2.absdiff(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 
                                    cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
            diff = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
        else:
            diff = cv2.absdiff(img1, img2)
        
        # Generate output path
        if output_path is None:
            temp_dir = Path(tempfile.mkdtemp(prefix="frame_diff_"))
            output_path = str(temp_dir / "difference.png")
        else:
            output_path = normalize_path(output_path)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save difference image
        cv2.imwrite(output_path, diff)
        
        if Path(output_path).exists():
            return True, output_path, ""
        else:
            return False, None, "Failed to save difference image"
            
    except Exception as e:
        return False, None, f"Error creating difference: {str(e)}"


def video_to_frames(
    video_path: str,
    output_dir: str,
    name_pattern: str = "frame_%05d.png",
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    fps: Optional[float] = None
) -> Tuple[bool, List[str], str]:
    """
    Convert video to frame sequence.
    
    Args:
        video_path: Path to input video
        output_dir: Directory for output frames
        name_pattern: Printf-style pattern for frame names
        start_frame: First frame to extract (0-based)
        end_frame: Last frame to extract (None = all)
        fps: Output FPS (None = use source FPS)
        
    Returns:
        (success, list_of_frame_paths, error_message)
    """
    from .path_utils import normalize_path
    from .error_handling import check_ffmpeg_available
    
    if not check_ffmpeg_available():
        return False, [], "FFmpeg not available"
    
    try:
        video_path = normalize_path(video_path)
        output_dir = normalize_path(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Build ffmpeg command
        cmd = ["ffmpeg", "-y", "-i", video_path]
        
        if start_frame > 0:
            from .path_utils import get_media_fps
            source_fps = get_media_fps(video_path) or 30.0
            start_time = start_frame / source_fps
            cmd.extend(["-ss", str(start_time)])
        
        if end_frame is not None and end_frame > start_frame:
            frame_count = end_frame - start_frame
            cmd.extend(["-vframes", str(frame_count)])
        
        if fps:
            cmd.extend(["-r", str(fps)])
        
        output_pattern = str(Path(output_dir) / name_pattern)
        cmd.extend(["-q:v", "1", output_pattern])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            return False, [], result.stderr or "FFmpeg failed"
        
        # Collect generated frames
        frames = sorted(Path(output_dir).glob("*.png"))
        frame_paths = [str(f) for f in frames]
        
        return True, frame_paths, ""
        
    except Exception as e:
        return False, [], f"Error: {str(e)}"

