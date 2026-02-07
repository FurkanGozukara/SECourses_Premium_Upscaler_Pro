"""
Video Encoding Utilities with Advanced Features

Provides comprehensive video encoding with:
- Two-pass encoding for optimal quality
- Multiple codec support (H.264, H.265, ProRes, VP9, AV1)
- Pixel format selection (yuv420p, yuv444p, 10-bit, RGB)
- Quality presets and CRF control
- Audio codec options
- FPS handling and preservation
- Metadata embedding
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from .video_codec_options import build_ffmpeg_video_encode_args


def encode_video(
    input_frames_path: str,
    output_video_path: str,
    fps: float = 30.0,
    codec: str = "h264",
    crf: int = 18,
    preset: str = "medium",
    pixel_format: str = "yuv420p",
    audio_codec: str = "copy",
    audio_bitrate: Optional[str] = None,
    two_pass: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    on_progress: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Encode video from frame sequence with advanced options.
    
    Args:
        input_frames_path: Path pattern for input frames (e.g., "frames/frame_%05d.png")
        output_video_path: Output video file path
        fps: Frame rate
        codec: Video codec (libx264, libx265, libvpx-vp9)
        crf: Constant Rate Factor (0-51, lower = better quality)
        preset: Encoding preset (ultrafast to veryslow)
        two_pass: Use two-pass encoding for better quality
        metadata: Metadata to embed in video
        on_progress: Progress callback
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if on_progress:
            on_progress(f"Encoding video: {codec} CRF={crf} preset={preset}{'(two-pass)' if two_pass else ''}\n")
        
        # Auto-select codec if 'auto'
        if codec == "auto":
            codec = "h264"  # Default to most compatible
        
        # Build base ffmpeg command
        base_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", input_frames_path,
        ]
        
        # Add metadata if provided
        if metadata:
            metadata_str = json.dumps(metadata)
            base_cmd.extend(["-metadata", f"comment={metadata_str}"])
        
        # Use comprehensive codec builder
        codec_args = build_ffmpeg_video_encode_args(
            codec=codec,
            quality=crf,
            pixel_format=pixel_format,
            preset=preset,
            audio_codec=audio_codec,
            audio_bitrate=audio_bitrate
        )
        
        if two_pass:
            # Two-pass encoding for better quality/size ratio
            if on_progress:
                on_progress("Pass 1/2: Analyzing...\n")
            
            # Create temp file for pass log
            with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as passlog:
                passlog_path = passlog.name
            
            try:
                # First pass
                pass1_cmd = base_cmd + codec_args + [
                    "-pass", "1",
                    "-passlogfile", passlog_path,
                    "-f", "null",  # No output for first pass
                    "-" if Path(output_video_path).suffix == ".mp4" else "/dev/null"
                ]
                
                result1 = subprocess.run(
                    pass1_cmd,
                    capture_output=True,
                    text=True
                )
                
                if result1.returncode != 0:
                    if on_progress:
                        on_progress(f"⚠️ Pass 1 failed: {result1.stderr}\n")
                    # Clean up and try single-pass
                    Path(passlog_path).unlink(missing_ok=True)
                    return encode_video(
                        input_frames_path=input_frames_path,
                        output_video_path=output_video_path,
                        fps=fps,
                        codec=codec,
                        crf=crf,
                        preset=preset,
                        pixel_format=pixel_format,
                        audio_codec=audio_codec,
                        audio_bitrate=audio_bitrate,
                        two_pass=False,
                        metadata=metadata,
                        on_progress=on_progress,
                    )
                
                if on_progress:
                    on_progress("Pass 2/2: Encoding...\n")
                
                # Second pass
                pass2_cmd = base_cmd + codec_args + [
                    "-pass", "2",
                    "-passlogfile", passlog_path,
                    str(output_video_path)
                ]
                
                result2 = subprocess.run(
                    pass2_cmd,
                    capture_output=True,
                    text=True
                )
                
                # Clean up pass log files
                Path(passlog_path).unlink(missing_ok=True)
                Path(passlog_path + "-0.log").unlink(missing_ok=True)
                Path(passlog_path + "-0.log.mbtree").unlink(missing_ok=True)
                
                if result2.returncode != 0:
                    if on_progress:
                        on_progress(f"❌ Pass 2 failed: {result2.stderr}\n")
                    return False
                
            finally:
                # Ensure cleanup of pass log files
                for log_file in Path(passlog_path).parent.glob(f"{Path(passlog_path).stem}*"):
                    log_file.unlink(missing_ok=True)
        
        else:
            # Single-pass encoding
            cmd = base_cmd + codec_args + [str(output_video_path)]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                if on_progress:
                    on_progress(f"❌ Encoding failed: {result.stderr}\n")
                return False
        
        if on_progress:
            on_progress(f"✅ Video encoded: {output_video_path}\n")
        
        return Path(output_video_path).exists()
        
    except Exception as e:
        if on_progress:
            on_progress(f"❌ Encoding error: {e}\n")
        return False


def embed_metadata_in_video(
    video_path: str,
    metadata: Dict[str, Any],
    format: str = "json",
    on_progress: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Embed metadata in video file.
    
    Args:
        video_path: Path to video file
        metadata: Metadata dictionary to embed
        format: Metadata format ("json", "xml", "exif")
        on_progress: Progress callback
        
    Returns:
        True if successful
    """
    try:
        if format == "json":
            # Embed as JSON in comment metadata field
            metadata_str = json.dumps(metadata)
            
            temp_output = Path(video_path).with_stem(f"{Path(video_path).stem}_meta")
            
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-c", "copy",
                "-metadata", f"comment={metadata_str}",
                str(temp_output)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and temp_output.exists():
                # Replace original with metadata-embedded version
                Path(video_path).unlink()
                temp_output.rename(video_path)
                if on_progress:
                    on_progress("✅ Metadata embedded\n")
                return True
                
        elif format == "xml":
            # Create sidecar XML file
            xml_path = Path(video_path).with_suffix(".xml")
            
            # Simple XML generation
            xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n<metadata>\n'
            for key, value in metadata.items():
                xml_content += f"  <{key}>{value}</{key}>\n"
            xml_content += '</metadata>\n'
            
            xml_path.write_text(xml_content, encoding='utf-8')
            
            if on_progress:
                on_progress(f"✅ Metadata saved to {xml_path}\n")
            return True
            
        elif format == "exif":
            # EXIF embedding (requires exiftool if available)
            # For now, save as JSON sidecar
            json_path = Path(video_path).with_suffix('.json')
            json_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            
            if on_progress:
                on_progress(f"✅ Metadata saved to {json_path}\n")
            return True
        
        return False
        
    except Exception as e:
        if on_progress:
            on_progress(f"⚠️ Metadata embedding failed: {e}\n")
        return False


def set_video_fps(
    input_video: str,
    output_video: str,
    fps: float,
    on_progress: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Change video FPS by re-encoding or stream copying.
    
    Args:
        input_video: Input video path
        output_video: Output video path
        fps: Target FPS
        on_progress: Progress callback
        
    Returns:
        True if successful
    """
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-r", str(fps),
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            output_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and Path(output_video).exists():
            if on_progress:
                on_progress(f"✅ FPS set to {fps}\n")
            return True
        else:
            if on_progress:
                on_progress(f"❌ FPS change failed\n")
            return False
            
    except Exception as e:
        if on_progress:
            on_progress(f"❌ FPS error: {e}\n")
        return False

