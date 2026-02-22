"""
Advanced Video Comparison System

Provides multiple comparison layouts:
- Side-by-side video comparison
- Slider-based comparison (using HTML/JS)
- Pin reference frame for iterative comparison
- Fullscreen comparison mode
- Automatic layout detection based on video properties
"""

import json
import re
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

COMPARISON_BRAND_TEXT = "SECourses Upscaler Pro"
COMPARISON_BRAND_RIGHT_PAD = 20
FFMPEG_PROGRESS_TIME_RE = re.compile(r"time=(\d{2}:\d{2}:\d{2}(?:\.\d+)?)")


@dataclass
class ComparisonConfig:
    """Configuration for video comparison"""
    layout: str = "slider"  # "slider", "side_by_side", "stacked", "auto"
    enable_fullscreen: bool = True
    show_info_overlay: bool = True
    sync_playback: bool = True
    pin_reference: bool = False
    pinned_frame_path: Optional[str] = None


def _coerce_even_dimension(value: Any) -> Optional[int]:
    """Convert a numeric value into a positive even integer dimension."""
    try:
        dim = int(float(value))
    except Exception:
        return None
    if dim <= 0:
        return None
    if dim % 2 != 0:
        dim += 1
    return max(2, dim)


def _coerce_font_size(value: Any, default: int = 32) -> int:
    try:
        size = int(float(value))
    except Exception:
        size = int(default)
    return max(8, min(200, size))


def _resolve_pair_base_dimensions(
    dims_a: Tuple[int, int],
    dims_b: Tuple[int, int],
) -> Tuple[int, int]:
    """
    Pick the larger source resolution as the common merge base.
    """
    a_w, a_h = int(dims_a[0] or 0), int(dims_a[1] or 0)
    b_w, b_h = int(dims_b[0] or 0), int(dims_b[1] or 0)
    if a_w <= 0 or a_h <= 0:
        return max(2, b_w), max(2, b_h)
    if b_w <= 0 or b_h <= 0:
        return max(2, a_w), max(2, a_h)

    a_area = a_w * a_h
    b_area = b_w * b_h
    if a_area > b_area:
        return a_w, a_h
    if b_area > a_area:
        return b_w, b_h

    if (a_w, a_h) >= (b_w, b_h):
        return a_w, a_h
    return b_w, b_h


def normalize_comparison_layout(layout: Optional[str], width: int, height: int) -> str:
    """
    Normalize user-facing layout values into ffmpeg merge modes.

    Accepted values:
    - horizontal aliases: horizontal, left_to_right, side_by_side
    - vertical aliases: vertical, top_to_bottom, stacked
    - auto: pick layout closest to 16:9 output aspect ratio
    """
    key = str(layout or "auto").strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"horizontal", "left_to_right", "side_by_side"}:
        return "horizontal"
    if key in {"vertical", "top_to_bottom", "stacked"}:
        return "vertical"
    if width > 0 and height > 0:
        return get_smart_comparison_layout(width, height)
    return "horizontal"


def predict_comparison_dimensions(base_width: int, base_height: int, layout: str = "auto") -> Tuple[str, int, int]:
    """
    Predict final merged video dimensions from base output dimensions.

    Returns:
        (resolved_layout, final_width, final_height)
    """
    width = int(base_width or 0)
    height = int(base_height or 0)
    if width <= 0 or height <= 0:
        return "horizontal", 0, 0

    resolved_layout = normalize_comparison_layout(layout, width, height)
    if resolved_layout == "vertical":
        merged_w, merged_h = width, height * 2
    else:
        merged_w, merged_h = width * 2, height

    safe_w = _coerce_even_dimension(merged_w) or 2
    safe_h = _coerce_even_dimension(merged_h) or 2
    return resolved_layout, safe_w, safe_h


def _escape_drawtext_text(value: str) -> str:
    """
    Escape text for ffmpeg drawtext `text=...` values.
    """
    txt = str(value or "").replace("\r", " ").replace("\n", " ")
    replacements = [
        ("\\", "\\\\"),
        (":", r"\:"),
        ("'", r"\'"),
        ("%", r"\%"),
        (",", r"\,"),
        (";", r"\;"),
        ("=", r"\="),
        ("[", r"\["),
        ("]", r"\]"),
    ]
    for src, dst in replacements:
        txt = txt.replace(src, dst)
    return txt


def _parse_ffmpeg_time_to_seconds(value: str) -> Optional[float]:
    try:
        parts = str(value or "").strip().split(":")
        if len(parts) != 3:
            return None
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return (hours * 3600.0) + (minutes * 60.0) + seconds
    except Exception:
        return None


def _run_ffmpeg_with_progress(
    cmd: List[str],
    timeout_seconds: int,
    duration_seconds: Optional[float],
    on_progress: Optional[callable],
) -> Tuple[int, str]:
    """
    Execute ffmpeg and stream best-effort progress callbacks from stderr.
    """
    stderr_tail: deque[str] = deque(maxlen=120)
    duration = float(duration_seconds or 0.0)
    last_emit_time = 0.0
    last_pct = -1.0

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    start = time.time()

    try:
        if proc.stderr is not None:
            while True:
                if time.time() - start > timeout_seconds:
                    proc.kill()
                    raise subprocess.TimeoutExpired(cmd, timeout_seconds)

                line = proc.stderr.readline()
                if line == "" and proc.poll() is not None:
                    break
                if not line:
                    continue

                stripped = line.strip()
                if not stripped:
                    continue
                stderr_tail.append(stripped)

                match = FFMPEG_PROGRESS_TIME_RE.search(stripped)
                if not match:
                    continue
                elapsed_seconds = _parse_ffmpeg_time_to_seconds(match.group(1))
                if elapsed_seconds is None:
                    continue

                now = time.time()
                if duration > 0.0:
                    pct = max(0.0, min(100.0, (elapsed_seconds / duration) * 100.0))
                    if (pct - last_pct) >= 0.3 or (now - last_emit_time) >= 0.6 or pct >= 99.9:
                        if on_progress:
                            on_progress(
                                f"COMPARISON_PROGRESS {pct:.1f}% "
                                f"({elapsed_seconds:.1f}s/{duration:.1f}s)\n"
                            )
                        last_pct = pct
                        last_emit_time = now
                elif (now - last_emit_time) >= 1.0:
                    if on_progress:
                        on_progress(f"COMPARISON_PROGRESS time={elapsed_seconds:.1f}s\n")
                    last_emit_time = now

        returncode = proc.wait(timeout=5)
        return returncode, "\n".join(stderr_tail)
    finally:
        try:
            if proc.stderr is not None:
                proc.stderr.close()
        except Exception:
            pass


def create_side_by_side_video(
    input_video: str,
    output_video: str,
    comparison_output: str,
    label_left: str = "Original",
    label_right: str = "Upscaled"
) -> Tuple[bool, str, str]:
    """
    Create side-by-side comparison video using ffmpeg.
    
    Args:
        input_video: Path to original video
        output_video: Path to upscaled video
        comparison_output: Path for comparison output
        label_left: Label for left video
        label_right: Label for right video
        
    Returns:
        (success, output_path, error_message)
    """
    from .path_utils import normalize_path
    from .error_handling import check_ffmpeg_available
    
    try:
        # Check ffmpeg
        if not check_ffmpeg_available():
            return False, "", "FFmpeg not found in PATH"
        
        input_video = normalize_path(input_video)
        output_video = normalize_path(output_video)
        comparison_output = normalize_path(comparison_output)
        
        # Build ffmpeg command for side-by-side with labels
        cmd = [
            "ffmpeg",
            "-i", input_video,
            "-i", output_video,
            "-filter_complex",
            f"[0:v]scale=iw/2:ih/2,drawtext=text='{label_left}':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[left];"
            f"[1:v]scale=iw/2:ih/2,drawtext=text='{label_right}':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[right];"
            f"[left][right]hstack[out]",
            "-map", "[out]",
            "-map", "0:a?",  # Copy audio from first input if exists
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-c:a", "copy",
            "-y",
            comparison_output
        ]
        
        # Run ffmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        if result.returncode == 0 and Path(comparison_output).exists():
            return True, comparison_output, ""
        else:
            error = result.stderr or result.stdout or "Unknown ffmpeg error"
            return False, "", error
            
    except subprocess.TimeoutExpired:
        return False, "", "FFmpeg process timed out (>10 minutes)"
    except Exception as e:
        return False, "", f"Error creating comparison: {str(e)}"


def create_stacked_video(
    input_video: str,
    output_video: str,
    comparison_output: str,
    label_top: str = "Original",
    label_bottom: str = "Upscaled"
) -> Tuple[bool, str, str]:
    """
    Create top/bottom stacked comparison video.
    
    Returns:
        (success, output_path, error_message)
    """
    from .path_utils import normalize_path
    from .error_handling import check_ffmpeg_available
    
    try:
        if not check_ffmpeg_available():
            return False, "", "FFmpeg not found in PATH"
        
        input_video = normalize_path(input_video)
        output_video = normalize_path(output_video)
        comparison_output = normalize_path(comparison_output)
        
        # Build ffmpeg command for stacked
        cmd = [
            "ffmpeg",
            "-i", input_video,
            "-i", output_video,
            "-filter_complex",
            f"[0:v]scale=iw:ih/2,drawtext=text='{label_top}':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[top];"
            f"[1:v]scale=iw:ih/2,drawtext=text='{label_bottom}':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[bottom];"
            f"[top][bottom]vstack[out]",
            "-map", "[out]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-c:a", "copy",
            "-y",
            comparison_output
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0 and Path(comparison_output).exists():
            return True, comparison_output, ""
        else:
            error = result.stderr or result.stdout or "Unknown ffmpeg error"
            return False, "", error
            
    except Exception as e:
        return False, "", f"Error creating stacked comparison: {str(e)}"


def create_comparison_slider_html(
    input_video: str,
    output_video: str,
    output_html: str,
    config: Optional[ComparisonConfig] = None
) -> Tuple[bool, str, str]:
    """
    Create HTML-based video comparison with slider.
    
    Uses custom HTML/JS for interactive slider control.
    
    Returns:
        (success, html_path, error_message)
    """
    if config is None:
        config = ComparisonConfig()
    
    try:
        from .path_utils import normalize_path
        
        input_video = Path(normalize_path(input_video))
        output_video = Path(normalize_path(output_video))
        output_html = Path(normalize_path(output_html))
        
        if not input_video.exists():
            return False, "", f"Input video not found: {input_video}"
        if not output_video.exists():
            return False, "", f"Output video not found: {output_video}"
        
        # Create HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Comparison Slider</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            background: #1a1a1a;
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            overflow: hidden;
        }}
        
        .container {{
            position: relative;
            width: 100vw;
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .comparison-container {{
            position: relative;
            width: 90%;
            max-width: 1920px;
            height: auto;
            overflow: hidden;
            border-radius: 8px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }}
        
        .video-wrapper {{
            position: relative;
            width: 100%;
            padding-bottom: 56.25%; /* 16:9 aspect ratio */
        }}
        
        video {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}
        
        .original-video {{
            z-index: 1;
        }}
        
        .upscaled-video {{
            z-index: 2;
            clip-path: inset(0 50% 0 0);
        }}
        
        .slider-control {{
            position: absolute;
            top: 0;
            left: 50%;
            width: 4px;
            height: 100%;
            background: #fff;
            z-index: 3;
            cursor: ew-resize;
            box-shadow: 0 0 10px rgba(0,0,0,0.5);
        }}
        
        .slider-handle {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 60px;
            height: 60px;
            background: #fff;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
        
        .slider-handle::before,
        .slider-handle::after {{
            content: '';
            position: absolute;
            width: 0;
            height: 0;
            border-style: solid;
        }}
        
        .slider-handle::before {{
            left: 12px;
            border-width: 8px 12px 8px 0;
            border-color: transparent #1a1a1a transparent transparent;
        }}
        
        .slider-handle::after {{
            right: 12px;
            border-width: 8px 0 8px 12px;
            border-color: transparent transparent transparent #1a1a1a;
        }}
        
        .label {{
            position: absolute;
            top: 20px;
            padding: 8px 16px;
            background: rgba(0,0,0,0.7);
            border-radius: 4px;
            font-size: 14px;
            font-weight: 600;
            z-index: 4;
            backdrop-filter: blur(10px);
        }}
        
        .label-left {{
            left: 20px;
        }}
        
        .label-right {{
            right: 20px;
        }}
        
        .controls {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 12px;
            z-index: 4;
        }}
        
        .btn {{
            padding: 12px 24px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            color: #fff;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
            backdrop-filter: blur(10px);
        }}
        
        .btn:hover {{
            background: rgba(255,255,255,0.2);
        }}
        
        .btn.active {{
            background: rgba(99, 102, 241, 0.8);
            border-color: rgb(99, 102, 241);
        }}
        
        .info-overlay {{
            position: absolute;
            top: 60px;
            right: 20px;
            padding: 12px 16px;
            background: rgba(0,0,0,0.7);
            border-radius: 4px;
            font-size: 12px;
            z-index: 4;
            max-width: 300px;
            backdrop-filter: blur(10px);
        }}
        
        .fullscreen-btn {{
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="comparison-container">
            <div class="video-wrapper">
                <video class="original-video" id="originalVideo" loop>
                    <source src="{input_video.name}" type="video/mp4">
                </video>
                <video class="upscaled-video" id="upscaledVideo" loop>
                    <source src="{output_video.name}" type="video/mp4">
                </video>
                <div class="slider-control" id="sliderControl">
                    <div class="slider-handle"></div>
                </div>
                <div class="label label-left">Original</div>
                <div class="label label-right">Upscaled</div>
                {f'<div class="info-overlay" id="infoOverlay"></div>' if config.show_info_overlay else ''}
            </div>
            
            <div class="controls">
                <button class="btn" id="playBtn">Play / Pause</button>
                <button class="btn" id="muteBtn">Mute / Unmute</button>
                <button class="btn" id="resetBtn">Reset Slider</button>
                {f'<button class="btn fullscreen-btn" id="fullscreenBtn">Fullscreen</button>' if config.enable_fullscreen else ''}
            </div>
        </div>
    </div>
    
    <script>
        const originalVideo = document.getElementById('originalVideo');
        const upscaledVideo = document.getElementById('upscaledVideo');
        const sliderControl = document.getElementById('sliderControl');
        const playBtn = document.getElementById('playBtn');
        const muteBtn = document.getElementById('muteBtn');
        const resetBtn = document.getElementById('resetBtn');
        const fullscreenBtn = document.getElementById('fullscreenBtn');
        const infoOverlay = document.getElementById('infoOverlay');
        
        let isDragging = false;
        let sliderPosition = 50;
        
        // Sync video playback
        {'const syncPlayback = true;' if config.sync_playback else 'const syncPlayback = false;'}
        
        if (syncPlayback) {{
            originalVideo.addEventListener('timeupdate', () => {{
                if (Math.abs(originalVideo.currentTime - upscaledVideo.currentTime) > 0.1) {{
                    upscaledVideo.currentTime = originalVideo.currentTime;
                }}
            }});
        }}
        
        // Slider functionality
        function updateSlider(clientX) {{
            const rect = sliderControl.parentElement.getBoundingClientRect();
            const x = clientX - rect.left;
            sliderPosition = Math.max(0, Math.min(100, (x / rect.width) * 100));
            
            sliderControl.style.left = sliderPosition + '%';
            upscaledVideo.style.clipPath = `inset(0 ${{100 - sliderPosition}}% 0 0)`;
        }}
        
        sliderControl.addEventListener('mousedown', (e) => {{
            isDragging = true;
            e.preventDefault();
        }});
        
        document.addEventListener('mousemove', (e) => {{
            if (isDragging) {{
                updateSlider(e.clientX);
            }}
        }});
        
        document.addEventListener('mouseup', () => {{
            isDragging = false;
        }});
        
        // Touch support
        sliderControl.addEventListener('touchstart', (e) => {{
            isDragging = true;
            e.preventDefault();
        }});
        
        document.addEventListener('touchmove', (e) => {{
            if (isDragging) {{
                updateSlider(e.touches[0].clientX);
            }}
        }});
        
        document.addEventListener('touchend', () => {{
            isDragging = false;
        }});
        
        // Controls
        playBtn.addEventListener('click', () => {{
            if (originalVideo.paused) {{
                originalVideo.play();
                upscaledVideo.play();
                playBtn.classList.add('active');
            }} else {{
                originalVideo.pause();
                upscaledVideo.pause();
                playBtn.classList.remove('active');
            }}
        }});
        
        muteBtn.addEventListener('click', () => {{
            originalVideo.muted = !originalVideo.muted;
            upscaledVideo.muted = !upscaledVideo.muted;
            muteBtn.classList.toggle('active');
        }});
        
        resetBtn.addEventListener('click', () => {{
            sliderPosition = 50;
            sliderControl.style.left = '50%';
            upscaledVideo.style.clipPath = 'inset(0 50% 0 0)';
        }});
        
        {f'''
        if (fullscreenBtn) {{
            fullscreenBtn.addEventListener('click', () => {{
                const container = document.querySelector('.comparison-container');
                if (!document.fullscreenElement) {{
                    container.requestFullscreen();
                }} else {{
                    document.exitFullscreen();
                }}
            }});
        }}
        ''' if config.enable_fullscreen else ''}
        
        {f'''
        // Update info overlay
        if (infoOverlay) {{
            function updateInfo() {{
                const time = originalVideo.currentTime.toFixed(1);
                const duration = originalVideo.duration.toFixed(1);
                infoOverlay.innerHTML = `
                    <div>Time: ${{time}}s / ${{duration}}s</div>
                    <div>Slider: ${{sliderPosition.toFixed(0)}}%</div>
                `;
            }}
            
            originalVideo.addEventListener('timeupdate', updateInfo);
            setInterval(updateInfo, 100);
        }}
        ''' if config.show_info_overlay else ''}
        
        // Mute by default
        originalVideo.muted = true;
        upscaledVideo.muted = true;
        muteBtn.classList.add('active');
    </script>
</body>
</html>
"""
        
        # Write HTML file
        output_html.parent.mkdir(parents=True, exist_ok=True)
        output_html.write_text(html_content, encoding='utf-8')
        
        return True, str(output_html), ""
        
    except Exception as e:
        return False, "", f"Error creating comparison HTML: {str(e)}"


def auto_select_comparison_layout(
    input_video: str,
    output_video: str
) -> str:
    """
    Automatically select best comparison layout based on video properties.

    Returns:
        "slider", "side_by_side", or "stacked"
    """
    from .path_utils import get_media_dimensions

    try:
        # Get dimensions
        in_w, in_h = get_media_dimensions(input_video)
        out_w, out_h = get_media_dimensions(output_video)

        if not in_w or not in_h or not out_w or not out_h:
            return "slider"  # Default

        # If resolutions are very different, use stacked
        if abs(in_w - out_w) > 100 or abs(in_h - out_h) > 100:
            return "stacked"

        # If both are high resolution (>1080p), use slider
        if min(in_w, in_h) > 1080 and min(out_w, out_h) > 1080:
            return "slider"

        # Otherwise, side-by-side works well
        return "side_by_side"

    except Exception:
        return "slider"


def create_input_vs_output_comparison_video(
    original_input_video: str,
    upscaled_output_video: str,
    comparison_output: Optional[str] = None,
    layout: str = "auto",
    label_input: str = "Original",
    label_output: str = "Upscaled",
    on_progress: Optional[callable] = None,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    font_size: int = 32,
    include_branding: bool = True,
) -> Tuple[bool, str, str]:
    """
    Create a comparison video of original input vs upscaled output.

    The original input is scaled UP to match the output resolution,
    then both are merged horizontally or vertically based on aspect ratio.

    Args:
        original_input_video: Path to the ORIGINAL input video (before any downscaling)
        upscaled_output_video: Path to the upscaled output video
        comparison_output: Output path for comparison video (auto-generated if None)
        layout: "auto" (choose based on aspect ratio), "horizontal", or "vertical"
        label_input: Label for the input/original side
        label_output: Label for the output/upscaled side
        on_progress: Optional progress callback
        target_width: Optional forced final width (even integer, <=0 ignored)
        target_height: Optional forced final height (even integer, <=0 ignored)
        font_size: Label drawtext font size (8-200)
        include_branding: When True, draws COMPARISON_BRAND_TEXT on right/bottom video

    Returns:
        (success, comparison_video_path, error_message)
    """
    from .path_utils import (
        normalize_path,
        get_media_dimensions,
        get_media_duration_seconds,
    )
    from .error_handling import check_ffmpeg_available

    try:
        if not check_ffmpeg_available():
            return False, "", "FFmpeg not found in PATH"

        input_path = Path(normalize_path(original_input_video))
        output_path = Path(normalize_path(upscaled_output_video))

        if not input_path.exists():
            return False, "", f"Original input not found: {input_path}"
        if not output_path.exists():
            return False, "", f"Upscaled output not found: {output_path}"

        # Get dimensions of both videos
        in_dims = get_media_dimensions(str(input_path))
        out_dims = get_media_dimensions(str(output_path))

        if not in_dims or not out_dims:
            return False, "", "Could not determine video dimensions"

        in_w, in_h = in_dims
        out_w, out_h = out_dims
        base_w, base_h = _resolve_pair_base_dimensions((in_w, in_h), (out_w, out_h))

        layout, auto_w, auto_h = predict_comparison_dimensions(base_w, base_h, layout)
        custom_w = _coerce_even_dimension(target_width)
        custom_h = _coerce_even_dimension(target_height)
        use_custom_size = bool(custom_w and custom_h)
        final_w = custom_w if use_custom_size else auto_w
        final_h = custom_h if use_custom_size else auto_h

        # Generate output path if not specified
        if comparison_output is None:
            comparison_output = str(output_path.parent / f"{output_path.stem}_comparison.mp4")
        else:
            comparison_output = normalize_path(comparison_output)

        Path(comparison_output).parent.mkdir(parents=True, exist_ok=True)

        if on_progress:
            on_progress(
                f"Creating comparison video ({layout} layout, {final_w}x{final_h})...\n"
            )

        # Build ffmpeg filter complex
        # 1. Scale original input to match output resolution using lanczos for quality
        # 2. Add labels to both videos
        # 3. Merge based on layout
        safe_label_input = _escape_drawtext_text(label_input)
        safe_label_output = _escape_drawtext_text(label_output)
        safe_brand_text = _escape_drawtext_text(COMPARISON_BRAND_TEXT)
        safe_font_size = _coerce_font_size(font_size, default=32)

        if layout == "horizontal":
            # Side by side - each video takes half the final width
            # Final video will be 2x the width of output video
            right_drawtext = (
                f"[1:v]scale={base_w}:{base_h}:flags=lanczos,"
                f"drawtext=text='{safe_label_output}':x=10:y=10:fontsize={safe_font_size}:fontcolor=white:"
                f"box=1:boxcolor=black@0.6:boxborderw=5"
            )
            if include_branding:
                right_drawtext += (
                    f",drawtext=text='{safe_brand_text}':x=w-tw-{COMPARISON_BRAND_RIGHT_PAD}:y=10:"
                    f"fontsize={safe_font_size}:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=5"
                )
            right_drawtext += "[right];"

            stacked_filter = (
                f"[0:v]scale={base_w}:{base_h}:flags=lanczos,"
                f"drawtext=text='{safe_label_input}':x=10:y=10:fontsize={safe_font_size}:fontcolor=white:"
                f"box=1:boxcolor=black@0.6:boxborderw=5[left];"
                f"{right_drawtext}"
                f"[left][right]hstack=inputs=2[stacked]"
            )
        else:
            # Stacked (vertical) - each video takes half the final height
            # Final video will be 2x the height of output video
            bottom_drawtext = (
                f"[1:v]scale={base_w}:{base_h}:flags=lanczos,"
                f"drawtext=text='{safe_label_output}':x=10:y=10:fontsize={safe_font_size}:fontcolor=white:"
                f"box=1:boxcolor=black@0.6:boxborderw=5"
            )
            if include_branding:
                bottom_drawtext += (
                    f",drawtext=text='{safe_brand_text}':x=w-tw-{COMPARISON_BRAND_RIGHT_PAD}:y=10:"
                    f"fontsize={safe_font_size}:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=5"
                )
            bottom_drawtext += "[bottom];"

            stacked_filter = (
                f"[0:v]scale={base_w}:{base_h}:flags=lanczos,"
                f"drawtext=text='{safe_label_input}':x=10:y=10:fontsize={safe_font_size}:fontcolor=white:"
                f"box=1:boxcolor=black@0.6:boxborderw=5[top];"
                f"{bottom_drawtext}"
                f"[top][bottom]vstack=inputs=2[stacked]"
            )

        if use_custom_size:
            filter_complex = (
                f"{stacked_filter};"
                f"[stacked]scale={final_w}:{final_h}:flags=lanczos:force_original_aspect_ratio=decrease,"
                f"pad={final_w}:{final_h}:(ow-iw)/2:(oh-ih)/2:black[out]"
            )
        else:
            filter_complex = f"{stacked_filter};[stacked]null[out]"

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i", str(input_path),
            "-i", str(output_path),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-map", "1:a?",  # Use audio from upscaled output if available
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(comparison_output)
        ]

        if on_progress:
            on_progress(f"Running ffmpeg for comparison video...\n")

        duration_seconds = (
            get_media_duration_seconds(str(output_path))
            or get_media_duration_seconds(str(input_path))
            or 0.0
        )
        result_code, stderr_tail = _run_ffmpeg_with_progress(
            cmd=cmd,
            timeout_seconds=1200,
            duration_seconds=duration_seconds,
            on_progress=on_progress,
        )

        if result_code == 0 and Path(comparison_output).exists():
            if on_progress:
                on_progress("COMPARISON_PROGRESS 100.0%\n")
                on_progress(f"Comparison video created: {comparison_output}\n")
            return True, comparison_output, ""
        else:
            error = stderr_tail or "Unknown ffmpeg error"
            return False, "", f"FFmpeg error: {error[:500]}"

    except subprocess.TimeoutExpired:
        return False, "", "Comparison video generation timed out (>20 minutes)"
    except Exception as e:
        return False, "", f"Error creating comparison video: {str(e)}"


def create_input_vs_output_slider_comparison_video(
    original_input_video: str,
    upscaled_output_video: str,
    comparison_output: Optional[str] = None,
    layout: str = "auto",
    label_input: str = "Original",
    label_output: str = "Upscaled",
    on_progress: Optional[callable] = None,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    font_size: int = 32,
    include_branding: bool = True,
    slider_pass_duration_seconds: Optional[float] = None,
) -> Tuple[bool, str, str]:
    """
    Create an animated slider comparison video (left->right then right->left).

    The output contains two passes:
    1) Reveal from left to right (or top to bottom for vertical layout)
    2) Reverse back to the start
    """
    from .path_utils import (
        normalize_path,
        get_media_dimensions,
        get_media_duration_seconds,
        get_media_fps,
    )
    from .error_handling import check_ffmpeg_available

    try:
        if not check_ffmpeg_available():
            return False, "", "FFmpeg not found in PATH"

        input_path = Path(normalize_path(original_input_video))
        output_path = Path(normalize_path(upscaled_output_video))

        if not input_path.exists():
            return False, "", f"Original input not found: {input_path}"
        if not output_path.exists():
            return False, "", f"Upscaled output not found: {output_path}"

        in_dims = get_media_dimensions(str(input_path))
        out_dims = get_media_dimensions(str(output_path))
        if not in_dims or not out_dims:
            return False, "", "Could not determine video dimensions"
        in_w, in_h = in_dims
        out_w, out_h = out_dims
        base_w, base_h = _resolve_pair_base_dimensions((in_w, in_h), (out_w, out_h))

        layout = normalize_comparison_layout(layout, base_w, base_h)
        custom_w = _coerce_even_dimension(target_width)
        custom_h = _coerce_even_dimension(target_height)
        use_custom_size = bool(custom_w and custom_h)
        final_w = custom_w if use_custom_size else base_w
        final_h = custom_h if use_custom_size else base_h

        if comparison_output is None:
            comparison_output = str(output_path.parent / f"{output_path.stem}_comparison_slider.mp4")
        else:
            comparison_output = normalize_path(comparison_output)
        Path(comparison_output).parent.mkdir(parents=True, exist_ok=True)

        dur_out = get_media_duration_seconds(str(output_path)) or 0.0
        dur_in = get_media_duration_seconds(str(input_path)) or 0.0
        source_duration = max(float(dur_out), float(dur_in), 0.0)
        try:
            pass_duration = float(slider_pass_duration_seconds or 0.0)
        except Exception:
            pass_duration = 0.0
        if pass_duration <= 0.0:
            pass_duration = float(source_duration or 0.0)
        pass_duration = max(0.5, pass_duration)
        source_fps = (
            get_media_fps(str(output_path))
            or get_media_fps(str(input_path))
            or 30.0
        )
        source_fps = max(1.0, float(source_fps))
        pass_frames = max(1, int(round(pass_duration * source_fps)))
        total_frames = pass_frames * 2
        pass_duration = pass_frames / source_fps
        total_duration = total_frames / source_fps

        if on_progress:
            on_progress(
                f"Creating slider comparison video ({layout} layout, {final_w}x{final_h}, "
                f"pass={pass_duration:.2f}s)...\n"
            )

        safe_label_input = _escape_drawtext_text(label_input)
        safe_label_output = _escape_drawtext_text(label_output)
        safe_brand_text = _escape_drawtext_text(COMPARISON_BRAND_TEXT)
        safe_font_size = _coerce_font_size(font_size, default=32)

        if layout == "vertical":
            frame_idx_expr = "max(N-1,0)"
            boundary_expr = (
                f"if(lt({frame_idx_expr},{pass_frames}),"
                f"{base_h}*({frame_idx_expr}/{float(pass_frames):.6f}),"
                f"if(lt({frame_idx_expr},{total_frames}),"
                f"{base_h}*(({float(total_frames):.6f}-{frame_idx_expr})/{float(pass_frames):.6f}),0))"
            )
            edge_min = 8
            edge_max = max(edge_min, int(base_h) - edge_min)
            split_expr = f"if(lt(Y,{boundary_expr}),A,B)"
            right_drawtext = (
                f"[1:v]scale={base_w}:{base_h}:flags=lanczos,fps={source_fps:.6f},format=gbrp,"
                f"drawtext=text='{safe_label_output}':x=10:y=10:fontsize={safe_font_size}:fontcolor=white:"
                f"box=1:boxcolor=black@0.6:boxborderw=5"
            )
            if include_branding:
                right_drawtext += (
                    f",drawtext=text='{safe_brand_text}':x=w-tw-{COMPARISON_BRAND_RIGHT_PAD}:y=10:"
                    f"fontsize={safe_font_size}:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=5"
                )
            right_drawtext += "[right];"

            blend_expr = (
                f"if(between({boundary_expr},{edge_min},{edge_max}),"
                f"if(lte(abs(Y-({boundary_expr})),2),255,"
                f"if(lte(abs(Y-({boundary_expr})),6),20,{split_expr})),"
                f"{split_expr})"
            )
            base_filter = (
                f"[0:v]scale={base_w}:{base_h}:flags=lanczos,fps={source_fps:.6f},format=gbrp,"
                f"drawtext=text='{safe_label_input}':x=10:y=10:fontsize={safe_font_size}:fontcolor=white:"
                f"box=1:boxcolor=black@0.6:boxborderw=5[left];"
                f"{right_drawtext}"
                f"[left][right]blend=all_expr='{blend_expr}'[slider]"
            )
        else:
            frame_idx_expr = "max(N-1,0)"
            boundary_expr = (
                f"if(lt({frame_idx_expr},{pass_frames}),"
                f"{base_w}*({frame_idx_expr}/{float(pass_frames):.6f}),"
                f"if(lt({frame_idx_expr},{total_frames}),"
                f"{base_w}*(({float(total_frames):.6f}-{frame_idx_expr})/{float(pass_frames):.6f}),0))"
            )
            edge_min = 8
            edge_max = max(edge_min, int(base_w) - edge_min)
            split_expr = f"if(lt(X,{boundary_expr}),A,B)"
            right_drawtext = (
                f"[1:v]scale={base_w}:{base_h}:flags=lanczos,fps={source_fps:.6f},format=gbrp,"
                f"drawtext=text='{safe_label_output}':x=10:y=10:fontsize={safe_font_size}:fontcolor=white:"
                f"box=1:boxcolor=black@0.6:boxborderw=5"
            )
            if include_branding:
                right_drawtext += (
                    f",drawtext=text='{safe_brand_text}':x=w-tw-{COMPARISON_BRAND_RIGHT_PAD}:y=10:"
                    f"fontsize={safe_font_size}:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=5"
                )
            right_drawtext += "[right];"

            blend_expr = (
                f"if(between({boundary_expr},{edge_min},{edge_max}),"
                f"if(lte(abs(X-({boundary_expr})),2),255,"
                f"if(lte(abs(X-({boundary_expr})),6),20,{split_expr})),"
                f"{split_expr})"
            )
            base_filter = (
                f"[0:v]scale={base_w}:{base_h}:flags=lanczos,fps={source_fps:.6f},format=gbrp,"
                f"drawtext=text='{safe_label_input}':x=10:y=10:fontsize={safe_font_size}:fontcolor=white:"
                f"box=1:boxcolor=black@0.6:boxborderw=5[left];"
                f"{right_drawtext}"
                f"[left][right]blend=all_expr='{blend_expr}'[slider]"
            )

        if use_custom_size:
            filter_complex = (
                f"{base_filter};"
                f"[slider]scale={final_w}:{final_h}:flags=lanczos:force_original_aspect_ratio=decrease,"
                f"pad={final_w}:{final_h}:(ow-iw)/2:(oh-ih)/2:black[out]"
            )
        else:
            filter_complex = f"{base_filter};[slider]null[out]"

        cmd = [
            "ffmpeg",
            "-y",
            "-stream_loop",
            "-1",
            "-i",
            str(input_path),
            "-stream_loop",
            "-1",
            "-i",
            str(output_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-map",
            "1:a?",
            "-t",
            f"{total_duration:.6f}",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-r",
            f"{source_fps:.6f}",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(comparison_output),
        ]

        if on_progress:
            on_progress(f"Running ffmpeg for slider comparison video...\n")

        result_code, stderr_tail = _run_ffmpeg_with_progress(
            cmd=cmd,
            timeout_seconds=1200,
            duration_seconds=total_duration,
            on_progress=on_progress,
        )

        if result_code == 0 and Path(comparison_output).exists():
            if on_progress:
                on_progress("COMPARISON_PROGRESS 100.0%\n")
                on_progress(f"Slider comparison video created: {comparison_output}\n")
            return True, comparison_output, ""

        error = stderr_tail or "Unknown ffmpeg error"
        return False, "", f"FFmpeg error: {error[:500]}"

    except subprocess.TimeoutExpired:
        return False, "", "Slider comparison video generation timed out (>20 minutes)"
    except Exception as e:
        return False, "", f"Error creating slider comparison video: {str(e)}"


def create_input_vs_output_comparison_preview_image(
    original_input_video: str,
    upscaled_output_video: str,
    preview_output: Optional[str] = None,
    layout: str = "auto",
    label_input: str = "Original",
    label_output: str = "Upscaled",
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    font_size: int = 32,
    include_branding: bool = True,
    slider_mode: bool = False,
    slider_pass_duration_seconds: Optional[float] = None,
    on_progress: Optional[callable] = None,
) -> Tuple[bool, str, str]:
    """
    Render a fast first-frame preview image for comparison settings.
    """
    from .path_utils import (
        normalize_path,
        get_media_dimensions,
        get_media_duration_seconds,
        get_media_fps,
    )
    from .error_handling import check_ffmpeg_available

    try:
        if not check_ffmpeg_available():
            return False, "", "FFmpeg not found in PATH"

        input_path = Path(normalize_path(original_input_video))
        output_path = Path(normalize_path(upscaled_output_video))

        if not input_path.exists():
            return False, "", f"Original input not found: {input_path}"
        if not output_path.exists():
            return False, "", f"Upscaled output not found: {output_path}"

        in_dims = get_media_dimensions(str(input_path))
        out_dims = get_media_dimensions(str(output_path))
        if not in_dims or not out_dims:
            return False, "", "Could not determine video dimensions"
        in_w, in_h = in_dims
        out_w, out_h = out_dims
        base_w, base_h = _resolve_pair_base_dimensions((in_w, in_h), (out_w, out_h))

        layout = normalize_comparison_layout(layout, base_w, base_h)
        custom_w = _coerce_even_dimension(target_width)
        custom_h = _coerce_even_dimension(target_height)
        use_custom_size = bool(custom_w and custom_h)
        if slider_mode:
            final_w = custom_w if use_custom_size else base_w
            final_h = custom_h if use_custom_size else base_h
        else:
            _resolved_layout, auto_w, auto_h = predict_comparison_dimensions(base_w, base_h, layout)
            final_w = custom_w if use_custom_size else auto_w
            final_h = custom_h if use_custom_size else auto_h

        if preview_output is None:
            preview_output = str(output_path.parent / f"{output_path.stem}_comparison_preview.png")
        else:
            preview_output = normalize_path(preview_output)
        Path(preview_output).parent.mkdir(parents=True, exist_ok=True)

        safe_label_input = _escape_drawtext_text(label_input)
        safe_label_output = _escape_drawtext_text(label_output)
        safe_brand_text = _escape_drawtext_text(COMPARISON_BRAND_TEXT)
        safe_font_size = _coerce_font_size(font_size, default=32)

        if slider_mode:
            dur_out = get_media_duration_seconds(str(output_path)) or 0.0
            dur_in = get_media_duration_seconds(str(input_path)) or 0.0
            source_duration = max(float(dur_out), float(dur_in), 0.0)
            try:
                pass_duration = float(slider_pass_duration_seconds or 0.0)
            except Exception:
                pass_duration = 0.0
            if pass_duration <= 0.0:
                pass_duration = float(source_duration or 0.0)
            pass_duration = max(0.5, pass_duration)
            source_fps = (
                get_media_fps(str(output_path))
                or get_media_fps(str(input_path))
                or 30.0
            )
            source_fps = max(1.0, float(source_fps))
            pass_frames = max(1, int(round(pass_duration * source_fps)))
            total_frames = pass_frames * 2
            pass_duration = pass_frames / source_fps
            total_duration = total_frames / source_fps

            if layout == "vertical":
                frame_idx_expr = "max(N-1,0)"
                boundary_expr = (
                    f"if(lt({frame_idx_expr},{pass_frames}),"
                    f"{base_h}*({frame_idx_expr}/{float(pass_frames):.6f}),"
                    f"if(lt({frame_idx_expr},{total_frames}),"
                    f"{base_h}*(({float(total_frames):.6f}-{frame_idx_expr})/{float(pass_frames):.6f}),0))"
                )
                edge_min = 8
                edge_max = max(edge_min, int(base_h) - edge_min)
                split_expr = f"if(lt(Y,{boundary_expr}),A,B)"
                right_drawtext = (
                    f"[1:v]scale={base_w}:{base_h}:flags=lanczos,fps={source_fps:.6f},format=gbrp,"
                    f"drawtext=text='{safe_label_output}':x=10:y=10:fontsize={safe_font_size}:fontcolor=white:"
                    f"box=1:boxcolor=black@0.6:boxborderw=5"
                )
                if include_branding:
                    right_drawtext += (
                        f",drawtext=text='{safe_brand_text}':x=w-tw-{COMPARISON_BRAND_RIGHT_PAD}:y=10:"
                        f"fontsize={safe_font_size}:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=5"
                    )
                right_drawtext += "[right];"
                blend_expr = (
                    f"if(between({boundary_expr},{edge_min},{edge_max}),"
                    f"if(lte(abs(Y-({boundary_expr})),2),255,"
                    f"if(lte(abs(Y-({boundary_expr})),6),20,{split_expr})),"
                    f"{split_expr})"
                )
                base_filter = (
                    f"[0:v]scale={base_w}:{base_h}:flags=lanczos,fps={source_fps:.6f},format=gbrp,"
                    f"drawtext=text='{safe_label_input}':x=10:y=10:fontsize={safe_font_size}:fontcolor=white:"
                    f"box=1:boxcolor=black@0.6:boxborderw=5[left];"
                    f"{right_drawtext}"
                    f"[left][right]blend=all_expr='{blend_expr}'[slider]"
                )
            else:
                frame_idx_expr = "max(N-1,0)"
                boundary_expr = (
                    f"if(lt({frame_idx_expr},{pass_frames}),"
                    f"{base_w}*({frame_idx_expr}/{float(pass_frames):.6f}),"
                    f"if(lt({frame_idx_expr},{total_frames}),"
                    f"{base_w}*(({float(total_frames):.6f}-{frame_idx_expr})/{float(pass_frames):.6f}),0))"
                )
                edge_min = 8
                edge_max = max(edge_min, int(base_w) - edge_min)
                split_expr = f"if(lt(X,{boundary_expr}),A,B)"
                right_drawtext = (
                    f"[1:v]scale={base_w}:{base_h}:flags=lanczos,fps={source_fps:.6f},format=gbrp,"
                    f"drawtext=text='{safe_label_output}':x=10:y=10:fontsize={safe_font_size}:fontcolor=white:"
                    f"box=1:boxcolor=black@0.6:boxborderw=5"
                )
                if include_branding:
                    right_drawtext += (
                        f",drawtext=text='{safe_brand_text}':x=w-tw-{COMPARISON_BRAND_RIGHT_PAD}:y=10:"
                        f"fontsize={safe_font_size}:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=5"
                    )
                right_drawtext += "[right];"
                blend_expr = (
                    f"if(between({boundary_expr},{edge_min},{edge_max}),"
                    f"if(lte(abs(X-({boundary_expr})),2),255,"
                    f"if(lte(abs(X-({boundary_expr})),6),20,{split_expr})),"
                    f"{split_expr})"
                )
                base_filter = (
                    f"[0:v]scale={base_w}:{base_h}:flags=lanczos,fps={source_fps:.6f},format=gbrp,"
                    f"drawtext=text='{safe_label_input}':x=10:y=10:fontsize={safe_font_size}:fontcolor=white:"
                    f"box=1:boxcolor=black@0.6:boxborderw=5[left];"
                    f"{right_drawtext}"
                    f"[left][right]blend=all_expr='{blend_expr}'[slider]"
                )
            if use_custom_size:
                filter_complex = (
                    f"{base_filter};"
                    f"[slider]scale={final_w}:{final_h}:flags=lanczos:force_original_aspect_ratio=decrease,"
                    f"pad={final_w}:{final_h}:(ow-iw)/2:(oh-ih)/2:black[out]"
                )
            else:
                filter_complex = f"{base_filter};[slider]null[out]"
        else:
            if layout == "horizontal":
                right_drawtext = (
                    f"[1:v]scale={base_w}:{base_h}:flags=lanczos,"
                    f"drawtext=text='{safe_label_output}':x=10:y=10:fontsize={safe_font_size}:fontcolor=white:"
                    f"box=1:boxcolor=black@0.6:boxborderw=5"
                )
                if include_branding:
                    right_drawtext += (
                        f",drawtext=text='{safe_brand_text}':x=w-tw-{COMPARISON_BRAND_RIGHT_PAD}:y=10:"
                        f"fontsize={safe_font_size}:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=5"
                    )
                right_drawtext += "[right];"
                stacked_filter = (
                    f"[0:v]scale={base_w}:{base_h}:flags=lanczos,"
                    f"drawtext=text='{safe_label_input}':x=10:y=10:fontsize={safe_font_size}:fontcolor=white:"
                    f"box=1:boxcolor=black@0.6:boxborderw=5[left];"
                    f"{right_drawtext}"
                    f"[left][right]hstack=inputs=2[stacked]"
                )
            else:
                bottom_drawtext = (
                    f"[1:v]scale={base_w}:{base_h}:flags=lanczos,"
                    f"drawtext=text='{safe_label_output}':x=10:y=10:fontsize={safe_font_size}:fontcolor=white:"
                    f"box=1:boxcolor=black@0.6:boxborderw=5"
                )
                if include_branding:
                    bottom_drawtext += (
                        f",drawtext=text='{safe_brand_text}':x=w-tw-{COMPARISON_BRAND_RIGHT_PAD}:y=10:"
                        f"fontsize={safe_font_size}:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=5"
                    )
                bottom_drawtext += "[bottom];"
                stacked_filter = (
                    f"[0:v]scale={base_w}:{base_h}:flags=lanczos,"
                    f"drawtext=text='{safe_label_input}':x=10:y=10:fontsize={safe_font_size}:fontcolor=white:"
                    f"box=1:boxcolor=black@0.6:boxborderw=5[top];"
                    f"{bottom_drawtext}"
                    f"[top][bottom]vstack=inputs=2[stacked]"
                )
            if use_custom_size:
                filter_complex = (
                    f"{stacked_filter};"
                    f"[stacked]scale={final_w}:{final_h}:flags=lanczos:force_original_aspect_ratio=decrease,"
                    f"pad={final_w}:{final_h}:(ow-iw)/2:(oh-ih)/2:black[out]"
                )
            else:
                filter_complex = f"{stacked_filter};[stacked]null[out]"

        if on_progress:
            mode_txt = "slider" if slider_mode else "merged"
            on_progress(f"Rendering {mode_txt} preview frame...\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-i",
            str(output_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-frames:v",
            "1",
            str(preview_output),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode == 0 and Path(preview_output).exists() and Path(preview_output).stat().st_size > 0:
            return True, str(preview_output), ""
        err = result.stderr or result.stdout or "Unknown ffmpeg error"
        return False, "", f"FFmpeg preview error: {err[:500]}"
    except Exception as e:
        return False, "", f"Error creating comparison preview image: {str(e)}"


def get_smart_comparison_layout(width: int, height: int) -> str:
    """
    Determine the best comparison layout based on output dimensions.

    The goal is to produce a comparison video with aspect ratio closest to 16:9
    since most monitors are 16:9.

    Args:
        width: Output video width
        height: Output video height

    Returns:
        "horizontal" or "vertical" depending on which produces closer to 16:9
    """
    target_ratio = 16.0 / 9.0  # 1.778

    # Calculate resulting aspect ratios for each merge option
    horizontal_ratio = (2 * width) / height if height > 0 else 2.0
    vertical_ratio = width / (2 * height) if height > 0 else 0.5

    # Choose whichever is closer to 16:9
    if abs(horizontal_ratio - target_ratio) <= abs(vertical_ratio - target_ratio):
        return "horizontal"
    else:
        return "vertical"

