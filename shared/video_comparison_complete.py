"""
Complete Video Comparison Implementation

Provides video comparison using HTML5 video elements with synchronized playback.
Since Gradio doesn't have native VideoSlider (only ImageSlider), we create a custom HTML component.
"""

from pathlib import Path
from typing import Optional, Tuple
import base64


def create_video_comparison_html(
    video_left: Optional[str],
    video_right: Optional[str],
    width: int = 1280,
    height: int = 720,
    slider_position: float = 50.0
) -> str:
    """
    Create HTML for side-by-side video comparison with synchronized playback.
    
    Args:
        video_left: Path to left video (original)
        video_right: Path to right video (upscaled)
        width: Container width in pixels
        height: Container height in pixels
        slider_position: Initial slider position (0-100)
        
    Returns:
        HTML string for embedding in gr.HTML component
    """
    if not video_left or not video_right:
        return """
        <div style="text-align: center; padding: 40px; background: #f5f5f5; border-radius: 8px;">
            <p style="color: #666; font-size: 16px;">
                No videos to compare. Process a video first.
            </p>
        </div>
        """
    
    # Convert paths to data URLs or file:// URLs for local playback
    left_url = _get_video_url(video_left)
    right_url = _get_video_url(video_right)
    
    html = f"""
    <div id="video-comparison-container" style="position: relative; width: {width}px; max-width: 100%; margin: 0 auto; background: #000;">
        <div style="position: relative; width: 100%; height: {height}px; overflow: hidden;">
            <!-- Left video (original) -->
            <video id="video-left" 
                   style="position: absolute; left: 0; top: 0; width: 100%; height: 100%; object-fit: contain;"
                   src="{left_url}">
                Your browser does not support the video tag.
            </video>
            
            <!-- Right video (upscaled) with clip-path -->
            <video id="video-right"
                   style="position: absolute; left: 0; top: 0; width: 100%; height: 100%; object-fit: contain; clip-path: inset(0 0 0 {slider_position}%);"
                   src="{right_url}">
                Your browser does not support the video tag.
            </video>
            
            <!-- Slider line -->
            <div id="slider-line" 
                 style="position: absolute; left: {slider_position}%; top: 0; width: 3px; height: 100%; background: #fff; cursor: ew-resize; box-shadow: 0 0 10px rgba(0,0,0,0.5); z-index: 10;">
                <div style="position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); width: 40px; height: 40px; background: #fff; border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 10px rgba(0,0,0,0.3);">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#333" stroke-width="2">
                        <path d="M15 18l-6-6 6-6"/>
                        <path d="M9 18l6-6-6-6"/>
                    </svg>
                </div>
            </div>
        </div>
        
        <!-- Controls -->
        <div style="display: flex; gap: 10px; padding: 15px; background: #1a1a1a; align-items: center; justify-content: center; flex-wrap: wrap;">
            <button id="play-pause-btn" style="padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 14px;">
                ▶ Play
            </button>
            <button id="restart-btn" style="padding: 10px 20px; background: #2196F3; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 14px;">
                ⟲ Restart
            </button>
            <div style="display: flex; gap: 10px; align-items: center;">
                <label style="color: #fff; font-size: 14px;">Speed:</label>
                <select id="playback-speed" style="padding: 8px; border-radius: 5px; border: 1px solid #555; background: #333; color: #fff;">
                    <option value="0.25">0.25x</option>
                    <option value="0.5">0.5x</option>
                    <option value="1" selected>1x</option>
                    <option value="1.5">1.5x</option>
                    <option value="2">2x</option>
                </select>
            </div>
            <div style="flex: 1; display: flex; align-items: center; gap: 10px; min-width: 200px;">
                <span id="current-time" style="color: #fff; font-size: 14px;">0:00</span>
                <input type="range" id="timeline" min="0" max="100" value="0" 
                       style="flex: 1; height: 4px; border-radius: 2px; background: #555; outline: none;">
                <span id="duration-time" style="color: #fff; font-size: 14px;">0:00</span>
            </div>
            <button id="fullscreen-btn" style="padding: 10px 20px; background: #FF9800; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 14px;">
                ⛶ Fullscreen
            </button>
        </div>
        
        <!-- Labels -->
        <div style="display: flex; justify-content: space-between; padding: 10px 15px; background: #2a2a2a;">
            <span style="color: #fff; font-size: 13px;">◀ Original</span>
            <span style="color: #fff; font-size: 13px;">Upscaled ▶</span>
        </div>
    </div>

    <script>
    (function() {{
        const container = document.getElementById('video-comparison-container');
        const videoLeft = document.getElementById('video-left');
        const videoRight = document.getElementById('video-right');
        const sliderLine = document.getElementById('slider-line');
        const playPauseBtn = document.getElementById('play-pause-btn');
        const restartBtn = document.getElementById('restart-btn');
        const playbackSpeed = document.getElementById('playback-speed');
        const timeline = document.getElementById('timeline');
        const currentTimeSpan = document.getElementById('current-time');
        const durationTimeSpan = document.getElementById('duration-time');
        const fullscreenBtn = document.getElementById('fullscreen-btn');
        
        let isDragging = false;
        let sliderPosition = {slider_position};
        
        // Synchronize playback
        function syncVideos() {{
            videoRight.currentTime = videoLeft.currentTime;
        }}
        
        // Update slider visual
        function updateSlider(percent) {{
            sliderPosition = Math.max(0, Math.min(100, percent));
            sliderLine.style.left = sliderPosition + '%';
            videoRight.style.clipPath = `inset(0 0 0 ${{sliderPosition}}%)`;
        }}
        
        // Slider dragging
        function startDrag(e) {{
            isDragging = true;
            updateSliderPosition(e);
        }}
        
        function stopDrag() {{
            isDragging = false;
        }}
        
        function updateSliderPosition(e) {{
            if (!isDragging) return;
            
            const rect = container.querySelector('div').getBoundingClientRect();
            const x = (e.clientX || e.touches[0].clientX) - rect.left;
            const percent = (x / rect.width) * 100;
            updateSlider(percent);
        }}
        
        // Event listeners for dragging
        sliderLine.addEventListener('mousedown', startDrag);
        document.addEventListener('mouseup', stopDrag);
        document.addEventListener('mousemove', updateSliderPosition);
        sliderLine.addEventListener('touchstart', startDrag);
        document.addEventListener('touchend', stopDrag);
        document.addEventListener('touchmove', updateSliderPosition);
        
        // Play/Pause
        playPauseBtn.addEventListener('click', () => {{
            if (videoLeft.paused) {{
                videoLeft.play();
                videoRight.play();
                playPauseBtn.textContent = '⏸ Pause';
            }} else {{
                videoLeft.pause();
                videoRight.pause();
                playPauseBtn.textContent = '▶ Play';
            }}
        }});
        
        // Restart
        restartBtn.addEventListener('click', () => {{
            videoLeft.currentTime = 0;
            videoRight.currentTime = 0;
            syncVideos();
        }});
        
        // Playback speed
        playbackSpeed.addEventListener('change', (e) => {{
            const speed = parseFloat(e.target.value);
            videoLeft.playbackRate = speed;
            videoRight.playbackRate = speed;
        }});
        
        // Timeline
        timeline.addEventListener('input', (e) => {{
            const percent = parseFloat(e.target.value);
            const time = (percent / 100) * videoLeft.duration;
            videoLeft.currentTime = time;
            syncVideos();
        }});
        
        // Update timeline as video plays
        videoLeft.addEventListener('timeupdate', () => {{
            if (!isDragging) {{
                const percent = (videoLeft.currentTime / videoLeft.duration) * 100;
                timeline.value = percent;
                currentTimeSpan.textContent = formatTime(videoLeft.currentTime);
            }}
            syncVideos();
        }});
        
        // Set duration
        videoLeft.addEventListener('loadedmetadata', () => {{
            durationTimeSpan.textContent = formatTime(videoLeft.duration);
        }});
        
        // Format time
        function formatTime(seconds) {{
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return mins + ':' + (secs < 10 ? '0' : '') + secs;
        }}
        
        // Fullscreen
        fullscreenBtn.addEventListener('click', () => {{
            if (container.requestFullscreen) {{
                container.requestFullscreen();
            }} else if (container.webkitRequestFullscreen) {{
                container.webkitRequestFullscreen();
            }} else if (container.msRequestFullscreen) {{
                container.msRequestFullscreen();
            }}
        }});
        
        // Sync on play
        videoLeft.addEventListener('play', () => videoRight.play());
        videoLeft.addEventListener('pause', () => videoRight.pause());
        videoLeft.addEventListener('seeking', syncVideos);
        
        // Auto-sync periodically (fallback)
        setInterval(syncVideos, 100);
    }})();
    </script>
    """
    
    return html


def _get_video_url(video_path: str) -> str:
    """
    Convert video path to URL usable in HTML.
    
    For local files, returns file:// URL.
    For already-served files, returns as-is.
    """
    path = Path(video_path)
    
    if path.exists():
        # Use file:// URL for local files
        # Note: This works in most browsers but may have security restrictions
        return f"file:///{path.as_posix()}"
    else:
        # Assume it's already a URL
        return video_path


def create_image_comparison(
    image_left: Optional[str],
    image_right: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Prepare images for gr.ImageSlider component.
    
    Args:
        image_left: Path to left image (original)
        image_right: Path to right image (upscaled)
        
    Returns:
        Tuple of (left_path, right_path) suitable for ImageSlider value
    """
    if not image_left or not image_right:
        return (None, None)
    
    # Gradio ImageSlider accepts file paths directly
    return (image_left, image_right)


def build_comparison_layout(
    output_path: Optional[str],
    input_path: Optional[str],
    input_type: str
) -> dict:
    """
    Build comparison data for displaying in UI.
    
    Args:
        output_path: Path to upscaled output
        input_path: Path to original input
        input_type: Type of input ("video" or "image")
        
    Returns:
        Dictionary with comparison data
    """
    result = {
        "has_comparison": False,
        "type": input_type,
        "video_html": "",
        "image_tuple": (None, None),
        "message": ""
    }
    
    if not output_path or not input_path:
        result["message"] = "No files to compare"
        return result
    
    if input_type == "video":
        result["has_comparison"] = True
        result["video_html"] = create_video_comparison_html(input_path, output_path)
        result["message"] = "Video comparison ready (use slider to compare)"
    elif input_type == "image":
        result["has_comparison"] = True
        result["image_tuple"] = create_image_comparison(input_path, output_path)
        result["message"] = "Image comparison ready"
    else:
        result["message"] = "Unsupported file type for comparison"
    
    return result

