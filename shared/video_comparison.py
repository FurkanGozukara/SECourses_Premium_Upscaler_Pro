"""
Advanced video and image comparison using Gradio's latest features.

Features:
- Native Gradio ImageSlider for images with enhanced controls
- Custom video comparison with slider controls
- Side-by-side and overlay comparison modes
- Fullscreen support and export capabilities
- Responsive design for different screen sizes
"""

import gradio as gr
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


def create_image_comparison(
    input_image: Optional[str],
    output_image: Optional[str],
    label: str = "Before/After Comparison"
):
    """
    Create an enhanced image comparison update for Gradio's ImageSlider.

    Features:
    - Native slider with smooth transitions
    - Position control for precise comparison
    - Export and fullscreen capabilities
    - Responsive design
    
    Returns:
        gr.update() object for ImageSlider component
    """
    if not input_image or not output_image:
        # Return update to hide slider when no images available
        return gr.update(visible=False)

    return gr.update(
        value=(input_image, output_image),
        visible=True
    )


def create_video_comparison_html(
    input_video: Optional[str],
    output_video: Optional[str],
    width: int = 800,
    height: int = 450
) -> str:
    """
    Create a custom HTML video comparison with slider controls.

    Since Gradio doesn't have native VideoSlider, we create a custom
    HTML implementation with advanced features.
    """
    if not input_video or not output_video:
        return '<div class="no-comparison"><p>No videos available for comparison</p></div>'

    # Normalize paths for web display
    input_path = str(Path(input_video)).replace('\\', '/')
    output_path = str(Path(output_video)).replace('\\', '/')

    html = f"""
    <div class="video-comparison-container" style="width: 100%; max-width: {width}px; margin: 0 auto;">
        <div class="video-wrapper" style="position: relative; width: 100%; height: {height}px; background: #000; border-radius: 8px; overflow: hidden; border: 1px solid #333;">
            <!-- Input video (background) -->
            <video id="input-video" style="width: 100%; height: 100%; object-fit: contain;" muted>
                <source src="file:///{input_path}" type="video/mp4">
                Your browser does not support video comparison.
            </video>

            <!-- Output video (overlay) -->
            <video id="output-video" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; clip-path: inset(0 50% 0 0);" muted>
                <source src="file:///{output_path}" type="video/mp4">
            </video>

            <!-- Slider control -->
            <input type="range" id="comparison-slider" min="0" max="100" value="50"
                   style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                          width: 80%; height: 6px; background: rgba(255,255,255,0.3);
                          border-radius: 3px; cursor: ew-resize; z-index: 10;">

            <!-- Label overlay -->
            <div class="comparison-label" style="position: absolute; top: 10px; left: 10px;
                        background: rgba(0,0,0,0.7); color: white; padding: 4px 8px;
                        border-radius: 4px; font-size: 12px; z-index: 5;">
                Drag slider to compare
            </div>
        </div>

        <!-- Control buttons -->
        <div class="video-controls" style="display: flex; gap: 10px; margin-top: 10px; justify-content: center;">
            <button id="play-btn" style="padding: 8px 16px; background: #007bff; color: white;
                    border: none; border-radius: 4px; cursor: pointer;">Play Both</button>
            <button id="pause-btn" style="padding: 8px 16px; background: #6c757d; color: white;
                    border: none; border-radius: 4px; cursor: pointer;">Pause Both</button>
            <button id="reset-btn" style="padding: 8px 16px; background: #28a745; color: white;
                    border: none; border-radius: 4px; cursor: pointer;">Reset Slider</button>
        </div>

        <script>
        (function() {{
            const inputVideo = document.getElementById('input-video');
            const outputVideo = document.getElementById('output-video');
            const slider = document.getElementById('comparison-slider');
            const playBtn = document.getElementById('play-btn');
            const pauseBtn = document.getElementById('pause-btn');
            const resetBtn = document.getElementById('reset-btn');

            // Sync video playback
            function syncVideos() {{
                outputVideo.currentTime = inputVideo.currentTime;
            }}

            inputVideo.addEventListener('timeupdate', syncVideos);
            inputVideo.addEventListener('play', () => outputVideo.play());
            inputVideo.addEventListener('pause', () => outputVideo.pause());

            // Slider control
            slider.addEventListener('input', function(e) {{
                const percentage = e.target.value;
                outputVideo.style.clipPath = `inset(0 ${{100 - percentage}}% 0 0)`;
            }});

            // Control buttons
            playBtn.addEventListener('click', function() {{
                inputVideo.play();
                outputVideo.play();
            }});

            pauseBtn.addEventListener('click', function() {{
                inputVideo.pause();
                outputVideo.pause();
            }});

            resetBtn.addEventListener('click', function() {{
                slider.value = 50;
                outputVideo.style.clipPath = 'inset(0 50% 0 0)';
            }});

            // Initialize
            slider.value = 50;
            outputVideo.style.clipPath = 'inset(0 50% 0 0)';
        }})();
        </script>

        <style>
        .video-comparison-container {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}

        .video-wrapper {{
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: box-shadow 0.3s ease;
        }}

        .video-wrapper:hover {{
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }}

        #comparison-slider:hover {{
            background: rgba(255,255,255,0.5);
        }}

        .video-controls button:hover {{
            opacity: 0.8;
            transform: translateY(-1px);
        }}

        .video-controls button:active {{
            transform: translateY(0);
        }}

        @media (max-width: 768px) {{
            .video-wrapper {{
                height: {int(height * 0.7)}px;
            }}

            .video-controls {{
                flex-wrap: wrap;
            }}

            .video-controls button {{
                flex: 1 1 100%;
                margin-bottom: 5px;
            }}
        }}
        </style>
    </div>
    """

    return html


def create_side_by_side_comparison(
    input_path: Optional[str],
    output_path: Optional[str],
    title: str = "Side-by-Side Comparison"
) -> str:
    """
    Create a side-by-side comparison layout.
    Works for both images and videos.
    """
    if not input_path or not output_path:
        return '<div class="no-comparison"><p>No content available for comparison</p></div>'

    input_ext = Path(input_path).suffix.lower()
    output_ext = Path(output_path).suffix.lower()

    is_video = input_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']

    if is_video:
        media_html = f"""
        <div class="side-by-side-video">
            <div class="video-item">
                <h4>Input Video</h4>
                <video controls style="width: 100%; max-height: 400px;">
                    <source src="file:///{input_path.replace(chr(92), '/')}" type="video/mp4">
                </video>
            </div>
            <div class="video-item">
                <h4>Output Video</h4>
                <video controls style="width: 100%; max-height: 400px;">
                    <source src="file:///{output_path.replace(chr(92), '/')}" type="video/mp4">
                </video>
            </div>
        </div>
        """
    else:
        media_html = f"""
        <div class="side-by-side-image">
            <div class="image-item">
                <h4>Input Image</h4>
                <img src="file:///{input_path.replace(chr(92), '/')}" style="width: 100%; max-height: 400px; object-fit: contain;" />
            </div>
            <div class="image-item">
                <h4>Output Image</h4>
                <img src="file:///{output_path.replace(chr(92), '/')}" style="width: 100%; max-height: 400px; object-fit: contain;" />
            </div>
        </div>
        """

    html = f"""
    <div class="comparison-container">
        <h3 style="text-align: center; margin-bottom: 20px;">{title}</h3>
        {media_html}
    </div>

    <style>
    .comparison-container {{
        width: 100%;
        max-width: 1200px;
        margin: 0 auto;
    }}

    .side-by-side-video, .side-by-side-image {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        align-items: start;
    }}

    .video-item, .image-item {{
        text-align: center;
    }}

    .video-item h4, .image-item h4 {{
        margin-bottom: 10px;
        color: #333;
        font-weight: 600;
    }}

    .video-item video, .image-item img {{
        border: 1px solid #ddd;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}

    @media (max-width: 768px) {{
        .side-by-side-video, .side-by-side-image {{
            grid-template-columns: 1fr;
            gap: 15px;
        }}
    }}
    </style>
    """

    return html


def create_comparison_selector(
    input_path: Optional[str],
    output_path: Optional[str],
    comparison_mode: str = "slider",
    pinned_reference_path: Optional[str] = None,
    pin_enabled: bool = False
) -> Tuple[str, Any]:
    """
    Create the appropriate comparison component based on mode and content type.
    
    Args:
        input_path: Path to input/original file
        output_path: Path to output/upscaled file
        comparison_mode: "slider", "side_by_side", or "stacked"
        pinned_reference_path: If set, use this as reference instead of input_path
        pin_enabled: Whether pin reference feature is active
    
    Returns:
        Tuple of (HTML content, gr.update() for ImageSlider or None)
    """
    if not output_path:
        return '<div class="no-comparison"><p>No output available for comparison</p></div>', gr.update(visible=False)
    
    # Use pinned reference if available and enabled
    effective_input = pinned_reference_path if (pin_enabled and pinned_reference_path) else input_path
    
    if not effective_input:
        return '<div class="no-comparison"><p>No reference available for comparison</p></div>', gr.update(visible=False)

    input_ext = Path(effective_input).suffix.lower()

    if input_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
        # Image comparison
        if comparison_mode == "slider":
            return "", create_image_comparison(effective_input, output_path)
        elif comparison_mode == "side_by_side":
            return create_side_by_side_comparison(effective_input, output_path), gr.update(visible=False)
        else:
            return create_side_by_side_comparison(effective_input, output_path), gr.update(visible=False)
    else:
        # Video comparison
        if comparison_mode == "slider":
            return create_video_comparison_html(effective_input, output_path), gr.update(visible=False)
        elif comparison_mode == "side_by_side":
            return create_side_by_side_comparison(effective_input, output_path), gr.update(visible=False)
        else:
            return create_video_comparison_html(effective_input, output_path), gr.update(visible=False)


def _pin_js(val: bool) -> str:
    return "true" if val else "false"


def build_video_comparison(
    input_video: str,
    output_video: str,
    pin_reference: bool = False,
    start_fullscreen: bool = False,
    use_fallback_assets: bool = False,
) -> str:
    if not input_video or not output_video:
        return "<p>No comparison available.</p>"
    if use_fallback_assets:
        try:
            from pathlib import Path

            # Serve the bundled slider HTML (assumes Video_Comparison_Slider assets are present)
            slider_html = Path(__file__).parent.parent / "Video_Comparison_Slider" / "Start_Slider_App.html"
            if slider_html.exists():
                with slider_html.open("r", encoding="utf-8") as f:
                    html = f.read()

                # Inject the video paths into the HTML
                # This assumes the HTML has placeholders or we inject via script
                inject_script = f"""
                <script>
                // Auto-load videos when page loads
                document.addEventListener('DOMContentLoaded', function() {{
                    // Find video input elements and set values
                    const videoInputs = document.querySelectorAll('input[type="file"], input[type="url"]');
                    if (videoInputs.length >= 2) {{
                        // Create data URLs or file paths for local files
                        const inputPath = "{input_video.replace(chr(92), '/')}";
                        const outputPath = "{output_video.replace(chr(92), '/')}";

                        // If there are URL inputs, use file:// protocol
                        const urlInputs = document.querySelectorAll('input[type="url"]');
                        if (urlInputs.length >= 2) {{
                            urlInputs[0].value = "file:///" + inputPath;
                            urlInputs[1].value = "file:///" + outputPath;
                        }}

                        // Try to auto-submit or trigger comparison
                        setTimeout(() => {{
                            const compareBtn = document.querySelector('button[id*="compare"], button:contains("Compare")');
                            if (compareBtn) {{
                                compareBtn.click();
                            }}
                        }}, 500);
                    }}
                }});
                </script>
                """

                # Insert the injection script before the closing body tag
                html = html.replace('</body>', inject_script + '</body>')
                return html
        except Exception as e:
            # Fallback to basic HTML if the slider file can't be loaded
            return f"<p>Video comparison slider unavailable: {str(e)}</p>"

    pin_js = _pin_js(pin_reference)
    fs_js = _pin_js(start_fullscreen)
    # Normalize paths for file:// usage
    safe_inp = input_video.replace("\\", "/")
    safe_out = output_video.replace("\\", "/")
    return f"""
<style>
.vid-cmp-shell {{
  width: 100%;
  max-width: 1280px;
  margin: 0 auto;
  font-family: Inter, system-ui, sans-serif;
  color: #eee;
}}
.vid-cmp-container {{
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #0f0f0f;
  overflow: hidden;
  border-radius: 12px;
  border: 1px solid #2b2b2b;
}}
.vid-base, .vid-overlay {{
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #000;
}}
.vid-overlay {{
  position: absolute;
  top: 0;
  left: 0;
  clip-path: inset(0 50% 0 0);
  pointer-events: none; /* Keep controls on base video */
}}
.vid-range {{
  width: 100%;
  margin-top: 10px;
  accent-color: #6d8bff;
}}
.vid-toolbar {{
  display: flex;
  gap: 8px;
  margin-top: 8px;
  flex-wrap: wrap;
}}
.vid-toolbar button, .vid-toolbar a {{
  flex: 1 1 auto;
  min-width: 120px;
  padding: 9px 10px;
  background: #1f2430;
  color: #eee;
  text-decoration: none;
  text-align: center;
  border: 1px solid #2f3650;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
}}
.vid-toolbar button:hover, .vid-toolbar a:hover {{
  border-color: #6d8bff;
}}
.vid-hint {{
  margin-top: 6px;
  font-size: 13px;
  color: #cfd7ff;
}}
</style>
<div class="vid-cmp-shell" id="vidCmpShell">
  <div class="vid-cmp-container" id="vidCmpRoot">
    <video class="vid-base" id="vidBase" src="file:///{safe_inp}" controls preload="metadata"></video>
    <video id="vidOverlay" class="vid-overlay" src="file:///{safe_out}" controls muted preload="metadata"></video>
  </div>
  <input id="vidRange" type="range" min="0" max="100" value="50" class="vid-range">
  <div class="vid-toolbar">
    <button id="swapBtn">Swap</button>
    <button id="pinBtn">Pin Reference</button>
    <button id="fsBtn">Fullscreen</button>
    <button id="resetBtn">Reset</button>
    <a id="popoutBtn" href="#" title="Open comparison in a new window">Pop-out</a>
  </div>
  <div class="vid-hint">
    Keys: ←/→ adjust slider, S swap, P toggle pin, F fullscreen, R reset.
  </div>
</div>
<script>
(() => {{
  const r = document.getElementById('vidRange');
  const o = document.getElementById('vidOverlay');
  const b = document.getElementById('vidBase');
  const swapBtn = document.getElementById('swapBtn');
  const pinBtn = document.getElementById('pinBtn');
  const fsBtn = document.getElementById('fsBtn');
  const resetBtn = document.getElementById('resetBtn');
  const popoutBtn = document.getElementById('popoutBtn');
  const root = document.getElementById('vidCmpRoot');
  const shell = document.getElementById('vidCmpShell');

  let pinned = {pin_js};

  const applyClip = (val) => {{
    o.style.clipPath = `inset(0 ${100 - val}% 0 0)`;
  }};

  const syncPinLabel = () => {{
    pinBtn.textContent = pinned ? "Unpin" : "Pin Reference";
  }};

  r.oninput = () => applyClip(r.value);
  applyClip(r.value);
  syncPinLabel();

  swapBtn.onclick = () => {{
    const tmp = o.src;
    o.src = b.src;
    b.src = tmp;
  }};

  pinBtn.onclick = () => {{
    pinned = !pinned;
    syncPinLabel();
    if (pinned) {{
      o.pause(); b.pause();
    }}
  }};

  fsBtn.onclick = () => {{
    if (root.requestFullscreen) root.requestFullscreen();
  }};

  resetBtn.onclick = () => {{
    r.value = 50;
    applyClip(50);
  }};

  popoutBtn.onclick = (e) => {{
    e.preventDefault();
    const popup = window.open("", "_blank", "width=1400,height=900");
    if (!popup) return;
    const tpl = shell.outerHTML;
    popup.document.write(`<!doctype html><html><head><title>Video Comparison</title></head><body style="margin:0;background:#0f0f0f;color:#eee;font-family:Inter,system-ui,sans-serif;">${{tpl}}</body></html>`);
    popup.document.close();
  }};

  // Keyboard shortcuts
  window.addEventListener('keydown', (ev) => {{
    const step = 2;
    if (ev.key === 'ArrowLeft') {{ r.value = Math.max(0, r.value - step); r.dispatchEvent(new Event('input')); }}
    if (ev.key === 'ArrowRight') {{ r.value = Math.min(100, parseInt(r.value) + step); r.dispatchEvent(new Event('input')); }}
    if (ev.key.toLowerCase() === 's') swapBtn.click();
    if (ev.key.toLowerCase() === 'p' || ev.key === ' ') {{ ev.preventDefault(); pinBtn.click(); }}
    if (ev.key.toLowerCase() === 'f') fsBtn.click();
    if (ev.key.toLowerCase() === 'r') resetBtn.click();
  }});

  // Initialize pin state
  if (pinned) {{
    o.pause(); b.pause();
  }}

  // Optional fullscreen on load
  if ({fs_js} && root.requestFullscreen) {{
    setTimeout(() => fsBtn.click(), 300);
  }}
}})();
</script>
"""


def build_image_comparison(input_path: str, output_path: str, pin_reference: bool = False) -> str:
    if not input_path or not output_path:
        return "<p>No comparison available.</p>"
    pin_js = _pin_js(pin_reference)
    return f"""
<style>
.img-cmp-shell {{
  width: 100%;
  max-width: 1080px;
  margin: 0 auto;
  font-family: Inter, system-ui, sans-serif;
}}
.img-cmp-container {{
  position: relative;
  width: 100%;
  overflow: hidden;
  border-radius: 10px;
  border: 1px solid #333;
}}
.img-base, .img-overlay {{
  width: 100%;
  display: block;
}}
.img-overlay {{
  position: absolute;
  top: 0;
  left: 0;
  clip-path: inset(0 50% 0 0);
}}
.img-range {{
  width: 100%;
  margin-top: 10px;
  accent-color: #08f;
}}
.img-toolbar {{
  display: flex;
  gap: 8px;
  margin-top: 8px;
  flex-wrap: wrap;
}}
.img-toolbar button {{
  flex: 1 1 auto;
  min-width: 120px;
  padding: 8px 10px;
  background: #1e1e1e;
  color: #eee;
  border: 1px solid #333;
  border-radius: 6px;
  cursor: pointer;
}}
.img-toolbar button:hover {{
  border-color: #08f;
}}
</style>
<div class="img-cmp-shell">
  <div class="img-cmp-container" id="imgCmpRoot">
    <img class="img-base" id="imgBase" src="file:///{input_path}" />
    <img class="img-overlay" id="imgOverlay" src="file:///{output_path}" />
  </div>
  <input id="imgRange" type="range" min="0" max="100" value="50" class="img-range">
  <div class="img-toolbar">
    <button id="imgSwapBtn">Swap</button>
    <button id="imgPinBtn">Pin Reference</button>
    <button id="imgResetBtn">Reset Slider</button>
  </div>
</div>
<script>
(() => {{
  const r = document.getElementById('imgRange');
  const o = document.getElementById('imgOverlay');
  const b = document.getElementById('imgBase');
  const swapBtn = document.getElementById('imgSwapBtn');
  const pinBtn = document.getElementById('imgPinBtn');
  const resetBtn = document.getElementById('imgResetBtn');

  let pinned = {pin_js};

  const applyClip = (val) => {{
    o.style.clipPath = `inset(0 ${100 - val}% 0 0)`;
  }};

  r.oninput = () => applyClip(r.value);
  applyClip(r.value);

  swapBtn.onclick = () => {{
    const tmp = o.src;
    o.src = b.src;
    b.src = tmp;
  }};

  pinBtn.onclick = () => {{
    pinned = !pinned;
    pinBtn.textContent = pinned ? "Unpin" : "Pin Reference";
  }};

  resetBtn.onclick = () => {{
    r.value = 50;
    applyClip(50);
  }};

  if (pinned) {{
    pinBtn.textContent = "Unpin";
  }}
}})();
</script>
"""

