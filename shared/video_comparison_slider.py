"""
Custom HTML5 video comparison slider for Gradio.

Notes:
- Gradio HTML updates do not execute inline <script> tags by default.
- `get_video_comparison_js_on_load()` enables safe execution for scripts included
  in updated HTML payloads.
"""

import hashlib
import html
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
from typing import Dict, Optional
import urllib.parse


def get_video_comparison_js_on_load() -> str:
    """
    JavaScript hook for gr.HTML(js_on_load=...).
    Executes inline scripts whenever HTML content changes.
    """
    return """
    const executeInlineScripts = () => {
        const scripts = element.querySelectorAll("script");
        scripts.forEach((scriptEl) => {
            if (scriptEl.dataset.grExecuted === "1") return;
            scriptEl.dataset.grExecuted = "1";
            const runner = document.createElement("script");
            if (scriptEl.type) runner.type = scriptEl.type;
            runner.textContent = scriptEl.textContent || "";
            document.body.appendChild(runner);
            document.body.removeChild(runner);
        });
    };

    const observer = new MutationObserver(() => executeInlineScripts());
    observer.observe(element, { childList: true, subtree: true });
    executeInlineScripts();
    """


def _get_gradio_upload_root() -> Path:
    """
    Resolve Gradio's upload/cache directory used by /gradio_api/file=.
    """
    try:
        from gradio.utils import get_upload_folder

        return Path(get_upload_folder()).resolve()
    except Exception:
        fallback = os.environ.get("GRADIO_TEMP_DIR") or str(
            (Path(tempfile.gettempdir()) / "gradio").resolve()
        )
        return Path(fallback).resolve()


def _ensure_gradio_servable_file(file_path: str) -> str:
    """
    Ensure a media file can be served by Gradio's file route.

    Gradio 6 blocks arbitrary absolute paths in custom HTML unless they are in
    allowed paths or in its upload/cache directory. We stage external files into
    the upload folder so `/gradio_api/file=...` works reliably.
    """
    src = Path(file_path).resolve()
    if not src.exists():
        return str(src)

    upload_root = _get_gradio_upload_root()
    try:
        src.relative_to(upload_root)
        return str(src)
    except ValueError:
        pass

    try:
        stat = src.stat()
        cache_key = f"{src.as_posix()}|{stat.st_size}|{stat.st_mtime_ns}"
        cache_dir = upload_root / hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_file = cache_dir / src.name

        if not cached_file.exists() or cached_file.stat().st_size != stat.st_size:
            try:
                if cached_file.exists():
                    cached_file.unlink()
                # Prefer hard-linking to avoid duplicating large video files.
                os.link(src, cached_file)
            except Exception:
                temp_name = (
                    f"{src.name}.{os.getpid()}.{random.randint(1000, 9999)}.tmp"
                )
                temp_file = cache_dir / temp_name
                shutil.copy2(src, temp_file)
                os.replace(temp_file, cached_file)

        return str(cached_file)
    except Exception:
        # Fallback to original path; launch(allowed_paths=...) may still permit it.
        return str(src)


def _media_version_token(file_path: str) -> str:
    """
    Build a short deterministic token from path + stat metadata for cache busting.
    """
    try:
        p = Path(file_path).resolve()
        st = p.stat()
        raw = f"{p.as_posix()}|{st.st_size}|{st.st_mtime_ns}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return f"rand{random.randint(100000, 999999)}"


def _probe_video_meta(video_path: str) -> Dict[str, str]:
    """
    Best-effort ffprobe for codec and dimensions.
    """
    out: Dict[str, str] = {}
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height",
                "-of",
                "default=noprint_wrappers=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if proc.returncode != 0:
            return out
        for raw in str(proc.stdout or "").splitlines():
            line = str(raw or "").strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = str(k).strip().lower()
            val = str(v).strip()
            if key and val:
                out[key] = val
    except Exception:
        return out
    return out


def _is_browser_friendly_codec(codec_name: str) -> bool:
    """
    Conservative browser-safe codec check for HTML5 video playback.
    """
    codec = str(codec_name or "").strip().lower()
    # Keep this strict to maximize cross-browser reliability.
    return codec in {"h264", "vp8", "vp9", "av1"}


def _ensure_browser_video_playable(file_path: str) -> str:
    """
    Ensure the comparison player receives a browser-friendly video codec.

    If source codec is not broadly HTML5-compatible (e.g., mpeg4 part2),
    create a cached H.264 preview and return that path.
    """
    src = Path(file_path).resolve()
    if not src.exists() or not src.is_file():
        return str(src)

    meta = _probe_video_meta(str(src))
    codec = str(meta.get("codec_name") or "").strip().lower()
    if _is_browser_friendly_codec(codec):
        return str(src)

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return str(src)

    try:
        st = src.stat()
        key = hashlib.sha256(
            f"{src.as_posix()}|{st.st_size}|{st.st_mtime_ns}|h264_preview_v1".encode("utf-8")
        ).hexdigest()[:20]
        out = src.with_name(f"{src.stem}.__h264_preview_{key}.mp4")
        if out.exists() and out.stat().st_size > 1024:
            return str(out)

        tmp = out.with_suffix(".tmp.mp4")
        cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(tmp),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 1024:
            os.replace(tmp, out)
            return str(out)
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
    except Exception:
        return str(src)
    return str(src)


def _build_video_badge(base_label: str, video_path: str) -> str:
    meta = _probe_video_meta(video_path)
    width = str(meta.get("width") or "").strip()
    height = str(meta.get("height") or "").strip()
    codec = str(meta.get("codec_name") or "").strip().lower()

    extras = []
    if width.isdigit() and height.isdigit():
        extras.append(f"{width}x{height}")
    if codec:
        extras.append(codec)

    if extras:
        return f"{base_label} ({', '.join(extras)})"
    return base_label


def create_video_comparison_html(
    original_video: Optional[str],
    upscaled_video: Optional[str],
    height: int = 600,
    slider_position: float = 50.0,
) -> str:
    """
    Create HTML for video comparison slider.

    Args:
        original_video: Path to original video file
        upscaled_video: Path to upscaled video file
        height: Height of video player in pixels
        slider_position: Initial slider position (0-100%)

    Returns:
        HTML string with embedded JavaScript for video comparison
    """
    if not original_video or not upscaled_video:
        return """
        <div style="text-align:center;padding:40px;background:#f0f0f0;border-radius:8px;">
            <p style="color:#666;font-size:16px;margin:0;">
                Upload and upscale videos to see comparison.
            </p>
        </div>
        """

    original_path = str(Path(original_video).resolve())
    upscaled_path = str(Path(upscaled_video).resolve())
    try:
        if Path(original_path).resolve() == Path(upscaled_path).resolve():
            same = html.escape(original_path)
            return f"""
            <div style="text-align:center;padding:24px;background:#fff4e5;border:1px solid #ffd59a;border-radius:8px;">
                <p style="margin:0;color:#8a5300;font-size:14px;">
                    Comparison skipped: original and upscaled paths are identical.<br>{same}
                </p>
            </div>
            """
    except Exception:
        pass

    original_served_path = _ensure_gradio_servable_file(original_path)
    upscaled_served_path = _ensure_gradio_servable_file(upscaled_path)
    original_served_path = _ensure_browser_video_playable(original_served_path)
    upscaled_served_path = _ensure_browser_video_playable(upscaled_served_path)

    # Gradio 6.x file route.
    original_path_encoded = urllib.parse.quote(
        original_served_path.replace("\\", "/"), safe=":/"
    )
    upscaled_path_encoded = urllib.parse.quote(
        upscaled_served_path.replace("\\", "/"), safe=":/"
    )
    original_token = _media_version_token(original_served_path)
    upscaled_token = _media_version_token(upscaled_served_path)
    original_url = f"/gradio_api/file={original_path_encoded}?v={original_token}&side=original"
    upscaled_url = f"/gradio_api/file={upscaled_path_encoded}?v={upscaled_token}&side=upscaled"

    original_badge = html.escape(_build_video_badge("Original", original_path))
    upscaled_badge = html.escape(_build_video_badge("Upscaled", upscaled_path))

    safe_height = max(260, int(height or 600))
    initial_slider = max(0.0, min(100.0, float(slider_position or 50.0)))
    unique_id = f"vc_{random.randint(100000, 999999)}"

    return f"""
    <div id="{unique_id}_container" class="video-comparison-container" style="position:relative;width:100%;max-width:1200px;margin:0 auto;background:#000;border-radius:8px;overflow:hidden;">
        <div id="{unique_id}_loading" style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#fff;font-size:16px;z-index:100;text-align:center;">
            Loading videos...
        </div>

        <div id="{unique_id}_wrapper" class="video-container" style="position:relative;width:100%;height:{safe_height}px;overflow:hidden;cursor:grab;">
            <div class="video-wrapper video-left" style="position:absolute;top:0;left:0;width:100%;height:100%;overflow:hidden;z-index:1;">
                <video id="{unique_id}_video1" style="width:100%;height:100%;object-fit:contain;display:block;" preload="metadata" playsinline>
                    <source src="{original_url}" type="video/mp4">
                </video>
            </div>

            <div id="{unique_id}_videoRight" class="video-wrapper video-right" style="position:absolute;top:0;left:0;width:100%;height:100%;overflow:hidden;z-index:2;clip-path:polygon(50% 0%,100% 0%,100% 100%,50% 100%);">
                <video id="{unique_id}_video2" style="width:100%;height:100%;object-fit:contain;display:block;" preload="metadata" playsinline>
                    <source src="{upscaled_url}" type="video/mp4">
                </video>
            </div>

            <div id="{unique_id}_slider" class="slider-container" style="position:absolute;top:0;left:50%;width:4px;height:100%;background:#fff;z-index:10;cursor:ew-resize;transform:translateX(-50%);">
                <div class="slider-handle" style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:40px;height:40px;background:#fff;border-radius:50%;box-shadow:0 2px 10px rgba(0,0,0,0.3);display:flex;align-items:center;justify-content:center;cursor:ew-resize;">
                    <span style="color:#3a76d8;font-weight:bold;font-size:16px;">&#8646;</span>
                </div>
            </div>

            <div style="position:absolute;top:10px;left:10px;background:rgba(0,0,0,0.7);color:#fff;padding:8px 15px;border-radius:20px;font-size:14px;z-index:20;">
                {original_badge}
            </div>
            <div style="position:absolute;top:10px;right:10px;background:rgba(0,0,0,0.7);color:#fff;padding:8px 15px;border-radius:20px;font-size:14px;z-index:20;">
                {upscaled_badge}
            </div>
        </div>

        <div id="{unique_id}_controls" style="padding:15px;background:linear-gradient(to top,rgba(0,0,0,0.9),rgba(0,0,0,0.7));display:flex;gap:15px;align-items:center;flex-wrap:wrap;">
            <button id="{unique_id}_playPause" style="padding:10px 20px;background:linear-gradient(45deg,#4CAF50,#45a049);color:white;border:none;border-radius:8px;cursor:pointer;font-size:14px;min-width:90px;font-weight:500;">Play</button>
            <div style="flex:1;min-width:200px;">
                <input id="{unique_id}_timeline" type="range" min="0" max="100" value="0" style="width:100%;cursor:pointer;accent-color:#4CAF50;">
            </div>
            <span id="{unique_id}_time" style="color:#fff;font-family:monospace;font-size:14px;min-width:100px;text-align:right;">0:00 / 0:00</span>
            <button id="{unique_id}_mute" style="padding:10px;background:rgba(255,255,255,0.2);color:white;border:none;border-radius:8px;cursor:pointer;font-size:14px;">Mute</button>
            <button id="{unique_id}_sync" style="padding:10px 15px;background:linear-gradient(45deg,#2196F3,#1976D2);color:white;border:none;border-radius:8px;cursor:pointer;font-size:14px;font-weight:500;">Sync</button>
            <button id="{unique_id}_fullscreen" style="padding:10px;background:rgba(255,255,255,0.2);color:white;border:none;border-radius:8px;cursor:pointer;font-size:14px;">Fullscreen</button>
        </div>
    </div>

    <script>
    (function() {{
        const uid = "{unique_id}";
        const container = document.getElementById(uid + "_container");
        const video1 = document.getElementById(uid + "_video1");
        const video2 = document.getElementById(uid + "_video2");
        const videoRight = document.getElementById(uid + "_videoRight");
        const slider = document.getElementById(uid + "_slider");
        const wrapper = document.getElementById(uid + "_wrapper");
        const playPauseBtn = document.getElementById(uid + "_playPause");
        const timeline = document.getElementById(uid + "_timeline");
        const timeDisplay = document.getElementById(uid + "_time");
        const muteBtn = document.getElementById(uid + "_mute");
        const syncBtn = document.getElementById(uid + "_sync");
        const fullscreenBtn = document.getElementById(uid + "_fullscreen");
        const loadingEl = document.getElementById(uid + "_loading");

        if (!container || !video1 || !video2 || !videoRight || !slider || !wrapper) {{
            return;
        }}
        if (container.dataset.vcInitialized === "1") {{
            return;
        }}
        container.dataset.vcInitialized = "1";

        let isPlaying = false;
        let isMuted = false;
        let isDragging = false;
        let sliderPos = {initial_slider};
        let ready1 = false;
        let ready2 = false;
        let syncTimer = null;
        let readyWatchdog = null;

        const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

        function formatTime(seconds) {{
            if (!seconds || isNaN(seconds)) return "0:00";
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return mins + ":" + String(secs).padStart(2, "0");
        }}

        function getEffectiveDuration() {{
            const d1 = Number(video1.duration || 0);
            const d2 = Number(video2.duration || 0);
            if (d1 > 0 && d2 > 0) return Math.min(d1, d2);
            return Math.max(d1, d2, 0);
        }}

        function renderTime() {{
            const duration = getEffectiveDuration();
            const current = clamp(Number(video1.currentTime || 0), 0, duration || 0);
            if (timeline && duration > 0) {{
                timeline.value = String(clamp((current / duration) * 100, 0, 100));
            }}
            if (timeDisplay) {{
                timeDisplay.textContent = formatTime(current) + " / " + formatTime(duration);
            }}
        }}

        function updateMuteButton() {{
            // Keep right-side video muted to avoid double-audio/echo.
            video2.muted = true;
            video1.muted = isMuted;
            if (muteBtn) muteBtn.textContent = isMuted ? "Unmute" : "Mute";
        }}

        function updateSliderPosition(percent) {{
            const p = clamp(percent, 0, 100);
            sliderPos = p;
            slider.style.left = p + "%";
            videoRight.style.clipPath = `polygon(${{p}}% 0%, 100% 0%, 100% 100%, ${{p}}% 100%)`;
        }}

        function syncTo(timeValue) {{
            const duration = getEffectiveDuration();
            const t = duration > 0 ? clamp(Number(timeValue || 0), 0, duration) : Math.max(0, Number(timeValue || 0));
            if (isFinite(t)) {{
                if (Math.abs((video1.currentTime || 0) - t) > 0.04) video1.currentTime = t;
                if (Math.abs((video2.currentTime || 0) - t) > 0.04) video2.currentTime = t;
            }}
            renderTime();
        }}

        function stopSyncLoop() {{
            if (syncTimer) {{
                window.clearInterval(syncTimer);
                syncTimer = null;
            }}
        }}

        function startSyncLoop() {{
            stopSyncLoop();
            syncTimer = window.setInterval(() => {{
                if (!isPlaying) return;
                if (video1.paused) {{
                    pauseBoth();
                    return;
                }}
                const d = getEffectiveDuration();
                if (d > 0 && (video1.currentTime || 0) >= d - 0.02) {{
                    pauseBoth();
                    return;
                }}
                const delta = Math.abs((video2.currentTime || 0) - (video1.currentTime || 0));
                if (delta > 0.05) {{
                    video2.currentTime = video1.currentTime || 0;
                }}
                if (video2.paused) {{
                    video2.play().catch(() => {{}});
                }}
                renderTime();
            }}, 80);
        }}

        async function safePlay(videoEl) {{
            try {{
                await videoEl.play();
                return true;
            }} catch (_) {{
                return false;
            }}
        }}

        function pauseBoth() {{
            video1.pause();
            video2.pause();
            isPlaying = false;
            if (playPauseBtn) playPauseBtn.textContent = "Play";
            stopSyncLoop();
            renderTime();
        }}

        async function playBoth() {{
            if (!(ready1 && ready2)) {{
                if (loadingEl) {{
                    loadingEl.style.display = "block";
                    loadingEl.innerHTML = "Waiting for videos to finish loading...";
                    loadingEl.style.color = "#ffb347";
                }}
                return;
            }}
            syncTo(Math.min(video1.currentTime || 0, video2.currentTime || 0));
            const leftOk = await safePlay(video1);
            await safePlay(video2);
            isPlaying = leftOk && !video1.paused;
            if (playPauseBtn) playPauseBtn.textContent = isPlaying ? "Pause" : "Play";
            if (isPlaying) {{
                startSyncLoop();
            }} else if (loadingEl) {{
                loadingEl.style.display = "block";
                loadingEl.innerHTML = "Play blocked by browser or unsupported stream";
                loadingEl.style.color = "#ffb347";
            }}
        }}

        function handleDragEvent(e) {{
            const rect = wrapper.getBoundingClientRect();
            const x = (e.clientX || e.pageX) - rect.left;
            const pct = (x / Math.max(1, rect.width)) * 100;
            updateSliderPosition(pct);
        }}

        function markReady(which) {{
            if (which === 1) ready1 = true;
            if (which === 2) ready2 = true;
            renderTime();
            if (ready1 && ready2) {{
                if (readyWatchdog) {{
                    window.clearTimeout(readyWatchdog);
                    readyWatchdog = null;
                }}
                if (loadingEl) loadingEl.style.display = "none";
                syncTo(0);
            }}
        }}

        updateSliderPosition(sliderPos);
        updateMuteButton();
        renderTime();

        video1.addEventListener("loadedmetadata", () => markReady(1), {{ once: true }});
        video2.addEventListener("loadedmetadata", () => markReady(2), {{ once: true }});
        video1.addEventListener("canplay", () => {{ if (!ready1) markReady(1); }}, {{ once: true }});
        video2.addEventListener("canplay", () => {{ if (!ready2) markReady(2); }}, {{ once: true }});
        video1.addEventListener("loadeddata", () => {{ if (!ready1) markReady(1); }}, {{ once: true }});
        video2.addEventListener("loadeddata", () => {{ if (!ready2) markReady(2); }}, {{ once: true }});

        video1.addEventListener("error", () => {{
            if (!loadingEl) return;
            loadingEl.innerHTML = "Error loading original video";
            loadingEl.style.color = "#ff6b6b";
        }});
        video2.addEventListener("error", () => {{
            if (!loadingEl) return;
            loadingEl.innerHTML = "Error loading upscaled video";
            loadingEl.style.color = "#ff6b6b";
        }});

        // If metadata is already available (cached browser state), mark ready now.
        if (video1.readyState >= 1) markReady(1);
        if (video2.readyState >= 1) markReady(2);

        readyWatchdog = window.setTimeout(() => {{
            if (ready1 && ready2) return;
            if (!loadingEl) return;
            const missing = [];
            if (!ready1) missing.push("original");
            if (!ready2) missing.push("upscaled");
            loadingEl.style.display = "block";
            loadingEl.innerHTML = `Still loading ${{missing.join(" and ")}} video...`;
            loadingEl.style.color = "#ffb347";
        }}, 5000);

        wrapper.addEventListener("pointerdown", (e) => {{
            isDragging = true;
            if (wrapper.setPointerCapture) {{
                try {{ wrapper.setPointerCapture(e.pointerId); }} catch (_) {{}}
            }}
            handleDragEvent(e);
            e.preventDefault();
        }});

        wrapper.addEventListener("pointermove", (e) => {{
            if (!isDragging) return;
            handleDragEvent(e);
        }});

        wrapper.addEventListener("pointerup", () => {{
            isDragging = false;
        }});
        wrapper.addEventListener("pointercancel", () => {{
            isDragging = false;
        }});

        if (playPauseBtn) {{
            playPauseBtn.addEventListener("click", async () => {{
                if (isPlaying) {{
                    pauseBoth();
                    return;
                }}
                await playBoth();
            }});
        }}

        if (timeline) {{
            timeline.addEventListener("input", (e) => {{
                const v = Number(e.target.value || 0);
                const effectiveDuration = getEffectiveDuration();
                const t = (v / 100) * effectiveDuration;
                syncTo(t);
            }});
        }}

        video1.addEventListener("timeupdate", () => {{
            renderTime();
        }});
        video2.addEventListener("timeupdate", () => renderTime());

        video1.addEventListener("play", () => {{
            isPlaying = true;
            if (playPauseBtn) playPauseBtn.textContent = "Pause";
            startSyncLoop();
        }});
        video1.addEventListener("pause", () => {{
            if (isPlaying) pauseBoth();
        }});

        video1.addEventListener("ended", () => {{
            pauseBoth();
        }});
        video2.addEventListener("ended", () => {{
            pauseBoth();
        }});

        if (syncBtn) {{
            syncBtn.addEventListener("click", () => {{
                syncTo(video1.currentTime || 0);
            }});
        }}

        if (muteBtn) {{
            muteBtn.addEventListener("click", () => {{
                isMuted = !isMuted;
                updateMuteButton();
            }});
        }}

        if (fullscreenBtn) {{
            fullscreenBtn.addEventListener("click", () => {{
                const fsEl =
                    document.fullscreenElement ||
                    document.webkitFullscreenElement ||
                    document.msFullscreenElement ||
                    null;

                // If already fullscreen, behave like ESC and exit.
                if (fsEl) {{
                    if (document.exitFullscreen) document.exitFullscreen();
                    else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
                    else if (document.msExitFullscreen) document.msExitFullscreen();
                    return;
                }}

                if (container.requestFullscreen) container.requestFullscreen();
                else if (container.webkitRequestFullscreen) container.webkitRequestFullscreen();
                else if (container.msRequestFullscreen) container.msRequestFullscreen();
            }});
        }}

        // Keep controls consistent if user leaves fullscreen.
        document.addEventListener("fullscreenchange", () => renderTime());

        video1.load();
        video2.load();
    }})();
    </script>

    <style>
    .video-comparison-container {{
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }}
    .video-comparison-container button:hover {{
        opacity: 0.92;
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }}
    .video-comparison-container button:active {{
        transform: translateY(0);
    }}
    .slider-container:hover {{
        width: 6px !important;
    }}
    .video-comparison-container:fullscreen {{
        display: flex;
        flex-direction: column;
        background: #000;
    }}
    .video-comparison-container:fullscreen .video-container {{
        flex: 1;
        height: auto !important;
    }}
    </style>
    """


def create_comparison_selector(original: Optional[str], upscaled: Optional[str]) -> str:
    """
    Compatibility wrapper: use custom slider for videos.
    """
    if not original or not upscaled:
        return create_video_comparison_html(None, None)

    original_path = Path(original) if isinstance(original, str) else original
    upscaled_path = Path(upscaled) if isinstance(upscaled, str) else upscaled
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if (
        original_path.suffix.lower() in video_exts
        and upscaled_path.suffix.lower() in video_exts
    ):
        return create_video_comparison_html(str(original), str(upscaled))

    return """
    <div style="text-align:center;padding:20px;">
        <p>Use ImageSlider component for image comparison.</p>
    </div>
    """
