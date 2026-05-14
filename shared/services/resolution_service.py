import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import gradio as gr

from shared.preset_manager import PresetManager
from shared.path_utils import (
    collision_safe_dir,
    get_media_duration_seconds,
    normalize_path,
    sanitize_filename,
)
from shared.path_utils import get_media_dimensions
from shared.resolution_calculator import (
    calculate_resolution, calculate_chunk_count, calculate_disk_space_required,
    get_available_disk_space, ResolutionConfig, ResolutionResult,
    estimate_seedvr2_upscale_plan_from_dims,
    estimate_fixed_scale_upscale_plan_from_dims,
)
from shared.input_detector import detect_input, validate_batch_directory


def resolution_defaults(models: List[str]) -> Dict[str, Any]:
    return {
        # NEW: Auto-detect scene count (PySceneDetect) for info panels.
        # This affects UI-only sizing/preview displays; it does NOT change processing unless auto_chunk is enabled.
        "auto_detect_scenes": True,
        # NEW (vNext): Auto chunking via PySceneDetect scenes (recommended default)
        "auto_chunk": True,
        # NEW: Frame-accurate splitting for chunking (lossless re-encode).
        # OFF = faster stream-copy (keyframe-limited).
        "frame_accurate_split": True,
        "chunk_size": 0,
        # Default 0 because overlap is not meaningful for scene cuts (auto chunking).
        "chunk_overlap": 0.0,
        "per_chunk_cleanup": False,
        "scene_threshold": 27.0,  # PySceneDetect sensitivity
        "min_scene_len": 1.0,  # Minimum scene length in seconds
    }


RESOLUTION_ORDER: List[str] = [
    "auto_detect_scenes",
    "auto_chunk",
    "frame_accurate_split",
    "per_chunk_cleanup",
    "chunk_size",
    "chunk_overlap",
    "scene_threshold",  # PySceneDetect sensitivity (for chunking)
    "min_scene_len",    # Minimum scene length for PySceneDetect
]


def _res_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    return dict(zip(RESOLUTION_ORDER, args))


def _apply_resolution_preset(
    preset: Dict[str, Any],
    defaults: Dict[str, Any],
    preset_manager: PresetManager,
    current: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    base = defaults.copy()
    if current:
        base.update(current)
    merged = preset_manager.merge_config(base, preset)
    # If auto chunking is enabled (PySceneDetect scenes), overlap is forced off.
    if bool(merged.get("auto_chunk", True)):
        merged["chunk_overlap"] = 0.0
    if merged.get("chunk_size", 0) and merged.get("chunk_overlap", 0) >= merged.get("chunk_size", 0):
        merged["chunk_overlap"] = max(0, merged.get("chunk_size", 0) - 1)
    return [merged[k] for k in RESOLUTION_ORDER]


def _get_aspect_ratio_str(aspect_ratio: float) -> Tuple[int, int]:
    """Convert aspect ratio to simplified fraction (e.g., 1.777 -> 16:9)"""
    common_ratios = {
        (16, 9): 1.778,
        (4, 3): 1.333,
        (21, 9): 2.333,
        (1, 1): 1.0,
        (3, 2): 1.5,
        (5, 4): 1.25,
    }
    
    # Find closest common ratio
    min_diff = float('inf')
    best_ratio = (16, 9)
    
    for (w, h), ratio in common_ratios.items():
        diff = abs(aspect_ratio - ratio)
        if diff < min_diff:
            min_diff = diff
            best_ratio = (w, h)
    
    # If very close to common ratio, use it
    if min_diff < 0.01:
        return best_ratio
    
    # Otherwise, calculate from actual ratio
    from math import gcd
    w = int(aspect_ratio * 1000)
    h = 1000
    divisor = gcd(w, h)
    return (w // divisor, h // divisor)


def _resolve_standalone_split_root(state: Dict[str, Any], base_dir: Optional[Path]) -> Path:
    seed_controls = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
    global_settings = seed_controls.get("global_settings", {}) if isinstance(seed_controls, dict) else {}
    configured_root = global_settings.get("output_dir") if isinstance(global_settings, dict) else None
    if configured_root:
        return Path(normalize_path(str(configured_root)))
    return Path(base_dir or Path.cwd()).resolve() / "outputs"


def _copy_or_remux_full_video_to_mp4(input_path: Path, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if input_path.suffix.lower() == ".mp4":
        try:
            shutil.copy2(input_path, output_path)
            return output_path.exists() and output_path.stat().st_size > 0
        except Exception:
            pass

    copy_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        proc = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return True
    except Exception:
        pass

    encode_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        proc = subprocess.run(encode_cmd, capture_output=True, text=True, timeout=600)
        return proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def chunk_estimate(chunk_size: float, chunk_overlap: float):
    if chunk_size <= 0:
        return gr.update(value="Chunking disabled.")
    if chunk_overlap >= chunk_size:
        return gr.update(value="⚠️ Chunk overlap must be smaller than chunk size.")
    return gr.update(
        value=f"Chunking enabled: size={chunk_size}s, overlap={chunk_overlap}s. Estimated chunks = ceil(duration / (size-overlap))."
    )


def build_resolution_callbacks(
    preset_manager: PresetManager,
    shared_state: gr.State,
    models: List[str],
    base_dir: Optional[Path] = None,
):
    defaults = resolution_defaults(models)

    def refresh_presets(model_name: str, select_name: Optional[str] = None):
        presets = preset_manager.list_presets("resolution", model_name)
        last_used = preset_manager.get_last_used_name("resolution", model_name)
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        return gr.update(choices=presets, value=value)

    def save_preset(preset_name: str, *args):
        if not preset_name.strip():
            return gr.update(), gr.update(value="⚠️ Enter a preset name before saving"), *list(args)

        try:
            payload = _res_dict_from_args(list(args))
            model_name = "default"
            preset_manager.save_preset_safe("resolution", model_name, preset_name.strip(), payload)
            dropdown = refresh_presets(model_name, select_name=preset_name.strip())

            current_map = dict(zip(RESOLUTION_ORDER, list(args)))
            loaded_vals = _apply_resolution_preset(payload, defaults, preset_manager, current=current_map)

            return dropdown, gr.update(value=f"✅ Saved preset '{preset_name}' for {model_name}"), *loaded_vals
        except Exception as e:
            return gr.update(), gr.update(value=f"❌ Error saving preset: {str(e)}"), *list(args)

    def load_preset(preset_name: str, model_name: str, current_values: List[Any]):
        """
        Load a preset.
        
        FIXED: Now returns (*values, status_message) to match UI output expectations.
        """
        try:
            model_name = model_name or "default"
            preset = preset_manager.load_preset_safe("resolution", model_name, preset_name)
            if preset:
                preset_manager.set_last_used("resolution", model_name, preset_name)

            defaults_with_model = defaults.copy()
            current_map = dict(zip(RESOLUTION_ORDER, current_values))
            values = _apply_resolution_preset(preset or {}, defaults_with_model, preset_manager, current=current_map)
            
            # Return values + status message (status is LAST)
            status_msg = f"✅ Loaded preset '{preset_name}'" if preset else "ℹ️ Preset not found"
            return (*values, gr.update(value=status_msg))
        except Exception as e:
            print(f"Error loading preset {preset_name}: {e}")
            # Return current values + error status
            return (*current_values, gr.update(value=f"❌ Error: {str(e)}"))

    def safe_defaults():
        return [defaults[k] for k in RESOLUTION_ORDER]

    def calculate_auto_resolution(
        input_path: str,
        upscale_factor: float,
        max_res: int,
        enable_max: bool,
        _auto_mode_unused: bool,
        pre_downscale_then_upscale: bool,
        model_scale: Optional[int],
        state: Dict,
    ) -> Tuple[str, Dict]:
        """
        Auto-calculate target sizing plan based on input and the new Upscale-x rules.
        
        Returns:
            (info_message, updated_state)
        """
        if not input_path or not input_path.strip():
            return "⚠️ No input path provided", state
        
        try:
            # Detect input
            input_info = detect_input(input_path)
            if not input_info.is_valid:
                return f"⚠️ {input_info.error_message}", state
            
            # Get dimensions
            from shared.path_utils import get_media_dimensions
            dims = get_media_dimensions(input_path)
            if not dims:
                return "⚠️ Could not determine input dimensions", state
            w, h = dims

            # Build sizing plan
            max_edge = int(max_res if enable_max else 0)
            if model_scale and model_scale > 1:
                plan = estimate_fixed_scale_upscale_plan_from_dims(
                    w,
                    h,
                    requested_scale=float(upscale_factor),
                    model_scale=int(model_scale),
                    max_edge=max_edge,
                    force_pre_downscale=True,
                )
            else:
                plan = estimate_seedvr2_upscale_plan_from_dims(
                    w,
                    h,
                    upscale_factor=float(upscale_factor),
                    max_edge=max_edge,
                    pre_downscale_then_upscale=bool(pre_downscale_then_upscale),
                )

            # Update state with calculated values (for other tabs to consume if desired)
            seed_controls = state.get("seed_controls", {})
            seed_controls["calculated_output_width"] = plan.final_saved_width or plan.resize_width
            seed_controls["calculated_output_height"] = plan.final_saved_height or plan.resize_height
            seed_controls["needs_downscale_first"] = bool(plan.pre_downscale_then_upscale and plan.preprocess_scale < 0.999999)
            seed_controls["input_resize_width"] = plan.preprocess_width if plan.pre_downscale_then_upscale else None
            seed_controls["input_resize_height"] = plan.preprocess_height if plan.pre_downscale_then_upscale else None
            state["seed_controls"] = seed_controls
            
            # Build info message
            info = f"### Auto-Resolution Result\n\n"
            info += f"📐 Input: {plan.input_width}×{plan.input_height}\n\n"
            info += f"🎯 Target: {plan.requested_scale:.2f}x"
            if max_edge and max_edge > 0:
                info += f", max edge {max_edge}px\n\n"
            else:
                info += " (no max edge cap)\n\n"

            if plan.pre_downscale_then_upscale and plan.preprocess_scale < 0.999999:
                info += f"🧩 Preprocess: downscale to {plan.preprocess_width}×{plan.preprocess_height} (×{plan.preprocess_scale:.3f})\n\n"

            info += f"✅ Expected output: {plan.final_saved_width or plan.resize_width}×{plan.final_saved_height or plan.resize_height}\n"
            info += f"(Effective: {plan.effective_scale:.2f}x)\n"

            if plan.notes:
                info += "\n" + "\n".join([f"ℹ️ {n}" for n in plan.notes])
            
            return info, state
            
        except Exception as e:
            return f"❌ Error calculating resolution: {str(e)}", state
    
    def calculate_chunk_estimate(
        input_path: str,
        auto_chunk: bool,
        chunk_size: float,
        chunk_overlap: float,
        state: Dict,
    ) -> Tuple[str, Dict]:
        """
        Estimate number of chunks and processing info.
        
        Returns:
            (info_message, updated_state)
        """
        if not input_path or not input_path.strip():
            return "⚠️ No input path provided", state
        
        if auto_chunk:
            # Note: this can be expensive on long videos (requires decoding frames).
            # We run it only on explicit user action (Estimate Chunks button).
            try:
                input_info = detect_input(input_path)
                if not input_info.is_valid:
                    return f"⚠️ {input_info.error_message}", state
                if input_info.input_type != "video":
                    return (
                        "ℹ️ Auto Chunk (PySceneDetect) is enabled, but scene counting applies to VIDEO inputs only.\n\n"
                        f"- Detected input type: {input_info.input_type.upper()}\n"
                        "- For non-video inputs, chunking uses folder/image chunking rules.",
                        state,
                    )

                seed_controls = state.get("seed_controls", {})
                scene_threshold = float(seed_controls.get("scene_threshold", 27.0) or 27.0)
                min_scene_len = float(seed_controls.get("min_scene_len", 1.0) or 1.0)
                norm_path = normalize_path(input_path)

                # If we already scanned this exact input with the same settings, return cached result.
                scan = seed_controls.get("last_scene_scan") or {}
                try:
                    scan_path = normalize_path(scan.get("input_path")) if scan.get("input_path") else None
                except Exception:
                    scan_path = scan.get("input_path")

                cached_scene_count = int(scan.get("scene_count", 0) or 0)
                if (
                    scan_path
                    and scan_path == norm_path
                    and abs(float(scan.get("scene_threshold", scene_threshold)) - scene_threshold) < 1e-6
                    and abs(float(scan.get("min_scene_len", min_scene_len)) - min_scene_len) < 1e-6
                    and cached_scene_count > 0
                ):
                    dur = get_media_duration_seconds(norm_path) or 0.0
                    avg = (float(dur) / cached_scene_count) if dur and cached_scene_count > 0 else 0.0

                    msg = "✅ Auto Chunk (PySceneDetect) scan (cached).\n\n"
                    msg += f"- Detected scenes: **{cached_scene_count}**\n"
                    msg += f"- Settings: threshold={scene_threshold:g}, min scene len={min_scene_len:g}s (overlap forced 0)\n"
                    if dur:
                        msg += f"- Duration: ~{float(dur):.1f}s → avg scene ~{avg:.1f}s\n"
                    if cached_scene_count == 1:
                        msg += "- Note: single continuous scene → no physical split will occur.\n"
                    return msg, state

                from shared.chunking import detect_scenes

                scenes = detect_scenes(
                    norm_path,
                    threshold=scene_threshold,
                    min_scene_len=min_scene_len,
                )
                if not scenes:
                    return (
                        "⚠️ Could not detect scenes (PySceneDetect unavailable or detection failed).\n\n"
                        "- Chunk count will be determined at run time using fallback chunking.\n"
                        "- Tip: ensure `scenedetect` is installed in the venv.",
                        state,
                    )

                # Cache last scan for info panels (SeedVR2/others can display without rescanning).
                seed_controls["last_scene_scan"] = {
                    "input_path": norm_path,
                    "scene_threshold": scene_threshold,
                    "min_scene_len": min_scene_len,
                    "scene_count": int(len(scenes)),
                }
                state["seed_controls"] = seed_controls

                dur = get_media_duration_seconds(norm_path) or 0.0
                avg = (float(dur) / len(scenes)) if dur and len(scenes) > 0 else 0.0

                msg = "✅ Auto Chunk (PySceneDetect) scan complete.\n\n"
                msg += f"- Detected scenes: **{len(scenes)}**\n"
                msg += f"- Settings: threshold={scene_threshold:g}, min scene len={min_scene_len:g}s (overlap forced 0)\n"
                if dur:
                    msg += f"- Duration: ~{float(dur):.1f}s → avg scene ~{avg:.1f}s\n"
                if len(scenes) == 1:
                    msg += "- Note: single continuous scene → no physical split will occur.\n"
                msg += "\nℹ️ This scan decodes the video once on CPU; for long videos it may take a while."
                return msg, state
            except Exception as e:
                return f"⚠️ Scene scan failed: {str(e)}", state

        if chunk_size <= 0:
            return "ℹ️ Chunking disabled (Auto Chunk off, chunk size = 0)", state
        
        try:
            # Get chunk estimate
            chunk_count, duration, info = calculate_chunk_count(input_path, chunk_size, 1.0)
            
            if chunk_count == 0:
                return info, state
            
            # Add disk space estimate (rough) based on Upscale-x plan
            try:
                dims = get_media_dimensions(input_path)
                if dims:
                    w, h = dims
                else:
                    w, h = 0, 0

                seed_controls = state.get("seed_controls", {})
                scale_x = float(seed_controls.get("upscale_factor_val", 4.0) or 4.0)
                max_edge = 0

                plan = estimate_seedvr2_upscale_plan_from_dims(
                    w, h, upscale_factor=scale_x, max_edge=max_edge, pre_downscale_then_upscale=bool(seed_controls.get("ratio_downscale", True))
                ) if w and h else None

                result = ResolutionResult(
                    output_width=int(plan.final_saved_width or plan.resize_width) if plan else 0,
                    output_height=int(plan.final_saved_height or plan.resize_height) if plan else 0,
                )

                space_required, space_str = calculate_disk_space_required(input_path, result, "mp4", duration)
                
                # Get available space
                output_dir = state.get("seed_controls", {}).get("last_output_dir", ".")
                avail_bytes, avail_str = get_available_disk_space(output_dir)
                
                info += f"\n\n### Disk Space\n"
                info += f"Estimated required: **{space_str}**\n"
                info += f"Available: **{avail_str}**\n"
                
                if space_required > 0 and avail_bytes > 0:
                    if space_required > avail_bytes * 0.9:  # Using >90% of available
                        info += f"\n⚠️ **Warning:** Insufficient disk space!"
                    
            except Exception:
                pass
            
            # Update state
            seed_controls = state.get("seed_controls", {})
            seed_controls["estimated_chunk_count"] = chunk_count
            state["seed_controls"] = seed_controls
            
            return info, state
            
        except Exception as e:
            return f"❌ Error estimating chunks: {str(e)}", state
    
    def standalone_scene_split(
        input_path: str,
        auto_chunk: bool,
        frame_accurate_split: bool,
        chunk_size: float,
        chunk_overlap: float,
        scene_threshold: float,
        min_scene_len: float,
        state: Dict,
        progress=gr.Progress(track_tqdm=False),
    ) -> Any:
        """
        Standalone scene/chunk splitter for the Resolution tab.

        Creates one output directory containing MP4 scene files with video and
        audio together, without running any upscaler pipeline.
        """
        state = state if isinstance(state, dict) else {"seed_controls": {}}
        seed_controls = state.setdefault("seed_controls", {})
        if not isinstance(seed_controls, dict):
            seed_controls = {}
            state["seed_controls"] = seed_controls

        progress_messages: List[str] = []

        def _result(message: str, output_dir: Optional[Path] = None, show_open: bool = False):
            has_output_dir = bool(output_dir)
            return (
                gr.update(value=message, visible=True),
                gr.update(value=str(output_dir) if output_dir else "", visible=has_output_dir),
                gr.update(visible=bool(show_open and output_dir)),
                state,
            )

        def _collect_progress(message: str) -> None:
            if not message:
                return
            clean = str(message).strip()
            if clean:
                progress_messages.append(clean)

        def _status(title: str, details: Optional[List[str]] = None) -> str:
            lines = ["### Standalone Split Progress", "", f"**{title}**"]
            if details:
                lines.append("")
                lines.extend([f"- {line}" for line in details if line])
            tail = [msg for msg in progress_messages[-8:] if msg]
            if tail:
                lines.append("")
                lines.append("Recent log:")
                lines.extend([f"- {msg}" for msg in tail])
            return "\n".join(lines)

        def _emit_progress(value: float, desc: str) -> None:
            try:
                progress(max(0.0, min(1.0, float(value))), desc=desc)
            except Exception:
                pass

        def _scene_seconds_label(start_sec: float, end_sec: float) -> str:
            return f"{float(start_sec):.3f}s to {float(end_sec):.3f}s"

        def _is_full_scene(scene: Tuple[float, float], duration: float) -> bool:
            if duration <= 0:
                return False
            start_sec, end_sec = float(scene[0]), float(scene[1])
            tol = max(0.05, duration * 0.005)
            return abs(start_sec) <= tol and abs(end_sec - duration) <= tol

        try:
            _emit_progress(0.01, "Starting standalone split")
            yield _result(_status("Validating input..."))

            if not input_path or not str(input_path).strip():
                input_path = str(seed_controls.get("last_input_path") or "").strip()
            if not input_path:
                yield _result("ERROR: No input video path provided.")
                return

            input_info = detect_input(input_path)
            if not input_info.is_valid:
                yield _result(f"ERROR: {input_info.error_message}")
                return
            if input_info.input_type != "video":
                yield _result(f"ERROR: Standalone scene split requires a video file. Detected: {input_info.input_type}.")
                return

            norm_path = normalize_path(input_path)
            if not norm_path:
                yield _result("ERROR: Could not normalize input path.")
                return
            source_path = Path(norm_path)
            if not source_path.exists() or not source_path.is_file():
                yield _result(f"ERROR: Input video does not exist: {norm_path}")
                return

            # Keep shared caches aligned with the latest Resolution tab controls.
            seed_controls["last_input_path"] = str(source_path)
            seed_controls["auto_chunk"] = bool(auto_chunk)
            seed_controls["frame_accurate_split"] = bool(frame_accurate_split)
            seed_controls["chunk_size_sec"] = float(chunk_size or 0)
            seed_controls["chunk_overlap_sec"] = 0.0 if bool(auto_chunk) else float(chunk_overlap or 0)
            seed_controls["scene_threshold"] = float(scene_threshold or 27.0)
            seed_controls["min_scene_len"] = float(min_scene_len or 1.0)
            state["seed_controls"] = seed_controls

            scenes = []
            mode_label = "PySceneDetect scenes"
            total_duration = float(get_media_duration_seconds(str(source_path)) or 0.0)
            if bool(auto_chunk):
                _emit_progress(0.04, "Detecting scenes")
                yield _result(
                    _status(
                        "Detecting scene cuts...",
                        [
                            f"Input: {source_path.name}",
                            f"Threshold: {float(scene_threshold or 27.0):g}",
                            f"Minimum scene length: {float(min_scene_len or 1.0):g}s",
                        ],
                    )
                )
                from shared.chunking import detect_scenes

                scan_events: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
                scan_result: Dict[str, Any] = {"scenes": [], "error": None}

                def _scan_worker() -> None:
                    try:
                        scan_result["scenes"] = detect_scenes(
                            str(source_path),
                            threshold=float(scene_threshold or 27.0),
                            min_scene_len=float(min_scene_len or 1.0),
                            overlap_sec=0.0,
                            on_progress=lambda msg: scan_events.put(("log", msg)),
                            on_progress_pct=lambda pct: scan_events.put(("pct", pct)),
                        )
                    except Exception as exc:
                        scan_result["error"] = exc
                    finally:
                        scan_events.put(("done", None))

                scan_thread = threading.Thread(target=_scan_worker, daemon=True)
                scan_thread.start()
                scene_pct = 0
                scan_done = False
                last_yield = 0.0

                while not scan_done:
                    try:
                        kind, value = scan_events.get(timeout=0.25)
                        if kind == "pct":
                            scene_pct = max(0, min(100, int(value)))
                        elif kind == "log":
                            _collect_progress(str(value))
                        elif kind == "done":
                            scan_done = True
                    except queue.Empty:
                        pass

                    now = time.monotonic()
                    if scan_done or (now - last_yield) >= 0.5:
                        _emit_progress(0.04 + (0.38 * scene_pct / 100.0), f"Detecting scenes {scene_pct}%")
                        yield _result(
                            _status(
                                f"Detecting scene cuts... {scene_pct}%",
                                [
                                    "This scans the video once on CPU.",
                                    "Long videos can stay on this step for a while.",
                                ],
                            )
                        )
                        last_yield = now

                scan_thread.join(timeout=1.0)
                if scan_result.get("error") is not None:
                    raise scan_result["error"]
                scenes = scan_result.get("scenes") or []
                if not scenes:
                    if total_duration > 0:
                        scenes = [(0.0, total_duration)]
                        progress_messages.append("Scene detection returned no cuts; exporting the full video as scene_0001.mp4.")
                    else:
                        yield _result("ERROR: Scene detection failed and video duration could not be read.")
                        return
                _emit_progress(0.43, f"Detected {len(scenes)} scene(s)")
                yield _result(
                    _status(
                        f"Detected {len(scenes)} scene(s).",
                        [
                            f"Duration: {total_duration:.2f}s" if total_duration > 0 else "",
                            "Preparing output folder...",
                        ],
                    )
                )
            else:
                fixed_seconds = float(chunk_size or 0)
                if fixed_seconds <= 0:
                    yield _result(
                        "ERROR: Auto Chunk is off and Chunk Size is 0. Enable Auto Chunk for scene splitting, "
                        "or set a fixed Chunk Size for standalone chunk export."
                    )
                    return
                from shared.chunking import fallback_scenes

                mode_label = f"fixed {fixed_seconds:g}s chunks"
                _emit_progress(0.08, "Creating fixed chunks")
                scenes = fallback_scenes(
                    str(source_path),
                    chunk_seconds=fixed_seconds,
                    overlap_seconds=float(chunk_overlap or 0),
                )
                yield _result(
                    _status(
                        f"Prepared {len(scenes)} fixed chunk(s).",
                        [f"Chunk size: {fixed_seconds:g}s", f"Overlap: {float(chunk_overlap or 0):g}s"],
                    )
                )

            if not scenes:
                yield _result("ERROR: No scenes/chunks were produced for this input.")
                return

            output_root = _resolve_standalone_split_root(state, base_dir)
            safe_stem = sanitize_filename(source_path.stem) or "video"
            split_dir = collision_safe_dir(output_root / f"{safe_stem}_scene_split")
            split_dir.mkdir(parents=True, exist_ok=True)
            work_root = split_dir / "_split_work"
            work_root.mkdir(parents=True, exist_ok=True)

            _emit_progress(0.46, "Output folder ready")
            yield _result(
                _status(
                    "Output folder ready.",
                    [
                        f"Folder: {split_dir}",
                        f"Writing {len(scenes)} MP4 file(s) with video and audio...",
                    ],
                ),
                split_dir,
                show_open=False,
            )

            from shared.chunking import split_video

            source_resolved = source_path.resolve()
            final_paths: List[Path] = []
            total_scenes = max(1, len(scenes))
            for idx, scene in enumerate(scenes, 1):
                start_sec, end_sec = float(scene[0]), float(scene[1])
                chunk_dir = work_root / f"scene_{idx:04d}"
                chunk_dir.mkdir(parents=True, exist_ok=True)
                target = split_dir / f"scene_{idx:04d}.mp4"

                base_progress = 0.48 + (0.46 * (idx - 1) / total_scenes)
                _emit_progress(base_progress, f"Writing scene {idx}/{total_scenes}")
                yield _result(
                    _status(
                        f"Writing scene {idx}/{total_scenes}...",
                        [
                            f"Time range: {_scene_seconds_label(start_sec, end_sec)}",
                            f"Output: {target.name}",
                            f"Completed files: {len(final_paths)}/{total_scenes}",
                        ],
                    ),
                    split_dir,
                    show_open=False,
                )

                raw_paths = split_video(
                    str(source_path),
                    [(start_sec, end_sec)],
                    chunk_dir,
                    precise=bool(frame_accurate_split),
                    preserve_quality=True,
                    include_audio=True,
                    on_progress=_collect_progress,
                )

                raw_resolved = [Path(p).resolve() for p in raw_paths if p]
                if len(raw_resolved) == 1 and raw_resolved[0] == source_resolved:
                    if len(scenes) == 1 and _is_full_scene((start_sec, end_sec), total_duration):
                        if not _copy_or_remux_full_video_to_mp4(source_path, target):
                            yield _result("ERROR: Could not export the single-scene MP4.", split_dir, show_open=True)
                            return
                    else:
                        yield _result(
                            "ERROR: ffmpeg did not produce a split file for this scene. "
                            "Try Frame-Accurate Split, or check ffmpeg support for this source file.",
                            split_dir,
                            show_open=True,
                        )
                        return
                elif len(raw_paths) == 1:
                    src = Path(raw_paths[0])
                    if src.resolve() != target.resolve():
                        try:
                            src.replace(target)
                        except Exception:
                            shutil.copy2(src, target)
                            try:
                                src.unlink(missing_ok=True)
                            except Exception:
                                pass
                else:
                    yield _result(
                        f"ERROR: Expected one MP4 for scene {idx}, but ffmpeg returned {len(raw_paths)} file(s).",
                        split_dir,
                        show_open=True,
                    )
                    return

                if not target.exists() or target.stat().st_size <= 0:
                    yield _result(f"ERROR: Scene {idx} output is missing or empty: {target}", split_dir, show_open=True)
                    return

                final_paths.append(target)
                _emit_progress(0.48 + (0.46 * idx / total_scenes), f"Finished scene {idx}/{total_scenes}")
                yield _result(
                    _status(
                        f"Finished scene {idx}/{total_scenes}.",
                        [
                            f"Created: {target.name}",
                            f"Completed files: {len(final_paths)}/{total_scenes}",
                        ],
                    ),
                    split_dir,
                    show_open=False,
                )

            try:
                shutil.rmtree(work_root, ignore_errors=True)
            except Exception:
                pass

            final_paths = [p for p in final_paths if p.exists() and p.stat().st_size > 0]
            if len(final_paths) != len(scenes):
                yield _result(
                    f"ERROR: Expected {len(scenes)} MP4 files but found {len(final_paths)} complete files.",
                    split_dir,
                    show_open=True,
                )
                return

            seed_controls["last_standalone_scene_split_dir"] = str(split_dir)
            seed_controls["last_output_dir"] = str(split_dir)
            seed_controls["last_output_path"] = str(final_paths[0]) if final_paths else None
            seed_controls["last_scene_scan"] = {
                "input_path": str(source_path),
                "scene_threshold": float(scene_threshold or 27.0),
                "min_scene_len": float(min_scene_len or 1.0),
                "scene_count": int(len(scenes)),
            }
            state["seed_controls"] = seed_controls

            _emit_progress(1.0, "Standalone split complete")

            listed = "\n".join(f"- {p.name}" for p in final_paths[:30])
            if len(final_paths) > 30:
                listed += f"\n- ... {len(final_paths) - 30} more"

            message = (
                "SUCCESS: Standalone split complete.\n\n"
                f"- Output directory: `{split_dir}`\n"
                f"- Files created: **{len(final_paths)}**\n"
                f"- Mode: {mode_label}\n"
                f"- Split: {'frame-accurate lossless' if frame_accurate_split else 'fast stream-copy'}\n\n"
                f"{listed}"
            )
            if progress_messages:
                tail = [msg for msg in progress_messages[-4:] if msg]
                if tail:
                    message += "\n\nRecent log:\n" + "\n".join(f"- {msg}" for msg in tail)

            yield _result(message, split_dir, show_open=True)

        except Exception as e:
            yield _result(f"ERROR: Standalone scene split failed: {str(e)}")

    def apply_to_seed(*args):
        """Apply resolution settings to ALL upscaler pipelines via shared state"""
        state = args[-1]
        values = list(args[:-1])
        
        # Extract resolution settings from inputs
        settings_dict = _res_dict_from_args(values)
        
        # Update shared state with resolution settings for all pipelines
        seed_controls = state.get("seed_controls", {})
        
        # Resolution tab now manages chunking/splitting only.
        seed_controls["auto_detect_scenes"] = bool(settings_dict.get("auto_detect_scenes", True))
        seed_controls["auto_chunk"] = bool(settings_dict.get("auto_chunk", True))
        seed_controls["frame_accurate_split"] = bool(settings_dict.get("frame_accurate_split", True))
        seed_controls["chunk_size_sec"] = settings_dict.get("chunk_size", 0)
        # Overlap is only meaningful for static chunking; force 0 when auto chunking is enabled.
        seed_controls["chunk_overlap_sec"] = 0.0 if seed_controls["auto_chunk"] else float(settings_dict.get("chunk_overlap", 0) or 0)
        seed_controls["per_chunk_cleanup"] = settings_dict.get("per_chunk_cleanup", False)
        seed_controls["scene_threshold"] = settings_dict.get("scene_threshold", 27.0)
        seed_controls["min_scene_len"] = settings_dict.get("min_scene_len", 1.0)
        # Global max-resolution propagation is removed.
        seed_controls.pop("max_resolution_val", None)
        seed_controls.pop("enable_max_target", None)
        
        state["seed_controls"] = seed_controls
        
        status_msg = f"✅ Applied resolution settings to ALL upscalers:\n"
        status_msg += f"- Auto Detect Scenes (info): {seed_controls.get('auto_detect_scenes', True)}\n"
        if seed_controls.get("auto_chunk", True):
            status_msg += "- Chunking: Auto (PySceneDetect scenes, overlap forced 0)\n"
            status_msg += f"- Split: {'Frame-accurate (lossless)' if seed_controls.get('frame_accurate_split', True) else 'Fast (keyframe-limited)'}\n"
            status_msg += f"- Scene Detection: threshold={seed_controls['scene_threshold']}, min_len={seed_controls['min_scene_len']}s\n"
        elif seed_controls['chunk_size_sec'] > 0:
            status_msg += f"- Chunking: Static {seed_controls['chunk_size_sec']}s (overlap: {seed_controls['chunk_overlap_sec']}s)\n"
            status_msg += f"- Split: {'Frame-accurate (lossless)' if seed_controls.get('frame_accurate_split', True) else 'Fast (keyframe-limited)'}\n"
        return gr.update(value=status_msg), state

    def estimate_from_input(size, ov, state):
        """Estimate chunks from cached input path in shared state"""
        seed_controls = state.get("seed_controls", {})
        path = seed_controls.get("last_input_path")
        if not path:
            return gr.update(value="Provide input path (upload or textbox) to estimate chunks."), state
        path = normalize_path(path)
        dur = get_media_duration_seconds(path) if path else None
        if not dur:
            return chunk_estimate(size, ov), state
        if size <= 0 or ov >= size:
            return chunk_estimate(size, ov), state
        import math

        est = math.ceil(dur / max(0.001, size - ov))
        return gr.update(value=f"Duration ~{dur:.1f}s → est. {est} chunks (size {size}s, overlap {ov}s)."), state

    return {
        "defaults": defaults,
        "order": RESOLUTION_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "apply_to_seed": apply_to_seed,
        "chunk_estimate": chunk_estimate,
        "estimate_from_input": estimate_from_input,
        "calculate_auto_resolution": calculate_auto_resolution,
        "calculate_chunk_estimate": calculate_chunk_estimate,
        "standalone_scene_split": standalone_scene_split,
    }



