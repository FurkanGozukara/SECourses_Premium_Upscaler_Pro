from typing import Any, Dict, List, Optional, Tuple
import gradio as gr

from shared.preset_manager import PresetManager
from shared.path_utils import normalize_path, get_media_duration_seconds
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
        "model": models[0] if models else "",
        "auto_resolution": True,
        "enable_max_target": True,
        # NEW: Auto-detect scene count (PySceneDetect) for info panels.
        # This affects UI-only sizing/preview displays; it does NOT change processing unless auto_chunk is enabled.
        "auto_detect_scenes": True,
        # NEW (vNext): Auto chunking via PySceneDetect scenes (recommended default)
        "auto_chunk": True,
        # NEW: Frame-accurate splitting for chunking (lossless re-encode).
        # OFF = faster stream-copy (keyframe-limited).
        "frame_accurate_split": True,
        # NEW (vNext): Target sizing expressed as an upscale factor (relative to input)
        # Default 4x as requested.
        "upscale_factor": 4.0,
        # Max edge cap (LONG side). 0 = unlimited.
        "max_target_resolution": 0,
        "chunk_size": 0,
        # Default 0 because overlap is not meaningful for scene cuts (auto chunking).
        "chunk_overlap": 0.0,
        "ratio_downscale_then_upscale": True,
        "per_chunk_cleanup": False,
        "scene_threshold": 27.0,  # PySceneDetect sensitivity
        "min_scene_len": 1.0,  # Minimum scene length in seconds
    }


RESOLUTION_ORDER: List[str] = [
    "model",
    "auto_resolution",
    "enable_max_target",
    "auto_detect_scenes",
    "upscale_factor",
    "max_target_resolution",
    "ratio_downscale_then_upscale",
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
):
    defaults = resolution_defaults(models)

    def _ensure_model_cache(model: str, state: Dict[str, Any]) -> Dict[str, Any]:
        cache_root = state["seed_controls"].setdefault("resolution_cache", {})
        model_key = model or defaults["model"]
        return cache_root.setdefault(model_key, {})

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
            model_name = payload["model"]
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
            model_name = model_name or defaults["model"]
            preset = preset_manager.load_preset_safe("resolution", model_name, preset_name)
            if preset:
                preset_manager.set_last_used("resolution", model_name, preset_name)

            defaults_with_model = defaults.copy()
            defaults_with_model["model"] = model_name
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
                max_edge = int(seed_controls.get("max_resolution_val", 0) or 0)
                enable_max = bool(seed_controls.get("enable_max_target", True))
                if not enable_max:
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
    
    def apply_to_seed(*args):
        """Apply resolution settings to ALL upscaler pipelines via shared state"""
        state = args[-1]
        values = list(args[:-1])
        
        # Extract resolution settings from inputs
        settings_dict = _res_dict_from_args(values)
        
        # Update shared state with resolution settings for all pipelines
        seed_controls = state.get("seed_controls", {})
        
        # Cache resolution values for all upscalers to use (GLOBAL level)
        seed_controls["upscale_factor_val"] = float(settings_dict.get("upscale_factor", 4.0) or 4.0)
        seed_controls["max_resolution_val"] = int(settings_dict.get("max_target_resolution", 0) or 0)
        seed_controls["enable_max_target"] = settings_dict.get("enable_max_target", True)
        seed_controls["auto_resolution"] = settings_dict.get("auto_resolution", True)  # kept for backward compat
        seed_controls["auto_detect_scenes"] = bool(settings_dict.get("auto_detect_scenes", True))
        seed_controls["auto_chunk"] = bool(settings_dict.get("auto_chunk", True))
        seed_controls["frame_accurate_split"] = bool(settings_dict.get("frame_accurate_split", True))
        seed_controls["chunk_size_sec"] = settings_dict.get("chunk_size", 0)
        # Overlap is only meaningful for static chunking; force 0 when auto chunking is enabled.
        seed_controls["chunk_overlap_sec"] = 0.0 if seed_controls["auto_chunk"] else float(settings_dict.get("chunk_overlap", 0) or 0)
        # Repurposed: now controls "pre-downscale then upscale when clamped"
        seed_controls["ratio_downscale"] = settings_dict.get("ratio_downscale_then_upscale", True)
        seed_controls["per_chunk_cleanup"] = settings_dict.get("per_chunk_cleanup", False)
        seed_controls["scene_threshold"] = settings_dict.get("scene_threshold", 27.0)
        seed_controls["min_scene_len"] = settings_dict.get("min_scene_len", 1.0)
        
        # FIXED: Also cache per-model for proper isolation as requested
        # Write to per-model cache so each model can have different resolution settings
        model_name = settings_dict.get("model", "")
        if model_name:
            model_cache = _ensure_model_cache(model_name, state)
            model_cache["upscale_factor_val"] = float(settings_dict.get("upscale_factor", 4.0) or 4.0)
            model_cache["max_resolution_val"] = int(settings_dict.get("max_target_resolution", 0) or 0)
            model_cache["enable_max_target"] = settings_dict.get("enable_max_target", True)
            model_cache["auto_resolution"] = settings_dict.get("auto_resolution", True)
            model_cache["auto_detect_scenes"] = bool(settings_dict.get("auto_detect_scenes", True))
            model_cache["auto_chunk"] = bool(settings_dict.get("auto_chunk", True))
            model_cache["frame_accurate_split"] = bool(settings_dict.get("frame_accurate_split", True))
            model_cache["chunk_size_sec"] = float(settings_dict.get("chunk_size", 0) or 0)
            model_cache["chunk_overlap_sec"] = 0.0 if model_cache["auto_chunk"] else float(settings_dict.get("chunk_overlap", 0) or 0)
            model_cache["ratio_downscale"] = settings_dict.get("ratio_downscale_then_upscale", True)
            model_cache["per_chunk_cleanup"] = settings_dict.get("per_chunk_cleanup", False)
            model_cache["scene_threshold"] = float(settings_dict.get("scene_threshold", 27.0))
            model_cache["min_scene_len"] = float(settings_dict.get("min_scene_len", 1.0))
        
        state["seed_controls"] = seed_controls
        
        status_msg = f"✅ Applied resolution settings to ALL upscalers:\n"
        status_msg += f"- Upscale Factor: {seed_controls['upscale_factor_val']}x\n"
        status_msg += f"- Max Resolution: {seed_controls['max_resolution_val']}px\n"
        status_msg += f"- Auto-Resolution: {seed_controls['auto_resolution']}\n"
        status_msg += f"- Auto Detect Scenes (info): {seed_controls.get('auto_detect_scenes', True)}\n"
        if seed_controls.get("auto_chunk", True):
            status_msg += "- Chunking: Auto (PySceneDetect scenes, overlap forced 0)\n"
            status_msg += f"- Split: {'Frame-accurate (lossless)' if seed_controls.get('frame_accurate_split', True) else 'Fast (keyframe-limited)'}\n"
            status_msg += f"- Scene Detection: threshold={seed_controls['scene_threshold']}, min_len={seed_controls['min_scene_len']}s\n"
        elif seed_controls['chunk_size_sec'] > 0:
            status_msg += f"- Chunking: Static {seed_controls['chunk_size_sec']}s (overlap: {seed_controls['chunk_overlap_sec']}s)\n"
            status_msg += f"- Split: {'Frame-accurate (lossless)' if seed_controls.get('frame_accurate_split', True) else 'Fast (keyframe-limited)'}\n"
        if model_name:
            status_msg += f"\n💾 Settings also saved for model: {model_name}"
        
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

    def cache_resolution(scale_x, m_res, model, state):
        model_cache = _ensure_model_cache(model, state)
        model_cache["upscale_factor_val"] = float(scale_x or 4.0)
        model_cache["max_resolution_val"] = int(m_res or 0)
        return gr.update(value=f"Resolution cached for {model}."), state

    def cache_resolution_flags(auto_res, enable_max, chunk_sz, chunk_ov, ratio_down, per_cleanup, scene_thresh, min_scene, model, state):
        model_cache = _ensure_model_cache(model, state)
        model_cache["auto_resolution"] = auto_res
        model_cache["enable_max_target"] = enable_max
        model_cache["chunk_size_sec"] = float(chunk_sz or 0)
        model_cache["chunk_overlap_sec"] = float(chunk_ov or 0)
        model_cache["ratio_downscale"] = ratio_down
        model_cache["per_chunk_cleanup"] = per_cleanup
        model_cache["scene_threshold"] = float(scene_thresh or 27.0)
        model_cache["min_scene_len"] = float(min_scene or 1.0)
        return gr.update(value=f"Resolution options cached for {model}."), state

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
        "cache_resolution": lambda *args: cache_resolution(*args[:-1], args[-1]),
        "cache_resolution_flags": lambda *args: cache_resolution_flags(*args[:-1], args[-1]),
        "calculate_auto_resolution": calculate_auto_resolution,
        "calculate_chunk_estimate": calculate_chunk_estimate,
    }


