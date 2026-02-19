"""
SeedVR2 Service Module - Complete Rewrite
Handles all SeedVR2 processing logic, presets, and callbacks
"""

import html
import re
import shutil
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr

from shared.preset_manager import PresetManager
from shared.preset_auto_serializer import (
    auto_detect_inputs,
    serialize_component_values,
    deserialize_to_component_values,
    create_auto_order
)
from shared.runner import Runner, RunResult
from shared.path_utils import (
    normalize_path,
    collision_safe_path,
    collision_safe_dir,
    ffmpeg_set_fps,
    get_media_dimensions,
    get_media_duration_seconds,
    get_media_fps,
    detect_input_type,
    resolve_batch_output_dir,
)
from shared.resolution_calculator import estimate_seedvr2_upscale_plan_from_dims
from shared.chunking import chunk_and_process, check_resume_available
from shared.output_run_manager import (
    prepare_single_video_run,
    downscaled_video_path,
    resolve_resume_input_from_run_dir,
    ensure_image_input_artifact,
    finalize_run_context,
)
from shared.ffmpeg_utils import scale_video
from shared.face_restore import restore_image, restore_video
from shared.models.seedvr2_meta import get_seedvr2_model_names, model_meta_map
from shared.logging_utils import RunLogger
from shared.comparison_unified import create_unified_comparison, build_comparison_selector
from shared.oom_alert import clear_vram_oom_alert, maybe_set_vram_oom_alert, show_vram_oom_modal
from shared.video_comparison import create_comparison_selector
from shared.global_rife import maybe_apply_global_rife, global_rife_enabled
from shared.model_manager import get_model_manager, ModelType
from shared.gpu_utils import expand_cuda_device_spec, get_global_gpu_override, validate_cuda_device_spec
from shared.models.rife_meta import get_rife_default_model
from shared.error_handling import (
    validate_input_path,
    validate_cuda_device as validate_cuda_spec,
    validate_batch_size,
    check_ffmpeg_available,
    check_disk_space,
    safe_execute,
    logger as error_logger,
)

# Constants --------------------------------------------------------------------
SEEDVR2_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
SEEDVR2_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
I32_INDEX_LIMIT = 2_147_483_647


def _largest_4n_plus_1_leq(n: int) -> int:
    """Largest value <= n that satisfies 4k+1 (minimum 1)."""
    try:
        n_i = int(n)
    except Exception:
        return 1
    if n_i <= 1:
        return 1
    return int(((n_i - 1) // 4) * 4 + 1)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Parse booleans safely from bool/int/float/string values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", ""}:
            return False
    return bool(default)


def _estimate_seedvr2_output_dims(in_w: int, in_h: int, resolution_short: int, max_edge: int = 0) -> Tuple[int, int]:
    """
    Mirror the SeedVR2 CLI sizing logic sufficiently for batch-safety checks.
    """
    w = max(1, int(in_w or 1))
    h = max(1, int(in_h or 1))
    res = max(16, int(resolution_short or 16))
    m_edge = max(0, int(max_edge or 0))

    if w <= h:
        out_w = res
        out_h = int(res * h / max(1, w))
    else:
        out_h = res
        out_w = int(res * w / max(1, h))

    if m_edge > 0 and max(out_w, out_h) > m_edge:
        s = float(m_edge) / float(max(out_w, out_h))
        out_w = int(round(out_w * s))
        out_h = int(round(out_h * s))

    # CLI eventually saves even dimensions after alignment/trimming.
    out_w = max(2, int(out_w) - (int(out_w) % 2))
    out_h = max(2, int(out_h) - (int(out_h) % 2))
    return out_w, out_h


def _is_path_within(path: Path, parent: Path) -> bool:
    """Return True when `path` is equal to or inside `parent`."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _predict_batch_video_output_path(item_out_dir: Path, input_file: Path, output_format: Any) -> Path:
    """
    Predict the stable output path for a batch video item without creating directories.
    """
    fmt = "png" if str(output_format or "auto").lower() == "png" else "mp4"
    stem = input_file.stem or "output"
    return item_out_dir / stem if fmt == "png" else item_out_dir / f"{stem}.mp4"


def _looks_like_image_input(path_value: Any, original_name: Any = None) -> bool:
    """
    Robust image-input detection for upload temp paths that may not keep extension.
    """
    path_text = str(path_value or "").strip()
    if path_text:
        try:
            ext = Path(path_text).suffix.lower()
            if ext in SEEDVR2_IMAGE_EXTS:
                return True
        except Exception:
            pass
        try:
            if detect_input_type(path_text) == "image":
                return True
        except Exception:
            pass

    original_text = str(original_name or "").strip()
    if original_text:
        try:
            return Path(original_text).suffix.lower() in SEEDVR2_IMAGE_EXTS
        except Exception:
            return False
    return False


def _apply_image_output_preferences(
    image_path: Optional[str],
    image_output_format: Any,
    image_output_quality: Any,
) -> Optional[str]:
    """
    Convert final image output to the globally selected image format/quality.
    Returns updated path (or original path on failure/no-op).
    """
    if not image_path:
        return image_path
    try:
        src = Path(str(image_path))
        if (not src.exists()) or src.is_dir():
            return image_path

        fmt = str(image_output_format or "png").strip().lower()
        if fmt not in {"png", "jpg", "webp"}:
            fmt = "png"

        try:
            quality = int(float(image_output_quality or 95))
        except Exception:
            quality = 95
        quality = max(1, min(100, quality))

        target_ext = ".jpg" if fmt == "jpg" else f".{fmt}"
        needs_reencode = src.suffix.lower() != target_ext or fmt in {"jpg", "webp"}
        if not needs_reencode:
            return image_path

        from PIL import Image

        dst = collision_safe_path(src.with_suffix(target_ext))
        with Image.open(src) as img:
            if fmt == "jpg":
                img = img.convert("RGB")
                img.save(dst, format="JPEG", quality=quality)
            elif fmt == "webp":
                img.save(dst, format="WEBP", quality=quality)
            else:
                img.save(dst, format="PNG")

        try:
            if dst.resolve() != src.resolve():
                src.unlink(missing_ok=True)
        except Exception:
            pass

        return str(dst)
    except Exception:
        return image_path


# Defaults and ordering --------------------------------------------------------
def _get_default_attention_mode() -> str:
    """
    Get optimal default attention mode with GPU compute capability detection.
    
    INTELLIGENT PRIORITY based on GPU architecture:
    - **Blackwell (12.x)**: sageattn_3 > flash_attn_3 > flash_attn_2 > sdpa
    - **Hopper (9.x)**: flash_attn_3 > sageattn_2 > flash_attn_2 > sdpa
    - **Ada/Ampere (8.x-8.9)**: flash_attn_2 > sdpa
    - **Turing (7.5-7.x)**: sdpa (flash_attn slower on older arch)
    - **Older (<7.5)**: sdpa only
    
    Automatically selects the fastest attention backend for detected GPU.
    
    Returns:
        Best attention mode for detected GPU architecture
    """
    # IMPORTANT:
    # Do NOT import torch / flash-attn / sageattention in the parent Gradio process.
    # Importing torch can permanently increase RAM usage, and some CUDA queries can
    # create a CUDA context that reserves hundreds of MB of VRAM for the life of the UI.
    #
    # We use NVML (nvidia-smi) via shared.gpu_utils to detect compute capability without
    # initializing CUDA, and `importlib.util.find_spec()` to check optional backends
    # without importing their CUDA extensions.
    try:
        from shared.gpu_utils import get_gpu_info
        import importlib.util

        gpus = get_gpu_info()
        if not gpus:
            return "sdpa"

        compute_cap = gpus[0].compute_capability
        has_flash_attn = importlib.util.find_spec("flash_attn") is not None
        has_sageattention = importlib.util.find_spec("sageattention") is not None
        has_sageattn3 = importlib.util.find_spec("sageattn3") is not None

        # If we can't detect architecture, pick a safe default.
        if not compute_cap:
            return "flash_attn_2" if has_flash_attn else "sdpa"

        major, minor = compute_cap

        # Blackwell (12.x): prefer sageattn_3 if installed, else flash_attn_2, else sdpa.
        if major >= 12:
            if has_sageattn3:
                return "sageattn_3"
            return "flash_attn_2" if has_flash_attn else "sdpa"

        # Hopper (9.x-11.x): flash_attn_2 is generally safe if installed, else sageattn_2, else sdpa.
        if major >= 9:
            if has_flash_attn:
                return "flash_attn_2"
            if has_sageattention:
                return "sageattn_2"
            return "sdpa"

        # Ampere/Ada (8.x): flash_attn_2 if installed.
        if major >= 8:
            return "flash_attn_2" if has_flash_attn else "sdpa"

        return "sdpa"
    except Exception as e:
        print(f"Warning: Attention mode detection failed ({e}), using sdpa")
        return "sdpa"


def seedvr2_defaults(model_name: Optional[str] = None, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get default SeedVR2 settings aligned with CLI defaults.
    Applies model-specific metadata when model_name is provided.
    
    Args:
        model_name: Optional model name to apply metadata from
        base_dir: Optional base directory (app root) to resolve model_dir path
    """
    # IMPORTANT: do not import torch here (parent Gradio process must remain GPU-free).
    # Use NVML-based detection (nvidia-smi) instead.
    try:
        from shared.gpu_utils import get_gpu_info
        cuda_default = "0" if get_gpu_info() else ""
    except Exception:
        cuda_default = ""
    
    # Compute correct model directory path
    # SeedVR2 models are stored in <base_dir>/SeedVR2/models/
    if base_dir:
        model_dir_path = str(Path(base_dir) / "SeedVR2" / "models")
    else:
        # Fallback: try to compute from this file's location
        import os
        service_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        app_base = service_dir.parent.parent  # shared/services -> shared -> app root
        model_dir_path = str(app_base / "SeedVR2" / "models")
    
    # Get model metadata if specific model is provided
    model_meta = None
    # Default to 7B fp16 model (best quality/speed balance)
    available_models = get_seedvr2_model_names()
    default_model = "seedvr2_ema_7b_fp16.safetensors" if "seedvr2_ema_7b_fp16.safetensors" in available_models else (available_models[0] if available_models else "seedvr2_ema_3b_fp16.safetensors")
    target_model = model_name or default_model
    
    if target_model:
        from shared.models.seedvr2_meta import model_meta_map
        meta_map = model_meta_map()
        model_meta = meta_map.get(target_model)
    
    # Apply model-specific defaults if metadata available
    if model_meta:
        default_attention = model_meta.preferred_attention if hasattr(model_meta, 'preferred_attention') else _get_default_attention_mode()
        default_batch_size = model_meta.default_batch_size if hasattr(model_meta, 'default_batch_size') else 5
        compile_compatible = model_meta.compile_compatible if hasattr(model_meta, 'compile_compatible') else True
        max_res_cap = model_meta.max_resolution if hasattr(model_meta, 'max_resolution') else 0
    else:
        default_attention = _get_default_attention_mode()
        default_batch_size = 5
        compile_compatible = True
        max_res_cap = 0
    
    return {
        "input_path": "",
        "output_override": "",
        "output_format": "auto",
        "model_dir": model_dir_path,
        "dit_model": target_model,
        "batch_enable": False,
        "batch_input_path": "",
        "batch_output_path": "",
        # PySceneDetect chunking removed - now configured in Resolution tab
        # Legacy chunk_enable, scene_threshold, scene_min_len removed from ORDER
        "chunk_size": 0,  # SeedVR2 native chunking (frames per chunk, 0=disabled)
        "resolution": 1080,
        "max_resolution": 1920,  # Default 1920 (model cap only limits max slider, not default)
        # NEW (vNext): replace manual short-side target with an Upscale factor (default 4x).
        # The actual SeedVR2 CLI `resolution` is computed at runtime from the input dimensions.
        "upscale_factor": 4.0,
        # NEW (vNext): when max edge would clamp the requested upscale, optionally downscale the input first,
        # then upscale with the full factor to reach the capped target (useful for fixed-scale models).
        "pre_downscale_then_upscale": True,
        "batch_size": default_batch_size,  # Apply model-specific default
        "uniform_batch_size": False,
        "seed": 42,
        "skip_first_frames": 0,
        "load_cap": 0,
        "prepend_frames": 0,
        "temporal_overlap": 0,
        "color_correction": "lab",
        "input_noise_scale": 0.0,
        "latent_noise_scale": 0.1,
        # UI semantics: False = keep default safeguard (force 0 for images),
        # True = allow custom latent-noise slider value for images.
        "force_latent_noise_zero_for_images": False,
        "cuda_device": cuda_default,
        "dit_offload_device": "cpu",
        "vae_offload_device": "cpu",
        "tensor_offload_device": "cpu",
        "blocks_to_swap": 0,
        "swap_io_components": False,
        "vae_encode_tiled": True,
        "vae_encode_tile_size": 1024,
        "vae_encode_tile_overlap": 128,
        "vae_decode_tiled": True,
        "vae_decode_tile_size": 1024,
        "vae_decode_tile_overlap": 128,
        "tile_debug": "false",
        "attention_mode": default_attention,  # Apply model-specific attention
        "compile_dit": False,  # Default off, will validate against model metadata
        "compile_vae": False,  # Default off, will validate against model metadata
        "compile_backend": "inductor",
        "compile_mode": "default",
        "compile_fullgraph": False,
        "compile_dynamic": False,
        "compile_dynamo_cache_size_limit": 64,
        "compile_dynamo_recompile_limit": 128,
        "cache_dit": False,
        "cache_vae": False,
        "debug": False,
        "resume_chunking": False,
        "resume_run_dir": "",
        "save_metadata": True,  # Per-run metadata toggle (respects global telemetry)
        "fps_override": 0,  # FPS override (0 = use source FPS, >0 = set specific FPS)
        # ADDED v2.5.22: FFmpeg 10-bit encoding support for reduced banding in gradients
        "video_backend": "opencv",  # "opencv" (default, 8-bit) or "ffmpeg" (10-bit capable)
        "use_10bit": False,  # Enable 10-bit color depth (requires video_backend="ffmpeg", x265 codec)
        # Persist per-run face restoration toggle from SeedVR2 tab.
        "face_restore_after_upscale": False,
        "_compile_compatible": compile_compatible,  # Store for validation
    }


"""
 PRESET SERIALIZATION ORDER & ROBUSTNESS DESIGN
==================================================

CURRENT APPROACH: Manual Synchronization (Option C)
----------------------------------------------------

This list defines the order of parameters for preset save/load.
MUST match inputs_list order in ui/seedvr2_tab.py.

 MANUAL SYNCHRONIZATION REQUIRED:
Adding a new control requires updates across 3 locations:

1. **seedvr2_defaults()** (this file, lines ~108-194)
   - Add default value for new control
   - Include model-specific metadata if applicable
   
2. **SEEDVR2_ORDER** (this file, lines ~244-308)
   - Append new key name at END (preserves backward compatibility)
   - CRITICAL: Order determines serialization sequence
   
3. **inputs_list** (ui/seedvr2_tab.py, lines ~743-759)
   - Add Gradio component at SAME POSITION as SEEDVR2_ORDER
   - Component values align with ORDER by index

 BACKWARD COMPATIBILITY GUARANTEE:
------------------------------------
Old presets automatically work with new features via PresetManager.merge_config():
- Keys in preset  loaded values
- Keys NOT in preset  current defaults (new controls get default values)
- NO migration scripts needed when adding features
- Graceful degradation for removed controls (ignored keys are harmless)

 RUNTIME VALIDATION & SAFETY:
--------------------------------
save_preset() callback validates len(inputs_list) == len(SEEDVR2_ORDER) at runtime.
Catches integration bugs immediately with detailed error message.
If mismatch detected:
- Preset save is aborted to prevent corruption
- Error shown in UI with exact counts and missing keys
- Development-time validation in seedvr2_tab.py logs warnings on load

 ROBUSTNESS FEATURES IMPLEMENTED:
------------------------------------
1. **Type Validation**: merge_config() preserves types (int/float/str/bool auto-converted)
2. **Tab-Specific Constraints**: validate_preset_constraints() enforces model rules:
   - SeedVR2: batch_size 4n+1, tile overlap < size, BlockSwap requires offload
   - GAN: scale factor validation
   - RIFE: single GPU enforcement, FPS multiplier limits
   - FlashVSR+: tile constraints, precision compatibility
3. **Model Metadata Integration**: Model-specific defaults and constraints from metadata registry
4. **Collision-Safe Storage**: Presets use sanitized names, atomic writes (tmp  rename)
5. **Last-Used Tracking**: Auto-restore last preset per model on tab load
6. **Missing Preset Warnings**: Non-blocking warnings if last-used preset file is missing

 ALTERNATIVE APPROACHES EVALUATED:
------------------------------------

**Option A: Auto-Serialization** (preset_auto_serializer module exists but unused)
Pros:
  + Eliminates manual ORDER maintenance
  + Auto-detects component IDs/labels from inputs_list
  + Reduces human error in synchronization
Cons:
  - Requires strict component naming conventions (not currently enforced)
  - Less explicit control over serialization sequence
  - Harder to debug when component detection fails
  - Breaking change (would need refactoring all tabs)
Status: Module implemented but not wired up (lines 15-21 in service imports)

**Option B: Schema-Driven with Dataclasses**
Pros:
  + Single source of truth (@dataclass defines both defaults and UI)
  + Type safety with Python type hints
  + Auto-generate UI components from schema
  + IDE autocomplete for settings access
Cons:
  - Complete architectural change (breaking)
  - Requires custom UI component generator
  - Loss of granular UI customization flexibility
  - Significant refactoring effort across all 7 tabs
Status: Not implemented (would be major rewrite)

**Option C: Current Manual Approach**  CHOSEN
Pros:
  + Proven stable and predictable behavior
  + Easy to debug when things break (explicit ordering)
  + Full control over UI component layout and styling
  + Runtime validation catches mismatches immediately
  + Works well for ~50 controls per tab (manageable scale)
Cons:
  - Requires discipline to maintain sync across 3 files
  - Human error possible (mitigated by runtime validation)
  - Adding controls is not "one-line" (requires 3 edits)

RATIONALE for Option C:
- Backward compatibility auto-merge works excellently (proven in production)
- Runtime validation catches errors immediately (no silent corruption)
- Current scale (~50 controls/tab) is manageable with discipline
- Refactoring to A/B is high-risk, high-effort for unclear benefit
- Explicit control valuable for complex UIs with custom validation

 EASE OF INTEGRATION (Adding New Controls):
----------------------------------------------
**Step-by-Step Process:**

1. Add default to seedvr2_defaults():
   ```python
   "new_control": default_value,  # Line ~193
   ```

2. Append to SEEDVR2_ORDER (at END for backward compat):
   ```python
   "new_control",  # Line ~308
   ```

3. Add Gradio component in seedvr2_tab.py:
   ```python
   new_control_widget = gr.Checkbox(...)  # Create widget
   inputs_list.append(new_control_widget)  # Line ~759
   ```

4. Validation happens automatically:
   - Runtime check warns if sync breaks
   - Old presets auto-merge (get default value for new control)
   - No migration scripts needed

**Estimated Time**: ~2 minutes per new control
**Error Rate**: Low (runtime validation catches mistakes)
**Maintenance Cost**: Acceptable for current scale

 ROBUSTNESS ASSESSMENT:
-------------------------
The current preset system IS robust and easy to manage:
-  Auto-merge handles feature additions seamlessly
-  Runtime validation prevents corruption
-  Model-specific constraints enforced automatically
-  Type safety via merge_config() type preservation
-  Collision-safe storage with atomic writes
-  Manual sync required but validated at runtime
-  Developer discipline needed (mitigated by validation)

CONCLUSION: Current approach meets "extremely robust and easy to manage" requirement
given the scale and complexity. Alternative approaches would add complexity without
clear benefit at current scale.

To validate sync, check: len(inputs_list) == len(SEEDVR2_ORDER)
Development-time check at line 762 in seedvr2_tab.py logs warnings if mismatched.
"""

SEEDVR2_ORDER: List[str] = [
    # Input/Output
    "input_path",
    "output_override",
    "output_format",
    "model_dir",
    "dit_model",
    # Batch processing (internal, not exposed as batch UI in SeedVR2 - use directory input instead)
    "batch_enable",
    "batch_input_path",
    "batch_output_path",
    # PySceneDetect chunking - REMOVED LEGACY CONTROLS (use Resolution tab instead)
    # Former: "chunk_enable", "scene_threshold", "scene_min_len" - now managed by Resolution tab
    # SeedVR2 native streaming (CLI --chunk_size, works WITH PySceneDetect)
    "chunk_size",
    # Resolution and processing
    "resolution",
    "max_resolution",
    "batch_size",
    "uniform_batch_size",
    "seed",
    "skip_first_frames",
    "load_cap",
    "prepend_frames",
    "temporal_overlap",
    # Quality controls
    "color_correction",
    "input_noise_scale",
    "latent_noise_scale",
    # Device configuration
    "cuda_device",
    "dit_offload_device",
    "vae_offload_device",
    "tensor_offload_device",
    # BlockSwap memory optimization
    "blocks_to_swap",
    "swap_io_components",
    # VAE tiling
    "vae_encode_tiled",
    "vae_encode_tile_size",
    "vae_encode_tile_overlap",
    "vae_decode_tiled",
    "vae_decode_tile_size",
    "vae_decode_tile_overlap",
    "tile_debug",
    # Performance and compilation
    "attention_mode",
    "compile_dit",
    "compile_vae",
    "compile_backend",
    "compile_mode",
    "compile_fullgraph",
    "compile_dynamic",
    "compile_dynamo_cache_size_limit",
    "compile_dynamo_recompile_limit",
    # Model caching (batch processing optimization)
    "cache_dit",
    "cache_vae",
    # Debug and resume
    "debug",
    "resume_chunking",
    # vNext sizing (app-level; does not directly map 1:1 to CLI flags)
    "upscale_factor",
    "pre_downscale_then_upscale",
    # Per-run post-upscale face restoration toggle (SeedVR2 tab checkbox).
    "face_restore_after_upscale",
    # Resume from an existing chunk run folder (shared across chunk-capable upscalers).
    "resume_run_dir",
    # Optional image latent-noise policy (see default key notes above).
    "force_latent_noise_zero_for_images",
]


# Preset Migration -------------------------------------------------------------
def _migrate_preset_values(cfg: Dict[str, Any], defaults: Dict[str, Any], silent: bool = False) -> None:
    """
    Migrate old preset values to new ones for backward compatibility.
    
    This function modifies cfg in-place and handles:
    - Renamed values (e.g., flash_attn  flash_attn_2)
    - Deprecated values (replace with defaults)
    - Type conversions (if needed)
    
    Args:
        cfg: Configuration dictionary to migrate (modified in-place)
        defaults: Default values to use as fallback
        silent: If True, suppress migration logs (useful during bulk operations)
    """
    # Attention mode migration: old values to new valid choices
    # Valid choices: ['sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', 'sageattn_3']
    attention_migrations = {
        "flash_attn": "flash_attn_2",  # Old generic name  v2
        "flash_attention": "flash_attn_2",  # Alternative old name
        "flash": "flash_attn_2",  # Short form
        "sageattn": "sageattn_2",  # Old generic name  v2
        "sage": "sageattn_2",  # Short form
    }
    
    current_attention = cfg.get("attention_mode", "")
    if current_attention in attention_migrations:
        old_val = current_attention
        new_val = attention_migrations[old_val]
        cfg["attention_mode"] = new_val
        if not silent:
            error_logger.info(f"Migrated attention_mode: '{old_val}'  '{new_val}'")
    elif current_attention and current_attention not in ["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"]:
        # Unknown/invalid value  fallback to default
        default_attention = defaults.get("attention_mode", "sdpa")
        error_logger.warning(f"Unknown attention_mode '{current_attention}', falling back to default '{default_attention}'")
        cfg["attention_mode"] = default_attention
    
    # Video backend migration (added in v2.5.22)
    # Valid choices: ['opencv', 'ffmpeg']
    current_backend = cfg.get("video_backend", "")
    if current_backend not in ["opencv", "ffmpeg"]:
        # Old presets might have numeric values or missing values
        default_backend = defaults.get("video_backend", "opencv")
        if current_backend not in ["", None]:
            error_logger.info(f"Migrated video_backend: '{current_backend}'  '{default_backend}'")
        cfg["video_backend"] = default_backend
    
    # Use_10bit migration (added in v2.5.22)
    # Must be boolean
    current_10bit = cfg.get("use_10bit", False)
    if not isinstance(current_10bit, bool):
        # Convert numeric or string values to boolean
        default_10bit = defaults.get("use_10bit", False)
        cfg["use_10bit"] = bool(current_10bit) if current_10bit else default_10bit
        if current_10bit not in [None, ""]:
            error_logger.info(f"Migrated use_10bit: '{current_10bit}'  '{cfg['use_10bit']}'")
    
    # Add more migrations here as needed for other settings
    # Example:
    # if "old_setting_name" in cfg:
    #     cfg["new_setting_name"] = cfg.pop("old_setting_name")
    

# Guardrails -------------------------------------------------------------------
def _enforce_seedvr2_guardrails(cfg: Dict[str, Any], defaults: Dict[str, Any], state: Optional[Dict[str, Any]] = None, silent_migration: bool = False) -> Dict[str, Any]:
    """Apply SeedVR2-specific validation rules and apply resolution tab settings if available."""
    cfg = cfg.copy()
    
    # Migrate old preset values to new ones for backward compatibility
    _migrate_preset_values(cfg, defaults, silent=silent_migration)

    # Optional image latent-noise policy:
    # False => safeguard ON (force 0 for images), True => allow custom image latent noise.
    cfg["force_latent_noise_zero_for_images"] = _coerce_bool(
        cfg.get("force_latent_noise_zero_for_images", defaults.get("force_latent_noise_zero_for_images", False)),
        default=bool(defaults.get("force_latent_noise_zero_for_images", False)),
    )
    
    # Apply model-specific metadata constraints
    model_name = cfg.get("dit_model", "")
    if model_name:
        from shared.models.seedvr2_meta import model_meta_map
        meta_map = model_meta_map()
        model_meta = meta_map.get(model_name)
        
        if model_meta:
            # Disable compile if model doesn't support it (e.g., GGUF quantized models)
            compile_compatible = getattr(model_meta, 'compile_compatible', True)
            if not compile_compatible:
                if cfg.get("compile_dit") or cfg.get("compile_vae"):
                    error_logger.warning(f"Model {model_name} doesn't support torch.compile - disabling compile flags")
                    cfg["compile_dit"] = False
                    cfg["compile_vae"] = False
                    cfg["_compile_disabled_reason"] = f"Model {model_name} is not compile-compatible"
            
            # Enforce multi-GPU support check
            supports_multi_gpu = getattr(model_meta, 'supports_multi_gpu', True)
            cuda_device_str = str(cfg.get("cuda_device", ""))
            devices = [d.strip() for d in cuda_device_str.replace(" ", "").split(",") if d.strip() and d.strip().isdigit()]
            
            if not supports_multi_gpu and len(devices) > 1:
                error_logger.warning(f"Model {model_name} doesn't support multi-GPU - forcing single GPU (using first: {devices[0]})")
                cfg["cuda_device"] = devices[0]
                cfg["_multi_gpu_disabled_reason"] = f"Model {model_name} doesn't support multi-GPU"
            
            # Apply model max resolution cap if set
            model_max_res = getattr(model_meta, 'max_resolution', 0)
            if model_max_res > 0:
                current_res = cfg.get("resolution", 1080)
                if current_res > model_max_res:
                    error_logger.warning(f"Model {model_name} max resolution is {model_max_res}, clamping from {current_res}")
                    cfg["resolution"] = model_max_res
                    cfg["_resolution_clamped_reason"] = f"Model max resolution: {model_max_res}"
            
            # Set preferred attention mode if not explicitly set by user.
            # Normalize legacy names (e.g., flash_attn -> flash_attn_2) so CLI args stay valid.
            preferred_attention = getattr(model_meta, 'preferred_attention', None)
            if preferred_attention and cfg.get("attention_mode") == _get_default_attention_mode():
                cfg["attention_mode"] = preferred_attention
                _migrate_preset_values(cfg, defaults, silent=True)

    # Apply resolution tab settings from shared state if available
    if state:
        seed_controls = state.get("seed_controls", {})
        
        # Apply shared sizing values (Resolution tab / global cache)
        if seed_controls.get("upscale_factor_val") is not None:
            try:
                cfg["upscale_factor"] = float(seed_controls["upscale_factor_val"])
            except Exception:
                pass
        # Repurposed global flag: "pre-downscale then upscale when capped"
        if "ratio_downscale" in seed_controls:
            cfg["pre_downscale_then_upscale"] = bool(seed_controls.get("ratio_downscale", True))
        # PySceneDetect chunking now ONLY controlled by Resolution tab
        # No more chunk_enable - chunking triggers automatically when chunk_size_sec > 0
        if "chunk_overlap_sec" in seed_controls:
            cfg["chunk_overlap"] = float(seed_controls["chunk_overlap_sec"])

    # Batch size must be 4n+1 using centralized validation
    bs = int(cfg.get("batch_size", defaults["batch_size"]))
    is_valid, error_msg = validate_batch_size(bs, must_be_4n_plus_1=True)
    if not is_valid:
        error_logger.warning(f"Invalid batch size {bs}, correcting: {error_msg}")
        cfg["batch_size"] = max(5, (bs // 4) * 4 + 1)

    # VAE tiling constraints
    if cfg.get("vae_encode_tiled"):
        tile_size = cfg.get("vae_encode_tile_size", defaults["vae_encode_tile_size"])
        overlap = cfg.get("vae_encode_tile_overlap", 0)
        if overlap >= tile_size:
            cfg["vae_encode_tile_overlap"] = max(0, tile_size - 1)

    if cfg.get("vae_decode_tiled"):
        tile_size = cfg.get("vae_decode_tile_size", defaults["vae_decode_tile_size"])
        overlap = cfg.get("vae_decode_tile_overlap", 0)
        if overlap >= tile_size:
            cfg["vae_decode_tile_overlap"] = max(0, tile_size - 1)

    # BlockSwap requires dit_offload_device
    blockswap_enabled = cfg.get("blocks_to_swap", 0) > 0 or cfg.get("swap_io_components", False)
    if blockswap_enabled and str(cfg.get("dit_offload_device", "none")).lower() in ("none", ""):
        cfg["dit_offload_device"] = "cpu"

    # Multi-GPU constraints
    devices = [d.strip() for d in str(cfg.get("cuda_device", "")).split(",") if d.strip()]
    if len(devices) > 1:
        if cfg.get("cache_dit"):
            cfg["cache_dit"] = False
        if cfg.get("cache_vae"):
            cfg["cache_vae"] = False
    
    # ADDED v2.5.22: Validate video backend and 10-bit encoding consistency
    # Auto-disable 10-bit if ffmpeg backend not selected (prevents CLI errors)
    if cfg.get("use_10bit") and cfg.get("video_backend") != "ffmpeg":
        error_logger.warning("10-bit encoding requires ffmpeg backend, auto-disabling 10-bit")
        cfg["use_10bit"] = False
        cfg["_10bit_disabled_reason"] = "Requires video_backend=ffmpeg"

    return cfg


# Helper functions -------------------------------------------------------------
def _validate_cuda_devices(cuda_spec: str) -> Optional[str]:
    """Validate CUDA device specification using centralized error handling."""
    is_valid, error_msg = validate_cuda_spec(cuda_spec)
    return error_msg if not is_valid else None


def _expand_cuda_spec(cuda_spec: str) -> str:
    """
    DEPRECATED: Use shared.gpu_utils.expand_cuda_device_spec instead.
    Kept for backward compatibility.
    """
    return expand_cuda_device_spec(cuda_spec)


def _ffmpeg_available() -> bool:
    """Check if ffmpeg is available in PATH using centralized error handling."""
    is_available, _ = check_ffmpeg_available()
    return is_available


def _resolve_input_path(file_upload: Optional[str], manual_path: str, batch_enable: bool, batch_input: str) -> Tuple[str, Optional[str]]:
    """
    Resolve the input path from various sources with priority order:
    1. Batch input (if batch enabled)
    2. File upload (highest priority for single files)
    3. Manual path entry (fallback)
    
    Returns:
        Tuple of (input_path, original_filename)
        - input_path: Actual file path to process
        - original_filename: User's original filename (if from upload), None otherwise
    """
    if batch_enable and batch_input and batch_input.strip():
        return batch_input.strip(), None
    
    # Check if file_upload is a FileData dict or path string
    original_filename = None
    if file_upload:
        # Gradio can pass FileData as dict with orig_name
        if isinstance(file_upload, dict) and 'orig_name' in file_upload:
            original_filename = file_upload.get('orig_name')
            input_path = file_upload.get('path', '')
        else:
            input_path = str(file_upload).strip()
        
        if input_path:
            return input_path, original_filename
    
    return manual_path.strip() if manual_path else "", None


def _list_media_files(folder: str, video_exts: set, image_exts: set) -> List[str]:
    """List media files in a folder."""
    try:
        p = Path(normalize_path(folder))
        if not p.exists() or not p.is_dir():
            return []
        items = []
        for f in sorted(p.iterdir()):
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext in video_exts or ext in image_exts:
                items.append(str(f))
        return items
    except Exception:
        return []


# Preset helpers ---------------------------------------------------------------
def _seedvr2_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    """
    Convert argument list to settings dictionary.
    
    Uses SEEDVR2_ORDER for backward compatibility, but could be auto-detected
    using auto_detect_inputs() from preset_auto_serializer for future-proofing.
    
    To add a new control:
    1. Add to seedvr2_defaults() with default value
    2. Add to SEEDVR2_ORDER list (append at end for backward compat)
    3. Add to inputs_list in seedvr2_tab.py (same order as SEEDVR2_ORDER)
    4. That's it! Auto-merging ensures old presets still work.
    """
    return dict(zip(SEEDVR2_ORDER, args))


def _validate_preset_completeness(components: List[gr.components.Component]) -> None:
    """
    Validate that components list matches SEEDVR2_ORDER length.
    Helps catch mismatches during development.
    
    This is a development-time helper that ensures inputs_list and SEEDVR2_ORDER
    stay in sync. Remove or disable in production.
    """
    if len(components) != len(SEEDVR2_ORDER):
        error_msg = (
            f" PRESET MISMATCH: inputs_list has {len(components)} components "
            f"but SEEDVR2_ORDER has {len(SEEDVR2_ORDER)} keys.\n"
            f"This will cause preset save/load errors.\n"
            f"Expected keys: {SEEDVR2_ORDER}\n"
            f"Component count: {len(components)}"
        )
        error_logger.warning(error_msg)
        # Don't raise - just warn, as this is recoverable


def _apply_preset_to_values(
    preset: Dict[str, Any],
    defaults: Dict[str, Any],
    preset_manager: PresetManager,
    current: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Apply preset values to current settings."""
    base = defaults.copy()
    if current:
        base.update(current)
    merged = preset_manager.merge_config(base, preset)
    merged = _enforce_seedvr2_guardrails(merged, defaults, state=None)  # No state during preset load
    return [merged[key] for key in SEEDVR2_ORDER]


# Core processing functions -----------------------------------------------------
def _save_preprocessed_artifact(pre_path: Path, output_path_str: str) -> Optional[str]:
    """
    Save the preprocessed (downscaled) input next to outputs for inspection.

    Requirement:
    - Save into a `pre_processed/` folder inside the output folder
    - Use the SAME base name as the final output
    """
    try:
        if not pre_path or not pre_path.exists():
            return None

        outp = Path(output_path_str)
        parent = outp.parent if outp.suffix else outp.parent
        pre_dir = parent / "pre_processed"
        pre_dir.mkdir(parents=True, exist_ok=True)

        if outp.suffix:
            base = outp.stem
        else:
            base = outp.name

        if pre_path.is_dir():
            dest_dir = collision_safe_dir(pre_dir / base)
            shutil.copytree(pre_path, dest_dir, dirs_exist_ok=False)
            return str(dest_dir)

        dest_file = collision_safe_path(pre_dir / f"{base}{pre_path.suffix}")
        shutil.copy2(pre_path, dest_file)
        return str(dest_file)
    except Exception:
        return None


def _process_single_file(
    runner: Runner,
    settings: Dict[str, Any],
    global_settings: Dict[str, Any],
    seed_controls: Dict[str, Any],
    face_apply: bool,
    face_strength: float,
    run_logger: RunLogger,
    output_dir: Path,
    preview_only: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None
) -> Tuple[str, str, Optional[str], Optional[str], str, str, str]:
    """
    Process a single file with SeedVR2.
    Returns: (status, logs, output_video, output_image, chunk_info, chunk_summary, chunk_progress)
    """
    local_logs = []
    output_video = None
    output_image = None
    chunk_info_msg = "No chunking performed."
    chunk_summary = "Single pass (no chunking)."
    chunk_progress_msg = ""
    status = "Processing exited unexpectedly"
    audio_already_replaced = False
    final_returncode: Optional[int] = None

    # CONSOLE LOGGING: Print startup info so users can see what's happening
    print("\n" + "=" * 70, flush=True)
    print("SEEDVR2 SERVICE - STARTING PROCESSING", flush=True)
    print("=" * 70, flush=True)
    print(f"Input Path: {settings.get('input_path', 'Not set')}", flush=True)
    print(f"Model: {settings.get('dit_model', 'Not set')}", flush=True)
    print(f"Target Resolution: {settings.get('resolution', 'Auto')}p", flush=True)
    print(f"Batch Size: {settings.get('batch_size', 'Default')}", flush=True)
    print(f"CUDA Device: {settings.get('cuda_device', 'Default (0)')}", flush=True)
    print(f"Face Restore: {'Yes' if face_apply else 'No'}", flush=True)
    print(f"Preview Only: {'Yes' if preview_only else 'No'}", flush=True)
    print("-" * 70, flush=True)

    try:
        # Handle first-frame preview mode
        if preview_only:
            input_type = detect_input_type(settings["input_path"])
            
            if input_type == "video":
                # Extract first frame
                from shared.frame_utils import extract_first_frame
                
                if progress_cb:
                    progress_cb("Extracting first frame for preview...\n")
                
                success, frame_path, error = extract_first_frame(
                    settings["input_path"],
                    format="png"
                )
                
                if not success or not frame_path:
                    return (
                        f"Frame extraction failed: {error}",
                        error or "",
                        None,
                        None,
                        "Preview failed",
                        "Preview failed",
                        "Preview failed",
                    )
                
                # Process the extracted frame as an image
                preview_settings = settings.copy()
                preview_settings["input_path"] = frame_path
                preview_settings["output_format"] = "png"
                preview_settings["load_cap"] = 1
                
                if progress_cb:
                    progress_cb("Upscaling first frame...\n")
                
                result = runner.run_seedvr2(
                    preview_settings,
                    on_progress=lambda x: progress_cb(x) if progress_cb else None,
                    preview_only=True
                )
                
                if result.output_path and Path(result.output_path).exists():
                    output_image = result.output_path
                    status = "First-frame preview complete"
                    local_logs.append("Preview mode: Processed first frame only")
                    chunk_info_msg = "Preview: First frame extracted and upscaled"
                    chunk_summary = f"Preview output: {output_image}"
                    chunk_progress_msg = "Preview mode: 1/1 frames"
                else:
                    status = "Preview upscaling failed"
                    local_logs.append(result.log)
                    chunk_progress_msg = "Preview failed"
                    
                return status, "\n".join(local_logs), None, output_image, chunk_info_msg, chunk_summary, chunk_progress_msg
                
            else:
                # For images, just process normally with load_cap=1
                settings["load_cap"] = 1
                settings["output_format"] = "png"

        # Optional image latent-noise policy:
        # safeguard ON by default; checkbox can allow custom image latent noise.
        try:
            allow_custom_image_latent_noise = _coerce_bool(
                settings.get("force_latent_noise_zero_for_images"),
                default=False,
            )
            input_is_image = _looks_like_image_input(
                settings.get("input_path"),
                settings.get("_original_filename"),
            )
            if (not allow_custom_image_latent_noise) and input_is_image:
                prev_latent_noise = settings.get("latent_noise_scale")
                settings["latent_noise_scale"] = 0.0
                if prev_latent_noise not in (None, 0, 0.0, "0", "0.0"):
                    local_logs.append(
                        f"Image safeguard active: forcing latent_noise_scale 0 (was {prev_latent_noise})."
                    )
        except Exception:
            pass

        # -----------------------------------------------------------------
        #  Upscale-x sizing (compute SeedVR2 CLI params + optional pre-downscale)
        # -----------------------------------------------------------------
        # Single-image outputs: keep artifacts inside the per-run output folder.
        try:
            if (not preview_only) and detect_input_type(settings["input_path"]) == "image":
                run_root = Path(settings.get("_run_dir") or output_dir)
                run_root.mkdir(parents=True, exist_ok=True)
                override_raw = (settings.get("output_override") or "").strip()
                orig_name = settings.get("_original_filename") or Path(settings["input_path"]).name
                if override_raw:
                    override_path = Path(normalize_path(override_raw))
                    if override_path.suffix:
                        # Explicit file path provided by the user -> honor as-is.
                        settings["output_override"] = str(override_path)
                    else:
                        # Directory/root override -> use run-local file name.
                        settings["output_override"] = str(
                            collision_safe_path(Path(override_path) / f"{Path(orig_name).stem}.png")
                        )
                else:
                    # Default: per-run folder output path.
                    settings["output_override"] = str(
                        collision_safe_path(run_root / f"{Path(orig_name).stem}.png")
                    )
        except Exception:
            pass

        try:
            max_edge = int(settings.get("max_resolution", 0) or 0)
            scale_x = float(settings.get("upscale_factor") or seed_controls.get("upscale_factor_val") or 4.0)
            pre_down = bool(settings.get("pre_downscale_then_upscale", True))

            dims = get_media_dimensions(settings["input_path"])
            if dims:
                w, h = dims
                plan = estimate_seedvr2_upscale_plan_from_dims(
                    w,
                    h,
                    upscale_factor=scale_x,
                    max_edge=max_edge,
                    pre_downscale_then_upscale=pre_down,
                )

                # Apply computed CLI parameters
                if plan.seedvr2_resolution is not None:
                    settings["resolution"] = int(plan.seedvr2_resolution)
                settings["max_resolution"] = int(max_edge or 0)

                # If enabled and capped, pre-downscale the input media so the model runs at full scale_x
                if plan.pre_downscale_then_upscale and plan.preprocess_scale < 0.999999:
                    in_path = settings["input_path"]
                    in_type = detect_input_type(in_path)
                    pre_root = Path(output_dir) / "pre_processed"
                    pre_root.mkdir(parents=True, exist_ok=True)

                    if progress_cb:
                        progress_cb(
                            f" Preprocessing input: {w}x{h} -> {plan.preprocess_width}x{plan.preprocess_height} (x{plan.preprocess_scale:.3f})\n"
                        )

                    pre_out: Optional[Path] = None
                    if in_type == "video":
                        # Save the downscaled input into the user-visible output folder (run dir).
                        original_name = settings.get("_original_filename") or Path(in_path).name
                        pre_out = downscaled_video_path(output_dir, str(original_name))
                        ok, _err = scale_video(
                            Path(in_path),
                            Path(pre_out),
                            int(plan.preprocess_width),
                            int(plan.preprocess_height),
                            lossless=True,
                            audio_copy_first=True,
                            on_progress=(lambda x: progress_cb(x) if progress_cb else None),
                        )
                        if (not ok) or (not Path(pre_out).exists()):
                            pre_out = None
                    elif in_type == "image":
                        try:
                            import cv2  # type: ignore

                            img = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
                            if img is not None:
                                resized = cv2.resize(
                                    img,
                                    (int(plan.preprocess_width), int(plan.preprocess_height)),
                                    interpolation=cv2.INTER_LANCZOS4,
                                )
                                pre_out = collision_safe_path(
                                    pre_root / f"{Path(in_path).stem}_pre{plan.preprocess_width}x{plan.preprocess_height}{Path(in_path).suffix}"
                                )
                                cv2.imwrite(str(pre_out), resized)
                                if not pre_out.exists():
                                    pre_out = None
                        except Exception:
                            pre_out = None

                    # Directories (batch folders / frame folders) are handled per-item in batch mode.
                    if pre_out:
                        settings["_original_input_path_before_preprocess"] = settings["input_path"]
                        settings["_preprocessed_input_path"] = str(pre_out)
                        settings["input_path"] = str(pre_out)
                        if progress_cb:
                            progress_cb(f"Preprocess complete: {pre_out.name}\n")
        except Exception as e:
            if progress_cb:
                progress_cb(f"Preprocess sizing skipped: {str(e)[:120]}\n")

        # Guardrail: LAB color-correction can hit PyTorch 32-bit index limits for
        # very large per-batch tensors (frames * H * W * C). Clamp batch_size
        # to a safe 4n+1 value based on estimated output dimensions.
        try:
            if (not preview_only) and detect_input_type(settings["input_path"]) == "video":
                color_mode = str(settings.get("color_correction", "lab") or "").strip().lower()
                if color_mode == "lab":
                    dims_for_guard = get_media_dimensions(settings["input_path"])
                    if dims_for_guard:
                        in_w, in_h = int(dims_for_guard[0]), int(dims_for_guard[1])
                        out_w, out_h = _estimate_seedvr2_output_dims(
                            in_w,
                            in_h,
                            int(settings.get("resolution", 1080) or 1080),
                            int(settings.get("max_resolution", 0) or 0),
                        )
                        elems_per_frame = int(out_w) * int(out_h) * 3  # RGB
                        if elems_per_frame > 0:
                            max_safe_frames = max(1, int((I32_INDEX_LIMIT - 1) // elems_per_frame))
                            safe_bs = _largest_4n_plus_1_leq(max_safe_frames)
                            current_bs = int(settings.get("batch_size", 5) or 5)
                            if current_bs > safe_bs:
                                settings["batch_size"] = safe_bs
                                guard_msg = (
                                    f"Adjusting batch_size {current_bs}->{safe_bs} to avoid 32-bit index overflow "
                                    f"in LAB color correction ({out_w}x{out_h} output)."
                                )
                                local_logs.append(guard_msg)
                                if progress_cb:
                                    progress_cb(f"⚠️ {guard_msg}\n")
        except Exception:
            # Never block processing due to guardrail estimation issues.
            pass

        # Model loading check
        model_manager = get_model_manager()
        dit_model = settings.get("dit_model", "")
        # Model loading check
        model_manager = get_model_manager()
        dit_model = settings.get("dit_model", "")

        if not model_manager.is_model_loaded(ModelType.SEEDVR2, dit_model, **settings):
            if progress_cb:
                progress_cb(f"Loading model: {dit_model}...\n")
            if not runner.ensure_seedvr2_model_loaded(settings, lambda x: progress_cb(x) if progress_cb else None):
                return (
                    "Model load failed",
                    "",
                    None,
                    None,
                    "Model load failed",
                    "Model load failed",
                    "Model load failed",
                )
            if progress_cb:
                progress_cb("Model loaded successfully!\n")

            #  CHUNKING SYSTEM ARCHITECTURE - Two Complementary Methods:
        # 
        # METHOD 1: PySceneDetect Chunking (PREFERRED, UNIVERSAL)
        # --------------------------------------------------------
        # - Controlled by: Resolution & Scene Split tab  chunk_size_sec setting in shared state
        # - Settings: chunk_size_sec, chunk_overlap_sec, scene_threshold, min_scene_len
        # - How it works: Externally splits video into scenes using PySceneDetect,
        #   processes each scene separately, then concatenates with blending
        # - Works with: ALL models (SeedVR2, GAN, RIFE, FlashVSR+)
        # - Use when: Processing long videos, managing VRAM, or optimizing per-scene quality
        # 
        # METHOD 2: SeedVR2 Native Streaming (SEEDVR2-SPECIFIC OPTIMIZATION)
        # --------------------------------------------------------------------
        # - Controlled by: SeedVR2 tab  "Streaming Chunk Size (frames)" control
        # - Settings: chunk_size (frames), temporal_overlap
        # - How it works: CLI-internal memory-bounded processing, streams frames through model
        # - Works with: ONLY SeedVR2 (built into the CLI via --chunk_size flag)
        # - Use when: Processing VERY long videos where even scene chunks exceed VRAM
        # - Can combine: Both methods work together! PySceneDetect splits into scenes,
        #   then each scene uses native streaming internally for maximum efficiency.
        # 
        # Priority: PySceneDetect chunking happens FIRST (external), then native streaming
        # is applied within each chunk if enabled (settings["chunk_size"] > 0 passed to CLI).
        
        # Check if PySceneDetect chunking should be enabled (ONLY from Resolution tab now)
        # Legacy chunk_enable removed - chunking is now purely controlled by Resolution tab
        auto_chunk = bool(seed_controls.get("auto_chunk", True))
        chunk_size_sec = float(seed_controls.get("chunk_size_sec", 0) or 0)
        chunk_enabled_resolution_tab = auto_chunk or (chunk_size_sec > 0)
        
        should_chunk = (
            chunk_enabled_resolution_tab
            and not preview_only
            and detect_input_type(settings["input_path"]) == "video"
        )
        resume_requested = bool(
            str(settings.get("resume_run_dir") or "").strip() or bool(settings.get("_resume_run_requested", False))
        )
        if resume_requested and not should_chunk:
            msg = (
                "Resume run folder is only supported for chunk/scene-based video processing. "
                "Enable Resolution tab chunking and use the same settings as the original run."
            )
            local_logs.append(msg)
            return (
                "Resume unavailable for current mode",
                "\n".join(local_logs),
                None,
                None,
                "Resume requires chunk/scene mode",
                msg,
                msg,
            )
        chunk_count = 0
        scene_threshold = float(seed_controls.get("scene_threshold", 27.0))
        min_scene_len = float(seed_controls.get("min_scene_len", 1.0))

        if should_chunk:
            # Process with external PySceneDetect chunking
            # This splits video into scenes, processes each, then concatenates
            if resume_requested and progress_cb:
                progress_cb(
                    "Resume run folder detected. Resuming from last processed chunk "
                    "(same settings required) and continuing remaining chunks.\n"
                )
            if resume_requested:
                resume_ok, resume_msg = check_resume_available(Path(output_dir), "mp4")
                if not resume_ok:
                    msg = (
                        f"Resume requested but no resumable chunk outputs were found in {output_dir}. "
                        f"{resume_msg}"
                    )
                    local_logs.append(msg)
                    return (
                        "Resume failed",
                        "\n".join(local_logs),
                        None,
                        None,
                        "Resume folder is not resumable",
                        msg,
                        msg,
                    )
            completed_chunks = 0
            processing_chunk_idx = 0
            total_chunks_estimate = 1
            chunk_progress_updates: List[str] = []
            chunk_gallery_items: List[Any] = []
            chunk_media_by_index: Dict[int, Any] = {}

            def chunk_progress_callback(progress_val, desc="", **info):
                nonlocal completed_chunks, processing_chunk_idx, total_chunks_estimate
                nonlocal chunk_progress_updates, chunk_gallery_items, chunk_media_by_index

                import re

                chunk_idx = info.get("chunk_index")
                chunk_total = info.get("chunk_total")
                phase = str(info.get("phase") or "").strip().lower()
                desc_text = str(desc or "").strip()
                desc_lc = desc_text.lower()

                if chunk_total:
                    try:
                        total_chunks_estimate = max(1, int(chunk_total))
                    except Exception:
                        pass
                if chunk_idx is None:
                    m = re.search(r"(\d+)/(\d+)", desc_text)
                    if m:
                        try:
                            chunk_idx = int(m.group(1))
                            total_chunks_estimate = max(1, int(m.group(2)))
                        except Exception:
                            chunk_idx = None

                is_processing_event = phase == "processing" or desc_lc.startswith("processing chunk")
                is_completed_event = (
                    phase == "completed"
                    or "completed chunk" in desc_lc
                    or "finished chunk" in desc_lc
                    or "chunk complete" in desc_lc
                )

                if chunk_idx is not None:
                    try:
                        idx_int = int(chunk_idx)
                        if is_processing_event:
                            processing_chunk_idx = max(1, idx_int)
                        if is_completed_event:
                            completed_chunks = max(int(completed_chunks), idx_int)
                            processing_chunk_idx = max(processing_chunk_idx, idx_int)
                    except Exception:
                        pass
                elif is_completed_event:
                    completed_chunks = min(total_chunks_estimate, completed_chunks + 1)
                    processing_chunk_idx = max(processing_chunk_idx, completed_chunks)

                try:
                    idx_int = int(chunk_idx or completed_chunks or 0)
                except Exception:
                    idx_int = 0

                try:
                    chunk_output = info.get("chunk_output")
                    target_path = Path(chunk_output) if chunk_output else None
                    if target_path and not target_path.exists():
                        target_path = None

                    processed_dir = Path(seed_controls.get("processed_chunks_dir") or (Path(output_dir) / "processed_chunks"))
                    if target_path is None and idx_int > 0:
                        cand_mp4 = processed_dir / f"chunk_{idx_int:04d}_upscaled.mp4"
                        cand_dir = processed_dir / f"chunk_{idx_int:04d}_upscaled"
                        if cand_mp4.exists():
                            target_path = cand_mp4
                        elif cand_dir.exists() and cand_dir.is_dir():
                            target_path = cand_dir

                    if idx_int > 0 and target_path and idx_int not in chunk_media_by_index:
                        gallery_item: Optional[Any] = None
                        if target_path.is_file():
                            gallery_item = str(target_path)
                        elif target_path.is_dir():
                            video_candidates: List[Path] = []
                            for ext in sorted(SEEDVR2_VIDEO_EXTS):
                                video_candidates.extend(sorted(target_path.glob(f"*{ext}")))
                            if video_candidates:
                                gallery_item = str(video_candidates[0])
                            else:
                                image_candidates: List[Path] = []
                                for ext in sorted(SEEDVR2_IMAGE_EXTS):
                                    image_candidates.extend(sorted(target_path.glob(f"*{ext}")))
                                if image_candidates:
                                    gallery_item = str(image_candidates[0])

                        if gallery_item:
                            chunk_media_by_index[idx_int] = gallery_item
                            chunk_gallery_items = [chunk_media_by_index[i] for i in sorted(chunk_media_by_index)]
                            seed_controls["chunk_gallery_items"] = list(chunk_gallery_items)
                            # Backward-compatible key for older readers.
                            seed_controls["chunk_thumbnails"] = list(chunk_gallery_items)
                except Exception:
                    pass

                pct = max(0.0, min(100.0, float(progress_val or 0.0) * 100.0))
                if is_processing_event:
                    label = f"Processing chunk {max(1, processing_chunk_idx)}/{total_chunks_estimate} ({pct:.1f}%)"
                elif is_completed_event:
                    label = f"Completed chunk {completed_chunks}/{total_chunks_estimate} ({pct:.1f}%)"
                else:
                    label = f"Progress {pct:.1f}%"

                line = f"{label} | {desc_text}" if desc_text and desc_text.lower() not in label.lower() else label
                chunk_progress_updates.append(line)
                if len(chunk_progress_updates) > 200:
                    chunk_progress_updates = chunk_progress_updates[-200:]

                if progress_cb:
                    progress_cb(f"{line}\n")

            # Get ALL chunking params from Resolution tab (via seed_controls)
            # PySceneDetect parameters now managed centrally in Resolution tab
            scene_threshold = float(seed_controls.get("scene_threshold", 27.0))
            min_scene_len = float(seed_controls.get("min_scene_len", 1.0))
            settings["frame_accurate_split"] = bool(seed_controls.get("frame_accurate_split", True))

            # Forward per-chunk SeedVR2 CLI progress (e.g. "Upscaling batch 6/13")
            # so the Gradio UI can show intra-chunk percentages.
            def _seedvr2_chunk_process(chunk_settings: Dict[str, Any], on_progress=None) -> RunResult:
                return runner.run_seedvr2(
                    chunk_settings,
                    on_progress=(lambda x: progress_cb(x) if progress_cb else None),
                    preview_only=False,
                )

            rc, clog, final_out, chunk_count = chunk_and_process(
                runner,
                settings,
                scene_threshold=scene_threshold,
                min_scene_len=min_scene_len,
                work_dir=Path(output_dir),
                on_progress=lambda msg: progress_cb(msg) if progress_cb else None,
                chunk_seconds=0.0 if auto_chunk else chunk_size_sec,
                chunk_overlap=0.0 if auto_chunk else float(seed_controls.get("chunk_overlap_sec", 0) or 0),
                per_chunk_cleanup=bool(seed_controls.get("per_chunk_cleanup", False)),
                resume_from_partial=bool(settings.get("resume_chunking", False) or resume_requested),
                allow_partial=True,
                global_output_dir=str(output_dir),
                progress_tracker=chunk_progress_callback,
                process_func=_seedvr2_chunk_process,
                model_type="seedvr2",  # Explicitly specify SeedVR2 processing
            )

            output_path = final_out if final_out else None
            runner_canceled = bool(getattr(runner, "is_canceled", lambda: False)())
            if rc == 0:
                status = "Chunked upscale complete"
            elif output_path and runner_canceled:
                status = "Cancelled - partial upscale complete"
            elif output_path:
                status = f"Chunked upscale produced partial output ({rc})"
            else:
                status = f"Chunked upscale ended early ({rc})"
            output_video = output_path if output_path and output_path.lower().endswith(".mp4") else None
            output_image = None
            audio_already_replaced = bool(output_video)
            local_logs.append(clog)
            
            # Enhanced summary showing both chunking methods if applicable
            native_streaming_info = ""
            if settings.get("chunk_size", 0) > 0:
                native_streaming_info = f" + native streaming ({settings['chunk_size']} frames/chunk)"
            mode_label = "Auto (PySceneDetect scenes)" if auto_chunk else f"Static ({chunk_size_sec:g}s)"
            chunk_summary = f"{mode_label}: processed {chunk_count} chunks{native_streaming_info}. Final: {output_path}"
            
            latest_line = chunk_progress_updates[-1] if chunk_progress_updates else "Chunking in progress..."
            active_chunk = processing_chunk_idx if processing_chunk_idx > 0 else max(1, completed_chunks)
            chunk_info_msg = (
                f"Chunks completed: {completed_chunks}/{total_chunks_estimate}\n"
                f"Current chunk: {active_chunk}/{total_chunks_estimate}\n"
                f"Latest: {latest_line}"
            )
            chunk_progress_msg = "\n".join(chunk_progress_updates[-12:]) if chunk_progress_updates else "Chunking in progress..."
            
            # Final fallback: discover processed chunk outputs if callback data was sparse.
            if chunk_count > 0 and not chunk_gallery_items:
                try:
                    processed_dir = Path(seed_controls.get("processed_chunks_dir") or (Path(output_dir) / "processed_chunks"))
                    chunk_video_files = sorted(processed_dir.glob("chunk_*_upscaled.mp4"))
                    if not chunk_video_files:
                        chunk_video_files = sorted(processed_dir.glob("chunk_*_out.mp4"))

                    if chunk_video_files:
                        chunk_gallery_items = [str(f) for f in chunk_video_files]
                        seed_controls["chunk_gallery_items"] = list(chunk_gallery_items)
                        seed_controls["chunk_thumbnails"] = list(chunk_gallery_items)
                except Exception as e:
                    # Don't fail processing if fallback discovery fails.
                    if progress_cb:
                        progress_cb(f"Chunk output discovery failed: {str(e)}\n")
            
            result = RunResult(rc, output_path, clog)
            final_returncode = int(rc)
        else:
            # Process without external chunking
            # NOTE: SeedVR2 native streaming (--chunk_size in frames) is handled
            # directly by the CLI via settings["chunk_size"] parameter
            # This is independent of PySceneDetect chunking above
            result = runner.run_seedvr2(
                settings,
                on_progress=lambda x: progress_cb(x) if progress_cb else None,
                preview_only=preview_only
            )
            final_returncode = int(result.returncode)
            runner_canceled = bool(getattr(runner, "is_canceled", lambda: False)())
            
            # Add informative message if native streaming is being used
            native_chunk_size = settings.get("chunk_size", 0)
            if native_chunk_size > 0:
                if result.returncode == 0:
                    status = "Upscale complete (SeedVR2 native streaming)"
                elif runner_canceled:
                    status = "Cancelled (SeedVR2 native streaming)"
                else:
                    status = f"Upscale exited with code {result.returncode}"
                chunk_summary = f"SeedVR2 native streaming: {native_chunk_size} frames/chunk (CLI-internal, memory-efficient)"
                chunk_info_msg = f"Native streaming enabled: {native_chunk_size} frames/chunk\nTemporal overlap: {settings.get('temporal_overlap', 0)}\nMemory-bounded processing for long videos."
                chunk_progress_msg = (
                    "Cancelled by user during native streaming."
                    if runner_canceled
                    else "Native streaming: CLI-internal chunking"
                )
            else:
                if result.returncode == 0:
                    status = "Upscale complete"
                elif runner_canceled:
                    status = "Cancelled by user"
                else:
                    status = f"Upscale exited with code {result.returncode}"
                chunk_summary = "Single pass (entire video loaded at once)"
                chunk_info_msg = (
                    "Cancelled by user during single-pass processing"
                    if runner_canceled
                    else "Single-pass processing (entire video in memory)"
                )
                chunk_progress_msg = (
                    "Cancelled by user"
                    if runner_canceled
                    else "Single pass: no chunking"
                )

        # Preserve subprocess logs for ALL paths so OOM/error detection can inspect
        # the full CLI traceback even when processing exits with a non-zero code.
        if result.log and (not local_logs or local_logs[-1] != result.log):
            local_logs.append(result.log)

        # Extract output paths
        if result.output_path:
            output_video = result.output_path if result.output_path.lower().endswith(".mp4") else None
            output_image = result.output_path if not result.output_path.lower().endswith(".mp4") else None

            # Save preprocessed input (if we created one) alongside outputs
            pre_in = settings.get("_preprocessed_input_path")
            if pre_in and Path(pre_in).exists():
                local_logs.append(f"Downscaled input saved: {pre_in}")

            # Update state - track both directory AND file path for pinned comparison
            try:
                outp = Path(result.output_path)
                seed_controls["last_output_dir"] = str(outp.parent if outp.is_file() else outp)
                seed_controls["last_output_path"] = str(outp) if outp.is_file() else None
            except Exception:
                pass

            # Log the run - use live output_dir from global settings
            live_output_dir = Path(global_settings.get("output_dir", output_dir))
            telemetry_enabled = bool(global_settings.get("telemetry", True))
            if telemetry_enabled and bool(settings.get("save_metadata", True)):
                run_logger.write_summary(
                    Path(result.output_path) if result.output_path else live_output_dir,
                    {
                        "input": settings.get("_original_input_path_before_preprocess") or settings["input_path"],
                        "output": result.output_path,
                        "returncode": result.returncode,
                        "args": settings,
                        "face_global": face_apply,
                        "chunk_summary": chunk_summary,
                    },
                )

        # Apply face restoration if enabled
        if face_apply and output_video and Path(output_video).exists():
            restored = restore_video(
                output_video,
                strength=face_strength,
                on_progress=lambda x: progress_cb(x) if progress_cb else None
            )
            if restored:
                local_logs.append(f"Face-restored video saved to {restored} (strength {face_strength})")
                output_video = restored

        if face_apply and output_image and Path(output_image).exists():
            restored_img = restore_image(output_image, strength=face_strength)
            if restored_img:
                local_logs.append(f"Face-restored image saved to {restored_img} (strength {face_strength})")
                output_image = restored_img

        # Apply global image output format/quality preferences.
        if output_image and Path(output_image).exists() and not Path(output_image).is_dir():
            converted_image = _apply_image_output_preferences(
                output_image,
                settings.get("image_output_format", "png"),
                settings.get("image_output_quality", 95),
            )
            if converted_image and converted_image != output_image:
                local_logs.append(
                    f"Converted image output to {Path(converted_image).suffix.lower()} "
                    f"(quality {int(settings.get('image_output_quality', 95) or 95)}): {converted_image}"
                )
                output_image = converted_image

        # Apply FPS override from global Output settings.
        fps_val = settings.get("fps_override", 0) or seed_controls.get("fps_override_val", 0)
        if fps_val and fps_val > 0 and output_video and Path(output_video).exists():
            try:
                adjusted = ffmpeg_set_fps(Path(output_video), float(fps_val))
                output_video = str(adjusted)
                local_logs.append(f"FPS overridden to {fps_val}: {adjusted}")
            except Exception as e:
                local_logs.append(f"FPS override failed: {str(e)}")

        # Audio replacement is controlled by user-selected Output settings.
        # Do not force hidden fallback codecs here.
        if (not audio_already_replaced) and output_video and Path(output_video).exists():
            try:
                from shared.audio_utils import ensure_audio_on_video
                audio_src = settings.get("_original_input_path_before_preprocess") or settings.get("input_path")
                audio_codec = str(settings.get("audio_codec") or "copy")
                audio_bitrate = str(settings.get("audio_bitrate")) if settings.get("audio_bitrate") else None
                src_for_mux = Path(audio_src) if audio_src and Path(audio_src).exists() else Path(output_video)
                _changed, maybe_final, audio_err = ensure_audio_on_video(
                    video_path=Path(output_video),
                    audio_source_path=src_for_mux,
                    audio_codec=audio_codec,
                    audio_bitrate=audio_bitrate,
                    force_replace=True,
                    on_progress=(lambda x: progress_cb(x) if progress_cb else None),
                )
                if maybe_final and Path(maybe_final).exists():
                    output_video = str(maybe_final)
                if audio_err:
                    local_logs.append(f"Audio note: {audio_err}")
            except Exception as e:
                # Never fail the whole operation due to audio issues
                local_logs.append(f"Audio replacement skipped: {str(e)}")

        # Global RIFE post-process (adds *_xFPS file while preserving original output).
        if output_video and Path(output_video).exists():
            try:
                if global_rife_enabled(seed_controls):
                    out_settings = seed_controls.get("output_settings", {}) if isinstance(seed_controls, dict) else {}
                    if not isinstance(out_settings, dict):
                        out_settings = {}
                    mult = str(
                        seed_controls.get("global_rife_multiplier_val", out_settings.get("global_rife_multiplier", "x2")) or "x2"
                    )
                    precision = str(
                        seed_controls.get("global_rife_precision_val", out_settings.get("global_rife_precision", "fp32")) or "fp32"
                    ).lower()
                    local_logs.append(f"Global RIFE requested: {mult} ({'fp16' if precision == 'fp16' else 'fp32'})")
                rife_out, rife_msg = maybe_apply_global_rife(
                    runner=runner,
                    output_video_path=output_video,
                    seed_controls=seed_controls,
                    on_log=(lambda m: progress_cb(m) if progress_cb and m else None),
                    chunking_context={
                        "enabled": bool(chunk_count and chunk_count > 0),
                        "auto_chunk": bool(auto_chunk),
                        "chunk_size_sec": float(chunk_size_sec or 0),
                        "chunk_overlap_sec": 0.0 if auto_chunk else float(seed_controls.get("chunk_overlap_sec", 0) or 0),
                        "scene_threshold": float(scene_threshold or 27.0),
                        "min_scene_len": float(min_scene_len or 1.0),
                        "frame_accurate_split": bool(settings.get("frame_accurate_split", True)),
                        "per_chunk_cleanup": bool(seed_controls.get("per_chunk_cleanup", False)),
                    },
                )
                if rife_out and Path(rife_out).exists():
                    local_logs.append(f"Global RIFE output: {rife_out}")
                    output_video = rife_out
                elif rife_msg:
                    local_logs.append(rife_msg)
            except Exception as e:
                local_logs.append(f"Global RIFE skipped: {str(e)}")

        # Generate comparison video if enabled (new input vs output comparison feature)
        generate_comparison_video = seed_controls.get("generate_comparison_video_val", True)
        if generate_comparison_video and output_video and Path(output_video).exists():
            from shared.video_comparison_advanced import create_input_vs_output_comparison_video

            # Use original input before any downscaling/preprocessing
            original_input = settings.get("_original_input_path_before_preprocess") or settings["input_path"]
            video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}

            if Path(original_input).exists() and Path(original_input).suffix.lower() in video_exts:
                comparison_layout = seed_controls.get("comparison_video_layout_val", "auto")
                comparison_path = Path(output_video).parent / f"{Path(output_video).stem}_comparison.mp4"

                if progress_cb:
                    progress_cb("Generating input vs output comparison video...\n")

                success, comp_path, err = create_input_vs_output_comparison_video(
                    original_input_video=str(original_input),
                    upscaled_output_video=str(output_video),
                    comparison_output=str(comparison_path),
                    layout=comparison_layout,
                    label_input="Original",
                    label_output="Upscaled",
                    on_progress=progress_cb
                )

                if success:
                    local_logs.append(f"Comparison video created: {comp_path}")
                    if progress_cb:
                        progress_cb(f"Comparison video saved: {comp_path}\n")
                else:
                    local_logs.append(f"Comparison video generation failed: {err}")
                    if progress_cb:
                        progress_cb(f"Comparison video failed: {err}\n")

    except Exception as e:
        import traceback
        error_msg = f"Processing failed: {str(e)}"
        traceback_str = traceback.format_exc()
        
        # CONSOLE LOGGING: Print error details so users can see what went wrong
        print("\n" + "=" * 70, flush=True)
        print("SEEDVR2 PROCESSING FAILED", flush=True)
        print("=" * 70, flush=True)
        print(f"Error: {error_msg}", flush=True)
        print("-" * 70, flush=True)
        print("FULL TRACEBACK:", flush=True)
        print(traceback_str, flush=True)
        print("=" * 70, flush=True)
        
        local_logs.append(error_msg)
        local_logs.append(f"Traceback:\n{traceback_str}")
        status = "Processing failed"
        chunk_summary = "Failed"
        chunk_info_msg = f"Error: {error_msg}"
        chunk_progress_msg = "Error occurred"

    # Keep run folder self-documented for both video and image runs.
    try:
        run_dir_raw = settings.get("_run_dir")
        if run_dir_raw and (not preview_only):
            final_output_path = output_video or output_image
            effective_input = settings.get("_effective_input_path") or settings.get("input_path")
            input_kind_for_ctx = detect_input_type(str(effective_input or ""))
            pre_in = settings.get("_preprocessed_input_path")
            if input_kind_for_ctx == "image" and (not pre_in):
                snap = ensure_image_input_artifact(
                    Path(run_dir_raw),
                    str(effective_input or ""),
                    preferred_stem=Path(settings.get("_original_filename") or "used_input").stem,
                )
                if snap:
                    pre_in = str(snap)
                    settings["_preprocessed_input_path"] = pre_in
            finalize_run_context(
                Path(run_dir_raw),
                pipeline="seedvr2",
                status=str(status or ""),
                returncode=final_returncode,
                output_path=str(final_output_path) if final_output_path else None,
                original_input_path=str(
                    settings.get("_original_input_path_before_preprocess")
                    or settings.get("input_path")
                    or ""
                ),
                effective_input_path=str(effective_input) if effective_input else None,
                preprocessed_input_path=str(pre_in) if pre_in else None,
                input_kind=input_kind_for_ctx,
            )
    except Exception:
        pass

    return status, "\n".join(local_logs), output_video, output_image, chunk_info_msg, chunk_summary, chunk_progress_msg


# Comparison initialization ---------------------------------------------------
def comparison_html_slider():
    """
    Initialize comparison note with helpful instructions.
    
    The actual video comparison slider is created dynamically during processing
    using create_video_comparison_html from shared.video_comparison_slider.
    """
    return {
        'value': """
        <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;">
            <p style="color: #495057; font-size: 14px; margin: 0;">
                 <strong>Comparison View:</strong> Process a video or image to see before/after comparison.<br>
                Videos use interactive HTML5 slider with fullscreen support. Images use Gradio's ImageSlider.
            </p>
        </div>
        """
    }


# Core run/cancel/preset callbacks --------------------------------------------
def build_seedvr2_callbacks(
    preset_manager: PresetManager,
    runner: Runner,
    run_logger: RunLogger,
    global_settings: Dict[str, Any],
    shared_state: gr.State,
    output_dir: Path,
    temp_dir: Path,
):
    """
    Build SeedVR2 callback functions for the UI.
    
    Returns a dict of callbacks that can be wired to Gradio components.
    The callbacks handle preset management, validation, and processing orchestration.
    
    Note: Component order validation is performed when inputs_list is passed to save_preset.
    If you add new controls, update both SEEDVR2_ORDER and inputs_list in seedvr2_tab.py.
    """
    defaults = seedvr2_defaults()

    def refresh_presets(model_name: str = None, select_name: Optional[str] = None):
        """Refresh preset dropdown (all presets for tab, not model-specific)."""
        presets = preset_manager.list_presets("seedvr2")  # List all presets for tab
        
        # If no presets exist yet, return empty choices with no value
        if not presets:
            return gr.update(choices=[], value=None)
        
        # Determine which preset to select
        last_used = preset_manager.get_last_used_name("seedvr2", model_name) if model_name else None
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used and last_used in presets else presets[0])
        
        return gr.update(choices=presets, value=value)

    def save_preset(preset_name: str, model_name: str, current_preset_selection: str, *args):
        """
        Save a preset with automatic validation.
        
        If no name is entered, uses the currently selected preset name.
        Validates that args length matches SEEDVR2_ORDER to catch integration bugs early.
        """
        # If no name entered, use the currently selected preset from dropdown
        if not preset_name.strip() and current_preset_selection and current_preset_selection.strip():
            preset_name = current_preset_selection.strip()
        elif not preset_name.strip():
            return gr.update(), gr.update(value=" Enter a preset name or select an existing preset to overwrite"), *list(args)

        try:
            # Validate component count matches ORDER
            if len(args) != len(SEEDVR2_ORDER):
                error_msg = (
                    f" Preset schema mismatch: received {len(args)} values but expected {len(SEEDVR2_ORDER)}. "
                    f"This indicates inputs_list in seedvr2_tab.py is out of sync with SEEDVR2_ORDER. "
                    f"Preset saving aborted to prevent corruption."
                )
                error_logger.error(error_msg)
                return gr.update(), gr.update(value=f" {error_msg[:200]}..."), *list(args)
            
            payload = _seedvr2_dict_from_args(list(args))
            validated_payload = _enforce_seedvr2_guardrails(payload, defaults, state=None)
            
            preset_manager.save_preset_safe("seedvr2", model_name, preset_name.strip(), validated_payload)
            
            # Refresh dropdown with all presets (not model-specific)
            all_presets = preset_manager.list_presets("seedvr2")
            dropdown = gr.update(choices=all_presets, value=preset_name.strip())

            # Reload the validated values to ensure UI consistency
            current_map = dict(zip(SEEDVR2_ORDER, list(args)))
            loaded_vals = _apply_preset_to_values(validated_payload, defaults, preset_manager, current=current_map)

            return dropdown, gr.update(value=f" Saved preset '{preset_name}'"), *loaded_vals
        except Exception as e:
            import traceback
            error_logger.error(f"Error saving preset: {str(e)}\n{traceback.format_exc()}")
            return gr.update(), gr.update(value=f" Error saving preset: {str(e)}"), *list(args)

    def load_preset(preset_name: str, model_name: str, current_values: List[Any]):
        """
        Load a preset.
        
        FIXED: Now returns (*values, status_message) to match UI output expectations.
        UI expects: inputs_list + [preset_status, shared_state]
        So we return: (*values, status_markdown_update) and UI lambda appends shared_state
        """
        try:
            preset = preset_manager.load_preset_safe("seedvr2", model_name, preset_name)
            if preset:
                preset_manager.set_last_used("seedvr2", model_name, preset_name)
                preset = preset_manager.validate_preset_constraints(preset, "seedvr2", model_name)
                preset = _enforce_seedvr2_guardrails(preset, defaults, state=None)

            current_map = dict(zip(SEEDVR2_ORDER, current_values))
            values = _apply_preset_to_values(preset or {}, defaults, preset_manager, current=current_map)
            
            # Return values + status message (status is second-to-last, before shared_state)
            status_msg = f" Loaded preset '{preset_name}'" if preset else " Preset not found"
            return (*values, gr.update(value=status_msg))
        except Exception as e:
            print(f"Error loading preset {preset_name}: {e}")
            # Return current values + error status
            return (*current_values, gr.update(value=f" Error: {str(e)}"))

    def safe_defaults():
        """Get safe default values."""
        return [defaults[key] for key in SEEDVR2_ORDER]

    def check_resume_status(global_settings, output_format):
        """Check chunking resume status."""
        out_root = Path(global_settings.get("output_dir", output_dir))
        fmt = output_format or "mp4"

        # Search newest run folders first (0001, 0002, ...). This enables resuming even after restart.
        candidates: List[Path] = []
        try:
            for d in out_root.iterdir():
                if d.is_dir() and d.name.isdigit() and len(d.name) == 4:
                    candidates.append(d)
        except Exception:
            candidates = []

        candidates.sort(key=lambda p: int(p.name), reverse=True)
        for run_dir in candidates:
            try:
                available, message = check_resume_available(run_dir, fmt)
                if available:
                    return gr.update(value=f" {run_dir.name}: {message}", visible=True)
            except Exception:
                continue

        return gr.update(value=" No partial chunking session found to resume in the outputs folder.", visible=True)

    def cancel():
        """
        Cancel current processing and compile any partial outputs if available.

        Priority:
        1) Current/recent output run directories (new architecture)
        2) Recently completed batch outputs in outputs root
        3) Legacy temp-folder fallback (backward compatibility)
        """
        def _cancel_media_updates(path: Optional[str]) -> Tuple[Any, Any]:
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
            image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
            try:
                if path and Path(path).is_file():
                    ext = Path(path).suffix.lower()
                    if ext in video_exts:
                        src = Path(path)
                        preview_dir = Path(temp_dir) / "output_preview_cache"
                        preview_dir.mkdir(parents=True, exist_ok=True)
                        st = src.stat()
                        preview_name = f"{src.stem}_{st.st_size}_{st.st_mtime_ns}{src.suffix}"
                        preview_path = preview_dir / preview_name
                        if not preview_path.exists():
                            shutil.copy2(src, preview_path)
                        return gr.update(value=str(preview_path), visible=True), gr.update(value=None, visible=False)
                    if ext in image_exts:
                        return gr.update(value=None, visible=False), gr.update(value=path, visible=True)
            except Exception:
                pass
            return gr.update(value=None, visible=False), gr.update(value=None, visible=False)

        canceled = runner.cancel()
        if not canceled:
            return gr.update(value="No active process to cancel"), "", gr.update(), gr.update()

        compiled_output: Optional[str] = None
        merge_method = "none"
        recent_outputs: List[Path] = []
        live_output_root = Path(global_settings.get("output_dir", output_dir))

        state_snapshot = {}
        try:
            state_snapshot = shared_state.value if isinstance(shared_state.value, dict) else {}
        except Exception:
            state_snapshot = {}
        seed_state = state_snapshot.get("seed_controls", {}) if isinstance(state_snapshot, dict) else {}
        last_run_dir = seed_state.get("last_run_dir")
        audio_source = (
            seed_state.get("last_input_path")
            or seed_state.get("_original_input_path_before_preprocess")
            or None
        )
        audio_codec = str(seed_state.get("audio_codec_val") or "copy")
        audio_bitrate = seed_state.get("audio_bitrate_val") or None
        output_settings = seed_state.get("output_settings", {}) if isinstance(seed_state, dict) else {}
        if not isinstance(output_settings, dict):
            output_settings = {}

        # Preferred: salvage from run directories in outputs.
        try:
            from shared.chunking import salvage_partial_from_run_dir
            from shared.output_run_manager import recent_output_run_dirs

            run_dirs = recent_output_run_dirs(
                live_output_root,
                last_run_dir=str(last_run_dir) if last_run_dir else None,
                limit=30,
            )
            for run_dir in run_dirs:
                partial_path, method = salvage_partial_from_run_dir(
                    run_dir,
                    partial_basename="cancelled_partial",
                    audio_source=str(audio_source) if audio_source else None,
                    audio_codec=audio_codec,
                    audio_bitrate=str(audio_bitrate) if audio_bitrate else None,
                    encode_settings=output_settings,
                )
                if partial_path and Path(partial_path).exists():
                    compiled_output = str(partial_path)
                    merge_method = method
                    break
        except Exception as e:
            error_logger.warning(f"Output-run salvage failed during cancel: {e}")

        # Batch fallback: recently created outputs.
        if not compiled_output:
            try:
                batch_outputs: List[Path] = []
                for ext in [".mp4", ".avi", ".mov", ".mkv", ".png", ".jpg", ".jpeg", ".webp"]:
                    batch_outputs.extend(list(live_output_root.rglob(f"*_upscaled{ext}")))

                import time

                current_time = time.time()
                recent_outputs = [
                    f for f in batch_outputs if f.is_file() and (current_time - f.stat().st_mtime < 86400)
                ]
                if recent_outputs:
                    compiled_output = str(live_output_root)
                    merge_method = "batch_partial"
            except Exception as e:
                error_logger.warning(f"Batch partial recovery failed: {e}")

        # Legacy fallback: check temp files from older flow.
        if not compiled_output:
            try:
                temp_base = Path(global_settings.get("temp_dir", temp_dir))
                temp_files: List[Path] = []
                for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                    temp_files.extend(list(temp_base.glob(f"*{ext}")))
                for ext in [".png", ".jpg", ".jpeg"]:
                    temp_files.extend(list(temp_base.glob(f"*{ext}")))

                temp_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                if temp_files:
                    most_recent = temp_files[0]
                    from shared.path_utils import collision_safe_path

                    final_output = live_output_root / f"cancelled_partial_{most_recent.name}"
                    final_output = collision_safe_path(final_output)
                    if most_recent.is_file():
                        shutil.copy2(most_recent, final_output)
                        compiled_output = str(final_output)
                        merge_method = "latest_file"
                    elif most_recent.is_dir():
                        shutil.copytree(most_recent, final_output, dirs_exist_ok=True)
                        compiled_output = str(final_output)
                        merge_method = "latest_dir"
            except Exception as e:
                error_logger.warning(f"Fallback recovery failed during cancel: {e}")

        if compiled_output:
            merge_info = {
                "simple": "Chunks concatenated (no overlap detected)",
                "png_collection": "PNG frames collected from chunks",
                "batch_partial": f"Batch processing cancelled - {len(recent_outputs)} completed files saved in output folder",
                "latest_file": "Best-effort: Latest temp file copied (no proper merge)",
                "latest_dir": "Best-effort: Latest temp directory copied (no proper merge)",
            }.get(merge_method, "Unknown merge method")
            vid_upd, img_upd = _cancel_media_updates(compiled_output)
            return (
                gr.update(value=f"Cancelled - Partial output saved: {Path(compiled_output).name}\nMerge method: {merge_info}"),
                f"Partial results saved to: {compiled_output}\n\nMerge method: {merge_info}",
                vid_upd,
                img_upd,
            )

        return (
            gr.update(value="Cancelled - No partial outputs found"),
            "Processing was cancelled. No recoverable partial outputs were found in output runs or temp fallback.",
            gr.update(),
            gr.update(),
        )

    def open_outputs_folder_seedvr2(state: Dict[str, Any]):
        """Open outputs folder - delegates to shared utility (no code duplication)"""
        from shared.services.global_service import open_outputs_folder
        # Use live output_dir from global settings (may have changed since tab load)
        live_output_dir = str(global_settings.get("output_dir", output_dir))
        return open_outputs_folder(live_output_dir)
    
    def clear_temp_folder_seedvr2(confirm: bool):
        """Clear temp folder - delegates to shared utility (no code duplication)"""
        from shared.services.global_service import clear_temp_folder
        # Use live temp_dir from global settings (may have changed since tab load)
        live_temp_dir = str(global_settings.get("temp_dir", temp_dir))
        return clear_temp_folder(live_temp_dir, confirm)

    def get_model_loading_status():
        """Get current model loading status for UI display."""
        try:
            model_manager = get_model_manager()
            loaded_models = model_manager.get_loaded_models_info()
            current_model = model_manager.current_model_id

            if not loaded_models:
                return "No models loaded"

            status_lines = []
            for model_id, info in loaded_models.items():
                state = info["state"]
                marker = "" if state == "loaded" else "" if state == "loading" else ""
                current_marker = "  current" if model_id == current_model else ""
                status_lines.append(f"{marker} {info['model_name']} ({state}){current_marker}")

            return "\n".join(status_lines)
        except Exception as e:
            return f"Error getting model status: {str(e)}"

    def _format_int(value: Any) -> str:
        """Format integer with separators for UI display."""
        try:
            return f"{int(value):,}"
        except Exception:
            return "n/a"

    def _probe_video_frame_count(video_path: str) -> Tuple[Optional[int], str]:
        """
        Return total frame count for a video.

        Returns:
            (frame_count, source) where source is one of:
            - "exact": ffprobe decoded frame count (`nb_read_frames`)
            - "metadata": stream metadata (`nb_frames`)
            - "estimated": duration * fps fallback
            - "unknown": unavailable
        """
        def _parse_positive_int(raw_text: str) -> Optional[int]:
            raw = str(raw_text or "").strip()
            if not raw:
                return None
            for line in raw.splitlines():
                token = line.strip()
                if token.isdigit():
                    val = int(token)
                    if val > 0:
                        return val
            return None

        # 1) Exact frame count (can be slower but most accurate).
        try:
            proc = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0:
                val = _parse_positive_int(proc.stdout)
                if val is not None:
                    return val, "exact"
        except Exception:
            pass

        # 2) Stream metadata frame count (fast when present).
        try:
            proc = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_frames",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0:
                val = _parse_positive_int(proc.stdout)
                if val is not None:
                    return val, "metadata"
        except Exception:
            pass

        # 3) Fallback estimate.
        try:
            dur = get_media_duration_seconds(str(video_path))
            fps = get_media_fps(str(video_path))
            if dur and dur > 0 and fps and fps > 0:
                return max(1, int(round(float(dur) * float(fps)))), "estimated"
        except Exception:
            pass

        return None, "unknown"

    def _calculate_scene_frame_stats(scenes: List[Tuple[float, float]], fps: Optional[float]) -> Dict[str, Any]:
        """
        Convert scene time ranges to frame-count stats using frame-accurate boundaries.
        Mirrors the frame alignment logic used by chunk splitting.
        """
        if not scenes:
            return {}

        try:
            fps_val = float(fps or 0.0)
        except Exception:
            fps_val = 0.0

        if fps_val <= 0:
            return {"scene_count": len(scenes)}

        import math

        frame_counts: List[int] = []
        for start_sec, end_sec in scenes:
            try:
                start_f = float(start_sec)
                end_f = float(end_sec)
            except Exception:
                continue

            if end_f <= start_f:
                continue

            start_frame = int(math.floor(start_f * fps_val + 1e-9))
            end_frame = int(math.ceil(end_f * fps_val - 1e-9))
            if end_frame <= start_frame:
                end_frame = start_frame + 1

            frame_counts.append(max(1, end_frame - start_frame))

        if not frame_counts:
            return {"scene_count": len(scenes)}

        avg_frames = float(sum(frame_counts)) / float(len(frame_counts))
        return {
            "scene_count": len(frame_counts),
            "chunk_min_frames": int(min(frame_counts)),
            "chunk_avg_frames": float(round(avg_frames, 1)),
            "chunk_max_frames": int(max(frame_counts)),
        }

    def _auto_res_from_input(
        input_path: str,
        state: Dict[str, Any],
        on_progress: Optional[Callable[[int, str], None]] = None,
    ):
        """Compute dynamic sizing info for the new Upscale-x feature."""
        state = state or {"seed_controls": {}}
        state.setdefault("seed_controls", {})
        seed_controls = state.get("seed_controls", {})
        model_name = seed_controls.get("current_model") or defaults.get("dit_model")
        model_cache = seed_controls.get("resolution_cache", {}).get(model_name, {})

        def _emit_progress(pct: int, note: str = "") -> None:
            if not on_progress:
                return
            safe_pct = max(0, min(100, int(pct)))
            try:
                on_progress(safe_pct, str(note or ""))
            except TypeError:
                try:
                    on_progress(safe_pct)
                except Exception:
                    pass
            except Exception:
                pass

        _emit_progress(2, "Reading media metadata...")

        if not input_path:
            return gr.update(value="Provide an input to calculate sizing."), state

        p = Path(normalize_path(input_path))
        if not p.exists():
            return gr.update(value="Input path not found."), state

        dims = get_media_dimensions(str(p))
        if not dims:
            return gr.update(value=" Could not determine input dimensions."), state

        _emit_progress(16, "Calculating target resize plan...")
        w, h = dims
        input_short = min(w, h)
        input_long = max(w, h)

        # Pull settings with explicit precedence:
        # 1) live seed_controls (SeedVR2/Resolution tab current UI state),
        # 2) per-model cache fallback,
        # 3) service defaults.
        max_edge = int(model_cache.get("max_resolution_val", defaults.get("max_resolution", 0)) or 0)

        scale_raw = seed_controls.get("upscale_factor_val")
        if scale_raw is None:
            scale_raw = model_cache.get("upscale_factor_val", defaults.get("upscale_factor", 4.0))
        scale_x = float(scale_raw or defaults.get("upscale_factor", 4.0) or 4.0)

        pre_down = bool(seed_controls.get("ratio_downscale", model_cache.get("ratio_downscale", True)))

        plan = estimate_seedvr2_upscale_plan_from_dims(
            w,
            h,
            upscale_factor=scale_x,
            max_edge=max_edge,
            pre_downscale_then_upscale=pre_down,
        )
        _emit_progress(32, "Building sizing summary...")

        out_w = plan.final_saved_width or plan.resize_width
        out_h = plan.final_saved_height or plan.resize_height
        out_short = min(out_w, out_h)
        out_long = max(out_w, out_h)

        input_kind = detect_input_type(str(p))
        if input_kind == "unknown" and _looks_like_image_input(str(p), seed_controls.get("_original_filename")):
            input_kind = "image"
        is_video_input = input_kind == "video"
        input_format = (p.suffix or "").replace(".", "").upper() if p.suffix else "N/A"
        allow_custom_image_latent_noise = _coerce_bool(
            seed_controls.get(
                "force_latent_noise_zero_for_images_val",
                defaults.get("force_latent_noise_zero_for_images", False),
            ),
            default=bool(defaults.get("force_latent_noise_zero_for_images", False)),
        )

        def _safe(text: Any) -> str:
            return html.escape(str(text))

        def _stat_row(label: str, value: str, value_class: str = "") -> str:
            cls = "resolution-stat-val"
            if value_class:
                cls = f"{cls} {value_class}"
            return (
                '<div class="resolution-stat-row">'
                f'<div class="resolution-stat-key">{_safe(label)}</div>'
                f'<div class="{_safe(cls)}">{_safe(value)}</div>'
                "</div>"
            )

        def _build_card(title: str, rows: List[str]) -> str:
            body = "".join(rows) if rows else _stat_row("Info", "n/a")
            return (
                '<div class="resolution-stat-card">'
                f'<div class="resolution-stat-card-title">{_safe(title)}</div>'
                f"{body}"
                "</div>"
            )

        sizing_rows: List[str] = []
        runtime_rows: List[str] = []
        input_rows: List[str] = []
        chunk_rows: List[str] = []
        notes: List[str] = []

        sizing_rows.append(_stat_row("Input", f"{w}x{h} (short side: {input_short}px)"))

        target_line = f"upscale {scale_x:g}x"
        if max_edge and max_edge > 0:
            target_line += f", max edge {max_edge}px"
        if max_edge and max_edge > 0 and plan.cap_ratio < 0.999999:
            target_line += f" (effective {plan.effective_scale:.2f}x)"
        sizing_rows.append(_stat_row("Target Setting", target_line))

        # When max-edge cap reduces effective scale, show the cap-aware base size
        # that corresponds to the requested upscale factor.
        if scale_x > 0 and max_edge and max_edge > 0 and plan.cap_ratio < 0.999999:
            if plan.pre_downscale_then_upscale and plan.preprocess_scale < 0.999999:
                cap_base_line = (
                    f"actual preprocess: downscaled to {plan.preprocess_width}x{plan.preprocess_height}px, "
                    f"then upscaled {scale_x:g}x"
                )
            else:
                cap_base_w = max(1, int(round(float(out_w) / float(scale_x))))
                cap_base_h = max(1, int(round(float(out_h) / float(scale_x))))
                cap_base_line = (
                    f"equivalent cap-aware base: ~{cap_base_w}x{cap_base_h}px for {scale_x:g}x "
                    f"(no actual preprocess unless checkbox is ON)"
                )
            sizing_rows.append(_stat_row("Cap-Aware Upscale Path", cap_base_line))

        if plan.pre_downscale_then_upscale and plan.preprocess_scale < 0.999999:
            sizing_rows.append(
                _stat_row(
                    "Preprocess",
                    f"{w}x{h} -> {plan.preprocess_width}x{plan.preprocess_height} (x{plan.preprocess_scale:.3f})",
                )
            )

        resized_short = min(plan.resize_width, plan.resize_height)
        sizing_rows.append(
            _stat_row(
                "Resize Result",
                f"{plan.resize_width}x{plan.resize_height} (short side: {resized_short}px)",
            )
        )

        if plan.padded_width and plan.padded_height:
            sizing_rows.append(
                _stat_row(
                    "Padded for Model (16)",
                    f"{plan.padded_width}x{plan.padded_height} (trimmed after processing)",
                )
            )

        sizing_rows.append(
            _stat_row(
                "Final Saved Output",
                f"{int(out_w)}x{int(out_h)} (trimmed to even numbers)",
            )
        )

        mode_class = "is-neutral"
        if out_short < input_short:
            mode_line = f"Downscaling vs original input ({out_short}px < {input_short}px short side)"
            mode_class = "is-down"
            notes.append("Tip: set Upscale x >= 1.0 and/or increase Max Resolution to avoid downscaling.")
        elif out_short > input_short:
            mode_line = f"Upscaling vs original input ({out_short}px > {input_short}px short side)"
            mode_class = "is-up"
        else:
            mode_line = "Keep size vs original input (output short side matches input)"

        runtime_rows.append(_stat_row("Mode", mode_line, value_class=mode_class))

        if max_edge and max_edge > 0 and plan.cap_ratio < 0.999999:
            if plan.pre_downscale_then_upscale and plan.preprocess_scale < 0.999999:
                runtime_rows.append(
                    _stat_row(
                        "Actual Preprocess",
                        f"ON: input is pre-downscaled to {plan.preprocess_width}x{plan.preprocess_height} before model pass",
                    )
                )
            else:
                runtime_rows.append(
                    _stat_row(
                        "Actual Preprocess",
                        "OFF: input is NOT pre-downscaled; cap is enforced during CLI resize via --max_resolution",
                    )
                )

            requested_long = int(round(input_long * scale_x))
            if plan.pre_downscale_then_upscale and plan.preprocess_scale < 0.999999:
                clamp_line = (
                    f"original request ~{requested_long}px long side; preprocess+upscale path saved ~{out_long}px "
                    f"(ratio {plan.cap_ratio:.3f})"
                )
            else:
                clamp_line = (
                    f"requested ~{requested_long}px long side (from original input), "
                    f"capped to ~{out_long}px (ratio {plan.cap_ratio:.3f})"
                )
            runtime_rows.append(
                _stat_row(
                    "Max Edge Clamp",
                    clamp_line,
                )
            )

        if plan.seedvr2_resolution is not None:
            cli_line = f"--resolution {plan.seedvr2_resolution} --max_resolution {max_edge if max_edge > 0 else 0}"
            if max_edge and max_edge > 0 and plan.cap_ratio < 0.999999:
                if plan.pre_downscale_then_upscale and plan.preprocess_scale < 0.999999:
                    cli_line += " (preprocess path)"
                else:
                    cli_line += " (direct cap path, no preprocess)"
            runtime_rows.append(
                _stat_row(
                    "SeedVR2 CLI",
                    cli_line,
                )
            )

        if plan.notes:
            notes.extend([str(n) for n in plan.notes if str(n).strip()])

        input_rows.append(_stat_row("Input Detected", f"{input_kind.upper()}"))
        input_rows.append(_stat_row("Format", input_format))

        duration_sec: Optional[float] = None
        fps_val: Optional[float] = None
        total_frames: Optional[int] = None
        frame_count_source = "unknown"

        if is_video_input:
            _emit_progress(42, "Probing duration/FPS/frame stats...")
            cache_key = str(p)
            try:
                stat = p.stat()
                cache_key = f"{str(p)}|{int(stat.st_size)}|{int(stat.st_mtime_ns)}"
            except Exception:
                pass

            frame_cache = seed_controls.get("last_video_probe") or {}
            if frame_cache.get("cache_key") == cache_key:
                duration_sec = frame_cache.get("duration_sec")
                fps_val = frame_cache.get("fps")
                total_frames = frame_cache.get("total_frames")
                frame_count_source = str(frame_cache.get("frame_count_source") or "unknown")
            else:
                duration_sec = get_media_duration_seconds(str(p))
                fps_val = get_media_fps(str(p))
                total_frames, frame_count_source = _probe_video_frame_count(str(p))
                seed_controls["last_video_probe"] = {
                    "cache_key": cache_key,
                    "duration_sec": duration_sec,
                    "fps": fps_val,
                    "total_frames": total_frames,
                    "frame_count_source": frame_count_source,
                }
                state["seed_controls"] = seed_controls

            _emit_progress(56, "Collecting input stats...")

            if duration_sec and duration_sec > 0:
                input_rows.append(_stat_row("Duration", f"{float(duration_sec):.2f}s"))
            else:
                input_rows.append(_stat_row("Duration", "Unavailable"))

            if fps_val and fps_val > 0:
                input_rows.append(_stat_row("FPS", f"{float(fps_val):.3f}".rstrip("0").rstrip(".")))
            else:
                input_rows.append(_stat_row("FPS", "Unavailable"))

            if total_frames is not None and int(total_frames) > 0:
                frame_src_label = {
                    "exact": "exact",
                    "metadata": "stream metadata",
                    "estimated": "estimated",
                }.get(frame_count_source, "unknown")
                input_rows.append(_stat_row("Total Frames", f"{_format_int(total_frames)} ({frame_src_label})"))
            else:
                input_rows.append(_stat_row("Total Frames", "Unavailable"))

            # FPS forecast shown in the Sizing card (base + Global RIFE when enabled).
            output_settings = seed_controls.get("output_settings", {}) if isinstance(seed_controls, dict) else {}
            if not isinstance(output_settings, dict):
                output_settings = {}

            def _fmt_fps(v: Optional[float]) -> Optional[str]:
                try:
                    if v is None:
                        return None
                    fv = float(v)
                    if fv <= 0:
                        return None
                    return f"{fv:.3f}".rstrip("0").rstrip(".")
                except Exception:
                    return None

            try:
                fps_override = float(seed_controls.get("fps_override_val", output_settings.get("fps_override", 0)) or 0)
            except Exception:
                fps_override = 0.0

            base_output_fps: Optional[float] = fps_override if fps_override > 0 else (float(fps_val) if fps_val and fps_val > 0 else None)
            base_fps_text = _fmt_fps(base_output_fps)
            if base_fps_text:
                sizing_rows.append(_stat_row("Output FPS (Base)", base_fps_text))

            global_rife_on = global_rife_enabled(seed_controls)
            if global_rife_on:
                mult_raw = str(
                    seed_controls.get("global_rife_multiplier_val", output_settings.get("global_rife_multiplier", "x2")) or "x2"
                ).strip().lower()
                if mult_raw.startswith("x"):
                    mult_raw = mult_raw[1:]
                try:
                    mult_val = int(float(mult_raw))
                except Exception:
                    mult_val = 2
                mult_val = 2 if mult_val <= 2 else (4 if mult_val <= 4 else 8)

                if base_output_fps and base_output_fps > 0:
                    rife_output_fps = base_output_fps * float(mult_val)
                    rife_fps_text = _fmt_fps(rife_output_fps)
                    if rife_fps_text:
                        sizing_rows.append(_stat_row(f"Output FPS (+Global RIFE {mult_val}x)", rife_fps_text))
                else:
                    sizing_rows.append(_stat_row(f"Output FPS (+Global RIFE {mult_val}x)", "Unavailable (input FPS unknown)"))

            # Chunking info (Resolution tab global settings)
            _emit_progress(62, "Preparing chunk analysis...")
            auto_chunk = bool(seed_controls.get("auto_chunk", True))
            if auto_chunk:
                scene_threshold = float(seed_controls.get("scene_threshold", 27.0) or 27.0)
                min_scene_len = float(seed_controls.get("min_scene_len", 1.0) or 1.0)
                auto_detect_scenes = bool(seed_controls.get("auto_detect_scenes", True))
                scan = seed_controls.get("last_scene_scan") or {}

                try:
                    scan_path = normalize_path(scan.get("input_path")) if scan.get("input_path") else None
                except Exception:
                    scan_path = scan.get("input_path")

                cached_valid = (
                    bool(scan_path)
                    and scan_path == str(p)
                    and abs(float(scan.get("scene_threshold", scene_threshold)) - scene_threshold) < 1e-6
                    and abs(float(scan.get("min_scene_len", min_scene_len)) - min_scene_len) < 1e-6
                    and "scene_count" in scan
                )

                scene_count = int(scan.get("scene_count", 0) or 0) if cached_valid else 0
                chunk_min_frames = scan.get("chunk_min_frames") if cached_valid else None
                chunk_avg_frames = scan.get("chunk_avg_frames") if cached_valid else None
                chunk_max_frames = scan.get("chunk_max_frames") if cached_valid else None
                scene_scan_error = str(scan.get("error") or "").strip() if cached_valid else ""

                missing_scene_stats = (
                    scene_count > 0
                    and (chunk_min_frames is None or chunk_avg_frames is None or chunk_max_frames is None)
                )

                if (not cached_valid or missing_scene_stats) and auto_detect_scenes:
                    try:
                        from shared.chunking import detect_scenes

                        _emit_progress(68, "Running scene detection...")

                        def _scene_scan_progress(scene_pct: int) -> None:
                            safe_scene_pct = max(0, min(100, int(scene_pct)))
                            # Reserve the upper portion of analysis progress for scene scanning.
                            mapped_pct = 68 + int(round((safe_scene_pct / 100.0) * 28.0))
                            _emit_progress(mapped_pct, f"Running scene detection... {safe_scene_pct}%")

                        scenes = detect_scenes(
                            str(p),
                            threshold=scene_threshold,
                            min_scene_len=min_scene_len,
                            on_progress_pct=_scene_scan_progress,
                        )
                        scene_stats = _calculate_scene_frame_stats(scenes or [], fps_val)
                        scene_count = int(scene_stats.get("scene_count", len(scenes or [])) or 0)
                        chunk_min_frames = scene_stats.get("chunk_min_frames")
                        chunk_avg_frames = scene_stats.get("chunk_avg_frames")
                        chunk_max_frames = scene_stats.get("chunk_max_frames")
                        scene_scan_error = ""

                        scan_payload: Dict[str, Any] = {
                            "input_path": str(p),
                            "scene_threshold": scene_threshold,
                            "min_scene_len": min_scene_len,
                            "scene_count": scene_count,
                            "success": scene_count > 0,
                        }
                        if chunk_min_frames is not None:
                            scan_payload["chunk_min_frames"] = int(chunk_min_frames)
                        if chunk_avg_frames is not None:
                            scan_payload["chunk_avg_frames"] = float(chunk_avg_frames)
                        if chunk_max_frames is not None:
                            scan_payload["chunk_max_frames"] = int(chunk_max_frames)
                        if total_frames is not None:
                            scan_payload["total_frames"] = int(total_frames)

                        seed_controls["last_scene_scan"] = scan_payload
                        state["seed_controls"] = seed_controls
                        _emit_progress(96, "Building summary cards...")
                    except Exception as e:
                        scene_count = 0
                        chunk_min_frames = None
                        chunk_avg_frames = None
                        chunk_max_frames = None
                        scene_scan_error = str(e)
                        seed_controls["last_scene_scan"] = {
                            "input_path": str(p),
                            "scene_threshold": scene_threshold,
                            "min_scene_len": min_scene_len,
                            "scene_count": 0,
                            "success": False,
                            "error": str(e),
                        }
                        state["seed_controls"] = seed_controls
                elif cached_valid and scene_count > 0:
                    _emit_progress(82, "Using cached scene detection results...")
                elif not auto_detect_scenes:
                    _emit_progress(74, "Scene detection disabled; using static chunk rules.")

                chunk_rows.append(_stat_row("Chunk Mode", "Auto Scene Detect (PySceneDetect)"))
                chunk_rows.append(
                    _stat_row(
                        "Scene Settings",
                        f"threshold={scene_threshold:g}, min_len={min_scene_len:g}s, overlap=0",
                    )
                )

                if scene_count > 0:
                    chunk_rows.append(_stat_row("Detected Scenes", _format_int(scene_count)))
                    has_all_chunk_stats = True
                    if chunk_min_frames is not None:
                        chunk_rows.append(_stat_row("Min Frames / Chunk", _format_int(chunk_min_frames)))
                    else:
                        has_all_chunk_stats = False
                    if chunk_avg_frames is not None:
                        chunk_rows.append(_stat_row("Avg Frames / Chunk", f"{float(chunk_avg_frames):.1f}"))
                    else:
                        has_all_chunk_stats = False
                    if chunk_max_frames is not None:
                        chunk_rows.append(_stat_row("Max Frames / Chunk", _format_int(chunk_max_frames)))
                    else:
                        has_all_chunk_stats = False
                    if not has_all_chunk_stats:
                        chunk_rows.append(_stat_row("Chunk Frame Stats", "Unavailable for current cached scan."))
                elif scene_scan_error:
                    chunk_rows.append(_stat_row("Auto Chunk Status", f"Scene scan failed: {scene_scan_error}"))
                elif not auto_detect_scenes:
                    chunk_rows.append(_stat_row("Auto Chunk Status", "Auto scene detection is disabled in Resolution tab."))
                else:
                    chunk_rows.append(_stat_row("Auto Chunk Status", "Scene scan failed."))
            else:
                _emit_progress(74, "Static chunking mode selected.")
                chunk_size = float(model_cache.get("chunk_size_sec", seed_controls.get("chunk_size_sec", 0) or 0))
                chunk_overlap = float(model_cache.get("chunk_overlap_sec", seed_controls.get("chunk_overlap_sec", 0) or 0))

                chunk_rows.append(_stat_row("Chunk Mode", "Static"))
                if chunk_size <= 0:
                    chunk_rows.append(_stat_row("Chunking", "Disabled (chunk size = 0)"))
                elif chunk_overlap >= chunk_size:
                    chunk_rows.append(_stat_row("Chunking", "Invalid settings: overlap must be smaller than chunk size."))
                else:
                    import math

                    if duration_sec and duration_sec > 0:
                        est_chunks = math.ceil(float(duration_sec) / max(0.001, chunk_size - chunk_overlap))
                        chunk_rows.append(_stat_row("Estimated Chunks", f"~{_format_int(est_chunks)}"))
                        chunk_rows.append(
                            _stat_row(
                                "Window",
                                f"{chunk_size:g}s with {chunk_overlap:g}s overlap",
                            )
                        )
                    if fps_val and fps_val > 0:
                        est_chunk_frames = max(1, int(round(float(chunk_size) * float(fps_val))))
                        chunk_rows.append(_stat_row("Approx Frames / Chunk", _format_int(est_chunk_frames)))
        else:
            _emit_progress(74, "Input is not a video; scene detection skipped.")
            chunk_rows.append(_stat_row("Chunk Stats", "Chunk frame stats are available for video inputs."))
            if input_kind == "image":
                chunk_rows.append(
                    _stat_row(
                        "Latent Noise Scale",
                        (
                            "Custom mode ON: using configured slider value for image inputs."
                            if allow_custom_image_latent_noise
                            else "Default safeguard ON: forced to 0 for image inputs."
                        ),
                    )
                )

        left_cards = [
            _build_card("Sizing", sizing_rows),
            _build_card("Runtime", runtime_rows),
        ]
        right_cards = [
            _build_card("Input Stats", input_rows),
            _build_card("Chunk Stats", chunk_rows),
        ]

        notes_html = ""
        if notes:
            note_items = "".join(
                f'<div class="resolution-note-item">{_safe(note)}</div>' for note in notes if str(note).strip()
            )
            if note_items:
                notes_html = f'<div class="resolution-notes">{note_items}</div>'

        html_block = (
            '<div class="resolution-stats-shell">'
            '<div class="resolution-stats-grid">'
            f'<div class="resolution-stats-col">{"".join(left_cards)}</div>'
            f'<div class="resolution-stats-col">{"".join(right_cards)}</div>'
            "</div>"
            f"{notes_html}"
            "</div>"
        )
        _emit_progress(100, "Finalizing analysis report...")
        return gr.update(value=html_block, visible=True), state

    def run_action(
        uploaded_file,
        *args,
        preview_only: bool = False,
        state: Dict[str, Any] = None,
        progress=None,
        global_settings_snapshot: Dict[str, Any] | None = None,
        _global_settings: Dict[str, Any] = global_settings,
    ):
        """Main processing action with streaming support and gr.Progress integration."""
        try:
            global_settings = (
                dict(global_settings_snapshot)
                if isinstance(global_settings_snapshot, dict)
                else dict(_global_settings)
            )
            state = state or {"seed_controls": {}, "operation_status": "ready"}
            state["operation_status"] = "running"
            # Ensure a previous cancel does not block a fresh run.
            try:
                reset_cancel = getattr(runner, "reset_cancel_state", None)
                if callable(reset_cancel):
                    reset_cancel()
            except Exception:
                pass
            # Clear any previous VRAM OOM banner at the start of a new run.
            clear_vram_oom_alert(state)
            seed_controls = state.get("seed_controls", {})
            output_settings = seed_controls.get("output_settings", {}) if isinstance(seed_controls, dict) else {}
            if not isinstance(output_settings, dict):
                output_settings = {}
            global_gpu_device = get_global_gpu_override(seed_controls, global_settings)
            seed_controls["global_gpu_device_val"] = global_gpu_device
            seed_controls["global_rife_cuda_device_val"] = "" if global_gpu_device == "cpu" else global_gpu_device
            state["seed_controls"] = seed_controls

            # Keep legacy cached keys synced with Output tab source-of-truth values.
            # This prevents tab switches/run starts from reverting Global RIFE state.
            if output_settings:
                global_rife_on = bool(output_settings.get("frame_interpolation", False))
                seed_controls["frame_interpolation_val"] = global_rife_on
                seed_controls["global_rife_enabled_val"] = global_rife_on
                seed_controls["global_rife_multiplier_val"] = output_settings.get("global_rife_multiplier", "x2") or "x2"
                seed_controls["global_rife_model_val"] = (
                    str(output_settings.get("global_rife_model", "") or "").strip() or get_rife_default_model()
                )
                precision = str(output_settings.get("global_rife_precision", "fp32") or "fp32").strip().lower()
                seed_controls["global_rife_precision_val"] = "fp16" if precision == "fp16" else "fp32"
                seed_controls["global_rife_cuda_device_val"] = "" if global_gpu_device == "cpu" else global_gpu_device
                seed_controls["global_rife_process_chunks_val"] = bool(output_settings.get("global_rife_process_chunks", True))
                state["seed_controls"] = seed_controls

            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
            image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

            def _prepare_output_video_preview_copy(video_path: str) -> str:
                """
                Create/reuse an isolated preview copy for Gradio's Video component.
                This prevents UI preview/transcode logic from touching the real output file.
                """
                src = Path(str(video_path))
                preview_dir = Path(temp_dir) / "output_preview_cache"
                preview_dir.mkdir(parents=True, exist_ok=True)
                st = src.stat()
                preview_name = f"{src.stem}_{st.st_size}_{st.st_mtime_ns}{src.suffix}"
                preview_path = preview_dir / preview_name
                if not preview_path.exists():
                    shutil.copy2(src, preview_path)
                return str(preview_path)

            def _media_updates(video_path: Optional[str], image_path: Optional[str]) -> tuple[Any, Any]:
                """
                Return (output_video_update, output_image_update) for the merged output panel.
                """
                try:
                    if video_path and not Path(video_path).is_dir():
                        if Path(video_path).suffix.lower() in video_exts:
                            preview_path = _prepare_output_video_preview_copy(video_path)
                            return gr.update(value=preview_path, visible=True), gr.update(value=None, visible=False)
                    if image_path and not Path(image_path).is_dir():
                        if Path(image_path).suffix.lower() in image_exts:
                            return gr.update(value=None, visible=False), gr.update(value=image_path, visible=True)
                except Exception:
                    pass
                return gr.update(value=None, visible=False), gr.update(value=None, visible=False)

            def _processing_indicator(title: str = "Upscale in progress..."):
                html = (
                    '<div class="processing-banner">'
                    '<div class="processing-spinner"></div>'
                    '<div class="processing-col">'
                    f'<div class="processing-text">{title}</div>'
                    '<div class="processing-sub">Scroll down to see run logs and completed chunk previews.</div>'
                    "</div></div>"
                )
                return gr.update(value=html, visible=True)

            indicator_off = gr.update(value="", visible=False)

            # Parse settings
            settings = dict(zip(SEEDVR2_ORDER, list(args)))
            settings["force_latent_noise_zero_for_images"] = _coerce_bool(
                settings.get("force_latent_noise_zero_for_images"),
                default=bool(defaults.get("force_latent_noise_zero_for_images", False)),
            )
            seed_controls["force_latent_noise_zero_for_images_val"] = settings["force_latent_noise_zero_for_images"]
            if preview_only:
                # Preview should never trigger run-folder resume semantics.
                settings["resume_run_dir"] = ""
            if settings.get("batch_enable"):
                # Resume folder mode is single-run only.
                settings["resume_run_dir"] = ""
            settings["cuda_device"] = "" if global_gpu_device == "cpu" else global_gpu_device

            def _normalize_offload_device(value: Any) -> str:
                text = str(value or "").strip().lower()
                if not text:
                    return "none"
                if text in {"none", "cpu"}:
                    return text
                if text.startswith("cuda:"):
                    text = text.split(":", 1)[1].strip()
                if text.isdigit():
                    return "cpu" if global_gpu_device == "cpu" else global_gpu_device
                return "cpu" if global_gpu_device == "cpu" else "none"

            settings["dit_offload_device"] = _normalize_offload_device(settings.get("dit_offload_device"))
            settings["vae_offload_device"] = _normalize_offload_device(settings.get("vae_offload_device"))
            settings["tensor_offload_device"] = _normalize_offload_device(settings.get("tensor_offload_device"))

            # Shared Output tab values are the source of truth for these cross-model controls.
            settings["save_metadata"] = bool(
                seed_controls.get(
                    "save_metadata_val",
                    output_settings.get("save_metadata", defaults.get("save_metadata", True)),
                )
            )
            try:
                settings["fps_override"] = float(
                    seed_controls.get(
                        "fps_override_val",
                        output_settings.get("fps_override", defaults.get("fps_override", 0)),
                    )
                    or 0
                )
            except Exception:
                settings["fps_override"] = 0.0

            backend_val = str(
                seed_controls.get(
                    "seedvr2_video_backend_val",
                    output_settings.get("seedvr2_video_backend", defaults.get("video_backend", "opencv")),
                )
                or "opencv"
            ).strip().lower()
            if backend_val not in {"opencv", "ffmpeg"}:
                backend_val = "opencv"
            settings["video_backend"] = backend_val
            settings["use_10bit"] = bool(
                seed_controls.get(
                    "seedvr2_use_10bit_val",
                    output_settings.get("seedvr2_use_10bit", defaults.get("use_10bit", False)),
                )
            ) and backend_val == "ffmpeg"

            # Sync live SeedVR2 sizing controls into shared cache before guardrails.
            # Prevents stale global/per-model cached values from overriding current UI controls.
            try:
                seed_controls["upscale_factor_val"] = float(settings.get("upscale_factor") or seed_controls.get("upscale_factor_val") or 4.0)
            except Exception:
                pass
            try:
                live_max_res = int(settings.get("max_resolution") or 0)
                if live_max_res >= 0:
                    seed_controls.pop("max_resolution_val", None)
                    seed_controls.pop("enable_max_target", None)
            except Exception:
                pass
            seed_controls["ratio_downscale"] = bool(
                settings.get("pre_downscale_then_upscale", seed_controls.get("ratio_downscale", True))
            )
            state["seed_controls"] = seed_controls

            settings = _enforce_seedvr2_guardrails(settings, defaults, state=state)  # Pass state for resolution tab integration

            # Validate inputs - now extracts original filename from upload
            input_path, original_filename = _resolve_input_path(
                uploaded_file,
                settings["input_path"],
                settings["batch_enable"],
                settings["batch_input_path"]
            )
            settings["input_path"] = normalize_path(input_path)
            # Ensure original filename is always available for output naming.
            # Gradio may provide only a path string (without FileData.orig_name).
            if original_filename and str(original_filename).strip():
                original_filename = Path(str(original_filename).strip()).name
            elif settings["input_path"]:
                original_filename = Path(settings["input_path"]).name
            else:
                original_filename = None

            # Resume folder mode ignores normal input fields; recover source input from run folder metadata/artifacts.
            resume_run_dir_raw = str(settings.get("resume_run_dir") or "").strip()
            if resume_run_dir_raw:
                resume_run_dir = Path(normalize_path(resume_run_dir_raw))
                if not (resume_run_dir.exists() and resume_run_dir.is_dir()):
                    yield (
                        "Resume folder not found",
                        f"Configured resume folder does not exist: {resume_run_dir}",
                        "",
                        gr.update(value=None, visible=False),
                        gr.update(value=None, visible=False),
                        "No chunks",
                        "",
                        "",
                        gr.update(value="", visible=False),
                        gr.update(value=None, visible=False),
                        gr.update(value="", visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        state,
                    )
                    return

                recovered_input, recovered_name, _recovered_source = resolve_resume_input_from_run_dir(resume_run_dir)
                if recovered_input is None:
                    yield (
                        "Resume input not found",
                        (
                            f"Could not recover input source from resume folder: {resume_run_dir}. "
                            "Expected run_context.json input path or run_metadata/downscaled artifact."
                        ),
                        "",
                        gr.update(value=None, visible=False),
                        gr.update(value=None, visible=False),
                        "No chunks",
                        "",
                        "",
                        gr.update(value="", visible=False),
                        gr.update(value=None, visible=False),
                        gr.update(value="", visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        state,
                    )
                    return

                settings["input_path"] = str(recovered_input)
                if recovered_name and str(recovered_name).strip():
                    original_filename = Path(str(recovered_name).strip()).name
                elif not original_filename:
                    original_filename = Path(str(recovered_input)).name

            settings["_original_filename"] = original_filename  # Store for output naming
            state["seed_controls"]["last_input_path"] = settings["input_path"]
            state["seed_controls"]["_original_filename"] = original_filename

            if not settings["input_path"] or not Path(settings["input_path"]).exists():
                yield (
                    "Input path missing or not found",  # status_box
                    "",  # log_box
                    "",  # progress_indicator
                    gr.update(value=None, visible=False),  # output_video
                    gr.update(value=None, visible=False),  # output_image
                    "No chunks",  # chunk_info
                    "",  # resume_status
                    "",  # chunk_progress
                    gr.update(value="", visible=False),  # comparison_note
                    gr.update(value=None, visible=False),  # image_slider
                    gr.update(value="", visible=False),  # video_comparison_html
                    gr.update(visible=False),  # chunk_gallery
                    gr.update(visible=False),  # chunk_preview_video
                    gr.update(visible=False),  # batch_gallery
                    state  # shared_state
                )
                return

            # Expand "all" to device list if specified
            cuda_device_raw = settings.get("cuda_device", "")
            if cuda_device_raw:
                settings["cuda_device"] = _expand_cuda_spec(cuda_device_raw)
            
            # Validate CUDA devices (using shared GPU utility)
            cuda_warning = validate_cuda_device_spec(settings.get("cuda_device", ""))
            if cuda_warning:
                yield (
                    f"{cuda_warning}",  # status_box
                    "",  # log_box
                    "",  # progress_indicator
                    gr.update(value=None, visible=False),  # output_video
                    gr.update(value=None, visible=False),  # output_image
                    "No chunks",  # chunk_info
                    "",  # resume_status
                    "",  # chunk_progress
                    gr.update(value="", visible=False),  # comparison_note
                    gr.update(value=None, visible=False),  # image_slider
                    gr.update(value="", visible=False),  # video_comparison_html
                    gr.update(visible=False),  # chunk_gallery
                    gr.update(visible=False),  # chunk_preview_video
                    gr.update(visible=False),  # batch_gallery
                    state  # shared_state
                )
                return

            # Check ffmpeg availability
            if not _ffmpeg_available():
                yield (
                    "ffmpeg not found in PATH. Install ffmpeg and retry.",  # status_box
                    "",  # log_box
                    "",  # progress_indicator
                    gr.update(value=None, visible=False),  # output_video
                    gr.update(value=None, visible=False),  # output_image
                    "No chunks",  # chunk_info
                    "",  # resume_status
                    "",  # chunk_progress
                    gr.update(value="", visible=False),  # comparison_note
                    gr.update(value=None, visible=False),  # image_slider
                    gr.update(value="", visible=False),  # video_comparison_html
                    gr.update(visible=False),  # chunk_gallery
                    gr.update(visible=False),  # chunk_preview_video
                    gr.update(visible=False),  # batch_gallery
                    state  # shared_state
                )
                return

            # Check disk space (require at least 5GB free)
            output_path = Path(global_settings.get("output_dir", output_dir))
            has_space, space_warning = check_disk_space(output_path, required_mb=5000)
            if not has_space:
                yield (
                    space_warning or "Insufficient disk space",  # status_box
                    f"Free up disk space before processing. Recommended: 5GB+ free",  # log_box
                    "",  # progress_indicator
                    gr.update(value=None, visible=False),  # output_video
                    gr.update(value=None, visible=False),  # output_image
                    "No chunks",  # chunk_info
                    "",  # resume_status
                    "",  # chunk_progress
                    gr.update(value="", visible=False),  # comparison_note
                    gr.update(value=None, visible=False),  # image_slider
                    gr.update(value="", visible=False),  # video_comparison_html
                    gr.update(visible=False),  # chunk_gallery
                    gr.update(visible=False),  # chunk_preview_video
                    gr.update(visible=False),  # batch_gallery
                    state  # shared_state
                )
                return
            elif space_warning:
                # Low space but might work - show warning
                yield (
                    f"{space_warning}",  # status_box
                    "Low disk space detected. Processing may fail if output is large.",  # log_box
                    "",  # progress_indicator
                    gr.update(value=None, visible=False),  # output_video
                    gr.update(value=None, visible=False),  # output_image
                    "Disk space warning",  # chunk_info
                    "",  # resume_status
                    "",  # chunk_progress
                    gr.update(value="", visible=False),  # comparison_note
                    gr.update(value=None, visible=False),  # image_slider
                    gr.update(value="", visible=False),  # video_comparison_html
                    gr.update(visible=False),  # chunk_gallery
                    gr.update(visible=False),  # chunk_preview_video
                    gr.update(visible=False),  # batch_gallery
                    state  # shared_state
                )

            # Setup processing parameters
            face_apply = bool(settings.get("face_restore_after_upscale", False)) or bool(global_settings.get("face_global", False))
            face_strength = float(global_settings.get("face_strength", 0.5))

            # Apply shared sizing values from Resolution & Scene Split tab (global cache)
            if seed_controls.get("upscale_factor_val") is not None:
                try:
                    settings["upscale_factor"] = float(seed_controls["upscale_factor_val"])
                except Exception:
                    pass
            if "ratio_downscale" in seed_controls:
                settings["pre_downscale_then_upscale"] = bool(seed_controls.get("ratio_downscale", True))

            chunk_size_sec = float(seed_controls.get("chunk_size_sec", 0) or 0)
            chunk_overlap_sec = float(seed_controls.get("chunk_overlap_sec", 0) or 0)
            per_chunk_cleanup = seed_controls.get("per_chunk_cleanup", False)

            settings["chunk_size_sec"] = chunk_size_sec
            settings["chunk_overlap_sec"] = chunk_overlap_sec
            settings["per_chunk_cleanup"] = per_chunk_cleanup
            settings["frame_accurate_split"] = bool(seed_controls.get("frame_accurate_split", True))

            # NOTE: SeedVR2 CLI `--resolution` is now computed per-input from Upscale-x rules.
            # This happens inside _process_single_file() so it also applies to batch items.

            # Apply Output tab format preferences when SeedVR2 tab format is not explicitly set.
            cached_fmt = str(seed_controls.get("output_format_val") or "").strip().lower()
            png_seq_enabled = bool(
                seed_controls.get(
                    "png_sequence_enabled_val",
                    output_settings.get("png_sequence_enabled", False) if isinstance(output_settings, dict) else False,
                )
            )
            if settings.get("output_format") in (None, "auto"):
                if cached_fmt in ("mp4", "png"):
                    settings["output_format"] = cached_fmt
                elif png_seq_enabled:
                    settings["output_format"] = "png"

            if settings["output_format"] == "auto":
                settings["output_format"] = None

            # Global image output preferences (final image files across all model tabs).
            image_fmt = str(
                seed_controls.get(
                    "image_output_format_val",
                    output_settings.get("image_output_format", "png") if isinstance(output_settings, dict) else "png",
                )
                or "png"
            ).strip().lower()
            if image_fmt not in {"png", "jpg", "webp"}:
                image_fmt = "png"
            settings["image_output_format"] = image_fmt
            try:
                settings["image_output_quality"] = int(
                    float(
                        seed_controls.get(
                            "image_output_quality_val",
                            output_settings.get("image_output_quality", 95) if isinstance(output_settings, dict) else 95,
                        )
                        or 95
                    )
                )
            except Exception:
                settings["image_output_quality"] = 95
            settings["image_output_quality"] = max(1, min(100, int(settings["image_output_quality"])))

            # Apply PNG padding/basename settings from Output tab if available
            # These are used for PNG sequence outputs and collision-safe path generation
            if seed_controls.get("png_padding_val") is not None:
                settings["png_padding"] = int(seed_controls["png_padding_val"])
            if seed_controls.get("png_keep_basename_val") is not None:
                settings["png_keep_basename"] = bool(seed_controls["png_keep_basename_val"])

            # Apply audio mux preferences from Output tab (used by chunking + final output postprocessing)
            if seed_controls.get("audio_codec_val") is not None:
                settings["audio_codec"] = seed_controls.get("audio_codec_val") or "copy"
            if seed_controls.get("audio_bitrate_val") is not None:
                settings["audio_bitrate"] = seed_controls.get("audio_bitrate_val") or ""
            
            # Apply metadata and telemetry settings from Output tab
            if seed_controls.get("save_metadata_val") is not None:
                settings["save_metadata"] = bool(seed_controls["save_metadata_val"])
            if seed_controls.get("telemetry_enabled_val") is not None:
                # Note: telemetry is controlled globally via runner.set_telemetry()
                # This flag controls per-run metadata emission only
                settings["telemetry_enabled"] = bool(seed_controls["telemetry_enabled_val"])
            # Apply advanced encoding settings from Output tab so downstream chunk concat and
            # post-processing paths do not fall back to hardcoded ffmpeg defaults.
            encode_overrides = {
                "video_codec": seed_controls.get("video_codec_val"),
                "video_quality": seed_controls.get("video_quality_val"),
                "video_preset": seed_controls.get("video_preset_val"),
                "pixel_format": seed_controls.get("pixel_format_val"),
                "two_pass_encoding": seed_controls.get("two_pass_encoding_val"),
            }
            for key in (
                "video_codec",
                "video_quality",
                "video_preset",
                "pixel_format",
                "two_pass_encoding",
                "metadata_format",
                "log_level",
            ):
                value = encode_overrides.get(key)
                if value is None and isinstance(output_settings, dict):
                    value = output_settings.get(key)
                if value is not None:
                    settings[key] = value

            # SeedVR2 encoding controls moved to Output tab and apply globally.
            backend_pref = seed_controls.get(
                "seedvr2_video_backend_val",
                output_settings.get("seedvr2_video_backend", settings.get("video_backend", "opencv")),
            )
            backend_pref = str(backend_pref or "opencv").strip().lower()
            if backend_pref not in {"opencv", "ffmpeg"}:
                backend_pref = "opencv"
            settings["video_backend"] = backend_pref
            use_10bit_pref = seed_controls.get(
                "seedvr2_use_10bit_val",
                output_settings.get("seedvr2_use_10bit", settings.get("use_10bit", False)),
            )
            settings["use_10bit"] = bool(use_10bit_pref) and backend_pref == "ffmpeg"

            # Apply FPS override from Output tab (0 = keep source FPS).
            if seed_controls.get("fps_override_val") is not None:
                try:
                    settings["fps_override"] = float(seed_controls["fps_override_val"] or 0)
                except Exception:
                    settings["fps_override"] = 0.0
            
            # Apply skip_first_frames and load_cap from Output tab ONLY if not explicitly set in SeedVR2 tab
            # SeedVR2 tab values take precedence over Output tab cached values
            if seed_controls.get("skip_first_frames_val") is not None:
                if settings.get("skip_first_frames", 0) == 0:  # Only if not set in SeedVR2 tab
                    settings["skip_first_frames"] = int(seed_controls["skip_first_frames_val"])
            if seed_controls.get("load_cap_val") is not None:
                if settings.get("load_cap", 0) == 0:  # Only if not set in SeedVR2 tab
                    settings["load_cap"] = int(seed_controls["load_cap_val"])

            # Batch processing
            if settings.get("batch_enable"):
                # Use the batch processor for multiple files
                from shared.batch_processor import BatchProcessor, BatchJob

                batch_input_raw = str(settings.get("batch_input_path") or settings.get("input_path") or "").strip()
                batch_input_path = Path(normalize_path(batch_input_raw)) if batch_input_raw else Path()

                if not batch_input_path.exists():
                    yield (
                        "Batch input path does not exist",  # status_box
                        "",  # log_box
                        "",  # progress_indicator
                        gr.update(value=None, visible=False),  # output_video
                        gr.update(value=None, visible=False),  # output_image
                        "No chunks",  # chunk_info
                        "",  # resume_status
                        "",  # chunk_progress
                        gr.update(value="", visible=False),  # comparison_note
                        gr.update(value=None, visible=False),  # image_slider
                        gr.update(value="", visible=False),  # video_comparison_html
                        gr.update(visible=False),  # chunk_gallery
                        gr.update(visible=False),  # chunk_preview_video
                        gr.update(visible=False),  # batch_gallery
                        state  # shared_state
                    )
                    return

                batch_output_folder = resolve_batch_output_dir(
                    batch_input_path=str(batch_input_path),
                    batch_output_path=settings.get("batch_output_path"),
                    fallback_output_dir=Path(global_settings.get("output_dir", output_dir)),
                    default_subdir_name="upscaled_files",
                )
                settings["batch_output_path"] = str(batch_output_folder)

                # Collect all files to process
                supported_exts = SEEDVR2_VIDEO_EXTS | SEEDVR2_IMAGE_EXTS
                batch_files = []
                if batch_input_path.is_dir():
                    excluded_subtree: Optional[Path] = None
                    try:
                        # If output folder sits inside input folder, don't re-ingest produced files.
                        if _is_path_within(batch_output_folder, batch_input_path) and batch_output_folder.resolve() != batch_input_path.resolve():
                            excluded_subtree = batch_output_folder.resolve()
                    except Exception:
                        excluded_subtree = None

                    for ext in supported_exts:
                        for candidate in batch_input_path.glob(f"**/*{ext}"):
                            try:
                                if not candidate.is_file():
                                    continue
                                if excluded_subtree and _is_path_within(candidate, excluded_subtree):
                                    continue
                                batch_files.append(candidate)
                            except Exception:
                                continue
                elif batch_input_path.suffix.lower() in supported_exts:
                    batch_files = [batch_input_path]

                # Detect if this is an all-image batch or mixed/video batch
                video_count = sum(1 for f in batch_files if Path(f).suffix.lower() in SEEDVR2_VIDEO_EXTS)
                image_count = sum(1 for f in batch_files if Path(f).suffix.lower() in SEEDVR2_IMAGE_EXTS)
                is_image_only_batch = image_count > 0 and video_count == 0

                if not batch_files:
                    yield (
                        "No supported files found in batch input",  # status_box
                        "",  # log_box
                        "",  # progress_indicator
                        gr.update(value=None, visible=False),  # output_video
                        gr.update(value=None, visible=False),  # output_image
                        "No chunks",  # chunk_info
                        "",  # resume_status
                        "",  # chunk_progress
                        gr.update(value="", visible=False),  # comparison_note
                        gr.update(value=None, visible=False),  # image_slider
                        gr.update(value="", visible=False),  # video_comparison_html
                        gr.update(visible=False),  # chunk_gallery
                        gr.update(visible=False),  # chunk_preview_video
                        gr.update(visible=False),  # batch_gallery
                        state  # shared_state
                    )
                    return
                
                # ADDED: Disk space pre-check before batch processing
                from shared.path_utils import get_disk_free_gb
                try:
                    temp_dir_path = Path(global_settings.get("temp_dir", "temp"))
                    output_dir_path = batch_output_folder
                    
                    temp_free_gb = get_disk_free_gb(temp_dir_path)
                    output_free_gb = get_disk_free_gb(output_dir_path)
                    
                    # Estimate required space (rough: 10GB per video, 100MB per image)
                    estimated_gb = (video_count * 10.0) + (image_count * 0.1)
                    
                    warnings = []
                    errors = []
                    
                    # Critical: Less than 5GB free
                    if temp_free_gb < 5.0:
                        errors.append(f"CRITICAL: Only {temp_free_gb:.1f}GB free in temp directory. Need at least 5GB for processing.")
                    elif temp_free_gb < estimated_gb:
                        warnings.append(f"LOW TEMP SPACE: {temp_free_gb:.1f}GB free, estimated need: {estimated_gb:.1f}GB. May fail during processing.")
                    
                    if output_free_gb < 5.0:
                        errors.append(f"CRITICAL: Only {output_free_gb:.1f}GB free in output directory. Need at least 5GB.")
                    elif output_free_gb < estimated_gb * 0.5:
                        warnings.append(f"LOW OUTPUT SPACE: {output_free_gb:.1f}GB free, estimated need: {estimated_gb * 0.5:.1f}GB")
                    
                    if errors:
                        error_msg = "INSUFFICIENT DISK SPACE - Cannot start batch processing:\n" + "\n".join(errors)
                        if warnings:
                            error_msg += "\n\nAdditional warnings:\n" + "\n".join(warnings)
                        
                        yield (
                            error_msg,  # status_box
                            "",  # log_box
                            "",  # progress_indicator
                            gr.update(value=None, visible=False),  # output_video
                            gr.update(value=None, visible=False),  # output_image
                            f"Batch aborted: {len(batch_files)} files (insufficient disk space)",  # chunk_info
                            "",  # resume_status
                            "",  # chunk_progress
                            gr.update(value="", visible=False),  # comparison_note
                            gr.update(value=None, visible=False),  # image_slider
                            gr.update(value="", visible=False),  # video_comparison_html
                            gr.update(visible=False),  # chunk_gallery
                            gr.update(visible=False),  # chunk_preview_video
                            gr.update(visible=False),  # batch_gallery
                            state  # shared_state
                        )
                        return
                    
                    if warnings:
                        warning_msg = "DISK SPACE WARNINGS:\n" + "\n".join(warnings) + "\n\nProceeding with batch processing..."
                        yield (
                            warning_msg,  # status_box
                            f"Starting batch: {len(batch_files)} files\nTemp: {temp_free_gb:.1f}GB free | Output: {output_free_gb:.1f}GB free",  # log_box
                            "",  # progress_indicator
                            gr.update(value=None, visible=False),  # output_video
                            gr.update(value=None, visible=False),  # output_image
                            f"Batch: {len(batch_files)} files queued",  # chunk_info
                            "",  # resume_status
                            "",  # chunk_progress
                            gr.update(value="", visible=False),  # comparison_note
                            gr.update(value=None, visible=False),  # image_slider
                            gr.update(value="", visible=False),  # video_comparison_html
                            gr.update(visible=False),  # chunk_gallery
                            gr.update(visible=False),  # chunk_preview_video
                            gr.update(visible=False),  # batch_gallery
                            state  # shared_state
                        )
                except Exception as e:
                    # Disk check failed - log warning but don't block processing
                    print(f"Warning: Disk space check failed: {e}")
                
                # Continue with batch processing
                if not batch_files:
                    yield (
                        "No supported files found after validation",  # status_box
                        "",  # log_box
                        "",  # progress_indicator
                        gr.update(value=None, visible=False),  # output_video
                        gr.update(value=None, visible=False),  # output_image
                        "No chunks",  # chunk_info
                        "",  # resume_status
                        "",  # chunk_progress
                        gr.update(value="", visible=False),  # comparison_note
                        gr.update(value=None, visible=False),  # image_slider
                        gr.update(value="", visible=False),  # video_comparison_html
                        gr.update(visible=False),  # chunk_gallery
                        gr.update(visible=False),  # chunk_preview_video
                        gr.update(visible=False),  # batch_gallery
                        state  # shared_state
                    )
                    return

                # Create batch processor
                def _batch_progress_cb(bp):
                    try:
                        if progress and bp and bp.total_files:
                            done = int(bp.completed_files + bp.failed_files + bp.skipped_files)
                            total = int(bp.total_files or 1)
                            desc = f"Batch: {done}/{total} files processed"
                            if bp.current_file:
                                desc += f" ({Path(bp.current_file).name})"
                            progress(min(1.0, done / total), desc=desc)
                    except Exception:
                        pass

                batch_processor = BatchProcessor(
                    output_dir=str(batch_output_folder),
                    max_workers=1,  # Sequential processing for memory management
                    # SeedVR2 already writes run_summary.json via run_logger in the single-file pipeline
                    # (and batch writes consolidated metadata at the end). Disable BatchProcessor's
                    # own per-file summaries to avoid duplicates.
                    telemetry_enabled=False,
                    progress_callback=_batch_progress_cb,
                )

                # Create batch jobs
                jobs: List[BatchJob] = []
                for input_file in sorted(set(batch_files)):
                    # For image-only batches, disable per-file telemetry (we will write one consolidated metadata file).
                    job_global_settings = global_settings.copy()
                    if is_image_only_batch:
                        job_global_settings["telemetry"] = False

                    jobs.append(
                        BatchJob(
                            input_path=str(input_file),
                            metadata={
                                "settings": settings.copy(),
                                "global_settings": job_global_settings,
                                "face_apply": face_apply,
                                "face_strength": face_strength,
                                "seed_controls": seed_controls.copy(),
                                "is_image": Path(input_file).suffix.lower() in SEEDVR2_IMAGE_EXTS,
                            },
                        )
                    )

                # Define processing function for each job (reuses the single-file pipeline)
                def process_single_batch_job(job: BatchJob) -> bool:
                    try:
                        # Process single file with current settings
                        single_settings = job.metadata["settings"].copy()
                        single_settings["input_path"] = job.input_path
                        single_settings["batch_enable"] = False  # Disable batch for individual processing
                        single_settings["_original_filename"] = Path(job.input_path).name

                        overwrite_existing = bool(seed_controls.get("overwrite_existing_batch_val", False))

                        input_file = Path(job.input_path)

                        from shared.output_run_manager import batch_item_dir, prepare_batch_video_run_dir
                        from shared.path_utils import sanitize_filename

                        is_video = input_file.suffix.lower() in SEEDVR2_VIDEO_EXTS
                        if not is_video:
                            # Batch images: write directly into the batch output folder using the original name.
                            safe_name = sanitize_filename(input_file.name)
                            target_path = batch_output_folder / safe_name
                            if target_path.exists() and not overwrite_existing:
                                job.status = "skipped"
                                job.output_path = str(target_path)
                                return True
                            if overwrite_existing:
                                try:
                                    target_path.unlink(missing_ok=True)
                                except Exception:
                                    pass
                            single_settings["output_override"] = str(target_path)
                            batch_work_dir = batch_output_folder
                        else:
                            # Batch videos: stable per-input folder under outputs/<input_stem>/ with chunk artifacts inside.
                            item_out_dir = batch_item_dir(batch_output_folder, input_file.name)
                            predicted_output = _predict_batch_video_output_path(
                                item_out_dir=item_out_dir,
                                input_file=input_file,
                                output_format=single_settings.get("output_format"),
                            )
                            if Path(predicted_output).exists() and not overwrite_existing:
                                job.status = "skipped"
                                job.output_path = str(predicted_output)
                                return True
                            run_paths = prepare_batch_video_run_dir(
                                batch_output_folder,
                                input_file.name,
                                input_path=str(input_file),
                                model_label="SeedVR2",
                                mode=str(getattr(runner, "get_mode", lambda: "subprocess")() or "subprocess"),
                                overwrite_existing=overwrite_existing,
                            )
                            if not run_paths:
                                if not overwrite_existing:
                                    job.status = "skipped"
                                    job.output_path = str(predicted_output)
                                    return True
                                job.error_message = f"Could not create batch output folder: {item_out_dir}"
                                return False
                            single_settings["_run_dir"] = str(run_paths.run_dir)
                            single_settings["_processed_chunks_dir"] = str(run_paths.processed_chunks_dir)
                            single_settings["output_override"] = str(run_paths.run_dir)
                            batch_work_dir = run_paths.run_dir

                        # Run single-file processing (no per-file streaming in batch)
                        _status, logs, output_video, output_image, _chunk_info, _chunk_summary, _chunk_progress = _process_single_file(
                            runner,
                            single_settings,
                            job.metadata["global_settings"],
                            job.metadata["seed_controls"],
                            job.metadata["face_apply"],
                            job.metadata["face_strength"],
                            run_logger,
                            Path(batch_work_dir),
                            False,  # not preview
                            None,   # progress_cb
                        )

                        if output_video or output_image:
                            job.output_path = output_video or output_image
                            return True

                        job.error_message = logs
                        return False

                    except Exception as e:
                        job.error_message = str(e)
                        return False

                # Run batch processing
                batch_result = batch_processor.process_batch(
                    jobs=jobs,
                    processor_func=process_single_batch_job,
                    max_concurrent=1,
                )

                # Summarize results and collect output paths for gallery
                completed = int(batch_result.completed_files)
                failed = int(batch_result.failed_files)
                skipped = int(batch_result.skipped_files)
                
                # Collect successful outputs for gallery
                batch_outputs = []
                for job in jobs:
                    if job.status == "completed" and job.output_path and Path(job.output_path).exists():
                        batch_outputs.append(str(job.output_path))

                summary_msg = f"Batch complete: {completed}/{len(jobs)} succeeded"
                if skipped > 0:
                    summary_msg += f", {skipped} skipped"
                if failed > 0:
                    summary_msg += f", {failed} failed"

                # Write consolidated metadata for ALL batches (images AND videos)
                # User requirement: "only have 1 metadata for batch image upscale and have individual metadata for videos upscale"
                # Implementation: 
                # - Image batches: Single consolidated metadata (no per-file metadata to avoid spam)
                # - Video batches: Individual per-video metadata (already written by _process_single_file) + batch summary
                # - Mixed batches: Both image and video summaries
                
                try:
                    metadata_dir = batch_output_folder
                    
                    if is_image_only_batch:
                        # Image-only batch: Single consolidated metadata (no per-file metadata)
                        batch_metadata = {
                            "batch_type": "images",
                            "total_files": len(jobs),
                            "completed": completed,
                            "skipped": skipped,
                            "failed": failed,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "settings": settings,
                            "outputs": batch_outputs,
                            "failed_files": [
                                {"input": job.input_path, "error": job.error_message}
                                for job in jobs if job.status == "failed"
                            ]
                        }
                        metadata_path = metadata_dir / "batch_images_metadata.json"
                    else:
                        # Video batch or mixed batch: Batch summary + individual per-video metadata
                        # Note: Individual video metadata already written by _process_single_file  run_logger.write_summary
                        batch_metadata = {
                            "batch_type": "videos" if video_count > 0 and image_count == 0 else "mixed",
                            "total_files": len(jobs),
                            "video_files": video_count,
                            "image_files": image_count,
                            "completed": completed,
                            "skipped": skipped,
                            "failed": failed,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "settings": settings,
                            "outputs": batch_outputs,
                            "individual_metadata_note": "Each video has its own run_metadata.json in its output directory",
                            "failed_files": [
                                {"input": job.input_path, "error": job.error_message}
                                for job in jobs if job.status == "failed"
                            ]
                        }
                        metadata_path = metadata_dir / "batch_videos_summary.json"
                    
                    import json
                    with metadata_path.open("w", encoding="utf-8") as f:
                        json.dump(batch_metadata, f, indent=2)
                    
                except Exception as e:
                    # Don't fail batch on metadata error
                    pass

                # Update gr.Progress to 100%
                if progress:
                    progress(1.0, desc="Batch complete!")

                # Compact failure summary for the log box
                log_lines = [f"{summary_msg}"]
                if skipped:
                    log_lines.append("Skipped files were left unchanged (enable overwrite to reprocess existing outputs).")
                if failed:
                    for j in [x for x in jobs if x.status == "failed"][:10]:
                        name = Path(j.input_path).name
                        err = (j.error_message or "").strip()
                        err = (err[:180] + "") if len(err) > 180 else err
                        log_lines.append(f"{name}: {err}" if err else f"{name}")

                yield (
                    f"{summary_msg}",  # status_box
                    "\n".join(log_lines),  # log_box
                    "",  # progress_indicator
                    gr.update(value=None, visible=False),  # output_video
                    gr.update(value=None, visible=False),  # output_image
                    f"Batch: {completed} completed, {skipped} skipped, {failed} failed",  # chunk_info
                    "",  # resume_status
                    "",  # chunk_progress
                    gr.update(value="", visible=False),  # comparison_note
                    gr.update(value=None, visible=False),  # image_slider
                    gr.update(value="", visible=False),  # video_comparison_html
                    gr.update(visible=False),  # chunk_gallery
                    gr.update(visible=False),  # chunk_preview_video
                    gr.update(value=batch_outputs[:50], visible=True) if batch_outputs else gr.update(visible=False),  # batch_gallery
                    state  # shared_state
                )
                return

            # Single file processing with streaming updates
            processing_complete = False
            chunk_info = "Waiting for first progress update..."
            prepare_log_msg = "Preparing run..."

            # Start processing with progress tracking
            effective_output_dir = Path(global_settings.get("output_dir", output_dir))

            # Per-run output folder for single video/image runs (0001/0002/...) to avoid collisions
            # and to keep all artifacts in one user-visible location.
            input_kind_single = detect_input_type(settings["input_path"])
            if (not preview_only) and input_kind_single in {"video", "image"}:
                resume_run_dir_raw = str(settings.get("resume_run_dir") or "").strip()
                if resume_run_dir_raw:
                    resume_run_dir = Path(normalize_path(resume_run_dir_raw))
                    if not (resume_run_dir.exists() and resume_run_dir.is_dir()):
                        yield (
                            "Resume folder not found",
                            f"Configured resume folder does not exist: {resume_run_dir}",
                            "",
                            gr.update(value=None, visible=False),
                            gr.update(value=None, visible=False),
                            "Resume folder invalid",
                            "",
                            "",
                            gr.update(value="", visible=False),
                            gr.update(value=None, visible=False),
                            gr.update(value="", visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            state,
                        )
                        return

                    effective_output_dir = resume_run_dir
                    processed_chunks_dir = effective_output_dir / "processed_chunks"
                    processed_chunks_dir.mkdir(parents=True, exist_ok=True)
                    seed_controls["last_run_dir"] = str(effective_output_dir)
                    seed_controls["processed_chunks_dir"] = str(processed_chunks_dir)
                    settings["_run_dir"] = str(effective_output_dir)
                    settings["_processed_chunks_dir"] = str(processed_chunks_dir)
                    settings["_resume_run_requested"] = bool(input_kind_single == "video")
                    settings["_user_output_override_raw"] = settings.get("output_override") or ""
                    # Resume mode ignores fresh output overrides and keeps writing into the selected run folder.
                    settings["output_override"] = str(effective_output_dir)
                    if input_kind_single != "video":
                        settings["resume_run_dir"] = ""
                    prepare_log_msg = (
                        f"Resuming from run folder {effective_output_dir} "
                        "(continuing from last processed chunk; keep same settings)."
                    )
                else:
                    try:
                        run_paths, explicit_final = prepare_single_video_run(
                            output_root_fallback=effective_output_dir,
                            output_override_raw=settings.get("output_override"),
                            input_path=settings["input_path"],
                            original_filename=original_filename,
                            model_label="SeedVR2",
                            mode=str(getattr(runner, "get_mode", lambda: "subprocess")() or "subprocess"),
                        )
                        effective_output_dir = Path(run_paths.run_dir)
                        # Store for UI (chunk gallery, resume helpers, user visibility)
                        seed_controls["last_run_dir"] = str(effective_output_dir)
                        seed_controls["processed_chunks_dir"] = str(run_paths.processed_chunks_dir)
                        settings["_run_dir"] = str(effective_output_dir)
                        settings["_processed_chunks_dir"] = str(run_paths.processed_chunks_dir)

                        # Output override now targets the run folder (or an explicit file inside it)
                        settings["_user_output_override_raw"] = settings.get("output_override") or ""
                        settings["output_override"] = str(explicit_final) if explicit_final else str(effective_output_dir)
                    except Exception:
                        # Fail open: fall back to global output dir if run folder creation fails.
                        seed_controls["last_run_dir"] = str(effective_output_dir)
                        settings["_run_dir"] = str(effective_output_dir)

            yield (
                "Starting processing...",  # status_box
                prepare_log_msg,  # log_box
                _processing_indicator("Upscale in progress..."),  # progress_indicator
                gr.update(value=None, visible=False),  # output_video
                gr.update(value=None, visible=False),  # output_image
                "Waiting for first progress update...",  # chunk_info
                "",  # resume_status
                "",  # chunk_progress
                gr.update(value="", visible=False),  # comparison_note
                gr.update(value=None, visible=False),  # image_slider
                gr.update(value="", visible=False),  # video_comparison_html
                gr.update(visible=False),  # chunk_gallery
                gr.update(visible=False),  # chunk_preview_video
                gr.update(visible=False),  # batch_gallery
                state  # shared_state
            )

            # Create a queue for progress updates
            progress_queue = queue.Queue()

            def processing_thread():
                try:
                    status, logs, output_video, output_image, chunk_info, chunk_summary, chunk_progress = _process_single_file(
                        runner,
                        settings,
                        global_settings,
                        seed_controls,
                        face_apply,
                        face_strength,
                        run_logger,
                        effective_output_dir,
                        preview_only,
                        lambda msg: progress_queue.put(("progress", msg))
                    )
                    progress_queue.put(("complete", (status, logs, output_video, output_image, chunk_info, chunk_summary, chunk_progress)))
                except Exception as e:
                    progress_queue.put(("error", str(e)))

            # Start processing in background thread
            import threading
            proc_thread = threading.Thread(target=processing_thread, daemon=True)
            proc_thread.start()

            # Stream progress updates with gr.Progress integration and throttling
            completed_chunk_count = 0
            active_chunk_idx = 0
            total_chunks_estimate = 1
            last_progress_value = 0.0
            last_ui_update_time = 0
            ui_update_throttle = 0.12
            accumulated_messages = []
            active_stage_name = ""
            active_stage_batch_current = 0
            active_stage_batch_total = 0
            active_stage_percent = 0.0
            active_stage_chunk_idx = 1
            last_gallery_update_time = 0.0
            gallery_update_interval = 1.0
            last_gallery_signature: Tuple[str, ...] = ()
            thumb_cache: Dict[str, str] = {}
            try:
                auto_chunk_enabled_ui = bool(seed_controls.get("auto_chunk", True))
                static_chunk_size_sec_ui = float(seed_controls.get("chunk_size_sec", 0) or 0)
            except Exception:
                auto_chunk_enabled_ui = True
                static_chunk_size_sec_ui = 0.0
            try:
                native_chunk_size_frames = int(settings.get("chunk_size", 0) or 0)
            except Exception:
                native_chunk_size_frames = 0

            if auto_chunk_enabled_ui:
                chunk_mode_label = "Auto Chunk (PySceneDetect scenes)"
            elif static_chunk_size_sec_ui > 0:
                chunk_mode_label = f"Static Chunk ({static_chunk_size_sec_ui:g}s)"
            elif native_chunk_size_frames > 0:
                chunk_mode_label = f"SeedVR2 Native Streaming ({native_chunk_size_frames} frames/chunk)"
            else:
                chunk_mode_label = "Single Pass (no external chunking)"

            ansi_escape_re = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
            chunk_progress_re = re.compile(
                r"(processing|completed)\s+chunk\s+(\d+)\s*/\s*(\d+)",
                re.IGNORECASE,
            )
            stage_batch_re = re.compile(
                r"\b(?P<stage>upscaling|decoding|encoding)\s+batch\s+(?P<current>\d+)\s*/\s*(?P<total>\d+)\b",
                re.IGNORECASE,
            )
            chunk_duration_by_index: Dict[int, float] = {}
            chunk_duration_total = 0.0
            chunk_duration_loaded = False

            def _try_load_chunk_duration_weights() -> None:
                """
                Best-effort chunk weight discovery.

                For video chunking, this uses split chunk durations from `run_dir/input_chunks`.
                For frame-folder chunking, it falls back to frame counts as relative work units.
                """
                nonlocal chunk_duration_by_index, chunk_duration_total, chunk_duration_loaded
                if chunk_duration_loaded:
                    return

                try:
                    run_dir_raw = settings.get("_run_dir")
                    if not run_dir_raw:
                        return
                    input_chunks_dir = Path(str(run_dir_raw)) / "input_chunks"
                    if not input_chunks_dir.exists() or not input_chunks_dir.is_dir():
                        return

                    discovered: Dict[int, float] = {}
                    for cand in sorted(input_chunks_dir.iterdir()):
                        m = re.search(r"chunk_(\d+)", cand.stem.lower())
                        if not m:
                            continue
                        idx = int(m.group(1))
                        unit = 0.0
                        try:
                            if cand.is_file():
                                unit = float(get_media_duration_seconds(str(cand)) or 0.0)
                            elif cand.is_dir():
                                unit = float(
                                    sum(
                                        1
                                        for f in cand.iterdir()
                                        if f.is_file() and f.suffix.lower() in SEEDVR2_IMAGE_EXTS
                                    )
                                )
                        except Exception:
                            unit = 0.0

                        discovered[idx] = unit if unit > 0 else 1.0

                    if discovered:
                        chunk_duration_by_index = discovered
                        chunk_duration_total = float(sum(discovered.values()))
                        if chunk_duration_total > 0:
                            chunk_duration_loaded = True
                except Exception:
                    # Weighted progress is best-effort; fallback logic below remains valid.
                    pass

            def _overall_fraction_for_chunk(chunk_idx: int, in_chunk_fraction: float) -> float:
                in_chunk_fraction = max(0.0, min(1.0, float(in_chunk_fraction)))
                safe_total_chunks = max(1, int(total_chunks_estimate))
                safe_chunk_idx = min(max(1, int(chunk_idx or 1)), safe_total_chunks)

                if safe_total_chunks <= 1:
                    return max(0.0, min(0.999, in_chunk_fraction))

                _try_load_chunk_duration_weights()
                if chunk_duration_loaded and chunk_duration_total > 0:
                    completed_units = 0.0
                    for idx in range(1, safe_chunk_idx):
                        completed_units += float(chunk_duration_by_index.get(idx, 0.0))

                    current_unit = float(chunk_duration_by_index.get(safe_chunk_idx, 0.0))
                    if current_unit <= 0:
                        remaining_units = max(0.0, float(chunk_duration_total) - completed_units)
                        remaining_chunks = max(1, safe_total_chunks - safe_chunk_idx + 1)
                        current_unit = (
                            (remaining_units / remaining_chunks)
                            if remaining_units > 0
                            else (float(chunk_duration_total) / safe_total_chunks)
                        )

                    weighted = (completed_units + current_unit * in_chunk_fraction) / max(1e-9, float(chunk_duration_total))
                    return max(0.0, min(0.999, weighted))

                # Fallback: equal-weight chunks.
                uniform = (max(0, safe_chunk_idx - 1) + in_chunk_fraction) / safe_total_chunks
                return max(0.0, min(0.999, uniform))

            def _normalize_stage(stage_raw: str) -> Tuple[str, str]:
                stage_key = str(stage_raw or "").strip().lower()
                if stage_key.startswith("decod"):
                    return "decoding", "VAE Decoding"
                if stage_key.startswith("encod"):
                    return "encoding", "VAE Encoding"
                if stage_key.startswith("upscal"):
                    return "upscaling", "Upscaling"
                return stage_key, (str(stage_raw).title() if stage_raw else "Processing")

            def _chunk_fraction_from_stage(stage_key: str, batch_fraction: float) -> float:
                batch_fraction = max(0.0, min(1.0, float(batch_fraction)))
                if stage_key == "encoding":
                    return 0.25 * batch_fraction
                if stage_key == "upscaling":
                    return 0.25 + 0.5 * batch_fraction
                if stage_key == "decoding":
                    return 0.75 + 0.25 * batch_fraction
                return batch_fraction

            try:
                from shared.frame_utils import extract_video_thumbnail
            except Exception:
                extract_video_thumbnail = None

            while proc_thread.is_alive() or not progress_queue.empty():
                try:
                    update_type, data = progress_queue.get(timeout=0.1)
                    current_time = time.time()                    if update_type == "progress":
                        raw_message = str(data or "").strip()
                        if not raw_message:
                            continue

                        message = ansi_escape_re.sub("", raw_message).strip()
                        if not message:
                            continue

                        accumulated_messages.append(message)
                        if len(accumulated_messages) > 200:
                            accumulated_messages = accumulated_messages[-200:]

                        message_lc = message.lower()
                        if maybe_set_vram_oom_alert(state, model_label="SeedVR2", text=message, settings=settings):
                            state["operation_status"] = "error"
                            show_vram_oom_modal(state, title="Out of VRAM (GPU) - SeedVR2", duration=None)

                        oom_visible = bool(state.get("alerts", {}).get("oom", {}).get("visible"))
                        chunk_match = chunk_progress_re.search(message)
                        stage_match = stage_batch_re.search(message)
                        pct_match = re.search(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*%", message)
                        pct_value = float(pct_match.group(1)) if pct_match else None
                        if chunk_match:
                            phase = str(chunk_match.group(1) or "").lower()
                            chunk_idx = int(chunk_match.group(2))
                            prev_total_chunks_estimate = total_chunks_estimate
                            total_chunks_estimate = max(1, int(chunk_match.group(3)))
                            if (
                                total_chunks_estimate > prev_total_chunks_estimate
                                and completed_chunk_count < total_chunks_estimate
                            ):
                                # If we previously assumed a single chunk, stale near-100% progress
                                # must be capped once we discover multiple chunks.
                                cap_chunk_idx = max(1, chunk_idx)
                                conservative_cap = _overall_fraction_for_chunk(cap_chunk_idx, 0.999)
                                last_progress_value = min(last_progress_value, conservative_cap)
                            active_chunk_idx = max(1, chunk_idx)
                            if phase == "processing" and active_stage_chunk_idx != active_chunk_idx:
                                active_stage_name = ""
                                active_stage_batch_current = 0
                                active_stage_batch_total = 0
                                active_stage_percent = 0.0
                            if phase == "completed":
                                completed_chunk_count = max(completed_chunk_count, chunk_idx)
                                if active_stage_chunk_idx <= chunk_idx:
                                    active_stage_name = ""
                                    active_stage_batch_current = 0
                                    active_stage_batch_total = 0
                                    active_stage_percent = 0.0
                            if pct_value is not None:
                                pct_fraction = max(0.0, min(1.0, pct_value / 100.0))
                                if total_chunks_estimate > 1:
                                    # Log percentages on "processing chunk X/Y (...%)" are chunk-local,
                                    # so convert them into overall progress.
                                    if phase == "completed":
                                        overall_fraction = _overall_fraction_for_chunk(chunk_idx, 1.0)
                                    else:
                                        overall_fraction = _overall_fraction_for_chunk(chunk_idx, pct_fraction)
                                    last_progress_value = max(last_progress_value, min(0.999, overall_fraction))
                                else:
                                    last_progress_value = max(last_progress_value, min(0.999, pct_fraction))
                            elif phase == "completed":
                                last_progress_value = max(last_progress_value, _overall_fraction_for_chunk(chunk_idx, 1.0))
                            else:
                                last_progress_value = max(last_progress_value, _overall_fraction_for_chunk(chunk_idx, 0.0))
                        if stage_match:
                            stage_key, stage_label = _normalize_stage(stage_match.group("stage"))
                            batch_current = int(stage_match.group("current"))
                            batch_total = max(1, int(stage_match.group("total")))
                            batch_fraction = max(0.0, min(1.0, float(batch_current) / float(batch_total)))

                            if active_chunk_idx <= 0:
                                active_chunk_idx = max(1, completed_chunk_count + 1)
                            if total_chunks_estimate < active_chunk_idx:
                                total_chunks_estimate = active_chunk_idx

                            active_stage_name = stage_label
                            active_stage_batch_current = batch_current
                            active_stage_batch_total = batch_total
                            active_stage_percent = batch_fraction
                            active_stage_chunk_idx = active_chunk_idx

                            chunk_fraction = _chunk_fraction_from_stage(stage_key, batch_fraction)
                            if total_chunks_estimate > 1:
                                overall_fraction = _overall_fraction_for_chunk(active_chunk_idx, chunk_fraction)
                            else:
                                overall_fraction = chunk_fraction
                            last_progress_value = max(last_progress_value, min(0.999, overall_fraction))
                        elif not chunk_match:
                            # Parse only explicit progress-style ratios to avoid false
                            # positives from file paths/CLI args (e.g. ".../0002/16.mp4").
                            ratio_ctx = re.search(
                                r"\b(?:(chunk|frame|step|batch)\s*(\d+)\s*/\s*(\d+)|(\d+)\s*/\s*(\d+)\s*(chunks?|frames?|steps?|batches?))\b",
                                message_lc,
                            )
                            if ratio_ctx:
                                if ratio_ctx.group(1):
                                    kind = str(ratio_ctx.group(1) or "").strip().lower()
                                    current = int(ratio_ctx.group(2))
                                    total = max(1, int(ratio_ctx.group(3)))
                                else:
                                    unit = str(ratio_ctx.group(6) or "").strip().lower()
                                    kind = unit[:-1] if unit.endswith("s") else unit
                                    current = int(ratio_ctx.group(4))
                                    total = max(1, int(ratio_ctx.group(5)))

                                is_chunk_ratio = kind == "chunk"
                                if is_chunk_ratio:
                                    prev_total_chunks_estimate = total_chunks_estimate
                                    total_chunks_estimate = max(total_chunks_estimate, total)
                                    if (
                                        total_chunks_estimate > prev_total_chunks_estimate
                                        and completed_chunk_count < total_chunks_estimate
                                    ):
                                        cap_chunk_idx = active_chunk_idx if active_chunk_idx > 0 else max(
                                            1, completed_chunk_count + 1
                                        )
                                        conservative_cap = _overall_fraction_for_chunk(cap_chunk_idx, 0.999)
                                        last_progress_value = min(last_progress_value, conservative_cap)
                                if total_chunks_estimate > 1 and is_chunk_ratio:
                                    if current <= 0:
                                        overall_fraction = 0.0
                                    else:
                                        overall_fraction = _overall_fraction_for_chunk(current, 1.0)
                                    last_progress_value = max(last_progress_value, min(0.999, overall_fraction))
                                elif total_chunks_estimate <= 1:
                                    last_progress_value = max(last_progress_value, min(0.999, current / total))
                            elif pct_value is not None and any(
                                tok in message_lc for tok in ("progress", "processing chunk", "completed chunk", "done")
                            ):
                                pct_fraction = max(0.0, min(1.0, pct_value / 100.0))
                                if total_chunks_estimate > 1:
                                    inferred_active_chunk = active_chunk_idx if active_chunk_idx > 0 else max(
                                        1, completed_chunk_count + 1
                                    )
                                    overall_fraction = _overall_fraction_for_chunk(inferred_active_chunk, pct_fraction)
                                    last_progress_value = max(last_progress_value, min(0.999, overall_fraction))
                                else:
                                    last_progress_value = max(last_progress_value, min(0.999, pct_fraction))

                        # Multi-chunk overall progress is defined strictly by completed chunks.
                        if total_chunks_estimate > 1:
                            display_progress_value = max(
                                0.0,
                                min(1.0, completed_chunk_count / max(1, total_chunks_estimate)),
                            )
                        else:
                            display_progress_value = max(0.0, min(0.999, last_progress_value))

                        if progress:
                            progress(display_progress_value, desc=message[:100] if message else "Processing...")

                        stage_summary_short = ""
                        stage_summary_line = ""
                        if active_stage_name and active_stage_batch_total > 0:
                            stage_pct = int(round(max(0.0, min(1.0, active_stage_percent)) * 100.0))
                            stage_chunk = active_stage_chunk_idx if active_stage_chunk_idx > 0 else max(1, active_chunk_idx or 1)
                            stage_summary_short = f"Chunk {stage_chunk} {active_stage_name} {stage_pct}%"
                            stage_summary_line = (
                                f"{stage_summary_short} "
                                f"(batch {active_stage_batch_current}/{active_stage_batch_total})"
                            )

                        is_key_event = (
                            "processing chunk" in message_lc
                            or "completed chunk" in message_lc
                            or "error" in message_lc
                            or "failed" in message_lc
                            or bool(stage_match)
                        )
                        if (current_time - last_ui_update_time) < ui_update_throttle and not is_key_event:
                            continue

                        chunk_gallery_update = gr.update()
                        chunk_preview_update = gr.update()
                        current_chunk_items = (state or {}).get("seed_controls", {}).get("chunk_gallery_items", [])
                        if not current_chunk_items:
                            current_chunk_items = (state or {}).get("seed_controls", {}).get("chunk_thumbnails", [])
                        current_signature = tuple(
                            str(item)
                            for item in current_chunk_items
                            if isinstance(item, str)
                        )
                        should_refresh_gallery = (
                            is_key_event
                            or current_signature != last_gallery_signature
                            or (current_time - last_gallery_update_time) >= gallery_update_interval
                        )
                        if should_refresh_gallery:
                            gallery_display_items = []
                            chunk_video_paths = []
                            if current_chunk_items:
                                for idx, item in enumerate(current_chunk_items):
                                    if not (isinstance(item, str) and Path(item).exists()):
                                        continue
                                    item_path = Path(item)
                                    if item_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
                                        thumb_path = thumb_cache.get(item)
                                        if thumb_path and Path(thumb_path).exists():
                                            gallery_display_items.append((thumb_path, f"Chunk {idx+1}"))
                                            chunk_video_paths.append(item)
                                        elif extract_video_thumbnail:
                                            ok, generated_thumb, _ = extract_video_thumbnail(item, width=280)
                                            if ok and generated_thumb:
                                                thumb_cache[item] = generated_thumb
                                                gallery_display_items.append((generated_thumb, f"Chunk {idx+1}"))
                                                chunk_video_paths.append(item)
                                        # Never pass live chunk video paths to Gallery during processing.
                                        # Some UI stacks may transcode/normalize video for preview and mutate files.
                                    else:
                                        gallery_display_items.append((item, f"Chunk {idx+1}"))
                                        chunk_video_paths.append(None)

                            if chunk_video_paths:
                                state["seed_controls"]["chunk_video_paths"] = chunk_video_paths

                            chunk_gallery_update = (
                                gr.update(
                                    value=gallery_display_items,
                                    visible=len(gallery_display_items) > 0,
                                    columns=4,
                                    rows=2,
                                    height=200,
                                )
                                if gallery_display_items
                                else gr.update(visible=False)
                            )

                            # Keep chunk preview video hidden during processing to avoid any
                            # Gradio-side preview re-encode touching chunk outputs.
                            chunk_preview_update = gr.update(visible=False)
                            last_gallery_signature = current_signature
                            last_gallery_update_time = current_time

                        if total_chunks_estimate > 1:
                            overall_fraction_for_text = max(
                                0.0,
                                min(1.0, completed_chunk_count / max(1, total_chunks_estimate)),
                            )
                        else:
                            overall_fraction_for_text = max(0.0, min(1.0, last_progress_value))

                        percent_done = int(overall_fraction_for_text * 100.0)
                        completed_shown = min(max(0, completed_chunk_count), total_chunks_estimate)
                        current_shown = active_chunk_idx if active_chunk_idx > 0 else max(1, completed_shown)
                        if total_chunks_estimate > 1:
                            status_text = (
                                f"Processing... {completed_shown}/{total_chunks_estimate} chunks completed "
                                f"| overall {percent_done}%"
                            )
                            if stage_summary_short:
                                status_text = f"{status_text} | {stage_summary_short}"
                                indicator_title = (
                                    f"Processing... ({completed_shown}/{total_chunks_estimate} chunks completed, "
                                    f"overall {percent_done}% done, {stage_summary_short})"
                                )
                            else:
                                indicator_title = (
                                    f"Processing... ({completed_shown}/{total_chunks_estimate} chunks completed, "
                                    f"overall {percent_done}% done)"
                                )
                        else:
                            if stage_summary_short:
                                status_text = f"Processing... {stage_summary_short} (overall {percent_done}%)"
                                indicator_title = f"Processing... ({stage_summary_short}, overall {percent_done}% done)"
                            else:
                                status_text = f"Processing... {percent_done}%"
                                indicator_title = f"Processing... ({percent_done}% done)"

                        if oom_visible:
                            status_text = "Out of VRAM (GPU) - see banner above"
                            indicator_title = "Out of VRAM (GPU) - see banner above"

                        if total_chunks_estimate > 1:
                            chunk_status_lines = [
                                f"Chunk mode: {chunk_mode_label}",
                                f"Chunks completed: {completed_shown}/{total_chunks_estimate}",
                                f"Current chunk: {current_shown}/{total_chunks_estimate}",
                            ]
                        else:
                            chunk_status_lines = [
                                f"Chunk mode: {chunk_mode_label}",
                                f"Chunk {current_shown}/{max(1, total_chunks_estimate)}",
                            ]
                        if stage_summary_line:
                            chunk_status_lines.append(stage_summary_line)
                        chunk_status_lines.append(f"Latest: {message}")
                        chunk_status_text = "\n".join(chunk_status_lines)

                        chunk_progress_lines: List[str] = []
                        if stage_summary_line:
                            chunk_progress_lines.append(stage_summary_line)
                        tail_count = 11 if stage_summary_line else 12
                        chunk_progress_lines.extend(accumulated_messages[-tail_count:])
                        chunk_progress_text = "\n".join(chunk_progress_lines)

                        yield (
                            status_text,
                            "\n".join(accumulated_messages[-30:]),
                            _processing_indicator(indicator_title),
                            gr.update(value=None, visible=False),
                            gr.update(value=None, visible=False),
                            chunk_status_text,
                            "",
                            chunk_progress_text,
                            gr.update(value="", visible=False),
                            gr.update(value=None, visible=False),
                            gr.update(value="", visible=False),
                            chunk_gallery_update,
                            chunk_preview_update,  # chunk preview video
                            gr.update(visible=False),
                            state,
                        )
                        last_ui_update_time = current_time
                    elif update_type == "complete":
                        status, logs, output_video, output_image, chunk_info, chunk_summary, chunk_progress = data
                        processing_complete = True
                        if progress:
                            progress(1.0, desc="Complete!")
                        break                    elif update_type == "error":
                        if progress:
                            progress(0, desc="Error occurred")
                        if maybe_set_vram_oom_alert(state, model_label="SeedVR2", text=data, settings=settings):
                            state["operation_status"] = "error"
                            show_vram_oom_modal(state, title="Out of VRAM (GPU) - SeedVR2", duration=None)

                        yield (
                            ("Out of VRAM (GPU) - see banner above" if state.get("alerts", {}).get("oom", {}).get("visible") else "Processing failed"),
                            f"Error: {data}",
                            indicator_off,
                            gr.update(value=None, visible=False),
                            gr.update(value=None, visible=False),
                            "Error occurred",
                            "",
                            "",
                            gr.update(value="", visible=False),
                            gr.update(value=None, visible=False),
                            gr.update(value="", visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            state,
                        )
                        return
                except queue.Empty:
                    continue            if not processing_complete:
                yield (
                    "Processing timed out",  # status_box
                    "Processing did not complete within expected time",  # log_box
                    indicator_off,  # progress_indicator
                    gr.update(value=None, visible=False),  # output_video
                    gr.update(value=None, visible=False),  # output_image
                    "Timeout",  # chunk_info
                    "",  # resume_status
                    "",  # chunk_progress
                    gr.update(value="", visible=False),  # comparison_note
                    gr.update(value=None, visible=False),  # image_slider
                    gr.update(value="", visible=False),  # video_comparison_html
                    gr.update(visible=False),  # chunk_gallery
                    gr.update(visible=False),  # chunk_preview_video
                    gr.update(visible=False),  # batch_gallery
                    state  # shared_state
                )
                return

            # Final pass: if logs contain VRAM OOM signatures, raise the global banner.            if maybe_set_vram_oom_alert(state, model_label="SeedVR2", text=logs, settings=settings):
                state["operation_status"] = "error"
                status = "Out of VRAM (GPU) - see banner above"
                show_vram_oom_modal(state, title="Out of VRAM (GPU) - SeedVR2", duration=None)

            # Create comparison based on mode from Output tab
            comparison_mode = seed_controls.get("comparison_mode_val", "native")
            original_path_for_compare = settings.get("_original_input_path_before_preprocess") or settings.get("input_path")
            comparison_html = ""
            
            if comparison_mode == "native":
                # Use gradio's native ImageSlider for images
                if output_image and Path(output_image).exists():
                    comparison_html = ""
                    # Check for pinned reference
                    pinned_ref = seed_controls.get("pinned_reference_path")
                    pin_enabled = seed_controls.get("pin_reference_val", False)
                    
                    image_slider_update = gr.update(
                        value=(pinned_ref if (pin_enabled and pinned_ref) else original_path_for_compare, output_image),
                        visible=True
                    )
                else:
                    # Check for pinned reference
                    pinned_ref = seed_controls.get("pinned_reference_path")
                    pin_enabled = seed_controls.get("pin_reference_val", False)
                    
                    comparison_html, image_slider_update = create_comparison_selector(
                        input_path=original_path_for_compare,
                        output_path=output_video or output_image,
                        comparison_mode="slider",
                        pinned_reference_path=pinned_ref,
                        pin_enabled=pin_enabled
                    )
            else:
                # Use custom HTML comparisons for other modes
                pinned_ref = seed_controls.get("pinned_reference_path")
                pin_enabled = seed_controls.get("pin_reference_val", False)
                
                comparison_html, image_slider_update = create_comparison_selector(
                    input_path=original_path_for_compare,
                    output_path=output_video or output_image,
                    comparison_mode=comparison_mode,
                    pinned_reference_path=pinned_ref,
                    pin_enabled=pin_enabled
                )

            # Build video comparison HTML for videos
            video_comparison_html_update = gr.update(value="", visible=False)
            try:
                if output_video and Path(output_video).exists():
                    original_path = original_path_for_compare or ""
                    if original_path and Path(original_path).exists():
                        # Use new video comparison slider
                        from shared.video_comparison_slider import create_video_comparison_html as create_vid_comp
                        try:
                            same_comp_paths = Path(original_path).resolve() == Path(output_video).resolve()
                        except Exception:
                            same_comp_paths = False

                        video_comp_html = create_vid_comp(
                            original_video=original_path,
                            upscaled_video=output_video,
                            height=600,
                            slider_position=50.0
                        )
                        video_comparison_html_update = gr.update(value=video_comp_html, visible=True)
                        print(
                            f"[DEBUG] Video comparison slider created "
                            f"(original: {original_path}, upscaled: {output_video}, same_path={same_comp_paths})",
                            flush=True,
                        )
                    else:
                        print(f"[DEBUG] Video comparison skipped - original path missing: {original_path}", flush=True)
                    # For videos, hide the image slider
                    image_slider_update = gr.update(value=None, visible=False)
                else:
                    print(f"[DEBUG] Video comparison skipped - output_video: {output_video}", flush=True)
            except Exception as e:
                import traceback
                print(f"[ERROR] Video comparison HTML error: {str(e)}", flush=True)
                print(f"[ERROR] Traceback: {traceback.format_exc()}", flush=True)
            
            # If no HTML comparison, use ImageSlider for images
            if not comparison_html and output_image and not output_video:
                image_slider_update = gr.update(
                    value=(original_path_for_compare, output_image),
                    visible=True
                )
            elif not image_slider_update:
                image_slider_update = gr.update(value=None, visible=False)

            state["operation_status"] = "completed" if "complete" in str(status or "").lower() else "ready"
            
            # Prepare final chunk gallery display with thumbnail images
            # gr.Gallery in Gradio 6.x doesn't display videos well, so we convert to thumbnails
            final_chunk_items = (state or {}).get("seed_controls", {}).get("chunk_gallery_items", [])
            if not final_chunk_items:
                final_chunk_items = (state or {}).get("seed_controls", {}).get("chunk_thumbnails", [])

            # Convert video paths to thumbnail images for gallery
            final_gallery_display = []
            if final_chunk_items:
                from shared.frame_utils import extract_video_thumbnail
                for idx, item in enumerate(final_chunk_items):
                    if isinstance(item, str) and Path(item).exists():
                        if Path(item).suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}:
                            # Generate thumbnail for video
                            success, thumb_path, _ = extract_video_thumbnail(item, width=320)
                            if success and thumb_path:
                                final_gallery_display.append((thumb_path, f"Chunk {idx+1}"))
                        else:
                            final_gallery_display.append((item, f"Chunk {idx+1}"))

            final_chunk_gallery = gr.update(
                value=final_gallery_display,
                visible=len(final_gallery_display) > 0,
                columns=4,
                rows=2,
                height=320
            ) if final_gallery_display else gr.update(visible=False)
            
            # Debug final state before yielding
            print(f"[DEBUG] Final output_video: {output_video}", flush=True)
            print(f"[DEBUG] Final output_video exists: {Path(output_video).exists() if output_video else False}", flush=True)

            vid_upd, img_upd = _media_updates(output_video, output_image)
            print(f"[DEBUG] vid_upd: {vid_upd}", flush=True)

            yield (
                status,  # status_box
                logs,  # log_box
                indicator_off,  # progress_indicator
                vid_upd,  # output_video
                img_upd,  # output_image
                chunk_info,  # chunk_info
                "",  # resume_status
                chunk_progress,  # chunk_progress - NOW POPULATED with actual chunk progress
                gr.update(value="", visible=False),  # comparison_note
                image_slider_update,  # image_slider
                video_comparison_html_update,  # video_comparison_html
                final_chunk_gallery,  # chunk_gallery - SHOW completed chunk thumbnails!
                gr.update(visible=False),  # chunk_preview_video - preview shown via gallery select
                gr.update(visible=False),  # batch_gallery - Hide for single file
                state  # shared_state
            )

        except Exception as e:
            error_msg = f"Critical error in SeedVR2 processing: {str(e)}"
            state["operation_status"] = "error"
            # If this was VRAM OOM, show the big banner.
            if maybe_set_vram_oom_alert(state, model_label="SeedVR2", text=str(e), settings=locals().get("settings")):
                show_vram_oom_modal(state, title="Out of VRAM (GPU) - SeedVR2", duration=None)
            yield (
                "Critical error",  # status_box
                error_msg,  # log_box
                indicator_off,  # progress_indicator
                gr.update(value=None, visible=False),  # output_video
                gr.update(value=None, visible=False),  # output_image
                "Error",  # chunk_info
                "",  # resume_status
                "",  # chunk_progress
                gr.update(value="", visible=False),  # comparison_note
                gr.update(value=None, visible=False),  # image_slider
                gr.update(value="", visible=False),  # video_comparison_html
                gr.update(visible=False),  # chunk_gallery
                gr.update(visible=False),  # chunk_preview_video
                gr.update(visible=False),  # batch_gallery
                state  # shared_state
            )

    return {
        "defaults": defaults,
        "order": SEEDVR2_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "check_resume_status": check_resume_status,
        "run_action": run_action,
        "cancel_action": cancel,
        "open_outputs_folder": open_outputs_folder_seedvr2,
        "clear_temp_folder": clear_temp_folder_seedvr2,
        "comparison_html_slider": comparison_html_slider,
        "auto_res_on_input": _auto_res_from_input,
        "get_model_loading_status": get_model_loading_status,
    }
