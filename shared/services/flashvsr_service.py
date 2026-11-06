"""
FlashVSR Service Module
Handles FlashVSR processing logic, presets, and callbacks
"""

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr

from shared.preset_manager import PresetManager
from shared.logging_utils import RunLogger
from shared.models.flashvsr_meta import (
    get_flashvsr_metadata,
    get_flashvsr_default_model,
    flashvsr_version_to_internal,
    flashvsr_version_to_ui,
)
from shared.vram_autotune.common import DEFAULT_SAVE_VRAM_GB, clamp_save_vram_gb
from shared.services.video_sr_service import (  # noqa: F401 - re-exported helpers
    _save_preprocessed_artifact,
    _apply_image_output_preferences,
    _move_or_convert_image_output,
    _safe_ui_video_preview_path,
    _to_bool,
    _to_int,
    _to_float,
    VideoSREngine,
    build_video_sr_callbacks,
)

# Cancel event for FlashVSR processing (the shared flow reads it from FLASHVSR_ENGINE)
_flashvsr_cancel_event = threading.Event()

FLASHVSR_VAE_OPTIONS = ["Wan2.1", "LightVAE_W2.1"]
FLASHVSR_PRECISION_OPTIONS = ["auto", "bf16", "fp16", "int8_convrot"]
FLASHVSR_ATTENTION_OPTIONS = ["sparse_sage_attention", "block_sparse_attention", "flash_attention_2", "sdpa"]
FLASHVSR_CODEC_OPTIONS = ["libx264", "libx265", "h264_nvenc"]


def run_flashvsr(*args, **kwargs):
    """Lazy-import FlashVSR runner so the Gradio process stays backend-light."""
    from shared.flashvsr_runner import run_flashvsr as _run_flashvsr_impl

    return _run_flashvsr_impl(*args, **kwargs)


def _nearest_supported_scale(value: Any, default: int = 4) -> int:
    raw = _to_float(value, float(default))
    return 2 if raw <= 3.0 else 4


def canonical_flashvsr_scale(
    *,
    scale_value: Any = None,
    upscale_factor_value: Any = None,
    default: Any = 4,
) -> int:
    """
    Resolve FlashVSR scale to backend-supported values (2x or 4x).

    `upscale_factor_value` takes precedence over `scale_value` so UI slider
    updates cannot be overridden by stale hidden/state fields.
    """
    default_scale = _nearest_supported_scale(default, 4)
    if upscale_factor_value is not None:
        return _nearest_supported_scale(upscale_factor_value, default_scale)
    return _nearest_supported_scale(scale_value, default_scale)


def flashvsr_defaults(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Defaults aligned to ComfyUI-FlashVSR_Stable with quality-first tuning.
    """
    try:
        from shared.gpu_utils import get_gpu_info

        cuda_default = "auto" if get_gpu_info() else "cpu"
    except Exception:
        cuda_default = "cpu"

    default_model = model_name or get_flashvsr_default_model()
    model_meta = get_flashvsr_metadata(default_model)

    if model_meta:
        default_precision = "bf16"
        default_tile_size = model_meta.default_tile_size
        default_overlap = model_meta.default_overlap
        default_attention = "flash_attention_2"
        default_vae = "LightVAE_W2.1"  # measured best decoder (fidelity, flicker, speed) on two clips
        mode = model_meta.mode
        scale = int(model_meta.scale)
        frame_chunk_size = 241
        keep_models_on_cpu = bool(model_meta.default_keep_models_on_cpu)
        version = flashvsr_version_to_ui(model_meta.version)
        tiled_vae = False
        tiled_dit = True
    else:
        default_precision = "bf16"
        default_tile_size = 256
        default_overlap = 24
        default_attention = "flash_attention_2"
        default_vae = "LightVAE_W2.1"  # measured best decoder (fidelity, flicker, speed) on two clips
        version = "1.1"
        mode = "full"
        scale = 4
        frame_chunk_size = 241
        keep_models_on_cpu = True
        tiled_vae = False
        tiled_dit = True

    # User decision (V9.5): INT8 ConvRot is the FlashVSR default on every platform (small cache, same quality,
    # about 1 GB less VRAM and slightly faster than BF16).
    default_precision = "int8_convrot"

    return {
        "input_path": "",
        "output_override": "",
        "output_format": "mp4",
        "scale": scale,
        "version": version,
        "mode": mode,
        "vae_model": default_vae,
        "precision": default_precision,
        "attention_mode": default_attention,
        "tiled_vae": tiled_vae,
        "tiled_dit": tiled_dit,
        "tile_size": default_tile_size,
        "overlap": default_overlap,
        "unload_dit": True,
        "stream_decode": False,
        "cfg_scale": 1.0,
        "denoise_amount": 1.0,
        "sparse_ratio": 2.0,
        "kv_ratio": 3.0,
        "local_range": 11,
        "frame_chunk_size": frame_chunk_size,
        "keep_models_on_cpu": keep_models_on_cpu,
        "force_offload": True,
        "enable_debug": False,
        "color_fix": True,
        "seed": 0,
        "auto_transfer_output_to_input": False,
        "device": cuda_default,
        "fps": 0.0,
        "codec": "libx264",
        "crf": 18,
        "start_frame": 0,
        "end_frame": -1,
        "models_dir": str((Path(__file__).resolve().parents[2] / "ComfyUI-FlashVSR_Stable" / "models")),
        "save_metadata": True,
        "face_restore_after_upscale": False,
        "batch_enable": False,
        "keep_only_output_files": False,
        "batch_input_path": "",
        "batch_output_path": "",
        "resume_run_dir": "",
        "use_resolution_tab": True,
        "upscale_factor": float(scale),
        "max_target_resolution": 1920,
        "pre_downscale_then_upscale": True,
        "save_vram_gb": DEFAULT_SAVE_VRAM_GB,
        "auto_tune_on_run": True,
        # FlashVSR+ (official Turbo / Base) only
        "temporal_compression": 4,
        "decode_every": 16,
    }


FLASHVSR_ORDER: List[str] = [
    "input_path",
    "output_override",
    "output_format",
    "scale",
    "version",
    "mode",
    "vae_model",
    "precision",
    "attention_mode",
    "tiled_vae",
    "tiled_dit",
    "tile_size",
    "overlap",
    "unload_dit",
    "stream_decode",
    "cfg_scale",
    "denoise_amount",
    "sparse_ratio",
    "kv_ratio",
    "local_range",
    "frame_chunk_size",
    "keep_models_on_cpu",
    "force_offload",
    "enable_debug",
    "color_fix",
    "seed",
    "auto_transfer_output_to_input",
    "device",
    "fps",
    "codec",
    "crf",
    "start_frame",
    "end_frame",
    "models_dir",
    "save_metadata",
    "face_restore_after_upscale",
    "batch_enable",
    "batch_input_path",
    "batch_output_path",
    "use_resolution_tab",
    "upscale_factor",
    "max_target_resolution",
    "pre_downscale_then_upscale",
    "resume_run_dir",
    "save_vram_gb",
    "keep_only_output_files",
    "auto_tune_on_run",
    "temporal_compression",
    "decode_every",
]


def _flashvsr_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    return dict(zip(FLASHVSR_ORDER, args))


def _enforce_flashvsr_guardrails(cfg: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {**defaults, **(cfg or {})}

    cfg["output_format"] = "mp4"
    cfg["save_metadata"] = _to_bool(cfg.get("save_metadata"), _to_bool(defaults.get("save_metadata", True), True))
    cfg["face_restore_after_upscale"] = _to_bool(
        cfg.get("face_restore_after_upscale"),
        _to_bool(defaults.get("face_restore_after_upscale", False), False),
    )

    cfg["version"] = flashvsr_version_to_ui(cfg.get("version", defaults.get("version", "1.1")))
    mode = str(cfg.get("mode", defaults.get("mode", "full")) or "full").strip().lower()
    cfg["mode"] = mode if mode in {"tiny", "tiny-long", "full"} else "full"
    scale = canonical_flashvsr_scale(
        scale_value=cfg.get("scale", defaults.get("scale", 4)),
        upscale_factor_value=cfg.get("upscale_factor"),
        default=defaults.get("upscale_factor", defaults.get("scale", 4)),
    )
    cfg["scale"] = str(scale)
    cfg["upscale_factor"] = float(scale)

    precision = str(cfg.get("precision", cfg.get("dtype", defaults.get("precision", "auto")))).strip().lower()
    cfg["precision"] = precision if precision in {"auto", "bf16", "fp16", "int8_convrot"} else "auto"

    att_raw = str(cfg.get("attention_mode", cfg.get("attention", defaults.get("attention_mode", "flash_attention_2")))).strip().lower()
    att_map = {
        "sage": "sparse_sage_attention",
        "sparse_sage": "sparse_sage_attention",
        "sparse_sage_attention": "sparse_sage_attention",
        "block": "block_sparse_attention",
        "block_sparse": "block_sparse_attention",
        "block_sparse_attention": "block_sparse_attention",
        "flash_attn_2": "flash_attention_2",
        "flash_attention_2": "flash_attention_2",
        "sdpa": "sdpa",
    }
    cfg["attention_mode"] = att_map.get(att_raw, "flash_attention_2")

    vae_model = str(cfg.get("vae_model", defaults.get("vae_model", "LightVAE_W2.1"))).strip()
    cfg["vae_model"] = vae_model if vae_model in FLASHVSR_VAE_OPTIONS else "LightVAE_W2.1"

    cfg["tiled_vae"] = _to_bool(cfg.get("tiled_vae"), _to_bool(defaults.get("tiled_vae", True), True))
    cfg["tiled_dit"] = _to_bool(cfg.get("tiled_dit"), _to_bool(defaults.get("tiled_dit", False), False))
    # Guardrail: full mode + tiled_vae can produce heavily corrupted output on some systems.
    if cfg.get("mode") == "full" and cfg.get("tiled_vae"):
        cfg["tiled_vae"] = False
        cfg["_tiled_vae_note"] = "Disabled tiled_vae in full mode to avoid known corruption artifacts."
    cfg["unload_dit"] = _to_bool(cfg.get("unload_dit"), _to_bool(defaults.get("unload_dit", False), False))
    cfg["stream_decode"] = _to_bool(cfg.get("stream_decode"), _to_bool(defaults.get("stream_decode", False), False))
    cfg["cfg_scale"] = max(0.5, min(2.0, _to_float(cfg.get("cfg_scale"), _to_float(defaults.get("cfg_scale"), 1.0))))
    cfg["denoise_amount"] = max(
        0.5,
        min(
            2.0,
            _to_float(
                cfg.get("denoise_amount", cfg.get("denoising_strength")),
                _to_float(defaults.get("denoise_amount", defaults.get("denoising_strength", 1.0)), 1.0),
            ),
        ),
    )
    cfg["color_fix"] = _to_bool(cfg.get("color_fix"), _to_bool(defaults.get("color_fix", True), True))
    cfg["auto_transfer_output_to_input"] = _to_bool(
        cfg.get("auto_transfer_output_to_input"),
        _to_bool(defaults.get("auto_transfer_output_to_input", False), False),
    )
    cfg["keep_models_on_cpu"] = _to_bool(
        cfg.get("keep_models_on_cpu"),
        _to_bool(defaults.get("keep_models_on_cpu", True), True),
    )
    cfg["force_offload"] = _to_bool(cfg.get("force_offload"), _to_bool(defaults.get("force_offload", True), True))
    cfg["enable_debug"] = _to_bool(cfg.get("enable_debug"), _to_bool(defaults.get("enable_debug", False), False))

    cfg["seed"] = max(0, _to_int(cfg.get("seed"), 0))
    cfg["tile_size"] = max(32, min(1024, _to_int(cfg.get("tile_size"), 256)))
    cfg["overlap"] = max(8, min(512, _to_int(cfg.get("overlap"), 24)))
    if cfg["overlap"] >= cfg["tile_size"]:
        cfg["overlap"] = max(8, cfg["tile_size"] - 8)

    cfg["sparse_ratio"] = max(1.5, min(2.0, _to_float(cfg.get("sparse_ratio"), 2.0)))
    cfg["kv_ratio"] = max(1.0, min(3.0, _to_float(cfg.get("kv_ratio"), 3.0)))
    cfg["local_range"] = 9 if _to_int(cfg.get("local_range"), 11) == 9 else 11
    cfg["frame_chunk_size"] = max(0, _to_int(cfg.get("frame_chunk_size"), 0))
    cfg["save_vram_gb"] = clamp_save_vram_gb(cfg.get("save_vram_gb"), defaults.get("save_vram_gb", DEFAULT_SAVE_VRAM_GB))
    cfg["auto_tune_on_run"] = _to_bool(cfg.get("auto_tune_on_run", defaults.get("auto_tune_on_run", True)), True)
    tc = _to_int(cfg.get("temporal_compression"), 4)
    cfg["temporal_compression"] = tc if tc in (1, 2, 4) else 4
    cfg["decode_every"] = max(4, min(64, (_to_int(cfg.get("decode_every"), 16) // 4) * 4))

    cfg["fps"] = max(0.0, _to_float(cfg.get("fps"), 0.0))
    codec = str(cfg.get("codec", defaults.get("codec", "libx264")) or "libx264").strip()
    cfg["codec"] = codec if codec else "libx264"
    cfg["crf"] = max(0, min(51, _to_int(cfg.get("crf"), 18)))
    cfg["start_frame"] = max(0, _to_int(cfg.get("start_frame"), 0))
    cfg["end_frame"] = _to_int(cfg.get("end_frame"), -1)

    cfg["models_dir"] = str(cfg.get("models_dir", defaults.get("models_dir", "")) or "").strip()
    cfg["batch_enable"] = _to_bool(cfg.get("batch_enable"), _to_bool(defaults.get("batch_enable", False), False))
    cfg["keep_only_output_files"] = _to_bool(
        cfg.get("keep_only_output_files"),
        _to_bool(defaults.get("keep_only_output_files", False), False),
    )
    cfg["use_resolution_tab"] = _to_bool(
        cfg.get("use_resolution_tab"),
        _to_bool(defaults.get("use_resolution_tab", True), True),
    )
    cfg["max_target_resolution"] = max(0, min(8192, _to_int(cfg.get("max_target_resolution"), 1920)))
    cfg["pre_downscale_then_upscale"] = _to_bool(
        cfg.get("pre_downscale_then_upscale"),
        _to_bool(defaults.get("pre_downscale_then_upscale", True), True),
    )

    # Single-GPU guardrail
    device_str = str(cfg.get("device", defaults.get("device", "auto")) or "auto").strip()
    if "," in device_str:
        cfg["device"] = device_str.split(",")[0].strip() or "auto"
        cfg["_multi_gpu_disabled_reason"] = "FlashVSR is single-GPU."
    else:
        cfg["device"] = device_str or "auto"

    # Apply metadata defaults for selected version/mode/scale.
    internal_version = flashvsr_version_to_internal(cfg.get("version", "1.1"))
    model_id = f"v{internal_version}_{cfg.get('mode', 'full')}_{cfg.get('scale', '4')}x"
    model_meta = get_flashvsr_metadata(model_id)
    if model_meta:
        if cfg.get("tile_size", 0) <= 0:
            cfg["tile_size"] = int(model_meta.default_tile_size)
        if cfg.get("overlap", 0) <= 0:
            cfg["overlap"] = int(model_meta.default_overlap)

    # Compatibility aliases for older presets/components.
    cfg["dtype"] = cfg.get("precision", "auto")
    cfg["attention"] = cfg.get("attention_mode", "flash_attention_2")
    cfg["quality"] = cfg.get("crf", 18)

    return cfg


def _flashvsr_autotune_module():
    from shared.vram_autotune import flashvsr as flash_plan

    return flash_plan


def _flashvsr_preset_scope(payload: Dict[str, Any]) -> str:
    internal_version = flashvsr_version_to_internal(payload.get("version", "1.1"))
    return f"v{internal_version}_{payload.get('mode', 'full')}"


def _apply_flashvsr_preset(
    preset: Dict[str, Any],
    defaults: Dict[str, Any],
    preset_manager: PresetManager,
    current: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    from shared.services.video_sr_service import _apply_engine_preset

    return _apply_engine_preset(FLASHVSR_ENGINE, preset, defaults, preset_manager, current=current)


FLASHVSR_ENGINE = VideoSREngine(
    key="flashvsr",
    label="FlashVSR",
    order=FLASHVSR_ORDER,
    defaults_fn=lambda: flashvsr_defaults(),
    guardrails=lambda cfg, defaults: _enforce_flashvsr_guardrails(cfg, defaults),
    canonical_scale=lambda **kw: canonical_flashvsr_scale(**kw),
    scales_text="2x/4x",
    run=lambda *a, **kw: run_flashvsr(*a, **kw),
    autotune_module=_flashvsr_autotune_module,
    process_overhead_gb=0.0,  # set below from the Auto Tune module (lazy import keeps startup light)
    preset_scope=_flashvsr_preset_scope,
    preset_scope_keys=("version", "mode"),
    is_progress_line=lambda text_lc: text_lc.startswith("[flashvsr] processing..."),
    cancel_event=_flashvsr_cancel_event,
)


def build_flashvsr_callbacks(
    preset_manager: PresetManager,
    runner,
    run_logger: RunLogger,
    global_settings: Dict[str, Any],
    shared_state: gr.State,
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path,
):
    """Build FlashVSR callback functions for the UI (the shared video SR flow with the FlashVSR spec)."""
    FLASHVSR_ENGINE.process_overhead_gb = _flashvsr_autotune_module().FLASHVSR_PROCESS_OVERHEAD_GB
    return build_video_sr_callbacks(FLASHVSR_ENGINE, preset_manager, runner, run_logger, global_settings,
                                    shared_state, base_dir, temp_dir, output_dir)
