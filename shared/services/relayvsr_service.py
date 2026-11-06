"""
RelayVSR service: settings, defaults and guardrails; the run / batch / chunk / preview / preset flow is the shared video
SR service (shared/services/video_sr_service.py) driven by RELAYVSR_ENGINE.
"""
from pathlib import Path
from typing import Any, Dict, List

import gradio as gr

from shared.logging_utils import RunLogger
from shared.preset_manager import PresetManager
from shared.relayvsr_constants import (
    RELAYVSR_PRECISIONS,
    RELAYVSR_SCALE,
    RELAYVSR_WAN_DEFAULT,
    RELAYVSR_WAN_NAMES,
    RELAYVSR_WINDOW_DEFAULT,
    RELAYVSR_WINDOW_MAX,
    RELAYVSR_WINDOW_MIN,
)
from shared.services.video_sr_service import VideoSREngine, _to_bool, _to_float, _to_int, build_video_sr_callbacks
from shared.vram_autotune.common import DEFAULT_SAVE_VRAM_GB, clamp_save_vram_gb
from shared.vram_autotune.relayvsr import RELAYVSR_PROCESS_OVERHEAD_GB

RELAYVSR_ORDER: List[str] = [
    "input_path",
    "output_override",
    "output_format",
    "scale",
    "wan_model",
    "precision",
    "window_size",
    "wan_gpu_blocks",
    "seed",
    "auto_transfer_output_to_input",
    "device",
    "fps",
    "codec",
    "crf",
    "start_frame",
    "end_frame",
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
]


def canonical_relayvsr_scale(*, scale_value: Any = None, upscale_factor_value: Any = None, default: Any = 4) -> float:
    """Requested factor (1-4x in 0.5 steps). The model itself always runs at 4x; smaller factors pre-downscale."""
    raw = upscale_factor_value if upscale_factor_value is not None else (scale_value if scale_value is not None else default)
    value = _to_float(raw, _to_float(default, 4.0))
    return max(1.0, min(float(RELAYVSR_SCALE), round(value * 2.0) / 2.0))


def relayvsr_defaults(model_name: Any = None) -> Dict[str, Any]:
    try:
        from shared.gpu_utils import get_gpu_info

        device = "auto" if get_gpu_info() else "cpu"
    except Exception:
        device = "cpu"
    return {
        "input_path": "",
        "output_override": "",
        "output_format": "mp4",
        "scale": str(RELAYVSR_SCALE),
        "wan_model": RELAYVSR_WAN_DEFAULT,
        "precision": "auto",
        "window_size": RELAYVSR_WINDOW_DEFAULT,
        "wan_gpu_blocks": -1,
        "seed": 42,
        "auto_transfer_output_to_input": False,
        "device": device,
        "fps": 0.0,
        "codec": "libx264",
        "crf": 18,
        "start_frame": 0,
        "end_frame": -1,
        "save_metadata": True,
        "face_restore_after_upscale": False,
        "batch_enable": False,
        "batch_input_path": "",
        "batch_output_path": "",
        "use_resolution_tab": True,
        "upscale_factor": float(RELAYVSR_SCALE),
        "max_target_resolution": 3840,
        "pre_downscale_then_upscale": True,
        "resume_run_dir": "",
        "save_vram_gb": DEFAULT_SAVE_VRAM_GB,
        "keep_only_output_files": False,
        "auto_tune_on_run": True,
    }


def _enforce_relayvsr_guardrails(cfg: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {**defaults, **(cfg or {})}
    cfg["output_format"] = "mp4"
    cfg["scale"] = str(RELAYVSR_SCALE)
    cfg["upscale_factor"] = canonical_relayvsr_scale(upscale_factor_value=cfg.get("upscale_factor"),
                                                     default=defaults.get("upscale_factor", 4))
    wan = str(cfg.get("wan_model") or "")
    cfg["wan_model"] = wan if wan in RELAYVSR_WAN_NAMES else RELAYVSR_WAN_DEFAULT
    precision = str(cfg.get("precision") or "auto").strip().lower()
    cfg["precision"] = precision if precision in RELAYVSR_PRECISIONS else "auto"
    cfg["window_size"] = max(RELAYVSR_WINDOW_MIN, min(RELAYVSR_WINDOW_MAX, _to_int(cfg.get("window_size"), RELAYVSR_WINDOW_DEFAULT)))
    cfg["wan_gpu_blocks"] = max(-1, min(30, _to_int(cfg.get("wan_gpu_blocks"), -1)))
    cfg["seed"] = max(0, _to_int(cfg.get("seed"), 42))
    for key in ("auto_transfer_output_to_input", "save_metadata", "face_restore_after_upscale", "batch_enable",
                "keep_only_output_files", "use_resolution_tab", "pre_downscale_then_upscale", "auto_tune_on_run"):
        cfg[key] = _to_bool(cfg.get(key), _to_bool(defaults.get(key), False))
    cfg["fps"] = max(0.0, _to_float(cfg.get("fps"), 0.0))
    cfg["codec"] = str(cfg.get("codec") or "libx264").strip() or "libx264"
    cfg["crf"] = max(0, min(51, _to_int(cfg.get("crf"), 18)))
    cfg["start_frame"] = max(0, _to_int(cfg.get("start_frame"), 0))
    cfg["end_frame"] = _to_int(cfg.get("end_frame"), -1)
    cfg["max_target_resolution"] = max(0, min(8192, _to_int(cfg.get("max_target_resolution"), 3840)))
    cfg["save_vram_gb"] = clamp_save_vram_gb(cfg.get("save_vram_gb"), defaults.get("save_vram_gb", DEFAULT_SAVE_VRAM_GB))
    device = str(cfg.get("device") or "auto").strip() or "auto"
    if "," in device:
        cfg["device"] = device.split(",")[0].strip() or "auto"
        cfg["_multi_gpu_disabled_reason"] = "RelayVSR is single-GPU."
    else:
        cfg["device"] = device
    return cfg


def _relayvsr_run(*args, **kwargs):
    from shared.relayvsr_runner import run_relayvsr

    return run_relayvsr(*args, **kwargs)


def _relayvsr_autotune_module():
    from shared.vram_autotune import relayvsr

    return relayvsr


RELAYVSR_ENGINE = VideoSREngine(
    key="relayvsr",
    label="RelayVSR",
    order=RELAYVSR_ORDER,
    defaults_fn=relayvsr_defaults,
    guardrails=_enforce_relayvsr_guardrails,
    canonical_scale=canonical_relayvsr_scale,
    scales_text="1x-4x (the model runs at 4x; smaller factors pre-downscale the input)",
    run=_relayvsr_run,
    autotune_module=_relayvsr_autotune_module,
    process_overhead_gb=RELAYVSR_PROCESS_OVERHEAD_GB,
    is_progress_line=lambda text_lc: text_lc.startswith("frame_progress "),
    preview_frame_dir=False,
    fixed_model_scale=RELAYVSR_SCALE,
)


def build_relayvsr_callbacks(
    preset_manager: PresetManager,
    runner,
    run_logger: RunLogger,
    global_settings: Dict[str, Any],
    shared_state: gr.State,
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path,
):
    return build_video_sr_callbacks(RELAYVSR_ENGINE, preset_manager, runner, run_logger, global_settings, shared_state,
                                    base_dir, temp_dir, output_dir)
