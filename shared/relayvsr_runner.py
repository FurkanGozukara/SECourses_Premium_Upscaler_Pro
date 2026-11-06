"""
RelayVSR runner: model download, output naming, VRAM cap, and the tools/relayvsr_inference.py subprocess with live
progress and cancel. The subprocess / env / probe helpers are the LTX 2.5 and Open-H3 runners' own.
"""
from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .command_logger import get_command_logger
from .ltx25_runner import _build_env, _format_command_for_log, _probe_ltx25_video, _resolve_python_executable, _stream_process
from .open_h3_runner import _popen, _resolve_device
from .path_utils import collision_safe_path, detect_input_type, get_media_fps, normalize_path, resolve_output_location
from .relayvsr_constants import (
    RELAYVSR_FLASH_FILE,
    RELAYVSR_MODELS_DIRNAME,
    RELAYVSR_WAN_DEFAULT,
    RELAYVSR_WAN_FILES,
    RELAYVSR_WINDOW_DEFAULT,
    RELAYVSR_WINDOW_MAX,
    RELAYVSR_WINDOW_MIN,
)
from .vram_autotune.common import apply_vram_cap_env, is_tight

TAG = "RelayVSR"
CODECS = {"libx264", "libx265", "libsvtav1"}


@dataclass
class RelayVSRResult:
    returncode: int
    output_path: Optional[str]
    log: str
    input_fps: float = 30.0
    output_fps: float = 30.0


def _int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def model_paths(base_dir: Path, settings: Dict[str, Any]) -> Dict[str, Path]:
    models = Path(base_dir) / RELAYVSR_MODELS_DIRNAME
    wan = RELAYVSR_WAN_FILES.get(str(settings.get("wan_model") or ""), RELAYVSR_WAN_FILES[RELAYVSR_WAN_DEFAULT])
    return {"wan": models / wan, "flash": models / RELAYVSR_FLASH_FILE}


def build_command(python_exe: str, script: Path, settings: Dict[str, Any], paths: Dict[str, Path], input_path: str,
                  output_file: Path) -> List[str]:
    precision = str(settings.get("precision") or "auto").lower()
    codec = str(settings.get("codec") or "libx264")
    cmd = [
        python_exe, "-u", str(script),
        "--input", str(input_path), "--output", str(output_file),
        "--wan", str(paths["wan"]), "--flash", str(paths["flash"]),
        "--window_size", str(max(RELAYVSR_WINDOW_MIN, min(RELAYVSR_WINDOW_MAX, _int(settings.get("window_size"), RELAYVSR_WINDOW_DEFAULT)))),
        "--seed", str(max(0, _int(settings.get("seed"), 42))),
        "--dtype", "fp16" if precision == "fp16" else "bf16",
        "--start_frame", str(max(0, _int(settings.get("start_frame"), 0))),
        "--end_frame", str(_int(settings.get("end_frame"), -1)),
        "--max_gpu_blocks", str(_int(settings.get("wan_gpu_blocks"), -1)),
        "--save_vram_gb", str(float(settings.get("save_vram_gb") or 0.0)),
        "--codec", codec if codec in CODECS else "libx264",
        "--crf", str(max(0, min(51, _int(settings.get("crf"), 15)))),
        "--preset", str(settings.get("video_preset") or "medium"),
        "--pixel_format", str(settings.get("pixel_format") or "yuv420p"),
    ]
    try:
        fps = float(settings.get("fps") or 0.0)
    except Exception:
        fps = 0.0
    if fps > 0:
        cmd += ["--fps", f"{fps:.6f}"]
    return cmd


def _output_file(settings: Dict[str, Any], base_dir: Path, original: str, effective: str, is_image: bool) -> Path:
    ext = ".png" if is_image else ".mp4"
    stem = Path(str(settings.get("_original_filename") or original or effective)).stem or "output"
    override = str(settings.get("output_override") or "").strip()
    if override:
        path = Path(normalize_path(override))
        target = path.with_suffix(ext) if path.suffix else path / f"{stem}{ext}"
    else:
        resolved = resolve_output_location(
            input_path=original or effective, output_format="png" if is_image else "mp4",
            global_output_dir=settings.get("global_output_dir", str(Path(base_dir) / "outputs")), batch_mode=False,
            original_filename=settings.get("_original_filename"))
        target = resolved if resolved.suffix else resolved / f"{stem}{ext}"
    target = collision_safe_path(Path(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def run_relayvsr(
    settings: Dict[str, Any],
    base_dir: Path,
    on_progress: Optional[Callable[[str], None]] = None,
    cancel_event=None,
    process_handle: Optional[Dict] = None,
) -> RelayVSRResult:
    settings = dict(settings or {})
    start_time = time.time()
    log_lines: List[str] = []
    cmd: List[str] = []
    result: Optional[RelayVSRResult] = None

    def log(msg: str, transient: bool = False) -> None:
        text = str(msg)
        if not transient:
            log_lines.append(text)
        print(text, flush=True)
        if on_progress:
            with suppress(Exception):
                on_progress(text + "\n")

    try:
        original = normalize_path(settings.get("input_path", ""))
        effective = normalize_path(settings.get("_effective_input_path") or original)
        if not effective or not Path(effective).exists():
            return RelayVSRResult(1, None, "Invalid input path")
        is_image = detect_input_type(original or effective) == "image"
        paths = model_paths(base_dir, settings)
        if not all(p.is_file() for p in paths.values()):
            from .model_downloads import ensure_relayvsr_model

            ok, err = ensure_relayvsr_model(Path(base_dir), str(settings.get("wan_model") or RELAYVSR_WAN_DEFAULT),
                                            on_progress=on_progress, cancel_event=cancel_event)
            if not ok:
                return RelayVSRResult(1, None, f"RelayVSR model download failed:\n{err}")
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            return RelayVSRResult(1, None, "RelayVSR model files missing after download:\n" + "\n".join(missing))
        if str(settings.get("auto_tune_on_run", "")).strip().lower() in {"1", "true", "yes", "on"}:
            from .vram_autotune.relayvsr import auto_apply as relayvsr_auto_apply

            relayvsr_auto_apply(settings, effective, log=log)

        output_file = _output_file(settings, Path(base_dir), original, effective, is_image)
        script = Path(base_dir) / "tools" / "relayvsr_inference.py"
        python_exe = _resolve_python_executable(Path(base_dir))
        visible_gpu, gpu_note = _resolve_device(settings.get("device", "auto"))
        if gpu_note:
            log(gpu_note)
        cmd = build_command(python_exe, script, settings, paths, effective, output_file)
        log(f"Running RelayVSR 4x ({settings.get('wan_model') or RELAYVSR_WAN_DEFAULT}, window "
            f"{settings.get('window_size') or RELAYVSR_WINDOW_DEFAULT})")
        log(f"Command: {_format_command_for_log(cmd)}")
        env = _build_env(Path(base_dir), visible_gpu, log, tag=TAG)
        if visible_gpu != "":
            apply_vram_cap_env(env, visible_gpu if visible_gpu is not None else settings.get("device", "auto"),
                               settings.get("save_vram_gb"), overhead_gb=_overhead_gb(), log=log, tag=TAG,
                               settings_tight=is_tight(settings))
        proc = _popen(cmd, Path(base_dir), env)
        if process_handle is not None:
            process_handle["proc"] = proc
        returncode, reported, cancelled = _stream_process(proc, log, cancel_event, process_handle, tag=TAG, label=TAG)
        if process_handle is not None:
            process_handle["proc"] = None
        if cancelled:
            with suppress(Exception):
                output_file.unlink(missing_ok=True)
            return RelayVSRResult(1, None, "\n".join(log_lines + ["[Cancelled by user]"]))
        output_path = reported if reported and Path(reported).exists() else (str(output_file) if output_file.exists() else None)
        if returncode != 0 or not output_path:
            log(f"RelayVSR exited with code {returncode}; no output kept.")
            with suppress(Exception):
                output_file.unlink(missing_ok=True)
            result = RelayVSRResult(returncode or 1, None, "\n".join(log_lines))
        else:
            if not is_image and _probe_ltx25_video(Path(output_path), count_frames=False) is None:
                log("RelayVSR output has no readable video stream.")
                result = RelayVSRResult(1, None, "\n".join(log_lines))
            else:
                log(f"Output saved: {output_path}")
                in_fps = 1.0 if is_image else (get_media_fps(effective) or 30.0)
                result = RelayVSRResult(0, output_path, "\n".join(log_lines), input_fps=in_fps,
                                        output_fps=float(settings.get("fps") or 0.0) or in_fps)
    except Exception as exc:
        log_lines.append(f"RelayVSR error: {exc}")
        result = RelayVSRResult(1, None, "\n".join(log_lines))
    finally:
        if cmd:
            with suppress(Exception):
                get_command_logger(Path(base_dir) / "executed_commands").log_command(
                    tab_name="relayvsr", command=cmd, settings=settings,
                    returncode=result.returncode if result else -1,
                    output_path=result.output_path if result else None,
                    error_logs=log_lines[-50:] if result and result.returncode != 0 else None,
                    execution_time=time.time() - start_time,
                    additional_info={"wan_model": settings.get("wan_model"), "window_size": settings.get("window_size"),
                                     "precision": settings.get("precision")},
                )
    return result if result else RelayVSRResult(1, None, "\n".join(log_lines))


def _overhead_gb() -> float:
    from .vram_autotune.relayvsr import RELAYVSR_PROCESS_OVERHEAD_GB

    return RELAYVSR_PROCESS_OVERHEAD_GB
