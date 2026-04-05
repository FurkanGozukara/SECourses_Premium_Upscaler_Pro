from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .path_utils import collision_safe_path, detect_input_type, get_media_fps, normalize_path, sanitize_filename


def format_fps_value(value: Optional[float]) -> Optional[str]:
    try:
        if value is None:
            return None
        fps_val = float(value)
        if fps_val <= 0:
            return None
        return f"{fps_val:.3f}".rstrip("0").rstrip(".")
    except Exception:
        return None


def normalize_rife_multiplier(raw: Any) -> int:
    text = str(raw or "x2").strip().lower()
    if text.startswith("x"):
        text = text[1:]
    try:
        val = int(float(text))
    except Exception:
        val = 2
    if val <= 1:
        return 1
    if val <= 2:
        return 2
    if val <= 4:
        return 4
    return 8


def build_output_fps_summary(
    *,
    input_fps: Optional[float],
    seed_controls: Optional[Dict[str, Any]],
    output_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from .global_rife import global_rife_enabled

    seed_controls = seed_controls if isinstance(seed_controls, dict) else {}
    output_settings = output_settings if isinstance(output_settings, dict) else {}

    try:
        fps_override = float(seed_controls.get("fps_override_val", output_settings.get("fps_override", 0)) or 0.0)
    except Exception:
        fps_override = 0.0

    base_fps: Optional[float]
    try:
        src_fps = float(input_fps) if input_fps is not None else 0.0
    except Exception:
        src_fps = 0.0
    base_fps = fps_override if fps_override > 0 else (src_fps if src_fps > 0 else None)

    rife_on = bool(global_rife_enabled(seed_controls))
    mult_val = normalize_rife_multiplier(
        seed_controls.get(
            "global_rife_multiplier_val",
            output_settings.get("global_rife_multiplier", "x2"),
        )
    )

    final_fps = (base_fps * float(mult_val)) if (rife_on and base_fps and base_fps > 0) else base_fps
    value_text = format_fps_value(final_fps)

    if rife_on and fps_override > 0:
        label = f"Output FPS (FPS Override + Global RIFE x{mult_val})"
        value_class = "is-override"
    elif rife_on:
        label = f"Output FPS (Global RIFE x{mult_val})"
        value_class = "is-override"
    elif fps_override > 0:
        label = "Output FPS (FPS Override)"
        value_class = "is-override"
    else:
        label = "Output FPS (Base)"
        value_class = ""

    if not value_text:
        value_text = "Unavailable (input FPS unknown)"

    return {
        "label": label,
        "value": value_text,
        "value_class": value_class,
        "base_fps": base_fps,
        "final_fps": final_fps,
        "fps_override": fps_override,
        "global_rife_enabled": rife_on,
        "global_rife_multiplier": mult_val,
    }


def remux_video_fps(
    input_path: Path,
    output_path: Path,
    fps: float,
    *,
    on_progress: Optional[Callable[[str], None]] = None,
) -> tuple[bool, str]:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if fps <= 0:
        return False, "Invalid FPS override"
    if not input_path.exists():
        return False, f"Input not found: {input_path}"
    if shutil.which("ffmpeg") is None:
        return False, "ffmpeg not available"

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0",
        "-c",
        "copy",
        "-r",
        str(float(fps)),
        "-avoid_negative_ts",
        "make_zero",
    ]
    if output_path.suffix.lower() in {".mp4", ".m4v", ".mov"}:
        cmd.extend(["-movflags", "+faststart"])
    cmd.append(str(output_path))

    if on_progress:
        try:
            on_progress(f"Applying FPS override preprocess -> {output_path.name}\n")
        except Exception:
            pass

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as exc:
        return False, str(exc)

    if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1024:
        return True, ""
    return False, (proc.stderr or proc.stdout or "ffmpeg FPS remux failed").strip()


def apply_video_fps_override_preprocess(
    settings: Dict[str, Any],
    *,
    fps_key: str,
    run_dir: Path,
    on_progress: Optional[Callable[[str], None]] = None,
    input_key: str = "input_path",
    effective_input_key: Optional[str] = "_effective_input_path",
    original_input_key: str = "_original_input_path_before_preprocess",
    preprocessed_input_key: str = "_preprocessed_input_path",
) -> tuple[bool, str]:
    if not isinstance(settings, dict):
        return True, ""

    try:
        target_fps = float(settings.get(fps_key) or 0.0)
    except Exception:
        target_fps = 0.0

    if target_fps <= 0:
        return True, ""

    input_path_raw = settings.get(effective_input_key) if effective_input_key else None
    if not input_path_raw:
        input_path_raw = settings.get(input_key)
    input_path = normalize_path(str(input_path_raw or ""))
    if not input_path:
        return True, ""
    if detect_input_type(input_path) != "video":
        return True, ""

    source_fps = get_media_fps(input_path)
    source_fps_text = format_fps_value(source_fps)
    target_fps_text = format_fps_value(target_fps) or str(float(target_fps))

    if source_fps and abs(float(source_fps) - float(target_fps)) <= 0.01:
        settings[fps_key] = 0.0
        settings["_fps_override_requested"] = float(target_fps)
        settings["_fps_override_source_fps"] = float(source_fps)
        settings["_fps_override_target_fps"] = float(target_fps)
        return True, f"FPS override matches source FPS ({target_fps_text}); no remux needed."

    run_dir = Path(run_dir)
    original_name = str(settings.get("_original_filename") or Path(input_path).name or "input.mp4")
    original_suffix = Path(original_name).suffix or Path(input_path).suffix or ".mp4"
    safe_stem = sanitize_filename(Path(original_name).stem or "input")
    fps_token = target_fps_text.replace(".", "_")
    output_path = collision_safe_path(run_dir / f"fps_override_{safe_stem}_{fps_token}{original_suffix}")

    ok, err = remux_video_fps(Path(input_path), output_path, target_fps, on_progress=on_progress)
    if not ok:
        return False, (
            f"FPS override preprocess failed for {Path(input_path).name}: "
            f"{err or 'ffmpeg remux failed'}"
        )

    original_input = normalize_path(str(settings.get(original_input_key) or "")) or input_path
    settings[original_input_key] = original_input
    settings[preprocessed_input_key] = str(output_path)
    settings["_fps_override_preprocessed_input_path"] = str(output_path)
    settings["_fps_override_requested"] = float(target_fps)
    settings["_fps_override_source_fps"] = float(source_fps) if source_fps else 0.0
    settings["_fps_override_target_fps"] = float(target_fps)
    settings[input_key] = str(output_path)
    if effective_input_key:
        settings[effective_input_key] = str(output_path)
    settings[fps_key] = 0.0

    if source_fps_text:
        return True, (
            f"FPS override preprocess applied: {source_fps_text} -> {target_fps_text} FPS "
            f"({output_path.name})"
        )
    return True, f"FPS override preprocess applied: target {target_fps_text} FPS ({output_path.name})"
