"""
Open-H3 Regenerate subprocess runner for the SECourses pipeline.

Launches tools/open_h3_inference.py inside the app venv, streams its stdout
progress lines to the service, supports cancellation via process-tree kill,
downloads the selected model files on first use and logs every executed
command. Mirrors shared/ltx25_runner.py.
"""

from __future__ import annotations

import math
import os
import random
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .command_logger import get_command_logger
from .vram_autotune.open_h3 import auto_apply as open_h3_auto_apply
from .ltx25_runner import (
    _build_env,
    _format_command_for_log,
    _probe_ltx25_video,
    _resolve_python_executable,
    _stream_process,
)
from .open_h3_constants import (
    OPENH3_AUDIO_VAE_FILE,
    OPENH3_CURVES_FILE,
    OPENH3_DEFAULT_SEED,
    OPENH3_JXL_LORA_FILE,
    OPENH3_JXL_MAX_SIDE_DEFAULT,
    OPENH3_JXL_PROMPT,
    OPENH3_JXL_SEED,
    OPENH3_JXL_STEPS,
    is_jxl_workflow,
    jxl_output_size,
    OPENH3_DIT_FILES,
    OPENH3_DIT_DEFAULT,
    OPENH3_DIT_INT8,
    OPENH3_LOW_BIT_INT8_FALLBACK,
    OPENH3_LORA_FILE,
    OPENH3_MODELS_DIRNAME,
    OPENH3_RESOLUTIONS,
    OPENH3_TABLE_FILE,
    OPENH3_TE_FILES,
    OPENH3_TE_NVFP4,
    OPENH3_UPSCALER_FILE,
    OPENH3_VAE_FILES,
    OPENH3_VAE_SOL_FP16,
)
from .path_utils import (
    collision_safe_path,
    detect_input_type,
    get_media_dimensions,
    get_media_fps,
    normalize_path,
    resolve_output_location,
)

TAG = "OpenH3"
LABEL = "Open-H3 Regenerate"


@dataclass
class OpenH3Result:
    returncode: int
    output_path: Optional[str]
    log: str
    input_fps: float = 24.0
    output_fps: float = 24.0


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def fps_mode(settings: Dict[str, Any]) -> str:
    """'24' (reference resample) or 'source' (1:1 frames); accepts engine values and UI labels."""
    value = str(settings.get("fps_mode") or "").strip()
    return "24" if value == "24" or ("24" in value and "resample" in value.lower()) else "source"


def audio_mode(settings: Dict[str, Any]) -> str:
    value = str(settings.get("audio_mode") or "source").strip().lower()
    return "source" if "source" in value else "silence"


def _even(value: float) -> int:
    return max(2, int(round(float(value) / 2.0)) * 2)


def jxl_max_side(settings: Dict[str, Any]) -> int:
    """JonXL largest output side: the JonXL Max Size setting, lowered by Max Edge when that is set."""
    side = max(256, min(2560, _parse_int(settings.get("jxl_max_side"), OPENH3_JXL_MAX_SIDE_DEFAULT)))
    max_edge = max(0, _parse_int(settings.get("max_target_resolution"), 0))
    return min(side, max_edge) if max_edge >= 256 else side


def compute_target_size(src_w: int, src_h: int, settings: Dict[str, Any]) -> Tuple[int, int, str]:
    """Output size for a src_w x src_h input: resolution preset (short side) or upscale factor, then max-edge cap.
    The JonXL H3 Upscale workflow fits the picture under its max size instead."""
    if is_jxl_workflow(settings):
        side = jxl_max_side(settings)
        width, height = jxl_output_size(src_w, src_h, side)
        return width, height, f"JonXL H3 Upscale, fitted under {side}x{side}"
    preset = str(settings.get("resolution_preset") or "").strip().lower()
    src_w, src_h = max(1, int(src_w)), max(1, int(src_h))
    short_side = None
    for key, (_pw, ph) in OPENH3_RESOLUTIONS.items():
        if preset.startswith(key):
            short_side = ph
            break
    if short_side is not None:
        scale = float(short_side) / float(min(src_w, src_h))
        note = f"preset {preset} (short side {short_side})"
    else:
        scale = max(1.0, _parse_float(settings.get("upscale_factor"), 2.0))
        note = f"upscale x{scale:g}"
    width, height = src_w * scale, src_h * scale
    max_edge = max(0, _parse_int(settings.get("max_target_resolution"), 0))
    if max_edge > 0 and max(width, height) > max_edge:
        shrink = float(max_edge) / max(width, height)
        width, height = width * shrink, height * shrink
        note += f", capped at max edge {max_edge}"
    return _even(width), _even(height), note


def _resolve_device(device_value: Any) -> Tuple[Optional[str], Optional[str]]:
    raw = str(device_value or "").strip().lower()
    if raw in {"cpu", "none", "off"}:
        return "", f"[{TAG}] GPU isolation: CPU mode (CUDA_VISIBLE_DEVICES cleared)"
    if raw in {"", "auto", "cuda"}:
        return None, None
    if raw.startswith("cuda:"):
        raw = raw.split(":", 1)[1].strip()
    if raw.isdigit():
        return raw, f"[{TAG}] GPU isolation: CUDA_VISIBLE_DEVICES={raw}"
    return None, None


def _draft_size(settings: Dict[str, Any], src_w: int, src_h: int) -> str:
    value = str(settings.get("draft_size") or "").strip().lower()
    if value.startswith("576"):
        return "576p"
    if value.startswith("768"):
        return "768p"
    if value.startswith("native"):
        return "native"
    # auto: H3-class drafts for 720p+ sources, the 576P setting for lower-resolution sources
    return "768p" if min(int(src_w), int(src_h)) >= 700 else "576p"


def resolve_model_files(base_dir: Path, settings: Dict[str, Any]) -> Tuple[Optional[Dict[str, Path]], List[str], Optional[str]]:
    notes: List[str] = []
    models_dir = Path(base_dir) / OPENH3_MODELS_DIRNAME
    dit_name = str(settings.get("model_name") or OPENH3_DIT_DEFAULT).strip()
    dit_file = OPENH3_DIT_FILES.get(dit_name)
    if dit_file is None:
        return None, notes, f"Unknown Open-H3 model: {dit_name}"
    dit_path = models_dir / dit_file
    override = normalize_path(settings.get("model_path") or "")
    if override:
        if Path(override).is_file():
            dit_path = Path(override)
            notes.append(f"[{TAG}] Using DiT path override: {override}")
        else:
            return None, notes, f"Open-H3 DiT override not found: {override}"
    te_file = OPENH3_TE_FILES.get(str(settings.get("text_encoder") or "")) or OPENH3_TE_FILES[OPENH3_TE_NVFP4]
    vae_file = OPENH3_VAE_FILES.get(str(settings.get("video_vae") or "")) or OPENH3_VAE_FILES[OPENH3_VAE_SOL_FP16]
    return {
        "dit": dit_path,
        "text_encoder": models_dir / te_file,
        "vae": models_dir / vae_file,
        "audio_vae": models_dir / OPENH3_AUDIO_VAE_FILE,
        "upscaler": models_dir / OPENH3_UPSCALER_FILE,
        "lora": models_dir / OPENH3_LORA_FILE,
        "curves": models_dir / OPENH3_CURVES_FILE,
        "table": models_dir / OPENH3_TABLE_FILE,
        "jxl_lora": models_dir / OPENH3_JXL_LORA_FILE,
    }, notes, None


def _needed_paths(paths: Dict[str, Path], settings: Dict[str, Any]) -> Dict[str, Path]:
    """Model files the selected workflow reads."""
    if is_jxl_workflow(settings):
        return {k: paths[k] for k in ("dit", "text_encoder", "vae", "jxl_lora")}
    out = {k: v for k, v in paths.items() if k != "jxl_lora"}
    if _parse_float(settings.get("curve_strength"), 1.0) == 0:
        out.pop("curves", None)
    return out


def _expected_frames(source_frames: int, source_fps: float, settings: Dict[str, Any]) -> int:
    first = max(0, _parse_int(settings.get("start_frame"), 0))
    end = _parse_int(settings.get("end_frame"), -1)
    stop = source_frames if end < 0 else min(source_frames, end + 1)
    count = max(0, stop - first)
    if fps_mode(settings) == "24" and abs(float(source_fps) - 24.0) > 1e-3:
        return max(1, int(round(count / float(source_fps) * 24.0)))
    return count


def _validate_video_output(input_path: Path, output_path: Path, settings: Dict[str, Any],
                           expected_fps: float) -> Tuple[bool, str]:
    """Complete, frame-exact output before reporting model success (same checks as LTX 2.5)."""
    source = _probe_ltx25_video(Path(input_path), count_frames=True)
    output = _probe_ltx25_video(Path(output_path), count_frames=True)
    if source is None:
        return False, "input has no probeable video stream"
    if output is None:
        return False, "output has no usable video stream"

    def _frames(stream) -> Optional[int]:
        try:
            value = int(str(stream.get("nb_read_frames") or stream.get("nb_frames") or "").strip())
            return value if value > 0 else None
        except Exception:
            return None

    def _rate(raw) -> float:
        try:
            value = float(Fraction(str(raw)))
            return value if math.isfinite(value) and value > 0 else 0.0
        except Exception:
            return 0.0

    source_frames, output_frames = _frames(source), _frames(output)
    if source_frames is None or output_frames is None:
        return False, "decoded frame count is unavailable"
    source_fps = _rate(source.get("r_frame_rate")) or _rate(source.get("avg_frame_rate")) or 24.0
    expected = _expected_frames(source_frames, source_fps, settings)
    if output_frames != expected:
        return False, f"decoded frame count mismatch {output_frames}/{expected}"
    actual_fps = _rate(output.get("r_frame_rate")) or _rate(output.get("avg_frame_rate"))
    if actual_fps <= 0:
        return False, "output frame rate is unavailable"
    if expected_fps > 0 and abs(actual_fps - float(expected_fps)) > max(1e-6, 1e-4 * float(expected_fps)):
        return False, f"frame rate mismatch {actual_fps:.9f}/{float(expected_fps):.9f}"
    return True, f"video={output_frames} frames at {actual_fps:.9f} fps"


def _write_prompt_file(base_dir: Path, settings: Dict[str, Any]) -> Path:
    folder = Path(base_dir) / "temp" / "open_h3_prompt"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"prompt_{os.getpid()}_{int(time.time() * 1000)}.txt"
    text = OPENH3_JXL_PROMPT if is_jxl_workflow(settings) else str(settings.get("prompt") or "")
    path.write_text(text, encoding="utf-8")
    return path



def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)

def build_command(python_exe: str, script_path: Path, settings: Dict[str, Any], paths: Dict[str, Path],
                  input_path: str, output_file: Path, seed_value: int, target: Tuple[int, int],
                  draft_size: str, prompt_file: Optional[Path], cache_dir: Path) -> List[str]:
    cmd = [
        python_exe, "-u", str(script_path),
        "--input", str(input_path),
        "--output", str(output_file),
        "--dit", str(paths["dit"]),
        "--text_encoder", str(paths["text_encoder"]),
        "--vae", str(paths["vae"]),
        "--audio_vae", str(paths["audio_vae"]),
        "--upscaler", str(paths["upscaler"]),
        "--lora", str(paths["lora"]),
        "--curves", str(paths["curves"]) if _parse_float(settings.get("curve_strength"), 1.0) != 0 else "",
        "--table", str(paths["table"]),
        "--cache_dir", str(cache_dir),
        "--seed", str(int(seed_value)),
        "--steps", str(max(1, min(8, _parse_int(settings.get("steps"), 3)))),
        "--start_sigma", str(max(0.0, min(1.0, _parse_float(settings.get("start_sigma"), 0.0)))),
        "--lora_strength", str(_parse_float(settings.get("lora_strength"), 1.0)),
        "--curve_strength", str(_parse_float(settings.get("curve_strength"), 1.0)),
        "--target_width", str(int(target[0])),
        "--target_height", str(int(target[1])),
        "--draft_size", draft_size,
        "--max_window_frames", str(max(5, min(361, _parse_int(settings.get("max_window_frames"), 121)))),
        "--overlap_frames", str(max(0, _parse_int(settings.get("overlap_frames"), 8))),
        "--fps_mode", fps_mode(settings),
        "--audio_mode", audio_mode(settings),
        "--start_frame", str(max(0, _parse_int(settings.get("start_frame"), 0))),
        "--end_frame", str(_parse_int(settings.get("end_frame"), -1)),
        "--sparse_tau", str(_parse_float(settings.get("sparse_tau"), 1.0)),
        "--sparse_end_percent", str(_parse_float(settings.get("sparse_end_percent"), 0.9)),
        "--sparse_dense_steps", str(_parse_int(settings.get("sparse_dense_steps"), 1)),
        "--second_pass_sigma", str(max(0.0, min(1.0, _parse_float(settings.get("second_pass_sigma"), 0.0)))),
        "--dense_blocks", str(settings.get("dense_blocks") or ""),
        "--sink_conditioning", str(settings.get("sink_conditioning") or "exact_kv_and_rows"),
        "--ffn_chunk_tokens", str(max(0, _parse_int(settings.get("ffn_chunk_tokens"), 32768))),
        "--token_budget_mode", "manual" if "manual" in str(settings.get("token_budget_mode") or "").lower() else "auto",
        "--max_tokens", str(max(20000, _parse_int(settings.get("max_tokens"), 160000))),
        "--save_vram_gb", str(max(0.0, _parse_float(settings.get("reserve_vram_gb"), 2.0))),
        "--max_gpu_blocks", str(max(0, min(50, _parse_int(settings.get("max_gpu_blocks"), 50)))),
        "--attention", str(settings.get("attention_backend") or "auto"),
        "--codec", str(settings.get("codec") or "libx264"),
        "--crf", str(max(0, min(51, _parse_int(settings.get("crf"), 15)))),
        "--preset", str(settings.get("video_preset") or settings.get("preset") or "medium"),
        "--pixel_format", str(settings.get("pixel_format") or "yuv420p"),
    ]
    if prompt_file is not None:
        cmd += ["--prompt_file", str(prompt_file)]
    if not _bool(settings.get("sparse_attention"), True):
        cmd.append("--no_sparse")
    if str(settings.get("audio_codec") or "").strip().lower() == "none":
        cmd.append("--no_audio")
    if _bool(settings.get("keep_latents"), False):
        cmd.append("--keep_latents")
    if is_jxl_workflow(settings):
        cmd += ["--workflow", "jxl", "--jxl_lora", str(paths["jxl_lora"]),
                "--jxl_max_side", str(jxl_max_side(settings)), "--jxl_steps", str(OPENH3_JXL_STEPS)]
    return cmd


def _prepare_workflow(settings: Dict[str, Any], log: Callable[[str], None]) -> None:
    """JonXL H3 Upscale runs every frame 1:1 in one process (no 24 fps resample, no separate stage processes)."""
    if not is_jxl_workflow(settings):
        return
    settings["fps_mode"] = "source"
    if _bool(settings.get("isolated_stages"), False):
        settings["isolated_stages"] = False
        log(f"[{TAG}] JonXL H3 Upscale runs in one process; 'Run each stage as a separate process' is not used.")


def _resolve_seed(settings: Dict[str, Any], log: Callable[[str], None]) -> int:
    if _bool(settings.get("randomize_seed"), False):
        seed_value = random.randint(0, 2**31 - 1)
        log(f"[{TAG}] Randomized seed: {seed_value}")
        return seed_value
    seed_value = _parse_int(settings.get("seed"), OPENH3_DEFAULT_SEED)
    if is_jxl_workflow(settings) and seed_value == OPENH3_DEFAULT_SEED:
        return OPENH3_JXL_SEED  # JonXL's fixed workflow seed while Seed holds the Open-H3 default
    return seed_value


def _popen(cmd: List[str], base_dir: Path, env: Dict[str, str]) -> subprocess.Popen:
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    if os.name == "nt":
        creationflags |= subprocess.CREATE_NO_WINDOW
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        cwd=str(base_dir), env=env, creationflags=creationflags,
        preexec_fn=None if os.name == "nt" else os.setsid,
    )


ENGINE_STAGES = ("text", "encode", "refine", "decode")
STAGE_LABELS = {
    "text": "text encoding",
    "encode": "VAE encode + latent upscale",
    "refine": "DiT refine",
    "decode": "VAE decode + write",
}


def _run_engine(cmd: List[str], settings: Dict[str, Any], base_dir: Path, env: Dict[str, str],
                log: Callable[[str], None], cancel_event, process_handle: Optional[Dict]):
    """Run the engine command; with "Run each stage as a separate process" the four stages run one
    after another in their own processes, handing results over through files, so a finished stage
    (text encoder, VAE) holds no RAM or VRAM while the next one runs.

    Returns (returncode, reported_output, cancelled)."""
    runs: List[List[str]] = [cmd]
    work_dir: Optional[Path] = None
    if _bool(settings.get("isolated_stages"), False):
        work_dir = Path(base_dir) / "temp" / "open_h3_stages" / f"run_{os.getpid()}_{int(time.time() * 1000)}"
        work_dir.mkdir(parents=True, exist_ok=True)
        runs = [cmd + ["--stages", stage, "--work_dir", str(work_dir)] for stage in ENGINE_STAGES]
    returncode, reported, cancelled = 1, None, False
    try:
        for index, stage_cmd in enumerate(runs):
            if work_dir is not None:
                stage = ENGINE_STAGES[index]
                log(f"[{TAG}] Stage {index + 1}/{len(runs)}: {STAGE_LABELS[stage]} (separate process)")
            proc = _popen(stage_cmd, Path(base_dir), env)
            if process_handle is not None:
                process_handle["proc"] = proc
            returncode, reported, cancelled = _stream_process(proc, log, cancel_event, process_handle,
                                                              tag=TAG, label=LABEL)
            if process_handle is not None:
                process_handle["proc"] = None
            if cancelled or returncode not in (0, 3):  # 3 = some batch items failed, the rest go on
                break
    finally:
        if work_dir is not None:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)
    return returncode, reported, cancelled


def _item_plan(settings: Dict[str, Any], base_dir: Path, input_path: str, input_kind: str):
    """Per-input sizes and output file, or an error string."""
    dims = get_media_dimensions(input_path)
    if not dims:
        return f"Could not read the input size: {input_path}"
    src_w, src_h = int(dims[0]), int(dims[1])
    target_w, target_h, size_note = compute_target_size(src_w, src_h, settings)
    draft = _draft_size(settings, src_w, src_h)
    input_fps = (get_media_fps(input_path) or 24.0) if input_kind == "video" else 24.0
    output_fps = 24.0 if (fps_mode(settings) == "24" and input_kind == "video") else input_fps

    out_ext = ".png" if input_kind == "image" else ".mp4"
    base_name = settings.get("_original_filename") or Path(input_path).name
    base_stem = Path(str(base_name)).stem or "output"
    explicit_output = normalize_path(settings.get("output_path") or "")
    if explicit_output:
        output_file = Path(explicit_output)
    else:
        override = str(settings.get("output_override") or "").strip()
        if override:
            override_path = Path(normalize_path(override))
            if override_path.suffix:
                output_file = collision_safe_path(override_path.with_suffix(out_ext))
            else:
                output_file = collision_safe_path(override_path / f"{base_stem}{out_ext}")
        else:
            resolved = resolve_output_location(
                input_path=input_path,
                output_format="png" if input_kind == "image" else "mp4",
                global_output_dir=settings.get("global_output_dir", str(Path(base_dir) / "outputs")),
                batch_mode=False,
                original_filename=settings.get("_original_filename"),
            )
            output_file = collision_safe_path(resolved if resolved.suffix else (resolved / f"{base_stem}{out_ext}"))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    return (src_w, src_h), (target_w, target_h), size_note, draft, input_fps, output_fps, output_file


def run_open_h3(
    settings: Dict[str, Any],
    base_dir: Path,
    on_progress: Optional[Callable[[str], None]] = None,
    cancel_event=None,
    process_handle: Optional[Dict] = None,
) -> OpenH3Result:
    settings = dict(settings or {})
    start_time = time.time()
    log_lines: List[str] = []
    cmd: List[str] = []
    result: Optional[OpenH3Result] = None
    prompt_file: Optional[Path] = None

    def log(message: str) -> None:
        text = str(message or "")
        if text:
            log_lines.append(text)
            if on_progress:
                with suppress(Exception):
                    on_progress(text)

    try:
        input_path = normalize_path(settings.get("_effective_input_path") or settings.get("input_path") or "")
        if not input_path or not Path(input_path).exists():
            return OpenH3Result(1, None, f"Open-H3 input path not found: {input_path}")
        input_kind = detect_input_type(input_path)
        if input_kind not in {"video", "image"}:
            return OpenH3Result(1, None, f"Open-H3 expects a video or an image (got: {input_kind}).")

        # Output-tab FPS override: convert the whole input to the requested CFR first so duration
        # and audio sync stay exact (same guard as LTX 2.5).
        requested_fps = max(0.0, _parse_float(settings.get("fps"), 0.0))
        if requested_fps > 0 and input_kind == "video":
            from .video_fps_utils import apply_video_fps_override_preprocess

            preprocess_root = Path(settings.get("_run_dir") or (Path(base_dir) / "temp" / "open_h3_fps")) / "fps_preprocess"
            fps_ok, fps_note = apply_video_fps_override_preprocess(settings, fps_key="fps", run_dir=preprocess_root, on_progress=log)
            if fps_note:
                log(fps_note)
            if not fps_ok:
                return OpenH3Result(1, None, fps_note or "Open-H3 FPS preprocess failed")
            input_path = normalize_path(settings.get("_effective_input_path") or settings.get("input_path") or "")

        _prepare_workflow(settings, log)
        seed_value = _resolve_seed(settings, log)
        low_bit_int8 = OPENH3_LOW_BIT_INT8_FALLBACK.get(str(settings.get("model_name") or OPENH3_DIT_DEFAULT).strip())
        if low_bit_int8 and not normalize_path(settings.get("model_path") or ""):
            from .models.seedvr2_w4a8 import w4a8_fallback_reason

            reason = w4a8_fallback_reason(settings.get("device"))
            if reason:
                log(f"[Open-H3] {settings.get('model_name') or OPENH3_DIT_DEFAULT} cannot run here ({reason}); "
                    f"using {low_bit_int8} instead.")
                settings["model_name"] = low_bit_int8
        if not normalize_path(settings.get("model_path") or ""):
            from .model_downloads import ensure_open_h3_model

            ok, error = ensure_open_h3_model(
                base_dir,
                str(settings.get("model_name") or OPENH3_DIT_DEFAULT),
                str(settings.get("text_encoder") or OPENH3_TE_NVFP4),
                str(settings.get("video_vae") or OPENH3_VAE_SOL_FP16),
                on_progress,
                cancel_event=cancel_event,
                jxl=is_jxl_workflow(settings),
            )
            if not ok:
                if cancel_event is not None and cancel_event.is_set():
                    return OpenH3Result(130, None, "[Cancelled by user] Model download stopped.")
                return OpenH3Result(1, None, f"Open-H3 model download failed:\n{error}")

        paths, notes, error = resolve_model_files(base_dir, settings)
        for note in notes:
            log(note)
        if error or paths is None:
            return OpenH3Result(1, None, error or "Failed to resolve Open-H3 model files.")
        missing = [str(p) for p in _needed_paths(paths, settings).values() if not Path(p).is_file()]
        if missing:
            return OpenH3Result(1, None, "Open-H3 model files missing after download step:\n" + "\n".join(missing))

        item = _item_plan(settings, base_dir, input_path, input_kind)
        if isinstance(item, str):
            return OpenH3Result(1, None, item)
        (src_w, src_h), (target_w, target_h), size_note, draft, input_fps, output_fps, output_file = item

        python_exe = _resolve_python_executable(base_dir)
        script_path = Path(base_dir) / "tools" / "open_h3_inference.py"
        if not script_path.exists():
            return OpenH3Result(1, None, f"Open-H3 inference script not found: {script_path}")
        visible_gpu, gpu_note = _resolve_device(settings.get("device", "auto"))
        if gpu_note:
            log(gpu_note)

        if _truthy(settings.get("auto_tune_on_run")):
            open_h3_auto_apply(settings, log)
        prompt_file = _write_prompt_file(base_dir, settings)
        cache_dir = Path(base_dir) / OPENH3_MODELS_DIRNAME / ".cache"
        cmd = build_command(python_exe, script_path, settings, paths, input_path, output_file, seed_value,
                            (target_w, target_h), draft, prompt_file, cache_dir)
        if is_jxl_workflow(settings):
            log(f"Running JonXL H3 Upscale: {src_w}x{src_h} -> {target_w}x{target_h} ({size_note}), "
                f"model={settings.get('model_name')}, {OPENH3_JXL_STEPS} steps, seed={seed_value}")
        else:
            log(f"Running Open-H3 Regenerate: {src_w}x{src_h} -> {target_w}x{target_h} ({size_note}), draft {draft}, "
                f"model={settings.get('model_name')}, steps={settings.get('steps')}, seed={seed_value}")
        log(f"Command: {_format_command_for_log(cmd)}")

        env = _build_env(Path(base_dir), visible_gpu, log, tag=TAG)
        returncode, reported, cancelled = _run_engine(cmd, settings, Path(base_dir), env, log, cancel_event,
                                                      process_handle)
        if cancelled or returncode == 130:
            return OpenH3Result(1, None, "\n".join(log_lines + ["[Cancelled by user]"]))

        output_path: Optional[str] = None
        if reported and Path(reported).exists():
            output_path = reported
        elif output_file.exists():
            output_path = str(output_file)

        if returncode != 0:
            log(f"Open-H3 exited with code {returncode}; refusing partial/stale output.")
            if output_path and input_kind == "video":
                with suppress(Exception):
                    Path(output_path).unlink(missing_ok=True)
            result = OpenH3Result(returncode or 1, None, "\n".join(log_lines))
        elif not output_path:
            log("No Open-H3 output file generated.")
            result = OpenH3Result(1, None, "\n".join(log_lines))
        else:
            valid, detail = True, "image written"
            if input_kind == "video":
                valid, detail = _validate_video_output(Path(input_path), Path(output_path), settings, float(output_fps))
            if not valid:
                log(f"Open-H3 output validation failed: {detail}")
                with suppress(Exception):
                    Path(output_path).unlink(missing_ok=True)
                result = OpenH3Result(1, None, "\n".join(log_lines))
            else:
                log(f"Open-H3 output validation passed: {detail}")
                log(f"Output saved: {output_path}")
                result = OpenH3Result(0, output_path, "\n".join(log_lines), input_fps=float(input_fps),
                                      output_fps=float(output_fps))
    except Exception as exc:
        log_lines.append(f"Open-H3 error: {exc}")
        result = OpenH3Result(1, None, "\n".join(log_lines))
    finally:
        if prompt_file is not None:
            with suppress(Exception):
                prompt_file.unlink(missing_ok=True)
        if cmd:
            try:
                get_command_logger(Path(base_dir) / "executed_commands").log_command(
                    tab_name="open_h3",
                    command=cmd,
                    settings=settings,
                    returncode=result.returncode if result else -1,
                    output_path=result.output_path if result else None,
                    error_logs=log_lines[-50:] if result and result.returncode != 0 else None,
                    execution_time=time.time() - start_time,
                    additional_info={
                        "model_name": settings.get("model_name"),
                        "text_encoder": settings.get("text_encoder"),
                        "video_vae": settings.get("video_vae"),
                        "resolution_preset": settings.get("resolution_preset"),
                        "steps": settings.get("steps"),
                        "start_sigma": settings.get("start_sigma"),
                        "draft_size": settings.get("draft_size"),
                        "sparse_attention": settings.get("sparse_attention"),
                    },
                )
                log_lines.append("Command logged to executed_commands folder")
            except Exception as exc:
                log_lines.append(f"Failed to log command: {exc}")

    return result if result else OpenH3Result(1, None, "\n".join(log_lines))


def run_open_h3_batch(
    items: List[Dict[str, Any]],
    base_dir: Path,
    on_progress: Optional[Callable[[str], None]] = None,
    cancel_event=None,
    process_handle: Optional[Dict] = None,
) -> List[OpenH3Result]:
    """Several inputs in ONE engine process: every model loads once for the whole batch.

    All items share the settings of items[0] except their input/output paths, so the
    caller only groups items that differ in nothing else."""
    start_time = time.time()
    log_lines: List[str] = []
    results: List[OpenH3Result] = [OpenH3Result(1, None, "not processed") for _ in items]
    if not items:
        return results

    def log(message: str) -> None:
        text = str(message or "")
        if text:
            log_lines.append(text)
            if on_progress:
                with suppress(Exception):
                    on_progress(text)

    base = dict(items[0])
    cmd: List[str] = []
    prompt_file: Optional[Path] = None
    batch_file: Optional[Path] = None
    try:
        _prepare_workflow(base, log)
        for settings in items:
            if is_jxl_workflow(settings):
                settings["fps_mode"] = "source"
        seed_value = _resolve_seed(base, log)
        if not normalize_path(base.get("model_path") or ""):
            from .model_downloads import ensure_open_h3_model

            ok, error = ensure_open_h3_model(base_dir, str(base.get("model_name") or OPENH3_DIT_DEFAULT),
                                             str(base.get("text_encoder") or OPENH3_TE_NVFP4),
                                             str(base.get("video_vae") or OPENH3_VAE_SOL_FP16),
                                             on_progress, cancel_event=cancel_event, jxl=is_jxl_workflow(base))
            if not ok:
                message = "[Cancelled by user] Model download stopped." if (cancel_event and cancel_event.is_set()) \
                    else f"Open-H3 model download failed:\n{error}"
                return [OpenH3Result(1, None, message) for _ in items]
        paths, notes, error = resolve_model_files(base_dir, base)
        for note in notes:
            log(note)
        if error or paths is None:
            return [OpenH3Result(1, None, error or "Failed to resolve Open-H3 model files.") for _ in items]
        missing = [str(p) for p in _needed_paths(paths, base).values() if not Path(p).is_file()]
        if missing:
            return [OpenH3Result(1, None, "Open-H3 model files missing after download step:\n" + "\n".join(missing))
                    for _ in items]

        plans = []
        for index, settings in enumerate(items):
            input_path = normalize_path(settings.get("_effective_input_path") or settings.get("input_path") or "")
            kind = detect_input_type(input_path) if input_path and Path(input_path).exists() else "missing"
            if kind not in {"video", "image"}:
                results[index] = OpenH3Result(1, None, f"Open-H3 input not usable ({kind}): {input_path}")
                continue
            item = _item_plan(settings, base_dir, input_path, kind)
            if isinstance(item, str):
                results[index] = OpenH3Result(1, None, item)
                continue
            plans.append((index, input_path, kind, item))
        if not plans:
            return results

        import json

        work = Path(base_dir) / "temp" / "open_h3_batch"
        work.mkdir(parents=True, exist_ok=True)
        batch_file = work / f"batch_{os.getpid()}_{int(time.time() * 1000)}.json"
        batch_file.write_text(json.dumps([
            {"input": input_path, "output": str(item[6]), "target_width": int(item[1][0]),
             "target_height": int(item[1][1]), "draft_size": item[3], "start_frame": max(0, _parse_int(items[index].get("start_frame"), 0)),
             "end_frame": _parse_int(items[index].get("end_frame"), -1)}
            for index, input_path, _kind, item in plans
        ], indent=1), encoding="utf-8")
        first_index, first_input, _k, first_item = plans[0]
        if _truthy(base.get("auto_tune_on_run")):
            open_h3_auto_apply(base, log)
        prompt_file = _write_prompt_file(base_dir, base)
        python_exe = _resolve_python_executable(base_dir)
        script_path = Path(base_dir) / "tools" / "open_h3_inference.py"
        cmd = build_command(python_exe, script_path, base, paths, first_input, first_item[6], seed_value,
                            first_item[1], first_item[3], prompt_file, Path(base_dir) / OPENH3_MODELS_DIRNAME / ".cache")
        cmd += ["--batch_json", str(batch_file)]
        log(f"Running {'JonXL H3 Upscale' if is_jxl_workflow(base) else 'Open-H3 Regenerate'} on {len(plans)} input(s) "
            f"in one engine process (models load once), seed={seed_value}")
        log(f"Command: {_format_command_for_log(cmd)}")
        visible_gpu, gpu_note = _resolve_device(base.get("device", "auto"))
        if gpu_note:
            log(gpu_note)
        env = _build_env(Path(base_dir), visible_gpu, log, tag=TAG)
        returncode, _reported, cancelled = _run_engine(cmd, base, Path(base_dir), env, log, cancel_event,
                                                       process_handle)
        if cancelled or returncode == 130:
            return [OpenH3Result(1, None, "\n".join(log_lines + ["[Cancelled by user]"])) for _ in items]

        done: Dict[int, str] = {}
        failed: Dict[int, str] = {}
        for text in log_lines:
            if "ITEM_DONE" in text and "OUTPUT_PATH:" in text:
                try:
                    done[int(text.split("ITEM_DONE", 1)[1].split()[0])] = text.split("OUTPUT_PATH:", 1)[1].strip()
                except Exception:
                    pass
            elif "ITEM_ERROR" in text:
                try:
                    head, msg = text.split("ITEM_ERROR", 1)[1].split(":", 1)
                    failed[int(head.strip())] = msg.strip()
                except Exception:
                    pass
        if len(plans) == 1 and returncode == 0 and Path(first_item[6]).exists():
            done.setdefault(0, str(first_item[6]))
        full_log = "\n".join(log_lines)
        for position, (index, input_path, kind, item) in enumerate(plans):
            output = done.get(position)
            if not output or not Path(output).exists():
                reason = failed.get(position) or f"engine exit code {returncode}"
                results[index] = OpenH3Result(1, None, full_log + f"\n[OpenH3] {Path(input_path).name} failed: {reason}")
                continue
            valid, detail = True, "image written"
            if kind == "video":
                valid, detail = _validate_video_output(Path(input_path), Path(output), items[index], float(item[5]))
            if not valid:
                with suppress(Exception):
                    Path(output).unlink(missing_ok=True)
                results[index] = OpenH3Result(1, None, full_log + f"\nOpen-H3 output validation failed: {detail}")
            else:
                results[index] = OpenH3Result(0, output, full_log + f"\nOpen-H3 output validation passed: {detail}",
                                              input_fps=float(item[4]), output_fps=float(item[5]))
    except Exception as exc:
        log_lines.append(f"Open-H3 batch error: {exc}")
        results = [r if r.returncode == 0 else OpenH3Result(1, None, "\n".join(log_lines)) for r in results]
    finally:
        for path in (prompt_file, batch_file):
            if path is not None:
                with suppress(Exception):
                    path.unlink(missing_ok=True)
        if cmd:
            with suppress(Exception):
                get_command_logger(Path(base_dir) / "executed_commands").log_command(
                    tab_name="open_h3",
                    command=cmd,
                    settings=base,
                    returncode=0 if all(r.returncode == 0 for r in results) else 1,
                    output_path=next((r.output_path for r in results if r.output_path), None),
                    error_logs=log_lines[-50:] if any(r.returncode != 0 for r in results) else None,
                    execution_time=time.time() - start_time,
                    additional_info={"batch_items": len(items), "model_name": base.get("model_name")},
                )
    return results


def run_open_h3_plan(settings: Dict[str, Any], base_dir: Path, on_progress: Optional[Callable[[str], None]] = None,
                     cancel_event=None) -> Tuple[int, str]:
    """--plan_only run: geometry and window plan, no model loaded or downloaded."""
    log_lines: List[str] = []

    def log(message: str) -> None:
        text = str(message or "")
        if text:
            log_lines.append(text)
            if on_progress:
                with suppress(Exception):
                    on_progress(text)

    try:
        input_path = normalize_path(settings.get("_effective_input_path") or settings.get("input_path") or "")
        if not input_path or not Path(input_path).exists():
            return 1, f"Open-H3 input path not found: {input_path}"
        paths, _notes, error = resolve_model_files(base_dir, settings)
        if error or paths is None:
            return 1, error or "Failed to resolve Open-H3 model files."
        dims = get_media_dimensions(input_path)
        if not dims:
            return 1, f"Could not read the input size: {input_path}"
        target_w, target_h, size_note = compute_target_size(int(dims[0]), int(dims[1]), settings)
        draft = _draft_size(settings, int(dims[0]), int(dims[1]))
        log(f"[Plan] target {target_w}x{target_h} ({size_note}), draft {draft}")
        python_exe = _resolve_python_executable(base_dir)
        script_path = Path(base_dir) / "tools" / "open_h3_inference.py"
        cmd = build_command(python_exe, script_path, settings, paths, input_path,
                            Path(base_dir) / "temp" / "open_h3_plan_preview.mp4", 0, (target_w, target_h), draft,
                            None, Path(base_dir) / OPENH3_MODELS_DIRNAME / ".cache")
        cmd.append("--plan_only")
        visible_gpu, _note = _resolve_device(settings.get("device", "auto"))
        env = _build_env(Path(base_dir), visible_gpu, log, tag=TAG)
        proc = _popen(cmd, Path(base_dir), env)
        returncode, _reported, cancelled = _stream_process(proc, log, cancel_event, None, tag=TAG, label=LABEL)
        if cancelled:
            return 130, "\n".join(log_lines + ["[Cancelled by user]"])
        return int(returncode), "\n".join(log_lines)
    except Exception as exc:
        log_lines.append(f"Open-H3 plan error: {exc}")
        return 1, "\n".join(log_lines)
