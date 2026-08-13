"""
LTX 2.5 Upscaler subprocess runner for the SECourses pipeline.

Launches tools/ltx25_inference.py inside the app venv, streams its stdout
progress lines to the service, supports cancellation via process-tree kill,
and logs every executed command. Mirrors shared/sparkvsr_runner.py.
"""

from __future__ import annotations

import os
import queue
import random
import shlex
import subprocess
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .command_logger import get_command_logger
from .ltx25_constants import (
    LTX25_AUDIO_VAE_FILE,
    LTX25_IC_LORA_FILE,
    LTX25_MODELS_DIRNAME,
    LTX25_TE_FILES,
    LTX25_TE_INT8,
    LTX25_TRANSFORMER_FILES,
    LTX25_VAE_CONV,
    LTX25_VAE_FILES,
    ltx25_is_distilled,
)
from .model_downloads import ensure_ltx25_model
from .path_utils import (
    collision_safe_path,
    detect_input_type,
    get_media_fps,
    normalize_path,
    resolve_output_location,
)
from .process_control import terminate_process_tree


@dataclass
class Ltx25Result:
    returncode: int
    output_path: Optional[str]
    log: str
    input_fps: float = 30.0
    output_fps: float = 30.0


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


def _format_command_for_log(cmd: List[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([str(part) for part in cmd])
    return shlex.join([str(part) for part in cmd])


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds or 0))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _resolve_python_executable(base_dir: Path) -> str:
    if os.name == "nt":
        candidate = base_dir / "venv" / "Scripts" / "python.exe"
    else:
        candidate = base_dir / "venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def _normalize_cuda_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("cuda:"):
        text = text.split(":", 1)[1].strip()
    return text


def _resolve_ltx25_device(device_value: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve the device selection into a CUDA_VISIBLE_DEVICES value.

    The LTX 2.5 CLI has no --device flag; GPU isolation happens purely through
    the environment. Returns (visible_gpu_or_None, log_note_or_None) where an
    empty string clears CUDA (CPU mode) and None leaves the environment as-is.
    """
    raw = str(device_value or "").strip()
    raw_lower = raw.lower()
    if raw_lower in {"cpu", "none", "off"}:
        return "", "[LTX25] GPU isolation: CPU mode (CUDA_VISIBLE_DEVICES cleared)"
    if raw_lower in {"", "auto", "cuda"}:
        return None, None
    gpu_id = _normalize_cuda_token(raw)
    if gpu_id.isdigit():
        return gpu_id, f"[LTX25] GPU isolation: CUDA_VISIBLE_DEVICES={gpu_id}"
    return None, None


def _map_token_budget_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "manual" in text:
        return "manual"
    return "auto"


def _resolve_schedule(settings: Dict[str, Any], model_name: str) -> str:
    schedule = str(settings.get("schedule") or "").strip().lower()
    if schedule in {"distilled", "dev"}:
        return schedule
    return "distilled" if ltx25_is_distilled(model_name) else "dev"


def _resolve_model_files(
    base_dir: Path,
    settings: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Path]], List[str], Optional[str]]:
    """
    Resolve every checkpoint path required by the CLI from LTX25_Models.

    Returns (paths_dict_or_None, log_notes, error_message_or_None).
    """
    notes: List[str] = []
    models_dir = Path(base_dir) / LTX25_MODELS_DIRNAME
    model_name = str(settings.get("model_name") or "").strip()
    transformer_file = LTX25_TRANSFORMER_FILES.get(model_name)
    if transformer_file is None:
        return None, notes, f"Unknown LTX 2.5 model: {model_name}"

    transformer_path = models_dir / transformer_file
    override = normalize_path(settings.get("model_path") or "")
    if override:
        override_path = Path(override)
        if override_path.is_file():
            transformer_path = override_path
            notes.append(f"[LTX25] Using transformer path override: {override_path}")
        else:
            return None, notes, f"LTX 2.5 transformer override not found: {override_path}"

    te_name = str(settings.get("text_encoder") or "").strip()
    te_file = LTX25_TE_FILES.get(te_name) or LTX25_TE_FILES[LTX25_TE_INT8]
    vae_name = str(settings.get("video_vae") or "").strip()
    vae_file = LTX25_VAE_FILES.get(vae_name) or LTX25_VAE_FILES[LTX25_VAE_CONV]

    paths = {
        "transformer": transformer_path,
        "text_encoder": models_dir / te_file,
        "video_vae": models_dir / vae_file,
        "audio_vae": models_dir / LTX25_AUDIO_VAE_FILE,
        "ic_lora": models_dir / LTX25_IC_LORA_FILE,
    }
    return paths, notes, None


def _build_ltx25_command(
    python_exe: str,
    script_path: Path,
    settings: Dict[str, Any],
    paths: Dict[str, Path],
    input_path: str,
    output_file: Path,
    seed_value: int,
) -> List[str]:
    model_name = str(settings.get("model_name") or "").strip()
    schedule = _resolve_schedule(settings, model_name)
    cmd = [
        python_exe,
        "-u",
        str(script_path),
        "--input",
        str(input_path),
        "--output",
        str(output_file),
        "--transformer",
        str(paths["transformer"]),
        "--text_encoder",
        str(paths["text_encoder"]),
        "--video_vae",
        str(paths["video_vae"]),
        "--audio_vae",
        str(paths["audio_vae"]) if Path(paths["audio_vae"]).is_file() else "",
        "--ic_lora",
        str(paths["ic_lora"]),
        "--positive_prompt",
        str(settings.get("positive_prompt") or ""),
        "--negative_prompt",
        str(settings.get("negative_prompt") or ""),
        "--seed",
        str(int(seed_value)),
        "--steps",
        str(max(1, _parse_int(settings.get("steps"), 8))),
        "--cfg",
        str(_parse_float(settings.get("cfg"), 1.0)),
        "--sampler",
        str(settings.get("sampler") or "euler_ancestral"),
        "--schedule",
        schedule,
        "--ic_lora_strength",
        str(_parse_float(settings.get("ic_lora_strength"), 1.0)),
        "--guide_strength",
        str(_parse_float(settings.get("guide_strength"), 1.0)),
        "--token_budget_mode",
        _map_token_budget_mode(settings.get("token_budget_mode")),
        "--max_latent_tokens",
        str(max(4096, _parse_int(settings.get("max_latent_tokens"), 18000))),
        "--max_chunk_frames",
        str(max(9, _parse_int(settings.get("max_chunk_frames"), 121))),
        "--overlap_frames",
        str(max(1, _parse_int(settings.get("overlap_frames"), 1))),
        "--encode_tile_size",
        str(max(64, _parse_int(settings.get("encode_tile_size"), 512))),
        "--encode_tile_overlap",
        str(max(0, _parse_int(settings.get("encode_tile_overlap"), 64))),
        "--decode_tile_size",
        str(max(64, _parse_int(settings.get("decode_tile_size"), 512))),
        "--decode_tile_overlap",
        str(max(0, _parse_int(settings.get("decode_tile_overlap"), 64))),
        "--decode_temporal_size",
        str(max(8, _parse_int(settings.get("decode_temporal_size"), 128))),
        "--decode_temporal_overlap",
        str(max(0, _parse_int(settings.get("decode_temporal_overlap"), 32))),
        "--pre_resize_longer_edge",
        str(max(0, _parse_int(settings.get("pre_resize_longer_edge"), 0))),
        "--fps_override",
        str(max(0.0, _parse_float(settings.get("fps"), 0.0))),
        "--start_frame",
        str(max(0, _parse_int(settings.get("start_frame"), 0))),
        "--end_frame",
        str(_parse_int(settings.get("end_frame"), -1)),
        "--codec",
        str(settings.get("codec") or "libx264"),
        "--crf",
        str(max(0, min(51, _parse_int(settings.get("crf"), 15)))),
        "--preset",
        str(settings.get("video_preset") or settings.get("preset") or "medium"),
        "--pixel_format",
        str(settings.get("pixel_format") or "yuv420p"),
        "--reserve_vram_gb",
        str(max(0.0, _parse_float(settings.get("reserve_vram_gb"), 1.0))),
        "--attention",
        str(settings.get("attention_backend") or settings.get("attention") or "auto"),
        "--vram_mode",
        str(settings.get("vram_mode") or "auto"),
    ]
    if str(settings.get("audio_codec") or "").strip().lower() == "none":
        cmd.append("--no_audio")
    return cmd


def _build_env(base_dir: Path, visible_gpu: Optional[str], log: Callable[[str], None]) -> Dict[str, str]:
    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    }
    legacy_alloc_conf = env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if legacy_alloc_conf and not env.get("PYTORCH_ALLOC_CONF"):
        env["PYTORCH_ALLOC_CONF"] = legacy_alloc_conf
        log("[LTX25] Migrated PYTORCH_CUDA_ALLOC_CONF -> PYTORCH_ALLOC_CONF")
    if visible_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible_gpu
    triton_cache_dir = base_dir / "temp" / "triton_cache"
    with suppress(Exception):
        triton_cache_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("TRITON_CACHE_DIR", str(triton_cache_dir))
    return env


def _resolve_seed(settings: Dict[str, Any], log: Callable[[str], None]) -> int:
    if _bool(settings.get("randomize_seed"), False):
        seed_value = random.randint(0, 2**31 - 1)
        log(f"[LTX25] Randomized seed: {seed_value}")
        return seed_value
    return _parse_int(settings.get("seed"), 42)


def _stream_process(
    proc: subprocess.Popen,
    log: Callable[[str], None],
    cancel_event,
    process_handle: Optional[Dict],
) -> Tuple[int, Optional[str], bool]:
    """
    Stream stdout lines, heartbeat on silence, poll cancellation.

    Returns (returncode, reported_output_path, cancelled).
    """
    line_queue: "queue.Queue[Optional[str]]" = queue.Queue()

    def _read_output() -> None:
        try:
            if proc.stdout is None:
                return
            token: List[str] = []
            while True:
                char = proc.stdout.read(1)
                if char == "":
                    break
                if char in ("\n", "\r"):
                    line = "".join(token).strip()
                    token = []
                    if line:
                        line_queue.put(line)
                else:
                    token.append(char)
            tail = "".join(token).strip()
            if tail:
                line_queue.put(tail)
        except Exception:
            pass
        finally:
            line_queue.put(None)

    output_reader = threading.Thread(target=_read_output, daemon=True)
    output_reader.start()

    reported_output_path: Optional[str] = None
    last_activity = time.time()
    proc_started = time.time()
    last_progress_text = ""
    process_exit_seen_at: Optional[float] = None

    def _record_output_line(line: str) -> None:
        nonlocal last_activity, last_progress_text, reported_output_path
        text = str(line or "").strip()
        if not text:
            return
        if "OUTPUT_PATH:" in text:
            candidate = text.split("OUTPUT_PATH:", 1)[1].strip()
            if candidate:
                reported_output_path = candidate
        log(text)
        last_progress_text = text
        last_activity = time.time()

    while True:
        if cancel_event and cancel_event.is_set():
            log("Cancellation requested - terminating LTX 2.5 process")
            terminate_process_tree(proc)
            if process_handle is not None:
                process_handle["proc"] = None
            return 130, reported_output_path, True

        try:
            item = line_queue.get(timeout=0.25)
        except queue.Empty:
            now = time.time()
            if proc.poll() is not None:
                if process_exit_seen_at is None:
                    process_exit_seen_at = now
                elif now - process_exit_seen_at >= 1.0:
                    break
            if now - last_activity > 10:
                elapsed = _format_duration(now - proc_started)
                if last_progress_text:
                    heartbeat = f"[LTX25] still running | elapsed={elapsed} | last={last_progress_text[:180]}"
                else:
                    heartbeat = f"[LTX25] still running | elapsed={elapsed} | waiting for first progress output"
                log(heartbeat)
                last_activity = now
            continue

        if item is None:
            if proc.poll() is not None:
                break
            continue
        _record_output_line(item)

    while True:
        try:
            item = line_queue.get_nowait()
        except queue.Empty:
            break
        if item is not None:
            _record_output_line(item)

    return int(proc.wait()), reported_output_path, False


def run_ltx25(
    settings: Dict[str, Any],
    base_dir: Path,
    on_progress: Optional[Callable[[str], None]] = None,
    cancel_event=None,
    process_handle: Optional[Dict] = None,
) -> Ltx25Result:
    start_time = time.time()
    log_lines: List[str] = []
    cmd: List[str] = []
    result: Optional[Ltx25Result] = None

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
            return Ltx25Result(1, None, f"LTX 2.5 input path not found: {input_path}")
        input_kind = detect_input_type(input_path)
        if input_kind != "video":
            return Ltx25Result(
                1,
                None,
                (
                    f"LTX 2.5 expects a video input (got: {input_kind}). "
                    "Convert image sequences to a video first, or use another tab for single images."
                ),
            )

        seed_value = _resolve_seed(settings, log)

        model_name = str(settings.get("model_name") or "").strip()
        transformer_override = normalize_path(settings.get("model_path") or "")
        if not transformer_override:
            # ensure_ltx25_model performs its own existence check and returns
            # immediately when every required file is already on disk. The
            # cancel event terminates the downloader mid-transfer; verified
            # partial ranges are kept so the next run resumes the download.
            download_ok, download_error = ensure_ltx25_model(
                base_dir,
                model_name,
                str(settings.get("text_encoder") or ""),
                str(settings.get("video_vae") or ""),
                on_progress,
                cancel_event=cancel_event,
            )
            if not download_ok:
                if cancel_event is not None and cancel_event.is_set():
                    return Ltx25Result(130, None, "[Cancelled by user] Model download stopped.")
                return Ltx25Result(1, None, f"LTX 2.5 model download failed:\n{download_error}")

        paths, path_notes, path_error = _resolve_model_files(base_dir, settings)
        for note in path_notes:
            log(note)
        if path_error or paths is None:
            return Ltx25Result(1, None, path_error or "Failed to resolve LTX 2.5 model files.")
        missing = [
            str(p)
            for key, p in paths.items()
            if key != "audio_vae" and not Path(p).is_file()
        ]
        if missing:
            return Ltx25Result(
                1,
                None,
                "LTX 2.5 model files missing after download step:\n" + "\n".join(missing),
            )

        input_fps = get_media_fps(input_path) or 30.0
        fps_override = max(0.0, _parse_float(settings.get("fps"), 0.0))
        output_fps = fps_override if fps_override > 0 else input_fps

        base_name = settings.get("_original_filename") or Path(input_path).name
        base_stem = Path(str(base_name)).stem or "output"
        explicit_output = normalize_path(settings.get("output_path") or "")
        if explicit_output:
            output_file = Path(explicit_output)
        else:
            output_override = str(settings.get("output_override") or "").strip()
            if output_override:
                override_path = Path(normalize_path(output_override))
                if override_path.suffix:
                    output_file = collision_safe_path(override_path)
                else:
                    output_file = collision_safe_path(override_path / f"{base_stem}.mp4")
            else:
                resolved = resolve_output_location(
                    input_path=input_path,
                    output_format="mp4",
                    global_output_dir=settings.get("global_output_dir", str(base_dir / "outputs")),
                    batch_mode=False,
                    original_filename=settings.get("_original_filename"),
                )
                output_file = collision_safe_path(resolved if resolved.suffix else (resolved / f"{base_stem}.mp4"))
        output_file.parent.mkdir(parents=True, exist_ok=True)

        python_exe = _resolve_python_executable(base_dir)
        script_path = base_dir / "tools" / "ltx25_inference.py"
        if not script_path.exists():
            return Ltx25Result(1, None, f"LTX 2.5 inference script not found: {script_path}")

        visible_gpu, gpu_note = _resolve_ltx25_device(settings.get("device", "auto"))
        if gpu_note:
            log(gpu_note)
        if python_exe != sys.executable:
            log(f"[LTX25] Using venv python: {python_exe}")

        cmd = _build_ltx25_command(python_exe, script_path, settings, paths, input_path, output_file, seed_value)

        schedule = _resolve_schedule(settings, model_name)
        log(
            f"Running LTX 2.5 Upscaler (fixed 2x): model={model_name}, schedule={schedule}, "
            f"steps={settings.get('steps')}, cfg={settings.get('cfg')}, sampler={settings.get('sampler')}, "
            f"token_budget={_map_token_budget_mode(settings.get('token_budget_mode'))}, seed={seed_value}"
        )
        log(f"Command: {_format_command_for_log(cmd)}")

        env = _build_env(base_dir, visible_gpu, log)
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        preexec_fn = None if os.name == "nt" else os.setsid
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(base_dir),
            env=env,
            creationflags=creationflags,
            preexec_fn=preexec_fn,
        )
        if process_handle is not None:
            process_handle["proc"] = proc

        returncode, reported_output_path, cancelled = _stream_process(proc, log, cancel_event, process_handle)
        if process_handle is not None:
            process_handle["proc"] = None
        if cancelled:
            result = Ltx25Result(1, None, "\n".join(log_lines + ["[Cancelled by user]"]))
            return result

        output_path: Optional[str] = None
        if reported_output_path and Path(reported_output_path).exists():
            output_path = reported_output_path
        elif output_file.exists():
            output_path = str(output_file)

        if returncode == 130:
            log("LTX 2.5 process reported cancellation (exit code 130).")
            result = Ltx25Result(1, None, "\n".join(log_lines + ["[Cancelled by user]"]))
            return result
        if output_path and returncode != 0:
            log(f"LTX 2.5 exited with code {returncode} after producing output; treating run as successful.")
            returncode = 0
        if not output_path:
            log("No LTX 2.5 output file generated.")
            result = Ltx25Result(returncode or 1, None, "\n".join(log_lines))
        else:
            log(f"Output saved: {output_path}")
            result = Ltx25Result(
                int(returncode),
                output_path,
                "\n".join(log_lines),
                input_fps=input_fps,
                output_fps=output_fps,
            )

    except Exception as exc:
        log_lines.append(f"LTX 2.5 error: {exc}")
        result = Ltx25Result(1, None, "\n".join(log_lines))
    finally:
        if cmd:
            try:
                execution_time = time.time() - start_time
                get_command_logger(base_dir / "executed_commands").log_command(
                    tab_name="ltx25",
                    command=cmd,
                    settings=settings,
                    returncode=result.returncode if result else -1,
                    output_path=result.output_path if result else None,
                    error_logs=log_lines[-50:] if result and result.returncode != 0 else None,
                    execution_time=execution_time,
                    additional_info={
                        "model_name": settings.get("model_name"),
                        "text_encoder": settings.get("text_encoder"),
                        "video_vae": settings.get("video_vae"),
                        "schedule": settings.get("schedule"),
                        "steps": settings.get("steps"),
                        "cfg": settings.get("cfg"),
                        "sampler": settings.get("sampler"),
                        "token_budget_mode": settings.get("token_budget_mode"),
                        "max_latent_tokens": settings.get("max_latent_tokens"),
                        "pre_resize_longer_edge": settings.get("pre_resize_longer_edge"),
                        "attention_backend": settings.get("attention_backend"),
                    },
                )
                log_lines.append("Command logged to executed_commands folder")
            except Exception as exc:
                log_lines.append(f"Failed to log command: {exc}")

    return result if result else Ltx25Result(1, None, "\n".join(log_lines))


def run_ltx25_plan(
    settings: Dict[str, Any],
    base_dir: Path,
    on_progress: Optional[Callable[[str], None]] = None,
    cancel_event=None,
) -> Tuple[int, str]:
    """
    Run the engine with --plan_only to print the chunk plan without sampling.

    Does NOT download model weights: the plan path only reads the input video
    and queries free VRAM, so the checkpoint files are never opened.
    """
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
            return 1, f"LTX 2.5 input path not found: {input_path}"
        if detect_input_type(input_path) != "video":
            return 1, "LTX 2.5 chunk planning requires a video input."

        paths, path_notes, path_error = _resolve_model_files(base_dir, settings)
        for note in path_notes:
            log(note)
        if path_error or paths is None:
            return 1, path_error or "Failed to resolve LTX 2.5 model files."

        python_exe = _resolve_python_executable(base_dir)
        script_path = base_dir / "tools" / "ltx25_inference.py"
        if not script_path.exists():
            return 1, f"LTX 2.5 inference script not found: {script_path}"

        visible_gpu, gpu_note = _resolve_ltx25_device(settings.get("device", "auto"))
        if gpu_note:
            log(gpu_note)

        plan_output = Path(base_dir) / "temp" / "ltx25_plan_preview.mp4"
        seed_value = _parse_int(settings.get("seed"), 42)
        cmd = _build_ltx25_command(python_exe, script_path, settings, paths, input_path, plan_output, seed_value)
        cmd.append("--plan_only")
        log(f"[LTX25] Planning chunks (no sampling): {Path(input_path).name}")
        log(f"Command: {_format_command_for_log(cmd)}")

        env = _build_env(base_dir, visible_gpu, log)
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        preexec_fn = None if os.name == "nt" else os.setsid
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(base_dir),
            env=env,
            creationflags=creationflags,
            preexec_fn=preexec_fn,
        )
        returncode, _reported, cancelled = _stream_process(proc, log, cancel_event, None)
        if cancelled:
            return 130, "\n".join(log_lines + ["[Cancelled by user]"])
        return int(returncode), "\n".join(log_lines)
    except Exception as exc:
        log_lines.append(f"LTX 2.5 plan error: {exc}")
        return 1, "\n".join(log_lines)
