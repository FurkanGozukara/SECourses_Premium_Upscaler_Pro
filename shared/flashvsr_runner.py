"""
FlashVSR+ Runner - Interface for FlashVSR+ Video Super-Resolution

Provides subprocess wrapper for FlashVSR+ CLI (run.py) with:
- Local model directory resolution for offline use
- Support for video and image sequence inputs
- Tiled processing for memory efficiency
- Color correction and FPS control
- Multiple pipeline modes (tiny, tiny-long, full)
"""

import subprocess
import sys
import time
import shutil
import tempfile
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from contextlib import suppress

from .path_utils import (
    normalize_path,
    collision_safe_path,
    detect_input_type,
    IMAGE_EXTENSIONS,
    get_media_fps,
    resolve_output_location
)
from .command_logger import get_command_logger
from .models.flashvsr_meta import flashvsr_version_to_internal, flashvsr_version_to_ui


@dataclass
class FlashVSRResult:
    """Result of FlashVSR+ processing"""
    returncode: int
    output_path: Optional[str]
    log: str
    input_fps: float = 30.0
    output_fps: float = 30.0


def _normalize_cuda_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("cuda:"):
        text = text.split(":", 1)[1].strip()
    return text


def _resolve_flashvsr_device(device_value: Any) -> tuple[str, Optional[str], Optional[str]]:
    """
    Convert app GPU selection into a FlashVSR-safe runtime tuple:
    - CLI device arg (`-d`)
    - CUDA_VISIBLE_DEVICES value (or None to leave unchanged)
    - Optional log note
    """
    raw = str(device_value or "").strip()
    raw_lower = raw.lower()

    if raw_lower in {"cpu", "none", "off"}:
        return "cpu", "", "[FlashVSR] GPU isolation: CPU mode (CUDA_VISIBLE_DEVICES cleared)"

    if raw_lower in {"", "auto"}:
        return "auto", None, None

    gpu_id = _normalize_cuda_token(raw)
    if gpu_id.isdigit():
        # Restrict to the selected physical GPU and remap to local cuda:0.
        return "cuda:0", gpu_id, f"[FlashVSR] GPU isolation: CUDA_VISIBLE_DEVICES={gpu_id}, device remapped to cuda:0"

    # Unknown format: pass through untouched.
    return raw, None, None


def _resolve_python_executable(base_dir: Path) -> str:
    """
    Prefer project venv Python for subprocess consistency.
    Falls back to current interpreter when venv executable is missing.
    """
    if os.name == "nt":
        candidate = base_dir / "venv" / "Scripts" / "python.exe"
    else:
        candidate = base_dir / "venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


_FLASHVSR_REQUIRED_FILES = (
    "diffusion_pytorch_model_streaming_dmd.safetensors",
    "Wan2.1_VAE.pth",
    "LQ_proj_in.ckpt",
    "TCDecoder.ckpt",
)


def _expected_flashvsr_model_dir_name(version_internal: str) -> str:
    return "FlashVSR-v1.1" if str(version_internal) == "11" else "FlashVSR"


def _missing_required_model_files(model_dir: Path) -> List[str]:
    missing: List[str] = []
    for name in _FLASHVSR_REQUIRED_FILES:
        if not (model_dir / name).exists():
            missing.append(name)
    return missing


def _ensure_local_flashvsr_model_layout(base_dir: Path, version_internal: str) -> tuple[bool, str]:
    """
    Validate local FlashVSR model folders expected by upstream FlashVSR_plus/run.py:
    - FlashVSR_plus/models/FlashVSR
    - FlashVSR_plus/models/FlashVSR-v1.1
    """
    models_root = base_dir / "FlashVSR_plus" / "models"
    target_name = _expected_flashvsr_model_dir_name(version_internal)
    target_dir = models_root / target_name

    if not models_root.exists():
        return False, f"[FlashVSR] models root not found: {models_root}"

    if not target_dir.exists():
        return False, (
            f"[FlashVSR] Local model folder missing: {target_dir}. "
            f"Expected files: {', '.join(_FLASHVSR_REQUIRED_FILES)}"
        )

    missing = _missing_required_model_files(target_dir)
    if missing:
        return False, (
            f"[FlashVSR] Local model folder is incomplete: {target_dir}. "
            f"Missing: {', '.join(missing)}"
        )

    return True, f"[FlashVSR] Using local model folder: {target_dir}"


def _is_known_tiled_dit_temp_name_bug(log_text: str) -> bool:
    text = str(log_text or "").lower()
    return (
        "unboundlocalerror" in text
        and "temp_name" in text
        and "referenced before assignment" in text
    )


def _is_bf16_runtime_issue(log_text: str) -> bool:
    text = str(log_text or "").lower()
    needles = (
        "bfloat16",
        "bf16",
        "not implemented for",
        "does not support",
        "unsupported datatype",
        "unsupported dtype",
        "expected scalar type",
    )
    return any(n in text for n in needles)


def _ffmpeg_python_api_compatible() -> bool:
    """
    FlashVSR_plus/run.py expects ffmpeg-python API (`probe`, `Error`).
    Some environments have a different `ffmpeg` package installed.
    """
    try:
        import ffmpeg as _ffmpeg  # type: ignore

        return bool(hasattr(_ffmpeg, "probe") and hasattr(_ffmpeg, "Error"))
    except Exception:
        return False


def _build_flashvsr_ffmpeg_shim_source() -> str:
    """
    Minimal ffmpeg-python compatibility layer for FlashVSR_plus/run.py.
    It only implements the API surface used by `merge_video_with_audio`.
    """
    return """# Auto-generated compatibility shim for FlashVSR+ upstream ffmpeg import.
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List


class Error(Exception):
    def __init__(self, cmd=None, stdout=None, stderr=None):
        super().__init__("ffmpeg command failed")
        self.cmd = cmd
        self.stdout = stdout
        self.stderr = stderr


def _run(cmd: List[str], quiet: bool = False) -> subprocess.CompletedProcess:
    kwargs: Dict[str, Any] = {"check": False}
    if quiet:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    proc = subprocess.run(cmd, **kwargs)
    if proc.returncode != 0:
        raise Error(
            cmd=cmd,
            stdout=getattr(proc, "stdout", None),
            stderr=getattr(proc, "stderr", None),
        )
    return proc


def probe(path: str) -> Dict[str, Any]:
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        raise Error(cmd=["ffprobe"], stderr=b"ffprobe not found in PATH")

    proc = _run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-print_format",
            "json",
            str(path),
        ],
        quiet=True,
    )
    raw = proc.stdout if isinstance(proc.stdout, (bytes, bytearray)) else b""
    return json.loads(raw.decode("utf-8", errors="replace") or "{}")


@dataclass
class _StreamRef:
    path: str
    selector: str


class _InputRef:
    def __init__(self, path: str):
        self.path = str(path)

    def __getitem__(self, selector: str) -> _StreamRef:
        return _StreamRef(path=self.path, selector=str(selector))


def input(path: str) -> _InputRef:
    return _InputRef(path)


def _selector_with_default(selector: str, default_selector: str) -> str:
    sel = str(selector or "").strip()
    return sel or default_selector


class _OutputRef:
    def __init__(
        self,
        video_stream: _StreamRef,
        audio_stream: _StreamRef,
        output_path: str,
        vcodec: str = "copy",
        acodec: str = "copy",
    ):
        self.video_stream = video_stream
        self.audio_stream = audio_stream
        self.output_path = str(output_path)
        self.vcodec = str(vcodec or "copy")
        self.acodec = str(acodec or "copy")

    def run(self, overwrite_output: bool = False, quiet: bool = False):
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise Error(cmd=["ffmpeg"], stderr=b"ffmpeg not found in PATH")

        video_sel = _selector_with_default(self.video_stream.selector, "v:0")
        audio_sel = _selector_with_default(self.audio_stream.selector, "a:0")

        cmd = [ffmpeg_bin]
        if overwrite_output:
            cmd.append("-y")
        cmd.extend(
            [
                "-i",
                self.video_stream.path,
                "-i",
                self.audio_stream.path,
                "-map",
                f"0:{video_sel}",
                "-map",
                f"1:{audio_sel}",
                "-c:v",
                self.vcodec,
                "-c:a",
                self.acodec,
                self.output_path,
            ]
        )
        _run(cmd, quiet=quiet)
        return None


def output(*args, **kwargs) -> _OutputRef:
    if len(args) < 3:
        raise ValueError("output() expects at least: video_stream, audio_stream, output_path")

    out_path = args[-1]
    streams = args[:-1]
    if len(streams) != 2:
        raise ValueError("output() shim only supports one video and one audio stream")

    video_stream, audio_stream = streams
    if not isinstance(video_stream, _StreamRef) or not isinstance(audio_stream, _StreamRef):
        raise TypeError("output() expects stream refs returned by input(...)[...]")

    return _OutputRef(
        video_stream=video_stream,
        audio_stream=audio_stream,
        output_path=str(out_path),
        vcodec=str(kwargs.get("vcodec", "copy")),
        acodec=str(kwargs.get("acodec", "copy")),
    )
"""


def _prepare_flashvsr_ffmpeg_shim(base_dir: Path) -> Optional[Path]:
    """
    Create/update a local `ffmpeg.py` shim that emulates the tiny subset
    of ffmpeg-python used by FlashVSR_plus/run.py.
    """
    try:
        shim_dir = base_dir / "temp" / "_flashvsr_ffmpeg_shim"
        shim_dir.mkdir(parents=True, exist_ok=True)
        shim_file = shim_dir / "ffmpeg.py"
        src = _build_flashvsr_ffmpeg_shim_source()
        if (not shim_file.exists()) or shim_file.read_text(encoding="utf-8") != src:
            shim_file.write_text(src, encoding="utf-8")
        return shim_dir
    except Exception:
        return None


def run_flashvsr(
    settings: Dict[str, Any],
    base_dir: Path,
    on_progress: Optional[Callable[[str], None]] = None,
    cancel_event=None,
    process_handle: Optional[Dict] = None
) -> FlashVSRResult:
    """
    Run FlashVSR+ upscaling.
    
    settings must include:
    - input_path: str (video or image folder)
    - output_path: str (output directory)
    - scale: int (2 or 4)
    - version: str ("1.0"/"1.1" UI, or legacy "10"/"11")
    - mode: str ("tiny", "tiny-long", "full")
    - tiled_vae: bool
    - tiled_dit: bool
    - tile_size: int
    - overlap: int
    - unload_dit: bool
    - color_fix: bool
    - seed: int
    - dtype: str ("fp16" or "bf16")
    - device: str (GPU ID or "auto")
    - fps: int (for image sequences)
    - quality: int (1-10, video quality)
    - attention: str ("sage" or "block")
    
    Returns:
        FlashVSRResult with processing outcome
    """
    start_time = time.time()
    log_lines = []
    cmd: List[str] = []
    result: Optional[FlashVSRResult] = None

    def log(msg: str):
        log_lines.append(msg)
        try:
            print(str(msg), flush=True)
        except Exception:
            pass
        if on_progress:
            try:
                on_progress(msg + "\n")
            except Exception:
                # Never let UI/console callback issues fail the actual run.
                pass
    temp_input_dir: Optional[Path] = None

    try:
        # Validate input (support preprocessed effective input path)
        original_input_path = normalize_path(settings.get("input_path", ""))
        effective_input_path = normalize_path(settings.get("_effective_input_path") or original_input_path)
        if not effective_input_path or not Path(effective_input_path).exists():
            return FlashVSRResult(returncode=1, output_path=None, log="Invalid input path")

        # FlashVSR CLI expects either a video path or a directory of images.
        # For single-image inputs, wrap the image into a temporary one-frame folder.
        effective_type = detect_input_type(effective_input_path)
        if effective_type == "image":
            try:
                src_img = Path(effective_input_path)
                temp_input_dir = Path(tempfile.mkdtemp(prefix="flashvsr_single_frame_"))
                ext = src_img.suffix.lower()
                if ext not in IMAGE_EXTENSIONS:
                    ext = ".png"
                dst_img = temp_input_dir / f"frame_000001{ext}"
                shutil.copy2(src_img, dst_img)
                effective_input_path = str(temp_input_dir)
                log(f"[FlashVSR] Single image input wrapped as frame sequence: {effective_input_path}")
            except Exception as e:
                return FlashVSRResult(
                    returncode=1,
                    output_path=None,
                    log=f"Failed to prepare single-image input for FlashVSR: {e}",
                )
        
        # Determine output path (naming should follow ORIGINAL input)
        output_override = settings.get("output_override", "")
        explicit_output_file: Optional[Path] = None
        if output_override:
            override_path = Path(normalize_path(output_override))
            # FlashVSR CLI writes into an output FOLDER; support file-path override by
            # running in the parent folder and renaming the produced mp4 after.
            if override_path.suffix:
                explicit_output_file = collision_safe_path(override_path)
                output_folder = str(explicit_output_file.parent)
            else:
                output_folder = str(override_path)
        else:
            # Use default output naming
            output_folder = resolve_output_location(
                input_path=original_input_path,
                output_format="mp4",
                global_output_dir=settings.get("global_output_dir", str(base_dir / "outputs")),
                batch_mode=False,
                original_filename=settings.get("_original_filename"),
            )
            # FlashVSR expects a folder, not a file
            if Path(output_folder).suffix:
                output_folder = str(Path(output_folder).parent)
        
        output_folder_path = Path(output_folder)
        output_folder_path.mkdir(parents=True, exist_ok=True)
        # Track pre-existing outputs so we can reliably locate the output from THIS run.
        pre_existing_mp4 = {p.resolve() for p in output_folder_path.glob("*.mp4")}
        
        # Get settings with defaults
        scale = int(settings.get("scale", 4))
        raw_version = str(settings.get("version", "1.0"))
        version = flashvsr_version_to_internal(raw_version)
        version_ui = flashvsr_version_to_ui(raw_version)
        mode = settings.get("mode", "tiny")
        dtype = settings.get("dtype", "bf16")
        device = settings.get("device", "auto")
        device_arg, visible_gpu, gpu_note = _resolve_flashvsr_device(device)
        fps = int(settings.get("fps", 30))
        quality = int(settings.get("quality", 6))
        attention = settings.get("attention", "sage")
        seed = int(settings.get("seed", 0))

        layout_ok, layout_msg = _ensure_local_flashvsr_model_layout(base_dir, version)
        log(layout_msg)
        if not layout_ok:
            return FlashVSRResult(
                returncode=1,
                output_path=None,
                log="\n".join(log_lines),
            )

        # FlashVSR+ naming (mirrors FlashVSR_plus/run.py):
        #   FlashVSR_{mode}_{name.split('.')[0]}_{seed}.mp4
        base_name = os.path.basename(original_input_path.rstrip("/\\"))
        base_no_ext = base_name.split(".")[0] if base_name else "FlashVSR"
        expected_output_path = output_folder_path / f"FlashVSR_{mode}_{base_no_ext}_{seed}.mp4"
        
        # Tile settings
        tiled_vae = bool(settings.get("tiled_vae", False))
        tiled_dit = bool(settings.get("tiled_dit", False))
        tile_size = int(settings.get("tile_size", 256))
        overlap = int(settings.get("overlap", 24))
        unload_dit = bool(settings.get("unload_dit", True))
        color_fix = bool(settings.get("color_fix", False))

        mode_norm = str(mode or "").strip().lower()
        # Upstream FlashVSR_plus/run.py has a known bug in tiled_dit path for non tiny-long modes:
        # `temp_name` is referenced before assignment. Disable proactively to avoid a noisy first failure.
        if tiled_dit and mode_norm != "tiny-long":
            log(
                "[FlashVSR] Upstream tiled_dit bug for mode!='tiny-long' detected; "
                "disabling tiled_dit for this run."
            )
            tiled_dit = False
        
        # Build command
        flashvsr_script = base_dir / "FlashVSR_plus" / "run.py"
        if not flashvsr_script.exists():
            return FlashVSRResult(
                returncode=1,
                output_path=None,
                log=f"FlashVSR+ script not found at {flashvsr_script}"
            )

        python_exe = _resolve_python_executable(base_dir)
        if python_exe != sys.executable:
            log(f"[FlashVSR] Using venv python: {python_exe}")

        # Run subprocess with cancellation support
        import platform

        # Platform-specific process group creation
        creationflags = 0
        preexec_fn = None
        if platform.system() == "Windows":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            preexec_fn = os.setsid
        
        triton_cache_dir = base_dir / "temp" / "triton_cache"
        with suppress(Exception):
            triton_cache_dir.mkdir(parents=True, exist_ok=True)

        proc_env = {
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
        # Force offline/local model usage for FlashVSR+ to avoid HuggingFace fetch.
        proc_env.setdefault("HF_HUB_OFFLINE", "1")
        proc_env.setdefault("TRANSFORMERS_OFFLINE", "1")
        if visible_gpu is not None:
            proc_env["CUDA_VISIBLE_DEVICES"] = visible_gpu
        if gpu_note:
            log(gpu_note)
        legacy_alloc_conf = proc_env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        if legacy_alloc_conf and not proc_env.get("PYTORCH_ALLOC_CONF"):
            proc_env["PYTORCH_ALLOC_CONF"] = legacy_alloc_conf
            log("[FlashVSR] Migrated PYTORCH_CUDA_ALLOC_CONF -> PYTORCH_ALLOC_CONF")
        proc_env.setdefault("TRITON_CACHE_DIR", str(triton_cache_dir))

        if not _ffmpeg_python_api_compatible():
            shim_dir = _prepare_flashvsr_ffmpeg_shim(base_dir)
            if shim_dir is not None:
                existing_pythonpath = str(proc_env.get("PYTHONPATH") or "")
                shim_path = str(shim_dir)
                proc_env["PYTHONPATH"] = (
                    f"{shim_path}{os.pathsep}{existing_pythonpath}"
                    if existing_pythonpath
                    else shim_path
                )
                log(
                    "[FlashVSR] Incompatible Python 'ffmpeg' package detected; "
                    "using compatibility shim to avoid upstream audio-merge errors."
                )
            else:
                log(
                    "[FlashVSR] Python package 'ffmpeg' is not ffmpeg-python compatible "
                    "(missing probe/Error). Upstream audio-merge traceback can appear, "
                    "but video output is still usable."
                )

        def _build_cmd(run_dtype: str, run_tiled_dit: bool) -> List[str]:
            local_cmd = [
                python_exe,
                str(flashvsr_script),
                "-i", effective_input_path,
                "-s", str(scale),
                "-v", version,
                "-m", mode,
                "-t", str(run_dtype),
                "-d", device_arg,
                "-f", str(fps),
                "-q", str(quality),
                "-a", attention,
                "--seed", str(seed),
            ]
            if tiled_vae:
                local_cmd.append("--tiled-vae")
            if run_tiled_dit:
                local_cmd.append("--tiled-dit")
            if run_tiled_dit or tiled_vae:
                local_cmd.extend(["--tile-size", str(tile_size)])
                local_cmd.extend(["--overlap", str(overlap)])
            if unload_dit:
                local_cmd.append("--unload-dit")
            if color_fix:
                local_cmd.append("--color-fix")
            local_cmd.append(output_folder)  # Positional arg (must stay last)
            return local_cmd

        def _run_command(cmd_to_run: List[str]) -> tuple[int, str, bool]:
            import queue as _queue
            import threading as _threading

            proc = subprocess.Popen(
                cmd_to_run,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=base_dir,
                env=proc_env,
                creationflags=creationflags,
                preexec_fn=preexec_fn,
            )
            if process_handle is not None:
                process_handle["proc"] = proc

            output_lines: List[str] = []
            line_queue: "_queue.Queue[Optional[str]]" = _queue.Queue()

            def _reader_thread() -> None:
                try:
                    if proc.stdout is None:
                        return
                    token: List[str] = []
                    while True:
                        ch = proc.stdout.read(1)
                        if ch == "":
                            break
                        if ch in ("\n", "\r"):
                            line = "".join(token).strip()
                            token = []
                            if line:
                                line_queue.put(line)
                        else:
                            token.append(ch)
                    tail = "".join(token).strip()
                    if tail:
                        line_queue.put(tail)
                except Exception:
                    pass
                finally:
                    line_queue.put(None)

            reader = _threading.Thread(target=_reader_thread, daemon=True)
            reader.start()
            attempt_started = time.time()
            last_activity_ts = attempt_started
            heartbeat_every_sec = 10.0

            while True:
                if cancel_event and cancel_event.is_set():
                    log("Cancellation requested - terminating FlashVSR+ process")
                    try:
                        if platform.system() == "Windows":
                            proc.terminate()
                        else:
                            proc.kill()
                        proc.wait(timeout=5.0)
                    except Exception:
                        pass
                    if process_handle is not None:
                        process_handle["proc"] = None
                    return 1, "\n".join(output_lines), True

                try:
                    item = line_queue.get(timeout=0.5)
                    if item is None:
                        if proc.poll() is not None:
                            break
                    else:
                        output_lines.append(item)
                        log(item)
                        last_activity_ts = time.time()
                except _queue.Empty:
                    if proc.poll() is not None:
                        break
                    now = time.time()
                    if now - last_activity_ts >= heartbeat_every_sec:
                        elapsed = int(now - attempt_started)
                        hb = f"[FlashVSR] still processing... {elapsed}s elapsed"
                        output_lines.append(hb)
                        log(hb)
                        last_activity_ts = now

            while True:
                try:
                    item = line_queue.get_nowait()
                except _queue.Empty:
                    break
                if item:
                    output_lines.append(item)
                    log(item)

            returncode_local = proc.wait()

            if process_handle is not None:
                process_handle["proc"] = None
            return returncode_local, "\n".join(output_lines), False

        def _discover_output_path() -> Optional[str]:
            if expected_output_path.exists():
                return str(expected_output_path)
            post_mp4 = {p.resolve() for p in output_folder_path.glob("*.mp4")}
            new_files = list(post_mp4 - pre_existing_mp4)
            if new_files:
                newest = max(new_files, key=lambda p: p.stat().st_mtime)
                return str(newest)
            if post_mp4:
                newest = max(post_mp4, key=lambda p: p.stat().st_mtime)
                return str(newest)
            return None

        log(f"Running FlashVSR+ with scale={scale}, mode={mode}, version={version_ui}")
        if original_input_path != effective_input_path:
            log(f"Preprocessed input: {effective_input_path} (original: {original_input_path})")

        attempts: List[Dict[str, Any]] = [{"dtype": str(dtype), "tiled_dit": bool(tiled_dit)}]
        attempted_signatures = {(str(dtype).lower(), bool(tiled_dit))}
        returncode = 1
        output_path: Optional[str] = None
        attempt_idx = 0

        while attempt_idx < len(attempts):
            attempt = attempts[attempt_idx]
            run_dtype = str(attempt.get("dtype") or "bf16")
            run_tiled_dit = bool(attempt.get("tiled_dit", tiled_dit))
            cmd = _build_cmd(run_dtype, run_tiled_dit)

            if attempt_idx > 0:
                log(
                    "[FlashVSR] Retrying with adjusted runtime settings: "
                    f"dtype={run_dtype}, tiled_dit={run_tiled_dit}"
                )
            log(f"Command: {' '.join(cmd)}")

            returncode, attempt_blob, was_canceled = _run_command(cmd)
            if was_canceled:
                result = FlashVSRResult(
                    returncode=1,
                    output_path=None,
                    log="\n".join(log_lines + ["[Cancelled by user]"]),
                )
                return result

            output_path = _discover_output_path()
            if output_path:
                break

            queued_retry = False
            if (
                returncode != 0
                and run_tiled_dit
                and mode != "tiny-long"
                and _is_known_tiled_dit_temp_name_bug(attempt_blob)
            ):
                log(
                    "[FlashVSR] Known upstream tiled_dit bug detected for non-tiny-long mode; "
                    "retrying with tiled_dit disabled."
                )
                sig = (run_dtype.lower(), False)
                if sig not in attempted_signatures:
                    attempts.append({"dtype": run_dtype, "tiled_dit": False})
                    attempted_signatures.add(sig)
                    queued_retry = True

            if (
                not queued_retry
                and returncode != 0
                and run_dtype.lower() == "bf16"
                and _is_bf16_runtime_issue(attempt_blob)
            ):
                log("[FlashVSR] bf16 runtime issue detected; retrying once with dtype=fp16.")
                sig = ("fp16", run_tiled_dit)
                if sig not in attempted_signatures:
                    attempts.append({"dtype": "fp16", "tiled_dit": run_tiled_dit})
                    attempted_signatures.add(sig)
                    queued_retry = True

            attempt_idx += 1
            if not queued_retry:
                break

        if output_path:
            # Optional: rename/move to an explicit file path override.
            if explicit_output_file:
                try:
                    import shutil
                    desired = explicit_output_file
                    desired.parent.mkdir(parents=True, exist_ok=True)
                    if Path(output_path).resolve() != desired.resolve():
                        shutil.move(output_path, desired)
                    output_path = str(desired)
                    log(f"✅ Output renamed to: {output_path}")
                except Exception as exc:
                    log(f"⚠️ Failed to rename output to override path: {exc}")

            log(f"✅ Output saved: {output_path}")
            
            effective_returncode = int(returncode)
            if effective_returncode != 0:
                log(
                    f"FlashVSR+ exited with code {effective_returncode} after producing output; "
                    "treating run as successful."
                )
                effective_returncode = 0

            result = FlashVSRResult(
                returncode=effective_returncode,
                output_path=output_path,
                log="\n".join(log_lines),
                input_fps=get_media_fps(original_input_path) or 30.0,
                output_fps=fps
            )
        else:
            log("❌ No output file generated")
            result = FlashVSRResult(
                returncode=int(returncode or 1),
                output_path=None,
                log="\n".join(log_lines)
            )
            
    except Exception as e:
        error_msg = f"FlashVSR+ error: {str(e)}"
        log_lines.append(error_msg)
        result = FlashVSRResult(
            returncode=1,
            output_path=None,
            log="\n".join(log_lines)
        )
    
    finally:
        if temp_input_dir:
            try:
                shutil.rmtree(temp_input_dir, ignore_errors=True)
            except Exception:
                pass
        # Log command to executed_commands folder
        execution_time = time.time() - start_time
        try:
            command_logger = get_command_logger(base_dir.parent / "executed_commands")
            
            command_logger.log_command(
                tab_name="flashvsr",
                command=cmd if cmd else ["flashvsr_run.py", "--input", settings.get("input_path", "unknown")],
                settings=settings,
                returncode=result.returncode if result else -1,
                output_path=result.output_path if result else None,
                error_logs=log_lines[-50:] if result and result.returncode != 0 else None,
                execution_time=execution_time,
                additional_info={
                    "scale": settings.get("scale", "unknown"),
                    "version": settings.get("version", "unknown"),
                    "mode": settings.get("mode", "unknown")
                }
            )
            log("✅ Command logged to executed_commands folder")
        except Exception as e:
            log(f"⚠️ Failed to log command: {e}")

    return result if result else FlashVSRResult(returncode=1, output_path=None, log="\n".join(log_lines))


def discover_flashvsr_models(base_dir: Path) -> List[str]:
    """
    Discover available FlashVSR+ models.
    
    Args:
        base_dir: Base directory containing FlashVSR_plus
        
    Returns:
        List of available model versions
    """
    # FlashVSR+ runtime uses these model folder names
    models_dir = base_dir / "FlashVSR_plus" / "models"
    expected = ["FlashVSR", "FlashVSR-v1.1"]

    if not models_dir.exists():
        return ["FlashVSR"]

    available = [name for name in expected if (models_dir / name).is_dir()]
    return available if available else ["FlashVSR"]

