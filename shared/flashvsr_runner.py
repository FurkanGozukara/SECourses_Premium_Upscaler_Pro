"""
FlashVSR runner backed by ComfyUI-FlashVSR_Stable CLI.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import math
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .command_logger import get_command_logger
from .models.flashvsr_meta import (
    flashvsr_internal_to_model_name,
    flashvsr_version_to_internal,
    flashvsr_version_to_ui,
)
from .path_utils import (
    IMAGE_EXTENSIONS,
    collision_safe_path,
    detect_input_type,
    get_media_dimensions,
    get_media_fps,
    normalize_path,
    resolve_output_location,
)


@dataclass
class FlashVSRResult:
    returncode: int
    output_path: Optional[str]
    log: str
    input_fps: float = 30.0
    output_fps: float = 30.0


_FLASHVSR_REQUIRED_FILES = (
    "diffusion_pytorch_model_streaming_dmd.safetensors",
    "LQ_proj_in.ckpt",
    "TCDecoder.ckpt",
)
_SINGLE_IMAGE_FAST_TARGET_PIXELS = 4_194_304  # 2048 x 2048
_SINGLE_IMAGE_REPEAT_FRAMES = 21


def _normalize_cuda_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("cuda:"):
        text = text.split(":", 1)[1].strip()
    return text


def _resolve_flashvsr_device(device_value: Any) -> tuple[str, Optional[str], Optional[str]]:
    raw = str(device_value or "").strip()
    raw_lower = raw.lower()

    if raw_lower in {"cpu", "none", "off"}:
        return "cpu", "", "[FlashVSR] GPU isolation: CPU mode (CUDA_VISIBLE_DEVICES cleared)"

    if raw_lower in {"", "auto"}:
        return "auto", None, None

    gpu_id = _normalize_cuda_token(raw)
    if gpu_id.isdigit():
        return "cuda:0", gpu_id, f"[FlashVSR] GPU isolation: CUDA_VISIBLE_DEVICES={gpu_id}, device remapped to cuda:0"

    return raw, None, None


def _resolve_python_executable(base_dir: Path) -> str:
    if os.name == "nt":
        candidate = base_dir / "venv" / "Scripts" / "python.exe"
    else:
        candidate = base_dir / "venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def _resolve_flashvsr_root(base_dir: Path) -> Path:
    preferred = base_dir / "ComfyUI-FlashVSR_Stable"
    if preferred.exists():
        return preferred
    # Compatibility fallback
    legacy = base_dir / "FlashVSR_plus"
    if legacy.exists():
        return legacy
    return preferred


def _expected_flashvsr_model_dir_name(version_internal: str) -> str:
    return "FlashVSR-v1.1" if str(version_internal) == "11" else "FlashVSR"


def _missing_required_model_files(model_dir: Path) -> List[str]:
    missing: List[str] = []
    for name in _FLASHVSR_REQUIRED_FILES:
        if not (model_dir / name).exists():
            missing.append(name)
    return missing


def _resolve_models_root(base_dir: Path, settings: Dict[str, Any]) -> Path:
    settings_models_dir = normalize_path(settings.get("models_dir") or "")
    if settings_models_dir:
        return Path(settings_models_dir)
    return _resolve_flashvsr_root(base_dir) / "models"


def _ensure_local_flashvsr_model_layout(
    models_root: Path, version_internal: str
) -> tuple[bool, str, Optional[Path]]:
    target_name = _expected_flashvsr_model_dir_name(version_internal)
    target_dir = models_root / target_name

    if not models_root.exists():
        return False, f"[FlashVSR] models root not found: {models_root}", None

    if not target_dir.exists():
        return (
            False,
            f"[FlashVSR] Model directory missing: {target_dir}",
            None,
        )

    missing = _missing_required_model_files(target_dir)
    if missing:
        return (
            False,
            f"[FlashVSR] Incomplete model directory: {target_dir}. Missing: {', '.join(missing)}",
            None,
        )

    return True, f"[FlashVSR] Using local model directory: {target_dir}", target_dir


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


def _sanitize_mode(value: Any) -> str:
    mode = str(value or "tiny").strip().lower()
    return mode if mode in {"tiny", "tiny-long", "full"} else "tiny"


def _sanitize_precision(value: Any) -> str:
    precision = str(value or "auto").strip().lower()
    return precision if precision in {"auto", "fp16", "bf16"} else "auto"


def _sanitize_attention(value: Any) -> str:
    raw = str(value or "sparse_sage_attention").strip().lower()
    mapping = {
        "sage": "sparse_sage_attention",
        "sparse_sage": "sparse_sage_attention",
        "sparse_sage_attention": "sparse_sage_attention",
        "block": "block_sparse_attention",
        "block_sparse": "block_sparse_attention",
        "block_sparse_attention": "block_sparse_attention",
        "flash_attention_2": "flash_attention_2",
        "flash_attn_2": "flash_attention_2",
        "sdpa": "sdpa",
    }
    return mapping.get(raw, "sparse_sage_attention")


def _sanitize_vae_model(value: Any) -> str:
    valid = {"wan2.1", "wan2.2", "lightvae_w2.1", "tae_w2.2", "lighttae_hy1.5"}
    text = str(value or "Wan2.1").strip()
    return text if text.lower() in valid else "Wan2.1"


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


def _build_temp_video_from_frames(
    image_paths: List[Path],
    output_path: Path,
    fps: float,
) -> tuple[bool, str]:
    try:
        import cv2  # type: ignore
    except Exception as e:
        return False, f"OpenCV (cv2) is required for image/directory input conversion: {e}"

    if not image_paths:
        return False, "No image frames found to build temporary video input."

    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        return False, f"Failed to read first image frame: {image_paths[0]}"
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, max(1.0, float(fps or 30.0)), (w, h))
    if not writer.isOpened():
        return False, f"Failed to create temporary video writer: {output_path}"

    try:
        for frame_path in image_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()

    if not output_path.exists():
        return False, f"Temporary video was not created: {output_path}"
    return True, ""


def _extract_single_image_from_video(
    video_path: Path,
    image_format: str = "png",
    image_quality: int = 95,
) -> tuple[Optional[str], Optional[str]]:
    """
    Extract a high-quality representative frame from a single-image FlashVSR run.
    For multi-frame outputs, use pixel-wise median across the decoded frames.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:
        return None, f"OpenCV/numpy are required for single-image postprocess: {e}"

    if not video_path.exists():
        return None, f"Output video not found for single-image extraction: {video_path}"

    fmt = str(image_format or "png").strip().lower()
    if fmt not in {"png", "jpg", "webp"}:
        fmt = "png"
    quality = max(1, min(100, int(image_quality)))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, f"Failed to open output video for single-image extraction: {video_path}"

    frames_bgr = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is not None:
                frames_bgr.append(frame)
            if len(frames_bgr) >= 64:
                break
    finally:
        cap.release()

    if not frames_bgr:
        return None, f"No frames decoded from output video: {video_path}"

    if len(frames_bgr) == 1:
        out_frame = frames_bgr[0]
    else:
        stack = np.stack(frames_bgr, axis=0).astype(np.float32)
        out_frame = np.median(stack, axis=0).astype(np.uint8)

    ext = ".jpg" if fmt == "jpg" else f".{fmt}"
    out_path = collision_safe_path(video_path.with_name(f"{video_path.stem}_image{ext}"))
    imwrite_params: List[int] = []
    if fmt == "jpg":
        imwrite_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    elif fmt == "webp":
        imwrite_params = [int(cv2.IMWRITE_WEBP_QUALITY), quality]

    ok = cv2.imwrite(str(out_path), out_frame, imwrite_params)
    if not ok:
        return None, f"Failed to save extracted single-image output: {out_path}"
    return str(out_path), None


def _prepare_cli_input(
    input_path: str,
    fps_hint: float,
    log: Callable[[str], None],
    single_image_repeat_frames: int = 1,
) -> tuple[Optional[str], Optional[Path], Optional[str]]:
    kind = detect_input_type(input_path)
    if kind == "video":
        return input_path, None, None

    tmp_root = Path(tempfile.mkdtemp(prefix="flashvsr_cli_input_"))
    temp_video = tmp_root / "input_video.mp4"

    if kind == "image":
        repeat = max(1, int(single_image_repeat_frames or 1))
        image_frames = [Path(input_path)] * repeat
        ok, err = _build_temp_video_from_frames(image_frames, temp_video, fps_hint)
        if not ok:
            shutil.rmtree(tmp_root, ignore_errors=True)
            return None, None, err
        if repeat > 1:
            log(
                f"[FlashVSR] Converted single image to temporary video input "
                f"with {repeat} repeated frames: {temp_video}"
            )
        else:
            log(f"[FlashVSR] Converted single image to temporary video input: {temp_video}")
        return str(temp_video), tmp_root, None

    if kind == "directory":
        image_paths = [
            p
            for p in sorted(Path(input_path).iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not image_paths:
            shutil.rmtree(tmp_root, ignore_errors=True)
            return None, None, f"Input directory has no supported image files: {input_path}"
        ok, err = _build_temp_video_from_frames(image_paths, temp_video, fps_hint)
        if not ok:
            shutil.rmtree(tmp_root, ignore_errors=True)
            return None, None, err
        log(f"[FlashVSR] Converted image directory to temporary video input: {temp_video}")
        return str(temp_video), tmp_root, None

    shutil.rmtree(tmp_root, ignore_errors=True)
    return None, None, f"Unsupported FlashVSR input type: {kind}"


def run_flashvsr(
    settings: Dict[str, Any],
    base_dir: Path,
    on_progress: Optional[Callable[[str], None]] = None,
    cancel_event=None,
    process_handle: Optional[Dict] = None,
) -> FlashVSRResult:
    start_time = time.time()
    log_lines: List[str] = []
    cmd: List[str] = []
    temp_input_dir: Optional[Path] = None

    def log(msg: str):
        text = str(msg)
        log_lines.append(text)
        try:
            print(text, flush=True)
        except Exception:
            pass
        if on_progress:
            try:
                on_progress(text + "\n")
            except Exception:
                pass

    result: Optional[FlashVSRResult] = None

    try:
        original_input_path = normalize_path(settings.get("input_path", ""))
        effective_input_path = normalize_path(settings.get("_effective_input_path") or original_input_path)
        original_input_kind = detect_input_type(original_input_path or effective_input_path)
        if not effective_input_path or not Path(effective_input_path).exists():
            return FlashVSRResult(returncode=1, output_path=None, log="Invalid input path")

        fps_hint = _parse_float(settings.get("fps"), 30.0)

        scale = 2 if str(settings.get("scale", "2")).strip() == "2" else 4
        raw_version = str(settings.get("version", "1.1"))
        version_internal = flashvsr_version_to_internal(raw_version)
        version_ui = flashvsr_version_to_ui(raw_version)
        model_name = flashvsr_internal_to_model_name(version_internal)
        mode = _sanitize_mode(settings.get("mode", "tiny"))
        precision = _sanitize_precision(settings.get("precision", settings.get("dtype", "auto")))
        vae_model = _sanitize_vae_model(settings.get("vae_model", "Wan2.1"))
        attention_mode = _sanitize_attention(settings.get("attention_mode", settings.get("attention", "sparse_sage_attention")))
        device_arg, visible_gpu, gpu_note = _resolve_flashvsr_device(settings.get("device", "auto"))

        color_fix = _bool(settings.get("color_fix", True), default=True)
        tiled_vae = _bool(settings.get("tiled_vae", True), default=True)
        tiled_dit = _bool(settings.get("tiled_dit", True), default=True)
        unload_dit = _bool(settings.get("unload_dit", False), default=False)
        stream_decode = _bool(settings.get("stream_decode", False), default=False)
        keep_models_on_cpu = _bool(settings.get("keep_models_on_cpu", True), default=True)
        force_offload = _bool(settings.get("force_offload", True), default=True)
        enable_debug = _bool(settings.get("enable_debug", False), default=False)

        tile_size = max(32, min(1024, _parse_int(settings.get("tile_size"), 256)))
        overlap = max(8, min(512, _parse_int(settings.get("overlap", settings.get("tile_overlap")), 24)))
        if tiled_dit and tile_size < 128:
            log(
                f"[FlashVSR] tile_size={tile_size} is too small for stable DiT tiling. "
                "Using tile_size=128."
            )
            tile_size = 128
        if overlap >= tile_size:
            overlap = max(8, tile_size - 8)

        sparse_ratio = max(1.5, min(2.0, _parse_float(settings.get("sparse_ratio"), 2.0)))
        kv_ratio = max(1.0, min(3.0, _parse_float(settings.get("kv_ratio"), 3.0)))
        local_range = 9 if _parse_int(settings.get("local_range"), 11) == 9 else 11
        frame_chunk_size = max(0, _parse_int(settings.get("frame_chunk_size"), 0))
        resize_factor = max(0.1, min(1.0, _parse_float(settings.get("resize_factor"), 1.0)))
        seed = max(0, _parse_int(settings.get("seed"), 0))
        start_frame = max(0, _parse_int(settings.get("start_frame"), 0))
        end_frame = _parse_int(settings.get("end_frame"), -1)
        codec = str(settings.get("codec", "libx264") or "libx264").strip()
        crf = max(0, min(51, _parse_int(settings.get("crf"), 18)))
        fps = _parse_float(settings.get("fps"), 0.0)
        single_image_profile_applied = False
        single_image_repeat_frames = 1

        if original_input_kind == "image":
            single_image_profile_applied = True
            single_image_repeat_frames = _SINGLE_IMAGE_REPEAT_FRAMES
            log("[FlashVSR] Single-image input detected. Applying automatic one-frame profile.")
            frame_chunk_size = 0
            start_frame = 0
            end_frame = -1
            if not unload_dit:
                unload_dit = True
                log("[FlashVSR] Single-image profile: enabling unload_dit to reduce peak VRAM.")

            if mode == "tiny-long":
                mode = "tiny"
                log("[FlashVSR] Single-image profile: switching mode tiny-long -> tiny for one-frame throughput.")

            target_pixels: Optional[int] = None
            try:
                dim_probe_path = effective_input_path or original_input_path
                dims = get_media_dimensions(dim_probe_path)
                if dims:
                    in_w, in_h = int(dims[0]), int(dims[1])
                    target_pixels = max(1, in_w * scale) * max(1, in_h * scale)
            except Exception:
                target_pixels = None

            if mode in {"tiny", "tiny-long"}:
                if target_pixels is not None and target_pixels <= _SINGLE_IMAGE_FAST_TARGET_PIXELS:
                    if tiled_dit:
                        log("[FlashVSR] Single-image fast path: disabling DiT tiling for better GPU utilization.")
                    tiled_dit = False
                    stream_decode = True
                else:
                    if target_pixels is not None:
                        log(
                            "[FlashVSR] Single-image profile: target is large; keeping DiT tiling "
                            "to avoid OOM."
                        )
                    stream_decode = False
                    tiled_dit = True

        prepared_input_path, tmp_root, prep_err = _prepare_cli_input(
            effective_input_path,
            fps_hint,
            log,
            single_image_repeat_frames=single_image_repeat_frames,
        )
        if prep_err:
            return FlashVSRResult(returncode=1, output_path=None, log=prep_err)
        if not prepared_input_path:
            return FlashVSRResult(returncode=1, output_path=None, log="Failed to prepare input for FlashVSR CLI.")
        temp_input_dir = tmp_root

        models_root = _resolve_models_root(base_dir, settings)
        layout_ok, layout_msg, _ = _ensure_local_flashvsr_model_layout(models_root, version_internal)
        log(layout_msg)
        if not layout_ok:
            return FlashVSRResult(returncode=1, output_path=None, log="\n".join(log_lines))

        output_override = settings.get("output_override", "")
        explicit_output_file: Optional[Path] = None
        if output_override:
            override_path = Path(normalize_path(output_override))
            if override_path.suffix:
                explicit_output_file = collision_safe_path(override_path)
            else:
                base_stem = Path(original_input_path).stem if original_input_path else "FlashVSR"
                explicit_output_file = collision_safe_path(
                    Path(override_path) / f"FlashVSR_{mode}_{base_stem}_{seed}.mp4"
                )
        else:
            resolved = resolve_output_location(
                input_path=original_input_path or prepared_input_path,
                output_format="mp4",
                global_output_dir=settings.get("global_output_dir", str(base_dir / "outputs")),
                batch_mode=False,
                original_filename=settings.get("_original_filename"),
            )
            explicit_output_file = collision_safe_path(
                resolved if resolved.suffix else (resolved / "FlashVSR_output.mp4")
            )

        explicit_output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file = explicit_output_file

        flashvsr_root = _resolve_flashvsr_root(base_dir)
        flashvsr_script = flashvsr_root / "cli_main.py"
        if not flashvsr_script.exists():
            return FlashVSRResult(
                returncode=1,
                output_path=None,
                log=f"FlashVSR stable CLI not found at {flashvsr_script}",
            )

        python_exe = _resolve_python_executable(base_dir)
        if python_exe != sys.executable:
            log(f"[FlashVSR] Using venv python: {python_exe}")
        if gpu_note:
            log(gpu_note)

        if stream_decode:
            if mode in {"tiny", "tiny-long"}:
                if tiled_dit:
                    log(
                        "[FlashVSR] stream_decode enabled. "
                        "Disabling DiT tiling to activate CLI streaming decode path."
                    )
                    tiled_dit = False
            else:
                log("[FlashVSR] stream_decode is only supported in tiny/tiny-long modes; ignoring for full mode.")

        dit_tiling_lines: List[str] = []
        if tiled_dit:
            dims_for_tiling = get_media_dimensions(prepared_input_path)
            if dims_for_tiling:
                in_w, in_h = int(dims_for_tiling[0]), int(dims_for_tiling[1])
                stride = max(1, int(tile_size - overlap))
                rows = max(1, int(math.ceil((in_h - overlap) / float(stride))))
                cols = max(1, int(math.ceil((in_w - overlap) / float(stride))))
                total_tiles = int(rows * cols)
                dit_tiling_lines = [
                    "[FlashVSR] DiT tiling math (applies to both image and video):",
                    f"  - tile_size={tile_size}",
                    f"  - overlap={overlap}",
                    (
                        f"  - stride={stride}  "
                        "(stride = tile_size - overlap; distance between neighboring tile starts)"
                    ),
                    f"  - input_for_tiling={in_w}x{in_h} (before upscale)",
                    f"  - rows=ceil((H-overlap)/stride)=ceil(({in_h}-{overlap})/{stride})={rows}",
                    f"  - cols=ceil((W-overlap)/stride)=ceil(({in_w}-{overlap})/{stride})={cols}",
                    f"  - total_tiles=rows*cols={rows}*{cols}={total_tiles}",
                    "  - Note: tile count uses input size at DiT stage, not final output size.",
                ]
                for line in dit_tiling_lines:
                    log(line)

        def _build_cmd(run_precision: str) -> List[str]:
            local_cmd = [
                python_exe,
                "-u",
                str(flashvsr_script),
                "--input",
                prepared_input_path,
                "--output",
                str(output_file),
                "--model",
                model_name,
                "--mode",
                mode,
                "--vae_model",
                vae_model,
                "--precision",
                run_precision,
                "--device",
                device_arg,
                "--attention_mode",
                attention_mode,
                "--scale",
                str(scale),
                "--tile_size",
                str(tile_size),
                "--tile_overlap",
                str(overlap),
                "--sparse_ratio",
                str(sparse_ratio),
                "--kv_ratio",
                str(kv_ratio),
                "--local_range",
                str(local_range),
                "--seed",
                str(seed),
                "--frame_chunk_size",
                str(frame_chunk_size),
                "--resize_factor",
                str(resize_factor),
                "--codec",
                codec,
                "--crf",
                str(crf),
                "--start_frame",
                str(start_frame),
                "--end_frame",
                str(end_frame),
                "--models_dir",
                str(models_root),
            ]
            if fps > 0:
                local_cmd.extend(["--fps", str(fps)])
            local_cmd.append("--color_fix" if color_fix else "--no_color_fix")
            if tiled_vae:
                local_cmd.append("--tiled_vae")
            if tiled_dit:
                local_cmd.append("--tiled_dit")
            if unload_dit:
                local_cmd.append("--unload_dit")
            if enable_debug:
                local_cmd.append("--enable_debug")
            local_cmd.append("--keep_models_on_cpu" if keep_models_on_cpu else "--no_keep_models_on_cpu")
            local_cmd.append("--force_offload" if force_offload else "--no_force_offload")
            return local_cmd

        def _run_command(cmd_to_run: List[str]) -> tuple[int, str, bool]:
            import queue as _queue
            import threading as _threading

            proc_env = {
                **os.environ,
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
            }
            legacy_alloc_conf = proc_env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            if legacy_alloc_conf and not proc_env.get("PYTORCH_ALLOC_CONF"):
                proc_env["PYTORCH_ALLOC_CONF"] = legacy_alloc_conf
                log("[FlashVSR] Migrated PYTORCH_CUDA_ALLOC_CONF -> PYTORCH_ALLOC_CONF")
            if visible_gpu is not None:
                proc_env["CUDA_VISIBLE_DEVICES"] = visible_gpu
            triton_cache_dir = base_dir / "temp" / "triton_cache"
            with suppress(Exception):
                triton_cache_dir.mkdir(parents=True, exist_ok=True)
            proc_env.setdefault("TRITON_CACHE_DIR", str(triton_cache_dir))

            creationflags = 0
            preexec_fn = None
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                preexec_fn = os.setsid

            proc = subprocess.Popen(
                cmd_to_run,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=flashvsr_root,
                env=proc_env,
                creationflags=creationflags,
                preexec_fn=preexec_fn,
            )
            if process_handle is not None:
                process_handle["proc"] = proc

            output_lines: List[str] = []
            line_queue: "_queue.Queue[Optional[str]]" = _queue.Queue()
            last_progress_hint: str = ""
            preflight_tiling_emitted = False

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
                    log("Cancellation requested - terminating FlashVSR process")
                    try:
                        proc.terminate()
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
                        try:
                            text = str(item)
                            if (
                                (not preflight_tiling_emitted)
                                and dit_tiling_lines
                                and ("PRE-FLIGHT RESOURCE CHECK" in text.upper())
                            ):
                                preflight_tiling_emitted = True
                                for math_line in dit_tiling_lines:
                                    output_lines.append(math_line)
                                    log(math_line)
                            if "%" in text or "Processed:" in text or "Processing:" in text:
                                last_progress_hint = text
                        except Exception:
                            pass
                        last_activity_ts = time.time()
                except _queue.Empty:
                    if proc.poll() is not None:
                        break
                    now = time.time()
                    if now - last_activity_ts >= heartbeat_every_sec:
                        elapsed = int(now - attempt_started)
                        if last_progress_hint:
                            hb = f"[FlashVSR] processing... {elapsed}s elapsed | {last_progress_hint[:140]}"
                        else:
                            hb = f"[FlashVSR] processing... {elapsed}s elapsed (waiting for progress output)"
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

        attempts: List[str] = [precision]
        if precision == "bf16":
            attempts.append("fp16")

        output_path: Optional[str] = None
        returncode = 1
        for idx, run_precision in enumerate(attempts):
            cmd = _build_cmd(run_precision)
            if idx > 0:
                log(f"[FlashVSR] Retrying with precision={run_precision}")
            log(
                f"Running FlashVSR Stable with scale={scale}x, model={model_name}, "
                f"mode={mode}, vae={vae_model}, precision={run_precision}, version={version_ui}"
            )
            log(f"Command: {' '.join(cmd)}")
            returncode, attempt_blob, was_canceled = _run_command(cmd)
            if was_canceled:
                return FlashVSRResult(
                    returncode=1,
                    output_path=None,
                    log="\n".join(log_lines + ["[Cancelled by user]"]),
                )
            if output_file.exists():
                output_path = str(output_file)
                break
            if returncode != 0 and run_precision == "bf16" and _is_bf16_runtime_issue(attempt_blob):
                continue
            break

        if output_path:
            if returncode != 0:
                log(
                    f"FlashVSR exited with code {returncode} after producing output; "
                    "treating run as successful."
                )
                returncode = 0
            log(f"Output saved: {output_path}")
            result = FlashVSRResult(
                returncode=int(returncode),
                output_path=output_path,
                log="\n".join(log_lines),
                input_fps=get_media_fps(original_input_path or prepared_input_path) or 30.0,
                output_fps=(fps if fps > 0 else (get_media_fps(prepared_input_path) or 30.0)),
            )
            if original_input_kind == "image" and output_path:
                fmt_pref = str(settings.get("image_output_format", "png") or "png").strip().lower()
                quality_pref = max(1, min(100, _parse_int(settings.get("image_output_quality"), 95)))
                img_path, img_err = _extract_single_image_from_video(
                    Path(output_path),
                    image_format=fmt_pref,
                    image_quality=quality_pref,
                )
                if img_path:
                    log(f"[FlashVSR] Single-image input detected. Exported image output: {img_path}")
                    result.output_path = str(img_path)
                elif img_err:
                    log(f"[FlashVSR] Single-image export skipped: {img_err}")
        else:
            log("No output file generated")
            result = FlashVSRResult(
                returncode=int(returncode or 1),
                output_path=None,
                log="\n".join(log_lines),
            )

    except Exception as e:
        log_lines.append(f"FlashVSR error: {e}")
        result = FlashVSRResult(returncode=1, output_path=None, log="\n".join(log_lines))

    finally:
        if temp_input_dir:
            with suppress(Exception):
                shutil.rmtree(temp_input_dir, ignore_errors=True)

        execution_time = time.time() - start_time
        if cmd:
            try:
                command_logger = get_command_logger(base_dir / "executed_commands")
                command_logger.log_command(
                    tab_name="flashvsr",
                    command=cmd,
                    settings=settings,
                    returncode=result.returncode if result else -1,
                    output_path=result.output_path if result else None,
                    error_logs=log_lines[-50:] if result and result.returncode != 0 else None,
                    execution_time=execution_time,
                    additional_info={
                        "scale": scale if "scale" in locals() else settings.get("scale", "unknown"),
                        "version": version_ui if "version_ui" in locals() else settings.get("version", "unknown"),
                        "mode": mode if "mode" in locals() else settings.get("mode", "unknown"),
                        "vae_model": vae_model if "vae_model" in locals() else settings.get("vae_model", "unknown"),
                        "stream_decode": bool(stream_decode) if "stream_decode" in locals() else bool(settings.get("stream_decode", False)),
                        "single_image_profile": bool(single_image_profile_applied) if "single_image_profile_applied" in locals() else False,
                        "single_image_repeat_frames": int(single_image_repeat_frames) if "single_image_repeat_frames" in locals() else 1,
                    },
                )
                log_lines.append("Command logged to executed_commands folder")
            except Exception as e:
                log_lines.append(f"Failed to log command: {e}")

    return result if result else FlashVSRResult(returncode=1, output_path=None, log="\n".join(log_lines))


def discover_flashvsr_models(base_dir: Path) -> List[str]:
    models_dir = _resolve_flashvsr_root(base_dir) / "models"
    expected = ["FlashVSR", "FlashVSR-v1.1"]
    if not models_dir.exists():
        return ["FlashVSR-v1.1"]
    available = [name for name in expected if (models_dir / name).is_dir()]
    return available if available else ["FlashVSR-v1.1"]
