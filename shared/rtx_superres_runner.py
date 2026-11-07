"""
RTX Super Resolution runner backed by NVIDIA Maxine nvvfx Python bindings.
"""

from __future__ import annotations

import json
import os
import platform
import queue
import re
import shutil
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from shared.image_io import imwrite, load_image_bgr8
from shared.vram_autotune.rtx import check_or_pick as rtx_check_or_pick
from shared.path_utils import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    collision_safe_dir,
    collision_safe_path,
    detect_input_type,
    get_media_dimensions,
    get_media_fps,
    list_files_sorted,
    normalize_path,
    resolve_output_location,
)


@dataclass
class RTXSuperResResult:
    returncode: int
    output_path: Optional[str]
    log: str
    input_fps: float = 30.0
    output_fps: float = 30.0
    frames_processed: int = 0
    total_frames: int = 0
    elapsed_seconds: float = 0.0


NVIDIA_VFX_VERSION = "0.2.0.0"
SAME_RES_QUALITY_RE = re.compile(r"^(DENOISE_|DEBLUR_)", flags=re.IGNORECASE)


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


def _to_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _format_seconds(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    sec = int(round(seconds))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _format_finish_eta(eta_seconds: Optional[float]) -> str:
    if eta_seconds is None:
        return "ETA unknown"
    if eta_seconds <= 0:
        return "ETA 0s"
    finish_ts = time.time() + float(eta_seconds)
    finish_local = time.strftime("%H:%M:%S", time.localtime(finish_ts))
    return f"ETA {_format_seconds(eta_seconds)} (finish ~{finish_local})"


def _sorted_image_files(folder: Path) -> List[Path]:
    return list_files_sorted(folder, IMAGE_EXTENSIONS)


def _make_image_output_path(
    *,
    input_path: str,
    image_format: str,
    global_output_dir: Optional[str],
    output_override: Optional[str],
    original_filename: Optional[str] = None,
    keep_name: bool = False,
) -> Path:
    fmt = str(image_format or "png").strip().lower()
    if fmt not in {"png", "jpg", "webp"}:
        fmt = "png"
    ext = ".jpg" if fmt == "jpg" else f".{fmt}"

    override_raw = str(output_override or "").strip()
    if override_raw:
        override_path = Path(normalize_path(override_raw))
        # Batch: the per-input folder of `my.photo.jpg` is `my.photo/`, a folder, not a file name.
        if override_path.suffix and not (keep_name and override_path.is_dir()):
            if override_path.suffix.lower() != ext:
                override_path = override_path.with_suffix(ext)
            return collision_safe_path(override_path)
        base_name = Path(original_filename or Path(input_path).name).stem or "image"
        return collision_safe_path(override_path / f"{base_name}{ext}")

    default_png = resolve_output_location(
        input_path=input_path,
        output_format="png",
        global_output_dir=global_output_dir,
        batch_mode=False,
        original_filename=original_filename,
    )
    target = default_png if default_png.suffix else default_png / f"{Path(input_path).stem}.png"
    if target.suffix.lower() != ext:
        target = target.with_suffix(ext)
    return collision_safe_path(target)


def _make_video_output_path(
    *,
    input_path: str,
    global_output_dir: Optional[str],
    output_override: Optional[str],
    original_filename: Optional[str] = None,
    keep_name: bool = False,
) -> Path:
    override_raw = str(output_override or "").strip()
    if override_raw:
        override_path = Path(normalize_path(override_raw))
        # Batch: the per-input folder of `my.clip.mp4` is `my.clip/`, a folder, not a file name.
        if override_path.suffix and not (keep_name and override_path.is_dir()):
            if override_path.suffix.lower() != ".mp4":
                override_path = override_path.with_suffix(".mp4")
            return collision_safe_path(override_path)
        base_stem = Path(original_filename or Path(input_path).name).stem or "video"
        return collision_safe_path(override_path / f"{base_stem}.mp4")

    resolved = resolve_output_location(
        input_path=input_path,
        output_format="mp4",
        global_output_dir=global_output_dir,
        batch_mode=False,
        original_filename=original_filename,
    )
    target = resolved if resolved.suffix else (resolved / f"{Path(input_path).stem}.mp4")
    if target.suffix.lower() != ".mp4":
        target = target.with_suffix(".mp4")
    return collision_safe_path(target)


def _same_dir(a: Any, b: Any) -> bool:
    if not a or not b:
        return False
    try:
        return Path(normalize_path(str(a))).resolve() == Path(normalize_path(str(b))).resolve()
    except Exception:
        return False


def _make_sequence_output_dir(
    *,
    input_path: str,
    global_output_dir: Optional[str],
    output_override: Optional[str],
    original_filename: Optional[str] = None,
    keep_name: bool = False,
) -> Path:
    override_raw = str(output_override or "").strip()
    if override_raw:
        override_path = Path(normalize_path(override_raw))
        if override_path.suffix and not (keep_name and override_path.is_dir()):
            return collision_safe_dir(override_path.parent / override_path.stem)
        if keep_name:
            # Batch: the per-input folder already exists, so the frames go into a folder named
            # after the input inside it (clip/clip/) instead of a numbered sibling (clip_0001/).
            base_stem = Path(original_filename or Path(input_path).name).stem or "frames"
            return collision_safe_dir(override_path / base_stem)
        return collision_safe_dir(override_path)

    resolved = resolve_output_location(
        input_path=input_path,
        output_format="png",
        global_output_dir=global_output_dir,
        batch_mode=False,
        original_filename=original_filename,
    )
    return collision_safe_dir(resolved if not resolved.suffix else (resolved.parent / resolved.stem))


def _build_dimensions_plan(
    *,
    input_width: int,
    input_height: int,
    upscale_factor: float,
    max_edge: int,
    pre_downscale_then_upscale: bool,
    quality_name: str,
) -> Dict[str, Any]:
    in_w = max(1, int(input_width))
    in_h = max(1, int(input_height))
    requested_scale = max(1.0, float(upscale_factor or 1.0))
    max_edge_val = max(0, int(max_edge or 0))
    long_side = max(in_w, in_h)

    cap_ratio = 1.0
    if max_edge_val > 0:
        requested_long = long_side * requested_scale
        if requested_long > float(max_edge_val):
            cap_ratio = max(0.01, float(max_edge_val) / float(requested_long))

    same_res_mode = bool(SAME_RES_QUALITY_RE.match(str(quality_name or "")))
    if same_res_mode:
        # DENOISE/DEBLUR modes are same-resolution modes.
        cap_ratio = 1.0
        requested_scale = 1.0

    effective_scale = requested_scale * cap_ratio
    # Cap must always be enforced:
    # - ON: pre-downscale input first, then run model at requested scale.
    # - OFF: keep input as-is, reduce model scale to effective scale.
    use_pre_down = bool(
        pre_downscale_then_upscale
        and cap_ratio < 0.999999
        and requested_scale <= 4.0
    )
    preprocess_scale = cap_ratio if use_pre_down else 1.0
    preprocess_w = max(1, int(round(in_w * preprocess_scale)))
    preprocess_h = max(1, int(round(in_h * preprocess_scale)))

    out_w = max(1, int(round(in_w * effective_scale)))
    out_h = max(1, int(round(in_h * effective_scale)))

    # Keep encoded video dimensions codec-friendly.
    out_w = max(2, int(out_w))
    out_h = max(2, int(out_h))
    preprocess_w = max(1, int(preprocess_w))
    preprocess_h = max(1, int(preprocess_h))

    if out_w % 2:
        out_w -= 1
    if out_h % 2:
        out_h -= 1
    if preprocess_w % 2:
        preprocess_w -= 1
    if preprocess_h % 2:
        preprocess_h -= 1
    out_w = max(2, out_w)
    out_h = max(2, out_h)
    preprocess_w = max(2, preprocess_w)
    preprocess_h = max(2, preprocess_h)

    # The NVIDIA binding supports up to 4x per inference call. Keep the AI
    # pass inside that contract and perform only the residual resize afterward.
    # If a max-edge cap requests an output smaller than the source, preprocess
    # to the final size instead of asking VSR to downscale.
    if out_w < preprocess_w or out_h < preprocess_h:
        preprocess_w = out_w
        preprocess_h = out_h
    native_scale = max(
        float(out_w) / float(max(1, preprocess_w)),
        float(out_h) / float(max(1, preprocess_h)),
    )
    if native_scale > 4.0:
        model_out_w = min(out_w, max(2, int(round(preprocess_w * 4.0))))
        model_out_h = min(out_h, max(2, int(round(preprocess_h * 4.0))))
    else:
        model_out_w = out_w
        model_out_h = out_h
    if model_out_w % 2:
        model_out_w -= 1
    if model_out_h % 2:
        model_out_h -= 1
    model_out_w = max(2, model_out_w)
    model_out_h = max(2, model_out_h)
    model_scale = max(
        float(model_out_w) / float(max(1, preprocess_w)),
        float(model_out_h) / float(max(1, preprocess_h)),
    )

    return {
        "input_width": in_w,
        "input_height": in_h,
        "requested_scale": float(upscale_factor or 1.0),
        "effective_scale": float(effective_scale),
        "model_scale": float(model_scale),
        "cap_ratio": float(cap_ratio),
        "same_res_mode": same_res_mode,
        "preprocess_width": preprocess_w,
        "preprocess_height": preprocess_h,
        "model_output_width": model_out_w,
        "model_output_height": model_out_h,
        "post_resize_required": bool(model_out_w != out_w or model_out_h != out_h),
        "output_width": out_w,
        "output_height": out_h,
    }


_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
# ffprobe colour-matrix tag -> swscale matrix name, for an RGB -> YUV conversion that matches the source.
_SWS_MATRIX = {
    "bt709": "bt709",
    "smpte170m": "bt601",
    "bt470bg": "bt601",
    "bt2020nc": "bt2020",
    "bt2020c": "bt2020",
    "smpte240m": "smpte240m",
    "fcc": "fcc",
}
# HDR10 mastering display: BT.2020 primaries, D65 white (x265 units of 0.00002 / 0.0001 cd/m2).
_BT2020_MASTER = "G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)"
# VFX SDK effects that only run on newer GPU generations (compute capability).
STREAMING_MIN_CAPABILITY = (8, 0)  # STREAMING_* levels: Ampere (RTX 30) or newer


def _probe_video_stream(path: str) -> Dict[str, Any]:
    """Size after rotation and colour tags of the first video stream (empty dict when unknown)."""
    exe = shutil.which("ffprobe")
    if not exe:
        return {}
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,color_space,color_primaries,color_transfer,color_range,pix_fmt"
             ":stream_side_data=rotation:stream_tags=rotate", "-of", "json", str(path)],
            capture_output=True, text=True, creationflags=_NO_WINDOW, timeout=60,
        ).stdout
        stream = (json.loads(out or "{}").get("streams") or [{}])[0]
    except Exception:
        return {}
    info: Dict[str, Any] = {k: str(stream.get(k) or "") for k in
                            ("color_space", "color_primaries", "color_transfer", "color_range", "pix_fmt")}
    try:
        w, h = int(stream.get("width") or 0), int(stream.get("height") or 0)
        rotation = 0
        for side in stream.get("side_data_list") or []:
            if "rotation" in side:
                rotation = int(float(side["rotation"]))
        if not rotation:
            rotation = int(float((stream.get("tags") or {}).get("rotate") or 0))
        if abs(rotation) % 180 == 90:
            w, h = h, w
        if w > 0 and h > 0:
            info["size"] = (w, h)
    except Exception:
        pass
    return info


def _sdr_color_args(color: Dict[str, Any]) -> List[str]:
    """Encode RGB with the source's YUV matrix and tag the output like the source.

    Frames are decoded with the source's own matrix (ffmpeg honours the tags), so the encode must use
    the same one; an untagged source keeps ffmpeg's defaults (the old behaviour)."""
    space = str(color.get("color_space") or "").lower()
    matrix = _SWS_MATRIX.get(space)
    if not matrix:
        return []
    rng = "pc" if str(color.get("color_range") or "").lower() in {"pc", "jpeg", "full"} else "tv"
    # ffmpeg 8/9 encoders take colorimetry from the frames, so label them with setparams.
    params = [f"colorspace={space}", f"range={rng}"]
    prim = str(color.get("color_primaries") or "").lower()
    trc = str(color.get("color_transfer") or "").lower()
    if prim and prim not in {"unknown", "reserved"}:
        params.append(f"color_primaries={prim}")
    if trc and trc not in {"unknown", "reserved"}:
        params.append(f"color_trc={trc}")
    return ["-vf", f"scale=out_color_matrix={matrix}:out_range={rng},setparams={':'.join(params)}"]


class _FrameReader:
    """RGB frames from an ffmpeg pipe (rgb24, or rgb48le for the 10-bit path); optional prefetch thread."""

    def __init__(self, path: str, width: int, height: int, deep: bool, prefetch: int = 0):
        self.width, self.height, self.deep = int(width), int(height), bool(deep)
        self.frame_bytes = self.width * self.height * 3 * (2 if self.deep else 1)
        cmd = [shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0",
               "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb48le" if self.deep else "rgb24", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     creationflags=_NO_WINDOW, bufsize=self.frame_bytes * 2)
        self.queue: Optional["queue.Queue"] = None
        self.thread: Optional[threading.Thread] = None
        if prefetch > 0:
            self.queue = queue.Queue(maxsize=prefetch)
            self.thread = threading.Thread(target=self._pump, name="rtx-reader", daemon=True)
            self.thread.start()

    def _read_raw(self):
        import numpy as np  # type: ignore

        buf = bytearray(self.frame_bytes)
        view = memoryview(buf)
        got = 0
        while got < self.frame_bytes:
            n = self.proc.stdout.readinto(view[got:])
            if not n:
                break
            got += n
        if got < self.frame_bytes:
            return None
        dtype = np.dtype("<u2") if self.deep else np.uint8
        return np.frombuffer(buf, dtype=dtype).reshape(self.height, self.width, 3)

    def _pump(self) -> None:
        try:
            while True:
                frame = self._read_raw()
                self.queue.put(frame)
                if frame is None:
                    break
        except Exception:
            self.queue.put(None)

    def read(self):
        if self.queue is not None:
            return self.queue.get()
        return self._read_raw()

    def close(self) -> None:
        with suppress(Exception):
            self.proc.stdout.close()
        with suppress(Exception):
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait(timeout=5)


def _pack_rgb10a2(torch, chw):
    """(3,H,W) float in [0,1] -> (H,W) uint32 R10G10B10A2 (R in the low bits)."""
    rgb = chw.clamp(0.0, 1.0).mul(1023.0).round().to(torch.int32)
    packed = rgb[0] | (rgb[1] << 10) | (rgb[2] << 20) | -1073741824
    return packed.contiguous().view(torch.uint32)


def _unpack_rgb10a2(torch, packed):
    """(H,W) uint32 R10G10B10A2 -> (3,H,W) float in [0,1]."""
    p = packed.view(torch.int32)
    return torch.stack(
        (p.bitwise_and(0x3FF), p.bitwise_right_shift(10).bitwise_and(0x3FF), p.bitwise_right_shift(20).bitwise_and(0x3FF)),
        dim=0,
    ).to(torch.float32).div_(1023.0)


def run_rtx_superres(
    settings: Dict[str, Any],
    base_dir: Path,
    on_progress: Optional[Callable[[str], None]] = None,
    cancel_event=None,
) -> RTXSuperResResult:
    start_ts = time.time()
    logs: List[str] = []
    processed = 0
    total_frames = 0
    input_fps = 30.0
    output_fps = 30.0
    inline_progress_active = False
    inline_progress_width = 0
    sr = None
    thdr = None
    cap = None
    reader: Optional[_FrameReader] = None
    video_writer = None
    ffmpeg_writer = None
    writer_queue: Optional["queue.Queue"] = None
    writer_thread: Optional[threading.Thread] = None
    writer_error: List[str] = []

    def log(msg: str) -> None:
        nonlocal inline_progress_active, inline_progress_width
        text = str(msg or "").rstrip("\r\n")
        if not text:
            return
        logs.append(text)
        is_frame_progress = text.startswith("FRAME_PROGRESS ")
        try:
            if is_frame_progress:
                padded = text
                if inline_progress_width > len(text):
                    padded = text + (" " * (inline_progress_width - len(text)))
                print(f"\r{padded}", end="", flush=True)
                inline_progress_active = True
                inline_progress_width = len(text)
            else:
                if inline_progress_active:
                    print("", flush=True)
                    inline_progress_active = False
                    inline_progress_width = 0
                print(text, flush=True)
        except Exception:
            pass
        if on_progress:
            try:
                on_progress(text + "\n")
            except Exception:
                pass

    def _fail(message: str, frames_done: int = 0) -> RTXSuperResResult:
        return RTXSuperResResult(
            returncode=1,
            output_path=None,
            log="\n".join(logs + [message]) if logs else message,
            input_fps=float(input_fps),
            output_fps=float(output_fps),
            frames_processed=int(frames_done),
            total_frames=int(total_frames),
            elapsed_seconds=max(0.0, time.time() - start_ts),
        )

    try:
        try:
            import cv2  # type: ignore
            import torch  # type: ignore
            import nvvfx  # type: ignore
            from nvvfx import VideoSuperRes  # type: ignore
        except Exception as import_err:
            return RTXSuperResResult(
                returncode=1,
                output_path=None,
                log=(
                    "RTX Super Resolution runtime could not be loaded. The app requirements install "
                    f"nvidia-vfx=={NVIDIA_VFX_VERSION} on both Windows and Linux. Re-run the installer in this "
                    f"app's venv and verify the NVIDIA driver. Platform={platform.system()}. "
                    f"Details: {import_err}"
                ),
                elapsed_seconds=max(0.0, time.time() - start_ts),
            )
        image_encoding_enum = getattr(nvvfx, "ImageEncoding", None)
        true_hdr_cls = getattr(nvvfx, "TrueHDR", None)

        input_path = normalize_path(settings.get("input_path", ""))
        if not input_path or not Path(input_path).exists():
            return RTXSuperResResult(
                returncode=1,
                output_path=None,
                log="Input path is missing or does not exist.",
                elapsed_seconds=max(0.0, time.time() - start_ts),
            )

        input_kind = detect_input_type(input_path)
        if input_kind not in {"video", "image", "directory"}:
            return RTXSuperResResult(
                returncode=1,
                output_path=None,
                log=f"Unsupported input type for RTX Super Resolution: {input_kind}",
                elapsed_seconds=max(0.0, time.time() - start_ts),
            )

        device_raw = str(settings.get("device", "0") or "0").strip().lower()
        if device_raw in {"cpu", "none", "off"}:
            return RTXSuperResResult(
                returncode=1,
                output_path=None,
                log="RTX Super Resolution requires CUDA GPU; CPU mode is not supported.",
                elapsed_seconds=max(0.0, time.time() - start_ts),
            )
        if device_raw.startswith("cuda:"):
            device_raw = device_raw.split(":", 1)[1].strip()
        if not device_raw.isdigit():
            device_raw = "0"
        device_idx = int(device_raw)

        if not torch.cuda.is_available():
            return RTXSuperResResult(
                returncode=1,
                output_path=None,
                log="CUDA is not available. RTX Super Resolution requires CUDA.",
                elapsed_seconds=max(0.0, time.time() - start_ts),
            )
        device_count = int(torch.cuda.device_count())
        if device_idx < 0 or device_idx >= device_count:
            return RTXSuperResResult(
                returncode=1,
                output_path=None,
                log=(
                    f"CUDA device {device_idx} is unavailable; detected {device_count} CUDA device(s). "
                    "Select a valid RTX GPU in Global Settings."
                ),
                elapsed_seconds=max(0.0, time.time() - start_ts),
            )
        torch.cuda.set_device(device_idx)
        props = torch.cuda.get_device_properties(device_idx)
        capability = (int(getattr(props, "major", 0)), int(getattr(props, "minor", 0)))
        if capability < (7, 5):
            return RTXSuperResResult(
                returncode=1,
                output_path=None,
                log=(
                    f"GPU '{getattr(props, 'name', device_idx)}' has CUDA capability "
                    f"{capability[0]}.{capability[1]}. nvidia-vfx requires a Tensor Core GPU "
                    "from NVIDIA Turing/RTX 2000 or newer."
                ),
                elapsed_seconds=max(0.0, time.time() - start_ts),
            )
        quality_members = VideoSuperRes.QualityLevel.__members__

        def _usable_quality(name: str) -> str:
            q = str(name or "ULTRA").strip().upper()
            if q.startswith("STREAMING_"):
                if q not in quality_members:
                    fallback = q.split("_", 1)[1]
                    log(f"[RTX] {q} needs nvidia-vfx 0.2.0.0 or newer; using {fallback}.")
                    return fallback if fallback in quality_members else "ULTRA"
                if capability < STREAMING_MIN_CAPABILITY:
                    fallback = q.split("_", 1)[1]
                    log(f"[RTX] {q} needs an RTX 30 series (Ampere) or newer GPU; using {fallback}.")
                    return fallback
            return q if q in quality_members else "ULTRA"

        quality_name = _usable_quality(settings.get("quality_preset", "ULTRA"))
        quality_enum = VideoSuperRes.QualityLevel[quality_name]

        upscale_factor = _to_float(settings.get("upscale_factor"), 2.0)
        max_edge = _to_int(settings.get("max_resolution"), 0)
        pre_down = _bool(settings.get("pre_downscale_then_upscale"), True)
        non_blocking = _bool(settings.get("non_blocking_inference"), True)
        stream_ptr = _to_int(settings.get("cuda_stream_ptr"), 0)
        strength = max(0.0, min(1.0, _to_float(settings.get("vsr_strength"), 1.0)))
        want_10bit = _bool(settings.get("rtx_10bit"), False)
        want_hdr = _bool(settings.get("rtx_hdr_output"), False)
        hdr_luminance = max(400, min(2000, _to_int(settings.get("rtx_hdr_luminance"), 1000)))
        # Reader/writer threads overlap decode and encode with the SDK: +5.6 % at 1080p -> 4K
        # (3 of 3 interleaved repeats on RTX 5090), within noise at small sizes.
        pipelined = _bool(settings.get("rtx_pipeline"), True)

        global_output_dir = settings.get("global_output_dir")
        output_override = settings.get("output_override")
        original_filename = settings.get("_original_filename")

        image_output_format = str(settings.get("image_output_format", "png") or "png").strip().lower()
        if image_output_format not in {"png", "jpg", "webp"}:
            image_output_format = "png"
        image_output_quality = max(1, min(100, _to_int(settings.get("image_output_quality"), 95)))

        output_format_setting = str(settings.get("output_format", "auto") or "auto").strip().lower()
        if output_format_setting not in {"auto", "mp4", "png"}:
            output_format_setting = "auto"

        if output_format_setting == "auto":
            if input_kind == "video":
                resolved_output_format = "mp4"
            elif input_kind == "directory":
                resolved_output_format = "png"
            else:
                resolved_output_format = image_output_format
        elif output_format_setting == "png":
            resolved_output_format = "png" if input_kind != "image" else image_output_format
        else:
            resolved_output_format = "mp4"

        dims = get_media_dimensions(input_path)
        if not dims:
            return RTXSuperResResult(
                returncode=1,
                output_path=None,
                log="Failed to detect input dimensions.",
                elapsed_seconds=max(0.0, time.time() - start_ts),
            )
        in_w, in_h = int(dims[0]), int(dims[1])
        plan = _build_dimensions_plan(
            input_width=in_w,
            input_height=in_h,
            upscale_factor=upscale_factor,
            max_edge=max_edge,
            pre_downscale_then_upscale=pre_down,
            quality_name=quality_name,
        )
        preprocess_w = int(plan["preprocess_width"])
        preprocess_h = int(plan["preprocess_height"])
        model_out_w = int(plan["model_output_width"])
        model_out_h = int(plan["model_output_height"])
        out_w = int(plan["output_width"])
        out_h = int(plan["output_height"])

        # Save VRAM: the SDK allocates outside PyTorch, so check the budget before loading it
        # (Auto Tune picks the best level that fits) instead of letting Windows spill.
        vram_ok, vram_reason = rtx_check_or_pick(
            settings, in_w=preprocess_w, in_h=preprocess_h,
            model_out_w=model_out_w, model_out_h=model_out_h, log=log,
        )
        if not vram_ok:
            log(f"[RTX] Not started: {vram_reason}")
            return RTXSuperResResult(
                returncode=1,
                output_path=None,
                log="\n".join(logs + [f"Not enough free VRAM: {vram_reason}"]),
                elapsed_seconds=max(0.0, time.time() - start_ts),
            )
        tuned_quality = str(settings.get("quality_preset") or quality_name).strip().upper()
        if tuned_quality != quality_name:
            quality_name = _usable_quality(tuned_quality)
            quality_enum = VideoSuperRes.QualityLevel[quality_name]

        if resolved_output_format == "mp4" and (out_w > 8192 or out_h > 8192):
            return RTXSuperResResult(
                returncode=1,
                output_path=None,
                log=(
                    f"Target resolution {out_w}x{out_h} exceeds 8192 for MP4 output. "
                    "Lower Upscale x / Max Resolution, or switch output format to PNG frames."
                ),
                elapsed_seconds=max(0.0, time.time() - start_ts),
            )

        if bool(plan.get("same_res_mode")):
            log(
                f"[RTX] Quality preset {quality_name} is same-resolution mode. "
                f"Output dimensions forced to {out_w}x{out_h}."
            )
        else:
            log(
                f"[RTX] Sizing plan: input {in_w}x{in_h} -> preprocess {preprocess_w}x{preprocess_h} -> output {out_w}x{out_h} "
                f"(requested {upscale_factor:g}x, effective {float(plan.get('effective_scale', 1.0)):.3f}x)."
            )
        if bool(plan.get("post_resize_required")):
            log(
                f"[RTX] Native VFX pass is capped at 4x ({model_out_w}x{model_out_h}); "
                f"a final Lanczos resize will produce the exact {out_w}x{out_h} target."
            )

        # 10-bit (RGB10A2) and HDR10 need nvidia-vfx 0.2+; HDR10 needs a video output.
        use_10bit = bool(want_10bit and image_encoding_enum is not None)
        if want_10bit and not use_10bit:
            log("[RTX] 10-bit processing needs nvidia-vfx 0.2.0.0 or newer; using 8-bit.")
        use_hdr = bool(want_hdr and true_hdr_cls is not None and resolved_output_format == "mp4")
        if want_hdr and not use_hdr:
            if true_hdr_cls is None:
                log("[RTX] HDR10 output needs nvidia-vfx 0.2.0.0 or newer; writing SDR.")
            else:
                log("[RTX] HDR10 output is written for video (MP4) output only; writing SDR frames.")
        deep_out = bool(use_10bit or use_hdr)

        output_path: Optional[str] = None
        output_dir_path: Optional[Path] = None

        if input_kind == "image":
            output_img_path = _make_image_output_path(
                input_path=input_path,
                image_format=image_output_format if resolved_output_format != "mp4" else "png",
                global_output_dir=global_output_dir,
                output_override=output_override,
                original_filename=original_filename,
                keep_name=bool(settings.get("_batch_keep_name")),
            )
            output_img_path.parent.mkdir(parents=True, exist_ok=True)
            output_path = str(output_img_path)
            total_frames = 1
            input_fps = _to_float(settings.get("fps"), 30.0)
            output_fps = input_fps
        elif resolved_output_format == "png":
            output_dir = _make_sequence_output_dir(
                input_path=input_path,
                global_output_dir=global_output_dir,
                output_override=output_override,
                original_filename=original_filename,
                # Only the batch item's own run folder (not a chunk's output folder).
                keep_name=bool(settings.get("_batch_keep_name")) and _same_dir(output_override, settings.get("_run_dir")),
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir_path = output_dir
            output_path = str(output_dir)
        else:
            output_video_path = _make_video_output_path(
                input_path=input_path,
                global_output_dir=global_output_dir,
                output_override=output_override,
                original_filename=original_filename,
                keep_name=bool(settings.get("_batch_keep_name")),
            )
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            output_path = str(output_video_path)

        input_fps = float(get_media_fps(input_path) or _to_float(settings.get("fps"), 30.0) or 30.0)
        if input_fps <= 0:
            input_fps = 30.0
        output_fps = float(_to_float(settings.get("fps"), input_fps) or input_fps)
        if output_fps <= 0:
            output_fps = input_fps

        try:
            sr = VideoSuperRes(device=device_idx, quality=quality_enum)
            if hasattr(sr, "strength") and strength < 1.0:
                sr.strength = float(strength)
            elif strength < 1.0:
                log("[RTX] VSR strength needs nvidia-vfx 0.2.0.0 or newer; using full strength.")
            if use_10bit:
                sr.image_encoding = image_encoding_enum.RGB10A2
            sr.output_width = model_out_w
            sr.output_height = model_out_h
            sr.load()
            if use_hdr:
                thdr = true_hdr_cls(luminance=int(hdr_luminance), device=device_idx)
                thdr.load()
        except Exception as load_err:
            linux_driver_hint = (
                " Linux VSR requires a supported NVIDIA driver branch (570.190+, 580.82+, or 590.44+)."
                if platform.system() == "Linux"
                else " Windows VSR requires NVIDIA driver 570.65 or newer."
            )
            return RTXSuperResResult(
                returncode=1,
                output_path=None,
                log=f"NVIDIA VFX failed to initialize on cuda:{device_idx}: {load_err}.{linux_driver_hint}",
                elapsed_seconds=max(0.0, time.time() - start_ts),
            )

        log(
            f"[RTX] Runtime ready: quality={quality_name}, strength={strength:g}, "
            f"encoding={'RGB10A2 (10-bit)' if use_10bit else 'RGB 8-bit'}"
            f"{', HDR10 output (TrueHDR ' + str(hdr_luminance) + ' nits)' if use_hdr else ''}, "
            f"pipelined={bool(pipelined)}, non_blocking={bool(non_blocking)}, device=cuda:{device_idx}"
        )
        torch_device = torch.device(f"cuda:{device_idx}")

        def run_frame(frame_rgb):
            """RGB uint8 (or uint16 when 10-bit input) at input size -> RGB uint8/uint16 at output size."""
            if frame_rgb is None:
                return None
            if preprocess_w != int(frame_rgb.shape[1]) or preprocess_h != int(frame_rgb.shape[0]):
                frame_rgb = cv2.resize(frame_rgb, (preprocess_w, preprocess_h), interpolation=cv2.INTER_AREA)
            scale_in = 65535.0 if frame_rgb.dtype.itemsize == 2 else 255.0
            if frame_rgb.dtype.itemsize == 2:
                frame_rgb = frame_rgb.astype("int32")
            src = torch.from_numpy(frame_rgb).to(torch_device, non_blocking=True)
            x = src.permute(2, 0, 1).to(torch.float32).div_(scale_in)
            if use_10bit:
                inp = _pack_rgb10a2(torch, x)
            else:
                inp = x.contiguous()
            out = sr.run(inp, non_blocking=bool(non_blocking), stream_ptr=int(stream_ptr or 0))
            if non_blocking:
                torch.cuda.synchronize(device_idx)
            y = torch.from_dlpack(out.image).clone()
            if use_10bit:
                y = _unpack_rgb10a2(torch, y)
            if use_hdr:
                hdr_res = thdr.run(y.clamp(0.0, 1.0).contiguous())
                y = _unpack_rgb10a2(torch, torch.from_dlpack(hdr_res.image).clone())
            y = y.clamp(0.0, 1.0)
            if deep_out:
                arr = y.mul(65535.0).round().to(torch.int32).permute(1, 2, 0).contiguous().cpu().numpy().astype("uint16")
            else:
                # Round, not truncate: truncation made every frame ~0.5 level darker.
                arr = y.mul(255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
            if int(arr.shape[1]) != out_w or int(arr.shape[0]) != out_h:
                interpolation = (
                    cv2.INTER_LANCZOS4
                    if out_w > int(arr.shape[1]) or out_h > int(arr.shape[0])
                    else cv2.INTER_AREA
                )
                arr = cv2.resize(arr, (out_w, out_h), interpolation=interpolation)
            return arr

        def to_bgr_for_file(frame_rgb):
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if frame.dtype.itemsize == 2 and not use_10bit:
                frame = (frame.astype("float32") / 257.0 + 0.5).astype("uint8")
            return frame

        # Build frame source: ffmpeg decodes with the source's own colour matrix (OpenCV assumes
        # BT.601 for every file, which shifted BT.709 colours by about 1.4 levels on average).
        frame_paths: List[Path] = []
        single_image = None
        src_color: Dict[str, Any] = {}
        try:
            if input_kind == "video":
                probe = _probe_video_stream(input_path)
                src_color = probe
                count_cap = cv2.VideoCapture(str(input_path))
                raw_total = int(count_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if count_cap.isOpened() else 0
                with suppress(Exception):
                    count_cap.release()
                total_frames = raw_total if raw_total > 0 else 0
                if shutil.which("ffmpeg") and probe.get("size") == (in_w, in_h):
                    reader = _FrameReader(input_path, in_w, in_h, deep=use_10bit, prefetch=8 if pipelined else 0)
                else:
                    cap = cv2.VideoCapture(str(input_path))
                    if not cap.isOpened():
                        return _fail(f"Failed to open input video: {input_path}")
                    src_color = {}
                    log("[RTX] ffmpeg/ffprobe unavailable or size mismatch: decoding with OpenCV (BT.601 colours).")
            elif input_kind == "directory":
                frame_paths = _sorted_image_files(Path(input_path))
                if not frame_paths:
                    return RTXSuperResResult(
                        returncode=1,
                        output_path=None,
                        log=f"No supported images found in frame folder: {input_path}",
                        elapsed_seconds=max(0.0, time.time() - start_ts),
                    )
                total_frames = len(frame_paths)
            else:
                single_image = load_image_bgr8(input_path)
                if single_image is None:
                    return RTXSuperResResult(
                        returncode=1,
                        output_path=None,
                        log=f"Failed to read input image: {input_path}",
                        elapsed_seconds=max(0.0, time.time() - start_ts),
                    )
                single_image = cv2.cvtColor(single_image, cv2.COLOR_BGR2RGB)
                total_frames = 1
        except Exception as io_err:
            return RTXSuperResResult(
                returncode=1,
                output_path=None,
                log=f"Input read failed: {io_err}",
                elapsed_seconds=max(0.0, time.time() - start_ts),
            )

        # Build sink: FFmpeg with the Output tab's codec settings (H.264 by default). OpenCV's MP4 writer
        # produced MPEG-4 Part 2 ("mp4v"), a lower-quality codec many players and editors handle poorly, and is
        # only used when ffmpeg is missing.
        if output_path and Path(output_path).suffix.lower() in VIDEO_EXTENSIONS:
            from .video_codec_options import build_ffmpeg_video_encode_args

            ffmpeg_exe = shutil.which("ffmpeg")
            if ffmpeg_exe:
                codec_key = str(settings.get("video_codec") or "h264").strip().lower()
                codec_key = {"libx264": "h264", "x264": "h264", "avc": "h264", "libx265": "h265", "x265": "h265",
                             "hevc": "h265", "libsvtav1": "av1", "svtav1": "av1", "libvpx-vp9": "vp9"}.get(codec_key, codec_key)
                if codec_key not in ("h264", "h265", "av1", "vp9") or (
                        codec_key == "vp9" and Path(output_path).suffix.lower() != ".webm"):
                    codec_key = "h264"  # ProRes / VP9 need .mov / .webm containers; the MP4 path stays H.264
                if use_hdr and codec_key not in ("h265", "av1"):
                    log(f"[RTX] HDR10 output uses H.265 10-bit (the selected {codec_key.upper()} has no HDR10 path here).")
                    codec_key = "h265"
                pixel_format = str(settings.get("pixel_format") or "yuv420p")
                if deep_out and "10" not in pixel_format:
                    pixel_format = "yuv420p10le"
                encode_args = build_ffmpeg_video_encode_args(
                    codec=codec_key,
                    quality=_to_int(settings.get("video_quality"), 18),
                    pixel_format=pixel_format,
                    preset=str(settings.get("video_preset") or "medium"),
                    h265_tune=str(settings.get("h265_tune") or "none"),
                    av1_film_grain=_to_int(settings.get("av1_film_grain"), 8),
                    av1_film_grain_denoise=_bool(settings.get("av1_film_grain_denoise"), False),
                    audio_codec="none",
                )
                if deep_out and "-pix_fmt" in encode_args:
                    # libx264 and SVT-AV1 encode 10-bit too; the app's profile list only names 8-bit for them.
                    encode_args[encode_args.index("-pix_fmt") + 1] = pixel_format if "10" in pixel_format else "yuv420p10le"
                if use_hdr:
                    # MaxCLL = the TrueHDR peak target (the output cannot exceed it); MaxFALL 0 = unknown.
                    max_light = int(hdr_luminance) * 10000
                    color_args = ["-vf", "scale=out_color_matrix=bt2020:out_range=tv,setparams=colorspace=bt2020nc:"
                                  "range=tv:color_primaries=bt2020:color_trc=smpte2084",
                                  "-color_primaries", "bt2020", "-color_trc", "smpte2084",
                                  "-colorspace", "bt2020nc", "-color_range", "tv"]
                    if codec_key == "h265":
                        encode_args += ["-x265-params",
                                        "hdr10=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:"
                                        f"colormatrix=bt2020nc:master-display={_BT2020_MASTER}L({max_light},50):"
                                        f"max-cll={int(hdr_luminance)},0"]
                    else:
                        # SVT-AV1 takes chromaticities and nits as decimals (x265 uses integer units).
                        svt = "-svtav1-params"
                        extra = ("mastering-display=G(0.1700,0.7970)B(0.1310,0.0460)R(0.7080,0.2920)"
                                 f"WP(0.3127,0.3290)L({int(hdr_luminance)}.0,0.0050):"
                                 f"content-light={int(hdr_luminance)},0")
                        if svt in encode_args:
                            i = encode_args.index(svt) + 1
                            encode_args[i] = encode_args[i] + ":" + extra
                        else:
                            encode_args += [svt, extra]
                else:
                    color_args = _sdr_color_args(src_color)
                ffmpeg_cmd = [
                    ffmpeg_exe, "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le" if deep_out else "rgb24",
                    "-s:v", f"{out_w}x{out_h}", "-r", f"{float(output_fps):.8f}", "-i", "-",
                    *color_args, *encode_args, "-movflags", "+faststart", str(output_path),
                ]
                ffmpeg_writer = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=_NO_WINDOW,
                )
                log(f"[RTX] Encoding {codec_key.upper()} with FFmpeg ({' '.join(encode_args[:6])})"
                    + (" as HDR10 (BT.2020 PQ)." if use_hdr else (f", colours tagged {src_color.get('color_space')}." if color_args else ".")))
            else:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(output_path), fourcc, float(output_fps), (out_w, out_h))
                if not video_writer.isOpened():
                    with suppress(Exception):
                        video_writer.release()
                    return RTXSuperResResult(
                        returncode=1,
                        output_path=None,
                        log=(
                            f"Failed to create a video writer for {output_path}: "
                            "ffmpeg was not found in PATH and OpenCV video encoding is unavailable."
                        ),
                        elapsed_seconds=max(0.0, time.time() - start_ts),
                    )
                log("[RTX] ffmpeg not found: falling back to OpenCV's MPEG-4 writer (lower quality; install ffmpeg).")

        def _write_now(frame_rgb) -> None:
            if video_writer is not None:
                video_writer.write(to_bgr_for_file(frame_rgb))
                return
            if ffmpeg_writer is not None and ffmpeg_writer.stdin is not None:
                data = frame_rgb if frame_rgb.flags["C_CONTIGUOUS"] else frame_rgb.copy()
                ffmpeg_writer.stdin.write(memoryview(data).cast("B"))
                return
            raise RuntimeError("RTX video output writer is not initialized")

        def _writer_loop() -> None:
            while True:
                item = writer_queue.get()
                if item is None:
                    break
                if writer_error:
                    continue
                try:
                    _write_now(item)
                except Exception as werr:
                    writer_error.append(str(werr))

        if pipelined and (ffmpeg_writer is not None or video_writer is not None):
            writer_queue = queue.Queue(maxsize=8)
            writer_thread = threading.Thread(target=_writer_loop, name="rtx-writer", daemon=True)
            writer_thread.start()

        def write_video_frame(frame_rgb) -> None:
            if writer_queue is not None:
                if writer_error:
                    raise RuntimeError(writer_error[0])
                writer_queue.put(frame_rgb)
            else:
                _write_now(frame_rgb)

        ema_frame_sec: Optional[float] = None
        log_eta_interval = 0.8
        last_eta_log_ts = 0.0

        def emit_progress(frame_idx: int, total: int) -> None:
            nonlocal ema_frame_sec, last_eta_log_ts
            now_ts = time.time()
            elapsed = max(1e-6, now_ts - start_ts)
            if frame_idx > 0:
                frame_sec = elapsed / float(frame_idx)
                if ema_frame_sec is None:
                    ema_frame_sec = frame_sec
                else:
                    ema_frame_sec = (ema_frame_sec * 0.7) + (frame_sec * 0.3)
            pct = (float(frame_idx) / float(total)) if total > 0 else 0.0
            eta_seconds = None
            if ema_frame_sec is not None and total > 0 and frame_idx > 0:
                eta_seconds = max(0.0, (float(total - frame_idx) * float(ema_frame_sec)))
            eta_text = _format_finish_eta(eta_seconds)
            line = (
                f"FRAME_PROGRESS {frame_idx}/{max(1, total)} | {pct * 100.0:.1f}% | "
                f"elapsed {_format_seconds(elapsed)} | {eta_text}"
            )
            if now_ts - last_eta_log_ts >= log_eta_interval or frame_idx >= total:
                log(line)
                last_eta_log_ts = now_ts
            elif on_progress:
                try:
                    on_progress(line + "\n")
                except Exception:
                    pass

        frame_idx = 0

        if single_image is not None:
            if cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)()):
                return RTXSuperResResult(
                    returncode=1,
                    output_path=None,
                    log="\n".join(logs + ["Cancelled before inference"]),
                    input_fps=input_fps,
                    output_fps=output_fps,
                    frames_processed=0,
                    total_frames=1,
                    elapsed_seconds=max(0.0, time.time() - start_ts),
                )
            out_frame = run_frame(single_image)
            if out_frame is None:
                return _fail("Failed to process input image frame")
            frame_idx = 1
            emit_progress(frame_idx, total_frames or 1)
            out_path = Path(output_path or "")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if video_writer is not None or ffmpeg_writer is not None:
                write_video_frame(out_frame)
            else:
                save_params: List[int] = []
                if out_path.suffix.lower() in {".jpg", ".jpeg"}:
                    save_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(image_output_quality)]
                elif out_path.suffix.lower() == ".webp":
                    save_params = [int(cv2.IMWRITE_WEBP_QUALITY), int(image_output_quality)]
                bgr = to_bgr_for_file(out_frame)
                if bgr.dtype.itemsize == 2 and out_path.suffix.lower() != ".png":
                    bgr = (bgr.astype("float32") / 257.0 + 0.5).astype("uint8")
                ok = imwrite(out_path, bgr, save_params)
                if not ok:
                    return _fail(f"Failed to save output image: {out_path}")
        else:
            while True:
                if cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)()):
                    return _fail("Cancelled by user", frame_idx)

                frame_rgb = None
                if reader is not None:
                    frame_rgb = reader.read()
                    if frame_rgb is None:
                        break
                    if total_frames <= 0:
                        total_frames = max(total_frames, frame_idx + 1)
                elif cap is not None:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if total_frames <= 0:
                        total_frames = max(total_frames, frame_idx + 1)
                else:
                    if frame_idx >= len(frame_paths):
                        break
                    frame_file = frame_paths[frame_idx]
                    frame_bgr = load_image_bgr8(frame_file)
                    if frame_bgr is None:
                        return _fail(
                            f"Unreadable input frame: {frame_file}. "
                            "Processing stopped to prevent a frame-count/timing mismatch.",
                            frame_idx,
                        )
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                out_frame = run_frame(frame_rgb)
                if out_frame is None:
                    return _fail(f"Failed to process frame {frame_idx + 1}", frame_idx)

                next_idx = frame_idx + 1
                emit_progress(next_idx, total_frames if total_frames > 0 else max(next_idx, 1))

                if video_writer is not None or ffmpeg_writer is not None:
                    write_video_frame(out_frame)
                elif output_dir_path is not None:
                    frame_name = f"{Path(input_path).stem}_{next_idx:06d}.png"
                    out_file = output_dir_path / frame_name
                    if not imwrite(out_file, to_bgr_for_file(out_frame)):
                        return _fail(f"Failed to save output frame: {out_file}", frame_idx)
                frame_idx = next_idx

        if writer_queue is not None and writer_thread is not None:
            writer_queue.put(None)
            writer_thread.join()
            writer_queue = None
            if writer_error:
                return _fail(f"Writing the output video failed: {writer_error[0]}", frame_idx)
        if video_writer is not None:
            video_writer.release()
            video_writer = None
        if ffmpeg_writer is not None:
            if ffmpeg_writer.stdin is not None:
                ffmpeg_writer.stdin.close()
            ffmpeg_returncode = ffmpeg_writer.wait(timeout=3600)  # slow presets (AV1, 8K) finish after stdin closes
            ffmpeg_writer = None
            if ffmpeg_returncode != 0:
                return _fail(f"FFmpeg video encoding failed with code {ffmpeg_returncode}.", frame_idx)

        processed = frame_idx
        elapsed = max(0.0, time.time() - start_ts)
        if processed <= 0:
            return _fail("No input frames were decoded; RTX output is incomplete.")
        if input_kind == "video" and total_frames > 0 and processed + 1 < total_frames:
            return _fail(
                f"Input decoding stopped early at {processed}/{total_frames} frames. "
                "Processing failed instead of returning a truncated video.",
                processed,
            )
        fps_proc = (float(processed) / elapsed) if elapsed > 0 else 0.0
        log(
            f"[RTX] Complete: {processed} frame(s) processed in {_format_seconds(elapsed)} "
            f"({fps_proc:.2f} FPS)."
        )

        final_output = output_path
        if output_dir_path is not None:
            final_output = str(output_dir_path)
        final_path = Path(final_output) if final_output else None
        if not final_path or not final_path.exists():
            return _fail("RTX processing ended without creating the expected output.", processed)
        if final_path.is_file() and final_path.stat().st_size <= 0:
            return _fail(f"RTX output file is empty: {final_path}", processed)

        return RTXSuperResResult(
            returncode=0,
            output_path=final_output,
            log="\n".join(logs),
            input_fps=float(input_fps),
            output_fps=float(output_fps),
            frames_processed=int(processed),
            total_frames=int(total_frames),
            elapsed_seconds=float(elapsed),
        )

    except Exception as e:
        logs.append(f"[RTX] Error: {e}")
        return RTXSuperResResult(
            returncode=1,
            output_path=None,
            log="\n".join(logs),
            input_fps=float(input_fps),
            output_fps=float(output_fps),
            frames_processed=int(processed),
            total_frames=int(total_frames),
            elapsed_seconds=max(0.0, time.time() - start_ts),
        )
    finally:
        # Ensure resources are released even on early returns.
        with suppress(Exception):
            if writer_queue is not None:
                writer_queue.put(None)
            if writer_thread is not None:
                writer_thread.join(timeout=30)
        with suppress(Exception):
            if reader is not None:
                reader.close()
        with suppress(Exception):
            if cap is not None:
                cap.release()
        with suppress(Exception):
            if video_writer is not None:
                video_writer.release()
        with suppress(Exception):
            if ffmpeg_writer is not None:
                if ffmpeg_writer.stdin is not None:
                    ffmpeg_writer.stdin.close()
                if ffmpeg_writer.poll() is None:
                    ffmpeg_writer.terminate()
                ffmpeg_writer.wait(timeout=5)
        for effect in (thdr, sr):
            with suppress(Exception):
                if effect is not None:
                    close_fn = getattr(effect, "close", None)
                    if callable(close_fn):
                        close_fn()
