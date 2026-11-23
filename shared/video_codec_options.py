"""
Video Codec and Encoding Options

Provides comprehensive video encoding configuration:
- Multiple codec support (H.264, H.265, ProRes, VP9, AV1, and NVIDIA NVENC H.264 / H.265 / AV1 with a CPU fallback)
- Pixel format options (yuv420p, yuv444p, rgb24)
- Quality presets (ultrafast to veryslow)
- Audio codec options
- FFmpeg command building

Designed for both CLI and programmatic use.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CodecProfile:
    """Profile for video codec configuration"""
    name: str
    ffmpeg_codec: str
    description: str
    quality_param: str  # "crf" or "qp" or "quality"
    quality_range: Tuple[int, int]  # (best, worst)
    quality_default: int
    supports_pixel_formats: List[str]
    supports_presets: bool
    recommended_for: List[str]  # Use cases


# Codec profiles database
CODEC_PROFILES = {
    "h264": CodecProfile(
        name="H.264 (AVC)",
        ffmpeg_codec="libx264",
        description="Best compatibility, fast encoding, good quality. Works everywhere.",
        quality_param="crf",
        quality_range=(0, 51),  # 0=lossless, 51=worst
        quality_default=23,
        supports_pixel_formats=["yuv420p", "yuv422p", "yuv444p"],
        supports_presets=True,
        recommended_for=["general", "compatibility", "youtube", "streaming"]
    ),
    "h265": CodecProfile(
        name="H.265 (HEVC)",
        ffmpeg_codec="libx265",
        description="Better compression than H.264, smaller files, slower encoding. Modern devices.",
        quality_param="crf",
        quality_range=(0, 51),
        quality_default=28,
        supports_pixel_formats=["yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p10le", "yuv444p10le"],
        supports_presets=True,
        recommended_for=["archival", "4k", "file_size"]
    ),
    "prores": CodecProfile(
        name="ProRes",
        ffmpeg_codec="prores_ks",
        description=(
            "Professional editing codec. Large files, excellent quality, fast decoding. "
            "Saved as .mov (an MP4 file cannot hold ProRes). The Pixel Format picks the profile: "
            "yuv422p10le = ProRes 422 HQ, yuv444p10le = ProRes 4444. The Video Quality (CRF) slider "
            "does not apply to ProRes. Written by Open-H3 Regenerate and RTX Super Resolution."
        ),
        quality_param="profile",
        quality_range=(0, 4),  # profile follows the pixel format (see prores_profile_for_pixel_format)
        quality_default=2,
        supports_pixel_formats=["yuv422p10le", "yuv444p10le"],
        supports_presets=False,
        recommended_for=["editing", "professional", "color_grading"]
    ),
    "vp9": CodecProfile(
        name="VP9 (WebM)",
        ffmpeg_codec="libvpx-vp9",
        description="Open-source, excellent compression. Great for web. Slow encoding.",
        quality_param="crf",
        quality_range=(0, 63),
        quality_default=31,
        supports_pixel_formats=["yuv420p", "yuv444p"],
        supports_presets=False,
        recommended_for=["web", "youtube", "open_source"]
    ),
    "av1": CodecProfile(
        name="AV1",
        ffmpeg_codec="libsvtav1",
        description="Next-gen codec. Best compression but very slow encoding. Cutting edge.",
        quality_param="crf",
        quality_range=(0, 63),
        quality_default=35,
        supports_pixel_formats=["yuv420p", "yuv420p10le", "yuv444p"],
        supports_presets=True,
        recommended_for=["future", "archival", "best_compression"]
    ),
    # NVIDIA NVENC: the GPU's video encoder chip does the encoding, so a few-vCPU cloud machine no longer waits on
    # the CPU. Quality is NVENC's constant-quality mode (CQ, same scale as CRF); the Encoding Preset maps to NVENC
    # p1 (fastest) ... p7 (best). When the GPU or FFmpeg cannot encode it, the matching CPU encoder is used instead.
    "h264_nvenc": CodecProfile(
        name="H.264 NVENC (NVIDIA GPU)",
        ffmpeg_codec="h264_nvenc",
        description=(
            "H.264 encoded on the NVIDIA GPU (NVENC): very fast and light on the CPU, ideal on cloud GPUs with few "
            "vCPUs. Up to 4096 pixels wide/high (larger videos use libx264). Video Quality is the NVENC CQ value "
            "(lower = better). Falls back to libx264 when NVENC is not available."
        ),
        quality_param="cq",
        quality_range=(1, 51),
        quality_default=19,
        supports_pixel_formats=["yuv420p", "yuv444p"],
        supports_presets=True,
        recommended_for=["fast_gpu_encode", "cloud", "compatibility"]
    ),
    "hevc_nvenc": CodecProfile(
        name="H.265 / HEVC NVENC (NVIDIA GPU)",
        ffmpeg_codec="hevc_nvenc",
        description=(
            "H.265 encoded on the NVIDIA GPU (NVENC): very fast and light on the CPU, 10-bit (Main10) with "
            "yuv420p10le, up to 8192 pixels. Video Quality is the NVENC CQ value (lower = better). For final "
            "masters at the smallest size, libx265 (h265) still compresses better per bit. Falls back to libx265 "
            "when NVENC is not available."
        ),
        quality_param="cq",
        quality_range=(1, 51),
        quality_default=19,
        supports_pixel_formats=["yuv420p", "yuv420p10le", "yuv444p", "yuv444p10le"],
        supports_presets=True,
        recommended_for=["fast_gpu_encode", "cloud", "4k", "10bit"]
    ),
    "av1_nvenc": CodecProfile(
        name="AV1 NVENC (NVIDIA GPU, RTX 40 / 50)",
        ffmpeg_codec="av1_nvenc",
        description=(
            "AV1 encoded on the NVIDIA GPU (NVENC): RTX 40 series, RTX 50 series, RTX 6000 Ada, L4, L40 and newer "
            "(RTX 30 / A6000 / A100 have no AV1 encoder). 8-bit or 10-bit, up to 8192 pixels. Video Quality is the "
            "NVENC CQ value (lower = better). Falls back to libsvtav1 when NVENC AV1 is not available."
        ),
        quality_param="cq",
        quality_range=(1, 63),
        quality_default=24,
        supports_pixel_formats=["yuv420p", "yuv420p10le"],
        supports_presets=True,
        recommended_for=["fast_gpu_encode", "cloud", "best_compression"]
    ),
}


PIXEL_FORMATS = {
    "yuv420p": {
        "name": "YUV 4:2:0 (Standard)",
        "description": "Best compatibility. Used by most players and platforms. Chroma subsampling.",
        "bit_depth": 8,
        "chroma_subsampling": "4:2:0",
        "file_size": "smallest",
        "quality": "good",
        "compatibility": "universal"
    },
    "yuv422p": {
        "name": "YUV 4:2:2 (Professional)",
        "description": "Professional broadcast standard. Better chroma detail than 4:2:0.",
        "bit_depth": 8,
        "chroma_subsampling": "4:2:2",
        "file_size": "medium",
        "quality": "excellent",
        "compatibility": "professional"
    },
    "yuv444p": {
        "name": "YUV 4:4:4 (No Subsampling)",
        "description": "No chroma subsampling. Full color detail. Large files.",
        "bit_depth": 8,
        "chroma_subsampling": "4:4:4",
        "file_size": "large",
        "quality": "maximum",
        "compatibility": "limited"
    },
    "yuv420p10le": {
        "name": "YUV 4:2:0 10-bit",
        "description": "10-bit color depth. Reduces banding. Better HDR support.",
        "bit_depth": 10,
        "chroma_subsampling": "4:2:0",
        "file_size": "medium",
        "quality": "excellent",
        "compatibility": "modern"
    },
    "yuv444p10le": {
        "name": "YUV 4:4:4 10-bit",
        "description": "10-bit with no subsampling. Professional grading. Very large files.",
        "bit_depth": 10,
        "chroma_subsampling": "4:4:4",
        "file_size": "very large",
        "quality": "reference",
        "compatibility": "professional"
    },
    "rgb24": {
        "name": "RGB 24-bit",
        "description": "Lossless RGB. Maximum quality but huge files. Editing only.",
        "bit_depth": 8,
        "chroma_subsampling": "none",
        "file_size": "huge",
        "quality": "lossless",
        "compatibility": "editing"
    },
}


ENCODING_PRESETS = [
    "ultrafast",  # Fastest encoding, largest files
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",     # Default balanced
    "slow",
    "slower",
    "veryslow",   # Best compression, slowest
]

# libsvtav1 exposes a numeric speed preset (0 is slowest/best compression, 13 is
# fastest), unlike x264/x265's named presets. Passing values such as ``ultrafast``
# makes ffmpeg exit before it reads the first frame, which previously broke the
# concat re-encode fallback whenever AV1 was selected.
SVT_AV1_PRESET_MAP = {
    "veryslow": 2,
    "slower": 3,
    "slow": 4,
    "medium": 6,
    "fast": 8,
    "faster": 9,
    "veryfast": 10,
    "superfast": 11,
    "ultrafast": 12,
}

H265_TUNE_OPTIONS = [
    "none",
    "grain",
    "psnr",
    "ssim",
    "fastdecode",
    "zerolatency",
    "animation",
]

DEFAULT_AV1_FILM_GRAIN = 8
DEFAULT_AV1_FILM_GRAIN_DENOISE = False


AUDIO_CODECS = {
    "copy": "Copy original (no re-encoding)",
    "aac": "AAC (best compatibility)",
    "opus": "Opus (best quality per bitrate)",
    "flac": "FLAC (lossless)",
    "none": "No audio (remove audio track)",
}


def _normalize_h265_tune(value: Optional[str]) -> str:
    tune = str(value or "none").strip().lower()
    return tune if tune in H265_TUNE_OPTIONS else "none"


def _normalize_av1_film_grain(value: Optional[int]) -> int:
    try:
        grain = int(float(value if value is not None else DEFAULT_AV1_FILM_GRAIN))
    except Exception:
        grain = DEFAULT_AV1_FILM_GRAIN
    return max(0, min(50, grain))


PRORES_CODEC_NAMES = {"prores", "prores_ks", "prores_aw"}
PRORES_PIXEL_FORMATS = ("yuv422p10le", "yuv444p10le")


def is_prores_codec(codec: Optional[str]) -> bool:
    """True for the Output tab's ProRes choice (any of its encoder names)."""
    return str(codec or "").strip().lower() in PRORES_CODEC_NAMES


def prores_pixel_format(pixel_format: Optional[str]) -> str:
    """The ProRes pixel format for an Output tab value (4:2:2 10-bit unless 4:4:4 was chosen)."""
    pix = str(pixel_format or "").strip().lower()
    if pix in PRORES_PIXEL_FORMATS:
        return pix
    return "yuv444p10le" if "444" in pix else "yuv422p10le"


def prores_profile_for_pixel_format(pixel_format: Optional[str]) -> str:
    """prores_ks profile for a pixel format: ProRes 4444 for 4:4:4, ProRes 422 HQ otherwise."""
    return "4444" if "444" in prores_pixel_format(pixel_format) else "hq"


def video_container_suffix(codec: Optional[str]) -> str:
    """File suffix a codec is written in: .mov for ProRes (MP4 cannot hold it), .mp4 for the rest."""
    return ".mov" if is_prores_codec(codec) else ".mp4"


# --------------------------------------------------------------------------------------------------------------
# NVIDIA NVENC (Output tab codecs h264_nvenc / hevc_nvenc / av1_nvenc).
#
# The encoder chip of the GPU encodes, so the CPU is free; on cloud GPUs with few vCPUs a libx265 encode was the
# slowest part of a run. Every NVENC encode is checked first with a tiny test encode in the same place it will
# run (same GPU, same size, same options); when NVENC cannot do it (no NVIDIA GPU or driver, an FFmpeg without
# NVENC, a GPU without that codec such as AV1 on RTX 30, a size above the encoder's limit) the matching CPU
# encoder (libx264 / libx265 / libsvtav1) is used with the same quality value, and the log says why.
# --------------------------------------------------------------------------------------------------------------
NVENC_CODECS: Tuple[str, ...] = ("h264_nvenc", "hevc_nvenc", "av1_nvenc")
NVENC_CPU_FALLBACK = {"h264_nvenc": "h264", "hevc_nvenc": "h265", "av1_nvenc": "av1"}
NVENC_DISABLE_ENV = "SECOURSES_DISABLE_NVENC"  # 1 = treat NVENC as unavailable (tests, troubleshooting)
# Environment variable through which the app hands its resolved FFmpeg video arguments (a JSON list, from -c:v
# on) to engine CLIs in other repositories (RIFE, SeedVR2, FlashVSR); a CLI that does not read it uses the CPU
# codec it is given on its command line.
ENGINE_VIDEO_ARGS_ENV = "SECOURSES_FFMPEG_VIDEO_ARGS"

_NVENC_ALIASES = {
    "h264_nvenc": "h264_nvenc", "nvenc_h264": "h264_nvenc", "nvenc": "h264_nvenc",
    "hevc_nvenc": "hevc_nvenc", "h265_nvenc": "hevc_nvenc", "nvenc_hevc": "hevc_nvenc", "nvenc_h265": "hevc_nvenc",
    "av1_nvenc": "av1_nvenc", "nvenc_av1": "av1_nvenc",
}
# x264-style preset names -> NVENC p1 (fastest) ... p7 (slowest, best quality), following NVENC's own labels.
NVENC_PRESET_MAP = {
    "ultrafast": "p1", "superfast": "p1", "veryfast": "p1", "faster": "p2", "fast": "p3",
    "medium": "p4", "slow": "p5", "slower": "p6", "veryslow": "p7",
}
# Output tab pixel formats -> the input formats NVENC takes (it has no planar 10-bit input).
_NVENC_INPUT_PIX = {"yuv420p": "yuv420p", "yuv420p10le": "p010le", "yuv444p": "yuv444p", "yuv444p10le": "yuv444p16le"}
_NVENC_PROFILES = {
    "h264_nvenc": {"yuv420p": "high", "yuv444p": "high444p"},
    "hevc_nvenc": {"yuv420p": "main", "yuv420p10le": "main10", "yuv444p": "rext", "yuv444p10le": "rext"},
}
_NVENC_MAX_SIDE = {"h264_nvenc": 4096, "hevc_nvenc": 8192, "av1_nvenc": 8192}
_NVENC_LABEL = {"h264_nvenc": "H.264 NVENC", "hevc_nvenc": "H.265 NVENC", "av1_nvenc": "AV1 NVENC"}

_CODEC_KEY_ALIASES = {
    "h264": "h264", "libx264": "h264", "x264": "h264", "avc": "h264",
    "h265": "h265", "libx265": "h265", "x265": "h265", "hevc": "h265",
    "prores": "prores", "prores_ks": "prores", "prores_aw": "prores",
    "vp9": "vp9", "libvpx-vp9": "vp9",
    "av1": "av1", "libsvtav1": "av1", "svtav1": "av1", "libaom-av1": "av1",
    **_NVENC_ALIASES,
}


def nvenc_codec_key(value: object) -> Optional[str]:
    """'h264_nvenc' / 'hevc_nvenc' / 'av1_nvenc' for an NVENC codec name, else None."""
    return _NVENC_ALIASES.get(str(value or "").strip().lower())


def is_nvenc_codec(value: object) -> bool:
    return nvenc_codec_key(value) is not None


def normalize_codec_key(value: object, default: str = "h264") -> str:
    """The Output tab codec key (h264, h265, prores, vp9, av1, h264_nvenc, hevc_nvenc, av1_nvenc) for any alias."""
    return _CODEC_KEY_ALIASES.get(str(value or "").strip().lower(), default)


def nvenc_cpu_fallback_key(value: object) -> Optional[str]:
    """The CPU codec key an NVENC codec falls back to (h264_nvenc -> h264, hevc_nvenc -> h265, av1_nvenc -> av1)."""
    key = nvenc_codec_key(value)
    return NVENC_CPU_FALLBACK.get(key) if key else None


def nvenc_cpu_fallback_encoder(value: object) -> Optional[str]:
    """The CPU encoder an NVENC codec falls back to (libx264 / libx265 / libsvtav1)."""
    key = nvenc_cpu_fallback_key(value)
    return CODEC_PROFILES[key].ffmpeg_codec if key else None


def nvenc_pixel_format(codec: object, pixel_format: Optional[str]) -> str:
    """The Output tab pixel format an NVENC codec encodes (the nearest one it supports)."""
    key = nvenc_codec_key(codec) or "h264_nvenc"
    supported = CODEC_PROFILES[key].supports_pixel_formats
    pix = str(pixel_format or "").strip().lower()
    if pix in supported:
        return pix
    if "444" in pix:
        deep = "yuv444p10le" if "10" in pix or "12" in pix or "16" in pix else "yuv444p"
        if deep in supported:
            return deep
        if "yuv444p" in supported:
            return "yuv444p"
    if ("10" in pix or "12" in pix or "16" in pix) and "yuv420p10le" in supported:
        return "yuv420p10le"
    return supported[0]


def nvenc_video_args(codec: object, quality: object = 19, pixel_format: Optional[str] = "yuv420p",
                     preset: Optional[str] = "medium", spatial_aq: bool = True) -> List[str]:
    """FFmpeg video arguments for an NVENC codec: constant quality (CQ = the Video Quality value), preset p1-p7."""
    key = nvenc_codec_key(codec)
    if key is None:
        raise ValueError(f"not an NVENC codec: {codec!r}")
    profile = CODEC_PROFILES[key]
    lo, hi = profile.quality_range
    try:
        cq = int(round(float(quality)))
    except Exception:
        cq = profile.quality_default
    cq = max(lo, min(hi, cq))  # CQ 0 means "automatic" to NVENC, not lossless
    pix = nvenc_pixel_format(key, pixel_format)
    args = ["-c:v", key, "-preset", NVENC_PRESET_MAP.get(str(preset or "").strip().lower(), "p4"),
            "-tune", "hq", "-rc", "vbr", "-cq", str(cq), "-b:v", "0"]
    if spatial_aq:
        args += ["-spatial-aq", "1"]
    nv_profile = _NVENC_PROFILES.get(key, {}).get(pix)
    if nv_profile:
        args += ["-profile:v", nv_profile]
    args += ["-pix_fmt", _NVENC_INPUT_PIX[pix]]
    return args


def nvenc_describe(codec: object, pixel_format: Optional[str]) -> str:
    """Short label, e.g. 'H.265 NVENC 10-bit (Main10)'."""
    key = nvenc_codec_key(codec) or "h264_nvenc"
    pix = nvenc_pixel_format(key, pixel_format)
    depth = "10-bit" if "10" in pix else "8-bit"
    chroma = " 4:4:4" if "444" in pix else ""
    profile = _NVENC_PROFILES.get(key, {}).get(pix)
    extra = {"main10": " (Main10)", "rext": " (RExt)", "high444p": " (High 4:4:4)"}.get(profile or "", "")
    return f"{_NVENC_LABEL[key]} {depth}{chroma}{extra}"


_NVENC_PROBE_CACHE: Dict[tuple, Tuple[Optional[str], float]] = {}  # key -> (result, time of the check)
_NVENC_PROBE_LOCK = None  # created lazily (threading imported on first use)
_NVENC_NOTES_SHOWN: set = set()
# A failure that can pass later (no free video memory or encoder session while another job runs, a timeout) is
# checked again after this many seconds; a missing encoder, GPU or driver stays cached for the session.
_NVENC_TRANSIENT_RETRY_S = 120.0
_NVENC_PERMANENT_ERRORS = ("no capable devices", "unknown encoder", "encoder not found", "cannot load",
                           "driver does not support", "no nvenc capable", "not supported", "unsupported")


def _ffmpeg_exe() -> Optional[str]:
    import os
    import shutil

    env = os.environ.get("FFMPEG_PATH")
    if env and os.path.isfile(env):
        return env
    return shutil.which("ffmpeg")


def _clean_ffmpeg_error(stderr: str, limit: int = 3) -> str:
    """FFmpeg's error text without the '[... @ 0x..]' prefixes; the encoder's own lines first (the real cause,
    e.g. 'No capable devices found'), else the last lines."""
    import re

    lines: List[str] = []
    encoder_lines: List[str] = []
    for raw in (stderr or "").strip().splitlines():
        raw = raw.strip()
        own = re.match(r"^\[(?:h264|hevc|av1)_nvenc @ [0-9a-fA-Fx]+\]", raw) is not None
        text = re.sub(r"^(\[[^\]]*@ [0-9a-fA-Fx]+\]\s*)+", "", raw)
        if text and text not in lines:
            lines.append(text)
            if own:
                encoder_lines.append(text)
    if encoder_lines:
        return " | ".join(encoder_lines[:limit])
    return " | ".join(lines[-limit:])


def nvenc_disabled() -> bool:
    import os

    return str(os.environ.get(NVENC_DISABLE_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}


def probe_nvenc_args(args: List[str], width: Optional[int] = None, height: Optional[int] = None,
                     gpu: Optional[str] = None, timeout: float = 45.0) -> Optional[str]:
    """None when FFmpeg encodes a few test frames with these NVENC arguments here, else the reason.

    width / height: the real video size (NVENC limits depend on it); 1280x720 when unknown.
    gpu: a CUDA device index the encode will see (CUDA_VISIBLE_DEVICES), None = this process's own view.
    Results are cached for the process, so a long batch checks once.
    """
    import os
    import subprocess
    import threading
    import time

    global _NVENC_PROBE_LOCK
    if nvenc_disabled():
        return f"NVENC is turned off ({NVENC_DISABLE_ENV}=1)"
    ffmpeg = _ffmpeg_exe()
    if not ffmpeg:
        return "ffmpeg not found"
    try:
        w = int(width or 0)
        h = int(height or 0)
    except Exception:
        w = h = 0
    if w <= 0 or h <= 0:
        w, h = 1280, 720
    w, h = max(256, w + (w % 2)), max(144, h + (h % 2))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") if gpu is None else str(gpu)
    key = (ffmpeg, tuple(str(a) for a in args), w, h, visible)
    if _NVENC_PROBE_LOCK is None:
        _NVENC_PROBE_LOCK = threading.Lock()
    with _NVENC_PROBE_LOCK:
        if key in _NVENC_PROBE_CACHE:
            cached, checked_at = _NVENC_PROBE_CACHE[key]
            transient = cached is not None and not any(p in cached.lower() for p in _NVENC_PERMANENT_ERRORS)
            if not transient or time.time() - checked_at < _NVENC_TRANSIENT_RETRY_S:
                return cached
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-f", "lavfi",
               "-i", f"color=c=gray:s={w}x{h}:r=25,format=rgb24", "-frames:v", "3",
               *[str(a) for a in args], "-an", "-f", "null", "-"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env,
                                  creationflags=0x08000000 if os.name == "nt" else 0)
            if proc.returncode == 0:
                result = None
            else:
                result = _clean_ffmpeg_error(proc.stderr) or f"ffmpeg exit code {proc.returncode}"
        except subprocess.TimeoutExpired:
            result = "the NVENC test encode did not finish in time"
        except Exception as exc:
            result = str(exc) or exc.__class__.__name__
        _NVENC_PROBE_CACHE[key] = (result, time.time())
        return result


def resolve_nvenc_video_args(codec: object, quality: object = 19, pixel_format: Optional[str] = "yuv420p",
                             preset: Optional[str] = "medium", width: Optional[int] = None,
                             height: Optional[int] = None, gpu: Optional[str] = None
                             ) -> Tuple[Optional[List[str]], Optional[str]]:
    """(NVENC arguments, None) when NVENC can encode this video here, else (None, reason)."""
    key = nvenc_codec_key(codec)
    if key is None:
        return None, f"{codec!r} is not an NVENC codec"
    try:
        w, h = int(width or 0), int(height or 0)
    except Exception:
        w = h = 0
    limit = _NVENC_MAX_SIDE[key]
    if max(w, h) > limit:
        return None, f"{_NVENC_LABEL[key]} encodes up to {limit}x{limit} pixels and this video is {w}x{h}"
    args = nvenc_video_args(key, quality, pixel_format, preset, spatial_aq=True)
    err = probe_nvenc_args(args, w or None, h or None, gpu)
    if err is None:
        return args, None
    plain = nvenc_video_args(key, quality, pixel_format, preset, spatial_aq=False)
    if probe_nvenc_args(plain, w or None, h or None, gpu) is None:
        return plain, None
    return None, err


def nvenc_fallback_note(codec: object, reason: Optional[str], tag: str = "") -> str:
    key = nvenc_codec_key(codec) or "h264_nvenc"
    cpu = nvenc_cpu_fallback_encoder(key)
    prefix = f"{tag} " if tag else ""
    return (f"{prefix}{key} (NVIDIA GPU encoding) is not available here: {reason or 'unknown reason'}. "
            f"Encoding with {cpu} (CPU) instead, same quality value.")


def _report_note_once(note: str, on_note=None) -> None:
    if on_note is not None:
        try:
            on_note(note)
            return
        except Exception:
            pass
    if note not in _NVENC_NOTES_SHOWN:
        _NVENC_NOTES_SHOWN.add(note)
        print(f"[Video Encode] {note}", flush=True)


def resolve_encoder_name(codec: object, quality: object = 19, pixel_format: Optional[str] = "yuv420p",
                         preset: Optional[str] = "medium", width: Optional[int] = None,
                         height: Optional[int] = None, gpu: Optional[str] = None
                         ) -> Tuple[str, Optional[str]]:
    """(encoder for an engine writer, note): an NVENC codec stays NVENC when it works here, else its CPU encoder."""
    key = nvenc_codec_key(codec)
    if key is None:
        return str(codec or "libx264"), None
    args, reason = resolve_nvenc_video_args(key, quality, pixel_format, preset, width, height, gpu)
    if args:
        return key, None
    return str(nvenc_cpu_fallback_encoder(key)), nvenc_fallback_note(key, reason)


def engine_encoder_for_run(codec: object, quality: object, pixel_format: Optional[str], preset: Optional[str],
                           width: Optional[int] = None, height: Optional[int] = None, gpu: Optional[str] = None,
                           log=None, tag: str = "") -> str:
    """The encoder an engine run writes with, logged: NVENC when it works for this video and GPU, else the CPU one."""
    encoder, note = resolve_encoder_name(codec, quality, pixel_format, preset, width, height, gpu)
    prefix = f"[{tag}] " if tag else ""
    message = None
    if note:
        message = prefix + note
    elif is_nvenc_codec(encoder):
        message = (f"{prefix}Video encode on the NVIDIA GPU: {nvenc_describe(encoder, pixel_format)}, CQ {quality}, "
                   f"preset {NVENC_PRESET_MAP.get(str(preset or '').strip().lower(), 'p4')}")
    if message and log is not None:
        try:
            log(message)
        except Exception:
            pass
    return encoder


def nvenc_status_text(codec: object, pixel_format: Optional[str], width: Optional[int] = None,
                      height: Optional[int] = None) -> str:
    """One line for the Output tab: whether the selected NVENC codec works on this machine."""
    key = nvenc_codec_key(codec)
    if key is None:
        return ""
    args, reason = resolve_nvenc_video_args(key, 19, pixel_format, "medium", width, height)
    label = nvenc_describe(key, pixel_format)
    if args:
        return (f"✅ **{label} works on this machine.** Runs encode on the NVIDIA GPU; each run checks it again "
                f"for its own video size and GPU.")
    return (f"⚠️ **{label} is not available here** ({reason}). Runs will encode with "
            f"{nvenc_cpu_fallback_encoder(key)} (CPU) instead, with the same quality value.")


_CLI_ARGS_SUPPORT_CACHE: Dict[str, Tuple[int, bool]] = {}


def cli_takes_app_video_args(cli_path) -> bool:
    """True when an engine CLI from another repository (RIFE, SeedVR2, FlashVSR) reads the app's FFmpeg video
    arguments (ENGINE_VIDEO_ARGS_ENV); an older checkout gets the CPU codec on its command line instead."""
    from pathlib import Path

    path = Path(cli_path)
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return False
    cached = _CLI_ARGS_SUPPORT_CACHE.get(str(path))
    if cached and cached[0] == stamp:
        return cached[1]
    try:
        supported = ENGINE_VIDEO_ARGS_ENV in path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        supported = False
    _CLI_ARGS_SUPPORT_CACHE[str(path)] = (stamp, supported)
    return supported


def engine_args_env_value(args: Optional[List[str]]) -> str:
    """The ENGINE_VIDEO_ARGS_ENV value for a list of FFmpeg video arguments ('' = none)."""
    import json

    return json.dumps([str(a) for a in args]) if args else ""


def engine_args_from_env(environ=None) -> Optional[List[str]]:
    """The FFmpeg video arguments the app handed over in ENGINE_VIDEO_ARGS_ENV (None when unset or malformed)."""
    import json
    import os

    raw = str((environ if environ is not None else os.environ).get(ENGINE_VIDEO_ARGS_ENV, "") or "").strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except Exception:
        return None
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        return value
    return None


# --------------------------------------------------------------------------------------------------------------
# Custom FFmpeg video arguments (Output tab): the user's own encoder options from -c:v on, used instead of the
# Video Codec / Quality / Preset / Pixel Format settings by the engines that pipe frames into FFmpeg.
# --------------------------------------------------------------------------------------------------------------
CUSTOM_VIDEO_ARGS_MAX_CHARS = 2000

# Options the app sets itself (inputs, mapping, timing, audio, overwrite); a custom value would break the run.
_CUSTOM_ARGS_REFUSED = {
    "-i": "the app gives the input",
    "-y": "the app overwrites its own files",
    "-n": "the app overwrites its own files",
    "-map": "the app maps the video and audio",
    "-filter_complex": "use -vf for filters",
    "-lavfi": "use -vf for filters",
    "-ss": "trimming would drop frames", "-sseof": "trimming would drop frames",
    "-t": "trimming would drop frames", "-to": "trimming would drop frames",
    "-frames": "trimming would drop frames", "-frames:v": "trimming would drop frames",
    "-vframes": "trimming would drop frames",
    "-r": "the frame rate follows the input (FPS settings)", "-r:v": "the frame rate follows the input (FPS settings)",
    "-an": "audio follows the Audio Codec setting", "-c:a": "audio follows the Audio Codec setting",
    "-acodec": "audio follows the Audio Codec setting", "-codec:a": "audio follows the Audio Codec setting",
    "-b:a": "audio follows the Audio Codec setting",
    "-c": "write -c:v (it would also change the audio codec)",
    "-codec": "write -c:v (it would also change the audio codec)",
}
_CUSTOM_ARGS_CODEC_OPTS = ("-c:v", "-vcodec", "-codec:v")
_CUSTOM_ARGS_VALUE_FREE = {"-shortest", "-bitexact", "-autoscale", "-noautoscale"}

# Encoders an MP4 file cannot hold: QuickTime-family codecs go into .mov, lossless / raw ones into .mkv.
_MOV_ENCODERS = {"prores", "prores_ks", "prores_aw", "prores_videotoolbox", "dnxhd", "cfhd", "qtrle", "hap",
                 "mjpeg", "png", "v210", "r210", "v410", "v308", "v408"}
_MKV_ENCODERS = {"ffv1", "utvideo", "huffyuv", "ffvhuff", "magicyuv", "rawvideo", "libvpx", "vp8", "libtheora",
                 "theora", "zlib", "flashsv", "flashsv2", "snow"}
_FORMAT_SUFFIX = {"mov": ".mov", "mp4": ".mp4", "matroska": ".mkv", "mkv": ".mkv"}

_DEFAULT_ENCODER_NAMES = {"h264": "libx264", "h265": "libx265", "prores": "prores_ks", "vp9": "libvpx-vp9",
                          "av1": "libsvtav1", "h264_nvenc": "h264_nvenc", "hevc_nvenc": "hevc_nvenc",
                          "av1_nvenc": "av1_nvenc"}


def normalize_custom_video_args(value: object) -> str:
    """The Output tab's custom FFmpeg video arguments as one trimmed line ('' when unset)."""
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    return text[:CUSTOM_VIDEO_ARGS_MAX_CHARS]


def custom_video_args_container(args: List[str]) -> str:
    """File suffix for custom arguments from their encoder: .mov / .mkv for codecs MP4 cannot hold, else .mp4."""
    encoder = ""
    for i, tok in enumerate(args[:-1]):
        low = tok.lower()
        if low in _CUSTOM_ARGS_CODEC_OPTS:
            encoder = args[i + 1].strip().lower()
    if encoder in _MOV_ENCODERS:
        return ".mov"
    if encoder in _MKV_ENCODERS:
        return ".mkv"
    return ".mp4"


def parse_custom_video_args(text: object, default_codec: Optional[str] = "h264") -> Tuple[List[str], str, Optional[str]]:
    """(FFmpeg arguments, container suffix, error) for the Output tab's custom video arguments.

    The text holds FFmpeg output options, for example `-c:v prores_ks -profile:v 4444xq -pix_fmt yuv444p10le`.
    Without -c:v, the encoder of the selected Video Codec is used. Every option takes a value; a lone word
    (which FFmpeg would treat as a second output file) is refused, as are the options the app sets itself.
    Empty text gives ([], "", None): the Video Codec settings apply.
    """
    import shlex

    normalized = normalize_custom_video_args(text)
    if not normalized:
        return [], "", None
    try:
        tokens = shlex.split(normalized, posix=True)
    except ValueError as exc:
        return [], "", f"could not read the arguments ({exc}); check the quotes"
    if not tokens:
        return [], "", None
    args: List[str] = []
    forced_suffix = ""
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("-") or len(tok) < 2:
            return [], "", (f"'{tok}' is not an option. Write options with their values, for example "
                            "-c:v libx265 -crf 12 -pix_fmt yuv444p10le (no input or output file names)")
        low = tok.lower()
        if low in _CUSTOM_ARGS_REFUSED:
            return [], "", f"{tok} cannot be used here: {_CUSTOM_ARGS_REFUSED[low]}"
        if low in _CUSTOM_ARGS_VALUE_FREE:
            args.append(tok)
            i += 1
            continue
        if i + 1 >= len(tokens):
            return [], "", f"{tok} needs a value"
        value = tokens[i + 1]
        if value.startswith("-") and not value[1:2].isdigit():
            return [], "", f"{tok} needs a value (found {value})"
        if low == "-f":
            # The container follows the file name; -f only picks it (mov, mp4 or matroska).
            if value.strip().lower() not in _FORMAT_SUFFIX:
                return [], "", f"-f {value} is not supported here; use mov, mp4 or matroska"
            forced_suffix = _FORMAT_SUFFIX[value.strip().lower()]
            i += 2
            continue
        args += [tok, value]
        i += 2
    if not any(t.lower() in _CUSTOM_ARGS_CODEC_OPTS for t in args[::2]):
        codec_key = str(default_codec or "h264").strip().lower()
        encoder = _DEFAULT_ENCODER_NAMES.get(codec_key, codec_key if codec_key.startswith("lib") or "_" in codec_key
                                             else "libx264")
        args = ["-c:v", encoder] + args
    return args, forced_suffix or custom_video_args_container(args), None


def video_output_suffix(codec: Optional[str], custom_args: object = None) -> str:
    """Container of a video output: the custom arguments' when they are set and valid, else the codec's."""
    if normalize_custom_video_args(custom_args):
        args, suffix, err = parse_custom_video_args(custom_args, default_codec=codec)
        if args and not err:
            return suffix
    return video_container_suffix(codec)


def video_suffix_fits(required_suffix: str, actual_suffix: str) -> bool:
    """True when a file with actual_suffix can hold what needs required_suffix (.mp4 codecs also fit .mov / .mkv)."""
    need = str(required_suffix or ".mp4").lower()
    have = str(actual_suffix or "").lower()
    if have == need:
        return True
    if need == ".mp4":
        return have in (".mp4", ".m4v", ".mov", ".mkv")
    if need == ".mov":
        return have in (".mov", ".mkv")
    return False


def describe_custom_video_args(args: List[str], suffix: str) -> str:
    encoder = next((args[i + 1] for i in range(len(args) - 1) if args[i].lower() in _CUSTOM_ARGS_CODEC_OPTS), "?")
    return f"encoder {encoder}, saved as {suffix}"


def check_custom_video_args(args: List[str], suffix: str, timeout: float = 60.0) -> Optional[str]:
    """Encode a few test frames with the arguments; None when FFmpeg accepts them, else its error text."""
    import os
    import shutil
    import subprocess
    import tempfile

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return "ffmpeg not found"
    tmp_dir = tempfile.mkdtemp(prefix="secourses_custom_args_")
    out = os.path.join(tmp_dir, f"check{suffix or '.mp4'}")
    cmd = [ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=256x144:r=24:d=0.25,format=rgb24",
           *args, "-an"]
    if suffix in (".mp4", ".mov"):
        cmd += ["-movflags", "+faststart"]
    cmd.append(out)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              creationflags=0x08000000 if os.name == "nt" else 0)
        if proc.returncode != 0 or not os.path.isfile(out) or os.path.getsize(out) <= 0:
            import re

            lines = []
            for raw in (proc.stderr or "").strip().splitlines():
                text = re.sub(r"^\[[^\]]*@ [0-9a-fA-Fx]+\]\s*", "", raw.strip())  # "[vost#0:0 @ 0x..] "
                if text and tmp_dir not in text and text not in lines:
                    lines.append(text)
            return " | ".join(lines[-4:]) or f"ffmpeg exit code {proc.returncode}"
        return None
    except subprocess.TimeoutExpired:
        return "the test encode did not finish in time"
    except Exception as exc:
        return str(exc)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def get_codec_choices() -> List[str]:
    """Get list of available codec choices for UI"""
    return list(CODEC_PROFILES.keys())


def get_pixel_format_choices(codec: str = "h264") -> List[str]:
    """Get compatible pixel formats for a codec"""
    if codec in CODEC_PROFILES:
        return CODEC_PROFILES[codec].supports_pixel_formats
    return ["yuv420p"]  # Safe default


def get_codec_info(codec: str) -> str:
    """Get formatted info string for codec"""
    if codec in CODEC_PROFILES:
        profile = CODEC_PROFILES[codec]
        return f"**{profile.name}**\n{profile.description}\n\nBest for: {', '.join(profile.recommended_for)}"
    return "Unknown codec"


def get_pixel_format_info(pix_fmt: str) -> str:
    """Get formatted info string for pixel format"""
    if pix_fmt in PIXEL_FORMATS:
        info = PIXEL_FORMATS[pix_fmt]
        return f"**{info['name']}**\n{info['description']}\n\n" \
               f"File size: {info['file_size']} | Quality: {info['quality']} | Compatibility: {info['compatibility']}"
    return "Unknown pixel format"


def build_ffmpeg_video_encode_args(
    codec: str = "h264",
    quality: int = 23,
    pixel_format: str = "yuv420p",
    preset: str = "medium",
    audio_codec: str = "copy",
    audio_bitrate: Optional[str] = None,
    h265_tune: str = "none",
    av1_film_grain: Optional[int] = None,
    av1_film_grain_denoise: Optional[bool] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    on_note=None,
) -> List[str]:
    """
    Build ffmpeg encoding arguments from settings.

    Args:
        codec: Codec key (h264, h265, prores, vp9, av1, h264_nvenc, hevc_nvenc, av1_nvenc)
        quality: Quality value (meaning depends on codec; CQ for NVENC)
        pixel_format: Pixel format (yuv420p, yuv444p, etc.)
        preset: Encoding preset (ultrafast to veryslow; NVENC p1-p7)
        audio_codec: Audio codec (copy, aac, opus, flac, none)
        audio_bitrate: Audio bitrate (e.g., "192k", None = default)
        h265_tune: x265 tune value (none, grain, psnr, ssim, ...)
        av1_film_grain: SVT-AV1 synthetic film grain strength (0-50)
        av1_film_grain_denoise: SVT-AV1 denoise flag for film grain synthesis
        width / height: the video size when known (NVENC is checked for it)
        on_note: callback for the NVENC fallback note (printed once when None)

    Returns:
        List of ffmpeg arguments to append to command
    """
    args = []

    # NVIDIA NVENC: checked with a test encode; the matching CPU codec when it cannot encode here.
    nv_key = nvenc_codec_key(codec)
    if nv_key:
        nv_args, reason = resolve_nvenc_video_args(nv_key, quality, pixel_format, preset, width, height)
        if nv_args:
            return nv_args + _audio_encode_args(audio_codec, audio_bitrate)
        _report_note_once(nvenc_fallback_note(nv_key, reason), on_note)
        codec = NVENC_CPU_FALLBACK[nv_key]

    # Get codec profile
    if codec not in CODEC_PROFILES:
        codec = "h264"  # Safe fallback
    
    profile = CODEC_PROFILES[codec]
    
    # Video codec
    args.extend(["-c:v", profile.ffmpeg_codec])
    
    # Quality parameter
    if profile.quality_param == "crf":
        args.extend(["-crf", str(quality)])
    elif profile.quality_param == "qp":
        args.extend(["-qp", str(quality)])
    elif profile.quality_param == "profile":
        # ProRes: the profile follows the pixel format (422 HQ / 4444), as a CRF value means nothing here.
        # A CRF-indexed profile tagged 4:2:2 frames as 4444 (a non-standard file editors may refuse).
        args.extend(["-profile:v", prores_profile_for_pixel_format(pixel_format)])
    
    # Encoding preset (if supported)
    if profile.supports_presets and preset in ENCODING_PRESETS:
        preset_value = str(SVT_AV1_PRESET_MAP[preset]) if codec == "av1" else preset
        args.extend(["-preset", preset_value])

    # Codec-specific tuning hooks.
    if codec == "h265":
        tune = _normalize_h265_tune(h265_tune)
        if tune != "none":
            args.extend(["-tune", tune])
    elif codec == "av1":
        film_grain_val = _normalize_av1_film_grain(
            av1_film_grain if av1_film_grain is not None else DEFAULT_AV1_FILM_GRAIN
        )
        denoise = (
            bool(av1_film_grain_denoise)
            if av1_film_grain_denoise is not None
            else bool(DEFAULT_AV1_FILM_GRAIN_DENOISE)
        )
        args.extend(
            [
                "-svtav1-params",
                f"film-grain={film_grain_val}:film-grain-denoise={1 if denoise else 0}",
            ]
        )

    # Pixel format (validate compatibility)
    if codec == "prores":
        args.extend(["-pix_fmt", prores_pixel_format(pixel_format)])
    elif pixel_format in profile.supports_pixel_formats:
        args.extend(["-pix_fmt", pixel_format])
    else:
        # Fallback to first supported format
        args.extend(["-pix_fmt", profile.supports_pixel_formats[0]])

    # Audio codec
    args.extend(_audio_encode_args(audio_codec, audio_bitrate))
    return args


def _audio_encode_args(audio_codec: str = "copy", audio_bitrate: Optional[str] = None) -> List[str]:
    if audio_codec == "none":
        return ["-an"]  # No audio
    if audio_codec == "copy":
        return ["-c:a", "copy"]
    args = ["-c:a", audio_codec]
    if audio_bitrate:
        args.extend(["-b:a", audio_bitrate])
    elif audio_codec == "aac":
        args.extend(["-b:a", "192k"])  # Default for AAC
    elif audio_codec == "opus":
        args.extend(["-b:a", "128k"])  # Default for Opus
    return args


def get_recommended_settings(use_case: str) -> Dict[str, any]:
    """
    Get recommended encoding settings for common use cases.
    
    Args:
        use_case: "youtube", "archival", "editing", "compatibility", "web"
        
    Returns:
        Dictionary with recommended settings
    """
    presets = {
        "youtube": {
            "codec": "h264",
            "quality": 18,
            "pixel_format": "yuv420p",
            "preset": "slow",
            "audio_codec": "aac",
            "audio_bitrate": "192k"
        },
        "archival": {
            "codec": "h265",
            "quality": 20,
            "pixel_format": "yuv420p10le",
            "preset": "slower",
            "audio_codec": "flac",
            "audio_bitrate": None
        },
        "editing": {
            "codec": "prores",
            "quality": 2,  # HQ
            "pixel_format": "yuv422p10le",
            "preset": "medium",
            "audio_codec": "copy",
            "audio_bitrate": None
        },
        "compatibility": {
            "codec": "h264",
            "quality": 23,
            "pixel_format": "yuv420p",
            "preset": "medium",
            "audio_codec": "aac",
            "audio_bitrate": "128k"
        },
        "web": {
            "codec": "vp9",
            "quality": 31,
            "pixel_format": "yuv420p",
            "preset": "medium",
            "audio_codec": "opus",
            "audio_bitrate": "128k"
        },
        "max_quality": {
            "codec": "h265",
            "quality": 15,
            "pixel_format": "yuv444p10le",
            "preset": "veryslow",
            "audio_codec": "flac",
            "audio_bitrate": None
        },
        "fast_preview": {
            "codec": "h264",
            "quality": 28,
            "pixel_format": "yuv420p",
            "preset": "ultrafast",
            "audio_codec": "copy",
            "audio_bitrate": None
        },
    }
    
    return presets.get(use_case, presets["compatibility"])

