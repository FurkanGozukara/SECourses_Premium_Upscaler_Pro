"""
Video Codec and Encoding Options

Provides comprehensive video encoding configuration:
- Multiple codec support (H.264, H.265, ProRes, VP9)
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
        supports_pixel_formats=["yuv420p", "yuv444p"],
        supports_presets=True,
        recommended_for=["future", "archival", "best_compression"]
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
                          "av1": "libsvtav1"}


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
) -> List[str]:
    """
    Build ffmpeg encoding arguments from settings.
    
    Args:
        codec: Codec key (h264, h265, prores, vp9, av1)
        quality: Quality value (meaning depends on codec)
        pixel_format: Pixel format (yuv420p, yuv444p, etc.)
        preset: Encoding preset (ultrafast to veryslow)
        audio_codec: Audio codec (copy, aac, opus, flac, none)
        audio_bitrate: Audio bitrate (e.g., "192k", None = default)
        h265_tune: x265 tune value (none, grain, psnr, ssim, ...)
        av1_film_grain: SVT-AV1 synthetic film grain strength (0-50)
        av1_film_grain_denoise: SVT-AV1 denoise flag for film grain synthesis
        
    Returns:
        List of ffmpeg arguments to append to command
    """
    args = []
    
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
    if audio_codec == "none":
        args.append("-an")  # No audio
    elif audio_codec == "copy":
        args.extend(["-c:a", "copy"])
    else:
        args.extend(["-c:a", audio_codec])
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

