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
        supports_pixel_formats=["yuv420p", "yuv422p", "yuv444p"],
        supports_presets=True,
        recommended_for=["archival", "4k", "file_size"]
    ),
    "prores": CodecProfile(
        name="ProRes",
        ffmpeg_codec="prores_ks",
        description="Professional editing codec. Large files, excellent quality, fast decoding.",
        quality_param="profile",
        quality_range=(0, 4),  # 0=Proxy, 4=4444
        quality_default=2,  # ProRes HQ
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


AUDIO_CODECS = {
    "copy": "Copy original (no re-encoding)",
    "aac": "AAC (best compatibility)",
    "opus": "Opus (best quality per bitrate)",
    "flac": "FLAC (lossless)",
    "none": "No audio (remove audio track)",
}


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
    audio_bitrate: Optional[str] = None
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
        # ProRes profiles
        prores_profiles = ["proxy", "lt", "standard", "hq", "4444"]
        profile_name = prores_profiles[min(quality, len(prores_profiles)-1)]
        args.extend(["-profile:v", profile_name])
    
    # Encoding preset (if supported)
    if profile.supports_presets and preset in ENCODING_PRESETS:
        args.extend(["-preset", preset])
    
    # Pixel format (validate compatibility)
    if pixel_format in profile.supports_pixel_formats:
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

