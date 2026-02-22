"""
FlashVSR metadata registry for UI defaults and guardrails.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def flashvsr_version_to_internal(version: Any) -> str:
    """
    Normalize FlashVSR version to internal/CLI form.

    Supported inputs:
    - "10", "v10", "1.0", "v1.0", "FlashVSR"
    - "11", "v11", "1.1", "v1.1", "FlashVSR-v1.1"
    """
    raw = str(version or "").strip().lower()
    if raw in {"11", "v11", "1.1", "v1.1", "flashvsr-v1.1"}:
        return "11"
    return "10"


def flashvsr_version_to_ui(version: Any) -> str:
    """Normalize FlashVSR version to UI form: '1.0' or '1.1'."""
    return "1.1" if flashvsr_version_to_internal(version) == "11" else "1.0"


def flashvsr_internal_to_model_name(version_internal: Any) -> str:
    """Convert internal version ('10'/'11') to CLI model name."""
    return "FlashVSR-v1.1" if str(version_internal) == "11" else "FlashVSR"


@dataclass
class FlashVSRModel:
    name: str
    version: str
    mode: str
    scale: int
    estimated_vram_gb: float = 8.0
    supports_multi_gpu: bool = False
    supports_tiled_vae: bool = True
    supports_tiled_dit: bool = True
    default_tile_size: int = 256
    default_overlap: int = 24
    default_precision: str = "auto"
    default_attention_mode: str = "flash_attention_2"
    default_vae_model: str = "Wan2.1"
    default_keep_models_on_cpu: bool = True
    recommended_frame_chunk_size: int = 0
    recommended_resize_factor: float = 1.0
    speed_tier: str = "medium"
    quality_tier: str = "high"
    notes: str = ""


def _build_model(
    version: str,
    mode: str,
    scale: int,
    estimated_vram_gb: float,
    *,
    default_vae_model: str,
    recommended_frame_chunk_size: int,
    recommended_resize_factor: float,
    default_precision: str = "auto",
    default_keep_models_on_cpu: bool = True,
    tiled_vae: bool = True,
    tiled_dit: bool = True,
    speed_tier: str = "medium",
    quality_tier: str = "high",
    notes: str = "",
) -> FlashVSRModel:
    return FlashVSRModel(
        name=f"v{version}_{mode}_{int(scale)}x",
        version=str(version),
        mode=str(mode),
        scale=int(scale),
        estimated_vram_gb=float(estimated_vram_gb),
        default_vae_model=str(default_vae_model),
        recommended_frame_chunk_size=int(recommended_frame_chunk_size),
        recommended_resize_factor=float(recommended_resize_factor),
        default_precision=str(default_precision),
        default_keep_models_on_cpu=bool(default_keep_models_on_cpu),
        supports_tiled_vae=bool(tiled_vae),
        supports_tiled_dit=bool(tiled_dit),
        speed_tier=str(speed_tier),
        quality_tier=str(quality_tier),
        notes=str(notes or ""),
    )


def _get_flashvsr_models() -> List[FlashVSRModel]:
    models: List[FlashVSRModel] = []
    for version in ("10", "11"):
        for scale in (2, 4):
            # tiny
            models.append(
                _build_model(
                    version,
                    "tiny",
                    scale,
                    9.0 if scale == 4 else 6.5,
                    default_vae_model="Wan2.1",
                    recommended_frame_chunk_size=64 if scale == 4 else 0,
                    recommended_resize_factor=1.0,
                    default_precision="auto",
                    default_keep_models_on_cpu=True,
                    tiled_vae=True,
                    tiled_dit=False,
                    speed_tier="fast",
                    quality_tier="high",
                    notes=f"v{version} tiny {scale}x: balanced quality and speed.",
                )
            )
            # tiny-long
            models.append(
                _build_model(
                    version,
                    "tiny-long",
                    scale,
                    7.0 if scale == 4 else 5.0,
                    default_vae_model="LightVAE_W2.1",
                    recommended_frame_chunk_size=24 if scale == 4 else 40,
                    recommended_resize_factor=0.8 if scale == 4 else 1.0,
                    default_precision="fp16",
                    default_keep_models_on_cpu=True,
                    tiled_vae=True,
                    tiled_dit=True,
                    speed_tier="medium",
                    quality_tier="high",
                    notes=f"v{version} tiny-long {scale}x: lowest VRAM profile for long clips.",
                )
            )
            # full
            models.append(
                _build_model(
                    version,
                    "full",
                    scale,
                    13.0 if scale == 4 else 10.0,
                    default_vae_model="Wan2.2",
                    recommended_frame_chunk_size=0,
                    recommended_resize_factor=1.0,
                    default_precision="auto",
                    default_keep_models_on_cpu=False,
                    tiled_vae=False,
                    tiled_dit=False,
                    speed_tier="slow",
                    quality_tier="very_high",
                    notes=f"v{version} full {scale}x: highest quality, highest VRAM usage.",
                )
            )
    return models


_FLASHVSR_MODELS = _get_flashvsr_models()
_FLASHVSR_MODEL_MAP = {m.name: m for m in _FLASHVSR_MODELS}


def get_flashvsr_model_names() -> List[str]:
    return [m.name for m in _FLASHVSR_MODELS]


def get_flashvsr_default_model() -> str:
    return "v11_full_4x"


def get_flashvsr_metadata(model_name: str) -> Optional[FlashVSRModel]:
    return _FLASHVSR_MODEL_MAP.get(model_name)


def flashvsr_model_map() -> Dict[str, FlashVSRModel]:
    return dict(_FLASHVSR_MODEL_MAP)

