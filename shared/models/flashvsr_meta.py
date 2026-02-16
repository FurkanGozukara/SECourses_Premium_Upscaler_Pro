"""
FlashVSR+ Model Metadata Registry
Provides comprehensive model metadata with VRAM, compile, and multi-GPU constraints
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def flashvsr_version_to_internal(version: Any) -> str:
    """
    Normalize FlashVSR version to internal/CLI form.

    Supported inputs:
    - "10", "v10", "1.0", "v1.0"
    - "11", "v11", "1.1", "v1.1"
    """
    raw = str(version or "").strip().lower()
    if raw in {"11", "v11", "1.1", "v1.1"}:
        return "11"
    return "10"


def flashvsr_version_to_ui(version: Any) -> str:
    """Normalize FlashVSR version to UI form: '1.0' or '1.1'."""
    return "1.1" if flashvsr_version_to_internal(version) == "11" else "1.0"


@dataclass
class FlashVSRModel:
    """Comprehensive metadata for FlashVSR+ model configurations"""
    name: str  # e.g., "v10_tiny_4x"
    version: str  # "10" or "11"
    mode: str  # "tiny", "tiny-long", "full"
    scale: int  # 2 or 4
    
    # Multi-GPU support
    supports_multi_gpu: bool = False  # FlashVSR+ is primarily single-GPU optimized
    
    # Performance characteristics
    estimated_vram_gb: float = 6.0
    supports_fp16: bool = True
    supports_bf16: bool = True
    default_dtype: str = "bf16"
    
    # Compilation support
    compile_compatible: bool = True
    preferred_compile_backend: str = "inductor"
    
    # Tiling support
    supports_tiled_vae: bool = True
    supports_tiled_dit: bool = True
    default_tile_size: int = 256
    default_overlap: int = 24
    
    # Quality vs Speed trade-offs
    speed_tier: str = "medium"  # "fast", "medium", "slow"
    quality_tier: str = "high"  # "medium", "high", "very_high"
    
    # Resolution constraints
    max_resolution: int = 0  # 0 = no cap
    min_resolution: int = 256
    
    # Attention mode
    default_attention: str = "sage"  # "sage" or "block"
    
    # Description
    notes: str = ""


def _get_flashvsr_models() -> List[FlashVSRModel]:
    """
    Define all FlashVSR+ model configurations with comprehensive metadata.
    
    FlashVSR+ offers multiple versions and modes:
    - v10: Original release, stable
    - v11: Latest, improved quality
    - Modes:
      - tiny: Fastest, lowest VRAM (4-6GB)
      - tiny-long: Balanced temporal consistency
      - full: Highest quality, more VRAM (8-12GB)
    """
    models = []
    
    # v10 models - Original release
    for scale in [2, 4]:
        models.append(FlashVSRModel(
            name=f"v10_tiny_{scale}x",
            version="10",
            mode="tiny",
            scale=scale,
            estimated_vram_gb=4.0 if scale == 2 else 6.0,
            speed_tier="fast",
            quality_tier="high",
            notes=f"v10 tiny {scale}x - Fastest mode, lowest VRAM. Good for real-time or batch processing."
        ))
        
        models.append(FlashVSRModel(
            name=f"v10_tiny-long_{scale}x",
            version="10",
            mode="tiny-long",
            scale=scale,
            estimated_vram_gb=5.0 if scale == 2 else 7.0,
            speed_tier="medium",
            quality_tier="high",
            notes=f"v10 tiny-long {scale}x - Balanced mode with better temporal consistency."
        ))
        
        models.append(FlashVSRModel(
            name=f"v10_full_{scale}x",
            version="10",
            mode="full",
            scale=scale,
            estimated_vram_gb=8.0 if scale == 2 else 10.0,
            speed_tier="slow",
            quality_tier="very_high",
            notes=f"v10 full {scale}x - Highest quality, more VRAM required. Best results for archival."
        ))
    
    # v11 models - Latest release with improvements
    for scale in [2, 4]:
        models.append(FlashVSRModel(
            name=f"v11_tiny_{scale}x",
            version="11",
            mode="tiny",
            scale=scale,
            estimated_vram_gb=4.5 if scale == 2 else 6.5,
            speed_tier="fast",
            quality_tier="high",
            notes=f"v11 tiny {scale}x - Latest fast mode with quality improvements over v10."
        ))
        
        models.append(FlashVSRModel(
            name=f"v11_tiny-long_{scale}x",
            version="11",
            mode="tiny-long",
            scale=scale,
            estimated_vram_gb=5.5 if scale == 2 else 7.5,
            speed_tier="medium",
            quality_tier="high",
            notes=f"v11 tiny-long {scale}x - Enhanced temporal consistency, balanced performance."
        ))
        
        models.append(FlashVSRModel(
            name=f"v11_full_{scale}x",
            version="11",
            mode="full",
            scale=scale,
            estimated_vram_gb=9.0 if scale == 2 else 12.0,
            speed_tier="slow",
            quality_tier="very_high",
            notes=f"v11 full {scale}x - Latest highest quality mode. Recommended for final output."
        ))
    
    return models


_FLASHVSR_MODELS = _get_flashvsr_models()
_FLASHVSR_MODEL_MAP = {m.name: m for m in _FLASHVSR_MODELS}


def get_flashvsr_model_names() -> List[str]:
    """
    Get available FlashVSR+ model identifiers.
    
    Returns:
        List of model identifier strings (e.g., "v10_tiny_2x", "v11_full_4x")
    """
    return [m.name for m in _FLASHVSR_MODELS]


def get_flashvsr_default_model() -> str:
    """Get default FlashVSR+ model identifier."""
    return "v10_tiny_4x"


def get_flashvsr_metadata(model_name: str) -> Optional[FlashVSRModel]:
    """
    Get comprehensive metadata for a FlashVSR+ model.
    
    Args:
        model_name: Model identifier (e.g., "v10_tiny_4x")
        
    Returns:
        FlashVSRModel metadata or None if not found
    """
    return _FLASHVSR_MODEL_MAP.get(model_name)


def flashvsr_model_map() -> Dict[str, FlashVSRModel]:
    """Get mapping of model names to metadata."""
    return dict(_FLASHVSR_MODEL_MAP)

