from dataclasses import dataclass
from typing import Dict, List
from threading import Lock
import time

from pathlib import Path

DEFAULT_BATCH_SIZE = 5
# Default attention mode: flash_attn preferred, falls back to sdpa if unavailable
# This is used as fallback for unknown models. Runtime detection (_get_default_attention_mode) 
# tests actual GPU compatibility and takes precedence.
DEFAULT_ATTENTION = "flash_attn"  # Preferred default, runtime will fall back to sdpa if needed

# Local scan will look for common weight extensions inside ./models/seedvr2 (and siblings).
MODEL_EXTS = {".safetensors", ".gguf"}
_SEED_SCAN_CACHE_TTL_SEC = 20.0
_SEED_SCAN_CACHE_LOCK = Lock()
_SEED_SCAN_CACHE: tuple[float, tuple[tuple[str, float], ...], List[str]] = (0.0, tuple(), [])


@dataclass
class SeedVR2Model:
    name: str
    size: str
    precision: str
    variant: str
    # Multi-GPU support
    supports_multi_gpu: bool = True
    # Performance defaults
    default_batch_size: int = DEFAULT_BATCH_SIZE
    preferred_attention: str = DEFAULT_ATTENTION
    # Compilation support
    compile_compatible: bool = True
    preferred_compile_backend: str = "inductor"  # inductor, cudagraphs
    preferred_compile_mode: str = "default"  # default, reduce-overhead, max-autotune
    # Resolution constraints
    max_resolution: int = 0  # 0 = no cap
    min_resolution: int = 256  # Minimum safe resolution
    # Offload recommendations
    recommended_dit_offload: str = "none"  # none, cpu, or specific GPU
    recommended_vae_offload: str = "none"
    # BlockSwap support
    supports_blockswap: bool = True
    max_blocks_to_swap: int = 36  # Maximum safe blocks for this model
    # Caching support (CUDA graphs)
    supports_cache_dit: bool = True
    supports_cache_vae: bool = True
    requires_single_gpu_for_cache: bool = True  # Most models require single GPU for caching
    # Memory requirements (rough estimates in GB)
    estimated_vram_gb: float = 8.0  # Estimated VRAM usage
    # Special notes
    notes: str = ""


def _built_ins() -> List[SeedVR2Model]:
    return [
        # 3B models - lower VRAM, faster, good for most use cases
        SeedVR2Model(
            name="seedvr2_ema_3b_fp8_e4m3fn.safetensors",
            size="3B", precision="fp8_e4m3fn", variant="standard",
            preferred_attention="flash_attn", supports_multi_gpu=True,
            estimated_vram_gb=6.0, max_resolution=4096,
            notes="Lightweight 3B model with fp8 quantization. Good balance of speed and quality."
        ),
        SeedVR2Model(
            name="seedvr2_ema_3b_fp16.safetensors",
            size="3B", precision="fp16", variant="standard",
            preferred_attention="flash_attn", supports_multi_gpu=True,
            estimated_vram_gb=8.0, max_resolution=4096,
            notes="Standard 3B model with fp16 precision. Recommended for 8GB+ VRAM."
        ),
        # GGUF 3B models - quantized, limited features
        SeedVR2Model(
            name="seedvr2_ema_3b-Q8_0.gguf",
            size="3B", precision="Q8_0", variant="standard",
            compile_compatible=False, supports_multi_gpu=False,
            supports_cache_dit=False, supports_cache_vae=False,
            estimated_vram_gb=4.0, max_resolution=2160,
            notes="GGUF Q8_0 quantized. Low VRAM but no torch.compile or multi-GPU support."
        ),
        SeedVR2Model(
            name="seedvr2_ema_3b-Q4_K_M.gguf",
            size="3B", precision="Q4_K_M", variant="standard",
            compile_compatible=False, supports_multi_gpu=False,
            supports_cache_dit=False, supports_cache_vae=False,
            estimated_vram_gb=3.0, max_resolution=2160,
            notes="GGUF Q4 quantized. Minimal VRAM, quality trade-offs. No compile/multi-GPU."
        ),
        # 7B models - higher quality, more VRAM
        SeedVR2Model(
            name="seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
            size="7B", precision="fp8_e4m3fn_mixed_block35_fp16", variant="standard",
            preferred_attention="flash_attn", supports_multi_gpu=True,
            estimated_vram_gb=12.0, max_resolution=4096,
            default_batch_size=5, max_blocks_to_swap=35,
            notes="Mixed precision 7B. Balances VRAM usage and quality. 12GB+ recommended."
        ),
        SeedVR2Model(
            name="seedvr2_ema_7b_fp16.safetensors",
            size="7B", precision="fp16", variant="standard",
            preferred_attention="flash_attn", supports_multi_gpu=True,
            estimated_vram_gb=16.0, max_resolution=4096,
            default_batch_size=5,
            notes="Full fp16 7B model. Highest quality, requires 16GB+ VRAM."
        ),
        SeedVR2Model(
            name="seedvr2_ema_7b-Q4_K_M.gguf",
            size="7B", precision="Q4_K_M", variant="standard",
            compile_compatible=False, supports_multi_gpu=False,
            supports_cache_dit=False, supports_cache_vae=False,
            estimated_vram_gb=5.0, max_resolution=2160,
            notes="GGUF Q4 7B. Reduced VRAM for 7B quality, but no compile/multi-GPU."
        ),
        SeedVR2Model(
            name="seedvr2_ema_7b-Q8_0.gguf",
            size="7B", precision="Q8_0", variant="standard",
            compile_compatible=False, supports_multi_gpu=False,
            supports_cache_dit=False, supports_cache_vae=False,
            estimated_vram_gb=8.0, max_resolution=2160,
            notes="GGUF Q8_0 7B. Better quality than Q4 with moderate VRAM, no compile/multi-GPU."
        ),
        # Sharp variants - enhanced edge preservation
        SeedVR2Model(
            name="seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
            size="7B", precision="fp8_e4m3fn_mixed_block35_fp16", variant="sharp",
            preferred_attention="flash_attn", supports_multi_gpu=True,
            estimated_vram_gb=12.0, max_resolution=4096,
            default_batch_size=5, max_blocks_to_swap=35,
            notes="Sharp variant with enhanced edge detail. Mixed precision for efficiency."
        ),
        SeedVR2Model(
            name="seedvr2_ema_7b_sharp_fp16.safetensors",
            size="7B", precision="fp16", variant="sharp",
            preferred_attention="flash_attn", supports_multi_gpu=True,
            estimated_vram_gb=16.0, max_resolution=4096,
            default_batch_size=5,
            notes="Sharp variant fp16. Maximum quality with edge enhancement. 16GB+ VRAM."
        ),
        SeedVR2Model(
            name="seedvr2_ema_7b_sharp-Q4_K_M.gguf",
            size="7B", precision="Q4_K_M", variant="sharp",
            compile_compatible=False, supports_multi_gpu=False,
            supports_cache_dit=False, supports_cache_vae=False,
            estimated_vram_gb=5.0, max_resolution=2160,
            notes="GGUF Q4 sharp variant. Low VRAM sharp quality, limited features."
        ),
        SeedVR2Model(
            name="seedvr2_ema_7b_sharp-Q8_0.gguf",
            size="7B", precision="Q8_0", variant="sharp",
            compile_compatible=False, supports_multi_gpu=False,
            supports_cache_dit=False, supports_cache_vae=False,
            estimated_vram_gb=8.0, max_resolution=2160,
            notes="GGUF Q8_0 sharp variant. Strong detail with lower VRAM, no compile/multi-GPU."
        ),
    ]


_SEED_BUILT_INS = _built_ins()


def _scan_local_weights() -> List[str]:
    """
    Discover locally downloaded SeedVR2 weights to keep dropdowns in sync with the filesystem.
    Searches common local weight locations relative to the app root:
    - ./SeedVR2/models (bundled CLI repo folder in this project)
    - ./models/SEEDVR2 (common shared models folder)
    - ./models/SeedVR2 / ./models/seedvr2 (legacy/case variants)
    """
    global _SEED_SCAN_CACHE
    try:
        root = Path(__file__).resolve().parents[2]
        candidates = [
            # Bundled SeedVR2 repo inside this app (default used by seedvr2_defaults()).
            root / "SeedVR2" / "models",

            # Shared/global models folder (various case conventions).
            root / "models" / "SEEDVR2",
            root / "models" / "seedvr2",
            root / "models" / "SeedVR2",
        ]
        fingerprint: list[tuple[str, float]] = []
        for candidate in candidates:
            try:
                if candidate.exists():
                    fingerprint.append((str(candidate.resolve()), float(candidate.stat().st_mtime)))
            except Exception:
                continue
        fingerprint_key = tuple(sorted(fingerprint))
        now = time.time()
        with _SEED_SCAN_CACHE_LOCK:
            ts, cached_fingerprint, cached_models = _SEED_SCAN_CACHE
            if (now - ts) < _SEED_SCAN_CACHE_TTL_SEC and cached_fingerprint == fingerprint_key:
                return list(cached_models)

        found: List[str] = []
        for base in candidates:
            if not base.exists():
                continue
            for f in base.iterdir():
                if f.is_file() and f.suffix.lower() in MODEL_EXTS:
                    found.append(f.name)
        result = sorted(list({*found}))
        with _SEED_SCAN_CACHE_LOCK:
            _SEED_SCAN_CACHE = (now, fingerprint_key, result)
        return result
    except Exception:
        return []


def get_seedvr2_models() -> List[SeedVR2Model]:
    disk = _scan_local_weights()
    built = _SEED_BUILT_INS
    # Add disk-only models with conservative defaults
    known_names = {m.name for m in built}
    extras = [
        SeedVR2Model(name=name, size="unknown", precision="unknown", variant="custom", supports_multi_gpu=True)
        for name in disk
        if name not in known_names
    ]
    return built + extras


def get_seedvr2_model_names() -> List[str]:
    """
    Get list of available SeedVR2 model names.
    Filtered to show only preferred production-ready models.
    """
    # Preferred models to display (in order of preference)
    preferred_models = [
        "seedvr2_ema_7b_fp16.safetensors",  # Default
        "seedvr2_ema_7b_sharp_fp16.safetensors",
        "seedvr2_ema_3b_fp16.safetensors",
        "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
        "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
        "seedvr2_ema_3b-Q8_0.gguf",
        "seedvr2_ema_7b-Q8_0.gguf",
        "seedvr2_ema_7b_sharp-Q8_0.gguf",
    ]
    
    # Get all available models
    all_models = [m.name for m in get_seedvr2_models()]
    
    # Filter to only show preferred models that exist
    filtered_models = [model for model in preferred_models if model in all_models]
    
    # If no preferred models found, fall back to all models
    return filtered_models if filtered_models else all_models


def model_meta_map() -> Dict[str, SeedVR2Model]:
    return {m.name: m for m in get_seedvr2_models()}
