"""
RIFE Model Metadata Registry
Provides comprehensive model metadata with VRAM, compile, and multi-GPU constraints
"""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import time
from typing import Dict, List, Optional


_RIFE_DISCOVERY_CACHE_TTL_SEC = 20.0
_RIFE_DISCOVERY_CACHE_LOCK = Lock()
_RIFE_DISCOVERY_CACHE: dict[str, tuple[float, tuple[tuple[str, float], ...], List[str]]] = {}


@dataclass
class RifeModel:
    """Comprehensive metadata for RIFE frame interpolation models"""
    name: str  # e.g., "rife-v4.17"
    version: str  # "4.6", "4.17", etc.
    variant: str  # "standard" or "anime"
    
    # Multi-GPU support
    supports_multi_gpu: bool = False  # RIFE is single-GPU optimized
    
    # Performance characteristics
    estimated_vram_gb: float = 4.0
    supports_fp16: bool = True
    supports_uhd: bool = True  # UHD mode for 4K+
    default_precision: str = "fp16"
    
    # Compilation support
    compile_compatible: bool = True  # RIFE supports torch.compile
    preferred_compile_backend: str = "inductor"
    
    # Processing capabilities
    supports_img_mode: bool = True  # Can interpolate between two images
    max_fps_multiplier: int = 8  # Maximum safe FPS multiplication
    
    # Resolution constraints
    max_resolution: int = 0  # 0 = no cap (but UHD mode recommended for 4K+)
    min_resolution: int = 64
    recommended_uhd_threshold: int = 2160  # Enable UHD mode for 4K+
    
    # Quality settings
    recommended_scale: float = 1.0  # Spatial scale (1.0 = no scaling)
    
    # Description
    notes: str = ""


def _get_rife_models() -> List[RifeModel]:
    """
    Define all known RIFE model configurations with comprehensive metadata.
    
    RIFE versions progress with improved temporal consistency and quality.
    Latest versions (v4.15+) offer best results.
    """
    models = [
        # v4.6 - Stable baseline
        RifeModel(
            name="rife-v4.6",
            version="4.6",
            variant="standard",
            estimated_vram_gb=3.0,
            notes="Stable baseline version. Good for general use."
        ),
        
        # v4.13-v4.15 - Progressive improvements
        RifeModel(
            name="rife-v4.13",
            version="4.13",
            variant="standard",
            estimated_vram_gb=3.5,
            notes="Improved temporal consistency over v4.6."
        ),
        RifeModel(
            name="rife-v4.14",
            version="4.14",
            variant="standard",
            estimated_vram_gb=3.5,
            notes="Enhanced motion estimation, fewer artifacts."
        ),
        RifeModel(
            name="rife-v4.15",
            version="4.15",
            variant="standard",
            estimated_vram_gb=4.0,
            notes="Significant quality improvements, recommended for most use cases."
        ),
        
        # v4.16-v4.17 - Latest stable
        RifeModel(
            name="rife-v4.16",
            version="4.16",
            variant="standard",
            estimated_vram_gb=4.0,
            notes="Latest quality improvements with better scene change handling."
        ),
        RifeModel(
            name="rife-v4.17",
            version="4.17",
            variant="standard",
            estimated_vram_gb=4.0,
            max_fps_multiplier=8,
            notes="Current recommended version. Best quality/performance balance."
        ),
        
        # v4.18 - Experimental/latest
        RifeModel(
            name="rife-v4.18",
            version="4.18",
            variant="standard",
            estimated_vram_gb=4.5,
            notes="Latest experimental version. May have cutting-edge features."
        ),

        RifeModel(
            name="rife-v4.20",
            version="4.20",
            variant="standard",
            estimated_vram_gb=4.5,
            notes="Improved post-processing quality for diffusion-generated videos."
        ),
        RifeModel(
            name="rife-v4.21",
            version="4.21",
            variant="standard",
            estimated_vram_gb=4.5,
            notes="Quality-focused update for temporal consistency."
        ),
        RifeModel(
            name="rife-v4.22",
            version="4.22",
            variant="standard",
            estimated_vram_gb=4.5,
            notes="Further refinement of motion handling and stability."
        ),
        RifeModel(
            name="rife-v4.25",
            version="4.25",
            variant="standard",
            estimated_vram_gb=4.8,
            notes="Modern high-quality model with stronger detail retention."
        ),
        RifeModel(
            name="rife-v4.26",
            version="4.26",
            variant="standard",
            estimated_vram_gb=5.0,
            notes="Recommended latest quality model when available."
        ),
        
        # Anime-specific variant
        RifeModel(
            name="rife-anime",
            version="4.x",
            variant="anime",
            estimated_vram_gb=4.0,
            notes="Specialized for anime content. Better handles hard edges and stylized motion."
        ),
    ]
    
    return models


_RIFE_KNOWN_MODELS = _get_rife_models()
_RIFE_MODEL_MAP = {m.name: m for m in _RIFE_KNOWN_MODELS}
_RIFE_MODEL_MAP_LOWER = {m.name.lower(): m for m in _RIFE_KNOWN_MODELS}


def _resolve_model_bundle_dir(model_dir: Path) -> Optional[Path]:
    """
    Resolve a model bundle directory that contains `flownet.pkl`.

    Supports either:
    - direct layout: <model_dir>/flownet.pkl
    - one-level nested zip layout: <model_dir>/<inner>/flownet.pkl
    """
    if not model_dir.exists() or not model_dir.is_dir():
        return None

    if (model_dir / "flownet.pkl").exists():
        return model_dir

    children = [p for p in model_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if len(children) == 1 and (children[0] / "flownet.pkl").exists():
        return children[0]
    return None


def _discover_rife_models_from_layout(base_dir: Path) -> List[str]:
    """
    Discover installed RIFE models from supported local layouts:
    - RIFE/models/<model_name>/...
    - RIFE/train_log/<model_name>/...
    """
    base_path = Path(base_dir).resolve()
    rife_root = base_path / "RIFE"
    models_dir = rife_root / "models"
    train_log_dir = rife_root / "train_log"
    fingerprint: list[tuple[str, float]] = []
    for scan_dir in (models_dir, train_log_dir):
        try:
            if scan_dir.exists():
                fingerprint.append((str(scan_dir), float(scan_dir.stat().st_mtime)))
        except Exception:
            continue
    cache_key = str(base_path)
    fingerprint_key = tuple(sorted(fingerprint))
    now = time.time()
    with _RIFE_DISCOVERY_CACHE_LOCK:
        cached = _RIFE_DISCOVERY_CACHE.get(cache_key)
        if cached:
            ts, cached_fingerprint, cached_models = cached
            if (now - ts) < _RIFE_DISCOVERY_CACHE_TTL_SEC and cached_fingerprint == fingerprint_key:
                return list(cached_models)

    discovered_models: set[str] = set()
    discovered_train_log: set[str] = set()

    # Preferred new layout: RIFE/models/<name>/...
    if models_dir.exists() and models_dir.is_dir():
        for item in models_dir.iterdir():
            if not item.is_dir() or item.name.startswith("_") or item.name.startswith("."):
                continue
            bundle_dir = _resolve_model_bundle_dir(item)
            if bundle_dir and (bundle_dir / "flownet.pkl").exists():
                discovered_models.add(item.name)

    # If models/ has valid bundles, use it as source-of-truth for dropdowns.
    if discovered_models:
        result = sorted(discovered_models)
        with _RIFE_DISCOVERY_CACHE_LOCK:
            _RIFE_DISCOVERY_CACHE[cache_key] = (now, fingerprint_key, result)
        return result

    # Fallback layout: RIFE/train_log/<name>/...
    if train_log_dir.exists() and train_log_dir.is_dir():
        for item in train_log_dir.iterdir():
            if not item.is_dir() or item.name.startswith("_") or item.name.startswith("."):
                continue
            bundle_dir = _resolve_model_bundle_dir(item)
            if bundle_dir and (bundle_dir / "flownet.pkl").exists():
                discovered_train_log.add(item.name)

    # Intentionally do NOT expose legacy train_log root (no folder) as a selectable model.
    result = sorted(discovered_train_log)
    with _RIFE_DISCOVERY_CACHE_LOCK:
        _RIFE_DISCOVERY_CACHE[cache_key] = (now, fingerprint_key, result)
    return result


def get_rife_model_names(base_dir: Path = None) -> List[str]:
    """
    Get locally installed RIFE model names.
    
    Scans supported local layouts and returns only discovered models.
    
    Args:
        base_dir: Base directory of the application (to find RIFE/train_log)
    
    Returns:
        List of installed model names.
    """
    # User requested behavior: list only installed local models.
    if base_dir:
        return _discover_rife_models_from_layout(Path(base_dir))
    return []


def get_rife_default_model() -> str:
    """Get default RIFE model identifier."""
    return "4.26"


def get_rife_metadata(model_name: str) -> Optional[RifeModel]:
    """
    Get comprehensive metadata for a RIFE model.
    
    Args:
        model_name: Model identifier (e.g., "rife-v4.17")
        
    Returns:
        RifeModel metadata or default metadata for unknown models
    """
    models_map = _RIFE_MODEL_MAP
    models_map_l = _RIFE_MODEL_MAP_LOWER
    text = str(model_name or "").strip()
    text_l = text.lower()

    # Return exact match if found
    if text in models_map:
        return models_map[text]
    if text_l in models_map_l:
        return models_map_l[text_l]

    # Alias support for folder-style names like "4.26", "v4.26", "anime".
    candidates: List[str] = []
    if text_l:
        candidates.append(text_l)
        if text_l.startswith("rife-v"):
            tail = text_l[len("rife-v"):].strip()
            if tail:
                candidates.extend([tail, f"v{tail}"])
        elif text_l.startswith("v") and len(text_l) > 1 and text_l[1].isdigit():
            tail = text_l[1:]
            candidates.extend([tail, f"rife-v{tail}"])
        elif text_l[0].isdigit():
            candidates.extend([f"v{text_l}", f"rife-v{text_l}"])
        elif text_l in {"anime", "rife-anime"}:
            candidates.extend(["anime", "rife-anime"])

    for cand in candidates:
        if cand in models_map_l:
            return models_map_l[cand]
    
    # Fallback for unknown models - conservative defaults
    return RifeModel(
        name=model_name,
        version="unknown",
        variant="standard",
        estimated_vram_gb=4.0,
        notes="Unknown RIFE model - using conservative defaults"
    )


def rife_model_map() -> Dict[str, RifeModel]:
    """Get mapping of model names to metadata."""
    return dict(_RIFE_MODEL_MAP)

