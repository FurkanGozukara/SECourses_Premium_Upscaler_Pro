"""
Shared model discovery and metadata functions.
"""

from pathlib import Path
from threading import Lock
import time
from typing import List, Tuple


GAN_MODEL_EXTS = (".pth", ".safetensors")
_GAN_SCAN_CACHE_TTL_SEC = 20.0
_GAN_SCAN_CACHE_LOCK = Lock()
_GAN_SCAN_CACHE: dict[str, tuple[float, Tuple[Tuple[str, float], ...], List[str]]] = {}


def _iter_gan_model_dirs(base_dir: Path) -> List[Path]:
    """
    Return model directories that may contain GAN / image upscaler weights.

    This project historically used `Image_Upscale_Models/`, but newer layouts store
    weights under `models/`. We support both for backwards compatibility.
    """
    dirs: List[Path] = []
    for folder_name in ("models", "Image_Upscale_Models"):
        d = base_dir / folder_name
        if d.exists() and d.is_dir():
            dirs.append(d)
    return dirs


def _gan_dirs_fingerprint(model_dirs: List[Path]) -> Tuple[Tuple[str, float], ...]:
    fingerprint: list[tuple[str, float]] = []
    for model_dir in model_dirs:
        try:
            mtime = float(model_dir.stat().st_mtime)
        except Exception:
            mtime = 0.0
        fingerprint.append((str(model_dir.resolve()), mtime))
    return tuple(sorted(fingerprint))


def scan_gan_models(base_dir: Path) -> List[str]:
    """
    Scan for GAN / image upscaler model weights.
    
    Args:
        base_dir: App base directory
        
    Returns:
        Sorted list of model filenames
    """
    model_dirs = _iter_gan_model_dirs(base_dir)
    fingerprint = _gan_dirs_fingerprint(model_dirs)
    cache_key = str(base_dir.resolve())
    now = time.time()

    with _GAN_SCAN_CACHE_LOCK:
        cached = _GAN_SCAN_CACHE.get(cache_key)
        if cached:
            ts, cached_fingerprint, cached_models = cached
            if (now - ts) < _GAN_SCAN_CACHE_TTL_SEC and cached_fingerprint == fingerprint:
                return list(cached_models)

    choices: set[str] = set()
    for models_dir in model_dirs:
        try:
            for f in models_dir.iterdir():
                if f.is_file() and f.suffix.lower() in GAN_MODEL_EXTS:
                    choices.add(f.name)
        except Exception:
            # Ignore unreadable dirs; keep scanning others.
            continue
    result = sorted(choices)
    with _GAN_SCAN_CACHE_LOCK:
        _GAN_SCAN_CACHE[cache_key] = (now, fingerprint, result)
    return result


# Re-export model metadata functions for convenience
from .seedvr2_meta import get_seedvr2_model_names, get_seedvr2_models, model_meta_map
from .flashvsr_meta import (
    get_flashvsr_model_names, 
    get_flashvsr_default_model,
    get_flashvsr_metadata,
    flashvsr_model_map
)
from .rife_meta import (
    get_rife_model_names, 
    get_rife_default_model,
    get_rife_metadata,
    rife_model_map
)


__all__ = [
    "scan_gan_models",
    "get_seedvr2_model_names",
    "get_seedvr2_models", 
    "model_meta_map",
    "get_flashvsr_model_names",
    "get_flashvsr_default_model",
    "get_flashvsr_metadata",
    "flashvsr_model_map",
    "get_rife_model_names",
    "get_rife_default_model",
    "get_rife_metadata",
    "rife_model_map"
]
