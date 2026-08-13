"""
LTX 2.5 Upscaler metadata registry.

Mirrors shared/models/sparkvsr_meta.py. All variants are fixed 2x pixel
spatial video upscalers (IC-LoRA on the 22B DiT); they differ only in
transformer quantization, download size, VRAM class, and sampling defaults.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from shared.ltx25_constants import (
    LTX25_DEV_BF16,
    LTX25_DEV_INT4,
    LTX25_DEV_INT8,
    LTX25_DISTILLED_BF16,
    LTX25_DISTILLED_INT4,
    LTX25_DISTILLED_INT8,
    LTX25_DISTILLED_NVFP4,
    LTX25_MODEL_NAMES,
    LTX25_MODELS_DIRNAME,
    LTX25_TRANSFORMER_FILES,
    ltx25_is_distilled,
)

# Download sources (kept in sync with the root Models_Downloader.py).
LTX25_REPO_MAIN = "MonsterMMORPG/Wan_GGUF"
LTX25_REPO_INT4 = "tsolful/LTX_2.5_INT4_W4A8_ConvRot"


@dataclass(frozen=True)
class Ltx25Model:
    name: str
    transformer_file: str
    repo_id: str
    estimated_vram_gb: float
    size_gb: float
    is_distilled: bool
    default_steps: int
    default_cfg: float
    default_sampler: str
    schedule: str
    notes: str = ""


_BUILTIN_MODELS: List[Ltx25Model] = [
    Ltx25Model(
        name=LTX25_DISTILLED_INT8,
        transformer_file=LTX25_TRANSFORMER_FILES[LTX25_DISTILLED_INT8],
        repo_id=LTX25_REPO_MAIN,
        estimated_vram_gb=22.0,
        size_gb=21.5,
        is_distilled=True,
        default_steps=8,
        default_cfg=1.0,
        default_sampler="euler_ancestral",
        schedule="distilled",
        notes=(
            "Recommended default. ComfyUI-parity INT8 ConvRot kernels (Hadamard "
            "rotation + MSE-optimized scales) give the best speed/quality balance."
        ),
    ),
    Ltx25Model(
        name=LTX25_DISTILLED_NVFP4,
        transformer_file=LTX25_TRANSFORMER_FILES[LTX25_DISTILLED_NVFP4],
        repo_id=LTX25_REPO_MAIN,
        estimated_vram_gb=19.0,
        size_gb=18.7,
        is_distilled=True,
        default_steps=8,
        default_cfg=1.0,
        default_sampler="euler_ancestral",
        schedule="distilled",
        notes=(
            "Native FP4 on RTX 50 series (Blackwell); emulated on older GPUs "
            "(same quality, slower)."
        ),
    ),
    Ltx25Model(
        name=LTX25_DISTILLED_INT4,
        transformer_file=LTX25_TRANSFORMER_FILES[LTX25_DISTILLED_INT4],
        repo_id=LTX25_REPO_INT4,
        estimated_vram_gb=16.0,
        size_gb=15.4,
        is_distilled=True,
        default_steps=8,
        default_cfg=1.0,
        default_sampler="euler_ancestral",
        schedule="distilled",
        notes="Smallest download. INT4 weights with W4A8 mixed activation quantization.",
    ),
    Ltx25Model(
        name=LTX25_DISTILLED_BF16,
        transformer_file=LTX25_TRANSFORMER_FILES[LTX25_DISTILLED_BF16],
        repo_id=LTX25_REPO_MAIN,
        estimated_vram_gb=43.0,
        size_gb=42.0,
        is_distilled=True,
        default_steps=8,
        default_cfg=1.0,
        default_sampler="euler_ancestral",
        schedule="distilled",
        notes=(
            "Highest fidelity distilled variant. Needs weight streaming on GPUs "
            "with less than 48GB VRAM."
        ),
    ),
    Ltx25Model(
        name=LTX25_DEV_INT8,
        transformer_file=LTX25_TRANSFORMER_FILES[LTX25_DEV_INT8],
        repo_id=LTX25_REPO_MAIN,
        estimated_vram_gb=22.0,
        size_gb=21.5,
        is_distilled=False,
        default_steps=20,
        default_cfg=3.0,
        default_sampler="euler_ancestral",
        schedule="dev",
        notes="Quality base (Dev) model with INT8 ConvRot kernels; 20-step CFG sampling.",
    ),
    Ltx25Model(
        name=LTX25_DEV_INT4,
        transformer_file=LTX25_TRANSFORMER_FILES[LTX25_DEV_INT4],
        repo_id=LTX25_REPO_INT4,
        estimated_vram_gb=16.0,
        size_gb=15.4,
        is_distilled=False,
        default_steps=20,
        default_cfg=3.0,
        default_sampler="euler_ancestral",
        schedule="dev",
        notes="Dev (quality) base model in the smallest INT4 W4A8 download.",
    ),
    Ltx25Model(
        name=LTX25_DEV_BF16,
        transformer_file=LTX25_TRANSFORMER_FILES[LTX25_DEV_BF16],
        repo_id=LTX25_REPO_MAIN,
        estimated_vram_gb=43.0,
        size_gb=42.0,
        is_distilled=False,
        default_steps=20,
        default_cfg=3.0,
        default_sampler="euler_ancestral",
        schedule="dev",
        notes=(
            "Maximum quality. Full BF16 Dev transformer; needs weight streaming "
            "on GPUs with less than 48GB VRAM."
        ),
    ),
]

_SCAN_CACHE_TTL_SEC = 20.0
_SCAN_LOCK = Lock()
_SCAN_CACHE: tuple[float, tuple[tuple[str, float], ...], List[Ltx25Model]] | None = None


def _candidate_roots(base_dir: Optional[Path] = None) -> List[Path]:
    if base_dir is not None:
        base = Path(base_dir)
    else:
        base = Path(__file__).resolve().parents[2]
    return [base / LTX25_MODELS_DIRNAME]


def _fingerprint(paths: List[Path]) -> tuple[tuple[str, float], ...]:
    items: List[tuple[str, float]] = []
    for path in paths:
        try:
            items.append((str(path.resolve()), float(path.stat().st_mtime)))
        except Exception:
            items.append((str(path), 0.0))
    return tuple(sorted(items))


def discover_ltx25_model_paths(base_dir: Optional[Path] = None) -> List[Path]:
    """Return local transformer checkpoints that already exist on disk."""
    found: List[Path] = []
    for root in _candidate_roots(base_dir):
        if not (root.exists() and root.is_dir()):
            continue
        for model in _BUILTIN_MODELS:
            candidate = root / model.transformer_file
            if candidate.is_file():
                found.append(candidate)
    seen = set()
    unique: List[Path] = []
    for item in found:
        key = str(item.resolve()) if item.exists() else str(item)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def get_ltx25_models(base_dir: Optional[Path] = None) -> List[Ltx25Model]:
    global _SCAN_CACHE
    roots = _candidate_roots(base_dir)
    fp = _fingerprint(roots)
    now = time.time()
    with _SCAN_LOCK:
        if _SCAN_CACHE is not None:
            ts, cached_fp, cached_models = _SCAN_CACHE
            if (now - ts) < _SCAN_CACHE_TTL_SEC and cached_fp == fp:
                return list(cached_models)

    models: List[Ltx25Model] = list(_BUILTIN_MODELS)

    with _SCAN_LOCK:
        _SCAN_CACHE = (now, fp, list(models))
    return models


def get_ltx25_model_names(base_dir: Optional[Path] = None) -> List[str]:
    return [m.name for m in get_ltx25_models(base_dir)]


def get_ltx25_default_model() -> str:
    # User requirement: INT8 ConvRot is the recommended default on ALL platforms.
    return LTX25_DISTILLED_INT8


def get_ltx25_metadata(model_name: str, base_dir: Optional[Path] = None) -> Optional[Ltx25Model]:
    return ltx25_model_map(base_dir).get(str(model_name or ""))


def ltx25_model_map(base_dir: Optional[Path] = None) -> Dict[str, Ltx25Model]:
    return {m.name: m for m in get_ltx25_models(base_dir)}


def ltx25_sampling_defaults(model_name: str) -> Dict[str, object]:
    """
    Sampling defaults for a variant, used by the service guardrails and the
    tab's model-switch handler (steps/cfg/sampler stay user-editable after).
    """
    meta = get_ltx25_metadata(model_name)
    if meta is None:
        name = str(model_name or "")
        if name in LTX25_MODEL_NAMES and not ltx25_is_distilled(name):
            return {"steps": 20, "cfg": 3.0, "sampler": "euler_ancestral", "schedule": "dev"}
        return {"steps": 8, "cfg": 1.0, "sampler": "euler_ancestral", "schedule": "distilled"}
    return {
        "steps": int(meta.default_steps),
        "cfg": float(meta.default_cfg),
        "sampler": str(meta.default_sampler),
        "schedule": str(meta.schedule),
    }
