"""
SparkVSR metadata registry and local model discovery.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SparkVSRModel:
    name: str
    repo_id: str
    relative_path: str
    stage: str
    estimated_vram_gb: float
    default_dtype: str = "bfloat16"
    default_scale: int = 4
    default_chunk_len: int = 0
    default_overlap_t: int = 8
    default_tile_height: int = 0
    default_tile_width: int = 0
    default_overlap_height: int = 32
    default_overlap_width: int = 32
    notes: str = ""


_BUILTIN_MODELS: List[SparkVSRModel] = [
    SparkVSRModel(
        name="SparkVSR-S2",
        repo_id="JiongzeYu/SparkVSR",
        relative_path="SparkVSR",
        stage="Stage-2 final",
        estimated_vram_gb=18.0,
        default_dtype="bfloat16",
        default_scale=4,
        notes="Official Stage-2 final SparkVSR model. Recommended default.",
    ),
    SparkVSRModel(
        name="SparkVSR-S1",
        repo_id="JiongzeYu/SparkVSR-S1",
        relative_path="SparkVSR-S1",
        stage="Stage-1 intermediate",
        estimated_vram_gb=18.0,
        default_dtype="bfloat16",
        default_scale=4,
        notes="Official Stage-1 checkpoint, mainly useful for comparisons.",
    ),
]

_SCAN_CACHE_TTL_SEC = 20.0
_SCAN_LOCK = Lock()
_SCAN_CACHE: tuple[float, tuple[tuple[str, float], ...], List[SparkVSRModel]] | None = None


def sparkvsr_version_to_internal(version: object) -> str:
    raw = str(version or "").strip().lower()
    if raw in {"s1", "stage1", "stage-1", "1", "sparkvsr-s1"}:
        return "s1"
    return "s2"


def sparkvsr_version_to_ui(version: object) -> str:
    return "Stage-1" if sparkvsr_version_to_internal(version) == "s1" else "Stage-2"


def _candidate_roots(base_dir: Optional[Path] = None) -> List[Path]:
    roots: List[Path] = []
    if base_dir is not None:
        base = Path(base_dir)
    else:
        base = Path(__file__).resolve().parents[2]
    roots.extend(
        [
            base / "SparkVSR" / "models",
            base / "models" / "SparkVSR",
            base / "models" / "sparkvsr",
        ]
    )
    return roots


def _fingerprint(paths: List[Path]) -> tuple[tuple[str, float], ...]:
    items: List[tuple[str, float]] = []
    for path in paths:
        try:
            items.append((str(path.resolve()), float(path.stat().st_mtime)))
        except Exception:
            items.append((str(path), 0.0))
    return tuple(sorted(items))


def _is_diffusers_model_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "model_index.json").exists()
        and (path / "transformer").is_dir()
        and (path / "vae").is_dir()
    )


def discover_sparkvsr_model_paths(base_dir: Optional[Path] = None) -> List[Path]:
    found: List[Path] = []
    for root in _candidate_roots(base_dir):
        if _is_diffusers_model_dir(root):
            found.append(root)
        if root.exists() and root.is_dir():
            for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
                if _is_diffusers_model_dir(child):
                    found.append(child)
    seen = set()
    unique: List[Path] = []
    for item in found:
        key = str(item.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def get_sparkvsr_models(base_dir: Optional[Path] = None) -> List[SparkVSRModel]:
    global _SCAN_CACHE
    roots = _candidate_roots(base_dir)
    fp = _fingerprint(roots)
    now = time.time()
    with _SCAN_LOCK:
        if _SCAN_CACHE is not None:
            ts, cached_fp, cached_models = _SCAN_CACHE
            if (now - ts) < _SCAN_CACHE_TTL_SEC and cached_fp == fp:
                return list(cached_models)

    models: List[SparkVSRModel] = list(_BUILTIN_MODELS)
    builtin_names = {m.name for m in models}
    for path in discover_sparkvsr_model_paths(base_dir):
        if path.name in {"SparkVSR", "SparkVSR-S1"}:
            continue
        name = path.name
        if name in builtin_names:
            continue
        models.append(
            SparkVSRModel(
                name=name,
                repo_id="local",
                relative_path=str(path),
                stage="Local",
                estimated_vram_gb=18.0,
                notes=f"Local SparkVSR diffusers model at {path}",
            )
        )

    with _SCAN_LOCK:
        _SCAN_CACHE = (now, fp, list(models))
    return models


def get_sparkvsr_model_names(base_dir: Optional[Path] = None) -> List[str]:
    return [m.name for m in get_sparkvsr_models(base_dir)]


def get_sparkvsr_default_model() -> str:
    return "SparkVSR-S2"


def get_sparkvsr_metadata(model_name: str, base_dir: Optional[Path] = None) -> Optional[SparkVSRModel]:
    return sparkvsr_model_map(base_dir).get(str(model_name or ""))


def sparkvsr_model_map(base_dir: Optional[Path] = None) -> Dict[str, SparkVSRModel]:
    return {m.name: m for m in get_sparkvsr_models(base_dir)}

