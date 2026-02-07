"""
Helpers for queue job state isolation.

Queued jobs must run against the settings snapshot captured when the user
clicked process, while preserving current live UI settings for later jobs.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


_RUNTIME_TOP_LEVEL_KEYS = {
    "operation_status",
    "alerts",
    "health_banner",
}

_RUNTIME_SEED_KEYS = {
    "_original_filename",
    "current_model",
    "gan_chunk_preview",
    "flashvsr_chunk_preview",
    "rife_chunk_preview",
    "seedvr2_chunk_preview",
    "chunk_gallery_items",
    "chunk_thumbnails",
    "chunk_video_paths",
    "processed_chunks_dir",
    "last_run_dir",
    "last_video_probe",
    "last_scene_scan",
    "resolution_cache",
}


def snapshot_queue_state(state: Any) -> Dict[str, Any]:
    """Deep-copy the shared state at click time for a queued job."""
    if isinstance(state, dict):
        try:
            return deepcopy(state)
        except Exception:
            pass
    return {"seed_controls": {}, "operation_status": "ready"}


def snapshot_global_settings(settings: Any) -> Dict[str, Any]:
    """Deep-copy global settings at click time for a queued job."""
    if isinstance(settings, dict):
        try:
            return deepcopy(settings)
        except Exception:
            try:
                return dict(settings)
            except Exception:
                pass
    return {}


def _is_runtime_seed_key(key: str) -> bool:
    if not key:
        return False
    if key in _RUNTIME_SEED_KEYS:
        return True
    if key.startswith("last_"):
        return True
    if key.endswith("_chunk_preview"):
        return True
    return False


def merge_runtime_state(live_state: Any, worker_state: Any) -> Dict[str, Any]:
    """
    Merge only runtime outputs from worker_state into the live shared state.

    This keeps current UI/user settings intact while still updating runtime
    pointers like last outputs, chunk preview payloads, and operation status.
    """
    live = live_state if isinstance(live_state, dict) else {}
    if not isinstance(worker_state, dict):
        return live

    for key in _RUNTIME_TOP_LEVEL_KEYS:
        if key in worker_state:
            try:
                live[key] = deepcopy(worker_state[key])
            except Exception:
                live[key] = worker_state[key]

    worker_seed = worker_state.get("seed_controls", {})
    if not isinstance(worker_seed, dict):
        return live

    live_seed = live.get("seed_controls", {})
    if not isinstance(live_seed, dict):
        live_seed = {}
        live["seed_controls"] = live_seed

    for key, value in worker_seed.items():
        if _is_runtime_seed_key(str(key)):
            try:
                live_seed[key] = deepcopy(value)
            except Exception:
                live_seed[key] = value

    return live


def merge_payload_state(payload: Any, live_state: Any) -> Any:
    """Replace payload's trailing state with merged live runtime state."""
    if isinstance(payload, tuple) and payload:
        merged = merge_runtime_state(live_state, payload[-1])
        return (*payload[:-1], merged)
    return payload

