"""VRAM-budget Auto Tune: instant per-model settings for (GPU total - in use - Save VRAM)."""

from .common import (  # noqa: F401
    DEFAULT_SAVE_VRAM_GB,
    SAVE_VRAM_INFO,
    SAVE_VRAM_LABEL,
    SAVE_VRAM_MAX_GB,
    SAVE_VRAM_MIN_GB,
    SAVE_VRAM_STEP_GB,
    VramBudget,
    apply_vram_cap_env,
    clamp_save_vram_gb,
    query_gpu_memory,
    resolve_budget,
)
