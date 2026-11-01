"""
RTX Super Resolution Auto Tune.

NVIDIA's VFX SDK (TensorRT) upscales one frame per call; it has no batch, tile or precision
setting, so the only quality lever is the quality level and the VRAM it needs grows with the
output size. Auto Tune keeps the family the user picked (standard upscale, High Bitrate, Denoise,
Deblur) and selects its highest level whose VRAM fits (GPU total - in use - Save VRAM). The SDK
allocates outside PyTorch, so no allocator cap can stop a spill: the run itself checks the same
estimate and refuses to start instead of using shared GPU memory.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from .common import TunePlan, VramBudget, resolve_budget

LEVELS = ("ULTRA", "HIGH", "MEDIUM", "LOW")
FAMILIES = ("HIGHBITRATE", "DENOISE", "DEBLUR", "STREAMING")
# Families whose SDK level set differs from LEVELS (nvidia-vfx 0.2: STREAMING has only these two).
FAMILY_LEVELS = {"STREAMING": ("ULTRA", "MEDIUM")}

# Whole-process GB = base + per megapixel of model output + per megapixel of model input, measured
# in a separate process with the Windows per-process counters (benchmarks/vram/rtx.jsonl): 0.75 GB
# for 480p -> 1080p, 1.7 GB for 1080p -> 4K, 3.0 GB for 1080p -> 8K (High Bitrate Ultra, the
# largest level; Ultra/Low measured 0.2-0.4 GB lower). The base includes the CUDA context.
_BASE_GB = {"ULTRA": 0.6, "HIGH": 0.6, "MEDIUM": 0.6, "LOW": 0.6, "BICUBIC": 0.5}
_GB_PER_OUT_MP = {"ULTRA": 0.06, "HIGH": 0.06, "MEDIUM": 0.06, "LOW": 0.06, "BICUBIC": 0.04}
_GB_PER_IN_MP = 0.30
PROCESS_OVERHEAD_GB = 0.0  # included in the measured base
SAFETY = 1.15
# nvidia-vfx 0.2.0.0 re-measurement (2026-11, benchmarks/vram/rtx.jsonl "v020_*"): the levels above are
# unchanged (1080p -> 4K HBU 1.68 GB, -> 8K 2.98 GB); STREAMING_ULTRA = HBU, STREAMING_MEDIUM 1.30 GB.
# TrueHDR at the output size adds 0.44 GB at 4K and the 10-bit (RGB10A2) path 0.09 GB.
_HDR_GB_PER_OUT_MP = 0.06
_TEN_BIT_GB_PER_OUT_MP = 0.015


def extra_gb_per_out_mp(settings: Optional[Dict[str, Any]]) -> float:
    s = settings or {}

    def on(key: str) -> bool:
        return str(s.get(key, False)).strip().lower() in {"1", "true", "yes", "on"}

    return (_HDR_GB_PER_OUT_MP if on("rtx_hdr_output") else 0.0) + (_TEN_BIT_GB_PER_OUT_MP if on("rtx_10bit") else 0.0)


def family_of(quality: str) -> str:
    q = str(quality or "").strip().upper()
    for fam in FAMILIES:
        if q.startswith(fam + "_"):
            return fam
    return ""


def ladder(quality: str) -> List[str]:
    fam = family_of(quality)
    if fam:
        return [f"{fam}_{lvl}" for lvl in FAMILY_LEVELS.get(fam, LEVELS)]
    return list(LEVELS) + ["BICUBIC"]


def _level(quality: str) -> str:
    q = str(quality or "").strip().upper()
    return q.split("_", 1)[1] if family_of(q) else (q if q in _BASE_GB else "ULTRA")


def estimate_gb(quality: str, in_w: int, in_h: int, out_w: int, out_h: int, extra_per_out_mp: float = 0.0) -> float:
    lvl = _level(quality)
    out_mp = max(1, out_w) * max(1, out_h) / 1e6
    in_mp = max(1, in_w) * max(1, in_h) / 1e6
    return (_BASE_GB[lvl] + (_GB_PER_OUT_MP[lvl] + extra_per_out_mp) * out_mp + _GB_PER_IN_MP * in_mp) * SAFETY + PROCESS_OVERHEAD_GB


def choose(quality: str, in_w: int, in_h: int, out_w: int, out_h: int,
           budget: Optional[VramBudget], extra_per_out_mp: float = 0.0) -> Tuple[Optional[str], float]:
    """Highest level of the user's family that fits; (None, need) when even the lowest does not."""
    need = 0.0
    for q in ladder(quality):
        need = estimate_gb(q, in_w, in_h, out_w, out_h, extra_per_out_mp)
        if budget is None or need <= budget.limit_gb:
            return q, need
    return None, need


def plan(settings: Dict[str, Any], *, in_w: int, in_h: int, model_out_w: int, model_out_h: int,
         budget: Optional[VramBudget]) -> TunePlan:
    out = TunePlan(budget=budget)
    current = str(settings.get("quality_preset") or "HIGHBITRATE_ULTRA").upper()
    picked, need = choose(current, in_w, in_h, model_out_w, model_out_h, budget, extra_gb_per_out_mp(settings))
    fam = family_of(current) or "standard upscale"
    out.lines.append(f"Model output {model_out_w}x{model_out_h} from {in_w}x{in_h} ({fam.title()} family)")
    if picked is None:
        out.fits = False
        out.warnings.append(f"Even the lowest level needs about {need:.1f} GB but only "
                            f"{budget.limit_gb if budget else 0:.1f} GB is available. Lower Upscale x or Max "
                            "Resolution, close other GPU apps or lower Save VRAM.")
        return out
    out.updates = {"quality_preset": picked}
    out.estimate_gb = need
    best = ladder(current)[0]
    out.lines.append(f"Quality preset {picked}" + (" (highest level)" if picked == best else " (highest level that fits)"))
    return out


def check_or_pick(settings: Dict[str, Any], *, in_w: int, in_h: int, model_out_w: int, model_out_h: int,
                  log: Optional[Callable[[str], None]] = None) -> Tuple[bool, str]:
    """Runner guard (every run): with Auto Tune on, pick the best level that fits; always refuse to
    start when nothing fits, because the SDK would otherwise spill into shared GPU memory."""
    budget = resolve_budget(settings.get("device"), settings.get("save_vram_gb"), overhead_gb=0.0)
    auto = str(settings.get("auto_tune_on_run", True)).strip().lower() not in {"false", "0", "no", "off"}
    current = str(settings.get("quality_preset") or "ULTRA").upper()
    if auto:
        tp = plan(settings, in_w=in_w, in_h=in_h, model_out_w=model_out_w, model_out_h=model_out_h, budget=budget)
        if not tp.fits:
            return False, " ".join(tp.warnings)
        settings.update(tp.updates)
        if log:
            log(f"[RTX Auto Tune] {tp.updates['quality_preset']} for {model_out_w}x{model_out_h}, about "
                f"{tp.estimate_gb:.1f} GB" + (f" of {budget.limit_gb:.1f} GB available" if budget else ""))
        return True, ""
    need = estimate_gb(current, in_w, in_h, model_out_w, model_out_h, extra_gb_per_out_mp(settings))
    if budget is not None and need > budget.limit_gb:
        return False, (f"{current} at {model_out_w}x{model_out_h} needs about {need:.1f} GB but only "
                       f"{budget.limit_gb:.1f} GB is free after Save VRAM; it would spill into shared GPU memory. "
                       "Enable Auto Tune on every Upscale, pick a lower quality level, lower the output size or "
                       "lower Save VRAM.")
    return True, ""
