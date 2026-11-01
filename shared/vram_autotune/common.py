"""
Save VRAM budget shared by every upscaler tab (like ComfyUI's ``--reserve-vram``).

Budget for one run = GPU total - VRAM already in use - Save VRAM. Every engine
subprocess is capped at that budget through PyTorch's allocator setting
``per_process_memory_fraction`` (``PYTORCH_ALLOC_CONF``): an allocation past the
cap raises a normal CUDA out-of-memory error instead of the Windows driver
spilling into shared GPU memory (system RAM, about 20x slower).

Everything here reads memory with ``nvidia-smi`` (NVML), so the Gradio process
never creates a CUDA context. ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` (set at app
start) keeps the nvidia-smi index and the CUDA index of a GPU the same.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

GIB = 1024 ** 3

DEFAULT_SAVE_VRAM_GB = 1.5
SAVE_VRAM_MIN_GB = 0.0
SAVE_VRAM_MAX_GB = 64.0
SAVE_VRAM_STEP_GB = 0.1
SAVE_VRAM_LABEL = "Save VRAM (GB)"
SAVE_VRAM_INFO = (
    "Kept free for other apps, like ComfyUI's --reserve-vram. A run uses at most "
    "GPU total - in use - this, never shared GPU memory."
)

# VRAM a CUDA process uses outside the PyTorch allocator: the CUDA context,
# loaded kernels and library handles (measured 0.43 GB for a bare torch 2.13 /
# CUDA 13 process on an RTX 5090; engines with more kernel libraries use more).
DEFAULT_PROCESS_OVERHEAD_GB = 0.6
# The smallest cap ever applied; below this no upscaler can load at all.
MIN_ALLOCATOR_CAP_GB = 1.0

ENV_SAVE_VRAM = "SECOURSES_SAVE_VRAM_GB"
MAX_SPLIT_SIZE_CONF = "max_split_size_mb:512"
ENV_VRAM_LIMIT = "SECOURSES_VRAM_LIMIT_GB"
_FRACTION_RE = re.compile(r"(?:^|,)\s*per_process_memory_fraction\s*:\s*[^,]*")


def clamp_save_vram_gb(value: Any, default: float = DEFAULT_SAVE_VRAM_GB) -> float:
    """Save VRAM value from a slider/preset; None/garbage falls back to the default, 0 stays 0."""
    try:
        raw = float(default if value is None or value == "" else value)
    except Exception:
        raw = float(default)
    if raw != raw:  # NaN
        raw = float(default)
    return round(max(SAVE_VRAM_MIN_GB, min(SAVE_VRAM_MAX_GB, raw)), 1)


@dataclass
class GpuMemory:
    index: int
    name: str
    uuid: str
    total_gb: float
    reserved_gb: float  # driver-reserved pool (not usable by any process)
    used_gb: float      # used by all processes (dedicated)
    free_gb: float

    @property
    def in_use_gb(self) -> float:
        """What the user sees as 'in use': processes plus the driver's reserved pool."""
        return max(0.0, self.total_gb - self.free_gb)


def _nvidia_smi() -> Optional[str]:
    return shutil.which("nvidia-smi")


def query_gpu_memory() -> Dict[int, GpuMemory]:
    """Live (uncached) memory of every NVIDIA GPU, keyed by nvidia-smi index."""
    exe = _nvidia_smi()
    if not exe:
        return {}
    fields_sets = (
        "index,name,uuid,memory.total,memory.reserved,memory.used,memory.free",
        "index,name,uuid,memory.total,memory.used,memory.free",
    )
    for fields in fields_sets:
        try:
            proc = subprocess.run(
                [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except Exception:
            return {}
        if proc.returncode != 0 or not (proc.stdout or "").strip():
            continue
        out: Dict[int, GpuMemory] = {}
        has_reserved = "memory.reserved" in fields
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            try:
                if has_reserved and len(parts) >= 7:
                    idx, name, uuid, total, reserved, used, free = parts[:7]
                elif len(parts) >= 6:
                    idx, name, uuid, total, used, free = parts[:6]
                    reserved = "0"
                else:
                    continue

                def _gb(v: str) -> float:
                    try:
                        return float(v) / 1024.0
                    except Exception:
                        return 0.0

                out[int(idx)] = GpuMemory(
                    index=int(idx), name=name, uuid=uuid,
                    total_gb=_gb(total), reserved_gb=_gb(reserved), used_gb=_gb(used), free_gb=_gb(free),
                )
            except Exception:
                continue
        if out:
            return out
    return {}


def parse_device_ids(device_spec: Any) -> List[int]:
    """'0' / 'cuda:1' / '0,1' -> [ids]; 'cpu'/'none' -> []; ''/'auto' -> [] (caller picks)."""
    raw = str(device_spec or "").strip().lower()
    if not raw or raw in {"cpu", "none", "auto"}:
        return []
    ids: List[int] = []
    for token in raw.replace(";", ",").split(","):
        tok = token.strip()
        if tok.startswith("cuda:"):
            tok = tok[5:].strip()
        if tok.isdigit() and int(tok) not in ids:
            ids.append(int(tok))
    return ids


def is_cpu_device(device_spec: Any) -> bool:
    return str(device_spec or "").strip().lower() in {"cpu", "none"}


@dataclass
class VramBudget:
    """VRAM one run may use on one GPU."""

    gpu: GpuMemory
    save_gb: float
    overhead_gb: float = DEFAULT_PROCESS_OVERHEAD_GB
    notes: List[str] = field(default_factory=list)

    @property
    def limit_gb(self) -> float:
        """Whole-process limit: total - in use - Save VRAM (the user's formula)."""
        return max(0.0, self.gpu.free_gb - self.save_gb)

    @property
    def allocator_cap_gb(self) -> float:
        """What the PyTorch allocator may hold (limit minus context/kernels)."""
        return max(MIN_ALLOCATOR_CAP_GB, self.limit_gb - self.overhead_gb)

    @property
    def fraction(self) -> float:
        if self.gpu.total_gb <= 0:
            return 1.0
        return max(0.01, min(1.0, self.allocator_cap_gb / self.gpu.total_gb))

    def describe(self) -> str:
        g = self.gpu
        return (
            f"GPU {g.index} {g.name}: {g.total_gb:.1f} GB total, {g.in_use_gb:.1f} GB in use, "
            f"Save VRAM {self.save_gb:g} GB -> this run may use up to {self.limit_gb:.1f} GB"
        )


def pick_gpu_index(device_spec: Any, gpus: Optional[Dict[int, GpuMemory]] = None) -> Optional[int]:
    """GPU index a run on `device_spec` lands on (first listed id; auto = most VRAM)."""
    if is_cpu_device(device_spec):
        return None
    gpus = gpus if gpus is not None else query_gpu_memory()
    if not gpus:
        return None
    ids = parse_device_ids(device_spec)
    for idx in ids:
        if idx in gpus:
            return idx
    if ids:
        return None
    # "auto": the engine runs on CUDA device 0, i.e. the first visible GPU in PCI bus order.
    for idx in parse_device_ids(os.environ.get("CUDA_VISIBLE_DEVICES", "")):
        if idx in gpus:
            return idx
    return min(gpus)


def resolve_budget(
    device_spec: Any,
    save_gb: Any,
    *,
    overhead_gb: float = DEFAULT_PROCESS_OVERHEAD_GB,
    gpus: Optional[Dict[int, GpuMemory]] = None,
) -> Optional[VramBudget]:
    """Budget on the GPU a run will use; None for CPU or when no NVIDIA GPU is visible."""
    gpus = gpus if gpus is not None else query_gpu_memory()
    idx = pick_gpu_index(device_spec, gpus)
    if idx is None:
        return None
    return VramBudget(gpu=gpus[idx], save_gb=clamp_save_vram_gb(save_gb), overhead_gb=float(overhead_gb))


def resolve_budgets(device_spec: Any, save_gb: Any, *, overhead_gb: float = DEFAULT_PROCESS_OVERHEAD_GB) -> List[VramBudget]:
    """One budget per listed GPU (multi-GPU runs); falls back to the auto pick."""
    gpus = query_gpu_memory()
    ids = [i for i in parse_device_ids(device_spec) if i in gpus]
    if not ids:
        one = resolve_budget(device_spec, save_gb, overhead_gb=overhead_gb, gpus=gpus)
        return [one] if one else []
    save = clamp_save_vram_gb(save_gb)
    return [VramBudget(gpu=gpus[i], save_gb=save, overhead_gb=float(overhead_gb)) for i in ids]


def merge_alloc_conf(existing: str, fraction: float) -> str:
    """Set per_process_memory_fraction inside a PYTORCH_ALLOC_CONF string (other options kept)."""
    base = _FRACTION_RE.sub("", str(existing or "")).strip().strip(",")
    item = f"per_process_memory_fraction:{max(0.01, min(1.0, float(fraction))):.4f}"
    return f"{base},{item}" if base else item


def apply_vram_cap_env(
    env: Dict[str, str],
    device_spec: Any,
    save_gb: Any,
    *,
    overhead_gb: float = DEFAULT_PROCESS_OVERHEAD_GB,
    log: Optional[Callable[[str], None]] = None,
    tag: str = "VRAM",
    settings_tight: Optional[bool] = None,
    limit_override_gb: Optional[float] = None,
) -> Optional[VramBudget]:
    """Cap an engine subprocess (env is the child's environment) at total - in use - Save VRAM.

    Child processes the engine spawns inherit the same cap. Returns the budget used
    (the tightest one for multi-GPU runs) or None when nothing was capped (CPU, no
    NVIDIA GPU, or nvidia-smi unavailable)."""
    if is_cpu_device(device_spec):
        return None
    budgets = resolve_budgets(device_spec, save_gb, overhead_gb=overhead_gb)
    if not budgets:
        return None
    budget = min(budgets, key=lambda b: b.fraction)
    full_limit = budget.limit_gb
    try:
        override = float(limit_override_gb) if limit_override_gb is not None else 0.0
    except (TypeError, ValueError):
        override = 0.0
    if 0.0 < override < full_limit:
        # Auto Tune's estimate for this run: the caching allocator would otherwise keep freed blocks up to the
        # whole budget, so VRAM in use would read far above the estimate whenever the GPU has room.
        budget = VramBudget(gpu=budget.gpu, save_gb=max(0.0, budget.gpu.free_gb - override),
                            overhead_gb=budget.overhead_gb, notes=list(budget.notes))
    legacy = env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    current = env.get("PYTORCH_ALLOC_CONF") or legacy or ""
    # The cudaMallocAsync backend ignores the fraction (verified: 40 GB reserved on a 31.8 GB
    # card with fraction 0.15), so the cap always runs on the native caching allocator.
    current = re.sub(r"(?:^|,)\s*backend\s*:\s*cudaMallocAsync\s*", "", str(current)).strip().strip(",")
    # Near the cap, fragmentation decides success: without max_split_size_mb a SeedVR2 1080p batch
    # failed at a 19.7 GB cap with 14.9 GB allocated and 4.0 GB stranded in split blocks; with 512 MB
    # it completed (expandable segments, the usual fix, is not supported on Windows).
    # It costs speed (74 s -> 84.5 s on the same SeedVR2 run with room to spare: large blocks are no
    # longer reused), so it is only added when the run is close to its budget or its fit is unknown.
    tight = settings_tight if settings_tight is not None else True
    if budget.limit_gb < full_limit - 1e-6:
        tight = True  # capped at the estimate: keep fragmentation down (measured OOM at the cap without it)
    if tight and "max_split_size_mb" not in current and not os.environ.get("SECOURSES_NO_MAX_SPLIT"):
        current = f"{current},{MAX_SPLIT_SIZE_CONF}" if current else MAX_SPLIT_SIZE_CONF
    env["PYTORCH_ALLOC_CONF"] = merge_alloc_conf(current, budget.fraction)
    env[ENV_SAVE_VRAM] = f"{budget.save_gb:g}"
    env[ENV_VRAM_LIMIT] = f"{budget.limit_gb:.3f}"
    if log:
        msg = (
            f"[{tag}] {budget.describe()}; PyTorch capped at {budget.allocator_cap_gb:.1f} GB "
            f"(fraction {budget.fraction:.3f}), so it never spills into shared GPU memory."
        )
        if budget.limit_gb < full_limit - 1e-6:
            msg += (f" The cap follows Auto Tune's estimate for this run ({budget.limit_gb:.1f} of the "
                    f"{full_limit:.1f} GB Save VRAM leaves).")
        if budget.limit_gb < budget.overhead_gb + MIN_ALLOCATOR_CAP_GB:
            msg += f" WARNING: only {budget.gpu.free_gb:.1f} GB is free; close other GPU apps or lower Save VRAM."
        log(msg)
    return budget


@dataclass
class MediaInfo:
    path: str
    kind: str  # video | image | directory | unknown
    width: int
    height: int
    frames: int
    fps: float

    def describe(self) -> str:
        if self.kind == "image":
            return f"image {self.width}x{self.height}"
        unit = "frames" if self.kind == "video" else "images"
        return f"{self.kind} {self.width}x{self.height}, {self.frames} {unit}" + (f" @ {self.fps:.3g} fps" if self.fps else "")


_MEDIA_CACHE: Dict[tuple, MediaInfo] = {}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".exr"}


def _ffprobe_video(path: str) -> Optional[MediaInfo]:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,nb_frames,avg_frame_rate,r_frame_rate:format=duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=20, check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        import json

        data = json.loads(proc.stdout or "{}")
    except Exception:
        return None
    streams = data.get("streams") or []
    if not streams:
        return None
    st = streams[0]

    def _rate(text: Any) -> float:
        try:
            num, _, den = str(text or "0/1").partition("/")
            return float(num) / float(den or 1) if float(den or 1) else 0.0
        except Exception:
            return 0.0

    fps = _rate(st.get("avg_frame_rate")) or _rate(st.get("r_frame_rate"))
    frames = 0
    try:
        frames = int(st.get("nb_frames") or 0)
    except Exception:
        frames = 0
    if frames <= 0:
        try:
            frames = int(round(float((data.get("format") or {}).get("duration") or 0) * fps))
        except Exception:
            frames = 0
    return MediaInfo(path=path, kind="video", width=int(st.get("width") or 0), height=int(st.get("height") or 0),
                     frames=max(1, frames), fps=fps)


def probe_media(path: Any) -> Optional[MediaInfo]:
    """Size and frame count of an input (video, image or frame folder); cached by mtime/size."""
    p = str(path or "").strip().strip('"')
    if not p or not os.path.exists(p):
        return None
    try:
        st = os.stat(p)
        key = (os.path.normcase(os.path.abspath(p)), int(st.st_mtime), int(st.st_size))
    except Exception:
        key = (p, 0, 0)
    if key in _MEDIA_CACHE:
        return _MEDIA_CACHE[key]
    info: Optional[MediaInfo] = None
    if os.path.isdir(p):
        try:
            files = sorted(f for f in os.listdir(p) if os.path.splitext(f)[1].lower() in _IMAGE_EXTS)
        except Exception:
            files = []
        if files:
            dims = _image_size(os.path.join(p, files[0]))
            if dims:
                info = MediaInfo(path=p, kind="directory", width=dims[0], height=dims[1], frames=len(files), fps=0.0)
    elif os.path.splitext(p)[1].lower() in _IMAGE_EXTS:
        dims = _image_size(p)
        if dims:
            info = MediaInfo(path=p, kind="image", width=dims[0], height=dims[1], frames=1, fps=0.0)
    else:
        info = _ffprobe_video(p)
    if info and info.width > 0 and info.height > 0:
        if len(_MEDIA_CACHE) > 256:
            _MEDIA_CACHE.clear()
        _MEDIA_CACHE[key] = info
        return info
    return None


def _image_size(path: str) -> Optional[tuple]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        pass
    try:
        from shared.path_utils import get_media_dimensions

        dims = get_media_dimensions(path)
        return (int(dims[0]), int(dims[1])) if dims else None
    except Exception:
        return None


def frames_per_unit(media: Optional[MediaInfo], seed_controls: Optional[Dict[str, Any]] = None,
                    native_chunk_frames: int = 0) -> int:
    """Most frames one model call can receive: the whole clip, a static app chunk, or a native chunk.

    Scene-split chunks are unknown before PySceneDetect runs, so the whole clip is the worst case."""
    if media is None:
        return 1
    frames = max(1, int(media.frames))
    sc = seed_controls or {}
    try:
        auto_chunk = bool(sc.get("auto_chunk", True))
        chunk_sec = float(sc.get("chunk_size_sec", 0) or 0)
        overlap_sec = float(sc.get("chunk_overlap_sec", 0) or 0)
    except Exception:
        auto_chunk, chunk_sec, overlap_sec = True, 0.0, 0.0
    if media.kind == "video" and not auto_chunk and chunk_sec > 0 and media.fps > 0:
        frames = min(frames, int(round((chunk_sec + max(0.0, overlap_sec)) * media.fps)) + 1)
    if native_chunk_frames and int(native_chunk_frames) > 0:
        frames = min(frames, int(native_chunk_frames))
    return max(1, frames)


@dataclass
class TunePlan:
    """What Auto Tune sets for one input on one GPU budget."""

    updates: Dict[str, Any] = field(default_factory=dict)
    estimate_gb: float = 0.0
    budget: Optional[VramBudget] = None
    lines: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fits: bool = True

    def report(self, title: str = "Auto Tune") -> str:
        head = [f"{title}: " + ("settings applied." if self.updates else "no change needed.")]
        if self.budget is not None:
            head.append(self.budget.describe() + ".")
        body = list(self.lines)
        if self.estimate_gb:
            body.append(f"Estimated peak VRAM: {self.estimate_gb:.1f} GB" + (
                f" of {self.budget.limit_gb:.1f} GB" if self.budget is not None else ""))
        return "\n".join(head + body + [f"Warning: {w}" for w in self.warnings])


TIGHT_SHARE = 0.8  # a plan using more than this share of its budget runs with fragmentation limits


def mark_tightness(settings: Dict[str, Any], plan: "TunePlan") -> None:
    """Remember for the launcher whether this run's planned peak is close to its budget."""
    if plan is not None and plan.budget is not None and plan.estimate_gb > 0:
        settings["_vram_tight"] = (not plan.fits) or plan.estimate_gb > TIGHT_SHARE * plan.budget.limit_gb


def is_tight(settings: Dict[str, Any]) -> Optional[bool]:
    value = settings.get("_vram_tight")
    return None if value is None else bool(value)


def describe_gb(value: float) -> str:
    return f"{float(value):.1f} GB"


def round_down_4n1(value: int, minimum: int = 1) -> int:
    v = max(int(minimum), int(value))
    return max(int(minimum), ((v - 1) // 4) * 4 + 1)


def round_up_4n1(value: int) -> int:
    v = max(1, int(value))
    return ((v - 1 + 3) // 4) * 4 + 1


def round_down_8n1(value: int, minimum: int = 1) -> int:
    v = max(int(minimum), int(value))
    return max(int(minimum), ((v - 1) // 8) * 8 + 1)


def round_up_8n1(value: int) -> int:
    v = max(1, int(value))
    return ((v - 1 + 7) // 8) * 8 + 1
