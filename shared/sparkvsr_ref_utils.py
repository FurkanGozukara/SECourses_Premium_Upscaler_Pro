from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SPARKVSR_TEMPORAL_REF_MANIFEST_NAME = "sparkvsr_temporal_references.json"

PISA_RUNTIME_KEYS = (
    "pisa_python_executable",
    "pisa_script_path",
    "pisa_sd_model_path",
    "pisa_chkpt_path",
)

_PISA_ENV_KEYS = {
    "pisa_python_executable": ("SEC_PISA_PYTHON_EXECUTABLE", "PISA_PYTHON_EXECUTABLE"),
    "pisa_script_path": ("SEC_PISA_SCRIPT_PATH", "PISA_SCRIPT_PATH"),
    "pisa_sd_model_path": ("SEC_PISA_SD_MODEL_PATH", "PISA_SD_MODEL_PATH"),
    "pisa_chkpt_path": ("SEC_PISA_CHKPT_PATH", "PISA_CHKPT_PATH"),
}


def _unique_paths(values: List[Path]) -> List[Path]:
    unique: List[Path] = []
    seen = set()
    for value in values:
        try:
            candidate = Path(value).expanduser().resolve()
        except Exception:
            continue
        token = os.path.normcase(str(candidate))
        if token not in seen:
            seen.add(token)
            unique.append(candidate)
    return unique


def _resolve_candidate(value: Any, base_dir: Path) -> Optional[Path]:
    raw = os.path.expandvars(os.path.expanduser(str(value or "").strip()))
    if not raw:
        return None
    try:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        return candidate.resolve()
    except Exception:
        return None


def _valid_pisa_candidate(key: str, value: Any, base_dir: Path) -> Optional[str]:
    if key == "pisa_python_executable":
        raw = str(value or "").strip()
        if not raw:
            return None
        resolved_command = shutil.which(raw)
        if resolved_command:
            return str(Path(resolved_command).resolve())
        candidate = _resolve_candidate(raw, base_dir)
        if candidate and candidate.is_file():
            return str(candidate)
        return None

    candidate = _resolve_candidate(value, base_dir)
    if not candidate or not candidate.exists():
        return None
    if key in {"pisa_script_path", "pisa_chkpt_path"} and not candidate.is_file():
        return None
    if key == "pisa_sd_model_path" and not candidate.is_dir():
        return None
    return str(candidate)


def discover_pisa_runtime(settings: Optional[Dict[str, Any]], base_dir: Path) -> Dict[str, str]:
    """Find a usable PiSA-SR installation without depending on the process CWD."""
    base_dir = Path(base_dir).expanduser().resolve()
    settings = settings if isinstance(settings, dict) else {}

    roots: List[Path] = []
    explicit_script = _resolve_candidate(settings.get("pisa_script_path"), base_dir)
    if explicit_script:
        roots.append(explicit_script.parent)
    for env_key in ("SEC_PISA_ROOT", "PISA_ROOT"):
        env_root = _resolve_candidate(os.environ.get(env_key), base_dir)
        if env_root:
            roots.append(env_root)
    roots.extend(
        [
            base_dir / "PiSA-SR",
            base_dir / "pisasr",
            base_dir / "external" / "PiSA-SR",
            base_dir.parent / "PiSA-SR",
            Path.cwd() / "PiSA-SR",
        ]
    )
    roots = _unique_paths(roots)

    candidates: Dict[str, List[Any]] = {key: [settings.get(key)] for key in PISA_RUNTIME_KEYS}
    for key, env_keys in _PISA_ENV_KEYS.items():
        candidates[key].extend(os.environ.get(env_key) for env_key in env_keys)

    for root in roots:
        candidates["pisa_script_path"].append(root / "test_pisasr.py")
        candidates["pisa_sd_model_path"].extend(
            [
                root / "preset" / "models" / "stable-diffusion-2-1-base",
                root / "models" / "stable-diffusion-2-1-base",
            ]
        )
        candidates["pisa_chkpt_path"].extend(
            [
                root / "preset" / "models" / "pisa_sr.pkl",
                root / "models" / "pisa_sr.pkl",
            ]
        )
        for env_name in ("venv", ".venv", "env"):
            env_root = root / env_name
            candidates["pisa_python_executable"].append(
                env_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            )

    home = Path.home()
    for conda_home in (home / "miniconda3", home / "anaconda3", home / "miniforge3"):
        for env_name in ("PiSA-SR", "pisa-sr", "pisasr"):
            env_root = conda_home / "envs" / env_name
            candidates["pisa_python_executable"].append(
                env_root / ("python.exe" if os.name == "nt" else "bin/python")
            )

    # PiSA can also be installed into SEC's environment. Keep this behind all
    # dedicated-environment candidates because the official project recommends
    # a separate Python 3.10 environment.
    candidates["pisa_python_executable"].extend(
        [
            base_dir / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
            sys.executable,
        ]
    )

    resolved: Dict[str, str] = {}
    for key in PISA_RUNTIME_KEYS:
        for candidate in candidates[key]:
            valid = _valid_pisa_candidate(key, candidate, base_dir)
            if valid:
                resolved[key] = valid
                break
    return resolved


def resolve_pisa_runtime(
    settings: Dict[str, Any],
    base_dir: Path,
) -> Tuple[Dict[str, Any], Optional[str], List[str]]:
    """Resolve and validate every path needed by the PiSA-SR reference mode."""
    merged = dict(settings or {})
    if str(merged.get("ref_mode") or "").strip().lower() != "pisasr":
        return merged, None, []

    discovered = discover_pisa_runtime(merged, Path(base_dir))
    notes: List[str] = []
    missing: List[str] = []
    labels = {
        "pisa_python_executable": "PiSA Python executable",
        "pisa_script_path": "PiSA test_pisasr.py script",
        "pisa_sd_model_path": "Stable Diffusion 2.1 base model directory",
        "pisa_chkpt_path": "PiSA pisa_sr.pkl checkpoint",
    }
    for key in PISA_RUNTIME_KEYS:
        original = str(merged.get(key) or "").strip()
        resolved = discovered.get(key, "")
        if not resolved:
            missing.append(labels[key])
            continue
        merged[key] = resolved
        if not original or os.path.normcase(original) != os.path.normcase(resolved):
            notes.append(f"[SparkVSR] Auto-resolved {labels[key]}: {resolved}")

    if missing:
        standard_root = Path(base_dir).expanduser().resolve() / "PiSA-SR"
        error = (
            "PiSA-SR reference mode is not fully configured. Missing or invalid: "
            + ", ".join(missing)
            + ". Install PiSA-SR under "
            + str(standard_root)
            + " (with test_pisasr.py, preset/models/stable-diffusion-2-1-base, "
            "preset/models/pisa_sr.pkl, and a venv/.venv), fill the four PiSA fields in "
            "SparkVSR Advanced Reference Controls, or set SEC_PISA_PYTHON_EXECUTABLE, "
            "SEC_PISA_SCRIPT_PATH, SEC_PISA_SD_MODEL_PATH, and SEC_PISA_CHKPT_PATH."
        )
        return merged, error, notes
    return merged, None, notes


def temporal_padding_to_vae_grid(frame_count: int, period: int = 8) -> int:
    """SparkVSR/CogVideoX VAE round-trips frame counts exactly on the 8n+1 grid."""
    frame_count = int(frame_count)
    period = max(1, int(period or 1))
    if frame_count <= 0:
        return 0
    remainder = (frame_count - 1) % period
    return 0 if remainder == 0 else period - remainder


def make_temporal_chunks(frame_count: int, chunk_len: int, overlap_t: int = 8) -> List[Tuple[int, int]]:
    frame_count = int(frame_count)
    if int(chunk_len or 0) <= 0 or frame_count <= int(chunk_len or 0):
        return [(0, frame_count)]
    chunk = int(chunk_len)
    overlap = int(overlap_t or 0)
    effective_stride = chunk - overlap
    if effective_stride <= 0:
        raise ValueError("chunk_len must be greater than overlap_t")
    starts = [0]
    while starts[-1] + chunk < frame_count:
        next_start = starts[-1] + effective_stride
        if next_start + chunk >= frame_count:
            if next_start < frame_count and next_start != starts[-1]:
                starts.append(next_start)
            break
        starts.append(next_start)
    starts = sorted(set(max(0, min(int(s), max(0, frame_count - 1))) for s in starts))
    return [(s, min(s + chunk, frame_count)) for s in starts]


def effective_frame_window(total_source_frames: int, start_frame: int = 0, end_frame: int = -1) -> Tuple[int, int, int]:
    """
    Return (start, end_exclusive, count) after SparkVSR start/end frame settings.
    """
    total = max(0, int(total_source_frames or 0))
    if total <= 0:
        return 0, 0, 0
    start = max(0, int(start_frame or 0))
    end_raw = int(end_frame) if end_frame is not None else -1
    end = total if end_raw < 0 else min(total, end_raw + 1)
    if start >= end:
        return start, start, 0
    return start, end, max(0, end - start)


def build_temporal_reference_specs(
    total_source_frames: int,
    chunk_len: int,
    overlap_t: int = 8,
    *,
    start_frame: int = 0,
    end_frame: int = -1,
) -> List[Dict[str, int]]:
    """
    Build the exact source-frame references needed for SparkVSR temporal chunks.

    `t_start` is local to the decoded SparkVSR input after start/end trimming.
    `source_frame` is the corresponding frame in the original source, clamped to
    the last real frame when a padded chunk start lands in the repeated tail.
    """
    source_start, _source_end, effective_count = effective_frame_window(
        total_source_frames,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    if effective_count <= 0:
        return []

    padded_count = effective_count + temporal_padding_to_vae_grid(effective_count)
    chunks = make_temporal_chunks(padded_count, int(chunk_len or 0), int(overlap_t or 0))
    specs: List[Dict[str, int]] = []
    last_real_offset = max(0, effective_count - 1)
    for order, (t_start, t_end) in enumerate(chunks, 1):
        source_frame = source_start + min(max(0, int(t_start)), last_real_offset)
        specs.append(
            {
                "order": int(order),
                "t_start": int(t_start),
                "t_end": int(t_end),
                "source_frame": int(source_frame),
                "effective_frame_count": int(effective_count),
                "padded_frame_count": int(padded_count),
            }
        )
    return specs


def write_temporal_reference_manifest(
    ref_dir: str | os.PathLike,
    *,
    source_path: str,
    entries: List[Dict[str, Any]],
    chunk_len: int,
    overlap_t: int,
    start_frame: int = 0,
    end_frame: int = -1,
    upscaler: str = "",
) -> Path:
    root = Path(ref_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "sparkvsr_temporal_references_v1",
        "source_path": str(source_path or ""),
        "chunk_len": int(chunk_len or 0),
        "overlap_t": int(overlap_t or 0),
        "start_frame": int(start_frame or 0),
        "end_frame": int(end_frame if end_frame is not None else -1),
        "upscaler": str(upscaler or ""),
        "entries": entries,
    }
    manifest = root / SPARKVSR_TEMPORAL_REF_MANIFEST_NAME
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_temporal_reference_manifest(source_path: str | os.PathLike | None) -> Dict[int, Path]:
    if not source_path:
        return {}
    source = Path(str(source_path)).expanduser()
    manifest = source / SPARKVSR_TEMPORAL_REF_MANIFEST_NAME if source.is_dir() else source
    if manifest.suffix.lower() != ".json" or not manifest.exists():
        return {}
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return {}

    raw_entries: Any = payload.get("entries") if isinstance(payload, dict) else payload
    if not isinstance(raw_entries, list):
        return {}

    index: Dict[int, Path] = {}
    for raw in raw_entries:
        if not isinstance(raw, dict):
            continue
        frame_value = raw.get("t_start", raw.get("chunk_start_idx", raw.get("frame_idx")))
        rel_path = raw.get("reference_path", raw.get("path"))
        if frame_value is None or not rel_path:
            continue
        try:
            frame_idx = int(frame_value)
        except Exception:
            continue
        ref_path = Path(str(rel_path))
        if not ref_path.is_absolute():
            ref_path = manifest.parent / ref_path
        if ref_path.exists():
            index[frame_idx] = ref_path
    return dict(sorted(index.items(), key=lambda item: item[0]))


def choose_temporal_reference_path(index: Dict[int, Path], t_start: int) -> Optional[Path]:
    if not index:
        return None
    start = int(t_start or 0)
    if start in index:
        return index[start]
    previous = [idx for idx in index.keys() if idx <= start]
    if previous:
        return index[max(previous)]
    return index[min(index.keys())]


def _select_indices(total_frames: int) -> List[int]:
    """Select first, middle, and last frame indices, matching SparkVSR defaults."""
    total = int(total_frames or 0)
    if total <= 0:
        return []
    if total == 1:
        return [0]
    if total == 2:
        return [0, 1]
    return [0, total // 2, total - 1]


def save_ref_frames_locally(
    video_path: str | os.PathLike | None = None,
    output_dir: str | os.PathLike | None = None,
    video_id: Optional[str] = None,
    target_frames: Optional[int] = None,
    is_match: bool = False,
    specific_indices: Optional[List[int]] = None,
) -> List[Tuple[int, str]]:
    """Extract selected reference frames locally."""
    if output_dir is None:
        raise ValueError("output_dir must be provided")
    if video_path is None:
        return []

    import cv2  # type: ignore

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{video_id}_" if video_id else ""

    path = str(video_path)
    if "LQ-Video" in path:
        gt_path = path.replace("LQ-Video", "GT-Video")
        if Path(gt_path).exists():
            path = gt_path

    use_decord = False
    vr = None
    cap = None
    try:
        from decord import VideoReader, cpu  # type: ignore

        vr = VideoReader(path, ctx=cpu(0))
        total_frames = len(vr)
        use_decord = True
    except Exception:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    effective_frames = total_frames
    if is_match and total_frames > 0:
        remainder = (total_frames - 1) % 8
        if remainder != 0:
            effective_frames = total_frames + (8 - remainder)

    indices = list(specific_indices) if specific_indices is not None else _select_indices(target_frames or effective_frames)
    saved: List[Tuple[int, str]] = []

    try:
        for idx in indices:
            if total_frames <= 0:
                break
            read_idx = min(max(0, int(idx)), total_frames - 1)
            if use_decord and vr is not None:
                frame_obj = vr[read_idx]
                frame_np = frame_obj.asnumpy() if hasattr(frame_obj, "asnumpy") else frame_obj.cpu().numpy()
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                ok = True
            else:
                assert cap is not None
                cap.set(cv2.CAP_PROP_POS_FRAMES, read_idx)
                ok, frame_bgr = cap.read()
            if not ok:
                continue
            frame_path = out_dir / f"{prefix}frame_{int(idx):05d}.png"
            cv2.imwrite(str(frame_path), frame_bgr)
            saved.append((int(idx), str(frame_path)))
    finally:
        if cap is not None:
            cap.release()

    return saved
