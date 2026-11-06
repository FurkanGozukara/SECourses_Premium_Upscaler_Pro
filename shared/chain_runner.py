"""Chains: run several tabs one after another on the same video (step N's output is step N+1's input).

Every step runs the tab's own ``run_action`` with the settings currently set in that tab (from the
shared state), so a chain reuses every engine, option, VRAM Auto Tune and output rule as they are.
A step overrides only the input, its output folder, batch / resume / auto-transfer and, when given,
its scale (upscalers) or FPS multiplier (RIFE). Global post-processes (DLSS 5, RIFE, comparison
videos) are switched off inside steps; a chain adds them as explicit steps instead.
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv", ".wmv", ".ts", ".mts", ".m2ts"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class ChainEngine:
    key: str            # universal-preset tab name (seed_controls["<key>_settings"])
    label: str
    icon: str
    run_action: Callable
    style: str = "upload"   # "upload": run_action(upload, *values, ...); "values": run_action(*values, ...)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    accepts_progress: bool = True
    accepts_snapshot: bool = True
    factor_kind: str = "scale"  # "scale" (upscale factor), "fps" (RIFE multiplier) or "" (none)
    summary_keys: tuple = ()
    cancel: Optional[Callable] = None  # the tab's own cancel action (stops the running step)


ENGINES: Dict[str, ChainEngine] = {}
# Steps that only take videos: on an image input they are skipped, the chain goes on.
VIDEO_ONLY = {"ltx25", "rife"}
_RUNNING: Dict[str, Any] = {"engine": None}
# Display order of the engine palette.
ENGINE_ORDER = ["seedvr2", "flashvsr", "relayvsr", "sparkvsr", "gan", "rtx", "ltx25", "open_h3", "dlss5", "face", "rife"]
ENGINE_META = {
    "seedvr2": ("SeedVR2", "🎬", "scale", ("dit_model", "resolution")),
    "flashvsr": ("FlashVSR+", "⚡", "scale", ("version", "mode")),
    "relayvsr": ("RelayVSR", "🔁", "scale", ("wan_model", "window_size")),
    "sparkvsr": ("SparkVSR", "✨", "scale", ("model_name", "upscale_mode")),
    "gan": ("Image GAN", "🖼️", "scale", ("model",)),
    "rtx": ("RTX Super Res", "🚀", "scale", ("quality_preset",)),
    "ltx25": ("LTX 2.5", "🎞️", "", ("model_name",)),
    "open_h3": ("Open-H3", "🌀", "scale", ("model_name", "resolution_preset")),
    "dlss5": ("DLSS 5", "💎", "", ("profile", "intensity")),
    "face": ("Face Restore", "👤", "", ("restoration_model", "face_strength")),
    "rife": ("RIFE FPS", "⏱️", "fps", ("model",)),
}


def register_chain_engine(key: str, run_action: Callable, *, style: str = "upload", progress: bool = True,
                          snapshot: bool = True, cancel: Optional[Callable] = None, **kwargs: Any) -> None:
    """Register a tab's ``run_action``; ``kwargs`` are fixed keyword arguments (e.g. preview_only=False)."""
    label, icon, factor_kind, summary = ENGINE_META.get(key, (key, "🔧", "", ()))
    ENGINES[key] = ChainEngine(key, label, icon, run_action, style, dict(kwargs), progress, snapshot,
                               factor_kind, summary, cancel)


def cancel_running() -> str:
    """Stop the step that is running now with its tab's own cancel action (the chain stops after it)."""
    engine = _RUNNING.get("engine")
    if engine is None or engine.cancel is None:
        return ""
    for args in ((), (None,)):
        try:
            engine.cancel(*args)
            return engine.label
        except TypeError:
            continue
        except Exception:
            return ""
    return ""


def media_info(path: str) -> Dict[str, Any]:
    """Kind, size, frame rate, frame count and length of a video or image (what the preview shows)."""
    info: Dict[str, Any] = {"path": path, "kind": "", "width": 0, "height": 0, "fps": 0.0, "frames": 0, "duration": 0.0}
    suffix = Path(str(path)).suffix.lower()
    info["kind"] = "video" if suffix in VIDEO_EXTS else "image" if suffix in IMAGE_EXTS else ""
    try:
        from shared.path_utils import get_media_dimensions, get_media_duration_seconds, get_media_fps

        dims = get_media_dimensions(path)
        if dims:
            info["width"], info["height"] = int(dims[0]), int(dims[1])
        if info["kind"] == "video":
            info["fps"] = float(get_media_fps(path) or 0.0)
            info["duration"] = float(get_media_duration_seconds(path) or 0.0)
            info["frames"] = int(round(info["fps"] * info["duration"])) if info["fps"] and info["duration"] else 0
    except Exception:
        pass
    return info


def _tab_factor(engine: str, state: Dict[str, Any]) -> Optional[float]:
    seed = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
    value = (seed.get(f"{engine}_settings") or {}).get("upscale_factor")
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


_CAP_KEYS = ("max_resolution", "max_target_resolution")


def _tab_settings(engine: str, state: Dict[str, Any]) -> Dict[str, Any]:
    seed = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
    value = seed.get(f"{engine}_settings") if isinstance(seed, dict) else None
    return value if isinstance(value, dict) else {}


def _cap(size: tuple, settings: Dict[str, Any]) -> tuple:
    """Apply the tab's max-edge cap (0 = no cap) to a projected size."""
    cap = 0
    for key in _CAP_KEYS:
        try:
            cap = int(float(settings.get(key) or 0))
        except (TypeError, ValueError):
            cap = 0
        if cap:
            break
    if cap and max(size) > cap:
        ratio = cap / float(max(size))
        return int(round(size[0] * ratio)), int(round(size[1] * ratio))
    return size


def _open_h3_size(size: tuple, settings: Dict[str, Any], factor: Optional[float]) -> Optional[tuple]:
    from .open_h3_constants import is_jxl_workflow, jxl_output_size

    if is_jxl_workflow(settings):  # JonXL H3 Upscale: fitted under its max size, no factor
        from .open_h3_runner import jxl_max_side

        return jxl_output_size(size[0], size[1], jxl_max_side(settings))
    preset = str(settings.get("resolution_preset") or "")
    if factor or preset.startswith("Custom"):
        f = factor or float(settings.get("upscale_factor") or 2.0)
        return int(round(size[0] * f)), int(round(size[1] * f))
    match = re.search(r"(\d{3,4})p", preset)
    if not match:
        return None
    short = int(match.group(1))
    ratio = short / float(min(size))
    return int(round(size[0] * ratio)), int(round(size[1] * ratio))


def chain_label(steps: List[Dict[str, Any]], preset_name: str = "", base_dir: Optional[Path] = None) -> str:
    """The preset name when the steps are that preset's, otherwise the steps themselves (queue / chain.json)."""
    active = [s for s in normalize_steps(steps) if s.get("enabled", True)]
    name = str(preset_name or "").strip()
    if name and base_dir is not None and normalize_steps(load_chain_preset(base_dir, name)) == normalize_steps(steps):
        return name
    parts: List[str] = []
    for step in active:
        meta = ENGINE_META.get(step["engine"], (step["engine"], "", "", ()))
        factor = step.get("factor")
        text = meta[0] + (f" {factor:g}x" if factor and meta[2] == "scale" else f" {int(factor)}x FPS" if factor else "")
        if parts and parts[-1].split(" ×")[0] == text:
            head, _, count = parts[-1].partition(" ×")
            parts[-1] = f"{head} ×{int(count or 1) + 1}"
        else:
            parts.append(text)
    return ("Custom chain: " + " → ".join(parts)) if parts else (name or "Chain")


def project_sizes(steps: List[Dict[str, Any]], state: Dict[str, Any], width: int, height: int,
                  kind: str = "video") -> List[Optional[tuple]]:
    """Approximate frame size after each step for an input of ``width`` x ``height`` (None = unknown).

    A scale step uses its own factor or its tab's upscale factor; LTX 2.5 doubles; DLSS 5, Face Restore and
    RIFE keep the size; a video-only step is skipped on images. Engines round to their own multiples and tab
    caps (max resolution) can lower a result, so the UI shows these with "≈"."""
    sizes: List[Optional[tuple]] = []
    current: Optional[tuple] = (int(width), int(height)) if width and height else None
    for step in normalize_steps(steps):
        meta = ENGINE_META.get(step["engine"], (step["engine"], "", "", ()))
        if current is not None and step.get("enabled", True) and not (kind == "image" and step["engine"] in VIDEO_ONLY):
            settings = _tab_settings(step["engine"], state)
            if step["engine"] == "open_h3":
                current = _open_h3_size(current, settings, step.get("factor"))
                current = _cap(current, settings) if current else None
            elif meta[2] == "scale":
                factor = step.get("factor") or _tab_factor(step["engine"], state)
                current = (int(round(current[0] * factor)), int(round(current[1] * factor))) if factor else None
                current = _cap(current, settings) if current else None
            elif step["engine"] == "ltx25":  # LTX 2.5 always upscales 2x ("Upscale 2x")
                current = (current[0] * 2, current[1] * 2)
        sizes.append(current)
    return sizes


def available_engines() -> List[str]:
    return [k for k in ENGINE_ORDER if k in ENGINES] + [k for k in ENGINES if k not in ENGINE_ORDER]


def engine_label(key: str) -> str:
    meta = ENGINE_META.get(key)
    return f"{meta[1]} {meta[0]}" if meta else key


# --- chain presets -----------------------------------------------------------------------------

BUILTIN_CHAINS: Dict[str, List[Dict[str, Any]]] = {
    "⭐ SeedVR2 2x → DLSS 5": [{"engine": "seedvr2", "factor": 2.0}, {"engine": "dlss5"}],
    "⭐ SeedVR2 2x → FlashVSR+ 2x → DLSS 5": [
        {"engine": "seedvr2", "factor": 2.0}, {"engine": "flashvsr", "factor": 2.0}, {"engine": "dlss5"}],
    "⭐ RTX Super Res 2x → DLSS 5 → RIFE 2x FPS": [
        {"engine": "rtx", "factor": 2.0}, {"engine": "dlss5"}, {"engine": "rife", "factor": 2}],
    "⭐ DLSS 5 → RTX Super Res 2x": [{"engine": "dlss5"}, {"engine": "rtx", "factor": 2.0}],
    # Multi-pass DLSS 5: every pass renders the previous pass again, so the photoreal look gets stronger.
    "⭐ DLSS 5 ×3 (multi-pass, strongest look)": [{"engine": "dlss5"}, {"engine": "dlss5"}, {"engine": "dlss5"}],
    "⭐ SeedVR2 2x → DLSS 5 ×3": [{"engine": "seedvr2", "factor": 2.0}, {"engine": "dlss5"}, {"engine": "dlss5"},
                                 {"engine": "dlss5"}],
}


DEFAULT_CHAIN = "⭐ SeedVR2 2x → FlashVSR+ 2x → DLSS 5"
# App (universal) presets keep only the NAME of the selected chain preset (key "chains"); empty =
# the newest chain you saved, else the default example.
CHAINS_ORDER = ["chain_preset"]


def chains_defaults() -> Dict[str, Any]:
    return {"chain_preset": ""}


def resolve_chain_preset(base_dir: Path, name: Any) -> str:
    """The chain preset to load for an app preset: its saved name when that chain still exists,
    otherwise the newest chain the user saved, otherwise the default example."""
    text = str(name or "").strip()
    if text in BUILTIN_CHAINS or (text and (chains_dir(base_dir) / f"{_safe_name(text)}.json").is_file()):
        return text
    saved = sorted(chains_dir(base_dir).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return saved[0].stem if saved else DEFAULT_CHAIN


def chains_dir(base_dir: Path) -> Path:
    path = Path(base_dir) / "presets" / "chains"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_name(name: str) -> str:
    text = re.sub(r"[^\w\- .+()→,]", "_", str(name or "").strip()).strip(" .")
    return text[:80] or "chain"


def list_chain_presets(base_dir: Path) -> List[str]:
    saved = sorted(p.stem for p in chains_dir(base_dir).glob("*.json"))
    return list(BUILTIN_CHAINS) + [name for name in saved if name not in BUILTIN_CHAINS]


def normalize_steps(steps: Any) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for step in steps or []:
        if not isinstance(step, dict) or not step.get("engine"):
            continue
        item = {"engine": str(step["engine"]), "enabled": bool(step.get("enabled", True))}
        factor = step.get("factor")
        if factor not in (None, "", 0):
            try:
                item["factor"] = float(factor)
            except (TypeError, ValueError):
                pass
        result.append(item)
    return result


def save_chain_preset(base_dir: Path, name: str, steps: List[Dict[str, Any]]) -> str:
    safe = _safe_name(name)
    payload = {"name": safe, "saved_at": datetime.now().isoformat(timespec="seconds"), "steps": normalize_steps(steps)}
    target = chains_dir(base_dir) / f"{safe}.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)
    return safe


def load_chain_preset(base_dir: Path, name: str) -> List[Dict[str, Any]]:
    if name in BUILTIN_CHAINS:
        return normalize_steps(copy.deepcopy(BUILTIN_CHAINS[name]))
    path = chains_dir(base_dir) / f"{_safe_name(name)}.json"
    if not path.is_file():
        return []
    try:
        return normalize_steps(json.loads(path.read_text(encoding="utf-8")).get("steps"))
    except (OSError, ValueError):
        return []


def delete_chain_preset(base_dir: Path, name: str) -> bool:
    """Saved chains only (built-in examples stay). Removes the preset's own JSON file."""
    if name in BUILTIN_CHAINS:
        return False
    path = chains_dir(base_dir) / f"{_safe_name(name)}.json"
    if path.is_file() and path.parent == chains_dir(base_dir):
        path.unlink()
        return True
    return False


# --- running ----------------------------------------------------------------------------------

@dataclass
class StepState:
    index: int
    engine: str
    status: str = "pending"   # pending | running | done | failed | skipped
    detail: str = ""
    started: float = 0.0
    ended: float = 0.0
    output: Optional[str] = None


@dataclass
class ChainEvent:
    """One UI update: the step table, the live status line and (at the end) the final output."""
    steps: List[StepState]
    video_index: int
    video_count: int
    source: str
    status: str
    log: List[str]
    final_output: Optional[str] = None
    finished: bool = False
    run_dir: str = ""
    runs: List[Dict[str, Any]] = field(default_factory=list)  # per source: source, run_dir, ok (for Resume)


def _tab_values(engine: ChainEngine, state: Dict[str, Any], defaults: Dict[str, Dict[str, Any]],
                overrides: Dict[str, Any]) -> List[Any]:
    from shared.universal_preset import TAB_CONFIGS, dict_to_values

    seed = state.get("seed_controls", {}) if isinstance(state, dict) else {}
    settings = dict(seed.get(f"{engine.key}_settings") or {})
    order = TAB_CONFIGS[engine.key]["order"]
    for key, value in overrides.items():
        if key in order:
            settings[key] = value
    return dict_to_values(engine.key, settings, defaults.get(engine.key, {}))


def _step_overrides(engine: ChainEngine, step: Dict[str, Any], source: str, step_dir: Path) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "input_path": source,
        "output_override": str(step_dir),
        "batch_enable": False,
        "auto_transfer_output_to_input": False,
        "resume_run_dir": "",
        "resume_chunking": False,
        "global_postprocess": False,
        # Face tab names its single-run fields differently.
        "standalone_input_path": source,
        "standalone_output_override": str(step_dir),
        "standalone_batch_enable": False,
    }
    factor = step.get("factor")
    if factor and engine.factor_kind == "scale":
        overrides["upscale_factor"] = float(factor)
        overrides["use_resolution_tab"] = False
        if engine.key == "open_h3":  # Open-H3 reads the factor only in its custom mode
            overrides["resolution_preset"] = "Custom upscale factor"
    elif factor and engine.factor_kind == "fps":
        overrides["fps_multiplier"] = f"x{int(float(factor))}"
    return overrides


def _quiet_globals(state: Dict[str, Any]) -> Dict[str, Any]:
    """The step's own copy of the state with every global post-process off."""
    step_state = copy.deepcopy(state) if isinstance(state, dict) else {"seed_controls": {}}
    seed = step_state.setdefault("seed_controls", {})
    out = dict(seed.get("output_settings") or {})
    out["frame_interpolation"] = False
    out["generate_comparison_video"] = False
    seed["output_settings"] = out
    seed["global_rife_enabled_val"] = False
    seed["frame_interpolation_val"] = False
    seed["generate_comparison_video_val"] = False
    dl = dict(seed.get("dlss5_settings") or {})
    dl["global_postprocess"] = False
    seed["dlss5_settings"] = dl
    seed["last_output_path"] = None
    seed["auto_tune_on_run_chain"] = True
    return step_state


def _text(item: Any) -> str:
    if isinstance(item, dict):
        item = item.get("value")
    return item if isinstance(item, str) else ""


def _payload_text(payload: Any) -> str:
    return _text(payload[0]) if isinstance(payload, (tuple, list)) and payload else ""


def _payload_logs(payload: Any) -> str:
    return _text(payload[1]) if isinstance(payload, (tuple, list)) and len(payload) > 1 else ""


def _output_from_state(step_state: Dict[str, Any]) -> Optional[str]:
    seed = step_state.get("seed_controls", {}) if isinstance(step_state, dict) else {}
    out = seed.get("last_output_path")
    if out and Path(str(out)).is_file():
        return str(out)
    return None


def collect_videos(uploads: Any, path_text: str) -> List[str]:
    files: List[str] = []
    for item in uploads or []:
        path = getattr(item, "name", item)
        if path and Path(str(path)).is_file():
            files.append(str(path))
    text = str(path_text or "").strip().strip('"')
    if text:
        target = Path(text)
        if target.is_dir():
            from shared.path_utils import list_directory_entries_sorted

            files.extend(str(p) for p in map(Path, list_directory_entries_sorted(target))
                         if p.is_file() and p.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS)
        elif target.is_file():
            files.append(str(target))
    seen, unique = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


# --- batch folders ------------------------------------------------------------------------------

BATCH_SUFFIX = "_chain"
BATCH_REPORT = "chain_batch_report.json"


def default_batch_output(output_root: Path, folder: str) -> Path:
    """Where a batch writes when its output folder is empty: <outputs>/<input folder name>_chain."""
    name = Path(str(folder or "batch").strip().strip('"').rstrip("/\\")).name or "batch"
    return Path(output_root) / f"{name}{BATCH_SUFFIX}"


def list_batch_sources(folder: str, subfolders: bool = False, exclude: Optional[str] = None) -> List[str]:
    """Every video and image in ``folder`` (and its subfolders when asked), in natural order; files inside
    ``exclude`` (the batch output folder, when it lies inside the input folder) are left out."""
    root = Path(str(folder or "").strip().strip('"'))
    if not root.is_dir():
        return []
    from shared.path_utils import list_directory_entries_sorted

    skip = None
    if exclude:
        try:
            skip = Path(str(exclude).strip().strip('"')).resolve()
        except OSError:
            skip = None
    found: List[str] = []

    def walk(directory: Path) -> None:
        try:
            if skip is not None and directory.resolve() == skip:
                return
        except OSError:
            return
        entries = [Path(p) for p in list_directory_entries_sorted(directory, include_dirs=subfolders)]
        for entry in entries:
            if entry.is_file() and entry.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS:
                found.append(str(entry))
        if subfolders:
            for entry in entries:
                if entry.is_dir() and not entry.is_symlink():
                    walk(entry)

    walk(root)
    return found


def batch_target(source: str, batch: Dict[str, Any], test_seconds: float = 0.0) -> Path:
    """The result path (without suffix) of ``source`` in the batch output folder, keeping subfolders."""
    out = Path(batch["output"])
    try:
        rel = Path(source).resolve().parent.relative_to(Path(batch["input"]).resolve())
    except (ValueError, OSError):
        rel = Path()
    tag = f"_first{test_seconds:g}s" if test_seconds else ""
    return out / rel / f"{Path(source).stem}{BATCH_SUFFIX}{tag}"


def batch_done(source: str, batch: Dict[str, Any]) -> Optional[str]:
    """The existing full result of ``source`` in the batch output folder (quick-test results do not count)."""
    target = batch_target(source, batch)
    if not target.parent.is_dir():
        return None
    want = VIDEO_EXTS if Path(source).suffix.lower() in VIDEO_EXTS else IMAGE_EXTS | VIDEO_EXTS
    for candidate in sorted(target.parent.glob(target.name + ".*")):
        if candidate.suffix.lower() in want and candidate.stem == target.name and candidate.stat().st_size > 0:
            return str(candidate)
    return None


def _write_batch_report(batch: Dict[str, Any], chain_name: str, started: str, entries: List[Dict[str, Any]]) -> None:
    payload = {"chain": chain_name, "input_folder": str(batch["input"]), "output_folder": str(batch["output"]),
               "started": started, "updated": datetime.now().isoformat(timespec="seconds"), "files": entries}
    try:
        Path(batch["output"]).mkdir(parents=True, exist_ok=True)
        target = Path(batch["output"]) / BATCH_REPORT
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
    except OSError:
        pass


def _write_chain_json(run_dir: Path, chain_name: str, source: str, original: str, active: List[Dict[str, Any]],
                      table: List[StepState], started: str, test_seconds: float) -> None:
    payload = {
        "name": chain_name, "source": original, "chain_input": source, "test_seconds": test_seconds or 0,
        "steps": active, "started": started, "updated": datetime.now().isoformat(timespec="seconds"),
        "results": [{"index": r.index, "engine": r.engine, "status": r.status, "output": r.output,
                     "seconds": round(max(0.0, r.ended - r.started), 1) if r.ended else None} for r in table],
    }
    try:
        (run_dir / "chain.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _resume_plan(run_dir: Path, active: List[Dict[str, Any]]) -> tuple:
    """(chain input, outputs of the leading steps that match ``active`` and still exist) of an earlier run."""
    try:
        data = json.loads((Path(run_dir) / "chain.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, []
    done: List[Optional[str]] = []
    for step, old_step, result in zip(active, data.get("steps") or [], data.get("results") or []):
        same = step["engine"] == old_step.get("engine") and step.get("factor") == old_step.get("factor")
        out = result.get("output")
        if not same or result.get("status") not in ("done", "skipped-image") or (out and not Path(out).is_file()):
            break
        done.append(out)
    return data.get("chain_input") or data.get("source"), done


def _trim_for_test(source: str, seconds: float, run_dir: Path, log: List[str]) -> str:
    """Quick test: the first ``seconds`` of a video, re-encoded near-lossless so every frame is kept."""
    if Path(source).suffix.lower() not in VIDEO_EXTS or seconds <= 0:
        return source
    target = run_dir / f"{Path(source).stem}_first{seconds:g}s.mp4"
    import subprocess

    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-t", f"{seconds:g}", "-map", "0:v:0", "-map", "0:a?",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "10", "-pix_fmt", "yuv420p", "-c:a", "aac", str(target)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        log.append(f"    quick test: first {seconds:g} s -> {target.name}")
        return str(target)
    except Exception as exc:
        log.append(f"    quick test trim failed ({exc}); using the whole video")
        return source


def run_chain(
    sources: List[str],
    steps: List[Dict[str, Any]],
    state: Dict[str, Any],
    *,
    base_dir: Path,
    output_root: Path,
    chain_name: str = "",
    progress: Optional[Callable] = None,
    cancel_flag: Optional[Callable[[], bool]] = None,
    global_settings_snapshot: Optional[Dict[str, Any]] = None,
    test_seconds: float = 0.0,
    resume_dirs: Optional[Dict[str, str]] = None,
    batch: Optional[Dict[str, Any]] = None,
) -> Iterator[ChainEvent]:
    """Run ``steps`` over every source; yields ChainEvents for the UI.

    ``test_seconds`` > 0 runs the chain on the first seconds of each video (quick test).
    ``resume_dirs`` maps a source to the run folder of an earlier, unfinished run of it: the
    leading steps that match and whose outputs still exist are reused, the chain goes on from there.
    ``batch`` = {"input": folder, "output": folder, "skip_done": bool}: each final result is also saved as
    ``<output>/<subfolders>/<name>_chain.<ext>``, files that already have one are skipped (skip_done) and
    ``chain_batch_report.json`` in the output folder lists every file. Cancel stops the whole run.
    """
    from shared.output_run_manager import allocate_sequential_run_dir
    from shared.universal_preset import get_all_defaults

    active = [s for s in normalize_steps(steps) if s.get("enabled", True) and s["engine"] in ENGINES]
    if not active:
        yield ChainEvent([], 0, len(sources), "", "⚠️ Add at least one step to the chain.", [], finished=True)
        return
    defaults = get_all_defaults(base_dir)
    root = Path(output_root) / "chains"
    root.mkdir(parents=True, exist_ok=True)
    finals: List[str] = []
    log: List[str] = []
    runs: List[Dict[str, Any]] = []
    table: List[StepState] = []
    resume_dirs = dict(resume_dirs or {})
    batch_started = datetime.now().isoformat(timespec="seconds")
    report: List[Dict[str, Any]] = []
    skipped_done = 0
    if batch:
        log.append(f"=== Batch: {len(sources)} file(s) from {batch['input']} -> {batch['output']}")
    for video_index, source in enumerate(sources, 1):
        if cancel_flag and cancel_flag():
            log.append(f"=== Cancelled: {len(sources) - video_index + 1} file(s) not started")
            for rest in sources[video_index - 1:]:
                runs.append({"source": rest, "run_dir": "", "ok": False})
            break
        if batch and batch.get("skip_done") and not test_seconds and not resume_dirs.get(source):
            existing = batch_done(source, batch)
            if existing:
                skipped_done += 1
                log.append(f"=== Skipped {Path(source).name}: already done ({existing})")
                runs.append({"source": source, "run_dir": "", "ok": True, "skipped": True})
                report.append({"source": source, "status": "skipped (already done)", "output": existing})
                _write_batch_report(batch, chain_name, batch_started, report)
                yield ChainEvent([], video_index, len(sources), source, f"⏭️ {Path(source).name}: already done",
                                 log, runs=runs)
                continue
        table = [StepState(i + 1, s["engine"]) for i, s in enumerate(active)]
        reused: List[Optional[str]] = []
        chain_input = source
        old_dir = resume_dirs.get(source)
        if old_dir and (Path(old_dir) / "chain.json").is_file():
            run_dir = Path(old_dir)
            chain_input, reused = _resume_plan(run_dir, active)
            chain_input = chain_input if chain_input and Path(chain_input).is_file() else source
        else:
            run_dir = allocate_sequential_run_dir(root)
            run_dir.mkdir(parents=True, exist_ok=True)
        started = datetime.now().isoformat(timespec="seconds")
        if not reused and test_seconds and chain_input == source:
            chain_input = _trim_for_test(source, float(test_seconds), run_dir, log)
        is_image = Path(chain_input).suffix.lower() in IMAGE_EXTS
        current = chain_input
        log.append(f"=== Chain {chain_name or ''} on {Path(source).name} -> {run_dir}")
        for row, out in zip(table, reused):
            row.status = "done" if out else "skipped-image"
            row.output = out
            row.detail = (f"{Path(out).name} (kept from the earlier run)" if out else "skipped: video-only step")
            if out:
                current = out
        if reused:
            log.append(f"    resume: {len(reused)} finished step(s) kept, continuing at step {len(reused) + 1}")
        runs.append({"source": source, "run_dir": str(run_dir), "ok": False})
        _write_chain_json(run_dir, chain_name, chain_input, source, active, table, started, test_seconds)
        failed = False
        for position, (step, row) in enumerate(zip(active, table)):
            if row.status in ("done", "skipped-image"):
                continue
            if cancel_flag and cancel_flag():
                row.status, row.detail = "skipped", "cancelled"
                failed = True
                continue
            engine = ENGINES[step["engine"]]
            if is_image and engine.key in VIDEO_ONLY:
                row.status, row.detail = "skipped-image", f"skipped: {engine.label} takes videos only"
                log.append(f"--- Step {row.index}/{len(table)}: {engine.label} skipped (image input)")
                continue
            step_dir = run_dir / f"step{row.index}_{engine.key}"
            step_dir.mkdir(parents=True, exist_ok=True)
            row.status, row.started = "running", time.time()
            factor = step.get("factor")
            suffix = f" ×{factor:g}" if isinstance(factor, float) and factor else ""
            log.append(f"--- Step {row.index}/{len(table)}: {engine.label}{suffix} on {Path(current).name}")
            yield ChainEvent(table, video_index, len(sources), source, f"⏳ Step {row.index}/{len(table)} {engine.label}", log,
                             run_dir=str(run_dir), runs=runs)
            step_state = _quiet_globals(state)
            if factor and engine.factor_kind == "scale":
                # SeedVR2 sizes from the shared Resolution-tab factor whenever it is set; the step's own
                # copy of the state carries the step's scale so every engine reads the same factor.
                step_state["seed_controls"]["upscale_factor_val"] = float(factor)
            values = _tab_values(engine, step_state, defaults, _step_overrides(engine, step, current, step_dir))
            kwargs = dict(engine.kwargs)
            kwargs["state"] = step_state
            if engine.accepts_progress:
                kwargs["progress"] = None
            if engine.accepts_snapshot and global_settings_snapshot is not None:
                kwargs["global_settings_snapshot"] = copy.deepcopy(global_settings_snapshot)
            last_status, last_logs = "", ""
            _RUNNING["engine"] = engine
            try:
                call = engine.run_action(current, *values, **kwargs) if engine.style == "upload" else engine.run_action(*values, **kwargs)
                payloads = call if hasattr(call, "__iter__") and not isinstance(call, tuple) else [call]
                last_emit = 0.0
                for payload in payloads:
                    text = _payload_text(payload)
                    if text:
                        last_status = text
                    logs_text = _payload_logs(payload)
                    if logs_text:
                        last_logs = logs_text
                    tail = [ln for ln in last_logs.splitlines() if ln.strip()][-1:] if last_logs else []
                    row.detail = (tail[0] if tail else last_status)[:160]
                    now = time.time()
                    if now - last_emit > 0.75:
                        last_emit = now
                        if progress is not None:
                            try:
                                progress(((video_index - 1) + (position + 0.5) / len(table)) / max(1, len(sources)),
                                         desc=f"{engine.label}: {row.detail[:80]}")
                            except Exception:
                                pass
                        yield ChainEvent(table, video_index, len(sources), source,
                                         f"⏳ Step {row.index}/{len(table)} {engine.label}", log, run_dir=str(run_dir), runs=runs)
                    if cancel_flag and cancel_flag():
                        break
            except Exception as exc:  # a failing tab never takes the app down
                last_status = f"ERROR: {exc}"
            finally:
                _RUNNING["engine"] = None
            row.ended = time.time()
            output = _output_from_state(step_state)
            if output and Path(output).resolve() != Path(current).resolve() and not (cancel_flag and cancel_flag()):
                row.status, row.output = "done", output
                row.detail = f"{Path(output).name} ({_dims(output)}) in {_fmt_secs(row.ended - row.started)}"
                log.append(f"    done: {output}")
                current = output
            else:
                cancelled = bool(cancel_flag and cancel_flag())
                row.status = "failed"
                row.detail = ("cancelled" if cancelled else (last_status or "no output produced"))[:200]
                log.append(f"    {'CANCELLED' if cancelled else 'FAILED'}: {row.detail}")
                if last_logs and not cancelled:
                    log.extend(["    " + ln for ln in last_logs.splitlines()[-12:] if ln.strip()])
                failed = True
                for rest in table[position + 1:]:
                    rest.status, rest.detail = "skipped", ("cancelled" if cancelled else "previous step failed")
                _write_chain_json(run_dir, chain_name, chain_input, source, active, table, started, test_seconds)
                break
            _write_chain_json(run_dir, chain_name, chain_input, source, active, table, started, test_seconds)
            yield ChainEvent(table, video_index, len(sources), source, f"✅ Step {row.index} {engine.label} done", log,
                             run_dir=str(run_dir), runs=runs)
        if not failed and current != chain_input:
            final = run_dir / f"{Path(source).stem}_chain{Path(current).suffix}"
            try:
                shutil.copy2(current, final)
            except OSError:
                final = Path(current)
            if batch:
                target = batch_target(source, batch, test_seconds)
                target = target.with_name(target.name + Path(current).suffix)
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = target.with_name(target.name + ".part")
                    shutil.copy2(final, tmp)
                    tmp.replace(target)
                    final = target
                except OSError as exc:
                    log.append(f"    batch copy failed ({exc}); the result stays in {run_dir}")
            finals.append(str(final))
            runs[-1]["ok"] = True
            log.append(f"=== Final: {final}")
            _maybe_comparison(chain_input, str(final), state, log)
        elif not failed:
            failed = True
            log.append("    no step changed the file (every step was skipped)")
        if batch:
            failed_row = next((r for r in table if r.status == "failed"), None)
            report.append({"source": source, "status": "failed" if failed else "done",
                           "output": finals[-1] if finals and not failed else None, "run_dir": str(run_dir),
                           "detail": (f"step {failed_row.index} {failed_row.engine}: {failed_row.detail}"
                                      if failed_row else ("every step was skipped" if failed else ""))})
            _write_batch_report(batch, chain_name, batch_started, report)
        _write_chain_json(run_dir, chain_name, chain_input, source, active, table, started, test_seconds)
        yield ChainEvent(table, video_index, len(sources), chain_input,
                         ("❌ Chain stopped" if failed else f"✅ Chain finished for {Path(source).name}"), log,
                         final_output=finals[-1] if finals and not failed else None, run_dir=str(run_dir), runs=runs)
    ok = len(finals)
    if batch:
        done_all = ok + skipped_done
        extra = f" ({skipped_done} already done, skipped)" if skipped_done else ""
        status = (f"✅ Batch complete: {done_all} of {len(sources)} file(s){extra} → {batch['output']}"
                  if done_all == len(sources)
                  else f"⚠️ Batch finished: {done_all} of {len(sources)} file(s) done{extra} → {batch['output']}")
    else:
        status = (f"✅ Chain complete: {ok} of {len(sources)} video(s)" if ok == len(sources)
                  else f"⚠️ Chain finished: {ok} of {len(sources)} video(s) succeeded")
    last_source = runs[-1]["source"] if runs else (sources[-1] if sources else "")
    yield ChainEvent(table, len(sources), len(sources), last_source, status, log,
                     final_output=finals[-1] if finals else None, finished=True,
                     run_dir=runs[-1]["run_dir"] if runs else "", runs=runs)


def _maybe_comparison(source: str, final: str, state: Dict[str, Any], log: List[str]) -> None:
    if Path(final).suffix.lower() not in VIDEO_EXTS:
        return
    try:
        from shared.comparison_video_service import maybe_generate_input_vs_output_comparison

        seed = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
        comp, err = maybe_generate_input_vs_output_comparison(source, final, seed, label_input="Original",
                                                              label_output="Chain")
        if comp:
            log.append(f"    comparison: {comp}")
        elif err:
            log.append(f"    comparison skipped: {err}")
    except Exception as exc:
        log.append(f"    comparison skipped: {exc}")


def _dims(path: str) -> str:
    try:
        from shared.path_utils import get_media_dimensions

        dims = get_media_dimensions(path)
        return f"{dims[0]}x{dims[1]}" if dims else "?"
    except Exception:
        return "?"


def _fmt_secs(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"
