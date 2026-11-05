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


ENGINES: Dict[str, ChainEngine] = {}
# Display order of the engine palette.
ENGINE_ORDER = ["seedvr2", "flashvsr", "sparkvsr", "gan", "rtx", "ltx25", "open_h3", "dlss5", "face", "rife"]
ENGINE_META = {
    "seedvr2": ("SeedVR2", "🎬", "scale", ("dit_model", "resolution")),
    "flashvsr": ("FlashVSR+", "⚡", "scale", ("version", "mode")),
    "sparkvsr": ("SparkVSR", "✨", "scale", ("model", "upscale_mode")),
    "gan": ("Image GAN", "🖼️", "scale", ("model",)),
    "rtx": ("RTX Super Res", "🚀", "scale", ("quality_preset",)),
    "ltx25": ("LTX 2.5", "🎞️", "", ("model",)),
    "open_h3": ("Open-H3", "🌀", "scale", ("model",)),
    "dlss5": ("DLSS 5", "💎", "", ("profile", "intensity")),
    "face": ("Face Restore", "👤", "", ("restoration_model", "face_strength")),
    "rife": ("RIFE FPS", "⏱️", "fps", ("model",)),
}


def register_chain_engine(key: str, run_action: Callable, *, style: str = "upload", progress: bool = True,
                          snapshot: bool = True, **kwargs: Any) -> None:
    """Register a tab's ``run_action``; ``kwargs`` are fixed keyword arguments (e.g. preview_only=False)."""
    label, icon, factor_kind, summary = ENGINE_META.get(key, (key, "🔧", "", ()))
    ENGINES[key] = ChainEngine(key, label, icon, run_action, style, dict(kwargs), progress, snapshot,
                               factor_kind, summary)


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
}


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
) -> Iterator[ChainEvent]:
    """Run ``steps`` over every source; yields ChainEvents for the UI."""
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
    for video_index, source in enumerate(sources, 1):
        run_dir = allocate_sequential_run_dir(root)
        run_dir.mkdir(parents=True, exist_ok=True)
        table = [StepState(i + 1, s["engine"]) for i, s in enumerate(active)]
        (run_dir / "chain.json").write_text(json.dumps(
            {"name": chain_name, "source": source, "steps": active, "started": datetime.now().isoformat(timespec="seconds")},
            indent=2, ensure_ascii=False), encoding="utf-8")
        current = source
        log.append(f"=== Chain {chain_name or ''} on {Path(source).name} -> {run_dir}")
        failed = False
        for position, (step, row) in enumerate(zip(active, table)):
            if cancel_flag and cancel_flag():
                row.status, row.detail = "skipped", "cancelled"
                failed = True
                continue
            engine = ENGINES[step["engine"]]
            step_dir = run_dir / f"step{row.index}_{engine.key}"
            step_dir.mkdir(parents=True, exist_ok=True)
            row.status, row.started = "running", time.time()
            factor = step.get("factor")
            suffix = f" ×{factor:g}" if isinstance(factor, float) and factor else ""
            log.append(f"--- Step {row.index}/{len(table)}: {engine.label}{suffix} on {Path(current).name}")
            yield ChainEvent(table, video_index, len(sources), source, f"⏳ Step {row.index}/{len(table)} {engine.label}", log)
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
                                         f"⏳ Step {row.index}/{len(table)} {engine.label}", log)
                    if cancel_flag and cancel_flag():
                        break
            except Exception as exc:  # a failing tab never takes the app down
                last_status = f"ERROR: {exc}"
            row.ended = time.time()
            output = _output_from_state(step_state)
            if output and Path(output).resolve() != Path(current).resolve():
                row.status, row.output = "done", output
                row.detail = f"{Path(output).name} ({_dims(output)}) in {_fmt_secs(row.ended - row.started)}"
                log.append(f"    done: {output}")
                current = output
            else:
                row.status = "failed"
                row.detail = (last_status or "no output produced")[:200]
                log.append(f"    FAILED: {row.detail}")
                if last_logs:
                    log.extend(["    " + ln for ln in last_logs.splitlines()[-12:] if ln.strip()])
                failed = True
                for rest in table[position + 1:]:
                    rest.status, rest.detail = "skipped", "previous step failed"
                break
            yield ChainEvent(table, video_index, len(sources), source, f"✅ Step {row.index} {engine.label} done", log)
        if not failed and current != source:
            final = run_dir / f"{Path(source).stem}_chain{Path(current).suffix}"
            try:
                shutil.copy2(current, final)
            except OSError:
                final = Path(current)
            finals.append(str(final))
            log.append(f"=== Final: {final}")
            _maybe_comparison(source, str(final), state, log)
        yield ChainEvent(table, video_index, len(sources), source,
                         ("❌ Chain stopped" if failed else f"✅ Chain finished for {Path(source).name}"), log,
                         final_output=finals[-1] if finals and not failed else None)
    ok = len(finals)
    status = (f"✅ Chain complete: {ok} of {len(sources)} video(s)" if ok == len(sources)
              else f"⚠️ Chain finished: {ok} of {len(sources)} video(s) succeeded")
    yield ChainEvent(table, len(sources), len(sources), sources[-1] if sources else "", status, log,
                     final_output=finals[-1] if finals else None, finished=True)


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
