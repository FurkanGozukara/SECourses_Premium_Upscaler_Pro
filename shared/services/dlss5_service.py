"""DLSS 5 tab service: settings, the post-process hook used after every upscaler, and the run action."""
from __future__ import annotations

import html
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from shared.dlss5.runner import (
    DLSS5_DEFAULTS,
    IMAGE_EXTS,
    VIDEO_EXTS,
    run_dlss5,
)

DLSS5_ORDER: List[str] = [
    "input_path",
    "output_override",
    "batch_enable",
    "batch_input_path",
    "batch_output_path",
    "profile",
    "intensity",
    "detail_strength",
    "colour_strength",
    "detail_radius",
    "processing_scale",
    "temporal",
    "motion_quality",
    "scene_cut_threshold",
    "tone_strength",
    "structure_strength",
    "precision",
    "speed_mode",
    "max_frames",
    "keep_model_loaded",
    "global_postprocess",
]

# Output-tab encoding keys a DLSS 5 render inherits (same codec, quality and audio as the upscalers).
_ENCODE_KEYS = ("video_codec", "video_quality", "video_preset", "h265_tune", "av1_film_grain",
                "av1_film_grain_denoise", "pixel_format", "audio_codec", "audio_bitrate",
                "image_output_format", "image_output_quality")


def dlss5_defaults() -> Dict[str, Any]:
    values = {
        "input_path": "",
        "output_override": "",
        "batch_enable": False,
        "batch_input_path": "",
        "batch_output_path": "",
        "global_postprocess": False,
    }
    values.update(DLSS5_DEFAULTS)
    return {key: values.get(key) for key in DLSS5_ORDER}


def settings_from_values(values: List[Any]) -> Dict[str, Any]:
    defaults = dlss5_defaults()
    data = dict(zip(DLSS5_ORDER, list(values) + [None] * (len(DLSS5_ORDER) - len(values))))
    return {key: (defaults[key] if data.get(key) is None else data[key]) for key in DLSS5_ORDER}


def _tab_settings(seed_controls: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dlss5_defaults()
    raw = (seed_controls or {}).get("dlss5_settings") if isinstance(seed_controls, dict) else None
    if isinstance(raw, dict):
        merged.update({k: v for k, v in raw.items() if v is not None})
    return merged


def render_settings(tab: Dict[str, Any], seed_controls: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Tab settings plus the Output tab's encoding and the global GPU."""
    settings = dict(tab)
    output = (seed_controls or {}).get("output_settings") if isinstance(seed_controls, dict) else None
    if isinstance(output, dict):
        for key in _ENCODE_KEYS:
            if output.get(key) not in (None, ""):
                settings[key] = output[key]
    try:
        from shared.gpu_utils import get_global_gpu_override

        settings["cuda_device"] = get_global_gpu_override(seed_controls or {})
    except Exception:
        settings["cuda_device"] = "0"
    return settings


def global_dlss5_enabled(seed_controls: Optional[Dict[str, Any]]) -> bool:
    return bool(_tab_settings(seed_controls).get("global_postprocess"))


def maybe_apply_global_dlss5(
    output_path: Optional[str],
    seed_controls: Optional[Dict[str, Any]],
    on_log: Optional[Callable[[str], None]] = None,
    base_dir: Optional[Path] = None,
) -> Tuple[Optional[str], str]:
    """Post-process an upscaler's finished output with the DLSS 5 tab settings (keeps the original)."""
    if not output_path:
        return None, ""
    source = Path(str(output_path))
    if not source.is_file() or not global_dlss5_enabled(seed_controls):
        return None, ""
    tab = _tab_settings(seed_controls)
    if source.suffix.lower() not in VIDEO_EXTS:
        return None, ""
    if source.stem.endswith("_DLSS5"):
        return None, ""  # already rendered (chained post-processing)
    settings = render_settings(tab, seed_controls)
    settings.update({"input_path": str(source), "output_dir": str(source.parent), "output_path": "",
                     "max_frames": 0, "keep_model_loaded": False})
    if on_log:
        on_log(f"DLSS 5 post-process: {tab.get('profile')} profile, intensity {tab.get('intensity')}...\n")
    root = Path(base_dir) if base_dir else Path(__file__).resolve().parents[2]
    result = run_dlss5(settings, root, on_progress=(lambda m: on_log(f"[DLSS 5] {m}") if on_log and not str(m).startswith("FRAME_PROGRESS") else None))
    if result.returncode == 0 and result.output_path and Path(result.output_path).exists():
        message = f"DLSS 5 post-process complete: {result.output_path}"
        if on_log:
            on_log(message + "\n")
        return result.output_path, message
    tail = [line for line in str(result.log).splitlines() if line.strip()][-1:] or ["unknown error"]
    message = f"DLSS 5 post-process failed: {tail[0]}"
    if on_log:
        on_log(message + "\n")
    return None, message


def _status_html(title: str, subtitle: str, spinning: bool = True) -> str:
    spinner = "" if spinning else ' style="opacity:0.45; animation:none;"'
    return (
        '<div class="processing-banner">'
        f'<div class="processing-spinner"{spinner}></div>'
        '<div class="processing-col">'
        f'<div class="processing-text">{html.escape(title)}</div>'
        f'<div class="processing-sub">{html.escape(subtitle)}</div>'
        "</div></div>"
    )


def _batch_items(folder: Path) -> List[str]:
    from shared.path_utils import list_directory_entries_sorted

    items: List[str] = []
    for child in list_directory_entries_sorted(folder):
        child = Path(child)
        if child.is_file() and child.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS:
            items.append(str(child))
    return items


def build_dlss5_callbacks(
    *,
    global_settings: Dict[str, Any],
    base_dir: Path,
    output_dir: Path,
):
    import gradio as gr

    cancel_event = threading.Event()

    def _output_root(state: Dict[str, Any]) -> Path:
        seed = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
        glob = seed.get("global_settings", {}) if isinstance(seed, dict) else {}
        return Path(str((glob or {}).get("output_dir") or global_settings.get("output_dir") or output_dir))

    def _media(out_path: Optional[str]):
        if out_path and Path(out_path).is_file():
            suffix = Path(out_path).suffix.lower()
            if suffix in VIDEO_EXTS:
                return gr.update(value=None, visible=False), gr.update(value=out_path, visible=True)
            if suffix in IMAGE_EXTS:
                return gr.update(value=out_path, visible=True), gr.update(value=None, visible=False)
        return gr.update(), gr.update()

    def _comparison(src: str, out: Optional[str]):
        if not out or not Path(out).is_file():
            return gr.update(), gr.update()
        if Path(out).suffix.lower() in VIDEO_EXTS:
            from shared.comparison_unified import create_video_comparison_slider

            return gr.update(value=None, visible=False), gr.update(value=create_video_comparison_slider(src, out), visible=True)
        return gr.update(value=(src, out), visible=True), gr.update(value="", visible=False)

    def _single_target(source: str, override: str, root: Path, image_format: str = "png") -> str:
        from shared.output_run_manager import numbered_single_image_output_path, prepare_single_video_run

        if Path(source).suffix.lower() in IMAGE_EXTS:
            if override and Path(override).suffix:
                return override
            ext = {"jpg": ".jpg", "jpeg": ".jpg", "webp": ".webp"}.get(str(image_format or "png").lower(), ".png")
            return str(numbered_single_image_output_path(Path(override) if override else root,
                                                         f"{Path(source).stem}_DLSS5{ext}", ext))
        if Path(source).is_dir():
            return override or str(root / f"{Path(source).name}_DLSS5")
        run_paths, explicit = prepare_single_video_run(
            output_root_fallback=root, output_override_raw=override or None, input_path=source,
            original_filename=Path(source).name, model_label="DLSS5", mode="in_app",
        )
        return str(explicit or (run_paths.run_dir / f"{Path(source).stem}_DLSS5.mp4"))

    def run_action(upload, *values, state=None, progress=None):
        """Generator: (status, logs, progress_html, image, video, slider, comparison_html, state)."""
        state = state if isinstance(state, dict) else {"seed_controls": {}}
        seed_controls = state.setdefault("seed_controls", {})
        tab = settings_from_values(list(values))
        seed_controls["dlss5_settings"] = dict(tab)
        cancel_event.clear()
        jobs: List[Tuple[str, str]] = []
        root = _output_root(state)
        if bool(tab.get("batch_enable")):
            folder = Path(str(tab.get("batch_input_path") or "").strip().strip('"'))
            if not folder.is_dir():
                yield ("❌ Batch input folder not found.", "", gr.update(visible=False), gr.update(), gr.update(),
                       gr.update(), gr.update(), state)
                return
            out_dir = Path(str(tab.get("batch_output_path") or "").strip().strip('"') or (root / f"{folder.name}_DLSS5"))
            out_dir.mkdir(parents=True, exist_ok=True)
            for item in _batch_items(folder):
                suffix = ".mp4" if Path(item).suffix.lower() in VIDEO_EXTS else ".png"
                jobs.append((item, str(out_dir / f"{Path(item).stem}_DLSS5{suffix}")))
        else:
            source = str(upload or "").strip() or str(tab.get("input_path") or "").strip().strip('"')
            if not source or not Path(source).exists():
                yield ("⚠️ Upload a video / image or enter an Input Path first.", "", gr.update(visible=False),
                       gr.update(), gr.update(), gr.update(), gr.update(), state)
                return
            image_format = render_settings(tab, seed_controls).get("image_output_format", "png")
            jobs.append((source, _single_target(source, str(tab.get("output_override") or "").strip(), root, image_format)))
        if not jobs:
            yield ("⚠️ No videos or images found in the batch folder.", "", gr.update(visible=False), gr.update(),
                   gr.update(), gr.update(), gr.update(), state)
            return

        logs: List[str] = []
        last_output: Optional[str] = None
        failures = 0
        started = time.time()
        for index, (source, target) in enumerate(jobs, 1):
            settings = render_settings(tab, seed_controls)
            settings.update({"input_path": source, "output_path": target})
            lines: "queue.Queue[str]" = queue.Queue()
            holder: Dict[str, Any] = {}

            def _work(s=settings):
                holder["result"] = run_dlss5(s, base_dir, on_progress=lambda m: lines.put(str(m).strip()),
                                             cancel_event=cancel_event)

            worker = threading.Thread(target=_work, name="dlss5-run", daemon=True)
            worker.start()
            label = f"[{index}/{len(jobs)}] {Path(source).name}" if len(jobs) > 1 else Path(source).name
            detail = "Loading DLSS 5..."
            last_yield = 0.0
            while worker.is_alive() or not lines.empty():
                try:
                    line = lines.get(timeout=0.2)
                except queue.Empty:
                    line = ""
                if line:
                    if line.startswith("FRAME_PROGRESS "):
                        detail = line[len("FRAME_PROGRESS "):]
                        if logs and logs[-1].startswith("FRAME_PROGRESS "):
                            logs[-1] = line
                        else:
                            logs.append(line)
                        if progress is not None:
                            try:
                                done, _, rest = detail.partition("/")
                                total = rest.split(" ", 1)[0]
                                if done.strip().isdigit() and total.isdigit() and int(total) > 0:
                                    progress((index - 1 + int(done) / int(total)) / len(jobs), desc=f"DLSS 5 {label}: {detail}")
                            except Exception:
                                pass
                    else:
                        logs.append(line)
                        detail = line[:140]
                now = time.time()
                if now - last_yield > 0.6:
                    last_yield = now
                    yield (f"⏳ DLSS 5 rendering {label}", "\n".join(logs[-300:]),
                           gr.update(value=_status_html(f"DLSS 5 — {label}", f"{detail} | {int(now - started)}s"), visible=True),
                           gr.update(), gr.update(), gr.update(), gr.update(), state)
            worker.join()
            result = holder.get("result")
            if result is None or result.returncode != 0:
                failures += 1
                if cancel_event.is_set():
                    break
                continue
            last_output = result.output_path
            seed_controls["dlss5_last_output_path"] = last_output
            seed_controls["last_output_dir"] = str(Path(last_output).parent)
            seed_controls["last_output_path"] = last_output if Path(last_output).is_file() else None
            if Path(last_output).suffix.lower() in VIDEO_EXTS:
                try:
                    from shared.comparison_video_service import maybe_generate_input_vs_output_comparison

                    comp, comp_err = maybe_generate_input_vs_output_comparison(
                        source, last_output, seed_controls, label_input="Original", label_output="DLSS 5",
                        on_progress=lambda m: logs.append(str(m).strip()),
                    )
                    if comp:
                        logs.append(f"Comparison video: {comp}")
                    elif comp_err:
                        logs.append(f"Comparison video skipped: {comp_err}")
                except Exception as exc:
                    logs.append(f"Comparison video skipped: {exc}")

        elapsed = int(time.time() - started)
        if cancel_event.is_set():
            status = "⏹️ DLSS 5 cancelled."
        elif failures and not last_output:
            status = f"❌ DLSS 5 failed ({failures} of {len(jobs)}). See the log."
        elif failures:
            status = f"⚠️ DLSS 5 finished with {failures} failure(s) in {elapsed}s."
        else:
            status = f"✅ DLSS 5 complete in {elapsed}s" + (f" ({len(jobs)} files)" if len(jobs) > 1 else "")
        img_upd, vid_upd = _media(last_output)
        slider_upd, comp_upd = _comparison(jobs[-1][0], last_output) if len(jobs) == 1 else (gr.update(), gr.update())
        if progress is not None and last_output:
            progress(1.0, desc="DLSS 5 complete")
        yield (status, "\n".join(logs[-400:]), gr.update(value="", visible=False), img_upd, vid_upd, slider_upd, comp_upd, state)

    def cancel_action():
        cancel_event.set()
        return "⏹️ Cancelling DLSS 5 after the current frame..."

    return {"run_action": run_action, "cancel_action": cancel_action, "defaults": dlss5_defaults()}
