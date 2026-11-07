"""🗂️ Past Generations tab: browse, filter, inspect, compare and reuse every generation of the app.

The whole browser is one gr.HTML component (ui/history_assets/history.js + .css). Its server functions are
direct POSTs outside the processing queue, so browsing stays instant while a GPU job runs. "Reload parameters"
and "Send output to" go through one queued event: it writes the engine tab's settings into the shared state, resets
that tab's sync signature and clears its upload box only when it holds a file; the browser then opens the tab, whose
tab-select sync puts every value into the controls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import gradio as gr

from shared.generation_history import ENGINES, GenerationHistory, kind_of, settings_for_reload

ASSETS = Path(__file__).resolve().parent / "history_assets"

# Opens the target tab after the reload event wrote its settings. The short wait lets a cleared upload box of that
# tab finish its own change event (its previews) before the tab sync writes the controls.
NAV_JS = """
async (target) => {
  if (!target) return;
  const [elemId, label, path] = String(target).split('|');
  await new Promise((r) => setTimeout(r, 420));
  let btn = document.getElementById(elemId + '-button');
  if (!btn) btn = [...document.querySelectorAll('button[role="tab"]')].find((b) => b.textContent.trim() === label);
  if (!btn) return;
  btn.click();
  const tabs = document.getElementById('secourses-main-tabs');
  if (tabs) tabs.scrollIntoView({ behavior: 'smooth', block: 'start' });
  if (!path) return;
  // Once the tab sync has written the restored path, press Enter in that field (as the 📄 picker does),
  // so the tab shows its input preview and size info exactly as for a typed path.
  for (let i = 0; i < 30; i++) {
    await new Promise((r) => setTimeout(r, 150));
    const panel = document.getElementById(elemId);
    const field = panel && [...panel.querySelectorAll('.path-row textarea, .path-row input[type=text]')].find((f) => f.value === path);
    if (field) { field.dispatchEvent(new KeyboardEvent('keypress', { key: 'Enter', code: 'Enter', bubbles: true })); return; }
  }
}
"""

_HISTORY: Optional[GenerationHistory] = None


def get_history(root_fn: Callable[[], Path]) -> GenerationHistory:
    global _HISTORY
    if _HISTORY is None:
        _HISTORY = GenerationHistory(root_fn)
    else:
        _HISTORY.root_fn = root_fn
    return _HISTORY


def _component_js() -> str:
    css = (ASSETS / "history.css").read_text(encoding="utf-8")
    js = (ASSETS / "history.js").read_text(encoding="utf-8")
    return "const GH_CSS = " + json.dumps(css) + ";\n" + js


def history_tab(*, global_settings: Dict[str, Any], shared_state: gr.State, output_dir: Path, temp_dir: Path,
                tab_uis: Dict[str, Dict[str, Any]], tab_targets: Dict[str, tuple], chain_ui: Dict[str, Any],
                tab_sync_states: Optional[Dict[str, gr.State]] = None):
    """tab_uis: universal-preset tab key -> that tab's ui dict (needs "input_file");
    tab_targets: tab key -> (elem_id, tab label) used to open the tab;
    tab_sync_states: tab key -> the gr.State that holds that tab's tab-select sync signature."""
    from shared.universal_preset import TAB_CONFIGS

    def _root() -> Path:
        return Path(str(global_settings.get("output_dir") or output_dir))

    history = get_history(_root)

    # ---- server functions (called from the browser as server.<name>(data)) -----------------------------
    def _ids(data: Any) -> List[str]:
        return [str(i) for i in ((data or {}).get("ids") or [])] if isinstance(data, dict) else []

    def hist_index(data):
        return history.index(force=bool((data or {}).get("force")) if isinstance(data, dict) else False)

    def hist_poll(data):
        return {"sig": history.signature()}

    def hist_thumbs(data):
        return history.thumbs(_ids(data))

    def hist_details(data):
        rid = data.get("id") if isinstance(data, dict) else data
        return history.details(str(rid or ""))

    def hist_favorite(data):
        return {"changed": history.set_favorite(_ids(data), bool((data or {}).get("value")))}

    def hist_open(data):
        data = data if isinstance(data, dict) else {}
        return history.open_location(str(data.get("id") or ""), str(data.get("what") or "folder"))

    def hist_delete(data):
        return history.delete(_ids(data))

    def hist_zip(data):
        temp = Path(str(global_settings.get("temp_dir") or temp_dir))
        return history.zip_selection(_ids(data), temp)

    def hist_playable(data):
        data = data if isinstance(data, dict) else {}
        return history.playable_url(str(data.get("id") or ""), str(data.get("which") or "out"))

    view = gr.HTML(
        value="",
        html_template='<div class="gh-mount"></div>',
        js_on_load=_component_js(),
        apply_default_css=False,
        container=False,
        padding=False,
        elem_id="gh-history",
        server_functions=[hist_index, hist_poll, hist_thumbs, hist_details, hist_favorite, hist_open, hist_delete,
                          hist_zip, hist_playable],
    )
    nav = gr.Textbox(value="", visible=False)

    upload_tabs = [k for k, ui in tab_uis.items() if ui and ui.get("input_file") is not None]
    upload_boxes = [tab_uis[k]["input_file"] for k in upload_tabs]
    chain_steps = chain_ui.get("steps_state")
    chain_path = chain_ui.get("path_box")
    sync_tabs = [k for k, s in (tab_sync_states or {}).items() if s is not None]
    sync_states = [tab_sync_states[k] for k in sync_tabs]

    def _nav(tab: str, path: str = "") -> str:
        elem_id, label = tab_targets.get(tab, ("", ""))
        return f"{elem_id}|{label}|{path or ''}" if elem_id or label else ""

    def on_action(state, evt: gr.EventData, *upload_values):
        payload = getattr(evt, "_data", None) or {}
        data = payload.get("value") if isinstance(payload, dict) and isinstance(payload.get("value"), dict) else payload
        data = data if isinstance(data, dict) else {}
        skip = gr.skip()
        uploads = [skip] * len(upload_boxes)
        syncs = [skip] * len(sync_states)
        result = [state, "", *uploads, skip, skip, *syncs]

        rec = history.get(str(data.get("id") or ""))
        if rec is None:
            gr.Warning("This generation is no longer in the outputs folder. Press Refresh.")
            return result
        action = str(data.get("action") or "")
        state = state if isinstance(state, dict) else {}
        seed = state.setdefault("seed_controls", {})
        name = Path(rec["out"]).name if rec["out"] else Path(rec["dir"]).name

        def _clear_upload(tab: str) -> None:
            # An uploaded file would win over the new Input Path, so a box that holds one is cleared. An empty box
            # is left alone: Gradio fires a File's change event for every value the server sets, even None on a box
            # that is already empty.
            if tab in upload_tabs:
                i = upload_tabs.index(tab)
                if i < len(upload_values) and upload_values[i]:
                    uploads[i] = gr.update(value=None)

        def _resync(tab: str) -> None:
            # Forget the tab's last sync signature, so opening the tab writes the new settings into the controls
            # even when they equal the settings it synced last time (the controls may have been edited since).
            if tab in sync_tabs:
                syncs[sync_tabs.index(tab)] = ""

        if action == "reload" and rec["src"] == "chain":
            from shared import chain_runner as cr

            chain_json = Path(rec["dir"]) / "chain.json"
            try:
                steps = json.loads(chain_json.read_text(encoding="utf-8")).get("steps") or []
            except (OSError, ValueError):
                steps = []
            steps = cr.normalize_steps(steps)
            if not steps:
                gr.Warning("This chain run has no saved steps.")
                return result
            source = rec["inp"] if rec["inp"] and Path(rec["inp"]).exists() else ""
            gr.Info(f"🔗 Chain loaded: {len(steps)} steps" + (f", input {Path(source).name}" if source else
                                                             " (its input file is no longer on disk)"))
            return [state, _nav("chains", source), *uploads, steps if chain_steps is not None else skip,
                    (source or gr.update()) if chain_path is not None else skip, *syncs]

        if action == "reload":
            tab = ENGINES.get(rec["engine"], ENGINES["other"])[3]
            if not tab or tab not in TAB_CONFIGS or not rec["args"]:
                gr.Warning("No saved settings for this file, so there is nothing to reload.")
                return result
            current = seed.get(f"{tab}_settings") if isinstance(seed.get(f"{tab}_settings"), dict) else {}
            merged, applied = settings_for_reload(rec, current, list(TAB_CONFIGS[tab]["order"]))
            seed[f"{tab}_settings"] = merged
            _clear_upload(tab)
            _resync(tab)
            label = tab_targets.get(tab, ("", tab))[1]
            missing_input = rec["inp"] and not Path(str(merged.get("input_path") or "")).exists()
            gr.Info(f"↻ {applied} settings of {name} loaded into {label}"
                    + (" (its input file is no longer on disk)" if missing_input else ""))
            return [state, _nav(tab, str(merged.get("input_path") or "")), *uploads, skip, skip, *syncs]

        if action == "send":
            tab = str(data.get("tab") or "")
            out = rec["out"]
            if not out or not Path(out).is_file():
                gr.Warning("The output file is no longer on disk.")
                return result
            if tab == "chains":
                gr.Info(f"➜ {name} is now the Chains input")
                return [state, _nav("chains", out), *uploads, skip, out if chain_path is not None else skip, *syncs]
            if tab not in TAB_CONFIGS:
                return result
            if kind_of(out) == "image" and tab in ("ltx25", "rife"):
                gr.Warning("That tab only takes videos.")
                return result
            settings = dict(seed.get(f"{tab}_settings") or {})
            settings["input_path"] = out
            if "batch_enable" in settings or "batch_enable" in TAB_CONFIGS[tab]["order"]:
                settings["batch_enable"] = False
            if "resume_run_dir" in TAB_CONFIGS[tab]["order"]:
                settings["resume_run_dir"] = ""
            seed[f"{tab}_settings"] = settings
            _clear_upload(tab)
            _resync(tab)
            gr.Info(f"➜ {name} is now the input of {tab_targets.get(tab, ('', tab))[1]}")
            return [state, _nav(tab, out), *uploads, skip, skip, *syncs]
        return result

    outputs = [shared_state, nav, *upload_boxes,
               chain_steps if chain_steps is not None else nav, chain_path if chain_path is not None else nav,
               *sync_states]
    view.select(on_action, inputs=[shared_state, *upload_boxes], outputs=outputs, show_progress="hidden").then(
        fn=None, inputs=[nav], outputs=None, js=NAV_JS, show_progress="hidden")
    def sync_root(state):
        history.follow_state(state)

    return {"view": view, "history": history, "sync_root": sync_root}
