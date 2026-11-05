"""Chains tab: build a pipeline of tabs (e.g. SeedVR2 2x -> FlashVSR+ 2x -> DLSS 5), save it, run it on videos."""
from __future__ import annotations

import html
import threading
from pathlib import Path
from typing import Any, Dict, List

import gradio as gr

from shared import chain_runner as cr
from shared.video_comparison_slider import get_video_comparison_js_on_load

DEFAULT_CHAIN = "⭐ SeedVR2 2x → FlashVSR+ 2x → DLSS 5"

HERO_HTML = """
<div class="chain-hero">
  <div class="chain-hero-title">🔗 Chains</div>
  <div class="chain-hero-sub">Stack upscalers and effects into one pipeline: the output of each step feeds the next.
  Every step runs with the settings you set on its own tab (model, quality, VRAM Auto Tune, output codec); a step can
  override only its scale. Save chains as presets and run them on one video or a whole folder.</div>
</div>
"""


def _step_summary(engine: str, state: Dict[str, Any]) -> str:
    meta = cr.ENGINE_META.get(engine)
    seed = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
    settings = seed.get(f"{engine}_settings") or {}
    parts = []
    for key in (meta[3] if meta else ()):
        value = settings.get(key)
        if value not in (None, ""):
            text = str(value)
            parts.append(text if len(text) <= 38 else text[:35] + "...")
    return " · ".join(parts) if parts else "tab defaults"


def flow_html(steps: List[Dict[str, Any]], state: Dict[str, Any]) -> str:
    steps = cr.normalize_steps(steps)
    nodes = ['<div class="chain-node chain-node-io">🎬 Input</div>']
    total_scale = 1.0
    fps = 1
    for i, step in enumerate(steps, 1):
        engine = step["engine"]
        meta = cr.ENGINE_META.get(engine, (engine, "🔧", "", ()))
        factor = step.get("factor")
        if step.get("enabled", True):
            if factor and meta[2] == "scale":
                total_scale *= float(factor)
            if factor and meta[2] == "fps":
                fps *= int(float(factor))
        badge = ""
        if factor and meta[2] == "scale":
            badge = f'<span class="chain-badge">×{float(factor):g}</span>'
        elif factor and meta[2] == "fps":
            badge = f'<span class="chain-badge">{int(float(factor))}× FPS</span>'
        cls = "chain-node" + ("" if step.get("enabled", True) else " chain-node-off")
        nodes.append(
            f'<div class="{cls}"><div class="chain-node-top"><span class="chain-node-num">{i}</span> {meta[1]} '
            f'{html.escape(meta[0])} {badge}</div><div class="chain-node-sub">{html.escape(_step_summary(engine, state))}</div></div>'
        )
    nodes.append('<div class="chain-node chain-node-io">📦 Output</div>')
    arrow = '<div class="chain-arrow">➜</div>'
    summary = f"{len(steps)} step(s)"
    if total_scale != 1.0:
        summary += f" · total scale ×{total_scale:g}"
    if fps != 1:
        summary += f" · {fps}× frame rate"
    if not steps:
        summary = "Empty chain: add steps with the buttons above."
    return f'<div class="chain-flow">{arrow.join(nodes)}</div><div class="chain-flow-summary">{html.escape(summary)}</div>'


def timeline_html(event: "cr.ChainEvent | None") -> str:
    if event is None or not event.steps:
        return ""
    icons = {"pending": "⏸️", "running": "⏳", "done": "✅", "failed": "❌", "skipped": "⏭️"}
    rows = []
    for row in event.steps:
        meta = cr.ENGINE_META.get(row.engine, (row.engine, "🔧", "", ()))
        rows.append(
            f'<div class="chain-tl-row chain-tl-{row.status}"><span class="chain-tl-icon">{icons.get(row.status, "•")}</span>'
            f'<span class="chain-tl-name">{row.index}. {meta[1]} {html.escape(meta[0])}</span>'
            f'<span class="chain-tl-detail">{html.escape(row.detail or row.status)}</span></div>'
        )
    head = f"Video {event.video_index} of {event.video_count}: {html.escape(Path(event.source).name)}" if event.source else ""
    return f'<div class="chain-timeline"><div class="chain-tl-head">{head}</div>{"".join(rows)}</div>'


def chain_tab(*, global_settings: Dict[str, Any], shared_state: gr.State, base_dir: Path, output_dir: Path):
    from shared.processing_queue import get_processing_queue_manager, resolve_queue_gpu_resources
    from shared.queue_state import snapshot_global_settings, snapshot_queue_state

    cancel = threading.Event()
    initial = cr.load_chain_preset(base_dir, DEFAULT_CHAIN)
    steps_state = gr.State(initial)

    with gr.Row(elem_classes=["chain-tab"]):
        with gr.Column(scale=3):
            gr.HTML(HERO_HTML)
            with gr.Group(elem_classes=["chain-card"]):
                gr.Markdown("#### ① Videos", elem_classes=["chain-card-title"])
                with gr.Row():
                    files = gr.File(label="Videos or images (several allowed)", file_count="multiple", type="filepath",
                                    file_types=["video", "image"], scale=3)
                    path_box = gr.Textbox(label="...or a file / folder path", placeholder="D:/videos  or  D:/videos/clip.mp4",
                                          scale=2)

            with gr.Group(elem_classes=["chain-card"]):
                gr.Markdown("#### ② Build the chain", elem_classes=["chain-card-title"])
                gr.Markdown("Click an engine to append it. Each step uses the current settings of its tab.",
                            elem_classes=["resolution-info"])
                palette = {}
                with gr.Row(elem_classes=["chain-palette"]):
                    for key in cr.ENGINE_ORDER:
                        label, icon, _kind, _summary = cr.ENGINE_META[key]
                        palette[key] = gr.Button(f"➕ {icon} {label}", size="sm", elem_classes=["chain-add-btn"], min_width=110)
                flow = gr.HTML(flow_html(initial, shared_state.value if isinstance(shared_state.value, dict) else {}))

                @gr.render(inputs=[steps_state])
                def render_steps(steps):
                    steps = cr.normalize_steps(steps)
                    if not steps:
                        gr.Markdown("*No steps yet.*")
                        return
                    for i, step in enumerate(steps):
                        meta = cr.ENGINE_META.get(step["engine"], (step["engine"], "🔧", "", ()))
                        with gr.Row(elem_classes=["chain-step"], equal_height=True, key=f"row-{i}-{step['engine']}"):
                            gr.HTML(f'<div class="chain-step-title"><span class="chain-node-num">{i + 1}</span> '
                                    f'{meta[1]} {html.escape(meta[0])}</div>', key=f"title-{i}-{step['engine']}")
                            enabled = gr.Checkbox(value=step.get("enabled", True), label="On", scale=0, min_width=70,
                                                  key=f"on-{i}-{step['engine']}")
                            factor = None
                            if meta[2] == "scale":
                                factor = gr.Dropdown(choices=[("Tab setting", 0.0), ("1x", 1.0), ("1.5x", 1.5), ("2x", 2.0),
                                                              ("3x", 3.0), ("4x", 4.0)],
                                                     value=float(step.get("factor") or 0.0), label="Scale", scale=1,
                                                     min_width=120, key=f"f-{i}-{step['engine']}")
                            elif meta[2] == "fps":
                                factor = gr.Dropdown(choices=[("Tab setting", 0.0), ("2x FPS", 2.0), ("4x FPS", 4.0), ("8x FPS", 8.0)],
                                                     value=float(step.get("factor") or 0.0), label="Frame rate", scale=1,
                                                     min_width=120, key=f"f-{i}-{step['engine']}")
                            up = gr.Button("▲", size="sm", scale=0, min_width=44, key=f"up-{i}-{step['engine']}")
                            down = gr.Button("▼", size="sm", scale=0, min_width=44, key=f"dn-{i}-{step['engine']}")
                            remove = gr.Button("✕", size="sm", scale=0, min_width=44, variant="stop",
                                               key=f"rm-{i}-{step['engine']}")

                        def _move(cur, idx=i, delta=-1):
                            cur = cr.normalize_steps(cur)
                            j = idx + delta
                            if 0 <= j < len(cur):
                                cur[idx], cur[j] = cur[j], cur[idx]
                            return cur

                        def _remove(cur, idx=i):
                            cur = cr.normalize_steps(cur)
                            if 0 <= idx < len(cur):
                                cur.pop(idx)
                            return cur

                        def _toggle(value, cur, idx=i):
                            cur = cr.normalize_steps(cur)
                            if 0 <= idx < len(cur):
                                cur[idx]["enabled"] = bool(value)
                            return cur

                        def _factor(value, cur, idx=i):
                            cur = cr.normalize_steps(cur)
                            if 0 <= idx < len(cur):
                                if value:
                                    cur[idx]["factor"] = float(value)
                                else:
                                    cur[idx].pop("factor", None)
                            return cur

                        up.click(lambda cur, idx=i: _move(cur, idx, -1), inputs=[steps_state], outputs=[steps_state], show_progress="hidden")
                        down.click(lambda cur, idx=i: _move(cur, idx, 1), inputs=[steps_state], outputs=[steps_state], show_progress="hidden")
                        remove.click(_remove, inputs=[steps_state], outputs=[steps_state], show_progress="hidden")
                        enabled.input(_toggle, inputs=[enabled, steps_state], outputs=[steps_state], show_progress="hidden")
                        if factor is not None:
                            factor.input(_factor, inputs=[factor, steps_state], outputs=[steps_state], show_progress="hidden")

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear chain", size="sm", elem_classes=["sec-btn-slate"])

            with gr.Group(elem_classes=["chain-card"]):
                gr.Markdown("#### ③ Chain presets", elem_classes=["chain-card-title"])
                with gr.Row():
                    preset_dd = gr.Dropdown(label="Saved chains", choices=cr.list_chain_presets(base_dir), value=DEFAULT_CHAIN,
                                            scale=3, allow_custom_value=False)
                    preset_name = gr.Textbox(label="Name for Save", placeholder="My 4K remaster", scale=2)
                with gr.Row():
                    load_btn = gr.Button("📂 Load", elem_classes=["sec-btn-blue"])
                    save_btn = gr.Button("💾 Save", elem_classes=["sec-btn-teal"])
                    delete_btn = gr.Button("🗑️ Delete", elem_classes=["sec-btn-orange"])
                preset_status = gr.Markdown("")

        with gr.Column(scale=2):
            gr.Markdown("### Run")
            status = gr.Markdown("Ready. Pick videos, build or load a chain, then press **Run Chain**.",
                                 elem_classes=["runtime-status-box"])
            with gr.Row():
                run_btn = gr.Button("▶️ Run Chain", variant="primary", elem_classes=["action-btn", "sec-btn-emerald"], scale=3)
                cancel_btn = gr.Button("⏹️ Cancel", variant="stop", elem_classes=["action-btn", "sec-btn-red"], scale=1)
            timeline = gr.HTML("")
            out_video = gr.Video(label="Final output", interactive=False, visible=False)
            out_image = gr.Image(label="Final output", type="filepath", interactive=False, visible=False)
            slider = gr.ImageSlider(label="Before / After", interactive=False, visible=False, max_height=900,
                                    buttons=["download", "fullscreen"], elem_classes=["native-image-comparison-slider"])
            comparison = gr.HTML(value="", js_on_load=get_video_comparison_js_on_load(), visible=False)
            with gr.Accordion("Log", open=False):
                log_box = gr.Textbox(label="Log", lines=16, max_lines=28, interactive=False, autoscroll=True)

    # --- editing -----------------------------------------------------------------------------
    for key, button in palette.items():
        button.click(lambda cur, k=key: cr.normalize_steps(cur) + [{"engine": k, "enabled": True}],
                     inputs=[steps_state], outputs=[steps_state], show_progress="hidden")
    clear_btn.click(lambda: [], outputs=[steps_state], show_progress="hidden")
    # A state written by a non-queued event does not fire .change in this app, so every editing event
    # above is queued; the flow view follows the state through .change.
    steps_state.change(flow_html, inputs=[steps_state, shared_state], outputs=[flow], show_progress="hidden")

    def _load(name):
        steps = cr.load_chain_preset(base_dir, name)
        if not steps:
            return gr.update(), f"⚠️ Chain '{name}' not found."
        return steps, f"✅ Loaded chain **{name}** ({len(steps)} steps)."

    def _save(name, selected, steps):
        target = str(name or "").strip() or (selected if selected and selected not in cr.BUILTIN_CHAINS else "")
        if not target:
            return gr.update(), "⚠️ Type a name for the chain first."
        if not cr.normalize_steps(steps):
            return gr.update(), "⚠️ The chain is empty."
        saved = cr.save_chain_preset(base_dir, target, steps)
        return gr.update(choices=cr.list_chain_presets(base_dir), value=saved), f"💾 Saved chain **{saved}**."

    def _delete(name):
        if name in cr.BUILTIN_CHAINS:
            return gr.update(), "ℹ️ Built-in example chains cannot be deleted."
        ok = cr.delete_chain_preset(base_dir, name)
        choices = cr.list_chain_presets(base_dir)
        return gr.update(choices=choices, value=choices[0] if choices else None), (
            f"🗑️ Deleted chain **{name}**." if ok else f"⚠️ Chain '{name}' not found.")

    load_btn.click(_load, inputs=[preset_dd], outputs=[steps_state, preset_status], show_progress="hidden")
    preset_dd.input(_load, inputs=[preset_dd], outputs=[steps_state, preset_status], show_progress="hidden")
    save_btn.click(_save, inputs=[preset_name, preset_dd, steps_state], outputs=[preset_dd, preset_status], queue=False)
    delete_btn.click(_delete, inputs=[preset_dd], outputs=[preset_dd, preset_status], queue=False)

    # --- running -----------------------------------------------------------------------------
    def _final_updates(final: str | None, source: str):
        hidden = gr.update(value=None, visible=False)
        if not final or not Path(final).is_file():
            return hidden, hidden, gr.update(visible=False), gr.update(value="", visible=False)
        if Path(final).suffix.lower() in cr.VIDEO_EXTS:
            from shared.comparison_unified import create_video_comparison_slider

            return (gr.update(value=final, visible=True), hidden, gr.update(visible=False),
                    gr.update(value=create_video_comparison_slider(source, final), visible=True))
        return hidden, gr.update(value=final, visible=True), gr.update(value=(source, final), visible=True), \
            gr.update(value="", visible=False)

    def run_chain_ui(uploads, path_text, steps, preset_label, state, progress=gr.Progress()):
        live_state = state if isinstance(state, dict) else {}
        sources = cr.collect_videos(uploads, path_text)
        if not sources:
            yield ("⚠️ Pick at least one video or image (upload or path).", gr.update(), gr.update(),
                   gr.update(), gr.update(), gr.update(), gr.update(), live_state)
            return
        missing = [s["engine"] for s in cr.normalize_steps(steps) if s["engine"] not in cr.ENGINES]
        if missing:
            yield (f"❌ Unknown chain step(s): {', '.join(missing)}", gr.update(), gr.update(), gr.update(), gr.update(),
                   gr.update(), gr.update(), live_state)
            return
        cancel.clear()
        queued_state = snapshot_queue_state(live_state)
        queued_globals = snapshot_global_settings(global_settings)
        manager = get_processing_queue_manager()
        keys, label = resolve_queue_gpu_resources(queued_state, queued_globals)
        ticket = manager.submit("Chain", str(preset_label or "Chain"), resource_keys=keys, resource_label=label)
        try:
            while not ticket.start_event.wait(timeout=0.5):
                if ticket.cancel_event.is_set() or cancel.is_set():
                    yield ("⏹️ Chain cancelled while waiting in the queue.", gr.update(), gr.update(), gr.update(),
                           gr.update(), gr.update(), gr.update(), live_state)
                    return
                yield (f"⏳ Waiting in the processing queue ({ticket.job_id})...", gr.update(), gr.update(), gr.update(),
                       gr.update(), gr.update(), gr.update(), live_state)
            seed = queued_state.get("seed_controls", {})
            glob = seed.get("global_settings", {}) if isinstance(seed, dict) else {}
            out_root = Path(str((glob or {}).get("output_dir") or global_settings.get("output_dir") or output_dir))
            last_final, last_source = None, sources[-1]
            for event in cr.run_chain(sources, steps, queued_state, base_dir=base_dir, output_root=out_root,
                                      chain_name=str(preset_label or ""), progress=progress,
                                      cancel_flag=lambda: cancel.is_set() or ticket.cancel_event.is_set(),
                                      global_settings_snapshot=queued_globals):
                if event.final_output:
                    last_final, last_source = event.final_output, event.source
                if event.finished:
                    vid, img, sl, comp = _final_updates(last_final, last_source)
                    if last_final:
                        live_state.setdefault("seed_controls", {})["last_output_path"] = last_final
                        live_state["seed_controls"]["last_output_dir"] = str(Path(last_final).parent)
                    yield (event.status, "\n".join(event.log[-500:]), timeline_html(event), vid, img, sl, comp, live_state)
                else:
                    yield (event.status, "\n".join(event.log[-500:]), timeline_html(event), gr.update(), gr.update(),
                           gr.update(), gr.update(), live_state)
        finally:
            manager.complete(ticket.job_id)

    run_btn.click(run_chain_ui, inputs=[files, path_box, steps_state, preset_dd, shared_state],
                  outputs=[status, log_box, timeline, out_video, out_image, slider, comparison, shared_state],
                  concurrency_limit=32, concurrency_id="app_processing_queue", trigger_mode="multiple")

    def _cancel():
        cancel.set()
        return "⏹️ Cancelling: the chain stops after the current step (use that tab's Cancel to stop the step itself)."

    cancel_btn.click(_cancel, outputs=[status], queue=False)
    return {"steps_state": steps_state, "flow": flow}
