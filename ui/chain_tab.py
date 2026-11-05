"""Chains tab: build a pipeline of tabs (e.g. SeedVR2 2x -> FlashVSR+ 2x -> DLSS 5), save it, run it on videos."""
from __future__ import annotations

import html
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr

from shared import chain_runner as cr
from shared.video_comparison_slider import get_video_comparison_js_on_load
from ui.media_preview import preview_updates

DEFAULT_CHAIN = cr.DEFAULT_CHAIN

HERO_HTML = """
<div class="chain-hero">
  <div class="chain-hero-title">🔗 Chains</div>
  <div class="chain-hero-sub">Stack upscalers and effects into one pipeline: the output of each step feeds the next.
  Every step runs with the settings you set on its own tab (model, quality, VRAM Auto Tune, output codec); a step can
  override only its scale. Save chains as presets and run them on one video or a whole folder.</div>
</div>
"""

PRESET_NOTE = ("Chain presets store only the chain: its steps, their order, On / Off and scale (`presets/chains`). "
               "Loading or saving one never changes a tab's settings. An app preset (Global Settings) remembers "
               "which chain preset you use and loads it with the app preset; it never stores the steps themselves.")

ICONS = {"pending": "⏸️", "running": "⏳", "done": "✅", "failed": "❌", "skipped": "⏭️", "skipped-image": "⏭️"}


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


def _size(size: Optional[tuple]) -> str:
    return f"{size[0]}×{size[1]}" if size else "?"


def flow_html(steps: List[Dict[str, Any]], state: Dict[str, Any], dims: Optional[Dict[str, Any]] = None) -> str:
    steps = cr.normalize_steps(steps)
    dims = dims if isinstance(dims, dict) else {}
    width, height, kind = int(dims.get("width") or 0), int(dims.get("height") or 0), dims.get("kind") or "video"
    sizes = cr.project_sizes(steps, state, width, height, kind) if width and height else [None] * len(steps)
    input_label = f'🎬 Input<div class="chain-node-sub">{width}×{height}</div>' if width and height else "🎬 Input"
    nodes = [f'<div class="chain-node chain-node-io">{input_label}</div>']
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
        size_line = ""
        if width and height and step.get("enabled", True):
            if kind == "image" and engine in cr.VIDEO_ONLY:
                size_line = '<div class="chain-node-size">skipped on images</div>'
            else:
                size_line = f'<div class="chain-node-size">≈ {_size(sizes[i - 1])}</div>'
        nodes.append(
            f'<div class="{cls}"><div class="chain-node-top"><span class="chain-node-num">{i}</span> {meta[1]} '
            f'{html.escape(meta[0])} {badge}</div><div class="chain-node-sub">{html.escape(_step_summary(engine, state))}</div>'
            f'{size_line}</div>'
        )
    final = next((s for s in reversed(sizes) if s), None) if sizes and sizes[-1] else None
    out_label = f'📦 Output<div class="chain-node-sub">≈ {_size(final)}</div>' if final else "📦 Output"
    nodes.append(f'<div class="chain-node chain-node-io">{out_label}</div>')
    arrow = '<div class="chain-arrow">➜</div>'
    summary = f"{len(steps)} step(s)"
    if total_scale != 1.0:
        summary += f" · total scale ×{total_scale:g}"
    if fps != 1:
        summary += f" · {fps}× frame rate"
    if final and max(final) > 7680:
        summary += " · ⚠️ the result is larger than 8K: lower a scale"
    if not steps:
        summary = "Empty chain: add steps with the buttons above."
    return f'<div class="chain-flow">{arrow.join(nodes)}</div><div class="chain-flow-summary">{html.escape(summary)}</div>'


def timeline_html(event: "cr.ChainEvent | None") -> str:
    if event is None or not event.steps:
        return ""
    rows = []
    for row in event.steps:
        meta = cr.ENGINE_META.get(row.engine, (row.engine, "🔧", "", ()))
        status_cls = "skipped" if row.status == "skipped-image" else row.status
        rows.append(
            f'<div class="chain-tl-row chain-tl-{status_cls}"><span class="chain-tl-icon">{ICONS.get(row.status, "•")}</span>'
            f'<span class="chain-tl-name">{row.index}. {meta[1]} {html.escape(meta[0])}</span>'
            f'<span class="chain-tl-detail">{html.escape(row.detail or row.status)}</span></div>'
        )
    head = f"Video {event.video_index} of {event.video_count}: {html.escape(Path(event.source).name)}" if event.source else ""
    return f'<div class="chain-timeline"><div class="chain-tl-head">{head}</div>{"".join(rows)}</div>'


def input_info_md(info: Dict[str, Any], count: int, videos: int) -> str:
    if not info.get("path"):
        return ""
    name = html.escape(Path(info["path"]).name)
    parts = [f"{'🎬' if info.get('kind') == 'video' else '🖼️'} **{name}**"]
    if info.get("width"):
        parts.append(f"{info['width']}×{info['height']}")
    if info.get("kind") == "video":
        if info.get("fps"):
            parts.append(f"{info['fps']:.2f}".rstrip("0").rstrip(".") + " fps")  # 29.97, 30, 23.98
        if info.get("frames"):
            parts.append(f"{info['frames']} frames")
        if info.get("duration"):
            parts.append(f"{info['duration']:.1f} s")
    text = " · ".join(parts)
    if count > 1:
        text += f"  \n📚 {count} inputs: {videos} video(s), {count - videos} image(s). The chain runs on each in turn."
    return text


def chain_tab(*, global_settings: Dict[str, Any], shared_state: gr.State, base_dir: Path, output_dir: Path):
    from shared.processing_queue import get_processing_queue_manager, resolve_queue_gpu_resources
    from shared.queue_state import snapshot_global_settings, snapshot_queue_state

    cancel = threading.Event()
    start_seed = shared_state.value.get("seed_controls", {}) if isinstance(shared_state.value, dict) else {}
    start_name = cr.resolve_chain_preset(base_dir, (start_seed.get("chains_settings") or {}).get("chain_preset"))
    if isinstance(start_seed, dict):
        start_seed["chains_settings"] = {"chain_preset": start_name}
    initial = cr.load_chain_preset(base_dir, start_name) or cr.load_chain_preset(base_dir, DEFAULT_CHAIN)
    start_epoch = start_seed.get("preset_epoch") if isinstance(start_seed, dict) else ""
    sync_sig = gr.State(f"{start_epoch}|{start_name}")
    steps_state = gr.State(initial)
    dims_state = gr.State({})
    runs_state = gr.State([])
    step_outputs = gr.State([])

    with gr.Row(elem_classes=["chain-tab"]):
        with gr.Column(scale=3):
            gr.HTML(HERO_HTML)
            with gr.Group(elem_classes=["chain-card"]):
                gr.Markdown("#### ① Videos and images", elem_classes=["chain-card-title"])
                with gr.Row():
                    files = gr.File(label="Videos or images (several allowed)", file_count="multiple", type="filepath",
                                    file_types=["video", "image"], scale=3)
                    with gr.Column(scale=2):
                        path_box = gr.Textbox(label="...or a file / folder path", placeholder="D:/videos  or  D:/videos/clip.mp4")
                        preview_pick = gr.Dropdown(label="Preview input", choices=[], value=None, visible=False,
                                                   allow_custom_value=False)
                input_info = gr.Markdown("", elem_classes=["chain-input-info"])
                with gr.Row(elem_classes=["chain-preview"]):
                    in_video = gr.Video(label="Input preview", interactive=False, height=300, visible=False)
                    in_image = gr.Image(label="Input preview", type="filepath", interactive=False, height=300, visible=False)
                with gr.Row(equal_height=True):
                    quick_test = gr.Checkbox(label="⚡ Quick test: run the chain on the first seconds of each video only",
                                             value=False, scale=3,
                                             info="Try a chain on a short piece first; images are not affected.")
                    quick_seconds = gr.Slider(label="Seconds", minimum=1, maximum=30, step=1, value=3, scale=2)

            with gr.Group(elem_classes=["chain-card"]):
                gr.Markdown("#### ② Build the chain", elem_classes=["chain-card-title"])
                gr.Markdown("Click an engine to append it. Each step uses the current settings of its tab; "
                            "the ≈ sizes follow the previewed input.", elem_classes=["resolution-info"])
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
                            dup = gr.Button("⧉", size="sm", scale=0, min_width=44, key=f"dup-{i}-{step['engine']}")
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

                        def _duplicate(cur, idx=i):
                            cur = cr.normalize_steps(cur)
                            if 0 <= idx < len(cur):
                                cur.insert(idx + 1, dict(cur[idx]))
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
                        dup.click(_duplicate, inputs=[steps_state], outputs=[steps_state], show_progress="hidden")
                        remove.click(_remove, inputs=[steps_state], outputs=[steps_state], show_progress="hidden")
                        enabled.input(_toggle, inputs=[enabled, steps_state], outputs=[steps_state], show_progress="hidden")
                        if factor is not None:
                            factor.input(_factor, inputs=[factor, steps_state], outputs=[steps_state], show_progress="hidden")

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear chain", size="sm", elem_classes=["sec-btn-slate"])

            with gr.Group(elem_classes=["chain-card"]):
                gr.Markdown("#### ③ Chain presets", elem_classes=["chain-card-title"])
                gr.Markdown(PRESET_NOTE, elem_classes=["resolution-info"])
                with gr.Row():
                    preset_dd = gr.Dropdown(label="Saved chains", choices=cr.list_chain_presets(base_dir), value=start_name,
                                            scale=3, allow_custom_value=False)
                    preset_name = gr.Textbox(label="Name for Save", placeholder="My 4K remaster", scale=2)
                with gr.Row():
                    load_btn = gr.Button("📂 Load chain", elem_classes=["sec-btn-blue"])
                    save_btn = gr.Button("💾 Save chain", elem_classes=["sec-btn-teal"])
                    delete_btn = gr.Button("🗑️ Delete chain", elem_classes=["sec-btn-orange"])
                preset_status = gr.Markdown("")

        with gr.Column(scale=2):
            gr.Markdown("### Run")
            status = gr.Markdown("Ready. Pick videos, build or load a chain, then press **Run Chain**.",
                                 elem_classes=["runtime-status-box"])
            with gr.Row():
                run_btn = gr.Button("▶️ Run Chain", variant="primary", elem_classes=["action-btn", "sec-btn-emerald"], scale=3)
                cancel_btn = gr.Button("⏹️ Cancel", variant="stop", elem_classes=["action-btn", "sec-btn-red"], scale=1)
            resume_btn = gr.Button("↻ Resume unfinished run", elem_classes=["sec-btn-amber"], visible=False)
            timeline = gr.HTML("")
            step_pick = gr.Dropdown(label="🔍 View a step's result (compared with the step before it)", choices=[],
                                    value=None, visible=False, allow_custom_value=False)
            out_video = gr.Video(label="Final output", interactive=False, visible=False)
            out_image = gr.Image(label="Final output", type="filepath", interactive=False, visible=False)
            slider = gr.ImageSlider(label="Before / After", interactive=False, visible=False, max_height=900,
                                    buttons=["download", "fullscreen"], elem_classes=["native-image-comparison-slider"])
            comparison = gr.HTML(value="", js_on_load=get_video_comparison_js_on_load(), visible=False)
            open_btn = gr.Button("📂 Open run folder", size="sm", elem_classes=["sec-btn-slate"], visible=False)
            with gr.Accordion("Log", open=False):
                log_box = gr.Textbox(label="Log", lines=16, max_lines=28, interactive=False, autoscroll=True)

    # --- input preview -------------------------------------------------------------------------
    def _preview(path: Optional[str], sources: List[str], steps, state):
        img_upd, vid_upd = preview_updates(path) if path else (gr.update(value=None, visible=False),
                                                               gr.update(value=None, visible=False))
        info = cr.media_info(path) if path else {}
        videos = sum(1 for s in sources if Path(s).suffix.lower() in cr.VIDEO_EXTS)
        dims = {"width": info.get("width", 0), "height": info.get("height", 0), "kind": info.get("kind", "video")} if info else {}
        return vid_upd, img_upd, input_info_md(info, len(sources), videos), dims, flow_html(steps, state, dims)

    def _inputs_changed(uploads, path_text, steps, state):
        sources = cr.collect_videos(uploads, path_text)
        first = sources[0] if sources else None
        names = [(f"{i + 1}. {Path(s).name}", s) for i, s in enumerate(sources)]
        pick = gr.update(choices=names, value=first, visible=len(sources) > 1)
        return (pick, *_preview(first, sources, steps, state))

    preview_outputs = [in_video, in_image, input_info, dims_state, flow]
    for event in (files.change, path_box.submit, path_box.blur):
        event(_inputs_changed, inputs=[files, path_box, steps_state, shared_state],
              outputs=[preview_pick] + preview_outputs, show_progress="hidden")
    preview_pick.input(lambda p, u, t, s, st: _preview(p, cr.collect_videos(u, t), s, st),
                       inputs=[preview_pick, files, path_box, steps_state, shared_state], outputs=preview_outputs,
                       show_progress="hidden")

    # --- editing -----------------------------------------------------------------------------
    for key, button in palette.items():
        button.click(lambda cur, k=key: cr.normalize_steps(cur) + [{"engine": k, "enabled": True}],
                     inputs=[steps_state], outputs=[steps_state], show_progress="hidden")
    clear_btn.click(lambda: [], outputs=[steps_state], show_progress="hidden")
    # A state written by a non-queued event does not fire .change in this app, so every editing event
    # above is queued; the flow view follows the state through .change.
    steps_state.change(flow_html, inputs=[steps_state, shared_state, dims_state], outputs=[flow], show_progress="hidden")

    def _remember(state, name):
        """The app preset keeps the chosen chain preset name (Save in Global Settings stores it)."""
        if isinstance(state, dict):
            seed = state.setdefault("seed_controls", {})
            seed["chains_settings"] = {"chain_preset": str(name or "")}
        return state

    def _load(name, state):
        steps = cr.load_chain_preset(base_dir, name)
        if not steps:
            return gr.update(), f"⚠️ Chain '{name}' not found.", state
        return steps, f"✅ Loaded chain **{name}** ({len(steps)} steps). Tab settings are unchanged.", _remember(state, name)

    def _save(name, selected, steps, state):
        target = str(name or "").strip() or (selected if selected and selected not in cr.BUILTIN_CHAINS else "")
        if not target:
            return gr.update(), "⚠️ Type a name for the chain first.", state
        if not cr.normalize_steps(steps):
            return gr.update(), "⚠️ The chain is empty.", state
        existed = target in cr.list_chain_presets(base_dir)
        saved = cr.save_chain_preset(base_dir, target, steps)
        verb = "Updated" if existed else "Saved"
        return (gr.update(choices=cr.list_chain_presets(base_dir), value=saved),
                f"💾 {verb} chain **{saved}** ({len(cr.normalize_steps(steps))} steps, chain only).", _remember(state, saved))

    def _delete(name, state):
        if name in cr.BUILTIN_CHAINS:
            return gr.update(), "ℹ️ Built-in example chains cannot be deleted.", state
        ok = cr.delete_chain_preset(base_dir, name)
        choices = cr.list_chain_presets(base_dir)
        keep = choices[0] if choices else None
        return gr.update(choices=choices, value=keep), (
            f"🗑️ Deleted chain **{name}**." if ok else f"⚠️ Chain '{name}' not found."), _remember(state, keep)

    load_btn.click(_load, inputs=[preset_dd, shared_state], outputs=[steps_state, preset_status, shared_state],
                   show_progress="hidden")
    preset_dd.input(_load, inputs=[preset_dd, shared_state], outputs=[steps_state, preset_status, shared_state],
                    show_progress="hidden")
    save_btn.click(_save, inputs=[preset_name, preset_dd, steps_state, shared_state],
                   outputs=[preset_dd, preset_status, shared_state], queue=False)
    delete_btn.click(_delete, inputs=[preset_dd, shared_state], outputs=[preset_dd, preset_status, shared_state], queue=False)

    def on_tab_select(state, sig, steps, dims):
        """Opening the tab after an app preset was loaded or reset (from any tab) loads that preset's
        chain; otherwise the chain being edited stays. The step cards always follow the tabs' settings."""
        seed = state.get("seed_controls", {}) if isinstance(state, dict) else {}
        seed = seed if isinstance(seed, dict) else {}
        wanted = (seed.get("chains_settings") or {}).get("chain_preset") or ""
        new_sig = f"{seed.get('preset_epoch')}|{wanted}"
        if new_sig == str(sig or ""):
            return gr.update(), gr.update(), flow_html(steps, state, dims), sig, gr.update()
        name = cr.resolve_chain_preset(base_dir, wanted)
        loaded = cr.load_chain_preset(base_dir, name)
        if not loaded:
            return gr.update(), gr.update(), flow_html(steps, state, dims), new_sig, gr.update()
        _remember(state, name)
        return (loaded, gr.update(choices=cr.list_chain_presets(base_dir), value=name), flow_html(loaded, state, dims),
                f"{seed.get('preset_epoch')}|{name}", f"✅ Loaded chain **{name}** with the app preset.")

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

    def _step_choices(event: "cr.ChainEvent", final: Optional[str], source: str) -> List[tuple]:
        """[(label, [previous, current])] for the step viewer: the final result first, then every step."""
        items: List[tuple] = []
        if final:
            items.append((f"🏁 Final result (input → chain)", [source, final]))
        previous = source
        for row in event.steps:
            if row.output and Path(row.output).is_file():
                meta = cr.ENGINE_META.get(row.engine, (row.engine, "🔧", "", ()))
                items.append((f"Step {row.index} · {meta[1]} {meta[0]} ({cr._dims(row.output)})", [previous, row.output]))
                previous = row.output
        return items

    NO = gr.update()

    def _run(sources, steps, preset_label, state, progress, test_seconds=0.0, resume_dirs=None):
        live_state = state if isinstance(state, dict) else {}

        def idle(text):
            return (text, NO, NO, NO, NO, NO, NO, live_state, NO, NO, NO, NO, NO)

        if not sources:
            yield idle("⚠️ Pick at least one video or image (upload or path).")
            return
        missing = [s["engine"] for s in cr.normalize_steps(steps) if s["engine"] not in cr.ENGINES]
        if missing:
            yield idle(f"❌ Unknown chain step(s): {', '.join(missing)}")
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
                    yield idle("⏹️ Chain cancelled while waiting in the queue.")
                    return
                yield idle(f"⏳ Waiting in the processing queue ({ticket.job_id})...")
            seed = queued_state.get("seed_controls", {})
            glob = seed.get("global_settings", {}) if isinstance(seed, dict) else {}
            out_root = Path(str((glob or {}).get("output_dir") or global_settings.get("output_dir") or output_dir))
            last_final, last_source = None, sources[-1]
            for event in cr.run_chain(sources, steps, queued_state, base_dir=base_dir, output_root=out_root,
                                      chain_name=str(preset_label or ""), progress=progress,
                                      cancel_flag=lambda: cancel.is_set() or ticket.cancel_event.is_set(),
                                      global_settings_snapshot=queued_globals, test_seconds=test_seconds,
                                      resume_dirs=resume_dirs):
                if event.final_output:
                    last_final, last_source = event.final_output, event.source
                if not event.finished:
                    yield (event.status, "\n".join(event.log[-500:]), timeline_html(event), NO, NO, NO, NO, live_state,
                           NO, NO, NO, NO, NO)
                    continue
                source = last_source if last_final else event.source
                vid, img, sl, comp = _final_updates(last_final, source)
                if last_final:
                    live_state.setdefault("seed_controls", {})["last_output_path"] = last_final
                    live_state["seed_controls"]["last_output_dir"] = str(Path(last_final).parent)
                choices = _step_choices(event, last_final, source)
                unfinished = [r for r in event.runs if not r.get("ok")]
                resume_upd = gr.update(visible=bool(unfinished),
                                       value=f"↻ Resume unfinished run ({len(unfinished)})" if unfinished else "↻ Resume")
                pick_upd = gr.update(choices=[c[0] for c in choices], value=choices[0][0] if choices else None,
                                     visible=bool(choices))
                if not last_final and choices:  # stopped: show the last finished step
                    vid, img, sl, comp = _final_updates(choices[-1][1][1], choices[-1][1][0])
                yield (event.status, "\n".join(event.log[-500:]), timeline_html(event), vid, img, sl, comp, live_state,
                       event.runs, resume_upd, pick_upd, choices, gr.update(visible=bool(event.run_dir)))
        finally:
            manager.complete(ticket.job_id)

    def run_chain_ui(uploads, path_text, steps, preset_label, test_on, test_secs, state, progress=gr.Progress()):
        sources = cr.collect_videos(uploads, path_text)
        yield from _run(sources, steps, preset_label, state, progress, float(test_secs) if test_on else 0.0)

    def resume_chain_ui(runs, steps, preset_label, state, progress=gr.Progress()):
        unfinished = [r for r in (runs or []) if not r.get("ok") and Path(str(r.get("run_dir") or "")).is_dir()]
        sources = [r["source"] for r in unfinished]
        yield from _run(sources, steps, preset_label, state, progress, 0.0, {r["source"]: r["run_dir"] for r in unfinished})

    run_outputs = [status, log_box, timeline, out_video, out_image, slider, comparison, shared_state, runs_state, resume_btn,
                   step_pick, step_outputs, open_btn]
    run_btn.click(run_chain_ui, inputs=[files, path_box, steps_state, preset_dd, quick_test, quick_seconds, shared_state],
                  outputs=run_outputs, concurrency_limit=32, concurrency_id="app_processing_queue", trigger_mode="multiple")
    resume_btn.click(resume_chain_ui, inputs=[runs_state, steps_state, preset_dd, shared_state], outputs=run_outputs,
                     concurrency_limit=32, concurrency_id="app_processing_queue", trigger_mode="multiple")

    def _view_step(label, choices):
        for name, pair in choices or []:
            if name == label:
                return _final_updates(pair[1], pair[0])
        return NO, NO, NO, NO

    step_pick.input(_view_step, inputs=[step_pick, step_outputs], outputs=[out_video, out_image, slider, comparison],
                    show_progress="hidden")

    def _open_folder(runs):
        from shared.services.global_service import open_outputs_folder

        folder = (runs or [{}])[-1].get("run_dir") if runs else ""
        return open_outputs_folder(folder) if folder else gr.update(value="⚠️ No run folder yet.")

    open_btn.click(_open_folder, inputs=[runs_state], outputs=[status], queue=False)

    def _cancel():
        cancel.set()
        stopped = cr.cancel_running()
        if stopped:
            return f"⏹️ Cancelling: {stopped} is stopping now; the chain stops after it (finished steps are kept for Resume)."
        return "⏹️ Cancelling: the chain stops after the current step (finished steps are kept for Resume)."

    cancel_btn.click(_cancel, outputs=[status], queue=False)
    return {"steps_state": steps_state, "flow": flow, "dims_state": dims_state, "on_select": on_tab_select,
            "select_inputs": [shared_state, sync_sig, steps_state, dims_state],
            "select_outputs": [steps_state, preset_dd, flow, sync_sig, preset_status]}
