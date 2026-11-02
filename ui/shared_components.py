import gradio as gr


CANCEL_CONFIRMATION_MESSAGE = (
    "Cancellation was not sent. Enable the 'Confirm cancel' checkbox first, "
    "then click Cancel again."
)


def warn_cancel_confirmation() -> str:
    """Show an explicit warning when Cancel is clicked without confirmation."""
    gr.Warning(CANCEL_CONFIRMATION_MESSAGE, title="Cancel confirmation required")
    return CANCEL_CONFIRMATION_MESSAGE


def autotune_modal_dismiss_js(elem_id: str) -> str:
    """Hide a terminal Auto Tune notice immediately, even while a queued event settles."""
    return f"""
() => {{
  const modal = document.getElementById({elem_id!r});
  if (modal) {{
    modal.dataset.dismissed = "true";
    modal.style.setProperty("display", "none", "important");
  }}
  return [];
}}
"""


def autotune_modal_reset_js(elem_id: str) -> str:
    """Allow a newly started Auto Tune run to show its own terminal notice."""
    return f"""
(...args) => {{
  const modal = document.getElementById({elem_id!r});
  if (modal) {{
    delete modal.dataset.dismissed;
    modal.style.removeProperty("display");
  }}
  return args;
}}
"""


AUTO_TUNE_BUTTON_LABEL = "Auto Tune for Max Quality - VRAM Optimized"
AUTO_TUNE_ON_RUN_LABEL = "Auto Tune on every Upscale"
AUTO_TUNE_ON_RUN_INFO = "Every Upscale first applies Auto Tune for its input and the VRAM free at that moment."


def autotune_vram_controls(save_vram_value, auto_tune_on_run_value=True, *, button_label: str = AUTO_TUNE_BUTTON_LABEL):
    """Auto Tune button, Save VRAM (GB) slider and the auto-apply checkbox, laid out the same on every tab.

    Returns (auto_tune_btn, save_vram_gb, auto_tune_on_run)."""
    from shared.vram_autotune.common import (
        DEFAULT_SAVE_VRAM_GB,
        SAVE_VRAM_INFO,
        SAVE_VRAM_LABEL,
        SAVE_VRAM_MAX_GB,
        SAVE_VRAM_MIN_GB,
        SAVE_VRAM_STEP_GB,
        clamp_save_vram_gb,
    )

    auto_tune_btn = gr.Button(
        button_label,
        elem_classes=["action-btn", "action-btn-optimize", "sec-btn-crimson"],
    )
    with gr.Group(elem_classes=["autotune-vram-group"]):
        save_vram_gb = gr.Slider(
            label=SAVE_VRAM_LABEL,
            minimum=SAVE_VRAM_MIN_GB,
            maximum=SAVE_VRAM_MAX_GB,
            step=SAVE_VRAM_STEP_GB,
            value=clamp_save_vram_gb(save_vram_value, DEFAULT_SAVE_VRAM_GB),
            info=SAVE_VRAM_INFO,
        )
        auto_tune_on_run = gr.Checkbox(
            label=AUTO_TUNE_ON_RUN_LABEL,
            value=bool(True if auto_tune_on_run_value is None else auto_tune_on_run_value),
            info=AUTO_TUNE_ON_RUN_INFO,
        )
    return auto_tune_btn, save_vram_gb, auto_tune_on_run


def _report_markdown(report: str) -> str:
    lines = [ln.strip() for ln in str(report or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    return "**" + lines[0] + "**\n\n" + "\n".join(f"- {ln}" for ln in lines[1:])


def wire_auto_tune(
    *,
    auto_tune_btn,
    action,
    inputs: list,
    tuned: dict,
    status_box,
    shared_state,
    notice_text,
    notice_modal,
    modal_id: str,
    global_settings=None,
):
    """Instant Auto Tune click: `action(*inputs_without_state, state=...)` returns
    {"status", "report", "updates": {setting key: value}, "state"}; every key in `tuned`
    (setting key -> component) gets its new value and the report opens in the notice modal."""
    keys = list(tuned)

    def _on_click(*args):
        live_state = args[-1] if (args and isinstance(args[-1], dict)) else {}
        try:
            result = action(*args[:-1], state=live_state, global_settings_snapshot=global_settings)
        except Exception as exc:  # never leave the button dead
            result = {"status": f"Auto Tune failed: {exc}", "report": f"Auto Tune failed: {exc}", "updates": {}, "state": live_state}
        updates = result.get("updates") or {}
        report = result.get("report") or result.get("status") or ""
        return (
            gr.update(value=str(result.get("status") or ""), visible=True),
            *[gr.update(value=updates[k]) if k in updates else gr.update() for k in keys],
            result.get("state", live_state),
            gr.update(value=_report_markdown(report)),
            gr.update(visible=bool(report)),
        )

    return auto_tune_btn.click(
        fn=_on_click,
        inputs=inputs + [shared_state],
        outputs=[status_box, *[tuned[k] for k in keys], shared_state, notice_text, notice_modal],
        queue=False,
        show_progress="hidden",
        js=autotune_modal_reset_js(modal_id),
    )


def wire_auto_tune_on_run(*, upscale_btn, action, inputs: list, tuned: dict, shared_state, auto_tune_on_run, global_settings=None):
    """When "Auto Tune on every Upscale" is on, show the tuned values in the UI as the Upscale
    click starts (the job itself re-tunes at its real start, so queued jobs size for the VRAM
    free when they run). Skipped while another job holds the GPU, whose VRAM would distort it.

    Wire this before the Upscale job. Gradio runs a button's listeners one after another and
    reads each listener's inputs only when its turn comes, so a server call registered before
    the job would delay the moment the job reads its settings: a setting changed during that
    call (seconds while other jobs keep the server busy) would leak into the queued job. The
    tuned values are therefore shown from a browser-only listener's .then(), which runs in its
    own chain while the job reads its settings at the click."""
    keys = list(tuned)

    def _out(values):
        # One output takes the bare update: Gradio would store a 1-tuple as the component's value.
        values = tuple(values)
        return values[0] if len(values) == 1 else values

    def _on_click(enabled, *args):
        no_change = _out(gr.update() for _ in keys)
        if not enabled:
            return no_change
        try:
            from shared.processing_queue import get_processing_queue_manager

            snap = get_processing_queue_manager().snapshot()
            if any(float(job.get("wait_seconds") or 0.0) > 3.0 for job in snap.get("active_jobs") or []):
                print("[Auto Tune] Upscale clicked while another job runs: the job tunes itself when it starts.",
                      flush=True)
                return no_change
            live_state = args[-1] if (args and isinstance(args[-1], dict)) else {}
            # Work on a copy: the running job owns the real state.
            import copy

            result = action(*args[:-1], state=copy.deepcopy(live_state), global_settings_snapshot=global_settings)
        except Exception as exc:
            print(f"[Auto Tune] Could not show the tuned values in the UI ({type(exc).__name__}: {exc}); "
                  "the job still tunes itself when it starts.", flush=True)
            return no_change
        updates = result.get("updates") or {}
        return _out(gr.update(value=updates[k]) if k in updates else gr.update() for k in keys)

    start = upscale_btn.click(fn=None, js="() => []", queue=False, show_progress="hidden")
    return start.then(
        fn=_on_click,
        inputs=[auto_tune_on_run] + inputs + [shared_state],
        outputs=[tuned[k] for k in keys],
        queue=False,
        show_progress="hidden",
    )


def with_pre_run_output_signature(run_fn, capture_signature):
    """Wrap an Upscale job so it also outputs the latest-output signature taken as the job starts.

    Auto Transfer Output to Input compares it with the newest output after the run. The job
    captures it itself instead of a listener before it, which would delay the moment the job
    reads its settings (see wire_auto_tune_on_run). Register the job with the signature state
    as its last output."""

    def _run(*args, progress=gr.Progress()):
        live_state = args[-1] if (args and isinstance(args[-1], dict)) else {}
        try:
            signature = capture_signature(live_state)
        except Exception:
            signature = ""
        updates = run_fn(*args, progress=progress)
        try:
            for payload in updates:
                yield (*payload, signature) if isinstance(payload, (tuple, list)) else (payload, signature)
        finally:
            updates.close()

    return _run


def mode_banner(text: str):
    """Reusable mode / health banner."""
    return gr.Markdown(text)


def comparison_help(
    message: str = (
        "Use the comparison slider to swipe between input and output. Pin keeps the reference fixed; "
        "Fullscreen opens a lightbox-style view. For video we use an HTML/fallback slider (native video slider not available)."
    )
):
    """Reusable comparison info block with pin/fullscreen guidance."""
    return gr.Markdown(message)


def preset_section(
    tab_name: str,
    preset_manager,
    model_name: str,
    choices: list,
    last_used: str,
    safe_defaults_label: str = "Safe Defaults",
):
    """
    Shared preset UI block used across tabs to keep layout consistent.
    Returns tuple of components for downstream wiring.
    Note: Refresh button removed - presets auto-refresh after save/load.
    """
    preset_dropdown = gr.Dropdown(
        label=f"{tab_name} Presets",
        choices=choices,
        value=last_used,
        allow_custom_value=True,  # Allow saving new presets that aren't in the list yet
    )
    preset_name = gr.Textbox(label="Preset Name", placeholder="my_preset")
    with gr.Row():
        save_preset_btn = gr.Button("Save Preset")
        load_preset_btn = gr.Button("Load Preset")
    preset_status = gr.Markdown("")
    safe_defaults_btn = gr.Button(f"🔄 {safe_defaults_label}", variant="secondary", size="lg")
    return (
        preset_dropdown,
        preset_name,
        save_preset_btn,
        load_preset_btn,
        preset_status,
        safe_defaults_btn,
    )

