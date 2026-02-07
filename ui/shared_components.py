import gradio as gr


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

