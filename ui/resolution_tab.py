"""
Resolution & Scene Split Tab - Chunking-focused implementation
UPDATED: Uses Universal Preset System
"""

import gradio as gr
from pathlib import Path

from shared.services.resolution_service import (
    build_resolution_callbacks, RESOLUTION_ORDER
)
from shared.models import (
    get_seedvr2_model_names,
    scan_gan_models,
    get_flashvsr_model_names,
    get_rife_model_names
)
from ui.universal_preset_section import (
    universal_preset_section,
    wire_universal_preset_events,
)


def resolution_tab(preset_manager, shared_state: gr.State, base_dir: Path):
    """
    Self-contained Resolution & Scene Split tab.

    This tab applies universal chunking/splitting behavior across all upscaler models.
    Model-specific sizing (scale/max resolution) is configured in each model tab.
    """

    seed_controls = shared_state.value.get("seed_controls", {})
    shared_models = seed_controls.get("available_models", [])
    if not isinstance(shared_models, list):
        shared_models = []
    combined_models = sorted(
        {str(model_name).strip() for model_name in (shared_models or []) if str(model_name).strip()}
    )
    if not combined_models:
        # Fallback for legacy state/migrations: discover model lists directly.
        seedvr2_models = get_seedvr2_model_names()
        gan_models = scan_gan_models(base_dir)
        flashvsr_models = get_flashvsr_model_names()
        rife_models = get_rife_model_names(base_dir)
        combined_models = sorted(list({
            *seedvr2_models,
            *gan_models,
            *flashvsr_models,
            *rife_models
        }))
    if not combined_models:
        combined_models = ["default"]

    # Build service callbacks
    service = build_resolution_callbacks(preset_manager, shared_state, combined_models)

    # Get defaults
    defaults = service["defaults"]
    
    # UNIVERSAL PRESET: Load from shared_state
    seed_controls = shared_state.value.get("seed_controls", {})
    resolution_settings = seed_controls.get("resolution_settings", {})
    models_list = seed_controls.get("available_models", combined_models)
    
    # Merge with defaults
    merged_defaults = defaults.copy()
    for key, value in resolution_settings.items():
        if value is not None:
            merged_defaults[key] = value
    
    values = [merged_defaults[k] for k in RESOLUTION_ORDER]
    gr.Markdown("### Resolution & Scene Split Settings")
    gr.Markdown(
        "*Manage universal PySceneDetect scene splitting and chunking for all upscalers. "
        "Upscale factor / max resolution are now configured directly in each upscaler tab.*"
    )

    # Legacy model/sizing fields are hidden but retained for RESOLUTION_ORDER compatibility.
    model_selector_value = values[0]
    if model_selector_value not in combined_models:
        model_selector_value = combined_models[0] if combined_models else "default"

    model_selector = gr.Textbox(
        value=model_selector_value,
        visible=False,
        interactive=False,
    )
    auto_resolution = gr.Checkbox(
        value=values[1],
        visible=False,
        interactive=False,
    )
    enable_max_target = gr.Checkbox(
        value=values[2],
        visible=False,
        interactive=False,
    )
    upscale_factor = gr.Number(
        value=values[4],
        precision=2,
        visible=False,
        interactive=False,
    )
    max_target_resolution = gr.Number(
        value=values[5],
        precision=0,
        visible=False,
        interactive=False,
    )
    ratio_downscale_then_upscale = gr.Checkbox(
        value=values[6],
        visible=False,
        interactive=False,
    )

    with gr.Row():
        # Left Column: Settings
        with gr.Column(scale=2):
            gr.Markdown("#### Scene Splitting & Chunking")

            with gr.Group():
                with gr.Row():
                    auto_detect_scenes = gr.Checkbox(
                        label="🎬 Auto Detect Scenes (on input)",
                        value=values[3],
                        info="When Auto Chunk is ON and input is a video, auto-scan scene cuts to show the scene count. Can be slow for long videos."
                    )
            
            with gr.Group():
                with gr.Row():
                    auto_chunk = gr.Checkbox(
                        label="Auto Chunk (PySceneDetect Scenes)",
                        value=values[7],
                        info="Recommended. Splits by detected scene cuts (content-based). Uses scene sensitivity + minimum scene length."
                    )

                    frame_accurate_split = gr.Checkbox(
                        label="Frame-Accurate Split (Lossless)",
                        value=values[8],
                        info="Enabled: frame-accurate splitting via lossless re-encode (slower). Disabled: fast stream-copy splitting (keyframe-limited)."
                    )

                    per_chunk_cleanup = gr.Checkbox(
                        label="Delete Chunk Files After Processing",
                        value=values[9],
                        info="Deletes chunk artifacts from the run output folder (input_chunks/processed_chunks) to save disk space. Thumbnails are kept for the chunk gallery."
                    )

                with gr.Row():
                    chunk_size = gr.Slider(
                        label="Chunk Size (seconds, 0=disabled)",
                        minimum=0, maximum=600, step=10,
                        value=values[10],
                        interactive=not bool(values[7]),
                        info="Static chunking only (when Auto Chunk is OFF). 0=off, 60=1min chunks, 300=5min chunks."
                    )

                    chunk_overlap = gr.Slider(
                        label="Chunk Overlap (seconds)",
                        minimum=0.0, maximum=5.0, step=0.1,
                        value=0.0 if bool(values[7]) else values[11],
                        interactive=not bool(values[7]),
                        info="Static chunking only. Auto Chunk forces overlap to 0 to avoid blending across scene cuts."
                    )

                with gr.Row():
                    scene_threshold = gr.Slider(
                        label="Scene Detection Sensitivity",
                        minimum=5.0, maximum=50.0, step=1.0,
                        value=values[12],
                        interactive=bool(values[7]),
                        info="PySceneDetect ContentDetector threshold. Lower = more cuts, higher = fewer. 27 is a balanced default."
                    )

                    min_scene_len = gr.Slider(
                        label="Minimum Scene Length (seconds)",
                        minimum=0.5, maximum=10.0, step=0.5,
                        value=values[13],
                        interactive=bool(values[7]),
                        info="Minimum duration for a detected scene. Prevents very short chunks. 1.0s recommended."
                    )

                def _toggle_chunk_mode(auto_enabled: bool):
                    auto_enabled = bool(auto_enabled)
                    if auto_enabled:
                        chunk_overlap_update = gr.update(value=0.0, interactive=False)
                    else:
                        chunk_overlap_update = gr.update(interactive=True)
                    return (
                        gr.update(interactive=not auto_enabled),  # chunk_size
                        chunk_overlap_update,  # chunk_overlap
                        gr.update(interactive=auto_enabled),  # scene_threshold
                        gr.update(interactive=auto_enabled),  # min_scene_len
                    )

                auto_chunk.change(
                    fn=_toggle_chunk_mode,
                    inputs=auto_chunk,
                    outputs=[chunk_size, chunk_overlap, scene_threshold, min_scene_len],
                    queue=False,
                    show_progress="hidden",
                    trigger_mode="always_last",
                )

        # Right Column: Chunk Estimation & Preview
        with gr.Column(scale=1):
            gr.Markdown("#### Estimate & Preview")
            
            # Input path for estimation
            calc_input_path = gr.Textbox(
                label="Input Path (for estimation)",
                placeholder="Paste input video/image path or use SeedVR2 tab input",
                info="Path to estimate scene/chunk splitting for"
            )
            
            with gr.Row():
                calc_chunks_btn = gr.Button("📊 Estimate Chunks", variant="primary")
            
            # Results display
            calc_result = gr.Markdown("", visible=False)
            
            # Disk space warning
            disk_space_warning = gr.Markdown("", visible=False)
            
            # Quick actions
            gr.Markdown("#### ⚡ Quick Actions")
            
            with gr.Row():
                use_seedvr2_input_btn = gr.Button("📥 Use SeedVR2 Input", size="lg")

    # UNIVERSAL PRESET MANAGEMENT
    (
        preset_dropdown,
        preset_name_input,
        save_preset_btn,
        load_preset_btn,
        preset_status,
        reset_defaults_btn,
        delete_preset_btn,
        preset_callbacks,
    ) = universal_preset_section(
        preset_manager=preset_manager,
        shared_state=shared_state,
        tab_name="resolution",
        inputs_list=[],
        base_dir=base_dir,
        models_list=models_list,
        open_accordion=True,
    )
    # Collect inputs - MUST match RESOLUTION_ORDER exactly
    inputs_list = [
        model_selector, auto_resolution, enable_max_target, auto_detect_scenes, upscale_factor,
        max_target_resolution, ratio_downscale_then_upscale,
        auto_chunk, frame_accurate_split, per_chunk_cleanup,
        chunk_size, chunk_overlap,
        scene_threshold, min_scene_len
    ]

    # UNIVERSAL PRESET EVENT WIRING
    preset_events = wire_universal_preset_events(
        preset_dropdown=preset_dropdown,
        preset_name_input=preset_name_input,
        save_btn=save_preset_btn,
        load_btn=load_preset_btn,
        preset_status=preset_status,
        reset_btn=reset_defaults_btn,
        delete_btn=delete_preset_btn,
        callbacks=preset_callbacks,
        inputs_list=inputs_list,
        shared_state=shared_state,
        tab_name="resolution",
    )

    def _auto_apply_resolution_state(*args):
        # Reuse service logic so global + per-model resolution caches stay in sync.
        _, updated_state = service["apply_to_seed"](*args)
        return updated_state

    auto_apply_inputs = inputs_list + [shared_state]
    auto_apply_kwargs = {
        "fn": _auto_apply_resolution_state,
        "inputs": auto_apply_inputs,
        "outputs": [shared_state],
        "queue": False,
        "show_progress": "hidden",
    }

    # Apply immediately whenever any Resolution control changes.
    auto_apply_triggers = []
    for comp in inputs_list:
        if hasattr(comp, "change"):
            auto_apply_triggers.append(comp.change)
        if hasattr(comp, "release"):
            auto_apply_triggers.append(comp.release)

    if hasattr(gr, "on"):
        gr.on(
            triggers=auto_apply_triggers,
            trigger_mode="always_last",
            **auto_apply_kwargs,
        )
    else:
        for comp in inputs_list:
            if hasattr(comp, "change"):
                comp.change(
                    trigger_mode="always_last",
                    **auto_apply_kwargs,
                )

    # Preset load/reset updates component values; chain an explicit apply so caches
    # update even if programmatic value updates do not emit component change events.
    for key in ("load_click", "load_change", "reset"):
        dep = (preset_events or {}).get(key)
        if dep is not None and hasattr(dep, "then"):
            dep.then(
                fn=_auto_apply_resolution_state,
                inputs=auto_apply_inputs,
                outputs=[shared_state],
                queue=False,
                show_progress="hidden",
            )

    # Estimation callbacks
    def calculate_chunks_wrapper(input_path, auto_chunk_enabled, chunk_size, chunk_overlap, state):
        """Wrapper for chunk estimation"""
        if not input_path or not input_path.strip():
            input_path = state.get("seed_controls", {}).get("last_input_path", "")
            if not input_path:
                return gr.update(value="⚠️ No input path provided", visible=True), state, gr.update(visible=False)
        
        info, updated_state = service["calculate_chunk_estimate"](
            input_path,
            bool(auto_chunk_enabled),
            float(chunk_size),
            float(chunk_overlap),
            state
        )
        
        # Check for disk space warnings in the info
        if "⚠️" in info and "disk space" in info.lower():
            disk_warning = "⚠️ **DISK SPACE WARNING**: Insufficient space detected!"
            return gr.update(value=info, visible=True), updated_state, gr.update(value=disk_warning, visible=True)
        
        return gr.update(value=info, visible=True), updated_state, gr.update(visible=False)

    calc_chunks_btn.click(
        fn=calculate_chunks_wrapper,
        inputs=[calc_input_path, auto_chunk, chunk_size, chunk_overlap, shared_state],
        outputs=[calc_result, shared_state, disk_space_warning]
    )

    # Use SeedVR2 input button
    def use_seedvr2_input(state):
        input_path = state.get("seed_controls", {}).get("last_input_path", "")
        if input_path:
            return gr.update(value=input_path), gr.update(value=f"✅ Using input from SeedVR2 tab: {input_path}", visible=True)
        else:
            return gr.update(), gr.update(value="⚠️ No input set in SeedVR2 tab yet", visible=True)

    use_seedvr2_input_btn.click(
        fn=use_seedvr2_input,
        inputs=shared_state,
        outputs=[calc_input_path, calc_result]
    )

    # Auto-update estimation hint when chunk settings change
    for component in [auto_detect_scenes, auto_chunk, chunk_size, chunk_overlap, scene_threshold, min_scene_len]:
        component.change(
            fn=lambda *args: gr.update(value="ℹ️ Settings changed. Click 'Estimate Chunks' to update.", visible=True),
            outputs=calc_result
        )

    return {
        "inputs_list": inputs_list,
        "preset_dropdown": preset_dropdown,
        "preset_status": preset_status,
    }

