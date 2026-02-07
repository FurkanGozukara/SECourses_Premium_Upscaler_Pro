"""
Resolution & Scene Split Tab - Complete Implementation with Auto-Calculation
UPDATED: Now uses Universal Preset System
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any

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
from shared.universal_preset import dict_to_values


def resolution_tab(preset_manager, shared_state: gr.State, base_dir: Path):
    """
    Self-contained Resolution & Scene Split tab with auto-calculation features.
    
    This tab's settings apply to ALL upscaler models (SeedVR2, GAN, FlashVSR+, RIFE).
    Settings are saved per-model for flexibility.
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
    # Header
    gr.Markdown("### 📐 Resolution & Scene Split Settings")
    gr.Markdown("""
    *Configure resolution, aspect ratio handling, and **universal PySceneDetect chunking** for all upscaler models*
    
    **🎬 PySceneDetect Chunking**: Intelligent scene-based video splitting that works with **ALL models** (SeedVR2, GAN, RIFE, FlashVSR+).
    This is the **PREFERRED chunking method** for long videos, managing VRAM, and optimizing quality per scene.
    """)

    # Model selector (guard against stale preset values not present in choices)
    model_selector_value = values[0]
    if model_selector_value not in combined_models:
        model_selector_value = combined_models[0] if combined_models else "default"

    model_selector = gr.Dropdown(
        label="Model Context (for presets)",
        choices=combined_models,
        value=model_selector_value,
        info="Settings are saved/loaded per model"
    )

    with gr.Row():
        # Left Column: Settings
        with gr.Column(scale=2):
            gr.Markdown("#### 🎯 Resolution Settings")
             
            with gr.Group():
                with gr.Row():
                    auto_resolution = gr.Checkbox(
                        label="🔄 Auto-Resolution (Aspect Ratio Aware)",
                        value=values[1],
                        info="Automatically calculate optimal resolution maintaining aspect ratio"
                    )

                    enable_max_target = gr.Checkbox(
                        label="🎯 Enable Max Target Resolution",
                        value=values[2],
                        info="Apply maximum resolution cap to prevent excessive upscaling"
                    )

                    auto_detect_scenes = gr.Checkbox(
                        label="🎬 Auto Detect Scenes (on input)",
                        value=values[3],
                        info="When Auto Chunk is ON and input is a video, auto-scan scene cuts to show the scene count. Can be slow for long videos."
                    )

                upscale_factor = gr.Number(
                    label="Upscale x (any factor)",
                    value=values[4],
                    precision=2,
                    info="Target scale factor relative to input (e.g., 4.0 = 4x). Works for both images and videos."
                )

                with gr.Row():
                    max_target_resolution = gr.Slider(
                        label="Max Resolution (max edge, 0 = no cap)",
                        minimum=0, maximum=8192, step=16,
                        value=values[5],
                        info="Caps the LONG side (max(width,height)) of the target. 0 = unlimited."
                    )

                    ratio_downscale_then_upscale = gr.Checkbox(
                        label="⬇️➡️⬆️ Pre-downscale then upscale (when capped)",
                        value=values[6],
                        info="If max edge would reduce effective scale, downscale input first so the model still runs at the full Upscale x."
                    )

            gr.Markdown("#### 🎬 Scene Splitting & Chunking")
            
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

        # Right Column: Auto-Calculation & Preview
        with gr.Column(scale=1):
            gr.Markdown("#### 🔍 Auto-Calculate & Preview")
            
            # Input path for calculation
            calc_input_path = gr.Textbox(
                label="Input Path (for estimation)",
                placeholder="Paste input video/image path or use SeedVR2 tab input",
                info="Path to calculate resolution for"
            )
            
            calc_model_scale = gr.Number(
                label="Model Scale (2, 4, etc. - leave 0 for auto)",
                value=0,
                precision=0,
                info="For GAN models with fixed scale. 0=auto-detect, 2=2x model, 4=4x model"
            )
            
            with gr.Row():
                calc_resolution_btn = gr.Button("🎯 Calculate Resolution", variant="primary")
                calc_chunks_btn = gr.Button("📊 Estimate Chunks", variant="secondary")
            
            # Results display
            calc_result = gr.Markdown("", visible=False)
            
            # Disk space warning
            disk_space_warning = gr.Markdown("", visible=False)
            
            # Quick actions
            gr.Markdown("#### ⚡ Quick Actions")
            
            with gr.Row():
                use_seedvr2_input_btn = gr.Button("📥 Use SeedVR2 Input", size="lg")
                refresh_calc_btn = gr.Button("🔄 Refresh", size="lg")

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

    # Apply to pipeline
    gr.Markdown("#### 🔗 Apply to Pipeline")
    apply_to_seed_btn = gr.Button(
        "✅ Apply Resolution Settings to All Upscalers",
        variant="primary",
        size="lg"
    )
    apply_status = gr.Markdown("")

    # Collect inputs - MUST match RESOLUTION_ORDER exactly
    inputs_list = [
        model_selector, auto_resolution, enable_max_target, auto_detect_scenes, upscale_factor,
        max_target_resolution, ratio_downscale_then_upscale,
        auto_chunk, frame_accurate_split, per_chunk_cleanup,
        chunk_size, chunk_overlap,
        scene_threshold, min_scene_len
    ]

    # UNIVERSAL PRESET EVENT WIRING
    wire_universal_preset_events(
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

    apply_to_seed_btn.click(
        fn=lambda *args: service["apply_to_seed"](*args),
        inputs=inputs_list + [shared_state],
        outputs=[apply_status, shared_state]
    )

    # Auto-calculation callbacks
    def calculate_resolution_wrapper(input_path, scale_x, max_res, enable_max, auto_mode, 
                                    pre_downscale, model_scale, state):
        """Wrapper for sizing calculation (new Upscale-x rules)"""
        if not input_path or not input_path.strip():
            # Try to get from shared state
            input_path = state.get("seed_controls", {}).get("last_input_path", "")
            if not input_path:
                return "⚠️ No input path provided. Upload in SeedVR2 tab or enter path above.", state, gr.update(visible=False)
        
        # Call service function
        info, updated_state = service["calculate_auto_resolution"](
            input_path, 
            float(scale_x), 
            int(max_res),
            enable_max,
            auto_mode,
            pre_downscale,
            int(model_scale) if model_scale > 0 else None,
            state
        )
        
        return gr.update(value=info, visible=True), updated_state, gr.update(visible=False)

    def calculate_chunks_wrapper(input_path, auto_chunk_enabled, chunk_size, chunk_overlap, state):
        """Wrapper for chunk estimation"""
        if not input_path or not input_path.strip():
            input_path = state.get("seed_controls", {}).get("last_input_path", "")
            if not input_path:
                return "⚠️ No input path provided", state
        
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

    calc_resolution_btn.click(
        fn=calculate_resolution_wrapper,
        inputs=[calc_input_path, upscale_factor, max_target_resolution, 
                enable_max_target, auto_resolution, ratio_downscale_then_upscale, 
                calc_model_scale, shared_state],
        outputs=[calc_result, shared_state, disk_space_warning]
    )

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

    # Auto-update calculation when settings change
    for component in [upscale_factor, max_target_resolution, auto_resolution, enable_max_target, auto_detect_scenes, ratio_downscale_then_upscale]:
        component.change(
            fn=lambda *args: gr.update(value="ℹ️ Settings changed. Click 'Calculate Resolution' to update.", visible=True),
            outputs=calc_result
        )

    return {
        "inputs_list": inputs_list,
        "preset_dropdown": preset_dropdown,
        "preset_status": preset_status,
    }

