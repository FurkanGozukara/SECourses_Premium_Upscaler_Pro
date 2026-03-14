"""
Universal Preset Section - Shared UI component for ALL tabs

This component provides a unified preset management interface that:
- Saves/loads ALL settings from ALL tabs in a single preset
- Is identical across all tabs for consistency
- Syncs via shared_state when any tab saves/loads
"""

import copy
import gradio as gr
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path

from shared.preset_manager import PresetManager
from shared.models.rife_meta import get_rife_default_model
from shared.gpu_utils import resolve_global_gpu_device
from shared.services.output_service import OUTPUT_ORDER
from shared.video_codec_options import (
    get_pixel_format_choices,
    DEFAULT_AV1_FILM_GRAIN,
    DEFAULT_AV1_FILM_GRAIN_DENOISE,
)
from shared.universal_preset import (
    TAB_CONFIGS,
    values_to_dict,
    dict_to_values,
    get_all_defaults,
    update_shared_state_from_preset,
    collect_preset_from_shared_state,
    merge_preset_with_defaults,
)


def create_universal_preset_callbacks(
    preset_manager: PresetManager,
    tab_name: str,
    base_dir: Path = None,
    models_list: List[str] = None,
) -> Dict[str, Callable]:
    """
    Create callbacks for universal preset operations.
    
    Args:
        preset_manager: PresetManager instance
        tab_name: Current tab name (for updating that tab's values)
        base_dir: App base directory
        models_list: List of available models
    
    Returns:
        Dict of callback functions
    """
    
    def get_presets_list() -> List[str]:
        """Get list of all universal presets."""
        return preset_manager.list_universal_presets()

    def _coerce_tab_values_for_ui(tab_values: List[Any]) -> List[Any]:
        """Ensure tab-specific UI invariants (choices/value compatibility)."""
        if tab_name != "output":
            return tab_values
        coerced = list(tab_values)
        try:
            codec_idx = OUTPUT_ORDER.index("video_codec")
            pix_idx = OUTPUT_ORDER.index("pixel_format")
            codec_val = str(coerced[codec_idx] or "h264").strip().lower()
            pix_choices = get_pixel_format_choices(codec_val)
            pix_fallback = pix_choices[0] if pix_choices else "yuv420p"
            pix_val = str(coerced[pix_idx] or pix_fallback).strip().lower()
            if pix_val not in pix_choices:
                pix_val = pix_fallback
            coerced[pix_idx] = gr.update(choices=pix_choices, value=pix_val)
        except Exception:
            return tab_values
        return coerced
    
    def save_preset(
        preset_name: str,
        selected_preset_name: str | None,
        current_tab_values: List[Any],
        state: Dict[str, Any],
    ) -> Tuple:
        """
        Save a universal preset from current shared_state.

        The current tab's values are updated in state first,
        then the full state is saved as a universal preset.

        Returns:
            (updated_dropdown, status_message, updated_state)
        """
        typed_name = str(preset_name or "").strip()
        selected_name = str(selected_preset_name or "").strip()
        state_current_name = ""
        if isinstance(state, dict):
            seed_controls = state.get("seed_controls", {})
            if isinstance(seed_controls, dict):
                state_current_name = str(seed_controls.get("current_preset_name") or "").strip()

        target_name = typed_name or selected_name or state_current_name
        if not target_name:
            return (
                gr.update(),
                "⚠️ Enter a preset name or select a preset to overwrite",
                state,
            )

        fallback_overwrite = not typed_name

        try:
            # Update current tab's settings in state
            seed_controls = state.get("seed_controls", {})
            tab_settings_key = f"{tab_name}_settings"

            # Convert current tab values to dict
            tab_dict = values_to_dict(tab_name, current_tab_values)
            seed_controls[tab_settings_key] = tab_dict
            state["seed_controls"] = seed_controls

            # Collect full preset from state
            preset_data = collect_preset_from_shared_state(state)

            # Save universal preset
            saved_name = preset_manager.save_universal_preset(target_name, preset_data)

            # Update state with new preset name
            state["seed_controls"]["current_preset_name"] = saved_name
            state["seed_controls"]["preset_dirty"] = False

            # Refresh dropdown
            presets = get_presets_list()
            action = "Overwrote" if fallback_overwrite else "Saved"

            return (
                gr.update(choices=presets, value=saved_name),
                f"✅ {action} universal preset '{saved_name}' (all tabs)",
                state,
            )

        except Exception as e:
            return (
                gr.update(),
                f"❌ Error saving preset: {str(e)}",
                state,
            )


    def load_preset(
        preset_name: str,
        state: Dict[str, Any],
    ) -> Tuple:
        """
        Load a universal preset and update ALL tabs via shared_state.
        
        Returns:
            (tab_values..., status_message, updated_state)
            
            tab_values are the values for the CURRENT tab only.
            Other tabs read from shared_state on next access.
        """
        if not preset_name or not preset_name.strip():
            defaults = get_all_defaults(base_dir, models_list)
            seed_controls = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
            current_tab_settings = seed_controls.get(f"{tab_name}_settings", {}) if isinstance(seed_controls, dict) else {}
            if not isinstance(current_tab_settings, dict):
                current_tab_settings = {}
            source = current_tab_settings if current_tab_settings else defaults.get(tab_name, {})
            tab_values = dict_to_values(tab_name, source, defaults.get(tab_name, {}))
            return (*_coerce_tab_values_for_ui(tab_values), "ℹ️ No preset selected", state)
        
        try:
            # Load the universal preset
            preset_data = preset_manager.load_universal_preset(preset_name)
            
            if not preset_data:
                defaults = get_all_defaults(base_dir, models_list)
                seed_controls = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
                current_tab_settings = seed_controls.get(f"{tab_name}_settings", {}) if isinstance(seed_controls, dict) else {}
                if not isinstance(current_tab_settings, dict):
                    current_tab_settings = {}
                source = current_tab_settings if current_tab_settings else defaults.get(tab_name, {})
                tab_values = dict_to_values(tab_name, source, defaults.get(tab_name, {}))
                return (*_coerce_tab_values_for_ui(tab_values), f"⚠️ Preset '{preset_name}' not found", state)
            
            # Merge with defaults to fill any missing keys
            merged_preset = merge_preset_with_defaults(preset_data, base_dir, models_list)
            
            # Update shared_state with all tab settings
            state = update_shared_state_from_preset(state, merged_preset, preset_name)
            
            # Update last used
            preset_manager.set_last_used_universal_preset(preset_name)
            
            # Get values for current tab
            tab_data = merged_preset.get(tab_name, {})
            defaults = get_all_defaults(base_dir, models_list)
            tab_values = dict_to_values(tab_name, tab_data, defaults.get(tab_name, {}))
            
            return (*_coerce_tab_values_for_ui(tab_values), f"✅ Loaded universal preset '{preset_name}' (all tabs)", state)
        
        except Exception as e:
            defaults = get_all_defaults(base_dir, models_list)
            tab_values = dict_to_values(tab_name, defaults.get(tab_name, {}))
            return (*_coerce_tab_values_for_ui(tab_values), f"❌ Error loading preset: {str(e)}", state)
    
    def reset_to_defaults(state: Dict[str, Any]) -> Tuple:
        """
        Reset ALL tabs to default values.
        
        Returns:
            (tab_values..., status_message, updated_state)
        """
        try:
            defaults = get_all_defaults(base_dir, models_list)
            
            # Clear preset settings in state
            seed_controls = state.get("seed_controls", {})
            for tab in TAB_CONFIGS:
                seed_controls[f"{tab}_settings"] = defaults.get(tab, {})
            seed_controls["current_preset_name"] = None
            seed_controls["preset_dirty"] = False
            state["seed_controls"] = seed_controls
            
            # Update shared values
            state = update_shared_state_from_preset(state, defaults, None)
            
            # Get values for current tab
            tab_values = dict_to_values(tab_name, defaults.get(tab_name, {}))
            
            return (*_coerce_tab_values_for_ui(tab_values), "✅ Reset all tabs to defaults", state)
        
        except Exception as e:
            defaults = get_all_defaults(base_dir, models_list)
            tab_values = dict_to_values(tab_name, defaults.get(tab_name, {}))
            return (*_coerce_tab_values_for_ui(tab_values), f"❌ Error resetting: {str(e)}", state)
    
    def refresh_dropdown() -> gr.update:
        """Refresh the preset dropdown choices."""
        presets = get_presets_list()
        return gr.update(choices=presets, value=(presets[-1] if presets else None))
    
    def delete_preset(preset_name: str) -> Tuple:
        """Delete a universal preset."""
        if not preset_name or not preset_name.strip():
            return gr.update(), "⚠️ No preset selected to delete"
        
        try:
            if preset_manager.delete_universal_preset(preset_name):
                presets = get_presets_list()
                selected = presets[-1] if presets else None
                return gr.update(choices=presets, value=selected), f"✅ Deleted preset '{preset_name}'"
            else:
                return gr.update(), f"⚠️ Could not delete preset '{preset_name}'"
        except Exception as e:
            return gr.update(), f"❌ Error deleting: {str(e)}"
    
    return {
        "get_presets_list": get_presets_list,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "reset_to_defaults": reset_to_defaults,
        "refresh_dropdown": refresh_dropdown,
        "delete_preset": delete_preset,
    }


def universal_preset_section(
    preset_manager: PresetManager,
    shared_state: gr.State,
    tab_name: str,
    inputs_list: List[gr.components.Component],
    base_dir: Path = None,
    models_list: List[str] = None,
    open_accordion: bool = True,
) -> Tuple[gr.Dropdown, gr.Textbox, gr.Button, gr.Button, gr.Markdown, gr.Button, gr.Button, Dict]:
    """
    Create a universal preset management UI section.
    
    This is IDENTICAL across all tabs - save/load affects ALL tabs.
    
    Args:
        preset_manager: PresetManager instance
        shared_state: Gradio shared state
        tab_name: Name of current tab (seedvr2, gan, etc.)
        inputs_list: List of Gradio components for this tab (matching ORDER)
        base_dir: App base directory
        models_list: List of available models
        open_accordion: Whether accordion is open by default
    
    Returns:
        (dropdown, name_input, save_btn, load_btn, status, reset_btn, delete_btn, callbacks)
    """
    # Get callbacks
    callbacks = create_universal_preset_callbacks(
        preset_manager, tab_name, base_dir, models_list
    )
    
    # Get current preset name from state
    current_preset = None
    if shared_state and shared_state.value:
        current_preset = shared_state.value.get("seed_controls", {}).get("current_preset_name")
    
    # Get list of presets
    presets = callbacks["get_presets_list"]()
    selected_preset = current_preset if current_preset in presets else (presets[-1] if presets else None)
    
    with gr.Accordion("📦 Universal Preset (All Tabs)", open=open_accordion):
        gr.Markdown("""
        **Save/Load affects ALL tabs simultaneously.**  
        One preset contains Global Settings, SeedVR2, GAN, RIFE, FlashVSR+, RTX Super Resolution, Face, Resolution, and Output settings.
        """)
        
        preset_dropdown = gr.Dropdown(
            label="Select Preset",
            choices=presets,
            value=selected_preset,
            allow_custom_value=False,
            info="All your settings from every tab in one file"
        )
        
        with gr.Row():
            preset_name_input = gr.Textbox(
                label="New Preset Name",
                placeholder="my_settings",
                scale=3
            )
            save_preset_btn = gr.Button("💾 Save", variant="primary", scale=1)
        
        with gr.Row():
            load_preset_btn = gr.Button("📂 Load Selected", scale=1)
            reset_defaults_btn = gr.Button("🔄 Reset All", variant="secondary", scale=1)
            delete_preset_btn = gr.Button("🗑️ Delete", variant="stop", scale=1)
        
        preset_status = gr.Markdown("")
    
    return (
        preset_dropdown,
        preset_name_input,
        save_preset_btn,
        load_preset_btn,
        preset_status,
        reset_defaults_btn,
        delete_preset_btn,
        callbacks,
    )


def wire_universal_preset_events(
    preset_dropdown: gr.Dropdown,
    preset_name_input: gr.Textbox,
    save_btn: gr.Button,
    load_btn: gr.Button,
    preset_status: gr.Markdown,
    reset_btn: gr.Button,
    delete_btn: gr.Button,
    callbacks: Dict[str, Callable],
    inputs_list: List[gr.components.Component],
    shared_state: gr.State,
    tab_name: Optional[str] = None,
    enable_auto_sync: bool = True,
) -> Dict[str, Any]:
    """
    Wire up event handlers for universal preset UI.
    
    This connects the buttons to their callbacks, passing the correct inputs/outputs.
    
    Args:
        preset_dropdown: Preset selection dropdown
        preset_name_input: Text input for new preset name
        save_btn: Save button
        load_btn: Load button
        preset_status: Status display
        reset_btn: Reset to defaults button
        delete_btn: Delete preset button
        callbacks: Dict of callback functions from create_universal_preset_callbacks
        inputs_list: List of Gradio components for the current tab
        shared_state: Gradio shared state
    """
    
    # Save preset - wrap to collect unpacked inputs_list back into a list
    def save_preset_wrapper(preset_name, selected_preset_name, *args):
        """Wrapper to collect unpacked inputs_list values"""
        # args = (value1, value2, ..., valueN, shared_state)
        # Split into tab_values and state
        tab_values = list(args[:-1])  # All except last
        state = args[-1]  # Last argument is shared_state
        return callbacks["save_preset"](preset_name, selected_preset_name, tab_values, state)
    
    save_event = save_btn.click(
        fn=save_preset_wrapper,
        inputs=[preset_name_input, preset_dropdown] + inputs_list + [shared_state],
        outputs=[preset_dropdown, preset_status, shared_state],
    )
    
    # Load preset - updates all tab components
    def load_preset_wrapper(preset_name, state):
        """Wrapper to unpack return values for UI components"""
        result = callbacks["load_preset"](preset_name, state)
        # result = (value1, value2, ..., valueN, status_message, updated_state)
        return result
    
    load_click_event = load_btn.click(
        fn=load_preset_wrapper,
        inputs=[preset_dropdown, shared_state],
        outputs=inputs_list + [preset_status, shared_state],
    )
    
    # Auto-load on dropdown change
    load_change_event = preset_dropdown.change(
        fn=load_preset_wrapper,
        inputs=[preset_dropdown, shared_state],
        outputs=inputs_list + [preset_status, shared_state],
    )
    
    # Reset to defaults
    def reset_defaults_wrapper(state):
        """Wrapper to unpack return values for UI components"""
        result = callbacks["reset_to_defaults"](state)
        return result
    
    reset_event = reset_btn.click(
        fn=reset_defaults_wrapper,
        inputs=[shared_state],
        outputs=inputs_list + [preset_status, shared_state],
    )
    
    # Delete preset
    delete_event = delete_btn.click(
        fn=callbacks["delete_preset"],
        inputs=[preset_dropdown],
        outputs=[preset_dropdown, preset_status],
    )

    # ------------------------------------------------------------------ #
    # Auto-sync current tab values into shared_state on ANY change.
    #
    # Why: Universal presets save ALL tabs from shared_state. Without auto-sync,
    # changes made in other tabs may not be reflected when saving from a different tab.
    #
    # Uses Gradio 6.x `gr.on()` to avoid wiring dozens of separate change handlers.
    # ------------------------------------------------------------------ #
    sync_event = None
    if enable_auto_sync and tab_name:
        def _sync_wrapper(*args):
            # args = (value1, value2, ..., valueN, shared_state)
            tab_values = list(args[:-1])
            state = args[-1]
            next_state, changed = sync_tab_to_shared_state(
                tab_name,
                tab_values,
                state,
                return_changed=True,
            )
            if not changed:
                return gr.skip()
            return next_state

        triggers = []
        for comp in inputs_list:
            # Use a single best trigger per component to reduce fan-out:
            # release (sliders) -> input (text-like) -> change (fallback).
            if hasattr(comp, "release"):
                triggers.append(comp.release)
            elif hasattr(comp, "input"):
                triggers.append(comp.input)
            elif hasattr(comp, "change"):
                triggers.append(comp.change)

        # Prefer gr.on() for a single endpoint; fall back to per-component wiring if needed.
        if hasattr(gr, "on") and triggers:
            sync_event = gr.on(
                triggers=triggers,
                fn=_sync_wrapper,
                inputs=inputs_list + [shared_state],
                outputs=[shared_state],
                queue=False,
                show_progress="hidden",
                trigger_mode="always_last",
            )
        else:
            # Fallback: register one event per component (heavier, but compatible).
            for comp in inputs_list:
                if hasattr(comp, "release"):
                    comp.release(
                        fn=_sync_wrapper,
                        inputs=inputs_list + [shared_state],
                        outputs=[shared_state],
                        queue=False,
                        show_progress="hidden",
                        trigger_mode="always_last",
                    )
                elif hasattr(comp, "input"):
                    comp.input(
                        fn=_sync_wrapper,
                        inputs=inputs_list + [shared_state],
                        outputs=[shared_state],
                        queue=False,
                        show_progress="hidden",
                        trigger_mode="always_last",
                    )
                elif hasattr(comp, "change"):
                    comp.change(
                        fn=_sync_wrapper,
                        inputs=inputs_list + [shared_state],
                        outputs=[shared_state],
                        queue=False,
                        show_progress="hidden",
                        trigger_mode="always_last",
                    )

    return {
        "save": save_event,
        "load_click": load_click_event,
        "load_change": load_change_event,
        "reset": reset_event,
        "delete": delete_event,
        "sync": sync_event,
    }


def sync_tab_to_shared_state(
    tab_name: str,
    values: List[Any],
    state: Dict[str, Any],
    return_changed: bool = False,
) -> Dict[str, Any] | Tuple[Dict[str, Any], bool]:
    """
    Sync current tab values to shared_state.
    
    Call this when any value in the tab changes to keep state updated.
    """
    state = state if isinstance(state, dict) else {}
    seed_controls = state.get("seed_controls", {})
    if not isinstance(seed_controls, dict):
        seed_controls = {}
    try:
        before_seed_controls = copy.deepcopy(seed_controls)
    except Exception:
        before_seed_controls = seed_controls.copy()

    tab_settings_key = f"{tab_name}_settings"
    tab_dict = values_to_dict(tab_name, values)
    existing_tab_settings = seed_controls.get(tab_settings_key, {})
    if not isinstance(existing_tab_settings, dict):
        existing_tab_settings = {}
    try:
        existing_normalized = values_to_dict(
            tab_name,
            dict_to_values(tab_name, existing_tab_settings),
        )
    except Exception:
        existing_normalized = existing_tab_settings

    tab_settings_changed = tab_dict != existing_normalized
    if tab_settings_changed:
        seed_controls[tab_settings_key] = tab_dict
    elif tab_settings_key not in seed_controls:
        seed_controls[tab_settings_key] = tab_dict
        tab_settings_changed = True

    # Keep derived cross-tab caches in sync so other pipelines immediately
    # see updated Resolution/Output settings without requiring "Apply" buttons.
    if tab_name == "global":
        seed_controls["face_strength_val"] = float(tab_dict.get("face_strength", 0.5) or 0.5)
        seed_controls["queue_enabled_val"] = bool(tab_dict.get("queue_enabled", True))
        seed_controls["theme_mode_val"] = str(tab_dict.get("theme_mode", "dark") or "dark")
        seed_controls["pinned_reference_path"] = tab_dict.get("pinned_reference_path")
        global_gpu_device = resolve_global_gpu_device(tab_dict.get("global_gpu_device"))
        tab_dict["global_gpu_device"] = global_gpu_device
        seed_controls["global_gpu_device_val"] = global_gpu_device
        seed_controls["global_rife_cuda_device_val"] = "" if global_gpu_device == "cpu" else global_gpu_device
        output_settings = seed_controls.get("output_settings", {})
        if isinstance(output_settings, dict):
            output_settings["global_rife_cuda_device"] = seed_controls["global_rife_cuda_device_val"]
            seed_controls["output_settings"] = output_settings

    if tab_name == "resolution":
        seed_controls["auto_detect_scenes"] = bool(tab_dict.get("auto_detect_scenes", True))
        seed_controls["auto_chunk"] = bool(tab_dict.get("auto_chunk", True))
        seed_controls["frame_accurate_split"] = bool(tab_dict.get("frame_accurate_split", True))
        seed_controls["chunk_size_sec"] = float(tab_dict.get("chunk_size", 0) or 0)
        if seed_controls["auto_chunk"]:
            tab_dict["chunk_overlap"] = 0.0
            seed_controls["chunk_overlap_sec"] = 0.0
        else:
            seed_controls["chunk_overlap_sec"] = float(tab_dict.get("chunk_overlap", 0.0) or 0.0)
        seed_controls["per_chunk_cleanup"] = bool(tab_dict.get("per_chunk_cleanup", False))
        seed_controls["scene_threshold"] = float(tab_dict.get("scene_threshold", 27.0) or 27.0)
        seed_controls["min_scene_len"] = float(tab_dict.get("min_scene_len", 1.0) or 1.0)

    if tab_name == "output":
        seed_controls["png_padding_val"] = int(tab_dict.get("png_padding", 6) or 6)
        seed_controls["png_keep_basename_val"] = bool(tab_dict.get("png_keep_basename", True))
        seed_controls["png_sequence_enabled_val"] = bool(tab_dict.get("png_sequence_enabled", False))
        seed_controls["overwrite_existing_batch_val"] = bool(tab_dict.get("overwrite_existing_batch", False))
        seed_controls["skip_first_frames_val"] = int(tab_dict.get("skip_first_frames", 0) or 0)
        seed_controls["load_cap_val"] = int(tab_dict.get("load_cap", 0) or 0)
        seed_controls["fps_override_val"] = float(tab_dict.get("fps_override", 0) or 0)
        seed_controls["image_output_format_val"] = str(tab_dict.get("image_output_format", "png") or "png")
        seed_controls["image_output_quality_val"] = int(tab_dict.get("image_output_quality", 95) or 95)
        seed_controls["seedvr2_video_backend_val"] = str(tab_dict.get("seedvr2_video_backend", "ffmpeg") or "ffmpeg")
        seed_controls["seedvr2_use_10bit_val"] = bool(tab_dict.get("seedvr2_use_10bit", False))
        seed_controls["video_codec_val"] = str(tab_dict.get("video_codec", "h264") or "h264")
        seed_controls["video_quality_val"] = int(tab_dict.get("video_quality", 18) or 18)
        seed_controls["video_preset_val"] = str(tab_dict.get("video_preset", "medium") or "medium")
        seed_controls["h265_tune_val"] = str(tab_dict.get("h265_tune", "none") or "none")
        seed_controls["av1_film_grain_val"] = int(
            tab_dict.get("av1_film_grain", DEFAULT_AV1_FILM_GRAIN) or DEFAULT_AV1_FILM_GRAIN
        )
        seed_controls["av1_film_grain_denoise_val"] = bool(
            tab_dict.get("av1_film_grain_denoise", DEFAULT_AV1_FILM_GRAIN_DENOISE)
        )
        seed_controls["two_pass_encoding_val"] = bool(tab_dict.get("two_pass_encoding", False))
        seed_controls["pixel_format_val"] = str(tab_dict.get("pixel_format", "yuv420p") or "yuv420p")
        seed_controls["frame_interpolation_val"] = bool(tab_dict.get("frame_interpolation", False))
        seed_controls["global_rife_enabled_val"] = bool(tab_dict.get("frame_interpolation", False))
        seed_controls["global_rife_multiplier_val"] = tab_dict.get("global_rife_multiplier", "x2") or "x2"
        model_name = str(tab_dict.get("global_rife_model", "") or "").strip() or get_rife_default_model()
        tab_dict["global_rife_model"] = model_name
        seed_controls["global_rife_model_val"] = model_name
        precision = str(tab_dict.get("global_rife_precision", "fp32") or "fp32").lower()
        seed_controls["global_rife_precision_val"] = "fp16" if precision == "fp16" else "fp32"
        current_global = seed_controls.get("global_settings", {})
        current_global = current_global if isinstance(current_global, dict) else {}
        global_gpu_device = resolve_global_gpu_device(
            current_global.get("global_gpu_device", seed_controls.get("global_gpu_device_val"))
        )
        seed_controls["global_gpu_device_val"] = global_gpu_device
        tab_dict["global_rife_cuda_device"] = "" if global_gpu_device == "cpu" else global_gpu_device
        seed_controls["global_rife_cuda_device_val"] = tab_dict["global_rife_cuda_device"]
        seed_controls["global_rife_process_chunks_val"] = bool(tab_dict.get("global_rife_process_chunks", True))
        seed_controls["output_format_val"] = tab_dict.get("output_format", "auto") or "auto"
        seed_controls["comparison_mode_val"] = tab_dict.get("comparison_mode", "slider") or "slider"
        seed_controls["pin_reference_val"] = bool(tab_dict.get("pin_reference", False))
        seed_controls["fullscreen_val"] = bool(tab_dict.get("fullscreen_enabled", True))
        seed_controls["save_metadata_val"] = bool(tab_dict.get("save_metadata", True))
        seed_controls["telemetry_enabled_val"] = bool(tab_dict.get("telemetry_enabled", True))
        seed_controls["audio_codec_val"] = str(tab_dict.get("audio_codec", "copy") or "copy")
        seed_controls["audio_bitrate_val"] = str(tab_dict.get("audio_bitrate", "") or "")
        seed_controls["generate_comparison_video_val"] = bool(tab_dict.get("generate_comparison_video", True))
        seed_controls["comparison_video_layout_val"] = str(tab_dict.get("comparison_video_layout", "auto") or "auto")

    # Mark dirty only when user-facing tab settings actually changed.
    if tab_settings_changed:
        seed_controls["preset_dirty"] = True

    state["seed_controls"] = seed_controls
    changed = before_seed_controls != seed_controls
    if return_changed:
        return state, changed
    return state


def get_tab_values_from_state(
    tab_name: str,
    state: Dict[str, Any],
    base_dir: Path = None,
    models_list: List[str] = None,
) -> List[Any]:
    """
    Get tab values from shared_state.
    
    Call this when tab is loaded/shown to get values synced from other tabs.
    """
    seed_controls = state.get("seed_controls", {})
    tab_settings = seed_controls.get(f"{tab_name}_settings", {})
    
    if not tab_settings:
        # No settings in state, use defaults
        defaults = get_all_defaults(base_dir, models_list)
        tab_settings = defaults.get(tab_name, {})
    
    return dict_to_values(tab_name, tab_settings)

