"""
Shared Preset Utilities - DRY Preset Management
Eliminates code duplication across all service modules
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import gradio as gr
from shared.preset_manager import PresetManager


def create_preset_callbacks(
    tab_name: str,
    preset_manager: PresetManager,
    order: List[str],
    defaults_func: Callable[[], Dict[str, Any]],
    apply_guardrails_func: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = None,
    validate_constraints_func: Optional[Callable[[Dict[str, Any], str, Optional[str]], Dict[str, Any]]] = None
) -> Dict[str, Callable]:
    """
    Create standardized preset callbacks for any tab.
    
    This eliminates code duplication across SeedVR2/GAN/RIFE/FlashVSR services.
    
    Args:
        tab_name: Tab identifier for preset storage (e.g., "seedvr2", "gan")
        preset_manager: PresetManager instance
        order: List of parameter keys in UI order
        defaults_func: Function that returns default values dict
        apply_guardrails_func: Optional function to apply tab-specific validation
        validate_constraints_func: Optional function for preset constraint validation
    
    Returns:
        Dict with standard callbacks: refresh_presets, save_preset, load_preset, safe_defaults
    """
    
    def _dict_from_args(args: List[Any]) -> Dict[str, Any]:
        """Convert argument list to settings dict"""
        return dict(zip(order, args))
    
    def _apply_preset_to_values(
        preset: Dict[str, Any],
        defaults: Dict[str, Any],
        current: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """Apply preset values to current settings"""
        base = defaults.copy()
        if current:
            base.update(current)
        merged = preset_manager.merge_config(base, preset)
        
        # Apply guardrails if provided
        if apply_guardrails_func:
            merged = apply_guardrails_func(merged, defaults)
        
        return [merged.get(key, defaults.get(key)) for key in order]
    
    def refresh_presets(model_name: str, select_name: Optional[str] = None):
        """Refresh preset dropdown for given model"""
        presets = preset_manager.list_presets(tab_name, model_name)
        last_used = preset_manager.get_last_used_name(tab_name, model_name)
        
        # Priority: select_name > last_used > last in list
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        
        return gr.update(choices=presets, value=value)
    
    def save_preset(preset_name: str, model_name: str, *args):
        """
        Save preset with validation.
        
        Args:
            preset_name: Name for the preset
            model_name: Model identifier
            *args: Flattened UI values (must match order length)
        
        Returns:
            Tuple of (dropdown_update, status_message, *validated_values)
        """
        if not preset_name.strip():
            return (
                gr.update(),
                gr.update(value="⚠️ Enter a preset name before saving"),
                *list(args)
            )
        
        try:
            # Validate component count
            if len(args) != len(order):
                error_msg = (
                    f"⚠️ Preset schema mismatch: received {len(args)} values but expected {len(order)}. "
                    f"Check inputs_list in {tab_name}_tab.py"
                )
                return (
                    gr.update(),
                    gr.update(value=error_msg),
                    *list(args)
                )
            
            # Convert to dict
            payload = _dict_from_args(list(args))
            
            # Apply guardrails if provided
            defaults = defaults_func()
            if apply_guardrails_func:
                payload = apply_guardrails_func(payload, defaults)
            
            # Save to disk
            preset_manager.save_preset_safe(tab_name, model_name, preset_name.strip(), payload)
            
            # Refresh dropdown with new preset selected
            dropdown = refresh_presets(model_name, select_name=preset_name.strip())
            
            # Reload validated values for UI consistency
            current_map = _dict_from_args(list(args))
            loaded_vals = _apply_preset_to_values(payload, defaults, current=current_map)
            
            return (
                dropdown,
                gr.update(value=f"✅ Saved preset '{preset_name}' for {model_name}"),
                *loaded_vals
            )
            
        except Exception as e:
            return (
                gr.update(),
                gr.update(value=f"❌ Error saving preset: {str(e)}"),
                *list(args)
            )
    
    def load_preset(preset_name: str, model_name: str, current_values: List[Any]):
        """
        Load preset and return values + status message.
        
        Args:
            preset_name: Preset to load
            model_name: Model identifier
            current_values: Current UI values (for fallback)
        
        Returns:
            Tuple of (*values, status_message) - note status is LAST
        """
        try:
            # Load preset from disk
            preset = preset_manager.load_preset_safe(tab_name, model_name, preset_name)
            
            if preset:
                # Mark as last used
                preset_manager.set_last_used(tab_name, model_name, preset_name)
                
                # Apply validation constraints if provided
                if validate_constraints_func:
                    preset = validate_constraints_func(preset, tab_name, model_name)
                elif hasattr(preset_manager, 'validate_preset_constraints'):
                    preset = preset_manager.validate_preset_constraints(preset, tab_name, model_name)
            
            # Get defaults
            defaults = defaults_func()
            if model_name:
                # Update model in defaults for model-specific settings
                defaults_copy = defaults.copy()
                # Handle different model key names
                for key in ["model", "dit_model", "rife_model", "gan_model"]:
                    if key in defaults_copy:
                        defaults_copy[key] = model_name
                        break
                defaults = defaults_copy
            
            # Merge with current values
            current_map = _dict_from_args(current_values)
            values = _apply_preset_to_values(preset or {}, defaults, current=current_map)
            
            # Return values + status message (status LAST to match UI expectations)
            status_msg = f"✅ Loaded preset '{preset_name}'" if preset else "ℹ️ Preset not found, using defaults"
            return (*values, gr.update(value=status_msg))
            
        except Exception as e:
            print(f"Error loading preset {preset_name} for {tab_name}/{model_name}: {e}")
            # Return current values + error status
            return (*current_values, gr.update(value=f"❌ Error: {str(e)}"))
    
    def safe_defaults():
        """Get safe default values in order"""
        defaults = defaults_func()
        return [defaults.get(key) for key in order]
    
    return {
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
    }
