import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sanitize_name(name: str) -> str:
    """Make a filesystem-safe filename from a preset or model name."""
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(name))
    return safe.strip("._") or "default"


# File that tracks the last used universal preset
LAST_USED_PRESET_FILE = ".last_used_preset.txt"


class PresetManager:
    """
    File-based preset manager with UNIVERSAL PRESET support.

    NEW Layout (v2.0 - Universal Presets):
    presets/
      global.json                    # Global app settings
      .last_used_preset.txt          # Name of last used universal preset
      my_preset.json                 # Universal preset (contains ALL tabs)
      another_preset.json            # Another universal preset
    
    Universal preset format:
    {
      "_meta": {"version": "2.0", "format": "universal", ...},
      "seedvr2": {...},              # All SeedVR2 settings
      "gan": {...},                  # All GAN settings
      "rife": {...},                 # All RIFE settings
      "flashvsr": {...},             # All FlashVSR settings
      "face": {...},                 # All Face settings
      "resolution": {...},           # All Resolution settings
      "output": {...}                # All Output settings
    }
    
    OLD Layout (deprecated, auto-migrated):
    presets/<tab>/<preset_name>.json
    """

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._universal_presets_cache_ts = 0.0
        self._universal_presets_cache_dir_mtime = 0.0
        self._universal_presets_cache: List[str] = []
        # Auto-migrate old preset structure on first load
        self._migrate_old_preset_structure()

    # ------------------------------------------------------------------ #
    # Core helpers
    # ------------------------------------------------------------------ #
    def _tab_dir(self, tab: str) -> Path:
        return self.base_dir / _sanitize_name(tab)

    def _preset_path(self, tab: str, preset_name: str) -> Path:
        """Preset files are stored directly in tab folder (no model subfolder)"""
        return self._tab_dir(tab) / f"{_sanitize_name(preset_name)}.json"

    def _last_used_path(self, tab: str, model: Optional[str]) -> Path:
        """Last used tracking is per-model, stored in .last_used subfolder"""
        last_used_dir = self._tab_dir(tab) / ".last_used"
        last_used_dir.mkdir(parents=True, exist_ok=True)
        model_name = _sanitize_name(model) if model else "default"
        return last_used_dir / f"{model_name}.txt"

    # ------------------------------------------------------------------ #
    # Global settings
    # ------------------------------------------------------------------ #
    def load_global_settings(self, defaults: Dict[str, Any]) -> Dict[str, Any]:
        path = self.base_dir / "global.json"
        if not path.exists():
            return defaults.copy()
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            merged = defaults.copy()
            merged.update(data)
            return merged
        except Exception:
            return defaults.copy()

    def save_global_settings(self, settings: Dict[str, Any]) -> None:
        path = self.base_dir / "global.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        tmp_path.replace(path)

    # ------------------------------------------------------------------ #
    # Preset operations
    # ------------------------------------------------------------------ #
    def list_presets(self, tab: str, model: Optional[str] = None) -> List[str]:
        """List all presets for a tab (model parameter kept for compatibility but not used)"""
        folder = self._tab_dir(tab)
        if not folder.exists():
            return []
        # List all JSON files in tab folder (excluding .last_used subfolder)
        return sorted([p.stem for p in folder.glob("*.json")])

    def save_preset(self, tab: str, model: Optional[str], preset_name: str, data: Dict[str, Any]) -> str:
        """Save preset in tab folder (not model-specific subfolder)"""
        folder = self._tab_dir(tab)
        folder.mkdir(parents=True, exist_ok=True)
        preset_path = self._preset_path(tab, preset_name)
        tmp_path = preset_path.with_suffix(".json.tmp")
        
        # Ensure model name is stored in the preset data itself
        if model and "dit_model" in data:
            # Model is already in data for seedvr2/gan/rife tabs
            pass
        elif model:
            # For other tabs, store model reference
            data["_model"] = model
        
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(preset_path)
        # Track last used per model
        self.set_last_used(tab, model, preset_name)
        return preset_path.stem

    def load_preset(self, tab: str, model: Optional[str], preset_name: str) -> Optional[Dict[str, Any]]:
        """Load preset from tab folder (model parameter kept for compatibility)"""
        preset_path = self._preset_path(tab, preset_name)
        if not preset_path.exists():
            return None
        try:
            with preset_path.open("r", encoding="utf-8") as f:
                preset_data = json.load(f)
            
            # Validate and clean loaded preset
            cleaned = self.validate_and_clean_preset(preset_data, tab, model)
            
            # Apply tab-specific constraints
            validated = self.validate_preset_constraints(cleaned, tab, model)
            
            # Tab-specific validation on load
            if tab == "seedvr2":
                # Validate batch_size 4n+1 formula
                batch_size = validated.get("batch_size", 5)
                if (batch_size - 1) % 4 != 0:
                    corrected = max(5, (batch_size // 4) * 4 + 1)
                    validated["batch_size"] = corrected
                    print(f"⚠️ Preset '{preset_name}' had invalid batch_size {batch_size}, auto-corrected to {corrected}")
                
                # Check cache + multi-GPU conflict
                cache_enabled = validated.get("cache_dit") or validated.get("cache_vae")
                cuda_device = str(validated.get("cuda_device", ""))
                if cache_enabled and cuda_device:
                    devices = [d.strip() for d in cuda_device.split(",") if d.strip()]
                    if len(devices) > 1:
                        validated["cache_dit"] = False
                        validated["cache_vae"] = False
                        print(f"⚠️ Preset '{preset_name}' had cache enabled with multi-GPU, auto-disabled caching")
                
                # ENHANCED: Model-specific validation
                model_name = str(model or validated.get("dit_model", ""))
                
                # GGUF model limitations
                if model_name.lower().endswith(".gguf"):
                    # GGUF models don't support torch.compile
                    if validated.get("compile_dit") or validated.get("compile_vae"):
                        validated["compile_dit"] = False
                        validated["compile_vae"] = False
                        print(f"⚠️ Preset '{preset_name}': GGUF models don't support torch.compile, auto-disabled")
                    
                    # GGUF models don't support multi-GPU
                    if cuda_device and "," in cuda_device:
                        # Keep only first GPU
                        first_gpu = cuda_device.split(",")[0].strip()
                        validated["cuda_device"] = first_gpu
                        print(f"⚠️ Preset '{preset_name}': GGUF models don't support multi-GPU, using GPU {first_gpu} only")
                
                # BlockSwap requires dit_offload_device != none
                blocks_to_swap = int(validated.get("blocks_to_swap", 0))
                dit_offload = str(validated.get("dit_offload_device", "none"))
                if blocks_to_swap > 0 and dit_offload == "none":
                    validated["blocks_to_swap"] = 0
                    print(f"⚠️ Preset '{preset_name}': BlockSwap requires dit_offload_device != 'none', auto-disabled BlockSwap")
                
                # VAE tiling overlap must be < tile size
                vae_encode_tiled = validated.get("vae_encode_tiled", False)
                vae_encode_tile_size = int(validated.get("vae_encode_tile_size", 1024))
                vae_encode_overlap = int(validated.get("vae_encode_tile_overlap", 128))
                if vae_encode_tiled and vae_encode_overlap >= vae_encode_tile_size:
                    validated["vae_encode_tile_overlap"] = vae_encode_tile_size // 2
                    print(f"⚠️ Preset '{preset_name}': VAE encode overlap must be < tile size, auto-corrected to {vae_encode_tile_size // 2}")
                
                vae_decode_tiled = validated.get("vae_decode_tiled", False)
                vae_decode_tile_size = int(validated.get("vae_decode_tile_size", 1024))
                vae_decode_overlap = int(validated.get("vae_decode_tile_overlap", 128))
                if vae_decode_tiled and vae_decode_overlap >= vae_decode_tile_size:
                    validated["vae_decode_tile_overlap"] = vae_decode_tile_size // 2
                    print(f"⚠️ Preset '{preset_name}': VAE decode overlap must be < tile size, auto-corrected to {vae_decode_tile_size // 2}")
                
                # Attention mode GPU compatibility check
                attention_mode = str(validated.get("attention_mode", "sdpa"))
                try:
                    # Use NVML-based GPU probing to avoid importing torch in the parent process.
                    from .gpu_utils import get_gpu_info

                    gpus = get_gpu_info()
                    compute_cap = gpus[0].compute_capability if gpus else None
                    if compute_cap:
                        # flash_attn_3 and sageattn_3 require Hopper+ (9.0+)
                        if attention_mode in ("flash_attn_3", "sageattn_3") and compute_cap[0] < 9:
                            validated["attention_mode"] = "flash_attn_2" if compute_cap[0] >= 8 else "sdpa"
                            print(
                                f"⚠️ Preset '{preset_name}': {attention_mode} requires Hopper+ GPU (9.0+), "
                                f"falling back to {validated['attention_mode']}"
                            )

                        # flash_attn_2 works best on Ampere+ (8.0+)
                        elif attention_mode == "flash_attn_2" and compute_cap[0] < 8:
                            validated["attention_mode"] = "sdpa"
                            print(
                                f"⚠️ Preset '{preset_name}': flash_attn_2 requires Ampere+ GPU (8.0+), "
                                f"falling back to sdpa"
                            )
                except Exception:
                    pass  # Skip GPU check if torch unavailable
            
            return validated
        except Exception as e:
            print(f"Error loading preset {preset_name} for {tab}/{model}: {e}")
            return None

    def delete_preset(self, tab: str, model: Optional[str], preset_name: str) -> bool:
        """Delete preset (model parameter kept for compatibility but not used)"""
        preset_path = self._preset_path(tab, preset_name)
        if preset_path.exists():
            preset_path.unlink()
            return True
        return False

    # ------------------------------------------------------------------ #
    # Last used tracking
    # ------------------------------------------------------------------ #
    def set_last_used(self, tab: str, model: Optional[str], preset_name: str) -> None:
        path = self._last_used_path(tab, model)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(_sanitize_name(preset_name))

    def get_last_used_name(self, tab: str, model: Optional[str]) -> Optional[str]:
        path = self._last_used_path(tab, model)
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8").strip() or None
        except Exception:
            return None

    def load_last_used(self, tab: str, model: Optional[str]) -> Optional[Dict[str, Any]]:
        """Load last used preset for a model (model used for tracking, not storage location)"""
        name = self.get_last_used_name(tab, model)
        if not name:
            return None
        return self.load_preset(tab, None, name)  # Load from tab folder (not model subfolder)

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def merge_config(current: Dict[str, Any], preset: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge preset values onto current values, preserving current values for missing preset keys.
        
        This allows older presets to work with new features - if a preset doesn't have a key,
        the current default value is preserved.
        """
        if not preset:
            return current.copy()
        
        merged = current.copy()
        
        # Only update keys that exist in the preset
        # Keys in current that aren't in preset are preserved with their default values
        for key, value in preset.items():
            # Type checking to ensure compatibility
            if key in current:
                # Try to preserve type of current value
                try:
                    current_type = type(current[key])
                    if current_type in (int, float, str, bool):
                        # Convert to expected type
                        merged[key] = current_type(value)
                    else:
                        # For complex types, use as-is
                        merged[key] = value
                except (ValueError, TypeError):
                    # If conversion fails, use default
                    merged[key] = current[key]
            else:
                # New key from preset not in current defaults - add it
                merged[key] = value
        
        return merged

    @staticmethod
    def validate_and_clean_preset(preset: Dict[str, Any], tab: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Validate and clean preset data before saving/loading."""
        if not isinstance(preset, dict):
            return {}

        cleaned = {}
        for key, value in preset.items():
            # Basic type validation - ensure values are JSON serializable
            try:
                import json
                json.dumps({key: value})  # Test serialization
                cleaned[key] = value
            except (TypeError, ValueError):
                # Skip non-serializable values with warning
                print(f"Warning: Skipping non-serializable preset value for key '{key}' in {tab}/{model}")
                continue

        return cleaned

    def save_preset_safe(self, tab: str, model: Optional[str], preset_name: str, data: Dict[str, Any]) -> str:
        """Safe preset saving with validation and error handling."""
        try:
            # Validate and clean the data
            cleaned_data = self.validate_and_clean_preset(data, tab, model)
            if not cleaned_data:
                raise ValueError("No valid data to save")

            # Save the preset
            result = self.save_preset(tab, model, preset_name, cleaned_data)
            return result
        except Exception as e:
            print(f"Error saving preset {preset_name} for {tab}/{model}: {e}")
            raise

    def load_preset_safe(self, tab: str, model: Optional[str], preset_name: str) -> Optional[Dict[str, Any]]:
        """Safe preset loading with error handling."""
        try:
            preset = self.load_preset(tab, model, preset_name)
            if preset is not None:
                # Validate loaded data
                cleaned = self.validate_and_clean_preset(preset, tab, model)
                return cleaned
            return None
        except Exception as e:
            print(f"Error loading preset {preset_name} for {tab}/{model}: {e}")
            return None

    def validate_preset_constraints(self, preset: Dict[str, Any], tab: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Apply tab-specific validation constraints to presets."""
        validated = preset.copy()

        if tab == "seedvr2":
            # SeedVR2 specific validations
            validated = self._validate_seedvr2_constraints(validated)
        elif tab == "gan":
            # GAN specific validations
            validated = self._validate_gan_constraints(validated)
        elif tab == "flashvsr":
            # FlashVSR+ specific validations
            validated = self._validate_flashvsr_constraints(validated)
        elif tab == "rife":
            # RIFE specific validations
            validated = self._validate_rife_constraints(validated)

        return validated

    def _validate_seedvr2_constraints(self, preset: Dict[str, Any]) -> Dict[str, Any]:
        """Apply SeedVR2-specific validation rules."""
        validated = preset.copy()

        # Batch size must be 4n+1
        try:
            bs = int(validated.get("batch_size", 5))
            if bs % 4 != 1:
                validated["batch_size"] = max(1, (bs // 4) * 4 + 1)
        except (ValueError, TypeError):
            validated["batch_size"] = 5  # Safe default

        # Resolution must be multiple of 16
        try:
            res = int(validated.get("resolution", 1080))
            if res % 16 != 0:
                validated["resolution"] = (res // 16) * 16
        except (ValueError, TypeError):
            validated["resolution"] = 1080

        # Max resolution validation
        try:
            max_res = int(validated.get("max_resolution", 0))
            if max_res < 0:
                validated["max_resolution"] = 0
        except (ValueError, TypeError):
            validated["max_resolution"] = 0

        # VAE tiling constraints
        if validated.get("vae_encode_tiled"):
            try:
                tile_size = int(validated.get("vae_encode_tile_size", 1024))
                overlap = int(validated.get("vae_encode_tile_overlap", 128))
                if overlap >= tile_size:
                    validated["vae_encode_tile_overlap"] = max(0, tile_size - 1)
                if tile_size < 64:
                    validated["vae_encode_tile_size"] = 512
            except (ValueError, TypeError):
                validated["vae_encode_tile_size"] = 1024
                validated["vae_encode_tile_overlap"] = 128

        if validated.get("vae_decode_tiled"):
            try:
                tile_size = int(validated.get("vae_decode_tile_size", 1024))
                overlap = int(validated.get("vae_decode_tile_overlap", 128))
                if overlap >= tile_size:
                    validated["vae_decode_tile_overlap"] = max(0, tile_size - 1)
                if tile_size < 64:
                    validated["vae_decode_tile_size"] = 512
            except (ValueError, TypeError):
                validated["vae_decode_tile_size"] = 1024
                validated["vae_decode_tile_overlap"] = 128

        # BlockSwap requires dit_offload_device
        try:
            blocks_to_swap = int(validated.get("blocks_to_swap", 0))
            swap_io = bool(validated.get("swap_io_components", False))
            blockswap_enabled = blocks_to_swap > 0 or swap_io
            if blockswap_enabled:
                offload_device = str(validated.get("dit_offload_device", "none")).lower().strip()
                if offload_device in ("none", ""):
                    validated["dit_offload_device"] = "cpu"
        except (ValueError, TypeError):
            pass

        # Multi-GPU constraints - improved parsing
        try:
            cuda_device_str = str(validated.get("cuda_device", "0"))
            # Remove all whitespace and split by comma
            devices = [d.strip() for d in cuda_device_str.replace(" ", "").split(",") if d.strip() and d.strip().isdigit()]
            
            if len(devices) > 1:
                # Multi-GPU detected - disable cache options
                if validated.get("cache_dit"):
                    validated["cache_dit"] = False
                if validated.get("cache_vae"):
                    validated["cache_vae"] = False
        except Exception:
            # If parsing fails, assume single GPU
            pass

        # Validate compile cache limits
        try:
            cache_limit = int(validated.get("compile_dynamo_cache_size_limit", 64))
            if cache_limit < 1:
                validated["compile_dynamo_cache_size_limit"] = 64
        except (ValueError, TypeError):
            validated["compile_dynamo_cache_size_limit"] = 64

        try:
            recompile_limit = int(validated.get("compile_dynamo_recompile_limit", 128))
            if recompile_limit < 1:
                validated["compile_dynamo_recompile_limit"] = 128
        except (ValueError, TypeError):
            validated["compile_dynamo_recompile_limit"] = 128

        # Validate noise scales (0.0 to 1.0)
        for noise_key in ["input_noise_scale", "latent_noise_scale"]:
            try:
                noise_val = float(validated.get(noise_key, 0.0))
                validated[noise_key] = max(0.0, min(1.0, noise_val))
            except (ValueError, TypeError):
                validated[noise_key] = 0.0

        # Validate seed
        try:
            seed = int(validated.get("seed", 42))
            # Allow -1 for random
            if seed < -1:
                validated["seed"] = 42
        except (ValueError, TypeError):
            validated["seed"] = 42
        
        # ADDED v2.5.22: Validate video backend and 10-bit encoding
        # Ensure video_backend is valid choice
        video_backend = str(validated.get("video_backend", "opencv")).lower()
        if video_backend not in ("opencv", "ffmpeg"):
            validated["video_backend"] = "opencv"
            print(f"⚠️ Preset had invalid video_backend '{video_backend}', reset to 'opencv'")
        
        # Auto-disable 10-bit if backend is not ffmpeg
        use_10bit = bool(validated.get("use_10bit", False))
        if use_10bit and validated.get("video_backend") != "ffmpeg":
            validated["use_10bit"] = False
            print(f"⚠️ Preset had 10-bit enabled without ffmpeg backend, auto-disabled")

        return validated

    def _validate_gan_constraints(self, preset: Dict[str, Any]) -> Dict[str, Any]:
        """Apply GAN-specific validation rules."""
        # GAN models typically don't have the same constraints as SeedVR2
        # Add any GAN-specific validations here if needed
        return preset
    
    def _validate_flashvsr_constraints(self, preset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply FlashVSR+-specific validation rules using metadata registry.
        
        Enforces:
        - Single GPU requirement (multi-GPU not supported)
        - Tile size/overlap constraints
        - Valid version/mode/scale combinations
        """
        validated = preset.copy()
        
        # Import metadata functions
        from shared.models.flashvsr_meta import get_flashvsr_metadata
        
        # Build model identifier
        model_id = f"v{validated.get('version', '10')}_{validated.get('mode', 'tiny')}_{validated.get('scale', 4)}x"
        model_meta = get_flashvsr_metadata(model_id)
        
        if model_meta:
            # Enforce single GPU
            device_str = str(validated.get("device", "auto"))
            if device_str not in ("auto", "cpu", ""):
                devices = [d.strip() for d in device_str.replace(" ", "").split(",") if d.strip()]
                if len(devices) > 1:
                    validated["device"] = devices[0]  # Use first GPU only
                    print(f"WARNING: FlashVSR+ preset: Multi-GPU not supported, using single GPU: {devices[0]}")
            
            # Validate tile constraints
            if validated.get("tiled_vae") or validated.get("tiled_dit"):
                tile_size = int(validated.get("tile_size", 256))
                overlap = int(validated.get("overlap", 24))
                
                if overlap >= tile_size:
                    validated["overlap"] = max(0, tile_size - 1)
                    print(f"WARNING: FlashVSR+ preset: Tile overlap >= tile size, correcting to {validated['overlap']}")
                
                if tile_size < 64:
                    validated["tile_size"] = model_meta.default_tile_size
                    print(f"WARNING: FlashVSR+ preset: Tile size too small, resetting to {validated['tile_size']}")
        
        return validated
    
    def _validate_rife_constraints(self, preset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply RIFE-specific validation rules using metadata registry.
        
        Enforces:
        - Single GPU requirement (multi-GPU not supported)
        - FPS multiplier limits from model metadata
        - Scale factor validation
        """
        validated = preset.copy()
        
        # Import metadata functions
        from shared.models.rife_meta import get_rife_metadata, get_rife_default_model
        
        # Get model metadata
        model_name = validated.get("rife_model", get_rife_default_model())
        model_meta = get_rife_metadata(model_name)
        
        if model_meta:
            # Enforce single GPU
            gpu_device_str = str(validated.get("gpu_device", ""))
            if gpu_device_str and gpu_device_str not in ("", "cpu"):
                devices = [d.strip() for d in gpu_device_str.replace(" ", "").split(",") if d.strip()]
                if len(devices) > 1:
                    validated["gpu_device"] = devices[0]  # Use first GPU only
                    print(f"WARNING: RIFE preset: Multi-GPU not supported, using single GPU: {devices[0]}")
            
            # Validate FPS multiplier
            fps_mult_str = str(validated.get("fps_multiplier", "x2"))
            try:
                mult_value = int(fps_mult_str.replace("x", "").strip())
                max_mult = model_meta.max_fps_multiplier
                
                if mult_value > max_mult:
                    validated["fps_multiplier"] = f"x{max_mult}"
                    print(f"WARNING: RIFE preset: FPS multiplier {mult_value}x exceeds limit {max_mult}x, clamping")
            except (ValueError, AttributeError):
                pass  # Keep original if parsing fails
            
            # Validate scale factor
            try:
                scale = float(validated.get("scale", 1.0))
                if scale <= 0:
                    validated["scale"] = 1.0
                    print(f"WARNING: RIFE preset: Invalid scale {scale}, resetting to 1.0")
            except (ValueError, TypeError):
                validated["scale"] = 1.0
        
        return validated

    def _migrate_old_preset_structure(self):
        """
        Migrate presets from old structure (per-model subfolders) to new structure (flat per-tab).
        
        Old: presets/<tab>/<model>/<preset>.json
        New: presets/<tab>/<preset>.json
        
        This runs once on startup and moves all presets to the new location.
        """
        try:
            for tab_dir in self.base_dir.iterdir():
                if not tab_dir.is_dir() or tab_dir.name.startswith("."):
                    continue
                
                # Check for model subfolders (old structure)
                for model_dir in tab_dir.iterdir():
                    if not model_dir.is_dir() or model_dir.name == ".last_used":
                        continue
                    
                    # Move preset files from model subfolder to tab folder
                    for preset_file in model_dir.glob("*.json"):
                        target_path = tab_dir / preset_file.name
                        
                        # Only move if target doesn't exist (avoid overwriting newer presets)
                        if not target_path.exists():
                            try:
                                preset_file.rename(target_path)
                                print(f"✅ Migrated preset: {preset_file.name} → {tab_dir.name}/")
                            except Exception as e:
                                print(f"⚠️ Could not migrate {preset_file}: {e}")
                    
                    # Move last_used_preset.txt to .last_used subfolder
                    last_used_file = model_dir / "last_used_preset.txt"
                    if last_used_file.exists():
                        try:
                            last_used_dir = tab_dir / ".last_used"
                            last_used_dir.mkdir(exist_ok=True)
                            target_last_used = last_used_dir / f"{model_dir.name}.txt"
                            if not target_last_used.exists():
                                last_used_file.rename(target_last_used)
                        except Exception:
                            pass  # Non-critical
                    
                    # Remove empty model subfolder
                    try:
                        if not any(model_dir.iterdir()):
                            model_dir.rmdir()
                    except Exception:
                        pass  # Non-critical
        
        except Exception as e:
            print(f"⚠️ Preset migration warning: {e}")
            # Non-fatal - continue even if migration fails

    # ------------------------------------------------------------------ #
    # UNIVERSAL PRESET SYSTEM (v2.0)
    # All tabs save/load from single preset file
    # ------------------------------------------------------------------ #
    
    def _universal_preset_path(self, preset_name: str) -> Path:
        """Get path for a universal preset file (stored directly in presets/)"""
        return self.base_dir / f"{_sanitize_name(preset_name)}.json"
    
    def _last_used_preset_path(self) -> Path:
        """Get path for the last used preset tracker file"""
        return self.base_dir / LAST_USED_PRESET_FILE

    def _invalidate_universal_presets_cache(self) -> None:
        self._universal_presets_cache_ts = 0.0
        self._universal_presets_cache_dir_mtime = 0.0
        self._universal_presets_cache = []
    
    def list_universal_presets(self) -> List[str]:
        """
        List all universal presets (JSON files directly in presets/ folder).
        
        Excludes:
        - global.json (app settings)
        - Subfolders (old per-tab structure)
        - Hidden files
        """
        if not self.base_dir.exists():
            return []

        now = time.time()
        try:
            dir_mtime = float(self.base_dir.stat().st_mtime)
        except Exception:
            dir_mtime = 0.0
        if (
            (now - self._universal_presets_cache_ts) < 5.0
            and dir_mtime == self._universal_presets_cache_dir_mtime
        ):
            return list(self._universal_presets_cache)

        presets = []
        for path in self.base_dir.glob("*.json"):
            if path.name == "global.json":
                continue
            if path.is_file():
                presets.append(path.stem)

        result = sorted(presets)
        self._universal_presets_cache_ts = now
        self._universal_presets_cache_dir_mtime = dir_mtime
        self._universal_presets_cache = list(result)
        return result
    
    def save_universal_preset(self, preset_name: str, data: Dict[str, Any]) -> str:
        """
        Save a universal preset containing all tab settings.
        
        Args:
            preset_name: Name for the preset
            data: Universal preset dict with structure:
                  {"seedvr2": {...}, "gan": {...}, ..., "_meta": {...}}
        
        Returns:
            Sanitized preset name
        """
        if not preset_name or not preset_name.strip():
            raise ValueError("Preset name cannot be empty")
        
        safe_name = _sanitize_name(preset_name.strip())
        preset_path = self._universal_preset_path(safe_name)
        
        # Add/update metadata
        if "_meta" not in data:
            data["_meta"] = {}
        data["_meta"]["version"] = "2.0"
        data["_meta"]["format"] = "universal"
        data["_meta"]["last_modified"] = datetime.now().isoformat()
        if "created_at" not in data["_meta"]:
            data["_meta"]["created_at"] = data["_meta"]["last_modified"]
        
        # Atomic write: write to tmp file then rename
        tmp_path = preset_path.with_suffix(".json.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp_path.replace(preset_path)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        
        # Update last used
        self.set_last_used_universal_preset(safe_name)
        self._invalidate_universal_presets_cache()
        
        return safe_name
    
    def load_universal_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a universal preset.
        
        Args:
            preset_name: Name of the preset to load
        
        Returns:
            Universal preset dict, or None if not found
        """
        if not preset_name:
            return None
        
        preset_path = self._universal_preset_path(preset_name)
        if not preset_path.exists():
            return None
        
        try:
            with preset_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading universal preset {preset_name}: {e}")
            return None
    
    def delete_universal_preset(self, preset_name: str) -> bool:
        """Delete a universal preset."""
        preset_path = self._universal_preset_path(preset_name)
        if preset_path.exists():
            try:
                preset_path.unlink()
                self._invalidate_universal_presets_cache()
                return True
            except Exception:
                return False
        return False
    
    def set_last_used_universal_preset(self, preset_name: str) -> None:
        """
        Save the name of the last used universal preset.
        
        This is stored in presets/.last_used_preset.txt
        """
        path = self._last_used_preset_path()
        try:
            with path.open("w", encoding="utf-8") as f:
                f.write(_sanitize_name(preset_name))
        except Exception as e:
            print(f"⚠️ Could not save last used preset: {e}")
    
    def get_last_used_universal_preset(self) -> Optional[str]:
        """
        Get the name of the last used universal preset.
        
        Returns:
            Preset name, or None if not set
        """
        path = self._last_used_preset_path()
        if not path.exists():
            return None
        
        try:
            name = path.read_text(encoding="utf-8").strip()
            return name if name else None
        except Exception:
            return None
    
    def load_last_used_universal_preset(self) -> Optional[Dict[str, Any]]:
        """
        Load the last used universal preset.
        
        Returns:
            Universal preset dict, or None if not found
        """
        preset_name = self.get_last_used_universal_preset()
        if not preset_name:
            return None
        
        return self.load_universal_preset(preset_name)
    
    def universal_preset_exists(self, preset_name: str) -> bool:
        """Check if a universal preset exists."""
        return self._universal_preset_path(preset_name).exists()
