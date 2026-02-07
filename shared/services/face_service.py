from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import gradio as gr

from shared.preset_manager import PresetManager
from shared.path_utils import (
    normalize_path,
    collision_safe_path,
    collision_safe_dir,
    detect_input_type,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from shared.face_restore import restore_image, restore_video


def face_defaults(models: List[str]) -> Dict[str, Any]:
    """Get default face restoration settings"""
    from shared.face_restore import get_available_backends
    available = get_available_backends()
    default_backend = available[0] if available else "auto"
    
    return {
        "model": models[0] if models else "",
        "face_detector": "retinaface",
        "detection_confidence": 0.7,
        "min_face_size": 64,
        "max_faces": 0,
        "restoration_model": "auto",
        "face_strength": 0.5,
        "restore_blindly": False,
        "upscale_faces": False,
        "face_padding": 0.3,
        "use_landmarks": True,
        "color_correction": True,
        "gpu_acceleration": True,
        "batch_faces": True,
        "output_quality": 0.8,
        "preserve_original": True,
        "artifact_reduction": False,
        "save_face_masks": False,
        # Standalone processing (Face tab) - kept in presets so inputs persist like other tabs
        "input_path": "",
        "output_override": "",
        "batch_enable": False,
        "batch_input_path": "",
        "batch_output_path": "",
        "backend": default_backend,  # Keep for backward compatibility with old presets
    }


FACE_ORDER: List[str] = [
    # IMPORTANT: This order MUST match inputs_list in ui/face_tab.py exactly!
    # The UI creates components in this exact sequence (lines 344-349 in face_tab.py)
    "model",                  # 0: model_selector
    "face_detector",          # 1: face_detector (Face Detection tab)
    "detection_confidence",   # 2: detection_confidence
    "min_face_size",          # 3: min_face_size
    "max_faces",              # 4: max_faces
    "restoration_model",      # 5: restoration_model (Restoration tab)
    "face_strength",          # 6: face_strength (Restoration tab)
    "restore_blindly",        # 7: restore_blindly
    "upscale_faces",          # 8: upscale_faces
    "face_padding",           # 9: face_padding (Advanced tab)
    "use_landmarks",          # 10: face_landmarks (UI name differs)
    "color_correction",       # 11: color_correction
    "gpu_acceleration",       # 12: gpu_acceleration
    "batch_faces",            # 13: batch_face_processing (UI name differs)
    "output_quality",         # 14: output_quality (Quality tab)
    "preserve_original",      # 15: preserve_original
    "artifact_reduction",     # 16: artifact_reduction
    "save_face_masks",        # 17: save_face_masks
    # Standalone processing controls (Face tab)
    "input_path",             # 18: input_path (standalone)
    "output_override",        # 19: output_override (standalone)
    "batch_enable",           # 20: batch_enable (standalone)
    "batch_input_path",       # 21: batch_input_path (standalone)
    "batch_output_path",      # 22: batch_output_path (standalone)
]


def _face_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    return dict(zip(FACE_ORDER, args))


def _apply_face_preset(
    preset: Dict[str, Any],
    defaults: Dict[str, Any],
    preset_manager: PresetManager,
    current: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    base = defaults.copy()
    if current:
        base.update(current)
    merged = preset_manager.merge_config(base, preset)
    return [merged[k] for k in FACE_ORDER]


def build_face_callbacks(
    preset_manager: PresetManager,
    global_settings: Dict[str, Any],
    models: List[str],
    shared_state: gr.State = None,
):
    defaults = face_defaults(models)

    def refresh_presets(model_name: str, select_name: Optional[str] = None):
        presets = preset_manager.list_presets("face", model_name)
        last_used = preset_manager.get_last_used_name("face", model_name)
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        return gr.update(choices=presets, value=value)

    def save_preset(preset_name: str, *args):
        if not preset_name.strip():
            return gr.update(), gr.update(value="⚠️ Enter a preset name before saving"), *list(args)

        try:
            payload = _face_dict_from_args(list(args))
            model_name = payload["model"]
            preset_manager.save_preset_safe("face", model_name, preset_name.strip(), payload)
            dropdown = refresh_presets(model_name, select_name=preset_name.strip())

            current_map = dict(zip(FACE_ORDER, list(args)))
            loaded_vals = _apply_face_preset(payload, defaults, preset_manager, current=current_map)

            return dropdown, gr.update(value=f"✅ Saved preset '{preset_name}' for {model_name}"), *loaded_vals
        except Exception as e:
            return gr.update(), gr.update(value=f"❌ Error saving preset: {str(e)}"), *list(args)

    def load_preset(preset_name: str, model_name: str, current_values: List[Any]):
        """
        Load a preset.
        
        FIXED: Now returns (*values, status_message) to match UI output expectations.
        Note: Face tab has special logic - skips model_selector in outputs.
        """
        try:
            model_name = model_name or defaults["model"]
            preset = preset_manager.load_preset_safe("face", model_name, preset_name)
            if preset:
                preset_manager.set_last_used("face", model_name, preset_name)

            defaults_with_model = defaults.copy()
            defaults_with_model["model"] = model_name
            current_map = dict(zip(FACE_ORDER, current_values))
            values = _apply_face_preset(preset or {}, defaults_with_model, preset_manager, current=current_map)

            # Keep global face strength in sync with the face tab's strength (best-effort).
            try:
                strength_idx = FACE_ORDER.index("face_strength")
                global_settings["face_strength"] = float(values[strength_idx])
                preset_manager.save_global_settings(global_settings)
            except Exception:
                pass

            # Return values + status message (status is LAST)
            # Face tab skips model_selector in outputs, so return values[1:] + status
            status_msg = f"✅ Loaded preset '{preset_name}'" if preset else "ℹ️ Preset not found"
            return (*values[1:], gr.update(value=status_msg))
        except Exception as e:
            print(f"Error loading preset {preset_name}: {e}")
            # Return current values + error status (skip first value for model_selector)
            return (*current_values[1:], gr.update(value=f"❌ Error: {str(e)}"))

    def safe_defaults():
        return [defaults[k] for k in FACE_ORDER]

    def set_face_global(val, state=None):
        global_settings["face_global"] = bool(val)
        preset_manager.save_global_settings(global_settings)
        if state:
            state["seed_controls"]["face_strength_val"] = global_settings.get("face_strength", 0.5)
        return gr.update(value="✅ Global face restoration updated"), state or {}

    def cache_strength(strength_val: float, state=None):
        try:
            strength_num = float(strength_val)
        except Exception:
            strength_num = defaults.get("face_strength", 0.5)
        strength_num = max(0.0, min(1.0, strength_num))
        global_settings["face_strength"] = strength_num
        preset_manager.save_global_settings(global_settings)
        if state:
            state["seed_controls"]["face_strength_val"] = strength_num
        return gr.update(value=f"✅ Face strength set to {strength_num}"), state or {}

    # ------------------------------------------------------------------ #
    # Standalone processing (Face tab): restore faces on a single file or batch folder.
    # Uses the SAME `restore_image` / `restore_video` utilities used post-upscale.
    # ------------------------------------------------------------------ #
    def _coerce_backend(model_choice: str) -> Tuple[str, Optional[str]]:
        """
        Returns (backend_for_runtime, warning_message_if_any).
        """
        m = (model_choice or "").strip().lower()
        if not m or m == "auto":
            return "auto", None
        if m in ("gfpgan", "codeformer"):
            return m, None
        # UI may expose future backends; keep safe fallback.
        return "auto", f"⚠️ Unsupported face restoration model '{model_choice}'. Falling back to auto."

    def _ensure_out_dir(path_val: str) -> Path:
        p = Path(normalize_path(path_val)) if path_val else Path.cwd() / "outputs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _build_output_path(out_dir: Path, input_file: Path, suffix_tag: str = "_face_restored") -> Path:
        return collision_safe_path(out_dir / f"{input_file.stem}{suffix_tag}{input_file.suffix}")

    def _collect_batch_files(batch_in: Path) -> List[Path]:
        exts = sorted({*IMAGE_EXTENSIONS, *VIDEO_EXTENSIONS})
        batch_files: List[Path] = []
        if batch_in.is_dir():
            for ext in exts:
                batch_files.extend(batch_in.glob(f"**/*{ext}"))
        elif batch_in.is_file() and batch_in.suffix.lower() in exts:
            batch_files = [batch_in]
        return sorted(batch_files)

    def run_action(
        uploaded_file,
        *args,
        state: Dict[str, Any] = None,
        progress=None,
        global_settings_snapshot: Dict[str, Any] | None = None,
        _global_settings: Dict[str, Any] = global_settings,
    ):
        """
        Face restoration run entrypoint for the Face tab.

        Inputs:
        - uploaded_file: gr.File(type="filepath") value (optional)
        - *args: values matching FACE_ORDER (inputs_list in ui/face_tab.py)
        - state: shared_state dict (last argument from UI)

        Outputs are handled in ui/face_tab.py.
        """
        global_settings = (
            dict(global_settings_snapshot)
            if isinstance(global_settings_snapshot, dict)
            else dict(_global_settings)
        )
        state = state or {}
        settings = _face_dict_from_args(list(args))

        logs: List[str] = []

        # Resolve runtime settings
        backend, backend_warn = _coerce_backend(str(settings.get("restoration_model", "auto")))
        if backend_warn:
            logs.append(backend_warn)

        try:
            strength = float(settings.get("face_strength", defaults.get("face_strength", 0.5)))
        except Exception:
            strength = float(defaults.get("face_strength", 0.5))
        strength = max(0.0, min(1.0, strength))

        use_gpu = bool(settings.get("gpu_acceleration", True))

        # Output dirs
        out_dir = _ensure_out_dir(global_settings.get("output_dir") or "")

        # Standalone input resolution: upload wins, then textbox
        manual_in = str(settings.get("input_path") or "").strip()
        in_path = uploaded_file or manual_in

        # Batch mode
        if bool(settings.get("batch_enable")):
            batch_in_str = str(settings.get("batch_input_path") or "").strip()
            if not batch_in_str:
                return (
                    gr.update(value="❌ Batch enabled but no batch input path set."),
                    gr.update(value=""),
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                    gr.update(value=None),
                    state,
                )

            batch_in = Path(normalize_path(batch_in_str))
            if not batch_in.exists():
                return (
                    gr.update(value=f"❌ Batch input path not found: {batch_in}"),
                    gr.update(value=""),
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                    gr.update(value=None),
                    state,
                )

            batch_out_str = str(settings.get("batch_output_path") or "").strip()
            batch_out_dir = _ensure_out_dir(batch_out_str) if batch_out_str else out_dir

            files = _collect_batch_files(batch_in)
            if not files:
                return (
                    gr.update(value=f"⚠️ No supported media found in: {batch_in}"),
                    gr.update(value=""),
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                    gr.update(value=None),
                    state,
                )

            logs.append(f"📦 Batch: {len(files)} file(s)")
            logs.append(f"🧠 Model: {backend} | strength={strength:g} | gpu={'on' if use_gpu else 'off'}")
            logs.append(f"📤 Output dir: {batch_out_dir}")

            output_paths: List[str] = []
            last_img: Optional[str] = None
            last_vid: Optional[str] = None

            for idx, fp in enumerate(files, start=1):
                itype = detect_input_type(str(fp))
                out_path = _build_output_path(batch_out_dir, fp)
                try:
                    if itype == "image":
                        restored = restore_image(str(fp), strength=strength, backend=backend, use_gpu=use_gpu, output_path=str(out_path))
                        last_img = restored or None
                        output_paths.append(str(restored or out_path))
                    elif itype == "video":
                        restored = restore_video(str(fp), strength=strength, backend=backend, use_gpu=use_gpu, output_path=str(out_path))
                        last_vid = restored or None
                        output_paths.append(str(restored or out_path))
                    else:
                        logs.append(f"⚠️ Skipped unsupported: {fp.name}")
                        continue
                    logs.append(f"[{idx}/{len(files)}] ✅ {fp.name} → {Path(output_paths[-1]).name}")
                except Exception as e:
                    logs.append(f"[{idx}/{len(files)}] ❌ {fp.name}: {str(e)}")

            # Update shared state pointers (best-effort)
            try:
                state.setdefault("seed_controls", {})
                state["seed_controls"]["last_output_dir"] = str(batch_out_dir)
                state["seed_controls"]["last_output_path"] = output_paths[-1] if output_paths else None
                state["seed_controls"]["last_input_path"] = str(batch_in)
            except Exception:
                pass

            # Batch output as a downloadable multi-file list (gr.File output)
            return (
                gr.update(value=f"✅ Batch complete: {len(output_paths)} output(s)"),
                gr.update(value="\n".join(logs)),
                gr.update(value=last_img, visible=bool(last_img)),
                gr.update(value=last_vid, visible=bool(last_vid)),
                gr.update(value=output_paths if output_paths else None),
                state,
            )

        # Single mode
        if not in_path or not str(in_path).strip():
            return (
                gr.update(value="⚠️ No input set. Upload a file or enter an input path."),
                gr.update(value=""),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None),
                state,
            )

        resolved = normalize_path(str(in_path))
        itype = detect_input_type(resolved)
        in_file = Path(resolved)
        if not in_file.exists():
            return (
                gr.update(value=f"❌ Input path not found: {resolved}"),
                gr.update(value=""),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None),
                state,
            )

        override = str(settings.get("output_override") or "").strip()
        if override:
            ov = Path(normalize_path(override))
            if ov.suffix == "":
                ov.mkdir(parents=True, exist_ok=True)
                out_path = _build_output_path(ov, in_file)
            else:
                ov.parent.mkdir(parents=True, exist_ok=True)
                out_path = collision_safe_path(ov)
        else:
            out_path = _build_output_path(out_dir, in_file)

        logs.append(f"🧠 Model: {backend} | strength={strength:g} | gpu={'on' if use_gpu else 'off'}")
        logs.append(f"📥 Input: {resolved}")
        logs.append(f"📤 Output: {out_path}")

        out_img: Optional[str] = None
        out_vid: Optional[str] = None
        if itype == "image":
            restored = restore_image(resolved, strength=strength, backend=backend, use_gpu=use_gpu, output_path=str(out_path))
            out_img = restored or str(out_path)
        elif itype == "video":
            restored = restore_video(resolved, strength=strength, backend=backend, use_gpu=use_gpu, output_path=str(out_path))
            out_vid = restored or str(out_path)
        else:
            return (
                gr.update(value=f"❌ Unsupported input type for face restoration: {resolved}"),
                gr.update(value="\n".join(logs)),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None),
                state,
            )

        try:
            state.setdefault("seed_controls", {})
            state["seed_controls"]["last_output_dir"] = str(out_path.parent)
            state["seed_controls"]["last_output_path"] = out_img or out_vid
            state["seed_controls"]["last_input_path"] = resolved
        except Exception:
            pass

        return (
            gr.update(value="✅ Face restoration complete"),
            gr.update(value="\n".join(logs)),
            gr.update(value=out_img, visible=bool(out_img)),
            gr.update(value=out_vid, visible=bool(out_vid)),
            gr.update(value=None),
            state,
        )

    return {
        "defaults": defaults,
        "order": FACE_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "set_face_global": lambda *args: set_face_global(*args[:-1], args[-1]) if len(args) > 1 else set_face_global(args[0]),
        "cache_strength": lambda *args: cache_strength(*args[:-1], args[-1]) if len(args) > 1 else cache_strength(args[0]),
        "run_action": run_action,
    }

