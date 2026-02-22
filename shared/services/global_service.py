import os
import platform
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from shared.preset_manager import PresetManager
from shared.gpu_utils import (
    describe_gpu_selection,
    resolve_global_gpu_device,
)


def _normalize_mode(mode_choice: str) -> str:
    mode = str(mode_choice or "subprocess").strip().lower()
    if mode not in ("subprocess", "in_app"):
        return "subprocess"
    return mode


def _normalize_path(value: str, fallback: str) -> str:
    raw = str(value or "").strip()
    if raw:
        return raw
    return str(fallback or "").strip()


def _normalize_theme_mode(value: str, fallback: str = "dark") -> str:
    mode = str(value or fallback or "dark").strip().lower()
    if mode not in {"dark", "light"}:
        mode = str(fallback or "dark").strip().lower()
    return mode if mode in {"dark", "light"} else "dark"


def _normalize_directory_path(value: str, fallback: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalize user-provided directory path robustly across Windows/Linux:
    - accepts '/' and '\\' separators
    - supports env vars and '~'
    - requires absolute path (full path)
    """
    # Keep backwards compatibility for missing values from older presets,
    # but require explicit non-empty input when the field is present.
    if value is None:
        raw = _normalize_path(value, fallback)
    else:
        raw = str(value).strip()
    if not raw:
        return None, "Path is empty. Please enter a full absolute path."

    expanded = os.path.expanduser(os.path.expandvars(raw))
    if os.name == "nt":
        # Keep UNC semantics and normalize slash style for Windows.
        if expanded.startswith("//"):
            expanded = "\\\\" + expanded.lstrip("/\\")
        expanded = expanded.replace("/", "\\")
    else:
        # Treat Windows-style separators as path separators on Unix too.
        expanded = expanded.replace("\\", "/")

    path_obj = Path(expanded)
    if not path_obj.is_absolute():
        return None, f"Path must be absolute: {raw}"

    try:
        normalized = str(path_obj.resolve())
    except Exception:
        normalized = str(path_obj)

    return normalized, None


def _ensure_writable_dir(path_value: str, label: str) -> Optional[str]:
    try:
        target = Path(path_value)
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".secourses_write_test"
        with probe.open("w", encoding="utf-8") as f:
            f.write("ok")
        probe.unlink(missing_ok=True)
        return None
    except Exception as exc:
        return f"{label} is not writable: {exc}"


def _update_runtime_env(global_settings: dict):
    models_dir = str(global_settings.get("models_dir") or "").strip()
    hf_home = str(global_settings.get("hf_home") or "").strip()
    transformers_cache = str(global_settings.get("transformers_cache") or "").strip()
    global_gpu_device = resolve_global_gpu_device(global_settings.get("global_gpu_device"))

    if models_dir:
        os.environ["MODELS_DIR"] = models_dir
    if hf_home:
        os.environ["HF_HOME"] = hf_home
    if transformers_cache:
        os.environ["TRANSFORMERS_CACHE"] = transformers_cache
    os.environ["SECOURSES_GLOBAL_GPU_DEVICE"] = global_gpu_device


def _build_restart_note(changed_keys: list[str]) -> str:
    if not changed_keys:
        return ""
    keys = ", ".join(changed_keys)
    return (
        "\n\nRestart note:\n"
        "The following settings are saved now but are guaranteed across all toolchains only after app restart:\n"
        f"- {keys}"
    )


def apply_global_settings_live(
    output_dir_val: str,
    temp_dir_val: str,
    theme_mode_val: str,
    telemetry_enabled: bool,
    face_strength: float,
    queue_enabled: bool,
    global_gpu_device_val: str,
    mode_choice: str,
    models_dir_val: str,
    hf_home_val: str,
    transformers_cache_val: str,
    runner,
    preset_manager: PresetManager,
    global_settings: dict,
    run_logger=None,
    state: dict | None = None,
):
    """
    Apply global settings immediately and persist them.
    """
    state = state or {}
    seed_controls = state.get("seed_controls", {}) if isinstance(state, dict) else {}
    seed_controls = seed_controls if isinstance(seed_controls, dict) else {}
    state_global_settings = seed_controls.get("global_settings", {})
    state_global_settings = state_global_settings if isinstance(state_global_settings, dict) else {}
    pinned_ref = state_global_settings.get(
        "pinned_reference_path",
        seed_controls.get("pinned_reference_path") or global_settings.get("pinned_reference_path"),
    )
    face_global_enabled = bool(state_global_settings.get("face_global", global_settings.get("face_global", False)))

    old_models = str(global_settings.get("models_dir") or "")
    old_hf = str(global_settings.get("hf_home") or "")
    old_trans = str(global_settings.get("transformers_cache") or "")

    output_dir, output_dir_err = _normalize_directory_path(output_dir_val, global_settings.get("output_dir"))
    temp_dir, temp_dir_err = _normalize_directory_path(temp_dir_val, global_settings.get("temp_dir"))
    theme_mode = _normalize_theme_mode(theme_mode_val, str(global_settings.get("theme_mode", "dark")))
    models_dir = _normalize_path(models_dir_val, global_settings.get("models_dir"))
    hf_home = _normalize_path(hf_home_val, global_settings.get("hf_home"))
    transformers_cache = _normalize_path(transformers_cache_val, global_settings.get("transformers_cache"))
    global_gpu_device = resolve_global_gpu_device(
        global_gpu_device_val if global_gpu_device_val is not None else global_settings.get("global_gpu_device")
    )

    mode_requested = _normalize_mode(mode_choice)
    try:
        runner.set_mode(mode_requested)
        actual_mode = runner.get_mode()
    except Exception:
        runner.set_mode("subprocess")
        actual_mode = "subprocess"

    path_errors: list[str] = []
    if output_dir_err:
        path_errors.append(f"Output Directory: {output_dir_err}")
    if temp_dir_err:
        path_errors.append(f"Temp Directory: {temp_dir_err}")
    if output_dir:
        writable_err = _ensure_writable_dir(output_dir, "Output Directory")
        if writable_err:
            path_errors.append(writable_err)
    if temp_dir:
        writable_err = _ensure_writable_dir(temp_dir, "Temp Directory")
        if writable_err:
            path_errors.append(writable_err)

    if path_errors:
        status = "Settings not applied due to path validation errors:\n- " + "\n- ".join(path_errors)
        if isinstance(state, dict):
            state.setdefault("seed_controls", {})
            seed_controls = state["seed_controls"] if isinstance(state["seed_controls"], dict) else {}
            state["seed_controls"] = seed_controls
            current_global = seed_controls.get("global_settings", {})
            current_global = dict(current_global) if isinstance(current_global, dict) else {}
            current_global["output_dir"] = str(global_settings.get("output_dir", ""))
            current_global["temp_dir"] = str(global_settings.get("temp_dir", ""))
            current_global["theme_mode"] = str(global_settings.get("theme_mode", "dark") or "dark")
            seed_controls["global_settings"] = current_global
        return gr.update(value=status), gr.update(value=actual_mode), state

    global_settings.update(
        {
            "output_dir": output_dir,
            "temp_dir": temp_dir,
            "theme_mode": theme_mode,
            "telemetry": bool(telemetry_enabled),
            # Face global on/off remains managed from Face tab, but still persists in unified presets.
            "face_global": face_global_enabled,
            "face_strength": float(face_strength),
            "queue_enabled": bool(queue_enabled),
            "global_gpu_device": global_gpu_device,
            "mode": actual_mode,
            "pinned_reference_path": pinned_ref,
            "models_dir": models_dir,
            "hf_home": hf_home,
            "transformers_cache": transformers_cache,
        }
    )

    runner.temp_dir = Path(temp_dir)
    runner.output_dir = Path(output_dir)
    runner.set_telemetry(bool(telemetry_enabled))
    if run_logger is not None:
        try:
            run_logger.enabled = bool(telemetry_enabled)
        except Exception:
            pass

    _update_runtime_env(global_settings)

    if isinstance(state, dict):
        state.setdefault("seed_controls", {})
        seed_controls = state["seed_controls"] if isinstance(state["seed_controls"], dict) else {}
        state["seed_controls"] = seed_controls
        seed_controls["face_strength_val"] = float(face_strength)
        seed_controls["queue_enabled_val"] = bool(queue_enabled)
        seed_controls["theme_mode_val"] = theme_mode
        seed_controls["global_gpu_device_val"] = global_gpu_device
        seed_controls["global_rife_cuda_device_val"] = "" if global_gpu_device == "cpu" else global_gpu_device
        seed_controls["pinned_reference_path"] = pinned_ref
        seed_controls["global_settings"] = {
            "output_dir": output_dir,
            "temp_dir": temp_dir,
            "theme_mode": theme_mode,
            "telemetry": bool(telemetry_enabled),
            "face_global": face_global_enabled,
            "face_strength": float(face_strength),
            "queue_enabled": bool(queue_enabled),
            "global_gpu_device": global_gpu_device,
            "mode": actual_mode,
            "models_dir": models_dir,
            "hf_home": hf_home,
            "transformers_cache": transformers_cache,
            "pinned_reference_path": pinned_ref,
        }

    changed_restart_keys: list[str] = []
    if models_dir != old_models:
        changed_restart_keys.append("MODELS_DIR")
    if hf_home != old_hf:
        changed_restart_keys.append("HF_HOME")
    if transformers_cache != old_trans:
        changed_restart_keys.append("TRANSFORMERS_CACHE")

    status = (
        "Applied immediately.\n"
        "Save a Universal Preset to persist across restarts.\n"
        f"Theme: {theme_mode}\n"
        f"Active mode: {actual_mode}\n"
        f"Global GPU: {describe_gpu_selection(global_gpu_device)}"
        f"{_build_restart_note(changed_restart_keys)}"
    )
    return gr.update(value=status), gr.update(value=actual_mode), state


def save_global_settings(
    output_dir_val: str,
    temp_dir_val: str,
    theme_mode_val: str,
    telemetry_enabled: bool,
    face_strength: float,
    queue_enabled: bool,
    global_gpu_device_val: str,
    models_dir_val: str,
    hf_home_val: str,
    transformers_cache_val: str,
    runner,
    preset_manager: PresetManager,
    global_settings: dict,
    run_logger=None,
    state: dict | None = None,
):
    """
    Backward-compatible wrapper.
    """
    return apply_global_settings_live(
        output_dir_val=output_dir_val,
        temp_dir_val=temp_dir_val,
        theme_mode_val=theme_mode_val,
        telemetry_enabled=telemetry_enabled,
        face_strength=face_strength,
        queue_enabled=queue_enabled,
        global_gpu_device_val=global_gpu_device_val,
        mode_choice=str(global_settings.get("mode", "subprocess")),
        models_dir_val=models_dir_val,
        hf_home_val=hf_home_val,
        transformers_cache_val=transformers_cache_val,
        runner=runner,
        preset_manager=preset_manager,
        global_settings=global_settings,
        run_logger=run_logger,
        state=state,
    )


def apply_mode_selection(
    mode_choice: str,
    confirm: bool,
    runner,
    preset_manager: PresetManager,
    global_settings: dict,
    state: dict | None = None,
):
    """
    Backward-compatible immediate mode apply.
    """
    _ = confirm  # retained for compatibility
    status_upd, mode_upd, state = apply_global_settings_live(
        output_dir_val=str(global_settings.get("output_dir", "")),
        temp_dir_val=str(global_settings.get("temp_dir", "")),
        theme_mode_val=str(global_settings.get("theme_mode", "dark") or "dark"),
        telemetry_enabled=bool(global_settings.get("telemetry", True)),
        face_strength=float(global_settings.get("face_strength", 0.5)),
        queue_enabled=bool(global_settings.get("queue_enabled", True)),
        global_gpu_device_val=str(global_settings.get("global_gpu_device", "")),
        mode_choice=mode_choice,
        models_dir_val=str(global_settings.get("models_dir", "")),
        hf_home_val=str(global_settings.get("hf_home", "")),
        transformers_cache_val=str(global_settings.get("transformers_cache", "")),
        runner=runner,
        preset_manager=preset_manager,
        global_settings=global_settings,
        run_logger=None,
        state=state,
    )
    return mode_upd, gr.update(value=False), status_upd, state


def open_outputs_folder(path: str):
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    try:
        if platform.system() == "Windows":
            os.startfile(path_obj)  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.run(["open", str(path_obj)])
        else:
            subprocess.run(["xdg-open", str(path_obj)])
        return gr.update(value=f"Opened: {path_obj}")
    except Exception as exc:
        return gr.update(value=f"Could not open folder: {exc}")


def clear_temp_folder(path: str, confirm: bool = False):
    if not confirm:
        return gr.update(value="Enable 'Confirm delete' before clearing temp.")
    target = Path(path)
    if target.exists():
        for child in target.iterdir():
            if child.is_file():
                child.unlink(missing_ok=True)
            else:
                import shutil

                shutil.rmtree(child, ignore_errors=True)
    return gr.update(value=f"Temp cleared at {target}")
