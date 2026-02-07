import os
import platform
import subprocess
from pathlib import Path

import gradio as gr

from shared.preset_manager import PresetManager


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


def _update_runtime_env(global_settings: dict):
    models_dir = str(global_settings.get("models_dir") or "").strip()
    hf_home = str(global_settings.get("hf_home") or "").strip()
    transformers_cache = str(global_settings.get("transformers_cache") or "").strip()

    if models_dir:
        os.environ["MODELS_DIR"] = models_dir
    if hf_home:
        os.environ["HF_HOME"] = hf_home
    if transformers_cache:
        os.environ["TRANSFORMERS_CACHE"] = transformers_cache


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
    telemetry_enabled: bool,
    face_global: bool,
    face_strength: float,
    queue_enabled: bool,
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
    pinned_ref = seed_controls.get("pinned_reference_path") or global_settings.get("pinned_reference_path")

    old_models = str(global_settings.get("models_dir") or "")
    old_hf = str(global_settings.get("hf_home") or "")
    old_trans = str(global_settings.get("transformers_cache") or "")

    output_dir = _normalize_path(output_dir_val, global_settings.get("output_dir"))
    temp_dir = _normalize_path(temp_dir_val, global_settings.get("temp_dir"))
    models_dir = _normalize_path(models_dir_val, global_settings.get("models_dir"))
    hf_home = _normalize_path(hf_home_val, global_settings.get("hf_home"))
    transformers_cache = _normalize_path(transformers_cache_val, global_settings.get("transformers_cache"))

    mode_requested = _normalize_mode(mode_choice)
    try:
        runner.set_mode(mode_requested)
        actual_mode = runner.get_mode()
    except Exception:
        runner.set_mode("subprocess")
        actual_mode = "subprocess"

    global_settings.update(
        {
            "output_dir": output_dir,
            "temp_dir": temp_dir,
            "telemetry": bool(telemetry_enabled),
            "face_global": bool(face_global),
            "face_strength": float(face_strength),
            "queue_enabled": bool(queue_enabled),
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
    preset_manager.save_global_settings(global_settings)

    if isinstance(state, dict):
        state.setdefault("seed_controls", {})
        state["seed_controls"]["face_strength_val"] = float(face_strength)
        state["seed_controls"]["queue_enabled_val"] = bool(queue_enabled)

    changed_restart_keys: list[str] = []
    if models_dir != old_models:
        changed_restart_keys.append("MODELS_DIR")
    if hf_home != old_hf:
        changed_restart_keys.append("HF_HOME")
    if transformers_cache != old_trans:
        changed_restart_keys.append("TRANSFORMERS_CACHE")

    status = (
        "Applied immediately and saved to global config.\n"
        f"Active mode: {actual_mode}"
        f"{_build_restart_note(changed_restart_keys)}"
    )
    return gr.update(value=status), gr.update(value=actual_mode), state


def save_global_settings(
    output_dir_val: str,
    temp_dir_val: str,
    telemetry_enabled: bool,
    face_global: bool,
    face_strength: float,
    queue_enabled: bool,
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
        telemetry_enabled=telemetry_enabled,
        face_global=face_global,
        face_strength=face_strength,
        queue_enabled=queue_enabled,
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
        telemetry_enabled=bool(global_settings.get("telemetry", True)),
        face_global=bool(global_settings.get("face_global", False)),
        face_strength=float(global_settings.get("face_strength", 0.5)),
        queue_enabled=bool(global_settings.get("queue_enabled", True)),
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
