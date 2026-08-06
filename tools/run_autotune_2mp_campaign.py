"""Run the requested 2 MP, 4x auto-tune profiling campaign.

This is an operator tool, not an application entry point. It calls the same
service callbacks as Gradio with fresh-install defaults, while an opt-in
environment flag extends the search toward a 4 GB measured peak. The normal
2 GB free-VRAM cutoff remains active, so a probe crossing 30 GB on a 32 GB GPU
is terminated and logged instead of spilling into shared memory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


# The model CLIs emit Unicode progress markers. Keep the operator process and
# every child pipe on UTF-8 so a Windows legacy console cannot stop log draining.
os.environ["PYTHONIOENCODING"] = "utf-8"
for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from shared.logging_utils import RunLogger
from shared.path_utils import get_media_dimensions
from shared.preset_manager import PresetManager
from shared.runner import Runner
from shared.services.flashvsr_service import (
    FLASHVSR_ORDER,
    build_flashvsr_callbacks,
)
from shared.services.seedvr2_service import (
    SEEDVR2_ORDER,
    build_seedvr2_callbacks,
)
from shared.services.sparkvsr_service import (
    SPARKVSR_ORDER,
    build_sparkvsr_callbacks,
)


PLAN_PATH = BASE_DIR / "tools" / "autotune_campaign_plan.json"
CAMPAIGN_INPUT_DIR = BASE_DIR / "temp" / "autotune_2mp_sources"
VRAM_DIR = BASE_DIR / "vram_usages"
MANIFEST_PATH = VRAM_DIR / "autotune_2mp_campaign_manifest.json"
PROFILE_MIN_PEAK_GB = 4.0


CONFIGURATIONS: List[Dict[str, Any]] = [
    {
        "engine": "seedvr2",
        "name": "seedvr2_7b_sharp_fp16",
        "model": "seedvr2_ema_7b_sharp_fp16.safetensors",
        "overrides": {
            "dit_model": "seedvr2_ema_7b_sharp_fp16.safetensors",
            "int8_convrot": False,
        },
    },
    {
        "engine": "seedvr2",
        "name": "seedvr2_3b_fp16",
        "model": "seedvr2_ema_3b_fp16.safetensors",
        "overrides": {
            "dit_model": "seedvr2_ema_3b_fp16.safetensors",
            "int8_convrot": False,
        },
    },
    {
        "engine": "seedvr2",
        "name": "seedvr2_7b_sharp_int8_convrot",
        "model": "seedvr2_ema_7b_sharp_fp16.safetensors",
        "overrides": {
            "dit_model": "seedvr2_ema_7b_sharp_fp16.safetensors",
            "int8_convrot": True,
        },
    },
    {
        "engine": "seedvr2",
        "name": "seedvr2_3b_int8_convrot",
        "model": "seedvr2_ema_3b_fp16.safetensors",
        "overrides": {
            "dit_model": "seedvr2_ema_3b_fp16.safetensors",
            "int8_convrot": True,
        },
    },
    {
        "engine": "flashvsr",
        "name": "flashvsrplus_1_1_bf16",
        "model": "FlashVSR+ 1.1",
        "overrides": {"version": "1.1", "precision": "bf16"},
    },
    {
        "engine": "flashvsr",
        "name": "flashvsrplus_1_1_int8_convrot",
        "model": "FlashVSR+ 1.1",
        "overrides": {"version": "1.1", "precision": "int8_convrot"},
    },
    {
        "engine": "sparkvsr",
        "name": "sparkvsr_bf16",
        "model": "SparkVSR-bf16",
        "overrides": {"model_name": "SparkVSR-bf16"},
    },
    {
        "engine": "sparkvsr",
        "name": "sparkvsr_int8_convrot",
        "model": "SparkVSR-int8-convrot",
        "overrides": {"model_name": "SparkVSR-int8-convrot"},
    },
]


def _print(message: str) -> None:
    print(str(message), flush=True)


def _load_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _prepare_inputs(plan: Dict[str, Any]) -> None:
    CAMPAIGN_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for clip in plan.get("clips", []):
        source = Path(str(clip["source"]))
        output = CAMPAIGN_INPUT_DIR / f"{clip['name']}.mp4"
        expected = (int(clip["campaign_input_width"]), int(clip["campaign_input_height"]))
        if not source.exists():
            raise FileNotFoundError(f"Missing source clip: {source}")
        if output.exists() and get_media_dimensions(str(output)) == expected:
            continue
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"scale={expected[0]}:{expected[1]}:flags=lanczos",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "16",
            "-an",
            str(output),
        ]
        _print(f"Preparing {clip['name']} at {expected[0]}x{expected[1]}")
        subprocess.run(cmd, check=True, cwd=str(BASE_DIR))


def _build_runtime(gpu_id: int) -> Tuple[Runner, Dict[str, Any], Dict[str, Dict[str, Any]]]:
    temp_dir = BASE_DIR / "temp"
    output_dir = BASE_DIR / "outputs"
    global_settings = {
        "global_gpu_device": str(gpu_id),
        "temp_dir": str(temp_dir),
        "output_dir": str(output_dir),
        "telemetry": True,
        "mode": "subprocess",
    }
    runner = Runner(BASE_DIR, temp_dir=temp_dir, output_dir=output_dir, telemetry_enabled=True)
    runner.set_mode("subprocess")
    preset_manager = PresetManager(BASE_DIR / "presets")
    run_logger = RunLogger(enabled=False)
    callbacks = {
        "seedvr2": build_seedvr2_callbacks(
            preset_manager,
            runner,
            run_logger,
            global_settings,
            None,
            output_dir,
            temp_dir,
        ),
        "flashvsr": build_flashvsr_callbacks(
            preset_manager,
            runner,
            run_logger,
            global_settings,
            None,
            BASE_DIR,
            temp_dir,
            output_dir,
        ),
        "sparkvsr": build_sparkvsr_callbacks(
            preset_manager,
            runner,
            run_logger,
            global_settings,
            None,
            BASE_DIR,
            temp_dir,
            output_dir,
        ),
    }
    return runner, global_settings, callbacks


def _case_state(gpu_id: int) -> Dict[str, Any]:
    return {
        "operation_status": "ready",
        "seed_controls": {
            "global_gpu_device_val": str(gpu_id),
            "global_settings": {"global_gpu_device": str(gpu_id)},
        },
    }


def _log_prefix(engine: str) -> str:
    return {
        "seedvr2": "seedvr2_autotune_",
        "flashvsr": "flashvsrplus_autotune_",
        "sparkvsr": "sparkvsr_autotune_",
    }[engine]


def _ordered_settings(callback: Dict[str, Any], config: Dict[str, Any], input_path: Path) -> List[Any]:
    settings = dict(callback["defaults"])
    settings.update(dict(config.get("overrides") or {}))
    settings["input_path"] = str(input_path)
    expected_order = {
        "seedvr2": SEEDVR2_ORDER,
        "flashvsr": FLASHVSR_ORDER,
        "sparkvsr": SPARKVSR_ORDER,
    }[str(config["engine"])]
    if list(callback["order"]) != list(expected_order):
        raise RuntimeError(f"Callback order mismatch for {config['engine']}")
    return [settings[key] for key in expected_order]


def _run_generator(generator: Iterable[Any]) -> str:
    last_status = ""
    for update in generator:
        status = ""
        if isinstance(update, (tuple, list)) and update:
            status = str(update[0] or "").strip()
        if status and status != last_status:
            _print(f"  {status}")
            last_status = status
    return last_status


def _find_created_log(engine: str, before: set[Path], started_at: float) -> Path:
    prefix = _log_prefix(engine)
    current = set(VRAM_DIR.glob(f"{prefix}*.json"))
    new_paths = [path for path in current - before if path.is_file()]
    if not new_paths:
        new_paths = [
            path
            for path in current
            if path.is_file() and path.stat().st_mtime >= started_at - 1.0
        ]
    if not new_paths:
        raise RuntimeError(f"No {engine} auto-tune log was created")
    return max(new_paths, key=lambda path: path.stat().st_mtime)


def _annotate_log(
    log_path: Path,
    *,
    case_id: str,
    config: Dict[str, Any],
    clip: Dict[str, Any],
    campaign_input: Path,
    plan: Dict[str, Any],
) -> Path:
    payload = _load_json(log_path, {})
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid auto-tune log: {log_path}")
    payload["campaign"] = {
        "id": "autotune_2mp_4x_20260806",
        "case_id": case_id,
        "engine": config["engine"],
        "configuration": config["name"],
        "model": config["model"],
        "overrides_from_fresh_defaults": dict(config.get("overrides") or {}),
        "source_video": str(clip["source"]),
        "campaign_input": str(campaign_input),
        "campaign_input_width": int(clip["campaign_input_width"]),
        "campaign_input_height": int(clip["campaign_input_height"]),
        "planned_target_width": int(clip["target_width"]),
        "planned_target_height": int(clip["target_height"]),
        "planned_target_pixels": int(clip["target_width"]) * int(clip["target_height"]),
        "scale": int(plan["scale"]),
        "max_long_edge": int(plan["max_long_edge"]),
        "max_target_pixels": int(plan["max_target_pixels"]),
        "profile_min_peak_vram_gb": PROFILE_MIN_PEAK_GB,
        "over_30gb_probe_cutoff_enabled": True,
        "safe_probe_early_success_stops_disabled": True,
        "search_continues_after_first_safe_result": True,
        "fresh_measurements_forced": True,
    }
    _write_json(log_path, payload)
    suffix = f"__{case_id}.json"
    if not log_path.name.endswith(suffix):
        renamed = log_path.with_name(f"{log_path.stem}{suffix}")
        if renamed.exists():
            renamed = log_path.with_name(f"{log_path.stem}_{int(time.time())}{suffix}")
        log_path.replace(renamed)
        log_path = renamed
    return log_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ordered 2 MP auto-tune campaign")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--engines",
        default="seedvr2,flashvsr,sparkvsr",
        help="Comma-separated ordered subset of seedvr2,flashvsr,sparkvsr",
    )
    parser.add_argument("--case-filter", default="", help="Optional exact case-id substring")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    plan = _load_json(PLAN_PATH, {})
    if not isinstance(plan, dict) or not isinstance(plan.get("clips"), list):
        raise RuntimeError(f"Invalid campaign plan: {PLAN_PATH}")
    _prepare_inputs(plan)
    VRAM_DIR.mkdir(parents=True, exist_ok=True)

    allowed_engines = [part.strip() for part in str(args.engines).split(",") if part.strip()]
    unknown = [name for name in allowed_engines if name not in {"seedvr2", "flashvsr", "sparkvsr"}]
    if unknown:
        raise ValueError(f"Unknown engines: {unknown}")

    os.environ["SECOURSES_AUTOTUNE_PROFILE_MIN_VRAM_GB"] = str(PROFILE_MIN_PEAK_GB)
    os.environ["SECOURSES_GLOBAL_GPU_DEVICE"] = str(int(args.gpu_id))
    os.environ["PYTHONUNBUFFERED"] = "1"

    _runner, global_settings, callbacks = _build_runtime(int(args.gpu_id))
    manifest = _load_json(MANIFEST_PATH, {})
    if not isinstance(manifest, dict):
        manifest = {}
    manifest.update(
        {
            "campaign_id": "autotune_2mp_4x_20260806",
            "plan_path": str(PLAN_PATH),
            "gpu_id": int(args.gpu_id),
            "profile_min_peak_vram_gb": PROFILE_MIN_PEAK_GB,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    entries = manifest.setdefault("cases", {})
    if not isinstance(entries, dict):
        entries = {}
        manifest["cases"] = entries

    any_failed = False
    for engine in allowed_engines:
        for config in [item for item in CONFIGURATIONS if item["engine"] == engine]:
            for clip in plan["clips"]:
                case_id = f"{config['name']}__{clip['name']}"
                if str(args.case_filter) and str(args.case_filter) not in case_id:
                    continue
                prior = entries.get(case_id)
                if bool(args.resume) and isinstance(prior, dict) and prior.get("status") == "completed":
                    prior_log = Path(str(prior.get("log_path") or ""))
                    if prior_log.exists():
                        _print(f"[SKIP] {case_id}")
                        continue

                campaign_input = CAMPAIGN_INPUT_DIR / f"{clip['name']}.mp4"
                before = set(VRAM_DIR.glob(f"{_log_prefix(engine)}*.json"))
                started_at = time.time()
                entries[case_id] = {
                    "status": "running",
                    "engine": engine,
                    "configuration": config["name"],
                    "clip": clip["name"],
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                _write_json(MANIFEST_PATH, manifest)
                _print(f"[RUN ] {case_id}")
                try:
                    state = _case_state(int(args.gpu_id))
                    values = _ordered_settings(callbacks[engine], config, campaign_input)
                    generator = callbacks[engine]["auto_tune_action"](
                        str(campaign_input),
                        *values,
                        state=state,
                        progress=None,
                        global_settings_snapshot=global_settings,
                    )
                    final_status = _run_generator(generator)
                    created_log = _find_created_log(engine, before, started_at)
                    created_log = _annotate_log(
                        created_log,
                        case_id=case_id,
                        config=config,
                        clip=clip,
                        campaign_input=campaign_input,
                        plan=plan,
                    )
                    log_payload = _load_json(created_log, {})
                    completed = bool(
                        isinstance(log_payload, dict)
                        and bool(log_payload.get("finalized", False))
                        and str(log_payload.get("status") or "") in {"completed", "threshold_reached"}
                    )
                    entries[case_id].update(
                        {
                            "status": "completed" if completed else "failed",
                            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "elapsed_sec": round(time.time() - started_at, 3),
                            "final_status": final_status,
                            "log_path": str(created_log),
                            "test_count": len(log_payload.get("tests") or []) if isinstance(log_payload, dict) else 0,
                        }
                    )
                    if not completed:
                        any_failed = True
                        _print(f"[FAIL] {case_id}: log did not finalize successfully")
                    else:
                        _print(f"[DONE] {case_id} -> {created_log.name}")
                except Exception as exc:
                    any_failed = True
                    entries[case_id].update(
                        {
                            "status": "failed",
                            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "elapsed_sec": round(time.time() - started_at, 3),
                            "error": str(exc),
                        }
                    )
                    _print(f"[FAIL] {case_id}: {exc}")
                finally:
                    manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    _write_json(MANIFEST_PATH, manifest)

    _print(f"Campaign manifest: {MANIFEST_PATH}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
