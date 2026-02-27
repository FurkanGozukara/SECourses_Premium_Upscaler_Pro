"""
Validate optimizer-picked FlashVSR cases for unique effective scenarios.

This script deduplicates scenarios by effective preprocess/output resolution,
computes the current optimizer recommendation for each unique scenario, and
either reuses an equivalent validated CSV row or runs a bounded 5-minute
validation case and appends the result to the sweep CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from shared.flashvsr_optimizer import FlashVSROptimizedSettings, optimize_flashvsr_settings
from shared.path_utils import get_media_dimensions
from shared.resolution_calculator import estimate_fixed_scale_upscale_plan_from_dims
from tools.flashvsr_vram_campaign import Scenario, _default_scenarios
from tools.flashvsr_vram_sweep import (
    _append_csv_row,
    _build_case_id,
    _ensure_csv,
    _prepare_effective_input,
    _run_case,
)


def _safe_print(text: str) -> None:
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        sys.stdout.buffer.write((text + "\n").encode(enc, errors="replace"))
        sys.stdout.flush()


def _to_bool(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate optimizer-picked FlashVSR cases")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--mode", type=str, default="full")
    parser.add_argument("--version", type=str, default="1.1")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--vae-model", type=str, default="Wan2.2")
    parser.add_argument("--keep-models-on-cpu", action="store_true", default=True)
    parser.add_argument("--no-keep-models-on-cpu", action="store_false", dest="keep_models_on_cpu")
    parser.add_argument("--reserve-vram-gb", type=float, default=2.0)
    parser.add_argument("--scenario-set", type=str, choices=["common", "wide"], default="common")
    parser.add_argument(
        "--scenario-manifest",
        type=str,
        default="",
        help="Optional JSON scenario list: {\"scenarios\": [{name,input_path,scale,max_target_resolution}, ...]}",
    )
    parser.add_argument("--scenario-filter", type=str, default="")
    parser.add_argument("--max-scenarios", type=int, default=0, help="0 = unlimited")
    parser.add_argument(
        "--exclude-signature-from-calibration",
        action="store_true",
        default=False,
        help="Exclude rows with the same effective signature during per-scenario prediction.",
    )
    parser.add_argument("--timeout-minutes", type=float, default=5.0)
    parser.add_argument("--stall-seconds", type=float, default=120.0)
    parser.add_argument("--min-effective-fps", type=float, default=0.20)
    parser.add_argument("--shared-low-util-pct", type=float, default=12.0)
    parser.add_argument("--shared-near-full-margin-mb", type=float, default=768.0)
    parser.add_argument("--shared-pressure-seconds", type=float, default=120.0)
    parser.add_argument("--min-shared-check-runtime-sec", type=float, default=90.0)
    parser.add_argument("--accept-timeout-profile", action="store_true", default=True)
    parser.add_argument("--no-accept-timeout-profile", action="store_false", dest="accept_timeout_profile")
    parser.add_argument("--profile-min-elapsed-sec", type=float, default=75.0)
    parser.add_argument(
        "--records-csv",
        type=str,
        default=str(BASE_DIR / "outputs" / "flashvsr_vram_sweeps" / "flashvsr_vram_records.csv"),
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default=str(BASE_DIR / "outputs" / "flashvsr_vram_sweeps" / "optimizer_validation_cases"),
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(BASE_DIR / "outputs" / "flashvsr_vram_sweeps" / "optimizer_validation_manifest.json"),
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default=str(BASE_DIR / "outputs" / "flashvsr_vram_sweeps" / "optimizer_validation_summary.md"),
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(BASE_DIR / "outputs" / "flashvsr_vram_sweeps" / "optimizer_validation_summary.json"),
    )
    return parser.parse_args()


def _load_manifest(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"runs": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"runs": []}


def _save_manifest(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _upsert_manifest_run(manifest: Dict[str, object], entry: Dict[str, object]) -> List[Dict[str, object]]:
    runs = [run for run in manifest.get("runs", []) if isinstance(run, dict)]
    primary_name = str(entry.get("primary_name") or "")
    kept = [run for run in runs if str(run.get("primary_name") or "") != primary_name]
    kept.append(entry)
    manifest["runs"] = kept
    return kept


def _effective_signature(scenario: Scenario) -> Tuple[int, int, int, int, int]:
    dims = get_media_dimensions(str(scenario.input_path))
    if not dims:
        raise RuntimeError(f"could not read dimensions for {scenario.input_path}")
    plan = estimate_fixed_scale_upscale_plan_from_dims(
        int(dims[0]),
        int(dims[1]),
        requested_scale=float(scenario.scale),
        model_scale=int(scenario.scale),
        max_edge=int(scenario.max_target_resolution),
        force_pre_downscale=True,
    )
    return (
        int(scenario.scale),
        int(plan.preprocess_width),
        int(plan.preprocess_height),
        int(plan.resize_width),
        int(plan.resize_height),
    )


def _load_manifest_scenarios(manifest_path: Path) -> List[Scenario]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to read scenario manifest {manifest_path}: {exc}") from exc

    raw_items = payload.get("scenarios") if isinstance(payload, dict) else payload
    if not isinstance(raw_items, list):
        raise RuntimeError("scenario manifest must be a list or an object with 'scenarios' list")

    scenarios: List[Scenario] = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or f"custom_{idx+1}")
        input_path_raw = str(item.get("input_path") or "").strip()
        if not input_path_raw:
            continue
        try:
            scale = 2 if int(float(item.get("scale", 4))) <= 2 else 4
        except Exception:
            scale = 4
        try:
            max_edge = max(0, int(float(item.get("max_target_resolution", 0))))
        except Exception:
            max_edge = 0
        scenarios.append(
            Scenario(
                name=name,
                input_path=Path(input_path_raw).resolve(),
                scale=int(scale),
                max_target_resolution=int(max_edge),
            )
        )
    return scenarios


def _collect_unique_scenarios(args: argparse.Namespace) -> List[Dict[str, object]]:
    if str(args.scenario_manifest or "").strip():
        scenarios = _load_manifest_scenarios(Path(str(args.scenario_manifest)).resolve())
    else:
        scenarios = _default_scenarios(BASE_DIR, str(args.scenario_set))
    if str(args.scenario_filter or "").strip():
        allowed = {part.strip() for part in str(args.scenario_filter).split(",") if part.strip()}
        scenarios = [s for s in scenarios if s.name in allowed]

    unique: Dict[Tuple[int, int, int, int, int], Dict[str, object]] = {}
    for scenario in scenarios:
        signature = _effective_signature(scenario)
        current = unique.get(signature)
        if current is None:
            unique[signature] = {
                "primary_name": scenario.name,
                "input_path": str(scenario.input_path),
                "requested_scale": int(scenario.scale),
                "requested_max_target_resolution": int(scenario.max_target_resolution),
                "effective_signature": {
                    "scale": int(signature[0]),
                    "preprocess_resolution": f"{int(signature[1])}x{int(signature[2])}",
                    "output_resolution": f"{int(signature[3])}x{int(signature[4])}",
                },
                "aliases": [],
            }
            continue
        current["aliases"].append(scenario.name)

    out = list(unique.values())
    out.sort(
        key=lambda item: (
            int(item["requested_scale"]),
            int(item["requested_max_target_resolution"]),
            str(item["primary_name"]),
        )
    )
    return out


def _load_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            return [row for row in reader if isinstance(row, dict)]
    except Exception:
        return []


def _matches_effective_signature(row: Dict[str, str], signature: Dict[str, object]) -> bool:
    try:
        pre = str(signature.get("preprocess_resolution") or "")
        out = str(signature.get("output_resolution") or "")
        if "x" not in pre or "x" not in out:
            return False
        pre_w_text, pre_h_text = pre.split("x", 1)
        out_w_text, out_h_text = out.split("x", 1)
        return (
            _to_int(row.get("scale"), 0) == int(signature.get("scale", 0))
            and _to_int(row.get("preprocess_width"), -1) == int(pre_w_text)
            and _to_int(row.get("preprocess_height"), -1) == int(pre_h_text)
            and _to_int(row.get("output_width"), -1) == int(out_w_text)
            and _to_int(row.get("output_height"), -1) == int(out_h_text)
        )
    except Exception:
        return False


def _build_holdout_calibration_csv(
    *,
    source_csv: Path,
    signature: Dict[str, object],
    out_csv: Path,
) -> Tuple[int, int]:
    total_rows = 0
    kept_rows = 0
    with source_csv.open("r", encoding="utf-8", newline="") as in_fp:
        reader = csv.DictReader(in_fp)
        fieldnames = list(reader.fieldnames or [])
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as out_fp:
            writer = csv.DictWriter(out_fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                if not isinstance(row, dict):
                    continue
                total_rows += 1
                if _matches_effective_signature(row, signature):
                    continue
                writer.writerow(row)
                kept_rows += 1
    return total_rows, kept_rows


def _find_equivalent_validated_case(
    rows: List[Dict[str, str]],
    *,
    mode: str,
    precision: str,
    vae_model: str,
    gpu_id: int,
    keep_models_on_cpu: bool,
    optimized: FlashVSROptimizedSettings,
) -> Optional[Dict[str, str]]:
    matches: List[Dict[str, str]] = []
    for row in rows:
        if str(row.get("mode") or "").strip().lower() != str(mode).strip().lower():
            continue
        if str(row.get("precision") or "").strip().lower() != str(precision).strip().lower():
            continue
        if str(row.get("vae_model") or "").strip().lower() != str(vae_model).strip().lower():
            continue
        if _to_int(row.get("gpu_id"), -1) != int(gpu_id):
            continue
        if _to_int(row.get("scale"), 0) != int(optimized.scale):
            continue
        if _to_int(row.get("max_target_resolution"), -1) != int(optimized.max_target_resolution):
            continue
        if _to_int(row.get("preprocess_width"), -1) != int(optimized.preprocess_width):
            continue
        if _to_int(row.get("preprocess_height"), -1) != int(optimized.preprocess_height):
            continue
        if _to_int(row.get("output_width"), -1) != int(optimized.output_width):
            continue
        if _to_int(row.get("output_height"), -1) != int(optimized.output_height):
            continue
        if _to_int(row.get("tile_size"), -1) != int(optimized.tile_size):
            continue
        if _to_int(row.get("overlap"), -1) != int(optimized.overlap):
            continue
        if _to_int(row.get("frame_chunk_size"), -1) != int(optimized.frame_chunk_size):
            continue
        if _to_bool(row.get("keep_models_on_cpu")) != bool(keep_models_on_cpu):
            continue
        if not _to_bool(row.get("tiled_dit")) or _to_bool(row.get("tiled_vae")) or _to_bool(row.get("stream_decode")):
            continue
        if not _to_bool(row.get("effective_success")):
            continue
        if _to_bool(row.get("shared_vram_suspect")) or _to_bool(row.get("oom_recovery_override")):
            continue
        matches.append(row)

    if not matches:
        return None
    matches.sort(
        key=lambda row: (
            _to_float(row.get("peak_vram_gb"), 0.0),
            _to_bool(row.get("profile_partial")),
            str(row.get("timestamp") or ""),
        )
    )
    return matches[-1]


def _append_validation_row(
    *,
    csv_path: Path,
    input_path: Path,
    input_dims: Tuple[int, int],
    optimized: FlashVSROptimizedSettings,
    args: argparse.Namespace,
    result,
) -> None:
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "case_id": _build_case_id(optimized.scale, optimized.frame_chunk_size, optimized.tile_size, optimized.overlap),
        "input_path": str(input_path),
        "input_width": int(input_dims[0]),
        "input_height": int(input_dims[1]),
        "preprocess_width": int(optimized.preprocess_width),
        "preprocess_height": int(optimized.preprocess_height),
        "output_width": int(optimized.output_width),
        "output_height": int(optimized.output_height),
        "mode": str(args.mode),
        "precision": str(args.precision),
        "vae_model": str(args.vae_model),
        "scale": int(optimized.scale),
        "max_target_resolution": int(optimized.max_target_resolution),
        "tile_size": int(optimized.tile_size),
        "overlap": int(optimized.overlap),
        "frame_chunk_size": int(optimized.frame_chunk_size),
        "keep_models_on_cpu": bool(args.keep_models_on_cpu),
        "tiled_dit": True,
        "tiled_vae": False,
        "stream_decode": False,
        "gpu_id": int(args.gpu_id),
        "gpu_total_gb": float(result.gpu_total_gb),
        "peak_vram_gb": float(result.peak_vram_gb),
        "success": bool(result.success),
        "oom": bool(result.oom),
        "returncode": int(result.returncode),
        "elapsed_sec": float(result.elapsed_sec),
        "output_file": str(result.output_file),
        "log_file": str(result.log_file),
        "effective_success": bool(result.effective_success),
        "raw_success": bool(result.raw_success),
        "profile_partial": bool(result.profile_partial),
        "shared_vram_suspect": bool(result.shared_vram_suspect),
        "oom_recovery_override": bool(result.oom_recovery_override),
        "failure_reason": str(result.failure_reason),
        "processing_fps": float(result.processing_fps),
        "peak_vram_cli_gb": float(result.peak_vram_cli_gb),
        "prepared_input_path": str(result.prepared_input_path),
    }
    _append_csv_row(csv_path, row)


def _entry_accuracy_metrics(entry: Dict[str, object]) -> Dict[str, Optional[float]]:
    optimized = entry.get("optimized", {})
    if not isinstance(optimized, dict):
        return {
            "actual_peak_vram_gb": None,
            "target_headroom_gb": None,
            "estimated_error_gb": None,
            "guarded_error_gb": None,
            "actual_over_target_gb": None,
            "actual_over_guarded_gb": None,
        }

    actual = entry.get("actual_peak_vram_gb")
    if actual is None:
        return {
            "actual_peak_vram_gb": None,
            "target_headroom_gb": None,
            "estimated_error_gb": None,
            "guarded_error_gb": None,
            "actual_over_target_gb": None,
            "actual_over_guarded_gb": None,
        }

    actual_f = float(actual)
    estimated = float(optimized.get("estimated_peak_vram_gb") or 0.0)
    guarded = float(optimized.get("estimated_guarded_vram_gb") or 0.0)
    target = float(optimized.get("target_vram_gb") or 0.0)
    return {
        "actual_peak_vram_gb": actual_f,
        "target_headroom_gb": float(target - actual_f),
        "estimated_error_gb": float(actual_f - estimated),
        "guarded_error_gb": float(actual_f - guarded),
        "actual_over_target_gb": max(0.0, float(actual_f - target)),
        "actual_over_guarded_gb": max(0.0, float(actual_f - guarded)),
    }


def _summarize_accuracy(results: List[Dict[str, object]]) -> Dict[str, float]:
    metrics = [_entry_accuracy_metrics(item) for item in results]
    validated = [item for item in metrics if item.get("actual_peak_vram_gb") is not None]
    if not validated:
        return {
            "validated_count": 0,
            "median_abs_estimated_error_gb": 0.0,
            "median_abs_guarded_error_gb": 0.0,
            "max_actual_over_guard_gb": 0.0,
            "max_actual_over_target_gb": 0.0,
            "guard_failures": 0,
            "target_failures": 0,
        }

    est_abs = sorted(abs(float(item["estimated_error_gb"])) for item in validated)
    guard_abs = sorted(abs(float(item["guarded_error_gb"])) for item in validated)
    over_guard = [float(item["actual_over_guarded_gb"]) for item in validated]
    over_target = [float(item["actual_over_target_gb"]) for item in validated]
    mid = len(validated) // 2

    def _median(sorted_vals: List[float]) -> float:
        if not sorted_vals:
            return 0.0
        if len(sorted_vals) % 2 == 1:
            return float(sorted_vals[mid])
        return float((sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0)

    return {
        "validated_count": int(len(validated)),
        "median_abs_estimated_error_gb": float(_median(est_abs)),
        "median_abs_guarded_error_gb": float(_median(guard_abs)),
        "max_actual_over_guard_gb": float(max(over_guard) if over_guard else 0.0),
        "max_actual_over_target_gb": float(max(over_target) if over_target else 0.0),
        "guard_failures": int(sum(1 for v in over_guard if v > 1e-9)),
        "target_failures": int(sum(1 for v in over_target if v > 1e-9)),
    }


def _build_markdown(
    results: List[Dict[str, object]],
    *,
    records_csv: Path,
    reserve_vram_gb: float,
) -> str:
    accuracy = _summarize_accuracy(results)
    holdout_mode = any(bool(item.get("calibration_holdout_enabled")) for item in results)
    lines: List[str] = []
    lines.append("# FlashVSR Optimizer Validation Summary")
    lines.append("")
    lines.append(f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- Source CSV: `{records_csv}`")
    lines.append(f"- GPU reserve target: `{float(reserve_vram_gb):.2f} GB`")
    lines.append(f"- Unique scenarios checked: `{len(results)}`")
    lines.append(
        f"- Calibration holdout mode: `{'ON' if holdout_mode else 'OFF'}` "
        "(exclude same effective signature during prediction)"
    )
    lines.append(f"- Validated scenarios with actual peak data: `{int(accuracy['validated_count'])}`")
    lines.append(f"- Median absolute estimate error: `{float(accuracy['median_abs_estimated_error_gb']):.3f} GB`")
    lines.append(f"- Guard overruns: `{int(accuracy['guard_failures'])}` (max `{float(accuracy['max_actual_over_guard_gb']):.3f} GB`)")  # noqa: E501
    lines.append(f"- Target overruns: `{int(accuracy['target_failures'])}` (max `{float(accuracy['max_actual_over_target_gb']):.3f} GB`)")  # noqa: E501
    lines.append("")
    lines.append("| Scenario | Aliases | Optimized Settings | Estimated / Guarded / Target (GB) | Actual Peak (GB) | Headroom (GB) | Actual-Est (GB) | Actual-Guard (GB) | Holdout Removed | Source | Status |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    for item in results:
        optimized = item["optimized"]
        metrics = _entry_accuracy_metrics(item)
        aliases = ", ".join(item["aliases"]) if item["aliases"] else "-"
        setting_text = (
            f"s={optimized['scale']}x, max={optimized['max_target_resolution']}, "
            f"chunk={optimized['frame_chunk_size']}, tile={optimized['tile_size']}, overlap={optimized['overlap']}"
        )
        actual_peak = metrics.get("actual_peak_vram_gb")
        actual_text = "-" if actual_peak is None else f"{float(actual_peak):.3f}"
        headroom_text = "-" if metrics.get("target_headroom_gb") is None else f"{float(metrics['target_headroom_gb']):.3f}"
        est_delta_text = "-" if metrics.get("estimated_error_gb") is None else f"{float(metrics['estimated_error_gb']):.3f}"
        guard_delta_text = "-" if metrics.get("guarded_error_gb") is None else f"{float(metrics['guarded_error_gb']):.3f}"
        holdout_removed = item.get("calibration_rows_removed")
        holdout_text = "-" if holdout_removed is None else str(int(holdout_removed))
        lines.append(
            "| "
            f"{item['primary_name']} | "
            f"{aliases} | "
            f"{setting_text} | "
            f"{float(optimized['estimated_peak_vram_gb']):.3f} / "
            f"{float(optimized['estimated_guarded_vram_gb']):.3f} / "
            f"{float(optimized['target_vram_gb']):.3f} | "
            f"{actual_text} | "
            f"{headroom_text} | "
            f"{est_delta_text} | "
            f"{guard_delta_text} | "
            f"{holdout_text} | "
            f"{item['source']} | "
            f"{item['status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    records_csv = Path(args.records_csv).resolve()
    run_root = Path(args.run_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    out_md = Path(args.out_md).resolve()
    out_json = Path(args.out_json).resolve()
    python_exe = BASE_DIR / "venv" / "Scripts" / "python.exe"
    cli_path = BASE_DIR / "ComfyUI-FlashVSR_Stable" / "cli_main.py"
    models_dir = BASE_DIR / "ComfyUI-FlashVSR_Stable" / "models"

    if not python_exe.exists():
        _safe_print(f"ERROR: missing venv python: {python_exe}")
        return 2
    if not cli_path.exists():
        _safe_print(f"ERROR: missing CLI: {cli_path}")
        return 2
    if not models_dir.exists():
        _safe_print(f"ERROR: missing models dir: {models_dir}")
        return 2

    _ensure_csv(records_csv)
    manifest = _load_manifest(manifest_path)
    manifest["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["records_csv"] = str(records_csv)
    manifest["reserve_vram_gb"] = float(args.reserve_vram_gb)
    manifest["scenario_set"] = str(args.scenario_set)
    manifest["scenario_manifest"] = str(args.scenario_manifest or "")
    manifest["exclude_signature_from_calibration"] = bool(args.exclude_signature_from_calibration)
    manifest.setdefault("runs", [])

    scenarios = _collect_unique_scenarios(args)
    if not scenarios:
        _safe_print("ERROR: no scenarios selected.")
        return 2

    if int(args.max_scenarios) > 0:
        scenarios = scenarios[: int(args.max_scenarios)]

    existing_rows = _load_rows(records_csv)
    prepared_cache: Dict[Tuple[str, int, int], object] = {}
    summaries: List[Dict[str, object]] = [run for run in manifest.get("runs", []) if isinstance(run, dict)]

    _safe_print(f"Unique scenarios: {len(scenarios)}")
    _safe_print(f"Records CSV: {records_csv}")
    _safe_print(f"Run root: {run_root}")
    _safe_print(
        "Calibration holdout mode: "
        + ("ON (excluding same effective signature per scenario)" if bool(args.exclude_signature_from_calibration) else "OFF")
    )
    _safe_print("-" * 72)

    for scenario in scenarios:
        input_path = Path(str(scenario["input_path"])).resolve()
        dims = get_media_dimensions(str(input_path))
        if not dims:
            _safe_print(f"[SKIP] failed to read dimensions: {input_path}")
            continue

        calibration_csv_for_optimization = str(records_csv)
        calibration_rows_removed = None
        if bool(args.exclude_signature_from_calibration):
            holdout_dir = run_root / "_calibration_holdout"
            holdout_csv = holdout_dir / f"{str(scenario['primary_name'])}_holdout.csv"
            total_rows, kept_rows = _build_holdout_calibration_csv(
                source_csv=records_csv,
                signature=dict(scenario["effective_signature"]),
                out_csv=holdout_csv,
            )
            calibration_rows_removed = max(0, int(total_rows - kept_rows))
            calibration_csv_for_optimization = str(holdout_csv)

        optimized = optimize_flashvsr_settings(
            input_path=str(input_path),
            requested_scale=int(scenario["requested_scale"]),
            mode=str(args.mode),
            precision=str(args.precision),
            vae_model=str(args.vae_model),
            keep_models_on_cpu=bool(args.keep_models_on_cpu),
            stream_decode=False,
            selected_gpu_value=str(int(args.gpu_id)),
            max_target_resolution=int(scenario["requested_max_target_resolution"]),
            pre_downscale_then_upscale=True,
            reserve_vram_gb=float(args.reserve_vram_gb),
            records_csv_path=calibration_csv_for_optimization,
        )

        entry: Dict[str, object] = {
            "primary_name": scenario["primary_name"],
            "aliases": list(scenario["aliases"]),
            "input_path": str(input_path),
            "requested_scale": int(scenario["requested_scale"]),
            "requested_max_target_resolution": int(scenario["requested_max_target_resolution"]),
            "effective_signature": dict(scenario["effective_signature"]),
            "optimized": {
                "scale": int(optimized.scale),
                "max_target_resolution": int(optimized.max_target_resolution),
                "tile_size": int(optimized.tile_size),
                "overlap": int(optimized.overlap),
                "frame_chunk_size": int(optimized.frame_chunk_size),
                "estimated_peak_vram_gb": float(optimized.estimated_peak_vram_gb),
                "estimated_guarded_vram_gb": float(optimized.estimated_guarded_vram_gb),
                "target_vram_gb": float(optimized.budget.target_vram_gb),
                "stage_label": str(optimized.stage_label),
            },
            "notes": list(optimized.notes),
            "source": "optimizer",
            "status": "pending",
            "calibration_holdout_enabled": bool(args.exclude_signature_from_calibration),
        }
        if calibration_rows_removed is not None:
            entry["calibration_rows_removed"] = int(calibration_rows_removed)
        summaries = _upsert_manifest_run(manifest, entry)
        _save_manifest(manifest_path, manifest)

        if optimized.stage_label in {"invalid_input", "missing_dimensions", "cpu_mode"}:
            entry["status"] = optimized.stage_label
            summaries = _upsert_manifest_run(manifest, entry)
            _save_manifest(manifest_path, manifest)
            _safe_print(f"[SKIP] {scenario['primary_name']} -> {optimized.stage_label}")
            continue

        reused = _find_equivalent_validated_case(
            existing_rows,
            mode=str(args.mode),
            precision=str(args.precision),
            vae_model=str(args.vae_model),
            gpu_id=int(args.gpu_id),
            keep_models_on_cpu=bool(args.keep_models_on_cpu),
            optimized=optimized,
        )
        if reused is not None:
            entry["source"] = "existing_csv"
            entry["status"] = "validated_existing"
            entry["actual_peak_vram_gb"] = float(_to_float(reused.get("peak_vram_gb"), 0.0))
            entry["profile_partial"] = bool(_to_bool(reused.get("profile_partial")))
            entry["log_file"] = str(reused.get("log_file") or "")
            entry["output_file"] = str(reused.get("output_file") or "")
            summaries = _upsert_manifest_run(manifest, entry)
            _save_manifest(manifest_path, manifest)
            _safe_print(
                f"[OK ] {scenario['primary_name']} reused existing validated row "
                f"peak={float(entry['actual_peak_vram_gb']):.2f}GB"
            )
            continue

        run_dir = run_root / str(scenario["primary_name"])
        run_dir.mkdir(parents=True, exist_ok=True)
        prepared = _prepare_effective_input(
            input_path=input_path,
            run_dir=run_dir,
            input_width=int(dims[0]),
            input_height=int(dims[1]),
            scale=int(optimized.scale),
            max_target_resolution=int(optimized.max_target_resolution),
            cache=prepared_cache,
        )

        _safe_print(
            f"[RUN ] {scenario['primary_name']} -> "
            f"scale={optimized.scale}x chunk={optimized.frame_chunk_size} "
            f"tile={optimized.tile_size} overlap={optimized.overlap} "
            f"preprocess={optimized.preprocess_width}x{optimized.preprocess_height} "
            f"output={optimized.output_width}x{optimized.output_height}"
        )

        result = _run_case(
            python_exe=python_exe,
            cli_path=cli_path,
            models_dir=models_dir,
            run_dir=run_dir,
            input_path=str(input_path),
            prepared_input_path=str(prepared.effective_input_path),
            mode=str(args.mode),
            precision=str(args.precision),
            vae_model=str(args.vae_model),
            scale=int(optimized.scale),
            max_target_resolution=int(optimized.max_target_resolution),
            tile_size=int(optimized.tile_size),
            overlap=int(optimized.overlap),
            frame_chunk_size=int(optimized.frame_chunk_size),
            gpu_id=int(args.gpu_id),
            version_ui=str(args.version),
            keep_models_on_cpu=bool(args.keep_models_on_cpu),
            timeout_minutes=float(args.timeout_minutes),
            stall_seconds=float(args.stall_seconds),
            min_effective_fps=float(args.min_effective_fps),
            shared_low_util_pct=float(args.shared_low_util_pct),
            shared_near_full_margin_mb=float(args.shared_near_full_margin_mb),
            shared_pressure_seconds=float(args.shared_pressure_seconds),
            min_shared_check_runtime_sec=float(args.min_shared_check_runtime_sec),
            accept_timeout_profile=bool(args.accept_timeout_profile),
            profile_min_elapsed_sec=float(args.profile_min_elapsed_sec),
        )

        _append_validation_row(
            csv_path=records_csv,
            input_path=input_path,
            input_dims=(int(dims[0]), int(dims[1])),
            optimized=optimized,
            args=args,
            result=result,
        )
        existing_rows = _load_rows(records_csv)

        entry["source"] = "validation_run"
        entry["status"] = "validated_run" if bool(result.effective_success) else "validation_failed"
        entry["actual_peak_vram_gb"] = float(result.peak_vram_gb)
        entry["profile_partial"] = bool(result.profile_partial)
        entry["failure_reason"] = str(result.failure_reason)
        entry["log_file"] = str(result.log_file)
        entry["output_file"] = str(result.output_file)
        summaries = _upsert_manifest_run(manifest, entry)
        _save_manifest(manifest_path, manifest)
        _safe_print(
            f"[DONE] {scenario['primary_name']} "
            f"peak={float(result.peak_vram_gb):.2f}GB status={entry['status']} "
            f"reason={str(result.failure_reason)}"
        )
        _safe_print("-" * 72)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "records_csv": str(records_csv),
        "reserve_vram_gb": float(args.reserve_vram_gb),
        "scenario_manifest": str(args.scenario_manifest or ""),
        "exclude_signature_from_calibration": bool(args.exclude_signature_from_calibration),
        "accuracy_summary": _summarize_accuracy(summaries),
        "results": summaries,
    }

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        _build_markdown(summaries, records_csv=records_csv, reserve_vram_gb=float(args.reserve_vram_gb)),
        encoding="utf-8",
    )
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _save_manifest(manifest_path, manifest)

    _safe_print(f"Validation summary written: {out_md}")
    _safe_print(f"Validation JSON written: {out_json}")
    _safe_print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
