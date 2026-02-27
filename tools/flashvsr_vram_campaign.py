"""
Run a resumable multi-scenario FlashVSR VRAM campaign.

This wraps `tools/flashvsr_vram_sweep.py` so long campaigns can be resumed
after interruption by re-running the same command.

Usage:
    .\venv\Scripts\python.exe tools\flashvsr_vram_campaign.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from shared.path_utils import get_media_dimensions
from shared.resolution_calculator import estimate_fixed_scale_upscale_plan_from_dims


@dataclass(frozen=True)
class Scenario:
    name: str
    input_path: Path
    scale: int
    max_target_resolution: int


def _safe_print(text: str) -> None:
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        sys.stdout.buffer.write((text + "\n").encode(enc, errors="replace"))
        sys.stdout.flush()


def _default_scenarios(base_dir: Path, scenario_set: str) -> List[Scenario]:
    in960 = base_dir / "450frame960.mp4"
    in1280 = base_dir / "450frame1280.mp4"
    in832 = base_dir / "450frame832.mp4"
    common = [
        Scenario("960_s4_max0", in960, 4, 0),
        Scenario("960_s2_max0", in960, 2, 0),
        Scenario("1280_s4_max3840", in1280, 4, 3840),
        Scenario("1280_s2_max1920", in1280, 2, 1920),
        Scenario("832_s4_max0", in832, 4, 0),
        Scenario("832_s2_max0", in832, 2, 0),
    ]
    if scenario_set == "common":
        return common

    # Wider resolution coverage for sensitivity analysis and capping behavior.
    wide = list(common)
    wide.extend(
        [
            Scenario("960_s4_max3840", in960, 4, 3840),
            Scenario("960_s4_max2160", in960, 4, 2160),
            Scenario("960_s2_max1920", in960, 2, 1920),
            Scenario("960_s2_max1080", in960, 2, 1080),
            Scenario("1280_s4_max2160", in1280, 4, 2160),
            Scenario("1280_s2_max1080", in1280, 2, 1080),
            Scenario("832_s4_max3840", in832, 4, 3840),
            Scenario("832_s4_max2160", in832, 4, 2160),
            Scenario("832_s2_max1920", in832, 2, 1920),
            Scenario("832_s2_max1080", in832, 2, 1080),
        ]
    )
    return wide


def _scenario_effective_signature(scenario: Scenario) -> Tuple[int, int, int, int, int]:
    dims = get_media_dimensions(str(scenario.input_path))
    if not dims:
        raise RuntimeError(f"could not read dimensions for {scenario.input_path}")
    input_w, input_h = int(dims[0]), int(dims[1])
    plan = estimate_fixed_scale_upscale_plan_from_dims(
        input_w,
        input_h,
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


def _dedupe_effective_scenarios(
    scenarios: List[Scenario],
) -> Tuple[List[Scenario], List[Dict[str, object]]]:
    kept: List[Scenario] = []
    skipped: List[Dict[str, object]] = []
    seen: Dict[Tuple[int, int, int, int, int], Scenario] = {}

    for scenario in scenarios:
        signature = _scenario_effective_signature(scenario)
        existing = seen.get(signature)
        if existing is None:
            seen[signature] = scenario
            kept.append(scenario)
            continue
        skipped.append(
            {
                "skipped": scenario.name,
                "kept": existing.name,
                "signature": {
                    "scale": int(signature[0]),
                    "preprocess_resolution": f"{int(signature[1])}x{int(signature[2])}",
                    "output_resolution": f"{int(signature[3])}x{int(signature[4])}",
                },
            }
        )
    return kept, skipped


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


def _run_streaming(cmd: List[str], cwd: Path, env: Dict[str, str]) -> int:
    _safe_print("COMMAND: " + " ".join(f'"{x}"' if " " in x else x for x in cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _safe_print(line.rstrip("\n"))
    return int(proc.wait())


def _parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Resumable FlashVSR VRAM campaign runner")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--mode", type=str, default="full")
    parser.add_argument("--version", type=str, default="1.1")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--vae-model", type=str, default="Wan2.2")
    parser.add_argument("--tile-overlap", type=int, default=48)
    parser.add_argument("--chunk-list", type=str, default="450,384,320,256,224,192,160,128,96,64,48,32")
    parser.add_argument("--keep-models-on-cpu", action="store_true", default=True)
    parser.add_argument("--no-keep-models-on-cpu", action="store_false", dest="keep_models_on_cpu")
    parser.add_argument("--timeout-minutes", type=float, default=5.0)
    parser.add_argument("--stall-seconds", type=float, default=240.0)
    parser.add_argument("--min-effective-fps", type=float, default=0.20)
    parser.add_argument("--shared-low-util-pct", type=float, default=12.0)
    parser.add_argument("--shared-near-full-margin-mb", type=float, default=768.0)
    parser.add_argument("--shared-pressure-seconds", type=float, default=120.0)
    parser.add_argument("--min-shared-check-runtime-sec", type=float, default=90.0)
    parser.add_argument("--accept-timeout-profile", action="store_true", default=True)
    parser.add_argument("--no-accept-timeout-profile", action="store_false", dest="accept_timeout_profile")
    parser.add_argument("--profile-min-elapsed-sec", type=float, default=75.0)
    parser.add_argument("--max-cases-per-scenario", type=int, default=0)
    parser.add_argument("--scenario-set", type=str, choices=["common", "wide"], default="common")
    parser.add_argument(
        "--scenario-filter",
        type=str,
        default="",
        help="Optional comma-separated scenario names to run (exact names).",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument(
        "--records-csv",
        type=str,
        default=str(base_dir / "outputs" / "flashvsr_vram_sweeps" / "flashvsr_vram_records.csv"),
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default=str(base_dir / "outputs" / "flashvsr_vram_sweeps" / "campaign_runs"),
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Optional manifest path (JSON).",
    )
    parser.add_argument(
        "--report-md",
        type=str,
        default=str(base_dir / "outputs" / "flashvsr_vram_sweeps" / "flashvsr_vram_report.md"),
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=str(base_dir / "outputs" / "flashvsr_vram_sweeps" / "flashvsr_vram_report.json"),
    )
    parser.add_argument("--skip-report", action="store_true", default=False)
    parser.add_argument("--dedupe-effective-scenarios", action="store_true", default=True)
    parser.add_argument("--no-dedupe-effective-scenarios", action="store_false", dest="dedupe_effective_scenarios")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base_dir = Path(__file__).resolve().parents[1]
    python_exe = base_dir / "venv" / "Scripts" / "python.exe"
    sweep_script = base_dir / "tools" / "flashvsr_vram_sweep.py"
    report_script = base_dir / "tools" / "flashvsr_vram_report.py"
    records_csv = Path(args.records_csv).resolve()
    run_root = Path(args.run_root).resolve()

    if not python_exe.exists():
        _safe_print(f"ERROR: missing python executable: {python_exe}")
        return 2
    if not sweep_script.exists():
        _safe_print(f"ERROR: missing sweep script: {sweep_script}")
        return 2
    if not report_script.exists():
        _safe_print(f"ERROR: missing report script: {report_script}")
        return 2

    scenarios = _default_scenarios(base_dir, args.scenario_set)
    if str(args.scenario_filter or "").strip():
        allowed = {part.strip() for part in str(args.scenario_filter).split(",") if part.strip()}
        scenarios = [s for s in scenarios if s.name in allowed]
    if not scenarios:
        _safe_print("ERROR: no scenarios selected.")
        return 2
    missing_inputs = [s.input_path for s in scenarios if not s.input_path.exists()]
    if missing_inputs:
        for p in missing_inputs:
            _safe_print(f"ERROR: missing input video: {p}")
        return 2

    deduped_scenarios: List[Dict[str, object]] = []
    if bool(args.dedupe_effective_scenarios):
        scenarios, deduped_scenarios = _dedupe_effective_scenarios(scenarios)
        if deduped_scenarios:
            _safe_print("[INFO] skipping spatially redundant scenarios with matching effective preprocess/output sizes:")
            for item in deduped_scenarios:
                sig = item["signature"]
                _safe_print(
                    "  "
                    f"{item['skipped']} -> {item['kept']} "
                    f"(scale={sig['scale']}, preprocess={sig['preprocess_resolution']}, output={sig['output_resolution']})"
                )
            _safe_print("-" * 72)

    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
    else:
        manifest_path = (
            base_dir
            / "outputs"
            / "flashvsr_vram_sweeps"
            / f"campaign_manifest_{args.scenario_set}_gpu{int(args.gpu_id)}.json"
        )

    manifest = _load_manifest(manifest_path)
    manifest["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["scenario_set"] = str(args.scenario_set)
    manifest["gpu_id"] = int(args.gpu_id)
    manifest["records_csv"] = str(records_csv)
    manifest["guardrails"] = {
        "timeout_minutes": float(args.timeout_minutes),
        "stall_seconds": float(args.stall_seconds),
        "min_effective_fps": float(args.min_effective_fps),
        "accept_timeout_profile": bool(args.accept_timeout_profile),
        "profile_min_elapsed_sec": float(args.profile_min_elapsed_sec),
        "shared_low_util_pct": float(args.shared_low_util_pct),
        "shared_near_full_margin_mb": float(args.shared_near_full_margin_mb),
        "shared_pressure_seconds": float(args.shared_pressure_seconds),
        "min_shared_check_runtime_sec": float(args.min_shared_check_runtime_sec),
    }
    manifest["dedupe_effective_scenarios"] = bool(args.dedupe_effective_scenarios)
    manifest["deduped_scenarios"] = deduped_scenarios
    manifest.setdefault("runs", [])
    _save_manifest(manifest_path, manifest)

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    run_entries = list(manifest.get("runs", []))
    any_fail = False

    _safe_print(f"Campaign scenario set: {args.scenario_set}")
    _safe_print(f"GPU: cuda:{int(args.gpu_id)}")
    _safe_print(f"Records CSV: {records_csv}")
    _safe_print(f"Manifest: {manifest_path}")
    _safe_print(
        "Guards: "
        f"timeout={float(args.timeout_minutes):.1f}m, "
        f"stall={float(args.stall_seconds):.0f}s, "
        f"min_fps={float(args.min_effective_fps):.3f}, "
        f"timeout_profile={bool(args.accept_timeout_profile)} "
        f"(min_elapsed={float(args.profile_min_elapsed_sec):.0f}s), "
        f"shared(low_util<={float(args.shared_low_util_pct):.1f}%, "
        f"margin={float(args.shared_near_full_margin_mb):.0f}MB, "
        f"duration>={float(args.shared_pressure_seconds):.0f}s)"
    )
    _safe_print("-" * 72)

    for scenario in scenarios:
        run_dir = run_root / scenario.name
        run_dir.mkdir(parents=True, exist_ok=True)

        start_ts = time.time()
        entry: Dict[str, object] = {
            "scenario": scenario.name,
            "input_path": str(scenario.input_path),
            "scale": int(scenario.scale),
            "max_target_resolution": int(scenario.max_target_resolution),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "running",
            "run_dir": str(run_dir),
        }
        run_entries.append(entry)
        manifest["runs"] = run_entries
        _save_manifest(manifest_path, manifest)

        cmd = [
            str(python_exe),
            "-u",
            str(sweep_script),
            "--input",
            str(scenario.input_path),
            "--gpu-id",
            str(int(args.gpu_id)),
            "--mode",
            str(args.mode),
            "--version",
            str(args.version),
            "--precision",
            str(args.precision),
            "--vae-model",
            str(args.vae_model),
            "--tile-overlap",
            str(int(args.tile_overlap)),
            "--max-target-resolution",
            str(int(scenario.max_target_resolution)),
            "--chunk-list",
            str(args.chunk_list),
            "--scales",
            str(int(scenario.scale)),
            "--records-csv",
            str(records_csv),
            "--run-dir",
            str(run_dir),
            "--timeout-minutes",
            str(float(args.timeout_minutes)),
            "--stall-seconds",
            str(float(args.stall_seconds)),
            "--min-effective-fps",
            str(float(args.min_effective_fps)),
            "--profile-min-elapsed-sec",
            str(float(args.profile_min_elapsed_sec)),
            "--shared-low-util-pct",
            str(float(args.shared_low_util_pct)),
            "--shared-near-full-margin-mb",
            str(float(args.shared_near_full_margin_mb)),
            "--shared-pressure-seconds",
            str(float(args.shared_pressure_seconds)),
            "--min-shared-check-runtime-sec",
            str(float(args.min_shared_check_runtime_sec)),
        ]
        if bool(args.accept_timeout_profile):
            cmd.append("--accept-timeout-profile")
        else:
            cmd.append("--no-accept-timeout-profile")
        if bool(args.resume):
            cmd.append("--resume")
        else:
            cmd.append("--no-resume")
        if bool(args.keep_models_on_cpu):
            cmd.append("--keep-models-on-cpu")
        else:
            cmd.append("--no-keep-models-on-cpu")
        if int(args.max_cases_per_scenario) > 0:
            cmd.extend(["--max-cases", str(int(args.max_cases_per_scenario))])

        _safe_print(f"[RUN] {scenario.name}")
        rc = _run_streaming(cmd, cwd=base_dir, env=env)
        elapsed = time.time() - start_ts
        entry["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        entry["elapsed_sec"] = round(float(elapsed), 3)
        entry["returncode"] = int(rc)
        entry["status"] = "ok" if rc == 0 else "failed"
        _save_manifest(manifest_path, manifest)

        if rc != 0:
            any_fail = True
            _safe_print(f"[FAIL] {scenario.name} (rc={rc})")
        else:
            _safe_print(f"[DONE] {scenario.name} ({elapsed:.1f}s)")
        _safe_print("-" * 72)

    if not bool(args.skip_report):
        report_cmd = [
            str(python_exe),
            "-u",
            str(report_script),
            "--records-csv",
            str(records_csv),
            "--out-md",
            str(Path(args.report_md).resolve()),
            "--out-json",
            str(Path(args.report_json).resolve()),
        ]
        _safe_print("[RUN] Generating VRAM report")
        report_rc = _run_streaming(report_cmd, cwd=base_dir, env=env)
        if report_rc != 0:
            any_fail = True
            _safe_print(f"[FAIL] report generation (rc={report_rc})")
        else:
            _safe_print("[DONE] report generation")
        _safe_print("-" * 72)

    _safe_print(f"Campaign complete. Manifest: {manifest_path}")
    _safe_print(f"Records CSV: {records_csv}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
