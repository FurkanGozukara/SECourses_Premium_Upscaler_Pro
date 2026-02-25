"""
Systematic FlashVSR VRAM sweep with resume support.

Default target:
- input: 450frame960.mp4
- scales: 4x and 2x (common output sizes: 3840x2160 and 1920x1080)
- overlap: 48
- tile search: maximize tile size under OOM limit for each chunk size

Usage:
    .\venv\Scripts\python.exe tools\flashvsr_vram_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from shared.resolution_calculator import estimate_fixed_scale_upscale_plan_from_dims

TILES_ASC: Tuple[int, ...] = tuple(range(128, 1025, 32))
CHUNK_SIZES: Tuple[int, ...] = (450, 384, 320, 256, 224, 192, 160, 128, 96, 64, 48, 32)
CSV_FIELDS: Tuple[str, ...] = (
    "timestamp",
    "case_id",
    "input_path",
    "input_width",
    "input_height",
    "preprocess_width",
    "preprocess_height",
    "output_width",
    "output_height",
    "mode",
    "precision",
    "vae_model",
    "scale",
    "max_target_resolution",
    "tile_size",
    "overlap",
    "frame_chunk_size",
    "keep_models_on_cpu",
    "tiled_dit",
    "tiled_vae",
    "stream_decode",
    "gpu_id",
    "gpu_total_gb",
    "peak_vram_gb",
    "success",
    "oom",
    "returncode",
    "elapsed_sec",
    "output_file",
    "log_file",
)


@dataclass(frozen=True)
class CaseKey:
    input_path: str
    mode: str
    precision: str
    vae_model: str
    scale: int
    max_target_resolution: int
    tile_size: int
    overlap: int
    frame_chunk_size: int
    gpu_id: int


@dataclass
class CaseResult:
    success: bool
    oom: bool
    returncode: int
    elapsed_sec: float
    peak_vram_gb: float
    gpu_total_gb: float
    log_file: str
    output_file: str


def _safe_print(text: str) -> None:
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        sys.stdout.buffer.write((text + "\n").encode(enc, errors="replace"))
        sys.stdout.flush()


def _run_nvidia_smi_query() -> Optional[str]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        if proc.returncode != 0:
            return None
        out = (proc.stdout or "").strip()
        return out or None
    except Exception:
        return None


def _query_gpu_memory_mb(gpu_id: int) -> Tuple[Optional[float], Optional[float]]:
    out = _run_nvidia_smi_query()
    if not out:
        return None, None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            if idx != int(gpu_id):
                continue
            used = float(parts[1])
            total = float(parts[2])
            return used, total
        except Exception:
            continue
    return None, None


class GPUPeakWatcher:
    def __init__(self, gpu_id: int, interval_sec: float = 0.35):
        self._gpu_id = int(gpu_id)
        self._interval = max(0.1, float(interval_sec))
        self._peak_mb = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def peak_mb(self) -> float:
        return float(self._peak_mb)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            used_mb, _ = _query_gpu_memory_mb(self._gpu_id)
            if used_mb is not None:
                self._peak_mb = max(float(self._peak_mb), float(used_mb))
            self._stop.wait(self._interval)


def _contains_oom(log_text: str) -> bool:
    text = str(log_text or "").lower()
    needles = (
        "out of memory",
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "failed to allocate",
        "oom",
        "std::bad_alloc",
        "hip out of memory",
        "insufficient memory",
    )
    return any(n in text for n in needles)


def _ensure_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(CSV_FIELDS))
        writer.writeheader()


def _case_key_from_row(row: Dict[str, str]) -> Optional[CaseKey]:
    try:
        return CaseKey(
            input_path=str(row.get("input_path") or ""),
            mode=str(row.get("mode") or ""),
            precision=str(row.get("precision") or ""),
            vae_model=str(row.get("vae_model") or ""),
            scale=int(float(row.get("scale") or 0)),
            max_target_resolution=int(float(row.get("max_target_resolution") or 0)),
            tile_size=int(float(row.get("tile_size") or 0)),
            overlap=int(float(row.get("overlap") or 0)),
            frame_chunk_size=int(float(row.get("frame_chunk_size") or 0)),
            gpu_id=int(float(row.get("gpu_id") or 0)),
        )
    except Exception:
        return None


def _load_existing_cases(path: Path) -> Dict[CaseKey, Dict[str, str]]:
    if not path.exists():
        return {}
    found: Dict[CaseKey, Dict[str, str]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                key = _case_key_from_row(row)
                if key is None:
                    continue
                found[key] = row
    except Exception:
        return {}
    return found


def _append_csv_row(path: Path, row: Dict[str, object]) -> None:
    out: Dict[str, object] = {k: row.get(k, "") for k in CSV_FIELDS}
    with path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(CSV_FIELDS))
        writer.writerow(out)


def _build_case_id(scale: int, chunk: int, tile: int, overlap: int) -> str:
    return f"s{int(scale)}_c{int(chunk)}_t{int(tile)}_o{int(overlap)}"


def _resolve_model_name(version_ui: str) -> str:
    return "FlashVSR-v1.1" if str(version_ui).strip() == "1.1" else "FlashVSR"


def _run_case(
    *,
    python_exe: Path,
    cli_path: Path,
    models_dir: Path,
    run_dir: Path,
    input_path: str,
    mode: str,
    precision: str,
    vae_model: str,
    scale: int,
    max_target_resolution: int,
    tile_size: int,
    overlap: int,
    frame_chunk_size: int,
    gpu_id: int,
    version_ui: str,
    keep_models_on_cpu: bool,
    timeout_minutes: float,
) -> CaseResult:
    case_id = _build_case_id(scale, frame_chunk_size, tile_size, overlap)
    output_file = run_dir / f"{case_id}.mp4"
    log_file = run_dir / f"{case_id}.log"
    model_name = _resolve_model_name(version_ui)
    timeout_sec = max(60.0, float(timeout_minutes) * 60.0)

    cmd = [
        str(python_exe),
        "-u",
        str(cli_path),
        "--input",
        str(input_path),
        "--output",
        str(output_file),
        "--model",
        model_name,
        "--mode",
        str(mode),
        "--vae_model",
        str(vae_model),
        "--precision",
        str(precision),
        "--device",
        f"cuda:{int(gpu_id)}",
        "--attention_mode",
        "flash_attention_2",
        "--scale",
        str(int(scale)),
        "--tiled_dit",
        "--tile_size",
        str(int(tile_size)),
        "--tile_overlap",
        str(int(overlap)),
        "--unload_dit",
        "--sparse_ratio",
        "2.0",
        "--kv_ratio",
        "3.0",
        "--local_range",
        "11",
        "--cfg_scale",
        "1.0",
        "--denoise_amount",
        "1.0",
        "--seed",
        "0",
        "--frame_chunk_size",
        str(int(frame_chunk_size)),
        "--resize_factor",
        "1.0",
        "--codec",
        "libx264",
        "--crf",
        "18",
        "--start_frame",
        "0",
        "--end_frame",
        "-1",
        "--models_dir",
        str(models_dir),
        "--color_fix",
        "--force_offload",
    ]
    cmd.append("--keep_models_on_cpu" if bool(keep_models_on_cpu) else "--no_keep_models_on_cpu")

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    triton_cache = run_dir.parent / "triton_cache"
    triton_cache.mkdir(parents=True, exist_ok=True)
    env["TRITON_CACHE_DIR"] = str(triton_cache)

    watcher = GPUPeakWatcher(gpu_id=int(gpu_id))
    gpu_used_mb, gpu_total_mb = _query_gpu_memory_mb(int(gpu_id))
    gpu_total_gb = float(gpu_total_mb or 0.0) / 1024.0
    start = time.time()
    returncode = 1
    log_lines: List[str] = []
    oom = False

    with log_file.open("w", encoding="utf-8", newline="") as fp:
        fp.write("COMMAND:\n")
        fp.write(" ".join(f'"{x}"' if " " in x else x for x in cmd))
        fp.write("\n\nOUTPUT:\n")
        fp.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cli_path.parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        watcher.start()
        try:
            assert proc.stdout is not None
            while True:
                line = proc.stdout.readline()
                if line:
                    text = line.rstrip("\r\n")
                    log_lines.append(text)
                    fp.write(text + "\n")
                    fp.flush()
                    _safe_print(text)
                    if _contains_oom(text):
                        oom = True
                    if (time.time() - start) > timeout_sec:
                        fp.write(f"\n[Timeout] Terminating process after {timeout_sec:.0f}s.\n")
                        fp.flush()
                        try:
                            proc.terminate()
                            proc.wait(timeout=10.0)
                        except Exception:
                            try:
                                proc.kill()
                            except Exception:
                                pass
                        oom = oom or _contains_oom("\n".join(log_lines))
                        returncode = 124
                        break
                    continue
                if proc.poll() is not None:
                    break
                if (time.time() - start) > timeout_sec:
                    fp.write(f"\n[Timeout] Terminating process after {timeout_sec:.0f}s.\n")
                    fp.flush()
                    try:
                        proc.terminate()
                        proc.wait(timeout=10.0)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    oom = oom or _contains_oom("\n".join(log_lines))
                    returncode = 124
                    break
            if returncode != 124:
                returncode = int(proc.wait())
        finally:
            watcher.stop()

    elapsed = time.time() - start
    full_log = "\n".join(log_lines)
    oom = oom or _contains_oom(full_log)
    output_ok = output_file.exists() and output_file.is_file() and output_file.stat().st_size > 1024
    success = bool(returncode == 0 and output_ok and not oom)

    return CaseResult(
        success=success,
        oom=bool(oom),
        returncode=int(returncode),
        elapsed_sec=float(elapsed),
        peak_vram_gb=float(watcher.peak_mb / 1024.0),
        gpu_total_gb=float(gpu_total_gb),
        log_file=str(log_file),
        output_file=str(output_file),
    )


def _iter_scales(raw: str) -> Iterable[int]:
    seen = set()
    for token in str(raw or "").split(","):
        t = token.strip()
        if not t:
            continue
        try:
            s = 2 if int(float(t)) <= 2 else 4
        except Exception:
            continue
        if s in seen:
            continue
        seen.add(s)
        yield s


def _build_resolution_info(
    *,
    input_width: int,
    input_height: int,
    scale: int,
    max_target_resolution: int,
) -> Tuple[int, int, int, int]:
    plan = estimate_fixed_scale_upscale_plan_from_dims(
        input_width,
        input_height,
        requested_scale=float(scale),
        model_scale=int(scale),
        max_edge=int(max_target_resolution),
        force_pre_downscale=True,
    )
    return (
        int(plan.preprocess_width),
        int(plan.preprocess_height),
        int(plan.resize_width),
        int(plan.resize_height),
    )


def _parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Resumable FlashVSR VRAM sweep")
    parser.add_argument("--input", type=str, default=str(base_dir / "450frame960.mp4"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--mode", type=str, default="full")
    parser.add_argument("--version", type=str, default="1.1")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--vae-model", type=str, default="Wan2.2")
    parser.add_argument("--tile-overlap", type=int, default=48)
    parser.add_argument("--max-target-resolution", type=int, default=0)
    parser.add_argument("--chunk-list", type=str, default=",".join(str(v) for v in CHUNK_SIZES))
    parser.add_argument("--scales", type=str, default="4,2")
    parser.add_argument("--keep-models-on-cpu", action="store_true", default=True)
    parser.add_argument("--no-keep-models-on-cpu", action="store_false", dest="keep_models_on_cpu")
    parser.add_argument("--max-cases", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--timeout-minutes", type=float, default=35.0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument(
        "--records-csv",
        type=str,
        default=str(base_dir / "outputs" / "flashvsr_vram_sweeps" / "flashvsr_vram_records.csv"),
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Optional explicit run directory for logs and outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base_dir = Path(__file__).resolve().parents[1]
    input_path = Path(args.input).resolve()
    cli_path = base_dir / "ComfyUI-FlashVSR_Stable" / "cli_main.py"
    models_dir = base_dir / "ComfyUI-FlashVSR_Stable" / "models"
    python_exe = base_dir / "venv" / "Scripts" / "python.exe"
    records_csv = Path(args.records_csv).resolve()

    if not input_path.exists():
        _safe_print(f"ERROR: missing input: {input_path}")
        return 2
    if not cli_path.exists():
        _safe_print(f"ERROR: missing CLI: {cli_path}")
        return 2
    if not models_dir.exists():
        _safe_print(f"ERROR: missing models dir: {models_dir}")
        return 2
    if not python_exe.exists():
        _safe_print(f"ERROR: missing venv python: {python_exe}")
        return 2

    dims = None
    try:
        from shared.path_utils import get_media_dimensions

        dims = get_media_dimensions(str(input_path))
    except Exception:
        dims = None
    if not dims:
        _safe_print("ERROR: failed to read input dimensions (ffprobe).")
        return 2
    input_w, input_h = int(dims[0]), int(dims[1])

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = base_dir / "outputs" / "flashvsr_vram_sweeps" / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    _ensure_csv(records_csv)
    existing = _load_existing_cases(records_csv) if args.resume else {}

    overlap = max(24, min(512, int(args.tile_overlap)))
    max_target_resolution = max(0, int(args.max_target_resolution))

    chunk_values: List[int] = []
    for tok in str(args.chunk_list).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            chunk_values.append(max(32, int(float(tok))))
        except Exception:
            continue
    if not chunk_values:
        chunk_values = list(CHUNK_SIZES)

    scales = list(_iter_scales(args.scales))
    if not scales:
        scales = [4, 2]

    max_cases = max(0, int(args.max_cases))
    executed_cases = 0
    success_cases = 0

    _safe_print(f"Input: {input_path} ({input_w}x{input_h})")
    _safe_print(f"GPU: cuda:{int(args.gpu_id)}")
    _safe_print(f"Scales: {scales}")
    _safe_print(f"Chunk list: {chunk_values}")
    _safe_print(f"Records CSV: {records_csv}")
    _safe_print(f"Run dir: {run_dir}")
    _safe_print("-" * 72)

    def run_or_reuse_case(scale: int, chunk: int, tile: int) -> Tuple[CaseResult, bool]:
        key = CaseKey(
            input_path=str(input_path),
            mode=str(args.mode),
            precision=str(args.precision),
            vae_model=str(args.vae_model),
            scale=int(scale),
            max_target_resolution=int(max_target_resolution),
            tile_size=int(tile),
            overlap=int(overlap),
            frame_chunk_size=int(chunk),
            gpu_id=int(args.gpu_id),
        )
        if args.resume and key in existing:
            row = existing[key]
            cached = CaseResult(
                success=str(row.get("success", "")).strip().lower() in {"1", "true", "yes"},
                oom=str(row.get("oom", "")).strip().lower() in {"1", "true", "yes"},
                returncode=int(float(row.get("returncode") or 1)),
                elapsed_sec=float(row.get("elapsed_sec") or 0.0),
                peak_vram_gb=float(row.get("peak_vram_gb") or 0.0),
                gpu_total_gb=float(row.get("gpu_total_gb") or 0.0),
                log_file=str(row.get("log_file") or ""),
                output_file=str(row.get("output_file") or ""),
            )
            return cached, True

        preprocess_w, preprocess_h, out_w, out_h = _build_resolution_info(
            input_width=input_w,
            input_height=input_h,
            scale=int(scale),
            max_target_resolution=int(max_target_resolution),
        )
        case_id = _build_case_id(scale, chunk, tile, overlap)
        _safe_print(
            f"[RUN ] {case_id} | scale={scale}x chunk={chunk} tile={tile} overlap={overlap} "
            f"| preprocess={preprocess_w}x{preprocess_h} output={out_w}x{out_h}"
        )
        result = _run_case(
            python_exe=python_exe,
            cli_path=cli_path,
            models_dir=models_dir,
            run_dir=run_dir,
            input_path=str(input_path),
            mode=str(args.mode),
            precision=str(args.precision),
            vae_model=str(args.vae_model),
            scale=int(scale),
            max_target_resolution=int(max_target_resolution),
            tile_size=int(tile),
            overlap=int(overlap),
            frame_chunk_size=int(chunk),
            gpu_id=int(args.gpu_id),
            version_ui=str(args.version),
            keep_models_on_cpu=bool(args.keep_models_on_cpu),
            timeout_minutes=float(args.timeout_minutes),
        )

        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "case_id": case_id,
            "input_path": str(input_path),
            "input_width": int(input_w),
            "input_height": int(input_h),
            "preprocess_width": int(preprocess_w),
            "preprocess_height": int(preprocess_h),
            "output_width": int(out_w),
            "output_height": int(out_h),
            "mode": str(args.mode),
            "precision": str(args.precision),
            "vae_model": str(args.vae_model),
            "scale": int(scale),
            "max_target_resolution": int(max_target_resolution),
            "tile_size": int(tile),
            "overlap": int(overlap),
            "frame_chunk_size": int(chunk),
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
        }
        _append_csv_row(records_csv, row)
        existing[key] = {k: str(v) for k, v in row.items()}
        return result, False

    for scale in scales:
        for chunk in chunk_values:
            low = 0
            high = len(TILES_ASC) - 1
            best_success_tile: Optional[int] = None

            while low <= high:
                if max_cases and executed_cases >= max_cases:
                    _safe_print("Max case limit reached, stopping.")
                    _safe_print(f"Executed cases: {executed_cases}, successful cases: {success_cases}")
                    _safe_print(f"Records CSV: {records_csv}")
                    return 0

                mid = (low + high) // 2
                tile = int(TILES_ASC[mid])
                result, reused = run_or_reuse_case(scale, chunk, tile)
                if not reused:
                    executed_cases += 1
                if result.success:
                    success_cases += 1 if not reused else 0
                    best_success_tile = tile
                    low = mid + 1
                    _safe_print(
                        f"[ OK ] scale={scale} chunk={chunk} tile={tile} "
                        f"peak={result.peak_vram_gb:.2f}GB elapsed={result.elapsed_sec:.1f}s"
                    )
                else:
                    high = mid - 1
                    _safe_print(
                        f"[FAIL] scale={scale} chunk={chunk} tile={tile} rc={result.returncode} "
                        f"oom={result.oom} peak={result.peak_vram_gb:.2f}GB"
                    )

            if best_success_tile is None:
                _safe_print(f"[INFO] no successful tile for scale={scale} chunk={chunk}")
                continue

            next_tile = best_success_tile + 32
            if next_tile <= 1024:
                if max_cases and executed_cases >= max_cases:
                    _safe_print("Max case limit reached, stopping.")
                    _safe_print(f"Executed cases: {executed_cases}, successful cases: {success_cases}")
                    _safe_print(f"Records CSV: {records_csv}")
                    return 0
                res_top, reused_top = run_or_reuse_case(scale, chunk, next_tile)
                if not reused_top:
                    executed_cases += 1
                if res_top.success and not reused_top:
                    success_cases += 1
                _safe_print(
                    f"[TOP ] scale={scale} chunk={chunk} tile={next_tile} "
                    f"success={res_top.success} oom={res_top.oom} peak={res_top.peak_vram_gb:.2f}GB"
                )

            _safe_print(f"[DONE] scale={scale} chunk={chunk} best_success_tile={best_success_tile}")

    _safe_print("-" * 72)
    _safe_print(f"Sweep completed. Executed new cases: {executed_cases}")
    _safe_print(f"New successful cases: {success_cases}")
    _safe_print(f"Records CSV: {records_csv}")
    _safe_print(f"Run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
