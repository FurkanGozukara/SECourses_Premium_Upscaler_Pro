"""
Systematic FlashVSR VRAM sweep with resume support.

Default target:
- input: 450frame960.mp4
- scales: 4x and 2x (common output sizes: 3840x2160 and 1920x1080)
- overlap: 48
- tile search: maximize tile size under OOM/shared-VRAM guardrails for each chunk size

Usage:
    .\venv\Scripts\python.exe tools\flashvsr_vram_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import re
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

from shared.ffmpeg_utils import scale_video
from shared.path_utils import get_media_dimensions
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
    "effective_success",
    "raw_success",
    "profile_partial",
    "shared_vram_suspect",
    "oom_recovery_override",
    "failure_reason",
    "processing_fps",
    "peak_vram_cli_gb",
    "prepared_input_path",
)

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
FPS_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"Total Processing Time:\s*([0-9]+(?:\.[0-9]+)?)s\s*\(([0-9]+(?:\.[0-9]+)?)\s*FPS\)", re.IGNORECASE),
    re.compile(r"Total Time:\s*[0-9:]+\s*\(([0-9]+(?:\.[0-9]+)?)\s*FPS\)", re.IGNORECASE),
)
PEAK_VRAM_PATTERN = re.compile(r"Peak VRAM Used:\s*([0-9]+(?:\.[0-9]+)?)\s*GB", re.IGNORECASE)
TOTAL_FRAMES_PATTERN = re.compile(r"Total Frames Processed:\s*([0-9]+)\s*/\s*([0-9]+)", re.IGNORECASE)
TILE_PROGRESS_PATTERN = re.compile(r"Processing\s+Tiles:\s*([0-9]+)\s*/\s*([0-9]+)", re.IGNORECASE)
STEP_PROGRESS_PATTERN = re.compile(r"Processing:\s*([0-9]+)\s*/\s*([0-9]+)", re.IGNORECASE)
OOM_RECOVERY_HINTS: Tuple[str, ...] = (
    "auto-enabling tiled vae to prevent oom",
    "auto-enabling tiled dit to prevent oom",
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
    raw_success: bool
    effective_success: bool
    profile_partial: bool
    shared_vram_suspect: bool
    oom_recovery_override: bool
    failure_reason: str
    oom: bool
    returncode: int
    elapsed_sec: float
    peak_vram_gb: float
    peak_vram_cli_gb: float
    processing_fps: float
    gpu_total_gb: float
    log_file: str
    output_file: str
    prepared_input_path: str


@dataclass(frozen=True)
class PreparedInput:
    effective_input_path: str
    preprocess_width: int
    preprocess_height: int
    output_width: int
    output_height: int


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


def _strip_ansi(line: str) -> str:
    return ANSI_ESCAPE_RE.sub("", str(line or ""))


def _run_nvidia_smi_query() -> Optional[str]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
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


def _query_gpu_stats_mb(gpu_id: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    out = _run_nvidia_smi_query()
    if not out:
        return None, None, None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            if idx != int(gpu_id):
                continue
            used = float(parts[1])
            total = float(parts[2])
            util = float(parts[3])
            return used, total, util
        except Exception:
            continue
    return None, None, None


class GPUPeakWatcher:
    def __init__(
        self,
        gpu_id: int,
        *,
        interval_sec: float = 0.35,
        low_util_pct: float = 12.0,
        near_full_margin_mb: float = 768.0,
        pressure_seconds: float = 120.0,
    ):
        self._gpu_id = int(gpu_id)
        self._interval = max(0.1, float(interval_sec))
        self._low_util_pct = max(0.0, float(low_util_pct))
        self._near_full_margin_mb = max(64.0, float(near_full_margin_mb))
        self._pressure_seconds = max(10.0, float(pressure_seconds))
        self._peak_mb = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._low_util_near_full_sec = 0.0
        self._shared_vram_suspect = False
        self._util_sum = 0.0
        self._util_samples = 0

    @property
    def peak_mb(self) -> float:
        return float(self._peak_mb)

    @property
    def shared_vram_suspect(self) -> bool:
        return bool(self._shared_vram_suspect)

    @property
    def average_util_pct(self) -> float:
        if self._util_samples <= 0:
            return 0.0
        return float(self._util_sum / float(self._util_samples))

    @property
    def pressure_seconds(self) -> float:
        return float(self._low_util_near_full_sec)

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
            used_mb, total_mb, util_pct = _query_gpu_stats_mb(self._gpu_id)
            if used_mb is not None:
                self._peak_mb = max(float(self._peak_mb), float(used_mb))
            if util_pct is not None:
                self._util_sum += float(util_pct)
                self._util_samples += 1
            if used_mb is not None and total_mb is not None and util_pct is not None:
                near_full = float(used_mb) >= max(0.0, float(total_mb) - self._near_full_margin_mb)
                low_util = float(util_pct) <= self._low_util_pct
                if near_full and low_util:
                    self._low_util_near_full_sec += self._interval
                else:
                    self._low_util_near_full_sec = max(0.0, self._low_util_near_full_sec - (self._interval * 0.5))
                if self._low_util_near_full_sec >= self._pressure_seconds:
                    self._shared_vram_suspect = True
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


def _contains_oom_recovery_override(log_text: str) -> bool:
    text = str(log_text or "").lower()
    return any(hint in text for hint in OOM_RECOVERY_HINTS)


def _extract_processing_metrics(log_text: str, elapsed_sec: float) -> Tuple[float, float, int, bool]:
    clean = _strip_ansi(str(log_text or ""))
    processing_fps = 0.0
    peak_cli = 0.0
    total_frames = 0
    progress_observed = False

    for pat in FPS_PATTERNS:
        for match in pat.finditer(clean):
            groups = match.groups()
            if not groups:
                continue
            try:
                fps_val = float(groups[-1])
                if fps_val > 0:
                    processing_fps = max(processing_fps, fps_val)
            except Exception:
                continue

    for match in PEAK_VRAM_PATTERN.finditer(clean):
        try:
            peak_cli = max(peak_cli, float(match.group(1)))
        except Exception:
            continue

    for match in TOTAL_FRAMES_PATTERN.finditer(clean):
        try:
            total_frames = max(total_frames, int(match.group(1)))
        except Exception:
            continue

    if TILE_PROGRESS_PATTERN.search(clean) or STEP_PROGRESS_PATTERN.search(clean):
        progress_observed = True

    if processing_fps <= 0.0 and total_frames > 0 and elapsed_sec > 0:
        processing_fps = float(total_frames) / float(elapsed_sec)

    return float(processing_fps), float(peak_cli), int(total_frames), bool(progress_observed)


def _ensure_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(CSV_FIELDS))
            writer.writeheader()
        return

    try:
        with path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            header = list(reader.fieldnames or [])
            rows = list(reader)
    except Exception:
        with path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(CSV_FIELDS))
            writer.writeheader()
        return

    if header == list(CSV_FIELDS):
        return

    missing = [f for f in CSV_FIELDS if f not in header]
    if not missing:
        return

    backup = path.with_suffix(path.suffix + ".bak")
    try:
        if not backup.exists():
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            safe_row = row if isinstance(row, dict) else {}
            writer.writerow({k: safe_row.get(k, "") for k in CSV_FIELDS})
    tmp_path.replace(path)
    _safe_print(f"[CSV ] migrated schema with new fields: {', '.join(missing)}")


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
        int(input_width),
        int(input_height),
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


def _prepare_effective_input(
    *,
    input_path: Path,
    run_dir: Path,
    input_width: int,
    input_height: int,
    scale: int,
    max_target_resolution: int,
    cache: Dict[Tuple[str, int, int], PreparedInput],
) -> PreparedInput:
    cache_key = (str(input_path), int(scale), int(max_target_resolution))
    cached = cache.get(cache_key)
    if cached is not None:
        if Path(cached.effective_input_path).exists():
            return cached

    preprocess_w, preprocess_h, output_w, output_h = _build_resolution_info(
        input_width=int(input_width),
        input_height=int(input_height),
        scale=int(scale),
        max_target_resolution=int(max_target_resolution),
    )

    if int(preprocess_w) == int(input_width) and int(preprocess_h) == int(input_height):
        prepared = PreparedInput(
            effective_input_path=str(input_path),
            preprocess_width=int(preprocess_w),
            preprocess_height=int(preprocess_h),
            output_width=int(output_w),
            output_height=int(output_h),
        )
        cache[cache_key] = prepared
        return prepared

    prepared_dir = run_dir / "_prepared_inputs"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_name = (
        f"{input_path.stem}_pre_s{int(scale)}_m{int(max_target_resolution)}_"
        f"{int(preprocess_w)}x{int(preprocess_h)}.mp4"
    )
    prepared_path = prepared_dir / prepared_name
    if not (prepared_path.exists() and prepared_path.stat().st_size > 1024):
        _safe_print(
            f"[PREP] downscaling input for max-edge enforcement: "
            f"{input_width}x{input_height} -> {preprocess_w}x{preprocess_h}"
        )
        ok, err = scale_video(
            input_path=input_path,
            output_path=prepared_path,
            width=int(preprocess_w),
            height=int(preprocess_h),
            video_codec="libx264",
            preset="veryfast",
            lossless=True,
            audio_copy_first=True,
        )
        if not ok:
            raise RuntimeError(f"failed to preprocess input via ffmpeg: {err}")

    prepared = PreparedInput(
        effective_input_path=str(prepared_path),
        preprocess_width=int(preprocess_w),
        preprocess_height=int(preprocess_h),
        output_width=int(output_w),
        output_height=int(output_h),
    )
    cache[cache_key] = prepared
    return prepared


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=10.0)
        return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=5.0)
    except Exception:
        pass


def _run_case(
    *,
    python_exe: Path,
    cli_path: Path,
    models_dir: Path,
    run_dir: Path,
    input_path: str,
    prepared_input_path: str,
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
    stall_seconds: float,
    min_effective_fps: float,
    shared_low_util_pct: float,
    shared_near_full_margin_mb: float,
    shared_pressure_seconds: float,
    min_shared_check_runtime_sec: float,
    accept_timeout_profile: bool,
    profile_min_elapsed_sec: float,
) -> CaseResult:
    case_id = _build_case_id(scale, frame_chunk_size, tile_size, overlap)
    output_file = run_dir / f"{case_id}.mp4"
    log_file = run_dir / f"{case_id}.log"
    if output_file.exists():
        try:
            output_file.unlink()
        except Exception:
            pass
    model_name = _resolve_model_name(version_ui)
    timeout_sec = max(60.0, float(timeout_minutes) * 60.0)
    stall_sec = max(45.0, float(stall_seconds))

    cmd = [
        str(python_exe),
        "-u",
        str(cli_path),
        "--input",
        str(prepared_input_path),
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
    # Keep sweep measurements strict: do not allow runtime OOM auto-fallback to flip tiling modes.
    env["FLASHVSR_DISABLE_OOM_RECOVERY"] = "1"
    triton_cache = run_dir.parent / "triton_cache"
    triton_cache.mkdir(parents=True, exist_ok=True)
    env["TRITON_CACHE_DIR"] = str(triton_cache)

    watcher = GPUPeakWatcher(
        gpu_id=int(gpu_id),
        low_util_pct=float(shared_low_util_pct),
        near_full_margin_mb=float(shared_near_full_margin_mb),
        pressure_seconds=float(shared_pressure_seconds),
    )
    _gpu_used_mb, gpu_total_mb, _gpu_util = _query_gpu_stats_mb(int(gpu_id))
    gpu_total_gb = float(gpu_total_mb or 0.0) / 1024.0
    start = time.time()
    returncode: Optional[int] = None
    log_lines: List[str] = []
    oom = False
    termination_reason = ""

    with log_file.open("w", encoding="utf-8", newline="") as fp:
        fp.write("COMMAND:\n")
        fp.write(" ".join(f'"{x}"' if " " in x else x for x in cmd))
        fp.write("\n\nSWEEP_META:\n")
        fp.write(f"source_input={input_path}\n")
        fp.write(f"prepared_input={prepared_input_path}\n")
        fp.write(f"timeout_sec={timeout_sec:.1f}\n")
        fp.write(f"stall_sec={stall_sec:.1f}\n")
        fp.write(f"min_effective_fps={float(min_effective_fps):.4f}\n")
        fp.write(f"accept_timeout_profile={bool(accept_timeout_profile)}\n")
        fp.write(f"profile_min_elapsed_sec={float(profile_min_elapsed_sec):.1f}\n")
        fp.write("disable_oom_recovery_env=1\n")
        fp.write(
            "shared_guard="
            f"low_util<={float(shared_low_util_pct):.1f}% "
            f"and near_full_margin={float(shared_near_full_margin_mb):.1f}MB "
            f"for >= {float(shared_pressure_seconds):.1f}s\n"
        )
        fp.write("\nOUTPUT:\n")
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

        line_queue: "queue.Queue[object]" = queue.Queue()
        done_marker = object()

        def _pump_output() -> None:
            try:
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    line_queue.put(raw_line)
            except Exception:
                pass
            finally:
                line_queue.put(done_marker)

        pump_thread = threading.Thread(target=_pump_output, daemon=True)
        pump_thread.start()
        last_output_ts = time.time()

        try:
            while True:
                now = time.time()
                elapsed = now - start

                if proc.poll() is None and elapsed > timeout_sec:
                    termination_reason = "timeout"
                    fp.write(f"\n[Timeout] terminating process after {timeout_sec:.0f}s.\n")
                    fp.flush()
                    _terminate_process(proc)
                    returncode = 124
                    break

                if (
                    proc.poll() is None
                    and elapsed >= max(30.0, float(min_shared_check_runtime_sec))
                    and watcher.shared_vram_suspect
                ):
                    termination_reason = "shared_vram_pressure"
                    fp.write(
                        "\n[Guard] suspected shared-VRAM thrash: "
                        f"low-util-near-full window reached {watcher.pressure_seconds:.1f}s. Terminating.\n"
                    )
                    fp.flush()
                    _terminate_process(proc)
                    returncode = 125
                    break

                try:
                    item = line_queue.get(timeout=0.5)
                except queue.Empty:
                    item = None

                if item is done_marker:
                    if proc.poll() is not None:
                        break
                    continue

                if isinstance(item, str):
                    text = item.rstrip("\r\n")
                    clean = _strip_ansi(text)
                    log_lines.append(clean)
                    fp.write(clean + "\n")
                    fp.flush()
                    _safe_print(clean)
                    last_output_ts = time.time()
                    if _contains_oom(clean):
                        oom = True

                if proc.poll() is not None and line_queue.empty():
                    break

                if proc.poll() is None and (time.time() - last_output_ts) > stall_sec:
                    termination_reason = "stalled_no_output"
                    fp.write(f"\n[Stall] no output for {stall_sec:.0f}s; terminating process.\n")
                    fp.flush()
                    _terminate_process(proc)
                    returncode = 126
                    break

            if returncode is None:
                try:
                    returncode = int(proc.wait(timeout=10.0))
                except Exception:
                    _terminate_process(proc)
                    returncode = 127
        finally:
            watcher.stop()
            try:
                pump_thread.join(timeout=2.0)
            except Exception:
                pass

    elapsed = time.time() - start
    full_log = "\n".join(log_lines)
    oom = oom or _contains_oom(full_log)
    oom_recovery_override = _contains_oom_recovery_override(full_log)
    processing_fps, peak_cli_gb, _total_frames, progress_observed = _extract_processing_metrics(full_log, elapsed)
    peak_sampled_gb = float(watcher.peak_mb / 1024.0)
    peak_vram_gb = max(float(peak_sampled_gb), float(peak_cli_gb))
    output_ok = output_file.exists() and output_file.is_file() and output_file.stat().st_size > 1024
    raw_success = bool(returncode == 0 and output_ok and not oom and not oom_recovery_override)

    shared_vram_suspect = bool(watcher.shared_vram_suspect)
    if raw_success and float(processing_fps) > 0 and float(processing_fps) < max(0.0, float(min_effective_fps)):
        shared_vram_suspect = True
        if not termination_reason:
            termination_reason = f"fps_below_floor_{float(processing_fps):.4f}"

    if shared_vram_suspect and not termination_reason:
        termination_reason = "shared_vram_suspect"

    profile_partial = bool(
        bool(accept_timeout_profile)
        and int(returncode or 0) == 124
        and (not oom)
        and (not oom_recovery_override)
        and (not shared_vram_suspect)
        and bool(progress_observed)
        and float(elapsed) >= max(30.0, float(profile_min_elapsed_sec))
        and float(peak_vram_gb) > 0.25
    )

    effective_success = bool((raw_success or profile_partial) and not shared_vram_suspect)
    success = bool(effective_success)
    if oom_recovery_override:
        termination_reason = "oom_recovery_override"
    elif profile_partial:
        termination_reason = "profile_timeout_ok"

    if not termination_reason:
        if success:
            termination_reason = "ok"
        elif oom_recovery_override:
            termination_reason = "oom_recovery_override"
        elif oom:
            termination_reason = "oom_detected"
        elif returncode == 124:
            termination_reason = "timeout"
        elif returncode == 126:
            termination_reason = "stalled_no_output"
        elif returncode == 125:
            termination_reason = "shared_vram_pressure"
        elif returncode != 0:
            termination_reason = f"returncode_{int(returncode)}"
        elif not output_ok:
            termination_reason = "missing_output"
        else:
            termination_reason = "failed"

    return CaseResult(
        success=success,
        raw_success=bool(raw_success),
        effective_success=bool(effective_success),
        profile_partial=bool(profile_partial),
        shared_vram_suspect=bool(shared_vram_suspect),
        oom_recovery_override=bool(oom_recovery_override),
        failure_reason=str(termination_reason),
        oom=bool(oom),
        returncode=int(returncode),
        elapsed_sec=float(elapsed),
        peak_vram_gb=float(peak_vram_gb),
        peak_vram_cli_gb=float(peak_cli_gb),
        processing_fps=float(processing_fps),
        gpu_total_gb=float(gpu_total_gb),
        log_file=str(log_file),
        output_file=str(output_file),
        prepared_input_path=str(prepared_input_path),
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

    dims = get_media_dimensions(str(input_path))
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
    prepared_cache: Dict[Tuple[str, int, int], PreparedInput] = {}

    _safe_print(f"Input: {input_path} ({input_w}x{input_h})")
    _safe_print(f"GPU: cuda:{int(args.gpu_id)}")
    _safe_print(f"Scales: {scales}")
    _safe_print(f"Chunk list: {chunk_values}")
    _safe_print(f"Records CSV: {records_csv}")
    _safe_print(f"Run dir: {run_dir}")
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
            has_effective = str(row.get("effective_success", "")).strip() != ""
            if has_effective:
                raw_success = _to_bool(row.get("raw_success")) if str(row.get("raw_success", "")).strip() else _to_bool(row.get("success"))
                shared_suspect = _to_bool(row.get("shared_vram_suspect"))
                effective_success = _to_bool(row.get("effective_success"))
                cached = CaseResult(
                    success=bool(effective_success),
                    raw_success=bool(raw_success),
                    effective_success=bool(effective_success),
                    profile_partial=_to_bool(row.get("profile_partial")),
                    shared_vram_suspect=bool(shared_suspect),
                    oom_recovery_override=_to_bool(row.get("oom_recovery_override")),
                    failure_reason=str(row.get("failure_reason") or ""),
                    oom=_to_bool(row.get("oom")),
                    returncode=_to_int(row.get("returncode"), 1),
                    elapsed_sec=_to_float(row.get("elapsed_sec"), 0.0),
                    peak_vram_gb=_to_float(row.get("peak_vram_gb"), 0.0),
                    peak_vram_cli_gb=_to_float(row.get("peak_vram_cli_gb"), 0.0),
                    processing_fps=_to_float(row.get("processing_fps"), 0.0),
                    gpu_total_gb=_to_float(row.get("gpu_total_gb"), 0.0),
                    log_file=str(row.get("log_file") or ""),
                    output_file=str(row.get("output_file") or ""),
                    prepared_input_path=str(row.get("prepared_input_path") or row.get("input_path") or ""),
                )
                return cached, True
            _safe_print(
                f"[RERUN] legacy row without effective_success for "
                f"scale={scale} chunk={chunk} tile={tile}; rerunning with shared-VRAM guards."
            )

        prepared = _prepare_effective_input(
            input_path=input_path,
            run_dir=run_dir,
            input_width=input_w,
            input_height=input_h,
            scale=int(scale),
            max_target_resolution=int(max_target_resolution),
            cache=prepared_cache,
        )
        case_id = _build_case_id(scale, chunk, tile, overlap)
        _safe_print(
            f"[RUN ] {case_id} | scale={scale}x chunk={chunk} tile={tile} overlap={overlap} "
            f"| preprocess={prepared.preprocess_width}x{prepared.preprocess_height} "
            f"output={prepared.output_width}x{prepared.output_height}"
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
            scale=int(scale),
            max_target_resolution=int(max_target_resolution),
            tile_size=int(tile),
            overlap=int(overlap),
            frame_chunk_size=int(chunk),
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

        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "case_id": case_id,
            "input_path": str(input_path),
            "input_width": int(input_w),
            "input_height": int(input_h),
            "preprocess_width": int(prepared.preprocess_width),
            "preprocess_height": int(prepared.preprocess_height),
            "output_width": int(prepared.output_width),
            "output_height": int(prepared.output_height),
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
                        f"peak={result.peak_vram_gb:.2f}GB fps={result.processing_fps:.3f} "
                        f"elapsed={result.elapsed_sec:.1f}s profile_partial={result.profile_partial} "
                        f"oom_recovery={result.oom_recovery_override}"
                    )
                else:
                    high = mid - 1
                    _safe_print(
                        f"[FAIL] scale={scale} chunk={chunk} tile={tile} rc={result.returncode} "
                        f"oom={result.oom} shared={result.shared_vram_suspect} "
                        f"peak={result.peak_vram_gb:.2f}GB fps={result.processing_fps:.3f} "
                        f"reason={result.failure_reason} oom_recovery={result.oom_recovery_override}"
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
                    f"success={res_top.success} shared={res_top.shared_vram_suspect} "
                    f"oom={res_top.oom} peak={res_top.peak_vram_gb:.2f}GB "
                    f"fps={res_top.processing_fps:.3f} reason={res_top.failure_reason} "
                    f"profile_partial={res_top.profile_partial} "
                    f"oom_recovery={res_top.oom_recovery_override}"
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
