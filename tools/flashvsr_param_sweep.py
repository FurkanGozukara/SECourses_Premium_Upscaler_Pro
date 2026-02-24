"""
Run a fixed 10-case FlashVSR parameter sweep on example0.mp4 (first 1 second).

Usage:
    .\venv\Scripts\python.exe tools\flashvsr_param_sweep.py
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Variant:
    name: str
    sparse_ratio: float
    kv_ratio: float
    local_range: int
    seed: int
    color_fix: bool


def _safe_console_write(text: str) -> None:
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        sys.stdout.buffer.write(text.encode(enc, errors="replace"))
    sys.stdout.flush()


def main() -> int:
    base_dir = Path(__file__).resolve().parents[1]
    input_video = base_dir / "example0.mp4"
    cli_path = base_dir / "ComfyUI-FlashVSR_Stable" / "cli_main.py"
    models_dir = base_dir / "ComfyUI-FlashVSR_Stable" / "models"
    python_exe = base_dir / "venv" / "Scripts" / "python.exe"

    if not input_video.exists():
        print(f"ERROR: missing input video: {input_video}")
        return 2
    if not cli_path.exists():
        print(f"ERROR: missing CLI: {cli_path}")
        return 2
    if not python_exe.exists():
        print(f"ERROR: missing venv python: {python_exe}")
        return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / "outputs" / f"flashvsr_param_sweep_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    variants: List[Variant] = [
        Variant("v01_baseline", 2.0, 3.0, 11, 0, True),
        Variant("v02_sparse15", 1.5, 3.0, 11, 0, True),
        Variant("v03_sparse17", 1.7, 3.0, 11, 0, True),
        Variant("v04_kv10", 2.0, 1.0, 11, 0, True),
        Variant("v05_kv20", 2.0, 2.0, 11, 0, True),
        Variant("v06_lr09", 2.0, 3.0, 9, 0, True),
        Variant("v07_sparse15_kv10_lr09", 1.5, 1.0, 9, 0, True),
        Variant("v08_seed123", 2.0, 3.0, 11, 123, True),
        Variant("v09_seed999", 2.0, 3.0, 11, 999, True),
        Variant("v10_nocolorfix", 2.0, 3.0, 11, 0, False),
    ]

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    triton_cache = base_dir / "temp" / "triton_cache"
    triton_cache.mkdir(parents=True, exist_ok=True)
    env["TRITON_CACHE_DIR"] = str(triton_cache)

    summary_rows = []
    total_start = time.time()

    print(f"Sweep output folder: {run_dir}")
    print(f"Input: {input_video}")
    print(f"Model: FlashVSR-v1.1, mode=full, scale=4x")
    print(f"Frames: start=0, end=30 (first 1 second @ 30 fps)")
    print("-" * 72)

    for idx, v in enumerate(variants, start=1):
        out_file = run_dir / f"{idx:02d}_{v.name}.mp4"
        log_file = run_dir / f"{idx:02d}_{v.name}.log"

        cmd = [
            str(python_exe),
            "-u",
            str(cli_path),
            "--input",
            str(input_video),
            "--output",
            str(out_file),
            "--model",
            "FlashVSR-v1.1",
            "--mode",
            "full",
            "--vae_model",
            "Wan2.2",
            "--precision",
            "bf16",
            "--device",
            "cuda:0",
            "--attention_mode",
            "sparse_sage_attention",
            "--scale",
            "4",
            "--tiled_dit",
            "--tile_size",
            "256",
            "--tile_overlap",
            "24",
            "--unload_dit",
            "--sparse_ratio",
            str(v.sparse_ratio),
            "--kv_ratio",
            str(v.kv_ratio),
            "--local_range",
            str(v.local_range),
            "--seed",
            str(v.seed),
            "--frame_chunk_size",
            "0",
            "--resize_factor",
            "1.0",
            "--codec",
            "libx264",
            "--crf",
            "18",
            "--start_frame",
            "0",
            "--end_frame",
            "30",
            "--models_dir",
            str(models_dir),
            "--color_fix" if v.color_fix else "--no_color_fix",
        ]

        print(
            f"[{idx:02d}/{len(variants)}] {v.name} "
            f"(sparse={v.sparse_ratio}, kv={v.kv_ratio}, lr={v.local_range}, seed={v.seed}, color_fix={v.color_fix})"
        )
        run_start = time.time()
        return_code = 1

        with log_file.open("w", encoding="utf-8", newline="") as log_fp:
            log_fp.write("COMMAND:\n")
            log_fp.write(" ".join(f'"{x}"' if " " in x else x for x in cmd))
            log_fp.write("\n\nOUTPUT:\n")
            proc = subprocess.Popen(
                cmd,
                cwd=base_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                _safe_console_write(line)
                log_fp.write(line)
            return_code = proc.wait()

        elapsed = time.time() - run_start
        exists = out_file.exists()
        size_mb = (out_file.stat().st_size / (1024 * 1024)) if exists else 0.0
        status = "ok" if return_code == 0 and exists else "failed"
        print(f"  -> {status} | {elapsed:.1f}s | {size_mb:.1f} MB")

        summary_rows.append(
            {
                "index": idx,
                "variant": v.name,
                "sparse_ratio": v.sparse_ratio,
                "kv_ratio": v.kv_ratio,
                "local_range": v.local_range,
                "seed": v.seed,
                "color_fix": int(v.color_fix),
                "return_code": return_code,
                "status": status,
                "elapsed_sec": round(elapsed, 3),
                "output_file": str(out_file),
                "output_size_mb": round(size_mb, 3),
                "log_file": str(log_file),
            }
        )

    manifest = run_dir / "sweep_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    readme = run_dir / "README.txt"
    ok_count = sum(1 for r in summary_rows if r["status"] == "ok")
    total_elapsed = time.time() - total_start
    with readme.open("w", encoding="utf-8", newline="") as fp:
        fp.write("FlashVSR Parameter Sweep\n")
        fp.write("=======================\n\n")
        fp.write(f"Input: {input_video}\n")
        fp.write("Model: FlashVSR-v1.1\n")
        fp.write("Mode: full\n")
        fp.write("Scale: 4x\n")
        fp.write("Frames: start=0, end=30 (first 1 second at 30fps)\n")
        fp.write(f"Successful runs: {ok_count}/{len(summary_rows)}\n")
        fp.write(f"Total elapsed: {total_elapsed:.1f}s\n\n")
        fp.write("See sweep_manifest.csv for all parameters and outputs.\n")

    print("-" * 72)
    print(f"Done. Success: {ok_count}/{len(summary_rows)}")
    print(f"Run folder: {run_dir}")
    print(f"Manifest: {manifest}")
    return 0 if ok_count == len(summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
