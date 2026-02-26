"""
Generate a parameter-to-VRAM relationship report from FlashVSR sweep CSV data.

Usage:
    .\venv\Scripts\python.exe tools\flashvsr_vram_report.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class SweepRow:
    input_path: str
    mode: str
    precision: str
    vae_model: str
    scale: int
    max_target_resolution: int
    preprocess_width: int
    preprocess_height: int
    output_width: int
    output_height: int
    tile_size: int
    overlap: int
    frame_chunk_size: int
    keep_models_on_cpu: bool
    tiled_dit: bool
    tiled_vae: bool
    stream_decode: bool
    gpu_id: int
    gpu_total_gb: float
    peak_vram_gb: float
    success: bool
    effective_success: bool
    raw_success: bool
    profile_partial: bool
    shared_vram_suspect: bool
    oom_recovery_override: bool
    oom: bool
    failure_reason: str
    processing_fps: float

    @property
    def preprocess_pixels(self) -> int:
        return int(self.preprocess_width * self.preprocess_height)


def _to_bool(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


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


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    q = max(0.0, min(1.0, float(q)))
    pos = q * (len(vals) - 1)
    lo = int(pos)
    hi = min(len(vals) - 1, lo + 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _iter_adjacent_slopes(items: List[Tuple[int, float]], normalize_step: float) -> Iterable[float]:
    if len(items) < 2:
        return []
    out: List[float] = []
    s = sorted(items, key=lambda x: x[0])
    for i in range(len(s) - 1):
        x1, y1 = s[i]
        x2, y2 = s[i + 1]
        dx = float(x2 - x1)
        if abs(dx) < 1e-9:
            continue
        out.append(((y2 - y1) / dx) * float(normalize_step))
    return out


def _load_rows(csv_path: Path) -> List[SweepRow]:
    rows: List[SweepRow] = []
    if not csv_path.exists():
        return rows
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if not isinstance(row, dict):
                continue
            try:
                rows.append(
                    SweepRow(
                        input_path=str(row.get("input_path") or ""),
                        mode=str(row.get("mode") or ""),
                        precision=str(row.get("precision") or ""),
                        vae_model=str(row.get("vae_model") or ""),
                        scale=2 if _to_int(row.get("scale"), 4) <= 2 else 4,
                        max_target_resolution=max(0, _to_int(row.get("max_target_resolution"), 0)),
                        preprocess_width=max(1, _to_int(row.get("preprocess_width"), 1)),
                        preprocess_height=max(1, _to_int(row.get("preprocess_height"), 1)),
                        output_width=max(1, _to_int(row.get("output_width"), 1)),
                        output_height=max(1, _to_int(row.get("output_height"), 1)),
                        tile_size=max(1, _to_int(row.get("tile_size"), 1)),
                        overlap=max(0, _to_int(row.get("overlap"), 0)),
                        frame_chunk_size=max(0, _to_int(row.get("frame_chunk_size"), 0)),
                        keep_models_on_cpu=_to_bool(row.get("keep_models_on_cpu")),
                        tiled_dit=_to_bool(row.get("tiled_dit")),
                        tiled_vae=_to_bool(row.get("tiled_vae")),
                        stream_decode=_to_bool(row.get("stream_decode")),
                        gpu_id=_to_int(row.get("gpu_id"), 0),
                        gpu_total_gb=_to_float(row.get("gpu_total_gb"), 0.0),
                        peak_vram_gb=max(0.0, _to_float(row.get("peak_vram_gb"), 0.0)),
                        success=_to_bool(row.get("success")),
                        effective_success=(
                            _to_bool(row.get("effective_success"))
                            if str(row.get("effective_success", "")).strip()
                            else _to_bool(row.get("success"))
                        ),
                        raw_success=(
                            _to_bool(row.get("raw_success"))
                            if str(row.get("raw_success", "")).strip()
                            else _to_bool(row.get("success"))
                        ),
                        profile_partial=_to_bool(row.get("profile_partial")),
                        shared_vram_suspect=_to_bool(row.get("shared_vram_suspect")),
                        oom_recovery_override=_to_bool(row.get("oom_recovery_override")),
                        oom=_to_bool(row.get("oom")),
                        failure_reason=str(row.get("failure_reason") or ""),
                        processing_fps=max(0.0, _to_float(row.get("processing_fps"), 0.0)),
                    )
                )
            except Exception:
                continue
    return rows


def _build_sensitivity(rows: List[SweepRow]) -> Dict[str, Dict[str, float]]:
    ok_rows = [
        r
        for r in rows
        if r.effective_success and (not r.shared_vram_suspect) and (not r.oom_recovery_override) and r.peak_vram_gb > 0.0
    ]

    tile_slopes: List[float] = []
    chunk_slopes: List[float] = []
    pixel_slopes: List[float] = []

    tile_groups: Dict[Tuple[object, ...], List[Tuple[int, float]]] = {}
    chunk_groups: Dict[Tuple[object, ...], List[Tuple[int, float]]] = {}
    pixel_groups: Dict[Tuple[object, ...], List[Tuple[int, float]]] = {}

    for r in ok_rows:
        tile_key = (
            r.input_path,
            r.mode,
            r.precision,
            r.vae_model,
            r.scale,
            r.max_target_resolution,
            r.preprocess_width,
            r.preprocess_height,
            r.frame_chunk_size,
            r.overlap,
            r.keep_models_on_cpu,
            r.tiled_dit,
            r.tiled_vae,
            r.stream_decode,
            r.gpu_id,
        )
        tile_groups.setdefault(tile_key, []).append((int(r.tile_size), float(r.peak_vram_gb)))

        chunk_key = (
            r.input_path,
            r.mode,
            r.precision,
            r.vae_model,
            r.scale,
            r.max_target_resolution,
            r.preprocess_width,
            r.preprocess_height,
            r.tile_size,
            r.overlap,
            r.keep_models_on_cpu,
            r.tiled_dit,
            r.tiled_vae,
            r.stream_decode,
            r.gpu_id,
        )
        chunk_groups.setdefault(chunk_key, []).append((int(r.frame_chunk_size), float(r.peak_vram_gb)))

        pixel_key = (
            r.mode,
            r.precision,
            r.vae_model,
            r.scale,
            r.tile_size,
            r.overlap,
            r.frame_chunk_size,
            r.keep_models_on_cpu,
            r.tiled_dit,
            r.tiled_vae,
            r.stream_decode,
            r.gpu_id,
        )
        pixel_groups.setdefault(pixel_key, []).append((int(r.preprocess_pixels), float(r.peak_vram_gb)))

    for vals in tile_groups.values():
        tile_slopes.extend(list(_iter_adjacent_slopes(vals, normalize_step=32.0)))
    for vals in chunk_groups.values():
        chunk_slopes.extend(list(_iter_adjacent_slopes(vals, normalize_step=64.0)))
    for vals in pixel_groups.values():
        # Normalize by 1 megapixel.
        slopes = list(_iter_adjacent_slopes(vals, normalize_step=1_000_000.0))
        pixel_slopes.extend(slopes)

    def summarize(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"count": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
        return {
            "count": float(len(values)),
            "median": float(statistics.median(values)),
            "p10": float(_quantile(values, 0.10)),
            "p90": float(_quantile(values, 0.90)),
        }

    return {
        "tile_32px_gb": summarize(tile_slopes),
        "chunk_64f_gb": summarize(chunk_slopes),
        "preprocess_1mp_gb": summarize(pixel_slopes),
    }


def _best_tile_map(rows: List[SweepRow]) -> List[Dict[str, object]]:
    best: Dict[Tuple[object, ...], SweepRow] = {}
    for r in rows:
        if (not r.effective_success) or r.shared_vram_suspect or r.oom_recovery_override:
            continue
        key = (
            r.gpu_id,
            r.input_path,
            r.scale,
            r.max_target_resolution,
            r.preprocess_width,
            r.preprocess_height,
            r.output_width,
            r.output_height,
            r.frame_chunk_size,
            r.overlap,
            r.mode,
            r.precision,
            r.vae_model,
            r.keep_models_on_cpu,
            r.tiled_dit,
            r.tiled_vae,
            r.stream_decode,
        )
        prev = best.get(key)
        if prev is None or r.tile_size > prev.tile_size or (
            r.tile_size == prev.tile_size and r.peak_vram_gb < prev.peak_vram_gb
        ):
            best[key] = r

    out: List[Dict[str, object]] = []
    for row in best.values():
        out.append(
            {
                "gpu_id": row.gpu_id,
                "input_path": row.input_path,
                "scale": row.scale,
                "max_target_resolution": row.max_target_resolution,
                "preprocess_resolution": f"{row.preprocess_width}x{row.preprocess_height}",
                "output_resolution": f"{row.output_width}x{row.output_height}",
                "frame_chunk_size": row.frame_chunk_size,
                "overlap": row.overlap,
                "best_tile_size": row.tile_size,
                "peak_vram_gb": round(row.peak_vram_gb, 4),
            }
        )
    out.sort(
        key=lambda item: (
            int(item["gpu_id"]),
            str(item["input_path"]),
            int(item["scale"]),
            int(item["frame_chunk_size"]),
            int(item["best_tile_size"]),
        )
    )
    return out


def _build_markdown(
    *,
    csv_path: Path,
    rows: List[SweepRow],
    sensitivity: Dict[str, Dict[str, float]],
    best_tiles: List[Dict[str, object]],
) -> str:
    total = len(rows)
    raw_success = sum(1 for r in rows if r.raw_success)
    effective_success = sum(1 for r in rows if r.effective_success and not r.shared_vram_suspect)
    profile_partial = sum(1 for r in rows if r.profile_partial and r.effective_success and not r.shared_vram_suspect)
    shared_suspects = sum(1 for r in rows if r.shared_vram_suspect)
    oom_recovery_overrides = sum(1 for r in rows if r.oom_recovery_override)
    oom = sum(1 for r in rows if r.oom)
    effective_rows = [
        r
        for r in rows
        if r.effective_success and not r.shared_vram_suspect and not r.oom_recovery_override and r.processing_fps > 0.0
    ]
    if effective_rows:
        fps_vals = [r.processing_fps for r in effective_rows]
        fps_median = statistics.median(fps_vals)
        fps_p10 = _quantile(fps_vals, 0.10)
        fps_p90 = _quantile(fps_vals, 0.90)
    else:
        fps_median = 0.0
        fps_p10 = 0.0
        fps_p90 = 0.0
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    lines.append("# FlashVSR VRAM Relationship Report")
    lines.append("")
    lines.append(f"- Generated: `{now}`")
    lines.append(f"- Source CSV: `{csv_path}`")
    lines.append(f"- Total cases: `{total}`")
    lines.append(f"- Raw successful cases: `{raw_success}`")
    lines.append(f"- Effective successful cases (shared-VRAM filtered): `{effective_success}`")
    lines.append(f"- Effective partial profile cases (timeout accepted): `{profile_partial}`")
    lines.append(f"- Shared-VRAM suspect cases: `{shared_suspects}`")
    lines.append(f"- OOM recovery override cases (invalid for strict tiling): `{oom_recovery_overrides}`")
    lines.append(f"- OOM cases: `{oom}`")
    lines.append("")
    lines.append("## Throughput Snapshot")
    lines.append("")
    lines.append(
        "- Effective processing FPS (`processing_fps` from logs): "
        f"median `{fps_median:.3f}`, P10 `{fps_p10:.3f}`, P90 `{fps_p90:.3f}`."
    )
    lines.append("")
    lines.append("## Sensitivity Estimates")
    lines.append("")
    lines.append("| Variable | Samples | Median | P10 | P90 |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        "| Tile size (+32 px) -> Delta VRAM (GB) | "
        f"{int(sensitivity['tile_32px_gb']['count'])} | "
        f"{sensitivity['tile_32px_gb']['median']:.4f} | "
        f"{sensitivity['tile_32px_gb']['p10']:.4f} | "
        f"{sensitivity['tile_32px_gb']['p90']:.4f} |"
    )
    lines.append(
        "| Frame chunk (+64 frames) -> Delta VRAM (GB) | "
        f"{int(sensitivity['chunk_64f_gb']['count'])} | "
        f"{sensitivity['chunk_64f_gb']['median']:.4f} | "
        f"{sensitivity['chunk_64f_gb']['p10']:.4f} | "
        f"{sensitivity['chunk_64f_gb']['p90']:.4f} |"
    )
    lines.append(
        "| Preprocess pixels (+1 MP) -> Delta VRAM (GB) | "
        f"{int(sensitivity['preprocess_1mp_gb']['count'])} | "
        f"{sensitivity['preprocess_1mp_gb']['median']:.4f} | "
        f"{sensitivity['preprocess_1mp_gb']['p10']:.4f} | "
        f"{sensitivity['preprocess_1mp_gb']['p90']:.4f} |"
    )
    lines.append("")
    lines.append("## Best Successful Tile By Scenario")
    lines.append("")
    lines.append("| GPU | Input | Scale | Max Edge | Chunk | Overlap | Best Tile | Peak VRAM (GB) | Preprocess | Output |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|---|")
    for item in best_tiles:
        lines.append(
            "| "
            f"{item['gpu_id']} | "
            f"{Path(str(item['input_path'])).name} | "
            f"{item['scale']} | "
            f"{item['max_target_resolution']} | "
            f"{item['frame_chunk_size']} | "
            f"{item['overlap']} | "
            f"{item['best_tile_size']} | "
            f"{float(item['peak_vram_gb']):.3f} | "
            f"{item['preprocess_resolution']} | "
            f"{item['output_resolution']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate FlashVSR VRAM parameter relationship report")
    parser.add_argument(
        "--records-csv",
        type=str,
        default=str(base_dir / "outputs" / "flashvsr_vram_sweeps" / "flashvsr_vram_records.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default=str(base_dir / "outputs" / "flashvsr_vram_sweeps" / "flashvsr_vram_report.md"),
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(base_dir / "outputs" / "flashvsr_vram_sweeps" / "flashvsr_vram_report.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    csv_path = Path(args.records_csv).resolve()
    out_md = Path(args.out_md).resolve()
    out_json = Path(args.out_json).resolve()

    rows = _load_rows(csv_path)
    if not rows:
        print(f"ERROR: no rows found in {csv_path}")
        return 2

    sensitivity = _build_sensitivity(rows)
    best_tiles = _best_tile_map(rows)
    md = _build_markdown(csv_path=csv_path, rows=rows, sensitivity=sensitivity, best_tiles=best_tiles)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "records_csv": str(csv_path),
        "total_cases": len(rows),
        "raw_successful_cases": sum(1 for r in rows if r.raw_success),
        "effective_successful_cases": sum(1 for r in rows if r.effective_success and not r.shared_vram_suspect),
        "effective_profile_partial_cases": sum(
            1 for r in rows if r.profile_partial and r.effective_success and not r.shared_vram_suspect
        ),
        "shared_vram_suspect_cases": sum(1 for r in rows if r.shared_vram_suspect),
        "oom_recovery_override_cases": sum(1 for r in rows if r.oom_recovery_override),
        "oom_cases": sum(1 for r in rows if r.oom),
        "sensitivity": sensitivity,
        "best_successful_tiles": best_tiles,
    }

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Report written: {out_md}")
    print(f"JSON written: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
