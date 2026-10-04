"""Opt-in GPU smoke campaign: every image-capable model on awkward image files.

Some files decode differently than an ordinary 8-bit RGB PNG: 16-bit PNG/TIFF
(OpenCV returns 0-65535), 16-bit grayscale (PIL clips it to white), float TIFF,
palette, CMYK JPEG, alpha, and non-ASCII file names (OpenCV cannot open them on
Windows). A loader that mishandles one of them produces a white, black or failed
upscale. This tool runs the real model runners on each case, shrinks every output
back to the input size and compares it with the 8-bit reference.

Like ``actual_model_smoke.py`` it downloads/loads real models, so it is not a
``test_*.py`` file. Example::

    python tools/image_input_smoke.py --models seedvr2,gan,flashvsr,sparkvsr,rtx
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.runner import Runner  # noqa: E402
from tools.actual_model_smoke import ModelAdapter, _common_settings, _settings_for  # noqa: E402

IMAGE_MODELS = ("seedvr2", "gan", "flashvsr", "sparkvsr", "rtx")
MIN_PSNR_DB = 20.0


def _read_any(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags) if data.size else None


def _write_any(path: Path, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    buf.tofile(str(path))


def _synthetic_master(width: int, height: int) -> np.ndarray:
    """16-bit BGR test card: smooth gradients, dark areas and saturated shapes."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    b = 0.15 + 0.35 * (xx / max(1, width - 1))
    g = 0.05 + 0.25 * (yy / max(1, height - 1))
    r = 0.10 + 0.20 * np.sin(xx / 17.0) * np.cos(yy / 23.0) + 0.1
    img = np.clip(np.dstack([b, g, r]), 0.0, 1.0)
    cv2.circle(img, (width // 3, height // 2), height // 4, (0.95, 0.85, 0.1), -1)
    cv2.rectangle(img, (width // 2, height // 4), (width - width // 8, height - height // 4), (0.2, 0.1, 0.9), -1)
    cv2.putText(img, "16BIT", (width // 2 + 6, height // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (1.0, 1.0, 1.0), 2)
    return (img * 65535.0 + 0.5).astype(np.uint16)


def build_corpus(out_dir: Path, source: Optional[Path], width: int, height: int) -> Dict[str, Dict[str, Any]]:
    """Write every awkward case next to its 8-bit BGR reference."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if source is not None:
        master = _read_any(source)
        if master is None:
            raise RuntimeError(f"Cannot read source image {source}")
        if master.ndim == 2:
            master = cv2.cvtColor(master, cv2.COLOR_GRAY2BGR)
        master = master[..., :3]
        if master.dtype == np.uint8:
            master = master.astype(np.uint16) * 257
        elif master.dtype != np.uint16:
            master = (np.clip(master.astype(np.float32), 0, 1) * 65535.0 + 0.5).astype(np.uint16)
        master = cv2.resize(master, (width, height), interpolation=cv2.INTER_AREA)
    else:
        master = _synthetic_master(width, height)

    rgb8 = (master.astype(np.float32) / 257.0 + 0.5).astype(np.uint8)
    gray16 = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)
    gray8 = (gray16.astype(np.float32) / 257.0 + 0.5).astype(np.uint8)
    gray8_bgr = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    alpha16 = np.full(gray16.shape, 65535, np.uint16)
    alpha16[:, : width // 4] = 0

    cases: Dict[str, Dict[str, Any]] = {}

    def add(name: str, writer: Callable[[Path], None], reference: np.ndarray) -> None:
        path = out_dir / name
        writer(path)
        cases[name] = {"path": path, "reference": reference}

    add("rgb8.png", lambda p: _write_any(p, rgb8), rgb8)
    add("rgb16.png", lambda p: _write_any(p, master), rgb8)
    add("gray16.png", lambda p: _write_any(p, gray16), gray8_bgr)
    add("rgba16.png", lambda p: _write_any(p, np.dstack([master, alpha16])), rgb8)
    add("gray8.png", lambda p: _write_any(p, gray8), gray8_bgr)
    add(
        "gray_alpha8.png",
        lambda p: Image.fromarray(np.dstack([gray8, (alpha16 >> 8).astype(np.uint8)]), "LA").save(p),
        gray8_bgr,
    )
    palette_rgb = np.asarray(Image.fromarray(cv2.cvtColor(rgb8, cv2.COLOR_BGR2RGB)).quantize(256).convert("RGB"))
    add(
        "palette8.png",
        lambda p: Image.fromarray(cv2.cvtColor(rgb8, cv2.COLOR_BGR2RGB)).quantize(256).save(p),
        cv2.cvtColor(palette_rgb, cv2.COLOR_RGB2BGR),
    )
    add("rgb16.tif", lambda p: _write_any(p, master), rgb8)
    add("rgb32f.tif", lambda p: _write_any(p, master.astype(np.float32) / 65535.0), rgb8)
    add(
        "cmyk.jpg",
        lambda p: Image.fromarray(cv2.cvtColor(rgb8, cv2.COLOR_BGR2RGB)).convert("CMYK").save(p, quality=95),
        rgb8,
    )
    add("görsel_çğüşıö_16bit.png", lambda p: _write_any(p, master), rgb8)
    return cases


def score_output(output_path: Optional[str], reference: np.ndarray) -> Dict[str, Any]:
    if not output_path or not Path(output_path).is_file():
        return {"ok": False, "reason": f"no output file ({output_path})"}
    out = _read_any(Path(output_path))
    if out is None:
        return {"ok": False, "reason": f"output unreadable ({output_path})"}
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    out = out[..., :3]
    if out.dtype != np.uint8:
        scale = 257.0 if out.dtype == np.uint16 else 1.0 / 255.0
        out = np.clip(out.astype(np.float32) / scale + 0.5, 0, 255).astype(np.uint8)
    h, w = reference.shape[:2]
    small = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
    diff = small.astype(np.float32) - reference.astype(np.float32)
    mse = float(np.mean(diff * diff))
    psnr = 99.0 if mse <= 1e-9 else float(10.0 * np.log10(255.0 * 255.0 / mse))
    result = {
        "ok": bool(psnr >= MIN_PSNR_DB),
        "psnr_db": round(psnr, 2),
        "mean_out": round(float(small.mean()), 2),
        "mean_ref": round(float(reference.mean()), 2),
        "output_size": [int(out.shape[1]), int(out.shape[0])],
        "output_path": str(output_path),
    }
    if not result["ok"]:
        result["reason"] = f"PSNR {psnr:.1f} dB < {MIN_PSNR_DB} dB (blown out, wrong colors or garbled)"
    return result


def _image_settings(model: str, input_path: Path, case_dir: Path) -> Dict[str, Any]:
    settings = _settings_for(model, _common_settings(input_path, case_dir))
    if model in ("flashvsr", "sparkvsr"):
        override = case_dir / f"{input_path.stem}.mp4"  # what their services pass for images too
    elif model == "gan":
        override = case_dir / "gan_output.png"  # explicit file: the runner moves the result here
    else:
        override = case_dir  # SeedVR2 and RTX services pass the run folder
    settings.update(
        output_format="png",
        image_output_format="png",
        output_override=str(override),
        # Keep model inputs as they are; the resize step has its own unit tests.
        pre_downscale_then_upscale=False,
        upscale_factor=2.0,
    )
    if model == "seedvr2":
        settings.update(_original_filename=input_path.name)
    return settings


def run_case(adapter: ModelAdapter, model: str, case_name: str, case: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    case_dir = work_dir / model / Path(case_name).stem
    shutil.rmtree(case_dir, ignore_errors=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    settings = _image_settings(model, case["path"], case_dir)
    log_lines: List[str] = []
    started = time.time()
    try:
        result = getattr(adapter, f"run_{'rtx_superres' if model == 'rtx' else model}")(
            settings, on_progress=lambda msg: log_lines.append(str(msg))
        )
        returncode = int(getattr(result, "returncode", 1))
        output_path = getattr(result, "output_path", None)
        log_text = str(getattr(result, "log", "") or "")
    except Exception as exc:  # report and continue with the next case
        returncode, output_path, log_text = 1, None, f"{type(exc).__name__}: {exc}"
    (case_dir / "run.log").write_text("".join(log_lines) + "\n" + log_text, encoding="utf-8", errors="replace")
    scored = score_output(output_path, case["reference"]) if returncode == 0 else {
        "ok": False,
        "reason": f"returncode {returncode}: {log_text.strip().splitlines()[-1][:200] if log_text.strip() else 'see run.log'}",
    }
    scored["seconds"] = round(time.time() - started, 1)
    return scored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=",".join(IMAGE_MODELS))
    parser.add_argument("--cases", default="", help="Comma-separated case file names (default: all)")
    parser.add_argument("--source", default="", help="Optional source image (any depth); default is a synthetic card")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--work-dir", default=str(REPO_ROOT / "temp" / "image_input_smoke"))
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    cases = build_corpus(work_dir / "corpus", Path(args.source) if args.source else None, args.width, args.height)
    wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
    if wanted:
        cases = {k: v for k, v in cases.items() if k in wanted}

    runner = Runner(REPO_ROOT, REPO_ROOT / "temp", work_dir, telemetry_enabled=False)
    adapter = ModelAdapter(runner)
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    report: Dict[str, Dict[str, Any]] = {}
    failures = 0
    for model in models:
        report[model] = {}
        for case_name, case in cases.items():
            print(f"=== {model} :: {case_name}", flush=True)
            res = run_case(adapter, model, case_name, case, work_dir)
            report[model][case_name] = res
            failures += 0 if res["ok"] else 1
            status = "PASS" if res["ok"] else "FAIL"
            detail = f"PSNR {res['psnr_db']} dB, mean {res['mean_out']} vs {res['mean_ref']}" if "psnr_db" in res else res.get("reason", "")
            print(f"    {status} {detail} ({res['seconds']} s)", flush=True)

    report_path = work_dir / "image_input_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSummary (PSNR dB of output shrunk to input size vs 8-bit reference):")
    names = list(cases)
    print("case".ljust(28) + "".join(m.ljust(12) for m in models))
    for name in names:
        row = name.ljust(28)
        for model in models:
            res = report[model][name]
            row += (f"{res['psnr_db']:.1f}" if res["ok"] else ("FAIL " + (f"{res['psnr_db']:.1f}" if "psnr_db" in res else "err"))).ljust(12)
        print(row)
    print(f"\nReport: {report_path}\n{failures} failing case(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
