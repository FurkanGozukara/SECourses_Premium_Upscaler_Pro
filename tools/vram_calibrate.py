"""Measure real engine VRAM for the Auto Tune planners (operator tool, not used by the app).

Each case runs one real upscale through the app's own runner (with the Save VRAM cap the
app applies) and records:
- per process: PyTorch peak reserved/allocated (tools/vram_profile_site/sitecustomize.py)
- whole process tree: Windows dedicated and shared GPU memory peaks (tools/gpu_process_memory.py)

Results append to benchmarks/vram/<engine>.jsonl. GPU 0 work takes the shared gpu0_turn lock.

    python tools/vram_calibrate.py seedvr2 --cases cases.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import threading
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from gpu_process_memory import ProcessGpuMemorySampler, process_tree_pids  # noqa: E402

PROFILE_SITE = REPO / "tools" / "vram_profile_site"
OUT_DIR = REPO / "benchmarks" / "vram"
WORK = REPO / "temp" / "vram_calibrate"
TURN_FILE = Path(r"C:/Users/Furkan/AppData/Local/gpu0_turn/gpu0_turn.py")
GPU0_LUID = os.environ.get("CALIB_GPU_LUID", "")


@contextlib.contextmanager
def gpu_turn(purpose: str):
    # CALIB_NO_TURN=1: the case runs on another GPU (GPU 1), so it must not hold GPU 0's turn.
    if not TURN_FILE.is_file() or os.environ.get("CALIB_NO_TURN") == "1":
        yield
        return
    ns: Dict[str, Any] = {}
    exec(TURN_FILE.read_text(encoding="utf-8"), ns)
    with ns["gpu0_turn"]("claude-upscaler-vram", purpose):
        yield


def make_clip(src: str, width: int, height: int, frames: int, name: str) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / f"{name}_{width}x{height}_{frames}f_cfr.mp4"
    if out.is_file():
        return out
    # Constant 30 fps timestamps (a looped cut of the source otherwise reports avg 23.7 fps).
    vf = f"fps=30,scale={width}:{height}:flags=lanczos,setsar=1"
    cmd = ["ffmpeg", "-y", "-v", "error", "-stream_loop", "-1", "-i", src, "-vf", vf, "-frames:v", str(frames),
           "-an", "-fps_mode", "cfr", "-r", "30", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    return out


def measure(case_name: str, run: Callable[[], Any], profile_path: Path) -> Dict[str, Any]:
    if profile_path.exists():
        profile_path.unlink()
    os.environ["SECOURSES_VRAM_PROFILE_OUT"] = str(profile_path)
    old_pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(PROFILE_SITE) + (os.pathsep + old_pp if old_pp else "")
    me = os.getpid()
    sampler = ProcessGpuMemorySampler(lambda: [p for p in process_tree_pids(me) if p != me],
                                      interval=0.15, luid=GPU0_LUID or None).start()
    t0 = time.time()
    err = None
    result = None
    spill = {"killed": False}
    stop_watch = threading.Event()

    def _watch_spill():
        # A spill into shared GPU memory is the failure this whole feature prevents: stop the
        # case at once instead of letting it crawl at sysmem speed.
        import psutil

        while not stop_watch.wait(1.0):
            if sampler.shared_growth_gb > 1.0 and not spill["killed"]:
                spill["killed"] = True
                for pid in process_tree_pids(me):
                    if pid != me:
                        with contextlib.suppress(Exception):
                            psutil.Process(pid).kill()

    watcher = threading.Thread(target=_watch_spill, daemon=True)
    watcher.start()
    try:
        result = run()
    except Exception as exc:  # recorded, the campaign continues
        err = f"{type(exc).__name__}: {exc}"
    finally:
        stop_watch.set()
        sampler.stop()
        if spill["killed"]:
            err = (err + "; " if err else "") + "SPILL: shared GPU memory grew > 1 GB, case killed"
        os.environ["PYTHONPATH"] = old_pp
        os.environ.pop("SECOURSES_VRAM_PROFILE_OUT", None)
    procs: List[Dict[str, Any]] = []
    if profile_path.exists():
        for line in profile_path.read_text(encoding="utf-8").splitlines():
            with contextlib.suppress(Exception):
                procs.append(json.loads(line))
    return {
        "case": case_name,
        "seconds": round(time.time() - t0, 1),
        "tree_peak_dedicated_gb": round(sampler.peak_dedicated_gb, 3),
        "tree_peak_shared_gb": round(sampler.peak_shared_gb, 3),
        "tree_shared_growth_gb": round(sampler.shared_growth_gb, 3),
        "processes": procs,
        "error": err,
        "result": result,
    }


# --------------------------------------------------------------------------- SeedVR2
def run_seedvr2_case(case: Dict[str, Any]) -> Dict[str, Any]:
    from shared.runner import Runner
    from shared.services.seedvr2_service import seedvr2_defaults

    model = case.get("model", "seedvr2_ema_7b_sharp_fp16.safetensors")
    clip = make_clip(case.get("src", "C:/upgraded_upscaler_test/aa_16x9_widescreen.mp4"),
                     int(case["in_w"]), int(case["in_h"]), int(case["frames"]), "seed")
    out_dir = WORK / "out_seedvr2"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(REPO, REPO / "temp", out_dir, telemetry_enabled=False)
    s = seedvr2_defaults(model, REPO)
    s.update({
        "input_path": str(clip), "output_override": str(out_dir), "output_format": "mp4",
        "dit_model": model, "cuda_device": "0", "save_metadata": False,
        "resolution": int(case["resolution"]), "max_resolution": int(case.get("max_resolution", 0)),
        "batch_size": int(case["batch_size"]), "uniform_batch_size": False,
        "blocks_to_swap": int(case.get("blocks_to_swap", 0)),
        "swap_io_components": bool(case.get("swap_io_components", False)),
        "vae_encode_tiled": bool(case.get("encode_tiled", False)),
        "vae_encode_tile_size": int(case.get("encode_tile", 1024)),
        "vae_encode_tile_overlap": int(case.get("encode_overlap", 128)),
        "vae_decode_tiled": bool(case.get("decode_tiled", False)),
        "vae_decode_tile_size": int(case.get("decode_tile", 1024)),
        "vae_decode_tile_overlap": int(case.get("decode_overlap", 128)),
        "int8_convrot": bool(case.get("int8", True)),
        "split_phase_subprocesses": bool(case.get("split", True)),
        "compile_dit": False, "compile_vae": False,
        "save_vram_gb": float(case.get("save_vram_gb", 1.5)),
        "load_cap": int(case.get("load_cap", 0)),
        "video_backend": "ffmpeg",
        "auto_tune_on_run": False,
    })
    s.update(case.get("extra", {}))
    log_lines: List[str] = []
    rr = runner.run_seedvr2(s, on_progress=lambda m: log_lines.append(str(m)))
    text = "".join(log_lines)
    oom = ("out of memory" in text.lower()) or ("outofmemory" in text.lower())
    cap_line = next((l for l in text.splitlines() if "PyTorch capped at" in l), "")
    return {"returncode": rr.returncode, "output": str(rr.output_path or ""), "oom": oom, "cap": cap_line.strip()[:300],
            "log_tail": text[-1500:] if rr.returncode != 0 else ""}


# --------------------------------------------------------------------------- FlashVSR+
def run_flashvsr_case(case: Dict[str, Any]) -> Dict[str, Any]:
    from shared.flashvsr_runner import run_flashvsr
    from shared.services.flashvsr_service import flashvsr_defaults

    clip = make_clip(case.get("src", "C:/upgraded_upscaler_test/aa_16x9_widescreen.mp4"),
                     int(case["in_w"]), int(case["in_h"]), int(case["frames"]), "flash")
    out_dir = WORK / "out_flashvsr"
    out_dir.mkdir(parents=True, exist_ok=True)
    s = flashvsr_defaults()
    s.update({
        "input_path": str(clip), "output_override": str(out_dir / f"{case['name']}.mp4"), "output_format": "mp4",
        "device": "0", "save_metadata": False, "scale": int(case.get("scale", 4)),
        "version": str(case.get("version", "1.1")), "mode": str(case.get("mode", "full")),
        "precision": str(case.get("precision", "bf16")), "attention_mode": str(case.get("attention", "flash_attention_2")),
        "frame_chunk_size": int(case.get("chunk", 0)), "tiled_dit": bool(case.get("tiled_dit", False)),
        "tile_size": int(case.get("tile", 256)), "overlap": int(case.get("overlap", 24)),
        "kv_ratio": float(case.get("kv_ratio", 3.0)), "tiled_vae": False,
        "keep_models_on_cpu": bool(case.get("keep_models_on_cpu", False)),
        "unload_dit": bool(case.get("unload_dit", False)), "force_offload": bool(case.get("force_offload", False)),
        "save_vram_gb": float(case.get("save_vram_gb", 1.5)), "auto_tune_on_run": False,
        "crf": 18, "codec": "libx264",
    })
    s.update(case.get("extra", {}))
    log_lines: List[str] = []
    rr = run_flashvsr(s, REPO, on_progress=lambda m: log_lines.append(str(m) + "\n"))
    text = "".join(log_lines) + str(getattr(rr, "log", "") or "")
    oom = ("out of memory" in text.lower()) or ("outofmemory" in text.lower())
    peaks = [ln.strip() for ln in text.splitlines() if "Peak VRAM" in ln][-4:]
    cap_line = next((ln for ln in text.splitlines() if "PyTorch capped at" in ln), "")
    return {"returncode": rr.returncode, "output": str(rr.output_path or ""), "oom": oom, "cap": cap_line.strip()[:300],
            "peak_lines": peaks, "log_tail": text[-1500:] if rr.returncode != 0 or oom else ""}


# --------------------------------------------------------------------------- SparkVSR
def run_sparkvsr_case(case: Dict[str, Any]) -> Dict[str, Any]:
    from shared.services.sparkvsr_service import sparkvsr_defaults
    from shared.sparkvsr_runner import run_sparkvsr

    clip = make_clip(case.get("src", "C:/upgraded_upscaler_test/aa_16x9_widescreen.mp4"),
                     int(case["in_w"]), int(case["in_h"]), int(case["frames"]), "spark")
    out_dir = WORK / "out_sparkvsr"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = case.get("model", "SparkVSR-bf16")
    s = sparkvsr_defaults(model)
    s.update({
        "input_path": str(clip), "output_override": str(out_dir / f"{case['name']}.mp4"), "output_format": "mp4",
        "device": "0", "save_metadata": False, "model_name": model, "scale": int(case.get("scale", 4)),
        "chunk_len": int(case.get("chunk_len", 0)), "overlap_t": int(case.get("overlap_t", 8)),
        "tile_height": int(case.get("tile", 0)), "tile_width": int(case.get("tile", 0)),
        "overlap_height": int(case.get("tile_overlap", 32)), "overlap_width": int(case.get("tile_overlap", 32)),
        "cpu_offload": bool(case.get("cpu_offload", True)), "vae_tiling": bool(case.get("vae_tiling", True)),
        "split_stage_subprocesses": bool(case.get("split", True)),
        "group_offload": bool(case.get("group_offload", False)), "num_blocks_per_group": int(case.get("blocks_per_group", 1)),
        "save_vram_gb": float(case.get("save_vram_gb", 1.5)), "auto_tune_on_run": False,
        "start_frame": 0, "end_frame": -1, "crf": 18, "codec": "libx264",
    })
    s.update(case.get("extra", {}))
    log_lines: List[str] = []
    rr = run_sparkvsr(s, REPO, on_progress=lambda m: log_lines.append(str(m) + "\n"))
    text = "".join(log_lines) + str(getattr(rr, "log", "") or "")
    oom = ("out of memory" in text.lower()) or ("outofmemory" in text.lower())
    cap_line = next((ln for ln in text.splitlines() if "PyTorch capped at" in ln), "")
    lines = text.splitlines()
    oom_ctx = []
    for i, ln in enumerate(lines):
        if "out of memory" in ln.lower() or "Auto Tune" in ln or "Group offload" in ln or "tiling" in ln.lower():
            oom_ctx.extend(lines[max(0, i - 3): i + 2])
    return {"returncode": rr.returncode, "output": str(getattr(rr, "output_path", "") or ""), "oom": oom,
            "cap": cap_line.strip()[:300], "context": oom_ctx[-60:],
            "log_tail": text[-2000:] if rr.returncode != 0 or oom else ""}


# --------------------------------------------------------------------------- RTX Super Resolution
_RTX_CHILD = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from shared.rtx_superres_runner import run_rtx_superres
settings = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
res = run_rtx_superres(settings, Path(sys.argv[1]), on_progress=lambda m: None)
print("RTX_RESULT", json.dumps({"returncode": int(res.returncode), "output": str(res.output_path or "")}))
'''


def run_rtx_case(case: Dict[str, Any]) -> Dict[str, Any]:
    clip = make_clip(case.get("src", "C:/upgraded_upscaler_test/aa_16x9_widescreen.mp4"),
                     int(case["in_w"]), int(case["in_h"]), int(case.get("frames", 30)), "rtx")
    out_dir = WORK / "out_rtx"
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "input_path": str(clip), "output_override": str(out_dir / f"{case['name']}.mp4"), "output_format": "mp4",
        "device": "0", "quality_preset": str(case.get("quality", "HIGHBITRATE_ULTRA")),
        "upscale_factor": float(case.get("scale", 4.0)), "max_resolution": int(case.get("max_resolution", 0)),
        "pre_downscale_then_upscale": False, "non_blocking_inference": True, "cuda_stream_ptr": 0,
        "codec": "libx264", "crf": 18, "save_metadata": False, "audio_codec": "copy",
        "auto_tune_on_run": False, "save_vram_gb": 0.0,
    }
    settings.update(case.get("extra", {}))
    cfg = WORK / f"rtx_{case['name']}.json"
    cfg.write_text(json.dumps(settings), encoding="utf-8")
    child = WORK / "rtx_child.py"
    child.write_text(_RTX_CHILD, encoding="utf-8")
    env = dict(os.environ)
    proc = subprocess.run([sys.executable, str(child), str(REPO), str(cfg)], capture_output=True, text=True,
                          env=env, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    out = (proc.stdout or "") + (proc.stderr or "")
    line = next((ln for ln in out.splitlines() if ln.startswith("RTX_RESULT ")), "")
    res = json.loads(line[len("RTX_RESULT "):]) if line else {"returncode": proc.returncode}
    res["oom"] = "out of memory" in out.lower()
    res["log_tail"] = out[-1500:] if res.get("returncode") else ""
    return res


# --------------------------------------------------------------------------- LTX 2.5 / Open-H3
def _run_engine(case: Dict[str, Any], defaults_fn, run_fn, prefix: str) -> Dict[str, Any]:
    clip = make_clip(case.get("src", "C:/upgraded_upscaler_test/aa_16x9_widescreen.mp4"),
                     int(case["in_w"]), int(case["in_h"]), int(case["frames"]), prefix)
    out_dir = WORK / f"out_{prefix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    s = defaults_fn()
    s.update({"input_path": str(clip), "output_override": str(out_dir / f"{case['name']}.mp4"), "device": "0",
              "save_metadata": False})
    s.update(case.get("extra", {}))
    log_lines: List[str] = []
    rr = run_fn(s, REPO, on_progress=lambda m: log_lines.append(str(m) + "\n"))
    text = "".join(log_lines) + str(getattr(rr, "log", "") or "")
    oom = ("out of memory" in text.lower()) or ("outofmemory" in text.lower())
    keep = [ln.strip() for ln in text.splitlines()
            if any(k in ln for k in ("VRAM limit", "Auto Tune", "placement", "token budget", "Text encoder", "streaming", "[Plan]", "peak VRAM"))]
    return {"returncode": int(getattr(rr, "returncode", 1)), "output": str(getattr(rr, "output_path", "") or ""),
            "oom": oom, "notes": keep[-14:], "log_tail": text[-2500:] if getattr(rr, "returncode", 1) != 0 or oom else ""}


def run_ltx25_case(case: Dict[str, Any]) -> Dict[str, Any]:
    from shared.ltx25_runner import run_ltx25
    from shared.services.ltx25_service import ltx25_defaults

    return _run_engine(case, ltx25_defaults, run_ltx25, "ltx")


def run_open_h3_case(case: Dict[str, Any]) -> Dict[str, Any]:
    from shared.open_h3_runner import run_open_h3
    from shared.services.open_h3_service import open_h3_defaults

    return _run_engine(case, open_h3_defaults, run_open_h3, "openh3")


ENGINES = {"seedvr2": run_seedvr2_case, "flashvsr": run_flashvsr_case, "sparkvsr": run_sparkvsr_case,
           "rtx": run_rtx_case, "ltx25": run_ltx25_case, "open_h3": run_open_h3_case}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=sorted(ENGINES))
    ap.add_argument("--cases", required=True, help="JSON file: list of case dicts (each needs 'name')")
    ap.add_argument("--only", default="", help="comma-separated case names")
    ap.add_argument("--skip-done", action="store_true")
    args = ap.parse_args()
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    only = {c.strip() for c in args.only.split(",") if c.strip()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"{args.engine}.jsonl"
    done = set()
    if args.skip_done and out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            with contextlib.suppress(Exception):
                rec = json.loads(line)
                if not rec.get("error") and (rec.get("result") or {}).get("returncode") == 0:
                    done.add(rec["case"])
    for case in cases:
        name = case["name"]
        if only and name not in only:
            continue
        if name in done:
            continue
        print(f"=== {args.engine} {name}", flush=True)
        if case.get("target_cap_gb"):
            # Boundary test: pick Save VRAM so the allocator cap lands on target_cap_gb now.
            from shared.vram_autotune.common import query_gpu_memory

            free = query_gpu_memory().get(0).free_gb
            overhead = float(case.get("overhead_gb", 1.2))
            case = dict(case, save_vram_gb=round(max(0.0, free - overhead - float(case["target_cap_gb"])), 2))
            print(f"    boundary: free {free:.2f} GB -> save_vram_gb {case['save_vram_gb']}", flush=True)
        saved_conf = os.environ.get("PYTORCH_ALLOC_CONF")
        if case.get("alloc_conf"):
            os.environ["PYTORCH_ALLOC_CONF"] = str(case["alloc_conf"])
        case_env = {k: str(v) for k, v in (case.get("env") or {}).items()}
        saved_env = {k: os.environ.get(k) for k in case_env}
        os.environ.update(case_env)
        try:
            with gpu_turn(f"VRAM calibration {args.engine} {name}"):
                rec = measure(name, lambda: ENGINES[args.engine](case), WORK / f"profile_{args.engine}.jsonl")
        finally:
            if saved_conf is None:
                os.environ.pop("PYTORCH_ALLOC_CONF", None)
            else:
                os.environ["PYTORCH_ALLOC_CONF"] = saved_conf
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        rec["params"] = case
        rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with out_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        res = rec.get("result") or {}
        peaks = [round(max((d.get("peak_reserved_gb", 0) for d in p.get("devices", [])), default=0), 2)
                 for p in rec["processes"]]
        print(f"    rc={res.get('returncode')} oom={res.get('oom')} {rec['seconds']}s tree={rec['tree_peak_dedicated_gb']} GB "
              f"shared_growth={rec['tree_shared_growth_gb']} proc_peaks={peaks} err={rec['error']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
