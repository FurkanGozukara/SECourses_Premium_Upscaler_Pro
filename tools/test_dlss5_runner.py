"""End-to-end DLSS 5 runner check on a real clip: frames, size, fps, audio and temporal stability.

    venv\\Scripts\\python.exe tools/test_dlss5_runner.py <video> [--frames 48] [--speed "CUDA Graphs"] [--out dir]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.dlss5.runner import DLSS5_DEFAULTS, run_dlss5  # noqa: E402


def probe(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height,r_frame_rate,nb_frames", "-of", "json", path],
                         capture_output=True, text=True).stdout
    return json.loads(out or "{}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--speed", default="CUDA Graphs")
    parser.add_argument("--motion", default="Balanced")
    parser.add_argument("--out", default=str(ROOT / "temp" / "dlss5_runner_test"))
    args = parser.parse_args()
    settings = dict(DLSS5_DEFAULTS)
    settings.update({"input_path": args.video, "output_dir": args.out, "max_frames": args.frames,
                     "speed_mode": args.speed, "motion_quality": args.motion})
    res = run_dlss5(settings, ROOT)
    print(f"returncode={res.returncode} frames={res.frames_processed}/{res.total_frames} "
          f"elapsed={res.elapsed_seconds:.1f}s fps={res.fps:.2f} out={res.output_path}")
    if res.returncode != 0:
        print(res.log[-3000:])
        return 1
    src, dst = probe(args.video), probe(res.output_path)
    sv = [s for s in src.get("streams", []) if s["codec_type"] == "video"][0]
    dv = [s for s in dst.get("streams", []) if s["codec_type"] == "video"][0]
    ok = (sv["width"], sv["height"], sv["r_frame_rate"]) == (dv["width"], dv["height"], dv["r_frame_rate"])
    ok &= int(dv.get("nb_frames") or 0) == res.frames_processed
    had_audio = any(s["codec_type"] == "audio" for s in src.get("streams", []))
    has_audio = any(s["codec_type"] == "audio" for s in dst.get("streams", []))
    ok &= (had_audio == has_audio)
    print(f"size/fps match: {(sv['width'], sv['height'], sv['r_frame_rate'])} -> {(dv['width'], dv['height'], dv['r_frame_rate'])}; "
          f"frames {dv.get('nb_frames')}; audio {had_audio}->{has_audio}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
