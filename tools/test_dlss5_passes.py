"""DLSS 5 multi-pass checks (needs an RTX GPU).

1. Passes = 1 renders exactly what a single render did before multi-pass existed (bit-identical).
2. Passes = 3 on an image equals three single renders in a row (float hand-off between passes, so the
   difference comes only from rounding the two intermediate 8-bit images, mean below 1 level), and each pass moves
   further from the input (the effect gets stronger).
3. A video with Passes = 2 keeps its frame count, size and frame rate; the log names the passes.
Run: venv/Scripts/python.exe tools/test_dlss5_passes.py [image]
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.dlss5 import runner  # noqa: E402
from shared.dlss5.engine import DLSS5Session, to_array, to_tensor  # noqa: E402

FAILS: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + (f" ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> int:
    from PIL import Image

    image = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "test_media" / "test_image.png"
    frame = np.asarray(Image.open(image).convert("RGB"))
    h, w = frame.shape[:2]
    if max(h, w) > 1920:  # keep the test quick
        frame = np.asarray(Image.fromarray(frame).resize((w // 2, h // 2), Image.LANCZOS))
        h, w = frame.shape[:2]
    settings = {"profile": "Standard"}
    device = runner._resolve_device(settings)
    engine = runner._plan_engine(settings, ROOT, device, w, h, lambda *_: None)

    def single(arr):
        options = runner.build_options(settings)
        options.temporal = False
        return to_array(DLSS5Session(engine, options).render(to_tensor(arr, device)))

    one_old = single(frame)
    one_new = runner._render_still(engine, {**settings, "passes": 1}, frame, False)
    check("passes=1 is bit-identical to a single render", np.array_equal(one_old, one_new))

    chained = frame
    distances = []
    for _ in range(3):
        chained = single(chained)
        distances.append(float(np.abs(chained.astype(np.float32) - frame.astype(np.float32)).mean()))
    three = runner._render_still(engine, {**settings, "passes": 3}, frame, False)
    mae = float(np.abs(three.astype(np.float32) - chained.astype(np.float32)).mean())
    check("passes=3 equals three renders in a row", mae < 1.0, f"mean difference {mae:.3f} levels")
    check("each pass strengthens the effect", distances[0] < distances[1] < distances[2],
          " < ".join(f"{d:.2f}" for d in distances))
    check("passes clamp to 1..4", runner.passes_of({"passes": 9}) == 4 and runner.passes_of({"passes": 0}) == 1
          and runner.passes_of({}) == 1)

    with tempfile.TemporaryDirectory(prefix="dlss5_passes_") as tmp:
        clip = Path(tmp) / "clip.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=1",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        out = Path(tmp) / "clip_DLSS5.mp4"
        lines: list = []
        result = runner.run_dlss5({"input_path": str(clip), "output_path": str(out), "passes": 2, "keep_model_loaded": True},
                                  ROOT, on_progress=lines.append)
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                                "stream=width,height,nb_read_frames,r_frame_rate", "-of", "csv=p=0", str(out)],
                               capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
        check("video with 2 passes renders", result.returncode == 0 and out.is_file(),
              result.log[-300:] if result.returncode else "")
        check("video keeps size, rate and frame count", probe == "640,360,24/1,24", probe)
        check("log names the passes", any("2 passes" in line for line in lines))
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
