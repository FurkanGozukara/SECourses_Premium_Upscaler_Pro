"""Chain regression: build the real UI (no launch), then run a chain through the registered tab engines.

    venv\\Scripts\\python.exe tools/test_chain_runner.py <video> [--steps rtx:2,dlss5] [--max-dlss-frames 0]

Checks that every step produced its own output, the final file exists with the expected size
(the product of the scale steps), the frame count is kept and global post-processes stayed off
inside the steps. Also checks chain preset save / load / delete round trips.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402


class _Built(Exception):
    def __init__(self, demo):
        super().__init__("built")
        self.demo = demo


def build_demo():
    import secourses_app

    def _capture(self, *a, **k):
        raise _Built(self)

    with patch.object(gr.Blocks, "launch", _capture):
        try:
            secourses_app.main([])
        except _Built as built:
            return built.demo
    raise AssertionError("UI did not build")


def shared_state_value(demo) -> dict:
    for block in demo.blocks.values():
        if isinstance(block, gr.State) and isinstance(block.value, dict) and "seed_controls" in block.value:
            return copy.deepcopy(block.value)
    raise AssertionError("shared_state not found")


def probe(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=width,height,nb_read_frames", "-of", "json", path], capture_output=True, text=True).stdout
    return (json.loads(out or "{}").get("streams") or [{}])[0]


def preset_round_trip() -> bool:
    from shared import chain_runner as cr

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        name = cr.save_chain_preset(base, "Test chain", [{"engine": "seedvr2", "factor": 2}, {"engine": "dlss5"}])
        ok = name in cr.list_chain_presets(base)
        ok &= cr.load_chain_preset(base, name) == [{"engine": "seedvr2", "enabled": True, "factor": 2.0},
                                                   {"engine": "dlss5", "enabled": True}]
        ok &= cr.delete_chain_preset(base, name) and name not in cr.list_chain_presets(base)
        ok &= not cr.delete_chain_preset(base, next(iter(cr.BUILTIN_CHAINS)))
    print(f"[{'PASS' if ok else 'FAIL'}] chain preset save/load/delete")
    return bool(ok)


def postprocess_check(video: str, out_dir: Path) -> bool:
    """An upscaler tab run with 'Apply DLSS 5 after every upscale' on must end in a *_DLSS5 video."""
    from shared import chain_runner as cr

    engine = cr.ENGINES["rtx"]
    state = shared_state_value(_DEMO[0])
    seed = state["seed_controls"]
    seed.setdefault("dlss5_settings", {}).update({"global_postprocess": True, "max_frames": 0})
    seed.setdefault("output_settings", {})["frame_interpolation"] = False
    values = cr._tab_values(engine, state, __import__("shared.universal_preset", fromlist=["x"]).get_all_defaults(ROOT),
                            {"input_path": video, "output_override": str(out_dir), "batch_enable": False,
                             "upscale_factor": 2.0, "use_resolution_tab": False})
    for _payload in engine.run_action(video, *values, preview_only=False, state=state, progress=None):
        pass
    final = seed.get("last_output_path") or ""
    ok = bool(final) and Path(final).stem.endswith("_DLSS5") and Path(final).is_file()
    print(f"[{'PASS' if ok else 'FAIL'}] RTX upscale with DLSS 5 post-process -> {final}")
    return ok


_DEMO: list = []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--steps", default="rtx:2,dlss5")
    parser.add_argument("--out", default=str(ROOT / "temp" / "chain_test"))
    parser.add_argument("--postprocess", action="store_true", help="only run the post-process check")
    args = parser.parse_args()
    ok = preset_round_trip()
    demo = build_demo()
    _DEMO.append(demo)
    from shared import chain_runner as cr

    if args.postprocess:
        ok &= postprocess_check(args.video, Path(args.out) / "postprocess")
        print("ALL PASS" if ok else "SOME CHECKS FAILED")
        return 0 if ok else 1

    print("registered engines:", cr.available_engines())
    ok &= set(cr.ENGINE_ORDER) <= set(cr.ENGINES)
    steps = []
    for token in args.steps.split(","):
        engine, _, factor = token.partition(":")
        steps.append({"engine": engine, **({"factor": float(factor)} if factor else {})})
    state = shared_state_value(demo)
    seed = state["seed_controls"]
    # Globals ON in the live state: the chain must switch them off inside its steps.
    seed.setdefault("dlss5_settings", {})["global_postprocess"] = True
    seed.setdefault("output_settings", {})["frame_interpolation"] = True
    events = list(cr.run_chain([args.video], steps, state, base_dir=ROOT, output_root=Path(args.out), chain_name="test"))
    final = events[-1]
    for row in final.steps:
        print(f"  step {row.index} {row.engine}: {row.status} - {row.detail}")
    print("status:", final.status, "| final:", final.final_output)
    if not final.final_output:
        print("\n".join(final.log[-40:]))
        print("FAIL")
        return 1
    src, dst = probe(args.video), probe(final.final_output)
    scale = 1.0
    for s in steps:
        if s.get("factor") and cr.ENGINE_META[s["engine"]][2] == "scale":
            scale *= s["factor"]
    want = (round(int(src["width"]) * scale), round(int(src["height"]) * scale))
    got = (int(dst["width"]), int(dst["height"]))
    size_ok = abs(got[0] - want[0]) <= 8 and abs(got[1] - want[1]) <= 8
    frames_ok = int(dst.get("nb_read_frames") or 0) == int(src.get("nb_read_frames") or -1)
    rife_free = not any("xFPS" in Path(r.output or "").name for r in final.steps)
    dlss_once = sum("_DLSS5" in Path(r.output or "").stem for r in final.steps) <= sum(s["engine"] == "dlss5" for s in steps)
    print(f"[{'PASS' if size_ok else 'FAIL'}] size {got} (expected ~{want})")
    print(f"[{'PASS' if frames_ok else 'FAIL'}] frames {dst.get('nb_read_frames')} == {src.get('nb_read_frames')}")
    print(f"[{'PASS' if rife_free and dlss_once else 'FAIL'}] no global RIFE / extra DLSS 5 inside steps")
    ok &= size_ok and frames_ok and rife_free and dlss_once and all(r.status == "done" for r in final.steps)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
