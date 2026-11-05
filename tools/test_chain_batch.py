"""Chains batch-folder checks without a GPU: a stand-in engine (copies its input) replaces the real ones.

Covers: listing (natural order, subfolders, the output folder inside the input folder left out), result names
and subfolders in the batch output folder, skip already-done files, the batch report, a failing file not
stopping the batch, quick-test results not counted as done, Cancel stopping the whole batch, Resume of the
files that never started, video-only steps on images, and multi-pass flow labels.
Run: venv/Scripts/python.exe tools/test_chain_batch.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from shared import chain_runner as cr  # noqa: E402

FAILS: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + (f" ({detail})" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def fake_engine(key: str, fail_on: str = "", cancel_after: list | None = None):
    calls: list = []

    def run_action(upload, *values, state=None, progress=None, global_settings_snapshot=None, **_kw):
        from shared.universal_preset import TAB_CONFIGS

        order = TAB_CONFIGS[key]["order"]
        settings = dict(zip(order, values))
        out_dir = Path(settings.get("output_override") or settings.get("standalone_output_override"))
        calls.append(Path(upload).name)
        if fail_on and fail_on in Path(upload).name:
            yield ("❌ stand-in failure", "stand-in engine failed on purpose")
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / (Path(upload).stem + f"_{key}" + Path(upload).suffix)
        shutil.copy2(upload, target)
        state["seed_controls"]["last_output_path"] = str(target)
        if cancel_after is not None:
            cancel_after.append(Path(upload).name)
        yield ("✅ done", f"wrote {target.name}")

    return run_action, calls


def make_media(folder: Path) -> None:
    (folder / "sub").mkdir(parents=True)
    for name in ("b.png", "a.png", "a10.png", "a2.jpg"):
        Image.new("RGB", (64, 36), (40, 80, 120)).save(folder / name)
    Image.new("RGB", (64, 36), (10, 20, 30)).save(folder / "sub" / "c.png")
    (folder / "notes.txt").write_text("not media", encoding="utf-8")
    (folder / "clip.mp4").write_bytes(b"\x00" * 64)  # listed as a video (never decoded by the stand-in)


def run(sources, steps, out_root, batch=None, cancel=None, test_seconds=0.0, resume=None):
    events = list(cr.run_chain(sources, steps, {"seed_controls": {}}, base_dir=ROOT, output_root=out_root,
                               chain_name="batch test", cancel_flag=cancel, batch=batch, test_seconds=test_seconds,
                               resume_dirs=resume))
    return events[-1]


def main() -> int:
    saved = dict(cr.ENGINES)
    tmp = Path(tempfile.mkdtemp(prefix="chain_batch_"))
    try:
        src = tmp / "in"
        src.mkdir()
        make_media(src)
        out_root = tmp / "outputs"

        files = [Path(f).name for f in cr.list_batch_sources(str(src))]
        check("list: media only, Explorer order", files == ["a2.jpg", "a10.png", "a.png", "b.png", "clip.mp4"], str(files))
        deep = [Path(f).name for f in cr.list_batch_sources(str(src), subfolders=True)]
        check("list: subfolders", deep == files + ["c.png"], str(deep))
        inside = src / "results"
        inside.mkdir()
        Image.new("RGB", (8, 8)).save(inside / "a_chain.png")
        listed = cr.list_batch_sources(str(src), subfolders=True, exclude=str(inside))
        check("list: output folder inside the input folder is left out", all("results" not in f for f in listed))
        check("default output", cr.default_batch_output(out_root, str(src)) == out_root / "in_chain")

        run_action, calls = fake_engine("dlss5", fail_on="a2")
        cr.register_chain_engine("dlss5", run_action)
        images = [f for f in cr.list_batch_sources(str(src), subfolders=True, exclude=str(inside))
                  if not f.endswith(".mp4")]
        batch = {"input": str(src), "output": str(tmp / "done"), "skip_done": True}
        last = run(images, [{"engine": "dlss5"}, {"engine": "dlss5"}], out_root, batch)
        names = sorted(p.relative_to(tmp / "done").as_posix() for p in (tmp / "done").rglob("*.png"))
        check("results named <name>_chain, subfolders kept",
              names == ["a10_chain.png", "a_chain.png", "b_chain.png", "sub/c_chain.png"], str(names))
        check("a failing file does not stop the batch", "Batch finished: 4 of 5" in last.status, last.status)
        report = json.loads((tmp / "done" / cr.BATCH_REPORT).read_text(encoding="utf-8"))
        statuses = {Path(e["source"]).name: e["status"] for e in report["files"]}
        check("report lists every file", statuses == {"a.png": "done", "a2.jpg": "failed", "a10.png": "done",
                                                      "b.png": "done", "c.png": "done"}, str(statuses))
        check("report keeps the failing step", "dlss5" in next(e["detail"] for e in report["files"] if e["status"] == "failed"))
        check("two DLSS 5 passes ran per file", calls.count("a.png") == 1 and calls.count("a_dlss5.png") == 1, str(calls))

        calls.clear()
        last = run(images, [{"engine": "dlss5"}], out_root, batch)
        check("second run skips finished files", calls == ["a2.jpg"], str(calls))
        check("status counts skipped files", "4 already done" in last.status, last.status)

        calls.clear()
        a_png = next(f for f in images if f.endswith("a.png"))
        run([a_png], [{"engine": "dlss5"}], out_root, {**batch, "output": str(tmp / "qt")}, test_seconds=2)
        qt = sorted(p.name for p in (tmp / "qt").glob("*.png"))
        check("quick test result is tagged and not counted as done",
              qt == ["a_chain_first2s.png"] and cr.batch_done(a_png, {**batch, "output": str(tmp / "qt")}) is None, str(qt))

        cancel_log: list = []
        run_action, calls = fake_engine("dlss5", cancel_after=cancel_log)
        cr.register_chain_engine("dlss5", run_action)
        cbatch = {"input": str(src), "output": str(tmp / "cancel"), "skip_done": True}
        last = run(images, [{"engine": "dlss5"}], out_root, cbatch, cancel=lambda: len(cancel_log) >= 2)
        started = [r for r in last.runs if r.get("run_dir")]
        not_started = [r for r in last.runs if not r.get("run_dir")]
        check("cancel stops the batch", len(calls) == 2 and len(not_started) == 3, f"calls={calls} runs={last.runs}")
        unfinished = [r for r in last.runs if not r.get("ok")]
        dirs = {r["source"]: r["run_dir"] for r in unfinished if r.get("run_dir")}
        cancel_log.clear()
        calls.clear()
        last = run([r["source"] for r in unfinished], [{"engine": "dlss5"}], out_root, cbatch,
                   cancel=lambda: False, resume=dirs)
        done_names = sorted(p.name for p in (tmp / "cancel").rglob("*_chain.*"))
        check("Resume finishes the files that never started", len(done_names) == 5 and "Batch complete" in last.status,
              f"{done_names} {last.status}")
        # a2 finished before Cancel; a10 was stopped by it and runs again with the 3 files never started.
        check("Resume does not redo finished files", len(started) == 2 and "a2.jpg" not in calls and len(calls) == 4, str(calls))

        run_action, calls = fake_engine("dlss5")
        cr.register_chain_engine("dlss5", run_action)
        rife_action, rife_calls = fake_engine("rife")
        cr.register_chain_engine("rife", rife_action)
        last = run([a_png], [{"engine": "rife"}, {"engine": "dlss5"}], out_root,
                   {"input": str(src), "output": str(tmp / "vo"), "skip_done": False})
        check("video-only step skipped on an image, chain goes on", rife_calls == [] and calls == ["a.png"]
              and (tmp / "vo" / "a_chain.png").is_file(), f"{rife_calls} {calls}")

        from ui.chain_tab import flow_html

        html_text = flow_html([{"engine": "dlss5"}] * 3, {})
        check("flow labels DLSS 5 passes", "pass 1/3" in html_text and "pass 3/3" in html_text and "DLSS 5 ×3" in html_text)
        check("built-in multi-pass chains", all(n in cr.BUILTIN_CHAINS for n in
                                                ("⭐ DLSS 5 ×3 (multi-pass, strongest look)", "⭐ SeedVR2 2x → DLSS 5 ×3")))
    finally:
        cr.ENGINES.clear()
        cr.ENGINES.update(saved)
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
