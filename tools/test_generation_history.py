"""Past Generations checks without a GPU, on a temporary outputs folder.

Covers: run folders (settings from run_summary.json, engine from run_context, helper files left out, status,
run time, comparison video), Chains runs (steps, collapsed model label, batch-report folders skipped), loose
DLSS 5 images with their settings from the journal, results outside the outputs folder found through the
journal, failed and unfinished runs, media probes and thumbnails (image and video), the cheap change signature,
favorites, details (run-internal paths hidden, groups), Reload parameters (run paths reset, a batch item
reloads as one file), Delete limited to the outputs folder, and the zip of a selection.
Run: venv/Scripts/python.exe tools/test_generation_history.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from shared import generation_history as gh  # noqa: E402

FAILS: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + (f" ({detail})" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def make_video(path: Path, size: str = "320x180") -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=24:duration=1",
                           "-pix_fmt", "yuv420p", "-c:v", "libx264", str(path)], capture_output=True)
    return proc.returncode == 0 and path.is_file()


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="gh_test_"))
    saved = (gh.DATA_DIR, gh.JOURNAL_PATH)
    gh.DATA_DIR = tmp / "history_data"
    gh.JOURNAL_PATH = gh.DATA_DIR / "journal.jsonl"
    try:
        out = tmp / "outputs"
        inputs = tmp / "inputs"
        inputs.mkdir(parents=True)
        src_img = inputs / "photo.png"
        Image.new("RGB", (160, 90), (40, 90, 160)).save(src_img)
        src_vid = inputs / "clip.mp4"
        have_video = make_video(src_vid, "160x90")

        # 1) a finished RelayVSR video run folder with its summary, a comparison and helper files
        run = out / "0007"
        res_vid = run / "clip.mp4"
        if have_video:
            make_video(res_vid, "640x360")
            shutil.copy2(res_vid, run / "clip_comparison.mp4")
            shutil.copy2(res_vid, run / "downscaled_clip.mp4")
        args = {"input_path": str(src_vid), "output_override": str(res_vid), "_user_output_override_raw": "",
                "wan_model": "Wan INT8 ConvRot (5 GB)", "window_size": 16, "seed": 42, "upscale_factor": 4.0,
                "resume_run_dir": str(run), "global_output_dir": str(run), "batch_enable": False}
        write_json(run / "run_context.json", {"created_at": "2026-11-06 07:15:23", "input_path": str(src_vid),
                                              "pipeline": "relayvsr", "last_updated_at": "2026-11-06 07:16:58",
                                              "last_status": "✅ RelayVSR upscaling complete", "last_returncode": 0,
                                              "last_output_path": str(res_vid)})
        write_json(run / "run_summary.json", [{"input": str(src_vid), "output": str(res_vid), "returncode": 0,
                                               "args": args, "pipeline": "relayvsr", "timestamp": "2026-11-06 07:16:58"}])

        # 2) a failed run without output, 3) an unfinished run (context only) that must be skipped
        write_json(out / "0008" / "run_context.json", {"created_at": "2026-11-06 08:00:00", "pipeline": "seedvr2",
                                                       "last_status": "❌ Out of memory", "last_returncode": 1})
        write_json(out / "0009" / "run_context.json", {"created_at": "2026-11-06 08:10:00", "model": "ltx25"})

        # 4) a Chains run: three DLSS 5 steps
        chain = out / "chains" / "0000"
        chain_final = chain / "photo_chain.png"
        chain.mkdir(parents=True)
        Image.new("RGB", (640, 360), (200, 120, 40)).save(chain_final)
        steps_out = []
        for i in range(1, 4):
            p = chain / f"step{i}_dlss5" / "0000" / f"photo_DLSS5_{i}.png"
            p.parent.mkdir(parents=True)
            Image.new("RGB", (160, 90), (i * 50, 20, 20)).save(p)
            write_json(p.parent / "run_context.json", {"pipeline": "dlss5"})  # step folders are not separate items
            steps_out.append(str(p))
        write_json(chain / "chain.json", {"name": "x3", "source": str(src_img), "steps": [{"engine": "dlss5", "enabled": True}] * 3,
                                          "started": "2026-11-06T09:00:00", "updated": "2026-11-06T09:01:00",
                                          "results": [{"index": i + 1, "engine": "dlss5", "status": "done",
                                                       "output": steps_out[i], "seconds": 10.0} for i in range(3)]})

        # 5) a Chains batch output folder (copies; must not be listed twice) and a DLSS 5 loose image
        batch_copy = out / "photos_chain"
        batch_copy.mkdir()
        shutil.copy2(chain_final, batch_copy / "photo_chain.png")
        write_json(batch_copy / "chain_batch_report.json", {"files": []})
        loose = out / "0003_photo_DLSS5.png"
        Image.new("RGB", (160, 90), (10, 200, 10)).save(loose)
        gh.record_generation(loose, src_img, "dlss5", {"input_path": str(src_img), "profile": "Neutral", "passes": 2})

        # 6) a result written outside the outputs folder (custom batch folder), found through the journal
        far = tmp / "elsewhere" / "far_DLSS5.png"
        far.parent.mkdir()
        Image.new("RGB", (100, 100), (90, 90, 90)).save(far)
        gh.record_generation(far, src_img, "dlss5", {"input_path": str(src_img), "batch_enable": True,
                                                     "batch_input_path": str(inputs)})

        h = gh.GenerationHistory(lambda: out)
        idx = h.index()
        if idx["pending"]:
            h._wait_pending(10)
            idx = h.index()
        items = {Path(i["out"]).name if i["out"] else Path(i["dir"]).name: i for i in idx["items"]}
        check("run folder, chain, loose, journal and failed items listed (unfinished and step folders skipped)",
              set(items) == ({"clip.mp4"} if have_video else set()) | {"0008", "photo_chain.png", "0003_photo_DLSS5.png",
                                                                        "far_DLSS5.png"}, str(sorted(items)))
        if have_video:
            it = items["clip.mp4"]
            check("run folder: engine, model, status, run time", it["engine"] == "relayvsr" and it["model"].startswith("Wan INT8")
                  and it["status"] == "ok" and abs((it["elapsed"] or 0) - 95.0) < 0.01, str(it))
            check("video probe: size, frames, fps, playable, scale", it["w"] == 640 and it["h"] == 360 and it["frames"] == 24
                  and abs(it["fps"] - 24) < 0.01 and it["playable"] and it["scale"] == 4.0, str(it))
            check("comparison video found, helper files not listed", bool(it["cmp_url"]) and "downscaled_clip.mp4" not in items)
        check("failed run without output listed as failed", items["0008"]["status"] == "failed" and items["0008"]["kind"] == "none")
        ch = items["photo_chain.png"]
        check("chain: engine, collapsed model label, steps", ch["engine"] == "chain" and ch["model"] == "DLSS 5 ×3"
              and ch["steps"] == 3 and ch["in_exists"], str(ch))
        lo = items["0003_photo_DLSS5.png"]
        check("loose DLSS 5 image with journal settings", lo["engine"] == "dlss5" and lo["has_args"] and lo["model"] == "Neutral",
              str(lo))
        check("result outside the outputs folder found via the journal", items["far_DLSS5.png"]["engine"] == "dlss5")

        # thumbnails
        th = h.thumbs([i["id"] for i in idx["items"] if i["out"]])
        check("thumbnails for every item with an output", all(th.values()) and len(th) == len([i for i in idx["items"] if i["out"]]),
              str(th))
        thumb_file = next(iter((gh.DATA_DIR / "thumbs").glob("*.jpg")))
        with Image.open(thumb_file) as im:
            check("thumbnail is a small JPEG", max(im.size) <= gh.THUMB_SIZE and im.format == "JPEG", str(im.size))

        # signature changes when a run appears
        sig = h.signature()
        write_json(out / "0010" / "run_context.json", {"created_at": "2026-11-06 10:00:00", "pipeline": "gan"})
        check("signature changes when a new run folder appears", h.signature() != sig)

        # details
        if have_video:
            d = h.details(items["clip.mp4"]["id"])
            check("details: run-internal paths hidden, groups present", "global_output_dir" not in d["args"]
                  and d["args"].get("output_override") == "" and "resume_run_dir" not in d["args"]
                  and any(g["name"] == "Model" for g in d["groups"]), json.dumps(d["args"])[:300])
            # reload
            rec = h.get(items["clip.mp4"]["id"])
            merged, n = gh.settings_for_reload(rec, {"input_path": "old", "output_override": "x", "resume_run_dir": "r",
                                                     "keep": 1}, ["input_path", "output_override", "resume_run_dir",
                                                                  "wan_model", "window_size", "seed", "batch_enable"])
            check("reload: values applied, run paths reset, other keys kept",
                  merged["wan_model"].startswith("Wan INT8") and merged["window_size"] == 16 and merged["seed"] == 42
                  and merged["output_override"] == "" and merged["resume_run_dir"] == "" and merged["keep"] == 1
                  and merged["input_path"] == str(src_vid), str(merged))
        far_rec = h.get(items["far_DLSS5.png"]["id"])
        merged, _ = gh.settings_for_reload(far_rec, {}, ["input_path", "batch_enable", "batch_input_path"])
        check("reload of one batch item: that file, batch off", merged["input_path"] == str(src_img)
              and merged["batch_enable"] is False, str(merged))
        dch = h.details(ch["id"])
        check("chain details list every step with a URL", len(dch["steps"]) == 3 and all(s["url"] for s in dch["steps"]))

        # favorites persist
        h.set_favorite([lo["id"]], True)
        h2 = gh.GenerationHistory(lambda: out)
        fav = {i["id"]: i["fav"] for i in h2.index()["items"]}
        check("favorites persist across instances", fav.get(lo["id"]) is True)

        # zip
        z = h.zip_selection([lo["id"], ch["id"]], tmp / "temp")
        zpath = Path(z["url"].split("file=", 1)[1].split("?", 1)[0])
        check("zip of a selection", z["count"] == 2 and sorted(zipfile.ZipFile(zpath).namelist()) ==
              ["0003_photo_DLSS5.png", "photo_chain.png"], str(z))

        # delete targets: inside the outputs folder the run folder; outside only the file (+ comparison)
        far_target = h._owned_target(far_rec)
        check("delete outside the outputs folder: only that file", far_target[0] is None and far_target[1] == [far])
        if have_video:
            run_target = h._owned_target(h.get(items["clip.mp4"]["id"]))
            check("delete inside the outputs folder: the run folder", run_target[0] == run.resolve())
        root_rec = dict(h.get(items["0008"]["id"]), dir=str(out))
        check("delete never takes the outputs folder itself", h._owned_target(root_rec)[0] is None)
        real_bin = gh._to_recycle_bin
        gh._to_recycle_bin = lambda _p: False  # keep test files out of the user's Recycle Bin: the app trash path
        try:
            res = h.delete([lo["id"]])
        finally:
            gh._to_recycle_bin = real_bin
        check("delete moves the item to the trash", res["removed"] == [lo["id"]] and not loose.exists()
              and any((gh.DATA_DIR / "trash").rglob("0003_photo_DLSS5.png")), str(res))
        check("engine names", gh.engine_key("rtx_super_resolution") == "rtx" and gh.engine_key("FlashVSR+") == "flashvsr"
              and gh.engine_key("open_h3") == "open_h3" and gh.engine_key("RIFE") == "rife" and gh.engine_key("DLSS5") == "dlss5")
    finally:
        gh.DATA_DIR, gh.JOURNAL_PATH = saved
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
