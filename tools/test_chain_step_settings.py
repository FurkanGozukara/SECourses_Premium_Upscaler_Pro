"""Chains: every step keeps its tab's settings from the moment it was added (no GPU; stand-in engines).

Covers: Add takes a copy of the tab (per-run paths left out) that later tab changes do not touch; the same engine
twice with different settings (Open-H3 Regenerate, then JonXL H3 Upscale with 6 steps) shows and runs as two
different steps; size preview per step; a setting newer than the step keeps its default; steps without settings
(built-in examples, older saved chains) still follow the tab; Duplicate / Retake; chain preset save (freezes steps
that followed their tab) / load round trip and old preset files; chain.json, Resume and Past Generations reload.
Run: venv/Scripts/python.exe tools/test_chain_step_settings.py
"""
from __future__ import annotations

import copy
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
from shared.open_h3_constants import OPENH3_WORKFLOW_JXL, OPENH3_WORKFLOW_REGEN  # noqa: E402
from shared.services.open_h3_service import open_h3_defaults  # noqa: E402

FAILS: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + (f" ({detail})" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def recording_engine(key: str):
    """A stand-in run_action that records the settings it was called with and copies its input."""
    calls: list = []

    def run_action(upload, *values, state=None, progress=None, global_settings_snapshot=None, **_kw):
        from shared.universal_preset import TAB_CONFIGS

        settings = dict(zip(TAB_CONFIGS[key]["order"], values))
        calls.append({"values": settings, "state_tab": dict(state["seed_controls"].get(f"{key}_settings") or {})})
        out_dir = Path(settings["output_override"])
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{Path(upload).stem}_{key}{len(calls)}{Path(upload).suffix}"
        shutil.copy2(upload, target)
        state["seed_controls"]["last_output_path"] = str(target)
        yield ("✅ done", f"wrote {target.name}")

    return run_action, calls


def main() -> int:
    regen = {**open_h3_defaults(), "workflow": OPENH3_WORKFLOW_REGEN, "resolution_preset": "1080p",
             "model_name": "FL2VA Pruned W4A8", "input_path": "D:/in/clip.mp4", "output_override": "D:/out",
             "batch_input_path": "D:/batch", "_multi_gpu_disabled_reason": "x"}
    state = {"seed_controls": {"open_h3_settings": copy.deepcopy(regen)}}

    # 1) Add: a copy of the tab, per-run fields left out, independent of later tab changes
    step1 = cr.new_step("open_h3", state)
    check("Add keeps the tab's settings", step1["settings"]["workflow"] == OPENH3_WORKFLOW_REGEN
          and step1["settings"]["resolution_preset"] == "1080p")
    check("per-run fields and private keys are not kept", not ({"input_path", "output_override", "batch_input_path",
                                                               "_multi_gpu_disabled_reason"} & set(step1["settings"])))
    state["seed_controls"]["open_h3_settings"].update({"workflow": OPENH3_WORKFLOW_JXL, "jxl_steps": 6,
                                                       "model_name": "FL2VA Pruned INT8 ConvRot"})
    step2 = cr.new_step("open_h3", state)
    check("a later tab change leaves the first step alone", step1["settings"]["workflow"] == OPENH3_WORKFLOW_REGEN
          and step1["settings"]["model_name"] == "FL2VA Pruned W4A8")
    check("the second step has the new settings", step2["settings"]["workflow"] == OPENH3_WORKFLOW_JXL
          and step2["settings"]["jxl_steps"] == 6)
    steps = [step1, step2]

    # 2) Cards and sizes show each step's own settings
    from ui import chain_tab

    s1, s2 = (chain_tab._step_summary("open_h3", state, s) for s in steps)
    check("card 1 shows Regenerate", s1 == "FL2VA Pruned W4A8 · 1080p", s1)
    check("card 2 shows JonXL with its steps", s2 == "JonXL H3 Upscale · FL2VA Pruned INT8 ConvRot · 6 steps", s2)
    flow = chain_tab.flow_html(steps, state, {"width": 960, "height": 540, "kind": "image"})
    check("flow shows both settings", "W4A8 · 1080p" in flow and "6 steps" in flow and flow.count("JonXL") == 1)
    sizes = cr.project_sizes(steps, state, 960, 540, "image")
    check("sizes follow each step", sizes == [(1920, 1080), (1920, 1080)], str(sizes))
    sizes = cr.project_sizes(steps, state, 480, 270, "image")
    check("JonXL step fits its own max size", sizes[1] == (1920, 1080), str(sizes))
    legacy = {"engine": "open_h3", "enabled": True}
    check("a step without settings follows the tab (marked)",
          chain_tab._step_summary("open_h3", state, legacy).endswith("· follows the tab")
          and "JonXL" in chain_tab._step_summary("open_h3", state, legacy))

    # 3) normalize / duplicate / retake / freeze
    norm = cr.normalize_steps(steps)
    check("normalize keeps settings", norm[0]["settings"] == step1["settings"] and norm[1]["settings"] == step2["settings"])
    norm[0]["settings"]["workflow"] = "changed"
    check("normalize copies (no shared dicts)", step1["settings"]["workflow"] == OPENH3_WORKFLOW_REGEN)
    check("old steps normalize unchanged", cr.normalize_steps([{"engine": "seedvr2", "factor": 2}]) ==
          [{"engine": "seedvr2", "enabled": True, "factor": 2.0}])
    frozen = cr.freeze_steps([legacy, step1], state)
    check("freeze fills a step that followed its tab, keeps own settings",
          frozen[0]["settings"]["workflow"] == OPENH3_WORKFLOW_JXL and frozen[1]["settings"] == step1["settings"])
    check("freeze with no tab settings leaves the step as it was", cr.freeze_steps([legacy], {}) == [legacy])

    # 4) chain presets: save / load keeps every step's settings; old files load as before
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        name = cr.save_chain_preset(base, "Regen then JonXL", steps)
        loaded = cr.load_chain_preset(base, name)
        check("preset round trip keeps both steps' settings", loaded == cr.normalize_steps(steps))
        saved = json.loads((cr.chains_dir(base) / f"{name}.json").read_text(encoding="utf-8"))
        check("preset file has no input / output paths", "input_path" not in json.dumps(saved)
              and "D:/out" not in json.dumps(saved))
        check("chain label = the preset name", cr.chain_label(loaded, name, base) == name)
        changed = copy.deepcopy(loaded)
        changed[1]["settings"]["jxl_steps"] = 9
        check("other settings = not that preset", cr.chain_label(changed, name, base) != name)
        (cr.chains_dir(base) / "Old.json").write_text(json.dumps({"name": "Old", "steps": [
            {"engine": "open_h3", "enabled": True}, {"engine": "dlss5", "enabled": True}]}), encoding="utf-8")
        old = cr.load_chain_preset(base, "Old")
        check("an old preset loads without settings (follows the tabs)", old == [
            {"engine": "open_h3", "enabled": True}, {"engine": "dlss5", "enabled": True}])
        builtin = cr.load_chain_preset(base, cr.DEFAULT_CHAIN)
        check("built-in examples unchanged", all(not cr.has_own_settings(s) for s in builtin) and len(builtin) == 3)

    # 5) run: each step runs with its own settings, whatever the tab holds now
    from shared.universal_preset import get_all_defaults

    defaults = get_all_defaults(ROOT)
    engine = cr.ChainEngine("open_h3", "Open-H3", "🌀", lambda *a, **k: None)
    state["seed_controls"]["open_h3_settings"].update({"workflow": OPENH3_WORKFLOW_REGEN, "jxl_steps": 30,
                                                       "resolution_preset": "2K (1440p)"})
    run_values = cr._tab_values(engine, state, defaults, {"input_path": "x.png"}, step2["settings"])
    from shared.services.open_h3_service import OPENH3_ORDER

    as_dict = dict(zip(OPENH3_ORDER, run_values))
    check("run values = the step's settings, not the tab's", as_dict["workflow"] == OPENH3_WORKFLOW_JXL
          and as_dict["jxl_steps"] == 6 and as_dict["input_path"] == "x.png")
    no_new_key = {k: v for k, v in step2["settings"].items() if k != "jxl_steps"}
    as_dict = dict(zip(OPENH3_ORDER, cr._tab_values(engine, state, defaults, {}, no_new_key)))
    check("a setting newer than the step keeps its default", as_dict["jxl_steps"] == 20, str(as_dict["jxl_steps"]))
    as_dict = dict(zip(OPENH3_ORDER, cr._tab_values(engine, state, defaults, {})))
    check("a step without settings runs the tab's settings", as_dict["jxl_steps"] == 30
          and as_dict["workflow"] == OPENH3_WORKFLOW_REGEN)

    saved_engines = dict(cr.ENGINES)
    try:
        run_action, calls = recording_engine("open_h3")
        cr.ENGINES.clear()
        cr.ENGINES["open_h3"] = cr.ChainEngine("open_h3", "Open-H3", "🌀", run_action, "upload", {}, True, True,
                                               "scale", ("model_name", "resolution_preset"))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "in.png"
            Image.new("RGB", (64, 36), (90, 120, 150)).save(src)
            events = list(cr.run_chain([str(src)], steps + [legacy], state, base_dir=ROOT, output_root=tmp / "out"))
            final = events[-1]
            check("chain finished", final.finished and final.final_output and Path(final.final_output).is_file(),
                  final.status)
            got = [(c["values"]["workflow"], c["values"]["jxl_steps"], c["values"]["resolution_preset"]) for c in calls]
            check("step 1 ran Regenerate 1080p, step 2 JonXL 6 steps, step 3 (no settings) the tab", got == [
                (OPENH3_WORKFLOW_REGEN, 20, "1080p"), (OPENH3_WORKFLOW_JXL, 6, "1080p"),
                (OPENH3_WORKFLOW_REGEN, 30, "2K (1440p)")], str(got))
            check("a step's state copy carries its settings", calls[1]["state_tab"].get("workflow") == OPENH3_WORKFLOW_JXL
                  and calls[1]["state_tab"].get("jxl_steps") == 6)
            check("the live tab settings are untouched", state["seed_controls"]["open_h3_settings"]["jxl_steps"] == 30)
            check("the log names each step's settings", "settings: JonXL H3 Upscale · FL2VA Pruned INT8 ConvRot · 6 steps"
                  in "\n".join(final.log) and "(current tab settings)" in "\n".join(final.log))
            run_dir = Path(final.run_dir)
            chain_json = json.loads((run_dir / "chain.json").read_text(encoding="utf-8"))
            check("chain.json keeps each step's settings", chain_json["steps"][1]["settings"]["jxl_steps"] == 6
                  and "settings" not in chain_json["steps"][2])
            reload_steps = cr.normalize_steps(chain_json["steps"])  # what Past Generations' Reload puts in the chain
            check("Past Generations reload brings the settings back", reload_steps == cr.normalize_steps(steps + [legacy]))
            _input, reused = cr._resume_plan(run_dir, cr.normalize_steps(steps + [legacy]))
            check("Resume reuses steps with the same settings", len(reused) == 3)
            other = cr.normalize_steps(steps + [legacy])
            other[1]["settings"]["jxl_steps"] = 8
            _input, reused = cr._resume_plan(run_dir, other)
            check("Resume redoes a step whose settings changed", len(reused) == 1)
    finally:
        cr.ENGINES.clear()
        cr.ENGINES.update(saved_engines)

    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
