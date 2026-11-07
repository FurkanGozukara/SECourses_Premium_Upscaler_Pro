"""Past Generations "Reload parameters" and "Send output to" put the input into the engine tab and keep it there.

Seen in the tutorial recording on 7 November 2026 (V10.1): Reload parameters of a RelayVSR result opened the RelayVSR
tab, the Input Path first still showed the previous input and then went empty, although the run's input file exists.

Cause: the event cleared the tab's upload box with gr.update(value=None). Gradio 6.31 fires a File's change event for
every value the server sets, also None on a box that was already empty (checked in a headless browser). The tab's
"upload cleared" handler listened to that change event and emptied the Input Path; the tab's auto-sync then copied
all of the tab's old controls over the restored settings. And when the restored settings equalled those the tab
showed the last time it was opened, the tab-select sync skipped them (its signature does not see later edits), so
the previous input stayed.

These checks build the real interface without launching it and replay the browser's order of events with the app's
own handlers: the Past Generations event, then the change events of every upload box it sets (the cleared-upload
handlers and the auto-sync of the tab's settings), then the click on the tab (its select sync).
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
from gradio import helpers as gr_helpers  # noqa: E402

from shared import generation_history as gh  # noqa: E402
from shared.universal_preset import TAB_CONFIGS  # noqa: E402

ENGINE_TABS = ("seedvr2", "flashvsr", "relayvsr", "open_h3", "ltx25", "sparkvsr", "rtx", "dlss5", "gan", "rife")
# Tabs whose ✕ on the upload box empties the Input Path (the user's own clear must keep doing that).
CLEAR_EMPTIES_PATH = ("seedvr2", "flashvsr", "relayvsr", "open_h3", "ltx25", "sparkvsr", "rtx", "gan", "rife")


class _Built(Exception):
    def __init__(self, demo):
        super().__init__("UI built")
        self.demo = demo


def _build_demo():
    import secourses_app

    def _capture(self, *args, **kwargs):
        raise _Built(self)

    with patch.object(gr.Blocks, "launch", _capture):
        try:
            secourses_app.main([])
        except _Built as built:
            return built.demo
    raise AssertionError("secourses_app.main() returned without launching the UI")


def _walk(block):
    yield block
    for child in getattr(block, "children", None) or []:
        yield from _walk(child)


def _deps(demo, comp, event):
    return [d for d in demo.fns.values() if d.fn is not None
            and any(t[0] == comp._id and t[1] == event for t in (d.targets or []))]


def _is_update(value):
    return isinstance(value, dict) and value.get("__type__") == "update"


class _Page:
    """The browser side of one session: component values plus the session's shared state."""

    def __init__(self, demo, app):
        self.demo, self.app, self.ui = demo, app, {}

    def get(self, comp):
        return self.ui.get(comp._id)

    def call(self, dep, event_data=None):
        values = [self.ui.get(c._id) for c in dep.inputs]
        args, *_ = gr_helpers.special_args(dep.fn, values, None, event_data)
        result = dep.fn(*args)
        if len(dep.outputs) == 1:
            result = [result]
        elif _is_update(result):  # one gr.skip() for every output
            result = [result] * len(dep.outputs)
        changed = []
        for comp, value in zip(dep.outputs, result):
            if _is_update(value):
                if "value" not in value:
                    continue
                value = value["value"]
            if isinstance(comp, gr.File) or self.ui.get(comp._id) != value:
                changed.append(comp)
            self.ui[comp._id] = value
        return changed

    def settle(self, changed):
        """Run what the browser runs after server-set values: a set upload box fires change (Gradio 6.31, also for
        None on an empty box), an edited control of an engine tab fires that tab's auto-sync."""
        files = [c for c in changed if isinstance(c, gr.File)]
        for box in files:
            for dep in _deps(self.demo, box, "change"):
                writes = set(id(c) for c in dep.outputs)
                tab = self.app.tab_of_box.get(box._id)
                if tab and writes & set(id(c) for c in self.app.tabs[tab]["inputs"] + [self.app.shared_state]):
                    self.settle(self.call(dep))
        for tab, info in self.app.tabs.items():
            if any(c in info["inputs"] for c in changed if not isinstance(c, gr.File)):
                self.call(info["auto_sync"])

    def open_tab(self, tab):
        self.settle(self.call(self.app.tabs[tab]["select"]))

    def type_path(self, tab, path):
        info = self.app.tabs[tab]
        self.ui[info["input_path"]._id] = path
        self.settle([info["input_path"]])

    def upload(self, tab, path):
        info = self.app.tabs[tab]
        self.ui[info["box"]._id] = path
        self.ui[info["input_path"]._id] = path  # what the tab's upload handler writes
        self.settle([info["input_path"]])

    def press_clear(self, tab):
        """The user's ✕ on the upload box: the box becomes empty and fires clear and change."""
        box = self.app.tabs[tab]["box"]
        self.ui[box._id] = None
        changed = []
        for dep in _deps(self.demo, box, "clear"):
            changed += self.call(dep)
        self.settle(changed + [box])

    def history(self, action, rid, tab=""):
        evt = gr.EventData(None, {"value": {"action": action, "id": rid, "tab": tab}})
        changed = self.call(self.app.history_dep, evt)
        self.settle(changed)

    def settings(self, tab):
        return (self.ui[self.app.shared_state._id].get("seed_controls") or {}).get(f"{tab}_settings") or {}


class HistoryReloadInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import ui.history_tab as history_tab

        cls.demo = demo = _build_demo()
        fns = list(demo.fns.values())
        cls.history_dep = next(d for d in fns if d.name == "on_action")
        cls.shared_state = cls.history_dep.inputs[0]
        boxes = [c for c in cls.history_dep.outputs if isinstance(c, gr.File)]
        cls.tabs, cls.tab_of_box = {}, {}
        for key in ENGINE_TABS:
            tab = next(b for b in demo.blocks.values() if isinstance(b, gr.Tab) and b.elem_id == f"sec-tab-{key}")
            select = next(d for d in fns if d.name == "_sync" and any(t[0] == tab._id and t[1] == "select"
                                                                      for t in d.targets))
            inputs = list(select.outputs[:-3])
            inside = {b._id for b in _walk(tab)}
            box = next(b for b in boxes if b._id in inside)
            auto_sync = next(d for d in fns if d.name == "_sync_wrapper" and list(d.inputs[:-1]) == inputs)
            order = TAB_CONFIGS[key]["order"]
            cls.tabs[key] = {"select": select, "inputs": inputs, "box": box, "auto_sync": auto_sync,
                             "input_path": inputs[order.index("input_path")], "order": order}
            cls.tab_of_box[box._id] = key

        # one finished RelayVSR run in a temporary outputs folder, with settings that differ from the defaults
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="history_reload_")
        cls.tmp = Path(cls._tmpdir.name)
        cls.input_file = cls.tmp / "inputs" / "man_in_suit_960x540.mp4"
        run = cls.tmp / "outputs" / "0000"
        cls.output_file = run / "man_in_suit_960x540.mp4"
        for p in (cls.input_file, cls.output_file):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\0" * 1024)
        cls.args = {"input_path": str(cls.input_file), "output_override": "", "_user_output_override_raw": "",
                    "wan_model": TAB_CONFIGS["relayvsr"]["defaults_fn"]()["wan_model"], "window_size": 12,
                    "wan_gpu_blocks": 7, "seed": 1234, "upscale_factor": 4.0, "max_target_resolution": 3840,
                    "batch_enable": False, "resume_run_dir": str(run)}
        (run / "run_context.json").write_text(json.dumps(
            {"created_at": "2026-11-07 15:00:00", "input_path": str(cls.input_file), "pipeline": "relayvsr",
             "last_status": "✅ RelayVSR upscaling complete", "last_returncode": 0,
             "last_output_path": str(cls.output_file)}), encoding="utf-8")
        (run / "run_summary.json").write_text(json.dumps(
            [{"input": str(cls.input_file), "output": str(cls.output_file), "returncode": 0, "args": cls.args,
              "pipeline": "relayvsr", "timestamp": "2026-11-07 15:01:00"}]), encoding="utf-8")

        cls._saved_journal = gh.JOURNAL_PATH
        gh.JOURNAL_PATH = cls.tmp / "history_data" / "journal.jsonl"
        cls.history = history_tab._HISTORY
        cls._saved_root = cls.history.root_override
        cls.history.root_override = cls.tmp / "outputs"
        cls.rid = next(r["id"] for r in cls.history.scan().values() if r["out"] == str(cls.output_file))

    @classmethod
    def tearDownClass(cls):
        cls.history.root_override = cls._saved_root
        gh.JOURNAL_PATH = cls._saved_journal
        cls._tmpdir.cleanup()

    def _page(self, tab):
        page = _Page(self.demo, self)
        page.ui[self.shared_state._id] = copy.deepcopy(self.shared_state.value)
        page.open_tab(tab)  # the tab as the user last saw it
        return page

    def _value(self, page, tab, key):
        info = self.tabs[tab]
        return page.get(info["inputs"][info["order"].index(key)])

    def _assert_restored(self, page, label):
        self.assertEqual(page.get(self.tabs["relayvsr"]["input_path"]), str(self.input_file), label)
        self.assertEqual(self._value(page, "relayvsr", "window_size"), 12, label)
        self.assertEqual(self._value(page, "relayvsr", "wan_gpu_blocks"), 7, label)
        self.assertEqual(self._value(page, "relayvsr", "seed"), 1234, label)
        self.assertEqual(page.settings("relayvsr").get("input_path"), str(self.input_file), label)
        self.assertEqual(page.settings("relayvsr").get("window_size"), 12, label)

    def test_reload_after_a_typed_path(self):
        page = self._page("relayvsr")
        page.type_path("relayvsr", "E:/video_upscaler_examples/multi_scene_movie_960x540.mp4")
        page.history("reload", self.rid)
        page.open_tab("relayvsr")
        self._assert_restored(page, "typed path")

    def test_reload_when_the_tab_last_showed_these_settings(self):
        # The run was reloaded once and the tab opened with its settings; then another input was picked. Reloading
        # the same run again gives exactly the settings the tab synced last time: they must still reach the controls.
        page = self._page("relayvsr")
        page.history("reload", self.rid)
        page.open_tab("relayvsr")
        page.type_path("relayvsr", "E:/video_upscaler_examples/multi_scene_movie_960x540.mp4")
        page.history("reload", self.rid)
        page.open_tab("relayvsr")
        self._assert_restored(page, "same settings as at the last tab select")

    def test_reload_after_an_upload(self):
        page = self._page("relayvsr")
        page.upload("relayvsr", str(self.tmp / "gradio_cache" / "clip.mp4"))
        page.history("reload", self.rid)
        page.open_tab("relayvsr")
        self._assert_restored(page, "uploaded file")
        self.assertIsNone(page.get(self.tabs["relayvsr"]["box"]), "the upload box is cleared (it would win the run)")

    def test_send_output_to_every_engine_tab(self):
        for tab in ENGINE_TABS:
            for how in ("typed", "uploaded"):
                with self.subTest(tab=tab, how=how):
                    page = self._page(tab)
                    if how == "typed":
                        page.type_path(tab, str(self.tmp / "other.mp4"))
                    else:
                        page.upload(tab, str(self.tmp / "gradio_cache" / "clip.mp4"))
                    page.history("send", self.rid, tab)
                    page.open_tab(tab)
                    self.assertEqual(page.get(self.tabs[tab]["input_path"]), str(self.output_file))
                    self.assertEqual(page.settings(tab).get("input_path"), str(self.output_file))
                    self.assertIsNone(page.get(self.tabs[tab]["box"]))

    def test_users_clear_button_still_empties_the_input_path(self):
        for tab in CLEAR_EMPTIES_PATH:
            with self.subTest(tab=tab):
                page = self._page(tab)
                upload = str(self.tmp / "gradio_cache" / "clip.mp4")
                page.upload(tab, upload)
                self.assertEqual(page.get(self.tabs[tab]["input_path"]), upload)
                page.press_clear(tab)
                self.assertEqual(page.get(self.tabs[tab]["input_path"]), "")
                self.assertEqual(page.settings(tab).get("input_path"), "")


if __name__ == "__main__":
    unittest.main()
