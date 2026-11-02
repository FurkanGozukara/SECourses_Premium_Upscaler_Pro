"""Every processing button's queued job must read its settings at the click (V9.6).

Gradio runs a button's click listeners one after another and reads each listener's inputs only
when its turn comes. A server listener registered before the job therefore delayed the moment
the job read its settings, and a setting changed meanwhile (seconds while other jobs kept the
app busy) leaked into the queued job. Only browser-only listeners (no server function) may come
before the job. Builds the real UI without launching it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr

QUEUE_ID = "app_processing_queue"


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


class UpscaleClickOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = _build_demo()

    def test_queued_job_is_the_first_server_listener_of_its_button(self):
        functions = sorted(self.demo.fns.values(), key=lambda fn: fn._id)
        job_buttons = {
            target_id
            for fn in functions
            if fn.concurrency_id == QUEUE_ID
            for target_id, event in fn.targets
            if event == "click"
        }
        checked = []
        for button_id in sorted(job_buttons):
            listeners = [fn for fn in functions if (button_id, "click") in [tuple(t) for t in fn.targets]]
            first_job = next(i for i, fn in enumerate(listeners) if fn.concurrency_id == QUEUE_ID)
            before = [fn.name for fn in listeners[:first_job] if fn.fn is not None]
            label = getattr(self.demo.blocks.get(button_id), "value", button_id)
            self.assertEqual(before, [], f"button {label!r}: server listeners run before its job: {before}")
            checked.append(label)
        # SeedVR2, FlashVSR, Open-H3, LTX 2.5, SparkVSR, RTX, GAN, RIFE, Face (Upscale) and their previews.
        self.assertGreaterEqual(len(checked), 9, checked)

    def test_upscale_jobs_output_their_pre_run_signature(self):
        """Auto Transfer Output to Input reads the signature the job now captures as it starts."""
        functions = sorted(self.demo.fns.values(), key=lambda fn: fn._id)
        wrapped = [fn for fn in functions if fn.concurrency_id == QUEUE_ID and fn.name == "_run"]
        self.assertEqual(len(wrapped), 7, [fn.name for fn in wrapped])
        for fn in wrapped:
            last = fn.outputs[-1]
            self.assertIsInstance(last, gr.State)
            self.assertTrue(any(last in other.inputs for other in functions if other.trigger_after == fn._id),
                            f"{fn.name}#{fn._id}: no follow-up reads the signature state")


if __name__ == "__main__":
    unittest.main()
