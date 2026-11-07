"""Settings that nothing reads are hidden, yet presets that hold them still load (V10.1 fixes).

Output & Comparison: 'Default Zoom Level' and 'Show Difference Overlay' are hidden, and Comparison Mode no longer
offers 'overlay' (no overlay view exists; it showed the slider). A preset saved with 'overlay' loads as 'slider'.
Metadata & Logging: 'Metadata Format' and 'Log Verbosity' are hidden (metadata is always JSON).
The UI checks build the real interface without launching it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from shared.services.output_service import (  # noqa: E402
    COMPARISON_MODE_CHOICES,
    _normalize_output_fields,
    normalize_comparison_mode,
    output_defaults,
)
from shared.universal_preset import (  # noqa: E402
    _normalize_output_settings,
    dict_to_values,
    merge_preset_with_defaults,
    TAB_CONFIGS,
)


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


def _top_tab(demo, label):
    return next(b for b in demo.blocks.values() if isinstance(b, gr.Tab) and b.label == label)


def _by_label(root, label):
    found = [b for b in _walk(root) if isinstance(b, gr.components.Component) and getattr(b, "label", None) == label]
    if len(found) != 1:
        raise AssertionError(f"expected one component labelled {label!r}, found {len(found)}")
    return found[0]


def _choice_values(dropdown):
    return [c[1] if isinstance(c, (tuple, list)) else c for c in dropdown.choices]


def _wired_both_ways(demo, component):
    """The component is still an input (preset save) and an output (preset load / reset) of server events."""
    fns = list(demo.fns.values())
    return (any(component in fn.inputs for fn in fns if fn.fn is not None),
            any(component in fn.outputs for fn in fns if fn.fn is not None))


class OutputPresetLoadTests(unittest.TestCase):
    def test_overlay_maps_to_slider(self):
        self.assertEqual(normalize_comparison_mode("overlay"), "slider")
        self.assertEqual(normalize_comparison_mode(" Overlay "), "slider")
        for mode in COMPARISON_MODE_CHOICES:
            self.assertEqual(normalize_comparison_mode(mode), mode)
        self.assertNotIn("overlay", COMPARISON_MODE_CHOICES)

    def test_old_preset_with_overlay_loads_as_slider(self):
        self.assertEqual(_normalize_output_settings({"comparison_mode": "overlay"})["comparison_mode"], "slider")
        self.assertEqual(_normalize_output_fields({"comparison_mode": "overlay"})["comparison_mode"], "slider")
        merged = merge_preset_with_defaults({"output": {"comparison_mode": "overlay"}}, ROOT, ["model"])
        self.assertEqual(merged["output"]["comparison_mode"], "slider")
        self.assertEqual(
            merge_preset_with_defaults({"output": {"comparison_mode": "side_by_side"}}, ROOT, ["model"])
            ["output"]["comparison_mode"], "side_by_side")

    def test_hidden_comparison_values_still_load(self):
        order = TAB_CONFIGS["output"]["order"]
        values = dict_to_values("output", {"comparison_zoom": 150, "show_difference": True},
                                output_defaults(["model"]))
        self.assertEqual(values[order.index("comparison_zoom")], 150)
        self.assertIs(values[order.index("show_difference")], True)

    def test_hidden_metadata_values_still_load(self):
        order = TAB_CONFIGS["output"]["order"]
        values = dict_to_values("output", {"metadata_format": "xml", "log_level": "debug"}, output_defaults(["model"]))
        self.assertEqual(values[order.index("metadata_format")], "xml")
        self.assertEqual(values[order.index("log_level")], "debug")


class HiddenControlsUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = _build_demo()
        cls.output_tab = _top_tab(cls.demo, "🔍 Output & Comparison")

    def test_comparison_mode_offers_no_overlay(self):
        mode = _by_label(self.output_tab, "Comparison Mode")
        self.assertEqual(_choice_values(mode), ["native", "slider", "side_by_side"])
        self.assertIn(mode.value, ("native", "slider", "side_by_side"))

    def test_dead_comparison_controls_hidden_but_wired(self):
        for label in ("Default Zoom Level", "Show Difference Overlay"):
            component = _by_label(self.output_tab, label)
            self.assertIs(component.visible, False, label)
            self.assertEqual(_wired_both_ways(self.demo, component), (True, True), label)
        for label in ("Comparison Mode", "Pin Reference Image", "Enable Fullscreen Comparison",
                      "Generate Input vs Output Comparison Video"):
            self.assertIsNot(_by_label(self.output_tab, label).visible, False, label)

    def test_dead_metadata_controls_hidden_but_wired(self):
        for label in ("Metadata Format", "Log Verbosity"):
            component = _by_label(self.output_tab, label)
            self.assertIs(component.visible, False, label)
            self.assertEqual(_wired_both_ways(self.demo, component), (True, True), label)
        for label in ("Save Processing Metadata", "Enable Run Telemetry"):
            self.assertIsNot(_by_label(self.output_tab, label).visible, False, label)


if __name__ == "__main__":
    unittest.main()
