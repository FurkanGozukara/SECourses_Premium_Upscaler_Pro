"""The Before/After image comparison lines up in fullscreen (V10.2 fix, 7 November 2026).

Seen in the tutorial recording (Image-Based (GAN), a 960x661 photo upscaled): the in-page slider was aligned, but in
fullscreen the two sides showed the picture at different scales. Checked in the real app in Chrome: in fullscreen the
input was drawn at its own 960x661 pixels and the result scaled to the screen (1454x1000).

Cause: the app's fullscreen shim (CUSTOM_HEAD) makes the slider's .image-container the fullscreen element. Gradio's
Image preview stylesheet, which the app loads for its gr.Image components, has the unscoped rule
".image-container:fullscreen img { object-fit: scale-down }". It also matches the slider's two pictures and outranks
the slider's own "object-fit: contain", so a picture smaller than the screen keeps its pixel size while a larger one
is scaled down.

These checks build the real interface without launching it, read the CSS and head the app passes to launch(), and
compare the rules' selectors with Gradio's packaged stylesheet. The browser check is in the provenance report.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

GRADIO_ASSETS = Path(gr.__file__).resolve().parent / "templates" / "frontend" / "assets"


class _Built(Exception):
    def __init__(self, demo, kwargs):
        super().__init__("UI built")
        self.demo, self.kwargs = demo, kwargs


def _build():
    import secourses_app

    def _capture(self, *args, **kwargs):
        raise _Built(self, kwargs)

    with patch.object(gr.Blocks, "launch", _capture):
        try:
            secourses_app.main([])
        except _Built as built:
            return built.demo, built.kwargs
    raise AssertionError("secourses_app.main() returned without launching the UI")


def _rules(css: str):
    """(selector, declarations) for every plain rule, media blocks flattened, comments removed."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"@media[^{]*\{", "", css)
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        for selector in selectors.split(","):
            yield " ".join(selector.split()), body


def _specificity(selector: str):
    ids = len(re.findall(r"#[\w-]+", selector))
    classes = len(re.findall(r"\.[\w-]+", selector)) + len(re.findall(r"(?<!:):(?!:)[\w-]+", selector))
    elements = len(re.findall(r"(?:^|[\s>+~])([a-zA-Z][\w-]*)", selector))
    return ids, classes, elements


def _object_fit(body: str):
    match = re.search(r"object-fit\s*:\s*([\w-]+)", body)
    return match.group(1) if match else None


class ImageSliderFullscreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo, cls.launch_kwargs = _build()

    def _gradio_fullscreen_rule(self):
        for sheet in sorted(GRADIO_ASSETS.glob("*.css")):
            for selector, body in _rules(sheet.read_text(encoding="utf-8", errors="replace")):
                if selector == ".image-container:fullscreen img" and _object_fit(body) == "scale-down":
                    return selector
        return None

    def test_the_comparison_pictures_are_fitted_alike_in_fullscreen(self):
        gradio_rule = self._gradio_fullscreen_rule()
        if gradio_rule is None:
            self.skipTest("this Gradio build has no unscoped fullscreen scale-down rule")
        overrides = [(selector, _object_fit(body)) for selector, body in _rules(self.launch_kwargs.get("css") or "")
                     if ".native-image-comparison-slider" in selector
                     and selector.endswith(".image-container:fullscreen img") and _object_fit(body)]
        self.assertTrue(overrides, "no app rule fits the comparison pictures in fullscreen")
        for selector, fit in overrides:
            self.assertEqual(fit, "contain", selector)
            self.assertGreater(_specificity(selector), _specificity(gradio_rule),
                               f"{selector} does not outrank Gradio's {gradio_rule}")

    def test_the_fullscreen_shim_targets_the_slider_container(self):
        head = self.launch_kwargs.get("head") or ""
        self.assertIn('btn.closest(".native-image-comparison-slider")', head)
        self.assertIn('wrapper.querySelector(".image-container")', head)

    def test_every_image_slider_with_fullscreen_carries_the_class(self):
        sliders = [b for b in self.demo.blocks.values() if isinstance(b, gr.ImageSlider)
                   and "fullscreen" in (getattr(b, "buttons", None) or [])]
        self.assertTrue(sliders)
        for slider in sliders:
            self.assertIn("native-image-comparison-slider", slider.elem_classes or [], slider.label)


if __name__ == "__main__":
    unittest.main()
