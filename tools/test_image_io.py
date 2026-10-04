from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from shared import image_io


def _card16() -> np.ndarray:
    """Dark 16-bit BGR card, like the AI-generated PNGs that came out white."""
    yy, xx = np.mgrid[0:24, 0:32].astype(np.float32)
    img = np.dstack([0.05 + 0.3 * xx / 31.0, 0.1 + 0.2 * yy / 23.0, np.full_like(xx, 0.25)])
    return (img * 65535.0 + 0.5).astype(np.uint16)


class ImageIoTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.card16 = _card16()
        self.card8 = ((self.card16.astype(np.uint32) + 128) // 257).astype(np.uint8)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name: str, image: np.ndarray) -> Path:
        path = self.root / name
        self.assertTrue(image_io.imwrite(path, image), name)
        self.assertTrue(path.is_file(), name)
        return path

    def assertClose(self, actual: np.ndarray, expected: np.ndarray, tol: int = 1):
        self.assertEqual(actual.dtype, np.uint8)
        self.assertEqual(actual.shape, expected.shape)
        self.assertLessEqual(int(np.abs(actual.astype(int) - expected.astype(int)).max()), tol)

    def test_non_ascii_paths_round_trip(self):
        for name in ("görsel_çğüşıö.png", "café_日本語.png"):
            with self.subTest(name=name):
                path = self._write(name, self.card8)
                self.assertTrue(np.array_equal(image_io.imread(path), self.card8))
        self.assertIsNone(image_io.imread(self.root / "missing_ş.png"))

    def test_imread_matches_cv2_for_ascii_paths(self):
        path = self._write("plain16.png", self.card16)
        for flags in (cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED, cv2.IMREAD_GRAYSCALE):
            self.assertTrue(np.array_equal(image_io.imread(path, flags), cv2.imread(str(path), flags)))

    def test_load_image_bgr8_scales_every_depth(self):
        gray16 = cv2.cvtColor(self.card16, cv2.COLOR_BGR2GRAY)
        gray8 = cv2.cvtColor(((gray16.astype(np.uint32) + 128) // 257).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        alpha = np.full(self.card16.shape[:2], 65535, np.uint16)
        cases = {
            "rgb16.png": (self.card16, self.card8),
            "gray16.png": (gray16, gray8),
            "rgba16.png": (np.dstack([self.card16, alpha]), self.card8),
            "rgb16.tif": (self.card16, self.card8),
            "rgb32f.tif": (self.card16.astype(np.float32) / 65535.0, self.card8),
            "rgb8_ç.png": (self.card8, self.card8),
        }
        for name, (stored, expected) in cases.items():
            with self.subTest(name=name):
                self.assertClose(image_io.load_image_bgr8(self._write(name, stored)), expected)

    def test_load_image_rgb_pil_keeps_pillow_for_ordinary_files(self):
        palette = Image.fromarray(cv2.cvtColor(self.card8, cv2.COLOR_BGR2RGB)).quantize(64)
        path = self.root / "palette.png"
        palette.save(path)
        with Image.open(path) as img:
            expected = np.asarray(img.convert("RGB"))
        self.assertTrue(np.array_equal(np.asarray(image_io.load_image_rgb_pil(path)), expected))

    def test_load_image_rgb_pil_does_not_clip_high_depth(self):
        gray16 = cv2.cvtColor(self.card16, cv2.COLOR_BGR2GRAY)
        gray_path = self._write("gray16.png", gray16)
        with Image.open(gray_path) as img:
            self.assertEqual(img.mode, "I;16")  # the mode Pillow clips to white
        gray8 = ((gray16.astype(np.uint32) + 128) // 257).astype(np.uint8)
        self.assertClose(np.asarray(image_io.load_image_rgb_pil(gray_path)), np.dstack([gray8] * 3))

        float_path = self._write("rgb32f.tif", self.card16.astype(np.float32) / 65535.0)
        self.assertClose(
            np.asarray(image_io.load_image_rgb_pil(float_path)), cv2.cvtColor(self.card8, cv2.COLOR_BGR2RGB)
        )

    def test_to_uint8_ranges(self):
        self.assertTrue(np.array_equal(image_io.to_uint8(np.array([0, 128, 65535], np.uint16)), [0, 0, 255]))
        self.assertTrue(np.array_equal(image_io.to_uint8(np.array([-1.0, 0.5, 2.0, np.nan], np.float32)), [0, 128, 255, 0]))


if __name__ == "__main__":
    unittest.main()
