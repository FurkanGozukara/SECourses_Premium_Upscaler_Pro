"""V9.5 defaults: W4A8 for SeedVR2 / LTX 2.5 / MiniMax Open-H3, INT8 ConvRot for FlashVSR, and the INT8 fallback
of the LTX / Open-H3 runners on GPUs that cannot run W4A8. CPU only: the model download step is replaced by a stub
that records the model it was asked for and stops the run."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from shared.ltx25_constants import LTX25_DEV_INT4, LTX25_DEV_INT8, LTX25_DISTILLED_INT4, LTX25_DISTILLED_INT8  # noqa: E402
from shared.open_h3_constants import OPENH3_DIT_INT8, OPENH3_DIT_W4A8, OPENH3_DIT_W6A8  # noqa: E402

SAMPLE = next((p for p in (APP / "test_media").rglob("*.mp4")), None) or next(
    p for p in (APP.parent / "upstream_scan_20261101" / "eval" / "inputs").glob("*.mp4"))


def _defaults(windows: bool):
    env = {"SECOURSES_WINDOWS_INT8_DEFAULTS": "1" if windows else "0"}
    with mock.patch.dict(os.environ, env):
        from shared.services.flashvsr_service import flashvsr_defaults
        from shared.services.ltx25_service import ltx25_defaults
        from shared.services.open_h3_service import open_h3_defaults
        from shared.services.seedvr2_service import seedvr2_defaults

        return seedvr2_defaults(base_dir=APP), flashvsr_defaults(), ltx25_defaults(), open_h3_defaults()


class DefaultsTest(unittest.TestCase):
    def test_defaults_on_both_platforms(self):
        for windows in (True, False):
            seed, flash, ltx, h3 = _defaults(windows)
            self.assertEqual(seed["dit_model"], "seedvr2_7b_sharp_convrot_w4a8.safetensors")
            self.assertEqual(flash["precision"], "int8_convrot")
            self.assertEqual(ltx["model_name"], LTX25_DISTILLED_INT4)
            self.assertEqual(h3["model_name"], OPENH3_DIT_W4A8)

    def test_open_h3_guardrail_falls_back_to_default(self):
        from shared.services.open_h3_service import _enforce_open_h3_guardrails, open_h3_defaults

        d = open_h3_defaults()
        self.assertEqual(_enforce_open_h3_guardrails(dict(d, model_name="no such model"), d)["model_name"], OPENH3_DIT_W4A8)
        self.assertEqual(_enforce_open_h3_guardrails(dict(d, model_name=OPENH3_DIT_INT8), d)["model_name"], OPENH3_DIT_INT8)


class _Stop(Exception):
    pass


class RunnerFallbackTest(unittest.TestCase):
    def _ltx_model(self, model_name: str, reason):
        import shared.ltx25_runner as runner
        from shared.services.ltx25_service import ltx25_defaults

        seen = {}

        def fake_ensure(base_dir, name, *args, **kwargs):
            seen["model"] = name
            return False, "stopped by test"

        s = dict(ltx25_defaults(), input_path=str(SAMPLE), model_name=model_name, auto_tune_on_run=False)
        with mock.patch("shared.models.seedvr2_w4a8.w4a8_fallback_reason", return_value=reason), \
                mock.patch.object(runner, "ensure_ltx25_model", side_effect=fake_ensure):
            runner.run_ltx25(s, APP, on_progress=lambda m: None)
        return seen.get("model")

    def _h3_model(self, model_name: str, reason):
        import shared.open_h3_runner as runner
        from shared.services.open_h3_service import open_h3_defaults

        seen = {}

        def fake_ensure(base_dir, name, *args, **kwargs):
            seen["model"] = name
            return False, "stopped by test"

        s = dict(open_h3_defaults(), input_path=str(SAMPLE), model_name=model_name, auto_tune_on_run=False)
        with mock.patch("shared.models.seedvr2_w4a8.w4a8_fallback_reason", return_value=reason), \
                mock.patch("shared.model_downloads.ensure_open_h3_model", side_effect=fake_ensure):
            runner.run_open_h3(s, APP, on_progress=lambda m: None)
        return seen.get("model")

    def test_ltx_w4a8_falls_back_to_int8(self):
        old_gpu = "GPU 0 has compute capability 7.5"
        self.assertEqual(self._ltx_model(LTX25_DISTILLED_INT4, old_gpu), LTX25_DISTILLED_INT8)
        self.assertEqual(self._ltx_model(LTX25_DEV_INT4, old_gpu), LTX25_DEV_INT8)
        self.assertEqual(self._ltx_model(LTX25_DISTILLED_INT4, None), LTX25_DISTILLED_INT4)
        self.assertEqual(self._ltx_model(LTX25_DISTILLED_INT8, old_gpu), LTX25_DISTILLED_INT8)

    def test_open_h3_low_bit_falls_back_to_int8(self):
        old_gpu = "GPU 0 has compute capability 7.5"
        self.assertEqual(self._h3_model(OPENH3_DIT_W4A8, old_gpu), OPENH3_DIT_INT8)
        self.assertEqual(self._h3_model(OPENH3_DIT_W6A8, old_gpu), OPENH3_DIT_INT8)
        self.assertEqual(self._h3_model(OPENH3_DIT_W4A8, None), OPENH3_DIT_W4A8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
