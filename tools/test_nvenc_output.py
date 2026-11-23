r"""NVIDIA NVENC codecs in Output & Comparison > Video Codec, and the Custom FFmpeg Video Arguments in RelayVSR, RIFE
and the final fixed-size resize (user request 2026-11-23: on a 6-vCPU cloud A6000 about 60 % of a RelayVSR 4x ->
2160x3840 resize -> RIFE x2 run was libx265 on the CPU while the GPU sat idle).

CPU only by default: NVENC is switched off for these tests (SECOURSES_DISABLE_NVENC=1) and its availability is
simulated, so the CPU fallback, the replay of a failed NVENC start and every argument list are checked on any
machine. Set SECOURSES_TEST_REAL_NVENC=1 to also run real NVENC encodes on the GPU (RealNvencTests).

    venv\Scripts\python.exe -m unittest tools.test_nvenc_output -v
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REAL_NVENC = str(os.environ.get("SECOURSES_TEST_REAL_NVENC", "")).strip() == "1"
if not REAL_NVENC:
    os.environ["SECOURSES_DISABLE_NVENC"] = "1"

from shared import video_codec_options as vco  # noqa: E402

TEMP_ROOT = ROOT / "temp" / "nvenc_output_tests"
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
CQ_ARGS = ["-tune", "hq", "-rc", "vbr"]


def _probe(path: Path) -> dict:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "stream=codec_type,codec_name,profile,pix_fmt,width,height",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return next(s for s in json.loads(out)["streams"] if s["codec_type"] == "video")


def _frames(path: Path) -> int:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=nb_read_frames", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(out.strip())


def _make_clip(path: Path, frames: int = 12, size: str = "320x240", audio: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=s={size}:r=24:d={frames / 24.0}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={frames / 24.0}"]
    cmd += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    subprocess.run(cmd + [str(path)], check=True, capture_output=True)
    return path


def _nvenc_works(*_args, **_kwargs):
    return None


@unittest.skipUnless(FFMPEG and FFPROBE, "ffmpeg/ffprobe not on PATH")
class _TempCase(unittest.TestCase):
    def setUp(self):
        self.root = TEMP_ROOT / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)
        vco._NVENC_PROBE_CACHE.clear()

    def tearDown(self):
        # Only this test's own folder under <app>/temp/nvenc_output_tests.
        target = self.root.resolve()
        if target.parent == TEMP_ROOT.resolve() and not target.is_symlink():
            shutil.rmtree(target, ignore_errors=True)


class CodecChoiceTests(unittest.TestCase):
    def test_choices_profiles_and_pixel_formats(self):
        choices = vco.get_codec_choices()
        self.assertEqual(choices[:5], ["h264", "h265", "prores", "vp9", "av1"])  # existing order kept
        for key in ("h264_nvenc", "hevc_nvenc", "av1_nvenc"):
            self.assertIn(key, choices)
            self.assertEqual(vco.CODEC_PROFILES[key].ffmpeg_codec, key)
        self.assertEqual(vco.get_pixel_format_choices("hevc_nvenc"),
                         ["yuv420p", "yuv420p10le", "yuv444p", "yuv444p10le"])
        self.assertEqual(vco.get_pixel_format_choices("av1_nvenc"), ["yuv420p", "yuv420p10le"])
        self.assertEqual(vco.get_pixel_format_choices("h264_nvenc"), ["yuv420p", "yuv444p"])
        self.assertEqual(vco.get_pixel_format_choices("h264"), ["yuv420p", "yuv422p", "yuv444p"])
        self.assertIn("NVENC", vco.get_codec_info("hevc_nvenc"))

    def test_output_tab_normalization_keeps_nvenc(self):
        from shared.services.output_service import _normalize_output_fields

        cfg = _normalize_output_fields({"video_codec": "hevc_nvenc", "pixel_format": "yuv420p10le"})
        self.assertEqual((cfg["video_codec"], cfg["pixel_format"]), ("hevc_nvenc", "yuv420p10le"))
        cfg = _normalize_output_fields({"video_codec": "av1_nvenc", "pixel_format": "yuv444p"})
        self.assertEqual((cfg["video_codec"], cfg["pixel_format"]), ("av1_nvenc", "yuv420p"))
        cfg = _normalize_output_fields({"video_codec": "not_a_codec"})
        self.assertEqual(cfg["video_codec"], "h264")

    def test_aliases(self):
        self.assertEqual(vco.normalize_codec_key("nvenc_hevc"), "hevc_nvenc")
        self.assertEqual(vco.normalize_codec_key("libx265"), "h265")
        self.assertEqual(vco.nvenc_cpu_fallback_encoder("hevc_nvenc"), "libx265")
        self.assertEqual(vco.nvenc_cpu_fallback_encoder("av1_nvenc"), "libsvtav1")
        self.assertEqual(vco.nvenc_cpu_fallback_encoder("h264_nvenc"), "libx264")
        self.assertFalse(vco.is_nvenc_codec("h265"))


class NvencArgumentTests(unittest.TestCase):
    def test_hevc_main10_cq_and_preset(self):
        args = vco.nvenc_video_args("hevc_nvenc", 10, "yuv420p10le", "slow")
        self.assertEqual(args, ["-c:v", "hevc_nvenc", "-preset", "p5", *CQ_ARGS, "-cq", "10", "-b:v", "0",
                                "-spatial-aq", "1", "-profile:v", "main10", "-pix_fmt", "p010le"])

    def test_preset_map_quality_clamp_and_pixel_formats(self):
        presets = {p: vco.nvenc_video_args("h264_nvenc", 18, "yuv420p", p)[3] for p in vco.ENCODING_PRESETS}
        self.assertEqual(presets, {"ultrafast": "p1", "superfast": "p1", "veryfast": "p1", "faster": "p2",
                                   "fast": "p3", "medium": "p4", "slow": "p5", "slower": "p6", "veryslow": "p7"})
        self.assertEqual(vco.nvenc_video_args("hevc_nvenc", 0, "yuv420p", "medium")[9], "1")  # CQ 0 = auto
        self.assertEqual(vco.nvenc_video_args("av1_nvenc", 63, "yuv420p", "medium")[9], "63")
        self.assertEqual(vco.nvenc_video_args("hevc_nvenc", 99, "yuv420p", "medium")[9], "51")
        h264_444 = vco.nvenc_video_args("h264_nvenc", 18, "yuv444p", "medium")
        self.assertEqual(h264_444[-4:], ["-profile:v", "high444p", "-pix_fmt", "yuv444p"])
        self.assertEqual(vco.nvenc_video_args("h264_nvenc", 18, "yuv420p10le", "medium")[-1], "yuv420p")
        self.assertEqual(vco.nvenc_video_args("hevc_nvenc", 18, "yuv444p10le", "medium")[-4:],
                         ["-profile:v", "rext", "-pix_fmt", "yuv444p16le"])
        self.assertEqual(vco.nvenc_video_args("av1_nvenc", 18, "yuv420p10le", "medium")[-2:], ["-pix_fmt", "p010le"])

    def test_size_limit_skips_the_test_encode(self):
        with mock.patch.object(vco, "probe_nvenc_args", side_effect=AssertionError("no probe")):
            args, reason = vco.resolve_nvenc_video_args("h264_nvenc", 18, "yuv420p", "slow", 7680, 4320)
        self.assertIsNone(args)
        self.assertIn("4096", reason)

    def test_resolve_retries_without_spatial_aq(self):
        calls = []

        def probe(args, *_a, **_k):
            calls.append(list(args))
            return "spatial AQ not supported" if "-spatial-aq" in args else None

        with mock.patch.object(vco, "probe_nvenc_args", side_effect=probe):
            args, reason = vco.resolve_nvenc_video_args("hevc_nvenc", 18, "yuv420p", "slow", 1920, 1080)
        self.assertIsNone(reason)
        self.assertNotIn("-spatial-aq", args)
        self.assertEqual(len(calls), 2)

    def test_transient_failures_are_checked_again(self):
        calls = []

        def fake_run(cmd, **_kwargs):
            calls.append(cmd)
            return types.SimpleNamespace(returncode=1, stderr=fake_run.stderr)

        vco._NVENC_PROBE_CACHE.clear()
        args = ["-c:v", "hevc_nvenc", "-cq", "18"]
        with mock.patch.dict(os.environ, {"SECOURSES_DISABLE_NVENC": ""}), \
                mock.patch("subprocess.run", side_effect=fake_run), mock.patch("time.time") as now:
            fake_run.stderr = "[hevc_nvenc @ 0x1] OpenEncodeSessionEx failed: out of memory (10)"
            now.return_value = 1000.0
            self.assertIn("out of memory", vco.probe_nvenc_args(args, 640, 480))
            now.return_value = 1010.0
            vco.probe_nvenc_args(args, 640, 480)
            self.assertEqual(len(calls), 1)  # cached for a moment
            now.return_value = 1000.0 + vco._NVENC_TRANSIENT_RETRY_S + 1
            vco.probe_nvenc_args(args, 640, 480)
            self.assertEqual(len(calls), 2)  # checked again later
            fake_run.stderr = "[hevc_nvenc @ 0x1] No capable devices found"
            other = args + ["-preset", "p7"]
            vco.probe_nvenc_args(other, 640, 480)
            now.return_value = 99999.0
            vco.probe_nvenc_args(other, 640, 480)
            self.assertEqual(len(calls), 3)  # a missing device stays cached
        vco._NVENC_PROBE_CACHE.clear()

    def test_reason_is_the_encoder_line(self):
        stderr = ("[av1_nvenc @ 000001e73727d000] No capable devices found\n"
                  "[vost#0:0/av1_nvenc @ 000001e73727cd80] [enc:av1_nvenc @ 000001e73726c240] Error while opening "
                  "encoder - maybe incorrect parameters such as bit_rate, rate, width or height.\n"
                  "[vost#0:0/av1_nvenc @ 000001e73727cd80] Task finished with error code: -22 (Invalid argument)\n")
        self.assertEqual(vco._clean_ffmpeg_error(stderr), "No capable devices found")
        self.assertEqual(vco._clean_ffmpeg_error("[out#0/null @ 0x1] a\n[x @ 0x2] b"), "a | b")

    def test_disabled_nvenc_reports_reason(self):
        with mock.patch.dict(os.environ, {"SECOURSES_DISABLE_NVENC": "1"}):
            self.assertIn("turned off", vco.probe_nvenc_args(["-c:v", "hevc_nvenc"]))
            encoder, note = vco.resolve_encoder_name("hevc_nvenc", 18, "yuv420p10le", "slow", 3840, 2160)
        self.assertEqual(encoder, "libx265")
        self.assertIn("libx265 (CPU)", note)


class SharedBuilderTests(unittest.TestCase):
    def test_existing_codecs_keep_their_arguments(self):
        build = vco.build_ffmpeg_video_encode_args
        self.assertEqual(build(codec="h264", quality=18, pixel_format="yuv420p", preset="slow", audio_codec="none"),
                         ["-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p", "-an"])
        self.assertEqual(build(codec="h265", quality=20, pixel_format="yuv444p10le", preset="medium",
                               h265_tune="grain", audio_codec="aac"),
                         ["-c:v", "libx265", "-crf", "20", "-preset", "medium", "-tune", "grain",
                          "-pix_fmt", "yuv444p10le", "-c:a", "aac", "-b:a", "192k"])
        self.assertEqual(build(codec="av1", quality=30, pixel_format="yuv420p", preset="slow", audio_codec="copy"),
                         ["-c:v", "libsvtav1", "-crf", "30", "-preset", "4", "-svtav1-params",
                          "film-grain=8:film-grain-denoise=0", "-pix_fmt", "yuv420p", "-c:a", "copy"])
        self.assertEqual(build(codec="prores", pixel_format="yuv444p10le", audio_codec="none"),
                         ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuv444p10le", "-an"])

    def test_nvenc_when_available(self):
        with mock.patch.object(vco, "probe_nvenc_args", side_effect=_nvenc_works):
            args = vco.build_ffmpeg_video_encode_args(codec="hevc_nvenc", quality=10, pixel_format="yuv420p10le",
                                                      preset="slow", audio_codec="aac", width=2160, height=3840)
        self.assertEqual(args[:2], ["-c:v", "hevc_nvenc"])
        self.assertIn("main10", args)
        self.assertNotIn("-crf", args)
        self.assertEqual(args[-4:], ["-c:a", "aac", "-b:a", "192k"])

    def test_cpu_fallback_with_same_quality(self):
        notes = []
        with mock.patch.dict(os.environ, {"SECOURSES_DISABLE_NVENC": "1"}):
            args = vco.build_ffmpeg_video_encode_args(codec="hevc_nvenc", quality=10, pixel_format="yuv420p10le",
                                                      preset="slow", audio_codec="none", on_note=notes.append)
        self.assertEqual(args, ["-c:v", "libx265", "-crf", "10", "-preset", "slow", "-pix_fmt", "yuv420p10le",
                                "-an"])
        self.assertEqual(len(notes), 1)
        with mock.patch.dict(os.environ, {"SECOURSES_DISABLE_NVENC": "1"}):
            av1 = vco.build_ffmpeg_video_encode_args(codec="av1_nvenc", quality=28, pixel_format="yuv420p10le",
                                                     preset="medium", audio_codec="none", on_note=notes.append)
        self.assertEqual(av1[:2], ["-c:v", "libsvtav1"])
        self.assertEqual(av1[av1.index("-pix_fmt") + 1], "yuv420p10le")  # 10-bit kept on the CPU

    def test_custom_args_default_to_the_nvenc_encoder(self):
        self.assertEqual(vco.parse_custom_video_args("-cq 18 -preset p7", default_codec="hevc_nvenc"),
                         (["-c:v", "hevc_nvenc", "-cq", "18", "-preset", "p7"], ".mp4", None))
        self.assertTrue(vco.video_suffix_fits(".mp4", ".mov"))
        self.assertTrue(vco.video_suffix_fits(".mov", ".mov"))
        self.assertFalse(vco.video_suffix_fits(".mov", ".mp4"))
        self.assertFalse(vco.video_suffix_fits(".mkv", ".mp4"))

    def test_engine_env_round_trip(self):
        value = vco.engine_args_env_value(["-c:v", "hevc_nvenc", "-cq", "18"])
        self.assertEqual(vco.engine_args_from_env({vco.ENGINE_VIDEO_ARGS_ENV: value}), ["-c:v", "hevc_nvenc", "-cq", "18"])
        self.assertIsNone(vco.engine_args_from_env({vco.ENGINE_VIDEO_ARGS_ENV: "not json"}))
        self.assertIsNone(vco.engine_args_from_env({}))
        self.assertEqual(vco.engine_args_env_value(None), "")


class EngineCodecMapTests(unittest.TestCase):
    def test_maps_keep_nvenc(self):
        from shared.chunking import _normalize_video_encode_settings
        from shared.gan_runner import _normalize_video_codec_key as gan_key
        from shared.relayvsr_runner import CODECS
        from shared.rtx_superres_runner import _rtx_video_codec_key
        from shared.runner import Runner
        from shared.services.rife_service import _normalize_video_codec_key as rife_edit_key, rife_tab_video_codec

        for key in ("h264_nvenc", "hevc_nvenc", "av1_nvenc"):
            self.assertEqual(_normalize_video_encode_settings({"video_codec": key})["codec"], key)
            self.assertEqual(gan_key(key), key)
            self.assertIn(key, CODECS)
            self.assertEqual(_rtx_video_codec_key(key), (key, None))
            self.assertEqual(Runner._normalize_rife_video_codec(key), key)
            self.assertEqual(Runner._rife_video_codec_key(key), key)
            self.assertEqual(rife_edit_key(key), key)
        self.assertEqual(rife_tab_video_codec("hevc_nvenc"), "libx265")
        self.assertEqual(rife_tab_video_codec("h264_nvenc"), "libx264")
        self.assertEqual(rife_tab_video_codec("av1_nvenc"), "libx264")
        self.assertEqual(Runner._normalize_rife_video_codec("h265"), "libx265")  # unchanged

    def test_seedvr2_command_gets_the_cpu_codec(self):
        from shared.runner import Runner

        runner = Runner.__new__(Runner)
        runner.base_dir = ROOT
        runner._get_python_executable = lambda: "python"
        for codec, cpu, preset in (("hevc_nvenc", "libx265", "slow"), ("h264_nvenc", "libx264", "slow"),
                                   ("av1_nvenc", "libsvtav1", "4"), ("h265", "libx265", "slow"),
                                   ("h264", "libx264", "slow"), ("av1", "libsvtav1", "4")):
            cmd = runner._build_seedvr2_cmd(Path("cli.py"), {"input_path": "x.mp4", "video_codec": codec,
                                                             "video_quality": 12, "video_preset": "slow"},
                                            "mp4", False, "o.mp4")
            self.assertEqual(cmd[cmd.index("--video_codec") + 1], cpu, codec)
            self.assertEqual(cmd[cmd.index("--video_quality") + 1], "12")
            self.assertEqual(cmd[cmd.index("--video_preset") + 1], preset, codec)  # SVT-AV1: numeric preset

    def test_frame_pattern_size(self):
        from shared.video_encoder import _first_frame_size

        root = TEMP_ROOT / uuid.uuid4().hex
        try:
            frames = root / "frames"
            frames.mkdir(parents=True)
            from PIL import Image

            for i in (2, 1):
                Image.new("RGB", (96, 64)).save(frames / f"frame_{i:06d}_out.png")
            self.assertEqual(_first_frame_size(str(frames / "frame_%06d_out.png")), (96, 64))
            self.assertEqual(_first_frame_size(str(frames / "nothing_%06d.png")), (None, None))
        finally:
            target = root.resolve()
            if target.parent == TEMP_ROOT.resolve() and not target.is_symlink():
                shutil.rmtree(target, ignore_errors=True)


class SeedVR2EnvTests(unittest.TestCase):
    def _runner(self, root: Path, marker: bool):
        from shared.runner import Runner

        cli = root / "SeedVR2" / "inference_cli.py"
        cli.parent.mkdir(parents=True, exist_ok=True)
        cli.write_text("APP_VIDEO_ARGS_ENV = 'SECOURSES_FFMPEG_VIDEO_ARGS'\n" if marker else "# old\n",
                       encoding="utf-8")
        runner = Runner.__new__(Runner)
        runner.base_dir = root
        return runner

    def setUp(self):
        self.root = TEMP_ROOT / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)
        vco._NVENC_PROBE_CACHE.clear()

    def tearDown(self):
        target = self.root.resolve()
        if target.parent == TEMP_ROOT.resolve() and not target.is_symlink():
            shutil.rmtree(target, ignore_errors=True)

    def test_env_args(self):
        runner = self._runner(self.root, marker=True)
        logs = []
        env = {"CUDA_VISIBLE_DEVICES": "0"}
        settings = {"video_codec": "hevc_nvenc", "video_quality": 12, "use_10bit": True, "video_preset": "slow"}
        with mock.patch.object(vco, "probe_nvenc_args", side_effect=_nvenc_works):
            runner._seedvr2_nvenc_env(settings, env, logs.append)
        args = json.loads(env[vco.ENGINE_VIDEO_ARGS_ENV])
        self.assertEqual(args[:2], ["-c:v", "hevc_nvenc"])
        self.assertIn("p010le", args)  # 10-bit toggle -> Main10

        env2 = {}
        with mock.patch.dict(os.environ, {"SECOURSES_DISABLE_NVENC": "1"}):
            runner._seedvr2_nvenc_env(settings, env2, logs.append)
        self.assertNotIn(vco.ENGINE_VIDEO_ARGS_ENV, env2)
        self.assertTrue(any("libx265 (CPU)" in line for line in logs))

        env3 = {vco.ENGINE_VIDEO_ARGS_ENV: "stale"}
        runner._seedvr2_nvenc_env(dict(settings, video_backend="opencv"), env3, logs.append)
        self.assertNotIn(vco.ENGINE_VIDEO_ARGS_ENV, env3)
        runner._seedvr2_nvenc_env({"video_codec": "h265"}, env3, logs.append)
        self.assertNotIn(vco.ENGINE_VIDEO_ARGS_ENV, env3)

    def test_old_checkout_keeps_cpu(self):
        runner = self._runner(self.root, marker=False)
        logs, env = [], {}
        with mock.patch.object(vco, "probe_nvenc_args", side_effect=_nvenc_works):
            runner._seedvr2_nvenc_env({"video_codec": "hevc_nvenc"}, env, logs.append)
        self.assertNotIn(vco.ENGINE_VIDEO_ARGS_ENV, env)
        self.assertTrue(any("update it" in line for line in logs))


class RifeEncodePlanTests(_TempCase):
    def _runner(self, takes_args: bool = True):
        from shared.runner import Runner

        runner = Runner.__new__(Runner)
        runner.base_dir = ROOT
        runner._rife_cli_takes_app_video_args = lambda: takes_args
        return runner

    def test_custom_args_fit_and_container(self):
        clip = _make_clip(self.root / "clip.mp4", audio=False)
        runner, logs = self._runner(), []
        settings = {"video_codec": "h265", "video_custom_args": "-c:v libx265 -crf 8 -preset ultrafast"}
        args, override = runner._rife_encode_plan(settings, str(clip), self.root / "o.mp4", True, None, logs.append)
        self.assertEqual(args, ["-c:v", "libx265", "-crf", "8", "-preset", "ultrafast"])
        self.assertIsNone(override)
        prores = {"video_codec": "h264", "video_custom_args": "-c:v prores_ks -profile:v 3"}
        args, override = runner._rife_encode_plan(prores, str(clip), self.root / "o.mp4", True, None, logs.append)
        self.assertIsNone(args)  # a .mov codec is not written into RIFE's .mp4
        self.assertTrue(any(".mov file" in line for line in logs))
        args, _ = runner._rife_encode_plan(prores, str(clip), self.root / "o.mov", True, None, logs.append)
        self.assertEqual(args[:2], ["-c:v", "prores_ks"])
        bad = {"video_codec": "h264", "video_custom_args": "-c:v libx264 -pix_fmt not_a_format"}
        args, _ = runner._rife_encode_plan(bad, str(clip), self.root / "o.mp4", True, None, logs.append)
        self.assertIsNone(args)  # FFmpeg refuses them: Video Codec settings instead
        self.assertEqual(runner._rife_encode_plan(settings, str(clip), self.root / "seq", False, None)[0], None)

    def test_nvenc_plan(self):
        clip = _make_clip(self.root / "clip.mp4", audio=False)
        runner, logs = self._runner(), []
        settings = {"video_codec": "hevc_nvenc", "video_quality": 10, "pixel_format": "yuv420p10le",
                    "video_preset": "slow"}
        with mock.patch.object(vco, "probe_nvenc_args", side_effect=_nvenc_works):
            args, override = runner._rife_encode_plan(settings, str(clip), self.root / "o.mp4", True, "0", logs.append)
        self.assertEqual(args[:2], ["-c:v", "hevc_nvenc"])
        self.assertEqual(override, "h265")
        with mock.patch.dict(os.environ, {"SECOURSES_DISABLE_NVENC": "1"}):
            args, override = runner._rife_encode_plan(settings, str(clip), self.root / "o.mp4", True, "0", logs.append)
        self.assertIsNone(args)
        self.assertEqual(override, "h265")
        old = self._runner(takes_args=False)
        args, override = old._rife_encode_plan(settings, str(clip), self.root / "o.mp4", True, "0", logs.append)
        self.assertEqual((args, override), (None, "h265"))

    def test_rife_cli_hook(self):
        """The RIFE checkout's encoder arguments: the app's list wins, its -vf runs after the colour conversion."""
        cli = ROOT / "RIFE" / "inference_video.py"
        if not cli.is_file():
            self.skipTest("RIFE checkout not present")
        tree = ast.parse(cli.read_text(encoding="utf-8"))
        wanted = {"apply_codec_tuning", "_app_video_args", "video_encode_args"}
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
        self.assertEqual({f.name for f in funcs}, wanted)
        ns = {"os": os, "json": json, "APP_VIDEO_ARGS_ENV": vco.ENGINE_VIDEO_ARGS_ENV,
              "args": types.SimpleNamespace(video_codec="libx265", video_preset="slow", video_crf=10,
                                            pixel_format="yuv420p10le", h265_tune="grain", av1_film_grain=8,
                                            av1_film_grain_denoise=False)}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), str(cli), "exec"), ns)
        ns["app_video_args"] = None
        color = ["-vf", "scale=out_color_matrix=bt709"]
        self.assertEqual(ns["video_encode_args"](color),
                         color + ["-c:v", "libx265", "-preset", "slow", "-crf", "10", "-pix_fmt", "yuv420p10le",
                                  "-tune", "grain"])
        ns["app_video_args"] = ["-c:v", "hevc_nvenc", "-vf", "unsharp", "-cq", "12"]
        self.assertEqual(ns["video_encode_args"](color),
                         ["-vf", "scale=out_color_matrix=bt709,unsharp", "-c:v", "hevc_nvenc", "-cq", "12"])
        self.assertEqual(ns["video_encode_args"](), ["-vf", "unsharp", "-c:v", "hevc_nvenc", "-cq", "12"])
        with mock.patch.dict(os.environ, {vco.ENGINE_VIDEO_ARGS_ENV: json.dumps(["-c:v", "h264_nvenc"])}):
            self.assertEqual(ns["_app_video_args"](), ["-c:v", "h264_nvenc"])
        with mock.patch.dict(os.environ, {vco.ENGINE_VIDEO_ARGS_ENV: ""}):
            self.assertIsNone(ns["_app_video_args"]())

    def test_other_engine_cli_hooks(self):
        for rel in ("SeedVR2/inference_cli.py", "ComfyUI-FlashVSR_Stable/cli_main.py"):
            cli = ROOT / rel
            if not cli.is_file():
                continue
            self.assertTrue(vco.cli_takes_app_video_args(cli), rel)
            tree = ast.parse(cli.read_text(encoding="utf-8"))
            func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_app_video_args")
            ns = {"os": os, "APP_VIDEO_ARGS_ENV": vco.ENGINE_VIDEO_ARGS_ENV}
            exec(compile(ast.Module(body=[func], type_ignores=[]), rel, "exec"), ns)
            with mock.patch.dict(os.environ, {vco.ENGINE_VIDEO_ARGS_ENV: json.dumps(["-c:v", "av1_nvenc"])}):
                self.assertEqual(ns["_app_video_args"](), ["-c:v", "av1_nvenc"], rel)
            with mock.patch.dict(os.environ, {vco.ENGINE_VIDEO_ARGS_ENV: "[1, 2]"}):
                self.assertIsNone(ns["_app_video_args"](), rel)


class WriterFallbackTests(_TempCase):
    """shared.video_io.FfmpegWriter (RelayVSR, Open-H3, LTX 2.5 NVENC): a failed NVENC start is redone on the CPU."""

    def _writer(self, out: Path, w: int, h: int, nv_args):
        from shared.video_io import FfmpegWriter

        with mock.patch.object(vco, "resolve_nvenc_video_args", return_value=(nv_args, None)):
            return FfmpegWriter(str(out), w, h, 24, codec="hevc_nvenc", crf=20, preset="ultrafast",
                                pixel_format="yuv420p")

    def test_failed_start_replays_on_cpu_large_frames(self):
        import torch

        out = self.root / "big.mp4"
        writer = self._writer(out, 1280, 720, ["-c:v", "nvenc_that_does_not_exist", "-pix_fmt", "yuv420p"])
        for _ in range(4):
            writer.write(torch.randint(0, 255, (2, 720, 1280, 3), dtype=torch.uint8))
        writer.close()
        self.assertEqual(_probe(out)["codec_name"], "hevc")  # libx265 took over
        self.assertEqual(_frames(out), 8)

    def test_failed_start_replays_on_cpu_short_clip(self):
        import torch

        out = self.root / "short.mp4"
        writer = self._writer(out, 64, 48, ["-c:v", "nvenc_that_does_not_exist"])
        writer.write(torch.randint(0, 255, (3, 48, 64, 3), dtype=torch.uint8))
        writer.close()
        self.assertEqual(_frames(out), 3)

    def test_working_encoder_releases_the_kept_frames(self):
        import torch

        out = self.root / "ok.mp4"
        writer = self._writer(out, 320, 240, ["-c:v", "libx264", "-crf", "30", "-preset", "ultrafast",
                                              "-pix_fmt", "yuv420p"])
        for _ in range(40):
            writer.write(torch.randint(0, 255, (1, 240, 320, 3), dtype=torch.uint8))
        self.assertIsNone(writer._pending)  # encoder running: no more replay copies in RAM
        writer.close()
        self.assertEqual(_probe(out)["codec_name"], "h264")
        self.assertEqual(_frames(out), 40)

    def test_older_file_at_the_output_path_keeps_the_replay(self):
        out = self.root / "old.mp4"
        out.write_bytes(b"x" * 4096)  # a result of an earlier run at the same path
        writer = self._writer(out, 64, 48, ["-c:v", "nvenc_that_does_not_exist"])
        try:
            self.assertIsNotNone(writer._stale_sig)
            self.assertFalse(writer._encoder_running())  # the old bytes are not the new encoder's output
        finally:
            writer.abort()

    def test_unavailable_nvenc_uses_cpu_from_the_start(self):
        import torch

        from shared.video_io import FfmpegWriter

        out = self.root / "cpu.mp4"
        with mock.patch.dict(os.environ, {"SECOURSES_DISABLE_NVENC": "1"}):
            writer = FfmpegWriter(str(out), 64, 48, 24, codec="av1_nvenc", crf=40, pixel_format="yuv420p")
        self.assertIsNone(writer._pending)
        writer.write(torch.randint(0, 255, (4, 48, 64, 3), dtype=torch.uint8))
        writer.close()
        self.assertEqual(_probe(out)["codec_name"], "av1")
        self.assertEqual(_frames(out), 4)

    def test_cpu_codecs_unchanged(self):
        from shared.video_io import _codec_video_args

        self.assertEqual(_codec_video_args("libx265", 15, "slow", "yuv420p10le"),
                         ["-c:v", "libx265", "-crf", "15", "-preset", "slow", "-pix_fmt", "yuv420p10le"])
        self.assertEqual(_codec_video_args("libsvtav1", 30, "slow", "yuv420p"),
                         ["-c:v", "libsvtav1", "-crf", "30", "-preset", "8", "-pix_fmt", "yuv420p"])
        self.assertEqual(_codec_video_args("prores_ks", 15, "slow", "yuv444p10le"),
                         ["-c:v", "prores_ks", "-profile:v", "4444", "-vendor", "apl0", "-pix_fmt", "yuv444p10le"])


class FixedSizeResizeTests(_TempCase):
    def test_resize_plans(self):
        from shared.fixed_scale_output import resize_video_args

        base = {"video_codec": "h265", "video_quality": 12, "video_preset": "slow", "pixel_format": "yuv420p10le"}
        self.assertEqual(resize_video_args(base, 640, 480),
                         (["-c:v", "libx265", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p10le"], None, ""))
        custom = resize_video_args(dict(base, video_custom_args="-c:v prores_ks -profile:v 3"), 640, 480)
        self.assertEqual(custom, (["-c:v", "prores_ks", "-profile:v", "3"], None, ".mov"))
        # SVT-AV1 gets its numeric preset (a name made ffmpeg refuse the encode).
        av1 = resize_video_args(dict(base, video_codec="av1", pixel_format="yuv420p"), 640, 480)
        self.assertEqual(av1[0], ["-c:v", "libsvtav1", "-preset", "4", "-crf", "12", "-pix_fmt", "yuv420p"])
        logs = []
        with mock.patch.dict(os.environ, {"SECOURSES_DISABLE_NVENC": "1"}):
            cpu = resize_video_args(dict(base, video_codec="hevc_nvenc"), 640, 480, logs.append)
        self.assertEqual(cpu[0][:2], ["-c:v", "libx265"])
        self.assertIsNone(cpu[1])
        self.assertTrue(logs)
        with mock.patch.object(vco, "probe_nvenc_args", side_effect=_nvenc_works):
            nv = resize_video_args(dict(base, video_codec="hevc_nvenc"), 2160, 3840)
        self.assertEqual(nv[0][:2], ["-c:v", "hevc_nvenc"])
        self.assertEqual(nv[1][:2], ["-c:v", "libx265"])

    def test_failed_gpu_encode_is_retried_on_cpu(self):
        from shared.fixed_scale_output import _resize_video

        src = _make_clip(self.root / "clip.mp4")
        dst = self.root / "clip_final.mp4"
        logs = []
        plan = (["-c:v", "nvenc_that_does_not_exist"],
                ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p"], ".mp4")
        ok, err = _resize_video(src, dst, 160, 120, {}, on_log=logs.append, video_plan=plan)
        self.assertTrue(ok, err)
        info = _probe(dst)
        self.assertEqual((info["codec_name"], info["width"], info["height"]), ("h264", 160, 120))
        self.assertTrue(any("retrying" in line for line in logs))

    def test_custom_filter_runs_after_the_resize(self):
        from shared.fixed_scale_output import _resize_video, _split_video_filters

        self.assertEqual(_split_video_filters(["-c:v", "libx264", "-vf", "unsharp", "-crf", "8"]),
                         (["-c:v", "libx264", "-crf", "8"], "unsharp"))
        src = _make_clip(self.root / "clip.mp4", audio=False)
        dst = self.root / "clip_final.mp4"
        plan = (["-c:v", "libx264", "-vf", "hflip", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p"],
                None, ".mp4")
        ok, err = _resize_video(src, dst, 160, 120, {}, video_plan=plan)
        self.assertTrue(ok, err)
        info = _probe(dst)
        self.assertEqual((info["width"], info["height"]), (160, 120))  # the scale filter was not dropped

    def test_enforce_uses_custom_args_container(self):
        from shared.fixed_scale_output import enforce_fixed_scale_output_size

        src = _make_clip(self.root / "in" / "clip.mp4", size="160x120")
        out = _make_clip(self.root / "out" / "clip.mp4", size="640x480")
        settings = {"video_codec": "h264", "video_custom_args": "-c:v ffv1 -level 3"}
        path, resized, msg = enforce_fixed_scale_output_size(
            output_path=str(out), source_input_path=str(src), requested_scale=2, model_scale=4,
            pre_downscale_then_upscale=False, settings=settings)
        self.assertTrue(resized, msg)
        self.assertEqual(Path(path).suffix, ".mkv")
        info = _probe(Path(path))
        self.assertEqual((info["codec_name"], info["width"], info["height"]), ("ffv1", 320, 240))


class RelayVSRCommandTests(_TempCase):
    def test_command_codec_and_custom_args(self):
        from shared.relayvsr_runner import _output_file, build_command

        paths = {"wan": Path("w.safetensors"), "flash": Path("f.safetensors")}
        cmd = build_command("python", Path("r.py"), {"codec": "hevc_nvenc", "crf": 10, "pixel_format": "yuv420p10le",
                                                    "video_preset": "slow"}, paths, "in.mp4", Path("o.mp4"))
        self.assertEqual(cmd[cmd.index("--codec") + 1], "hevc_nvenc")
        self.assertNotIn("--video_args", cmd)
        cmd = build_command("python", Path("r.py"), {"codec": "libx265", "video_custom_args": "-crf 8 -preset fast"},
                            paths, "in.mp4", Path("o.mp4"))
        self.assertEqual(cmd[cmd.index("--video_args") + 1], "-c:v libx265 -crf 8 -preset fast")
        target = _output_file({"output_override": str(self.root / "run" / "clip.mp4"), "codec": "libx264",
                               "video_custom_args": "-c:v prores_ks -profile:v 3"}, ROOT, "clip.mp4", "clip.mp4",
                              False)
        self.assertEqual(target.suffix, ".mov")

    def test_chunked_custom_args_use_lossless_chunks(self):
        from shared.services.video_sr_service import _custom_args_chunk_settings

        logs = []
        same, plan = _custom_args_chunk_settings({"codec": "hevc_nvenc", "video_custom_args": ""}, "RelayVSR",
                                                 logs.append)
        self.assertIsNone(plan)
        self.assertEqual(same["codec"], "hevc_nvenc")
        chunk, plan = _custom_args_chunk_settings({"codec": "libx265", "video_codec": "h265",
                                                   "video_custom_args": "-c:v libx265 -crf 6"}, "RelayVSR",
                                                  logs.append)
        self.assertEqual((chunk["codec"], chunk["crf"], chunk["video_custom_args"]), ("libx264", 0, ""))
        self.assertEqual(plan["args"], ["-c:v", "libx265", "-crf", "6"])
        self.assertTrue(logs)
        # A ProRes Video Codec without custom arguments keeps the RelayVSR chunk path as before.
        same, plan = _custom_args_chunk_settings({"video_codec": "prores", "video_custom_args": ""}, "RelayVSR",
                                                 logs.append)
        self.assertIsNone(plan)


@unittest.skipUnless(REAL_NVENC and FFMPEG and FFPROBE, "set SECOURSES_TEST_REAL_NVENC=1 to encode on the GPU")
class RealNvencTests(_TempCase):
    def test_real_encodes(self):
        import torch

        from shared.fixed_scale_output import _resize_video
        from shared.video_io import FfmpegWriter

        for codec, pix, expect, profile in (("hevc_nvenc", "yuv420p10le", "hevc", "Main 10"),
                                            ("h264_nvenc", "yuv420p", "h264", "High"),
                                            ("av1_nvenc", "yuv420p10le", "av1", None)):
            args, reason = vco.resolve_nvenc_video_args(codec, 18, pix, "slow", 1280, 720)
            if args is None:
                print(f"[real NVENC] {codec}: not available here ({reason})")
                continue
            out = self.root / f"{codec}.mp4"
            writer = FfmpegWriter(str(out), 1280, 720, 24, codec=codec, crf=18, preset="slow", pixel_format=pix)
            for _ in range(24):
                writer.write(torch.randint(0, 255, (1, 720, 1280, 3), dtype=torch.uint8))
            writer.close()
            info = _probe(out)
            self.assertEqual(info["codec_name"], expect, codec)
            if profile:
                self.assertEqual(info.get("profile"), profile, codec)
            self.assertEqual(_frames(out), 24)
            print(f"[real NVENC] {codec}: {info}")
        src = _make_clip(self.root / "clip.mp4", frames=24)
        dst = self.root / "clip_final.mp4"
        ok, err = _resize_video(src, dst, 1080, 1920, {"video_codec": "hevc_nvenc", "video_quality": 10,
                                                      "pixel_format": "yuv420p10le", "video_preset": "slow"})
        self.assertTrue(ok, err)
        info = _probe(dst)
        self.assertEqual((info["codec_name"], info["width"], info["height"], info["pix_fmt"]),
                         ("hevc", 1080, 1920, "yuv420p10le"))


if __name__ == "__main__":
    unittest.main()
