r"""ProRes output from the Output tab (user report 2026-11-08: Open-H3 Regenerate and RTX Super Resolution
returned H.264 4:2:0 8-bit MP4 when Video Codec = prores). CPU only, no models; needs ffmpeg/ffprobe.

Covers the shared encoder arguments (profile follows the pixel format), the .mov container choice, the
Open-H3 engine writer, the RTX runner's output path and codec choice, the Open-H3 output plan, and the
chunked-run delivery (lossless H.264 chunks -> one ProRes encode). Other codecs must keep their arguments.

    venv\Scripts\python.exe -m unittest tools.test_prores_output -v
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.video_codec_options import (  # noqa: E402
    build_ffmpeg_video_encode_args,
    get_codec_info,
    is_prores_codec,
    prores_pixel_format,
    prores_profile_for_pixel_format,
    video_container_suffix,
)
from shared import prores_delivery  # noqa: E402

TEMP_ROOT = ROOT / "temp" / "prores_output_tests"
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _probe(path: Path) -> dict:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,profile,pix_fmt,codec_tag_string:format=format_name",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def _video(info: dict) -> dict:
    return next(s for s in info["streams"] if s["codec_type"] == "video")


def _has_audio(info: dict) -> bool:
    return any(s["codec_type"] == "audio" for s in info["streams"])


def _frames(path: Path) -> int:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=nb_read_frames", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(out.strip())


def _make_clip(path: Path, *, frames: int = 12, audio: bool = True, lossless: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=s=320x240:r=24:d={frames / 24.0}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={frames / 24.0}"]
    cmd += ["-c:v", "libx264"]
    cmd += ["-crf", "0", "-pix_fmt", "yuv444p"] if lossless else ["-crf", "18", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


@unittest.skipUnless(FFMPEG and FFPROBE, "ffmpeg/ffprobe not on PATH")
class _TempCase(unittest.TestCase):
    def setUp(self):
        self.root = TEMP_ROOT / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # Only this test's own folder under <app>/temp/prores_output_tests.
        target = self.root.resolve()
        if target.parent == TEMP_ROOT.resolve() and not target.is_symlink():
            shutil.rmtree(target, ignore_errors=True)


class EncoderArgumentTests(unittest.TestCase):
    def test_prores_profile_follows_pixel_format(self):
        args = build_ffmpeg_video_encode_args(codec="prores", quality=18, pixel_format="yuv422p10le",
                                              audio_codec="none")
        self.assertEqual(args, ["-c:v", "prores_ks", "-profile:v", "hq", "-pix_fmt", "yuv422p10le", "-an"])
        args = build_ffmpeg_video_encode_args(codec="prores", quality=2, pixel_format="yuv444p10le",
                                              audio_codec="none")
        self.assertEqual(args, ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuv444p10le", "-an"])

    def test_prores_with_a_non_prores_pixel_format(self):
        self.assertEqual(prores_pixel_format("yuv420p"), "yuv422p10le")
        self.assertEqual(prores_pixel_format("yuv444p"), "yuv444p10le")
        self.assertEqual(prores_profile_for_pixel_format("yuv420p10le"), "hq")
        args = build_ffmpeg_video_encode_args(codec="prores", pixel_format="yuv444p", audio_codec="none")
        self.assertIn("4444", args)
        self.assertEqual(args[args.index("-pix_fmt") + 1], "yuv444p10le")

    def test_other_codecs_keep_their_arguments(self):
        self.assertEqual(
            build_ffmpeg_video_encode_args(codec="h264", quality=18, pixel_format="yuv420p", preset="slow",
                                           audio_codec="none"),
            ["-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p", "-an"],
        )
        self.assertEqual(
            build_ffmpeg_video_encode_args(codec="h265", quality=20, pixel_format="yuv444p10le", preset="medium",
                                           h265_tune="grain", audio_codec="copy"),
            ["-c:v", "libx265", "-crf", "20", "-preset", "medium", "-tune", "grain", "-pix_fmt", "yuv444p10le",
             "-c:a", "copy"],
        )
        # A pixel format the codec lacks still falls back to its first one.
        self.assertEqual(
            build_ffmpeg_video_encode_args(codec="h264", pixel_format="yuv422p10le", audio_codec="none")[-2],
            "yuv420p",
        )

    def test_container_and_names(self):
        for name in ("prores", "prores_ks", "PRORES", "prores_aw"):
            self.assertTrue(is_prores_codec(name))
            self.assertEqual(video_container_suffix(name), ".mov")
        for name in ("h264", "libx264", "h265", "av1", "vp9", "", None):
            self.assertFalse(is_prores_codec(name))
            self.assertEqual(video_container_suffix(name), ".mp4")
        self.assertIn(".mov", get_codec_info("prores"))


class EngineWriterTests(_TempCase):
    """shared.video_io.FfmpegWriter, the writer of the Open-H3, LTX 2.5 and RelayVSR engines."""

    def _write(self, out: Path, **kwargs) -> Path:
        import torch

        from shared.video_io import FfmpegWriter

        writer = FfmpegWriter(str(out), 64, 48, 24, **kwargs)
        frames = torch.randint(0, 255, (6, 48, 64, 3), dtype=torch.uint8)
        writer.write(frames)
        writer.close()
        return out

    def test_prores_writer_makes_prores_mov(self):
        for pix, profile in (("yuv422p10le", "HQ"), ("yuv444p10le", "4444")):
            out = self._write(self.root / f"p_{pix}.mov", codec="prores_ks", crf=15, pixel_format=pix)
            video = _video(_probe(out))
            self.assertEqual(video["codec_name"], "prores")
            self.assertEqual(video["profile"], profile)
            self.assertEqual(_frames(out), 6)

    def test_h264_writer_unchanged(self):
        out = self._write(self.root / "h.mp4", codec="libx264", crf=15, pixel_format="yuv420p")
        video = _video(_probe(out))
        self.assertEqual((video["codec_name"], video["pix_fmt"]), ("h264", "yuv420p"))


class RtxRunnerTests(_TempCase):
    def test_codec_choice(self):
        from shared.rtx_superres_runner import _rtx_video_codec_key

        self.assertEqual(_rtx_video_codec_key("prores"), ("prores", None))
        self.assertEqual(_rtx_video_codec_key("prores_ks"), ("prores", None))
        self.assertEqual(_rtx_video_codec_key("h264"), ("h264", None))
        self.assertEqual(_rtx_video_codec_key("libx265"), ("h265", None))
        self.assertEqual(_rtx_video_codec_key("av1"), ("av1", None))
        self.assertEqual(_rtx_video_codec_key(None), ("h264", None))
        key, note = _rtx_video_codec_key("vp9")
        self.assertEqual(key, "h264")
        self.assertIn("vp9", note)

    def test_output_path_container(self):
        from shared.rtx_superres_runner import _make_video_output_path

        src = str(self.root / "clip.mp4")
        override_file = self.root / "out" / "clip.mp4"
        self.assertEqual(
            _make_video_output_path(input_path=src, global_output_dir=None, output_override=str(override_file)),
            override_file,
        )
        self.assertEqual(
            _make_video_output_path(input_path=src, global_output_dir=None, output_override=str(override_file),
                                    suffix=".mov"),
            override_file.with_suffix(".mov"),
        )
        folder = self.root / "run"
        folder.mkdir()
        self.assertEqual(
            _make_video_output_path(input_path=src, global_output_dir=None, output_override=str(folder),
                                    suffix=".mov"),
            folder / "clip.mov",
        )
        self.assertEqual(
            _make_video_output_path(input_path=src, global_output_dir=None, output_override=str(folder)),
            folder / "clip.mp4",
        )


class OpenH3PlanTests(_TempCase):
    def test_output_file_follows_the_codec(self):
        from shared.open_h3_runner import _item_plan

        src = _make_clip(self.root / "in" / "clip.mp4", audio=False, lossless=False)
        base = {"resolution_preset": "1080p", "output_override": str(self.root / "run" / "clip.mp4")}
        for codec, ext in (("prores_ks", ".mov"), ("libx264", ".mp4"), ("libx265", ".mp4")):
            plan = _item_plan(dict(base, codec=codec), ROOT, str(src), "video")
            self.assertNotIsInstance(plan, str, plan)
            self.assertEqual(plan[-1], self.root / "run" / f"clip{ext}")
        # Chunk prepass outputs keep their explicit MP4 path.
        explicit = self.root / "stage" / "chunk_0001_upscaled.mp4"
        plan = _item_plan(dict(base, codec="libx264", output_path=str(explicit)), ROOT, str(src), "video")
        self.assertEqual(plan[-1], explicit)


class ChunkedDeliveryTests(_TempCase):
    def test_lossless_chunk_settings_and_records(self):
        requested = {"video_codec": "prores", "pixel_format": "yuv444p10le", "video_quality": 18,
                     "codec": "prores_ks", "crf": 18, "video_preset": "slow", "other": 1}
        self.assertTrue(prores_delivery.wants_prores(requested))
        self.assertTrue(prores_delivery.wants_prores({"codec": "prores_ks"}))
        self.assertFalse(prores_delivery.wants_prores({"video_codec": "h264", "codec": "libx264"}))
        chunk = prores_delivery.lossless_chunk_settings(requested)
        self.assertEqual(
            {k: chunk[k] for k in ("video_codec", "video_quality", "pixel_format", "codec", "crf")},
            {"video_codec": "h264", "video_quality": 0, "pixel_format": "yuv444p", "codec": "libx264", "crf": 0},
        )
        self.assertEqual(prores_delivery.lossless_chunk_settings(requested, deep=True)["pixel_format"],
                         "yuv444p10le")
        self.assertFalse(prores_delivery.wants_prores(chunk))
        self.assertEqual(requested["video_codec"], "prores")  # the caller's dict is untouched
        kept = {k: requested[k] for k in prores_delivery.ENCODE_KEYS if k in requested}
        restored = prores_delivery.restore_requested_encode(chunk, kept)
        self.assertEqual({k: restored[k] for k in kept}, kept)
        self.assertEqual(restored["other"], 1)

    def test_lossless_h264_chunk_args_are_lossless(self):
        # RTX chunks: the runner's encoder arguments for the chunk settings must be lossless 4:4:4 H.264.
        chunk = prores_delivery.lossless_chunk_settings({"video_codec": "prores"})
        args = build_ffmpeg_video_encode_args(codec=chunk["video_codec"], quality=chunk["video_quality"],
                                              pixel_format=chunk["pixel_format"], preset=chunk["video_preset"],
                                              audio_codec="none")
        self.assertEqual(args, ["-c:v", "libx264", "-crf", "0", "-preset", "veryfast", "-pix_fmt", "yuv444p", "-an"])

    def test_deliver_prores_replaces_the_merged_mp4(self):
        merged = _make_clip(self.root / "run" / "clip.mp4")
        frames = _frames(merged)
        lines = []
        out, err = prores_delivery.deliver_prores(str(merged), "yuv422p10le", on_progress=lines.append)
        self.assertIsNone(err)
        out = Path(out)
        self.assertEqual(out, self.root / "run" / "clip.mov")
        info = _probe(out)
        video = _video(info)
        self.assertEqual((video["codec_name"], video["profile"], video["pix_fmt"]), ("prores", "HQ", "yuv422p10le"))
        self.assertTrue(_has_audio(info))
        self.assertEqual(_frames(out), frames)
        self.assertFalse(merged.exists())
        self.assertFalse(any("__final_tmp" in p.name for p in out.parent.iterdir()))
        self.assertTrue(any("ProRes 422 HQ" in line for line in lines))
        # Already ProRes: returned as is.
        self.assertEqual(prores_delivery.deliver_prores(str(out), "yuv422p10le"), (str(out), None))

    def test_deliver_prores_4444_keep_source_and_no_audio(self):
        merged = _make_clip(self.root / "run" / "silent.mp4", audio=False)
        out, err = prores_delivery.deliver_prores(str(merged), "yuv444p10le", keep_source=True)
        self.assertIsNone(err)
        video = _video(_probe(Path(out)))
        self.assertEqual((video["codec_name"], video["profile"]), ("prores", "4444"))
        self.assertTrue(merged.exists())
        self.assertFalse(_has_audio(_probe(Path(out))))

    def test_deliver_prores_errors(self):
        self.assertEqual(prores_delivery.deliver_prores(None, "yuv422p10le")[0], None)
        out, err = prores_delivery.deliver_prores(str(self.root / "missing.mp4"), "yuv422p10le")
        self.assertIsNone(out)
        self.assertIn("not found", err)

    def test_open_h3_service_helpers(self):
        from shared.services import open_h3_service as svc

        logs = []
        same, plan = svc._final_encode_chunk_settings({"video_codec": "h264", "codec": "libx264"}, logs.append)
        self.assertIsNone(plan)
        self.assertEqual(same["codec"], "libx264")
        self.assertEqual(logs, [])
        chunk, plan = svc._final_encode_chunk_settings(
            {"video_codec": "prores", "codec": "prores_ks", "pixel_format": "yuv444p10le"}, logs.append)
        self.assertEqual(plan["suffix"], ".mov")
        self.assertEqual((chunk["codec"], chunk["crf"], chunk["pixel_format"]), ("libx264", 0, "yuv444p"))
        merged = _make_clip(self.root / "run" / "clip.mp4")
        # Not a final-encode run, or a failed run: the merged file is returned unchanged.
        self.assertEqual(svc._final_encode_finish(str(merged), None, 0, logs.append), str(merged))
        self.assertEqual(svc._final_encode_finish(str(merged), plan, 1, logs.append), str(merged))
        out = svc._final_encode_finish(str(merged), plan, 0, logs.append)
        self.assertEqual(Path(out).suffix, ".mov")
        self.assertEqual(_video(_probe(Path(out)))["profile"], "4444")


class CustomArgumentParseTests(unittest.TestCase):
    def test_accepted_arguments_and_containers(self):
        from shared.video_codec_options import parse_custom_video_args as parse

        self.assertEqual(parse("", "h264"), ([], "", None))
        self.assertEqual(parse("   ", "h264"), ([], "", None))
        self.assertEqual(
            parse("-c:v prores_ks -profile:v 4444xq -pix_fmt yuv444p10le"),
            (["-c:v", "prores_ks", "-profile:v", "4444xq", "-pix_fmt", "yuv444p10le"], ".mov", None),
        )
        self.assertEqual(parse("-c:v dnxhd -profile:v dnxhr_hqx -pix_fmt yuv422p10le")[1], ".mov")
        self.assertEqual(parse("-c:v ffv1 -level 3")[1:], (".mkv", None))
        self.assertEqual(parse("-c:v libx265 -crf 8 -f matroska"),
                         (["-c:v", "libx265", "-crf", "8"], ".mkv", None))
        # Without -c:v, the selected Video Codec's encoder is used.
        self.assertEqual(parse('-crf 10 -x265-params "aq-mode=3:psy-rd=1.0"', "h265"),
                         (["-c:v", "libx265", "-crf", "10", "-x265-params", "aq-mode=3:psy-rd=1.0"], ".mp4", None))
        self.assertEqual(parse("-profile:v 4444xq", "prores")[0][:2], ["-c:v", "prores_ks"])
        self.assertEqual(parse("-c:v libx264 -qp -1")[0], ["-c:v", "libx264", "-qp", "-1"])
        self.assertEqual(parse("-c:v libx264\n-crf 12")[0], ["-c:v", "libx264", "-crf", "12"])

    def test_refused_arguments(self):
        from shared.video_codec_options import parse_custom_video_args as parse

        for text, word in (("out.mp4", "not an option"), ("-c:v libx264 out.mp4", "not an option"),
                           ("-i other.mp4", "input"), ("-c libx264", "-c:v"), ("-crf", "needs a value"),
                           ("-f null", "mov, mp4 or matroska"), ("-map 0:v", "maps"), ("-r 60", "frame rate"),
                           ("-c:a aac", "Audio Codec"), ("-t 5", "trimming"), ('-x264-params "a=1', "quotes")):
            args, suffix, err = parse(text)
            self.assertEqual((args, suffix), ([], ""), text)
            self.assertIn(word, err, text)


@unittest.skipUnless(FFMPEG and FFPROBE, "ffmpeg/ffprobe not on PATH")
class CustomArgumentEncodeTests(_TempCase):
    def test_check_encode(self):
        from shared.video_codec_options import check_custom_video_args, parse_custom_video_args

        args, suffix, _ = parse_custom_video_args("-c:v prores_ks -profile:v 4444xq -pix_fmt yuv444p10le")
        self.assertIsNone(check_custom_video_args(args, suffix))
        args, suffix, _ = parse_custom_video_args("-c:v no_such_encoder_xyz")
        err = check_custom_video_args(args, suffix)
        self.assertIn("Unknown encoder 'no_such_encoder_xyz'", err)
        self.assertNotIn("@ ", err)  # FFmpeg's "[vost#0:0 @ 0x..]" prefixes are dropped
        self.assertNotIn("secourses_custom_args_", err)  # and the test file's path
        args, suffix, _ = parse_custom_video_args("-c:v libx264 -pix_fmt not_a_format")
        self.assertIsNotNone(check_custom_video_args(args, suffix))

    def test_state_check(self):
        self.assertIsNone(prores_delivery.custom_video_args_error({}))
        self.assertIsNone(prores_delivery.custom_video_args_error({"video_custom_args_val": ""}))
        self.assertIsNone(prores_delivery.custom_video_args_error(
            {"video_custom_args_val": "-c:v ffv1 -level 3", "video_codec_val": "h264"}))
        self.assertIn("input", prores_delivery.custom_video_args_error(
            {"output_settings": {"video_custom_args": "-i x.mp4"}}))

    def test_engine_writer_uses_custom_args(self):
        import torch

        from shared.video_io import FfmpegWriter

        out = self.root / "custom.mkv"
        writer = FfmpegWriter(str(out), 64, 48, 24, codec="libx264", crf=15, pixel_format="yuv420p",
                              video_args="-c:v ffv1 -level 3 -pix_fmt yuv444p")
        writer.write(torch.randint(0, 255, (5, 48, 64, 3), dtype=torch.uint8))
        writer.close()
        video = _video(_probe(out))
        self.assertEqual((video["codec_name"], video["pix_fmt"]), ("ffv1", "yuv444p"))
        self.assertEqual(_frames(out), 5)

    def test_open_h3_plan_and_engine_argument(self):
        from shared.open_h3_runner import _custom_video_args_text, _item_plan

        src = _make_clip(self.root / "in" / "clip.mp4", audio=False, lossless=False)
        base = {"resolution_preset": "1080p", "output_override": str(self.root / "run" / "clip.mp4"),
                "codec": "libx264"}
        for text, ext in (("-c:v ffv1", ".mkv"), ("-c:v prores_ks -profile:v 4444xq", ".mov"),
                          ("-c:v libx265 -crf 6", ".mp4"), ("", ".mp4")):
            plan = _item_plan(dict(base, video_custom_args=text), ROOT, str(src), "video")
            self.assertEqual(plan[-1], self.root / "run" / f"clip{ext}", text)
        self.assertEqual(_custom_video_args_text(dict(base, video_custom_args='-crf 4 -x264-params "a=1:b=2"')),
                         "-c:v libx264 -crf 4 -x264-params a=1:b=2")
        self.assertEqual(_custom_video_args_text(dict(base, video_custom_args="")), "")
        self.assertEqual(_custom_video_args_text(dict(base, video_custom_args="-i x")), "")

    def test_rtx_filter_merge_and_path(self):
        from shared.rtx_superres_runner import _make_video_output_path, _pop_video_filter

        args = ["-c:v", "libx264", "-vf", "unsharp=5:5:0.5", "-crf", "8", "-filter:v", "eq=gamma=1.1"]
        self.assertEqual(_pop_video_filter(args), "unsharp=5:5:0.5,eq=gamma=1.1")
        self.assertEqual(args, ["-c:v", "libx264", "-crf", "8"])
        self.assertEqual(_pop_video_filter(["-c:v", "ffv1"]), "")
        self.assertEqual(
            _make_video_output_path(input_path=str(self.root / "clip.mp4"), global_output_dir=None,
                                    output_override=str(self.root / "o" / "clip.mp4"), suffix=".mkv"),
            self.root / "o" / "clip.mkv",
        )

    def test_final_encode_plan_and_in_place_delivery(self):
        plan = prores_delivery.final_encode_plan(
            {"video_codec": "prores", "video_custom_args": "-c:v libx265 -crf 30 -preset ultrafast"})
        self.assertEqual(plan["suffix"], ".mp4")  # custom arguments win over the codec choice
        merged = _make_clip(self.root / "run" / "clip.mp4")
        frames = _frames(merged)
        out, err = prores_delivery.deliver_final_encode(str(merged), plan)
        self.assertIsNone(err)
        self.assertEqual(Path(out), merged)  # same suffix: replaced in place, same name
        info = _probe(merged)
        self.assertEqual(_video(info)["codec_name"], "hevc")
        self.assertTrue(_has_audio(info))
        self.assertEqual(_frames(merged), frames)
        self.assertEqual(sorted(p.name for p in merged.parent.iterdir()), ["clip.mp4"])

        mkv_plan = prores_delivery.final_encode_plan({"video_custom_args": "-c:v ffv1 -level 3"})
        merged2 = _make_clip(self.root / "run2" / "clip.mp4")
        out, err = prores_delivery.deliver_final_encode(str(merged2), mkv_plan)
        self.assertIsNone(err)
        self.assertEqual(Path(out).suffix, ".mkv")
        info = _probe(Path(out))
        self.assertEqual(_video(info)["codec_name"], "ffv1")
        self.assertTrue(_has_audio(info))  # AAC copied into MKV
        self.assertFalse(merged2.exists())
        self.assertIsNone(prores_delivery.final_encode_plan({"video_custom_args": "-i bad"}))
        self.assertIsNone(prores_delivery.final_encode_plan({"video_codec": "h265"}))


class OutputTabSettingTests(unittest.TestCase):
    def test_key_order_defaults_and_state_sync(self):
        from shared.services.output_service import OUTPUT_ORDER, _normalize_output_fields, output_defaults
        from shared.universal_preset import dict_to_values, values_to_dict
        from ui.universal_preset_section import sync_tab_to_shared_state

        self.assertEqual(OUTPUT_ORDER[-1], "video_custom_args")
        defaults = output_defaults(["default"])
        self.assertEqual(defaults["video_custom_args"], "")
        # A preset saved before this setting existed loads it as empty.
        legacy = {k: v for k, v in defaults.items() if k != "video_custom_args"}
        values = dict_to_values("output", legacy, defaults)
        self.assertEqual(values[OUTPUT_ORDER.index("video_custom_args")], "")
        self.assertEqual(_normalize_output_fields({"video_custom_args": "  -c:v ffv1\n -level 3 "})["video_custom_args"],
                         "-c:v ffv1 -level 3")
        vals = [defaults[k] for k in OUTPUT_ORDER]
        vals[OUTPUT_ORDER.index("video_custom_args")] = "-c:v prores_ks -profile:v 4444xq"
        state = sync_tab_to_shared_state("output", vals, {"seed_controls": {}})
        seed = state["seed_controls"]
        self.assertEqual(seed["video_custom_args_val"], "-c:v prores_ks -profile:v 4444xq")
        self.assertEqual(seed["output_settings"]["video_custom_args"], "-c:v prores_ks -profile:v 4444xq")
        self.assertEqual(values_to_dict("output", vals)["video_custom_args"], "-c:v prores_ks -profile:v 4444xq")


if __name__ == "__main__":
    unittest.main()
