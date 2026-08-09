from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ui import media_preview


class MediaPreviewTests(unittest.TestCase):
    def test_incompatible_video_is_converted_without_touching_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.mp4"
            source.write_bytes(b"original-video")

            def convert(_ffmpeg, _source, destination, *, copy_streams):
                self.assertFalse(copy_streams)
                destination.write_bytes(b"browser-preview")
                return True, ""

            with (
                patch.object(media_preview.shutil, "which", return_value="ffmpeg"),
                patch.object(
                    media_preview,
                    "_first_stream_codecs",
                    return_value=("hevc", "aac"),
                ),
                patch.object(
                    media_preview,
                    "_is_browser_playable",
                    side_effect=lambda path, codecs=None: Path(path) != source,
                ),
                patch.object(media_preview, "_preview_cache_root", return_value=root / "cache"),
                patch.object(media_preview, "_run_ffmpeg_preview", side_effect=convert),
            ):
                image_path, video_path = media_preview.pick_preview_paths(str(source))

            self.assertIsNone(image_path)
            self.assertIsNotNone(video_path)
            preview = Path(video_path)
            self.assertNotEqual(preview, source)
            self.assertEqual(preview.read_bytes(), b"browser-preview")
            self.assertEqual(source.read_bytes(), b"original-video")

    def test_browser_compatible_video_does_not_need_a_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.mp4"
            source.write_bytes(b"browser-compatible-video")

            with (
                patch.object(media_preview.shutil, "which", return_value="ffmpeg"),
                patch.object(
                    media_preview,
                    "_first_stream_codecs",
                    return_value=("h264", "aac"),
                ),
                patch.object(media_preview, "_is_browser_playable", return_value=True),
            ):
                _, video_path = media_preview.pick_preview_paths(str(source))

            self.assertEqual(Path(video_path), source.resolve())

    def test_failed_preview_conversion_never_exposes_processing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.mkv"
            source.write_bytes(b"video")

            with (
                patch.object(media_preview.shutil, "which", return_value="ffmpeg"),
                patch.object(
                    media_preview,
                    "_first_stream_codecs",
                    return_value=("vp9", "opus"),
                ),
                patch.object(media_preview, "_is_browser_playable", return_value=False),
                patch.object(media_preview, "_preview_cache_root", return_value=root / "cache"),
                patch.object(
                    media_preview,
                    "_run_ffmpeg_preview",
                    return_value=(False, "encoder failed"),
                ),
                patch("builtins.print"),
            ):
                image_path, video_path = media_preview.pick_preview_paths(str(source))

            self.assertIsNone(image_path)
            self.assertIsNone(video_path)

    def test_h264_aac_in_mkv_uses_lossless_remux(self):
        command = media_preview._ffmpeg_preview_command(
            "ffmpeg",
            Path("input.mkv"),
            Path("preview.mp4"),
            copy_streams=True,
        )

        self.assertIn("copy", command)
        self.assertNotIn("libx264", command)
        self.assertTrue(media_preview._can_remux_to_mp4(("h264", "aac")))
        self.assertFalse(media_preview._can_remux_to_mp4(("hevc", "aac")))


if __name__ == "__main__":
    unittest.main()
