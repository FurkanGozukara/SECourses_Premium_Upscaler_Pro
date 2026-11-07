r"""Batch outputs keep their input's name (user decision, 2026-11-08); CPU only, no models.

Covers the shared naming helper, the post-step chain (global DLSS 5 / RIFE stand-ins), the
keep-only cleanup on reruns and skipped items, the Face Restoration and DLSS 5 tab batches with
their engines replaced by file copies, SeedVR2's in-place JPG/WebP re-encode and the RTX
runner's batch output folders. Single-run names must stay as they were.

    venv\Scripts\python.exe -m unittest tools.test_batch_output_names -v
"""
from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.batch_naming import BATCH_KEEP_NAME_KEY, adopt_output_name, batch_keep_name, same_name_target
from shared.batch_output_cleanup import keep_only_batch_outputs
from shared.output_run_manager import find_flattened_batch_output, prepare_batch_video_run_dir

TEMP_ROOT = ROOT / "temp" / "batch_output_names"


def _write(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class _TempCase(unittest.TestCase):
    def setUp(self):
        self.root = TEMP_ROOT / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # Only this test's own folder under <app>/temp/batch_output_names.
        target = self.root.resolve()
        if target.parent == TEMP_ROOT.resolve() and not target.is_symlink():
            shutil.rmtree(target, ignore_errors=True)


class AdoptOutputNameTests(_TempCase):
    def test_post_step_file_replaces_the_output_under_its_name(self):
        base = _write(self.root / "clip" / "clip.mp4", b"upscaled")
        produced = _write(self.root / "clip" / "clip_face_restored.mp4", b"restored")
        result = adopt_output_name(base, produced)
        self.assertEqual(Path(result), base)
        self.assertEqual(base.read_bytes(), b"restored")
        self.assertFalse(produced.exists())

    def test_other_extension_keeps_the_stem_and_drops_the_old_version(self):
        base = _write(self.root / "photo.png", b"png")
        produced = _write(self.root / "photo_final2048x2048.jpg", b"jpg")
        result = adopt_output_name(base, produced)
        self.assertEqual(Path(result), self.root / "photo.jpg")
        self.assertFalse(base.exists())
        self.assertFalse(produced.exists())

    def test_directory_result_replaces_the_directory(self):
        base = _write(self.root / "clip" / "frame_0001.png", b"old").parent
        produced = _write(self.root / "clip_final1920x1080" / "frame_0001.png", b"new").parent
        result = adopt_output_name(base, produced)
        self.assertEqual(Path(result), base)
        self.assertEqual((base / "frame_0001.png").read_bytes(), b"new")
        self.assertFalse(produced.exists())

    def test_noops(self):
        base = _write(self.root / "clip.mp4")
        self.assertEqual(adopt_output_name(base, base), str(base))  # failed post-step returns its input
        self.assertIsNone(adopt_output_name(base, None))
        missing = self.root / "missing_2xFPS.mp4"
        self.assertEqual(adopt_output_name(base, missing), str(missing))
        self.assertTrue(base.exists())

    def test_batch_flag_and_same_name_target(self):
        self.assertTrue(batch_keep_name({BATCH_KEEP_NAME_KEY: True}))
        self.assertFalse(batch_keep_name({}))
        self.assertFalse(batch_keep_name(None))
        src = _write(self.root / "in" / "photo.jpg")
        self.assertEqual(same_name_target(self.root / "out", src, ".png", "_DLSS5"), self.root / "out" / "photo.png")
        # Output folder = input folder: the input is never the target.
        self.assertEqual(same_name_target(src.parent, src, None, "_DLSS5"), src.parent / "photo_DLSS5.jpg")


class GlobalPostStepChainTests(_TempCase):
    """maybe_apply_global_rife runs DLSS 5 then RIFE; the stand-ins write the files the real steps write."""

    def _run(self, keep_name: bool, dlss5_on: bool = True, rife_on: bool = True):
        from shared import global_rife

        base = _write(self.root / "clip" / "clip.mp4", b"upscaled")

        def fake_dlss5(path, seed_controls, on_log=None):
            if not dlss5_on:
                return None, ""
            out = Path(path).with_name(Path(path).stem + "_DLSS5.mp4")
            out.write_bytes(Path(path).read_bytes() + b"+dlss5")
            return str(out), "ok"

        def fake_rife(runner, path, seed_controls, on_log=None, chunking_context=None):
            if not rife_on:
                return None, "Global RIFE disabled"
            out = Path(path).with_name(Path(path).stem + "_2xFPS.mp4")
            out.write_bytes(Path(path).read_bytes() + b"+rife")
            return str(out), "ok"

        with mock.patch("shared.services.dlss5_service.maybe_apply_global_dlss5", fake_dlss5), \
                mock.patch.object(global_rife, "_maybe_apply_global_rife_only", fake_rife):
            out, _msg = global_rife.maybe_apply_global_rife(None, str(base), {}, keep_name=keep_name)
        return base, out

    def test_batch_chain_keeps_the_name_and_leaves_no_copies(self):
        base, out = self._run(keep_name=True)
        self.assertEqual(Path(out), base)
        self.assertEqual(base.read_bytes(), b"upscaled+dlss5+rife")
        self.assertEqual(sorted(p.name for p in base.parent.iterdir()), ["clip.mp4"])

    def test_batch_rife_only_and_dlss5_only(self):
        base, out = self._run(keep_name=True, dlss5_on=False)
        self.assertEqual(Path(out), base)
        self.assertEqual(base.read_bytes(), b"upscaled+rife")
        _TempCase.tearDown(self)
        _TempCase.setUp(self)
        base, out = self._run(keep_name=True, rife_on=False)
        self.assertEqual(Path(out), base)
        self.assertEqual(base.read_bytes(), b"upscaled+dlss5")

    def test_single_run_names_unchanged(self):
        base, out = self._run(keep_name=False)
        self.assertEqual(Path(out).name, "clip_DLSS5_2xFPS.mp4")
        self.assertTrue(base.exists())


class KeepOnlyCleanupTests(_TempCase):
    def test_rerun_replaces_the_earlier_result_instead_of_numbering(self):
        _write(self.root / "clip.mp4", b"old run")
        new = _write(self.root / "clip" / "clip.mp4", b"new run")
        _write(self.root / "clip" / "processed_chunks" / "chunk_0001_upscaled.mp4")
        result = keep_only_batch_outputs(self.root, [str(new)])
        self.assertEqual(result["final_outputs"], [str((self.root / "clip.mp4").resolve())])
        self.assertEqual((self.root / "clip.mp4").read_bytes(), b"new run")
        self.assertFalse((self.root / "clip_0001.mp4").exists())
        self.assertFalse((self.root / "clip").exists())

    def test_batch_input_in_the_output_folder_is_never_replaced_or_deleted(self):
        # Batch output folder = input folder: clip.mp4 there is the input, not an earlier result.
        source = _write(self.root / "clip.mp4", b"input")
        other_input = _write(self.root / "other.mp4", b"other input")
        new = _write(self.root / "clip" / "clip.mp4", b"result")
        result = keep_only_batch_outputs(self.root, [str(new)], inputs=[str(source), str(other_input)])
        self.assertEqual(source.read_bytes(), b"input")
        self.assertEqual(other_input.read_bytes(), b"other input")
        self.assertEqual(result["final_outputs"], [str((self.root / "clip_0001.mp4").resolve())])

    def test_results_of_skipped_items_survive(self):
        root_result = _write(self.root / "older.mp4", b"flattened earlier")
        skipped = _write(self.root / "skipped" / "skipped.mp4", b"skipped item")
        _write(self.root / "skipped" / "skipped_comparison.mp4")
        new = _write(self.root / "clip" / "clip.mp4", b"new")
        result = keep_only_batch_outputs(self.root, [str(new)], also_keep=[str(skipped)])
        self.assertEqual(root_result.read_bytes(), b"flattened earlier")
        self.assertEqual((self.root / "skipped.mp4").read_bytes(), b"skipped item")
        self.assertEqual(result["kept_outputs"], 1)  # skipped results are kept, not reported
        self.assertFalse((self.root / "skipped" / "skipped_comparison.mp4").exists())

    def test_nothing_new_leaves_everything_untouched(self):
        skipped = _write(self.root / "skipped" / "skipped.mp4")
        result = keep_only_batch_outputs(self.root, [], also_keep=[str(skipped)])
        self.assertEqual(result["status"], "skipped")
        self.assertTrue(skipped.exists())

    def test_frame_folder_output_keeps_frame_names(self):
        frames = self.root / "clip" / "clip"
        _write(frames / "clip_000001.png")
        _write(frames / "clip_000002.png")
        _write(self.root / "clip" / "run_metadata.json")
        result = keep_only_batch_outputs(self.root, [str(frames)])
        self.assertEqual(sorted(Path(p).name for p in result["final_outputs"]), ["clip_000001.png", "clip_000002.png"])
        self.assertTrue((frames / "clip_000001.png").exists())
        self.assertFalse((self.root / "clip" / "run_metadata.json").exists())


class SkipFlattenedResultTests(_TempCase):
    def test_keep_only_result_in_the_batch_folder_counts_as_existing(self):
        src = _write(self.root / "in" / "clip.mov")
        out = self.root / "out"
        _write(out / "clip.mp4", b"earlier keep-only result")
        self.assertEqual(find_flattened_batch_output(out, "clip.mov", input_path=str(src)), out / "clip.mp4")
        kwargs = dict(input_path=str(src), model_label="test", mode="subprocess")
        self.assertIsNone(prepare_batch_video_run_dir(out, "clip.mov", flattened_output_counts=True, **kwargs))
        self.assertFalse((out / "clip").exists())
        # Without keep-only (or with overwrite) the item runs as before.
        self.assertIsNotNone(prepare_batch_video_run_dir(out, "clip.mov", overwrite_existing=True,
                                                         flattened_output_counts=True, **kwargs))

    def test_the_input_itself_is_not_a_result(self):
        src = _write(self.root / "clip.mp4")
        self.assertIsNone(find_flattened_batch_output(self.root, "clip.mp4", input_path=str(src)))


class FaceTabBatchTests(_TempCase):
    def _run(self, batch_in: Path, batch_out: str = "", overwrite: bool = False):
        from shared.services import face_service

        def fake_restore(path, output_path=None, **_kw):
            Path(output_path).write_bytes(Path(path).read_bytes() + b"+face")
            return str(output_path)

        cb = face_service.build_face_callbacks(None, {"output_dir": str(self.root / "app_out")}, [])
        values = dict(face_service.face_defaults([]))
        values.update({"batch_enable": True, "batch_input_path": str(batch_in), "batch_output_path": batch_out})
        args = [values.get(k) for k in face_service.FACE_ORDER]
        state = {"seed_controls": {"overwrite_existing_batch_val": overwrite}}
        with mock.patch.object(face_service, "restore_image", fake_restore), \
                mock.patch.object(face_service, "restore_video", fake_restore):
            result = cb["run_action"](None, *args, state=state)
        return result

    def test_outputs_keep_input_names_and_subfolders(self):
        batch_in = self.root / "in"
        _write(batch_in / "photo.png", b"p")
        _write(batch_in / "sub" / "photo.png", b"s")
        _write(batch_in / "clip.mp4", b"v")
        self._run(batch_in)
        out = batch_in / "upscaled_files"
        self.assertEqual((out / "photo.png").read_bytes(), b"p+face")
        self.assertEqual((out / "sub" / "photo.png").read_bytes(), b"s+face")
        self.assertEqual((out / "clip.mp4").read_bytes(), b"v+face")
        self.assertFalse(list(out.rglob("*_face_restored*")))
        self.assertFalse(list(out.rglob(".*face_tmp*")))

    def test_rerun_skips_or_overwrites_and_never_reads_its_own_outputs(self):
        batch_in = self.root / "in"
        _write(batch_in / "photo.png", b"p")
        self._run(batch_in)
        self._run(batch_in)  # skip: output exists, upscaled_files is not scanned as input
        out = batch_in / "upscaled_files"
        self.assertEqual(sorted(p.name for p in out.rglob("*") if p.is_file()), ["photo.png"])
        (batch_in / "photo.png").write_bytes(b"q")
        self._run(batch_in, overwrite=True)
        self.assertEqual((out / "photo.png").read_bytes(), b"q+face")

    def test_output_folder_equal_to_input_folder_never_overwrites_inputs(self):
        batch_in = self.root / "in"
        _write(batch_in / "photo.png", b"p")
        self._run(batch_in, batch_out=str(batch_in))
        self.assertEqual((batch_in / "photo.png").read_bytes(), b"p")
        self.assertEqual((batch_in / "photo_face_restored.png").read_bytes(), b"p+face")
        self._run(batch_in, batch_out=str(batch_in))
        self.assertFalse((batch_in / "photo_face_restored_face_restored.png").exists())


class Dlss5TabBatchTests(_TempCase):
    def _run(self, folder: Path, out: Path, overwrite: bool = False, image_format: str = "png"):
        from shared.services import dlss5_service

        calls = []

        def fake_run(settings, base_dir, on_progress=None, cancel_event=None):
            target = Path(settings["output_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(settings["input_path"]).read_bytes() + b"+dlss5")
            calls.append(target.name)
            return SimpleNamespace(returncode=0, output_path=str(target), log="")

        cb = dlss5_service.build_dlss5_callbacks(global_settings={"output_dir": str(self.root)}, base_dir=ROOT,
                                                 output_dir=self.root)
        tab = dict(cb["defaults"])
        tab.update({"batch_enable": True, "batch_input_path": str(folder), "batch_output_path": str(out),
                    "image_output_format": image_format})
        values = [tab.get(k) for k in dlss5_service.DLSS5_ORDER]
        state = {"seed_controls": {"overwrite_existing_batch_val": overwrite}}
        with mock.patch.object(dlss5_service, "run_dlss5", fake_run), \
                mock.patch("shared.generation_history.record_generation", lambda *a, **k: None):
            for _ in cb["run_action"](None, *values, state=state):
                pass
        return calls

    def test_batch_outputs_keep_input_names(self):
        folder = self.root / "in"
        _write(folder / "clip.mp4", b"v")
        _write(folder / "photo.jpg", b"i")
        out = self.root / "out"
        self._run(folder, out)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["clip.mp4", "photo.png"])
        self.assertEqual(self._run(folder, out), [])  # rerun: existing outputs skipped
        self.assertEqual(sorted(self._run(folder, out, overwrite=True)), ["clip.mp4", "photo.png"])

    def test_same_stem_inputs_do_not_share_one_output_and_inputs_are_never_targets(self):
        folder = self.root / "in"
        _write(folder / "clip.mp4", b"a")
        _write(folder / "clip.mov", b"b")
        calls = self._run(folder, self.root / "out")
        self.assertEqual(len(set(calls)), 2)
        self._run(folder, folder)
        self.assertEqual((folder / "clip.mp4").read_bytes(), b"a")
        self.assertTrue((folder / "clip_DLSS5.mp4").exists())


class SeedVR2ImageFormatTests(_TempCase):
    def test_jpg_reencode_keeps_the_name(self):
        from PIL import Image
        from shared.services.seedvr2_service import _apply_image_output_preferences

        src = self.root / "photo.jpg"
        Image.new("RGB", (8, 8), (200, 10, 10)).save(src, format="JPEG", quality=100)
        out = _apply_image_output_preferences(str(src), "jpg", 80)
        self.assertEqual(Path(out), src)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["photo.jpg"])

    def test_format_change_still_converts(self):
        from PIL import Image
        from shared.services.seedvr2_service import _apply_image_output_preferences

        src = self.root / "photo.png"
        Image.new("RGB", (8, 8)).save(src, format="PNG")
        out = _apply_image_output_preferences(str(src), "webp", 90)
        self.assertEqual(Path(out), self.root / "photo.webp")
        self.assertFalse(src.exists())


class RtxBatchFolderTests(_TempCase):
    def test_frame_folder_inside_the_item_folder(self):
        from shared.rtx_superres_runner import _make_sequence_output_dir

        run_dir = self.root / "clip"
        run_dir.mkdir()
        kw = dict(input_path=str(self.root / "clip.mp4"), global_output_dir=None, original_filename="clip.mp4")
        self.assertEqual(_make_sequence_output_dir(output_override=str(run_dir), keep_name=True, **kw), run_dir / "clip")
        self.assertEqual(_make_sequence_output_dir(output_override=str(run_dir), **kw).name, "clip_0001")

    def test_dotted_input_name_keeps_all_its_parts(self):
        from shared.rtx_superres_runner import _make_video_output_path

        run_dir = self.root / "my.clip"
        run_dir.mkdir()
        kw = dict(input_path=str(self.root / "my.clip.mp4"), global_output_dir=None, output_override=str(run_dir),
                  original_filename="my.clip.mp4")
        self.assertEqual(_make_video_output_path(keep_name=True, **kw), run_dir / "my.clip.mp4")


if __name__ == "__main__":
    unittest.main()
