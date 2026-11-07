"""Models_Downloader.py resumes an interrupted download instead of starting it again (V10.1 fix, 7 November 2026).

Seen while recording the tutorial: Ctrl+C during a native Xet download printed "[INTERRUPTED] ...: the transferred
bytes were kept for the next run", but the re-run printed "[DOWNLOADING]" (not "[RESUMING]") and fetched the whole
file again.

Cause: the byte map (<file>.part.json) is written by _write_state(), which first flushes the staging file with
os.fsync() on a descriptor opened O_RDONLY. Windows flushes only through a handle opened for writing: the fsync failed
with EBADF, the OSError was logged at debug level and the map was never written, on every save (Xet and HTTP alike).
Without a map the re-run discards the staging file and starts from zero.

The downloader is a release file one folder above the app (as in every install); the tests are skipped without it.
A fake hf_xet stands in for the network: it delivers the first bytes and then the user presses Ctrl+C.
"""
from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADER = ROOT.parent / "Models_Downloader.py"

SIZE = 4096
CONTENT = bytes(i % 251 for i in range(SIZE))


def _load_downloader():
    spec = importlib.util.spec_from_file_location("Models_Downloader_resume_test", DOWNLOADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeXet:
    """hf_xet's session / stream group / unordered stream, serving CONTENT; Ctrl+C after `stop_after` bytes."""

    def __init__(self, stop_after=None):
        self.stop_after = stop_after
        self.sent = 0
        self.module = SimpleNamespace(
            XetConfig=lambda: SimpleNamespace(with_config=lambda updates: "config"),
            XetSession=lambda config: self,
            XetFileInfo=lambda file_hash, size: ("info", file_hash, size),
        )

    def new_download_stream_group(self, **kwargs):
        return self

    def sigint_abort(self):
        pass

    def download_unordered_stream(self, info, start, end):
        fake = self

        class _Stream:
            def __iter__(self):
                for offset in range(start, end, 512):
                    if fake.stop_after is not None and fake.sent >= fake.stop_after:
                        raise KeyboardInterrupt  # the user's Ctrl+C
                    data = CONTENT[offset:min(end, offset + 512)]
                    fake.sent += len(data)
                    yield offset - start, data

            def cancel(self):
                pass

        return _Stream()


@unittest.skipUnless(DOWNLOADER.is_file(), "Models_Downloader.py is not one folder above the app")
class DownloaderResumeTests(unittest.TestCase):
    def setUp(self):
        self.md = _load_downloader()
        self.tmp = Path(tempfile.mkdtemp(prefix="downloader_resume_"))
        self._cache = mock.patch.object(self.md, "CACHE_DIR", self.tmp / "download_cache")
        self._cache.start()
        self.remote = self.md.RemoteFile("example/repo", "model.safetensors", SIZE, "sha256",
                                         hashlib.sha256(CONTENT).hexdigest(), "c0ffee")
        self.target = self.tmp / "models" / "model.safetensors"

    def tearDown(self):
        self._cache.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _downloader(self, fake, log):
        dl = self.md.RobustDownloader({})
        dl.log = log.append
        dl.show_progress_line = lambda text: None
        dl.finalize_progress_line = lambda final_text=None: log.append(final_text) if final_text else None
        dl._xet_details = lambda remote: (fake.module, "xethash", "https://example.invalid/refresh")
        return dl

    def test_the_byte_map_is_saved(self):
        dl = self._downloader(_FakeXet(), [])
        staging = dl._staging_path(self.target)
        dl._create_staging(staging, SIZE)
        dl._write_state(self.target, self.remote, self.md._Ranges([(0, 1536)]))
        self.assertTrue(dl._state_path(self.target).is_file(), "the resume map was not written")
        self.assertEqual(dl._read_state(self.target, self.remote).total, 1536)

    def test_ctrl_c_during_a_xet_download_resumes_on_the_next_run(self):
        first_log = []
        dl = self._downloader(_FakeXet(stop_after=1536), first_log)
        with self.assertRaises(KeyboardInterrupt):
            dl.ensure_file(self.remote, self.target)
        self.assertTrue(any("[INTERRUPTED]" in line for line in first_log), first_log)
        self.assertTrue(dl._state_path(self.target).is_file(), "Ctrl+C left no resume map")

        # the re-run (a new process in real life): it must continue, not start again
        second_log = []
        fake = _FakeXet()
        dl2 = self._downloader(fake, second_log)
        self.assertTrue(dl2.ensure_file(self.remote, self.target), second_log)
        self.assertTrue(any(line and line.startswith("[RESUMING]") for line in second_log), second_log)
        self.assertFalse(any(line and line.startswith("[DOWNLOADING]") for line in second_log), second_log)
        self.assertEqual(fake.sent, SIZE - 1536, "the re-run fetched bytes that were already on disk")
        self.assertEqual(self.target.read_bytes(), CONTENT)


if __name__ == "__main__":
    unittest.main()
