"""Commit video segments and the matching Wan state at keyframe boundaries."""

import hashlib
import json
import subprocess
from pathlib import Path

import torch
from filelock import FileLock


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


class ResumeStore:
    def __init__(self, root, settings):
        key = hashlib.sha256(json.dumps({"version": 1, "settings": settings}, sort_keys=True).encode()).hexdigest()[:24]
        self.root = Path(root).resolve() / key
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "segments.json"
        self.lock = FileLock(str(self.root / "run.lock"), timeout=0)
        self.frame = 0
        self.count = 0
    def load(self):
        if self.manifest.is_file():
            data = json.loads(self.manifest.read_text(encoding="utf-8"))
            self.frame, self.count = data["frame"], data["count"]
        if not self.count:
            return None
        data = torch.load(self.state(self.count), map_location="cpu", weights_only=True)
        if data["frame"] != self.frame or not all(self.part(i).is_file() for i in range(self.count)):
            raise RuntimeError(f"RelayVSR resume manifest and part files disagree in {self.root}")
        return data

    def part(self, index):
        return self.root / f"part-{index:07d}.mp4"

    def state(self, index):
        return self.root / f"state-{index % 2}.pt"

    def commit(self, checkpoint):
        new_count = self.count + 1
        temp = self.root / "state.tmp"
        torch.save(checkpoint, temp)
        temp.replace(self.state(new_count))
        temp = self.root / "segments.tmp"
        temp.write_text(json.dumps(dict(frame=checkpoint["frame"], count=new_count)), encoding="utf-8")
        temp.replace(self.manifest)
        self.count, self.frame = new_count, checkpoint["frame"]

    def finalize(self, ffmpeg, output, audio, offset, duration, has_tail):
        parts = self.root / "parts.ffconcat"
        parts.write_text(
            "ffconcat version 1.0\n" + "".join(
                f"file '{self.part(i).name}'\n" for i in range(self.count + int(has_tail))
            ),
            encoding="utf-8",
        )
        audio_input = ["-ss", f"{offset:.9f}", "-t", f"{duration:.9f}", "-i", audio] if audio else []
        audio_map = ["-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k"] if audio else []
        subprocess.run([ffmpeg, "-v", "error", "-y", "-nostdin", "-f", "concat", "-safe", "1", "-i", str(parts),
                        *audio_input, "-map", "0:v:0", "-c:v", "copy", *audio_map, output],
                       check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
