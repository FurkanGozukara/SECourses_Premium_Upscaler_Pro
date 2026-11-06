"""Past Generations: one index of everything the app has made.

Sources, merged and de-duplicated by output file:
  * run folders (``<outputs>/0060/``, batch items ``<outputs>/<stem>/``): run_context.json + run_summary.json
    (the summary keeps every setting of the run, so "Reload parameters" restores the tab exactly);
  * Chains runs (``<outputs>/chains/NNNN/chain.json``) with every step's result;
  * the generation journal (``history_data/journal.jsonl``): every run summary and every DLSS 5 result appends a
    line, so outputs written to custom batch / override folders outside the outputs folder are found too;
  * loose media (single-image outputs such as ``0003_photo_DLSS5.png``, batch output folders).

Everything here is cached: per-folder records by folder / metadata mtimes, media probes and thumbnails by
path + size + mtime, so opening the tab again costs a few ``stat`` calls.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import threading
import time
import urllib.parse
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from shared.path_utils import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "history_data"
JOURNAL_PATH = DATA_DIR / "journal.jsonl"
THUMB_SIZE = 560
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW: never flash a console for ffmpeg

# key: (label, icon, color, universal-preset tab or "")
ENGINES: Dict[str, Tuple[str, str, str, str]] = {
    "seedvr2": ("SeedVR2", "🎬", "#8b5cf6", "seedvr2"),
    "flashvsr": ("FlashVSR+", "⚡", "#f59e0b", "flashvsr"),
    "relayvsr": ("RelayVSR", "🔁", "#06b6d4", "relayvsr"),
    "open_h3": ("Open-H3", "🌀", "#ec4899", "open_h3"),
    "ltx25": ("LTX 2.5", "🎞️", "#f97316", "ltx25"),
    "sparkvsr": ("SparkVSR", "✨", "#eab308", "sparkvsr"),
    "rtx": ("RTX Super Res", "🚀", "#22c55e", "rtx"),
    "dlss5": ("DLSS 5", "💎", "#14b8a6", "dlss5"),
    "gan": ("Image GAN", "🖼️", "#3b82f6", "gan"),
    "rife": ("RIFE FPS", "⏱️", "#a855f7", "rife"),
    "face": ("Face Restore", "👤", "#f43f5e", "face"),
    "chain": ("Chain", "🔗", "#6366f1", "chains"),
    "other": ("Other", "📄", "#64748b", ""),
}
# Engines that only take videos (Chains VIDEO_ONLY): "Send to" offers them for videos only.
VIDEO_ONLY_TABS = {"ltx25", "rife"}
SEND_TABS = ["seedvr2", "flashvsr", "relayvsr", "open_h3", "sparkvsr", "rtx", "dlss5", "gan", "ltx25", "rife", "face",
             "chains"]

_ALIASES = {
    "rtx_super_resolution": "rtx", "rtxsuperresolution": "rtx", "rtx": "rtx",
    "flashvsr": "flashvsr", "flashvsrplus": "flashvsr", "flashvsr+": "flashvsr",
    "seedvr2": "seedvr2", "relayvsr": "relayvsr", "openh3": "open_h3", "open_h3": "open_h3",
    "ltx25": "ltx25", "ltx2.5": "ltx25", "sparkvsr": "sparkvsr", "dlss5": "dlss5", "dlss": "dlss5",
    "gan": "gan", "imagegan": "gan", "realesrgan": "gan", "rife": "rife", "face": "face", "chain": "chain",
    "chains": "chain",
}
# Files in a run folder that are not the result itself.
_HELPER_SUFFIXES = ("_comparison", "_chunk_metadata", "_partial", "_preview")
_HELPER_PREFIXES = ("downscaled_", "cancelled_partial", "chunk_")
_HELPER_MARKERS = (".__ui_preview_", ".__h264_preview_", ".tmp.", ".part")
_SKIP_DIRS = {".run_ids", ".history", "direct_video_compare", "input_chunks", "processed_chunks", "pre_processed",
              "thumbs", ".ui_preview_cache", "playwright", "history_data"}
MEDIA_EXTS = set(VIDEO_EXTENSIONS) | set(IMAGE_EXTENSIONS)


def engine_key(raw: Any) -> str:
    text = str(raw or "").strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    if not text:
        return "other"
    if text in _ALIASES:
        return _ALIASES[text]
    for alias, key in _ALIASES.items():
        if alias.replace("_", "") and text.startswith(alias.replace("_", "")):
            return key
    return "other"


def engine_from_name(name: str) -> str:
    stem = Path(name).stem.lower()
    if stem.endswith("_dlss5") or "_dlss5_" in stem:
        return "dlss5"
    if stem.endswith("_face_restored"):
        return "face"
    if stem.endswith("_chain"):
        return "chain"
    if "xfps" in stem or stem.endswith("_rife"):
        return "rife"
    return "other"


def _norm(path: Any) -> str:
    try:
        return os.path.normcase(os.path.abspath(str(path)))
    except Exception:
        return str(path)


def _hid(path: Any) -> str:
    return hashlib.sha1(_norm(path).encode("utf-8", "surrogatepass")).hexdigest()[:14]


def _stat_key(path: Path) -> str:
    try:
        st = path.stat()
        return f"{_norm(path)}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        return ""


def file_url(path: Any) -> str:
    """Gradio file route for a local path (served through launch(allowed_paths=...)), cache-busted by mtime."""
    p = str(path).replace("\\", "/")
    try:
        version = int(Path(str(path)).stat().st_mtime)
    except OSError:
        version = 0
    return f"/gradio_api/file={urllib.parse.quote(p, safe=':/')}?v={version}"


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _parse_ts(text: Any) -> Optional[float]:
    raw = str(text or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(raw[:26], fmt).timestamp()
        except ValueError:
            continue
    return None


def is_media(path: Path) -> bool:
    return path.suffix.lower() in MEDIA_EXTS


def kind_of(path: Any) -> str:
    suffix = Path(str(path)).suffix.lower()
    return "video" if suffix in VIDEO_EXTENSIONS else "image" if suffix in IMAGE_EXTENSIONS else "none"


def is_helper_file(name: str) -> bool:
    low = name.lower()
    stem = Path(low).stem
    return (any(stem.endswith(s) for s in _HELPER_SUFFIXES) or low.startswith(_HELPER_PREFIXES)
            or any(m in low for m in _HELPER_MARKERS))


# --------------------------------------------------------------------------------------------------
# Journal: one JSON line per finished generation (written by RunLogger.write_summary and DLSS 5).
# --------------------------------------------------------------------------------------------------

_journal_lock = threading.Lock()


def journal_append(entry: Dict[str, Any]) -> None:
    """Best effort; never raises (a history problem must not fail a finished upscale)."""
    try:
        out = str(entry.get("output") or "").strip()
        if not out:
            return
        row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **{k: v for k, v in entry.items() if v not in (None, "")}}
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False, default=str)
        with _journal_lock, JOURNAL_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def record_generation(output: Any, source: Any, engine: str, settings: Optional[Dict[str, Any]] = None,
                      status: str = "ok") -> None:
    """Journal a result that has no run_summary.json of its own (DLSS 5 images, batch folders)."""
    clean = {k: v for k, v in (settings or {}).items() if isinstance(v, (str, int, float, bool)) or v is None}
    journal_append({"output": str(output or ""), "input": str(source or ""), "pipeline": engine, "status": status,
                    "args": clean})


def _read_journal() -> Dict[str, Dict[str, Any]]:
    """{normalized output path: newest entry}."""
    out: Dict[str, Dict[str, Any]] = {}
    try:
        with JOURNAL_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("output"):
                    out[_norm(row["output"])] = row
    except OSError:
        pass
    return out


# --------------------------------------------------------------------------------------------------
# Media probing and thumbnails (cached by path + size + mtime)
# --------------------------------------------------------------------------------------------------

def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def probe_media(path: Path) -> Dict[str, Any]:
    """Width, height, duration, fps, frames, codec and audio of a video, or the size of an image."""
    info: Dict[str, Any] = {}
    kind = kind_of(path)
    if kind == "image":
        try:
            from PIL import Image

            with Image.open(path) as im:
                info["w"], info["h"] = int(im.size[0]), int(im.size[1])
        except Exception:
            pass
        return info
    if kind != "video":
        return info
    ffprobe = _which("ffprobe")
    if not ffprobe:
        return info
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries",
             "stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames:format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=20, creationflags=_NO_WINDOW,
        )
        data = json.loads(proc.stdout or "{}")
    except Exception:
        return info
    for stream in data.get("streams") or []:
        if stream.get("codec_type") == "video" and "w" not in info:
            info["w"], info["h"] = int(stream.get("width") or 0), int(stream.get("height") or 0)
            info["codec"] = str(stream.get("codec_name") or "")
            for key in ("avg_frame_rate", "r_frame_rate"):
                num, _, den = str(stream.get(key) or "0/0").partition("/")
                try:
                    fps = float(num) / float(den or 1)
                except (ValueError, ZeroDivisionError):
                    fps = 0.0
                if fps > 0:
                    info["fps"] = round(fps, 3)
                    break
            try:
                info["frames"] = int(stream.get("nb_frames") or 0)
            except ValueError:
                pass
        elif stream.get("codec_type") == "audio":
            info["audio"] = True
    try:
        info["dur"] = round(float((data.get("format") or {}).get("duration") or 0.0), 3)
    except ValueError:
        pass
    if not info.get("frames") and info.get("fps") and info.get("dur"):
        info["frames"] = int(round(info["fps"] * info["dur"]))
    return info


def browser_playable(info: Dict[str, Any], path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".webm", ".m4v", ".mov", ".mkv"} and \
        str(info.get("codec") or "").lower() in {"h264", "vp8", "vp9", "av1"}


class _ProbeCache:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Dict[str, Any]] = {}
        self.dirty = False
        self.lock = threading.Lock()
        loaded = _load_json(path)
        if isinstance(loaded, dict):
            self.data = loaded

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.data.get(key) if key else None

    def put(self, key: str, info: Dict[str, Any]) -> None:
        if not key:
            return
        with self.lock:
            self.data[key] = info
            self.dirty = True

    def save(self) -> None:
        with self.lock:
            if not self.dirty:
                return
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.data, separators=(",", ":")), encoding="utf-8")
                tmp.replace(self.path)
                self.dirty = False
            except OSError:
                pass


def _thumb_path(source: Path) -> Optional[Path]:
    key = _stat_key(source)
    if not key:
        return None
    return DATA_DIR / "thumbs" / (hashlib.sha1(key.encode("utf-8", "surrogatepass")).hexdigest()[:24] + ".jpg")


def make_thumbnail(source: Path, duration: float = 0.0) -> Optional[Path]:
    """A small JPEG of an image or of a video frame at 30 % of its length (cached)."""
    target = _thumb_path(source)
    if target is None:
        return None
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.stem + f".{os.getpid()}.{threading.get_ident()}.tmp.jpg")
    try:
        if kind_of(source) == "image":
            from PIL import Image

            with Image.open(source) as im:
                im.draft("RGB", (THUMB_SIZE * 2, THUMB_SIZE * 2))
                im = im.convert("RGBA") if im.mode in ("P", "LA") else im
                if im.mode == "RGBA":
                    bg = Image.new("RGB", im.size, (24, 28, 40))
                    bg.paste(im, mask=im.split()[-1])
                    im = bg
                im = im.convert("RGB")
                im.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
                im.save(tmp, "JPEG", quality=86, optimize=True)
        else:
            ffmpeg = _which("ffmpeg")
            if not ffmpeg:
                return None
            at = max(0.0, min(duration * 0.3, max(0.0, duration - 0.1))) if duration else 0.0
            scale = f"scale='min({THUMB_SIZE},iw)':'min({THUMB_SIZE},ih)':force_original_aspect_ratio=decrease"
            cmd = [ffmpeg, "-v", "error", "-y", "-ss", f"{at:.3f}", "-i", str(source), "-frames:v", "1",
                   "-vf", scale, "-q:v", "4", str(tmp)]
            proc = subprocess.run(cmd, capture_output=True, timeout=45, creationflags=_NO_WINDOW)
            if (proc.returncode != 0 or not tmp.is_file()) and at > 0:
                cmd[5] = "0"
                subprocess.run(cmd, capture_output=True, timeout=45, creationflags=_NO_WINDOW)
        if tmp.is_file() and tmp.stat().st_size > 0:
            os.replace(tmp, target)
            return target
    except Exception:
        pass
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    return None


def browser_preview(source: Path) -> Optional[Path]:
    """An H.264 copy (long side <= 1920) of a video the browser cannot decode (HEVC, ProRes, MPEG-4...)."""
    ffmpeg = _which("ffmpeg")
    key = _stat_key(source)
    if not ffmpeg or not key:
        return None
    target = DATA_DIR / "previews" / (hashlib.sha1(key.encode("utf-8", "surrogatepass")).hexdigest()[:24] + ".mp4")
    if target.is_file() and target.stat().st_size > 1024:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp.mp4")
    cmd = [ffmpeg, "-v", "error", "-y", "-i", str(source), "-map", "0:v:0", "-map", "0:a?",
           "-vf", "scale='min(1920,iw)':'min(1920,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(tmp)]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=3600, creationflags=_NO_WINDOW)
        if proc.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 1024:
            os.replace(tmp, target)
            return target
    except Exception:
        pass
    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        pass
    return None


# --------------------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------------------

def _status_of(text: str, returncode: Any, has_output: bool) -> str:
    low = str(text or "").lower()
    if "cancel" in low:
        return "cancelled"
    try:
        rc = int(returncode) if returncode not in (None, "") else None
    except (TypeError, ValueError):
        rc = None
    if rc == 0 or any(w in low for w in ("complete", "success", "[ok]", "✅", "done")):
        return "ok" if has_output else "failed"
    if rc not in (None, 0) or any(w in low for w in ("error", "fail", "❌", "exited with code", "[warn]")):
        return "failed"
    return "ok" if has_output else "incomplete"


def _model_of(args: Dict[str, Any], engine: str) -> str:
    for key in ("dit_model", "wan_model", "model_name", "model", "restoration_model", "profile", "quality_preset"):
        value = args.get(key)
        if value not in (None, "") and not isinstance(value, (dict, list)):
            text = str(value)
            for ext in (".safetensors", ".pth", ".ckpt", ".gguf", ".onnx"):
                if text.lower().endswith(ext):
                    text = text[: -len(ext)]
            if engine == "open_h3" and str(args.get("workflow") or "").startswith("JonXL"):
                return f"JonXL · {text}"
            return text
    return ""


def _summary_entry(run_dir: Path, output_norm: Optional[str]) -> Optional[Dict[str, Any]]:
    payload = _load_json(run_dir / "run_summary.json")
    entries = [payload] if isinstance(payload, dict) else [e for e in (payload or []) if isinstance(e, dict)]
    if not entries:
        return None
    if output_norm:
        for entry in reversed(entries):
            if entry.get("output") and _norm(entry["output"]) == output_norm:
                return entry
    for entry in reversed(entries):
        if isinstance(entry.get("args"), dict):
            return entry
    return entries[-1]


def _media_in_dir(folder: Path) -> List[Path]:
    try:
        files = [p for p in folder.iterdir() if p.is_file() and is_media(p) and not is_helper_file(p.name)]
    except OSError:
        return []
    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return files


def _comparison_for(output: Path) -> Optional[Path]:
    for ext in (".mp4", output.suffix):
        cand = output.with_name(f"{output.stem}_comparison{ext}")
        if cand.is_file():
            return cand
    return None


def _base_record(output: Optional[Path], engine: str, rid_source: Any) -> Dict[str, Any]:
    label, icon, color, _tab = ENGINES.get(engine, ENGINES["other"])
    return {"id": _hid(rid_source), "engine": engine, "out": str(output) if output else "",
            "kind": kind_of(output) if output else "none", "inp": "", "dir": "", "ts": 0.0, "elapsed": None,
            "status": "ok", "status_text": "", "model": "", "args": None, "steps": None, "extras": [],
            "cmp": "", "batch": False, "chain_name": "", "target": "", "src": ""}


def record_from_run_dir(run_dir: Path, journal: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ctx = _load_json(run_dir / "run_context.json")
    ctx = ctx if isinstance(ctx, dict) else {}
    candidates: List[Path] = []
    for raw in (ctx.get("last_output_path"),):
        if raw and Path(str(raw)).is_file() and is_media(Path(str(raw))):
            candidates.append(Path(str(raw)))
    summary = _summary_entry(run_dir, _norm(candidates[0]) if candidates else None)
    if not candidates and summary and summary.get("output"):
        p = Path(str(summary["output"]))
        if p.is_file() and is_media(p):
            candidates.append(p)
    media = _media_in_dir(run_dir)
    if not candidates and media:
        candidates.append(media[0])
    output = candidates[0] if candidates else None
    jrow = journal.get(_norm(output)) if output else None
    engine = engine_key(ctx.get("pipeline") or (summary or {}).get("pipeline") or (jrow or {}).get("pipeline")
                        or ctx.get("model"))
    if engine == "other" and output:
        engine = engine_from_name(output.name)
    rec = _base_record(output, engine, run_dir)
    rec["dir"] = str(run_dir)
    args = (summary or {}).get("args") if isinstance((summary or {}).get("args"), dict) else None
    if args is None and jrow and isinstance(jrow.get("args"), dict):
        args = jrow["args"]
    rec["args"] = args
    rec["src"] = "run"
    inp = (args or {}).get("_original_input_path_before_preprocess") or ctx.get("original_input_path") or \
        ctx.get("input_path") or (summary or {}).get("input") or (args or {}).get("input_path") or ""
    rec["inp"] = str(inp or "")
    status_text = str(ctx.get("last_status") or (summary or {}).get("status") or "")
    returncode = ctx.get("last_returncode", (summary or {}).get("returncode"))
    rec["status_text"] = status_text
    rec["status"] = _status_of(status_text, returncode, output is not None)
    if output is not None and "_partial" in output.stem.lower():
        rec["status"] = "partial"
    started = _parse_ts(ctx.get("created_at"))
    finished = _parse_ts(ctx.get("last_updated_at")) or _parse_ts((summary or {}).get("timestamp"))
    if finished is None:
        try:
            finished = (output or run_dir).stat().st_mtime
        except OSError:
            finished = started or 0.0
    rec["ts"] = float(finished or started or 0.0)
    if started and finished and 0 <= finished - started < 7 * 86400 and rec["status"] == "ok":
        rec["elapsed"] = round(finished - started, 1)
    rec["model"] = _model_of(args or {}, engine)
    rec["target"] = str((summary or {}).get("target") or (args or {}).get("_preprocess_plan_note") or "")
    rec["batch"] = bool((args or {}).get("batch_enable")) or not run_dir.name.isdigit()
    if output is not None:
        cmp = _comparison_for(output)
        rec["cmp"] = str(cmp) if cmp else ""
        rec["extras"] = [str(p) for p in media if p != output][:12]
    return rec


def record_from_chain_dir(chain_dir: Path) -> Optional[Dict[str, Any]]:
    data = _load_json(chain_dir / "chain.json")
    if not isinstance(data, dict):
        return None
    media = _media_in_dir(chain_dir)
    finals = [p for p in media if p.stem.endswith("_chain")] or media
    results = [r for r in (data.get("results") or []) if isinstance(r, dict)]
    output = finals[0] if finals else None
    if output is None:
        for r in reversed(results):
            if r.get("output") and Path(str(r["output"])).is_file():
                output = Path(str(r["output"]))
                break
    step_engines = [engine_key((s or {}).get("engine")) for s in (data.get("steps") or []) if (s or {}).get("enabled", True)]
    # a one-step chain is a run of that engine (it still reloads as a chain)
    engine = step_engines[0] if len(step_engines) == 1 and step_engines[0] in ENGINES else "chain"
    rec = _base_record(output, engine, chain_dir)
    rec["dir"] = str(chain_dir)
    rec["src"] = "chain"
    rec["inp"] = str(data.get("source") or data.get("chain_input") or "")
    rec["chain_name"] = str(data.get("name") or "")
    statuses = [str(r.get("status") or "") for r in results]
    if any(s == "failed" for s in statuses):
        rec["status"] = "failed"
    elif finals and finals[0].stem.endswith("_chain"):
        rec["status"] = "ok"
    elif any(s in ("running", "pending", "") for s in statuses):
        rec["status"] = "incomplete"
    else:
        rec["status"] = "ok" if output else "failed"
    rec["status_text"] = " → ".join(f"{r.get('engine')}: {r.get('status')}" for r in results)
    steps = []
    for r in results:
        out = str(r.get("output") or "")
        meta = ENGINES.get(engine_key(r.get("engine")), ENGINES["other"])
        steps.append({"index": r.get("index"), "engine": engine_key(r.get("engine")), "label": meta[0], "icon": meta[1],
                      "status": r.get("status"), "out": out if out and Path(out).is_file() else "",
                      "seconds": r.get("seconds")})
    rec["steps"] = steps
    runs: List[List[Any]] = []  # consecutive repeats collapse: DLSS 5 → DLSS 5 → DLSS 5 = "DLSS 5 ×3"
    for s in steps:
        if runs and runs[-1][0] == s["label"]:
            runs[-1][1] += 1
        else:
            runs.append([s["label"], 1])
    rec["model"] = " → ".join(f"{label} ×{n}" if n > 1 else label for label, n in runs)
    started, finished = _parse_ts(data.get("started")), _parse_ts(data.get("updated"))
    rec["ts"] = float(finished or started or (chain_dir.stat().st_mtime if chain_dir.exists() else 0.0))
    secs = [float(r.get("seconds") or 0) for r in results if r.get("seconds")]
    if secs and rec["status"] == "ok":
        rec["elapsed"] = round(sum(secs), 1)
    if output is not None:
        cmp = _comparison_for(output)
        rec["cmp"] = str(cmp) if cmp else ""
    return rec


def record_from_file(path: Path, journal: Dict[str, Dict[str, Any]], batch: bool = False) -> Dict[str, Any]:
    jrow = journal.get(_norm(path)) or {}
    engine = engine_key(jrow.get("pipeline")) if jrow.get("pipeline") else engine_from_name(path.name)
    rec = _base_record(path, engine, path)
    rec["src"] = "file"
    rec["dir"] = str(path.parent)
    rec["batch"] = batch
    args = jrow.get("args") if isinstance(jrow.get("args"), dict) else None
    rec["args"] = args
    rec["inp"] = str(jrow.get("input") or (args or {}).get("input_path") or "")
    rec["status"] = "failed" if str(jrow.get("status") or "ok") not in ("ok", "") else "ok"
    rec["ts"] = _parse_ts(jrow.get("ts")) or (path.stat().st_mtime if path.exists() else 0.0)
    rec["model"] = _model_of(args or {}, engine)
    cmp = _comparison_for(path)
    rec["cmp"] = str(cmp) if cmp else ""
    return rec


# --------------------------------------------------------------------------------------------------
# The index
# --------------------------------------------------------------------------------------------------

class GenerationHistory:
    """Thread-safe, incrementally refreshed index of the outputs folder (one per app process)."""

    def __init__(self, root_fn: Callable[[], Path]):
        self.root_fn = root_fn
        self.root_override: Optional[Path] = None
        self.lock = threading.RLock()
        self.dir_cache: Dict[str, Tuple[str, Optional[Dict[str, Any]]]] = {}
        self.records: Dict[str, Dict[str, Any]] = {}
        self.probes = _ProbeCache(DATA_DIR / "probe_cache.json")
        self.pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="history")
        self.pending: Dict[str, Any] = {}
        self.journal_sig = ""
        self.journal: Dict[str, Dict[str, Any]] = {}
        self.favorites = self._load_favorites()

    # ---- persistence ----------------------------------------------------------------------------
    def _load_favorites(self) -> set:
        data = _load_json(DATA_DIR / "favorites.json")
        return set(data.get("paths") or []) if isinstance(data, dict) else set()

    def _save_favorites(self) -> None:
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            (DATA_DIR / "favorites.json").write_text(json.dumps({"paths": sorted(self.favorites)}, indent=1),
                                                     encoding="utf-8")
        except OSError:
            pass

    def root(self) -> Path:
        if self.root_override is not None:
            return self.root_override
        try:
            return Path(self.root_fn())
        except Exception:
            return APP_DIR / "outputs"

    def follow_state(self, state: Any) -> None:
        """Use the outputs folder of the user's session (Global Settings), where the engines write new runs."""
        try:
            seed = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
            folder = str(((seed or {}).get("global_settings") or {}).get("output_dir") or "").strip()
            self.root_override = Path(folder) if folder and Path(folder).is_dir() else None
        except Exception:
            self.root_override = None

    # ---- scanning ---------------------------------------------------------------------------------
    def _journal(self) -> Dict[str, Dict[str, Any]]:
        sig = _stat_key(JOURNAL_PATH)
        if sig != self.journal_sig:
            self.journal = _read_journal()
            self.journal_sig = sig
        return self.journal

    def _dir_sig(self, folder: Path, names: Iterable[str]) -> str:
        parts = []
        for p in [folder] + [folder / n for n in names]:
            try:
                parts.append(str(p.stat().st_mtime_ns))
            except OSError:
                parts.append("-")
        return "|".join(parts)

    def _run_sig(self, folder: Path) -> str:
        sig = self._dir_sig(folder, ["run_context.json", "run_summary.json"])
        # only folders without a summary take their settings from the journal (DLSS 5)
        return sig if (folder / "run_summary.json").is_file() else sig + "|" + self.journal_sig

    def _cached(self, key: str, sig: str, build: Callable[[], Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        hit = self.dir_cache.get(key)
        if hit and hit[0] == sig:
            return hit[1]
        rec = build()
        self.dir_cache[key] = (sig, rec)
        return rec

    def _scan_folder(self, folder: Path, journal, found: Dict[str, Dict[str, Any]], depth: int, seen_out: set) -> None:
        try:
            entries = list(os.scandir(folder))
        except OSError:
            return
        names = {e.name for e in entries}
        if "chain_batch_report.json" in names:
            return  # Chains batch copies: the chain run folders already list these results
        for entry in entries:
            name = entry.name
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            path = Path(entry.path)
            if is_dir:
                if name in _SKIP_DIRS or name.startswith("."):
                    continue
                if name == "chains" and depth == 0:
                    try:
                        chain_dirs = [Path(c.path) for c in os.scandir(path) if c.is_dir()]
                    except OSError:
                        chain_dirs = []
                    for cdir in chain_dirs:
                        if (cdir / "chain.json").is_file():
                            rec = self._cached(_norm(cdir), self._dir_sig(cdir, ["chain.json"]),
                                               lambda d=cdir: record_from_chain_dir(d))
                            if rec:
                                found[rec["id"]] = rec
                                if rec["out"]:
                                    seen_out.add(_norm(rec["out"]))
                    continue
                if (path / "run_context.json").is_file() or (path / "run_summary.json").is_file():
                    rec = self._cached(_norm(path), self._run_sig(path), lambda d=path: record_from_run_dir(d, journal))
                    if rec and (rec["out"] or rec["status"] != "incomplete" or rec["args"]):
                        found[rec["id"]] = rec
                        if rec["out"]:
                            seen_out.add(_norm(rec["out"]))
                    continue
                if (path / "chain.json").is_file():
                    rec = self._cached(_norm(path), self._dir_sig(path, ["chain.json"]),
                                       lambda d=path: record_from_chain_dir(d))
                    if rec:
                        found[rec["id"]] = rec
                    continue
                if depth < 2:
                    self._scan_folder(path, journal, found, depth + 1, seen_out)
            elif is_media(path) and not is_helper_file(name):
                key = _norm(path)
                if key in seen_out:
                    continue
                rec = self._cached(key, self._dir_sig(path, []) + "|" + self.journal_sig,
                                   lambda p=path: record_from_file(p, journal, batch=depth > 0))
                if rec:
                    found[rec["id"]] = rec

    def scan(self) -> Dict[str, Dict[str, Any]]:
        with self.lock:
            journal = self._journal()
            root = self.root()
            found: Dict[str, Dict[str, Any]] = {}
            seen_out: set = set()
            # run folders first so their outputs are not listed again as loose files
            self._scan_folder(root, journal, found, 0, seen_out)
            for rec in found.values():
                if rec["out"]:
                    seen_out.add(_norm(rec["out"]))
            # journal entries outside the outputs folder (custom batch / override folders)
            for key, row in journal.items():
                if key in seen_out:
                    continue
                out = Path(str(row.get("output")))
                if not out.is_file():
                    continue
                parent = out.parent
                if (parent / "run_context.json").is_file():
                    rec = self._cached(_norm(parent), self._run_sig(parent),
                                       lambda d=parent: record_from_run_dir(d, journal))
                else:
                    rec = record_from_file(out, journal, batch=True)
                if rec:
                    found[rec["id"]] = rec
                    seen_out.add(key)
            # drop cached dirs that vanished
            live = {_norm(r["dir"]) for r in found.values()} | {_norm(r["out"]) for r in found.values() if r["out"]}
            for key in [k for k in self.dir_cache if k not in live]:
                self.dir_cache.pop(key, None)
            self.records = found
            return found

    def signature(self) -> str:
        """Cheap change detector: the outputs folder, its newest entries and the journal."""
        root = self.root()
        parts = [_stat_key(JOURNAL_PATH)]
        try:
            entries = sorted(os.scandir(root), key=lambda e: e.stat().st_mtime_ns, reverse=True)[:6]
            parts.append(str(root.stat().st_mtime_ns))
            for e in entries:
                p = Path(e.path)
                parts.append(f"{e.name}:{e.stat().st_mtime_ns}")
                for name in ("run_context.json", "chain.json"):
                    try:
                        parts.append(str((p / name).stat().st_mtime_ns))
                    except OSError:
                        pass
            chains = root / "chains"
            if chains.is_dir():
                newest = sorted(os.scandir(chains), key=lambda e: e.stat().st_mtime_ns, reverse=True)[:2]
                for e in newest:
                    parts.append(f"c{e.name}:{e.stat().st_mtime_ns}")
                    try:
                        parts.append(str((Path(e.path) / "chain.json").stat().st_mtime_ns))
                    except OSError:
                        pass
        except OSError:
            parts.append("missing")
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]

    # ---- probes -----------------------------------------------------------------------------------
    def _probe(self, path_text: str) -> Optional[Dict[str, Any]]:
        if not path_text:
            return None
        path = Path(path_text)
        key = _stat_key(path)
        if not key:
            return None
        hit = self.probes.get(key)
        if hit is not None:
            return hit
        if key not in self.pending:
            self.pending[key] = self.pool.submit(self._probe_job, key, path)
        return None

    def _probe_job(self, key: str, path: Path) -> None:
        try:
            info = probe_media(path)
            info["size"] = path.stat().st_size
            self.probes.put(key, info)
        except Exception:
            self.probes.put(key, {})
        finally:
            self.pending.pop(key, None)

    def _wait_pending(self, timeout: float) -> None:
        deadline = time.time() + timeout
        for fut in list(self.pending.values()):
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                fut.result(timeout=remaining)
            except Exception:
                pass

    # ---- client payload ---------------------------------------------------------------------------
    def _client(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        out = rec["out"]
        info = self._probe(out) or {}
        inp = rec["inp"]
        in_exists = bool(inp) and Path(inp).is_file()
        in_info = (self._probe(inp) or {}) if in_exists else {}
        label, icon, color, tab = ENGINES.get(rec["engine"], ENGINES["other"])
        thumb = None
        if out:
            tp = _thumb_path(Path(out))
            if tp is not None and tp.is_file():
                thumb = file_url(tp)
        name = Path(out).name if out else (Path(rec["dir"]).name if rec["dir"] else "")
        scale = None
        if info.get("w") and in_info.get("w"):
            scale = round(info["w"] / max(1, in_info["w"]), 2)
        search = " ".join(str(x) for x in (name, Path(inp).name if inp else "", label, rec["model"], rec["chain_name"],
                                           rec["status"], Path(rec["dir"]).name if rec["dir"] else "",
                                           (rec["args"] or {}).get("prompt") or "") if x).lower()
        return {
            "id": rec["id"], "kind": rec["kind"], "engine": rec["engine"], "label": label, "icon": icon, "color": color,
            "tab": tab, "name": name, "out": out, "url": file_url(out) if out else "", "thumb": thumb,
            "inp": inp, "in_exists": in_exists, "in_url": file_url(inp) if in_exists else "",
            "in_kind": kind_of(inp) if inp else "none", "in_w": in_info.get("w"), "in_h": in_info.get("h"),
            "dir": rec["dir"], "ts": rec["ts"], "elapsed": rec["elapsed"], "status": rec["status"],
            "status_text": rec["status_text"][:240], "model": rec["model"], "has_args": bool(rec["args"]),
            "w": info.get("w"), "h": info.get("h"), "dur": info.get("dur"), "fps": info.get("fps"),
            "frames": info.get("frames"), "codec": info.get("codec"), "audio": bool(info.get("audio")),
            "size": info.get("size"), "probed": bool(info) or not out,
            "playable": (browser_playable(info, Path(out)) if (out and rec["kind"] == "video" and info) else True),
            "scale": scale, "cmp_url": file_url(rec["cmp"]) if rec["cmp"] else "", "batch": rec["batch"],
            "chain": rec["chain_name"], "steps": len(rec["steps"] or []), "chain_run": rec["src"] == "chain", "fav": _norm(out) in self.favorites if out
            else False, "search": search,
        }

    def index(self, force: bool = False) -> Dict[str, Any]:
        if force:
            with self.lock:
                self.dir_cache.clear()
        records = self.scan()
        for rec in records.values():
            self._probe(rec["out"])
            if rec["inp"] and Path(rec["inp"]).is_file():
                self._probe(rec["inp"])
        self._wait_pending(1.8)
        items = [self._client(r) for r in records.values()]
        items.sort(key=lambda r: r["ts"] or 0, reverse=True)
        self.probes.save()
        total_size = sum(int(i.get("size") or 0) for i in items)
        return {"items": items, "root": str(self.root()), "sig": self.signature(), "pending": len(self.pending),
                "total_size": total_size,
                "engines": {k: {"label": v[0], "icon": v[1], "color": v[2], "tab": v[3]} for k, v in ENGINES.items()},
                "send_tabs": SEND_TABS, "video_only": sorted(VIDEO_ONLY_TABS)}

    def get(self, rid: str) -> Optional[Dict[str, Any]]:
        rec = self.records.get(str(rid or ""))
        if rec is None:
            self.scan()
            rec = self.records.get(str(rid or ""))
        return rec

    def thumbs(self, ids: List[str]) -> Dict[str, str]:
        jobs = {}
        for rid in list(ids or [])[:32]:
            rec = self.records.get(str(rid))
            if not rec or not rec["out"]:
                continue
            path = Path(rec["out"])
            info = self.probes.get(_stat_key(path)) or {}
            jobs[rid] = self.pool.submit(make_thumbnail, path, float(info.get("dur") or 0.0))
        out: Dict[str, str] = {}
        for rid, fut in jobs.items():
            try:
                tp = fut.result(timeout=60)
            except Exception:
                tp = None
            out[rid] = file_url(tp) if tp else ""
        return out

    def details(self, rid: str) -> Dict[str, Any]:
        rec = self.get(rid)
        if not rec:
            return {"error": "This generation is no longer in the outputs folder."}
        args = rec["args"] or {}
        groups: Dict[str, List[List[str]]] = {}
        clean: Dict[str, Any] = {}
        shown = dict(args)
        # run-internal values (the run's own folder) are not settings the user chose
        if "_user_output_override_raw" in args:
            shown["output_override"] = args.get("_user_output_override_raw") or ""
        elif rec["dir"] and _inside(args.get("output_override"), rec["dir"]):
            shown["output_override"] = ""
        for key in ("global_output_dir", "resume_run_dir"):
            shown.pop(key, None)
        for key in sorted(shown):
            value = shown[key]
            if key.startswith("_") or isinstance(value, (dict, list)):
                continue
            clean[key] = value
            groups.setdefault(param_group(key), []).append([key, _fmt_value(value)])
        order = ["Input & Output", "Model", "Size & Scale", "Quality", "Frames & Timing", "Memory & Speed",
                 "Encoding", "Other"]
        steps = []
        for s in rec["steps"] or []:
            info = (self._probe(s["out"]) or {}) if s["out"] else {}
            steps.append({**s, "url": file_url(s["out"]) if s["out"] else "", "kind": kind_of(s["out"]),
                          "w": info.get("w"), "h": info.get("h")})
        extras = [{"name": Path(p).name, "url": file_url(p), "kind": kind_of(p), "path": p} for p in rec["extras"]]
        return {
            "id": rec["id"], "groups": [{"name": g, "items": groups[g]} for g in order if g in groups],
            "args": clean, "steps": steps, "extras": extras, "target": rec["target"],
            "status_text": rec["status_text"], "tab": ENGINES.get(rec["engine"], ENGINES["other"])[3],
        }

    def set_favorite(self, ids: List[str], value: bool) -> int:
        changed = 0
        with self.lock:
            for rid in ids or []:
                rec = self.records.get(str(rid))
                if not rec or not rec["out"]:
                    continue
                key = _norm(rec["out"])
                if value and key not in self.favorites:
                    self.favorites.add(key)
                    changed += 1
                elif not value and key in self.favorites:
                    self.favorites.discard(key)
                    changed += 1
            self._save_favorites()
        return changed

    def playable_url(self, rid: str, which: str = "out") -> Dict[str, Any]:
        rec = self.get(rid)
        if not rec:
            return {"url": ""}
        path = rec["inp"] if which == "inp" else rec["out"]
        if not path or not Path(path).is_file():
            return {"url": ""}
        info = probe_media(Path(path)) if kind_of(path) == "video" else {}
        if kind_of(path) != "video" or browser_playable(info, Path(path)):
            return {"url": file_url(path)}
        preview = browser_preview(Path(path))
        return {"url": file_url(preview) if preview else file_url(path), "converted": bool(preview)}

    # ---- file actions -------------------------------------------------------------------------------
    def _owned_target(self, rec: Dict[str, Any]) -> Tuple[Optional[Path], List[Path]]:
        """What Delete removes: the run / chain folder when it lies inside the outputs folder, otherwise the
        output file and its comparison video. Returns (folder or None, files)."""
        root = self.root().resolve()
        if rec["src"] in ("run", "chain") and rec["dir"]:
            folder = Path(rec["dir"]).resolve()
            try:
                inside = folder != root and folder.is_relative_to(root)
            except (OSError, ValueError):
                inside = False
            if inside and not os.path.islink(folder):
                return folder, []
        files = [Path(rec["out"])] if rec["out"] else []
        if rec["cmp"]:
            files.append(Path(rec["cmp"]))
        return None, [f for f in files if f.is_file() and not os.path.islink(f)]

    def delete(self, ids: List[str]) -> Dict[str, Any]:
        removed, failed = [], []
        trash_root = None
        for rid in ids or []:
            rec = self.records.get(str(rid))
            if not rec:
                continue
            folder, files = self._owned_target(rec)
            targets = [folder] if folder else files
            if not targets:
                failed.append(rec["id"])
                continue
            ok = True
            for target in targets:
                if not _to_recycle_bin(target):
                    trash_root = trash_root or (DATA_DIR / "trash" / time.strftime("%Y%m%d_%H%M%S"))
                    try:
                        trash_root.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(target), str(trash_root / target.name))
                    except OSError:
                        ok = False
            if ok:
                removed.append(rec["id"])
                with self.lock:
                    self.records.pop(rec["id"], None)
                    self.favorites.discard(_norm(rec["out"]))
            else:
                failed.append(rec["id"])
        if removed:
            self._save_favorites()
        where = "the Recycle Bin" if platform.system() == "Windows" and not trash_root else \
            (str(trash_root) if trash_root else "the trash")
        return {"removed": removed, "failed": failed, "where": where}

    def open_location(self, rid: str, what: str) -> Dict[str, Any]:
        if rid == "__root__":
            target = self.root()
        else:
            rec = self.get(rid)
            if not rec:
                return {"ok": False, "msg": "Not found."}
            path = rec["out"] if what == "file" else (rec["dir"] or str(Path(rec["out"]).parent))
            target = Path(path)
        if not target.exists():
            return {"ok": False, "msg": f"Not found: {target}"}
        try:
            if platform.system() == "Windows":
                if what == "select" and target.is_file():
                    subprocess.Popen(["explorer", "/select,", str(target)], creationflags=_NO_WINDOW)
                else:
                    os.startfile(str(target))  # type: ignore[attr-defined]
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", "-R", str(target)] if what == "select" else ["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target if target.is_dir() or what == "file" else target.parent)])
            return {"ok": True, "msg": f"Opened on the computer running the app: {target}"}
        except Exception as exc:
            return {"ok": False, "msg": f"Could not open: {exc}"}

    def zip_selection(self, ids: List[str], temp_dir: Path) -> Dict[str, Any]:
        files = []
        for rid in ids or []:
            rec = self.records.get(str(rid))
            if rec and rec["out"] and Path(rec["out"]).is_file():
                files.append(Path(rec["out"]))
        if not files:
            return {"url": "", "msg": "Nothing to download."}
        out_dir = Path(temp_dir) / "history_downloads"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"generations_{time.strftime('%Y%m%d_%H%M%S')}_{len(files)}.zip"
        used: set = set()
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for f in files:
                name = f.name
                n = 2
                while name.lower() in used:
                    name = f"{f.stem}_{n}{f.suffix}"
                    n += 1
                used.add(name.lower())
                zf.write(f, arcname=name)
        return {"url": file_url(target), "name": target.name, "count": len(files), "size": target.stat().st_size}


def _to_recycle_bin(path: Path) -> bool:
    """Windows Recycle Bin (recoverable). False elsewhere or on failure (the caller moves it to the app trash)."""
    if platform.system() != "Windows":
        return False
    try:
        from ctypes import wintypes

        class SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                        ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_uint16), ("fAnyOperationsAborted", wintypes.BOOL),
                        ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]

        FO_DELETE, FOF_SILENT, FOF_NOCONFIRMATION, FOF_ALLOWUNDO, FOF_NOERRORUI = 3, 0x4, 0x10, 0x40, 0x400
        op = SHFILEOPSTRUCTW(None, FO_DELETE, str(path.resolve()) + "\0", None,
                             FOF_SILENT | FOF_NOCONFIRMATION | FOF_ALLOWUNDO | FOF_NOERRORUI, False, None, None)
        result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))  # type: ignore[attr-defined]
        return result == 0 and not op.fAnyOperationsAborted and not path.exists()
    except Exception:
        return False


# --------------------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------------------

_GROUP_RULES = [
    ("Input & Output", ("input_path", "output_override", "batch", "output_format", "resume", "save_metadata",
                        "auto_transfer", "keep_only", "overwrite", "skip")),
    ("Model", ("model", "dit", "vae", "wan", "precision", "attention", "quant", "int8", "lora", "encoder", "profile",
               "workflow", "version", "mode", "backend", "compile")),
    ("Size & Scale", ("scale", "resolution", "upscale", "max_target", "downscale", "width", "height", "tile", "size",
                      "crop", "pad", "max_side", "aspect")),
    ("Quality", ("step", "seed", "prompt", "cfg", "sigma", "strength", "noise", "denoise", "sharp", "intensity",
                 "detail", "color", "curve", "guidance", "sampler", "scheduler", "face", "quality_preset")),
    ("Frames & Timing", ("fps", "frame", "chunk", "window", "overlap", "scene", "temporal", "duration", "start", "end")),
    ("Memory & Speed", ("vram", "block", "swap", "offload", "cpu", "gpu", "device", "batch_size", "token", "cache",
                        "sparse", "dense", "auto_tune", "ffn", "reserve", "memory")),
    ("Encoding", ("codec", "crf", "quality", "pixel", "audio", "video_", "h265", "av1", "two_pass", "metadata",
                  "image_output", "png", "bitrate")),
]


def param_group(key: str) -> str:
    low = key.lower()
    if "face" in low:  # face_restore_after_upscale is about quality, not size
        return "Quality"
    for name, needles in _GROUP_RULES:
        if any(n in low for n in needles):
            return name
    return "Other"


def _fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "On" if value else "Off"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def settings_for_reload(rec: Dict[str, Any], current: Dict[str, Any], order: List[str]) -> Tuple[Dict[str, Any], int]:
    """The tab's settings with every saved value of the run applied; run-internal paths are reset."""
    args = rec.get("args") or {}
    merged = dict(current or {})
    applied = 0
    for key in order:
        if key in args:
            merged[key] = args[key]
            applied += 1
    if "output_override" in order:
        merged["output_override"] = str(args.get("_user_output_override_raw") or "") \
            if "_user_output_override_raw" in args else ("" if _inside(args.get("output_override"), rec.get("dir"))
                                                         else str(args.get("output_override") or ""))
    if "resume_run_dir" in order:
        merged["resume_run_dir"] = ""
    original = str(args.get("_original_input_path_before_preprocess") or args.get("input_path") or rec.get("inp") or "")
    if "input_path" in order and original:
        if Path(original).is_file() or (Path(original).is_dir() and not merged.get("batch_enable")):
            merged["input_path"] = original
        elif rec.get("inp") and Path(rec["inp"]).exists():
            merged["input_path"] = rec["inp"]
    # One item of a batch: reload that file, not the whole folder again.
    if merged.get("batch_enable") and "batch_enable" in order and Path(str(merged.get("input_path") or "")).is_file():
        merged["batch_enable"] = False
    return merged, applied


def _inside(path: Any, folder: Any) -> bool:
    if not path or not folder:
        return False
    try:
        return Path(str(path)).resolve().is_relative_to(Path(str(folder)).resolve())
    except (OSError, ValueError):
        return False
