"""Opt-in GPU smoke campaign for the real model runners and universal chunker.

This is intentionally not named ``test_*.py``: the large model downloads and GPU work
must never run as part of the ordinary unit-test suite. Example::

    python tools/actual_model_smoke.py --models gan,rtx,rife,ltx25
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.chunking import chunk_and_process  # noqa: E402
from shared.flashvsr_runner import run_flashvsr  # noqa: E402
from shared.ltx25_runner import run_ltx25  # noqa: E402
from shared.rtx_superres_runner import run_rtx_superres  # noqa: E402
from shared.runner import Runner  # noqa: E402
from shared.sparkvsr_runner import run_sparkvsr  # noqa: E402


class ModelAdapter:
    """Expose standalone runners through the interface used by ``chunk_and_process``."""

    def __init__(self, runner: Runner):
        self._runner = runner

    def reset_cancel_state(self) -> None:
        self._runner.reset_cancel_state()

    def is_canceled(self) -> bool:
        return self._runner.is_canceled()

    def run_seedvr2(self, settings, on_progress=None, preview_only=False):
        return self._runner.run_seedvr2(settings, on_progress=on_progress, preview_only=preview_only)

    def run_gan(self, settings, on_progress=None):
        return self._runner.run_gan(settings, on_progress=on_progress)

    def run_rife(self, settings, on_progress=None):
        return self._runner.run_rife(settings, on_progress=on_progress)

    def run_flashvsr(self, settings, on_progress=None):
        return run_flashvsr(settings, REPO_ROOT, on_progress=on_progress)

    def run_sparkvsr(self, settings, on_progress=None):
        return run_sparkvsr(settings, REPO_ROOT, on_progress=on_progress)

    def run_ltx25(self, settings, on_progress=None):
        return run_ltx25(settings, REPO_ROOT, on_progress=on_progress)

    def run_rtx_superres(self, settings, on_progress=None):
        return run_rtx_superres(settings, REPO_ROOT, on_progress=on_progress)


def _run(cmd: list[str], *, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _make_fixture(path: Path, *, frames: int, fps: Fraction) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = float(Fraction(frames, 1) / fps)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=128x128:rate={fps.numerator}/{fps.denominator}",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=733:sample_rate=48000",
        "-frames:v",
        str(frames),
        "-t:a",
        f"{duration:.12f}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "15",
        "-bf",
        "3",
        "-g",
        "120",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        str(path),
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or "fixture generation failed")


def _probe(path: Path) -> dict[str, Any]:
    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,start_time,duration,nb_frames,nb_read_frames",
            "-of",
            "json",
            str(path),
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or f"ffprobe failed: {path}")
    return json.loads(proc.stdout or "{}")


def _rate(raw: Any) -> float:
    try:
        return float(Fraction(str(raw)))
    except Exception:
        return 0.0


def _validate_output(
    path: Path,
    model: str,
    *,
    source_frames: int,
    source_fps: float,
    rife_target_fps: float = 0.0,
) -> dict[str, Any]:
    payload = _probe(path)
    streams = payload.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = [item for item in streams if item.get("codec_type") == "audio"]
    if not video:
        raise AssertionError(f"{model}: no video stream")
    frames_raw = str(video.get("nb_read_frames") or video.get("nb_frames") or "")
    if not frames_raw.isdigit():
        raise AssertionError(f"{model}: decoded frame count is unavailable")
    frames = int(frames_raw)
    fps = _rate(video.get("r_frame_rate")) or _rate(video.get("avg_frame_rate"))
    start = float(video.get("start_time") or 0.0)
    duration = float(video.get("duration") or 0.0)
    if abs(start) > 0.001:
        raise AssertionError(f"{model}: video starts at {start:.6f}s")
    if not audio:
        raise AssertionError(f"{model}: source audio was not restored")

    if model == "rife":
        expected_fps = float(rife_target_fps) if rife_target_fps > 0 else (source_fps * 2.0)
        expected_frames = int(round((source_frames - 1) * expected_fps / source_fps)) + 1
        if frames != expected_frames:
            raise AssertionError(f"{model}: expected {expected_frames} interpolated frames, got {frames}")
        if abs(fps - expected_fps) > 0.02:
            raise AssertionError(f"{model}: expected {expected_fps:.6f} fps, got {fps:.6f}")
    else:
        if frames != source_frames:
            raise AssertionError(f"{model}: expected {source_frames} frames, got {frames}")
        if abs(fps - source_fps) > 0.02:
            raise AssertionError(f"{model}: expected {source_fps:.6f} fps, got {fps:.6f}")
    expected_duration = frames / fps
    if abs(duration - expected_duration) > max(0.10, 1.25 / fps):
        raise AssertionError(
            f"{model}: duration {duration:.6f}s does not match {frames} frames at {fps:.6f} fps"
        )
    return {
        "path": str(path),
        "frames": frames,
        "fps": fps,
        "duration": duration,
        "start": start,
        "video_codec": video.get("codec_name"),
        "audio_streams": len(audio),
        "size": [video.get("width"), video.get("height")],
    }


def _common_settings(input_path: Path, model_dir: Path) -> dict[str, Any]:
    return {
        "input_path": str(input_path),
        "_effective_input_path": str(input_path),
        "_original_filename": input_path.name,
        "_run_dir": str(model_dir),
        "global_output_dir": str(model_dir),
        "output_format": "mp4",
        "frame_accurate_split": True,
        "audio_codec": "copy",
        "audio_bitrate": "",
        "video_codec": "h264",
        "codec": "libx264",
        "video_quality": 15,
        "crf": 15,
        "video_preset": "ultrafast",
        "pixel_format": "yuv420p",
        "two_pass_encoding": False,
        "fps": 0.0,
        "fps_override": 0.0,
        "device": "0",
        "cuda_device": "0",
        "save_metadata": False,
    }


def _settings_for(model: str, common: dict[str, Any]) -> dict[str, Any]:
    settings = dict(common)
    if model == "gan":
        settings.update(model="4x-UltraSharpV2.safetensors", batch_size=4)
    elif model == "rtx":
        settings.update(
            quality_preset="SUPER_RES",
            upscale_factor=2.0,
            max_resolution=0,
            pre_downscale_then_upscale=False,
            non_blocking_inference=True,
        )
    elif model == "rife":
        settings.update(
            model="4.26",
            fps_multiplier=2,
            fp16_mode=True,
            scale=1.0,
            no_audio=False,
            output_quality=15,
        )
    elif model == "ltx25":
        settings.update(
            model_name="Distilled INT8 ConvRot",
            text_encoder="Gemma 4 12B INT8 ConvRot",
            video_vae="Video VAE Conv",
            positive_prompt="Faithfully upscale the source video without changing its content or timing.",
            negative_prompt="changed timing, changed content, artifacts",
            seed=12,
            randomize_seed=False,
            steps=8,
            cfg=1.0,
            sampler="euler_ancestral",
            schedule="distilled",
            ic_lora_strength=1.0,
            guide_strength=1.0,
            token_budget_mode="auto",
            max_latent_tokens=4096,
            max_chunk_frames=17,
            overlap_frames=1,
            encode_tile_size=128,
            encode_tile_overlap=32,
            decode_tile_size=128,
            decode_tile_overlap=32,
            decode_temporal_size=16,
            decode_temporal_overlap=4,
            reserve_vram_gb=1.0,
            attention_backend="auto",
            vram_mode="auto",
        )
    elif model == "seedvr2":
        from shared.services.seedvr2_service import seedvr2_defaults

        defaults = seedvr2_defaults("seedvr2_ema_3b-Q4_K_M.gguf", REPO_ROOT)
        defaults.update(settings)
        defaults.update(
            dit_model="seedvr2_ema_3b-Q4_K_M.gguf",
            batch_size=5,
            resolution=256,
            max_resolution=384,
            upscale_factor=2.0,
            pre_downscale_then_upscale=False,
            int8_convrot=False,
            compile_dit=False,
            compile_vae=False,
        )
        settings = defaults
    elif model == "flashvsr":
        settings.update(
            model="FlashVSR-v1.1",
            mode="tiny-long",
            vae_model="Wan2.2",
            precision="bf16",
            scale=2,
            frame_chunk_size=0,
            attention_mode="sdpa",
            tiled_vae=True,
            tiled_dit=True,
            tile_size=64,
            tile_overlap=16,
        )
    elif model == "sparkvsr":
        settings.update(
            model_name="SparkVSR-bf16",
            precision="bfloat16",
            scale=2,
            chunk_len=0,
            overlap_t=0,
            cpu_offload=True,
            tile_height=0,
            tile_width=0,
        )
    else:
        raise ValueError(f"Unsupported model route: {model}")
    return settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="gan,rtx,rife,ltx25")
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--fps", default="17/1")
    parser.add_argument("--rife-target-fps", type=float, default=0.0)
    parser.add_argument(
        "--two-chunk-models",
        default="gan,rtx,rife",
        help="Comma-separated routes that must cross a real app-level chunk boundary.",
    )
    parser.add_argument("--work-dir", default=str(REPO_ROOT / "temp" / "actual_model_smoke"))
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    fps = Fraction(str(args.fps))
    frames = max(3, int(args.frames))
    work_root = Path(args.work_dir).resolve()
    if work_root.exists() and not args.keep:
        allowed_root = (REPO_ROOT / "temp").resolve()
        try:
            relative_target = work_root.relative_to(allowed_root)
        except ValueError as exc:
            raise RuntimeError(
                f"Refusing to clear a smoke work directory outside {allowed_root}: {work_root}"
            ) from exc
        if not relative_target.parts:
            raise RuntimeError(f"Refusing to clear the shared temp root: {work_root}")
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    fixture = work_root / f"fixture_{frames}f_{fps.numerator}_{fps.denominator}.mp4"
    _make_fixture(fixture, frames=frames, fps=fps)

    runner = Runner(REPO_ROOT, REPO_ROOT / "temp", work_root, telemetry_enabled=False)
    adapter = ModelAdapter(runner)
    requested = [part.strip().lower() for part in str(args.models).split(",") if part.strip()]
    two_chunk_models = {
        part.strip().lower()
        for part in str(args.two_chunk_models).split(",")
        if part.strip()
    }
    report: dict[str, Any] = {
        "fixture": _validate_output(fixture, "fixture", source_frames=frames, source_fps=float(fps)),
        "models": {},
    }

    for model in requested:
        model_dir = work_root / model
        model_dir.mkdir(parents=True, exist_ok=True)
        settings = _settings_for(model, _common_settings(fixture, model_dir))
        if model == "rife" and float(args.rife_target_fps or 0.0) > 0:
            # Runner._build_rife_cmd uses the same fps_override key as the GUI.
            settings["fps_override"] = float(args.rife_target_fps)
        # RIFE is included by default because q*(N-1)+1 interpolation needs a shared source
        # frame at every app-level boundary. Release validation can opt every route into the
        # same two-chunk exercise without making the ordinary smoke command unexpectedly slow.
        chunk_seconds = (9.0 / float(fps)) if model in two_chunk_models and frames > 9 else 60.0
        print(f"\n=== actual {model} route ({frames} frames, chunk_seconds={chunk_seconds:.6f}) ===", flush=True)
        rc, log_text, output_path, chunk_count = chunk_and_process(
            adapter,
            settings,
            scene_threshold=1_000_000.0,
            min_scene_len=0.1,
            work_dir=model_dir,
            on_progress=lambda msg: print(str(msg), end="", flush=True),
            chunk_seconds=chunk_seconds,
            chunk_overlap=0.0,
            per_chunk_cleanup=False,
            allow_partial=False,
            global_output_dir=str(model_dir),
            model_type=model,
        )
        if rc != 0:
            raise RuntimeError(f"{model} failed (chunks={chunk_count}): {log_text}\noutput={output_path}")
        output = Path(output_path)
        report["models"][model] = {
            "chunk_count": chunk_count,
            **_validate_output(
                output,
                model,
                source_frames=frames,
                source_fps=float(fps),
                rife_target_fps=float(args.rife_target_fps or 0.0),
            ),
        }
        print(json.dumps(report["models"][model], indent=2), flush=True)

    report_path = work_root / "actual_model_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nActual model smoke report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
