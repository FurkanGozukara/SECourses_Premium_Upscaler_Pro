"""
LTX 2.5 Pixel Spatial Upscaler 2x — standalone CLI (no ComfyUI backend).

Launched by shared/ltx25_runner.py inside the app venv. All progress lines go
to stdout unbuffered so the service can mirror them into the Gradio log.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
LTX25_DIR = APP_DIR / "LTX25"
sys.path.insert(0, str(LTX25_DIR))

if sys.platform == "win32":
    try:
        if sys.stdout.encoding != "utf-8":
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True)
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", write_through=True)
    except Exception:
        pass


def emit(text: str) -> None:
    print(text, flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LTX 2.5 Pixel Spatial Upscaler 2x")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--transformer", required=True)
    p.add_argument("--text_encoder", required=True)
    p.add_argument("--video_vae", required=True)
    p.add_argument("--audio_vae", default="")
    p.add_argument("--ic_lora", required=True)

    p.add_argument("--positive_prompt", default="")
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--sampler", default="euler_ancestral")
    p.add_argument("--schedule", default="distilled", choices=["distilled", "dev"])
    p.add_argument("--ic_lora_strength", type=float, default=1.0)
    p.add_argument("--guide_strength", type=float, default=1.0)

    p.add_argument("--token_budget_mode", default="auto", choices=["auto", "manual"])
    p.add_argument("--max_latent_tokens", type=int, default=18000)
    p.add_argument("--max_chunk_frames", type=int, default=121)
    p.add_argument("--overlap_frames", type=int, default=1)

    p.add_argument("--encode_tile_size", type=int, default=512)
    p.add_argument("--encode_tile_overlap", type=int, default=64)
    p.add_argument("--decode_tile_size", type=int, default=512)
    p.add_argument("--decode_tile_overlap", type=int, default=64)
    p.add_argument("--decode_temporal_size", type=int, default=128)
    p.add_argument("--decode_temporal_overlap", type=int, default=32)

    p.add_argument("--pre_resize_longer_edge", type=int, default=0)
    p.add_argument("--fps_override", type=float, default=0.0)
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, default=-1)

    p.add_argument("--codec", default="libx264")
    p.add_argument("--crf", type=int, default=15)
    p.add_argument("--preset", default="medium")
    p.add_argument("--pixel_format", default="yuv420p")
    p.add_argument("--no_audio", action="store_true")

    p.add_argument("--reserve_vram_gb", type=float, default=1.0)
    p.add_argument("--bytes_per_token", type=float, default=1_400_000.0)
    p.add_argument("--attention", default="auto", choices=["auto", "sage", "flash", "sdpa"])
    p.add_argument("--vram_mode", default="auto",
                   choices=["auto", "gpu_only", "highvram", "lowvram", "novram"])
    p.add_argument("--plan_only", action="store_true", help="Print the chunk plan and exit")
    return p


def main() -> int:
    args = build_parser().parse_args()

    os.environ.setdefault("SECOURSES_LTX25_ATTENTION", args.attention)
    os.environ.setdefault("SECOURSES_LTX25_VRAM_MODE", args.vram_mode)
    os.environ.setdefault("SECOURSES_LTX25_RESERVE_VRAM_GB", str(args.reserve_vram_gb))

    from comfy.cli_args import configure_from_env

    configure_from_env()

    # Heavy imports after args are configured.
    from ltx25_engine import planner as planner_mod
    from ltx25_engine.pipeline import Ltx25Pipeline, Ltx25Progress, Ltx25Settings
    from ltx25_engine.video_io import read_video_frames

    if args.plan_only:
        frames_u8, fps, _ = read_video_frames(args.input)
        total = frames_u8.shape[0]
        h, w = frames_u8.shape[1], frames_u8.shape[2]
        budget = args.max_latent_tokens
        if args.token_budget_mode == "auto":
            import comfy.model_management as mm

            budget, reason = planner_mod.auto_token_budget(
                total, w, h,
                free_vram_bytes=int(mm.get_free_memory(mm.get_torch_device())),
                weight_resident_bytes=0,
                reserve_gb=args.reserve_vram_gb,
                bytes_per_token=args.bytes_per_token,
            )
            emit(f"[Plan] auto budget {budget} ({reason})")
        plan = planner_mod.plan_chunks(total, w, h, budget, args.max_chunk_frames, args.overlap_frames)
        for line in plan.summary(args.steps).splitlines():
            emit(f"[Plan] {line}")
        return 0

    settings = Ltx25Settings(
        input_path=args.input,
        output_path=args.output,
        transformer_path=args.transformer,
        text_encoder_path=args.text_encoder,
        video_vae_path=args.video_vae,
        audio_vae_path=args.audio_vae,
        ic_lora_path=args.ic_lora,
        positive_prompt=args.positive_prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        steps=args.steps,
        cfg=args.cfg,
        sampler=args.sampler,
        schedule=args.schedule,
        ic_lora_strength=args.ic_lora_strength,
        guide_strength=args.guide_strength,
        token_budget_mode=args.token_budget_mode,
        max_latent_tokens=args.max_latent_tokens,
        max_chunk_frames=args.max_chunk_frames,
        overlap_frames=args.overlap_frames,
        encode_tile_size=args.encode_tile_size,
        encode_tile_overlap=args.encode_tile_overlap,
        decode_tile_size=args.decode_tile_size,
        decode_tile_overlap=args.decode_tile_overlap,
        decode_temporal_size=args.decode_temporal_size,
        decode_temporal_overlap=args.decode_temporal_overlap,
        pre_resize_longer_edge=args.pre_resize_longer_edge,
        fps_override=args.fps_override,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset,
        pixel_format=args.pixel_format,
        keep_audio=not args.no_audio,
        reserve_vram_gb=args.reserve_vram_gb,
        bytes_per_token=args.bytes_per_token,
    )

    emit(f"[LTX25] Starting pixel spatial 2x upscale | {Path(args.input).name}")
    emit(f"[LTX25] Transformer: {Path(args.transformer).name}")
    emit(f"[LTX25] Schedule: {args.schedule} | steps {args.steps} | cfg {args.cfg} | sampler {args.sampler} | seed {args.seed}")
    started = time.time()
    pipeline = Ltx25Pipeline(settings, Ltx25Progress(on_line=emit))
    try:
        output = pipeline.run()
    except KeyboardInterrupt:
        emit("[LTX25] Cancelled by user")
        return 130
    except Exception as exc:  # surface a clean error for the service log
        import traceback

        traceback.print_exc()
        emit(f"[LTX25] ERROR: {exc}")
        return 1
    emit(f"[LTX25] OUTPUT_PATH: {output}")
    emit(f"[LTX25] Total time: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
