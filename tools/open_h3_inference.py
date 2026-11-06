"""
Open-H3 Regenerate (MiniMax H3, NVIDIA Open-H3-Regenerate-2K recipe) - standalone CLI (no ComfyUI backend).

Launched by shared/open_h3_runner.py inside the app venv. All progress lines go
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
OPENH3_DIR = APP_DIR / "OpenH3"
sys.path.insert(0, str(OPENH3_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(1, str(APP_DIR))  # shared.video_io

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
    p = argparse.ArgumentParser(description="Open-H3 Regenerate (MiniMax H3 3-step regenerate)")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dit", required=True)
    p.add_argument("--text_encoder", required=True)
    p.add_argument("--vae", required=True)
    p.add_argument("--decode_vae", default="")
    p.add_argument("--audio_vae", required=True)
    p.add_argument("--upscaler", required=True)
    p.add_argument("--lora", required=True)
    p.add_argument("--curves", default="")
    p.add_argument("--table", default="")
    p.add_argument("--cache_dir", default="")

    p.add_argument("--prompt", default="")
    p.add_argument("--prompt_file", default="", help="UTF-8 file holding the prompt (overrides --prompt)")
    p.add_argument("--seed", type=int, default=1766580932)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--start_sigma", type=float, default=0.0)
    p.add_argument("--lora_strength", type=float, default=1.0)
    p.add_argument("--curve_strength", type=float, default=1.0)

    p.add_argument("--target_width", type=int, default=0)
    p.add_argument("--target_height", type=int, default=0)
    p.add_argument("--draft_size", default="768p", choices=["768p", "576p", "native"])
    p.add_argument("--max_window_frames", type=int, default=121)
    p.add_argument("--overlap_frames", type=int, default=8)
    p.add_argument("--fps_mode", default="source", choices=["source", "24"])
    p.add_argument("--audio_mode", default="silence", choices=["silence", "source"])
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, default=-1)

    p.add_argument("--no_sparse", action="store_true")
    p.add_argument("--sparse_tau", type=float, default=1.2)
    p.add_argument("--sparse_end_percent", type=float, default=0.9)
    p.add_argument("--sparse_dense_steps", type=int, default=-1,
                   help="-1: sparse inside the start/end percent window; N >= 0: first N refine steps dense, the rest sparse")
    p.add_argument("--second_pass_sigma", type=float, default=0.0,
                   help="> 0: refine each window a second time at the same size from this noise level "
                        "(experimental: at 4K a 0.7 + 0.6 ladder added grain and changed objects, so it is off)")
    p.add_argument("--dense_blocks", default="0,1,49")
    p.add_argument("--sparse_min_tokens", type=int, default=12288)
    p.add_argument("--sparse_extra_tokens", type=int, default=256)
    p.add_argument("--sink_conditioning", default="exact_kv_and_rows",
                   choices=["exact_kv_and_rows", "exact_kv", "off"])
    p.add_argument("--ffn_chunk_tokens", type=int, default=32768)

    p.add_argument("--token_budget_mode", default="auto", choices=["auto", "manual"])
    p.add_argument("--max_tokens", type=int, default=160000)
    p.add_argument("--save_vram_gb", "--reserve_vram_gb", dest="reserve_vram_gb", type=float, default=2.0,
                   help="VRAM always left free; the engine caps itself below it and streams instead of spilling")
    p.add_argument("--max_gpu_blocks", type=int, default=50, help="Most DiT blocks kept on the GPU (block swap)")
    p.add_argument("--bytes_per_token", type=float, default=145000.0)
    p.add_argument("--attention", default="auto", choices=["auto", "sage", "flash", "sdpa"])

    p.add_argument("--codec", default="libx264")
    p.add_argument("--crf", type=int, default=15)
    p.add_argument("--preset", default="medium")
    p.add_argument("--pixel_format", default="yuv420p")
    p.add_argument("--no_audio", action="store_true")
    p.add_argument("--keep_latents", action="store_true")
    p.add_argument("--plan_only", action="store_true", help="Print the geometry / window plan and exit")
    p.add_argument("--stages", default="",
                   help="Comma list of text,encode,refine,decode to run (default: all four in this process)")
    p.add_argument("--work_dir", default="", help="Hand-over folder between stages run as separate processes")
    p.add_argument("--workflow", default="regenerate", choices=["regenerate", "jxl"],
                   help="regenerate: Open-H3 Regenerate (NVIDIA recipe); jxl: JonXL H3 Upscale (h3_upscale_1920 LoRA, "
                        "fixed prompt, 20 steps er_sde / beta57, the last frame of a 24-frame clip per picture)")
    p.add_argument("--jxl_lora", default="", help="JonXL LoRA file (h3_upscale_1920.safetensors)")
    p.add_argument("--jxl_max_side", type=int, default=1920, help="JonXL: largest output side (the LoRA is trained at 1920)")
    p.add_argument("--jxl_steps", type=int, default=20)
    p.add_argument("--jxl_geometry", default="keep", choices=["keep", "workflow"],
                   help="keep: exact aspect on a padded canvas; workflow: the ComfyUI graph's crop + stretch (parity tests)")
    p.add_argument("--batch_json", default="",
                   help="JSON list of per-input overrides (input, output, target_width, target_height, draft_size, "
                        "start_frame, end_frame); every input runs with the models loaded once")
    return p


def main() -> int:
    args = build_parser().parse_args()
    os.environ.setdefault("SECOURSES_OPENH3_ATTENTION", args.attention)
    os.environ.setdefault("SECOURSES_OPENH3_RESERVE_VRAM_GB", str(args.reserve_vram_gb))

    from comfy.cli_args import configure_from_env

    configure_from_env()

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    if args.workflow == "jxl":
        return run_jxl_cli(args, prompt)

    from open_h3_engine.pipeline import OpenH3Pipeline, OpenH3Progress, OpenH3Settings

    settings = OpenH3Settings(
        input_path=args.input, output_path=args.output, dit_path=args.dit, text_encoder_path=args.text_encoder,
        vae_path=args.vae, decode_vae_path=args.decode_vae, audio_vae_path=args.audio_vae,
        upscaler_path=args.upscaler, lora_path=args.lora, curve_path=args.curves, table_path=args.table,
        cache_dir=args.cache_dir, prompt=prompt, seed=args.seed, steps=args.steps, start_sigma=args.start_sigma,
        lora_strength=args.lora_strength, curve_strength=args.curve_strength,
        target_width=args.target_width, target_height=args.target_height, draft_size=args.draft_size,
        max_window_frames=args.max_window_frames, overlap_frames=args.overlap_frames, fps_mode=args.fps_mode,
        audio_mode=args.audio_mode, start_frame=args.start_frame, end_frame=args.end_frame,
        sparse_attention=not args.no_sparse, sparse_tau=args.sparse_tau, sparse_end_percent=args.sparse_end_percent,
        sparse_dense_steps=args.sparse_dense_steps, second_pass_sigma=args.second_pass_sigma,
        dense_blocks=args.dense_blocks, sparse_min_tokens=args.sparse_min_tokens,
        sparse_extra_tokens=args.sparse_extra_tokens, sink_conditioning=args.sink_conditioning,
        ffn_chunk_tokens=args.ffn_chunk_tokens, reserve_vram_gb=args.reserve_vram_gb,
        bytes_per_token=args.bytes_per_token, token_budget_mode=args.token_budget_mode, max_tokens=args.max_tokens,
        max_gpu_blocks=args.max_gpu_blocks,
        codec=args.codec, crf=args.crf, preset=args.preset, pixel_format=args.pixel_format,
        keep_audio=not args.no_audio, keep_latents=args.keep_latents,
    )

    from open_h3_engine.pipeline import STAGES, run_many

    stages = tuple(x.strip() for x in str(args.stages or "").split(",") if x.strip()) or STAGES
    unknown = [x for x in stages if x not in STAGES]
    if unknown:
        emit(f"[OpenH3] ERROR: unknown stage(s) {unknown}")
        return 2
    if stages != STAGES and not args.work_dir:
        emit("[OpenH3] ERROR: --work_dir is required when running a subset of the stages")
        return 2
    work_dir = args.work_dir or None
    if stages != STAGES:
        emit(f"[OpenH3] Stage {'+'.join(stages)} (separate process; results handed over in {work_dir})")

    if args.plan_only:
        from open_h3_engine.pipeline import plan_only

        for line in plan_only(settings):
            emit(f"[Plan] {line}")
        return 0

    if args.batch_json:
        import dataclasses
        import json

        items = json.loads(Path(args.batch_json).read_text(encoding="utf-8"))
        allowed = {"input_path", "output_path", "target_width", "target_height", "draft_size", "start_frame", "end_frame"}
        pipelines = []
        for item in items:
            overrides = {"input_path": item["input"], "output_path": item["output"]}
            overrides.update({k: v for k, v in item.items() if k in allowed})
            pipelines.append(OpenH3Pipeline(dataclasses.replace(settings, **overrides), OpenH3Progress(on_line=emit)))
        emit(f"[OpenH3] Starting Open-H3 regenerate batch | {len(pipelines)} input(s), models load once")
        started = time.time()
        try:
            results = run_many(pipelines, OpenH3Progress(on_line=emit), stages=stages, work_dir=work_dir)
        except KeyboardInterrupt:
            emit("[OpenH3] Cancelled by user")
            return 130
        except Exception as exc:
            import traceback

            traceback.print_exc()
            emit(f"[OpenH3] ERROR: {exc}")
            return 1
        ok = sum(1 for out, _err in results if out)
        emit(f"[OpenH3] Batch finished: {ok}/{len(results)} succeeded | total {time.time() - started:.1f}s")
        return 0 if ok == len(results) else (3 if ok else 1)

    emit(f"[OpenH3] Starting Open-H3 regenerate | {Path(args.input).name}")
    emit(f"[OpenH3] DiT: {Path(args.dit).name} | steps {args.steps} | start sigma {args.start_sigma or 'release grid'} | seed {args.seed}")
    started = time.time()
    pipeline = OpenH3Pipeline(settings, OpenH3Progress(on_line=emit))
    try:
        output, error = run_many([pipeline], pipeline.progress, stages=stages, work_dir=work_dir)[0]
        if error is not None:
            raise error
    except KeyboardInterrupt:
        emit("[OpenH3] Cancelled by user")
        return 130
    except Exception as exc:
        import traceback

        traceback.print_exc()
        emit(f"[OpenH3] ERROR: {exc}")
        return 1
    if "decode" in stages:
        emit(f"[OpenH3] OUTPUT_PATH: {output}")
    emit(f"[OpenH3] Total time: {time.time() - started:.1f}s")
    return 0


def run_jxl_cli(args, prompt: str) -> int:
    """JonXL H3 Upscale: one input, or every input of --batch_json, with the models loaded once."""
    import dataclasses
    import json

    from open_h3_engine.jxl import JXL_PROMPT, JXLSettings, run_jxl

    if args.plan_only:
        from open_h3_engine.jxl import plan_geometry, video_io

        if video_io.is_image(args.input):
            img = video_io.read_image(args.input)
            w, h, what = img.shape[1], img.shape[0], "image"
        else:
            info = video_io.probe_video(args.input)
            w, h, what = info.width, info.height, f"{info.frames} frame(s) @ {float(info.fps):.6g} fps, one H3 clip per frame"
        g = plan_geometry(w, h, int(args.jxl_max_side), int(args.jxl_max_side), args.jxl_geometry)
        emit(f"[Plan] JonXL H3 Upscale: {w}x{h} -> {g.out_w}x{g.out_h} (model canvas {g.width}x{g.height}), {what}")
        return 0
    if not args.jxl_lora or not Path(args.jxl_lora).is_file():
        emit(f"[OpenH3] ERROR: JonXL LoRA not found: {args.jxl_lora}")
        return 2
    base = JXLSettings(
        input_path=args.input, output_path=args.output, dit_path=args.dit, text_encoder_path=args.text_encoder,
        vae_path=args.decode_vae or args.vae, lora_path=args.jxl_lora, cache_dir=args.cache_dir,
        prompt=(prompt or "").strip() and prompt or JXL_PROMPT, seed=args.seed, steps=max(1, int(args.jxl_steps)),
        max_width=int(args.jxl_max_side), max_height=int(args.jxl_max_side), geometry=args.jxl_geometry,
        start_frame=args.start_frame, end_frame=args.end_frame, reserve_vram_gb=args.reserve_vram_gb,
        max_gpu_blocks=args.max_gpu_blocks, bytes_per_token=args.bytes_per_token, ffn_chunk_tokens=args.ffn_chunk_tokens,
        codec=args.codec, crf=args.crf, preset=args.preset, pixel_format=args.pixel_format, keep_audio=not args.no_audio)
    items = [base]
    if args.batch_json:
        allowed = {"input_path", "output_path", "start_frame", "end_frame"}
        items = []
        for item in json.loads(Path(args.batch_json).read_text(encoding="utf-8")):
            overrides = {"input_path": item["input"], "output_path": item["output"]}
            overrides.update({k: v for k, v in item.items() if k in allowed})
            items.append(dataclasses.replace(base, **overrides))
    emit(f"[OpenH3] Starting JonXL H3 Upscale | {len(items)} input(s) | LoRA {Path(args.jxl_lora).name} | "
         f"seed {args.seed} | max side {args.jxl_max_side}")
    started = time.time()
    try:
        results = run_jxl(items, emit)
    except KeyboardInterrupt:
        emit("[OpenH3] Cancelled by user")
        return 130
    except Exception as exc:
        import traceback

        traceback.print_exc()
        emit(f"[OpenH3] ERROR: {exc}")
        return 1
    ok = sum(1 for out, _err in results if out)
    if len(items) > 1:
        emit(f"[OpenH3] Batch finished: {ok}/{len(results)} succeeded | total {time.time() - started:.1f}s")
        return 0 if ok == len(results) else (3 if ok else 1)
    if results[0][1] is not None:
        emit(f"[OpenH3] ERROR: {results[0][1]}")
        return 1
    emit(f"[OpenH3] OUTPUT_PATH: {results[0][0]}")
    emit(f"[OpenH3] Total time: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
