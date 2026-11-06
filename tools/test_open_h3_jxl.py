"""Open-H3 "JonXL H3 Upscale" workflow checks.

CPU (default):
1. Geometry: the app-side size (shared.open_h3_constants.jxl_output_size) equals the engine's plan_geometry; the
   canvas is a multiple of 32 around the exact-aspect picture; the workflow geometry of a 960x540 picture is
   JonXL's 1920x1056 canvas; first frames are exact uint8 pictures of the canvas size.
2. Schedule: beta57 (20 steps, shift 12) gives ComfyUI's values (cross-checked against ComfyUI's beta_scheduler and
   sample_er_sde: identical); er_sde is deterministic per seed.
3. LoRA: a kohya 'lora_unet_blocks_<n>_<layer>' file maps to 'diffusion_model.blocks.<n>.<layer>.weight' with the
   alpha / rank scale.
4. Runner / service / Chains: JonXL sizes, seed, fixed prompt, command flags, needed model files, settings
   guardrails, the Chains step size and label; the root downloader lists only the JonXL files.
--gpu [image]: a real 1-picture run through tools/open_h3_inference.py; with the ComfyUI reference of
   dlss5_work/v10_jxl (JonXL's graph in ComfyUI V92) it must match it (PSNR above 40 dB, workflow geometry).
Run: venv/Scripts/python.exe tools/test_open_h3_jxl.py [--gpu]
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "OpenH3"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FAILS: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + (f" ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def cpu_checks() -> None:
    import torch
    from comfy.cli_args import args as comfy_args, configure_from_env

    configure_from_env()
    if not torch.cuda.is_available():
        comfy_args.cpu = True  # the vendored model management picks its device at import
    from safetensors.torch import save_file

    from open_h3_engine import jxl, lora
    from shared import open_h3_constants as C

    # 1) geometry
    sizes = [(960, 540), (1672, 941), (1264, 720), (2528, 1440), (1080, 1920), (512, 512), (3840, 2160), (641, 359)]
    agree = all(C.jxl_output_size(w, h, s) == (jxl.plan_geometry(w, h, s, s).out_w, jxl.plan_geometry(w, h, s, s).out_h)
                for w, h in sizes for s in C.OPENH3_JXL_MAX_SIDES)
    check("app size == engine geometry", agree)
    g = jxl.plan_geometry(960, 540)
    check("960x540 -> 1920x1080 on a 1920x1088 canvas", (g.out_w, g.out_h, g.width, g.height, g.y0) == (1920, 1080, 1920, 1088, 4),
          f"{g}")
    ok = all(jxl.plan_geometry(w, h).width % 32 == 0 and jxl.plan_geometry(w, h).height % 32 == 0
             and jxl.plan_geometry(w, h).out_w % 2 == 0 and max(jxl.plan_geometry(w, h).out_w, jxl.plan_geometry(w, h).out_h) <= 1920
             for w, h in sizes)
    check("canvas multiple of 32, even output, longest side <= 1920", ok)
    gw = jxl.plan_geometry(960, 540, mode="workflow")
    check("workflow geometry 960x540 -> 1920x1056 (JonXL's canvas)", (gw.width, gw.height) == (1920, 1056))
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (540, 960, 3), dtype=np.uint8)
    first = jxl.prepare_first_frame(frame, g)
    check("keep first frame fills the canvas", first.shape == (1088, 1920, 3) and first.dtype == np.uint8)
    check("keep padding is the edge row", bool((first[0] == first[4]).all() and (first[-1] == first[-5]).all()))
    check("workflow first frame on the canvas", jxl.prepare_first_frame(frame, gw).shape == (1056, 1920, 3))

    # 2) schedule + sampler
    sig = jxl.beta57_sigmas(20)
    expect_head = [1.0, 0.997341, 0.992709, 0.986396]
    expect_tail = [0.171862, 0.056872, 0.0]
    check("beta57: 21 sigmas", len(sig) == 21)
    check("beta57 values (ComfyUI parity)",
          np.allclose(sig[:4].numpy(), expect_head, atol=2e-6) and np.allclose(sig[-3:].numpy(), expect_tail, atol=2e-6),
          f"{[round(float(v), 6) for v in sig[:4]]} ... {[round(float(v), 6) for v in sig[-3:]]}")
    x0 = torch.randn(1, 1, 512, generator=torch.Generator().manual_seed(1))

    def toy(x, s):
        return torch.tanh(x) * (1 - s.view(-1, 1, 1))

    a = jxl.sample_er_sde(toy, x0.clone(), sig.clone(), 505817956432961)
    b = jxl.sample_er_sde(toy, x0.clone(), sig.clone(), 505817956432961)
    c = jxl.sample_er_sde(toy, x0.clone(), sig.clone(), 1)
    check("er_sde deterministic per seed", torch.equal(a, b) and not torch.equal(a, c))
    check("frame grid: 24 -> 39 frames, 12 latent frames", (jxl.frame_count(24), jxl.video_latent_t(39)) == (39, 12))

    # 3) kohya LoRA mapping
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "k.safetensors"
        down, up = torch.randn(4, 16), torch.randn(48, 4)
        save_file({"lora_unet_blocks_3_attn_qkv_proj.lora_down.weight": down,
                   "lora_unet_blocks_3_attn_qkv_proj.lora_up.weight": up,
                   "lora_unet_blocks_3_attn_qkv_proj.alpha": torch.tensor(2.0),
                   "lora_unet_blocks_0_mlp_fc1.lora_down.weight": down.clone(),
                   "lora_unet_blocks_0_mlp_fc1.lora_up.weight": up.clone()}, str(path))
        pairs = lora.load_kohya_pairs(str(path), torch.float32)
        a_, b_, key, slot = pairs["lora_unet_blocks_3_attn_qkv_proj"]
        check("kohya key -> diffusion_model.blocks.3.attn.qkv_proj.weight", key == "diffusion_model.blocks.3.attn.qkv_proj.weight" and slot is None)
        check("kohya alpha / rank scale", torch.allclose(a_, down * 0.5) and torch.equal(b_, up))
        check("kohya without alpha = scale 1", torch.allclose(pairs["lora_unet_blocks_0_mlp_fc1"][0], down)
              and pairs["lora_unet_blocks_0_mlp_fc1"][2] == "diffusion_model.blocks.0.mlp.fc1.weight")

    # 4) runner, service, Chains
    from shared import open_h3_runner as R
    from shared.services import open_h3_service as S

    jxl_set = {"workflow": C.OPENH3_WORKFLOW_JXL, "jxl_max_side": 1920, "resolution_preset": "2k (2560x1440)",
               "seed": C.OPENH3_DEFAULT_SEED, "prompt": "something else", "max_target_resolution": 0}
    check("compute_target_size JonXL", R.compute_target_size(960, 540, jxl_set)[:2] == (1920, 1080))
    check("Max Edge lowers the JonXL size", R.compute_target_size(960, 540, {**jxl_set, "max_target_resolution": 1280})[:2] == (1280, 720))
    regen = {"resolution_preset": "1080p (1920x1080)", "upscale_factor": 2.0}
    check("Regenerate sizing unchanged", R.compute_target_size(960, 540, regen)[:2] == (1920, 1080))
    check("JonXL default seed", R._resolve_seed(jxl_set, print) == C.OPENH3_JXL_SEED)
    check("a changed seed is kept", R._resolve_seed({**jxl_set, "seed": 42}, print) == 42)
    check("Regenerate seed unchanged", R._resolve_seed({"seed": C.OPENH3_DEFAULT_SEED}, print) == C.OPENH3_DEFAULT_SEED)
    with tempfile.TemporaryDirectory() as tmp:
        pf = R._write_prompt_file(Path(tmp), jxl_set)
        check("fixed JonXL prompt", pf.read_text(encoding="utf-8") == C.OPENH3_JXL_PROMPT)
        paths, _notes, err = R.resolve_model_files(Path(tmp), jxl_set)
        need = R._needed_paths(paths, jxl_set)
        check("JonXL needs DiT + TE + VAE + LoRA only", sorted(need) == ["dit", "jxl_lora", "text_encoder", "vae"]
              and need["jxl_lora"].name == "h3_upscale_1920.safetensors")
        check("Regenerate needs no JonXL LoRA", "jxl_lora" not in R._needed_paths(paths, {}))
        cmd = R.build_command("py", Path("x.py"), jxl_set, paths, "in.mp4", Path("out.mp4"), 1, (1920, 1080), "768p", pf, Path(tmp))
        check("command selects the JonXL workflow", cmd[cmd.index("--workflow") + 1] == "jxl" and "--jxl_lora" in cmd
              and cmd[cmd.index("--jxl_steps") + 1] == "20")
        check("Regenerate command has no workflow flag", "--workflow" not in R.build_command(
            "py", Path("x.py"), {}, paths, "in.mp4", Path("out.mp4"), 1, (1920, 1080), "768p", pf, Path(tmp)))
    st = {"isolated_stages": True, **jxl_set, "fps_mode": "24 fps (reference resample)"}
    R._prepare_workflow(st, print)
    check("JonXL: 1:1 frames, one process", st["fps_mode"] == "source" and st["isolated_stages"] is False)
    defaults = S.open_h3_defaults() if hasattr(S, "open_h3_defaults") else None
    cfg = S._enforce_open_h3_guardrails({"workflow": "jonxl h3 upscale", "jxl_max_side": 777}, defaults or {})
    check("service keeps the workflow, resets a bad max size", cfg["workflow"] == C.OPENH3_WORKFLOW_JXL and cfg["jxl_max_side"] == 1920)
    check("settings order has the new keys last", S.OPENH3_ORDER[-2:] == ["workflow", "jxl_max_side"])
    from shared import chain_runner as CR

    check("Chains size of a JonXL step", CR._open_h3_size((960, 540), jxl_set, 4.0) == (1920, 1080))
    check("Chains size of a Regenerate step unchanged", CR._open_h3_size((960, 540), {"resolution_preset": "1080p"}, None) == (1920, 1080))
    from ui import chain_tab

    label = chain_tab._step_summary("open_h3", {"seed_controls": {"open_h3_settings": {**jxl_set, "model_name": "FL2VA Pruned W4A8"}}})
    check("Chains label names JonXL", label.startswith("JonXL H3 Upscale"), label)

    dl = subprocess.run([sys.executable, "-I", str(ROOT.parent / "Models_Downloader.py"), "--ensure-openh3", "FL2VA Pruned W4A8",
                         "--openh3-jxl", "--dry-run"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                        cwd=str(ROOT.parent), timeout=600,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    out = dl.stdout + dl.stderr
    check("downloader --openh3-jxl runs", dl.returncode == 0, out.strip().splitlines()[-1] if out.strip() else "")
    check("downloader skips the Regenerate-only files", "open_h3_regenerate_2k_lora" not in out and "AdaLN table" not in out)


def gpu_check(image: Path) -> None:
    from PIL import Image

    work = ROOT.parent / "dlss5_work" / "v10_jxl"
    ref = work / "comfy_out" / "suit_f45_last_00001_.png"
    m = ROOT / "OpenH3_Models"
    lora_file = m / "h3_upscale_1920.safetensors"
    if not lora_file.is_file():
        lora_file = ROOT.parent / "test_media" / "h3_upscale_1920.safetensors"
    swarm_vae = Path(r"F:/SwarmUI_Model_Downloader_v155/SwarmUI/Models/vae/MiniMaxH3/minimax_h3_video_vae_int8_convrot.safetensors")
    parity = ref.is_file() and swarm_vae.is_file() and image.name == "suit_f45.png"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "jxl.png"
        prompt = Path(tmp) / "p.txt"
        prompt.write_text("A static still blurry pixelated picture crossfades into a clear sharp detailed picture,  4psc4l3, "
                          if parity else "", encoding="utf-8")
        cmd = [sys.executable, "-u", str(ROOT / "tools" / "open_h3_inference.py"), "--workflow", "jxl",
               "--input", str(image), "--output", str(out),
               "--dit", str(m / ("minimax_h3_fl2va_pruned_int8_convrot.safetensors" if parity else "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors")),
               "--text_encoder", str(m / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
               "--vae", str(swarm_vae if parity else m / "minimax_h3_sol_vae_fp16.safetensors"),
               "--audio_vae", "x", "--upscaler", "x", "--lora", "x", "--jxl_lora", str(lora_file),
               "--prompt_file", str(prompt), "--seed", "505817956432961", "--cache_dir", str(Path(tmp) / "cache")]
        if parity:
            cmd += ["--jxl_geometry", "workflow"]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT),
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        log = proc.stdout + proc.stderr
        check("GPU run exits 0", proc.returncode == 0 and out.is_file(), log.strip().splitlines()[-1] if log.strip() else "")
        if not out.is_file():
            print(log[-3000:])
            return
        img = np.asarray(Image.open(out).convert("RGB"), dtype=np.float64)
        src = Image.open(image)
        if parity:
            r = np.asarray(Image.open(ref).convert("RGB"), dtype=np.float64)
            mse = ((img - r) ** 2).mean()
            psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-12))
            check("matches JonXL's graph in ComfyUI (PSNR > 40 dB)", psnr > 40, f"{psnr:.2f} dB")
        else:
            from shared.open_h3_constants import jxl_output_size

            check("output size", (img.shape[1], img.shape[0]) == jxl_output_size(*src.size), f"{img.shape}")
        check("log names the workflow", "JonXL H3 Upscale complete" in log and "20 steps er_sde / beta57" in log)


def main() -> int:
    cpu_checks()
    if "--gpu" in sys.argv:
        rest = [a for a in sys.argv[1:] if a != "--gpu"]
        default = ROOT.parent / "dlss5_work" / "v10_jxl" / "ref_in" / "suit_f45.png"
        gpu_check(Path(rest[0]) if rest else default)
    print("ALL PASS" if not FAILS else f"SOME FAILED: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
