"""RelayVSR checks.

CPU (default):
1. Settings: defaults / guardrails / ORDER, the requested scale (1-4x, model always 4x), universal preset round trip.
2. VRAM plan: KV cache and activation formula, blocks on the GPU per budget (one formula for Auto Tune and the engine).
3. Runner command flags and model paths; downloader knows the files (mirror first, pinned upstream files).
4. INT8 ConvRot modules: quantize a small model, save, load into a meta-built model, same outputs.
--gpu [clip]: the real CLI on 17 frames: INT8 vs BF16 (> 40 dB) and a 10 GB-capped run identical to the uncapped one.
Run: venv/Scripts/python.exe tools/test_relayvsr.py [--gpu]
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "RelayVSR"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
FAILS: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + (f" ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def cpu_checks() -> None:
    import torch
    import torch.nn as nn

    from shared.relayvsr_constants import RELAYVSR_WAN_FILES, RELAYVSR_WAN_INT8
    from shared.relayvsr_runner import build_command, model_paths
    from shared.services.relayvsr_service import (
        RELAYVSR_ENGINE,
        RELAYVSR_ORDER,
        _enforce_relayvsr_guardrails,
        canonical_relayvsr_scale,
        relayvsr_defaults,
    )
    from shared.universal_preset import dict_to_values, get_all_defaults, values_to_dict
    from shared.vram_autotune import relayvsr as tune

    # 1) settings
    d = relayvsr_defaults()
    check("defaults cover ORDER", all(k in d for k in RELAYVSR_ORDER), str([k for k in RELAYVSR_ORDER if k not in d]))
    g = _enforce_relayvsr_guardrails({"upscale_factor": 2.2, "window_size": 500, "wan_model": "x", "scale": "2"}, d)
    check("guardrails", g["scale"] == "4" and g["upscale_factor"] == 2.0 and g["window_size"] == 64
          and g["wan_model"] == RELAYVSR_WAN_INT8, str({k: g[k] for k in ("scale", "upscale_factor", "window_size")}))
    check("requested scale 1-4x in 0.5 steps", [canonical_relayvsr_scale(upscale_factor_value=v) for v in (0.3, 2.2, 3.3, 9)]
          == [1.0, 2.0, 3.5, 4.0])
    check("model always 4x", RELAYVSR_ENGINE.fixed_model_scale == 4)
    defaults = get_all_defaults()
    check("universal preset has relayvsr", "relayvsr" in defaults)
    vals = dict_to_values("relayvsr", dict(defaults["relayvsr"], window_size=24), defaults["relayvsr"])
    back = values_to_dict("relayvsr", vals)
    check("preset round trip", back["window_size"] == 24 and len(vals) == len(RELAYVSR_ORDER))

    # 2) VRAM plan (measured on an RTX 5090: INT8, all blocks: 8.8 GB at 1080p, 18.8 GB at 4K output)
    p1080, p4k = tune.peak_gb("int8", 1920, 1080, 30), tune.peak_gb("int8", 3840, 2160, 30)
    check("peak formula covers the measured peaks", 8.8 <= p1080 <= 10.0 and 18.8 <= p4k <= 20.5, f"{p1080:.1f} / {p4k:.1f}")
    check("KV cache 4K", 11.0 < tune.kv_block_gb(3840, 2160) * 30 < 11.5, f"{tune.kv_block_gb(3840, 2160) * 30:.2f}")
    check("blocks per budget", tune.blocks_on_gpu(31.2, "int8", 3840, 2160) == 30
          and 0 < tune.blocks_on_gpu(10.0, "int8", 3840, 2160) < 30 and tune.blocks_on_gpu(3.0, "int8", 3840, 2160) == 0)

    # 3) runner + downloader
    s = dict(d, input_path="in.mp4", wan_model=RELAYVSR_WAN_INT8, window_size=20, wan_gpu_blocks=12, crf=17)
    paths = model_paths(ROOT, s)
    cmd = build_command("py", ROOT / "tools/relayvsr_inference.py", s, paths, "in.mp4", Path("out.mp4"))
    check("command flags", cmd[cmd.index("--window_size") + 1] == "20" and cmd[cmd.index("--max_gpu_blocks") + 1] == "12"
          and cmd[cmd.index("--crf") + 1] == "17" and Path(cmd[cmd.index("--wan") + 1]).name == RELAYVSR_WAN_FILES[RELAYVSR_WAN_INT8])
    dl = (ROOT.parent / "Models_Downloader.py").read_text(encoding="utf-8")
    check("downloader lists the files", all(f in dl for f in list(RELAYVSR_WAN_FILES.values()) + ["relayvsr_flash.safetensors"])
          and "--ensure-relayvsr" in dl)

    # 4) INT8 ConvRot modules on a small model
    from shared.int8_convrot_modules import Int8ConvRotLinear, load_int8_state, quantize_linears, save_int8_model
    from safetensors.torch import load_file, save_file

    torch.manual_seed(0)
    class Tiny(nn.Module):  # the layer policy quantizes block layers (blocks.N.*), like a transformer
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256))
                                         for _ in range(4)])

        def forward(self, x):
            for block in self.blocks:
                x = x + block(x)
            return x

    model = Tiny().to(torch.bfloat16)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src.safetensors"
        save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(src))
        x = torch.randn(4, 256, dtype=torch.bfloat16)
        ref = model(x).float()
        os.environ["SECOURSES_INT8_RESCUE_MB"] = "0"  # the tiny model would fit the rescue budget whole
        n = quantize_linears(model, src, log_prefix="[test INT8]")
        os.environ.pop("SECOURSES_INT8_RESCUE_MB", None)
        out = Path(tmp) / "int8.safetensors"
        save_int8_model(model, out, src, format_id="test-int8", marker_key="test_int8")
        with torch.device("meta"):
            fresh = Tiny()
        layers = load_int8_state(fresh, load_file(str(out)), torch.bfloat16)
        y = fresh(x).float()
        rel = float((y - ref).norm() / ref.norm())
        check("INT8 quantize / save / load", n >= 1 and layers == n and isinstance(fresh.blocks[0][0], Int8ConvRotLinear)
              and rel < 0.05, f"{n} layers, rel err {rel:.4f}")


def gpu_check(clip: str) -> None:
    py = sys.executable
    models = ROOT / "RelayVSR_Models"
    with tempfile.TemporaryDirectory() as tmp:
        def run(name, wan, *extra, cap=None):
            out = Path(tmp) / f"{name}.mp4"
            base = [str(ROOT / "tools/relayvsr_inference.py")] if cap is None else [str(ROOT.parent / "relayvsr_work/capped_run.py"), str(cap)]
            cmd = [py, "-u", *base, "--input", clip, "--output", str(out), "--end_frame", "16", "--wan", str(models / wan),
                   "--flash", str(models / "relayvsr_flash.safetensors"), "--no_audio", "--crf", "12", *extra]
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT),
                                  env=dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            done = [l for l in proc.stdout.splitlines() if "Done:" in l]
            check(f"{name} run", proc.returncode == 0 and out.is_file(), done[-1] if done else proc.stdout[-300:])
            return out

        int8 = run("int8", "relayvsr_wan_int8_convrot.safetensors")
        bf16 = run("bf16", "relayvsr_wan_bf16.safetensors")
        capped = run("int8_capped_10gb", "relayvsr_wan_int8_convrot.safetensors", cap=10)
        sys.path.insert(0, str(ROOT.parent / "relayvsr_work"))
        cmp = [py, str(ROOT.parent / "relayvsr_work/compare_videos.py")]
        r1 = subprocess.run(cmp + [str(bf16), str(int8)], capture_output=True, text=True).stdout.strip()
        r2 = subprocess.run(cmp + [str(int8), str(capped)], capture_output=True, text=True).stdout.strip()
        mean = lambda r: float(r.split("PSNR mean ")[1].split(" dB")[0]) if "PSNR mean" in r else 0.0  # noqa: E731
        check("INT8 vs BF16 > 40 dB", mean(r1) > 40, r1)
        check("10 GB capped run identical (block + KV swap)", mean(r2) > 100, r2)


if __name__ == "__main__":
    cpu_checks()
    if "--gpu" in sys.argv:
        rest = [a for a in sys.argv[1:] if a != "--gpu"]
        gpu_check(rest[0] if rest else str(ROOT.parent / "relayvsr_work/out/probe_960x540.mp4"))
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
