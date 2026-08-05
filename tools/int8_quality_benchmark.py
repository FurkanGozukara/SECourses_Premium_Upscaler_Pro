"""
INT8 ConvRot weight-quality benchmark: V6.0 baseline vs the V6.1 pipeline.

Measures, per layer and model-wide, how close the quantized weights stay to
the BF16/FP16 originals (SQNR in dB / relative error), for:

  legacy  V6.0: old per-model exclusion lists, MSE clip search 0.55-1.0 x 80,
          clamp [-127, 127], per-row scales, nothing else.
  new     V6.1: generic sensitive-layer policy, clamp [-128, 127], LS scale
          refit, outlier clamp, budgeted rescue, plus - when calibration
          artifacts sit next to the source checkpoint - energy-weighted
          search, GPTQ rounding and Hessian/energy-weighted ARA low-rank
          recovery. Exactly the pipeline the converters ship.

Usage (from the repo root, venv python):

  python tools/int8_quality_benchmark.py --model sparkvsr
  python tools/int8_quality_benchmark.py --model seedvr2
  python tools/int8_quality_benchmark.py --model flashvsr --source <ckpt.safetensors>
  python tools/int8_quality_benchmark.py --source <any .safetensors> [--limit 20]

Signal base is the union of layers either configuration quantizes, so keeping
a sensitive layer in BF16 counts as zero error for that configuration (and is
charged in the reported cache-size column instead).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from safetensors import safe_open

from shared.int8_convert_engine import Int8ConversionEngine
from shared.int8_convrot import (
    best_int8_convrot_groupsize,
    build_hadamard,
    rotate_weight,
    weight_error_metrics,
)

DEFAULT_SOURCES = {
    "sparkvsr": (
        "SparkVSR/models/SparkVSR-bf16/transformer/diffusion_pytorch_model.safetensors",
    ),
    "seedvr2": (
        "SeedVR2/models/seedvr2_ema_7b_sharp_fp16.safetensors",
        "SeedVR2/models/*.safetensors",
    ),
    "flashvsr": (
        "ComfyUI-FlashVSR_Stable/models/**/*iffusion*.safetensors",
        "ComfyUI-FlashVSR_Stable/models/**/*.safetensors",
        "models/FlashVSR/**/*.safetensors",
    ),
}


def resolve_source(model: Optional[str], source: Optional[str]) -> Path:
    if source:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"--source not found: {path}")
        return path
    if not model:
        raise SystemExit("Pass --model {sparkvsr,seedvr2,flashvsr} or --source <safetensors>")
    for pattern in DEFAULT_SOURCES[model]:
        matches = sorted(REPO_ROOT.glob(pattern))
        matches = [m for m in matches if m.is_file() and "_int8_convrot" not in m.name.lower()]
        if matches:
            biggest = max(matches, key=lambda p: p.stat().st_size)
            return biggest
    raise SystemExit(
        f"Could not auto-locate the {model} source checkpoint; pass --source <path>. "
        f"Tried patterns: {DEFAULT_SOURCES[model]}"
    )


# --------------------------------------------------------------------- #
# V6.0 baseline, reproduced exactly
# --------------------------------------------------------------------- #
def legacy_quantize_rowwise(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    absmax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    best_mse = torch.full_like(absmax, float("inf"))
    best_scale = (absmax / 127.0).clamp(min=1e-30)
    best_q = None
    for ratio in torch.linspace(0.55, 1.0, 80, device=x.device, dtype=torch.float32):
        scale = (absmax * ratio / 127.0).clamp(min=1e-30)
        q = (x / scale).round().clamp(-127, 127)
        mse = ((q * scale - x) ** 2).mean(dim=1, keepdim=True)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        best_scale = torch.where(better, scale, best_scale)
        best_q = q if best_q is None else torch.where(better.expand_as(q), q, best_q)
    return best_q.to(torch.int8), best_scale


def legacy_plan(model: Optional[str], shapes: Dict[str, Tuple[int, int]]) -> Dict[str, int]:
    """Reproduce the V6.0 per-model layer selection. Returns name -> group size."""
    plan: Dict[str, int] = {}
    for name, (out_features, in_features) in shapes.items():
        group = best_int8_convrot_groupsize(in_features)
        if group is None:
            continue
        if model == "sparkvsr":
            if not name.startswith("transformer_blocks.") or ".norm" in name:
                continue
        elif model == "flashvsr":
            if name.endswith("head.head"):
                continue
        elif model == "seedvr2":
            if name.endswith(("vid_in.proj", "vid_out.proj", "txt_in")) or name.startswith("emb_in."):
                continue
        plan[name] = group
    return plan


def fmt_db(err_energy: float, sig_energy: float) -> str:
    if err_energy <= 0:
        return "lossless"
    return f"{10.0 * torch.log10(torch.tensor(sig_energy / err_energy)).item():.2f} dB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("sparkvsr", "seedvr2", "flashvsr"), default=None)
    parser.add_argument("--source", default=None, help="Source BF16/FP16 safetensors checkpoint")
    parser.add_argument("--limit", type=int, default=0, help="Only benchmark the first N candidate layers")
    parser.add_argument("--out", default=None, help="Markdown report path (default: benchmarks/int8_quality_<model>.md)")
    args = parser.parse_args()

    source = resolve_source(args.model, args.source)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label = args.model or source.stem
    print(f"[bench] source: {source}")
    print(f"[bench] device: {device}")

    engine = Int8ConversionEngine(source, calc_device=device, log_prefix="[bench:new]")

    started = time.monotonic()
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        shapes: Dict[str, Tuple[int, int]] = {}
        elem_bytes: Dict[str, int] = {}
        for key in keys:
            if not key.endswith(".weight"):
                continue
            sl = handle.get_slice(key)
            shape = sl.get_shape()
            if len(shape) == 2:
                name = key[: -len(".weight")]
                shapes[name] = (int(shape[0]), int(shape[1]))

        old_plan = legacy_plan(args.model, shapes)
        decisions = engine.plan(shapes)
        new_plan = {n: d.group_size for n, d in decisions.items() if d.quantize}
        union = sorted(set(old_plan) | set(new_plan))
        if args.limit > 0:
            union = union[: args.limit]
        print(f"[bench] layers: legacy quantizes {len(old_plan)}, new policy {len(new_plan)}, union {len(union)}")

        rows = []
        results = {}
        sig_total = 0.0
        err_old_total = 0.0
        old_metrics: Dict[str, Dict[str, float]] = {}
        for index, name in enumerate(union, 1):
            tensor = handle.get_tensor(f"{name}.weight")
            elem_bytes[name] = tensor.element_size()
            w = tensor.detach().to(device=device, dtype=torch.float32)
            sig = float((w * w).sum())
            sig_total += sig

            if name in old_plan:
                group = old_plan[name]
                h = build_hadamard(group, device=device, dtype=torch.float32)
                w_rot = rotate_weight(w, h, group)
                q_old, s_old = legacy_quantize_rowwise(w_rot)
                err = float(((q_old.float() * s_old - w_rot) ** 2).sum())
                old_metrics[name] = {"err": err, "sig": sig}
                err_old_total += err
                del w_rot, q_old, s_old
            else:
                old_metrics[name] = {"err": 0.0, "sig": sig}

            if name in new_plan:
                results[name] = engine.quantize_layer(
                    name,
                    tensor,
                    new_plan[name],
                    has_bias=False,  # bias handling does not affect weight metrics
                    source_bytes_per_element=tensor.element_size(),
                )
            del tensor, w
            if index % 50 == 0 or index == len(union):
                print(f"[bench] {index}/{len(union)} layers", flush=True)

        rescued = engine.select_rescue(results)
        err_new_total = 0.0
        for name in union:
            entry = {"name": name, "params": shapes[name][0] * shapes[name][1]}
            entry["old"] = old_metrics[name]
            if name in results and name not in rescued:
                result = results[name]
                # Unweighted metrics for a fair old-vs-new comparison
                group = new_plan[name]
                w = handle.get_tensor(f"{name}.weight").to(device=device, dtype=torch.float32)
                hmat = build_hadamard(group, device=device, dtype=torch.float32)
                w_rot = rotate_weight(w, hmat, group)
                metrics = weight_error_metrics(
                    w_rot,
                    result.q.to(device),
                    result.scale.to(device),
                    ara_up=result.ara_up,
                    ara_down=result.ara_down,
                )
                entry["new"] = {"err": metrics["err_energy"], "sig": metrics["sig_energy"]}
                entry["new_flags"] = ("gptq " if result.used_gptq else "") + (
                    "ara " if result.ara_up is not None else ""
                ) + ("energy" if result.used_energy else "")
                err_new_total += metrics["err_energy"]
                del w, w_rot
            else:
                reason = "rescued" if name in rescued else decisions.get(name).reason if name in decisions else "excluded"
                entry["new"] = {"err": 0.0, "sig": old_metrics[name]["sig"]}
                entry["new_flags"] = f"bf16 ({reason})"
            rows.append(entry)

    # Cache-size estimate: int8 weight + fp32 scales + bf16 ARA vs source bytes
    def config_bytes(plan, results_map, rescued_set):
        total = 0
        for name, (out_f, in_f) in shapes.items():
            n = out_f * in_f
            src = n * elem_bytes.get(name, 2)
            if name in plan and not (rescued_set and name in rescued_set):
                total += n  # int8
                result = results_map.get(name) if results_map else None
                if result is not None:
                    total += result.scale.numel() * 4
                    if result.ara_up is not None:
                        total += (result.ara_up.numel() + result.ara_down.numel()) * 2
                else:
                    total += out_f * 4
            else:
                total += src
        return total

    old_bytes = config_bytes(old_plan, None, None)
    new_bytes = config_bytes(new_plan, results, rescued)

    lines = []
    lines.append(f"# INT8 ConvRot quality benchmark - {label}")
    lines.append("")
    lines.append(f"- source: `{source}`")
    lines.append(f"- calibration: {'yes' if engine.stats else 'no'}, hessians (GPTQ): {'yes' if engine.hessians else 'no'}")
    lines.append(f"- features: `{engine.report.features}`")
    lines.append("")
    lines.append("## Model-wide (signal base = union of quantized layers)")
    lines.append("")
    lines.append("| config | quantized layers | weight SQNR | est. quantized-weights size |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| V6.0 legacy | {len(old_plan)} | {fmt_db(err_old_total, sig_total)} | {old_bytes / 1e9:.2f} GB |"
    )
    lines.append(
        f"| V6.1 new | {len(new_plan) - len(rescued)} (+{len(rescued)} rescued) | "
        f"{fmt_db(err_new_total, sig_total)} | {new_bytes / 1e9:.2f} GB |"
    )
    lines.append("")

    def sqnr(entry, cfg):
        err = entry[cfg]["err"]
        return float("inf") if err <= 0 else 10.0 * torch.log10(torch.tensor(entry[cfg]["sig"] / err)).item()

    worst = sorted((r for r in rows if r["new"]["err"] > 0), key=lambda r: sqnr(r, "new"))[:10]
    lines.append("## Worst 10 layers under the new pipeline")
    lines.append("")
    lines.append("| layer | params | V6.0 SQNR | V6.1 SQNR | V6.1 features |")
    lines.append("|---|---|---|---|---|")
    for r in worst:
        old_s = sqnr(r, "old")
        old_txt = "kept bf16" if r["old"]["err"] <= 0 else f"{old_s:.2f} dB"
        lines.append(
            f"| {r['name']} | {r['params'] / 1e6:.1f}M | {old_txt} | {sqnr(r, 'new'):.2f} dB | {r.get('new_flags', '')} |"
        )
    lines.append("")

    improved = sum(
        1 for r in rows
        if r["old"]["err"] > 0 and r["new"]["err"] > 0 and sqnr(r, "new") > sqnr(r, "old")
    )
    both = sum(1 for r in rows if r["old"]["err"] > 0 and r["new"]["err"] > 0)
    lines.append(f"Layers quantized by both where V6.1 wins on SQNR: {improved}/{both}")
    lines.append("")
    lines.append(f"Benchmark wall time: {time.monotonic() - started:.1f}s")

    report = "\n".join(lines)
    print("\n" + report)
    out_path = Path(args.out) if args.out else REPO_ROOT / "benchmarks" / f"int8_quality_{label}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\n[bench] wrote {out_path}")


if __name__ == "__main__":
    main()
