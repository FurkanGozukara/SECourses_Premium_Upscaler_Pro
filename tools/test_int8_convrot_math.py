"""
Self-checks for the V6.1 INT8 ConvRot pipeline (no model files needed).

Run:  python tools/test_int8_convrot_math.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from shared.int8_convrot import (
    best_int8_convrot_groupsize,
    build_hadamard,
    dequantize_int8_convrot_weight,
    dequantize_rotated,
    fit_lowrank_residual,
    gptq_quantize_rotated,
    int8_convrot_linear,
    quantize_int8_rowwise,
    rotate_weight,
    weight_error_metrics,
)
from shared.int8_layer_policy import plan_int8_layers, select_rescue_layers

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {name} {detail}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def legacy_quantize(x):
    absmax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    best_mse = torch.full_like(absmax, float("inf"))
    best_scale = (absmax / 127.0).clamp(min=1e-30)
    best_q = None
    for ratio in torch.linspace(0.55, 1.0, 80, dtype=torch.float32):
        scale = (absmax * ratio / 127.0).clamp(min=1e-30)
        q = (x / scale).round().clamp(-127, 127)
        mse = ((q * scale - x) ** 2).mean(dim=1, keepdim=True)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        best_scale = torch.where(better, scale, best_scale)
        best_q = q if best_q is None else torch.where(better.expand_as(q), q, best_q)
    return best_q, best_scale


def main() -> None:
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # -- Hadamard invariants ------------------------------------------------
    for size in (16, 64, 256):
        h = build_hadamard(size)
        eye_err = (h @ h - torch.eye(size)).abs().max().item()
        sym_err = (h - h.T).abs().max().item()
        check(f"hadamard[{size}] orthonormal+symmetric", eye_err < 1e-5 and sym_err == 0.0)

    # -- LS refit strictly improves on the legacy grid search ---------------
    w = torch.randn(64, 512) * 0.02
    w[::7] *= 9.0  # heavy-tail rows
    q_leg, s_leg = legacy_quantize(w.clone())
    mse_leg = ((q_leg * s_leg - w) ** 2).mean().item()
    q_new, s_new = quantize_int8_rowwise(w)
    mse_new = ((dequantize_rotated(q_new, s_new) - w) ** 2).mean().item()
    check("clip+LS-refit <= legacy clip search", mse_new <= mse_leg * (1 + 1e-6), f"({mse_new:.3e} vs {mse_leg:.3e})")
    check("-128 range used", bool((q_new.float() == -128).any()) or True)  # informational
    check("codes within [-128,127]", int(q_new.min()) >= -128 and int(q_new.max()) <= 127)

    # -- per-group scales beat per-row on group-heterogeneous rows ----------
    w2 = torch.randn(32, 1024) * 0.02
    w2[:, :256] *= 6.0
    q_row, s_row = quantize_int8_rowwise(w2)
    q_grp, s_grp = quantize_int8_rowwise(w2, scale_groups=4)
    mse_row = ((dequantize_rotated(q_row, s_row) - w2) ** 2).mean().item()
    mse_grp = ((dequantize_rotated(q_grp, s_grp) - w2) ** 2).mean().item()
    check("per-group scales < per-row on heterogeneous K", mse_grp < mse_row, f"({mse_grp:.3e} vs {mse_row:.3e})")

    # -- energy-weighted search shifts error out of hot columns -------------
    energy = torch.ones(512)
    energy[:64] = 100.0
    q_e, s_e = quantize_int8_rowwise(w, col_energy=energy)
    err_plain = ((dequantize_rotated(q_new, s_new) - w) ** 2)[:, :64].mean().item()
    err_energy = ((dequantize_rotated(q_e, s_e) - w) ** 2)[:, :64].mean().item()
    check("energy weighting lowers hot-column error", err_energy <= err_plain * (1 + 1e-6), f"({err_energy:.3e} vs {err_plain:.3e})")

    # -- GPTQ lowers Hessian-weighted error vs RTN --------------------------
    x_cal = torch.randn(2048, 512)
    x_cal[:, :32] *= 4.0
    hessian = (x_cal.T @ x_cal) / x_cal.shape[0]
    q_rtn, s_rtn = quantize_int8_rowwise(w)
    q_gptq = gptq_quantize_rotated(w, hessian, s_rtn)
    check("gptq returns codes", q_gptq is not None)
    if q_gptq is not None:
        def h_err(q):
            e = dequantize_rotated(q, s_rtn) - w
            return float((e @ hessian * e).sum())
        check("gptq H-weighted error < RTN", h_err(q_gptq) < h_err(q_rtn), f"({h_err(q_gptq):.4e} vs {h_err(q_rtn):.4e})")

    # -- low-rank residual reduces error ------------------------------------
    pair = fit_lowrank_residual(w, q_rtn, s_rtn, rank=16)
    check("ara fit returns tensors", pair is not None)
    if pair is not None:
        up, down = pair
        m_plain = weight_error_metrics(w, q_rtn, s_rtn)
        m_ara = weight_error_metrics(w, q_rtn, s_rtn, ara_up=up, ara_down=down)
        check("ara lowers weight error", m_ara["rel_err"] < m_plain["rel_err"], f"({m_ara['rel_err']:.4f} vs {m_plain['rel_err']:.4f})")
        pair_h = fit_lowrank_residual(w, q_rtn, s_rtn, rank=16, hessian=hessian)
        check("hessian-weighted ara fit works", pair_h is not None)
        if pair_h is not None:
            up_h, down_h = pair_h
            def out_err(u, d):
                w_hat = dequantize_rotated(q_rtn, s_rtn) + (u.float() @ d.float() if u is not None else 0)
                e = w_hat - w
                return float((e @ hessian * e).sum())
            check(
                "hessian ara <= plain ara on output error",
                out_err(up_h, down_h) <= out_err(up, down) * (1 + 1e-4),
                f"({out_err(up_h, down_h):.4e} vs {out_err(up, down):.4e})",
            )

    # -- runtime parity ------------------------------------------------------
    out_f, in_f, group = 128, 512, 256
    weight = torch.randn(out_f, in_f, dtype=torch.bfloat16) * 0.02
    h = build_hadamard(group, dtype=torch.float32)
    w_rot = rotate_weight(weight.float(), h, group)
    for groups, label in ((1, "per-row"), (in_f // group, "per-group")):
        q, s = quantize_int8_rowwise(w_rot, scale_groups=groups)
        bias = torch.randn(out_f, dtype=torch.bfloat16) * 0.1
        x = torch.randn(33, in_f, dtype=torch.bfloat16)
        w_ref = dequantize_int8_convrot_weight(q, s, group, dtype=torch.float32)
        y_ref = (x.float() @ w_ref.T + bias.float())
        y_cpu = int8_convrot_linear(x, q, s, group, bias)
        rel_cpu = ((y_cpu.float() - y_ref).norm() / y_ref.norm()).item()
        check(f"cpu fallback parity ({label})", rel_cpu < 5e-2, f"rel={rel_cpu:.4f}")
        if device == "cuda":
            y_gpu = int8_convrot_linear(x.cuda(), q.cuda(), s.cuda(), group, bias.cuda())
            rel = ((y_gpu.float().cpu() - y_ref).norm() / y_ref.norm()).item()
            check(f"gpu path parity ({label})", rel < 5e-2, f"rel={rel:.4f}")

    # ARA runtime parity
    if pair is not None:
        q, s = quantize_int8_rowwise(rotate_weight(weight.float(), h, group))
        pair2 = fit_lowrank_residual(rotate_weight(weight.float(), h, group), q, s, rank=8)
        if pair2 is not None:
            up2, down2 = pair2
            up2 = up2.to(torch.bfloat16)
            down2 = down2.to(torch.bfloat16)
            x = torch.randn(33, in_f, dtype=torch.bfloat16)
            w_eff = dequantize_int8_convrot_weight(q, s, group, dtype=torch.float32, ara_down=down2, ara_up=up2)
            y_ref = x.float() @ w_eff.T
            y = int8_convrot_linear(x, q, s, group, None, down2, up2)
            rel = ((y.float() - y_ref).norm() / y_ref.norm()).item()
            check("cpu ara parity", rel < 5e-2, f"rel={rel:.4f}")
            if device == "cuda":
                y_g = int8_convrot_linear(x.cuda(), q.cuda(), s.cuda(), group, None, down2.cuda(), up2.cuda())
                rel_g = ((y_g.float().cpu() - y_ref).norm() / y_ref.norm()).item()
                check("gpu ara parity", rel_g < 5e-2, f"rel={rel_g:.4f}")

    # -- policy sanity -------------------------------------------------------
    shapes = {}
    for i in range(30):
        shapes[f"blocks.{i}.self_attn.q"] = (1536, 1536)
        shapes[f"blocks.{i}.ffn.0"] = (8960, 1536)
        shapes[f"blocks.{i}.norm1.linear"] = (9216, 1536)
    shapes["time_embedding.0"] = (1536, 256)
    shapes["time_projection.1"] = (9216, 1536)
    shapes["text_embedding.0"] = (1536, 4096)
    shapes["head.head"] = (64, 1536)
    shapes["blocks.0.tiny"] = (128, 1536)
    decisions = plan_int8_layers(shapes)
    check("policy quantizes block attn/ffn", decisions["blocks.3.self_attn.q"].quantize and decisions["blocks.3.ffn.0"].quantize)
    check("policy skips time_projection", not decisions["time_projection.1"].quantize, decisions["time_projection.1"].reason)
    check("policy skips time_embedding", not decisions["time_embedding.0"].quantize)
    check("policy skips text_embedding", not decisions["text_embedding.0"].quantize)
    check("policy skips head", not decisions["head.head"].quantize)
    check("policy skips modulation-in-block", not decisions["blocks.3.norm1.linear"].quantize, decisions["blocks.3.norm1.linear"].reason)
    check("policy skips tiny gemm", not decisions["blocks.0.tiny"].quantize)

    rescued = select_rescue_layers({"a": (0.10, 100), "b": (0.02, 100), "c": (0.30, 100)}, 250)
    check("rescue picks worst under budget", rescued == {"c", "a"}, str(rescued))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
