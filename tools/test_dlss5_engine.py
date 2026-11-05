"""DLSS 5 engine checks: exact ops against the NumPy reference recipe, model parity, speed and VRAM.

    venv\\Scripts\\python.exe tools/test_dlss5_engine.py            # CPU op checks (no GPU, no weights)
    venv\\Scripts\\python.exe tools/test_dlss5_engine.py --gpu      # + network on the visible CUDA GPU
    venv\\Scripts\\python.exe tools/test_dlss5_engine.py --gpu --reference <MLX-DLSS python dir> --image a.png
    venv\\Scripts\\python.exe tools/test_dlss5_engine.py --gpu --vram 1920x1080

The NumPy reference functions below restate the recovered recipe (noise, scaled
colour, Catmull-Rom history sampling); the torch ops must match them.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.dlss5 import DLSS5_MODELS_DIRNAME, DLSS5_WEIGHTS_FILE  # noqa: E402
from shared.dlss5 import ops  # noqa: E402


# --- NumPy reference (uint32 arithmetic) ------------------------------------------------------

def _np_half(v):
    return np.asarray(v, dtype=np.float32).astype(np.float16).astype(np.float32)


def _np_mix(value):
    value = value.astype(np.uint32, copy=False)
    shift = (value >> np.uint32(28)) + np.uint32(4)
    mixed = value ^ (value >> shift)
    return (mixed * np.uint32(0x108EF2D9)).astype(np.uint32)


def _np_uniform24(value):
    mixed = _np_mix(value)
    bits = (mixed >> np.uint32(30)) ^ (mixed >> np.uint32(8))
    return (bits + np.uint32(1)).astype(np.float32) * np.float32(5.960464477539063e-8)


def np_noise(height, width, frame_index):
    yy, xx = np.indices((height, width), dtype=np.uint32)
    seed = yy * np.uint32(0xD8163841)
    seed ^= xx * np.uint32(0x8DA6B343)
    seed ^= np.uint32((int(frame_index) * 0x9E3779B9) & 0xFFFFFFFF)
    seed ^= np.uint32(0x243F6A88)
    m = _np_mix(seed)
    mixed = m ^ (m >> np.uint32(22))
    ra = _np_uniform24(mixed * np.uint32(0xCAA5B80D) + np.uint32(0x21DD796B))
    ab = _np_uniform24(mixed * np.uint32(0x83232C31) + np.uint32(0x3463E0AC))
    rb = _np_uniform24(mixed * np.uint32(0x2C9277B5) + np.uint32(0xAC564B05))
    aa = _np_uniform24(mixed * np.uint32(0xFA6DC5F9) + np.uint32(0x4712A88E))
    ra = np.sqrt(np.float32(-2) * np.log(ra)).astype(np.float32)
    rb = np.sqrt(np.float32(-2) * np.log(rb)).astype(np.float32)
    tau = np.float32(6.2831854820251465)
    aa = (tau * aa).astype(np.float32)
    ab = (tau * ab).astype(np.float32)
    return np.stack((_np_half(rb * np.cos(aa)), _np_half(rb * np.sin(aa)), _np_half(ra * np.cos(ab))), axis=-1)


def np_sample_linear(image, x, y):
    height, width = image.shape[:2]
    px = x - np.float32(0.5)
    py = y - np.float32(0.5)
    x0 = np.clip(np.floor(px), 0, width - 1).astype(np.int64)
    y0 = np.clip(np.floor(py), 0, height - 1).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    tx = np.clip(px - x0, 0, 1).astype(np.float32)[..., None]
    ty = np.clip(py - y0, 0, 1).astype(np.float32)[..., None]
    top = image[y0, x0] * (1 - tx) + image[y0, x1] * tx
    bottom = image[y1, x0] * (1 - tx) + image[y1, x1] * tx
    return top * (1 - ty) + bottom * ty


def _np_catmull(normalized, dimension):
    pixel = normalized * np.float32(dimension) - np.float32(0.5)
    base_index = np.floor(pixel)
    t = np.clip(pixel - base_index, 0, 1).astype(np.float32)
    s = t * t
    c = s * t
    w0 = -0.5 * t + s - 0.5 * c
    w1 = 1 - 2.5 * s + 1.5 * c
    w2 = 0.5 * t + 2 * s - 1.5 * c
    w3 = -0.5 * s + 0.5 * c
    g = w1 + w2
    base = base_index + np.float32(0.5)
    lo, hi = np.float32(0.5), np.float32(dimension) - np.float32(0.5)
    return (np.clip(base - 1, lo, hi), np.clip(base + w2 / g, lo, hi), np.clip(base + 2, lo, hi), w0, w3, g)


def np_sample_history(history, u, v):
    height, width = history.shape[:2]
    xo0, xm, xo3, xw0, xw3, xg = _np_catmull(u, width)
    yo0, ym, yo3, yw0, yw3, yg = _np_catmull(v, height)
    weights = [xw0 * yg, xg * yw0, xg * yg, xg * yw3, xw3 * yg]
    taps = [np_sample_linear(history, xo0, ym), np_sample_linear(history, xm, yo0), np_sample_linear(history, xm, ym),
            np_sample_linear(history, xm, yo3), np_sample_linear(history, xo3, ym)]
    total = sum(w[..., None] * tap for w, tap in zip(weights, taps))
    return (total / sum(weights)[..., None]).astype(np.float32)


# --- checks -----------------------------------------------------------------------------------

def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""), flush=True)
    return ok


def cpu_checks() -> bool:
    ok = True
    cpu = torch.device("cpu")
    for (h, w, f) in ((64, 96, 0), (37, 53, 7), (320, 384, 123456)):
        # The integer hash must match bit for bit; sqrt/log/cos round 1 ulp differently between
        # NumPy and torch (and the GPU), which flips a few half-rounded values by one quantum.
        yy, xx = np.indices((h, w), dtype=np.uint32)
        seed = yy * np.uint32(0xD8163841)
        seed ^= xx * np.uint32(0x8DA6B343)
        seed ^= np.uint32((f * 0x9E3779B9) & 0xFFFFFFFF)
        seed ^= np.uint32(0x243F6A88)
        m = _np_mix(seed)
        ref_u = _np_uniform24((m ^ (m >> np.uint32(22))) * np.uint32(0xCAA5B80D) + np.uint32(0x21DD796B))
        s = ops._noise_base(h, w, "cpu") ^ ((f * 0x9E3779B9) & 0xFFFFFFFF) ^ 0x243F6A88
        mm = ops._dynamic_shift_mix(s)
        our_u = ops._uniform24(ops._u32((mm ^ (mm >> 22)) * 0xCAA5B80D + 0x21DD796B)).numpy()
        ok &= check(f"noise hash {h}x{w} frame {f} is bit-exact", bool(np.array_equal(our_u, ref_u)))
        diff = np.abs(ops.deterministic_noise(h, w, f, cpu).numpy() - np_noise(h, w, f))
        frac = float((diff > 0).mean())
        ok &= check(f"noise {h}x{w} frame {f} matches", frac < 1e-3 and float(diff.max()) <= 0.002,
                    f"{frac:.2e} of values one quantum off, max {float(diff.max()):.4f}")
    g = ops.NetworkGeometry.vendor_aligned(300, 200)
    ok &= check("vendor alignment", (g.network_width, g.network_height) == (320, 320), str(g))
    rows, cols = ops.mirror_indices(g, cpu)
    expect_rows = np.where(np.arange(320) < 200, np.arange(320), np.maximum(0, 2 * 200 - 2 - np.arange(320)))
    ok &= check("mirror extension index 2*extent-2-x", bool(np.array_equal(rows.numpy(), expect_rows)))
    rng = np.random.default_rng(1)
    hist = rng.random((40, 56, 3), dtype=np.float32)
    u = rng.uniform(-0.05, 1.05, (40, 56)).astype(np.float32)
    v = rng.uniform(-0.05, 1.05, (40, 56)).astype(np.float32)
    ours = ops.sample_history(torch.from_numpy(hist), torch.from_numpy(u), torch.from_numpy(v)).numpy()
    ref = np_sample_history(hist, u, v)
    diff = float(np.abs(ours - ref).max())
    ok &= check("Catmull-Rom history sampling matches", diff < 2e-5, f"max diff {diff:.2e}")
    color = rng.random((50, 70, 3), dtype=np.float32)
    feats = ops.make_features(torch.from_numpy(color), frame_index=3, geometry=ops.NetworkGeometry.vendor_aligned(70, 50))
    sc = _np_half(_np_half(_np_half(color) - np.float32(0.5)) * np.float32(0.125))
    ok &= check("scaled colour channels", float(np.abs(feats[0, :50, :70, 4:7].numpy() - sc).max()) == 0.0)
    ok &= check("first-frame history equals colour", bool(torch.equal(feats[..., 4:7], feats[..., 7:10])))
    ok &= check("feature extent", tuple(feats.shape) == (1, 320, 320, 16), str(tuple(feats.shape)))
    return ok


def load_image(path: str | None, size=(960, 544)) -> np.ndarray:
    from PIL import Image

    if path:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:size[1], 0:size[0]].astype(np.float32)
    base = np.stack((xx / size[0], yy / size[1], 0.5 + 0.5 * np.sin(xx / 13.0) * np.cos(yy / 17.0)), -1)
    return np.clip(base + rng.normal(0, 0.02, base.shape).astype(np.float32), 0, 1)


def gpu_checks(args) -> bool:
    from shared.dlss5.engine import DLSS5Engine, DLSS5Options, DLSS5Session

    ok = True
    device = torch.device("cuda", 0)
    name = torch.cuda.get_device_name(device)
    cap = torch.cuda.get_device_capability(device)
    print(f"GPU: {name} sm{cap[0]}{cap[1]}", flush=True)
    probe = torch.tensor([0.1234, 3.3, -500.0], device=device, dtype=torch.float16)
    cast = probe.clamp(-448, 448).to(torch.float8_e4m3fn).to(torch.float16).float().tolist()
    ok &= check("float8 E4M3 cast works on this GPU", cast[2] == -448.0, str(cast))
    weights = Path(args.weights)
    t0 = time.perf_counter()
    engine = DLSS5Engine.get(weights, device, args.precision)
    print(f"load {time.perf_counter() - t0:.1f}s, topology {engine.topology.spec}, dtype {engine.dtype}", flush=True)
    image = load_image(args.image)
    frame = torch.from_numpy(image).to(device)
    session = DLSS5Session(engine, DLSS5Options(temporal=False))
    torch.cuda.synchronize()
    out = session.render(frame)
    torch.cuda.synchronize()
    times = []
    for _ in range(3):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = session.render(frame)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t)
    h, w = image.shape[:2]
    print(f"{w}x{h}: {min(times) * 1000:.0f} ms/frame, peak {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB", flush=True)
    result = out.float().cpu().numpy()
    ok &= check("output finite and in range", bool(np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1))
    change = float(np.abs(result - image).mean())
    ok &= check("neural rendering changes the frame", change > 1e-3, f"mean |out - in| {change:.4f}")
    if args.save:
        from PIL import Image

        Image.fromarray((result * 255 + 0.5).astype(np.uint8)).save(args.save)
        print("saved", args.save)
    if args.reference:
        sys.path.insert(0, str(Path(args.reference)))
        from mlxdlss.pipeline import NeuralRenderingPipeline  # type: ignore

        for precision in ("fast", "reference"):
            ref_pipe = NeuralRenderingPipeline.from_safetensors(weights, device=device, precision=precision)
            ref = ref_pipe.enhance(image, frame_index=1).image
            # E4M3 rounding turns tiny GEMM order differences into one-quantum flips: MLX's own fast
            # and reference paths differ by MAE ~0.0028 on a 540p photo, so 0.004 bounds a faithful port.
            s2 = DLSS5Session(engine, DLSS5Options(temporal=False))
            s2.frame_index = 1
            ours = s2.render(frame).float().cpu().numpy()
            mae = float(np.abs(ours - ref).mean())
            ok &= check(f"parity with MLX-DLSS precision={precision}", mae < 4e-3, f"MAE {mae:.5f}")
            del ref_pipe
            torch.cuda.empty_cache()
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--weights", default=str(ROOT / DLSS5_MODELS_DIRNAME / DLSS5_WEIGHTS_FILE))
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--image", default=None)
    parser.add_argument("--save", default=None)
    parser.add_argument("--reference", default=None)
    args = parser.parse_args()
    ok = cpu_checks()
    if args.gpu:
        ok &= gpu_checks(args)
    print("ALL PASS" if ok else "SOME CHECKS FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
