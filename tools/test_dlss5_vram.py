"""DLSS 5 VRAM checks: the chunk planner, the out-of-memory retry and the sliced global attention.

    venv\\Scripts\\python.exe tools/test_dlss5_vram.py          # planner + retry (CPU, no weights)
    venv\\Scripts\\python.exe tools/test_dlss5_vram.py --gpu    # + 4K: sliced == whole tensors, peak under 3.5 GiB
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from shared.dlss5 import network  # noqa: E402
from shared.dlss5.engine import DLSS5Engine  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' (' + detail + ')') if detail else ''}", flush=True)
    return ok


class _FakeEngine(DLSS5Engine):
    """The planner and head() without weights: a runner that runs out of memory ``fail`` times."""

    def __init__(self, fail: int = 0, graphs: bool = True):  # noqa: D401 - no super().__init__
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.precision, self.dtype = "fp16", torch.float32
        self._graphs, self.graph_error = {}, ""
        self.use_graphs = graphs
        self.speed_label = "CUDA Graphs" if graphs else "eager"
        self.chunk_tokens = 1 << 16
        self.notes = []
        self.on_notice = self.notes.append
        self.fail = fail

        def runner(features):
            if self.fail > 0:
                self.fail -= 1
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")
            return features[..., :4] * 2

        self._runner = runner

    def _graph_head(self, features):
        return self._runner(features).to(torch.float32)


def cpu_checks() -> bool:
    ok = True
    gib = 2**30
    eng = _FakeEngine(graphs=True)
    ok &= check("graphs: fastest 64k chunk when VRAM is plentiful", eng.plan_chunk(3840 * 2176, 24 * gib) == 1 << 16)
    eng.use_graphs = False
    ok &= check("eager: 256k chunk when VRAM is plentiful", eng.plan_chunk(3840 * 2176, 24 * gib) == 1 << 18)
    eng.use_graphs = True
    small = eng.plan_chunk(3840 * 2176, int(4.75 * gib))
    ok &= check("tight VRAM: smaller chunk", small < 1 << 16, f"{small} tokens at 4.75 GiB for 4K")
    ok &= check("never below 16k tokens", eng.plan_chunk(3840 * 2176, 1 * gib) == 1 << 14)
    feats = torch.ones(1, 8, 8, 16)
    eng = _FakeEngine(fail=2, graphs=True)
    eng.chunk_tokens = 1 << 16
    out = eng.head(feats)
    ok &= check("OOM twice: chunk halves twice, frame rendered", eng.chunk_tokens == 1 << 14 and out.shape[-1] == 4,
                f"chunk {eng.chunk_tokens}, notes {len(eng.notes)}")
    eng = _FakeEngine(fail=2, graphs=True)
    eng.chunk_tokens = 1 << 15
    eng.head(feats)
    ok &= check("OOM at the smallest chunk: CUDA Graphs off, then rendered", not eng.use_graphs and eng.oom_retries == 2,
                "; ".join(eng.notes))
    eng = _FakeEngine(fail=5, graphs=False)
    eng.chunk_tokens = 1 << 14
    try:
        eng.head(feats)
        raised = False
    except torch.cuda.OutOfMemoryError:
        raised = True
    ok &= check("OOM with nothing left to lower: the error is reported", raised)
    network.set_chunk_tokens(1 << 18)
    return ok


def gpu_checks() -> bool:
    import numpy as np

    from shared.dlss5.engine import DLSS5Options, DLSS5Session
    from shared.dlss5.runner import weights_path

    ok = True
    dev = torch.device("cuda:0")
    engine = DLSS5Engine.get(weights_path(ROOT), dev, "fp16")
    gen = torch.Generator(device="cpu").manual_seed(7)
    frame = torch.rand((2160, 3840, 3), generator=gen).to(dev)
    results = {}
    for label, chunk in (("whole", 0), ("sliced", 1 << 16)):
        network.set_chunk_tokens(chunk)
        engine.chunk_tokens = chunk
        engine.set_speed_mode("Eager")
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        out = DLSS5Session(engine, DLSS5Options(temporal=False)).render(frame)
        torch.cuda.synchronize(dev)
        results[label] = (out.float().cpu().numpy(), (torch.cuda.max_memory_allocated(dev) - base) / 2**30)
        del out
    diff = float(np.abs(results["whole"][0] - results["sliced"][0]).max())
    ok &= check("4K sliced global attention == whole tensors", diff <= 2e-4, f"max diff {diff:.6f}")
    ok &= check("4K peak under 3.5 GiB with the 64k chunk", results["sliced"][1] < 3.5,
                f"{results['sliced'][1]:.2f} GiB (whole tensors {results['whole'][1]:.2f} GiB)")
    network.set_chunk_tokens(1 << 18)
    DLSS5Engine.release_all()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    ok = cpu_checks()
    if args.gpu:
        ok &= gpu_checks()
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
