"""
Sol-Attn block-sparse self-attention for the LTX 2.5 transformer (comfy_kitchen kernel).

The generic attention-override path of OpenH3/open_h3_engine/sparse.py (vendored from
ComfyUI's BlockSparseAttention node), without the MiniMax-H3 producer: that module
imports the Open-H3 vendored comfy package, which cannot load next to LTX25/comfy.

Policy follows NVlabs/Sana `models/ltx2.5-refiner`: video self-attention of block 0
dense, blocks 1-47 sparse; cross-attention and the small audio attention stay dense
(their q/k shapes differ or they fall under min_tokens); tau may rise per step.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import torch

log = logging.getLogger("ltx25.sparse")

HEAD_DIM = 128


def sol_attn_available(device) -> bool:
    try:
        import comfy_kitchen as ck

        return bool(ck.sol_attn_is_available(device))
    except Exception:
        return False


class SparsePolicy:
    def __init__(self, taus: Sequence[float], dense_steps: int = 0, dense_blocks=(0,), min_tokens: int = 12288,
                 extra_tokens: int = 256):
        self.taus = [float(t) for t in taus] or [1.2]
        self.dense_steps = int(dense_steps)
        self.dense_blocks = set(int(b) for b in dense_blocks)
        self.min_tokens = int(min_tokens)
        self.extra_tokens = int(extra_tokens)
        self.step = 0
        self.block = -1
        self.sink_tokens = 0            # guide (IC-LoRA reference) tokens at the end of the video sequence
        self.calls_sparse = 0
        self.calls_dense = 0

    def tau(self) -> Optional[float]:
        """None = dense this step."""
        k = self.step - self.dense_steps
        if k < 0:
            return None
        return self.taus[min(k, len(self.taus) - 1)]


def install(model, policy: SparsePolicy) -> dict:
    """Return transformer_options carrying the override; block indices come from forward pre-hooks."""
    import comfy_kitchen as ck

    for i, block in enumerate(model.transformer_blocks):
        def _pre(_mod, _args, _i=i):
            policy.block = _i
        block.register_forward_pre_hook(_pre)

    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        def dense():
            policy.calls_dense += 1
            return func(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape, **kwargs)

        tau = policy.tau()
        if mask is not None or tau is None or policy.block in policy.dense_blocks:
            return dense()
        tokens = q.shape[2] if skip_reshape else q.shape[1]
        if tokens < policy.min_tokens or q.shape != k.shape or q.shape != v.shape or q.device.type != "cuda":
            return dense()
        if skip_reshape:
            b, _, _, dim_head = q.shape
            qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            qs, ks, vs = (t.view(b, -1, heads, dim_head) for t in (q, k, v))
        if dim_head != HEAD_DIM:
            return dense()
        if qs.dtype == torch.float32:
            qs, ks, vs = (t.to(torch.bfloat16) for t in (qs, ks, vs))
        # The appended guide tokens are the conditioning: every query attends them exactly.
        n_blocks = (tokens + 63) // 64
        sink = [0, 0]
        if 0 < policy.sink_tokens < tokens:
            sink = [(tokens - policy.sink_tokens) // 64, n_blocks]
        out = ck.sol_attn(qs, ks, vs, tau=tau, scale=kwargs.get("scale"), sink_blocks=sink, sink_q=[0, 0],
                          topk_ratio=0.0, token_aug=policy.extra_tokens).to(q.dtype)
        policy.calls_sparse += 1
        if skip_output_reshape:
            return out.transpose(1, 2)
        return out.reshape(b, -1, heads * dim_head)

    return {"optimized_attention_override": override}
