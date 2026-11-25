"""Opt-in E4M3 weight storage and matrix multiplication for video transformers."""

import torch
from torch import nn


class FP8Linear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        weight = linear.weight.detach()
        scale = weight.abs().amax().float().div(448).clamp_min(1e-12).reshape(1)
        self.register_buffer("weight", (weight.float() / scale).to(torch.float8_e4m3fn))
        self.register_buffer("scale", scale)
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach())
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        value = value.reshape(-1, self.in_features)
        scale = value.abs().amax().float().div(448).clamp_min(1e-12).reshape(1)
        rows = 16_384
        if value.shape[0] <= rows:
            result = self._product(value, scale)
        else:
            # Hold one quantization workspace when a 4K attention projection contains many token rows.
            result = value.new_empty((value.shape[0], self.out_features))
            for start in range(0, value.shape[0], rows):
                end = min(start + rows, value.shape[0])
                result[start:end].copy_(self._product(value[start:end], scale))
        if self.bias is not None:
            result.add_(self.bias)
        return result.reshape(*shape[:-1], self.out_features)

    def _product(self, value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        codes = (value.float().div_(scale).clamp_(-448, 448)).to(torch.float8_e4m3fn).contiguous()
        return torch._scaled_mm(
            codes, self.weight.t(), scale_a=scale, scale_b=self.scale, out_dtype=value.dtype,
        )


@torch.no_grad()
def convert_fp8_blocks(blocks, device, dtype, log) -> int:
    """Convert aligned linear projections after a probe on the selected CUDA device."""
    device = torch.device(device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device) < (8, 9) or dtype != torch.bfloat16:
        log("FP8 requires an SM 8.9 or newer CUDA GPU and BF16 compute. Using the selected weight format.")
        return 0
    try:
        probe = FP8Linear(nn.Linear(64, 64, device=device, dtype=dtype))
        result = probe(torch.ones((16, 64), device=device, dtype=dtype))
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("FP8 matrix product returned non-finite values")
    except (RuntimeError, AttributeError, NotImplementedError, TypeError) as exc:
        log(f"FP8 matrix product unavailable ({exc}). Using BF16.")
        return 0

    converted = 0
    for block in blocks:
        for parent in tuple(block.modules()):
            for name, linear in tuple(parent.named_children()):
                if (isinstance(linear, nn.Linear) and min(linear.in_features, linear.out_features) >= 512
                        and linear.in_features % 16 == linear.out_features % 16 == 0):
                    setattr(parent, name, FP8Linear(linear))
                    converted += 1
    log(f"FP8 matrix multiplication enabled for {converted} linear layers.")
    return converted
