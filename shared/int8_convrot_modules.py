"""
INT8 ConvRot for any torch model's Linear layers (app-level engines; the FlashVSR ComfyUI node keeps its own copy).

- Int8ConvRotLinear: drop-in nn.Linear replacement (Hadamard-rotated row-wise INT8 weight, float32 scales, INT8 GEMM).
- quantize_linears: convert the eligible Linear layers of a loaded model in place with the shared conversion engine
  (shared.int8_convert_engine; the layer policy keeps embeddings / heads in bf16).
- save_int8_model / load_int8_state: one safetensors file per model; INT8 layers are recognised by their
  "<layer>.int8_convrot_groupsize" key, everything else loads as the model's float weights.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

from shared.int8_convrot import int8_convrot_linear

GROUPSIZE_SUFFIX = ".int8_convrot_groupsize"
ARA_DOWN_SUFFIX = ".int8_ara_down"
ARA_UP_SUFFIX = ".int8_ara_up"


class Int8ConvRotLinear(nn.Module):
    """nn.Linear replacement whose weight is INT8 ConvRot (moves between devices, never changes dtype)."""

    def __init__(self, in_features: int, out_features: int, q_weight: torch.Tensor, scale: torch.Tensor,
                 group_size: int, bias: Optional[torch.Tensor], ara_down: Optional[torch.Tensor] = None,
                 ara_up: Optional[torch.Tensor] = None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.weight = nn.Parameter(q_weight, requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        self.register_buffer("scale_weight", scale.float())
        self.register_buffer("int8_ara_down", ara_down)
        self.register_buffer("int8_ara_up", ara_up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return int8_convrot_linear(x, self.weight, self.scale_weight, self.group_size, self.bias,
                                   self.int8_ara_down, self.int8_ara_up)

    def _apply(self, fn, recurse=True):
        # Module.to(dtype=...) must not round the float32 scales; only their device follows.
        scale = self._buffers.pop("scale_weight")
        try:
            super()._apply(fn, recurse)
        finally:
            target = fn(torch.empty(0, dtype=torch.int8, device=scale.device)).device
            self._buffers["scale_weight"] = scale.to(target)
        return self

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int8_convrot gs={self.group_size}"


def set_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, module)


def count_int8_linears(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, Int8ConvRotLinear))


def quantize_linears(model: nn.Module, source_checkpoint: str | Path, *, log_prefix: str = "[INT8]",
                     include: Optional[Callable[[str], bool]] = None) -> int:
    """Convert the eligible Linear layers of a loaded model in place; returns the number converted."""
    from shared.int8_convert_engine import Int8ConversionEngine

    engine = Int8ConversionEngine(source_checkpoint, log_prefix=log_prefix)
    candidates: Dict[str, nn.Linear] = {
        name: m for name, m in model.named_modules()
        if isinstance(m, nn.Linear) and m.weight.ndim == 2 and m.weight.device.type != "meta"
        and (include is None or include(name))
    }
    decisions = engine.plan({n: (int(m.weight.shape[0]), int(m.weight.shape[1])) for n, m in candidates.items()})
    results = {}
    for name, module in candidates.items():
        decision = decisions.get(name)
        if decision is None or not decision.quantize:
            continue
        results[name] = engine.quantize_layer(name, module.weight, decision.group_size,
                                              has_bias=module.bias is not None,
                                              source_bytes_per_element=module.weight.element_size())
    rescued = engine.select_rescue(results)
    patched = 0
    for name, result in results.items():
        if name in rescued:
            continue
        lin = candidates[name]
        bias = None
        if lin.bias is not None:
            bias = lin.bias.detach()
            if result.bias_delta is not None:
                bias = (bias.float() - result.bias_delta.to(bias.device)).to(bias.dtype)
        set_module(model, name, Int8ConvRotLinear(
            lin.in_features, lin.out_features, result.q.cpu(), result.scale.cpu(), int(result.group_size),
            None if bias is None else bias.cpu(),
            None if result.ara_down is None else result.ara_down.cpu(),
            None if result.ara_up is None else result.ara_up.cpu(),
        ))
        patched += 1
    model._int8_conversion_report = engine.report.metadata_json()
    print(engine.summary_line(results, rescued), flush=True)
    return patched


def save_int8_model(model: nn.Module, path: str | Path, source_checkpoint: str | Path, *, format_id: str,
                    marker_key: str, extra_metadata: Optional[Dict[str, str]] = None) -> Path:
    """The whole model (float + INT8 layers) as one atomic safetensors file with a format marker."""
    output = Path(path)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    quantized = 0
    for name, module in model.named_modules():
        if isinstance(module, Int8ConvRotLinear):
            state[f"{name}{GROUPSIZE_SUFFIX}"] = torch.tensor(module.group_size, dtype=torch.int32)
            quantized += 1
    if not quantized:
        raise RuntimeError("Refusing to save an INT8 ConvRot file without INT8 layers")
    source = Path(source_checkpoint)
    metadata = {
        "format": "pt",
        "int8_convrot_format": format_id,
        marker_key: "true",
        "quantized_linear_layers": str(quantized),
        "scale_dtype": "float32",
        "rotation": "hadamard-regular",
        "source": source.name,
        "source_size": str(source.stat().st_size) if source.is_file() else "",
    }
    metadata.update(extra_metadata or {})
    report = getattr(model, "_int8_conversion_report", None)
    if report:
        metadata["conversion_report"] = str(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_name(output.name + f".{os.getpid()}.tmp")
    try:
        save_file(state, str(temp_path), metadata=metadata)
        os.replace(temp_path, output)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return output


def file_metadata(path: str | Path) -> Dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def is_int8_file(path: str | Path, marker_key: str) -> bool:
    try:
        return file_metadata(path).get(marker_key) == "true"
    except Exception:
        return False


def load_int8_state(model: nn.Module, state: Dict[str, torch.Tensor], dtype: torch.dtype, *,
                    log_prefix: str = "[INT8]") -> int:
    """Swap the Linear layers named by the state's group-size keys for INT8 layers and assign every other tensor
    (model built on the meta device; float tensors are cast to dtype). Returns the INT8 layer count."""
    layers = {k[: -len(GROUPSIZE_SUFFIX)] for k in state if k.endswith(GROUPSIZE_SUFFIX)}
    modules = dict(model.named_modules())
    unknown = [n for n in layers if not isinstance(modules.get(n), nn.Linear)]
    if unknown:
        raise RuntimeError(f"{log_prefix} file does not match the model; unknown layers: {sorted(unknown)[:6]}")
    for name in sorted(layers):
        lin = modules[name]
        bias = state.pop(f"{name}.bias", None)
        set_module(model, name, Int8ConvRotLinear(
            lin.in_features, lin.out_features, state.pop(f"{name}.weight"), state.pop(f"{name}.scale_weight"),
            int(state.pop(f"{name}{GROUPSIZE_SUFFIX}").item()), None if bias is None else bias.to(dtype),
            state.pop(f"{name}{ARA_DOWN_SUFFIX}", None), state.pop(f"{name}{ARA_UP_SUFFIX}", None),
        ))
    rest = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in state.items()}
    expected = {k for k in model.state_dict(keep_vars=True)
                if not k.endswith(("scale_weight", "int8_ara_down", "int8_ara_up"))
                and not any(k in (f"{n}.weight", f"{n}.bias") for n in layers)}
    missing, unexpected = sorted(expected - set(rest)), sorted(set(rest) - expected)
    if missing or unexpected:
        raise RuntimeError(f"{log_prefix} file keys do not match the model (missing {missing[:6]}, "
                           f"unexpected {unexpected[:6]})")
    model.load_state_dict(rest, strict=False, assign=True)
    return len(layers)


def model_config_json(config) -> str:
    return json.dumps(dict(config), separators=(",", ":"), default=str)
