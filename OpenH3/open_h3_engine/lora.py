"""
Open-H3-Regenerate-2K adapters on the vendored MiniMax H3 DiT.

- Merged LoRA (refiner + LightX2V FL2VA Turbo + LMS Sharp, Diffusers keys, rank 576):
  keys mapped to the H3 layout with regen_core (q/k/v -> row slices of qkv_proj,
  SwiGLU halves swapped), then merged the way ComfyUI applies a LoRA to a
  quantized layer: dequantize, W += strength * B @ A in the compute dtype,
  requantize with the layer key as stochastic-rounding seed. Layers whose
  format cannot requantize keep the patch as a comfy weight_function instead.
- LMS Sharp's 50 AdaLN curve branches: delta(t) = B @ A @ curve(t), added to
  every block's AdaLN projection (ported from ComfyUI-Open-H3-Regenerate).
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import re
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

import comfy.float
import comfy.utils
from comfy.quant_ops import QuantizedTensor

from . import regen_core as core

log = logging.getLogger("openh3.lora")
Emit = Optional[Callable[[str], None]]


def _emit(emit: Emit, text: str) -> None:
    if emit:
        emit(text)


# ------------------------------------------------------------------ merged LoRA

def load_lora_pairs(path: str, dtype: torch.dtype) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, str, object]]:
    """{diffusers module: (A, B in ComfyUI row order, ComfyUI weight key, slot)}, read one tensor at a time."""
    from safetensors import safe_open

    pairs = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for module in core.lora_modules(set(f.keys())):
            target, slot = core.comfy_target(module)
            b = f.get_tensor(module + ".lora_B.weight").to(dtype)
            if slot == "swiglu":
                b = core.swap_swiglu_halves(b)
            pairs[module] = (f.get_tensor(module + ".lora_A.weight").to(dtype), b, target, slot)
    return pairs


class _LoraWeightPatch:
    """comfy weight_function entry (applied after the cast / dequantize): W' = W + strength * B @ A on rows."""

    def __init__(self, items, strength: float):
        self.items = items  # [(a, b, row_start or None)]
        self.strength = float(strength)

    def __call__(self, weight: torch.Tensor) -> torch.Tensor:
        weight = weight.clone()
        for a, b, start in self.items:
            delta = (b.to(weight.device, weight.dtype) @ a.to(weight.device, weight.dtype)) * self.strength
            if start is None:
                weight += delta
            else:
                weight[start:start + delta.shape[0]] += delta
        return weight


_KOHYA_MODULE = re.compile(r"^lora_unet_blocks_(\d+)_(attn_qkv_proj|attn_out_proj|mlp_fc1|mlp_fc2)$")
_KOHYA_LEAF = {"attn_qkv_proj": "attn.qkv_proj", "attn_out_proj": "attn.out_proj", "mlp_fc1": "mlp.fc1", "mlp_fc2": "mlp.fc2"}


def load_kohya_pairs(path: str, dtype: torch.dtype) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, str, object]]:
    """A kohya-format LoRA trained on the ComfyUI H3 layout ('lora_unet_blocks_<n>_<layer>' with lora_down / lora_up /
    alpha, such as JonXL's h3_upscale_1920): the names map one to one onto 'diffusion_model.blocks.<n>.<layer>.weight'
    (fused qkv and SwiGLU rows as the checkpoint stores them). A carries the alpha / rank scale.
    Returns {module: (A, B, weight key, None)}."""
    from safetensors import safe_open

    pairs = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        modules = sorted({k.rsplit(".lora_down.weight", 1)[0] for k in keys if k.endswith(".lora_down.weight")})
        if not modules:
            raise ValueError(f"{os.path.basename(str(path))} holds no kohya lora_down / lora_up pairs")
        for module in modules:
            match = _KOHYA_MODULE.match(module)
            if match is None or f"{module}.lora_up.weight" not in keys:
                raise ValueError(f"unexpected LoRA module {module} (expected lora_unet_blocks_<n>_<layer> pairs)")
            down = f.get_tensor(module + ".lora_down.weight").to(torch.float32)
            up = f.get_tensor(module + ".lora_up.weight").to(torch.float32)
            rank = down.shape[0]
            alpha = float(f.get_tensor(module + ".alpha")) if f"{module}.alpha" in keys else float(rank)
            target = f"diffusion_model.blocks.{int(match.group(1))}.{_KOHYA_LEAF[match.group(2)]}.weight"
            pairs[module] = ((down * (alpha / rank)).to(dtype), up.to(dtype), target, None)
    return pairs


@torch.no_grad()
def apply_kohya_lora(model: nn.Module, path: str, strength: float, device: torch.device,
                     compute_dtype: torch.dtype, emit: Emit = None) -> str:
    """Merge a kohya-format H3 LoRA (LoraLoaderModelOnly parity) into the DiT. Returns a one-line report."""
    if float(strength) == 0.0:
        return "LoRA: strength 0 (skipped)"
    _emit(emit, f"Loading LoRA: {os.path.basename(str(path))}")
    return _merge_pairs(model, load_kohya_pairs(path, compute_dtype), path, strength, device, compute_dtype, emit)


@torch.no_grad()
def apply_merged_lora(model: nn.Module, path: str, strength: float, device: torch.device,
                      compute_dtype: torch.dtype, emit: Emit = None) -> str:
    """Merge the Open-H3 LoRA into the DiT. Returns a one-line report."""
    if float(strength) == 0.0:
        return "LoRA: strength 0 (skipped - not the reference recipe)"
    _emit(emit, f"Loading Open-H3 merged LoRA: {os.path.basename(str(path))}")
    return _merge_pairs(model, load_lora_pairs(path, compute_dtype), path, strength, device, compute_dtype, emit)


def _merge_pairs(model: nn.Module, pairs, path: str, strength: float, device: torch.device,
                 compute_dtype: torch.dtype, emit: Emit = None) -> str:
    modules = dict(model.named_modules())

    grouped: Dict[str, list] = {}
    for module_name, (a, b, key, slot) in pairs.items():
        grouped.setdefault(key, []).append((module_name, a, b, slot))

    merged = patched = 0
    for key, items in grouped.items():
        layer_name = key[len("diffusion_model."):-len(".weight")]
        layer = modules.get(layer_name)
        if layer is None:
            raise ValueError(f"{key} is not in the model: select a MiniMax H3 checkpoint")
        weight = layer.weight
        rows = weight.shape[0]
        slices = []
        for module_name, a, b, slot in items:
            if isinstance(slot, int):
                if rows != 3 * b.shape[0]:
                    raise ValueError(f"{key}: {rows} rows, LoRA {module_name} gives {b.shape[0]} per q/k/v")
                slices.append((a, b, slot * b.shape[0]))
            else:
                if rows != b.shape[0]:
                    raise ValueError(f"{key}: {rows} rows, LoRA {module_name} gives {b.shape[0]}")
                slices.append((a, b, None))

        home = weight.device
        w = weight.data if isinstance(weight, nn.Parameter) else weight
        try:
            w_dev = w.to(device)
            if isinstance(w_dev, QuantizedTensor):
                full = w_dev.dequantize().to(compute_dtype)
            else:
                full = w_dev.to(compute_dtype)
            for a, b, start in slices:
                delta = (b.to(device, compute_dtype) @ a.to(device, compute_dtype)) * float(strength)
                if start is None:
                    full += delta
                else:
                    full[start:start + delta.shape[0]] += delta
                del delta
            seed = comfy.utils.string_to_seed(key)
            if isinstance(w_dev, QuantizedTensor):
                new = w_dev.requantize_from_float(full, scale="recalculate", stochastic_rounding=seed)
            else:
                new = comfy.float.stochastic_rounding(full, w.dtype, seed=seed) if w.dtype != compute_dtype else full
            layer.weight = nn.Parameter(new.to(home), requires_grad=False)
            merged += 1
            del full, w_dev, new
        except Exception as exc:  # format without requantize support: runtime patch (ComfyUI lowvram path)
            log.info("LoRA %s: runtime patch (%s)", key, exc)
            existing = list(getattr(layer, "weight_function", []) or [])
            layer.weight_function = existing + [_LoraWeightPatch(slices, strength)]
            patched += 1
    del pairs
    report = (f"LoRA {os.path.basename(str(path))}: {sum(len(v) for v in grouped.values())} modules on "
              f"{len(grouped)} weights, strength {float(strength):g}"
              + (f" ({merged} merged" + (f", {patched} runtime" if patched else "") + ")"))
    _emit(emit, report)
    return report


# ------------------------------------------------------------- Sharp curve branches

def read_table(path: str) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        names = [k for k in f.keys() if k.endswith("adaln_t_table")]
        if not names:
            raise ValueError(f"{os.path.basename(str(path))} has no adaln_t_table")
        return f.get_tensor(names[0]).to(torch.float32).contiguous()


def table_sha256(table: torch.Tensor) -> str:
    return hashlib.sha256(table.to(torch.float32).contiguous().numpy().tobytes()).hexdigest()


class _CurveState:
    """Shared by every block's wrapper: where the model time of a forward comes from."""

    def __init__(self, table, mode, model_table=None):
        self.table = table
        self.mode = mode                # "same" | "invert" | "recorded"
        self.model_table = model_table
        self.t = None
        self.warned = False
        self._device_tables = {}

    def table_on(self, device):
        if device not in self._device_tables:
            self._device_tables[device] = self.table.to(device)
        return self._device_tables[device]

    def coords(self, adaln_input):
        device = adaln_input.device
        if self.mode == "same":
            return adaln_input.to(torch.float32)
        if self.mode == "invert":
            t, err = core.invert_table(self.model_table, adaln_input.to(torch.float32))
            if err > 1e-3 and not self.warned:
                self.warned = True
                log.warning("Open-H3 Sharp curve: AdaLN input is %.2g off the model's curve", err)
            return core.table_coords(self.table_on(device), t)
        t = self.t
        if t is None or t.shape[0] != adaln_input.shape[0]:
            if not self.warned:
                self.warned = True
                log.warning("Open-H3 Sharp curve: no matching timestep recorded, curve branches skipped")
            return None
        return core.table_coords(self.table_on(device), t.to(device))


def _subclass(module, name, body):
    """Shallow copy of `module` with a subclass of its class: same parameters / child modules; only forward differs."""
    out = copy.copy(module)
    out.__class__ = type(name, (module.__class__,), body)
    return out


def _curve_linear(linear, weight, state, strength):
    def forward(self, *args, **kwargs):
        out = super(self.__class__, self).forward(*args, **kwargs)
        coords = state.coords(args[0] if args else kwargs["input"])
        if coords is None:
            return out
        w = self._open_h3_weights.get(coords.device)
        if w is None:
            w = self._open_h3_weights[coords.device] = weight.to(coords.device, torch.float32)
        return out + (coords @ w.t() * strength).to(out.dtype)

    wrapped = _subclass(linear, "OpenH3SharpCurveLinear", {"forward": forward})
    object.__setattr__(wrapped, "_open_h3_weights", {})
    return wrapped


def _recording_embedder(embedder, state):
    def forward(self, t):
        state.t = t.detach().to(torch.float32)
        return super(self.__class__, self).forward(t)

    return _subclass(embedder, "OpenH3RecordingTimeEmbedder", {"forward": forward})


@torch.no_grad()
def apply_sharp_curves(model: nn.Module, curve_path: str, reference_table: torch.Tensor,
                       strength: float, emit: Emit = None) -> str:
    from safetensors import safe_open

    if float(strength) == 0.0 or not curve_path:
        return "Sharp curve branches: off (not the reference recipe)"
    curve_form = bool(getattr(model, "use_adaln_curves", False))
    own = model.adaln_t_table.detach().to("cpu", torch.float32) if curve_form else None
    table = reference_table.to(torch.float32)
    sha = table_sha256(table)
    tag = "reference table" if sha == core.REFERENCE_TABLE_SHA256 else f"NOT the reference table (sha {sha[:12]})"
    if curve_form:
        mode = "same" if own.shape == table.shape and torch.equal(own, table) else "invert"
    else:
        mode = "recorded"
    state = _CurveState(table, mode, own)
    with safe_open(str(curve_path), framework="pt", device="cpu") as f:
        blocks = core.curve_blocks(set(f.keys()))
        if len(blocks) != len(model.blocks):
            raise ValueError(f"{os.path.basename(curve_path)}: {len(blocks)} curve branches for {len(model.blocks)} H3 blocks")
        for index, (key_a, key_b) in sorted(blocks.items()):
            a, b = f.get_tensor(key_a).to(torch.float32), f.get_tensor(key_b).to(torch.float32)
            proj = model.blocks[index].adaln_proj
            linear = proj.linear
            if b.shape[0] != linear.weight.shape[0] or a.shape[1] != table.shape[1]:
                raise ValueError(f"curve branch {index} {tuple(b.shape)} @ {tuple(a.shape)} does not fit "
                                 f"blocks.{index}.adaln_proj.linear {tuple(linear.weight.shape)}")
            proj.linear = _curve_linear(linear, b @ a, state, float(strength))
    if mode == "recorded":
        model.time_embedder = _recording_embedder(model.time_embedder, state)
    how = {"same": "the model's AdaLN curve is the reference curve",
           "invert": "model time read back from the model's own curve, branches use the reference curve",
           "recorded": "full (non-pruned) model, model time recorded from the time embedder"}[mode]
    report = f"Sharp curve branches: {len(blocks)} at {float(strength):g}, {tag}; {how}"
    _emit(emit, report)
    return report
