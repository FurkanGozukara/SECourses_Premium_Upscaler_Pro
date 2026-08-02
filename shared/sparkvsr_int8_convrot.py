"""
SparkVSR INT8 ConvRot cache generation and runtime loading.

Mirrors shared/sparkvsr_fp8_scaled.py, but the target Linear weights are
quantized to INT8 after a group-wise Hadamard rotation (ConvRot) with
per-output-channel MSE-optimized scales. Unlike the FP8-scaled path (which
dequantizes back to BF16 for every matmul), the INT8 path runs a real fused
INT8 GEMM at inference time - lower VRAM than BF16 *and* faster on
Turing (SM 7.5) or newer NVIDIA GPUs.

Quality: group-wise Hadamard rotation + per-row MSE clipping gives roughly
41 dB weight SQNR versus roughly 32 dB for scaled FP8, so outputs track the
BF16 reference more closely than FP8-scaled does.

The cache is generated automatically from the local SparkVSR-bf16 weights on
first use (exactly like the FP8-scaled cache) and reused afterwards.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from diffusers import CogVideoXTransformer3DModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import T5Config, T5EncoderModel

from shared.int8_convrot import (
    best_int8_convrot_groupsize,
    int8_convrot_linear,
    quantize_int8_convrot_weight,
)
from shared.sparkvsr_constants import SPARKVSR_BF16_MODEL_NAME, SPARKVSR_INT8_CONVROT_MODEL_NAME
from shared.sparkvsr_fp8_scaled import (
    _component_file,
    _copy_layout,
    _format_bytes,
    _linear_weight_names_for_component,
)

INT8_MANIFEST_NAME = "sparkvsr_int8_convrot_manifest.json"
INT8_GROUPSIZE_KEY_SUFFIX = ".int8_convrot_groupsize"


def is_int8_convrot_model_path(path: str | Path) -> bool:
    model_path = Path(path)
    return model_path.name == SPARKVSR_INT8_CONVROT_MODEL_NAME or (model_path / INT8_MANIFEST_NAME).exists()


def default_int8_convrot_model_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / "SparkVSR" / "models" / SPARKVSR_INT8_CONVROT_MODEL_NAME


def _convert_component_to_int8_convrot(
    source: Path,
    output: Path,
    component: str,
    *,
    force: bool,
    calc_device: str = "cpu",
) -> Dict[str, object]:
    src_file = _component_file(source, component)
    out_file = _component_file(output, component)
    if out_file.exists() and not force:
        return {"component": component, "status": "exists", "bytes": out_file.stat().st_size}
    if not src_file.exists():
        raise FileNotFoundError(f"SparkVSR BF16 source component not found: {src_file}")

    target_weights = _linear_weight_names_for_component(source, component)
    state: Dict[str, torch.Tensor] = {}
    optimized = 0
    skipped_groupsize = 0
    total = 0
    started = time.monotonic()
    print(
        f"[SparkVSR INT8] converting {component}: {src_file} ({_format_bytes(src_file.stat().st_size)})",
        flush=True,
    )
    with safe_open(str(src_file), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for index, key in enumerate(keys, 1):
            tensor = handle.get_tensor(key)
            group_size = None
            if key in target_weights and tensor.ndim == 2:
                group_size = best_int8_convrot_groupsize(int(tensor.shape[1]))
            if group_size is not None:
                quantized, scale = quantize_int8_convrot_weight(
                    tensor, group_size=group_size, calc_device=calc_device, mse_clip=True
                )
                base = key[: -len(".weight")]
                state[key] = quantized
                state[base + ".scale_weight"] = scale
                state[base + INT8_GROUPSIZE_KEY_SUFFIX] = torch.tensor(int(group_size), dtype=torch.int32)
                optimized += 1
            else:
                if key in target_weights:
                    skipped_groupsize += 1
                state[key] = tensor.detach().cpu().contiguous()
            total += 1
            if index % 100 == 0 or index == len(keys):
                print(
                    f"[SparkVSR INT8] {component}: {index}/{len(keys)} tensors, quantized={optimized}",
                    flush=True,
                )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state,
        str(out_file),
        metadata={
            "format": "pt",
            "sparkvsr_int8_convrot": "true",
            "source": str(source.resolve()),
            "component": component,
            "scale_dtype": "float32",
            "rotation": "hadamard-regular",
            "mse_clip": "true",
        },
    )
    del state
    for index_name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        stale = out_file.parent / index_name
        if stale.exists():
            stale.unlink()
    elapsed = time.monotonic() - started
    print(
        f"[SparkVSR INT8] {component}: wrote {out_file} ({_format_bytes(out_file.stat().st_size)}) in {elapsed:.1f}s",
        flush=True,
    )
    return {
        "component": component,
        "status": "converted",
        "source_bytes": src_file.stat().st_size,
        "bytes": out_file.stat().st_size,
        "tensors": total,
        "quantized_linear_weights": optimized,
        "skipped_groupsize": skipped_groupsize,
        "seconds": elapsed,
    }


def _has_valid_int8_cache(path: Path) -> bool:
    return (
        (path / "model_index.json").exists()
        and (path / INT8_MANIFEST_NAME).exists()
        and _component_file(path, "text_encoder").exists()
        and _component_file(path, "transformer").exists()
        and (path / "vae" / "diffusion_pytorch_model.safetensors").exists()
    )


def ensure_sparkvsr_int8_convrot_cache(
    *,
    int8_model_path: str | Path,
    bf16_model_path: str | Path,
    force: bool = False,
    calc_device: str = "cpu",
) -> Path:
    output = Path(int8_model_path)
    source = Path(bf16_model_path)
    if _has_valid_int8_cache(output) and not force:
        return output
    if not (source / "model_index.json").exists():
        raise FileNotFoundError(f"SparkVSR BF16 model is required to build the INT8 ConvRot cache: {source}")
    if not _component_file(source, "text_encoder").exists() or not _component_file(source, "transformer").exists():
        raise FileNotFoundError(f"SparkVSR BF16 model is missing single-file text_encoder/transformer weights: {source}")

    started = time.monotonic()
    if calc_device == "cpu" and torch.cuda.is_available():
        # The MSE clip search is 80 quantization passes per weight; the GPU
        # finishes the whole model in well under a minute.
        calc_device = "cuda"
    print(f"[SparkVSR INT8] building ConvRot cache from {source} -> {output} (calc device: {calc_device})", flush=True)
    _copy_layout(source, output, force=force)
    component_results = [
        _convert_component_to_int8_convrot(source, output, "text_encoder", force=force, calc_device=calc_device),
        _convert_component_to_int8_convrot(source, output, "transformer", force=force, calc_device=calc_device),
    ]
    manifest = {
        "format": "sparkvsr-int8-convrot-v1",
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "rotation": "hadamard-regular",
        "scale": "per-row-mse-clip",
        "group_sizes": "auto (256/64/16)",
        "components": component_results,
        "seconds": time.monotonic() - started,
    }
    (output / INT8_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def _patch_int8_linear_forward(module: nn.Linear) -> None:
    def int8_convrot_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        group_size = getattr(self, "_int8_convrot_gs", None)
        if group_size is None:
            group_size = int(self.int8_convrot_groupsize.item())
            self._int8_convrot_gs = group_size
        return int8_convrot_linear(x, self.weight, self.scale_weight, group_size, self.bias)

    module.forward = int8_convrot_forward.__get__(module, type(module))
    module._sparkvsr_int8_convrot = True


def _register_int8_buffers_and_patch(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> int:
    quantized_layers = {
        key[: -len(INT8_GROUPSIZE_KEY_SUFFIX)]
        for key in state_dict
        if key.endswith(INT8_GROUPSIZE_KEY_SUFFIX)
    }
    patched = 0
    for name, module in model.named_modules():
        if name in quantized_layers and isinstance(module, nn.Linear):
            scale = state_dict[f"{name}.scale_weight"]
            module.register_buffer("scale_weight", torch.empty(tuple(scale.shape), dtype=torch.float32, device="meta"))
            module.register_buffer(
                "int8_convrot_groupsize", torch.empty((), dtype=torch.int32, device="meta"), persistent=True
            )
            # load_state_dict(assign=True) re-wraps the checkpoint tensor as a
            # Parameter that inherits requires_grad from the existing one, and
            # integer Parameters cannot require grad - clear it up front.
            module.weight.requires_grad_(False)
            _patch_int8_linear_forward(module)
            patched += 1
    return patched


def _finalize_int8_model(model: nn.Module) -> nn.Module:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_int8_convrot_text_encoder(model_path: str | Path) -> T5EncoderModel:
    path = Path(model_path)
    state = load_file(str(_component_file(path, "text_encoder")), device="cpu")
    config = T5Config.from_pretrained(str(path / "text_encoder"))
    with init_empty_weights():
        model = T5EncoderModel(config)
    patched = _register_int8_buffers_and_patch(model, state)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"Unexpected SparkVSR INT8 text encoder keys: {unexpected[:8]}")
    missing = [key for key in missing if not key.endswith("encoder.embed_tokens.weight")]
    if missing:
        raise RuntimeError(f"Missing SparkVSR INT8 text encoder keys: {missing[:8]}")
    from shared.sparkvsr_fp8_scaled import _retie_t5_embeddings

    _retie_t5_embeddings(model)
    print(f"[SparkVSR INT8] loaded text encoder with {patched} INT8 ConvRot Linear layers", flush=True)
    return _finalize_int8_model(model)


def load_int8_convrot_transformer(model_path: str | Path) -> CogVideoXTransformer3DModel:
    path = Path(model_path)
    state = load_file(str(_component_file(path, "transformer")), device="cpu")
    config = CogVideoXTransformer3DModel.load_config(str(path / "transformer"))
    with init_empty_weights():
        model = CogVideoXTransformer3DModel.from_config(config)
    patched = _register_int8_buffers_and_patch(model, state)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"Unexpected SparkVSR INT8 transformer keys: {unexpected[:8]}")
    if missing:
        raise RuntimeError(f"Missing SparkVSR INT8 transformer keys: {missing[:8]}")
    print(f"[SparkVSR INT8] loaded transformer with {patched} INT8 ConvRot Linear layers", flush=True)
    return _finalize_int8_model(model)


def count_int8_convrot_linears(model: nn.Module) -> int:
    return sum(
        1
        for module in model.modules()
        if isinstance(module, nn.Linear) and getattr(module, "_sparkvsr_int8_convrot", False)
    )


def summarize_int8_convrot_cache(model_path: str | Path) -> Dict[str, object]:
    path = Path(model_path)
    summary: Dict[str, object] = {}
    for component in ("text_encoder", "transformer"):
        file_path = _component_file(path, component)
        dtype_counts: Dict[str, int] = {}
        tensor_count = 0
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor_count += 1
                dtype = str(handle.get_slice(key).get_dtype())
                shape = handle.get_slice(key).get_shape()
                params = 1
                for dim in shape:
                    params *= int(dim)
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + params
        summary[component] = {
            "file": str(file_path),
            "bytes": file_path.stat().st_size,
            "tensors": tensor_count,
            "dtypes": dtype_counts,
        }
    return summary
