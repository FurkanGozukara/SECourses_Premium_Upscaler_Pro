"""Single-file SparkVSR transformer INT8 ConvRot cache support."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from accelerate import init_empty_weights

from shared.torch_runtime_compat import configure_torch_runtime_compat

configure_torch_runtime_compat()

from diffusers import CogVideoXTransformer3DModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from shared.int8_convert_engine import Int8ConversionEngine
from shared.int8_convrot import int8_convrot_linear
from shared.sparkvsr_constants import (
    SPARKVSR_BF16_MODEL_NAME,
    SPARKVSR_INT8_CONVROT_CACHE_NAME,
    SPARKVSR_INT8_CONVROT_MODEL_NAME,
)
from shared.sparkvsr_fp8_scaled import _component_file, _format_bytes

# v3: V6.1 pipeline (policy exclusions, -128 range, LS scale refit, optional
# calibration features, ARA low-rank recovery, budgeted rescue) + portable
# validation so locally generated caches can be shipped to other machines.
INT8_CACHE_FORMAT = "sparkvsr-transformer-int8-convrot-v3"
INT8_GROUPSIZE_KEY_SUFFIX = ".int8_convrot_groupsize"
INT8_ARA_DOWN_SUFFIX = ".int8_ara_down"
INT8_ARA_UP_SUFFIX = ".int8_ara_up"

# Legacy (V6.0) directory-cache marker, only used to recognize old paths.
INT8_MANIFEST_NAME = "sparkvsr_int8_convrot_manifest.json"


def _normalize_cache_path(path: str | Path) -> Path:
    cache_path = Path(path)
    if cache_path.suffix.lower() == ".safetensors":
        return cache_path
    if cache_path.name == SPARKVSR_INT8_CONVROT_MODEL_NAME:
        return cache_path.parent / SPARKVSR_INT8_CONVROT_CACHE_NAME
    return cache_path


def is_int8_convrot_model_path(path: str | Path) -> bool:
    model_path = Path(path)
    return (
        model_path.name in {
            SPARKVSR_INT8_CONVROT_MODEL_NAME,
            SPARKVSR_INT8_CONVROT_CACHE_NAME,
        }
        or (model_path.is_dir() and (model_path / INT8_MANIFEST_NAME).exists())
    )


def default_int8_convrot_model_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / "SparkVSR" / "models" / SPARKVSR_INT8_CONVROT_CACHE_NAME


def int8_convrot_base_model_path(cache_path: str | Path) -> Path:
    path = _normalize_cache_path(cache_path)
    return path.parent / SPARKVSR_BF16_MODEL_NAME


def _source_metadata(source: Path) -> Dict[str, str]:
    transformer_file = _component_file(source, "transformer")
    stat = transformer_file.stat()
    return {
        "source": str(source.resolve()),
        "source_transformer": str(transformer_file.resolve()),
        "source_size": str(stat.st_size),
        "source_mtime_ns": str(stat.st_mtime_ns),
    }


def _has_valid_int8_cache(cache_path: Path, source: Path) -> bool:
    """
    Portable validation: the cache must carry the current format marker and
    per-layer group-size keys. When the BF16 source file is present its size
    must match the recorded one; path and mtime are informational only, so a
    cache generated on one machine loads on any other (shipped caches).
    """
    if not cache_path.is_file():
        return False
    try:
        with safe_open(str(cache_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            keys = list(handle.keys())
        if metadata.get("sparkvsr_int8_convrot") != "true":
            return False
        if metadata.get("int8_convrot_format") != INT8_CACHE_FORMAT:
            return False
        if not any(key.endswith(INT8_GROUPSIZE_KEY_SUFFIX) for key in keys):
            return False
        try:
            source_file = _component_file(source, "transformer")
        except Exception:
            return True
        if not source_file.is_file():
            return True
        recorded = metadata.get("source_size")
        return recorded is None or recorded == str(source_file.stat().st_size)
    except Exception:
        return False


def ensure_sparkvsr_int8_convrot_cache(
    *,
    int8_model_path: str | Path,
    bf16_model_path: str | Path,
    force: bool = False,
    calc_device: str = "cpu",
) -> Path:
    """Create or reuse one transformer-only safetensors cache."""
    output = _normalize_cache_path(int8_model_path)
    source = Path(bf16_model_path)

    if not force and _has_valid_int8_cache(output, source):
        print(f"[SparkVSR INT8] cache hit: {output}", flush=True)
        return output

    source_file = _component_file(source, "transformer")
    if not source_file.is_file():
        raise FileNotFoundError(
            f"SparkVSR BF16 transformer is required to build the INT8 cache: {source_file}"
        )

    engine = Int8ConversionEngine(
        source_file, calc_device=calc_device, log_prefix="[SparkVSR INT8]"
    )
    started = time.monotonic()
    print(
        f"[SparkVSR INT8] building transformer cache from {source_file} -> {output} "
        f"(calc device: {engine.calc_device})",
        flush=True,
    )

    state: Dict[str, torch.Tensor] = {}
    results = {}
    with safe_open(str(source_file), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        shapes: Dict[str, tuple] = {}
        for key in keys:
            if not key.endswith(".weight"):
                continue
            shape = handle.get_slice(key).get_shape()
            if len(shape) == 2:
                shapes[key[: -len(".weight")]] = (int(shape[0]), int(shape[1]))
        decisions = engine.plan(shapes)
        key_set = set(keys)

        for index, key in enumerate(keys, 1):
            base = key[: -len(".weight")] if key.endswith(".weight") else None
            decision = decisions.get(base) if base else None
            if decision is None or not decision.quantize:
                state[key] = handle.get_tensor(key).detach().cpu().contiguous()
            else:
                tensor = handle.get_tensor(key)
                result = engine.quantize_layer(
                    base,
                    tensor,
                    decision.group_size,
                    has_bias=f"{base}.bias" in key_set,
                    source_bytes_per_element=tensor.element_size(),
                )
                results[base] = result
                del tensor
            if index % 100 == 0 or index == len(keys):
                print(
                    f"[SparkVSR INT8] transformer: {index}/{len(keys)} tensors, "
                    f"quantized={len(results)}",
                    flush=True,
                )

        rescued = engine.select_rescue(results)
        for base, result in results.items():
            key = f"{base}.weight"
            if base in rescued:
                state[key] = handle.get_tensor(key).detach().cpu().contiguous()
                continue
            state[key] = result.q
            state[base + ".scale_weight"] = result.scale
            state[base + INT8_GROUPSIZE_KEY_SUFFIX] = torch.tensor(
                result.group_size, dtype=torch.int32
            )
            if result.ara_up is not None and result.ara_down is not None:
                state[base + INT8_ARA_UP_SUFFIX] = result.ara_up
                state[base + INT8_ARA_DOWN_SUFFIX] = result.ara_down
            if result.bias_delta is not None:
                bias_key = f"{base}.bias"
                bias = state.get(bias_key)
                if bias is not None:
                    state[bias_key] = (
                        (bias.float() - result.bias_delta).to(bias.dtype).contiguous()
                    )
    print(engine.summary_line(results, rescued), flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_name(output.name + f".{os.getpid()}.tmp")
    metadata = {
        "format": "pt",
        "int8_convrot_format": INT8_CACHE_FORMAT,
        "sparkvsr_int8_convrot": "true",
        "component": "transformer",
        "scale_dtype": "float32",
        "rotation": "hadamard-regular",
        "mse_clip": "true",
        "conversion_report": engine.report.metadata_json(),
        **_source_metadata(source),
    }
    try:
        save_file(state, str(temp_path), metadata=metadata)
        os.replace(temp_path, output)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        del state

    print(
        f"[SparkVSR INT8] wrote {output} ({_format_bytes(output.stat().st_size)}) "
        f"in {time.monotonic() - started:.1f}s",
        flush=True,
    )
    return output


def _patch_int8_linear_forward(module: nn.Linear) -> None:
    def int8_convrot_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        group_size = getattr(self, "_int8_convrot_gs", None)
        if group_size is None:
            group_size = int(self.int8_convrot_groupsize.item())
            self._int8_convrot_gs = group_size
        return int8_convrot_linear(
            x,
            self.weight,
            self.scale_weight,
            group_size,
            self.bias,
            getattr(self, "int8_ara_down", None),
            getattr(self, "int8_ara_up", None),
        )

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
        if name not in quantized_layers or not isinstance(module, nn.Linear):
            continue
        scale = state_dict[f"{name}.scale_weight"]
        module.register_buffer(
            "scale_weight",
            torch.empty(tuple(scale.shape), dtype=torch.float32, device="meta"),
        )
        module.register_buffer(
            "int8_convrot_groupsize",
            torch.empty((), dtype=torch.int32, device="meta"),
            persistent=True,
        )
        ara_down = state_dict.get(f"{name}{INT8_ARA_DOWN_SUFFIX}")
        ara_up = state_dict.get(f"{name}{INT8_ARA_UP_SUFFIX}")
        if ara_down is not None and ara_up is not None:
            module.register_buffer(
                INT8_ARA_DOWN_SUFFIX.lstrip("."),
                torch.empty(tuple(ara_down.shape), dtype=ara_down.dtype, device="meta"),
                persistent=True,
            )
            module.register_buffer(
                INT8_ARA_UP_SUFFIX.lstrip("."),
                torch.empty(tuple(ara_up.shape), dtype=ara_up.dtype, device="meta"),
                persistent=True,
            )
        module.weight.requires_grad_(False)
        _patch_int8_linear_forward(module)
        patched += 1
    return patched


def load_int8_convrot_transformer(
    cache_path: str | Path,
    bf16_model_path: str | Path | None = None,
) -> CogVideoXTransformer3DModel:
    path = _normalize_cache_path(cache_path)
    base_model = Path(bf16_model_path) if bf16_model_path else int8_convrot_base_model_path(path)
    state = load_file(str(path), device="cpu")
    config = CogVideoXTransformer3DModel.load_config(str(base_model / "transformer"))
    with init_empty_weights():
        model = CogVideoXTransformer3DModel.from_config(config)
    patched = _register_int8_buffers_and_patch(model, state)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"Unexpected SparkVSR INT8 transformer keys: {unexpected[:8]}")
    if missing:
        raise RuntimeError(f"Missing SparkVSR INT8 transformer keys: {missing[:8]}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(
        f"[SparkVSR INT8] cache hit: loaded {patched} transformer INT8 ConvRot Linear layers",
        flush=True,
    )
    return model


def count_int8_convrot_linears(model: nn.Module) -> int:
    return sum(
        1
        for module in model.modules()
        if isinstance(module, nn.Linear) and getattr(module, "_sparkvsr_int8_convrot", False)
    )


def summarize_int8_convrot_cache(model_path: str | Path) -> Dict[str, object]:
    path = _normalize_cache_path(model_path)
    dtype_counts: Dict[str, int] = {}
    tensor_count = 0
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor_count += 1
            tensor_slice = handle.get_slice(key)
            dtype = str(tensor_slice.get_dtype())
            params = 1
            for dim in tensor_slice.get_shape():
                params *= int(dim)
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + params
    return {
        "transformer": {
            "file": str(path),
            "bytes": path.stat().st_size,
            "tensors": tensor_count,
            "dtypes": dtype_counts,
        }
    }
