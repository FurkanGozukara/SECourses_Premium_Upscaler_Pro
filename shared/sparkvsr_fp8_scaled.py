"""
SparkVSR FP8-scaled cache generation and runtime loading.

This follows the same high-level strategy used by kohya-ss/musubi-tuner:
target Linear weights are quantized to FP8 E4M3 with block-wise scale tensors,
and Linear.forward dequantizes those weights back to the requested compute dtype.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from diffusers import CogVideoXTransformer3DModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import T5Config, T5EncoderModel

from shared.sparkvsr_constants import SPARKVSR_BF16_MODEL_NAME, SPARKVSR_FP8_SCALED_MODEL_NAME

FP8_MANIFEST_NAME = "sparkvsr_fp8_scaled_manifest.json"
FP8_BLOCK_SIZE = 64
FP8_DTYPE = torch.float8_e4m3fn


def is_fp8_scaled_model_path(path: str | Path) -> bool:
    model_path = Path(path)
    return model_path.name == SPARKVSR_FP8_SCALED_MODEL_NAME or (model_path / FP8_MANIFEST_NAME).exists()


def default_bf16_model_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / "SparkVSR" / "models" / SPARKVSR_BF16_MODEL_NAME


def default_fp8_scaled_model_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / "SparkVSR" / "models" / SPARKVSR_FP8_SCALED_MODEL_NAME


def _format_bytes(value: int) -> str:
    return f"{value / 1024 ** 3:.2f} GiB"


def _copy_file(src: Path, dst: Path, *, force: bool) -> None:
    if dst.exists() and not force:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path, *, force: bool) -> None:
    if not src.exists():
        return
    if dst.exists() and force:
        shutil.rmtree(dst)
    if not dst.exists():
        shutil.copytree(src, dst)


def _copy_layout(source: Path, output: Path, *, force: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in ("model_index.json", "LICENSE", "README.md", "README_zh.md"):
        src = source / name
        if src.exists():
            _copy_file(src, output / name, force=force)
    for name in ("assets", "prompt_embeddings", "scheduler", "tokenizer", "vae"):
        _copy_tree(source / name, output / name, force=force)
    for component in ("text_encoder", "transformer"):
        cfg = source / component / "config.json"
        if cfg.exists():
            _copy_file(cfg, output / component / "config.json", force=force)


def _component_file(model_path: Path, component: str) -> Path:
    if component == "text_encoder":
        return model_path / "text_encoder" / "model.safetensors"
    if component == "transformer":
        return model_path / "transformer" / "diffusion_pytorch_model.safetensors"
    raise ValueError(f"Unsupported SparkVSR FP8 component: {component}")


def _linear_weight_names_for_component(model_path: Path, component: str) -> Set[str]:
    if component == "text_encoder":
        # Musubi's scaled FP8 integrations primarily target DiT/transformer
        # blocks. Keep T5 in BF16 by default; text-encoder FP8 is a separate
        # quality tradeoff and made the real SparkVSR probe collapse to black.
        _ = T5Config.from_pretrained(str(model_path / "text_encoder"))
        return set()
    elif component == "transformer":
        config = CogVideoXTransformer3DModel.load_config(str(model_path / "transformer"))
        with init_empty_weights():
            model = CogVideoXTransformer3DModel.from_config(config)
    else:
        raise ValueError(f"Unsupported SparkVSR FP8 component: {component}")
    names: Set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if component == "transformer":
            if not name.startswith("transformer_blocks."):
                continue
            if ".norm" in name:
                continue
        names.add(f"{name}.weight")
    return names


def _quantize_linear_weight(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    original_dtype = tensor.dtype
    max_value = float(torch.finfo(FP8_DTYPE).max)
    min_value = -max_value
    work = tensor.detach().to(torch.float32)

    quant_mode = "tensor"
    original_shape = tuple(work.shape)
    if work.ndim == 2:
        out_features, in_features = int(work.shape[0]), int(work.shape[1])
        if in_features % FP8_BLOCK_SIZE == 0:
            quant_mode = "block"
            work_view = work.contiguous().view(out_features, in_features // FP8_BLOCK_SIZE, FP8_BLOCK_SIZE)
            scale = torch.amax(torch.abs(work_view), dim=2, keepdim=True) / max_value
        else:
            quant_mode = "channel"
            work_view = work
            scale = torch.amax(torch.abs(work_view), dim=1, keepdim=True) / max_value
    else:
        work_view = work
        scale = torch.amax(torch.abs(work_view).reshape(-1), dim=0, keepdim=True) / max_value

    scale = torch.clamp(scale, min=1e-8).to(torch.float32)
    quantized = torch.div(work_view, scale).nan_to_num_(0.0).clamp_(min=min_value, max=max_value).to(FP8_DTYPE)
    if quant_mode == "block":
        quantized = quantized.view(original_shape)
    return quantized.contiguous(), scale.to(dtype=original_dtype).contiguous()


def _convert_component_to_fp8_scaled(source: Path, output: Path, component: str, *, force: bool) -> Dict[str, object]:
    src_file = _component_file(source, component)
    out_file = _component_file(output, component)
    if out_file.exists() and not force:
        return {"component": component, "status": "exists", "bytes": out_file.stat().st_size}
    if not src_file.exists():
        raise FileNotFoundError(f"SparkVSR BF16 source component not found: {src_file}")

    target_weights = _linear_weight_names_for_component(source, component)
    state: Dict[str, torch.Tensor] = {}
    optimized = 0
    total = 0
    started = time.monotonic()
    print(f"[SparkVSR FP8] converting {component}: {src_file} ({_format_bytes(src_file.stat().st_size)})", flush=True)
    with safe_open(str(src_file), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for index, key in enumerate(keys, 1):
            tensor = handle.get_tensor(key)
            if key in target_weights:
                quantized, scale = _quantize_linear_weight(tensor)
                state[key] = quantized
                state[key.replace(".weight", ".scale_weight")] = scale
                optimized += 1
            else:
                state[key] = tensor.detach().cpu().contiguous()
            total += 1
            if index % 100 == 0 or index == len(keys):
                print(f"[SparkVSR FP8] {component}: {index}/{len(keys)} tensors, optimized={optimized}", flush=True)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state,
        str(out_file),
        metadata={
            "format": "pt",
            "sparkvsr_fp8_scaled": "true",
            "source": str(source.resolve()),
            "component": component,
            "fp8_dtype": "float8_e4m3fn",
            "scale_dtype": "source",
            "block_size": str(FP8_BLOCK_SIZE),
        },
    )
    del state
    for index_name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        stale = out_file.parent / index_name
        if stale.exists():
            stale.unlink()
    elapsed = time.monotonic() - started
    print(f"[SparkVSR FP8] {component}: wrote {out_file} ({_format_bytes(out_file.stat().st_size)}) in {elapsed:.1f}s", flush=True)
    return {
        "component": component,
        "status": "converted",
        "source_bytes": src_file.stat().st_size,
        "bytes": out_file.stat().st_size,
        "tensors": total,
        "optimized_linear_weights": optimized,
        "seconds": elapsed,
    }


def _has_valid_fp8_cache(path: Path) -> bool:
    return (
        (path / "model_index.json").exists()
        and _component_file(path, "text_encoder").exists()
        and _component_file(path, "transformer").exists()
        and (path / "vae" / "diffusion_pytorch_model.safetensors").exists()
    )


def ensure_sparkvsr_fp8_scaled_cache(
    *,
    fp8_model_path: str | Path,
    bf16_model_path: str | Path,
    force: bool = False,
) -> Path:
    output = Path(fp8_model_path)
    source = Path(bf16_model_path)
    if _has_valid_fp8_cache(output) and not force:
        return output
    if not (source / "model_index.json").exists():
        raise FileNotFoundError(f"SparkVSR BF16 model is required to build FP8 cache: {source}")
    if not _component_file(source, "text_encoder").exists() or not _component_file(source, "transformer").exists():
        raise FileNotFoundError(f"SparkVSR BF16 model is missing single-file text_encoder/transformer weights: {source}")

    started = time.monotonic()
    print(f"[SparkVSR FP8] building cache from {source} -> {output}", flush=True)
    _copy_layout(source, output, force=force)
    component_results = [
        _convert_component_to_fp8_scaled(source, output, "text_encoder", force=force),
        _convert_component_to_fp8_scaled(source, output, "transformer", force=force),
    ]
    manifest = {
        "format": "sparkvsr-fp8-scaled-v1",
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "fp8_dtype": "float8_e4m3fn",
        "block_size": FP8_BLOCK_SIZE,
        "components": component_results,
        "seconds": time.monotonic() - started,
    }
    (output / FP8_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def _patch_fp8_linear_forward(module: nn.Linear) -> None:
    def fp8_scaled_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        original_dtype = self.scale_weight.dtype
        if x.dtype != original_dtype:
            x = x.to(original_dtype)
        scale = self.scale_weight
        if scale.ndim < 3:
            weight = self.weight.to(original_dtype) * scale
        else:
            out_features, num_blocks, _ = scale.shape
            weight = self.weight.to(original_dtype).contiguous().view(out_features, num_blocks, -1)
            weight = (weight * scale).view(self.weight.shape)
        return F.linear(x, weight, self.bias)

    module.forward = fp8_scaled_forward.__get__(module, type(module))
    module._sparkvsr_fp8_scaled = True


def _register_scale_buffers_and_patch(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> int:
    scale_shapes = {
        key.rsplit(".scale_weight", 1)[0]: tuple(value.shape)
        for key, value in state_dict.items()
        if key.endswith(".scale_weight") and isinstance(value, torch.Tensor)
    }
    patched = 0
    for name, module in model.named_modules():
        if name in scale_shapes and isinstance(module, nn.Linear):
            module.register_buffer("scale_weight", torch.empty(scale_shapes[name], dtype=torch.bfloat16, device="meta"))
            _patch_fp8_linear_forward(module)
            patched += 1
    return patched


def _finalize_fp8_model(model: nn.Module) -> nn.Module:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_fp8_scaled_text_encoder(model_path: str | Path) -> T5EncoderModel:
    path = Path(model_path)
    state = load_file(str(_component_file(path, "text_encoder")), device="cpu")
    config = T5Config.from_pretrained(str(path / "text_encoder"))
    with init_empty_weights():
        model = T5EncoderModel(config)
    patched = _register_scale_buffers_and_patch(model, state)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"Unexpected SparkVSR FP8 text encoder keys: {unexpected[:8]}")
    missing = [key for key in missing if not key.endswith("encoder.embed_tokens.weight")]
    if missing:
        raise RuntimeError(f"Missing SparkVSR FP8 text encoder keys: {missing[:8]}")
    print(f"[SparkVSR FP8] loaded text encoder with {patched} FP8-scaled Linear layers", flush=True)
    return _finalize_fp8_model(model)


def load_fp8_scaled_transformer(model_path: str | Path) -> CogVideoXTransformer3DModel:
    path = Path(model_path)
    state = load_file(str(_component_file(path, "transformer")), device="cpu")
    config = CogVideoXTransformer3DModel.load_config(str(path / "transformer"))
    with init_empty_weights():
        model = CogVideoXTransformer3DModel.from_config(config)
    patched = _register_scale_buffers_and_patch(model, state)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"Unexpected SparkVSR FP8 transformer keys: {unexpected[:8]}")
    if missing:
        raise RuntimeError(f"Missing SparkVSR FP8 transformer keys: {missing[:8]}")
    print(f"[SparkVSR FP8] loaded transformer with {patched} FP8-scaled Linear layers", flush=True)
    return _finalize_fp8_model(model)


def summarize_fp8_scaled_cache(model_path: str | Path) -> Dict[str, object]:
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


def count_fp8_scaled_linears(model: nn.Module) -> int:
    return sum(1 for module in model.modules() if isinstance(module, nn.Linear) and getattr(module, "_sparkvsr_fp8_scaled", False))


def iter_linear_weight_dtypes(model: nn.Module) -> Iterable[torch.dtype]:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            yield module.weight.dtype
