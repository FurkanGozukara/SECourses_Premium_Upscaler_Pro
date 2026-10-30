"""
Model loading for the standalone LTX 2.5 engine.

- Transformer: vendored LTXAV model, built from the checkpoint's embedded
  ComfyUI config metadata; quantized checkpoints (NVFP4 / INT8-ConvRot /
  INT4 W4A8 ConvRot) load through the vendored comfy mixed-precision ops on
  comfy_kitchen kernels — identical to how ComfyUI executes them.
- IC LoRA: additive runtime adapters on the exact layers the file targets
  (keeps quant fast paths intact); baked into plain bf16 weights when possible.
- Text encoder: vendored Gemma 4 12B stack (comfy sd1_clip framework) with
  the tokenizer loaded from the checkpoint's embedded tokenizer_json.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

import comfy.model_management as mm
import comfy.ops
import comfy.utils
import comfy.text_encoders.gemma4 as gemma4
import comfy.text_encoders.lt as lt
from comfy.ldm.lightricks.av_model import LTXAVModel
from comfy.quant_ops import QuantizedTensor

log = logging.getLogger("ltx25.loader")

GIB = 1024 ** 3
_VRAM_CAP: Dict[str, float] = {"bytes": 0.0, "save": 0.0}


def device_free_bytes(device: torch.device) -> int:
    """Free VRAM that other processes cannot take from us.

    On Windows (WDDM) cudaMemGetInfo does not see memory other processes hold, so a shared GPU
    looks empty; nvidia-smi reports the device-wide figure, so the smaller of the two is used."""
    free = int(mm.get_free_memory(device))
    if getattr(device, "type", "cuda") != "cuda" or not torch.cuda.is_available():
        return free
    try:
        import os
        import subprocess

        uuid = str(torch.cuda.get_device_properties(device).uuid)
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        ).stdout
        cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and parts[0].endswith(uuid):
                free = min(free, int(float(parts[1])) * 1024 * 1024 + max(0, cached))
    except Exception:
        pass
    return free


def apply_vram_cap(device: torch.device, save_gb: float, emit: Optional[Callable[[str], None]] = None) -> int:
    """Cap this process at (free VRAM now - Save VRAM): past it PyTorch raises an out-of-memory
    error instead of the Windows driver spilling into shared GPU memory (about 20x slower)."""
    if getattr(device, "type", "cuda") != "cuda" or not torch.cuda.is_available():
        return 0
    torch.empty(1, device=device)  # the CUDA context exists before the free figure is read
    total = int(torch.cuda.get_device_properties(device).total_memory)
    save = max(0.0, float(save_gb)) * GIB
    cap = int(torch.cuda.memory_reserved(device) + device_free_bytes(device) - save)
    cap = max(int(1 * GIB), min(cap, total))
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total), device)
    _VRAM_CAP.update(bytes=float(cap), save=save)
    if emit:
        emit(f"VRAM limit: {cap / GIB:.1f} GB of {total / GIB:.1f} GB (Save VRAM {float(save_gb):g} GB kept free); "
             "weights that do not fit stream from RAM instead of spilling into shared GPU memory")
    return cap


def budget_free_bytes(device: torch.device) -> int:
    """Free VRAM for planning, never above what the cap still allows this process."""
    free = device_free_bytes(device)
    if _VRAM_CAP["bytes"]:
        free = min(free, int(_VRAM_CAP["bytes"]) - torch.cuda.memory_allocated(device) + int(_VRAM_CAP["save"]))
    return max(0, int(free))

# Keys in the embedded transformer config that are not LTXAVModel kwargs.
_NON_MODEL_CONFIG_KEYS = {"image_model", "_class_name", "_diffusers_version", "scheduler"}


def _quant_disabled_set(load_device) -> set:
    disabled = set()
    if not mm.supports_nvfp4_compute(load_device):
        disabled.add("nvfp4")
    if not mm.supports_mxfp8_compute(load_device):
        disabled.add("mxfp8")
    if not mm.supports_fp8_compute(load_device):
        disabled.add("float8_e4m3fn")
        disabled.add("float8_e5m2")
    return disabled


def load_transformer(
    path: str,
    load_device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Tuple[LTXAVModel, Optional[dict], dict]:
    """Load the LTX 2.5 AV transformer with quant-aware operations.

    Returns (model_on_cpu, quant_config, transformer_config).
    """
    emit = on_progress or (lambda s: None)
    emit(f"Loading transformer checkpoint: {path}")
    sd, metadata = comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)
    sd = comfy.utils.state_dict_prefix_replace(sd, {"model.diffusion_model.": ""})
    metadata = dict(metadata or {})

    # _quantization_metadata layer names carry the on-disk prefix; strip them so
    # convert_old_quants injects tags that match the stripped state dict keys.
    if "_quantization_metadata" in metadata:
        try:
            qmeta = json.loads(metadata["_quantization_metadata"])
            layers = qmeta.get("layers", {})
            qmeta["layers"] = {
                (k[len("model.diffusion_model."):] if k.startswith("model.diffusion_model.") else k): v
                for k, v in layers.items()
            }
            metadata["_quantization_metadata"] = json.dumps(qmeta)
        except (ValueError, TypeError, AttributeError):
            pass

    sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)
    quant_config = comfy.utils.detect_layer_quantization(sd, "")

    if metadata is None or "config" not in metadata:
        raise RuntimeError(
            "The LTX 2.5 transformer checkpoint is missing its embedded config metadata."
        )
    config = json.loads(metadata["config"]).get("transformer")
    if not config:
        raise RuntimeError("Embedded metadata has no 'transformer' config section.")
    model_kwargs = {k: v for k, v in config.items() if k not in _NON_MODEL_CONFIG_KEYS}

    if quant_config is not None:
        disabled = _quant_disabled_set(load_device)
        operations = comfy.ops.mixed_precision_ops(quant_config, dtype, disabled=disabled)
        emit(
            "Quantized checkpoint detected — using ComfyUI mixed-precision ops"
            + (f" (emulated: {', '.join(sorted(disabled))})" if disabled else " (native kernels)")
        )
        # MixedPrecisionOps.Linear defers weight allocation to load time and
        # places loaded tensors on factory_kwargs["device"], so build on CPU.
        model = LTXAVModel(
            **model_kwargs,
            dtype=dtype,
            device="cpu",
            operations=operations,
        ).eval()
        emit("Attaching weights (quant-aware load)...")
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    else:
        operations = comfy.ops.disable_weight_init
        emit("BF16 checkpoint — using standard cast operations")
        # Meta construction avoids a full-size empty allocation (Windows commit
        # charge); assign=True then adopts the state dict tensors zero-copy.
        with torch.device("meta"):
            model = LTXAVModel(
                **model_kwargs,
                dtype=dtype,
                device="meta",
                operations=operations,
            )
        model = model.eval()
        emit("Attaching weights (zero-copy assign)...")
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)

    meta_left = [n for n, p in model.named_parameters() if p is not None and p.device.type == "meta"]
    meta_left += [n for n, b in model.named_buffers() if b is not None and b.device.type == "meta"]
    if meta_left:
        raise RuntimeError(
            f"Transformer load incomplete; {len(meta_left)} tensors missing, e.g. {meta_left[:5]}"
        )
    real_missing = [m for m in missing if not m.endswith(".comfy_quant")]
    if real_missing:
        log.info(f"Transformer: {len(real_missing)} missing keys, e.g. {real_missing[:5]}")
    if unexpected:
        log.info(f"Transformer: {len(unexpected)} unexpected keys ignored")
    return model, quant_config, config


class _LoraWeightPatch:
    """comfy `weight_function` entry: W' = W + strength * (B @ A).

    Applied by cast_bias_weight after the weight is cast (and dequantized for
    quantized layers) — identical to how ComfyUI's ModelPatcher applies LoRA.
    """

    def __init__(self, lora_a: torch.Tensor, lora_b: torch.Tensor, strength: float):
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.strength = float(strength)

    def __call__(self, weight: torch.Tensor) -> torch.Tensor:
        lora_a = self.lora_a
        if lora_a.device != weight.device:
            # Keep the tiny rank-32 factors resident on the compute device.
            self.lora_a = lora_a = lora_a.to(device=weight.device)
            self.lora_b = self.lora_b.to(device=weight.device)
        delta = (
            self.lora_b.to(dtype=weight.dtype) @ lora_a.to(dtype=weight.dtype)
        ) * self.strength
        return weight + delta


def apply_ic_lora(
    model: nn.Module,
    lora_path: str,
    strength: float = 1.0,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Tuple[int, float]:
    """Attach the IC LoRA. Bakes into plain weights, wraps quantized layers.

    Returns (patched_layer_count, reference_downscale_factor).
    """
    emit = on_progress or (lambda s: None)
    lora_sd, lora_meta = comfy.utils.load_torch_file(str(lora_path), safe_load=True, return_metadata=True)
    try:
        ref_downscale = float((lora_meta or {}).get("reference_downscale_factor", 1.0))
    except (TypeError, ValueError):
        ref_downscale = 1.0

    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in lora_sd.items():
        name = key
        if name.startswith("diffusion_model."):
            name = name[len("diffusion_model."):]
        elif name.startswith("transformer."):
            name = name[len("transformer."):]
        if name.endswith(".lora_A.weight"):
            pairs.setdefault(name[: -len(".lora_A.weight")], {})["a"] = tensor
        elif name.endswith(".lora_B.weight"):
            pairs.setdefault(name[: -len(".lora_B.weight")], {})["b"] = tensor

    modules = dict(model.named_modules())
    patched = 0
    if strength == 0.0:
        return 0, ref_downscale
    for target, ab in pairs.items():
        if "a" not in ab or "b" not in ab:
            continue
        module = modules.get(target)
        if module is None:
            log.warning(f"IC LoRA target not found in model: {target}")
            continue
        weight = getattr(module, "weight", None)
        if (
            isinstance(weight, torch.Tensor)
            and not isinstance(weight, QuantizedTensor)
            and weight.device.type != "meta"
        ):
            # Plain weight: bake once (no runtime cost). W' = W + strength * B @ A
            with torch.no_grad():
                new_weight = weight.to(torch.float32)
                new_weight += (
                    ab["b"].to(torch.float32) @ ab["a"].to(torch.float32)
                ) * float(strength)
                module.weight = nn.Parameter(new_weight.to(weight.dtype), requires_grad=False)
        else:
            # Quantized weight: register a comfy weight_function patch. The cast
            # dequantizes and applies it exactly like ComfyUI's ModelPatcher.
            patch = _LoraWeightPatch(
                ab["a"].to(torch.bfloat16), ab["b"].to(torch.bfloat16), strength
            )
            existing = list(getattr(module, "weight_function", []) or [])
            module.weight_function = existing + [patch]
        patched += 1
    emit(f"IC LoRA applied to {patched} layers (strength {strength}, ref downscale {ref_downscale:g})")
    return patched, ref_downscale


def place_transformer(
    model: LTXAVModel,
    device: torch.device,
    free_vram_bytes: int,
    activation_budget_bytes: int,
    reserve_bytes: int,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Tuple[int, int]:
    """
    ComfyUI-style partial residency: keep as many transformer blocks on the GPU
    as fit after reserving room for activations; the rest stay in system RAM
    and are streamed per-forward by the comfy cast ops (async offload stream).

    Returns (resident_blocks, total_blocks).
    """
    emit = on_progress or (lambda s: None)

    module_bytes = _module_bytes

    blocks = list(model.transformer_blocks)
    non_block_bytes = module_bytes(model) - sum(module_bytes(b) for b in blocks)

    budget = int(free_vram_bytes) - int(activation_budget_bytes) - int(reserve_bytes)
    budget -= non_block_bytes

    # Non-block components (embedders, connectors, adaln, out) always resident.
    for name, child in model.named_children():
        if name != "transformer_blocks":
            child.to(device)

    resident = 0
    used = 0
    for block in blocks:
        size = module_bytes(block)
        if used + size <= budget:
            block.to(device)
            used += size
            resident += 1
        else:
            break
    # CPU-resident blocks stream through the comfy cast ops; plain (BF16)
    # layers need comfy_cast_weights=True — the same flag ComfyUI's lowvram
    # loader sets — so their weights are cast to the input device per call.
    for block in blocks[resident:]:
        for module in block.modules():
            if hasattr(module, "comfy_cast_weights"):
                module.comfy_cast_weights = True
    sizes = (f" [free {free_vram_bytes / 1024 ** 3:.1f} GB, activations reserved {activation_budget_bytes / 1024 ** 3:.1f} GB, "
             f"non-block weights {non_block_bytes / 1024 ** 3:.2f} GB, block {module_bytes(blocks[0]) / 1024 ** 3:.3f} GB]")
    if resident < len(blocks):
        # Pin the CPU-resident tail for faster async streaming when allowed.
        emit(
            f"VRAM placement: {resident}/{len(blocks)} transformer blocks resident on GPU, "
            f"{len(blocks) - resident} streamed from RAM per step (ComfyUI-style offload)" + sizes
        )
    else:
        emit(f"VRAM placement: full model resident on GPU ({len(blocks)} blocks)" + sizes)
    return resident, len(blocks)


def transformer_weight_bytes(model: nn.Module) -> int:
    return _module_bytes(model)


def _tensor_bytes(t: torch.Tensor) -> int:
    """Bytes the tensor occupies on its device. A QuantizedTensor reports its original (bf16) dtype,
    which counted INT8 blocks at twice their size (0.72 instead of 0.37 GB) and kept half of the
    blocks that fit in RAM; its real storage is the quantized data plus the scale tensors."""
    if isinstance(t, QuantizedTensor):
        total = int(t._qdata.nbytes)
        params = getattr(t, "_params", None)
        fields = vars(params).values() if hasattr(params, "__dict__") else (
            getattr(params, name) for name in getattr(params, "__dataclass_fields__", {}))
        return total + sum(int(v.nbytes) for v in fields if torch.is_tensor(v))
    return t.numel() * t.element_size()


def _logical_bytes(module: nn.Module) -> int:
    """Size once dequantized (a QuantizedTensor reports its original dtype), e.g. a streamed layer's cast copy."""
    return sum(t.numel() * t.element_size() for t in list(module.parameters()) + list(module.buffers()))


def _module_bytes(module: nn.Module) -> int:
    total = 0
    for p in module.parameters(recurse=True):
        total += _tensor_bytes(p)
    for b in module.buffers(recurse=True):
        total += _tensor_bytes(b)
    return total


class Ltx25TextEncoder:
    """Gemma 4 12B (+dual projection) text encoder — vendored comfy stack."""

    def __init__(self, path: str, device: torch.device, dtype: torch.dtype = torch.bfloat16,
                 on_progress: Optional[Callable[[str], None]] = None):
        emit = on_progress or (lambda s: None)
        emit(f"Loading text encoder: {path}")
        sd, _metadata = comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)

        te_kwargs = {}
        norm_key = "model.layers.0.input_layernorm.weight"
        if norm_key in sd:
            te_kwargs["dtype_llama"] = sd[norm_key].dtype
        quant = comfy.utils.detect_layer_quantization(sd, "")
        if quant is not None:
            te_kwargs["llama_quantization_metadata"] = quant
            emit("Quantized text encoder detected — using ComfyUI mixed-precision ops")
        te_kwargs.update(lt.sd_detect([sd]))

        te_class = lt.ltxav_te(
            **te_kwargs,
            text_encoder_model=gemma4.gemma4_text_encoder_model(gemma4.Gemma4_12B),
            text_encoder_key="gemma4",
        )
        tokenizer_class = lt.ltxav_gemma4_tokenizer(gemma4.Gemma4_12B.tokenizer)
        tokenizer_data = {}
        if "tokenizer_json" in sd:
            tokenizer_data["tokenizer_json"] = sd.pop("tokenizer_json")
        self.tokenizer = tokenizer_class(tokenizer_data=tokenizer_data)

        self.model = te_class(device="cpu", dtype=dtype, model_options={})
        self.model.can_assign_sd = True
        missing, unexpected = self.model.load_sd(sd)
        real_missing = [m for m in missing if not m.startswith("visual.") and "rotary" not in m]
        if real_missing:
            log.info(f"Text encoder missing keys (non-fatal): {len(real_missing)}")
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer.tokenize_with_weights(text, return_word_ids=False)
        self.model.set_clip_options({"execution_device": self.device})
        out, _pooled, extra = self.model.encode_token_weights(tokens)
        # extra == {"unprocessed_ltxav_embeds": True}; connectors run in the DiT.
        return out

    def to_gpu(self, budget_bytes: Optional[int] = None, emit: Optional[Callable[[str], None]] = None):
        """Move the encoder to the GPU; with a budget, keep only the decoder layers that fit and
        stream the rest from RAM per call (comfy cast ops), so a small Save VRAM budget never
        pushes the 12B encoder into shared GPU memory."""
        transformer = self.model.gemma3_12b.transformer
        self.model.text_embedding_projection.to(self.device)
        layers = None
        for module in transformer.modules():
            if isinstance(module, nn.ModuleList) and (layers is None or len(module) > len(layers)):
                layers = module
        if budget_bytes is None or layers is None:
            transformer.to(self.device)
            return
        layer_ids = {id(layer) for layer in layers}
        # The encoder is budgeted at its dequantized (bf16) size: streamed INT8 layers are cast to bf16 on
        # the GPU, and counting the real INT8 size kept so many layers resident that 9-10 GB budgets failed
        # in the forward (the transformer, which reserves its cast buffers explicitly, uses real sizes).
        used = _logical_bytes(self.model.text_embedding_projection)
        for name, child in transformer.named_children():
            if child is layers:
                continue
            if not any(id(m) in layer_ids for m in child.modules()):
                child.to(self.device)
                used += _logical_bytes(child)
        sizes = [_logical_bytes(layer) for layer in layers]
        resident = 0
        for layer, size in zip(layers, sizes):
            if used + size <= budget_bytes:
                layer.to(self.device)
                used += size
                resident += 1
            else:
                for module in layer.modules():
                    if hasattr(module, "comfy_cast_weights"):
                        module.comfy_cast_weights = True
        if emit and resident < len(layers):
            emit(f"Text encoder: {resident}/{len(layers)} layers on the GPU, the rest stream from RAM (Save VRAM budget)")

    def unload(self):
        self.model.gemma3_12b.transformer.to("cpu")
        mm.soft_empty_cache()


def read_audio_latent_constants(audio_vae_path: Optional[str]) -> dict:
    """Read latent geometry constants from the audio VAE metadata (no weights)."""
    defaults = {"z_channels": 8, "freq_bins": 16, "latents_per_second": 25.0}
    if not audio_vae_path:
        return defaults
    try:
        with open(audio_vae_path, "rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_size))
        metadata = header.get("__metadata__", {})
        config = metadata.get("config")
        if isinstance(config, str):
            config = json.loads(config)
        audio_cfg = (config or {}).get("audio_vae", {})
        sample_rate = float(audio_cfg.get("sampling_rate", 16000))
        hop = float(audio_cfg.get("mel_hop_length", 160))
        mel_bins = int(audio_cfg.get("mel_bins", 64))
        z_channels = int(audio_cfg.get("z_channels", 8))
        return {
            "z_channels": z_channels,
            "freq_bins": mel_bins // 4,
            "latents_per_second": sample_rate / hop / 4.0,
        }
    except Exception as exc:  # pragma: no cover - metadata is advisory
        log.warning(f"Could not read audio VAE metadata ({exc}); using LTX 2.5 defaults")
        return defaults


def audio_latent_frames(frames_number: int, frame_rate: float, latents_per_second: float) -> int:
    return int(round((float(frames_number) / float(frame_rate)) * float(latents_per_second)))
