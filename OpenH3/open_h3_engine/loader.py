"""
Model loading for the standalone Open-H3 Regenerate engine.

Every model is built from the vendored ComfyUI 0.43 code with the same
operations ComfyUI picks for it:

- DiT: MiniMaxH3Model, config detected from the checkpoint (model_detection
  parity); quantized checkpoints (INT8 ConvRot / FP8 / W6A8 ...) load through
  comfy mixed-precision ops on comfy_kitchen kernels.
- Text encoder: Qwen3-VL-32B truncated to 50 layers (comfy.text_encoders.minimax).
- Video VAE: MiniMaxH3VideoVAE (stock decoder or the Sol fast decoder).
- Audio VAE: MiniMaxH3AudioVAE (only used to encode the held audio stream).
- Latent upscaler: MiniMaxH3LatentUpscaler (NVIDIA / LBH 3D-conv checkpoint).

`place_blocks` keeps as many blocks on the GPU as fit and streams the rest
from system RAM through the comfy cast ops (ComfyUI's lowvram behaviour).

`apply_vram_cap` limits this process to (free VRAM - Save VRAM): past the limit
PyTorch raises an out-of-memory error instead of the Windows driver spilling into
shared GPU memory (about 20x slower), and the pipeline then streams more blocks
(`stream_more_blocks`) and retries.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

import comfy.model_management as mm
import comfy.ops
import comfy.utils
from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
from comfy.ldm.minimax.latent_upscaler import MiniMaxH3LatentUpscaler
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy.ldm.minimax.vae import LATENTS_MEAN, LATENTS_STD, MiniMaxH3VideoVAE
from comfy.quant_ops import QuantizedTensor

log = logging.getLogger("openh3.loader")

Emit = Optional[Callable[[str], None]]


def _emit(emit: Emit, text: str) -> None:
    if emit:
        emit(text)


def device_free_bytes(device: torch.device) -> int:
    """Free VRAM that other processes cannot take from us.

    On Windows (WDDM) cudaMemGetInfo does not see memory other processes hold, so a shared
    GPU looks empty and an over-full placement pages through system RAM; nvidia-smi reports
    the device-wide figure, so the smaller of the two is used."""
    free = int(mm.get_free_memory(device))
    if getattr(device, "type", "cuda") != "cuda" or not torch.cuda.is_available():
        return free
    try:
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


GIB = 1024 ** 3
_VRAM_CAP: Dict[str, float] = {"bytes": 0.0, "save": 0.0}


def apply_vram_cap(device: torch.device, save_gb: float, emit: Emit = None) -> int:
    """Cap this process at (free VRAM now - save_gb) so it can never spill into shared memory.

    Returns the cap in bytes (0 when there is no CUDA device)."""
    if getattr(device, "type", "cuda") != "cuda" or not torch.cuda.is_available():
        return 0
    torch.empty(1, device=device)  # the CUDA context exists before the free figure is read
    total = int(torch.cuda.get_device_properties(device).total_memory)
    save = max(0.0, float(save_gb)) * GIB
    cap = int(torch.cuda.memory_reserved(device) + device_free_bytes(device) - save)
    cap = max(int(1 * GIB), min(cap, total))
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total), device)
    _VRAM_CAP.update(bytes=float(cap), save=save)
    _emit(emit, f"VRAM limit: {cap / GIB:.1f} GB of {total / GIB:.1f} GB (Save VRAM {float(save_gb):g} GB kept free); "
                f"anything larger streams from RAM instead of spilling into shared GPU memory")
    return cap


def vram_cap_bytes(device: torch.device) -> int:
    return int(_VRAM_CAP["bytes"]) or int(torch.cuda.get_device_properties(device).total_memory)


def usable_vram_bytes(device: torch.device) -> int:
    """VRAM this process can still allocate: under its cap and under what other processes left."""
    if getattr(device, "type", "cuda") != "cuda" or not torch.cuda.is_available():
        return 0
    usable = device_free_bytes(device) - int(_VRAM_CAP["save"])
    if _VRAM_CAP["bytes"]:
        usable = min(usable, int(_VRAM_CAP["bytes"]) - torch.cuda.memory_allocated(device))
    return max(0, int(usable))


class VramLimit(BaseException):
    """An out-of-memory raised inside a kernel wrapper that would otherwise swallow it.

    ComfyUI's sage attention catches every exception and quietly falls back to slower PyTorch
    attention; at the VRAM limit that hid the shortage from the block-swap auto-fix."""


def guard_attention_oom() -> None:
    import comfy.ldm.modules.attention as attention

    kernel = getattr(attention, "sageattn", None)
    if kernel is None or getattr(kernel, "_open_h3_guard", False):
        return

    def guarded(*args, **kwargs):
        try:
            return kernel(*args, **kwargs)
        except torch.OutOfMemoryError as exc:
            raise VramLimit(oom_note(exc) + " (sage attention)") from None

    guarded._open_h3_guard = True
    attention.sageattn = guarded


def oom_shortfall(exc: BaseException, device: torch.device) -> int:
    """Bytes that must be freed for the failed allocation to fit under the cap (+0.5 GB margin).

    Called inside the except block, while the failed step's tensors are still alive."""
    import re

    match = re.search(r"Tried to allocate ([0-9.]+) ([KMG])iB", str(exc))
    requested = float(match.group(1)) * {"K": 1024, "M": 1024 ** 2, "G": GIB}[match.group(2)] if match else GIB
    cap = int(_VRAM_CAP["bytes"]) or int(torch.cuda.get_device_properties(device).total_memory)
    short = requested + torch.cuda.memory_reserved(device) - cap
    return int(max(0.0, short) + 0.5 * GIB)


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, (torch.OutOfMemoryError, VramLimit)) or (
        isinstance(exc, Exception) and "out of memory" in str(exc).lower())


def oom_note(exc: BaseException) -> str:
    """'Tried to allocate X GiB' plus where it happened, for the run log."""
    import re
    import traceback

    text = str(exc)
    if isinstance(exc, VramLimit):
        return text
    match = re.search(r"Tried to allocate [0-9.]+ [KMG]iB", text)
    where = ""
    try:
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        where = f" in {os.path.basename(frame.filename)}:{frame.lineno} {frame.name}"
    except Exception:
        pass
    return (match.group(0) if match else text.splitlines()[0][:120]) + where


def tensor_bytes(t: torch.Tensor) -> int:
    """Real storage size (a QuantizedTensor reports its logical full-precision shape)."""
    data = t.data if isinstance(t, nn.Parameter) else t
    if isinstance(data, QuantizedTensor):
        total = data._qdata.numel() * data._qdata.element_size()
        for value in vars(data._params).values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
        return total
    return data.numel() * data.element_size()


def module_bytes(module: nn.Module) -> int:
    total = 0
    for p in module.parameters(recurse=True):
        total += tensor_bytes(p)
    for b in module.buffers(recurse=True):
        total += tensor_bytes(b)
    return total


def _strip_prefix(sd: Dict[str, torch.Tensor], prefixes) -> Dict[str, torch.Tensor]:
    for prefix in prefixes:
        if any(k.startswith(prefix) for k in sd):
            return comfy.utils.state_dict_prefix_replace(sd, {prefix: ""}, filter_keys=True)
    return sd


# --------------------------------------------------------------------------- DiT

def _count_blocks(keys, fmt: str) -> int:
    count = 0
    while any(k.startswith(fmt.format(count)) for k in keys):
        count += 1
    return count


def detect_h3_config(sd: Dict[str, torch.Tensor], metadata: Optional[dict]) -> dict:
    """comfy.model_detection MiniMax H3 branch."""
    keys = set(sd.keys())
    if "video_patch_proj.weight" not in keys or "audio_patch_proj.weight" not in keys:
        raise RuntimeError("This checkpoint is not a MiniMax H3 diffusion model (video/audio patch projections missing).")
    cfg: dict = {"image_model": "minimax_h3"}
    cfg["num_layers"] = _count_blocks(keys, "blocks.{}.")
    cfg["token_refiner_num_layers"] = _count_blocks(keys, "token_refiner.blocks.{}.")
    cfg["hidden_size"] = sd["video_patch_proj.weight"].shape[0]
    pdd_steps = 1
    if metadata is not None and "config" in metadata:
        pdd_steps = json.loads(metadata["config"]).get("transformer", {}).get("pdd_steps", 1)
    cfg["latents_dim"] = sd["final_layer.video_out.weight"].shape[0] // (4 * pdd_steps)
    cfg["audio_latents_dim"] = sd["final_layer.audio_out.weight"].shape[0] // pdd_steps
    cfg["attention_head_dim"] = sd["blocks.0.attn.q_norm.weight"].shape[0]
    qkv = sd["blocks.0.attn.qkv_proj.weight"]
    cfg["num_attention_heads"] = qkv.shape[0] // (3 * cfg["attention_head_dim"])
    cfg["ffn_hidden_size"] = sd["blocks.0.mlp.fc1.weight"].shape[0] // 2
    cfg["text_dim"] = sd["condition_proj.weight"].shape[1]
    if "adaln_t_table" in keys:
        table = sd["adaln_t_table"].shape
        cfg["adaln_curve_grid"] = table[0]
        cfg["time_embed_dim"] = table[1]
    else:
        te = sd["time_embedder.proj_in.weight"]
        cfg["timestep_input_dim"] = te.shape[1]
        cfg["time_embed_hidden_size"] = te.shape[0]
        cfg["time_embed_dim"] = sd["time_embedder.proj_out.weight"].shape[0]
    cfg["rope_inv_freq_len"] = sd["rope.inv_freq"].shape[0]
    cfg["gate_compress"] = "blocks.0.attn.to_gate_compress.weight" in keys
    if metadata is not None and "config" in metadata:
        cfg.update(json.loads(metadata["config"]).get("transformer", {}))
    return cfg


def load_dit(path: str, device: torch.device, emit: Emit = None) -> Tuple[MiniMaxH3Model, dict, torch.dtype, bool]:
    """Load the H3 DiT on the CPU. Returns (model, config, compute_dtype, quantized)."""
    _emit(emit, f"Loading MiniMax H3 DiT: {path}")
    sd, metadata = comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)
    sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)
    sd = _strip_prefix(sd, ("model.diffusion_model.", "diffusion_model."))
    sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)
    config = detect_h3_config(sd, metadata)
    quant_config = comfy.utils.detect_layer_quantization(sd, "")

    supported = [torch.bfloat16, torch.float32]
    params = comfy.utils.calculate_parameters(sd)
    weight_dtype = None if quant_config is not None else comfy.utils.weight_dtype(sd)
    unet_dtype = mm.unet_dtype(model_params=params, supported_dtypes=supported, weight_dtype=weight_dtype)
    manual_cast = mm.unet_manual_cast(None if quant_config is not None else unet_dtype, device, supported)
    compute_dtype = manual_cast if manual_cast is not None else unet_dtype

    kwargs = {k: v for k, v in config.items()}
    if quant_config is not None:
        disabled = comfy.ops.get_disabled_quant_formats(device)
        operations = comfy.ops.mixed_precision_ops(quant_config, compute_dtype, disabled=disabled)
        _emit(emit, "Quantized checkpoint - ComfyUI mixed-precision ops"
              + (f" (emulated: {', '.join(sorted(disabled & set(_quant_formats(sd))))})"
                 if disabled & set(_quant_formats(sd)) else " (native kernels)"))
        model = MiniMaxH3Model(**kwargs, dtype=unet_dtype, device="cpu", operations=operations).eval()
    else:
        operations = comfy.ops.pick_operations(unet_dtype, manual_cast, load_device=device)
        _emit(emit, f"{str(unet_dtype).replace('torch.', '').upper()} checkpoint - standard cast operations")
        with torch.device("meta"):
            model = MiniMaxH3Model(**kwargs, dtype=unet_dtype, device="meta", operations=operations)
        model = model.eval()
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    left = [n for n, p in model.named_parameters() if p is not None and p.device.type == "meta"]
    left += [n for n, b in model.named_buffers() if b is not None and b.device.type == "meta"]
    if left:
        raise RuntimeError(f"H3 DiT load incomplete: {len(left)} tensors missing, e.g. {left[:4]}")
    real_missing = [m for m in missing if not m.endswith(".comfy_quant")]
    if real_missing:
        log.info("H3 DiT: %d missing keys, e.g. %s", len(real_missing), real_missing[:4])
    if unexpected:
        log.info("H3 DiT: %d unexpected keys ignored", len(unexpected))
    model.requires_grad_(False)
    _emit(emit, f"H3 DiT ready: {config['num_layers']} blocks, hidden {config['hidden_size']}, "
                f"{'curve-form AdaLN' if model.use_adaln_curves else 'time embedder'}, compute {str(compute_dtype).replace('torch.', '')}")
    return model, config, compute_dtype, quant_config is not None


def _quant_formats(sd) -> List[str]:
    formats = set()
    for key, value in sd.items():
        if key.endswith(".comfy_quant"):
            try:
                formats.add(json.loads(value.numpy().tobytes()).get("format", ""))
            except Exception:
                pass
    return sorted(f for f in formats if f)


def place_blocks(
    model: nn.Module,
    blocks: List[nn.Module],
    device: torch.device,
    budget_bytes: int,
    emit: Emit = None,
    label: str = "blocks",
    max_resident: Optional[int] = None,
) -> Tuple[int, int]:
    """Keep non-block modules and as many blocks as fit on the GPU; stream the rest.

    Streaming blocks stay in system RAM and their layers cast weights to the
    compute device per call (comfy_cast_weights=True - the flag ComfyUI's lowvram
    loader sets), so the result is identical to a fully resident model.
    max_resident caps the GPU-resident count (block swap).
    """
    block_ids = {id(b) for b in blocks}
    non_block = 0
    for module in _top_level_non_blocks(model, block_ids):
        module.to(device)
        non_block += module_bytes(module)
    budget = int(budget_bytes) - non_block
    resident = 0
    used = 0
    limit = len(blocks) if max_resident is None else max(0, min(len(blocks), int(max_resident)))
    for block in blocks[:limit]:
        size = module_bytes(block)
        if used + size <= budget:
            block.to(device)
            block._open_h3_resident = True
            used += size
            resident += 1
        else:
            break
    for block in blocks[resident:]:
        _stream_block(block)
    if resident < len(blocks):
        _emit(emit, f"VRAM placement: {resident}/{len(blocks)} {label} on GPU, "
                    f"{len(blocks) - resident} streamed from RAM per step")
    else:
        _emit(emit, f"VRAM placement: all {len(blocks)} {label} resident on GPU")
    return resident, len(blocks)


def _stream_block(block: nn.Module) -> None:
    block.to("cpu")
    block._open_h3_resident = False
    for module in block.modules():
        if hasattr(module, "comfy_cast_weights"):
            if not hasattr(module, "_open_h3_cast_before"):
                module._open_h3_cast_before = module.comfy_cast_weights
            module.comfy_cast_weights = True


def stream_more_blocks(blocks: List[nn.Module], need_bytes: int, emit: Emit = None,
                       label: str = "DiT blocks") -> int:
    """Move GPU-resident blocks (last first) to RAM streaming until need_bytes are freed.

    Returns the number of blocks moved (0 when none was resident)."""
    resident = [b for b in blocks if getattr(b, "_open_h3_resident", False)]
    moved = freed = 0
    for block in reversed(resident):
        if freed >= need_bytes and moved:
            break
        freed += module_bytes(block)
        _stream_block(block)
        moved += 1
    if moved:
        mm.soft_empty_cache()
        left = len(resident) - moved
        _emit(emit, f"VRAM placement: {left}/{len(blocks)} {label} on GPU, {len(blocks) - left} streamed from RAM "
                    f"(freed {freed / GIB:.1f} GB)")
    return moved


def restore_blocks(blocks: List[nn.Module], device: torch.device, spare_bytes: int, emit: Emit = None,
                   label: str = "DiT blocks") -> int:
    """Move streamed blocks back onto the GPU while spare_bytes allow (first streamed block first)."""
    moved = used = 0
    for block in blocks:
        if getattr(block, "_open_h3_resident", True):
            continue
        size = module_bytes(block)
        if used + size > spare_bytes:
            break
        block.to(device)
        block._open_h3_resident = True
        for module in block.modules():
            if hasattr(module, "_open_h3_cast_before"):
                module.comfy_cast_weights = module._open_h3_cast_before
        used += size
        moved += 1
    if moved:
        resident = sum(1 for b in blocks if getattr(b, "_open_h3_resident", False))
        _emit(emit, f"VRAM placement: {resident}/{len(blocks)} {label} on GPU "
                    f"({moved} moved back from RAM using {used / GIB:.1f} GB of spare VRAM)")
    return moved


def _top_level_non_blocks(model: nn.Module, block_ids) -> List[nn.Module]:
    """Leaf-most modules/parameters outside the block list (recursively skipping containers of blocks)."""
    out: List[nn.Module] = []

    def walk(module: nn.Module):
        for child in module.children():
            if id(child) in block_ids:
                continue
            if any(id(m) in block_ids for m in child.modules()):
                walk(child)
            else:
                out.append(child)
        # parameters / buffers owned directly by this container
        direct = list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False))
        if direct:
            out.append(_DirectTensors(module))

    walk(model)
    return out


class _DirectTensors:
    """Moves the parameters/buffers a container owns directly (nn.Module.to would move its children too)."""

    def __init__(self, module: nn.Module):
        self.module = module

    def to(self, device):
        for name, p in list(self.module.named_parameters(recurse=False)):
            if p is not None:
                setattr(self.module, name, nn.Parameter(p.data.to(device), requires_grad=False))
        for name, b in list(self.module.named_buffers(recurse=False)):
            if b is not None:
                self.module._buffers[name] = b.to(device)
        return self

    def parameters(self, recurse=True):
        return self.module.parameters(recurse=False)

    def buffers(self, recurse=True):
        return self.module.buffers(recurse=False)


def offload(model: nn.Module) -> None:
    model.to("cpu")
    mm.soft_empty_cache()


# ------------------------------------------------------------------ text encoder

def _llama_detect(sd: Dict[str, torch.Tensor]) -> dict:
    """comfy.text_encoders.hunyuan_video.llama_detect."""
    out = {}
    for key in ("model.norm.weight", "model.layers.0.input_layernorm.weight"):
        if key in sd:
            out["dtype_llama"] = sd[key].dtype
            break
    quant = comfy.utils.detect_layer_quantization(sd, "")
    if quant is not None:
        out["llama_quantization_metadata"] = quant
    return out


class H3TextEncoder:
    """Qwen3-VL-32B (first 50 layers) conditioning encoder, comfy CLIP-wrapper parity."""

    def __init__(self, path: str, device: torch.device, emit: Emit = None):
        import comfy.text_encoders.minimax as minimax_te

        _emit(emit, f"Loading H3 text encoder: {path}")
        sd = comfy.utils.load_torch_file(str(path), safe_load=True)
        if "visual.deepstack_merger_list.0.norm.weight" not in sd or "model.layers.49.self_attn.q_proj.weight" not in sd:
            raise RuntimeError("This file is not the MiniMax H3 text encoder (Qwen3-VL-32B, 50 layers).")
        detect = _llama_detect(sd)
        if "llama_quantization_metadata" in detect:
            _emit(emit, "Quantized text encoder - ComfyUI mixed-precision ops")
        te_class = minimax_te.te(**detect)
        self.tokenizer = minimax_te.MiniMaxH3Tokenizer(tokenizer_data={})
        self.model = te_class(device="cpu", dtype=mm.text_encoder_dtype(device), model_options={})
        self.model.can_assign_sd = True
        missing, unexpected = self.model.load_sd(sd)
        del sd
        real_missing = [m for m in missing if "rotary" not in m]
        if real_missing:
            log.info("Text encoder missing keys (non-fatal): %d", len(real_missing))
        self.device = device

    @property
    def layers(self) -> List[nn.Module]:
        return list(self.model.qwen3vl_32b.transformer.model.layers)

    def place(self, budget_bytes: int, emit: Emit = None) -> None:
        transformer = self.model.qwen3vl_32b.transformer
        layers = list(transformer.model.layers)
        # the vision tower is unused for text prompts: leave it in RAM
        keep = [m for m in (transformer.model,) ]
        place_blocks(transformer.model, layers, self.device, budget_bytes, emit=emit, label="text encoder layers")
        del keep

    def place_vision(self, emit: Emit = None) -> None:
        """Pictures in the prompt (JonXL first frame): keep the vision tower on the GPU (about 1 GB)."""
        visual = getattr(self.model.qwen3vl_32b.transformer, "visual", None)
        if visual is not None:
            visual.to(self.device)
            _emit(emit, f"Text encoder vision tower on {self.device} ({module_bytes(visual) / GIB:.2f} GB)")

    @torch.no_grad()
    def encode(self, text: str, images=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """images: [1, H, W, 3] float pictures in [0, 1], tokenized as '<Picture n>: ' before the text."""
        tokens = self.tokenizer.tokenize_with_weights(text, return_word_ids=False, images=list(images or []))
        self.model.set_clip_options({"execution_device": self.device})
        out = self.model.encode_token_weights(tokens)
        self.model.reset_clip_options()
        cond = out[0]
        extra = out[2] if len(out) > 2 and isinstance(out[2], dict) else {}
        return cond, extra.get("minimax_token_tags")

    def unload(self) -> None:
        self.model.to("cpu")
        mm.soft_empty_cache()


# ----------------------------------------------------------------------- VAEs

class H3VideoVAE:
    """MiniMax H3 video VAE (stock or Sol fast decoder), comfy.sd.VAE parity for this model."""

    def __init__(self, path: str, device: torch.device, emit: Emit = None):
        self._emit = emit
        _emit(emit, f"Loading H3 video VAE: {path}")
        sd, metadata = comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)
        if "blocks.0.scale1" in sd:  # Sol-H3-VAE as released: decoder unprefixed, possibly merged with the encoder
            top = ("encoder", "quant_conv", "post_quant_conv", "latents_mean", "latents_std")
            sd = {k if k.split(".")[0] in top else "decoder." + k: v for k, v in sd.items()}
            sd.setdefault("latents_mean", torch.tensor(LATENTS_MEAN))
            sd.setdefault("latents_std", torch.tensor(LATENTS_STD))
        quant = comfy.utils.detect_layer_quantization(sd, "")
        self.dtype = mm.vae_dtype(device, [torch.float16, torch.float32])
        ops = comfy.ops.disable_weight_init
        if quant is not None:
            ops = comfy.ops.mixed_precision_ops(quant, self.dtype)
        noncausal = "decoder.blocks.0.scale1" in sd
        block_prefix = "decoder.blocks." if noncausal else "decoder.transformer_blocks."
        layers = sum(k.startswith(block_prefix) and k.endswith(".scale1") for k in sd)
        still_frame = None
        if metadata is not None and metadata.get("h3_t1_direct") == "true" and not noncausal:
            still_frame = int(metadata.get("h3_t1_output_slice", 3))
        kept = {(int(k.split(".")[2]), int(k.split(".")[4])) for k in sd
                if k.startswith("encoder.down.") and k.endswith(".conv1.weight")}
        self.model = MiniMaxH3VideoVAE(
            operations=ops, num_layers=layers, still_frame=still_frame, encoder_kept_blocks=kept,
            encoder="encoder.conv_in.weight" in sd, decoder=layers > 0, noncausal_decoder=noncausal)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            log.info("H3 video VAE missing keys: %s", missing[:6])
        self.model = self.model.to(self.dtype).eval().requires_grad_(False)
        self.has_encoder = hasattr(self.model, "encoder")
        self.has_decoder = layers > 0
        self.noncausal = noncausal
        self.device = device
        _emit(emit, f"H3 video VAE ready: {'Sol fast decoder' if noncausal else 'released decoder'} "
                    f"({layers} layers){', encoder' if self.has_encoder else ''}, {str(self.dtype).replace('torch.', '')}")

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames float [T, H, W, 3] in [0, 1] (CPU) -> normalized latent [1, 24, T_lat, H/16, W/16] (CPU fp32)."""
        if not self.has_encoder:
            raise RuntimeError("The selected H3 VAE file has no encoder.")
        self.model.to(self.device)
        pixels = frames.movedim(-1, 0).unsqueeze(0)  # [1, 3, T, H, W]
        pixels = (pixels * 2.0 - 1.0).to(self.dtype)  # comfy.sd.VAE.process_input
        latent = self.model.encode(pixels, device=self.device)
        return latent.float().cpu()

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """normalized latent [1, 24, T, h, w] -> fp16 pixels [1, 3, frames, H, W] in [0, 1] (CPU).

        Finalized frames stream out of VRAM into a half-precision buffer (half the RAM of float32)."""
        if not self.has_decoder:
            raise RuntimeError("The selected H3 VAE file has no decoder.")
        self.model.to(self.device)
        buffer = torch.empty(self.model.decode_output_shape(tuple(latent.shape)), dtype=torch.float16,
                             device=mm.intermediate_device())
        try:
            return self.model.decode(latent.to(self.device, self.dtype), output_buffer=buffer)
        except torch.OutOfMemoryError:
            pass
        # The whole-window decode did not fit the VRAM cap (a 4K window needs about 20.5 GB allocated,
        # 24.6 GB reserved: benchmarks/vram/open_h3_decode.jsonl). Decode overlapping spatial tiles with
        # the whole window's frames each and feather them together, like ComfyUI's tiled VAE decode.
        gc.collect()
        mm.soft_empty_cache()
        frames_of = lambda t: self.model.decode_output_shape((1, 1, int(t), 1, 1))[2]  # noqa: E731
        for tile in (96, 64, 48):
            if tile >= max(latent.shape[-2:]):
                continue
            _emit(self._emit, f"H3 VAE decode did not fit the VRAM limit: decoding {tile * 16}px tiles")
            try:
                pixels = comfy.utils.tiled_scale_multidim(
                    latent, lambda z: self.model.decode(z.to(self.device, self.dtype)),
                    tile=(latent.shape[2], tile, tile), overlap=(0, 8, 8),
                    upscale_amount=(frames_of, 16, 16), index_formulas=(frames_of, 16, 16),
                    out_channels=self.model.decoder.out_channels, output_device=mm.intermediate_device())
                return buffer.copy_(pixels)
            except torch.OutOfMemoryError:
                gc.collect()
                mm.soft_empty_cache()
        raise torch.OutOfMemoryError("H3 VAE decode does not fit even with 768px tiles")

    def offload(self) -> None:
        self.model.to("cpu")
        mm.soft_empty_cache()


class H3AudioVAE:
    def __init__(self, path: str, device: torch.device, emit: Emit = None):
        _emit(emit, f"Loading H3 audio VAE: {path}")
        sd = comfy.utils.load_torch_file(str(path), safe_load=True)
        if "pre_block.attn.zero_k_bias" not in sd:
            raise RuntimeError("This file is not the MiniMax H3 audio VAE.")
        self.model = MiniMaxH3AudioVAE()
        self.model.load_state_dict(sd, strict=False)
        self.model = self.model.float().eval().requires_grad_(False)
        self.sample_rate = 32000
        self.device = device

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform [1, 2, samples] float32 at 32 kHz -> latent [1, 32, 2, T]."""
        self.model.to(self.device)
        out = self.model.encode(waveform.to(self.device, torch.float32))
        self.model.to("cpu")
        return out.float().cpu()


class H3LatentUpscaler:
    def __init__(self, path: str, device: torch.device, emit: Emit = None):
        _emit(emit, f"Loading H3 latent upscaler: {path}")
        sd = comfy.utils.load_torch_file(str(path), safe_load=True)
        if not ("in_blocks.0.emb_layers.1.weight" in sd and "conv_in.weight" in sd and "in_blocks.1.dwconv.weight" in sd):
            raise RuntimeError("This file is not the MiniMax H3 3D-conv latent upscaler.")
        self.dtype = mm.vae_dtype(device, [torch.float16, torch.bfloat16, torch.float32])
        self.model = MiniMaxH3LatentUpscaler(
            in_channels=sd["conv_in.weight"].shape[1],
            channels=sd["conv_in.weight"].shape[0],
            in_blocks=len({k.split(".")[1] for k in sd if k.startswith("in_blocks.") and ".in_layers." in k}),
            out_blocks=len({k.split(".")[1] for k in sd if k.startswith("out_blocks.") and ".in_layers." in k}),
            temporal_kernel=sd["in_blocks.1.dwconv.weight"].shape[2],
            dtype=self.dtype,
            operations=comfy.ops.disable_weight_init,
        )
        self.model.load_state_dict(sd, strict=False)
        self.model = self.model.to(self.dtype).eval().requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def upscale(self, video: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
        """MiniMaxH3LatentUpscaleWithModel: [1, 24, T, h, w] -> [1, 24, T, latent_h, latent_w]."""
        size = (video.shape[2], int(latent_h), int(latent_w))
        if size == tuple(video.shape[2:]):
            return video
        self.model.to(self.device)
        out = self.model(video.to(device=self.device, dtype=self.dtype), size)
        return out.float().cpu()

    def offload(self) -> None:
        self.model.to("cpu")
        mm.soft_empty_cache()
