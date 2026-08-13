"""
Static replacement for ComfyUI's comfy/cli_args.py.

The vendored ComfyUI modules in this package read launch flags from
``comfy.cli_args.args``.  The standalone LTX 2.5 engine does not parse
ComfyUI's command line; instead the inference CLI calls :func:`configure`
before the heavy modules are imported.  Defaults mirror the ComfyUI
launchers shipped with SECourses (sage attention optional, triton backend
enabled, pinned memory disabled on Windows).
"""

from __future__ import annotations

import enum
import os
import types


class PerformanceFeature(enum.Enum):
    Fp16Accumulation = "fp16_accumulation"
    Fp8MatrixMultiplication = "fp8_matrix_mult"
    CublasOps = "cublas_ops"
    AutoTune = "autotune"


class LatentPreviewMethod(enum.Enum):
    NoPreviews = "none"
    Auto = "auto"
    Latent2RGB = "latent2rgb"
    TAESD = "taesd"


def _default_args() -> types.SimpleNamespace:
    ns = types.SimpleNamespace(
        # devices / precision
        cpu=False,
        cuda_device=None,
        default_device=None,
        directml=None,
        cuda_malloc=False,
        disable_cuda_malloc=True,
        force_fp32=False,
        force_fp16=False,
        fp32_unet=False,
        fp64_unet=False,
        bf16_unet=False,
        fp16_unet=False,
        fp8_e4m3fn_unet=False,
        fp8_e5m2_unet=False,
        fp8_e8m0fnu_unet=False,
        fp16_vae=False,
        fp32_vae=False,
        bf16_vae=False,
        cpu_vae=False,
        fp8_e4m3fn_text_enc=False,
        fp8_e5m2_text_enc=False,
        fp16_text_enc=False,
        fp32_text_enc=False,
        bf16_text_enc=False,
        fp16_intermediates=False,
        force_channels_last=False,
        supports_fp8_compute=False,
        # attention
        use_split_cross_attention=False,
        use_quad_cross_attention=False,
        use_pytorch_cross_attention=False,
        use_sage_attention=False,
        use_flash_attention=False,
        use_ck_attention=False,
        disable_xformers=True,
        force_upcast_attention=False,
        dont_upcast_attention=False,
        # vram policy
        gpu_only=False,
        highvram=False,
        normalvram=False,
        lowvram=False,
        novram=False,
        reserve_vram=None,
        async_offload=None,
        disable_async_offload=False,
        fast_disk=False,
        force_non_blocking=False,
        disable_smart_memory=False,
        deterministic=False,
        high_ram=False,
        # kernels / misc
        enable_triton_backend=True,
        disable_triton_backend=False,
        fast=set(),
        disable_pinned_memory=(os.name == "nt"),
        mmap_torch_files=False,
        disable_mmap=False,
        in_training=False,
    )
    return ns


args = _default_args()


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def configure(**overrides: object) -> None:
    """Apply engine settings onto the static args namespace."""
    for key, value in overrides.items():
        setattr(args, key, value)


def configure_from_env() -> None:
    """Read SECOURSES_LTX25_* environment overrides (set by the runner)."""
    env = os.environ
    attention = env.get("SECOURSES_LTX25_ATTENTION", "auto").strip().lower()
    if attention in {"sage", "sageattention"}:
        args.use_sage_attention = True
    elif attention in {"flash", "flash_attn", "flashattention"}:
        args.use_flash_attention = True
    elif attention in {"sdpa", "pytorch", "torch"}:
        args.use_pytorch_cross_attention = True
    elif attention == "auto":
        try:
            import sageattention  # noqa: F401
            args.use_sage_attention = True
        except ImportError:
            args.use_pytorch_cross_attention = True
    if "SECOURSES_LTX25_RESERVE_VRAM_GB" in env:
        try:
            args.reserve_vram = float(env["SECOURSES_LTX25_RESERVE_VRAM_GB"])
        except ValueError:
            pass
    vram_mode = env.get("SECOURSES_LTX25_VRAM_MODE", "auto").strip().lower()
    if vram_mode == "gpu_only":
        args.gpu_only = True
    elif vram_mode == "highvram":
        args.highvram = True
    elif vram_mode == "lowvram":
        args.lowvram = True
    elif vram_mode == "novram":
        args.novram = True
    if "SECOURSES_LTX25_PINNED_MEMORY" in env:
        args.disable_pinned_memory = not _as_bool(env["SECOURSES_LTX25_PINNED_MEMORY"])
    if "SECOURSES_LTX25_TRITON" in env:
        enabled = _as_bool(env["SECOURSES_LTX25_TRITON"])
        args.enable_triton_backend = enabled
        args.disable_triton_backend = not enabled
    if _as_bool(env.get("SECOURSES_LTX25_ASYNC_OFFLOAD", "1")):
        args.async_offload = None  # auto (on for NVIDIA)
    else:
        args.disable_async_offload = True
