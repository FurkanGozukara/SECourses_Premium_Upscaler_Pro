"""
Open-H3 Regenerate constants shared by the tab, service, runner and downloader bridge.

Keep the name maps in sync with Models_Downloader.py (one folder above the app).
"""

from __future__ import annotations

# Folder (relative to the app base dir) that holds every Open-H3 checkpoint.
OPENH3_MODELS_DIRNAME = "OpenH3_Models"

# MiniMax H3 FL2VA base (the reference base of the regenerate recipe).
OPENH3_DIT_INT8 = "FL2VA Pruned INT8 ConvRot"
OPENH3_DIT_FP8 = "FL2VA Pruned FP8 Scaled"
OPENH3_DIT_W6A8 = "FL2VA Pruned W6A8"
OPENH3_DIT_W4A8 = "FL2VA Pruned W4A8"
OPENH3_DIT_BF16 = "FL2VA Pruned BF16"
OPENH3_DIT_NAMES = [OPENH3_DIT_INT8, OPENH3_DIT_FP8, OPENH3_DIT_W6A8, OPENH3_DIT_W4A8, OPENH3_DIT_BF16]
OPENH3_DIT_FILES = {
    OPENH3_DIT_INT8: "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    OPENH3_DIT_FP8: "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
    OPENH3_DIT_W6A8: "minimax_h3_fl2va_pruned_w6a8.safetensors",
    OPENH3_DIT_W4A8: "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
    OPENH3_DIT_BF16: "minimax_h3_fl2va_pruned_bf16.safetensors",
}
# W4A8 / W6A8 need compute capability 8.0+ (comfy_kitchen); on older GPUs a run switches to INT8 ConvRot.
OPENH3_DIT_DEFAULT = OPENH3_DIT_W4A8
OPENH3_LOW_BIT_INT8_FALLBACK = {OPENH3_DIT_W4A8: OPENH3_DIT_INT8, OPENH3_DIT_W6A8: OPENH3_DIT_INT8}
OPENH3_DIT_SIZES_GB = {OPENH3_DIT_INT8: 21.0, OPENH3_DIT_FP8: 21.0, OPENH3_DIT_W6A8: 16.0, OPENH3_DIT_W4A8: 12.5,
                       OPENH3_DIT_BF16: 40.2}

# Qwen3-VL-32B conditioning encoder (first 50 layers).
OPENH3_TE_NVFP4 = "Qwen3-VL 32B NVFP4 AWQ"
OPENH3_TE_INT8 = "Qwen3-VL 32B INT8 ConvRot"
OPENH3_TE_BF16 = "Qwen3-VL 32B BF16"
OPENH3_TE_NAMES = [OPENH3_TE_NVFP4, OPENH3_TE_INT8, OPENH3_TE_BF16]
OPENH3_TE_FILES = {
    OPENH3_TE_NVFP4: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    OPENH3_TE_INT8: "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    OPENH3_TE_BF16: "qwen3vl_32b_minimax_h3_bf16.safetensors",
}
OPENH3_TE_SIZES_GB = {OPENH3_TE_NVFP4: 15.7, OPENH3_TE_INT8: 27.1, OPENH3_TE_BF16: 51.5}

# Video VAE: the Sol file holds the H3 encoder plus NVIDIA's fast 24-layer decoder.
OPENH3_VAE_SOL_FP16 = "Sol Fast VAE FP16"
OPENH3_VAE_SOL_INT8 = "Sol Fast VAE INT8 ConvRot"
OPENH3_VAE_RELEASED = "Released H3 VAE FP16"
OPENH3_VAE_NAMES = [OPENH3_VAE_SOL_FP16, OPENH3_VAE_SOL_INT8, OPENH3_VAE_RELEASED]
OPENH3_VAE_FILES = {
    OPENH3_VAE_SOL_FP16: "minimax_h3_sol_vae_fp16.safetensors",
    OPENH3_VAE_SOL_INT8: "minimax_h3_sol_vae_int8_convrot.safetensors",
    OPENH3_VAE_RELEASED: "minimax_h3_video_vae_fp16.safetensors",
}
OPENH3_VAE_SIZES_GB = {OPENH3_VAE_SOL_FP16: 3.6, OPENH3_VAE_SOL_INT8: 2.0, OPENH3_VAE_RELEASED: 5.2}

OPENH3_LORA_FILE = "open_h3_regenerate_2k_lora.safetensors"
OPENH3_CURVES_FILE = "sharp_curve_branches.safetensors"
OPENH3_UPSCALER_FILE = "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
OPENH3_AUDIO_VAE_FILE = "minimax_h3_audio_vae_fp32.safetensors"
OPENH3_TABLE_FILE = "open_h3_reference_adaln_t_table.safetensors"
OPENH3_ALWAYS_FILES = (OPENH3_LORA_FILE, OPENH3_CURVES_FILE, OPENH3_UPSCALER_FILE, OPENH3_AUDIO_VAE_FILE)
OPENH3_ALWAYS_SIZE_GB = 12.5 + 1.2 + 0.7 + 0.6

# Recipe defaults (NVIDIA Open-H3-Regenerate-2K release).
OPENH3_DEFAULT_SEED = 1766580932
OPENH3_RESOLUTIONS = {
    "768p": (1344, 768),
    "1080p": (1920, 1080),
    "2k": (2560, 1440),
    "4k": (3840, 2160),
}


def openh3_download_size_gb(dit: str, text_encoder: str, vae: str) -> float:
    return (
        OPENH3_DIT_SIZES_GB.get(dit, 21.0)
        + OPENH3_TE_SIZES_GB.get(text_encoder, 15.7)
        + OPENH3_VAE_SIZES_GB.get(vae, 3.6)
        + OPENH3_ALWAYS_SIZE_GB
    )


# JonXL H3 Upscale: JonXL's MiniMax H3 image upscaler (LoRA + workflow), the second Open-H3 workflow. Every picture
# (an image, or every frame of a video) becomes the last frame of a 24-frame H3 clip that cross-fades from the blurry
# input to a sharp picture (OpenH3/open_h3_engine/jxl.py). Needs only the DiT, the text encoder, the VAE and this LoRA.
OPENH3_WORKFLOW_REGEN = "Open-H3 Regenerate (default)"
OPENH3_WORKFLOW_JXL = "JonXL H3 Upscale"
OPENH3_WORKFLOWS = [OPENH3_WORKFLOW_REGEN, OPENH3_WORKFLOW_JXL]
OPENH3_JXL_LORA_FILE = "h3_upscale_1920.safetensors"
OPENH3_JXL_LORA_SIZE_GB = 0.6
OPENH3_JXL_PROMPT = "A static still blurry pixelated picture crossfades into a clear sharp detailed picture, 4psc4l3,"
OPENH3_JXL_SEED = 505817956432961   # the workflow's fixed seed (used while Seed holds the Open-H3 default)
OPENH3_JXL_STEPS = 20
OPENH3_JXL_MAX_SIDES = [1920, 1536, 1280]
OPENH3_JXL_MAX_SIDE_DEFAULT = 1920


def is_jxl_workflow(settings) -> bool:
    value = str((settings or {}).get("workflow") or "").strip().lower()
    return value.startswith("jonxl") or value == "jxl"


def jxl_output_size(src_w: int, src_h: int, max_side: int = OPENH3_JXL_MAX_SIDE_DEFAULT):
    """Output size of a JonXL upscale: the picture fitted under max_side x max_side at its exact aspect, even sides
    (mirrors open_h3_engine.jxl.plan_geometry 'keep'; the model renders a 32-pixel padded canvas around it)."""
    import math

    src_w, src_h, max_side = max(1, int(src_w)), max(1, int(src_h)), int(max_side)
    total = (max_side * max_side / 1e6) * 1024 * 1024
    scale = min(math.sqrt(total / (src_w * src_h)), max_side / src_w, max_side / src_h)
    out_w, out_h = max(2, int(round(src_w * scale / 2)) * 2), max(2, int(round(src_h * scale / 2)) * 2)
    return min(out_w, max_side - max_side % 2), min(out_h, max_side - max_side % 2)


def openh3_jxl_download_size_gb(dit: str, text_encoder: str, vae: str) -> float:
    return (OPENH3_DIT_SIZES_GB.get(dit, 21.0) + OPENH3_TE_SIZES_GB.get(text_encoder, 15.7)
            + OPENH3_VAE_SIZES_GB.get(vae, 3.6) + OPENH3_JXL_LORA_SIZE_GB)
