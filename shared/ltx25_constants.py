"""
LTX 2.5 constants shared by the tab, service, runner, meta registry and downloader bridge.

Model naming: "<base> <quant>" where base is Distilled (8-step fast) or Dev
(20-step quality) and quant is one of INT8 ConvRot / INT4 W4A8 / NVFP4 / BF16.
The names are stored in presets, so they must remain stable.
"""

# Folder (relative to the app base dir) that holds every LTX 2.5 checkpoint.
LTX25_MODELS_DIRNAME = "LTX25_Models"

# Transformer variants (dropdown values).
LTX25_DISTILLED_INT8 = "Distilled INT8 ConvRot"
LTX25_DISTILLED_NVFP4 = "Distilled NVFP4"
LTX25_DISTILLED_INT4 = "Distilled INT4 W4A8 ConvRot"
LTX25_DISTILLED_BF16 = "Distilled BF16"
LTX25_DEV_INT8 = "Dev INT8 ConvRot"
LTX25_DEV_INT4 = "Dev INT4 W4A8 ConvRot"
LTX25_DEV_BF16 = "Dev BF16"

LTX25_MODEL_NAMES = [
    LTX25_DISTILLED_INT8,
    LTX25_DISTILLED_NVFP4,
    LTX25_DISTILLED_INT4,
    LTX25_DISTILLED_BF16,
    LTX25_DEV_INT8,
    LTX25_DEV_INT4,
    LTX25_DEV_BF16,
]

# Transformer checkpoint file for each variant.
LTX25_TRANSFORMER_FILES = {
    LTX25_DISTILLED_INT8: "LTX-2.5-22b-Distilled-Transformer-Int8-ConvRot-Premium-SECourses.safetensors",
    LTX25_DISTILLED_NVFP4: "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
    LTX25_DISTILLED_INT4: "ltx-2.5-22b-distilled-transformer_W4A8_Mixed.safetensors",
    LTX25_DISTILLED_BF16: "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    LTX25_DEV_INT8: "LTX-2.5-22b-Dev-Transformer-Int8-ConvRot-Premium-SECourses.safetensors",
    LTX25_DEV_INT4: "ltx-2.5-22b-dev-transformer_W4A8_Mixed.safetensors",
    LTX25_DEV_BF16: "ltx-2.5-22b-dev-transformer-bf16.safetensors",
}

# Text encoder variants (dropdown values).
LTX25_TE_INT8 = "Gemma 4 12B INT8 ConvRot"
LTX25_TE_BF16 = "Gemma 4 12B BF16"
LTX25_TE_NAMES = [LTX25_TE_INT8, LTX25_TE_BF16]
LTX25_TE_FILES = {
    LTX25_TE_INT8: "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    LTX25_TE_BF16: "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
}

# Video VAE variants (dropdown values).
LTX25_VAE_CONV = "Video VAE Conv"
LTX25_VAE_REGULAR = "Video VAE Regular"
LTX25_VAE_NAMES = [LTX25_VAE_CONV, LTX25_VAE_REGULAR]
LTX25_VAE_FILES = {
    LTX25_VAE_CONV: "ltx-2.5-video-vae-conv-bf16.safetensors",
    LTX25_VAE_REGULAR: "ltx-2.5-video-vae-bf16.safetensors",
}

# Always-required auxiliary files.
LTX25_IC_LORA_FILE = "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
LTX25_AUDIO_VAE_FILE = "ltx-2.5-audio-vae-bf16.safetensors"
LTX25_ALWAYS_FILES_TUPLE = (LTX25_AUDIO_VAE_FILE, LTX25_IC_LORA_FILE)

# Optional extras kept in the same folder for power users / future pipelines.
LTX25_EXTRA_FILES = (
    "ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
    "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
    "ltx-2.5-duration-head-bf16.safetensors",
)

# Distilled variants run the 8-step distilled sigma schedule by default;
# Dev variants default to the 20-step quality schedule from the Auto
# Resolution preset (cfg 3).
LTX25_DISTILLED_BASES = {
    LTX25_DISTILLED_INT8,
    LTX25_DISTILLED_NVFP4,
    LTX25_DISTILLED_INT4,
    LTX25_DISTILLED_BF16,
}


def ltx25_is_distilled(model_name: str) -> bool:
    return str(model_name or "") in LTX25_DISTILLED_BASES
