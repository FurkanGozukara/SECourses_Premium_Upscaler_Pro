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

# Same-size (reference downscale 1) IC-LoRAs, downloaded only when a pass uses them.
LTX25_REFINE_DETAILS_LORA_FILE = "ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors"
LTX25_RESTORE_LORA_FILE = "ltx-2.5-22b-ic-lora-restore-1.0.safetensors"
LTX25_DECOMPRESSION_LORA_FILE = "ltx-2.5-22b-ic-lora-decompression-0.9.safetensors"
LTX25_DEBLUR_LORA_FILE = "ltx-2.5-22b-ic-lora-deblur-0.9.safetensors"

# Optional pre-clean pass (same size, before the x2 upscale) and its trained prompt convention.
LTX25_PRECLEAN_OFF = "Off"
LTX25_PRECLEAN_LORAS = {
    "Decompression 0.9": LTX25_DECOMPRESSION_LORA_FILE,
    "Deblur 0.9": LTX25_DEBLUR_LORA_FILE,
    "Restore 1.0": LTX25_RESTORE_LORA_FILE,
}
LTX25_PRECLEAN_NAMES = [LTX25_PRECLEAN_OFF, *LTX25_PRECLEAN_LORAS]
LTX25_PRECLEAN_PROMPTS = {
    "Decompression 0.9": (
        "Reference shows the scene, heavily compressed with visible macroblocking, chroma bleed, and ringing "
        "artifacts. Edited shows the same scene restored to high quality with sharp detail, clean edges, and no "
        "compression artifacts. ENHANCE QUALITY the scene. Subject identity, framing, and background geometry are "
        "identical to the reference; only compression artifacts and image quality differ between reference and edited."
    ),
    "Deblur 0.9": (
        "Reference shows the scene, heavily out of focus with soft defocused blur and no fine detail. Edited shows "
        "the same scene in sharp focus with crisp detail and clean edges. DEBLUR the scene Subject identity, framing, "
        "and background geometry are identical to the reference; only focus and sharpness differ between reference "
        "and edited."
    ),
    "Restore 1.0": (
        "natural colour, sharp photographic detail, crisp faces and clothing texture, natural grain, "
        "high resolution footage"
    ),
}

# Refine mode: a same-size IC-LoRA on a lanczos 2x canvas through spatial tiling (tiled fusion), with the
# window each LoRA was trained on.
LTX25_REFINE_LORAS = {
    "Refine Details 1.0": (LTX25_REFINE_DETAILS_LORA_FILE, 1024, 576),
    "Restore 1.0": (LTX25_RESTORE_LORA_FILE, 960, 544),
}
LTX25_MODE_X2 = "Pixel Spatial x2"
# Offered in the tab: Restore (tested 2026-11-01 on archive-style footage). Refine Details is wired but its
# gated repository has not been accepted on this account, so it is not offered until it is mirrored and tested.
LTX25_MODE_NAMES = [LTX25_MODE_X2, "Refine: Restore 1.0"]
LTX25_REFINE_PROMPTS = {
    "Refine Details 1.0": "sharp photographic detail, crisp natural texture, fine surface detail, clean edges, "
                          "natural film grain, high resolution footage",
    "Restore 1.0": "natural colour, sharp photographic detail, crisp faces and clothing texture, natural grain, "
                   "high resolution footage",
}
LTX25_REFINE_NEGATIVE = "blurry, soft, plastic, smeared detail, oversharpened halos, warped faces"

# Size and sha256 of every downloadable LTX 2.5 file as released. A local file whose size differs, or whose
# verified sha256 is not this one, is checked against the repository and replaced when it is stale (the
# pre-2026-08-16 NVFP4 without keyframes_abs_pos_embedding; the Aug 13 Gemma 4 INT8 text encoder).
LTX25_FILE_SHA256 = {
    "LTX-2.5-22b-Distilled-Transformer-Int8-ConvRot-Premium-SECourses.safetensors":
        (21939612288, "a379237e52ddcfd49e623f92246f8d61251e67dcc6c804b9fed0520a1a928edb"),
    "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors":
        (18721548408, "4b94231e734c1950f8f6826cb8bd8715d94be5b3e04f8256ee060c5bc3886c30"),
    "ltx-2.5-22b-distilled-transformer_W4A8_Mixed.safetensors":
        (15412311232, "fce4041034c3b6dde4bbe36d527d5c56a547e3ccb5a5bd2fe4fdf08c8e536c44"),
    "ltx-2.5-22b-distilled-transformer-bf16.safetensors":
        (42018190584, "31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4"),
    "LTX-2.5-22b-Dev-Transformer-Int8-ConvRot-Premium-SECourses.safetensors":
        (21956355240, "277d64a759cd935a960765158cdcd5102ac7cc83c4f87f87fd719c52c97ccc29"),
    "ltx-2.5-22b-dev-transformer_W4A8_Mixed.safetensors":
        (15412311232, "ecba8d93dcf5946961bdaa9b4f1b97552380bd0349293742d72bba72c2ceae78"),
    "ltx-2.5-22b-dev-transformer-bf16.safetensors":
        (42018190584, "792a2bad501ca03262c0bc2ce7a2949e85b142ce18e30894aad5bc849c8e7584"),
    "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors":
        (15372969374, "6ce688a0aa98a5fa36a9f1e6c3f42152a498cc2b53ee8c15674c64244f91487f"),
    "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors":
        (26263858182, "ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1"),
    "ltx-2.5-video-vae-conv-bf16.safetensors":
        (1452269922, "685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d"),
    "ltx-2.5-video-vae-bf16.safetensors":
        (1472223346, "847e14ca7f3355debca0cea4eaa24ac0fbcdf0061da054ac89ca638a869ddba3"),
    "ltx-2.5-audio-vae-bf16.safetensors":
        (364866540, "c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5"),
    "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors":
        (327322640, "984851b769ea2bcb4c9e0a239a7676239e42c6a6001ddc69943b41ff0b283c1d"),
    LTX25_REFINE_DETAILS_LORA_FILE:
        (1308787534, "771a84f70e143af89867fc714ebbf7bd6edcaa4974c0360cba4a2d292a99f0e6"),
    LTX25_RESTORE_LORA_FILE:
        (1711643914, "357728bc704b3f56e59c5f09cf0540ef3b3fb88a93bbd0b050dbfacd5e05c8b0"),
    LTX25_DECOMPRESSION_LORA_FILE:
        (906080571, "dd42b8fb53e2fb004bc36479ce4f1ea977ec5ba2160aba51e0fbfdcce00db0b1"),
    LTX25_DEBLUR_LORA_FILE:
        (906080571, "62dce32f671429c0ca98fdba997cabb9925fcfe79bbf788c5aa62a3a4c4a5a93"),
}

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
