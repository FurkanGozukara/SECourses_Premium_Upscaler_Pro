"""RelayVSR model names, files and sizes (app side; the downloader has its own copy of the file list)."""

RELAYVSR_MODELS_DIRNAME = "RelayVSR_Models"
RELAYVSR_MIRROR_SUBDIR = "RelayVSR"                     # MonsterMMORPG/Wan_GGUF RelayVSR/<file>

RELAYVSR_WAN_INT8 = "Wan INT8 ConvRot (5 GB)"
RELAYVSR_WAN_BF16 = "Wan BF16 (10 GB)"
RELAYVSR_WAN_NAMES = [RELAYVSR_WAN_INT8, RELAYVSR_WAN_BF16]
RELAYVSR_WAN_DEFAULT = RELAYVSR_WAN_INT8
RELAYVSR_WAN_FILES = {
    RELAYVSR_WAN_INT8: "relayvsr_wan_int8_convrot.safetensors",
    RELAYVSR_WAN_BF16: "relayvsr_wan_bf16.safetensors",
}
RELAYVSR_FLASH_FILE = "relayvsr_flash.safetensors"
RELAYVSR_WAN_SIZES_GB = {RELAYVSR_WAN_INT8: 5.1, RELAYVSR_WAN_BF16: 10.1}
RELAYVSR_FLASH_SIZE_GB = 0.13

RELAYVSR_SCALE = 4                                      # the model always upscales 4x
RELAYVSR_WINDOW_DEFAULT = 16                            # one Wan keyframe every 15 frames (upstream default)
RELAYVSR_WINDOW_MIN, RELAYVSR_WINDOW_MAX = 2, 64
RELAYVSR_PRECISIONS = ["auto", "bf16", "fp16"]


def relayvsr_download_size_gb(wan_name: str) -> float:
    return RELAYVSR_WAN_SIZES_GB.get(wan_name, 5.1) + RELAYVSR_FLASH_SIZE_GB
