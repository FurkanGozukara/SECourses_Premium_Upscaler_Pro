"""RelayVSR model names, files and sizes (app side; the downloader has its own copy of the file list)."""

RELAYVSR_MODELS_DIRNAME = "RelayVSR_Models"
RELAYVSR_MIRROR_SUBDIR = "RelayVSR"                     # MonsterMMORPG/Wan_GGUF RelayVSR/<file>

RELAYVSR_WAN_INT8 = "Wan INT8 ConvRot (5 GB)"
RELAYVSR_WAN_BF16 = "Wan BF16 (10 GB)"
RELAYVSR_WAN_2X = "Native 2x BF16 (10 GB)"
RELAYVSR_WAN_NAMES = [RELAYVSR_WAN_INT8, RELAYVSR_WAN_BF16, RELAYVSR_WAN_2X]
RELAYVSR_WAN_DEFAULT = RELAYVSR_WAN_INT8
RELAYVSR_WAN_FILES = {
    RELAYVSR_WAN_INT8: "relayvsr_wan_int8_convrot.safetensors",
    RELAYVSR_WAN_BF16: "relayvsr_wan_bf16.safetensors",
    RELAYVSR_WAN_2X: "wan_x2.safetensors",
}
RELAYVSR_FLASH_FILE = "relayvsr_flash.safetensors"
RELAYVSR_FLASH_2X_FILE = "flash_x2.safetensors"
RELAYVSR_NATIVE_REPO = "lightx2v/RelayVSR_X2"
RELAYVSR_NATIVE_REVISION = "bdaba6f33b6ec0b7f8a7d0c32c871e2539d1193b"
RELAYVSR_NATIVE_FILE_SIZES = {
    RELAYVSR_WAN_FILES[RELAYVSR_WAN_2X]: 10127172392,
    RELAYVSR_FLASH_2X_FILE: 127759464,
}
RELAYVSR_WAN_SIZES_GB = {RELAYVSR_WAN_INT8: 5.1, RELAYVSR_WAN_BF16: 10.1, RELAYVSR_WAN_2X: 10.1}
RELAYVSR_FLASH_SIZE_GB = 0.13

RELAYVSR_SCALE = 4                                      # original checkpoint scale
RELAYVSR_WINDOW_DEFAULT = 16                            # one Wan keyframe every 15 frames (upstream default)
RELAYVSR_WINDOW_MIN, RELAYVSR_WINDOW_MAX = 2, 64
RELAYVSR_PRECISIONS = ["auto", "bf16", "fp16"]


def relayvsr_model_scale(wan_name: str) -> int:
    return 2 if wan_name == RELAYVSR_WAN_2X else RELAYVSR_SCALE


def relayvsr_flash_file(wan_name: str) -> str:
    return RELAYVSR_FLASH_2X_FILE if wan_name == RELAYVSR_WAN_2X else RELAYVSR_FLASH_FILE


def relayvsr_download_size_gb(wan_name: str) -> float:
    return RELAYVSR_WAN_SIZES_GB.get(wan_name, 5.1) + RELAYVSR_FLASH_SIZE_GB
