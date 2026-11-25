"""Download the pinned RelayVSR native 2x checkpoint pair into a model directory."""

import argparse
from pathlib import Path
import sys

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.relayvsr_constants import (
    RELAYVSR_NATIVE_FILE_SIZES,
    RELAYVSR_NATIVE_REPO,
    RELAYVSR_NATIVE_REVISION,
)


def download(model_dir: Path) -> None:
    for filename, size in RELAYVSR_NATIVE_FILE_SIZES.items():
        target = model_dir / filename
        if target.is_file() and target.stat().st_size == size:
            continue
        print(f"Downloading native RelayVSR 2x: {filename}", flush=True)
        downloaded = Path(hf_hub_download(
            RELAYVSR_NATIVE_REPO, filename, revision=RELAYVSR_NATIVE_REVISION,
            local_dir=model_dir, force_download=target.is_file(),
        ))
        if downloaded.stat().st_size != size:
            raise RuntimeError(f"Unexpected checkpoint size for {filename}: {downloaded.stat().st_size} bytes")
    print("Native RelayVSR 2x checkpoints are ready.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    download(parser.parse_args().model_dir.resolve())
