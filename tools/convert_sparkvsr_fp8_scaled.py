from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.sparkvsr_fp8_scaled import (
    SPARKVSR_BF16_MODEL_NAME,
    SPARKVSR_FP8_SCALED_MODEL_NAME,
    ensure_sparkvsr_fp8_scaled_cache,
    summarize_fp8_scaled_cache,
)


DEFAULT_SOURCE = REPO_ROOT / "SparkVSR" / "models" / SPARKVSR_BF16_MODEL_NAME
DEFAULT_OUTPUT = REPO_ROOT / "SparkVSR" / "models" / SPARKVSR_FP8_SCALED_MODEL_NAME


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the SparkVSR FP8-scaled cache from SparkVSR-bf16.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = ensure_sparkvsr_fp8_scaled_cache(
        fp8_model_path=args.output,
        bf16_model_path=args.source,
        force=bool(args.force),
    )
    summary = summarize_fp8_scaled_cache(output)
    for component, info in summary.items():
        print(
            f"{component}: {info['bytes'] / 1024 ** 3:.2f} GiB, "
            f"tensors={info['tensors']}, dtypes={info['dtypes']}",
            flush=True,
        )
    print(f"SparkVSR FP8-scaled cache ready: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
