"""
Build a self-contained SparkVSR BF16 distribution folder.

The official SparkVSR checkpoint is stored as FP32 safetensor shards. This tool
keeps the source folder untouched and writes a new Diffusers model folder with:
  - text_encoder/model.safetensors in BF16
  - transformer/diffusion_pytorch_model.safetensors in BF16
  - copied tokenizer/scheduler/VAE/config files

The output avoids shard index files so Diffusers/Transformers load the single
safetensors files directly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from safetensors import safe_open
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "SparkVSR" / "models" / "SparkVSR"
DEFAULT_OUTPUT = REPO_ROOT / "SparkVSR" / "models" / "SparkVSR-bf16"


def _format_bytes(value: int) -> str:
    return f"{value / 1024 ** 3:.2f} GiB"


def _copy_file(src: Path, dst: Path, *, force: bool) -> None:
    if dst.exists() and not force:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path, *, force: bool) -> None:
    if not src.exists():
        return
    if dst.exists() and force:
        shutil.rmtree(dst)
    if not dst.exists():
        shutil.copytree(src, dst)


def _copy_root_files(source: Path, output: Path, *, force: bool) -> None:
    for name in ("model_index.json", "LICENSE", "README.md", "README_zh.md"):
        src = source / name
        if src.exists():
            _copy_file(src, output / name, force=force)
    for name in ("assets", "scheduler", "tokenizer", "vae"):
        _copy_tree(source / name, output / name, force=force)


def _copy_component_config(source: Path, output: Path, component: str, *, force: bool) -> None:
    src = source / component / "config.json"
    if src.exists():
        _copy_file(src, output / component / "config.json", force=force)


def _component_files(source: Path, component: str) -> List[Path]:
    return sorted((source / component).glob("*.safetensors"), key=lambda p: p.name)


def _convert_component(
    source: Path,
    output: Path,
    *,
    component: str,
    output_filename: str,
    force: bool,
) -> Path:
    started = time.monotonic()
    out_path = output / component / output_filename
    if out_path.exists() and not force:
        print(f"[skip] {component}: {out_path} already exists ({_format_bytes(out_path.stat().st_size)})", flush=True)
        return out_path

    files = _component_files(source, component)
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {source / component}")

    _copy_component_config(source, output, component, force=force)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    state: Dict[str, torch.Tensor] = {}
    source_bytes = sum(p.stat().st_size for p in files)
    print(f"[convert] {component}: {len(files)} source file(s), {_format_bytes(source_bytes)} FP32 on disk", flush=True)
    for file_index, path in enumerate(files, 1):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            for key_index, key in enumerate(keys, 1):
                tensor = handle.get_tensor(key)
                state[key] = tensor.to(dtype=torch.bfloat16, device="cpu").contiguous()
                del tensor
                if key_index % 100 == 0 or key_index == len(keys):
                    print(
                        f"[convert] {component}: {file_index}/{len(files)} {path.name}, "
                        f"{key_index}/{len(keys)} tensors",
                        flush=True,
                    )

    metadata = {
        "format": "pt",
        "source": str(source.resolve()),
        "component": component,
        "converted_dtype": "bfloat16",
    }
    print(f"[save] {component}: writing {out_path}", flush=True)
    save_file(state, str(out_path), metadata=metadata)
    del state

    for index_name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        stale = out_path.parent / index_name
        if stale.exists():
            stale.unlink()

    elapsed = time.monotonic() - started
    print(f"[done] {component}: {_format_bytes(out_path.stat().st_size)} in {elapsed:.1f}s", flush=True)
    return out_path


def _summarize_component(path: Path) -> None:
    dtype_counts: Dict[str, int] = {}
    tensor_count = 0
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor_count += 1
            dtype = str(handle.get_slice(key).get_dtype())
            shape = handle.get_slice(key).get_shape()
            params = 1
            for dim in shape:
                params *= int(dim)
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + params
    print(f"[verify] {path}: {tensor_count} tensors, dtypes={json.dumps(dtype_counts, sort_keys=True)}", flush=True)


def _verify_outputs(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Expected output missing: {path}")
        _summarize_component(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert SparkVSR text encoder and transformer shards to single BF16 safetensors.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--component",
        choices=["all", "text_encoder", "transformer"],
        default="all",
        help="Convert one component or both. Root files are copied for every run.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not (source / "model_index.json").exists():
        raise FileNotFoundError(f"SparkVSR source model not found: {source}")

    output.mkdir(parents=True, exist_ok=True)
    _copy_root_files(source, output, force=bool(args.force))

    outputs: List[Path] = []
    if args.component in {"all", "text_encoder"}:
        outputs.append(
            _convert_component(
                source,
                output,
                component="text_encoder",
                output_filename="model.safetensors",
                force=bool(args.force),
            )
        )
    if args.component in {"all", "transformer"}:
        outputs.append(
            _convert_component(
                source,
                output,
                component="transformer",
                output_filename="diffusion_pytorch_model.safetensors",
                force=bool(args.force),
            )
        )

    _verify_outputs(outputs)
    print(f"[complete] SparkVSR BF16 model folder: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
