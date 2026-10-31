"""
Torch-free GAN architecture detection for the GAN tab.

The tab must not import torch/spandrel at startup, so it reads only the tensor
names of a model file (the safetensors JSON header, or the pickle of a .pth with
every class stubbed out, which never runs code from the file) and matches them
against spandrel 0.4.2's own key conditions, in spandrel's registry order.
The runner still loads the model with spandrel; this is only the label.
"""

from __future__ import annotations

import collections
import json
import pickle
import struct
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

# (spandrel architecture name, alternatives). An alternative matches when all of its
# keys are present; a key given as a tuple matches when any of its names is present.
_ARCH_CONDITIONS = (
    ("RealESRGAN Compact", [("body.0.weight", "body.1.weight")]),
    ("Swift-SRGAN", [("initial.cnn.depthwise.weight", "final_conv.pointwise.weight")]),
    ("HAT", [(
        "relative_position_index_SA",
        "conv_first.weight",
        "layers.0.residual_group.blocks.0.conv_block.cab.0.weight",
        "layers.0.residual_group.overlap_attn.qkv.weight",
    )]),
    ("GRL", [
        (
            f"{prefix}conv_first.weight",
            f"{prefix}norm_start.weight",
            f"{prefix}norm_end.weight",
            f"{prefix}layers.0.blocks.0.attn.window_attn.attn_transform.logit_scale",
        )
        for prefix in ("", "model.", "model_g.")
    ]),
    ("Swin2SR", [(
        "layers.0.residual_group.blocks.0.norm1.weight",
        "patch_embed.proj.weight",
        "conv_first.weight",
        "layers.0.residual_group.blocks.0.mlp.fc1.bias",
        "layers.0.residual_group.blocks.0.attn.relative_position_index",
    )]),
    ("SwinIR", [(
        "layers.0.residual_group.blocks.0.norm1.weight",
        ("conv_first.weight", "conv_first.1.weight"),
        "layers.0.residual_group.blocks.0.mlp.fc1.bias",
        "layers.0.residual_group.blocks.0.attn.relative_position_index",
    )]),
    ("OmniSR", [("residual_layer.0.residual_layer.0.layer.0.fn.0.weight", "input.weight", "up.0.weight")]),
    ("RGT", [(
        "conv_first.weight",
        "before_RG.1.weight",
        "layers.0.blocks.0.gamma",
        "layers.0.blocks.0.attn.attns.0.rpe_biases",
    )]),
    ("DAT", [(
        "conv_first.weight",
        "before_RG.1.weight",
        "layers.0.blocks.0.ffn.sg.conv.weight",
        "layers.0.blocks.0.attn.qkv.weight",
    )]),
    ("CRAFT", [("conv_first.weight", "layers.0.residual_group.hf_blocks.0.attn.temperature")]),
    ("SPAN", [("conv_1.sk.weight", "block_1.c1_r.sk.weight", "conv_cat.weight", "upsampler.0.weight")]),
    ("RealCUGAN", [("unet1.conv1.conv.0.weight", "unet2.conv1.conv.0.weight", "unet2.conv_bottom.weight")]),
    ("SAFMN", [("to_feat.weight", "feats.0.safm.mfr.0.weight", "to_img.0.weight")]),
    ("ATD", [(
        "relative_position_index_SA",
        "conv_first.weight",
        "layers.0.residual_group.td",
        "layers.0.residual_group.layers.0.wqkv.weight",
    )]),
    ("DRCT", [("conv_first.weight", "layers.0.swin1.attn.qkv.weight", "layers.0.adjust1.weight")]),
    ("ESRGAN", [
        ("model.0.weight", "model.1.sub.0.RDB1.conv1.0.weight"),
        ("conv_first.weight", "body.0.rdb1.conv1.weight", "conv_body.weight", "conv_last.weight"),
        ("conv_first.weight", "RRDB_trunk.0.RDB1.conv1.weight", "trunk_conv.weight", "conv_last.weight"),
        ("model.0.weight", "model.1.sub.0.RDB1.conv1x1.weight"),
    ]),
    ("PLKSR", [(
        "feats.0.weight",
        ("feats.1.lk.conv.weight", "feats.1.lk.convs.0.weight", "feats.1.lk.mn_conv.weight"),
        "feats.1.refine.weight",
        ("feats.1.channe_mixer.0.weight", "feats.1.channel_mixer.0.weight"),
    )]),
    ("MoSR", [("gblocks.0.weight", "gblocks.1.norm.weight", "gblocks.1.fc1.weight", "gblocks.1.conv.weight")]),
    ("RCAN", [(
        ("head.0.weight", "head.1.weight"),
        "tail.1.weight",
        "body.0.body.0.body.0.weight",
        "body.0.body.0.body.3.conv_du.0.weight",
    )]),
    ("FDAT", [(
        ("conv_first.weight", "conv_first.1.weight"),
        "groups.0.blocks.0.attn.bias",
        "groups.0.blocks.0.inter.cg.1.weight",
        "upsampler.MetaUpsample",
    )]),
)

# Same unwrapping as spandrel's canonicalize_state_dict.
_UNWRAP_KEYS = ("model_state_dict", "state_dict", "params_ema", "params-ema", "params", "model", "net")
_COMMON_PREFIXES = ("module.", "netG.")

_ARCH_CACHE: Dict[Tuple[str, int, int], str] = {}


class _Stub:
    """Stands in for every class a .pth pickle references (tensors, storages, ...)."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __setitem__(self, key, value):
        pass

    def append(self, value):
        pass

    def extend(self, values):
        pass


class _KeysUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        return _Stub

    def persistent_load(self, pid):
        return None


def _safetensors_keys(path: Path) -> Set[str]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        if header_len <= 0 or header_len > 100 * 1024 * 1024:
            return set()
        header = json.loads(f.read(header_len))
    return {k for k in header if k != "__metadata__"}


def _pickle_state(path: Path) -> Optional[dict]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            pkl = next((n for n in zf.namelist() if n.endswith("data.pkl")), None)
            if not pkl:
                return None
            with zf.open(pkl) as f:
                obj = _KeysUnpickler(f).load()
            return obj if isinstance(obj, dict) else None
    # Legacy torch.save format: magic number, protocol version and sys info pickles come first.
    with path.open("rb") as f:
        obj = _KeysUnpickler(f).load()
        if isinstance(obj, int):
            _KeysUnpickler(f).load()
            _KeysUnpickler(f).load()
            obj = _KeysUnpickler(f).load()
    return obj if isinstance(obj, dict) else None


def _canonical_keys(state: dict) -> Set[str]:
    for key in _UNWRAP_KEYS:
        if isinstance(state.get(key), dict):
            state = state[key]
            break
    if len(state) == 1:
        single = next(iter(state.values()))
        if isinstance(single, dict):
            state = single
    keys = [str(k) for k in state]
    for prefix in _COMMON_PREFIXES:
        if keys and all(k.startswith(prefix) for k in keys):
            keys = [k[len(prefix):] for k in keys]
    return set(keys)


def read_state_dict_keys(path: Path) -> Set[str]:
    path = Path(path)
    if path.suffix.lower() == ".safetensors":
        return _safetensors_keys(path)
    state = _pickle_state(path)
    return _canonical_keys(state) if state else set()


def _matches(keys: Set[str], alternative: Iterable) -> bool:
    for key in alternative:
        if isinstance(key, tuple):
            if not any(k in keys for k in key):
                return False
        elif key not in keys:
            return False
    return True


def detect_architecture_from_keys(keys: Set[str]) -> str:
    for name, alternatives in _ARCH_CONDITIONS:
        if any(_matches(keys, alt) for alt in alternatives):
            return name
    return ""


def detect_gan_architecture(path: Path) -> str:
    """Architecture name of a GAN model file ("" when the file is missing or not recognized)."""
    path = Path(path)
    try:
        st = path.stat()
    except OSError:
        return ""
    cache_key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if cache_key not in _ARCH_CACHE:
        try:
            _ARCH_CACHE[cache_key] = detect_architecture_from_keys(read_state_dict_keys(path))
        except Exception:
            _ARCH_CACHE[cache_key] = ""
    return _ARCH_CACHE[cache_key]
