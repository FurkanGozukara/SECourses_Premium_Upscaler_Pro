"""Block topology of the DLSS 5 neural-rendering U-Net (HT 71 blocks, small 31 blocks).

The vendor builds every network of this family from five stage depths, one
per encoder width (32, 64, 128 and 256 channels) plus the 512-channel split
stage, mirrored by the decoder, and eight global blocks between them. The
block indices of the packed weights follow that order:

    pre | encoder 32 | 64 | 128 | 256 | split 512 | global 1024 | bridge
        | split 512 | decoder 256 | 128 | 64 | 32 | post

Known networks, both in ``nvngx_dlssnr.dll`` 310.9.16 (``CCNetwork`` factory,
one config of five depths each):

* ``HT``: depths ``(4, 4, 6, 8, 8)``, 71 blocks, preset 1. The only network of
  310.8.0.
* ``SMALL``: depths ``(2, 2, 2, 2, 2)``, 31 blocks, preset 2 (the shipping
  default of 310.9.16, ``hnet-crazy-cuckoo-small-fp8``).

Both use the same kernel families and layer packing. The window origin of a
block depends only on its position in its stage: the factory passes
``i & 3`` for the ``i``-th encoder block of a stage and ``(depth + j) & 3`` for
the ``j``-th decoder block (``depth`` of the mirrored stage), and a fixed phase
1 for the post block.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Mapping, Sequence

ENCODER_CHANNELS = (32, 64, 128, 256)
SPLIT_CHANNELS = 512
GLOBAL_CHANNELS = 1024
HEADS = {32: 1, 64: 2, 128: 4, 256: 8, SPLIT_CHANNELS: 16, GLOBAL_CHANNELS: 32}
HIDDEN = {32: 128, 64: 224, 128: 384, 256: 704, SPLIT_CHANNELS: 512, GLOBAL_CHANNELS: 4096}
GLOBAL_BLOCKS = 8
# (y, x) window origin of each phase: unshifted, both, x only, y only.
WINDOW_ORIGINS = ((0, 0), (-4, -4), (0, -4), (-4, 0))
POST_PHASE = 1

FAMILIES = (
    "pre", "window", "downsample", "split", "split_transition",
    "global", "bridge", "upsample", "post",
)


@dataclass(frozen=True)
class Block:
    index: int
    family: str
    channels: int
    heads: int
    hidden: int
    phase: int | None
    """Window-origin phase, ``None`` for blocks without window attention."""

    @property
    def window_origin(self) -> tuple[int, int]:
        return (0, 0) if self.phase is None else WINDOW_ORIGINS[self.phase % 4]


@dataclass(frozen=True)
class Topology:
    depths: tuple[int, int, int, int, int]
    global_blocks: int = GLOBAL_BLOCKS

    def __post_init__(self) -> None:
        if len(self.depths) != 5 or any(int(d) < 1 for d in self.depths):
            raise ValueError("a topology needs five stage depths of at least 1")
        if self.global_blocks < 1:
            raise ValueError("a topology needs at least one global block")
        object.__setattr__(self, "depths", tuple(int(d) for d in self.depths))

    @property
    def name(self) -> str:
        return {HT.depths: "ht", SMALL.depths: "small"}.get(self.depths, "custom") if self.global_blocks == GLOBAL_BLOCKS else "custom"

    @property
    def spec(self) -> str:
        """Stable textual form, ``"4,4,6,8,8"`` (``"/G"`` appended for a non-default global count)."""
        text = ",".join(str(d) for d in self.depths)
        return text if self.global_blocks == GLOBAL_BLOCKS else f"{text}/{self.global_blocks}"

    @classmethod
    def parse(cls, text: str) -> "Topology":
        depths, _, global_blocks = text.partition("/")
        return cls(tuple(int(v) for v in depths.split(",")), int(global_blocks) if global_blocks else GLOBAL_BLOCKS)

    @cached_property
    def blocks(self) -> tuple[Block, ...]:
        blocks: list[Block] = []

        def add(family: str, channels: int, phase: int | None) -> None:
            blocks.append(Block(len(blocks), family, channels, HEADS[channels], HIDDEN[channels], phase))

        add("pre", 32, 0)
        for stage, channels in enumerate(ENCODER_CHANNELS):
            depth = self.depths[stage]
            for i in range(depth):
                add("downsample" if i == depth - 1 else "window", channels, i & 3)
        for i in range(self.depths[4]):
            add("split_transition" if i == self.depths[4] - 1 else "split", SPLIT_CHANNELS, i & 3)
        for _ in range(self.global_blocks):
            add("global", GLOBAL_CHANNELS, None)
        add("bridge", GLOBAL_CHANNELS, None)
        for j in range(self.depths[4]):
            add("split", SPLIT_CHANNELS, (self.depths[4] + j) & 3)
        for stage in reversed(range(len(ENCODER_CHANNELS))):
            depth = self.depths[stage]
            for j in range(depth):
                add("upsample" if j == 0 else "window", ENCODER_CHANNELS[stage], (depth + j) & 3)
        add("post", 32, POST_PHASE)
        return tuple(blocks)

    @property
    def block_count(self) -> int:
        return 3 + self.global_blocks + 2 * sum(self.depths)

    @property
    def post(self) -> int:
        return self.block_count - 1

    def block(self, index: int) -> Block:
        if not 0 <= index < self.block_count:
            raise ValueError(f"block {index} is outside the {self.block_count}-block topology {self.spec}")
        return self.blocks[index]

    def window_origin(self, index: int) -> tuple[int, int]:
        return self.block(index).window_origin

    def encoder_stage(self, stage: int) -> range:
        """Block indices of encoder stage ``stage`` (0-3), the downsample last; stage 0 excludes the pre block."""
        start = 1 + sum(self.depths[:stage])
        return range(start, start + self.depths[stage])

    @property
    def split_encoder(self) -> range:
        start = 1 + sum(self.depths[:4])
        return range(start, start + self.depths[4])

    @property
    def global_stage(self) -> range:
        start = self.split_encoder.stop
        return range(start, start + self.global_blocks)

    @property
    def bridge(self) -> int:
        return self.global_stage.stop

    @property
    def split_decoder(self) -> range:
        return range(self.bridge + 1, self.bridge + 1 + self.depths[4])

    def decoder_stage(self, stage: int) -> range:
        """Block indices of the decoder stage mirroring encoder ``stage``, the upsample first."""
        start = self.split_decoder.stop + sum(self.depths[stage + 1 : 4])
        return range(start, start + self.depths[stage])

    @property
    def last_encoder_downsample(self) -> int:
        """The 256-channel downsample that pads its output to whole windows for the split stage."""
        return self.encoder_stage(3)[-1]


HT = Topology((4, 4, 6, 8, 8))
SMALL = Topology((2, 2, 2, 2, 2))
KNOWN = {"ht": HT, "small": SMALL}

_BLOCK_NAME = re.compile(r"^block(?P<block>\d+)\.")


def _layer0(shapes: Mapping[str, Sequence[int]], block: int, name: str) -> Sequence[int] | None:
    return shapes.get(f"block{block}.layer0.{name}")


def classify_logical_block(shapes: Mapping[str, Sequence[int]], block: int) -> tuple[str, int] | None:
    """``(family, channels)`` of a block in a logical (decoded) weight map."""
    if f"block{block}.layer0.first_projection_weight" in shapes:
        return ("split_transition" if f"block{block}.layer4.weight" in shapes else "split", SPLIT_CHANNELS)
    if (expand := _layer0(shapes, block, "weight")) is not None and list(expand) == [GLOBAL_CHANNELS, HIDDEN[GLOBAL_CHANNELS]]:
        return ("global", GLOBAL_CHANNELS)
    if _layer0(shapes, block, "conv_weight") is not None:
        return ("bridge", GLOBAL_CHANNELS)
    qkv = _layer0(shapes, block, "qkv_weight")
    if qkv is None:
        return None
    channels = int(qkv[0])
    if _layer0(shapes, block, "input_adapter_weight") is not None:
        return ("pre", channels)
    if _layer0(shapes, block, "inp_merge_sin") is not None:
        return ("post", channels)
    if _layer0(shapes, block, "sin") is not None:
        return ("upsample", channels)
    if (weight0 := _layer0(shapes, block, "weight0")) is not None and list(weight0) == [channels, 2 * channels]:
        return ("downsample", channels)
    return ("window", channels)


def infer_from_families(families: Sequence[tuple[str, int] | None]) -> Topology:
    """Recover the stage depths from the ordered ``(family, channels)`` of every block.

    A map with unrecognised or missing blocks (``None``), such as a partial
    fixture, resolves to the first known preset that agrees with every block
    that is present.
    """
    try:
        return _infer_complete(families)
    except ValueError:
        if None not in families:
            raise
        for preset in KNOWN.values():
            expected = [(block.family, block.channels) for block in preset.blocks]
            if len(families) <= len(expected) and all(
                family is None or family == expected[index] for index, family in enumerate(families)
            ):
                return preset
        raise


def _infer_complete(families: Sequence[tuple[str, int] | None]) -> Topology:
    position = 0

    def take(predicate) -> int:
        nonlocal position
        start = position
        while position < len(families) and families[position] is not None and predicate(families[position]):
            position += 1
        return position - start

    if not families or families[0] != ("pre", 32):
        raise ValueError("block 0 is not the 32-channel pre block")
    position = 1
    depths: list[int] = []
    for channels in ENCODER_CHANNELS:
        regular = take(lambda f, c=channels: f == ("window", c))
        if take(lambda f, c=channels: f == ("downsample", c)) != 1:
            raise ValueError(f"encoder stage {channels} has no single downsample block at {position}")
        depths.append(regular + 1)
    split = take(lambda f: f == ("split", SPLIT_CHANNELS))
    if take(lambda f: f == ("split_transition", SPLIT_CHANNELS)) != 1:
        raise ValueError(f"split encoder has no single transition block at {position}")
    depths.append(split + 1)
    global_blocks = take(lambda f: f[0] == "global")
    if take(lambda f: f[0] == "bridge") != 1:
        raise ValueError(f"no bridge block at {position}")
    topology = Topology(tuple(depths), global_blocks)
    expected = [(block.family, block.channels) for block in topology.blocks]
    actual = list(families)
    if actual != expected:
        mismatch = next(i for i, (a, b) in enumerate(zip(actual + [None] * len(expected), expected)) if a != b)
        raise ValueError(
            f"weights do not follow the mirrored topology {topology.spec}: block {mismatch} is "
            f"{actual[mismatch] if mismatch < len(actual) else 'missing'}, expected {expected[mismatch]}"
        )
    return topology


def block_count(names: Iterable[str]) -> int:
    indices = [int(m.group("block")) for name in names if (m := _BLOCK_NAME.match(name))]
    if not indices:
        raise ValueError("no blockN tensors")
    return max(indices) + 1


def infer_logical(shapes: Mapping[str, Sequence[int]]) -> Topology:
    """Topology of a logical weight map (``name -> shape``)."""
    count = block_count(shapes)
    return infer_from_families([classify_logical_block(shapes, block) for block in range(count)])
