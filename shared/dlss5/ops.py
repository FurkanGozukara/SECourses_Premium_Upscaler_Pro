"""GPU frame operations around the DLSS 5 network (all tensors HWC float32 on one device).

Network input, 16 channels (half-rounded values):
  0-2  deterministic Gaussian noise (per network pixel and frame index)
  3    constant 1
  4-6  scaled colour ((half(c) - 0.5) * 0.125)
  7-9  scaled history: the reprojected previous output (the colour again on a first frame)
  10   normalised style, 11 local tone, 12 local structure,
  13   skin structure, 14 automatic-mask structure, 15 zero
The network runs on a vendor-aligned extent (at least 320, multiples of 64): the
image sits at the origin and the extension mirrors it without repeating the edge
(index 2*extent-2-x), while the noise comes from the network coordinates.
Output head: RGB residual (x0.25) and one temporal blend logit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F

MINIMUM_EXTENT = 320
EXTENT_MULTIPLE = 64
FEATURE_CHANNELS = 16
BLEND_SCALE = 0.73974609375  # half(blend_scale) of the HT package
_MASK32 = 0xFFFFFFFF

# Style index / 128, local tone and local structure of the vendor profiles.
PROFILES: dict[str, dict[str, float]] = {
    "Standard": {"normalized_style": 0.0, "local_tone_strength": 1.0, "local_structure_strength": 1.0},
    "Natural": {"normalized_style": 1.0 / 128, "local_tone_strength": 1.0, "local_structure_strength": 1.0},
    "Cinematic": {"normalized_style": 2.0 / 128, "local_tone_strength": 1.0, "local_structure_strength": 1.0},
    "Neutral": {"normalized_style": 0.0, "local_tone_strength": 0.0, "local_structure_strength": 0.0},
}


def half(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.float16).to(torch.float32)


def _half_scalar(value: float) -> float:
    return float(torch.tensor(float(value), dtype=torch.float16).item())


@dataclass(frozen=True)
class NetworkGeometry:
    output_width: int
    output_height: int
    network_width: int
    network_height: int

    @staticmethod
    def aligned_extent(extent: int) -> int:
        minimum = max(MINIMUM_EXTENT, int(extent))
        return (minimum + EXTENT_MULTIPLE - 1) // EXTENT_MULTIPLE * EXTENT_MULTIPLE

    @classmethod
    def vendor_aligned(cls, width: int, height: int) -> "NetworkGeometry":
        return cls(int(width), int(height), cls.aligned_extent(width), cls.aligned_extent(height))

    @property
    def is_identity(self) -> bool:
        return (self.network_width, self.network_height) == (self.output_width, self.output_height)

    @property
    def tokens(self) -> int:
        return self.network_width * self.network_height


@lru_cache(maxsize=16)
def _mirror_index_cpu(count: int, extent: int) -> torch.Tensor:
    index = torch.arange(count, dtype=torch.long)
    return torch.where(index < extent, index, torch.clamp(2 * extent - 2 - index, min=0))


def mirror_indices(geometry: NetworkGeometry, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = _mirror_index_cpu(geometry.network_height, geometry.output_height).to(device)
    columns = _mirror_index_cpu(geometry.network_width, geometry.output_width).to(device)
    return rows, columns


def extend(image: torch.Tensor, geometry: NetworkGeometry) -> torch.Tensor:
    """Mirror a logical (H, W, C) tensor onto the network extent."""
    if geometry.is_identity:
        return image
    rows, columns = mirror_indices(geometry, image.device)
    return image.index_select(0, rows).index_select(1, columns)


# --- deterministic noise (exact uint32 arithmetic emulated in int64) ---------------------------

def _u32(value: torch.Tensor) -> torch.Tensor:
    return value & _MASK32


def _dynamic_shift_mix(value: torch.Tensor) -> torch.Tensor:
    shift = (value >> 28) + 4
    mixed = value ^ (value >> shift)
    return _u32(mixed * 0x108EF2D9)


def _uniform24(value: torch.Tensor) -> torch.Tensor:
    mixed = _dynamic_shift_mix(value)
    bits = (mixed >> 30) ^ (mixed >> 8)
    return (bits + 1).to(torch.float32) * 5.960464477539063e-8


@lru_cache(maxsize=4)
def _noise_base(height: int, width: int, device_key: str) -> torch.Tensor:
    device = torch.device(device_key)
    yy = torch.arange(height, dtype=torch.int64, device=device).unsqueeze(1)
    xx = torch.arange(width, dtype=torch.int64, device=device).unsqueeze(0)
    return _u32(_u32(yy * 0xD8163841) ^ _u32(xx * 0x8DA6B343))


def deterministic_noise(height: int, width: int, frame_index: int, device: torch.device) -> torch.Tensor:
    """The three half-rounded Gaussian noise channels of a network extent, (H, W, 3) float32."""
    seed = _noise_base(int(height), int(width), str(device))
    seed = seed ^ ((int(frame_index) * 0x9E3779B9) & _MASK32) ^ 0x243F6A88
    multiplied = _dynamic_shift_mix(seed)
    mixed = multiplied ^ (multiplied >> 22)
    radius_a = _uniform24(_u32(mixed * 0xCAA5B80D + 0x21DD796B))
    angle_b = _uniform24(_u32(mixed * 0x83232C31 + 0x3463E0AC))
    radius_b = _uniform24(_u32(mixed * 0x2C9277B5 + 0xAC564B05))
    angle_a = _uniform24(_u32(mixed * 0xFA6DC5F9 + 0x4712A88E))
    del mixed, multiplied, seed
    radius_a = torch.sqrt(-2.0 * torch.log(radius_a))
    radius_b = torch.sqrt(-2.0 * torch.log(radius_b))
    tau = 6.2831854820251465
    angle_a = tau * angle_a
    angle_b = tau * angle_b
    return torch.stack(
        (half(radius_b * torch.cos(angle_a)), half(radius_b * torch.sin(angle_a)), half(radius_a * torch.cos(angle_b))),
        dim=-1,
    )


def scaled_color(color: torch.Tensor) -> torch.Tensor:
    return half(half(half(color) - 0.5) * 0.125)


def control_values(profile: str, overrides: Optional[dict] = None) -> dict[str, float]:
    controls = dict(PROFILES.get(profile) or PROFILES["Standard"])
    for key, value in (overrides or {}).items():
        if value is not None and key in controls:
            controls[key] = float(value)
    return controls


def make_features(
    color: torch.Tensor,
    *,
    frame_index: int,
    geometry: NetworkGeometry,
    normalized_style: float = 0.0,
    local_tone_strength: float = 1.0,
    local_structure_strength: float = 1.0,
    history: Optional[torch.Tensor] = None,
    history_confidence: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the (1, network_height, network_width, 16) network input.

    ``color`` and ``history`` are logical (H, W, 3) float32 in [0, 1]; ``history`` is
    the reprojected previous output and ``history_confidence`` (H, W, 1) mixes it with
    the current colour where the correspondence is unreliable.
    """
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("color must be (height, width, 3)")
    if (int(color.shape[1]), int(color.shape[0])) != (geometry.output_width, geometry.output_height):
        raise ValueError("geometry output extent must match the colour image")
    device = color.device
    features = torch.empty((geometry.network_height, geometry.network_width, FEATURE_CHANNELS), dtype=torch.float32, device=device)
    features[..., 0:3] = deterministic_noise(geometry.network_height, geometry.network_width, frame_index, device)
    features[..., 3] = 1.0
    current = scaled_color(color)
    features[..., 4:7] = extend(current, geometry)
    if history is None:
        features[..., 7:10] = features[..., 4:7]
    else:
        reprojected = scaled_color(history)
        if history_confidence is not None:
            c = history_confidence.to(torch.float32)
            mixed = current + c * (reprojected - current)
            reprojected = torch.where(c == 0, current, torch.where(c == 1, reprojected, mixed))
        features[..., 7:10] = extend(reprojected, geometry)
    features[..., 10] = _half_scalar(normalized_style)
    features[..., 11] = _half_scalar(local_tone_strength)
    features[..., 12] = _half_scalar(local_structure_strength)
    features[..., 13] = -1.0
    features[..., 14] = -1.0
    features[..., 15] = 0.0
    return features.unsqueeze(0).to(dtype)


def compose_head(head: torch.Tensor, color: torch.Tensor, *, intensity: float = 1.0) -> torch.Tensor:
    """First frame: RGB = colour + 0.25 * half(head[..., :3]), blended by intensity."""
    predicted = torch.clamp(color + half(head[..., :3].to(torch.float32)) * 0.25, 0.0, 1.0)
    blend = min(max(float(intensity), 0.0), 1.0)
    if blend == 1.0:
        return predicted
    return torch.clamp(color + blend * (predicted - color), 0.0, 1.0)


def compose_temporal(
    head: torch.Tensor,
    color: torch.Tensor,
    history: torch.Tensor,
    *,
    blend_scale: float = BLEND_SCALE,
    history_confidence: Optional[torch.Tensor] = None,
    intensity: float = 1.0,
) -> torch.Tensor:
    """predicted + alpha * (history - predicted), alpha = clamp(sigmoid(half(logit)) * half(blend_scale)).

    ``history`` is the history the network saw (channels 7-9 unscaled: x8 + 0.5).
    """
    head = head.to(torch.float32)
    logit = half(head[..., 3:4])
    alpha = torch.clamp(torch.sigmoid(logit) * _half_scalar(blend_scale), 0.0, 1.0)
    if history_confidence is not None:
        alpha = alpha * history_confidence
    predicted = torch.clamp(color + half(head[..., :3]) * 0.25, 0.0, 1.0)
    temporal = predicted + alpha * (history - predicted)
    blend = min(max(float(intensity), 0.0), 1.0)
    if blend == 1.0:
        return temporal
    return torch.clamp(color + blend * (temporal - color), 0.0, 1.0)


def network_history(features: torch.Tensor, geometry: NetworkGeometry) -> torch.Tensor:
    """The history the network saw, cropped to the logical extent and unscaled."""
    return features[0, : geometry.output_height, : geometry.output_width, 7:10].to(torch.float32) * 8.0 + 0.5


@lru_cache(maxsize=8)
def _gaussian_kernel(radius: float, device_key: str) -> torch.Tensor:
    extent = int(math.ceil(3 * radius))
    offsets = torch.arange(-extent, extent + 1, dtype=torch.float32, device=torch.device(device_key))
    kernel = torch.exp(-offsets * offsets / float(2 * radius * radius))
    return kernel / kernel.sum()


def gaussian_blur(image: torch.Tensor, radius: float) -> torch.Tensor:
    """Separable Gaussian blur of (H, W, C) with edge replication."""
    kernel = _gaussian_kernel(float(radius), str(image.device))
    extent = (kernel.numel() - 1) // 2
    channels = image.shape[2]
    x = image.permute(2, 0, 1).unsqueeze(0)
    x = F.conv2d(F.pad(x, (extent, extent, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels)
    x = F.conv2d(F.pad(x, (0, 0, extent, extent), mode="replicate"), kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels)
    return x.squeeze(0).permute(1, 2, 0)


def compose_detail(
    source: torch.Tensor,
    output: torch.Tensor,
    *,
    detail_strength: float = 1.0,
    colour_strength: float = 1.0,
    radius: float = 4.0,
) -> torch.Tensor:
    """result = source + colour * lowpass(change) + detail * highpass(change), clamped to [0, 1]."""
    if detail_strength == 1.0 and colour_strength == 1.0:
        return output
    change = output - source
    low = gaussian_blur(change, max(0.5, float(radius)))
    return torch.clamp(source + float(colour_strength) * low + float(detail_strength) * (change - low), 0.0, 1.0)


def resample(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Resize (H, W, C): box average for integer downscales, antialiased bicubic otherwise."""
    source_height, source_width = int(image.shape[0]), int(image.shape[1])
    width, height = int(width), int(height)
    if (width, height) == (source_width, source_height):
        return image
    x = image.permute(2, 0, 1).unsqueeze(0)
    if (
        source_width % width == 0 and source_height % height == 0
        and source_width // width == source_height // height and source_width // width > 1
    ):
        factor = source_width // width
        x = F.avg_pool2d(x, factor)
    else:
        x = F.interpolate(x, size=(height, width), mode="bicubic", align_corners=False, antialias=width < source_width)
    return x.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)


def _catmull_coordinates(normalized: torch.Tensor, dimension: int):
    pixel = normalized * float(dimension) - 0.5
    base_index = torch.floor(pixel)
    t = torch.clamp(pixel - base_index, 0.0, 1.0)
    square = t * t
    cube = square * t
    w0 = -0.5 * t + square - 0.5 * cube
    w1 = 1.0 - 2.5 * square + 1.5 * cube
    w2 = 0.5 * t + 2.0 * square - 1.5 * cube
    w3 = -0.5 * square + 0.5 * cube
    g = w1 + w2
    base = base_index + 0.5
    lower, upper = 0.5, float(dimension) - 0.5
    return (
        torch.clamp(base - 1.0, lower, upper), torch.clamp(base + w2 / g, lower, upper), torch.clamp(base + 2.0, lower, upper),
        w0, w3, g,
    )


def sample_history(history: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Five-tap Catmull-Rom approximation (a cross of bilinear taps) at normalised (u, v)."""
    height, width = int(history.shape[0]), int(history.shape[1])
    x0, xm, x3, xw0, xw3, xg = _catmull_coordinates(u, width)
    y0, ym, y3, yw0, yw3, yg = _catmull_coordinates(v, height)
    weights = (xw0 * yg, xg * yw0, xg * yg, xg * yw3, xw3 * yg)
    coords = ((x0, ym), (xm, y0), (xm, ym), (xm, y3), (x3, ym))
    source = history.permute(2, 0, 1).unsqueeze(0)
    total = None
    for weight, (cx, cy) in zip(weights, coords):
        grid = torch.stack((cx * (2.0 / width) - 1.0, cy * (2.0 / height) - 1.0), dim=-1).unsqueeze(0)
        tap = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=False)
        tap = tap.squeeze(0).permute(1, 2, 0) * weight.unsqueeze(-1)
        total = tap if total is None else total + tap
    return total / (weights[0] + weights[1] + weights[2] + weights[3] + weights[4]).unsqueeze(-1)


def reproject(history: torch.Tensor, motion_uv: torch.Tensor) -> torch.Tensor:
    """Sample the previous output at currentUV + motion (normalised offsets, +x right, +y down)."""
    height, width = int(history.shape[0]), int(history.shape[1])
    yy = (torch.arange(height, dtype=torch.float32, device=history.device) + 0.5) / float(height)
    xx = (torch.arange(width, dtype=torch.float32, device=history.device) + 0.5) / float(width)
    u = xx.unsqueeze(0) + motion_uv[..., 0]
    v = yy.unsqueeze(1) + motion_uv[..., 1]
    return sample_history(history, u, v)


def luma(image: torch.Tensor) -> torch.Tensor:
    return image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722
