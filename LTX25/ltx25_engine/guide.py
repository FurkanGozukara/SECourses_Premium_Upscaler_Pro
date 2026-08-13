"""
IC-LoRA guide construction for the pixel spatial upscaler (2x).

Exact port of LTXAddVideoICLoRAGuide (ComfyUI-LTXVideo/iclora.py) with the
workflow's fixed parameters: frame_idx=0, crop=disabled, tiled encode 512/64,
latent_downscale_factor from the LoRA metadata (2 for this IC LoRA).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

import comfy.utils
from comfy.ldm.lightricks.symmetric_patchifier import SymmetricPatchifier, latent_to_pixel_coords

SCALE_FACTORS = (8, 32, 32)
_PATCHIFIER = SymmetricPatchifier(1, start_end=True)


@dataclass
class GuideBundle:
    latent: torch.Tensor            # [1, 128, 2L, H, W] video latent with guide appended
    noise_mask: torch.Tensor        # [1, 1, 2L, H, W] 1=denoise, 0=frozen guide, -2=filler
    keyframe_idxs: torch.Tensor     # [1, 3, L*H*W, 2] pre-filter guide coords
    guide_attention_entries: List[dict]
    num_guide_latent_frames: int


def build_guide(
    vae,
    frames_bhwc: torch.Tensor,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    guide_strength: float = 1.0,
    latent_downscale_factor: float = 2.0,
    tile_size: int = 512,
    tile_overlap: int = 64,
    use_tiled_encode: bool = True,
) -> GuideBundle:
    """frames_bhwc: padded source chunk [F(8n+1), Hpad, Wpad, C] float 0..1."""
    factor = int(latent_downscale_factor)
    time_scale = SCALE_FACTORS[0]

    images = frames_bhwc
    num_frames_to_keep = ((images.shape[0] - 1) // time_scale) * time_scale + 1
    images = images[:num_frames_to_keep]

    target_width = int(latent_width * SCALE_FACTORS[1] / latent_downscale_factor)
    target_height = int(latent_height * SCALE_FACTORS[2] / latent_downscale_factor)
    if images.shape[2] != target_width or images.shape[1] != target_height:
        pixels = comfy.utils.common_upscale(
            images.movedim(-1, 1), target_width, target_height, "bilinear", crop="disabled"
        ).movedim(1, -1)
    else:
        pixels = images
    encode_pixels = pixels[:, :, :, :3]

    if use_tiled_encode:
        guide_latent = vae.encode_tiled(encode_pixels, tile_x=tile_size, tile_y=tile_size, overlap=tile_overlap)
    else:
        guide_latent = vae.encode(encode_pixels)

    guide_mask = None
    if factor > 1:
        if latent_width % factor != 0 or latent_height % factor != 0:
            raise ValueError(
                f"Latent spatial size {latent_width}x{latent_height} must be divisible by "
                f"latent_downscale_factor {factor}"
            )
        dilated_shape = guide_latent.shape[:3] + (
            guide_latent.shape[3] * factor,
            guide_latent.shape[4] * factor,
        )
        dilated = torch.zeros(dilated_shape, device=guide_latent.device, dtype=guide_latent.dtype)
        dilated[..., ::factor, ::factor] = guide_latent
        guide_mask = torch.full(
            (dilated.shape[0], 1, dilated.shape[2], dilated.shape[3], dilated.shape[4]),
            -1.0,
            device=guide_latent.device,
            dtype=guide_latent.dtype,
        )
        guide_mask[..., ::factor, ::factor] = 1.0
        guide_orig_shape = list(guide_latent.shape[2:])
        guide_latent = dilated
    else:
        guide_orig_shape = list(guide_latent.shape[2:])

    if guide_latent.shape[2] > latent_frames:
        raise ValueError("Conditioning frames exceed the length of the latent sequence.")

    # Keyframe RoPE coords on the full (dilated) grid, frame_idx=0, causal fix on.
    _, latent_coords = _PATCHIFIER.patchify(guide_latent)
    pixel_coords = latent_to_pixel_coords(latent_coords, SCALE_FACTORS, causal_fix=True)
    spatial_end_offset = (latent_downscale_factor - 1) * torch.tensor(
        SCALE_FACTORS[1:], device=pixel_coords.device
    ).view(1, -1, 1, 1)
    pixel_coords[:, 1:, :, 1:] += spatial_end_offset.to(pixel_coords.dtype)
    keyframe_idxs = pixel_coords

    # Base latent (zeros) + guide appended along frames.
    base_latent = torch.zeros(
        (1, 128, latent_frames, latent_height, latent_width), dtype=torch.float32
    )
    base_mask = torch.ones((1, 1, latent_frames, latent_height, latent_width), dtype=torch.float32)

    if guide_mask is not None:
        mask = guide_mask.float() - float(guide_strength)
    else:
        mask = torch.full(
            (1, 1, guide_latent.shape[2], latent_height, latent_width),
            max(0.0, 1.0 - float(guide_strength)),
            dtype=torch.float32,
        )

    latent = torch.cat([base_latent, guide_latent.float()], dim=2)
    noise_mask = torch.cat([base_mask, mask], dim=2)

    pre_filter_count = guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
    entries = [
        {
            "pre_filter_count": int(pre_filter_count),
            "strength": float(guide_strength),
            "pixel_mask": None,
            "latent_shape": guide_orig_shape,
        }
    ]
    return GuideBundle(
        latent=latent,
        noise_mask=noise_mask,
        keyframe_idxs=keyframe_idxs,
        guide_attention_entries=entries,
        num_guide_latent_frames=int(guide_latent.shape[2]),
    )


def crop_guides(latent: torch.Tensor, num_guide_latent_frames: int) -> torch.Tensor:
    """LTXVCropGuides: drop the appended guide latent frames after sampling."""
    if num_guide_latent_frames <= 0:
        return latent
    return latent[:, :, :-num_guide_latent_frames]
