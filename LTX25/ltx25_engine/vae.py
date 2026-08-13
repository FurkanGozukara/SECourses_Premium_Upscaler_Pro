"""
Slim LTX 2.5 video VAE wrapper (ComfyUI-parity paths only).

Wraps the vendored `VideoVAE` (conv decoder) / `CausalDiffusionVAE`
(diffusion decoder) with exactly the tiled encode/decode code paths the
"IC LoRA Pixel Spatial Upscaler" workflow uses:

- encode_tiled: tiled_scale_multidim, pixel tiles (T=all, 512, 512), overlap
  (1, 64, 64), downscale with index formulas (8, 32, 32).
- decode_tiled: latent tiles from the VAEDecodeTiled node params
  (tile 512px -> 16, overlap 64px -> 2, temporal 128f -> 16, overlap 32f -> 4).

Pixel IO matches comfy.sd.VAE: process_input = x*2-1 (BHWC 0..1 input),
process_output = clamp((x+1)/2, 0, 1).
"""

from __future__ import annotations

import json
import logging
import math

import torch

import comfy.utils
import comfy.model_management as mm
from comfy.ldm.lightricks.vae.causal_video_autoencoder import VideoVAE
from comfy.ldm.lightricks.vae.na_diffusion_decoder import CausalDiffusionVAE


class Ltx25VideoVae:
    UPSCALE_RATIO = (lambda a: max(0, a * 8 - 7), 32, 32)
    DOWNSCALE_RATIO = (lambda a: max(0, math.floor((a + 7) / 8)), 32, 32)
    INDEX_FORMULA = (8, 32, 32)

    def __init__(self, path: str, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        sd, metadata = comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)
        vae_config = None
        if metadata is not None and "config" in metadata:
            vae_config = json.loads(metadata["config"]).get("vae", None)

        self.is_diffusion_decoder = "decoder.conv_in_x_t.weight" in sd
        if self.is_diffusion_decoder:
            self.first_stage_model = CausalDiffusionVAE(config=vae_config)
            self.memory_used_decode = lambda shape, dt: (
                1700 * shape[2] * shape[3] * shape[4] * (8 * 8 * 8)
            ) * mm.dtype_size(dt)
        else:
            tensor_conv1 = sd["decoder.up_blocks.0.res_blocks.0.conv1.conv.weight"]
            version = 0
            if tensor_conv1.shape[0] == 1024:
                version = 1
                if "encoder.down_blocks.1.conv.conv.bias" in sd:
                    version = 2
            self.first_stage_model = VideoVAE(version=version, config=vae_config)
            self.memory_used_decode = lambda shape, dt: (
                1200 * shape[2] * shape[3] * shape[4] * (8 * 8 * 8)
            ) * mm.dtype_size(dt)
        self.memory_used_encode = lambda shape, dt: (
            80 * max(shape[2], 7) * shape[3] * shape[4]
        ) * mm.dtype_size(dt)

        self.first_stage_model = self.first_stage_model.eval()
        missing, unexpected = self.first_stage_model.load_state_dict(sd, strict=False)
        if missing:
            logging.warning(f"LTX25 VAE missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            logging.debug(f"LTX25 VAE unexpected keys: {len(unexpected)}")

        self.device = device
        self.dtype = dtype
        self.output_device = torch.device("cpu")
        self.latent_channels = 128
        self.first_stage_model.to(dtype)
        # `downscale_index_formula` mirrors comfy.sd.VAE for the guide encoder.
        self.downscale_index_formula = self.INDEX_FORMULA

    # -- residency -------------------------------------------------------
    def to_gpu(self):
        self.first_stage_model.to(self.device)

    def to_cpu(self):
        self.first_stage_model.to("cpu")
        mm.soft_empty_cache()

    # -- pixel conversion (comfy.sd.VAE parity) ---------------------------
    @staticmethod
    def process_input(image: torch.Tensor) -> torch.Tensor:
        return image * 2.0 - 1.0

    @staticmethod
    def process_output(image: torch.Tensor) -> torch.Tensor:
        return torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)

    # -- encode ------------------------------------------------------------
    @torch.no_grad()
    def encode_tiled(self, pixels_bhwc: torch.Tensor, tile_x: int = 512, tile_y: int = 512, overlap: int = 64) -> torch.Tensor:
        """pixels_bhwc: [F, H, W, C] float 0..1 -> latent [1, 128, Tl, H/32, W/32]."""
        pixels = pixels_bhwc.movedim(-1, 1)  # F,C,H,W
        pixels = pixels.movedim(1, 0).unsqueeze(0)  # 1,C,F,H,W
        self.to_gpu()
        encode_fn = lambda a: self.first_stage_model.encode(
            self.process_input(a).to(self.dtype).to(self.device)
        ).to(dtype=torch.float32)
        samples = comfy.utils.tiled_scale_multidim(
            pixels,
            encode_fn,
            tile=(9999, tile_x, tile_y),
            overlap=(1, overlap, overlap),
            upscale_amount=self.DOWNSCALE_RATIO,
            out_channels=self.latent_channels,
            downscale=True,
            index_formulas=self.INDEX_FORMULA,
            output_device=self.output_device,
        )
        return samples.float()

    @torch.no_grad()
    def encode(self, pixels_bhwc: torch.Tensor) -> torch.Tensor:
        pixels = pixels_bhwc.movedim(-1, 1).movedim(1, 0).unsqueeze(0)
        self.to_gpu()
        try:
            out = self.first_stage_model.encode(self.process_input(pixels).to(self.dtype).to(self.device))
            return out.float().to(self.output_device)
        except mm.OOM_EXCEPTION:
            logging.warning("LTX25 VAE encode OOM; falling back to tiled encode.")
            mm.soft_empty_cache()
            return self.encode_tiled(pixels_bhwc)

    # -- decode ------------------------------------------------------------
    @torch.no_grad()
    def decode_tiled(
        self,
        samples: torch.Tensor,
        tile_size_px: int = 512,
        overlap_px: int = 64,
        temporal_size_frames: int = 128,
        temporal_overlap_frames: int = 32,
    ) -> torch.Tensor:
        """samples: [1, 128, T, H, W] -> pixels [T', H*32, W*32, C] on CPU (0..1).

        Parameter conversion mirrors the ComfyUI VAEDecodeTiled node with the
        workflow defaults (512 / 64 / 128 / 32).
        """
        temporal_compression = 8
        tile_t = max(2, temporal_size_frames // temporal_compression)
        overlap_t = max(1, min(tile_t // 2, temporal_overlap_frames // temporal_compression))
        compression = 32
        tile_x = max(1, tile_size_px // compression)
        tile_y = max(1, tile_size_px // compression)
        overlap = max(1, min(tile_x // 4, overlap_px // compression))

        self.to_gpu()
        decode_fn = lambda a: self.first_stage_model.decode(
            a.to(self.dtype).to(self.device)
        ).to(dtype=torch.float32)
        pixels = comfy.utils.tiled_scale_multidim(
            samples,
            decode_fn,
            tile=(tile_t, tile_x, tile_y),
            overlap=(overlap_t, overlap, overlap),
            upscale_amount=self.UPSCALE_RATIO,
            out_channels=3,
            index_formulas=self.INDEX_FORMULA,
            output_device=self.output_device,
        )
        pixels = self.process_output(pixels)
        return self._to_bhwc(pixels)

    @staticmethod
    def _to_bhwc(pixels: torch.Tensor) -> torch.Tensor:
        # [1, C, T, H, W] -> [T, H, W, C]
        return pixels.squeeze(0).movedim(0, -1).contiguous()
