"""
Sigma schedules and samplers for LTX 2.5 (ComfyUI-parity).

- Distilled 8-step schedule: exact port of LTX25DistilledSigmaSchedule.
- Dev schedule: exact port of ComfyUI's LTXVScheduler (token-shifted, stretched).
- Samplers: euler, euler_ancestral (rectified-flow variant used by ComfyUI for
  CONST models) and res_multistep, ported from comfy/k_diffusion/sampling.py.
- The denoise-mask pinning replicates comfy.samplers.KSamplerX0Inpaint with
  LTXV's scale_latent_inpaint override (masked regions pinned to the clean
  latent, no re-noising).
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional

import torch

OFFICIAL_DISTILLED_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)

SAMPLER_NAMES = ["euler_ancestral", "euler", "res_multistep"]


def distilled_sigmas(sampling_steps: int) -> torch.Tensor:
    """LTX25DistilledSigmaSchedule — linear resample of the official 9-value curve."""
    official = torch.tensor(OFFICIAL_DISTILLED_SIGMAS, dtype=torch.float32)
    positions = torch.linspace(0, len(official) - 1, int(sampling_steps) + 1)
    lower = positions.floor().to(torch.long)
    upper = positions.ceil().to(torch.long)
    fraction = positions - lower.to(positions.dtype)
    return official[lower] * (1.0 - fraction) + official[upper] * fraction


def ltxv_scheduler_sigmas(
    steps: int,
    tokens: int,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> torch.Tensor:
    """ComfyUI LTXVScheduler (nodes_lt.py) — token-count shifted sigmas."""
    sigmas = torch.linspace(1.0, 0.0, steps + 1)
    x1, x2 = 1024, 4096
    mm = (max_shift - base_shift) / (x2 - x1)
    b = base_shift - mm * x1
    sigma_shift = tokens * mm + b
    power = 1
    sigmas = torch.where(
        sigmas != 0,
        math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power),
        0,
    )
    if stretch:
        non_zero_mask = sigmas != 0
        non_zero_sigmas = sigmas[non_zero_mask]
        one_minus_z = 1.0 - non_zero_sigmas
        scale_factor = one_minus_z[-1] / (1.0 - terminal)
        stretched = 1.0 - (one_minus_z / scale_factor)
        sigmas[non_zero_mask] = stretched
    return sigmas


def prepare_noise(latent: torch.Tensor, seed: int) -> torch.Tensor:
    """comfy.sample.prepare_noise parity: CPU randn with a manual-seeded generator."""
    generator = torch.manual_seed(int(seed))
    return torch.randn(
        latent.size(), dtype=latent.dtype, layout=latent.layout, generator=generator, device="cpu"
    )


def _default_noise_sampler(x: torch.Tensor, seed: Optional[int]):
    if seed is not None:
        if x.device == torch.device("cpu"):
            seed += 1
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
    else:
        generator = None
    return lambda sigma, sigma_next: torch.randn(
        x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator
    )


def _to_d(x, sigma, denoised):
    return (x - denoised) / sigma


def _get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(
        sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5
    )
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


@torch.no_grad()
def sample_euler(model, x, sigmas, seed=None, callback=None):
    for i in range(len(sigmas) - 1):
        sigma = float(sigmas[i])
        denoised = model(x, sigma)
        d = _to_d(x, sigma, denoised)
        dt = float(sigmas[i + 1]) - sigma
        x = x + d * dt
        if callback is not None:
            callback(i)
    return x


@torch.no_grad()
def sample_euler_ancestral_rf(model, x, sigmas, seed=None, eta=1.0, s_noise=1.0, callback=None):
    """comfy sample_euler_ancestral_RF — the CONST/flow path ComfyUI dispatches to."""
    noise_sampler = _default_noise_sampler(x, seed)
    for i in range(len(sigmas) - 1):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        denoised = model(x, sigma)
        if sigma_next == 0:
            x = denoised
        else:
            downstep_ratio = 1 + (sigma_next / sigma - 1) * eta
            sigma_down = sigma_next * downstep_ratio
            alpha_ip1 = 1 - sigma_next
            alpha_down = 1 - sigma_down
            renoise_coeff = (sigma_next ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2) ** 0.5
            sigma_down_i_ratio = sigma_down / sigma
            x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised
            if eta > 0:
                x = (alpha_ip1 / alpha_down) * x + noise_sampler(sigma, sigma_next) * s_noise * renoise_coeff
        if callback is not None:
            callback(i)
    return x


@torch.no_grad()
def sample_res_multistep(model, x, sigmas, seed=None, s_noise=1.0, eta=0.0, callback=None):
    """comfy res_multistep (cfg_pp=False). With eta=0 this is sample_res_multistep."""
    noise_sampler = _default_noise_sampler(x, seed)
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    phi1_fn = lambda t: torch.expm1(t) / t
    phi2_fn = lambda t: (phi1_fn(t) - 1.0) / t

    old_sigma_down = None
    old_denoised = None
    for i in range(len(sigmas) - 1):
        sigma = float(sigmas[i])
        denoised = model(x, sigma)
        sigma_down, sigma_up = _get_ancestral_step(sigma, float(sigmas[i + 1]), eta=eta)
        if sigma_down == 0 or old_denoised is None:
            d = _to_d(x, sigma, denoised)
            dt = sigma_down - sigma
            x = x + d * dt
        else:
            t, t_old, t_next, t_prev = (
                t_fn(torch.tensor(sigma)),
                t_fn(torch.tensor(old_sigma_down)),
                t_fn(torch.tensor(sigma_down)),
                t_fn(sigmas[i - 1].clone().detach()),
            )
            h = t_next - t
            c2 = (t_prev - t_old) / h
            phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
            b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)
            x = sigma_fn(h) * x + h * (b1 * denoised + b2 * old_denoised)
        if sigma_up > 0:
            x = x + noise_sampler(sigma, float(sigmas[i + 1])) * s_noise * sigma_up
        old_denoised = denoised
        old_sigma_down = sigma_down
        if callback is not None:
            callback(i)
    return x


SAMPLERS: dict = {
    "euler": sample_euler,
    "euler_ancestral": sample_euler_ancestral_rf,
    "res_multistep": sample_res_multistep,
}


class MaskedFlowModel:
    """
    Wraps the LTXAV apply-model call with KSamplerX0Inpaint semantics:

      x_in  = x * mask + clean * (1 - mask)       (scale_latent_inpaint == latent_image)
      out   = apply(x_in, sigma)                  (per-token timestep = sigma * mask)
      out   = out * mask + clean * (1 - mask)

    Only the video latent is integrated; the audio latent is a fully frozen
    zero tensor handled inside `apply_fn`.
    """

    def __init__(
        self,
        apply_fn: Callable[[torch.Tensor, float], torch.Tensor],
        denoise_mask: torch.Tensor,
        clean_latent: torch.Tensor,
    ):
        self.apply_fn = apply_fn
        self.mask = denoise_mask
        self.latent_image = clean_latent

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        latent_mask = 1.0 - self.mask
        x = x * self.mask + self.latent_image * latent_mask
        out = self.apply_fn(x, sigma)
        out = out * self.mask + self.latent_image * latent_mask
        return out


def run_sampler(
    sampler_name: str,
    model: Callable[[torch.Tensor, float], torch.Tensor],
    noise: torch.Tensor,
    sigmas: torch.Tensor,
    seed: int,
    callback: Optional[Callable[[int], None]] = None,
) -> torch.Tensor:
    sampler = SAMPLERS.get(str(sampler_name or "euler_ancestral"))
    if sampler is None:
        raise ValueError(f"Unknown sampler: {sampler_name}. Valid: {SAMPLER_NAMES}")
    # sigma_max is 1.0 for these schedules; initial x = noise * sigma_max.
    x = noise * float(sigmas[0])
    return sampler(model, x, sigmas, seed=seed, callback=callback)
