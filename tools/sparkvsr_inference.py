from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import imageio.v3 as iio
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import set_seed

from diffusers import CogVideoXDPMScheduler, CogVideoXImageToVideoPipeline
from diffusers.models.embeddings import get_3d_rotary_pos_embed

import decord  # isort:skip

decord.bridge.set_bridge("torch")

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    try:
        total = max(0, int(round(float(seconds))))
    except Exception:
        return "unknown"
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def emit_progress(
    pct: float,
    phase: str,
    detail: str = "",
    *,
    step: Optional[int] = None,
    total: Optional[int] = None,
    started_at: Optional[float] = None,
) -> None:
    pct = max(0.0, min(1.0, float(pct)))
    parts = [f"SparkVSR Progress: {pct * 100.0:.1f}%", f"phase={phase}"]
    if detail:
        parts.append(str(detail))
    if step is not None and total is not None:
        parts.append(f"step={int(step)}/{max(1, int(total))}")
    if started_at is not None:
        elapsed = max(0.0, time.monotonic() - started_at)
        parts.append(f"elapsed={format_duration(elapsed)}")
        if pct > 0.01 and pct < 0.999:
            eta = (elapsed / pct) - elapsed
            parts.append(f"eta={format_duration(eta)}")
    print(" | ".join(parts), flush=True)


def no_grad(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return wrapper


def is_video_file(filename: str | os.PathLike) -> bool:
    return str(filename).lower().endswith(VIDEO_EXTENSIONS)


def image_to_tensor(path: str | os.PathLike) -> torch.Tensor:
    import numpy as np

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def center_crop_to_aspect_ratio(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, src_h, src_w = tensor.shape
    target_ar = target_w / target_h
    src_ar = src_w / src_h
    if abs(target_ar - src_ar) < 1e-3:
        return tensor
    if src_ar > target_ar:
        new_w = int(src_h * target_ar)
        start_w = max(0, (src_w - new_w) // 2)
        return tensor[:, :, start_w : start_w + new_w]
    new_h = int(src_w / target_ar)
    start_h = max(0, (src_h - new_h) // 2)
    return tensor[:, start_h : start_h + new_h, :]


def interpolate_2d(video: torch.Tensor, size: Tuple[int, int], mode: str) -> torch.Tensor:
    mode = str(mode or "bilinear").lower()
    if mode == "nearest":
        return torch.nn.functional.interpolate(video, size=size, mode=mode)
    return torch.nn.functional.interpolate(video, size=size, mode=mode, align_corners=False)


def save_frames_as_png(video: torch.Tensor, output_dir: str | os.PathLike) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = (video[0].permute(1, 2, 3, 0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    for idx, frame in enumerate(frames):
        Image.fromarray(frame).save(output / f"{idx:05d}.png")


def save_video_with_imageio(video: torch.Tensor, output_path: str | os.PathLike, fps: float, pixel_format: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = (video[0].permute(1, 2, 3, 0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    pix_fmt = "yuv444p" if str(pixel_format).lower() == "yuv444p" else "yuv420p"
    crf = "0" if pix_fmt == "yuv444p" else "10"
    iio.imwrite(
        path,
        frames,
        fps=max(1.0, float(fps or 30.0)),
        codec="libx264",
        pixelformat=pix_fmt,
        macro_block_size=None,
        ffmpeg_params=["-crf", crf],
    )


def preprocess_video_match(
    video_path: str | os.PathLike,
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    is_match: bool = True,
) -> Tuple[torch.Tensor, int, int, int, Tuple[int, int, int, int]]:
    video_reader = decord.VideoReader(uri=Path(video_path).as_posix())
    total_frames = len(video_reader)
    start = max(0, int(start_frame) if start_frame is not None else 0)
    end_raw = int(end_frame) if end_frame is not None else -1
    end = total_frames if end_raw < 0 else min(total_frames, end_raw + 1)
    if start >= end:
        raise ValueError(f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}, total={total_frames}")

    frames = video_reader.get_batch(list(range(start, end)))
    frame_count, height, width, channels = frames.shape
    original_shape = (int(frame_count), int(height), int(width), int(channels))

    pad_f = 0
    pad_h = 0
    pad_w = 0

    if is_match:
        remainder = (frame_count - 1) % 8
        if remainder != 0:
            pad_f = 8 - remainder
            frames = torch.cat([frames, frames[-1:].repeat(pad_f, 1, 1, 1)], dim=0)
        pad_h = (4 - height % 4) % 4
        pad_w = (4 - width % 4) % 4
        if pad_h > 0 or pad_w > 0:
            frames = torch.nn.functional.pad(frames, pad=(0, 0, 0, pad_w, 0, pad_h))

    return frames.float().permute(0, 3, 1, 2).contiguous(), pad_f, pad_h, pad_w, original_shape


def remove_padding_and_extra_frames(video: torch.Tensor, pad_f: int, pad_h: int, pad_w: int) -> torch.Tensor:
    if pad_f > 0:
        video = video[:, :, :-pad_f, :, :]
    if pad_h > 0:
        video = video[:, :, :, :-pad_h, :]
    if pad_w > 0:
        video = video[:, :, :, :, :-pad_w]
    return video


def make_temporal_chunks(frame_count: int, chunk_len: int, overlap_t: int = 8) -> List[Tuple[int, int]]:
    if int(chunk_len or 0) <= 0:
        return [(0, int(frame_count))]
    chunk = int(chunk_len)
    overlap = int(overlap_t or 0)
    effective_stride = chunk - overlap
    if effective_stride <= 0:
        raise ValueError("chunk_len must be greater than overlap_t")
    chunk_starts = list(range(0, frame_count - overlap, effective_stride))
    if not chunk_starts:
        chunk_starts = [0]
    if chunk_starts[-1] + chunk < frame_count:
        chunk_starts.append(max(0, frame_count - chunk))
    time_chunks = [(s, min(s + chunk, frame_count)) for s in chunk_starts]
    if len(time_chunks) >= 2 and time_chunks[-1][1] - time_chunks[-1][0] < chunk:
        last = time_chunks.pop()
        prev_start, _ = time_chunks[-1]
        time_chunks[-1] = (prev_start, last[1])
    return time_chunks


def make_spatial_tiles(height: int, width: int, tile_size_hw: Tuple[int, int], overlap_hw: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
    tile_h, tile_w = int(tile_size_hw[0]), int(tile_size_hw[1])
    overlap_h, overlap_w = int(overlap_hw[0]), int(overlap_hw[1])
    if tile_h <= 0 or tile_w <= 0:
        return [(0, height, 0, width)]
    stride_h = tile_h - overlap_h
    stride_w = tile_w - overlap_w
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError("Tile size must be greater than overlap")

    h_tiles = list(range(0, max(1, height - overlap_h), stride_h))
    if not h_tiles or h_tiles[-1] + tile_h < height:
        h_tiles.append(max(0, height - tile_h))
    if len(h_tiles) >= 2 and h_tiles[-1] + tile_h > height:
        h_tiles.pop()

    w_tiles = list(range(0, max(1, width - overlap_w), stride_w))
    if not w_tiles or w_tiles[-1] + tile_w < width:
        w_tiles.append(max(0, width - tile_w))
    if len(w_tiles) >= 2 and w_tiles[-1] + tile_w > width:
        w_tiles.pop()

    tiles: List[Tuple[int, int, int, int]] = []
    for h_start in h_tiles:
        h_end = min(h_start + tile_h, height)
        if h_end + stride_h > height:
            h_end = height
        for w_start in w_tiles:
            w_end = min(w_start + tile_w, width)
            if w_end + stride_w > width:
                w_end = width
            tiles.append((h_start, h_end, w_start, w_end))
    return tiles


def get_valid_tile_region(
    t_start: int,
    t_end: int,
    h_start: int,
    h_end: int,
    w_start: int,
    w_end: int,
    video_shape: Tuple[int, int, int, int, int],
    overlap_t: int,
    overlap_h: int,
    overlap_w: int,
) -> Dict[str, int]:
    _, _, frame_count, height, width = video_shape
    t_len = t_end - t_start
    h_len = h_end - h_start
    w_len = w_end - w_start
    valid_t_start = 0 if t_start == 0 else overlap_t // 2
    valid_t_end = t_len if t_end == frame_count else t_len - overlap_t // 2
    valid_h_start = 0 if h_start == 0 else overlap_h // 2
    valid_h_end = h_len if h_end == height else h_len - overlap_h // 2
    valid_w_start = 0 if w_start == 0 else overlap_w // 2
    valid_w_end = w_len if w_end == width else w_len - overlap_w // 2
    return {
        "valid_t_start": valid_t_start,
        "valid_t_end": valid_t_end,
        "valid_h_start": valid_h_start,
        "valid_h_end": valid_h_end,
        "valid_w_start": valid_w_start,
        "valid_w_end": valid_w_end,
        "out_t_start": t_start + valid_t_start,
        "out_t_end": t_start + valid_t_end,
        "out_h_start": h_start + valid_h_start,
        "out_h_end": h_start + valid_h_end,
        "out_w_start": w_start + valid_w_start,
        "out_w_end": w_start + valid_w_end,
    }


def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))
    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))
    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    transformer_config,
    vae_scale_factor_spatial: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
    grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)
    p = transformer_config.patch_size
    p_t = transformer_config.patch_size_t
    base_size_width = transformer_config.sample_width // p
    base_size_height = transformer_config.sample_height // p
    if p_t is None:
        grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
        return get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            device=device,
        )
    base_num_frames = (num_frames + p_t - 1) // p_t
    return get_3d_rotary_pos_embed(
        embed_dim=transformer_config.attention_head_dim,
        crops_coords=None,
        grid_size=(grid_height, grid_width),
        temporal_size=base_num_frames,
        grid_type="slice",
        max_size=(max(base_size_height, grid_height), max(base_size_width, grid_width)),
        device=device,
    )


@no_grad
def process_video_ref_i2v(
    pipe: CogVideoXImageToVideoPipeline,
    video: torch.Tensor,
    prompt: str = "",
    ref_frames: Optional[List[torch.Tensor]] = None,
    ref_indices: Optional[List[int]] = None,
    chunk_start_idx: int = 0,
    noise_step: int = 0,
    sr_noise_step: int = 399,
    empty_prompt_embedding: Optional[torch.Tensor] = None,
    ref_guidance_scale: float = 1.0,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> torch.Tensor:
    def mark(frac: float, detail: str) -> None:
        if progress_cb:
            progress_cb(max(0.0, min(1.0, float(frac))), detail)

    ref_frames = ref_frames or []
    ref_indices = ref_indices or []
    mark(0.02, "moving tile tensors to device")
    execution_device = getattr(pipe, "_execution_device", None) or pipe.device
    video = video.to(execution_device, dtype=pipe.dtype)
    mark(0.08, "encoding low-resolution video latents")
    latent_dist = pipe.vae.encode(video).latent_dist
    lq_latent = latent_dist.sample() * pipe.vae.config.scaling_factor
    batch_size, _, num_frames, height, width = lq_latent.shape
    device = lq_latent.device
    dtype = lq_latent.dtype

    mark(0.18, "encoding reference keyframes")
    full_ref_latent = torch.zeros_like(lq_latent)
    for i, idx in enumerate(ref_indices):
        if i >= len(ref_frames):
            break
        local_frame_idx = int(idx) - int(chunk_start_idx)
        target_lat_idx = local_frame_idx // 4
        if 0 <= target_lat_idx < num_frames:
            r_frame = ref_frames[i].to(device, dtype=dtype)
            chunk = r_frame.unsqueeze(0).unsqueeze(2).repeat(1, 1, 4, 1, 1)
            lat = pipe.vae.encode(chunk).latent_dist.sample() * pipe.vae.config.scaling_factor
            full_ref_latent[:, :, target_lat_idx, :, :] = lat[0, :, 0, :, :]

    mark(0.28, "preparing conditioning")
    do_cfg = abs(float(ref_guidance_scale) - 1.0) > 1e-3
    if do_cfg:
        input_latent_cond = torch.cat([lq_latent, full_ref_latent], dim=1)
        input_latent_uncond = torch.cat([lq_latent, torch.zeros_like(full_ref_latent)], dim=1)
        input_latent = torch.cat([input_latent_uncond, input_latent_cond], dim=0)
    else:
        input_latent = torch.cat([lq_latent, full_ref_latent], dim=1)

    patch_size_t = pipe.transformer.config.patch_size_t
    ncopy = 0
    if patch_size_t is not None:
        ncopy = input_latent.shape[2] % patch_size_t
        if ncopy:
            input_latent = torch.cat([input_latent[:, :, :1].repeat(1, 1, ncopy, 1, 1), input_latent], dim=2)

    if prompt == "" and empty_prompt_embedding is not None:
        prompt_embedding = empty_prompt_embedding.to(device, dtype=dtype)
        if prompt_embedding.shape[0] != batch_size:
            prompt_embedding = prompt_embedding.repeat(batch_size, 1, 1)
    else:
        mark(0.36, "encoding text prompt")
        prompt_token_ids = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.transformer.config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids
        prompt_embedding = pipe.text_encoder(prompt_token_ids.to(device))[0]
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=dtype)

    latents = input_latent.permute(0, 2, 1, 3, 4)
    if do_cfg:
        prompt_embedding = torch.cat([prompt_embedding, prompt_embedding], dim=0)

    mark(0.42, "preparing noise/timesteps")
    if int(noise_step or 0) != 0:
        lq_part = latents[:, :, :16, :, :]
        ref_part = latents[:, :, 16:, :, :]
        noise = torch.randn_like(lq_part)
        add_timesteps = torch.full((latents.shape[0],), fill_value=int(noise_step), dtype=torch.long, device=device)
        lq_part = pipe.scheduler.add_noise(lq_part.transpose(1, 2), noise.transpose(1, 2), add_timesteps).transpose(1, 2)
        latents = torch.cat([lq_part, ref_part], dim=2)

    timesteps = torch.full((latents.shape[0],), fill_value=int(sr_noise_step), dtype=torch.long, device=device)
    vae_scale_factor_spatial = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    rotary_emb = (
        prepare_rotary_positional_embeddings(
            height=height * vae_scale_factor_spatial,
            width=width * vae_scale_factor_spatial,
            num_frames=num_frames,
            transformer_config=pipe.transformer.config,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            device=device,
        )
        if pipe.transformer.config.use_rotary_positional_embeddings
        else None
    )
    ofs = None
    if pipe.transformer.config.ofs_embed_dim is not None:
        ofs = torch.full((latents.shape[0],), fill_value=2.0, device=device, dtype=dtype)

    mark(0.55, "running SparkVSR transformer")
    predicted_noise = pipe.transformer(
        hidden_states=latents,
        encoder_hidden_states=prompt_embedding,
        timestep=timesteps,
        image_rotary_emb=rotary_emb,
        ofs=ofs,
        return_dict=False,
    )[0]

    mark(0.78, "applying scheduler update")
    predicted_noise_slice = predicted_noise[:, :, :16, :, :].transpose(1, 2)
    lq_sample = latents[:, :, :16, :, :].transpose(1, 2)
    if do_cfg:
        noise_pred_uncond, noise_pred_cond = predicted_noise_slice.chunk(2)
        predicted_noise_slice = noise_pred_uncond + float(ref_guidance_scale) * (noise_pred_cond - noise_pred_uncond)
        lq_sample = lq_sample.chunk(2)[1]
        timesteps = timesteps.chunk(2)[0]

    latent_generate = pipe.scheduler.get_velocity(predicted_noise_slice, lq_sample, timesteps)
    if patch_size_t is not None and ncopy > 0:
        latent_generate = latent_generate[:, :, ncopy:, :, :]

    mark(0.88, "decoding output video tile")
    video_generate = pipe.vae.decode(latent_generate / pipe.vae.config.scaling_factor).sample
    mark(1.0, "tile complete")
    return (video_generate * 0.5 + 0.5).clamp(0.0, 1.0)


def parse_ref_indices(text: Optional[str]) -> Optional[List[int]]:
    if text is None:
        return None
    if isinstance(text, list):
        return [int(x) for x in text]
    values = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(float(part)))
    return values if values else None


def find_empty_prompt_embedding(model_path: Path) -> Optional[torch.Tensor]:
    candidates = [
        Path.cwd() / "pretrained_models" / "prompt_embeddings" / "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors",
        model_path / "prompt_embeddings" / "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors",
        model_path.parent / "prompt_embeddings" / "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors",
    ]
    for candidate in candidates:
        if candidate.exists():
            return load_file(str(candidate))["prompt_embedding"]
    return None


def _align_reference_tensor(tensor: torch.Tensor, target_h: int, target_w: int, mode: str = "bicubic") -> torch.Tensor:
    ref = tensor.float().contiguous()
    if ref.shape[-2:] != (target_h, target_w):
        ref = center_crop_to_aspect_ratio(ref, target_h, target_w)
    if ref.shape[-2:] != (target_h, target_w):
        ref = interpolate_2d(ref.unsqueeze(0), (target_h, target_w), mode).squeeze(0)
    return ref.clamp(-1.0, 1.0)


def _read_reference_video_frame(path: Path, frame_idx: int) -> torch.Tensor:
    reader = decord.VideoReader(uri=path.as_posix())
    if len(reader) <= 0:
        raise ValueError(f"Reference video has no frames: {path}")
    read_idx = min(max(0, int(frame_idx)), len(reader) - 1)
    frame_obj = reader[read_idx]
    if hasattr(frame_obj, "asnumpy"):
        frame_np = frame_obj.asnumpy()
    else:
        frame_np = frame_obj.cpu().numpy()
    tensor = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0
    return tensor * 2.0 - 1.0


def _list_reference_images(path: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(path.glob(f"*{ext}"))
        files.extend(path.glob(f"*{ext.upper()}"))
    return sorted(set(files), key=lambda p: p.name.lower())


def _load_local_sr_reference(
    source_path: str,
    ref_idx: int,
    ref_order: int,
    ref_indices: List[int],
) -> torch.Tensor:
    source = Path(source_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Local SR reference path not found: {source}")

    if source.is_dir():
        images = _list_reference_images(source)
        if not images:
            raise FileNotFoundError(f"No reference images found in directory: {source}")
        if len(images) == len(ref_indices):
            chosen = images[min(ref_order, len(images) - 1)]
        elif len(images) > int(ref_idx):
            chosen = images[int(ref_idx)]
        elif len(images) == 1:
            chosen = images[0]
        else:
            chosen = images[min(ref_order, len(images) - 1)]
        print(f"Using local SR reference image for frame {ref_idx}: {chosen}", flush=True)
        return image_to_tensor(chosen) * 2.0 - 1.0

    if is_video_file(source):
        print(f"Using local SR reference video frame {ref_idx}: {source}", flush=True)
        return _read_reference_video_frame(source, int(ref_idx))

    if source.suffix.lower() in IMAGE_EXTENSIONS:
        if len(ref_indices) > 1:
            print(
                "Warning: one reference image was provided for multiple reference indices; "
                "the same image will be reused.",
                flush=True,
            )
        print(f"Using local SR reference image for frame {ref_idx}: {source}", flush=True)
        return image_to_tensor(source) * 2.0 - 1.0

    raise ValueError(f"Unsupported local SR reference path: {source}")


def build_ref_frames(args, video_name: str, video_path: str, video: torch.Tensor, video_lr: torch.Tensor, original_shape, effective_upscale: int) -> Tuple[List[int], List[torch.Tensor]]:
    from shared.sparkvsr_ref_utils import _select_indices, save_ref_frames_locally

    if args.ref_mode == "no_ref":
        print("Running in No-Ref mode (0 reference frames).", flush=True)
        return [], []

    ref_indices = parse_ref_indices(args.ref_indices)
    if ref_indices is not None:
        ref_indices = sorted(set(ref_indices))
        for i in range(len(ref_indices) - 1):
            if ref_indices[i + 1] - ref_indices[i] < 4:
                raise ValueError("Reference frame interval must be >= 4.")
        print(f"Using manually specified reference indices: {ref_indices}", flush=True)
    else:
        ref_indices = _select_indices(video.shape[2])
        print(f"Using auto-selected reference indices: {ref_indices}", flush=True)

    ref_frames_list: List[torch.Tensor] = []
    target_h, target_w = video.shape[-2], video.shape[-1]

    if args.ref_mode == "gt":
        saved = save_ref_frames_locally(
            video_path=video_path,
            output_dir=os.path.join(args.output_path, "ref_gt_cache", Path(video_name).stem),
            video_id=Path(video_name).stem,
            is_match=True,
            specific_indices=ref_indices,
        )
        for idx in ref_indices:
            match = next((path for saved_idx, path in saved if saved_idx == idx), None)
            if match:
                t_img = image_to_tensor(match) * 2.0 - 1.0
                if t_img.shape[-2:] != (target_h, target_w):
                    gt_h, gt_w = t_img.shape[-2], t_img.shape[-1]
                    orig_h_up = original_shape[1] * effective_upscale
                    orig_w_up = original_shape[2] * effective_upscale
                    if gt_h == orig_h_up and gt_w == orig_w_up:
                        t_img = torch.nn.functional.pad(t_img, (0, target_w - gt_w, 0, target_h - gt_h), mode="replicate")
                    else:
                        t_img = interpolate_2d(t_img.unsqueeze(0), (target_h, target_w), "bilinear").squeeze(0)
                ref_frames_list.append(t_img)
            else:
                print(f"Warning: GT frame {idx} not found. Using LQ frame.", flush=True)
                ref_frames_list.append(video[0, :, idx])
        return ref_indices, ref_frames_list

    if args.ref_mode == "sr_image":
        ref_source_path = str(args.ref_source_path or "").strip()
        if not ref_source_path:
            ref_source_path = str(video_path)
            print(
                "No local SR reference path was provided. "
                "Using the input video as the local reference source; for best quality, "
                "provide a locally upscaled keyframe/video or use PiSA-SR.",
                flush=True,
            )
        for order, idx in enumerate(ref_indices):
            ref = _load_local_sr_reference(ref_source_path, idx, order, ref_indices)
            ref_frames_list.append(_align_reference_tensor(ref, target_h, target_w, mode="bicubic"))
        return ref_indices, ref_frames_list

    if args.ref_mode == "pisasr":
        cache_dir = Path(args.ref_pisa_cache_dir or Path(args.output_path) / "ref_pisasr_cache") / Path(video_name).stem
        cache_dir.mkdir(parents=True, exist_ok=True)
        required = [args.pisa_python_executable, args.pisa_script_path, args.pisa_sd_model_path, args.pisa_chkpt_path]
        if not all(required):
            raise ValueError("PiSA-SR mode requires pisa_python_executable, pisa_script_path, pisa_sd_model_path, and pisa_chkpt_path.")
        for idx in ref_indices:
            frame_path = cache_dir / f"{Path(video_name).stem}_frame_{idx:05d}.png"
            if not frame_path.exists():
                lr_frame = video_lr[min(idx, video_lr.shape[0] - 1)].cpu().permute(1, 2, 0).numpy()
                with tempfile.TemporaryDirectory(prefix="sparkvsr_pisa_") as tmpdir:
                    lr_path = Path(tmpdir) / "input_frame.png"
                    Image.fromarray(lr_frame.astype("uint8")).save(lr_path)
                    out_dir = Path(tmpdir) / "out"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        args.pisa_python_executable,
                        args.pisa_script_path,
                        "--input_image",
                        str(lr_path),
                        "--output_dir",
                        str(out_dir),
                        "--pretrained_model_path",
                        args.pisa_sd_model_path,
                        "--pretrained_path",
                        args.pisa_chkpt_path,
                        "--upscale",
                        str(args.upscale),
                        "--align_method",
                        "adain",
                        "--lambda_pix",
                        "1.0",
                        "--lambda_sem",
                        "1.0",
                    ]
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(args.pisa_gpu or "0")
                    subprocess.run(cmd, env=env, check=True, cwd=str(Path(args.pisa_script_path).parent))
                    generated = out_dir / "input_frame.png"
                    if generated.exists():
                        frame_path.write_bytes(generated.read_bytes())
            if frame_path.exists():
                t_img = image_to_tensor(frame_path) * 2.0 - 1.0
                if t_img.shape[-2:] != (target_h, target_w):
                    t_img = interpolate_2d(t_img.unsqueeze(0), (target_h, target_w), "bilinear").squeeze(0)
                ref_frames_list.append(t_img)
            else:
                print(f"Warning: PiSA-SR frame {idx} not generated. Using LQ frame.", flush=True)
                ref_frames_list.append(video[0, :, idx])
        return ref_indices, ref_frames_list

    raise ValueError(f"Unsupported ref_mode: {args.ref_mode}")


def collect_video_files(input_dir: str) -> List[str]:
    path = Path(input_dir)
    if path.is_file():
        if is_video_file(path) or path.suffix.lower() in IMAGE_EXTENSIONS:
            return [str(path)]
        raise ValueError(f"Unsupported input file: {path}")
    files: List[str] = []
    for ext in VIDEO_EXTENSIONS:
        files.extend(glob.glob(str(path / f"*{ext}")))
    return sorted(files)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SparkVSR inference entrypoint for SECourses.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--input_json", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--upscale_mode", type=str, default="bilinear", choices=["bilinear", "bicubic", "nearest"])
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--output_resolution", type=int, nargs=2, default=None, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--noise_step", type=int, default=0)
    parser.add_argument("--sr_noise_step", type=int, default=399)
    parser.add_argument("--is_cpu_offload", action="store_true")
    parser.add_argument("--is_vae_st", action="store_true")
    parser.add_argument("--group_offload", action="store_true")
    parser.add_argument("--num_blocks_per_group", type=int, default=1)
    parser.add_argument("--png_save", action="store_true")
    parser.add_argument("--save_format", type=str, default="yuv444p", choices=["yuv444p", "yuv420p"])
    parser.add_argument("--tile_size_hw", type=int, nargs=2, default=(0, 0), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--overlap_hw", type=int, nargs=2, default=(32, 32), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--chunk_len", type=int, default=0)
    parser.add_argument("--overlap_t", type=int, default=8)
    parser.add_argument("--ref_mode", type=str, default="sr_image", choices=["sr_image", "pisasr", "no_ref", "gt"])
    parser.add_argument("--ref_indices", type=str, default="")
    parser.add_argument("--ref_guidance_scale", type=float, default=1.0)
    parser.add_argument("--ref_source_path", type=str, default="")
    parser.add_argument("--ref_pisa_cache_dir", type=str, default="")
    parser.add_argument("--pisa_python_executable", type=str, default="")
    parser.add_argument("--pisa_script_path", type=str, default="")
    parser.add_argument("--pisa_sd_model_path", type=str, default="")
    parser.add_argument("--pisa_chkpt_path", type=str, default="")
    parser.add_argument("--pisa_gpu", type=str, default="0")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def main() -> int:
    run_started_at = time.monotonic()
    args = build_parser().parse_args()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    set_seed(int(args.seed or 0))

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"SparkVSR model path not found: {model_path}")
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_dict: Dict[str, str] = {}
    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as handle:
            prompt_dict = json.load(handle)

    video_files = collect_video_files(args.input_dir)
    if not video_files:
        raise FileNotFoundError(f"No supported video files found in {args.input_dir}")

    print(f"Loading SparkVSR model from {model_path}", flush=True)
    emit_progress(0.005, "startup", "validated inputs", started_at=run_started_at)
    emit_progress(0.01, "model_load", f"loading pipeline from {model_path}", started_at=run_started_at)
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(str(model_path), torch_dtype=dtype, low_cpu_mem_usage=True)
    emit_progress(0.10, "model_load", "pipeline weights loaded", started_at=run_started_at)
    if args.lora_path:
        print(f"Loading LoRA from {args.lora_path}", flush=True)
        emit_progress(0.105, "lora", f"loading LoRA {args.lora_path}", started_at=run_started_at)
        pipe.load_lora_weights(args.lora_path, adapter_name="dove_ref_i2v")
        pipe.fuse_lora(lora_scale=1.0)
        emit_progress(0.11, "lora", "LoRA fused", started_at=run_started_at)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    device = str(args.device or "cuda").lower()
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.", flush=True)
        device = "cpu"
    if args.is_cpu_offload and device != "cpu":
        emit_progress(0.115, "memory", "enabling CPU offload", started_at=run_started_at)
        pipe.enable_model_cpu_offload(device=device)
    else:
        emit_progress(0.115, "memory", f"moving pipeline to {device}", started_at=run_started_at)
        pipe.to(device)

    if args.group_offload and device != "cpu":
        try:
            from diffusers.hooks import apply_group_offloading

            apply_group_offloading(
                pipe.transformer,
                onload_device=torch.device("cuda"),
                offload_type="block_level",
                num_blocks_per_group=max(1, int(args.num_blocks_per_group or 1)),
            )
            print(f"Group offload enabled: num_blocks_per_group={args.num_blocks_per_group}", flush=True)
        except Exception as exc:
            print(f"Group offload unavailable: {exc}", flush=True)

    if args.is_vae_st:
        emit_progress(0.12, "memory", "enabling VAE slicing/tiling", started_at=run_started_at)
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    empty_prompt_embedding = find_empty_prompt_embedding(model_path)
    emit_progress(0.13, "ready", "model ready; starting videos", started_at=run_started_at)

    for index, video_path in enumerate(video_files, 1):
        video_base = 0.13 + (float(index - 1) / max(1, len(video_files))) * 0.86
        video_span = 0.86 / max(1, len(video_files))

        def video_pct(local: float) -> float:
            return video_base + max(0.0, min(1.0, float(local))) * video_span

        video_name = Path(video_path).name
        prompt = prompt_dict.get(video_name, "")
        print(f"[{index}/{len(video_files)}] Reading {video_name}", flush=True)
        emit_progress(
            video_pct(0.02),
            "read_input",
            f"video {index}/{len(video_files)}: {video_name}",
            step=index,
            total=len(video_files),
            started_at=run_started_at,
        )
        video, pad_f, pad_h, pad_w, original_shape = preprocess_video_match(
            video_path,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            is_match=True,
        )
        h_orig, w_orig = int(video.shape[2]), int(video.shape[3])
        emit_progress(
            video_pct(0.07),
            "prepare_input",
            f"decoded {int(video.shape[0])} frame(s), source={w_orig}x{h_orig}",
            step=index,
            total=len(video_files),
            started_at=run_started_at,
        )

        if args.output_resolution is not None:
            target_h, target_w = int(args.output_resolution[0]), int(args.output_resolution[1])
            scale_factor = max(target_h / h_orig, target_w / w_orig)
            scaled_h = int(h_orig * scale_factor)
            scaled_w = int(w_orig * scale_factor)
            print(f"Output resolution mode: {target_h}x{target_w}", flush=True)
            video_up = interpolate_2d(video, (scaled_h, scaled_w), args.upscale_mode)
            crop_top = max(0, (scaled_h - target_h) // 2)
            crop_left = max(0, (scaled_w - target_w) // 2)
            video_up = video_up[:, :, crop_top : crop_top + target_h, crop_left : crop_left + target_w]
            pad_h_extra = (8 - target_h % 8) % 8
            pad_w_extra = (8 - target_w % 8) % 8
            if pad_h_extra > 0 or pad_w_extra > 0:
                video_up = torch.nn.functional.pad(video_up, (0, pad_w_extra, 0, pad_h_extra))
            effective_upscale = 1
            emit_progress(
                video_pct(0.11),
                "resize_input",
                f"resized/cropped to {target_w}x{target_h}",
                step=index,
                total=len(video_files),
                started_at=run_started_at,
            )
        else:
            effective_upscale = max(1, int(args.upscale or 4))
            video_up = interpolate_2d(video, (h_orig * effective_upscale, w_orig * effective_upscale), args.upscale_mode)
            emit_progress(
                video_pct(0.11),
                "resize_input",
                f"upscaled input to {w_orig * effective_upscale}x{h_orig * effective_upscale} ({effective_upscale}x)",
                step=index,
                total=len(video_files),
                started_at=run_started_at,
            )

        video_lr = video
        video = (video_up / 255.0 * 2.0) - 1.0
        video = video.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()

        emit_progress(video_pct(0.13), "references", f"preparing reference mode: {args.ref_mode}", step=index, total=len(video_files), started_at=run_started_at)
        ref_indices, ref_frames_list = build_ref_frames(args, video_name, video_path, video, video_lr, original_shape, effective_upscale)
        emit_progress(
            video_pct(0.16),
            "references",
            f"reference frames ready: {len(ref_frames_list)}",
            step=index,
            total=len(video_files),
            started_at=run_started_at,
        )

        _, _, frame_count, height, width = video.shape
        overlap_t = int(args.overlap_t if args.chunk_len > 0 else 0)
        tile_size_hw = (int(args.tile_size_hw[0]), int(args.tile_size_hw[1]))
        overlap_hw = (int(args.overlap_hw[0]), int(args.overlap_hw[1])) if tile_size_hw != (0, 0) else (0, 0)
        time_chunks = make_temporal_chunks(frame_count, int(args.chunk_len), overlap_t)
        spatial_tiles = make_spatial_tiles(height, width, tile_size_hw, overlap_hw)
        output_video = torch.zeros_like(video)
        print(
            f"Processing: F={frame_count} H={height} W={width} | "
            f"Chunks={len(time_chunks)} Tiles={len(spatial_tiles)}",
            flush=True,
        )
        emit_progress(
            video_pct(0.18),
            "processing",
            f"frames={frame_count}, output={width}x{height}, chunks={len(time_chunks)}, tiles={len(spatial_tiles)}",
            step=index,
            total=len(video_files),
            started_at=run_started_at,
        )

        total_steps = max(1, len(time_chunks) * len(spatial_tiles))
        step_idx = 0
        for t_start, t_end in time_chunks:
            for h_start, h_end, w_start, w_end in spatial_tiles:
                step_idx += 1
                tile_start_local = 0.18 + ((step_idx - 1) / total_steps) * 0.68
                pct = (step_idx - 1) / total_steps * 100.0
                print(
                    f"Processing Tiles: {step_idx}/{total_steps} | "
                    f"t={t_start}:{t_end} h={h_start}:{h_end} w={w_start}:{w_end} ({pct:.1f}%)",
                    flush=True,
                )
                emit_progress(
                    video_pct(tile_start_local),
                    "tile",
                    f"tile {step_idx}/{total_steps}: frames {t_start}:{t_end}, h {h_start}:{h_end}, w {w_start}:{w_end}",
                    step=step_idx,
                    total=total_steps,
                    started_at=run_started_at,
                )
                video_chunk = video[:, :, t_start:t_end, h_start:h_end, w_start:w_end]
                current_ref_frames = [rf[:, h_start:h_end, w_start:w_end] for rf in ref_frames_list]

                def tile_progress(local_frac: float, detail: str) -> None:
                    tile_local = 0.18 + (((step_idx - 1) + max(0.0, min(1.0, float(local_frac)))) / total_steps) * 0.68
                    emit_progress(
                        video_pct(tile_local),
                        "tile",
                        f"tile {step_idx}/{total_steps}: {detail}",
                        step=step_idx,
                        total=total_steps,
                        started_at=run_started_at,
                    )

                generated = process_video_ref_i2v(
                    pipe=pipe,
                    video=video_chunk,
                    prompt=prompt,
                    ref_frames=current_ref_frames,
                    ref_indices=ref_indices,
                    chunk_start_idx=t_start,
                    noise_step=args.noise_step,
                    sr_noise_step=args.sr_noise_step,
                    empty_prompt_embedding=empty_prompt_embedding,
                    ref_guidance_scale=args.ref_guidance_scale,
                    progress_cb=tile_progress,
                )
                region = get_valid_tile_region(
                    t_start,
                    t_end,
                    h_start,
                    h_end,
                    w_start,
                    w_end,
                    video.shape,
                    overlap_t,
                    overlap_hw[0],
                    overlap_hw[1],
                )
                output_video[
                    :,
                    :,
                    region["out_t_start"] : region["out_t_end"],
                    region["out_h_start"] : region["out_h_end"],
                    region["out_w_start"] : region["out_w_end"],
                ] = generated[
                    :,
                    :,
                    region["valid_t_start"] : region["valid_t_end"],
                    region["valid_h_start"] : region["valid_h_end"],
                    region["valid_w_start"] : region["valid_w_end"],
                ]
                emit_progress(
                    video_pct(0.18 + (step_idx / total_steps) * 0.68),
                    "tile",
                    f"tile {step_idx}/{total_steps} merged",
                    step=step_idx,
                    total=total_steps,
                    started_at=run_started_at,
                )

        emit_progress(video_pct(0.88), "postprocess", "removing padding and extra frames", step=index, total=len(video_files), started_at=run_started_at)
        video_generate = remove_padding_and_extra_frames(
            output_video,
            pad_f,
            pad_h * int(effective_upscale),
            pad_w * int(effective_upscale),
        )

        if args.output_file:
            out_file_path = Path(args.output_file)
        else:
            out_file_path = output_dir / Path(video_name).with_suffix(".mp4").name
        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        if args.png_save:
            png_dir = out_file_path.with_suffix("")
            emit_progress(video_pct(0.92), "save_png", f"saving PNG frames to {png_dir}", step=index, total=len(video_files), started_at=run_started_at)
            save_frames_as_png(video_generate, png_dir)
            print(f"PNG frames saved: {png_dir}", flush=True)
        emit_progress(video_pct(0.95), "encode_output", f"writing video {out_file_path}", step=index, total=len(video_files), started_at=run_started_at)
        save_video_with_imageio(video_generate, out_file_path, fps=args.fps, pixel_format=args.save_format)
        print(f"Output saved: {out_file_path}", flush=True)
        emit_progress(video_pct(1.0), "video_complete", f"saved {out_file_path}", step=index, total=len(video_files), started_at=run_started_at)

    print("SparkVSR complete.", flush=True)
    emit_progress(1.0, "complete", "SparkVSR complete", started_at=run_started_at)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
