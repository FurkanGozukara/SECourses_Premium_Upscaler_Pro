"""Bounded-lookahead collaboration with immediate, ordered frame delivery."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice

import torch
import torch.nn.functional as F

from .timing import DeviceSpan


@dataclass(frozen=True)
class InputFrame:
    index: int
    rgb: torch.Tensor  # uint8 CHW, native LR
    source_time: float = 0.0  # seconds relative to the session origin
    arrival_time: float = 0.0


@dataclass(frozen=True)
class OutputFrame:
    index: int
    rgb: torch.Tensor  # GPU/CPU float BCHW in [-1,1]
    source_time: float
    arrival_time: float
    window_start: int
    window_end: int
    timings: tuple[tuple[str, DeviceSpan], ...] = ()


def prepare_lr(frame: torch.Tensor, *, device, dtype, stride: int = 8):
    """Normalize native RGB and pad for the Wan latent/patch grid."""
    if frame.ndim != 3 or frame.shape[0] != 3 or frame.dtype != torch.uint8:
        raise ValueError("Streaming input must be uint8 RGB [3,H,W].")
    h, w = frame.shape[-2:]
    if min(h, w) <= 0:
        raise ValueError("Frame dimensions must be positive.")
    x = frame.unsqueeze(0).to(device=device, non_blocking=True).to(dtype)
    x = x.div(127.5).sub(1.0)
    return F.pad(x, (0, (-w) % stride, 0, (-h) % stride), mode="replicate")


class CollaborativeFrameStream:
    """Model-only generator shared by deployment and paced benchmarking.

    It buffers at most K native frames, retains four (configured) Wan keyframes,
    and resets Flash video history at each overlapping endpoint window. Yielding
    suspends before the next Flash decode, so downstream sees individual frames.
    """

    def __init__(
        self,
        *,
        wan_model,
        flash_model,
        wan_pipeline,
        prompt_embeds,
        device,
        compute_dtype,
        window_size=16,
        seed=42,
        reuse_endpoint_kv=True,
        precompute_rope=True,
        quantized_kv=False,
    ):
        if window_size < 2:
            raise ValueError("Collaborative window_size must be at least 2.")
        if flash_model.condition_mode != "endpoints":
            raise ValueError("Collaborative streaming requires endpoint conditions.")
        if flash_model.training or wan_model.training:
            raise ValueError("Streaming models must be in eval() mode.")
        self.wan = wan_model
        self.flash = flash_model
        self.scale = int(flash_model.upscale_factor)
        self.latent_stride = 16 // self.scale
        self.wan_input_scale = 4 if self.scale == 4 else 1
        self.pipeline = wan_pipeline
        self.prompt = prompt_embeds
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = compute_dtype
        self.window_size = int(window_size)
        self.seed = seed
        self.reuse_endpoint_kv = reuse_endpoint_kv
        self.precompute_rope = precompute_rope
        self.quantized_kv = quantized_kv
        self.last_keyframe_indices: deque[int] = deque(maxlen=10000)
        self.keyframe_count = 0
        self.capture_state = False

    @property
    def stride(self):
        patch = self.wan.config.patch_size
        return self.latent_stride * max(int(patch[1]), int(patch[2]))

    @torch.no_grad()
    def prepare(self, height, width):
        stride = self.stride
        geometry = (
            height + (-height) % stride,
            width + (-width) % stride,
            self.window_size,
            self.dtype,
            self.device,
        )
        if self.precompute_rope and getattr(self, "_prepared_geometry", None) != geometry:
            self.flash.prepare_streaming(geometry[0], geometry[1], self.window_size)
            self._prepared_geometry = geometry

    @torch.no_grad()
    def frames(self, source: Iterable[InputFrame], *, resume=None, checkpoint=False) -> Iterator[OutputFrame]:
        iterator = iter(source)
        first = next(iterator, None)
        if first is None:
            raise ValueError("Cannot infer an empty stream.")
        expected = int(resume["frame"]) if resume else 0
        if first.index != expected:
            raise ValueError(f"Stream must start at frame {expected}.")
        shape = tuple(first.rgb.shape)
        h, w = shape[-2:]
        stride = self.stride
        ph, pw = h + (-h) % stride, w + (-w) % stride
        self.prepare(h, w)
        wan_state = self.pipeline.create_streaming_state(
            self.wan, **({"quantized_kv": True} if self.quantized_kv else {})
        )
        generator = (
            None
            if self.seed is None
            else [torch.Generator(device=self.device).manual_seed(self.seed)]
        )
        self.last_keyframe_indices.clear()
        self.keyframe_count = 0
        self._checkpoint = None
        first_window = resume is None
        if resume:
            model_state = wan_state.model_state
            model_state.frame_index = resume["wan_frame"]
            model_state.grid_signature = (*resume["grid_shape"], self.device, self.dtype)
            if len(model_state.block_caches) != len(resume["caches"]):
                raise ValueError("RelayVSR checkpoint has a different number of Wan blocks.")
            for block, cache, saved in zip(self.wan.blocks, model_state.block_caches, resume["caches"]):
                cache.next_frame = saved["next_frame"]
                cache.tokens_per_frame = saved["tokens"]
                target = "cpu" if getattr(block, "_relay_swapped", False) else self.device
                capacity = cache.window_size * cache.tokens_per_frame
                storage = []
                for item in saved["storage"]:
                    buffer_shape = (item.shape[0], capacity, *item.shape[2:])
                    buffer = torch.empty(buffer_shape, dtype=item.dtype, device=target)
                    buffer[:, :item.shape[1]].copy_(item)
                    storage.append(buffer)
                cache.move_storage(tuple(storage))
            if generator:
                generator[0].set_state(resume["rng"])
            self.keyframe_count = model_state.frame_index

        pending_timings = []

        def keyframe(packet):
            span = DeviceSpan(self.device)
            native = prepare_lr(packet.rgb, device=self.device, dtype=self.dtype, stride=stride)
            wan_input = native
            if self.wan_input_scale != 1:
                wan_input = F.interpolate(native, scale_factor=self.wan_input_scale, mode="bilinear", align_corners=False)
            pending_timings.append(("wan_input", span.finish()))
            span = DeviceSpan(self.device)
            latent = self.pipeline.sample_streaming_latent_step(
                model=self.wan,
                lq_frame=wan_input.unsqueeze(2),
                spatial_scale_factor=self.latent_stride * self.wan_input_scale,
                prompt_embeds=self.prompt,
                streaming_state=wan_state,
                generator=generator,
                device=self.device,
                compute_dtype=self.dtype,
            )[:, :, 0].to(self.dtype)
            pending_timings.append(("wan", span.finish()))
            self.last_keyframe_indices.append(packet.index)
            self.keyframe_count += 1
            if self.capture_state:
                self.last_wan_latent = latent
            span = DeviceSpan(self.device)
            prepared = self.flash.prepare_condition(latent) if self.reuse_endpoint_kv else None
            pending_timings.append(("flash_condition", span.finish()))
            return latent, prepared

        # First condition can be computed while the producer collects lookahead.
        if resume:
            start_latent = resume["latent"].to(device=self.device, dtype=self.dtype)
            start_prepared = self.flash.prepare_condition(start_latent) if self.reuse_endpoint_kv else None
            resume.clear()
        else:
            start_latent, start_prepared = keyframe(first)
        start = first
        while True:
            window = [start, *islice(iterator, self.window_size - 1)]
            for offset, packet in enumerate(window):
                if packet.index != start.index + offset or tuple(packet.rgb.shape) != shape:
                    raise ValueError(
                        "Streaming frames must be consecutive with constant dimensions."
                    )
            if len(window) == 1 and not first_window:
                return
            end = window[-1]
            if end.index == start.index:
                end_latent, end_prepared = start_latent, start_prepared
            else:
                end_latent, end_prepared = keyframe(end)
            count = max(2, len(window))
            span = DeviceSpan(self.device)
            state = (
                self.flash.begin_stream(
                    start_prepared,
                    end_prepared,
                    total_frames=count,
                    input_height=ph,
                    input_width=pw,
                )
                if self.reuse_endpoint_kv
                else None
            )
            pending_timings.append(("flash_condition", span.finish()))
            for offset, packet in enumerate(window):
                span = DeviceSpan(self.device)
                native = prepare_lr(packet.rgb, device=self.device, dtype=self.dtype, stride=stride)
                kwargs = {}
                if offset == 0 and not self.reuse_endpoint_kv:
                    kwargs = {
                        "condition_latent": start_latent,
                        "end_condition_latent": end_latent,
                        "total_frames": count,
                    }
                pending_timings.append(("flash_input", span.finish()))
                span = DeviceSpan(self.device)
                prediction, state = self.flash.decode_step(native, state, **kwargs)
                pending_timings.append(("flash", span.finish()))
                if self.capture_state:
                    self.last_flash_state = state
                # The overlap must update history, but has already been emitted.
                if first_window or offset:
                    timings = tuple(pending_timings)
                    pending_timings.clear()
                    if checkpoint and offset == len(window) - 1 and len(window) == self.window_size:
                        rng = generator[0] if generator else None
                        self._checkpoint = (packet.index, end_latent, wan_state.model_state, rng)
                    yield OutputFrame(
                        packet.index,
                        prediction[..., : h * self.scale, : w * self.scale],
                        packet.source_time,
                        packet.arrival_time,
                        start.index,
                        end.index,
                        timings,
                    )
            first_window = False
            start, start_latent, start_prepared = end, end_latent, end_prepared
            if len(window) < self.window_size:
                return

    def checkpoint(self, frame: int) -> dict:
        if self._checkpoint is None or self._checkpoint[0] != frame:
            raise ValueError("RelayVSR can checkpoint only at the end of a complete keyframe window.")
        _, latent, state, rng = self._checkpoint
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)  # Swapped cache slices may still be copying to pinned RAM.
        caches = []
        for cache in state.block_caches:
            valid = min(cache.next_frame, cache.window_size) * cache.tokens_per_frame
            caches.append(dict(next_frame=cache.next_frame, tokens=cache.tokens_per_frame,
                               storage=tuple(t[:, :valid].to("cpu", copy=True).contiguous()
                                             for t in cache._storage)))
        return dict(frame=frame, latent=latent.cpu(), wan_frame=state.frame_index,
                    grid_shape=state.grid_signature[:-2], rng=rng.get_state() if rng else None, caches=caches)
