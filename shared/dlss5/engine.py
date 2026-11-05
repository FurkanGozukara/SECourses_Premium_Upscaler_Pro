"""DLSS 5 neural rendering engine: weights, the network on one GPU and a temporal session.

``DLSS5Engine`` loads the logical weights once per (file, device, precision) and
runs the network; ``DLSS5Session`` renders a frame sequence with reprojected
history and the learned temporal blend (or independent frames).
"""
from __future__ import annotations

import gc
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import network as _network
from . import ops
from . import topology as _topology
from .motion import MotionEstimate

WEIGHT_FORMATS = tuple(f"dlssnr-logical-v{v}" for v in range(8, 19))
_SPEC_PATH = Path(__file__).with_name("weight_spec.json")
PRECISIONS = ("fp16", "fp32")
# Activation bytes per network token at the chunk ceiling, measured on the HT network (see
# tools/test_dlss5_engine.py --vram); the planner keeps the peak inside the free VRAM.
_FIXED_BYTES_PER_TOKEN = {"fp16": 1100, "fp32": 2200}
_CHUNK_BYTES_PER_TOKEN = {"fp16": 9000, "fp32": 18000}


def _load_spec() -> dict:
    import json

    return json.loads(_SPEC_PATH.read_text(encoding="utf-8"))["tensors"]


def load_weights(path: str | Path) -> dict[str, torch.Tensor]:
    """Logical safetensors (``dlssnr-logical-v8`` … ``v18``, fully logical), checked against the HT spec."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
        if metadata.get("format") not in WEIGHT_FORMATS:
            raise ValueError(f"Unsupported DLSS 5 weight format {metadata.get('format')!r} in {path}")
        if metadata.get("fully_logical") != "true":
            raise ValueError(f"DLSS 5 weights must be fully logical: {path}")
        weights = {name: source.get_tensor(name) for name in source.keys()}
    topology = _topology.infer_logical({name: tuple(v.shape) for name, v in weights.items()})
    if topology == _topology.HT:
        spec = _load_spec()
        missing = sorted(set(spec) - set(weights))
        if missing:
            raise ValueError(f"DLSS 5 weights are missing {len(missing)} tensors (first {missing[:3]})")
        for name, entry in spec.items():
            if list(weights[name].shape) != entry["shape"]:
                raise ValueError(f"DLSS 5 weight {name}: expected {entry['shape']}, got {list(weights[name].shape)}")
    return weights


class DLSS5Engine:
    _cache: dict[tuple[str, str, str], "DLSS5Engine"] = {}
    _lock = threading.Lock()

    def __init__(self, weights_path: str | Path, device: torch.device, precision: str = "fp16"):
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        self.path = str(weights_path)
        self.device = device
        self.precision = precision
        self.dtype = torch.float16 if precision == "fp16" and device.type == "cuda" else torch.float32
        weights = load_weights(weights_path)
        model = _network.NeuralRenderingModel(weights).eval()
        del weights
        if self.dtype != torch.float32:
            model = model.to(self.dtype)
        self.model = model.to(device)
        self.topology = self.model.topology
        self.blend_scale = self.model.blend_scale or ops.BLEND_SCALE
        self.chunk_tokens = _network.CHUNK_TOKENS
        self._graphs: dict = {}
        self.graph_error = ""
        self._runner = self.model

    @classmethod
    def get(cls, weights_path: str | Path, device: str | torch.device = "cuda:0", precision: str = "fp16") -> "DLSS5Engine":
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        key = (str(Path(weights_path).resolve()), str(device), precision)
        with cls._lock:
            engine = cls._cache.get(key)
            if engine is None:
                cls.release_all()
                engine = cls(weights_path, device, precision)
                cls._cache[key] = engine
            return engine

    @classmethod
    def release_all(cls) -> None:
        """Drop every cached engine and give its VRAM back (other upscalers run next)."""
        if not cls._cache:
            return
        devices = {engine.device for engine in cls._cache.values()}
        cls._cache.clear()
        gc.collect()
        for device in devices:
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

    def plan_chunk(self, network_tokens: int, budget_bytes: Optional[int] = None) -> int:
        """Pick the token chunk that keeps one frame's activations inside ``budget_bytes``."""
        if budget_bytes is None and self.device.type == "cuda":
            free, _total = torch.cuda.mem_get_info(self.device)
            budget_bytes = int(free * 0.85)
        if budget_bytes is None:
            return _network.CHUNK_TOKENS
        fixed = _FIXED_BYTES_PER_TOKEN[self.precision] * network_tokens
        per_chunk_token = _CHUNK_BYTES_PER_TOKEN[self.precision]
        room = max(0, budget_bytes - fixed)
        chunk = int(room // per_chunk_token)
        chunk = max(1 << 14, min(1 << 20, chunk))
        chunk = 1 << int(np.floor(np.log2(chunk)))
        self.chunk_tokens = chunk
        _network.set_chunk_tokens(chunk)
        return chunk

    @torch.inference_mode()
    def head(self, features: torch.Tensor) -> torch.Tensor:
        """(1, H, W, 16) network input -> (1, H, W, 4) float32 head."""
        features = features.to(self.device, self.dtype)
        if self.use_graphs and self.device.type == "cuda":
            return self._graph_head(features)
        return self._runner(features).to(torch.float32)

    use_graphs: bool = True
    speed_label: str = "CUDA Graphs"

    def set_speed_mode(self, mode: str, log=None) -> str:
        """``CUDA Graphs`` (default) or ``Eager``; returns the active label.

        torch.compile was measured and rejected (2026-11-05, RTX 3090): 5x faster, but Inductor's
        fused arithmetic moves the half / E4M3 rounding points and the head drifted by 0.2 on average.
        """
        eager = str(mode or "").lower().startswith("eager")
        self.use_graphs = (not eager) and self.device.type == "cuda"
        self.speed_label = "CUDA Graphs" if self.use_graphs else "eager"
        self._graphs.clear()
        return self.speed_label

    def _graph_head(self, features: torch.Tensor) -> torch.Tensor:
        """Replay a captured CUDA graph per input shape: the graph has ~10k small kernels whose
        launches otherwise cost more CPU time than the GPU spends on them (RTX 3090, 540p:
        281 ms eager vs 167 ms of GPU work)."""
        key = (tuple(features.shape), self.chunk_tokens)
        entry = self._graphs.get(key)
        if entry is None:
            try:
                static_in = features.clone()
                stream = torch.cuda.Stream(self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        self._runner(static_in)
                torch.cuda.current_stream(self.device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    static_out = self._runner(static_in).to(torch.float32)
                entry = (graph, static_in, static_out)
                self._graphs.clear()  # one frame size at a time keeps the graph pool small
                self._graphs[key] = entry
            except Exception as exc:  # capture unsupported here: fall back to eager for this engine
                self.use_graphs = False
                self.graph_error = str(exc)
                torch.cuda.synchronize(self.device)
                return self._runner(features).to(torch.float32)
        graph, static_in, static_out = entry
        static_in.copy_(features)
        graph.replay()
        return static_out.clone()


@dataclass
class DLSS5Options:
    profile: str = "Standard"
    intensity: float = 1.0
    detail_strength: float = 1.0
    colour_strength: float = 1.0
    detail_radius: float = 4.0
    processing_scale: float = 1.0
    temporal: bool = True
    blend_scale: Optional[float] = None
    tone_strength: Optional[float] = None
    structure_strength: Optional[float] = None
    style: Optional[float] = None

    def controls(self) -> dict[str, float]:
        return ops.control_values(
            self.profile,
            {
                "normalized_style": self.style,
                "local_tone_strength": self.tone_strength,
                "local_structure_strength": self.structure_strength,
            },
        )


@dataclass
class DLSS5Session:
    engine: DLSS5Engine
    options: DLSS5Options = field(default_factory=DLSS5Options)
    history: Optional[torch.Tensor] = None
    frame_index: int = 0
    scene_cuts: int = 0

    def reset(self) -> None:
        self.history = None
        self.frame_index = 0

    def _guide(self, estimate: Optional[MotionEstimate], width: int, height: int):
        if estimate is None:
            return None, None
        device = self.engine.device
        motion = torch.from_numpy(np.ascontiguousarray(estimate.motion_uv)).to(device)
        if motion.shape[:2] != (height, width):
            # Normalised UV offsets keep their units when the grid is resized.
            motion = torch.nn.functional.interpolate(
                motion.permute(2, 0, 1).unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
            ).squeeze(0).permute(1, 2, 0)
        confidence = None
        if estimate.confidence is not None:
            confidence = torch.from_numpy(np.ascontiguousarray(estimate.confidence)).to(device)
            if confidence.shape[:2] != (height, width):
                confidence = torch.nn.functional.interpolate(
                    confidence.permute(2, 0, 1).unsqueeze(0), size=(height, width), mode="nearest"
                ).squeeze(0).permute(1, 2, 0)
        return motion, confidence

    @torch.inference_mode()
    def render(self, frame: torch.Tensor, estimate: Optional[MotionEstimate] = None) -> torch.Tensor:
        """Render one (H, W, 3) float32 frame in [0, 1] on the engine's device."""
        options = self.options
        if not self.options.temporal:
            self.history = None
        elif estimate is not None and estimate.reset:
            self.reset()
            self.scene_cuts += 1
        height, width = int(frame.shape[0]), int(frame.shape[1])
        scale = float(options.processing_scale)
        processing = frame
        if scale != 1.0:
            processing = ops.resample(frame, int(round(width * scale)), int(round(height * scale)))
        if self.history is not None and self.history.shape != processing.shape:
            self.reset()
        p_height, p_width = int(processing.shape[0]), int(processing.shape[1])
        geometry = ops.NetworkGeometry.vendor_aligned(p_width, p_height)
        controls = options.controls()
        if self.history is None:
            features = ops.make_features(processing, frame_index=self.frame_index, geometry=geometry,
                                         dtype=self.engine.dtype, **controls)
            head = self.engine.head(features)[0, :p_height, :p_width]
            del features
            output = ops.compose_head(head, processing, intensity=options.intensity)
        else:
            motion, confidence = self._guide(estimate, p_width, p_height)
            if motion is None:
                motion = torch.zeros((p_height, p_width, 2), dtype=torch.float32, device=processing.device)
            reprojected = ops.reproject(self.history, motion)
            features = ops.make_features(processing, frame_index=self.frame_index, geometry=geometry,
                                         history=reprojected, history_confidence=confidence,
                                         dtype=self.engine.dtype, **controls)
            seen_history = ops.network_history(features, geometry)
            head = self.engine.head(features)[0, :p_height, :p_width]
            del features, reprojected
            blend_scale = options.blend_scale if options.blend_scale is not None else self.engine.blend_scale
            output = ops.compose_temporal(head, processing, seen_history, blend_scale=blend_scale,
                                          history_confidence=confidence, intensity=options.intensity)
        self.history = output if options.temporal else None
        self.frame_index += 1
        if (p_width, p_height) != (width, height):
            output = ops.resample(output, width, height)
        return ops.compose_detail(frame, output, detail_strength=options.detail_strength,
                                  colour_strength=options.colour_strength, radius=options.detail_radius)


def to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 / uint16 / float (H, W, 3) RGB -> float32 tensor in [0, 1] on ``device``."""
    if frame.dtype == np.uint8:
        return torch.from_numpy(np.ascontiguousarray(frame)).to(device, non_blocking=True).to(torch.float32).div_(255.0)
    if frame.dtype == np.uint16:
        array = np.ascontiguousarray(frame).view(np.int16)
        tensor = torch.from_numpy(array).to(device, non_blocking=True).to(torch.int32) & 0xFFFF
        return tensor.to(torch.float32).div_(65535.0)
    return torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32)).to(device).clamp_(0.0, 1.0)


def to_array(frame: torch.Tensor, deep: bool = False) -> np.ndarray:
    """float32 [0, 1] tensor -> uint8 (or uint16 when ``deep``) (H, W, 3) array."""
    if deep:
        values = frame.clamp(0.0, 1.0).mul(65535.0).round().to(torch.int32)
        return values.to(torch.int16).cpu().numpy().view(np.uint16)
    return frame.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
