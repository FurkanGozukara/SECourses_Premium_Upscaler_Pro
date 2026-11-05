"""Motion guides for DLSS 5 on decoded video.

A decoded video carries no engine motion vectors, so they are estimated with
dense optical flow (OpenCV DIS) between consecutive source frames, in both
directions. The forward/backward consistency and the photometric error of the
warped previous frame give a per-pixel history confidence, and they decide
scene cuts (history reset). Everything runs on the CPU at a reduced flow
resolution in a worker thread while the GPU renders the previous frame.
Motion is a normalised history-UV offset: +x samples right, +y samples down.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Long edge of the flow image per quality (0 = the frame's own size).
MOTION_QUALITY = {
    "Fast": ("ultrafast", 640),
    "Balanced": ("fast", 960),
    "Quality": ("medium", 1280),
    "Maximum": ("medium", 0),
}
DEFAULT_MOTION_QUALITY = "Balanced"


@dataclass
class MotionEstimate:
    motion_uv: np.ndarray            # (h, w, 2) float32 at the flow resolution
    confidence: Optional[np.ndarray]  # (h, w, 1) float32 in [0, 1], or None
    reset: bool
    reason: Optional[str] = None


def _gray(frame_rgb: np.ndarray) -> np.ndarray:
    frame = frame_rgb.astype(np.float32)
    if frame_rgb.dtype == np.uint16:
        frame *= 1.0 / 65535.0
    elif frame_rgb.dtype == np.uint8:
        frame *= 1.0 / 255.0
    luma = frame[..., 0] * 0.2126 + frame[..., 1] * 0.7152 + frame[..., 2] * 0.0722
    return (np.clip(luma, 0, 1) * 255 + 0.5).astype(np.uint8)


class FlowMotionEstimator:
    """Bidirectional DIS flow with a forward/backward confidence map (MLX-DLSS recipe)."""

    def __init__(self, quality: str = DEFAULT_MOTION_QUALITY, scene_cut_threshold: float = 0.3):
        import cv2

        preset_name, self.target_dim = MOTION_QUALITY.get(quality, MOTION_QUALITY[DEFAULT_MOTION_QUALITY])
        presets = {
            "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
            "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
            "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        }
        self._cv2 = cv2
        self._flow = cv2.DISOpticalFlow_create(presets[preset_name])
        self.scene_cut_threshold = float(scene_cut_threshold)

    def _small(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Downscaled float RGB and its uint8 luma at the flow resolution."""
        cv2 = self._cv2
        height, width = frame.shape[:2]
        if self.target_dim and max(height, width) > self.target_dim:
            scale = self.target_dim / float(max(height, width))
            size = (max(16, int(round(width * scale))), max(16, int(round(height * scale))))
            small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        else:
            small = frame
        rgb = small.astype(np.float32) * (1.0 / (65535.0 if small.dtype == np.uint16 else 255.0))
        return rgb, _gray(small)

    def estimate(self, current: np.ndarray, previous: np.ndarray) -> MotionEstimate:
        cv2 = self._cv2
        cur_rgb, cur_gray = self._small(current)
        prev_rgb, prev_gray = self._small(previous)
        height, width = cur_gray.shape
        scale = np.array([width, height], np.float32)
        backward = self._flow.calc(cur_gray, prev_gray, None)   # current -> previous, pixels
        forward = self._flow.calc(prev_gray, cur_gray, None)    # previous -> current, pixels
        yy, xx = np.indices((height, width), dtype=np.float32)
        mx, my = xx + backward[..., 0], yy + backward[..., 1]
        inside = (mx >= 0) & (mx <= width - 1) & (my >= 0) & (my <= height - 1)
        reverse = cv2.remap(forward, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        warped = cv2.remap(prev_rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fb_squared = np.square(backward + reverse).sum(-1)
        tolerance = 0.01 * (np.square(backward).sum(-1) + np.square(reverse).sum(-1)) + 0.5
        error = np.abs(cur_rgb - warped).mean(-1)
        photometric = np.clip((0.12 - error) / 0.09, 0, 1)
        valid = inside & (fb_squared <= tolerance) & (photometric > 0)
        # Keep newly exposed boundaries out of the history filter's footprint
        # (7 px at the full resolution, scaled to the flow image).
        kernel = max(3, int(round(7 * width / float(current.shape[1]))) | 1)
        valid = cv2.erode(valid.astype(np.uint8), np.ones((kernel, kernel), np.uint8), borderType=cv2.BORDER_REPLICATE)
        confidence = (valid * photometric).astype(np.float32)[..., None]
        reset, reason = False, None
        if self.scene_cut_threshold > 0:
            coverage = float((confidence > 0.5).mean())
            luma_cur = cur_rgb[..., 0] * 0.2126 + cur_rgb[..., 1] * 0.7152 + cur_rgb[..., 2] * 0.0722
            luma_warp = warped[..., 0] * 0.2126 + warped[..., 1] * 0.7152 + warped[..., 2] * 0.0722
            warped_error = float(np.abs(luma_cur - luma_warp).mean())
            # Camera motion can change every pixel without changing the scene.
            if coverage < 0.5 and warped_error > self.scene_cut_threshold:
                reset, reason = True, "scene change"
            elif coverage < 0.15 and warped_error > 0.12:
                reset, reason = True, "lost correspondence"
        motion_uv = (backward / scale).astype(np.float32)
        return MotionEstimate(motion_uv, confidence, reset, reason)


class StillMotionEstimator:
    """Zero motion for a sequence known to be static (no flow cost); scene cuts by luma change."""

    def __init__(self, scene_cut_threshold: float = 0.3):
        self.scene_cut_threshold = float(scene_cut_threshold)

    def estimate(self, current: np.ndarray, previous: np.ndarray) -> MotionEstimate:
        step = 8
        cur = _gray(current[::step, ::step]).astype(np.float32) / 255.0
        prev = _gray(previous[::step, ::step]).astype(np.float32) / 255.0
        cut = self.scene_cut_threshold > 0 and float(np.abs(cur - prev).mean()) > self.scene_cut_threshold
        h, w = cur.shape
        return MotionEstimate(np.zeros((h, w, 2), np.float32), None, cut, "scene change" if cut else None)


def make_estimator(quality: str, scene_cut_threshold: float):
    if str(quality) == "Static camera (no flow)":
        return StillMotionEstimator(scene_cut_threshold)
    return FlowMotionEstimator(quality, scene_cut_threshold)
