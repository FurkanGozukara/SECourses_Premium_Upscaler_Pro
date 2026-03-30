import argparse
import gc
import json
import math
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch


def _install_torchvision_stub() -> None:
    if "torchvision.transforms" in sys.modules:
        return

    torchvision_stub = types.ModuleType("torchvision")
    transforms_stub = types.ModuleType("torchvision.transforms")

    class _UnusedTransform:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("torchvision transforms are not used in this script")

    transforms_stub.ToTensor = lambda: _UnusedTransform()
    transforms_stub.ToPILImage = lambda: _UnusedTransform()
    torchvision_stub.transforms = transforms_stub
    sys.modules.setdefault("torchvision", torchvision_stub)
    sys.modules.setdefault("torchvision.transforms", transforms_stub)


_install_torchvision_stub()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SeedVR2"))

from src.utils import color_fix  # noqa: E402


class _DebugCapture:
    def __init__(self):
        self.messages = []

    def log(self, message: str, **kwargs):
        self.messages.append(str(message))


def _legacy_lab_color_transfer(content_feat: torch.Tensor, style_feat: torch.Tensor, luminance_weight: float = 0.8) -> torch.Tensor:
    content_feat = color_fix.wavelet_reconstruction(content_feat.clone(), style_feat.clone(), debug=None)

    if content_feat.shape != style_feat.shape:
        style_feat = color_fix.safe_interpolate_operation(
            style_feat,
            size=content_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    else:
        style_feat = style_feat.clone()

    device = content_feat.device
    content_feat, original_dtype = color_fix.ensure_float32_precision(content_feat)
    style_feat, _ = color_fix.ensure_float32_precision(style_feat)

    rgb_to_xyz_matrix = torch.tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=torch.float32,
        device=device,
    )
    xyz_to_rgb_matrix = torch.tensor(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=torch.float32,
        device=device,
    )

    epsilon = 6.0 / 29.0
    kappa = (29.0 / 3.0) ** 3

    content_feat.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
    style_feat.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)

    content_lab = color_fix._rgb_to_lab_batch(content_feat, device, rgb_to_xyz_matrix, epsilon, kappa)
    style_lab = color_fix._rgb_to_lab_batch(style_feat, device, rgb_to_xyz_matrix, epsilon, kappa)

    matched_a = color_fix._histogram_matching_channel_exact(content_lab[:, 1], style_lab[:, 1], device)
    matched_b = color_fix._histogram_matching_channel_exact(content_lab[:, 2], style_lab[:, 2], device)

    if luminance_weight < 1.0:
        matched_l = color_fix._histogram_matching_channel_exact(content_lab[:, 0], style_lab[:, 0], device)
        result_l = content_lab[:, 0].mul(luminance_weight).add_(matched_l.mul(1.0 - luminance_weight))
    else:
        result_l = content_lab[:, 0]

    result_lab = torch.stack([result_l, matched_a, matched_b], dim=1)
    result_rgb = color_fix._lab_to_rgb_batch(result_lab, device, xyz_to_rgb_matrix, epsilon, kappa)
    result = result_rgb.mul_(2.0).sub_(1.0)

    if result.dtype != original_dtype:
        result = result.to(original_dtype)

    return result


def _load_video_frames(video_path: Path, frame_count: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    try:
        while len(frames) < frame_count:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    if len(frames) < frame_count:
        raise RuntimeError(f"Requested {frame_count} frames, got {len(frames)} from {video_path}")
    return np.stack(frames, axis=0)


def _frames_bgr_to_tensor(frames_bgr: np.ndarray) -> torch.Tensor:
    frames_rgb = frames_bgr[..., ::-1].copy()
    tensor = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).float().div_(255.0)
    return tensor.mul_(2.0).sub_(1.0)


def _tensor_to_bgr_images(tensor: torch.Tensor) -> np.ndarray:
    rgb = tensor.detach().cpu().clamp(-1.0, 1.0).add(1.0).mul(0.5)
    rgb = rgb.mul(255.0).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    return rgb[..., ::-1].copy()


def _compute_metrics(old_bgr: np.ndarray, new_bgr: np.ndarray) -> dict:
    diff = old_bgr.astype(np.float32) - new_bgr.astype(np.float32)
    abs_diff = np.abs(diff)
    mse = np.mean(diff ** 2)
    psnr = float("inf") if mse <= 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    return {
        "mae": float(np.mean(abs_diff)),
        "rmse": float(math.sqrt(mse)),
        "max_abs": float(np.max(abs_diff)),
        "psnr_db": float(psnr),
    }


def _make_synthetic_content(style_bgr: np.ndarray) -> np.ndarray:
    frames = style_bgr.astype(np.float32)
    blurred = np.stack([cv2.GaussianBlur(frame, (0, 0), 1.1) for frame in frames], axis=0)
    sharpened = cv2.addWeighted(frames, 1.22, blurred, -0.22, 0.0)

    # Apply a moderate channel/gamma shift to simulate output that needs color correction.
    shifted = sharpened.copy()
    shifted[..., 0] *= 1.05
    shifted[..., 1] *= 0.97
    shifted[..., 2] *= 0.91
    shifted = np.clip(255.0 * np.power(np.clip(shifted / 255.0, 0.0, 1.0), 0.92), 0.0, 255.0)
    return shifted.round().astype(np.uint8)


def _save_comparison_sheet(
    frame_idx: int,
    content_bgr: np.ndarray,
    style_bgr: np.ndarray,
    old_bgr: np.ndarray,
    new_bgr: np.ndarray,
    output_dir: Path,
) -> dict:
    abs_diff = cv2.absdiff(old_bgr, new_bgr)
    boosted = cv2.normalize(abs_diff, None, 0, 255, cv2.NORM_MINMAX)
    boosted = cv2.applyColorMap(boosted, cv2.COLORMAP_INFERNO)

    separator = np.full((old_bgr.shape[0], 12, 3), 32, dtype=np.uint8)
    sheet = np.concatenate([content_bgr, separator, style_bgr, separator, old_bgr, separator, new_bgr, separator, boosted], axis=1)

    label_bar = np.full((48, sheet.shape[1], 3), 18, dtype=np.uint8)
    pane_w = old_bgr.shape[1] + 12
    cv2.putText(label_bar, "CONTENT", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(label_bar, "STYLE", (pane_w + 8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(label_bar, "OLD / exact", (pane_w * 2 + 8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(label_bar, "NEW / streamed", (pane_w * 3 + 8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(label_bar, "ABS DIFF", (pane_w * 4 + 8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2, cv2.LINE_AA)
    composed = np.concatenate([label_bar, sheet], axis=0)

    out_path = output_dir / f"frame_{frame_idx:03d}_context_old_new_diff.png"
    cv2.imwrite(str(out_path), composed)
    return {"frame": int(frame_idx), "path": str(out_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare SeedVR2 LAB old vs new paths on a real video batch.")
    parser.add_argument("--input", type=Path, default=Path("test/1280x720.mp4"))
    parser.add_argument("--frames", type=int, default=21)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_path = (repo_root / args.input).resolve() if not args.input.is_absolute() else args.input.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (repo_root / "outputs" / f"seedvr2_lab_compare_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    style_frames_bgr = _load_video_frames(input_path, args.frames)
    content_frames_bgr = _make_synthetic_content(style_frames_bgr)
    style_tensor = _frames_bgr_to_tensor(style_frames_bgr)
    content_tensor = _frames_bgr_to_tensor(content_frames_bgr)
    total_pixels = int(content_tensor.shape[0]) * int(content_tensor.shape[-2]) * int(content_tensor.shape[-1])

    old_start = time.perf_counter()
    old_tensor = _legacy_lab_color_transfer(content_tensor.clone(), style_tensor.clone(), luminance_weight=0.8)
    old_seconds = time.perf_counter() - old_start
    gc.collect()

    debug_capture = _DebugCapture()
    new_start = time.perf_counter()
    new_tensor = color_fix.lab_color_transfer(content_tensor.clone(), style_tensor.clone(), debug_capture, luminance_weight=0.8)
    new_seconds = time.perf_counter() - new_start
    gc.collect()

    old_bgr = _tensor_to_bgr_images(old_tensor)
    new_bgr = _tensor_to_bgr_images(new_tensor)

    overall = {
        "old_vs_new": _compute_metrics(old_bgr, new_bgr),
        "content_vs_style": _compute_metrics(content_frames_bgr, style_frames_bgr),
        "old_vs_style": _compute_metrics(old_bgr, style_frames_bgr),
        "new_vs_style": _compute_metrics(new_bgr, style_frames_bgr),
    }
    per_frame = []
    for idx in range(old_bgr.shape[0]):
        item = {
            "frame": int(idx),
            "old_vs_new": _compute_metrics(old_bgr[idx], new_bgr[idx]),
            "content_vs_style": _compute_metrics(content_frames_bgr[idx], style_frames_bgr[idx]),
            "old_vs_style": _compute_metrics(old_bgr[idx], style_frames_bgr[idx]),
            "new_vs_style": _compute_metrics(new_bgr[idx], style_frames_bgr[idx]),
        }
        per_frame.append(item)

    selected_frames = sorted({0, max(0, args.frames // 2), args.frames - 1})
    artifacts = [
        _save_comparison_sheet(
            idx,
            content_frames_bgr[idx],
            style_frames_bgr[idx],
            old_bgr[idx],
            new_bgr[idx],
            output_dir,
        )
        for idx in selected_frames
    ]

    result = {
        "input_video": str(input_path),
        "frame_count": int(args.frames),
        "frame_dimensions": {
            "width": int(old_bgr.shape[2]),
            "height": int(old_bgr.shape[1]),
        },
        "batch_pixels": int(total_pixels),
        "lab_sort_match_threshold": int(color_fix.LAB_SORT_MATCH_MAX_ELEMENTS),
        "new_path_expected": bool(total_pixels > int(color_fix.LAB_SORT_MATCH_MAX_ELEMENTS)),
        "frames_per_chunk_new_path": int(color_fix._lab_frames_per_chunk(content_tensor)),
        "old_exact_runtime_sec": float(old_seconds),
        "new_streamed_runtime_sec": float(new_seconds),
        "overall_metrics": overall,
        "per_frame_metrics": per_frame,
        "selected_frame_artifacts": artifacts,
        "new_path_logs": debug_capture.messages,
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({
        "metrics_path": str(metrics_path),
        "overall_metrics": overall,
        "old_exact_runtime_sec": round(old_seconds, 4),
        "new_streamed_runtime_sec": round(new_seconds, 4),
        "new_path_logs": debug_capture.messages,
        "artifacts": artifacts,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
