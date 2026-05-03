from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple


def _select_indices(total_frames: int) -> List[int]:
    """Select first, middle, and last frame indices, matching SparkVSR defaults."""
    total = int(total_frames or 0)
    if total <= 0:
        return []
    if total == 1:
        return [0]
    if total == 2:
        return [0, 1]
    return [0, total // 2, total - 1]


def save_ref_frames_locally(
    video_path: str | os.PathLike | None = None,
    output_dir: str | os.PathLike | None = None,
    video_id: Optional[str] = None,
    target_frames: Optional[int] = None,
    is_match: bool = False,
    specific_indices: Optional[List[int]] = None,
) -> List[Tuple[int, str]]:
    """Extract selected reference frames without any external API."""
    if output_dir is None:
        raise ValueError("output_dir must be provided")
    if video_path is None:
        return []

    import cv2  # type: ignore

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{video_id}_" if video_id else ""

    path = str(video_path)
    if "LQ-Video" in path:
        gt_path = path.replace("LQ-Video", "GT-Video")
        if Path(gt_path).exists():
            path = gt_path

    use_decord = False
    vr = None
    cap = None
    try:
        from decord import VideoReader, cpu  # type: ignore

        vr = VideoReader(path, ctx=cpu(0))
        total_frames = len(vr)
        use_decord = True
    except Exception:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    effective_frames = total_frames
    if is_match and total_frames > 0:
        remainder = (total_frames - 1) % 8
        if remainder != 0:
            effective_frames = total_frames + (8 - remainder)

    indices = list(specific_indices) if specific_indices is not None else _select_indices(target_frames or effective_frames)
    saved: List[Tuple[int, str]] = []

    try:
        for idx in indices:
            if total_frames <= 0:
                break
            read_idx = min(max(0, int(idx)), total_frames - 1)
            if use_decord and vr is not None:
                frame_obj = vr[read_idx]
                frame_np = frame_obj.asnumpy() if hasattr(frame_obj, "asnumpy") else frame_obj.cpu().numpy()
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                ok = True
            else:
                assert cap is not None
                cap.set(cv2.CAP_PROP_POS_FRAMES, read_idx)
                ok, frame_bgr = cap.read()
            if not ok:
                continue
            frame_path = out_dir / f"{prefix}frame_{int(idx):05d}.png"
            cv2.imwrite(str(frame_path), frame_bgr)
            saved.append((int(idx), str(frame_path)))
    finally:
        if cap is not None:
            cap.release()

    return saved


def get_ref_frames_api(*args, **kwargs):
    """Lazy placeholder for optional external FAL reference generation."""
    if not os.environ.get("FAL_KEY"):
        raise RuntimeError("SparkVSR API reference mode requires FAL_KEY in the environment.")
    try:
        import fal_client  # noqa: F401
    except Exception as exc:
        raise RuntimeError("SparkVSR API reference mode requires fal-client to be installed.") from exc
    raise RuntimeError(
        "SparkVSR API reference generation is not bundled in this redistributable build. "
        "Use no_ref, gt, or PiSA-SR mode."
    )
