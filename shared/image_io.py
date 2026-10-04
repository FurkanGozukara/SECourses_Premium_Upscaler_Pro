"""
Image file I/O that works for every image the app accepts.

Two decoder pitfalls broke upscales for some images:

* Bit depth: ``cv2.imread(..., IMREAD_UNCHANGED)`` returns 16-bit PNG/TIFF files as
  uint16 (0-65535) and float TIFFs as float32, so code that divides by 255 blows the
  picture out to white. Pillow's ``convert("RGB")`` clips 16-bit grayscale ("I;16")
  to white and cannot open float RGB TIFFs at all.
* Paths: on Windows ``cv2.imread`` cannot open non-ASCII paths (Turkish ç ğ ı ö ş ü,
  accents, CJK...) and ``cv2.imwrite`` reports success while writing a mangled name.

``imread``/``imwrite`` are drop-in replacements for the OpenCV functions (pixels and
EXIF handling identical to ``cv2.imread``), and the ``load_*`` helpers return
model-ready 8-bit images from any supported file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np

PathLike = Union[str, Path]

# Pillow modes whose convert("RGB") clips instead of scaling.
_PIL_HIGH_DEPTH_MODES = {"I;16", "I;16L", "I;16B", "I;16N", "I", "F"}


def imread(path: PathLike, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """``cv2.imread`` that also opens non-ASCII paths on Windows."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path: PathLike, image: np.ndarray, params: Optional[Sequence[int]] = None) -> bool:
    """``cv2.imwrite`` that also writes non-ASCII paths on Windows."""
    suffix = Path(str(path)).suffix
    if not suffix:
        return False
    try:
        ok, encoded = cv2.imencode(suffix, image, list(params) if params else [])
        if not ok:
            return False
        encoded.tofile(str(path))
    except (cv2.error, OSError, ValueError):
        return False
    return True


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Scale an image of any depth to uint8 (16-bit by 1/257, float from [0, 1])."""
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.uint16:
        return ((image.astype(np.uint32) + 128) // 257).astype(np.uint8)
    if np.issubdtype(image.dtype, np.integer):
        scaled = image.astype(np.float64) * (255.0 / float(np.iinfo(image.dtype).max))
    else:
        scaled = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0) * 255.0
    return np.clip(scaled + 0.5, 0.0, 255.0).astype(np.uint8)


def load_image_bgr8(path: PathLike) -> Optional[np.ndarray]:
    """
    Read any supported image as 3-channel BGR uint8, like ``cv2.imread(path)``.

    IMREAD_COLOR already converts 16-bit, grayscale and palette files correctly; the
    fallback covers files it rejects, such as 32-bit float TIFFs.
    """
    image = imread(path, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    image = imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    image = to_uint8(image)
    if image.ndim == 2 or image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(image[..., :3])


def load_image_rgb_pil(path: PathLike):
    """
    Open an image as an RGB Pillow image without clipping 16-bit or float data.

    Ordinary files keep Pillow's exact conversion; 16-bit grayscale, 32-bit and float
    files, and files Pillow cannot open, are decoded through OpenCV instead.
    """
    from PIL import Image

    try:
        with Image.open(path) as img:
            if img.mode not in _PIL_HIGH_DEPTH_MODES:
                return img.convert("RGB")
    except Exception:
        pass
    bgr = load_image_bgr8(path)
    if bgr is None:
        raise ValueError(f"Cannot open image file: {path}")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
