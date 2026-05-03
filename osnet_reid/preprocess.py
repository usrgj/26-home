"""人物框裁剪与 OSNet 输入预处理。

作用：
    根据 YOLO 返回的 xyxy 人物框从 BGR 画面中裁出人物，并按比例缩放到
    256x128。短边不足部分用黑色像素填充，保证最终图像满足 OSNet 输入要求。

用法：
    crop = crop_and_letterbox_person(frame_bgr, bbox)
    # crop 的 shape 固定为 (256, 128, 3)，可直接传给 OSNetReID.extract()
"""

from __future__ import annotations

from typing import TypeAlias

from .core import OSNET_IMAGE_HEIGHT, OSNET_IMAGE_WIDTH


BBox: TypeAlias = tuple[int, int, int, int]


def _load_numpy():
    """Import NumPy only when a padded crop is created."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Missing NumPy dependency. Install numpy before running "
            "osnet_reid image preprocessing."
        ) from exc

    return np


def _load_cv2():
    """Import OpenCV only when image resizing is needed."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Missing OpenCV dependency. Install opencv-python before running "
            "osnet_reid image preprocessing."
        ) from exc

    return cv2


def clip_bbox(bbox: BBox, width: int, height: int) -> BBox | None:
    """Clip one xyxy box to the image bounds."""

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def crop_person(frame_bgr: np.ndarray, bbox: BBox) -> np.ndarray | None:
    """Crop a person box from a BGR frame."""

    if frame_bgr is None or frame_bgr.ndim != 3:
        return None

    height, width = frame_bgr.shape[:2]
    clipped = clip_bbox(bbox, width, height)
    if clipped is None:
        return None

    x1, y1, x2, y2 = clipped
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return crop.copy()


def letterbox_person_crop(
    person_bgr: np.ndarray,
    target_height: int = OSNET_IMAGE_HEIGHT,
    target_width: int = OSNET_IMAGE_WIDTH,
) -> np.ndarray:
    """Resize a BGR person crop with black padding to the OSNet input size."""

    if person_bgr is None or person_bgr.ndim != 3 or person_bgr.shape[2] != 3:
        raise ValueError("person_bgr must be a BGR image with shape HxWx3")

    src_height, src_width = person_bgr.shape[:2]
    if src_width <= 0 or src_height <= 0:
        raise ValueError("person_bgr has an empty width or height")

    scale = min(target_width / src_width, target_height / src_height)
    resized_width = max(1, int(round(src_width * scale)))
    resized_height = max(1, int(round(src_height * scale)))

    cv2 = _load_cv2()
    resized = cv2.resize(
        person_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )

    np = _load_numpy()
    canvas = np.zeros((target_height, target_width, 3), dtype=person_bgr.dtype)
    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2
    canvas[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized
    return canvas


def crop_and_letterbox_person(frame_bgr: np.ndarray, bbox: BBox) -> np.ndarray | None:
    """Crop one person box and letterbox it to 256x128 BGR."""

    crop = crop_person(frame_bgr, bbox)
    if crop is None:
        return None

    return letterbox_person_crop(crop)
