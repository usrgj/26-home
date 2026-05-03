"""OSNet 特征提取与余弦相似度计算。

作用：
    封装 torchreid 的 OSNet 模型，输入已经裁剪并 letterbox 到 256x128 的
    BGR 人物图，输出 L2 归一化后的 ReID 特征。也提供 prototype 构建和
    余弦相似度判断。

用法：
    reid = OSNetReID()
    feat = reid.extract(person_crop_256x128)
    prototype = reid.build_prototype([crop1, crop2, crop3])
    score = reid.similarity(feat, prototype)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "weights" / "osnet_ain_d_m_c.pth.tar"
)
OSNET_IMAGE_HEIGHT = 256
OSNET_IMAGE_WIDTH = 128


def _load_numpy():
    """Import NumPy only when feature vectors are created or compared."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Missing NumPy dependency. Install numpy before running osnet_reid."
        ) from exc

    return np


def _load_torchreid_stack():
    """Import the heavy ReID dependencies only when the extractor is built."""

    try:
        import torch
        import torch.nn.functional as functional
        from torchreid.utils import FeatureExtractor
    except ImportError as exc:
        raise RuntimeError(
            "Missing OSNet ReID dependencies. Install torch, torchvision, "
            "and torchreid/deep-person-reid before running osnet_reid."
        ) from exc

    return torch, functional, FeatureExtractor


def _resolve_device(device: str, torch_module) -> str:
    """Resolve auto/cuda/cpu into the actual torch device string."""

    if device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"

    if device == "cuda" and not torch_module.cuda.is_available():
        warnings.warn(
            "Requested cuda but CUDA is unavailable; falling back to cpu.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")

    return device


class OSNetReID:
    """Extract OSNet person features and compare them with cosine similarity.

    Usage:
        reid = OSNetReID()
        prototype = reid.build_prototype([person_crop_256x128])
        score = reid.similarity(reid.extract(query_crop_256x128), prototype)
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        device: str = "auto",
        threshold: float = 0.80,
    ) -> None:
        """Build the OSNet extractor from a local weight file."""

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"OSNet model file not found: {self.model_path}")

        self.threshold = float(threshold)
        self._torch, self._functional, extractor_cls = _load_torchreid_stack()
        self.device = _resolve_device(device, self._torch)
        self._extractor = extractor_cls(
            model_name="osnet_ain_x1_0",
            model_path=str(self.model_path),
            image_size=(OSNET_IMAGE_HEIGHT, OSNET_IMAGE_WIDTH),
            device=self.device,
            verbose=False,
        )

    def extract(self, person_bgr_256x128: np.ndarray) -> np.ndarray:
        """Extract a normalized feature vector from one BGR person crop."""

        np = _load_numpy()
        crop = self._validate_crop(person_bgr_256x128)
        rgb_crop = crop[:, :, ::-1].copy()

        with self._torch.no_grad():
            feat = self._extractor(rgb_crop)
            if feat.ndim != 2 or feat.shape[0] != 1:
                raise RuntimeError(f"Unexpected OSNet feature shape: {tuple(feat.shape)}")

            feat = self._functional.normalize(feat[0], p=2, dim=0)

        return feat.detach().cpu().numpy().astype(np.float32)

    def build_prototype(self, crops: Iterable[np.ndarray]) -> np.ndarray:
        """Average several person crops into one normalized prototype."""

        np = _load_numpy()
        feats = [self.extract(crop) for crop in crops]
        if not feats:
            raise ValueError("At least one crop is required to build a prototype")

        prototype = np.mean(np.stack(feats, axis=0), axis=0)
        norm = np.linalg.norm(prototype)
        if norm < 1e-12:
            raise RuntimeError("Prototype feature norm is too small")

        return (prototype / norm).astype(np.float32)

    def similarity(self, feat: np.ndarray, proto: np.ndarray) -> float:
        """Return cosine similarity between a feature and a prototype."""

        return cosine_similarity(feat, proto)

    def is_match(self, feat: np.ndarray, proto: np.ndarray) -> bool:
        """Return whether the feature matches the prototype threshold."""

        return self.similarity(feat, proto) >= self.threshold

    @staticmethod
    def _validate_crop(crop: np.ndarray) -> np.ndarray:
        """Validate that the crop is exactly the OSNet BGR input shape."""

        np = _load_numpy()
        if crop is None:
            raise ValueError("crop must not be None")
        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError("crop must be a BGR image with shape HxWx3")
        if crop.shape[:2] != (OSNET_IMAGE_HEIGHT, OSNET_IMAGE_WIDTH):
            raise ValueError(
                "crop must be letterboxed to "
                f"{OSNET_IMAGE_HEIGHT}x{OSNET_IMAGE_WIDTH}, got {crop.shape[:2]}"
            )

        return np.ascontiguousarray(crop)


def cosine_similarity(feat: np.ndarray, proto: np.ndarray) -> float:
    """Compute cosine similarity for two feature vectors."""

    np = _load_numpy()
    feat_arr = np.asarray(feat, dtype=np.float32).reshape(-1)
    proto_arr = np.asarray(proto, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(feat_arr) * np.linalg.norm(proto_arr))
    if denom < 1e-12:
        return 0.0

    return float(np.dot(feat_arr, proto_arr) / denom)
