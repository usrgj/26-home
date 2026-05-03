"""OSNet ReID 特征库存取。

作用：
    管理注册得到的身份特征库，落盘为压缩 NPZ。每个身份同时保存平均特征
    prototype 和多个原始样本 features，后续可按“快筛 + 多特征确认”匹配。

用法：
    gallery = FeatureGallery.load(path) if path.exists() else FeatureGallery()
    gallery.upsert(identity)
    gallery.save(path)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .core import OSNET_IMAGE_HEIGHT, OSNET_IMAGE_WIDTH


SCHEMA_VERSION = 1
DEFAULT_GALLERY_PATH = Path(__file__).resolve().parent / "gallery" / "feature_library.npz"
DEFAULT_MODEL_NAME = "osnet_ain_x1_0"


def _load_numpy():
    """Import NumPy only when a gallery file is loaded or saved."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Missing NumPy dependency. Install numpy before using the ReID gallery."
        ) from exc

    return np


def utc_timestamp() -> str:
    """Return a compact UTC timestamp for gallery metadata."""

    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass(eq=False)
class GalleryIdentity:
    """One registered identity with both prototype and raw sample features."""

    person_id: str
    name: str
    prototype: Any
    features: Any
    threshold: float = 0.80
    created_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)

    @property
    def feature_count(self) -> int:
        """Return how many sample features this identity stores."""

        return int(getattr(self.features, "shape", [0])[0])


class FeatureGallery:
    """Load, update, and save registered OSNet identities.

    Usage:
        gallery = FeatureGallery.load(path) if path.exists() else FeatureGallery()
        gallery.upsert(identity)
        gallery.save(path)
    """

    def __init__(
        self,
        identities: list[GalleryIdentity] | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        image_size: tuple[int, int] = (OSNET_IMAGE_HEIGHT, OSNET_IMAGE_WIDTH),
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        """Create an in-memory feature gallery."""

        now = utc_timestamp()
        self.identities = identities or []
        self.model_name = model_name
        self.image_size = image_size
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    @classmethod
    def load(cls, path: str | Path) -> "FeatureGallery":
        """Load a feature gallery from a compressed NPZ file."""

        np = _load_numpy()
        gallery_path = Path(path)
        with np.load(gallery_path, allow_pickle=False) as data:
            schema_version = int(data["schema_version"].item())
            if schema_version != SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported gallery schema version {schema_version}; "
                    f"expected {SCHEMA_VERSION}"
                )

            person_ids = data["person_ids"].astype(str).tolist()
            names = data["names"].astype(str).tolist()
            thresholds = data["thresholds"].astype("float32")
            prototypes = data["prototypes"].astype("float32")
            feature_matrix = data["features"].astype("float32")
            feature_counts = data["feature_counts"].astype("int32")
            identity_created_at = data["identity_created_at"].astype(str).tolist()
            identity_updated_at = data["identity_updated_at"].astype(str).tolist()

            identities: list[GalleryIdentity] = []
            for index, person_id in enumerate(person_ids):
                count = int(feature_counts[index])
                identities.append(
                    GalleryIdentity(
                        person_id=person_id,
                        name=names[index],
                        prototype=prototypes[index].copy(),
                        features=feature_matrix[index, :count].copy(),
                        threshold=float(thresholds[index]),
                        created_at=identity_created_at[index],
                        updated_at=identity_updated_at[index],
                    )
                )

            image_size_array = data["image_size"].astype("int32")
            return cls(
                identities=identities,
                model_name=str(data["model_name"].item()),
                image_size=(int(image_size_array[0]), int(image_size_array[1])),
                created_at=str(data["created_at"].item()),
                updated_at=str(data["updated_at"].item()),
            )

    def save(self, path: str | Path) -> None:
        """Save the gallery to a compressed NPZ file."""

        if not self.identities:
            raise ValueError("Cannot save an empty feature gallery")

        np = _load_numpy()
        gallery_path = Path(path)
        gallery_path.parent.mkdir(parents=True, exist_ok=True)

        feature_dim = self._feature_dim()
        max_count = max(identity.feature_count for identity in self.identities)
        prototypes = np.zeros((len(self.identities), feature_dim), dtype=np.float32)
        features = np.zeros(
            (len(self.identities), max_count, feature_dim),
            dtype=np.float32,
        )
        feature_counts = np.zeros((len(self.identities),), dtype=np.int32)

        for index, identity in enumerate(self.identities):
            identity_features = np.asarray(identity.features, dtype=np.float32)
            identity_prototype = np.asarray(identity.prototype, dtype=np.float32)
            if identity_features.ndim != 2:
                raise ValueError(f"{identity.person_id} features must be a 2D array")
            if identity_features.shape[1] != feature_dim:
                raise ValueError(f"{identity.person_id} feature dimension mismatch")
            if identity_prototype.shape != (feature_dim,):
                raise ValueError(f"{identity.person_id} prototype dimension mismatch")

            count = identity_features.shape[0]
            prototypes[index] = identity_prototype
            features[index, :count] = identity_features
            feature_counts[index] = count

        self.updated_at = utc_timestamp()
        np.savez_compressed(
            gallery_path,
            schema_version=np.array(SCHEMA_VERSION, dtype=np.int32),
            created_at=np.array(self.created_at),
            updated_at=np.array(self.updated_at),
            model_name=np.array(self.model_name),
            image_size=np.array(self.image_size, dtype=np.int32),
            person_ids=np.array(
                [identity.person_id for identity in self.identities],
                dtype="U128",
            ),
            names=np.array(
                [identity.name for identity in self.identities],
                dtype="U128",
            ),
            thresholds=np.array(
                [identity.threshold for identity in self.identities],
                dtype=np.float32,
            ),
            identity_created_at=np.array(
                [identity.created_at for identity in self.identities],
                dtype="U32",
            ),
            identity_updated_at=np.array(
                [identity.updated_at for identity in self.identities],
                dtype="U32",
            ),
            prototypes=prototypes,
            features=features,
            feature_counts=feature_counts,
        )

    def get(self, person_id: str) -> GalleryIdentity | None:
        """Return one registered identity by stable person_id."""

        for identity in self.identities:
            if identity.person_id == person_id:
                return identity
        return None

    def upsert(self, identity: GalleryIdentity) -> None:
        """Insert or replace an identity by person_id."""

        existing = self.get(identity.person_id)
        identity.updated_at = utc_timestamp()
        if existing is None:
            self.identities.append(identity)
            return

        identity.created_at = existing.created_at
        for index, item in enumerate(self.identities):
            if item.person_id == identity.person_id:
                self.identities[index] = identity
                return

    def _feature_dim(self) -> int:
        """Return the shared feature dimension for all registered identities."""

        if not self.identities:
            raise ValueError("Feature gallery is empty")

        first = self.identities[0]
        return int(getattr(first.prototype, "shape", [0])[0])
