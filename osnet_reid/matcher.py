"""OSNet ReID 身份匹配规则。

作用：
    把“当前检测到的人是否为特征库中的目标身份”的阈值判断封装成可复用
    工具。独立识别脚本和后续 task1 接入都应复用这里，避免各处写出不一致
    的匹配逻辑。

用法：
    matcher = IdentityMatcher(mode="strict", proto_threshold=0.75, feature_threshold=0.80)
    result = matcher.score(query_feat, identity)
    if result.is_match:
        ...
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import cosine_similarity


MATCH_MODES = ("strict", "max-only", "prototype-only")


def _load_numpy():
    """Import NumPy only when vector matching is performed."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Missing NumPy dependency. Install numpy before matching ReID features."
        ) from exc

    return np


@dataclass(frozen=True)
class MatchScore:
    """One ReID score result for query feature against one identity."""

    proto_score: float
    max_feature_score: float
    final_score: float
    is_match: bool
    mode: str
    proto_threshold: float
    feature_threshold: float
    reason: str = ""


class IdentityMatcher:
    """Score one query feature against one registered identity.

    Usage:
        matcher = IdentityMatcher(mode="strict")
        result = matcher.score(query_feat, identity)
    """

    def __init__(
        self,
        mode: str = "strict",
        proto_threshold: float | None = None,
        feature_threshold: float | None = None,
        proto_margin: float = 0.05,
    ) -> None:
        """Create a matcher with explicit or identity-derived thresholds."""

        if mode not in MATCH_MODES:
            raise ValueError(f"mode must be one of: {', '.join(MATCH_MODES)}")

        self.mode = mode
        self.proto_threshold = proto_threshold
        self.feature_threshold = feature_threshold
        self.proto_margin = float(proto_margin)

    def score(
        self,
        query_feat,
        identity,
        proto_score: float | None = None,
    ) -> MatchScore:
        """Return prototype score, max-feature score, and final decision."""

        thresholds = self._resolve_thresholds(identity)
        if proto_score is None:
            proto_score = self.prototype_score(query_feat, identity)

        max_feature_score = 0.0
        if self.mode != "prototype-only":
            max_feature_score = self.max_feature_score(query_feat, identity)

        if self.mode == "strict":
            is_match = (
                proto_score >= thresholds["proto"]
                and max_feature_score >= thresholds["feature"]
            )
            final_score = max_feature_score
        elif self.mode == "max-only":
            is_match = max_feature_score >= thresholds["feature"]
            final_score = max_feature_score
        else:
            is_match = proto_score >= thresholds["proto"]
            final_score = proto_score

        return MatchScore(
            proto_score=float(proto_score),
            max_feature_score=float(max_feature_score),
            final_score=float(final_score),
            is_match=bool(is_match),
            mode=self.mode,
            proto_threshold=thresholds["proto"],
            feature_threshold=thresholds["feature"],
        )

    def filtered_score(
        self,
        proto_score: float,
        identity,
        reason: str = "filtered",
    ) -> MatchScore:
        """Return a non-match result for candidates skipped by prefiltering."""

        thresholds = self._resolve_thresholds(identity)
        return MatchScore(
            proto_score=float(proto_score),
            max_feature_score=0.0,
            final_score=float(proto_score),
            is_match=False,
            mode=self.mode,
            proto_threshold=thresholds["proto"],
            feature_threshold=thresholds["feature"],
            reason=reason,
        )

    def thresholds_for(self, identity) -> dict[str, float]:
        """Return the effective prototype and feature thresholds."""

        return self._resolve_thresholds(identity)

    def prototype_score(self, query_feat, identity) -> float:
        """Return cosine score against the identity prototype."""

        return cosine_similarity(query_feat, identity.prototype)

    def max_feature_score(self, query_feat, identity) -> float:
        """Return the best cosine score against all stored sample features."""

        np = _load_numpy()
        features = np.asarray(identity.features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] == 0:
            return 0.0

        query = np.asarray(query_feat, dtype=np.float32).reshape(-1)
        feature_norms = np.linalg.norm(features, axis=1)
        query_norm = np.linalg.norm(query)
        denom = feature_norms * query_norm
        valid = denom > 1e-12
        if not np.any(valid):
            return 0.0

        scores = np.full((features.shape[0],), -1.0, dtype=np.float32)
        scores[valid] = features[valid].dot(query) / denom[valid]
        return float(np.max(scores))

    def _resolve_thresholds(self, identity) -> dict[str, float]:
        """Resolve matcher thresholds from CLI overrides or identity metadata."""

        identity_threshold = float(getattr(identity, "threshold", 0.80))
        feature_threshold = (
            float(self.feature_threshold)
            if self.feature_threshold is not None
            else identity_threshold
        )
        proto_threshold = (
            float(self.proto_threshold)
            if self.proto_threshold is not None
            else feature_threshold - self.proto_margin
        )

        return {
            "proto": proto_threshold,
            "feature": feature_threshold,
        }
