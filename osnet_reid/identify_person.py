"""OSNet ReID 指定人物在线识别脚本。

作用：
    从特征库加载指定 person_id，使用 OpenCV 打开相机 6，检测画面中的所有人，
    并判断谁是该目标身份。可选在线更新只修改内存里的运行时特征，不写回硬盘
    特征库。

用法：
    python3 -m osnet_reid.identify_person --person-id host
    python3 -m osnet_reid.identify_person --person-id host --online-update
    python3 -m osnet_reid.identify_person --person-id host --match-mode max-only

    窗口按键：
        u 开关在线更新
        q 或 Esc 退出
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from .core import DEFAULT_MODEL_PATH, OSNetReID
from .detector import PersonDetection, PersonDetector
from .gallery import DEFAULT_GALLERY_PATH, FeatureGallery, GalleryIdentity
from .matcher import MATCH_MODES, IdentityMatcher, MatchScore
from .preprocess import crop_and_letterbox_person
from .sources import OpenCVCameraSource


WINDOW_NAME = "OSNet ReID Identify Person"


def _load_cv2():
    """Import OpenCV only when the live identify UI starts."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Missing OpenCV dependency. Install opencv-python before running "
            "python3 -m osnet_reid.identify_person."
        ) from exc

    return cv2


def _load_numpy():
    """Import NumPy only when runtime identity features are updated."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Missing NumPy dependency. Install numpy before identifying ReID targets."
        ) from exc

    return np


@dataclass
class DetectionMatch:
    """One detected person with optional ReID matching result."""

    detection: PersonDetection
    crop: object | None = None
    query_feat: object | None = None
    score: MatchScore | None = None
    error: str = ""


class RuntimeIdentity:
    """In-memory identity that can absorb online features without saving to disk."""

    def __init__(self, identity: GalleryIdentity) -> None:
        """Copy disk features into runtime memory."""

        np = _load_numpy()
        self.person_id = identity.person_id
        self.name = identity.name
        self.threshold = float(identity.threshold)
        self.created_at = identity.created_at
        self.updated_at = identity.updated_at
        self.base_features = np.asarray(identity.features, dtype=np.float32).copy()
        self.runtime_features = np.zeros(
            (0, self.base_features.shape[1]),
            dtype=np.float32,
        )
        self.prototype = np.asarray(identity.prototype, dtype=np.float32).copy()

    @property
    def runtime_count(self) -> int:
        """Return how many online features have been added in this process."""

        return int(self.runtime_features.shape[0])

    @property
    def total_count(self) -> int:
        """Return total base and online features visible to the matcher."""

        return int(self.base_features.shape[0] + self.runtime_features.shape[0])

    def as_identity(self) -> GalleryIdentity:
        """Return a GalleryIdentity view for matcher compatibility."""

        return GalleryIdentity(
            person_id=self.person_id,
            name=self.name,
            prototype=self.prototype,
            features=self._all_features(),
            threshold=self.threshold,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def add_feature(self, query_feat, max_total_features: int) -> bool:
        """Add one online feature and recompute runtime prototype in memory."""

        np = _load_numpy()
        base_count = int(self.base_features.shape[0])
        online_cap = max(0, int(max_total_features) - base_count)
        if online_cap <= 0:
            return False

        query = np.asarray(query_feat, dtype=np.float32).reshape(1, -1)
        self.runtime_features = np.concatenate(
            [self.runtime_features, query],
            axis=0,
        )
        if self.runtime_features.shape[0] > online_cap:
            self.runtime_features = self.runtime_features[-online_cap:]

        all_features = self._all_features()
        prototype = np.mean(all_features, axis=0)
        norm = np.linalg.norm(prototype)
        if norm < 1e-12:
            return False

        self.prototype = (prototype / norm).astype(np.float32)
        return True

    def _all_features(self):
        """Return base and online features as one feature matrix."""

        np = _load_numpy()
        if self.runtime_features.shape[0] == 0:
            return self.base_features.copy()

        return np.concatenate([self.base_features, self.runtime_features], axis=0)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for live identity matching."""

    parser = argparse.ArgumentParser(
        description="Load a registered identity and identify it from camera 6."
    )
    parser.add_argument("--person-id", required=True, help="Registered person ID to match")
    parser.add_argument(
        "--gallery",
        type=Path,
        default=DEFAULT_GALLERY_PATH,
        help="Feature gallery NPZ path",
    )
    parser.add_argument("--camera-index", type=int, default=6, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=1280, help="Camera frame width")
    parser.add_argument("--height", type=int, default=720, help="Camera frame height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS request")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Local OSNet weight path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device for OSNet",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model used for person detection",
    )
    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.35,
        help="YOLO person detection confidence",
    )
    parser.add_argument(
        "--min-box-area",
        type=int,
        default=5000,
        help="Ignore person boxes smaller than this pixel area",
    )
    parser.add_argument(
        "--match-mode",
        choices=MATCH_MODES,
        default="strict",
        help="Identity matching mode",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the identity threshold used as the base threshold",
    )
    parser.add_argument(
        "--proto-threshold",
        type=float,
        default=None,
        help="Override prototype threshold; default is feature threshold minus 0.05",
    )
    parser.add_argument(
        "--feature-threshold",
        type=float,
        default=None,
        help="Override max-feature threshold; default is identity threshold",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Strict-mode candidates allowed through prototype prefilter; 0 disables",
    )
    parser.add_argument(
        "--online-update",
        action="store_true",
        help="Update runtime features in memory after high-confidence matches",
    )
    parser.add_argument(
        "--update-threshold",
        type=float,
        default=None,
        help="Final-score threshold for online update; default feature threshold plus 0.05",
    )
    parser.add_argument(
        "--update-min-interval-s",
        type=float,
        default=0.50,
        help="Minimum seconds between online feature updates",
    )
    parser.add_argument(
        "--max-runtime-features",
        type=int,
        default=80,
        help="Maximum total in-memory features; base disk features are never dropped",
    )
    return parser.parse_args()


def _draw_text(cv2, frame, line: str, y: int, color=(255, 255, 255)) -> None:
    """Draw one readable status line on the frame."""

    cv2.putText(
        frame,
        line,
        (16, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        line,
        (16, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_detection(cv2, frame, match: DetectionMatch, is_best: bool) -> None:
    """Draw one detected person box and ReID score."""

    x1, y1, x2, y2 = match.detection.bbox
    if match.score is None:
        color = (0, 255, 255)
        label = match.error or "no score"
    elif match.score.is_match:
        color = (0, 220, 0)
        label = _format_score_label(match.score, suffix="MATCH")
    else:
        color = (0, 0, 255)
        suffix = match.score.reason.upper() if match.score.reason else "NO"
        label = _format_score_label(match.score, suffix=suffix)

    thickness = 3 if is_best else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        frame,
        label[:80],
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _format_score_label(score: MatchScore, suffix: str) -> str:
    """Format compact per-box score text."""

    return (
        f"proto={score.proto_score:.3f} "
        f"max={score.max_feature_score:.3f} {suffix}"
    )


def _valid_detections(
    detections: list[PersonDetection],
    min_box_area: int,
) -> list[PersonDetection]:
    """Filter tiny person boxes before feature extraction."""

    return [detection for detection in detections if detection.area >= min_box_area]


def _resolve_thresholds(args: argparse.Namespace, identity: GalleryIdentity) -> dict[str, float]:
    """Resolve CLI threshold overrides into matcher and update thresholds."""

    base_threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(identity.threshold)
    )
    feature_threshold = (
        float(args.feature_threshold)
        if args.feature_threshold is not None
        else base_threshold
    )
    proto_threshold = (
        float(args.proto_threshold)
        if args.proto_threshold is not None
        else feature_threshold - 0.05
    )
    update_threshold = (
        float(args.update_threshold)
        if args.update_threshold is not None
        else feature_threshold + 0.05
    )
    return {
        "feature": feature_threshold,
        "proto": proto_threshold,
        "update": update_threshold,
    }


def _load_identity(args: argparse.Namespace) -> GalleryIdentity:
    """Load one identity from the feature gallery."""

    gallery = FeatureGallery.load(args.gallery)
    identity = gallery.get(args.person_id)
    if identity is None:
        known = ", ".join(item.person_id for item in gallery.identities) or "<empty>"
        raise KeyError(f"person-id {args.person_id!r} not found. Known IDs: {known}")
    return identity


def _score_detections(
    frame,
    detections: list[PersonDetection],
    reid: OSNetReID,
    matcher: IdentityMatcher,
    runtime_identity: RuntimeIdentity,
    top_k: int,
) -> list[DetectionMatch]:
    """Extract query features and score all detections against the target."""

    identity_view = runtime_identity.as_identity()
    matches: list[DetectionMatch] = []

    for detection in detections:
        match = DetectionMatch(detection=detection)
        try:
            match.crop = crop_and_letterbox_person(frame, detection.bbox)
            if match.crop is None:
                match.error = "invalid crop"
                matches.append(match)
                continue

            match.query_feat = reid.extract(match.crop)
            proto_score = matcher.prototype_score(match.query_feat, identity_view)
            match.score = MatchScore(
                proto_score=proto_score,
                max_feature_score=0.0,
                final_score=proto_score,
                is_match=False,
                mode=matcher.mode,
                proto_threshold=matcher.thresholds_for(identity_view)["proto"],
                feature_threshold=matcher.thresholds_for(identity_view)["feature"],
                reason="pending",
            )
        except Exception as exc:
            match.error = str(exc)
        matches.append(match)

    candidates = [
        item
        for item in matches
        if item.query_feat is not None and item.score is not None
    ]
    candidates.sort(key=lambda item: item.score.proto_score, reverse=True)

    if matcher.mode == "strict" and top_k > 0:
        allowed_ids = {id(item) for item in candidates[:top_k]}
    else:
        allowed_ids = {id(item) for item in candidates}

    for item in candidates:
        if id(item) not in allowed_ids:
            item.score = matcher.filtered_score(
                item.score.proto_score,
                identity_view,
                reason="prefiltered",
            )
            continue
        item.score = matcher.score(
            item.query_feat,
            identity_view,
            proto_score=item.score.proto_score,
        )

    return matches


def _best_match(matches: list[DetectionMatch]) -> DetectionMatch | None:
    """Return the highest scoring matched target in the current frame."""

    matched = [
        item
        for item in matches
        if item.score is not None and item.score.is_match
    ]
    if not matched:
        return None

    return max(matched, key=lambda item: item.score.final_score)


def _maybe_update_runtime_identity(
    args: argparse.Namespace,
    runtime_identity: RuntimeIdentity,
    matches: list[DetectionMatch],
    update_threshold: float,
    last_update_time: float,
    online_update: bool,
) -> tuple[float, bool]:
    """Update runtime features in memory after one unambiguous strong match."""

    if not online_update:
        return last_update_time, False

    now = time.time()
    if now - last_update_time < args.update_min_interval_s:
        return last_update_time, False

    matched = [
        item
        for item in matches
        if item.score is not None and item.score.is_match
    ]
    if len(matched) != 1:
        return last_update_time, False

    best = matched[0]
    if best.query_feat is None or best.score is None:
        return last_update_time, False
    if best.score.final_score < update_threshold:
        return last_update_time, False

    updated = runtime_identity.add_feature(
        best.query_feat,
        max_total_features=args.max_runtime_features,
    )
    if not updated:
        return last_update_time, False

    return now, True


def run_identification(args: argparse.Namespace) -> None:
    """Run live camera identification for one registered person."""

    cv2 = _load_cv2()
    identity = _load_identity(args)
    runtime_identity = RuntimeIdentity(identity)
    thresholds = _resolve_thresholds(args, identity)
    matcher = IdentityMatcher(
        mode=args.match_mode,
        proto_threshold=thresholds["proto"],
        feature_threshold=thresholds["feature"],
    )
    reid = OSNetReID(
        model_path=args.model_path,
        device=args.device,
        threshold=thresholds["feature"],
    )
    detector = PersonDetector(args.yolo_model, conf=args.det_conf)
    source = OpenCVCameraSource(
        index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    online_update = bool(args.online_update)
    last_update_time = 0.0
    last_status = ""

    source.start()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = source.read()
            if frame is None:
                time.sleep(0.02)
                continue

            detections = _valid_detections(
                detector.detect(frame),
                min_box_area=args.min_box_area,
            )
            display = frame.copy()
            matches = _score_detections(
                frame,
                detections,
                reid,
                matcher,
                runtime_identity,
                top_k=args.top_k,
            )
            best = _best_match(matches)

            last_update_time, updated = _maybe_update_runtime_identity(
                args,
                runtime_identity,
                matches,
                update_threshold=thresholds["update"],
                last_update_time=last_update_time,
                online_update=online_update,
            )
            if updated:
                last_status = "runtime feature updated"

            for match in matches:
                _draw_detection(cv2, display, match, match is best)

            best_score = best.score.final_score if best and best.score else 0.0
            update_text = "on" if online_update else "off"
            _draw_text(
                cv2,
                display,
                (
                    f"Target: {runtime_identity.person_id} ({runtime_identity.name}) "
                    f"| mode={args.match_mode} | best={best_score:.3f} "
                    f"| runtime={runtime_identity.runtime_count} | update={update_text}"
                ),
                28,
            )
            _draw_text(
                cv2,
                display,
                (
                    f"proto>={thresholds['proto']:.2f} "
                    f"max>={thresholds['feature']:.2f} "
                    f"update>={thresholds['update']:.2f} | u=toggle update q/Esc=quit"
                ),
                56,
            )
            if last_status:
                _draw_text(cv2, display, last_status, 84, (0, 255, 255))

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("u"):
                online_update = not online_update
                last_status = f"online update {'enabled' if online_update else 'disabled'}"
    finally:
        source.stop()
        cv2.destroyWindow(WINDOW_NAME)


def main() -> None:
    """Run the CLI entry point."""

    run_identification(parse_args())


if __name__ == "__main__":
    main()
