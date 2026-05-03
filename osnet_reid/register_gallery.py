"""OSNet ReID 人物注册脚本。

作用：
    使用 OpenCV 打开相机 6，采集指定人物的多帧人体裁剪图，提取 OSNet
    特征后生成 prototype，并把 prototype 与多样本 features 一起写入特征库。
    默认多人入镜时暂停采样，避免把不同人注册到同一个身份。

用法：
    python3 -m osnet_reid.register_gallery --person-id host --name Host
    python3 -m osnet_reid.register_gallery --person-id guest_a --append
    python3 -m osnet_reid.register_gallery --person-id host --append --merge-existing

    窗口按键：
        s 开始采样
        p 暂停采样
        c 清空本次已采样内容
        q 或 Esc 取消注册
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .core import DEFAULT_MODEL_PATH, OSNetReID
from .detector import PersonDetection, PersonDetector, select_largest
from .gallery import DEFAULT_GALLERY_PATH, FeatureGallery, GalleryIdentity
from .preprocess import crop_and_letterbox_person
from .sources import OpenCVCameraSource


WINDOW_NAME = "OSNet ReID Gallery Registration"


def _load_cv2():
    """Import OpenCV only when the registration UI starts."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Missing OpenCV dependency. Install opencv-python before running "
            "python3 -m osnet_reid.register_gallery."
        ) from exc

    return cv2


def _load_numpy():
    """Import NumPy only when registered features are processed."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Missing NumPy dependency. Install numpy before registering ReID features."
        ) from exc

    return np


def parse_args() -> argparse.Namespace:
    """Parse command-line options for gallery registration."""

    parser = argparse.ArgumentParser(
        description="Open camera 6 and register one person into an OSNet feature gallery."
    )
    parser.add_argument("--person-id", required=True, help="Stable ID, e.g. host")
    parser.add_argument("--name", default="", help="Display name; defaults to person-id")
    parser.add_argument("--camera-index", type=int, default=6, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=1280, help="Camera frame width")
    parser.add_argument("--height", type=int, default=720, help="Camera frame height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS request")
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of valid person crops to register",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=0.20,
        help="Minimum seconds between accepted samples",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_GALLERY_PATH,
        help="Feature gallery NPZ output path",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Load an existing gallery and add or update this person",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="When person-id exists, merge old sample features with new samples",
    )
    parser.add_argument(
        "--max-stored-features",
        type=int,
        default=60,
        help="Maximum features kept for one identity when merging",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Stored same-person threshold for this identity",
    )
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
        "--allow-multiple",
        action="store_true",
        help="Allow sampling the largest box even when multiple people are visible",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start collecting immediately instead of waiting for key s",
    )
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Save accepted 256x128 person crops for debugging",
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "gallery" / "samples",
        help="Directory for optional accepted sample crops",
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


def _draw_detection(cv2, frame, detection: PersonDetection, selected: bool) -> None:
    """Draw one detected person box."""

    x1, y1, x2, y2 = detection.bbox
    color = (0, 255, 255) if selected else (255, 255, 0)
    label = f"person {detection.confidence:.2f} area {detection.area}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def _valid_detections(
    detections: list[PersonDetection],
    min_box_area: int,
) -> list[PersonDetection]:
    """Filter tiny person boxes before registration sampling."""

    return [detection for detection in detections if detection.area >= min_box_area]


def _can_sample(
    detections: list[PersonDetection],
    selected: PersonDetection | None,
    allow_multiple: bool,
) -> tuple[bool, str]:
    """Return whether the current frame is safe to sample."""

    if selected is None:
        return False, "No valid person detected"
    if len(detections) > 1 and not allow_multiple:
        return False, "Multiple people detected; sampling paused"
    return True, "Ready"


def _save_sample_crop(cv2, crop, sample_dir: Path, person_id: str, index: int) -> None:
    """Save one accepted registration crop for later visual inspection."""

    person_dir = sample_dir / person_id
    person_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(person_dir / f"{index:03d}.jpg"), crop)


def collect_registration_crops(args: argparse.Namespace):
    """Collect fixed-size person crops from camera 6 for one identity."""

    cv2 = _load_cv2()
    source = OpenCVCameraSource(
        index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    detector = PersonDetector(args.yolo_model, conf=args.det_conf)

    crops = []
    collecting = bool(args.auto_start)
    status = "Collecting" if collecting else "Press s to start collecting"
    last_sample_time = 0.0

    source.start()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while len(crops) < args.samples:
            frame = source.read()
            if frame is None:
                time.sleep(0.02)
                continue

            detections = _valid_detections(
                detector.detect(frame),
                min_box_area=args.min_box_area,
            )
            selected = select_largest(detections)
            can_sample, reason = _can_sample(
                detections,
                selected,
                allow_multiple=args.allow_multiple,
            )
            display = frame.copy()

            for detection in detections:
                _draw_detection(cv2, display, detection, detection is selected)

            now = time.time()
            if (
                collecting
                and can_sample
                and selected is not None
                and now - last_sample_time >= args.interval_s
            ):
                crop = crop_and_letterbox_person(frame, selected.bbox)
                if crop is not None:
                    crops.append(crop)
                    last_sample_time = now
                    status = f"Accepted sample {len(crops)}/{args.samples}"
                    if args.save_samples:
                        _save_sample_crop(
                            cv2,
                            crop,
                            args.sample_dir,
                            args.person_id,
                            len(crops),
                        )
                else:
                    status = "Selected person crop is invalid"
            elif collecting and not can_sample:
                status = reason

            _draw_text(
                cv2,
                display,
                f"Register {args.person_id}: {len(crops)}/{args.samples}",
                28,
            )
            _draw_text(
                cv2,
                display,
                status,
                56,
                (0, 255, 255) if collecting else (255, 255, 255),
            )
            _draw_text(
                cv2,
                display,
                "s=start  p=pause  c=clear  q/Esc=cancel",
                84,
            )

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("Registration cancelled")
            if key == ord("s"):
                collecting = True
                status = "Collecting"
            if key == ord("p"):
                collecting = False
                status = "Paused"
            if key == ord("c"):
                crops.clear()
                status = "Cleared samples"
                last_sample_time = 0.0
    finally:
        source.stop()
        cv2.destroyWindow(WINDOW_NAME)

    return crops


def _normalize_feature_matrix(features: list) -> tuple:
    """Stack normalized sample features and build their average prototype."""

    np = _load_numpy()
    feature_matrix = np.stack(features, axis=0).astype(np.float32)
    prototype = np.mean(feature_matrix, axis=0)
    norm = np.linalg.norm(prototype)
    if norm < 1e-12:
        raise RuntimeError("Prototype feature norm is too small")
    prototype = (prototype / norm).astype(np.float32)
    return feature_matrix, prototype


def _merge_features(old_features, new_features, max_count: int):
    """Merge old and new sample features, keeping the newest samples if capped."""

    np = _load_numpy()
    merged = np.concatenate([old_features, new_features], axis=0)
    if max_count > 0 and merged.shape[0] > max_count:
        merged = merged[-max_count:]
    return merged.astype(np.float32)


def build_registered_identity(
    args: argparse.Namespace,
    crops,
    existing: GalleryIdentity | None,
) -> GalleryIdentity:
    """Extract features from accepted crops and create the gallery identity."""

    reid = OSNetReID(
        model_path=args.model_path,
        device=args.device,
        threshold=args.threshold,
    )
    features = [reid.extract(crop) for crop in crops]
    feature_matrix, prototype = _normalize_feature_matrix(features)

    if existing is not None and args.merge_existing:
        feature_matrix = _merge_features(
            existing.features,
            feature_matrix,
            max_count=args.max_stored_features,
        )
        feature_matrix, prototype = _normalize_feature_matrix(
            [row for row in feature_matrix]
        )

    return GalleryIdentity(
        person_id=args.person_id,
        name=args.name or args.person_id,
        prototype=prototype,
        features=feature_matrix,
        threshold=args.threshold,
    )


def save_registration(args: argparse.Namespace, identity: GalleryIdentity) -> None:
    """Save one registered identity into a new or existing gallery."""

    if args.output.exists():
        if not args.append:
            raise FileExistsError(
                f"Gallery already exists: {args.output}. Use --append to update it."
            )
        gallery = FeatureGallery.load(args.output)
    else:
        gallery = FeatureGallery()

    gallery.upsert(identity)
    gallery.save(args.output)


def run_registration(args: argparse.Namespace) -> None:
    """Collect samples, extract features, and save the feature gallery."""

    if args.samples <= 0:
        raise ValueError("--samples must be greater than 0")
    if args.merge_existing and not args.append:
        raise ValueError("--merge-existing requires --append")
    if args.output.exists() and not args.append:
        raise FileExistsError(
            f"Gallery already exists: {args.output}. Use --append to update it."
        )

    existing = None
    if args.output.exists() and args.append:
        existing = FeatureGallery.load(args.output).get(args.person_id)

    crops = collect_registration_crops(args)
    identity = build_registered_identity(args, crops, existing)
    save_registration(args, identity)
    print(
        f"Registered {identity.person_id} ({identity.name}) with "
        f"{identity.feature_count} features -> {args.output}"
    )


def main() -> None:
    """Run the CLI entry point."""

    try:
        run_registration(parse_args())
    except KeyboardInterrupt as exc:
        print(str(exc) or "Registration cancelled", file=sys.stderr)
        raise SystemExit(130) from exc


if __name__ == "__main__":
    main()
