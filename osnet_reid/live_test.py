"""OSNet ReID 独立实时测试脚本。

作用：
    使用 OpenCV 打开相机 6，YOLO 检测人体，按最大人物框注册目标，并在
    后续画面中实时显示 ReID 相似度和匹配结果。不依赖 camera_manager.py。

用法：
    python3 -m osnet_reid.live_test --camera-index 6

    窗口按键：
        r 注册当前最大人物框
        c 清空目标
        q 或 Esc 退出
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .core import DEFAULT_MODEL_PATH, OSNetReID
from .detector import PersonDetection, PersonDetector, select_largest
from .preprocess import crop_and_letterbox_person
from .sources import OpenCVCameraSource


WINDOW_NAME = "OSNet ReID Live Test"


def _load_cv2():
    """Import OpenCV only when the live test starts."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Missing OpenCV dependency. Install opencv-python before running "
            "python3 -m osnet_reid.live_test."
        ) from exc

    return cv2


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the standalone live test."""

    parser = argparse.ArgumentParser(
        description="Open camera 6, register a person, and test OSNet ReID live."
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
        "--threshold",
        type=float,
        default=0.80,
        help="Cosine threshold for same-person matching",
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
    return parser.parse_args()


def _draw_text(cv2, frame: np.ndarray, line: str, y: int, color=(255, 255, 255)) -> None:
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


def _score_detection(
    reid: OSNetReID,
    frame: np.ndarray,
    detection: PersonDetection,
    target_proto: np.ndarray | None,
) -> tuple[float | None, bool, np.ndarray | None]:
    """Prepare one detection crop and optionally score it against the target."""

    crop = crop_and_letterbox_person(frame, detection.bbox)
    if crop is None:
        return None, False, None

    if target_proto is None:
        return None, False, crop

    feat = reid.extract(crop)
    score = reid.similarity(feat, target_proto)
    return score, score >= reid.threshold, crop


def _draw_detection(
    cv2,
    frame: np.ndarray,
    detection: PersonDetection,
    score: float | None,
    is_match: bool,
    is_selected: bool,
) -> None:
    """Draw one detection box and optional ReID score."""

    x1, y1, x2, y2 = detection.bbox
    if score is None:
        color = (0, 255, 255) if is_selected else (255, 255, 0)
        label = f"person {detection.confidence:.2f}"
    else:
        color = (0, 220, 0) if is_match else (0, 0, 255)
        label = f"score {score:.3f}"
        if is_match:
            label += " match"

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


def run_live_test(args: argparse.Namespace) -> None:
    """Run the OpenCV camera loop for registration and live matching."""

    cv2 = _load_cv2()
    source = OpenCVCameraSource(
        index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    detector = PersonDetector(args.yolo_model, conf=args.det_conf)
    reid = OSNetReID(
        model_path=args.model_path,
        device=args.device,
        threshold=args.threshold,
    )

    target_proto: np.ndarray | None = None
    selected_crop: np.ndarray | None = None
    status = "Press r to register largest person, c to clear, q/Esc to quit"
    last_error = ""

    source.start()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = source.read()
            if frame is None:
                time.sleep(0.02)
                continue

            detections = detector.detect(frame)
            selected = select_largest(detections)
            selected_crop = None
            display = frame.copy()
            best_score: float | None = None

            for detection in detections:
                try:
                    score, is_match, crop = _score_detection(
                        reid,
                        frame,
                        detection,
                        target_proto,
                    )
                except Exception as exc:
                    last_error = f"ReID error: {exc}"
                    score, is_match, crop = None, False, None

                if detection is selected:
                    selected_crop = crop
                if score is not None:
                    best_score = score if best_score is None else max(best_score, score)

                _draw_detection(
                    cv2,
                    display,
                    detection,
                    score,
                    is_match,
                    detection is selected,
                )

            if target_proto is None:
                status_line = "Target: not registered"
            elif best_score is None:
                status_line = "Target: registered, no valid score"
            else:
                status_line = f"Target: registered, best score {best_score:.3f}"

            _draw_text(cv2, display, status_line, 28, (255, 255, 255))
            _draw_text(cv2, display, status, 56, (255, 255, 255))
            if last_error:
                _draw_text(cv2, display, last_error[:100], 84, (0, 180, 255))

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                target_proto = None
                status = "Target cleared"
                last_error = ""
            if key == ord("r"):
                if selected_crop is None:
                    status = "No valid person crop to register"
                else:
                    try:
                        target_proto = reid.build_prototype([selected_crop])
                        status = "Registered largest person as target"
                        last_error = ""
                    except Exception as exc:
                        status = "Registration failed"
                        last_error = str(exc)
    finally:
        source.stop()
        cv2.destroyWindow(WINDOW_NAME)


def main() -> None:
    """Run the CLI entry point."""

    run_live_test(parse_args())


if __name__ == "__main__":
    main()
