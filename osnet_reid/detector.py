"""YOLO 人体检测工具。

作用：
    延迟加载 ultralytics YOLO，只检测 person 类，并把结果整理成
    PersonDetection。注册和实时测试脚本都通过这里获取人物框。

用法：
    detector = PersonDetector("yolov8n.pt", conf=0.35)
    detections = detector.detect(frame_bgr)
    selected = select_largest(detections)
"""

from __future__ import annotations

from dataclasses import dataclass

from .preprocess import BBox


@dataclass(frozen=True)
class PersonDetection:
    """One person detection in xyxy image coordinates."""

    bbox: BBox
    confidence: float

    @property
    def area(self) -> int:
        """Return the bounding-box area in pixels."""

        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


def _load_yolo():
    """Import Ultralytics YOLO only when the detector is created."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Missing YOLO dependency. Install ultralytics before running "
            "osnet_reid live detection."
        ) from exc

    return YOLO


class PersonDetector:
    """Detect person boxes with an Ultralytics YOLO model.

    Usage:
        detector = PersonDetector("yolov8n.pt")
        detections = detector.detect(frame_bgr)
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.35,
        person_class_id: int = 0,
    ) -> None:
        """Load the YOLO model used for person detection."""

        yolo_cls = _load_yolo()
        self.model_path = model_path
        self.conf = float(conf)
        self.person_class_id = int(person_class_id)
        self._model = yolo_cls(model_path)

    def detect(self, frame_bgr: np.ndarray) -> list[PersonDetection]:
        """Return person detections from one BGR frame."""

        if frame_bgr is None:
            return []

        detections: list[PersonDetection] = []
        results = self._model(
            frame_bgr,
            conf=self.conf,
            classes=[self.person_class_id],
            verbose=False,
            stream=True,
        )

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                if cls_id != self.person_class_id or score < self.conf:
                    continue

                x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append(
                    PersonDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=score,
                    )
                )

        return detections


def select_largest(detections: list[PersonDetection]) -> PersonDetection | None:
    """Select the largest person box from the current frame."""

    if not detections:
        return None

    return max(detections, key=lambda item: item.area)
