"""task2 厨房视觉识别适配层。

本模块只封装视觉识别，不在导入时启动相机或加载模型。状态执行时按需
调用 KitchenDetector.detect()，将不同模型输出统一成 DetectedObject。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from task2 import config
from task2.context import (
    DetectedObject,
    classify_label,
    normalize_label,
)

log = logging.getLogger("task2.kitchen_detector")



class KitchenDetector:
    """懒加载 YOLO 模型，并对多帧检测结果做简单去重。"""

    def __init__(self):
        self._model: Any | None = None

    def detect(self, camera_serial: str, source_area: str) -> list[DetectedObject]:
        """从指定相机采样并返回 task2 统一识别结果。"""
        try:
            camera = self._get_camera(camera_serial)
        except Exception as exc:
            log.warning("相机不可用，返回空识别结果: %s", exc)
            return []

        tracks: list[DetectedObject] = []

        for _ in range(config.VISUAL_SAMPLE_COUNT):
            frame = self._wait_for_frame(camera)
            if frame is None:
                continue

            for detected in self._detect_frame(self._get_model(), frame, source_area):
                self._merge_detection(tracks, detected)

            time.sleep(config.VISUAL_SAMPLE_INTERVAL_S)

        return tracks

    def _get_camera(self, camera_serial: str):
        """按需获取相机对象，未预热时由 camera_manager 懒启动。"""
        from common.skills.camera import camera_manager

        return camera_manager.get(camera_serial)

    def _get_model(self) -> Any:
        """按需加载自训练模型。"""
        if self._model is not None:
            return self._model

        from ultralytics import YOLO
        self._model = YOLO(config.CUSTOM_MODEL_PATH)
        return self._model

    def _wait_for_frame(self, camera, timeout_s: float = 1.5):
        """等待相机缓存产生一帧彩色图像。"""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            color_frame, _ = camera.get_frames()
            if color_frame is not None:
                return color_frame
            time.sleep(0.05)
        return None

    def _detect_frame(self, model, frame, source_area: str) -> list[DetectedObject]:
        """对单帧图像执行检测并转为 DetectedObject。"""
        detections: list[DetectedObject] = []
        try:
            results = model(
                frame,
                conf=config.DETECTION_CONFIDENCE,
                verbose=False,
                stream=True,
            )
        except Exception as exc:
            log.warning("视觉模型推理失败: %s", exc)
            return detections

        for result in results:
            names = getattr(result, "names", None) or getattr(model, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                detected = self._box_to_detected_object(box, names, source_area)
                if detected is not None:
                    detections.append(detected)

        return detections

    def _box_to_detected_object(
        self,
        box,
        names: dict[int, str],
        source_area: str,
    ) -> DetectedObject | None:
        """把 YOLO box 对象转为统一识别结果。"""
        try:
            class_id = int(box.cls[0])
            raw_label = str(names.get(class_id, class_id))
            label = normalize_label(raw_label)
            if label in config.IGNORED_LABELS:
                return None

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        except Exception as exc:
            log.debug("解析检测框失败: %s", exc)
            return None

        category = classify_label(label)
        return DetectedObject(
            label=label,
            category=category,
            confidence=confidence,
            bbox=(x1, y1, x2, y2),
            source_area=source_area,
        )

    def _merge_detection(
        self,
        tracks: list[DetectedObject],
        detected: DetectedObject,
    ) -> None:
        """按同类标签和 bbox 重叠合并多帧检测结果。"""
        for index, current in enumerate(tracks):
            if current.label != detected.label:
                continue
            if _iou(current.bbox, detected.bbox) < config.DETECTION_IOU_THRESHOLD:
                continue
            if detected.confidence > current.confidence:
                tracks[index] = detected
            return

        tracks.append(detected)


def _iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    """计算两个 xyxy 检测框的 IOU。"""
    x1, y1, x2, y2 = first
    a1, b1, a2, b2 = second

    inter_x1 = max(x1, a1)
    inter_y1 = max(y1, b1)
    inter_x2 = min(x2, a2)
    inter_y2 = min(y2, b2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    first_area = max(0, x2 - x1) * max(0, y2 - y1)
    second_area = max(0, a2 - a1) * max(0, b2 - b1)
    union_area = first_area + second_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area
