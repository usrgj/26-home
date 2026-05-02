"""task2 拾取与放置 —— 状态间共享数据容器"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from task2 import config


@dataclass
class DetectedObject:
    """视觉识别结果，bbox 坐标为相机图像像素坐标 xyxy。"""

    label: str
    category: str
    confidence: float
    bbox: tuple[int, int, int, int]
    source_area: str


@dataclass
class ShelfLayerSummary:
    """架子单层的类别判断结果。"""

    layer: int
    category: str
    objects: list[DetectedObject]


def normalize_label(label: str) -> str:
    """统一模型输出标签，便于跨模型匹配和分类。"""
    normalized = (label or "").strip().lower().replace("-", "_").replace(" ", "_")
    return config.LABEL_ALIASES.get(normalized, normalized)


def classify_label(label: str) -> str:
    """根据标签给出 task2 的任务类别。"""
    normalized = normalize_label(label)
    return config.OBJECT_CATEGORY_MAP.get(normalized, config.SHELF_CATEGORY_UNKNOWN)


class TaskContext:
    """task2 状态间共享数据，避免识别结果和播报计划散落在状态里。"""

    def __init__(self):
        self.detector: Optional[object] = None
        self.table_objects: list[DetectedObject] = []
        self.shelf_objects: list[DetectedObject] = []
        self.current_area: str = ""

        self.failed_state: str = ""
        self.nav_failures: int = 0
        self.table_scan_failures: int = 0
        self.shelf_scan_failures: int = 0

    def add_detection(self, detected: DetectedObject) -> None:
        """记录单个识别结果。"""
        if detected.source_area == "table":
            self.table_objects.append(detected)
        elif detected.source_area == "shelf":
            self.shelf_objects.append(detected)

    def replace_detections(
        self,
        source_area: str,
        detections: Iterable[DetectedObject],
    ) -> None:
        """用一次扫描结果替换指定区域的旧结果。"""
        items = list(detections)
        if source_area == "table":
            self.table_objects = []
        elif source_area == "shelf":
            self.shelf_objects = []

        for detected in items:
            self.add_detection(detected)

    def summarize_shelf_layers(self) -> list[ShelfLayerSummary]:
        """根据检测框中心点和层框配置，统计每层最可能的物品类别。"""
        summaries: list[ShelfLayerSummary] = []
        for layer_config in config.SHELF_LAYER_BOXES:
            layer = int(layer_config["layer"])
            layer_box = tuple(layer_config["box"])
            if not _is_valid_box(layer_box):
                summaries.append(_unknown_layer_summary(layer))
                continue

            layer_objects = [
                detected
                for detected in self.shelf_objects
                if _point_in_box(_bbox_center(detected.bbox), layer_box)
            ]
            category = _select_layer_category(layer_objects)
            summaries.append(
                ShelfLayerSummary(
                    layer=layer,
                    category=category,
                    objects=layer_objects,
                )
            )

        return summaries


def _unknown_layer_summary(layer: int) -> ShelfLayerSummary:
    """生成未配置或未识别层的默认结果。"""
    return ShelfLayerSummary(
        layer=layer,
        category=config.SHELF_CATEGORY_UNKNOWN,
        objects=[],
    )


def _is_valid_box(box: tuple[int, int, int, int]) -> bool:
    """判断层框是否为有效 xyxy 像素框。"""
    x1, y1, x2, y2 = box
    return x2 > x1 and y2 > y1


def _bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    """计算检测框中心点。"""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _point_in_box(point: tuple[float, float], box: tuple[int, int, int, int]) -> bool:
    """判断点是否落在层框内，边界视为框内。"""
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _select_layer_category(objects: list[DetectedObject]) -> str:
    """按数量投票选择层类别；数量相同时用最高置信度打破平局。"""
    scores: dict[str, dict[str, float]] = {}
    for detected in objects:
        if detected.category == config.SHELF_CATEGORY_UNKNOWN:
            continue
        score = scores.setdefault(detected.category, {"count": 0.0, "confidence": 0.0})
        score["count"] += 1
        score["confidence"] = max(score["confidence"], detected.confidence)

    if not scores:
        return config.SHELF_CATEGORY_UNKNOWN

    return max(
        scores,
        key=lambda category: (
            scores[category]["count"],
            scores[category]["confidence"],
            category,
        ),
    )
