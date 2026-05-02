"""task2 主流程状态：导航、识别和播报集中在一个状态内完成。"""

from __future__ import annotations

import logging

from common.state_machine import State
from task2 import config
from task2.behaviors.kitchen_detector import KitchenDetector
from task2.states.utils import (
    camera_serial_for_role,
    navigate_to_station,
    safe_speak,
)

log = logging.getLogger("task2.kitchen_task")


class KitchenTask(State):
    """执行 task2 初版的全部非抓取主流程。"""

    def execute(self, ctx) -> str:
        """顺序完成餐桌物体播报和架子层类别播报。"""
        detector = _require_detector(ctx)

        self._scan_table(ctx, detector)
        
        # self._scan_table2(ctx, detector)
        self._scan_shelf(ctx, detector)

        return "release"
    
    def _scan_table2(self, ctx, detector: KitchenDetector) -> None:
        """导航到餐桌并识别餐桌上的待整理物品。"""
        safe_speak("I am moving to the kitchen table.")
        ok = navigate_to_station(config.STATION_TABLE2, timeout=config.NAV_TIMEOUT)
        if not ok:
            raise RuntimeError("导航到餐桌失败")

        ctx.current_area = "table"
        # safe_speak("I have arrived at the table. I will scan the objects now.")

        camera_serial = camera_serial_for_role(config.TABLE_SCAN_CAMERA_ROLE)
        detections = detector.detect(camera_serial, source_area="table")
        ctx.replace_detections("table", detections)

        if not detections:
            safe_speak("I did not recognize table objects clearly.")
            log.warning("餐桌扫描没有识别到物体")
            return

        safe_speak(f"I recognized {len(detections)} objects on the table.")
        for detected in detections:
            safe_speak(f"{detected.label}. ")

    def _scan_table(self, ctx, detector: KitchenDetector) -> None:
        """导航到餐桌并识别餐桌上的待整理物品。"""
        safe_speak("I am moving to the kitchen table.")
        ok = navigate_to_station(config.STATION_TABLE, timeout=config.NAV_TIMEOUT)
        if not ok:
            raise RuntimeError("导航到餐桌失败")

        ctx.current_area = "table"
        # safe_speak("I have arrived at the table. I will scan the objects now.")

        camera_serial = camera_serial_for_role(config.TABLE_SCAN_CAMERA_ROLE)
        detections = detector.detect(camera_serial, source_area="table")
        ctx.replace_detections("table", detections)

        if not detections:
            safe_speak("I did not recognize table objects clearly.")
            log.warning("餐桌扫描没有识别到物体")
            return

        safe_speak(f"I recognized {len(detections)} objects on the table.")
        for detected in detections:
            safe_speak(f"{detected.label}. ")

    def _scan_shelf(self, ctx, detector: KitchenDetector) -> None:
        """导航到橱柜并感知货架物品和早餐物品线索。"""
        safe_speak("I am moving to the cabinet to inspect the shelf.")
        ok = navigate_to_station(config.STATION_CABINET, timeout=config.NAV_TIMEOUT)
        if not ok:
            ctx.nav_failures += 1
            safe_speak("I could not reach the cabinet, so I will skip shelf perception.")
            log.warning("导航到橱柜失败，跳过货架识别")
            return

        ctx.current_area = "shelf"
        camera_serial = camera_serial_for_role(config.SHELF_SCAN_CAMERA_ROLE)
        detections = detector.detect(camera_serial, source_area="shelf")
        ctx.replace_detections("shelf", detections)

        if detections:
            labels = _unique_labels(detections)
            safe_speak(f"I recognized {', '.join(labels)} on the shelf.")
        else:
            safe_speak("I did not recognize shelf objects clearly.")
            log.warning("货架扫描没有识别到物体")

        for summary in ctx.summarize_shelf_layers():
            category_text = config.CATEGORY_SPEECH.get(summary.category, summary.category)
            ordinal = _ordinal(summary.layer)
            safe_speak(f"The {ordinal} shelf is for {category_text}.")

def _require_detector(ctx) -> KitchenDetector:
    """获取初始化阶段创建的视觉适配器。"""
    detector = getattr(ctx, "detector", None)
    if detector is None:
        detector = KitchenDetector()
    return detector


def _unique_labels(detections) -> list[str]:
    """按首次出现顺序返回识别标签，避免重复播报同类物品。"""
    labels: list[str] = []
    seen: set[str] = set()
    for detected in detections:
        if detected.label in seen:
            continue
        seen.add(detected.label)
        labels.append(detected.label)
    return labels


def _ordinal(number: int) -> str:
    """将 1-4 层编号转成英文序数。"""
    names = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
    }
    return names.get(number, str(number))
