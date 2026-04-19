"""
状态1：等待铃声 → 导航到门 → 问好 → 询问姓名饮品
     → 后台提取视觉特征 → 引导就坐

本状态内部循环执行两次（两位客人），全部就座后进入状态2。
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from common.config import CAMERA_HEAD
from common.skills.agv_api import agv, wait_nav
from common.skills.arm import left_arm, right_arm
from common.skills.audio_module.voice_assiant import (
    doorbell,
    extract_drink,
    extract_name,
    voice_assistant,
)
from common.skills.camera import camera_manager
from common.skills.head_control import pan_tilt
from common.state_machine import State
from task1 import config
from task1.behaviors.vision.client import analyze_person_features
from task1.behaviors.vision import GazeAPI

_DEFAULT_DETECTION_CONF = 0.35
_GUEST_CROP_WINDOW_S = 2.0
_GUEST_CROP_MAX_SAMPLES = 5
_FEATURE_WAIT_TIMEOUT_S = 2.0
_GAZE_LOOP_INTERVAL_S = 0.1


@dataclass
class FeatureExtractionJob:
    """后台外貌提取任务。"""

    guest_index: int
    done_event: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    thread: threading.Thread | None = None


class ReceiveGuest(State):
    def __init__(self):
        self._model: YOLO | None = None
        self._cam_head = camera_manager.get(CAMERA_HEAD)
        self._feature_jobs: dict[int, FeatureExtractionJob] = {}

    def execute(self, ctx) -> str:
        model = self._get_model()
        self.gaze_api = GazeAPI(model)
        self._feature_jobs = {}

        voice_assistant.set_recording(1)

        while ctx.current_guest_index < len(ctx.guests):
            guest_index = ctx.current_guest_index
            guest = ctx.current_guest

            pan_tilt.home()
            agv.navigate_to(agv.get_current_station(), config.STATION_START)
            wait_nav(timeout=config.NAV_TIMEOUT)

            
            # 在这个位置进行第一次观察，座位状态更新，
            update_seats(ctx, model, self._cam_head, box_key="box1")
            # 等待门铃
            doorbell.start()
            doorbell.wait_for_doorbell(timeout=30)
            doorbell.stop()
            # 导航到门口
            agv.navigate_to(config.STATION_START, config.STATION_DOOR)
            wait_nav(timeout=config.NAV_TIMEOUT)

            # TODO 开门
            # 注视
            gaze_thread, gaze_stop_event = self.gaze_api.start_gaze_tracking_nearest_person(pan_tilt, self._cam_head, duration=45)
            # 询问姓名和喜爱饮品，同时在后台提取视觉特征
            try:
                text = quest_and_answer("Welcome! May I know your name ?")
                guest.name = extract_name(text)

                if guest_index == 0:
                    crop = _select_best_guest_crop(model, self._cam_head)
                    if crop is not None:
                        self._feature_jobs[guest_index] = _start_feature_extraction(
                            guest_index=guest_index,
                            crop=crop,
                        )

                text = quest_and_answer("What is your favorite drink ?")
                guest.favorite_drink = extract_drink(text)
            finally:
                # 结束注视
                gaze_stop_event.set()
                gaze_thread.join()
                pan_tilt.home()#回中

            # 导航到空位置
            seat_id = ctx.find_free_seat()
            # 如果找不到空座位，则去第二个位置观察，再找一次
            if seat_id is None:
                agv.navigate_to(agv.get_current_station(), config.STATION_OBSERVATION)
                wait_nav(timeout=config.NAV_TIMEOUT)
                update_seats(ctx, model, self._cam_head, box_key="box2")
                seat_id = ctx.find_free_seat()

            if seat_id is not None:
                nav_id, angle = _get_seat_navigation_target(seat_id)
                voice_assistant.speak("Please follow me, I will show you to your seat.")
                pan_tilt.home()
                agv.navigate_to(agv.get_current_station() or "", nav_id, angle)
                wait_nav(timeout=config.NAV_TIMEOUT)
                voice_assistant.speak("Please have a seat here.")

                guest.seat_id = seat_id
                ctx.occupy_seat(seat_id)
            else:
                voice_assistant.speak("I'm sorry, there are no free seats available.")

            ctx.current_guest_index += 1

        for guest_index, job in self._feature_jobs.items():
            features = _collect_feature_result(job, timeout_s=_FEATURE_WAIT_TIMEOUT_S)
            if features:
                ctx.guests[guest_index].visual_features = features

        return "introduce"

    def _get_model(self) -> YOLO:
        """延迟初始化 YOLO，避免模块导入时立刻加载模型。"""
        if self._model is None:
            self._model = YOLO("yolov8n.pt")
        return self._model


def quest_and_answer(text: str) -> str:
    """进行一次询问，和一次识别回答结果
    text: 询问内容
    return: 识别结果
    """

    for i in range(config.ASK_RETRIES):
        voice_assistant.speak(text)
        recognized_text = _record_and_recognize_text()
        print(f"识别到回答：{recognized_text}")

        if recognized_text:
            return recognized_text

    return ""



def _record_and_recognize_text() -> str:
    """录制一段语音并调用现有识别接口。"""
    audio_frames = voice_assistant.record_utterance()
    if not audio_frames:
        return ""

    recognize_speech = getattr(voice_assistant, "recognize_speech", None)
    if callable(recognize_speech):
        return (recognize_speech(audio_frames) or "").strip()

    return (voice_assistant.recognize(audio_frames) or "").strip()



def update_seats(ctx, yolo_model, camera, box_key: str) -> None:
    """用当前视角里可见的座位框更新占用状态。
    boxe key就是对应的座位框，第一个观察用 box1，第二次观察用 box2，后续如果需要再加 box3 之类的
    """
    color_frame, depth_frame = _wait_for_latest_frame(camera, timeout_s=1.5)
    if color_frame is None:
        return

    person_boxes = _detect_person_boxes(
        yolo_model,
        color_frame,
        conf=_DEFAULT_DETECTION_CONF,
        person_class_id=0,
    )

    for seat in ctx.seats:
        seat_box = tuple(seat.get(box_key, (0, 0, 0, 0)))
        if not _is_valid_bbox(seat_box):
            continue

        occupied = any(_person_overlaps_seat(person_box, seat_box) for person_box in person_boxes)
        if occupied:
            seat["occupied"] = True
        elif seat["occupied"] is None:
            seat["occupied"] = False


def _get_seat_navigation_target(seat_id: str) -> tuple[str, float]:
    """把 seat_id 映射到真正的导航点。"""
    for seat_mapping in config.SEATS_MAPPING:
        if seat_mapping["seat_id"] == seat_id:
            return seat_mapping["nav_id"], seat_mapping["angle"]
    raise KeyError(f"未找到 seat_id={seat_id} 对应的导航配置")


def _select_best_guest_crop(
    yolo_model,
    camera,
    window_s: float = _GUEST_CROP_WINDOW_S,
    max_samples: int = _GUEST_CROP_MAX_SAMPLES,
) -> np.ndarray | None:
    """在稳定注视窗口内抓取若干帧，选出最佳人物裁剪图。"""
    deadline = time.time() + window_s
    best_crop: np.ndarray | None = None
    best_score = float("-inf")
    accepted = 0

    while time.time() < deadline and accepted < max_samples:
        color_frame, depth_frame = _wait_for_latest_frame(camera, timeout_s=0.4)
        if color_frame is None:
            continue

        person_boxes = _detect_person_boxes(
            yolo_model,
            color_frame,
            conf=_DEFAULT_DETECTION_CONF,
            person_class_id=0,
        )
        if not person_boxes:
            time.sleep(0.05)
            continue

        bbox = _select_closest_person(person_boxes, depth_frame)
        crop = _crop_person_with_margin(color_frame, bbox)
        if crop is None:
            time.sleep(0.05)
            continue

        score = _score_person_crop(bbox, color_frame.shape[1], color_frame.shape[0])
        if score > best_score:
            best_score = score
            best_crop = crop.copy()

        accepted += 1
        time.sleep(0.08)

    return best_crop


def _start_feature_extraction(guest_index: int, crop: np.ndarray) -> FeatureExtractionJob:
    """启动后台外貌特征提取任务。"""
    job = FeatureExtractionJob(guest_index=guest_index)

    def worker() -> None:
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = tmp.name

            if not cv2.imwrite(temp_path, crop):
                raise RuntimeError("无法写入外貌提取临时图像")

            features = analyze_person_features(temp_path)
            if isinstance(features, dict):
                job.result = features
        except Exception as exc:
            job.error = str(exc)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            job.done_event.set()

    job.thread = threading.Thread(
        target=worker,
        name=f"feature_guest_{guest_index}",
        daemon=True,
    )
    job.thread.start()
    return job


def _collect_feature_result(job: FeatureExtractionJob, timeout_s: float) -> dict[str, Any]:
    """在进入 introduce 前按短超时收集后台外貌提取结果。"""
    if not job.done_event.wait(timeout_s):
        return {}

    if job.error:
        return {}

    return job.result


def _wait_for_latest_frame(camera, timeout_s: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    """等待相机缓存中出现一帧可用图像。"""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        color_frame, depth_frame = _split_frames(camera.get_frames())
        if color_frame is not None:
            return color_frame, depth_frame
        time.sleep(0.05)
    return None, None


def _split_frames(frames) -> tuple[np.ndarray | None, np.ndarray | None]:
    """兼容 get_frames 返回 frame 或 (color, depth)。"""
    if frames is None:
        return None, None
    if isinstance(frames, tuple):
        if len(frames) >= 2:
            return frames[0], frames[1]
        if len(frames) == 1:
            return frames[0], None
    return frames, None


def _detect_person_boxes(
    yolo_model,
    frame: np.ndarray,
    conf: float,
    person_class_id: int,
) -> list[tuple[int, int, int, int]]:
    """从 YOLO 结果中提取 person 框。"""
    results = yolo_model(frame, conf=conf, verbose=False)
    if not results:
        return []

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes: list[tuple[int, int, int, int]] = []
    for box, cls_id, score in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        if int(cls_id.item()) != person_class_id:
            continue
        if float(score.item()) < conf:
            continue

        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))

    return boxes


def _select_closest_person(
    bboxes: list[tuple[int, int, int, int]],
    depth_frame: np.ndarray | None,
) -> tuple[int, int, int, int]:
    """优先用深度图选最近的人；没有深度图时退化为面积最大的框。"""
    if depth_frame is None:
        return max(bboxes, key=_bbox_area)

    best_bbox: tuple[int, int, int, int] | None = None
    best_depth = float("inf")

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = x2 - x1

        rx1 = max(0, int(x1 + 0.2 * w))
        ry1 = max(0, int(y1 + 0.2 * h))
        rx2 = min(depth_frame.shape[1], int(x2 - 0.2 * w))
        ry2 = min(depth_frame.shape[0], int(y2 - 0.2 * h))
        if rx2 <= rx1 or ry2 <= ry1:
            continue

        roi = depth_frame[ry1:ry2, rx1:rx2]
        valid = roi[np.isfinite(roi) & (roi > 0)]
        if valid.size == 0:
            continue

        median_depth = float(np.median(valid))
        if median_depth < best_depth:
            best_depth = median_depth
            best_bbox = bbox

    return best_bbox if best_bbox is not None else max(bboxes, key=_bbox_area)


def _crop_person_with_margin(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """在人物框周围保留少量边缘，提升外貌描述信息完整性。"""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    pad_x = int((x2 - x1) * 0.08)
    pad_y = int((y2 - y1) * 0.08)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _score_person_crop(bbox: tuple[int, int, int, int], frame_w: int, frame_h: int) -> float:
    """综合框面积和居中程度给候选 crop 打分。"""
    x1, y1, x2, y2 = bbox
    area_ratio = _bbox_area(bbox) / max(1, frame_w * frame_h)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    center_penalty = abs(center_x - frame_w / 2.0) / frame_w + abs(center_y - frame_h / 2.0) / frame_h
    return area_ratio * 100.0 - center_penalty * 10.0


def _is_valid_bbox(bbox: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    return x2 > x1 and y2 > y1


def _person_overlaps_seat(
    person_bbox: tuple[int, int, int, int],
    seat_bbox: tuple[int, int, int, int],
) -> bool:
    """用重叠和下半身位置做简单座位占用判断。"""
    px1, py1, px2, py2 = person_bbox
    sx1, sy1, sx2, sy2 = seat_bbox

    inter_x1 = max(px1, sx1)
    inter_y1 = max(py1, sy1)
    inter_x2 = min(px2, sx2)
    inter_y2 = min(py2, sy2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return False

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    seat_area = max(1, (sx2 - sx1) * (sy2 - sy1))
    overlap_ratio = inter_area / seat_area
    center_x = (px1 + px2) / 2.0
    bottom_y = py2
    legs_in_seat = sx1 <= center_x <= sx2 and sy1 <= bottom_y <= sy2 + 0.15 * (sy2 - sy1)
    return overlap_ratio >= 0.3


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)

