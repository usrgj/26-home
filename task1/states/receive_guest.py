"""
状态1：等待铃声 → 导航到门 → 问好 → 询问姓名饮品
     → 后台提取视觉特征 → 引导就坐

本状态内部循环执行两次（两位客人），全部就座后进入状态2。
"""

from __future__ import annotations

import logging
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

log = logging.getLogger("task1.receive_guest")

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
        self._feature_jobs = {}

        voice_assistant.set_recording(1)

        while ctx.current_guest_index < len(ctx.guests):

            agv.navigate_to(agv.get_current_station(), config.STATION_START)
            wait_nav(timeout=config.NAV_TIMEOUT)

            pan_tilt.home()
            # 在这个位置进行第一次观察，座位状态更新，
            _observe_visible_seats(ctx, model, self._cam_head, box_key="box1")

            guest_index = ctx.current_guest_index
            guest = ctx.current_guest
            log.info("========== 接待客人 #%d ==========", guest_index)

            # 等待门铃
            doorbell.start()
            detected = doorbell.wait_for_doorbell(timeout=60)
            doorbell.stop()
            log.info("门铃检测结果: %s", "detected" if detected else "timeout")
            
            # 导航到门口
            pan_tilt.home()
            agv.navigate_to(config.STATION_START, config.STATION_DOOR)
            wait_nav(timeout=config.NAV_TIMEOUT)

            # TODO 开门
            # 注视
            gaze_stop_event = threading.Event()
            gaze_thread = threading.Thread(
                target=_gaze_loop,
                args=(gaze_stop_event, model, self._cam_head, pan_tilt),
                name=f"gaze_guest_{guest_index}",
                daemon=True,
            )
            gaze_thread.start()

            # 询问姓名和喜爱饮品，同时在后台提取视觉特征
            try:
                guest.name = _ask_guest_name()

                crop = _select_best_guest_crop(model, self._cam_head)
                if crop is not None:
                    self._feature_jobs[guest_index] = _start_feature_extraction(
                        guest_index=guest_index,
                        crop=crop,
                    )

                guest.favorite_drink = _ask_guest_drink()
            finally:
                # 结束注视
                _stop_gaze_thread(gaze_stop_event, gaze_thread) 

            # 导航到空位置
            seat_id = ctx.find_free_seat()
            if seat_id is None:
                seat_id = _find_free_seat_after_additional_observation(
                    ctx,
                    model,
                    self._cam_head,
                )

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
                log.warning("没有找到可用座位，guest=%s", guest.name or guest_index)

            ctx.current_guest_index += 1

            left_arm.rm_movej(
                config.LEFT_HOME_JOINTS,
                v=config.ARM_SPEED,
                r=0,
                connect=0,
                block=0,
            )
            right_arm.rm_movej(
                config.RIGHT_HOME_JOINTS,
                v=config.ARM_SPEED,
                r=0,
                connect=0,
                block=0,
            )

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


def _ask_guest_name() -> str:
    """通过语音询问并提取客人姓名。"""
    prompt = "你好，欢迎来到我家。请问你叫什么?"

    for attempt in range(config.ASK_RETRIES):
        voice_assistant.speak(prompt)
        recognized_text = _record_and_recognize_text()
        if recognized_text:
            log.info("姓名语音识别结果: %s", recognized_text)
        name = extract_name(recognized_text)
        if name:
            log.info("提取到姓名: %s", name)
            return name

    log.warning("未能提取到客人姓名")
    return ""


def _ask_guest_drink() -> str:
    """通过语音询问并提取客人喜爱饮品。"""
    prompt = "你要喝点什么？"

    for attempt in range(config.ASK_RETRIES):
        voice_assistant.speak(prompt)
        recognized_text = _record_and_recognize_text()
        if recognized_text:
            log.info("饮品语音识别结果: %s", recognized_text)
        drink = _extract_favorite_drink(recognized_text)
        if drink:
            log.info("提取到饮品: %s", drink)
            return drink

    log.warning("未能提取到客人饮品")
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


def _extract_favorite_drink(text: str) -> str:
    """优先走英文饮品抽取，失败后回退到配置里的直匹配。"""
    if not text:
        return ""

    drink = extract_drink(text)
    if drink:
        return drink

    for candidate in config.COMMON_DRINKS:
        if candidate in text:
            return candidate
    return ""


def _find_free_seat_after_additional_observation(ctx, yolo_model, camera) -> str | None:
    """导航到额外观察位，补齐未知座位的可见状态。"""
    if not config.STATION_OBSERVATION:
        return None

    pan_tilt.home()
    agv.navigate_to(agv.get_current_station() or "", config.STATION_OBSERVATION)
    wait_nav(timeout=config.NAV_TIMEOUT)
    _observe_visible_seats(ctx, yolo_model, camera, box_key="box2")
    return ctx.find_free_seat()


def _observe_visible_seats(ctx, yolo_model, camera, box_key: str) -> None:
    """用当前视角里可见的座位框更新占用状态。"""
    color_frame, depth_frame = _wait_for_latest_frame(camera, timeout_s=1.5)
    if color_frame is None:
        log.warning("座位观察失败：未获取到图像")
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

    if best_crop is None:
        log.warning("未能抓取到稳定的人物裁剪图")
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
            log.exception("客人 #%d 外貌提取失败", guest_index)
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
        log.warning("客人 #%d 外貌提取超时", job.guest_index)
        return {}

    if job.error:
        log.warning("客人 #%d 外貌提取失败: %s", job.guest_index, job.error)
        return {}

    return job.result


def gaze(
    yolo_model,
    camera,
    gimbal,
    conf: float = _DEFAULT_DETECTION_CONF,
    tolerance_px: int = 50,
    person_class_id: int = 0,
) -> dict[str, Any]:
    """
    检测最近的人，将其“上 1/3 位置”对准画面中心，并把像素偏差发送给云台。
    """
    color_frame, depth_frame = _split_frames(camera.get_frames())
    if color_frame is None:
        return {
            "found": False,
            "bbox": None,
            "target_point": None,
            "raw_offset": None,
            "command_offset": None,
            "centered": False,
        }

    bboxes = _detect_person_boxes(
        yolo_model=yolo_model,
        frame=color_frame,
        conf=conf,
        person_class_id=person_class_id,
    )
    if not bboxes:
        return {
            "found": False,
            "bbox": None,
            "target_point": None,
            "raw_offset": None,
            "command_offset": None,
            "centered": False,
        }

    bbox = _select_closest_person(bboxes, depth_frame)
    x1, y1, x2, y2 = bbox

    frame_h, frame_w = color_frame.shape[:2]
    frame_cx = frame_w // 2
    frame_cy = frame_h // 2

    target_x = int((x1 + x2) / 2)
    target_y = int(y1 + (y2 - y1) / 3)

    raw_dx = target_x - frame_cx
    raw_dy = target_y - frame_cy

    cmd_dx = 0 if abs(raw_dx) <= tolerance_px else raw_dx
    cmd_dy = 0 if abs(raw_dy) <= tolerance_px else raw_dy

    if cmd_dx != 0 or cmd_dy != 0:
        _send_offset_to_gimbal(gimbal, cmd_dx, cmd_dy)

    return {
        "found": True,
        "bbox": bbox,
        "target_point": (target_x, target_y),
        "raw_offset": (raw_dx, raw_dy),
        "command_offset": (cmd_dx, cmd_dy),
        "centered": cmd_dx == 0 and cmd_dy == 0,
    }


def _gaze_loop(
    stop_event: threading.Event,
    yolo_model,
    camera,
    gimbal,
    interval_s: float = _GAZE_LOOP_INTERVAL_S,
) -> None:
    """后台循环调用 gaze，直到主线程置位 stop_event。"""
    while not stop_event.is_set():
        try:
            gaze(yolo_model, camera, gimbal)
        except Exception:
            log.exception("后台 gaze 线程异常退出")
            break
        stop_event.wait(interval_s)


def _stop_gaze_thread(stop_event: threading.Event, gaze_thread: threading.Thread | None) -> None:
    """统一停止并回收 gaze 线程。"""
    stop_event.set()
    if gaze_thread is not None:
        gaze_thread.join(timeout=1.0)


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
    return overlap_ratio >= 0.1 or legs_in_seat


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _send_offset_to_gimbal(gimbal, dx: int, dy: int) -> None:
    """把像素偏差发给云台。"""
    if callable(gimbal):
        gimbal(dx, dy)
        return

    move_relative = getattr(gimbal, "move_relative", None)
    if callable(move_relative):
        move_relative(dx * 200, dy * 200)
        return

    raise TypeError("gimbal 不可用：需要是可调用对象或实现 move_relative(horizontal, vertical)")
