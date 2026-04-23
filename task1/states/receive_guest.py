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

from common.config import (CAMERA_HEAD,CAMERA_CHEST)
from common.skills.agv_api import agv, wait_nav
from common.skills.arm import left_arm, left_gripper
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
from task1.behaviors.vision import (GazeAPI,SeatManager)

_DEFAULT_DETECTION_CONF = 0.35
_GUEST_CROP_WINDOW_S = 2.0
_GUEST_CROP_MAX_SAMPLES = 5
_FEATURE_WAIT_TIMEOUT_S = 2.0


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
        self._cam_chest = camera_manager.get(CAMERA_CHEST)
        self._feature_jobs: dict[int, FeatureExtractionJob] = {}
        seat_coords = [seat["box1"] for seat in config.SEATS if any(seat["box1"])]
        self.seat_manager = SeatManager(seat_coords, min_empty=2)

        
    def execute(self, ctx) -> str:
        from common.utils.drag_and_play.dragTeach_play import play_robot_trajectory
        self._model = self._get_model()
        self.gaze_api = GazeAPI(self._model)
        self._feature_jobs = {}
        voice_assistant.set_recording(1)

        while ctx.current_guest_index < len(ctx.guests):
            guest_index = ctx.current_guest_index
            guest = ctx.current_guest
        
            pan_tilt.home()
            agv.navigate_to(agv.get_current_station(), config.STATION_START)
            wait_nav(timeout=config.NAV_TIMEOUT)
            #==== 空座位识别能力接口调用 START ==== author:xxy
            for _ in range(5):
                color_frame, _ = self._cam_chest.get_frames()
                if color_frame is None:
                    continue
                person_boxes = self.gaze_api.detect_persons(color_frame)
                self.seat_manager.update_from_detections(person_boxes)
                 # 画座位框
                frame = color_frame.copy() 
                for idx, seat in enumerate(self.seat_manager.seat_coords):
                    color = (0,255,0) if self.seat_manager.seat_status[idx] == "empty" else (0,0,255)
                    cv2.rectangle(frame, (seat[0], seat[1]), (seat[2], seat[3]), color, 2)
                    cv2.putText(frame, f"{idx+1}:{self.seat_manager.seat_status[idx]}", (seat[0], seat[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                # 画人体框
                for box in person_boxes:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255,255,0), 2)
                cv2.imshow('Seat Status', frame)
                cv2.waitKey(1)
                time.sleep(0.08)
            seat_status = self.seat_manager.seat_status
            print(f"当前座位状态: {seat_status}")
            empty_indices = [i for i, s in enumerate(seat_status) if s == "empty"]
            print(f"当前空座位编号: {empty_indices}")
            # 分配空座位
            if empty_indices:
                seat_idx = empty_indices[0]
                seat_id = config.SEATS[seat_idx]["id"]   
            else:
                seat_id = None
    
            # 等待门铃
            doorbell.start()
            is_detected = doorbell.wait_for_doorbell(timeout=30)
            doorbell.stop()
            print("检测到门铃" if is_detected else "等待门铃超时")
            
            # 导航到门口
            agv.navigate_to(config.STATION_START, config.STATION_DOOR)
            wait_nav(timeout=config.NAV_TIMEOUT)

            # TODO 开门
            left_gripper.open()
            success = play_robot_trajectory(trajectory_file=config.TRAJECTORY_GET_PATH, arm=left_arm)
            left_gripper.grab(force=700, block=True, timeout=5)
            success = play_robot_trajectory(trajectory_file=config.TRAJECTORY_MOVE_PATH, arm=left_arm)
            left_gripper.open(block=True)
            success = play_robot_trajectory(trajectory_file=config.TRAJECTORY_LEAVE_PATH, arm=left_arm)
            
            
            
            
            # 注视 author:xxy
            gaze_thread, gaze_stop_event = self.gaze_api.start_gaze_tracking_nearest_person(pan_tilt, self._cam_head, duration=45)
            # 询问姓名和喜爱饮品，同时在后台提取视觉特征
            try:
                text = quest_and_answer("Welcome! May I know your name ?")
                guest.name = extract_name(text)
                print(f"the guest's name is {guest.name}")

                if guest_index == 0:
                    crop = _select_best_guest_crop(self._model, self._cam_head)
                    if crop is not None:
                        self._feature_jobs[guest_index] = _start_feature_extraction(
                            guest_index=guest_index,
                            crop=crop,
                        )

                text = quest_and_answer("What is your favorite drink ?")
                guest.favorite_drink = extract_drink(text)
                print(f"the guest's favorite drink is {guest.favorite_drink}")
                
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
                # 再采集多帧
                for _ in range(3):
                    color_frame, _ = self._cam_head.get_frames()
                    if color_frame is None:
                        continue
                    person_boxes = self.gaze_api.detect_persons(color_frame)
                    self.seat_manager.update_from_detections(person_boxes)
                    time.sleep(0.08)
                seat_status = self.seat_manager.seat_status
                empty_indices = [i for i, s in enumerate(seat_status) if s == "empty"]
                if empty_indices:
                    seat_idx = empty_indices[0]
                    seat_id = config.SEATS[seat_idx]["id"]
                else:
                    seat_id = None
            if seat_id is None:
                # 默认导航到某个备选点/接待区
                nav_id, angle = _get_seat_navigation_target("seat_default")
                agv.navigate_to(agv.get_current_station() or "", nav_id, angle)
                wait_nav(timeout=config.NAV_TIMEOUT)
                voice_assistant.speak("Sorry, no free seat detected, please wait for staff assistance.")
                        
            if seat_id is not None:
                nav_id, angle = _get_seat_navigation_target(seat_id)
                voice_assistant.speak("Please follow me, I will show you to your seat.")
                pan_tilt.home()
                agv.navigate_to(agv.get_current_station() or "", nav_id, angle)
                wait_nav(timeout=config.NAV_TIMEOUT)
                
                left_arm.rm_movej([-44.246,-59.463,-58.874,20.883,0.296,12.781], 20, 0, 0, 0)
                
                voice_assistant.speak("Please have a seat here.")
                guest.seat_id = seat_id
                ctx.occupy_seat(seat_id)
            else:
                voice_assistant.speak("I'm sorry, there are no free seats available.")

            left_arm.rm_movej(config.LEFT_HOME_JOINTS, 20, 0, 0, 0)
            

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

def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)
