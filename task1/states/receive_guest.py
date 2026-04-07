"""
状态1：等待铃声 → 导航到门 → 开门迎接 → 问好 → 询问姓名饮品
     → 提取人脸/视觉特征 → 引导就坐

本状态内部循环执行两次（两位客人），全部就座后进入状态2。

评分项:
  - 门铃声识别为客人到达信号 (2×30 奖励)
  - 为客人开门 (2×200 奖励)
  - 朝导航方向观察 (2×15)
  - 注视正在交谈的对象 (2×50)
  - 为客人提供空闲座位 (2×100)
  - 向第二位客人描述第一位客人特征 (4×20 奖励)
  - 不提非必要问题确认信息 (4×15 奖励)
"""

import time
import logging

from ultralytics import YOLO

from common.state_machine import State
from common.skills.agv_api import agv
from common.skills.camera import camera_manager
from common.config import CAMERA_HEAD, CAMERA_CHEST, CAMERA_LEFT, CAMERA_RIGHT
from common.skills.head_control.head_control import HeadCameraController
from common.skills.audio_module.voice_assiant import (
    VoiceAssistant, DoorbellDetector, extract_name,
)
from task1.behaviors.vision import RoboCupReIDTracker
from task1 import config

log = logging.getLogger("task1.receive_guest")

# ── 导航辅助 ─────────────────────────────────────────────────────────────

def _wait_nav(timeout: float = 30.0) -> bool:
    """轮询等待导航完成"""
    start = time.time()
    while time.time() - start < timeout:
        status = agv.get_task_status()
        if status is None:
            time.sleep(0.5)
            continue
        ts = status.get("task_status", "")
        if ts in ("completed", "none", ""):
            return True
        if ts == "failed":
            log.warning("导航失败")
            return False
        time.sleep(0.5)
    log.warning("导航超时")
    return False


# ── 语音问答辅助 ─────────────────────────────────────────────────────────

def _ask(va: VoiceAssistant, prompt: str, retries: int = 3) -> str:
    """speak → record → recognize，失败自动重试"""
    for i in range(retries):
        va.speak(prompt)
        frames = va.record_utterance()
        text = va.recognize_speech(frames)
        if text:
            return text
        if i < retries - 1:
            va.speak("Sorry, I didn't catch that. Could you say it again?")
    return ""


# ── 视觉检测辅助 ─────────────────────────────────────────────────────────

def _detect_and_bind(yolo, tracker: RoboCupReIDTracker, cam, guest_name: str):
    """
    从相机取帧 → YOLO 检测 → tracker 更新 → 绑定最近的未分配人物
    返回 (person_id, frame)，失败返回 (-1, None)
    """
    for _ in range(5):  # 多帧尝试，等人脸被检测到
        color, depth = cam.get_frames()
        if color is None:
            time.sleep(0.1)
            continue

        results = yolo(color, conf=0.5, classes=0, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({"bbox": (x1, y1, x2, y2)})

        if not detections:
            time.sleep(0.2)
            continue

        matched = tracker.update(detections, color)
        pid = tracker.get_closest_unassigned_person(matched)
        if pid is not None:
            tracker.assign_guest_name(pid, guest_name, color)
            log.info("客人 %s 绑定 → person_id=%d", guest_name, pid)
            return pid, color

        time.sleep(0.2)

    log.warning("视觉绑定失败: %s", guest_name)
    return -1, None


# ═════════════════════════════════════════════════════════════════════════
#  状态类
# ═════════════════════════════════════════════════════════════════════════

class ReceiveGuest(State):

    def __init__(self):
        self.va: VoiceAssistant | None = None
        self.doorbell: DoorbellDetector | None = None
        self.tracker: RoboCupReIDTracker | None = None
        self.yolo = None
        self.head: HeadCameraController | None = None

    # ── 初始化 ────────────────────────────────────────────────────────

    def on_enter(self, ctx):
        log.info("初始化语音 / 视觉 / 云台模块...")

        # 语音
        self.va = VoiceAssistant(use_rnnoise=True)
        self.va.start_stream()
        self.va.calibrate_noise(duration_ms=800)

        # 门铃
        self.doorbell = DoorbellDetector(threshold=0.5)

        # 视觉
        self.yolo = YOLO("yolov8n.pt")
        self.tracker = RoboCupReIDTracker(debug=False)

        # 云台
        self.head = HeadCameraController()

    # ── 主流程 ────────────────────────────────────────────────────────

    def execute(self, ctx) -> str:
        cam = camera_manager.get(CAMERA_HEAD)

        while ctx.current_guest_index < 2:
            i = ctx.current_guest_index
            guest = ctx.current_guest
            log.info("========== 接待客人 #%d ==========", i)

            # ── 1. 等待门铃（奖励 +30） ──────────────────────────────
            self.doorbell.start()
            detected = self.doorbell.wait_for_doorbell(timeout=60)
            self.doorbell.stop()
            if detected:
                log.info("门铃响了，出发迎接")
            else:
                log.info("未检测到门铃，仍然出发")

            # ── 2. 导航到门口（云台朝前 +15） ────────────────────────
            self.head.home()
            agv.navigate_to("", config.STATION_DOOR)
            _wait_nav(timeout=config.NAV_TIMEOUT)

            # ── 3. 问好 + 询问姓名（+15 不提非必要问题） ────────────
            self.va.speak("Hello! Welcome. I am your host robot.")

            raw_name = _ask(self.va, "May I have your name please?")
            guest.name = extract_name(raw_name) if raw_name else ""
            log.info("客人姓名: %s (原始: %s)", guest.name, raw_name)

            # ── 4. 询问饮品 ──────────────────────────────────────────
            guest.favorite_drink = _ask(
                self.va,
                "What is your favorite drink?"
            )
            log.info("喜爱饮品: %s", guest.favorite_drink)

            # ── 5. 视觉绑定（人脸 + LLM 特征提取） ──────────────────
            pid, frame = _detect_and_bind(
                self.yolo, self.tracker, cam,
                guest.name if guest.name else f"guest_{i}",
            )
            guest.person_id = pid
            if pid >= 0:
                info = self.tracker.get_person_info(pid)
                if info:
                    guest.visual_features = info.get("features", {})

            # ── 6. 第二位客人时描述第一位外观（奖励 4×20） ───────────
            if i == 1 and ctx.guests[0].name:
                desc = self.tracker.describe_guest(ctx.guests[0].name)
                if desc and desc != "未知":
                    self.va.speak(
                        f"By the way, the other guest {ctx.guests[0].name} "
                        f"looks like: {desc}."
                    )

            # ── 7. 导航到空座位（+100 提供空闲座位） ─────────────────
            seat_id = ctx.find_free_seat()
            if seat_id:
                self.va.speak("Please follow me, I will show you to your seat.")
                self.head.home()  # 导航时朝前看
                agv.navigate_to("", seat_id)
                _wait_nav(timeout=config.NAV_TIMEOUT)
                self.va.speak("Please have a seat here.")

                guest.seat_id = seat_id
                ctx.occupy_seat(seat_id)
            else:
                self.va.speak("I'm sorry, there are no free seats available.")

            # ── 8. 更新索引 ──────────────────────────────────────────
            ctx.current_guest_index += 1

        return "introduce"

    # ── 清理 ──────────────────────────────────────────────────────────

    def on_exit(self, ctx):
        if self.doorbell:
            self.doorbell.stop()
        if self.va:
            self.va.close()
        log.info("receive_guest 资源已释放")
