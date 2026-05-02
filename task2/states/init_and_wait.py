"""初始化状态：准备底盘、相机和语音播报，等待比赛开始。"""

from __future__ import annotations

import logging

from common.state_machine import State
from task2.states.utils import safe_speak

log = logging.getLogger("task2.init_and_wait")


class InitAndWait(State):
    """启动 task2 需要的非抓取能力。"""

    def execute(self, ctx) -> str:
        """初始化导航和视觉模块，并等待裁判开始信号。"""
        self._start_agv()
        self._start_cameras()
        self._prepare_detector(ctx)

        safe_speak("Ready.")
        input("[状态0] Task2 导航、视觉和语音就绪，按 Enter 开始比赛...")
        return "kitchen_task"

    def _start_agv(self) -> None:
        """启动底盘通信，失败时交给状态机异常恢复。"""
        from common.skills.agv_api import agv

        ok = agv.start()
        if not ok:
            raise RuntimeError("AGV 启动失败")

    def _start_cameras(self) -> None:
        """异步预热头部和胸部相机，不阻塞比赛流程。"""
        try:
            from common.config import CAMERA_CHEST, CAMERA_HEAD
            from common.skills.camera import camera_manager as cams

            cams.start_async(CAMERA_HEAD)
            cams.start_async(CAMERA_CHEST)
        except Exception as exc:
            log.warning("相机预热失败，后续识别会尝试懒启动: %s", exc)

    def _prepare_detector(self, ctx) -> None:
        """创建视觉适配器实例；模型和相机仍由适配器在检测时懒加载。"""
        from task2.behaviors.kitchen_detector import KitchenDetector

        ctx.detector = KitchenDetector()
