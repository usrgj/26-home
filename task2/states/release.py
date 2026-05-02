"""释放 task2 使用的导航与视觉资源。"""

from __future__ import annotations

import logging

from common.state_machine import State

log = logging.getLogger("task2.release")


class Release(State):
    """停止底盘和相机，不处理任何抓取硬件。"""

    def execute(self, ctx) -> str:
        """释放资源后进入 finished。"""
        self._stop_agv()
        self._stop_cameras()
        return "finished"

    def _stop_agv(self) -> None:
        """停止底盘通信。"""
        try:
            from common.skills.agv_api import agv

            agv.stop()
        except Exception as exc:
            log.warning("底盘释放失败: %s", exc)

    def _stop_cameras(self) -> None:
        """停止可能已经启动的相机流。"""
        try:
            from common.skills.camera import camera_manager as cams

            cams.stop_all()
        except Exception as exc:
            log.warning("相机释放失败: %s", exc)
