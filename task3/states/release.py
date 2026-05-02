"""释放硬件资源状态"""

from __future__ import annotations
import logging

from common.state_machine import State
from task3 import config

log = logging.getLogger("task3.release")


class Release(State):

    def execute(self, ctx) -> str:
        """停止运动设备并释放可释放的硬件资源。"""
        self._stop_agv()
        self._stop_cameras()
        self._release_arm()
        return "finished"

    def _stop_agv(self):
        """停止底盘通信。"""
        try:
            from common.skills.agv_api import agv

            agv.stop()
        except Exception as exc:
            log.warning("底盘释放失败: %s", exc)

    def _stop_cameras(self):
        """停止可能已启动的相机流。"""
        try:
            from common.skills.camera import camera_manager as cams

            cams.stop_all()
        except Exception as exc:
            log.warning("相机释放失败: %s", exc)

    def _release_arm(self):
        """打开夹爪，并在配置了安全关节角时回收机械臂。"""
        try:
            from common.skills.arm import left_arm, left_gripper
        except Exception as exc:
            log.warning("机械臂模块不可用，跳过机械臂释放: %s", exc)
            return

        try:
            left_gripper.open()
        except Exception as exc:
            log.warning("夹爪释放失败: %s", exc)

        home_joints = getattr(config, "ARM_HOME_JOINTS", None)
        if home_joints:
            try:
                left_arm.rm_movej(
                    home_joints,
                    v=config.ARM_SPEED,
                    r=0,
                    connect=0,
                    block=1,
                )
            except Exception as exc:
                log.warning("机械臂回安全位失败: %s", exc)

        try:
            left_arm.rm_delete_robot_arm()
        except Exception as exc:
            log.warning("机械臂连接释放失败: %s", exc)
