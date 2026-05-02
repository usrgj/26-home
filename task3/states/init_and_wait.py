"""初始化状态：准备硬件并等待比赛开始"""

from __future__ import annotations
import logging

from common.state_machine import State
from task3 import config

log = logging.getLogger("task3.init_and_wait")


class InitAndWait(State):

    def execute(self, ctx) -> str:
        """初始化底盘和机械臂，并在裁判开始前阻塞等待。"""
        self._start_agv()
        self._prepare_arm()

        input("[状态0] Task3 硬件就绪，按 Enter 开始从洗衣机取衣...")
        return "nav_to_washer"

    def _start_agv(self):
        """启动底盘通信，失败时交给状态机异常恢复。"""
        from common.skills.agv_api import agv

        ok = agv.start()
        if not ok:
            raise RuntimeError("AGV 启动失败")

    def _prepare_arm(self):
        """打开夹爪，并在配置了安全关节角时让机械臂回安全位。"""
        try:
            from common.skills.arm import left_arm, left_gripper
        except Exception as exc:
            log.warning("机械臂模块不可用，跳过机械臂初始化: %s", exc)
            return

        try:
            left_gripper.open()
        except Exception as exc:
            log.warning("夹爪打开失败: %s", exc)

        home_joints = getattr(config, "ARM_HOME_JOINTS", None)
        if home_joints:
            left_arm.rm_movej(
                home_joints,
                v=config.ARM_SPEED,
                r=0,
                connect=0,
                block=1,
            )
        else:
            log.info("未配置 ARM_HOME_JOINTS，跳过机械臂回安全位")
