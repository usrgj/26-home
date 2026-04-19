
"""
状态0：初始化与等待开始
功能：完成机器人硬件、软件模块初始化，等待任务启动信号
"""


import time
from Robotic_Arm.rm_robot_interface import *

from common.state_machine import State
from common.skills.slide_control import slide_control
from common.skills.agv_api import agv
from common.skills.arm import left_arm, right_arm, Gripper, IOGripper, GripperError, IOGripperError
from common.skills.head_control import pan_tilt
from common.skills.camera import camera_manager as cams

from common.skills.audio_module import voice_assistant, doorbell, extract_name

from common.config import CAMERA_HEAD, CAMERA_CHEST, CAMERA_LEFT, CAMERA_RIGHT
from task1.config import LEFT_HOME_JOINTS, RIGHT_HOME_JOINTS, ARM_SPEED


class InitAndWait(State):

    def execute(self, ctx) -> str:

# ═══════════════════════════════════════════════════════════════════════════
        # 导轨回中（在运行前需要检查使能并清除故障）
        slide_control.send_axis(0000000)

        # 建立底盘通讯(需要释放)
        agv.start()

        # 异步预热相机，后续状态中 get_frames() 直接读取最新缓存，不阻塞状态机
        cams.start_async(CAMERA_HEAD)
        cams.start_async(CAMERA_CHEST)
        cams.start_async(CAMERA_LEFT)
        cams.start_async(CAMERA_RIGHT)

        # 机械臂回初始位置（需要释放）
        left_arm.rm_movej(LEFT_HOME_JOINTS, v=ARM_SPEED, r=0, connect=0, block=0)
        right_arm.rm_movej(RIGHT_HOME_JOINTS, v=ARM_SPEED, r=0, connect=0, block=0)

        # 云台回中
        pan_tilt.home()


        input("[状态0] 硬件就绪，按 Enter 开始比赛...")

        return "1_navigation.py"






