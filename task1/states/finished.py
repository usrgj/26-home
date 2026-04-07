"""
状态5：比赛结束
"""
from Robotic_Arm.rm_robot_interface import *

from common.state_machine import State
from common.skills.slide_control import slide_control
from common.skills.agv_api import agv
from common.skills.arm import left_arm, right_arm, Gripper, IOGripper, GripperError, IOGripperError
from common.skills.head_control import pan_tilt
from common.skills.camera import camera_manager as cams
from common.config import CAMERA_HEAD, CAMERA_CHEST, CAMERA_LEFT, CAMERA_RIGHT
from task1.config import LEFT_HOME_JOINTS, RIGHT_HOME_JOINTS, ARM_SPEED


class Finished(State):

    def execute(self, ctx) -> str:
        """
        执行系统复位与资源清理操作。
        """
        #关闭底盘通讯
        agv.stop()
        # 关闭相机连接
        cams.stop_all()

        # 控制左右机械臂回归 home 位并断开连接
        left_arm.rm_movej(LEFT_HOME_JOINTS, v=ARM_SPEED, r=0, connect=0, block=1)
        right_arm.rm_movej(RIGHT_HOME_JOINTS, v=ARM_SPEED, r=0, connect=0, block=1)

        left_arm.rm_delete_robot_arm()
        right_arm.rm_delete_robot_arm()

        # 云台复位
        pan_tilt.home()

        # 导轨回中
        slide_control.send_axis(0000000)
        return "finished"
