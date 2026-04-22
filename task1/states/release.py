"""
状态5：释放硬件资源
"""
import time
from Robotic_Arm.rm_robot_interface import *

from common.state_machine import State
from common.skills.slide_control import slide_control
from common.skills.agv_api import agv
# from common.skills.arm import left_arm, right_arm, Gripper, IOGripper, GripperError, IOGripperError
from common.skills.arm import left_arm, left_gripper
from common.skills.head_control import pan_tilt
from common.skills.camera import camera_manager as cams
from task1.behaviors.show import viewer
from task1.config import LEFT_HOME_JOINTS, ARM_SPEED


class Release(State):

    def execute(self, ctx) -> str:
        """
        执行系统复位与资源清理操作。
        """
        try:
            agv.stop()
        except Exception:
            pass
        
        try:
            viewer.stop()
        except Exception:
            pass

        try:
            cams.stop_all()
        except Exception:
            pass

        try:
            left_gripper.open()
            left_arm.rm_movej(LEFT_HOME_JOINTS, v=ARM_SPEED, r=0, connect=0, block=1)
            # right_arm.rm_movej(RIGHT_HOME_JOINTS, v=ARM_SPEED, r=0, connect=0, block=1)
            time.sleep(3)
        except Exception:
            pass

        try:
            left_arm.rm_delete_robot_arm()
        except Exception:
            pass

        # try:
        #     right_arm.rm_delete_robot_arm()
        # except Exception:
        #     pass

        try:
            pan_tilt.home()
        except Exception:
            pass

        try:
            slide_control.device_speed_set(200)
            slide_control.send_axis(0)
        except Exception:
            pass

        return "finished"
