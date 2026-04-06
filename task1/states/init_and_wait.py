"""
状态0：初始化硬件软件，等待按下 Enter 开始任务
"""
import time


from common.state_machine import State
from Robotic_Arm.rm_robot_interface import *
from common.skills.slide_control import slide_control, send_axis



class InitAndWait(State):

    def execute(self, ctx) -> str:
        # TODO: 初始化硬件（云台回中、机械臂归位、相机启动等）
        # 检查是否开启
        if not slide_control.serial.is_open:
                slide_control.serial.open()

        # 检查使能
        #slide_control.switch_to_position_mode()
        sw = slide_control.read_status_word()
        if sw is not None and (sw & 0x6F) == 0x27:
            print("电机已使能，开始运动...")
            send_axis(0000000)
        else:
            print("使能失败，尝试清除故障...")
            slide_control.clear_fault()
            print("重新设为位置模式")
            slide_control.switch_to_position_mode()
            # 发送速度
            slide_control.device_speed_set(200)
            print("已恢复，请重新运行脚本")

        input("[状态0] 硬件就绪，按 Enter 开始比赛...")
        return "receive_guest"
