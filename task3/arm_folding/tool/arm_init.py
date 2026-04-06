'''
让机械臂达到一个折衣服的初始位置
'''
from Robotic_Arm.rm_robot_interface import *

LEFT_ARM_IP = "192.168.192.18"
RIGHT_ARM_IP = "192.168.192.19"
ARM_PORT = 8080

LEFT_TARGET_JOINTS = [-121.994, -27.868, -56.127, 47.455, -77.515, 44.013]  # 目标关节角度（度）
RIGHT_TARGET_JOINTS = [141.882, -54.571, -12.117, -62.232, -113.411, -27.554]  # 目标关节角度（度）

SPEED         = 20                                  # 运动速度 1~100

left_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
left_arm.rm_create_robot_arm(LEFT_ARM_IP, ARM_PORT)
right_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
right_arm.rm_create_robot_arm(RIGHT_ARM_IP, ARM_PORT)

code = left_arm.rm_movej(LEFT_TARGET_JOINTS, v=SPEED, r=0, connect=0, block=1)
code = right_arm.rm_movej(RIGHT_TARGET_JOINTS, v=SPEED, r=0, connect=0, block=1)
print("完成" if code == 0 else f"失败，错误码: {code}")

left_arm.rm_delete_robot_arm()
right_arm.rm_delete_robot_arm()