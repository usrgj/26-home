from common.skills.gripper.gripper_servo import Gripper
from Robotic_Arm.rm_robot_interface import *

arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = arm.rm_create_robot_arm("192.168.192.18", 8080)

gripper = Gripper(arm)
gripper.set_route(min_pos=1, max_pos=500)