"""
状态3：向第二位客人拿包

评分项:
  - 通过手递手方式从客人处接过包 (+50)

与状态4独立：即使拿包失败，也可以拿下跟随的分数。
"""

from common.state_machine import State
from common.skills.agv_api import agv, wait_nav
from common.skills.arm import left_arm, right_arm, left_gripper
from common.skills.audio_module.voice_assiant import voice_assistant
from task1 import config
import time
import math

class ReceiveBag(State):

    def execute(self, ctx) -> str:
        
        # ── 1. 导航到第二位客人 ────────────────────────
        # ── 2. 请求递包 ────────────────────────
        # ── 3. 接包 ────────────────────────


        # 1. 面向第二位客人
        # TODO: 导航/转向到 guest_b 的座位附近
        target_angle = config.INTRO_LOOK_ANGLES_DEG.get(agv.get_current_station(), {}).get(ctx.guests[1].seat_id)
        pose = agv.get_pose()
        if pose is None:
          print("无法获取机器人位姿，跳过底盘转向")
        try:
            x = float(pose["x"])
            y = float(pose["y"])
            agv.free_navigate_to(x, y, math.radians(target_angle))
        except (KeyError, TypeError, ValueError) as exc:
            print("机器人位姿不完整，跳过底盘转向: %s", exc)

        # 2. 请求递包
        voice_assistant.speak("请把包递给我，我来帮您拿。")

        # 3. 机械臂到接包位置，夹爪张开
        print(left_arm.rm_movej([-132.172,-69.632,-27.875,96.759,-5.229,-244.059], 20, 0, 0, 1))
        left_gripper.open()
        time.sleep(7)

        left_gripper.grab(force=500)
        print(left_arm.rm_movej([-112.305,-115.771,-67.401,2.997,7.318,-263.746], 20, 0, 0, 1))
        time.sleep(2)


        ctx.bag_received = True

        
        return "follow_and_place"