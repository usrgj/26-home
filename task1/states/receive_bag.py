"""
状态3：向第二位客人拿包

评分项:
  - 通过手递手方式从客人处接过包 (+50)

与状态4独立：即使拿包失败，也可以拿下跟随的分数。
"""

from common.state_machine import State
from common.skills.arm import left_arm, right_arm, left_gripper
import time

class ReceiveBag(State):

    def execute(self, ctx) -> str:
        
        # ── 1. 导航到第二位客人 ────────────────────────
        # ── 2. 请求递包 ────────────────────────
        # ── 3. 接包 ────────────────────────


        # 1. 面向第二位客人
        # TODO: 导航/转向到 guest_b 的座位附近

        # 2. 请求递包
        # speech.say(f"{ctx.guests[1].name}, could you please hand me your bag?")

        # 3. 机械臂到接包位置，夹爪张开
        print(left_arm.rm_movej([-132.172,-69.632,-27.875,96.759,-5.229,-244.059], 20, 0, 0, 1))
        left_gripper.open()
        time.sleep(7)

        left_gripper.grab(force=500)
        print(left_arm.rm_movej([-112.305,-115.771,-67.401,2.997,7.318,-263.746], 20, 0, 0, 1))
        time.sleep(2)
        # arm.go_receive_bag()
        # arm.release()

        # 4. 等待包被放入（力反馈 / 视觉确认）
        # TODO: 检测是否成功接到包

        # 5. 夹爪闭合
        # arm.grip()

        ctx.bag_received = True

        # speech.say("Thank you. I'll take this to the host.")
        return "follow_and_place"