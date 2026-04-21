"""
状态3：向第二位客人拿包

评分项:
  - 通过手递手方式从客人处接过包 (+50)

与状态4独立：即使拿包失败，也可以拿下跟随的分数。
"""

import sys
from pathlib import Path

# 将项目根目录添加到 sys.path，以便导入 common.config
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 向上2级到 26-home
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import LANGUAGE
from common.state_machine import State
from common.skills.agv_api import agv, wait_nav
# from common.skills.arm import left_arm, right_arm, left_gripper
from common.skills.arm import left_arm,  left_gripper
from common.skills.audio_module.voice_assiant import voice_assistant
from task1 import config
import time
import math

class ReceiveBag(State):

    def execute(self, ctx) -> str:
        
        # 根据语言选择提示语
        if LANGUAGE == "en":
            speak_text = "Please hand me the bag, I will take it for you."
        else:
            speak_text = "请把包递给我，我来帮你拿。"

        # 1. 面向第二位客人
        target_degree = config.INTRO_LOOK_ANGLES_DEG.get(agv.get_current_station(), {}).get(ctx.guests[1].seat_id)
        angle = math.radians(target_degree)
        
        agv.navigate_to(agv.get_current_station(), agv.get_current_station(), angle=angle)


        # 2. 请求递包
        voice_assistant.speak(speak_text)

        # 3. 机械臂到接包位置，夹爪张开
        print(left_arm.rm_movej([-83.018,-42.675,-73.002,-16.54,-29.304,52.168], 20, 0, 0, 1))
        left_gripper.open()
        time.sleep(7)

        left_gripper.grab(force=500)
        print(left_arm.rm_movej([-112.305,-115.771,-67.401,2.997,7.318,-263.746], 20, 0, 0, 1))
        time.sleep(2)

        ctx.bag_received = True

        return "follow_and_place"
