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
        
      # 1. 面向第二位客人
      _face_guest(agv.get_current_station(), getattr(ctx.guests[1], "seat_id", ""))

      # 3. 机械臂到接包位置，夹爪张开
      print(left_arm.rm_movej([-38.651, -98.036, -36.142, -42.949, -51.924, 38.235], 20, 0, 0, 1))
      left_gripper.set_route(0, 500)
      left_gripper.open()
      
      # 请求递包
      voice_assistant.speak("Please pass me your bag, and I'll help you with it.")
      time.sleep(1.5)

      voice_assistant.speak("I'll grab your bag. Please be careful.")
      left_gripper.grab(force=500)
      print(left_arm.rm_movej([-23.77,-93.721,-52.2,-19.763,-47.238,19.638], 40, 0, 0, 0))
      left_gripper.set_route(0, 1000)
      # time.sleep(2)

      # ctx.bag_received = True

      return "follow_and_place"

def _face_guest(view_station_id: str, target_seat_id: str) -> None:

    if not view_station_id or not target_seat_id:
        return

    angle_deg = config.INTRO_LOOK_ANGLES_DEG.get(view_station_id, {}).get(target_seat_id) or None
    if angle_deg is None:
        print(f"未配置介绍角度: view_station={view_station_id}, target_seat={target_seat_id}")
        return
    theta = math.radians(angle_deg)
    ok = agv.navigate_to(agv.get_current_station(), view_station_id, angle=theta)
    if not ok:
        print(f"原地转向指令发送失败: view_station={view_station_id} target_seat={target_seat_id}")
        return

    wait_nav(timeout=config.NAV_TIMEOUT)
