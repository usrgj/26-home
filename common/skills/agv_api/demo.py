'''
底盘通讯的使用demo
'''
from __future__ import annotations
import os
import sys
import math

# 获取当前脚本的绝对路径 (26-home/task3/arm_folding/tool/slide_locate.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 向上跳三级回到真正的根目录 (26-home)
# tool -> arm_folding -> task3 -> 26-home
root_dir = os.path.abspath(os.path.join(current_dir, "../../.."))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
    print(f"已修正项目根目录为: {root_dir}")
import time
from common.skills.agv_api import agv, wait_nav

# ── Demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":


    agv.start()

    # agv.send_velocity(0.3, 0.0, duration=0) #开环运动
    # time.sleep(5)

    print(agv.navigate_to("", "LM3", math.radians(18.850)))
    time.sleep(3)
    wait_nav()

    agv.stop()
