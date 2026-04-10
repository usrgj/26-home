'''
底盘通讯的使用demo
'''
from __future__ import annotations
import time
from agv_api import agv, wait_nav

# ── Demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":


    agv.start()

    # agv.send_velocity(0.3, 0.0, duration=0) #开环运动
    # time.sleep(5)

    agv.navigate_to("", "LM1")
    wait_nav()

    agv.stop()