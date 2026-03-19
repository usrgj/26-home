from __future__ import annotations
from follow.phase1_lidar_follow import main
from agv_api import AGVManager, agv_manager
import time
# from camera.config import CAMERA_CHEST
# from camera import camera_manager

PORT_STATUS = 19204
PORT_CONTROL = 19205
PORT_NAVIGATION = 19206
PORT_CONFIGUE = 19207
PORT_OTHER = 19210
PROT_PUSH = 19301

if __name__ == "__main__":
    agv_manager.start()
    try:
        main()
    finally:
        agv_manager.stop()