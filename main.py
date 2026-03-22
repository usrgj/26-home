from __future__ import annotations
from follow.main import main
from agv_api import  agv_manager
import time
from camera.config import CAMERA_CHEST,CAMERA_HEAD
from camera import camera_manager

PORT_STATUS = 19204
PORT_CONTROL = 19205
PORT_NAVIGATION = 19206
PORT_CONFIGUE = 19207
PORT_OTHER = 19210
PROT_PUSH = 19301

if __name__ == "__main__":
    agv_manager.start()
    cams = camera_manager
    for serial in (CAMERA_HEAD, CAMERA_CHEST):
        try:
            cams.start(serial)
        except RuntimeError as e:
            print(f"[警告] 相机 {serial} 启动失败: {e}")

    try:
        main()
    finally:
        agv_manager.stop()
        cams.stop_all()