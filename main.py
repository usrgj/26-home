from __future__ import annotations
from common.follow.main import main
from common.agv_api import  agv
import time
from common.camera.config import CAMERA_CHEST,CAMERA_HEAD
from common.camera import camera_manager



if __name__ == "__main__":
    agv.start()
    cams = camera_manager
    for serial in (CAMERA_HEAD, CAMERA_CHEST):
        try:
            cams.start(serial)
        except RuntimeError as e:
            print(f"[警告] 相机 {serial} 启动失败: {e}")

    try:
        main()
    finally:
        agv.stop()
        cams.stop_all()