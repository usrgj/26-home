from head_control import HeadCameraController
import time
with HeadCameraController() as camera:
    camera.rotate_horizontal("+8000")   # 测试左右
    time.sleep(3)
    for i in range(10):
        camera.rotate_horizontal_rel("-800")
        time.sleep(0.1)