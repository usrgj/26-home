import threading
import time
import cv2
import numpy as np
from common.skills.camera import camera_manager
from common.config import CAMERA_HEAD

class ColorViewer:
    def __init__(self, camera):
        self.camera = camera
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._show_loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        cv2.destroyAllWindows()

    def _show_loop(self):
        while not self.stop_event.is_set():
            try:
                color_frame, depth_frame = self.camera.get_frames()
                # 如果你的 get_frames() 只返回彩色帧，这里改成：
                # color_frame = self.camera.get_frames()
            except Exception as e:
                print(f"get_frames() failed: {e}")
                time.sleep(0.05)
                continue

            if color_frame is None:
                time.sleep(0.01)
                continue

            try:
                img = np.asarray(color_frame)

                cv2.imshow("Color", img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    self.stop_event.set()
                    break

            except Exception as e:
                print(f"imshow failed: {e}")
                time.sleep(0.05)


def main():
    cam = camera_manager.get(CAMERA_HEAD)
    # 替换成你的相机对象
    camera = cam

    viewer = ColorViewer(camera)
    viewer.start()

    try:
        input("按回车退出...\n")
    except KeyboardInterrupt:
        pass
    finally:
        viewer.stop()


if __name__ == "__main__":
    main()