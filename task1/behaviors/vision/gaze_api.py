import threading
import time
from ultralytics import YOLO
from .gaze_tracking import get_person_direction  
KP = 0.015
MAX_STEP_HORIZONTAL = 42000
MAX_STEP_VERTICAL = 32000

class GazeAPI:
    def __init__(self, model: YOLO | str = "yolov8n.pt"):
        """允许直接传已加载 YOLO 模型，或传模型路径字符串。"""
        self.model = model if isinstance(model, YOLO) else YOLO(model)

    def detect_persons(self, frame, conf=0.5):
        """返回当前帧的所有人体框 [(x1, y1, x2, y2), ...]"""
        results = self.model(frame, conf=conf, classes=0, verbose=False, stream=True)
        bboxes = []
        for r in results:
            for b in r.boxes:
                bboxes.append(tuple(map(int, b.xyxy[0])))
        return bboxes

    def get_person_direction(self, frame, conf=0.5, threshold=0.15, k=0.15):
        """对一帧图片返回第一个人体的方向、偏移等信息"""
        bboxes = self.detect_persons(frame, conf)
        if not bboxes:
            return None
        h, w = frame.shape[:2]
        return get_person_direction(bboxes[0], w, h, threshold, k)

    def start_gaze_tracking_nearest_person(self, head_controller, cam, duration=30):
        """
        能力接口：实时追踪画面中面积最大的人体，无需ID
        """
        stop_event = threading.Event()
        def worker():
            start_time = time.time()
            current_pos_h = 0
            current_pos_v = 0
            threshold = 50
            step = 0x500
            while not stop_event.is_set() and time.time() - start_time < duration:
                need_move = False
                # 兼容OpenCV/RealSense相机
                frame = None
                frame, *_ = cam.get_frames()
                
                # if hasattr(cam, "get_color_frame"):
                #     frame = cam.get_color_frame()
                # elif hasattr(cam, "get_frames"):
                #     color_frame, *_ = cam.get_frames()
                #     frame = color_frame
                # elif hasattr(cam, "read"):
                #     ret, frame = cam.read()
                #     if not ret:
                #         frame = None
                if frame is None:
                    stop_event.wait(0.01)
                    continue
                bboxes = self.detect_persons(frame)
                if not bboxes:
                    stop_event.wait(0.01)
                    continue
                bbox = max(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                ctrl = get_person_direction(bbox, frame.shape[1], frame.shape[0])
                offset_x = ctrl['offset_x']
                offset_y = ctrl['offset_y']
                if abs(offset_x) > threshold:
                    need_move = True
                    if offset_x > 0:
                        current_pos_h -= min(step * abs(offset_x) * KP, MAX_STEP_HORIZONTAL)
                    else:
                        current_pos_h += min(step * abs(offset_x) * KP, MAX_STEP_HORIZONTAL)
                    # head_controller.rotate_horizontal(current_pos_h)
                if abs(offset_y) > threshold:
                    need_move = True
                    if offset_y > 0:
                        current_pos_v -= min(step * abs(offset_y) * KP, MAX_STEP_VERTICAL)
                    else:
                        current_pos_v += min(step * abs(offset_y) * KP, MAX_STEP_VERTICAL)
                    # head_controller.rotate_vertical(current_pos_v)
                if need_move:
                    head_controller.move_absolute(current_pos_h, current_pos_v)
                # time.sleep(0.5)
            print("✓ 最近人体追踪线程结束")
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        return t, stop_event