<<<<<<< HEAD
import threading
import time
from ultralytics import YOLO
from .gaze_tracking import get_person_direction  

class GazeAPI:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect_persons(self, frame, conf=0.5):
        """返回当前帧的所有人体框 [(x1, y1, x2, y2), ...]"""
        results = self.model(frame, conf=conf, classes=0, verbose=False)
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
            threshold = 25
            step = 0x400
            while not stop_event.is_set() and time.time() - start_time < duration:
                # 兼容OpenCV/RealSense相机
                frame = None
                if hasattr(cam, "get_color_frame"):
                    frame = cam.get_color_frame()
                elif hasattr(cam, "get_frames"):
                    color_frame, *_ = cam.get_frames()
                    frame = color_frame
                elif hasattr(cam, "read"):
                    ret, frame = cam.read()
                    if not ret:
                        frame = None
                if frame is None:
                    time.sleep(0.1)
                    continue
                bboxes = self.detect_persons(frame)
                if not bboxes:
                    time.sleep(0.1)
                    continue
                bbox = max(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                ctrl = get_person_direction(bbox, frame.shape[1], frame.shape[0])
                offset_x = ctrl['offset_x']
                offset_y = ctrl['offset_y']
                if abs(offset_x) > threshold:
                    if offset_x > 0:
                        current_pos_h -= step
                    else:
                        current_pos_h += step
                    head_controller.rotate_horizontal(current_pos_h)
                if abs(offset_y) > threshold:
                    if offset_y > 0:
                        current_pos_v -= step
                    else:
                        current_pos_v += step
                    head_controller.rotate_vertical(current_pos_v)
                time.sleep(0.1)
            print("✓ 最近人体追踪线程结束")
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        return t, stop_event

=======
import threading
import time
from ultralytics import YOLO
from .gaze_tracking import get_person_direction  

class GazeAPI:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect_persons(self, frame, conf=0.5):
        """返回当前帧的所有人体框 [(x1, y1, x2, y2), ...]"""
        results = self.model(frame, conf=conf, classes=0, verbose=False)
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
            threshold = 25
            step = 0x400
            while not stop_event.is_set() and time.time() - start_time < duration:
                # 兼容OpenCV/RealSense相机
                frame = None
                if hasattr(cam, "get_color_frame"):
                    frame = cam.get_color_frame()
                elif hasattr(cam, "get_frames"):
                    color_frame, *_ = cam.get_frames()
                    frame = color_frame
                elif hasattr(cam, "read"):
                    ret, frame = cam.read()
                    if not ret:
                        frame = None
                if frame is None:
                    time.sleep(0.1)
                    continue
                bboxes = self.detect_persons(frame)
                if not bboxes:
                    time.sleep(0.1)
                    continue
                bbox = max(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                ctrl = get_person_direction(bbox, frame.shape[1], frame.shape[0])
                offset_x = ctrl['offset_x']
                offset_y = ctrl['offset_y']
                if abs(offset_x) > threshold:
                    if offset_x > 0:
                        current_pos_h -= step
                    else:
                        current_pos_h += step
                    head_controller.rotate_horizontal(current_pos_h)
                if abs(offset_y) > threshold:
                    if offset_y > 0:
                        current_pos_v -= step
                    else:
                        current_pos_v += step
                    head_controller.rotate_vertical(current_pos_v)
                time.sleep(0.1)
            print("✓ 最近人体追踪线程结束")
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        return t, stop_event

>>>>>>> dev
