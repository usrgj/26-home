import threading
import time
from ultralytics import YOLO
from .gaze_tracking import get_person_direction

yolo_model = YOLO('yolov8n.pt')

def detect_persons(frame, conf=0.5):
    results = yolo_model(frame, conf=conf, classes=0, verbose=False)
    bboxes = []
    for r in results:
        for b in r.boxes:
            bboxes.append(tuple(map(int, b.xyxy[0])))
    return bboxes

def start_gaze_tracking_nearest_person(head_controller, cam, duration=30, k=0.15):
    """
    能力接口：实时追踪画面中面积最大（最近）的人，不用ID绑定
    """
    stop_event = threading.Event()
    def worker():
        start_time = time.time()
        current_pos_h = 0
        current_pos_v = 0
        threshold = 25
        step = 0x400
        while not stop_event.is_set() and time.time() - start_time < duration:
            ret, frame = cam.read()
            if not ret:
                time.sleep(0.1)
                continue
            bboxes = detect_persons(frame)
            if not bboxes:
                time.sleep(0.1)
                continue
            # 找面积最大的人体框
            bbox = max(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            ctrl = get_person_direction(bbox, frame.shape[1], frame.shape[0], k=k)
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

