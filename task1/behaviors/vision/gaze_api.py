from ultralytics import YOLO
from .gaze_tracking import get_person_direction

class GazeAPI:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect_persons(self, frame, conf=0.5):
        results = self.model(frame, conf=conf, classes=0, verbose=False)
        bboxes = []
        for r in results:
            for b in r.boxes:
                bboxes.append(tuple(map(int, b.xyxy[0])))
        return bboxes

    def get_person_direction(self, frame, conf=0.5, threshold=0.15):
        bboxes = self.detect_persons(frame, conf)
        if not bboxes:
            return None
        h, w = frame.shape[:2]
        ctrl = get_person_direction(bboxes[0], w, h, threshold)
        return ctrl