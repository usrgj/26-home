from ultralytics import YOLO
import cv2
import numpy as np
import pyrealsense2 as rs

# ====================== 深度相机序列号（你原来的） ======================
CAMERA_SERIAL = {
    "head": "151222072331",
    "chest": "151222073707",
    "left_arm": "141722075710",
    "right_arm": "239722070896"
}

# ====================== 模型路径（你原来的） ======================
COCO_MODEL_PATH   = "/home/blinx/桌面/26-home/task2/behaviors/AI/yolov8n.pt"
PLATE_MODEL_PATH  = "/home/blinx/桌面/26-home/task2/behaviors/AI/best.pt"

coco_model = YOLO(COCO_MODEL_PATH, task="detect", verbose=False)
plate_model = YOLO(PLATE_MODEL_PATH, task="detect", verbose=False)

# ====================== 打开深度相机（稳定版） ======================
def open_camera_by_position(camera_name):
    pipeline = rs.pipeline()
    config = rs.config()

    # 不写死序列号，直接打开任意已连接的深度相机
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        pipeline.start(config)
        print("✅ 深度相机已打开，正在显示画面...")
        return pipeline
    except Exception as e:
        print(f"❌ 相机打开失败: {e}")
        return None

# ====================== IOU ======================
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    xi1 = max(x1, a1)
    yi1 = max(y1, b1)
    xi2 = min(x2, a2)
    yi2 = min(y2, b2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (a2 - a1) * (b2 - b1)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# ====================== 检测绘制 ======================
def detect_and_draw(frame, conf_threshold=0.5, iou_threshold=0.4):
    plate_detections = []
    plate_results = plate_model(frame, verbose=False)
    for r in plate_results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            cls = int(box.cls[0])
            label = plate_model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            plate_detections.append((x1, y1, x2, y2, label, conf))

    coco_keep_labels = ["milk", "dining table"]
    coco_detections = []
    coco_results = coco_model(frame, verbose=False)
    for r in coco_results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            cls = int(box.cls[0])
            label = coco_model.names[cls]
            if label not in coco_keep_labels:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            coco_box = (x1, y1, x2, y2)

            duplicate = False
            for p_box in plate_detections:
                if iou(coco_box, p_box[:4]) > iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                coco_detections.append((x1, y1, x2, y2, label, conf))

    all_detections = plate_detections + coco_detections

    for det in all_detections:
        x1, y1, x2, y2, label, conf = det
        if label == "plate":
            color = (0, 255, 0)
        elif label == "oatmeal":
            color = (0, 255, 255)
        elif label == "crash_can":
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# ====================== 主函数：只保证打开相机 + 出画面 ======================
def start_detection(camera_name="head"):
    pipeline = open_camera_by_position(camera_name)
    if not pipeline:
        return

    align = rs.align(rs.stream.color)

    while True:
        try:
            frames = pipeline.wait_for_frames()
        except:
            continue

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            continue

        # 转成图像
        color_image = np.asanyarray(color_frame.get_data())

        # 检测
        color_image = detect_and_draw(color_image)

        # 显示画面
        cv2.imshow("Kitchen Camera", color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_detection()

