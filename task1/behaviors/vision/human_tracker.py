import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import tempfile
import os
import json
from client import analyze_person_features
from seat_manager import SeatManager
from head_control import HeadCameraController
from gaze_tracking import get_person_direction, draw_direction_indicator

class RoboCupReIDTracker:
    def __init__(self, debug=False):
        self.persons = {}
        self.frame_count = 0
        self.person_counter = 0
        self.target_guests = {}
        self.guest_embeddings = {}
        self.guest_assigned = set()
        self.guest_features = {}
        self.debug = debug
        
        self.face_analyzer = FaceAnalysis(name='buffalo_l')
        self.face_analyzer.prepare(ctx_id=0, det_size=(160, 160))
    
    def extract_face_embedding(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        # ★ 扩大裁剪区域，提高人脸检测率
        h, w = frame.shape[:2]
        x1 = max(0, x1 - 10)
        y1 = max(0, y1 - 10)
        x2 = min(w, x2 + 10)
        y2 = min(h, y2 + 10)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, 0.0
        
        try:
            faces = self.face_analyzer.get(crop)
            if faces:
                face = faces[0]
                return face.embedding, float(face.det_score) if hasattr(face, 'det_score') else 0.9
        except Exception as e:
            if self.debug:
                print(f"❗ 人脸提取失败: {e}")
        return None, 0.0
    
    def match_guest(self, embedding, threshold=0.55):  # ★ 降低阈值，提高匹配率
        if not self.guest_embeddings or embedding is None:
            return None, 0.0
        
        best_name, best_sim = None, threshold
        for guest_name, emb_list in self.guest_embeddings.items():
            for emb in emb_list:
                sim = np.dot(embedding, emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(emb) + 1e-5
                )
                if sim > best_sim:
                    best_sim, best_name = sim, guest_name
        return best_name, best_sim
    def match_person(self, embedding):
        best_pid, best_sim = None, 0.5
        for pid, data in self.persons.items():
            emb = data.get('face_embedding')
            if emb is None:
               continue
            sim = np.dot(embedding, emb) / (
            np.linalg.norm(embedding) * np.linalg.norm(emb) + 1e-5)
            if sim > best_sim:
              best_sim, best_pid = sim, pid
        return best_pid, best_sim
    def calc_distance(self, bbox):
         x1, y1, x2, y2 = bbox
         return (x2 - x1) * (y2 - y1)


    
    def update(self, detections, frame):
        self.frame_count += 1
        matched = set()
        do_face_extract = (self.frame_count % 5 == 0)  # ★ 改成每5帧提取一次，提高识别频率
        
        for det in detections:
            bbox = det['bbox']
            
            if do_face_extract:
                embedding, face_conf = self.extract_face_embedding(frame, bbox)
            else:
                embedding = None
                face_conf = 0.0
                # ★ 改进缓存匹配逻辑
                for pid, data in self.persons.items():
                    px1, py1, px2, py2 = data['bbox']
                    x1, y1, x2, y2 = bbox
                    # 计算IoU
                    ix1, iy1 = max(px1, x1), max(py1, y1)
                    ix2, iy2 = min(px2, x2), min(py2, y2)
                    if ix1 < ix2 and iy1 < iy2:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        union = (px2-px1)*(py2-py1) + (x2-x1)*(y2-y1) - inter
                        iou = inter / (union + 1e-5)
                        if iou > 0.3:  # IoU阈值
                            embedding = data['face_embedding']
                            face_conf = data.get('face_confidence', 0.0)
                            break
            
            if embedding is None:
                continue
            
            # ★ 先匹配已绑定客人
            matched_guest, guest_sim = self.match_guest(embedding, 0.55)
            if matched_guest:
                pid = self.target_guests[matched_guest]
                if pid in self.persons:
                    self.persons[pid].update({
                        'face_embedding': embedding, 'face_confidence': face_conf,
                        'guest_similarity': guest_sim, 'bbox': bbox,
                        'distance': self.calc_distance(bbox), 'last_seen': self.frame_count
                    })
                    matched.add(pid)
                    continue
            
            pid, person_sim = self.match_person(embedding)
            if pid is None:
                pid = self.person_counter
                self.person_counter += 1
                self.persons[pid] = {
                    'face_embedding': embedding, 'face_confidence': face_conf,
                    'person_similarity': 0.0, 'guest_similarity': 0.0, 'name': '',
                    'bbox': bbox, 'distance': self.calc_distance(bbox),
                    'last_seen': self.frame_count, 'features': {}
                }
            else:
                self.persons[pid].update({
                    'face_embedding': embedding, 'face_confidence': face_conf,
                    'person_similarity': person_sim, 'bbox': bbox,
                    'distance': self.calc_distance(bbox), 'last_seen': self.frame_count
                })
            matched.add(pid)
        
        return matched
    
    def assign_guest_name(self, person_id, guest_name, frame):
        if person_id in self.guest_assigned or guest_name in self.target_guests:
            print(f"⚠️ 已分配或已绑定")
            return
        
        info = self.persons.get(person_id)
        if not info:
            return
        
        emb = info['face_embedding']
        if emb is None:
            print("❌ 无人脸特征，请重试")
            return
        
        x1, y1, x2, y2 = info['bbox']
        crop = frame[y1:y2, x1:x2]
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, crop)
                features = analyze_person_features(tmp.name)
                os.unlink(tmp.name)
            
            # ★ 存储多个embedding样本，提高鲁棒性
            self.guest_embeddings[guest_name] = [emb]
            self.target_guests[guest_name] = person_id
            self.guest_assigned.add(person_id)
            self.guest_features[guest_name] = features
            self.persons[person_id]['features'] = features
            self.persons[person_id]['name'] = guest_name
            
            print(f"✓ {guest_name.upper()} 已绑定 (相似度阈值: 0.55)")
            print(f"特征: {json.dumps(features, ensure_ascii=False)}")
            
            with open("guest_features.json", "w", encoding="utf-8") as f:
                json.dump(self.guest_features, f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            print(f"❗ 绑定失败: {e}")

    
    def get_closest_unassigned_person(self, matched_ids):
        unassigned = [pid for pid in matched_ids if pid not in self.guest_assigned]
        return min(unassigned, key=lambda pid: self.persons[pid]['distance']) if unassigned else None
    
    def get_person_info(self, person_id):
        return self.persons.get(person_id)
    
    def describe_guest(self, guest_name):
        if guest_name not in self.guest_features:
            return "未绑定"
        f = self.guest_features[guest_name]
        desc = f"{f.get('性别')},{f.get('头发颜色')}头发,穿{f.get('衣服颜色')}"
        if f.get('眼镜') == '佩戴眼镜':
            desc += ",戴眼镜"
        if f.get('帽子') == '戴帽子':
            desc += ",戴帽子"
        return desc
def smooth_gaze_tracking(head_controller, tracker, frame, model, cap, guest_name, axis='horizontal'):
    """平滑目光追踪 - 让人物移动到画面中心"""
    info = tracker.get_person_info(tracker.target_guests.get(guest_name))
    if not info:
        print(f"❌ {guest_name}未绑定")
        return
    
    print(f"\n=== 平滑转向{guest_name}（{axis}） ===")
    
    current_pos = 0
    max_iter = 20  # ★ 增加迭代次数
    step = 0x300
    threshold = 15  # ★ 减小阈值到15px，更精确
    
    for iteration in range(max_iter):
        ctrl = get_person_direction(info['bbox'], frame.shape[1], frame.shape[0])
        
        # ★ 根据轴选择偏移量
        if axis == 'horizontal':
            offset = ctrl['offset_x']
        else:
            offset = ctrl['offset_y']
        
        print(f"迭代{iteration+1}: 偏移{offset:.0f}px")
        
        if abs(offset) < threshold:
            print(f"✓ 转向完成，误差{abs(offset):.0f}px < {threshold}px")
            break
        
        try:
            if abs(offset) > 150:
                current_step = step * 2
            elif abs(offset) > 80:
                current_step = step
            else:
                current_step = step // 2
            
            # 反向：人在右边/下边，头向左/上转
            if offset < -threshold:
                print(f"  → 向正方向转 {current_step:X}")
                current_pos += current_step
            elif offset > threshold:
                print(f"  → 向负方向转 {current_step:X}")
                current_pos -= current_step
            
            if axis == 'horizontal':
                head_controller.rotate_horizontal(current_pos)
            else:
                head_controller.rotate_vertical(current_pos)
            
            print(f"  当前位置: {current_pos:X}")
            
            import time
            time.sleep(0.2)
            
            ret, frame = cap.read()
            if ret:
                dets = model(frame, conf=0.5, classes=0, verbose=False)
                detections = [{'bbox': tuple(map(int, b.xyxy[0]))} for r in dets for b in r.boxes]
                matched = tracker.update(detections, frame)
                info = tracker.get_person_info(tracker.target_guests.get(guest_name))
                if not info:
                    print("❌ 人物丢失")
                    break
        
        except Exception as e:
            print(f"  ❌ 转向失败: {e}")
            break
    
    print(f"✓ {guest_name}转向完成\n")

def continuous_gaze_tracking(head_controller, tracker, frame, model, cap, target, duration=30):
    """持续目光追踪 - 自动跟随人物移动"""
    import time
    start_time = time.time()
    
    current_pos_h = 0
    current_pos_v = 0
    
    threshold = 25
    step = 0x250
    
    print(f"🎯 追踪 {target.upper()}，持续 {duration} 秒（按ESC提前退出）...")
    
    frame_count = 0
    while time.time() - start_time < duration:
        info = tracker.get_person_info(tracker.target_guests.get(target))
        if not info:
            print("⚠️ 人物丢失，等待重新检测...")
            time.sleep(0.1)
            continue
        
        ctrl = get_person_direction(info['bbox'], frame.shape[1], frame.shape[0])
        offset_x = ctrl['offset_x']
        offset_y = ctrl['offset_y']
        
        adjusted = False
        
        if abs(offset_x) > threshold:
            if offset_x > 0:
                current_pos_h -= step
            else:
                current_pos_h += step
            head_controller.rotate_horizontal(current_pos_h)
            adjusted = True
            print(f"  ↔ 水平调整: {offset_x:.0f}px")
        
        if abs(offset_y) > threshold:
            if offset_y > 0:
                current_pos_v -= step
            else:
                current_pos_v += step
            head_controller.rotate_vertical(current_pos_v)
            adjusted = True
            print(f"  ↕ 竖直调整: {offset_y:.0f}px")
        
        ret, frame = cap.read()
        if ret:
            dets = model(frame, conf=0.5, classes=0, verbose=False)
            detections = [{'bbox': tuple(map(int, b.xyxy[0]))} for r in dets for b in r.boxes]
            matched = tracker.update(detections, frame)
            
            # 显示画面
            for pid in matched:
                info_display = tracker.get_person_info(pid)
                if info_display:
                    x1, y1, x2, y2 = info_display['bbox']
                    name = info_display['name'] or f"ID{pid}"
                    col = (0, 255, 0) if info_display['name'] else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                    
                    # ★ 显示偏移信息
                    if info_display['name'] == target:
                        cv2.putText(frame, f"Offset: X={offset_x:.0f} Y={offset_y:.0f}", 
                                   (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # ★ 显示追踪状态
            elapsed = int(time.time() - start_time)
            cv2.putText(frame, f"TRACKING: {target.upper()} ({elapsed}s/{duration}s)", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow('RoboCup@Home', frame)
            
            # ★ 按ESC退出
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\n⏹ 用户中断追踪")
                break
        
        time.sleep(0.1 if adjusted else 0.05)
    
    print(f"✓ {target.upper()} 追踪完成")


def main():
    print("\n=== RoboCup@Home ReID ===\n")
    
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(10)
    
    # ★ 改进摄像头设置
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 提高分辨率
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 开启自动对焦
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 自动曝光
    
    if not cap.isOpened():
        print("❗ 无法打开摄像头")
        return
    
    # 等待摄像头稳定
    import time
    time.sleep(1)
    
    tracker = RoboCupReIDTracker(debug=False)
    seat_manager = SeatManager(debug=False)
    head_controller = HeadCameraController(port='/dev/ttyS1')
    
    try:
        head_controller.enable()
        print("✓ 头部已激活")
    except Exception as e:
        print(f"⚠️ 头部初始化失败: {e}")
    
    print("快捷键: 1=Jack  2=Tom  d=描述  i=看Tom  j=看Jack  h=回中  q=退出\n")
    
    detections = []  # ★ 初始化
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if tracker.frame_count % 2 == 0:
            dets = model(frame, conf=0.5, classes=0, verbose=False)
            detections = [{'bbox': tuple(map(int, b.xyxy[0]))} for r in dets for b in r.boxes]
        
        matched = tracker.update(detections, frame)
        
        for pid in matched:
            info = tracker.get_person_info(pid)
            if info:
                x1, y1, x2, y2 = info['bbox']
                name = info['name'] or f"ID{pid}"
                col = (0, 255, 0) if info['name'] else (0, 165, 255)
                
                # ★ 显示相似度
                if info['name']:
                    sim_text = f"{info.get('guest_similarity', 0):.2f}"
                    cv2.putText(frame, sim_text, (x1, y2 + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        
        cv2.putText(frame, f"Frame:{tracker.frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('RoboCup@Home', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            if matched:
                pid = tracker.get_closest_unassigned_person(matched)
                if pid is not None:
                    tracker.assign_guest_name(pid, 'jack', frame)
                else:
                    print("⚠️ 无未分配人物")
            else:
                print("⚠️ 无检测人物")
        elif key == ord('2'):
            if matched:
                pid = tracker.get_closest_unassigned_person(matched)
                if pid is not None:
                    tracker.assign_guest_name(pid, 'tom', frame)
                else:
                    print("⚠️ 无未分配人物")
            else:
                print("⚠️ 无检测人物")
        elif key == ord('d'):
            print(f"Jack: {tracker.describe_guest('jack')}")
            print(f"Tom: {tracker.describe_guest('tom')}\n")
        elif key == ord('i'):
            smooth_gaze_tracking(head_controller, tracker, frame, model, cap, 'tom', axis='horizontal')
            smooth_gaze_tracking(head_controller, tracker, frame, model, cap, 'tom', axis='vertical')
        elif key == ord('j'):
            smooth_gaze_tracking(head_controller, tracker, frame, model, cap, 'jack', axis='horizontal')
            smooth_gaze_tracking(head_controller, tracker, frame, model, cap, 'jack', axis='vertical')
        elif key == ord('t'):  # ★ 新增：测试持续追踪
            if matched:
                pid = tracker.get_closest_unassigned_person(matched)
                if pid is None and tracker.target_guests:
            # 如果没有未分配的，就用已绑定的第一个
                   target_name = list(tracker.target_guests.keys())[0]
                   print(f"\n🎯 开始持续追踪 {target_name.upper()}，按'q'停止")
                   continuous_gaze_tracking(head_controller, tracker, frame, model, cap, target_name, duration=30)
                else:
                   print("⚠️ 请先绑定一个人")
            else:
                print("⚠️ 无检测人物")


        elif key == ord('h'):
            try:
                print("正在回中...")
                head_controller.home()
                print("✓ 头部回中成功")
            except Exception as e:
                print(f"❌ 头部回中失败: {e}")
    
    try:
        head_controller.close()
    except:
        pass
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()