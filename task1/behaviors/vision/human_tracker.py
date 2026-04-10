import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import time
import tempfile
import os
import json

try:
    from .client import analyze_person_features
    from .seat_manager import SeatManager
except ImportError:
    # 兼容直接运行该文件进行单独调试
    from client import analyze_person_features
    from seat_manager import SeatManager


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
        
        # 人脸分析器
        self.face_analyzer = FaceAnalysis(name='buffalo_l')
        self.face_analyzer.prepare(ctx_id=0, det_size=(160, 160))
    
    def extract_face_embedding(self, frame, bbox):
        """提取人脸特征向量"""
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, 0.0
        
        try:
            faces = self.face_analyzer.get(crop)
            if faces:
                face = faces[0]
                embedding = face.embedding
                confidence = float(face.det_score) if hasattr(face, 'det_score') else 0.9
                return embedding, confidence
        except Exception as e:
            if self.debug:
                print(f"❗ 人脸提取失败: {e}")
        return None, 0.0
    
    def calc_distance(self, bbox):
        """计算人体距离（用面积代替）"""
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        return -area
    
    def match_person(self, embedding, threshold=0.5):
        """匹配已知人物"""
        if not self.persons or embedding is None:
            return None, 0.0
        
        best_id = None
        best_sim = threshold
        
        for pid, data in self.persons.items():
            if data['face_embedding'] is not None:
                sim = np.dot(embedding, data['face_embedding']) / (
                    np.linalg.norm(embedding) * np.linalg.norm(data['face_embedding']) + 1e-5
                )
                if sim > best_sim:
                    best_sim = sim
                    best_id = pid
        
        return best_id, best_sim
    
    def match_guest(self, embedding, threshold=0.6):
        """匹配已绑定的客人"""
        if not self.guest_embeddings or embedding is None:
            return None, 0.0
        
        best_name = None
        best_sim = threshold
        
        for guest_name, embeddings_list in self.guest_embeddings.items():
            for emb in embeddings_list:
                sim = np.dot(embedding, emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(emb) + 1e-5
                )
                if sim > best_sim:
                    best_sim = sim
                    best_name = guest_name
        
        return best_name, best_sim
    
    def update(self, detections, frame):
        """更新追踪状态"""
        self.frame_count += 1
        matched = set()
        
        # 正常情况下每10帧提取一次人脸特征以加速；
        # 但对首次出现或附近没有可复用缓存的人，必须立即提取，
        # 否则新客人永远无法在短时间窗口内完成绑定。
        do_face_extract = (self.frame_count % 10 == 0)
        
        for det in detections:
            bbox = det['bbox']
            
            if do_face_extract:
                embedding, face_conf = self.extract_face_embedding(frame, bbox)
            else:
                # 使用缓存特征
                embedding = None
                face_conf = 0.0
                for pid, data in self.persons.items():
                    if abs(data['bbox'][0] - bbox[0]) < 100 and abs(data['bbox'][1] - bbox[1]) < 100:
                        embedding = data['face_embedding']
                        face_conf = data.get('face_confidence', 0.0)
                        break

                if embedding is None:
                    embedding, face_conf = self.extract_face_embedding(frame, bbox)
            
            if embedding is None:
                continue
            
            # 先匹配已绑定的客人
            matched_guest, guest_sim = self.match_guest(embedding, threshold=0.6)
            if matched_guest:
                pid = self.target_guests[matched_guest]
                if pid in self.persons:
                    self.persons[pid]['face_embedding'] = embedding
                    self.persons[pid]['face_confidence'] = face_conf
                    self.persons[pid]['guest_similarity'] = guest_sim
                    self.persons[pid]['bbox'] = bbox
                    self.persons[pid]['distance'] = self.calc_distance(bbox)
                    self.persons[pid]['last_seen'] = self.frame_count
                    matched.add(pid)
                    continue
            
            # 再匹配已知人物
            pid, person_sim = self.match_person(embedding)
            
            if pid is None:
                # 新人物
                pid = self.person_counter
                self.person_counter += 1
                self.persons[pid] = {
                    'face_embedding': embedding,
                    'face_confidence': face_conf,
                    'person_similarity': 0.0,
                    'guest_similarity': 0.0,
                    'name': '',
                    'bbox': bbox,
                    'distance': self.calc_distance(bbox),
                    'last_seen': self.frame_count,
                    'features': {}
                }
            else:
                # 更新已知人物
                self.persons[pid]['face_embedding'] = embedding
                self.persons[pid]['face_confidence'] = face_conf
                self.persons[pid]['person_similarity'] = person_sim
                self.persons[pid]['bbox'] = bbox
                self.persons[pid]['distance'] = self.calc_distance(bbox)
                self.persons[pid]['last_seen'] = self.frame_count
            
            matched.add(pid)
        
        return matched
    
    def assign_guest_name(self, person_id, guest_name, frame):
        """分配客人名字并分析外貌特征"""
        if person_id in self.persons:
            if person_id in self.guest_assigned:
                print(f"⚠️ Person {person_id} 已被分配")
                return
            if guest_name in self.target_guests:
                print(f"⚠️ {guest_name} 已绑定")
                return
            
            embedding = self.persons[person_id]['face_embedding']
            
            if embedding is not None:
                # ★ 裁剪bbox区域
                bbox = self.persons[person_id]['bbox']
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                
                # ★ 保存临时图像
                try:
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        cv2.imwrite(tmp.name, crop)
                        # ★ 调用大模型（会显示推理中）
                        features = analyze_person_features(tmp.name)
                        os.unlink(tmp.name)
                    
                    self.guest_embeddings[guest_name] = [embedding]
                    self.persons[person_id]['name'] = guest_name
                    self.persons[person_id]['features'] = features
                    self.target_guests[guest_name] = person_id
                    self.guest_assigned.add(person_id)
                    self.guest_features[guest_name] = features
                    
                    print(f"✓ {guest_name.upper()} 已绑定 (ID {person_id})")
                    print(f"  外貌特征: {json.dumps(features, ensure_ascii=False)}")
                    save_path = "guest_features.json"
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(self.guest_features, f, ensure_ascii=False, indent=2)
                    print(f"✅ 特征已保存到: {save_path}")                
                    if "jack" in self.guest_features and "tom" in self.guest_features:
                        print("\n==== 两位客人已全部识别完成 ====")              
                        jack_feat = self.guest_features["jack"]
                        tom_feat = self.guest_features["tom"]
                    
                        jack_desc = self.describe_guest("jack")
                        tom_desc = self.describe_guest("tom")
                    
                    # 向第二个客人介绍第一个
                        print(f"\n【对 Tom 介绍】：{jack_desc}")
                    # 向第一个客人介绍第二个
                        print(f"【对 Jack 介绍】：{tom_desc}")

                    
                except Exception as e:
                    print(f"❗ 绑定客人失败: {e}")

    
    def get_closest_unassigned_person(self, matched_ids):
        """获取最近的未分配人物"""
        unassigned = [pid for pid in matched_ids if pid not in self.guest_assigned]
        if not unassigned:
            return None
        return min(unassigned, key=lambda pid: self.persons[pid]['distance'])
    
    def get_person_info(self, person_id):
        """获取人物信息"""
        if person_id in self.persons:
            return self.persons[person_id]
        return None
    
    def describe_guest(self, guest_name):
        """返回客人外貌描述"""
        if guest_name in self.guest_features:
            features = self.guest_features[guest_name]
            desc = f"{features.get('性别', '未知')}，{features.get('头发颜色', '未知')}头发，" \
                   f"穿着{features.get('衣服颜色', '未知')}衣服"
            if features.get('眼镜') == '佩戴眼镜':
                desc += "，戴眼镜"
            if features.get('帽子') == '戴帽子':
                desc += "，戴帽子"
            return desc
        return "未知"

feature_extraction = RoboCupReIDTracker(debug=False)

def main():
    print("\n=== RoboCup@Home ReID + 大模型 ===")
    print("初始化中...\n")
    
    # 初始化YOLOv8
    model = YOLO('yolov8n.pt')
    
    # 打开摄像头
    cap = cv2.VideoCapture(10)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❗ 无法打开摄像头 ID 10")
        return
    
    tracker = RoboCupReIDTracker(debug=False)
    seat_manager = SeatManager(debug=False)  # ★ 初始化座位管理器
    # ★ 等导航组提供座位配置后加载
    # seats_config = [...]
    # seat_manager.load_seats(seats_config)
    
    print("按 '1' 标记为 Jack")
    print("按 '2' 标记为 Tom")
    print("按 'd' 查看客人外貌描述")
    print("按 'q' 退出\n")
    
    detections = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❗ 读取视频帧失败")
            break
        
        # 每2帧检测一次（加速）
        if tracker.frame_count % 2 == 0:
            results = model(frame, conf=0.5, classes=0, verbose=False)
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({'bbox': (x1, y1, x2, y2)})
        
        # 更新追踪
        matched = tracker.update(detections, frame)
           # ★ 更新座位状态（同一个frame）
        person_bboxes = [det['bbox'] for det in detections]
        seat_manager.update_seat_status(frame, person_bboxes, use_pose=False)
        # ★ 绘制座位
        frame = seat_manager.draw_seats(frame)
        # 绘制追踪结果
        for pid in matched:
            info = tracker.get_person_info(pid)
            if info:
                x1, y1, x2, y2 = info['bbox']
                name = info['name'] if info['name'] else f"ID {pid}"
                color = (0, 255, 0) if info['name'] else (0, 165, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 显示帧数
        cv2.putText(frame, f"Frame: {tracker.frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('RoboCup@Home ReID', frame)
        
        # 键盘输入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            if matched:
                closest_pid = tracker.get_closest_unassigned_person(matched)
                if closest_pid is not None:
                    print(f"→ 标记 Person {closest_pid} 为 Jack...")
                    tracker.assign_guest_name(closest_pid, 'jack', frame)
                else:
                    print("⚠️ 没有未分配的人物")
            else:
                print("⚠️ 没有检测到人物")
        elif key == ord('2'):
            if matched:
                closest_pid = tracker.get_closest_unassigned_person(matched)
                if closest_pid is not None:
                    print(f"→ 标记 Person {closest_pid} 为 Tom...")
                    tracker.assign_guest_name(closest_pid, 'tom', frame)
                else:
                    print("⚠️ 没有未分配的人物")
            else:
                print("⚠️ 没有检测到人物")
        elif key == ord('s'):  # ★ 按's'查看座位状态
            print("\n--- 座位状态 ---")
            for seat_id, status in seat_manager.seat_status.items():
                print(f"座位{seat_id}: {'占用' if status['occupied'] else '空'} (置信度{status['confidence']:.2f})")
            empty = seat_manager.get_empty_seats()
            print(f"空座位: {[s['id'] for s in empty]}")
        elif key == ord('d'):
            print("\n--- 客人外貌描述 ---")
            for name in ['jack', 'tom']:
                desc = tracker.describe_guest(name)
                if desc != "未知":
                    print(f"{name.upper()}: {desc}")
                else:
                    print(f"{name.upper()}: 未绑定")
            print()
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n程序已退出")


if __name__ == '__main__':
    main()
