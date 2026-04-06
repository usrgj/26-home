import cv2
import numpy as np
from collections import defaultdict

class SeatManager:
    def __init__(self, debug=False):
        self.seats = []  # [{'id': 1, 'bbox': (x1,y1,x2,y2), 'position': (px,py)}, ...]
        self.seat_status = {}  # {seat_id: {'occupied': bool, 'confidence': float, 'frames': int}}
        self.debug = debug
        self.frame_count = 0
        self.occupy_threshold = 3  # 连续3帧判断为占用
    
    def load_seats(self, seat_config):
        """加载座位配置（导航组提供）"""
        self.seats = seat_config
        for seat in self.seats:
            self.seat_status[seat['id']] = {
                'occupied': False,
                'confidence': 0.0,
                'frames': 0
            }
        print(f"✓ 加载 {len(self.seats)} 个座位")
    
    def is_person_in_seat(self, person_bbox, seat_bbox, use_pose=False, pose_data=None):
        """更严谨的座位占用判断"""
        px1, py1, px2, py2 = person_bbox
        sx1, sy1, sx2, sy2 = seat_bbox
        
        # 条件1：人的bbox与座位bbox有显著重叠
        iou = self._calculate_iou(person_bbox, seat_bbox)
        if iou < 0.2:  # 重叠不足20%，肯定不在座位上
            return False, 0.0
        
        # 条件2：人的下半身在座位范围内（关键！）
        person_center_x = (px1 + px2) / 2
        person_bottom_y = py2
        person_top_y = py1
        
        # 人的下半身（腿部）应该在座位范围内
        legs_in_seat = sx1 < person_center_x < sx2 and sy1 < person_bottom_y < sy2
        
        # 条件3：人的高度合理（不是站着）
        person_height = py2 - py1
        seat_height = sy2 - sy1
        height_ratio = person_height / seat_height if seat_height > 0 else 0
        
        # 坐着时人的高度应该 < 座位高度的1.5倍
        reasonable_height = height_ratio < 1.5
        
        # 综合判断
        basic_check = iou > 0.2 and legs_in_seat and reasonable_height
        
        if not basic_check:
            return False, 0.0
        
        # 条件4：姿态确认（如果有）
        if use_pose and pose_data:
            is_sitting = self._check_sitting_pose(pose_data)
            confidence = 0.95 if is_sitting else 0.4
            return is_sitting, confidence
        
        # 没有姿态数据时，基于几何判断
        confidence = 0.8 if legs_in_seat else 0.5
        return True, confidence

    
    def _check_sitting_pose(self, pose_data):
        """检查姿态是否为坐着（需要你的姿态识别代码）"""
        # pose_data 格式：{'keypoints': [...], 'confidence': [...]}
        # 判断标准：膝盖高度 < 臀部高度
        try:
            keypoints = pose_data.get('keypoints', [])
            if len(keypoints) < 12:  # 需要足够的关键点
                return False
            
            # COCO格式：11=左膝, 12=右膝, 8=左臀, 9=右臀
            left_knee_y = keypoints[11][1] if len(keypoints) > 11 else float('inf')
            right_knee_y = keypoints[12][1] if len(keypoints) > 12 else float('inf')
            left_hip_y = keypoints[8][1] if len(keypoints) > 8 else 0
            right_hip_y = keypoints[9][1] if len(keypoints) > 9 else 0
            
            knee_y = min(left_knee_y, right_knee_y)
            hip_y = max(left_hip_y, right_hip_y)
            
            # 坐着时：膝盖在臀部下方
            return knee_y > hip_y
        except:
            return False
    
    def update_seat_status(self, frame, person_detections, pose_list=None, use_pose=False):
        """更新座位占用状态"""
        self.frame_count += 1
        occupied_seats = set()
        
        # 检查每个人是否在座位上
        for idx, person_bbox in enumerate(person_detections):
            pose_data = pose_list[idx] if use_pose and pose_list and idx < len(pose_list) else None
            
            for seat in self.seats:
                is_in_seat, confidence = self.is_person_in_seat(
                    person_bbox, 
                    seat['bbox'],
                    use_pose=use_pose,
                    pose_data=pose_data
                )
                
                if is_in_seat:
                    occupied_seats.add(seat['id'])
                    if self.debug:
                        print(f"  座位{seat['id']}: 有人 (置信度{confidence:.2f})")
        
        # 更新座位状态（使用时间序列判断）
        for seat in self.seats:
            seat_id = seat['id']
            
            if seat_id in occupied_seats:
                self.seat_status[seat_id]['frames'] += 1
                self.seat_status[seat_id]['confidence'] = min(1.0, self.seat_status[seat_id]['frames'] / self.occupy_threshold)
            else:
                self.seat_status[seat_id]['frames'] = max(0, self.seat_status[seat_id]['frames'] - 1)
                self.seat_status[seat_id]['confidence'] = max(0.0, self.seat_status[seat_id]['frames'] / self.occupy_threshold)
            # 判断座位是否真的被占用
            self.seat_status[seat_id]['occupied'] = self.seat_status[seat_id]['frames'] >= self.occupy_threshold
    
    def get_empty_seats(self):
        """获取所有空座位（按ID排序）"""
        empty = [s for s in self.seats if not self.seat_status[s['id']]['occupied']]
        return sorted(empty, key=lambda s: s['id'])
    
    def get_nearest_empty_seat(self, robot_position=None):
        """获取最近的空座位"""
        empty_seats = self.get_empty_seats()
        
        if not empty_seats:
            return None
        
        if robot_position is None:
            # 没有机器人位置，返回第一个空座位
            return empty_seats[0]
        
        # 按距离排序
        robot_x, robot_y = robot_position
        empty_seats.sort(key=lambda s: (s['position'][0] - robot_x)**2 + (s['position'][1] - robot_y)**2)
        return empty_seats[0]
    
    def draw_seats(self, frame):
        """在图像上绘制座位"""
        for seat in self.seats:
            x1, y1, x2, y2 = seat['bbox']
            seat_id = seat['id']
            occupied = self.seat_status[seat_id]['occupied']
            
            # 颜色：红=占用，绿=空
            color = (0, 0, 255) if occupied else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 显示座位ID和状态
            status = "占用" if occupied else "空"
            cv2.putText(frame, f"Seat {seat_id} ({status})", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def get_seat_info(self):
        """获取所有座位信息"""
        info = {}
        for seat in self.seats:
            seat_id = seat['id']
            info[seat_id] = {
                'occupied': self.seat_status[seat_id]['occupied'],
                'confidence': self.seat_status[seat_id]['confidence'],
                'position': seat['position']
            }
        return info
    
    def reset(self):
        """重置座位状态"""
        for seat in self.seats:
            self.seat_status[seat['id']] = {
                'occupied': False,
                'confidence': 0.0,
                'frames': 0
            }


# 使用示例
if __name__ == '__main__':
    # 初始化座位管理器
    seat_manager = SeatManager(debug=True)
    
    # 导航组提供座位配置
    seats_config = [
        {'id': 1, 'bbox': (100, 200, 300, 400), 'position': (1.5, 2.0)},
        {'id': 2, 'bbox': (400, 200, 600, 400), 'position': (3.0, 2.0)},
        {'id': 3, 'bbox': (700, 200, 900, 400), 'position': (4.5, 2.0)},
    ]
    seat_manager.load_seats(seats_config)
    
    # 模拟人体检测结果
    person_detections = [
        (120, 250, 280, 450),  # 在座位1
        (420, 250, 580, 450),  # 在座位2
    ]
    
    # 更新座位状态
    seat_manager.update_seat_status(None, person_detections, use_pose=False)
    
    # 获取空座位
    empty_seats = seat_manager.get_empty_seats()
    print(f"空座位: {[s['id'] for s in empty_seats]}")
    
    # 获取最近的空座位
    nearest = seat_manager.get_nearest_empty_seat(robot_position=(0, 0))
    print(f"最近的空座位: {nearest['id'] if nearest else '无'}")
