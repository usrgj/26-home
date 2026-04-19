"""
vision_detector.py — 视觉检测与ReID模块（改进版：在线特征更新 + 位置约束匹配）
"""
import math
import time
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from .config import (
    CAMERAS, PRIMARY_CAMERAS,
    DETECTION_CONFIDENCE_THRESHOLD, REID_SIMILARITY_THRESHOLD,
    REID_FEATURE_DIM,
)
from .robot_api import RobotAPI, CameraFrame, RobotPose


@dataclass
class PersonDetection:
    """单个人物检测结果"""
    # 图像空间
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x1, y1, x2, y2)
    confidence: float = 0.0
    
    # 3D空间 (机器人坐标系)
    local_x: float = 0.0       # 前方距离 (m)
    local_y: float = 0.0       # 左方偏移 (m)
    depth: float = 0.0         # 深度 (m)
    
    # 3D空间 (世界坐标系)
    world_x: float = 0.0
    world_y: float = 0.0
    
    # ReID特征
    feature: Optional[np.ndarray] = None  # 外观特征向量
    
    # 是否为跟随目标
    is_target: bool = False
    
    camera_name: str = ""
    timestamp: float = 0.0


class VisionDetector:
    """视觉检测器（改进版）"""
    
    def __init__(self):
        """
        初始化检测模型。
        """
        self._model = None
        self._reid_model = None
        
        # 跟随目标的外观特征模板
        self._target_feature: Optional[np.ndarray] = None
        # ----- 新增：目标特征在线更新系数（指数移动平均）-----
        self._feature_update_alpha = 0.2    # 新特征权重
        
        # ----- 新增：记录上次目标位置用于空间约束 -----
        self._last_target_world_pos: Optional[Tuple[float, float]] = None
        self._last_target_timestamp: float = 0.0
        
        from ultralytics import YOLO
        self._model = YOLO("yolov8s.pt")  
        
        print("[VisionDetector] 初始化完成（改进版：在线特征更新 + 位置约束）")
    
    # =====================================================================
    # 核心检测流程
    # =====================================================================
    def detect(self, robot_api: RobotAPI, robot_pose: RobotPose,
               predicted_target_pos: Optional[Tuple[float, float]] = None
               ) -> List[PersonDetection]:
        """
        从所有相机中检测人物。
        
        参数:
            robot_api: 机器人API实例
            robot_pose: 当前机器人位姿
            predicted_target_pos: (x,y) EKF预测的目标世界坐标，用于空间约束匹配
        
        返回:
            所有检测到的人物列表
        """
        all_detections = []
        
        for cam_name in PRIMARY_CAMERAS:
            try:
                frame = robot_api.get_camera_frame(cam_name)
                if frame.color_image is None:
                    continue
                
                detections = self._detect_in_frame(frame, cam_name, robot_pose)
                all_detections.extend(detections)
                
            except Exception as e:
                pass
        
        # 去重
        all_detections = self._deduplicate(all_detections)
        
        # ----- 修改：匹配目标时传入预测位置 -----
        self._match_target(all_detections, predicted_target_pos)
        
        # ----- 新增：更新目标特征（若找到目标）-----
        for det in all_detections:
            if det.is_target:
                self._update_target_feature(det)
                self._last_target_world_pos = (det.world_x, det.world_y)
                self._last_target_timestamp = time.time()
                break
        
        return all_detections
    
    def lock_target(self, detection: PersonDetection):
        """
        锁定跟随目标——记录目标的外观特征。
        """
        if detection.feature is not None:
            self._target_feature = detection.feature.copy()
            self._last_target_world_pos = (detection.world_x, detection.world_y)
            self._last_target_timestamp = time.time()
            print(f"[VisionDetector] 已锁定跟随目标，特征维度: {len(self._target_feature)}")
        else:
            print("[VisionDetector] 警告: 检测结果没有特征向量，无法锁定目标")

    def reset_target(self):
        """清除当前已锁定的目标特征。"""
        self._target_feature = None
        self._last_target_world_pos = None
    
    def lock_target_from_frame(self, robot_api: RobotAPI, camera_name: str = "head"):
        """
        从当前画面中自动选择最近的人物作为跟随目标并锁定。
        """
        try:
            frame = robot_api.get_camera_frame(camera_name)
            pose = robot_api.get_robot_pose()
            detections = self._detect_in_frame(frame, camera_name, pose)
            
            if not detections:
                print("[VisionDetector] 当前画面中未检测到人物")
                return False
            
            nearest = min(detections, key=lambda d: d.depth)
            self.lock_target(nearest)
            return True
            
        except Exception as e:
            print(f"[VisionDetector] 锁定目标失败: {e}")
            return False
    
    # =====================================================================
    # 内部方法
    # =====================================================================
    def _detect_in_frame(self, frame: CameraFrame, camera_name: str,
                          robot_pose: RobotPose) -> List[PersonDetection]:
        """
        在单个相机帧中检测人物。
        """
        # 位姿不可用时直接跳过，避免坐标换算抛异常。
        if robot_pose.x is None or robot_pose.y is None or robot_pose.theta is None:
            return []
        detections = []
        cam_config = CAMERAS[camera_name]
        timestamp = time.time()
        
        bboxes = self._run_detector(frame.color_image)
        
        for (x1, y1, x2, y2, conf) in bboxes:
            depth_m = self._get_depth_at_bbox(frame.depth_image, x1, y1, x2, y2)
            if depth_m <= 0 or depth_m > 15.0:
                continue
            
            # 像素坐标转相机坐标系
            cx_pixel = (x1 + x2) / 2.0
            cy_pixel = (y1 + y2) / 2.0
            
            fx = frame.fx
            fy = frame.fy
            ppx = frame.ppx
            ppy = frame.ppy

            cam_z = depth_m
            cam_x = (cx_pixel - ppx) * depth_m / fx
            cam_y = (cy_pixel - ppy) * depth_m / fy
            
            # 相机坐标系 → 机器人坐标系
            cam_yaw_rad = math.radians(cam_config["yaw"])
            cam_pitch_rad = math.radians(cam_config["pitch"])

            cos_p = math.cos(cam_pitch_rad)
            sin_p = math.sin(cam_pitch_rad)
            cam_z_rot = cam_z * cos_p + cam_y * sin_p

            cos_y = math.cos(cam_yaw_rad)
            sin_y = math.sin(cam_yaw_rad)
            robot_x = cam_config["x"] + cam_z_rot * cos_y + cam_x * sin_y
            robot_y = cam_config["y"] + cam_z_rot * sin_y - cam_x * cos_y
            
            # 机器人坐标系 → 世界坐标系
            cos_t = math.cos(robot_pose.theta)
            sin_t = math.sin(robot_pose.theta)
            world_x = robot_pose.x + robot_x * cos_t - robot_y * sin_t
            world_y = robot_pose.y + robot_x * sin_t + robot_y * cos_t
            
            feature = self._extract_feature(frame.color_image, x1, y1, x2, y2)
            
            det = PersonDetection(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                local_x=robot_x,
                local_y=robot_y,
                depth=depth_m,
                world_x=world_x,
                world_y=world_y,
                feature=feature,
                camera_name=camera_name,
                timestamp=timestamp,
            )
            detections.append(det)
        
        return detections
    
    def _run_detector(self, color_image: np.ndarray
                      ) -> List[Tuple[int, int, int, int, float]]:
        results = self._model(color_image, verbose=False)
        bboxes = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls != 0:
                    continue
                conf = float(box.conf[0])
                if conf < DETECTION_CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                bboxes.append((x1, y1, x2, y2, conf))
        return bboxes
    
    def _get_depth_at_bbox(self, depth_image: Optional[np.ndarray],
                           x1: int, y1: int, x2: int, y2: int) -> float:
        if depth_image is None:
            return -1.0
        
        h, w = depth_image.shape[:2]
        
        cx1 = int(x1 + (x2 - x1) * 0.25)
        cy1 = int(y1 + (y2 - y1) * 0.25)
        cx2 = int(x1 + (x2 - x1) * 0.75)
        cy2 = int(y1 + (y2 - y1) * 0.75)
        
        cx1 = max(0, min(cx1, w - 1))
        cx2 = max(0, min(cx2, w - 1))
        cy1 = max(0, min(cy1, h - 1))
        cy2 = max(0, min(cy2, h - 1))
        
        roi = depth_image[cy1:cy2, cx1:cx2]
        if roi.size == 0:
            return -1.0
        
        valid = roi[roi > 0]
        if len(valid) == 0:
            return -1.0
        
        depth_mm = float(np.median(valid))
        depth_m = depth_mm / 1000.0
        return depth_m
    
    def _extract_feature(self, color_image: np.ndarray,
                         x1: int, y1: int, x2: int, y2: int
                         ) -> np.ndarray:
        """
        提取人物外观特征向量（改进版：增加 HSV 直方图归一化稳健性）
        """
        try:
            import cv2
        except ImportError:
            return np.random.randn(REID_FEATURE_DIM).astype(np.float32)
        
        h, w = color_image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        person_crop = color_image[y1:y2, x1:x2]
        if person_crop.size == 0:
            return np.zeros(REID_FEATURE_DIM, dtype=np.float32)
        
        hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
        mid_y = person_crop.shape[0] // 2
        
        upper = hsv[:mid_y]
        lower = hsv[mid_y:]
        
        bins_h, bins_s = 32, 32
        
        features = []
        for region in [upper, lower]:
            if region.size == 0:
                features.extend([0.0] * (bins_h + bins_s))
                continue
            hist_h = cv2.calcHist([region], [0], None, [bins_h], [0, 180])
            hist_s = cv2.calcHist([region], [1], None, [bins_s], [0, 256])
            hist_h = hist_h.flatten() / (hist_h.sum() + 1e-6)
            hist_s = hist_s.flatten() / (hist_s.sum() + 1e-6)
            features.extend(hist_h.tolist())
            features.extend(hist_s.tolist())
        
        feature = np.array(features, dtype=np.float32)
        
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature /= norm
        
        return feature
    
    def _match_target(self, detections: List[PersonDetection],
                      predicted_pos: Optional[Tuple[float, float]] = None):
        """
        匹配已锁定的目标，综合外观相似度与空间位置约束。
        
        参数:
            predicted_pos: (x,y) EKF 预测的世界坐标，用于计算位置得分
        """
        if self._target_feature is None:
            return
        
        best_idx = -1
        best_score = -1.0
        
        # 使用上次目标位置或预测位置作为空间中心
        center_pos = predicted_pos if predicted_pos is not None else self._last_target_world_pos
        
        for i, det in enumerate(detections):
            if det.feature is None:
                continue
            
            # 外观相似度（余弦相似度）
            app_score = float(np.dot(self._target_feature, det.feature))
            
            # ----- 新增：位置得分（距离越近越高）-----
            pos_score = 0.0
            if center_pos is not None:
                dist = math.hypot(det.world_x - center_pos[0], det.world_y - center_pos[1])
                # 距离在 1.0m 内得满分，超过 2.0m 得 0 分
                pos_score = max(0.0, 1.0 - (dist - 1.0) / 1.0) if dist > 1.0 else 1.0
            
            # ----- 修改：综合得分（外观 0.7，位置 0.3）-----
            total_score = 0.7 * app_score + 0.3 * pos_score
            
            # 外观阈值必须满足最低要求，防止完全靠位置匹配
            if app_score < REID_SIMILARITY_THRESHOLD * 0.8:
                continue
            
            if total_score > REID_SIMILARITY_THRESHOLD and total_score > best_score:
                best_score = total_score
                best_idx = i
        
        if best_idx >= 0:
            detections[best_idx].is_target = True
    
    def _update_target_feature(self, detection: PersonDetection):
        """
        在线更新目标特征（指数移动平均），适应外观缓慢变化。
        """
        if self._target_feature is None or detection.feature is None:
            return
        
        # 指数移动平均
        new_feature = (1 - self._feature_update_alpha) * self._target_feature + \
                      self._feature_update_alpha * detection.feature
        norm = np.linalg.norm(new_feature)
        if norm > 0:
            self._target_feature = new_feature / norm
        else:
            self._target_feature = detection.feature.copy()
    
    def _deduplicate(self, detections: List[PersonDetection],
                     dist_threshold: float = 0.5) -> List[PersonDetection]:
        if len(detections) <= 1:
            return detections
        
        keep = [True] * len(detections)
        
        for i in range(len(detections)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(detections)):
                if not keep[j]:
                    continue
                dist = math.hypot(
                    detections[i].world_x - detections[j].world_x,
                    detections[i].world_y - detections[j].world_y,
                )
                if dist < dist_threshold:
                    if detections[i].confidence >= detections[j].confidence:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
        
        return [d for d, k in zip(detections, keep) if k]
