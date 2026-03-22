"""
=============================================================================
vision_detector.py — 视觉检测与ReID模块
=============================================================================
职责：
1. 用YOLO或其他检测器在相机图像中检测人物
2. 从深度图获取目标距离
3. 将检测结果转换到机器人坐标系/世界坐标系
4. 维护目标人物的外观特征 (ReID)，用于区分跟随对象和其他人

★★★
你需要根据你实际使用的检测模型来修改检测部分。
本文件提供基于 ultralytics YOLO 的参考实现。
如果你使用其他检测器（如MediaPipe、OpenPose等），替换相应部分即可。
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
    """视觉检测器"""
    
    def __init__(self):
        """
        初始化检测模型。
        
        TODO 根据你使用的检测框架修改。
        """
        self._model = None
        self._reid_model = None
        
        # 跟随目标的外观特征模板 (在 lock_target 时设置)
        self._target_feature: Optional[np.ndarray] = None
        
        from ultralytics import YOLO
        self._model = YOLO("yolov8s.pt")  
        
        # ReID模型 (可选，用于多人场景下区分目标) 使用onnet时需要在这里加载模型
        # self._reid_model = 
        # 简单方案: 不用深度学习ReID，直接用颜色直方图作为特征
        # 复杂方案: 用 torchreid 或 OSNet 等轻量级ReID模型
        
        print("[VisionDetector] 初始化完成")
    
    # =====================================================================
    # 核心检测流程
    # =====================================================================
    def detect(self, robot_api: RobotAPI, robot_pose: RobotPose
               ) -> List[PersonDetection]:
        """
        从所有相机中检测人物。
        
        按优先级遍历相机 (头部 → 胸部)，
        合并所有检测结果，去重，返回统一列表。
        
        参数:
            robot_api: 机器人API实例
            robot_pose: 当前机器人位姿
        
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
                # 某个相机可能临时不可用，不影响其他相机
                pass
        
        # 去重: 如果多个相机看到了同一个人 (世界坐标距离 < 0.5m)，保留置信度最高的
        all_detections = self._deduplicate(all_detections)
        
        # 对每个检测做ReID匹配，标记哪个是跟随目标
        self._match_target(all_detections)
        
        return all_detections
    
    def lock_target(self, detection: PersonDetection):
        """
        锁定跟随目标——记录目标的外观特征。
        
        在跟随开始前调用一次，传入你想跟随的那个人的检测结果。
        之后 detect() 会自动标记与该特征最匹配的人为 is_target=True。
        
        参数:
            detection: 要锁定的目标人物检测结果
        """
        if detection.feature is not None:
            self._target_feature = detection.feature.copy()
            print(f"[VisionDetector] 已锁定跟随目标，特征维度: {len(self._target_feature)}")
        else:
            print("[VisionDetector] 警告: 检测结果没有特征向量，无法锁定目标")
    
    def lock_target_from_frame(self, robot_api: RobotAPI, camera_name: str = "head"):
        """
        从当前画面中自动选择最近的人物作为跟随目标并锁定。
        
        这是一个便捷方法，用于初始化时自动锁定正前方最近的人。
        """
        try:
            frame = robot_api.get_camera_frame(camera_name)
            pose = robot_api.get_robot_pose()
            detections = self._detect_in_frame(frame, camera_name, pose)
            
            if not detections:
                print("[VisionDetector] 当前画面中未检测到人物")
                return False
            
            # 选择最近的人
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
        
         这里是实际调用检测模型的地方。
        """
        detections = []
        cam_config = CAMERAS[camera_name]
        timestamp = time.time()
        
        bboxes = self._run_detector(frame.color_image)
        
        for (x1, y1, x2, y2, conf) in bboxes:
            # --- 从深度图获取目标距离 ---
            depth_m = self._get_depth_at_bbox(frame.depth_image, x1, y1, x2, y2)
            if depth_m <= 0 or depth_m > 15.0:
                continue  # 深度无效
            
            # --- 将像素坐标 + 深度 转换为相机坐标系中的3D点 ---
            # 边界框中心的像素坐标
            cx_pixel = (x1 + x2) / 2.0
            cy_pixel = (y1 + y2) / 2.0
            
            # 像素坐标 → 相机坐标系（使用 RealSense 提供的真实内参）
            fx = frame.fx
            fy = frame.fy
            ppx = frame.ppx
            ppy = frame.ppy

            # 相机坐标系: Z朝前, X朝右, Y朝下 (标准相机坐标系)
            cam_z = depth_m  # 前方距离
            cam_x = (cx_pixel - ppx) * depth_m / fx  # 水平偏移
            cam_y = (cy_pixel - ppy) * depth_m / fy  # 垂直偏移 (向下为正)
            
            # --- 相机坐标系 → 机器人坐标系 ---
            # 相机坐标系: X右, Y下, Z前
            # 机器人坐标系: X前, Y左
            cam_yaw_rad = math.radians(cam_config["yaw"])
            cam_pitch_rad = math.radians(cam_config["pitch"])

            # 1. pitch 旋转 (绕相机水平轴): 修正前方距离的垂直投影
            cos_p = math.cos(cam_pitch_rad)
            sin_p = math.sin(cam_pitch_rad)
            cam_z_rot = cam_z * cos_p + cam_y * sin_p

            # 2. yaw 旋转 + 平移
            #    cam_z_rot (相机前方) → robot_x 分量
            #    cam_x (相机右方) → -robot_y 分量 (机器人Y轴朝左)
            cos_y = math.cos(cam_yaw_rad)
            sin_y = math.sin(cam_yaw_rad)
            robot_x = cam_config["x"] + cam_z_rot * cos_y + cam_x * sin_y
            robot_y = cam_config["y"] + cam_z_rot * sin_y - cam_x * cos_y
            
            # --- 机器人坐标系 → 世界坐标系 ---
            cos_t = math.cos(robot_pose.theta)
            sin_t = math.sin(robot_pose.theta)
            world_x = robot_pose.x + robot_x * cos_t - robot_y * sin_t
            world_y = robot_pose.y + robot_x * sin_t + robot_y * cos_t
            
            # --- 提取外观特征 (用于ReID) ---
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
        """
        运行人物检测模型。
        
        参数:
            color_image: BGR图像 (H, W, 3)
        
        返回:
            检测框列表: [(x1, y1, x2, y2, confidence), ...]
        """
        results = self._model(color_image, verbose=False)
        bboxes = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls != 0:  # COCO person class
                    continue
                conf = float(box.conf[0])
                if conf < DETECTION_CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                bboxes.append((x1, y1, x2, y2, conf))
        return bboxes
    
    def _get_depth_at_bbox(self, depth_image: Optional[np.ndarray],
                           x1: int, y1: int, x2: int, y2: int) -> float:
        """
        从深度图中获取目标检测框内的代表性深度值。
        
        策略：取检测框中心区域（中间50%面积）的中位数深度，
        这样可以避免边缘噪声和背景干扰。
        
        返回: 深度 (m)，无效返回 -1
        """
        if depth_image is None:
            return -1.0
        
        h, w = depth_image.shape[:2]
        
        # 检测框中心区域
        cx1 = int(x1 + (x2 - x1) * 0.25)
        cy1 = int(y1 + (y2 - y1) * 0.25)
        cx2 = int(x1 + (x2 - x1) * 0.75)
        cy2 = int(y1 + (y2 - y1) * 0.75)
        
        # 边界检查
        cx1 = max(0, min(cx1, w - 1))
        cx2 = max(0, min(cx2, w - 1))
        cy1 = max(0, min(cy1, h - 1))
        cy2 = max(0, min(cy2, h - 1))
        
        roi = depth_image[cy1:cy2, cx1:cx2]
        
        if roi.size == 0:
            return -1.0
        
        # 取有效深度值的中位数
        valid = roi[roi > 0]
        if len(valid) == 0:
            return -1.0
        
        # 深度图通常单位是毫米 (Intel RealSense)，转为米
        # 如果你的深度图单位不是毫米，请修改
        depth_mm = float(np.median(valid))
        depth_m = depth_mm / 1000.0
        
        return depth_m
    
    def _extract_feature(self, color_image: np.ndarray,
                         x1: int, y1: int, x2: int, y2: int
                         ) -> np.ndarray:
        """
        提取人物外观特征向量，用于ReID。
        
        这里使用简单的颜色直方图方案 (不需要额外模型)：
        - 将检测框内的图像转换到HSV色彩空间
        - 分别计算上半身和下半身的H/S通道直方图
        - 拼接并归一化
        
        如果你需要更强的ReID能力（多人场景、光照变化大等），
        可以替换为深度学习ReID模型 (如 OSNet)。
        """
        try:
            import cv2
        except ImportError:
            # 如果没有cv2，返回随机特征 (仅用于开发调试)
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
        
        # 上半身和下半身分别提取直方图
        upper = hsv[:mid_y]
        lower = hsv[mid_y:]
        
        bins_h, bins_s = 32, 32  # 直方图bins数
        
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
        
        # 归一化到单位向量 (便于余弦相似度计算)
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature /= norm
        
        return feature
    
    def _match_target(self, detections: List[PersonDetection]):
        """
        将检测列表中的人物与已锁定的目标特征做匹配。
        置信度最高且相似度超过阈值的标记为 is_target=True。
        """
        if self._target_feature is None:
            # 未锁定目标，所有人都不是目标
            return
        
        best_idx = -1
        best_score = -1.0
        
        for i, det in enumerate(detections):
            if det.feature is None:
                continue
            # 余弦相似度
            similarity = float(np.dot(self._target_feature, det.feature))
            if similarity > REID_SIMILARITY_THRESHOLD and similarity > best_score:
                best_score = similarity
                best_idx = i
        
        if best_idx >= 0:
            detections[best_idx].is_target = True
    
    def _deduplicate(self, detections: List[PersonDetection],
                     dist_threshold: float = 0.5) -> List[PersonDetection]:
        """
        去重：世界坐标距离小于阈值的检测视为同一个人，保留置信度最高的。
        """
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
                    # 保留置信度高的
                    if detections[i].confidence >= detections[j].confidence:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
        
        return [d for d, k in zip(detections, keep) if k]
