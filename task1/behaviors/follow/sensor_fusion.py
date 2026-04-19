"""
=============================================================================
sensor_fusion.py — 扩展卡尔曼滤波 (EKF) 传感器融合跟踪器（改进版）
=============================================================================
新增功能：
1. 暴露速度方差供运动控制器使用
2. 改进 LiDAR 关联时的中心计算（融合视觉锚点与 EKF 预测）
3. 增加运动模型适应性（静止时降低过程噪声）
"""
import math
import time
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .config import (
    EKF_PROCESS_NOISE_POS, EKF_PROCESS_NOISE_VEL,
    EKF_MEASUREMENT_NOISE_LIDAR, EKF_MEASUREMENT_NOISE_VISION,
    EKF_MAX_COAST_TIME,
)
from .lidar_processor import PersonCandidate
from .vision_detector import PersonDetection


@dataclass
class TargetState:
    """跟踪目标的状态"""
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    
    speed: float = 0.0
    heading: float = 0.0
    
    is_valid: bool = False
    is_coasting: bool = False
    
    last_vision_time: float = 0.0
    last_lidar_time: float = 0.0
    last_any_time: float = 0.0
    
    predicted_x: float = 0.0
    predicted_y: float = 0.0
    
    # ----- 新增：速度方差（由 fusion 填充）-----
    speed_variance: float = 1.0


class SensorFusion:
    """EKF传感器融合跟踪器"""
    
    def __init__(self):
        # 状态向量: [x, y, vx, vy]
        self._state = np.zeros(4)
        
        # 协方差矩阵
        self._P = np.eye(4) * 10.0
        
        self._last_time: float = 0.0
        self._initialized = False
        self._coast_time: float = 0.0
        
        self._last_vision_time: float = 0.0
        self._last_lidar_time: float = 0.0
        
        # 视觉锚点（用于 LiDAR 关联）
        self._vision_anchor: Optional[Tuple[float, float]] = None
        self._vision_anchor_time: float = 0.0
    
    # =====================================================================
    # 外部接口
    # =====================================================================
    def initialize(self, x: float, y: float, timestamp: float):
        self._state = np.array([x, y, 0.0, 0.0])
        self._P = np.diag([0.5, 0.5, 1.0, 1.0])
        self._last_time = timestamp
        self._initialized = True
        self._coast_time = 0.0
        self._last_vision_time = timestamp
        self._last_lidar_time = timestamp
        print(f"[SensorFusion] 初始化: ({x:.2f}, {y:.2f})")
    
    def update_with_vision(self, detection: PersonDetection):
        if not self._initialized:
            self.initialize(detection.world_x, detection.world_y, detection.timestamp)
            return
        
        dt = detection.timestamp - self._last_time
        if dt > 0:
            # 根据目标速度调整过程噪声（静止时降低）
            process_noise_scale = self._compute_process_noise_scale()
            self._predict(dt, process_noise_scale)
        
        z = np.array([detection.world_x, detection.world_y])
        R = np.eye(2) * EKF_MEASUREMENT_NOISE_VISION
        self._update(z, R)
        
        self._last_time = detection.timestamp
        self._last_vision_time = detection.timestamp
        self._coast_time = 0.0
    
    def update_with_lidar(self, candidate: PersonCandidate):
        if not self._initialized:
            self.initialize(candidate.world_x, candidate.world_y, candidate.timestamp)
            return
        
        dt = candidate.timestamp - self._last_time
        if dt > 0:
            process_noise_scale = self._compute_process_noise_scale()
            self._predict(dt, process_noise_scale)
        
        z = np.array([candidate.world_x, candidate.world_y])
        R = np.eye(2) * EKF_MEASUREMENT_NOISE_LIDAR
        self._update(z, R)
        
        self._last_time = candidate.timestamp
        self._last_lidar_time = candidate.timestamp
        self._coast_time = 0.0
    
    def predict_only(self, timestamp: float):
        if not self._initialized:
            return
        
        dt = timestamp - self._last_time
        if dt > 0:
            process_noise_scale = self._compute_process_noise_scale()
            self._predict(dt, process_noise_scale)
            self._last_time = timestamp
            self._coast_time += dt
    
    def get_target_state(self, prediction_horizon: float = 0.5) -> TargetState:
        if not self._initialized:
            return TargetState(is_valid=False)
        
        x, y, vx, vy = self._state
        speed = math.hypot(vx, vy)
        heading = math.atan2(vy, vx) if speed > 0.05 else 0.0
        
        is_valid = self._coast_time < EKF_MAX_COAST_TIME
        is_coasting = self._coast_time > 0.3
        
        pred_x = x + vx * prediction_horizon
        pred_y = y + vy * prediction_horizon
        
        # 计算速度方差
        speed_variance = self.get_speed_variance()
        
        return TargetState(
            x=x, y=y, vx=vx, vy=vy,
            speed=speed, heading=heading,
            is_valid=is_valid, is_coasting=is_coasting,
            last_vision_time=self._last_vision_time,
            last_lidar_time=self._last_lidar_time,
            last_any_time=max(self._last_vision_time, self._last_lidar_time),
            predicted_x=pred_x, predicted_y=pred_y,
            speed_variance=speed_variance,
        )
    
    def set_vision_anchor(self, world_x: float, world_y: float):
        self._vision_anchor = (world_x, world_y)
        self._vision_anchor_time = time.time()
    
    def associate_lidar_candidates(self, candidates: List[PersonCandidate],
                                    gate_distance: float = 0.8
                                    ) -> Optional[PersonCandidate]:
        """
        改进的 LiDAR 关联：
        - 优先使用新鲜视觉锚点（<0.5s）
        - 若锚点与 EKF 预测接近，取两者平均
        - 否则以 EKF 预测为中心（若 EKF 有效），或锚点（若无 EKF）
        """
        if not self._initialized or len(candidates) == 0:
            return None
        
        now = time.time()
        
        # 确定关联中心
        anchor = self._vision_anchor
        anchor_time = self._vision_anchor_time
        
        ekf_x, ekf_y = self._state[0], self._state[1]
        ekf_valid = self._initialized
        
        # 情况 1：有新鲜锚点 (<0.5s) 且 EKF 有效
        if anchor is not None and (now - anchor_time) < 0.5 and ekf_valid:
            dist_anchor_ekf = math.hypot(anchor[0] - ekf_x, anchor[1] - ekf_y)
            if dist_anchor_ekf < 0.6:
                # 锚点与 EKF 一致，取平均
                cx = (anchor[0] + ekf_x) / 2
                cy = (anchor[1] + ekf_y) / 2
            else:
                # 差异大，优先相信新鲜锚点
                cx, cy = anchor
        elif anchor is not None and (now - anchor_time) < 1.0:
            # 锚点较新（<1s），使用锚点
            cx, cy = anchor
        elif ekf_valid:
            # 无锚点或锚点过旧，使用 EKF 位置
            cx, cy = ekf_x, ekf_y
        else:
            # 都没有，无法关联
            return None
        
        # 动态门控：coasting 时放宽
        effective_gate = gate_distance
        if self._coast_time > 0.5:
            effective_gate = min(gate_distance + self._coast_time * 0.3, 1.8)
        
        best_candidate = None
        best_dist = float('inf')
        
        for cand in candidates:
            dist = math.hypot(cand.world_x - cx, cand.world_y - cy)
            if dist < effective_gate and dist < best_dist:
                best_dist = dist
                best_candidate = cand
        
        return best_candidate
    
    def get_speed_variance(self) -> float:
        """返回速度估计的方差（取 vx, vy 方差的最大值）"""
        if not self._initialized:
            return 1.0
        var_vx = self._P[2, 2]
        var_vy = self._P[3, 3]
        return float(max(var_vx, var_vy))
    
    def reset(self):
        self._state = np.zeros(4)
        self._P = np.eye(4) * 10.0
        self._last_time = 0.0
        self._initialized = False
        self._coast_time = 0.0
        self._vision_anchor = None
        print("[SensorFusion] 跟踪器已重置")
    
    # =====================================================================
    # EKF 内部实现
    # =====================================================================
    def _compute_process_noise_scale(self) -> float:
        """
        根据当前估计速度调整过程噪声缩放因子。
        目标静止时降低过程噪声，避免漂移。
        """
        if not self._initialized:
            return 1.0
        speed = math.hypot(self._state[2], self._state[3])
        if speed < 0.1:
            return 0.3   # 静止时降低噪声
        elif speed < 0.5:
            return 0.7
        else:
            return 1.0
    
    def _predict(self, dt: float, noise_scale: float = 1.0):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])
        
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        
        q_pos = EKF_PROCESS_NOISE_POS * noise_scale
        q_vel = EKF_PROCESS_NOISE_VEL * noise_scale
        
        Q = np.array([
            [dt4 * q_pos, 0,           dt3 * q_pos, 0          ],
            [0,           dt4 * q_pos, 0,           dt3 * q_pos],
            [dt3 * q_pos, 0,           dt2 * q_vel, 0          ],
            [0,           dt3 * q_pos, 0,           dt2 * q_vel],
        ])
        
        self._state = F @ self._state
        self._P = F @ self._P @ F.T + Q
    
    def _update(self, z: np.ndarray, R: np.ndarray):
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        
        y = z - H @ self._state
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        
        self._state = self._state + K @ y
        
        I = np.eye(4)
        self._P = (I - K @ H) @ self._P
