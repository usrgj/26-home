"""
=============================================================================
sensor_fusion.py — 扩展卡尔曼滤波 (EKF) 传感器融合跟踪器
=============================================================================
职责：
1. 维护跟随目标的状态估计: [x, y, vx, vy] (世界坐标系)
2. 融合来自视觉和LiDAR的观测
3. 在无观测时做预测 (匀速模型)
4. 关联LiDAR检测到的人物候选与视觉检测到的目标
5. 输出目标的世界坐标、速度、预测位置
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
        
        # 协方差矩阵 (初始不确定度较大)
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
            self._predict(dt)
        
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
            self._predict(dt)
        
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
            self._predict(dt)
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
        
        return TargetState(
            x=x, y=y, vx=vx, vy=vy,
            speed=speed, heading=heading,
            is_valid=is_valid, is_coasting=is_coasting,
            last_vision_time=self._last_vision_time,
            last_lidar_time=self._last_lidar_time,
            last_any_time=max(self._last_vision_time, self._last_lidar_time),
            predicted_x=pred_x, predicted_y=pred_y,
            speed_variance=self.get_speed_variance(),
        )
    
    def set_vision_anchor(self, world_x: float, world_y: float):
        self._vision_anchor = (world_x, world_y)
        self._vision_anchor_time = time.time()
    
    def associate_lidar_candidates(self, candidates: List[PersonCandidate],
                                    gate_distance: float = 0.8
                                    ) -> Optional[PersonCandidate]:
        if not self._initialized or len(candidates) == 0:
            return None
        
        now = time.time()

        anchor = self._vision_anchor
        anchor_time = self._vision_anchor_time
        if anchor is not None and (now - anchor_time) < 1.0:
            cx, cy = anchor
        else:
            cx, cy = self._state[0], self._state[1]

        effective_gate = gate_distance
        if self._coast_time > 0.3:
            effective_gate = min(gate_distance + self._coast_time * 0.3, 1.5)
        
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
        self._vision_anchor_time = 0.0
        print("[SensorFusion] 跟踪器已重置")
    
    # =====================================================================
    # EKF 内部实现
    # =====================================================================
    def _predict(self, dt: float):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])
        
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        
        q_pos = EKF_PROCESS_NOISE_POS
        q_vel = EKF_PROCESS_NOISE_VEL
        
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
