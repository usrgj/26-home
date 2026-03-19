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

状态向量: [x, y, vx, vy]
- x, y: 目标在世界坐标系中的位置 (m)
- vx, vy: 目标在世界坐标系中的速度 (m/s)

观测向量: [x, y]
- 来自视觉或LiDAR的位置观测 (世界坐标系)

运动模型: 匀速运动 (Constant Velocity)
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
    x: float = 0.0           # 世界坐标X (m)
    y: float = 0.0           # 世界坐标Y (m)
    vx: float = 0.0          # 世界坐标系速度X (m/s)
    vy: float = 0.0          # 世界坐标系速度Y (m/s)
    
    speed: float = 0.0       # 速度标量 (m/s)
    heading: float = 0.0     # 运动方向 (rad)
    
    is_valid: bool = False   # 估计是否有效
    is_coasting: bool = False  # 是否在无观测预测 (coasting)
    
    last_vision_time: float = 0.0   # 上次视觉观测时间
    last_lidar_time: float = 0.0    # 上次LiDAR观测时间
    last_any_time: float = 0.0      # 上次任意观测时间
    
    # 预测位置 (根据当前速度外推一小段时间后的位置)
    predicted_x: float = 0.0
    predicted_y: float = 0.0


class SensorFusion:
    """EKF传感器融合跟踪器"""
    
    def __init__(self):
        # 状态向量: [x, y, vx, vy]
        self._state = np.zeros(4)
        
        # 协方差矩阵 (初始不确定度较大)
        self._P = np.eye(4) * 10.0
        
        # 上次更新时间
        self._last_time: float = 0.0
        
        # 是否已初始化
        self._initialized = False
        
        # 连续无观测时间计数
        self._coast_time: float = 0.0
        
        # 上次各传感器观测时间
        self._last_vision_time: float = 0.0
        self._last_lidar_time: float = 0.0
    
    # =====================================================================
    # 外部接口
    # =====================================================================
    def initialize(self, x: float, y: float, timestamp: float):
        """
        用第一个观测初始化跟踪器。
        
        参数:
            x, y: 目标初始世界坐标
            timestamp: 时间戳 (s)
        """
        self._state = np.array([x, y, 0.0, 0.0])
        self._P = np.diag([0.5, 0.5, 1.0, 1.0])  # 位置较确定，速度不确定
        self._last_time = timestamp
        self._initialized = True
        self._coast_time = 0.0
        self._last_vision_time = timestamp
        self._last_lidar_time = timestamp
        print(f"[SensorFusion] 初始化: ({x:.2f}, {y:.2f})")
    
    def update_with_vision(self, detection: PersonDetection):
        """
        用视觉检测结果更新状态估计。
        
        参数:
            detection: 标记为 is_target=True 的视觉检测结果
        """
        if not self._initialized:
            self.initialize(detection.world_x, detection.world_y, detection.timestamp)
            return
        
        # 先做预测
        dt = detection.timestamp - self._last_time
        if dt > 0:
            self._predict(dt)
        
        # 再做更新
        z = np.array([detection.world_x, detection.world_y])
        R = np.eye(2) * EKF_MEASUREMENT_NOISE_VISION  # 视觉观测噪声
        self._update(z, R)
        
        self._last_time = detection.timestamp
        self._last_vision_time = detection.timestamp
        self._coast_time = 0.0
    
    def update_with_lidar(self, candidate: PersonCandidate):
        """
        用LiDAR人物候选更新状态估计。
        
        参数:
            candidate: 与跟踪目标关联成功的LiDAR人物候选
        """
        if not self._initialized:
            self.initialize(candidate.world_x, candidate.world_y, candidate.timestamp)
            return
        
        dt = candidate.timestamp - self._last_time
        if dt > 0:
            self._predict(dt)
        
        z = np.array([candidate.world_x, candidate.world_y])
        # LiDAR观测噪声比视觉小 (更精确)
        R = np.eye(2) * EKF_MEASUREMENT_NOISE_LIDAR
        self._update(z, R)
        
        self._last_time = candidate.timestamp
        self._last_lidar_time = candidate.timestamp
        self._coast_time = 0.0
    
    def predict_only(self, timestamp: float):
        """
        仅做预测（无观测时调用），用匀速模型外推位置。
        """
        if not self._initialized:
            return
        
        dt = timestamp - self._last_time
        if dt > 0:
            self._predict(dt)
            self._last_time = timestamp
            self._coast_time += dt
    
    def get_target_state(self, prediction_horizon: float = 0.5) -> TargetState:
        """
        获取当前目标状态估计。
        
        参数:
            prediction_horizon: 预测时间跨度 (s)，用于计算预测位置
        
        返回:
            TargetState 数据结构
        """
        if not self._initialized:
            return TargetState(is_valid=False)
        
        x, y, vx, vy = self._state
        speed = math.hypot(vx, vy)
        heading = math.atan2(vy, vx) if speed > 0.05 else 0.0
        
        # 判断是否仍然有效
        is_valid = self._coast_time < EKF_MAX_COAST_TIME
        is_coasting = self._coast_time > 0.3  # 0.3秒无观测算coasting
        
        # 预测位置 (匀速外推)
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
        )
    
    def associate_lidar_candidates(self, candidates: List[PersonCandidate],
                                    gate_distance: float = 1.5
                                    ) -> Optional[PersonCandidate]:
        """
        将LiDAR检测到的人物候选与当前跟踪目标做数据关联。
        
        使用马氏距离 (Mahalanobis distance) 门控：
        - 在跟踪状态的不确定度范围内找最近的候选
        - 距离超过 gate_distance 的候选被拒绝
        
        如果同时有视觉确认 (is_target=True)，优先信任视觉的关联结果，
        然后在LiDAR候选中找与视觉目标位置最近的那个。
        
        参数:
            candidates: LiDAR检测到的所有人物候选
            gate_distance: 关联门控距离 (m)
        
        返回:
            关联成功的候选，None表示没有匹配的
        """
        if not self._initialized or len(candidates) == 0:
            return None
        
        x, y = self._state[0], self._state[1]
        
        best_candidate = None
        best_dist = float('inf')
        
        for cand in candidates:
            dist = math.hypot(cand.world_x - x, cand.world_y - y)
            if dist < gate_distance and dist < best_dist:
                best_dist = dist
                best_candidate = cand
        
        return best_candidate
    
    # =====================================================================
    # EKF 内部实现
    # =====================================================================
    def _predict(self, dt: float):
        """
        EKF预测步骤 (匀速运动模型)。
        
        状态转移:
            x(k+1)  = x(k)  + vx(k) * dt
            y(k+1)  = y(k)  + vy(k) * dt
            vx(k+1) = vx(k)
            vy(k+1) = vy(k)
        
        状态转移矩阵 F:
            [1 0 dt 0 ]
            [0 1 0  dt]
            [0 0 1  0 ]
            [0 0 0  1 ]
        """
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])
        
        # 过程噪声矩阵 Q (连续白噪声加速度模型)
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
        
        # 状态预测
        self._state = F @ self._state
        
        # 协方差预测
        self._P = F @ self._P @ F.T + Q
    
    def _update(self, z: np.ndarray, R: np.ndarray):
        """
        EKF更新步骤 (观测模型为线性: z = H * x)。
        
        观测矩阵 H:
            [1 0 0 0]
            [0 1 0 0]
        
        即观测的是位置 (x, y)，不直接观测速度。
        
        参数:
            z: 观测向量 [x, y]
            R: 观测噪声协方差矩阵 (2x2)
        """
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        
        # 创新 (innovation): y = z - H * x_pred
        y = z - H @ self._state
        
        # 创新协方差: S = H * P * H^T + R
        S = H @ self._P @ H.T + R
        
        # 卡尔曼增益: K = P * H^T * S^(-1)
        K = self._P @ H.T @ np.linalg.inv(S)
        
        # 状态更新: x = x + K * y
        self._state = self._state + K @ y
        
        # 协方差更新: P = (I - K * H) * P
        I = np.eye(4)
        self._P = (I - K @ H) @ self._P
    
    def reset(self):
        """重置跟踪器"""
        self._state = np.zeros(4)
        self._P = np.eye(4) * 10.0
        self._last_time = 0.0
        self._initialized = False
        self._coast_time = 0.0
        print("[SensorFusion] 跟踪器已重置")
