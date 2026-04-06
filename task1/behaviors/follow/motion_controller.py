"""
=============================================================================
motion_controller.py — 运动控制模块
=============================================================================
职责：
1. 直接跟随控制: PID控制器，保持与目标的距离和朝向
2. 反应式避障: VFH (Vector Field Histogram) 向量场直方图
3. 速度调节: 根据障碍物距离和目标状态动态调速
4. 运动指令生成: 输出线速度和角速度

控制策略:
- 计算到目标的"期望方向"和"期望速度"
- 用VFH在期望方向附近找到"最优安全方向"
- PID控制角速度使机器人朝向最优方向
- PID控制线速度使距离趋近于 FOLLOW_DISTANCE
"""
import math
import time
import numpy as np
from typing import Optional, Tuple

from .config import (
    ROBOT_MAX_LINEAR_VEL, ROBOT_MAX_ANGULAR_VEL, ROBOT_RADIUS,
    MAX_LINEAR_ACCEL, MAX_ANGULAR_ACCEL, MAIN_LOOP_RATE,
    FOLLOW_DISTANCE, FOLLOW_DISTANCE_TOLERANCE, FOLLOW_ANGLE_TOLERANCE,
    PID_LINEAR_KP, PID_LINEAR_KI, PID_LINEAR_KD,
    PID_ANGULAR_KP, PID_ANGULAR_KI, PID_ANGULAR_KD,
    OBSTACLE_DANGER_DIST, OBSTACLE_SLOW_DIST, OBSTACLE_AVOID_DIST,
    VFH_SECTOR_ANGLE, VFH_THRESHOLD, LIDAR_MAX_RANGE,
)
from .robot_api import RobotAPI, RobotPose
from .sensor_fusion import TargetState


class PIDController:
    """简单的PID控制器"""
    
    def __init__(self, kp: float, ki: float, kd: float,
                 output_min: float = -float('inf'),
                 output_max: float = float('inf')):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = 0.0
        self._first = True
    
    def compute(self, error: float, current_time: float) -> float:
        """
        计算PID输出。
        
        参数:
            error: 当前误差 (期望值 - 当前值)
            current_time: 当前时间戳 (s)
        
        返回:
            控制量
        """
        if self._first:
            self._prev_error = error
            self._prev_time = current_time
            self._first = False
            return self.kp * error
        
        dt = current_time - self._prev_time
        if dt <= 0:
            dt = 0.001
        
        # 比例项
        p_out = self.kp * error
        
        # 积分项 (带anti-windup)
        self._integral += error * dt
        self._integral = np.clip(self._integral, -10.0, 10.0)
        i_out = self.ki * self._integral
        
        # 微分项
        d_out = self.kd * (error - self._prev_error) / dt
        
        self._prev_error = error
        self._prev_time = current_time
        
        output = p_out + i_out + d_out
        return np.clip(output, self.output_min, self.output_max)
    
    def reset(self):
        """重置PID状态"""
        self._integral = 0.0
        self._prev_error = 0.0
        self._first = True


class MotionController:
    """运动控制器"""
    
    def __init__(self, robot_api: RobotAPI):
        self._robot_api = robot_api

        # PID控制器: 线速度 (控制跟随距离)
        self._linear_pid = PIDController(
            kp=PID_LINEAR_KP, ki=PID_LINEAR_KI, kd=PID_LINEAR_KD,
            output_min=-ROBOT_MAX_LINEAR_VEL,
            output_max=ROBOT_MAX_LINEAR_VEL,
        )

        # PID控制器: 角速度 (控制朝向)
        self._angular_pid = PIDController(
            kp=PID_ANGULAR_KP, ki=PID_ANGULAR_KI, kd=PID_ANGULAR_KD,
            output_min=-ROBOT_MAX_ANGULAR_VEL,
            output_max=ROBOT_MAX_ANGULAR_VEL,
        )

        # 上一帧速度（用于斜坡限制）
        self._last_linear_vel = 0.0
        self._last_angular_vel = 0.0
    
    # =====================================================================
    # 主控制接口
    # =====================================================================
    def compute_velocity(self, target: TargetState, robot_pose: RobotPose,
                         obstacle_sectors: np.ndarray) -> Tuple[float, float]:
        """
        计算跟随目标所需的运动指令。
        
        这是运动控制的核心方法，整合了跟随控制和避障。
        
        参数:
            target: 目标状态 (来自EKF)
            robot_pose: 当前机器人位姿
            obstacle_sectors: 每个扇区最近障碍物距离 (来自LiDAR处理器)
        
        返回:
            (linear_vel, angular_vel) 运动指令
        """
        now = time.time()
        
        # --- Step 1: 计算到目标的距离和方向 (机器人坐标系) ---
        dx = target.x - robot_pose.x
        dy = target.y - robot_pose.y
        dist_to_target = math.hypot(dx, dy)
        
        # 目标相对于机器人的方位角 (世界坐标系)
        angle_to_target_world = math.atan2(dy, dx)
        
        # 转换为机器人坐标系中的相对角度
        angle_error = self._normalize_angle(angle_to_target_world - robot_pose.theta)
        
        # --- Step 2: 计算期望方向 (考虑目标运动预测) ---
        # 如果目标在移动，瞄准预测位置而非当前位置
        if target.speed > 0.1:
            pred_dx = target.predicted_x - robot_pose.x
            pred_dy = target.predicted_y - robot_pose.y
            desired_angle_world = math.atan2(pred_dy, pred_dx)
            desired_angle = self._normalize_angle(desired_angle_world - robot_pose.theta)
        else:
            desired_angle = angle_error
        
        # --- Step 3: VFH避障 — 在期望方向附近找安全方向 ---
        safe_angle = self._vfh_find_safe_direction(
            desired_angle, obstacle_sectors
        )
        
        # --- Step 4: 检查前方是否有紧急障碍物 ---
        emergency_stop, speed_factor = self._check_obstacles(
            safe_angle, obstacle_sectors
        )
        
        if emergency_stop:
            self._robot_api.stop()
            return 0.0, 0.0
        
        # --- Step 5: PID计算线速度 ---
        # 距离误差 = 当前距离 - 期望跟随距离 (正值表示太远，需要前进)
        dist_error = dist_to_target - FOLLOW_DISTANCE

        # 距离死区：误差在容差范围内时不动
        if abs(dist_error) < FOLLOW_DISTANCE_TOLERANCE:
            linear_vel = 0.0
        else:
            linear_vel = self._linear_pid.compute(dist_error, now)

        # 如果目标在后方 (|angle_error| > 90°)，先转身再前进
        if abs(angle_error) > math.pi / 2:
            linear_vel = 0.0  # 先不前进，只转身

        # 距离太近时后退
        if dist_to_target < FOLLOW_DISTANCE - FOLLOW_DISTANCE_TOLERANCE:
            linear_vel = min(linear_vel, -0.1)

        # 应用速度因子 (根据障碍物距离减速)
        linear_vel *= speed_factor

        # 限幅
        linear_vel = np.clip(linear_vel, -ROBOT_MAX_LINEAR_VEL, ROBOT_MAX_LINEAR_VEL)

        # --- Step 6: PID计算角速度 ---
        # 角度死区：角度偏差在容差范围内时不转
        if abs(safe_angle) < FOLLOW_ANGLE_TOLERANCE:
            angular_vel = 0.0
        else:
            angular_vel = self._angular_pid.compute(safe_angle, now)
        angular_vel = np.clip(angular_vel, -ROBOT_MAX_ANGULAR_VEL, ROBOT_MAX_ANGULAR_VEL)
        
        # --- Step 7: 高角速度时降低线速度 (差速底盘稳定性) ---
        angular_ratio = abs(angular_vel) / ROBOT_MAX_ANGULAR_VEL
        if angular_ratio > 0.5:
            linear_vel *= (1.0 - 0.5 * angular_ratio)

        # --- Step 8: 斜坡限制 (slew rate limiter) ---
        # 限制相邻周期速度变化量，使运动更平滑
        max_linear_delta = MAX_LINEAR_ACCEL / MAIN_LOOP_RATE
        max_angular_delta = MAX_ANGULAR_ACCEL / MAIN_LOOP_RATE

        linear_vel = np.clip(
            linear_vel,
            self._last_linear_vel - max_linear_delta,
            self._last_linear_vel + max_linear_delta,
        )
        angular_vel = np.clip(
            angular_vel,
            self._last_angular_vel - max_angular_delta,
            self._last_angular_vel + max_angular_delta,
        )

        self._last_linear_vel = linear_vel
        self._last_angular_vel = angular_vel

        return linear_vel, angular_vel
    
    def rotate_search(self, direction: float = 1.0) -> Tuple[float, float]:
        """
        搜索模式: 原地旋转寻找目标。
        
        参数:
            direction: 1.0 左转, -1.0 右转
        
        返回:
            (linear_vel, angular_vel) — 原地旋转指令
        """
        from .config import SEARCH_ROTATION_SPEED
        return 0.0, direction * SEARCH_ROTATION_SPEED
    
    def stop(self):
        """停止运动"""
        self._robot_api.stop()
        self._linear_pid.reset()
        self._angular_pid.reset()
        self._last_linear_vel = 0.0
        self._last_angular_vel = 0.0
    
    # =====================================================================
    # VFH 向量场直方图避障
    # =====================================================================
    def _vfh_find_safe_direction(self, desired_angle: float,
                                  obstacle_sectors: np.ndarray
                                  ) -> float:
        """
        VFH (Vector Field Histogram) 避障算法。
        
        原理：
        1. 将360°范围划分为多个扇区，每个扇区有最近障碍物距离
        2. 将障碍物距离转换为"障碍密度" (越近密度越高)
        3. 障碍密度低于阈值的扇区为"开放扇区"
        4. 在所有开放扇区中，选择与期望方向最近的扇区
        
        参数:
            desired_angle: 期望运动方向 (rad, 机器人坐标系, 0=正前方)
            obstacle_sectors: 每个扇区最近障碍物距离
        
        返回:
            安全运动方向 (rad, 机器人坐标系)
        """
        num_sectors = len(obstacle_sectors)
        sector_size_rad = 2 * math.pi / num_sectors
        
        # 计算每个扇区的障碍密度
        # 密度 = max(0, 1 - dist / OBSTACLE_AVOID_DIST)
        densities = np.maximum(0, 1.0 - obstacle_sectors / OBSTACLE_AVOID_DIST)
        
        # 找到所有"开放"扇区 (密度 < 阈值)
        open_sectors = densities < VFH_THRESHOLD
        
        if not np.any(open_sectors):
            # 没有安全方向，返回期望方向 (紧急情况由其他逻辑处理)
            return desired_angle
        
        # 在开放扇区中找与期望方向最近的
        desired_sector = self._angle_to_sector(desired_angle, num_sectors)
        
        best_sector = -1
        best_cost = float('inf')
        
        for i in range(num_sectors):
            if not open_sectors[i]:
                continue
            
            # 计算该扇区与期望扇区的角度差 (考虑环绕)
            sector_diff = min(
                abs(i - desired_sector),
                num_sectors - abs(i - desired_sector)
            )
            
            # 代价 = 角度差 (偏离期望方向的代价)
            cost = sector_diff
            
            if cost < best_cost:
                best_cost = cost
                best_sector = i
        
        if best_sector < 0:
            return desired_angle
        
        # 将扇区索引转回角度
        safe_angle = self._sector_to_angle(best_sector, num_sectors)
        
        return safe_angle
    
    def _check_obstacles(self, heading_angle: float,
                          obstacle_sectors: np.ndarray
                          ) -> Tuple[bool, float]:
        """
        检查运动方向前方的障碍物，决定是否紧急停止或减速。
        
        参数:
            heading_angle: 即将运动的方向 (rad, 机器人坐标系)
            obstacle_sectors: 扇区障碍物距离
        
        返回:
            (emergency_stop, speed_factor)
            - emergency_stop: True则立即停止
            - speed_factor: 0.0~1.0 速度缩放因子
        """
        num_sectors = len(obstacle_sectors)
        center_sector = self._angle_to_sector(heading_angle, num_sectors)
        
        # 检查运动方向前方 ±30° 范围内的障碍物
        check_range = int(30.0 / (360.0 / num_sectors))
        
        min_dist = LIDAR_MAX_RANGE
        for offset in range(-check_range, check_range + 1):
            idx = (center_sector + offset) % num_sectors
            if obstacle_sectors[idx] < min_dist:
                min_dist = obstacle_sectors[idx]
        
        # 减去机器人半径得到实际净空距离
        clearance = min_dist - ROBOT_RADIUS
        
        # 紧急停止
        if clearance < OBSTACLE_DANGER_DIST:
            return True, 0.0
        
        # 减速
        if clearance < OBSTACLE_SLOW_DIST:
            factor = (clearance - OBSTACLE_DANGER_DIST) / (OBSTACLE_SLOW_DIST - OBSTACLE_DANGER_DIST)
            factor = np.clip(factor, 0.1, 1.0)
            return False, factor
        
        return False, 1.0
    
    # =====================================================================
    # 工具方法
    # =====================================================================
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """将角度归一化到 [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    @staticmethod
    def _angle_to_sector(angle_rad: float, num_sectors: int) -> int:
        """
        将机器人坐标系中的角度 (rad) 转换为扇区索引。
        扇区0 = 正前方 (0°), 逆时针递增。
        """
        angle_deg = math.degrees(angle_rad) % 360
        sector_size = 360.0 / num_sectors
        return int(angle_deg / sector_size) % num_sectors
    
    @staticmethod
    def _sector_to_angle(sector: int, num_sectors: int) -> float:
        """将扇区索引转换为角度 (rad, -pi ~ pi)"""
        sector_size = 360.0 / num_sectors
        angle_deg = sector * sector_size + sector_size / 2.0
        if angle_deg > 180:
            angle_deg -= 360
        return math.radians(angle_deg)
