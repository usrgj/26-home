"""
=============================================================================
motion_controller.py — 运动控制模块
=============================================================================
职责：
1. 直接跟随控制: PID控制器，保持与目标的距离和朝向
2. 反应式避障: VFH (Vector Field Histogram) 向量场直方图
3. 速度调节: 根据障碍物距离和目标状态动态调速
4. 运动指令生成: 输出线速度和角速度
"""

import math
import time
import numpy as np
from typing import Tuple

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
        """基于当前误差计算一次控制输出。"""
        if self._first:
            self._prev_error = error
            self._prev_time = current_time
            self._first = False
            return self.kp * error

        dt = current_time - self._prev_time
        if dt <= 0:
            dt = 0.001

        p_out = self.kp * error

        self._integral += error * dt
        self._integral = np.clip(self._integral, -10.0, 10.0)
        i_out = self.ki * self._integral

        d_out = self.kd * (error - self._prev_error) / dt

        self._prev_error = error
        self._prev_time = current_time

        output = p_out + i_out + d_out
        return np.clip(output, self.output_min, self.output_max)

    def reset(self):
        """清空积分项和历史状态，避免模式切换后残留控制量。"""
        self._integral = 0.0
        self._prev_error = 0.0
        self._first = True


class MotionController:
    """将目标状态和避障信息转换成底盘速度指令。"""
    def __init__(self, robot_api: RobotAPI):
        self._robot_api = robot_api
        self._linear_pid = PIDController(
            kp=PID_LINEAR_KP, ki=PID_LINEAR_KI, kd=PID_LINEAR_KD,
            output_min=-ROBOT_MAX_LINEAR_VEL,
            output_max=ROBOT_MAX_LINEAR_VEL,
        )
        self._angular_pid = PIDController(
            kp=PID_ANGULAR_KP, ki=PID_ANGULAR_KI, kd=PID_ANGULAR_KD,
            output_min=-ROBOT_MAX_ANGULAR_VEL,
            output_max=ROBOT_MAX_ANGULAR_VEL,
        )
        self._last_linear_vel = 0.0
        self._last_angular_vel = 0.0

    def compute_velocity(self, target: TargetState, robot_pose: RobotPose,
                         obstacle_sectors: np.ndarray) -> Tuple[float, float]:
        """
        计算直接跟随模式下的线速度和角速度。

        流程是：先根据目标位置/预测位置生成期望方向，再用 VFH 调整成
        安全方向，最后叠加 PID、限速和加速度限幅。
        """
        now = time.time()
        dx = target.x - robot_pose.x
        dy = target.y - robot_pose.y
        dist_to_target = math.hypot(dx, dy)
        angle_to_target_world = math.atan2(dy, dx)
        angle_error = self._normalize_angle(angle_to_target_world - robot_pose.theta)

        if target.speed > 0.1:
            # 目标在移动时优先追预测点，减少拐弯和滞后。
            pred_dx = target.predicted_x - robot_pose.x
            pred_dy = target.predicted_y - robot_pose.y
            desired_angle_world = math.atan2(pred_dy, pred_dx)
            desired_angle = self._normalize_angle(desired_angle_world - robot_pose.theta)
        else:
            desired_angle = angle_error

        safe_angle = self._vfh_find_safe_direction(desired_angle, obstacle_sectors)

        emergency_stop, speed_factor = self._check_obstacles(safe_angle, obstacle_sectors)
        if emergency_stop:
            self._robot_api.stop()
            return 0.0, 0.0

        dist_error = dist_to_target - FOLLOW_DISTANCE
        if abs(dist_error) < FOLLOW_DISTANCE_TOLERANCE:
            linear_vel = 0.0
        else:
            linear_vel = self._linear_pid.compute(dist_error, now)

        if abs(angle_error) > math.pi / 2:
            linear_vel = 0.0

        if dist_to_target < FOLLOW_DISTANCE - FOLLOW_DISTANCE_TOLERANCE:
            linear_vel = min(linear_vel, -0.1)

        linear_vel *= speed_factor
        linear_vel = np.clip(linear_vel, -ROBOT_MAX_LINEAR_VEL, ROBOT_MAX_LINEAR_VEL)

        if abs(safe_angle) < FOLLOW_ANGLE_TOLERANCE:
            angular_vel = 0.0
        else:
            angular_vel = self._angular_pid.compute(safe_angle, now)
        angular_vel = np.clip(angular_vel, -ROBOT_MAX_ANGULAR_VEL, ROBOT_MAX_ANGULAR_VEL)

        angular_ratio = abs(angular_vel) / ROBOT_MAX_ANGULAR_VEL
        if angular_ratio > 0.5:
            linear_vel *= (1.0 - 0.5 * angular_ratio)

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
        """搜索模式下原地匀速旋转，不前进。"""
        from .config import SEARCH_ROTATION_SPEED
        return 0.0, direction * SEARCH_ROTATION_SPEED

    def stop(self):
        """停止底盘并清空内部 PID 状态。"""
        self._robot_api.stop()
        self._linear_pid.reset()
        self._angular_pid.reset()
        self._last_linear_vel = 0.0
        self._last_angular_vel = 0.0

    # ========== VFH 避障 ==========
    def _vfh_find_safe_direction(self, desired_angle: float,
                                  obstacle_sectors: np.ndarray) -> float:
        """从所有开放扇区里选择最接近期望方向的安全角度。"""
        num_sectors = len(obstacle_sectors)
        densities = np.maximum(0, 1.0 - obstacle_sectors / OBSTACLE_AVOID_DIST)
        open_sectors = densities < VFH_THRESHOLD
        if not np.any(open_sectors):
            return desired_angle
        desired_sector = self._angle_to_sector(desired_angle, num_sectors)
        best_sector = -1
        best_cost = float('inf')
        for i in range(num_sectors):
            if not open_sectors[i]:
                continue
            sector_diff = min(abs(i - desired_sector),
                              num_sectors - abs(i - desired_sector))
            cost = sector_diff
            if cost < best_cost:
                best_cost = cost
                best_sector = i
        if best_sector < 0:
            return desired_angle
        return self._sector_to_angle(best_sector, num_sectors)

    def _check_obstacles(self, heading_angle: float,
                          obstacle_sectors: np.ndarray) -> Tuple[bool, float]:
        """
        检查当前航向附近的净空。

        返回 `(是否急停, 速度缩放因子)`，供上层统一做减速或停车。
        """
        num_sectors = len(obstacle_sectors)
        center_sector = self._angle_to_sector(heading_angle, num_sectors)
        check_range = int(30.0 / (360.0 / num_sectors))
        min_dist = LIDAR_MAX_RANGE
        for offset in range(-check_range, check_range + 1):
            idx = (center_sector + offset) % num_sectors
            if obstacle_sectors[idx] < min_dist:
                min_dist = obstacle_sectors[idx]
        clearance = min_dist - ROBOT_RADIUS
        if clearance < OBSTACLE_DANGER_DIST:
            return True, 0.0
        if clearance < OBSTACLE_SLOW_DIST:
            factor = (clearance - OBSTACLE_DANGER_DIST) / (OBSTACLE_SLOW_DIST - OBSTACLE_DANGER_DIST)
            factor = np.clip(factor, 0.1, 1.0)
            return False, factor
        return False, 1.0

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """将角度归一化到 [-pi, pi] 区间。"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    @staticmethod
    def _angle_to_sector(angle_rad: float, num_sectors: int) -> int:
        """将弧度角映射到 VFH 扇区编号。"""
        angle_deg = math.degrees(angle_rad) % 360
        sector_size = 360.0 / num_sectors
        return int(angle_deg / sector_size) % num_sectors

    @staticmethod
    def _sector_to_angle(sector: int, num_sectors: int) -> float:
        """将扇区中心编号反算为弧度角。"""
        sector_size = 360.0 / num_sectors
        angle_deg = sector * sector_size + sector_size / 2.0
        if angle_deg > 180:
            angle_deg -= 360
        return math.radians(angle_deg)
