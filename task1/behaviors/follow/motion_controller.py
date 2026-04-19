"""
motion_controller.py — 运动控制模块（改进版：脱困状态机 + 前馈优化 + 死区防抖）
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
    VFH_THRESHOLD, LIDAR_MAX_RANGE,
)
from .robot_api import RobotAPI, RobotPose
from .sensor_fusion import TargetState


class PIDController:
    """简单的PID控制器，支持动态修改 kp"""
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
        if self._first:
            self._prev_error = error
            self._prev_time = current_time
            self._first = False
            return self.kp * error

        dt = current_time - self._prev_time
        if dt <= 0:
            dt = 0.001

        p_out = self.kp * error

        # 积分分离：误差较大时禁用积分
        if abs(error) < 0.3:
            self._integral += error * dt
            self._integral = np.clip(self._integral, -10.0, 10.0)
        i_out = self.ki * self._integral

        d_out = self.kd * (error - self._prev_error) / dt

        self._prev_error = error
        self._prev_time = current_time

        output = p_out + i_out + d_out
        return np.clip(output, self.output_min, self.output_max)

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._first = True


class MotionController:
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
        self._prev_angle_error = 0.0
        self._prev_time = 0.0

        # ----- 新增：脱困状态机 -----
        self._escape_state = 0          # 0: 未激活, 1: 后退, 2: 旋转寻找出口
        self._escape_timer = 0.0
        self._escape_rotation_dir = 1.0  # 旋转方向：1 左转，-1 右转

    def compute_velocity(self, target: TargetState, robot_pose: RobotPose,
                         obstacle_sectors: np.ndarray) -> Tuple[float, float]:
        now = time.time()
        dt = now - self._prev_time if self._prev_time > 0 else 0.02
        self._prev_time = now

        # ---------- 基本量计算 ----------
        dx = target.x - robot_pose.x
        dy = target.y - robot_pose.y
        dist_to_target = math.hypot(dx, dy)
        angle_to_target_world = math.atan2(dy, dx)
        angle_error = self._normalize_angle(angle_to_target_world - robot_pose.theta)

        # ----- 修改：预测方向仅在速度估计可靠时使用 -----
        # 假设 TargetState 中添加了 speed_variance 属性，若没有则默认高方差
        speed_variance = getattr(target, 'speed_variance', 1.0)
        if target.speed > 0.3 and speed_variance < 0.5 and target.is_valid:
            pred_dx = target.predicted_x - robot_pose.x
            pred_dy = target.predicted_y - robot_pose.y
            desired_angle_world = math.atan2(pred_dy, pred_dx)
            desired_angle = self._normalize_angle(desired_angle_world - robot_pose.theta)
        else:
            desired_angle = angle_error

        # VFH 安全方向
        safe_angle = self._vfh_find_safe_direction(desired_angle, obstacle_sectors)

        # ----- 新增：VFH 无安全方向时进入脱困模式 -----
        if not self._has_safe_direction(obstacle_sectors):
            # 触发脱困，忽略正常控制
            linear_vel, angular_vel = self._escape_stuck(obstacle_sectors, now)
            # 紧急停止检查仍保留（避免碰撞）
            emergency_stop, _ = self._check_obstacles(safe_angle, obstacle_sectors)
            if emergency_stop:
                linear_vel = 0.0
                angular_vel = 0.0
            self._last_linear_vel = linear_vel
            self._last_angular_vel = angular_vel
            return linear_vel, angular_vel
        else:
            # 有安全方向时重置脱困状态
            self._escape_state = 0

        # 障碍物紧急停止/减速
        emergency_stop, speed_factor = self._check_obstacles(safe_angle, obstacle_sectors)
        if emergency_stop:
            self._robot_api.stop()
            return 0.0, 0.0

        # ---------- 动态跟随距离（转弯时增大） ----------
        angle_error_rate = (angle_error - self._prev_angle_error) / dt if dt > 0 else 0.0
        self._prev_angle_error = angle_error
        target_angular_speed = abs(angle_error_rate)
        dynamic_follow_dist = FOLLOW_DISTANCE
        if target_angular_speed > 0.8:
            dynamic_follow_dist = FOLLOW_DISTANCE * 1.4
        elif target_angular_speed > 0.4:
            dynamic_follow_dist = FOLLOW_DISTANCE * 1.2

        dist_error = dist_to_target - dynamic_follow_dist

        # ---------- 前馈 + PID 线速度（改进：只在可靠时启用） ----------
        if abs(dist_error) < FOLLOW_DISTANCE_TOLERANCE:
            pid_linear = 0.0
        else:
            pid_linear = self._linear_pid.compute(dist_error, now)

        # ----- 修改：前馈速度仅在目标快速且稳定时加入，并打折扣 -----
        feedforward = 0.0
        if target.speed > 0.4 and speed_variance < 0.5 and dist_to_target > 1.0:
            feedforward = target.speed * 0.6  # 打折扣避免过冲

        linear_vel = feedforward + pid_linear

        # 目标在后方时不前进
        if abs(angle_error) > math.pi / 2:
            linear_vel = 0.0

        # 转弯时降低线速度
        linear_scale = max(0.3, 1.0 - min(1.0, target_angular_speed / 1.5))
        linear_vel *= linear_scale
        linear_vel = np.clip(linear_vel, -ROBOT_MAX_LINEAR_VEL, ROBOT_MAX_LINEAR_VEL)
        linear_vel *= speed_factor

        # ----- 修改：角速度死区，防止微小误差抖动 -----
        if abs(safe_angle) < math.radians(3):  # 3度死区
            angular_vel = 0.0
        else:
            # 动态角速度增益
            abs_angle_error = abs(safe_angle)
            angular_gain = 1.0 + min(1.5, abs_angle_error / math.radians(60))
            angular_vel = self._angular_pid.compute(safe_angle, now) * angular_gain

        angular_vel = np.clip(angular_vel, -ROBOT_MAX_ANGULAR_VEL, ROBOT_MAX_ANGULAR_VEL)

        # ----- 新增：当 VFH 安全方向与期望方向偏差大时，进一步减速 -----
        angle_diff = abs(self._normalize_angle(safe_angle - desired_angle))
        if angle_diff > math.radians(30):
            linear_vel *= 0.5
        elif angle_diff > math.radians(15):
            linear_vel *= 0.8

        # 高角速度时降低线速度
        angular_ratio = abs(angular_vel) / ROBOT_MAX_ANGULAR_VEL
        if angular_ratio > 0.5:
            linear_vel *= (1.0 - 0.3 * angular_ratio)

        # 斜坡限制（平滑）
        max_linear_delta = MAX_LINEAR_ACCEL / MAIN_LOOP_RATE
        max_angular_delta = MAX_ANGULAR_ACCEL / MAIN_LOOP_RATE
        linear_vel = np.clip(linear_vel,
                             self._last_linear_vel - max_linear_delta,
                             self._last_linear_vel + max_linear_delta)
        angular_vel = np.clip(angular_vel,
                              self._last_angular_vel - max_angular_delta,
                              self._last_angular_vel + max_angular_delta)

        self._last_linear_vel = linear_vel
        self._last_angular_vel = angular_vel

        return linear_vel, angular_vel

    # ---------- 新增：检查是否存在安全方向 ----------
    def _has_safe_direction(self, obstacle_sectors: np.ndarray) -> bool:
        densities = np.maximum(0, 1.0 - obstacle_sectors / OBSTACLE_AVOID_DIST)
        open_sectors = densities < VFH_THRESHOLD
        return np.any(open_sectors)

    # ---------- 新增：脱困状态机 ----------
    def _escape_stuck(self, obstacle_sectors: np.ndarray, now: float) -> Tuple[float, float]:
        """
        当 VFH 没有安全方向时执行脱困序列：
        - 先后退 1.0 秒
        - 然后原地旋转，直到前方有足够空间
        """
        if self._escape_state == 0:
            self._escape_state = 1
            self._escape_timer = now
            self._escape_rotation_dir = 1.0  # 默认左转

        if self._escape_state == 1:
            # 后退
            linear_vel = -0.15
            angular_vel = 0.0
            if now - self._escape_timer > 1.0:
                self._escape_state = 2
                self._escape_timer = now
        elif self._escape_state == 2:
            # 旋转寻找出口
            linear_vel = 0.0
            angular_vel = 0.5 * self._escape_rotation_dir
            # 检查前方 60° 扇区是否已开放
            front_open = np.any(obstacle_sectors[30:42] > OBSTACLE_AVOID_DIST * 1.5)
            if front_open or now - self._escape_timer > 3.0:
                # 找到出口或超时，重置脱困状态，下一周期恢复正常控制
                self._escape_state = 0
        else:
            linear_vel = angular_vel = 0.0

        return linear_vel, angular_vel

    def rotate_search(self, direction: float = 1.0) -> Tuple[float, float]:
        from .config import SEARCH_ROTATION_SPEED
        return 0.0, direction * SEARCH_ROTATION_SPEED

    def stop(self):
        self._robot_api.stop()
        self._linear_pid.reset()
        self._angular_pid.reset()
        self._last_linear_vel = 0.0
        self._last_angular_vel = 0.0
        self._prev_angle_error = 0.0
        self._escape_state = 0   # 重置脱困状态

    # ========== VFH 避障 ==========
    def _vfh_find_safe_direction(self, desired_angle: float,
                                  obstacle_sectors: np.ndarray) -> float:
        num_sectors = len(obstacle_sectors)
        sector_size_rad = 2 * math.pi / num_sectors
        densities = np.maximum(0, 1.0 - obstacle_sectors / OBSTACLE_AVOID_DIST)
        open_sectors = densities < VFH_THRESHOLD
        if not np.any(open_sectors):
            return desired_angle   # 无开放扇区，返回原方向（上层会进入脱困）
        desired_sector = self._angle_to_sector(desired_angle, num_sectors)
        best_sector = -1
        best_cost = float('inf')
        for i in range(num_sectors):
            if not open_sectors[i]:
                continue
            sector_diff = min(abs(i - desired_sector),
                              num_sectors - abs(i - desired_sector))
            # 增加障碍物距离惩罚，鼓励选择更空旷的方向
            dist_penalty = 1.0 - min(obstacle_sectors[i] / OBSTACLE_AVOID_DIST, 1.0)
            cost = sector_diff + dist_penalty * 2.0
            if cost < best_cost:
                best_cost = cost
                best_sector = i
        if best_sector < 0:
            return desired_angle
        return self._sector_to_angle(best_sector, num_sectors)

    def _check_obstacles(self, heading_angle: float,
                          obstacle_sectors: np.ndarray) -> Tuple[bool, float]:
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
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    @staticmethod
    def _angle_to_sector(angle_rad: float, num_sectors: int) -> int:
        angle_deg = math.degrees(angle_rad) % 360
        sector_size = 360.0 / num_sectors
        return int(angle_deg / sector_size) % num_sectors

    @staticmethod
    def _sector_to_angle(sector: int, num_sectors: int) -> float:
        sector_size = 360.0 / num_sectors
        angle_deg = sector * sector_size + sector_size / 2.0
        if angle_deg > 180:
            angle_deg -= 360
        return math.radians(angle_deg)
