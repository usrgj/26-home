"""
=============================================================================
state_machine.py — 分层控制状态机
=============================================================================
职责：
管理机器人跟随行为的三种模式之间的切换逻辑。

三种模式（状态）:
1. DIRECT_FOLLOW (直接跟随)
   - 目标可见且路径通畅
   - 使用运动指令直接控制 + 反应式避障
   - 流畅性最好

2. NAV_FOLLOW (导航跟随)
   - 目标丢失/被遮挡/需要过门拐角
   - 切换到导航API，将目标最后已知位置（或预测位置）作为目标点
   - 导航系统负责全局路径规划 + 避障

3. SEARCH (搜索模式)
   - 导航到目标最后位置后仍未找到目标
   - 原地旋转搜索
   - 超时则停止

状态转换条件:
  DIRECT_FOLLOW → NAV_FOLLOW:  视觉连续丢失超过 TARGET_LOST_TIMEOUT
  DIRECT_FOLLOW → SEARCH:      不应直接跳转
  
  NAV_FOLLOW → DIRECT_FOLLOW:  重新看到目标且持续 TARGET_RECOVER_TIMEOUT
  NAV_FOLLOW → SEARCH:         到达导航目标但仍未看到目标
  
  SEARCH → DIRECT_FOLLOW:      重新看到目标
  SEARCH → NAV_FOLLOW:         不应直接跳转 (搜索失败则停止)
"""
import math
import time
import logging
from enum import Enum, auto
from typing import Optional

from .config import (
    TARGET_LOST_TIMEOUT, TARGET_RECOVER_TIMEOUT,
    NAV_GOAL_REACHED_DIST, SEARCH_TIMEOUT,
    FOLLOW_DISTANCE,
    STUCK_TIMEOUT, STUCK_SPEED_THRESHOLD, STUCK_MIN_TARGET_DIST,
)
from .robot_api import RobotAPI, RobotPose
from .sensor_fusion import TargetState

logger = logging.getLogger("StateMachine")


class FollowState(Enum):
    """跟随状态枚举"""
    IDLE = auto()            # 空闲，未开始跟随
    DIRECT_FOLLOW = auto()   # 直接跟随模式
    NAV_FOLLOW = auto()      # 导航跟随模式
    SEARCH = auto()          # 搜索模式
    LOST = auto()            # 彻底丢失，已停止


class StateMachine:
    """分层控制状态机"""
    
    def __init__(self, robot_api: RobotAPI):
        self._robot_api = robot_api
        self._state = FollowState.IDLE
        
        # 各状态的时间追踪
        self._state_enter_time: float = 0.0       # 进入当前状态的时间
        self._vision_lost_since: float = 0.0       # 视觉开始丢失的时间
        self._vision_recover_since: float = 0.0    # 视觉重新看到的时间
        self._target_visible: bool = False          # 目标当前帧是否可见
        
        # 导航目标 (NAV_FOLLOW 模式使用)
        self._nav_goal_x: float = 0.0
        self._nav_goal_y: float = 0.0
        self._nav_goal_theta: float = 0.0
        self._nav_sent: bool = False
        
        # 搜索模式
        self._search_direction: float = 1.0  # 1.0=左转, -1.0=右转

        # 卡住检测 (直接跟随中被避障阻挡)
        self._stuck_since: float = 0.0       # 开始卡住的时间
        self._is_stuck: bool = False          # 当前是否处于卡住状态
    
    @property
    def state(self) -> FollowState:
        return self._state
    
    def start(self):
        """开始跟随"""
        self._transition_to(FollowState.DIRECT_FOLLOW)
        logger.info("跟随开始 → DIRECT_FOLLOW")
    
    def stop(self):
        """停止跟随"""
        self._robot_api.stop()
        try:
            self._robot_api.cancel_navigation()
        except:
            pass
        self._transition_to(FollowState.IDLE)
        logger.info("跟随停止 → IDLE")
    
    # =====================================================================
    # 核心更新逻辑
    # =====================================================================
    def update(self, target: TargetState, robot_pose: RobotPose,
               cmd_linear_vel: float = None) -> FollowState:
        """
        根据当前目标状态和机器人位姿，更新状态机。

        每个主循环周期调用一次。状态机根据条件决定是否切换模式，
        并返回当前状态供主循环据此选择控制策略。

        参数:
            target: 当前目标状态估计 (来自EKF)
            robot_pose: 当前机器人位姿
            cmd_linear_vel: 本帧实际下发的线速度 (m/s)，用于卡住检测

        返回:
            当前状态 (供主循环使用)
        """
        now = time.time()

        # 更新目标可见性
        # 如果EKF不在coasting状态，说明最近有传感器观测，目标"可见"
        target_visible_now = target.is_valid and not target.is_coasting

        # 检测可见性变化
        if target_visible_now and not self._target_visible:
            # 从不可见变为可见
            self._vision_recover_since = now
        elif not target_visible_now and self._target_visible:
            # 从可见变为不可见
            self._vision_lost_since = now

        self._target_visible = target_visible_now

        # --- 根据当前状态执行转换逻辑 ---
        if self._state == FollowState.IDLE:
            pass  # 等待 start() 调用

        elif self._state == FollowState.DIRECT_FOLLOW:
            self._update_direct_follow(target, robot_pose, now, cmd_linear_vel)

        elif self._state == FollowState.NAV_FOLLOW:
            self._update_nav_follow(target, robot_pose, now)

        elif self._state == FollowState.SEARCH:
            self._update_search(target, robot_pose, now)

        elif self._state == FollowState.LOST:
            # 如果重新看到目标，复活
            if target_visible_now:
                self._transition_to(FollowState.DIRECT_FOLLOW)
                logger.info("目标重新出现 → DIRECT_FOLLOW")

        return self._state
    
    # =====================================================================
    # 各状态的更新逻辑
    # =====================================================================
    def _update_direct_follow(self, target: TargetState,
                               robot_pose: RobotPose, now: float,
                               cmd_linear_vel: float = None):
        """
        DIRECT_FOLLOW 状态的转换逻辑。

        退出条件：
        1. 目标连续不可见超过阈值 → NAV_FOLLOW
        2. 目标可见但被避障卡住（速度≈0且目标较远）超过阈值 → NAV_FOLLOW
        """
        # --- 条件1：目标丢失 ---
        if not self._target_visible:
            self._is_stuck = False  # 不可见时不算卡住
            lost_duration = now - self._vision_lost_since

            if lost_duration > TARGET_LOST_TIMEOUT:
                self._switch_to_nav(target, robot_pose, "目标丢失")
                return

        # --- 条件2：被避障卡住（目标可见但走不动） ---
        if self._target_visible and cmd_linear_vel is not None and target.is_valid:
            dist_to_target = math.hypot(target.x - robot_pose.x,
                                        target.y - robot_pose.y)
            is_stuck_now = (abs(cmd_linear_vel) < STUCK_SPEED_THRESHOLD and
                           dist_to_target > STUCK_MIN_TARGET_DIST)

            if is_stuck_now and not self._is_stuck:
                self._stuck_since = now
                self._is_stuck = True
            elif not is_stuck_now:
                self._is_stuck = False

            if self._is_stuck and (now - self._stuck_since) > STUCK_TIMEOUT:
                self._is_stuck = False
                self._switch_to_nav(target, robot_pose, "避障卡住")
                return

    def _switch_to_nav(self, target: TargetState, robot_pose: RobotPose,
                       reason: str):
        """从 DIRECT_FOLLOW 切换到 NAV_FOLLOW 的共用逻辑"""
        if target.is_valid:
            self._nav_goal_x = target.predicted_x
            self._nav_goal_y = target.predicted_y
            self._nav_goal_theta = target.heading
        else:
            self._nav_goal_x = target.x
            self._nav_goal_y = target.y
            self._nav_goal_theta = 0.0

        self._nav_sent = False
        self._transition_to(FollowState.NAV_FOLLOW)
        logger.info(f"{reason} → NAV_FOLLOW "
                    f"目标: ({self._nav_goal_x:.2f}, {self._nav_goal_y:.2f})")
    
    def _update_nav_follow(self, target: TargetState,
                            robot_pose: RobotPose, now: float):
        """
        NAV_FOLLOW 状态的更新逻辑。
        
        在导航过程中:
        - 如果重新看到目标并持续了足够时间 → 切回 DIRECT_FOLLOW
        - 如果导航到达目标位置但仍未看到 → 进入 SEARCH
        - 如果导航失败 → 进入 SEARCH
        """
        # 检查是否重新看到目标
        if self._target_visible:
            recover_duration = now - self._vision_recover_since
            if recover_duration > TARGET_RECOVER_TIMEOUT:
                # 目标稳定恢复，切回直接跟随
                try:
                    self._robot_api.cancel_navigation()
                except:
                    pass
                self._transition_to(FollowState.DIRECT_FOLLOW)
                logger.info("目标恢复可见 → DIRECT_FOLLOW")
                return
        
        # 发送导航目标 (只发一次)
        if not self._nav_sent:
            try:
                success = self._robot_api.navigate_to(
                    self._nav_goal_x, self._nav_goal_y, self._nav_goal_theta
                )
                self._nav_sent = True
                if not success:
                    logger.warning("导航请求失败 → SEARCH")
                    self._transition_to(FollowState.SEARCH)
                    return
            except NotImplementedError:
                # 导航API未实现，直接进搜索模式
                logger.warning("导航API未实现 → SEARCH")
                self._transition_to(FollowState.SEARCH)
                return
        
        # 检查是否到达导航目标
        dist_to_goal = math.hypot(
            robot_pose.x - self._nav_goal_x,
            robot_pose.y - self._nav_goal_y,
        )
        
        if dist_to_goal < NAV_GOAL_REACHED_DIST:
            # 到达目标位置但仍未看到目标 → 搜索
            logger.info(f"到达导航目标 (距离{dist_to_goal:.2f}m) 但目标不可见 → SEARCH")
            self._transition_to(FollowState.SEARCH)
            return
        
        # 也可以检查导航状态
        try:
            nav_status = self._robot_api.get_navigation_status()
            if nav_status.status == "FAILED":
                logger.warning("导航失败 → SEARCH")
                self._transition_to(FollowState.SEARCH)
            elif nav_status.status == "COMPLETED":
                if not self._target_visible:
                    logger.info("导航到达但目标不可见 → SEARCH")
                    self._transition_to(FollowState.SEARCH)
        except NotImplementedError:
            pass  # 导航状态查询未实现，忽略
    
    def _update_search(self, target: TargetState,
                       robot_pose: RobotPose, now: float):
        """
        SEARCH 状态的更新逻辑。
        
        原地旋转搜索目标。如果找到则切回 DIRECT_FOLLOW，
        超时则标记为 LOST。
        """
        # 找到目标了
        if self._target_visible:
            self._transition_to(FollowState.DIRECT_FOLLOW)
            logger.info("搜索找到目标 → DIRECT_FOLLOW")
            return
        
        # 超时
        search_duration = now - self._state_enter_time
        if search_duration > SEARCH_TIMEOUT:
            self._robot_api.stop()
            self._transition_to(FollowState.LOST)
            logger.warning(f"搜索超时 {SEARCH_TIMEOUT}s → LOST")
    
    # =====================================================================
    # 工具方法
    # =====================================================================
    def _transition_to(self, new_state: FollowState):
        """状态转换"""
        old_state = self._state
        self._state = new_state
        self._state_enter_time = time.time()
        
        # 进入新状态时的初始化
        if new_state == FollowState.NAV_FOLLOW:
            self._nav_sent = False
        elif new_state == FollowState.SEARCH:
            # 搜索方向：如果有目标的最后运动方向信息，朝那个方向搜索
            self._search_direction = 1.0  # 默认左转
    
    def get_search_direction(self) -> float:
        """获取搜索旋转方向"""
        return self._search_direction
    
    def get_status_str(self) -> str:
        """获取当前状态的可读字符串"""
        elapsed = time.time() - self._state_enter_time
        return f"[{self._state.name}] {elapsed:.1f}s | 目标{'可见' if self._target_visible else '不可见'}"
