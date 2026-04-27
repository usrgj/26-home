"""
=============================================================================
state_machine.py — 分层控制状态机
=============================================================================
职责：
管理机器人跟随行为的三种模式之间的切换逻辑。
"""
import math
import time
import logging
from enum import Enum, auto
from .config import (
    TARGET_LOST_TIMEOUT, TARGET_RECOVER_TIMEOUT,
    NAV_GOAL_REACHED_DIST, SEARCH_TIMEOUT,
    STUCK_TIMEOUT, STUCK_SPEED_THRESHOLD, STUCK_MIN_TARGET_DIST,
)
from .robot_api import RobotAPI, RobotPose
from .sensor_fusion import TargetState

logger = logging.getLogger("StateMachine")


class FollowState(Enum):
    """跟随子系统的高层行为模式。"""
    IDLE = auto()
    DIRECT_FOLLOW = auto()
    NAV_FOLLOW = auto()
    SEARCH = auto()
    LOST = auto()


class StateMachine:
    """管理直接跟随、导航接续和搜索之间的切换。"""
    def __init__(self, robot_api: RobotAPI):
        self._robot_api = robot_api
        self._state = FollowState.IDLE
        self._state_enter_time: float = 0.0
        self._vision_lost_since: float = 0.0
        self._vision_recover_since: float = 0.0
        self._target_visible: bool = False
        self._nav_goal_x: float = 0.0
        self._nav_goal_y: float = 0.0
        self._nav_goal_theta: float = 0.0
        self._nav_sent: bool = False
        self._search_direction: float = 1.0
        self._stuck_since: float = 0.0
        self._is_stuck: bool = False
    
    @property
    def state(self) -> FollowState:
        """返回当前高层状态。"""
        return self._state
    
    def start(self):
        """重置内部计时器并进入直接跟随模式。"""
        self._vision_lost_since = 0.0
        self._vision_recover_since = 0.0
        self._target_visible = False
        self._nav_goal_x = 0.0
        self._nav_goal_y = 0.0
        self._nav_goal_theta = 0.0
        self._nav_sent = False
        self._search_direction = 1.0
        self._stuck_since = 0.0
        self._is_stuck = False
        self._transition_to(FollowState.DIRECT_FOLLOW)
        logger.info("跟随开始 → DIRECT_FOLLOW")
    
    def stop(self, send_stop: bool = True):
        """停止导航并回到空闲状态，必要时下发一次底盘停止。"""
        if send_stop:
            self._robot_api.stop()
        try:
            self._robot_api.cancel_navigation()
        except:
            pass
        self._transition_to(FollowState.IDLE)
        logger.info("跟随停止 → IDLE")
    
    def update(self, target: TargetState, robot_pose: RobotPose,
               cmd_linear_vel: float = None) -> FollowState:
        """
        根据当前目标状态和控制输出更新高层模式。

        这里不直接下发速度，只负责判断“继续直跟 / 切导航 / 原地搜索”。
        """
        now = time.time()
        target_visible_now = target.is_valid and not target.is_coasting

        # 目标可见性变化检测
        if target_visible_now and not self._target_visible:
            self._vision_recover_since = now
        elif not target_visible_now and self._target_visible:
            self._vision_lost_since = now
        self._target_visible = target_visible_now

        if self._state == FollowState.IDLE:
            pass
        elif self._state == FollowState.DIRECT_FOLLOW:
            self._update_direct_follow(target, robot_pose, now, cmd_linear_vel)
        elif self._state == FollowState.NAV_FOLLOW:
            self._update_nav_follow(target, robot_pose, now)
        elif self._state == FollowState.SEARCH:
            self._update_search(target, robot_pose, now)
        elif self._state == FollowState.LOST:
            if target_visible_now:
                self._transition_to(FollowState.DIRECT_FOLLOW)
                logger.info("目标重新出现 → DIRECT_FOLLOW")

        return self._state
    
    def _update_direct_follow(self, target: TargetState, robot_pose: RobotPose,
                               now: float, cmd_linear_vel: float = None):
        """处理直接跟随模式下的丢失检测和卡住检测。"""
        if not self._target_visible:
            self._is_stuck = False
            lost_duration = now - self._vision_lost_since
            if lost_duration > TARGET_LOST_TIMEOUT:
                self._switch_to_nav(target, robot_pose, "目标丢失")
                return

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
        """切换到导航模式，设置导航目标点。"""
        if target.is_valid:
            # 使用预测位置作为导航目标，略微超前
            self._nav_goal_x = target.predicted_x
            self._nav_goal_y = target.predicted_y
            self._nav_goal_theta = target.heading
        else:
            # 目标无效时，用最后已知位置
            self._nav_goal_x = target.x
            self._nav_goal_y = target.y
            self._nav_goal_theta = 0.0
        self._nav_sent = False
        self._transition_to(FollowState.NAV_FOLLOW)
        logger.info(f"{reason} → NAV_FOLLOW 目标: ({self._nav_goal_x:.2f}, {self._nav_goal_y:.2f})")
    
    def _update_nav_follow(self, target: TargetState, robot_pose: RobotPose, now: float):
        """处理导航接续模式：发导航、监控恢复、决定是否转搜索。"""
        # 目标重新可见足够时间 → 切回直接跟随
        if self._target_visible:
            recover_duration = now - self._vision_recover_since
            if recover_duration > TARGET_RECOVER_TIMEOUT:
                try:
                    self._robot_api.cancel_navigation()
                except:
                    pass
                self._transition_to(FollowState.DIRECT_FOLLOW)
                logger.info("目标恢复可见 → DIRECT_FOLLOW")
                return
        
        # 发送导航请求（仅一次）
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
                logger.warning("导航API未实现 → SEARCH")
                self._transition_to(FollowState.SEARCH)
                return
        
        # 检查是否到达导航目标
        dist_to_goal = math.hypot(robot_pose.x - self._nav_goal_x,
                                  robot_pose.y - self._nav_goal_y)
        if dist_to_goal < NAV_GOAL_REACHED_DIST:
            logger.info(f"到达导航目标但目标不可见 → SEARCH")
            self._transition_to(FollowState.SEARCH)
            return
        
        # 检查导航状态
        try:
            nav_status = self._robot_api.get_navigation_status()
            if nav_status.status == "FAILED":
                logger.warning("导航失败 → SEARCH")
                self._transition_to(FollowState.SEARCH)
            elif nav_status.status == "COMPLETED":
                if not self._target_visible:
                    logger.info("导航完成但目标不可见 → SEARCH")
                    self._transition_to(FollowState.SEARCH)
        except NotImplementedError:
            pass
    
    def _update_search(self, target: TargetState, robot_pose: RobotPose, now: float):
        """处理原地搜索模式，找到目标就回直跟，超时则宣告丢失。"""
        if self._target_visible:
            self._transition_to(FollowState.DIRECT_FOLLOW)
            logger.info("搜索找到目标 → DIRECT_FOLLOW")
            return
        
        if now - self._state_enter_time > SEARCH_TIMEOUT:
            self._transition_to(FollowState.LOST)
            logger.warning(f"搜索超时 {SEARCH_TIMEOUT}s → LOST")
    
    def _transition_to(self, new_state: FollowState):
        """执行状态切换并重置该状态对应的辅助变量。"""
        self._state = new_state
        self._state_enter_time = time.time()
        if new_state == FollowState.NAV_FOLLOW:
            self._nav_sent = False
        elif new_state == FollowState.SEARCH:
            self._search_direction = 1.0
    
    def get_search_direction(self) -> float:
        """返回搜索模式下的旋转方向。"""
        return self._search_direction
    
    def get_status_str(self) -> str:
        """生成适合日志输出的状态摘要。"""
        elapsed = time.time() - self._state_enter_time
        return f"[{self._state.name}] {elapsed:.1f}s | 目标{'可见' if self._target_visible else '不可见'}"
