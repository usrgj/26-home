"""
state_machine.py — 分层控制状态机（改进版：灵敏卡住检测 + 脱困协同）
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
    IDLE = auto()
    DIRECT_FOLLOW = auto()
    NAV_FOLLOW = auto()
    SEARCH = auto()
    LOST = auto()
    ARRIVED = auto()


class StateMachine:
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
        
        # ----- 新增：记录机器人历史位置用于检测实际移动 -----
        self._last_robot_pos: Optional[tuple] = None
        self._stuck_start_pos: Optional[tuple] = None
    
    @property
    def state(self) -> FollowState:
        return self._state
    
    def start(self):
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
        self._last_robot_pos = None
        self._stuck_start_pos = None
        self._transition_to(FollowState.DIRECT_FOLLOW)
        logger.info("跟随开始 → DIRECT_FOLLOW")
    
    def stop(self):
        self._robot_api.stop()
        try:
            self._robot_api.cancel_navigation()
        except:
            pass
        self._transition_to(FollowState.IDLE)
        logger.info("跟随停止 → IDLE")
    
    def update(self, target: TargetState, robot_pose: RobotPose,
               cmd_linear_vel: float = None) -> FollowState:
        now = time.time()
        target_visible_now = target.is_valid and not target.is_coasting

        # 目标可见性变化检测
        if target_visible_now and not self._target_visible:
            self._vision_recover_since = now
        elif not target_visible_now and self._target_visible:
            self._vision_lost_since = now
        self._target_visible = target_visible_now

        # ----- 新增：更新机器人历史位置 -----
        current_pos = (robot_pose.x, robot_pose.y)
        if self._last_robot_pos is None:
            self._last_robot_pos = current_pos

        if self._state == FollowState.IDLE:
            pass
        elif self._state == FollowState.DIRECT_FOLLOW:
            self._update_direct_follow(target, robot_pose, now, cmd_linear_vel, current_pos)
        elif self._state == FollowState.NAV_FOLLOW:
            self._update_nav_follow(target, robot_pose, now)
        elif self._state == FollowState.SEARCH:
            self._update_search(target, robot_pose, now)
        elif self._state == FollowState.LOST:
            if target_visible_now:
                self._transition_to(FollowState.DIRECT_FOLLOW)
                logger.info("目标重新出现 → DIRECT_FOLLOW")
        elif self._state == FollowState.ARRIVED:
            if target_visible_now and target.speed > 0.15:
                self._transition_to(FollowState.DIRECT_FOLLOW)
                logger.info("目标开始移动 → DIRECT_FOLLOW")

        # 更新历史位置
        self._last_robot_pos = current_pos

        return self._state
    
    def _update_direct_follow(self, target: TargetState, robot_pose: RobotPose,
                               now: float, cmd_linear_vel: float, current_pos: tuple):
        # 计算距离
        dist_to_target = math.hypot(target.x - robot_pose.x, target.y - robot_pose.y)
        
        # 目标静止且距离合适 → ARRIVED
        if target.is_valid and target.speed < 0.15 and abs(dist_to_target - FOLLOW_DISTANCE) < 0.3:
            self._transition_to(FollowState.ARRIVED)
            logger.info("目标已静止且距离合适 → ARRIVED")
            return
        
        # 目标丢失超时 → 导航
        if not self._target_visible:
            self._is_stuck = False
            self._stuck_start_pos = None
            lost_duration = now - self._vision_lost_since
            if lost_duration > TARGET_LOST_TIMEOUT:
                self._switch_to_nav(target, robot_pose, "目标丢失")
                return
        
        # ----- 修改：更灵敏的卡住检测 -----
        if self._target_visible and cmd_linear_vel is not None and target.is_valid:
            # 条件1：指令速度很低（基本没动）
            cmd_speed_low = abs(cmd_linear_vel) < STUCK_SPEED_THRESHOLD
            
            # 条件2：目标距离足够远（不是正常停车）
            target_far = dist_to_target > FOLLOW_DISTANCE + 0.5
            
            # 条件3：机器人实际没有移动（通过位置变化判断）
            if self._stuck_start_pos is None:
                self._stuck_start_pos = current_pos
            actual_move = math.hypot(current_pos[0] - self._stuck_start_pos[0],
                                     current_pos[1] - self._stuck_start_pos[1])
            not_moving = actual_move < 0.05  # 5cm内算没动
            
            is_stuck_now = cmd_speed_low and target_far and not_moving
            
            if is_stuck_now and not self._is_stuck:
                self._stuck_since = now
                self._is_stuck = True
                self._stuck_start_pos = current_pos
                logger.debug(f"疑似卡住: cmd={cmd_linear_vel:.3f}, dist={dist_to_target:.2f}")
            elif not is_stuck_now:
                self._is_stuck = False
                self._stuck_start_pos = None
            
            # 卡住超时，切换到导航
            if self._is_stuck and (now - self._stuck_since) > STUCK_TIMEOUT:
                self._is_stuck = False
                self._stuck_start_pos = None
                self._switch_to_nav(target, robot_pose, "避障卡住")
                return
        else:
            self._is_stuck = False
            self._stuck_start_pos = None
    
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
        if self._target_visible:
            self._transition_to(FollowState.DIRECT_FOLLOW)
            logger.info("搜索找到目标 → DIRECT_FOLLOW")
            return
        
        if now - self._state_enter_time > SEARCH_TIMEOUT:
            self._robot_api.stop()
            self._transition_to(FollowState.LOST)
            logger.warning(f"搜索超时 {SEARCH_TIMEOUT}s → LOST")
    
    def _transition_to(self, new_state: FollowState):
        old_state = self._state
        self._state = new_state
        self._state_enter_time = time.time()
        if new_state == FollowState.NAV_FOLLOW:
            self._nav_sent = False
        elif new_state == FollowState.SEARCH:
            self._search_direction = 1.0
        elif new_state == FollowState.ARRIVED:
            self._robot_api.stop()
        logger.debug(f"状态转换: {old_state} → {new_state}")
    
    def get_search_direction(self) -> float:
        return self._search_direction
    
    def get_status_str(self) -> str:
        elapsed = time.time() - self._state_enter_time
        return f"[{self._state.name}] {elapsed:.1f}s | 目标{'可见' if self._target_visible else '不可见'}"
