"""
可嵌入任务状态机的人物跟随运行器。

职责：
- 管理跟随子系统的初始化、单周期执行和停止/清理
- 保留 follow 子系统内部的 EKF、跟随状态机和控制逻辑
- 为 task1 外层状态机提供 start/step/stop 风格接口
"""
from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import MAIN_LOOP_RATE, MAP_POINTS_NPY_PATH
from .robot_api import RobotAPI, RobotPose
from .lidar_processor import LidarProcessor
from .vision_detector import VisionDetector
from .sensor_fusion import SensorFusion, TargetState
from .motion_controller import MotionController
from .state_machine import StateMachine, FollowState


logger = logging.getLogger("FollowRunner")


@dataclass
class FollowStepResult:
    """单次跟随控制周期的摘要结果。"""

    timestamp: float
    loop_count: int
    follow_state: FollowState
    robot_pose: RobotPose
    target_state: TargetState
    target_locked: bool
    vision_detection_count: int
    lidar_candidate_count: int


class FollowRunner:
    """可被 task1 外层状态机调用的跟随运行器。"""

    def __init__(
        self,
        robot_api: Optional[RobotAPI] = None,
        lidar_proc: Optional[LidarProcessor] = None,
        vision_det: Optional[VisionDetector] = None,
        fusion: Optional[SensorFusion] = None,
        motion_ctrl: Optional[MotionController] = None,
        state_machine: Optional[StateMachine] = None,
        *,
        loop_rate: float = MAIN_LOOP_RATE,
        vision_interval: int = 3,
        map_points_npy_path: str = MAP_POINTS_NPY_PATH,
    ):
        self._robot_api = robot_api
        self._lidar_proc = lidar_proc
        self._vision_det = vision_det
        self._fusion = fusion
        self._motion_ctrl = motion_ctrl
        self._state_machine = state_machine

        self._loop_rate = loop_rate
        self._loop_period = 1.0 / loop_rate if loop_rate > 0 else 0.0
        self._vision_interval = max(1, vision_interval)
        self._status_log_interval = max(1, int(loop_rate * 2))
        self._map_points_npy_path = map_points_npy_path

        self._initialized = False
        self._running = False
        self._closed = False
        self._target_locked = False
        self._loop_count = 0
        self._last_result: Optional[FollowStepResult] = None
        self._last_robot_pose = RobotPose()
        self._last_target_state = TargetState(is_valid=False)

    @property
    def target_locked(self) -> bool:
        """当前是否已锁定跟随目标。"""
        return self._target_locked

    def start(self, lock_target: bool = True, target_camera: str = "head") -> bool:
        """
        初始化并启动跟随循环。

        返回值表示当前会话是否已完成视觉锁目标。
        """
        if self._closed:
            raise RuntimeError("FollowRunner 已关闭，不能再次启动")

        self._ensure_initialized()

        if self._running:
            return self._target_locked

        self._loop_count = 0
        self._last_result = None
        self._last_robot_pose = RobotPose()
        self._last_target_state = TargetState(is_valid=False)
        self._target_locked = False

        if hasattr(self._vision_det, "reset_target"):
            self._vision_det.reset_target()
        if hasattr(self._fusion, "reset"):
            self._fusion.reset()

        if lock_target:
            logger.info("正在锁定跟随目标...")
            logger.info("  请确保目标人物站在机器人正前方，面朝相机")
            try:
                self._target_locked = bool(
                    self._vision_det.lock_target_from_frame(self._robot_api, target_camera)
                )
            except NotImplementedError:
                logger.warning("相机接口未实现，跳过视觉目标锁定")
            except Exception as exc:
                logger.warning("视觉目标锁定失败: %s", exc)

            if not self._target_locked:
                logger.warning("未能通过视觉锁定目标，将在运行中尝试自动锁定")

        self._state_machine.start()
        self._running = True
        logger.info("跟随运行器已启动")
        return self._target_locked

    def step(self) -> FollowStepResult:
        """执行一个跟随控制周期。"""
        if not self._running:
            raise RuntimeError("FollowRunner 尚未启动，请先调用 start()")

        loop_start = time.time()
        self._loop_count += 1

        robot_pose = self._last_robot_pose
        target_state = self._last_target_state
        current_state = self._state_machine.state
        vision_count = 0
        lidar_count = 0

        try:
            try:
                self._robot_api.get_state()
            except Exception as exc:
                logger.warning("拉取推送失败: %s", exc)
                return self._finalize_step(
                    loop_start,
                    robot_pose,
                    target_state,
                    current_state,
                    vision_count,
                    lidar_count,
                )

            try:
                robot_pose = self._robot_api.get_robot_pose()
            except NotImplementedError:
                logger.warning("获取机器人世界坐标失败，跳过本周期")
                return self._finalize_step(
                    loop_start,
                    robot_pose,
                    target_state,
                    current_state,
                    vision_count,
                    lidar_count,
                )
            except Exception as exc:
                logger.warning("读取机器人位姿异常: %s", exc)
                return self._finalize_step(
                    loop_start,
                    robot_pose,
                    target_state,
                    current_state,
                    vision_count,
                    lidar_count,
                )

            lidar_candidates = []
            obstacle_sectors = np.full(72, 40.0)

            try:
                scans = self._robot_api.get_lidar_scans()
                if scans:
                    lidar_candidates = self._lidar_proc.process(scans, robot_pose)
                    obstacle_sectors = self._lidar_proc.get_obstacle_sectors(scans)
            except NotImplementedError:
                pass
            except Exception as exc:
                logger.debug("LiDAR处理异常: %s", exc)

            lidar_count = len(lidar_candidates)

            vision_detections = []
            target_detection = None

            if self._loop_count % self._vision_interval == 0:
                try:
                    vision_detections = self._vision_det.detect(self._robot_api, robot_pose)
                    for det in vision_detections:
                        if det.is_target:
                            target_detection = det
                            break

                    if target_detection is None and not self._target_locked and vision_detections:
                        nearest = min(vision_detections, key=lambda d: d.depth)
                        self._vision_det.lock_target(nearest)
                        target_detection = nearest
                        target_detection.is_target = True
                        self._target_locked = True
                        logger.info("自动锁定目标: depth=%.2fm", nearest.depth)

                except NotImplementedError:
                    pass
                except Exception as exc:
                    logger.debug("视觉检测异常: %s", exc)

            vision_count = len(vision_detections)
            now = time.time()

            if target_detection is not None:
                self._fusion.update_with_vision(target_detection)
                self._fusion.set_vision_anchor(
                    target_detection.world_x,
                    target_detection.world_y,
                )

            matched_lidar = None
            if lidar_candidates:
                matched_lidar = self._fusion.associate_lidar_candidates(lidar_candidates)
                if matched_lidar is not None:
                    self._fusion.update_with_lidar(matched_lidar)

            if target_detection is None and matched_lidar is None:
                self._fusion.predict_only(now)

            target_state = self._fusion.get_target_state()

            cmd_linear = 0.0
            cmd_angular = 0.0
            if target_state.is_valid:
                cmd_linear, cmd_angular = self._motion_ctrl.compute_velocity(
                    target_state,
                    robot_pose,
                    obstacle_sectors,
                )

            current_state = self._state_machine.update(
                target_state,
                robot_pose,
                cmd_linear_vel=cmd_linear,
            )

            if current_state == FollowState.DIRECT_FOLLOW:
                if target_state.is_valid:
                    try:
                        self._robot_api.send_velocity(cmd_linear, cmd_angular)
                    except NotImplementedError:
                        pass
                else:
                    self._motion_ctrl.stop()

            elif current_state == FollowState.NAV_FOLLOW:
                pass

            elif current_state == FollowState.SEARCH:
                direction = self._state_machine.get_search_direction()
                linear_vel, angular_vel = self._motion_ctrl.rotate_search(direction)
                try:
                    self._robot_api.send_velocity(linear_vel, angular_vel)
                except NotImplementedError:
                    pass

            elif current_state == FollowState.LOST:
                self._motion_ctrl.stop()

            elif current_state == FollowState.IDLE:
                pass

            if self._loop_count % self._status_log_interval == 0:
                status = self._state_machine.get_status_str()
                if target_state.is_valid:
                    dist = math.hypot(
                        target_state.x - robot_pose.x,
                        target_state.y - robot_pose.y,
                    )
                    logger.info(
                        "%s | 目标(%.2f,%.2f) 距离=%.2fm 速度=%.2fm/s | LiDAR候选=%d",
                        status,
                        target_state.x,
                        target_state.y,
                        dist,
                        target_state.speed,
                        lidar_count,
                    )
                else:
                    logger.info("%s | 目标状态无效", status)

            return self._finalize_step(
                loop_start,
                robot_pose,
                target_state,
                current_state,
                vision_count,
                lidar_count,
            )

        except Exception:
            try:
                self._robot_api.stop()
            except Exception:
                pass
            logger.error("跟随 step 异常", exc_info=True)
            raise

    def stop(self) -> None:
        """停止当前跟随，不释放共享硬件资源。"""
        if not self._initialized:
            return

        try:
            if self._motion_ctrl is not None:
                self._motion_ctrl.stop()
        except Exception:
            pass

        try:
            if self._state_machine is not None:
                self._state_machine.stop()
        except Exception:
            pass

        self._running = False

    def close(self) -> None:
        """完整关闭跟随运行器，仅供独立脚本使用。"""
        if self._closed:
            return

        try:
            self.stop()
        finally:
            try:
                if self._robot_api is not None and hasattr(self._robot_api, "release"):
                    self._robot_api.release()
            finally:
                self._closed = True

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        self._robot_api = self._robot_api or RobotAPI()
        self._lidar_proc = self._lidar_proc or LidarProcessor()
        self._vision_det = self._vision_det or VisionDetector()
        self._fusion = self._fusion or SensorFusion()
        self._motion_ctrl = self._motion_ctrl or MotionController(self._robot_api)
        self._state_machine = self._state_machine or StateMachine(self._robot_api)

        self._load_map()
        self._initialized = True

    def _load_map(self) -> None:
        logger.info("正在加载全局地图...")
        if self._map_points_npy_path:
            self._lidar_proc.load_map_from_npy(self._map_points_npy_path)
            return

        try:
            map_data = self._robot_api.get_global_map()
            if map_data is not None:
                self._lidar_proc.load_map_from_dict(map_data)
        except NotImplementedError:
            logger.warning("get_global_map() 未实现且无预处理地图文件")
            logger.warning("  → LiDAR将无法做地图差分，所有LiDAR点都视为动态物体")
            logger.warning("  → 请运行: python map_preprocessor.py <log文件> ./maps/")
        except Exception as exc:
            logger.error("地图加载失败: %s", exc)

    def _finalize_step(
        self,
        loop_start: float,
        robot_pose: RobotPose,
        target_state: TargetState,
        current_state: FollowState,
        vision_count: int,
        lidar_count: int,
    ) -> FollowStepResult:
        elapsed = time.time() - loop_start
        sleep_time = self._loop_period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        self._last_robot_pose = robot_pose
        self._last_target_state = target_state
        self._last_result = FollowStepResult(
            timestamp=time.time(),
            loop_count=self._loop_count,
            follow_state=current_state,
            robot_pose=robot_pose,
            target_state=target_state,
            target_locked=self._target_locked,
            vision_detection_count=vision_count,
            lidar_candidate_count=lidar_count,
        )
        return self._last_result
