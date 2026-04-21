"""
robot_api.py — 机器人硬件抽象层
"""
import math
import time
import json
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from task1.behaviors.follow.config import ROBOT_MAX_LINEAR_VEL, ROBOT_MAX_ANGULAR_VEL
else:
    from .config import ROBOT_MAX_LINEAR_VEL, ROBOT_MAX_ANGULAR_VEL


@dataclass
class RobotPose:
    """机器人在世界坐标系中的位姿"""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    timestamp: float = 0.0


@dataclass
class RobotVelocity:
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    timestamp: float = 0.0


@dataclass
class LidarBeam:
    angle: float = 0.0
    dist: float = 0.0
    rssi: float = 0.0
    valid: bool = False


@dataclass
class LidarScan:
    beams: List[LidarBeam] = field(default_factory=list)
    device_name: str = ""
    install_x: float = 0.0
    install_yaw: float = 0.0
    timestamp: float = 0.0


@dataclass
class CameraFrame:
    color_image: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None
    camera_name: str = ""
    timestamp: float = 0.0
    fx: float = 0.0
    fy: float = 0.0
    ppx: float = 0.0
    ppy: float = 0.0


@dataclass
class Obstacle:
    x: float = 0.0
    y: float = 0.0
    distance: float = 0.0


@dataclass
class NavigationResult:
    success: bool = False
    status: str = ""


state_map = {
    0: None,
    1: "WAITING",
    2: "RUNNING",
    3: "SUSPENDED",
    4: "COMPLETED",
    5: "FAILED",
    6: "CANCELED",
}


class RobotAPI:
    """封装跟随子系统需要的底盘、LiDAR、相机和导航接口。"""
    def __init__(self):
        from common.skills.agv_api import agv
        from common.skills.camera import camera_manager
        from common.config import CAMERA_CHEST, CAMERA_HEAD
        self._agv = agv

        self._state: Dict = {}
        # 缓存上次有效位姿，防止None值
        self._last_valid_pose: Optional[RobotPose] = None

        self.wait_for_data()

        self.camera_head = camera_manager.get(CAMERA_HEAD)
        self.camera_chest = camera_manager.get(CAMERA_CHEST)

        print("自动跟随初始化")

    def wait_for_data(self, timeout: float = 6.0) -> bool:
        """配置 AGV 推送字段，并等待第一帧有效推送到达。"""
        result = self._agv.configure_push(interval=50, fields=[
            "x", "y", "angle", "task_status",
            "vx", "w", "create_on", "block_x", "block_y",
        ])
        print(result)
        time.sleep(3)
        start = time.time()
        while time.time() - start < timeout:
            try:
                result = self._agv.poll_push().response["data"]
                return True
            except:
                time.sleep(0.1)
                print(self._agv.poll_push())
                print("err")
        print("ERROR: 等待更新配置后的推送超时!!!")
        return False

    def get_state(self):
        """拉取一帧新的 AGV 推送，失败时显式报错。"""
        result = self._agv.poll_push()
        if result is None or not getattr(result, "ok", True):
            raise RuntimeError("未收到新的 AGV 推送")

        response = getattr(result, "response", None) or {}
        data = response.get("data")
        if not data:
            raise RuntimeError("AGV 推送数据为空")

        self._state = data
        return True

    def has_valid_pose(self) -> bool:
        """当前缓存推送中是否包含可用于闭环控制的位姿。"""
        return self._state.get("x") is not None and self._state.get("y") is not None

    def get_robot_pose(self) -> RobotPose:
        """
        获取机器人当前位姿，若数据无效则返回上次有效值或默认值。
        """
        x = self._state.get("x")
        y = self._state.get("y")
        theta = self._state.get("angle", 0.0)
        timestamp = self._state.get("create_on", time.time())

        # 防御：若关键字段为 None，使用缓存值或默认值
        if x is None or y is None:
            if self._last_valid_pose is not None:
                # 沿用上次位姿
                pose = self._last_valid_pose
                # 更新时间戳
                pose.timestamp = time.time()
                return pose
            else:
                # 首次启动且无有效数据，返回原点
                x, y, theta = 0.0, 0.0, 0.0
                timestamp = time.time()

        # 确保类型转换安全
        try:
            pose = RobotPose(
                x=float(x),
                y=float(y),
                theta=float(theta) if theta is not None else 0.0,
                timestamp=float(timestamp) if timestamp is not None else time.time(),
            )
        except (TypeError, ValueError):
            if self._last_valid_pose is not None:
                pose = self._last_valid_pose
                pose.timestamp = time.time()
            else:
                pose = RobotPose(0.0, 0.0, 0.0, time.time())

        self._last_valid_pose = pose
        return pose

    def get_lidar_scans(self) -> List[LidarScan]:
        """读取底盘返回的全部 LiDAR 原始扫描，并整理成统一数据结构。"""
        raw_data = self._agv.get_lidar()
        if raw_data is None:
            return []

        scans = []
        for laser_data in raw_data:
            install_info = laser_data.get('install_info', {})
            device_info = laser_data.get('device_info', {})

            scan = LidarScan(
                device_name=device_info.get('device_name', ''),
                install_x=install_info.get('x', 0.0),
                install_yaw=install_info.get('yaw', 0.0),
                timestamp=time.time(),
            )

            for beam_data in laser_data.get('beams', []):
                valid = beam_data.get('valid', False)
                if isinstance(valid, int):
                    valid = valid != 0

                scan.beams.append(LidarBeam(
                    angle=beam_data.get('angle', 0.0),
                    dist=beam_data.get('dist', 0.0),
                    rssi=beam_data.get('rssi', 0.0),
                    valid=valid,
                ))

            scans.append(scan)

        return scans

    def get_camera_frame(self, camera_name: str) -> CameraFrame:
        """按名称获取一帧彩色图、深度图和相机内参。"""
        camera_map = {
            "head": self.camera_head,
            "chest": self.camera_chest,
        }
        cam = camera_map.get(camera_name)
        if cam is None:
            raise ValueError(f"未知相机名称: {camera_name}")

        if not cam.started:
            cam.start()

        color_image, depth_image = cam.get_frames()
        intr = cam.intrinsics
        return CameraFrame(
            color_image=color_image,
            depth_image=depth_image,
            camera_name=camera_name,
            timestamp=time.time(),
            fx=intr.fx if intr else 0.0,
            fy=intr.fy if intr else 0.0,
            ppx=intr.ppx if intr else 0.0,
            ppy=intr.ppy if intr else 0.0,
        )

    def send_velocity(self, linear_vel: float, angular_vel: float):
        """下发差速底盘速度，并在接口层做一次限幅保护。"""
        vx = max(-ROBOT_MAX_LINEAR_VEL, min(ROBOT_MAX_LINEAR_VEL, linear_vel))
        w = max(-ROBOT_MAX_ANGULAR_VEL, min(ROBOT_MAX_ANGULAR_VEL, angular_vel))
        self._agv.send_velocity(vx=vx, w=w)

    def send_arc_motion(self, linear_vel: float, radius: float):
        """用圆弧半径换算角速度，再复用通用速度接口。"""
        if abs(radius) < 0.01:
            angular_vel = 0.0
        else:
            angular_vel = linear_vel / radius
        self.send_velocity(linear_vel, angular_vel)

    def stop(self):
        """立即停止底盘。"""
        self.send_velocity(0.0, 0.0)

    def navigate_to(self, x: float, y: float, theta: float) -> bool:
        """调用底盘自由导航，到达世界坐标系下的目标位姿。"""
        try:
            return self._agv.free_navigate_to(x, y, theta)
        except:
            return False

    def get_navigation_status(self) -> NavigationResult:
        """将底盘任务状态码转换为跟随模块可直接消费的导航状态。"""
        task_code = self._state.get("task_status")
        success = (task_code == 4)
        return NavigationResult(success=success, status=state_map.get(task_code, "UNKNOWN"))

    def cancel_navigation(self):
        """取消当前导航任务。"""
        self._agv.cancel_navigation()

    def get_global_map(self) -> Optional[Dict]:
        """预留接口：如需在线取图，可在此接入地图服务。"""
        raise NotImplementedError("请实现 get_global_map()")

    def robot_to_world(self, local_x: float, local_y: float, pose: RobotPose) -> Tuple[float, float]:
        """将机器人坐标系中的点转换到世界坐标系。"""
        cos_t = math.cos(pose.theta)
        sin_t = math.sin(pose.theta)
        world_x = pose.x + local_x * cos_t - local_y * sin_t
        world_y = pose.y + local_x * sin_t + local_y * cos_t
        return world_x, world_y

    def world_to_robot(self, world_x: float, world_y: float, pose: RobotPose) -> Tuple[float, float]:
        """将世界坐标系中的点转换到机器人坐标系。"""
        dx = world_x - pose.x
        dy = world_y - pose.y
        cos_t = math.cos(pose.theta)
        sin_t = math.sin(pose.theta)
        local_x = dx * cos_t + dy * sin_t
        local_y = -dx * sin_t + dy * cos_t
        return local_x, local_y

    def release(self):
        """恢复较轻量的 AGV 推送配置，供独立脚本退出时清理使用。"""
        self._agv.configure_push(interval=9990, fields=["create_on"])
