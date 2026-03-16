"""
=============================================================================
robot_api.py — 机器人硬件抽象层
=============================================================================
★★★ 这是你需要重点适配的文件 ★★★

本文件封装了所有与机器人硬件交互的接口。你需要将每个方法的实现
替换为你自己机器人SDK/API的实际调用。

当前所有方法都是占位实现 (placeholder)，用 NotImplementedError
或模拟数据标注。
"""
import math
import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


# =============================================================================
# 数据结构定义
# =============================================================================
@dataclass
class RobotPose:
    """机器人在世界坐标系中的位姿"""
    x: float = 0.0          # 世界坐标X (m)
    y: float = 0.0          # 世界坐标Y (m)
    theta: float = 0.0      # 朝向角 (rad)，以世界坐标系X轴正方向为0
    timestamp: float = 0.0  # 时间戳 (s)


@dataclass
class RobotVelocity:
    """机器人在机器人坐标系中的速度"""
    vx: float = 0.0         # 前向线速度 (m/s)
    vy: float = 0.0         # 横向线速度 (m/s)，差速底盘通常为0
    omega: float = 0.0      # 角速度 (rad/s)
    timestamp: float = 0.0


@dataclass
class LidarBeam:
    """单个LiDAR光束"""
    angle: float = 0.0      # 角度 (deg)
    dist: float = 0.0       # 距离 (m)
    rssi: float = 0.0       # 信号强度
    valid: bool = False      # 是否有效


@dataclass
class LidarScan:
    """一帧完整的LiDAR扫描数据"""
    beams: List[LidarBeam] = field(default_factory=list)
    device_name: str = ""   # "laser" (前) 或 "laser1" (后)
    install_x: float = 0.0  # 安装位置X偏移
    install_yaw: float = 0.0  # 安装朝向偏移 (deg)
    timestamp: float = 0.0


@dataclass
class CameraFrame:
    """一帧相机数据"""
    color_image: Optional[np.ndarray] = None    # BGR彩色图 (H, W, 3)
    depth_image: Optional[np.ndarray] = None    # 深度图 (H, W)，单位 mm
    camera_name: str = ""
    timestamp: float = 0.0


@dataclass
class Obstacle:
    """障碍物信息"""
    x: float = 0.0          # 世界坐标X
    y: float = 0.0          # 世界坐标Y
    distance: float = 0.0   # 到机器人的距离


@dataclass
class NavigationResult:
    """导航结果"""
    success: bool = False
    status: str = ""        # "reached", "planning", "moving", "failed"


# =============================================================================
# 机器人API类
# =============================================================================
class RobotAPI:
    """
    机器人硬件交互的统一接口。
    
    ★★★ [需适配] ★★★
    你需要将这个类中的每个方法替换为你自己机器人的实际API调用。
    例如：
    - 如果你的机器人通过HTTP REST API通信，这里用requests库调用
    - 如果通过TCP/Protobuf通信，这里用socket + protobuf库
    - 如果通过共享内存或SDK，这里调用相应SDK
    """
    
    def __init__(self):
        """
        [需适配] 初始化与机器人的连接。
        例如：建立TCP连接、初始化SDK、连接相机等。
        """
        self._connected = False
        # ---------------------------------------------------------------
        # 示例：如果你的机器人用TCP通信
        # import socket
        # self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self._socket.connect(("192.168.1.100", 19200))
        # ---------------------------------------------------------------
        
        # ---------------------------------------------------------------
        # 示例：如果你用Intel RealSense相机
        # import pyrealsense2 as rs
        # self._pipelines = {}
        # for cam_name, cam_serial in CAMERA_SERIALS.items():
        #     pipeline = rs.pipeline()
        #     config = rs.config()
        #     config.enable_device(cam_serial)
        #     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        #     config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        #     pipeline.start(config)
        #     self._pipelines[cam_name] = pipeline
        # ---------------------------------------------------------------
        print("[RobotAPI] 初始化完成 (当前为占位实现，请替换为实际API)")
    
    # =====================================================================
    # 位姿与速度获取
    # =====================================================================
    def get_robot_pose(self) -> RobotPose:
        """
        [需适配] 获取机器人当前世界坐标系下的位姿。
        
        返回: RobotPose(x, y, theta, timestamp)
        - x, y: 世界坐标 (m)
        - theta: 朝向角 (rad)
        - timestamp: 秒级时间戳
        
        你的API可能返回的是角度制，需要转为弧度制。
        你的API可能返回的是protobuf消息，需要解析。
        """
        # ---------------------------------------------------------------
        # 示例实现 (替换为你的实际代码):
        # response = self._send_command("GET_POSE")
        # return RobotPose(
        #     x=response.x,
        #     y=response.y,
        #     theta=math.radians(response.angle),  # 如果API返回角度制
        #     timestamp=time.time()
        # )
        # ---------------------------------------------------------------
        raise NotImplementedError("请实现 get_robot_pose()，调用你的机器人API获取世界坐标位姿")
    
    def get_robot_velocity(self) -> RobotVelocity:
        """
        [需适配] 获取机器人在机器人坐标系下的速度。
        
        返回: RobotVelocity(vx, vy, omega, timestamp)
        - vx: 前向速度 (m/s)
        - vy: 横向速度 (m/s)，差速底盘通常为0
        - omega: 角速度 (rad/s)
        """
        raise NotImplementedError("请实现 get_robot_velocity()")
    
    # =====================================================================
    # LiDAR数据获取
    # =====================================================================
    def get_lidar_scans(self) -> List[LidarScan]:
        """
        [需适配] 获取所有LiDAR的最新一帧扫描数据。
        
        返回: 列表，每个元素是一个LidarScan
        
        从你给的point_cloud.txt数据格式来看，你的API返回的是一个列表，
        包含两个激光雷达的数据。每个雷达的数据包含:
        - beams: 光束列表，每个beam有 angle, dist, rssi, valid
        - device_info: 设备信息
        - install_info: 安装信息 (x, yaw, z, upside)
        - header: 时间戳信息
        
        你需要将原始数据解析并填入 LidarScan 数据结构。
        """
        # ---------------------------------------------------------------
        # 示例实现:
        # raw_data = self._get_point_cloud()  # 调用你的API获取原始点云
        # scans = []
        # for laser_data in raw_data:
        #     scan = LidarScan(
        #         device_name=laser_data['device_info']['device_name'],
        #         install_x=laser_data['install_info']['x'],
        #         install_yaw=laser_data['install_info'].get('yaw', 0.0),
        #         timestamp=float(laser_data['header']['data_nsec']) * 1e-9,
        #     )
        #     for beam_data in laser_data['beams']:
        #         if beam_data.get('valid', False):
        #             scan.beams.append(LidarBeam(
        #                 angle=beam_data.get('angle', 0.0),
        #                 dist=beam_data['dist'],
        #                 rssi=beam_data.get('rssi', 0.0),
        #                 valid=True,
        #             ))
        #     scans.append(scan)
        # return scans
        # ---------------------------------------------------------------
        raise NotImplementedError("请实现 get_lidar_scans()，解析你的点云数据")
    
    # =====================================================================
    # 相机数据获取
    # =====================================================================
    def get_camera_frame(self, camera_name: str) -> CameraFrame:
        """
        [需适配] 获取指定相机的最新一帧彩色图+深度图。
        
        参数:
            camera_name: 相机名称 ("head", "chest", "left_arm", "right_arm")
        
        返回: CameraFrame
            - color_image: numpy数组 (H, W, 3)，BGR格式
            - depth_image: numpy数组 (H, W)，单位毫米 (uint16)
        
        如果你用的是Intel RealSense，可以用pyrealsense2库:
        """
        # ---------------------------------------------------------------
        # 示例实现 (Intel RealSense):
        # pipeline = self._pipelines[camera_name]
        # frames = pipeline.wait_for_frames()
        # color_frame = frames.get_color_frame()
        # depth_frame = frames.get_depth_frame()
        # 
        # color_image = np.asanyarray(color_frame.get_data())
        # depth_image = np.asanyarray(depth_frame.get_data())
        # 
        # return CameraFrame(
        #     color_image=color_image,
        #     depth_image=depth_image,
        #     camera_name=camera_name,
        #     timestamp=time.time(),
        # )
        # ---------------------------------------------------------------
        raise NotImplementedError(f"请实现 get_camera_frame('{camera_name}')")
    
    # =====================================================================
    # 运动控制指令
    # =====================================================================
    def send_velocity(self, linear_vel: float, angular_vel: float):
        """
        [需适配] 发送线速度和角速度指令（直接运动控制）。
        
        参数:
            linear_vel: 线速度 (m/s)，正值前进，负值后退
            angular_vel: 角速度 (rad/s)，正值左转，负值右转
        
        注意：差速底盘将 (v, omega) 转化为左右轮速度：
            v_left  = linear_vel - angular_vel * wheel_base / 2
            v_right = linear_vel + angular_vel * wheel_base / 2
        
        你的API可能接受的是左右轮速度，也可能直接接受 (v, omega)。
        如果你的API支持发送"平动"和"转动"指令，这里需要做对应的转换。
        """
        # ---------------------------------------------------------------
        # 示例实现:
        # self._send_command("MOVE", {
        #     "linear_velocity": linear_vel,
        #     "angular_velocity": angular_vel
        # })
        # ---------------------------------------------------------------
        raise NotImplementedError("请实现 send_velocity()")
    
    def send_arc_motion(self, linear_vel: float, radius: float):
        """
        [需适配] 发送圆弧运动指令。
        
        参数:
            linear_vel: 线速度 (m/s)
            radius: 转弯半径 (m)，正值左转，负值右转
        
        圆弧运动等价于: angular_vel = linear_vel / radius
        如果你的API不直接支持圆弧运动，可以用 send_velocity 代替。
        """
        if abs(radius) < 0.01:
            angular_vel = 0.0
        else:
            angular_vel = linear_vel / radius
        self.send_velocity(linear_vel, angular_vel)
    
    def stop(self):
        """[需适配] 紧急停止，发送零速度。"""
        self.send_velocity(0.0, 0.0)
    
    # =====================================================================
    # 导航接口
    # =====================================================================
    def navigate_to(self, x: float, y: float, theta: float) -> bool:
        """
        [需适配] 发送导航目标点，让机器人自主导航到指定世界坐标。
        
        参数:
            x, y: 目标世界坐标 (m)
            theta: 目标朝向 (rad)
        
        返回: 
            bool — 导航请求是否发送成功（不代表已到达）
        
        这个接口是非阻塞的，发送导航目标后立即返回。
        用 get_navigation_status() 查询导航状态。
        """
        # ---------------------------------------------------------------
        # 示例实现:
        # angle_deg = math.degrees(theta)
        # result = self._send_command("NAVIGATE", {
        #     "x": x, "y": y, "angle": angle_deg
        # })
        # return result.get("success", False)
        # ---------------------------------------------------------------
        raise NotImplementedError("请实现 navigate_to()")
    
    def get_navigation_status(self) -> NavigationResult:
        """
        [需适配] 查询当前导航任务的状态。
        
        返回: NavigationResult
            - success: 是否到达
            - status: "idle" / "planning" / "moving" / "reached" / "failed"
        """
        raise NotImplementedError("请实现 get_navigation_status()")
    
    def cancel_navigation(self):
        """
        [需适配] 取消当前导航任务。
        当需要从导航模式切回直接控制模式时调用。
        """
        # ---------------------------------------------------------------
        # 示例: self._send_command("CANCEL_NAV")
        # ---------------------------------------------------------------
        raise NotImplementedError("请实现 cancel_navigation()")
    
    # =====================================================================
    # 地图接口
    # =====================================================================
    def get_global_map(self) -> Optional[Dict]:
        """
        [需适配] 获取protobuf格式的2D全局地图并转为字典。
        
        返回: 解析后的地图字典，格式取决于你的地图protobuf定义。
        
        你提到可以"获取protobuf格式的2D全局地图并转为json进行解析"，
        这里需要你实现具体的获取和解析逻辑。
        
        地图数据通常包含:
        - origin: 地图原点在世界坐标系中的位置
        - resolution: 分辨率 (m/pixel)
        - width, height: 地图尺寸 (pixels)
        - data: 栅格数据 (0=空闲, 100=占据, -1=未知)
        """
        # ---------------------------------------------------------------
        # 示例实现:
        # raw_proto = self._send_command("GET_MAP")
        # map_msg = MapMessage()
        # map_msg.ParseFromString(raw_proto)
        # 
        # # 转为json再转为dict
        # from google.protobuf.json_format import MessageToDict
        # map_dict = MessageToDict(map_msg)
        # return map_dict
        # ---------------------------------------------------------------
        raise NotImplementedError("请实现 get_global_map()")
    
    def get_nearest_obstacle(self) -> Optional[Obstacle]:
        """
        [需适配] 获取最近障碍物的世界坐标。
        
        你提到"能获取最近的障碍物的世界坐标（不能指定获取动态障碍物的坐标）"。
        这个接口返回地图上标记的最近障碍物，可能不包括动态障碍物。
        """
        raise NotImplementedError("请实现 get_nearest_obstacle()")
    
    # =====================================================================
    # 工具方法
    # =====================================================================
    def robot_to_world(self, local_x: float, local_y: float,
                       pose: RobotPose) -> Tuple[float, float]:
        """
        将机器人坐标系中的点转换到世界坐标系。
        
        参数:
            local_x, local_y: 机器人坐标系中的坐标 (m)
                              机器人坐标系: X朝前, Y朝左
            pose: 机器人当前世界位姿
        
        返回: (world_x, world_y) 世界坐标
        """
        cos_t = math.cos(pose.theta)
        sin_t = math.sin(pose.theta)
        world_x = pose.x + local_x * cos_t - local_y * sin_t
        world_y = pose.y + local_x * sin_t + local_y * cos_t
        return world_x, world_y
    
    def world_to_robot(self, world_x: float, world_y: float,
                       pose: RobotPose) -> Tuple[float, float]:
        """
        将世界坐标系中的点转换到机器人坐标系。
        
        返回: (local_x, local_y) 机器人坐标
        """
        dx = world_x - pose.x
        dy = world_y - pose.y
        cos_t = math.cos(pose.theta)
        sin_t = math.sin(pose.theta)
        local_x = dx * cos_t + dy * sin_t
        local_y = -dx * sin_t + dy * cos_t
        return local_x, local_y
    
    def disconnect(self):
        """[需适配] 断开与机器人的连接，释放资源。"""
        # ---------------------------------------------------------------
        # for pipeline in self._pipelines.values():
        #     pipeline.stop()
        # self._socket.close()
        # ---------------------------------------------------------------
        print("[RobotAPI] 已断开连接")
