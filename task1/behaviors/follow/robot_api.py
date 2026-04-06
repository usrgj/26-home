"""
=============================================================================
robot_api.py — 机器人硬件抽象层
=============================================================================

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
from .config import ROBOT_MAX_LINEAR_VEL, ROBOT_MAX_ANGULAR_VEL



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
    fx: float = 0.0     # 焦距 x (像素)
    fy: float = 0.0     # 焦距 y (像素)
    ppx: float = 0.0    # 主点 x (像素)
    ppy: float = 0.0    # 主点 y (像素)


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
    status: str = ""        # "idle"/"running"/"completed"/"failed"/"canceled"
    # task_type: int = 0        # 导航类型

state_map  = {
                0: None,
                1: "WAITING",
                2: "RUNNING", 
                3: "SUSPENDED",
                4: "COMPLETED",
                5: "FAILED",
                6: "CANCELED",
                }
# =============================================================================
# 机器人API类
# =============================================================================
class RobotAPI:
    """
    机器人硬件交互的统一接口。
    
    ★★★ TODO★★★
    你需要将这个类中的每个方法替换为你自己机器人的实际API调用。
    例如：
    - 如果你的机器人通过HTTP REST API通信，这里用requests库调用
    - 如果通过TCP/Protobuf通信，这里用socket + protobuf库
    - 如果通过共享内存或SDK，这里调用相应SDK
    """
    
    def __init__(self):
        """
        TODO初始化与机器人的连接。
        例如：建立TCP连接、初始化SDK、连接相机等。
        """
        from agv_api import agv
        from camera import camera_manager
        from common.config import CAMERA_CHEST, CAMERA_HEAD
        self._agv = agv
        
        self._state: Dict = {}
        # 切换推送配置
        self.wait_for_data()

        self.camera_head = camera_manager.get(CAMERA_HEAD)
        self.camera_chest = camera_manager.get(CAMERA_CHEST)


        
        print("自动跟随初始化")
    
    def wait_for_data(self, timeout: float = 6.0) -> bool:
        """
        切换配置需要时间
        等待首次推送数据到达。在主循环开始前调用。

        返回: True=数据已到达, False=超时
        """
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
                # debug
                # for i in range(20):
                #     print(self._agv.poll().response["data"])
                #     time.sleep(0.1)

                return
            except:
                time.sleep(0.1)
                print(self._agv.poll_push())
                print("err")
        print("ERROR: 等待更新配置后的推送超时!!!")

    def get_state(self):
        self._state = self._agv.poll_push().response["data"]
    # =====================================================================
    # 位姿与速度获取
    # =====================================================================
    def get_robot_pose(self) -> RobotPose:
        """
        获取机器人当前世界坐标系下的位姿。

        返回: RobotPose(x, y, theta, timestamp)
        - x, y: 世界坐标 (m)
        - theta: 朝向角 (rad)
        - timestamp: 秒级时间戳
        """
        return RobotPose(
            x=self._state.get("x"),
            y=self._state.get("y"),
            theta=self._state.get("angle", 0.0),
            timestamp=self._state.get("create_on"),
        )        

    '''
    def get_robot_velocity(self) -> RobotVelocity:
        """
        TODO获取机器人在机器人坐标系下的速度。
        
        返回: RobotVelocity(vx, vy, omega, timestamp)
        - vx: 前向速度 (m/s)
        - vy: 横向速度 (m/s)，差速底盘通常为0
        - omega: 角速度 (rad/s)
        """
        raise NotImplementedError("请实现 get_robot_velocity()")
    '''
    # =====================================================================
    # LiDAR数据获取
    # =====================================================================
    def get_lidar_scans(self) -> List[LidarScan]:
        """
        获取所有LiDAR的最新一帧扫描数据。
        
        返回: 列表，每个元素是一个LidarScan
        
        从你给的point_cloud.txt数据格式来看，你的API返回的是一个列表，
        包含两个激光雷达的数据。每个雷达的数据包含:
        - beams: 光束列表，每个beam有 angle, dist, rssi, valid
        - device_info: 设备信息
        - install_info: 安装信息 (x, yaw, z, upside)
        - header: 时间戳信息
        
        你需要将原始数据解析并填入 LidarScan 数据结构。
        """
        raw_data = self._agv.get_lidar()
        if raw_data is None:
            raise NotImplementedError("获取激光雷达为空")

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
    
    # =====================================================================
    # 相机数据获取
    # =====================================================================
    def get_camera_frame(self, camera_name: str) -> CameraFrame:
        """
        获取指定相机的最新一帧彩色图+深度图。

        参数:
            camera_name: 相机名称 ("head", "chest")

        返回: CameraFrame
            - color_image: numpy数组 (H, W, 3)，BGR格式
            - depth_image: numpy数组 (H, W)，单位毫米 (uint16)
        """
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

    
    # =====================================================================
    # 运动控制指令
    # =====================================================================
    def send_velocity(self, linear_vel: float, angular_vel: float):
        """
        发送线速度和角速度指令（直接运动控制）。
        
        参数:
            linear_vel: 线速度 (m/s)，正值前进，负值后退
            angular_vel: 角速度 (rad/s)，正值左转，负值右转
        
          vx = linear_vel    前向速度 (m/s)
          vy = 0             差速底盘横向为0
          w  = angular_vel   角速度 (rad/s)，逆时针为正
          duration = 200     200ms看门狗，程序崩溃后自动停
        """
        vx = max(-ROBOT_MAX_LINEAR_VEL, min(ROBOT_MAX_LINEAR_VEL, linear_vel))
        w = max(-ROBOT_MAX_ANGULAR_VEL, min(ROBOT_MAX_ANGULAR_VEL, angular_vel))

        self._agv.send_velocity(vx=vx, w=w)
        print(f'vx:{vx}, w:{w}')  


    def send_arc_motion(self, linear_vel: float, radius: float):
        """
        发送圆弧运动指令。
        
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
        """紧急停止，发送零速度。"""
        self.send_velocity(0.0, 0.0)
    
    # =====================================================================
    # 导航接口
    # =====================================================================
    def navigate_to(self, x: float, y: float, theta: float) -> bool:
        """
        TODO
        发送导航目标点，让机器人自主导航到指定世界坐标。
        
        参数:
            x, y: 目标世界坐标 (m)
            theta: 目标朝向 (rad)
        
        返回: 
            bool — 导航请求是否发送成功（不代表已到达）
        
        这个接口是非阻塞的，发送导航目标后立即返回。
        用 get_navigation_status() 查询导航状态。
        """
        return self._agv.free_navigate_to(x, y, theta)
    
    def get_navigation_status(self) -> NavigationResult:
        """
        查询当前导航任务的状态。
        
        返回: NavigationResult
            - success: 是否到达
            - status: 
        """
        sucess = True
        task_code = self._state.get("task_status")
        if task_code != 4:
            sucess = False
        
        return NavigationResult(success=sucess, 
                                status=state_map.get(task_code))
        
    
    def cancel_navigation(self):
        """
        取消当前导航任务。
        当需要从导航模式切回直接控制模式时调用。
        """
        self._agv.cancel_navigation()
    
    # =====================================================================
    # 地图接口
    # =====================================================================
    def get_global_map(self) -> Optional[Dict]:
        """
        TODO
        获取protobuf格式的2D全局地图并转为字典。

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

    '''
    def get_nearest_obstacle(self) -> Optional[Obstacle]:
        """
        TODO
        获取最近障碍物的世界坐标。

        """
        return Obstacle(
            x=self._state.get("block_x"),
            y=self._state.get("block_y"),

        )
        '''

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
    
    def release(self):
        '''
        关闭机器人推送
        '''
        self._agv.configure_push(interval=9990, fields=["create_on"])
        