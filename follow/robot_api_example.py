"""
=============================================================================
robot_api_example.py — 基于 AGVManager SDK 的参考适配示例
=============================================================================
★★★ 这不是直接可用的代码 ★★★

根据你的 log 输出，你使用的是 AGVManager SDK，连接了多个端口:
  - 19204, 19205, 19206, 19207, 19210

本文件展示了如何将 robot_api.py 中的占位方法替换为 AGVManager 的调用。
你需要根据你的 AGVManager 的实际API文档来补充细节。

使用方式:
  1. 复制本文件为 robot_api.py (替换原来的)
  2. 根据你的 AGVManager API 修改标记为 [TODO] 的部分
  3. 运行 phase1_lidar_follow.py 测试

提示: 如果你不确定 AGVManager 的某个方法名称或返回格式，
可以写一个小脚本来探索:
  import AGVManager
  help(AGVManager)         # 查看模块文档
  dir(AGVManager)          # 查看所有方法
  help(AGVManager.某方法)   # 查看特定方法文档
"""
import math
import time
import ast
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

# 导入原始数据结构 (不变)
from dataclasses import dataclass, field


# --- 数据结构 (与 robot_api.py 相同) ---
@dataclass
class RobotPose:
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

@dataclass
class NavigationResult:
    success: bool = False
    status: str = ""

@dataclass
class Obstacle:
    x: float = 0.0
    y: float = 0.0
    distance: float = 0.0


class RobotAPI:
    """
    基于 AGVManager SDK 的实际适配。
    """
    
    def __init__(self):
        # =================================================================
        # [TODO] 初始化 AGVManager 连接
        # 
        # 从你的log可以看到，AGVManager 连接了5个端口:
        #   19204, 19205, 19206, 19207, 19210
        # 这些端口可能分别对应不同的服务:
        #   - 一个端口用于控制指令
        #   - 一个端口用于传感器数据
        #   - 一个端口用于导航
        #   - 等等
        #
        # 请根据你的AGVManager文档确认各端口的用途。
        # =================================================================
        
        import AGVManager
        
        # [TODO] 替换为你的机器人IP和端口
        self._robot_ip = "192.168.1.100"
        self._agv = AGVManager  # 或 AGVManager.connect(self._robot_ip)
        
        # 等待连接就绪
        # [TODO] 根据实际情况修改
        # self._agv.connect(self._robot_ip)
        # time.sleep(1.0)
        
        print("[RobotAPI] AGVManager 已连接")
    
    # =====================================================================
    # 位姿获取
    # =====================================================================
    def get_robot_pose(self) -> RobotPose:
        """
        [TODO] 获取世界坐标位姿。
        
        AGVManager 可能有类似这样的方法:
          pose = self._agv.get_pose()
          pose = self._agv.get_location()
          pose = self._agv.get_robot_position()
        
        返回值可能是:
          - 字典: {"x": ..., "y": ..., "angle": ...}
          - 元组: (x, y, angle)
          - 对象: pose.x, pose.y, pose.angle
        
        注意角度单位：如果API返回度数，需要转为弧度。
        """
        # [TODO] 替换为实际调用
        # raw = self._agv.get_pose()  
        # return RobotPose(
        #     x=raw['x'],           # 或 raw[0], raw.x
        #     y=raw['y'],           
        #     theta=math.radians(raw['angle']),  # 度→弧度
        #     timestamp=time.time(),
        # )
        raise NotImplementedError("请实现 get_robot_pose()")
    
    # =====================================================================
    # LiDAR数据获取 — 这是最关键的适配
    # =====================================================================
    def get_lidar_scans(self) -> List[LidarScan]:
        """
        获取LiDAR点云。
        
        从你的 point_cloud.txt 可以看到，你的API返回的格式是:
        [
            {
                "beams": [{"angle": ..., "dist": ..., "rssi": ..., "valid": ...}, ...],
                "device_info": {"device_name": "laser"},
                "install_info": {"x": 0.299, "z": 0.238, ...},
                "header": {"data_nsec": ...}
            },
            {
                "beams": [...],
                "device_info": {"device_name": "laser1"},
                "install_info": {"x": -0.299, "yaw": 180, "z": 0.238, ...},
                ...
            }
        ]
        
        关键点:
        - 第一个是前LiDAR (device_name="laser", x=0.299, 无yaw偏移=0°)
        - 第二个是后LiDAR (device_name="laser1", x=-0.299, yaw=180°)
        """
        # [TODO] 替换为你获取点云的实际方法
        # raw_data = self._agv.get_point_cloud()
        # 
        # 如果返回的是字符串，需要解析:
        # raw_data = ast.literal_eval(raw_str)
        # 
        # 如果返回的是protobuf，需要解码:
        # raw_data = parse_point_cloud_proto(proto_bytes)
        
        raw_data = None  # [TODO] 替换
        
        if raw_data is None:
            raise NotImplementedError("请实现 get_lidar_scans()")
        
        scans = []
        for laser_data in raw_data:
            # 获取安装信息
            install_info = laser_data.get('install_info', {})
            device_info = laser_data.get('device_info', {})
            
            scan = LidarScan(
                device_name=device_info.get('device_name', ''),
                install_x=install_info.get('x', 0.0),
                install_yaw=install_info.get('yaw', 0.0),
                timestamp=time.time(),
            )
            
            # 解析光束数据
            for beam_data in laser_data.get('beams', []):
                valid = beam_data.get('valid', False)
                # 注意: 你的数据中 valid 字段可能是布尔值或整数
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
    # 运动控制
    # =====================================================================
    def send_velocity(self, linear_vel: float, angular_vel: float):
        """
        [TODO] 发送运动指令。
        
        你提到可以"发送平动、转动和圆弧运动的指令"。
        
        方式A: 如果你的API支持直接发送 (v, ω):
          self._agv.set_velocity(linear_vel, angular_vel)
        
        方式B: 如果你的API只支持分别发送平动和转动:
          # 将 (v, ω) 分解为平动+转动的组合
          # 对于差速底盘, 可以用圆弧运动近似:
          if abs(angular_vel) < 0.01:
              self._agv.move_translation(linear_vel)
          else:
              radius = linear_vel / angular_vel
              self._agv.move_arc(linear_vel, radius)
        
        方式C: 如果你的API接受左右轮速度:
          v_left  = linear_vel - angular_vel * WHEEL_BASE / 2
          v_right = linear_vel + angular_vel * WHEEL_BASE / 2
          self._agv.set_wheel_speeds(v_left, v_right)
        """
        # [TODO] 替换为实际调用
        raise NotImplementedError("请实现 send_velocity()")
    
    def stop(self):
        """紧急停止"""
        # [TODO] 
        # self._agv.stop() 或 self.send_velocity(0, 0)
        try:
            self.send_velocity(0.0, 0.0)
        except NotImplementedError:
            pass
    
    # =====================================================================
    # 导航 (Phase 2+ 才需要)
    # =====================================================================
    def navigate_to(self, x: float, y: float, theta: float) -> bool:
        """
        [TODO] 发送导航目标。Phase 1 不需要实现。
        
        你提到"能自由导航到世界坐标系内的任意坐标和角度"。
        """
        # [TODO]
        # angle_deg = math.degrees(theta)
        # return self._agv.navigate_to(x, y, angle_deg)
        raise NotImplementedError("Phase 2+ 实现")
    
    def get_navigation_status(self) -> NavigationResult:
        raise NotImplementedError("Phase 2+ 实现")
    
    def cancel_navigation(self):
        # [TODO] self._agv.cancel_navigation()
        raise NotImplementedError("Phase 2+ 实现")
    
    # =====================================================================
    # 其他 (按需实现)
    # =====================================================================
    def get_robot_velocity(self) -> RobotVelocity:
        # [TODO]
        # raw = self._agv.get_velocity()
        # return RobotVelocity(vx=raw['vx'], omega=raw['omega'], ...)
        raise NotImplementedError()
    
    def get_camera_frame(self, camera_name: str) -> CameraFrame:
        """
        [TODO] Phase 2+ 实现。
        
        Intel RealSense 示例:
          import pyrealsense2 as rs
          pipeline = self._pipelines[camera_name]
          frames = pipeline.wait_for_frames()
          ...
        """
        raise NotImplementedError("Phase 2+ 实现")
    
    def get_global_map(self) -> Optional[Dict]:
        """不需要运行时调用，用预处理的 .npy 文件代替"""
        return None
    
    def get_nearest_obstacle(self) -> Optional[Obstacle]:
        raise NotImplementedError()
    
    # =====================================================================
    # 坐标变换工具 (不需要修改)
    # =====================================================================
    def robot_to_world(self, local_x, local_y, pose):
        cos_t = math.cos(pose.theta)
        sin_t = math.sin(pose.theta)
        return (pose.x + local_x * cos_t - local_y * sin_t,
                pose.y + local_x * sin_t + local_y * cos_t)
    
    def world_to_robot(self, world_x, world_y, pose):
        dx = world_x - pose.x
        dy = world_y - pose.y
        cos_t = math.cos(pose.theta)
        sin_t = math.sin(pose.theta)
        return (dx * cos_t + dy * sin_t,
                -dx * sin_t + dy * cos_t)
    
    def disconnect(self):
        # [TODO] self._agv.disconnect()
        print("[RobotAPI] 已断开")
