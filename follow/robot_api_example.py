"""
=============================================================================
robot_api.py — 机器人硬件抽象层 (基于 AGVManager + 4B65推送)
=============================================================================
架构说明:
  机器人通过端口19301主动推送状态数据 (cmd_id=4B65)，包含位姿、速度、
  障碍物、导航状态、电池等几乎所有信息。

  本模块的核心设计:
  1. 启动时注册 4B65 回调，所有推送数据缓存到 _state 字典
  2. get_robot_pose() / get_robot_velocity() 等方法直接读缓存，零延迟
  3. 运动指令 / 导航指令通过 AGVManager SDK 主动发送
  4. 线程安全: 推送回调在SDK的接收线程中执行，读取时用锁保护

  ★★★ [需适配] 标记的地方需要你替换为 AGVManager SDK 的实际方法名 ★★★

4B65 推送字段与本模块方法的对应关系:
┌─────────────────┬──────────────────────────────────────────────────┐
│ 推送字段         │ 用途                                             │
├─────────────────┼──────────────────────────────────────────────────┤
│ x, y, angle     │ → get_robot_pose()       世界坐标位姿             │
│ vx, vy, w       │ → get_robot_velocity()   机器人坐标系实际速度      │
│ blocked         │ → is_blocked()           是否被障碍物阻挡          │
│ block_x, block_y│ → get_nearest_obstacle() 最近障碍物世界坐标        │
│ block_reason    │                          阻挡原因                 │
│ task_status     │ → get_navigation_status() 导航任务状态             │
│ task_type       │                          导航类型                 │
│ target_dist     │                          剩余路径距离              │
│ emergency       │ → is_emergency()         急停按钮状态              │
│ battery_level   │ → get_battery_level()    电池电量                 │
│ confidence      │ → get_loc_confidence()   定位置信度               │
│ current_lock    │ → get_control_owner()    当前控制权所有者          │
│ create_on       │                          推送时间戳               │
└─────────────────┴──────────────────────────────────────────────────┘
"""
import math
import time
import threading
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from config import ROBOT_MAX_LINEAR_VEL, ROBOT_MAX_ANGULAR_VEL

logger = logging.getLogger("RobotAPI")


# =============================================================================
# 数据结构定义
# =============================================================================
@dataclass
class RobotPose:
    """机器人在世界坐标系中的位姿"""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0       # 朝向角 (rad)
    timestamp: float = 0.0

@dataclass
class RobotVelocity:
    """机器人在机器人坐标系中的速度"""
    vx: float = 0.0          # 前向线速度 (m/s)
    vy: float = 0.0          # 横向线速度 (m/s)
    omega: float = 0.0       # 角速度 (rad/s)
    timestamp: float = 0.0

@dataclass
class LidarBeam:
    """单个LiDAR光束"""
    angle: float = 0.0
    dist: float = 0.0
    rssi: float = 0.0
    valid: bool = False

@dataclass
class LidarScan:
    """一帧完整的LiDAR扫描数据"""
    beams: List[LidarBeam] = field(default_factory=list)
    device_name: str = ""
    install_x: float = 0.0
    install_yaw: float = 0.0
    timestamp: float = 0.0

@dataclass
class CameraFrame:
    """一帧相机数据"""
    color_image: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None
    camera_name: str = ""
    timestamp: float = 0.0

@dataclass
class Obstacle:
    """障碍物信息"""
    x: float = 0.0
    y: float = 0.0
    distance: float = 0.0
    reason: int = -1          # 阻挡原因编号

@dataclass
class NavigationResult:
    """导航结果"""
    success: bool = False
    status: str = ""          # "idle"/"running"/"completed"/"failed"/"canceled"
    target_dist: float = 0.0  # 剩余路径长度 (m)
    task_type: int = 0        # 导航类型


# =============================================================================
# 机器人API类
# =============================================================================
class RobotAPI:
    """
    基于 AGVManager SDK + 4B65 推送的机器人接口。

    数据流:
      [AGV硬件] --19301推送--> [_on_push_4B65回调] --> [_state缓存]
                                                           ↑ 读取
      [主循环] ----调用----> get_robot_pose() 等方法 --------┘
    """

    def __init__(self, robot_ip: str = "192.168.1.100"):
        """
        初始化连接并注册推送回调。

        参数:
            robot_ip: 机器人IP地址
                [需适配] 替换为你的机器人实际IP
        """
        # 推送数据缓存 + 线程锁
        self._state: Dict = {}
        self._state_lock = threading.Lock()
        self._last_push_time: float = 0.0
        self._push_count: int = 0
        self._connected = False

        # ---------------------------------------------------------------
        # [需适配] 初始化 AGVManager 连接
        #
        # import AGVManager  # 或 from agv_client import AGVClient
        #
        # # 连接控制端口 (用于发送指令)
        # self._client = AGVManager.connect(robot_ip)
        #
        # # 连接推送端口 19301 并注册 4B65 回调
        # self._push_client = AGVManager.connect_push(robot_ip, port=19301)
        # self._push_client.on("4B65", self._on_push_4B65)
        #
        # # [建议] 设置推送间隔 50ms (20Hz)，与主循环匹配
        # # 只请求我们需要的字段，减少带宽和解析开销
        # self._push_client.configure(
        #     interval_ms=50,
        #     fields=[
        #         "x", "y", "angle",           # 位姿
        #         "vx", "vy", "w",              # 速度
        #         "blocked", "block_x", "block_y", "block_reason",
        #         "task_status", "task_type", "target_dist",
        #         "emergency", "battery_level",
        #         "confidence", "loc_state",
        #         "current_lock",
        #         "create_on",
        #     ]
        # )
        # ---------------------------------------------------------------

        logger.info("[RobotAPI] 初始化完成，等待 4B65 推送数据...")

    # =====================================================================
    # 4B65 推送回调 (在SDK接收线程中执行)
    # =====================================================================
    def _on_push_4B65(self, result):
        """
        4B65 推送数据的回调函数。

        注册方式:  self._push_client.on("4B65", self._on_push_4B65)

        参数:
            result: SDK传入的响应字典
                result["data"] 或 result 本身包含所有推送字段
        """
        try:
            data = result.get("data", result)

            with self._state_lock:
                self._state.update(data)
                self._last_push_time = time.time()
                self._push_count += 1

            if not self._connected:
                self._connected = True
                logger.info("[RobotAPI] 首次收到推送数据，连接就绪")

        except Exception as e:
            logger.error(f"[RobotAPI] 推送回调异常: {e}")

    def wait_for_data(self, timeout: float = 10.0) -> bool:
        """
        等待首次推送数据到达。在主循环开始前调用。

        返回: True=数据已到达, False=超时
        """
        start = time.time()
        while time.time() - start < timeout:
            if self._connected and self._push_count > 0:
                logger.info(f"[RobotAPI] 数据就绪 (已收到 {self._push_count} 帧)")
                return True
            time.sleep(0.1)
        logger.error(f"[RobotAPI] 等待推送数据超时 ({timeout}s)")
        return False

    # =====================================================================
    # 位姿与速度 — 直接从推送缓存读取，零延迟
    # =====================================================================
    def get_robot_pose(self) -> RobotPose:
        """
        获取机器人世界坐标位姿。

        数据来源: 4B65 推送的 x, y, angle 字段
        注意: angle 单位已经是 rad，无需转换
        """
        with self._state_lock:
            return RobotPose(
                x=self._state.get("x", 0.0),
                y=self._state.get("y", 0.0),
                theta=self._state.get("angle", 0.0),
                timestamp=self._last_push_time,
            )

    def get_robot_velocity(self) -> RobotVelocity:
        """
        获取机器人坐标系下的实际速度。

        数据来源: 4B65 推送的 vx, vy, w 字段
        注意: 这是"实际速度"(vx/vy/w)，不是"指令速度"(r_vx/r_vy/r_w)
        """
        with self._state_lock:
            return RobotVelocity(
                vx=self._state.get("vx", 0.0),
                vy=self._state.get("vy", 0.0),
                omega=self._state.get("w", 0.0),
                timestamp=self._last_push_time,
            )

    # =====================================================================
    # 障碍物信息 — 从推送缓存读取
    # =====================================================================
    def get_nearest_obstacle(self) -> Optional[Obstacle]:
        """
        获取最近障碍物信息。

        数据来源: 4B65 推送的 blocked, block_x, block_y, block_reason

        返回: Obstacle 对象，如果未被阻挡返回 None
        """
        with self._state_lock:
            if not self._state.get("blocked", False):
                return None

            bx = self._state.get("block_x", 0.0)
            by = self._state.get("block_y", 0.0)
            rx = self._state.get("x", 0.0)
            ry = self._state.get("y", 0.0)

            return Obstacle(
                x=bx, y=by,
                distance=math.hypot(bx - rx, by - ry),
                reason=self._state.get("block_reason", -1),
            )

    def is_blocked(self) -> bool:
        """机器人当前是否被障碍物阻挡"""
        with self._state_lock:
            return self._state.get("blocked", False)

    # =====================================================================
    # 安全状态 — 从推送缓存读取
    # =====================================================================
    def is_emergency(self) -> bool:
        """急停按钮是否按下"""
        with self._state_lock:
            return self._state.get("emergency", False)

    def get_battery_level(self) -> float:
        """电池电量 [0, 1]"""
        with self._state_lock:
            return self._state.get("battery_level", 1.0)

    def get_loc_confidence(self) -> float:
        """定位置信度 [0, 1]"""
        with self._state_lock:
            return self._state.get("confidence", 1.0)

    def get_control_owner(self) -> Optional[Dict]:
        """
        获取当前控制权所有者。

        用于诊断 "control is preempted" (ret_code: 40020) 错误。
        发送开环指令前，需确认控制权在自己手上。
        """
        with self._state_lock:
            return self._state.get("current_lock", None)

    def is_data_fresh(self, max_age: float = 0.5) -> bool:
        """推送数据是否新鲜 (未超时)"""
        return (time.time() - self._last_push_time) < max_age

    # =====================================================================
    # LiDAR数据获取 — 需要主动请求 (不在4B65推送中)
    # =====================================================================
    def get_lidar_scans(self) -> List[LidarScan]:
        """
        [需适配] 获取LiDAR点云数据。

        LiDAR点云不在4B65推送中，需要通过其他接口获取。
        数据格式参考你之前提供的 point_cloud.txt。
        """
        # ---------------------------------------------------------------
        # [需适配] 替换为实际调用
        #
        # raw_data = self._client.get_point_cloud()
        # ---------------------------------------------------------------

        raw_data = None  # [需适配] 替换为实际调用结果
        if raw_data is None:
            raise NotImplementedError("请实现 get_lidar_scans()")

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
    # 相机数据获取 — 需要主动请求
    # =====================================================================
    def get_camera_frame(self, camera_name: str) -> CameraFrame:
        """[需适配] 获取相机帧。Phase 2+ 实现。"""
        raise NotImplementedError(f"Phase 2+: get_camera_frame('{camera_name}')")

    # =====================================================================
    # 运动控制 — 主动发送开环速度指令
    # =====================================================================
    def send_velocity(self, linear_vel: float, angular_vel: float):
        """
        发送开环速度指令。

        对应你的API字段:
          vx = linear_vel    前向速度 (m/s)
          vy = 0             差速底盘横向为0
          w  = angular_vel   角速度 (rad/s)，逆时针为正
          duration = 200     200ms看门狗，程序崩溃后自动停

        ★ 首次调用前需要先 acquire_control() 获取开环控制权，
          否则会报 "control is preempted" (ret_code: 40020)
        """
        vx = max(-ROBOT_MAX_LINEAR_VEL, min(ROBOT_MAX_LINEAR_VEL, linear_vel))
        w = max(-ROBOT_MAX_ANGULAR_VEL, min(ROBOT_MAX_ANGULAR_VEL, angular_vel))

        # ---------------------------------------------------------------
        # [需适配] 替换为你的 AGVManager 开环速度指令调用
        #
        # self._client.send_open_loop(vx=vx, vy=0, w=w, duration=200)
        # ---------------------------------------------------------------
        raise NotImplementedError(
            f"请替换为AGVManager开环速度调用: vx={vx:.3f}, vy=0, w={w:.3f}, duration=200"
        )

    def stop(self):
        """紧急停止"""
        try:
            self.send_velocity(0.0, 0.0)
        except NotImplementedError:
            pass

    # =====================================================================
    # 控制权管理 — 解决 "control is preempted" 问题
    # =====================================================================
    def acquire_control(self) -> bool:
        """
        [需适配] 获取开环控制权。

        在发送开环速度指令之前必须调用此方法。
        你的机器人有控制权优先级机制，导航模式占用时开环指令会被拒绝。

        在API文档中搜索以下关键词:
          - cmd_id "1060" (推送中 current_lock 字段引用了此接口)
          - "control" / "preempt" / "lock" / "acquire"
          - ret_code 40020

        返回: True=成功获取, False=失败
        """
        # ---------------------------------------------------------------
        # [需适配]
        #
        # 方式A: 直接请求控制权
        #   result = self._client.acquire_control()
        #   return result.ret_code == 0
        #
        # 方式B: 切换到手动/开环模式
        #   result = self._client.set_control_mode("open_loop")
        #   return result.ret_code == 0
        #
        # 方式C: 取消所有任务让控制权自动释放
        #   self._client.cancel_all_tasks()
        #   time.sleep(0.5)
        #   return True
        # ---------------------------------------------------------------
        raise NotImplementedError(
            "请实现 acquire_control()。"
            "在API文档中搜索 cmd_id 1060 或 'control'/'preempt'/'lock'"
        )

    def release_control(self):
        """
        [需适配] 释放开环控制权。
        在切换到导航模式或程序退出时调用。
        """
        # ---------------------------------------------------------------
        # [需适配]
        # self._client.release_control()
        # ---------------------------------------------------------------
        pass

    # =====================================================================
    # 导航 — 主动发送指令 + 从推送缓存读取状态
    # =====================================================================
    def navigate_to(self, x: float, y: float, theta: float) -> bool:
        """
        [需适配] 发送导航目标点 (非阻塞)。

        发送后，导航状态会通过 4B65 推送的 task_status / task_type /
        target_dist 字段实时更新，用 get_navigation_status() 读取。

        注意: 发导航指令前需先 release_control() 释放开环控制权，
        否则导航系统无法接管底盘。
        """
        # ---------------------------------------------------------------
        # [需适配]
        # self.release_control()
        # result = self._client.navigate_to_point(x=x, y=y, angle=theta)
        # return result.ret_code == 0
        # ---------------------------------------------------------------
        raise NotImplementedError("请实现 navigate_to()")

    def get_navigation_status(self) -> NavigationResult:
        """
        获取导航状态。无需主动请求，直接读推送缓存。

        task_status 映射:
          0=NONE, 2=RUNNING, 3=SUSPENDED,
          4=COMPLETED, 5=FAILED, 6=CANCELED
        """
        STATUS_MAP = {
            0: "idle", 2: "running", 3: "suspended",
            4: "completed", 5: "failed", 6: "canceled",
        }

        with self._state_lock:
            task_status = self._state.get("task_status", 0)
            return NavigationResult(
                success=(task_status == 4),
                status=STATUS_MAP.get(task_status, "idle"),
                target_dist=self._state.get("target_dist", 0.0),
                task_type=self._state.get("task_type", 0),
            )

    def cancel_navigation(self):
        """
        [需适配] 取消当前导航任务。
        取消后需重新 acquire_control() 才能发开环指令。
        """
        # ---------------------------------------------------------------
        # [需适配]
        # self._client.cancel_task()
        # ---------------------------------------------------------------
        raise NotImplementedError("请实现 cancel_navigation()")

    # =====================================================================
    # 地图 — 使用预处理的 .npy 文件
    # =====================================================================
    def get_global_map(self) -> Optional[Dict]:
        """不需要运行时调用，用预处理的 .npy 文件代替"""
        return None

    # =====================================================================
    # 坐标变换工具 (纯数学，不需要修改)
    # =====================================================================
    def robot_to_world(self, local_x: float, local_y: float,
                       pose: RobotPose) -> Tuple[float, float]:
        """将机器人坐标系中的点转换到世界坐标系"""
        cos_t = math.cos(pose.theta)
        sin_t = math.sin(pose.theta)
        return (pose.x + local_x * cos_t - local_y * sin_t,
                pose.y + local_x * sin_t + local_y * cos_t)

    def world_to_robot(self, world_x: float, world_y: float,
                       pose: RobotPose) -> Tuple[float, float]:
        """将世界坐标系中的点转换到机器人坐标系"""
        dx = world_x - pose.x
        dy = world_y - pose.y
        cos_t = math.cos(pose.theta)
        sin_t = math.sin(pose.theta)
        return (dx * cos_t + dy * sin_t,
                -dx * sin_t + dy * cos_t)

    def disconnect(self):
        """断开连接，释放资源"""
        try:
            self.release_control()
        except:
            pass
        # ---------------------------------------------------------------
        # [需适配]
        # self._client.disconnect()
        # self._push_client.disconnect()
        # ---------------------------------------------------------------
        logger.info("[RobotAPI] 已断开连接")

    # =====================================================================
    # 调试工具
    # =====================================================================
    def print_state_summary(self):
        """打印当前推送数据摘要，调试用"""
        with self._state_lock:
            s = self._state
            age = time.time() - self._last_push_time

            print(f"--- 机器人状态 (延迟: {age*1000:.0f}ms, 累计: {self._push_count}帧) ---")
            print(f"  位姿:  x={s.get('x',0):.3f}  y={s.get('y',0):.3f}  "
                  f"angle={s.get('angle',0):.3f}rad "
                  f"({math.degrees(s.get('angle',0)):.1f}°)")
            print(f"  速度:  vx={s.get('vx',0):.3f}  vy={s.get('vy',0):.3f}  "
                  f"w={s.get('w',0):.3f}")
            print(f"  障碍:  blocked={s.get('blocked',False)}  "
                  f"pos=({s.get('block_x','?')}, {s.get('block_y','?')})")
            print(f"  导航:  status={s.get('task_status',0)}  "
                  f"type={s.get('task_type',0)}  "
                  f"remain={s.get('target_dist',0):.2f}m")
            print(f"  安全:  emergency={s.get('emergency',False)}  "
                  f"battery={s.get('battery_level',0):.0%}  "
                  f"confidence={s.get('confidence',0):.2f}")
            print(f"  控制:  {s.get('current_lock', '未知')}")