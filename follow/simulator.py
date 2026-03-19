"""
=============================================================================
simulator.py — 离线仿真环境
=============================================================================
不连接真实机器人，模拟所有传感器数据和运动响应，用于:
1. 验证算法逻辑是否正确
2. 调参 (PID增益、跟随距离、VFH阈值等)
3. 可视化跟随行为

使用方式:
  python simulator.py

会弹出一个 matplotlib 动画窗口，显示:
- 灰色点: 静态地图障碍物
- 红色圆: 目标人物 (自动行走)
- 蓝色三角: 机器人
- 绿色线: 跟随路径

依赖: pip install numpy scipy matplotlib
"""
import math
import time
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

# 导入我们的模块
from .config import *
from .robot_api import (
    RobotAPI, RobotPose, RobotVelocity, LidarScan, LidarBeam,
    CameraFrame, NavigationResult, Obstacle
)
from .lidar_processor import LidarProcessor, PersonCandidate
from .sensor_fusion import SensorFusion
from .motion_controller import MotionController
from .state_machine import StateMachine, FollowState


# =============================================================================
# 模拟器配置
# =============================================================================
SIM_DT = 0.05                     # 仿真步长 (s)，对应20Hz
SIM_DURATION = 60.0               # 仿真总时长 (s)
SIM_VISUALIZE = True              # 是否显示可视化

# 目标人物运动路径定义 (一系列航点，目标会依次走过)
# 格式: [(x, y, 停留时间s), ...]
# [需适配] 根据你的地图修改这些航点
TARGET_WAYPOINTS = [
    (-5.0,  1.5, 1.0),    # 起点，停1秒
    (-8.0,  1.5, 0.5),    # 直线向左走
    (-8.0, -1.0, 0.5),    # 向下走
    (-12.0, -1.0, 1.0),   # 继续向左
    (-12.0,  1.5, 0.5),   # 向上走
    (-8.0,  1.5, 0.5),    # 返回
    (-5.0,  1.5, 1.0),    # 回到起点
]
TARGET_SPEED = 0.8                 # 人物行走速度 (m/s)

# 机器人初始位置
ROBOT_INIT_X = -4.0
ROBOT_INIT_Y = 1.5
ROBOT_INIT_THETA = math.pi        # 朝左


# =============================================================================
# 模拟目标人物
# =============================================================================
class SimTarget:
    """模拟一个按航点行走的人物"""
    
    def __init__(self, waypoints, speed):
        self.waypoints = waypoints
        self.speed = speed
        self.x = waypoints[0][0]
        self.y = waypoints[0][1]
        self.vx = 0.0
        self.vy = 0.0
        
        self._wp_idx = 0
        self._wait_timer = waypoints[0][2]  # 初始停留
        self._history = [(self.x, self.y)]
    
    def step(self, dt: float):
        """推进一个时间步"""
        # 如果在等待
        if self._wait_timer > 0:
            self._wait_timer -= dt
            self.vx = 0.0
            self.vy = 0.0
            return
        
        # 当前目标航点
        next_idx = (self._wp_idx + 1) % len(self.waypoints)
        tx, ty, tw = self.waypoints[next_idx]
        
        dx = tx - self.x
        dy = ty - self.y
        dist = math.hypot(dx, dy)
        
        if dist < 0.1:
            # 到达航点
            self._wp_idx = next_idx
            self._wait_timer = tw
            self.vx = 0.0
            self.vy = 0.0
            return
        
        # 向目标航点移动
        self.vx = self.speed * dx / dist
        self.vy = self.speed * dy / dist
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        self._history.append((self.x, self.y))


# =============================================================================
# 模拟机器人API
# =============================================================================
class SimRobotAPI(RobotAPI):
    """
    用仿真数据替代真实硬件的 RobotAPI。
    继承 RobotAPI 并实现所有方法。
    """
    
    def __init__(self, map_points: np.ndarray, target: SimTarget):
        # 不调用父类 __init__ (会打印硬件初始化信息)
        self._map_points = map_points
        self._target = target
        
        # 构建地图KD-Tree用于快速LiDAR模拟
        from scipy.spatial import cKDTree
        self._map_tree = cKDTree(map_points)
        
        # 机器人状态
        self.pose = RobotPose(
            x=ROBOT_INIT_X, y=ROBOT_INIT_Y,
            theta=ROBOT_INIT_THETA, timestamp=time.time()
        )
        self.velocity = RobotVelocity()
        
        # 运动指令缓存
        self._cmd_linear = 0.0
        self._cmd_angular = 0.0
        
        # 导航模拟
        self._nav_goal = None
        self._nav_status = "idle"
        
        # 轨迹记录
        self.trajectory = [(self.pose.x, self.pose.y)]
    
    def get_robot_pose(self) -> RobotPose:
        self.pose.timestamp = time.time()
        return self.pose
    
    def get_robot_velocity(self) -> RobotVelocity:
        return self.velocity
    
    def get_lidar_scans(self) -> List[LidarScan]:
        """
        模拟LiDAR扫描: 快速版本。
        
        用KD-Tree查找LiDAR附近的地图点，然后按角度分桶得到每个方向
        的最近距离。比逐光束光线投射快100倍+。
        """
        scans = []
        
        lidar_configs = [
            ("front", LIDAR_FRONT_X, 0.0),
            ("rear",  LIDAR_REAR_X, 180.0),
        ]
        
        for name, install_x, install_yaw in lidar_configs:
            scan = LidarScan(
                device_name=name, install_x=install_x,
                install_yaw=install_yaw, timestamp=time.time(),
            )
            
            # LiDAR在世界坐标系中的位置
            cos_t = math.cos(self.pose.theta)
            sin_t = math.sin(self.pose.theta)
            lidar_wx = self.pose.x + install_x * cos_t
            lidar_wy = self.pose.y + install_x * sin_t
            lidar_world_yaw = self.pose.theta + math.radians(install_yaw)
            
            # 用KD-Tree查找 LIDAR_MAX_RANGE 内的所有地图点
            nearby_idx = self._map_tree.query_ball_point(
                [lidar_wx, lidar_wy], r=min(15.0, LIDAR_MAX_RANGE)
            )
            
            # 计算各点相对于LiDAR的角度和距离
            beam_angles = np.arange(-90, 90.5, 0.5)
            beam_dists = np.full(len(beam_angles), LIDAR_MAX_RANGE)
            
            if nearby_idx:
                nearby_pts = self._map_points[nearby_idx]
                dx = nearby_pts[:, 0] - lidar_wx
                dy = nearby_pts[:, 1] - lidar_wy
                dists = np.sqrt(dx**2 + dy**2)
                angles_world = np.arctan2(dy, dx)
                
                # 转为LiDAR局部角度
                angles_local = np.degrees(angles_world - lidar_world_yaw)
                # 归一化到 [-180, 180]
                angles_local = (angles_local + 180) % 360 - 180
                
                # 将地图点分配到最近的beam
                for i, ba in enumerate(beam_angles):
                    mask = np.abs(angles_local - ba) < 0.5  # ±0.5°角分辨率
                    if np.any(mask):
                        beam_dists[i] = np.min(dists[mask])
            
            # 添加目标人物 (模拟为0.2m半径圆柱)
            tdx = self._target.x - lidar_wx
            tdy = self._target.y - lidar_wy
            target_dist = math.hypot(tdx, tdy)
            if 0.1 < target_dist < LIDAR_MAX_RANGE:
                target_angle_world = math.atan2(tdy, tdx)
                target_angle_local = math.degrees(target_angle_world - lidar_world_yaw)
                target_angle_local = (target_angle_local + 180) % 360 - 180
                # 人物在LiDAR视角中的角宽度
                angular_width = math.degrees(math.atan2(0.2, target_dist))
                
                for i, ba in enumerate(beam_angles):
                    if abs(ba - target_angle_local) < angular_width:
                        hit_dist = target_dist - 0.15  # 粗略近似
                        if 0 < hit_dist < beam_dists[i]:
                            beam_dists[i] = hit_dist
            
            # 构建beam列表
            for i, ba in enumerate(beam_angles):
                d = beam_dists[i]
                if d < LIDAR_MAX_RANGE:
                    d += np.random.normal(0, 0.01)  # 1cm噪声
                scan.beams.append(LidarBeam(
                    angle=ba, dist=max(0.01, d),
                    rssi=100.0 if d < LIDAR_MAX_RANGE else 0.0,
                    valid=d < LIDAR_MAX_RANGE,
                ))
            
            scans.append(scan)
        
        return scans
    
    def get_camera_frame(self, camera_name: str) -> CameraFrame:
        # 仿真中不使用真实图像，返回空帧
        # 视觉检测由 SimVisionOverride 模块模拟
        return CameraFrame(camera_name=camera_name, timestamp=time.time())
    
    def send_velocity(self, linear_vel: float, angular_vel: float):
        self._cmd_linear = np.clip(linear_vel, -ROBOT_MAX_LINEAR_VEL, ROBOT_MAX_LINEAR_VEL)
        self._cmd_angular = np.clip(angular_vel, -ROBOT_MAX_ANGULAR_VEL, ROBOT_MAX_ANGULAR_VEL)
        self._nav_goal = None  # 直接控制时取消导航
        self._nav_status = "idle"
    
    def stop(self):
        self._cmd_linear = 0.0
        self._cmd_angular = 0.0
    
    def navigate_to(self, x, y, theta) -> bool:
        self._nav_goal = (x, y, theta)
        self._nav_status = "moving"
        return True
    
    def get_navigation_status(self) -> NavigationResult:
        if self._nav_goal is None:
            return NavigationResult(success=False, status="idle")
        
        dist = math.hypot(self.pose.x - self._nav_goal[0],
                          self.pose.y - self._nav_goal[1])
        if dist < 0.3:
            return NavigationResult(success=True, status="reached")
        return NavigationResult(success=False, status="moving")
    
    def cancel_navigation(self):
        self._nav_goal = None
        self._nav_status = "idle"
    
    def get_global_map(self):
        return None  # 使用预加载的 npy
    
    def get_nearest_obstacle(self):
        return None
    
    def sim_step(self, dt: float):
        """
        推进机器人状态一个时间步。
        
        差速底盘运动学:
          x(t+dt) = x(t) + v * cos(θ) * dt
          y(t+dt) = y(t) + v * sin(θ) * dt
          θ(t+dt) = θ(t) + ω * dt
        """
        v = self._cmd_linear
        omega = self._cmd_angular
        
        # 导航模式: 简单模拟向目标点移动
        if self._nav_goal is not None:
            gx, gy, _ = self._nav_goal
            dx = gx - self.pose.x
            dy = gy - self.pose.y
            dist = math.hypot(dx, dy)
            
            if dist > 0.3:
                desired_angle = math.atan2(dy, dx)
                angle_err = desired_angle - self.pose.theta
                while angle_err > math.pi: angle_err -= 2 * math.pi
                while angle_err < -math.pi: angle_err += 2 * math.pi
                
                omega = np.clip(2.0 * angle_err, -ROBOT_MAX_ANGULAR_VEL, ROBOT_MAX_ANGULAR_VEL)
                if abs(angle_err) < 0.3:
                    v = min(0.5, dist)
                else:
                    v = 0.0
            else:
                v = 0.0
                omega = 0.0
                self._nav_status = "reached"
        
        # 运动学更新
        self.pose.x += v * math.cos(self.pose.theta) * dt
        self.pose.y += v * math.sin(self.pose.theta) * dt
        self.pose.theta += omega * dt
        
        # 归一化角度
        while self.pose.theta > math.pi: self.pose.theta -= 2 * math.pi
        while self.pose.theta < -math.pi: self.pose.theta += 2 * math.pi
        
        self.velocity = RobotVelocity(vx=v, omega=omega, timestamp=time.time())
        self.trajectory.append((self.pose.x, self.pose.y))


# =============================================================================
# 仿真主循环
# =============================================================================
def run_simulation():
    import os
    
    print("=" * 60)
    print("  离线仿真 — 人物跟随系统")
    print("=" * 60)
    
    # --- 加载地图 ---
    npy_path = MAP_POINTS_NPY_PATH
    if not os.path.exists(npy_path):
        # 尝试在当前目录下找
        alt_path = os.path.join(os.path.dirname(__file__), "maps", "map_points.npy")
        if os.path.exists(alt_path):
            npy_path = alt_path
        else:
            print(f"错误: 找不到地图文件 {npy_path}")
            print("请先运行: python map_preprocessor.py log.txt ./maps/")
            return
    
    map_points = np.load(npy_path)
    print(f"地图已加载: {len(map_points):,} 个障碍物点")
    
    # --- 创建模拟目标 ---
    target = SimTarget(TARGET_WAYPOINTS, TARGET_SPEED)
    print(f"目标人物起点: ({target.x:.1f}, {target.y:.1f})")
    
    # --- 创建模拟机器人API ---
    sim_api = SimRobotAPI(map_points, target)
    print(f"机器人起点: ({sim_api.pose.x:.1f}, {sim_api.pose.y:.1f})")
    
    # --- 初始化算法模块 ---
    lidar_proc = LidarProcessor()
    lidar_proc.load_map_from_npy(npy_path)
    
    fusion = SensorFusion()
    motion_ctrl = MotionController(sim_api)
    state_machine = StateMachine(sim_api)
    
    # 初始化EKF (用目标初始位置)
    fusion.initialize(target.x, target.y, time.time())
    state_machine.start()
    
    # --- 数据记录 ---
    log_time = []
    log_dist = []
    log_state = []
    
    # --- 仿真循环 ---
    num_steps = int(SIM_DURATION / SIM_DT)
    print(f"\n开始仿真: {SIM_DURATION}s, {num_steps} 步...")
    
    for step in range(num_steps):
        t = step * SIM_DT
        
        # 1. 推进目标人物
        target.step(SIM_DT)
        
        # 2. 获取传感器数据
        robot_pose = sim_api.get_robot_pose()
        
        try:
            scans = sim_api.get_lidar_scans()
            lidar_candidates = lidar_proc.process(scans, robot_pose)
            obstacle_sectors = lidar_proc.get_obstacle_sectors(scans)
        except Exception as e:
            lidar_candidates = []
            obstacle_sectors = np.full(72, LIDAR_MAX_RANGE)
        
        # 3. 传感器融合
        # 在仿真中，直接模拟视觉检测: 如果目标在视野内，提供精确观测
        dx = target.x - robot_pose.x
        dy = target.y - robot_pose.y
        angle_to_target = math.atan2(dy, dx) - robot_pose.theta
        while angle_to_target > math.pi: angle_to_target -= 2 * math.pi
        while angle_to_target < -math.pi: angle_to_target += 2 * math.pi
        
        # 模拟视觉 FOV ±45° (头部相机)
        if abs(angle_to_target) < math.radians(45):
            from vision_detector import PersonDetection
            det = PersonDetection(
                world_x=target.x + np.random.normal(0, 0.05),
                world_y=target.y + np.random.normal(0, 0.05),
                depth=math.hypot(dx, dy),
                is_target=True,
                timestamp=time.time(),
            )
            fusion.update_with_vision(det)
        
        # LiDAR关联
        if lidar_candidates:
            matched = fusion.associate_lidar_candidates(lidar_candidates)
            if matched:
                fusion.update_with_lidar(matched)
        
        # 无观测时预测
        target_state = fusion.get_target_state()
        if not target_state.is_valid:
            fusion.predict_only(time.time())
            target_state = fusion.get_target_state()
        
        # 4. 状态机
        current_state = state_machine.update(target_state, robot_pose)
        
        # 5. 控制
        if current_state == FollowState.DIRECT_FOLLOW and target_state.is_valid:
            lv, av = motion_ctrl.follow_target(target_state, robot_pose, obstacle_sectors)
            sim_api.send_velocity(lv, av)
        elif current_state == FollowState.SEARCH:
            direction = state_machine.get_search_direction()
            lv, av = motion_ctrl.rotate_search(direction)
            sim_api.send_velocity(lv, av)
        elif current_state == FollowState.LOST:
            sim_api.stop()
        
        # 6. 推进机器人
        sim_api.sim_step(SIM_DT)
        
        # 记录数据
        dist = math.hypot(target.x - robot_pose.x, target.y - robot_pose.y)
        log_time.append(t)
        log_dist.append(dist)
        log_state.append(current_state.name)
        
        # 每5秒打印一次
        if step % (int(5.0 / SIM_DT)) == 0:
            print(f"  t={t:5.1f}s | {current_state.name:15s} | "
                  f"距离={dist:.2f}m | "
                  f"机器人({robot_pose.x:.1f},{robot_pose.y:.1f}) "
                  f"目标({target.x:.1f},{target.y:.1f})")
    
    print("\n仿真完成!")
    
    # --- 统计 ---
    dists = np.array(log_dist)
    print(f"\n跟随距离统计:")
    print(f"  平均: {dists.mean():.2f}m (目标: {FOLLOW_DISTANCE:.2f}m)")
    print(f"  标准差: {dists.std():.2f}m")
    print(f"  最小: {dists.min():.2f}m, 最大: {dists.max():.2f}m")
    
    # --- 可视化 ---
    if SIM_VISUALIZE:
        try:
            visualize_results(map_points, sim_api, target, log_time, log_dist)
        except ImportError:
            print("matplotlib未安装，跳过可视化。 pip install matplotlib")


def visualize_results(map_points, sim_api, target, log_time, log_dist):
    """绘制仿真结果"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = ['DejaVu Sans']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- 左图: 平面轨迹 ---
    ax = axes[0]
    
    # 地图 (采样显示，太多点会卡)
    step = max(1, len(map_points) // 5000)
    ax.scatter(map_points[::step, 0], map_points[::step, 1],
               s=0.5, c='#cccccc', alpha=0.5, label='Map obstacles')
    
    # 目标轨迹
    th = np.array(target._history)
    ax.plot(th[:, 0], th[:, 1], 'r-', linewidth=1.5, alpha=0.6, label='Target path')
    ax.plot(th[-1, 0], th[-1, 1], 'ro', markersize=8)
    
    # 机器人轨迹
    rt = np.array(sim_api.trajectory)
    ax.plot(rt[:, 0], rt[:, 1], 'b-', linewidth=1.5, alpha=0.6, label='Robot path')
    ax.plot(rt[-1, 0], rt[-1, 1], 'b^', markersize=10)
    
    # 起点
    ax.plot(rt[0, 0], rt[0, 1], 'bs', markersize=10, label='Robot start')
    ax.plot(th[0, 0], th[0, 1], 'rs', markersize=10, label='Target start')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Following Trajectory')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # --- 右图: 跟随距离随时间变化 ---
    ax2 = axes[1]
    ax2.plot(log_time, log_dist, 'b-', linewidth=1)
    ax2.axhline(y=FOLLOW_DISTANCE, color='g', linestyle='--',
                label=f'Target dist = {FOLLOW_DISTANCE}m')
    ax2.axhline(y=FOLLOW_DISTANCE + FOLLOW_DISTANCE_TOLERANCE,
                color='g', linestyle=':', alpha=0.5)
    ax2.axhline(y=FOLLOW_DISTANCE - FOLLOW_DISTANCE_TOLERANCE,
                color='g', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Distance to target (m)')
    ax2.set_title('Following Distance Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("sim_result.png", dpi=150)
    print("仿真结果已保存: sim_result.png")
    plt.show()


# =============================================================================
# 入口
# =============================================================================
if __name__ == "__main__":
    run_simulation()
