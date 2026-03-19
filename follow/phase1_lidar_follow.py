"""
=============================================================================
phase1_lidar_follow.py — Phase 1 最简版: 纯LiDAR跟随
=============================================================================
这是最小可运行版本，只用 LiDAR 跟随最近的动态物体。
不需要视觉检测、不需要ReID、不需要导航API。

目的: 快速验证硬件接口 (robot_api.py) 是否正确。

只需要实现 robot_api.py 中的 3 个方法:
  1. get_robot_pose()      — 获取位姿
  2. get_lidar_scans()     — 获取LiDAR点云
  3. send_velocity()       — 发送运动指令

使用方式:
  1. 在 robot_api.py 中实现上述3个方法
  2. 确保地图已预处理: python map_preprocessor.py log.txt ./maps/
  3. 让一个人站在机器人前方1-2米
  4. 运行: python phase1_lidar_follow.py
  5. 机器人会跟随最近的动态物体

安全提示:
  - 首次测试时，建议把 ROBOT_MAX_LINEAR_VEL 设到 0.3 以下
  - 随时准备好 Ctrl+C 紧急停止
"""
import math
import time
import signal
import sys
import numpy as np

from .config import (
    ROBOT_MAX_LINEAR_VEL, ROBOT_MAX_ANGULAR_VEL,
    FOLLOW_DISTANCE, MAP_POINTS_NPY_PATH,
    OBSTACLE_DANGER_DIST, ROBOT_RADIUS,
)
from .robot_api import RobotAPI
from .lidar_processor import LidarProcessor

# =============================================================================
# 安全限速 (Phase 1 建议保守值)
# =============================================================================
PHASE1_MAX_LINEAR = 0.3    # 最大线速度 (m/s) — 先慢一点
PHASE1_MAX_ANGULAR = 0.8   # 最大角速度 (rad/s)
PHASE1_FOLLOW_DIST = 1.2   # 跟随距离 (m) — 比最终值稍远一些

# =============================================================================
# 全局退出
# =============================================================================
_shutdown = False

def _signal_handler(sig, frame):
    global _shutdown
    _shutdown = True
    print("\n[紧急停止] Ctrl+C")

signal.signal(signal.SIGINT, _signal_handler)


# =============================================================================
# 简化版控制器 (不依赖 motion_controller.py 的完整PID+VFH)
# =============================================================================
def simple_follow_control(target_local_x: float, target_local_y: float,
                          obstacle_min_dist: float
                          ) -> tuple:
    """
    最简单的跟随控制: 比例控制。
    
    参数:
        target_local_x: 目标在机器人坐标系中的X坐标 (前方为正)
        target_local_y: 目标在机器人坐标系中的Y坐标 (左方为正)
        obstacle_min_dist: 前方最近障碍物距离
    
    返回:
        (linear_vel, angular_vel)
    """
    dist = math.hypot(target_local_x, target_local_y)
    angle = math.atan2(target_local_y, target_local_x)  # 目标方位角
    
    # --- 角速度: 比例控制，让机器人朝向目标 ---
    angular_vel = 1.2 * angle  # Kp = 1.2
    angular_vel = max(-PHASE1_MAX_ANGULAR, min(PHASE1_MAX_ANGULAR, angular_vel))
    
    # --- 线速度: 比例控制，保持跟随距离 ---
    dist_error = dist - PHASE1_FOLLOW_DIST
    linear_vel = 0.6 * dist_error  # Kp = 0.6
    
    # 目标在后方 (角度 > 90°) 时不前进，只转身
    if abs(angle) > math.pi / 2:
        linear_vel = 0.0
    
    # 距离太近时后退
    if dist < PHASE1_FOLLOW_DIST * 0.5:
        linear_vel = -0.1
    
    # 转弯时减速
    if abs(angle) > 0.3:
        linear_vel *= 0.5
    
    # 障碍物减速
    clearance = obstacle_min_dist - ROBOT_RADIUS
    if clearance < OBSTACLE_DANGER_DIST:
        linear_vel = 0.0
        print("  [!] 障碍物过近，停止前进")
    elif clearance < 0.6:
        factor = clearance / 0.6
        linear_vel *= factor
    
    # 限幅
    linear_vel = max(-PHASE1_MAX_LINEAR, min(PHASE1_MAX_LINEAR, linear_vel))
    
    return linear_vel, angular_vel


def get_front_min_obstacle(obstacle_sectors: np.ndarray) -> float:
    """获取正前方 ±30° 范围内的最近障碍物距离"""
    num = len(obstacle_sectors)
    sector_size = 360.0 / num
    check_range = int(30.0 / sector_size)
    
    min_dist = 999.0
    for offset in range(-check_range, check_range + 1):
        idx = offset % num
        if obstacle_sectors[idx] < min_dist:
            min_dist = obstacle_sectors[idx]
    
    return min_dist


# =============================================================================
# 主函数
# =============================================================================
def main():
    global _shutdown
    
    print("=" * 50)
    print("  Phase 1: 纯LiDAR跟随 (最简版)")
    print("=" * 50)
    print(f"  跟随距离: {PHASE1_FOLLOW_DIST}m")
    print(f"  最大线速度: {PHASE1_MAX_LINEAR}m/s")
    print(f"  最大角速度: {PHASE1_MAX_ANGULAR}rad/s")
    print()
    
    # 1. 初始化
    robot_api = RobotAPI()
    lidar_proc = LidarProcessor()
    
    # 2. 加载地图
    import os
    if MAP_POINTS_NPY_PATH and os.path.exists(MAP_POINTS_NPY_PATH):
        lidar_proc.load_map_from_npy(MAP_POINTS_NPY_PATH)
    else:
        print("[警告] 未找到预处理地图，LiDAR差分将不可用")
        print("  所有LiDAR点都会被视为动态物体")
        print("  建议先运行: python map_preprocessor.py log.txt ./maps/")
    
    robot_api.wait_for_data()
    print(22)
    print("\n请让目标人物站在机器人前方 1~2 米处")
    print("按 Enter 开始跟随 (Ctrl+C 紧急停止)...")
    input()
    
    print("开始跟随!\n")
    
    loop_count = 0
    
    
    while not _shutdown:
        loop_start = time.time()
        loop_count += 1
        
        try:
            robot_api.get_state()
            # --- 获取位姿 ---
            print(1)
            pose = robot_api.get_robot_pose()
            
            # --- 获取LiDAR数据并处理 ---
            scans = robot_api.get_lidar_scans()
            
            if not scans:
                print("  [!] 未收到LiDAR数据")
                time.sleep(0.1)
                continue
            
            # 提取动态人物候选
            candidates = lidar_proc.process(scans, pose)
            
            # 获取障碍物扇区
            obstacle_sectors = lidar_proc.get_obstacle_sectors(scans)
            front_min = get_front_min_obstacle(obstacle_sectors)
            
            # --- 选择跟随目标: 取最近的候选 ---
            if not candidates:
                # 没有检测到动态物体
                robot_api.send_velocity(0.0, 0.0)
                print("本周期没有检测到动态物体")
                if loop_count % 20 == 0:
                    print(f"  t={loop_count/20:.1f}s | 未检测到动态物体, 停止")
                time.sleep(0.05)
                continue
            
            # 选择最近的候选 (按机器人坐标系中的距离)
            nearest = min(candidates,
                          key=lambda c: math.hypot(c.local_x, c.local_y))
            
            dist = math.hypot(nearest.local_x, nearest.local_y)
            angle = math.degrees(math.atan2(nearest.local_y, nearest.local_x))
            
            # --- 计算控制量 ---
            lv, av = simple_follow_control(
                nearest.local_x, nearest.local_y, front_min
            )
            
            # --- 发送指令 ---
            robot_api.send_velocity(lv, av)
            
            # --- 日志 ---
            if loop_count % 5 == 0:  # 每0.5秒打印一次
                print(f"  t={loop_count*0.1:.1f}s | "
                      f"候选={len(candidates)} | "
                      f"最近: 距离={dist:.2f}m 方位={angle:.1f}° | "
                      f"指令: v={lv:.2f} ω={av:.2f} | "
                      f"前方障碍={front_min:.2f}m")
        
        except NotImplementedError as e:
            print(f"\n[错误] 硬件接口未实现: {e}")
            print("请在 robot_api.py 中实现对应方法后重试")
            break
        
        except Exception as e:
            print(f"  [异常] {e}")
            try:
                robot_api.stop()
            except:
                pass
        
        # 控制频率 ~10Hz
        elapsed = time.time() - loop_start
        sleep_time = 0.1 - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    # 清理
    print("\n正在停止...")
    try:
        robot_api.stop()
    except:
        pass
    robot_api.disconnect()
    print("已退出")


if __name__ == "__main__":
    main()
