"""
=============================================================================
main.py — 主入口，跟随系统主循环
=============================================================================
这是整个系统的入口文件。它将所有模块串联起来，执行以下循环:

每个周期 (约 10Hz):
  1. 获取机器人位姿
  2. 获取LiDAR数据 → 处理 → 提取动态人物候选
  3. 获取相机数据 → 检测人物 → ReID匹配
  4. 传感器融合 (EKF) → 更新目标状态
  5. 状态机决策 → 选择控制模式
  6. 运动控制 → 下发指令

使用方式:
  python main.py

首次运行前:
  1. 确保已安装依赖: pip install numpy scipy opencv-python
  2. [可选] 安装YOLO: pip install ultralytics
  3. 在 robot_api.py 中实现你的硬件接口
  4. 在 config.py 中调整参数
"""
import math
import time
import signal
import sys
import logging
import numpy as np
from typing import Optional

from .config import MAIN_LOOP_RATE, LOG_LEVEL, MAP_POINTS_NPY_PATH
from .robot_api import RobotAPI
from .lidar_processor import LidarProcessor
from .vision_detector import VisionDetector
from .sensor_fusion import SensorFusion, TargetState
from .motion_controller import MotionController
from .state_machine import StateMachine, FollowState


# =============================================================================
# 日志配置
# =============================================================================
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Main")


# =============================================================================
# 全局退出标志 (Ctrl+C 优雅退出)
# =============================================================================
_shutdown = False

def _signal_handler(sig, frame):
    global _shutdown
    _shutdown = True
    print("\n收到退出信号，正在停止...")

signal.signal(signal.SIGINT, _signal_handler)


# =============================================================================
# 主函数
# =============================================================================
def main():
    global _shutdown
    
    logger.info("=" * 60)
    logger.info("   人物跟随系统 (方案四: 视觉+LiDAR融合+分层控制)")
    logger.info("=" * 60)
    
    # -----------------------------------------------------------------
    # 1. 初始化各模块
    # -----------------------------------------------------------------
    logger.info("正在初始化模块...")
    
    # 机器人API (你需要在 robot_api.py 中实现具体接口)
    robot_api = RobotAPI()
    
    # LiDAR处理器
    lidar_proc = LidarProcessor()
    
    # 视觉检测器
    vision_det = VisionDetector()
    
    # 传感器融合 (EKF)
    fusion = SensorFusion()
    
    # 运动控制器
    motion_ctrl = MotionController(robot_api)
    
    # 状态机
    state_machine = StateMachine(robot_api)
    
    # -----------------------------------------------------------------
    # 2. 加载静态地图 (用于LiDAR差分)
    # -----------------------------------------------------------------
    logger.info("正在加载全局地图...")
    
    # 优先从预处理的 .npy 文件加载 (推荐，快速)
    # 运行方法: python map_preprocessor.py log.txt ./maps/
    if MAP_POINTS_NPY_PATH:
        lidar_proc.load_map_from_npy(MAP_POINTS_NPY_PATH)
    else:
        # 备选: 运行时从机器人API下载地图
        try:
            map_data = robot_api.get_global_map()
            if map_data is not None:
                lidar_proc.load_map_from_dict(map_data)
        except NotImplementedError:
            logger.warning("get_global_map() 未实现且无预处理地图文件")
            logger.warning("  → LiDAR将无法做地图差分，所有LiDAR点都视为动态物体")
            logger.warning("  → 请运行: python map_preprocessor.py <log文件> ./maps/")
        except Exception as e:
            logger.error(f"地图加载失败: {e}")
    
    # -----------------------------------------------------------------
    # 3. 锁定跟随目标
    # -----------------------------------------------------------------
    logger.info("正在锁定跟随目标...")
    logger.info("  请确保目标人物站在机器人正前方，面朝相机")
    
    # 自动锁定当前画面中最近的人
    # 适合简单场景，机器人前方只有一个人
    target_locked = False
    try:
        target_locked = vision_det.lock_target_from_frame(robot_api, "head")
    except NotImplementedError:
        logger.warning("相机接口未实现，跳过视觉目标锁定")
    
    if not target_locked:
        logger.warning("未能通过视觉锁定目标")
        logger.info("将在运行中尝试锁定第一个检测到的人")
    
    # -----------------------------------------------------------------
    # 4. 主循环
    # -----------------------------------------------------------------
    logger.info("开始跟随主循环...")
    state_machine.start()
    
    loop_period = 1.0 / MAIN_LOOP_RATE  # 每个周期的目标时间
    loop_count = 0
    
    # 视觉检测频率控制 (不需要每帧都做视觉检测，太耗时)
    vision_interval = 3  # 每3个主循环做一次视觉检测
    
    while not _shutdown:
        loop_start = time.time()
        loop_count += 1
        
        try:
            try :
                robot_api.get_state()
            except:
                print("拉取推送失败")
                continue
            # =========================================================
            # Step 1: 获取机器人当前位姿
            # =========================================================
            try:
                robot_pose = robot_api.get_robot_pose()
            except NotImplementedError:
                print("ERROR: 获取机器人世界坐标失败！")
                continue
            
            # =========================================================
            # Step 2: LiDAR处理 (每帧都做，速度快)
            # =========================================================
            lidar_candidates = []
            obstacle_sectors = np.full(72, 40.0)  # 默认值: 无障碍
            
            try:
                scans = robot_api.get_lidar_scans()
                if scans:
                    # 提取人物候选
                    lidar_candidates = lidar_proc.process(scans, robot_pose)
                    
                    # 获取障碍物扇区 (用于避障)
                    obstacle_sectors = lidar_proc.get_obstacle_sectors(scans)
            except NotImplementedError:
                pass  # LiDAR接口未实现
            except Exception as e:
                logger.debug(f"LiDAR处理异常: {e}")
            
            # =========================================================
            # Step 3: 视觉检测 (降频执行)
            # =========================================================
            vision_detections = []
            target_detection = None
            
            if loop_count % vision_interval == 0:
                try:
                    vision_detections = vision_det.detect(robot_api, robot_pose)
                    
                    # 找到标记为目标的检测
                    for det in vision_detections:
                        if det.is_target:
                            target_detection = det
                            break
                    
                    # 如果尚未锁定目标且检测到了人，自动锁定第一个
                    if target_detection is None and not target_locked and vision_detections:
                        nearest = min(vision_detections, key=lambda d: d.depth)
                        vision_det.lock_target(nearest)
                        target_detection = nearest
                        target_detection.is_target = True
                        target_locked = True
                        logger.info(f"自动锁定目标: depth={nearest.depth:.2f}m")
                    
                except NotImplementedError:
                    pass
                except Exception as e:
                    logger.debug(f"视觉检测异常: {e}")
            
            # =========================================================
            # Step 4: 传感器融合 (EKF更新)
            # =========================================================
            now = time.time()
            
            # 4a: 视觉观测更新
            if target_detection is not None:
                fusion.update_with_vision(target_detection)
                # 视觉 ReID 确认的位置作为锚点，引导后续 LiDAR 关联
                fusion.set_vision_anchor(target_detection.world_x, target_detection.world_y)
            
            # 4b: LiDAR观测更新
            # 在LiDAR候选中找与跟踪目标最近的进行关联
            matched_lidar = None
            if lidar_candidates:
                matched_lidar = fusion.associate_lidar_candidates(lidar_candidates)
                if matched_lidar is not None:
                    fusion.update_with_lidar(matched_lidar)

            # 4c: 如果本帧没有任何观测，做纯预测
            if target_detection is None and matched_lidar is None:
                fusion.predict_only(now)
            
            # 获取当前目标状态
            target_state = fusion.get_target_state()
            
            # =========================================================
            # Step 5: 预计算控制指令 (用于卡住检测)
            # =========================================================
            cmd_linear = 0.0
            cmd_angular = 0.0
            if target_state.is_valid:
                cmd_linear, cmd_angular = motion_ctrl.compute_velocity(
                    target_state, robot_pose, obstacle_sectors
                )

            # =========================================================
            # Step 6: 状态机更新 (传入预计算速度用于卡住检测)
            # =========================================================
            current_state = state_machine.update(
                target_state, robot_pose, cmd_linear_vel=cmd_linear
            )

            # =========================================================
            # Step 7: 根据状态执行控制
            # =========================================================
            if current_state == FollowState.DIRECT_FOLLOW:
                # --- 直接跟随模式 ---
                if target_state.is_valid:
                    try:
                        robot_api.send_velocity(cmd_linear, cmd_angular)
                    except NotImplementedError:
                        pass
                else:
                    motion_ctrl.stop()

            elif current_state == FollowState.NAV_FOLLOW:
                # --- 导航跟随模式 ---
                # 导航指令由状态机内部发送 (调用 navigate_to)
                # 这里不需要额外操作，导航系统自行控制
                pass

            elif current_state == FollowState.SEARCH:
                # --- 搜索模式 ---
                direction = state_machine.get_search_direction()
                linear_vel, angular_vel = motion_ctrl.rotate_search(direction)
                try:
                    robot_api.send_velocity(linear_vel, angular_vel)
                except NotImplementedError:
                    pass

            elif current_state == FollowState.LOST:
                # --- 丢失 ---
                motion_ctrl.stop()

            elif current_state == FollowState.IDLE:
                # --- 空闲 ---
                pass
            
            # =========================================================
            # Step 8: 日志输出
            # =========================================================
            if loop_count % (MAIN_LOOP_RATE * 2) == 0:  # 每2秒打印一次
                status = state_machine.get_status_str()
                if target_state.is_valid:
                    dist = math.hypot(
                        target_state.x - robot_pose.x,
                        target_state.y - robot_pose.y,
                    )
                    logger.info(
                        f"{status} | "
                        f"目标({target_state.x:.2f},{target_state.y:.2f}) "
                        f"距离={dist:.2f}m 速度={target_state.speed:.2f}m/s | "
                        f"LiDAR候选={len(lidar_candidates)}"
                    )
                else:
                    logger.info(f"{status} | 目标状态无效")
        
        except Exception as e:
            logger.error(f"主循环异常: {e}", exc_info=True)
            try:
                robot_api.stop()
            except:
                pass
        
        # =========================================================
        # 循环频率控制
        # =========================================================
        elapsed = time.time() - loop_start
        sleep_time = loop_period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    # -----------------------------------------------------------------
    # 5. 清理退出
    # -----------------------------------------------------------------
    logger.info("正在停止...")
    state_machine.stop()
    logger.info("已退出人物跟随")
    robot_api.release()


# =============================================================================
# 入口
# =============================================================================
if __name__ == "__main__":
    main()
