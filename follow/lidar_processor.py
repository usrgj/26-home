"""
=============================================================================
lidar_processor.py — LiDAR数据处理模块
=============================================================================
职责：
1. 合并前后两个LiDAR的扫描数据为统一的360°点云
2. 与静态全局地图做差分，提取"非地图上"的动态点
3. 对动态点做聚类 (DBSCAN)
4. 从聚类中筛选出类似"人腿"的候选
5. 配对腿部聚类，输出"人物候选"的位置
"""
import math
import time
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

from config import (
    LIDAR_FRONT_X, LIDAR_FRONT_YAW, LIDAR_REAR_X, LIDAR_REAR_YAW,
    LIDAR_MAX_RANGE, LIDAR_MIN_RANGE,
    CLUSTER_EPS, CLUSTER_MIN_POINTS, CLUSTER_MAX_POINTS,
    LEG_MIN_RADIUS, LEG_MAX_RADIUS, LEG_PAIR_MIN_DIST, LEG_PAIR_MAX_DIST,
    MAP_DIFF_THRESHOLD,
)
from robot_api import LidarScan, LidarBeam, RobotPose


@dataclass
class PointXY:
    """2D点 (机器人坐标系)"""
    x: float = 0.0
    y: float = 0.0


@dataclass
class Cluster:
    """一个聚类"""
    points: List[PointXY] = field(default_factory=list)
    center_x: float = 0.0      # 聚类中心X (机器人坐标系)
    center_y: float = 0.0      # 聚类中心Y
    radius: float = 0.0        # 聚类半径
    is_leg_candidate: bool = False


@dataclass
class PersonCandidate:
    """
    从LiDAR检测到的人物候选。
    可能是由两条腿配对得到，也可能是单个较大的聚类。
    坐标为机器人坐标系。
    """
    local_x: float = 0.0       # 机器人坐标系X (前方为正)
    local_y: float = 0.0       # 机器人坐标系Y (左方为正)
    world_x: float = 0.0       # 世界坐标X (经过转换后填充)
    world_y: float = 0.0       # 世界坐标Y
    confidence: float = 0.0    # 置信度 (0~1)
    timestamp: float = 0.0


class LidarProcessor:
    """LiDAR数据处理器"""
    
    def __init__(self):
        # 静态地图的点云缓存 (世界坐标系)
        # 用于与实时扫描做差分，格式为 numpy数组 (N, 2)
        self._static_map_points: Optional[np.ndarray] = None
        
        # 地图的KD-Tree，用于快速最近邻查询
        self._map_kdtree = None
    
    # =====================================================================
    # 地图加载
    # =====================================================================
    def load_map_from_npy(self, npy_path: str):
        """
        从预处理的 .npy 文件加载静态地图点云。（推荐方式）
        
        .npy 文件由 map_preprocessor.py 生成，格式为 (N, 2) float64 数组，
        每行是一个障碍物点的世界坐标 [x, y]。
        
        参数:
            npy_path: .npy 文件路径 (由 map_preprocessor.py 生成)
        """
        import os
        if not os.path.exists(npy_path):
            print(f"[LidarProcessor] 警告: 地图文件不存在: {npy_path}")
            print(f"  请先运行: python map_preprocessor.py <log文件> ./maps/")
            return
        
        self._static_map_points = np.load(npy_path)
        self._build_kdtree()
    
    def load_map_from_dict(self, map_data: Dict):
        """
        从机器人API返回的原始地图字典中加载点云。
        
        你的地图格式:
        - map_data['normalPosList']: [{x:..., y:...}, ...] 障碍物点列表
        - map_data['header']: 元数据 (mapName, resolution等)
        
        这些点已经是世界坐标，无需任何转换。
        """
        raw_pts = map_data.get('normalPosList', [])
        if not raw_pts:
            print("[LidarProcessor] 警告: 地图中无 normalPosList")
            return
        
        # 过滤不完整的点 (你的数据中有少量缺少x或y的点)
        valid = [(p['x'], p['y']) for p in raw_pts if 'x' in p and 'y' in p]
        self._static_map_points = np.array(valid, dtype=np.float64)
        self._build_kdtree()
    
    def _build_kdtree(self):
        """构建 KD-Tree 索引"""
        if self._static_map_points is not None and len(self._static_map_points) > 0:
            from scipy.spatial import cKDTree
            self._map_kdtree = cKDTree(self._static_map_points)
            print(f"[LidarProcessor] 已加载静态地图: {len(self._static_map_points):,} 个点, "
                  f"范围 X[{self._static_map_points[:,0].min():.1f}, {self._static_map_points[:,0].max():.1f}] "
                  f"Y[{self._static_map_points[:,1].min():.1f}, {self._static_map_points[:,1].max():.1f}]")
        else:
            print("[LidarProcessor] 警告: 地图点云为空，将跳过地图差分")
    
    # =====================================================================
    # 核心处理流程
    # =====================================================================
    def process(self, scans: List[LidarScan], robot_pose: RobotPose
                ) -> List[PersonCandidate]:
        """
        处理一帧LiDAR数据，返回所有检测到的人物候选。
        
        处理流程:
        1. 合并前后LiDAR → 360°点云 (机器人坐标系)
        2. 转换到世界坐标系 → 与静态地图差分 → 提取动态点
        3. 对动态点做DBSCAN聚类
        4. 从聚类中识别腿部候选 → 配对 → 生成人物候选
        
        参数:
            scans: LiDAR扫描数据列表 (通常2个，前后各一)
            robot_pose: 当前机器人位姿 (用于坐标变换)
        
        返回:
            人物候选列表，坐标为世界坐标系和机器人坐标系
        """
        timestamp = time.time()
        
        # Step 1: 合并所有LiDAR扫描为统一的机器人坐标系点云
        all_local_points = self._merge_scans(scans)
        if len(all_local_points) == 0:
            return []
        
        # Step 2: 与静态地图差分，提取动态点
        dynamic_points = self._filter_dynamic_points(all_local_points, robot_pose)
        if len(dynamic_points) == 0:
            return []
        
        # Step 3: DBSCAN聚类
        clusters = self._cluster_points(dynamic_points)
        
        # Step 4: 腿部检测与人物配对
        candidates = self._detect_persons(clusters, robot_pose, timestamp)
        
        return candidates
    
    def get_obstacle_sectors(self, scans: List[LidarScan],
                             num_sectors: int = 72) -> np.ndarray:
        """
        将360°范围划分为若干扇区，返回每个扇区内的最近障碍物距离。
        用于反应式避障 (VFH 向量场直方图)。
        
        参数:
            scans: LiDAR扫描数据
            num_sectors: 扇区数量 (默认72个，每个5°)
        
        返回:
            numpy数组 (num_sectors,)，每个扇区最近的障碍物距离 (m)
            扇区0对应机器人正前方, 逆时针递增
        """
        sector_size = 360.0 / num_sectors
        min_dists = np.full(num_sectors, LIDAR_MAX_RANGE)
        
        all_points = self._merge_scans(scans)
        
        for px, py in all_points:
            # 计算该点相对于机器人中心的角度和距离
            angle = math.degrees(math.atan2(py, px))  # -180 ~ 180
            if angle < 0:
                angle += 360.0  # 0 ~ 360
            dist = math.hypot(px, py)
            
            sector_idx = int(angle / sector_size) % num_sectors
            if dist < min_dists[sector_idx]:
                min_dists[sector_idx] = dist
        
        return min_dists
    
    # =====================================================================
    # 内部方法
    # =====================================================================
    def _merge_scans(self, scans: List[LidarScan]) -> List[Tuple[float, float]]:
        """
        合并多个LiDAR扫描到统一的机器人坐标系。
        
        每个LiDAR有自己的安装偏移 (install_x, install_yaw)。
        光束角度 + 安装偏移 → 机器人坐标系中的点。
        
        前LiDAR: install_x=0.299, yaw=0° → 覆盖前方 ±90°
        后LiDAR: install_x=-0.299, yaw=180° → 覆盖后方 ±90°
        合起来形成完整的360°覆盖。
        """
        all_points = []
        
        for scan in scans:
            # 安装偏移：LiDAR在机器人坐标系中的位置和朝向
            install_x = scan.install_x
            install_yaw_rad = math.radians(scan.install_yaw)
            
            for beam in scan.beams:
                if not beam.valid:
                    continue
                if beam.dist < LIDAR_MIN_RANGE or beam.dist > LIDAR_MAX_RANGE:
                    continue
                
                # 光束在LiDAR局部坐标系中的坐标
                beam_angle_rad = math.radians(beam.angle)
                local_x = beam.dist * math.cos(beam_angle_rad)
                local_y = beam.dist * math.sin(beam_angle_rad)
                
                # 将LiDAR局部坐标转换到机器人坐标系
                # 先旋转 (按照LiDAR安装朝向)
                cos_yaw = math.cos(install_yaw_rad)
                sin_yaw = math.sin(install_yaw_rad)
                robot_x = install_x + local_x * cos_yaw - local_y * sin_yaw
                robot_y = local_x * sin_yaw + local_y * cos_yaw
                
                all_points.append((robot_x, robot_y))
        
        return all_points
    
    def _filter_dynamic_points(self, local_points: List[Tuple[float, float]],
                                robot_pose: RobotPose
                                ) -> List[Tuple[float, float]]:
        """
        通过与静态地图差分，过滤掉属于静态地图的点，保留动态物体的点。
        
        原理：
        1. 将机器人坐标系中的点转换到世界坐标系
        2. 在静态地图KD-Tree中查找最近邻
        3. 如果最近邻距离 < MAP_DIFF_THRESHOLD，说明该点属于静态地图，丢弃
        4. 否则保留为动态点
        
        返回的动态点仍然是机器人坐标系（方便后续聚类和控制）
        """
        if self._map_kdtree is None:
            # 没有静态地图，所有点都当作动态点
            return local_points
        
        cos_t = math.cos(robot_pose.theta)
        sin_t = math.sin(robot_pose.theta)
        
        # 批量转换到世界坐标系
        world_points = []
        for lx, ly in local_points:
            wx = robot_pose.x + lx * cos_t - ly * sin_t
            wy = robot_pose.y + lx * sin_t + ly * cos_t
            world_points.append([wx, wy])
        
        world_array = np.array(world_points)
        
        # KD-Tree最近邻查询
        dists, _ = self._map_kdtree.query(world_array)
        
        # 保留距离静态地图较远的点 (即动态物体)
        dynamic_mask = dists > MAP_DIFF_THRESHOLD
        dynamic_points = [local_points[i] for i in range(len(local_points)) 
                         if dynamic_mask[i]]
        
        return dynamic_points
    
    def _cluster_points(self, points: List[Tuple[float, float]]) -> List[Cluster]:
        """
        对2D点集进行DBSCAN聚类。
        
        使用简单的DBSCAN实现（不依赖sklearn），适用于点数较少的情况。
        如果你的场景点数很多，建议替换为sklearn.cluster.DBSCAN。
        """
        if len(points) == 0:
            return []
        
        pts = np.array(points)
        n = len(pts)
        labels = np.full(n, -1, dtype=int)  # -1表示未分类
        cluster_id = 0
        
        for i in range(n):
            if labels[i] != -1:
                continue
            
            # 找到所有距离 < CLUSTER_EPS 的邻居
            dists = np.sqrt(np.sum((pts - pts[i]) ** 2, axis=1))
            neighbors = np.where(dists < CLUSTER_EPS)[0]
            
            if len(neighbors) < CLUSTER_MIN_POINTS:
                continue  # 噪声点
            
            # 开始一个新聚类
            labels[i] = cluster_id
            seed_set = list(neighbors)
            j = 0
            while j < len(seed_set):
                q = seed_set[j]
                if labels[q] == -1 or labels[q] == -2:  # 未分类或噪声
                    labels[q] = cluster_id
                    q_dists = np.sqrt(np.sum((pts - pts[q]) ** 2, axis=1))
                    q_neighbors = np.where(q_dists < CLUSTER_EPS)[0]
                    if len(q_neighbors) >= CLUSTER_MIN_POINTS:
                        seed_set.extend(q_neighbors.tolist())
                j += 1
            
            cluster_id += 1
        
        # 构建Cluster对象
        clusters = []
        for cid in range(cluster_id):
            mask = labels == cid
            count = np.sum(mask)
            
            if count < CLUSTER_MIN_POINTS or count > CLUSTER_MAX_POINTS:
                continue
            
            cluster_pts = pts[mask]
            center = cluster_pts.mean(axis=0)
            
            # 计算聚类半径 (中心到最远点的距离)
            radius = np.max(np.sqrt(np.sum((cluster_pts - center) ** 2, axis=1)))
            
            cluster = Cluster(
                points=[PointXY(p[0], p[1]) for p in cluster_pts],
                center_x=center[0],
                center_y=center[1],
                radius=radius,
            )
            
            # 判断是否为腿部候选
            if LEG_MIN_RADIUS <= radius <= LEG_MAX_RADIUS:
                cluster.is_leg_candidate = True
            
            clusters.append(cluster)
        
        return clusters
    
    def _detect_persons(self, clusters: List[Cluster],
                        robot_pose: RobotPose, timestamp: float
                        ) -> List[PersonCandidate]:
        """
        从聚类中检测人物候选。
        
        策略：
        1. 先尝试将腿部候选两两配对（两条腿间距在合理范围内）
        2. 未配对的腿也作为单腿候选（置信度较低）
        3. 较大的非腿聚类也可能是人的躯干截面
        """
        leg_candidates = [c for c in clusters if c.is_leg_candidate]
        persons = []
        paired = set()
        
        # 两两配对腿部
        for i in range(len(leg_candidates)):
            if i in paired:
                continue
            best_j = -1
            best_dist = float('inf')
            
            for j in range(i + 1, len(leg_candidates)):
                if j in paired:
                    continue
                dist = math.hypot(
                    leg_candidates[i].center_x - leg_candidates[j].center_x,
                    leg_candidates[i].center_y - leg_candidates[j].center_y,
                )
                if LEG_PAIR_MIN_DIST <= dist <= LEG_PAIR_MAX_DIST:
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j
            
            if best_j >= 0:
                # 成功配对，人物位置取两腿中点
                paired.add(i)
                paired.add(best_j)
                
                cx = (leg_candidates[i].center_x + leg_candidates[best_j].center_x) / 2
                cy = (leg_candidates[i].center_y + leg_candidates[best_j].center_y) / 2
                
                # 转换到世界坐标
                cos_t = math.cos(robot_pose.theta)
                sin_t = math.sin(robot_pose.theta)
                wx = robot_pose.x + cx * cos_t - cy * sin_t
                wy = robot_pose.y + cx * sin_t + cy * cos_t
                
                persons.append(PersonCandidate(
                    local_x=cx, local_y=cy,
                    world_x=wx, world_y=wy,
                    confidence=0.9,  # 双腿配对置信度高
                    timestamp=timestamp,
                ))
        
        # 未配对的单腿也作为候选 (低置信度)
        for i in range(len(leg_candidates)):
            if i in paired:
                continue
            c = leg_candidates[i]
            cos_t = math.cos(robot_pose.theta)
            sin_t = math.sin(robot_pose.theta)
            wx = robot_pose.x + c.center_x * cos_t - c.center_y * sin_t
            wy = robot_pose.y + c.center_x * sin_t + c.center_y * cos_t
            
            persons.append(PersonCandidate(
                local_x=c.center_x, local_y=c.center_y,
                world_x=wx, world_y=wy,
                confidence=0.4,  # 单腿，置信度较低
                timestamp=timestamp,
            ))
        
        return persons
