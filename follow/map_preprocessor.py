"""
=============================================================================
map_preprocessor.py — 地图预处理工具
=============================================================================
将从机器人下载的原始地图数据转为高效的 numpy .npy 文件。

用法:
  python map_preprocessor.py <地图log文件路径> [输出目录]

示例:
  python map_preprocessor.py log.txt ./maps/

这只需要运行一次。生成的 .npy 文件会在 main.py 启动时快速加载。

为什么要预处理？
  1. 原始log文件约 1.1MB，包含日志噪声，需要 ast.literal_eval 解析 (慢)
  2. 转为 .npy 后只有 568KB，加载只需 <1ms (numpy原生二进制格式)
  3. 地图是静态的，不需要每次启动都从机器人下载
  4. 还可以额外保存地图元数据、特征线等信息供后续使用
"""
import os
import sys
import ast
import json
import time
import numpy as np
from pathlib import Path


def extract_map_from_log(log_path: str) -> dict:
    """
    从下载日志文件中提取地图数据。
    
    你的日志文件格式: 多行日志文本中，有一行是 Python dict 格式的地图数据。
    该行以 {'header': 开头，包含完整的地图信息。
    """
    print(f"读取日志文件: {log_path}")
    
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 查找包含地图数据的行 (以 {'header' 开头)
    map_line = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("{'header'"):
            map_line = stripped
            print(f"  在第 {i+1} 行找到地图数据 (长度: {len(stripped):,} 字符)")
            break
    
    if map_line is None:
        raise ValueError("日志文件中未找到地图数据行 (以 {'header' 开头的行)")
    
    # 解析 Python dict 格式
    print("  正在解析地图数据...")
    t0 = time.time()
    data = ast.literal_eval(map_line)
    print(f"  解析完成，耗时 {(time.time()-t0)*1000:.0f}ms")
    
    return data


def process_map(data: dict, output_dir: str):
    """
    处理地图数据，保存为高效的 numpy 格式。
    
    输出文件:
      - map_points.npy   : 障碍物点云 (N, 2) float64
      - map_meta.json     : 地图元数据 (名称、范围、分辨率等)
      - map_lines.json    : 特征线数据 (如果有)
      - map_areas.json    : 区域数据 (如果有)
    """
    os.makedirs(output_dir, exist_ok=True)
    header = data['header']
    
    # =====================================================================
    # 1. 处理障碍物点云 (normalPosList)
    # =====================================================================
    raw_pts = data.get('normalPosList', [])
    print(f"\n障碍物点云:")
    print(f"  原始点数: {len(raw_pts)}")
    
    # 过滤掉不完整的点 (缺少x或y)
    valid_pts = [(p['x'], p['y']) for p in raw_pts 
                 if 'x' in p and 'y' in p]
    skipped = len(raw_pts) - len(valid_pts)
    if skipped > 0:
        print(f"  跳过不完整的点: {skipped}")
    print(f"  有效点数: {len(valid_pts)}")
    
    points = np.array(valid_pts, dtype=np.float64)
    
    # 保存为 .npy (二进制格式，加载极快)
    npy_path = os.path.join(output_dir, "map_points.npy")
    np.save(npy_path, points)
    print(f"  已保存: {npy_path} ({os.path.getsize(npy_path)/1024:.0f} KB)")
    
    # 验证加载速度
    t0 = time.time()
    _ = np.load(npy_path)
    load_time = (time.time() - t0) * 1000
    print(f"  加载验证: {load_time:.1f}ms")
    
    # KD-Tree 构建测试
    from scipy.spatial import cKDTree
    t0 = time.time()
    tree = cKDTree(points)
    build_time = (time.time() - t0) * 1000
    print(f"  KD-Tree构建: {build_time:.1f}ms")
    
    # 模拟查询测试
    test_pts = np.random.uniform(points.min(axis=0), points.max(axis=0), size=(720, 2))
    t0 = time.time()
    dists, _ = tree.query(test_pts)
    query_time = (time.time() - t0) * 1000
    print(f"  720点查询: {query_time:.2f}ms")
    
    # =====================================================================
    # 2. 保存元数据
    # =====================================================================
    meta = {
        "map_name": header.get('mapName', 'unknown'),
        "map_type": header.get('mapType', '2D-Map'),
        "resolution": header.get('resolution', 0.005),
        "version": header.get('version', ''),
        "min_pos": header.get('minPos', {}),
        "max_pos": header.get('maxPos', {}),
        "num_points": len(valid_pts),
        "x_range": [float(points[:, 0].min()), float(points[:, 0].max())],
        "y_range": [float(points[:, 1].min()), float(points[:, 1].max())],
        "x_span_m": float(points[:, 0].max() - points[:, 0].min()),
        "y_span_m": float(points[:, 1].max() - points[:, 1].min()),
    }
    
    meta_path = os.path.join(output_dir, "map_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n元数据已保存: {meta_path}")
    
    # =====================================================================
    # 3. 保存特征线 (advancedLineList)
    #    这些特征线可能表示墙壁、门框等结构化信息，
    #    未来可以用于门检测或路径规划的补充信息。
    # =====================================================================
    lines_data = data.get('advancedLineList', [])
    if lines_data:
        # 简化存储: 只保留起点和终点坐标
        simplified_lines = []
        for line in lines_data:
            if 'line' in line:
                simplified_lines.append({
                    "class": line.get('className', ''),
                    "start": line['line'].get('startPos', {}),
                    "end": line['line'].get('endPos', {}),
                })
        
        lines_path = os.path.join(output_dir, "map_lines.json")
        with open(lines_path, "w", encoding="utf-8") as f:
            json.dump(simplified_lines, f, indent=2)
        print(f"特征线已保存: {lines_path} ({len(simplified_lines)} 条)")
    
    # =====================================================================
    # 4. 保存标记点 (advancedPointList)
    # =====================================================================
    points_data = data.get('advancedPointList', [])
    if points_data:
        marks_path = os.path.join(output_dir, "map_marks.json")
        simplified_marks = []
        for pt in points_data:
            simplified_marks.append({
                "name": pt.get('instanceName', ''),
                "class": pt.get('className', ''),
                "x": pt.get('pos', {}).get('x', 0),
                "y": pt.get('pos', {}).get('y', 0),
                "dir": pt.get('dir', 0),
            })
        with open(marks_path, "w", encoding="utf-8") as f:
            json.dump(simplified_marks, f, indent=2)
        print(f"标记点已保存: {marks_path} ({len(simplified_marks)} 个)")
    
    # =====================================================================
    # 5. 保存区域 (advancedAreaList)
    # =====================================================================
    areas_data = data.get('advancedAreaList', [])
    if areas_data:
        areas_path = os.path.join(output_dir, "map_areas.json")
        simplified_areas = []
        for area in areas_data:
            simplified_areas.append({
                "name": area.get('instanceName', ''),
                "class": area.get('className', ''),
                "vertices": area.get('posGroup', []),
            })
        with open(areas_path, "w", encoding="utf-8") as f:
            json.dump(simplified_areas, f, indent=2)
        print(f"区域已保存: {areas_path} ({len(simplified_areas)} 个)")
    
    # =====================================================================
    # 总结
    # =====================================================================
    print("\n" + "=" * 50)
    print("地图预处理完成!")
    print(f"  地图名称: {meta['map_name']}")
    print(f"  场地大小: {meta['x_span_m']:.1f}m × {meta['y_span_m']:.1f}m")
    print(f"  障碍物点: {meta['num_points']:,}")
    print(f"  输出目录: {output_dir}")
    print(f"  文件列表:")
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        print(f"    {fname:25s} {os.path.getsize(fpath)/1024:8.1f} KB")
    print("=" * 50)
    print("\n在 config.py 中设置:")
    print(f'  MAP_POINTS_PATH = "{os.path.abspath(npy_path)}"')


# =============================================================================
# 入口
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python map_preprocessor.py <地图log文件> [输出目录]")
        print("示例: python map_preprocessor.py log.txt ./maps/")
        sys.exit(1)
    
    log_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./maps"
    
    data = extract_map_from_log(log_path)
    process_map(data, output_dir)
