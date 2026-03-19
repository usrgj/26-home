"""可视化激光雷达点云数据 - 从txt文件读取并解析"""

import ast
import math
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_point_cloud(filepath):
    """从txt文件读取点云数据（Python字面量格式）"""
    with open(filepath, "r") as f:
        content = f.read()
    return ast.literal_eval(content)


def polar_to_cartesian(beams, install_info):
    """将极坐标beam数据转换为笛卡尔坐标，考虑安装偏移和朝向"""
    xs, ys = [], []
    yaw_rad = math.radians(install_info.get("yaw", 0))
    ox = install_info.get("x", 0)

    for b in beams:
        if not b.get("valid", False) or "dist" not in b:
            continue
        angle_rad = math.radians(b.get("angle", 0))
        d = b["dist"]
        # 激光雷达本体坐标: x朝前, y朝左
        lx = d * math.cos(angle_rad)
        ly = d * math.sin(angle_rad)
        # 旋转到车体坐标系
        rx = lx * math.cos(yaw_rad) - ly * math.sin(yaw_rad) + ox
        ry = lx * math.sin(yaw_rad) + ly * math.cos(yaw_rad)
        xs.append(rx)
        ys.append(ry)

    return np.array(xs), np.array(ys)


def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else "point cloud.txt"
    sensors = load_point_cloud(filepath)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    # --- 左图: 合并点云（车体坐标系） ---
    ax = axes[0]
    for i, sensor in enumerate(sensors):
        info = sensor.get("install_info", {})
        label = sensor.get("device_info", {}).get("device_name", f"lidar_{i}")
        xs, ys = polar_to_cartesian(sensor["beams"], info)
        ax.scatter(ys, xs, s=2, c=colors[i % len(colors)], label=label)

    ax.plot(0, 0, "k^", markersize=10, label="robot center")
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("X (m) - forward")
    ax.set_title("Combined Point Cloud (robot frame)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 右图: 极坐标 ---
    ax2 = fig.add_subplot(122, polar=True)
    axes[1].set_visible(False)

    for i, sensor in enumerate(sensors):
        info = sensor.get("install_info", {})
        yaw_deg = info.get("yaw", 0)
        label = sensor.get("device_info", {}).get("device_name", f"lidar_{i}")
        angles, dists = [], []
        for b in sensor["beams"]:
            if not b.get("valid", False) or "dist" not in b:
                continue
            angles.append(math.radians(b.get("angle", 0) + yaw_deg + 90))
            dists.append(b["dist"])
        ax2.scatter(angles, dists, s=2, c=colors[i % len(colors)], label=label)

    ax2.set_title("Polar View", pad=20)
    ax2.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig("lidar_visualization.png", dpi=150, bbox_inches="tight")
    print(f"Loaded {len(sensors)} lidar(s) from '{filepath}'")
    print("Saved to lidar_visualization.png")
    plt.show()


if __name__ == "__main__":
    main()
