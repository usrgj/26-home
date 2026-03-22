"""
=============================================================================
calibrate_camera.py — Intel RealSense 深度相机外参标定工具
=============================================================================
用 ArUco 标记 + RealSense 深度测量，自动计算相机外参 (x, y, yaw, pitch)。

使用方法:
    python calibrate_camera.py

准备工作:
    1. 打印一张 ArUco 标记 (4x4 字典, ID 0, 建议 15cm x 15cm)
    2. 准备卷尺
    3. 确保机器人静止不动

依赖: pyrealsense2, opencv-python, numpy, scipy, matplotlib
"""
import sys
import math
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# 相机序列号映射 (来自 camera/config.py)
# ---------------------------------------------------------------------------
SERIAL_MAP = {
    "head":      "151222072331",
    "chest":     "151222073707",
    "left_arm":  "141722075710",
    "right_arm": "239722070896",
}

# 当前 config.py 中的初始外参估计 (用作优化初值)
INITIAL_PARAMS = {
    "head":      {"x": 0.0, "y": 0.0, "yaw": 0.0, "pitch": -10.0},
    "chest":     {"x": 0.0, "y": 0.0, "yaw": 0.0, "pitch": 0.0},
    "left_arm":  {"x": 0.0, "y": 0.25, "yaw": 30.0, "pitch": 0.0},
    "right_arm": {"x": 0.0, "y": -0.25, "yaw": -30.0, "pitch": 0.0},
}

# ArUco 配置
ARUCO_DICT = cv2.aruco.DICT_4X4_50
DEPTH_PATCH_SIZE = 20  # 深度采样区域半径 (像素)
CAPTURE_FRAMES = 10    # 每次采集平均帧数


# =============================================================================
# ArUco 检测
# =============================================================================
def detect_aruco(color_image):
    """
    检测 ArUco 标记，返回 (corners, ids)。
    corners: list of (4,2) arrays — 每个标记的4个角点像素坐标
    ids: (N,1) array — 标记ID
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(color_image)
    return corners, ids


def get_marker_center_px(corners):
    """从4个角点计算标记中心像素坐标"""
    pts = corners[0]  # shape (4, 2)
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    return cx, cy


def sample_depth(depth_image, cx, cy, patch_radius=DEPTH_PATCH_SIZE):
    """
    在 (cx, cy) 周围取 patch 的中位数深度 (毫米 → 米)。
    返回 -1 表示无效。
    """
    h, w = depth_image.shape[:2]
    x1 = max(0, int(cx - patch_radius))
    x2 = min(w, int(cx + patch_radius))
    y1 = max(0, int(cy - patch_radius))
    y2 = min(h, int(cy + patch_radius))

    roi = depth_image[y1:y2, x1:x2]
    valid = roi[roi > 0]
    if len(valid) < 10:
        return -1.0
    return float(np.median(valid)) / 1000.0


def pixel_to_camera(cx, cy, depth_m, intrinsics):
    """
    像素坐标 + 深度 → 相机坐标系 (X右, Y下, Z前)。
    使用 RealSense 真实内参。
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    ppx, ppy = intrinsics["ppx"], intrinsics["ppy"]
    cam_x = (cx - ppx) * depth_m / fx   # 右方偏移
    cam_y = (cy - ppy) * depth_m / fy   # 下方偏移
    cam_z = depth_m                      # 前方距离
    return cam_x, cam_y, cam_z


# =============================================================================
# 坐标变换: 相机 → 机器人
# =============================================================================
def camera_to_robot(cam_x, cam_y, cam_z, tx, ty, yaw_rad, pitch_rad):
    """
    相机坐标系 → 机器人坐标系。

    相机坐标系: X右, Y下, Z前
    机器人坐标系: X前, Y左

    步骤:
    1. pitch 旋转 (绕相机X轴，即水平轴): 修正 cam_z 和 cam_y
    2. yaw 旋转 + 平移: 映射到机器人坐标系

    注意: cam_x (相机右方) 映射到 -robot_y (机器人左方为正)
    """
    # pitch 旋转
    cos_p = math.cos(pitch_rad)
    sin_p = math.sin(pitch_rad)
    cam_z_rot = cam_z * cos_p + cam_y * sin_p
    # cam_y_rot = -cam_z * sin_p + cam_y * cos_p  # 垂直方向，不用于2D

    # yaw 旋转 + 平移
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    robot_x = tx + cam_z_rot * cos_y + cam_x * sin_y
    robot_y = ty + cam_z_rot * sin_y - cam_x * cos_y
    return robot_x, robot_y


# =============================================================================
# 最小二乘标定
# =============================================================================
def calibrate(measurements, initial_guess):
    """
    从多组观测拟合 (tx, ty, yaw, pitch)。

    measurements: list of (x_gt, y_gt, cam_x, cam_y, cam_z)
    initial_guess: dict with keys x, y, yaw(deg), pitch(deg)

    返回: (tx, ty, yaw_deg, pitch_deg, result)
    """
    x0 = [
        initial_guess["x"],
        initial_guess["y"],
        math.radians(initial_guess["yaw"]),
        math.radians(initial_guess["pitch"]),
    ]

    def residuals(params):
        tx, ty, yaw_rad, pitch_rad = params
        errs = []
        for (x_gt, y_gt, cam_x, cam_y, cam_z) in measurements:
            rx, ry = camera_to_robot(cam_x, cam_y, cam_z, tx, ty, yaw_rad, pitch_rad)
            errs.append(rx - x_gt)
            errs.append(ry - y_gt)
        return errs

    result = least_squares(residuals, x0, method='lm')
    tx, ty, yaw_rad, pitch_rad = result.x
    return tx, ty, math.degrees(yaw_rad), math.degrees(pitch_rad), result


def compute_errors(measurements, tx, ty, yaw_deg, pitch_deg):
    """计算每个测量点的误差"""
    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)
    errors = []
    for (x_gt, y_gt, cam_x, cam_y, cam_z) in measurements:
        rx, ry = camera_to_robot(cam_x, cam_y, cam_z, tx, ty, yaw_rad, pitch_rad)
        err = math.hypot(rx - x_gt, ry - y_gt)
        errors.append((x_gt, y_gt, rx, ry, err))
    return errors


# =============================================================================
# 相机初始化
# =============================================================================
def init_camera(serial):
    """启动 RealSense 相机，返回 (pipeline, align, intrinsics)"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # 获取真实内参
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    intrinsics = {
        "fx": intr.fx, "fy": intr.fy,
        "ppx": intr.ppx, "ppy": intr.ppy,
        "width": intr.width, "height": intr.height,
    }

    # 等待自动曝光稳定
    for _ in range(30):
        pipeline.wait_for_frames()

    return pipeline, align, intrinsics


def capture_frame(pipeline, align):
    """采集一帧对齐的 color + depth"""
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color = aligned.get_color_frame()
    depth = aligned.get_depth_frame()
    if not color or not depth:
        return None, None
    return np.asanyarray(color.get_data()), np.asanyarray(depth.get_data())


def capture_averaged(pipeline, align, intrinsics, n=CAPTURE_FRAMES):
    """
    采集 n 帧，检测 ArUco 并平均 3D 坐标。
    返回 (cam_x, cam_y, cam_z) 平均值，或 None。
    """
    samples = []
    for _ in range(n):
        color, depth = capture_frame(pipeline, align)
        if color is None:
            continue
        corners, ids = detect_aruco(color)
        if ids is None or len(ids) == 0:
            continue
        # 取第一个检测到的标记
        cx, cy = get_marker_center_px(corners[0])
        d = sample_depth(depth, cx, cy)
        if d <= 0 or d > 10.0:
            continue
        cam_x, cam_y, cam_z = pixel_to_camera(cx, cy, d, intrinsics)
        samples.append((cam_x, cam_y, cam_z))
        time.sleep(0.03)

    if len(samples) < 3:
        return None
    arr = np.array(samples)
    return float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])), float(np.mean(arr[:, 2]))


# =============================================================================
# 可视化
# =============================================================================
def plot_results(measurements, params_before, params_after):
    """绘制标定前后对比图"""
    import matplotlib.pyplot as plt

    gt_x = [m[0] for m in measurements]
    gt_y = [m[1] for m in measurements]

    before_errs = compute_errors(measurements, *params_before)
    after_errs = compute_errors(measurements, *params_after)

    bx = [e[2] for e in before_errs]
    by = [e[3] for e in before_errs]
    ax = [e[2] for e in after_errs]
    ay = [e[3] for e in after_errs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 位置对比
    axes[0].scatter(gt_y, gt_x, c='green', marker='o', s=80, label='Ground Truth')
    axes[0].scatter(by, bx, c='red', marker='x', s=80, label='Before Calib')
    axes[0].scatter(ay, ax, c='blue', marker='^', s=80, label='After Calib')
    for i in range(len(measurements)):
        axes[0].plot([gt_y[i], ay[i]], [gt_x[i], ax[i]], 'b--', alpha=0.3)
    axes[0].set_xlabel('Y (left, m)')
    axes[0].set_ylabel('X (forward, m)')
    axes[0].set_title('Position: GT vs Predicted')
    axes[0].legend()
    axes[0].set_aspect('equal')
    axes[0].grid(True)

    # 误差柱状图
    before_e = [e[4] for e in before_errs]
    after_e = [e[4] for e in after_errs]
    idx = np.arange(len(measurements))
    axes[1].bar(idx - 0.15, before_e, 0.3, label='Before', color='red', alpha=0.7)
    axes[1].bar(idx + 0.15, after_e, 0.3, label='After', color='blue', alpha=0.7)
    axes[1].set_xlabel('Measurement #')
    axes[1].set_ylabel('Error (m)')
    axes[1].set_title('Per-point Error')
    axes[1].legend()
    axes[1].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('calibration_result.png', dpi=100)
    print("\n[可视化] 已保存 calibration_result.png")
    plt.show()


# =============================================================================
# 交互式数据采集
# =============================================================================
def live_preview_and_capture(pipeline, align, intrinsics):
    """
    显示实时画面 + ArUco 检测叠加，按 Enter 采集。
    返回 (cam_x, cam_y, cam_z) 或 None。
    """
    print("    [实时预览] 请将 ArUco 标记放到相机视野内")
    print("    按 Enter 采集 | 按 q 跳过此点")

    window_name = "Calibration Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    result = None
    while True:
        color, depth = capture_frame(pipeline, align)
        if color is None:
            continue

        display = color.copy()
        corners, ids = detect_aruco(color)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            cx, cy = get_marker_center_px(corners[0])
            d = sample_depth(depth, cx, cy)
            if d > 0:
                cam_x, cam_y, cam_z = pixel_to_camera(cx, cy, d, intrinsics)
                info = f"depth={d:.3f}m cam=({cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f})"
                cv2.putText(display, info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No ArUco detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(30) & 0xFF

        if key == 13 or key == 10:  # Enter
            result = capture_averaged(pipeline, align, intrinsics)
            break
        elif key == ord('q'):
            break

    cv2.destroyWindow(window_name)
    return result


def collect_measurements(pipeline, align, intrinsics, camera_name):
    """交互式采集多组测量数据"""
    measurements = []

    print(f"\n{'='*60}")
    print(f"  数据采集 — {camera_name} 相机")
    print(f"{'='*60}")
    print()
    print("  坐标系说明:")
    print("    X = 机器人正前方 (正值)")
    print("    Y = 机器人左方 (正值), 右方 (负值)")
    print("    原点 = 机器人底盘中心")
    print()
    print("  建议: 在相机视野内采集 6~8 组，")
    print("        距离 1.0~3.0m，覆盖视野左右两侧。")
    print()

    point_idx = 0
    while True:
        point_idx += 1
        print(f"\n--- 测量点 #{point_idx} ---")
        print("输入标记相对机器人中心的真实位置 (卷尺测量)")
        print("  输入 'd' = 完成采集 (当前已采集 {} 组)".format(len(measurements)))

        x_str = input("  x_gt (前方距离, m): ").strip()
        if x_str.lower() == 'd':
            if len(measurements) < 4:
                print(f"  [警告] 至少需要 4 组数据 (当前 {len(measurements)} 组)，请继续采集")
                point_idx -= 1
                continue
            break

        try:
            x_gt = float(x_str)
        except ValueError:
            print("  无效输入，请重试")
            point_idx -= 1
            continue

        y_str = input("  y_gt (左方偏移, m, 右方为负): ").strip()
        try:
            y_gt = float(y_str)
        except ValueError:
            print("  无效输入，请重试")
            point_idx -= 1
            continue

        # 实时预览并采集
        cam_result = live_preview_and_capture(pipeline, align, intrinsics)
        if cam_result is None:
            print("  [失败] 未检测到标记或深度无效，请重试")
            point_idx -= 1
            continue

        cam_x, cam_y, cam_z = cam_result
        measurements.append((x_gt, y_gt, cam_x, cam_y, cam_z))
        print(f"  [OK] GT=({x_gt:.3f}, {y_gt:.3f}) "
              f"Cam=({cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f})")

    return measurements


# =============================================================================
# 主流程
# =============================================================================
def print_banner():
    print()
    print("=" * 60)
    print("  Intel RealSense 深度相机外参标定工具")
    print("=" * 60)
    print()
    print("  准备工作:")
    print("    1. 打印 ArUco 标记 (4x4字典, ID 0, 建议 15cm)")
    print("    2. 将标记贴在平面上 (纸板/墙壁)")
    print("    3. 准备卷尺")
    print("    4. 确保机器人固定不动")
    print()


def select_camera():
    """交互式选择相机"""
    names = list(SERIAL_MAP.keys())
    print("  可用相机:")
    for i, name in enumerate(names):
        print(f"    {i+1}. {name} (SN: {SERIAL_MAP[name]})")

    # 检测已连接的设备
    ctx = rs.context()
    connected = [d.get_info(rs.camera_info.serial_number) for d in ctx.devices]
    if connected:
        print(f"\n  已连接设备: {', '.join(connected)}")
    else:
        print("\n  [警告] 未检测到已连接的 RealSense 设备!")

    while True:
        choice = input("\n  选择相机编号 (1-4): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(names):
                name = names[idx]
                serial = SERIAL_MAP[name]
                if serial not in connected:
                    print(f"  [警告] {name} (SN: {serial}) 未连接，是否继续? (y/n)")
                    if input("  ").strip().lower() != 'y':
                        continue
                return name, serial
        except ValueError:
            pass
        print("  无效输入，请重试")


def main():
    print_banner()

    # 1. 选择相机
    camera_name, serial = select_camera()
    print(f"\n  已选择: {camera_name} (SN: {serial})")

    # 2. 输入 z (卷尺测量)
    while True:
        z_str = input("\n  输入相机高度 z (卷尺测量, m): ").strip()
        try:
            z_measured = float(z_str)
            if z_measured > 0:
                break
        except ValueError:
            pass
        print("  请输入有效的正数")

    # 3. 初始化相机
    print(f"\n  正在初始化相机 {serial} ...")
    try:
        pipeline, align, intrinsics = init_camera(serial)
    except Exception as e:
        print(f"\n  [错误] 无法初始化相机: {e}")
        sys.exit(1)

    print(f"  内参: fx={intrinsics['fx']:.1f} fy={intrinsics['fy']:.1f} "
          f"ppx={intrinsics['ppx']:.1f} ppy={intrinsics['ppy']:.1f}")

    # 4. 采集数据
    try:
        measurements = collect_measurements(pipeline, align, intrinsics, camera_name)
    except KeyboardInterrupt:
        print("\n\n  用户中断")
        pipeline.stop()
        sys.exit(0)

    if len(measurements) < 4:
        print("\n  [错误] 数据不足 (至少4组)，退出")
        pipeline.stop()
        sys.exit(1)

    # 5. 标定
    print(f"\n{'='*60}")
    print(f"  开始标定 ({len(measurements)} 组数据)")
    print(f"{'='*60}")

    initial = INITIAL_PARAMS[camera_name]
    tx, ty, yaw_deg, pitch_deg, result = calibrate(measurements, initial)

    # 6. 输出结果
    print(f"\n  === {camera_name} 相机标定结果 ===")
    print(f"  x:     {tx:.4f} m  (前方偏移)")
    print(f"  y:     {ty:.4f} m  (左方偏移)")
    print(f"  z:     {z_measured:.4f} m  (卷尺测量)")
    print(f"  yaw:   {yaw_deg:.2f} deg")
    print(f"  pitch: {pitch_deg:.2f} deg")
    print(f"  优化残差: {result.cost:.6f}")

    # 7. 误差分析
    errors = compute_errors(measurements, tx, ty, yaw_deg, pitch_deg)
    rms = math.sqrt(np.mean([e[4]**2 for e in errors]))

    errors_before = compute_errors(
        measurements, initial["x"], initial["y"], initial["yaw"], initial["pitch"])
    rms_before = math.sqrt(np.mean([e[4]**2 for e in errors_before]))

    print(f"\n  RMS 误差: {rms_before:.4f}m (标定前) → {rms:.4f}m (标定后)")
    print()
    print("  逐点误差:")
    print(f"  {'#':>3}  {'GT(x,y)':>14}  {'Pred(x,y)':>14}  {'Err(m)':>8}")
    for i, (x_gt, y_gt, rx, ry, err) in enumerate(errors):
        print(f"  {i+1:>3}  ({x_gt:>5.2f},{y_gt:>5.2f})  ({rx:>5.2f},{ry:>5.2f})  {err:>8.4f}")

    if rms > 0.10:
        print(f"\n  [警告] RMS误差 {rms:.3f}m > 0.10m，建议检查测量是否准确后重新标定")

    # 8. 输出可粘贴的配置
    hfov_map = {"head": 79.15, "chest": 79.16, "left_arm": 78.69, "right_arm": 79.38}
    hfov = hfov_map.get(camera_name, 79.0)

    print(f"\n  复制到 follow/config.py:")
    print(f'    "{camera_name}": {{')
    print(f'        "x": {tx:.4f}, "y": {ty:.4f}, "z": {z_measured:.3f},')
    print(f'        "yaw": {yaw_deg:.2f}, "pitch": {pitch_deg:.2f},')
    print(f'        "hfov": {hfov},')
    print(f'        "image_width": 640,')
    print(f'        "image_height": 480,')
    print(f'    }},')

    # 9. 可视化
    try:
        params_before = (initial["x"], initial["y"], initial["yaw"], initial["pitch"])
        params_after = (tx, ty, yaw_deg, pitch_deg)
        plot_results(measurements, params_before, params_after)
    except Exception as e:
        print(f"\n  [可视化跳过] {e}")

    # 清理
    pipeline.stop()
    cv2.destroyAllWindows()
    print("\n  标定完成!")


if __name__ == "__main__":
    main()
