"""
ACT 模型真机推理脚本
加载训练好的 ACT checkpoint，以 50Hz 主循环执行模型输出的动作。

用法:
    conda run -n fold python tool/inference.py
    conda run -n fold python tool/inference.py --ckpt act/checkpoints/policy_best.ckpt
"""

import sys
import os
import time
import threading
import pickle
import argparse

import cv2
import numpy as np
import torch

from Robotic_Arm.rm_robot_interface import *

sys.path.insert(0, "camera")
from camera_manager import camera_manager

sys.path.insert(0, "gripper")
from gripper_servo import Gripper, GripperError
from gripper_io import IOGripper, IOGripperError

sys.path.insert(0, "act")
sys.path.insert(0, os.path.join("act", "detr"))
from policy import ACTPolicy

# ==================== 配置区 ====================
LEFT_ARM_IP = "192.168.192.18"
RIGHT_ARM_IP = "192.168.192.19"
ARM_PORT = 8080

# 相机序列号
CAM_HEAD = "151222072331"
CAM_LEFT_WRIST = "141722075710"
CAM_RIGHT_WRIST = "239722070896"

# 折衣服初始关节角度（度）
LEFT_INIT_JOINTS  = [-121.994, -27.868, -56.127, 47.455, -77.515, 44.013]
RIGHT_INIT_JOINTS = [141.882, -54.571, -12.117, -62.232, -113.411, -27.554]
INIT_SPEED = 20

# RM65-B 6 轴关节限位（度）
JOINT_LIMITS_MIN = np.array([-178, -130, -135, -178, -128, -360], dtype=np.float64)
JOINT_LIMITS_MAX = np.array([ 178,  130,  135,  178,  128,  360], dtype=np.float64)

FREQ = 50
DT = 1.0 / FREQ

# ACT 模型配置（必须与训练时一致）
POLICY_CONFIG = {
    'lr': 1e-5,
    'num_queries': 50,          # chunk_size
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
}

# 内部相机名 → 训练时相机名（决定图像排列顺序）
CAM_ORDER = [
    ('cam_head', 'cam_high'),
    ('cam_left_wrist', 'cam_left_wrist'),
    ('cam_right_wrist', 'cam_right_wrist'),
]
# ================================================


# ─── 复用 test.py 的硬件抽象 ────────────────────

class CameraGrabber:
    def __init__(self, serial: str, name: str):
        self.serial = serial
        self.name = name
        self.latest_color = None
        self._stop = threading.Event()

    def start(self):
        cam = camera_manager.start(self.serial)
        self._thread = threading.Thread(target=self._loop, args=(cam,), daemon=True)
        self._thread.start()

    def _loop(self, cam):
        while not self._stop.is_set():
            color, _ = cam.get_frames()
            if color is not None:
                self.latest_color = color

    def get_latest(self):
        return self.latest_color

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        camera_manager.stop(self.serial)


class ArmReader:
    def __init__(self, arm, name: str):
        self.arm = arm
        self.name = name
        self.latest_joints = None
        self._stop = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            code, joints = self.arm.rm_get_joint_degree()
            if code == 0:
                self.latest_joints = joints

    def get_latest(self):
        return self.latest_joints

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)


# ─── 模型加载 ──────────────────────────────────

def load_policy(ckpt_path, stats_path):
    """加载 ACT 模型和归一化统计量。"""
    print(f"加载模型: {ckpt_path}")
    policy = ACTPolicy(POLICY_CONFIG)
    policy.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=False))
    policy.cuda()
    policy.eval()

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  模型参数量: {total_params / 1e6:.1f}M")

    print(f"加载统计量: {stats_path}")
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    qpos_mean = stats['qpos_mean']      # (14,) numpy float32
    qpos_std = stats['qpos_std']
    action_mean = stats['action_mean']
    action_std = stats['action_std']

    return policy, qpos_mean, qpos_std, action_mean, action_std


# ─── 观测预处理 ─────────────────────────────────

def preprocess_qpos(left_deg, right_deg, left_grip, right_grip, qpos_mean, qpos_std):
    """关节角(度) + 夹爪状态 → 归一化 qpos tensor (1, 14)。"""
    left_rad = np.radians(left_deg).astype(np.float32)
    right_rad = np.radians(right_deg).astype(np.float32)
    lg = float(left_grip) if left_grip is not None else 1.0
    rg = float(right_grip) if right_grip is not None else 1.0

    qpos_raw = np.array([*left_rad, lg, *right_rad, rg], dtype=np.float32)
    qpos_norm = (qpos_raw - qpos_mean) / qpos_std
    return torch.from_numpy(qpos_norm).float().cuda().unsqueeze(0)


def preprocess_images(frames):
    """BGR 帧 dict → tensor (1, num_cams, C, H, W)，仅 /255 转 float。
    ImageNet normalize 由 ACTPolicy.__call__ 内部完成。
    """
    images = []
    for internal_name, _ in CAM_ORDER:
        bgr = frames[internal_name]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
        images.append(img)
    return torch.stack(images).unsqueeze(0).cuda()  # (1, 3, C, H, W)


# ─── 动作后处理与执行 ────────────────────────────

def postprocess_action(raw_action_tensor, action_mean, action_std):
    """反归一化 action tensor → numpy (14,)。"""
    action = raw_action_tensor.squeeze().cpu().numpy()
    return action * action_std + action_mean


def execute_action(action, left_arm, right_arm, left_gripper, right_gripper):
    """将 14 维 action 发送到双臂和夹爪。

    action: [left_6j_rad, left_grip, right_6j_rad, right_grip]
    """
    left_joints_deg = np.clip(np.degrees(action[0:6]), JOINT_LIMITS_MIN, JOINT_LIMITS_MAX).tolist()
    right_joints_deg = np.clip(np.degrees(action[7:13]), JOINT_LIMITS_MIN, JOINT_LIMITS_MAX).tolist()

    # 透传关节角度
    left_arm.rm_movej_canfd(left_joints_deg, follow=False)
    right_arm.rm_movej_canfd(right_joints_deg, follow=False)

    # 夹爪控制
    left_grip = action[6]
    right_grip = action[13]

    if left_grip > 0.5 and left_gripper.is_open != 1:
        left_gripper.open(block=False)
    elif left_grip <= 0.5 and left_gripper.is_open != 0:
        left_gripper.grab_hold(block=False)

    if right_grip > 0.5 and right_gripper.is_open != 1:
        right_gripper.open()
    elif right_grip <= 0.5 and right_gripper.is_open != 0:
        right_gripper.close()


# ─── 主程序 ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ACT 真机推理")
    parser.add_argument('--ckpt', default='act/checkpoints/policy_best.ckpt')
    parser.add_argument('--stats', default='act/checkpoints/dataset_stats.pkl')
    parser.add_argument('--max_steps', type=int, default=3000, help='最大执行步数')
    args = parser.parse_args()

    # ── 加载模型 ──
    policy, qpos_mean, qpos_std, action_mean, action_std = load_policy(args.ckpt, args.stats)
    chunk_size = POLICY_CONFIG['num_queries']

    # ── 连接双臂 ──
    print("\n连接左臂...")
    left_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    lh = left_arm.rm_create_robot_arm(LEFT_ARM_IP, ARM_PORT)
    assert lh.id != -1, "左臂连接失败"
    print(f"  左臂连接成功 (ID={lh.id})")

    print("连接右臂...")
    right_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    rh = right_arm.rm_create_robot_arm(RIGHT_ARM_IP, ARM_PORT)
    assert rh.id != -1, "右臂连接失败"
    print(f"  右臂连接成功 (ID={rh.id})")

    # ── 移动到 Home 位 ──
    print("\n移动到初始位...")
    t_left = threading.Thread(
        target=left_arm.rm_movej,
        args=(LEFT_INIT_JOINTS,),
        kwargs=dict(v=INIT_SPEED, r=0, connect=0, block=1),
    )
    t_right = threading.Thread(
        target=right_arm.rm_movej,
        args=(RIGHT_INIT_JOINTS,),
        kwargs=dict(v=INIT_SPEED, r=0, connect=0, block=1),
    )
    t_left.start()
    t_right.start()
    t_left.join()
    t_right.join()
    print("  ✅ 已到达初始位")

    # ── 初始化夹爪 ──
    left_gripper = Gripper(left_arm, raise_on_error=False)
    right_gripper = IOGripper(right_arm, raise_on_error=False)
    left_gripper.open(block=True)
    right_gripper.open()
    print("  ✅ 夹爪已张开")

    # ── 启动相机 ──
    print("\n启动相机...")
    cam_config = {
        'cam_head': CAM_HEAD,
        'cam_left_wrist': CAM_LEFT_WRIST,
        'cam_right_wrist': CAM_RIGHT_WRIST,
    }
    grabbers = []
    for name, serial in cam_config.items():
        g = CameraGrabber(serial, name)
        g.start()
        grabbers.append(g)

    print("  等待相机就绪...")
    deadline = time.time() + 10
    while time.time() < deadline:
        if all(g.latest_color is not None for g in grabbers):
            break
        time.sleep(0.1)
    for g in grabbers:
        status = "✅" if g.latest_color is not None else "❌"
        print(f"  {status} {g.name}")

    # ── 启动关节读取线程 ──
    left_reader = ArmReader(left_arm, "左臂")
    right_reader = ArmReader(right_arm, "右臂")
    left_reader.start()
    right_reader.start()
    time.sleep(0.2)

    # ── 等待开始 ──
    print("\n" + "═" * 50)
    print("  准备就绪，按 Enter 开始推理，Ctrl+C 中止")
    print("═" * 50)
    input()

    # ── 50Hz 推理主循环 ──
    print(f"开始推理 (最大 {args.max_steps} 步)...")
    t_start = time.perf_counter()
    all_actions = None

    try:
        with torch.inference_mode():
            for t in range(args.max_steps):
                t_tick = time.perf_counter()

                # 读取观测
                left_deg = left_reader.get_latest()
                right_deg = right_reader.get_latest()
                if left_deg is None or right_deg is None:
                    continue

                frames = {g.name: g.get_latest() for g in grabbers}
                if any(frames[g.name] is None for g in grabbers):
                    continue

                # 预处理
                qpos_tensor = preprocess_qpos(
                    left_deg, right_deg,
                    left_gripper.is_open, right_gripper.is_open,
                    qpos_mean, qpos_std,
                )
                image_tensor = preprocess_images(frames)

                # 模型推理（每 chunk_size 步查询一次）
                if t % chunk_size == 0:
                    all_actions = policy(qpos_tensor, image_tensor)  # (1, 50, 14)

                raw_action = all_actions[:, t % chunk_size]

                # 后处理
                action = postprocess_action(raw_action, action_mean, action_std)

                # 执行
                execute_action(action, left_arm, right_arm, left_gripper, right_gripper)

                # 进度打印
                if t % 100 == 0:
                    elapsed = time.perf_counter() - t_start
                    print(f"  步 {t}/{args.max_steps}  ({elapsed:.1f}s)")

                # 50Hz 节拍对齐
                next_tick = t_start + (t + 1) * DT
                now = time.perf_counter()
                sleep_time = next_tick - now
                if sleep_time > 0.001:
                    time.sleep(sleep_time - 0.001)
                while time.perf_counter() < next_tick:
                    pass

    except KeyboardInterrupt:
        print("\n  用户中断")

    elapsed = time.perf_counter() - t_start
    print(f"\n推理结束: {t+1} 步, {elapsed:.1f}s, 实际 {(t+1)/elapsed:.1f} Hz")

    # ── 清理 ──
    print("\n清理中...")
    left_reader.stop()
    right_reader.stop()
    for g in grabbers:
        g.stop()

    # 回到 Home 位
    print("  移动到初始位...")
    t_left = threading.Thread(
        target=left_arm.rm_movej,
        args=(LEFT_INIT_JOINTS,),
        kwargs=dict(v=INIT_SPEED, r=0, connect=0, block=1),
    )
    t_right = threading.Thread(
        target=right_arm.rm_movej,
        args=(RIGHT_INIT_JOINTS,),
        kwargs=dict(v=INIT_SPEED, r=0, connect=0, block=1),
    )
    t_left.start()
    t_right.start()
    t_left.join()
    t_right.join()

    left_gripper.open(block=True)
    right_gripper.open()
    right_gripper.stop()

    left_arm.rm_delete_robot_arm()
    right_arm.rm_delete_robot_arm()
    print("  ✅ 完成")


if __name__ == "__main__":
    main()
