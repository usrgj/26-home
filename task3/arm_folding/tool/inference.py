"""
ACT 模型真机推理脚本
加载训练好的 ACT checkpoint，以 50Hz 主循环执行模型输出的动作。

用法:
    conda run -n fold python task3/arm_folding/tool/inference.py
    conda run -n fold python task3/arm_folding/tool/inference.py --ckpt task3/arm_folding/act/checkpoints/policy_best.ckpt
"""

import sys
import time
import threading
import pickle
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
ARM_FOLDING_DIR = Path(__file__).resolve().parents[1]
ACT_DIR = ARM_FOLDING_DIR / "act"
ACT_DETR_DIR = ACT_DIR / "detr"
DEFAULT_CKPT = ACT_DIR / "checkpoints" / "policy_best.ckpt"
DEFAULT_STATS = ACT_DIR / "checkpoints" / "dataset_stats.pkl"

for path in (ACT_DETR_DIR, ACT_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    

try:
    import cv2
except ImportError:  # pragma: no cover - runtime environment dependent
    cv2 = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - runtime environment dependent
    np = None

try:
    import torch
except ImportError:  # pragma: no cover - runtime environment dependent
    torch = None

try:
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
except ImportError:  # pragma: no cover - runtime environment dependent
    RoboticArm = None
    rm_thread_mode_e = None

try:
    from common.skills.camera import camera_manager
except ImportError:  # pragma: no cover - runtime environment dependent
    camera_manager = None

try:
    from common.skills.arm.gripper.gripper_servo import Gripper, GripperError
    from common.skills.arm.gripper.gripper_io import IOGripper, IOGripperError
except ImportError:  # pragma: no cover - runtime environment dependent
    Gripper = None
    GripperError = Exception
    IOGripper = None
    IOGripperError = Exception

ACT_IMPORT_ERROR = None
try:
    from policy import ACTPolicy
except ImportError as exc:  # pragma: no cover - runtime environment dependent
    ACTPolicy = None
    ACT_IMPORT_ERROR = exc

# ==================== 配置区 ====================
LEFT_ARM_IP = "192.168.192.18"
RIGHT_ARM_IP = "192.168.192.19"
ARM_PORT = 8080

# 头部广角 USB 摄像头（设备号）
CAM_HEAD_DEV = 10
# 腕部 RealSense 序列号
CAM_LEFT_WRIST = "141722075710"
CAM_RIGHT_WRIST = "239722070896"

# 折衣服初始关节角度（度）
LEFT_INIT_JOINTS  = [-121.994, -27.868, -56.127, 47.455, -77.515, 44.013]
RIGHT_INIT_JOINTS = [141.882, -54.571, -12.117, -62.232, -113.411, -27.554]
INIT_SPEED = 20

# RM65-B 6 轴关节限位（度）
JOINT_LIMITS_MIN = (-178, -130, -135, -178, -128, -360)
JOINT_LIMITS_MAX = (178, 130, 135, 178, 128, 360)

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


def ensure_runtime_dependencies():
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if np is None:
        missing.append("numpy")
    if torch is None:
        missing.append("torch")
    if RoboticArm is None or rm_thread_mode_e is None:
        missing.append("Robotic_Arm.rm_robot_interface")
    if camera_manager is None:
        missing.append("common.skills.camera")
    if Gripper is None or IOGripper is None:
        missing.append("common.skills.arm.gripper")
    if ACTPolicy is None:
        if ACT_IMPORT_ERROR is not None:
            missing.append(f"task3/arm_folding/act ({ACT_IMPORT_ERROR})")
        else:
            missing.append("task3/arm_folding/act")
    if missing:
        raise RuntimeError(f"Missing dependencies: {', '.join(missing)}")


def resolve_resource_path(path_like: Path | str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()

    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    arm_folding_candidate = (ARM_FOLDING_DIR / path).resolve()
    if arm_folding_candidate.exists():
        return arm_folding_candidate

    return cwd_candidate


def resolve_torch_device(device_arg: str):
    """解析推理设备，并打印当前实际使用的后端。"""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求使用 CUDA，但当前 PyTorch 未检测到可用 GPU。")
        device_name = torch.cuda.get_device_name(device)
        print(f"推理设备: {device} ({device_name})")
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
    else:
        print("推理设备: cpu")

    return device


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


class USBCameraGrabber:
    def __init__(self, device_id: int, name: str, width=640, height=480, fps=30):
        self.device_id = device_id
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        self.latest_color = None
        self._stop = threading.Event()

    def start(self):
        cap = cv2.VideoCapture(self.device_id)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开USB摄像头 /dev/video{self.device_id}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap = cap
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if ret and frame is not None:
                self.latest_color = frame

    def get_latest(self):
        return self.latest_color

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        if hasattr(self, "_cap"):
            self._cap.release()


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


def cleanup_runtime(
    left_reader,
    right_reader,
    grabbers,
    left_gripper,
    right_gripper,
    left_arm,
    right_arm,
):
    """统一清理真机资源，确保异常退出也能释放相机和双臂。"""
    print("\n清理中...")

    if left_reader is not None:
        left_reader.stop()
    if right_reader is not None:
        right_reader.stop()

    for g in reversed(grabbers):
        try:
            g.stop()
        except Exception as exc:
            print(f"  释放相机 {getattr(g, 'name', '?')} 失败: {exc}")

    if camera_manager is not None:
        try:
            camera_manager.stop_all()
        except Exception as exc:
            print(f"  camera_manager.stop_all() 失败: {exc}")

    if grabbers:
        time.sleep(0.5)

    if left_arm is not None or right_arm is not None:
        print("  移动到初始位...")
    if left_arm is not None:
        try:
            left_arm.rm_movej(LEFT_INIT_JOINTS, v=INIT_SPEED, r=0, connect=0, block=1)
        except Exception:
            pass
    if right_arm is not None:
        try:
            right_arm.rm_movej(RIGHT_INIT_JOINTS, v=INIT_SPEED, r=0, connect=0, block=1)
        except Exception:
            pass

    if left_gripper is not None:
        try:
            left_gripper.open(block=True)
        except Exception:
            pass
    if right_gripper is not None:
        try:
            right_gripper.open()
        except Exception:
            pass
        try:
            right_gripper.stop()
        except Exception:
            pass

    if left_arm is not None:
        try:
            left_arm.rm_delete_robot_arm()
        except Exception:
            pass
    if right_arm is not None:
        try:
            right_arm.rm_delete_robot_arm()
        except Exception:
            pass

    print("  ✅ 完成")


# ─── 模型加载 ──────────────────────────────────

def load_policy(ckpt_path, stats_path, device):
    """加载 ACT 模型和归一化统计量。"""
    print(f"加载模型: {ckpt_path}")
    policy = ACTPolicy(POLICY_CONFIG)
    policy.load_state_dict(torch.load(str(ckpt_path), map_location='cpu', weights_only=False))
    policy.to(device)
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

def preprocess_qpos(left_deg, right_deg, left_grip, right_grip, qpos_mean, qpos_std, device):
    """关节角(度) + 夹爪状态 → 归一化 qpos tensor (1, 14)。"""
    left_rad = np.radians(left_deg).astype(np.float32)
    right_rad = np.radians(right_deg).astype(np.float32)
    lg = float(left_grip) if left_grip is not None else 1.0
    rg = float(right_grip) if right_grip is not None else 1.0

    qpos_raw = np.array([*left_rad, lg, *right_rad, rg], dtype=np.float32)
    qpos_norm = (qpos_raw - qpos_mean) / qpos_std
    return torch.from_numpy(qpos_norm).float().to(device).unsqueeze(0)


def preprocess_images(frames, device):
    """BGR 帧 dict → tensor (1, num_cams, C, H, W)，仅 /255 转 float。
    ImageNet normalize 由 ACTPolicy.__call__ 内部完成。
    """
    images = []
    for internal_name, _ in CAM_ORDER:
        bgr = frames[internal_name]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
        images.append(img)
    return torch.stack(images).unsqueeze(0).to(device)  # (1, 3, C, H, W)


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
    parser.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    parser.add_argument('--stats', type=Path, default=DEFAULT_STATS)
    parser.add_argument(
        '--device',
        choices=('auto', 'cuda', 'cpu'),
        default='auto',
        help='推理设备，默认 auto（优先 CUDA）',
    )
    parser.add_argument('--max_steps', type=int, default=3000, help='最大执行步数')
    args = parser.parse_args()
    ensure_runtime_dependencies()

    ckpt_path = resolve_resource_path(args.ckpt)
    stats_path = resolve_resource_path(args.stats)
    device = resolve_torch_device(args.device)

    left_arm = None
    right_arm = None
    left_gripper = None
    right_gripper = None
    left_reader = None
    right_reader = None
    grabbers = []
    steps_executed = 0
    elapsed = 0.0

    try:
        # ── 加载模型 ──
        policy, qpos_mean, qpos_std, action_mean, action_std = load_policy(ckpt_path, stats_path, device)
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

        print(f"  启动 cam_head (USB /dev/video{CAM_HEAD_DEV}) ...")
        head_grabber = USBCameraGrabber(CAM_HEAD_DEV, "cam_head")
        head_grabber.start()
        grabbers.append(head_grabber)

        for name, serial in (
            ("cam_left_wrist", CAM_LEFT_WRIST),
            ("cam_right_wrist", CAM_RIGHT_WRIST),
        ):
            print(f"  启动 {name} ({serial}) ...")
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

        with torch.inference_mode():
            for t in range(args.max_steps):
                # 读取观测
                left_deg = left_reader.get_latest()
                right_deg = right_reader.get_latest()
                if left_deg is None or right_deg is None:
                    continue

                frames = {g.name: g.get_latest() for g in grabbers}
                if any(frames[g.name] is None for g in grabbers):
                    continue

                # 模型推理（每 chunk_size 步查询一次）
                if t % chunk_size == 0:
                    qpos_tensor = preprocess_qpos(
                        left_deg, right_deg,
                        left_gripper.is_open, right_gripper.is_open,
                        qpos_mean, qpos_std,
                        device,
                    )
                    image_tensor = preprocess_images(frames, device)
                    all_actions = policy(qpos_tensor, image_tensor)  # (1, 50, 14)

                raw_action = all_actions[:, t % chunk_size]

                # 后处理
                action = postprocess_action(raw_action, action_mean, action_std)

                # 执行
                execute_action(action, left_arm, right_arm, left_gripper, right_gripper)
                steps_executed = t + 1

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
    finally:
        if 't_start' in locals():
            elapsed = time.perf_counter() - t_start
            hz = steps_executed / elapsed if elapsed > 0 else 0.0
            print(f"\n推理结束: {steps_executed} 步, {elapsed:.1f}s, 实际 {hz:.1f} Hz")

        cleanup_runtime(
            left_reader,
            right_reader,
            grabbers,
            left_gripper,
            right_gripper,
            left_arm,
            right_arm,
        )


if __name__ == "__main__":
    main()
