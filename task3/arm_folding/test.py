"""
双臂关节 + 3路RealSense 综合频率测试
- 双臂进入拖动示教模式（灵敏度100，不保存轨迹）
- 3路相机后台线程异步推流，主循环只取最新帧
- 主循环以50Hz节拍采集：双臂关节(多线程) + 相机最新帧(非阻塞)
- 键盘控制夹爪开合（非阻塞，不影响50Hz节拍）
- 统计各环节延迟，验证能否维持50Hz
"""

import sys
import os
import time
import tty
import termios
import select
import threading
import statistics
import numpy as np
import cv2
from Robotic_Arm.rm_robot_interface import *

sys.path.insert(0, "camera")
from common.skills.camera import camera_manager

sys.path.insert(0, "gripper")
from common.skills.gripper.gripper_servo import Gripper, GripperError
from common.skills.gripper.gripper_io import IOGripper, IOGripperError

sys.path.insert(0, "tool")
from task3.arm_folding.tool.collect_dataset import EpisodeRecorder

# ==================== 配置区 ====================
LEFT_ARM_IP = "192.168.192.18"
RIGHT_ARM_IP = "192.168.192.19"
ARM_PORT = 8080

# 头部广角USB摄像头（设备号）
CAM_HEAD_DEV = 10                  # /dev/video10
# 腕部RealSense序列号
CAM_LEFT_WRIST = "141722075710"   # 左腕
CAM_RIGHT_WRIST = "239722070896"  # 右腕

FREQ = 50              # 目标频率 Hz
DT = 1.0 / FREQ        # 20ms
SENSITIVITY = 100       # 拖动示教灵敏度

# 折衣服初始关节角度（度）
LEFT_INIT_JOINTS  = [-121.994, -27.868, -56.127, 47.455, -77.515, 44.013]
RIGHT_INIT_JOINTS = [141.882, -54.571, -12.117, -62.232, -113.411, -27.554]
INIT_SPEED = 20        # 复位运动速度 1~100
# ================================================


# ─── 相机异步取帧线程 ────────────────────────────

class CameraGrabber:
    """后台线程持续取帧（RealSense），外部只读最新帧（无锁竞争，引用赋值是原子的）。"""

    def __init__(self, serial: str, name: str):
        self.serial = serial
        self.name = name
        self.latest_color = None  # np.ndarray or None
        self.latest_depth = None
        self.frame_count = 0
        self.grab_latencies = []  # 每次 wait_for_frames 的耗时
        self._stop = threading.Event()

    def start(self):
        cam = camera_manager.start(self.serial)
        self._thread = threading.Thread(target=self._loop, args=(cam,), daemon=True)
        self._thread.start()

    def _loop(self, cam):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            color, depth = cam.get_frames()
            t1 = time.perf_counter()
            if color is not None:
                self.latest_color = color
                self.latest_depth = depth
                self.frame_count += 1
                self.grab_latencies.append((t1 - t0) * 1000)

    def get_latest(self):
        """非阻塞：返回最新帧引用，可能为 None（相机未就绪时）。"""
        return self.latest_color

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        camera_manager.stop(self.serial)


class USBCameraGrabber:
    """后台线程持续取帧（普通USB摄像头，OpenCV），接口与 CameraGrabber 一致。"""

    def __init__(self, device_id: int, name: str, width=640, height=480, fps=30):
        self.device_id = device_id
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        self.latest_color = None
        self.frame_count = 0
        self.grab_latencies = []
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
            t0 = time.perf_counter()
            ret, frame = self._cap.read()
            t1 = time.perf_counter()
            if ret and frame is not None:
                self.latest_color = frame
                self.frame_count += 1
                self.grab_latencies.append((t1 - t0) * 1000)

    def get_latest(self):
        """非阻塞：返回最新帧引用，可能为 None（相机未就绪时）。"""
        return self.latest_color

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        if hasattr(self, '_cap'):
            self._cap.release()


# ─── 关节读取线程 ────────────────────────────────

class ArmReader:
    """后台线程持续读取关节，外部取最新值。"""

    def __init__(self, arm, name: str):
        self.arm = arm
        self.name = name
        self.latest_joints = None
        self.read_count = 0
        self.latencies = []
        self._stop = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            code, joints = self.arm.rm_get_joint_degree()
            t1 = time.perf_counter()
            if code == 0:
                self.latest_joints = joints
                self.read_count += 1
                self.latencies.append((t1 - t0) * 1000)
            # 不做 sleep，尽可能快地刷新

    def get_latest(self):
        return self.latest_joints

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)


# ─── 非阻塞键盘输入 ────────────────────────────────

class RawTerminal:
    """将终端切换为 raw 模式的上下文管理器，退出时自动恢复。"""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = None

    def __enter__(self):
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)  # cbreak 模式：单字符、不回显，但保留信号
        return self

    def __exit__(self, *_):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    @staticmethod
    def read_char():
        """非阻塞读取单字符，无输入时返回 None。"""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


# ─── 夹爪动作（子线程执行，不阻塞主循环）──────────

def gripper_action(left_fn, right_fn, label: str):
    """在子线程中并行执行左右夹爪动作。"""
    def run():
        try:
            threads = []
            if left_fn:
                threads.append(threading.Thread(target=left_fn))
            if right_fn:
                threads.append(threading.Thread(target=right_fn))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            print(f"\n  ✅ {label} 完成")
        except (GripperError, IOGripperError) as e:
            print(f"\n  ❌ {label} 失败: {e}")
    threading.Thread(target=run, daemon=True).start()


# ─── 主循环 ──────────────────────────────────────

def reset_to_init(left_arm, right_arm):
    """退出拖动示教 → 双臂复位到初始位 → 重新进入拖动示教。"""
    print("\n  ⏳ 复位中...")
    left_arm.rm_stop_drag_teach()
    right_arm.rm_stop_drag_teach()

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

    left_arm.rm_start_drag_teach(0)
    right_arm.rm_start_drag_teach(0)
    left_arm.rm_set_drag_teach_sensitivity(SENSITIVITY)
    right_arm.rm_set_drag_teach_sensitivity(SENSITIVITY)
    print("  ✅ 复位完成，已重新进入拖动示教")


def main_loop(left_reader, right_reader, grabbers,
              left_gripper, right_gripper, recorder,
              left_arm, right_arm):
    """50Hz主循环：采集数据 + 非阻塞键盘控制夹爪 + HDF5录制。按 q 退出。"""
    tick_latencies = []
    loop_latencies = []
    cam_fetch_latencies = []

    tick_count = 0
    t_start = time.perf_counter()

    with RawTerminal():
        while True:
            t_tick_start = time.perf_counter()

            # 非阻塞键盘检测
            ch = RawTerminal.read_char()
            if ch:
                ch = ch.lower()
                if ch == "o":
                    gripper_action(
                        lambda: left_gripper.open(block=True),
                        lambda: right_gripper.open(),
                        "双手打开")
                elif ch == "c":
                    gripper_action(
                        lambda: left_gripper.grab_hold(block=True),
                        lambda: right_gripper.close(),
                        "双手闭合")
                elif ch == "1":
                    gripper_action(
                        lambda: left_gripper.open(block=True),
                        None, "左手打开")
                elif ch == "2":
                    gripper_action(
                        None,
                        lambda: right_gripper.open(),
                        "右手打开")
                elif ch == "3":
                    gripper_action(
                        lambda: left_gripper.grab_hold(block=True),
                        None, "左手闭合")
                elif ch == "4":
                    gripper_action(
                        None,
                        lambda: right_gripper.close(),
                        "右手闭合")
                elif ch == "r":
                    if recorder.toggle():
                        print("\n  ● [REC] 开始录制")
                    # toggle 内部会打印保存路径
                elif ch == "i":
                    if recorder.is_recording:
                        print("\n  ⚠ 录制中不可复位，请先按 r 停止录制")
                    else:
                        threading.Thread(
                            target=reset_to_init,
                            args=(left_arm, right_arm),
                            daemon=True,
                        ).start()
                elif ch in ("q", "\x03"):
                    print("\n  提前退出主循环")
                    break

            # 取关节（非阻塞，只读引用）
            left_joints = left_reader.get_latest()
            right_joints = right_reader.get_latest()

            # 取相机最新帧（非阻塞，只读引用）
            t_cam0 = time.perf_counter()
            frames = {}
            for g in grabbers:
                frames[g.name] = g.get_latest()
            t_cam1 = time.perf_counter()
            cam_fetch_latencies.append((t_cam1 - t_cam0) * 1000)

            # HDF5 录制
            if recorder.is_recording and left_joints is not None and right_joints is not None:
                all_frames_ready = all(
                    frames.get(g.name) is not None for g in grabbers
                )
                if all_frames_ready:
                    recorder.record_step(
                        left_joints_deg=left_joints,
                        right_joints_deg=right_joints,
                        left_gripper_open=left_gripper.is_open,
                        right_gripper_open=right_gripper.is_open,
                        frames=frames,
                        timestamp=time.perf_counter() - t_start,
                    )

            t_tick_end = time.perf_counter()
            tick_latencies.append((t_tick_end - t_tick_start) * 1000)

            tick_count += 1

            # 绝对时间戳节拍：避免 sleep 超调累积漂移
            next_tick = t_start + tick_count * DT
            now = time.perf_counter()
            sleep_time = next_tick - now
            if sleep_time > 0.001:  # >1ms 才 sleep，否则直接 busy-wait
                time.sleep(sleep_time - 0.001)  # 提前1ms醒来
            # busy-wait 精确对齐
            while time.perf_counter() < next_tick:
                pass

            t_after_sleep = time.perf_counter()
            loop_latencies.append((t_after_sleep - t_tick_start) * 1000)

    actual_duration = time.perf_counter() - t_start
    return tick_count, actual_duration, tick_latencies, loop_latencies, cam_fetch_latencies


# ─── 统计打印 ────────────────────────────────────

def print_stats(name: str, latencies: list, duration: float):
    if not latencies:
        print(f"\n  {name}: 无数据")
        return
    print(f"\n{'─' * 55}")
    print(f"  {name} ({len(latencies)} 次采样)")
    print(f"{'─' * 55}")
    print(f"  平均: {statistics.mean(latencies):.3f} ms")
    print(f"  中位: {statistics.median(latencies):.3f} ms")
    print(f"  最小: {min(latencies):.3f} ms")
    print(f"  最大: {max(latencies):.3f} ms")
    if len(latencies) > 1:
        print(f"  标准差: {statistics.stdev(latencies):.3f} ms")
    print(f"  实际频率: {len(latencies) / duration:.1f} Hz")


def main():
    # ── 检测相机 ──
    serials = camera_manager.list_devices()
    print(f"检测到 {len(serials)} 个 RealSense: {serials}")

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

    # ── 进入拖动示教 ──
    print("\n进入拖动示教模式（灵敏度100，不保存轨迹）...")
    left_arm.rm_start_drag_teach(0)
    right_arm.rm_start_drag_teach(0)
    left_arm.rm_set_drag_teach_sensitivity(SENSITIVITY)
    right_arm.rm_set_drag_teach_sensitivity(SENSITIVITY)
    print("  已进入拖动示教模式")

    # ── 启动相机后台取帧 ──
    print("\n启动相机...")
    grabbers = []

    # 头部：广角USB摄像头（OpenCV）
    print(f"  启动 cam_head (USB /dev/video{CAM_HEAD_DEV}) ...")
    g_head = USBCameraGrabber(CAM_HEAD_DEV, "cam_head")
    g_head.start()
    grabbers.append(g_head)

    # 腕部：RealSense
    for name, serial in [("cam_left_wrist", CAM_LEFT_WRIST),
                          ("cam_right_wrist", CAM_RIGHT_WRIST)]:
        print(f"  启动 {name} ({serial}) ...")
        g = CameraGrabber(serial, name)
        g.start()
        grabbers.append(g)

    # 等待所有相机出帧
    print("  等待相机就绪...")
    deadline = time.time() + 10
    while time.time() < deadline:
        if all(g.latest_color is not None for g in grabbers):
            break
        time.sleep(0.1)
    for g in grabbers:
        if g.latest_color is not None:
            h, w = g.latest_color.shape[:2]
            print(f"  ✅ {g.name} 就绪 ({w}x{h})")
        else:
            print(f"  ❌ {g.name} 未出帧")

    # ── 创建夹爪控制器 ──
    print("\n初始化夹爪...")
    left_gripper = Gripper(left_arm, raise_on_error=False)
    right_gripper = IOGripper(right_arm, raise_on_error=False)
    print("  ✅ 夹爪初始化完成")

    # ── 启动关节读取线程 ──
    left_reader = ArmReader(left_arm, "左臂")
    right_reader = ArmReader(right_arm, "右臂")
    left_reader.start()
    right_reader.start()
    time.sleep(0.1)  # 让读取线程跑几轮

    # ── 创建录制器 ──
    recorder = EpisodeRecorder("./data", freq=FREQ)
    print(f"  录制器就绪，输出目录: ./data (下一 episode: {recorder._episode_idx})")

    # ── 50Hz主循环 ──
    print(f"\n主循环启动 (目标 {FREQ}Hz，按 q 退出)")
    print("─" * 55)
    print("  [o]双手开 [c]双手合 [1/2]左/右开 [3/4]左/右合")
    print("  [r]开始/停止录制  [i]复位到初始位  [q]退出")
    print("─" * 55)
    tick_count, actual_duration, tick_lats, loop_lats, cam_fetch_lats = main_loop(
        left_reader, right_reader, grabbers,
        left_gripper, right_gripper, recorder,
        left_arm, right_arm,
    )

    # 确保录制中的 episode 被保存
    if recorder.is_recording:
        recorder.stop_episode()

    # ── 停止所有线程 ──
    left_reader.stop()
    right_reader.stop()
    for g in grabbers:
        g.stop()

    # ── 打印统计 ──
    print(f"\n{'═' * 55}")
    print(f"  综合测试结果 (共 {tick_count} tick, {actual_duration}s)")
    print(f"{'═' * 55}")

    print_stats("主循环 tick 耗时（取数据+组装，不含sleep）", tick_lats, actual_duration)
    print_stats("主循环实际间隔（含sleep，目标20ms）", loop_lats, actual_duration)
    print_stats("相机取帧引用耗时（非阻塞读引用）", cam_fetch_lats, actual_duration)
    print_stats("左臂 rm_get_joint_degree 调用耗时", left_reader.latencies, actual_duration)
    print_stats("右臂 rm_get_joint_degree 调用耗时", right_reader.latencies, actual_duration)

    for g in grabbers:
        print_stats(f"{g.name} wait_for_frames 耗时", g.grab_latencies, actual_duration)
        print(f"  总帧数: {g.frame_count}  ({g.frame_count / actual_duration:.1f} fps)")

    # 关键判定
    print(f"\n{'═' * 55}")
    actual_freq = tick_count / actual_duration
    mean_tick = statistics.mean(tick_lats)
    print(f"  主循环实际频率: {actual_freq:.1f} Hz (目标 {FREQ} Hz)")
    print(f"  主循环 tick 均值: {mean_tick:.3f} ms (预算 {DT*1000:.1f} ms)")
    if actual_freq >= FREQ * 0.95:
        print(f"  ✅ 50Hz 节拍可维持")
    else:
        print(f"  ❌ 50Hz 节拍不足，需排查瓶颈")
    print(f"{'═' * 55}")

    # ── 清理退出 ──
    print("\n松开夹爪...")
    try:
        left_gripper.open(block=True)
    except GripperError:
        pass
    try:
        right_gripper.open()
        right_gripper.stop()
    except IOGripperError:
        pass

    print("退出拖动示教模式...")
    left_arm.rm_stop_drag_teach()
    right_arm.rm_stop_drag_teach()
    left_arm.rm_delete_robot_arm()
    right_arm.rm_delete_robot_arm()
    print("完成，双臂已断开。")


if __name__ == "__main__":
    main()
