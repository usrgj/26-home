"""
双臂关节 + 3路相机 综合频率测试 / 数采脚本
- 双臂进入拖动示教模式（灵敏度100，不保存轨迹）
- 头部 USB 相机 + 2 路腕部 RealSense 后台更新
- 主循环以 50Hz 节拍采集：双臂关节 + 相机最新帧
- 键盘控制夹爪开合（非阻塞，不影响 50Hz 节拍）
- 统计各环节延迟，验证能否维持 50Hz
"""

import sys
import time
import tty
import termios
import select
import threading
import statistics
from pathlib import Path
from typing import Callable

import cv2
from Robotic_Arm.rm_robot_interface import *

ROOT_DIR = Path(__file__).resolve().parents[2]
ARM_FOLDING_DIR = ROOT_DIR / "task3" / "arm_folding"
DATA_DIR = ARM_FOLDING_DIR / "data"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.skills.camera import camera_manager
from common.skills.arm.gripper.gripper_servo import Gripper, GripperError
from common.skills.arm.gripper.gripper_io import IOGripper, IOGripperError

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


def run_parallel_actions(actions: dict[str, Callable[[], object]]) -> list[str]:
    """并行执行多个动作，返回错误描述列表。"""
    errors = []
    lock = threading.Lock()

    def worker(name: str, fn):
        try:
            result = fn()
            if isinstance(result, int) and result != 0:
                raise RuntimeError(f"返回错误码 {result}")
        except (GripperError, IOGripperError, RuntimeError) as e:
            with lock:
                errors.append(f"{name}: {e}")
        except Exception as e:  # pragma: no cover - 真机异常兜底
            with lock:
                errors.append(f"{name}: {e}")

    threads = [
        threading.Thread(target=worker, args=(name, fn), daemon=True)
        for name, fn in actions.items()
        if fn is not None
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


# ─── 相机异步取帧线程 ────────────────────────────

class CameraGrabber:
    """后台轮询 RealSense 缓存帧，避免主循环阻塞在相机读取上。"""

    def __init__(self, serial: str, name: str):
        self.serial = serial
        self.name = name
        self.latest_color = None  # np.ndarray or None
        self.latest_depth = None
        self.latest_timestamp = 0.0
        self.frame_count = 0
        self.stat_label = "新帧间隔"
        self.stat_samples = []
        self._stop = threading.Event()
        self._last_seen_timestamp = 0.0

    def start(self):
        cam = camera_manager.start(self.serial)
        self._cam = cam
        self._thread = threading.Thread(target=self._loop, args=(cam,), daemon=True)
        self._thread.start()

    def _loop(self, cam):
        while not self._stop.is_set():
            color, depth = cam.get_frames()
            timestamp = cam.latest_timestamp
            if color is not None and timestamp > 0 and timestamp != self._last_seen_timestamp:
                if self._last_seen_timestamp > 0:
                    self.stat_samples.append((timestamp - self._last_seen_timestamp) * 1000)
                self._last_seen_timestamp = timestamp
                self.latest_color = color
                self.latest_depth = depth
                self.latest_timestamp = timestamp
                self.frame_count += 1
            else:
                time.sleep(0.001)

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
        self.latest_timestamp = 0.0
        self.frame_count = 0
        self.stat_label = "OpenCV read 耗时"
        self.stat_samples = []
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
                now = time.time()
                self.latest_color = frame
                self.latest_timestamp = now
                self.frame_count += 1
                self.stat_samples.append((t1 - t0) * 1000)

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

def gripper_action(left_fn, right_fn, label: str, busy_event: threading.Event):
    """在子线程中并行执行左右夹爪动作。"""
    def run():
        try:
            errors = run_parallel_actions({
                "左夹爪": left_fn,
                "右夹爪": right_fn,
            })
            if errors:
                print(f"\n  ❌ {label} 失败: {'; '.join(errors)}")
            else:
                print(f"\n  ✅ {label} 完成")
        finally:
            busy_event.clear()

    busy_event.set()
    threading.Thread(target=run, daemon=True).start()


# ─── 主循环 ──────────────────────────────────────

def reset_to_init(left_arm, right_arm, busy_event: threading.Event):
    """退出拖动示教 → 双臂复位到初始位 → 重新进入拖动示教。"""
    print("\n  ⏳ 复位中...")
    reentered_drag = False
    try:
        left_arm.rm_stop_drag_teach()
        right_arm.rm_stop_drag_teach()

        errors = run_parallel_actions({
            "左臂复位": lambda: left_arm.rm_movej(
                LEFT_INIT_JOINTS, v=INIT_SPEED, r=0, connect=0, block=1
            ),
            "右臂复位": lambda: right_arm.rm_movej(
                RIGHT_INIT_JOINTS, v=INIT_SPEED, r=0, connect=0, block=1
            ),
        })
        if errors:
            raise RuntimeError("; ".join(errors))

        left_arm.rm_start_drag_teach(0)
        right_arm.rm_start_drag_teach(0)
        left_arm.rm_set_drag_teach_sensitivity(SENSITIVITY)
        right_arm.rm_set_drag_teach_sensitivity(SENSITIVITY)
        reentered_drag = True
        print("  ✅ 复位完成，已重新进入拖动示教")
    except Exception as e:
        print(f"\n  ❌ 复位失败: {e}")
    finally:
        if not reentered_drag:
            try:
                left_arm.rm_start_drag_teach(0)
                right_arm.rm_start_drag_teach(0)
                left_arm.rm_set_drag_teach_sensitivity(SENSITIVITY)
                right_arm.rm_set_drag_teach_sensitivity(SENSITIVITY)
            except Exception as e:  # pragma: no cover - 真机异常兜底
                print(f"\n  ⚠ 无法恢复拖动示教模式: {e}")
        busy_event.clear()


def wait_for_idle(busy_event: threading.Event, label: str, timeout: float = 10.0):
    """清理前等待后台动作结束，避免与断连或复位冲突。"""
    if not busy_event.is_set():
        return

    print(f"等待{label}结束...")
    deadline = time.time() + timeout
    while busy_event.is_set() and time.time() < deadline:
        time.sleep(0.05)

    if busy_event.is_set():
        print(f"  ⚠ {label} 超时未结束，继续执行清理")


def main_loop(left_reader, right_reader, grabbers,
              left_gripper, right_gripper, recorder,
              left_arm, right_arm,
              gripper_busy: threading.Event,
              reset_busy: threading.Event):
    """50Hz主循环：采集数据 + 非阻塞键盘控制夹爪 + HDF5录制。按 q 退出。"""
    tick_latencies = []
    loop_latencies = []
    cam_fetch_latencies = []

    tick_count = 0
    t_start = time.perf_counter()

    def trigger_gripper(left_fn, right_fn, label: str):
        if reset_busy.is_set():
            print("\n  ⚠ 复位中，暂不接受夹爪指令")
            return
        if gripper_busy.is_set():
            print("\n  ⚠ 夹爪正在执行动作，请稍候")
            return
        gripper_action(left_fn, right_fn, label, gripper_busy)

    with RawTerminal():
        while True:
            t_tick_start = time.perf_counter()

            # 非阻塞键盘检测
            ch = RawTerminal.read_char()
            if ch:
                ch = ch.lower()
                if ch == "o":
                    trigger_gripper(
                        lambda: left_gripper.open(block=True),
                        lambda: right_gripper.open(),
                        "双手打开")
                elif ch == "c":
                    trigger_gripper(
                        lambda: left_gripper.grab_hold(block=True),
                        lambda: right_gripper.close(),
                        "双手闭合")
                elif ch == "1":
                    trigger_gripper(
                        lambda: left_gripper.open(block=True),
                        None, "左手打开")
                elif ch == "2":
                    trigger_gripper(
                        None,
                        lambda: right_gripper.open(),
                        "右手打开")
                elif ch == "3":
                    trigger_gripper(
                        lambda: left_gripper.grab_hold(block=True),
                        None, "左手闭合")
                elif ch == "4":
                    trigger_gripper(
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
                    elif gripper_busy.is_set():
                        print("\n  ⚠ 夹爪动作执行中，请稍后再复位")
                    elif reset_busy.is_set():
                        print("\n  ⚠ 已在复位中，请勿重复触发")
                    else:
                        reset_busy.set()
                        threading.Thread(
                            target=reset_to_init,
                            args=(left_arm, right_arm, reset_busy),
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
    if duration > 0:
        print(f"  实际频率: {len(latencies) / duration:.1f} Hz")


def print_summary(run_stats, left_reader, right_reader, grabbers):
    tick_count, actual_duration, tick_lats, loop_lats, cam_fetch_lats = run_stats

    print(f"\n{'═' * 55}")
    print(f"  综合测试结果 (共 {tick_count} tick, {actual_duration:.2f}s)")
    print(f"{'═' * 55}")

    print_stats("主循环 tick 耗时（取数据+组装，不含sleep）", tick_lats, actual_duration)
    print_stats("主循环实际间隔（含sleep，目标20ms）", loop_lats, actual_duration)
    print_stats("相机取帧引用耗时（非阻塞读引用）", cam_fetch_lats, actual_duration)
    print_stats("左臂 rm_get_joint_degree 调用耗时", left_reader.latencies, actual_duration)
    print_stats("右臂 rm_get_joint_degree 调用耗时", right_reader.latencies, actual_duration)

    for g in grabbers:
        print_stats(f"{g.name} {g.stat_label}", g.stat_samples, actual_duration)
        fps = g.frame_count / actual_duration if actual_duration > 0 else 0.0
        print(f"  总帧数: {g.frame_count}  ({fps:.1f} fps)")

    print(f"\n{'═' * 55}")
    if tick_lats and actual_duration > 0:
        actual_freq = tick_count / actual_duration
        mean_tick = statistics.mean(tick_lats)
        print(f"  主循环实际频率: {actual_freq:.1f} Hz (目标 {FREQ} Hz)")
        print(f"  主循环 tick 均值: {mean_tick:.3f} ms (预算 {DT*1000:.1f} ms)")
        if actual_freq >= FREQ * 0.95:
            print("  ✅ 50Hz 节拍可维持")
        else:
            print("  ❌ 50Hz 节拍不足，需排查瓶颈")
    else:
        print("  ⚠ 主循环没有产生有效 tick，跳过频率判定")
    print(f"{'═' * 55}")


def main():
    left_arm = None
    right_arm = None
    left_gripper = None
    right_gripper = None
    left_reader = None
    right_reader = None
    grabbers = []
    recorder = None
    run_stats = None
    gripper_busy = threading.Event()
    reset_busy = threading.Event()

    try:
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

        print(f"  启动 cam_head (USB /dev/video{CAM_HEAD_DEV}) ...")
        g_head = USBCameraGrabber(CAM_HEAD_DEV, "cam_head")
        g_head.start()
        grabbers.append(g_head)

        for name, serial in [("cam_left_wrist", CAM_LEFT_WRIST),
                             ("cam_right_wrist", CAM_RIGHT_WRIST)]:
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
            if g.latest_color is not None:
                h, w = g.latest_color.shape[:2]
                print(f"  ✅ {g.name} 就绪 ({w}x{h})")
            else:
                print(f"  ❌ {g.name} 未出帧")

        # ── 创建夹爪控制器 ──
        print("\n初始化夹爪...")
        left_gripper = Gripper(left_arm)
        right_gripper = IOGripper(right_arm)
        print("  ✅ 夹爪初始化完成")

        # ── 启动关节读取线程 ──
        left_reader = ArmReader(left_arm, "左臂")
        right_reader = ArmReader(right_arm, "右臂")
        left_reader.start()
        right_reader.start()
        time.sleep(0.1)  # 让读取线程跑几轮

        # ── 创建录制器 ──
        recorder = EpisodeRecorder(str(DATA_DIR), freq=FREQ)
        print(f"  录制器就绪，输出目录: {DATA_DIR} (下一 episode: {recorder._episode_idx})")

        # ── 50Hz主循环 ──
        print(f"\n主循环启动 (目标 {FREQ}Hz，按 q 退出)")
        print("─" * 55)
        print("  [o]双手开 [c]双手合 [1/2]左/右开 [3/4]左/右合")
        print("  [r]开始/停止录制  [i]复位到初始位  [q]退出")
        print("─" * 55)
        run_stats = main_loop(
            left_reader, right_reader, grabbers,
            left_gripper, right_gripper, recorder,
            left_arm, right_arm,
            gripper_busy, reset_busy,
        )
    except KeyboardInterrupt:
        print("\n收到中断，准备清理...")
    finally:
        if recorder is not None and recorder.is_recording:
            recorder.stop_episode()

        wait_for_idle(reset_busy, "复位动作")
        wait_for_idle(gripper_busy, "夹爪动作")

        if left_reader is not None:
            left_reader.stop()
        if right_reader is not None:
            right_reader.stop()
        for g in grabbers:
            g.stop()

        if run_stats is not None and left_reader is not None and right_reader is not None:
            print_summary(run_stats, left_reader, right_reader, grabbers)

        if left_gripper is not None or right_gripper is not None:
            print("\n松开夹爪...")
        if left_gripper is not None:
            try:
                left_gripper.open(block=True)
            except GripperError:
                pass
        if right_gripper is not None:
            try:
                right_gripper.open()
                right_gripper.stop()
            except IOGripperError:
                pass

        if left_arm is not None or right_arm is not None:
            print("退出拖动示教模式...")
        if left_arm is not None:
            try:
                left_arm.rm_stop_drag_teach()
            except Exception:
                pass
            left_arm.rm_delete_robot_arm()
        if right_arm is not None:
            try:
                right_arm.rm_stop_drag_teach()
            except Exception:
                pass
            right_arm.rm_delete_robot_arm()
        if left_arm is not None or right_arm is not None:
            print("完成，双臂已断开。")


if __name__ == "__main__":
    main()
