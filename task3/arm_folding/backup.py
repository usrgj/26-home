
from Robotic_Arm.rm_robot_interface import *
from gripper_servo import Gripper, GripperError
from gripper_io import IOGripper, IOGripperError
import time
import sys
 
# ==================== 配置区 ====================
LEFT_ARM_IP = "192.168.192.18"
RIGHT_ARM_IP = "192.168.192.19"
ARM_PORT = 8080
# ================================================

#创建机械臂对象
# left_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
# left_arm.rm_create_robot_arm(LEFT_ARM_IP, ARM_PORT)
# right_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
# right_arm.rm_create_robot_arm(RIGHT_ARM_IP, ARM_PORT)

# left_gripper = Gripper(left_arm)
# right_gripper = IOGripper(right_arm)

# 同时控制两个机械臂的夹爪
# right_gripper.open()
# left_gripper.open()

# right_gripper.close()
# left_gripper.close()


import tty
import termios
import threading
def get_char() -> str:
    """Linux / macOS：切换终端为 raw 模式后读取单字符。"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

# ─── 状态打印 ────────────────────────────────────
 
def print_status(gripper: Gripper):
    """在非阻塞线程中打印夹爪状态，不影响主循环。"""
    try:
        s = gripper.state
        print(f"  状态 → pos={s.position}/1000  force={s.force}  temp={s.temperature}°C  err={s.error}")
    except GripperError as e:
        print(f"  [状态查询失败] {e}")
 
 
def async_action(fn, gripper: Gripper, label: str):
    """在子线程中执行夹爪动作，避免阻塞键盘监听循环。"""
    def run():
        print(f"\n▶ {label} ...")
        try:
            fn()
            print(f"  ✅ {label} 完成")
        except GripperError as e:
            print(f"  ❌ {label} 失败: {e}")
        print_status(gripper)
        print("\n按键: [o] 打开  [c] 闭合  [q] 退出", end="", flush=True)
 
    threading.Thread(target=run, daemon=True).start()
 
 
# ─── 主程序 ──────────────────────────────────────
 
def main():
    print("─" * 45)
    print("  睿尔曼夹爪键盘控制")
    print("─" * 45)
    print(f"  连接中: {LEFT_ARM_IP}:{ARM_PORT} ...")
 
    # 建立连接
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(LEFT_ARM_IP, ARM_PORT)
    if handle.id == -1:
        print("  ❌ 连接失败，请检查 IP/端口，程序退出。")
        return
    print(f"  ✅ 连接成功 (ID={handle.id})")
 
    gripper = Gripper(arm)
 
    # 显示初始状态
    print_status(gripper)
    print("─" * 45)
    print("  按键: [o] 打开  [c] 闭合  [q] 退出")
    print("─" * 45)
    print("\n按键: [o] 打开  [c] 闭合  [q] 退出", end="", flush=True)
 
    # 键盘监听主循环
    busy = threading.Event()   # 防止动作未完成时重复触发
 
    while True:
        ch = get_char().lower()
 
        if ch == "o":
            if busy.is_set():
                print("\n  ⚠ 夹爪正在运动，请稍候...", end="", flush=True)
                continue
            busy.set()
            def do_open():
                try:
                    gripper.open(block=True)
                finally:
                    busy.clear()
            async_action(do_open, gripper, "打开夹爪")
 
        elif ch == "c":
            if busy.is_set():
                print("\n  ⚠ 夹爪正在运动，请稍候...", end="", flush=True)
                continue
            busy.set()
            def do_close():
                try:
                    gripper.grab_hold(block=True)
                finally:
                    busy.clear()
            async_action(do_close, gripper, "闭合夹爪（持续力控）")
 
        elif ch in ("q", "\x03", "\x1b"):  # q / Ctrl+C / Esc
            print("\n\n  正在退出，松开夹爪...")
            try:
                gripper.open(block=True)
            except GripperError:
                pass
            print("  🔌 连接断开，再见！")
            arm.rm_delete_robot_arm()
            break
 
 
if __name__ == "__main__":
    main()



from Robotic_Arm.rm_robot_interface import *
from gripper_servo import Gripper, GripperError
from gripper_io import IOGripper, IOGripperError
import time
import sys
import tty
import termios
import threading

# ==================== 配置区 ====================
LEFT_ARM_IP = "192.168.192.18"
RIGHT_ARM_IP = "192.168.192.19"
ARM_PORT = 8080
# ================================================

'''
# 键盘控制夹爪
def get_char() -> str:
    """Linux / macOS：切换终端为 raw 模式后读取单字符。"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


# ─── 状态打印 ────────────────────────────────────

def print_left_status(gripper: Gripper):
    try:
        s = gripper.state
        print(f"  左臂(RS485) → pos={s.position}/1000  force={s.force}  temp={s.temperature}°C  err={s.error}")
    except GripperError as e:
        print(f"  左臂 [状态查询失败] {e}")


def print_right_status(gripper: IOGripper):
    try:
        s = gripper.state
        print(f"  右臂(IO)    → status={s.status}  IO1={s.io1}  IO2={s.io2}")
    except IOGripperError as e:
        print(f"  右臂 [状态查询失败] {e}")


def print_all_status(left: Gripper, right: IOGripper):
    print_left_status(left)
    print_right_status(right)


PROMPT = "\n按键: [o] 双手打开  [c] 双手闭合  [1/2] 仅左/右开  [3/4] 仅左/右合  [s] 状态  [q] 退出"


def async_action(fn, left: Gripper, right: IOGripper, label: str, busy: threading.Event):
    """在子线程中执行夹爪动作，避免阻塞键盘监听循环。"""
    def run():
        print(f"\n▶ {label} ...")
        try:
            fn()
            print(f"  ✅ {label} 完成")
        except (GripperError, IOGripperError) as e:
            print(f"  ❌ {label} 失败: {e}")
        print_all_status(left, right)
        print(PROMPT, end="", flush=True)
        busy.clear()

    threading.Thread(target=run, daemon=True).start()


# ─── 双臂并行动作 ─────────────────────────────────

def both_open(left: Gripper, right: IOGripper):
    """双手同时张开：左臂RS485阻塞 + 右臂IO延时，用线程并行。"""
    t_right = threading.Thread(target=right.open)
    t_right.start()
    left.open(block=True)
    t_right.join()


def both_close(left: Gripper, right: IOGripper):
    """双手同时闭合：左臂持续力控 + 右臂IO闭合，用线程并行。"""
    t_right = threading.Thread(target=right.close)
    t_right.start()
    left.grab_hold(block=True)
    t_right.join()


# ─── 主程序 ──────────────────────────────────────

def main():
    print("─" * 50)
    print("  睿尔曼双臂夹爪键盘控制")
    print("─" * 50)

    # 连接左臂
    print(f"  连接左臂: {LEFT_ARM_IP}:{ARM_PORT} ...")
    left_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    left_handle = left_arm.rm_create_robot_arm(LEFT_ARM_IP, ARM_PORT)
    if left_handle.id == -1:
        print("  ❌ 左臂连接失败，程序退出。")
        return
    print(f"  ✅ 左臂连接成功 (ID={left_handle.id})")

    # 连接右臂
    print(f"  连接右臂: {RIGHT_ARM_IP}:{ARM_PORT} ...")
    right_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    right_handle = right_arm.rm_create_robot_arm(RIGHT_ARM_IP, ARM_PORT)
    if right_handle.id == -1:
        print("  ❌ 右臂连接失败，程序退出。")
        left_arm.rm_delete_robot_arm()
        return
    print(f"  ✅ 右臂连接成功 (ID={right_handle.id})")

    # 创建夹爪控制器
    left_gripper = Gripper(left_arm)
    right_gripper = IOGripper(right_arm)

    # 显示初始状态
    print_all_status(left_gripper, right_gripper)
    print("─" * 50)
    print("  [o] 双手打开    [c] 双手闭合")
    print("  [1] 仅左手开    [2] 仅右手开")
    print("  [3] 仅左手合    [4] 仅右手合")
    print("  [s] 查询状态    [q] 退出")
    print("─" * 50)
    print(PROMPT, end="", flush=True)

    busy = threading.Event()

    while True:
        ch = get_char().lower()

        if busy.is_set():
            if ch not in ("q", "\x03", "\x1b"):
                print("\n  ⚠ 夹爪正在运动，请稍候...", end="", flush=True)
                continue

        if ch == "o":
            busy.set()
            async_action(
                lambda: both_open(left_gripper, right_gripper),
                left_gripper, right_gripper, "双手打开", busy,
            )

        elif ch == "c":
            busy.set()
            async_action(
                lambda: both_close(left_gripper, right_gripper),
                left_gripper, right_gripper, "双手闭合", busy,
            )

        elif ch == "1":
            busy.set()
            async_action(
                lambda: left_gripper.open(block=True),
                left_gripper, right_gripper, "左手打开", busy,
            )

        elif ch == "2":
            busy.set()
            async_action(
                lambda: right_gripper.open(),
                left_gripper, right_gripper, "右手打开", busy,
            )

        elif ch == "3":
            busy.set()
            async_action(
                lambda: left_gripper.grab_hold(block=True),
                left_gripper, right_gripper, "左手闭合（持续力控）", busy,
            )

        elif ch == "4":
            busy.set()
            async_action(
                lambda: right_gripper.close(),
                left_gripper, right_gripper, "右手闭合", busy,
            )

        elif ch == "s":
            print()
            print_all_status(left_gripper, right_gripper)
            print(PROMPT, end="", flush=True)

        elif ch in ("q", "\x03", "\x1b"):
            print("\n\n  正在退出，松开双手夹爪...")
            try:
                both_open(left_gripper, right_gripper)
            except (GripperError, IOGripperError):
                pass
            right_gripper.stop()
            left_arm.rm_delete_robot_arm()
            right_arm.rm_delete_robot_arm()
            print("  🔌 双臂断开，再见！")
            break


if __name__ == "__main__":
    main()'''

left_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
left_arm.rm_create_robot_arm(LEFT_ARM_IP, ARM_PORT)
right_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
right_arm.rm_create_robot_arm(RIGHT_ARM_IP, ARM_PORT)
print(right_arm.rm_get_joint_degree())
