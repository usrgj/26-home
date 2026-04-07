import os
import sys

# 获取当前脚本的绝对路径 (26-home/task3/arm_folding/tool/slide_locate.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 向上跳三级回到真正的根目录 (26-home)
# tool -> arm_folding -> task3 -> 26-home
root_dir = os.path.abspath(os.path.join(current_dir, "../../.."))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
    print(f"已修正项目根目录为: {root_dir}")

from common.skills.slide_control import slide_control
import time

def send_axis(abs_axis):
    # 发送坐标
    slide_control.device_location_set(abs_axis)
    time.sleep(0.5)
    # 启动控制
    slide_control.device_start("2F")
    time.sleep(0.5)
    slide_control.device_start("3F")
    print("等待运动完成...")
    for i in range(60):  # 最多等待30秒
        time.sleep(0.5)
        sw = slide_control.read_status_word()
        if sw is not None and (sw & 0x0400):  # bit10 = Target Reached
            print("✓ 目标位置已到达！")
            pos = slide_control.read_actual_position()
            break


# 检查是否开启
if not slide_control.serial.is_open:
        slide_control.serial.open()

# 检查使能
#slide_control.switch_to_position_mode()
sw = slide_control.read_status_word()
if sw is not None and (sw & 0x6F) == 0x27:
    print("电机已使能，开始运动...")
    send_axis(-3800000)
else:
    print("使能失败，尝试清除故障...")
    slide_control.clear_fault()
    print("重新设为位置模式")
    slide_control.switch_to_position_mode()
    # 发送速度
    slide_control.device_speed_set(200)
    print("已恢复，请重新运行脚本")