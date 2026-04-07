"""
头部相机运动控制模块

通过 Modbus RTU 协议控制头部相机的左右旋转（地址01）和上下旋转（地址02）。
两个设备均在 /dev/ttyS1，波特率 38400。

使用示例:
    from head_camera_controller import HeadCameraController

    camera = HeadCameraController(port='/dev/ttyS1')

    # 绝对位置控制（以下写法等价）
    camera.rotate_horizontal(0x1000)
    camera.rotate_horizontal("1000")        # 字符串自动按十六进制解析
    camera.rotate_horizontal("-1000")       # 负方向

    camera.rotate_vertical("2000", speed=400)

    # 相对位置控制（在当前位置基础上偏移）
    camera.rotate_horizontal_rel("500")
    camera.rotate_vertical_rel("-300")

    # 同时控制两个轴
    camera.move(horizontal="1000", vertical="2000")
    camera.move_rel(horizontal="100", vertical="-100")

    # 回到原点
    camera.home()

    # 查看 / 校准位置
    h, v = camera.get_position()
    camera.set_position(horizontal="1000", vertical="2000")

    camera.close()
"""

import serial
import time



# 设备地址
ADDR_HORIZONTAL = "01"  # 左右旋转
ADDR_VERTICAL = "02"    # 上下旋转

# 默认速度
DEFAULT_SPEED = 600


class HeadCameraController:
    def __init__(self, port='/dev/ttyS1', baudrate=38400, timeout=1):
        """
        初始化头部相机控制器

        参数:
            port: 串口端口号，Linux 如 '/dev/ttyS1'，Windows 如 'COM3'
            baudrate: 波特率，默认 38400
            timeout: 串口超时时间（秒）
        """
        self.serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=8,
            parity=serial.PARITY_NONE,
            stopbits=1,
            timeout=timeout
        )
        if not self.serial.is_open:
            self.serial.open()

        # 软件记录的当前位置（假设初始在原点）
        self._pos_horizontal = 0
        self._pos_vertical = 0

    # ---- 绝对位置控制 ----

    def rotate_horizontal(self, position, speed=DEFAULT_SPEED):
        """
        左右旋转到绝对位置

        参数:
            position: 整数(0x1000)或十六进制字符串("1000", "-1000")
            speed: 运动速度，默认 600
        """
        position = self._parse_position(position)
        self._pos_horizontal = position
        self._location_control(ADDR_HORIZONTAL, speed, position)

    def rotate_vertical(self, position, speed=DEFAULT_SPEED):
        """
        上下旋转到绝对位置

        参数:
            position: 整数(0x2000)或十六进制字符串("2000", "-2000")
            speed: 运动速度，默认 600
        """
        position = self._parse_position(position)
        self._pos_vertical = position
        self._location_control(ADDR_VERTICAL, speed, position)

    def move(self, horizontal=None, vertical=None, speed=DEFAULT_SPEED):
        """同时控制两个轴到绝对位置，None 表示不动"""
        if horizontal is not None:
            self.rotate_horizontal(horizontal, speed)
        if vertical is not None:
            self.rotate_vertical(vertical, speed)

    # ---- 相对位置控制 ----

    def rotate_horizontal_rel(self, offset, speed=DEFAULT_SPEED):
        """左右旋转相对偏移"""
        offset = self._parse_position(offset)
        self.rotate_horizontal(self._pos_horizontal + offset, speed)

    def rotate_vertical_rel(self, offset, speed=DEFAULT_SPEED):
        """上下旋转相对偏移"""
        offset = self._parse_position(offset)
        self.rotate_vertical(self._pos_vertical + offset, speed)

    def move_rel(self, horizontal=None, vertical=None, speed=DEFAULT_SPEED):
        """同时对两个轴做相对偏移，None 表示不动"""
        if horizontal is not None:
            self.rotate_horizontal_rel(horizontal, speed)
        if vertical is not None:
            self.rotate_vertical_rel(vertical, speed)

    # ---- 辅助功能 ----

    def home(self, speed=DEFAULT_SPEED):
        """两个轴都回到原点"""
        self.move(horizontal=0, vertical=0, speed=speed)

    def get_position(self):
        """获取当前记录的位置，返回 (horizontal, vertical)"""
        return self._pos_horizontal, self._pos_vertical

    def set_position(self, horizontal=0, vertical=0):
        """手动校准位置记录（不发送指令，仅更新内部状态）"""
        self._pos_horizontal = self._parse_position(horizontal)
        self._pos_vertical = self._parse_position(vertical)

    def enable(self):
        """发送使能指令，激活设备"""
        msg = "FA01F300"
        crc = self._checksum8(bytes.fromhex(msg))
        msg = msg + f"{crc:02X}"
        formatted_hex = ' '.join([msg[i:i + 2] for i in range(0, 10, 2)])
        self._send_command(formatted_hex)

    def close(self):
        """关闭串口连接"""
        if self.serial.is_open:
            self.serial.close()

    # ---- 内部方法（与原始工作脚本完全一致） ----

    def _location_control(self, addr, speed, absAxis):
        """拼装位置控制报文并发送（逻辑与原始脚本一致）"""
        msg = "FA" + str(addr) + "F5"

        speed = f"{speed:X}"
        speed_len = len(speed)
        if speed_len < 4:
            for i in range(4 - speed_len):
                speed = "0" + speed

        msg = msg + speed
        msg = msg + "02"

        absAxis = self._int_to_hex(absAxis)
        msg = msg + absAxis
        msg = str(msg).replace(" ", "")
        crc = self._checksum8(bytes.fromhex(msg))
        msg = msg + f"{crc:02X}"
        formatted_hex = ' '.join([msg[i:i + 2] for i in range(0, len(msg), 2)])
        self._send_command(formatted_hex)

    def _send_command(self, hex_str):
        """发送十六进制指令"""
        if not self.serial.is_open:
            print("串口未打开")
            return False
        try:
            command_bytes = bytes.fromhex(hex_str)
            self.serial.write(command_bytes)
            print(f"[发送指令] {hex_str}")
            return True
        except Exception as e:
            print(f"发送失败: {e}")
            return False

    @staticmethod
    def _parse_position(value):
        """
        解析位置值，支持整数和十六进制字符串

        "1000"  -> 0x1000 (4096)
        "-1000" -> -0x1000 (-4096)
        0x1000  -> 0x1000 (4096)
        """
        if isinstance(value, str):
            value = value.strip()
            negative = value.startswith('-')
            if negative:
                value = value[1:]
            result = int(value, 16)
            return -result if negative else result
        return int(value)

    @staticmethod
    def _int_to_hex(value):
        """将有符号整数转换为4字节大端十六进制字符串"""
        if value < -0x80000000 or value > 0x7FFFFFFF:
            raise ValueError("位置值超出32位有符号整数范围")
        unsigned_value = value & 0xFFFFFFFF
        hex_str = f"{unsigned_value:08X}"
        return ' '.join([hex_str[i:i + 2] for i in range(0, 8, 2)])

    @staticmethod
    def _checksum8(data):
        """计算 CHECKSUM-8 校验码"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return sum(data) & 0xFF

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

pan_tilt = HeadCameraController()