"""task1 人机交互 —— 可调参数，赛前换场地只改这里"""

# ═══════════════════════════════════════════════════════════════════════════
#  llm-vl 
# ═══════════════════════════════════════════════════════════════════════════
LLM_VL_SERVER = "http://127.0.0.1:8003/v1"

# ── 场地站点 ID（对应地图上的点位名称）────────────────────────────────────
STATION_START = "LM1"          # 机器人起始位置
STATION_DOOR = "LM2"            # 入口门
STATION_OBSERVATION = "LM6" # 第二个观察座位的站点
STATION_SEAT1 = "LM3" # 五个位置，需要3个站点才能全部到达
STATION_SEAT2 = "LM4"
STATION_SEAT3 = "LM5"

# ── 座位状态 初始设为未知  ─────────────────────────────────────────────────

SEATS = [
    {"id": "seat_1", "occupied": None, "box1": [0, 0, 0, 0], "box2": [138, 11, 239, 190]},  # box1 是在第一个观察位置看到的座位框，box2 是在第二个观察位置看到的座位框，如果某个位置看不到这个座位就设为 [0,0,0,0]
    {"id": "seat_2", "occupied": None, "box1": [375, 146, 637, 444], "box2": [0, 0, 0, 0]},
    {"id": "seat_3", "occupied": None, "box1": [0,0,0,0 ], "box2": [210, 132, 404, 338]},
    {"id": "seat_4", "occupied": None, "box1": [230, 23, 330, 103], "box2": [0, 0, 0, 0]},
    {"id": "seat_5", "occupied": None, "box1": [0, 0, 0, 0], "box2": [354, 10, 464, 125]},
]
PRE_OCCUPIED_SEATS = []
HOST_SEATS = "seat_3"

# 位置对应的导航站点；此处 angle 仍沿用导航接口既有配置语义
SEATS_MAPPING = [
    {"seat_id": "seat_1", "nav_id": STATION_SEAT1, "angle": -9.55},
    {"seat_id": "seat_2", "nav_id": STATION_SEAT1, "angle": -59.90},
    {"seat_id": "seat_3", "nav_id": STATION_SEAT2, "angle": 154.595},
    {"seat_id": "seat_4", "nav_id": STATION_SEAT2, "angle": 90},
    {"seat_id": "seat_5", "nav_id": STATION_SEAT3, "angle": 172.409},
]

# 介绍阶段使用的朝向配置：
# 键1 = 当前所在导航站点
# 键2 = 需要看向的目标座位
# 值 = 底盘原地转向的目标角度（度）
INTRO_LOOK_ANGLES_DEG = {
    STATION_SEAT1: {
        "seat_1": 0,
        "seat_2": -56.91,
        "seat_3": -41.50,
        "seat_4": -19.21,
        "seat_5": 0,
    },
    STATION_SEAT2: {
        "seat_1": 131.31,
        "seat_2": 156.99,
        "seat_3": 163.98,
        "seat_4": 93.63,
        "seat_5": 105.62,
    },
    STATION_SEAT3: {
        "seat_1": 175.8,
        "seat_2": -166.58,
        "seat_3": -143.23,
        "seat_4": -130.75,
        "seat_5": 175.8,
    },
}



# ── 超时（秒） ───────────────────────────────────────────────────────────
MATCH_TIMEOUT = 360        # 比赛总时限 6 分钟
NAV_TIMEOUT = 30           # 单次导航超时
LISTEN_TIMEOUT = 5         # 语音识别等待
ASK_RETRIES = 3            # 语音问答重试次数
FOLLOW_HOST_TIMEOUT = 90   # 跟随主人阶段超时

# ── 云台预设角度 ─────────────────────────────────────────────────────────


# ── 机械臂预设 ───────────────────────────────────────────────────────────
LEFT_HOME_JOINTS = [-4.297, -97.695, -22.042, 18.172, -12.501, 0]  # 初始关节角度（度）
RIGHT_HOME_JOINTS = [30.0, -94.4, -25.7, -60.9, -31.6, 0]  # 初始关节角度（度）
ARM_SPEED = 40  # 运动速度 1~100

# 开门预设
TRAJECTORY_GET_PATH = "/home/blinx/26-home/common/utils/drag_and_play/open_door_trajectory/get.txt"
TRAJECTORY_MOVE_PATH = "/home/blinx/26-home/common/utils/drag_and_play/open_door_trajectory/move.txt"
TRAJECTORY_LEAVE_PATH = "/home/blinx/26-home/common/utils/drag_and_play/open_door_trajectory/leave.txt"

