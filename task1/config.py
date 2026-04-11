"""task1 人机交互 —— 可调参数，赛前换场地只改这里"""

# ═══════════════════════════════════════════════════════════════════════════
#  llm-vl 
# ═══════════════════════════════════════════════════════════════════════════
LLM_VL_SERVER = "http://127.0.0.1:8004/v1"

# ── 场地站点 ID（对应地图上的点位名称）────────────────────────────────────
STATION_START = "start"          # 机器人起始位置
STATION_DOOR = "door"            # 入口门
STATION_GUEST1 = "" # 第一个客人坐的位置
STATION_GUEST2 = "" # 第二个客人坐的位置
STATION_OBSERVATION = "observation" # 第二个观察座位的站点
STATION_SEAT1 = "seat1" # 五个位置，需要3个站点才能全部到达
STATION_SEAT2 = "seat2"
STATION_SEAT3 = "seat3"

# ── 座位状态 初始设为未知  ─────────────────────────────────────────────────
SEATS = [
    {"id": "seat_1", "occupied": None, "box1": [0, 190, 165, 450], "box2": [0, 0, 0, 0]},  # box1 是在第一个观察位置看到的座位框，box2 是在第二个观察位置看到的座位框，如果某个位置看不到这个座位就设为 [0,0,0,0]
    {"id": "seat_2", "occupied": None, "box1": [515, 326, 637, 478], "box2": [0, 0, 0, 0]},
    {"id": "seat_3", "occupied": None, "box1": [0, 0, 0, 0], "box2": [0, 0, 0, 0]},
    {"id": "seat_4", "occupied": None, "box1": [236, 126, 308, 147], "box2": [0, 0, 0, 0]},
    {"id": "seat_5", "occupied": None, "box1": [0, 0, 0, 0], "box2": [0, 0, 0, 0]},
]
PRE_OCCUPIED_SEATS = []

# 位置对应的导航站点；此处 angle 仍沿用导航接口既有配置语义
SEATS_MAPPING = [
    {"seat_id": "seat_1", "nav_id": "seat_nav1", "angle": 0},
    {"seat_id": "seat_2", "nav_id": "seat_nav1", "angle": 0},
    {"seat_id": "seat_3", "nav_id": "seat_nav2", "angle": 0},
    {"seat_id": "seat_4", "nav_id": "seat_nav2", "angle": 0},
    {"seat_id": "seat_5", "nav_id": "seat_nav3", "angle": 0},
]

# 介绍阶段使用的朝向配置：
# 键1 = 当前所在导航站点
# 键2 = 需要看向的目标座位
# 值 = 底盘原地转向的目标角度（度）
INTRO_LOOK_ANGLES_DEG = {
    "seat_nav1": {
        "seat_1": 0.0,
        "seat_2": 0.0,
        "seat_3": 0.0,
        "seat_4": 0.0,
        "seat_5": 0.0,
    },
    "seat_nav2": {
        "seat_1": 0.0,
        "seat_2": 0.0,
        "seat_3": 0.0,
        "seat_4": 0.0,
        "seat_5": 0.0,
    },
    "seat_nav3": {
        "seat_1": 0.0,
        "seat_2": 0.0,
        "seat_3": 0.0,
        "seat_4": 0.0,
        "seat_5": 0.0,
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
ARM_RECEIVE_BAG = 500000
ARM_PLACE_BAG = 300000
ARM_HOME = 0
LEFT_HOME_JOINTS = [0, 0, 0, 0, 0, 0]  # 初始关节角度（度）
RIGHT_HOME_JOINTS = [0, 0, 0, 0, 0, 0]  # 初始关节角度（度）
ARM_SPEED = 50  # 运动速度 1~100

# ── 饮料列表 ───────────────────────────────────────────────────────────
COMMON_DRINKS = ["可乐", "雪碧", "芬达", "美年达", "七喜", "果汁", "橙汁", "苹果汁",
                              "牛奶", "酸奶", "水", "矿泉水", "茶", "红茶", "绿茶", "乌龙茶",
                              "咖啡", "拿铁", "卡布奇诺", "啤酒", "红酒"]
