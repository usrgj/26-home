"""task1 人机交互 —— 可调参数，赛前换场地只改这里"""


# ── 场地站点 ID（对应地图上的点位名称）────────────────────────────────────
STATION_START = "LM1"          # 机器人起始位置
STATION_DOOR = "LM2"            # 入口门

STATION_SEAT1 = "LM3" # 五个位置，需要3个站点才能全部到达
STATION_SEAT2 = "LM4"
STATION_SEAT3 = "LM5"

STATION_OBSERVATION = "observation" # 第二个观察座位的站点

# ── 座位状态 初始设为未知  ─────────────────────────────────────────────────
SEATS = [
    {"id": "seat_1", "occupied": None, "box1": [414, 253, 639, 478], "box2": [0, 0, 0, 0]},  # box1 是在第一个观察位置看到的座位框，box2 是在第二个观察位置看到的座位框，如果某个位置看不到这个座位就设为 [0,0,0,0]
    {"id": "seat_2", "occupied": None, "box1": [472, 102, 621, 217], "box2": [0, 0, 0, 0]},
    {"id": "seat_3", "occupied": None, "box1": [5,6,229,221 ], "box2": [0, 0, 0, 0]},
    {"id": "seat_4", "occupied": None, "box1": [5, 6, 229, 221], "box2": [0, 0, 0, 0]},
    {"id": "seat_5", "occupied": None, "box1": [11, 209, 347, 409], "box2": [0, 0, 0, 0]},
]
PRE_OCCUPIED_SEATS = []

# 位置对应的导航站点；此处 angle 仍沿用导航接口既有配置语义
SEATS_MAPPING = [
    {"seat_id": "seat_1", "nav_id": STATION_SEAT1, "angle": 69.414},
    {"seat_id": "seat_2", "nav_id": STATION_SEAT1, "angle": 5.466},
    {"seat_id": "seat_3", "nav_id": STATION_SEAT2, "angle": -109.08},
    {"seat_id": "seat_4", "nav_id": STATION_SEAT2, "angle": 166.559},
    {"seat_id": "seat_5", "nav_id": STATION_SEAT3, "angle": -92.636},
]

# 介绍阶段使用的朝向配置：
# 键1 = 当前所在导航站点
# 键2 = 需要看向的目标座位
# 值 = 底盘原地转向的目标角度（度）
INTRO_LOOK_ANGLES_DEG = {
    STATION_SEAT1: {
        "seat_1": 81.033,
        "seat_2": 8.405,
        "seat_3": 29.209,
        "seat_4": 58.281,
        "seat_5": 81.033,
    },
    STATION_SEAT2: {
        "seat_1": -142.277,
        "seat_2": -124.590,
        "seat_3": -91.496,
        "seat_4": 173.858,
        "seat_5": -167.985,
    },
    STATION_SEAT3: {
        "seat_1": -92.636,
        "seat_2": -68.807,
        "seat_3": -43.745,
        "seat_4": -8.669,
        "seat_5": -92.636,
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
LEFT_HOME_JOINTS = [-29.1, -95.6, -53.0, 0.1, -34.6, 0]  # 初始关节角度（度）
RIGHT_HOME_JOINTS = [30.0, -94.4, -25.7, -60.9, -31.6, 0]  # 初始关节角度（度）
ARM_SPEED = 40  # 运动速度 1~100

# ── 饮料列表 ───────────────────────────────────────────────────────────
COMMON_DRINKS_ZH = ["可乐", "雪碧", "芬达", "美年达", "七喜", "果汁", "橙汁", "苹果汁",
                              "牛奶", "酸奶", "水", "矿泉水", "茶", "红茶", "绿茶", "乌龙茶",
                              "咖啡", "拿铁", "卡布奇诺", "啤酒", "红酒"]
# ═══════════════════════════════════════════════════════════════════════════
# 饮料列表（英文，用于 ASR 热词和模糊匹配）
# ═══════════════════════════════════════════════════════════════════════════
COMMON_DRINKS_EN = [
    "coke", "coca cola", "pepsi", "sprite", "fanta", "7up", "juice", "orange juice",
    "apple juice", "milk", "yogurt", "water", "mineral water", "tea", "black tea",
    "green tea", "oolong tea", "coffee", "latte", "cappuccino", "beer", "wine",
    "red wine", "white wine", "cocktail", "lemonade", "soda"
]