"""task1 人机交互 —— 可调参数，赛前换场地只改这里"""

# ═══════════════════════════════════════════════════════════════════════════
#  llm-vl 
# ═══════════════════════════════════════════════════════════════════════════
LLM_VL_SERVER = "http://127.0.0.1:8004/v1"

# ── 场地站点 ID（对应地图上的点位名称）────────────────────────────────────
STATION_START = "start"          # 机器人起始位置
STATION_DOOR = "door"            # 入口门
STATION_LIVING_ROOM = "living"   # 客厅中心
STATION_GUEST1 = ""
STATION_GUEST2 = ""

# ── 座位站点（按需增减） ─────────────────────────────────────────────────
SEATS = [
    {"id": "seat_1", "occupied": False},
    {"id": "seat_2", "occupied": False},
    {"id": "seat_3", "occupied": False},
    {"id": "seat_4", "occupied": False},
    {"id": "seat_5", "occupied": False},
]

# 位置对应的站点，因为不是每个位置都需要打一个站点，所以单独维护一个映射,角度是弧度
SEATS_MAPPING = [
    {"seat_id": "seat_1", "nav_id": "seat_nav1", "angle": 0},
    {"seat_id": "seat_2", "nav_id": "seat_nav1", "angle": 0},
    {"seat_id": "seat_3", "nav_id": "seat_nav2", "angle": 0},
    {"seat_id": "seat_4", "nav_id": "seat_nav2", "angle": 0},
    {"seat_id": "seat_5", "nav_id": "seat_nav3", "angle": 0},
]

# 已被主人和无关人员占据的座位索引（赛前填入）
PRE_OCCUPIED_SEATS = [0, 1]  # 例如 seat_1, seat_2 被占

# ── 超时（秒） ───────────────────────────────────────────────────────────
MATCH_TIMEOUT = 360        # 比赛总时限 6 分钟
NAV_TIMEOUT = 30           # 单次导航超时
LISTEN_TIMEOUT = 5         # 语音识别等待
ASK_RETRIES = 3            # 语音问答重试次数
FOLLOW_HOST_TIMEOUT = 90   # 跟随主人阶段超时

# ── 云台预设角度 ─────────────────────────────────────────────────────────
HEAD_FORWARD = (0, 0)
HEAD_LOOK_DOWN = (0, -0x800)

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
