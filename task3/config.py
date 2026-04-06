"""task3 洗衣服 —— 可调参数，赛前换场地只改这里"""

# ── 场地站点 ID ──────────────────────────────────────────────────────────
STATION_START = "start"
STATION_LAUNDRY = "laundry"       # 洗衣区
STATION_WASHER = "washer"         # 洗衣机前
STATION_BASKET = "basket"         # 洗衣篮前
STATION_TABLE = "folding_table"   # 折叠台前

# ── 超时（秒） ───────────────────────────────────────────────────────────
MATCH_TIMEOUT = 420        # 比赛总时限 7 分钟
NAV_TIMEOUT = 30
FOLD_TIMEOUT = 60          # 单件折叠超时

# ── 衣物数量 ─────────────────────────────────────────────────────────────
BASKET_CLOTHES_COUNT = 6   # 篮子里的衣物数量 (6-8)
WASHER_CLOTHES_COUNT = 2   # 洗衣机里的衣物数量 (2-4)

# ── 机械臂预设 ───────────────────────────────────────────────────────────
ARM_PICK_BASKET = 400000
ARM_PICK_WASHER = 300000
ARM_TABLE_HEIGHT = 450000
ARM_HOME = 0
