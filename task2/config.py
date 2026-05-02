"""task2 拾取和放置 —— 可调参数，赛前换场地只改这里"""

from __future__ import annotations

from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
#  场地站点 ID（占位值，赛前按地图点位名称修改）
# ═══════════════════════════════════════════════════════════════════════════

STATION_KITCHEN_ENTRY = "LM1"
STATION_TABLE = "LM2"
STATION_DISHWASHER = "dishwasher"
STATION_TRASH_BIN = "trash_bin"
STATION_CABINET = "LM3"
STATION_TABLE2 = "LM5"

# ═══════════════════════════════════════════════════════════════════════════
#  比赛与导航超时
# ═══════════════════════════════════════════════════════════════════════════

MATCH_TIMEOUT = 420
NAV_TIMEOUT = 40

# ═══════════════════════════════════════════════════════════════════════════
#  视觉采样
# ═══════════════════════════════════════════════════════════════════════════

TABLE_SCAN_CAMERA_ROLE = "chest"
SHELF_SCAN_CAMERA_ROLE = "head"
VISUAL_SAMPLE_COUNT = 12
VISUAL_SAMPLE_INTERVAL_S = 0.12
DETECTION_CONFIDENCE = 0.6
DETECTION_IOU_THRESHOLD = 0.45


CUSTOM_MODEL_PATH = "task2.pt"

# 头部相机中架子四层区域框，按 [x1, y1, x2, y2] 填像素坐标。
# 赛前标定后只需要替换这里的 0 占位值。
SHELF_LAYER_BOXES = [
    {"layer": 1, "box": [166, 14, 477, 155]},
    {"layer": 2, "box": [177, 183, 468, 304]},
    {"layer": 3, "box": [189, 323, 455, 420]},
    {"layer": 4, "box": [208, 432, 446, 478]},
]

# ═══════════════════════════════════════════════════════════════════════════
#  物体分类与放置决策
# ═══════════════════════════════════════════════════════════════════════════

SHELF_CATEGORY_UNKNOWN = "unknown"
SHELF_CATEGORY_FOOD = "food"
SHELF_CATEGORY_DRINK = "drink"
SHELF_CATEGORY_CLEANING = "cleaning_stuff"
SHELF_CATEGORY_CUTLERY = "cutlery"

OBJECT_CATEGORY_MAP = {
    "biscuit": SHELF_CATEGORY_FOOD,
    "chip": SHELF_CATEGORY_FOOD,
    "lays": SHELF_CATEGORY_FOOD,
    "bread": SHELF_CATEGORY_FOOD,
    "cookie": SHELF_CATEGORY_FOOD,
    "cereal": SHELF_CATEGORY_FOOD,
    "handwash": SHELF_CATEGORY_CLEANING,
    "dishsoap": SHELF_CATEGORY_CLEANING,
    "water": SHELF_CATEGORY_DRINK,
    "sprite": SHELF_CATEGORY_DRINK,
    "cola": SHELF_CATEGORY_DRINK,
    "orangejuice": SHELF_CATEGORY_DRINK,
    "shampoo": SHELF_CATEGORY_CLEANING,
    "milk": SHELF_CATEGORY_DRINK,
    "bowl": SHELF_CATEGORY_CUTLERY,
    "plate": SHELF_CATEGORY_CUTLERY,
    "spoon": SHELF_CATEGORY_CUTLERY,
    "fork": SHELF_CATEGORY_CUTLERY,
    "bag": SHELF_CATEGORY_CLEANING,
    "tablet": SHELF_CATEGORY_CLEANING,
    "cup": SHELF_CATEGORY_CUTLERY,
}

CATEGORY_SPEECH = {
    SHELF_CATEGORY_FOOD: "food",
    SHELF_CATEGORY_DRINK: "drink",
    SHELF_CATEGORY_CLEANING: "cleaning stuff",
    SHELF_CATEGORY_CUTLERY: "cutlery",
    SHELF_CATEGORY_UNKNOWN: "unknown",
}

IGNORED_LABELS = {
    "person",
    "chair",
    "dining_table",
    "table",
    "sink",
    "refrigerator",
    "oven",
    "microwave",
    "dishwasher",
    "cabinet",
}

LABEL_ALIASES = {
    "chips": "chip",
    "lays_chip": "lays",
    "lays_chips": "lays",
    "orange_juice": "orangejuice",
    "hand_wash": "handwash",
    "dish_soap": "dishsoap",
    "dishwashing_tablet": "tablet",
    "dishwasher_tablet": "tablet",
    "wine_glass": "glass",
    "water_glass": "glass",
    "coffee_cup": "cup",
    "tea_cup": "cup",
    "dinner_plate": "plate",
    "dish": "plate",
    "cerealbox": "cereal",
    "cereal_box": "cereal",
    "corn_flakes": "cereal",
    "oatmeal_box": "oatmeal",
    "milk_box": "milk",
    "milk_carton": "milk",
    "trashcan": "trash_can",
    "garbage_bag": "trash_bag",
}
