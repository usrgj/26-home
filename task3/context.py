"""task3 洗衣服 —— 状态间共享数据容器"""

from __future__ import annotations
from task3 import config


class TaskContext:
    def __init__(self):
        # 衣物计数
        self.washer_remaining: int = config.WASHER_CLOTHES_COUNT
        self.clothes_on_table: int = 0

        # 洗衣机门状态
        self.washer_door_opened: bool = False

        # 当前是否已经从洗衣机拿到衣物，防止空手运输到桌面
        self.cloth_in_hand: bool = False

        # 错误恢复
        self.failed_state: str = ""
        self.nav_to_washer_failures: int = 0
        self.washer_pick_failures: int = 0
        self.transport_failures: int = 0
