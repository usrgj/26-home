"""task3 洗衣服 —— 状态间共享数据容器"""

from __future__ import annotations
from task3 import config


class TaskContext:
    def __init__(self):
        # 衣物计数
        self.basket_remaining: int = config.BASKET_CLOTHES_COUNT
        self.washer_remaining: int = config.WASHER_CLOTHES_COUNT
        self.clothes_on_table: int = 0
        self.clothes_folded: int = 0

        # 洗衣机门状态
        self.washer_door_opened: bool = False

        # 使用洗衣篮运输
        self.using_basket: bool = False
        self.basket_loaded: int = 0  # 篮子里已装的衣物数

        # 错误恢复
        self.failed_state: str = ""
