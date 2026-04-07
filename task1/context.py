"""task1 人机交互 —— 状态间共享数据容器"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from task1 import config


@dataclass
class GuestInfo:
    name: str = ""
    favorite_drink: str = ""
    person_id: int = -1                        # RoboCupReIDTracker 中的 person_id
    seat_id: str = ""
    visual_features: dict = field(default_factory=dict)  # LLM 提取的外貌特征


class TaskContext:
    def __init__(self):
        # 客人数据
        self.guests: list[GuestInfo] = [GuestInfo(), GuestInfo()]
        self.current_guest_index: int = 0  # 0=客人A, 1=客人B

        # 座位状态
        self.seats = [dict(s) for s in config.SEATS]
        for idx in config.PRE_OCCUPIED_SEATS:
            self.seats[idx]["occupied"] = True

        # 包相关
        self.bag_received: bool = False

        # 错误恢复
        self.failed_state: str = ""

    @property
    def current_guest(self) -> GuestInfo:
        return self.guests[self.current_guest_index]

    def find_free_seat(self) -> Optional[str]:
        """返回第一个空闲座位 ID，无空位返回 None"""
        for s in self.seats:
            if not s["occupied"]:
                return s["id"]
        return None

    def occupy_seat(self, seat_id: str):
        for s in self.seats:
            if s["id"] == seat_id:
                s["occupied"] = True
                break
