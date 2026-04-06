"""task1 人机交互 —— 状态间共享数据容器"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from task1 import config


@dataclass
class GuestInfo:
    name: str = ""
    favorite_drink: str = ""
    face_encoding: Optional[np.ndarray] = None
    seat_id: str = ""
    visual_features: dict = field(default_factory=dict)  # 衣服颜色等


class TaskContext:
    def __init__(self):
        # 客人数据
        self.guests: list[GuestInfo] = [GuestInfo(), GuestInfo()]
        self.current_guest_index: int = 0  # 0=客人A, 1=客人B

        # 座位状态
        self.seats = list(config.SEATS)
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
