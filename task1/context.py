"""task1 人机交互 —— 状态间共享数据容器"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from task1 import config


seat_priority_by_host = {
                "seat_1": ("seat_2", "seat_3", "seat_4", "seat_5"),
                "seat_2": ("seat_1", "seat_3", "seat_4", "seat_5"),
                "seat_3": ("seat_4", "seat_5", "seat_2", "seat_1"),
                "seat_4": ("seat_3", "seat_5", "seat_2", "seat_1"),
                "seat_5": ("seat_4", "seat_3", "seat_2", "seat_1"),
            }

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
        self.host_seat_id: str = config.HOST_SEATS
        

        # 座位状态
        self.seats = [dict(s) for s in config.SEATS]
        # for idx in config.PRE_OCCUPIED_SEATS:
        #     self.seats[idx]["occupied"] = True
        self.occupy_seat(self.host_seat_id)
        
        # 主人的信息
        self.host_angle = 0.0
        self.host_nav = ""
        for s in config.SEATS_MAPPING:
            if s["seat_id"] == self.host_seat_id :
                self.host_angle = s["angle"]
                self.host_nav = s["nav_id"]
                break

        # 包相关
        self.bag_received: bool = False

        # 错误恢复
        self.failed_state: str = ""

    @property
    def current_guest(self) -> GuestInfo:
        return self.guests[self.current_guest_index]

    def find_free_seat(self) -> Optional[str]:
        '''
        根据现在的客人挑选有我们期望的座位，比如第一个客人就选择离开始点近的，第二个客人就选择离主人近的
        '''
        
        if self.current_guest_index == 0:
            # 优先选择离开始点近的座位
            for s in self.seats:
                if s["occupied"] is False :
                    return s["id"]
                
        if self.current_guest_index == 1:
            # 优先选择离主人近的座位  
            '''
            主人在seat_1,就按2、3、4、5的顺序寻找空闲座位
            在seat_2,就按1、3、4、5的顺序寻找空闲座位
            在seat_3,就按4、5、2、1的顺序寻找空闲座位
            在seat_4,就按3、5、2、1的顺序寻找空闲座位
            在seat_5,就按4、3、2、1的顺序寻找空闲座位
            '''
            
            free_seats = {s["id"] for s in self.seats if s["occupied"] is False}
            for seat_id in seat_priority_by_host.get(self.host_seat_id, ()):
                if seat_id in free_seats:
                    return seat_id


        
        # """返回第一个空闲座位 ID，无空位返回 None"""
        # for s in self.seats:
        #     if s["occupied"] is False:
        #         return s["id"]
        # return None

    def occupy_seat(self, seat_id: str):
        for s in self.seats:
            if s["id"] == seat_id:
                s["occupied"] = True
                break
    def release_seat(self, seat_id: str):
        for s in self.seats:
            if s["id"] == seat_id:
                s["occupied"] = False
                break
