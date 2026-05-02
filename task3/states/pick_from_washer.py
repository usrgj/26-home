"""从洗衣机中取出衣物

奖励: 从洗衣机中拿出一件衣物 (+100，最多 4 次)
"""

from common.state_machine import State
from task3 import config


class PickFromWasher(State):

    def execute(self, ctx) -> str:
        """从已打开的洗衣机中取出一件衣物。"""
        if not ctx.washer_door_opened:
            '''门没开就去开'''
            return "nav_to_washer"

        if ctx.washer_remaining <= 0:
            """没有衣物了就返回"""
            return "decide_next"

        if ctx.washer_pick_failures >= config.MAX_PICK_RETRIES:
            """尝试次数过多就结束"""
            return "release"

        # TODO: 机械臂伸入洗衣机内部取出一件衣物
        # arm.pick_cloth_from_washer()

        ctx.washer_remaining -= 1
        ctx.cloth_in_hand = True
        return "transport_to_table"
