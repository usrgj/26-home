"""决策节点：判断是否继续从洗衣机取衣

优先级:
  1. 手里还有衣物 → transport_to_table
  2. 洗衣机还有衣物 → nav_to_washer，先回到洗衣机前再取衣
  3. 全部完成或失败过多 → release
"""

from common.state_machine import State
from task3 import config


class DecideNext(State):

    def execute(self, ctx) -> str:
        """根据上下文计数选择下一状态。"""
        if ctx.cloth_in_hand:
            return "transport_to_table"

        if ctx.washer_remaining > 0:
            if ctx.washer_pick_failures >= config.MAX_PICK_RETRIES:
                return "release"
            return "nav_to_washer"

        return "release"
