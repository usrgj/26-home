"""从洗衣机中取出衣物

奖励: 从洗衣机中拿出一件衣物 (+100，最多 4 次)
"""

from common.state_machine import State


class PickFromWasher(State):

    def execute(self, ctx) -> str:
        if ctx.washer_remaining <= 0:
            return "decide_next"

        # TODO: 机械臂伸入洗衣机内部取出一件衣物
        # arm.pick_cloth_from_washer()

        ctx.washer_remaining -= 1
        return "transport_to_table"
