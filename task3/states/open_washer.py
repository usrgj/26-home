"""打开洗衣机门

奖励: 打开洗衣机门 (+300)

洗衣机初始时门关闭，必须先打开门才能取出衣物。
"""

from common.state_machine import State
from task3 import config


class OpenWasher(State):

    def execute(self, ctx) -> str:
        if ctx.washer_door_opened:
            return "pick_from_washer"

        if ctx.washer_remaining <= 0:
            return "decide_next"

        # TODO: 导航到洗衣机前
        # navigation.go_to(config.STATION_WASHER)

        # TODO: 机械臂操作打开洗衣机门
        # arm.open_washer_door()

        ctx.washer_door_opened = True
        return "pick_from_washer"
