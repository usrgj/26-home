"""导航到洗衣区

评分: 导航前往洗衣区 (+15)
"""

from common.state_machine import State
from task3 import config


class NavToLaundry(State):

    def execute(self, ctx) -> str:
        # TODO: navigation.go_to(config.STATION_LAUNDRY)
        # TODO: navigation.wait_until_arrived(timeout=config.NAV_TIMEOUT)

        # 优先从篮子取衣物（主线任务），洗衣机为附加奖励
        return "pick_from_basket"
