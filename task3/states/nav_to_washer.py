"""导航到洗衣机

奖励: 打开洗衣机门 (+300)

每次从桌子返回继续取衣时都需要先导航到洗衣机前。
洗衣机门只在第一次到达且尚未打开时执行开门动作。
"""

from common.state_machine import State
from task3 import config


class NavToWasher(State):

    def execute(self, ctx) -> str:
        """导航到洗衣机，并在需要时打开门。"""
        if ctx.washer_remaining <= 0:
            '''如果没有衣服了，释放'''
            return "release"

        if ctx.nav_to_washer_failures >= config.MAX_NAV_TO_WASHER_RETRIES:
            '''导航失败次数过多，释放'''
            return "release"

        from common.skills.agv_api import agv, wait_nav

        nav_result = agv.navigate_to(
            agv.get_current_station() or "",
            config.STATION_WASHER,
        )
        if nav_result is None or not wait_nav(config.NAV_TIMEOUT):
            raise RuntimeError("导航到洗衣机失败")

        if ctx.washer_door_opened:
            return "pick_from_washer"

        # TODO: 机械臂操作打开洗衣机门
        # arm.open_washer_door()

        ctx.washer_door_opened = True
        return "pick_from_washer"
