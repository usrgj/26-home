"""将衣物运送到折叠台上

评分: 将衣物放在桌子上 (+100)

禁止在地上折叠，必须放到桌上。
"""

from common.state_machine import State
from task3 import config


class TransportToTable(State):

    def execute(self, ctx) -> str:
        """把手中的衣物运到桌面，并更新桌面衣物计数。"""
        if not ctx.cloth_in_hand:
            '''手里没衣服就返回'''
            return "decide_next"

        if ctx.transport_failures >= config.MAX_TRANSPORT_RETRIES:
            '''失败太多就释放'''
            return "release"

        from common.skills.agv_api import agv, wait_nav

        nav_result = agv.navigate_to(
            agv.get_current_station() or "",
            config.STATION_TABLE,
        )
        if nav_result is None or not wait_nav(config.NAV_TIMEOUT):
            raise RuntimeError("导航到桌子失败")

        # TODO: 将衣物放在桌上
        # arm.place_on_table()

        ctx.cloth_in_hand = False
        ctx.clothes_on_table += 1
        return "decide_next"
