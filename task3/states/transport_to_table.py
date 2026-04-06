"""将衣物运送到折叠台上

评分: 将衣物放在桌子上 (+100)

禁止在地上折叠，必须放到桌上。
"""

from common.state_machine import State
from task3 import config


class TransportToTable(State):

    def execute(self, ctx) -> str:
        # TODO: 导航到折叠台
        # navigation.go_to(config.STATION_TABLE)
        # navigation.wait_until_arrived()

        # TODO: 将衣物放在桌上
        # arm.place_on_table()

        ctx.clothes_on_table += 1
        return "fold_one"
