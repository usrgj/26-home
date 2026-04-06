"""初始状态：场外等待，门打开后出发"""

from common.state_machine import State


class Idle(State):

    def execute(self, ctx) -> str:
        # TODO: 等待开始信号（门打开 / 裁判指令）
        return "nav_to_laundry"
