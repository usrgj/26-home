"""比赛结束状态"""

from common.state_machine import State


class Finished(State):

    def execute(self, ctx) -> str:
        return "finished"
