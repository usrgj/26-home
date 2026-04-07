"""
状态6：比赛结束
"""

from common.state_machine import State


class Finished(State):

    def execute(self, ctx) -> str:
        """最终占位状态。release 完成资源释放后进入这里。"""
        return "finished"
