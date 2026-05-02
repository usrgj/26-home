"""task2 结束状态。"""

from __future__ import annotations

from common.state_machine import State


class Finished(State):
    """状态机最终占位状态。"""

    def execute(self, ctx) -> str:
        """保持与通用状态机约定一致，返回 finished 结束。"""
        return "finished"
