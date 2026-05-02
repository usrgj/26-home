"""task2 异常恢复：跳过失败阶段并尽量继续播报流程。"""

from __future__ import annotations

import logging

from common.state_machine import State

log = logging.getLogger("task2.error_recovery")

_STATE_ORDER = ["init", "kitchen_task", "release", "finished"]


class ErrorRecovery(State):
    """根据失败状态选择可继续的下一个状态。"""

    def execute(self, ctx) -> str:
        """init 或主流程失败时直接释放，避免反复触发硬件动作。"""
        failed = ctx.failed_state
        log.warning("从状态 [%s] 恢复", failed)

        if failed in ("init", "kitchen_task"):
            return "release"

        if failed in _STATE_ORDER:
            index = _STATE_ORDER.index(failed)
            if index + 1 < len(_STATE_ORDER):
                return _STATE_ORDER[index + 1]

        return "release"
