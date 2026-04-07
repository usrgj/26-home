"""异常恢复：跳到失败状态的下一个继续"""

import logging
from common.state_machine import State

log = logging.getLogger("task1.error_recovery")

_STATE_ORDER = [
    "init", "receive_guest", "introduce",
    "receive_bag", "follow_and_place", "release", "finished",
]


class ErrorRecovery(State):

    def execute(self, ctx) -> str:
        failed = ctx.failed_state
        log.warning("从状态 [%s] 恢复", failed)

        if failed in _STATE_ORDER:
            idx = _STATE_ORDER.index(failed)
            if idx + 1 < len(_STATE_ORDER):
                next_state = _STATE_ORDER[idx + 1]
                log.info("跳过 [%s] → [%s]", failed, next_state)
                return next_state

        return "release"
