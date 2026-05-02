"""异常恢复：根据失败位置决定重试或结束"""

import logging
from common.state_machine import State
from task3 import config

log = logging.getLogger("task3.error_recovery")

_STATE_ORDER = [
    "init", "nav_to_washer", "pick_from_washer",
    "transport_to_table", "decide_next", "release", "finished",
]


class ErrorRecovery(State):

    def execute(self, ctx) -> str:
        """记录失败次数，未超过上限时重试对应状态。"""
        failed = ctx.failed_state
        log.warning("从状态 [%s] 恢复", failed)

        if failed == "init":
            return "release"

        if failed == "nav_to_washer":
            ctx.nav_to_washer_failures += 1
            if ctx.nav_to_washer_failures < config.MAX_NAV_TO_WASHER_RETRIES:
                return "nav_to_washer"
            return "release"

        if failed == "pick_from_washer":
            ctx.washer_pick_failures += 1
            if ctx.washer_pick_failures < config.MAX_PICK_RETRIES:
                return "pick_from_washer"
            return "release"

        if failed == "transport_to_table":
            ctx.transport_failures += 1
            if ctx.transport_failures < config.MAX_TRANSPORT_RETRIES:
                return "transport_to_table"
            return "release"

        if failed in _STATE_ORDER:
            idx = _STATE_ORDER.index(failed)
            if idx + 1 < len(_STATE_ORDER):
                return _STATE_ORDER[idx + 1]

        return "release"
