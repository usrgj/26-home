"""异常恢复：根据失败位置决定继续策略"""

import logging
from common.state_machine import State

log = logging.getLogger("task3.error_recovery")

_STATE_ORDER = [
    "idle", "nav_to_laundry", "pick_from_basket", "open_washer",
    "pick_from_washer", "transport_to_table", "fold_one",
    "decide_next", "finished",
]


class ErrorRecovery(State):

    def execute(self, ctx) -> str:
        failed = ctx.failed_state
        log.warning("从状态 [%s] 恢复", failed)

        # 折叠/取衣失败 → 跳到决策节点继续下一件
        if failed in ("fold_one", "pick_from_basket", "pick_from_washer"):
            return "decide_next"

        # 其他状态 → 跳到下一个
        if failed in _STATE_ORDER:
            idx = _STATE_ORDER.index(failed)
            if idx + 1 < len(_STATE_ORDER):
                return _STATE_ORDER[idx + 1]

        return "finished"
