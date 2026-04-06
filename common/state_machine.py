"""
有限状态机引擎

用法:
    from common.state_machine import State, StateMachine

    class Idle(State):
        def execute(self, ctx):
            return "next_state"

    sm = StateMachine(timeout=360)
    sm.add("idle", Idle())
    sm.run(ctx, initial="idle")
"""

from __future__ import annotations
import time
import logging

log = logging.getLogger("StateMachine")


class State:
    """状态基类，每个状态继承此类并实现 execute()"""

    name: str = ""

    def on_enter(self, ctx):
        """进入状态时调用（可选覆写）"""
        pass

    def execute(self, ctx) -> str:
        """
        执行状态逻辑，返回下一个状态名称（字符串）。
        返回自身名称 = 重试，返回 "finished" = 结束。
        """
        raise NotImplementedError

    def on_exit(self, ctx):
        """离开状态时调用（可选覆写）"""
        pass


class StateMachine:
    """
    状态机引擎

    - while 循环驱动状态转移
    - 内置总时限超时保护
    - try-except 异常捕获 → error_recovery
    """

    def __init__(self, timeout: float = 360.0):
        """
        :param timeout: 比赛总时限（秒），默认 6 分钟
        """
        self.states: dict[str, State] = {}
        self.timeout = timeout

    def add(self, name: str, state: State):
        state.name = name
        self.states[name] = state

    def run(self, ctx, initial: str = "idle"):
        current = initial
        start = time.time()

        log.info("状态机启动, initial=%s, timeout=%.0fs", current, self.timeout)

        while current != "finished":
            elapsed = time.time() - start
            if elapsed > self.timeout:
                log.warning("比赛超时 (%.0fs), 强制结束", elapsed)
                break

            if current not in self.states:
                log.error("未知状态: %s, 终止", current)
                break

            state = self.states[current]
            log.info(">>> 进入状态: %s  (已用 %.1fs)", current, elapsed)

            try:
                state.on_enter(ctx)
                next_state = state.execute(ctx)
                state.on_exit(ctx)
                log.info("<<< 状态 %s → %s", current, next_state)
                current = next_state

            except Exception as e:
                log.error("状态 %s 异常: %s", current, e, exc_info=True)
                try:
                    state.on_exit(ctx)
                except Exception:
                    pass

                if "error_recovery" in self.states:
                    ctx.failed_state = current
                    current = "error_recovery"
                else:
                    log.error("无 error_recovery 状态, 终止")
                    break

        log.info("状态机结束, 总耗时 %.1fs", time.time() - start)
