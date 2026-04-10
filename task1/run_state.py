"""
task1 单状态测试脚本

目标：
- 保留 init 初始化
- 只运行一个指定状态做测试
- 保留 error_recovery 异常处理入口
- 始终经过 release -> finished 做资源释放

默认流程：
    init -> <target_state> -> release -> finished

异常流程：
    init/<target_state> 抛异常 -> error_recovery -> release -> finished

用法示例：
    python3 -m task1.run_state receive_guest
    python3 -m task1.run_state receive_bag --context ./task1/debug_ctx.json
"""

from __future__ import annotations
import os
import sys

# 获取当前脚本的绝对路径 (26-home/task3/arm_folding/tool/slide_locate.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 向上跳三级回到真正的根目录 (26-home)
# tool -> arm_folding -> task3 -> 26-home
root_dir = os.path.abspath(os.path.join(current_dir, "../../.."))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
    print(f"已修正项目根目录为: {root_dir}")
    
import argparse
import json
import logging
from pathlib import Path

from common.state_machine import State, StateMachine
from task1 import config
from task1.context import TaskContext
from task1.states import ALL_STATES


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("task1.run_state")

_INTERNAL_STATES = {"init", "release", "finished", "error_recovery"}
TESTABLE_STATES = tuple(name for name in ALL_STATES if name not in _INTERNAL_STATES)


class RedirectState(State):
    """
    运行真实状态，但忽略它原始返回值，强制跳到指定状态。

    用于单状态测试场景：
    - init 执行完后强制进入目标状态
    - 目标状态执行完后强制进入 release
    - error_recovery 执行完后强制进入 release
    """

    def __init__(self, wrapped: State, forced_next: str, label: str):
        self.wrapped = wrapped
        self.forced_next = forced_next
        self.label = label

    def on_enter(self, ctx):
        self.wrapped.on_enter(ctx)

    def execute(self, ctx) -> str:
        original_next = self.wrapped.execute(ctx)
        log.info(
            "[%s] 原始返回=%s，测试脚本重定向到 %s",
            self.label,
            original_next,
            self.forced_next,
        )
        return self.forced_next

    def on_exit(self, ctx):
        self.wrapped.on_exit(ctx)


def _load_context_overrides(ctx: TaskContext, path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))

    if "current_guest_index" in data:
        ctx.current_guest_index = int(data["current_guest_index"])
    if "bag_received" in data:
        ctx.bag_received = bool(data["bag_received"])
    if "failed_state" in data:
        ctx.failed_state = str(data["failed_state"])

    guests = data.get("guests", [])
    for i, guest_data in enumerate(guests[: len(ctx.guests)]):
        guest = ctx.guests[i]
        if "name" in guest_data:
            guest.name = str(guest_data["name"])
        if "favorite_drink" in guest_data:
            guest.favorite_drink = str(guest_data["favorite_drink"])
        if "person_id" in guest_data:
            guest.person_id = int(guest_data["person_id"])
        if "seat_id" in guest_data:
            guest.seat_id = str(guest_data["seat_id"])
        if "visual_features" in guest_data:
            guest.visual_features = dict(guest_data["visual_features"])

    seats = data.get("seats")
    if seats is not None:
        for seat, override in zip(ctx.seats, seats):
            seat.update(override)


def _build_state_machine(target_state: str, timeout: float) -> StateMachine:
    sm = StateMachine(timeout=timeout)

    sm.add(
        "init",
        RedirectState(
            ALL_STATES["init"],
            forced_next=target_state,
            label="init",
        ),
    )
    sm.add(
        target_state,
        RedirectState(
            ALL_STATES[target_state],
            forced_next="release",
            label=target_state,
        ),
    )
    sm.add(
        "error_recovery",
        RedirectState(
            ALL_STATES["error_recovery"],
            forced_next="release",
            label="error_recovery",
        ),
    )
    sm.add("release", ALL_STATES["release"])
    sm.add("finished", ALL_STATES["finished"])

    return sm


def main():
    parser = argparse.ArgumentParser(description="task1 单状态测试脚本")
    parser.add_argument(
        "state",
        choices=TESTABLE_STATES,
        help="要测试的目标状态名",
    )
    parser.add_argument(
        "--context",
        type=Path,
        help="可选：从 JSON 文件加载 TaskContext 覆盖值",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=config.MATCH_TIMEOUT,
        help="状态机总超时时间（秒）",
    )
    args = parser.parse_args()

    ctx = TaskContext()
    if args.context:
        _load_context_overrides(ctx, args.context)
        log.info("已加载上下文覆盖: %s", args.context)
    elif args.state != "receive_guest":
        log.warning(
            "状态 [%s] 可能依赖前置上下文；如需模拟前置结果，请传入 --context JSON",
            args.state,
        )

    sm = _build_state_machine(args.state, timeout=args.timeout)
    log.info(
        "开始单状态测试: init -> %s -> release -> finished",
        args.state,
    )
    sm.run(ctx, initial="init")


if __name__ == "__main__":
    main()
