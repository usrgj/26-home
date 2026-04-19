"""
task2   主入口

状态流:
状态0：等待开始，可以先初始化硬件软件，等待按下enter开始任务
状态1：
状态2：相互介绍
状态3：向第二位客人拿包
状态4：跟随主人并放包
状态5：finish
这样分的理由是，有些得分点必须前面的任务完成好了才能继续，比如状态1；而状态3和状态4之间没有依赖，即使没有成功拿到包，也可以拿下跟随的分数。
"""

from __future__ import annotations
import logging

from common.state_machine import StateMachine
from task1 import config
from task1.context import TaskContext
from task1.states import ALL_STATES

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s %(name)s] %(message)s",
                    datefmt="%H:%M:%S")


def main():
    """
    主函数，负责初始化任务上下文并启动状态机。

    该函数创建任务上下文和状态机实例，注册所有预定义的状态，
    并以 "idle" 为初始状态运行状态机。

    Returns:
        None
    """
    ctx = TaskContext()

    # 初始化状态机并注册所有预定义状态
    sm = StateMachine(timeout=config.MATCH_TIMEOUT)
    for name, state in ALL_STATES.items():
        sm.add(name, state)

    sm.run(ctx, initial="init")


if __name__ == "__main__":
    main()
