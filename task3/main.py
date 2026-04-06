"""
task3 洗衣服 —— 主入口

状态流:
状态0：初始化硬件软件，等待按下enter开始任务
状态1：导航到洗衣区，识别并拿起篮子
状态2：导航到洗衣机，放下篮子，打开洗衣机，取出衣服
状态3：拿起篮子，导航到桌子，放下篮子
状态4：观察篮子中是否有衣服，若无则结束，若有则取出衣服
状态5：折衣服
状态6：finish
"""

from __future__ import annotations
import logging

from common.state_machine import StateMachine
from task3 import config
from task3.context import TaskContext
from task3.states import ALL_STATES

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s %(name)s] %(message)s",
                    datefmt="%H:%M:%S")


def main():
    """
    主函数，负责初始化任务上下文和状态机，并启动状态流转。

    该函数创建任务上下文对象，配置状态机的超时时间，注册所有预定义的状态，
    最后以 "idle" 为初始状态启动状态机运行流程。
    """
    ctx = TaskContext()

    # 初始化状态机并注册所有可用状态
    sm = StateMachine(timeout=config.MATCH_TIMEOUT)
    for name, state in ALL_STATES.items():
        sm.add(name, state)

    # 启动状态机，从空闲状态开始执行
    sm.run(ctx, initial="idle")


if __name__ == "__main__":
    main()
