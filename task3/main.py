"""
task3 洗衣服 —— 主入口

状态流:
状态0：初始化硬件软件，等待按下enter开始任务
状态1：导航到洗衣机，若洗衣机门未打开则打开洗衣机
状态2：从洗衣机取出衣服
状态3：导航到桌子，把衣服放到桌面
状态4：判断是否继续从洗衣机取衣，或释放资源结束
状态5：finish
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
    最后以 "init" 为初始状态启动状态机运行流程。
    """
    ctx = TaskContext()

    # 初始化状态机并注册所有可用状态
    sm = StateMachine(timeout=config.MATCH_TIMEOUT)
    for name, state in ALL_STATES.items():
        sm.add(name, state)

    # 启动状态机，从初始化状态开始执行
    sm.run(ctx, initial="init")


if __name__ == "__main__":
    main()
