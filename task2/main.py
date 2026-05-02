"""
task2  拾取与放置挑战 —— 主入口

状态流:
状态0：初始化底盘、相机和语音，等待按下 Enter 开始任务
状态1：完成餐桌导航、餐桌识别、货架感知、清理计划和早餐播报
状态2：释放资源并结束
"""

from __future__ import annotations
import logging

from common.state_machine import StateMachine
from task2 import config
from task2.context import TaskContext
from task2.states import ALL_STATES

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s %(name)s] %(message)s",
                    datefmt="%H:%M:%S")


def main():
    """
    主函数，负责初始化任务上下文和状态机，并启动状态流转。

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
