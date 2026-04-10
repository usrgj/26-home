"""
人物跟随子系统的 CLI 入口。

本文件只负责：
- 配置日志
- 处理 Ctrl+C
- 创建 FollowRunner 并循环调用 step()

真正的跟随生命周期接口位于 runner.py 中。
"""
from __future__ import annotations

import signal
import sys
import logging
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from task1.behaviors.follow.config import LOG_LEVEL
    from task1.behaviors.follow.runner import FollowRunner
else:
    from .config import LOG_LEVEL
    from .runner import FollowRunner


logger = logging.getLogger("FollowMain")


def _configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    """独立运行人物跟随系统。"""
    _configure_logging()

    stop_requested = {"value": False}
    previous_handler = signal.getsignal(signal.SIGINT)

    def _signal_handler(sig, frame):
        stop_requested["value"] = True
        print("\n收到退出信号，正在停止...")

    signal.signal(signal.SIGINT, _signal_handler)

    runner = FollowRunner()

    logger.info("=" * 60)
    logger.info("   人物跟随系统 (方案四: 视觉+LiDAR融合+分层控制)")
    logger.info("=" * 60)

    try:
        runner.start()
        while not stop_requested["value"]:
            runner.step()
    finally:
        logger.info("正在停止...")
        runner.close()
        signal.signal(signal.SIGINT, previous_handler)
        logger.info("已退出人物跟随")


if __name__ == "__main__":
    main()
