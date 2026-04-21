"""
状态4：跟随主人前往放包区域 + 放置包

评分项:
  - 跟随主持人前往放包区域 (+200)
  - 将包放置在指定区域 (+50)

主人已经在等候，指示机器人跟随，移动到随机位置后告知放包。
"""

import logging
import time

from common.state_machine import State
# from common.skills.arm import left_arm, right_arm, left_gripper
from common.skills.arm import left_arm,  left_gripper
from task1 import config
from task1.behaviors.follow import FollowRunner


log = logging.getLogger("task1.follow_and_place")


def _safe_speak(text: str) -> None:
    """尽量播报提示，失败时不打断主流程。"""
    try:
        from common.skills.audio_module.voice_assiant import voice_assistant

        voice_assistant.speak(text)
    except Exception as exc:
        log.warning("语音播报失败: %s", exc)


def _heard_place_command(ctx) -> bool:
    """占位钩子：后续在这里接入“put it here / place it here”语音判定。"""
    return False


def _should_stop_follow(ctx, result, start_time: float) -> bool:
    """由外层任务状态机决定何时结束跟随。"""
    if time.time() - start_time >= config.FOLLOW_HOST_TIMEOUT:
        log.info("跟随主人超时 %.1fs，结束跟随", config.FOLLOW_HOST_TIMEOUT)
        return True

    if _heard_place_command(ctx):
        log.info("检测到放包指令，结束跟随")
        return True

    return False

class FollowAndPlace(State):

    def execute(self, ctx) -> str:
        # ── 跟随主人 (+200) ──
        runner = FollowRunner()
        follow_started_at = time.time()

        _safe_speak("")

        try:
            target_locked = runner.start()
            if not target_locked:
                log.info("跟随启动时未锁定主人，将在运行中尝试自动锁定")

            while True:
                result = runner.step()
                if _should_stop_follow(ctx, result, follow_started_at):
                    break
        finally:
            runner.stop()

        # ── 放置包 (+50) ──

        # 4. 机械臂到放包高度
        left_arm.rm_movej([-99.705,-89.426,-9.176,-87.258,79.38,-246.351], 20, 0, 0, 1)
        left_gripper.open()
        time.sleep(3)

        # 5. 夹爪张开释放
        # arm.release()

        # 6. 机械臂归位
        # arm.go_home()
        left_gripper.grab(force=500)
        time.sleep(3)
        left_arm.rm_movej([-112.305,-115.771,-67.401,2.997,7.318,-263.746], 20, 0, 0, 1)
        print(left_gripper.state)

        # speech.say("The bag has been placed.")
        return "release"
