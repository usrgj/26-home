"""
状态4：跟随主人前往放包区域 + 放置包

评分项:
  - 跟随主持人前往放包区域 (+200)
  - 将包放置在指定区域 (+50)

主人已经在等候，指示机器人跟随，移动到随机位置后告知放包。
"""

import logging
import time
import re
import threading

from common.state_machine import State
# from common.skills.arm import left_arm, right_arm, left_gripper
from common.skills.arm import left_arm,  left_gripper
from task1 import config
from task1.behaviors.follow import FollowRunner


log = logging.getLogger("task1.follow_and_place")



class FollowAndPlace(State):

    def execute(self, ctx) -> str:
        # ── 跟随主人 (+200) ──
        runner = FollowRunner()
        follow_started_at = time.time()
        
        #持续监听放包指令 author: mhl
        heard_event = threading.Event()
        stop_event = threading.Event()
        heard_text = {"text": ""}
        listener_thread = threading.Thread(
        target=_listen_for_place_command,
        args=(heard_event, stop_event, heard_text),
        name="place-command-listener",
        daemon=True,
        )

        _safe_speak("I will follow you. Please say put it here when you want me to place the bag.")
        listener_thread.start()

        try:
            target_locked = runner.start()
            if not target_locked:
                log.info("跟随启动时未锁定主人，将在运行中尝试自动锁定")

            while True:
                result = runner.step()
                if _should_stop_follow(heard_event, follow_started_at):
                    break
        finally:
            runner.stop()
            stop_event.set()
            listener_thread.join(timeout=2.0)

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



def _safe_speak(text: str) -> None:
    """尽量播报提示，失败时不打断主流程。"""
    try:
        from common.skills.audio_module.voice_assiant import voice_assistant

        voice_assistant.speak(text)
    except Exception as exc:
        log.warning("语音播报失败: %s", exc)


PLACE_COMMANDS = (
    "put it here",
    "place it here",
    "drop it here",
    "here"
)
def _match_place_command(text: str) -> bool:
    """判断识别结果里是否包含放包口令。"""
    # 统一小写、去掉标点，方便做口令匹配。
    text = (text or "").lower()
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return any(command in text for command in PLACE_COMMANDS)

def _listen_for_place_command(heard_event: threading.Event,
                              stop_event: threading.Event,
                              heard_text: dict) -> None:
    """后台循环录音并识别，命中口令后置位 heard_event。"""
    try:
        from common.skills.audio_module.voice_assiant import voice_assistant

        voice_assistant.set_recording(1)

        while not stop_event.is_set() and not heard_event.is_set():
            audio_frames = voice_assistant.record_utterance()
            if stop_event.is_set():
                break
            if not audio_frames:
                continue

            text = (voice_assistant.recognize(audio_frames) or "").strip()
            if not text:
                continue

            log.info("跟随阶段识别到语音: %s", text)

            if _match_place_command(text):
                heard_text["text"] = text
                print("检测到放包指令为: %s", text)
                heard_event.set()
                break

    except Exception as exc:
        log.warning("后台监听放包指令失败: %s", exc)
    finally:
        try:
            from common.skills.audio_module.voice_assiant import voice_assistant
            voice_assistant.set_recording(0)
        except Exception:
            pass

def _should_stop_follow(heard_event: threading.Event, start_time: float) -> bool:
    """由外层任务状态机决定何时结束跟随。"""
    if time.time() - start_time >= config.FOLLOW_HOST_TIMEOUT:
        log.info("跟随主人超时 %.1fs，结束跟随", config.FOLLOW_HOST_TIMEOUT)
        return True

    if heard_event.is_set():
        log.info("检测到放包指令，结束跟随")
        return True

    return False