"""
状态4：跟随主人前往放包区域 + 放置包

评分项:
  - 跟随主持人前往放包区域 (+200)
  - 将包放置在指定区域 (+50)

主人已经在等候，指示机器人跟随，移动到随机位置后告知放包。
"""

from common.state_machine import State


class FollowAndPlace(State):

    def execute(self, ctx) -> str:
        # ── 跟随主人 (+200) ──

        # 1. 导航到主人处
        # navigation.go_to("host_position")
        # navigation.wait_until_arrived()

        # 2. 告知准备就绪
        # speech.say("I have the bag. Please lead the way.")

        # 3. 启动跟随（使用 common/skills/follow）
        # follow.start()
        # 等待主人停下并发出语音指令 ("put it here" / "place it here")
        # follow.stop()

        # ── 放置包 (+50) ──

        # 4. 机械臂到放包高度
        # arm.go_place_bag()
        # arm.wait_slide_done()

        # 5. 夹爪张开释放
        # arm.release()

        # 6. 机械臂归位
        # arm.go_home()

        # speech.say("The bag has been placed.")
        return "finished"
