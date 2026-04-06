"""
状态1：等待铃声 → 导航到门 → 开门迎接 → 问好 → 询问姓名饮品
     → 提取人脸/视觉特征 → 引导就坐

本状态内部循环执行两次（两位客人），全部就座后进入状态2。

评分项:
  - 门铃声识别为客人到达信号 (2×30 奖励)
  - 为客人开门 (2×200 奖励)
  - 朝导航方向观察 (2×15)
  - 注视正在交谈的对象 (2×50)
  - 为客人提供空闲座位 (2×100)
  - 向第二位客人描述第一位客人特征 (4×20 奖励)
  - 不提非必要问题确认信息 (4×15 奖励)
"""

import logging
from common.state_machine import State

log = logging.getLogger("task1.receive_guest")


class ReceiveGuest(State):

    def execute(self, ctx) -> str:
        while ctx.current_guest_index < 2:
            guest = ctx.current_guest
            i = ctx.current_guest_index
            log.info("===== 接待客人 #%d =====", i)

            # ── 1. 等待门铃（奖励 +30） ──
            # TODO: doorbell.wait() 或 直接跳过

            # ── 2. 导航到门口（导航时云台朝前 +15） ──
            # head.look_forward()
            # navigation.go_to(config.STATION_DOOR)
            # navigation.wait_until_arrived(timeout=config.NAV_TIMEOUT)

            # ── 3. 开门（奖励 +200） ──
            # TODO: door.open()

            # ── 4. 启动注视跟踪（对话时持续看着客人 +50） ──
            # head.start_gaze_tracking()

            # ── 5. 问好 + 询问姓名 + 询问饮品 ──
            # speech.say("Hello! Welcome.")
            # guest.name = speech.ask("What is your name?")
            # guest.favorite_drink = speech.ask("What is your favorite drink?")

            # ── 6. 记录人脸编码 ──
            # color, depth = camera.get_frames()
            # guest.face_encoding = vision.encode_face(color)

            # ── 7. 采集视觉特征（第一位客人，用于后续描述奖励 4×20） ──
            # if i == 0:
            #     persons = vision.detect_persons(color)
            #     if persons:
            #         guest.visual_features = vision.describe_person(color, persons[0]["bbox"])

            # ── 8. 第二位客人时描述第一位客人外观（奖励 4×20） ──
            # if i == 1:
            #     f = ctx.guests[0].visual_features
            #     speech.say(f"The other guest is wearing a {f.get('upper_color','')} top "
            #                f"and {f.get('lower_color','')} pants.")

            # ── 9. 停止注视跟踪，导航引领客人到座位 ──
            # head.stop_gaze_tracking()
            # head.look_forward()
            seat_id = ctx.find_free_seat()
            # speech.say("Please follow me to your seat.")
            # navigation.go_to(seat_id)
            # navigation.wait_until_arrived()
            # speech.say("Please have a seat here.")

            if seat_id:
                guest.seat_id = seat_id
                ctx.occupy_seat(seat_id)

            ctx.current_guest_index += 1

        return "introduce"
