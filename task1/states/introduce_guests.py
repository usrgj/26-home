"""
状态2：相互介绍两位客人

评分项:
  - 准确说出每位客人的姓名及喜爱饮品 (4×30 = +120)
  - 提及一位客人时需注视另一位对应客人 (2×50 = +100)

流程: 看向A → 说B的信息，看向B → 说A的信息
"""

from common.state_machine import State


class IntroduceGuests(State):

    def execute(self, ctx) -> str:
        a = ctx.guests[0]
        b = ctx.guests[1]

        # 1. 看向客人A，介绍客人B
        # TODO: head.look_at(seat_a 方向)
        # speech.say(f"{a.name}, this is {b.name}. "
        #            f"Their favorite drink is {b.favorite_drink}.")

        # 2. 看向客人B，介绍客人A
        # TODO: head.look_at(seat_b 方向)
        # speech.say(f"{b.name}, this is {a.name}. "
        #            f"Their favorite drink is {a.favorite_drink}.")

        return "receive_bag"
