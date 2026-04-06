"""从洗衣篮中拾取衣物

评分: 拾取一件衣物 (+100)
奖励: 使用洗衣篮运送衣物 (+300) —— 双臂搬篮子到桌子旁

策略:
  - 简单路径: 逐件从篮子里取出 → 放桌上 → 折叠
  - 奖励路径: 双臂抱篮子 → 搬到桌旁 → 逐件取出折叠
"""

from common.state_machine import State


class PickFromBasket(State):

    def execute(self, ctx) -> str:
        if ctx.basket_remaining <= 0:
            # 篮子空了，尝试从洗衣机取（奖励项）
            return "open_washer"

        # === 奖励路径：双臂搬篮子 ===
        # if not ctx.using_basket:
        #     speech.say("I'll carry the basket to the table.")
        #     # TODO: 双臂抓取篮子
        #     # arm.grip_basket()
        #     ctx.using_basket = True
        #     return "transport_to_table"

        # === 简单路径：逐件取 ===
        # TODO: 导航到篮子前
        # navigation.go_to(config.STATION_BASKET)

        # TODO: 机械臂拾取一件衣物
        # arm.pick_cloth_from_basket()

        ctx.basket_remaining -= 1
        return "transport_to_table"
