"""决策节点：判断下一步取衣物还是结束

优先级:
  1. 篮子还有衣物 → pick_from_basket
  2. 洗衣机还有衣物 → open_washer (若未开门) / pick_from_washer
  3. 桌上还有未折叠的衣物 → fold_one
  4. 全部完成 → finished
"""

from common.state_machine import State


class DecideNext(State):

    def execute(self, ctx) -> str:
        # 还有未取的衣物
        if ctx.basket_remaining > 0:
            return "pick_from_basket"

        if ctx.washer_remaining > 0:
            if not ctx.washer_door_opened:
                return "open_washer"
            return "pick_from_washer"

        # 桌上有未折叠的
        if ctx.clothes_on_table > ctx.clothes_folded:
            return "fold_one"

        return "finished"
