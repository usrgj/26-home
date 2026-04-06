"""折叠一件衣物

评分:
  - 折叠一件衣物 (+400，第一件)
  - 额外多折叠一件衣物 (+300，最多 6 次)
  - 整齐堆叠折好的衣物 (+200，最多 6 次)

折叠度评估: 整齐度、布料是否被压平和叠放。
使用 ACT 模型 (task3/arm_folding/act/) 进行模仿学习折叠。
"""

from common.state_machine import State
from task3 import config


class FoldOne(State):

    def execute(self, ctx) -> str:
        # TODO: 执行折叠动作
        # 使用 ACT 模型推理，或预设的机械臂折叠轨迹
        # from task3.arm_folding.tool.inference import fold_one_cloth
        # fold_one_cloth()

        ctx.clothes_folded += 1

        # TODO: 整齐堆叠（奖励 +200）
        # arm.stack_neatly()

        return "decide_next"
