"""
状态5：比赛结束
"""

from common.state_machine import State


class Finished(State):

    def execute(self, ctx) -> str:
        # speech.say("All tasks completed. Thank you.")
        return "finished"
