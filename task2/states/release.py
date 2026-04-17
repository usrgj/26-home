"""
状态5：释放硬件资源
"""

class Release(State):

    def execute(self, ctx) -> str:
        
        return "finished"