import os
import sys

# 获取当前脚本的绝对路径 (26-home/task3/arm_folding/tool/slide_locate.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 向上跳三级回到真正的根目录 (26-home)
# tool -> arm_folding -> task3 -> 26-home
root_dir = os.path.abspath(os.path.join(current_dir, "../../.."))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
    print(f"已修正项目根目录为: {root_dir}")


from common.skills.agv_api import agv

agv.start()
print(agv.get_map_data("4-11"))
agv.stop()