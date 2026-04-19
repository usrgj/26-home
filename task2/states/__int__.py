
# robot_kitchen_system/__init__.py
from task2.states.navigation_1 import Navigation
from task2.states.object_recognition_2 import ObjectRecognition
from task2.states.shelf_perception_3 import ShelfPerception
from task2.states.grasping_transport_4 import GraspingTransport
from task2.states.placement_handling_5 import PlacementHandling
from task2.states.release import Release
from task2.states.finished import Finished
from task2.states.error_recovery import ErrorRecovery




# 定义所有厨房任务模块的映射，键为简洁标识，值为模块实例
ALL_KITCHEN_MODULES = {
    "navigation":        Navigation(),          # 1. 导航至餐桌
    "object_recognition": ObjectRecognition(),  # 2. 正确识别物体
    "shelf_perception":  ShelfPerception(),     # 3. 货架物品感知 + 指出放置位置
    "grasping_transport": GraspingTransport(),  # 4. 搬运物品（地板拾取/餐具/盘子/洗碗机栏）
    "release":          Release(),
    "finished":         Finished(),
    "error_recovery":   ErrorRecovery(),

}









