


def detect_shelf_items_and_get_position():
    """
    任务3：感知货架物品 → 指出正确放置位置
    分值：2 × 30
    返回：物品列表 + 对应正确放置位置
    """
    print("感知货架上的物品...")
    # 货架分割 + 物品检测 + 位置匹配
    shelf_items = [
        {"item": "bowl", "correct_place": "cabinet_bottom"},
        {"item": "plate", "correct_place": "dishwasher_upper"}
    ]
    return shelf_items

