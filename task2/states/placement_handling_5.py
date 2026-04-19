


def PlacementHandling(item, place_type):
    """
    任务5：将物体放到指定位置
    基础分值：12 × 40
    加分：
        - 正确放入洗碗机（仿人操作）3×+70
        - 橱柜同类相邻放置 2×+20
    参数：物品、目标位置类型
    返回：放置结果 + 加分
    """
    print(f"将 {item} 放置到: {place_type}")
    extra_score = 0

    if place_type == "dishwasher":
        extra_score += 70  # 洗碗机正确放置
    elif place_type == "cabinet_similar":
        extra_score += 20  # 橱柜同类相邻

    return {"success": True, "extra_score": extra_score}

