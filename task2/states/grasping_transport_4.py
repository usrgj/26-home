




def transport_item(item_type):
    """
    任务4：搬运物品
    基础分值：12 × 50
    加分：
        - 地板拾取 +30
        - 餐具 2×+50
        - 盘子 +100
        - 洗碗机栏 +100
    参数：物品类型
    返回：搬运结果 + 获得加分
    """
    print(f"开始搬运物品: {item_type}")
    extra_score = 0

    if item_type == "floor_item":
        extra_score += 30  # 地板拾取
    elif item_type == "tableware":
        extra_score += 50  # 餐具
    elif item_type == "plate":
        extra_score += 100  # 盘子
    elif item_type == "dishwasher_rack":
        extra_score += 100  # 洗碗机栏

    return {"success": True, "extra_score": extra_score}
