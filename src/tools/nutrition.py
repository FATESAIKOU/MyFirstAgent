"""
營養計算工具

這是 Agent 可以調用的 Tool

Tool 的定義方式：
1. 使用 @tool 裝飾器
2. 函數的 docstring 會成為 Tool 的描述（LLM 用來決定是否調用）
3. 參數的型別註解會成為 Tool 的 Schema
"""

from langchain_core.tools import tool


# 簡化的食材營養資料庫（每 100g）
NUTRITION_DB = {
    # 肉類
    "雞胸肉": {"calories": 165, "protein": 31, "fat": 3.6, "carbs": 0},
    "雞腿肉": {"calories": 209, "protein": 26, "fat": 11, "carbs": 0},
    "豬里肌": {"calories": 143, "protein": 21, "fat": 6, "carbs": 0},
    "牛肉": {"calories": 250, "protein": 26, "fat": 15, "carbs": 0},
    "鮭魚": {"calories": 208, "protein": 20, "fat": 13, "carbs": 0},
    
    # 蔬菜
    "花椰菜": {"calories": 25, "protein": 3, "fat": 0.3, "carbs": 5},
    "青菜": {"calories": 20, "protein": 2, "fat": 0.3, "carbs": 3},
    "紅蘿蔔": {"calories": 41, "protein": 0.9, "fat": 0.2, "carbs": 10},
    "番茄": {"calories": 18, "protein": 0.9, "fat": 0.2, "carbs": 4},
    "洋蔥": {"calories": 40, "protein": 1.1, "fat": 0.1, "carbs": 9},
    
    # 主食
    "白飯": {"calories": 130, "protein": 2.7, "fat": 0.3, "carbs": 28},
    "糙米飯": {"calories": 111, "protein": 2.6, "fat": 0.9, "carbs": 23},
    "麵條": {"calories": 138, "protein": 5, "fat": 0.5, "carbs": 28},
    "地瓜": {"calories": 86, "protein": 1.6, "fat": 0.1, "carbs": 20},
    
    # 蛋奶
    "雞蛋": {"calories": 155, "protein": 13, "fat": 11, "carbs": 1.1},
    "牛奶": {"calories": 42, "protein": 3.4, "fat": 1, "carbs": 5},
    
    # 豆類
    "豆腐": {"calories": 76, "protein": 8, "fat": 4.8, "carbs": 1.9},
    "毛豆": {"calories": 122, "protein": 11, "fat": 5, "carbs": 9},
}


@tool
def calculate_nutrition(food_name: str, weight_grams: float) -> str:
    """計算食材的營養成分和熱量
    
    根據食材名稱和重量，計算熱量、蛋白質、脂肪和碳水化合物。
    
    Args:
        food_name: 食材名稱（如：雞胸肉、花椰菜、白飯）
        weight_grams: 重量（克）
    
    Returns:
        營養成分的詳細資訊
    """
    # 嘗試在資料庫中找到匹配的食材
    nutrition = None
    matched_name = None
    
    for name, data in NUTRITION_DB.items():
        if name in food_name or food_name in name:
            nutrition = data
            matched_name = name
            break
    
    if nutrition is None:
        return f"抱歉，找不到「{food_name}」的營養資料。可用的食材：{', '.join(NUTRITION_DB.keys())}"
    
    # 計算實際營養（根據重量比例）
    ratio = weight_grams / 100
    
    result = f"""【{matched_name}】{weight_grams}g 的營養成分：
- 熱量：{nutrition['calories'] * ratio:.1f} 大卡
- 蛋白質：{nutrition['protein'] * ratio:.1f} g
- 脂肪：{nutrition['fat'] * ratio:.1f} g
- 碳水化合物：{nutrition['carbs'] * ratio:.1f} g"""
    
    return result


@tool
def list_available_foods() -> str:
    """列出所有可查詢營養的食材清單
    
    Returns:
        可用食材的分類列表
    """
    categories = {
        "肉類": ["雞胸肉", "雞腿肉", "豬里肌", "牛肉", "鮭魚"],
        "蔬菜": ["花椰菜", "青菜", "紅蘿蔔", "番茄", "洋蔥"],
        "主食": ["白飯", "糙米飯", "麵條", "地瓜"],
        "蛋奶": ["雞蛋", "牛奶"],
        "豆類": ["豆腐", "毛豆"],
    }
    
    result = "【可查詢營養的食材】\n"
    for category, foods in categories.items():
        result += f"\n{category}：{', '.join(foods)}"
    
    return result


# 匯出所有 Tools
nutrition_tools = [calculate_nutrition, list_available_foods]
