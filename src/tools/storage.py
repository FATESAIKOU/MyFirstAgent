"""
用戶資料存取工具

提供 JSON 檔案的讀寫功能，用於持久化用戶資料

資料結構：
{
    "name": "用戶名",
    "target_weight": 70.0,        # 目標體重 (kg)
    "current_weight": 75.0,       # 當前體重 (kg)
    "daily_calorie_limit": 1800,  # 每日熱量上限 (kcal)
    "preferences": {
        "diet_type": "一般",      # 一般/素食/純素
        "allergies": [],          # 過敏原
        "favorites": [],          # 喜愛食材
        "dislikes": []            # 不喜歡食材
    },
    "weight_history": [           # 體重記錄
        {"date": "2025-12-26", "weight": 75.0}
    ],
    "meal_records": [             # 飲食記錄
        {"date": "2025-12-26", "meal": "午餐", "foods": [...], "total_calories": 500}
    ],
    "planned_feasts": [           # 預定大餐
        {"date": "2025-12-28", "description": "火鍋", "estimated_calories": 1500}
    ]
}

學習重點：
1. 多個相關 Tools 的組合
2. 錯誤處理和預設值
3. 資料驗證
"""

import json
import os
from datetime import datetime
from langchain_core.tools import tool
from src.config.settings import USER_PROFILE_PATH, DATA_DIR


def _ensure_data_dir():
    """確保資料目錄存在"""
    os.makedirs(DATA_DIR, exist_ok=True)


def _get_default_profile() -> dict:
    """返回預設的用戶資料結構"""
    return {
        "name": None,
        "target_weight": None,
        "current_weight": None,
        "daily_calorie_limit": 1800,
        "preferences": {
            "diet_type": "一般",
            "allergies": [],
            "favorites": [],
            "dislikes": []
        },
        "weight_history": [],
        "meal_records": [],
        "planned_feasts": []
    }


@tool
def load_user_profile() -> str:
    """載入用戶資料
    
    讀取用戶的個人資料，包括：體重目標、當前體重、飲食偏好等。
    如果是新用戶（資料不存在），會返回空資料。
    
    Returns:
        用戶資料的 JSON 格式字串，或「新用戶」提示
    """
    _ensure_data_dir()
    
    if not os.path.exists(USER_PROFILE_PATH):
        return "這是新用戶，尚未建立個人資料。請引導用戶設定：姓名、目標體重、當前體重、飲食偏好。"
    
    try:
        with open(USER_PROFILE_PATH, 'r', encoding='utf-8') as f:
            profile = json.load(f)
        
        # 格式化輸出
        result = f"""【用戶資料】
姓名：{profile.get('name', '未設定')}
目標體重：{profile.get('target_weight', '未設定')} kg
當前體重：{profile.get('current_weight', '未設定')} kg
每日熱量上限：{profile.get('daily_calorie_limit', 1800)} kcal
飲食類型：{profile.get('preferences', {}).get('diet_type', '一般')}
過敏原：{', '.join(profile.get('preferences', {}).get('allergies', [])) or '無'}
喜愛食材：{', '.join(profile.get('preferences', {}).get('favorites', [])) or '未設定'}
"""
        
        # 檢查今日飲食記錄
        today = datetime.now().strftime('%Y-%m-%d')
        today_meals = [m for m in profile.get('meal_records', []) if m.get('date') == today]
        today_calories = sum(m.get('total_calories', 0) for m in today_meals)
        
        result += f"\n今日已攝取：{today_calories} kcal"
        result += f"\n今日剩餘額度：{profile.get('daily_calorie_limit', 1800) - today_calories} kcal"
        
        # 檢查預定大餐
        planned = profile.get('planned_feasts', [])
        upcoming = [p for p in planned if p.get('date', '') >= today]
        if upcoming:
            result += "\n\n【預定大餐】"
            for feast in upcoming:
                result += f"\n- {feast.get('date')}: {feast.get('description')} (約 {feast.get('estimated_calories')} kcal)"
        
        return result
    
    except Exception as e:
        return f"讀取用戶資料時發生錯誤：{str(e)}"


@tool
def save_user_profile(
    name: str = None,
    target_weight: float = None,
    current_weight: float = None,
    daily_calorie_limit: int = None,
    diet_type: str = None,
    allergies: list[str] = None,
    favorites: list[str] = None,
    dislikes: list[str] = None
) -> str:
    """儲存或更新用戶資料
    
    可以一次更新多個欄位，未提供的欄位會保留原值。
    
    Args:
        name: 用戶姓名
        target_weight: 目標體重 (kg)
        current_weight: 當前體重 (kg)
        daily_calorie_limit: 每日熱量上限 (kcal)
        diet_type: 飲食類型（一般/素食/純素）
        allergies: 過敏原列表
        favorites: 喜愛食材列表
        dislikes: 不喜歡食材列表
    
    Returns:
        儲存結果訊息
    """
    _ensure_data_dir()
    
    # 載入現有資料或建立新資料
    if os.path.exists(USER_PROFILE_PATH):
        with open(USER_PROFILE_PATH, 'r', encoding='utf-8') as f:
            profile = json.load(f)
    else:
        profile = _get_default_profile()
    
    # 更新提供的欄位
    updated_fields = []
    
    if name is not None:
        profile['name'] = name
        updated_fields.append(f"姓名: {name}")
    
    if target_weight is not None:
        profile['target_weight'] = target_weight
        updated_fields.append(f"目標體重: {target_weight} kg")
    
    if current_weight is not None:
        profile['current_weight'] = current_weight
        updated_fields.append(f"當前體重: {current_weight} kg")
        # 同時記錄到體重歷史
        today = datetime.now().strftime('%Y-%m-%d')
        # 移除今天的舊記錄（如果有）
        profile['weight_history'] = [w for w in profile.get('weight_history', []) if w.get('date') != today]
        profile['weight_history'].append({'date': today, 'weight': current_weight})
    
    if daily_calorie_limit is not None:
        profile['daily_calorie_limit'] = daily_calorie_limit
        updated_fields.append(f"每日熱量上限: {daily_calorie_limit} kcal")
    
    if 'preferences' not in profile:
        profile['preferences'] = {}
    
    if diet_type is not None:
        profile['preferences']['diet_type'] = diet_type
        updated_fields.append(f"飲食類型: {diet_type}")
    
    if allergies is not None:
        profile['preferences']['allergies'] = allergies
        updated_fields.append(f"過敏原: {', '.join(allergies) or '無'}")
    
    if favorites is not None:
        profile['preferences']['favorites'] = favorites
        updated_fields.append(f"喜愛食材: {', '.join(favorites)}")
    
    if dislikes is not None:
        profile['preferences']['dislikes'] = dislikes
        updated_fields.append(f"不喜歡食材: {', '.join(dislikes)}")
    
    # 儲存
    try:
        with open(USER_PROFILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        
        if updated_fields:
            return f"已成功更新用戶資料：\n" + "\n".join(f"- {f}" for f in updated_fields)
        else:
            return "未提供任何需要更新的欄位。"
    
    except Exception as e:
        return f"儲存用戶資料時發生錯誤：{str(e)}"


@tool
def record_meal(meal_type: str, foods: list[str], total_calories: float) -> str:
    """記錄一餐的飲食
    
    Args:
        meal_type: 餐別（早餐/午餐/晚餐/點心）
        foods: 食物列表
        total_calories: 總熱量 (kcal)
    
    Returns:
        記錄結果和今日飲食摘要
    """
    _ensure_data_dir()
    
    # 載入或建立資料
    if os.path.exists(USER_PROFILE_PATH):
        with open(USER_PROFILE_PATH, 'r', encoding='utf-8') as f:
            profile = json.load(f)
    else:
        profile = _get_default_profile()
    
    if 'meal_records' not in profile:
        profile['meal_records'] = []
    
    # 新增記錄
    today = datetime.now().strftime('%Y-%m-%d')
    record = {
        'date': today,
        'meal': meal_type,
        'foods': foods,
        'total_calories': total_calories,
        'time': datetime.now().strftime('%H:%M')
    }
    profile['meal_records'].append(record)
    
    # 計算今日總攝取
    today_meals = [m for m in profile['meal_records'] if m.get('date') == today]
    today_calories = sum(m.get('total_calories', 0) for m in today_meals)
    daily_limit = profile.get('daily_calorie_limit', 1800)
    remaining = daily_limit - today_calories
    
    # 儲存
    with open(USER_PROFILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    
    result = f"""已記錄 {meal_type}：
- 食物：{', '.join(foods)}
- 熱量：{total_calories} kcal

【今日飲食摘要】
- 已攝取：{today_calories} kcal
- 每日上限：{daily_limit} kcal
- 剩餘額度：{remaining} kcal"""
    
    if remaining < 0:
        result += f"\n⚠️ 警告：已超過每日熱量上限 {abs(remaining)} kcal！"
    elif remaining < 300:
        result += f"\n💡 提醒：剩餘額度較少，晚餐建議選擇低熱量食物。"
    
    return result


@tool
def plan_feast(date: str, description: str, estimated_calories: float) -> str:
    """規劃大餐日
    
    記錄預定的大餐，用於調整前後幾天的飲食建議。
    
    Args:
        date: 日期（格式：YYYY-MM-DD）
        description: 大餐描述（如：火鍋、聚餐）
        estimated_calories: 預估熱量 (kcal)
    
    Returns:
        規劃結果和調整建議
    """
    _ensure_data_dir()
    
    # 載入或建立資料
    if os.path.exists(USER_PROFILE_PATH):
        with open(USER_PROFILE_PATH, 'r', encoding='utf-8') as f:
            profile = json.load(f)
    else:
        profile = _get_default_profile()
    
    if 'planned_feasts' not in profile:
        profile['planned_feasts'] = []
    
    # 新增大餐規劃
    feast = {
        'date': date,
        'description': description,
        'estimated_calories': estimated_calories
    }
    
    # 移除同一天的舊規劃（如果有）
    profile['planned_feasts'] = [p for p in profile['planned_feasts'] if p.get('date') != date]
    profile['planned_feasts'].append(feast)
    
    # 儲存
    with open(USER_PROFILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    
    # 計算調整建議
    daily_limit = profile.get('daily_calorie_limit', 1800)
    excess = estimated_calories - daily_limit
    
    result = f"""已規劃大餐：
- 日期：{date}
- 內容：{description}
- 預估熱量：{estimated_calories} kcal
"""
    
    if excess > 0:
        # 計算需要分攤到前後幾天
        days_to_adjust = min(3, max(1, int(excess / 300)))  # 每天最多減少 300 kcal
        daily_reduction = excess / days_to_adjust
        
        result += f"""
【調整建議】
大餐預估超過每日上限 {excess:.0f} kcal
建議在大餐前 {days_to_adjust} 天，每天減少約 {daily_reduction:.0f} kcal
- 選擇低熱量主食（如糙米飯代替白飯）
- 增加蔬菜比例
- 減少油脂攝取
"""
    else:
        result += "\n✅ 大餐熱量在每日上限內，無需特別調整。"
    
    return result


# 匯出所有 Tools
storage_tools = [load_user_profile, save_user_profile, record_meal, plan_feast]
