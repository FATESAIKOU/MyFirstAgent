"""
Agent State 定義

State 是 LangGraph 的核心概念：
- State 定義了在 Graph 中流動的資料結構
- 每個 Node 接收 State，處理後返回更新
- State 的更新是透過 Reducer 函數合併

學習重點：
1. TypedDict 定義 State 結構
2. Annotated + Reducer 定義合併策略
3. add_messages 是處理訊息歷史的內建 Reducer
4. 自定義欄位不需要 Reducer（直接覆蓋）
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Agent 狀態定義
    
    Attributes:
        messages: 對話歷史，使用 add_messages Reducer
                  - 新訊息會被追加到列表末尾
                  - 自動處理訊息 ID 和去重
        
        pending_image: 待處理的圖片資料（base64）
                      - 用於暫存用戶上傳的圖片
                      - Vision Node 處理後清空
                      - 不需要 Reducer，直接覆蓋
    
    Reducer 說明：
        add_messages 是 LangGraph 內建的 Reducer
        當 Node 返回 {"messages": [new_msg]} 時
        實際執行: state["messages"] = add_messages(old_messages, new_msg)
        
        pending_image 沒有 Reducer，直接覆蓋：
        返回 {"pending_image": data} -> state["pending_image"] = data
    """
    messages: Annotated[list[BaseMessage], add_messages]
    pending_image: str | None  # base64 編碼的圖片，或 None


# 未來可擴展的欄位（步驟 7.4 加入）：
# class AgentState(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]
#     user_profile: dict | None  # 用戶資料
#     current_date: str          # 當前日期
