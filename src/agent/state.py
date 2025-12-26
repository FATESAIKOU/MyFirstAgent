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
    
    Reducer 說明：
        add_messages 是 LangGraph 內建的 Reducer
        當 Node 返回 {"messages": [new_msg]} 時
        實際執行: state["messages"] = add_messages(old_messages, new_msg)
    """
    messages: Annotated[list[BaseMessage], add_messages]


# 未來可擴展的欄位（步驟 7.4 加入）：
# class AgentState(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]
#     user_profile: dict | None  # 用戶資料
#     current_date: str          # 當前日期
