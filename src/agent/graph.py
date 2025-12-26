"""
LangGraph Agent 定義

這是 Agent 的核心：使用 StateGraph 建構對話流程

Graph 結構（步驟 7.1 ~ 7.2）：
    [START] → [chatbot] → [END]

步驟 7.2 新增：
    - MemorySaver: 記憶對話歷史
    - thread_id: 區分不同對話線程

學習重點：
1. StateGraph - 狀態圖的容器
2. add_node - 添加處理節點
3. add_edge - 添加邊（節點間的連接）
4. compile - 編譯成可執行的 Graph
5. MemorySaver - 對話記憶（Checkpointer）
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.config.settings import OLLAMA_MODEL, OLLAMA_BASE_URL


# System Prompt - 定義 AI 角色
SYSTEM_PROMPT = """你是一個專業的營養師助手，專門協助用戶規劃健康飲食。

你的職責：
1. 根據用戶提供的食材推薦健康食譜
2. 計算食物的營養成分和熱量
3. 提供飲食建議以幫助用戶達成體重目標

請用繁體中文回答，回答簡潔扼要。 /no_think"""


def create_llm() -> ChatOllama:
    """建立 LLM 實例"""
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,
    )


# 全域 LLM 實例
llm = create_llm()


def chatbot_node(state: AgentState) -> dict:
    """聊天節點 - Graph 中的處理單元
    
    Node 函數規範：
    - 輸入：當前 State
    - 輸出：State 的更新部分（會透過 Reducer 合併）
    
    這個節點的職責：
    1. 讀取對話歷史 (state["messages"])
    2. 加入 System Prompt
    3. 調用 LLM
    4. 返回 AI 回覆
    """
    # 組裝訊息：System Prompt + 對話歷史
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    # 調用 LLM
    response = llm.invoke(messages)
    
    # 清理 qwen3 的 think 標籤
    if hasattr(response, 'content'):
        response.content = response.content.replace("<think>\n\n</think>\n\n", "").strip()
    
    # 返回更新：新訊息會被 add_messages Reducer 追加到歷史中
    return {"messages": [response]}


def create_graph() -> StateGraph:
    """建立 Agent Graph
    
    Graph 建構步驟：
    1. 建立 StateGraph，指定 State 類型
    2. add_node: 添加節點
    3. add_edge: 連接節點
    4. compile: 編譯成可執行圖（加入 Checkpointer）
    
    步驟 7.2 新增：MemorySaver
        - Checkpointer 負責在每次執行後保存 State
        - 下次執行時可恢復之前的 State
        - MemorySaver 是記憶體版本（程式結束就消失）
    
    目前結構：
        START → chatbot → END
    """
    # 1. 建立 StateGraph
    graph_builder = StateGraph(AgentState)
    
    # 2. 添加節點
    graph_builder.add_node("chatbot", chatbot_node)
    
    # 3. 添加邊
    graph_builder.add_edge(START, "chatbot")  # 起點 → chatbot
    graph_builder.add_edge("chatbot", END)     # chatbot → 終點
    
    # 4. 建立 Checkpointer（對話記憶）
    memory = MemorySaver()
    
    # 5. 編譯（傳入 checkpointer）
    graph = graph_builder.compile(checkpointer=memory)
    
    return graph


# 建立全域 Graph 實例
agent_graph = create_graph()
