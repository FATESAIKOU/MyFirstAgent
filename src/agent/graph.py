"""
LangGraph Agent 定義

這是 Agent 的核心：使用 StateGraph 建構對話流程

Graph 結構演進：
- 步驟 7.1: [START] → [chatbot] → [END]
- 步驟 7.2: 加入 MemorySaver
- 步驟 7.3: 加入 Tool Calling（營養計算）
- 步驟 7.4: 加入用戶資料 Tools

當前結構：
    [START] → [chatbot] → [should_continue?]
                              ↓ "tools" (有 tool_calls)
                         [tools] → [chatbot]
                              ↓ END (無 tool_calls)
                           [END]

學習重點：
1. bind_tools() - 綁定工具到 LLM
2. ToolNode - 自動執行工具
3. conditional_edges - 條件分支
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.prompts import NUTRITIONIST_SYSTEM_PROMPT
from src.config.settings import OLLAMA_MODEL, OLLAMA_BASE_URL
from src.tools.nutrition import nutrition_tools
from src.tools.storage import storage_tools

# 合併所有 Tools
all_tools = nutrition_tools + storage_tools


# 使用 prompts.py 中定義的 System Prompt
SYSTEM_PROMPT = NUTRITIONIST_SYSTEM_PROMPT


def create_llm() -> ChatOllama:
    """建立 LLM 實例（綁定工具）"""
    base_llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,
    )
    # bind_tools: 告訴 LLM 有哪些工具可用
    return base_llm.bind_tools(all_tools)


# 全域 LLM 實例（已綁定工具）
llm = create_llm()


def chatbot_node(state: AgentState) -> dict:
    """聊天節點 - Graph 中的處理單元
    
    Node 函數規範：
    - 輸入：當前 State
    - 輸出：State 的更新部分（會透過 Reducer 合併）
    
    這個節點的職責：
    1. 讀取對話歷史 (state["messages"])
    2. 加入 System Prompt
    3. 調用 LLM（可能生成 tool_calls 或直接回覆）
    4. 返回 AI 回覆
    """
    # 組裝訊息：System Prompt + 對話歷史
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    # 調用 LLM
    response = llm.invoke(messages)
    
    # 清理 qwen3 的 think 標籤
    if hasattr(response, 'content') and response.content:
        response.content = response.content.replace("<think>\n\n</think>\n\n", "").strip()
    
    # 返回更新：新訊息會被 add_messages Reducer 追加到歷史中
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """條件分支：決定下一步去哪個節點
    
    檢查最後一條訊息是否包含 tool_calls：
    - 有 tool_calls → 去 "tools" 節點執行工具
    - 沒有 → 結束對話
    """
    last_message = state["messages"][-1]
    
    # 檢查是否有工具調用請求
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    
    return END


def create_graph() -> StateGraph:
    """建立 Agent Graph
    
    當前結構：
        [START] → [chatbot] → [should_continue?]
                                    ↓ "tools"
                               [tools] → [chatbot]
                                    ↓ END
                                 [END]
    """
    # 1. 建立 StateGraph
    graph_builder = StateGraph(AgentState)
    
    # 2. 添加節點
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_node("tools", ToolNode(all_tools))
    
    # 3. 添加邊
    graph_builder.add_edge(START, "chatbot")
    
    # 條件邊：根據 should_continue 的返回值決定去向
    graph_builder.add_conditional_edges(
        "chatbot",
        should_continue,
        {
            "tools": "tools",
            END: END,
        }
    )
    
    # 工具執行完後，回到 chatbot 讓 LLM 處理結果
    graph_builder.add_edge("tools", "chatbot")
    
    # 4. 建立 Checkpointer（對話記憶）
    memory = MemorySaver()
    
    # 5. 編譯
    graph = graph_builder.compile(checkpointer=memory)
    
    return graph


# 建立全域 Graph 實例
agent_graph = create_graph()
