"""
LangGraph Agent 定義

這是 Agent 的核心：使用 StateGraph 建構對話流程

Graph 結構演進：
- 步驟 7.1: [START] → [chatbot] → [END]
- 步驟 7.2: 加入 MemorySaver
- 步驟 7.3: 加入 Tool Calling（營養計算）
- 步驟 7.4: 加入用戶資料 Tools
- 步驟 9: 加入 Vision 圖片解析

當前結構：
    [START] → [check_image?]
                 ↓ "vision" (有圖片)
              [vision] → [chatbot] → [should_continue?]
                 ↓ "chatbot" (無圖片)      ↓ "tools" (有 tool_calls)
              [chatbot]                 [tools] → [chatbot]
                                          ↓ END (無 tool_calls)
                                       [END]

學習重點：
1. bind_tools() - 綁定工具到 LLM
2. ToolNode - 自動執行工具
3. conditional_edges - 條件分支
4. Vision 多模態處理 - ollama.chat() 支援 images 參數
5. State 擴展 - pending_image 欄位暫存圖片
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.prompts import NUTRITIONIST_SYSTEM_PROMPT
from src.config.settings import OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_VISION_MODEL
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


def create_vision_llm() -> ChatOllama:
    """建立 Vision LLM 實例（不綁定工具）"""
    return ChatOllama(
        model=OLLAMA_VISION_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,  # Vision 任務使用較低溫度以保持穩定性
    )


# 全域 LLM 實例
llm = create_llm()
vision_llm = create_vision_llm()


def vision_node(state: AgentState) -> dict:
    """Vision 節點 - 處理圖片分析
    
    職責：
    1. 讀取 pending_image（base64 圖片資料）
    2. 使用 Vision LLM (moondream) 分析圖片
    3. 將分析結果加入到 messages
    4. 清空 pending_image
    
    LangChain 多模態支援：
    - ✅ ChatOllama 完整支援 Vision
    - ✅ 使用標準的 HumanMessage 多模態格式
    - ✅ content 為 list，包含 text 和 image_url
    
    最佳實踐說明：
    - ✅ Vision 處理在 Graph Node 中（不是外部 helper）
    - ✅ 使用 State 傳遞圖片（pending_image 欄位）
    - ✅ 使用 LangChain API（統一風格，不需額外依賴 ollama SDK）
    - ✅ 多模態訊息格式符合 LangChain 標準
    """
    image_data = state.get("pending_image")
    
    if not image_data:
        return {}  # 沒有圖片，不做任何事
    
    print("[Vision] 正在分析圖片... (可能需要 3-5 秒)")
    
    # 建立多模態訊息（文字 + 圖片）
    # 注意：moondream 的中文支援較弱，使用英文 prompt 效果更好
    vision_message_input = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Describe the food in this image. Include food types and quantities if visible."
            },
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_data}"
            }
        ]
    )
    
    # 使用 Vision LLM 分析圖片
    response = vision_llm.invoke([vision_message_input])
    vision_result = response.content
    
    # 將 Vision 結果加入對話，讓主 LLM 根據這個英文描述來回應用戶
    result_message = HumanMessage(
        content=f"[圖片分析結果（來自 Vision 模型）]\n{vision_result}\n\n請根據以上 Vision 分析結果，用繁體中文回答用戶的問題。"
    )
    
    # 返回更新：
    # - 加入 Vision 分析訊息
    # - 清空 pending_image
    return {
        "messages": [result_message],
        "pending_image": None
    }


def check_image(state: AgentState) -> str:
    """條件分支：檢查是否有待處理的圖片
    
    - 有 pending_image → 去 "vision" 節點
    - 沒有 → 直接去 "chatbot" 節點
    """
    if state.get("pending_image"):
        return "vision"
    return "chatbot"


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
    
    當前結構（Step 9 - Vision）：
        [START] → [check_image?]
                     ↓ "vision" (有圖片)
                  [vision] → [chatbot] → [should_continue?]
                     ↓ "chatbot" (無圖片)     ↓ "tools"
                  [chatbot]               [tools] → [chatbot]
                                            ↓ END
                                         [END]
    
    設計說明：
    1. check_image 條件分支：START 後先檢查是否有圖片
    2. vision 節點：分析圖片並將結果加入 messages
    3. 圖片處理後流向 chatbot，讓 LLM 理解 Vision 結果
    4. 保留原有的 Tool Calling 流程
    """
    # 1. 建立 StateGraph
    graph_builder = StateGraph(AgentState)
    
    # 2. 添加節點
    graph_builder.add_node("vision", vision_node)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_node("tools", ToolNode(all_tools))
    
    # 3. 添加邊
    # START → check_image 條件分支
    graph_builder.add_conditional_edges(
        START,
        check_image,
        {
            "vision": "vision",
            "chatbot": "chatbot",
        }
    )
    
    # vision → chatbot（分析完圖片後，讓 LLM 處理）
    graph_builder.add_edge("vision", "chatbot")
    
    # chatbot → should_continue 條件分支
    graph_builder.add_conditional_edges(
        "chatbot",
        should_continue,
        {
            "tools": "tools",
            END: END,
        }
    )
    
    # tools → chatbot（工具執行完後，回到 chatbot）
    graph_builder.add_edge("tools", "chatbot")
    
    # 4. 建立 Checkpointer（對話記憶）
    memory = MemorySaver()
    
    # 5. 編譯
    graph = graph_builder.compile(checkpointer=memory)
    
    return graph


# 建立全域 Graph 實例
agent_graph = create_graph()
