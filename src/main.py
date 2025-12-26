"""
步驟 7.2：多輪對話記憶

改進點：
- 加入 MemorySaver（Checkpointer）
- 使用 thread_id 區分對話
- 能記住之前說過的話

Graph 結構：
    [START] → [chatbot] → [END]
    （加入 checkpointer 記憶）

學習重點：
1. config 參數傳遞 thread_id
2. Checkpointer 的工作原理
3. 對話歷史的自動累積
"""

from langchain_core.messages import HumanMessage
from src.agent.graph import agent_graph


def chat_with_memory(user_input: str, thread_id: str = "default") -> str:
    """使用 Graph 進行對話（帶記憶）
    
    thread_id 的作用：
    - 區分不同的對話線程
    - 同一個 thread_id 的對話會共享歷史
    - 不同 thread_id 是獨立的對話
    
    config 參數：
    - configurable: 傳遞可配置項
    - thread_id: 對話線程標識
    """
    # 準備輸入 State
    input_state = {
        "messages": [HumanMessage(content=user_input)]
    }
    
    # 配置：指定對話線程
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    # 執行 Graph（帶配置）
    result = agent_graph.invoke(input_state, config=config)
    
    # 從結果中提取最後一條訊息（AI 回覆）
    ai_message = result["messages"][-1]
    return ai_message.content


def main():
    """CLI 主迴圈"""
    print("=" * 50)
    print("🥗 營養師 AI 助手 (步驟 7.4: 用戶資料)")
    print("輸入 'quit' 或 'q' 退出")
    print("輸入 'new' 開始新對話")
    print("=" * 50)
    
    # 使用固定的 thread_id（同一次執行期間共享記憶）
    thread_id = "user_session_1"
    
    while True:
        try:
            user_input = input("\n你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ("quit", "q", "exit"):
                print("再見！")
                break
            
            if user_input.lower() == "new":
                # 產生新的 thread_id 開始新對話
                import time
                thread_id = f"session_{int(time.time())}"
                print("🔄 已開始新對話！")
                continue
            
            print("\n助手: ", end="", flush=True)
            response = chat_with_memory(user_input, thread_id)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n再見！")
            break


if __name__ == "__main__":
    main()
