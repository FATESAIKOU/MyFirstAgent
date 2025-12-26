"""
步驟 7.1：使用 LangGraph 的對話

改進點：
- 從直接調用 LLM → 使用 Graph 架構
- 雖然功能相同，但架構更易擴展

Graph 結構：
    [START] → [chatbot] → [END]

學習重點：
1. graph.invoke() 執行 Graph
2. State 的輸入輸出格式
3. 訊息的流動方式
"""

from langchain_core.messages import HumanMessage
from src.agent.graph import agent_graph


def chat_with_graph(user_input: str) -> str:
    """使用 Graph 進行對話
    
    invoke() 參數：
    - 輸入是初始 State（或部分 State）
    - 輸出是最終 State
    
    注意：目前沒有記憶功能，每次調用都是獨立對話
    """
    # 準備輸入 State
    input_state = {
        "messages": [HumanMessage(content=user_input)]
    }
    
    # 執行 Graph
    result = agent_graph.invoke(input_state)
    
    # 從結果中提取最後一條訊息（AI 回覆）
    ai_message = result["messages"][-1]
    return ai_message.content


def main():
    """CLI 主迴圈"""
    print("=" * 50)
    print("🥗 營養師 AI 助手 (步驟 7.1: LangGraph)")
    print("輸入 'quit' 或 'q' 退出")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ("quit", "q", "exit"):
                print("再見！")
                break
            
            print("\n助手: ", end="", flush=True)
            response = chat_with_graph(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n再見！")
            break


if __name__ == "__main__":
    main()
