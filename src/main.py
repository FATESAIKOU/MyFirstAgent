"""
步驟 6：最小 LLM 對話

這是最簡單的 LLM 調用示例：
- 無狀態（每次對話獨立）
- 直接調用 ChatOllama
- 單輪問答

學習重點：
1. ChatOllama 的基本使用
2. invoke() 方法
3. 訊息格式 (HumanMessage, AIMessage)
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from src.config.settings import OLLAMA_MODEL, OLLAMA_BASE_URL


def create_llm() -> ChatOllama:
    """建立 LLM 實例
    
    ChatOllama 參數說明：
    - model: Ollama 模型名稱
    - base_url: Ollama 服務地址
    - temperature: 生成隨機性 (0=確定性, 1=隨機)
    """
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,
    )


def chat_once(llm: ChatOllama, user_input: str) -> str:
    """單輪對話
    
    訊息類型：
    - SystemMessage: 系統提示，定義 AI 角色
    - HumanMessage: 用戶輸入
    - AIMessage: AI 回覆（invoke 返回）
    
    注意：qwen3 預設啟用 thinking mode，會輸出 <think> 標籤
    加入 /no_think 可關閉思考過程輸出
    """
    messages = [
        SystemMessage(content="你是一個專業的營養師助手，專門協助用戶規劃健康飲食。請用繁體中文回答。回答請簡潔扼要。 /no_think"),
        HumanMessage(content=user_input),
    ]
    
    # invoke() 是同步調用，返回 AIMessage
    response = llm.invoke(messages)
    
    # 清理可能的空 think 標籤
    content = response.content
    content = content.replace("<think>\n\n</think>\n\n", "").strip()
    return content


def main():
    """CLI 主迴圈"""
    print("=" * 50)
    print("🥗 營養師 AI 助手 (步驟 6: 最小對話)")
    print("輸入 'quit' 或 'q' 退出")
    print("=" * 50)
    
    llm = create_llm()
    
    while True:
        try:
            user_input = input("\n你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ("quit", "q", "exit"):
                print("再見！")
                break
            
            print("\n助手: ", end="", flush=True)
            response = chat_once(llm, user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n再見！")
            break


if __name__ == "__main__":
    main()
