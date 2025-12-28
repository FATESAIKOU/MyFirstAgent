"""
Phase 2：體重管理食譜推薦 AI Agent + Vision

功能：
- 多輪對話記憶
- 營養計算（Tool Calling）
- 用戶資料持久化（JSON）
- 個人化食譜推薦
- 大餐日規劃
- 圖片分析（Vision）

Graph 結構：
    [START] → [check_image?]
                 ↓ vision
              [vision] → [chatbot] → [should_continue?]
                 ↓ chatbot              ↓ tools
              [chatbot]            [tools] → [chatbot]
                                      ↓ END
                                   [END]
"""

from langchain_core.messages import HumanMessage
import base64
import os

from src.agent.graph import agent_graph


def chat_with_memory(user_input: str, thread_id: str = "default", image_path: str | None = None) -> str:
    """使用 Graph 進行對話（帶記憶 + Vision）
    
    thread_id 的作用：
    - 區分不同的對話線程
    - 同一個 thread_id 的對話會共享歷史
    - 不同 thread_id 是獨立的對話
    
    image_path 參數：
    - 圖片檔案路徑（可選）
    - 會被 base64 編碼後放入 State 的 pending_image
    
    config 參數：
    - configurable: 傳遞可配置項
    - thread_id: 對話線程標識
    """
    # 準備輸入 State
    input_state = {
        "messages": [HumanMessage(content=user_input)]
    }
    
    # 如果有圖片，讀取並 base64 編碼
    if image_path:
        try:
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            input_state["pending_image"] = image_data
        except Exception as e:
            print(f"⚠️  圖片載入失敗: {e}")
    
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
    print("=" * 60)
    print("🥗 營養師 AI 助手 (Phase 2: Vision)")
    print("輸入 'quit' 或 'q' 退出")
    print("輸入 'new' 開始新對話")
    print("圖片輸入: image:/path/to/image.jpg")
    print("=" * 60)
    
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
            
            # 檢查是否為圖片輸入
            image_path = None
            text_input = user_input
            
            if user_input.startswith("image:"):
                image_path = user_input[6:].strip()
                
                if not os.path.exists(image_path):
                    print(f"❌ 圖片檔案不存在: {image_path}")
                    continue
                
                text_input = "請分析這張圖片"
                print(f"📷 已載入圖片: {image_path}")
            
            print("\n助手: ", end="", flush=True)
            response = chat_with_memory(text_input, thread_id, image_path)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n再見！")
            break


if __name__ == "__main__":
    main()
