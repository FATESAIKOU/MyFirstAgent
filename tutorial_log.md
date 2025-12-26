# AI Agent 開發教學記錄

> 專案：體重管理食譜推薦 AI Agent  
> 框架：Python + LangGraph + Ollama  
> 日期：2025-12-26

---

## 步驟 1：理解程式執行環境 ✅

### 環境總覽

| 項目 | 數值 | 評估 |
|------|------|------|
| **Python** | 3.12.7 (asdf 管理) | ✅ 優秀，LangGraph 完全支援 |
| **套件管理** | Poetry 2.1.1 | ✅ 推薦使用 |
| **Ollama** | 0.6.8 (已安裝，服務未啟動) | ⚠️ 需啟動服務 |
| **GPU** | RTX 2070 SUPER (8GB VRAM) | ✅ 可運行 7B~13B 模型 |
| **RAM** | 62 GB | ✅ 非常充裕 |
| **CPU** | i7-10700 (16 threads) | ✅ 良好 |

### 硬體能力分析

| 模型大小 | 適用性 | 說明 |
|---------|--------|------|
| **7B 模型** | ✅ 推薦 | 流暢運行，約 4-5GB VRAM |
| **13B 模型** | ⚠️ 可用 | 勉強可用，需量化版本 (Q4) |
| **32B+ 模型** | ❌ 不推薦 | VRAM 不足 |

### 結論
- 環境非常適合開發 AI Agent
- 建議使用 7B 模型（如 qwen2.5:7b 或 llama3.2）

---

## 步驟 2：需求設計 & 定義 ✅

### 專案目標
完成一個協助個人透過生成午餐晚餐食譜來控管體重的 AI Agent

### 核心場景
1. 提供家裡食材 → 推薦午餐/晚餐
2. 上傳零食圖片 → 計算後果 + 可視化影響（Phase 3）
3. 設定大餐日 → 調整後續食譜推薦

### 確定的技術選型

| 決策點 | 選擇 | 原因 |
|--------|------|------|
| **互動介面** | CLI（命令行） | 最簡單，專注於 Agent 核心學習 |
| **資料持久化** | JSON 檔案 | 輕量、易於調試 |
| **資料追蹤** | 體重歷史、飲食記錄、食性偏好 | - |
| **目標設定** | Agent 引導式詢問 | 展示多輪對話能力 |

### Phase 1 範圍（最小可用版本）

✅ **包含的功能**：
- 基礎對話（LangGraph + Ollama）
- 引導式資料收集（多輪對話）
- 食譜推薦（基於 LLM 內建知識）
- 營養計算（內建簡化計算）
- 資料追蹤（JSON 檔案存取）

❌ **Phase 3 再加入**：
- 圖片解析（Vision）
- 網路搜尋（MCP）
- 可視化圖表

### 典型使用流程

```
第一次使用：
1. 啟動 Agent
2. Agent 引導詢問：目標體重、當前體重、食性偏好
3. 存入 JSON

日常使用（午餐推薦）：
用戶: "我家裡有雞胸肉、花椰菜、糙米，推薦午餐"
Agent: [讀取用戶檔案] → [調用營養計算工具] → [生成食譜] → [更新飲食記錄]

規劃大餐：
用戶: "週六要吃火鍋，大概 2000 大卡"
Agent: [記錄大餐日] → "建議週四五降低熱量攝取..."
```

### 核心學習目標

| 學習目標 | 對應實現 |
|---------|---------|
| LangGraph 基礎 | 建立 State + Node + Edge |
| State 管理 | 用戶資料在圖中流動 |
| Tool Calling | 營養計算、檔案讀寫 |
| 條件分支 | 是否需要引導、是否需要更新資料 |
| 與 Ollama 整合 | ChatOllama 配置 |
| Prompt Engineering | 食譜推薦、引導式詢問 |

---

## 步驟 3：規劃模型/MCP/套件 ✅

### Ollama 模型選型

基於硬體（RTX 2070 SUPER 8GB）和 2025 年最新模型列表：

| 模型 | 參數量 | VRAM 需求 | 推薦度 | 說明 |
|------|--------|-----------|--------|------|
| **qwen3:8b** | 8B | ~5GB | ⭐⭐⭐⭐⭐ | **最終選擇** - Qwen 最新版，中文強，Tool Calling 佳 |
| qwen2.5:7b | 7B | ~4.5GB | ⭐⭐⭐⭐ | 上一代旗艦，依然強勁 |
| llama3.2:3b | 3B | ~2.5GB | ⭐⭐⭐ | 更快但中文稍弱 |

**最終決定：`qwen3:8b`**
- 2025.10 月更新，最新技術
- 原生 Tool Calling 支援
- 中文食譜推薦表現優異

### 重要概念澄清：Tools vs MCP

**LangGraph 中的 Tools**：
```
Agent → LLM (Function Calling) → Python 函數 (Tools)
                                  ↓
                       - calculate_nutrition()
                       - save_to_json()
                       - read_user_profile()
```

**Phase 1 不需要 MCP！**
- Tools 是 Agent 透過 LLM 的 Function Calling 直接調用的 Python 函數
- MCP (Model Context Protocol) 用於連接外部服務（網路搜尋、瀏覽器等）
- Phase 3 需要網路搜尋時才會引入 MCP

### Python 套件規劃

**核心依賴**：
```toml
[tool.poetry.dependencies]
python = "^3.12"
langgraph = "^0.2.45"           # Agent 框架
langchain-ollama = "^0.2.0"     # Ollama 整合
langchain-core = "^0.3.21"      # LangChain 核心
pydantic = "^2.10.5"            # 資料驗證
```

**開發依賴**：
```toml
[tool.poetry.group.dev.dependencies]
pytest = "^8.3.4"               # 測試
black = "^24.12.0"              # 格式化
ruff = "^0.8.4"                 # Linter
ipython = "^8.31.0"             # 互動式開發
```

**Phase 1 不需要的套件**（Phase 3+ 再加）：
- `mcp` - Model Context Protocol（網路搜尋時需要）
- `pillow` - 圖片處理（Vision 功能時需要）
- `langchain-community` - 外部工具整合
- `matplotlib` - 資料視覺化

### 專案結構規劃

```
MyFirstAgent/
├── pyproject.toml              # Poetry 配置
├── README.md                   # 專案說明
├── tutorial_log.md             # 教學記錄
├── data/                       # 資料目錄
│   └── user_profile.json       # 用戶資料
├── src/
│   ├── __init__.py
│   ├── main.py                 # CLI 入口
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py            # LangGraph 定義
│   │   └── state.py            # State 定義
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── nutrition.py        # 營養計算工具
│   │   └── storage.py          # JSON 讀寫工具
│   └── config/
│       ├── __init__.py
│       └── settings.py         # 配置（模型名稱等）
└── tests/                      # 測試（Phase 2+）
    └── __init__.py
```

### 技術決策總結

| 決策項 | 選擇 | 理由 |
|--------|------|------|
| **LLM 模型** | qwen3:8b | 2025 最新、中文強、Tool Calling 佳 |
| **LangGraph 版本** | 0.2.45 | 最新穩定版 |
| **配置管理** | Pydantic Settings | 型別安全 |
| **專案管理** | Poetry | 依賴隔離佳 |
| **程式碼風格** | Black + Ruff | 業界標準 |

---

## 步驟 4：展開功能列表與實現大綱 ✅

### Phase 1 實現步驟大綱

| 步驟 | 步驟名 | 實施內容大綱 | 說明的技術 | 預期效果 | 新增・變更的檔案 |
|------|--------|-------------|-----------|---------|-----------------|
| **5** | 構建專案骨架 | Poetry 初始化、目錄結構、依賴安裝、Ollama 模型下載 | Poetry、專案結構設計 | 可執行 `poetry run python -c "import langgraph"` | `pyproject.toml`, `src/**/__init__.py` |
| **6** | 最小 LLM 對話 | 單輪對話，無狀態，直接調用 Ollama | LangChain-Ollama 整合、Prompt 基礎 | CLI 輸入問題 → 得到回覆 | `src/main.py`, `src/config/settings.py` |
| **7.1** | 引入 LangGraph State | 定義 AgentState、建立最簡單的單節點 Graph | LangGraph State、Graph 結構 | 同上，但用 Graph 架構 | `src/agent/state.py`, `src/agent/graph.py` |
| **7.2** | 多輪對話記憶 | State 中加入 messages 歷史 | Checkpointer、對話記憶 | 能記住上一輪說的話 | `src/agent/graph.py` (修改) |
| **7.3** | 第一個 Tool：營養計算 | 定義 Tool Schema、LLM 調用 Tool | Tool Calling、Function Schema | "雞胸肉 200g 多少熱量" → 自動計算 | `src/tools/nutrition.py`, `src/agent/graph.py` |
| **7.4** | 用戶資料持久化 | JSON 讀寫 Tool、首次使用引導流程 | Tool 組合、條件分支 | 新用戶 → 引導輸入 → 存入 JSON | `src/tools/storage.py`, `data/user_profile.json` |
| **7.5** | 食譜推薦功能 | 結合用戶資料 + 食材 → 推薦食譜 | Prompt Engineering、上下文注入 | "有雞肉和青菜" → 個人化食譜推薦 | `src/agent/prompts.py` |
| **7.6** | 大餐日規劃 | 記錄預定大餐、調整推薦策略 | 多步推理、狀態更新 | "週六吃火鍋" → "週四五建議減少攝取..." | `src/tools/storage.py` (修改) |

### 各步驟學習重點

**步驟 5 - 構建專案骨架**
- Poetry 專案管理
- Python 套件結構設計
- Ollama 模型下載與驗證

**步驟 6 - 最小 LLM 對話**
- ChatOllama 使用方式
- 基本 Prompt 設計
- 驗證 LLM 連接

**步驟 7.1 - 引入 LangGraph State**
- StateGraph 概念
- Node（節點）與 Edge（邊）
- Graph 編譯與執行

**步驟 7.2 - 多輪對話記憶**
- MemorySaver / Checkpointer
- messages 歷史管理
- thread_id 概念

**步驟 7.3 - 第一個 Tool：營養計算**
- @tool 裝飾器
- bind_tools() 綁定工具
- ToolNode 執行工具
- 條件分支（should_continue）

**步驟 7.4 - 用戶資料持久化**
- 多 Tool 組合使用
- 條件邏輯（新用戶 vs 舊用戶）
- JSON 檔案操作

**步驟 7.5 - 食譜推薦功能**
- System Prompt 設計
- 動態 Prompt 組裝
- 上下文注入（用戶資料）

**步驟 7.6 - 大餐日規劃**
- 多步推理
- 狀態連動更新
- 時間相關邏輯

---

## 步驟 5：構建專案骨架 ✅

### 執行記錄

1. **Poetry 專案配置** - `pyproject.toml`
   ```toml
   [tool.poetry]
   package-mode = false  # 不打包，純開發用
   
   [tool.poetry.dependencies]
   python = "^3.12"
   langgraph = "^0.2.45"
   langchain-ollama = "^0.2.0"
   langchain-core = "^0.3.21"
   pydantic = "^2.10.5"
   ```

2. **目錄結構建立**
   ```
   MyFirstAgent/
   ├── pyproject.toml
   ├── poetry.lock
   ├── README.md
   ├── tutorial_log.md
   ├── data/.gitkeep
   ├── src/
   │   ├── __init__.py
   │   ├── main.py
   │   ├── agent/__init__.py
   │   ├── tools/__init__.py
   │   └── config/__init__.py
   └── tests/__init__.py
   ```

3. **依賴安裝** - `poetry install` ✅

4. **Ollama 模型下載** - `ollama pull qwen3:8b` ✅
   - 模型大小：5.2 GB
   - 狀態：可用

### 驗證結果

```bash
$ poetry run python -c "import langgraph; import langchain_ollama; print('✅')"
✅ 依賴安裝成功！

$ ollama list | grep qwen3
qwen3:8b    500a1f067a9f    5.2 GB
```

### 學習重點

1. **Poetry package-mode = false**：表示這是應用程式專案，不是要發佈的套件
2. **專案結構設計**：
   - `src/agent/` - Agent 核心邏輯（Graph、State）
   - `src/tools/` - Agent 可調用的工具函數
   - `src/config/` - 配置管理
   - `data/` - 運行時資料存放

---

## 步驟 6：最小 LLM 對話 ✅

### 執行記錄

1. **建立配置檔** - `src/config/settings.py`
   ```python
   OLLAMA_BASE_URL = "http://localhost:11434"
   OLLAMA_MODEL = "qwen3:8b"
   ```

2. **實現 CLI 對話** - `src/main.py`
   - 使用 `ChatOllama` 連接本地 Ollama
   - `SystemMessage` 定義 AI 角色（營養師助手）
   - `HumanMessage` 傳遞用戶輸入
   - `invoke()` 同步調用 LLM

3. **qwen3 特性處理**
   - qwen3 預設啟用 thinking mode，會輸出 `<think>` 標籤
   - 加入 `/no_think` 指令關閉思考過程輸出
   - 清理殘留的空標籤

### 程式碼結構

```python
# 建立 LLM
llm = ChatOllama(model="qwen3:8b", temperature=0.7)

# 組裝訊息
messages = [
    SystemMessage(content="你是營養師助手..."),
    HumanMessage(content=user_input),
]

# 調用 LLM
response = llm.invoke(messages)
```

### 驗證結果

```bash
$ echo "100g雞胸肉有多少熱量" | poetry run python -m src.main
助手: 100g雞胸肉約有165-200大卡的熱量
```

### 學習重點

1. **ChatOllama** - LangChain 對 Ollama 的封裝
2. **訊息類型**：
   - `SystemMessage` - 系統提示，定義 AI 人設
   - `HumanMessage` - 用戶輸入
   - `AIMessage` - AI 回覆（invoke 返回）
3. **invoke() vs stream()** - 同步 vs 串流輸出
4. **temperature** - 控制回覆隨機性

### 目前限制（下一步解決）

- ❌ 無對話記憶（每次都是新對話）
- ❌ 無 Graph 架構（難以擴展）

---

## 步驟 7.1：引入 LangGraph State ✅

### 執行記錄

1. **定義 State** - `src/agent/state.py`
   ```python
   class AgentState(TypedDict):
       messages: Annotated[list[BaseMessage], add_messages]
   ```
   - `Annotated` + `add_messages` = 訊息自動追加合併

2. **建立 Graph** - `src/agent/graph.py`
   ```python
   graph_builder = StateGraph(AgentState)
   graph_builder.add_node("chatbot", chatbot_node)
   graph_builder.add_edge(START, "chatbot")
   graph_builder.add_edge("chatbot", END)
   graph = graph_builder.compile()
   ```

3. **更新 main.py** - 改用 `graph.invoke()`

### Graph 結構

```
[START] → [chatbot] → [END]
```

### 程式碼重點

```python
# Node 函數：接收 State，返回 State 更新
def chatbot_node(state: AgentState) -> dict:
    messages = [SystemMessage(...)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}  # Reducer 會自動合併

# 執行 Graph
result = agent_graph.invoke({"messages": [HumanMessage(content="...")]})
```

### 驗證結果

```bash
$ echo "推薦一個低熱量午餐" | poetry run python -m src.main
助手: 推薦一道低熱量的 雞胸肉沙拉...
```

### 學習重點

1. **StateGraph** - 狀態圖容器
2. **add_node()** - 添加處理節點
3. **add_edge()** - 連接節點（定義流程）
4. **compile()** - 編譯成可執行圖
5. **Reducer (add_messages)** - 定義 State 如何合併更新

### 目前限制（下一步解決）

- ❌ 無對話記憶（每次 invoke 都是新對話）

---

## 步驟 7.2：多輪對話記憶 ✅

### 執行記錄

1. **引入 MemorySaver** - `src/agent/graph.py`
   ```python
   from langgraph.checkpoint.memory import MemorySaver
   
   memory = MemorySaver()
   graph = graph_builder.compile(checkpointer=memory)
   ```

2. **使用 thread_id** - `src/main.py`
   ```python
   config = {"configurable": {"thread_id": "user_session_1"}}
   result = agent_graph.invoke(input_state, config=config)
   ```

### 程式碼重點

```python
# Checkpointer 工作原理：
# 1. 每次 invoke 結束後，自動保存 State
# 2. 下次 invoke 時，根據 thread_id 恢復 State
# 3. 新訊息透過 add_messages Reducer 合併到歷史

# thread_id 的作用：
# - 同一個 thread_id = 同一個對話（共享歷史）
# - 不同 thread_id = 獨立對話
```

### 驗證結果

```bash
$ printf "我叫小明\n我叫什麼名字\n" | poetry run python -m src.main

你: 我叫小明
助手: 你好，小明！很高興認識你...

你: 我叫什麼名字
助手: 你叫小明！  ← 成功記住！
```

### 學習重點

1. **MemorySaver** - 記憶體版本的 Checkpointer
   - 程式結束就消失
   - 可替換為 SQLite 版本持久化

2. **thread_id** - 對話線程標識
   - 區分不同用戶/對話
   - 是多租戶的基礎

3. **Checkpointer 生命週期**
   - invoke 前：根據 thread_id 載入舊 State
   - invoke 中：Node 處理並更新 State
   - invoke 後：保存新 State

### Graph 結構（不變）

```
[START] → [chatbot] → [END]
            ↓
      [MemorySaver] ← 新增：自動保存/恢復 State
```

---

## 步驟 7.3：第一個 Tool - 營養計算（進行中）
