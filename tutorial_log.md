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

## 步驟 7.3：第一個 Tool - 營養計算 ✅

### 執行記錄

1. **建立 Tool** - `src/tools/nutrition.py`
   ```python
   @tool
   def calculate_nutrition(food_name: str, weight_grams: float) -> str:
       """計算食材的營養成分和熱量"""
       # LLM 根據 docstring 決定是否調用
       ...
   
   @tool
   def list_available_foods() -> str:
       """列出所有可查詢營養的食材清單"""
       ...
   ```

2. **綁定工具到 LLM** - `src/agent/graph.py`
   ```python
   llm = base_llm.bind_tools(nutrition_tools)
   ```

3. **新增 ToolNode** - 自動執行工具
   ```python
   graph_builder.add_node("tools", ToolNode(nutrition_tools))
   ```

4. **條件分支** - 判斷是否需要調用工具
   ```python
   def should_continue(state):
       if last_message.tool_calls:
           return "tools"
       return END
   
   graph_builder.add_conditional_edges("chatbot", should_continue, {...})
   ```

### Graph 結構

```
[START] → [chatbot] → [should_continue?]
                           ↓ "tools" (有 tool_calls)
                      [tools] → [chatbot]
                           ↓ END (無 tool_calls)
                        [END]
```

### 驗證結果

```bash
$ echo "雞胸肉150克有多少熱量" | poetry run python -m src.main
助手: 雞胸肉150克的熱量是247.5大卡。  ← 165 * 1.5 = 247.5 ✅

$ echo "有哪些食材可以查詢營養" | poetry run python -m src.main
助手: 肉類：雞胸肉、雞腿肉... ← 調用 list_available_foods ✅
```

### 學習重點

1. **@tool 裝飾器**
   - 將 Python 函數轉為 LangChain Tool
   - docstring 會成為工具描述（LLM 用來決策）
   - 參數型別 → JSON Schema

2. **bind_tools()**
   - 告訴 LLM 有哪些工具可用
   - LLM 會在適當時機生成 tool_calls

3. **ToolNode**
   - LangGraph 內建的工具執行節點
   - 自動根據 tool_calls 執行對應工具
   - 結果作為 ToolMessage 加入對話

4. **add_conditional_edges()**
   - 實現條件分支
   - 根據函數返回值決定下一個節點

5. **Tool Calling 流程**
   ```
   用戶問問題 → LLM 生成 tool_calls
   → ToolNode 執行工具 → 返回 ToolMessage
   → LLM 根據結果生成最終回覆
   ```

---

## 步驟 7.4：用戶資料持久化 ✅

### 執行記錄

1. **建立 Storage Tools** - `src/tools/storage.py`
   ```python
   @tool
   def load_user_profile() -> str:
       """載入用戶資料"""
   
   @tool
   def save_user_profile(name, target_weight, ...) -> str:
       """儲存用戶資料"""
   
   @tool
   def record_meal(meal_type, foods, total_calories) -> str:
       """記錄飲食"""
   
   @tool
   def plan_feast(date, description, estimated_calories) -> str:
       """規劃大餐"""
   ```

2. **整合到 Graph** - 合併所有 Tools
   ```python
   all_tools = nutrition_tools + storage_tools
   llm = base_llm.bind_tools(all_tools)
   ```

3. **JSON 資料結構**
   ```json
   {
     "name": "小明",
     "target_weight": 70.0,
     "current_weight": 75.0,
     "daily_calorie_limit": 1800,
     "preferences": {...},
     "weight_history": [...],
     "meal_records": [...],
     "planned_feasts": [...]
   }
   ```

### 驗證結果

```bash
# 設定用戶資料
你: 我叫小明，目標體重70公斤，目前75公斤
助手: 已成功更新用戶資料... ✅

# 記錄飲食
你: 幫我記錄午餐，吃了雞胸肉150克和白飯200克
助手: 已記錄午餐，今日已攝取 508 大卡，剩餘 1292 大卡 ✅

# 規劃大餐
你: 週六要吃火鍋，預計1800大卡
助手: 已規劃大餐... ✅
```

### 學習重點

1. **多 Tool 組合** - 一個 Agent 可以有多個工具
2. **Tool 間資料共享** - 透過 JSON 檔案持久化
3. **複雜 Tool 設計**：
   - 參數驗證
   - 預設值處理
   - 錯誤處理
   - 格式化輸出

4. **LLM 決策能力** - 自動選擇合適的工具

### 資料持久化流程

```
用戶說話 → LLM 分析意圖
         → 決定調用哪個 Tool
         → ToolNode 執行 Tool
         → Tool 讀寫 JSON 檔案
         → 返回結果給 LLM
         → LLM 生成最終回覆
```

---

## 步驟 7.5：食譜推薦功能 ✅

### 執行記錄

1. **建立 Prompt 模板** - `src/agent/prompts.py`
   - 定義營養師角色的詳細 System Prompt
   - 包含食譜推薦原則
   - 考慮熱量預算、營養均衡、用戶偏好

2. **更新 Graph** - 使用新的 Prompt

### System Prompt 設計要點

```python
NUTRITIONIST_SYSTEM_PROMPT = """
## 你的職責
1. 協助用戶設定和管理個人資料
2. 根據食材推薦個人化健康食譜
3. 計算營養成分
...

## 食譜推薦原則
1. 熱量預算：根據每日上限計算剩餘額度
2. 營養均衡：蛋白質、碳水、脂肪比例
3. 用戶偏好：飲食類型、喜好、厭惡
4. 過敏原：避開過敏食材
5. 減重建議：高蛋白低脂、糙米替代白飯...
"""
```

### 驗證結果

```bash
你: 我有雞胸肉、花椰菜和糙米，推薦一個低熱量的晚餐

助手: 晚餐食譜：香煎雞胸肉配花椰菜與糙米飯
- 雞胸肉 150g
- 花椰菜 100g  
- 糙米 50g

總熱量：328 大卡
蛋白質：50.8 g
脂肪：6.2 g
碳水化合物：16.5 g

這個搭配高蛋白、低脂肪... ✅
```

### 學習重點

1. **Prompt Engineering 技巧**：
   - 清晰定義角色和職責
   - 列出可用工具及用法
   - 提供推薦原則作為決策依據
   - 定義回覆風格

2. **System Prompt 結構**：
   ```
   1. 角色定義
   2. 職責列表
   3. 可用工具說明
   4. 決策原則
   5. 回覆風格要求
   ```

3. **上下文利用** - LLM 結合：
   - 用戶資料（透過 Tool 獲取）
   - 營養資料（透過 Tool 計算）
   - Prompt 中的推薦原則

---

## 步驟 7.6：大餐日規劃 ✅

### 執行記錄

1. **plan_feast Tool** 已在步驟 7.4 建立，功能：
   - 記錄預定大餐（日期、描述、預估熱量）
   - 計算超出額度
   - 提供調整建議

### 驗證結果

```bash
$ echo "我週六想吃火鍋大餐，大概2000大卡" | poetry run python -m src.main

助手: 已為您規劃週六的火鍋大餐，預估熱量為 2000 大卡。
由於這會超過您每日的熱量上限，建議在大餐前 1 天減少約 200 大卡的攝取。

調整建議：
- 選擇低熱量主食，如糙米飯代替白飯
- 增加蔬菜比例
- 減少油脂攝取
```

### 學習重點

1. **多步推理**
   - LLM 分析大餐熱量 vs 每日上限
   - 計算需要調整的天數和額度
   - 生成具體建議

2. **狀態連動**
   - 大餐規劃儲存到 JSON
   - 後續推薦可讀取此資料
   - 影響未來的推薦策略

---

# Phase 1 完成總結 🎉

## 已完成的功能

| 功能 | 實現方式 | 對應步驟 |
|------|---------|---------|
| 基礎對話 | ChatOllama + LangGraph | 6, 7.1 |
| 多輪記憶 | MemorySaver + thread_id | 7.2 |
| 營養計算 | @tool + bind_tools + ToolNode | 7.3 |
| 用戶資料 | JSON 讀寫 Tools | 7.4 |
| 食譜推薦 | Prompt Engineering | 7.5 |
| 大餐規劃 | plan_feast Tool + 多步推理 | 7.6 |

## 專案結構

```
MyFirstAgent/
├── pyproject.toml          # Poetry 配置
├── poetry.lock
├── README.md
├── tutorial_log.md         # 教學記錄（本檔案）
├── data/
│   └── user_profile.json   # 用戶資料
├── src/
│   ├── __init__.py
│   ├── main.py             # CLI 入口
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py        # LangGraph 定義
│   │   ├── state.py        # State 定義
│   │   └── prompts.py      # Prompt 模板
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── nutrition.py    # 營養計算工具
│   │   └── storage.py      # JSON 讀寫工具
│   └── config/
│       ├── __init__.py
│       └── settings.py     # 配置
└── tests/
    └── __init__.py
```

## Graph 最終結構

```
[START] → [chatbot] → [should_continue?]
                           ↓ "tools" (有 tool_calls)
                      [tools] → [chatbot]
                           ↓ END (無 tool_calls)
                        [END]
         ↑
    [MemorySaver] - 對話記憶
```

## 核心技術學習清單

## Phase 1 總結

Phase 1 已完成以下核心技術學習：

### LangGraph 基礎
- ✅ StateGraph - 狀態圖容器
- ✅ add_node() - 添加節點
- ✅ add_edge() - 連接節點
- ✅ add_conditional_edges() - 條件分支
- ✅ compile() - 編譯圖

### State 管理
- ✅ TypedDict - 定義 State 結構
- ✅ Annotated + Reducer - 定義合併策略
- ✅ add_messages - 訊息歷史 Reducer

### Tool Calling
- ✅ @tool 裝飾器 - 定義工具
- ✅ bind_tools() - 綁定工具到 LLM
- ✅ ToolNode - 自動執行工具
- ✅ tool_calls 判斷 - 條件分支

### 對話記憶
- ✅ MemorySaver - 記憶體 Checkpointer
- ✅ thread_id - 對話線程標識
- ✅ config 參數傳遞

### Prompt Engineering
- ✅ System Prompt 設計
- ✅ 角色定義
- ✅ 工具使用指引
- ✅ 決策原則

---

# Phase 2: Vision - 圖片解析功能

> **目標**：讓 Agent 能夠「看懂」食物圖片和營養標籤，實現原始需求中的「上傳零食圖片計算後果」功能

## Phase 2 步驟規劃

| 步驟 | 步驟名 | 說明的技術 | 預期效果 |
|------|--------|-----------|---------|
| **8** | Vision 模型選型與下載 | 多模態模型比較 | 選定並下載合適的 Vision 模型 |
| **9** | 最小 Vision 對話 | 圖片輸入、多模態 Prompt | CLI 可上傳圖片並得到描述 |
| **10** | Vision Tool：食物識別 | Vision + Tool 整合 | 辨識食物種類和份量 |
| **11** | Vision Tool：營養標籤解析 | OCR + 結構化提取 | 從圖片提取營養數據 |
| **12** | 整合到 Agent | Graph 擴展、條件分支 | Agent 可處理文字或圖片輸入 |
| **13** | 影響分析與建議 | 多步推理、狀態運用 | 分析零食對後續飲食的影響 |

## Step 8：Vision 模型選型與下載（✅ 完成）

### 候選模型

| 模型 | 參數量 | VRAM 需求 | 中文能力 | 推薦度 | 說明 |
|------|--------|-----------|---------|--------|------|
| **moondream** | 1.6B | ~1.7GB | ⭐⭐ | ⭐⭐⭐⭐⭐ | 超小型，適合與文字模型共存 |
| **llava-phi3** | 3.8B | ~3GB | ⭐⭐⭐ | ⭐⭐⭐⭐ | Phi-3 based，平衡 |
| **llava:7b** | 7B | ~5GB | ⭐⭐⭐ | ⭐⭐⭐ | 較大版本，需獨立運行 |
| **llava:13b** | 13B | ~8GB | ⭐⭐⭐ | ⭐⭐ | 通用能力強，VRAM 需求高 |

### 選型決策

**問題發現：** 
- qwen3:8b (5.2GB) + llava:7b (4.7GB) = ~10GB
- RTX 2070 SUPER 只有 8GB VRAM
- 雙模型會觸發 **context switch**（模型交換載入）

**解決方案：選擇 moondream (1.6B, ~1.7GB)**
- qwen3:8b (5.2GB) + moondream (1.7GB) = ~6.9GB ✅
- 可同時載入在 VRAM 中，避免 context switch
- 體積小但保留基本 Vision 能力

**最終選擇：moondream**

### 下載模型

```bash
ollama pull moondream
```

**下載資訊：**
- 模型權重：828 MB
- Vision Encoder：909 MB
- 總大小：約 1.7 GB
- 下載時間：約 1 分鐘（視網速）

### 安裝相關套件

```bash
poetry add ollama pillow
```

**新增依賴：**
- `ollama ^0.6.1` - Ollama Python SDK
- `pillow ^12.0.0` - Python 圖片處理庫

### 驗證測試

**測試 1：基本 Vision 功能**

```python
import ollama
import base64
from PIL import Image, ImageDraw, ImageFont

# 創建營養標籤測試圖片
img = Image.new('RGB', (400, 300), color='white')
d = ImageDraw.Draw(img)
d.text((20, 20), "Nutrition Facts", fill='black')
d.text((20, 70), "Calories: 250 kcal", fill='black')
d.text((20, 110), "Protein: 15g", fill='black')
# ... 更多營養資訊

# 測試 moondream
with open('/tmp/test_nutrition.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

response = ollama.chat(
    model='moondream',
    messages=[{
        'role': 'user',
        'content': 'What do you see in this image?',
        'images': [image_data]
    }]
)
```

**測試結果：**
✅ moondream 能識別圖片為營養標籤
✅ 能理解標籤的結構和佈局
⚠️ **數值提取能力較弱** - 只能描述「有營養資訊」，但無法準確提取具體數字

**測試 2：雙模型 VRAM 共存**

測試 qwen3:8b + moondream 能否同時載入：
```
qwen3:8b    (5.2GB) -> 成功載入 ✅
moondream   (1.7GB) -> 成功載入 ✅
總計：~6.9GB < 8GB VRAM ✅
```

**結論：** moondream 體積小，可與 qwen3:8b 共存，無 context switch！

### 學習重點

1. **VRAM 管理與模型選型**
   - **Context Switch 問題**：當模型總大小超過 VRAM，Ollama 會在模型間切換
   - **解決方案**：選擇小型 Vision 模型（moondream 1.6B）與文字模型（qwen3:8b）共存
   - **計算公式**：總 VRAM 需求 = 文字模型 + Vision 模型 + 推理開銷

2. **多模態模型結構**
   - Vision Encoder：將圖片編碼為向量
   - Language Model：理解與生成文字
   - 兩者融合：圖片理解 + 文字回應

3. **Ollama API 圖片輸入**
   - 圖片需 base64 編碼
   - 通過 `images` 參數傳遞
   - 可同時處理多張圖片

4. **PIL (Pillow) 圖片處理**
   - `Image.new()` - 創建空白圖片
   - `ImageDraw.Draw()` - 繪圖介面
   - `ImageFont` - 字體載入

5. **Vision 模型能力評估 - moondream**
   - ✅ 基本物件識別（能識別營養標籤）
   - ✅ 佈局理解（知道標籤的結構）
   - ✅ 文字存在檢測（知道有文字）
   - ⚠️ **精確 OCR 能力弱**（無法準確提取數值）
   - ⚠️ 中文識別待測試
   - **適用場景**：粗略食物識別、輔助判斷，需搭配 LLM 推理

### moondream 的限制與應對策略

**限制：**
- 無法精確提取營養標籤的數值（250 kcal, 15g 等）
- 更適合「描述圖片內容」而非「精確數據提取」

**應對策略（Phase 2 後續步驟）：**
1. **粗略識別 + 用戶確認**：
   - moondream 識別「這是雞胸肉」
   - Agent 詢問用戶：「請問大約多少克？」
   
2. **結合 Prompt Engineering**：
   - 引導 moondream 關注特定區域
   - 「Find the weight information in this image」
   
3. **未來升級選項**：
   - 若需精確 OCR，考慮專門的 OCR 工具（如 PaddleOCR）
   - 或使用更大的 Vision 模型（當有更多 VRAM 時）

---

### 方案 2 測試：更換小型文字模型

**問題回顧：**
- qwen3:8b 實際 VRAM：~7.5GB（磁碟 5.2GB）
- moondream 實際 VRAM：~4.4GB（磁碟 1.7GB）
- 總計：11.9GB > 8GB VRAM ❌
- **原因**：VRAM 使用量 ≈ 磁碟大小 × 1.5~2 倍（KV cache、激活值等）

**嘗試小型文字模型：**

| 模型 | 磁碟大小 | 預估 VRAM | 中文能力 | Tool Calling |
|------|---------|----------|---------|-------------|
| **gemma3:4b** | 3.3 GB | ~5GB | ⭐⭐⭐⭐ | ✅ 支援 |
| **phi4-mini** | 3.2 GB | ~5GB | ⭐⭐⭐ | ✅ 支援 |
| qwen3:8b | 5.2 GB | ~7.5GB | ⭐⭐⭐⭐⭐ | ✅ 支援 |

**測試結果：gemma3:4b + moondream**

```python
# 測試腳本
1. gemma3:4b 首次對話 -> 0.27秒 ✅
2. moondream Vision   -> 7.20秒 ⚠️ (context switch)
3. gemma3:4b 再次對話 -> 2.67秒 ⚠️ (失去記憶)
```

**結論：**
- ❌ **仍然無法完美共存**
- ❌ **記憶丟失**：gemma3:4b 無法記住之前對話（MemorySaver 可能被影響）
- ⚠️ **切換延遲**：moondream 首次載入需 7 秒

**最終決策：**

保留 **qwen3:8b** + **moondream**，接受 context switch，理由：
1. **保持最佳能力**：qwen3:8b 中文能力最強，適合營養師角色
2. **Vision 非高頻**：圖片分析不是每次對話都需要
3. **真實場景學習**：資源限制是實際開發常見問題
4. **UX 優化空間**：可透過「正在分析圖片...」提示改善體驗
5. **記憶完整性**：保證對話記憶不會因切換而丟失

**Phase 2 實施策略調整：**
- 明確提示用戶 Vision 分析需要等待
- 優化 Vision 調用頻率（非必要不用）
- 未來可考慮：
  - 升級 GPU（更多 VRAM）
  - 使用量化版本
  - 分離 Vision 服務到另一台機器

---

## Step 9：最小 Vision 對話（待實施）

**目標**：驗證 Vision 模型能正確處理圖片輸入

**實現內容**：
- 圖片讀取與編碼（base64）
- 多模態訊息格式
- Vision LLM 調用

**驗證方式**：上傳食物圖片 → 得到描述

---

## Step 10：Vision Tool - 食物識別（待實施）

**目標**：建立 Tool 自動識別食物

**實現內容**：
- `identify_food_from_image` Tool
- 識別食物種類
- 估算份量/重量
- 返回結構化資料

---

## Step 11：Vision Tool - 營養標籤解析（待實施）

**目標**：從圖片提取營養資訊

**實現內容**：
- `parse_nutrition_label` Tool
- OCR 提取文字
- 結構化提取：熱量、蛋白質、脂肪、碳水
- 處理中英文標籤

---

## Step 12：整合到 Agent（待實施）

**目標**：Agent 同時支援文字和圖片輸入

**實現內容**：
- State 擴展（支援圖片）
- Graph 條件分支（判斷輸入類型）
- CLI 圖片上傳介面

---

## Step 13：影響分析與建議（待實施）

**目標**：分析零食對後續飲食的影響

**實現內容**：
- 計算剩餘熱量額度
- 預測對明後天的影響
- 調整推薦策略
- 可視化建議

---

# Phase 3: MCP - 網路搜尋整合（規劃中）

> **目標**：透過 MCP 協議整合外部服務，獲取最新食譜和食材資訊

## Phase 3 步驟規劃（待展開）

- Step 14：MCP 基礎理解
- Step 15：MCP Server 選型與配置
- Step 16：MCP Client 整合到 Agent
- Step 17：食譜搜尋功能
- Step 18：食材價格查詢

---

# Phase 4: 資料可視化（規劃中）

> **目標**：將用戶的體重、飲食數據視覺化

## Phase 4 步驟規劃（待展開）

- Step 19：matplotlib 基礎設置
- Step 20：體重變化折線圖
- Step 21：每日熱量柱狀圖
- Step 22：營養平衡雷達圖
- Step 23：週報告生成

---

# Phase 5: Web UI（規劃中）

> **目標**：建立 Web 介面，提升使用體驗

## Phase 5 步驟規劃（待展開）

- Step 24：FastAPI 後端架設
- Step 25：WebSocket 即時對話
- Step 26：圖片上傳介面
- Step 27：資料視覺化展示
- Step 28：響應式前端設計

---

## 使用方式

```bash
# 啟動 Ollama
ollama serve

# 執行 Agent
cd /home/fatesaikou/testPY/MyFirstAgent
poetry run python -m src.main
```

---

**教學完成日期：2025-12-26**
