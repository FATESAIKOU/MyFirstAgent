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

## 步驟 4：展開功能列表與實現大綱（進行中）

