# Meal Planning AI Agent 🍽️

一個使用 LangGraph + Ollama 打造的個人化食譜推薦 AI Agent。

## 功能

- 📊 追蹤體重變化與飲食記錄
- 🥗 根據家中食材推薦午餐/晚餐
- 🧮 自動計算營養與熱量
- 🎉 支援大餐日規劃與調整

## 環境需求

- Python 3.12+
- Ollama（本地運行）
- 模型：qwen3:8b

## 安裝

```bash
# 安裝依賴
poetry install

# 下載 Ollama 模型
ollama pull qwen3:8b
```

## 使用

```bash
# 啟動 Agent
poetry run python -m src.main
```

## 專案結構

```
MyFirstAgent/
├── src/
│   ├── main.py          # CLI 入口
│   ├── agent/           # LangGraph Agent 定義
│   ├── tools/           # Agent 可調用的工具
│   └── config/          # 設定管理
├── data/                # 用戶資料 (JSON)
└── tests/               # 測試
```

## 學習記錄

詳見 [tutorial_log.md](tutorial_log.md)
