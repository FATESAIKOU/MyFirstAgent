"""配置設定"""

# Ollama 設定
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:8b"

# 專案路徑
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
USER_PROFILE_PATH = os.path.join(DATA_DIR, "user_profile.json")
