import os
from pathlib import Path
from langchain_openai import ChatOpenAI

try:
    from dotenv import load_dotenv
    # 尝试加载项目根目录的 .env 文件
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    print("Warning: 'python-dotenv' is not installed. Environment variables from .env will not be automatically loaded.")

# ==========================================
# 系统全局配置
# ==========================================

# 获取工作目录并确保其为绝对路径 (所有 agent 能够操作的文件均被限制在此目录下)
WORKSPACE_DIR = Path(os.getenv("MOOS_WORKSPACE", "./workspace")).resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# 限制上下文保留的最大消息数，防止 Token 溢出
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))


# ==========================================
# 大模型 (LLM) 配置
# ==========================================

LLM_MODEL = os.getenv("OPENAI_API_MODEL")
LLM_API_BASE = os.getenv("OPENAI_API_BASE")
LLM_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

if not LLM_API_KEY:
    raise ValueError("环境变量 OPENAI_API_KEY 未设置！请检查 .env 文件。")

def get_llm():
    """获取大模型实例，集中管理配置信息"""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        base_url=LLM_API_BASE,
        api_key=LLM_API_KEY,
        max_retries=LLM_MAX_RETRIES
    )
