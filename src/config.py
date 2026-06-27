# ========================= src/config.py =========================
import os
from dotenv import load_dotenv


APP_TITLE = "HSC AI Tutoring Centre"


# Base paths
ACCOUNTS_DB = os.path.join("server", "users.json")
USERS_ROOT = os.path.join("users")


# Load .env for API keys
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl")