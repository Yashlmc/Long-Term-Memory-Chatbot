import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openrouter/free")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")
SESSIONS_DIR = os.getenv("SESSIONS_DIR", "data/sessions")
DEFAULT_USER_ID = os.getenv("CHATBOT_USER_ID", "default_user")
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
