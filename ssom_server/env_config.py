import os
from dotenv import load_dotenv

load_dotenv()

def getenv(key, default=None):
    v = os.environ.get(key)
    return v if v not in (None, "") else default

# 전역 환경변수
QDRANT_HOST = getenv("QDRANT_HOST")
QDRANT_PORT = int(getenv("QDRANT_PORT", "6333"))
OPENAI_API_KEY = getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(getenv("LLM_TEMPERATURE", "0.2"))
RETRIEVER_TOP_K = int(getenv("RETRIEVER_TOP_K", "3"))
COLLECTION_NAME = getenv("COLLECTION_NAME", "java-files")

# 값 검증 (필요시)
if not QDRANT_HOST:
    raise ValueError("QDRANT_HOST 환경변수가 필요합니다")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 환경변수가 필요합니다")
