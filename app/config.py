import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge" / "starbridge-rag-starter" / "knowledge_base"

load_dotenv(PROJECT_ROOT / ".env")

POSTGRES_URI = os.getenv("DB_URL")
MILVUS_URI = os.getenv("MILVUS_URI")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
SUPPORT_ACCESS_TOKEN = os.getenv("SUPPORT_ACCESS_TOKEN")
TOP_K = 5
COLLECTION_NAME = "knowledge_base"
JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_RERANK_URL = os.getenv("JINA_RERANK_URL", "https://api.jina.ai/v1/rerank")
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
RERANK_CANDIDATE_K = max(TOP_K, int(os.getenv("RERANK_CANDIDATE_K", "20")))

# 多 Query 先作为可选能力，通过对照评测后再决定是否默认启用。
MULTI_QUERY_ENABLED = os.getenv("MULTI_QUERY_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
MULTI_QUERY_MODEL = os.getenv("MULTI_QUERY_MODEL", "deepseek-v4-flash")
MULTI_QUERY_TIMEOUT = float(os.getenv("MULTI_QUERY_TIMEOUT", "10"))

# BM25 与向量双路召回，使用 RRF 融合排名。
BM25_ENABLED = os.getenv("BM25_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
BM25_REFRESH_SECONDS = float(os.getenv("BM25_REFRESH_SECONDS", "30"))
RRF_K = int(os.getenv("RRF_K", "60"))
if BM25_REFRESH_SECONDS < 0 or RRF_K < 1:
    raise ValueError("BM25_REFRESH_SECONDS 必须非负、RRF_K 必须为正")

RERANK_MAX_RETRIES = int(os.getenv("RERANK_MAX_RETRIES", "1"))
if not 0 <= RERANK_MAX_RETRIES <= 2:
    raise ValueError("RERANK_MAX_RETRIES 必须在 0～2 之间")
