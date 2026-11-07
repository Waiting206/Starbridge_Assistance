"""Jina 重排序；服务暂不可用时保留召回候选的顺序。"""

from dataclasses import dataclass
import logging
import math
import time

import requests

from app.config import JINA_API_KEY, JINA_RERANK_URL, RERANK_MAX_RETRIES


logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    hits: list[dict]
    status: str
    error_type: str | None = None
    attempts: int = 0


def rerank_with_status(query: str, hits: list[dict], top_k: int = 5) -> RerankResult:
    if not hits:
        return RerankResult([], "empty")
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    fallback = hits[:top_k]
    if not JINA_API_KEY or JINA_API_KEY == "replace-me":
        logger.warning("未配置 Jina 密钥，使用候选原排序")
        return RerankResult(fallback, "fallback", error_type="MissingAPIKey")
    attempts = 0
    try:
        for attempt in range(RERANK_MAX_RETRIES + 1):
            attempts += 1
            try:
                response = requests.post(
                    JINA_RERANK_URL,
                    json={"model": "jina-reranker-v3.5", "query": query,
                          "documents": [hit["entity"]["text"] for hit in hits],
                          "top_n": min(top_k, len(hits)), "return_documents": False},
                    headers={"Authorization": f"Bearer {JINA_API_KEY}"},
                    timeout=(3, 15),
                )
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", 0) or 0
                retryable = isinstance(exc, (requests.Timeout, requests.ConnectionError)) or status_code == 429 or status_code >= 500
                if not retryable or attempt == RERANK_MAX_RETRIES:
                    raise
                time.sleep(.25)
        results = response.json()["results"]
        if not isinstance(results, list) or not results:
            raise ValueError("Jina 未返回有效结果")
        ranked, seen = [], set()
        for item in results:
            index, score = item["index"], item["relevance_score"]
            if (not isinstance(index, int) or isinstance(index, bool)
                or not 0 <= index < len(hits) or index in seen
                or not isinstance(score, (int, float)) or isinstance(score, bool)
                or not math.isfinite(score)):
                raise ValueError("Jina 返回的下标或分数不合法")
            seen.add(index)
            ranked.append({**hits[index], "rerank_score": float(score)})
        ranked.sort(key=lambda hit: -hit["rerank_score"])
        return RerankResult(ranked[:top_k], "ok", attempts=attempts)
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        logger.warning("Jina 重排序不可用（%s），使用候选原排序", type(exc).__name__)
        return RerankResult(fallback, "fallback", error_type=type(exc).__name__, attempts=attempts)


def rerank(query: str, hits: list[dict], top_k: int = 5) -> list[dict]:
    return rerank_with_status(query, hits, top_k).hits
