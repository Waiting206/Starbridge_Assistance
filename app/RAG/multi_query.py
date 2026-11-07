"""复用单路检索，按片段主键去重并用 RRF 合并多个查询的排名。"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import time

from app.RAG.fusion import fuse_rankings
from app.RAG.query_expansion import QueryExpansion, expand_queries
from app.RAG.retriever import retriever
from app.security import AccessRole


logger = logging.getLogger(__name__)


@dataclass
class MultiQueryResult:
    hits: list[dict]
    expansion: QueryExpansion
    seconds: float
    successful_extra_queries: int = 0
    failed_queries: list[dict] = field(default_factory=list)


def retrieve_multi_query(
    *,
    client,
    collection_name: str,
    query: str,
    embedding_model,
    original_hits: list[dict],
    limit: int,
    role: AccessRole = "customer",
    query_model=None,
    retrieval_fn=None,
) -> MultiQueryResult:
    """原问题已经检索一次；扩展查询最多并发两路，最终仍只保留 limit 条。"""
    started = time.perf_counter()
    retrieval_fn = retrieval_fn or retriever
    authorized = lambda hits: [
        hit for hit in hits if role in hit["entity"].get("audience", [])
    ]
    original_hits = authorized(original_hits)
    expansion = expand_queries(query, model=query_model)
    result = MultiQueryResult(original_hits[:limit], expansion, 0.0)
    alternatives = expansion.queries[1:]
    rankings = [original_hits]
    if alternatives:
        with ThreadPoolExecutor(max_workers=2) as executor:
            jobs = [
                (
                    alternative,
                    executor.submit(
                        retrieval_fn,
                        client=client,
                        collection_name=collection_name,
                        query=alternative,
                        embedding_model=embedding_model,
                        limit=limit,
                        role=role,  # 角色始终由服务端传入，扩展模型不能改变它。
                    ),
                )
                for alternative in alternatives
            ]
            for alternative, future in jobs:
                try:
                    rankings.append(authorized(future.result()))
                    result.successful_extra_queries += 1
                except Exception as exc:
                    logger.warning("扩展查询检索失败（%s），保留其他检索结果", type(exc).__name__)
                    result.failed_queries.append({"query": alternative, "error_type": type(exc).__name__})
        if result.successful_extra_queries:
            result.hits = fuse_rankings(rankings, limit)
    result.seconds = time.perf_counter() - started
    return result
