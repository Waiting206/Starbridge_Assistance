"""复用单路检索，按片段主键去重并用 RRF 合并多个查询的排名。"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import time

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


def fuse_rankings(rankings: list[list[dict]], limit: int, k: int = 60) -> list[dict]:
    """RRF 只融合排名，避免直接相加不同查询的相似度分数。"""
    hits_by_id = {}
    scores = {}
    for ranking in rankings:
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            chunk_id = hit["id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            hits_by_id.setdefault(chunk_id, hit)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    # 相同分数时沿用首次出现顺序，优先保留原问题的排序。
    ordered = sorted(hits_by_id, key=lambda chunk_id: -scores[chunk_id])
    return [
        {**hits_by_id[chunk_id], "fusion_score": scores[chunk_id]}
        for chunk_id in ordered[:limit]
    ]


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
) -> MultiQueryResult:
    """原问题已经检索一次；扩展查询最多并发两路，最终仍只保留 limit 条。"""
    started = time.perf_counter()
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
                        retriever,
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
