"""向量 + BM25 双路召回，任一路不可用时继续使用另一条路。"""

from concurrent.futures import ThreadPoolExecutor
import logging

from app.RAG.fusion import fuse_rankings
from app.RAG.retriever import retriever


logger = logging.getLogger(__name__)


def hybrid_retriever(
    *, client, collection_name, query, embedding_model, keyword_index,
    limit=20, role="customer", dense_hits=None, keyword_hits=None,
):
    rankings = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = []
        if dense_hits is None:
            jobs.append(("dense", executor.submit(
                retriever, client=client, collection_name=collection_name, query=query,
                embedding_model=embedding_model, limit=limit, role=role,
            )))
        else:
            rankings.append([{**hit, "retrieval_routes": ["dense"]} for hit in dense_hits if role in hit["entity"].get("audience", [])])
        if keyword_hits is None:
            jobs.append(("bm25", executor.submit(keyword_index.search, query=query, limit=limit, role=role)))
        else:
            rankings.append([{**hit, "retrieval_routes": ["bm25"]} for hit in keyword_hits if role in hit["entity"].get("audience", [])])
        for route, future in jobs:
            try:
                hits = future.result()
                rankings.append([
                    {**hit, "retrieval_routes": [route]}
                    for hit in hits if role in hit["entity"].get("audience", [])
                ])
            except Exception as exc:
                logger.warning("%s 召回不可用（%s）", route, type(exc).__name__)
    if not rankings:
        raise RuntimeError("知识库召回服务暂不可用")
    return fuse_rankings(rankings, limit)
