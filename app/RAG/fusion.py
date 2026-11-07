"""按排名融合召回结果，不混加不同检索器的原始分数。"""

from app.config import RRF_K


def fuse_rankings(rankings: list[list[dict]], limit: int, k: int = RRF_K) -> list[dict]:
    hits_by_id, scores = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            chunk_id = hit["id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            if chunk_id not in hits_by_id:
                hits_by_id[chunk_id] = dict(hit)
            else:
                existing = hits_by_id[chunk_id]
                routes = dict.fromkeys(existing.get("retrieval_routes", []) + hit.get("retrieval_routes", []))
                if routes:
                    existing["retrieval_routes"] = list(routes)
                if "bm25_score" in hit:
                    existing["bm25_score"] = hit["bm25_score"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    # 同分时稳定保留首次出现顺序，不修改输入对象。
    ordered = sorted(hits_by_id, key=lambda chunk_id: -scores[chunk_id])
    return [{**hits_by_id[chunk_id], "fusion_score": scores[chunk_id]} for chunk_id in ordered[:limit]]
