"""原题集：向量召回与双路 RRF 召回的同题重排序对照。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import time

from pymilvus import MilvusClient
import yaml

from app.config import COLLECTION_NAME, JINA_API_KEY, MILVUS_URI, RERANK_CANDIDATE_K, RRF_K, TOP_K
from app.RAG.bm25 import BM25Index
from app.RAG.hybrid import hybrid_retriever
from app.RAG.reranker import rerank_with_status
from app.RAG.retriever import retriever
from app.RAG.vector_store import create_embedding_model
from evaluation.evaluate import load_questions, retrieval_metrics, unique_doc_ids


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data/knowledge/starbridge-rag-starter/evaluation/questions.jsonl"
KNOWLEDGE_BASE = QUESTIONS.parent.parent / "knowledge_base"
MODES = ("dense_rerank", "hybrid_rerank")


def doc_ids(hits):
    return unique_doc_ids([{"doc_id": h["entity"].get("doc_id")} for h in hits])


def timed(function, **kwargs):
    started = time.perf_counter()
    result = function(**kwargs)
    return result, time.perf_counter() - started


def validate_candidates(hits, allowed):
    for hit in hits:
        entity = hit["entity"]
        if entity.get("doc_id") not in allowed or not str(entity.get("source", "")).startswith(str(KNOWLEDGE_BASE) + "/"):
            raise RuntimeError("候选包含非内置模拟文档，已停止外部重排")


def summarize(rows, mode):
    scored = [row for row in rows if row["gold_doc_ids"]]
    avg = statistics.fmean
    summary = {"total": len(rows), "retrieval_scored": len(scored)}
    if scored:
        for key, name in (("hit_at_k", "hit_at_5"), ("recall_at_k", "macro_recall_at_5"),
                          ("reciprocal_rank", "mrr"), ("all_gold_found", "all_gold_found_rate")):
            summary[name] = round(avg(float(row[mode]["metrics"][key]) for row in scored), 4)
    if rows:
        summary.update({
            "evidence_coverage": round(avg(bool(row[mode]["chunk_ids"]) for row in rows), 4),
            "avg_total_seconds": round(avg(row[mode]["total_seconds"] for row in rows), 4),
            "rerank_fallback_ids": [row["id"] for row in rows if row[mode]["status"] == "fallback"],
            "empty_scored_ids": [row["id"] for row in scored if not row[mode]["chunk_ids"]],
            "unlabeled_with_evidence_ids": [row["id"] for row in rows if not row["gold_doc_ids"] and row[mode]["chunk_ids"]],
        })
    return summary


def report(metadata, rows):
    successful = [r for r in rows if r["dense_rerank"]["status"] == "ok" and r["hybrid_rerank"]["status"] == "ok"]
    return {
        "metadata": metadata,
        "summary": {mode: summarize(rows, mode) for mode in MODES},
        "both_reranks_succeeded": {mode: summarize(successful, mode) for mode in MODES},
        "candidate_recall": {
            route: round(statistics.fmean(r["candidate_metrics"][route]["recall_at_k"] for r in rows if r["gold_doc_ids"]), 4)
            for route in ("dense", "bm25", "hybrid")
        } if any(r["gold_doc_ids"] for r in rows) else {},
        "observations": {
            "hybrid_recovered_hit_ids": [r["id"] for r in rows if r["gold_doc_ids"] and not r["dense_rerank"]["metrics"]["hit_at_k"] and r["hybrid_rerank"]["metrics"]["hit_at_k"]],
            "note": "通过片段不等于模型回答错误或正确；相关性分数不是事实蕴含或准确率。",
        },
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="内置文档 + 真实 Embedding + 内存精确 COSINE；不代表 Milvus 在线性能")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--delay", type=float, default=.5)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须大于 0")
    if args.delay < 0:
        parser.error("--delay 不能为负数")
    if not JINA_API_KEY or JINA_API_KEY == "replace-me":
        raise RuntimeError("未配置 JINA_API_KEY")
    questions = load_questions(QUESTIONS)
    if args.limit:
        questions = questions[:args.limit]
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output = args.output or ROOT / "evaluation/results" / f"hybrid_compare_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    allowed = set()
    for path in KNOWLEDGE_BASE.rglob("*.md"):
        fields = yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1])
        if fields.get("source") != "synthetic":
            raise RuntimeError("评测仅允许内置模拟知识库")
        allowed.add(fields["doc_id"])
    model = create_embedding_model()
    if args.offline:
        from evaluation.offline_corpus import OfflineCorpusClient
        client = OfflineCorpusClient.from_builtin(model, ROOT / "evaluation/results/hybrid_offline_vectors.npz")
    else:
        client = MilvusClient(uri=MILVUS_URI, db_name="starbridge_rag", timeout=10)
    try:
        started = time.perf_counter()
        index = BM25Index(client, COLLECTION_NAME, refresh_seconds=3600)
        index.refresh()
        metadata = {
            "backend": "offline_exact_cosine" if args.offline else "milvus",
            "run_at": datetime.now().astimezone().isoformat(),
            "question_sha256": hashlib.sha256(QUESTIONS.read_bytes()).hexdigest(),
            "collection_rows": client.get_collection_stats(COLLECTION_NAME)["row_count"],
            "requested_questions": len(questions), "candidate_k": RERANK_CANDIDATE_K,
            "top_k": TOP_K, "rrf_k": RRF_K,
            "bm25": "rank-bm25 0.2.2 BM25Okapi + jieba 0.42.1; title and text; role-specific indexes",
            "index_warmup_seconds": time.perf_counter() - started,
            "multi_query": False,
            "rerank_model": "jina-reranker-v3.5",
            "method": "dense/BM25 queried in parallel, dense candidates reused; identical ordered candidate lists share a Jina request",
            "scope": "read-only retrieval, no answer generation; no-label questions observed separately, not counted as retrieval failures",
        }
        rows = []
        for number, question in enumerate(questions, 1):
            role, query = question.get("role", "customer"), question["question"]
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as executor:
                dense_job = executor.submit(timed, retriever, client=client, collection_name=COLLECTION_NAME,
                                            query=query, embedding_model=model, limit=RERANK_CANDIDATE_K, role=role)
                keyword_job = executor.submit(timed, index.search, query=query, limit=RERANK_CANDIDATE_K, role=role)
                dense, dense_seconds = dense_job.result()
                keyword, keyword_seconds = keyword_job.result()
            dense = [h for h in dense if role in h["entity"].get("audience", [])]
            hybrid = hybrid_retriever(client=client, collection_name=COLLECTION_NAME, query=query,
                                      embedding_model=model, keyword_index=index, limit=RERANK_CANDIDATE_K,
                                      role=role, dense_hits=dense, keyword_hits=keyword)
            hybrid_seconds = time.perf_counter() - started
            validate_candidates(dense, allowed)
            validate_candidates(hybrid, allowed)
            row = {
                "id": question["id"], "question": query, "role": role, "category": question["category"],
                "expected_behavior": question["expected_behavior"], "gold_doc_ids": question["gold_doc_ids"],
                "dense_retrieval_seconds": dense_seconds, "bm25_retrieval_seconds": keyword_seconds,
                "hybrid_retrieval_seconds": hybrid_seconds,
                "candidate_metrics": {name: retrieval_metrics(question["gold_doc_ids"], doc_ids(hits), RERANK_CANDIDATE_K)
                                      for name, hits in (("dense", dense), ("bm25", keyword), ("hybrid", hybrid))},
                "candidate_ids": {"dense": [h["id"] for h in dense], "bm25": [h["id"] for h in keyword], "hybrid": [h["id"] for h in hybrid]},
            }
            identical = row["candidate_ids"]["dense"] == row["candidate_ids"]["hybrid"]
            row["shared_rerank_request"] = identical
            ranked_results = {}
            modes = ("dense_rerank", "hybrid_rerank") if number % 2 else ("hybrid_rerank", "dense_rerank")
            for mode in modes:
                if identical and ranked_results:
                    result, seconds = next(iter(ranked_results.values()))
                else:
                    result, seconds = timed(rerank_with_status, query=query,
                                            hits=dense if mode == "dense_rerank" else hybrid,
                                            top_k=TOP_K)
                ranked_results[mode] = (result, seconds)
                hits = result.hits
                row[mode] = {
                    "doc_ids": doc_ids(hits), "chunk_ids": [h["id"] for h in hits],
                    "rerank_scores": [h.get("rerank_score") for h in hits],
                    "metrics": retrieval_metrics(question["gold_doc_ids"], doc_ids(hits), TOP_K),
                    "status": result.status, "error_type": result.error_type, "rerank_attempts": result.attempts,
                    "total_seconds": seconds + (dense_seconds if mode == "dense_rerank" else hybrid_seconds),
                }
            rows.append(row)
            output.write_text(json.dumps(report(metadata, rows), ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{number}/{len(questions)}] {row['id']} dense={row['dense_rerank']['metrics']['hit_at_k']} "
                  f"hybrid={row['hybrid_rerank']['metrics']['hit_at_k']} "
                  f"status={row['hybrid_rerank']['status']}", flush=True)
            if args.delay:
                time.sleep(args.delay)
        print("REPORT", output, flush=True)
        print("SUMMARY", json.dumps({k: v for k, v in report(metadata, rows).items() if k != "results"}, ensure_ascii=False), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
