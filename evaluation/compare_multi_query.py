"""在原题集上比较单 Query / 多 Query；只读 Milvus，不生成问答答案。"""

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import time

from pymilvus import MilvusClient
import yaml

from app.config import COLLECTION_NAME, JINA_API_KEY, MILVUS_URI, MULTI_QUERY_MODEL, RERANK_CANDIDATE_K, TOP_K
from app.RAG.multi_query import retrieve_multi_query
from app.RAG.query_expansion import create_query_model
from app.RAG.reranker import rerank
from app.RAG.retriever import retriever
from app.RAG.vector_store import create_embedding_model
from evaluation.evaluate import load_questions, retrieval_metrics, unique_doc_ids


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE = ROOT / "data/knowledge/starbridge-rag-starter/knowledge_base"
QUESTIONS = ROOT / "data/knowledge/starbridge-rag-starter/evaluation/questions.jsonl"


def doc_ids(hits):
    return unique_doc_ids([{"doc_id": hit["entity"].get("doc_id")} for hit in hits])


def validate_candidates(hits, allowed_doc_ids):
    for hit in hits:
        entity = hit["entity"]
        if (
            entity.get("doc_id") not in allowed_doc_ids
            or not str(entity.get("source", "")).startswith(str(KNOWLEDGE_BASE) + "/")
        ):
            raise RuntimeError("非内置模拟文档候选，已停止，避免向外部重排序服务发送正文")


def summarize(rows, mode):
    if not rows:
        return {"total": 0}
    metrics = [row[mode]["metrics"] for row in rows]
    candidate_metrics = [row[mode]["candidate_metrics"] for row in rows]
    avg = statistics.fmean
    return {
        "total": len(rows),
        f"hit_at_{TOP_K}": round(avg(float(m["hit_at_k"]) for m in metrics), 4),
        f"macro_recall_at_{TOP_K}": round(avg(m["recall_at_k"] for m in metrics), 4),
        "mrr": round(avg(m["reciprocal_rank"] for m in metrics), 4),
        "all_gold_found_rate": round(avg(float(m["all_gold_found"]) for m in metrics), 4),
        "candidate_macro_recall": round(avg(m["recall_at_k"] for m in candidate_metrics), 4),
        "candidate_all_gold_found_rate": round(avg(float(m["all_gold_found"]) for m in candidate_metrics), 4),
        "avg_total_seconds": round(avg(row[mode]["total_seconds"] for row in rows), 4),
        "avg_rerank_seconds": round(avg(row[mode]["rerank_seconds"] for row in rows), 4),
        "rerank_applied_count": sum(row[mode]["rerank_applied"] for row in rows),
        "rerank_fallback_ids": [row["id"] for row in rows if not row[mode]["rerank_applied"]],
    }


def build_report(metadata, rows, failures):
    both_succeeded = [row for row in rows if all(row[mode]["rerank_applied"] for mode in ("single_query", "multi_query"))]
    changes = {}
    for metric in ("hit_at_k", "recall_at_k", "all_gold_found", "reciprocal_rank"):
        changes[metric] = {
            "improved_ids": [r["id"] for r in rows if r["multi_query"]["metrics"][metric] > r["single_query"]["metrics"][metric]],
            "regressed_ids": [r["id"] for r in rows if r["multi_query"]["metrics"][metric] < r["single_query"]["metrics"][metric]],
        }
    return {
        "metadata": metadata,
        "single_query": summarize(rows, "single_query"),
        "multi_query": summarize(rows, "multi_query"),
        "both_reranks_succeeded": {
            "single_query": summarize(both_succeeded, "single_query"),
            "multi_query": summarize(both_succeeded, "multi_query"),
        },
        "expansion": {
            "expanded_count": sum(r["expansion"]["status"] == "expanded" for r in rows),
            "no_expansion_count": sum(r["expansion"]["status"] == "no_expansion" for r in rows),
            "fallback_ids": [r["id"] for r in rows if r["expansion"]["status"] not in {"expanded", "no_expansion"}],
            "extra_query_failure_ids": [r["id"] for r in rows if r["failed_queries"]],
            "avg_generation_seconds": round(statistics.fmean(r["expansion"]["seconds"] for r in rows), 4) if rows else 0,
            "input_tokens": sum(r["expansion"]["input_tokens"] for r in rows),
            "output_tokens": sum(r["expansion"]["output_tokens"] for r in rows),
            "extra_retrieval_count": sum(len(r["expansion"]["queries"]) - 1 for r in rows),
            "changed_candidate_set_count": sum(set(r["single_query"]["candidate_chunk_ids"]) != set(r["multi_query"]["candidate_chunk_ids"]) for r in rows),
        },
        "changes": changes,
        "failures": failures,
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须大于 0")
    if args.delay < 0:
        parser.error("--delay 不能小于 0")
    if not JINA_API_KEY or JINA_API_KEY == "replace-me":
        raise RuntimeError("未配置 JINA_API_KEY")
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output = args.output or ROOT / "evaluation/results" / f"multi_query_compare_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    all_questions = load_questions(QUESTIONS)
    questions = [q for q in all_questions if q["gold_doc_ids"]]
    if args.limit:
        questions = questions[:args.limit]
    allowed_doc_ids = set()
    for path in KNOWLEDGE_BASE.rglob("*.md"):
        fields = yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1])
        if fields.get("source") != "synthetic":
            raise RuntimeError("评测只允许使用内置模拟知识库")
        allowed_doc_ids.add(fields["doc_id"])

    # 只连接现有数据库和集合，不创建、不清空、不重新向量化。
    client = MilvusClient(uri=MILVUS_URI, db_name="starbridge_rag")
    try:
        metadata = {
            "run_at": datetime.now().astimezone().isoformat(),
            "questions_file": str(QUESTIONS),
            "question_sha256": hashlib.sha256(QUESTIONS.read_bytes()).hexdigest(),
            "question_total": len(all_questions),
            "requested_scored_questions": len(questions),
            "excluded_without_gold_ids": [q["id"] for q in all_questions if not q["gold_doc_ids"]],
            "collection": COLLECTION_NAME,
            "collection_row_count": client.get_collection_stats(COLLECTION_NAME)["row_count"],
            "candidate_k": RERANK_CANDIDATE_K,
            "top_k": TOP_K,
            "expansion_model": MULTI_QUERY_MODEL,
            "max_extra_queries": 2,
            "fusion": "RRF(k=60), deduplicate by chunk id, cap candidates before Jina",
            "rerank_model": "jina-reranker-v3.5",
            "rerank_order": "alternates by question to reduce ordering bias",
            "scope": "retrieval only, no answer generation or database writes",
        }
        embedding_model = create_embedding_model()
        query_model = create_query_model()
        rows, failures = [], []
        for index, question in enumerate(questions, 1):
            role = question.get("role", "customer")
            started = time.perf_counter()
            try:
                original_hits = retriever(
                    client=client, collection_name=COLLECTION_NAME, query=question["question"],
                    embedding_model=embedding_model, limit=RERANK_CANDIDATE_K, role=role,
                )
            except Exception as exc:
                failures.append({"id": question["id"], "stage": "original_retrieval", "error_type": type(exc).__name__})
                output.write_text(json.dumps(build_report(metadata, rows, failures), ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[{index}/{len(questions)}] {question['id']} retrieval_failed={type(exc).__name__}", flush=True)
                continue
            retrieval_seconds = time.perf_counter() - started
            original_hits = [h for h in original_hits if role in h["entity"].get("audience", [])]
            multi = retrieve_multi_query(
                client=client, collection_name=COLLECTION_NAME, query=question["question"],
                embedding_model=embedding_model, original_hits=original_hits,
                limit=RERANK_CANDIDATE_K, role=role, query_model=query_model,
            )
            validate_candidates(original_hits, allowed_doc_ids)
            validate_candidates(multi.hits, allowed_doc_ids)
            row = {
                "id": question["id"], "question": question["question"],
                "category": question["category"], "role": role,
                "gold_doc_ids": question["gold_doc_ids"],
                "original_retrieval_seconds": retrieval_seconds,
                "expansion": asdict(multi.expansion),
                "expansion_and_extra_retrieval_seconds": multi.seconds,
                "failed_queries": multi.failed_queries,
            }
            candidates = {"single_query": original_hits, "multi_query": multi.hits}
            modes = ("single_query", "multi_query") if index % 2 else ("multi_query", "single_query")
            for mode in modes:
                hits = candidates[mode]
                t0 = time.perf_counter()
                ranked = rerank(question["question"], hits, top_k=TOP_K)
                rerank_seconds = time.perf_counter() - t0
                row[mode] = {
                    "candidate_chunk_ids": [h["id"] for h in hits],
                    "candidate_doc_ids": doc_ids(hits),
                    "candidate_metrics": retrieval_metrics(question["gold_doc_ids"], doc_ids(hits), RERANK_CANDIDATE_K),
                    "doc_ids": doc_ids(ranked),
                    "metrics": retrieval_metrics(question["gold_doc_ids"], doc_ids(ranked), TOP_K),
                    "rerank_applied": bool(ranked and "rerank_score" in ranked[0]),
                    "rerank_seconds": rerank_seconds,
                    "total_seconds": retrieval_seconds + rerank_seconds + (multi.seconds if mode == "multi_query" else 0),
                }
            rows.append(row)
            output.write_text(json.dumps(build_report(metadata, rows, failures), ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{index}/{len(questions)}] {question['id']} queries={len(multi.expansion.queries)} expansion={multi.expansion.status} "
                  f"recall={row['single_query']['metrics']['recall_at_k']:.2f}->{row['multi_query']['metrics']['recall_at_k']:.2f} "
                  f"rerank={row['single_query']['rerank_applied']}/{row['multi_query']['rerank_applied']}", flush=True)
            if args.delay:
                time.sleep(args.delay)
        report = build_report(metadata, rows, failures)
        print("REPORT", output, flush=True)
        print("SUMMARY", json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False), flush=True)
        if failures:
            raise RuntimeError(f"{len(failures)} 道题原始检索失败，报告中已记录，不能视为完整评测")
    finally:
        client.close()


if __name__ == "__main__":
    main()
