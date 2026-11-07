import ast
from copy import deepcopy
import re
from types import SimpleNamespace

import pytest
import requests

from app.RAG.bm25 import BM25Index, tokenize
from app.RAG.hybrid import hybrid_retriever
from app.RAG.reranker import RerankResult, rerank_with_status
from app.agent.context import AgentContext


def record(chunk_id, text, audience=None):
    return {"id": chunk_id, "text": text, "doc_id": f"P-{chunk_id}", "title": "测试",
            "source": "test.md", "audience": audience or ["customer", "support"]}


def hit(chunk_id, score=None, audience=None):
    result = {"id": chunk_id, "entity": record(chunk_id, "参考资料", audience)}
    if score is not None:
        result["rerank_score"] = score
    return result


class FakeIterator:
    def __init__(self, rows):
        self.rows = deepcopy(rows)

    def next(self):
        batch, self.rows = self.rows, []
        return batch

    def close(self):
        pass


class FakeMilvus:
    def __init__(self, rows):
        self.rows = rows
        self.filters = []

    def query_iterator(self, **kwargs):
        return FakeIterator(self.rows)

    def query(self, **kwargs):
        expression = kwargs["filter"]
        self.filters.append(expression)
        ids = ast.literal_eval(re.search(r"id in (\[.*?\])", expression).group(1))
        role = re.search(r'JSON_CONTAINS\(audience, "(\w+)"\)', expression).group(1)
        return [deepcopy(row) for row in self.rows if row["id"] in ids and role in row["audience"]]


def test_tokenizer_preserves_identifiers_and_normalizes_case():
    tokens = tokenize("企业版 ＳＳＯ CSV E-401 task_id v2.1 10000")
    assert {"sso", "csv", "e-401", "task_id", "v2.1", "10000"}.issubset(tokens)


def test_keyword_only_matching_returns_no_unrelated_documents():
    client = FakeMilvus([record(1, "支持 SSO 登录"), record(2, "CSV 导出"), record(3, "成员邀请")])
    results = BM25Index(client, "kb").search("SSO", 5, "customer")
    assert [h["id"] for h in results] == [1]
    assert 'JSON_CONTAINS(audience, "customer")' in client.filters[0]


def test_single_document_keyword_index_is_searchable():
    index = BM25Index(FakeMilvus([record(1, "SSO")]), "kb")
    assert [h["id"] for h in index.search("SSO", 5, "customer")] == [1]


def test_keyword_permissions_are_checked_before_topk_and_again_live():
    client = FakeMilvus([record(1, "内部SSO流程", ["support"]), record(2, "SSO 帮助")])
    index = BM25Index(client, "kb")
    assert [h["id"] for h in index.search("SSO", 1, "customer")] == [2]
    client.rows[1]["audience"] = ["support"]  # 缓存未过期，但权限已被撤销。
    assert index.search("SSO", 1, "customer") == []


def test_upload_invalidation_makes_new_text_searchable():
    client = FakeMilvus([record(1, "SSO")])
    index = BM25Index(client, "kb", refresh_seconds=3600)
    index.refresh()
    client.rows.append(record(2, "E-401"))
    assert index.search("E-401", 5, "customer") == []
    index.invalidate()
    assert [h["id"] for h in index.search("E-401", 5, "customer")] == [2]


def test_ttl_refresh_observes_other_worker_writes():
    client = FakeMilvus([record(1, "SSO")])
    index = BM25Index(client, "kb", refresh_seconds=0)
    index.refresh()
    client.rows.append(record(2, "E-401"))
    assert [h["id"] for h in index.search("E-401", 5, "customer")] == [2]


def test_hybrid_rrf_deduplicates_and_combines_routes():
    keyword = SimpleNamespace(search=lambda **kwargs: [hit(2), hit(3)])
    results = hybrid_retriever(client=None, collection_name="kb", query="q", embedding_model=None,
                               keyword_index=keyword, dense_hits=[hit(1), hit(2)], limit=2)
    assert [h["id"] for h in results] == [2, 1]
    assert set(results[0]["retrieval_routes"]) == {"dense", "bm25"}


def test_keyword_failure_keeps_vector_candidates():
    def fail(**kwargs):
        raise TimeoutError()
    results = hybrid_retriever(client=None, collection_name="kb", query="q", embedding_model=None,
                               keyword_index=SimpleNamespace(search=fail), dense_hits=[hit(1)], limit=5)
    assert [h["id"] for h in results] == [1]


def test_vector_failure_keeps_keyword_candidates(monkeypatch):
    def fail(**kwargs):
        raise TimeoutError()
    monkeypatch.setattr("app.RAG.hybrid.retriever", fail)
    results = hybrid_retriever(client=None, collection_name="kb", query="q", embedding_model=None,
                               keyword_index=SimpleNamespace(search=lambda **kwargs: [hit(2)]), limit=5)
    assert [h["id"] for h in results] == [2]


def test_both_route_failures_raise(monkeypatch):
    def fail(**kwargs):
        raise TimeoutError()
    monkeypatch.setattr("app.RAG.hybrid.retriever", fail)
    with pytest.raises(RuntimeError):
        hybrid_retriever(client=None, collection_name="kb", query="q", embedding_model=None,
                          keyword_index=SimpleNamespace(search=fail), limit=5)


def test_fusion_filters_wrong_role_returned_by_a_backend():
    results = hybrid_retriever(client=None, collection_name="kb", query="q", embedding_model=None,
                               keyword_index=SimpleNamespace(search=lambda **kwargs: [hit(2, audience=["support"])]),
                               dense_hits=[hit(1), hit(3, audience=["support"])], limit=5)
    assert [h["id"] for h in results] == [1]


def test_reranker_sorts_finite_scores_including_negative_values(monkeypatch):
    monkeypatch.setattr("app.RAG.reranker.JINA_API_KEY", "test-only")
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
        "results": [{"index": 0, "relevance_score": -.02}, {"index": 1, "relevance_score": .69}]
    })
    monkeypatch.setattr("app.RAG.reranker.requests.post", lambda *args, **kwargs: response)
    result = rerank_with_status("q", [hit(1), hit(2)])
    assert result.status == "ok" and [h["id"] for h in result.hits] == [2, 1]
    assert result.hits[1]["rerank_score"] == -.02


def test_timeout_retries_then_preserves_candidates_without_fabricated_scores(monkeypatch):
    monkeypatch.setattr("app.RAG.reranker.JINA_API_KEY", "test-only")
    monkeypatch.setattr("app.RAG.reranker.RERANK_MAX_RETRIES", 1)
    monkeypatch.setattr("app.RAG.reranker.time.sleep", lambda seconds: None)
    def fail(*args, **kwargs):
        raise requests.Timeout("provider response must stay private")
    monkeypatch.setattr("app.RAG.reranker.requests.post", fail)
    original = [hit(1), hit(2)]
    result = rerank_with_status("q", original, top_k=1)
    assert result.hits == original[:1] and result.status == "fallback"
    assert result.attempts == 2 and "rerank_score" not in result.hits[0]


def test_missing_key_preserves_candidate_order(monkeypatch):
    monkeypatch.setattr("app.RAG.reranker.JINA_API_KEY", None)
    result = rerank_with_status("q", [hit(1)])
    assert result.status == "fallback" and result.error_type == "MissingAPIKey"
    assert [h["id"] for h in result.hits] == [1]


@pytest.mark.parametrize("item", [
    {"index": True, "relevance_score": .9},
    {"index": 100, "relevance_score": .9},
    {"index": 0, "relevance_score": float("nan")},
    {"index": 0, "relevance_score": "0.9"},
])
def test_malformed_provider_results_use_original_candidates(monkeypatch, item):
    monkeypatch.setattr("app.RAG.reranker.JINA_API_KEY", "test-only")
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"results": [item]})
    monkeypatch.setattr("app.RAG.reranker.requests.post", lambda *args, **kwargs: response)
    result = rerank_with_status("q", [hit(1)])
    assert result.status == "fallback" and "rerank_score" not in result.hits[0]


def test_tool_rechecks_role_before_building_context(monkeypatch):
    from app.agent import tools
    monkeypatch.setattr(tools, "BM25_ENABLED", False)
    monkeypatch.setattr(tools, "MULTI_QUERY_ENABLED", False)
    monkeypatch.setattr(tools, "RERANK_ENABLED", True)
    monkeypatch.setattr(tools, "retriever", lambda **kwargs: [hit(1)])
    monkeypatch.setattr(tools, "rerank_with_status", lambda *args, **kwargs: RerankResult(
        [hit(2, .9, ["support"]), hit(1, .2)], "ok"))
    tool = tools.build_search_tool(object(), object())
    content, sources = tool.func("q", SimpleNamespace(context=AgentContext()))
    assert len(sources) == 1 and sources[0]["score"] == .2
    assert sources[0]["score_type"] == "jina_relevance"
    assert "P-2" not in content


def test_transient_jina_failure_recovers_on_retry(monkeypatch):
    monkeypatch.setattr("app.RAG.reranker.JINA_API_KEY", "test-only")
    monkeypatch.setattr("app.RAG.reranker.RERANK_MAX_RETRIES", 1)
    monkeypatch.setattr("app.RAG.reranker.time.sleep", lambda seconds: None)
    calls = []
    def post(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.Timeout()
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"results": [{"index": 0, "relevance_score": .69}]})
    monkeypatch.setattr("app.RAG.reranker.requests.post", post)
    result = rerank_with_status("q", [hit(1)])
    assert len(calls) == 2 and result.attempts == 2
    assert result.status == "ok" and result.hits[0]["rerank_score"] == .69


def test_disabled_reranker_returns_authorized_rrf_sources(monkeypatch):
    from app.agent import tools
    monkeypatch.setattr(tools, "BM25_ENABLED", True)
    monkeypatch.setattr(tools, "MULTI_QUERY_ENABLED", False)
    monkeypatch.setattr(tools, "RERANK_ENABLED", False)
    monkeypatch.setattr(tools, "hybrid_retriever", lambda **kwargs: [{**hit(1), "fusion_score": .02}])
    tool = tools.build_search_tool(object(), object(), keyword_index=object())
    _, sources = tool.func("q", SimpleNamespace(context=AgentContext()))
    assert len(sources) == 1 and sources[0]["score_type"] == "rrf"
    assert sources[0]["score"] == .02


def test_upload_completion_invalidates_keyword_cache(monkeypatch, tmp_path):
    from app.api import knowledge
    events = []
    monkeypatch.setattr(knowledge, "load_uploaded_file", lambda *args: [object()])
    monkeypatch.setattr(knowledge, "document_spitter", lambda docs: docs)
    monkeypatch.setattr(knowledge, "index_documents", lambda *args: events.append("indexed"))
    state = SimpleNamespace(
        milvus_client=SimpleNamespace(flush=lambda name: events.append("flushed")), embedding_model=object(),
        bm25_index=SimpleNamespace(invalidate=lambda: events.append("invalidated")),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    assert knowledge._index_uploaded_file(request, tmp_path/"test.md", "test.md", "ID", "HASH") == 1
    assert events == ["indexed", "flushed", "invalidated"]


def test_offline_adapter_uses_cosine_and_keeps_role_filter():
    from evaluation.offline_corpus import OfflineCorpusClient
    records = [record(1, "a"), record(2, "b"), record(3, "private", ["support"])]
    client = OfflineCorpusClient(records, [[1, 0], [100, 100], [1, 0]])
    found = client.search(data=[[1, 0]], limit=3, filter='JSON_CONTAINS(audience, "customer")', output_fields=["doc_id"])[0]
    assert [h["id"] for h in found] == [1, 2]
    assert found[0]["distance"] == pytest.approx(1)


