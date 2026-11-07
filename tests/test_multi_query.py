from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from app.RAG.multi_query import fuse_rankings, retrieve_multi_query
from app.RAG.query_expansion import expand_queries
from app.agent.context import AgentContext
from app.RAG.reranker import RerankResult


class FakeQueryModel:
    def __init__(self, content):
        self.content = content

    def invoke(self, messages):
        return AIMessage(self.content)


def hit(chunk_id, audience=None):
    return {"id": chunk_id, "distance": 0.9, "entity": {
        "doc_id": f"P-{chunk_id}", "text": "测试知识", "source": "test.md",
        "audience": audience if audience is not None else ["customer", "support"],
    }}


def test_expansion_always_keeps_original_and_removes_duplicates():
    result = expand_queries("原问题", FakeQueryModel('{"queries":["原问题","查询一","查询一","查询二","查询三"]}'))
    assert result.queries == ["原问题", "查询一", "查询二"]
    assert result.status == "expanded"


@pytest.mark.parametrize("content", ["不是JSON", '{"queries":"错误类型"}', '{"queries":[1]}'])
def test_malformed_expansion_falls_back_to_original(content):
    result = expand_queries("原问题", FakeQueryModel(content))
    assert result.queries == ["原问题"]
    assert result.status == "invalid_response"


def test_expansion_timeout_falls_back_without_raising():
    class TimeoutModel:
        def invoke(self, messages):
            raise TimeoutError("do not expose provider response")
    result = expand_queries("原问题", TimeoutModel())
    assert result.queries == ["原问题"]
    assert result.status == "request_failed"


def test_rrf_deduplicates_by_chunk_and_keeps_candidate_budget():
    rankings = [[hit(1), hit(2)], [hit(2), hit(3)]]
    fused = fuse_rankings(rankings, limit=2)
    assert [h["id"] for h in fused] == [2, 1]
    assert "fusion_score" not in rankings[0][0]


def test_every_expanded_query_uses_same_role_and_filters_unauthorized_hits(monkeypatch):
    calls = []
    def search(**kwargs):
        calls.append(kwargs)
        return [hit(2), hit(99, ["support"])]
    monkeypatch.setattr("app.RAG.multi_query.retriever", search)
    result = retrieve_multi_query(
        client=object(), collection_name="kb", query="原问题", embedding_model=object(),
        original_hits=[hit(1), hit(88, ["support"])], limit=2, role="customer",
        query_model=FakeQueryModel('{"queries":["查询一","查询二"]}'),
    )
    assert len(calls) == 2
    assert all(c["role"] == "customer" for c in calls)
    assert all("customer" in h["entity"]["audience"] for h in result.hits)
    assert {h["id"] for h in result.hits} == {1, 2}


def test_all_extra_retrieval_failures_keep_original_order(monkeypatch):
    def fail(**kwargs):
        raise TimeoutError()
    monkeypatch.setattr("app.RAG.multi_query.retriever", fail)
    original = [hit(1), hit(2)]
    result = retrieve_multi_query(
        client=object(), collection_name="kb", query="原问题", embedding_model=object(),
        original_hits=original, limit=2, query_model=FakeQueryModel('{"queries":["查询一"]}'),
    )
    assert result.hits == original
    assert result.successful_extra_queries == 0
    assert len(result.failed_queries) == 1


def test_tool_reranks_with_original_query_and_preserves_source_structure(monkeypatch):
    from app.agent import tools
    observed = {}
    monkeypatch.setattr(tools, "BM25_ENABLED", False)
    monkeypatch.setattr(tools, "MULTI_QUERY_ENABLED", True)
    monkeypatch.setattr(tools, "RERANK_ENABLED", True)
    monkeypatch.setattr(tools, "retriever", lambda **kwargs: [hit(1 if kwargs["query"] == "原问题" else 2)])
    monkeypatch.setattr("app.RAG.query_expansion.create_query_model", lambda: FakeQueryModel('{"queries":["查询一"]}'))
    monkeypatch.setattr("app.RAG.multi_query.retriever", lambda **kwargs: [hit(2)])
    def rank(query, hits, **kwargs):
        observed.update(query=query, hits=hits)
        return RerankResult([{**hits[0], "rerank_score": 0.9}], "ok")
    monkeypatch.setattr(tools, "rerank_with_status", rank)
    tool = tools.build_search_tool(object(), object())
    content, sources = tool.func("原问题", SimpleNamespace(context=AgentContext(role="customer")))
    assert observed["query"] == "原问题"
    assert {h["id"] for h in observed["hits"]} == {1, 2}
    assert sources[0]["chunk_id"].startswith("chunk_")
    assert "文档编号" in content


def test_partial_extra_retrieval_failure_keeps_successful_results(monkeypatch):
    def search(**kwargs):
        if kwargs["query"] == "失败查询":
            raise TimeoutError()
        return [hit(2)]
    monkeypatch.setattr("app.RAG.multi_query.retriever", search)
    result = retrieve_multi_query(
        client=object(), collection_name="kb", query="原问题", embedding_model=object(),
        original_hits=[hit(1)], limit=2,
        query_model=FakeQueryModel('{"queries":["失败查询","成功查询"]}'),
    )
    assert {h["id"] for h in result.hits} == {1, 2}
    assert result.successful_extra_queries == 1
    assert len(result.failed_queries) == 1


def test_disabled_feature_does_not_generate_queries(monkeypatch):
    from app.agent import tools
    monkeypatch.setattr(tools, "BM25_ENABLED", False)
    monkeypatch.setattr(tools, "MULTI_QUERY_ENABLED", False)
    monkeypatch.setattr(tools, "RERANK_ENABLED", True)
    monkeypatch.setattr(tools, "rerank_with_status", lambda query, hits, **kwargs: RerankResult([{**hits[0], "rerank_score": 0.9}], "ok"))
    monkeypatch.setattr(tools, "retriever", lambda **kwargs: [hit(1)])
    def forbidden(**kwargs):
        raise AssertionError("Disabled feature must not call expansion")
    monkeypatch.setattr(tools, "retrieve_multi_query", forbidden)
    tool = tools.build_search_tool(object(), object())
    _, sources = tool.func("原问题", SimpleNamespace(context=AgentContext(role="customer")))
    assert len(sources) == 1
