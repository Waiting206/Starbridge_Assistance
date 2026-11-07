"""从 Milvus 正文构建可重建的 BM25 内存索引，Milvus 仍是唯一数据源。"""

from dataclasses import dataclass
import logging
import math
import re
from threading import RLock
import time
import unicodedata

import jieba
from rank_bm25 import BM25Okapi

from app.config import BM25_REFRESH_SECONDS
from app.RAG.retriever import OUTPUT_FIELDS, audience_filter_for_role


logger = logging.getLogger(__name__)
jieba.setLogLevel(logging.WARNING)
TOKEN_BLOCKS = re.compile(r"[a-z0-9]+(?:[_.:/-][a-z0-9]+)*|[\u3400-\u9fff]+")
STOP_WORDS = {"的", "了", "吗", "呢", "是", "和", "与", "及", "请问", "如何", "什么", "是否", "the", "a", "an"}


def tokenize(text: str) -> list[str]:
    """中英文采用一致分词；保留 SSO、CSV、版本号及带连字符的错误码。"""
    text = unicodedata.normalize("NFKC", text).lower()
    tokens = []
    for block in TOKEN_BLOCKS.findall(text):
        words = jieba.lcut(block, HMM=False) if "\u3400" <= block[0] <= "\u9fff" else [block]
        tokens.extend(word for word in words if word.strip() and word not in STOP_WORDS)
    return tokens


@dataclass(frozen=True)
class RoleIndex:
    records: tuple
    token_sets: tuple
    model: BM25Okapi


class BM25Index:
    def __init__(self, client, collection_name: str, refresh_seconds: float = BM25_REFRESH_SECONDS):
        self.client = client
        self.collection_name = collection_name
        self.refresh_seconds = refresh_seconds
        self._lock = RLock()
        self._roles = {}
        self._built_at = 0.0
        self._dirty = True

    def invalidate(self):
        """上传完成后使本进程索引失效；其他进程通过 TTL 自动刷新。"""
        with self._lock:
            self._dirty = True

    def refresh(self, force=False):
        with self._lock:
            if not force and not self._dirty and time.monotonic() - self._built_at < self.refresh_seconds:
                return
            iterator = self.client.query_iterator(
                collection_name=self.collection_name,
                filter="",
                output_fields=["id", *OUTPUT_FIELDS],
                batch_size=512,
                timeout=10,
                consistency_level="Strong",
            )
            records = []
            try:
                while batch := iterator.next():
                    records.extend(batch)
            finally:
                iterator.close()
            tokenized = []
            for record in records:
                if not isinstance(record.get("audience"), list) or not isinstance(record.get("text"), str):
                    continue
                tokens = tokenize(f"{record.get('title', '')}\n{record['text']}")
                if tokens:
                    tokenized.append((record, tokens))
            roles = {}
            for role in ("customer", "support"):
                allowed = [(record, tokens) for record, tokens in tokenized if role in record["audience"]]
                if allowed:
                    corpus = [tokens for _, tokens in allowed]
                    roles[role] = RoleIndex(
                        tuple(record for record, _ in allowed),
                        tuple(frozenset(tokens) for tokens in corpus),
                        BM25Okapi(corpus),
                    )
            self._roles = roles  # 仅在完整构建成功后替换快照。
            self._dirty = False
            self._built_at = time.monotonic()

    def search(self, query: str, limit: int, role: str) -> list[dict]:
        tokens = tokenize(query)
        if not tokens or limit < 1:
            return []
        self.refresh()
        with self._lock:
            snapshot = self._roles.get(role)
        if snapshot is None:
            return []
        scores = snapshot.model.get_scores(tokens)
        query_tokens = set(tokens)
        matched = [
            index for index, score in enumerate(scores)
            if math.isfinite(float(score)) and query_tokens.intersection(snapshot.token_sets[index])
        ]
        # 极小语料的 Okapi 分数可能为 0 或负数；是否命中依据词项交集，不以 0 为门槛。
        matched.sort(key=lambda index: (-float(scores[index]), int(snapshot.records[index]["id"])))
        selected = matched[:limit]
        if not selected:
            return []
        ids = [int(snapshot.records[index]["id"]) for index in selected]
        # 再读当前元数据：缓存尚未过期时也不能返回已撤销权限或已删除的片段。
        current = self.client.query(
            collection_name=self.collection_name,
            filter=f"id in {ids} and {audience_filter_for_role(role)}",
            output_fields=["id", *OUTPUT_FIELDS],
            limit=len(ids),
            timeout=10,
            consistency_level="Strong",
        )
        current_by_id = {int(record["id"]): record for record in current}
        hits = []
        for index in selected:
            chunk_id = int(snapshot.records[index]["id"])
            record = current_by_id.get(chunk_id)
            if record is not None and role in record.get("audience", []):
                hits.append({
                    "id": chunk_id,
                    "entity": {key: value for key, value in record.items() if key != "id"},
                    "bm25_score": float(scores[index]),
                    "retrieval_routes": ["bm25"],
                })
        return hits
