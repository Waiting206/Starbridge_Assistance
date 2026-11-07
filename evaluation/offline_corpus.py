"""仅供离线评测：真实 Embedding + 精确 COSINE，不代表 Milvus 在线性能。"""

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from app.RAG.Loader import load_documents
from app.RAG.spiltter import document_spitter
from app.RAG.vector_store import create_chunk_id


class RecordIterator:
    def __init__(self, rows, batch_size):
        self.rows, self.batch_size, self.offset = rows, batch_size, 0

    def next(self):
        batch = self.rows[self.offset:self.offset + self.batch_size]
        self.offset += self.batch_size
        return batch

    def close(self):
        pass


class OfflineCorpusClient:
    def __init__(self, records, vectors):
        self.records = records
        vectors = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if len(records) != len(vectors) or not np.isfinite(vectors).all() or (norms == 0).any():
            raise ValueError("离线向量数据不合法")
        self.vectors = vectors / norms

    @classmethod
    def from_builtin(cls, embedding_model, cache_path: Path):
        chunks = document_spitter(load_documents())
        if not chunks or any(chunk.metadata.get("source_type") != "synthetic" for chunk in chunks):
            raise RuntimeError("离线评测只允许内置模拟文档")
        records = [{**chunk.metadata, "id": create_chunk_id(chunk), "text": chunk.page_content} for chunk in chunks]
        digest = hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as cached:
                if str(cached["sha256"]) == digest:
                    return cls(records, cached["vectors"])
        vectors = embedding_model.embed_documents([chunk.page_content for chunk in chunks])
        client = cls(records, vectors)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, vectors=np.asarray(vectors), sha256=digest)
        return client

    @staticmethod
    def _allowed(record, expression):
        role_match = re.search(r'JSON_CONTAINS\(audience,\s*"(customer|support)"\)', expression)
        if role_match and role_match.group(1) not in record.get("audience", []):
            return False
        ids_match = re.search(r"id in (\[[0-9, ]*\])", expression)
        if ids_match and record["id"] not in json.loads(ids_match.group(1)):
            return False
        return True

    @staticmethod
    def _fields(record, output_fields):
        return {key: record[key] for key in output_fields if key in record}

    def query_iterator(self, *, output_fields, batch_size, filter="", **kwargs):
        return RecordIterator([self._fields(r, output_fields) for r in self.records if self._allowed(r, filter)], batch_size)

    def query(self, *, output_fields, filter="", limit=16384, **kwargs):
        return [self._fields(r, output_fields) for r in self.records if self._allowed(r, filter)][:limit]

    def search(self, *, data, limit, filter, output_fields, **kwargs):
        if len(data) != 1:
            raise ValueError("离线适配器只接受单个查询")
        query = np.asarray(data[0], dtype=np.float64)
        norm = np.linalg.norm(query)
        if not np.isfinite(query).all() or norm == 0:
            raise ValueError("查询向量不合法")
        scores = self.vectors @ (query / norm)
        indices = [i for i, r in enumerate(self.records) if self._allowed(r, filter)]
        indices.sort(key=lambda i: (-float(scores[i]), self.records[i]["id"]))
        return [[{"id": self.records[i]["id"], "distance": float(scores[i]),
                  "entity": self._fields(self.records[i], output_fields)} for i in indices[:limit]]]

    def get_collection_stats(self, collection_name):
        return {"row_count": len(self.records)}

    def close(self):
        pass
