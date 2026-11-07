"""星桥协作 RAG 的 FastAPI 应用入口。"""
"""
完整顺序是：

  1. 执行 uvicorn app.main:app --reload
  2. Uvicorn 导入 app/main.py
  3. Python 创建 FastAPI 对象：

     app = FastAPI(...)

  4. Uvicorn 准备启动 HTTP 服务
  5. Uvicorn 通知 FastAPI：“服务要启动了”
  6. FastAPI 调用 lifespan()
  7. 执行 lifespan 中 yield 前面的代码
  8. FastAPI 开始监听 8000 端口，接收请求

"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.RAG.Loader import load_documents
from app.RAG.spiltter import document_spitter
from app.RAG.vector_store import create_embedding_model, index_documents, init_milvus
from app.RAG.bm25 import BM25Index
from app.config import BM25_ENABLED, COLLECTION_NAME
from app.agent.init_agent import init_agent
from app.agent.tools import build_search_tool
from app.api.chat import router as chat_router
from app.api.knowledge import router as knowledge_router
from app.api.threads import router as threads_router



@asynccontextmanager
async def lifespan(app: FastAPI):
    """在服务启动时创建共享资源，在关闭时释放它们。"""
    client = init_milvus()
    embedding_model = create_embedding_model()
    keyword_index = BM25Index(client, COLLECTION_NAME) if BM25_ENABLED else None
    if keyword_index is not None:
        try:
            keyword_index.refresh()
        except Exception as exc:
            logging.getLogger(__name__).warning("BM25 初始化失败（%s），查询时重试", type(exc).__name__)
    search_tool = build_search_tool(client, embedding_model, keyword_index)

    with init_agent(tools=[search_tool]) as (agent,checkpointer):
        app.state.agent = agent
        app.state.milvus_client = client
        app.state.embedding_model = embedding_model
        app.state.bm25_index = keyword_index
        app.state.checkpointer = checkpointer
        yield

    client.close()


app = FastAPI(
    title="星桥协作知识助手 API",
    version="0.1.0",
    lifespan=lifespan,
)

# 前端本地开发地址；真实部署时应替换为实际前端域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(threads_router)
app.include_router(knowledge_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}



