"""Chroma 向量存储封装。

解决什么问题
    知识库需要三种能力：写入（建库）、语义检索（给 HybridRetriever 的向量路）、
    全量导出（给 BM25 建索引）。Chroma 的原生 API 在这三件事上都能用，
    但返回结构是"三个平行列表"（ids / documents / distances），
    直接用会让调用方到处写 zip 和索引对齐，且距离到相似度的换算规则
    会散落各处。本模块把它收敛成统一的 RetrievalResult。

核心设计决策
    1. embedding 模型由本模块自己加载并显式传给 Chroma，不用 Chroma 的
       默认 embedding function。默认函数用的是 all-MiniLM-L6-v2（纯英文），
       对中文金融语料几乎不可用。这是一个静默失效的坑：不会报错，
       只是检索结果全是噪音。
    2. 模型采用**进程级懒加载 + 缓存**。SentenceTransformer 加载要 3~10 秒
       并占几百 MB 内存；FastAPI 每个请求各加载一次会直接 OOM。
       同时懒加载保证不用 RAG 的场景（纯财务数据问题）不付这个启动成本。
    3. 距离统一换算成 [0,1] 的相似度分数。Chroma 用余弦距离
       （0 最相似），而 BM25 是分数越大越相关，两者不归一就没法比较，
       RRF 融合也会给出反向的排序。
    4. collection 的 metadata 里固定 hnsw:space="cosine"。Chroma 默认是 L2，
       而 sentence-transformers 的输出向量是为余弦相似度优化的，
       用 L2 检索质量会明显下降。

为什么不用其他方案
    - 不用 FAISS：FAISS 不带元数据存储，还要自己维护 id → 文档的映射
       和持久化，等于重写半个向量库。
    - 不用 pgvector / Milvus：需要额外的服务进程，与"docker-compose 一键起、
       本地也能跑"的部署目标冲突。Chroma 的本地持久化模式刚好合适。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Sequence

from config import get_config
from harness.types import Document, RetrievalResult

logger = logging.getLogger(__name__)

# chromadb 0.6.x 与新版 posthog 的接口不兼容，每次操作都会刷几行
# "Failed to send telemetry event ... capture() takes 1 positional argument but 3 were given"。
# 已经在 Settings 里关了 anonymized_telemetry，但它仍会初始化 posthog 并尝试上报。
# 这些是 ERROR 级别的日志，不静音会淹没真正的错误——可观测性的价值在于
# "有 ERROR 就是真出事了"，一旦掺进常态噪音，这条信号就废了。
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# 进程级 embedding 模型缓存：{model_name: SentenceTransformer}
_EMBEDDING_MODELS: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def get_embedding_model(model_name: str | None = None) -> Any:
    """加载并缓存 SentenceTransformer。线程安全（双重检查锁）。"""
    name = model_name or get_config().embedding_model
    if name in _EMBEDDING_MODELS:
        return _EMBEDDING_MODELS[name]

    with _MODEL_LOCK:
        if name in _EMBEDDING_MODELS:
            return _EMBEDDING_MODELS[name]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "未安装 sentence-transformers，无法使用向量检索。"
                "请执行 pip install -r requirements.txt"
            ) from exc

        logger.info("加载 embedding 模型: %s（首次运行需要下载约 120MB）", name)
        model = SentenceTransformer(name)
        _EMBEDDING_MODELS[name] = model
        return model


def embed_texts(texts: Sequence[str], model_name: str | None = None) -> list[list[float]]:
    """批量编码文本。归一化后可直接用点积当余弦相似度。"""
    model = get_embedding_model(model_name)
    vectors = model.encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 50,
        convert_to_numpy=True,
    )
    return [vector.tolist() for vector in vectors]


class RagStore:
    """Chroma 向量存储。"""

    def __init__(
        self,
        collection_name: str | None = None,
        persist_dir: str | Path | None = None,
        embedding_model: str | None = None,
    ):
        config = get_config()
        self.collection_name = collection_name or config.chroma_collection_name
        self.persist_dir = Path(persist_dir) if persist_dir else config.chroma_dir
        self.embedding_model_name = embedding_model or config.embedding_model
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client: Any = None
        self._collection: Any = None

    # ---------------- 连接 ----------------

    @property
    def collection(self) -> Any:
        """懒加载 collection。第一次访问时才连 Chroma 并加载模型。"""
        if self._collection is None:
            self._connect()
        return self._collection

    def _connect(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise RuntimeError(
                "未安装 chromadb，无法使用知识库。请执行 pip install -r requirements.txt"
            ) from exc

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            # 必须显式指定余弦空间，Chroma 默认 L2 与本项目的归一化向量不匹配
            metadata={"hnsw:space": "cosine", "description": "金融研报知识库"},
        )
        logger.info(
            "Chroma 已连接: collection=%s dir=%s count=%d",
            self.collection_name,
            self.persist_dir,
            self._collection.count(),
        )

    # ---------------- 写入 ----------------

    def add(self, documents: Sequence[Document], batch_size: int = 64) -> int:
        """写入文档。使用 upsert，重复建库不会产生重复条目。"""
        if not documents:
            return 0

        total = 0
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            embeddings = embed_texts(
                [doc.content for doc in batch], model_name=self.embedding_model_name
            )
            self.collection.upsert(
                ids=[doc.doc_id for doc in batch],
                documents=[doc.content for doc in batch],
                embeddings=embeddings,
                metadatas=[self._sanitize_metadata(doc.metadata) for doc in batch],
            )
            total += len(batch)
            logger.debug("已写入 %d/%d 个 chunk", total, len(documents))

        logger.info("知识库写入完成，共 %d 个 chunk", total)
        return total

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Chroma 的元数据只接受 str/int/float/bool，其他类型要转换。

        空 dict 也不被接受（部分版本会报错），所以至少塞一个占位字段。
        """
        cleaned: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, (list, tuple, set)):
                cleaned[key] = ", ".join(str(item) for item in value)
            else:
                cleaned[key] = str(value)
        return cleaned or {"_": "1"}

    # ---------------- 检索 ----------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """向量检索。返回按相似度降序的结果。"""
        limit = top_k or get_config().vector_top_k
        available = self.count()
        if available == 0:
            logger.warning("知识库为空，向量检索返回空结果。请先执行 python -m rag.prepare_data")
            return []

        query_embedding = embed_texts([query], model_name=self.embedding_model_name)[0]
        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, available),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (raw.get("ids") or [[]])[0]
        contents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: list[RetrievalResult] = []
        for rank, doc_id in enumerate(ids):
            distance = distances[rank] if rank < len(distances) else 1.0
            results.append(
                RetrievalResult(
                    doc_id=doc_id,
                    content=contents[rank] if rank < len(contents) else "",
                    metadata=dict(metadatas[rank] or {}) if rank < len(metadatas) else {},
                    score=self._distance_to_similarity(distance),
                    retrieval_method="vector",
                    ranks={"vector": rank + 1},
                )
            )
        return results

    @staticmethod
    def _distance_to_similarity(distance: float | None) -> float:
        """余弦距离 → [0,1] 相似度。

        Chroma 的余弦距离定义为 1 - cos_sim，值域 [0, 2]。
        直接用 1 - distance 在向量夹角大于 90° 时会得到负分，
        RRF 之前的归一化会因此出问题，所以做 clamp。
        """
        if distance is None:
            return 0.0
        similarity = 1.0 - float(distance)
        return max(0.0, min(1.0, similarity))

    # ---------------- 全量导出与维护 ----------------

    def get_all_documents(self, batch_size: int = 500) -> list[Document]:
        """导出全部文档，给 BM25 建索引用。

        分批取而不是一次 get()：知识库大了以后一次性取全量会占大量内存，
        而 BM25 建索引本身就要再存一份分词结果。
        """
        total = self.count()
        if total == 0:
            return []

        documents: list[Document] = []
        for offset in range(0, total, batch_size):
            raw = self.collection.get(
                limit=batch_size, offset=offset, include=["documents", "metadatas"]
            )
            ids = raw.get("ids") or []
            contents = raw.get("documents") or []
            metadatas = raw.get("metadatas") or []
            for index, doc_id in enumerate(ids):
                documents.append(
                    Document(
                        doc_id=doc_id,
                        content=contents[index] if index < len(contents) else "",
                        metadata=dict(metadatas[index] or {}) if index < len(metadatas) else {},
                    )
                )
        return documents

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception as exc:  # noqa: BLE001
            logger.error("读取知识库条目数失败: %s", exc)
            return 0

    def reset(self) -> None:
        """清空 collection。重建知识库时使用。"""
        if self._client is None:
            self._connect()
        try:
            self._client.delete_collection(self.collection_name)
        except Exception as exc:  # noqa: BLE001 — collection 不存在时也算成功
            logger.debug("删除 collection 时忽略异常: %s", exc)
        self._collection = None
        self._connect()
        logger.info("知识库已清空: %s", self.collection_name)

    def health(self) -> dict[str, Any]:
        try:
            return {
                "status": "ok",
                "collection": self.collection_name,
                "document_count": self.count(),
                "persist_dir": str(self.persist_dir),
                "embedding_model": self.embedding_model_name,
            }
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


__all__ = ["RagStore", "embed_texts", "get_embedding_model"]
