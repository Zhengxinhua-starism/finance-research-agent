"""混合检索：BM25 + 向量双路召回，RRF 融合。

解决什么问题
    单路向量检索在金融语料上有两个明确的短板：
    (1) 对精确词不敏感——查"商誉减值"时，语义上"资产减值准备"也很接近，
        向量检索会把它排在前面，但用户要的是商誉那一条；
    (2) 对代码和专有名词几乎无效——"002594"这种 token 在 embedding 空间里
        没有语义，查它等于随机召回。
    BM25 刚好补上这两点，但它又完全不懂"利润质量"和"现金流覆盖净利润"
    是同一件事。两路互补，用 RRF 融合。

核心设计决策
    1. 融合用 **RRF（Reciprocal Rank Fusion）** 而不是分数加权。
       分数加权（alpha * vector_score + (1-alpha) * bm25_score）的致命问题是
       两路分数不同分布：BM25 分数无上界且随语料规模变化，余弦相似度在 [0,1]，
       归一化方式一变，alpha 就要重调。RRF 只用**名次**不用分数，
       天然免疫分布差异，这也是它在 IR 领域成为默认融合方法的原因。
       公式：score(d) = Σ 1 / (k + rank_i(d))，k=60。
    2. k=60 是 Cormack 等人原始论文的经验值，含义是"名次在 60 以内的差异
       才显著影响得分"。k 越小越强调头部名次，越大越平均。这里保留为可配参数。
    3. 中文 BM25 需要分词。优先用 jieba（若已安装），否则退回
       "2-gram + 英文数字整词"的混合切分。2-gram 对中文 BM25 的效果
       接近分词，且零依赖——比强行按单字切好很多（单字切会让"率"这种
       高频字主导 IDF）。
    4. BM25 索引在内存里构建，随 Retriever 实例生命周期存在，
       并提供 refresh()。知识库是预填充的、更新极少，
       没必要引入 Elasticsearch 这类外部服务。

为什么保留 alpha 参数
    md 的接口签名里有 alpha。RRF 本身不需要它，但两路的**可信度**确实可能
    不对等（例如知识库很小的时候 BM25 更可靠）。这里把 alpha 实现为
    对两路 RRF 贡献的加权，默认 0.5 时退化为标准 RRF，不改变默认行为。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

from config import get_config
from harness.tracing import NullTracer, Tracer
from harness.types import Document, RetrievalResult
from rag.rag_store import RagStore

logger = logging.getLogger(__name__)

# 中文停用词（精简版）。BM25 对高频虚词很敏感，不去掉会让"的""是"主导匹配。
CHINESE_STOPWORDS = {
    "的", "了", "在", "是", "和", "与", "及", "或", "对", "为", "以", "于", "从",
    "到", "由", "被", "把", "个", "之", "其", "该", "这", "那", "有", "无", "不",
    "也", "都", "很", "更", "最", "会", "能", "可", "要", "就", "而", "但", "并",
    "等", "中", "上", "下", "内", "外", "前", "后", "时", "年", "月", "日",
}

TOKEN_PATTERN = re.compile(r"[a-zA-Z]+|\d+\.?\d*|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """中英混合分词。

    jieba 可用时走 jieba（精度更高），否则用 2-gram 近似：
    中文串切成相邻二字组合，英文和数字保持整词。
    """
    if not text:
        return []

    tokens: list[str] = []
    for segment in TOKEN_PATTERN.findall(text.lower()):
        if segment.isascii():
            tokens.append(segment)
            continue
        tokens.extend(_tokenize_chinese(segment))
    return [token for token in tokens if token not in CHINESE_STOPWORDS]


def _tokenize_chinese(segment: str) -> list[str]:
    jieba_module = _load_jieba()
    if jieba_module is not None:
        return [word for word in jieba_module.lcut(segment) if word.strip()]

    if len(segment) == 1:
        return [segment]
    # 2-gram：既保留单字（覆盖单字查询），也生成二元组（提升区分度）
    grams = [segment[i : i + 2] for i in range(len(segment) - 1)]
    return grams + list(segment)


_JIEBA_CACHE: list[Any] = []


def _load_jieba() -> Any:
    if _JIEBA_CACHE:
        return _JIEBA_CACHE[0]
    try:
        import jieba

        jieba.setLogLevel(logging.WARNING)
        module: Any = jieba
    except ImportError:
        logger.info("未安装 jieba，BM25 使用 2-gram 分词（效果略低但无需额外依赖）")
        module = None
    _JIEBA_CACHE.append(module)
    return module


def reciprocal_rank_fusion(
    results_list: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """RRF 融合。

    参数
        results_list: 多路检索结果，每路是一个按相关性降序排列的文档 ID 列表。
        k: 平滑常数，默认 60。
        weights: 各路权重，默认等权。

    返回
        {doc_id: rrf_score}，分数越高越相关。
    """
    if weights is None:
        weights = [1.0] * len(results_list)
    if len(weights) != len(results_list):
        raise ValueError("weights 长度必须与 results_list 一致")

    scores: dict[str, float] = {}
    for route_index, results in enumerate(results_list):
        weight = weights[route_index]
        for rank, doc_id in enumerate(results, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return scores


class BM25Index:
    """内存 BM25 索引。"""

    def __init__(self, documents: Sequence[Document]):
        self.documents: list[Document] = list(documents)
        self.doc_ids: list[str] = [doc.doc_id for doc in self.documents]
        self._bm25: Any = None
        if self.documents:
            self._build()

    def _build(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "未安装 rank-bm25，无法使用关键词检索。请执行 pip install -r requirements.txt"
            ) from exc

        corpus = [tokenize(doc.content) for doc in self.documents]
        # 全空语料会让 BM25Okapi 在计算 avgdl 时除零
        if not any(corpus):
            logger.warning("BM25 语料分词后为空，索引未构建")
            return
        self._bm25 = BM25Okapi(corpus)
        logger.info("BM25 索引构建完成，文档数: %d", len(self.documents))

    @property
    def ready(self) -> bool:
        return self._bm25 is not None

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if not self.ready:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]

        max_score = max((score for _, score in ranked), default=0.0)
        results: list[RetrievalResult] = []
        for rank, (index, score) in enumerate(ranked, start=1):
            if score <= 0:
                # BM25 分数为 0 表示没有任何查询词命中，这类结果是纯噪音
                continue
            document = self.documents[index]
            results.append(
                RetrievalResult(
                    doc_id=document.doc_id,
                    content=document.content,
                    metadata=document.metadata,
                    # 归一化到 [0,1] 仅为可读性，RRF 融合不使用这个分数
                    score=score / max_score if max_score > 0 else 0.0,
                    retrieval_method="bm25",
                    ranks={"bm25": rank},
                )
            )
        return results


class HybridRetriever:
    """BM25 + 向量双路检索 + RRF 融合。"""

    def __init__(
        self,
        vector_store: RagStore | None = None,
        documents: Sequence[Document] | None = None,
        bm25_index: BM25Index | None = None,
        tracer: Tracer | None = None,
    ):
        self.config = get_config()
        self.vector_store = vector_store or RagStore()
        self.tracer = tracer or NullTracer()
        self._bm25_index = bm25_index
        self._documents: list[Document] | None = list(documents) if documents else None

    @property
    def bm25(self) -> BM25Index:
        """懒加载 BM25 索引：首次检索时才从 Chroma 拉全量文档建索引。"""
        if self._bm25_index is None:
            documents = self._documents
            if documents is None:
                documents = self.vector_store.get_all_documents()
                self._documents = documents
            self._bm25_index = BM25Index(documents)
        return self._bm25_index

    def refresh(self) -> None:
        """知识库更新后重建 BM25 索引。"""
        self._documents = None
        self._bm25_index = None
        logger.info("BM25 索引已标记为待重建")

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        alpha: float = 0.5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """执行混合检索。

        参数
            query: 查询语句。
            top_k: 返回条数，默认取 config.bm25_top_k 与 vector_top_k 的较大值。
            alpha: 向量路的权重（0~1）。0.5 = 等权，退化为标准 RRF；
                   1.0 = 只信向量路；0.0 = 只信 BM25。
            where: Chroma 元数据过滤条件，例如 {"sector": "banking"}。
        """
        limit = top_k or max(self.config.bm25_top_k, self.config.vector_top_k)
        alpha = max(0.0, min(1.0, alpha))

        with self.tracer.span(
            "retrieval", "hybrid_retriever", input_summary=query, top_k=limit, alpha=alpha
        ) as span:
            vector_results = self._safe_vector_search(query, self.config.vector_top_k, where)
            bm25_results = self._safe_bm25_search(query, self.config.bm25_top_k)

            if not vector_results and not bm25_results:
                span.set_output("两路均无结果")
                return []

            fused = self._fuse(vector_results, bm25_results, alpha=alpha)[:limit]
            span.add_metadata(
                vector_hits=len(vector_results),
                bm25_hits=len(bm25_results),
                fused_hits=len(fused),
            )
            span.set_output(
                f"融合后 {len(fused)} 条，Top1: "
                + (fused[0].content[:60] if fused else "无")
            )
            return fused

    # ---------------- 内部实现 ----------------

    def _safe_vector_search(
        self, query: str, top_k: int, where: dict[str, Any] | None
    ) -> list[RetrievalResult]:
        """向量路失败时降级为空结果，让 BM25 单路继续工作。

        embedding 模型加载失败、Chroma 目录损坏都属于"这一路挂了"，
        不应该让整个检索失败——单路结果虽然差一些，但远好过没有结果。
        """
        try:
            return self.vector_store.search(query, top_k=top_k, where=where)
        except Exception as exc:  # noqa: BLE001
            logger.error("向量检索失败，降级为仅 BM25: %s", exc)
            return []

    def _safe_bm25_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        try:
            return self.bm25.search(query, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            logger.error("BM25 检索失败，降级为仅向量检索: %s", exc)
            return []

    def _fuse(
        self,
        vector_results: Sequence[RetrievalResult],
        bm25_results: Sequence[RetrievalResult],
        alpha: float,
    ) -> list[RetrievalResult]:
        vector_ids = [result.doc_id for result in vector_results]
        bm25_ids = [result.doc_id for result in bm25_results]

        rrf_scores = reciprocal_rank_fusion(
            [vector_ids, bm25_ids],
            k=self.config.rrf_k,
            # alpha=0.5 时两个权重都是 1.0，即标准等权 RRF
            weights=[alpha * 2, (1 - alpha) * 2],
        )

        # 合并两路的文档内容与名次信息
        merged: dict[str, RetrievalResult] = {}
        for result in list(vector_results) + list(bm25_results):
            existing = merged.get(result.doc_id)
            if existing is None:
                merged[result.doc_id] = result.model_copy(
                    update={"retrieval_method": "hybrid", "ranks": dict(result.ranks)}
                )
            else:
                existing.ranks.update(result.ranks)
                # 保留分数较高的那一路的原始分，仅用于展示
                existing.score = max(existing.score, result.score)

        for doc_id, result in merged.items():
            result.rrf_score = rrf_scores.get(doc_id, 0.0)

        return sorted(merged.values(), key=lambda item: item.rrf_score or 0.0, reverse=True)

    def stats(self) -> dict[str, Any]:
        return {
            "vector_documents": self.vector_store.count(),
            "bm25_documents": len(self.bm25.documents) if self._bm25_index else 0,
            "bm25_ready": self._bm25_index.ready if self._bm25_index else False,
            "rrf_k": self.config.rrf_k,
        }


__all__ = ["BM25Index", "HybridRetriever", "reciprocal_rank_fusion", "tokenize"]
