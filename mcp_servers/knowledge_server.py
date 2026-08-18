"""知识库 MCP Server。

解决什么问题
    把"混合检索 + 精排"这条链路包装成一个 Agent 可调用的工具。
    Agent 不需要知道背后有 BM25、Chroma、RRF、CrossEncoder 四个组件，
    它只看到一个 search_knowledge(query, top_k) 工具。

核心设计决策
    1. 检索链路（HybridRetriever → CrossEncoderReranker）在 Server 里组装，
       而不是暴露成两个工具让 LLM 自己串。让 LLM 决定"要不要精排"
       既没有意义（答案永远是要）又多一轮工具调用。
    2. 组件懒加载。embedding 模型 + CrossEncoder 加起来约 200MB 内存和
       10 秒加载时间。纯财务数据问题（"比亚迪 ROE 是多少"）根本用不到知识库，
       在 Server 构造时就加载会让每次 API 冷启动都白等 10 秒。
    3. 知识库为空时返回明确提示而不是空结果。空结果会让 LLM 以为
       "知识库里确实没有相关内容"，进而基于自身参数记忆编造答案；
       明确告知"知识库未初始化"能让它转而说明数据缺失。
    4. 检索结果一律标注 source_type="knowledge_base"。这类内容是
       分析框架和行业常识，不是公司事实，在证据门禁里只能支撑定性判断，
       永远拿不到 ✅已验证 标注。

为什么不用其他方案
    - 不把知识检索合并进 financial_data Server：两者的失败模式完全不同
      （一个依赖网络和外部接口，一个依赖本地模型和向量库），
      健康检查和降级策略应该独立。
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from config import get_config
from harness.tracing import NullTracer, Tracer
from harness.types import Evidence, RetrievalResult, ToolResult
from mcp_servers.base_server import MCPServer
from rag.hybrid_retriever import HybridRetriever
from rag.rag_store import RagStore
from rag.reranker import CrossEncoderReranker
from tools.data_client import BaseTool

logger = logging.getLogger(__name__)


class SearchKnowledgeTool(BaseTool):
    """知识库检索工具（BM25 + 向量混合检索 + CrossEncoder 精排）。"""

    name = "search_knowledge"
    description = (
        "检索金融研报知识库，获取行业分析框架、财务分析方法论、风险预警规则、"
        "A股信息披露制度等背景知识。适用于需要方法论支撑的问题，"
        "例如「毛利率下降该从哪些角度归因」「杜邦分析怎么拆」「银行业看什么指标」"
        "「应收账款增速过快意味着什么」。"
        "注意：知识库提供的是通用分析框架和行业常识，不包含具体公司的财务数据，"
        "公司数据请使用 get_financial_metrics 等数据工具获取。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索问题，用自然语言描述你想了解的分析方法或行业知识",
            },
            "top_k": {
                "type": "integer",
                "description": "返回的知识条目数，默认 5，最大 10",
                "default": 5,
            },
            "sector": {
                "type": "string",
                "description": (
                    "可选，按行业过滤。可选值：banking（银行）、"
                    "new_energy_vehicle（新能源汽车）、baijiu（白酒）、general（通用）。"
                    "留空则检索全部"
                ),
            },
        },
        "required": ["query"],
    }
    # 检索结果依赖知识库内容，内容更新后缓存会失效；
    # 而且检索本身很快（本地计算），缓存收益远小于陈旧风险
    cacheable = False

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        tracer: Tracer | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.tracer = tracer or NullTracer()
        self._retriever = retriever
        self._reranker = reranker
        # 最近一次检索的原始结果，供 KnowledgeMCPServer 构造 Evidence。
        # 走这条旁路而不是把结构塞进工具返回值，是为了让给 LLM 的文本
        # 保持精简（不带 doc_id、rrf_score 这类它用不上的字段）。
        self.last_results: list[RetrievalResult] = []

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(tracer=self.tracer)
        return self._retriever

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(tracer=self.tracer)
        return self._reranker

    def fetch(self, query: str, top_k: int = 5, sector: str | None = None) -> dict[str, Any]:
        config = get_config()
        limit = max(1, min(int(top_k), 10))

        if self.retriever.vector_store.count() == 0:
            self.last_results = []
            return {
                "query": query,
                "result_count": 0,
                "results": [],
                "status": "knowledge_base_empty",
                "note": (
                    "知识库尚未初始化（0 条记录）。这不代表相关知识不存在，"
                    "而是本次检索无法提供任何背景知识。"
                    "请勿基于此编造分析框架，应在结论中说明缺少方法论依据。"
                    "管理员可执行 python -m rag.prepare_data 初始化知识库。"
                ),
            }

        where = {"sector": sector} if sector else None
        # 粗排召回量取两路 top_k 的较大值，给精排留出足够的候选空间
        candidates = self.retriever.retrieve(
            query, top_k=max(config.bm25_top_k, config.vector_top_k), where=where
        )
        results = self.reranker.rerank(query, candidates, top_k=limit)
        self.last_results = results

        return {
            "query": query,
            "sector_filter": sector,
            "result_count": len(results),
            "results": [
                {
                    "rank": index + 1,
                    "title": item.metadata.get("title", "（无标题）"),
                    "content": item.content,
                    "sector": item.metadata.get("sector", "general"),
                    "knowledge_type": item.metadata.get("type", "unknown"),
                    "source": item.metadata.get("source", "手工整理"),
                    "relevance_score": item.rerank_score
                    if item.rerank_score is not None
                    else item.rrf_score,
                    "retrieved_by": item.ranks,
                }
                for index, item in enumerate(results)
            ],
            "retrieval_method": "BM25 + 向量混合检索 → RRF 融合 → CrossEncoder 精排",
            "evidence_level": "knowledge_base",
            "usage_note": (
                "以上为通用分析框架和行业常识，可用于指导分析思路和解释现象，"
                "但不能作为具体公司事实的证据。涉及具体数字的结论必须引用财报数据。"
            ),
        }


class KnowledgeMCPServer(MCPServer):
    """RAG 知识库 MCP Server。"""

    name = "knowledge_base"
    description = "金融研报知识库检索（BM25+向量混合检索+RRF融合+CrossEncoder精排）"

    def __init__(
        self,
        vector_store: RagStore | None = None,
        retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        tracer: Tracer | None = None,
    ):
        super().__init__(tracer=tracer)
        self._vector_store = vector_store
        self.search_tool = SearchKnowledgeTool(
            retriever=retriever, reranker=reranker, tracer=self.tracer
        )
        self.register_tools([self.search_tool])

    @property
    def retriever(self) -> HybridRetriever:
        return self.search_tool.retriever

    def refresh_index(self) -> None:
        """知识库更新后重建 BM25 索引。"""
        self.retriever.refresh()

    def evidence_from_tool_result(
        self, result: ToolResult, as_of_date: date | None = None
    ) -> list[Evidence]:
        """把检索结果转成 Evidence。

        disclosure_date 取 date.min 而不是今天：知识库内容是通用常识，
        不存在"披露时点"的概念，用最小日期保证它永远不会被前视偏差拦截。
        对应地，source_type="knowledge_base" 保证它拿不到 ✅已验证。
        """
        if not result.success or not isinstance(result.data, dict):
            return []

        evidences: list[Evidence] = []
        for item in result.data.get("results") or []:
            evidences.append(
                Evidence(
                    source_type="knowledge_base",
                    source_name=f"知识库：{item.get('title', '未命名条目')}",
                    disclosure_date=date.min,
                    content=str(item.get("content", "")),
                    numbers={},  # 知识库是方法论，不含可校验的公司数字
                    tool_name=result.tool_name,
                    metadata={
                        "sector": item.get("sector", "general"),
                        "knowledge_type": item.get("knowledge_type", "unknown"),
                        "relevance_score": item.get("relevance_score"),
                        "origin": item.get("source", "手工整理"),
                    },
                )
            )
        return evidences

    def health(self) -> dict[str, Any]:
        base = super().health()
        try:
            store = self._vector_store or self.retriever.vector_store
            base["knowledge_base"] = store.health()
        except Exception as exc:  # noqa: BLE001 — 健康检查本身不能抛异常
            base["knowledge_base"] = {"status": "error", "error": str(exc)}
        return base


__all__ = ["KnowledgeMCPServer", "SearchKnowledgeTool"]
