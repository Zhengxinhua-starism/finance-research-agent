"""CrossEncoder 精排。

解决什么问题
    混合检索召回的 top 20 里，真正相关的通常只有 3~5 条，其余是"词面像但
    答非所问"的噪音。把 20 条全塞给 LLM 有两个坏处：token 成本翻四倍，
    而且噪音会稀释注意力，让模型引用到不相关的知识条目。精排把这 20 条
    重新排序，只留最相关的 5 条。

核心设计决策
    1. 用 CrossEncoder 而不是再跑一次 Bi-Encoder。
       Bi-Encoder（召回阶段用的）把 query 和 document 分别编码再算相似度，
       两者之间没有交互，所以能预计算、能建索引、但精度有限。
       CrossEncoder 把 (query, document) 拼成一条输入过一遍模型，
       让每个 query token 都能注意到每个 document token，精度显著更高，
       代价是没法预计算——必须对每个候选实时算一次。
       "先粗排 20 条（快）再精排 5 条（准）"就是在这两者之间取平衡。
    2. 精排失败时**返回原顺序**而不是抛异常。CrossEncoder 模型有 80MB，
       首次运行要下载，国内网络经常失败。此时混合检索的结果虽然没那么精，
       但完全可用；为了精排而让整个检索链路失败是本末倒置。
    3. 分数不做归一化。CrossEncoder 输出的是 logit（可为负），
       归一化成 [0,1] 会丢掉"所有候选都不相关"这个重要信息——
       原始分数全为负时可以据此判断"知识库里没有相关内容"，
       这是拒答判断的依据之一。
    4. 提供 score_threshold 过滤。宁可返回 2 条高相关的，
       也不要为凑够 top_k 而返回 5 条里有 3 条是噪音。

已知局限（写在这里以免被当成 bug）
    ms-marco-MiniLM-L-6-v2 是在英文 MS MARCO 数据集上训练的，
    对中文的精排能力弱于专门的中文 reranker（如 bge-reranker-base）。
    这里仍按 md 规定选它，理由是体积小（80MB）、下载快、演示环境友好。
    生产环境建议换 BAAI/bge-reranker-base（约 1.1GB，中文效果明显更好）。
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Sequence

from config import get_config
from harness.tracing import NullTracer, Tracer
from harness.types import RetrievalResult

logger = logging.getLogger(__name__)

_CROSS_ENCODERS: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def get_cross_encoder(model_name: str | None = None) -> Any:
    """加载并缓存 CrossEncoder。线程安全。"""
    name = model_name or get_config().reranker_model
    if name in _CROSS_ENCODERS:
        return _CROSS_ENCODERS[name]

    with _MODEL_LOCK:
        if name in _CROSS_ENCODERS:
            return _CROSS_ENCODERS[name]
        from sentence_transformers import CrossEncoder

        logger.info("加载 CrossEncoder 模型: %s（首次运行需要下载约 80MB）", name)
        model = CrossEncoder(name, max_length=512)
        _CROSS_ENCODERS[name] = model
        return model


class CrossEncoderReranker:
    """对混合检索结果做 CrossEncoder 精排。"""

    def __init__(
        self,
        model_name: str | None = None,
        tracer: Tracer | None = None,
        score_threshold: float | None = None,
    ):
        self.config = get_config()
        self.model_name = model_name or self.config.reranker_model
        self.tracer = tracer or NullTracer()
        # None 表示不做阈值过滤。默认关闭是因为不同模型的分数量纲差异很大，
        # 需要针对具体模型标定后再启用。
        self.score_threshold = score_threshold
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        """模型是否可用。只探测一次，失败后不再反复尝试加载。"""
        if self._available is None:
            try:
                get_cross_encoder(self.model_name)
                self._available = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("CrossEncoder 不可用，精排将被跳过: %s", exc)
                self._available = False
        return self._available

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """精排并截断到 top_k。

        模型不可用或打分失败时返回按原顺序截断的结果——降级路径必须
        保证调用方总能拿到"排序合理的 top_k"，而不是空列表或异常。
        """
        limit = top_k or self.config.rerank_top_k
        if not candidates:
            return []
        if len(candidates) == 1:
            return list(candidates)
        if not self.available:
            return list(candidates[:limit])

        with self.tracer.span(
            "retrieval",
            "reranker",
            input_summary=f"{query} | {len(candidates)} 个候选",
            model=self.model_name,
        ) as span:
            try:
                model = get_cross_encoder(self.model_name)
                pairs = [(query, candidate.content) for candidate in candidates]
                scores = model.predict(pairs, show_progress_bar=False)
            except Exception as exc:  # noqa: BLE001
                logger.error("CrossEncoder 打分失败，保持原顺序: %s", exc)
                span.mark_failed(f"{type(exc).__name__}: {exc}")
                return list(candidates[:limit])

            reranked: list[RetrievalResult] = []
            for candidate, score in zip(candidates, scores):
                item = candidate.model_copy(
                    update={
                        "rerank_score": float(score),
                        "retrieval_method": "rerank",
                    }
                )
                reranked.append(item)

            reranked.sort(key=lambda item: item.rerank_score or float("-inf"), reverse=True)

            if self.score_threshold is not None:
                filtered = [
                    item
                    for item in reranked
                    if (item.rerank_score or float("-inf")) >= self.score_threshold
                ]
                # 全部被过滤掉时保留最高分的一条，让调用方能看到"最好也只有这么相关"
                reranked = filtered or reranked[:1]

            result = reranked[:limit]
            span.add_metadata(
                top_score=result[0].rerank_score if result else None,
                kept=len(result),
                dropped=len(candidates) - len(result),
            )
            span.set_output(
                f"精排后保留 {len(result)} 条，Top1 分数 "
                f"{result[0].rerank_score:.3f}" if result else "无结果"
            )
            return result


__all__ = ["CrossEncoderReranker", "get_cross_encoder"]
