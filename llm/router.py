"""两层模型路由。

解决什么问题
    一次完整研究要发十几次 LLM 请求，如果全用推理模型（deepseek-reasoner），
    成本和延迟都会翻好几倍。运行时四个 Agent 都走 chat；
    deep（reasoner）只留给评测 Judge。

核心设计决策
    1. 只分两层（quick / deep），不做更细的动态路由。动态路由（按问题难度
       实时选模型）听起来聪明，但会引入"路由决策本身也要调一次 LLM"的开销，
       而且路由错误难以复盘。静态分层的规则写在这里，一眼能看懂：
         - deep（reasoner）：评测 Judge
           —— 需要多步推理且错了代价大；
         - quick（chat）：Planner / Retriever / Verifier / Writer / 上下文压缩
           —— 任务明确、模式固定。Verifier 是结构化抽取，数字对错由
           Evidence Gate 确定性校验，不依赖推理深度；chat 还支持
           response_format，reasoner 单次 60-84 秒且不支持 JSON mode。
    2. Retriever 必须走 quick 层，这不只是成本考虑：reasoner 不支持
       function calling，走 deep 层它根本调不了工具。能力约束在这里
       与成本判断刚好一致。
    3. 客户端惰性创建 + 缓存。deep 层在很多请求里根本用不上（比如纯健康检查），
       启动时就建两个连接池是浪费；同时保证同一 tier 全局共享一个连接池。

为什么不用其他方案
    - 不用 LiteLLM 之类的多供应商网关：本项目只对接 DeepSeek 一家，
      引入网关等于为了将来可能的需求增加一个当下就要维护的依赖。
"""

from __future__ import annotations

import logging
from typing import Literal

from config import Config, get_config
from llm.client import LLMClient

logger = logging.getLogger(__name__)

Tier = Literal["quick", "deep"]

# 各 Agent 节点的默认层级。集中在一处声明，方便面试时直接指着这段讲路由策略。
AGENT_TIER_MAP: dict[str, Tier] = {
    "planner": "quick",  # 分类+填表，规则兜底
    "retriever": "quick",  # 工具调用，且 reasoner 不支持 function calling
    "verifier": "quick",  # 结构化抽取；数字对错由 Evidence Gate 校验
    "writer": "quick",  # 按固定模板写作，模式化任务
    "compactor": "quick",  # 高频调用，只做摘要
    "judge": "deep",  # 评测打分，需要严格推理
}


class LLMRouter:
    """按 tier 分发 LLMClient 的路由器。"""

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._clients: dict[Tier, LLMClient] = {}

    def get(self, tier: Tier = "quick") -> LLMClient:
        if tier not in ("quick", "deep"):
            raise ValueError(f"未知的模型层级: {tier!r}，只支持 'quick' / 'deep'")
        if tier not in self._clients:
            model = self.config.quick_model if tier == "quick" else self.config.deep_model
            self._clients[tier] = LLMClient(
                api_key=self.config.deepseek_api_key,
                base_url=self.config.deepseek_base_url,
                model=model,
            )
            logger.info("初始化 %s 层 LLM 客户端: %s", tier, model)
        return self._clients[tier]

    def for_agent(self, agent_name: str) -> LLMClient:
        """按 Agent 名取客户端。未登记的 Agent 默认走 quick 层（更便宜）。"""
        tier = AGENT_TIER_MAP.get(agent_name, "quick")
        return self.get(tier)

    @property
    def quick(self) -> LLMClient:
        return self.get("quick")

    @property
    def deep(self) -> LLMClient:
        return self.get("deep")

    def close(self) -> None:
        for client in self._clients.values():
            client.close()
        self._clients.clear()

    def describe(self) -> dict[str, str]:
        """给 /api/health 和 README 用的路由说明。"""
        return {
            "quick_model": self.config.quick_model,
            "deep_model": self.config.deep_model,
            **{agent: tier for agent, tier in AGENT_TIER_MAP.items()},
        }
