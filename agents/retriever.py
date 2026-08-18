"""Retriever Agent（quick model + MCP 工具调用）。

解决什么问题
    按 Planner 的计划把数据取回来，并把它们转成**带来源和披露日期的证据**，
    而不是一堆裸数字。证据池的质量直接决定后面 Verifier 能验证多少条结论——
    如果这一步只存了"营收 7771 亿"而没存"来自 2024 年报、披露于 2025-03-28"，
    那么后面的三级门禁里有两级（来源、时点）根本无从检查。

核心设计决策
    1. 工具调用走 MCP Broker，不直接 import 工具模块（md 硬性要求）。
       链路：Retriever → BrokerToolRegistry → MCPToolBroker → MCPServer
       → ToolRegistry → AKShare/Chroma。多这两层的收益是：
       新增数据源只要写一个 MCPServer 并注册进 Broker，本文件一行不改。
    2. 采用 **ReAct 循环 + 计划裁剪工具集**，而不是"按计划顺序硬调工具"。
       硬调的问题是：工具失败后没有补救（比如 get_income_history 挂了，
       其实可以改用 compare_periods 拿到同样的数据）。ReAct 让模型看见失败
       并自主换路。而裁剪工具集（只给 plan.required_data 里的工具）
       又避免了模型漫无目的地乱调。
    3. 证据在**工具结果回调里实时构造**（on_tool_results），不是等循环结束后
       统一处理。这样即使循环因 max_turns 中断，已经取到的数据也在证据池里，
       Writer 仍能写出一份带 ⚠️ 标注的降级研报。
    4. 补充检索（第二轮及以后）不重新执行整个计划，而是带着 Verifier 给出的
       expand_query 做定向补搜，并在 prompt 里显式告知"哪些结论还缺证据"。
       重跑整个计划只会把同样的工具再调一遍，拿到同样的数据。

为什么不用其他方案
    - 不用并行硬调所有工具再交给 LLM 总结：看起来更快，但会调到不需要的工具
      （单点事实问题也去拉三张报表），成本上升且证据池被无关数据稀释，
      Verifier 的匹配准确率会下降。
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING, Any, Sequence

from config import get_config
from harness.agent_loop import AgentLoop
from harness.tracing import NullTracer, Tracer
from harness.types import AgentResult, Evidence, ToolResult
from llm.client import LLMClient
from llm.router import LLMRouter
from mcp_servers.base_server import BrokerToolRegistry, MCPServer, MCPToolBroker

if TYPE_CHECKING:
    from agents.planner import ResearchPlan

logger = logging.getLogger(__name__)

RETRIEVER_SYSTEM_PROMPT = """你是一名金融数据检索助手，负责调用工具获取研究所需的数据。

## 你的职责
严格按研究计划调用工具取数，**不要做分析、不要下结论、不要编造任何数字**。
所有数字必须来自工具返回结果。

## 工作方式
1. 阅读研究计划，确定需要哪些数据。
2. 调用相应工具。可以在一轮里同时发起多个工具调用以节省时间。
3. 如果某个工具失败，尝试用其他工具获取等价数据；
   连续失败两次的数据项，直接说明"该数据获取失败"，不要反复重试。
4. 数据齐了之后，输出一段简短的取数总结（不超过 300 字），说明：
   - 成功获取了哪些数据（列出关键数字和对应的报告期）
   - 哪些数据没能获取到，原因是什么

## 硬性约束
- 禁止在总结里写任何工具没有返回过的数字。
- 禁止对数据做因果解释或投资判断，那是后续环节的工作。
- 如果所有工具都失败，明确说明"未获取到任何数据"。
"""

RETRIEVER_TASK_TEMPLATE = """## 研究问题
{question}

## 目标公司
{company}（{ticker}）

## 分析基准日
{as_of_date}
注意：只能使用在该日期之前已经披露的数据。

## 研究计划
{plan}

请开始取数。"""

SUPPLEMENT_TASK_TEMPLATE = """## 补充检索（第 {round_number} 轮）

## 原始研究问题
{question}

## 目标公司
{company}（{ticker}）

## 已有数据概况
{existing_summary}

## 验证环节发现的缺口
{gaps}

## 建议的补充检索方向
{expand_query}

请只针对上述缺口补充取数，不要重复调用已经成功返回过数据的工具。"""


# 补搜缺口关键词 → 额外开放的工具。
# 第二轮起不再把工具目录全部打开：实测一轮 ROE 对比题因此膨胀到 11 个工具、3 轮、286 秒。
GAP_TOOL_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("分部", "业务结构", "毛利率", "产品", "地区"), ("get_segment_breakdown",)),
    (("周转", "运营效率", "存货", "应收"), ("get_operating_efficiency",)),
    (("公告", "减值", "重组", "会计政策"), ("get_disclosure_events",)),
    (("新闻", "价格战", "外部", "政策"), ("search_news",)),
    (("框架", "行业", "知识", "银行", "杜邦", "口径"), ("search_knowledge",)),
    (("现金流", "资本开支", "自由现金"), ("get_cash_flow", "get_risk_snapshot")),
    (("负债", "杠杆", "偿债"), ("get_balance_sheet", "get_risk_snapshot")),
    (("ROE", "净资产收益率"), ("get_financial_metrics", "compare_periods")),
)

SUPPLEMENT_FALLBACK_TOOLS: tuple[str, ...] = (
    "get_financial_metrics",
    "compare_periods",
    "search_knowledge",
)


class RetrieverAgent:
    """数据检索 Agent。"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        mcp_servers: Sequence[MCPServer] | None = None,
        broker: MCPToolBroker | None = None,
        llm_router: LLMRouter | None = None,
        tracer: Tracer | None = None,
        max_turns: int | None = None,
    ):
        if llm_client is None:
            router = llm_router or LLMRouter()
            # 必须走 quick 层：deepseek-reasoner 不支持 function calling
            llm_client = router.get("quick")
        self.llm = llm_client
        self.tracer = tracer or NullTracer()

        if broker is None:
            if mcp_servers is None:
                from mcp_servers import build_mcp_servers

                mcp_servers = build_mcp_servers(tracer=self.tracer)
            broker = MCPToolBroker(mcp_servers)
        self.broker = broker
        self.servers: list[MCPServer] = list(mcp_servers or broker.servers)

        self.registry = BrokerToolRegistry(broker, tracer=self.tracer)
        self.agent_loop = AgentLoop(
            llm_client=self.llm,
            tool_registry=self.registry,
            max_turns=max_turns or get_config().max_turns,
            tracer=self.tracer,
            agent_name="retriever",
        )
        # 工具名 → 产出该工具证据的 Server，用于把 ToolResult 转成 Evidence
        self._evidence_sources: dict[str, MCPServer] = {
            tool_name: server for server in self.servers for tool_name in server.tool_names
        }

    # ---------------- 主入口 ----------------

    def retrieve(
        self,
        question: str,
        plan: "ResearchPlan",
        ticker: str = "",
        company: str = "",
        as_of_date: date | None = None,
        round_number: int = 1,
        existing_evidence: Sequence[Evidence] | None = None,
        gaps: Sequence[str] | None = None,
        expand_query: str | None = None,
    ) -> tuple[list[Evidence], AgentResult]:
        """执行一轮检索，返回（新增证据列表，Agent 运行结果）。

        返回元组而不是直接改 state：让节点函数决定怎么合并进状态，
        Agent 本身保持无副作用，便于单独测试。
        """
        as_of = as_of_date or date.today()
        collected: list[Evidence] = []

        def on_tool_results(results: list[ToolResult]) -> None:
            collected.extend(self._build_evidence(results, as_of))

        if round_number > 1:
            task = SUPPLEMENT_TASK_TEMPLATE.format(
                round_number=round_number,
                question=question,
                company=company or "（未指定）",
                ticker=ticker or "（未指定）",
                existing_summary=self._summarize_evidence(existing_evidence or []),
                gaps="\n".join(f"- {gap}" for gap in (gaps or [])) or "- （未指定具体缺口）",
                expand_query=expand_query or "（无具体建议，请按缺口自行判断）",
            )
            allowed_tools = self._tools_for_supplement(
                plan, gaps or [], expand_query
            )
        else:
            task = RETRIEVER_TASK_TEMPLATE.format(
                question=question,
                company=company or "（未指定）",
                ticker=ticker or "（未指定）",
                as_of_date=as_of.isoformat(),
                plan=plan.describe(),
            )
            allowed_tools = plan.required_data or None

        result = self.agent_loop.run(
            system_prompt=RETRIEVER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": task}],
            context={
                "ticker": ticker,
                "company": company,
                "as_of_date": as_of.isoformat(),
                "retrieval_round": round_number,
            },
            allowed_tools=allowed_tools,
            on_tool_results=on_tool_results,
        )

        deduped = self._deduplicate(collected, existing_evidence or [])
        logger.info(
            "检索第 %d 轮完成: status=%s 工具调用 %d 次（失败 %d），新增证据 %d 条",
            round_number,
            result.status,
            len(result.tool_results),
            result.failed_tool_calls,
            len(deduped),
        )
        return deduped, result

    # ---------------- 证据构造 ----------------

    def _build_evidence(self, results: Sequence[ToolResult], as_of: date) -> list[Evidence]:
        """把工具结果交给对应的 MCP Server 转成 Evidence。

        转换逻辑归属于 Server 而不是这里：只有 Server 知道自己的工具
        产出的是年报数据还是知识库常识，对应的 source_type 和披露日期是什么。
        """
        evidences: list[Evidence] = []
        for result in results:
            if not result.success:
                continue
            server = self._evidence_sources.get(result.tool_name)
            if server is None or not hasattr(server, "evidence_from_tool_result"):
                continue
            try:
                evidences.extend(server.evidence_from_tool_result(result, as_of_date=as_of))
            except Exception as exc:  # noqa: BLE001 — 证据构造失败不能中断检索
                logger.warning("构造 %s 的证据失败: %s", result.tool_name, exc)
        return evidences

    @staticmethod
    def _deduplicate(
        new_evidence: Sequence[Evidence], existing: Sequence[Evidence]
    ) -> list[Evidence]:
        """按 (来源名, 报告期, 数字集合) 去重。

        补充检索经常会把同一个工具再调一次（缓存命中，几乎不耗时），
        产生完全相同的证据。重复证据不会导致错误结论，但会让证据池膨胀，
        拖慢门禁匹配并干扰"证据覆盖率"这个评测指标。
        """
        seen: set[tuple[str, str, str]] = set()
        for evidence in existing:
            seen.add(RetrieverAgent._fingerprint(evidence))

        unique: list[Evidence] = []
        for evidence in new_evidence:
            fingerprint = RetrieverAgent._fingerprint(evidence)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique.append(evidence)
        return unique

    @staticmethod
    def _fingerprint(evidence: Evidence) -> tuple[str, str, str]:
        numbers = ",".join(f"{k}={v:.6g}" for k, v in sorted(evidence.numbers.items()))
        period = evidence.period_end.isoformat() if evidence.period_end else ""
        return (evidence.source_name, period, numbers or evidence.content[:80])

    @staticmethod
    def _summarize_evidence(evidences: Sequence[Evidence]) -> str:
        if not evidences:
            return "（尚无已获取的数据）"
        lines: list[str] = []
        for evidence in evidences[:12]:
            metrics = ", ".join(list(evidence.numbers)[:6]) or "（无数值）"
            lines.append(f"- {evidence.source_name}: {metrics}")
        if len(evidences) > 12:
            lines.append(f"- ...另有 {len(evidences) - 12} 条证据")
        return "\n".join(lines)

    @staticmethod
    def _tools_for_supplement(
        plan: "ResearchPlan",
        gaps: Sequence[str],
        expand_query: str | None,
    ) -> list[str]:
        """按缺口关键词开放补搜工具，而不是把目录全部打开。

        第一轮计划不够用，通常只缺一类数据（口径、分部、现金流），
        把研报、新闻、公告一并放开只会稀释证据池并拖慢评测。
        """
        text = " ".join([expand_query or "", *gaps])
        tools = list(plan.required_data or [])
        for keywords, extra in GAP_TOOL_HINTS:
            if any(keyword in text for keyword in keywords):
                tools.extend(extra)
        if not tools:
            tools.extend(SUPPLEMENT_FALLBACK_TOOLS)
        else:
            for fallback in SUPPLEMENT_FALLBACK_TOOLS:
                if fallback not in tools:
                    tools.append(fallback)
        return list(dict.fromkeys(tools))

    # ---------------- 观测 ----------------

    def available_tools(self) -> list[dict[str, Any]]:
        return self.broker.get_tool_definitions()

    def health(self) -> list[dict[str, Any]]:
        return self.broker.health()


__all__ = ["RetrieverAgent"]
