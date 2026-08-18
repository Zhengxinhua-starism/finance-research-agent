"""Planner Agent（quick model）。

解决什么问题
    如果把用户问题原样丢给 Retriever，会出现两种典型失败：
    (1) 问"毛利率为什么下降"，Retriever 只拉了一期利润表就开始回答——
        一期数据根本看不出"下降"；
    (2) 问"ROE 是多少"，Retriever 把六个工具全调了一遍——浪费 5 倍 token 和时间。
    Planner 的职责是把自然语言问题翻译成**结构化的研究计划**：
    这是什么类型的问题、需要哪些数据、按什么步骤分析、用什么框架写作。

核心设计决策
    1. 走 quick 层（deepseek-chat）。规划本质是分类 + 填工具清单，
       后面还有关键词规则兜底（类型上调、主题补工具、投资建议工具集）。
       reasoner 在这一步通常要 10~20 秒，却几乎不提高计划质量；
       推理预算留给 Verifier——门禁错判的代价远高于计划多带一个工具。
    2. 输出强约束为 ResearchPlan 结构。required_data 的取值被限制在
       已注册的工具名集合内，模型编出一个不存在的工具名时会在校验阶段
       被剔除并补上规则兜底，而不是等到 Retriever 调用时才报 tool_not_found。
    3. **规则兜底优先于重试**。LLM 结构化输出失败时不重试（重试同样可能失败，
       还要多等十几秒），而是用 question_type 的规则映射直接生成一份可用计划。
       md 里给出了四种问题类型到工具的确定性映射，这本身就是一份好计划。
    4. analysis_framework 字段告诉 Writer 用哪个分析框架（杜邦/现金流/成长性/
       风险）。让 Writer 自己临场决定框架会导致同一类问题两次运行的结构不同，
       评测时表现为分数波动。

为什么不用其他方案
    - 不用 ReAct 让 Planner 自己调工具探索：规划阶段调工具会把成本翻倍，
      而且"先看看数据再决定看什么数据"是个循环论证。计划应当基于
      问题类型的先验知识，不是基于数据。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from harness.tracing import NullTracer, Tracer
from llm.client import LLMClient
from llm.router import LLMRouter

logger = logging.getLogger(__name__)

QuestionType = Literal["single_fact", "cross_period", "causal", "risk"]
AnalysisFramework = Literal[
    "dupont", "cashflow_quality", "growth", "risk_screening", "investment_view", "general"
]

# md "问题类型与所需工具的对应关系"的代码化实现。
# 同时作为 LLM 输出的校验基准和失败时的兜底方案。
QUESTION_TYPE_TOOLS: dict[str, list[str]] = {
    "single_fact": ["get_financial_metrics"],
    # 跨期默认只要多期序列 + 程序化对比。分部拆解、知识库按主题叠加，
    # 不要默认带 get_segment_breakdown——问 ROE 变化时拉主营构成是过度取数。
    "cross_period": ["get_income_history", "compare_periods"],
    # 归因类是工具最多的一类，因为"为什么"需要多个角度交叉印证：
    # - compare_periods 定位变化发生在哪一期、幅度多大
    # - get_segment_breakdown 拆出是结构效应还是各业务自身效应（关键）
    # - get_operating_efficiency 看运营效率是否恶化（杜邦第二因子）
    # - search_news 提供财报解释不了的外部因素（价格战、原材料、政策）
    # - search_knowledge 提供该行业的归因框架
    # 少了 search_news 的后果是 Writer 只能用数字解释数字：
    # "毛利率降了所以净利降了"——这是同义反复，不是归因。
    "causal": [
        "get_income_history",
        "compare_periods",
        "get_segment_breakdown",
        "get_operating_efficiency",
        # 公告是第一层来源里唯一的"文字"证据：一次性减值、会计政策变更、
        # 资产重组这些归因线索不在报表数字里，只在公告里
        "get_disclosure_events",
        "search_news",
        "search_knowledge",
    ],
    "risk": [
        "get_balance_sheet",
        "get_cash_flow",
        "get_risk_snapshot",
        "get_operating_efficiency",
        "search_knowledge",
    ],
}

QUESTION_TYPE_FRAMEWORK: dict[str, AnalysisFramework] = {
    "single_fact": "general",
    "cross_period": "growth",
    "causal": "dupont",
    "risk": "risk_screening",
}

# 关键词 → 问题类型。用于 LLM 失败时的规则兜底，以及对 LLM 判断的交叉验证。
TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "risk": ("风险", "隐患", "暴雷", "减值", "偿债", "债务", "商誉", "应收", "现金流质量", "利润质量"),
    "causal": ("为什么", "原因", "为何", "怎么造成", "驱动", "归因", "受什么影响", "如何解释"),
    "cross_period": ("趋势", "变化", "近三年", "近几年", "同比", "环比", "相比", "对比", "增长", "下降"),
}

# 明确索取投资建议的表述。命中后按「投资观点」框架取数并给出倾向判断，
# 不再短路拒答。关键词仍刻意选「明确索取交易动作」的表述，
# 以便补上股价、风险、新闻、机构覆盖，而不是只调 get_financial_metrics。
ADVICE_KEYWORDS: tuple[str, ...] = (
    "推荐什么价", "什么价位", "什么价格买", "推荐买", "建议买", "能不能买", "该不该买",
    "值不值得买", "值得买吗", "可以买吗", "要不要买", "该买吗",
    "能不能卖", "该不该卖", "要不要卖", "建议卖",
    "目标价", "买入价", "入手价", "抄底", "加仓", "减仓", "清仓", "止损位", "止盈",
    "推荐股票", "推荐个股", "选股", "帮我买", "能涨到", "会涨吗", "会跌吗",
    "后市如何", "还能涨", "还会跌",
)

ADVICE_TOOLS: tuple[str, ...] = (
    "get_stock_price",
    "get_financial_metrics",
    "get_risk_snapshot",
    "search_news",
    "get_analyst_coverage",
)

ADVICE_STEPS: list[str] = [
    "获取最新收盘价与近期走势，作为价位判断的基准",
    "获取最新一期核心财务指标，判断盈利与增长质量",
    "执行风险规则检查（杠杆、现金流、应收等）",
    "检索近期新闻与机构覆盖主题，作为辅助观察",
    "综合基本面、价格位置与风险，给出偏多 / 中性 / 偏空的倾向判断",
    "若用户问具体建仓价，必须引用已核查的最新收盘价；没有股价数据则明确说无法给价位",
]

# 主题关键词 → 该主题必需的工具。
#
# 为什么需要这一层：四种 question_type 是按**提问方式**分的
# （单点/跨期/归因/风险），而工具是按**数据主题**组织的。两者不是一一对应。
# 实测暴露的问题：「经营现金流和资本开支状况如何，扩张是否可持续」
# 这个问题不含任何 risk 关键词（"风险""隐患""偿债"都没有），
# 被归为 single_fact，于是只调了 get_financial_metrics，
# 三个必需工具（现金流表、资产负债表、风险快照）一个都没调，事实分 0.20。
#
# 这一层与 question_type 是**叠加**关系不是替代：类型决定分析框架，
# 主题决定必须带上哪些数据。命中多个主题就都带上。
TOPIC_TOOL_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("行情", "股价", "股票价格", "涨跌", "涨幅", "跌幅", "收盘", "市值",
         "走势", "K线", "换手", "成交量", "波动"),
        ("get_stock_price", "search_news"),
    ),
    (
        ("现金流", "资本开支", "自由现金流", "扩张", "可持续", "烧钱",
         "在建工程", "产能建设", "投入"),
        ("get_cash_flow", "get_balance_sheet", "get_risk_snapshot"),
    ),
    (
        ("有息负债", "偿债", "杠杆", "负债率", "借款", "债务", "净现金"),
        ("get_balance_sheet", "get_risk_snapshot"),
    ),
    (
        ("分部", "业务结构", "各业务", "海外", "境外", "国内", "产品结构",
         "分产品", "分地区", "哪块业务", "毛利率"),
        ("get_segment_breakdown",),
    ),
    (
        ("周转", "存货", "应收", "运营效率", "回款", "库存"),
        ("get_operating_efficiency", "get_balance_sheet"),
    ),
    (
        ("公告", "减值", "会计政策", "重组", "并购", "一次性", "非经常"),
        ("get_disclosure_events",),
    ),
    (
        ("机构", "研报", "券商", "market关注", "市场怎么看"),
        ("get_analyst_coverage",),
    ),
    (
        ("银行", "不良", "净息差", "拨备", "非息", "净利息", "成本收入比"),
        ("search_knowledge",),
    ),
    (
        ("ROE", "净资产收益率", "杜邦"),
        ("get_financial_metrics", "compare_periods", "search_knowledge"),
    ),
    (
        ADVICE_KEYWORDS,
        ADVICE_TOOLS,
    ),
)

# 研究计划里仍保留这两个字段，避免旧会话反序列化失败；投资建议不再走拒答。
OUT_OF_SCOPE_REASON = ""

PLANNER_SYSTEM_PROMPT = """你是一名资深金融分析师，负责把用户的研究问题拆解成结构化的研究计划。

你的任务不是回答问题，而是规划"要查什么数据、按什么步骤分析"。

## 可用的数据工具
{tool_catalog}

## 问题类型判定标准
- single_fact（单点事实）：只问一个具体指标的当前值。例："比亚迪2024年ROE是多少"
- cross_period（跨期对比）：问变化趋势、同比、多期对比。例："平安银行近三年不良率变化趋势"
- causal（归因分析）：问原因、为什么。例："茅台净利率为什么这么高"
- risk（风险评估）：问风险、隐患、质量。例："比亚迪应收账款风险大吗"

## 规划原则
1. 归因类和跨期类问题必须获取至少 3 期数据，否则看不出趋势。
2. 归因类问题除了财务数据，还应检索知识库获取分析框架（用 search_knowledge）。
   银行、保险等特殊行业的跨期问题同样需要 search_knowledge——
   不能用制造业的毛利率框架解释银行的营收/利润背离。
3. 单点事实问题不要过度取数，调用 1~2 个工具即可。
   跨期对比默认 2~4 个工具，不要把归因类的全套工具搬过来。
   若用户在问能不能买、什么价位建仓、后市如何，必须取股价、财务、风险和新闻，
   并规划「给出倾向判断」这一步，不要当成超范围问题跳过。
4. research_steps 要写成可执行的分析动作，不要写"分析财务状况"这种空话。
5. expand_keywords 用于后续补充检索，应包含该问题涉及的业务概念，而非公司名。

## 特别注意
如果问题中的公司或标的明显不存在、无法识别，请把 question_type 设为 single_fact，
required_data 设为空列表，并在 reasoning 中说明"标的无法识别，应当拒答"。
"""

PLANNER_USER_TEMPLATE = """研究问题：{question}
公司代码：{ticker}
公司名称：{company}
分析基准日：{as_of_date}

请输出研究计划。"""


class ResearchPlan(BaseModel):
    """结构化研究计划。"""

    model_config = ConfigDict(extra="ignore")

    question_type: QuestionType = "single_fact"
    required_data: list[str] = Field(default_factory=list, description="需要调用的工具名列表")
    research_steps: list[str] = Field(default_factory=list, description="分析步骤")
    expand_keywords: list[str] = Field(default_factory=list, description="补充检索关键词")
    analysis_framework: AnalysisFramework = "general"
    reasoning: str = ""
    # 以下两个字段由代码填充，不由 LLM 生成
    fallback_used: bool = False
    # 问题超出系统职责范围（索取投资建议）→ 跳过检索与核查，直接拒答
    out_of_scope: bool = False
    out_of_scope_reason: str = ""

    def describe(self) -> str:
        """给下游 Agent 的 prompt 用的计划文本。"""
        steps = "\n".join(f"  {i}. {step}" for i, step in enumerate(self.research_steps, 1))
        return (
            f"问题类型: {self.question_type}\n"
            f"分析框架: {self.analysis_framework}\n"
            f"需要的数据: {', '.join(self.required_data) or '（无）'}\n"
            f"分析步骤:\n{steps or '  （无）'}\n"
            f"关注要点: {', '.join(self.expand_keywords) or '（无）'}"
        )


class PlannerAgent:
    """研究规划 Agent。"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        llm_router: LLMRouter | None = None,
        available_tools: list[dict[str, Any]] | None = None,
        tracer: Tracer | None = None,
    ):
        if llm_client is None:
            router = llm_router or LLMRouter()
            llm_client = router.get("quick")
        self.llm = llm_client
        self.tracer = tracer or NullTracer()
        self.available_tools = available_tools or []

    # ---------------- 主入口 ----------------

    def plan(
        self,
        question: str,
        ticker: str = "",
        company: str = "",
        as_of_date: str = "",
    ) -> ResearchPlan:
        """生成研究计划。任何失败都降级为规则计划，不抛异常。"""
        with self.tracer.span("node_enter", "planner", input_summary=question) as span:
            # 快捷路径：single_fact 类问题不需要 LLM 规划。
            # 关键词分类已经能准确判定，调 LLM 只是多花 2-5 秒拿回同样的结果。
            rule_type = self.classify_question(question)
            if rule_type == "single_fact" and not self.is_advice_request(question):
                plan = self.rule_based_plan(question)
                span.set_output(f"[快捷路径] {plan.question_type} / {plan.required_data}")
                span.add_metadata(
                    question_type=plan.question_type,
                    tool_count=len(plan.required_data),
                    framework=plan.analysis_framework,
                    shortcut=True,
                )
                return plan

            system_prompt = PLANNER_SYSTEM_PROMPT.format(tool_catalog=self._tool_catalog())
            user_message = PLANNER_USER_TEMPLATE.format(
                question=question,
                ticker=ticker or "（未指定）",
                company=company or "（未指定）",
                as_of_date=as_of_date or "（未指定，按最新数据）",
            )

            response = self.llm.chat_structured(
                messages=[{"role": "user", "content": user_message}],
                schema=ResearchPlan,
                system_prompt=system_prompt,
            )

            if isinstance(response, ResearchPlan):
                plan = self._validate(response, question)
                span.set_output(f"{plan.question_type} / {plan.required_data}")
            else:
                logger.warning("Planner 结构化输出失败，使用规则兜底。原始输出: %s", str(response)[:200])
                plan = self.rule_based_plan(question)
                plan.fallback_used = True
                span.add_metadata(fallback=True, raw_output=str(response)[:200])
                span.set_output(f"[规则兜底] {plan.question_type} / {plan.required_data}")

            span.add_metadata(
                question_type=plan.question_type,
                tool_count=len(plan.required_data),
                framework=plan.analysis_framework,
            )
            return plan

    # ---------------- 校验与兜底 ----------------

    def _validate(self, plan: ResearchPlan, question: str) -> ResearchPlan:
        """校正 LLM 输出：剔除不存在的工具、补齐必要工具、修正框架。"""
        known_tools = {tool["function"]["name"] for tool in self.available_tools}

        # LLM 偶尔把「近三年趋势」判成 single_fact。关键词规则更稳：
        # 信息需求只能升不能降，否则分析框架会从 growth 掉到 general，
        # Writer 就不会去解释营收/利润背离。
        rule_type = self.classify_question(question)
        stronger = {"single_fact": 0, "cross_period": 1, "causal": 2, "risk": 2}
        if stronger.get(rule_type, 0) > stronger.get(plan.question_type, 0):
            logger.info(
                "问题类型由关键词规则上调: %s → %s", plan.question_type, rule_type
            )
            plan.question_type = rule_type  # type: ignore[assignment]

        if known_tools:
            unknown = [name for name in plan.required_data if name not in known_tools]
            if unknown:
                logger.warning("Planner 请求了不存在的工具，已剔除: %s", unknown)
            plan.required_data = [name for name in plan.required_data if name in known_tools]

        # 计划为空时用规则补齐。空计划会让 Retriever 无工具可用，
        # 直接导致整份研报没有任何数据支撑。
        if not plan.required_data:
            rule_plan = self.rule_based_plan(question)
            plan.required_data = [
                name for name in rule_plan.required_data if not known_tools or name in known_tools
            ]
            plan.fallback_used = True
            logger.info("Planner 未给出工具，按规则补齐: %s", plan.required_data)

        # 跨期/归因类问题必须有多期数据源，否则趋势判断无从谈起
        if plan.question_type in ("cross_period", "causal"):
            has_multi_period = any(
                name in plan.required_data
                for name in ("get_income_history", "compare_periods")
            )
            if not has_multi_period:
                candidate = "compare_periods" if not known_tools or "compare_periods" in known_tools else None
                if candidate:
                    plan.required_data.append(candidate)
                    logger.info("%s 类问题缺少多期数据源，已补充 compare_periods", plan.question_type)

        # 按问题涉及的数据主题补齐工具。question_type 按提问方式分类，
        # 覆盖不到"这个问题需要哪些数据"——两者叠加才完整。
        for tool_name in self.suggest_tools_by_topic(question):
            if tool_name not in plan.required_data and (
                not known_tools or tool_name in known_tools
            ):
                plan.required_data.append(tool_name)
                logger.info("问题主题命中，补充工具 %s", tool_name)

        # LLM 容易给简单问题堆工具。类型基线 + 主题补齐之外的工具，
        # 在非归因/非风险问题上裁掉，避免第一轮就把检索面铺开。
        if plan.question_type in ("single_fact", "cross_period") and not self.is_advice_request(
            question
        ):
            keep = set(QUESTION_TYPE_TOOLS.get(plan.question_type, []))
            keep.update(self.suggest_tools_by_topic(question))
            keep.update(
                {
                    "get_financial_metrics",
                    "compare_periods",
                    "get_income_history",
                    "search_knowledge",
                }
            )
            pruned = [name for name in plan.required_data if name in keep]
            dropped = [name for name in plan.required_data if name not in keep]
            if pruned and dropped:
                logger.info("非归因类问题裁剪多余工具: %s", dropped)
                plan.required_data = pruned

        if self.is_advice_request(question):
            plan.analysis_framework = "investment_view"
            for tool_name in ADVICE_TOOLS:
                if tool_name not in plan.required_data and (
                    not known_tools or tool_name in known_tools
                ):
                    plan.required_data.append(tool_name)
            plan.research_steps = list(ADVICE_STEPS)
            logger.info("投资建议类问题，改用 investment_view 框架并补齐行情/风险/新闻工具")
        elif plan.analysis_framework == "general":
            plan.analysis_framework = QUESTION_TYPE_FRAMEWORK.get(plan.question_type, "general")

        if not plan.research_steps:
            plan.research_steps = self.rule_based_plan(question).research_steps

        # 去重但保持顺序（顺序反映了 Planner 认为的优先级）
        plan.required_data = list(dict.fromkeys(plan.required_data))
        return plan

    def rule_based_plan(self, question: str) -> ResearchPlan:
        """不调 LLM 的规则计划。用于兜底，也用于离线测试。"""
        question_type = self.classify_question(question)
        tools = list(QUESTION_TYPE_TOOLS[question_type])
        tools.extend(self.suggest_tools_by_topic(question))
        tools = list(dict.fromkeys(tools))
        known_tools = {tool["function"]["name"] for tool in self.available_tools}
        if known_tools:
            tools = [name for name in tools if name in known_tools]

        framework: AnalysisFramework = QUESTION_TYPE_FRAMEWORK[question_type]
        steps = self._default_steps(question_type)
        if self.is_advice_request(question):
            framework = "investment_view"
            steps = list(ADVICE_STEPS)

        return ResearchPlan(
            question_type=question_type,  # type: ignore[arg-type]
            required_data=tools,
            research_steps=steps,
            expand_keywords=self._extract_keywords(question),
            analysis_framework=framework,
            reasoning="由关键词规则生成（LLM 规划不可用或输出无效）",
            fallback_used=True,
        )

    @staticmethod
    def is_advice_request(question: str) -> bool:
        """判断问题是否在索取买卖时点、目标价或持仓建议。"""
        text = str(question or "").replace(" ", "")
        return any(keyword in text for keyword in ADVICE_KEYWORDS)

    @staticmethod
    def is_out_of_scope(question: str) -> bool:
        """兼容旧调用。投资建议已纳入研究范围，不再视为超范围。"""
        return False

    @staticmethod
    def suggest_tools_by_topic(question: str) -> list[str]:
        """按问题涉及的数据主题给出必需工具。命中多个主题就都返回。"""
        text = str(question or "")
        suggested: list[str] = []
        for keywords, tools in TOPIC_TOOL_HINTS:
            if any(keyword in text for keyword in keywords):
                suggested.extend(tools)
        return list(dict.fromkeys(suggested))

    @staticmethod
    def classify_question(question: str) -> QuestionType:
        """按关键词判定问题类型。

        判定顺序是 risk → causal → cross_period → single_fact，
        因为一个问题可能同时命中多类关键词（"应收账款风险近三年变化"），
        此时应取信息需求最强的那一类（风险评估需要的数据是跨期对比的超集）。
        """
        text = question.strip()
        for question_type in ("risk", "causal", "cross_period"):
            if any(keyword in text for keyword in TYPE_KEYWORDS[question_type]):
                return question_type  # type: ignore[return-value]
        return "single_fact"

    @staticmethod
    def _default_steps(question_type: str) -> list[str]:
        return {
            "single_fact": [
                "获取最新一期核心财务指标",
                "确认该指标的口径（归母/合并、年报/季报）和报告期",
                "给出数值并标注数据来源与披露日期",
            ],
            "cross_period": [
                "获取最近 3 期年报数据，建立时间序列",
                "计算各期同比变动（比率型用百分点，金额型用百分比）",
                "判断趋势方向，区分「增速放缓」与「绝对值下降」",
                "若营收与利润增速背离，结合行业分析框架给出可能原因"
                "（银行：拨备计提、非息收入、成本收入比；制造业：毛利率、费用、非经常性损益），"
                "取不到分项数据时明确标注为可能原因而非已验证事实",
                "涉及 ROE 时必须写明计算口径（期末摊薄 vs 平均净资产）",
                "标记变动超过阈值的显著异动并说明",
            ],
            "causal": [
                "获取最近 3 期利润表，定位变化发生的具体期间和幅度",
                "调 get_segment_breakdown 做分部拆解：整体毛利率变动有多少来自"
                "业务结构变化（结构效应）、多少来自各业务自身盈利能力变化（自身效应）",
                "按地区维度再拆一次，看境内外盈利能力差异及其变化",
                "拆解成本与费用端：营业成本增速 vs 营收增速、期间费用率变化",
                "检查利润构成：减值损失、投资收益、其他收益（政府补助）等"
                "非经营性项目对净利润的贡献占比",
                "调 get_operating_efficiency 看周转天数，判断运营效率是否恶化",
                "调 search_news 查找财报数据无法解释的外部因素"
                "（价格战、原材料涨价、政策变化、一次性事件）",
                "检索知识库获取该行业的归因框架，区分行业性因素与公司个体因素",
                "区分主因与次因：按各因子对目标指标的贡献大小排序，不要平铺罗列",
            ],
            "risk": [
                "获取资产负债表与现金流量表",
                "执行风险规则检查（应收背离、现金流覆盖、杠杆、商誉）",
                "对触发的每条预警，结合行业特性判断是否构成实质风险",
                "检索知识库获取对应风险项的核查方法",
                "明确说明哪些规则因数据缺失未能执行",
            ],
        }.get(question_type, ["获取相关财务数据", "分析并给出结论"])

    @staticmethod
    def _extract_keywords(question: str) -> list[str]:
        """从问题里提取业务概念关键词，用于后续补充检索。"""
        concept_keywords = [
            "毛利率", "净利率", "ROE", "净资产收益率", "营收", "营业收入", "净利润",
            "扣非", "现金流", "应收账款", "存货", "商誉", "负债率", "研发",
            "不良率", "净息差", "拨备", "预收款", "增速", "成本", "费用",
        ]
        found = [keyword for keyword in concept_keywords if keyword in question]
        # 补上问题里的年份，帮助后续检索定位期间
        found.extend(re.findall(r"20\d{2}", question))
        return list(dict.fromkeys(found))[:8]

    def _tool_catalog(self) -> str:
        if not self.available_tools:
            return "（工具列表未提供，请按标准工具名规划：" + ", ".join(
                sorted({name for names in QUESTION_TYPE_TOOLS.values() for name in names})
            ) + "）"
        lines: list[str] = []
        for tool in self.available_tools:
            function = tool.get("function", {})
            description = str(function.get("description", ""))
            lines.append(f"- {function.get('name')}: {description[:160]}")
        return "\n".join(lines)


__all__ = [
    "QUESTION_TYPE_FRAMEWORK",
    "QUESTION_TYPE_TOOLS",
    "PlannerAgent",
    "ResearchPlan",
]
