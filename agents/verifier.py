"""Verifier Agent（quick model + Evidence Gate）。

解决什么问题
    这是整个系统里唯一负责"拦住幻觉"的环节。它要回答两个问题：
    (1) 针对用户的问题，从现有证据里能得出哪些结论？每条结论有没有证据支撑？
    (2) 证据够不够？不够的话该补搜什么？
    没有这一环，Writer 会拿着一堆数字自由发挥，写出的研报里
    "已验证"和"编造"混在一起，而且没人分得出来。

核心设计决策
    1. **职责切分：LLM 只做抽取，判定全部交给代码。**
       LLM 负责把证据里的信息组织成一条条带数字的断言（自然语言 → 结构化），
       这是它擅长的；EvidenceGate 负责比对数字、检查日期、判定来源等级，
       这是确定性计算。让 LLM 自己判断"这句话有没有依据"等于让被告当法官。
    2. 走 quick 层（deepseek-chat）。抽取是结构化任务而非推理任务，
       断言的数字正确性由后续 Evidence Gate 确定性校验，不依赖推理深度。
       quick 层速度快 5-10 倍，且支持 response_format 保证输出格式合法。
    3. 补搜触发条件严格按 md：unverified_ratio > 0.3 且 retrieval_count < 3。
       两个条件缺一不可：只看比例会导致证据永远不够时无限补搜；
       只看轮次会导致明明证据充分也要跑满三轮。
    4. **数字必须由 LLM 显式抽成结构化字段**，而不是让门禁去正则原文。
       原文里的 "毛利率从 17.1% 提升到 21.8%" 有两个数字，
       正则无法知道哪个对应"本期"。抽取时要求 LLM 指明 metric 和 value，
       语义归属由生产方负责。
    5. 抽取失败（LLM 输出无法解析）时，**全部结论降级为 unverified 而不是放行**。
       宁可让研报里全是 ⚠️，也不能让未经检查的内容标成 ✅。

为什么不用其他方案
    - 不用"让 LLM 对着证据自查一遍"（self-critique）：幻觉的成因之一
      就是模型对自己的输出过度自信，让它自查往往会确认原答案。
      外部的确定性校验才有独立性。
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from config import get_config
from harness.evidence_gate import EvidenceGate
from harness.tracing import NullTracer, Tracer
from harness.types import (
    GATE_LOOKAHEAD_BLOCKED,
    GATE_NUMBER_MISMATCH,
    GATE_UNVERIFIED,
    GATE_VERIFIED,
    Claim,
    ClaimNumber,
    Evidence,
    GateResult,
)
from llm.client import LLMClient
from llm.router import LLMRouter

logger = logging.getLogger(__name__)


def _normalize_claim_ticker(value: str) -> str:
    """规范化 LLM 自报的股票代码。非 6 位数字一律视为未提供。

    模型偶尔会填公司名（"贵州茅台"）或带前缀（"SH600519"）。
    宽松接受纯数字部分，其余情况返回空串走兜底逻辑，
    而不是把脏值塞进 Claim.ticker——那会让门禁过滤掉所有证据。
    """
    digits = re.sub(r"\D", "", str(value or ""))
    return digits if len(digits) == 6 else ""


VERIFIER_SYSTEM_PROMPT = """你是一名严谨的金融研究审核员，负责把证据材料整理成可核查的结论清单。

## 你的职责
从提供的证据中抽取出能回答用户问题的**结论断言**，并为每条断言标出其中的数字。
你**不需要**判断这些断言是否可信——那由后续的自动校验程序完成。
你的任务是把信息组织成可被程序核查的结构。

## 最重要的一条：抽「回答问题的论断」，不是「数据点清单」
你的输出会直接成为研报的「核心结论」。如果你把每个数字都抽成一条断言，
研报的核心结论就会变成一张流水账（"营业成本为6613.05亿元"——
这回答不了任何问题），读者看完不知道结论是什么。

正确做法：**一条断言 = 一个能回答用户问题的判断 + 支撑它的数字**。
支撑数字放进 numbers 数组，不要为每个数字单独立一条断言。

  用户问："净利润为什么下降"
  ✅ 好："净利润下降的主因是毛利率收窄 1.70pp，对应毛利额减少 83.96 亿元；
         其中汽车业务毛利率下滑 1.82pp 是最大拖累"
         numbers: [{gross_margin,0.1774,2025}, {gross_margin,0.1944,2024},
                   {汽车、汽车相关产品及其他产品_毛利率,0.2049,2025}]
  ❌ 坏："2025年营业成本为6613.05亿元"（只是一个数据点，不回答问题）
  ❌ 坏："2025年毛利率为17.74%"+"2024年毛利率为19.44%"（拆成两条流水账，
         而不是一条带对比的论断）

数量上，一份研报 6~12 条论断是合理的；超过 20 条基本可以肯定是在罗列数据点。

## 抽取规则
1. 每条断言必须是一个完整、独立、可核查的陈述句。
   好的例子："2024年营业收入为7771.02亿元，同比增长29.0%"
   坏的例子："营收增长很快"（无法核查）、"公司经营向好"（无数字、无边界）
2. 断言中的每个数字都要在 numbers 里单独列出：
   - metric: 指标的英文标识，必须从下面的「允许的指标名」里选，不要自创。
   - value: 数值。金额一律换算成**元**（7771.02亿 → 777102000000）；
            比率一律用**小数**（21.8% → 0.218）；增速同样用小数（+29% → 0.29）。
   - kind: absolute（金额等绝对值）/ ratio（比率）/ growth（同比增速）
   - raw_text: 原文里的写法，例如 "7771.02亿元"
   - **period: 该数字属于哪个报告期**，填年份即可，如 "2025"。
     这一项在**跨期对比**的断言里是必填的。例如
     "毛利率由2024年的19.44%降至2025年的17.74%" 要拆成两个数字：
       {metric: "gross_margin", value: 0.1944, period: "2024"}
       {metric: "gross_margin", value: 0.1774, period: "2025"}
     两个数字的 metric 相同，只有 period 能区分它们。不填 period 会导致
     系统拿同一期的数据去核对两个值，其中一个必然被判为不一致。
3. 纯定性断言（不含数字）也要抽出来，numbers 留空数组即可。
4. section 标明这条断言属于研报的哪个部分：
   revenue（收入结构）/ profitability（盈利能力）/ risk（风险提示）/ conclusion（核心结论）
5. source_hint 填写这条断言依据的证据来源名称（从证据列表里复制）。
6. **ticker 填写这条断言所描述的公司代码**。默认是本次分析的目标公司，
   但如果这条断言讲的是**另一家公司**（例如同业对比、或用户问题提到了别的标的），
   必须填那家公司的真实代码。填错会导致系统拿错公司的数据去核对，
   把正确的结论误判为「数字不一致」。

## 允许的指标名
财务类：revenue, operating_cost, gross_profit, net_profit, deducted_net_profit,
gross_margin, net_margin, roe, asset_turnover, equity_multiplier, revenue_yoy,
net_profit_yoy, deducted_net_profit_yoy, rd_expense, rd_ratio, total_assets,
total_liabilities, net_assets, accounts_receivable, accounts_receivable_yoy,
inventory, goodwill, goodwill_to_equity, debt_ratio, operating_cashflow,
cashflow_to_profit

行情类（注意各自的确切含义，不要混用）：
- latest_close：**最新一个交易日**的收盘价
- period_return：**整个查询区间**的累计涨跌幅（不是某一天的涨跌幅）
- daily_volatility：区间内日收益率的标准差

## 关于找不到对应指标名的数字
如果某个数字在上面的列表里**找不到语义完全对应**的指标名
（例如"某一天的单日涨跌幅"、"目标价"、"销量"、"单车利润"），
**不要硬塞到语义相近但不同的指标名上**。正确做法是二选一：
  (a) 把这个数字从 numbers 里省略，断言文本照常保留（它会作为定性断言被核查）；
  (b) 如果这条断言的价值完全依赖那个数字，就不要抽取这条断言，
      改在 coverage_gaps 里说明"缺少 XX 指标的结构化支持"。
硬塞的后果是系统拿两个不同含义的数字做比对，必然报「数字不一致」，
把本来正确的内容错杀成拒答。

## 覆盖度：按研究计划逐步检查
任务里会给出「研究计划」的分析步骤。**逐条对照**，确保每个步骤都有对应的断言：
- 该步骤有证据支撑 → 抽出对应的论断；
- 该步骤的数据缺失 → 写进 coverage_gaps，不要沉默跳过。

对「为什么变化」这类归因问题，利润表是一条自上而下的链条，
只讲其中一环等于只回答了一部分。抽取时依次检查这几环是否都有证据：
  毛利（收入 - 成本）→ 期间费用（销售/管理/财务/研发）
  → 非经营性损益（政府补助、投资收益、减值损失）→ 所得税
哪一环有数据就抽哪一环的论断，都没有数据的环节写进 coverage_gaps。
实测中最容易被漏掉的是**期间费用**和**非经营性损益**这两环——
它们往往能解释"为什么净利率的降幅和毛利率的降幅对不上"。

## 硬性约束
- 只能使用证据中出现的数字，**绝对禁止**推算、估计或补全证据里没有的数字。
- 同一组数字、同一个判断只抽一条。禁止把「ROE 同比持平」重复写三遍。
- 如果证据不足以回答用户问题的某个方面，在 coverage_gaps 里写明缺什么数据，
  不要为了凑数而编造断言。
- 如果证据完全为空或与问题无关，claims 返回空数组，并在 coverage_gaps 里说明。
"""

VERIFIER_TASK_TEMPLATE = """## 用户问题
{question}

## 目标公司
{company}（{ticker}）

## 分析基准日
{as_of_date}

## 研究计划（供参考，说明本次分析的重点）
{plan}

## 可用证据
{evidence_block}

请抽取结论断言。"""


class ExtractedNumber(BaseModel):
    """LLM 抽取出的一个数字。"""

    model_config = ConfigDict(extra="ignore")

    metric: str
    value: float
    kind: Literal["absolute", "ratio", "growth"] = "absolute"
    raw_text: str = ""
    # 该数字属于哪个报告期，如 "2025" 或 "2025-12-31"。
    # 跨期对比断言（"由 2024 年的 19.44% 降至 2025 年的 17.74%"）必须填，
    # 否则两个数字的 metric 相同、期间不同，门禁无法区分（F-018）。
    period: str = ""


class ExtractedClaim(BaseModel):
    """LLM 抽取出的一条断言。"""

    model_config = ConfigDict(extra="ignore")

    text: str
    numbers: list[ExtractedNumber] = Field(default_factory=list)
    section: Literal["revenue", "profitability", "risk", "conclusion"] = "conclusion"
    source_hint: str = ""
    # 这条断言描述的是哪家公司。留空则回退为本次分析的目标公司。
    # 存在的理由见 F-015：原先把会话 ticker 强贴到所有断言上，
    # 导致"茅台收盘价 1355.99"这条断言被打成 ticker=002594，
    # 门禁过滤掉茅台证据后拿比亚迪的 89.78 去比，报出误导性的「数字不一致」。
    ticker: str = ""


class ClaimExtraction(BaseModel):
    """Verifier 的 LLM 抽取输出。"""

    model_config = ConfigDict(extra="ignore")

    claims: list[ExtractedClaim] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    suggested_query: str = ""


class VerifiedClaim(BaseModel):
    """经过证据门禁判定的断言。"""

    model_config = ConfigDict(extra="ignore")

    claim_id: str
    text: str
    section: str = "conclusion"
    # 门禁的四种状态
    gate_status: str = GATE_UNVERIFIED
    # 映射到研报的三级标注
    verdict: str = "unverified"
    label: str = "⚠️未验证"
    source: str | None = None
    reason: str = ""
    numbers: dict[str, float] = Field(default_factory=dict)
    checks: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def is_verified(self) -> bool:
        return self.gate_status == GATE_VERIFIED


class VerifyResult(BaseModel):
    """Verifier 的完整输出。"""

    model_config = ConfigDict(extra="ignore")

    claims: list[VerifiedClaim] = Field(default_factory=list)
    needs_more_retrieval: bool = False
    expand_query: str | None = None
    unverified_ratio: float = 0.0
    coverage_gaps: list[str] = Field(default_factory=list)
    # 抽取阶段是否失败（失败时所有断言被强制降级）
    extraction_failed: bool = False

    @property
    def verified_claims(self) -> list[VerifiedClaim]:
        return [claim for claim in self.claims if claim.gate_status == GATE_VERIFIED]

    @property
    def refused_claims(self) -> list[VerifiedClaim]:
        return [claim for claim in self.claims if claim.verdict == "refused"]

    @property
    def evidence_coverage(self) -> float:
        """已验证结论占总结论的比例（评测的 evidence_coverage 维度）。"""
        if not self.claims:
            return 0.0
        return len(self.verified_claims) / len(self.claims)

    def summary(self) -> dict[str, Any]:
        return {
            "total_claims": len(self.claims),
            "verified": len(self.verified_claims),
            "unverified": sum(1 for c in self.claims if c.verdict == "unverified"),
            "refused": len(self.refused_claims),
            "unverified_ratio": round(self.unverified_ratio, 3),
            "evidence_coverage": round(self.evidence_coverage, 3),
            "needs_more_retrieval": self.needs_more_retrieval,
        }


def _fingerprint_claim(claim: Any) -> str:
    """断言去重指纹：优先用结构化数字，否则用压缩后的原文。"""
    numbers = getattr(claim, "numbers", None) or {}
    if isinstance(numbers, dict) and numbers:
        parts = [
            f"{key}={round(float(value), 6)}"
            for key, value in sorted(numbers.items())
            if isinstance(value, (int, float))
        ]
        if parts:
            return "n:" + "|".join(parts)
    if isinstance(numbers, list) and numbers:
        parts: list[str] = []
        for item in numbers:
            if isinstance(item, dict):
                metric = str(item.get("metric") or "")
                value = item.get("value")
                period = str(item.get("period") or "")
            else:
                metric = str(getattr(item, "metric", "") or "")
                value = getattr(item, "value", None)
                period = str(getattr(item, "period", "") or "")
            if metric and isinstance(value, (int, float)):
                parts.append(f"{metric}:{round(float(value), 6)}:{period}")
        if parts:
            return "n:" + "|".join(sorted(parts))
    return "t:" + re.sub(r"\s+", "", getattr(claim, "text", "") or "")


def _dedupe_extracted_claims(claims: Sequence[ExtractedClaim]) -> list[ExtractedClaim]:
    seen: set[str] = set()
    unique: list[ExtractedClaim] = []
    for claim in claims:
        fingerprint = _fingerprint_claim(claim)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(claim)
    return unique


def _dedupe_verified_claims(claims: Sequence[VerifiedClaim]) -> list[VerifiedClaim]:
    rank = {"verified": 0, "unverified": 1, "refused": 2}
    best: dict[str, VerifiedClaim] = {}
    order: list[str] = []
    for claim in claims:
        fingerprint = _fingerprint_claim(claim)
        previous = best.get(fingerprint)
        if previous is None:
            best[fingerprint] = claim
            order.append(fingerprint)
            continue
        if rank.get(claim.verdict, 9) < rank.get(previous.verdict, 9):
            best[fingerprint] = claim
        elif claim.verdict == previous.verdict and len(claim.text) > len(previous.text):
            best[fingerprint] = claim
    return [best[key] for key in order]


class VerifierAgent:
    """证据核查 Agent。"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        evidence_gate: EvidenceGate | None = None,
        llm_router: LLMRouter | None = None,
        tracer: Tracer | None = None,
    ):
        self.config = get_config()
        if llm_client is None:
            router = llm_router or LLMRouter()
            llm_client = router.get("quick")
        self.llm = llm_client
        self.tracer = tracer or NullTracer()
        self.gate = evidence_gate or EvidenceGate(tracer=self.tracer)

    # ---------------- 主入口 ----------------

    def verify(
        self,
        question: str,
        evidence_pool: Sequence[Evidence],
        plan_description: str = "",
        ticker: str = "",
        company: str = "",
        as_of_date: date | None = None,
        retrieval_count: int = 1,
    ) -> VerifyResult:
        """抽取断言并逐条过证据门禁。"""
        as_of = as_of_date or date.today()

        if not evidence_pool:
            # 没有任何证据时不必调 LLM：结论只能是"无法回答"。
            # 这条捷径同时避免了模型在无证据情况下凭参数记忆编造答案。
            logger.warning("证据池为空，Verifier 直接返回拒答")
            return VerifyResult(
                claims=[],
                needs_more_retrieval=retrieval_count < self.config.max_retrieval_rounds,
                expand_query=question,
                unverified_ratio=1.0,
                coverage_gaps=["证据池为空，未获取到任何可用数据"],
            )

        with self.tracer.span(
            "node_enter", "verifier", input_summary=f"{len(evidence_pool)} 条证据"
        ) as span:
            extraction = self._extract_claims(
                question, evidence_pool, plan_description, ticker, company, as_of
            )
            extraction.claims = _dedupe_extracted_claims(extraction.claims)

            verified_claims: list[VerifiedClaim] = []
            for extracted in extraction.claims:
                claim = self._to_claim(extracted, as_of, ticker)
                gate_result = self.gate.check_financial_claim(claim, evidence_pool)
                verified_claims.append(
                    self._to_verified_claim(claim, extracted, gate_result, extraction_failed=False)
                )
            verified_claims = _dedupe_verified_claims(verified_claims)

            unverified_ratio = self._compute_unverified_ratio(verified_claims)
            needs_more = self._should_retrieve_more(
                unverified_ratio, retrieval_count, verified_claims, extraction.coverage_gaps
            )

            result = VerifyResult(
                claims=verified_claims,
                needs_more_retrieval=needs_more,
                expand_query=self._build_expand_query(extraction, question) if needs_more else None,
                unverified_ratio=unverified_ratio,
                coverage_gaps=extraction.coverage_gaps,
                extraction_failed=False,
            )

            span.add_metadata(**result.summary())
            span.set_output(
                f"{len(verified_claims)} 条断言，已验证 {len(result.verified_claims)} 条，"
                f"未验证率 {unverified_ratio:.1%}，{'需要补搜' if needs_more else '证据充分'}"
            )
            logger.info("Verifier 完成: %s", result.summary())
            return result

    # ---------------- 抽取 ----------------

    def _extract_claims(
        self,
        question: str,
        evidence_pool: Sequence[Evidence],
        plan_description: str,
        ticker: str,
        company: str,
        as_of: date,
    ) -> ClaimExtraction:
        task = VERIFIER_TASK_TEMPLATE.format(
            question=question,
            company=company or "（未指定）",
            ticker=ticker or "（未指定）",
            as_of_date=as_of.isoformat(),
            plan=plan_description or "（无）",
            evidence_block=self._format_evidence(evidence_pool),
        )

        response = self.llm.chat_structured(
            messages=[{"role": "user", "content": task}],
            schema=ClaimExtraction,
            system_prompt=VERIFIER_SYSTEM_PROMPT,
        )

        if isinstance(response, ClaimExtraction):
            return response

        logger.error("Verifier 断言抽取失败，原始输出: %s", str(response)[:300])
        # 抽取失败时不放弃：用证据本身生成保守的定性断言，
        # 它们会因为"无数字 + 官方来源"而被判为 verified，
        # 至少让研报能引用到真实数据，而不是整体拒答。
        return ClaimExtraction(
            claims=[],
            coverage_gaps=["结论抽取环节失败，本次结果仅包含原始数据，未做结构化核查"],
            suggested_query=question,
        )

    @staticmethod
    def _format_evidence(evidence_pool: Sequence[Evidence], limit: int = 30) -> str:
        """把证据池渲染成 prompt 文本。

        数字单独列出来（而不是只给 content），是为了让 LLM 抽取时
        直接复制这些值，避免它从 content 的自然语言里二次解析出错。
        """
        blocks: list[str] = []
        for index, evidence in enumerate(evidence_pool[:limit], start=1):
            numbers = (
                "\n".join(f"    {key} = {value!r}" for key, value in evidence.numbers.items())
                or "    （无结构化数字）"
            )
            blocks.append(
                f"[证据 {index}] 来源: {evidence.source_name}\n"
                f"  来源类型: {evidence.source_type}"
                f"（{'官方披露' if evidence.is_official else '非官方，只能标未验证'}）\n"
                f"  披露日期: {evidence.disclosure_date.isoformat()}"
                + (f" | 报告期: {evidence.period_end.isoformat()}" if evidence.period_end else "")
                + f"\n  内容: {evidence.content[:600]}\n"
                f"  可用数字（请直接引用这些值，单位已统一为元/小数）:\n{numbers}"
            )
        if len(evidence_pool) > limit:
            blocks.append(f"...（另有 {len(evidence_pool) - limit} 条证据未展示）")
        return "\n\n".join(blocks)

    # ---------------- 门禁与转换 ----------------

    @staticmethod
    def _to_claim(extracted: ExtractedClaim, as_of: date, ticker: str) -> Claim:
        """把抽取结果转成待核查的 Claim。

        ticker 的归属规则：断言自报的优先，会话代码只作兜底。
        绝不能反过来——用会话代码覆盖断言自报的代码，会让涉及其他标的的断言
        （同业对比、用户问了另一家公司）被拿去和目标公司的数据比对，
        产生"数字不一致"的误判（F-015）。
        """
        claim_ticker = _normalize_claim_ticker(extracted.ticker) or ticker or None
        return Claim(
            text=extracted.text,
            numbers=[
                ClaimNumber(
                    metric=number.metric,
                    value=number.value,
                    kind=number.kind,
                    raw_text=number.raw_text or None,
                    period=number.period or None,
                )
                for number in extracted.numbers
            ],
            as_of_date=as_of,
            source_hint=extracted.source_hint or None,
            ticker=claim_ticker,
            section=extracted.section,
        )

    @staticmethod
    def _to_verified_claim(
        claim: Claim,
        extracted: ExtractedClaim,
        gate_result: GateResult,
        extraction_failed: bool,
    ) -> VerifiedClaim:
        from harness.types import VERDICT_LABEL

        verdict = "unverified" if extraction_failed else gate_result.verdict
        return VerifiedClaim(
            claim_id=claim.claim_id,
            text=claim.text,
            section=extracted.section,
            gate_status=gate_result.status,
            verdict=verdict,
            label=VERDICT_LABEL.get(verdict, "⚠️未验证"),
            source=gate_result.matched_source,
            reason=gate_result.reason,
            numbers={number.metric: number.value for number in claim.numbers},
            checks=[check.model_dump() for check in gate_result.checks],
        )

    @staticmethod
    def _compute_unverified_ratio(claims: Sequence[VerifiedClaim]) -> float:
        """未验证率 = 非 verified 的断言占比。

        分母为 0（一条断言都没抽出来）时返回 1.0 而不是 0.0：
        "没有任何结论"应当被视为完全没有证据支撑，从而触发补搜；
        返回 0.0 会让系统误以为"全部已验证"。
        """
        if not claims:
            return 1.0
        unverified = sum(1 for claim in claims if claim.gate_status != GATE_VERIFIED)
        return unverified / len(claims)

    def _should_retrieve_more(
        self,
        unverified_ratio: float,
        retrieval_count: int,
        claims: Sequence[VerifiedClaim],
        coverage_gaps: Sequence[str],
    ) -> bool:
        """补搜判定。

        md 规定：unverified_ratio > 0.3 且 retrieval_count < 3。
        这里对"未验证率"做了收窄：只统计**带数字的断言**。

        为什么：知识库/新闻上的定性断言按设计永远拿不到 ✅（非官方来源）。
        把它们算进未验证率，系统会为了不可能验证的内容把检索跑满三轮——
        用例 1 因此从 2 个工具膨胀到 11 个工具、286 秒。
        """
        if retrieval_count >= self.config.max_retrieval_rounds:
            return False

        blocked = any(
            claim.gate_status in (GATE_NUMBER_MISMATCH, GATE_LOOKAHEAD_BLOCKED)
            for claim in claims
        )
        if blocked:
            return True

        # coverage_gaps 继续传给 expand_query；触发条件不再单独看缺口描述，
        # 避免「知识库框架未覆盖」这种永远补不完的缺口把循环跑满。
        numeric = [claim for claim in claims if claim.numbers]
        if numeric:
            ratio = sum(
                1 for claim in numeric if claim.gate_status != GATE_VERIFIED
            ) / len(numeric)
        elif coverage_gaps and not any(c.is_verified for c in claims):
            ratio = 1.0
        else:
            ratio = unverified_ratio

        return ratio > self.config.unverified_ratio_threshold

    @staticmethod
    def _build_expand_query(extraction: ClaimExtraction, question: str) -> str:
        if extraction.suggested_query:
            return extraction.suggested_query
        if extraction.coverage_gaps:
            return "；".join(extraction.coverage_gaps[:3])
        return question


__all__ = [
    "ClaimExtraction",
    "ExtractedClaim",
    "ExtractedNumber",
    "VerifiedClaim",
    "VerifierAgent",
    "VerifyResult",
]
