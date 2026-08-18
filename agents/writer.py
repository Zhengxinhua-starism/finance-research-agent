"""Writer Agent（quick model，三级标注输出）。

解决什么问题
    把核查过的结论组织成一份人能读的研报。难点不在写作，而在于
    **保证标注不被写歪**：如果让 LLM 自己决定哪句话标 ✅、哪句话标 ⚠️，
    它会倾向于把所有内容都标成"已验证"（模型对自己的输出天然自信），
    前面三级门禁做的所有工作在最后一步全部作废。

核心设计决策
    1. **混合渲染：标注部分由代码生成，叙述部分由 LLM 生成。**
       - "核心结论"清单、"数据来源"表格：直接从 VerifiedClaim 和 Evidence
         渲染，逐条带上门禁判定的标签。LLM 完全碰不到这部分。
       - "详细分析"正文：由 LLM 写，但它只能引用被门禁标记过的断言。
       这样做的效果是：即使 LLM 在正文里写了一句没依据的话，
       读者对照"核心结论"和"数据来源"两个确定性区块就能立刻发现。
       正文生成后仍过一遍门禁：已验证数字原样保留；来自 ⚠️ 结论的数字
       用样式标记为未验证，不堆「该数据未经核实」套话；列表里没有的数字
       从句子里删掉，而不是插入「已拦截」占位。来源统一写在文末表格，
       正文不再每个数字后跟「（来源：…）」。
       核心结论在渲染前按数字指纹去重，避免同一条 ROE 出现三次。
    2. 走 quick 层。写作是模式化任务（模板固定、内容已给定），
       用推理模型不会写得更好，只会更慢更贵。
    3. **拒答是一等公民**。没有任何已验证结论时，直接输出一份
       "无法回答"的研报，说明缺什么数据、为什么无法判断，
       而不是用 ⚠️ 标注硬凑一份看起来像研报的东西。
       查不到就说查不到；用户问买卖时，在有证据的前提下给出倾向判断。
    4. 免责声明和口径说明是硬编码的模板文本，不经过 LLM。
       这类内容一旦被模型改写就可能失去合规意义。

为什么不用其他方案
    - 不让 LLM 输出结构化 JSON 再由代码渲染全文：研报的"详细分析"部分
      需要自然的段落叙述和因果串联，强行结构化会写成机械的条目罗列，
      可读性大幅下降。分区处理才能兼顾可控性和可读性。
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any, Sequence

from harness.tracing import NullTracer, Tracer
from harness.types import VERDICT_LABEL, Evidence
from llm.client import LLMClient
from llm.router import LLMRouter

logger = logging.getLogger(__name__)

FRAMEWORK_GUIDANCE: dict[str, str] = {
    "dupont": (
        "使用杜邦分析框架组织「盈利能力」部分：先给出 ROE，再拆解成"
        "净利率 × 资产周转率 × 权益乘数三个因子，明确指出是哪个因子主导了变动，"
        "并给出该因子的具体变动幅度，说明该因子变动对应的经营含义"
        "（净利率→成本控制，周转率→资产效率，杠杆→债务风险）。\n"
        "必须写明 ROE 的计算口径（期末摊薄 vs 平均净资产）。不同口径差异可达数个百分点，"
        "不写口径等于没把数字说清楚。证据里若有「ROE 计算口径」字样，直接引用，不要省略。\n"
        "如果证据里有分部数据（各业务/各地区的毛利率与收入占比），"
        "必须用它把毛利率变动拆成「结构效应」与「各业务自身效应」——"
        "这是区分「业务结构调整」和「经营恶化」的唯一办法，两者含义完全相反。\n"
        "如果证据里有非经营性损益（投资收益、其他收益/政府补助、减值损失），"
        "要说明它占净利润的比重，判断利润有多少来自主营业务。"
    ),
    "cashflow_quality": (
        "重点分析利润质量：对比经营活动现金流与净利润，给出现金含量比值，"
        "并从应收账款、存货、非经常性损益三条线解释差异来源。"
    ),
    "growth": (
        "使用成长性三层框架：第一层给出营收/净利/扣非净利的增速；"
        "第二层比较三者的相对关系，判断增长质量；"
        "第三层评估可持续性（研发投入、资本开支、增速的二阶变化）。"
        "务必区分「增速放缓」与「绝对值下降」。\n"
        "若营收与利润增速出现背离，必须给出可能原因，而不是只陈述背离事实。"
        "制造业常见路径：毛利率、期间费用、非经常性损益。"
        "银行业没有毛利率，应沿拨备计提、非息收入、成本收入比排查；"
        "若这些科目未取到，把上述路径作为「可能原因」写出，并标注数据未核实。"
        "禁止用制造业的毛利率概念分析银行。"
    ),
    "risk_screening": (
        "按风险规则逐条呈现：每条风险给出触发的指标数值、预警阈值、"
        "以及该风险的实际含义。未触发的规则也要说明已检查过。"
        "对于因数据缺失而未能执行的检查，必须显式说明，不得默认为无风险。"
    ),
    "investment_view": (
        "用户在询问买卖时点、建仓价或后市方向。必须给出明确倾向："
        "偏多 / 中性 / 偏空（或买入 / 持有观望 / 减持），并列出依据。"
        "结构：先给最新价格位置，再给基本面（盈利、增长、现金流），再给风险约束，最后给倾向判断。"
        "倾向是研究观点，不得写成「✅已验证」事实；依据中的财务数字必须来自已核查结论。"
        "具体价位只能引用已核查的最新收盘价或近期均价，禁止编造目标价。"
        "没有股价数据时，可以说基本面倾向，但必须写明「缺少最新报价，无法给出建仓价」。"
    ),
    "general": "按「收入结构 → 盈利能力 → 风险提示」的顺序组织内容。",
}

WRITER_SYSTEM_PROMPT = """你是一名金融研究报告撰写员，负责把已核查的结论写成研报正文。

## 绝对约束（违反即为严重错误）
1. **只能使用「已核查结论」列表里出现的内容和数字。**
   禁止引入任何列表之外的数字，禁止自行推算、估计、补全数据。
2. **用户明确询问买卖、价位、后市时，必须给出倾向判断**（偏多 / 中性 / 偏空）。
   判断本身是研究观点，必须写明「以下为研究观点，不是已验证事实」。
   用户没有问买卖时，不要主动推荐买入或卖出。
3. 引用被标记为「⚠️未验证」的结论时，可以写数字，但不要写成「已经核实」；
   不要复述门禁内部原因（例如找不到某个 metric 名）。
4. 对于「❌拒答」的结论，只能说明"无法判断"及其原因，不得给出替代性猜测。
5. 如果某个分析角度缺少数据支撑，直接写"数据不足，无法判断"，不要绕过。
   没有最新股价时，可以给基本面倾向，但不得编造建仓价或目标价。

## 最重要的一条：不要用数字解释数字
"净利润下降是因为毛利率下降了 1.70pp" —— 这不是分析，这是把同一件事
换个说法讲了两遍。读者问"为什么"，你回答的必须是**业务层面发生了什么**，
数字只是证据。

  ❌ 同义反复："净利润下降，因为毛利率收窄、营收增速放缓"
  ❌ 同义反复："盈利能力下滑，主要由于净利率从 5.18% 降至 4.06%"
  ✅ 业务归因："毛利率收窄 1.70pp 的拆解显示，结构效应仅 +0.17pp、
     各业务自身效应 -1.87pp，说明不是业务结构变化所致，而是各业务
     自身盈利能力在下降；其中汽车业务毛利率下滑 1.82pp、贡献了 -1.19pp，
     是最主要的拖累。分地区看，境内毛利率下滑 3.52pp 而境外提升 1.88pp，
     指向境内市场的价格竞争。"

写归因时按这个顺序推进：
  发生了什么（现象+数字）→ 拆到哪一块（分部/成本项/费用项）
  → 业务上意味着什么（价格战/结构升级/一次性因素/投入期）
  → 有多确定（数据能证实到哪一步，哪一步只是推测）

## 区分主因与次因
不要把所有因素平铺罗列。必须按**对目标指标的贡献大小**排序，
并明确写出"主因是 X（贡献 A）、次因是 Y（贡献 B）"。
贡献无法量化时，说明排序依据是什么。

## 写作要求
- 只输出「详细分析」这一部分的 Markdown 内容，不要写标题行、核心结论、数据来源表格
  （这些由程序生成）。
- 用三级标题（###）组织小节，小节名从内容出发，不要生搬模板。
- **不要**在数字后面写「（来源：…）」或「（来源：某某报表推算）」。来源由程序在文末表格给出。
- 引用 ⚠️ 结论时，直接写数字即可，不要复述门禁内部原因（例如找不到某个 metric）。
- 比率变动用百分点（pp）表达，金额变动用百分比表达。
- 语言平实准确，不用"大幅""显著""强劲"这类没有量化边界的修饰词，
  除非紧跟具体数值。
- 对于只能推测、无法用数据证实的归因，必须写明"该判断为推测，
  现有数据无法证实"，不要和已证实的内容混在一起陈述。
- 篇幅控制在 600~1200 字。

## 本次分析框架
{framework_guidance}
"""

WRITER_TASK_TEMPLATE = """## 用户问题
{question}

## 公司
{company}（{ticker}）

## 分析基准日
{as_of_date}

## 已核查结论（这是你唯一可以使用的信息源）
{claims_block}

## 补充证据材料（可用于组织叙述，但其中的数字若未出现在上面的结论列表中，不得引用）
{evidence_block}

## 已知的数据缺口
{gaps_block}

请撰写「详细分析」部分。"""

DISCLAIMER = (
    "> **免责声明**：本报告由 AI Agent 自动生成，数据来源为公开财报与公开行情（AKShare）。"
    "其中的买卖倾向与价位判断为基于已披露信息的研究观点，不构成持牌投资顾问服务，不保证收益。"
    "标注为「⚠️未验证」的内容未通过数据一致性校验，"
    "标注为「❌拒答」的内容表示现有证据不足以支撑判断。"
    "投资决策请结合自身风险承受能力，并以上市公司正式披露文件为准。"
)

SCOPE_NOTE = (
    "> **口径说明**：金额单位为人民币元，比率以百分数表示；"
    "净利润默认指归属于母公司股东的净利润；报表数据均为合并报表口径；"
    "资产周转率与权益乘数使用期末值计算（非期初期末平均值），"
    "在资产规模快速变化时存在偏差。"
)

# 正文数字审计：只抓带单位的财务数字，年份/股票代码靠后续过滤排除。
_BODY_NUMBER_RE = re.compile(
    r"(\d[\d,]*\.?\d*)\s*(亿元|万亿|万元|亿|个百分点|pp|%|次|倍)"
)
_INLINE_SOURCE_RE = re.compile(
    r"[（(](?:来源|线索来源|出处)[：:][^）)]{1,120}[）)]"
)
_AUDIT_MAJORITY_TIP = "详细分析中超过半数数字未经门禁校验"
_AUDIT_MAX_ITEMS = 8
_BLOCKED_PLACEHOLDER = "〔已拦截：无门禁依据〕"
_UNVERIFIED_MARK = "（⚠️未验证）"
_UNVERIFIED_SPAN_OPEN = '<span class="num-unverified" title="未通过官方数据校验">'
_UNVERIFIED_SPAN_CLOSE = "</span>"
_YI = 100_000_000.0
_WAN = 10_000.0
_WAN_YI = 1_000_000_000_000.0


class ReportOutput:
    """研报输出。同时提供 Markdown 和结构化两种形态。"""

    def __init__(
        self,
        markdown: str,
        title: str,
        conclusions: list[dict[str, Any]],
        analysis_body: str,
        source_table: list[dict[str, Any]],
        stats: dict[str, Any],
        refused: bool = False,
    ):
        self.markdown = markdown
        self.title = title
        self.conclusions = conclusions
        self.analysis_body = analysis_body
        self.source_table = source_table
        self.stats = stats
        self.refused = refused

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "markdown": self.markdown,
            "conclusions": self.conclusions,
            "analysis_body": self.analysis_body,
            "source_table": self.source_table,
            "stats": self.stats,
            "refused": self.refused,
        }


class WriterAgent:
    """研报撰写 Agent。"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        llm_router: LLMRouter | None = None,
        tracer: Tracer | None = None,
    ):
        if llm_client is None:
            router = llm_router or LLMRouter()
            llm_client = router.get("quick")
        self.llm = llm_client
        self.tracer = tracer or NullTracer()

    # ---------------- 主入口 ----------------

    def write(
        self,
        question: str,
        company: str,
        ticker: str,
        as_of_date: date,
        verify_result: Any,
        evidence_pool: Sequence[Evidence],
        analysis_framework: str = "general",
        extra_notes: Sequence[str] = (),
        out_of_scope_reason: str = "",
    ) -> ReportOutput:
        """生成研报。

        verify_result 用 Any 而不是 VerifyResult：避免 agents 之间产生
        导入依赖（writer 不需要知道 verifier 的实现细节，只需要
        claims / coverage_gaps 这两个属性）。
        """
        claims = self._dedupe_claims(list(getattr(verify_result, "claims", []) or []))
        coverage_gaps = list(getattr(verify_result, "coverage_gaps", []) or [])
        data_date = self._latest_data_date(evidence_pool)

        with self.tracer.span(
            "node_enter", "writer", input_summary=f"{len(claims)} 条断言"
        ) as span:
            if out_of_scope_reason:
                report = self._build_out_of_scope_report(
                    question, company, ticker, as_of_date, out_of_scope_reason
                )
                span.add_metadata(out_of_scope=True)
                span.set_output("超出职责范围，输出范围拒答")
                return report

            verified = [c for c in claims if getattr(c, "verdict", "") == "verified"]

            if not verified:
                # 一条已验证结论都没有 → 输出拒答研报。
                # 这不是失败，而是系统应有的行为：查不到就明确说查不到。
                report = self._build_refusal_report(
                    question, company, ticker, as_of_date, data_date, claims, coverage_gaps
                )
                span.add_metadata(refused=True, claim_count=len(claims))
                span.set_output("无已验证结论，输出拒答研报")
                return report

            analysis_body = self._generate_analysis(
                question, company, ticker, as_of_date, claims, evidence_pool,
                coverage_gaps, analysis_framework,
            )
            report = self._assemble(
                question, company, ticker, as_of_date, data_date,
                claims, evidence_pool, analysis_body, coverage_gaps, extra_notes,
            )
            span.add_metadata(
                claim_count=len(claims),
                verified_count=len(verified),
                body_length=len(analysis_body),
            )
            span.set_output(f"研报生成完成，{len(report.markdown)} 字符")
            return report

    # ---------------- LLM 正文生成 ----------------

    def _generate_analysis(
        self,
        question: str,
        company: str,
        ticker: str,
        as_of_date: date,
        claims: Sequence[Any],
        evidence_pool: Sequence[Evidence],
        coverage_gaps: Sequence[str],
        analysis_framework: str,
    ) -> str:
        system_prompt = WRITER_SYSTEM_PROMPT.format(
            framework_guidance=FRAMEWORK_GUIDANCE.get(
                analysis_framework, FRAMEWORK_GUIDANCE["general"]
            )
        )
        task = WRITER_TASK_TEMPLATE.format(
            question=question,
            company=company or "（未指定）",
            ticker=ticker or "（未指定）",
            as_of_date=as_of_date.isoformat(),
            claims_block=self._format_claims_for_prompt(claims),
            evidence_block=self._format_evidence_for_prompt(evidence_pool),
            gaps_block="\n".join(f"- {gap}" for gap in coverage_gaps) or "- （无）",
        )

        response = self.llm.chat(
            messages=[{"role": "user", "content": task}],
            system_prompt=system_prompt,
        )
        if response.is_error or not response.content.strip():
            logger.error("Writer 正文生成失败: %s", response.error)
            # 降级为程序渲染的条目式正文。信息完整性优先于可读性——
            # 一份读起来生硬但数据准确的研报，好过没有研报。
            return self._render_fallback_body(claims)
        return response.content.strip()

    @staticmethod
    def _format_claims_for_prompt(claims: Sequence[Any]) -> str:
        if not claims:
            return "（无已核查结论）"
        lines: list[str] = []
        for index, claim in enumerate(claims, start=1):
            label = getattr(claim, "label", "⚠️未验证")
            source = getattr(claim, "source", None) or "无来源"
            numbers = getattr(claim, "numbers", {}) or {}
            number_text = (
                "，涉及数字: " + ", ".join(f"{k}={v!r}" for k, v in numbers.items())
                if numbers
                else ""
            )
            lines.append(
                f"{index}. [{label}] {getattr(claim, 'text', '')}\n"
                f"   来源: {source} | 归属小节: {getattr(claim, 'section', 'conclusion')}"
                f"{number_text}\n"
                f"   核查说明: {getattr(claim, 'reason', '')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_evidence_for_prompt(evidence_pool: Sequence[Evidence], limit: int = 12) -> str:
        if not evidence_pool:
            return "（无）"
        # 知识库证据排在前面：它提供的是分析框架和行业常识，
        # 对组织叙述最有帮助，而数字类证据已经在结论列表里给过了
        ordered = sorted(
            evidence_pool, key=lambda e: 0 if e.source_type == "knowledge_base" else 1
        )
        return "\n\n".join(
            f"[{evidence.source_name}] {evidence.content[:400]}" for evidence in ordered[:limit]
        )

    @staticmethod
    def _render_fallback_body(claims: Sequence[Any]) -> str:
        sections: dict[str, list[str]] = {}
        for claim in claims:
            section = getattr(claim, "section", "conclusion")
            label = getattr(claim, "label", "⚠️未验证")
            source = getattr(claim, "source", None) or "无来源"
            sections.setdefault(section, []).append(
                f"- [{label}] {getattr(claim, 'text', '')}（来源：{source}）"
            )
        section_titles = {
            "revenue": "收入结构",
            "profitability": "盈利能力",
            "risk": "风险提示",
            "conclusion": "其他结论",
        }
        blocks = ["> 说明：本节由程序直接渲染（正文生成环节不可用），内容为已核查结论的原样罗列。"]
        for key, title in section_titles.items():
            if key in sections:
                blocks.append(f"### {title}\n" + "\n".join(sections[key]))
        return "\n\n".join(blocks)

    # ---------------- 确定性区块渲染 ----------------

    def _assemble(
        self,
        question: str,
        company: str,
        ticker: str,
        as_of_date: date,
        data_date: str,
        claims: Sequence[Any],
        evidence_pool: Sequence[Evidence],
        analysis_body: str,
        coverage_gaps: Sequence[str],
        extra_notes: Sequence[str],
    ) -> ReportOutput:
        title = f"{company or ticker} — {self._topic(question)}"
        conclusions = self._build_conclusions(claims, evidence_pool)
        source_table = self._build_source_table(conclusions, evidence_pool)
        gated_body, flagged, blocked = self._enforce_gate_on_body(analysis_body, claims)

        parts: list[str] = [
            f"# {title}",
            f"> 分析日期：{as_of_date.isoformat()} | 数据截至：{data_date}",
            "",
            "## 核心结论",
            self._render_conclusions(conclusions),
            "",
            "## 详细分析",
            gated_body,
        ]

        if coverage_gaps:
            parts.extend(
                [
                    "",
                    "## 数据缺口",
                    "以下内容因数据缺失未能覆盖，相关判断不成立：",
                    "\n".join(f"- {gap}" for gap in coverage_gaps),
                ]
            )

        parts.extend(["", "## 数据来源", self._render_source_table(source_table)])

        if extra_notes:
            parts.extend(["", "## 备注", "\n".join(f"- {note}" for note in extra_notes)])

        audit_list = self._audit_notes(flagged, blocked, analysis_body)
        if audit_list:
            parts.extend(["", self._render_number_audit(audit_list)])

        parts.extend(["", SCOPE_NOTE, "", DISCLAIMER])

        unmatched_count = sum(1 for item in audit_list if item != _AUDIT_MAJORITY_TIP)
        stats = {
            "total_claims": len(claims),
            "verified": sum(1 for c in conclusions if c["verdict"] == "verified"),
            "unverified": sum(1 for c in conclusions if c["verdict"] == "unverified"),
            "refused": sum(1 for c in conclusions if c["verdict"] == "refused"),
            "evidence_count": len(evidence_pool),
            "data_date": data_date,
            "unverified_numbers_in_body": unmatched_count,
            "blocked_numbers_in_body": len(blocked),
            "body_audit_triggered": bool(audit_list),
        }

        return ReportOutput(
            markdown="\n".join(parts),
            title=title,
            conclusions=conclusions,
            analysis_body=gated_body,
            source_table=source_table,
            stats=stats,
            refused=False,
        )

    def _enforce_gate_on_body(
        self, analysis_body: str, claims: Sequence[Any]
    ) -> tuple[str, list[str], list[str]]:
        """把正文里的数字按门禁结果改写，避免未校验数字以「事实」面貌出现。

        已验证 → 原样保留；
        仅出现在 ⚠️ 结论里 → 数字保留并用样式标成未验证；
        结论列表里没有 → 从所在分句删除，不插入「已拦截」占位。
        """
        if not analysis_body:
            return analysis_body, [], []
        cleaned = _INLINE_SOURCE_RE.sub("", analysis_body)
        verified_claims = [c for c in claims if getattr(c, "verdict", "") == "verified"]
        unverified_claims = [c for c in claims if getattr(c, "verdict", "") == "unverified"]
        if not verified_claims and not unverified_claims:
            return cleaned, [], []

        verified_values, verified_forms = self._number_index(verified_claims)
        unverified_values, unverified_forms = self._number_index(unverified_claims)

        out: list[str] = []
        cursor = 0
        flagged: list[str] = []
        blocked: list[str] = []
        seen_flagged: set[str] = set()
        seen_blocked: set[str] = set()

        for match in _BODY_NUMBER_RE.finditer(cleaned):
            num_str, unit = match.group(1), match.group(2)
            if self._is_ignored_token(num_str):
                continue
            if self._already_gate_marked(cleaned, match.end()):
                continue
            display = f"{num_str}{unit}"
            out.append(cleaned[cursor : match.start()])
            if self._extracted_matches(num_str, unit, verified_forms, verified_values):
                out.append(match.group(0))
            elif self._extracted_matches(num_str, unit, unverified_forms, unverified_values):
                out.append(f"{_UNVERIFIED_SPAN_OPEN}{match.group(0)}{_UNVERIFIED_SPAN_CLOSE}")
                if display not in seen_flagged:
                    seen_flagged.add(display)
                    flagged.append(display)
            else:
                out.append(_BLOCKED_PLACEHOLDER)
                if display not in seen_blocked:
                    seen_blocked.add(display)
                    blocked.append(display)
            cursor = match.end()

        out.append(cleaned[cursor:])
        polished = self._drop_blocked_clauses("".join(out))
        return polished, flagged, blocked

    @staticmethod
    def _dedupe_claims(claims: Sequence[Any]) -> list[Any]:
        """同一组数字或同一句话只保留一条；已验证优先于未验证。"""
        rank = {"verified": 0, "unverified": 1, "refused": 2}
        best: dict[str, Any] = {}
        order: list[str] = []
        for claim in claims:
            fingerprint = WriterAgent._claim_fingerprint(claim)
            previous = best.get(fingerprint)
            if previous is None:
                best[fingerprint] = claim
                order.append(fingerprint)
                continue
            prev_rank = rank.get(getattr(previous, "verdict", ""), 9)
            new_rank = rank.get(getattr(claim, "verdict", ""), 9)
            if new_rank < prev_rank:
                best[fingerprint] = claim
            elif new_rank == prev_rank:
                old_text = getattr(previous, "text", "") or ""
                new_text = getattr(claim, "text", "") or ""
                if len(new_text) > len(old_text):
                    best[fingerprint] = claim
        return [best[key] for key in order]

    @staticmethod
    def _claim_fingerprint(claim: Any) -> str:
        numbers = getattr(claim, "numbers", None) or {}
        if isinstance(numbers, dict) and numbers:
            parts = [
                f"{key}={round(float(value), 6)}"
                for key, value in sorted(numbers.items())
                if isinstance(value, (int, float))
            ]
            if parts:
                return "n:" + "|".join(parts)
        text = re.sub(r"\s+", "", getattr(claim, "text", "") or "")
        return "t:" + text

    @staticmethod
    def _drop_blocked_clauses(text: str) -> str:
        """含拦截占位的分句整段删除，避免正文出现「毛利率为〔已拦截〕」。"""
        if _BLOCKED_PLACEHOLDER not in text:
            return text
        rebuilt_lines: list[str] = []
        for line in text.split("\n"):
            if _BLOCKED_PLACEHOLDER not in line:
                rebuilt_lines.append(line)
                continue
            if line.lstrip().startswith("#"):
                rebuilt_lines.append(line.replace(_BLOCKED_PLACEHOLDER, "").strip())
                continue
            pieces = re.split(r"(?<=[。！？])", line)
            kept_sentences: list[str] = []
            for sentence in pieces:
                if not sentence:
                    continue
                if _BLOCKED_PLACEHOLDER not in sentence:
                    kept_sentences.append(sentence)
                    continue
                filtered = WriterAgent._filter_blocked_sentence(sentence)
                if filtered:
                    kept_sentences.append(filtered)
            rebuilt_lines.append("".join(kept_sentences).strip())
        cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(rebuilt_lines))
        return cleaned.strip()

    @staticmethod
    def _filter_blocked_sentence(sentence: str) -> str:
        ending = ""
        core = sentence
        if core and core[-1] in "。！？":
            ending = core[-1]
            core = core[:-1]
        parts = re.split(r"[，；;]", core)
        kept = [part.strip() for part in parts if _BLOCKED_PLACEHOLDER not in part and part.strip()]
        if not kept:
            return ""
        joined = "，".join(kept)
        if not _BODY_NUMBER_RE.search(joined) and "num-unverified" not in joined:
            return ""
        if ending:
            return joined + ending
        return joined

    @staticmethod
    def strip_report_html(markdown: str) -> str:
        """CLI 打印时去掉正文里的校验标记标签，保留数字本身。"""
        text = re.sub(r'<span class="num-unverified"[^>]*>', "", markdown)
        return text.replace(_UNVERIFIED_SPAN_CLOSE, "")

    def _number_index(self, claims: Sequence[Any]) -> tuple[list[float], set[str]]:
        values = [value for claim in claims for value in self._claim_number_values(claim)]
        forms: set[str] = set()
        for value in values:
            forms.update(self._expand_verified_forms(value))
        for claim in claims:
            text = getattr(claim, "text", "") or ""
            for num_str, unit in _BODY_NUMBER_RE.findall(text):
                if self._is_ignored_token(num_str):
                    continue
                forms.add(f"{num_str}{unit}")
                forms.add(f"{num_str.replace(',', '')}{unit}")
        return values, forms

    @staticmethod
    def _already_gate_marked(text: str, end: int) -> bool:
        tail = text[end : end + 80]
        return (
            "⚠️" in tail
            or "未验证" in tail
            or "已拦截" in tail
            or "num-unverified" in tail
        )

    def _audit_analysis_numbers(
        self, analysis_body: str, claims: Sequence[Any]
    ) -> list[str]:
        """扫描正文中未能对上已验证结论的数字（单测与审计备注共用）。"""
        if not analysis_body or not claims:
            return []
        _, flagged, blocked = self._enforce_gate_on_body(analysis_body, claims)
        unmatched = flagged + blocked
        if not unmatched:
            return []
        result = unmatched[:_AUDIT_MAX_ITEMS]
        total = len(_BODY_NUMBER_RE.findall(analysis_body))
        if total > 0 and len(unmatched) / total > 0.5:
            result.append(_AUDIT_MAJORITY_TIP)
        return result

    def _audit_notes(
        self,
        flagged: Sequence[str],
        blocked: Sequence[str],
        original_body: str,
    ) -> list[str]:
        notes = [f"{item}（已标注未验证）" for item in flagged[:_AUDIT_MAX_ITEMS]]
        remain = _AUDIT_MAX_ITEMS - len(notes)
        notes.extend(f"{item}（已拦截）" for item in blocked[: max(remain, 0)])
        if not notes:
            return []
        scanned = [
            m
            for m in _BODY_NUMBER_RE.findall(original_body)
            if not self._is_ignored_token(m[0])
        ]
        if scanned and (len(flagged) + len(blocked)) / len(scanned) > 0.5:
            notes.append(_AUDIT_MAJORITY_TIP)
        return notes

    @staticmethod
    def _render_number_audit(items: Sequence[str]) -> str:
        numbers = [item for item in items if item != _AUDIT_MAJORITY_TIP]
        lines = [
            "## 正文数字审计",
            "> ⚠️ 「详细分析」中下列数字未作为已验证事实写入正文（已省略或标为未验证）：",
            f"> {', '.join(numbers)}。",
            "> 请以「核心结论」区块的三级标注为准。",
        ]
        if _AUDIT_MAJORITY_TIP in items:
            lines.append(f"> {_AUDIT_MAJORITY_TIP}。")
        return "\n".join(lines)

    @staticmethod
    def _claim_number_values(claim: Any) -> list[float]:
        numbers = getattr(claim, "numbers", None) or {}
        if isinstance(numbers, dict):
            return [float(value) for value in numbers.values() if isinstance(value, (int, float))]
        values: list[float] = []
        if isinstance(numbers, list):
            for item in numbers:
                raw = getattr(item, "value", None)
                if raw is None and isinstance(item, dict):
                    raw = item.get("value")
                if isinstance(raw, (int, float)):
                    values.append(float(raw))
        return values

    @staticmethod
    def _is_ignored_token(num_str: str) -> bool:
        compact = num_str.replace(",", "")
        if re.fullmatch(r"(19|20)\d{2}", compact):
            return True
        if re.fullmatch(r"\d{6}", compact):
            return True
        return False

    @staticmethod
    def _expand_verified_forms(value: float) -> set[str]:
        """把 claim 里的元/小数同时展开成正文常见写法。"""
        forms: set[str] = set()

        def add_scaled(magnitude: float, suffix: str) -> None:
            for digits in (2, 1, 0):
                plain = f"{magnitude:.{digits}f}"
                comma = f"{magnitude:,.{digits}f}"
                forms.add(plain + suffix)
                forms.add(comma + suffix)

        if abs(value) >= _WAN:
            add_scaled(value / _YI, "亿")
            add_scaled(value / _YI, "亿元")
            add_scaled(value / _WAN, "万")
            add_scaled(value / _WAN, "万元")
            add_scaled(value / _WAN_YI, "万亿")
        if abs(value) <= 5:
            add_scaled(value * 100.0, "%")
            add_scaled(abs(value) * 100.0, "pp")
            add_scaled(abs(value) * 100.0, "个百分点")
        if abs(value) <= 100:
            add_scaled(value, "%")
            add_scaled(abs(value), "pp")
            add_scaled(abs(value), "个百分点")
        return forms

    @staticmethod
    def _extracted_matches(
        num_str: str,
        unit: str,
        forms: set[str],
        values: Sequence[float],
    ) -> bool:
        compact = num_str.replace(",", "")
        try:
            magnitude = float(compact)
        except ValueError:
            return False
        unit_norm = {"亿元": "亿", "万元": "万", "个百分点": "pp"}.get(unit, unit)
        candidates = {
            f"{num_str}{unit}",
            f"{compact}{unit}",
            f"{compact}{unit_norm}",
            f"{magnitude:.2f}{unit_norm}",
            f"{magnitude:,.2f}{unit_norm}",
            f"{magnitude:.2f}{unit}",
            f"{magnitude:,.2f}{unit}",
            f"{magnitude:.1f}{unit_norm}",
            f"{magnitude:.0f}{unit_norm}",
        }
        if candidates & forms:
            return True
        for value in values:
            if WriterAgent._numeric_match(magnitude, unit_norm, value):
                return True
        return False

    @staticmethod
    def _numeric_match(extracted: float, unit: str, value: float) -> bool:
        if unit in {"亿", "亿元"}:
            return round(value / _YI, 2) == round(extracted, 2)
        if unit in {"万", "万元"}:
            return round(value / _WAN, 2) == round(extracted, 2)
        if unit == "万亿":
            return round(value / _WAN_YI, 2) == round(extracted, 2)
        if unit == "%":
            return (
                round(value * 100.0, 2) == round(extracted, 2)
                or round(value, 2) == round(extracted, 2)
            )
        if unit == "pp":
            return (
                round(abs(value) * 100.0, 2) == round(extracted, 2)
                or round(abs(value), 2) == round(extracted, 2)
            )
        return False

    def _build_conclusions(
        self, claims: Sequence[Any], evidence_pool: Sequence[Evidence] = ()
    ) -> list[dict[str, Any]]:
        """核心结论清单。排序：已验证 → 未验证 → 拒答。

        排序不是为了好看：读者从上往下读时应该先看到最可靠的信息，
        而不是被一堆 ⚠️ 淹没后才找到有依据的结论。
        """
        order = {"verified": 0, "unverified": 1, "refused": 2}
        unique_claims = self._dedupe_claims(list(claims))
        items = [
            {
                "verdict": getattr(claim, "verdict", "unverified"),
                "label": getattr(claim, "label", "⚠️未验证"),
                "text": getattr(claim, "text", ""),
                "source": self._resolve_citation_source(claim, evidence_pool),
                "reason": getattr(claim, "reason", ""),
                "section": getattr(claim, "section", "conclusion"),
                "gate_status": getattr(claim, "gate_status", "unverified"),
            }
            for claim in unique_claims
        ]
        items.sort(key=lambda item: order.get(item["verdict"], 3))
        return items

    @staticmethod
    def _render_conclusions(conclusions: Sequence[dict[str, Any]]) -> str:
        if not conclusions:
            return "- [❌拒答] 未能形成任何可核查的结论。"
        lines: list[str] = []
        for item in conclusions:
            # 来源和门禁内部原因放到文末表格 / title，不在核心结论里重复堆括号。
            lines.append(f"- [{item['label']}] {item['text']}")
        return "\n".join(lines)

    @staticmethod
    def _render_unverified_suffix(source: str | None, reason: str) -> str:
        """⚠️ 行同时给出线索来源和门禁未通过的原因。"""
        if source and reason:
            if source in reason:
                return f"（线索来源：{source}）"
            return f"（线索来源：{source}；未通过官方校验：{reason}）"
        if source:
            return f"（线索来源：{source}）"
        if reason:
            return f"（{reason}）"
        return ""

    @classmethod
    def _resolve_citation_source(
        cls, claim: Any, evidence_pool: Sequence[Evidence]
    ) -> str | None:
        """给 ⚠️ 结论找回新闻/研报线索，避免把门禁失败原因当成唯一出处。

        数字对不上官方报表时，Gate 常把 matched_source 回退成年报；
        那会让「2026 中报数字」看起来像来自 2025 年报。找不到官方覆盖时，
        只引用新闻/研报/知识库，否则不写来源。
        """
        source = getattr(claim, "source", None) or None
        reason = getattr(claim, "reason", "") or ""
        verdict = getattr(claim, "verdict", "")
        if verdict == "verified":
            return source

        by_name = {evidence.source_name: evidence for evidence in evidence_pool}
        attached = by_name.get(source) if source else None
        if attached and attached.source_type in {"news", "research_report", "knowledge_base"}:
            return attached.source_name

        clue = cls._match_clue_evidence(claim, evidence_pool)
        if clue is not None:
            return clue.source_name

        if "找不到" in reason:
            return None
        return source

    @classmethod
    def _match_clue_evidence(
        cls, claim: Any, evidence_pool: Sequence[Evidence]
    ) -> Evidence | None:
        clues = [
            evidence
            for evidence in evidence_pool
            if evidence.source_type in {"news", "research_report", "knowledge_base"}
        ]
        if not clues:
            return None

        needles: list[str] = []
        text = getattr(claim, "text", "") or ""
        for num_str, _unit in _BODY_NUMBER_RE.findall(text):
            compact = num_str.replace(",", "")
            if len(compact) >= 4:
                needles.append(compact)
        for value in cls._claim_number_values(claim):
            if abs(value) >= _WAN:
                rendered = f"{value / _YI:.2f}"
                if rendered != "0.00":
                    needles.append(rendered)
        phrases = re.findall(r"[\u4e00-\u9fff]{4,}", text)

        best: Evidence | None = None
        best_score = 0
        for evidence in clues:
            blob = f"{evidence.source_name}\n{evidence.content}".replace(",", "")
            score = sum(1 for needle in needles if needle in blob)
            score += sum(1 for phrase in phrases if phrase in blob)
            if score > best_score:
                best_score = score
                best = evidence
        return best if best_score >= 1 else None

    def _build_source_table(
        self, conclusions: Sequence[dict[str, Any]], evidence_pool: Sequence[Evidence]
    ) -> list[dict[str, Any]]:
        """数据来源表。与核心结论共用去重后的清单。"""
        evidence_by_name = {evidence.source_name: evidence for evidence in evidence_pool}
        rows: list[dict[str, Any]] = []
        for item in conclusions:
            source = item.get("source")
            evidence = evidence_by_name.get(source) if source else None
            rows.append(
                {
                    "item": (item.get("text") or "")[:60],
                    "source": source or "无来源",
                    "tier": evidence.tier_label if evidence else "—",
                    "disclosure_date": self._format_disclosure_date(evidence),
                    "status": item.get("label", "⚠️未验证"),
                }
            )
        return rows

    @staticmethod
    def _format_disclosure_date(evidence: Evidence | None) -> str:
        """格式化披露日期。

        知识库证据用 date.min 表示"没有披露时点概念"（这样它永远不会被
        前视偏差拦截），但在研报里渲染成 0001-01-01 会让读者困惑，
        所以单独显示成"通用知识"。
        """
        if evidence is None:
            return "—"
        if evidence.disclosure_date == date.min:
            return "通用知识"
        return evidence.disclosure_date.isoformat()

    @staticmethod
    def _render_source_table(rows: Sequence[dict[str, Any]]) -> str:
        if not rows:
            return "（本次分析未引用任何数据来源）"
        header = (
            "| 数据项 | 来源 | 可信度层级 | 披露日期 | 状态 |\n"
            "|--------|------|-----------|---------|------|"
        )
        body = "\n".join(
            f"| {row['item']} | {row['source']} | {row.get('tier', '—')} | "
            f"{row['disclosure_date']} | {row['status']} |"
            for row in rows
        )
        legend = (
            "\n\n**来源层级说明**：\n"
            "- ① 已披露事实（年报/季报/公告/行情）：上市公司依法披露或市场公开数据，可作为核心依据；\n"
            "- ② 机构判断（券商研报/分析框架）：有署名有方法论的专业观点，"
            "但是**判断不是事实**，最高只能标注为「⚠️未验证」；\n"
            "- ③ 待验证信息（新闻媒体）：仅作观察线索，需结合最新披露验证，不得写成确定结论。"
        )
        return f"{header}\n{body}{legend}"

    # ---------------- 拒答 ----------------

    def _build_out_of_scope_report(
        self,
        question: str,
        company: str,
        ticker: str,
        as_of_date: date,
        reason: str,
    ) -> ReportOutput:
        """职责范围拒答：问题本身超出系统能力边界，与数据是否充分无关。

        与"证据不足"型拒答分开，因为两者对用户的含义完全不同：
        前者是"这个问题我不该回答"，后者是"这个问题我答不了"。
        混为一谈会让用户以为补充数据就能得到买卖建议。
        """
        title = f"{company or ticker or '未指定标的'} — {self._topic(question)}"
        parts = [
            f"# {title}",
            f"> 分析日期：{as_of_date.isoformat()}",
            "",
            "## 核心结论",
            "- [❌拒答] 该问题超出本系统的职责范围，不予回答。",
            "",
            "## 原因",
            f"- {reason}",
            "",
            "## 本系统能做什么",
            "- 提取并核查已披露的财务数据（营收、利润、毛利率、ROE、现金流等）",
            "- 做跨期对比与趋势判断，区分「增速放缓」与「绝对值下降」",
            "- 做归因分析：拆解指标变动的驱动因子（杜邦分解、成本结构）",
            "- 执行风险规则检查（应收背离、现金流覆盖、杠杆、商誉减值）",
            "- 每条结论标注证据等级：✅已验证 / ⚠️未验证 / ❌拒答",
            "",
            "## 可以这样改写你的问题",
            f"- 「{company or ticker} 最近一期的营收和净利润是多少」（单点事实）",
            f"- 「{company or ticker} 近三年毛利率的变化趋势」（跨期对比）",
            f"- 「{company or ticker} 净利润下降的原因是什么」（归因分析）",
            f"- 「{company or ticker} 有哪些财务风险点」（风险评估）",
            "",
            DISCLAIMER,
        ]
        return ReportOutput(
            markdown="\n".join(parts),
            title=title,
            conclusions=[
                {
                    "verdict": "refused",
                    "label": VERDICT_LABEL["refused"],
                    "text": "该问题超出本系统的职责范围，不予回答。",
                    "source": None,
                    "reason": reason,
                    "section": "conclusion",
                    "gate_status": "out_of_scope",
                }
            ],
            analysis_body="",
            source_table=[],
            stats={
                "total_claims": 0,
                "verified": 0,
                "unverified": 0,
                "refused": 1,
                "evidence_count": 0,
                "data_date": "—",
                "refusal_type": "out_of_scope",
                "unverified_numbers_in_body": 0,
                "body_audit_triggered": False,
            },
            refused=True,
        )

    def _build_refusal_report(
        self,
        question: str,
        company: str,
        ticker: str,
        as_of_date: date,
        data_date: str,
        claims: Sequence[Any],
        coverage_gaps: Sequence[str],
    ) -> ReportOutput:
        title = f"{company or ticker or '未识别标的'} — {self._topic(question)}"
        conclusions = self._build_conclusions(claims)

        reasons: list[str] = list(coverage_gaps)
        if not claims:
            reasons.append("未能从现有证据中抽取出任何可核查的结论")
        blocked = [
            f"「{getattr(c, 'text', '')[:50]}」被拦截：{getattr(c, 'reason', '')}"
            for c in claims
            if getattr(c, "verdict", "") == "refused"
        ]
        reasons.extend(blocked)

        parts = [
            f"# {title}",
            f"> 分析日期：{as_of_date.isoformat()} | 数据截至：{data_date}",
            "",
            "## 核心结论",
            "- [❌拒答] 现有证据不足以回答该问题，无法给出可靠结论。",
            "",
            "## 无法回答的原因",
            "\n".join(f"- {reason}" for reason in reasons) or "- 未获取到任何可用数据",
            "",
            "## 已尝试但未通过核查的内容",
            self._render_conclusions(conclusions) if conclusions else "（无）",
            "",
            "## 建议",
            "\n".join(f"- {item}" for item in self._suggestions_for(claims, coverage_gaps)),
            "",
            DISCLAIMER,
        ]

        return ReportOutput(
            markdown="\n".join(parts),
            title=title,
            conclusions=conclusions,
            analysis_body="",
            source_table=[],
            stats={
                "total_claims": len(claims),
                "verified": 0,
                "unverified": sum(1 for c in conclusions if c["verdict"] == "unverified"),
                "refused": sum(1 for c in conclusions if c["verdict"] == "refused"),
                "evidence_count": 0,
                "data_date": data_date,
                "unverified_numbers_in_body": 0,
                "body_audit_triggered": False,
            },
            refused=True,
        )

    @staticmethod
    def _suggestions_for(
        claims: Sequence[Any], coverage_gaps: Sequence[str]
    ) -> list[str]:
        """按实际失败原因给建议，而不是套一段固定模板。

        之前无论什么原因都输出"确认股票代码是否正确"，
        但实测那次失败里代码完全正确（600519），失败原因是断言被门禁拦截。
        给出与现实不符的建议，比不给建议更糟——它会把用户引向错误的排查方向。
        """
        statuses = [getattr(claim, "gate_status", "") for claim in claims]
        suggestions: list[str] = []

        if not claims:
            suggestions.append("本次未能从数据中提取出任何可核查的结论，请确认股票代码是否正确（A股为 6 位数字）")
            suggestions.append("确认该公司是否已披露相关报告期的财务数据")
        if "number_mismatch" in statuses:
            suggestions.append(
                "部分结论因数字与来源不一致被拦截。这通常意味着分析环节引用了"
                "错误的指标或错误的报告期，可查看 trace 中的 gate_check 事件定位具体差异"
            )
        if "lookahead_blocked" in statuses:
            suggestions.append(
                "部分结论引用了分析基准日之后才披露的数据（前视偏差）。"
                "可把分析基准日调整到相应报告的披露日之后再试"
            )
        if "unverified" in statuses:
            suggestions.append(
                "部分结论找不到官方披露来源支撑。若问题依赖新闻或市场传闻，"
                "本系统无法将其作为已验证事实输出"
            )
        if coverage_gaps:
            suggestions.append(
                "存在数据缺口（见上文），可尝试把问题聚焦到已覆盖的维度："
                "财务指标、跨期趋势、风险规则检查"
            )

        return suggestions or ["请尝试换一种问法，或把问题拆分成更具体的子问题"]

    # ---------------- 工具方法 ----------------

    @staticmethod
    def _topic(question: str) -> str:
        """从问题里提炼研报副标题。"""
        topic = question.strip().rstrip("？?。.")
        return topic if len(topic) <= 40 else topic[:40] + "…"

    @staticmethod
    def _latest_data_date(evidence_pool: Sequence[Evidence]) -> str:
        """数据截至日 = 所有证据里最新的报告期。

        用报告期而不是披露日期：读者关心的是"这份研报用的是哪一期的数据"，
        而不是"这些数据什么时候被公布"。
        """
        periods = [e.period_end for e in evidence_pool if e.period_end is not None]
        return max(periods).isoformat() if periods else "—"


__all__ = ["VERDICT_LABEL", "ReportOutput", "WriterAgent"]
