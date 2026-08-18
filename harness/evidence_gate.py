"""证据门禁：LLM 输出进入研报前的最后一道拦截。

解决什么问题
    金融场景里 LLM 的三类致命错误：
    (1) 编造数据——"2024 年营收 8000 亿"，但没有任何来源支持；
    (2) 抄错数字——来源写 7771 亿，输出写成 7717 亿；
    (3) 前视偏差——用 2025-03 才披露的年报去支撑"截至 2024-06-30 的判断"，
        这在回测和投研里是能让整套结论作废的错误。
    通用的 RAG 引用检查只解决 (1)，本模块把三者都拦下来。

核心设计决策
    1. 三级串行门禁（来源 → 数字 → 时点），任一级不过就给出**具体的失败类型**，
       而不是笼统的 "not verified"。失败类型直接决定研报里的标注：
       verified → ✅，unverified → ⚠️，number_mismatch / lookahead_blocked → ❌拒答。
    2. 弃权语义借鉴 ai-hedge-fund：abstain ≠ neutral。
       "没有证据支持"不等于"证据表明不成立"，更不等于"中性判断"。
       所以 GateResult 没有 True/False，只有四种状态，调用方无法把
       "查不到"误读成"没问题"。
    3. 数字容差按语义分三档而不是一刀切 5%：
       - 绝对值（营收/净利润）：5%，容忍"7771 亿 vs 777.1 亿元"这类单位换算和四舍五入；
       - 比率（毛利率/ROE）：0.5 个百分点，因为 21.8% 和 22.9% 相差 5% 相对值，
         但在投研上是完全不同的结论；
       - 同比增速：1 个百分点，同理。
       用相对误差统一处理比率会让门禁形同虚设，这是最容易被面试官追问的点。
    4. 会计口径一致性作为第四级检查（md 的金融业务规则要求）：
       合并报表与母公司报表不能混用，"净利润"默认指归母净利润。
       口径混用产生的数字看起来都"对得上量级"，人工极难发现，必须由程序拦。

为什么不用其他方案
    - 不用"再让一个 LLM 判断这句话有没有依据"：那是用非确定性系统校验
      非确定性系统，幻觉会叠加。数字比对和日期比较是确定性的，就该用代码做。
      LLM 只负责把自然语言里的数字抽出来（agents/verifier.py 的职责），
      抽出来之后的判定全部交给这里的确定性规则。
    - 不用简单的子串匹配（"7771" 在原文里出现过就算通过）：
      原文里出现 7771 可能是另一个指标的值，还会漏掉单位换算的情况。
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Iterable, Sequence

from config import get_config
from harness.tracing import NullTracer, Tracer
from harness.types import (
    GATE_LOOKAHEAD_BLOCKED,
    GATE_NUMBER_MISMATCH,
    GATE_UNVERIFIED,
    GATE_VERIFIED,
    Claim,
    ClaimNumber,
    Evidence,
    GateCheckDetail,
    GateResult,
)

logger = logging.getLogger(__name__)

# A 股法定披露截止日（月, 日）。用于判断"这个报告期的数据在分析日是否应该已经公开"。
# 依据：《上市公司信息披露管理办法》——年报次年 4/30、中报当年 8/31、
# 一季报当年 4/30、三季报当年 10/31。
DISCLOSURE_DEADLINES: dict[str, tuple[int, int, int]] = {
    # 报告期(月-日) -> (相对年份偏移, 截止月, 截止日)
    "12-31": (1, 4, 30),  # 年报：次年 4 月 30 日
    "06-30": (0, 8, 31),  # 中报：当年 8 月 31 日
    "03-31": (0, 4, 30),  # 一季报：当年 4 月 30 日
    "09-30": (0, 10, 31),  # 三季报：当年 10 月 31 日
}

# 被视为"比率"的指标名片段（值域为 0~1 的小数）
RATIO_METRIC_HINTS = (
    "margin",
    "ratio",
    "roe",
    "roa",
    "rate",
    "率",
    "占比",
)
# 被视为"增速"的指标名片段
GROWTH_METRIC_HINTS = ("yoy", "qoq", "growth", "增速", "同比", "环比")


def infer_number_kind(metric: str) -> str:
    """按指标名推断数字语义，决定用哪档容差。"""
    lowered = metric.lower()
    if any(hint in lowered for hint in GROWTH_METRIC_HINTS):
        return "growth"
    if any(hint in lowered for hint in RATIO_METRIC_HINTS):
        return "ratio"
    return "absolute"


def disclosure_deadline(period_end: date) -> date | None:
    """给定报告期，返回法定披露截止日。非标准报告期返回 None。"""
    key = f"{period_end.month:02d}-{period_end.day:02d}"
    rule = DISCLOSURE_DEADLINES.get(key)
    if rule is None:
        return None
    year_offset, month, day = rule
    return date(period_end.year + year_offset, month, day)


class EvidenceGate:
    """三级（+ 会计口径第四级）证据门禁。"""

    def __init__(
        self,
        number_tolerance: float | None = None,
        ratio_tolerance_pp: float | None = None,
        growth_tolerance_pp: float | None = None,
        tracer: Tracer | None = None,
    ):
        config = get_config()
        self.number_tolerance = (
            number_tolerance if number_tolerance is not None else config.number_tolerance
        )
        self.ratio_tolerance_pp = (
            ratio_tolerance_pp if ratio_tolerance_pp is not None else config.ratio_tolerance_pp
        )
        self.growth_tolerance_pp = (
            growth_tolerance_pp if growth_tolerance_pp is not None else config.growth_tolerance_pp
        )
        self.tracer = tracer or NullTracer()

    # ================= 主入口 =================

    def check(self, claim: Claim, evidence_pool: Sequence[Evidence]) -> GateResult:
        """对单条断言执行门禁检查。

        检查顺序刻意是"来源 → 时点 → 会计口径 → 数字"，而不是 md 里写的
        "来源 → 数字 → 时点"。原因：如果一条证据本身就构成前视偏差，
        那么它的数字对不对已经无关紧要，先做时点过滤能给出更准确的失败归因
        （报 lookahead_blocked 而不是让它先在数字级通过再被时点否掉）。
        对外表现的四种状态与 md 完全一致。
        """
        checks: list[GateCheckDetail] = []

        # --- 第 1 级：来源检查 ---
        candidates = self._find_candidate_evidence(claim, evidence_pool)
        if not candidates:
            checks.append(
                GateCheckDetail(level="source", passed=False, reason="证据池中没有与该断言匹配的来源")
            )
            return self._emit(
                GateResult(
                    claim_id=claim.claim_id,
                    status=GATE_UNVERIFIED,
                    reason="无来源支撑，标注为未验证",
                    checks=checks,
                ),
                claim,
            )
        checks.append(
            GateCheckDetail(
                level="source",
                passed=True,
                reason=f"命中 {len(candidates)} 条候选证据，最高优先级来源: {candidates[0].source_name}",
            )
        )

        # --- 第 2 级：时点检查（前视偏差拦截）---
        timely, lookahead = self._split_by_timing(candidates, claim.as_of_date)
        if not timely:
            blocked = lookahead[0]
            checks.append(
                GateCheckDetail(
                    level="timing",
                    passed=False,
                    reason=(
                        f"仅有的证据 {blocked.source_name} 披露于 "
                        f"{blocked.disclosure_date}，晚于分析基准日 {claim.as_of_date}"
                    ),
                )
            )
            return self._emit(
                GateResult(
                    claim_id=claim.claim_id,
                    status=GATE_LOOKAHEAD_BLOCKED,
                    matched_evidence_id=blocked.evidence_id,
                    matched_source=blocked.source_name,
                    reason="存在前视偏差：引用了分析基准日之后才披露的数据",
                    checks=checks,
                ),
                claim,
            )
        checks.append(
            GateCheckDetail(
                level="timing",
                passed=True,
                reason=f"{len(timely)} 条证据披露日期不晚于 {claim.as_of_date}",
            )
        )

        # --- 第 3 级：会计口径一致性 ---
        scope_matched = [e for e in timely if e.statement_scope == claim.statement_scope]
        if not scope_matched:
            checks.append(
                GateCheckDetail(
                    level="accounting",
                    passed=False,
                    reason=(
                        f"口径不一致：断言按 {claim.statement_scope} 口径，"
                        f"但证据均为 {timely[0].statement_scope} 口径"
                    ),
                )
            )
            return self._emit(
                GateResult(
                    claim_id=claim.claim_id,
                    status=GATE_UNVERIFIED,
                    matched_evidence_id=timely[0].evidence_id,
                    matched_source=timely[0].source_name,
                    reason="合并报表与母公司报表口径混用，无法确认",
                    checks=checks,
                ),
                claim,
            )
        checks.append(GateCheckDetail(level="accounting", passed=True, reason="会计口径一致"))

        # --- 第 4 级：数字检查 ---
        if not claim.numbers:
            # 纯定性断言（"公司处于新能源汽车行业"）没有数字可校验。
            # 此时来源等级决定结论：官方披露 → 已验证；新闻/知识库 → 未验证。
            best = scope_matched[0]
            passed = best.is_official
            checks.append(
                GateCheckDetail(
                    level="number",
                    passed=passed,
                    reason="定性断言无数字，按来源可信度判定"
                    if passed
                    else f"来源类型 {best.source_type} 非官方披露，降级为未验证",
                )
            )
            return self._emit(
                GateResult(
                    claim_id=claim.claim_id,
                    status=GATE_VERIFIED if passed else GATE_UNVERIFIED,
                    matched_evidence_id=best.evidence_id,
                    matched_source=best.source_name,
                    reason="定性断言，来源可信" if passed else "定性断言，来源为非官方渠道",
                    checks=checks,
                ),
                claim,
            )

        number_result = self._check_numbers(claim.numbers, scope_matched)
        checks.append(
            GateCheckDetail(
                level="number", passed=number_result.passed, reason=number_result.reason
            )
        )
        if not number_result.passed:
            status = (
                GATE_NUMBER_MISMATCH if number_result.has_conflict else GATE_UNVERIFIED
            )
            return self._emit(
                GateResult(
                    claim_id=claim.claim_id,
                    status=status,
                    matched_evidence_id=number_result.evidence_id,
                    matched_source=number_result.source_name,
                    reason=number_result.reason,
                    checks=checks,
                    number_diffs=number_result.diffs,
                ),
                claim,
            )

        best_evidence = next(
            (e for e in scope_matched if e.evidence_id == number_result.evidence_id),
            scope_matched[0],
        )
        # 新闻来源即使数字对得上也只能标 ⚠️：媒体转载的数字无法作为一手依据。
        if not best_evidence.is_official:
            return self._emit(
                GateResult(
                    claim_id=claim.claim_id,
                    status=GATE_UNVERIFIED,
                    matched_evidence_id=best_evidence.evidence_id,
                    matched_source=best_evidence.source_name,
                    reason=f"数字一致但来源为 {best_evidence.source_type}（非官方披露），降级为未验证",
                    checks=checks,
                ),
                claim,
            )

        return self._emit(
            GateResult(
                claim_id=claim.claim_id,
                status=GATE_VERIFIED,
                matched_evidence_id=best_evidence.evidence_id,
                matched_source=best_evidence.source_name,
                reason=f"三级门禁全部通过（来源: {best_evidence.citation()}）",
                checks=checks,
            ),
            claim,
        )

    def check_batch(
        self, claims: Iterable[Claim], evidence_pool: Sequence[Evidence]
    ) -> list[GateResult]:
        return [self.check(claim, evidence_pool) for claim in claims]

    def check_financial_claim(self, claim: Claim, evidence_pool: Sequence[Evidence]) -> GateResult:
        """金融事实检查。

        md 把它写成独立方法，但通用三级门禁里已经内置了金融业务规则
        （来源优先级、分档容差、披露截止日、会计口径），拆成两套实现会导致
        规则漂移。这里保留方法名作为语义化别名，额外补一条"披露时点合理性"
        校验：如果某个报告期的数据在法定披露截止日之前就出现了，说明来源可疑。
        """
        result = self.check(claim, evidence_pool)
        if result.status != GATE_VERIFIED or result.matched_evidence_id is None:
            return result
        evidence = next(
            (e for e in evidence_pool if e.evidence_id == result.matched_evidence_id), None
        )
        if evidence is None or evidence.period_end is None:
            return result
        deadline = disclosure_deadline(evidence.period_end)
        if deadline and evidence.disclosure_date > deadline:
            # 晚于法定截止日披露不是造假，只是延迟披露，降级提示而非拦截
            result.checks.append(
                GateCheckDetail(
                    level="timing",
                    passed=True,
                    reason=f"注意：该数据披露于 {evidence.disclosure_date}，晚于法定截止日 {deadline}",
                )
            )
        return result

    # ================= 各级实现 =================

    # 定性断言（无数字）的候选门槛。对应约 27% 的 2-gram 重合度。
    # 这个门槛不能太低：定性断言没有数字可校验，最终判定完全取决于
    # "命中了哪条证据"，一条内容毫不相关但来源等级高的年报如果被选中，
    # 会让"ROE 的定义是什么"这种通用常识被标成 ✅已验证（引用年报）。
    QUALITATIVE_MATCH_THRESHOLD = 8
    NUMERIC_MATCH_THRESHOLD = 30

    def _find_candidate_evidence(
        self, claim: Claim, evidence_pool: Sequence[Evidence]
    ) -> list[Evidence]:
        """按"来源指认 > 指标名命中 > 文本重合"筛选候选并排序。

        排序键刻意是 (是否被断言指名, 匹配度, 来源优先级)，而不是
        (来源优先级, 匹配度)。后者会让"等级最高但内容无关"的证据排在
        "等级低但内容对口"的证据前面——对定性断言尤其致命。
        来源优先级降为最后的平手裁决，而在数字校验环节会重新按优先级取优
        （见 _check_numbers），所以"年报优先于新闻"的语义并没有丢。
        """
        claim_metrics = {number.metric for number in claim.numbers}
        candidates: list[tuple[int, int, int, Evidence]] = []

        for evidence in evidence_pool:
            if claim.ticker and evidence.ticker and claim.ticker != evidence.ticker:
                continue

            hint_matched = int(
                bool(claim.source_hint) and claim.source_hint in evidence.source_name
            )
            match_score = 0
            if claim_metrics and claim_metrics & set(evidence.numbers):
                match_score += 100  # 指标名直接对上，最强信号
            match_score += int(self._text_overlap(claim.text, evidence.content) * 30)

            threshold = (
                self.NUMERIC_MATCH_THRESHOLD if claim_metrics else self.QUALITATIVE_MATCH_THRESHOLD
            )
            if hint_matched or match_score >= threshold:
                candidates.append((hint_matched, match_score, evidence.priority, evidence))

        candidates.sort(key=lambda item: item[:3], reverse=True)
        return [evidence for *_, evidence in candidates]

    @staticmethod
    def _text_overlap(claim_text: str, evidence_text: str) -> float:
        """中文场景下的粗粒度重合度：2-gram 字符集合的 Jaccard 近似。

        中文没有空格分词，按词切需要引入 jieba；这里用 2-gram 近似，
        对"毛利率下降"这类固定搭配的匹配效果足够，且零依赖。
        """
        if not claim_text or not evidence_text:
            return 0.0
        normalized_claim = re.sub(r"\s+", "", claim_text)
        normalized_evidence = re.sub(r"\s+", "", evidence_text)
        if len(normalized_claim) < 2:
            return 0.0
        claim_grams = {normalized_claim[i : i + 2] for i in range(len(normalized_claim) - 1)}
        evidence_grams = {
            normalized_evidence[i : i + 2] for i in range(len(normalized_evidence) - 1)
        }
        if not claim_grams or not evidence_grams:
            return 0.0
        return len(claim_grams & evidence_grams) / len(claim_grams)

    @staticmethod
    def _split_by_timing(
        candidates: Sequence[Evidence], as_of_date: date
    ) -> tuple[list[Evidence], list[Evidence]]:
        timely = [e for e in candidates if e.disclosure_date <= as_of_date]
        lookahead = [e for e in candidates if e.disclosure_date > as_of_date]
        return timely, lookahead

    def _check_numbers(
        self, claim_numbers: Sequence[ClaimNumber], candidates: Sequence[Evidence]
    ) -> "_NumberCheckOutcome":
        """逐个数字在**其所属报告期**的证据里核对。

        为什么不是"找一条覆盖全部指标的证据"（旧实现）
            旧规则的初衷是防串期：避免"营收取自 2024 年报、毛利率取自 2023 年报"
            却整体判为通过。但它把两类正常情况一并误杀了（F-018）：
            1. **跨期对比断言**——"毛利率由 19.44% 降至 17.74%" 会产生两个
               metric 都是 gross_margin 的数字，任何单条证据都只能满足其中一个；
            2. **跨表复合断言**——营收在利润表、经营现金流在现金流量表，
               本来就分属两条证据，不存在能同时覆盖的单条证据。
            这两类恰恰是归因分析最常见的表述，导致同一个问题换个说法就大面积拒答。

        新规则：按报告期分组核对，串期防护由"期"承担而不是由"单条证据"承担
            - 数字**显式带 period** → 只在该报告期的证据里找，天然支持跨期对比；
            - 数字**不带 period** → 整组必须落在**同一个报告期**内
              （在该期内允许跨证据组合，解决跨表问题），
              优先选覆盖最完整的期，其次选最新的期。
            这样既放行了上面两类正常断言，又保留了防串期的核心保证。
        """
        if not claim_numbers:
            return _NumberCheckOutcome(True, False, None, None, "无数字需要核对", [])

        # 证据按报告期分组。没有 period_end 的（知识库、新闻、行情）归到 None 组，
        # 它们可以给任意报告期的数字提供佐证。
        grouped: dict[int | None, list[Evidence]] = {}
        for evidence in candidates:
            key = evidence.period_end.year if evidence.period_end else None
            grouped.setdefault(key, []).append(evidence)

        explicit = [n for n in claim_numbers if n.period_year() is not None]
        floating = [n for n in claim_numbers if n.period_year() is None]

        diffs: list[dict] = []
        uncovered: list[str] = []
        used: list[Evidence] = []

        # ---- 显式指定报告期的数字：各自回到自己的期核对 ----
        for number in explicit:
            year = number.period_year()
            pool = grouped.get(year, []) + grouped.get(None, [])
            matched = self._match_one(number, pool)
            if matched is None:
                uncovered.append(f"{number.metric}（{year}年）")
            elif matched[1] is None:
                used.append(matched[0])
            else:
                diffs.append(matched[1])
                used.append(matched[0])

        # ---- 未指定报告期的数字：整组落在同一期 ----
        if floating:
            best = self._match_floating_group(floating, grouped)
            if best is None:
                uncovered.extend(n.metric for n in floating)
            else:
                group_used, group_diffs, group_uncovered = best
                used.extend(group_used)
                diffs.extend(group_diffs)
                uncovered.extend(group_uncovered)

        primary = max(used, key=lambda e: e.priority) if used else (
            candidates[0] if candidates else None
        )

        if diffs:
            return _NumberCheckOutcome(
                passed=False,
                has_conflict=True,
                evidence_id=primary.evidence_id if primary else None,
                source_name=primary.source_name if primary else None,
                reason=(
                    f"数字与来源不一致（{len(diffs)} 处）: "
                    + "; ".join(
                        f"{d['metric']} 声称 {d['claimed']} vs 原始 {d['expected']}"
                        for d in diffs[:3]
                    )
                ),
                diffs=diffs,
            )

        if uncovered:
            # 明确指出**哪个指标**没有证据覆盖。旧实现会退而求其次去匹配
            # 另一个报告期的证据，然后报一个与真实问题无关的差异，
            # 把排查引向错误方向。
            return _NumberCheckOutcome(
                passed=False,
                has_conflict=False,
                evidence_id=primary.evidence_id if primary else None,
                source_name=primary.source_name if primary else None,
                reason=f"以下指标在证据池中找不到对应数据: {', '.join(uncovered[:5])}",
                diffs=[],
            )

        source_names = sorted({e.source_name for e in used})
        return _NumberCheckOutcome(
            passed=True,
            has_conflict=False,
            evidence_id=primary.evidence_id if primary else None,
            source_name=primary.source_name if primary else None,
            reason=f"{len(claim_numbers)} 个数字与 {'、'.join(source_names[:3])} 一致",
            diffs=[],
        )

    def _match_one(
        self, number: ClaimNumber, pool: Sequence[Evidence]
    ) -> tuple[Evidence, dict | None] | None:
        """在给定证据池里核对一个数字。

        返回 (命中的证据, 差异明细或 None)；找不到任何含该指标的证据时返回 None。
        同一指标被多条证据覆盖时，**优先取能对上的那条**——不同来源对同一指标
        可能有口径差异（例如 ROE 的摊薄 vs 平均），只要有一条口径能对上，
        就说明这个数字是有依据的，不该因为另一条口径不同而判为造假。
        """
        holders = [e for e in pool if number.metric in e.numbers]
        if not holders:
            return None

        kind = (
            number.kind if number.kind != "absolute" else infer_number_kind(number.metric)
        )
        fallback: tuple[Evidence, dict] | None = None
        for evidence in sorted(holders, key=lambda e: e.priority, reverse=True):
            expected = evidence.numbers[number.metric]
            ok, detail = self._compare(number.value, expected, kind)
            if ok:
                return evidence, None
            if fallback is None:
                fallback = (
                    evidence,
                    {
                        "metric": number.metric,
                        "claimed": number.value,
                        "expected": expected,
                        "kind": kind,
                        "detail": detail,
                        "source": evidence.source_name,
                        "period": number.period,
                    },
                )
        return fallback

    def _match_floating_group(
        self,
        floating: Sequence[ClaimNumber],
        grouped: dict[int | None, list[Evidence]],
        ) -> tuple[list[Evidence], list[dict], list[str]] | None:
        """未指定报告期的数字整组核对，要求全部落在同一个报告期。

        候选期按 (覆盖的指标数, 年份) 降序尝试：先看哪个期能覆盖最多指标，
        同等覆盖度下取最新期。这个顺序保证"2025年营收…"这种省略了年份、
        实际指最新期的常见写法能匹配到正确的期。
        """
        undated = grouped.get(None, [])
        years = sorted((y for y in grouped if y is not None), reverse=True)
        if not years and not undated:
            return None

        scored: list[tuple[int, int, int | None]] = []
        for year in years:
            pool = grouped[year] + undated
            covered = sum(
                1 for n in floating if any(n.metric in e.numbers for e in pool)
            )
            scored.append((covered, year, year))
        if undated:
            covered = sum(1 for n in floating if any(n.metric in e.numbers for e in undated))
            scored.append((covered, -1, None))

        scored.sort(reverse=True)
        if not scored or scored[0][0] == 0:
            return None

        best_year = scored[0][2]
        pool = grouped.get(best_year, []) + (undated if best_year is not None else [])

        used: list[Evidence] = []
        diffs: list[dict] = []
        uncovered: list[str] = []
        for number in floating:
            matched = self._match_one(number, pool)
            if matched is None:
                uncovered.append(number.metric)
                continue
            used.append(matched[0])
            if matched[1] is not None:
                diffs.append(matched[1])
        return used, diffs, uncovered

    def _compare(self, claimed: float, expected: float, kind: str) -> tuple[bool, str]:
        """按语义类型比对两个数字。"""
        if kind in ("ratio", "growth"):
            tolerance_pp = (
                self.ratio_tolerance_pp if kind == "ratio" else self.growth_tolerance_pp
            )
            # 统一到"百分点"维度：内部约定比率用小数存储（0.218 = 21.8%），
            # 但 LLM 有时会给 21.8。两种写法都接受，先归一。
            claimed_pp = self._to_percentage_points(claimed)
            expected_pp = self._to_percentage_points(expected)
            diff_pp = abs(claimed_pp - expected_pp)
            ok = diff_pp <= tolerance_pp
            return ok, f"差异 {diff_pp:.2f}pp，容差 {tolerance_pp}pp"

        if expected == 0:
            ok = abs(claimed) < 1e-9
            return ok, "基准值为 0，要求完全相等"
        relative_error = abs(claimed - expected) / abs(expected)
        ok = relative_error <= self.number_tolerance
        return ok, f"相对误差 {relative_error:.2%}，容差 {self.number_tolerance:.0%}"

    @staticmethod
    def _to_percentage_points(value: float) -> float:
        """把比率归一到百分点。|value| <= 1.5 视为小数形式，否则视为已是百分数。

        1.5 这个分界点的取舍：ROE 超过 150% 的 A 股公司极罕见，
        而写成小数的比率几乎不会超过 1.5，误判概率远低于其他分界点。
        """
        return value * 100 if abs(value) <= 1.5 else value

    # ================= trace =================

    def _emit(self, result: GateResult, claim: Claim) -> GateResult:
        self.tracer.log(
            "gate_check",
            agent_name="evidence_gate",
            input_summary=claim.text,
            output_summary=f"{result.status}: {result.reason}",
            success=result.status == GATE_VERIFIED,
            claim_id=claim.claim_id,
            gate_status=result.status,
            matched_source=result.matched_source,
        )
        return result


class _NumberCheckOutcome:
    """数字级检查的内部返回结构（不对外暴露，故不做成 Pydantic 模型）。"""

    __slots__ = ("passed", "has_conflict", "evidence_id", "source_name", "reason", "diffs")

    def __init__(
        self,
        passed: bool,
        has_conflict: bool,
        evidence_id: str | None,
        source_name: str | None,
        reason: str,
        diffs: list[dict],
    ):
        self.passed = passed
        # has_conflict 区分"数字对不上"(number_mismatch) 与
        # "找不到能对的数字"(unverified)——两者在研报里的处置方式不同
        self.has_conflict = has_conflict
        self.evidence_id = evidence_id
        self.source_name = source_name
        self.reason = reason
        self.diffs = diffs
