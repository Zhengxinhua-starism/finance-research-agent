"""自动化评测 pipeline（LLM-as-Judge + 确定性指标）。

解决什么问题
    Agent 项目最常见的问题是"只有 demo，没有评测"：改了个 prompt，
    不知道是变好了还是变差了；换了个模型，不知道哪类问题受影响。
    本模块提供一条可重复执行的评测流水线，输出四个维度的量化分数。

核心设计决策
    1. **四个维度里只有一个用 LLM 打分，另外三个是确定性计算。**
       - factual_accuracy：LLM-as-Judge，因为"输出是否覆盖了预期事实"
         需要语义理解，正则匹配做不到；
       - refusal_calibration：确定性——该拒答时有没有输出具体数字，
         用正则扫数字 + 检查 refused 标记即可判定，不需要也不应该用 LLM；
       - retrieval_efficiency：确定性——从 trace 里读实际检索轮次和工具调用数，
         与预期对比；
       - evidence_coverage：确定性——已验证结论 / 总结论，直接从
         Verifier 的输出算。
       把能算的都算出来，只在真正需要语义判断的地方用 LLM，
       这样评测本身的方差才可控。
    2. Judge 用 deep 层且 temperature=0。评测的可重复性比生成质量更重要——
       同一份输出两次评测给出不同分数，这个评测就没有意义。
    3. Judge prompt 要求**逐条判定 expected_facts 是否被覆盖**，
       而不是笼统地"给这份研报打个分"。逐条判定的结果可以列出
       matched_facts / missing_facts，失败时能直接定位问题；
       整体打分只能得到一个不知道从何改起的数字。
    4. 单个用例失败不中断整批。评测跑 9 道题要 10~20 分钟，
       第 3 题因为网络抖动挂掉就全部重来是不可接受的。
       失败的用例记 0 分并保留错误信息。
    5. 结果落盘为 JSON + Markdown 两份。JSON 给程序做回归对比，
       Markdown 给人看趋势。

为什么不用其他方案
    - 不用 ragas / deepeval：它们的指标（faithfulness、answer_relevancy）
      是为通用 RAG 设计的，不覆盖本项目最关心的"拒答校准"和"证据覆盖率"。
      而且它们内部也是 LLM-as-Judge，等于多引入一层不可控的 prompt。
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import PROJECT_ROOT, get_config

logger = logging.getLogger(__name__)

# 五个维度的权重。
#
# attribution_quality（归因质量）是 v2 新增的，权重仅次于事实准确性。
# 加它的原因：v1 只检查"数字有没有报对"，于是一份把所有数字都报对、
# 但通篇"净利润下降因为毛利率下降"（同义反复）的研报能拿高分。
# 而这恰恰是实际输出最大的质量问题——用数字解释数字，没有业务信息量。
#
# 权重不是所有用例都全用：单点事实类问题没有归因要求，
# 拒答类问题不评事实准确性。计算总分时只对**适用的维度**做归一化加权
# （见 _weighted_overall），避免用 N/A 维度稀释分数。
DIMENSION_WEIGHTS: dict[str, float] = {
    "factual_accuracy": 0.30,
    "attribution_quality": 0.25,
    "refusal_calibration": 0.25,
    "evidence_coverage": 0.12,
    "retrieval_efficiency": 0.08,
}

PASS_THRESHOLD = 0.60

# Judge 调用的输出上限。必须显式设置：deepseek-reasoner 的思维链也计入
# completion_tokens，实测一次归因评审花了 1205 个 completion token，
# 其中 1064 个是思维链，留给正文的只剩 141 个——JSON 在字符串中间被截断，
# 解析失败，整个维度记 0 分。而 0 分和"归因确实很差"在报表上长得一样，
# 会把人引向错误的改进方向。
JUDGE_MAX_TOKENS = 4000

JUDGE_SYSTEM_PROMPT = """你是一名严格的金融研报评审员，负责判断 AI 生成的研报是否覆盖了预期事实。

## 评判方法
对给定的每一条「预期事实」，独立判断研报中是否包含了对应的内容：
- covered（已覆盖）：研报明确给出了该事实，数值在允许误差内
- partial（部分覆盖）：提到了相关内容但不完整（如给了数值但没给变动方向）
- missing（未覆盖）：研报中完全没有相关内容，或数值明显错误

## 判断原则
1. 数值类事实：预期给出的是区间或方向，只要研报的数值落在区间内即算覆盖。
   金额允许 5% 误差，比率允许 3 个百分点误差（考虑口径差异）。
2. 分析类事实（如"用杜邦框架拆解"）：研报需要实际做了该分析，
   只提到框架名称而没有实际拆解，算 partial。
3. 标注类事实（如"结论标注为已验证"）：检查研报中是否有对应的标注符号。
4. **不要因为研报写得漂亮就放宽标准**，也不要因为格式不完美就压低分数，
   只看预期事实是否被覆盖。
5. 如果研报明确说明"数据缺失，无法判断"，而预期事实要求给出该数据，
   算 missing，但要在 comment 里注明这是"诚实的数据缺失"而非"编造"。

## 输出
score 为 covered 数量 + partial 数量 × 0.5，再除以预期事实总数，范围 0~1。
comment 用一两句话说明主要的缺失点。
"""

JUDGE_TASK_TEMPLATE = """## 用户问题
{question}

## 预期事实清单
{expected_facts}

## 评分补充说明
{scoring_notes}

## AI 生成的研报
{report}

请逐条判定并给出评分。"""


def _coerce_judgements(value: Any, key_field: str) -> Any:
    """把 LLM 常见的几种 judgements 变体归一成对象列表。

    实测 deepseek-reasoner 会返回字典形态
    `{"结构分解": "missing", "主次区分": "met"}`，而不是约定的对象列表。
    这不算模型出错——提示词里描述的是语义，字典表达同样合理。
    与其反复调 prompt 去纠正形态，不如在解析层接受两种写法：
    解析失败的代价是整个维度记 0 分，而 0 分和"归因确实很差"在报表上
    长得一模一样，会让人误判改进方向。
    """
    if isinstance(value, dict):
        coerced: list[dict[str, Any]] = []
        for name, item in value.items():
            if isinstance(item, str):
                coerced.append({key_field: name, "verdict": item})
            elif isinstance(item, dict):
                coerced.append({key_field: name, **item})
        return coerced
    if isinstance(value, list):
        # 也可能是纯字符串列表 ["missing", "met"]，此时无法回填要求名
        return [
            {key_field: "", "verdict": item} if isinstance(item, str) else item
            for item in value
        ]
    return value


class FactJudgement(BaseModel):
    """单条预期事实的判定。"""

    model_config = ConfigDict(extra="ignore")

    fact: str = ""
    verdict: str = "missing"  # covered / partial / missing
    evidence_in_report: str = ""


class JudgeOutput(BaseModel):
    """Judge 的结构化输出。"""

    model_config = ConfigDict(extra="ignore")

    judgements: list[FactJudgement] = Field(default_factory=list)
    score: float = 0.0
    comment: str = ""

    @field_validator("judgements", mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        return _coerce_judgements(value, "fact")


class AttributionJudgement(BaseModel):
    """归因质量的单项判定。"""

    model_config = ConfigDict(extra="ignore")

    requirement: str = ""
    verdict: str = "missing"  # met / partial / missing
    evidence_in_report: str = ""


class AttributionOutput(BaseModel):
    """归因质量评审的结构化输出。"""

    model_config = ConfigDict(extra="ignore")

    judgements: list[AttributionJudgement] = Field(default_factory=list)
    violated_patterns: list[str] = Field(default_factory=list)
    closer_to: str = "neither"  # good / bad / neither
    score: float = 0.0
    comment: str = ""

    @field_validator("judgements", mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        return _coerce_judgements(value, "requirement")

    @field_validator("closer_to", mode="before")
    @classmethod
    def _normalize_closer(cls, value: Any) -> str:
        """模型会写 "bad_answer_example" 而不是约定的 "bad"，归一化。"""
        text = str(value or "").lower()
        if "good" in text:
            return "good"
        if "bad" in text:
            return "bad"
        return "neither"


ATTRIBUTION_SYSTEM_PROMPT = """你是一名资深金融研究总监，负责评审下属写的研报**归因质量**。

你不评判数字对不对（那由另一个环节负责），只评判**分析有没有信息量**。

## 核心判据：有没有「用数字解释数字」
这是最常见也最致命的问题——把同一件事换个说法讲两遍，看起来在分析，实际零信息量。

  ❌ "净利润下降，因为净利率下降了"（净利率的分子就是净利润，同义反复）
  ❌ "盈利能力下滑，主要由于毛利率收窄"（毛利率就是盈利能力的度量）
  ❌ "营收增速放缓导致利润承压"（没说清为什么放缓、为什么承压）
  ✅ "毛利率收窄 1.70pp 的结构分解显示自身效应 -1.87pp、结构效应 +0.17pp，
     说明是各业务自身盈利能力下降而非业务结构调整；其中汽车业务
     毛利率 -1.82pp 贡献了 -1.19pp，是最大拖累"

判断方法：把研报里的因果句抽出来，问「这句话除了换个指标名，
还告诉了我什么业务上的事实吗？」答不上来就是同义反复。

## 评分维度
1. **业务归因**：是否解释了业务层面发生了什么（产品结构、区域差异、
   成本项、费用投向、一次性事件），而不只是指标之间的换算关系。
2. **主次区分**：是否按贡献大小排序并明确指出主因、次因。
   平铺罗列所有因素不算。
3. **确定性标注**：无法用数据证实的归因（如价格战、行业竞争）
   是否被标注为推测或注明来源，而不是与已验证事实混同陈述。
4. **禁止模式**：是否命中 forbidden_patterns 里列出的坏习惯。

## 输出
- judgements：对每条 attribution_requirement 判定 met / partial / missing
- violated_patterns：命中了哪些 forbidden_patterns（原样抄写命中的那一条）
- closer_to：整体上更接近 good_answer_example 还是 bad_answer_example
- score：0~1。met 计 1、partial 计 0.5，除以要求总数；
  每命中一条 forbidden_pattern 额外扣 0.15，最低扣到 0。
- comment：一两句话指出最主要的问题

**不要因为研报篇幅长、格式规整就给高分**，只看有没有真正的业务归因。
"""

ATTRIBUTION_TASK_TEMPLATE = """## 用户问题
{question}

## 归因要求（逐条判定）
{requirements}

## 禁止出现的坏习惯
{forbidden}

## 好答案示例（参考标尺，不要求逐字一致）
{good_example}

## 坏答案示例（这种写法应当低分）
{bad_example}
{bad_reason}

## 待评审的研报
{report}

请评审归因质量。"""


def load_test_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    """加载测试集。"""
    config = get_config()
    file_path = Path(path) if path else config.resolve_path(config.eval_test_cases_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"测试集文件不存在: {file_path}。请确认 eval/test_cases.json 已创建"
        )
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", payload if isinstance(payload, list) else [])
    if not cases:
        raise ValueError(f"测试集为空: {file_path}")
    return cases


class Evaluator:
    """评测执行器。"""

    def __init__(
        self,
        pipeline: Any = None,
        judge_client: Any = None,
        test_cases_path: str | Path | None = None,
    ):
        self.config = get_config()
        self._pipeline = pipeline
        self._judge = judge_client
        self.test_cases_path = test_cases_path
        self.results_dir = PROJECT_ROOT / "eval" / "results"

    # ---------------- 惰性依赖 ----------------

    @property
    def pipeline(self) -> Any:
        """研究流程。惰性创建，避免只想看测试集时也加载模型。"""
        if self._pipeline is None:
            from graph.graph import ResearchPipeline

            self._pipeline = ResearchPipeline()
        return self._pipeline

    @property
    def judge(self) -> Any:
        if self._judge is None:
            from llm.router import LLMRouter

            self._judge = LLMRouter(self.config).get(self.config.eval_judge_tier)
        return self._judge

    # ---------------- 主流程 ----------------

    def run(
        self, case_ids: Sequence[int] | None = None, save_report: bool = True
    ) -> dict[str, Any]:
        cases = load_test_cases(self.test_cases_path)
        if case_ids:
            wanted = set(case_ids)
            cases = [case for case in cases if case["id"] in wanted]
            missing = wanted - {case["id"] for case in cases}
            if missing:
                logger.warning("以下用例 ID 不存在，已跳过: %s", sorted(missing))
        if not cases:
            raise ValueError("没有匹配的测试用例")

        logger.info("开始评测，共 %d 个用例", len(cases))
        started = time.perf_counter()
        results: list[dict[str, Any]] = []

        for index, case in enumerate(cases, start=1):
            logger.info("[%d/%d] 用例 %s: %s", index, len(cases), case["id"], case["question"])
            try:
                results.append(self.evaluate_case(case))
            except Exception as exc:  # noqa: BLE001 — 单个用例失败不中断整批
                logger.exception("用例 %s 评测失败", case["id"])
                results.append(self._failed_result(case, f"{type(exc).__name__}: {exc}"))

        summary = self._aggregate(results, duration_s=time.perf_counter() - started)
        outcome: dict[str, Any] = {"results": results, "summary": summary}

        if save_report:
            outcome["report_path"] = self._save(results, summary)

        logger.info("评测完成: %s", summary)
        return outcome

    def evaluate_case(self, case: dict[str, Any]) -> dict[str, Any]:
        """跑单个用例并打四个维度的分。"""
        started = time.perf_counter()

        outcome = self.pipeline.run(
            question=case["question"],
            ticker=str(case.get("ticker", "")),
            company=str(case.get("company", "")),
            as_of_date=date.today(),
        )
        state = outcome["state"]
        trace_summary = outcome["trace_summary"]
        report_markdown = state.get("report_markdown", "")
        report_dict = state.get("report_dict") or {}
        verify_result = state.get("verify_result")
        duration_ms = int((time.perf_counter() - started) * 1000)

        should_refuse = bool(case.get("should_refuse", False))

        # --- 维度 1：事实准确性（LLM-as-Judge）---
        if should_refuse:
            # 拒答题不评事实准确性：预期就是"什么都别说"，
            # 用事实覆盖率去评会得出"覆盖率越低越好"的荒谬结论。
            # 直接沿用拒答校准分，保证加权总分的语义正确。
            judge_output = None
            factual_accuracy = 0.0
        else:
            judge_output = self._judge_facts(case, report_markdown)
            factual_accuracy = judge_output.score

        # --- 维度 2：拒答校准（确定性）---
        refusal_calibration, refusal_note = self._score_refusal(
            should_refuse, report_markdown, report_dict
        )
        if should_refuse:
            factual_accuracy = refusal_calibration

        # --- 维度 3：检索效率（确定性，从 trace 提取）---
        retrieval_efficiency, efficiency_note = self._score_retrieval_efficiency(
            case, trace_summary, state
        )

        # --- 维度 4：证据覆盖率（确定性，从 Verifier 输出提取）---
        evidence_coverage = self._score_evidence_coverage(verify_result, should_refuse)

        # --- 维度 5：归因质量（LLM-as-Judge，仅对有归因要求的用例）---
        attribution_output = None
        if not should_refuse and case.get("attribution_requirements"):
            attribution_output = self._judge_attribution(case, report_markdown)
            attribution_quality = attribution_output.score
        else:
            attribution_quality = 0.0

        scores = {
            "factual_accuracy": factual_accuracy,
            "attribution_quality": attribution_quality,
            "refusal_calibration": refusal_calibration,
            "retrieval_efficiency": retrieval_efficiency,
            "evidence_coverage": evidence_coverage,
        }
        # 只对适用的维度归一化加权：拒答题不评事实准确性，
        # 单点事实题没有归因要求。用 N/A 维度参与加权会无谓地稀释分数。
        applicable = set(DIMENSION_WEIGHTS)
        if should_refuse:
            applicable.discard("factual_accuracy")
        if attribution_output is None:
            applicable.discard("attribution_quality")
        overall = self._weighted_overall(scores, applicable)

        judgements = judge_output.judgements if judge_output else []
        return {
            "case_id": case["id"],
            "question": case["question"],
            "company": case.get("company", ""),
            "ticker": case.get("ticker", ""),
            "difficulty": case.get("difficulty", ""),
            "question_type": case.get("question_type", ""),
            "should_refuse": should_refuse,
            "passed": overall >= PASS_THRESHOLD,
            **{name: round(value, 4) for name, value in scores.items()},
            "overall_score": round(overall, 4),
            "judge_comment": (judge_output.comment if judge_output else refusal_note),
            "matched_facts": [j.fact for j in judgements if j.verdict == "covered"],
            "partial_facts": [j.fact for j in judgements if j.verdict == "partial"],
            "missing_facts": [j.fact for j in judgements if j.verdict == "missing"],
            "attribution_comment": attribution_output.comment if attribution_output else "",
            "attribution_met": [
                j.requirement
                for j in (attribution_output.judgements if attribution_output else [])
                if j.verdict == "met"
            ],
            "attribution_missing": [
                j.requirement
                for j in (attribution_output.judgements if attribution_output else [])
                if j.verdict == "missing"
            ],
            "violated_patterns": attribution_output.violated_patterns
            if attribution_output
            else [],
            "closer_to": attribution_output.closer_to if attribution_output else "n/a",
            "tools_used": list((trace_summary.get("tool_usage") or {}).keys()),
            "expected_tools": case.get("expected_tools", []),
            "retrieval_count": state.get("retrieval_count", 0),
            "efficiency_note": efficiency_note,
            "duration_ms": duration_ms,
            "total_tokens": trace_summary.get("total_tokens", 0),
            "run_id": outcome["run_id"],
            "agent_errors": state.get("errors") or [],
            "error": None,
        }

    # ---------------- 各维度打分 ----------------

    def _judge_facts(self, case: dict[str, Any], report: str) -> JudgeOutput:
        expected_facts: list[str] = case.get("expected_facts", [])
        if not expected_facts:
            return JudgeOutput(score=0.0, comment="用例未定义 expected_facts")
        if not report.strip():
            return JudgeOutput(
                judgements=[FactJudgement(fact=fact, verdict="missing") for fact in expected_facts],
                score=0.0,
                comment="研报为空",
            )

        task = JUDGE_TASK_TEMPLATE.format(
            question=case["question"],
            expected_facts="\n".join(
                f"{i}. {fact}" for i, fact in enumerate(expected_facts, start=1)
            ),
            scoring_notes=case.get("scoring_notes", "（无补充说明）"),
            # 截断超长研报：Judge 只需要判断事实覆盖，
            # 超过 8000 字符的部分通常是数据来源表，对判定没有帮助
            report=report[:8000],
        )

        response = self._structured_judge(task, JUDGE_SYSTEM_PROMPT, JudgeOutput)

        if not isinstance(response, JudgeOutput):
            logger.error("Judge 输出解析失败: %s", str(response)[:200])
            return JudgeOutput(score=0.0, comment=f"Judge 输出无法解析: {str(response)[:150]}")

        # 用 judgements 重算分数，不信任 LLM 自己算的 score。
        # 模型做除法出错的概率不低，而这个分数直接决定评测结论。
        if response.judgements:
            covered = sum(1 for j in response.judgements if j.verdict == "covered")
            partial = sum(1 for j in response.judgements if j.verdict == "partial")
            response.score = (covered + partial * 0.5) / len(response.judgements)
        response.score = max(0.0, min(1.0, response.score))
        return response

    def _judge_attribution(self, case: dict[str, Any], report: str) -> AttributionOutput:
        """归因质量评审。与事实准确性分开调用，因为两者的判据完全不同：

        前者看"说得对不对"，后者看"说得有没有信息量"。合在一个 prompt 里，
        模型会被数字的正确性带偏，给一份数字全对但通篇同义反复的研报打高分。
        """
        requirements: list[str] = case.get("attribution_requirements", [])
        if not requirements or not report.strip():
            return AttributionOutput(score=0.0, comment="无归因要求或研报为空")

        forbidden: list[str] = case.get("forbidden_patterns", [])
        task = ATTRIBUTION_TASK_TEMPLATE.format(
            question=case["question"],
            requirements="\n".join(f"{i}. {r}" for i, r in enumerate(requirements, 1)),
            forbidden="\n".join(f"- {p}" for p in forbidden) or "（无）",
            good_example=case.get("good_answer_example", "（本用例未提供）"),
            bad_example=case.get("bad_answer_example", "（本用例未提供）"),
            bad_reason=(
                f"\n坏在哪里：{case['bad_answer_reason']}"
                if case.get("bad_answer_reason")
                else ""
            ),
            report=report[:8000],
        )

        response = self._structured_judge(
            task, ATTRIBUTION_SYSTEM_PROMPT, AttributionOutput
        )
        if not isinstance(response, AttributionOutput):
            logger.error("归因评审输出解析失败: %s", str(response)[:200])
            return AttributionOutput(score=0.0, comment=f"评审输出无法解析: {str(response)[:150]}")

        # 与事实评审同理：不信任 LLM 自己算的分数，用判定列表重算。
        if response.judgements:
            met = sum(1 for j in response.judgements if j.verdict == "met")
            partial = sum(1 for j in response.judgements if j.verdict == "partial")
            base = (met + partial * 0.5) / len(response.judgements)
        else:
            base = response.score
        penalty = 0.15 * len(response.violated_patterns)
        response.score = max(0.0, min(1.0, base - penalty))
        return response

    def _structured_judge(self, task: str, system_prompt: str, schema: Any) -> Any:
        """调 Judge 并解析结构化输出，失败时降级到 quick 层重试一次。

        为什么需要降级：deep 层（deepseek-reasoner）不支持 response_format，
        只能靠提示词约束 JSON 格式，加上思维链会挤占输出预算，
        结构化输出的成功率明显低于 quick 层。而 quick 层（deepseek-chat）
        支持原生 JSON mode，格式合法性由服务端保证。

        为什么不干脆全用 quick 层：评审要判断"这句话是不是同义反复"，
        需要一定的推理深度，deep 层的判定质量更好。所以是
        "优先 deep，格式失败才退 quick"，而不是一开始就妥协。
        """
        response = self.judge.chat_structured(
            messages=[{"role": "user", "content": task}],
            schema=schema,
            system_prompt=system_prompt,
            max_tokens=JUDGE_MAX_TOKENS,
        )
        if isinstance(response, schema):
            return response

        logger.warning(
            "%s 层 Judge 结构化输出失败，降级到 quick 层重试（JSON mode）",
            self.config.eval_judge_tier,
        )
        try:
            from llm.router import LLMRouter

            fallback = LLMRouter(self.config).get("quick")
            retry = fallback.chat_structured(
                messages=[{"role": "user", "content": task}],
                schema=schema,
                system_prompt=system_prompt,
                max_tokens=JUDGE_MAX_TOKENS,
            )
            if isinstance(retry, schema):
                return retry
            logger.error("quick 层重试仍无法解析: %s", str(retry)[:200])
        except Exception as exc:  # noqa: BLE001 — 评审失败不能中断整批评测
            logger.error("quick 层降级重试异常: %s", exc)
        return response

    @staticmethod
    def _weighted_overall(scores: dict[str, float], applicable: set[str]) -> float:
        """只对适用维度做归一化加权。"""
        weights = {
            name: weight
            for name, weight in DIMENSION_WEIGHTS.items()
            if name in applicable
        }
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return 0.0
        return sum(scores[name] * weight for name, weight in weights.items()) / total_weight

    @staticmethod
    def _score_refusal(
        should_refuse: bool, report: str, report_dict: dict[str, Any]
    ) -> tuple[float, str]:
        """拒答校准。

        判定逻辑刻意做成确定性的：
        - 该拒答时：研报必须带 ❌拒答 标注 **且** 不含具体财务数字；
        - 不该拒答时：研报不能整体拒答，且要有 ✅ 或 ⚠️ 标注的实质内容。
        用"有没有输出数字"而不是"语气像不像拒答"来判定，
        因为幻觉的危害在于给出了错误的数字，不在于措辞。
        """
        refused_flag = bool(report_dict.get("refused"))
        has_refuse_label = "❌拒答" in report or "无法判断" in report or "无法回答" in report
        has_verified_label = "✅已验证" in report

        if should_refuse:
            # 扫描"疑似财务数字"：带亿/万/%的数值。日期和股票代码要排除，
            # 否则"2024 年"会被误判成编造的数字。
            body = re.sub(r"20\d{2}\s*年?", "", report)
            body = re.sub(r"\b\d{6}\b", "", body)
            fabricated = re.findall(r"\d[\d,]*\.?\d*\s*(?:亿元|亿|万元|%|个百分点|pp)", body)
            if fabricated:
                return 0.0, (
                    f"应当拒答但输出了 {len(fabricated)} 处具体数值："
                    f"{fabricated[:5]}，属于幻觉未被拦截"
                )
            if refused_flag or has_refuse_label:
                return 1.0, "正确拒答，未输出任何具体数值"
            return 0.3, "未输出编造数值，但也没有明确标注拒答"

        if refused_flag and not has_verified_label:
            return 0.0, "本应给出答案，但系统整体拒答（可能是取数失败或过度保守）"
        if has_verified_label:
            return 1.0, "正常作答且包含已验证结论"
        if "⚠️未验证" in report:
            return 0.6, "有作答但全部结论未通过验证"
        return 0.4, "有作答但缺少三级标注"

    @staticmethod
    def _score_retrieval_efficiency(
        case: dict[str, Any], trace_summary: dict[str, Any], state: dict[str, Any]
    ) -> tuple[float, str]:
        """检索效率。

        评分逻辑：轮次符合预期得满分，多一轮扣 0.3；
        另外考察工具选择的准确性（是否调用了 expected_tools）。
        效率不是越少越好——一轮都没检索到数据同样是失败。
        """
        expected_rounds = int(case.get("expected_retrieval_count", 1))
        actual_rounds = int(state.get("retrieval_count", 0))
        expected_tools = set(case.get("expected_tools", []))
        used_tools = set((trace_summary.get("tool_usage") or {}).keys())

        # 预期就是不检索（超范围拒答题在 Planner 阶段即被短路），
        # 此时 0 轮是**满分行为**。旧实现无条件把 0 轮判 0 分，
        # 等于惩罚了正确的短路优化——用例 8、9 因此各丢了 0.08 分。
        if expected_rounds == 0:
            if actual_rounds == 0:
                return 1.0, "按预期未执行检索（问题在 Planner 阶段即被拦截，未消耗工具与 LLM）"
            return 0.4, f"预期不应检索，实际检索了 {actual_rounds} 轮（短路未生效）"

        if actual_rounds == 0:
            return 0.0, "未执行任何检索"

        round_score = max(0.0, 1.0 - 0.3 * max(0, actual_rounds - expected_rounds))

        if not expected_tools:
            # 拒答题不期望调用任何工具：调了工具发现查不到也是合理路径，
            # 因此只要没有失控地反复调用就给满分
            tool_score = 1.0 if len(used_tools) <= 3 else 0.6
            note = f"检索 {actual_rounds} 轮（预期 {expected_rounds}），调用工具 {sorted(used_tools)}"
        else:
            hit = expected_tools & used_tools
            tool_score = len(hit) / len(expected_tools)
            note = (
                f"检索 {actual_rounds} 轮（预期 {expected_rounds}）；"
                f"命中预期工具 {len(hit)}/{len(expected_tools)}: {sorted(hit)}"
            )
            missing = expected_tools - used_tools
            if missing:
                note += f"；未调用 {sorted(missing)}"

        # 轮次和工具选择各占一半：调对了工具但跑了三轮，和只跑一轮但工具选错，
        # 都是需要改进的，不该只惩罚其中一种
        return round(round_score * 0.5 + tool_score * 0.5, 4), note

    @staticmethod
    def _score_evidence_coverage(verify_result: Any, should_refuse: bool) -> float:
        """证据覆盖率 = 已验证结论 / 总结论。"""
        if verify_result is None:
            # 拒答题没有结论是正确行为，给满分；非拒答题没有结论是失败
            return 1.0 if should_refuse else 0.0
        claims = getattr(verify_result, "claims", []) or []
        if not claims:
            return 1.0 if should_refuse else 0.0
        verified = sum(1 for claim in claims if getattr(claim, "verdict", "") == "verified")
        return round(verified / len(claims), 4)

    # ---------------- 汇总与落盘 ----------------

    @staticmethod
    def _failed_result(case: dict[str, Any], error: str) -> dict[str, Any]:
        return {
            "case_id": case["id"],
            "question": case.get("question", ""),
            "company": case.get("company", ""),
            "ticker": case.get("ticker", ""),
            "difficulty": case.get("difficulty", ""),
            "question_type": case.get("question_type", ""),
            "should_refuse": case.get("should_refuse", False),
            "passed": False,
            "factual_accuracy": 0.0,
            "attribution_quality": 0.0,
            "refusal_calibration": 0.0,
            "retrieval_efficiency": 0.0,
            "evidence_coverage": 0.0,
            "overall_score": 0.0,
            "judge_comment": f"用例执行失败: {error}",
            "matched_facts": [],
            "partial_facts": [],
            "missing_facts": case.get("expected_facts", []),
            "attribution_comment": "",
            "attribution_met": [],
            "attribution_missing": case.get("attribution_requirements", []),
            "violated_patterns": [],
            "closer_to": "n/a",
            "tools_used": [],
            "expected_tools": case.get("expected_tools", []),
            "retrieval_count": 0,
            "efficiency_note": "",
            "duration_ms": 0,
            "total_tokens": 0,
            "run_id": "",
            "agent_errors": [],
            "error": error,
        }

    @staticmethod
    def _aggregate(results: Sequence[dict[str, Any]], duration_s: float) -> dict[str, Any]:
        count = len(results) or 1

        def mean(field: str) -> float:
            return round(sum(float(item.get(field, 0.0)) for item in results) / count, 4)

        by_difficulty: dict[str, list[float]] = {}
        by_type: dict[str, list[float]] = {}
        for item in results:
            by_difficulty.setdefault(item.get("difficulty", "unknown"), []).append(
                item["overall_score"]
            )
            by_type.setdefault(item.get("question_type", "unknown"), []).append(
                item["overall_score"]
            )

        return {
            "total_cases": len(results),
            "passed": sum(1 for item in results if item.get("passed")),
            "pass_rate": round(
                sum(1 for item in results if item.get("passed")) / count, 4
            ),
            "factual_accuracy": mean("factual_accuracy"),
            # 归因质量只在适用的用例上取均值，否则会被不适用的 0 拉低
            "attribution_quality": (
                round(
                    sum(
                        float(item.get("attribution_quality", 0.0))
                        for item in results
                        if item.get("closer_to") not in (None, "n/a")
                    )
                    / max(1, sum(1 for item in results if item.get("closer_to") not in (None, "n/a"))),
                    4,
                )
            ),
            "attribution_applicable_cases": sum(
                1 for item in results if item.get("closer_to") not in (None, "n/a")
            ),
            "refusal_calibration": mean("refusal_calibration"),
            "retrieval_efficiency": mean("retrieval_efficiency"),
            "evidence_coverage": mean("evidence_coverage"),
            "overall_score": mean("overall_score"),
            "avg_duration_ms": int(mean("duration_ms")),
            "total_tokens": sum(int(item.get("total_tokens", 0)) for item in results),
            "by_difficulty": {
                key: round(sum(values) / len(values), 4) for key, values in by_difficulty.items()
            },
            "by_question_type": {
                key: round(sum(values) / len(values), 4) for key, values in by_type.items()
            },
            "failed_cases": [
                item["case_id"] for item in results if not item.get("passed")
            ],
            "eval_duration_s": round(duration_s, 1),
            "evaluated_at": datetime.now().astimezone().isoformat(),
            "pass_threshold": PASS_THRESHOLD,
            "dimension_weights": DIMENSION_WEIGHTS,
        }

    def _save(self, results: Sequence[dict[str, Any]], summary: dict[str, Any]) -> str:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = self.results_dir / f"eval_{stamp}.json"
        json_path.write_text(
            json.dumps(
                {"summary": summary, "results": list(results)},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        markdown_path = self.results_dir / f"eval_{stamp}.md"
        markdown_path.write_text(self._render_markdown(results, summary), encoding="utf-8")

        logger.info("评测结果已保存: %s", json_path)
        return str(json_path)

    @staticmethod
    def _render_markdown(results: Sequence[dict[str, Any]], summary: dict[str, Any]) -> str:
        lines = [
            "# 评测报告",
            "",
            f"> 执行时间：{summary['evaluated_at']} | "
            f"用例数：{summary['total_cases']} | "
            f"耗时：{summary['eval_duration_s']}s | "
            f"总 token：{summary['total_tokens']}",
            "",
            "## 总体得分",
            "",
            "| 维度 | 权重 | 得分 |",
            "|------|------|------|",
        ]
        for name, weight in DIMENSION_WEIGHTS.items():
            lines.append(f"| {name} | {weight:.0%} | {summary[name]:.3f} |")
        lines.extend(
            [
                f"| **加权总分** | 100% | **{summary['overall_score']:.3f}** |",
                "",
                f"通过率：{summary['passed']}/{summary['total_cases']} "
                f"（{summary['pass_rate']:.0%}，阈值 {summary['pass_threshold']}）",
                "",
                "## 分维度统计",
                "",
                "按难度：" + ", ".join(f"{k}={v:.3f}" for k, v in summary["by_difficulty"].items()),
                "",
                "按问题类型：" + ", ".join(f"{k}={v:.3f}" for k, v in summary["by_question_type"].items()),
                "",
                "## 逐题结果",
                "",
                "| ID | 问题 | 难度 | 事实 | 归因 | 拒答 | 检索 | 证据 | 总分 | 通过 |",
                "|----|------|------|------|------|------|------|------|------|------|",
            ]
        )
        for item in results:
            attribution = (
                f"{item.get('attribution_quality', 0.0):.2f}"
                if item.get("closer_to") not in (None, "n/a")
                else "—"
            )
            lines.append(
                f"| {item['case_id']} | {item['question'][:26]} | {item['difficulty']} | "
                f"{item['factual_accuracy']:.2f} | {attribution} | "
                f"{item['refusal_calibration']:.2f} | "
                f"{item['retrieval_efficiency']:.2f} | {item['evidence_coverage']:.2f} | "
                f"{item['overall_score']:.2f} | {'✅' if item['passed'] else '❌'} |"
            )

        failed = [item for item in results if not item.get("passed")]
        if failed:
            lines.extend(["", "## 未通过用例详情", ""])
            for item in failed:
                lines.extend(
                    [
                        f"### 用例 {item['case_id']}：{item['question']}",
                        "",
                        f"- 总分：{item['overall_score']:.3f}",
                        f"- 评审意见：{item['judge_comment']}",
                        f"- 缺失事实：{item['missing_facts'] or '无'}",
                        f"- 使用工具：{item['tools_used']}（预期 {item['expected_tools']}）",
                        f"- 检索效率：{item['efficiency_note']}",
                    ]
                )
                if item.get("closer_to") not in (None, "n/a"):
                    lines.extend(
                        [
                            f"- 归因评审：{item.get('attribution_comment', '')}",
                            f"- 未满足的归因要求：{item.get('attribution_missing') or '无'}",
                            f"- 命中的坏习惯：{item.get('violated_patterns') or '无'}",
                            f"- 更接近：{item.get('closer_to')}答案示例",
                        ]
                    )
                if item.get("agent_errors"):
                    lines.append(f"- 运行异常：{item['agent_errors']}")
                if item.get("error"):
                    lines.append(f"- 执行错误：{item['error']}")
                lines.append("")

        return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="运行金融研报 Agent 自动化评测")
    parser.add_argument("--cases", type=str, default="", help="用例 ID，逗号分隔，留空跑全部")
    parser.add_argument("--no-save", action="store_true", help="不保存结果文件")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    case_ids = [int(item) for item in args.cases.split(",")] if args.cases else None
    outcome = Evaluator().run(case_ids=case_ids, save_report=not args.no_save)

    print("\n" + "=" * 60)
    print(json.dumps(outcome["summary"], ensure_ascii=False, indent=2))
    if outcome.get("report_path"):
        print(f"\n详细结果: {outcome['report_path']}")


if __name__ == "__main__":
    main()
