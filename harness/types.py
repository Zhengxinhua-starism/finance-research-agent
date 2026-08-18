"""Harness 层的核心数据契约。

解决什么问题
    Agent 系统里最容易腐化的地方是"模块间靠 dict 传数据"：Retriever 塞了个
    {"ok": True}，Verifier 读的是 result["success"]，中间没有任何地方报错，
    只有最终研报少了一段内容。本模块把 Harness 各层之间流转的对象全部定为
    显式类型，让契约违约在构造对象时就暴露。

核心设计决策
    1. 状态用字符串字面量（Literal）而不是 Enum。理由：这些对象要频繁
       序列化进 JSON trace 和 Redis，Enum 需要额外的 encoder/decoder；
       Literal 天然是字符串，同时仍受 Pydantic 校验保护。为了避免手写魔法字符串
       拼错，另外提供了同名常量（TOOL_ERROR_TIMEOUT 等）。
    2. ToolResult 同时保留 data（结构化）与 content（给 LLM 看的文本）。
       LLM 只能读文本，但 Evidence Gate 需要拿原始数字做校验，两者不能互相替代。
    3. Evidence 强制带 disclosure_date 和 source_type。前视偏差（用未来数据
       做历史判断）是金融场景最严重的错误，把披露日期做成必填字段而不是
       metadata 里的可选键，是为了让"忘记填"变成构造期错误。
    4. Claim.numbers 存"结构化数字"而不是让门禁去正则原文。数字提取的
       职责归属于生产 claim 的一方（Verifier），门禁只做比对，职责单一。

为什么不用其他方案
    - 不用 dataclass：需要 JSON schema 导出（给 LLM 做结构化输出）和字段校验，
      Pydantic 一步到位，dataclass 要额外接 marshmallow / cattrs。
    - 不用 TypedDict：TypedDict 只有静态检查，运行期不校验；Agent 系统的
      错误大多来自 LLM 返回的脏数据，必须有运行期校验。
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

# ============================================================
# 状态字面量常量（避免手写魔法字符串）
# ============================================================

AgentStatus = Literal["completed", "max_turns", "token_limit", "error"]
AGENT_COMPLETED: AgentStatus = "completed"
AGENT_MAX_TURNS: AgentStatus = "max_turns"
AGENT_TOKEN_LIMIT: AgentStatus = "token_limit"
AGENT_ERROR: AgentStatus = "error"

ToolErrorType = Literal[
    "tool_not_found", "param_error", "execution_error", "timeout", "cancelled"
]
TOOL_ERROR_NOT_FOUND: ToolErrorType = "tool_not_found"
TOOL_ERROR_PARAM: ToolErrorType = "param_error"
TOOL_ERROR_EXECUTION: ToolErrorType = "execution_error"
TOOL_ERROR_TIMEOUT: ToolErrorType = "timeout"
TOOL_ERROR_CANCELLED: ToolErrorType = "cancelled"

GateStatus = Literal["verified", "unverified", "number_mismatch", "lookahead_blocked"]
GATE_VERIFIED: GateStatus = "verified"
GATE_UNVERIFIED: GateStatus = "unverified"
GATE_NUMBER_MISMATCH: GateStatus = "number_mismatch"
GATE_LOOKAHEAD_BLOCKED: GateStatus = "lookahead_blocked"

# 证据来源类型，同时决定门禁的可信度优先级
SourceType = Literal[
    "annual_report",  # 年报
    "interim_report",  # 中报
    "quarterly_report",  # 季报
    "announcement",  # 交易所公告（业绩预告、产销快报、重大事项）
    "derived",  # 由上述报表推算得出（例如毛利率 = (营收-成本)/营收）
    "market_data",  # 行情数据
    "research_report",  # 券商研报（机构判断，非事实）
    "knowledge_base",  # 内部知识库（分析框架、行业常识）
    "news",  # 新闻媒体（可信度最低，最高只能标 ⚠️未验证）
]

# 数值越大越可信；Evidence Gate 在多条证据命中时优先取高优先级来源
SOURCE_PRIORITY: dict[str, int] = {
    "annual_report": 100,
    "interim_report": 90,
    "quarterly_report": 80,
    "announcement": 70,
    "derived": 60,
    "market_data": 50,
    "research_report": 40,
    "knowledge_base": 30,
    "news": 10,
}

# ---- 三层可信度模型 ----
# 把 9 种来源类型归并成 3 个层级，用于研报里向读者呈现可信度。
#
# 为什么需要在 SOURCE_PRIORITY 之外再有一套层级：优先级是给**门禁**用的
# （多条证据命中时取哪条），是个连续的排序；层级是给**读者**用的，
# 需要的是"这条能不能信"的离散判断。9 个数字对读者没有意义，3 个层级有。
#
# 分层标准来自 A 股研究的通行实践：
#   第一层 已披露事实——上市公司依法披露的正式文件，可作为事实引用；
#   第二层 机构判断——有署名、有方法论，但是观点不是事实，必须标明"某机构认为"；
#   第三层 待验证信息——媒体转述与传闻，只能作为观察线索，不能写成结论。
SourceTier = Literal["disclosed_fact", "institutional_view", "unverified_info"]

SOURCE_TIER: dict[str, SourceTier] = {
    "annual_report": "disclosed_fact",
    "interim_report": "disclosed_fact",
    "quarterly_report": "disclosed_fact",
    "announcement": "disclosed_fact",
    # 由已披露数据按固定公式推算，可追溯到第一层，因此仍算事实
    "derived": "disclosed_fact",
    "market_data": "disclosed_fact",
    "research_report": "institutional_view",
    # 知识库是通用方法论，不是对该公司的判断，但也不是已披露事实。
    # 归到第二层是因为它和研报一样"有依据但需要人来判断适用性"。
    "knowledge_base": "institutional_view",
    "news": "unverified_info",
}

SOURCE_TIER_LABEL: dict[str, str] = {
    "disclosed_fact": "① 已披露事实",
    "institutional_view": "② 机构判断",
    "unverified_info": "③ 待验证信息",
}

SOURCE_TIER_USAGE: dict[str, str] = {
    "disclosed_fact": "可作为核心依据直接引用",
    "institutional_view": "作为辅助依据，须注明「机构观点」，不得写成事实",
    "unverified_info": "仅作观察线索，须注明「需结合最新披露验证」",
}

# 研报三级标注（对应 md "三级标注规则"）
ClaimVerdict = Literal["verified", "unverified", "refused"]
VERDICT_VERIFIED: ClaimVerdict = "verified"
VERDICT_UNVERIFIED: ClaimVerdict = "unverified"
VERDICT_REFUSED: ClaimVerdict = "refused"

VERDICT_LABEL: dict[str, str] = {
    "verified": "✅已验证",
    "unverified": "⚠️未验证",
    "refused": "❌拒答",
}

# 数字比对的语义类型，决定用哪种容差
NumberKind = Literal["absolute", "ratio", "growth"]


def new_id(prefix: str) -> str:
    """生成短 ID。用 uuid4 前 8 位，够用且不会污染 trace 可读性。"""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def utc_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


class HarnessModel(BaseModel):
    """所有 Harness 数据类的基类，统一序列化行为。"""

    model_config = ConfigDict(
        extra="forbid",  # 拼错字段名立刻报错，而不是静默丢失
        validate_assignment=True,
        ser_json_timedelta="iso8601",
    )


# ============================================================
# 工具调用
# ============================================================


class ToolCall(HarnessModel):
    """LLM 发起的一次工具调用请求（对齐 OpenAI function calling 的 tool_call）。"""

    id: str = Field(default_factory=lambda: new_id("call"))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # 原始 arguments 字符串。LLM 可能返回非法 JSON，保留原文便于 trace 定位问题
    raw_arguments: str | None = None

    def summary(self, limit: int = 200) -> str:
        text = f"{self.name}({self.arguments})"
        return text if len(text) <= limit else text[:limit] + "..."


class ToolResult(HarnessModel):
    """一次工具执行的结果。

    success=True 时 data/content 有效；success=False 时 error/error_type 有效。
    两者互斥，由 ok()/failed() 两个构造器保证，不要直接手搓。
    """

    tool_call_id: str
    tool_name: str
    success: bool
    # 结构化结果：给 Evidence Gate / 指标计算用，保留原始数字精度
    data: Any = None
    # 文本结果：给 LLM 读，已做过格式化和长度控制
    content: str = ""
    error: str | None = None
    error_type: ToolErrorType | None = None
    duration_ms: int = 0
    from_cache: bool = False
    # 缓存层级："redis" / "disk" / None
    cache_layer: str | None = None

    @classmethod
    def ok(
        cls,
        tool_call_id: str,
        tool_name: str,
        data: Any,
        content: str,
        duration_ms: int = 0,
        from_cache: bool = False,
        cache_layer: str | None = None,
    ) -> "ToolResult":
        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            success=True,
            data=data,
            content=content,
            duration_ms=duration_ms,
            from_cache=from_cache,
            cache_layer=cache_layer,
        )

    @classmethod
    def failed(
        cls,
        tool_call_id: str,
        tool_name: str,
        error: str,
        error_type: ToolErrorType,
        duration_ms: int = 0,
    ) -> "ToolResult":
        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            success=False,
            error=error,
            error_type=error_type,
            # 失败也要给 LLM 一段可读文本，否则它会以为工具没被调用而无限重试
            content=f"[工具执行失败] {tool_name}: {error_type} — {error}",
            duration_ms=duration_ms,
        )

    def to_message(self) -> dict[str, str]:
        """转成 OpenAI 格式的 tool 角色消息，回填进对话历史。"""
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.tool_name,
            "content": self.content,
        }


@runtime_checkable
class ToolProtocol(Protocol):
    """工具契约。任何实现了这四个成员的对象都能注册进 ToolRegistry。

    用 Protocol 而不是抽象基类：工具实现分散在 tools/ 和 rag/ 两个包里，
    继承会强制它们依赖 harness 包，形成"业务层依赖运行时层"的反向依赖。
    Protocol 是结构化类型，工具只需要长得对，不需要 import Harness。
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema

    def run(self, **kwargs: Any) -> Any: ...


# ============================================================
# 证据与断言
# ============================================================


class Evidence(HarnessModel):
    """一条可被引用的证据。

    numbers 字段是"这条证据里出现的、可被校验的数字"，键为指标名
    （revenue / gross_margin / roe ...），值为标准单位下的浮点数：
    金额单位统一为元，比率统一为小数（0.218 表示 21.8%）。
    单位不统一是金融数据校验最常见的 bug 来源，所以在类型层面固定死。
    """

    evidence_id: str = Field(default_factory=lambda: new_id("ev"))
    source_type: SourceType
    source_name: str  # 例如 "比亚迪2024年年报" / "利润表推算"
    # 披露日期：前视偏差检查的关键字段，强制必填
    disclosure_date: date
    # 报告期（数据本身对应的时点），与披露日期不同：2024 年报的报告期是
    # 2024-12-31，披露日期是 2025-03-28
    period_end: date | None = None
    content: str = ""
    numbers: dict[str, float] = Field(default_factory=dict)
    ticker: str | None = None
    company: str | None = None
    # 合并报表 / 母公司报表，用于会计口径一致性检查
    statement_scope: Literal["consolidated", "parent"] = "consolidated"
    tool_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def priority(self) -> int:
        return SOURCE_PRIORITY.get(self.source_type, 0)

    @property
    def is_official(self) -> bool:
        """是否为官方披露来源（第一层）。

        只有第一层能让断言拿到 ✅已验证。研报虽然可信度高于新闻，
        但它是**判断**不是**事实**——把机构观点标成"已验证"，
        等于把别人的推测包装成客观依据，这是本系统最该避免的事。
        """
        return self.tier == "disclosed_fact"

    @property
    def tier(self) -> str:
        return SOURCE_TIER.get(self.source_type, "unverified_info")

    @property
    def tier_label(self) -> str:
        return SOURCE_TIER_LABEL.get(self.tier, "③ 待验证信息")

    def citation(self) -> str:
        """生成研报里用的来源标注文本。"""
        return f"{self.source_name}（披露日 {self.disclosure_date.isoformat()}）"


class ClaimNumber(HarnessModel):
    """断言里出现的一个待校验数字。"""

    metric: str  # 指标名，需与 Evidence.numbers 的键对齐
    value: float  # 标准单位：金额=元，比率=小数
    kind: NumberKind = "absolute"
    # 原文里的展示形式，例如 "7771亿元" / "21.8%"，仅用于报错时给人看
    raw_text: str | None = None
    # 该数字所属的报告期。支持 "2024-12-31" 或 "2024" 两种写法，None 表示未指定。
    #
    # 这个字段是 F-018 的修复核心：没有它，"毛利率由 2024 年的 19.44% 降至
    # 2025 年的 17.74%" 会产生两个 metric 同为 gross_margin 的数字，
    # 门禁只能拿同一条证据去比，必然有一个对不上——而跨期对比恰恰是
    # 归因分析最常见的表述方式。
    period: str | None = None

    def period_year(self) -> int | None:
        """取报告期的年份，用于与证据的 period_end 比对。"""
        if not self.period:
            return None
        import re

        match = re.search(r"(19|20)\d{2}", str(self.period))
        return int(match.group()) if match else None


class Claim(HarnessModel):
    """研报中的一条断言，是证据门禁的检查单元。"""

    claim_id: str = Field(default_factory=lambda: new_id("claim"))
    text: str
    numbers: list[ClaimNumber] = Field(default_factory=list)
    # 分析基准日：任何披露日期晚于它的证据都构成前视偏差
    as_of_date: date
    # Verifier 认为这条断言来自哪里，用于缩小证据搜索范围（可为空）
    source_hint: str | None = None
    ticker: str | None = None
    statement_scope: Literal["consolidated", "parent"] = "consolidated"
    # 所属研报小节，用于渲染时归位
    section: str | None = None


class GateCheckDetail(HarnessModel):
    """单级门禁的检查明细，用于 trace 和失败归因。"""

    level: Literal["source", "number", "timing", "accounting"]
    passed: bool
    reason: str = ""


class GateResult(HarnessModel):
    """证据门禁的判定结果。"""

    claim_id: str
    status: GateStatus
    matched_evidence_id: str | None = None
    matched_source: str | None = None
    reason: str = ""
    checks: list[GateCheckDetail] = Field(default_factory=list)
    # 数字不一致时记录差异，便于人工复核
    number_diffs: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def verdict(self) -> ClaimVerdict:
        """把四种门禁状态映射到研报的三级标注。

        number_mismatch 和 lookahead_blocked 都归为 refused 而不是 unverified：
        前者说明模型算错或编造了数字，后者说明存在前视偏差，
        两者都属于"这句话不能出现在研报里"，而不是"存疑但可以写"。
        """
        if self.status == GATE_VERIFIED:
            return VERDICT_VERIFIED
        if self.status == GATE_UNVERIFIED:
            return VERDICT_UNVERIFIED
        return VERDICT_REFUSED


# ============================================================
# RAG 数据类型
# ============================================================


class Document(HarnessModel):
    """知识库文档单元（切分后的 chunk）。"""

    doc_id: str = Field(default_factory=lambda: new_id("doc"))
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(HarnessModel):
    """一条检索结果。

    同时保留 score（融合前的原始分）、rrf_score（融合分）、rerank_score（精排分）
    三个字段而不是覆盖同一个 score：调试检索质量时需要看清楚是哪一路召回的、
    RRF 有没有把它顶上来、Rerank 有没有把它压下去。
    """

    doc_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    rrf_score: float | None = None
    rerank_score: float | None = None
    retrieval_method: Literal["bm25", "vector", "hybrid", "rerank"] = "hybrid"
    # 该文档在各路检索中的名次，例如 {"bm25": 3, "vector": 11}
    ranks: dict[str, int] = Field(default_factory=dict)

    @property
    def final_score(self) -> float:
        """排序用的最终分数：有精排分用精排分，否则用融合分，再否则用原始分。"""
        if self.rerank_score is not None:
            return self.rerank_score
        if self.rrf_score is not None:
            return self.rrf_score
        return self.score


# ============================================================
# Agent 运行结果
# ============================================================


class TokenCounter(HarnessModel):
    """token 累计计数器。"""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "input": self.input_tokens,
            "output": self.output_tokens,
            "total": self.total,
        }


class AgentResult(HarnessModel):
    """一次 AgentLoop 运行的完整结果。

    注意 status != "completed" 时 final_text 仍可能非空（例如 max_turns 时
    保留最后一次 LLM 输出）。调用方必须显式检查 status，不能只看 final_text
    是否为空——这是"静默降级"类 bug 的高发点。
    """

    status: AgentStatus
    final_text: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    turns: int = 0
    token_usage: TokenCounter = Field(default_factory=TokenCounter)
    duration_ms: int = 0
    error: str | None = None
    agent_name: str = "unnamed"

    @property
    def is_ok(self) -> bool:
        return self.status == AGENT_COMPLETED

    @property
    def successful_tool_calls(self) -> int:
        return sum(1 for result in self.tool_results if result.success)

    @property
    def failed_tool_calls(self) -> int:
        return sum(1 for result in self.tool_results if not result.success)
