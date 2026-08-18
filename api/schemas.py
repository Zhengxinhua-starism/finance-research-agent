"""API 请求 / 响应模型。

解决什么问题
    API 的边界契约。这一层的模型与内部的 Harness 类型刻意分开：
    内部类型（Evidence、GateResult）会随实现演进频繁调整，
    而 API 契约一旦发布就要保持稳定。把它们混用会导致
    "重构内部数据结构 → API 响应结构跟着变 → 前端挂掉"。

核心设计决策
    1. 请求侧做严格校验（股票代码必须 6 位数字、日期必须 YYYY-MM-DD），
       在 Pydantic 层就拦掉错误输入，而不是等 AKShare 返回空表再报错。
       前者返回 422 和明确的字段错误，后者返回 500 和一句"数据获取失败"。
    2. as_of_date 允许留空并默认今天，但**不允许未来日期**。
       未来日期会让前视偏差检查失效（任何数据都"不晚于"未来日期），
       等于把整个证据门禁的时点检查关掉。
    3. 响应模型都带 examples，直接生成有意义的 OpenAPI 文档。
       /docs 页面能点"Try it out"跑通，是这类项目最省事的演示方式。
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TICKER_PATTERN = re.compile(r"^\d{6}$")


class ResearchRequest(BaseModel):
    """POST /api/research 请求体。"""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "比亚迪2024年毛利率为什么下降",
                    "company_ticker": "002594",
                    "as_of_date": "2025-06-30",
                }
            ]
        }
    )

    question: str = Field(min_length=2, max_length=500, description="研究问题")
    company_ticker: str = Field(description="A股股票代码，6位数字")
    as_of_date: str | None = Field(
        default=None, description="分析基准日 YYYY-MM-DD，留空则为今天"
    )
    company_name: str | None = Field(default=None, description="可选，公司名称")
    session_id: str | None = Field(default=None, description="可选，复用已有会话 ID")
    async_mode: bool = Field(
        default=False,
        description="true 则立即返回 session_id，通过 GET /api/session/{id} 轮询结果",
    )

    @field_validator("company_ticker")
    @classmethod
    def _validate_ticker(cls, value: str) -> str:
        cleaned = re.sub(r"[^0-9]", "", str(value).strip())
        if not TICKER_PATTERN.match(cleaned):
            raise ValueError(
                f"股票代码必须是 6 位数字（收到 {value!r}）。"
                "示例：000001（平安银行）、002594（比亚迪）、600519（贵州茅台）"
            )
        return cleaned

    @field_validator("as_of_date")
    @classmethod
    def _validate_as_of_date(cls, value: str | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        try:
            parsed = date.fromisoformat(str(value).strip())
        except ValueError as exc:
            raise ValueError(f"as_of_date 必须是 YYYY-MM-DD 格式（收到 {value!r}）") from exc
        if parsed > date.today():
            # 未来日期会让前视偏差检查形同虚设
            raise ValueError(
                f"as_of_date 不能是未来日期（收到 {parsed}，今天 {date.today()}）。"
                "分析基准日晚于当前日期会使前视偏差检查失效"
            )
        return parsed.isoformat()

    def resolved_as_of_date(self) -> date:
        return date.fromisoformat(self.as_of_date) if self.as_of_date else date.today()


class ConclusionItem(BaseModel):
    """研报里的一条结论。"""

    verdict: Literal["verified", "unverified", "refused"]
    label: str
    text: str
    source: str | None = None
    reason: str = ""
    section: str = "conclusion"


class SourceRow(BaseModel):
    """数据来源表的一行。"""

    item: str
    source: str
    disclosure_date: str
    status: str


class ReportPayload(BaseModel):
    """三级标注研报。"""

    title: str = ""
    markdown: str = ""
    conclusions: list[ConclusionItem] = Field(default_factory=list)
    analysis_body: str = ""
    source_table: list[SourceRow] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    refused: bool = False


class TraceSummaryPayload(BaseModel):
    """trace 聚合摘要。"""

    model_config = ConfigDict(extra="allow")

    run_id: str = ""
    total_duration_ms: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_call_count: int = 0
    tool_call_count: int = 0
    tool_failure_count: int = 0
    retrieval_count: int = 0
    gate_check_count: int = 0
    gate_blocked_count: int = 0
    tool_usage: dict[str, int] = Field(default_factory=dict)
    node_path: list[str] = Field(default_factory=list)
    status: str = "unknown"


class ResearchResponse(BaseModel):
    """POST /api/research 响应体。"""

    run_id: str
    session_id: str
    status: Literal["completed", "running", "failed", "pending"]
    report: ReportPayload | None = None
    trace_summary: TraceSummaryPayload | None = None
    errors: list[str] = Field(default_factory=list)
    message: str = ""


class SessionResponse(BaseModel):
    """GET /api/session/{session_id} 响应体。"""

    session_id: str
    status: str
    current_node: str = ""
    node_path: list[str] = Field(default_factory=list)
    progress: float = 0.0
    progress_label: str = ""
    question: str = ""
    company: str = ""
    ticker: str = ""
    created_at: str = ""
    updated_at: str = ""
    run_id: str | None = None
    result: ReportPayload | None = None
    trace_summary: TraceSummaryPayload | None = None
    error: str | None = None


class TraceEventPayload(BaseModel):
    """单条 trace 事件。"""

    model_config = ConfigDict(extra="allow")

    timestamp: str = ""
    event_type: str = ""
    agent_name: str = ""
    input_summary: str = ""
    output_summary: str = ""
    duration_ms: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)
    success: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceResponse(BaseModel):
    """GET /api/trace/{run_id} 响应体。"""

    run_id: str
    summary: TraceSummaryPayload
    final_result: str = ""
    events: list[TraceEventPayload] = Field(default_factory=list)


class EvalRequest(BaseModel):
    """POST /api/eval 请求体。"""

    model_config = ConfigDict(json_schema_extra={"examples": [{"test_case_ids": [1, 2, 3]}]})

    test_case_ids: list[int] | None = Field(
        default=None, description="要跑的用例 ID 列表，留空则跑全部 9 道题"
    )
    save_report: bool = Field(default=True, description="是否把评测结果落盘到 eval/results/")


class EvalCaseResult(BaseModel):
    """单个测试用例的评测结果。"""

    model_config = ConfigDict(extra="allow")

    case_id: int
    question: str
    company: str = ""
    difficulty: str = ""
    question_type: str = ""
    passed: bool = False
    factual_accuracy: float = 0.0
    refusal_calibration: float = 0.0
    retrieval_efficiency: float = 0.0
    evidence_coverage: float = 0.0
    overall_score: float = 0.0
    judge_comment: str = ""
    matched_facts: list[str] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    error: str | None = None


class EvalResponse(BaseModel):
    """POST /api/eval 响应体。"""

    results: list[EvalCaseResult] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    report_path: str | None = None


class HealthResponse(BaseModel):
    """GET /api/health 响应体。"""

    status: Literal["ok", "degraded", "error"]
    version: str = "1.0.0"
    redis: str = "unknown"
    chroma: str = "unknown"
    llm: dict[str, Any] = Field(default_factory=dict)
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    knowledge_base_documents: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """统一错误响应。"""

    error: str
    error_type: str = "internal_error"
    detail: str = ""


__all__ = [
    "ConclusionItem",
    "ErrorResponse",
    "EvalCaseResult",
    "EvalRequest",
    "EvalResponse",
    "HealthResponse",
    "ReportPayload",
    "ResearchRequest",
    "ResearchResponse",
    "SessionResponse",
    "SourceRow",
    "TraceEventPayload",
    "TraceResponse",
    "TraceSummaryPayload",
]
