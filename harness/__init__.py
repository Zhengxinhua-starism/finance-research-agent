"""自建 Agent 运行时（Harness）。

这一层刻意不依赖 LangGraph：LangGraph 负责"多个 Agent 之间怎么流转"（编排），
Harness 负责"单个 Agent 内部怎么跑"（ReAct 循环、工具调度、证据校验、上下文压缩）。
两者职责正交，分开后可以单独替换编排框架而不动运行时逻辑。
"""

from harness.types import (
    AGENT_COMPLETED,
    AGENT_ERROR,
    AGENT_MAX_TURNS,
    AGENT_TOKEN_LIMIT,
    GATE_LOOKAHEAD_BLOCKED,
    GATE_NUMBER_MISMATCH,
    GATE_UNVERIFIED,
    GATE_VERIFIED,
    VERDICT_LABEL,
    AgentResult,
    Claim,
    ClaimNumber,
    Document,
    Evidence,
    GateResult,
    RetrievalResult,
    TokenCounter,
    ToolCall,
    ToolProtocol,
    ToolResult,
)

__all__ = [
    "AGENT_COMPLETED",
    "AGENT_ERROR",
    "AGENT_MAX_TURNS",
    "AGENT_TOKEN_LIMIT",
    "GATE_LOOKAHEAD_BLOCKED",
    "GATE_NUMBER_MISMATCH",
    "GATE_UNVERIFIED",
    "GATE_VERIFIED",
    "VERDICT_LABEL",
    "AgentResult",
    "Claim",
    "ClaimNumber",
    "Document",
    "Evidence",
    "GateResult",
    "RetrievalResult",
    "TokenCounter",
    "ToolCall",
    "ToolProtocol",
    "ToolResult",
]
