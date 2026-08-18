"""LangGraph 状态定义。

解决什么问题
    四个 Agent 节点之间要传递的东西不止一个"消息列表"：研究计划、证据池、
    检索轮次、核查结果、研报、错误记录。如果用一个自由的 dict 传，
    很快就会出现"某个节点写了 evidence，另一个节点读的是 evidences"
    这类问题——而且不会报错，只会让研报少一半内容。

核心设计决策
    1. 用 TypedDict 而不是 Pydantic BaseModel。这不是偏好问题：
       LangGraph 的 StateGraph 要求状态是 dict-like 且支持通过
       "节点返回部分字典"来做增量更新（partial update）。
       Pydantic 模型需要每次全量构造，会把节点写成
       `return state.model_copy(update={...})`，既啰嗦又容易漏字段。
    2. evidence_pool 用 Annotated + 自定义 reducer 做**去重累加**。
       检索节点可能被执行多次（补搜循环），如果用默认的覆盖语义，
       第二轮检索会把第一轮的证据冲掉；用普通的 operator.add 又会
       在缓存命中时堆积大量重复证据。自定义 reducer 一次解决两个问题。
    3. retrieval_count 用累加 reducer 而不是让节点自己 +1。
       节点内部读旧值再写新值在并发或重试场景下会丢更新；
       让节点只返回增量（+1），由 reducer 负责合并，是更安全的写法。
    4. 状态里同时存 report_markdown（给人看）和 report_dict（给 API 返回），
       而不是只存一个再临时转换。API 层不应该承担渲染职责。

为什么不用其他方案
    - 不把 Evidence 存成 dict：证据要参与门禁计算（日期比较、数字比对），
      存 dict 会让每个使用点都要先反序列化。LangGraph 的状态本身
      不要求可 JSON 序列化，只有需要 checkpoint 持久化时才要求——
      本项目的会话持久化走 Redis（存的是渲染后的结果），不用 checkpointer。
"""

from __future__ import annotations

import operator
from datetime import date
from typing import Annotated, Any, Literal, Sequence, TypedDict

from agents.planner import ResearchPlan
from agents.verifier import VerifyResult
from harness.types import Evidence

# 研究流程的整体状态
ResearchStatus = Literal[
    "pending", "planning", "retrieving", "verifying", "writing", "completed", "failed"
]


def merge_evidence(
    existing: Sequence[Evidence] | None, incoming: Sequence[Evidence] | None
) -> list[Evidence]:
    """证据池 reducer：累加 + 按指纹去重。

    指纹取 (来源名, 报告期, 数字集合)。同一条证据在补搜时被重复获取
    （缓存命中，几乎零耗时）是常态，去重能避免证据池膨胀影响
    门禁匹配效率和 evidence_coverage 指标的准确性。
    """
    merged: list[Evidence] = list(existing or [])
    seen = {_evidence_fingerprint(evidence) for evidence in merged}
    for evidence in incoming or []:
        fingerprint = _evidence_fingerprint(evidence)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        merged.append(evidence)
    return merged


def _evidence_fingerprint(evidence: Evidence) -> tuple[str, str, str]:
    numbers = ",".join(f"{k}={v:.6g}" for k, v in sorted(evidence.numbers.items()))
    period = evidence.period_end.isoformat() if evidence.period_end else ""
    return (evidence.source_name, period, numbers or evidence.content[:80])


def merge_dict(existing: dict[str, Any] | None, incoming: dict[str, Any] | None) -> dict[str, Any]:
    """字典 reducer：浅合并。用于 token 统计这类逐节点累积的指标。"""
    result = dict(existing or {})
    for key, value in (incoming or {}).items():
        if isinstance(value, (int, float)) and isinstance(result.get(key), (int, float)):
            result[key] = result[key] + value
        else:
            result[key] = value
    return result


class ResearchState(TypedDict, total=False):
    """研究流程的完整状态。

    total=False 让所有键都是可选的，这样节点可以只返回自己修改的部分
    （LangGraph 的 partial update 语义）。
    """

    # ---- 输入（由调用方设置，节点只读）----
    run_id: str
    question: str
    ticker: str
    company: str
    as_of_date: date
    session_id: str

    # ---- Planner 产出 ----
    plan: ResearchPlan | None

    # ---- Retriever 产出 ----
    # 累加去重：补搜轮次的证据要叠加到已有证据上
    evidence_pool: Annotated[list[Evidence], merge_evidence]
    # 累加：节点返回 1 表示"本次又检索了一轮"
    retrieval_count: Annotated[int, operator.add]
    retriever_summary: str
    tool_calls_made: Annotated[list[str], operator.add]

    # ---- Verifier 产出 ----
    verify_result: VerifyResult | None
    needs_more_retrieval: bool
    expand_query: str | None

    # ---- Writer 产出 ----
    report_markdown: str
    report_dict: dict[str, Any]

    # ---- 流程控制与观测 ----
    status: ResearchStatus
    current_node: str
    # 累加：每个节点遇到的非致命错误都追加进来，不覆盖
    errors: Annotated[list[str], operator.add]
    node_path: Annotated[list[str], operator.add]
    token_usage: Annotated[dict[str, int], merge_dict]


def create_initial_state(
    run_id: str,
    question: str,
    ticker: str = "",
    company: str = "",
    as_of_date: date | None = None,
    session_id: str = "",
) -> ResearchState:
    """构造初始状态。

    所有带 reducer 的字段必须显式初始化：LangGraph 在第一次调用 reducer 时
    会把 None 作为 existing 传入，虽然本文件的 reducer 都做了 None 处理，
    但 operator.add 不行（None + 1 会抛 TypeError）。
    """
    return ResearchState(
        run_id=run_id,
        question=question,
        ticker=ticker,
        company=company,
        as_of_date=as_of_date or date.today(),
        session_id=session_id,
        plan=None,
        evidence_pool=[],
        retrieval_count=0,
        retriever_summary="",
        tool_calls_made=[],
        verify_result=None,
        needs_more_retrieval=False,
        expand_query=None,
        report_markdown="",
        report_dict={},
        status="pending",
        current_node="",
        errors=[],
        node_path=[],
        token_usage={"input": 0, "output": 0, "total": 0},
    )


def state_summary(state: ResearchState) -> dict[str, Any]:
    """给日志和 trace 用的状态摘要（不含大字段）。"""
    plan = state.get("plan")
    verify_result = state.get("verify_result")
    return {
        "run_id": state.get("run_id", ""),
        "status": state.get("status", "pending"),
        "current_node": state.get("current_node", ""),
        "question_type": getattr(plan, "question_type", None),
        "evidence_count": len(state.get("evidence_pool") or []),
        "retrieval_count": state.get("retrieval_count", 0),
        "claim_count": len(getattr(verify_result, "claims", []) or []),
        "unverified_ratio": getattr(verify_result, "unverified_ratio", None),
        "error_count": len(state.get("errors") or []),
        "node_path": state.get("node_path") or [],
        "token_usage": state.get("token_usage") or {},
    }


__all__ = [
    "ResearchState",
    "ResearchStatus",
    "create_initial_state",
    "merge_dict",
    "merge_evidence",
    "state_summary",
]
