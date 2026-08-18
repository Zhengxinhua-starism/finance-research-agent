"""运行 trace 记录。

解决什么问题
    Agent 是非确定性系统：同一个问题两次运行可能走不同的工具路径、
    产生不同的结论。出问题时"看日志"往往只能看到最后的异常栈，
    看不到"它为什么决定调这个工具"、"哪一步 token 爆了"、"证据门禁在哪句话上拦截了"。
    本模块把每次运行拆成结构化事件流并落盘为 JSON，让每次运行都可复盘。

核心设计决策
    1. 事件模型固定 7 个字段（时间、类型、Agent 名、输入摘要、输出摘要、
       耗时、token），额外信息一律塞 metadata。固定字段是为了让 trace 能直接
       转成表格给 Vue 时间线渲染；如果每种事件结构都不同，前端就得写 7 个分支。
    2. 输入/输出摘要截断到 200 字符。完整内容留在 metadata 里（可选），
       默认不存。工具返回的财务报表动辄几万字符，全存会让 trace 文件比
       研报本身大两个数量级，且没人会去读。
    3. Tracer 内部用锁保护事件列表。LangGraph 的节点在同一次运行里是串行的，
       但 FastAPI 下多个请求会并发，而 Tracer 实例是 per-run 创建的——
       加锁是为了防止将来有人把它做成单例后出现难以复现的丢事件问题。
    4. 提供 span() 上下文管理器。手写 start = time.time() ... duration 的写法
       在异常路径上会漏记录，用上下文管理器保证异常时也能落一条失败事件。

为什么不用其他方案
    - 不用 LangSmith / Langfuse：它们需要外部服务和账号，面试演示环境跑不起来；
      而且这个项目的卖点之一就是"自建可观测性"，用托管方案就没得讲了。
    - 不用 Python logging：logging 输出是非结构化文本行，做不到"按 run_id
      聚合、按事件类型统计、算 P95 工具耗时"。这里两者并存：logging 给人看，
      trace JSON 给程序读。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

from config import get_config

logger = logging.getLogger(__name__)

EventType = Literal[
    "llm_call",
    "tool_call",
    "gate_check",
    "compact",
    "node_enter",
    "node_exit",
    "retrieval",
    "error",
]

SUMMARY_LIMIT = 200


def truncate(text: Any, limit: int = SUMMARY_LIMIT) -> str:
    """把任意对象转成不超过 limit 的摘要字符串。"""
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = json.dumps(text, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(text)
    text = " ".join(text.split())  # 压掉换行，保证 trace 表格单行可读
    return text if len(text) <= limit else text[: limit - 3] + "..."


class TraceEvent(BaseModel):
    """一条 trace 事件。字段顺序即 JSON 输出顺序。"""

    model_config = ConfigDict(extra="forbid")

    timestamp: str = Field(default_factory=lambda: datetime.now().astimezone().isoformat())
    event_type: EventType
    agent_name: str
    input_summary: str = ""
    output_summary: str = ""
    duration_ms: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # TODO: md 的 TraceEvent 没有 success 字段，但失败事件（工具超时、门禁拦截）
    # 在复盘时需要能被直接筛出来，靠 output_summary 里找关键字太脆弱，这里补上。
    success: bool = True


class TraceSummary(BaseModel):
    """一次运行的聚合摘要，给 API 响应和 Vue 底部统计用。"""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    question: str
    started_at: str
    ended_at: str | None = None
    total_duration_ms: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    event_count: int = 0
    llm_call_count: int = 0
    tool_call_count: int = 0
    tool_failure_count: int = 0
    retrieval_count: int = 0
    gate_check_count: int = 0
    gate_blocked_count: int = 0
    compact_count: int = 0
    # 各工具调用次数，例如 {"get_income_history": 2}
    tool_usage: dict[str, int] = Field(default_factory=dict)
    node_path: list[str] = Field(default_factory=list)
    status: str = "running"


class Tracer:
    """单次运行的 trace 收集器。用法：

        tracer = Tracer()
        tracer.start_trace(run_id, question)
        with tracer.span("tool_call", "retriever", input_summary="...") as span:
            span.set_output(result)
        tracer.end_trace(final_result, total_tokens, total_duration_ms)
        tracer.export_json()
    """

    def __init__(self, trace_dir: str | Path | None = None, autosave: bool = True):
        config = get_config()
        self.trace_dir = Path(trace_dir) if trace_dir else config.traces_dir
        self.autosave = autosave
        self._lock = threading.Lock()
        self.events: list[TraceEvent] = []
        self.run_id: str = ""
        self.question: str = ""
        self.started_at: float = 0.0
        self.started_at_iso: str = ""
        self.ended_at_iso: str | None = None
        self.final_result: str = ""
        self.status: str = "not_started"
        self._total_duration_ms: int = 0
        self._node_path: list[str] = []

    # ---------------- 生命周期 ----------------

    def start_trace(self, run_id: str, question: str) -> None:
        with self._lock:
            self.run_id = run_id
            self.question = question
            self.started_at = time.perf_counter()
            self.started_at_iso = datetime.now().astimezone().isoformat()
            self.status = "running"
            self.events.clear()
            self._node_path.clear()
        logger.info("trace 开始 run_id=%s question=%s", run_id, truncate(question, 80))

    def log_event(self, event: TraceEvent) -> None:
        with self._lock:
            self.events.append(event)
            if event.event_type == "node_enter":
                self._node_path.append(event.agent_name)
        log_fn = logger.info if event.success else logger.warning
        log_fn(
            "[%s] %s | %s -> %s (%dms)",
            event.agent_name,
            event.event_type,
            event.input_summary,
            event.output_summary,
            event.duration_ms,
        )

    def end_trace(
        self,
        final_result: str,
        total_tokens: int = 0,
        total_duration_ms: int = 0,
        status: str = "completed",
    ) -> TraceSummary:
        with self._lock:
            self.final_result = final_result
            self.status = status
            self.ended_at_iso = datetime.now().astimezone().isoformat()
            self._total_duration_ms = total_duration_ms or int(
                (time.perf_counter() - self.started_at) * 1000
            )
            if total_tokens:
                # 显式传入的 token 总数优先于事件累加值（例如上游已做过统一统计）
                self._explicit_total_tokens: int | None = total_tokens
            else:
                self._explicit_total_tokens = None
        summary = self.summary()
        if self.autosave:
            try:
                self.export_json()
            except OSError as exc:  # 落盘失败不应该让整次研究失败
                logger.error("trace 落盘失败 run_id=%s: %s", self.run_id, exc)
        logger.info(
            "trace 结束 run_id=%s status=%s tokens=%d duration=%dms",
            self.run_id,
            status,
            summary.total_tokens,
            summary.total_duration_ms,
        )
        return summary

    # ---------------- 便捷记录接口 ----------------

    def log(
        self,
        event_type: EventType,
        agent_name: str,
        input_summary: Any = "",
        output_summary: Any = "",
        duration_ms: int = 0,
        token_usage: dict[str, int] | None = None,
        success: bool = True,
        **metadata: Any,
    ) -> None:
        """一行式记录，绝大多数场景用这个而不是手动构造 TraceEvent。"""
        self.log_event(
            TraceEvent(
                event_type=event_type,
                agent_name=agent_name,
                input_summary=truncate(input_summary),
                output_summary=truncate(output_summary),
                duration_ms=duration_ms,
                token_usage=token_usage or {},
                success=success,
                metadata=metadata,
            )
        )

    @contextmanager
    def span(
        self,
        event_type: EventType,
        agent_name: str,
        input_summary: Any = "",
        **metadata: Any,
    ) -> Iterator["_Span"]:
        """自动计时的事件上下文。异常会被记录成 success=False 的事件后再抛出。"""
        span = _Span(input_summary=truncate(input_summary), metadata=dict(metadata))
        started = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span.success = False
            span.output_summary = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.log_event(
                TraceEvent(
                    event_type=event_type,
                    agent_name=agent_name,
                    input_summary=span.input_summary,
                    output_summary=truncate(span.output_summary),
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    token_usage=span.token_usage,
                    success=span.success,
                    metadata=span.metadata,
                )
            )

    @contextmanager
    def node(self, node_name: str, state_summary: Any = "") -> Iterator[None]:
        """LangGraph 节点进出的成对事件。"""
        self.log("node_enter", node_name, input_summary=state_summary)
        started = time.perf_counter()
        success, output = True, "ok"
        try:
            yield
        except Exception as exc:
            success, output = False, f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.log(
                "node_exit",
                node_name,
                output_summary=output,
                duration_ms=int((time.perf_counter() - started) * 1000),
                success=success,
            )

    # ---------------- 聚合与导出 ----------------

    def summary(self) -> TraceSummary:
        with self._lock:
            events = list(self.events)
        input_tokens = sum(e.token_usage.get("input", 0) for e in events)
        output_tokens = sum(e.token_usage.get("output", 0) for e in events)
        explicit_total = getattr(self, "_explicit_total_tokens", None)
        tool_usage: dict[str, int] = {}
        for event in events:
            if event.event_type == "tool_call":
                tool_name = event.metadata.get("tool_name") or event.agent_name
                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        return TraceSummary(
            run_id=self.run_id,
            question=self.question,
            started_at=self.started_at_iso,
            ended_at=self.ended_at_iso,
            total_duration_ms=self._total_duration_ms
            or (int((time.perf_counter() - self.started_at) * 1000) if self.started_at else 0),
            total_tokens=explicit_total
            if explicit_total is not None
            else input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            event_count=len(events),
            llm_call_count=sum(1 for e in events if e.event_type == "llm_call"),
            tool_call_count=sum(1 for e in events if e.event_type == "tool_call"),
            tool_failure_count=sum(
                1 for e in events if e.event_type == "tool_call" and not e.success
            ),
            retrieval_count=sum(1 for e in events if e.event_type == "retrieval"),
            gate_check_count=sum(1 for e in events if e.event_type == "gate_check"),
            gate_blocked_count=sum(
                1 for e in events if e.event_type == "gate_check" and not e.success
            ),
            compact_count=sum(1 for e in events if e.event_type == "compact"),
            tool_usage=tool_usage,
            node_path=list(self._node_path),
            status=self.status,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary().model_dump(),
            "final_result": self.final_result,
            "events": [e.model_dump() for e in self.events],
        }

    def export_json(self, path: str | Path | None = None) -> str:
        """写入 traces/run_{run_id}_{timestamp}.json，返回文件路径。"""
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        if path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_run_id = "".join(c for c in self.run_id if c.isalnum() or c in "-_") or "unknown"
            path = self.trace_dir / f"run_{safe_run_id}_{stamp}.json"
        path = Path(path)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        self.saved_path = str(path)
        return str(path)


class _Span:
    """span() 上下文里暴露给调用方的可写句柄。"""

    def __init__(self, input_summary: str, metadata: dict[str, Any]):
        self.input_summary = input_summary
        self.output_summary: str = ""
        self.token_usage: dict[str, int] = {}
        self.metadata = metadata
        self.success = True

    def set_output(self, output: Any) -> None:
        self.output_summary = truncate(output)

    def set_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self.token_usage = {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        }

    def add_metadata(self, **kwargs: Any) -> None:
        self.metadata.update(kwargs)

    def mark_failed(self, reason: str) -> None:
        self.success = False
        self.output_summary = reason


class NullTracer(Tracer):
    """空实现。让所有下游模块可以无条件调用 tracer.xxx()，

    不必到处写 `if self.tracer is not None`。这类 None 检查散布在代码里
    会让主流程被噪音淹没，而且总有一处会漏掉导致 AttributeError。
    """

    def __init__(self) -> None:
        super().__init__(autosave=False)

    def log_event(self, event: TraceEvent) -> None:  # noqa: D102
        with self._lock:
            self.events.append(event)

    def export_json(self, path: str | Path | None = None) -> str:  # noqa: D102
        return ""


def load_trace(path: str | Path) -> dict[str, Any]:
    """读取已落盘的 trace 文件，给 /api/trace 路由用。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_trace_file(run_id: str, trace_dir: str | Path | None = None) -> Path | None:
    """按 run_id 查找 trace 文件。同一 run_id 有多份时取最新的。"""
    directory = Path(trace_dir) if trace_dir else get_config().traces_dir
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(f"run_{run_id}_*.json"), reverse=True)
    return candidates[0] if candidates else None
