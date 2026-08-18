"""ReAct 主循环——自建 Agent 运行时的核心。

解决什么问题
    "让 LLM 自己决定调什么工具、调几次、什么时候停"是 Agent 的本质，
    但这个自由度直接带来三个生产风险：
    (1) 模型在两个工具之间反复横跳，永不收敛（成本无上限）；
    (2) 每轮追加工具结果导致上下文爆炸（API 直接报错）；
    (3) 某个工具挂了，模型不知道，继续基于空数据编造结论。
    本模块用 max_turns 保护、上下文压缩、失败结果显式回喂三件事把风险闭合。

核心设计决策
    1. 循环终止条件是"LLM 返回纯文本（无 tool_calls）"，而不是"检测到某个
       结束标记"。前者是模型的自然行为，后者需要 prompt 约定，模型忘了写
       标记就会死循环。
    2. max_turns 耗尽时不抛异常，而是返回 status="max_turns" 的 AgentResult，
       并且**保留已获得的工具结果**。上层（Verifier）可以用不完整的证据
       写一份带 ⚠️ 标注的研报，这比整体失败对用户更有价值。
       这就是 md 里强调的"处理非确定性"：降级而不是崩溃。
    3. 工具失败的结果照样回填进 messages。让模型看见 "[工具执行失败]
       get_cash_flow: timeout"，它会自己改用别的工具或说明数据缺失；
       如果静默跳过，模型会以为工具返回了空数据。
    4. token 上限检查放在每次 LLM 调用**之后**而不是之前。之前检查需要精确
       预估请求体大小，而 API 返回的 usage 是权威值；用权威值判断，
       代价只是多花一次调用的钱，换来判断准确。
    5. 每轮先压缩再调用。压缩阈值（context_max_tokens）远低于模型窗口，
       给单轮工具返回留出余量——等到接近窗口上限再压就已经晚了。
       工具 JSON 的精确数字由 evidence_pool 供给门禁；消息历史里的工具
       结果按时间距离分层摘要，避免"全部保留"把预算一次性打满。

为什么不用其他方案
    - 不用 LangGraph 的 ToolNode 跑单 Agent 内循环：LangGraph 在这里的价值是
      "多 Agent 之间的状态流转"，把单 Agent 的 ReAct 也交给它，
      会让 max_turns、压缩时机、失败回喂这些策略藏进框架配置里，
      面试时既讲不清也改不动。分层是有意为之：编排交框架，运行时自建。
    - 不做流式输出：ReAct 中间轮次的输出用户不需要看，流式只增加复杂度。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Sequence

from config import get_config
from harness.context_compact import ContextCompact, count_messages_tokens
from harness.tool_registry import ToolRegistry
from harness.tracing import NullTracer, Tracer, truncate
from harness.types import (
    AGENT_COMPLETED,
    AGENT_ERROR,
    AGENT_MAX_TURNS,
    AGENT_TOKEN_LIMIT,
    AgentResult,
    TokenCounter,
    ToolResult,
)
from llm.schemas import LLMResponse

logger = logging.getLogger(__name__)

# max_turns 耗尽时追加的收尾提示。让模型基于已有信息给结论，
# 而不是继续尝试调工具——此时再调也不会被执行。
FINAL_TURN_INSTRUCTION = (
    "【系统提示】工具调用轮次已用尽。请立即基于当前已获得的数据给出结论，"
    "不要再请求调用任何工具。对于数据缺失的部分，必须明确写出"
    "「数据不足，无法判断」，禁止推测或编造。"
)


class AgentLoop:
    """单个 Agent 的 ReAct 执行循环。

    这个类是无状态的：所有运行期状态都在 run() 的局部变量里。
    这样同一个 AgentLoop 实例可以被多个 LangGraph 节点并发复用，
    不需要为每个节点各建一份（LLM 客户端和工具注册表都是重对象）。
    """

    def __init__(
        self,
        llm_client: Any,
        tool_registry: ToolRegistry | None = None,
        max_turns: int | None = None,
        tracer: Tracer | None = None,
        compactor: ContextCompact | None = None,
        max_total_tokens: int | None = None,
        agent_name: str = "agent",
    ):
        config = get_config()
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.max_turns = max_turns or config.max_turns
        self.tracer = tracer or NullTracer()
        self.compactor = compactor or ContextCompact(tracer=self.tracer)
        self.max_total_tokens = max_total_tokens or config.max_total_tokens
        self.agent_name = agent_name

    def run(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
        allowed_tools: Sequence[str] | None = None,
        on_tool_results: Callable[[list[ToolResult]], None] | None = None,
    ) -> AgentResult:
        """执行 ReAct 循环。

        参数
            system_prompt: 系统提示词，压缩时不会被动到。
            messages: 初始对话消息（通常只有一条 user）。函数不修改入参。
            context: 附加上下文，会被拼进 system prompt。典型用途是把
                ResearchPlan 或 as_of_date 这类结构化约束注入模型。
            allowed_tools: 本次允许调用的工具子集。按 Planner 的计划裁剪工具集
                能显著提升选择准确率——给模型 20 个工具让它挑 3 个，
                不如直接只给它那 3 个。
            on_tool_results: 每批工具执行后的回调。Retriever 用它把工具结果
                实时写进 evidence_pool，不必等循环结束。
        """
        started = time.perf_counter()
        token_counter = TokenCounter()
        collected_tool_results: list[ToolResult] = []

        effective_system_prompt = self._compose_system_prompt(system_prompt, context)
        working_messages: list[dict[str, Any]] = [
            {"role": "system", "content": effective_system_prompt},
            *messages,
        ]

        tool_definitions = self._resolve_tool_definitions(allowed_tools)
        last_text = ""
        turn = 0

        try:
            while turn < self.max_turns:
                turn += 1
                is_final_turn = turn == self.max_turns

                working_messages = self._maybe_compact(working_messages)

                # 最后一轮撤掉工具定义并给出收尾指令：既省 token，
                # 也从根本上杜绝"最后一轮又发起工具调用但无人执行"的空转。
                turn_tools = None if is_final_turn else tool_definitions
                if is_final_turn and tool_definitions:
                    working_messages.append({"role": "user", "content": FINAL_TURN_INSTRUCTION})

                response = self._call_llm(working_messages, effective_system_prompt, turn_tools, turn)

                if response.is_error:
                    return self._build_result(
                        status=AGENT_ERROR,
                        final_text=last_text,
                        messages=working_messages,
                        tool_results=collected_tool_results,
                        turns=turn,
                        token_counter=token_counter,
                        started=started,
                        error=response.error,
                    )

                token_counter.add(response.usage.prompt_tokens, response.usage.completion_tokens)
                working_messages.append(response.to_assistant_message())
                if response.content:
                    last_text = response.content

                # ---- 终止条件 1：模型给出纯文本回答 ----
                if not response.has_tool_calls:
                    return self._build_result(
                        status=AGENT_COMPLETED,
                        final_text=response.content,
                        messages=working_messages,
                        tool_results=collected_tool_results,
                        turns=turn,
                        token_counter=token_counter,
                        started=started,
                    )

                # ---- 执行工具 ----
                if self.tool_registry is None:
                    # 没有注册表却收到工具调用，说明 tools 定义与运行时不一致。
                    # 这是配置错误，不该静默吞掉。
                    return self._build_result(
                        status=AGENT_ERROR,
                        final_text=last_text,
                        messages=working_messages,
                        tool_results=collected_tool_results,
                        turns=turn,
                        token_counter=token_counter,
                        started=started,
                        error="模型请求调用工具，但该 AgentLoop 未配置 ToolRegistry",
                    )

                tool_results = self.tool_registry.execute(response.tool_calls)
                collected_tool_results.extend(tool_results)
                working_messages.extend(result.to_message() for result in tool_results)

                if on_tool_results is not None:
                    try:
                        on_tool_results(tool_results)
                    except Exception:  # noqa: BLE001 — 回调异常不能拖垮主循环
                        logger.exception("on_tool_results 回调抛出异常，已忽略")

                # ---- 终止条件 2：token 超限 ----
                if token_counter.total >= self.max_total_tokens:
                    logger.warning(
                        "[%s] token 超限：%d >= %d，提前结束",
                        self.agent_name,
                        token_counter.total,
                        self.max_total_tokens,
                    )
                    return self._build_result(
                        status=AGENT_TOKEN_LIMIT,
                        final_text=last_text,
                        messages=working_messages,
                        tool_results=collected_tool_results,
                        turns=turn,
                        token_counter=token_counter,
                        started=started,
                        error=f"累计 token {token_counter.total} 超过上限 {self.max_total_tokens}",
                    )

            # ---- 终止条件 3：轮次耗尽 ----
            logger.warning("[%s] max_turns=%d 耗尽", self.agent_name, self.max_turns)
            return self._build_result(
                status=AGENT_MAX_TURNS,
                final_text=last_text,
                messages=working_messages,
                tool_results=collected_tool_results,
                turns=turn,
                token_counter=token_counter,
                started=started,
                error=f"达到最大轮次 {self.max_turns}，可能存在工具调用循环",
            )

        except Exception as exc:  # noqa: BLE001 — 兜底，保证永远返回 AgentResult
            logger.exception("[%s] ReAct 循环未预期异常", self.agent_name)
            self.tracer.log(
                "error",
                agent_name=self.agent_name,
                output_summary=f"{type(exc).__name__}: {exc}",
                success=False,
            )
            return self._build_result(
                status=AGENT_ERROR,
                final_text=last_text,
                messages=working_messages,
                tool_results=collected_tool_results,
                turns=turn,
                token_counter=token_counter,
                started=started,
                error=f"{type(exc).__name__}: {exc}",
            )

    # ---------------- 内部步骤 ----------------

    def _compose_system_prompt(self, system_prompt: str, context: dict[str, Any] | None) -> str:
        if not context:
            return system_prompt
        lines = [f"- {key}: {value}" for key, value in context.items() if value not in (None, "")]
        if not lines:
            return system_prompt
        return f"{system_prompt}\n\n## 本次任务上下文\n" + "\n".join(lines)

    def _resolve_tool_definitions(
        self, allowed_tools: Sequence[str] | None
    ) -> list[dict[str, Any]] | None:
        if self.tool_registry is None:
            return None
        definitions = self.tool_registry.get_definitions(only=allowed_tools)
        return definitions or None

    def _maybe_compact(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.compactor.should_compact(messages):
            return messages
        return self.compactor.compact(messages, llm_client=self.llm_client)

    def _call_llm(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]] | None,
        turn: int,
    ) -> LLMResponse:
        # messages[0] 已是 system，client 内部会去重，这里传 body 部分即可
        body = [m for m in messages if m.get("role") != "system"]
        with self.tracer.span(
            "llm_call",
            self.agent_name,
            input_summary=f"turn={turn} messages={len(body)} tokens≈{count_messages_tokens(messages)}",
            turn=turn,
            tool_count=len(tools or []),
        ) as span:
            response: LLMResponse = self.llm_client.chat(
                messages=body, system_prompt=system_prompt, tools=tools
            )
            span.set_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)
            span.add_metadata(
                finish_reason=response.finish_reason,
                model=response.model,
                requested_tools=[call.name for call in response.tool_calls],
            )
            if response.is_error:
                span.mark_failed(response.error or "unknown error")
            else:
                span.set_output(
                    truncate(response.content)
                    or f"tool_calls: {[c.summary(80) for c in response.tool_calls]}"
                )
            return response

    def _build_result(
        self,
        status: str,
        final_text: str,
        messages: list[dict[str, Any]],
        tool_results: list[ToolResult],
        turns: int,
        token_counter: TokenCounter,
        started: float,
        error: str | None = None,
    ) -> AgentResult:
        result = AgentResult(
            status=status,  # type: ignore[arg-type]
            final_text=final_text,
            messages=messages,
            tool_results=tool_results,
            turns=turns,
            token_usage=token_counter,
            duration_ms=int((time.perf_counter() - started) * 1000),
            error=error,
            agent_name=self.agent_name,
        )
        logger.info(
            "[%s] 循环结束 status=%s turns=%d tools=%d(失败 %d) tokens=%d",
            self.agent_name,
            status,
            turns,
            len(tool_results),
            result.failed_tool_calls,
            token_counter.total,
        )
        return result
