"""LLM 层的数据契约。

解决什么问题
    DeepSeek 的原始响应是一层层嵌套的 dict（choices[0].message.tool_calls[0]
    .function.arguments 是个 JSON 字符串）。如果让 AgentLoop 直接去啃这个结构，
    上层代码会遍布 `.get("choices", [{}])[0]` 这种防御性索引，而且换模型供应商
    就要全改一遍。本模块把响应收敛成扁平的 LLMResponse。

核心设计决策
    1. tool_calls 在这一层就解析成 harness.types.ToolCall（arguments 已从
       JSON 字符串反序列化成 dict）。解析失败不抛异常，而是把 arguments 置空、
       raw_arguments 保留原文——LLM 生成非法 JSON 是常态而非异常，
       应该走"参数校验失败 → 提示重试"的正常路径，不该炸掉整个循环。
    2. 保留 reasoning_content 字段。deepseek-reasoner 会单独返回思维链，
       这部分不能回填进下一轮 messages（官方明确要求），但对 trace 复盘极有价值，
       所以存下来只用于观测。
    3. TokenUsage 带 cache_hit_tokens。DeepSeek 有上下文硬盘缓存，
       命中部分价格差 10 倍，做成本分析时必须能区分。

为什么不用其他方案
    - 不直接用 openai SDK 的 ChatCompletion 对象：那个类型绑定 SDK 版本，
      且携带大量本项目用不到的字段；自定义模型能让 trace 序列化更可控。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from harness.types import ToolCall

logger = logging.getLogger(__name__)

FinishReason = Literal[
    "stop",  # 正常结束
    "tool_calls",  # 请求调用工具
    "length",  # 达到 max_tokens 截断
    "content_filter",  # 内容审核拦截
    "error",  # 客户端侧构造的错误响应
    "unknown",
]


class TokenUsage(BaseModel):
    """单次 LLM 调用的 token 消耗。"""

    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # DeepSeek 上下文缓存命中/未命中的 prompt token 数（其他供应商没有则为 0）
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    # reasoner 模型的思维链 token（计入 completion 计费）
    reasoning_tokens: int = 0

    @classmethod
    def from_api(cls, usage: dict[str, Any] | None) -> "TokenUsage":
        if not usage:
            return cls()
        details = usage.get("completion_tokens_details") or {}
        return cls(
            prompt_tokens=usage.get("prompt_tokens", 0) or 0,
            completion_tokens=usage.get("completion_tokens", 0) or 0,
            total_tokens=usage.get("total_tokens", 0) or 0,
            cache_hit_tokens=usage.get("prompt_cache_hit_tokens", 0) or 0,
            cache_miss_tokens=usage.get("prompt_cache_miss_tokens", 0) or 0,
            reasoning_tokens=(details.get("reasoning_tokens", 0) or 0),
        )

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cache_hit_tokens=self.cache_hit_tokens + other.cache_hit_tokens,
            cache_miss_tokens=self.cache_miss_tokens + other.cache_miss_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def as_trace_dict(self) -> dict[str, int]:
        return {
            "input": self.prompt_tokens,
            "output": self.completion_tokens,
            "total": self.total_tokens or (self.prompt_tokens + self.completion_tokens),
        }


class LLMResponse(BaseModel):
    """一次 LLM 调用的归一化响应。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason = "unknown"
    model: str = ""
    usage: TokenUsage = Field(default_factory=TokenUsage)
    # reasoner 的思维链，仅用于 trace，禁止回填进 messages
    reasoning_content: str | None = None
    latency_ms: int = 0
    # 客户端侧错误（重试耗尽等），非空时 finish_reason == "error"
    error: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_error(self) -> bool:
        return self.finish_reason == "error" or self.error is not None

    @classmethod
    def from_api(cls, payload: dict[str, Any], latency_ms: int = 0) -> "LLMResponse":
        """从 OpenAI 兼容的响应体构造。"""
        choices = payload.get("choices") or []
        if not choices:
            return cls(
                finish_reason="error",
                error="响应中没有 choices 字段",
                model=payload.get("model", ""),
                usage=TokenUsage.from_api(payload.get("usage")),
                latency_ms=latency_ms,
            )

        message = choices[0].get("message") or {}
        raw_finish = choices[0].get("finish_reason") or "unknown"
        finish_reason: FinishReason = (
            raw_finish if raw_finish in {"stop", "tool_calls", "length", "content_filter"} else "unknown"  # type: ignore[assignment]
        )

        tool_calls = [
            parsed
            for parsed in (parse_tool_call(item) for item in message.get("tool_calls") or [])
            if parsed is not None
        ]
        # 有的兼容实现在返回 tool_calls 时 finish_reason 仍写 stop，这里纠正，
        # 否则 ReAct 循环会误判为"模型已给出最终答案"而提前退出。
        if tool_calls and finish_reason != "tool_calls":
            finish_reason = "tool_calls"

        return cls(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            model=payload.get("model", ""),
            usage=TokenUsage.from_api(payload.get("usage")),
            reasoning_content=message.get("reasoning_content"),
            latency_ms=latency_ms,
        )

    @classmethod
    def from_error(cls, error: str, model: str = "", latency_ms: int = 0) -> "LLMResponse":
        return cls(finish_reason="error", error=error, model=model, latency_ms=latency_ms)

    def to_assistant_message(self) -> dict[str, Any]:
        """转成可回填进 messages 的 assistant 消息。

        刻意不包含 reasoning_content：DeepSeek 文档明确要求
        不要把上一轮的思维链传回去，否则会报 400。
        """
        message: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.raw_arguments
                        or json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in self.tool_calls
            ]
        return message


def parse_tool_call(item: dict[str, Any]) -> ToolCall | None:
    """把 API 返回的 tool_call 结构解析成 ToolCall。

    arguments 是 JSON 字符串，模型偶尔会输出带尾随逗号、单引号、
    或者被 ```json 包裹的内容。这里做两次尝试：直接解析 → 清洗后解析。
    仍失败则返回空参数 + 保留原文，让 ToolRegistry 的参数校验去报错，
    错误信息会回喂给模型重试。
    """
    function = item.get("function") or {}
    name = function.get("name")
    if not name:
        logger.warning("tool_call 缺少 function.name: %s", item)
        return None

    raw_arguments = function.get("arguments") or "{}"
    arguments: dict[str, Any] = {}
    if isinstance(raw_arguments, dict):
        arguments, raw_arguments = raw_arguments, json.dumps(raw_arguments, ensure_ascii=False)
    else:
        try:
            parsed = json.loads(raw_arguments)
            arguments = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            cleaned = _clean_json_text(raw_arguments)
            try:
                parsed = json.loads(cleaned)
                arguments = parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                logger.warning("工具参数不是合法 JSON，保留原文: %s", raw_arguments[:200])

    fields: dict[str, Any] = {
        "name": name,
        "arguments": arguments,
        "raw_arguments": raw_arguments if isinstance(raw_arguments, str) else None,
    }
    # 有 id 才覆盖默认值：部分兼容实现不返回 id，此时用 ToolCall 自带的
    # default_factory 生成，保证 tool 消息始终能配对上。
    if item.get("id"):
        fields["id"] = item["id"]
    return ToolCall(**fields)


def _clean_json_text(text: str) -> str:
    """清洗 LLM 输出里常见的 JSON 污染：markdown 围栏、尾随逗号。"""
    import re

    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned.strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    """从自由文本里抠出第一个完整的 JSON 对象。

    用于 chat_structured 的降级路径：模型不支持 response_format 时，
    它会把 JSON 包在解释文字里返回。用括号配对扫描而不是正则，
    因为 JSON 是递归结构，正则无法正确处理嵌套。
    """
    if not text:
        return None

    candidate = _clean_json_text(text)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(candidate)):
            char = candidate[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    fragment = candidate[start : index + 1]
                    try:
                        parsed = json.loads(fragment)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        try:
                            parsed = json.loads(_clean_json_text(fragment))
                            if isinstance(parsed, dict):
                                return parsed
                        except json.JSONDecodeError:
                            break
                    break
        start = candidate.find("{", start + 1)
    return None
