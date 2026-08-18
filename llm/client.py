"""DeepSeek LLM 客户端。

解决什么问题
    LLM API 在生产里是最不可靠的一环：会 429 限流、会 502、会返回不符合约定的
    JSON、reasoner 模型还不支持 function calling。如果每个 Agent 各自 requests.post
    一遍，这些坑要踩四次。本模块把重试、超时、结构化输出降级、模型能力差异
    全部收敛在一个类里。

核心设计决策
    1. 用 httpx 直连而不是 openai SDK。理由不是"SDK 不好用"，而是：
       DeepSeek 有几个非标字段（reasoning_content、prompt_cache_hit_tokens）
       在 SDK 的类型定义里不存在，走 SDK 要么拿不到要么得访问私有属性；
       直连 REST 反而更直白。openai 仍留在依赖里，方便需要时切换。
    2. 重试只对**可重试错误**生效：429、5xx、网络超时。400（参数错误）
       和 401（鉴权失败）重试三次只是把同一个错误犯三遍，还浪费 14 秒。
       这个区分是"处理非确定性"的具体体现。
    3. 指数退避 2s / 4s / 8s，并叠加随机抖动。多个 Agent 节点并发被限流时，
       固定退避会让它们在同一时刻一起重试，形成惊群，抖动能把它们打散。
    4. reasoner 模型不支持 function calling / response_format / temperature，
       客户端在发请求前按模型能力过滤参数，而不是让调用方记住这些差异。
       调用方传了不支持的参数只会被静默丢弃并记一条 debug 日志。
    5. chat_structured 三级降级：原生 JSON mode → 提示词约束 + JSON 抠取 →
       返回原始文本。任何一级失败都不抛异常，因为上层（Planner/Verifier）
       都有基于文本的兜底路径，抛异常会让整次研究失败。

为什么不用其他方案
    - 不用 tenacity 装饰器做重试：需要根据 HTTP 状态码分类决定是否重试，
      还要在重试时改写 trace，装饰器里拿不到这些上下文，控制流反而更绕。
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Sequence, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from config import get_config
from llm.schemas import LLMResponse, extract_json_object

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# 这些 HTTP 状态码重试才有意义
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}

# 不支持 function calling / json 模式的模型前缀。
# DeepSeek 的 reasoner 系列走的是纯推理路径，传 tools 会直接 400。
REASONER_MODEL_MARKERS = ("reasoner", "-r1", "r1-")

STRUCTURED_OUTPUT_INSTRUCTION = """
你必须只输出一个合法的 JSON 对象，不要输出任何解释、前言或 markdown 代码围栏。
JSON 必须严格符合下面的 Schema：

{schema}

再次强调：只输出 JSON 对象本身。
""".strip()


class LLMClient:
    """单个模型的调用客户端。一个模型一个实例（quick / deep 各一份）。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: int | None = None,
        max_retries: int | None = None,
        backoff_base_seconds: float | None = None,
        temperature: float | None = None,
        http_client: httpx.Client | None = None,
    ):
        config = get_config()
        self.api_key = api_key or config.deepseek_api_key
        self.base_url = (base_url or config.deepseek_base_url).rstrip("/")
        self.model = model or config.quick_model
        self.timeout_seconds = timeout_seconds or config.llm_timeout_seconds
        self.max_retries = max_retries if max_retries is not None else config.llm_max_retries
        self.backoff_base_seconds = (
            backoff_base_seconds
            if backoff_base_seconds is not None
            else config.llm_backoff_base_seconds
        )
        self.temperature = temperature if temperature is not None else config.llm_temperature
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=httpx.Timeout(self.timeout_seconds, connect=10.0),
            # 连接池复用：Agent 一次运行会发十几次请求，每次重建 TLS 连接
            # 大约多花 200ms，累计起来不容忽视
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    # ---------------- 模型能力 ----------------

    @property
    def is_reasoner(self) -> bool:
        lowered = self.model.lower()
        return any(marker in lowered for marker in REASONER_MODEL_MARKERS)

    @property
    def supports_tools(self) -> bool:
        return not self.is_reasoner

    @property
    def supports_json_mode(self) -> bool:
        return not self.is_reasoner

    # ---------------- 主接口 ----------------

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        system_prompt: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """普通 chat 调用，支持 function calling。

        失败时返回 finish_reason="error" 的 LLMResponse 而不是抛异常：
        AgentLoop 需要把失败也当作一次可观测的状态转移来处理。
        """
        payload = self._build_payload(
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        return self._post_with_retry(payload)

    def chat_structured(
        self,
        messages: Sequence[dict[str, Any]],
        schema: Type[SchemaT],
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> SchemaT | str:
        """结构化输出。成功返回 schema 实例，彻底失败时返回原始文本。

        返回联合类型而不是抛异常，是因为调用方（Planner/Verifier）对
        "拿到文本但解析不出结构"有意义的降级处理：Planner 可以退回默认计划，
        Verifier 可以把全部结论标成 ⚠️未验证。强行抛异常会剥夺这个选择。
        """
        json_schema = schema.model_json_schema()
        instruction = STRUCTURED_OUTPUT_INSTRUCTION.format(
            schema=json.dumps(json_schema, ensure_ascii=False, indent=2)
        )
        combined_system = f"{system_prompt}\n\n{instruction}" if system_prompt else instruction

        # 第一级：原生 JSON mode（deepseek-chat 支持）
        response_format = {"type": "json_object"} if self.supports_json_mode else None
        response = self.chat(
            messages=messages,
            system_prompt=combined_system,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        if response.is_error:
            logger.warning("结构化输出调用失败: %s", response.error)
            return f"[LLM 调用失败] {response.error}"

        # 第二级：从自由文本里抠 JSON
        payload = extract_json_object(response.content)
        if payload is not None:
            try:
                return schema.model_validate(payload)
            except ValidationError as exc:
                logger.warning(
                    "JSON 结构不符合 %s: %s", schema.__name__, truncate_error(exc)
                )
                repaired = self._repair_with_schema(payload, schema)
                if repaired is not None:
                    return repaired

        # 第三级：返回原始文本，由调用方决定怎么降级
        logger.warning("无法从响应中解析出 %s，返回原始文本", schema.__name__)
        return response.content

    # ---------------- 请求构造与发送 ----------------

    def _build_payload(
        self,
        messages: Sequence[dict[str, Any]],
        system_prompt: str | None,
        tools: Sequence[dict[str, Any]] | None,
        tool_choice: str | None,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        body_messages: list[dict[str, Any]] = []
        if system_prompt:
            body_messages.append({"role": "system", "content": system_prompt})
        # 去掉调用方可能重复传入的 system，避免出现两条 system 消息
        body_messages.extend(m for m in messages if m.get("role") != "system")

        payload: dict[str, Any] = {"model": self.model, "messages": body_messages}

        if tools:
            if self.supports_tools:
                payload["tools"] = list(tools)
                if tool_choice:
                    payload["tool_choice"] = tool_choice
            else:
                logger.debug("模型 %s 不支持 function calling，已忽略 tools 参数", self.model)

        if response_format:
            if self.supports_json_mode:
                payload["response_format"] = response_format
            else:
                logger.debug("模型 %s 不支持 response_format，已忽略", self.model)

        if not self.is_reasoner:
            # reasoner 传 temperature 会被服务端忽略甚至报错，干脆不传
            payload["temperature"] = self.temperature

        if max_tokens:
            payload["max_tokens"] = max_tokens

        return payload

    def _post_with_retry(self, payload: dict[str, Any]) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = self._headers()
        last_error = "未知错误"
        started = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(url, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                last_error = f"请求超时（{self.timeout_seconds}s）: {exc}"
                if not self._sleep_before_retry(attempt, last_error):
                    break
                continue
            except httpx.HTTPError as exc:
                last_error = f"网络错误: {type(exc).__name__}: {exc}"
                if not self._sleep_before_retry(attempt, last_error):
                    break
                continue

            if response.status_code == 200:
                latency_ms = int((time.perf_counter() - started) * 1000)
                try:
                    return LLMResponse.from_api(response.json(), latency_ms=latency_ms)
                except json.JSONDecodeError as exc:
                    last_error = f"响应不是合法 JSON: {exc}"
                    break

            last_error = self._describe_http_error(response)
            if response.status_code not in RETRYABLE_STATUS_CODES:
                logger.error("不可重试的 API 错误: %s", last_error)
                break
            if not self._sleep_before_retry(attempt, last_error, response=response):
                break

        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.error("LLM 调用最终失败（model=%s）: %s", self.model, last_error)
        return LLMResponse.from_error(last_error, model=self.model, latency_ms=latency_ms)

    def _headers(self) -> dict[str, str]:
        api_key = self.api_key or get_config().require_deepseek_api_key()
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _sleep_before_retry(
        self, attempt: int, reason: str, response: httpx.Response | None = None
    ) -> bool:
        """决定是否继续重试。返回 False 表示放弃。"""
        if attempt >= self.max_retries:
            return False

        # 服务端明确给了 Retry-After 就听它的，这是限流场景下最有效的退避依据
        retry_after = None
        if response is not None:
            header_value = response.headers.get("Retry-After")
            if header_value:
                try:
                    retry_after = float(header_value)
                except ValueError:
                    retry_after = None

        delay = retry_after if retry_after is not None else self.backoff_base_seconds * (2**attempt)
        delay += random.uniform(0, 0.5)  # 抖动，避免多节点惊群
        logger.warning(
            "LLM 调用失败（第 %d/%d 次）: %s；%.1fs 后重试",
            attempt + 1,
            self.max_retries,
            reason,
            delay,
        )
        time.sleep(delay)
        return True

    @staticmethod
    def _describe_http_error(response: httpx.Response) -> str:
        try:
            body = response.json()
            detail = body.get("error", {}).get("message") or json.dumps(body, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            detail = response.text[:300]
        return f"HTTP {response.status_code}: {detail}"

    def _repair_with_schema(
        self, payload: dict[str, Any], schema: Type[SchemaT]
    ) -> SchemaT | None:
        """字段校验失败时的轻量修复：只保留 schema 里声明的字段再试一次。

        LLM 最常见的结构错误是"多加了一个自创字段"（extra 字段）。
        丢掉多余字段能救回大部分情况，比再发一次请求便宜得多。
        """
        allowed = set(schema.model_fields)
        filtered = {key: value for key, value in payload.items() if key in allowed}
        if filtered == payload:
            return None
        try:
            return schema.model_validate(filtered)
        except ValidationError:
            return None

    # ---------------- 资源 ----------------

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"LLMClient(model={self.model!r}, base_url={self.base_url!r})"


def truncate_error(exc: Exception, limit: int = 300) -> str:
    text = str(exc).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."
