"""工具注册表与调度器。

解决什么问题
    LLM 调工具这件事有三个必须处理的现实：
    (1) LLM 会调用不存在的工具名，会漏传必填参数，会把数字传成字符串；
    (2) 外部数据源（AKShare）会挂起不返回，没有超时保护就会拖死整次运行；
    (3) 工具抛的异常五花八门，直接把异常栈丢给 LLM 它会读不懂并陷入重试循环。
    本模块把"工具定义 → LLM function schema → 参数校验 → 带超时执行 →
    失败归类"这条链路收敛到一处。

核心设计决策
    1. 用注册表 + dispatch map，而不是 if/elif 分发。新增工具只需 register()，
       不用改任何分发代码；MCP Server 也能直接复用同一个注册表。
    2. 超时用 ThreadPoolExecutor + future.result(timeout)，不用 signal.alarm。
       signal 只能在主线程用，而 FastAPI 的同步端点跑在工作线程里，
       signal 方案在服务化后会直接失效。代价是超时后工作线程仍在后台跑
       （Python 无法安全强杀线程），所以线程池设为 daemon 且有容量上限，
       避免僵尸线程堆积。
    3. 失败一律转成 ToolResult(success=False) 返回，绝不向上抛。ReAct 循环
       需要把"这个工具失败了、失败原因是什么"作为一条观察喂回 LLM，让它
       换一个工具或换参数重试。异常直接冒泡等于剥夺了 Agent 的自愈能力。
    4. 参数校验只做 JSON Schema 的一个子集（required / type / enum）。
       引入 jsonschema 库会为了覆盖 5% 的边缘场景增加一个依赖，
       而 LLM 实际犯的错 95% 是"漏了必填参数"和"数字传成字符串"这两类。

为什么不用其他方案
    - 不用 LangChain 的 Tool/StructuredTool：它把校验、执行、回调绑死在一起，
      想单独换掉超时策略要读一大段继承链；自建这 200 行反而更好讲清楚。
    - 不用 asyncio 超时：工具底层是 AKShare（同步 requests + pandas），
      包成 async 只是把阻塞挪进事件循环，会卡住整个 FastAPI 进程。
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable, Iterable

from config import get_config
from harness.tracing import NullTracer, Tracer, truncate
from harness.types import (
    TOOL_ERROR_EXECUTION,
    TOOL_ERROR_NOT_FOUND,
    TOOL_ERROR_PARAM,
    TOOL_ERROR_TIMEOUT,
    ToolCall,
    ToolProtocol,
    ToolResult,
)

logger = logging.getLogger(__name__)

# 工具返回文本喂给 LLM 前的长度上限。财务报表原文可能上万字符，
# 超过这个长度会挤占上下文并显著推高成本，超出部分截断并显式告知 LLM。
MAX_TOOL_CONTENT_CHARS = 6000


class ToolExecutionError(Exception):
    """工具内部主动抛出的业务异常。与"未预期异常"区分，便于错误分类。"""

    def __init__(self, message: str, error_type: str = TOOL_ERROR_EXECUTION):
        super().__init__(message)
        self.error_type = error_type


class ToolRegistry:
    """工具注册表。线程安全的读多写少结构（注册在启动期完成，运行期只读）。"""

    def __init__(
        self,
        timeout_seconds: int | None = None,
        tracer: Tracer | None = None,
        max_workers: int = 8,
    ):
        config = get_config()
        self.timeout_seconds = timeout_seconds or config.tool_timeout_seconds
        self.tracer = tracer or NullTracer()
        self._tools: dict[str, ToolProtocol] = {}
        # thread_name_prefix 让超时后残留的线程在 py-spy / faulthandler 里可辨认
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="tool-worker"
        )

    # ---------------- 注册 ----------------

    def register(self, tool: ToolProtocol) -> None:
        """注册一个工具。重名会覆盖并告警——静默覆盖是排查半天的那种 bug。"""
        self._validate_tool_shape(tool)
        if tool.name in self._tools:
            logger.warning("工具重名，后注册的将覆盖先注册的: %s", tool.name)
        self._tools[tool.name] = tool
        logger.debug("已注册工具: %s", tool.name)

    def register_all(self, tools: Iterable[ToolProtocol]) -> None:
        for tool in tools:
            self.register(tool)

    @staticmethod
    def _validate_tool_shape(tool: Any) -> None:
        missing = [
            attr for attr in ("name", "description", "parameters", "run") if not hasattr(tool, attr)
        ]
        if missing:
            raise TypeError(
                f"{type(tool).__name__} 不满足 ToolProtocol，缺少: {', '.join(missing)}"
            )
        if not callable(tool.run):
            raise TypeError(f"{type(tool).__name__}.run 必须可调用")
        if not isinstance(tool.parameters, dict):
            raise TypeError(f"{tool.name}.parameters 必须是 JSON Schema dict")

    # ---------------- 查询 ----------------

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> ToolProtocol | None:
        return self._tools.get(name)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools)

    def get_definitions(self, only: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """生成 OpenAI function calling 格式的工具定义。

        only 参数用于按 ResearchPlan.required_data 裁剪工具集：给 LLM 传 20 个
        工具定义会显著降低它的选择准确率，只传计划里需要的 3~4 个效果更好。
        """
        selected = self._tools if only is None else {n: self._tools[n] for n in only if n in self._tools}
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                    or {"type": "object", "properties": {}, "required": []},
                },
            }
            for tool in selected.values()
        ]

    # ---------------- 执行 ----------------

    def execute(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """批量执行工具调用。

        同一轮里的多个工具调用之间没有依赖（LLM 并行发起的），因此并发执行；
        总耗时从"各工具耗时之和"降到"最慢的那个"。这在 Retriever 一次性
        拉利润表 + 资产负债表 + 现金流量表的场景下能省一半以上时间。
        """
        if not tool_calls:
            return []
        if len(tool_calls) == 1:
            return [self.execute_one(tool_calls[0])]

        futures: list[tuple[ToolCall, Future[ToolResult]]] = [
            (call, self._executor.submit(self.execute_one, call)) for call in tool_calls
        ]
        results: list[ToolResult] = []
        for call, future in futures:
            try:
                # 单个工具内部已有超时，这里加一点余量防止调度延迟造成误判
                results.append(future.result(timeout=self.timeout_seconds + 5))
            except FutureTimeoutError:
                results.append(
                    ToolResult.failed(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        error=f"工具调度超时（>{self.timeout_seconds + 5}s）",
                        error_type=TOOL_ERROR_TIMEOUT,
                    )
                )
        return results

    def execute_one(self, call: ToolCall) -> ToolResult:
        """执行单个工具调用，任何异常都转成失败的 ToolResult。"""
        started = time.perf_counter()

        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(sorted(self._tools)) or "（无）"
            return self._finish(
                call,
                ToolResult.failed(
                    call.id,
                    call.name,
                    f"工具不存在。可用工具: {available}",
                    TOOL_ERROR_NOT_FOUND,
                ),
                started,
            )

        param_error = self._validate_arguments(tool, call.arguments)
        if param_error:
            return self._finish(
                call,
                ToolResult.failed(call.id, call.name, param_error, TOOL_ERROR_PARAM),
                started,
            )

        coerced_arguments = self._coerce_arguments(tool, call.arguments)

        future = self._executor.submit(self._invoke, tool.run, coerced_arguments)
        try:
            raw_result = future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError:
            # 无法强杀线程，只能放弃等待。future.cancel() 对已开始执行的任务无效，
            # 这是 Python 线程模型的固有限制，写清楚以免后来者以为这里漏了清理。
            future.cancel()
            return self._finish(
                call,
                ToolResult.failed(
                    call.id,
                    call.name,
                    f"工具执行超时（>{self.timeout_seconds}s），后台线程可能仍在运行",
                    TOOL_ERROR_TIMEOUT,
                ),
                started,
            )
        except ToolExecutionError as exc:
            return self._finish(
                call,
                ToolResult.failed(call.id, call.name, str(exc), exc.error_type),  # type: ignore[arg-type]
                started,
            )
        except TypeError as exc:
            # run(**kwargs) 签名不匹配属于参数问题，不是执行问题，分开归类
            return self._finish(
                call,
                ToolResult.failed(call.id, call.name, f"参数不匹配: {exc}", TOOL_ERROR_PARAM),
                started,
            )
        except Exception as exc:
            logger.exception("工具 %s 执行异常", call.name)
            return self._finish(
                call,
                ToolResult.failed(
                    call.id, call.name, f"{type(exc).__name__}: {exc}", TOOL_ERROR_EXECUTION
                ),
                started,
            )

        duration_ms = int((time.perf_counter() - started) * 1000)
        data, content, from_cache, cache_layer = self._normalize_result(raw_result)
        return self._finish(
            call,
            ToolResult.ok(
                tool_call_id=call.id,
                tool_name=call.name,
                data=data,
                content=content,
                duration_ms=duration_ms,
                from_cache=from_cache,
                cache_layer=cache_layer,
            ),
            started,
        )

    @staticmethod
    def _invoke(run: Callable[..., Any], arguments: dict[str, Any]) -> Any:
        return run(**arguments)

    def _finish(self, call: ToolCall, result: ToolResult, started: float) -> ToolResult:
        if not result.duration_ms:
            result.duration_ms = int((time.perf_counter() - started) * 1000)
        self.tracer.log(
            "tool_call",
            agent_name=call.name,
            input_summary=call.arguments,
            output_summary=result.content if result.success else result.error,
            duration_ms=result.duration_ms,
            success=result.success,
            tool_name=call.name,
            error_type=result.error_type,
            from_cache=result.from_cache,
        )
        return result

    # ---------------- 结果与参数处理 ----------------

    @staticmethod
    def _normalize_result(raw: Any) -> tuple[Any, str, bool, str | None]:
        """把工具返回值归一成 (结构化数据, 给 LLM 的文本, 是否命中缓存, 缓存层)。

        约定：工具可以返回 str（纯文本）、dict（结构化）、
        或 dict 里带 `_cache` 元信息。文本形态由这里统一生成，
        保证所有工具喂给 LLM 的格式一致。
        """
        import json

        from_cache = False
        cache_layer: str | None = None

        if isinstance(raw, str):
            return raw, ToolRegistry._clip(raw), from_cache, cache_layer

        if isinstance(raw, dict):
            meta = raw.get("_cache")
            if isinstance(meta, dict):
                from_cache = bool(meta.get("hit"))
                cache_layer = meta.get("layer")
                raw = {k: v for k, v in raw.items() if k != "_cache"}
            text = json.dumps(raw, ensure_ascii=False, indent=2, default=str)
            return raw, ToolRegistry._clip(text), from_cache, cache_layer

        try:
            text = json.dumps(raw, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(raw)
        return raw, ToolRegistry._clip(text), from_cache, cache_layer

    @staticmethod
    def _clip(text: str) -> str:
        if len(text) <= MAX_TOOL_CONTENT_CHARS:
            return text
        return (
            text[:MAX_TOOL_CONTENT_CHARS]
            + f"\n...[输出过长已截断，原始长度 {len(text)} 字符。"
            "如需完整数据请缩小查询范围，例如减少 periods]"
        )

    @staticmethod
    def _validate_arguments(tool: ToolProtocol, arguments: dict[str, Any]) -> str | None:
        """JSON Schema 子集校验。返回错误描述，None 表示通过。

        错误信息写成"给 LLM 看的指令"而不是"给人看的报错"，
        因为它会被原样喂回模型，说清楚缺什么、该传什么能显著提高重试成功率。
        """
        schema = tool.parameters or {}
        properties: dict[str, Any] = schema.get("properties", {})
        required: list[str] = schema.get("required", [])

        missing = [name for name in required if name not in arguments or arguments[name] is None]
        if missing:
            return (
                f"缺少必填参数: {', '.join(missing)}。"
                f"该工具的参数定义为: {list(properties)}"
            )

        unknown = [name for name in arguments if name not in properties] if properties else []
        if unknown:
            return (
                f"传入了未定义的参数: {', '.join(unknown)}。"
                f"允许的参数为: {list(properties)}"
            )

        for name, value in arguments.items():
            spec = properties.get(name, {})
            enum_values = spec.get("enum")
            if enum_values and value not in enum_values:
                return f"参数 {name} 的值 {value!r} 不在允许范围内: {enum_values}"
        return None

    @staticmethod
    def _coerce_arguments(tool: ToolProtocol, arguments: dict[str, Any]) -> dict[str, Any]:
        """按 schema 声明的类型做温和转换。

        LLM 经常把 periods=3 写成 "3"。与其让工具内部各自 int() 一遍
        （总有一个会忘），不如在调度层统一转换。转换失败保留原值，
        让工具自己抛更具体的错误。
        """
        properties: dict[str, Any] = (tool.parameters or {}).get("properties", {})
        coerced: dict[str, Any] = {}
        for name, value in arguments.items():
            declared_type = properties.get(name, {}).get("type")
            try:
                if declared_type == "integer" and not isinstance(value, bool):
                    coerced[name] = int(value)
                elif declared_type == "number" and not isinstance(value, bool):
                    coerced[name] = float(value)
                elif declared_type == "string" and not isinstance(value, str):
                    coerced[name] = str(value)
                elif declared_type == "boolean" and isinstance(value, str):
                    coerced[name] = value.strip().lower() in {"true", "1", "yes", "y"}
                else:
                    coerced[name] = value
            except (TypeError, ValueError):
                coerced[name] = value
        return coerced

    # ---------------- 资源 ----------------

    def shutdown(self, wait: bool = False) -> None:
        """关闭线程池。wait=False 让进程退出时不被超时残留线程卡住。"""
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def describe(self) -> str:
        """给 system prompt 用的工具清单文本。"""
        if not self._tools:
            return "（当前没有可用工具）"
        return "\n".join(
            f"- {tool.name}: {truncate(tool.description, 120)}" for tool in self._tools.values()
        )
