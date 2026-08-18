"""MCP（Model Context Protocol）服务基类。

解决什么问题
    Agent 直接 `from tools.akshare_tools import GetIncomeHistoryTool` 会产生
    硬编码依赖：新增一个数据源要改 Retriever 的代码，工具定义散落在
    Agent 和工具模块两处，远程工具（跑在别的进程/机器上）根本接不进来。
    MCP 把"工具有哪些、怎么调"标准化成协议，Agent 只面向协议编程。

核心设计决策
    1. 遵循 MCP 的 JSON-RPC 2.0 语义（tools/list、tools/call、错误码），
       但**用进程内方法调用而非 HTTP/stdio 传输**（md 明确要求保持简单）。
       这个取舍的关键是：协议层（消息格式、方法名、错误码）与传输层
       （进程内 / stdio / HTTP）本来就是分离的。保留协议层意味着
       将来把 handle_request 挂到 stdio 或 HTTP 上就能变成真正的远程 Server，
       不用改任何工具代码。
    2. 同时提供两套接口：
       - handle_request(dict) -> dict：完整的 JSON-RPC 形态，
         用于演示协议本身、以及将来接远程传输；
       - list_tools() / call_tool()：Python 原生调用，
         Agent 走这条路径，省掉一层无意义的序列化开销。
       两者共用同一份实现，不会出现"协议层能用、直调不能用"的分裂。
    3. call_tool 永不抛异常，一律返回 MCP 的 isError 结果结构。
       Agent 需要把工具失败当成一条可观察的信息喂回 LLM，而不是
       让异常穿透到 ReAct 循环之外。
    4. 内部持有 ToolRegistry 而不是自己维护 dict。超时控制、参数校验、
       错误归类这些逻辑已经在 Registry 里实现过一次，MCP Server
       只做协议适配，不重复实现。

为什么不用官方 mcp SDK
    官方 SDK 面向 stdio/SSE 传输，会引入 asyncio 事件循环和子进程管理。
    本项目的工具是同步的（AKShare + pandas），套一层 async 传输只会
    增加复杂度而没有任何实际收益。协议的价值在于接口标准化，
    不在于必须走进程间通信。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Sequence

from harness.tool_registry import ToolRegistry
from harness.tracing import NullTracer, Tracer
from harness.types import ToolCall, ToolProtocol, ToolResult

logger = logging.getLogger(__name__)

JSONRPC_VERSION = "2.0"

# JSON-RPC 2.0 标准错误码
ERROR_PARSE = -32700
ERROR_INVALID_REQUEST = -32600
ERROR_METHOD_NOT_FOUND = -32601
ERROR_INVALID_PARAMS = -32602
ERROR_INTERNAL = -32603
# MCP 扩展：工具执行失败（区别于协议层错误）
ERROR_TOOL_EXECUTION = -32000


class MCPServer:
    """MCP 工具服务基类。

    子类只需要：设置 name / description，并在 __init__ 里
    调用 register_tools() 注册工具。
    """

    name: str = "base"
    description: str = ""
    protocol_version: str = "2024-11-05"

    def __init__(self, registry: ToolRegistry | None = None, tracer: Tracer | None = None):
        self.tracer = tracer or NullTracer()
        self.registry = registry or ToolRegistry(tracer=self.tracer)

    # ---------------- 注册 ----------------

    def register_tools(self, tools: Iterable[ToolProtocol]) -> None:
        self.registry.register_all(tools)

    # ---------------- MCP 原生接口 ----------------

    def list_tools(self) -> list[dict[str, Any]]:
        """返回 MCP 格式的工具列表。

        MCP 的字段名是 inputSchema（驼峰），而 OpenAI function calling 用
        parameters。两套格式的转换放在这里，不让 Agent 关心差异。
        """
        return [
            {
                "name": definition["function"]["name"],
                "description": definition["function"]["description"],
                "inputSchema": definition["function"]["parameters"],
            }
            for definition in self.registry.get_definitions()
        ]

    def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """调用工具，返回 MCP 格式的结果。

        返回结构遵循 MCP 的 CallToolResult：
            {"content": [{"type": "text", "text": "..."}], "isError": bool}
        额外附带 structuredContent（结构化数据）和 _meta（耗时、缓存命中），
        这两个字段是 MCP 允许的扩展，下游的 Evidence 构造需要原始数字。
        """
        call = ToolCall(name=tool_name, arguments=arguments or {})
        result: ToolResult = self.registry.execute_one(call)
        return self._to_mcp_result(result)

    @staticmethod
    def _to_mcp_result(result: ToolResult) -> dict[str, Any]:
        return {
            "content": [{"type": "text", "text": result.content}],
            "isError": not result.success,
            "structuredContent": result.data if result.success else None,
            "_meta": {
                "tool_name": result.tool_name,
                "duration_ms": result.duration_ms,
                "from_cache": result.from_cache,
                "cache_layer": result.cache_layer,
                "error_type": result.error_type,
                "error": result.error,
            },
        }

    # ---------------- 供 Agent 使用的便捷接口 ----------------

    def get_openai_tool_definitions(
        self, only: Sequence[str] | None = None
    ) -> list[dict[str, Any]]:
        """直接给 LLM function calling 用的工具定义。"""
        return self.registry.get_definitions(only=only)

    def execute_tool_call(self, call: ToolCall) -> ToolResult:
        """执行一次 ToolCall 并返回 Harness 的 ToolResult。

        Agent 走这条路径而不是 call_tool()：ToolResult 已经带了
        结构化数据、耗时、缓存标记，转成 MCP 字典再转回来是无谓的损耗。
        协议的价值在于"工具怎么被发现和描述"，这一点通过
        get_openai_tool_definitions() 已经体现了。
        """
        return self.registry.execute_one(call)

    def has_tool(self, tool_name: str) -> bool:
        return self.registry.has(tool_name)

    @property
    def tool_names(self) -> list[str]:
        return self.registry.tool_names

    # ---------------- JSON-RPC 2.0 传输层 ----------------

    def handle_request(self, request: dict[str, Any] | str) -> dict[str, Any]:
        """处理一条 JSON-RPC 2.0 请求。

        支持的方法：initialize / tools/list / tools/call / ping。
        这个方法目前只被 /api/mcp 调试路由和单测使用，
        但它的存在保证了"换成 stdio 或 HTTP 传输"是一个纯增量改动。
        """
        if isinstance(request, str):
            try:
                request = json.loads(request)
            except json.JSONDecodeError as exc:
                return self._error(None, ERROR_PARSE, f"JSON 解析失败: {exc}")

        if not isinstance(request, dict) or request.get("jsonrpc") != JSONRPC_VERSION:
            return self._error(
                request.get("id") if isinstance(request, dict) else None,
                ERROR_INVALID_REQUEST,
                "请求必须是 jsonrpc=2.0 的 JSON 对象",
            )

        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}

        if method == "initialize":
            return self._success(
                request_id,
                {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": self.name, "version": "1.0.0"},
                },
            )

        if method == "ping":
            return self._success(request_id, {})

        if method == "tools/list":
            return self._success(request_id, {"tools": self.list_tools()})

        if method == "tools/call":
            tool_name = params.get("name")
            if not tool_name:
                return self._error(request_id, ERROR_INVALID_PARAMS, "缺少参数 name")
            if not self.has_tool(tool_name):
                return self._error(
                    request_id,
                    ERROR_METHOD_NOT_FOUND,
                    f"工具 {tool_name} 不存在。可用工具: {self.tool_names}",
                )
            result = self.call_tool(tool_name, params.get("arguments") or {})
            # MCP 约定：工具执行失败通过 result.isError 表达，而不是 JSON-RPC error。
            # 协议层错误（方法不存在）才用 error——这个区分让客户端能分清
            # "调用姿势不对"和"工具本身失败了"。
            return self._success(request_id, result)

        return self._error(request_id, ERROR_METHOD_NOT_FOUND, f"未知方法: {method}")

    @staticmethod
    def _success(request_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    # ---------------- 观测 ----------------

    def health(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tool_count": len(self.tool_names),
            "tools": self.tool_names,
            "protocol_version": self.protocol_version,
        }

    def __repr__(self) -> str:
        return f"<MCPServer name={self.name!r} tools={len(self.tool_names)}>"


class MCPToolBroker:
    """多个 MCP Server 的聚合器。

    Retriever 面对的是"一堆工具"，不该关心某个工具属于哪个 Server。
    Broker 负责合并工具定义、按工具名路由调用，并在工具重名时报错
    ——重名会导致路由到错误的 Server，静默处理是灾难性的。
    """

    def __init__(self, servers: Sequence[MCPServer]):
        self.servers = list(servers)
        self._routing: dict[str, MCPServer] = {}
        for server in self.servers:
            for tool_name in server.tool_names:
                if tool_name in self._routing:
                    raise ValueError(
                        f"工具名冲突: {tool_name} 同时存在于 "
                        f"{self._routing[tool_name].name} 和 {server.name}"
                    )
                self._routing[tool_name] = server

    @property
    def tool_names(self) -> list[str]:
        return list(self._routing)

    def get_tool_definitions(self, only: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """合并所有 Server 的工具定义（OpenAI function calling 格式）。"""
        definitions: list[dict[str, Any]] = []
        for server in self.servers:
            wanted = (
                [name for name in only if server.has_tool(name)] if only is not None else None
            )
            if only is not None and not wanted:
                continue
            definitions.extend(server.get_openai_tool_definitions(only=wanted))
        return definitions

    def execute(self, call: ToolCall) -> ToolResult:
        """按工具名路由到对应 Server 执行。"""
        server = self._routing.get(call.name)
        if server is None:
            return ToolResult.failed(
                tool_call_id=call.id,
                tool_name=call.name,
                error=f"工具 {call.name} 未在任何 MCP Server 中注册。可用: {self.tool_names}",
                error_type="tool_not_found",
            )
        return server.execute_tool_call(call)

    def execute_batch(self, calls: Sequence[ToolCall]) -> list[ToolResult]:
        return [self.execute(call) for call in calls]

    def health(self) -> list[dict[str, Any]]:
        return [server.health() for server in self.servers]


class BrokerToolRegistry(ToolRegistry):
    """把 MCPToolBroker 伪装成 ToolRegistry，供 AgentLoop 使用。

    AgentLoop 依赖 ToolRegistry 接口，而 Retriever 需要通过 MCP 调工具。
    与其在 AgentLoop 里加一个"要么用 registry 要么用 broker"的分支，
    不如做一个适配器——AgentLoop 保持只认识一种协作方，
    MCP 的接入对它完全透明。
    """

    def __init__(self, broker: MCPToolBroker, tracer: Tracer | None = None):
        super().__init__(tracer=tracer)
        self.broker = broker

    def get_definitions(self, only: Iterable[str] | None = None) -> list[dict[str, Any]]:
        return self.broker.get_tool_definitions(only=list(only) if only is not None else None)

    def has(self, name: str) -> bool:
        return name in self.broker.tool_names

    @property
    def tool_names(self) -> list[str]:
        return self.broker.tool_names

    def execute_one(self, call: ToolCall) -> ToolResult:
        # 不在这里记 trace：MCPServer 内的 ToolRegistry.execute_one 已经
        # tracer.log 过一次。bind_tracer 让两层共用同一个 tracer，
        # 这里再记就会让每条工具调用在 trace 里出现两次。
        return self.broker.execute(call)

    # 不覆盖 execute()。父类用 ThreadPoolExecutor 并发调 self.execute_one，
    # 而本类的 execute_one 已经走 broker，所以 Retriever 同一轮里的
    # 利润表 + 资产负债表 + 现金流量表会真正并行。之前用列表推导串行覆盖，
    # 等于把这段并发逻辑整段废掉（见审查意见）。


__all__ = ["BrokerToolRegistry", "MCPServer", "MCPToolBroker"]
